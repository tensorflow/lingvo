# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Abstractions for composing layers."""

import re
from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import computation_cost
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import recurrent
from lingvo.core import summary_utils
from lingvo.core import symbolic
from lingvo.core import tshape


class FirstNLayer(base_layer.BaseLayer):
  """Returns the first n args."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('n', 0, 'The number of args to return.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.n > 0

  def FProp(self, theta, *args):
    """Return the first n args."""
    p = self.params
    assert len(args) >= p.n
    return tuple(args[:p.n]) if p.n > 1 else args[0]

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    if p.n > 1:
      out_shapes = args[:p.n]
    else:
      out_shapes = (args[0],)
    return py_utils.NestedMap(flops=0, out_shapes=out_shapes)


class ArgIndexLayer(base_layer.BaseLayer):
  """Select args with a list of indices."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('idx', [], 'The indices of args to return.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.idx

  def FProp(self, theta, *args):
    """Return the indexed args."""
    p = self.params
    assert p.idx
    for i in p.idx:
      assert 0 <= i <= len(args)
    r = [args[i] for i in p.idx]
    return tuple(r) if len(r) > 1 else r[0]


def _ToTuple(x):
  return x if isinstance(x, tuple) else (x,)


def _MaybeStackExtraTheta(theta, all_vars, repeat):
  var_set = set([key for key, _ in all_vars.FlattenItems()])
  values = []
  for key, value in theta.FlattenItems():
    if key not in var_set and value is not None:
      # Replicate non-variable theta by p.repeat times.
      value = tf.stack([value] * repeat)
    values.append(value)
  return theta.Pack(values)


class CreateNestedMapLayer(base_layer.BaseLayer):
  """Returns a NestedMap from FProp args."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('keys', None, 'A list of keys for the returned NestedMap.')
    return p

  def FProp(self, theta, *args):
    assert len(self.params.keys) == len(args)
    output = py_utils.NestedMap()
    for key, arg in zip(self.params.keys, args):
      output.Set(key, arg)
    return output


class RepeatLayer(base_layer.BaseLayer):
  """A layer which repeats itself sequentially using lingvo Recurrent."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('body', None, 'The param for the main network layer.')
    p.Define('repeat', 1,
             'Repeat layers specified in \'body\' this many times.')
    p.Define('per_layer_vars', False, 'Use separate variables for each layer')
    p.Define('unroll', 'never', 'Unroll the layers: never, eval_only, always.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.repeat > 0
    assert p.unroll in ('never', 'eval_only', 'always')
    if p.per_layer_vars:
      for i in range(p.repeat):
        self.CreateChild('body_iter_%05d' % i, p.body)
    else:
      self.CreateChild('body', p.body)

  @property
  def _body(self):
    """A child layer to be used as the loop body."""
    p = self.params
    if p.per_layer_vars:
      return self.body_iter_00000
    else:
      return self.body

  def _CreateChildrenVariables(self):
    p = self.params
    with tf.variable_scope(self.params.name):
      if p.per_layer_vars:
        for i in range(p.repeat):
          with tf.variable_scope('iter_%05d' % i):
            self.children['body_iter_%05d' % i].InstantiateVariables()
      else:
        with py_utils.VariableShapePrefixContext(self.params.repeat):
          self.body.InstantiateVariables()
    super()._CreateChildrenVariables()

  def _unrolled_fprop(self, theta, *args):
    p = self.params
    fprop_inputs = args
    with tf.name_scope(p.name):
      for layer_idx in range(p.repeat):
        if p.per_layer_vars:
          layer_theta = theta['body_iter_%05d' % layer_idx]
          body_instance = self.children['body_iter_%05d' % layer_idx]
        else:
          body_instance = self.body

          def _Slice(t, idx=layer_idx):
            return t[idx]

          layer_theta = tf.nest.map_structure(_Slice, theta.body)

        fprop_outputs = body_instance.FProp(layer_theta, *fprop_inputs)
        fprop_outputs = _ToTuple(fprop_outputs)
        assert len(fprop_outputs) == len(fprop_inputs)
        fprop_inputs = fprop_outputs
      return fprop_outputs[0] if len(fprop_outputs) == 1 else fprop_outputs

  def FProp(self, theta, *args):
    p = self.params

    if p.unroll == 'always' or (self.do_eval and p.unroll == 'eval_only'):
      return self._unrolled_fprop(theta, *args)

    # Make a copy of args for the bprop path in case 'args' are mutated between
    # fprop and bprop.
    saved_args = tf.nest.map_structure(lambda x: x, args)

    if p.per_layer_vars:
      all_iters = [theta['body_iter_%05d' % i] for i in range(p.repeat)]
      theta_stack = tf.nest.map_structure(lambda *t: tf.stack(list(t)),
                                          *all_iters)
    else:
      theta_stack = _MaybeStackExtraTheta(theta.body, self.body.vars, p.repeat)

    def _ArgsToState(arg_list):
      """Returns a NestedMap from a list of FProp args."""
      state = py_utils.NestedMap()
      # Maintains a mapping from arg_idx to tensor. states cannot contains
      # None tensors.
      for idx in range(len(args)):
        if isinstance(arg_list[idx], py_utils.NestedMap):
          # Make sure each value in the NestedMap is a tensor.
          if not all(isinstance(t, tf.Tensor) for t in arg_list[idx].Flatten()):
            raise ValueError(
                'Each value in the input NestedMap must be a tensor.')
        if arg_list[idx] is not None:
          state['_s{}'.format(idx)] = arg_list[idx]
      return state

    def _StateToArgs(state):
      """Returns a list of FProp args from a NestedMap."""
      arg_list = []
      for idx in range(len(args)):
        attr = '_s{}'.format(idx)
        arg_list.append(state[attr] if attr in state else None)
        if isinstance(arg_list[-1], py_utils.NestedMap):
          assert isinstance(saved_args[idx], py_utils.NestedMap)
          py_utils.SetShapes(arg_list[-1], saved_args[idx])
        elif isinstance(arg_list[-1], tf.Tensor):
          if arg_list[-1] is not None:
            arg_list[-1].set_shape(saved_args[idx].shape)
      return arg_list

    def _CellFn(unused_theta, state0, theta_i):
      """Recurrent cell function wrapper of body.FProp."""
      # Retrieves fprop arguments from state and sets shapes.
      fprop_inputs = _StateToArgs(state0)

      # Sets shapes for theta_i as well.
      for dst, src in zip(theta_i.Flatten(), theta_stack.Flatten()):
        if src is not None:
          dst.set_shape(tf.TensorShape(src.shape.as_list()[1:]))

      # Runs the actual body.FProp
      fprop_outputs = self._body.FProp(theta_i, *fprop_inputs)
      fprop_outputs = _ToTuple(fprop_outputs)
      assert len(fprop_outputs) == len(fprop_inputs)

      # Passes fprop outputs to the next layer through state.
      state1 = _ArgsToState(fprop_outputs)
      return state1, py_utils.NestedMap()

    with tf.name_scope(p.name):
      # Add FProp arg list to state0.
      state0 = _ArgsToState(args)
      # Runs body.FProp k times using Recurrent where k = dim 0 of var_nmap.
      _, state1 = recurrent.Recurrent(
          theta=py_utils.NestedMap(),
          state0=state0,
          inputs=theta_stack,  # Pass cell_fn theta through inputs.
          cell_fn=_CellFn,
          allow_implicit_capture=p.allow_implicit_capture)

      # Retrieves fprop outputs from state1 and sets shapes.
      output_tensors = _StateToArgs(state1)
      return output_tensors[0] if len(args) == 1 else tuple(output_tensors)

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    meta = p.body.cls.FPropMeta(p.body, *args)
    py_utils.CheckShapes(meta.out_shapes)
    total = meta.flops * p.repeat
    return py_utils.NestedMap(flops=total, out_shapes=args)


class SoftCondLayer(base_layer.BaseLayer):
  r"""A wrapper layer implements soft conditional computation.

  This layer computes

  output = p.body.FProp( \sum_i w_i theta_i, \*inputs)

  where the theta passed to p.body is the weighted average over p.num_experts
  copies of theta.body, and w_i is the example-dependent coefficient for the
  i-th expert (theta_i).

  Reference:
  Soft Conditional Computation, B. Yang, G. Bender, Q.V. Le, J. Ngiam
  https://arxiv.org/abs/1904.04971
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_tasks', None, 'The Params for the main network layer.')
    p.Define('body', None, 'The Params for the main network layer.')
    p.Define('num_experts', None, 'Number of experts.')
    p.Define(
        'cond_dim', None,
        'This layer maintains a weight matrix of shape [cond_dim, num_experts] '
        'to map from inputs to the expert dimension.')
    p.Define('nonzeros_mean', False, 'Whether to only use nonzero values for '
             'calculating the mean.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.num_experts
    assert p.cond_dim

    self.CreateChild('body', p.body)

  def _CreateChildrenVariables(self):
    with tf.variable_scope(self.params.name):
      # Prepends p.num_experts to the tensor shape of every variable created
      # by p.body.
      with py_utils.VariableShapePrefixContext(self.params.num_experts):
        self.body.InstantiateVariables()
    super()._CreateChildrenVariables()

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    # Create Variables for task weight mapping.
    collections = [
        self.__class__.__name__ + '_vars',
    ]
    w_p = py_utils.WeightParams(
        shape=[p.cond_dim, p.num_experts],
        init=p.params_init,  # TODO(huangyp): try zero init instead.
        dtype=p.dtype,
        collections=collections)
    self.CreateVariable('w', w_p)

  def _GetExpertDist(self, theta, inputs, *args):
    """Get the task id from inputs tensors."""
    # TODO(huangyp): support the more general case when batch size is not 1.
    # Input shape can be either [batch, length, dim] or [length, batch, dim]
    reshaped_inputs = tf.reshape(inputs, [-1, self.params.cond_dim])
    if self.params.nonzeros_mean:
      per_example_emb = tf.reduce_sum(reshaped_inputs, 0)
      nonzeros = tf.cast(
          tf.math.count_nonzero(reshaped_inputs, 0), dtype=tf.float32)
      per_example_emb /= (nonzeros + 1e-10)
    else:
      per_example_emb = tf.reduce_mean(reshaped_inputs, 0)
    expert_dist = tf.nn.sigmoid(tf.einsum('i,ij->j', per_example_emb, theta.w))
    return expert_dist

  def FProp(self, theta, inputs, *args):
    p = self.params
    with tf.name_scope(p.name) as scope:
      expert_dist = self._GetExpertDist(theta, inputs, *args)
      if not self.do_eval:
        summary_utils.histogram('soft_cond_{}'.format(scope), expert_dist)

      # Excludes non-variable extra_theta like global_step.
      var_set = set([key for key, _ in self.body.vars.FlattenItems()])
      values = []
      for key, value in theta.body.FlattenItems():
        if key in var_set and value is not None:
          # Weighted average for all variables created in the body layer.
          value = tf.einsum('i,i...->...', expert_dist, value)
        values.append(value)
      weighted_theta = theta.body.Pack(values)
      return self.body.FProp(weighted_theta, inputs, *args)


class ParallelRepeatLayer(RepeatLayer):
  """A layer that connects identical sublayers in parallel.

  This layer consists of p.repeat copies of sub-layers, each of which
  shares the same structure but with different weights. Conceptually it
  computes:

  output= [f(theta[i], input[i]) for i in range(p.repeat)]

  where f is the fprop function of the sublayer, theta[i] and input[i] are
  the weights and inputs of the i-th sublayer.
  """

  def _InferOutShapes(self, args):
    input_shapes = [
        None if arg is None else tshape.Shape(arg.get_shape().as_list()[1:])
        for arg in args
    ]
    out_shapes = self.body.FPropMeta(self.body.params, *input_shapes).out_shapes
    return [None if s is None else s.ToTensorShape() for s in out_shapes]

  def FProp(self, theta, *args):
    """Runs p.repeat copies of self.body.FProp independently.

    Args:
      theta: Layer model parameters. The shape of each variable in theta is
        always [p.repeat, ...]. And the i-th slice theta[i] becomes theta of the
        i-th copy of self.body.
      *args: Input arguments. The shape of each tensor in args is always
        [p.repeat, ....]. And the list [arg[i] for arg in args] becomes inputs
        to the i-th copy of self.body.FProp.

    Returns:
      The accumulated output_tensors. Each tensor t in the return has the shape
      [p.repeat, ....] and the tuple (t[i] for i in output_tensors) is the
      return tuple of the i-th self.body.FProp.
    """
    p = self.params
    for arg in args:
      if arg is not None:
        arg = py_utils.HasShape(arg, [p.repeat], ndims=1)

    theta_stack = _MaybeStackExtraTheta(theta.body, self.body.vars, p.repeat)
    inputs = py_utils.NestedMap(theta=theta_stack, args=list(args))
    # Infer out_shapes from FPropMeta.
    out_shapes = self._InferOutShapes(args)

    def _CellFn(unused_theta, unused_state0, inputs):
      """Recurrent cell function wrapper of body.FProp."""
      # Sets shapes for both theta and inputs to self.body.FProp.
      for dst, src in zip(inputs.args + inputs.theta.Flatten(),
                          list(args) + theta_stack.Flatten()):
        if src is not None:
          dst.set_shape(tf.TensorShape(src.shape.as_list()[1:]))

      # Runs the actual body.FProp
      fprop_outputs = self.body.FProp(inputs.theta, *inputs.args)
      fprop_outputs = _ToTuple(fprop_outputs)
      assert len(fprop_outputs) == len(out_shapes)
      # Passes fprop outputs to the next layer through state.
      state1 = py_utils.NestedMap(outputs=list(fprop_outputs))
      return state1, py_utils.NestedMap()

    with tf.name_scope(p.name):
      # Initiate state0 with inferred output shapes.
      state0 = py_utils.NestedMap(
          outputs=[tf.zeros(shape, args[0].dtype) for shape in out_shapes])
      # Runs body.FProp p.repeat times using Recurrent.
      acc_states, _ = recurrent.Recurrent(
          theta=py_utils.NestedMap(),
          state0=state0,
          inputs=inputs,
          cell_fn=_CellFn)

      # Retrieves fprop outputs from state1 and sets shapes.
      output_tensors = tuple(acc_states.outputs)
      for out_idx in range(len(output_tensors)):
        output_tensors[out_idx].set_shape(
            tf.TensorShape([p.repeat] + out_shapes[out_idx].as_list()))

      return output_tensors[0] if len(args) == 1 else tuple(output_tensors)

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    input_shapes = [
        None if arg is None else tshape.Shape(arg.get_shape().as_list()[1:])
        for arg in args
    ]
    meta = p.body.cls.FPropMeta(p.body, *input_shapes)
    py_utils.CheckShapes(meta.out_shapes)
    total = meta.flops * p.repeat
    out_shapes = [
        None if s is None else tshape.Shape([p.repeat] + s[:])
        for s in meta.out_shapes
    ]
    return py_utils.NestedMap(flops=total, out_shapes=tuple(out_shapes))


class SequentialLayer(base_layer.BaseLayer):
  """A layer which connects a few layers in a sequence."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('sub', [], 'A list of layers\' params.')
    p.Define('repeat', 1, 'Repeat layers specified in \'sub\' '
             'this many times.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    if p.repeat <= 1:
      self._seq = []
      for sub in p.sub:
        self.CreateChild(sub.name, sub)
        self._seq.append((sub.name, self.children[sub.name]))
    else:
      # We create 'repeat' number of sub layers. Each sub layer is a
      # sequential layer specified by 'sub'.  This allows us to name each
      # repetition with a unique name.
      children = []
      for i in range(p.repeat):
        children.append(p.Copy().Set(name='%03d' % i, repeat=1))
      self.CreateChildren('rep', children)

  def FProp(self, theta, *args):
    p = self.params
    with tf.name_scope(p.name):
      tf.logging.vlog(1, 'layer %s', self.params.name)
      if p.repeat <= 1:
        for (name, ch) in self._seq:
          th = theta[name]
          args = _ToTuple(args)
          tf.logging.vlog(1, 'SequentialLayer: call %s %s %d %s',
                          ch.params.name, ch, len(args), str(args))
          args = ch.FProp(th, *args)
      else:
        for (ch, th) in zip(self.rep, theta.rep):
          args = _ToTuple(args)
          tf.logging.vlog(1, '  call %s %s %d %s', ch.params.name, ch,
                          len(args), str(args))
          args = ch.FProp(th, *args)
      args = _ToTuple(args)
      return args[0] if len(args) == 1 else args

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    total = 0
    for _ in range(p.repeat):
      for sub in p.sub:
        tf.logging.vlog(1, '  seq abs fprop %s %s %d %s', sub.name, sub.cls,
                        len(args), str(args))
        meta = sub.cls.FPropMeta(sub, *args)
        py_utils.CheckShapes(meta.out_shapes)
        total += meta.flops
        args = meta.out_shapes
    return py_utils.NestedMap(flops=total, out_shapes=args)


class UnarySequentialLayer(base_layer.BaseLayer):
  """A layer which connects a few layers in a sequence.

  Each layer FProp must take a single input arg (besides theta) and its return
  value will be used as the input for the next layer or as the final output
  if it's the last layer.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('sub', [], 'A list of layers\' params.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self._seq = []
    for sub in p.sub:
      self.CreateChild(sub.name, sub)
      self._seq.append((sub.name, self.children[sub.name]))

  def FProp(self, theta, x):
    tf.logging.vlog(1, 'layer %s', self.params.name)
    with tf.name_scope(self.params.name):
      for (name, ch) in self._seq:
        th = theta[name]
        tf.logging.vlog(1, '  call %s %s %s', ch.params.name, ch, x)
        x = ch.FProp(th, x)
      return x

  @classmethod
  def FPropMeta(cls, p, x):
    total = 0
    for sub in p.sub:
      tf.logging.vlog(1, '  seq abs fprop %s %s %s', sub.name, sub.cls, x)
      meta = sub.cls.FPropMeta(sub, x)
      total += meta.flops
      x = meta.out_shapes
    return py_utils.NestedMap(flops=total, out_shapes=x)


class GraphTensors:
  """A collection of named tensors (or NestedMaps of tensors)."""

  def __init__(self):
    self._named_tensors = py_utils.NestedMap()

  def StoreTensor(self, path, tensor):
    """Add tensor 't' to 'named_tensors' at 'path'.

    A path may be a name or a path into a NestedMap. For instance,
    StoreTensor('a.b.c', [1]), is equivalent to this:
    {'a', {'b': {'c': [1]}}.

    NestedMaps will be created if they do not already exist, or modified if they
    do exist. However, tensors cannot be overwritten.

    Args:
      path: A path input a NestedMap.
      tensor: The item to store (may be a NestedMap or a tensor).
    """
    names = path.strip().split('.')
    named_tensors = self._named_tensors
    while len(names) > 1:
      n = names.pop(0)
      assert isinstance(named_tensors, py_utils.NestedMap), named_tensors
      if n not in named_tensors:
        named_tensors[n] = py_utils.NestedMap()
      named_tensors = named_tensors[n]
    n = names.pop(0)
    if n in named_tensors:
      raise ValueError('A tensor named "%s" (%s) already exists.' % (n, path))
    named_tensors[n] = tensor

  def GetTensor(self, path):
    """Returns the tensor at 'path' in 'named_tensors'.

    Path may be a NestedMap key or a path through a series of NestedMaps.
    For instance, a path of 'a.b.c' could be used to retrieve [1] from
    this structure:
    {'a': {'b': {'c': [1]}}}

    Args:
      path: A path through a series of NestedMaps.
    """
    names = path.strip().split('.')
    named_tensors = self._named_tensors
    while names:
      assert isinstance(named_tensors, py_utils.NestedMap), named_tensors
      n = names.pop(0)
      assert n in named_tensors, '%s not found in %s' % (n, named_tensors)
      named_tensors = named_tensors[n]
    return named_tensors


class GraphSignature:
  """Represents the input/output signature of a GraphLayer.

  A signature is of the form:
    input->output
  or, optionally just an input:
    input

  The output part may be:
    out1      (a tensor name)
    out1.a.b  (a path into a NestedMap)
    out1,myvar,out2.x  (or a sequence of these things)

  The input part may be:
    in1           (a tensor name)
    in1.b         (a path into a NestedMap)
    [in1,myvar.c] (a list)
    (a=in1,b=in2) (a NestedMap, using the syntax of the NestedMap constructor)
    in1,[a,b]     (or a sequence of these things)

  In BNF::

    key := [A-Za-z_][A-Za-z0-9_]*
    path := key | path '.' key
    path_seq := <empty> | path | path_seq ',' path
    item := path | list | map
    list := '[' item_seq ']'
    item_seq := <empty> | item | item_seq ',' item
    map := '(' map_pair_seq ')'
    map_pair := key '=' item
    map_pair_seq := <empty> | map_pair | map_pair_seq ',' map_pair
    input := item_seq
    output := path_seq
  """

  def __init__(self, signature):
    self.signature = signature
    self.symbols = set(['[', ']', '=', '(', ')', ','])
    parts = self.signature.split('->')
    self._input_signature = parts[0]
    self._output_signature = None
    if len(parts) > 1:
      self._output_signature = parts[1]

    self._tokens = self._TokenizeInputs(self._input_signature)
    self._outputs = self._ParseOutputs(self._output_signature)
    self._inputs = self._ParseInputs()

  def __str__(self):
    return '(%s->%s)' % (self._inputs, self.outputs)

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  def _ParseOutputs(self, outputs):
    """Splits the output spec into comma-delimited parts.

    Args:
      outputs: A string containing the output specification, like, 'a,b'.

    Returns:
      The parsed representation, e.g. ['a', 'b']
    """
    if outputs is None:
      return []
    o_tensors = [x.strip() for x in outputs.split(',')]
    for x in o_tensors:
      assert x
    return o_tensors

  def _TokenizeInputs(self, inputs):
    """Splits the input signature into tokens (not data structures).

    Args:
      inputs: A string containing the input speficiation, like "[a,b],c"

    Returns:
      The tokenized representation, like ['[', 'a', ',', 'b', ']', ',', 'c']
    """
    start = -1
    tokens = []
    for j in range(len(inputs)):
      # Each letter can either be a symbol or part of a variable/path.
      if inputs[j] in self.symbols:
        if start >= 0:
          # Output the last variable/path.
          tokens.append(inputs[start:j].strip())
          start = -1
        tokens.append(inputs[j])
      elif start < 0:
        start = j
    if start >= 0:
      tokens.append(inputs[start:].strip())
    id_regex = re.compile(r'[_a-zA-Z][_a-zA-Z0-9]*(\.[_a-zA-Z][_a-zA-Z0-9]*)*$')
    for token in tokens:
      if not (token in self.symbols or id_regex.match(token)):
        raise ValueError(
            'token(%s) is not a symbol(%r) or matched by id_regex.' %
            (token, self.symbols))
    # Wrapping the tokens in list brackets allows us to parse this using
    # _ConsumeList.
    return ['['] + tokens + [']']

  def _ConsumePath(self):
    """Return the path found at the current token position and increment.

    Returns:
      The path at the current token position.
    """
    if self._i >= len(self._tokens):
      raise ValueError('Ran out of tokens while looking for a path/key')
    if self._tokens[self._i] in self.symbols:
      raise ValueError('Found a symbol %s while looking for a path/key' %
                       (self._tokens[self._i]))
    self._i += 1
    return self._tokens[self._i - 1]

  def _ConsumeKey(self):
    """Return the key found at the current token position and increment.

    Returns:
      The key at the current token position.
    """
    token = self._ConsumePath()
    assert '.' not in token, token
    return token

  def _ConsumeSymbol(self, symbol):
    """Verify that the current token is this symbol, and increment.

    Args:
      symbol: The symbol that we expect at the current token position.
    """
    assert symbol in self.symbols
    if self._i >= len(self._tokens):
      raise ValueError('Ran out of tokens while looking for a %s' % symbol)
    if not self._MaybeConsumeSymbol(symbol):
      raise ValueError('Found a symbol %s while looking for %s' %
                       (self._tokens[self._i], symbol))

  def _MaybeConsumeSymbol(self, symbol):
    """Attempt to consume the given symbol.

    The symbol must be a member of the symbol set: []()=,

    Args:
      symbol: The symbol that we expect at the current token position.

    Returns:
      True if the symbol was consumed, False otherwise.
    """
    assert symbol in self.symbols
    if self._i < len(self._tokens) and self._tokens[self._i] == symbol:
      self._i += 1
      return True
    return False

  def _ConsumeItem(self):
    """Return whatever is at the current token position and increment.

    Returns:
      A list, NestedMap, or path/key.
    """
    if self._i >= len(self._tokens):
      raise ValueError(
          'Ran out of tokens while looking for a variable, list or NestedMap.')
    if self._tokens[self._i] == '(':
      return self._ConsumeMap()
    if self._tokens[self._i] == '[':
      return self._ConsumeList()
    return self._ConsumePath()

  def _ConsumeMap(self):
    """Return the NestedMap that starts at the current position, and increment.

    Returns:
      The NestedMap that starts at the current token position.
    """
    if self._i >= len(self._tokens):
      raise ValueError('Ran out of tokens while looking for a NestedMap.')
    if self._tokens[self._i] != '(':
      raise ValueError('Expected ( at token position %d' % self._i)
    self._i += 1
    if self._MaybeConsumeSymbol(')'):
      # Empty NestedMaps are allowed.
      return py_utils.NestedMap()
    result = py_utils.NestedMap()
    while self._i < len(self._tokens):
      name = self._ConsumeKey()
      self._ConsumeSymbol('=')
      result[name] = self._ConsumeItem()
      if self._MaybeConsumeSymbol(')'):
        return result
      self._ConsumeSymbol(',')
    raise ValueError('Ran out of tokens while looking for end of NestedMap.')

  def _ConsumeList(self):
    """Return the list that starts at the current position, and increment.

    Returns:
      The list that starts at the current token position.
    """
    if self._i >= len(self._tokens):
      raise ValueError('Ran out of tokens while looking for a list.')
    if self._tokens[self._i] != '[':
      raise ValueError('Expected [ at token position %d' % self._i)
    self._i += 1
    if self._MaybeConsumeSymbol(']'):
      # Empty lists are allowed.
      return []
    result = []
    while self._i < len(self._tokens):
      result.append(self._ConsumeItem())
      if self._MaybeConsumeSymbol(']'):
        return result
      self._ConsumeSymbol(',')
    raise ValueError('Ran out of tokens while looking for end of list.')

  def _ParseInputs(self):
    """Parse the inputs signature string.

    Returns:
      dict: The parsed inputs structure, like ``{'a': ['b', 'c']}``.
    """
    self._i = 0
    # The tokenization process adds fake '[' and ']' tokens so we can
    # parse them as a list.
    return self._ConsumeList()


class GraphLayer(base_layer.BaseLayer):
  r"""A layer that connects a few layers in a simple data flow graph.

  Params.sub specifies a list of (signature, layer param) pairs for all the
  sub-layers. 'Signature' of a layer specifies the signature of the
  corresponding layer's FProp function. It also names its input and output
  tensors.  A 'signature' is of the form 'x,b->c', where tensors on the left
  side of '->' specifies the input to the layer and tensors to the right
  specifies the output from the layer.  Note, the input to a layer has to be
  produced before, either as input to the GraphLayer or as produced by some
  previous layer. The output of a layer must be uniquely named, i.e. they can't
  reuse names assigned previous layer output or the input to this GraphLayer.

  The full grammar of the signature is described in the GraphSignature class
  definition above.

  Example
    input: ['a', 'b']
    'a->c', Fn(lambda x : tf.nn.relu(x))
    'c,b->d', Fn(lambda x, y : x + y)

  The above example computes relu(a) + b
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_endpoints', [], 'Names of the input tensors.')
    p.Define('output_endpoints', [], 'Names of the output tensors.')
    # TODO(yonghui): Define a NamedTuple for this pair.
    p.Define('sub', [], 'A list of (signature, layer params) pairs.')
    p.Define(
        'sub_layers', None, 'A list of layer params that GraphLayer '
        'should instantiate. If this is defined, sub is a list of '
        '(signature, index) pairs, where index refers to one of the '
        'layers in sub_layers. This allows sub to reference layers '
        'more than once.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_endpoints
    self._seq = []

    # If sub_layers exists, we instantiate layers from that param, otherwise
    # we get them from sub. reference_name helps us figure out what the layers
    # are called in the later loop.
    if self.params.sub_layers:
      for i, sub in enumerate(p.sub_layers):
        name = sub.name
        if not name:
          name = 'sub_layer_%d' % i
          sub.name = name
        self.CreateChild(name, sub)
      reference_name = lambda sub: self.params.sub_layers[sub].name
    else:
      for i, (signature, sub) in enumerate(p.sub):
        assert signature
        sig = GraphSignature(signature)
        assert sig.outputs, '{}'.format(signature)
        name = sub.name
        if not name:
          name = '%s_%02d' % (sig.outputs[0], i)
          sub.name = name
        self.CreateChild(name, sub)
      reference_name = lambda sub: sub.name

    for i, (signature, sub) in enumerate(p.sub):
      assert signature
      sig = GraphSignature(signature)
      assert sig.outputs, '{}'.format(signature)
      name = reference_name(sub)
      self._seq.append((name, sig, self.children[name]))

  def FProp(self, theta, *args):
    p = self.params

    graph_tensors = self._fprop = GraphTensors()
    with tf.name_scope(p.name):
      if len(p.input_endpoints) != len(args):
        raise ValueError(
            'Wrong number of inputs for {}: required={}, provided={}'.format(
                p.name, len(p.input_endpoints), len(args)))
      for n, t in zip(p.input_endpoints, args):
        if isinstance(t, py_utils.NestedMap):
          assert all(isinstance(x, tf.Tensor) for x in t.Flatten()), t
        else:
          assert isinstance(t,
                            tf.Tensor), (n, t, p.name, p.input_endpoints, args)
        graph_tensors.StoreTensor(n, t)

      ch_out = None
      for i, (name, sig, ch) in enumerate(self._seq):
        th = theta[name]
        template = py_utils.NestedMap(inputs=sig.inputs)
        packed = template.Transform(graph_tensors.GetTensor)
        input_args = packed.inputs
        tf.logging.vlog(1, 'signature: %s', p.sub[i][0])
        tf.logging.vlog(1, 'GraphLayer: call %s %s %d %s', ch.params.name, ch,
                        len(input_args), str(input_args))
        ch_out = ch.FProp(th, *input_args)
        if len(sig.outputs) == 1:
          ch_out = (ch_out,)
        assert len(sig.outputs) == len(ch_out)
        for n, t in zip(sig.outputs, ch_out):
          graph_tensors.StoreTensor(n, t)

      layer_out = tuple(graph_tensors.GetTensor(x) for x in p.output_endpoints)
      if len(layer_out) == 1:
        layer_out = layer_out[0]

      return layer_out

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    total = 0

    graph_tensors = GraphTensors()
    assert len(p.input_endpoints) == len(args)
    for n, t in zip(p.input_endpoints, args):
      graph_tensors.StoreTensor(n, t)

    ch_out = None
    for signature, sub in p.sub:
      sig = GraphSignature(signature)
      template = py_utils.NestedMap(inputs=sig.inputs)
      packed = template.Transform(graph_tensors.GetTensor)
      input_args = packed.inputs

      meta = sub.cls.FPropMeta(sub, *input_args)
      total += meta.flops
      ch_out = meta.out_shapes
      assert len(ch_out) == len(sig.outputs)
      for n, t in zip(sig.outputs, ch_out):
        graph_tensors.StoreTensor(n, t)

    layer_out = tuple(graph_tensors.GetTensor(x) for x in p.output_endpoints)
    return py_utils.NestedMap(flops=total, out_shapes=layer_out)


class ParallelLayer(base_layer.BaseLayer):
  """A layer which connects a few layers in a parallel."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'sub', [], 'A list of layers\' params. Each layer\'s '
        'FProp must return one Tensor or a tuple of Tensors. '
        'Their return values then can be merged according to the '
        'merge method. ')
    p.Define(
        'merge', None, 'Method to combine sub-layers\' outputs.'
        'It must be a callable list(tuple(tf.Tensor)) -> tuple(tf.Tensor).')
    p.Define(
        'merge_meta', None, 'Callable to compute the meta of merge(). It '
        'takes a list of tuples of TensorShape, and returns a NestedMap with '
        'flops and out_shapes, etc.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self._seq = []
    for sub in p.sub:
      self.CreateChild(sub.name, sub)
      self._seq.append((sub.name, self.children[sub.name]))

  def FProp(self, theta, *args):
    p = self.params

    with tf.name_scope(p.name):
      # Computes sub layers in parallel.
      outputs = []
      for (name, ch) in self._seq:
        th = theta[name]
        out = ch.FProp(th, *args)
        if isinstance(out, (list, tuple)):
          outputs.append(tuple(out))
        else:
          outputs.append((out,))
      rets = p.merge(outputs)
      return rets if len(rets) > 1 else rets[0]

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    total = 0
    outputs = []
    for sub in p.sub:
      tf.logging.vlog(1, '  par abs fprop %s %s %d %s', sub.name, sub.cls,
                      len(args), str(args))
      meta = sub.cls.FPropMeta(sub, *args)
      py_utils.CheckShapes(meta.out_shapes)
      meta.VLog(
          1, '  par abs fprop {} {} {} {}'.format(sub.name, sub.cls, len(args),
                                                  str(args)))
      total += meta.flops
      outputs.append(meta.out_shapes)

    meta = p.merge_meta(outputs)
    py_utils.CheckShapes(meta.out_shapes)
    meta.flops += total
    return meta


class MapLayer(base_layer.BaseLayer):
  """A layer applies a lambda on every argument."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('fn', None, 'A callable tensor->tensor.')
    p.Define('fn_meta', None, 'A callable shape->(flops, shape).')
    p.Define('kwargs', {}, 'Keyword arguments to fn.')
    return p

  def FProp(self, theta, *args):
    r"""Applies lambda(x, \*kwargs) for every non-None arg."""
    del theta
    p = self.params
    with tf.name_scope(p.name):
      ret = [None if x is None else p.fn(x, **p.kwargs) for x in args]
      return tuple(ret) if len(ret) > 1 else ret[0]

  @classmethod
  def FPropMeta(cls, p, *args):
    flops, rets = 0, []
    for x in args:
      if x is None:
        rets.append(None)
      else:
        cost, shape = p.fn_meta(x)
        py_utils.CheckShapes((shape,))
        flops += cost
        rets.append(shape)
    return py_utils.NestedMap(flops=flops, out_shapes=tuple(rets))


class LinearLayer(quant_utils.QuantizableLayer):
  """Linear layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    return p

  def __init__(self, params):
    """Constructs a LinearLayer object."""
    super().__init__(params)
    p = self.params
    self.CreateAqtWeight(
        'w', shape=[p.input_dims, p.output_dims], feature_axis=-1)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    if p.device_mesh is not None:
      assert p.weight_split_dims_mapping is not None
      assert len(p.weight_split_dims_mapping) == 2
    self.CreateVariable(
        'w',
        py_utils.WeightParams(
            shape=[p.input_dims, p.output_dims],
            init=p.params_init,
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=p.weight_split_dims_mapping,
            collections=[self.__class__.__name__ + '_vars']))

  def FProp(self, theta, inputs):
    """Apply projection to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: The inputs tensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    p = self.params
    with tf.name_scope(p.name):
      computation_cost.Add(
          self, 'flops',
          tf.reduce_prod(tf.cast(tf.shape(inputs)[:-1], tf.int64)) *
          tf.cast(symbolic.ToTensor(p.input_dims * p.output_dims), tf.int64) *
          2)
      inputs, w = self.ToAqtInputs(
          'w', act=inputs, weight=theta.w, w_feature_axis=-1)
      ret = py_utils.ProjectLastDim(inputs, w, p.input_dims, p.output_dims)
      ret = self.FromAqtMatmul('w', ret)
      return ret

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    assert p.input_dims == inputs[-1]
    # c_{ij} += x_{ik} * y_{kj} are considered 2 flops.
    return py_utils.NestedMap(
        flops=inputs.size * p.output_dims * 2,
        out_shapes=(inputs[:-1] + [p.output_dims],))


class BiasLayer(base_layer.BaseLayer):
  """Bias layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dims', 0, 'Depth of the input.')
    return p

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    if p.device_mesh is not None:
      assert p.weight_split_dims_mapping
      assert len(p.weight_split_dims_mapping) == 1
    self.CreateVariable(
        'b',
        py_utils.WeightParams(
            shape=[p.dims],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=p.weight_split_dims_mapping,
            collections=[self.__class__.__name__ + '_vars']))

  def FProp(self, theta, inputs):
    """Adds bias to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: The inputs tensor.  Shaped [..., dims].

    Returns:
      Inputs plus bias.
    """
    with tf.name_scope(self.params.name):
      return inputs + theta.b

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    assert inputs[-1] == p.dims
    return py_utils.NestedMap(flops=inputs.size, out_shapes=(inputs,))


class BranchLayer(base_layer.BaseLayer):
  """A layer to help constructing a network structure with multiple outputs."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('body', None, 'The param for the main network layer.')
    p.Define(
        'fetches', [], 'Fetch points within the body layer. Each fetch '
        'layers\' activation is appended to the output of body.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChild('body', p.body)

  def FProp(self, theta, *args):
    p = self.params
    with tf.name_scope(p.name):
      args = _ToTuple(self.body.FProp(theta.body, *args))
      for fetch in p.fetches:
        args += (self.body.GetDescendant(fetch).activation,)
      return args


class BatchParallelLayer(base_layer.BaseLayer):
  """A layer splits the batch and compute the FProp on multiple devices."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('sub', None, 'A layer param.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self.CreateChild('sub', p.sub)

  def FProp(self, theta, *args):
    """FProp through multiple devices in the split.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      *args: A tuple of Tensors (one or more). Every tensor's first dimension is
        the same (the batch dimension).

    Returns:
      The sub layer's output.
    """
    p = self.params
    with tf.name_scope(p.name):
      assert all(isinstance(x, tf.Tensor) for x in args)
      cluster = self.cluster
      num = cluster.num_devices_per_split
      if num == 1:
        return self.sub.FProp(theta.sub, *args)
      inps = py_utils.SplitRecursively(list(args), num, axis=0)
      outs = []
      for i, xs in enumerate(inps):
        device = cluster.WorkerDeviceInModelSplit(i)
        tf.logging.info('%d on device %s', i, device)
        with tf.device(device):
          ys = self.sub.FProp(theta.sub, *xs)
          if isinstance(ys, tuple):
            outs += [list(ys)]
          else:
            outs += [ys]  # ys is a single tensor
      ret = py_utils.ConcatRecursively(outs, axis=0)
      if isinstance(ret, list):
        return tuple(ret)
      else:
        return ret  # ys is a single tensor


class FnLayer(base_layer.BaseLayer):
  """A layer applies a function on a tuple of tensors."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('fn', None, 'A lambda tuple(Tensor) -> tuple(Tensor) '
             'or a single Tensor.')
    p.Define(
        'fn_meta', None, 'Callable to compute the meta of fn(). It '
        'takes a tuple of TensorShape, and returns a NestedMap with '
        'flops and out_shapes, etc.')
    return p

  def FProp(self, theta, *args):
    r"""Applies a function (p.fn) on args.

    Args:
      theta: Unused.
      *args: A tuple of Tensors (one or more).

    Returns:
      fn(\*args).
    """
    with tf.name_scope(self.params.name):
      return self.params.fn(*args)

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    meta = p.fn_meta(*args)
    py_utils.CheckShapes(meta.out_shapes)
    return meta


class RematerializationLayer(base_layer.BaseLayer):
  """A wrapper layer with rematerialization."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('body', None,
             'The main layer whose FProp will be wrapped by RematerializeFn.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateChild('body', self.params.body)

  def FProp(self, theta, *xs):
    input_list = theta.body.Flatten()
    theta_len = len(input_list)
    input_list += list(xs)
    input_len = len(input_list)

    def Fn(*args):
      body_theta = theta.body.Pack(args[:theta_len])
      return self.body.FProp(body_theta, *args[theta_len:input_len])

    return py_utils.RematerializeFn(Fn, *input_list)

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    return p.body.cls.FPropMeta(p.body, *args)


class PrintShapeLayer(base_layer.BaseLayer):
  """A layer prints debug information."""

  def FProp(self, theta, *args):
    p = self.params
    with tf.name_scope(p.name) as name_scope:
      for i, arg in enumerate(args):
        if not isinstance(arg, tf.Tensor):
          tf.logging.info(
              'FProp non-Tensor input in {}: arg_{} arg = {}'.format(
                  name_scope, i, arg))
        else:
          tf.logging.info(
              'FProp inputs in {}: arg_{} shape = {} dtype = {}'.format(
                  name_scope, i, arg.shape, arg.dtype.name))
    if len(args) == 1:
      return args[0]
    else:
      return args

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    return py_utils.NestedMap(flops=0, out_shapes=args)


class ReshapeLayer(base_layer.BaseLayer):
  """Reshape the inputs to the specified shape."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('shape', None, 'The output shape.')
    return p

  def FProp(self, theta, inp):
    """Reshape the inputs.

    Args:
      theta: A `.NestedMap` object containing variable values.
      inp: A tensor or `.NestedMap` object containing inputs to reshape.

    Returns:
      A tensor or `.NestedMap` with the same structure as the input.
    """
    p = self.params
    return tf.nest.map_structure(lambda t: tf.reshape(t, p.shape), inp)
