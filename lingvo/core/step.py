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
"""An abstract layer for processing sequences step-by-step.

E.g.::

  def ProcessSeq(step, external_inputs, input_batch):
    prepared_inputs = step.PrepareExternalInputs(
        step.theta, external_inputs)
    batch_size, T = tf.shape(input_batch.paddings)[:2]
    state = step.ZeroState(
        step.theta, prepared_inputs, batch_size)
    for t in range(T):
      step_inputs = input_batch.Transform(lambda x: x[:, t, ...])
      step_outputs, state = step.FProp(
          step.theta, prepared_inputs, step_inputs, state)
      (processing step_outputs...)
"""

import collections

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import builder_layers
from lingvo.core import py_utils
from lingvo.core import recurrent


class Step(base_layer.BaseLayer):
  """A layer that processes input sequences step-by-step.

  This can be seen as an RNNCell extended with optional external inputs.
  """

  def PrepareExternalInputs(self, theta, external_inputs):
    """Returns the prepared external inputs, e.g., packed_src for attention."""
    if not external_inputs:
      external_inputs = py_utils.NestedMap()
    packed = external_inputs.DeepCopy()
    for name, child in self.children.items():
      child_external_inputs = external_inputs.get(name, py_utils.NestedMap())
      if isinstance(child, (tuple, list)):
        output = []
        for i, sub in enumerate(child):
          if isinstance(sub, Step):
            output.append(
                sub.PrepareExternalInputs(theta[name][i],
                                          child_external_inputs))
        if output:
          if len(output) != len(child):
            raise ValueError('Expecting child list to be instances of Step.')
          packed[name] = type(child)(output)
      elif isinstance(child, Step):
        packed[name] = child.PrepareExternalInputs(theta[name],
                                                   child_external_inputs)
    return packed

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Returns the initial state given external inputs and batch size.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: External inputs returned by PrepareExternalInputs().
      batch_size: An int scalar representing the batch size of per-step inputs.

    Returns:
      A `.NestedMap` representing the initial state, which can be passed to
      FProp() for processing the first time step.
    """
    state0 = py_utils.NestedMap()
    for name, child in self.children.items():
      if isinstance(child, (tuple, list)):
        output = []
        for i, sub in enumerate(child):
          if isinstance(sub, Step):
            output.append(
                sub.ZeroState(theta[name][i], prepared_inputs[name][i],
                              batch_size))
        if output:
          if len(output) != len(child):
            raise ValueError('Expecting child list to be instances of Step.')
          state0[name] = type(child)(output)
      elif isinstance(child, Step):
        state0[name] = child.ZeroState(theta[name], prepared_inputs[name],
                                       batch_size)
    return state0

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Forward function.

    step_inputs, state0, step_outputs, and state1 should each be a `.NestedMap`
    of tensor values. Each tensor must be of shape [batch_size ...]. The
    structure of NestedMaps are determined by the implementation. state0 and
    state1 must have exactly the same structure and tensor shapes.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: External inputs returned by PrepareExternalInputs().
      step_inputs: The inputs for this time step.
      padding: A 0/1 float tensor of shape [batch_size]; 1.0 means that this
        batch element is empty in this step.
      state0: The previous recurrent state.

    Returns:
      A tuple (step_outputs, state1).

      - outputs: The outputs of this step.
      - state1: The next recurrent state.
    """
    raise NotImplementedError(type(self))


class StatelessLayerStep(Step):
  """Allows BaseLayer subclasses to be used as Steps.

  Layers used with this class should be stateless: they should not return
  anything that must be passed back in the next invocation.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('layer', None, 'Params for the layer that this step wraps.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = params
    self.CreateChild('layer', p.layer)

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Perform inference on a stateless layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: unused.
      step_inputs: A NestedMap containing 'inputs', which are passed directly to
        the layer.
      padding: A 0/1 float tensor of shape [batch_size]; 1.0 means that this
        batch element is empty in this step.
      state0: unused.

    Returns:
      (output, state1), where output is the output of the layer, and
      state1 is an empty NestedMap.
    """
    del state0
    del prepared_inputs
    args = {}
    if padding is not None:
      args['padding'] = padding
    output = self.layer.FProp(theta.layer, step_inputs.inputs, **args)
    return output, py_utils.NestedMap()


class StackStep(Step):
  """A stack of steps.

  Each sub-step is assumed to accept step_inputs of type NestedMap(inputs=[])
  and return a primary output of type NestedMap(output=tensor). The
  output of layer n-1 is sent to input of layer n.

  Per-step context vectors and per-sequence context vectors can also be
  supplied; see FProp for more details.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'sub', [], 'A list of sub-stack params. Each layer is '
        'expected to accept its input as NestedMap(inputs=[]), and '
        'produce output as NestedMap(output=tensor). '
        'The external_inputs parameter is passed directly to the '
        'PrepareExternalInputs method of each sub-step. ')
    p.Define(
        'residual_start', -1, 'An index of the layer where residual '
        'connections start. Setting this parameter to a negative value turns '
        'off residual connections.'
        'More precisely, when i >= residual_start, the output of each step '
        'is defined as: '
        'output[i] = output[i - residual_stride] + sub[i](output[i - 1]) '
        'where output[-1] is the step input.')
    p.Define(
        'residual_stride', 1, 'If residual connections are active, this '
        'is the number of layers that each connection skips. For '
        'instance, setting residual_stride = 2 means the output of layer '
        'n is added to layer n + 2')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = params
    self.sub_steps = []
    self.CreateChildren('sub', p.sub)

  def PrepareExternalInputs(self, theta, external_inputs):
    """Delegates external inputs preparation to sub-layers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      external_inputs: A `.NestedMap` object. The structure of the internal
        fields is defined by the sub-steps.

    Returns:
      A `.NestedMap` containing a pre-processed version of the external_inputs,
      one per sub-step.
    """
    packed = py_utils.NestedMap(sub=[])
    for i in range(len(self.sub)):
      packed.sub.append(self.sub[i].PrepareExternalInputs(
          theta.sub[i], external_inputs))
    return packed

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Computes a zero state for each sub-step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: An output from PrepareExternalInputs.
      batch_size: The number of items in the batch that FProp will process.

    Returns:
      A `.NestedMap` containing a state0 object for each sub-step.
    """
    state = py_utils.NestedMap(sub=[])
    for i in range(len(self.sub)):
      state.sub.append(self.sub[i].ZeroState(theta.sub[i], prepared_inputs,
                                             batch_size))
    return state

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Performs inference on the stack of sub-steps.

    There are three possible ways to feed input to the stack:

      * step_inputs.inputs: These tensors are fed only to the lowest layer.
      * step_inputs.context: [Optional] This tensor is fed to every layer.
      * prepared_inputs: [Optional] This tensor is fed to every layer and
          is assumed to stay constant over all steps.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: An output from PrepareExternalInputs.
      step_inputs: A `.NestedMap` containing a list called 'inputs', an
        optionally a tensor called 'context'.
      padding: A 0/1 float tensor of shape [batch_size]; 1.0 means that this
        batch element is empty in this step.
      state0: The previous recurrent state.

    Returns:
      A tuple (output, state1):

      - output: A `.NestedMap` containing the output of the top-most step.
      - state1: The recurrent state to feed to next invocation of this graph.
    """
    state1 = py_utils.NestedMap(sub=[])
    inputs = list(step_inputs.inputs)
    # We pretend that the input is the output of layer -1 for the purposes
    # of residual connections.
    residual_inputs = [tf.concat(inputs, axis=1)]
    additional = []
    if 'context' in step_inputs:
      additional.append(step_inputs.context)
    for i in range(len(self.sub)):
      sub_inputs = py_utils.NestedMap(inputs=inputs + additional)
      sub_output, state1_i = self.sub[i].FProp(theta.sub[i],
                                               prepared_inputs.sub[i],
                                               sub_inputs, padding,
                                               state0.sub[i])
      state1.sub.append(state1_i)
      output = sub_output.output
      if i >= self.params.residual_start >= 0:
        # residual_inputs contains the step input at residual_inputs[0].
        assert i + 1 - self.params.residual_stride < len(residual_inputs)
        output += residual_inputs[i + 1 - self.params.residual_stride]
      residual_inputs.append(output)
      inputs = [output]
    return py_utils.NestedMap(output=output), state1


class ParallelStep(Step):
  """Runs many steps on the same input and concatenates their outputs."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'sub', [], 'A list of step params. Each step is '
        'expected to accept its input as NestedMap(inputs=[]), and '
        'produce output as NestedMap(output=tensor). '
        'The external_inputs parameter is passed directly to the '
        'PrepareExternalInputs method of each sub-step. ')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = params
    self.CreateChildren('sub', p.sub)

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Performs inference on N steps at once and concatenates the result.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: An output from PrepareExternalInputs.
      step_inputs: A `.NestedMap` containing a list called 'inputs'.
      padding: A 0/1 float tensor of shape [batch_size]; 1.0 means that this
        batch element is empty in this step.
      state0: The previous recurrent state.

    Returns:
      A tuple (output, state1):

      - output: A `.NestedMap` containing the output of the top-most step.
      - state1: The recurrent state to feed to next invocation of this graph.
    """
    state1 = py_utils.NestedMap(sub=[None] * len(self.sub))
    outputs = [None] * len(self.sub)

    for i in range(len(self.sub)):
      outputs[i], state1.sub[i] = self.sub[i].FProp(theta.sub[i],
                                                    prepared_inputs.sub[i],
                                                    step_inputs, padding,
                                                    state0.sub[i])

    output = py_utils.NestedMap(output=tf.concat(outputs, axis=1))
    return output, state1


# signature: A GraphSignature string defining the input and output parameters
#   of this step. For example, (inputs=[a,b])->c means that step_inputs
#   should be NestedMap(inputs=[a,b]), and the output of FProp should be
#   stored in c.
# external_signature: A GraphSignature string defining the input to
#   PrepareExternalInputs. For example, 'external_inputs.foo' means that
#   the tensor external_inputs.foo should be the 'external_inputs' parameter
#   when calling PrepareExternalInputs on this sub-step.
# params: The parameters to use when constructing the sub-step.
SubStep = collections.namedtuple('SubStep',
                                 ['signature', 'external_signature', 'params'])


class GraphStep(Step):
  r"""A step that connects sub-steps in a simple data flow graph.

  This is an adaptation of builder_layers.GraphLayer to support steps.

  Params.sub specifies a list of Specs that define each sub-step.

  A spec contains:

    * step_inputs: The signature describing how to assemble the input and output
      for this step. The input part describes the 'step_inputs' parameter,
      while the output part describes the name of the output. The state0
      input and state1 output are handled automatically and should not be
      specified.
    * external_inputs: if this Step requires external_inputs, this
      is the signature describing how to find those inputs.
      This value can also be set to None.
    * params: the params used to construct the sub-step.

  The format of signature strings is defined in detail in the GraphSignature
  class documentation.

  All inputs to a layer must have been produced by some previous layer. No
  cycles are allowed. All outputs must be uniquely named; no overwriting
  of previous names is allowed.

  Example
    ('(act=[layer_0.output,step_inputs.context])->layer_1',
     'external_inputs.extra',
     step_params)

  This constructs the step defined by step_params. Its FProp method will be
  called with {act=[layer_0.output,step_inputs.context]} as the step_inputs
  parameter. Its PrepareExternalInputs method will be called with
  'external_inputs.extra' as the external_inputs parameter. The output of that
  method will be passed to ZeroState and FProp.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('output_signature', '', 'Signature of the step output.')
    p.Define('sub', [], 'A list of SubSteps (defined above).')
    p.Define('dict_type', py_utils.NestedMap, 'Type of nested dicts.')
    return p

  _seq = collections.namedtuple(
      '_Seq', ['name', 'signature', 'external_signature', 'step'])

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    self._seq = []
    for i, (signature, external_signature, sub_params) in enumerate(p.sub):
      assert signature
      sig = builder_layers.GraphSignature(signature)
      assert len(sig.inputs) == 1
      assert sig.outputs
      external_sig = None
      if external_signature:
        external_sig = builder_layers.GraphSignature(external_signature)
        assert len(external_sig.inputs) == 1
        assert not external_sig.outputs
      name = sub_params.name
      if not name:
        name = '%s_%02d' % (sig.outputs[0], i)
        sub_params.name = name
      self.CreateChild(name, sub_params)
      self._seq.append(
          GraphStep._seq(name, sig, external_sig, self.children[name]))
    self.output_signature = builder_layers.GraphSignature(p.output_signature)

  def PrepareExternalInputs(self, theta, external_inputs):
    """Prepares external inputs for each sub-step.

    The external_inputs parameter of this method is processed by the
    external_inputs of each sub-step, then processed by the sub-step's
    PrepareExternalInputs method.

    Args:
      theta: variables used by sub-steps.
      external_inputs: A NestedMap of [n_batch, ...] tensors.

    Returns:
      A NestedMap of prepared inputs, where the keys are the names of
        each sub-step.
    """
    graph_tensors = builder_layers.GraphTensors()
    graph_tensors.StoreTensor('external_inputs', external_inputs)
    prepared_inputs = py_utils.NestedMap()
    with tf.name_scope(self.params.name):
      for seq in self._seq:
        if seq.external_signature:
          template = py_utils.NestedMap(inputs=seq.external_signature.inputs)
          packed = template.Transform(graph_tensors.GetTensor)
          seq_external_inputs = packed.inputs[0]
          prepared_inputs[seq.name] = seq.step.PrepareExternalInputs(
              theta[seq.name], seq_external_inputs)
        else:
          prepared_inputs[seq.name] = py_utils.NestedMap()
    return prepared_inputs

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Creates a zero state NestedMap for this step.

    Args:
      theta: variables used by sub-steps.
      prepared_inputs: Output from a call to PrepareExternalInputs.
      batch_size: The number of items in the batch that FProp will process.

    Returns:
      A NestedMap of ZeroState results for each sub-step.
    """
    state0 = py_utils.NestedMap()
    with tf.name_scope(self.params.name):
      for seq in self._seq:
        state0[seq.name] = seq.step.ZeroState(theta[seq.name],
                                              prepared_inputs[seq.name],
                                              batch_size)
    return state0

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """A single inference step for this step graph.

    Args:
      theta: variables used by sub-steps.
      prepared_inputs: A NestedMap containing external_inputs that were
        pre-processed by the PrepareExternalInputs method of each sub-step. The
        keys are the names of the sub-steps.
      step_inputs: A NestedMap of [batch, ...] tensors. The structure of this
        depends on the graph implementation.
      padding: A 0/1 float tensor of shape [batch_size]; 1.0 means that this
        batch element is empty in this step.
      state0: A NestedMap of state variables produced by either ZeroState or a
        previous invocation of this FProp step. The keys are the names of the
        sub-steps.

    Returns:
      (output, state1), both of which are NestedMaps.
      output is implementation-dependent and is defined by the output_signature
      parameter.
      state1 is a NestedMap where the keys are names of sub-steps and the values
      are state outputs from their FProp methods.
    """
    p = self.params
    graph_tensors = builder_layers.GraphTensors()
    graph_tensors.StoreTensor('prepared_inputs', prepared_inputs)
    graph_tensors.StoreTensor('step_inputs', step_inputs)
    state1 = py_utils.NestedMap()
    with tf.name_scope(p.name):
      for seq in self._seq:
        tf.logging.vlog(1, 'GraphStep: call %s', seq.name)
        external = None
        if seq.external_signature:
          external = prepared_inputs[seq.name]
        template = py_utils.NestedMap(inputs=seq.signature.inputs)
        packed = template.Transform(graph_tensors.GetTensor)
        input_args = packed.inputs[0]
        out, seq_state1 = seq.step.FProp(theta[seq.name], external, input_args,
                                         padding, state0[seq.name])
        graph_tensors.StoreTensor(seq.signature.outputs[0], out)
        state1[seq.name] = seq_state1
    template = py_utils.NestedMap(inputs=self.output_signature.inputs)
    output_tensors = template.Transform(graph_tensors.GetTensor).inputs[0]
    return output_tensors, state1


class IteratorStep(Step):
  """An iterator over the time dimension of some tensors.

  It's common to have a tensor of shape [batch, time, ...] or
  [time, batch, ...]. This object will step through the time dimension,
  producing tensors of shape [batch, ...] in succession.

  The input tensors are passed to PrepareExternalInputs. The step_inputs
  argument of FProp is unused.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('axis', 1, 'The time dimension of the tensors.')
    return p

  def PrepareExternalInputs(self, theta, external_inputs):
    """Prepares the input for iteration.

    Args:
      theta: unused.
      external_inputs: A NestedMap containing tensors. The time axis of each
        tensor should be params.axis.

    Returns:
      A prepared NestedMap (current the same as the input).
    """
    return external_inputs

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Returns the initial iterator state.

    Args:
      theta: unused.
      prepared_inputs: Output from a call to PrepareExternalInputs.
      batch_size: The number of items in the batch that FProp will process.

    Returns:
      An initial state NestedMap.
    """
    return py_utils.NestedMap(t=tf.constant(0, dtype=tf.int32))

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Returns a A single inference step for this step graph.

    Args:
      theta: unused.
      prepared_inputs: Output from a call to PrepareExternalInputs.
      step_inputs: unused.
      padding: unused.
      state0: A NestedMap of state variables produced by either ZeroState or a
        previous invocation of this FProp step.

    Returns:
      (output, state1), both of which are NestedMaps.
      output is implementation-dependent and is defined by the output_signature
      parameter.
      state1 is a NestedMap where the keys are names of sub-steps and the values
      are state outputs from their FProp methods.
    """
    del theta
    del step_inputs
    del padding

    def _Slice(tensor):
      """Return a slice of this tensor at time=state0.t."""
      shape = py_utils.GetShape(tensor)
      # All zeros except for t in the time dimension.
      # e.g. if params.axis=1, begin is [0, t, 0, 0, 0, ...]
      begin = tf.one_hot(self.params.axis, tf.rank(tensor), on_value=state0.t)
      # Same as shape, but with a 1 in the time dimension.
      # e.g. if params.axis=1, shape is [shape[0], 1, shape[2], shape[3], ...]
      size = tf.concat([
          shape[0:self.params.axis],
          tf.constant([1], dtype=tf.int32), shape[self.params.axis + 1:]
      ],
                       axis=0)
      # Make a slice where the time dimension is fixed at state0.t.
      time_slice = tf.slice(tensor, begin, size)
      # Remove the time dimension.
      return tf.squeeze(time_slice, axis=self.params.axis)

    output = prepared_inputs.Transform(_Slice)
    state1 = py_utils.NestedMap(t=state0.t + 1)
    return output, state1


class RecurrentStepWrapper(base_layer.BaseLayer):
  """A layer that wraps a step in a recurrent.Recurrent call."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('step', None, 'The step params that this class wraps.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateChild('step', self.params.step)

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    self.step.InstantiateVariables()
    super()._CreateChildrenVariables()

  def PrepareExternalInputs(self, theta, external_inputs):
    """See Step.PrepareExternalInputs."""
    return self.step.PrepareExternalInputs(theta.step, external_inputs)

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """See Step.ZeroState."""
    return self.step.ZeroState(theta.step, prepared_inputs, batch_size)

  def FProp(self, theta, prepared_inputs, inputs, padding, state0, **kwargs):
    """Runs a Step layer over multiple timesteps using Recurrent.

    Args:
      theta: A NestedMap containing weights' values of this layer and its
        children layers.
      prepared_inputs: External inputs returned by Step.PrepareExternalInputs().
      inputs: A NestedMap of inputs of shape [time, batch_size, dim].
      padding: A 0/1 float tensor of shape [time, batch_size]; 1.0 means that
        this batch element is empty in this step.
      state0: A NestedMap containing the initial recurrent state.
      **kwargs: Additional kwargs to pass to Recurrent.

    Returns:
      A tuple (outputs, state1).

      - outputs: A NestedMap containing the accumulated outputs of all steps,
        containing Tensors shaped [time, batch_size, dim].
      - state1: A NestedMap containing the accumulated recurrent states,
        containing Tensors shaped [time, batch_size, dim].
    """

    def RnnStep(recurrent_theta, recurrent_state0, recurrent_inputs):
      """Compute a single timestep."""
      output, state1 = self.step.FProp(
          theta=recurrent_theta.theta,
          prepared_inputs=recurrent_theta.prepared_inputs,
          step_inputs=recurrent_inputs.inputs,
          padding=recurrent_inputs.padding,
          state0=recurrent_state0.state)
      recurrent_state1 = py_utils.NestedMap(output=output, state=state1)
      return recurrent_state1, py_utils.NestedMap()

    # In order to pass Step outputs through Recurrent, they need to be
    # included as part of state.
    output0, _ = self.step.FProp(theta.step, prepared_inputs,
                                 inputs.Transform(lambda x: x[0]), padding[0],
                                 state0)

    accumulated_states, _ = recurrent.Recurrent(
        theta=py_utils.NestedMap(
            theta=theta.step, prepared_inputs=prepared_inputs),
        state0=py_utils.NestedMap(output=output0, state=state0),
        inputs=py_utils.NestedMap(inputs=inputs, padding=padding),
        cell_fn=RnnStep,
        **kwargs)

    return accumulated_states.output, accumulated_states.state
