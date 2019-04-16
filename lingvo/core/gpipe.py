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
"""A recurrent model which enables pipelining model parallelism.

Reference:
'GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism'
https://arxiv.org/abs/1811.06965

Example implementation of Transformer Language model:
tasks/lm/layers.GPipeTransformerLm

Sample params for the one billion words task:
tasks/lm/params/one_billion_wds.OneBWdsGPipeTransformer.

More examples in machine translation, image classifications and others
will be included.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from six.moves import range

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import recurrent

_MICRO_BATCH_STATE_NAME = 'micro_batch_state'
_OVERWRITE_GLOBAL_STEP_COLLECTION = 'lingvo__OVERWRITE_GLOBAL_STEP_COLLECTION'


def GetOverWriteGlobalStep(graph=None):
  graph = graph or tf.get_default_graph()
  mb_tensors = graph.get_collection_ref(_OVERWRITE_GLOBAL_STEP_COLLECTION)
  if len(mb_tensors) == 1:
    mb_tensor = mb_tensors[0]
  else:
    mb_tensor = py_utils.GetGlobalStep()
  return mb_tensor


def SetOverWriteGlobalStep(tensor, graph=None):
  graph = graph or tf.get_default_graph()
  mb_tensors = graph.get_collection_ref(_OVERWRITE_GLOBAL_STEP_COLLECTION)
  if len(mb_tensors) == 1:
    mb_tensors[0] = tensor
  else:
    graph.add_to_collection(_OVERWRITE_GLOBAL_STEP_COLLECTION, tensor)


def GenerateStepSeedPair(p, unused_global_step=None, op_seed=None):
  """Override py_utils.GenerateStepSeedPair to use GetOverWriteGlobalStep."""
  seed_dtype = tf.int32 if py_utils.use_tpu() else tf.int64
  if p.is_inference and p.random_seed is None:
    # Unlike tf.random*, stateless random ops are completely determined by the
    # passed-in seeds. This means at inference time the same inputs will produce
    # the same outputs, even if the model is supposed to have randomness such as
    # dropout during inference. We inject additional randomness only during
    # inference if the graph is exported with random_seed=None as a workaround.
    return tf.random_uniform([2], maxval=seed_dtype.max, dtype=seed_dtype)

  with tf.name_scope('op_seed') as scope:
    global_step = tf.cast(GetOverWriteGlobalStep(), seed_dtype)
    step_seed = tf.cast(py_utils.GenerateSeedFromName(scope), seed_dtype)
    seeds = tf.stack([global_step, step_seed])

    if p.random_seed is not None:
      seeds += p.random_seed
    if op_seed is not None:
      seeds += op_seed
    return seeds


@contextlib.contextmanager
def CellFnFropOpReplacementWrapper():
  """Hacks to replace certain unwanted tensorflow ops."""
  # TODO(zhifengc/huangyp): Consider implementing assert_equal
  # op replacement for lingvo. As assert_equal doesn't support String on GPUs.
  # Hack to replace tf.assert_equal
  saved_assert_equal = tf.assert_equal
  # Hack to replace GenerateStepSeedPair since global_step is not available
  # in temp graph created by optional.while.
  saved_get_op_seed = py_utils.GenerateStepSeedPair

  # pylint: disable=unused-argument
  def NoOP(*args, **kwargs):
    return tf.no_op()

  # pylint: enable=unused-argument
  tf.assert_equal = NoOP  # Make assert_equal a no op.
  py_utils.GenerateStepSeedPair = GenerateStepSeedPair

  yield

  tf.assert_equal = saved_assert_equal
  py_utils.GenerateStepSeedPair = saved_get_op_seed


def _ToTuple(x):
  return x if isinstance(x, tuple) else (x,)


class FeatureExtractionLayer(base_layer.BaseLayer):
  """A layer that extrac features from a sequence of layers.

  FeatureExtractionLayer is a layer which connects a few layers in a sequence.
  It is also capable of fetching and forwarding activation endpoints.
  # TODO(huangyp): Allow keyworded argument dict in FProp.

  Args:
    fetch_activation_layers: names of fetch layers that extra activations.
    num_activation_inputs: # of activations forwarded from previous layers.
    num_activation_outputs: # of activations forwarded to next layers.
  """

  @classmethod
  def Params(cls):
    p = super(FeatureExtractionLayer, cls).Params()
    p.Define('variable_name_prefix', '',
             'Prefix for variable names in sub layers')
    p.Define('sub', [], 'A list of layers\' params.')
    p.Define('num_act_inputs', 0, 'Number of activation inputs.')
    p.Define('num_act_outputs', 0, 'Number of activation outputs.')
    p.Define('act_fetch_layers', [],
             'Names of fetch layers that cached extra activations')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(FeatureExtractionLayer, self).__init__(params)
    p = self.params
    assert p.num_act_inputs >= 0
    assert p.num_act_outputs >= 0
    p.act_fetch_layers = p.act_fetch_layers or []
    assert p.num_act_outputs == p.num_act_inputs + len(p.act_fetch_layers)
    self._seq = []
    for sub in p.sub:
      assert sub.name
      sub.name = p.variable_name_prefix + sub.name
      self.CreateChild(sub.name, sub)
      self._seq.append((sub.name, self.children[sub.name]))

  def FProp(self, theta, *args):
    p = self.params
    assert len(args) > p.num_act_inputs
    out_args = args[:-p.num_act_inputs] if p.num_act_inputs > 0 else args
    extra_args = args[-p.num_act_inputs:] if p.num_act_inputs > 0 else ()
    for (name, ch) in self._seq:
      th = theta[name]
      out_args = _ToTuple(out_args)
      out_args = ch.FProp(th, *out_args)
    # Append fetched activations to fprop outputs.
    for fetch_layer in p.act_fetch_layers:
      assert fetch_layer in self.children
      activation = self.children[fetch_layer].activation
      if isinstance(activation, (tuple, list)):
        activation = activation[0]
      extra_args += (activation,)
    if extra_args:
      out_args = _ToTuple(out_args) + extra_args
    return out_args

  @classmethod
  def FPropMeta(cls, p, *args):
    assert len(args) > p.num_act_inputs
    seq_args = args[:-p.num_act_inputs] if p.num_act_inputs > 0 else args
    extra_args = args[-p.num_act_inputs:] if p.num_act_inputs > 0 else ()
    total = 0
    act_fetch_metas = {}
    for sub in p.sub:
      meta = sub.cls.FPropMeta(sub, *seq_args)
      if sub.name in p.act_fetch_layers:
        act_fetch_metas[sub.name] = meta.out_shapes[0]
      total += meta.flops
      seq_args = meta.out_shapes
    for fetch_layer in p.act_fetch_layers:
      extra_args += (act_fetch_metas[fetch_layer],)
    return py_utils.NestedMap(flops=total, out_shapes=seq_args + extra_args)


class SeqLayer(base_layer.BaseLayer):
  """Round-robin every children cells in cell_tpl among worker devices."""

  @classmethod
  def Params(cls):
    p = super(SeqLayer, cls).Params()
    p.Define('before_tpl', [],
             'Config for the CNN layers that runs before pipelining.')
    p.Define('cell_tpl', [], 'A list of FeatureExtractionLayer layers.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SeqLayer, self).__init__(params)
    p = self.params
    assert p.name
    num_cells = len(p.cell_tpl)
    self._before_layers = []
    self._cells = []
    before_tpl_device = ''
    cell_devices = [''] * num_cells
    if py_utils.use_tpu():
      cluster = self.cluster
      before_tpl_device = cluster.WorkerDeviceInModelSplit(0)
      cell_devices = [
          cluster.WorkerDeviceInModelSplit(i) for i in range(num_cells)
      ]
    for l in p.before_tpl:
      with tf.device(before_tpl_device):
        assert l.name
        self.CreateChild(l.name, l)
        self._before_layers.append((l.name, self.children[l.name]))
    for i, l in enumerate(p.cell_tpl):
      with tf.device(cell_devices[i]):
        assert l.name
        self.CreateChild(l.name, l)
        self._cells.append((l.name, self.children[l.name]))

  def FProp(self, theta, *args):
    """Round-robin every children cells in cell_tpl among worker devices.

    Args:
      theta: A NestedMap object containing weights' values of this
          layer and its children layers.
      *args: Input args

    Returns:
      A list contains one tensor of [batch_size, feature_height, feature_width,
        channel].
    """
    num_layers = len(self.params.cell_tpl)
    cluster = self.cluster

    for (name, l) in self._before_layers:
      l_theta = theta[name]
      args = _ToTuple(args)
      args = l.FProp(l_theta, *args)
    for i in range(num_layers):
      with tf.device(cluster.WorkerDeviceInModelSplit(i)):
        cell_name, cell = self._cells[i]
        args = _ToTuple(args)
        args = cell.FProp(theta[cell_name], *args)

    return args


class PipeliningLayer(SeqLayer):
  """Pipelining a sequence of layers on multiple devices."""

  @classmethod
  def Params(cls):
    p = super(PipeliningLayer, cls).Params()
    p.Define('num_micro_batches', 1, 'Number of micro batches.')
    p.Define('batch_dim', 0, 'The batch dimension.')
    p.Define('state_dtype', None, 'Externally specify dtype for states.')
    return p

  def _CalculateOutputShapes(self, input_shapes):
    """Calcuate the output shape of intermediate layers.

    Given the FPropMeta function in each FeatureExtractionLayer, calcuates
    the shapes of outputs of that layer. This is used to recover the shape
    information in StackedRecurrent.

    Args:
      input_shapes: tuple of input shapes

    Returns:
      Return a list of K + 1 lists of shapes where K is the number of
      partitions.
    """
    state_shapes = []

    for (_, before_layer) in self._before_layers:
      meta = before_layer.FPropMeta(before_layer.params, *input_shapes)
      input_shapes = meta.out_shapes

    state_shape_list = []
    for state_shape in input_shapes:
      if state_shape is not None:
        state_shape_list.append(state_shape.as_list())
      else:
        state_shape_list.append(None)
    state_shapes.append(state_shape_list)

    for (_, cell) in self._cells:
      meta = cell.FPropMeta(cell.params, *input_shapes)
      input_shapes = meta.out_shapes
      state_shape_list = []
      for state_shape in input_shapes:
        if state_shape is not None:
          state_shape_list.append(state_shape.as_list())
        else:
          state_shape_list.append(None)
      state_shapes.append(state_shape_list)
    return state_shapes

  def FProp(self, theta, *args):
    """Run multiple cells in different devices in a pipelining manner.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      *args: Non-keyworded variable length argument list of input tensors.

    Returns:
      A list of output tensors
    """
    # TODO(huangyp): handle optional None inputs.
    p = self.params
    if p.is_eval:
      outputs = _ToTuple(args)
      for (name, l) in self._before_layers:
        outputs = _ToTuple(outputs)
        outputs = l.FProp(theta[name], *outputs)
      for (name, l) in self._cells:
        outputs = _ToTuple(outputs)
        outputs = l.FProp(theta[name], *outputs)
      return outputs

    num_cells = len(p.cell_tpl)
    cluster = self.cluster

    # Compute shapes of input and output tenors.
    input_tenors = _ToTuple(args)
    mini_batch_size = input_tenors[0].get_shape().as_list()[p.batch_dim]
    if p.state_dtype:
      state_dtype = p.state_dtype
    else:
      state_dtype = input_tenors[0].dtype
    if p.num_micro_batches > mini_batch_size:
      p.num_micro_batches = mini_batch_size
    micro_batch_size = mini_batch_size // p.num_micro_batches

    input_shapes = ()
    for input_tensor in input_tenors:
      if input_tensor is not None:
        input_shape = input_tensor.get_shape().as_list()
        input_shape[p.batch_dim] = micro_batch_size
        input_shapes += (tf.TensorShape(input_shape),)
      else:
        input_shapes += (None,)

    state_shapes = self._CalculateOutputShapes(input_shapes)

    def GetCellFn(i):
      """Get the ith feature extraction layer."""

      def CellFn(theta, state0, inputs):
        """A cell fn is exectued inside of StackedRecurrent."""
        del state0
        frop_inputs = []
        for input_idx in range(len(state_shapes[i])):
          name = 's{}'.format(input_idx)
          if state_shapes[i][input_idx] is not None:
            inputs[name].set_shape(state_shapes[i][input_idx])
            frop_inputs.append(inputs[name])
          else:
            frop_inputs.append(None)

        with CellFnFropOpReplacementWrapper():
          tf.logging.info('cell {} input {}'.format(i, frop_inputs))
          mb_tensor = inputs[_MICRO_BATCH_STATE_NAME]
          SetOverWriteGlobalStep(mb_tensor)
          _, cell = self._cells[i]
          outputs = cell.FProp(theta, *frop_inputs)

        state1 = py_utils.NestedMap()
        state1[_MICRO_BATCH_STATE_NAME] = mb_tensor
        outputs = _ToTuple(outputs)
        assert len(outputs) == len(state_shapes[i + 1])
        for output_idx in range(len(outputs)):
          if outputs[output_idx] is not None:
            name = 's{}'.format(output_idx)
            state1[name] = outputs[output_idx]
        return state1, py_utils.NestedMap()

      return CellFn

    cell_fns = []
    accumulator_layers = []
    thetas = []
    init_states = []
    devices = []
    for cell_idx in range(num_cells):
      cell_name, cell = self._cells[cell_idx]
      accumulator_layers.append(cell)
      cell_fns.append(GetCellFn(cell_idx))
      thetas.append(theta[cell_name])
      init_state = py_utils.NestedMap()
      init_state[_MICRO_BATCH_STATE_NAME] = tf.cast(0, dtype=state_dtype)
      for output_idx in range(len(state_shapes[cell_idx + 1])):
        name = 's{}'.format(output_idx)
        if state_shapes[cell_idx + 1][output_idx] is not None:
          init_state[name] = tf.zeros(
              state_shapes[cell_idx + 1][output_idx], dtype=state_dtype)
      init_states.append(init_state)
      devices.append(cluster.WorkerDeviceInModelSplit(cell_idx))

    cell_grads = [None] * num_cells
    cell_outs = [lambda x: x] * num_cells
    cell_out_grads = [lambda x: x] * num_cells

    with tf.device(devices[0]):
      previous = input_tenors
      for (name, l) in self._before_layers:
        previous = l.FProp(theta[name], *previous)
        previous = _ToTuple(previous)
      inputs = py_utils.NestedMap()
      gs_tensor = py_utils.GetGlobalStep()
      inputs[_MICRO_BATCH_STATE_NAME] = tf.stack([
          tf.cast(gs_tensor * p.num_micro_batches + t, dtype=state_dtype)
          for t in range(p.num_micro_batches)
      ])

      # TODO(huangyp, dehao): apply dehao's trick to reshape the input tensor
      # to [p.num_micro_batches, -1, 128].
      for output_idx, output_tenor in enumerate(previous):
        name = 's{}'.format(output_idx)
        if output_tenor is not None:
          output_tenor = tf.stack(
              tf.split(output_tenor, p.num_micro_batches, axis=p.batch_dim))
          inputs[name] = output_tenor

    output, _ = recurrent.StackedRecurrent(
        devices=devices,
        cell_fns=cell_fns,
        cell_grads=cell_grads,
        cell_outs=cell_outs,
        cell_out_grads=cell_out_grads,
        thetas=thetas,
        init_states=init_states,
        inputs=inputs,
        accumulator_layers=accumulator_layers,
        unused_acc_state=True)

    with tf.device(devices[-1]):
      output_tensors = []
      for output_idx in range(len(state_shapes[-1])):
        state_shape = state_shapes[-1][output_idx]
        if state_shape is None:
          output_tensors.append(None)
          continue
        output_name = 's{}'.format(output_idx)
        output_tensor = output[output_name]
        if p.batch_dim != 0:
          perm = list(range(1, p.batch_dim + 1)) + [0]
          perm += list(range(p.batch_dim + 1, len(state_shape) + 1))
          output_tensor = tf.transpose(output_tensor, perm=perm)
        state_shape[p.batch_dim] *= p.num_micro_batches
        output_tensor = tf.reshape(output_tensor, state_shape)
        output_tensors.append(output_tensor)
      tf.logging.info('pipeline output = {}'.format(output_tensors))
      if len(output_tensors) == 1:
        return output_tensors[0]
      return tuple(output_tensors)
