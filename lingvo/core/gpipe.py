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

import contextlib
import copy

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import builder_layers
from lingvo.core import py_utils
from lingvo.core import recurrent
from lingvo.core import tshape


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
    return tf.random.uniform([2], maxval=seed_dtype.max, dtype=seed_dtype)

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
def CellFnFPropOpReplacementWrapper():
  """Hacks to replace certain unwanted tensorflow ops."""
  # Hack to replace GenerateStepSeedPair since global_step is not available
  # in temp graph created by optional.while.
  saved_get_op_seed = py_utils.GenerateStepSeedPair
  py_utils.GenerateStepSeedPair = GenerateStepSeedPair

  yield

  py_utils.GenerateStepSeedPair = saved_get_op_seed


def _ToTuple(x):
  if isinstance(x, list):
    return tuple(x)
  return x if isinstance(x, tuple) else (x,)


class FeatureExtractionLayer(base_layer.BaseLayer):
  """A layer that extrac features from a sequence of layers.

  FeatureExtractionLayer is a layer which connects a few layers in a sequence.
  It is also capable of fetching and forwarding activation endpoints.
  # TODO(huangyp): Make it a sublayer of builder_layers.SequentialLayer
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('variable_name_prefix', '',
             'Prefix for variable names in sub layers')
    p.Define('sub', [], 'A list of layers\' params.')
    p.Define('num_act_inputs', 0, 'Number of activation inputs.')
    p.Define('num_act_outputs', 0, 'Number of activation outputs.')
    p.Define('act_fetch_layers', [],
             'Names of fetch layers that cached extra activations')
    return p

  def __init__(self, params):
    super().__init__(params)
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


def PartitionSequentialLayers(params, num_partitions, *shapes):
  r"""Partition a layer composed of sequential layers.

  This routine strives to partition layers so that each partition costs roughly
  the same flops given the input shapes.

  Args:
    params: A layer param or a list of layer param.
    num_partitions: The desired number of partitions.
    *shapes: A tuple of tshape.Shape representing input tensors to the first
      layer.

  Returns:
    A list of FeatureExtractionLayer params.
  """

  # Recursively concatenate SequentialLayer into a list.
  def FlattenSeq(p):
    if isinstance(p, list):
      return p
    if p.cls not in [builder_layers.SequentialLayer, FeatureExtractionLayer]:
      return [p.Copy()]
    subs = []
    for _ in range(p.repeat):
      for s in p.sub:
        subs += FlattenSeq(s)
    return subs

  subs = FlattenSeq(params)

  assert len(shapes) == 1
  tf.logging.info('num_partitions: {} input_shape: {}'.format(
      num_partitions, shapes[0]))

  # Computes the estimate cost for each sub layer.
  total, histo, output_shapes = 0, [], []
  for i, s in enumerate(subs):
    s.name = 'cell_%03d' % i
    meta = s.cls.FPropMeta(s, *shapes)
    total += meta.flops
    histo.append(total)
    output_shapes.append(meta.out_shapes)
    shapes = meta.out_shapes
  tf.logging.vlog(1, 'len %d histogram = %s', len(subs), histo)

  # Computes the normalized cumulative histogram of the layer's cost.
  histo_pct = [float(x / total) for x in histo]
  tf.logging.vlog(1, 'cost pct = %s', histo_pct)

  # i-th sub layer is put into partition j, where j is roughly i-th cumulative
  # histogram times num_partitions.
  parts = [[] for _ in range(num_partitions)]
  parts_cost = [0] * num_partitions
  pre_hist_cost = 0
  for i, s in enumerate(subs):
    j = min(int(histo_pct[i] * num_partitions), num_partitions - 1)
    # The boundary at parts[j] where j > 0
    if j > 0 and not parts[j]:
      parts_cost[j - 1] = histo_pct[i - 1] - pre_hist_cost
      pre_hist_cost = histo_pct[i - 1]
    parts[j].append(s)

  parts_cost[num_partitions - 1] = 1.0 - pre_hist_cost
  seqs = []
  for i, pa in enumerate(parts):
    tf.logging.info('Partition %d #subs %d #cost %.3f', i, len(pa),
                         parts_cost[i])

    seqs.append(FeatureExtractionLayer.Params().Set(name='d%d' % i, sub=pa))
  return seqs


class SeqLayer(base_layer.BaseLayer):
  """Round-robin every children cells in cell_tpl among worker devices."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('before_tpl', [],
             'Config for the CNN layers that runs before pipelining.')
    p.Define('cell_tpl', [], 'A list of FeatureExtractionLayer layers.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._before_layers = []
    self._cells = []
    for l in p.before_tpl:
      self.CreateChild(l.name, l)
      self._before_layers.append((l.name, self.children[l.name]))
    for l in p.cell_tpl:
      self.CreateChild(l.name, l)
      self._cells.append((l.name, self.children[l.name]))

  def _CreateChildrenVariables(self):
    p = self.params

    num_cells = len(p.cell_tpl)
    before_tpl_device = ''
    cell_devices = [''] * num_cells
    if py_utils.use_tpu():
      cluster = self.cluster
      before_tpl_device = cluster.WorkerDeviceInModelSplit(0)
      cell_devices = [
          cluster.WorkerDeviceInModelSplit(i) for i in range(num_cells)
      ]

    for unused_name, l in self._before_layers:
      with tf.device(before_tpl_device):
        l.InstantiateVariables()

    for i, (unused_name, l) in enumerate(self._cells):
      with tf.device(cell_devices[i]):
        l.InstantiateVariables()

    super()._CreateChildrenVariables()

  def FProp(self, theta, *args):
    """Round-robin every children cells in cell_tpl among worker devices.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
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
    p = super().Params()
    p.Define('num_micro_batches', 1, 'Number of micro batches.')
    p.Define('micro_batch_size', None, 'Size of a micro batch.')
    p.Define('batch_dim', 0, 'The batch dimension.')
    p.Define('state_dtype', None, 'Externally specify dtype for states.')
    p.Define(
        'nested_map_fprop', False, 'Whether arguments and returns of '
        'cell fprop functions are nested maps')
    return p

  def _CalculateOutputShapes(self, input_shapes):
    """Calcuate the output shape of intermediate layers.

    Given the FPropMeta function in each FeatureExtractionLayer, calcuates
    the shapes of outputs of that layer. This is used to recover the shape
    information in StackedRecurrent.

    Args:
      input_shapes: NestedMap or tuple of input TensorShapes.

    Returns:
      Return a list of K + 1 NestedMaps or lists of tShape where K is
      the number of partitions.
    """
    p = self.params
    shapes = []

    # Converts TensorShape to tshape.Shape.
    def _ToTShape(x):
      if x is None:
        return None
      return tshape.Shape(x.as_list())

    shapes = py_utils.Transform(_ToTShape, input_shapes)
    shapes = _ToTuple(shapes)

    state_shapes = []
    for (_, cell) in self._before_layers:
      shapes = cell.FPropMeta(cell.params, *shapes).out_shapes

    state_shapes.append(shapes[0] if p.nested_map_fprop else shapes)

    for (_, cell) in self._cells:
      shapes = cell.FPropMeta(cell.params, *shapes).out_shapes
      state_shapes.append(shapes[0] if p.nested_map_fprop else shapes)

    return state_shapes

  def _get_state_dtype(self, *args):
    if self.params.state_dtype:
      return self.params.state_dtype
    if self.params.nested_map_fprop:
      inputs = args[0].Filter(lambda x: x is not None)
      return py_utils.Flatten(inputs)[0].dtype
    return args[0].dtype

  def _get_input_shapes(self, *args):
    p = self.params
    if p.nested_map_fprop:
      assert len(args) == 1
      assert isinstance(args[0], py_utils.NestedMap)
      input_tensors = py_utils.Flatten(args[0])
    else:
      input_tensors = _ToTuple(args)
    # Get batch size from the first tensor which is not None.
    mini_batch_size = None
    for input_tensor in input_tensors:
      if input_tensor is not None:
        mini_batch_size = input_tensor.get_shape().as_list()[p.batch_dim]
    assert mini_batch_size is not None
    micro_batch_size = p.micro_batch_size
    if not micro_batch_size:
      if p.num_micro_batches > mini_batch_size:
        p.num_micro_batches = mini_batch_size
      micro_batch_size = mini_batch_size // p.num_micro_batches
    if mini_batch_size is not None:
      if micro_batch_size * p.num_micro_batches != mini_batch_size:
        raise ValueError('micro_batch_size * num_micro_batches != batch_size.')

    input_shapes = ()
    for input_tensor in input_tensors:
      if input_tensor is not None:
        input_shape = input_tensor.get_shape().as_list()
        input_shape[p.batch_dim] = micro_batch_size
        input_shapes += (tf.TensorShape(input_shape),)
      else:
        input_shapes += (None,)

    if p.nested_map_fprop:
      input_shapes = py_utils.Pack(args[0], input_shapes)
    return input_shapes

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
    if self.do_eval and self.cluster.num_devices_per_split == 1:
      outputs = copy.copy(args)
      for (name, l) in self._before_layers + self._cells:
        outputs = _ToTuple(outputs)
        outputs = l.FProp(theta[name], *outputs)
      return outputs

    num_cells = len(p.cell_tpl)
    cluster = self.cluster

    # Compute shapes of input and output tensors.
    input_shapes = self._get_input_shapes(*args)
    state_dtype = self._get_state_dtype(*args)
    state_shapes = self._CalculateOutputShapes(input_shapes)
    tf.logging.info('state_shapes={}'.format(state_shapes))

    def GetCellFn(i):
      """Get the ith feature extraction layer."""

      def CellFn(theta, state0, inputs):
        """A cell fn is exectued inside of StackedRecurrent."""
        del state0

        def _FPropInputSetShape(name, t_shape):
          if t_shape is None:
            return None
          inputs[name].set_shape(t_shape.ToTensorShape().as_list())
          return inputs[name]

        if p.nested_map_fprop:
          # pylint: disable=protected-access
          fprop_inputs = state_shapes[i]._RecursiveMap(_FPropInputSetShape)
          # pylint: enable=protected-access
        else:
          fprop_inputs = []
          for input_idx, input_shape in enumerate(state_shapes[i]):
            name = 's{}'.format(input_idx)
            fprop_inputs.append(_FPropInputSetShape(name, input_shape))

        with py_utils.RemoveAssertContext(remove=True):
          with CellFnFPropOpReplacementWrapper():
            tf.logging.info('cell {} input {}'.format(i, fprop_inputs))
            mb_tensor = inputs[_MICRO_BATCH_STATE_NAME]
            SetOverWriteGlobalStep(mb_tensor)
            _, cell = self._cells[i]
            fprop_inputs = _ToTuple(fprop_inputs)
            outputs = cell.FProp(theta, *fprop_inputs)

        if p.nested_map_fprop:
          assert py_utils.IsCompatible(outputs, state_shapes[i + 1])
          state1 = outputs.Filter(lambda x: x is not None)
        else:
          state1 = py_utils.NestedMap()
          outputs = _ToTuple(outputs)
          assert len(outputs) == len(state_shapes[i + 1])
          for output_idx in range(len(outputs)):
            if outputs[output_idx] is not None:
              name = 's{}'.format(output_idx)
              state1[name] = outputs[output_idx]
        state1[_MICRO_BATCH_STATE_NAME] = mb_tensor
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

      def _TfZeros(t_shape):
        if t_shape is None:
          return None
        return tf.zeros(t_shape.ToTensorShape().as_list(), dtype=state_dtype)

      if p.nested_map_fprop:
        init_state = py_utils.Transform(_TfZeros, state_shapes[cell_idx + 1])
        init_state = init_state.Filter(lambda x: x is not None)
      else:
        init_state = py_utils.NestedMap()
        for output_idx, state in enumerate(state_shapes[cell_idx + 1]):
          state = _TfZeros(state)
          if state is not None:
            name = 's{}'.format(output_idx)
            init_state[name] = state
      init_state[_MICRO_BATCH_STATE_NAME] = tf.cast(0, dtype=state_dtype)
      init_states.append(init_state)

      devices.append(cluster.WorkerDeviceInModelSplit(cell_idx))

    cell_grads = [None] * num_cells
    cell_outs = [lambda x: x] * num_cells
    cell_out_grads = [lambda x: x] * num_cells

    with tf.device(devices[0]):
      previous = _ToTuple(args)
      for (name, l) in self._before_layers:
        previous = l.FProp(theta[name], *previous)
        previous = _ToTuple(previous)

      def _StackAndSplit(x):
        # Split tensors into microbatches.
        if x is None:
          return None
        return tf.stack(tf.split(x, p.num_micro_batches, axis=p.batch_dim))

      if p.nested_map_fprop:
        inputs = py_utils.Transform(_StackAndSplit, previous[0])
        inputs = inputs.Filter(lambda x: x is not None)
      else:
        inputs = py_utils.NestedMap()
        for output_idx, output_tensor in enumerate(previous):
          output_tensor = _StackAndSplit(output_tensor)
          if output_tensor is not None:
            name = 's{}'.format(output_idx)
            inputs[name] = output_tensor
      gs_tensor = py_utils.GetGlobalStep()
      inputs[_MICRO_BATCH_STATE_NAME] = tf.stack([
          tf.cast(gs_tensor * p.num_micro_batches + t, dtype=state_dtype)
          for t in range(p.num_micro_batches)
      ])
    tf.logging.info('pipeline input = {}'.format(inputs))
    output_state, _ = recurrent.StackedRecurrent(
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

      def _ReshapeRetVal(name, t_shape):
        """Restore shape for tensors in microbatches."""
        if t_shape is None:
          return None
        output_tensor = output_state[name]
        if p.batch_dim != 0:
          perm = list(range(1, p.batch_dim + 1)) + [0]
          perm += list(range(p.batch_dim + 1, t_shape.rank + 1))
          output_tensor = tf.transpose(output_tensor, perm=perm)
        output_shape = t_shape.ToTensorShape().as_list()
        output_shape[p.batch_dim] *= p.num_micro_batches
        output_tensor = tf.reshape(output_tensor, output_shape)
        return output_tensor

      # Construct the final return values from output_state.
      if p.nested_map_fprop:
        # pylint: disable=protected-access
        output_tensors = state_shapes[-1]._RecursiveMap(_ReshapeRetVal)
        # pylint: enable=protected-access
      else:
        output_tensors = []
        for output_idx, state_shape in enumerate(state_shapes[-1]):
          output_name = 's{}'.format(output_idx)
          output_tensor = _ReshapeRetVal(output_name, state_shape)
          output_tensors.append(output_tensor)
        if len(output_tensors) == 1:
          output_tensors = output_tensors[0]
        else:
          output_tensors = tuple(output_tensors)
      tf.logging.info('pipeline output = {}'.format(output_tensors))
      return output_tensors
