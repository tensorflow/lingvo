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
"""Lingvo layers that depend on layers and gpipe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_function
from lingvo.core import base_layer
from lingvo.core import gpipe
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core.gpipe import FeatureExtractionLayer
from lingvo.core.gpipe import PipeliningLayer


class GPipeTransformerLayer(layers_with_attention.TransformerLayer):
  """GPipe compatible transformer layer."""

  @classmethod
  def Params(cls):
    """Configs for TransformerStack."""
    p = super(GPipeTransformerLayer, cls).Params()
    p.Define(
        'is_transparent', False,
        'If set, encoder outputs a list of layer outputs while decoder '
        'expects a list of source input vectors.')
    p.Define(
        'num_transparent_outputs', 0,
        'Number of transparent outputs. Only positive if this is the '
        'last encoder')
    p.Define(
        'transparent_merger_tpl', None,
        'Merger op for layer outputs. Not none if this is the last encoder')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeTransformerLayer, self).__init__(params)
    p = self.params
    if p.is_transparent and p.num_transparent_outputs > 0:
      transparent_params = []
      for i in range(p.num_transparent_outputs):
        transparent_param = p.transparent_merger_tpl.Copy()
        transparent_param.name = 'transparent_%d' % i
        transparent_params.append(transparent_param)
      self.CreateChildren('transparent_merger', transparent_params)
    assert p.name

  def FProp(self, theta, source_vecs, source_paddings, target_vecs,
            target_paddings, source_segment_id, target_segment_id,
            *more_source_vecs):
    p = self.params
    with tf.name_scope(p.name):
      if p.has_aux_atten:  # Decoder FProp
        assert target_vecs is not None
        assert target_paddings is not None
        h, _ = super(GPipeTransformerLayer, self).FProp(
            theta,
            target_vecs,
            target_paddings,
            aux_vecs=source_vecs,
            aux_paddings=source_paddings,
            source_segment_id=target_segment_id,
            aux_segment_id=source_segment_id)
        h.set_shape(target_vecs.shape)
        if p.is_transparent and more_source_vecs:
          source_vecs = more_source_vecs[0]
          more_source_vecs = more_source_vecs[1:]
        return (source_vecs, source_paddings, h, target_paddings,
                source_segment_id, target_segment_id) + more_source_vecs
      else:  # Encoder FProp
        h, _ = super(GPipeTransformerLayer, self).FProp(
            theta,
            source_vecs,
            source_paddings,
            source_segment_id=source_segment_id)
        h.set_shape(source_vecs.shape)
        if p.is_transparent:
          more_source_vecs += (source_vecs,)
          if p.num_transparent_outputs > 0:  # Merger layer.
            transformer_output = []
            for i in range(p.num_transparent_outputs):
              merged_outputs = self.transparent_merger[i].FProp(
                  theta.transparent_merger[i], list(more_source_vecs + (h,)))
              transformer_output.append(merged_outputs)
            h = transformer_output[0]
            if p.num_transparent_outputs == 1:
              more_source_vecs = ()
            else:
              more_source_vecs = tuple(transformer_output[1:])
        return (h, source_paddings, target_vecs, target_paddings,
                source_segment_id, target_segment_id) + more_source_vecs

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    # TODO(huangyp): return accurate estimate of flops.
    py_utils.CheckShapes((inputs,))
    flops_per_element = 5
    src_time, source_batch, dim = inputs.as_list()
    flops = flops_per_element * src_time * src_time * source_batch * dim
    args = args if isinstance(args, tuple) else (args,)
    if p.is_transparent:
      if p.has_aux_atten:  # Decoder FPropMeta
        args = args[:-1] if len(args) > 5 else args
      else:
        if p.num_transparent_outputs == 0:
          args += (inputs,)
        elif p.num_transparent_outputs == 1:
          # Switch back to non-transparent mode for decoder.
          args = args[:5]
        else:
          args += (inputs,) * (p.num_transparent_outputs - len(args) + 4)
    return py_utils.NestedMap(flops=flops, out_shapes=(inputs,) + args)


class GPipeTransformerStack(PipeliningLayer):
  """Stacked self- multi-head attention and fully connected layers.

  With optional layer normalization applied to the final output.

  See 'Attention Is All You Need' https://arxiv.org/abs/1706.03762
  for details.
  """

  @classmethod
  def Params(cls):
    """Configs for TransformerStack."""
    p = super(GPipeTransformerStack, cls).Params()

    # GPipe Related
    p.Define('num_splits', 1, 'Deprecated.')
    p.Define(
        'splits', 1,
        'Number of splits, or list of integers specifying the ending index for '
        'each split in ascending order. Last index should be num_layers.')

    # Transformer related
    p.Define('model_dim', 1024, 'Characteristic depth (dimension).')
    p.Define('num_encoder_layers', 0, 'Number of transformer encoder layers.')
    p.Define('num_decoder_layers', 0, 'Number of transformer encoder layers.')
    p.Define('encoder_tpl', GPipeTransformerLayer.Params(),
             'TransformerLayer Encoder params tpl.')
    p.Define('decoder_tpl', GPipeTransformerLayer.Params(),
             'TransformerLayer Decoder params tpl.')
    p.Define('transparent_merger_dropout_prob', 0.1,
             'Dropout probability in WeightedSumLayer')
    p.Define(
        'is_transparent', False,
        'If set, encoder outputs a merger of embeddings and '
        'layer outputs.')
    p.Define(
        'num_transparent_outputs', 0,
        'If set, the transparent merger outputs this number of weighted sums. '
        'Defaults to number of decoder layers if transparent.')
    p.Define('packed_input', False,
             'If True, assumes multiple training samples per input.')
    p.encoder_tpl.has_aux_atten = False
    p.decoder_tpl.has_aux_atten = True
    p.decoder_tpl.mask_self_atten = True
    p.batch_dim = 1
    return p

  @base_layer.initializer
  def __init__(self, params):
    p = params.Copy()
    num_layers = p.num_encoder_layers + p.num_decoder_layers

    if isinstance(p.splits, (list, tuple)):
      assert p.splits[-1] == num_layers
      for i, j in zip(p.splits[:-1], p.splits[1:]):
        assert i < j, 'Splits must be in increasing order.'
    else:
      num_splits = max(p.splits, p.num_splits)  # Supporting deprecated param.
      layers_per_split = num_layers // num_splits
      p.splits = []
      for i in range(num_splits):
        p.splits.append((i + 1) * layers_per_split)

    with tf.variable_scope(p.name):
      p.encoder_tpl.source_dim = p.model_dim
      p.decoder_tpl.source_dim = p.model_dim
      transformers = []
      for i in range(p.num_encoder_layers):
        params = p.encoder_tpl.Copy()
        params.name = 'encoder_%d' % (i)
        params.is_transparent = p.is_transparent
        params.packed_input = p.packed_input
        # Use DeterministicDropoutLayer when used in temp graphs.
        if len(p.splits) > 1 or p.num_micro_batches > 1:
          params = self.SetupDeterministicDropout(params)
        assert not params.has_aux_atten
        last_layer = (i == p.num_encoder_layers - 1)
        if p.is_transparent and last_layer:
          transparent_merger_tpl = DeterministicWeightedSumLayer.Params()
          transparent_merger_tpl.num_sources = p.num_encoder_layers + 1
          transparent_merger_tpl.dropout_tpl.keep_prob = (
              1 - p.transparent_merger_dropout_prob)
          params.transparent_merger_tpl = transparent_merger_tpl
          params.num_transparent_outputs = p.num_transparent_outputs
        transformers.append(params)
      for i in range(p.num_decoder_layers):
        params = p.decoder_tpl.Copy()
        params.name = 'decoder_%d' % (i)
        params.mask_self_atten = True
        params.packed_input = p.packed_input
        params.is_transparent = p.is_transparent and (
            p.num_transparent_outputs == p.num_decoder_layers)
        if len(p.splits) > 1 or p.num_micro_batches > 1:
          params = self.SetupDeterministicDropout(params)
        assert params.has_aux_atten
        transformers.append(params)
      cells = []
      cell_start = 0
      for split, cell_end in enumerate(p.splits):
        sub = transformers[cell_start:cell_end]
        cell = FeatureExtractionLayer.Params().Set(
            name='cell_{}'.format(split), sub=sub)
        cells.append(cell)
        cell_start = cell_end
      p.cell_tpl = cells
    super(GPipeTransformerStack, self).__init__(p)

  def SetupDeterministicDropout(self, params):
    """Replaced dropout layers in transformer with deterministic ones."""
    params.tr_atten_tpl.residual_dropout_tpl = (
        DeterministicDropoutLayer.Params())
    params.tr_atten_tpl.atten_tpl.atten_dropout_deterministic = True
    params.tr_atten_tpl.atten_tpl.inner_atten_params \
    .atten_dropout_deterministic = True
    params.tr_fflayer_tpl.residual_dropout_tpl = (
        DeterministicDropoutLayer.Params())
    params.tr_fflayer_tpl.fflayer_tpl.dropout = (
        DeterministicDropoutLayer.Params())
    return params

  def GetEncoders(self):
    encoders = []
    p = self.params
    cell_start = 0
    for split, cell_end in enumerate(p.splits):
      for encoder_id in xrange(cell_start, cell_end):
        if encoder_id >= p.num_encoder_layers:
          break
        encoder_l = self.children['cell_{}'.format(split)].children[
            'encoder_{}'.format(encoder_id)]
        encoders.append(encoder_l)
      cell_start = cell_end
    return encoders

  def GetDecoders(self):
    decoders = []
    p = self.params
    cell_start = 0
    for split, cell_end in enumerate(p.splits):
      for layer_id in xrange(cell_start, cell_end):
        decoder_id = layer_id - p.num_encoder_layers
        if decoder_id < 0:
          continue
        decoder_l = self.children['cell_{}'.format(split)].children[
            'decoder_{}'.format(decoder_id)]
        decoders.append(decoder_l)
      cell_start = cell_end
    assert len(decoders) == p.num_decoder_layers
    return decoders

  def EncoderFPropDefaultTheta(self,
                               source_vecs,
                               source_paddings,
                               source_segment_id=None):
    p = self.params
    more_source_vecs = ()
    for encoder_l in self.GetEncoders():
      encoder_outs = encoder_l.FProp(encoder_l.theta, source_vecs,
                                     source_paddings, None, None, None, None,
                                     *more_source_vecs)
      source_vecs = encoder_outs[0]
      more_source_vecs = encoder_outs[6:]

    assert p.is_transparent or not more_source_vecs

    if p.is_transparent and p.num_transparent_outputs > 1:
      source_vecs = more_source_vecs + (source_vecs,)
      if p.is_eval:
        source_vecs = tf.stack(list(source_vecs), 3)
    return source_vecs

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            target_vecs=None,
            target_paddings=None,
            source_segment_id=None,
            target_segment_id=None):
    """Transforms source sequence of Tensors with Transformers layers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_vecs: A sequence of input Tensors of [time, batch, dim] shape.
      source_paddings: A sequence of 0s and 1s indicating input paddings of
        [time, batch] shape.
      target_vecs: [target_time, target_batch, dim]
      target_paddings: [target_time, target_batch]
      source_segment_id: A sequence of ints indicating source segment ids of
        [time, batch] shape.
      target_segment_id: A sequence of ints indicating target segment ids of
        [time, batch] shape.

    Returns:
      transformer_output with shape [time, batch, dim]
    """
    p = self.params
    if p.num_decoder_layers > 0:
      assert target_vecs is not None
      assert target_paddings is not None
    if p.packed_input:
      assert source_segment_id is not None, (
          'Need to specify src_segment_id if packed input is supported.')
    gpipe_outputs = super(GPipeTransformerStack, self).FProp(
        theta, source_vecs, source_paddings, target_vecs, target_paddings,
        source_segment_id, target_segment_id)
    if p.num_decoder_layers > 0:
      transformer_output = gpipe_outputs[2]
    else:
      transformer_output = gpipe_outputs[0]
      more_source_vecs = gpipe_outputs[6:]
      if more_source_vecs:
        transformer_output = more_source_vecs + (gpipe_outputs[0],)
    return transformer_output


class AddingAccumulator(base_layer.Accumulator):
  """Accumulator for the sufficient statistics."""

  def __init__(self, shape, dtype):
    super(AddingAccumulator, self).__init__()
    self.dtype = dtype
    self.shape = shape

  def DefaultValue(self):
    return tf.zeros(self.shape, dtype=self.dtype)

  def Update(self, value):
    self.SetValue(self.GetValue() + tf.cast(value, self.dtype))


class BatchNormLayerNoPadding(base_layer.BaseLayer):
  """Batchnorm layer without padding."""

  @classmethod
  def Params(cls):
    p = super(BatchNormLayerNoPadding, cls).Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define(
        'decay', 0.997,
        'Decay in updating the mean and variance moving average used in'
        ' batch normalization.')
    p.Define('epsilon', 0.001,
             'Small float added to variance to avoid dividing by zero.')
    p.Define(
        'bn_group_size', 1,
        'The number of shards participating in normalization when distributed'
        ' batchnorm is used. Only used for TPU.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BatchNormLayerNoPadding, self).__init__(params)
    p = self.params
    assert p.name
    assert p.dim > 0
    p.fprop_dtype = None
    self.RegisterAccumulator('counts', AddingAccumulator([], p.dtype))
    self.RegisterAccumulator('mean_ss', AddingAccumulator([p.dim], p.dtype))
    self.RegisterAccumulator('variance_ss', AddingAccumulator([p.dim], p.dtype))
    collections = [
        self.__class__.__name__ + '_vars', py_utils.SKIP_LP_REGULARIZATION
    ]
    pc = py_utils.WeightParams(
        shape=[p.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=collections)

    with tf.variable_scope(p.name):
      self.CreateVariable('beta', pc)
      # Note, The real gamma to use is 1 + gamma.
      self.CreateVariable('gamma', pc, lambda x: 1.0 + x)

      moving_collections = [
          'moving_vars', tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
          self.__class__.__name__ + '_vars'
      ]
      mva = py_utils.WeightParams(
          shape=[p.dim],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=moving_collections)
      # Two statistics.
      self.CreateVariable('moving_mean', mva, trainable=False)
      mvv = py_utils.WeightParams(
          shape=[p.dim],
          init=py_utils.WeightInit.Constant(1.0),
          dtype=p.dtype,
          collections=moving_collections)
      self.CreateVariable('moving_variance', mvv, trainable=False)

  def PostTrainingStepUpdate(self, global_step):
    p = self.params
    counts = self.accumulators.counts.GetValue()
    mean_ss = self.accumulators.mean_ss.GetValue()
    variance_ss = self.accumulators.variance_ss.GetValue()
    mean, variance = tf.nn.normalize_moments(counts, mean_ss, variance_ss, None)
    decay = tf.convert_to_tensor(1.0 - p.decay, p.dtype)
    with tf.name_scope(p.name) as scope:
      with tf.colocate_with(self.vars.moving_mean):
        mean_update = tf.assign_sub(
            self.vars.moving_mean,
            (self.vars.moving_mean - tf.cast(mean, p.dtype)) * decay,
            name='moving_mean_update')
      with tf.colocate_with(self.vars.moving_variance):
        var_update = tf.assign_sub(
            self.vars.moving_variance,
            (self.vars.moving_variance - tf.cast(variance, p.dtype)) * decay,
            name='moving_variance_update')
      py_utils.CheckNumerics(
          self.vars.moving_mean,
          'moving mean of {} failed numeric check'.format(scope))
      py_utils.CheckNumerics(
          self.vars.moving_variance,
          'moving variance of {} failed numeric check'.format(scope))
    self.accumulators.counts.Reset()
    self.accumulators.mean_ss.Reset()
    self.accumulators.variance_ss.Reset()
    return tf.group(mean_update, var_update)

  def _Moments(self, inputs, group_size):
    """Computes mean and variance over N,H,W dimensions in inputs."""
    counts, mean_ss, variance_ss, _, = tf.nn.sufficient_statistics(
        inputs, axes=[0, 1, 2], keep_dims=False)
    self.accumulators.counts.Update(counts)
    self.accumulators.mean_ss.Update(mean_ss)
    self.accumulators.variance_ss.Update(variance_ss)
    if py_utils.use_tpu() and group_size > 1:
      num_shards = tpu_function.get_tpu_context().number_of_shards
      assert num_shards >= group_size
      assert num_shards % group_size == 0
      num_groups = num_shards // group_size
      group_assignment = []
      for g in range(num_groups):
        replica_ids = [g * group_size + i for i in range(group_size)]
        group_assignment.append(replica_ids)
      counts *= group_size
      mean_ss = tf.contrib.tpu.cross_replica_sum(mean_ss, group_assignment)
      variance_ss = tf.contrib.tpu.cross_replica_sum(variance_ss,
                                                     group_assignment)
    mean, variance = tf.nn.normalize_moments(counts, mean_ss, variance_ss, None)
    return mean, variance

  def FProp(self, theta, inputs):
    """Apply batch normalization.

    Using the implementation in github.com/
    tensorflow/tpu/blob/master/models/official/amoeba_net/network_utils.py#L550

    Args:
      theta: A nested map object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., dim].

    Returns:
      Output after applying batch normalization, with the same shape as
      'inputs'.
    """
    p = self.params
    inputs_dtype = inputs.dtype
    inputs = tf.cast(inputs, p.dtype)
    inputs = py_utils.with_dependencies(
        [py_utils.assert_shape_match([tf.shape(inputs)[-1]], [p.dim])], inputs)
    with tf.name_scope(p.name) as scope:
      if p.is_eval:
        outputs = tf.nn.batch_normalization(inputs, self.vars.moving_mean,
                                            self.vars.moving_variance,
                                            theta.beta, theta.gamma, p.epsilon)
      else:
        mean, variance = self._Moments(inputs, p.bn_group_size)
        mean = py_utils.CheckNumerics(
            mean, 'mean of {} failed numeric check'.format(scope))
        variance = py_utils.CheckNumerics(
            variance, 'variance of {} failed numeric check'.format(scope))
        outputs = tf.nn.batch_normalization(inputs, mean, variance, theta.beta,
                                            theta.gamma, p.epsilon)
      outputs.set_shape(inputs.get_shape())
      return tf.cast(outputs, inputs_dtype)

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    flops_per_element = 10  # Approximately 10 flops per element.
    return py_utils.NestedMap(
        flops=inputs.num_elements() * flops_per_element, out_shapes=(inputs,))


class ParallelLayer(base_layer.BaseLayer):
  """A layer which concats or adds a few layers in parallel."""

  @classmethod
  def Params(cls):
    p = super(ParallelLayer, cls).Params()
    p.Define(
        'sub', [], 'A list of sub layers\' params. Each sub layer\'s '
        'FProp must return one Tensor or a tuple of Tensors. '
        'Their return values then can be merged according to the '
        'merge method. ')
    p.Define('reduce_fn', tf.add_n, 'tf.add_n|lambda x: tf.concat(x, -1)')
    p.Define(
        'reduce_meta', None, 'Callable to compute the meta of reduce_fn '
        'It takes a list of tuples of TensorShape, and returns a '
        'NestedMap with flops and out_shapes, etc.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ParallelLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert len(p.sub) > 1
    with tf.variable_scope(p.name):
      self.CreateChildren('sub', p.sub)

  def FProp(self, theta, *args):
    """Compute the output given a list of inputs.

    Args:
      theta: weights.
      *args: A list of inputs, providing input to eac sub layer. Requiring
        len(args) == len(p.sub)

    Returns:
      A reduced output using p.reduce_fn
    """
    p = self.params
    assert len(args) == len(self.sub)
    # Computes sub layers in parallel.
    outputs = []
    for (ch, th, inputs) in zip(self.sub, theta.sub, list(args)):
      out = ch.FProp(th, inputs)
      if isinstance(out, (list, tuple)):
        outputs.append(list(out))
      else:
        outputs.append([out])

    assert all(len(x) == len(outputs[0])
               for x in outputs), 'outputs: {}'.format(outputs)
    rets = []
    for lane in zip(*outputs):
      if any(x is None for x in lane):
        rets.append(None)
      else:
        rets.append(p.reduce_fn(lane))

    return tuple(rets) if len(rets) > 1 else rets[0]

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    total = 0
    outputs = []
    for sub in p.sub:
      meta = sub.cls.FPropMeta(sub, *args)
      py_utils.CheckShapes(meta.out_shapes)
      total += meta.flops
      outputs.append(meta.out_shapes)

    meta = p.reduce_meta(outputs)
    py_utils.CheckShapes(meta.out_shapes)
    meta.flops += total
    return meta


class ReduceMeanLayer(base_layer.BaseLayer):
  """Construct to a layer that returns the spatial mean of the inputs."""

  @base_layer.initializer
  def __init__(self, params):
    super(ReduceMeanLayer, self).__init__(params)
    p = self.params
    assert p.name

  def FProp(self, theta, inputs, *args):
    return tf.reduce_mean(inputs, [1, 2])

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    py_utils.CheckShapes((inputs,))
    input_shape_list = inputs.as_list()
    out_shape = tf.TensorShape(input_shape_list[0:1] + input_shape_list[3:])
    return py_utils.NestedMap(
        flops=inputs.num_elements(), out_shapes=(out_shape,))


class TupleLayer(base_layer.BaseLayer):
  """A helper layer that returns a a duplicated tuple of the first argument."""

  @classmethod
  def Params(cls):
    p = super(TupleLayer, cls).Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TupleLayer, self).__init__(params)
    p = self.params
    assert p.name

  def FProp(self, theta, inputs, *args):
    return inputs, inputs

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(flops=0, out_shapes=(inputs, inputs))


class DeterministicDropoutLayer(base_layer.BaseLayer):
  """Dropout fully deterministic by scope_name and time step."""

  @classmethod
  def Params(cls):
    p = super(DeterministicDropoutLayer, cls).Params()
    p.Define('keep_prob', 1.0, 'Keep probability.')
    p.Define(
        'burn_in_steps', 0,
        'The droppath keep probability will increase linearly with time '
        'until drop_path_burn_in_steps')
    p.Define(
        'noise_shape_dim', None, 'Set noise_shape to input shape if -1 '
        'Otherwise noise_shape[noise_shape_dim]=inputs[noise_shape_dim]')
    p.Define('num_micro_batches', 128, 'Maximum number of micro-batches')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(DeterministicDropoutLayer, self).__init__(params)
    p = self.params
    assert p.keep_prob >= 0.0
    assert p.burn_in_steps >= 0
    cluster_params = self.cluster.params.Copy()
    if p.burn_in_steps > 0 and cluster_params.mode == 'sync':
      cluster_params.job = 'trainer_client'
      my_cluster = cluster_params.cls(cluster_params)
      splits = my_cluster.num_splits_per_client
      p.burn_in_steps /= splits

  def FProp(self, theta, inputs):
    """Apply dropout to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.

    Returns:
      inputs with dropout applied at training time.
    """
    p = self.params
    if p.keep_prob >= 1.0 or p.is_eval:
      return inputs

    with tf.name_scope(p.name):
      mb_tensor = gpipe.GetOverWriteGlobalStep()
      if p.burn_in_steps > 0:
        current_step = tf.cast(mb_tensor // p.num_micro_batches, inputs.dtype)
        current_ratio = current_step / tf.cast(p.burn_in_steps, inputs.dtype)
        current_ratio = tf.minimum(tf.cast(1.0, inputs.dtype), current_ratio)
        keep_prob = (1 - current_ratio * (1 - p.keep_prob))
      else:
        keep_prob = tf.cast(p.keep_prob, inputs.dtype)

      seeds = gpipe.GenerateStepSeedPair(p)
      noise_shape = py_utils.GetShape(inputs)
      if p.noise_shape_dim and p.noise_shape_dim < inputs.shape.ndims:
        for d in range(inputs.shape.ndims):
          if d != p.noise_shape_dim:
            noise_shape[d] = 1
      random_tensor = (
          tf.cast(keep_prob, tf.float32) +
          tf.contrib.stateless.stateless_random_uniform(
              noise_shape, seed=seeds, dtype=tf.float32))
      binary_tensor = tf.cast(tf.floor(random_tensor), inputs.dtype)
      ret = tf.div(inputs, keep_prob) * binary_tensor
      ret.set_shape(inputs.get_shape())
      return ret

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(
        flops=inputs.num_elements() * 5, out_shapes=(inputs,))


class DeterministicWeightedSumLayer(base_layer.BaseLayer):
  """WeightedSumLayer with deterministic dropout."""

  @classmethod
  def Params(cls):
    """Params for this MergerLayer class."""
    p = super(DeterministicWeightedSumLayer, cls).Params()
    p.Define('num_sources', 0, 'Number of input sources to combine.')
    p.Define('weighted_merger_dropout_prob', 0.0,
             'Applies dropout to the weights.')
    p.Define(
        'weighted_merger_softmax', True, 'If set, applies a softmax '
        'layer on top of the weights for normalization.')
    p.Define('global_weight_scale', 1.0, 'A global scale put on weights.')
    p.Define('minimal_prob', 0.0, 'The minimal weight for each component.')
    p.Define('dropout_tpl', DeterministicDropoutLayer.Params(), 'Dropout layer')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(DeterministicWeightedSumLayer, self).__init__(params)
    p = self.params
    if not p.name:
      raise ValueError('Layer must have a specified name!')

    assert p.num_sources > 0, ('Must specify num_sources > 0.')
    params_init = py_utils.WeightInit.Constant(0.0)
    # Weights to be learned.
    pw = py_utils.WeightParams(
        shape=[p.num_sources],
        init=params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('sum_weight', pw)
    p.dropout_tpl.name = 'dropout'
    self.CreateChild('weighted_merger_dropout', p.dropout_tpl)

  def FProp(self, theta, inputs):
    """Combines the list of input tensors into a single tensor.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A list of tensors of shape [time, batch, hidden_dim]

    Returns:
      A tensor of the same shape with input tensors.
    """
    p = self.params
    n_sources = len(inputs)

    if n_sources == 1:
      return inputs[0]

    # Weighted sum of all sources, all dims must match.
    # For weighted_sum, assume input is a list of rank 3 tensors
    inputs = tf.stack(inputs)
    inputs = py_utils.HasRank(inputs, 4)

    # The constant factor is just meant to support the non-normalized scenario.
    # If softmax is applied, this factor will cancel out.
    w = theta.sum_weight * p.global_weight_scale + (1 / p.num_sources)
    w = tf.reshape(w, [p.num_sources])
    w = self.weighted_merger_dropout.FProp(theta.weighted_merger_dropout, w)
    if p.weighted_merger_softmax:
      residual_weights = p.minimal_prob * p.num_sources
      assert residual_weights >= 0.0
      assert residual_weights < 1.0
      w = tf.nn.softmax(w, axis=0) * (1.0 - residual_weights) + p.minimal_prob

    w = tf.reshape(w, [p.num_sources, 1, 1, 1])
    output = tf.reduce_sum(inputs * w, axis=0)

    return output
