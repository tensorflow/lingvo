# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""GShard Builder. To be used with xla_sharding + SPMD."""

from lingvo import compat as tf
from lingvo.core import base_model
from lingvo.core import builder
from lingvo.core import builder_layers
from lingvo.core import entmax
from lingvo.core import gshard_layers
from lingvo.core import gshard_utils
from lingvo.core import layers
from lingvo.core import py_utils
import numpy as np
import six


def _ToInt32(t):
  return tf.cast(t, tf.int32)


class MoEBuilder(builder.Base):
  """Mixture-of-Experts, Dense and DenseSparse Builder.

  To be used with xla_sharding + SPMD.

  MoEBuilder can be used to construct MoE and non-MoE Transformer models.

  Such models are typically defined by introducing encoder and decoder layer
  stacks, for example::

      enc = builder.EncoderLayerStack('encoder', [
          builder.SelfAttention('self_attention'),
          builder.MoE('moe'),
          builder.SelfAttention('self_attention'),
          builder.DenseReluDense('dense_relu_dense'), ], 3)

      dec = builder.DecoderLayerStack('decoder', [
          builder.DecSelfAttention('dec_self_attention'),
          builder.DecEncAttention('dec_enc_attention'),
          builder.MoE('moe', decoder=True),
          builder.DecSelfAttention('dec_self_attention'),
          builder.DecEncAttention('dec_enc_attention'),
          builder.DenseReluDense('dense_relu_dense', decoder=True), ], 3)

  Each layer (e.g. builder.SelfAttention) is ultimately wrapped with
  builder.EncoderLayer or builder.DecoderLayer. These wrappers introduce
  Transformer residual connections and layer norm as well.

  Naturally supports input packing, where multiple segments are packed in a
  single vec row (e.g. packing 2 segments in a single row)::

      vec      [  4,   3,  24]
      segment_id  [  1,   1,   2] (0 would indicate padding)
      segment_pos [  0,   1,   0] (0 for first token in the segment etc)

  by adding Attention bias to Attention logits before applying tf.nn.softmax,
  bias calculated as follows::

      SelfAttention
        segment_id  [  1,   1,   2]
      =>
        bias       [[  0,   0,  -X],
                    [  0,   0,  -X],
                    [ -X,  -X,   0]], where X is a large number.

  Segments can only attend to itself::

      DecSelfAttention
        segment_id  [  1,   1,   2]
        segment_pos [  0,   1,   0]
      =>
        bias       [[  0,  -X,  -X],
                    [  0,   0,  -X],
                    [ -X,  -X,   0]], where X is a large number.

  Segments can only attend to itself, and pos 'i' can only attend to <= 'i'
  subsegment::

      DecEncAttention
        segment_id  [  1,   1,   2]
        encoder_segment_id  [  1,   2]
      =>
        bias       [[  0,  -X],
                    [  0,  -X],
                    [ -X,   0]], where X is a large number.

  Encoder layers must share same Graph input_endpoints, output_endpoints,
  Builder.{MoE,DenseReluDense,SelfAttention},
  so do Decoder layers (with decoder=true set where appropriate),
  Builder.{MoE,DenseReluDense,DecSelfAttention,DecEncAttention},
  so we can universally wrap them with Builder.{Encoder,Decoder}Layer and
  further stack with Builder.{Encoder,Decoder}LayerStack. To be moved from
  XlaShardingBuilder.

  TODO(lepikhin): enable MoE-Attention.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_devices', 1,
             'The number of cores to split weights and computation over.')
    p.Define('num_groups', None,
             'The number of groups. Set to None to use num_devices.')
    p.Define('layer_norm_epsilon', 1e-6,
             'Epsilon for layer norm numerical stability.')
    p.Define('dtype', tf.float32, 'Datatype to use.')

    p.Define(
        'model_dim', 1024, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')

    p.Define(
        'dropout_rate', 0.0,
        'Universal dropout rate that applies to inputs, residual, '
        'and other Transformer layers.')

    p.Define(
        'noise_shape_broadcast_dims', None,
        'A list of dimension where the noise shape is broadcasted. For '
        'example, noise_shape = [n, h, w, 1] when '
        'noise_shape_broadcast_dims=[-1] ')

    # attention params
    p.Define('attention_num_heads', 1, 'Attention number of heads.')
    p.Define(
        'attention_num_memory_heads', None,
        'Attention number of memory heads. We only support '
        'attention_num_memory_heads of 1 or None (default).')
    p.Define('attention_key_value_dim', None,
             'Shared dimensionality for Attention keys, values.')
    p.Define('attention_dropout_prob', 0.0, 'Attention dropout probability.')
    p.Define('moe_dropout_rate', 0.0, 'MoE dropout probability.')
    p.Define(
        'attention_combine_dims', False, 'Attention optimization. '
        'The heads and key/value dimensions are combined in the variables '
        'and the computation.')
    p.Define('attention_combine_qkv', True, 'Attention optimization. '
             'Combine qkv matmul.')

    p.Define('ff_dim', None, 'DenseReluDense hidden dim.')

    # MoE params
    p.Define('e_dim', None, 'E dimension. Number of experts.')
    p.Define('c_dim', None, 'C dimension. Per-expert capacity.')
    p.Define('moe_hidden_dim', None, 'Mixture-of-Experts hidden dim.')
    p.Define('moe_activation', 'RELU', 'MoE activation function.')

    p.Define('second_expert_policy', 'all',
             'Mixture-of-Experts dispatch policy.')
    p.Define('second_expert_threshold', 0.,
             'Mixture-of-Experts second-best gate threshold.')
    p.Define('legacy_mtf_behavior', True,
             'Mixture-of-Experts legacy mtf behavior. No renormalization.')
    p.Define('label_smoothing', 0.1, 'Label smoothing.')
    p.Define('capacity_factor', None, 'Capacity factor. Overrides c_dim.')

    # Used in DecSelfAttentionRelativeBias:
    p.Define('relative_attention_type', None,
             'Attention type. None is default. Alternative is "bias".')
    p.Define('relative_attention_num_buckets', 32,
             'Relative attention num buckets.')
    p.Define('relative_attention_max_distance', 128,
             'Max relative distance (outer bucket boundary).')
    p.Define(
        'relative_attention_use_universal_1d_position', False,
        'Relative attention could rely on fake 1d position tensor, '
        'since only the relative difference matters and extra large '
        'negative logit bias term is added for attention across segments '
        'anyway. Set to True to enable the hack.')
    p.Define(
        'inflate_universal_relative_bias_to_match_batch_dimension', False,
        'If true, inflate the relative_bias tensor to match the batch '
        'dimension of the BLM logits. When BLM is partitioned along the batch '
        'dimension, this avoids the all-reduce for the relative_bias '
        'activation gradients, but performs the all-reduce for relative_bias '
        'weights instead. This may cause computation overhead when batch_size'
        'is much larger than num_batch_partitions, so it should only be used'
        'when batch_size is not too large compared to num_batch_partitions. '
        'This flag is ignored if relative_attention_use_universal_1d_position '
        'is set to False. Please see b/173612674#comment2 for more details.')

    p.Define('attention_extra_logit', None,
             'Extra logit for attention softmax.')
    p.Define(
        'attention_logits_dtype', None,
        'Using float32 for attention logits with fprop_dtype=bfloat16 '
        'generally makes training giant models more stable.')
    p.Define(
        'mask_dtype', None,
        'Using bfloat16 for fprop_dtype could be problematic for '
        'mask tensors, mask_dtype is a special dtype for such tensors.')
    p.Define(
        'gating_logits_dtype', None,
        'Using bfloat16 for fprop_dtype could be problematic for '
        'gating logits, gating_logits_dtype is a special dtype for such '
        'tensors.')

    p.Define(
        'conv_vars_reshape', False, 'Boolean, whether or not to '
        'change the shape of conv variables. For checkpoint backward '
        'compatibility only, deprecated soon. If True, the variable shape '
        'of _LNConv will be based on model_dim_reshape_segment')
    p.Define(
        'use_fused_depthwise_conv_autoregressive', False,
        'If True, use CausalDepthwiseConv for '
        'DepthwiseConvAutoregressive.')
    p.Define('ln_no_scale', False,
             'Override Builder._LN with Builder._LNNoScale.')
    p.Define('model_dim_reshape_segments', None,
             'Size of N when reshaping model dimension M to Nm')
    p.Define('use_xla_dynamic_update_slice', True, 'internal optimization')
    return p

  @classmethod
  def SetFPropDtype(cls, p, fprop_dtype):
    p.fprop_dtype = fprop_dtype
    if fprop_dtype == tf.bfloat16:
      p.attention_logits_dtype = tf.float32
    return p

  @property
  def _device_mesh(self):
    return None

  @property
  def _model_dim_reshape_segments(self):
    if self.params.model_dim_reshape_segments is None:
      return None
    elif isinstance(self.params.model_dim_reshape_segments, list):
      return self.params.model_dim_reshape_segments
    return [self.params.model_dim_reshape_segments]

  def _AdjustMSplit(self, split, m_dim):
    """Adjusts split annotation according to model_dim_reshape_segments."""
    if split is None:
      return None
    if self._model_dim_reshape_segments is None:
      return split
    new_split = list(split)
    for _ in self._model_dim_reshape_segments:
      new_split.insert(m_dim + 1, -1)
    return new_split

  def _AdjustMSplitByName(self, p_split_name):
    split = getattr(self.params, p_split_name, None)
    m_axis = p_split_name.find('m')
    if m_axis >= 0:
      split = self._AdjustMSplit(split, m_axis)
    return split

  def _Dropout(self, name, keep_prob, noise_shape_broadcast_dims=None):
    return super()._Dropout(
        name, keep_prob, noise_shape_broadcast_dims or
        self.params.noise_shape_broadcast_dims)

  def _OneHotEncode(self, name, dim):
    fprop_dtype = py_utils.FPropDtype(self.params)
    return self._Fn(name, fn=lambda x: tf.one_hot(x, dim, dtype=fprop_dtype))

  def _Var(self, name, weights, shared_var_collection_suffix=None):
    return gshard_layers.VarLayer.Params().Set(
        name=name,
        dtype=self.params.dtype,
        fprop_dtype=self.params.fprop_dtype,
        weights=weights,
        shared_var_collection_suffix=shared_var_collection_suffix)

  def _ShardedVar(self, name, weights, device_mesh):
    """Creates a layer of variables potentially sharded in a device mesh.

    Args:
      name: name of the layer.
      weights: list of (name, gshard_layers.ShardedWeightParams).
      device_mesh: device mesh used in mesh_split. If None, the variables will
        not be sharded

    Returns:
      A layer of variables sharded according to device_mesh and the weights'
      sharding specification in ShardedWeightParams.
    """

    if device_mesh is None:
      return self._Var(name=name, weights=weights)
    else:
      return gshard_layers.ShardedVarLayer.Params().Set(
          name=name,
          dtype=self.params.dtype,
          fprop_dtype=self.params.fprop_dtype,
          weights=weights,
          device_mesh=device_mesh)

  def _ShardedVarOn1DDeviceArray(self, name, weights):
    """Variables sharded along dimension 0.

    Args:
      name: name of the layer.
      weights: list of (name, py_utils.WeightParams).

    Returns:
      A layer of variables sharded on dimension 0 across all devices.
    """
    sharded_weights = []
    for k, v in weights:
      assert v.shape is not None and v.shape
      dims_mapping = [0] + [-1] * (len(v.shape) - 1)
      sharded_weights.append((k,
                              gshard_layers.ShardedWeightParams(
                                  shape=v.shape,
                                  init=v.init,
                                  dtype=v.dtype,
                                  collections=v.collections,
                                  tensor_split_dims_mapping=dims_mapping)))
    return self._ShardedVar(
        name=name,
        weights=sharded_weights,
        device_mesh=np.arange(self.params.num_devices))

  def _EmbeddingWeight(self,
                       name,
                       vocab_dim,
                       device_mesh=None,
                       w_mesh_split=None):
    return self._ShardedVar(
        name=name,
        weights=[('embedding',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Gaussian(),
                      dtype=self.params.dtype,
                      shape=[vocab_dim, self.params.model_dim],
                      tensor_split_dims_mapping=w_mesh_split))],
        device_mesh=device_mesh)

  def SharedEmbSoftmax(self,
                       name,
                       vocab_size,
                       max_len,
                       logits_abs_max=None,
                       z_loss_coef=1e-4,
                       use_tgt_labels_size_as_loss_denominator=True):
    p = self.params
    return gshard_layers.SharedEmbeddingSoftmaxLayer.Params().Set(
        name=name,
        vocab_size=vocab_size,
        max_len=max_len,
        logits_abs_max=logits_abs_max,
        z_loss_coef=z_loss_coef,
        embedding_dim=p.model_dim,
        num_devices=p.num_devices,
        label_smoothing=p.label_smoothing,
        use_tgt_labels_size_as_loss_denominator=use_tgt_labels_size_as_loss_denominator
    )

  def Embedding(self, name, vocab_dim):
    return self._Graph(
        name, ['ids'], ['outputs'],
        ('->emb', self._EmbeddingWeight('w', vocab_dim)),
        ('ids->ids_split', self.Split('ids_split')),
        ('ids_split->one_hot_ids', self._OneHotEncode('one_hot_ids',
                                                      vocab_dim)),
        ('one_hot_ids->one_hot_ids_split', self.Split('one_hot_ids_split')),
        ('emb,one_hot_ids_split->outputs',
         self._Fn('einsum', fn=lambda w, x: tf.einsum('VH,BLV->BLH', w, x))))

  def SoftmaxWeight(self, name, vocab_dim):
    return self._Var(
        name=name,
        weights=[('softmax_weight',
                  py_utils.WeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.model_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.model_dim, vocab_dim]))])

  def Mask(self):

    def _apply_padding(x, segment_id):  # pylint: disable=invalid-name
      mask = tf.cast(tf.not_equal(segment_id, 0), x.dtype)
      for _ in range(len(x.shape) - len(mask.shape)):
        mask = tf.expand_dims(mask, -1)
      return x * mask

    return self._Fn('mask', fn=_apply_padding, fn_out=lambda x, y: x)

  def EncoderLayer(self, name, layer, residual_weight=1.0):
    """Returns params for lambda x: x + residual_weight * DropOut(layer(LN(x)))."""
    layer_input_keys = self._EncoderLayerInMapKeys
    layer_inputs = 'x,' + ','.join(['i.' + key for key in layer_input_keys[1:]])
    return self._Graph(
        name,
        ['i'],
        ['o'],
        ('i.vec,i.segment_id->input_masked', self.Mask()),
        ('input_masked->x', self._LN('ln')),
        (layer_inputs + '->y,o.aux_loss', layer),
        ('y->y_dropout', self._Dropout('y_dropout',
                                       1 - self.params.dropout_rate)),
        ('input_masked,y_dropout->o.vec',
         self._Add('add', residual_weight=residual_weight)),
    )

  @property
  def _EncoderLayerInMapKeys(self):
    return ['vec', 'segment_id', 'segment_pos']

  # We avoid Builder._Seq and Builder._Rep to improve theta / checkpoint
  # readability and reduce layer nesting.
  def EncoderLayerStack(self,
                        name,
                        sub_layers,
                        num=1,
                        use_repeat_layer=False,
                        spmd_pipeline_stages=1,
                        spmd_pipeline_microbatches=None):
    """Clean EncoderLayerStack with minimal layer nesting.

    E.g::

      encoder/
        layer_000/
          ln/w/
            scale
          self_attention/w/
            wq
            wk
            wv
            wo
        layer_001/
          ...

    will be constructed with::

      builder.EncoderLayerStack('encoder', [
          builder.SelfAttention('self_attention'),
          ...], ...)

    Args:
      name: Name of this layer
      sub_layers: Sublayers of the encoder layer.
      num: Number of encoder layers.
      use_repeat_layer: bool, whether to wrap num layers into a RepeatLayer.
      spmd_pipeline_stages: If > 1, use SPMD-shardable pipelining with this many
      pipeline stages.
      spmd_pipeline_microbatches: The number of microbatches when SPMD-shardable
      pipelining is used.

    Returns:
      The layer params.
    """
    return self._LayerStack(name, sub_layers, num, use_repeat_layer,
                            spmd_pipeline_stages, spmd_pipeline_microbatches,
                            self._EncoderLayerInMapKeys,
                            lambda n, p: self.EncoderLayer(name=n, layer=p))

  def DecoderLayer(self,
                   name,
                   layer,
                   conv_kernel_size=None,
                   norm_type='ln',
                   norm_policy='pre'):
    """A decoder layer.

    Args:
      name: Layer name.
      layer: Layer logic added to the compute graph.
      conv_kernel_size: The width of the kernel when convolution is added to
        layer norm.
      norm_type: String that describes the normalization type. Currently only
        supports 'ln'.
      norm_policy: String that describes the policy for applying normalzation.
        Currently only supports 'pre' for pre-transformation normalzation.

    Returns:
      Compute graph of decoder layer.
    """
    # BEGIN GOOGLE-INTERNAL
    # TODO(davidso): Expand the options for norm_type and norm_policy when
    #                making Primer public.
    # END GOOGLE-INTERNAL
    if conv_kernel_size is not None:
      if norm_type != 'ln':
        raise ValueError('Only ln supports conv. %s does not support conv.' %
                         norm_type)
      norm_layer = self._LNConv('ln', conv_kernel_size)
    elif norm_type == 'ln':
      norm_layer = self._LN('ln')
    elif norm_type == 'true_ln':
      norm_layer = self._TrueLN('true_ln')
    # BEGIN GOOGLE-INTERNAL
    # TODO(davidso): Make this public once the Primer paper is released.
    elif norm_type == 'pn':
      norm_layer = self._PN('pn')
    # END GOOGLE-INTERNAL
    else:
      raise ValueError('Norm type %s not supported.' % norm_type)
    layer_input_keys = self._DecoderLayerInMapKeys
    layer_inputs = 'x,' + ','.join(['i.' + key for key in layer_input_keys[1:]])
    if norm_policy == 'pre':
      return self._Graph(
          name,
          ['i'],
          ['o'],
          ('i.vec,i.segment_id->input_masked', self.Mask()),
          ('input_masked->x', norm_layer),
          (layer_inputs + '->y,o.aux_loss', layer),
          ('y->y_dropout',
           self._Dropout('y_dropout', 1 - self.params.dropout_rate)),
          ('input_masked,y_dropout->o.vec', self._Add('add')),
      )
    # BEGIN GOOGLE-INTERNAL
    # TODO(davidso): Make this public when Primer paper is released.
    if norm_policy == 'primer':
      indx = int(name[-3:])
      if indx % 2 == 0:
        return self._Graph(
            name,
            ['i'],
            ['o'],
            ('i.vec,i.segment_id->input_masked', self.Mask()),
            ('input_masked->x', norm_layer),
            (layer_inputs + '->y,o.aux_loss', layer),
            ('y->y_dropout',
             self._Dropout('y_dropout', 1 - self.params.dropout_rate)),
            ('input_masked,y_dropout->o.vec', self._Add('add')),
        )
      else:
        return self._Graph(
            name,
            ['i'],
            ['o'],
            ('i.vec,i.segment_id->x', self.Mask()),
            (layer_inputs + '->y,o.aux_loss', layer),
            ('y->y_norm', norm_layer),
            ('y_norm->y_dropout',
             self._Dropout('y_dropout', 1 - self.params.dropout_rate)),
            ('x,y_dropout->o.vec', self._Add('add')),
        )
    # END GOOGLE-INTERNAL
    raise ValueError('Unsupported norm policy: %s' % norm_policy)

  @property
  def _DecoderLayerInMapKeys(self):
    return [
        'vec', 'segment_id', 'segment_pos', 'encoder_output',
        'encoder_segment_id', 'encoder_segment_pos'
    ]

  def DecoderLayerStack(self,
                        name,
                        sub_layers,
                        num=1,
                        conv_kernel_size=None,
                        norm_type='ln',
                        norm_policy='pre',
                        use_repeat_layer=False,
                        spmd_pipeline_stages=1,
                        spmd_pipeline_microbatches=None):
    """Clean DecoderLayerStack, similar to EncoderLayerStack."""

    def _DecoderLayer(n, p):
      return self.DecoderLayer(
          n,
          p,
          conv_kernel_size=conv_kernel_size,
          norm_type=norm_type,
          norm_policy=norm_policy)

    return self._LayerStack(name, sub_layers, num, use_repeat_layer,
                            spmd_pipeline_stages, spmd_pipeline_microbatches,
                            self._DecoderLayerInMapKeys, _DecoderLayer)

  def Repeat(self, name, body, repeat=1, per_layer_vars=True):
    """Wrapper to call builder_layers.RepeatLayer."""
    return builder_layers.RepeatLayer.Params().Set(
        name=name,
        body=body,
        repeat=repeat,
        per_layer_vars=per_layer_vars,
        unroll='eval_only')

  def ShardablePipeline(self, name, body, stages):
    """Wrapper to call gshard_layers.LayerwiseShardablePipelinedLayer."""
    return gshard_layers.LayerwiseShardablePipelinedLayer.Params().Set(
        name=name, num_stages=stages, single_stage_body=body)

  def _LayerStack(self, name, sub_layers, num, use_repeat_layer,
                  spmd_pipeline_stages, spmd_pipeline_microbatches, imap_keys,
                  layer_fn):
    # TODO(yuanzx): Consider refactor this into a layer.
    assert 'segment_id' in imap_keys
    if use_repeat_layer:
      assert self.params.deterministic_dropout
    stack = []
    for key in imap_keys:
      stack.append(
          ('i.' + key + '->' + key + '_split', self.Split(key + '_split')))

    stack += [
        (imap_keys[0] + '_split->x_000',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('i.aux_loss->loss_000', self._Identity('loss_000')),
    ]

    def _SubLayersBlock(l, idx):
      map_inputs = 'x_%03d,' + ','.join(
          [key + '_split' for key in imap_keys[1:]])
      return [((map_inputs + '->imap_%03d') % (idx, idx),
               self._CreateNestedMap(name='imap_%03d' % idx, keys=imap_keys)),
              ('imap_%03d->omap_%03d' % (idx, idx),
               layer_fn('layer_%03d' % idx, l)),
              ('omap_%03d.vec->x_%03d' % (idx, idx + 1),
               self._Identity('vec_%03d' % idx)),
              ('loss_%03d,omap_%03d.aux_loss->loss_%03d' % (idx, idx, idx + 1),
               self._Add('loss_%03d' % (idx + 1)))]

    i = 0
    assert num % spmd_pipeline_stages == 0
    layers_per_stage = num // spmd_pipeline_stages
    main_stack = []
    if use_repeat_layer:
      blocks = []
      for l in sub_layers:
        blocks += _SubLayersBlock(l, i)
        i += 1
      body_inputs = 'x_000,loss_000,' + ','.join(
          [key + '_split' for key in imap_keys[1:]])
      body_outputs = 'x_%03d,loss_%03d,' % (i, i) + ','.join(
          [key + '_split' for key in imap_keys[1:]])
      body_p = self._Graph('blocks_body', body_inputs.split(','),
                           body_outputs.split(','), *blocks)
      repeat_p = self.Repeat(
          name='blocks', body=body_p, repeat=layers_per_stage)
      main_stack = [
          (body_inputs + '->' + body_outputs.replace('_split', '_split_out'),
           repeat_p)
      ]
    else:
      for _ in range(layers_per_stage):
        for l in sub_layers:
          # x_i, loss_i => x_{i+1}, loss_{i+1}
          main_stack += _SubLayersBlock(l, i)
          i += 1

    if spmd_pipeline_stages > 1:
      # TODO(yuanzx): Consider refactor this into a layer.

      # Reshape each input into microbatches.
      def _ToMicroBatches(key):

        def _ReshapeToMicroBatches(x):
          assert x.shape[0] % spmd_pipeline_microbatches == 0
          # First reshape to [microbatch_size, spmd_pipeline_microbatches, ..]
          # then transpose to [spmd_pipeline_microbatches, microbatch_size, ...]
          # because we want each microbatch to be sharded the same way as the
          # original batch dimension.
          new_shape = [
              x.shape[0] // spmd_pipeline_microbatches,
              spmd_pipeline_microbatches
          ]
          new_shape += x.shape[1:]
          perm = list(range(len(new_shape)))
          perm[0] = 1
          perm[1] = 0
          return tf.transpose(tf.reshape(x, new_shape), perm)

        return (key + '->' + key + '_m',
                self._Fn(key + '_microbatched', _ReshapeToMicroBatches))

      stack += [
          _ToMicroBatches(k)
          for k in ['x_000'] + [key + '_split' for key in imap_keys[1:]]
      ]

      # Pipelining requires each input/output to have a num_microbatches
      # dimension, but loss is a scalar. We pad it to the shape
      # [spmd_pipeline_microbatches] to compute per-microbatch loss, then sum
      # them together after the pipeline.
      def _PadLoss(x):
        return tf.pad(tf.reshape(x, [1]), [[0, spmd_pipeline_microbatches - 1]])

      stack.append(
          ('loss_000->loss_000_m', self._Fn('loss_padded_microbatched',
                                            _PadLoss)))
      body_inputs = 'x_000,loss_000,' + ','.join(
          [key + '_split' for key in imap_keys[1:]])
      body_outputs = 'x_%03d,loss_%03d,' % (i, i) + ','.join(
          [key + '_split' for key in imap_keys[1:]])
      body_p = self._Graph('pipeline_body', body_inputs.split(','),
                           body_outputs.split(','), *main_stack)
      pipeline_p = self.ShardablePipeline(
          name='pipeline', body=body_p, stages=spmd_pipeline_stages)
      pipeline_inputs = 'x_000_m,loss_000_m,' + ','.join(
          [key + '_split_m' for key in imap_keys[1:]])
      pipeline_outputs = 'x_%03d_m,loss_%03d_m,' % (i, i) + ','.join(
          [key + '_split_out_m' for key in imap_keys[1:]])
      main_stack = [(pipeline_inputs + '->' + pipeline_outputs, pipeline_p)]

      # Reshape outputs to the original shape without microbatches.
      def _ToBatches(key):

        def _ReshapeToBatches(x):
          perm = list(range(len(x.shape)))
          perm[0] = 1
          perm[1] = 0
          x = tf.transpose(x, perm)
          return tf.reshape(x, [x.shape[0] * x.shape[1]] + x.shape[2:])

        return (key + '_m->' + key,
                self._Fn(key + '_unmicrobatched', _ReshapeToBatches))

      main_stack += [
          _ToBatches(k) for k in (['x_%03d' % i] +
                                  [key + '_split_out' for key in imap_keys[1:]])
      ]
      # Sum the per-microbatch losses.
      main_stack.append(('loss_%03d_m->loss_%03d' % (i, i),
                         self._Fn('loss_combined', tf.reduce_sum)))

    stack += main_stack
    stack += [
        (('loss_%03d->o.aux_loss' % i), self._Identity('output_loss')),
        (('x_%03d->y_norm' % i), self._LN('final_layer_norm')),
        ('y_norm->y_dropout',
         self._Dropout('outputs_dropout', 1 - self.params.dropout_rate)),
        ('y_dropout,segment_id_split->o.vec', self.Mask()),
    ]
    return self._Graph(name, ['i'], ['o'], *stack)

  def _DenseReluDenseWeights(self,
                             name,
                             device_mesh=None,
                             wi_mesh_split=None,
                             wo_mesh_split=None):
    return self._ShardedVar(
        name=name,
        weights=[('wi',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.model_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.model_dim, self.params.ff_dim],
                      tensor_split_dims_mapping=wi_mesh_split)),
                 ('wo',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.ff_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.ff_dim, self.params.model_dim],
                      tensor_split_dims_mapping=wo_mesh_split))],
        device_mesh=device_mesh)

  def DenseReluDense(self, name, decoder=False, activation='relu'):
    if decoder:
      input_endpoints = self._DecoderLayerInMapKeys
    else:
      input_endpoints = self._EncoderLayerInMapKeys
    # Note that dropout is used here, but not in the MoE layer by default.

    if activation == 'relu':
      activation_fn = tf.nn.relu
    elif activation == 'gelu':
      activation_fn = lambda x: tf.nn.gelu(x, approximate=True)
    # BEGIN GOOGLE-INTERNAL
    # TODO(davidso): Make this public when Primer paper is released.
    elif activation == 'sqr_relu':
      activation_fn = lambda x: tf.math.square(tf.nn.relu(x))
    # END GOOGLE-INTERNAL
    else:
      raise ValueError('Activation %s not supported.' % activation)

    return self._Graph(
        name,
        input_endpoints,
        ['outputs', 'aux_loss'],
        ('->wi,wo', self._DenseReluDenseWeights('w')),
        ('wi,vec->h',
         self._Fn('wi', fn=lambda wi, vec: tf.einsum('MH,BLM->BLH', wi, vec))),
        ('h->h_%s' % activation, self._Fn(activation, activation_fn)),
        ('h_%s->h_dropout' % activation,
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('wo,h_dropout->outputs_pre_split',
         self._Fn(
             'wo',
             fn=lambda wo, h_dropout: tf.einsum('HM,BLH->BLM', wo, h_dropout))),
        ('outputs_pre_split->outputs', self.Split('outputs_split')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
    )

  def _DenseReluDenseWeightsGatedGELU(self,
                                      name,
                                      device_mesh=None,
                                      wi_mesh_split=None,
                                      wo_mesh_split=None):
    # Gated GELU.  There are two separate linear transformations applied in
    # parallel to the inputs.  You take the gelu of one of them and then
    # multiply the two componentwise.
    return self._ShardedVar(
        name=name,
        weights=[('wi_0',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.model_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.model_dim, self.params.ff_dim],
                      tensor_split_dims_mapping=wi_mesh_split)),
                 ('wi_1',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.model_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.model_dim, self.params.ff_dim],
                      tensor_split_dims_mapping=wi_mesh_split)),
                 ('wo',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.ff_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.ff_dim, self.params.model_dim],
                      tensor_split_dims_mapping=wo_mesh_split))],
        device_mesh=device_mesh)

  def DenseReluDenseGated(self, name, activation_fn, decoder=False):
    if decoder:
      input_endpoints = self._DecoderLayerInMapKeys
    else:
      input_endpoints = self._EncoderLayerInMapKeys

    def _Impl(wi_0, wi_1, inputs):
      return tf.math.multiply(
          activation_fn(tf.einsum('MH,BLM->BLH', wi_0, inputs)),
          # linear / pass-through
          tf.einsum('MH,BLM->BLH', wi_1, inputs))

    return self._Graph(
        name,
        input_endpoints,
        ['outputs', 'aux_loss'],
        ('->wi_0,wi_1,wo', self._DenseReluDenseWeightsGatedGELU('w')),
        ('wi_0,wi_1,vec->h', self._Fn('wi', fn=_Impl)),
        ('h->h_dropout',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('wo,h_dropout->outputs_pre_split',
         self._Fn(
             'wo',
             fn=lambda wo, h_dropout: tf.einsum('HM,BLH->BLM', wo, h_dropout))),
        ('outputs_pre_split->outputs', self.Split('outputs_split')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
    )

  def DenseReluDenseGatedGELU(self, name, decoder=False):
    return self.DenseReluDenseGated(
        name, lambda x: tf.nn.gelu(x, approximate=True), decoder=decoder)

  def DenseReluDenseGatedSILU(self, name, decoder=False):
    return self.DenseReluDenseGated(name, tf.nn.silu, decoder=decoder)

  def MoE(self, name, decoder=False):
    """Returns layer params to compute (outputs, scalar_aux_loss)."""
    if decoder:
      input_endpoints = self._DecoderLayerInMapKeys
    else:
      input_endpoints = self._EncoderLayerInMapKeys

    return self._Graph(
        name, input_endpoints, ['outputs', 'aux_loss'],
        ('vec->input_split', self.Split('input_split')),
        ('segment_id->segment_id_split', self.Split('segment_id_split')),
        ('->wi,wo', self._ShardedFeedForwardNetworksWeights(name)),
        ('input_split,segment_id_split,wi,wo->outputs_pre_split,aux_loss',
         self._ShardedMoEPositionWiseFeedForwardNetworks('ffw')),
        ('outputs_pre_split->outputs', self.Split('outputs_split')))

  # Multi-headed attention Tensors:
  # q: BLHD [batch, length,        heads, key_value]
  # k: BMHD [batch, memory_length, heads, key_value]
  # v: BMHD [batch, memory_length, heads, key_value]
  #
  # logits:  BLHM
  # bias:    BLM
  #
  # weights: BLHM [batch, length, heads, memory_length]
  #
  # outputs: BLHD [batch, length, heads, key_value]
  def Attention(self, name):
    """Attention with multiple attention heads.

    Keys, values share same dimensionality
    params.self.params.attention_key_value_dim.

    Args:
      name: name of the layer

    Returns:
      The Attention layer params.
    """
    p = self.params

    def _AddBiasF32(logits, bias):
      # logits: BLHM [batch, length, heads, memory_length]
      # bias: BLHM [batch, length, heads, memory_length]
      #       (in case of attention with relative bias) OR
      #
      #       BLM  [batch, length, memory_length]
      #       (default masking bias with very negative logits).
      bias = tf.cast(bias, logits.dtype)
      if bias.shape.ndims == 3:
        # Expanding the 'heads' dimension
        retval = logits + tf.expand_dims(bias, 2)
      else:
        assert bias.shape.ndims == 4
        retval = logits + bias
      return retval

    def _ReduceLogsumexp(x):
      max_logit = tf.math.reduce_max(
          tf.stop_gradient(x), axis=-1, keepdims=True)

      extra_logit = p.attention_extra_logit
      if extra_logit is not None:
        extra_logit = tf.convert_to_tensor(extra_logit, max_logit.dtype)
        max_logit = tf.math.maximum(max_logit, extra_logit)
      x -= max_logit
      exp_x = tf.math.exp(x)
      sum_exp_x = tf.math.reduce_sum(exp_x, axis=-1, keepdims=True)
      if extra_logit is not None:
        sum_exp_x += tf.math.exp(extra_logit - max_logit)
      return tf.math.log(sum_exp_x) + max_logit

    def _LogSoftmax(x):
      return x - _ReduceLogsumexp(x)

    def _LogitsFnF32(q, k):
      # logits.dtype == tf.float32 leads to better training stability
      if p.attention_logits_dtype is not None:
        q = tf.cast(q, p.attention_logits_dtype)
        k = tf.cast(k, p.attention_logits_dtype)
      return tf.einsum('BLHD,BMHD->BLHM', q, k)

    def _SoftmaxF32(x):
      # expecting x.dtype == tf.float32
      #
      # TODO(lepikhin): consider
      # if p.attention_extra_logit is None:
      #   return tf.nn.softmax(x)
      softmax = tf.math.exp(_LogSoftmax(x))
      softmax = tf.cast(softmax, py_utils.FPropDtype(self.params))
      return softmax

    return self._Graph(
        name,
        ['_q', '_k', '_v', 'bias'],
        ['outputs'],
        ('_q->q', self.Split('_q')),
        ('_k->k', self.Split('_k')),
        ('_v->v', self.Split('_v')),
        ('q,k->l', self._Fn('logits', fn=_LogitsFnF32)),
        ('l,bias->logits', self._Fn('bias', fn=_AddBiasF32)),
        ('logits->w', self._Fn('weights', _SoftmaxF32)),
        ('w->weights',
         self._Dropout('dropout', 1 - self.params.attention_dropout_prob)),
        ('weights,v->outputs',
         self._Fn(
             'outputs',
             fn=lambda weights, v: tf.einsum('BLHM,BMHD->BLHD', weights, v))),
    )

  def _ComputeAttenOutputs(self, o, wo):
    p = self.params
    if p.attention_combine_dims:
      wo = tf.reshape(
          wo, [p.attention_num_heads, p.attention_key_value_dim, p.model_dim])
    return tf.einsum('HDM,BLHD->BLM', wo, o)

  def _EncNotVisible(self, a, b):
    """a, b are encoder_segment_id, decoder_segment_id Tensors."""
    a, b = tf.expand_dims(a, -1), tf.expand_dims(b, -2)
    # NotVisible == (a != b) || !((a!=0) || (b != 0))
    #            == (a != b) || ((a==0) && (b == 0))
    # ignore segment_id == 0.
    ret = tf.equal(a, 0)
    ret = tf.logical_and(ret, tf.equal(b, 0))
    ret = tf.logical_or(ret, tf.not_equal(a, b))
    return tf.cast(ret, py_utils.FPropDtype(self.params))

  def SelfAttention(self,
                    name,
                    device_mesh=None,
                    w_qkv_mhd_mesh_split=None,
                    wo_hdm_mesh_split=None):
    """TransformerEncoder SelfAttention."""
    # pyformat: disable
    return self._Graph(
        name, self._EncoderLayerInMapKeys, [
            'outputs',
            'aux_loss'
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights(
            'w', device_mesh, w_qkv_mhd_mesh_split, wo_hdm_mesh_split)),
        ('segment_id->bias',
         self._Fn('bias',
                  fn=lambda x: self._EncNotVisible(x, x) * (-1e+09),
                  fn_out=lambda x: x + x[-1])),
        ('vec,wq,wk,wv->q,k,v', self._ComputeQKVCombine('qkv')),
        ('q,k,v,bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  def DecEncAttention(self,
                      name,
                      device_mesh=None,
                      w_qkv_mhd_mesh_split=None,
                      wo_hdm_mesh_split=None):
    """Transformer Decoder-Encoder Attention."""
    # pyformat: disable
    return self._Graph(
        name, self._DecoderLayerInMapKeys, [
            'outputs',
            'aux_loss',
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights(
            'w', device_mesh, w_qkv_mhd_mesh_split, wo_hdm_mesh_split)),
        ('segment_id,encoder_segment_id->bias',
         self._Fn('bias', fn=lambda a, b: -1e+09 * self._EncNotVisible(a, b))),
        ('vec,wq->q', self._ComputeQKV('q')),
        ('encoder_output,wk->k', self._ComputeQKV('k')),
        ('encoder_output,wv->v', self._ComputeQKV('v')),
        ('q,k,v,bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  def _DecNotVisible(self, segment_id, segment_pos):
    """Causal padding with segment_id and segment_pos."""
    a, b = tf.expand_dims(segment_id, -1), tf.expand_dims(segment_id, -2)
    ret = tf.equal(a, 0)
    ret = tf.logical_and(ret, tf.equal(b, 0))
    ret = tf.logical_or(ret, tf.not_equal(a, b))
    # position (~row) is less that memory position(~column)
    causal = tf.less(
        tf.expand_dims(segment_pos, -1), tf.expand_dims(segment_pos, -2))
    ret = tf.math.logical_or(causal, ret)
    return tf.cast(ret, py_utils.FPropDtype(self.params))

  def _DecComputeBiasGraphEdge(self):
    """Returns an edge of GraphLayer to compute attenion bias for Decoders."""
    return ('segment_id,segment_pos->qq_bias',
            self._Fn(
                'bias', fn=lambda x, y: self._DecNotVisible(x, y) * (-1e+09)))

  def DecSelfAttention(self,
                       name,
                       device_mesh=None,
                       w_qkv_mhd_mesh_split=None,
                       wo_hdm_mesh_split=None):
    """TransformerDecoder SelfAttention.

    Note that attention bias (see _DecNotvisible) ensures that current position
    (~row) is less that memory position(~column).

    Args:
      name: name of the layer.
      device_mesh: device_mesh for sharding (if specified)
      w_qkv_mhd_mesh_split: mesh split for qkv weigthts (if specified)
      wo_hdm_mesh_split: mesh split for output weights (if specified)

    Returns:
      layer params for TransformerDecoder SelfAttention.
    """
    p = self.params
    state_shape = [None, None, p.attention_num_heads, p.attention_key_value_dim]

    # pyformat: disable
    return self._Graph(
        name, self._DecoderLayerInMapKeys, [
            'outputs',
            'aux_loss',
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights(
            'w', device_mesh, w_qkv_mhd_mesh_split, wo_hdm_mesh_split)),
        ('vec,wq,wk,wv->q,k,v', self._ComputeQKVCombine('qkv')),
        ('k->k_full', self._AttentionState('k_state', state_shape)),
        ('v->v_full', self._AttentionState('v_state', state_shape)),
        self._DecComputeBiasGraphEdge(),
        ('qq_bias->bias_full', self._Override('dec_self_attention_bias')),
        ('q,k_full,v_full,bias_full->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  def _RelativePositionBucket(self, relative_position, bidirectional=False):
    p = self.params
    fprop_dtype = py_utils.FPropDtype(self.params)

    num_buckets = p.relative_attention_num_buckets
    max_distance = tf.cast(p.relative_attention_max_distance, fprop_dtype)
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += _ToInt32(tf.less(n, 0)) * num_buckets
      n = tf.math.abs(n)
    else:
      n = tf.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = tf.less(n, max_exact)
    # should be component-wise tf.math.log
    val_if_large = max_exact + _ToInt32(
        tf.math.log(tf.cast(n, fprop_dtype) / max_exact) /
        tf.math.log(max_distance / max_exact) * (num_buckets - max_exact))
    val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
    ret += tf.where(is_small, n, val_if_large)
    return ret

  # When training query_segment_pos = key_segment_pos, of shape [batch, time].
  # When decoding query_segment_pos is [batch, beam_size]
  # but key_segment_pos is [batch, memory_size] (because of k_pos StateLayer).
  def _AddRelativeBias(self,
                       bias,
                       query_segment_pos,
                       key_segment_pos,
                       relative_bias_weights,
                       bidirectional=False):
    p = self.params
    fprop_dtype = py_utils.FPropDtype(self.params)

    if p.relative_attention_use_universal_1d_position:
      assert (int(key_segment_pos.shape[-1]) == int(
          query_segment_pos.shape[-1])), (key_segment_pos.shape,
                                          query_segment_pos.shape)
      batch_size = query_segment_pos.shape.as_list()[0]
      len_dim = key_segment_pos.shape.as_list()[-1]
      key_segment_pos = query_segment_pos = tf.expand_dims(
          tf.range(len_dim), axis=0)

    # Relative position is defined in such a way that when query is in the
    # future relative to the key, the value of relative position is negative.
    relative_position = (
        tf.expand_dims(key_segment_pos, -2) -
        tf.expand_dims(query_segment_pos, -1))
    relative_bucket = self._RelativePositionBucket(relative_position,
                                                   bidirectional)

    relative_bucket_one_hot = tf.one_hot(
        relative_bucket, p.relative_attention_num_buckets, dtype=fprop_dtype)
    # relative_bucket_one_hot:
    # ..LJX - [batch?, length, memory_length, num_buckets]
    #
    # relative_bias_weights:
    # HX - [num_heads, num_buckets]
    #
    # relative_bias_inc:
    # [batch?, length, heads, memory_length]
    if (p.relative_attention_use_universal_1d_position and
        p.inflate_universal_relative_bias_to_match_batch_dimension):
      assert relative_bucket_one_hot.shape.ndims == 4
      relative_bucket_one_hot = tf.tile(relative_bucket_one_hot,
                                        [batch_size, 1, 1, 1])

    relative_bias_inc = tf.einsum('HX,...LJX->...LHJ', relative_bias_weights,
                                  relative_bucket_one_hot)
    if (relative_bias_inc.shape.ndims == 3 and
        not p.inflate_universal_relative_bias_to_match_batch_dimension):
      assert p.relative_attention_use_universal_1d_position
      relative_bias_inc = tf.expand_dims(relative_bias_inc, 0)

    # Eventually we add bias to BLHM [batch, length, heads, memory_length]
    # logits tensor, so we make 'heads' dim next to last.

    return tf.expand_dims(bias, -2) + relative_bias_inc

  def _EncoderAddRelativeBias(self, bias, segment_pos, relative_bias_weights):
    query_segment_pos, key_segment_pos = segment_pos, segment_pos
    bidirectional = True  # Encoder attention bias is always bidirectional.
    return self._AddRelativeBias(bias, query_segment_pos, key_segment_pos,
                                 relative_bias_weights, bidirectional)

  def SelfAttentionRelativeBias(self,
                                name,
                                device_mesh=None,
                                w_qkv_mhd_mesh_split=None,
                                wo_hdm_mesh_split=None):
    """TransformerEncoder SelfAttention with relative Attention Bias."""
    p = self.params
    collections = None
    if p.relative_attention_type == 'bias_shared':
      # Collection name is used as a unique ID to retrieve the shared variable.
      #
      # This name must be different for SelfAttentionRelativeBias (Encoder), and
      # must have a suffix matching shared_var_collection_suffix, e.g.
      # 'shared_var'.
      collections = ['_self_attention_shared_var']
    else:
      assert p.relative_attention_type == 'bias', p.relative_attention_type

    # pyformat: disable
    return self._Graph(
        name, self._EncoderLayerInMapKeys, [
            'outputs',
            'aux_loss'
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights(
            'w', device_mesh, w_qkv_mhd_mesh_split, wo_hdm_mesh_split)),
        ('->relative_bias_weights',
         self._RelativeAttentionBiasWeights('wrb', collections)),
        ('segment_id->segment_bias',
         self._Fn('bias',
                  fn=lambda x: self._EncNotVisible(x, x) * (-1e+09),
                  fn_out=lambda x: x + x[-1])),
        ('segment_bias,segment_pos,relative_bias_weights->bias',
         self._Fn('relative_bias', fn=self._EncoderAddRelativeBias)),
        ('vec,wq,wk,wv->q,k,v', self._ComputeQKVCombine('qkv')),
        ('q,k,v,bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  def DecSelfAttentionRelativeBias(self,
                                   name,
                                   device_mesh=None,
                                   w_qkv_mhd_mesh_split=None,
                                   wo_hdm_mesh_split=None):
    """DecSelfAttention with relative Attention Bias.

    Note that attention bias (see _DecNotVisible) ensures that current position
    (~row) is less that memory position(~column).

    In addition to masking bias we use per-head per-relative position bucket
    relative_bias_weights tensor (see _RelativeAttentionBiasWeights) of shape
    [num heads, num relative position buckets]
    (e.g. [128, 32] for Meena 64B).

    We compute relative position bucket for every position pair, relative_bucket
    tensor of shape [batch, length, length] and do
    tf.gather(relative_bias_weights, relative_bucket, axis=1)
    to compute per position-pair bias.

    Args:
      name: name of the layer.
      device_mesh: device_mesh for sharding (if specified)
      w_qkv_mhd_mesh_split: mesh split for qkv weigthts (if specified)
      wo_hdm_mesh_split: mesh split for output weights (if specified)

    Returns:
      The layer params.
    """
    p = self.params
    collections = None
    if p.relative_attention_type == 'bias_shared':
      # Collection name is used as a unique ID to retrieve the shared variable.
      #
      # This name must be different for SelfAttentionRelativeBias (Encoder), and
      # must have a suffix matching shared_var_collection_suffix, e.g.
      # 'shared_var'.
      collections = ['_dec_self_attention_shared_var']
    else:
      assert p.relative_attention_type == 'bias', p.relative_attention_type

    state_shape = [None, None, p.attention_num_heads, p.attention_key_value_dim]

    # pyformat: disable
    return self._Graph(
        name, self._DecoderLayerInMapKeys, [
            'outputs',
            'aux_loss',
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights(
            'w', device_mesh, w_qkv_mhd_mesh_split, wo_hdm_mesh_split)),
        ('->relative_bias_weights', self._RelativeAttentionBiasWeights('wrb', collections)),
        ('vec,wq,wk,wv->q,k,v', self._ComputeQKVCombine('qkv')),
        ('k->k_full', self._AttentionState('k_state', state_shape)),
        ('v->v_full', self._AttentionState('v_state', state_shape)),
        ('segment_pos->key_segment_pos',
         self._AttentionState('seg_pos', [None, None], dtype=tf.int32)),
        self._DecComputeBiasGraphEdge(),
        ('qq_bias->qk_bias', self._Override('dec_self_attention_bias')),
        ('qk_bias,segment_pos,key_segment_pos,relative_bias_weights->qhk_bias',
         # Decoder _AddRelativeBias always has bidirectional=False.
         self._Fn('relative_bias', fn=self._AddRelativeBias)),
        ('q,k_full,v_full,qhk_bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  # BEGIN GOOGLE-INTERNAL
  # TODO(davidso): Make this public once the Primer paper has been released.
  def DecMultiDconvHeadAttentionRelativeBias(self,
                                             name,
                                             device_mesh=None,
                                             w_qkv_mhd_mesh_split=None,
                                             wo_hdm_mesh_split=None):
    """Primer Multi-Dconv-Head Attention with relative attention bias.

    This follows the same logic as DecSelfAttentionRelativeBias(), with the
    Primer multi-head depthwise convolutions.

    Args:
      name: name of the layer.
      device_mesh: device_mesh for sharding (if specified).
      w_qkv_mhd_mesh_split: mesh split for qkv weigthts (if specified).
      wo_hdm_mesh_split: mesh split for output weights (if specified).

    Returns:
      The layer params.
    """
    p = self.params
    collections = None
    if p.relative_attention_type == 'bias_shared':
      # Collection name is used as a unique ID to retrieve the shared variable.
      #
      # This name must be different for SelfAttentionRelativeBias (Encoder), and
      # must have a suffix matching shared_var_collection_suffix, e.g.
      # 'shared_var'.
      collections = ['_dec_self_attention_shared_var']
    else:
      assert p.relative_attention_type == 'bias', p.relative_attention_type

    state_shape = [None, None, p.attention_num_heads, p.attention_key_value_dim]

    # pyformat: disable
    return self._Graph(
        name, self._DecoderLayerInMapKeys, [
            'outputs',
            'aux_loss',
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights(
            'w', device_mesh, w_qkv_mhd_mesh_split, wo_hdm_mesh_split)),
        ('->relative_bias_weights', self._RelativeAttentionBiasWeights('wrb', collections)),
        ('vec,wq,wk,wv->pre_q,pre_k,pre_v', self._ComputeQKVCombine('qkv')),
        # Note: This does not use shared Q and K representations.
        ('pre_q->q',
         self.DepthwiseConvAutoregressive('q_dconv',
                                          kernel_size=3,
                                          model_dims=[p.attention_num_heads,
                                                      p.attention_key_value_dim])),
        ('pre_k->k',
         self.DepthwiseConvAutoregressive('k_dconv',
                                          kernel_size=3,
                                          model_dims=[p.attention_num_heads,
                                                      p.attention_key_value_dim])),
        ('pre_v->v',
         self.DepthwiseConvAutoregressive('v_dconv',
                                          kernel_size=3,
                                          model_dims=[p.attention_num_heads,
                                                      p.attention_key_value_dim])),
        ('k->k_full', self._AttentionState('k_state', state_shape)),
        ('v->v_full', self._AttentionState('v_state', state_shape)),
        ('segment_pos->key_segment_pos',
         self._AttentionState('seg_pos', [None, None], dtype=tf.int32)),
        self._DecComputeBiasGraphEdge(),
        ('qq_bias->qk_bias', self._Override('dec_self_attention_bias')),
        ('qk_bias,segment_pos,key_segment_pos,relative_bias_weights->qhk_bias',
         # Decoder _AddRelativeBias always has bidirectional=False.
         self._Fn('relative_bias', fn=self._AddRelativeBias)),
        ('q,k_full,v_full,qhk_bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  # END GOOGLE-INTERNAL

  def _RelativeAttentionBiasWeights(self, name, collections=None):
    """Helper for '->rb' Graph edge."""
    p = self.params

    if collections is not None:
      name += collections[0]
    shared_var_collection_suffix = None
    if collections is not None and collections:
      shared_var_collection_suffix = 'shared_var'
    rb_stddev = (p.attention_num_heads * p.relative_attention_num_buckets)**-0.5
    rb_tpl = py_utils.WeightParams(
        shape=[p.attention_num_heads, p.relative_attention_num_buckets],
        dtype=self.params.dtype,
        collections=collections,
        init=py_utils.WeightInit.Gaussian(rb_stddev))
    return self._Var(
        name=name,
        weights=[('wrb', rb_tpl)],
        shared_var_collection_suffix=shared_var_collection_suffix)

  def _zero_aux_loss(self, name):  # pylint: disable=invalid-name
    return self._Fn(name,
                    lambda: tf.constant(0.0, py_utils.FPropDtype(self.params)))

  def CausalDepthwiseConv(self, name, kernel_size, model_dims=None):
    p = self.params
    model_dims = model_dims or [p.model_dim]
    model_dims = model_dims if p.conv_vars_reshape else [np.prod(model_dims)]
    return gshard_layers.CausalDepthwiseConv1DLayer.Params().Set(
        name=name, kernel_size=kernel_size, model_dims=model_dims)

  def DepthwiseConvAutoregressive(self, name, kernel_size, model_dims=None):
    r"""Depthwise convolution for autoregressive models.

    Same implementation as mesh_tensorflow/
    transformer/transformer.sublayer_depthwise_conv_autoregressive

    Given an input x of shape [B, L, M] and kernel_size K, there are K variables
    W[k] each with shape [M] so they represent the conv kernel [K, M].

    The output
      Y[:, t, :] = \sum_k W[k] * X[:, t - k, :]
      Y = W[0] * X + W[1] * Shift(X, 1) + W[2] * Shift(X, 2), ...
      where Shift(X, d) function rolls X forward in the time dimension by d.

    Args:
      name: Name of the layer.
      kernel_size: an integer.
      model_dims: Overridden model dimensions.

    Returns:
      A layer params that computes DepthwiseConvAutoregressive.
    """
    p = self.params
    if p.use_fused_depthwise_conv_autoregressive:
      return self.CausalDepthwiseConv(name, kernel_size, model_dims)

    var_shape = model_dims or [p.model_dim]

    def _GetScaleVar(shift_distance):
      init_const = 0.5 if shift_distance == 0 else 0.5 / kernel_size
      scale_var_weight_params = py_utils.WeightParams(
          init=py_utils.WeightInit.Constant(init_const),
          dtype=p.dtype,
          shape=var_shape if p.conv_vars_reshape else [np.prod(var_shape)])
      return self._Var(
          name='w_%d' % shift_distance,
          weights=[('scale', scale_var_weight_params)])

    def _Shift(x):
      """Shift x to right by 1 in time dim and pad with zeros."""
      return tf.concat([tf.zeros_like(x[:, -1:]), x[:, :-1]], axis=1)

    # Y = W[0] * X + W[1] * Shift(X, 1) + W[2] * Shift(X, 2) + ...
    # Iteratively:
    # Y_1 = W[0] * X_0
    # Y_2 = Y_1 + W[1] * X_1 = Y_1 + W[1] * _Shift(X_0)
    # ...
    # Y_{d+1} = Y_d + W[d] * X_d = Y_d + W[d] * _Shift(X_{d-1})
    sub_params = [('->w_0', _GetScaleVar(0)),
                  ('x_0,w_0->y_1',
                   self._Fn('mul0', lambda x, w: x * tf.reshape(w, var_shape)))]

    for d in range(1, kernel_size):
      sub_params += [
          ('x_%d->x_%d' % (d - 1, d), self._Fn('shift_%d' % d, _Shift)),
          ('->w_%d' % d, _GetScaleVar(d)),
          ('y_%d,x_%d,w_%d->y_%d' % (d, d, d, d + 1),
           self._Fn('scale_%d' % d,
                    lambda x, x2, w: x + x2 * tf.reshape(w, var_shape)))
      ]

    return self._Graph(name, ['x_0'], ['y_%d' % kernel_size], *sub_params)

  def _LNNoScale(self, name):

    def _RmsNormNoScale(x):
      eps = self.params.layer_norm_epsilon
      axis = [d + 2 for d in range(len(x.shape) - 2)]
      variance = tf.reduce_mean(tf.math.square(x), keepdims=True, axis=axis)
      return x * tf.math.rsqrt(variance + eps)

    return self._Fn(name, _RmsNormNoScale)

  def _LN(self, name):
    """Overriding with bias-less layer norm."""
    if self.params.ln_no_scale:
      return self._LNNoScale(name)

    return self._LNInternal(name)

  def _LNConv(self, name, conv_kernel_size):
    return self._Seq(name, self._LNNoScale('ln_no_scale'),
                     self.DepthwiseConvAutoregressive('conv', conv_kernel_size))

  def _LNInternal(self, name, ln_weight_reshape=None):
    """Internal implementation of _LN with optional reshape of the weight."""

    def LN(x, scale):
      eps = self.params.layer_norm_epsilon
      # BLm Tensor (m=1, reduced model_dim) or BLnm where model dim is split to
      # two dims.
      axis = [d + 2 for d in range(len(x.shape) - 2)]
      variance = tf.reduce_mean(tf.math.square(x), keepdims=True, axis=axis)
      if ln_weight_reshape is not None:
        scale = tf.reshape(scale, ln_weight_reshape)
      return x * tf.math.rsqrt(variance + eps) * scale

    ln_weight_params = py_utils.WeightParams(
        init=py_utils.WeightInit.Constant(1.0),
        dtype=self.params.dtype,
        shape=[self.params.model_dim])

    return self._Graph(
        name, ['x'], ['x_norm'],
        ('->scale', self._Var(name='w', weights=[('scale', ln_weight_params)])),
        ('x,scale->x_norm', self._Fn('ln', LN)))

  def _TrueLN(self, name, ln_weight_reshape=None):
    """True LN normalization."""

    def LN(x, scale, shift):
      eps = self.params.layer_norm_epsilon
      # BLm Tensor (m=1, reduced model_dim) or BLnm where model dim is split to
      # two dims.
      axis = [d + 2 for d in range(len(x.shape) - 2)]
      squared_mean_center = tf.math.square(
          (x - tf.reduce_mean(x, keepdims=True, axis=axis)))
      variance = tf.reduce_mean(squared_mean_center, keepdims=True, axis=axis)
      if ln_weight_reshape is not None:
        scale = tf.reshape(scale, ln_weight_reshape)
        shift = tf.reshape(shift, ln_weight_reshape)
      return x * tf.math.rsqrt(variance + eps) * scale + shift

    ln_scale_params = py_utils.WeightParams(
        init=py_utils.WeightInit.Constant(1.0),
        dtype=self.params.dtype,
        shape=[self.params.model_dim])
    ln_shift_params = py_utils.WeightParams(
        init=py_utils.WeightInit.Constant(0.0),
        dtype=self.params.dtype,
        shape=[self.params.model_dim])

    return self._Graph(
        name, ['x'], ['x_norm'],
        ('->scale',
         self._Var(name='w_scale', weights=[('scale', ln_scale_params)])),
        ('->shift',
         self._Var(name='w_shift', weights=[('shift', ln_shift_params)])),
        ('x,scale,shift->x_norm', self._Fn('true_ln', LN)))

  # BEGIN GOOGLE-INTERNAL
  # TODO(davidso): Make this public when the Primer paper is released.
  def _PN(self, name, ln_weight_reshape=None):
    """Primer normalization."""

    def PN(x, scale, shift):
      eps = self.params.layer_norm_epsilon
      # BLm Tensor (m=1, reduced model_dim) or BLnm where model dim is split to
      # two dims.
      axis = [d + 2 for d in range(len(x.shape) - 2)]
      temp = (x - tf.reduce_mean(x, keepdims=True, axis=axis)) * x
      mock_variance = tf.reduce_mean(temp, keepdims=True, axis=axis)
      if ln_weight_reshape is not None:
        scale = tf.reshape(scale, ln_weight_reshape)
        shift = tf.reshape(shift, ln_weight_reshape)
      return x * tf.math.rsqrt(mock_variance + eps) * scale + shift

    pn_scale_params = py_utils.WeightParams(
        init=py_utils.WeightInit.Constant(1.0),
        dtype=self.params.dtype,
        shape=[self.params.model_dim])
    pn_shift_params = py_utils.WeightParams(
        init=py_utils.WeightInit.Constant(0.0),
        dtype=self.params.dtype,
        shape=[self.params.model_dim])

    return self._Graph(
        name, ['x'], ['x_norm'],
        ('->scale',
         self._Var(name='w_scale', weights=[('scale', pn_scale_params)])),
        ('->shift',
         self._Var(name='w_shift', weights=[('shift', pn_shift_params)])),
        ('x,scale,shift->x_norm', self._Fn('pn', PN)))

  # END GOOGLE-INTERNAL

  def Split(self, name):
    """Sets sharding attribute for the Tensor. Split across dim=0."""
    tf.logging.warning('gshard_utils.Split is deprecated. '
                       'Please use gshard_utils.MeshSplit with specific '
                       'device_mesh and device_mesh_shape set in the Builder.')
    return self._Fn(
        name,
        lambda x: gshard_utils.Split(x, 0, num_devices=self.params.num_devices))

  def _Add(self, name, residual_weight=1.0):
    return self._Fn(
        name, fn=lambda x, y: x + residual_weight * y, fn_out=lambda x, y: x)

  def _Identity(self, name):
    """Apply identity transformation."""
    return layers.IdentityLayer.Params().Set(name=name)

  def _AttentionWeights(self,
                        name,
                        device_mesh=None,
                        w_qkv_mhd_mesh_split=None,
                        wo_hdm_mesh_split=None):
    """Helper for '->wq,wk,wv,wo' Graph edge."""

    p = self.params
    h = p.attention_num_heads
    hd_dims = ([h * p.attention_key_value_dim]
               if p.attention_combine_dims else [h, p.attention_key_value_dim])
    h = p.attention_num_memory_heads or p.attention_num_heads
    kv_hd_dims = ([h * p.attention_key_value_dim] if p.attention_combine_dims
                  else [h, p.attention_key_value_dim])
    q_stddev = (p.model_dim * p.attention_key_value_dim)**-0.5

    if p.attention_combine_dims:
      if w_qkv_mhd_mesh_split is None:
        w_qkv_mesh_split = None
      else:
        # hd can not be both sharded. Use negative indices in case there is an
        # additional leading pipeline stage dimension.
        assert (w_qkv_mhd_mesh_split[-1] < 0 or w_qkv_mhd_mesh_split[-2] < 0), (
            'hd can not be both sharded %s' % w_qkv_mhd_mesh_split)
        w_qkv_mesh_split = w_qkv_mhd_mesh_split[:-3] + [
            w_qkv_mhd_mesh_split[-3],
            max(w_qkv_mhd_mesh_split[-1], w_qkv_mhd_mesh_split[-2])
        ]

      if wo_hdm_mesh_split is None:
        wo_mesh_split = None
      else:
        # wo_hdm_mesh_split is almost always set via
        # DenseBuilder._attention_output_hdm_w_split, e.g.
        #   [p.mhd_w_split[-2], p.mhd_w_split[-1], p.mhd_w_split[-3]]
        #
        # TODO(lepikhin): this logic needs to be explicit, e.g. via
        # SetSplitsForCombinedAttentionDims utility function.
        assert (wo_hdm_mesh_split[-3] < 0 or
                wo_hdm_mesh_split[-2] < 0), ('hd can not be both sharded %s' %
                                             wo_hdm_mesh_split)
        wo_mesh_split = wo_hdm_mesh_split[:-3] + [
            max(wo_hdm_mesh_split[-3], wo_hdm_mesh_split[-2]),
            wo_hdm_mesh_split[-1],
        ]
    else:
      w_qkv_mesh_split = w_qkv_mhd_mesh_split
      wo_mesh_split = wo_hdm_mesh_split

    wq_tpl = gshard_layers.ShardedWeightParams(
        shape=[p.model_dim] + hd_dims,
        dtype=self.params.dtype,
        init=py_utils.WeightInit.Gaussian(q_stddev),
        tensor_split_dims_mapping=w_qkv_mesh_split)
    kv_stddev = (p.model_dim)**-0.5
    wkv_tpl = gshard_layers.ShardedWeightParams(
        shape=[p.model_dim] + kv_hd_dims,
        dtype=self.params.dtype,
        init=py_utils.WeightInit.Gaussian(kv_stddev),
        tensor_split_dims_mapping=w_qkv_mesh_split)
    o_stddev = (p.attention_num_heads * p.attention_key_value_dim)**-0.5
    wo_tpl = gshard_layers.ShardedWeightParams(
        shape=hd_dims + [p.model_dim],
        dtype=self.params.dtype,
        init=py_utils.WeightInit.Gaussian(o_stddev),
        tensor_split_dims_mapping=wo_mesh_split)

    return self._ShardedVar(
        name=name,
        weights=[('wq', wq_tpl), ('wk', wkv_tpl), ('wv', wkv_tpl),
                 ('wo', wo_tpl)],
        device_mesh=device_mesh)

  def _ComputeQKV(self, name):
    p = self.params

    def _Compute(x, w):
      if p.attention_combine_dims:
        w = tf.reshape(
            w, [p.model_dim, p.attention_num_heads, p.attention_key_value_dim])
      return tf.einsum('BLM,MHD->BLHD', x, w)

    return self._Fn(name, _Compute)

  def _ComputeQKVCombine(self, name):
    p = self.params

    def _Compute(x, wq, wk, wv):

      def _GetW(w, h):
        if p.attention_combine_dims:
          w = tf.reshape(w, [p.model_dim, h, p.attention_key_value_dim])
        return w

      wq = _GetW(wq, p.attention_num_heads)
      wk = _GetW(wk, p.attention_num_memory_heads or p.attention_num_heads)
      wv = _GetW(wv, p.attention_num_memory_heads or p.attention_num_heads)
      wc = [wq, wk, wv]

      if ((p.attention_num_memory_heads and
           p.attention_num_heads != p.attention_num_memory_heads) or
          not p.attention_combine_qkv):
        # Combined tf.einsum is not possible, falling back to individual
        # einsum ops.
        return [tf.einsum('BLM,MHD->BLHD', x, w) for w in wc]

      wc = [tf.expand_dims(w, 0) for w in wc]
      wc = tf.concat(wc, 0)
      return [
          tf.squeeze(y, 0)
          for y in tf.split(tf.einsum('BLM,KMHD->KBLHD', x, wc), 3, 0)
      ]

    return self._Fn(name, _Compute)

  def _Top2GatingWeights(self, name):
    p = self.params
    stddev = (1. / p.model_dim)**0.5
    init_scale = stddev * 3.**0.5
    return self._Var(
        name=name,
        weights=[('w',
                  py_utils.WeightParams(
                      shape=[p.model_dim, p.e_dim],
                      init=py_utils.WeightInit.Uniform(init_scale),
                      dtype=p.dtype))])

  def _ComputeTopKGating(self, name):
    p = self.params

    def _Compute(w, inputs, paddings):
      return gshard_layers.Top2Gating(
          w=w,
          inputs=inputs,
          paddings=paddings,
          num_devices=p.num_devices,
          experts_dim=p.e_dim,
          expert_capacity_dim=p.c_dim,
          model_dim_reshape_segments=self._model_dim_reshape_segments,
          local_dispatch=True,
          fprop_dtype=py_utils.FPropDtype(p),
          mask_dtype=p.mask_dtype,
          gating_logits_dtype=p.gating_logits_dtype,
          # We rely on sharding propagation here, Top2Gating is done
          # independently for each group and inputs are typically sharded by
          # group dimension.
          use_xla_sharding=False,
          second_expert_policy=p.second_expert_policy,
          second_expert_threshold=p.second_expert_threshold,
          legacy_mtf_behavior=p.legacy_mtf_behavior,
          capacity_factor=p.capacity_factor)

    return self._Fn(name, _Compute)

  def _ShardedFeedForwardNetworksWeights(self, name):
    """Gets the sharded weights for the two layer feedforward nets."""
    tf.logging.warning(
        'Deprecated. DenseBuilder should universally be used for MoE, '
        'Dense and Hybrid models.')
    p = self.params
    emh_shape = [p.e_dim, p.model_dim, p.moe_hidden_dim]
    # See VarianceScalingInitializer in py_utils
    #   scale        ~ 1.0
    #   reduced_dims ~ params.input_dim
    #   mode         ~ 'fan_in'
    #
    stddev = (1. / p.model_dim)**0.5
    wi_kernel_param_init_scale = stddev * 3.**0.5
    wi_pc = py_utils.WeightParams(
        shape=emh_shape,
        init=py_utils.WeightInit.Uniform(wi_kernel_param_init_scale),
        dtype=p.dtype)

    # EHM Tensor (output transformation after RELU)
    ehm_shape = [p.e_dim, p.moe_hidden_dim, p.model_dim]
    # See VarianceScalingInitializer in py_utils
    #   scale        ~ 1.0
    #   reduced_dims ~ params.moe_hidden_dim
    #   mode         ~ 'fan_in'
    #
    stddev = (1. / p.moe_hidden_dim)**0.5
    wo_kernel_param_init_scale = stddev * 3.**0.5
    wo_pc = py_utils.WeightParams(
        shape=ehm_shape,
        init=py_utils.WeightInit.Uniform(wo_kernel_param_init_scale),
        dtype=p.dtype)
    return self._ShardedVarOn1DDeviceArray(
        name=name, weights=[('wi', wi_pc), ('wo', wo_pc)])

  def _FeedForwardNetworksApplyGating(self, name):
    p = self.params
    if p.num_devices and p.num_devices > 1:
      tf.logging.warning('Split API is deprecated. '
                         'Use device_mesh and MeshSplit.')

    def _Compute(gating, inputs, reshaped_inputs, wi, wo):
      return gshard_layers.FeedForwardNetworksApplyGating(
          gating,
          inputs,
          reshaped_inputs,
          wi,
          wo,
          num_devices=p.num_devices,
          num_groups=p.num_groups or p.num_devices,
          dropout_rate=p.moe_dropout_rate,
          device_mesh=self._device_mesh,
          model_dim_reshape_segments=self._model_dim_reshape_segments,
          gsm_split=self._AdjustMSplitByName('blm_split'),
          egcm_split=self._AdjustMSplitByName('egcm_split'),
          gecm_split=self._AdjustMSplitByName('gecm_split'),
          gsec_split=self._AdjustMSplitByName('gsec_split'),
          eah_split=self._AdjustMSplitByName('eah_split'),
          eam_split=self._AdjustMSplitByName('eam_split'),
          activation_name=p.moe_activation)

    return self._Fn(name, _Compute)

  def _ShardedMoEPositionWiseFeedForwardNetworks(self, name):
    """Simple MoE FFN with xla_sharding."""
    tf.logging.warning(
        'Deprecated. DenseBuilder should universally be used for MoE, '
        'Dense and Hybrid models.')
    p = self.params
    num_groups = p.num_groups or p.num_devices

    reshape_input = gshard_layers.ReshapeInputLayer.Params().Set(
        num_groups=num_groups, num_devices=p.num_devices)
    return self._Graph(
        name, ['inputs', 'segment_id', 'wi', 'wo'], ['outputs', 'aux_loss'],
        ('inputs,segment_id->reshaped_inputs, paddings', reshape_input),
        ('->gw', self._Top2GatingWeights('top_2_gating')),
        ('gw,reshaped_inputs,paddings->gating',
         self._ComputeTopKGating('compute_gating')),
        ('gating,inputs,reshaped_inputs,wi,wo->outputs,aux_loss',
         self._FeedForwardNetworksApplyGating('process_gating')))

  def _AttentionState(self, name, shape, dtype=None):
    p = self.params
    dtype = dtype or py_utils.FPropDtype(p)
    return gshard_layers.MultiHeadAttentionStateLayer.Params().Set(
        name=name,
        shape=shape,
        dtype=dtype,
        use_xla_dynamic_update_slice=p.use_xla_dynamic_update_slice)

  def _Override(self, name, key=None):
    return gshard_layers.OverrideLayer.Params().Set(name=name, key=key or name)

  def _Softmax(self, dec_outs, tgt, w, vocab_dim):
    p = self.params

    def _MaybeSplit(x):
      return gshard_utils.Split(x, 0, p.num_devices)

    dec_outs *= (p.model_dim**-0.5)
    logits = _MaybeSplit(tf.einsum('BLM,VM->BLV', _MaybeSplit(dec_outs), w))
    label_smoothing = p.label_smoothing
    off_value = label_smoothing / vocab_dim
    on_value = 1.0 - label_smoothing + off_value
    soft_targets = _MaybeSplit(
        tf.one_hot(
            tgt.labels, vocab_dim, on_value=on_value, off_value=off_value))
    loss = _MaybeSplit(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=soft_targets, logits=logits))
    non_padding = _MaybeSplit(
        tf.cast(
            tf.not_equal(tgt.segment_ids, 0), py_utils.FPropDtype(self.params)))
    per_token_loss = _MaybeSplit(loss * non_padding)
    loss_denom = tf.reduce_sum(tf.ones_like(non_padding), 1)
    return py_utils.NestedMap(
        per_example_loss=tf.reduce_sum(per_token_loss, 1) / loss_denom)

  def SmoothedSoftmax(self, name, vocab_dim):
    """Returns the Softmax layer with optional label smoothing."""
    return self._Graph(
        name, ['i'], ['o'], ('->w', self._EmbeddingWeight('w', vocab_dim)),
        ('i.dec_outs,i.tgt,w->o',
         self._Fn('softmax',
                  lambda x, t, w: self._Softmax(x, t, w, vocab_dim))))


class DenseBuilder(MoEBuilder):
  """Desnse layrs with GShard annotations."""

  @classmethod
  def Params(cls):
    p = super().Params()
    # p.Delete('e_dim')
    # p.Delete('c_dim')
    # p.Delete('moe_hidden_dim')
    # p.Delete('second_expert_policy')
    # p.Delete('second_expert_threshold')
    # p.Delete('capacity_factor')
    p.Define('device_mesh_shape', None, 'Device mesh shape.')

    # Weight sharding configs.
    p.Define('emb_w_split', None, 'Mesh split for embedding weights.')
    p.Define('mhd_w_split', [0, 1, -1], 'Mesh split for attention MHD weight')
    p.Define('kv_mhd_w_split', None, 'Mesh split for K/V MHD weight')
    p.Define('mh_wi_split', [0, 1], 'Mesh split for dense MH weight')
    p.Define('hm_wo_split', [1, 0], 'Mesh split for dense HM weight')

    # Activation sharding configs.
    # one_hot_ids_split split should be inferred by XLA.
    p.Define('one_hot_ids_split', None, 'Mesh split for one_hot_ids.')
    p.Define('emb_out_split', [0, -1, -1], 'Mesh split for embedding outputs.')
    p.Define('qkv_split', [0, -1, 1, -1], 'Mesh split for Q, K, V')
    p.Define('blm_split', [0, -1, -1], 'Mesh split for BLM.')
    p.Define('blh_split', [0, -1, 1], 'Mesh split for BLH.')
    p.Define('egcm_split', [0, -1, -1, -1], 'Mesh split for EGCM.')
    p.Define('gecm_split', [0, -1, -1, -1], 'Mesh split for GECM.')
    p.Define('gsec_split', [0, -1, -1, -1], 'Mesh split for GSEC.')
    p.Define('eah_split', [0, -1, -1], 'Mesh split for EAH.')
    p.Define('eam_split', [0, -1, -1], 'Mesh split for EAM.')
    p.Define('emh_split', [0, -1, -1], 'Mesh split for EMH.')
    p.Define('ehm_split', [0, -1, -1], 'Mesh split for EHM.')
    p.Define('logits_split', [0, -1, -1], 'Mesh split for logits.')
    p.Define('experimental_fix_split_dims_mapping', False,
             'Mesh split dims mapping could require a fix for special cases.')

    p.Define('atten_logit_cap', 0.0, 'Atten logit cap.')

    p.attention_combine_dims = False
    return p

  def _ReshapedModelDims(self):
    """Returns the dimensions that M is reshaped into."""
    if self.params.model_dim_reshape_segments is None:
      return [self.params.model_dim]
    remaining_dim = self.params.model_dim
    for d in self._model_dim_reshape_segments:
      assert remaining_dim % d == 0
      remaining_dim = remaining_dim // d
    return self._model_dim_reshape_segments + [remaining_dim]

  def _ReshapeM(self, x, m_dim):
    """Reshapes tensor x according to model_dim_reshape_segments."""
    new_shape = x.shape
    if self._model_dim_reshape_segments is not None:
      new_shape = list(x.shape[0:m_dim])
      new_shape += self._ReshapedModelDims()
      new_shape.extend(d for d in x.shape[m_dim + 1:])
    return tf.reshape(x, new_shape)

  def _EinsumWithModelDim(self, equation, x, y):
    """Einsum with adjusted equation according to model_dim_reshape_segments.

    It changes each dimension named 'M' in the equation into two dimensions 'NM'
    if model_dim_reshape_segments is set in the params. Therefore the original
    equation should not have 'N', and only use 'M' when it is expected to be
    reshaped.

    Args:
      equation: a string describing the contraction, in the same format as
        numpy.einsum.
      x: First input to einsum.
      y: second input to einsum.

    Returns:
      tf.einsum(maybe_modified_equation, x, y)
    """
    return gshard_layers.EinsumWithModelDim(equation, x, y,
                                            self._model_dim_reshape_segments)

  def DepthwiseConvAutoregressive(self, name, kernel_size, model_dims=None):
    model_dims = model_dims or self._ReshapedModelDims()
    return super().DepthwiseConvAutoregressive(name, kernel_size, model_dims)

  def EinsumWithModelDim(self, name, equation):

    def _Fn(x, y):
      return gshard_layers.EinsumWithModelDim(equation, x, y,
                                              self._model_dim_reshape_segments)

    return self._Fn(name, _Fn)

  def _LN(self, name):
    """Overriding _LN to consider model_dim_reshape_segments."""
    if self._model_dim_reshape_segments is None:
      return super()._LN(name)

    assert not self.params.ln_no_scale, ('attempting to call self._LNInternal '
                                         'instead of self._LNNoScale')
    ln_weight_reshape = self._ReshapedModelDims()
    return self._LNInternal(name, ln_weight_reshape)

  # BEGIN GOOGLE-INTERNAL
  # TODO(davidso): Make this public when the Primer paper is released.
  def _PN(self, name):
    """Overriding _PN to consider model_dim_reshape_segments."""
    if self._model_dim_reshape_segments is None:
      return super()._PN(name)

    pn_weight_reshape = self._ReshapedModelDims()
    return super()._PN(name, pn_weight_reshape)

  # END GOOGLE-INTERNAL

  def _TrueLN(self, name):
    """Overriding _TrueLN to consider model_dim_reshape_segments."""
    if self._model_dim_reshape_segments is None:
      return super()._TrueLN(name)

    ln_weight_reshape = self._ReshapedModelDims()
    return super()._TrueLN(name, ln_weight_reshape)

  @property
  def _device_mesh(self):
    p = self.params
    device_mesh = p.device_mesh
    if device_mesh is None:
      if p.device_mesh_shape is None:
        return None
      num_devices = np.product(p.device_mesh_shape)
      device_mesh = np.reshape(np.arange(0, num_devices), p.device_mesh_shape)
    elif p.device_mesh_shape is not None:
      assert p.device_mesh_shape == list(
          device_mesh.shape), (p.device_mesh_shape, list(device_mesh.shape))
    return device_mesh

  def _MeshSplit(self, x, tensor_split_dims_mapping):
    if tensor_split_dims_mapping is None:
      return x

    if self.params.experimental_fix_split_dims_mapping:
      # TODO(lepikhin,yuanxz): fix me.
      # Required for split by attention head dim which could be 1 in special
      # cases.
      tensor_split_dims_mapping = tensor_split_dims_mapping.copy()
      for dim, idx in enumerate(tensor_split_dims_mapping):
        if idx < 0:
          continue

        d = x.shape.as_list()[dim]
        if d == 1:
          tf.logging.info('Fixing bad tensor_split_dims_mapping %s %s', x,
                          tensor_split_dims_mapping)
          tensor_split_dims_mapping[dim] = -1
          continue
        m = self._device_mesh.shape[idx]
        assert (d % m == 0), (x, self._device_mesh.shape,
                              tensor_split_dims_mapping)

    return gshard_utils.MeshSplit(x, self._device_mesh,
                                  tensor_split_dims_mapping)

  def MeshSplit(self, name, tensor_split_dims_mapping):
    return self._Fn(name,
                    lambda x: self._MeshSplit(x, tensor_split_dims_mapping))

  def MoE(self, name, decoder=False):
    """Returns layer params to compute (outputs, scalar_aux_loss)."""
    if decoder:
      input_endpoints = self._DecoderLayerInMapKeys
    else:
      input_endpoints = self._EncoderLayerInMapKeys
    p = self.params
    return self._Graph(
        name, input_endpoints, ['outputs', 'aux_loss'],
        ('vec->input_split', self.MeshSplit('input_split', p.blm_split)),
        ('segment_id->segment_id_split',
         self.MeshSplit('segment_id_split', p.blm_split[:-1])),
        ('->wi,wo', self._ShardedFeedForwardNetworksWeights(name)),
        ('input_split,segment_id_split,wi,wo->outputs_pre_split,aux_loss',
         self._ShardedMoEPositionWiseFeedForwardNetworks('ffw')),
        ('outputs_pre_split->outputs',
         self.MeshSplit('outputs_split', p.blm_split)))

  def _ShardedFeedForwardNetworksWeights(self, name, model_dim=None):
    """Gets the sharded weights for the two layer feedforward nets."""
    p = self.params
    device_mesh = self._device_mesh
    if model_dim is None:
      model_dim = p.model_dim
    emh_shape = [p.e_dim, model_dim, p.moe_hidden_dim]
    # See VarianceScalingInitializer in py_utils
    #   scale        ~ 1.0
    #   reduced_dims ~ params.input_dim
    #   mode         ~ 'fan_in'
    #
    stddev = (1. / model_dim)**0.5
    wi_kernel_param_init_scale = stddev * 3.**0.5
    wi_pc = gshard_layers.ShardedWeightParams(
        shape=emh_shape,
        init=py_utils.WeightInit.Uniform(wi_kernel_param_init_scale),
        dtype=p.dtype,
        tensor_split_dims_mapping=p.emh_split)

    # EHM Tensor (output transformation after RELU)
    ehm_shape = [p.e_dim, p.moe_hidden_dim, model_dim]
    # See VarianceScalingInitializer in py_utils
    #   scale        ~ 1.0
    #   reduced_dims ~ params.moe_hidden_dim
    #   mode         ~ 'fan_in'
    #
    stddev = (1. / p.moe_hidden_dim)**0.5
    wo_kernel_param_init_scale = stddev * 3.**0.5
    wo_pc = gshard_layers.ShardedWeightParams(
        shape=ehm_shape,
        init=py_utils.WeightInit.Uniform(wo_kernel_param_init_scale),
        dtype=p.dtype,
        tensor_split_dims_mapping=p.ehm_split)
    return self._ShardedVar(
        name=name,
        weights=[('wi', wi_pc), ('wo', wo_pc)],
        device_mesh=device_mesh)

  def _ShardedMoEPositionWiseFeedForwardNetworks(self, name):
    """Simple MoE FFN with xla_sharding."""
    p = self.params
    num_groups = p.num_groups or p.num_devices

    reshape_input = gshard_layers.ReshapeInputLayer.Params().Set(
        num_groups=num_groups,
        num_devices=p.num_devices,
        model_dims=self._ReshapedModelDims(),
        device_mesh=p.device_mesh)

    return self._Graph(
        name, ['inputs', 'segment_id', 'wi', 'wo'], ['outputs', 'aux_loss'],
        ('inputs,segment_id->reshaped_inputs, paddings', reshape_input),
        ('->gw', self._Top2GatingWeights('top_2_gating')),
        ('gw->gw_reshaped',
         self._Fn('reshape_gw', fn=lambda x: self._ReshapeM(x, 0))),
        ('wi->wi_reshaped',
         self._Fn('reshape_wi', fn=lambda x: self._ReshapeM(x, 1))),
        ('wo->wo_reshaped',
         self._Fn('reshape_wo', fn=lambda x: self._ReshapeM(x, 2))),
        ('gw_reshaped,reshaped_inputs,paddings->gating',
         self._ComputeTopKGating('compute_gating')),
        ('gating,inputs,reshaped_inputs,wi_reshaped,wo_reshaped'
         '->outputs,aux_loss',
         self._FeedForwardNetworksApplyGating('process_gating')))

  def Embedding(self, name, vocab_dim):
    p = self.params
    return self._Graph(
        name, ['ids'], ['outputs'],
        ('->emb_orig',
         self._EmbeddingWeight('w', vocab_dim, self._device_mesh,
                               p.emb_w_split)),
        ('emb_orig->emb',
         self._Fn('reshape_emb_w', fn=lambda x: self._ReshapeM(x, 1))),
        ('ids->one_hot_ids', self._OneHotEncode('one_hot_ids', vocab_dim)),
        ('one_hot_ids->one_hot_ids_split',
         self.MeshSplit('one_hot_ids_split', p.one_hot_ids_split)),
        ('emb,one_hot_ids_split->outputs_pre_split',
         self.EinsumWithModelDim('einsum', 'VM,BLV->BLM')),
        ('outputs_pre_split->outputs',
         self.MeshSplit('out_split', self._AdjustMSplit(p.emb_out_split, 2))))

  @property
  def _attention_output_hdm_w_split(self):
    p = self.params
    # wo: hdm
    if p.mhd_w_split is None:
      return None
    # Use negative indices in case there is an additional pipeline stage
    # dimension.
    return p.mhd_w_split[:-3] + [
        p.mhd_w_split[-2], p.mhd_w_split[-1], p.mhd_w_split[-3]
    ]

  def SelfAttention(self, name):
    return super().SelfAttention(name, self._device_mesh,
                                 self.params.mhd_w_split,
                                 self._attention_output_hdm_w_split)

  def DecEncAttention(self, name):
    return super().DecEncAttention(name, self._device_mesh,
                                   self.params.mhd_w_split,
                                   self._attention_output_hdm_w_split)

  def DecSelfAttention(self, name):
    return super().DecSelfAttention(name, self._device_mesh,
                                    self.params.mhd_w_split,
                                    self._attention_output_hdm_w_split)

  def SelfAttentionRelativeBias(self, name):
    return super().SelfAttentionRelativeBias(name, self._device_mesh,
                                             self.params.mhd_w_split,
                                             self._attention_output_hdm_w_split)

  def DecSelfAttentionRelativeBias(self, name):
    return super().DecSelfAttentionRelativeBias(
        name, self._device_mesh, self.params.mhd_w_split,
        self._attention_output_hdm_w_split)

  # BEGIN GOOGLE-INTERNAL
  # TODO(davidso): Make this public when the Primer paper is released.
  def DecMultiDconvHeadAttentionRelativeBias(self, name):
    return super().DecMultiDconvHeadAttentionRelativeBias(
        name, self._device_mesh, self.params.mhd_w_split,
        self._attention_output_hdm_w_split)

  # END GOOGLE-INTERNAL

  def ParallelDecSelfAttentionRelativeBiasFFN(self,
                                              name,
                                              activation_fn,
                                              conv_kernel_size=None,
                                              hidden_dim_reshape_segments=4):
    """Runs DecSelfAttentionRelativeBias and FFNWithConv in parallel."""
    p = self.params
    collections = None
    if p.relative_attention_type == 'bias_shared':
      # Collection name is used as a unique ID to retrieve the shared variable.
      #
      # This name must be different for SelfAttentionRelativeBias (Encoder), and
      # must have a suffix matching shared_var_collection_suffix, e.g.
      # 'shared_var'.
      collections = ['_dec_self_attention_shared_var']
    else:
      assert p.relative_attention_type == 'bias', p.relative_attention_type

    state_shape = [None, None, p.attention_num_heads, p.attention_key_value_dim]

    def _Output(o, h1, h2, wo1, wo2):
      h = tf.math.multiply(h1, h2)
      return self._ComputeCombinedOutputs(o, h, wo1, wo2,
                                          hidden_dim_reshape_segments)

    if conv_kernel_size:
      d = p.ff_dim // hidden_dim_reshape_segments // p.attention_key_value_dim
      model_dims = [hidden_dim_reshape_segments, d, p.attention_key_value_dim]
      optional_conv_layer = self.DepthwiseConvAutoregressive(
          name='conv', kernel_size=conv_kernel_size, model_dims=model_dims)
    else:
      optional_conv_layer = self._Identity('conv')

    graph_inputs = self._DecoderLayerInMapKeys
    graph_outputs = ['outputs', 'aux_loss']
    sub_layers = [
        ('->wq,wk,wv,wo1',
         self._AttentionWeights('w_atten', self._device_mesh, p.mhd_w_split,
                                self._attention_output_hdm_w_split)),
        ('->wi_0,wi_1,wo2',
         self._DenseReluDenseWeightsGatedGELU('w_fflayer', self._device_mesh,
                                              p.mh_wi_split, p.hm_wo_split)),
        ('->relative_bias_weights',
         self._RelativeAttentionBiasWeights('wrb', collections)),
        ('vec,wq,wk,wv,wi_0,wi_1->q,k,v,h1,h2',
         self._ComputeQKVH('qkvh', hidden_dim_reshape_segments)),
        ('k->k_full', self._AttentionState('k_state', state_shape)),
        ('v->v_full', self._AttentionState('v_state', state_shape)),
        ('segment_pos->key_segment_pos',
         self._AttentionState('seg_pos', [None, None], dtype=tf.int32)),
        self._DecComputeBiasGraphEdge(),
        ('qq_bias->qk_bias', self._Override('dec_self_attention_bias')),
        # Decoder _AddRelativeBias always has bidirectional=False.
        ('qk_bias,segment_pos,key_segment_pos,relative_bias_weights->qhk_bias',
         self._Fn('relative_bias', fn=self._AddRelativeBias)),
        ('q,k_full,v_full,qhk_bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('h1->h1_act', self._Fn('h1_act', fn=activation_fn)),
        ('h1_act->h1_conv', optional_conv_layer),
        ('o,h1_conv,h2,wo1,wo2->outputs', self._Fn('outputs', fn=_Output))
    ]
    return self._Graph(name, graph_inputs, graph_outputs, *sub_layers)

  def _CapLogits(self, logits):
    """When enabled, cap logits by p.atten_logit_cap with tanh."""
    p = self.params
    if not p.atten_logit_cap or p.atten_logit_cap <= 0.:
      return logits
    cap = tf.cast(p.atten_logit_cap, logits.dtype)
    # Note that since this caps the negative side as well, caller
    # must defer the pad-with-very-negative-logits logic to after
    # this function returns.
    logits = cap * tf.math.tanh(logits / cap)
    return logits

  def Attention(self, name):
    """Attention with multiple attention heads."""
    p = self.params

    def _AddBiasF32(logits, bias):
      # logits: BLHM [batch, length, heads, memory_length]
      # expecting logits.dtype == tf.float32
      #
      # bias: BLHM [batch, length, heads, memory_length]
      #       (in case of attention with relative bias) OR
      #
      #       BLM  [batch, length, memory_length]
      #       (default masking bias with very negative logits).
      bias = tf.cast(bias, logits.dtype)
      if bias.shape.ndims == 3:
        # Expanding the 'heads' dimension
        retval = logits + tf.expand_dims(bias, 2)
      else:
        assert bias.shape.ndims == 4
        retval = logits + bias
      return retval

    def _ReduceLogsumexp(x):
      max_logit = tf.math.reduce_max(
          tf.stop_gradient(x), axis=-1, keepdims=True)
      extra_logit = p.attention_extra_logit
      if extra_logit is not None:
        extra_logit = tf.convert_to_tensor(extra_logit, max_logit.dtype)
        max_logit = tf.math.maximum(max_logit, extra_logit)
      x -= max_logit
      exp_x = tf.math.exp(x)
      sum_exp_x = tf.math.reduce_sum(exp_x, axis=-1, keepdims=True)
      if extra_logit is not None:
        sum_exp_x += tf.math.exp(extra_logit - max_logit)
      return tf.math.log(sum_exp_x) + max_logit

    def _LogSoftmax(x):
      return x - _ReduceLogsumexp(x)

    def _SoftmaxF32(x):
      # expecting x.dtype == tf.float32
      #
      # TODO(lepikhin): consider
      # if p.attention_extra_logit is None:
      #   return tf.nn.softmax(x)
      softmax = tf.math.exp(_LogSoftmax(x))
      softmax = tf.cast(softmax, py_utils.FPropDtype(self.params))
      return softmax

    # p.attention_num_memory_heads == 1 special case is simple, we omit the
    # "heads" dimension H from the projection matrices wk and wv.
    #
    # Then remove the heads dimension H of wk, vv, k, or v in any einsum
    # formula in which it appears.
    def _LogitsFnF32(q, k):
      # logits.dtype == tf.float32 leads to better training stability
      if p.attention_logits_dtype is not None:
        q = tf.cast(q, p.attention_logits_dtype)
        k = tf.cast(k, p.attention_logits_dtype)
      if p.attention_num_memory_heads == 1:
        return tf.einsum('BLHD,BMD->BLHM', q, tf.squeeze(k, -2))
      assert p.attention_num_memory_heads is None, p.attention_num_memory_heads
      logits = tf.einsum('BLHD,BMHD->BLHM', q, k, name='attention_logits')
      return self._CapLogits(logits)

    def _OutputsFn(weights, v):
      if p.attention_num_memory_heads == 1:
        return tf.einsum('BLHM,BMD->BLHD', weights, tf.squeeze(v, -2))
      assert p.attention_num_memory_heads is None, p.attention_num_memory_heads
      return tf.einsum('BLHM,BMHD->BLHD', weights, v, name='attention_output')

    kv_split = p.qkv_split if p.attention_num_memory_heads is None else None

    return self._Graph(
        name, ['_q', '_k', '_v', 'bias'], ['outputs'],
        ('_q->q', self.MeshSplit('_q', p.qkv_split)),
        ('_k->k', self.MeshSplit('_k', kv_split)),
        ('_v->v', self.MeshSplit('_v', kv_split)),
        ('q,k->l', self._Fn('logits', fn=_LogitsFnF32)),
        ('l,bias->logits', self._Fn('bias', fn=_AddBiasF32)),
        ('logits->w', self._Fn('weights', _SoftmaxF32)),
        ('w->weights',
         self._Dropout('dropout', 1 - self.params.attention_dropout_prob)),
        ('weights->weights_split', self.MeshSplit('_wsplit', p.qkv_split)),
        ('weights_split,v->outputs_unsplitted',
         self._Fn('outputs', fn=_OutputsFn)),
        ('outputs_unsplitted->outputs', self.MeshSplit('_o', p.qkv_split)))

  def _ComputeQKV(self, name):
    p = self.params

    def _Compute(x, w):
      if p.attention_combine_dims:
        w = tf.reshape(
            w, [p.model_dim, p.attention_num_heads, p.attention_key_value_dim])
      w = self._MeshSplit(w, p.mhd_w_split)
      w = self._ReshapeM(w, 0)
      return self._EinsumWithModelDim('BLM,MHD->BLHD', x, w)

    return self._Fn(name, _Compute)

  def _ComputeQKVCombine(self, name):
    p = self.params

    def _Compute(x, wq, wk, wv):

      def _GetW(w, h):
        if p.attention_combine_dims:
          combined_split = None if p.mhd_w_split is None else p.mhd_w_split[:-1]
          w = self._MeshSplit(w, combined_split)
          w = tf.reshape(w, [p.model_dim, h, p.attention_key_value_dim])
        w = self._MeshSplit(w, p.mhd_w_split)
        return self._ReshapeM(w, 0)

      wq = _GetW(wq, p.attention_num_heads)
      wk = _GetW(wk, p.attention_num_memory_heads or p.attention_num_heads)
      wv = _GetW(wv, p.attention_num_memory_heads or p.attention_num_heads)
      wc = [wq, wk, wv]

      if ((p.attention_num_memory_heads and
           p.attention_num_heads != p.attention_num_memory_heads) or
          not p.attention_combine_qkv):
        # Combined tf.einsum is not possible, falling back to individual
        # einsum ops.
        return [self._EinsumWithModelDim('BLM,MHD->BLHD', x, w) for w in wc]

      wc = [tf.expand_dims(w, 0) for w in [wq, wk, wv]]
      wc = tf.concat(wc, 0)
      return [
          tf.squeeze(y, 0) for y in tf.split(
              self._EinsumWithModelDim('BLM,KMHD->KBLHD', x, wc), 3, 0)
      ]

    return self._Fn(name, _Compute)

  def _ComputeQKVH(self, name, hidden_dim_reshape_segments=4):
    p = self.params

    def _Compute(x, wq, wk, wv, wi0, wi1):

      def _GetW(w, h=0):
        if p.attention_num_heads > 1 and h == 1:
          return tf.reshape(w, self._ReshapedModelDims() + [h, -1])
        else:
          return tf.reshape(
              w,
              self._ReshapedModelDims() +
              [hidden_dim_reshape_segments, -1, p.attention_key_value_dim])

      wq = _GetW(wq, p.attention_num_heads)
      wk = _GetW(wk, p.attention_num_memory_heads or p.attention_num_heads)
      wv = _GetW(wv, p.attention_num_memory_heads or p.attention_num_heads)
      wi0 = _GetW(wi0)
      wi1 = _GetW(wi1)

      wc = [wq, wk, wv, wi0, wi1]

      if (p.attention_num_memory_heads and
          p.attention_num_heads != p.attention_num_memory_heads):
        # Combined tf.einsum is not possible, falling back to individual
        # einsum ops.
        k = self._EinsumWithModelDim('BLM,MHD->BLHD', x, wk)
        v = self._EinsumWithModelDim('BLM,MHD->BLHD', x, wv)
        wc = tf.concat([wq, wi0, wi1], -2)
        splits = [wq.shape.as_list()[-2]] + [wi0.shape.as_list()[-2]] * 2
        r = self._MeshSplit(
            self._EinsumWithModelDim('BLM,MSHD->BLSHD', x, wc),
            p.blh_split + [-1, -1])
        q, f1, f2 = tf.split(r, splits, -2)
        q = self._MeshSplit(
            tf.reshape(q,
                       q.shape.as_list()[:2] + [-1, q.shape[-1]]), p.qkv_split)

        # Only one head so there is nothing to shard the H in BLHD k,v tensors.
        kv_split = p.qkv_split[:2] + [-1, p.qkv_split[-1]]
        k = self._MeshSplit(k, kv_split)
        v = self._MeshSplit(v, kv_split)
        return q, k, v, f1, f2
      wc = tf.concat([wq, wk, wv, wi0, wi1], -2)
      splits = [wq.shape.as_list()[-2]] * 3 + [wi0.shape.as_list()[-2]] * 2
      ret = tf.split(
          self._EinsumWithModelDim('BLM,MSHD->BLSHD', x, wc), splits, -2)

      def _MeshSplitQKV(x):
        return tf.reshape(x, x.shape.as_list()[:2] + [-1, x.shape[-1]])

      return [_MeshSplitQKV(x) for x in ret[:3]] + ret[3:]

    return self._Fn(name, _Compute)

  def _ComputeAttenOutputs(self, o, wo):
    p = self.params
    hdm_split = self._attention_output_hdm_w_split
    if p.attention_combine_dims:
      wo = tf.reshape(
          wo, [p.attention_num_heads, p.attention_key_value_dim, p.model_dim])
    wo = self._MeshSplit(wo, hdm_split)
    wo = self._ReshapeM(wo, 2)
    return self._MeshSplit(
        self._EinsumWithModelDim('HDM,BLHD->BLM', wo, o),
        self._AdjustMSplit(p.blm_split, 2))

  def _ComputeCombinedOutputs(self,
                              o1,
                              o2,
                              wo1,
                              wo2,
                              hidden_dim_reshape_segments=4):
    p = self.params
    split_h_shape = [hidden_dim_reshape_segments, -1, p.attention_key_value_dim]

    wo1, wo2 = [
        tf.reshape(w, split_h_shape + self._ReshapedModelDims())
        for w in [wo1, wo2]
    ]

    o1 = tf.reshape(o1, o1.shape.as_list()[:2] + split_h_shape)
    o = self._MeshSplit(tf.concat([o1, o2], -2), p.blh_split + [-1, -1])
    wo = tf.concat([wo1, wo2], 1)
    return self._EinsumWithModelDim('SHDM,BLSHD->BLM', wo, o) * (2.0**-0.5)

  def DenseReluDense(self, name, decoder=False, activation='relu'):
    if decoder:
      input_endpoints = self._DecoderLayerInMapKeys
    else:
      input_endpoints = self._EncoderLayerInMapKeys

    if activation == 'relu':
      activation_fn = tf.nn.relu
    elif activation == 'gelu':
      activation_fn = lambda x: tf.nn.gelu(x, approximate=True)
    # BEGIN GOOGLE-INTERNAL
    # TODO(davidso): Make this public when Primer paper is released.
    elif activation == 'sqr_relu':
      activation_fn = lambda x: tf.math.square(tf.nn.relu(x))
    # END GOOGLE-INTERNAL
    else:
      raise ValueError('Activation %s not supported.' % activation)

    p = self.params
    # Note that dropout is used here, but not in the MoE layer by default.
    return self._Graph(
        name,
        input_endpoints,
        ['outputs', 'aux_loss'],
        ('->wi,wo',
         self._DenseReluDenseWeights('w', self._device_mesh, p.mh_wi_split,
                                     p.hm_wo_split)),
        ('wi->wi_reshaped',
         self._Fn('wi_reshape', fn=lambda x: self._ReshapeM(x, 0))),
        ('wo->wo_reshaped',
         self._Fn('wo_reshape', fn=lambda x: self._ReshapeM(x, 1))),
        ('wi_reshaped,vec->h', self.EinsumWithModelDim('wi', 'MH,BLM->BLH')),
        ('h->h_split', self.MeshSplit('_h_split', p.blh_split)),
        ('h_split->h_%s' % activation, self._Fn(activation, activation_fn)),
        ('h_%s->h_dropout' % activation,
         self._Dropout('input_dropout', 1 - p.dropout_rate)),
        ('wo_reshaped,h_dropout->outputs_pre_split',
         self.EinsumWithModelDim('wo', 'HM,BLH->BLM')),
        ('outputs_pre_split->outputs',
         self.MeshSplit('outputs_split', self._AdjustMSplit(p.blm_split, 2))),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
    )

  def DenseReluDenseGated(self, name, activation_fn, decoder=False):
    if decoder:
      input_endpoints = self._DecoderLayerInMapKeys
    else:
      input_endpoints = self._EncoderLayerInMapKeys

    def _Impl(wi_0, wi_1, inputs):
      wi = tf.concat([tf.expand_dims(wi_0, 0), tf.expand_dims(wi_1, 0)], 0)
      o1, o2 = [
          tf.squeeze(o, 0) for o in tf.split(
              self._EinsumWithModelDim('KMH,BLM->KBLH', wi, inputs), 2, 0)
      ]
      # To match historic behavior use approximate=True with tf.nn.gelu
      # activation.
      return tf.math.multiply(activation_fn(o1), o2)

    p = self.params
    return self._Graph(
        name,
        input_endpoints,
        ['outputs', 'aux_loss'],
        ('->wi_0,wi_1,wo',
         self._DenseReluDenseWeightsGatedGELU('w', self._device_mesh,
                                              p.mh_wi_split, p.hm_wo_split)),
        ('wi_0->wi_0_reshaped',
         self._Fn('wi0_reshape', fn=lambda x: self._ReshapeM(x, 0))),
        ('wi_1->wi_1_reshaped',
         self._Fn('wi1_reshape', fn=lambda x: self._ReshapeM(x, 0))),
        ('wo->wo_reshaped',
         self._Fn('wo_reshape', fn=lambda x: self._ReshapeM(x, 1))),
        ('wi_0_reshaped,wi_1_reshaped,vec->h', self._Fn('wi', fn=_Impl)),
        ('h->h_split', self.MeshSplit('_h_split', p.blh_split)),
        ('h_split->h_dropout',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('wo_reshaped,h_dropout->outputs_pre_split',
         self.EinsumWithModelDim('wo', 'HM,BLH->BLM')),
        ('outputs_pre_split->outputs',
         self.MeshSplit('outputs_split', self._AdjustMSplit(p.blm_split, 2))),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
    )

  def _MoEWeightsGatedGELU(self,
                           name,
                           device_mesh=None,
                           wi_mesh_split=None,
                           wo_mesh_split=None):
    # Gated GELU.  There are two separate linear transformations applied in
    # parallel to the inputs.  You take the gelu of one of them and then
    # multiply the two componentwise.
    p = self.params
    return self._ShardedVar(
        name=name,
        weights=[('wi_0',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / p.model_dim)**0.5) * 3.0**0.5)),
                      dtype=p.dtype,
                      shape=[p.e_dim, p.model_dim, p.moe_hidden_dim],
                      tensor_split_dims_mapping=p.emh_split)),
                 ('wi_1',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / p.model_dim)**0.5) * 3.0**0.5)),
                      dtype=p.dtype,
                      shape=[p.e_dim, p.model_dim, p.moe_hidden_dim],
                      tensor_split_dims_mapping=p.emh_split)),
                 ('wo',
                  gshard_layers.ShardedWeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / p.moe_hidden_dim)**0.5) * 3.0**0.5)),
                      dtype=p.dtype,
                      shape=[p.e_dim, p.moe_hidden_dim, p.model_dim],
                      tensor_split_dims_mapping=p.ehm_split))],
        device_mesh=device_mesh)

  def _ShardedMoEGLU(self, name):
    """Simple MoE GLU with xla_sharding."""
    p = self.params
    num_groups = p.num_groups or p.num_devices

    reshape_input = gshard_layers.ReshapeInputLayer.Params().Set(
        num_groups=num_groups,
        num_devices=p.num_devices,
        model_dims=self._ReshapedModelDims())

    def _GLUApplyGating(gating, inputs, reshaped_inputs, wi_0, wi_1, wo):
      wi = tf.concat([tf.expand_dims(wi_0, 0), tf.expand_dims(wi_1, 0)], 0)
      return gshard_layers.FeedForwardNetworksApplyGating(
          gating,
          inputs,
          reshaped_inputs,
          wi,
          wo,
          num_devices=p.num_devices,
          num_groups=p.num_groups or p.num_devices,
          dropout_rate=p.moe_dropout_rate,
          device_mesh=self._device_mesh,
          model_dim_reshape_segments=self._model_dim_reshape_segments,
          gsm_split=self._AdjustMSplitByName('blm_split'),
          egcm_split=self._AdjustMSplitByName('egcm_split'),
          gecm_split=self._AdjustMSplitByName('gecm_split'),
          gsec_split=self._AdjustMSplitByName('gsec_split'),
          eah_split=self._AdjustMSplitByName('eah_split'),
          eam_split=self._AdjustMSplitByName('eam_split'),
          use_glu=True,
          activation_name='GELU_APPROXIMATE')

    return self._Graph(
        name, ['inputs', 'segment_id', 'wi_0', 'wi_1', 'wo'],
        ['outputs', 'aux_loss'],
        ('inputs,segment_id->reshaped_inputs, paddings', reshape_input),
        ('->gw', self._Top2GatingWeights('top_2_gating')),
        ('gw->gw_reshaped',
         self._Fn('reshape_gw', fn=lambda x: self._ReshapeM(x, 0))),
        ('wi_0->wi0_reshaped',
         self._Fn('reshape_wi0', fn=lambda x: self._ReshapeM(x, 1))),
        ('wi_1->wi1_reshaped',
         self._Fn('reshape_wi1', fn=lambda x: self._ReshapeM(x, 1))),
        ('wo->wo_reshaped',
         self._Fn('reshape_wo', fn=lambda x: self._ReshapeM(x, 2))),
        ('gw_reshaped,reshaped_inputs,paddings->gating',
         self._ComputeTopKGating('compute_gating')),
        ('gating,inputs,reshaped_inputs,wi0_reshaped,wi1_reshaped,wo_reshaped'
         '->outputs,aux_loss', self._Fn('process_gating', _GLUApplyGating)))

  def MoEGated(self, name, decoder=False):
    """Returns a MoE layer with GLU experts."""
    if decoder:
      input_endpoints = self._DecoderLayerInMapKeys
    else:
      input_endpoints = self._EncoderLayerInMapKeys

    return self._Graph(
        name, input_endpoints, ['outputs', 'aux_loss'],
        ('vec->input_split', self.Split('input_split')),
        ('segment_id->segment_id_split', self.Split('segment_id_split')),
        ('->wi_0,wi_1,wo', self._MoEWeightsGatedGELU(name)),
        ('input_split,segment_id_split,wi_0,wi_1,wo'
         '->outputs_pre_split,aux_loss', self._ShardedMoEGLU('ffw')),
        ('outputs_pre_split->outputs', self.Split('outputs_split')))


class RecurrentDenseBuilderParallelDecode(DenseBuilder):
  """Same as RecurrentDenseBuilder but with micro variables.

  Projection variables created under this builder will be replaced by multiple
  micro variables, each with shape [model_dim, proj_weight_hdim, d_kv].
  For example, the original wi_o with shape [1024, 4096] was replaced with
  8 micro variables each with  shape [1024, 4, 128] when proj_weight_hdim = 4
  and d_kv = 128.

  Using smaller micro variables reduced the total size of variables created by
  RepeatLayer. For example, when the number of layers is 120, model_dim = 16k,
  ff_dim = 64k, d_kv = 128, then RepeatLayer will create variables with size
  120 * 16k * 64k * 4Bytes = 491GB, exceeding the host memoery limit.
  By setting proj_weight_hdim = 64, RepeatLayer will create 8x micro variables,
  each with shape [L, 16k, 64, 128] only. Using leasting loading placer those
  micro variables will be placed at different hosts.

  To disable micro variables, set proj_weight_hdim to None.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'proj_weight_hdim', 64,
        'weight hd_dims = [proj_weight_hdim, attention_key_value_dim]. '
        'Use None to disable micro variables.')
    p.deterministic_dropout = True
    return p

  @property
  def _num_input_proj_weights(self):
    p = self.params
    if p.proj_weight_hdim is None:
      return 1, 1, 1
    num_wq = p.attention_num_heads // p.proj_weight_hdim
    kv_h = p.attention_num_memory_heads or p.attention_num_heads
    if kv_h < p.proj_weight_hdim:
      num_wkv = 1
    else:
      num_wkv = kv_h // p.proj_weight_hdim
    num_wi = p.ff_dim // p.attention_key_value_dim // p.proj_weight_hdim
    return num_wq, num_wkv, num_wi

  @property
  def _num_output_proj_weights(self):
    p = self.params
    if p.proj_weight_hdim is None:
      return 1, 1
    num_atten = p.attention_num_heads // p.proj_weight_hdim
    num_wo = p.ff_dim // p.attention_key_value_dim // p.proj_weight_hdim
    return num_atten, num_wo

  @property
  def _ffw_var_h_dim_size(self):
    p = self.params
    assert p.ff_dim % p.attention_key_value_dim == 0
    if p.proj_weight_hdim is None:
      return p.ff_dim
    assert (p.ff_dim // p.attention_key_value_dim) % p.proj_weight_hdim == 0
    return p.proj_weight_hdim * p.attention_key_value_dim

  @property
  def _atten_q_var_heads(self):
    p = self.params
    if p.proj_weight_hdim is None:
      return p.attention_num_heads
    assert p.attention_num_heads % p.proj_weight_hdim == 0
    return p.proj_weight_hdim

  @property
  def _atten_kv_var_heads(self):
    p = self.params
    kv_h = p.attention_num_memory_heads or p.attention_num_heads
    if p.proj_weight_hdim is None or kv_h < p.proj_weight_hdim:
      return kv_h
    assert kv_h % p.proj_weight_hdim == 0
    return p.proj_weight_hdim

  def _DecoderLayerInputProjWeights(self, name):
    # Create multiple input projection weights, each of which has shape
    # [model_dim, proj_weight_hdim, d_kv].
    p = self.params
    input_w_split = [p.mhd_w_split[0], max(p.mhd_w_split[1], p.mhd_w_split[2])]
    kv_w_split = input_w_split if p.kv_mhd_w_split is None else [
        p.kv_mhd_w_split[0],
        max(p.kv_mhd_w_split[1], p.kv_mhd_w_split[2])
    ]

    q_stddev = (p.model_dim * p.attention_key_value_dim)**-0.5
    wq_tpl = gshard_layers.ShardedWeightParams(
        shape=[
            p.model_dim, self._atten_q_var_heads * p.attention_key_value_dim
        ],
        dtype=p.dtype,
        init=py_utils.WeightInit.Gaussian(q_stddev),
        tensor_split_dims_mapping=input_w_split)

    kv_stddev = (p.model_dim)**-0.5
    wkv_tpl = gshard_layers.ShardedWeightParams(
        shape=[
            p.model_dim, self._atten_kv_var_heads * p.attention_key_value_dim
        ],
        dtype=p.dtype,
        init=py_utils.WeightInit.Gaussian(kv_stddev),
        tensor_split_dims_mapping=kv_w_split)

    ffw_tpl = gshard_layers.ShardedWeightParams(
        init=py_utils.WeightInit.Uniform(((1. / p.model_dim)**0.5) * 3.0**0.5),
        dtype=p.dtype,
        shape=[p.model_dim, self._ffw_var_h_dim_size],
        tensor_split_dims_mapping=input_w_split)

    num_wq, num_wkv, num_wi = self._num_input_proj_weights

    weights_params = []
    for i in range(num_wq):
      weights_params.append(('wq_%d' % i, wq_tpl))
    for i in range(num_wkv):
      weights_params.append(('wk_%d' % i, wkv_tpl))
    for i in range(num_wkv):
      weights_params.append(('wv_%d' % i, wkv_tpl))
    for i in range(num_wi):
      weights_params.append(('wi0_%d' % i, ffw_tpl))
    for i in range(num_wi):
      weights_params.append(('wi1_%d' % i, ffw_tpl))
    return self._ShardedVar(
        name=name, weights=weights_params, device_mesh=self._device_mesh)

  def _DecoderLayerOutputProjWeights(self, name):
    p = self.params
    out_w_split = [max(p.mhd_w_split[1], p.mhd_w_split[2]), p.mhd_w_split[0]]

    atten_stddev = (p.attention_num_heads * p.attention_key_value_dim)**-0.5
    atten_tpl = gshard_layers.ShardedWeightParams(
        shape=[
            self._atten_q_var_heads * p.attention_key_value_dim, p.model_dim
        ],
        dtype=p.dtype,
        init=py_utils.WeightInit.Gaussian(atten_stddev),
        tensor_split_dims_mapping=out_w_split)
    ffw_tpl = gshard_layers.ShardedWeightParams(
        init=py_utils.WeightInit.Uniform(((1. / p.ff_dim)**0.5) * 3.0**0.5),
        dtype=p.dtype,
        shape=[self._ffw_var_h_dim_size, p.model_dim],
        tensor_split_dims_mapping=out_w_split)
    num_atten, num_wo = self._num_output_proj_weights
    weights_params = []
    for i in range(num_atten):
      weights_params.append(('atten_wo_%d' % i, atten_tpl))
    for i in range(num_wo):
      weights_params.append(('ffw_wo_%d' % i, ffw_tpl))
    return self._ShardedVar(
        name=name, weights=weights_params, device_mesh=self._device_mesh)

  def _ComputeQKVH(self, name, hidden_dim_reshape_segments=4):
    p = self.params
    d_kv = p.attention_key_value_dim

    def _Compute(x, *ws):
      num_wq, num_wkv, num_wi = self._num_input_proj_weights

      def _ReshapeW(w, h=0):
        if p.attention_num_heads > 1 and h == 1:
          return tf.reshape(w, self._ReshapedModelDims() + [h, -1])
        else:
          return tf.reshape(
              w,
              self._ReshapedModelDims() +
              [hidden_dim_reshape_segments, -1, d_kv])

      wqs = ws[:num_wq]
      wks = ws[num_wq:num_wq + num_wkv]
      wvs = ws[num_wq + num_wkv:num_wq + 2 * num_wkv]
      wi0s = ws[num_wq + 2 * num_wkv:num_wq + 2 * num_wkv + num_wi]
      wi1s = ws[num_wq + 2 * num_wkv + num_wi:]

      if (p.attention_num_memory_heads and
          p.attention_num_heads != p.attention_num_memory_heads):
        # Combined tf.einsum is not possible, falling back to individual
        # einsum ops.
        h_kv = p.attention_num_memory_heads or p.attention_num_heads
        wk = tf.concat([_ReshapeW(w, h_kv) for w in wks], -2)
        wv = tf.concat([_ReshapeW(w, h_kv) for w in wvs], -2)
        k = self._EinsumWithModelDim('BLM,MHD->BLHD', x, wk)
        v = self._EinsumWithModelDim('BLM,MHD->BLHD', x, wv)
        wc = tf.concat([_ReshapeW(w) for w in wqs + wi0s + wi1s], -2)
        splits = [p.attention_num_heads // hidden_dim_reshape_segments
                 ] + [p.ff_dim // d_kv // hidden_dim_reshape_segments] * 2
        r = self._MeshSplit(
            self._EinsumWithModelDim('BLM,MSHD->BLSHD', x, wc),
            p.blh_split + [-1, -1])
        q, f1, f2 = tf.split(r, splits, -2)
        f1 = self._MeshSplit(f1, p.blh_split + [-1, -1])
        f2 = self._MeshSplit(f2, p.blh_split + [-1, -1])
        q = self._MeshSplit(
            tf.reshape(q,
                       q.shape.as_list()[:2] + [-1, q.shape[-1]]), p.qkv_split)

        # Only one head so there is nothing to shard the H in BLHD k,v tensors.
        kv_split = p.qkv_split[:2] + [-1, p.qkv_split[-1]]
        k = self._MeshSplit(k, kv_split)
        v = self._MeshSplit(v, kv_split)
        return q, k, v, f1, f2
      wc = tf.concat([_ReshapeW(w) for w in ws], -2)
      splits = [p.attention_num_heads // hidden_dim_reshape_segments
               ] * 3 + [p.ff_dim // d_kv // hidden_dim_reshape_segments] * 2
      ret = tf.split(
          self._EinsumWithModelDim('BLM,MSHD->BLSHD', x, wc), splits, -2)

      def _MeshSplitQKV(x):
        return tf.reshape(x, x.shape.as_list()[:2] + [-1, x.shape[-1]])

      return [_MeshSplitQKV(x) for x in ret[:3]] + ret[3:]

    return self._Fn(name, _Compute)

  def DecoderLayer(self,
                   name,
                   activation_fn,
                   conv_kernel_size=None,
                   hidden_dim_reshape_segments=4,
                   norm_type='ln',
                   norm_policy='pre'):
    p = self.params

    collections = None
    if p.relative_attention_type == 'bias_shared':
      collections = ['_dec_self_attention_shared_var']
    else:
      assert p.relative_attention_type == 'bias', p.relative_attention_type

    def _ComputeBias(segment_id, segment_pos):
      return self._DecNotVisible(segment_id, segment_pos) * (-1e+09)

    state_shape = [None, None, p.attention_num_heads, p.attention_key_value_dim]

    if conv_kernel_size is not None:
      norm_layer = self._LNConv('ln', conv_kernel_size)
      d = p.ff_dim // hidden_dim_reshape_segments // p.attention_key_value_dim
      model_dims = [hidden_dim_reshape_segments, d, p.attention_key_value_dim]
      optional_conv_layer = self.DepthwiseConvAutoregressive(
          name='conv', kernel_size=conv_kernel_size, model_dims=model_dims)
    elif norm_type == 'ln':
      norm_layer = self._LN('ln')
      optional_conv_layer = self._Identity('conv')
    elif norm_type == 'true_ln':
      norm_layer = self._TrueLN('true_ln')
      optional_conv_layer = self._Identity('conv')
    # BEGIN GOOGLE-INTERNAL
    # TODO(davidso): Make this public once the Primer paper is released.
    elif norm_type == 'pn':
      norm_layer = self._PN('pn')
      optional_conv_layer = self._Identity('conv')
    # END GOOGLE-INTERNAL
    else:
      raise ValueError('Norm type %s not supported' % norm_type)
    num_q, num_kv, num_wi = self._num_input_proj_weights
    num_weights = num_q + 2 * num_kv + 2 * num_wi
    wi_str = ','.join(['wi_%d' % i for i in range(num_weights)])
    num_atten, num_wo = self._num_output_proj_weights
    num_weights = num_atten + num_wo
    wo_str = ','.join(['wo_%d' % i for i in range(num_weights)])

    def _Outputs(o_atten, o_ffw, *ws):
      h_shape = [hidden_dim_reshape_segments, -1, p.attention_key_value_dim]
      wo = tf.concat(
          [tf.reshape(w, h_shape + self._ReshapedModelDims()) for w in ws], 1)

      o_atten = tf.reshape(o_atten, o_atten.shape.as_list()[:2] + h_shape)
      o = self._MeshSplit(
          tf.concat([o_atten, o_ffw], -2), p.blh_split + [-1, -1])
      return self._EinsumWithModelDim('SHDM,BLSHD->BLM', wo, o) * (2.0**-0.5)

    if norm_policy != 'pre':
      raise ValueError('Normalization policy %s not supported' % norm_policy)

    return self._Graph(
        name,
        ['i'],
        ['o'],
        ('i.vec,i.segment_id->input_masked', self.Mask()),
        ('i.segment_id->o.segment_id', self._Identity('segment_id')),
        ('i.segment_pos->o.segment_pos', self._Identity('segment_pos')),
        ('input_masked->x', norm_layer),
        ('->%s' % wi_str, self._DecoderLayerInputProjWeights('get_w_in')),
        ('->relative_bias_ws',
         self._RelativeAttentionBiasWeights('wrb', collections)),
        ('x,%s->q,k,v,h1,h2' % wi_str,
         self._ComputeQKVH('qkvh', hidden_dim_reshape_segments)),
        ('k->k_full', self._AttentionState('k_state', state_shape)),
        ('v->v_full', self._AttentionState('v_state', state_shape)),
        ('i.segment_pos->key_segment_pos',
         self._AttentionState('seg_pos', [None, None], dtype=tf.int32)),
        ('i.segment_id,i.segment_pos->qq_bias', self._Fn(
            'bias', fn=_ComputeBias)),
        ('qq_bias->qk_bias', self._Override('dec_self_attention_bias')),
        # Decoder _AddRelativeBias always has bidirectional=False.
        ('qk_bias,i.segment_pos,key_segment_pos,relative_bias_ws->qhk_bias',
         self._Fn('relative_bias', fn=self._AddRelativeBias)),
        ('q,k_full,v_full,qhk_bias->o_atten', self.Attention('attention')),
        ('h1->h1_act', self._Fn('h1_act', fn=activation_fn)),
        ('h1_act->h1_conv', optional_conv_layer),
        ('h1_conv,h2->o_ffw', self._Fn('h_ffw', fn=tf.math.multiply)),
        ('->%s' % wo_str, self._DecoderLayerOutputProjWeights('get_w_out')),
        ('o_atten,o_ffw,%s->y' % wo_str, self._Fn('compute_output', _Outputs)),
        ('y->y_dropout', self._Dropout('y_dropout', 1 - p.dropout_rate)),
        ('input_masked,y_dropout->o.vec', self._Add('add')))

  @property
  def _DecoderLayerInMapKeys(self):
    return ['vec', 'segment_id', 'segment_pos']

  def DecoderLayerStack(self,
                        name,
                        sub_layers,
                        num=1,
                        conv_kernel_size=None,
                        norm_type=None,
                        norm_policy=None):
    """DecoderLayerStack with self attention and feedforward in parallel."""
    del norm_type, norm_policy
    p = self.params
    assert p.deterministic_dropout
    stack = [
        ('i.vec->inputs_split',
         self.MeshSplit('inputs_split', self._AdjustMSplit(p.blm_split, 2))),
        ('i.segment_id->segment_id_split',
         self.MeshSplit('segment_id_split', p.blm_split[:2])),
        ('i.segment_pos->segment_pos_split',
         self.MeshSplit('segment_pos_split', p.blm_split[:2])),
        ('inputs_split->input_dropout',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('input_dropout,segment_id_split,segment_pos_split->sub_layer_imap',
         self._CreateNestedMap(name='map', keys=self._DecoderLayerInMapKeys)),
        ('sub_layer_imap->sub_layer_omap', sub_layers[0]),
        ('sub_layer_omap.vec->y_norm', self._LNNoScale('final_layer_norm')),
        ('y_norm->y_dropout',
         self._Dropout('outputs_dropout', 1 - self.params.dropout_rate)),
        ('i.aux_loss->o.aux_loss', self._Identity('id')),
        ('y_dropout,segment_id_split->o.vec', self.Mask()),
    ]
    return self._Graph(name, ['i'], ['o'], *stack)


class UniTransformer(base_model.BaseTask):
  """LM TransformerModel with z-loss."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('debug', False, 'If true, outfeed additional per-example tensor.')

    p.Define('builder', None, 'GShard Builder.')
    p.Define('vocab_size', None, 'Vocabulary size')
    p.Define('sequence_length', None, 'Sequence length.')
    p.Define(
        'max_length', 512,
        'Max sequence length. Second pos_emb Tensor dim is set to ' +
        'max_length.')
    p.Define('batch_size', None, 'Batch size. Unused.')
    p.Define('num_transformer_layers', None,
             'Number of blocks in builder.{Decoder,Encoder}LayerStack.')
    p.Define(
        'loss_denominator', 0, 'If positive, ignore the value of '
        'use_tgt_labels_size_as_loss_denominator and set the denominator '
        'directly.')
    p.Define(
        'use_tgt_labels_size_as_loss_denominator', True,
        'False to use total number of non-padding tokens instead of '
        'fixed tgt_labels tensor size.')

    p.Define('aux_loss_coef', 0.01, 'Multiplier for GShard aux_loss.')
    p.Define('label_smoothing', 0.1, 'Label smoothing.')
    p.Define('logits_abs_max', None, 'Logits clipping.')
    p.Define(
        'z_loss', 1e-4, 'if z_loss is nonzero, we add a loss equal to '
        'z_loss * tf.math.square(tf.math.reduce_logsumexp(logits, -1))')
    p.Define('positional_embedding', True, 'Positional embs.')
    p.Define('gated_gelu', False, 'FFN gated GELU. '
             'Deprecated. Use gated_ffn_activation=gelu.')
    p.Define('moe_gated_gelu', False, 'Use gated GELU for the MoE layer.')
    p.Define('gated_ffn_activation', None, 'Transformer gated FFN activation.')
    p.Define('parallel_ffn', False,
             'Whether to make ffn and attention parallel.')
    p.Define(
        'hidden_dim_reshape_segments', 4,
        'Size of S when reshaping hidden dimension H to Sh. Only used when'
        ' parallel_ffn is true currently.')
    p.Define('conv_kernel_size', None,
             'Optional 1D depthwise convolutional kernel.')
    p.Define(
        'use_entmax', False, 'Flag to use the entmax for entropy and loss '
        'calculation. The entmax is following this publication: '
        'https://arxiv.org/pdf/2004.02644.pdf.')
    p.Define('entmax_alpha', 1.5, 'Define the alpha value for the entmax '
             'calculation.')
    p.Define('entmax_n_iters', 50, 'Define the number of iterations for the '
             'entmax.')
    p.Define(
        'entmax_ensure_sum_one', True, 'Define the if the summation of '
        'the output probabilities should be 1.')
    p.Define(
        'use_per_layer_vars_for_recurrent', False,
        'Create per-layer variables for RecurrentDenseBuilderParallelDecode, '
        'instead of  combined variables [num_layers, ...].')
    p.Define('use_repeat_layer', False,
             'Whether to use RepeatLayer to wrap the layer stack.')
    p.Define(
        'num_spmd_pipeline_stages', 1,
        'If > 1, SPMD-shardable pipelining is used with this many stages.')
    p.Define('num_spmd_pipeline_microbatches', None,
             'Number of microbatches when num_spmd_pipeline_stages > 1.')
    p.Define(
        'moe', False,
        'True for Mixture-of-Experts, False for canonical Transformer model, ')
    p.Define('activation', 'relu', 'Transformer non-gated FFN activation.')
    p.Define('norm_type', 'ln', 'Type of normalization. Options are: ln.')
    p.Define('norm_policy', 'pre', 'Policy for applying normalization. '
             'Options are: pre.')
    # BEGIN GOOGLE-INTERNAL
    # TODO(davidso): Expand the options for norm_type and norm_policy when
    #                making this public.
    # TODO(davidso): Make this public after releasing the Primer paper.
    p.Define('multi_dconv_head_att', False,
             "Whether or not to use Primer's Mutli-Dconv-Head attention.")
    # END GOOGLE-INTERNAL
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if p.use_repeat_layer or p.num_spmd_pipeline_stages > 1:
      p.builder.deterministic_dropout = True
    assert p.num_transformer_layers % p.num_spmd_pipeline_stages == 0
    b = p.builder.Instantiate()

    tgt_vocab_size = p.vocab_size

    if callable(p.gated_ffn_activation):
      gated_ffn_activation = p.gated_ffn_activation
    elif p.gated_ffn_activation == 'silu':
      gated_ffn_activation = tf.nn.silu
    elif p.gated_gelu or p.gated_ffn_activation == 'gelu':
      gated_ffn_activation = lambda x: tf.nn.gelu(x, approximate=True)
    else:
      assert not p.gated_ffn_activation, p.gated_ffn_activation
      gated_ffn_activation = None

    dec_emb = b.Embedding('dec_emb', tgt_vocab_size)
    self.CreateChild('dec_emb', dec_emb)

    if p.positional_embedding:
      dec_pos_emb = b.Embedding('dec_pos_emb', p.max_length)
      self.CreateChild('dec_pos_emb', dec_pos_emb)

    if p.parallel_ffn:  # Only works with RecurrentDenseBuilderParallelDecode.
      assert not p.positional_embedding
      assert gated_ffn_activation
      assert isinstance(b, RecurrentDenseBuilderParallelDecode)
      decoder_sub_layers = [
          b.Repeat(
              name='blocks',
              body=b.DecoderLayer(
                  'block',
                  gated_ffn_activation,
                  p.conv_kernel_size,
                  p.hidden_dim_reshape_segments,
                  norm_type=p.norm_type,
                  norm_policy=p.norm_policy),
              repeat=p.num_transformer_layers,
              per_layer_vars=p.use_per_layer_vars_for_recurrent)
      ]
      dec = b.DecoderLayerStack(
          'decoder',
          decoder_sub_layers,
          1,
          conv_kernel_size=p.conv_kernel_size,
          norm_type=p.norm_type,
          norm_policy=p.norm_policy)
    else:
      if p.positional_embedding:
        atten_layer = b.DecSelfAttention('dec_self_attention')
      # BEGIN GOOGLE-INTERNAL
      # TODO(davidso): Make this public when the Primer paper is released.
      elif p.multi_dconv_head_att:
        atten_layer = b.DecMultiDconvHeadAttentionRelativeBias(
            'multi_dconv_head_att')
      # END GOOGLE-INTERNAL
      else:
        atten_layer = b.DecSelfAttentionRelativeBias('dec_self_attention')
      if gated_ffn_activation is None:
        ffw_layer = b.DenseReluDense(
            'dense_relu_dense', decoder=True, activation=p.activation)
      else:
        ffw_layer = b.DenseReluDenseGated(
            'dense_relu_dense', gated_ffn_activation, decoder=True)
      if p.moe:
        if p.moe_gated_gelu:
          moe_layer = b.MoEGated('moe', decoder=True)
        else:
          moe_layer = b.MoE('moe', decoder=True)
        decoder_sub_layers = [atten_layer, moe_layer, atten_layer, ffw_layer]
        num_decoder_layers = p.num_transformer_layers // 2
      else:
        decoder_sub_layers = [atten_layer, ffw_layer]
        num_decoder_layers = p.num_transformer_layers
      dec = b.DecoderLayerStack(
          'decoder',
          decoder_sub_layers,
          num_decoder_layers,
          conv_kernel_size=p.conv_kernel_size,
          norm_type=p.norm_type,
          norm_policy=p.norm_policy,
          use_repeat_layer=p.use_repeat_layer,
          spmd_pipeline_stages=p.num_spmd_pipeline_stages,
          spmd_pipeline_microbatches=p.num_spmd_pipeline_microbatches)
    dec.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

    emb_w_split = b.MeshSplit('w_split', b.params.emb_w_split)
    dec_out_split = b.MeshSplit('dec_out_split',
                                b._AdjustMSplit(b.params.blm_split[-3:], 2))
    logits_split = b.MeshSplit('logits_split', b.params.logits_split)

    self.CreateChild('dec', dec)
    self.CreateChild('emb_w_split', emb_w_split)
    self.CreateChild('dec_out_split', dec_out_split)
    self.CreateChild('logits_split', logits_split)

  def _ComputeDecoderInput(self, theta, input_batch):
    y = self.dec_emb.FProp(theta.dec_emb, input_batch.tgt.ids)
    if self.params.positional_embedding:
      y += self.dec_pos_emb.FProp(theta.dec_pos_emb,
                                  input_batch.tgt.segment_pos)
    return py_utils.NestedMap(
        vec=y,
        segment_id=input_batch.tgt.segment_ids,
        segment_pos=input_batch.tgt.segment_pos,
        encoder_output=tf.zeros_like(y),
        encoder_segment_id=tf.zeros_like(input_batch.tgt.segment_ids),
        encoder_segment_pos=tf.zeros_like(input_batch.tgt.segment_pos),
        aux_loss=tf.convert_to_tensor(0.0, py_utils.FPropDtype(self.params)))

  def ComputePredictions(self, theta, input_batch):
    """Forward propagation through one tower of the model.

    Args:
      theta: A `.NestedMap` object containing variable values of this task
        copied to this tower's devices.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      A dict containing metrics pairs.
    """
    p = self.params

    with tf.name_scope(p.name):
      decoder_input = self._ComputeDecoderInput(theta, input_batch)
      all_outputs = self.dec.FProp(theta.dec, decoder_input)
      dec_outputs, aux_loss = all_outputs.vec, all_outputs.aux_loss
      dec_outputs *= (p.builder.model_dim**-0.5)
      dec_outputs = self.dec_out_split.FProp(theta.dec_out_split, dec_outputs)
      # TODO(lepikhin): we only support
      # shared_embedding_and_softmax_weights=True at the moment.
      softmax_weights = self.vars.dec_emb.w.embedding.read_value()
      softmax_weights = self.emb_w_split.FProp(theta.emb_w_split,
                                               softmax_weights)
      if dec_outputs.dtype != softmax_weights.dtype:
        # to enable fprop_dtype = tf.bfloat16
        softmax_weights = tf.cast(softmax_weights, dec_outputs.dtype)
      if p.builder.model_dim_reshape_segments is not None:
        dec_outputs = tf.reshape(
            dec_outputs, [dec_outputs.shape[0], dec_outputs.shape[1], -1])
      logits = tf.einsum('BLM,VM->BLV', dec_outputs, softmax_weights)
      logits = self.logits_split.FProp(theta.logits_split, logits)

      if p.logits_abs_max is not None:
        logits = py_utils.clip_by_value(logits, -p.logits_abs_max,
                                        p.logits_abs_max)
      logits = self.logits_split.FProp(theta.logits_split, logits)
      return logits, aux_loss

  def _ComputeNonPadding(self, input_batch):
    if 'paddings' in input_batch.tgt:
      return tf.cast(1.0 - input_batch.tgt.paddings,
                     py_utils.FPropDtype(self.params))

    non_padding = tf.cast(
        tf.not_equal(input_batch.tgt.segment_ids, 0),
        py_utils.FPropDtype(self.params))

    # Negative target labels now indicate tokens that are to be used as
    # autoregressive inputs, but not counted in the loss.
    non_padding *= tf.cast(
        tf.greater(input_batch.tgt.labels, 0), py_utils.FPropDtype(self.params))
    return non_padding

  def _ComputeSoftLabels(self, input_batch):
    p = self.params
    vocab_size = p.vocab_size
    if 'soft_labels' in input_batch.tgt:
      tf.logging.info('using input_batch.tgt.soft_labels: %r',
                      input_batch.tgt.soft_labels)
      soft_labels = input_batch.tgt.soft_labels
    else:
      label_smoothing = p.label_smoothing
      off_value = label_smoothing / vocab_size
      on_value = 1.0 - label_smoothing + off_value
      tf.logging.info({'on_value': on_value, 'off_value': off_value})
      soft_labels = tf.one_hot(
          tf.maximum(input_batch.tgt.labels, 0),
          vocab_size,
          on_value=on_value,
          off_value=off_value)
    return soft_labels

  def ComputePerTokenLoss(self, logits, input_batch):
    p = self.params

    with tf.name_scope(p.name):
      soft_labels = self._ComputeSoftLabels(input_batch)
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=soft_labels, logits=logits)

      if self.params.z_loss > 0.0:
        log_z = tf.math.reduce_logsumexp(logits, -1)
        z_loss_increment = self.params.z_loss * tf.math.square(log_z)
        loss += z_loss_increment

      non_padding = self._ComputeNonPadding(input_batch)
      return loss * non_padding

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params

    vocab_size = p.vocab_size

    with tf.name_scope(p.name):
      logits, aux_loss = predictions
      soft_labels = self._ComputeSoftLabels(input_batch)

      if p.use_entmax:
        entropy = entmax.entmax_loss(
            labels=tf.one_hot(input_batch.tgt.labels, vocab_size),
            inputs=logits,
            alpha=p.entmax_alpha,
            n_iter=p.entmax_n_iters,
            ensure_sum_one=p.entmax_ensure_sum_one)
        loss = entmax.entmax_loss(
            labels=soft_labels,
            inputs=logits,
            alpha=p.entmax_alpha,
            n_iter=p.entmax_n_iters,
            ensure_sum_one=p.entmax_ensure_sum_one)
      else:
        entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(input_batch.tgt.labels, vocab_size),
            logits=logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=soft_labels, logits=logits)

      top1 = tf.math.argmax(
          logits, -1, output_type=input_batch.tgt.labels.dtype)
      acc1 = tf.cast(tf.equal(input_batch.tgt.labels, top1), logits.dtype)
      assert acc1.shape == entropy.shape, (acc1.shape, entropy.shape)

      soft_labels_entropy = loss

      # To make sure the entmax works as intended, the z_loss should be set
      # to 0.
      if self.params.z_loss > 0.0:
        log_z = tf.math.reduce_logsumexp(logits, -1)
        z_loss_increment = self.params.z_loss * tf.math.square(log_z)
        loss += z_loss_increment

      non_padding = self._ComputeNonPadding(input_batch)

      per_token_loss = loss * non_padding
      if self.params.z_loss:
        per_token_z_loss_increment = z_loss_increment * non_padding

      if p.loss_denominator:
        loss_denom = p.loss_denominator
      elif p.use_tgt_labels_size_as_loss_denominator:
        # E.g. loss is going to be tiny if inputs are not packed and only a
        # fraction of tgt_labels are non-padding.
        loss_denom = tf.reduce_sum(tf.ones_like(non_padding))
      else:
        loss_denom = tf.reduce_sum(non_padding)
      avg_loss = tf.reduce_sum(per_token_loss) / loss_denom
      avg_z_loss_increment = (tf.reduce_sum(per_token_z_loss_increment) /
                              loss_denom) if p.z_loss else 0.0

      soft_labels_entropy = (
          tf.reduce_sum(soft_labels_entropy * non_padding) /
          tf.reduce_sum(non_padding))
      avg_loss += p.aux_loss_coef * aux_loss

      num_items_in_batch = tf.reduce_sum(
          tf.reduce_max(input_batch.tgt.segment_ids, axis=1))
      num_nonpadding = tf.reduce_sum(
          _ToInt32(tf.not_equal(input_batch.tgt.segment_ids, 0)))
      batch_capacity = tf.size(input_batch.tgt.labels)

      whole_tgt_correct = tf.cast(
          tf.equal(
              tf.reduce_sum(acc1 * non_padding, 1),
              tf.reduce_sum(non_padding, 1)), non_padding.dtype)

      # TODO(lepikhin): consider returning
      #   {'loss': (unnormalized per_token_loss, tf.reduce_sum(non_padding))}
      per_step_loss = {
          'loss': tf.reshape(avg_loss, [1]),
      }

      eval_metrics = {
          'num_packed_examples': (num_items_in_batch, 1.0),
          'batch_utilized_ratio': (num_nonpadding / batch_capacity, 1.0),
          'acc1':
              (tf.reduce_sum(acc1 * non_padding) / tf.reduce_sum(non_padding),
               tf.reduce_sum(non_padding)),
          'whole_tgt_accuracy':
              (tf.reduce_sum(whole_tgt_correct) /
               tf.cast(whole_tgt_correct.shape[0], whole_tgt_correct.dtype), 1.0
              ),
          'mean_xent': (tf.reduce_sum(entropy * non_padding) /
                        tf.reduce_sum(non_padding), tf.reduce_sum(non_padding)),
          'soft_labels_xent': (soft_labels_entropy, tf.reduce_sum(non_padding)),
          'weight': (tf.reduce_sum(non_padding), 1.0),
          'loss': (avg_loss, 1.0),
          'aux_loss': (p.aux_loss_coef * aux_loss, 1.0),
          'avg_z_loss_increment': (avg_z_loss_increment, 1.0),
      }
      if 'word_count' in input_batch.tgt:
        # +input_batch.num_sentences to account for the end of sequence symbol.
        num_words = tf.cast(
            tf.reduce_sum(
                input_batch.tgt.word_count +
                tf.cast(input_batch.tgt.num_sentences, dtype=tf.int32)),
            py_utils.FPropDtype(p))
        eval_metrics['num_words'] = (num_words, 1.0)
        eval_metrics['log_pplx_per_word'] = (
            tf.reduce_sum(entropy * non_padding) / num_words, num_words)
      # During training, the tpu summary tensors are added in _BPropGenTrainOps.
      if self.do_eval:
        for key, (val, wgt) in six.iteritems(py_utils.GetTpuSummaryTensors()):
          tf.logging.info('TpuSummaryTensor=>EvalMetric %r %r', key, (val, wgt))
          eval_metrics[key] = (val, wgt)
      return eval_metrics, per_step_loss

  def FilterPerExampleTensors(self, per_step):
    return per_step if self.params.debug else {}

  def ProcessFPropResults(self, sess, global_step, metrics, per_step):
    del sess, metrics
    if not per_step:
      return
    iterations_per_loop = 0
    for key, per_step_values in per_step.items():
      iterations_per_loop = iterations_per_loop or len(per_step_values)
      assert iterations_per_loop == len(per_step_values)

    for t in range(iterations_per_loop):
      for key, per_step_values in per_step.items():
        # Each per_step_values is an aggregation of outfeed tensor over
        # iterations_per_loop steps.
        tf.logging.info('Step = {}, {} = {}'.format(
            global_step + t - iterations_per_loop, key, per_step_values[t]))
