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
from lingvo.core import layers
from lingvo.core import moe_layers
from lingvo.core import py_utils
import numpy as np


class MoEBuilder(builder.Base):
  """Mixture-of-Experts Builder.

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
  single inputs row (e.g. packing 2 segments in a single row)::

      inputs      [  4,   3,  24]
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
        'Universal dropout rate that applies to inputs, attention, '
        'residual and other Transformer layers.')

    p.Define(
        'noise_shape_broadcast_dims', None,
        'A list of dimension where the noise shape is broadcasted. For '
        'example, noise_shape = [n, h, w, 1] when '
        'noise_shape_broadcast_dims=[-1] ')

    # attention params
    p.Define('attention_num_heads', 1, 'Attention number of heads.')
    p.Define('attention_key_value_dim', None,
             'Shared dimensionality for Attention keys, values.')
    p.Define('attention_dropout_prob', 0.0, 'Attention dropout probability.')
    p.Define('moe_dropout_rate', 0.0, 'MoE dropout probability.')
    p.Define(
        'attention_combine_dims', False, 'Attention optimization. '
        'The heads and key/value dimensions are combined in the variables '
        'and the computation.')

    p.Define('ff_dim', None, 'DenseReluDense hidden dim.')

    # MoE params
    p.Define('e_dim', None, 'E dimension. Number of experts.')
    p.Define('c_dim', None, 'C dimension. Per-expert capacity.')
    p.Define('moe_hidden_dim', None, 'Mixture-of-Experts hidden dim.')

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

    p.Define('attention_extra_logit', None,
             'Extra logit for attention softmax.')
    return p

  def _Dropout(self, name, keep_prob, noise_shape_broadcast_dims=None):
    return super()._Dropout(
        name, keep_prob, noise_shape_broadcast_dims or
        self.params.noise_shape_broadcast_dims)

  def _OneHotEncode(self, name, dim):
    fprop_dtype = py_utils.FPropDtype(self.params)
    return self._Fn(name, fn=lambda x: tf.one_hot(x, dim, dtype=fprop_dtype))

  def _Var(self, name, weights):
    return moe_layers.VarLayer.Params().Set(name=name, weights=weights)

  def _ShardedVar(self, name, weights):
    return moe_layers.ShardedVarLayer.Params().Set(
        name=name, weights=weights, num_devices=self.params.num_devices)

  def _EmbeddingWeight(self, name, vocab_dim):
    return self._Var(
        name=name,
        weights=[('embedding',
                  py_utils.WeightParams(
                      init=py_utils.WeightInit.Gaussian(),
                      dtype=self.params.dtype,
                      shape=[vocab_dim, self.params.model_dim]))])

  def SharedEmbSoftmax(self,
                       name,
                       vocab_size,
                       max_len,
                       logits_abs_max=None,
                       z_loss_coef=1e-4,
                       use_tgt_labels_size_as_loss_denominator=True):
    p = self.params
    return moe_layers.SharedEmbeddingSoftmaxLayer.Params().Set(
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

  def Mask(self):

    def _apply_padding(x, segment_id):  # pylint: disable=invalid-name
      mask = tf.cast(tf.not_equal(segment_id, 0), x.dtype)
      for _ in range(len(x.shape) - len(mask.shape)):
        mask = tf.expand_dims(mask, -1)
      return x * mask

    return self._Fn('mask', fn=_apply_padding, fn_out=lambda x, y: x)

  def EncoderLayer(self, name, layer):
    """Returns params for lambda x: x + DropOut(layer(LN(x)))."""
    return self._Graph(
        name,
        ['inputs', 'segment_id', 'segment_pos'],
        [
            'outputs',
            'aux_loss',
        ],
        ('inputs,segment_id->input_masked', self.Mask()),
        ('input_masked->x', self._LN('ln')),
        ('x,segment_id,segment_pos->' + 'y,aux_loss', layer),
        ('y->y_dropout', self._Dropout('y_dropout',
                                       1 - self.params.dropout_rate)),
        ('input_masked,y_dropout->outputs', self._Add('add')),
    )

  # We avoid Builder._Seq and Builder._Rep to improve theta / checkpoint
  # readability and reduce layer nesting.
  def EncoderLayerStack(self, name, sub_layers, num=1):
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

    Returns:
      The layer params.
    """
    stack = [
        ('inputs->inputs_split', self.Split('inputs_split')),
        ('segment_id->segment_id_split', self.Split('segment_id_split')),
        ('segment_pos->segment_pos_split', self.Split('segment_pos_split')),
    ]
    stack += [
        ('inputs_split->x_000',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('input_loss->loss_000', self._Identity('loss_000')),
    ]
    i = 0
    for _ in range(num):
      for l in sub_layers:
        # x_i, loss_i => x_{i+1}, loss_{i+1}
        stack += [('x_%03d,segment_id_split,segment_pos_split->'
                   'x_%03d,aux_loss_%03d' % (i, i + 1, i),
                   self.EncoderLayer('layer_%03d' % i, l)),
                  ('loss_%03d,aux_loss_%03d->loss_%03d' % (i, i, i + 1),
                   self._Add('loss_%03d' % (i + 1)))]
        i += 1

    stack += [
        (('loss_%03d->output_loss' % i), self._Identity('output_loss')),
        (('x_%03d->y_norm' % i), self._LN('final_layer_norm')),
        ('y_norm->y_dropout',
         self._Dropout('outputs_dropout', 1 - self.params.dropout_rate)),
        ('y_dropout,segment_id_split->outputs', self.Mask()),
    ]
    return self._Graph(name,
                       ['inputs', 'segment_id', 'segment_pos', 'input_loss'], [
                           'outputs',
                           'output_loss',
                       ], *stack)

  def DecoderLayer(self, name, layer, decoder=False):
    fprop_dtype = py_utils.FPropDtype(self.params)
    return self._Graph(
        name,
        [
            'inputs',
            'segment_id',
            'segment_pos',
            'encoder_output',
            'encoder_segment_id',
            'encoder_segment_pos',
        ],
        [
            'outputs',
            'aux_loss',
        ],
        ('inputs,segment_id->input_masked', self.Mask()),
        ('input_masked->x', self._LN('ln')),
        ('->zero_loss',
         self._Fn('zero_loss', lambda: tf.constant(0.0, fprop_dtype))),
        ('x,segment_id,segment_pos,' +
         'encoder_output,encoder_segment_id,encoder_segment_pos->' +
         'y,aux_loss', layer),
        ('y->y_dropout', self._Dropout('y_dropout',
                                       1 - self.params.dropout_rate)),
        ('input_masked,y_dropout->outputs', self._Add('add')),
    )

  def DecoderLayerStack(self, name, sub_layers, num=1):
    """Clean DecoderLayerStack."""
    stack = [
        ('inputs->inputs_split', self.Split('inputs_split')),
        ('segment_id->segment_id_split', self.Split('segment_id_split')),
        ('segment_pos->segment_pos_split', self.Split('segment_pos_split')),
        ('encoder_output->encoder_output_split',
         self.Split('encoder_output_split')),
        ('encoder_segment_id->encoder_segment_id_split',
         self.Split('encoder_segment_id_split')),
        ('encoder_segment_pos->encoder_segment_pos_split',
         self.Split('encoder_segment_pos_split')),
    ]
    stack += [
        ('inputs->x_000',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('input_loss->loss_000', self._Identity('loss_000')),
    ]
    i = 0
    for _ in range(num):
      for l in sub_layers:
        # x_i, loss_i => x_{i+1}, loss_{i+1}
        stack += [('x_%03d,segment_id_split,segment_pos_split,'
                   'encoder_output,encoder_segment_id_split,'
                   'encoder_segment_pos_split->'
                   'x_%03d,aux_loss_%03d' % (i, i + 1, i),
                   self.DecoderLayer('layer_%03d' % i, l)),
                  ('loss_%03d,aux_loss_%03d->loss_%03d' % (i, i, i + 1),
                   self._Add('loss_%03d' % (i + 1)))]
        i += 1

    stack += [
        (('loss_%03d->output_loss' % i), self._Identity('output_loss')),
        (('x_%03d->y_norm' % i), self._LN('final_layer_norm')),
        ('y_norm->y_dropout',
         self._Dropout('outputs_dropout', 1 - self.params.dropout_rate)),
        ('y_dropout,segment_id_split->outputs', self.Mask()),
    ]
    return self._Graph(name, [
        'inputs', 'segment_id', 'segment_pos', 'encoder_output',
        'encoder_segment_id', 'encoder_segment_pos', 'input_loss'
    ], [
        'outputs',
        'output_loss',
    ], *stack)

  def _DenseReluDenseWeights(self, name):
    return self._Var(
        name=name,
        weights=[('wi',
                  py_utils.WeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.model_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.model_dim, self.params.ff_dim])),
                 ('wo',
                  py_utils.WeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.ff_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.ff_dim, self.params.model_dim]))])

  def DenseReluDense(self, name, decoder=False):
    input_endpoints = ['inputs', 'segment_id', 'segment_pos']
    if decoder:
      input_endpoints += [
          'unused_encoder_output',
          'unused_encoder_segment_id',
          'unused_encoder_segment_pos',
      ]
    # Note that dropout is used here, but not in the MoE layer by default.
    return self._Graph(
        name,
        input_endpoints,
        ['outputs', 'aux_loss'],
        ('->wi,wo', self._DenseReluDenseWeights('w')),
        ('wi,inputs->h',
         self._Fn(
             'wi', fn=lambda wi, inputs: tf.einsum('MH,BLM->BLH', wi, inputs))),
        ('h->h_relu', self._Fn('relu', tf.nn.relu)),
        ('h_relu->h_dropout',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('wo,h_dropout->outputs_pre_split',
         self._Fn(
             'wo',
             fn=lambda wo, h_dropout: tf.einsum('HM,BLH->BLM', wo, h_dropout))),
        ('outputs_pre_split->outputs', self.Split('outputs_split')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
    )

  def _DenseReluDenseWeightsGatedGELU(self, name):
    # Gated GELU.  There are two separate linear transformations applied in
    # parallel to the inputs.  You take the gelu of one of them and then
    # multiply the two componentwise.
    return self._Var(
        name=name,
        weights=[('wi_0',
                  py_utils.WeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.model_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.model_dim, self.params.ff_dim])),
                 ('wi_1',
                  py_utils.WeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.model_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.model_dim, self.params.ff_dim])),
                 ('wo',
                  py_utils.WeightParams(
                      init=py_utils.WeightInit.Uniform(
                          (((1. / self.params.ff_dim)**0.5) * 3.0**0.5)),
                      dtype=self.params.dtype,
                      shape=[self.params.ff_dim, self.params.model_dim]))])

  def DenseReluDenseGatedGELU(self, name, decoder=False):
    # Need to unify.
    input_endpoints = ['inputs', 'segment_id', 'segment_pos']
    if decoder:
      input_endpoints += [
          'unused_encoder_output',
          'unused_encoder_segment_id',
          'unused_encoder_segment_pos',
      ]

    def _Impl(wi_0, wi_1, inputs):
      return tf.math.multiply(
          tf.nn.gelu(tf.einsum('MH,BLM->BLH', wi_0, inputs), approximate=True),
          # linear / pass-through
          tf.einsum('MH,BLM->BLH', wi_1, inputs))

    return self._Graph(
        name,
        input_endpoints,
        ['outputs', 'aux_loss'],
        ('->wi_0,wi_1,wo', self._DenseReluDenseWeightsGatedGELU('w')),
        ('wi_0,wi_1,inputs->h', self._Fn('wi', fn=_Impl)),
        ('h->h_dropout',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('wo,h_dropout->outputs_pre_split',
         self._Fn(
             'wo',
             fn=lambda wo, h_dropout: tf.einsum('HM,BLH->BLM', wo, h_dropout))),
        ('outputs_pre_split->outputs', self.Split('outputs_split')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
    )

  def MoE(self, name, decoder=False):
    input_endpoints = ['inputs', 'segment_id', 'segment_pos']
    if decoder:
      input_endpoints += [
          'unused_encoder_output',
          'unused_encoder_segment_id',
          'unused_encoder_segment_pos',
      ]
    return self._Graph(
        name, input_endpoints, ['outputs', 'aux_loss'],
        ('inputs->input_split', self.Split('input_split')),
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

    def _AddBias(logits, bias):
      # logits: BLHM [batch, length, heads, memory_length]
      # bias: BLHM [batch, length, heads, memory_length]
      #       (in case of attention with relative bias) OR
      #
      #       BLM  [batch, length, memory_length]
      #       (default masking bias with very negative logits).

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
        extra_logit = tf.convert_to_tensor(extra_logit, p.fprop_dtype)
        max_logit = tf.math.maximum(max_logit, extra_logit)
      x -= max_logit
      exp_x = tf.math.exp(x)
      sum_exp_x = tf.math.reduce_sum(exp_x, axis=-1, keepdims=True)
      if extra_logit is not None:
        sum_exp_x += tf.math.exp(extra_logit - max_logit)
      return tf.math.log(sum_exp_x) + max_logit

    def _LogSoftmax(x):
      return x - _ReduceLogsumexp(x)

    def _Softmax(x):
      # if p.attention_extra_logit is None:
      #   return tf.nn.softmax(x)
      # import ipdb; ipdb.set_trace()  # pyformat: disable
      return tf.math.exp(_LogSoftmax(x))

    return self._Graph(
        name,
        ['_q', '_k', '_v', 'bias'],
        ['outputs'],
        ('_q->q', self.Split('_q')),
        ('_k->k', self.Split('_k')),
        ('_v->v', self.Split('_v')),
        ('q,k->l',
         self._Fn('logits',
                  fn=lambda q, k: tf.einsum('BLHD,BMHD->BLHM', q, k))),
        ('l,bias->logits', self._Fn('bias', fn=_AddBias)),
        ('logits->w', self._Fn('weights', _Softmax)),
        ('w->weights', self._Dropout('dropout', 1 - self.params.dropout_rate)),
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

  def SelfAttention(self, name):
    """TransformerEncoder SelfAttention."""

    p = self.params

    def _Notvisible(x):
      a, b = tf.expand_dims(x, -1), tf.expand_dims(x, -2)
      return tf.cast(
          tf.math.logical_or(
              tf.not_equal(a, b),
              # also ignoring segment_id=0
              tf.math.logical_not(
                  tf.math.logical_or(tf.cast(a, tf.bool), tf.cast(b,
                                                                  tf.bool)))),
          py_utils.FPropDtype(p))

    # pyformat: disable
    return self._Graph(
        name,
        [
            'inputs',
            'segment_id',
            'segment_pos'
        ], [
            'outputs',
            'aux_loss'
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights('w')),
        ('segment_id->bias',
         self._Fn('bias',
                  fn=lambda x: _Notvisible(x) * (-1e+09),
                  fn_out=lambda x: x + x[-1])),
        ('inputs,wq->q', self._ComputeQKV('q')),
        ('inputs,wk->k', self._ComputeQKV('k')),
        ('inputs,wv->v', self._ComputeQKV('v')),
        ('q,k,v,bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  def DecEncAttention(self, name):
    """Transformer Decoder-Encoder Attention."""

    p = self.params

    def _Notvisible(a, b):
      """a, b are encoder_segment_id,(decoder_)segment_id Tensors."""
      a, b = tf.expand_dims(a, -1), tf.expand_dims(b, -2)
      return tf.cast(
          tf.math.logical_or(
              tf.not_equal(a, b),
              tf.math.logical_not(
                  tf.math.logical_or(tf.cast(a, tf.bool), tf.cast(b,
                                                                  tf.bool)))),
          py_utils.FPropDtype(p))

    # pyformat: disable
    return self._Graph(
        name,
        [
            'inputs',
            'segment_id',
            'segment_pos',
            'encoder_output',
            'encoder_segment_id',
            'encoder_segment_pos',
        ], [
            'outputs',
            'aux_loss',
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights('w')),
        ('segment_id,encoder_segment_id->bias',
         self._Fn('bias', fn=lambda a, b: -1e+09 * _Notvisible(a, b))),
        ('inputs,wq->q', self._ComputeQKV('q')),
        ('encoder_output,wk->k', self._ComputeQKV('k')),
        ('encoder_output,wv->v', self._ComputeQKV('v')),
        ('q,k,v,bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  def DecSelfAttention(self, name):
    """TransformerDecoder SelfAttention.

    Note that attention bias (see _Notvisible) ensures that current position
    (~row) is less that memory position(~column).

    Args:
      name: name of the layer.

    Returns:
      layer params for TransformerDecoder SelfAttention.
    """
    p = self.params
    fprop_dtype = py_utils.FPropDtype(self.params)

    def _Notvisible(
        segment_id,
        segment_pos,
    ):  # pylint: disable=missing-docstring
      a, b = tf.expand_dims(segment_id, -1), tf.expand_dims(segment_id, -2)
      return tf.cast(
          tf.math.logical_or(
              tf.less(  # position (~row) is less that memory position(~column)
                  tf.expand_dims(segment_pos, -1),
                  tf.expand_dims(segment_pos, -2)),
              tf.math.logical_or(
                  tf.not_equal(a, b),
                  # also ignoring segment_id=0
                  tf.math.logical_not(
                      tf.math.logical_or(
                          tf.cast(a, tf.bool), tf.cast(b, tf.bool))))),
          fprop_dtype)

    state_shape = [None, None, p.attention_num_heads, p.attention_key_value_dim]

    # pyformat: disable
    return self._Graph(
        name,
        [
            'inputs',
            'segment_id',
            'segment_pos',
            'unused_encoder_output',
            'unused_encoder_segment_id',
            'unused_encoder_segment_pos',
        ], [
            'outputs',
            'aux_loss',
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights('w')),
        ('inputs,wq->q', self._ComputeQKV('q')),
        ('inputs,wk->k', self._ComputeQKV('k')),
        ('inputs,wv->v', self._ComputeQKV('v')),
        ('k->k_full', self._State('k_state', state_shape)),
        ('v->v_full', self._State('v_state', state_shape)),
        ('segment_id,segment_pos->bias',
         self._Fn('bias',
                  fn=lambda x, y: _Notvisible(x, y) * (-1e+09),
                  fn_out=lambda x, y: x + x[-1])),
        ('bias->bias_full', self._Override('dec_self_attention_bias')),
        ('q,k_full,v_full,bias_full->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  def DecSelfAttentionRelativeBias(self, name):
    """DecSelfAttention with relative Attention Bias.

    Note that attention bias (see _Notvisible) ensures that current position
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

    Returns:
      The layer params.
    """
    p = self.params
    fprop_dtype = py_utils.FPropDtype(self.params)

    def _Notvisible(
        segment_id,
        segment_pos,
    ):  # pylint: disable=missing-docstring
      a, b = tf.expand_dims(segment_id, -1), tf.expand_dims(segment_id, -2)
      return tf.cast(
          tf.math.logical_or(
              tf.less(  # position (~row) is less that memory position(~column)
                  tf.expand_dims(segment_pos, -1),
                  tf.expand_dims(segment_pos, -2)),
              tf.math.logical_or(
                  tf.not_equal(a, b),
                  # also ignoring segment_id=0
                  tf.math.logical_not(
                      tf.math.logical_or(
                          tf.cast(a, tf.bool), tf.cast(b, tf.bool))))),
          fprop_dtype)

    def _ToInt32(t):
      return tf.cast(t, tf.int32)

    def _ToFloat(t):
      return tf.cast(t, fprop_dtype)

    def _RelativePositionBucket(relative_position, bidirectional=False):
      num_buckets = p.relative_attention_num_buckets
      max_distance = _ToFloat(p.relative_attention_max_distance)
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
          tf.math.log(_ToFloat(n) / max_exact) /
          tf.math.log(max_distance / max_exact) * (num_buckets - max_exact))
      val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
      ret += tf.where(is_small, n, val_if_large)
      return ret

    def _ComputeBias(segment_id, segment_pos):
      return _Notvisible(segment_id, segment_pos) * (-1e+09)

    # When training query_segment_pos = key_segment_pos, of shape [batch, time].
    # When decoding query_segment_pos is [batch, beam_size]
    # but key_segment_pos is [batch, memory_size] (because of k_pos StateLayer).
    def _AddRelativeBias(bias, query_segment_pos, key_segment_pos,
                         relative_bias_weights):
      if p.relative_attention_use_universal_1d_position:
        assert (int(key_segment_pos.shape[-1]) == int(
            query_segment_pos.shape[-1])), (key_segment_pos.shape,
                                            query_segment_pos.shape)
        len_dim = key_segment_pos.shape.as_list()[-1]
        key_segment_pos = query_segment_pos = tf.expand_dims(
            tf.range(len_dim), axis=0)

      # Relative position is defined in such a way that when query is in the
      # future relative to the key, the value of relative position is negative.
      relative_position = (
          tf.expand_dims(key_segment_pos, -2) -
          tf.expand_dims(query_segment_pos, -1))
      relative_bucket = _RelativePositionBucket(relative_position)

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
      relative_bias_inc = tf.einsum('HX,...LJX->...LHJ', relative_bias_weights,
                                    relative_bucket_one_hot)
      if relative_bias_inc.shape.ndims == 3:
        assert p.relative_attention_use_universal_1d_position
        relative_bias_inc = tf.expand_dims(relative_bias_inc, 0)

      # Eventually we add bias to BLHM [batch, length, heads, memory_length]
      # logits tensor, so we make 'heads' dim next to last.

      return tf.expand_dims(bias, -2) + relative_bias_inc

    state_shape = [None, None, p.attention_num_heads, p.attention_key_value_dim]

    # pyformat: disable
    return self._Graph(
        name,
        [
            'inputs',
            'segment_id',
            'segment_pos',
            'unused_encoder_output',
            'unused_encoder_segment_id',
            'unused_encoder_segment_pos',
        ], [
            'outputs',
            'aux_loss',
        ],
        ('->wq,wk,wv,wo', self._AttentionWeights('w')),
        ('->relative_bias_weights', self._RelativeAttentionBiasWeights('wrb')),
        ('inputs,wq->q', self._ComputeQKV('q')),
        ('inputs,wk->k', self._ComputeQKV('k')),
        ('inputs,wv->v', self._ComputeQKV('v')),
        ('k->k_full', self._State('k_state', state_shape)),
        ('v->v_full', self._State('v_state', state_shape)),
        ('segment_pos->key_segment_pos',
         self._State('seg_pos', [None, None], dtype=tf.int32)),
        ('segment_id,segment_pos->qq_bias', self._Fn('bias', fn=_ComputeBias)),
        ('qq_bias->qk_bias', self._Override('dec_self_attention_bias')),
        ('qk_bias,segment_pos,key_segment_pos,relative_bias_weights->qhk_bias',
         self._Fn('relative_bias', fn=_AddRelativeBias)),
        ('q,k_full,v_full,qhk_bias->o', self.Attention('attention')),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
        ('o,wo->outputs', self._Fn('outputs', fn=self._ComputeAttenOutputs)))
    # pyformat: enable

  def _RelativeAttentionBiasWeights(self, name):
    """Helper for '->rb' Graph edge."""
    p = self.params
    rb_stddev = (p.attention_num_heads * p.relative_attention_num_buckets)**-0.5
    rb_tpl = py_utils.WeightParams(
        shape=[p.attention_num_heads, p.relative_attention_num_buckets],
        dtype=self.params.dtype,
        init=py_utils.WeightInit.Gaussian(rb_stddev))
    return self._Var(name=name, weights=[('wrb', rb_tpl)])

  def _zero_aux_loss(self, name):  # pylint: disable=invalid-name
    return self._Fn(name,
                    lambda: tf.constant(0.0, py_utils.FPropDtype(self.params)))

  def _LN(self, name):
    """Overriding with bias-less layer norm."""
    return self._LNInternal(name)

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

  def Split(self, name):
    """Sets sharding attribute for the Tensor. Split across dim=0."""
    return self._Fn(
        name,
        lambda x: moe_layers.Split(x, 0, num_devices=self.params.num_devices))

  def _Add(self, name):
    return self._Fn(name, fn=lambda x, y: x + y, fn_out=lambda x, y: x)

  def _Identity(self, name):
    """Apply identity transformation."""
    return layers.IdentityLayer.Params().Set(name=name)

  def _AttentionWeights(self, name):
    """Helper for '->wq,wk,wv,wo' Graph edge."""

    p = self.params
    hd_dims = ([p.attention_num_heads *
                p.attention_key_value_dim] if p.attention_combine_dims else
               [p.attention_num_heads, p.attention_key_value_dim])
    q_stddev = (p.model_dim * p.attention_key_value_dim)**-0.5
    wq_tpl = py_utils.WeightParams(
        shape=[p.model_dim] + hd_dims,
        dtype=self.params.dtype,
        init=py_utils.WeightInit.Gaussian(q_stddev))
    kv_stddev = (p.model_dim)**-0.5
    wkv_tpl = py_utils.WeightParams(
        shape=[p.model_dim] + hd_dims,
        dtype=self.params.dtype,
        init=py_utils.WeightInit.Gaussian(kv_stddev))
    o_stddev = (p.attention_num_heads * p.attention_key_value_dim)**-0.5
    wo_tpl = py_utils.WeightParams(
        shape=hd_dims + [p.model_dim],
        dtype=self.params.dtype,
        init=py_utils.WeightInit.Gaussian(o_stddev))
    return self._Var(
        name=name,
        weights=[('wq', wq_tpl), ('wk', wkv_tpl), ('wv', wkv_tpl),
                 ('wo', wo_tpl)])

  def _ComputeQKV(self, name):
    p = self.params

    def _Compute(x, w):
      if p.attention_combine_dims:
        w = tf.reshape(
            w, [p.model_dim, p.attention_num_heads, p.attention_key_value_dim])
      return tf.einsum('BLM,MHD->BLHD', x, w)

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
      return moe_layers.Top2Gating(
          w=w,
          inputs=inputs,
          paddings=paddings,
          num_devices=p.num_devices,
          experts_dim=p.e_dim,
          expert_capacity_dim=p.c_dim,
          local_dispatch=True,
          fprop_dtype=py_utils.FPropDtype(p),
          use_xla_sharding=True,
          second_expert_policy=p.second_expert_policy,
          second_expert_threshold=p.second_expert_threshold,
          legacy_mtf_behavior=p.legacy_mtf_behavior,
          capacity_factor=p.capacity_factor)

    return self._Fn(name, _Compute)

  def _ShardedFeedForwardNetworksWeights(self, name):
    """Gets the sharded weights for the two layer feedforward nets."""
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
    return self._ShardedVar(name=name, weights=[('wi', wi_pc), ('wo', wo_pc)])

  def _FeedForwardNetworksApplyGating(self, name):
    p = self.params

    def _Compute(gating, inputs, reshaped_inputs, wi, wo):
      return moe_layers.FeedForwardNetworksApplyGating(
          gating,
          inputs,
          reshaped_inputs,
          wi,
          wo,
          num_devices=p.num_devices,
          num_groups=p.num_groups or p.num_devices,
          dropout_rate=p.moe_dropout_rate)

    return self._Fn(name, _Compute)

  def _ShardedMoEPositionWiseFeedForwardNetworks(self, name):
    """Simple MoE FFN with xla_sharding."""
    p = self.params
    num_groups = p.num_groups or p.num_devices

    def _ReshapeInputs(inputs, segment_id):
      """Prepare inputs and paddings for the gating layer."""
      paddings = tf.cast(tf.equal(segment_id, 0), inputs.dtype)
      orig_inputs = inputs
      inputs = tf.reshape(orig_inputs, [
          num_groups,
          (orig_inputs.shape[0] * orig_inputs.shape[1]) // num_groups,
          orig_inputs.shape[-1],
      ])
      inputs = moe_layers.Split(inputs, 0, p.num_devices)
      paddings = tf.reshape(paddings, inputs.shape[:2])
      return inputs, paddings

    return self._Graph(name, ['inputs', 'segment_id', 'wi', 'wo'],
                       ['outputs', 'aux_loss'],
                       ('inputs,segment_id->reshaped_inputs, paddings',
                        self._Fn('reshape_inputs', _ReshapeInputs)),
                       ('->gw', self._Top2GatingWeights('top_2_gating')),
                       ('gw,reshaped_inputs,paddings->gating',
                        self._ComputeTopKGating('compute_gating')),
                       ('gating,inputs,reshaped_inputs,wi,wo->outputs,aux_loss',
                        self._FeedForwardNetworksApplyGating('process_gating')))

  def _State(self, name, shape, dtype=None):
    dtype = dtype or py_utils.FPropDtype(self.params)
    return moe_layers.StateLayer.Params().Set(
        name=name, shape=shape, dtype=dtype)

  def _Override(self, name, key=None):
    return moe_layers.OverrideLayer.Params().Set(name=name, key=key or name)

  def _Softmax(self, dec_outs, tgt, w, vocab_dim):
    p = self.params

    def _MaybeSplit(x):
      return moe_layers.Split(x, 0, p.num_devices)

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
    p.Define(
        'device_mesh', None,
        'Device mesh as a numpy ND array of device IDs. If specified, '
        'device order in the array will be preserved, and the shape must equal '
        'device_mesh_shape if that is also specified.')

    # Weight sharding configs.
    p.Define('emb_w_split', None, 'Mesh split for embedding weights.')
    p.Define('mhd_w_split', [0, 1, -1], 'Mesh split for attention MHD weight')
    p.Define('mh_wi_split', [0, 1], 'Mesh split for dense MH weight')
    p.Define('hm_wo_split', [1, 0], 'Mesh split for dense HM weight')

    # Activation sharding configs.
    # one_hot_ids_split split should be inferred by XLA.
    p.Define('one_hot_ids_split', None, 'Mesh split for one_hot_ids.')
    p.Define('emb_out_split', [0, -1, -1], 'Mesh split for embedding outputs.')
    p.Define('qkv_split', [0, -1, 1, -1], 'Mesh split for Q, K, V')
    p.Define('blm_split', [0, -1, -1], 'Mesh split for BLM.')
    p.Define('blh_split', [0, -1, 1], 'Mesh split for BLH.')
    p.Define('logits_split', [0, -1, -1], 'Mesh split for logits.')
    p.Define('model_dim_reshape_segments', None,
             'Size of N when reshaping model dimension M to Nm')
    p.attention_combine_dims = False
    return p

  def _AdjustMSplit(self, split, m_dim):
    """Adjusts split annotation according to model_dim_reshape_segments."""
    if self.params.model_dim_reshape_segments is None:
      return split
    new_split = list(split)
    new_split.insert(m_dim + 1, -1)
    return new_split

  def _ReshapeM(self, x, m_dim):
    """Reshapes tensor x according to model_dim_reshape_segments."""
    new_shape = x.shape
    if self.params.model_dim_reshape_segments is not None:
      new_shape = list(x.shape[0:m_dim])
      new_shape.append(self.params.model_dim_reshape_segments)
      new_shape.append(x.shape[m_dim] // self.params.model_dim_reshape_segments)
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
    if self.params.model_dim_reshape_segments is None:
      return tf.einsum(equation, x, y)
    new_equation = ''
    for c in equation:
      assert c != 'N'
      if c == 'M':
        new_equation += 'N'
      new_equation += c
    return tf.einsum(new_equation, x, y)

  def EinsumWithModelDim(self, name, equation):
    return self._Fn(name, lambda x, y: self._EinsumWithModelDim(equation, x, y))

  def _LN(self, name):
    """Overriding _LN to consider model_dim_reshape_segments."""
    if self.params.model_dim_reshape_segments is None:
      return super()._LN(name)

    ln_weight_reshape = [
        self.params.model_dim_reshape_segments,
        self.params.model_dim // self.params.model_dim_reshape_segments
    ]
    return self._LNInternal(name, ln_weight_reshape)

  def _MeshSplit(self, x, tensor_split_dims_mapping):
    p = self.params
    device_mesh = p.device_mesh
    if device_mesh is None:
      if not p.device_mesh_shape:
        return x
      num_devices = np.product(p.device_mesh_shape)
      device_mesh = np.reshape(np.arange(0, num_devices), p.device_mesh_shape)
    elif p.device_mesh_shape is not None:
      assert p.device_mesh_shape == list(p.device_mesh.shape)
    return moe_layers.MeshSplit(x, device_mesh, tensor_split_dims_mapping)

  def MeshSplit(self, name, tensor_split_dims_mapping):
    return self._Fn(name,
                    lambda x: self._MeshSplit(x, tensor_split_dims_mapping))

  def Embedding(self, name, vocab_dim):
    p = self.params
    return self._Graph(
        name, ['ids'], ['outputs'],
        ('->emb', self._EmbeddingWeight('w', vocab_dim)),
        ('emb->emb_split', self.MeshSplit('w_split', p.emb_w_split)),
        ('ids->one_hot_ids', self._OneHotEncode('one_hot_ids', vocab_dim)),
        ('one_hot_ids->one_hot_ids_split',
         self.MeshSplit('one_hot_ids_split', p.one_hot_ids_split)),
        ('emb_split,one_hot_ids_split->outputs_pre_split',
         self._Fn('einsum', fn=lambda w, x: tf.einsum('VH,BLV->BLH', w, x))),
        ('outputs_pre_split->outputs_pre_reshape',
         self.MeshSplit('output_split', p.emb_out_split)),
        ('outputs_pre_reshape->outputs',
         self._Fn('outputs_reshape', fn=lambda x: self._ReshapeM(x, 2))))

  def Attention(self, name):
    """Attention with multiple attention heads."""
    p = self.params

    def _AddBias(logits, bias):
      # logits: BLHM [batch, length, heads, memory_length]
      # bias: BLHM [batch, length, heads, memory_length]
      #       (in case of attention with relative bias) OR
      #
      #       BLM  [batch, length, memory_length]
      #       (default masking bias with very negative logits).

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
        extra_logit = tf.convert_to_tensor(extra_logit, p.fprop_dtype)
        max_logit = tf.math.maximum(max_logit, extra_logit)
      x -= max_logit
      exp_x = tf.math.exp(x)
      sum_exp_x = tf.math.reduce_sum(exp_x, axis=-1, keepdims=True)
      if extra_logit is not None:
        sum_exp_x += tf.math.exp(extra_logit - max_logit)
      return tf.math.log(sum_exp_x) + max_logit

    def _LogSoftmax(x):
      return x - _ReduceLogsumexp(x)

    def _Softmax(x):
      # TODO(lepikhin): consider
      # if p.attention_extra_logit is None:
      #   return tf.nn.softmax(x)
      return tf.math.exp(_LogSoftmax(x))

    return self._Graph(
        name, ['_q', '_k', '_v', 'bias'], ['outputs'],
        ('_q->q', self.MeshSplit('_q', p.qkv_split)),
        ('_k->k', self.MeshSplit('_k', p.qkv_split)),
        ('_v->v', self.MeshSplit('_v', p.qkv_split)),
        ('q,k->l',
         self._Fn('logits',
                  fn=lambda q, k: tf.einsum('BLHD,BMHD->BLHM', q, k))),
        ('l,bias->logits', self._Fn('bias', fn=_AddBias)),
        ('logits->w', self._Fn('weights', _Softmax)),
        ('w->weights', self._Dropout('dropout', 1 - self.params.dropout_rate)),
        ('weights->weights_split', self.MeshSplit('_wsplit', p.qkv_split)),
        ('weights_split,v->outputs_unsplitted',
         self._Fn(
             'outputs',
             fn=lambda weights, v: tf.einsum('BLHM,BMHD->BLHD', weights, v))),
        ('outputs_unsplitted->outputs', self.MeshSplit('_o', p.qkv_split)))

  def _ComputeQKV(self, name):
    p = self.params

    def _Compute(x, w):
      if p.attention_combine_dims:
        combined_split = None if p.mhd_w_split is None else p.mhd_w_split[:-1]
        w = self._MeshSplit(w, combined_split)
        w = tf.reshape(
            w, [p.model_dim, p.attention_num_heads, p.attention_key_value_dim])
      w = self._MeshSplit(w, p.mhd_w_split)
      w = self._ReshapeM(w, 0)
      return self._EinsumWithModelDim('BLM,MHD->BLHD', x, w)

    return self._Fn(name, _Compute)

  def _ComputeAttenOutputs(self, o, wo):
    p = self.params
    hdm_split = None if p.mhd_w_split is None else [
        p.mhd_w_split[1], -1, p.mhd_w_split[0]
    ]
    if p.attention_combine_dims:
      combined_split = None if hdm_split is None else [
          hdm_split[0], hdm_split[2]
      ]
      wo = self._MeshSplit(wo, combined_split)
      wo = tf.reshape(
          wo, [p.attention_num_heads, p.attention_key_value_dim, p.model_dim])
    wo = self._MeshSplit(wo, hdm_split)
    wo = self._ReshapeM(wo, 2)
    return self._MeshSplit(
        self._EinsumWithModelDim('HDM,BLHD->BLM', wo, o),
        self._AdjustMSplit(p.blm_split, 2))

  def DenseReluDense(self, name, decoder=False):
    input_endpoints = ['inputs', 'segment_id', 'segment_pos']
    if decoder:
      input_endpoints += [
          'unused_encoder_output',
          'unused_encoder_segment_id',
          'unused_encoder_segment_pos',
      ]
    p = self.params
    # Note that dropout is used here, but not in the MoE layer by default.
    return self._Graph(
        name,
        input_endpoints,
        ['outputs', 'aux_loss'],
        ('->wi,wo', self._DenseReluDenseWeights('w')),
        ('wi->wi_split', self.MeshSplit('wi_split', p.mh_wi_split)),
        ('wo->wo_split', self.MeshSplit('wo_split', p.hm_wo_split)),
        ('wi_split->wi_reshaped',
         self._Fn('wi_reshape', fn=lambda x: self._ReshapeM(x, 0))),
        ('wo_split->wo_reshaped',
         self._Fn('wo_reshape', fn=lambda x: self._ReshapeM(x, 1))),
        ('wi_reshaped,inputs->h', self.EinsumWithModelDim('wi', 'MH,BLM->BLH')),
        ('h->h_split', self.MeshSplit('_h_split', p.blh_split)),
        ('h_split->h_relu', self._Fn('relu', tf.nn.relu)),
        ('h_relu->h_dropout', self._Dropout('input_dropout',
                                            1 - p.dropout_rate)),
        ('wo_reshaped,h_dropout->outputs_pre_split',
         self.EinsumWithModelDim('wo', 'HM,BLH->BLM')),
        ('outputs_pre_split->outputs',
         self.MeshSplit('outputs_split', self._AdjustMSplit(p.blm_split, 2))),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
    )

  def DenseReluDenseGatedGELU(self, name, decoder=False):
    # Need to unify.
    input_endpoints = ['inputs', 'segment_id', 'segment_pos']
    if decoder:
      input_endpoints += [
          'unused_encoder_output',
          'unused_encoder_segment_id',
          'unused_encoder_segment_pos',
      ]

    def _Impl(wi_0, wi_1, inputs):
      return tf.math.multiply(
          tf.nn.gelu(
              self._EinsumWithModelDim('MH,BLM->BLH', wi_0, inputs),
              approximate=True),
          # linear / pass-through
          self._EinsumWithModelDim('MH,BLM->BLH', wi_1, inputs))

    p = self.params
    return self._Graph(
        name,
        input_endpoints,
        ['outputs', 'aux_loss'],
        ('->wi_0,wi_1,wo', self._DenseReluDenseWeightsGatedGELU('w')),
        ('wi_0->wi_0_split', self.MeshSplit('wi_0_split', p.mh_wi_split)),
        ('wi_1->wi_1_split', self.MeshSplit('wi_1_split', p.mh_wi_split)),
        ('wo->wo_split', self.MeshSplit('wo_split', p.hm_wo_split)),
        ('wi_0_split->wi_0_reshaped',
         self._Fn('wi0_reshape', fn=lambda x: self._ReshapeM(x, 0))),
        ('wi_1_split->wi_1_reshaped',
         self._Fn('wi1_reshape', fn=lambda x: self._ReshapeM(x, 0))),
        ('wo_split->wo_reshaped',
         self._Fn('wo_reshape', fn=lambda x: self._ReshapeM(x, 1))),
        ('wi_0_reshaped,wi_1_reshaped,inputs->h', self._Fn('wi', fn=_Impl)),
        ('h->h_split', self.MeshSplit('_h_split', p.blh_split)),
        ('h_split->h_dropout',
         self._Dropout('input_dropout', 1 - self.params.dropout_rate)),
        ('wo_reshaped,h_dropout->outputs_pre_split',
         self.EinsumWithModelDim('wo', 'HM,BLH->BLM')),
        ('outputs_pre_split->outputs',
         self.MeshSplit('outputs_split', self._AdjustMSplit(p.blm_split, 2))),
        ('->aux_loss', self._zero_aux_loss('aux_loss')),
    )


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
    p.Define('batch_size', None, 'Batch size.')
    p.Define('num_transformer_layers', None,
             'Number of blocks in builder.{Decoder,Encoder}LayerStack.')

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
    p.Define('gated_gelu', False, 'FFN gated GELU.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    b = p.builder.Instantiate()

    tgt_vocab_size = p.vocab_size

    dec_emb = b.Embedding('dec_emb', tgt_vocab_size)
    self.CreateChild('dec_emb', dec_emb)

    if p.positional_embedding:
      dec_pos_emb = b.Embedding('dec_pos_emb', p.max_length)
      self.CreateChild('dec_pos_emb', dec_pos_emb)

    dec = b.DecoderLayerStack('decoder', [
        (b.DecSelfAttentionRelativeBias('dec_self_attention')
         if not p.positional_embedding else
         b.DecSelfAttention('dec_self_attention')),
        (b.DenseReluDense('dense_relu_dense', decoder=True) if not p.gated_gelu
         else b.DenseReluDenseGatedGELU('dense_relu_dense', decoder=True)),
    ], p.num_transformer_layers)
    dec.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

    emb_w_split = b.MeshSplit('w_split', b.params.emb_w_split)
    dec_out_split = b.MeshSplit('dec_out_split',
                                b._AdjustMSplit(b.params.blm_split, 2))
    logits_split = b.MeshSplit('logits_split', b.params.logits_split)

    self.CreateChild('dec', dec)
    self.CreateChild('emb_w_split', emb_w_split)
    self.CreateChild('dec_out_split', dec_out_split)
    self.CreateChild('logits_split', logits_split)

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
      # ops.text_packed:
      #   target_id_eos => tgt_labels
      #   target_bos_id => tgt_ids

      y = self.dec_emb.FProp(theta.dec_emb, input_batch.tgt.ids)

      if p.positional_embedding:
        y += self.dec_pos_emb.FProp(theta.dec_pos_emb,
                                    input_batch.tgt.segment_pos)

      dec_outputs, aux_loss = self.dec.FProp(
          theta.dec, y, input_batch.tgt.segment_ids,
          input_batch.tgt.segment_pos, tf.zeros_like(y),
          tf.zeros_like(input_batch.tgt.segment_ids),
          tf.zeros_like(input_batch.tgt.segment_pos),
          tf.convert_to_tensor(0.0, py_utils.FPropDtype(p)))
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
    non_padding = tf.cast(
        tf.not_equal(input_batch.tgt.segment_ids, 0),
        py_utils.FPropDtype(self.params))
    return non_padding

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params

    vocab_size = p.vocab_size

    with tf.name_scope(p.name):
      logits, aux_loss = predictions
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
            input_batch.tgt.labels,
            vocab_size,
            on_value=on_value,
            off_value=off_value)

      xent = tf.nn.softmax_cross_entropy_with_logits(
          labels=tf.one_hot(input_batch.tgt.labels, vocab_size), logits=logits)

      top1 = tf.math.argmax(
          logits, -1, output_type=input_batch.tgt.labels.dtype)
      acc1 = tf.cast(tf.equal(input_batch.tgt.labels, top1), logits.dtype)
      assert acc1.shape == xent.shape, (acc1.shape, xent.shape)

      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=soft_labels, logits=logits)
      soft_labels_xent = loss

      if self.params.z_loss > 0.0:
        log_z = tf.math.reduce_logsumexp(logits, -1)
        z_loss_increment = self.params.z_loss * tf.math.square(log_z)
        loss += z_loss_increment

      non_padding = self._ComputeNonPadding(input_batch)

      per_token_loss = loss * non_padding
      if self.params.z_loss:
        per_token_z_loss_increment = z_loss_increment * non_padding

      if p.use_tgt_labels_size_as_loss_denominator:
        # E.g. loss is going to be tiny if inputs are not packed and only a
        # fraction of tgt_labels are non-padding.
        loss_denom = tf.reduce_sum(tf.ones_like(non_padding))
      else:
        loss_denom = tf.reduce_sum(non_padding)
      avg_loss = tf.reduce_sum(per_token_loss) / loss_denom
      avg_z_loss_increment = (tf.reduce_sum(per_token_z_loss_increment) /
                              loss_denom) if p.z_loss else 0.0

      soft_labels_xent = (
          tf.reduce_sum(soft_labels_xent * non_padding) /
          tf.reduce_sum(non_padding))
      avg_loss += p.aux_loss_coef * aux_loss

      # TODO(lepikhin): consider returning
      #   {'loss': (unnormalized per_token_loss, tf.reduce_sum(non_padding))}
      per_step_loss = {
          'loss': tf.reshape(avg_loss, [1]),
      }

      eval_metrics = {
          'acc1':
              (tf.reduce_sum(acc1 * non_padding) / tf.reduce_sum(non_padding),
               tf.reduce_sum(non_padding)),
          'mean_xent':
              (tf.reduce_sum(xent * non_padding) / tf.reduce_sum(non_padding),
               tf.reduce_sum(non_padding)),
          'soft_labels_xent': (soft_labels_xent, tf.reduce_sum(non_padding)),
          'weight': (tf.reduce_sum(non_padding), 1.0),
          'loss': (avg_loss, 1.0),
          'aux_loss': (p.aux_loss_coef * aux_loss, 1.0),
          'avg_z_loss_increment': (avg_z_loss_increment, 1.0),
      }
      eval_metrics.update(py_utils.GetTpuSummaryTensors())
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
        tf.logging.info('Step = {}, {} = {}'.format(global_step + t, key,
                                                    per_step_values[t]))
