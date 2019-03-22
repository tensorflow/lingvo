# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Lingvo layers that depend on attention layers but are not recurrent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from lingvo.core import attention
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import py_utils


class TransformerAttentionLayer(base_layer.BaseLayer):
  """Multi-headed attention, add and norm used by 'Attention Is All You Need'.

  This class implements the first sub-layer of Transformer Layer. Input is
  first processed using a multi-headed (self) attention. Output of the
  attention layer is combined with the residual connection. And the finally,
  output is normalized using Layer Normalization.

  Layer can be used in three scenarios:

  1. Multi-Headed Self-Attention, where attention keys (source vectors),
     attention values (context vectors) and queries come from the same previous
     layer output, `query_vec`. This is the general use case for encoder
     Transformer Layers.
  2. Masked Multi-Headed Self-Attention, where attention keys, attention values
     and queries all come from the same previous layer output, but rightward
     activations are masked to prevent information flow from future. This is the
     use case for decoder self-attention Transformer Layers. Can be activated by
     setting is_masked flag of this layer.
  3. Multi-Headed Attention, where attention keys and attention values
     `source_vecs`, are coming from a different source (output of the encoder)
     and queries `query_vec`, coming from the previous layer outputs (decoder).
     This corresponds to the standard attention mechanism, decoder attending the
     encoder outputs.
  """

  @classmethod
  def Params(cls):
    p = super(TransformerAttentionLayer, cls).Params()
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define('atten_hidden_dim', 0, 'Dimension of the attention hidden dim.')
    p.Define('num_attention_heads', 8, 'Number of attention heads.')
    p.Define('is_masked', False, 'If set, uses masked MultiHededAttention.')
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define(
        'atten_tpl',
        attention.MultiHeadedAttention.Params().Set(
            use_source_vec_as_attention_value=False, enable_ctx_post_proj=True),
        'Multi-Headed Dot-Attention default params')
    p.Define(
        'atten_dropout_prob', 0.0,
        'Probability at which we apply dropout to the attention probs. '
        'This practically drops memory values at random positions.')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, y) = (x + dropout(y)).')
    p.Define(
        'residual_dropout_tpl', layers.DropoutLayer.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define('add_unnormalized_input', False, 'If set, uses unnormalized input '
             'in the residual add.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerAttentionLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim

    if not p.atten_hidden_dim:
      p.atten_hidden_dim = p.source_dim

    with tf.variable_scope(p.name):
      # Initialize multi-headed attention
      params = p.atten_tpl.Copy()
      params.name = 'multihead_atten'
      params.source_dim = p.source_dim
      params.query_dim = p.source_dim
      params.hidden_dim = p.atten_hidden_dim
      params.context_dim = p.source_dim
      params.ctx_post_proj_dim = p.source_dim
      params.num_attention_heads = p.num_attention_heads
      params.atten_dropout_prob = p.atten_dropout_prob
      params.packed_input = p.packed_input
      self.CreateChild('atten', params)

      # Initialize attention layer norm
      params = p.ln_tpl.Copy()
      params.name = 'atten_ln'
      params.input_dim = p.source_dim
      self.CreateChild('layer_norm', params)

      dropout_tpl = p.residual_dropout_tpl.Copy()
      dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
      self.CreateChild('residual_dropout', dropout_tpl)

  def FProp(self,
            theta,
            query_vec,
            source_paddings,
            source_vecs=None,
            query_segment_id=None,
            source_segment_id=None):
    """Transformer attention, residual and normalization layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [target_time, target_batch, dim]
      source_paddings: [source_time, source_batch]
      source_vecs: [source_time, source_batch, dim].
      query_segment_id: [target_time, target_batch]
      source_segment_id: [source_time, source_batch]

    Returns:
      (output, atten_probs). output is of shape [target_time, target_batch,
      source_dim], atten_probs is of shape [target_time, target_batch,
      source_time].
    """
    p = self.params
    unnormalized_query_vec = query_vec
    query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)

    if source_vecs is None:
      source_vecs = query_vec
      source_segment_id = query_segment_id

    if p.is_masked:
      assert source_vecs is not None
      query_vec = py_utils.with_dependencies([
          py_utils.assert_shape_match(
              tf.shape(source_vecs), tf.shape(query_vec))
      ], query_vec)
      # Prepares mask for self-attention
      # [time, time]
      target_time = tf.shape(query_vec)[0]
      target_bs = tf.shape(query_vec)[1]
      triangle_padding = 1.0 - tf.matrix_band_part(
          tf.ones([target_time, target_time], dtype=py_utils.FPropDtype(p)), -1,
          0)
      # [time,  batch, time]
      causal_padding = tf.tile(
          tf.expand_dims(triangle_padding, 1), [1, target_bs, 1])

      causal_padding = tf.reshape(causal_padding, [-1, target_time])
    else:
      causal_padding = None

    query_dim = tf.shape(query_vec)[-1]
    packed_src = self.atten.PackSource(theta.atten, source_vecs, source_vecs,
                                       source_paddings, source_segment_id)

    if query_segment_id is not None:
      query_segment_id = tf.reshape(query_segment_id, [-1])
    ctx_vec, atten_prob, _ = self.atten.ComputeContextVectorWithSource(
        theta.atten,
        packed_src,
        tf.reshape(query_vec, [-1, query_dim]),
        per_step_source_padding=causal_padding,
        query_segment_id=query_segment_id)
    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    h = input_to_add + tf.reshape(ctx_vec, tf.shape(query_vec))
    atten_prob = tf.reshape(atten_prob, [
        tf.shape(query_vec)[0],
        tf.shape(query_vec)[1],
        tf.shape(source_vecs)[0]
    ])
    return h, atten_prob

  def ExtendStep(self, theta, query_vec, prefix_state, t=None):
    """Extend prefix by one more time step.

    This function is expected to be called during fast decoding of the
    Transformer model.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [target_batch, dim]
      prefix_state: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
      t: a scalar, the current time step, 0-based.

    Returns:
      A triplet (cur_output, atten_prob, new_state) where cur_output is a tensor
      representing the output from the current state, and new_state is the new
      state `.NestedMap`.
    """
    p = self.params
    assert p.is_masked  # Must be causal attention.
    unnormalized_query_vec = query_vec
    query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)

    cached_packed_src = py_utils.NestedMap(
        source_vecs=prefix_state.key,
        source_contexts=prefix_state.value,
        source_padding=None,
        source_segment_id=None)
    extended_packed_src = self.atten.ExtendSourcePacked(
        theta.atten, query_vec, query_vec, None, None, cached_packed_src, t)
    if t is not None:
      source_seq_len = tf.shape(extended_packed_src.source_vecs)[0]
      zero_padding = tf.fill([source_seq_len],
                             tf.constant(0.0, dtype=query_vec.dtype))
      per_step_source_padding = tf.where(
          tf.less(tf.range(source_seq_len), tf.fill([source_seq_len], t + 1)),
          zero_padding, tf.ones_like(zero_padding, dtype=query_vec.dtype))
      query_batch_size = tf.shape(query_vec)[0]
      per_step_source_padding = tf.tile(
          tf.expand_dims(per_step_source_padding, axis=0),
          [query_batch_size, 1])
    else:
      per_step_source_padding = None
    ctx_vec, atten_prob, _ = self.atten.ComputeContextVectorWithCachedSource(
        theta.atten,
        extended_packed_src,
        query_vec,
        per_step_source_padding=per_step_source_padding)

    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    h = input_to_add + tf.reshape(ctx_vec, tf.shape(query_vec))

    new_states = py_utils.NestedMap(
        key=extended_packed_src.source_vecs,
        value=extended_packed_src.source_contexts)
    return h, atten_prob, new_states


class TransformerFeedForwardLayer(base_layer.BaseLayer):
  """Feed-forward, add and norm layer used by 'Attention Is All You Need'.

  This class implements the second sub-layer of Transformer Layer. First,
  input passes through a feed-forward neural network with one hidden layer and
  then projected back to the original input dimension to apply residual. Output
  of the layer, is then normalized using Layer Normalization.
  """

  @classmethod
  def Params(cls):
    p = super(TransformerFeedForwardLayer, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the layer input.')
    p.Define('output_dim', 0, 'Dimension of the layer output.')
    p.Define('hidden_dim', 0, 'Dimension of the hidden layer.')
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define('activation', 'RELU', 'Non-linearity.')
    p.Define('fflayer_tpl',
             layers.FeedForwardNet.Params().Set(activation=['RELU', 'NONE']),
             'Feed forward layer default params')
    p.Define(
        'res_proj_tpl', layers.ProjectionLayer.Params(),
        'Residual projection default params, used when input_dim != '
        'output_dim.')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, y) = (x + dropout(y)).')
    p.Define(
        'residual_dropout_tpl', layers.DropoutLayer.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define(
        'relu_dropout_prob', 0.0,
        'Probability at which we apply dropout to the hidden layer '
        'of feed-forward network.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerFeedForwardLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim
    assert p.hidden_dim

    with tf.variable_scope(p.name):
      # Initialize feed-forward layer
      params = p.fflayer_tpl.Copy()
      params.name = 'fflayer'
      params.input_dim = p.input_dim
      params.activation = [p.activation, 'NONE']
      if p.output_dim == 0:
        params.hidden_layer_dims = [p.hidden_dim, p.input_dim]
      else:
        params.hidden_layer_dims = [p.hidden_dim, p.output_dim]

        if p.output_dim != p.input_dim:
          pj = p.res_proj_tpl.Copy()
          pj.name = 'res_proj'
          pj.input_dim = p.input_dim
          pj.output_dim = p.output_dim
          pj.activation = 'NONE'
          self.CreateChild('res_proj_layer', pj)

      params.dropout = [
          params.dropout.cls.Params().Set(keep_prob=1.0 - p.relu_dropout_prob),
          params.dropout.cls.Params().Set(keep_prob=1.0)
      ]
      self.CreateChild('fflayer', params)

      # Initialize feed-forward layer norm
      params = p.ln_tpl.Copy()
      params.name = 'fflayer_ln'
      params.input_dim = p.input_dim
      self.CreateChild('layer_norm', params)

      dropout_tpl = p.residual_dropout_tpl.Copy()
      dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
      self.CreateChild('residual_dropout', dropout_tpl)

  def FProp(self, theta, inputs, paddings):
    """Feed-forward, residual and layer-norm.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: [time, batch, dim].
      paddings: [time, batch]

    Returns:
      tensor of the same shape with inputs
    """
    inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
    if hasattr(self, 'res_proj_layer'):
      inputs = self.res_proj_layer.FProp(theta.res_proj_layer, inputs)
    h = inputs + self.residual_dropout.FProp(
        theta.residual_dropout,
        self.fflayer.FProp(theta.fflayer, inputs_normalized,
                           tf.expand_dims(paddings, -1)))
    return h


class TransformerLayer(base_layer.BaseLayer):
  """Transformer Layer proposed by 'Attention Is All You Need'.

  Applies self-attention followed by a feed forward network and
  layer normalization. Uses residual connections between each consecutive
  layer. In particular, adds residuals from layer input and attention output
  and from attention output (feed-forward input) to feed-forward output.

  Implements the transformer block in 'Attention is All You Need':
  https://arxiv.org/abs/1706.03762.
  """

  @classmethod
  def Params(cls):
    p = super(TransformerLayer, cls).Params()
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define('output_dim', 0, 'Dimension of the transformer block output.')
    p.Define('tr_atten_tpl',
             TransformerAttentionLayer.Params().Set(num_attention_heads=8),
             'Transformer Attention Layer params.')
    p.Define('tr_fflayer_tpl',
             TransformerFeedForwardLayer.Params().Set(hidden_dim=2048),
             'Transformer Feed-Forward Layer params.')
    p.Define(
        'has_aux_atten', False,
        'If set, introduces a second attention layer, which attends to'
        ' the auxiliary source contexts.')
    p.Define('tr_aux_atten_tpl', None, 'Transformer Attention Layer params.')
    p.Define('mask_self_atten', False, 'If True, use masked self-attention.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define(
        'is_decoder', False, '(Deprecated) '
        'If true, forces both has_aux_atten and mask_self_atten to true.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim

    if p.is_decoder:
      tf.logging.warn('TransformerLayer.is_decoder is deprecated.')
      p.has_aux_atten = True
      p.mask_self_atten = True

    with tf.variable_scope(p.name):

      # Initialize multi-headed self-attention
      params = p.tr_atten_tpl.Copy()
      params.name = 'multihead_self_atten'
      params.source_dim = p.source_dim
      params.packed_input = p.packed_input
      params.is_masked = p.mask_self_atten
      self.CreateChild('self_atten', params)

      if p.has_aux_atten:
        # Initialize masked-multi-headed attention
        params = (
            p.tr_atten_tpl.Copy()
            if p.tr_aux_atten_tpl is None else p.tr_aux_atten_tpl.Copy())
        params.name = 'multihead_atten'
        params.source_dim = p.source_dim
        params.packed_input = p.packed_input
        self.CreateChild('atten', params)

      # Initialize feed-forward layer
      params = p.tr_fflayer_tpl.Copy()
      params.name = 'tr_fflayer'
      params.input_dim = p.source_dim
      params.output_dim = p.output_dim
      self.CreateChild('fflayer', params)

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            aux_vecs=None,
            aux_paddings=None,
            source_segment_id=None,
            aux_segment_id=None):
    """Transformer Layer.

    Transformer layer has the naming scheme as follows: `source_vecs` and
    `source_paddings` are all assumed to be coming from the activations of the
    layer below. When `TransformerLayer` is used in the Encoder (default
    behavior of this layer) `source_*` tensors correspond to the outputs of
    previous encoder layer. Further, keys, values and queries are all
    forked from `source_vecs`. When TransformerLayer is used in the Decoder
    (has_aux_atten=True), `source_*` tensors correspond to the outputs of
    previous decoder layer and used as the queries.

    For the cases when `TransformerLayer` is used in the decoder
    (has_aux_atten=True) `aux_*` tensors have to be provided.  Auxiliary inputs,
    `aux_*` tensors, are then correspond to the top-most layer encoder outputs
    and used by the second `TransformerAttentionLayer` as keys and values.

    Regardless of the encoder or decoder, queries are always assumed to be
    coming from the activations of layer below, in particular `source_vecs`.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_vecs: [source_time, source_batch, dim].
      source_paddings: [source_time, source_batch]
      aux_vecs: [aux_time, aux_batch, dim]
      aux_paddings: [aux_time, aux_batch]
      source_segment_id: [source_time, source_batch]
      aux_segment_id: [aux_time, aux_batch]

    Returns:
      The attention context vector, [source_time, source_batch, dim].

      The attention probability vector, [source_time, source_batch, source_time]
      if has_aux_atten is False, otherwise [source_time, source_batch,
      aux_time].
    """
    p = self.params
    if p.packed_input:
      assert source_segment_id is not None, ('Need to specify segment id for '
                                             'packed input.')

    if p.has_aux_atten:
      assert aux_vecs is not None
      assert aux_paddings is not None

    with tf.name_scope('self_atten'):
      atten_vec, atten_prob = self.self_atten.FProp(
          theta.self_atten,
          source_vecs,
          source_paddings,
          query_segment_id=source_segment_id)

    if p.has_aux_atten:
      with tf.name_scope('aux_atten'):
        atten_vec, atten_prob = self.atten.FProp(
            theta.atten, atten_vec, aux_paddings, aux_vecs, source_segment_id,
            aux_segment_id)

    h = self.fflayer.FProp(theta.fflayer, atten_vec, source_paddings)
    return h, atten_prob

  def ExtendStep(self,
                 theta,
                 source_vecs,
                 prefix_states,
                 aux_vecs=None,
                 aux_paddings=None,
                 t=None):
    """Transformer Layer, extend one step in decoding.

    This function is expected to be called during fast decoding of Transformer
    models.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_vecs: [source_batch, dim].
      prefix_states: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
      aux_vecs: [aux_time, aux_batch, dim]
      aux_paddings: [aux_time, aux_batch]
      t: a scalar, the current time step, 0-based.

    Returns:
      The attention context vector, [target_batch, source_dim]

      The attention probability vector, [source_time, target_batch]

      Updated prefix states
    """
    p = self.params

    if p.has_aux_atten:
      assert aux_vecs is not None
      assert aux_paddings is not None

    batch_size = tf.shape(source_vecs)[0]

    # First the self-attention layer.
    atten_vec, atten_prob, new_states = self.self_atten.ExtendStep(
        theta.self_atten, source_vecs, prefix_states, t)

    atten_vec = tf.expand_dims(atten_vec, axis=0)
    # Next the source attention layer.
    if p.has_aux_atten:
      atten_vec, atten_prob = self.atten.FProp(theta.atten, atten_vec,
                                               aux_paddings, aux_vecs)

    # Finally, the feedforward layer.
    h = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([1, batch_size], dtype=py_utils.FPropDtype(p)))
    h = tf.squeeze(h, 0)
    return h, atten_prob, new_states


class EvolvedTransformerEncoderBranchedConvsLayer(base_layer.BaseLayer):
  """Evolved Transformer encoder branched convolutions layer.

  This constructs the branched convolution portion of the Evolved Transformer
  encoder described in https://arxiv.org/abs/1901.11117 .
  """

  @classmethod
  def Params(cls):
    p = super(EvolvedTransformerEncoderBranchedConvsLayer, cls).Params()
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define('input_dim', 0, 'Dimension of the layer input.')
    p.Define('activation', 'RELU',
             'Activation applied after the left and right branches.')
    p.Define('dropout_tpl', layers.DropoutLayer.Params(),
             'Dropout applied to each layer output.')
    p.Define('dense_tpl', layers.FCLayer.Params(),
             'Fully connected "dense" layer.')
    p.Define('conv_tpl', layers.Conv2DLayer.Params(),
             'Standard convolution layer.')
    p.Define('separable_conv_tpl', layers.SeparableConv2DLayer.Params(),
             'Separable convolution layer.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(EvolvedTransformerEncoderBranchedConvsLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim

    with tf.variable_scope(p.name):
      # Initialize first layer norm.
      params = p.ln_tpl.Copy()
      params.name = 'first_layer_norm'
      params.input_dim = p.input_dim
      self.CreateChild('first_layer_norm', params)

      # Initialize second layer norm.
      params = p.ln_tpl.Copy()
      params.name = 'second_layer_norm'
      params.input_dim = p.input_dim * 4
      self.CreateChild('second_layer_norm', params)

      # Initialize dense layer.
      params = p.dense_tpl.Copy()
      params.name = 'dense_layer'
      params.input_dim = p.input_dim
      params.activation = p.activation
      params.output_dim = p.input_dim * 4
      self.CreateChild('dense_layer', params)

      # Initialize standard conv.
      params = p.conv_tpl.Copy()
      params.name = 'conv_layer'
      params.bias = True
      params.batch_norm = False
      params.activation = p.activation
      params.filter_stride = (1, 1)
      params.filter_shape = (3, 1, p.input_dim, int(p.input_dim / 2))
      self.CreateChild('conv_layer', params)

      # Initialize separable conv.
      params = p.separable_conv_tpl.Copy()
      params.name = 'separable_conv_layer'
      params.bias = True
      params.batch_norm = False
      params.activation = 'NONE'
      params.filter_stride = (1, 1)
      params.filter_shape = (9, 1, int(p.input_dim * 4), p.input_dim)
      self.CreateChild('separable_conv_layer', params)

      # Initialize dropout.
      dropout_tpl = p.dropout_tpl.Copy()
      self.CreateChild('dropout', dropout_tpl)

  def FProp(self, theta, inputs, paddings):
    inputs_normalized = self.first_layer_norm.FProp(theta.first_layer_norm,
                                                    inputs)

    left_branch = self.dense_layer.FProp(theta.dense_layer, inputs_normalized,
                                         tf.expand_dims(paddings, -1))
    left_branch = self.dropout.FProp(theta.dropout, left_branch)
    # Newly computed padding is discarded.
    right_branch = self.conv_layer.FProp(
        theta.conv_layer, tf.expand_dims(inputs_normalized, axis=2),
        paddings)[0]
    right_branch = tf.squeeze(right_branch, axis=2)
    right_branch = self.dropout.FProp(theta.dropout, right_branch)
    right_branch = tf.pad(
        right_branch,
        [[0, 0], [0, 0],
         [0, tf.shape(left_branch)[-1] - tf.shape(right_branch)[-1]]],
        constant_values=0)

    hidden_state = left_branch + right_branch

    hidden_state = self.second_layer_norm.FProp(theta.second_layer_norm,
                                                hidden_state)
    # Newly computed padding is discarded.
    hidden_state = self.separable_conv_layer.FProp(
        theta.separable_conv_layer, tf.expand_dims(hidden_state, axis=2),
        paddings)[0]
    hidden_state = tf.squeeze(hidden_state, axis=2)
    hidden_state = tf.pad(
        hidden_state, [[0, 0], [0, 0],
                       [0, tf.shape(inputs)[-1] - tf.shape(hidden_state)[-1]]],
        constant_values=0)
    hidden_state = self.dropout.FProp(theta.dropout, hidden_state)
    hidden_state += inputs

    return hidden_state


class EvolvedTransformerDecoderBranchedConvsLayer(base_layer.BaseLayer):
  """Evolved Transformer decoder branched convolutions layer.

  This constructs the branched convolution portion of the Evolved Transformer
  decoder described in https://arxiv.org/abs/1901.11117 .
  """

  @classmethod
  def Params(cls):
    p = super(EvolvedTransformerDecoderBranchedConvsLayer, cls).Params()
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define('input_dim', 0, 'Dimension of the layer input.')
    p.Define('activation', 'RELU',
             'Activation applied to the left convolution branch output.')
    p.Define('dropout_tpl', layers.DropoutLayer.Params(),
             'Dropout applied to each layer output.')
    p.Define('separable_conv_tpl',
             layers.SeparableConv2DLayer.Params().Set(causal_convolution=True),
             'Separable convolution layer.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(EvolvedTransformerDecoderBranchedConvsLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim

    with tf.variable_scope(p.name):
      # Initialize first layer norm.
      params = p.ln_tpl.Copy()
      params.name = 'first_layer_norm'
      params.input_dim = p.input_dim
      self.CreateChild('first_layer_norm', params)

      # Initialize second layer norm.
      params = p.ln_tpl.Copy()
      params.name = 'second_layer_norm'
      params.input_dim = p.input_dim * 2
      self.CreateChild('second_layer_norm', params)

      # Initialize separable conv.
      params = p.separable_conv_tpl.Copy()
      params.name = 'separable_conv_11x1_layer'
      params.bias = True
      params.batch_norm = False
      params.activation = p.activation
      params.filter_stride = (1, 1)
      params.filter_shape = (11, 1, p.input_dim, int(p.input_dim * 2))
      self.CreateChild('separable_conv_11x1_layer', params)

      # Initialize first separable conv.
      params = p.separable_conv_tpl.Copy()
      params.name = 'separable_conv_7x1_layer'
      params.bias = True
      params.batch_norm = False
      params.activation = 'NONE'
      params.filter_stride = (1, 1)
      params.filter_shape = (7, 1, p.input_dim, int(p.input_dim / 2))
      self.CreateChild('separable_conv_7x1_layer', params)

      # Initialize second separable conv.
      params = p.separable_conv_tpl.Copy()
      params.name = 'separable_conv_7x1_layer_2'
      params.bias = True
      params.batch_norm = False
      params.activation = 'NONE'
      params.filter_stride = (1, 1)
      params.filter_shape = (7, 1, int(p.input_dim * 2), p.input_dim)
      self.CreateChild('separable_conv_7x1_layer_2', params)

      # Initialize dropout.
      dropout_tpl = p.dropout_tpl.Copy()
      self.CreateChild('dropout', dropout_tpl)

  def FProp(self, theta, inputs, paddings):
    inputs_normalized = self.first_layer_norm.FProp(theta.first_layer_norm,
                                                    inputs)

    left_branch = self.separable_conv_11x1_layer.FProp(
        theta.separable_conv_11x1_layer,
        tf.expand_dims(inputs_normalized, axis=2), paddings)[0]
    left_branch = self.dropout.FProp(theta.dropout, left_branch)

    right_branch = self.separable_conv_7x1_layer.FProp(
        theta.separable_conv_7x1_layer, tf.expand_dims(
            inputs_normalized, axis=2), paddings)[0]
    right_branch = self.dropout.FProp(theta.dropout, right_branch)
    right_branch = tf.pad(
        right_branch,
        [[0, 0], [0, 0], [0, 0],
         [0, tf.shape(left_branch)[-1] - tf.shape(right_branch)[-1]]],
        constant_values=0)

    hidden_state = left_branch + right_branch
    hidden_state = self.second_layer_norm.FProp(theta.second_layer_norm,
                                                hidden_state)

    hidden_state = self.separable_conv_7x1_layer_2.FProp(
        theta.separable_conv_7x1_layer_2, hidden_state, paddings)[0]
    hidden_state = self.dropout.FProp(theta.dropout, hidden_state)

    hidden_state = tf.squeeze(hidden_state, axis=2)
    return hidden_state + inputs


class EvolvedTransformerEncoderLayer(base_layer.BaseLayer):
  """Evolved Transformer encoder layer.

  An Evolved Transformer encoder layer as described in
  https://arxiv.org/abs/1901.11117 .
  """

  @classmethod
  def Params(cls):
    p = super(EvolvedTransformerEncoderLayer, cls).Params()
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define('glu_tpl', layers.GluLayer.Params(), 'Glu layer.')
    p.Define('branched_convs_tpl',
             EvolvedTransformerEncoderBranchedConvsLayer.Params(),
             'Evolved Transformer branched convolutional layers.')
    p.Define('transformer_tpl', TransformerLayer.Params(), 'Transformer layer.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(EvolvedTransformerEncoderLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim

    with tf.variable_scope(p.name):

      # Initialize Glu layer.
      params = p.glu_tpl.Copy()
      params.name = 'glu_layer'
      params.input_dim = p.source_dim
      self.CreateChild('glu_layer', params)

      # Initialize branched convolutions layer.
      params = p.branched_convs_tpl.Copy()
      params.name = 'branched_convs_layer'
      params.input_dim = p.source_dim
      self.CreateChild('branched_convs_layer', params)

      # Initialize branched convolutional layers.
      params = p.transformer_tpl.Copy()
      params.name = 'transformer_layer'
      params.source_dim = p.source_dim
      params.output_dim = p.source_dim
      params.tr_fflayer_tpl.hidden_dim = 4 * p.source_dim
      # Decoder functionality is not supported so disable auxiliary attention.
      params.has_aux_atten = False
      params.tr_aux_atten_tpl = None
      params.mask_self_atten = False
      params.is_decoder = False
      self.CreateChild('transformer_layer', params)

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            aux_vecs=None,
            aux_paddings=None,
            source_segment_id=None,
            aux_segment_id=None):
    hidden_state = self.glu_layer.FProp(theta.glu_layer, source_vecs,
                                        source_paddings)

    hidden_state = self.branched_convs_layer.FProp(
        theta.branched_convs_layer, hidden_state, source_paddings)

    hidden_state, atten_prob = self.transformer_layer.FProp(
        theta.transformer_layer, hidden_state, source_paddings, aux_vecs,
        aux_paddings, source_segment_id, aux_segment_id)

    return hidden_state, atten_prob


class EvolvedTransformerDecoderLayer(base_layer.BaseLayer):
  """Evolved Transformer decoder layer.

  An Evolved Transformer decoder layer as described in
  https://arxiv.org/abs/1901.11117 .
  """

  @classmethod
  def Params(cls):
    p = super(EvolvedTransformerDecoderLayer, cls).Params()
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define('tr_atten_tpl',
             TransformerAttentionLayer.Params().Set(num_attention_heads=8),
             'Transformer attention layer params.')
    p.Define('tr_double_heads_atten_tpl',
             TransformerAttentionLayer.Params().Set(num_attention_heads=16),
             'Transformer double heads attention layer params.')
    p.Define('branched_convs_tpl',
             EvolvedTransformerDecoderBranchedConvsLayer.Params(),
             'Evolved Transformer branched convolutional layers.')
    p.Define('transformer_tpl', TransformerLayer.Params(), 'Transformer layer.')
    p.Define(
        'has_aux_atten', True,
        'If set, introduces a second attention layer, which attends to'
        ' the auxiliary source contexts.')
    p.Define('tr_aux_atten_tpl', None, 'Transformer Attention Layer params.')
    p.Define('mask_self_atten', False, 'If True, use masked self-attention.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(EvolvedTransformerDecoderLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim

    with tf.variable_scope(p.name):

      # Initialize multi-headed self-attention.
      params = p.tr_double_heads_atten_tpl.Copy()
      params.name = 'self_atten_double_heads'
      params.source_dim = p.source_dim
      params.is_masked = p.mask_self_atten
      # Packed input is not supported.
      params.packed_input = False
      self.CreateChild('self_atten_double_heads', params)

      if p.has_aux_atten:
        # Initialize masked-multi-headed encoder attention.
        params = (
            p.tr_aux_atten_tpl.Copy()
            if p.tr_aux_atten_tpl is not None else p.tr_atten_tpl.Copy())
        params.name = 'attend_to_encoder'
        params.source_dim = p.source_dim
        # Packed input is not supported.
        params.packed_input = False
        self.CreateChild('attend_to_encoder', params)

      # Initialize branched convolutional layers.
      params = p.branched_convs_tpl.Copy()
      params.name = 'branched_convs'
      params.input_dim = p.source_dim
      self.CreateChild('branched_convs', params)

      # Initialize transformer layer.
      params = p.transformer_tpl.Copy()
      params.name = 'transformer_layer'
      params.source_dim = p.source_dim
      params.output_dim = p.source_dim
      params.tr_fflayer_tpl.hidden_dim = 4 * p.source_dim
      params.tr_aux_atten_tpl = p.tr_aux_atten_tpl
      params.has_aux_atten = p.has_aux_atten
      params.mask_self_atten = p.mask_self_atten
      params.tr_fflayer_tpl.activation = 'SWISH'
      # Packed input is not supported.
      params.packed_input = False
      self.CreateChild('transformer_layer', params)

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            aux_vecs=None,
            aux_paddings=None,
            source_segment_id=None,
            aux_segment_id=None):
    p = self.params

    if p.has_aux_atten:
      assert aux_vecs is not None
      assert aux_paddings is not None

    with tf.name_scope('self_atten_double_heads'):
      left_branch, _ = self.self_atten_double_heads.FProp(
          theta.self_atten_double_heads,
          source_vecs,
          source_paddings,
          query_segment_id=source_segment_id)

    if p.has_aux_atten:
      with tf.name_scope('attend_to_encoder'):
        right_branch, _ = self.attend_to_encoder.FProp(
            theta.attend_to_encoder, source_vecs, aux_paddings, aux_vecs,
            source_segment_id, aux_segment_id)

      hidden_state = left_branch + right_branch + source_vecs
    else:
      hidden_state = left_branch + source_vecs

    hidden_state = self.branched_convs.FProp(theta.branched_convs, hidden_state,
                                             source_paddings)

    hidden_state, atten_prob = self.transformer_layer.FProp(
        theta.transformer_layer, hidden_state, source_paddings, aux_vecs,
        aux_paddings, source_segment_id, aux_segment_id)

    return hidden_state, atten_prob

  def ExtendStep(self,
                 theta,
                 source_vecs,
                 prefix_states,
                 aux_vecs=None,
                 aux_paddings=None,
                 t=None):
    """Evolved Transformer decoder layer, extended one step in decoding.

    This function is expected to be called during fast decoding of Evolved
    Transformer models.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_vecs: [source_batch, dim].
      prefix_states: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
      aux_vecs: [aux_time, aux_batch, dim]
      aux_paddings: [aux_time, aux_batch]
      t: a scalar, the current time step, 0-based.

    Returns:
      The attention context vector, [target_batch, source_dim].

      The attention probability vector, [source_time, target_batch].

      Updated prefix states.
    """
    p = self.params

    if p.has_aux_atten:
      assert aux_vecs is not None
      assert aux_paddings is not None

    inputs = tf.expand_dims(source_vecs, axis=0)
    new_states = prefix_states

    double_head_attention_states = prefix_states.double_head_attention_states
    # First the self-attention layer.
    (left_branch, _,
     double_head_attention_states) = self.self_atten_double_heads.ExtendStep(
         theta.self_atten_double_heads, source_vecs,
         double_head_attention_states, t)
    new_states.double_head_attention_states = double_head_attention_states
    left_branch = tf.expand_dims(left_branch, axis=0)

    hidden_state = left_branch + inputs

    # Next the source attention layer.
    if p.has_aux_atten:
      hidden_state += self.attend_to_encoder.FProp(
          theta.attend_to_encoder, inputs, aux_paddings, aux_vecs)[0]

    branched_convs_input = prefix_states.branched_convs_input
    branched_convs_input = tf.concat([branched_convs_input, hidden_state],
                                     axis=0)
    new_states.branched_convs_input = branched_convs_input
    # The receptive field of the branched convs is 17 and so we do not need
    # to consider inputs that come before that to compute the final position.
    # TODO(davidso): Create an ExtendStep method for branched_convs to make this
    # more efficient.
    inputs_length = tf.minimum(tf.shape(branched_convs_input)[0], 17)
    branched_convs_input = branched_convs_input[-inputs_length:, :, :]
    hidden_state = self.branched_convs.FProp(theta.branched_convs,
                                             branched_convs_input, None)

    transformer_layer_input = tf.squeeze(hidden_state[-1, :, :])
    transformer_layer_states = prefix_states.transformer_layer_states
    (hidden_state, atten_prob,
     transformer_layer_states) = self.transformer_layer.ExtendStep(
         theta.transformer_layer,
         transformer_layer_input,
         transformer_layer_states,
         aux_vecs=aux_vecs,
         aux_paddings=aux_paddings,
         t=t)

    new_states.transformer_layer_states = transformer_layer_states

    return hidden_state, atten_prob, new_states


class MergerLayer(base_layer.BaseLayer):
  """Merges a list of input tensors with various options into a single tensor.

  Implements a merger/combiner operator given a list of tensors. The merger
  operator outputs a single tensor with the following options (merger_op):

  - atten: Applies attention over the set of input tensors given query vector.
  - mean: Takes the mean of input tensors.
  - concat: Concatenates the input tensors over the last dimension.
  - sum: Sum up all the input tensors.
  - weighted_sum: Use learnt weights to combine input tensors.
  - gated_avg: Learnt input dependent gates are used to average tensors.

  This class is expected to be called by multi-source/multi-column models.
  """

  @classmethod
  def Params(cls):
    """Params for this MergerLayer class."""
    p = super(MergerLayer, cls).Params()
    p.Define('merger_op', None, 'How to merge input tensors.')
    p.Define('source_dim', 0, 'Number of source nodes.')
    p.Define('query_dim', 0, 'Number of query nodes.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    p.Define('attention_tpl', attention.AdditiveAttention.Params(),
             'Attention used by the merger layer when merger_op is atten.')
    p.Define(
        'pre_proj_input_dims', None,
        'If set, should be a list of depths for the tensors to be merged.'
        ' Setting this will result in a pre-projection to source_dim'
        ' before the merger.')
    p.Define('pre_proj_output_dim', 0, 'Depth to project all inputs to.')
    p.Define(
        'proj_tpl',
        layers.ProjectionLayer.Params().Set(
            batch_norm=False, weight_norm=True, has_bias=True),
        'Configs template for the projection layer.')
    p.Define('gated_avg_tpl', layers.GatedAverageLayer.Params(),
             'Configs template for the gated average layer.')
    p.Define('num_sources', 0, 'If merger_op=weighted_sum, then must specify '
             'num of sources.')
    return p

  # Merging operation keys supported by this layer.
  MERGER_OPS = ['mean', 'atten', 'concat', 'sum', 'weighted_sum', 'gated_avg']

  @base_layer.initializer
  def __init__(self, params):
    super(MergerLayer, self).__init__(params)
    p = self.params
    if not p.name:
      raise ValueError('Layer must have a specified name!')
    if p.merger_op not in set(self.MERGER_OPS):
      raise ValueError('Merger op must be one of: ', self.MERGER_OPS)

    if p.merger_op == 'atten':
      atten_params = p.attention_tpl.Copy()
      atten_params.source_dim = p.source_dim
      atten_params.query_dim = p.query_dim
      atten_params.hidden_dim = p.hidden_dim
      atten_params.dtype = p.dtype
      if atten_params.params_init is None:
        atten_params.params_init = py_utils.WeightInit.Gaussian(
            1. / math.sqrt(atten_params.source_dim + atten_params.query_dim))
      self.CreateChild('atten', atten_params)

    if p.pre_proj_input_dims:
      if not p.pre_proj_output_dim:
        raise ValueError('Output dim should be specified for projection.')
      pre_proj_params = []
      for i, pre_proj_dim in enumerate(p.pre_proj_input_dims):
        proj_p = p.proj_tpl.Copy()
        proj_p.name = 'merger_pre_proj_%d' % i
        proj_p.input_dim = pre_proj_dim
        proj_p.output_dim = p.pre_proj_output_dim
        pre_proj_params.append(proj_p)
      self.CreateChildren('pre_proj', pre_proj_params)

    if p.merger_op == 'weighted_sum':
      assert p.num_sources > 0, ('For merger_op=weighted_sum, must specify '
                                 'num_sources > 0.')
      params_init = py_utils.WeightInit.Constant(1.0 / p.num_sources)
      # Weights to be learned.
      pw = py_utils.WeightParams(
          shape=[p.num_sources],
          init=params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      with tf.variable_scope(p.name):
        _, self._sum_weight = py_utils.CreateVariable('sum_weight', pw)

    if p.merger_op == 'gated_avg':
      assert p.num_sources > 0, ('For merger_op=gated_avg, must specify '
                                 'num_sources > 0.')
      params = p.gated_avg_tpl.Copy()
      params.name = 'g_avg_merger'
      params.num_nodes = p.source_dim
      params.num_inputs = p.num_sources
      self.CreateChild('gated_average', params)

  def FProp(self, theta, inputs, query_vec=None):
    """Combines the list of input tensors into a single tensor.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A list of tensors of shape [..., hidden_dim] or [...,
        [pre_proj_input_dims[i]]] if pre_proj_input_dims is specified.
      query_vec: A tensor of shape [..., hidden_dim].

    Returns:
      A tensor of the same shape with input tensors.

    Raises:
      ValueError: p.merger_op is not defined.
    """
    p = self.params
    n_sources = len(inputs)

    if p.pre_proj_input_dims and len(p.pre_proj_input_dims) != n_sources:
      raise ValueError('pre_proj_input_dims must be specified for each input.')

    if n_sources == 1:
      return inputs[0]

    # Pre-projection operation.
    if p.pre_proj_input_dims:
      for i in range(n_sources):
        inputs[i] = self.pre_proj[i].FProp(theta.pre_proj[i], inputs[i])

    tensor_pairs = list(zip(inputs[:-1], inputs[1:]))
    if p.merger_op == 'mean':
      # Simply take the mean, all dims must match.
      with tf.control_dependencies([
          py_utils.assert_shape_match(tf.shape(t1), tf.shape(t2))
          for t1, t2 in tensor_pairs
      ]):
        output = tf.add_n(inputs) / n_sources

    elif p.merger_op == 'sum':
      # Sum up all sources, all dims must match.
      with tf.control_dependencies([
          py_utils.assert_shape_match(tf.shape(t1), tf.shape(t2))
          for t1, t2 in tensor_pairs
      ]):
        output = tf.add_n(inputs)

    elif p.merger_op == 'weighted_sum':
      # Weighted sum of all sources, all dims must match.
      # For weighted_sum, assume input is a list of rank 3 tensors
      inputs = tf.stack(inputs)
      inputs = py_utils.HasRank(inputs, 4)

      with tf.control_dependencies([
          py_utils.assert_shape_match(tf.shape(t1), tf.shape(t2))
          for t1, t2 in tensor_pairs
      ]):
        w = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(self._sum_weight, 1), 1), 1)
        w = tf.tile(
            w,
            [1,
             tf.shape(inputs)[1],
             tf.shape(inputs)[2],
             tf.shape(inputs)[3]])
        output = tf.reduce_sum(inputs * w, axis=0)

    elif p.merger_op == 'atten':
      # Apply attention over the concatenated tensor, all dims must match.
      with tf.control_dependencies([
          py_utils.assert_shape_match(tf.shape(t1), tf.shape(t2))
          for t1, t2 in tensor_pairs
      ]):
        inputs = tf.stack(inputs, axis=0)
        batch_size = tf.shape(inputs)[1]
        paddings = tf.zeros([n_sources, batch_size], dtype=inputs.dtype)
        self.atten.InitForSourcePacked(theta.atten, inputs, inputs, paddings)
        output, _, _ = self.atten.ComputeContextVector(
            theta.atten, tf.reshape(query_vec, [-1, p.query_dim]))

    elif p.merger_op == 'concat':
      # Concatenate over the last dim, all dims but last must match.
      with tf.control_dependencies([
          py_utils.assert_equal(tf.shape(t1)[:-1],
                                tf.shape(t2)[:-1]) for t1, t2 in tensor_pairs
      ]):
        output = tf.concat(inputs, axis=-1)

    elif p.merger_op == 'gated_avg':
      output = self.gated_average.FProp(theta.gated_average, inputs)

    else:
      raise ValueError('Unrecognized merge op!')

    return output


class StyleLayer(base_layer.BaseLayer):
  """A layer that performs weighted style emb lookup."""

  @classmethod
  def Params(cls):
    p = super(StyleLayer, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the input.')
    p.Define('output_dim', 0, 'Dimension of the output.')
    p.Define('num_styles', 0, 'Num of styles.')
    p.Define('num_heads', 4, 'Number of attention heads.')
    p.Define(
        'enable_ctx_post_proj', True,
        'If True, computed context is post projected into'
        ' ctx_post_proj_dim.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(StyleLayer, self).__init__(params)
    p = self.params
    assert p.num_styles > 0
    assert p.input_dim > 0
    assert p.output_dim > 0

    with tf.variable_scope(p.name):
      # The styles table.
      w_shape = [p.num_styles, 1, p.output_dim]
      w_init = py_utils.WeightInit.Gaussian(scale=1.0)
      w_pc = py_utils.WeightParams(
          shape=w_shape,
          init=w_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      self.CreateVariable('styles_w', w_pc)

      # Lastly the attention module.
      atten_p = attention.MultiHeadedAttention.Params().Set(
          source_dim=p.output_dim,
          context_dim=p.output_dim,
          hidden_dim=p.output_dim,
          query_dim=p.input_dim,
          ctx_post_proj_dim=p.output_dim,
          num_attention_heads=p.num_heads,
          use_source_vec_as_attention_value=False,
          enable_ctx_post_proj=p.enable_ctx_post_proj)
      self.CreateChild('atten', atten_p)

  def EmbLookup(self, theta, ids):
    """Looks up style embedding vectors for ids only for test purpose.

    Args:
      theta: Named tuple with the weight matrix for the embedding.
      ids: A rank-N int32 tensor.

    Returns:
      embs, A rank-(N+1) params.dtype tensor.
      embs[indices, :] is the embedding vector for ids[indices].
    """
    p = self.params
    # TODO(ngyuzh): call this function for virsualize big discrete table,
    # e.g. num_styles > 2^10.
    embs = tf.nn.embedding_lookup(theta.styles_w, tf.reshape(ids, [-1]))
    out_shape = tf.concat([tf.shape(ids), [p.output_dim]], 0)
    return tf.reshape(tf.nn.tanh(embs), out_shape)

  def StyleEmbFromProbs(self, theta, inp):
    """Look up style embedding based on feedin probabilities.

    Args:
      theta: params for this layer and its sub-layers.
      inp: attention probabilities of shape [batch_size, num_styles].

    Returns:
      style_emb - weighted combined style embedding based on inp.
    """
    p = self.params
    b_size = tf.shape(inp)[0]
    styles_w = tf.tile(tf.nn.tanh(theta.styles_w), [1, b_size, 1])
    styles_paddings = tf.zeros([p.num_styles, b_size],
                               dtype=py_utils.FPropDtype(p))
    atten_probs = tf.tile(tf.expand_dims(inp, 1), [1, p.num_heads, 1])
    atten_probs = tf.reshape(atten_probs, [-1, p.num_styles])
    packed_src = self.atten.InitForSourcePacked(theta.atten, styles_w, styles_w,
                                                styles_paddings)
    style_emb, _ = self.atten.ComputeContextVectorWithAttenProbs(
        theta.atten, packed_src.source_contexts, atten_probs)
    return style_emb

  def FProp(self, theta, inp):
    """Look up style embedding."""

    p = self.params
    b_size = tf.shape(inp)[0]
    styles_w = tf.tile(tf.nn.tanh(theta.styles_w), [1, b_size, 1])
    styles_paddings = tf.zeros([p.num_styles, b_size],
                               dtype=py_utils.FPropDtype(p))
    packed_src = self.atten.InitForSourcePacked(theta.atten, styles_w, styles_w,
                                                styles_paddings)
    style_emb, probs, _ = self.atten.ComputeContextVectorWithSource(
        theta.atten, packed_src, inp)
    # TODO(yonghui): Extract and return the attention probabilities.
    return style_emb, probs
