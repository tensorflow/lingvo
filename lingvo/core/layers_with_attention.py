# Lint as: python3
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

import lingvo.compat as tf
from lingvo.core import activations
from lingvo.core import attention
from lingvo.core import base_layer
from lingvo.core import gshard_builder
from lingvo.core import gshard_layers
from lingvo.core import gshard_utils
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import symbolic


class StochasticResidualLayer(base_layer.BaseLayer):
  """Stocahstic residual layer that randomly drop the residual branch.

  Originally proposed in "Deep Networks with Stochastic Depth" for ConvNets,
  https://arxiv.org/pdf/1603.09382.pdf
  """

  @classmethod
  def Params(cls):
    """Params for `StochasticResidualLayer`."""
    p = super().Params()
    p.Define('residual_weight', 1.0, 'Residual weight.')
    p.Define('survival_prob', 1.0,
             'Survival probability of the residual branch.')
    return p

  def _DropConnect(self, x):
    """Drop the entire residual layer with given survival probability."""
    if self.do_eval:
      return x

    # Compute tensor.
    batch_size = tf.shape(x)[0]
    random_tensor = self.params.survival_prob
    random_tensor += tf.random.uniform([batch_size], dtype=x.dtype)
    for _ in range(x.shape.rank - 1):
      random_tensor = tf.expand_dims(random_tensor, axis=-1)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = x / self.params.survival_prob * binary_tensor
    return output

  def FProp(self, theta, x, y):
    """Return combined inputs.

    Args:
      theta: weights defined in this layer.
      x: input tensor.
      y: input tensor to apply weight to.

    Returns:
      Added tensors.
    """
    return x + self.params.residual_weight * self._DropConnect(y)

  @classmethod
  def FPropMeta(cls, p, x, y):
    py_utils.CheckShapes((x, y))
    return py_utils.NestedMap(flops=x.num_elements() * 6, out_shapes=(x,))


class TransformerAttentionLayer(base_layer.BaseLayer):
  """Multi-headed attention, add and norm used by 'Attention Is All You Need'.

  This class implements the first sub-layer of Transformer Layer. Input is
  first processed using a multi-headed (self) attention. Output of the
  attention layer is combined with the residual connection. And the finally,
  output is normalized using Layer Normalization.

  Layer can be used in five scenarios:

  1. Multi-Headed Self-Attention, where attention keys (source vectors),
     attention values (context vectors) and queries come from the same previous
     layer output, `query_vec`. This is the general use case for encoder
     Transformer Layers.
  2. Masked Multi-Headed Self-Attention, where attention keys, attention values
     and queries all come from the same previous layer output, but rightward
     activations are masked to prevent information flow from future. This is the
     use case for decoder self-attention Transformer Layers. Can be activated by
     setting `is_masked` flag of this layer.
  3. Multi-Headed Attention, where attention keys and attention values
     `source_vecs`, are coming from a different source (output of the encoder)
     and queries `query_vec`, coming from the previous layer outputs (decoder).
     This corresponds to the standard attention mechanism, decoder attending the
     encoder outputs.
  4. Multi-Headed Attention, where attention values `context_vecs` are coming
     from a different source than queries and keys, e.g. for positional
     attention, where keys and queries are positional encodings and values are
     decoder states.
  5. Masked Multi-Headed Self-Attention, where attention keys, attention values
     and queries all come from the same previous layer output, but the
     activations for the current position are masked to reduce the impact of
     high self-similarity. This is the use case for non-autoregressive decoder
     self-attention Transformer Layers. Can be activated by setting `is_masked`
     flag of this layer and setting `mask_type="eye"`.
  6. Masked Multi-Headed Self-Attention, where attention keys, attention values
     and queries all come from the same previous layer output, but:
     . rightward activations are masked to prevent information flow from future.
     . leftward activations are also masked to prevent information flow from
     past tokens that are beyond the N-gram context [K-N+1, K-1] when predicting
     the target token in position K. This is the use case for decoder
     self-attention Transformer Layers in N-gram mode. Can be activated by
     setting `is_masked` flag of this layer, and setting both
     `mask_type="ngram"` and `mask_ngram_order=N-1` to use as context only the
     previous N-1 tokens (as expected for an N-gram model); for details and
     experimental results see https://arxiv.org/abs/2001.04589.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define('context_dim', 0, 'Dimension of the attention contexts.')
    p.Define('atten_hidden_dim', 0, 'Dimension of the attention hidden dim.')
    p.Define('num_attention_heads', 8, 'Number of attention heads.')
    p.Define('is_masked', False, 'If set, uses masked MultiHeadedAttention.')
    p.Define(
        'mask_ngram_order', 0, 'N-gram order, relevant only when'
        '`mask_type` is set to "ngram".')
    p.Define(
        'mask_type', 'future', 'Type of attention mask if `is_masked` is'
        'set. Either "future" for masking out attention to future'
        'positions or "eye" for masking out the token itself, or "ngram" for'
        'bounding the left context to the previous N-1 tokens, where N is set'
        'by `mask_ngram_order`.')
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params.')
    p.Define(
        'atten_tpl',
        attention.MultiHeadedAttention.Params().Set(
            use_source_vec_as_attention_value=False, enable_ctx_post_proj=True),
        'Multi-Headed Dot-Attention default params.')
    p.Define(
        'atten_dropout_prob', 0.0,
        'Probability at which we apply dropout to the attention probs. '
        'This practically drops memory values at random positions.')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, f(x)) = (x + dropout(f(x))).')
    p.Define(
        'residual_dropout_tpl', layers.DropoutLayer.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define('add_unnormalized_input', False, 'If set, uses unnormalized input '
             'in the residual add.')
    p.Define(
        'residual_function', None, 'When None (the default), use simple '
        'sum for the residual connection (output = x + f(x)). For example, can '
        'use layers.HighwaySkipLayer.Params() or layers.GatingLayer.Params() '
        'for gated residual add, where output is instead '
        'residual_function.FProp(x, f(x)).')
    p.Define(
        'pre_layer_norm', True, 'When True, layer norm is used before attention'
        'module, otherwise used after attention module which is consistent with'
        'Vaswani et al\'s paper')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim

    if not p.atten_hidden_dim:
      p.atten_hidden_dim = p.source_dim

    if not p.context_dim:
      p.context_dim = p.source_dim

    if p.is_masked:
      assert p.mask_type in ['future', 'eye', 'ngram']

    params = self._InitAttention(p.atten_tpl)
    self.CreateChild('atten', params)

    # Initialize attention layer norm
    params = p.ln_tpl.Copy()
    params.name = 'atten_ln'
    params.input_dim = p.source_dim
    self.CreateChild('layer_norm', params)

    dropout_tpl = p.residual_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.CreateChild('residual_dropout', dropout_tpl)

    if p.residual_function is not None:
      params = p.residual_function.Copy()
      params.input_dim = p.atten_hidden_dim
      self.CreateChild('residual_function', params)

  def _InitAttention(self, atten_tpl):
    p = self.params
    # Initialize multi-headed attention
    params = atten_tpl.Copy()
    params.name = 'multihead_atten'
    params.source_dim = p.source_dim
    params.query_dim = p.source_dim
    params.hidden_dim = p.atten_hidden_dim
    params.context_dim = p.context_dim
    params.ctx_post_proj_dim = p.source_dim
    params.num_attention_heads = p.num_attention_heads
    params.atten_dropout_prob = p.atten_dropout_prob
    params.packed_input = p.packed_input
    return params

  def _GetSourceLength(self, source_paddings):
    return py_utils.GetShape(source_paddings)[0]

  def FProp(self,
            theta,
            query_vec,
            source_paddings,
            source_vecs=None,
            query_segment_id=None,
            source_segment_id=None,
            context_vecs=None,
            **kwargs):
    """Transformer attention, residual and normalization layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [target_time, target_batch, dim]
      source_paddings: [source_time, source_batch]
      source_vecs: [source_time, source_batch, dim].
      query_segment_id: [target_time, target_batch]
      source_segment_id: [source_time, source_batch]
      context_vecs: [source_time, target_batch, dim]
      **kwargs: Can be optional params for the attention layer, eg. attention
        projection index tensor.

    Returns:
      (output, atten_probs). output is of shape [target_time, target_batch,
      context_dim], atten_probs is of shape [target_time, target_batch,
      source_time].
    """
    p = self.params
    unnormalized_query_vec = query_vec
    if p.pre_layer_norm:
      query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)

    if source_vecs is None:  # For self-attention: keys = queries.
      source_vecs = query_vec
      source_segment_id = query_segment_id

    if context_vecs is None:  # Inter/self-attention: keys = values/contexts.
      context_vecs = source_vecs

    target_time, target_bs, query_dim = py_utils.GetShape(query_vec, 3)
    if p.is_masked:
      assert source_vecs is not None
      query_vec = py_utils.with_dependencies([
          py_utils.assert_shape_match(
              tf.shape(source_vecs), tf.shape(query_vec))
      ], query_vec)
      # Prepares mask for self-attention
      # Padding is complemented, so time indexes that we want to mask out
      # receive padding weight 1.0.
      if p.mask_type == 'future':
        padding = py_utils.CausalSelfAttenPadding(
            target_time, dtype=py_utils.FPropDtype(p))
      elif p.mask_type == 'eye':
        padding = tf.eye(target_time, target_time, dtype=py_utils.FPropDtype(p))
      elif p.mask_type == 'ngram':  # Maybe apply N-gram mask.
        assert p.mask_ngram_order
        padding = 1.0 - tf.linalg.band_part(
            tf.ones([target_time, target_time], dtype=py_utils.FPropDtype(p)),
            tf.minimum(p.mask_ngram_order - 1, target_time - 1), 0)

      # [time,  batch, time]
      causal_padding = tf.tile(tf.expand_dims(padding, 1), [1, target_bs, 1])

      causal_padding = tf.reshape(causal_padding, [-1, target_time])
    else:
      causal_padding = None

    # Projects keys and values.
    packed_src = self.atten.PackSource(
        theta=theta.atten,
        source_vecs=source_vecs,  # keys
        source_contexts=context_vecs,  # values
        source_padding=source_paddings,
        source_segment_id=source_segment_id)

    if query_segment_id is not None:
      query_segment_id = tf.reshape(query_segment_id, [-1])

    ctx_vec, atten_prob, _ = self.atten.ComputeContextVectorWithSource(
        theta=theta.atten,
        packed_src=packed_src,
        query_vec=tf.reshape(query_vec, [-1, query_dim]),
        per_step_source_padding=causal_padding,
        query_segment_id=query_segment_id,
        **kwargs)

    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    input_after_sublayer = tf.reshape(
        ctx_vec,
        [
            target_time,
            target_bs,
            -1  # Either projected or not.
        ])
    if p.residual_function is None:
      h = input_to_add + input_after_sublayer
    else:
      h = self.residual_function.FProp(theta.residual_function, input_to_add,
                                       input_after_sublayer)
    if not p.pre_layer_norm:
      h = self.layer_norm.FProp(theta.layer_norm, h)
    atten_prob = tf.reshape(
        atten_prob,
        [target_time, target_bs,
         self._GetSourceLength(source_paddings)])
    return h, atten_prob

  def _FinishExtendStep(self,
                        theta,
                        query_vec,
                        unnormalized_query_vec,
                        extended_packed_src,
                        t=None):
    """Finish extending prefix by one more time step.

    Isolating this function from ExtendStep allows generalizing self-attention
    to causal attention on other inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [target_batch, dim]
      unnormalized_query_vec: [target_batch, dim]
      extended_packed_src: A `.NestedMap` object containing source_vecs,
        source_contexts, source_paddings, and source_segment_ids
      t: a scalar, the current time step, 0-based.

    Returns:
      A triplet (cur_output, atten_prob, new_state) where cur_output is a tensor
      representing the output from the current state, and new_state is the new
      state `.NestedMap`.
    """
    p = self.params

    # Compute per_step_source_padding. Padding is complemented, so time indexes
    # that we want to mask out receive padding weight 1.0.
    query_batch_size = py_utils.GetShape(query_vec)[0]
    source_seq_len = py_utils.GetShape(extended_packed_src.source_vecs)[0]
    zero_padding = tf.fill([source_seq_len],
                           tf.constant(0.0, dtype=query_vec.dtype))
    ones_padding = tf.ones_like(zero_padding, dtype=query_vec.dtype)
    if t is not None:
      per_step_source_padding = tf.where(
          tf.less(tf.range(source_seq_len), tf.fill([source_seq_len], t + 1)),
          zero_padding, ones_padding)
      per_step_source_padding = tf.tile(
          tf.expand_dims(per_step_source_padding, axis=0),
          [query_batch_size, 1])
    # Maybe apply N-gram masking.
    # TODO(ciprianchelba): As pointed out by miachen, to get the expected
    # speed-up we should go with per_step_source_padding=None here, and
    # everytime we update the prefix_states, we not only extend one step, but
    # also only keep the prefix_states for the most recent N steps instead of
    # the prefix states all the way from step 0.
    elif p.is_masked and p.mask_type == 'ngram':
      assert p.mask_ngram_order
      idx = tf.maximum(0, source_seq_len - p.mask_ngram_order)
      per_step_source_padding = tf.where(
          tf.less(tf.range(source_seq_len), tf.fill([source_seq_len], idx)),
          ones_padding, zero_padding)
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
    input_after_sublayer = tf.reshape(ctx_vec, py_utils.GetShape(query_vec))
    if p.residual_function is None:
      h = input_to_add + input_after_sublayer
    else:
      h = self.residual_function.FProp(theta.residual_function, input_to_add,
                                       input_after_sublayer)

    if not p.pre_layer_norm:
      h = self.layer_norm.FProp(theta.layer_norm, h)

    new_states = py_utils.NestedMap(
        key=extended_packed_src.source_vecs,
        value=extended_packed_src.source_contexts)
    return h, atten_prob, new_states

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
    if p.pre_layer_norm:
      query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)

    cached_packed_src = py_utils.NestedMap(
        source_vecs=prefix_state.key,
        source_contexts=prefix_state.value,
        source_padding=None,
        source_segment_id=None)
    extended_packed_src = self.atten.ExtendSourcePacked(theta.atten, query_vec,
                                                        query_vec, None, None,
                                                        cached_packed_src, t)
    return self._FinishExtendStep(theta, query_vec, unnormalized_query_vec,
                                  extended_packed_src, t)


class TransformerMultiSourceAttentionLayer(TransformerAttentionLayer):
  """Multi-source multi-headed attention.

  Only supports scenarios 3 and 4 in the base class. Now the two scenarios are:

  3. Multi-source multi-Headed Attention, where attention keys and attention
     values `source_vecs`, are different encodings and queries `query_vec`,
     coming from the previous layer outputs (decoder). In addition,
     attention keys and values are NestedMaps containing encodings of different
     sources. This corresponds to a multi-source decoder-to-encoder attention
     mechanism, i.e., decoder attends to encoder outputs and other sources.
  4. Similar to 3 but attention values `context_vecs` are coming from a
     different source than queries and keys.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_source', 0, 'Number of sources to attend to.')
    p.Define(
        'primary_source_index', 0, 'Index of the primary source whose '
        'attention probs will be returned.')
    p.Define('multi_source_atten', attention.MultiSourceAttention.Params(),
             'Multi-source attention params.')
    # Only used for case 3 and 4.
    p.is_masked = False
    return p

  def _InitAttention(self, atten_tpl):
    p = self.params
    source_atten_tpls = []
    # Set up each source attention.
    for i in range(p.num_source):
      src_key = 'source_%d' % i
      src_atten = atten_tpl.Copy()
      src_atten = super()._InitAttention(src_atten)
      src_atten.name = 'multihead_atten_%s' % src_key
      source_atten_tpls.append((src_key, src_atten))

    # Initialize multi-source attention.
    msa = p.multi_source_atten.Copy()
    msa.name = 'multi_source_atten'
    msa.source_dim = p.source_dim
    msa.query_dim = p.source_dim
    msa.source_atten_tpls = source_atten_tpls
    msa.primary_source_key = 'source_%d' % p.primary_source_index
    return msa

  def _GetSourceLength(self, source_paddings):
    return py_utils.GetShape(
        source_paddings['source_%d' % self.params.primary_source_index])[0]


class TransformerFeedForwardLayer(base_layer.BaseLayer):
  """Feed-forward, add and norm layer used by 'Attention Is All You Need'.

  This class implements the second sub-layer of Transformer Layer. First,
  input passes through a feed-forward neural network with one hidden layer and
  then projected back to the original input dimension to apply residual. Output
  of the layer, is then normalized using Layer Normalization.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the layer input.')
    p.Define('output_dim', 0, 'Dimension of the layer output.')
    p.Define('hidden_dim', 0, 'Dimension of the hidden layer.')
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define('activation', 'RELU', 'Non-linearity.')
    p.Define(
        'residual_weight', 1., 'Weight applied on residual connection.'
        'Final output is residual_weight * residual_fn(x) + x.'
        'Only effective when add_skip_connection is True.')
    p.Define('fflayer_tpl',
             layers.FeedForwardNet.Params().Set(activation=['RELU', 'NONE']),
             'Feed forward layer default params')
    p.Define(
        'res_proj_tpl',
        layers.ProjectionLayer.Params().Set(batch_norm=True),
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
    p.Define('add_skip_connection', True,
             'If True, add skip_connection from input to output.')
    p.Define('pre_layer_norm', True, 'Pre or post layer norm')
    p.Define('residual_droppath_prob', 0.0,
             'Probability at which we drop the entire residual path.')
    return p

  @classmethod
  def SetFPropDtype(cls, p, fprop_dtype):
    p.fprop_dtype = fprop_dtype
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim
    assert symbolic.ToStatic(p.hidden_dim) > 0

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

    if p.residual_droppath_prob > 0:
      assert p.add_skip_connection
      droppath_p = StochasticResidualLayer.Params().Set(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.CreateChild('residual_droppath', droppath_p)

  @property
  def output_dim(self):
    """Returns output dimension of the transformer layer."""
    return self.fflayer.output_dim

  @classmethod
  def NumOutputNodes(cls, p):
    return p.output_dim if p.output_dim else p.input_dim

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
    p = self.params
    with tf.name_scope(p.name):
      inputs = self._CastToFPropDtype(inputs)
      if self.params.pre_layer_norm:
        inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
      else:
        inputs_normalized = inputs
      if hasattr(self, 'res_proj_layer'):
        inputs = self.res_proj_layer.FProp(theta.res_proj_layer, inputs)
      h = self.residual_dropout.FProp(
          theta.residual_dropout,
          self.fflayer.FProp(theta.fflayer, inputs_normalized,
                             tf.expand_dims(paddings, -1)))
      if self.params.add_skip_connection:
        if p.residual_droppath_prob:
          h = self.residual_droppath.FProp(
              theta.residual_droppath,
              inputs,
              h,
          )
        else:
          h = inputs + h * self.params.residual_weight
      if not self.params.pre_layer_norm:
        h = self.layer_norm.FProp(theta.layer_norm, h)
      return h


class MoEFeedforwardLayer(base_layer.BaseLayer):
  """The feedforward net that is a MoE."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('moe_builder_p', gshard_builder.MoEBuilder.Params(),
             'A MoE builder params.')
    p.Define('fflayer_residual_weight', 0.5, 'fflayer residual weight.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    moe_builder = p.moe_builder_p.Instantiate()
    moe_layer_p = moe_builder.EncoderLayer(
        p.name,
        moe_builder.MoE(p.name),
        residual_weight=p.fflayer_residual_weight)
    self.CreateChild('moe_fflayer', moe_layer_p)

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
    p = self.params
    with tf.name_scope(p.name):
      # 0 - padded positions and 1 - non-padded positions.
      segment_ids = tf.cast(1. - paddings, tf.int32)
      segment_pos = tf.zeros_like(segment_ids)  # not used but required by MoE.
      moe_in = py_utils.NestedMap(
          vec=inputs, segment_id=segment_ids, segment_pos=segment_pos)
      moe_out = self.moe_fflayer.FProp(theta.moe_fflayer, moe_in)
      aux_loss_ctx = py_utils.AuxLossContext.Current()
      if aux_loss_ctx is None:
        raise ValueError('aux_loss_ctx should not be None.')
      aux_loss_ctx.AddLoss(moe_out.aux_loss)
      return moe_out.vec


class HybridFeedforwardLayer(base_layer.BaseLayer):
  """Hybrid Feedforward Net containing different fflayers, choosen by symbol.

  Example::

      hybrid_p = HybridFeedforwardLayer.Params()
      hybrid_p.sub = py_utils.NestedMap({'ff': fflayer_p, 'moe': moe_p})
      hybrid_p.sub_key = symbol
      hybrid_fflayer = HybridFeedforwardLayer(hybrid_p)
      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES,{symbol: 'moe'}):
        outputs_moe = hybrid_fflayer.FPropDefaultTheta(inputs, paddings)

  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'sub', None, 'A NestedMap of symbol to feedforward layers params. '
        'Each of them requires the NestedMap input outputs.')
    p.Define('sub_key', None,
             'A symbol to dynamically switch between sub layers.')
    return p

  @property
  def sub_key(self):
    p = self.params
    sub_key = p.sub_key
    if symbolic.IsExpr(p.sub_key):
      sub_key = symbolic.ToStatic(p.sub_key)
    return sub_key

  def __init__(self, params):
    super().__init__(params)
    for k, v in self.params.sub.items():
      self.CreateChild(k, v)

  def FProp(self, theta, inputs, paddings):
    p = self.params
    with tf.name_scope(p.name):
      return self.children[self.sub_key].FProp(theta[self.sub_key], inputs,
                                               paddings)


class TransformerShardedMoeLayer(base_layer.BaseLayer):
  """A sharded MOE layer.

  This is a drop-in replacement of the transformer feedforward layer. It is a
  composite of the following sub-layers.

  ln_inputs = ln(inputs)
  moe_output = moe(ln_inputs)
  drop_output = dropout(moe_output)
  output = inputs + drop_output
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the layer input.')
    p.Define('output_dim', 0, 'Dimension of the layer output.')
    p.Define('hidden_dim', 0, 'Dimension of the hidden layer.')
    # NOTE(yonghui): layer-norm as used in gshard doesn't have bias and assume
    # the mean is 0.0. See gshard_builder._LN for more details.
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define('activation', 'RELU', 'Non-linearity.')
    p.Define(
        'dropout_tpl', layers.DropoutLayer.Params(),
        'Dropout params template. keep_prob will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define('add_skip_connection', True,
             'If True, add skip_connection from input to output.')
    p.Define(
        'residual_weight', 1., 'Weight applied on residual connection.'
        'Final output is residual_weight * residual_fn(x) + x.'
        'Only effective when add_skip_connection is True.')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, y) = (x + dropout(y)).')
    p.Define(
        'relu_dropout_prob', 0.0,
        'Probability at which we apply dropout to the hidden layer '
        'of feed-forward network.')
    p.Define('pre_layer_norm', True, 'Pre or post layer norm')
    p.Define('residual_droppath_prob', 0.0,
             'Probability at which we drop the entire residual path.')
    p.Define('num_experts', 0, 'Total number of experts in this layer.')
    p.Define(
        'num_groups', 0,
        'Total number of groups for dispatching. num_groups typically'
        ' should be the same as num devices.')
    p.Define(
        'min_group_size', None,
        'If not None, num_groups will be adjusted so that there will be at '
        'least min_group_size tokens in each group.')
    p.Define(
        'expert_capacity_factor', 1.5,
        'Expert capacity factor. This should be set to a value greater'
        ' than or equal to 1.0. This is the ratio between max allowed'
        ' examples per expert over the average number of examples per '
        ' expert assuming routing is completely uniform.')
    p.Define(
        'expert_weight_shards', 1,
        'Shard each expert params into this many number of shards to reduce'
        ' the size of individual weight params.')
    p.Define('second_expert_policy', 'all',
             'How to pick second expert: all, sampling or random.')

    # SPMD partition related params.
    # M - model_dim, for both inputs and outputs.
    # E - experts dim
    # G - groups dim
    # C - experts capacity dim
    # H - hidden dim
    # S - sequence dim
    p.weight_split_dims_mapping = hyperparams.Params()
    wp = p.weight_split_dims_mapping
    wp.Define(
        'me', None, 'Sharding for the gating network weight, of shape'
        ' [input_dim, num_experts].')
    wp.Define(
        'emh', None, 'Sharding of the first projection matrix that maps from '
        ' input to hidden dim, of shape'
        ' [num_experts, input_dim, hidden_dim].')
    wp.Define(
        'ehm', None, 'Sharding of the second projection matrix that maps from '
        ' hidden to output dim, of shape'
        ' [num_experts, hidden_dim, output_dim].')

    p.activation_split_dims_mapping = hyperparams.Params()
    ap = p.activation_split_dims_mapping
    ap.Define('gsm', None, 'Sharding of the gsm tensors.')
    ap.Define('gs', None, 'Sharding of the gs tensors.')
    ap.Define('gsec', None, 'Sharding of the gsec tensors.')
    ap.Define('egcm', None, 'Sharding of the egcm tensors.')
    ap.Define('egch', None, 'Sharding of the egch tensors.')
    ap.Define('gecm', None, 'Sharding of the gecm tensors.')
    return p

  @classmethod
  def SetFPropDtype(cls, p, fprop_dtype):
    p.fprop_dtype = fprop_dtype
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim
    assert p.hidden_dim
    assert p.output_dim
    assert p.expert_capacity_factor >= 1.0
    assert p.num_experts > 0
    assert p.num_groups > 0

    # First create the gating network.
    wp = p.weight_split_dims_mapping
    stddev = (1. / p.input_dim)**0.5
    gate_scale = stddev * 3.**0.5
    gate_pc = py_utils.WeightParams(
        shape=[p.input_dim, p.num_experts],
        init=py_utils.WeightInit.Uniform(gate_scale),
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.me)
    self.CreateVariable('gate', gate_pc)

    # Next create the expert network.
    # Params initialization follows gshard_builder.py
    # emh tensor typically mesh-shard on first dim and last dim. Hence, here we
    # split the tensor manually into multiple tensors on the second dim.
    assert p.expert_weight_shards > 0
    emh_shape = [
        p.num_experts, p.input_dim // p.expert_weight_shards, p.hidden_dim
    ]
    stddev = (1. / p.input_dim)**0.5
    wi_init_scale = stddev * 3.**0.5
    wi_pc = py_utils.WeightParams(
        shape=emh_shape,
        init=py_utils.WeightInit.Uniform(wi_init_scale),
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.emh)

    for ii in range(p.expert_weight_shards):
      self.CreateVariable('wi_%d' % ii, wi_pc)

    # EHM Tensor (output transformation after RELU)
    # ehm tensor typically shard on the first dim and the second dim. Here we
    # manually split the tensor on the last dim into multiple tensors.
    ehm_shape = [
        p.num_experts, p.hidden_dim, p.output_dim // p.expert_weight_shards
    ]
    stddev = (1. / p.hidden_dim)**0.5
    wo_init_scale = stddev * 3.**0.5
    wo_pc = py_utils.WeightParams(
        shape=ehm_shape,
        init=py_utils.WeightInit.Uniform(wo_init_scale),
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.ehm)

    for ii in range(p.expert_weight_shards):
      self.CreateVariable('wo_%d' % ii, wo_pc)

    # TODO(yonghui): Possibly also add bias variables.

    # Initialize feed-forward layer norm
    params = p.ln_tpl.Copy()
    params.name = 'layer_norm'
    params.input_dim = p.input_dim
    self.CreateChild('layer_norm', params)

    dropout_tpl = p.dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.CreateChild('residual_dropout', dropout_tpl)

    dropout_tpl = p.dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.relu_dropout_prob)
    self.CreateChild('relu_dropout', dropout_tpl)

    if p.residual_droppath_prob > 0:
      assert p.add_skip_connection
      droppath_p = StochasticResidualLayer.Params().Set(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.CreateChild('residual_droppath', droppath_p)

  @property
  def output_dim(self):
    """Returns output dimension of the transformer layer."""
    return self.output_dim

  def FProp(self, theta, inputs, paddings):
    """Layer-norm, route, feed-forward, combine, residual.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: [batch, time, dim] or [batch, time, g, dim/g]. If the latter, the
        input is from some upstream component that applied the reshape trick to
        improve performance on tpu.
      paddings: [batch, time]

    Returns:
      tensor of the same shape with inputs
    """
    p = self.params
    ap = p.activation_split_dims_mapping

    assert inputs.shape.is_fully_defined()
    orig_shape = inputs.shape.as_list()
    assert len(orig_shape) == 3 or len(orig_shape) == 4
    if len(orig_shape) == 4:
      inputs = tf.reshape(inputs, orig_shape[:2] + [-1])

    with tf.name_scope(p.name):
      inputs = self._CastToFPropDtype(inputs)
      if p.pre_layer_norm:
        inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
      else:
        inputs_normalized = inputs
      inputs_normalized = py_utils.HasRank(inputs_normalized, 3)
      assert inputs_normalized.shape.is_fully_defined()
      bs, s_len, m_dim = py_utils.GetShape(inputs_normalized)
      paddings = py_utils.HasShape(paddings, [bs, s_len])
      num_groups = p.num_groups
      assert num_groups
      if (p.min_group_size is not None and
          bs * s_len / num_groups < p.min_group_size):
        num_groups = (bs * s_len + p.min_group_size - 1) // p.min_group_size
        tf.logging.info('num_groups adjusted to %s.' % num_groups)
      assert (bs * s_len) % num_groups == 0
      g_len = (bs * s_len) // num_groups
      reshaped_inputs = tf.reshape(inputs_normalized,
                                   [num_groups, g_len, m_dim])
      reshaped_paddings = tf.reshape(paddings, [num_groups, g_len])

      def _split(t_in, sharding):
        return gshard_utils.MeshSplit(t_in, p.device_mesh, sharding)

      # Sharding annotation.
      reshaped_inputs = _split(reshaped_inputs, ap.gsm)
      reshaped_paddings = _split(reshaped_paddings, ap.gs)

      fprop_dtype = py_utils.FPropDtype(p)
      logits = tf.einsum('gsm,me->gse', reshaped_inputs, theta.gate)

      # Here and below, we assume num devices equals num groups.
      # TODO(yonghui): Expose some of the options below through params.
      # NOTE(yonghui): The following code might break during beam search decode
      # due to much smaller group size.
      # TODO(yonghui): Avoid explicitly casting everything to fp32 once
      # Top2GatingOnLogits is stable in low-precision mode.
      aux_loss, combine_tensor, dispatch_tensor = gshard_layers.Top2GatingOnLogits(
          inputs=None,
          paddings=tf.cast(reshaped_paddings, tf.float32),
          logits=tf.cast(logits, tf.float32),
          num_devices=p.num_groups,
          experts_dim=p.num_experts,
          expert_capacity_dim=0,  # automatically decided.
          fprop_dtype=tf.float32,
          use_xla_sharding=False,
          second_expert_policy=p.second_expert_policy,
          second_expert_threshold=0.0,
          # legacy_mtf_behavior=True doesn't normalize gates when one expert is
          # being dropped. This is more appropriate for routing decisions like
          # 'random'.
          legacy_mtf_behavior=True,
          # x 2.0 because we choose top-2 experts per example.
          capacity_factor=2.0 * p.expert_capacity_factor)

      if fprop_dtype != tf.float32:
        aux_loss = tf.cast(aux_loss, fprop_dtype)
        dispatch_tensor = tf.cast(dispatch_tensor, fprop_dtype)
        combine_tensor = tf.cast(combine_tensor, fprop_dtype)

      # of shape [g, s, e, c]
      combine_tensor = _split(combine_tensor, ap.gsec)
      # of shape [g, s, e, c]
      dispatch_tensor = _split(dispatch_tensor, ap.gsec)

      theta_wis = []
      theta_wos = []
      for ii in range(p.expert_weight_shards):
        theta_wis.append(theta.get('wi_%d' % ii))
        theta_wos.append(theta.get('wo_%d' % ii))

      if len(theta_wis) == 1:
        theta_wi = theta_wis[0]
      else:
        theta_wi = tf.concat(theta_wis, 1)

      if len(theta_wos) == 1:
        theta_wo = theta_wos[0]
      else:
        theta_wo = tf.concat(theta_wos, 2)

      expert_inputs = _split(
          tf.einsum('gsec,gsm->egcm', dispatch_tensor, reshaped_inputs),
          ap.egcm)
      hidden = _split(
          tf.einsum('egcm,emh->egch', expert_inputs, theta_wi), ap.egch)
      # Activation function.
      hidden = activations.GetFn(p.activation)(hidden)
      # Dropout.
      hidden = self.relu_dropout.FProp(theta.relu_dropout, hidden)
      # Output
      expert_output = _split(
          tf.einsum('egch,ehm->egcm', hidden, theta_wo), ap.egcm)
      # Now transpose and reshard.
      transposed_expert_output = _split(
          tf.einsum('egcm->gecm', expert_output), ap.gecm)
      combined_output = _split(
          tf.einsum('gecm,gsec->gsm', transposed_expert_output, combine_tensor),
          ap.gsm)

      combined_output = tf.reshape(combined_output, [bs, s_len, p.output_dim])
      # Apply padding.
      combined_output *= tf.cast(1.0 - tf.expand_dims(paddings, -1),
                                 fprop_dtype)
      # Residual dropout.
      after_residual = self.residual_dropout.FProp(theta.residual_dropout,
                                                   combined_output)
      if p.add_skip_connection:
        if p.residual_droppath_prob:
          out = self.residual_droppath.FProp(
              theta.residual_droppath,
              inputs,
              after_residual,
          )
        else:
          out = inputs + after_residual * self.params.residual_weight

      if not p.pre_layer_norm:
        out = self.layer_norm.FProp(theta.layer_norm, out)

      if len(orig_shape) == 4:
        out = tf.reshape(out, orig_shape)

      # Add loss to a global collection. We don't return the loss to the caller
      # to avoid the change of the api here.
      aux_loss_ctx = py_utils.AuxLossContext.Current()
      if aux_loss_ctx is not None:
        aux_loss_ctx.AddLoss(aux_loss)

      return out


# TODO(shibow/wangtao) remove this after b/174094694 is done.
class ReshapedTransformerFeedForwardLayer(TransformerFeedForwardLayer):
  """TransformerFeedForward with model dim D reshaped as Md."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.ln_tpl = layers.ReshapedLayerNorm.Params()
    return p

  def FProp(self, theta, inputs, paddings):
    """Feed-forward, residual and layer-norm.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: [time, batch, dim_reshape_segments, dim // dim_reshape_segments].
        If a 3D tensor [time, batch, dim], the input (resp. output) rank is
        first augmented (resp. reduced) by splitting the last dimension
        according to the device_mesh (resp. merging the last two dimensions).
      paddings: [time, batch].

    Returns:
      tensor of the same shape with inputs.
    """
    p = self.params
    with tf.name_scope(p.name):
      inputs_shape = py_utils.GetShape(inputs)
      do_reshape = len(inputs_shape) == 3
      if do_reshape:
        inputs = gshard_utils.ReshapeDim(inputs, 2, p.device_mesh.shape[1])
      inputs = self._CastToFPropDtype(inputs)
      if self.params.pre_layer_norm:
        inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
      else:
        inputs_normalized = inputs
      if hasattr(self, 'res_proj_layer'):
        inputs = self.res_proj_layer.FProp(theta.res_proj_layer, inputs)

      theta.fflayer.fc[0].w = gshard_utils.ReshapeDim(theta.fflayer.fc[0].w, 0,
                                                      p.device_mesh.shape[1])
      theta.fflayer.fc[1].w = gshard_utils.ReshapeDim(theta.fflayer.fc[1].w, 1,
                                                      p.device_mesh.shape[1])
      if theta.fflayer.fc[1].b is not None:
        theta.fflayer.fc[1].b = gshard_utils.ReshapeDim(theta.fflayer.fc[1].b,
                                                        0,
                                                        p.device_mesh.shape[1])

      linear_paddings0 = tf.expand_dims(paddings, -1)
      linear_paddings1 = tf.expand_dims(linear_paddings0, -1)

      linear_out0 = tf.einsum('BLMd,MdH->BLH', inputs_normalized,
                              theta.fflayer.fc[0].w)
      if theta.fflayer.fc[0].b is not None:
        linear_out0 += theta.fflayer.fc[0].b
      linear_out0 = gshard_utils.MeshSplit(
          linear_out0, p.device_mesh,
          p.fflayer_tpl.activation_split_dims_mapping_list[0])
      linear_out0 = activations.GetFn(p.activation)(linear_out0)
      linear_out0 = py_utils.ApplyPadding(linear_paddings0, linear_out0)
      linear_out0 = self.fflayer.dropout[0].FProp(theta.fflayer.dropout[0],
                                                  linear_out0)
      linear_out1 = tf.einsum('BLH,HMd->BLMd', linear_out0,
                              theta.fflayer.fc[1].w)
      if theta.fflayer.fc[1].b is not None:
        linear_out1 += theta.fflayer.fc[1].b
      linear_out1 = gshard_utils.MeshSplit(
          linear_out1, p.device_mesh,
          p.fflayer_tpl.activation_split_dims_mapping_list[1])
      linear_out1 = py_utils.ApplyPadding(linear_paddings1, linear_out1)
      linear_out1 = self.fflayer.dropout[1].FProp(theta.fflayer.dropout[1],
                                                  linear_out1)

      h = self.residual_dropout.FProp(theta.residual_dropout, linear_out1)
      if self.params.add_skip_connection:
        h = inputs + h * self.params.residual_weight
      if not self.params.pre_layer_norm:
        h = self.layer_norm.FProp(theta.layer_norm, h)
      if do_reshape:
        shape = py_utils.GetShape(h, 2) + [-1]
        h = tf.reshape(h, shape)
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
    p = super().Params()
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define('output_dim', 0, 'Dimension of the transformer block output.')
    p.Define('tr_atten_tpl',
             TransformerAttentionLayer.Params().Set(num_attention_heads=8),
             'Transformer Attention Layer params.')
    p.Define('tr_post_ln_tpl', None,
             '(Optional) Layer norm at end of transformer layer.')
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
    p.Define(
        'num_aux_atten_post_proj', 1, 'Number of post projections for aux '
        'attention. This is usually used in multi-task setting, in which '
        'each task uses one dedicated projection layer.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim

    if p.is_decoder:
      tf.logging.warning('TransformerLayer.is_decoder is deprecated.')
      p.has_aux_atten = True
      p.mask_self_atten = True

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
      if hasattr(params.atten_tpl, 'num_post_proj'):
        params.atten_tpl.num_post_proj = p.num_aux_atten_post_proj
      self.CreateChild('atten', params)

    # Initialize feed-forward layer
    params = p.tr_fflayer_tpl.Copy()
    params.name = 'tr_fflayer'
    params.input_dim = p.source_dim
    params.output_dim = p.output_dim
    self.CreateChild('fflayer', params)

    # Initialize output layer norm
    if p.tr_post_ln_tpl:
      params = p.tr_post_ln_tpl.Copy()
      params.name = 'tr_post_layer_norm'
      params.input_dim = p.source_dim
      self.CreateChild('layer_norm', params)

  @property
  def output_dim(self):
    """Returns output dimension of the transformer layer."""
    # output_dim is equal to p.source_dim when p.output_dim is zero.
    return self.fflayer.output_dim

  @classmethod
  def NumOutputNodes(cls, p):
    return p.output_dim if p.output_dim else p.source_dim

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            aux_vecs=None,
            aux_paddings=None,
            source_segment_id=None,
            aux_segment_id=None,
            **kwargs):
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
      **kwargs: Can be optional params for the attention layer, eg. attention
        projection index tensor.

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

    with tf.name_scope('self_atten'):
      atten_vec, atten_prob = self.self_atten.FProp(
          theta.self_atten,
          source_vecs,
          source_paddings,
          query_segment_id=source_segment_id)

    if p.has_aux_atten:
      assert aux_vecs is not None
      assert aux_paddings is not None
      with tf.name_scope('aux_atten'):
        atten_vec, atten_prob = self.atten.FProp(theta.atten, atten_vec,
                                                 aux_paddings, aux_vecs,
                                                 source_segment_id,
                                                 aux_segment_id, **kwargs)

    with tf.name_scope('fflayer'):
      h = self.fflayer.FProp(theta.fflayer, atten_vec, source_paddings)
    if p.tr_post_ln_tpl:
      with tf.name_scope('layer_norm'):
        h = self.layer_norm.FProp(theta.layer_norm, h)
    return h, atten_prob

  def ExtendStep(self,
                 theta,
                 source_vecs,
                 prefix_states,
                 aux_vecs=None,
                 aux_paddings=None,
                 t=None,
                 **kwargs):
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
      **kwargs: Can be optional params for the attention layer, eg. attention
        projection index tensor.

    Returns:
      The attention context vector, [target_batch, source_dim]

      The attention probability vector, [source_time, target_batch]

      Updated prefix states
    """
    p = self.params

    if p.has_aux_atten:
      assert aux_vecs is not None
      assert aux_paddings is not None

    batch_size = py_utils.GetShape(source_vecs)[0]

    # First the self-attention layer.
    atten_vec, atten_prob, new_states = self.self_atten.ExtendStep(
        theta.self_atten, source_vecs, prefix_states, t)

    atten_vec = tf.expand_dims(atten_vec, axis=0)
    # Next the source attention layer.
    if p.has_aux_atten:
      atten_vec, atten_prob = self.atten.FProp(theta.atten, atten_vec,
                                               aux_paddings, aux_vecs, **kwargs)

    # Finally, the feedforward layer.
    h = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([1, batch_size], dtype=py_utils.FPropDtype(p)))
    if p.tr_post_ln_tpl:
      h = self.layer_norm.FProp(theta.layer_norm, h)
    h = tf.squeeze(h, 0)
    return h, atten_prob, new_states


class EvolvedTransformerEncoderBranchedConvsLayer(base_layer.BaseLayer):
  """Evolved Transformer encoder branched convolutions layer.

  This constructs the branched convolution portion of the Evolved Transformer
  encoder described in https://arxiv.org/abs/1901.11117 .
  """

  @classmethod
  def Params(cls):
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim

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
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim

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
        theta.separable_conv_7x1_layer,
        tf.expand_dims(inputs_normalized, axis=2), paddings)[0]
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


class EvolvedTransformerBaseLayer(base_layer.BaseLayer):
  """Base layer for the Evolved Transformer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define(
        'has_aux_atten', False,
        'If set, introduces a second attention layer, which attends to'
        ' the auxiliary source contexts.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    return p


class EvolvedTransformerEncoderLayer(EvolvedTransformerBaseLayer):
  """Evolved Transformer encoder layer.

  An Evolved Transformer encoder layer as described in
  https://arxiv.org/abs/1901.11117 .
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('glu_tpl', layers.GluLayer.Params(), 'Glu layer.')
    p.Define('branched_convs_tpl',
             EvolvedTransformerEncoderBranchedConvsLayer.Params(),
             'Evolved Transformer branched convolutional layers.')
    p.Define('transformer_tpl', TransformerLayer.Params(), 'Transformer layer.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim
    # Auxiliary attention not supported.
    if p.has_aux_atten:
      raise ValueError('Auxiliary attention not supported.')

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
    params.packed_input = p.packed_input
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

    hidden_state = tf.transpose(hidden_state, [1, 0, 2])
    source_paddings = tf.transpose(source_paddings, [1, 0])
    hidden_state = self.branched_convs_layer.FProp(theta.branched_convs_layer,
                                                   hidden_state,
                                                   source_paddings)
    hidden_state = tf.transpose(hidden_state, [1, 0, 2])
    source_paddings = tf.transpose(source_paddings, [1, 0])

    hidden_state, atten_prob = self.transformer_layer.FProp(
        theta.transformer_layer, hidden_state, source_paddings, aux_vecs,
        aux_paddings, source_segment_id, aux_segment_id)

    return hidden_state, atten_prob


class EvolvedTransformerDecoderLayer(EvolvedTransformerBaseLayer):
  """Evolved Transformer decoder layer.

  An Evolved Transformer decoder layer as described in
  https://arxiv.org/abs/1901.11117 .
  """

  @classmethod
  def Params(cls):
    p = super().Params()
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
    p.Define('tr_aux_atten_tpl', None, 'Transformer Attention Layer params.')
    p.Define('mask_self_atten', False, 'If True, use masked self-attention.')
    p.has_aux_atten = True
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim

    # Initialize multi-headed self-attention.
    params = p.tr_double_heads_atten_tpl.Copy()
    params.name = 'self_atten_double_heads'
    params.source_dim = p.source_dim
    params.is_masked = p.mask_self_atten
    # Packed input is not supported.
    params.packed_input = p.packed_input
    self.CreateChild('self_atten_double_heads', params)

    if p.has_aux_atten:
      # Initialize masked-multi-headed encoder attention.
      params = (
          p.tr_aux_atten_tpl.Copy()
          if p.tr_aux_atten_tpl is not None else p.tr_atten_tpl.Copy())
      params.name = 'attend_to_encoder'
      params.source_dim = p.source_dim
      # Packed input is not supported.
      params.packed_input = p.packed_input
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
    params.packed_input = p.packed_input
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
        right_branch, _ = self.attend_to_encoder.FProp(theta.attend_to_encoder,
                                                       source_vecs,
                                                       aux_paddings, aux_vecs,
                                                       source_segment_id,
                                                       aux_segment_id)

      hidden_state = left_branch + right_branch + source_vecs
    else:
      hidden_state = left_branch + source_vecs

    hidden_state = tf.transpose(hidden_state, [1, 0, 2])
    source_paddings = tf.transpose(source_paddings, [1, 0])
    hidden_state = self.branched_convs.FProp(theta.branched_convs, hidden_state,
                                             source_paddings)
    hidden_state = tf.transpose(hidden_state, [1, 0, 2])
    source_paddings = tf.transpose(source_paddings, [1, 0])

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
      hidden_state += self.attend_to_encoder.FProp(theta.attend_to_encoder,
                                                   inputs, aux_paddings,
                                                   aux_vecs)[0]

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
    branched_convs_input = tf.transpose(branched_convs_input, [1, 0, 2])
    hidden_state = self.branched_convs.FProp(theta.branched_convs,
                                             branched_convs_input, None)
    hidden_state = tf.transpose(hidden_state, [1, 0, 2])

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


class StyleLayer(base_layer.BaseLayer):
  """A layer that performs weighted style emb lookup."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the input.')
    p.Define('output_dim', 0, 'Dimension of the output.')
    p.Define('num_styles', 0, 'Num of styles.')
    p.Define('num_heads', 4, 'Number of attention heads.')
    p.Define(
        'enable_ctx_post_proj', True,
        'If True, computed context is post projected into'
        ' ctx_post_proj_dim.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.num_styles > 0
    assert p.input_dim > 0
    assert p.output_dim > 0

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

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    # The styles table.
    w_shape = [p.num_styles, 1, p.output_dim]
    w_init = py_utils.WeightInit.Gaussian(scale=1.0, seed=p.random_seed)
    w_pc = py_utils.WeightParams(
        shape=w_shape,
        init=w_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('styles_w', w_pc)

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


class TransformerLayerWithMultitaskAdapters(TransformerLayer):
  """Transformer Layer with multitask residual adapters.

  Applies transformer layer, followed by multitask adapters. Requires an
  additional input specifying the task_id for each input.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('adapter_tpl', layers.MultitaskAdapterLayer.Params(),
             'Template to use for multitask adapters.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    params = p.adapter_tpl.Copy()
    params.name = 'adapters'
    self.CreateChild('adapters', params)

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            aux_vecs=None,
            aux_paddings=None,
            source_segment_id=None,
            aux_segment_id=None,
            source_task_id=None):
    """Transformer Layer with multitask adapters.

    First applies the standard transformer layer. Then applies adapter layers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_vecs: [source_time, source_batch, dim].
      source_paddings: [source_time, source_batch]
      aux_vecs: [aux_time, aux_batch, dim]
      aux_paddings: [aux_time, aux_batch]
      source_segment_id: [source_time, source_batch]
      aux_segment_id: [aux_time, aux_batch]
      source_task_id: [source_time, source_batch]

    Returns:
      The attention context vector, [source_time, source_batch, dim].

      The attention probability vector, [source_time, source_batch, source_time]
      if has_aux_atten is False, otherwise [source_time, source_batch,
      aux_time].
    """
    p = self.params
    hidden, atten_prob = super().FProp(theta, source_vecs, source_paddings,
                                       aux_vecs, aux_paddings,
                                       source_segment_id, aux_segment_id)
    # Assumes the same task_id for the entire sequence during eval or when
    # not using packed_input.
    if not p.packed_input and not self.do_eval:
      source_task_id = source_task_id[0, :]
    hidden = self.adapters.FProp(theta.adapters, hidden, source_task_id)
    return hidden, atten_prob

  def ExtendStep(self,
                 theta,
                 source_vecs,
                 prefix_states,
                 aux_vecs=None,
                 aux_paddings=None,
                 timestep=None,
                 source_task_id=None):
    """Transformer Layer with adapters, extend one step in decoding.

    Applies TransformerLayer.ExtendStep, then applies adapters.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_vecs: [source_batch, dim].
      prefix_states: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
      aux_vecs: [aux_time, aux_batch, dim]
      aux_paddings: [aux_time, aux_batch]
      timestep: a scalar, the current time step, 0-based.
      source_task_id: [source_batch]

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
        theta.self_atten, source_vecs, prefix_states, timestep)

    atten_vec = tf.expand_dims(atten_vec, axis=0)
    # Next the source attention layer.
    if p.has_aux_atten:
      atten_vec, atten_prob = self.atten.FProp(theta.atten, atten_vec,
                                               aux_paddings, aux_vecs)

    # Finally, the feedforward layer.
    hidden = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([1, batch_size], dtype=py_utils.FPropDtype(p)))

    # Now adapter layers.
    hidden = self.adapters.FProp(theta.adapters, hidden, source_task_id)

    hidden = tf.squeeze(hidden, 0)
    return hidden, atten_prob, new_states


# TODO(ankurbpn): Implementation is slightly different from the original.
# In the original implementation the KV projection outputs were explicitly
# zeroed out by the gating networks. Here we control the inputs instead.
# Verify if this still works as well as the original implementation.
class CCTAttentionLayer(base_layer.BaseLayer):
  """Multi-headed attention, add and norm used by 'Attention Is All You Need'.

  Supports CCT attention gating as in the paper here:
  https://arxiv.org/abs/2002.07106
  """

  @classmethod
  def Params(cls):
    p = super().Params()

    # Transformer Attention params.
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define('context_dim', 0, 'Dimension of the attention contexts.')
    p.Define('atten_hidden_dim', 0, 'Dimension of the attention hidden dim.')
    p.Define('num_attention_heads', 8, 'Number of attention heads.')
    p.Define('is_masked', False, 'If set, uses masked MultiHeadedAttention.')
    p.Define(
        'mask_type', 'future', 'Type of attention mask if `is_masked` is'
        'set. Either "future" for masking out attention to future'
        'positions or "eye" for masking out the token itself.')
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

    # CCT params.
    p.Define('gating_tpl', layers.CCTGatingNetwork.Params(), '')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.source_dim

    if not p.atten_hidden_dim:
      p.atten_hidden_dim = p.source_dim

    if not p.context_dim:
      p.context_dim = p.source_dim

    if p.is_masked:
      assert p.mask_type in ['future', 'eye']

    # Initialize multi-headed attention
    params = p.atten_tpl.Copy()
    params.name = 'multihead_atten'
    params.source_dim = p.source_dim
    params.query_dim = p.source_dim
    params.hidden_dim = p.atten_hidden_dim
    params.context_dim = p.context_dim
    params.ctx_post_proj_dim = p.source_dim
    params.num_attention_heads = p.num_attention_heads
    params.atten_dropout_prob = p.atten_dropout_prob
    params.packed_input = p.packed_input
    self.CreateChild('atten', params)

    dropout_tpl = p.residual_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.CreateChild('residual_dropout', dropout_tpl)

    # Initialize attention layer norm
    params = p.ln_tpl.Copy()
    params.name = 'atten_ln'
    params.input_dim = p.source_dim
    self.CreateChild('layer_norm', params)

    # CCT specific operations.
    ff_gating = p.gating_tpl.Copy()
    ff_gating.input_dim = p.source_dim
    ff_gating.num_outputs = 1
    ff_gating.name = 'query_gating_net'
    self.CreateChild('query_gating', ff_gating)

    ff_gating = p.gating_tpl.Copy()
    ff_gating.input_dim = p.source_dim
    ff_gating.num_outputs = 1
    ff_gating.name = 'kv_gating_net'
    self.CreateChild('kv_gating', ff_gating)

    # Initialize source_vec layer norm
    params = p.ln_tpl.Copy()
    params.name = 'source_ln'
    params.input_dim = p.source_dim
    self.CreateChild('source_layer_norm', params)

    # Initialize ctx_vec layer norm
    params = p.ln_tpl.Copy()
    params.name = 'ctx_ln'
    params.input_dim = p.source_dim
    self.CreateChild('ctx_layer_norm', params)

  def FProp(self,
            theta,
            query_vec,
            source_paddings,
            source_vecs=None,
            query_segment_id=None,
            source_segment_id=None,
            **kwargs):
    """CCT attention, residual and normalization layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [target_time, target_batch, dim]
      source_paddings: [source_time, source_batch]
      source_vecs: [source_time, source_batch, dim].
      query_segment_id: [target_time, target_batch]
      source_segment_id: [source_time, source_batch]
      **kwargs: Can be optional params for the attention layer, eg. attention
        projection index tensor.

    Returns:
      (output, atten_probs). output is of shape [target_time, target_batch,
      context_dim], atten_probs is of shape [target_time, target_batch,
      source_time].
    """
    p = self.params
    unnormalized_query_vec = query_vec
    query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)

    if source_vecs is None:  # For self-attention: keys = queries.
      source_vecs = query_vec
      source_segment_id = query_segment_id
    else:
      source_vecs = self.source_layer_norm.FProp(theta.source_layer_norm,
                                                 source_vecs)

    # Gating the query computation.
    query_p_c = self.query_gating.FProp(theta.query_gating, query_vec)
    source_p_c = self.kv_gating.FProp(theta.kv_gating, source_vecs)
    source_vecs *= source_p_c  # Gate the source vectors.

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

      if p.mask_type == 'future':
        padding = 1.0 - tf.linalg.band_part(
            tf.ones([target_time, target_time], dtype=py_utils.FPropDtype(p)),
            -1, 0)
      elif p.mask_type == 'eye':
        padding = tf.eye(target_time, target_time, dtype=py_utils.FPropDtype(p))

      # [time,  batch, time]
      causal_padding = tf.tile(tf.expand_dims(padding, 1), [1, target_bs, 1])

      causal_padding = tf.reshape(causal_padding, [-1, target_time])
    else:
      causal_padding = None

    query_dim = tf.shape(query_vec)[-1]

    # Projects keys and values.
    packed_src = self.atten.PackSource(
        theta=theta.atten,
        source_vecs=source_vecs,  # keys
        source_contexts=source_vecs,  # values
        source_padding=source_paddings,
        source_segment_id=source_segment_id)

    if query_segment_id is not None:
      query_segment_id = tf.reshape(query_segment_id, [-1])

    ctx_vec, atten_prob, _ = self.atten.ComputeContextVectorWithSource(
        theta=theta.atten,
        packed_src=packed_src,
        query_vec=tf.reshape(query_vec, [-1, query_dim]),
        per_step_source_padding=causal_padding,
        query_segment_id=query_segment_id,
        **kwargs)

    # Gating operations
    ctx_vec = query_p_c * tf.reshape(
        self.ctx_layer_norm.FProp(theta.ctx_layer_norm, ctx_vec),
        tf.shape(query_vec))

    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    h = input_to_add + ctx_vec
    atten_prob = tf.reshape(atten_prob, [
        tf.shape(query_vec)[0],
        tf.shape(query_vec)[1],
        tf.shape(source_vecs)[0]
    ])
    return h, atten_prob, query_p_c, source_p_c

  def _FinishExtendStep(self,
                        theta,
                        query_vec,
                        unnormalized_query_vec,
                        extended_packed_src,
                        t=None):
    """Finish extending prefix by one more time step.

    Isolating this function from ExtendStep allows generalizing self-attention
    to causal attention on other inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [target_batch, dim]
      unnormalized_query_vec: [target_batch, dim]
      extended_packed_src: A `.NestedMap` object containing source_vecs,
        source_contexts, source_paddings, and source_segment_ids
      t: a scalar, the current time step, 0-based.

    Returns:
      A triplet (cur_output, atten_prob, new_state) where cur_output is a tensor
      representing the output from the current state, and new_state is the new
      state `.NestedMap`.
    """
    p = self.params
    # Gating operations
    query_p_c = self.query_gating.FProp(theta.query_gating, query_vec)

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

    # Gating operations
    ctx_vec = self.ctx_layer_norm.FProp(theta.ctx_layer_norm, ctx_vec)
    ctx_vec = query_p_c * tf.reshape(ctx_vec, tf.shape(query_vec))
    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    h = input_to_add + ctx_vec

    new_states = py_utils.NestedMap(
        key=extended_packed_src.source_vecs,
        value=extended_packed_src.source_contexts)
    return h, atten_prob, new_states

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

    # Gating operations
    unnormalized_query_vec = query_vec
    query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)
    source_p_c = self.kv_gating.FProp(theta.kv_gating, query_vec)
    source_vec = source_p_c * query_vec

    cached_packed_src = py_utils.NestedMap(
        source_vecs=prefix_state.key,
        source_contexts=prefix_state.value,
        source_padding=None,
        source_segment_id=None)
    extended_packed_src = self.atten.ExtendSourcePacked(theta.atten, source_vec,
                                                        source_vec, None, None,
                                                        cached_packed_src, t)
    return self._FinishExtendStep(theta, query_vec, unnormalized_query_vec,
                                  extended_packed_src, t)


class CCTFeedForwardLayer(base_layer.BaseLayer):
  """Transformer FF layer with CCT gating.

  https://arxiv.org/abs/2002.07106

  Differences from standard Transformer FF layer:
  1. Each feedforward layer is divided into num_blocks smaller layers (divided
  along the hidden dimension).
  2. Each block has its separate input layer norm.
  3. Each block has its separate output layer norm.
  4. Outputs from each block are gated with CCTGatingNetwork output - which is
  between 0 and 1 for training and either 0 or 1 during inference.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    # Transformer Feedforward params.
    p.Define('input_dim', 0, 'Dimension of the layer input.')
    p.Define('output_dim', 0, 'Dimension of the layer output.')  # Deprecated.
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

    # Expert params.
    p.Define('num_blocks', 1, 'Number of separately gated ff blocks.')
    p.Define('gating_tpl', layers.CCTGatingNetwork.Params(), 'gating template.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim
    assert p.hidden_dim
    assert not p.output_dim, 'output_dim should not be set.'

    # Initialize feed-forward layer
    params = p.fflayer_tpl.Copy()
    params.name = 'fflayer'
    params.input_dim = p.input_dim
    params.activation = [p.activation, 'NONE']
    if p.output_dim == 0:
      params.hidden_layer_dims = [p.hidden_dim, p.input_dim]
    else:
      params.hidden_layer_dims = [p.hidden_dim, p.output_dim]

    params.dropout = [
        params.dropout.cls.Params().Set(keep_prob=1.0 - p.relu_dropout_prob),
        params.dropout.cls.Params().Set(keep_prob=1.0)
    ]

    ffs = []
    ln_params = []
    out_layer_norm = []  # Required for stabilizing CCT.
    for i in range(p.num_blocks):
      ff_p = params.Copy()
      ff_p.name += '_%d' % i
      ffs.append(ff_p)

      ln_p = p.ln_tpl.Copy()
      ln_p.name = 'fflayer_ln_%d' % i
      ln_p.input_dim = p.input_dim
      ln_params.append(ln_p)

      ln_p = p.ln_tpl.Copy()
      ln_p.name = 'fflayer_ln_out_%d' % i
      ln_p.input_dim = p.input_dim
      out_layer_norm.append(ln_p)
    self.CreateChildren('fflayers', ffs)
    self.CreateChildren('layer_norm', ln_params)
    self.CreateChildren('out_layer_norm', out_layer_norm)

    # Note: Set gating noise and warmup in parent layer.
    ff_gating = p.gating_tpl.Copy()
    ff_gating.input_dim = p.input_dim
    ff_gating.num_outputs = p.num_blocks
    ff_gating.name = 'gating_net'
    self.CreateChild('ff_gating', ff_gating)

    dropout_tpl = p.residual_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.CreateChild('residual_dropout', dropout_tpl)

  def FProp(self, theta, inputs, paddings):
    """Feed-forward, layer-norm, residual, gating and layer-norm.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: [time, batch, dim].
      paddings: [time, batch]

    Returns:
      tensor of the same shape with inputs
    """
    p = self.params
    ff_outputs = []
    for i in range(p.num_blocks):
      inputs_normalized = self.layer_norm[i].FProp(theta.layer_norm[i], inputs)
      ff_output = self.fflayers[i].FProp(
          theta.fflayers[i],
          inputs_normalized,
          paddings=tf.expand_dims(paddings, -1))
      ff_output = self.out_layer_norm[i].FProp(theta.out_layer_norm[i],
                                               ff_output)
      ff_outputs.append(ff_output)
    p_c = self.ff_gating.FProp(theta.ff_gating, inputs_normalized)
    out = inputs + self.residual_dropout.FProp(
        theta.residual_dropout,
        tf.reduce_sum(
            tf.expand_dims(p_c, -1) * tf.stack(ff_outputs, -2), axis=-2))
    return out, p_c


class TransformerWithContextLayer(base_layer.BaseLayer):
  """A transformer layer with 3 attention layers.

     The same as layers_with_attention.TransformerLayer, but with an
     additional attention layer to attend to a third transformer stack
     representing context.

     self-attention => context attention (newly added as tertiary_atten) =>
     encoder attention (named aux_atten in TransformerLayer).

     The weights are *not* shared between these three attention layers.

     See https://arxiv.org/pdf/1810.03581.pdf
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('source_dim', 0, 'Dimension of the transformer block input.')
    p.Define('output_dim', 0, 'Dimension of the transformer block output.')
    p.Define(
        'tr_atten_tpl',
        TransformerAttentionLayer.Params().Set(num_attention_heads=8),
        'Transformer Attention Layer params. The same template is applied '
        'to all three attention layers.')
    p.Define(
        'tr_tertiary_atten_tpl', None,
        'Transformer Attention Layer params for the tertiary attention. '
        'When None, copies tr_atten_tpl above.')
    p.Define('tr_fflayer_tpl',
             TransformerFeedForwardLayer.Params().Set(hidden_dim=2048),
             'Transformer Feed-Forward Layer params.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    # Required params used by the decoder.
    p.Define('has_aux_atten', True, 'Must be True.')
    p.Define('mask_self_atten', True, 'Must be True.')
    # removed: p.num_aux_atten_post_proj, p.tr_post_ln_tpl
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.has_aux_atten
    assert p.mask_self_atten
    if not p.source_dim:
      raise ValueError('p.source_dim not set')

    # Initialize multi-headed self-attention
    params = p.tr_atten_tpl.Copy()
    params.name = 'multihead_self_atten'
    params.source_dim = p.source_dim
    params.packed_input = p.packed_input
    params.is_masked = True
    self.CreateChild('self_atten', params)

    # Initialize tertiary attention.
    # If p.tr_tertiary_atten_tpl is None, we fall back to using
    # p.tr_tertiary_atten_tpl.
    params = p.tr_tertiary_atten_tpl or p.tr_atten_tpl.Copy()
    params.name = 'tertiary_multihead_atten'
    params.source_dim = p.source_dim
    params.packed_input = p.packed_input
    self.CreateChild('tertiary_atten', params)

    # Initialize multi-headed encoder attention
    params = p.tr_atten_tpl.Copy()
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
            aux_vecs,
            aux_paddings,
            tertiary_vecs,
            tertiary_paddings,
            source_segment_id=None,
            aux_segment_id=None,
            tertiary_segment_id=None,
            **kwargs):
    """Transformer Layer.

    Please see docstring of TransformerAttentionLayer.FProp.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_vecs: [source_time, source_batch, dim].
      source_paddings: [source_time, source_batch]
      aux_vecs: [aux_time, aux_batch, dim]
      aux_paddings: [aux_time, aux_batch]
      tertiary_vecs: [tertiary_time, tertiary_batch, dim]
      tertiary_paddings: [tertiary_time, tertiary_batch]
      source_segment_id: [source_time, source_batch]
      aux_segment_id: [aux_time, aux_batch]
      tertiary_segment_id: [tertiary_time, tertiary_batch]
      **kwargs: Can be optional params for the attention layer, eg. attention
        projection index tensor.

    Returns:
      The attention context vector, [source_time, source_batch, dim].

      The attention probability vector, [source_time, source_batch, aux_time].
    """
    p = self.params
    if p.packed_input:
      assert source_segment_id is not None, ('Need to specify segment id for '
                                             'packed input.')
      assert aux_segment_id is not None, ('Need to specify segment id for '
                                          'packed input.')
      assert tertiary_segment_id is not None, ('Need to specify segment id for '
                                               'packed input.')

    atten_vec, atten_prob = self.self_atten.FProp(
        theta.self_atten,
        source_vecs,
        source_paddings,
        query_segment_id=source_segment_id)
    atten_vec, atten_prob = self.tertiary_atten.FProp(
        theta.tertiary_atten, atten_vec, tertiary_paddings, tertiary_vecs,
        source_segment_id, tertiary_segment_id)
    atten_vec, atten_prob = self.atten.FProp(theta.atten, atten_vec,
                                             aux_paddings, aux_vecs,
                                             source_segment_id, aux_segment_id,
                                             **kwargs)

    h = self.fflayer.FProp(theta.fflayer, atten_vec, source_paddings)
    return h, atten_prob

  def ExtendStep(self,
                 theta,
                 source_vecs,
                 prefix_states,
                 aux_vecs,
                 aux_paddings,
                 tertiary_vecs,
                 tertiary_paddings,
                 t=None,
                 **kwargs):
    """Transformer Layer, extend one step in decoding.

    Please see docstring of TransformerAttentionLayer.ExtendStep.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_vecs: [source_batch, dim].
      prefix_states: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
      aux_vecs: [aux_time, aux_batch, dim]
      aux_paddings: [aux_time, aux_batch] tertiary_vecs=None,
        tertiary_paddings=None,
      tertiary_vecs: [tertiary_time, tertiary_batch, dim]
      tertiary_paddings: [tertiary_time, tertiary_batch]
      t: a scalar, the current time step, 0-based.
      **kwargs: Can be optional params for the attention layer, eg. attention
        projection index tensor.

    Returns:
      The attention context vector, [target_batch, source_dim]

      The attention probability vector from the encoder attention layer (the
      last attention layer) only, [source_time, target_batch].
      TODO(zhouwk): Return also the attention prob from the tertiary attention.

      Updated prefix states
    """
    p = self.params

    batch_size = py_utils.GetShape(source_vecs)[0]

    # First the self-attention layer.
    atten_vec, _, new_states = self.self_atten.ExtendStep(
        theta.self_atten, source_vecs, prefix_states, t)

    # Next the context attention (tertiary_atten) layer.
    atten_vec = tf.expand_dims(atten_vec, axis=0)
    atten_vec, _ = self.tertiary_atten.FProp(theta.tertiary_atten, atten_vec,
                                             tertiary_paddings, tertiary_vecs)

    # Next the source attention (aux_atten) layer.
    atten_vec, atten_prob = self.atten.FProp(theta.atten, atten_vec,
                                             aux_paddings, aux_vecs, **kwargs)

    # Finally, the feedforward layer.
    h = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([1, batch_size], dtype=py_utils.FPropDtype(p)))
    h = tf.squeeze(h, 0)
    return h, atten_prob, new_states


class SelfAttentiveLayer(base_layer.BaseLayer):
  """Self-attentive structure for temporal pooling.

  See https://arxiv.org/abs/1703.03130
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_heads', 5, 'Number of attention heads to use.')
    p.Define('input_dim', 128, 'Dimension of the vectors.')
    p.Define('hidden_dim', 128, 'Dimension of the hidden vectors.')
    p.Define('penalty_coef', 1.0, 'Penalisation coefficient.')
    p.Define(
        'penalty_terms', 1.0,
        'A (list of) float values to penalise the attention heads.'
        '1 and 1/num_heads encourage the heads to be spiky and smooth')
    p.Define(
        'input_data_format', 'BTC',
        'String(enum) specifying the output data format of the encoder. '
        'Also used for output converters.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if not p.name:
      raise ValueError('Layer must have a specified name.')
    assert p.num_heads > 0, 'Number of attention heads must be positive.'
    assert p.input_dim > 0, 'Input vector dimension must be positive.'
    assert p.hidden_dim > 0, 'Hidden vector dimension must be positive.'
    assert p.penalty_coef >= 0.0, 'Penalisation coefficient for attention.'
    assert p.input_data_format in {'BTC', 'TBC'}, 'Expect TBC or BTC inputs.'
    if isinstance(p.penalty_terms, float):
      p.penalty_terms = [p.penalty_terms] * p.num_heads
    assert isinstance(p.penalty_terms, list)
    if len(p.penalty_terms) < p.num_heads:
      term = p.penalty_terms[-1]
      p.penalty_terms += [term] * (p.num_heads - len(p.penalty_terms))
    for eachterm in p.penalty_terms:
      assert eachterm > 0, 'Penalty term {} is not positive'.format(eachterm)
    # the trainable model parameters
    w1 = layers.FCLayer.Params().Set(
        name='self_attentive_layer_w1',
        input_dim=p.input_dim,
        output_dim=p.hidden_dim,
        activation='TANH',
        params_init=py_utils.WeightInit.Xavier(0.1))
    self.CreateChild('att_w1', w1)
    w2 = layers.FCLayer.Params().Set(
        name='self_attentive_layer_w2',
        input_dim=p.hidden_dim,
        output_dim=p.num_heads,
        activation='NONE',
        params_init=py_utils.WeightInit.Xavier(0.1))
    self.CreateChild('att_w2', w2)

  def FProp(self, theta, inputs, paddings=None):
    p = self.params
    if p.input_data_format == 'TBC':
      inputs = tf.transpose(inputs, [1, 0, 2])
      if paddings is not None:
        paddings = tf.transpose(paddings, [1, 0])
    assert inputs.shape[2] == p.input_dim, 'Input vector sizes do not match.'

    paddings = tf.expand_dims(paddings, axis=-1)
    hiddens = self.att_w1.FProp(theta.att_w1, inputs, paddings)
    logits = self.att_w2.FProp(theta.att_w2, hiddens, paddings)
    values_a = tf.nn.softmax(logits)
    values_at = tf.transpose(values_a, [0, 2, 1])
    outputs = tf.matmul(values_at, inputs)

    aux_loss_ctx = py_utils.AuxLossContext.Current()
    if aux_loss_ctx is not None:
      penmat = tf.linalg.diag(p.penalty_terms)
      terms = tf.tile(penmat, [inputs.shape[0], 1])
      terms = tf.reshape(terms, [inputs.shape[0], p.num_heads, p.num_heads])
      values = tf.matmul(values_at, values_a)
      penalty = tf.reduce_sum(tf.square(values - terms), axis=[-2, -1])
      aux_loss_ctx.AddLoss(p.penalty_coef * penalty)

    return outputs
