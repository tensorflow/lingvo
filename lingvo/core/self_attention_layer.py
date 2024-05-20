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
"""Self-attention layers.

[1] Attention is all you need.
    https://arxiv.org/pdf/1706.03762.pdf Section 3.
[2] Pay Less Attention with Lightweight and Dynamic Convolutions.
    https://arxiv.org/abs/1901.10430
"""
from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import batch_major_attention
from lingvo.core import py_utils

MultiHeadedSelfAttention = batch_major_attention.MultiHeadedAttention


class BlockSparseAttention(MultiHeadedSelfAttention):
  """Block sparse attention based on https://arxiv.org/pdf/2007.14062.pdf.

  This implmentation only has diagonal band mask and does not support
  global and random attention.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'src_block_size', None, 'Query Block size for block sparse attention.'
    )
    p.Define(
        'tgt_block_size',
        None,
        'Key/Value Block size for block sparse attention.',
    )
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.src_block_size is not None, 'src_block_size is not set'
    assert p.src_block_size > 0, 'src_block_size should be greater than 0'
    assert not p.packed_input, 'packed_input is not supported'
    assert (
        not p.use_scale_invariant_atten
    ), 'use_scale_invariant_atten is not supported.'
    assert (
        not p.enable_scaling_code_motion
    ), 'enable_scaling_code_motion is not supported.'

  def _DiagonalBandMaskFromInputs(self, src_blocked_mask, tgt_blocked_mask):
    """Constructs diagonal block mask to compute local attention.

    This mask is used to compute local attention for all the query blocks.

    Args:
      src_blocked_mask: 3D Tensor of shape [b, t//wt, wt].
      tgt_blocked_mask: 3D Tensor of shape [b, s//ws, ws].

    Returns:
      float Tensor of shape [b, 1, t//wt, ws, ws].
    """
    # Concatenating left rolled, fixed and right rolled copies of
    # tgt_blocked_mask to compute diagonal band mask.
    # [b, t//wt, wt] x [b, s//ws, ws] ==> [b, t//wt, wt, ws]
    # Assumption: t//wt = s//ws
    band_mask = self.QEinsum(
        'BLQ,BLK->BLQK', src_blocked_mask, tgt_blocked_mask
    )
    band_mask = tf.expand_dims(band_mask, 1)
    return band_mask

  def _ComputeLogits(self, theta, query, key, mask=None, eqn=None):
    del theta
    query, key = self.ToAqtActActInputs(query, key)
    if eqn is None:
      eqn = 'BNTH,BNSH->BNTS'
    logits = self.QEinsum(eqn, query, key)
    logits = self.FromAqtActActMatmul(logits)
    # Keep softmax computation in float32 otherwise the low precision can
    # can lead to worse quality.
    logits = tf.cast(logits, tf.float32)
    logits = self._CapLogits(logits)
    if mask is not None:
      logits += batch_major_attention.GetDtypeMin(logits.dtype) * (
          1.0 - tf.cast(mask, logits.dtype)
      )
    return logits

  def _ComputeAttnProbs(self, theta, logits):
    p = self.params
    dtype = p.fprop_dtype
    if dtype is None:
      dtype = p.dtype
    probs = tf.cast(
        py_utils.Softmax(logits, extra_logit=p.atten_extra_logit),
        dtype,
    )
    # Apply dropout to probs.
    probs = self.atten_dropout.FProp(theta.atten_dropout, probs)
    return probs

  def _ComputeContext(self, theta, probs, value, eqn=None):
    del theta
    probs, value = self.ToAqtActActInputs(
        probs,
        value,
        act_lhs_distribution='positive',
        act_rhs_distribution='symmetric',
    )
    if eqn is None:
      eqn = 'BNTS,BNSH->BNTH'
    encoded = self.QEinsum(eqn, probs, value)
    encoded = self.FromAqtActActMatmul(encoded)
    return encoded

  def ComputeLocalContext(self, theta, query, key, value, band_mask):
    """Computes diagonal context for query blocks.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query: [B, n, t//wt, wt, h]
      key: [B, n, s//ws, ws, h]
      value: [B, n, s//ws, ws h]
      band_mask: [B, 1, t//wt, wt, ws]

    Returns:
      [B, n, t//wt, wt, h]
    """
    # [b, n, t//wt, wt, h] x [b, n, t//wt, ws, h]
    #  ==> [b, n, t//wt, wt, ws]
    diagonal_logits = self._ComputeLogits(
        theta,
        query,  # [b, n, t//wt, wt, h]
        key,  # [b, n, t//wt, ws, h]
        mask=band_mask,  # [b, 1, t//wt, wt, ws]
        eqn='BNLTH,BNLSH->BNLTS',
    )
    band_probs = self._ComputeAttnProbs(theta, diagonal_logits)
    # [b, n, t//wt, wt, ws] x [b, n, s//ws, ws, h] ==>
    # [b, n, t//wt, wt, h]
    band_context = self._ComputeContext(
        theta,
        band_probs,  # [b, n, t//wt, wt, ws]
        value,  # [b, n, s//ws, ws, h]
        eqn='BNLTS,BNLSH->BNLTH',
    )
    return band_context, band_probs

  def BlockSparseAttention(
      self, theta, query, key, value, input_mask, band_mask
  ):
    del input_mask
    p = self.params
    # Scale the query projection.
    query = self._MaybeScaleQuery(theta, query)

    # Converting query to blocks.
    b, n, t, h = py_utils.GetShape(query, 4)
    wt = p.src_block_size
    src_num_blocks = t // wt
    blocked_query = tf.reshape(query, (b, n, src_num_blocks, wt, h))

    # Converting key/value to blocks.
    _, n, s, h = py_utils.GetShape(key, 5)
    ws = p.tgt_block_size or p.src_block_size
    tgt_num_blocks = s // ws
    assert tgt_num_blocks == src_num_blocks, 'tgt_num_blocks != src_num_blocks'
    blocked_key = tf.reshape(key, (b, n, tgt_num_blocks, ws, h))
    blocked_value = tf.reshape(value, (b, n, tgt_num_blocks, ws, h))

    with tf.name_scope('q_0_to_n_local_context'):
      ctx, _ = self.ComputeLocalContext(
          theta,
          blocked_query,
          blocked_key,
          blocked_value,
          band_mask,
      )
    encoded = tf.reshape(ctx, [b, n, t, h])
    return encoded, None

  def FProp(
      self,
      theta,
      query_vec,
      key_vec,
      value_vec,
      paddings,
      segment_mask=None,
      per_step_padding=None,
  ):
    """Computes the value vector given the current query output.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [B, T, D].
      key_vec:   [B, S, D].
      value_vec: [B, S, D].
      paddings:  [B, S].
      segment_mask: [B, 1, T, S]. A mask only applied if packed_input=encTrue.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, T, T] if
        not None.

    Returns:
      encoded: [B, T, D].
      atten_probs: [B, N, T, S].

    Raises:
      ValueError: If value projection is disabled.
    """
    p = self.params
    assert per_step_padding is None, 'per_step_padding is not supported.'
    assert segment_mask is None, 'segment_mask is not supported.'
    b, t, _ = py_utils.GetShape(query_vec, 3)
    _, s, _ = py_utils.GetShape(key_vec, 3)
    paddings = py_utils.HasShape(paddings, [b, s])
    assert t % p.src_block_size == 0, 'seq_length % src_block_size != 0'
    assert s % p.tgt_block_size == 0, 'seq_length % tgt_block_size != 0'
    input_mask = tf.cast(1.0 - tf.cast(paddings, tf.float32), paddings.dtype)
    # Computing band mask for local attention.
    block_mask = tf.reshape(
        input_mask, (-1, s // p.tgt_block_size, p.tgt_block_size)
    )
    band_mask = self._DiagonalBandMaskFromInputs(block_mask, block_mask)

    # Heads projection
    query_proj, key_proj, value_proj = self._HeadsProj(
        theta, query_vec, key_vec, value_vec, eqn='BTD,DNH->BNTH'
    )
    input_mask = tf.reshape(input_mask, [b, 1, 1, s])
    encoded, atten_probs = self.BlockSparseAttention(
        theta,
        query_proj,
        key_proj,
        value_proj,
        input_mask,
        band_mask,
    )
    # Post projection
    if p.enable_ctx_post_proj:
      encoded = self._PostProj(theta, encoded, eqn='BNTH,DNH->BTD')
    else:
      b, n, t, h = py_utils.GetShape(encoded, 4)
      encoded = tf.transpose(encoded, [0, 2, 1, 3])
      encoded = tf.reshape(encoded, [b, t, n * h])

    return encoded, atten_probs


class Builder(batch_major_attention.Builder):
  """Builder for self-attention layers."""

  def SelfAttention(self, name, layer_idx=None):
    p = self.params
    input_to_add = (
        'i.vec' if p.selfatten_add_unnormalized_input else 'after_ln')

    attention_inputs = 'after_ln,after_ln,after_ln,i.paddings'
    if p.packed_input:
      attention_inputs += ',i.segment_mask'
    if isinstance(p.atten_tpl, list):
      assert layer_idx is not None, 'layer_idx must be specified.'
      atten_tpl = p.atten_tpl[layer_idx]
    else:
      atten_tpl = p.atten_tpl

    sub_list = [
        ('i.vec->after_ln', self._DefaultLN('LN')),
        (
            '{}->after_att,unused_prob'.format(attention_inputs),
            self._MultiHeadedAtten(
                'atten',
                enable_qkv_proj_in_onestep=p.default_enable_qkv_proj_in_onestep,
                enable_qk_proj_in_onestep=atten_tpl.enable_qk_proj_in_onestep,
                atten_tpl=atten_tpl,
            ),
        ),
        (
            'after_att->after_dropout',
            self._Dropout('dropout', p.residual_dropout_prob),
        ),
        (
            '{},after_dropout->o.vec'.format(input_to_add),
            self._Add('add', apply_residual=p.atten_apply_residual),
        ),
        ('i.paddings->o.paddings', self._Id('id')),
    ]

    if p.packed_input:
      sub_list.append(('i.segment_mask->o.segment_mask', self._Id('mask')))

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list
    )

  def _TransformerLayerBlock(
      self, name, feed_forward_qdomain=None, layer_idx=None
  ):
    """(inputs, paddings) -> (encoded, paddings)."""
    return self._Seq(
        name,
        self.SelfAttention('self_atten', layer_idx=layer_idx),
        self.Feedforward('ff', qdomain=feed_forward_qdomain),
    )

  def TransformerStack(self, name, num_layers=1, feed_forward_qdomain=None):
    """Returns a stack of num_layers self-attention layers."""
    p = self.params
    if isinstance(p.atten_tpl, list):
      assert (
          len(p.atten_tpl) == num_layers
      ), 'atten_tpl list must have the same length as num_layers.'
    blocks = []
    for i in range(num_layers):
      blocks.append(
          self._Seq(
              'iter_%03d' % i,
              self._TransformerLayerBlock(
                  'block',
                  feed_forward_qdomain=feed_forward_qdomain,
                  layer_idx=i,
              ),
          )
      )
    return self._MaybeSplit(name, blocks) or self._Seq(name, *blocks)

  def _StridedTransformerLayerBlock(
      self,
      name,
      *,
      stride=1,
      first_n=None,
      feed_forward_qdomain=None,
      layer_idx=None
  ):
    """(inputs, paddings) -> (encoded, paddings)."""
    return self._Seq(
        name,
        self._StridedAttention(
            'self_atten', stride=stride, first_n=first_n, layer_idx=layer_idx
        ),
        self.Feedforward('ff', qdomain=feed_forward_qdomain),
    )

  def TransformerStackV2(
      self,
      name,
      num_layers=1,
      *,
      final_layer_first_n=None,
      final_layer_stride=1,
      feed_forward_qdomain=None
  ):
    """Returns a stack of num_layers self-attention layers."""
    p = self.params
    if isinstance(p.atten_tpl, list):
      assert (
          len(p.atten_tpl) == num_layers
      ), 'atten_tpl list must have the same length as num_layers.'
    blocks = []
    for i in range(num_layers):
      if i < num_layers - 1:
        stride, first_n = (1, None)
      else:
        stride, first_n = (final_layer_stride, final_layer_first_n)
      blocks.append(
          self._Seq(
              'iter_%03d' % i,
              self._StridedTransformerLayerBlock(
                  'block',
                  stride=stride,
                  first_n=first_n,
                  feed_forward_qdomain=feed_forward_qdomain,
                  layer_idx=i,
              ),
          )
      )
    return self._MaybeSplit(name, blocks) or self._Seq(name, *blocks)


class SimplifiedTransformerBuilder(Builder):
  """Builder for simplified transformer blocks based on https://arxiv.org/pdf/2311.01906.pdf."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'parallel_attention_mlp',
        False,
        'If Set, Attention and MLP are computed in parallel as in Fig. 10 of'
        'https://arxiv.org/pdf/2311.01906.pdf.',
    )
    return p

  def TransformerLayerBlock(
      self,
      name,
      stride=1,
      first_n=None,
      num_heads=None,
      feed_forward_qdomain=None,
      layer_idx=None,
  ):
    """(inputs, paddings) -> (encoded, paddings)."""
    p = self.params
    ff_hidden_dim = p.ff_hidden_dim
    if p.selfatten_add_unnormalized_input:
      tf.logging.warning(
          'This flag is a no-op since there is no residual add for self attn.'
      )

    attention_inputs = 'strided_query,after_ln,after_ln,i.paddings'
    sub_list = []
    if p.packed_input:
      if stride > 1:
        # TODO(huangyp): Make sure striding won't cross segment boundaries.
        tf.logging.warning(
            'Each segment in the packed input should has length '
            'divisible by stride.'
        )
      sub_list += [
          (
              'i.segment_mask->strided_segment_mask',
              self._Stride(
                  'segment_mask_query_stride', stride, first_n, axis=2
              ),
          ),
      ]
      attention_inputs += ',strided_segment_mask'

    if num_heads is None:
      num_heads = p.num_heads

    if isinstance(p.atten_tpl, list):
      assert layer_idx is not None, 'layer_idx must be specified.'
      atten_tpl = p.atten_tpl[layer_idx]
    else:
      atten_tpl = p.atten_tpl

    # compute qkv in one step only if default_enable_qkv_proj_in_onestep
    # and no striding (stride==1) and not first_n
    enable_qkv_proj_in_onestep = (
        p.default_enable_qkv_proj_in_onestep and stride == 1 and not first_n
    )
    # Overriding default param based on stride.
    enable_qk_proj_in_onestep = (
        atten_tpl.enable_qk_proj_in_onestep
        and stride == 1
        and not first_n
        and not atten_tpl.use_mqa
    )

    sub_list += [
        ('i.vec->strided_input', self._Stride('before_add', stride, first_n)),
        # TODO(b/323011070): Add rezero support.
        ('i.vec->after_ln', self._DefaultLN('LN')),
        (
            'after_ln->strided_query',
            self._Stride('query_after_stride', stride, first_n),
        ),
    ]

    sub_list += [
        (
            '{}->after_att,prob'.format(attention_inputs),
            self._MultiHeadedAtten(
                'atten',
                num_heads,
                enable_qkv_proj_in_onestep=enable_qkv_proj_in_onestep,
                enable_qk_proj_in_onestep=enable_qk_proj_in_onestep,
                query_stride=stride,
                query_first_n=first_n,
                atten_tpl=atten_tpl,
            ),
        ),
        (
            'after_att->attn_output.vec',
            self._Dropout('dropout', p.residual_dropout_prob),
        ),
        (
            'i.paddings->attn_output.paddings',
            self._Stride('padding_after_Stride', stride, first_n),
        ),
    ]
    if p.packed_input:
      sub_list += [
          (
              'strided_segment_mask->attn_output.segment_mask',
              self._Stride(
                  'segment_mask_context_stride', stride, first_n, axis=3
              ),
          ),
      ]

    if p.parallel_attention_mlp:
      if p.device_mesh is not None:
        assert p.device_mesh.ndim >= 2
        assert p.weight_split_dims_mapping.df is not None
        assert p.weight_split_dims_mapping.fd is not None
      bias_f_split = (
          [p.weight_split_dims_mapping.df[1]]
          if p.weight_split_dims_mapping.df is not None
          else None
      )
      bias_d_split = (
          [p.weight_split_dims_mapping.fd[1]]
          if p.weight_split_dims_mapping.fd is not None
          else None
      )
      sub_list += [
          (
              'strided_query->after_feedforward',
              self._Seq(
                  'feedforward',
                  self._Linear(
                      'linear01',
                      p.model_dim,
                      ff_hidden_dim,
                      device_mesh=p.device_mesh,
                      weight_split_dims_mapping=(
                          p.weight_split_dims_mapping.df
                      ),
                      qdomain=feed_forward_qdomain,
                  ),
                  self.MeshSplit(
                      'split01', p.activation_split_dims_mapping.blf
                  ),
                  self._Bias(
                      'bias01',
                      ff_hidden_dim,
                      device_mesh=p.device_mesh,
                      weight_split_dims_mapping=bias_f_split,
                  ),
                  self._Activation('act', p.ff_activation_fn),
                  self._Dropout('relu_dropout', p.relu_dropout_prob),
                  self._Linear(
                      'linear02',
                      ff_hidden_dim,
                      p.model_dim,
                      device_mesh=p.device_mesh,
                      weight_split_dims_mapping=(
                          p.weight_split_dims_mapping.fd
                      ),
                      qdomain=feed_forward_qdomain,
                  ),
                  self.MeshSplit(
                      'split02', p.activation_split_dims_mapping.bld
                  ),
                  self._Bias(
                      'bias02',
                      p.model_dim,
                      device_mesh=p.device_mesh,
                      weight_split_dims_mapping=bias_d_split,
                  ),
                  self._Dropout('dropout', p.residual_dropout_prob),
              ),
          ),
      ]
      sub_list += [
          (
              'attn_output.vec,after_feedforward->added_attn_ff',
              self._Add('attn_add_ff', p.ff_residual_weight),
          ),
          ('added_attn_ff,attn_output.paddings->o.vec', self._Pad('pad')),
          ('attn_output.paddings->o.paddings', self._Id('paddings_output')),
      ]
      if p.packed_input:
        sub_list += [
            (
                'attn_output.segment_mask->o.segment_mask',
                self._Id('segment_mask_output'),
            ),
        ]
    else:
      sub_list += [(
          'attn_output->o',
          self.Feedforward(
              'ff',
              ff_hidden_dim=ff_hidden_dim,
              qdomain=feed_forward_qdomain,
          ),
      )]

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list
    )

  def TransformerStack(self, name, num_layers=1, feed_forward_qdomain=None):
    """Returns a stack of num_layers self-attention layers."""
    p = self.params
    if isinstance(p.atten_tpl, list):
      assert (
          len(p.atten_tpl) == num_layers
      ), 'atten_tpl list must have the same length as num_layers.'
    blocks = []
    for i in range(num_layers):
      blocks.append(
          self._Seq(
              'iter_%03d' % i,
              self.TransformerLayerBlock(
                  'block',
                  feed_forward_qdomain=feed_forward_qdomain,
                  layer_idx=i,
              ),
          )
      )
    return self._MaybeSplit(name, blocks) or self._Seq(name, *blocks)

  def TransformerStackV2(
      self,
      name,
      num_layers=1,
      *,
      final_layer_first_n=None,
      final_layer_stride=1,
      feed_forward_qdomain=None
  ):
    """Returns a stack of num_layers self-attention layers."""
    p = self.params
    if isinstance(p.atten_tpl, list):
      assert (
          len(p.atten_tpl) == num_layers
      ), 'atten_tpl list must have the same length as num_layers.'
    blocks = []
    for i in range(num_layers):
      if i < num_layers - 1:
        stride, first_n = (1, None)
      else:
        stride, first_n = (final_layer_stride, final_layer_first_n)
      blocks.append(
          self._Seq(
              'iter_%03d' % i,
              self.TransformerLayerBlock(
                  'block',
                  stride=stride,
                  first_n=first_n,
                  feed_forward_qdomain=feed_forward_qdomain,
                  layer_idx=i,
              ),
          )
      )
    return self._MaybeSplit(name, blocks) or self._Seq(name, *blocks)


# TODO(huangyp): remove this layer after transition to nested maps is complete.
class StackedTransformerEncoderLayers(base_layer.BaseLayer):
  """Wrapper class for layers returned by Builder.TransformerStack."""

  @classmethod
  def Cast(cls, params):
    # Cast params returned from the builder to params in this class.
    params.Define('base_cls', params.cls, 'Store the base class in params.')
    params.cls = cls
    return params

  def __init__(self, params):
    # Make this class a sub-class of params.base_cls
    self.__class__ = type(self.__class__.__name__, (params.base_cls, object),
                          dict(self.__class__.__dict__))
    super(self.__class__, self).__init__(params)  # pylint: disable=bad-super-call

  def FProp(self, theta, vec, paddings, segment_mask=None):
    outputs = super(self.__class__, self).FProp(  # pylint: disable=bad-super-call
        theta,
        py_utils.NestedMap(
            vec=vec, paddings=paddings, segment_mask=segment_mask))
    return outputs.vec, outputs.paddings
