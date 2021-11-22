# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Attention layers."""
import functools
import string
from typing import Optional, Tuple, Union

import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import embedding_softmax
from lingvo.jax.layers import stochastics
import numpy as np

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
weight_params = py_utils.weight_params
InstantiableParams = py_utils.InstantiableParams
AuxLossContext = py_utils.AuxLossContext
JTensor = pytypes.JTensor


def _get_large_negative_number(dtype):
  # -0.7 is a float64 in Jax. Explicit cast output to target dtype.
  if jnp.issubdtype(dtype, jnp.inexact):
    dtype_max = jnp.finfo(dtype).max
  elif jnp.issubdtype(dtype, jnp.integer):
    dtype_max = jnp.iinfo(dtype).max
  else:
    raise ValueError('Unsupported dtype for inputs.')
  return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def causal_mask(input_t: JTensor) -> JTensor:
  """Computes and returns causal mask.

  Args:
    input_t: A JTensor of shape [B, T, D].

  Returns:
    An attention_mask JTensor of shape [1, 1, T, T]. Attention mask has
    already been converted large negative values.
  """
  assert (input_t.dtype == jnp.float32 or
          input_t.dtype == jnp.bfloat16), input_t.dtype
  large_negative_number = _get_large_negative_number(input_t.dtype)
  t = input_t.shape[1]
  col_idx = jnp.tile(jnp.arange(t)[jnp.newaxis, :], [t, 1])
  row_idx = jnp.tile(jnp.arange(t)[:, jnp.newaxis], [1, t])
  mask = (row_idx < col_idx).astype(input_t.dtype) * large_negative_number
  return mask[jnp.newaxis, jnp.newaxis, :, :]


def segment_mask(segment_ids: JTensor,
                 source_segment_ids: Optional[JTensor] = None,
                 dtype: jnp.dtype = jnp.float32) -> JTensor:
  """Computes (non-causal) segment mask.

  Args:
    segment_ids: a JTensor of shape [B, T], the segment that each token belongs
      to.
    source_segment_ids: a JTensor of shape [B, S], the segment that each source
      token belongs to (optional).
    dtype: data type of the input.

  Returns:
    A JTensor of shape [B, 1, T, S].
  """
  # [B, T, 1]
  segment_ids_1 = jnp.expand_dims(segment_ids, axis=-1)
  # [B, 1, S]
  if source_segment_ids is not None:
    segment_ids_2 = jnp.expand_dims(source_segment_ids, axis=1)
  else:
    segment_ids_2 = jnp.expand_dims(segment_ids, axis=1)
  # [B, T, S].
  mask = jnp.not_equal(segment_ids_1, segment_ids_2).astype(dtype)
  mask = jnp.expand_dims(mask, 1)
  mask *= _get_large_negative_number(dtype)
  return mask


def causal_segment_mask(segment_ids: JTensor,
                        dtype: jnp.dtype = jnp.float32) -> JTensor:
  """Computes the masks which combines causal masking and segment masks.

  Args:
    segment_ids: a JTensor of shape [B, T], the segment that each token belongs
      to.
    dtype: data type of the input.

  Returns:
    A JTensor of shape [B, 1, T, T].
  """
  # [B, 1, T, T]
  segment_mask_t = segment_mask(segment_ids, dtype=dtype)
  # [1, 1, T, T]
  b, t = segment_ids.shape
  causal_mask_t = causal_mask(jnp.zeros([b, t, 1], dtype=dtype))
  return jnp.minimum(segment_mask_t, causal_mask_t)


def convert_paddings_to_mask(paddings: JTensor,
                             dtype: jnp.dtype = jnp.float32) -> JTensor:
  """Converts binary paddings to a logit mask ready to add to attention matrix.

  Args:
    paddings: binary JTensor of shape [B, T], with 1 denoting padding token.
    dtype: data type of the input.

  Returns:
    A JTensor of shape [B, 1, 1, T] ready to add to attention logits.
  """
  attention_mask = paddings[:, jnp.newaxis, jnp.newaxis, :]
  attention_mask *= _get_large_negative_number(dtype)
  return attention_mask


def shift_1d(inputs: JTensor, offset: int, axis: int):
  """Shift the input tensor by offset in the dimension axis.

    To shift right the offset is positive and the input is padded at the
    beginning, while to shift left the offset is negative and the input is
    padded at the end.

  Args:
    inputs: The input tensor to shift.
    offset: The number of positions to shift. If the offset is positive, pad at
      the beginning of the sequence, if the offset is negative, then pad at the
      end of the sequence.
    axis: The dimension in which to shift the input.

  Returns:
    The shifted input.
  """
  paddings = [((max(offset, 0), -min(offset, 0)) if i == axis else (0, 0))
              for i in range(len(inputs.shape))]
  input_length = jnp.shape(inputs)[axis]
  padded_inputs = jnp.pad(inputs, paddings)
  if offset > 0:
    output = jax.lax.dynamic_slice_in_dim(
        padded_inputs, start_index=0, slice_size=input_length, axis=axis)
  else:
    output = jax.lax.dynamic_slice_in_dim(
        padded_inputs, start_index=offset, slice_size=input_length, axis=axis)
  return output


class PerDimScale(base_layer.BaseLayer):
  """A layer to scale individual dims of the input."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for `PerDimScale`."""
    p = super().Params()
    p.Define('dim', 0, 'Number of individual dims .')
    return p

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    pc = weight_params(
        shape=[p.dim], init=WeightInit.Constant(0.0), dtype=p.dtype)
    self.create_variable('per_dim_scale', pc)

  def fprop(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Return theta.scale * inputs / jnp.sqrt(dim)).

    Args:
      theta: A `.NestedMap` object containing weights defined in this layer.
      inputs: A JTensor with shape [..., p.dim].

    Returns:
      outpus: A JTensor with shape [..., p.dim].
    """
    p = self.params
    inputs_shape = inputs.shape
    assert inputs_shape[-1] == p.dim

    # 1.0/jax.nn.softplus(0.0) = 1.442695041. Hard code this number so that we
    # can avoid unnecessary XLA op fusion mess on TPU.
    r_softplus_0 = 1.442695041
    scale = jnp.array(r_softplus_0 / np.sqrt(p.dim), dtype=inputs.dtype)
    scale *= jax.nn.softplus(theta.per_dim_scale)
    return inputs * scale


class AttentionProjection(base_layer.BaseLayer):
  """Layer that computes multi heads projection.

    This layer is expected to be used within DotProductAttention below.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for AttentionProjection."""
    p = super().Params()
    p.Define('input_dim', 0, 'Input dimension.')
    p.Define('num_heads', 0, 'Number of heads.')
    p.Define('dim_per_head', 0, 'Size of each head.')
    p.Define(
        'is_output_projection', False,
        'Whether it is out projection or not. If False, we use '
        '"...D,DNH->...NH" for query,key,value projection. Otherwise we use '
        '"...NH,DNH->...D" for output projection.')
    p.Define('use_bias', True, 'If to add bias in projection.')
    return p

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    wp = p.weight_split_dims_mapping
    if p.device_mesh is not None:
      assert wp.wt is not None, self.path
    pc = weight_params(
        shape=[p.input_dim, p.num_heads, p.dim_per_head],
        init=p.params_init,
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.wt)
    self.create_variable('w', pc)
    if p.use_bias:
      if p.is_output_projection:
        if p.device_mesh is not None:
          bias_split_dims_mapping = [wp.wt[0]]
        else:
          bias_split_dims_mapping = None
        pc_bias = weight_params(
            shape=[p.input_dim],
            init=WeightInit.Constant(0.0),
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=bias_split_dims_mapping)
      else:
        if p.device_mesh is not None:
          bias_split_dims_mapping = [wp.wt[1], wp.wt[2]]
        else:
          bias_split_dims_mapping = None
        pc_bias = weight_params(
            shape=[p.num_heads, p.dim_per_head],
            init=WeightInit.Constant(0.0),
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=bias_split_dims_mapping)
      self.create_variable('b', pc_bias)

  def fprop(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Computes the multi headed projection for inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A JTensor of shape [..., num_heads, dim_per_head] if
        p.is_output_projection is True or [..., p.input_dim] otherwise..

    Returns:
      The projected JTensor with shape [..., p.input_dim] if
      p.is_output_projection is True or [..., num_heads, dim_per_head]
      otherwise.
    """
    p = self.params

    # Because tf.einsum is not fully optimized unless all the dimensions are
    # fully specified, we have to avoid using '...' for batch dimensions in the
    # equation in tf.einsum for optimized performance. This is only feasible
    # when the rank of the tensor is known.
    # Sort the available symbols to avoid nondeterminism.
    eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('DHN')))
    shape = inputs.shape
    rank = len(shape)

    inputs = self._cast_to_fprop_dtype(inputs)

    if p.is_output_projection:
      assert shape[-2:] == (p.num_heads, p.dim_per_head)
      batch_eqn = eqn_sym[:(rank - 2)]
      eqn = f'{batch_eqn}NH,DNH->{batch_eqn}D'
    else:
      assert shape[-1] == p.input_dim, (
          f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {p.input_dim}')
      batch_eqn = eqn_sym[:(rank - 1)] if rank else '...'
      eqn = f'{batch_eqn}D,DNH->{batch_eqn}NH'
    ret = jnp.einsum(eqn, inputs, theta.w)
    if p.use_bias:
      ret += theta.b
    return ret


class CombinedQKVProjectionLayer(base_layer.BaseLayer):
  """Layer that computes QKV projection with a combined weight.

  It may lead to faster collectives and step-time on TPU.

  This layer is expected to be used within DotProductAttention below.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for CombinedQkvProjectionLayer."""
    p = super().Params()
    p.Define('input_dim', 0, 'Input dimension.')
    p.Define('num_heads', 0, 'Number of heads.')
    p.Define('dim_per_head', 0, 'Size of each head.')
    p.Define('use_bias', True, 'If to add bias in projection.')
    return p

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    wp = p.weight_split_dims_mapping
    if p.device_mesh is not None:
      assert wp.wt is not None, self.path
      # Replicate the concat axis.
      assert len(wp.wt) == 3, ('wp.wt only specifies the sharding for '
                               'the last three dims of the weight tensor.')
      weight_split_dims_mapping = [None] + list(wp.wt)
      bias_split_dims_mapping = [None, wp.wt[1], wp.wt[2]]
    else:
      weight_split_dims_mapping = None
      bias_split_dims_mapping = None
    # Combined weight for q, k, v projections.
    pc = weight_params(
        shape=[3, p.input_dim, p.num_heads, p.dim_per_head],
        init=p.params_init,
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=weight_split_dims_mapping)
    self.create_variable('w', pc)
    if p.use_bias:
      # Combined bias weight for q, k, v projections.
      pc_bias = weight_params(
          shape=[3, p.num_heads, p.dim_per_head],
          init=WeightInit.Constant(0.0),
          dtype=p.dtype,
          device_mesh=p.device_mesh,
          tensor_split_dims_mapping=bias_split_dims_mapping)
      self.create_variable('b', pc_bias)

  # TODO(zhangqiaorjc): Take query, key, value as inputs to support all
  # attentions.
  def fprop(self, theta: NestedMap,
            inputs: JTensor) -> Tuple[JTensor, JTensor, JTensor]:
    """Computes the QKV projection for inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A JTensor of shape [..., p.input_dim].

    Returns:
      The three projected JTensor with shape [..., num_heads, dim_per_head]
      in q_proj, k_proj and v_proj order.
    """
    p = self.params

    # Because tf.einsum is not fully optimized unless all the dimensions are
    # fully specified, we have to avoid using '...' for batch dimensions in the
    # equation in tf.einsum for optimized performance. This is only feasible
    # when the rank of the tensor is known.
    # Sort the available symbols to avoid nondeterminism.
    eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('KDHN')))
    shape = inputs.shape
    rank = len(shape)
    assert rank > 0

    assert shape[-1] == p.input_dim
    batch_dims_rank = rank - 1
    batch_eqn = eqn_sym[:batch_dims_rank] if rank else '...'
    # K indexes qkv.
    eqn = f'{batch_eqn}D,KDNH->K{batch_eqn}NH'
    ret = jnp.einsum(eqn, inputs, theta.w)
    ret = checkpoint_name(ret, 'combined_qkv_proj')
    if p.use_bias:
      # Add newaxis to bias weight for each batch dim since ret is K...NH
      # and theta.b is KNH. Need to reshape theta.b to K...NH
      ret += jnp.expand_dims(theta.b, list(range(1, batch_dims_rank + 1)))
    # Split into three projections.
    query_proj, key_proj, value_proj = ret
    query_proj = checkpoint_name(query_proj, 'query_proj')
    key_proj = checkpoint_name(key_proj, 'key_proj')
    value_proj = checkpoint_name(value_proj, 'value_proj')
    return query_proj, key_proj, value_proj


class DotProductAttention(base_layer.BaseLayer):
  """Dot-product attention with multiple attention heads.

  This implementation heavily uses einsum to be efficient on TPUs.  We use the
  following capital letters to denote certain JTensor parameters.

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    D = model dimension
    N = number of attention heads
    H = dimensions of each attention head.

  The algorithm is sketched as follows. Each intermediate JTensor or weight
  JTensor is annotated with its shape. E.g., Wq, the weight JTensor for query's
  projection, its shape is [D, N, H].

  Trainable weights:
    Wq, Wk, Wv: [D, N, H]
    Wout: [D, N, H]

  Note it also allows k, v and q to have different input dimension by setting
  input_dim as a dict: {'key': key_dim, 'value': value_dim, 'query': query_dim}.

  Input q:[B, T, D]; k:[B, S, D]; v:[B, S, D]
  q_proj:[B, T, N, H] = einsum('BTD,DNH->BTNH', x, Wq)
  k_proj:[B, S, N, H] = einsum('BSD,DNH->BSNH', x, Wk)
  v_proj:[B, S, N, H] = einsum('BSD,DNH->BSNH', x, Wv)
  logits:[B, N, T, S] = einsum('BTNH,BSNH->BNTS', q_proj, k_proj) / sqrt(H)
  probs:[B, N, T, S] = softmax(logits)
  context:[B, T, N, H] = einsum('BNTS,BSNH->BTNH', probs, v_proj)
  Output y:[B, T, D] = einsum('BTNH,DNH>BTD', context, Wout)
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for DotProductAttention."""
    p = super().Params()
    p.Define(
        'input_dim', 0,
        'An integer or a dict of integer values as number of input nodes. If '
        'input_dim is a dict, keys must be key, value and query.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    p.Define('num_heads', 1, 'Num of attention heads.')
    # dim_per_head == hidden_dim // num_heads
    p.Define('dropout_tpl', stochastics.Dropout.Params(),
             'Params for dropout layer.')
    p.Define('atten_dropout_prob', 0.0,
             'Probability at which we apply dropout to the attention weights.')
    p.Define('proj_tpl', AttentionProjection.Params(), 'Params for '
             'projection layer.')
    p.Define(
        'dconv_qkv', False, 'If True then apply a depth-wise convolution of '
        '`dconv_kernel_size`x1 after the key, query and value projection as in'
        'Primer https://arxiv.org/abs/2109.08668. Note that this is currently'
        'only supported for self-attention.')
    p.Define(
        'dconv_kernel_size', 3, 'Size of the kernel window over the sequence '
        'dimension in the depth-wise convolution.')
    # Note: supported for encoder only.
    p.Define(
        'combine_qkv', False,
        'Whether to combine qkv tensor for optimizing qkv input gradient '
        'computation with SPMD. Only supports self-attention.')
    p.Define('combined_qkv_proj_tpl', CombinedQKVProjectionLayer.Params(),
             'Params for combined qkv projection layer.')

    p.Define('use_bias', True, 'Whether to use bias for projection layers.')
    p.Define(
        'internal_enable_per_dim_scale', True,
        'Internal. Setting to False disables rescaling of attention logits '
        'with 1/sqrt(dim) factor. Some Transformer variants (GShard, T5) use '
        'internal_enable_per_dim_scale=False and adjust initialization of the '
        'linear transformations(einsums), in conjunction with Adafactor '
        'optimizer.')
    p.Define(
        'atten_logit_cap', 0.0, 'Cap the absolute values of logits by '
        'tanh. Enabled when a positive value is specified. May not be '
        'supported by a subclass.')
    p.Define(
        'use_rotary_position_emb', False, 'Whether to add rotary position'
        'embedding to the queries and keys before computing self attention'
        'scores. This was proposed in https://arxiv.org/abs/2104.09864.')
    # SPMD partition related params.
    #
    # d - model_dim
    # n - num_heads
    # h - attention_dim_per_heads
    # b - batch_size
    # l - seq_len
    wp = p.weight_split_dims_mapping
    wp.Define(
        'proj', None,
        'How the projection weights should be sharded. All projection'
        ' matrix share the same sharding.')
    wp.Define(
        'dconv', None,
        'How the dconv weights should be sharded. All dconv weights'
        ' share the same sharding.')
    ap = p.activation_split_dims_mapping
    ap.Define(
        'blnh', None,
        'Mesh split for query, key, value, and encoded tensors with the '
        'shape of [batch_size, seq_len, num_heads, dim_per_head].')
    ap.Define(
        'bld', None,
        'Mesh split for output after post projection with the shape of '
        '[batch_size, seq_len, model_dim].')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructs a DotProductAttention object."""
    super().__init__(params)
    p = self.params
    wp = p.weight_split_dims_mapping
    assert p.input_dim, 'input_dim is {}'.format(p.input_dim)
    assert p.hidden_dim, 'hidden_dim is {}'.format(p.hidden_dim)

    dim_per_head = p.hidden_dim // p.num_heads
    assert dim_per_head * p.num_heads == p.hidden_dim, (
        f'{dim_per_head} * {p.num_heads} != {p.hidden_dim}')

    if p.device_mesh is not None:
      assert p.weight_split_dims_mapping is not None
      assert p.activation_split_dims_mapping is not None

    def project_input(input_dim):
      proj_p = p.proj_tpl.Copy().Set(
          input_dim=input_dim,
          num_heads=p.num_heads,
          dim_per_head=dim_per_head,
          use_bias=p.use_bias)
      proj_p.weight_split_dims_mapping.wt = wp.proj
      return proj_p

    def combined_qkv_project_input(input_dim):
      proj_p = p.combined_qkv_proj_tpl.Copy().Set(
          input_dim=input_dim,
          num_heads=p.num_heads,
          dim_per_head=dim_per_head,
          use_bias=p.use_bias)
      proj_p.weight_split_dims_mapping.wt = wp.proj
      return proj_p

    if isinstance(p.input_dim, dict):
      key_input_dim = p.input_dim['key']
      value_input_dim = p.input_dim['value']
      query_input_dim = p.input_dim['query']
      assert key_input_dim, f'key_input_dim is {key_input_dim}'
      assert query_input_dim, f'query_input_dim is {query_input_dim}'
    else:
      key_input_dim = p.input_dim
      value_input_dim = p.input_dim
      query_input_dim = p.input_dim

    if p.combine_qkv:
      assert key_input_dim == value_input_dim
      assert key_input_dim == query_input_dim
      self.create_child('combined_qkv',
                        combined_qkv_project_input(query_input_dim))
    else:
      self.create_child('key', project_input(key_input_dim))
      self.create_child('query', project_input(query_input_dim))
      self.create_child('value', project_input(value_input_dim))

    if p.use_rotary_position_emb:
      pos_emb_p = embedding_softmax.RotaryPositionalEmbedding.Params()
      pos_emb_p.embedding_dims = dim_per_head
      self.create_child('rotary_position_emb', pos_emb_p)

    if p.dconv_qkv:
      causal_dconv_p = CausalDepthwiseConv1D.Params().Set(
          kernel_size=p.dconv_kernel_size,
          hidden_dims=[p.num_heads, dim_per_head],
      )
      causal_dconv_p.weight_split_dims_mapping.wt = wp.dconv
      self.create_child('dconv_q', causal_dconv_p)
      self.create_child('dconv_k', causal_dconv_p)
      self.create_child('dconv_v', causal_dconv_p)

    if p.internal_enable_per_dim_scale:
      self.create_child('per_dim_scale',
                        PerDimScale.Params().Set(dim=dim_per_head))
    self.create_child('atten_dropout',
                      p.dropout_tpl.Set(keep_prob=1.0 - p.atten_dropout_prob))
    # Setting is_output_projection=True to set the projection direction
    # from hidden dim to input dim. Output projection follows query_input_dim.
    post_proj_p = p.proj_tpl.Copy().Set(
        input_dim=query_input_dim,
        num_heads=p.num_heads,
        dim_per_head=dim_per_head,
        is_output_projection=True,
        use_bias=p.use_bias)
    post_proj_p.weight_split_dims_mapping.wt = wp.proj
    self.create_child('post', post_proj_p)

  def _shard_lbnh(self, x: JTensor) -> JTensor:
    """Shards tensors of shape [l, b, n, h].

    Transformer states cached during decoding are of shape [l, b, n, h]. Note
    "l" is used to annotate the sequence length dim. throughout this class,
    sometimes it is denoted as "l", sometimes "t", and sometime "s".

    Args:
      x: A tensor of shape [l, b, n, h]

    Returns:
      x with proper sharding annotations.
    """
    p = self.params
    ap = p.activation_split_dims_mapping
    if p.mesh_axis_names is None:
      return x
    if ap.blnh is None:
      return x
    assert len(ap.blnh) == 4
    lbnh = [ap.blnh[1], ap.blnh[0], ap.blnh[2], ap.blnh[3]]
    return base_layer.maybe_shard(x, lbnh, p.mesh_axis_names)

  def _shard_bnh(self, x: JTensor) -> JTensor:
    """Shards tensors of shape [b, n, h].

    Single step decoder output are of shape [b, n, h].

    Args:
      x: A tensor of shape [b, n, h]

    Returns:
      x with proper sharding annotations.
    """
    p = self.params
    ap = p.activation_split_dims_mapping
    if p.mesh_axis_names is None:
      return x
    if ap.blnh is None:
      return x
    assert len(ap.blnh) == 4
    bnh = [ap.blnh[0], ap.blnh[2], ap.blnh[3]]
    return base_layer.maybe_shard(x, bnh, p.mesh_axis_names)

  def _shard_blnh(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, n, h]."""
    p = self.params
    ap = p.activation_split_dims_mapping
    return base_layer.maybe_shard(x, ap.blnh, p.mesh_axis_names)

  def _shard_bld(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, l, d]."""
    p = self.params
    ap = p.activation_split_dims_mapping
    return base_layer.maybe_shard(x, ap.bld, p.mesh_axis_names)

  def _shard_bd(self, x: JTensor) -> JTensor:
    """Adds sharding annotations to tensors of shape [b, d]."""
    p = self.params
    ap = p.activation_split_dims_mapping
    if p.mesh_axis_names is None:
      return x
    if ap.bld is None:
      return x
    assert len(ap.bld) == 3
    bd = [ap.bld[0], ap.bld[2]]
    return base_layer.maybe_shard(x, bd, p.mesh_axis_names)

  def _cap_logits(self, logits: JTensor) -> JTensor:
    """When enabled, caps the logits by p.atten_logit_cap with tanh."""
    p = self.params
    if not p.atten_logit_cap or p.atten_logit_cap <= 0.:
      return logits
    cap = jnp.array(p.atten_logit_cap, dtype=logits.dtype)
    # Note that since this caps the negative side as well, caller
    # must defer the pad-with-very-negative-logits logic to after
    # this function returns.
    logits = cap * jnp.tanh(logits / cap)
    return logits

  def _dot_atten(self, theta: NestedMap, query: JTensor, key: JTensor,
                 value: JTensor,
                 atten_mask: JTensor) -> Tuple[JTensor, JTensor]:
    """Main attention function.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query: JTensor of shape [B, T, N, H].
      key: JTensor of shape [B, S, N, H].
      value: JTensor of shape [B, S, N, H].
      atten_mask: JTensor of shape [1/B, 1, 1/T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.

    Returns:
      encoded: JTensor of shape [B, T, N, H].
      atten_probs: JTensor of shape [B, N, T, S].
    """
    # Add key sharding annotations.
    p = self.params
    query = self._shard_blnh(query)
    key = self._shard_blnh(key)
    value = self._shard_blnh(value)

    b, s, n, h = key.shape
    base_layer.assert_has_shape(value, [b, s, n, h])
    base_layer.assert_has_shape(query, [b, -1, n, h])
    t = query.shape[1]
    # If only padding bias is supplied, then atten_mask can be [B, 1, 1, S]
    # since each target token is prohibited from attending to the same set of
    # source tokens. In this case tiling is inefficient and unnecessary.
    # If there is no padding mask, and only causal mask then the shape can be
    # [1, 1, T, S]
    base_layer.assert_has_shape(atten_mask, [-1, 1, -1, s])
    assert atten_mask.shape[2] in [1, t]
    assert atten_mask.shape[0] in [1, b]
    if p.internal_enable_per_dim_scale:
      query = self.per_dim_scale.fprop(theta.per_dim_scale, query)
    logits = jnp.einsum('BTNH,BSNH->BNTS', query, key)
    logits = checkpoint_name(logits, 'logits')
    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking
    padded_logits = logits + atten_mask.astype(jnp.float32)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    # Apply attention dropout.
    probs = self.atten_dropout.fprop(theta.atten_dropout, probs)
    # Compute the attention context.
    encoded = jnp.einsum('BNTS,BSNH->BTNH', probs, value)
    encoded = checkpoint_name(encoded, 'context')
    encoded = self._shard_blnh(encoded)
    return encoded, probs

  def _dot_atten_one_step(self, theta: NestedMap, query: JTensor, key: JTensor,
                          value: JTensor, atten_mask: JTensor) -> JTensor:
    """Dot attention function for queries with 1 time step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query: JTensor of shape [B, N, H].
      key: JTensor of shape [S, B, N, H].
      value: JTensor of shape [S, B, N, H].
      atten_mask: JTensor of shape [1/B, 1, S] which is a mask that is applied
        to prevent attention between unwanted pairs. This has already been
        converted into large negative logits. The first dimension is allowed to
        be of size 1, if the mask is shared by all items in the batch (e.g.,
        only a causal mask).

    Returns:
      encoded: JTensor of shape [B, N, H].
    """

    # TODO(yonghui): switch the cached states to batch major.
    p = self.params
    key = self._shard_lbnh(key)
    value = self._shard_lbnh(value)
    # query is 3d.
    query = self._shard_bnh(query)

    s, b, n, h = key.shape
    base_layer.assert_has_shape(value, [s, b, n, h])
    base_layer.assert_has_shape(query, [b, n, h])
    base_layer.assert_has_shape(atten_mask, [-1, 1, s])
    assert atten_mask.shape[0] in [1, b]
    if p.internal_enable_per_dim_scale:
      query = self.per_dim_scale.fprop(theta.per_dim_scale, query)
    logits = jnp.einsum('BNH,SBNH->BNS', query, key)
    logits = self._cap_logits(logits)
    # Attention softmax is always carried out in fp32.
    logits = logits.astype(jnp.float32)
    # Apply attention masking
    padded_logits = logits + atten_mask.astype(jnp.float32)
    # Of shape [b, n, s]
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)
    # Compute the attention context.
    encoded = jnp.einsum('BNS,SBNH->BNH', probs, value)
    encoded = self._shard_bnh(encoded)
    return encoded, probs

  def fprop(self, theta: NestedMap, query_vec: JTensor, key_vec: JTensor,
            value_vec: JTensor, atten_mask: JTensor) -> Tuple[JTensor, JTensor]:
    """Computes the value vector given the current query output.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: JTensor of shape [B, T, D].
      key_vec: JTensor of shape [B, S, D].
      value_vec: JTensor of shape [B, S, D].
      atten_mask: JTensor of shape [1/B, 1, 1/T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.

    Returns:
      encoded: JTensor of shape [B, T, D].
      atten_probs: JTensor of shape [B, N, T, S].
    """
    p = self.params
    if p.combine_qkv:
      # Only supports self attention.
      assert query_vec is key_vec
      assert query_vec is value_vec
      # Project inputs to key, value and query using a combined weight for
      # faster performance on TPU.
      query_proj, key_proj, value_proj = self.combined_qkv.fprop(
          theta.combined_qkv, query_vec)
    else:
      # Project inputs to key, value and query, respectively has shape
      # [B, S, N, H], [B, S, N, H], and [B, T, N, H].
      query_proj = self.query.fprop(theta.query, query_vec)
      key_proj = self.key.fprop(theta.key, key_vec)
      value_proj = self.value.fprop(theta.value, value_vec)

    # Apply depth-wise convolution as in Primer.
    # Paper: https://arxiv.org/abs/2109.08668.
    if p.dconv_qkv:
      query_proj = self.dconv_q.fprop(theta.dconv_q, query_proj, axis=1)
      key_proj = self.dconv_k.fprop(theta.dconv_k, key_proj, axis=1)
      value_proj = self.dconv_v.fprop(theta.dconv_v, value_proj, axis=1)

    # Apply rotary position embeddings.
    # Paper: https://arxiv.org/abs/2104.09864.
    if p.use_rotary_position_emb:
      query_proj = self.rotary_position_emb.fprop(theta.rotary_position_emb,
                                                  query_proj)
      key_proj = self.rotary_position_emb.fprop(theta.rotary_position_emb,
                                                key_proj)

    encoded, atten_probs = self._dot_atten(theta, query_proj, key_proj,
                                           value_proj, atten_mask)

    # Post projection
    encoded = self.post.fprop(theta.post, encoded)
    encoded = self._shard_bld(encoded)
    encoded = checkpoint_name(encoded, 'out_proj')

    return encoded, atten_probs

  def init_states(self, theta: NestedMap, target_batch_size: int,
                  target_max_length: int) -> NestedMap:
    """Initializes cache for autoregressive cached decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      target_batch_size: The batch size of the target to be decoded.
      target_max_length: The sequence length of the target to be decoded.

    Returns:
      The cache as a `.NestedMap` with key and value initialized. In
      addition if we are using depth-wise convolution as in Primer, then we also
      cache the query, as well as the queries, keys and values post application
      of depth-wise convolution.
    """
    p = self.params
    num_heads = p.num_heads
    atten_dim = p.hidden_dim
    dim_per_head = atten_dim // num_heads
    # empty() is not supported for bfloat16 on CPU.
    dtype = self.fprop_dtype
    key = jnp.zeros(
        shape=(target_max_length, target_batch_size, num_heads, dim_per_head),
        dtype=dtype)
    value = jnp.zeros(
        shape=(target_max_length, target_batch_size, num_heads, dim_per_head),
        dtype=dtype)
    cache = NestedMap(key=key, value=value)

    # Apply depth-wise convolution as in Primer.
    # Paper: https://arxiv.org/abs/2109.08668.
    if p.dconv_qkv:
      # If using depth-wise convolution, we need to cache the query.
      query = jnp.zeros(
          shape=(target_max_length, target_batch_size, num_heads, dim_per_head),
          dtype=dtype)
      cache.query = query
      # Additionally, we also cache the post depth-wise convolution queries,
      # keys and values, so that we don't need to compute the convolution for
      # previous time steps in the sequence.
      query_post_dconv = jnp.zeros(
          shape=(target_max_length, target_batch_size, num_heads, dim_per_head),
          dtype=dtype)
      cache.query_post_dconv = query_post_dconv
      key_post_dconv = jnp.zeros(
          shape=(target_max_length, target_batch_size, num_heads, dim_per_head),
          dtype=dtype)
      cache.key_post_dconv = key_post_dconv
      value_post_dconv = jnp.zeros(
          shape=(target_max_length, target_batch_size, num_heads, dim_per_head),
          dtype=dtype)
      cache.value_post_dconv = value_post_dconv

    # Apply rotary position embeddings.
    # Paper: https://arxiv.org/abs/2104.09864.
    if p.use_rotary_position_emb:
      # We only need to cache the key, since query is only needed for that
      # particular time step.
      key_post_rotary_pos_emb = jnp.zeros(
          shape=(target_max_length, target_batch_size, num_heads, dim_per_head),
          dtype=dtype)
      cache.key_post_rotary_pos_emb = key_post_rotary_pos_emb

    # Add sharding annotations for all elements in the cache.
    cache = jax.tree_map(self._shard_lbnh, cache)
    return cache

  def extend_step(self, theta: NestedMap, cached_states: NestedMap,
                  query_vec: JTensor, *, atten_mask: JTensor,
                  time_step: JTensor) -> Tuple[JTensor, NestedMap]:
    """Computes the value vector given the query of the current step.

    This function is used by autoregressive decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. Contains key of
        shape [T, B, N, H] and value of shape [T, B, N, H].
      query_vec: JTensor of shape [B, D] corresponding to query vector at index
        time_step.
      atten_mask: JTensor of shape [B, 1, T, S]. atten_mask should have already
        taken care of causal masking for decoding, plus other maskings
        necessary.
      time_step: A scalar or JTensor. Current time-step, 0-based.

    Returns:
      updated_states: A `.NestedMap` of key and value pair updated at
        `time_step`. In addition if we are using depth-wise convolution as in
        Primer, then we also update the query, as well as the queries, keys and
        values post application of depth-wise convolution at `time_step`.
      encoded: JTensor of shape [B, D] which returns the attention output at
        `time_step`.
    """
    p = self.params
    time_step = jnp.array(time_step)
    assert time_step.ndim == 0
    if p.combine_qkv:
      # Project inputs to key, value and query using a combined weight for
      # faster performance on TPU.
      new_query_proj, new_key_proj, new_value_proj = self.combined_qkv.fprop(
          theta.combined_qkv, query_vec)
    else:
      # Project inputs to key, value and query. Each has shape [B, N, H].
      new_key_proj = self.key.fprop(theta.key, query_vec)
      new_value_proj = self.value.fprop(theta.value, query_vec)
      new_query_proj = self.query.fprop(theta.query, query_vec)

    updated_state = NestedMap()
    updated_state.key = cached_states.key.at[time_step].set(new_key_proj)
    updated_state.value = cached_states.value.at[time_step].set(new_value_proj)
    # Add sharding annotations for all elements in the updated state.
    updated_state = jax.tree_map(self._shard_lbnh, updated_state)
    extended_key = updated_state.key
    extended_value = updated_state.value

    # Apply depth-wise convolution as in Primer.
    # Paper: https://arxiv.org/abs/2109.08668.
    if p.dconv_qkv:
      # Assert that the queries are also cached.
      assert 'query' in cached_states

      # Assert that queries, keys and values post dconv are also cached.
      assert 'query_post_dconv' in cached_states
      assert 'key_post_dconv' in cached_states
      assert 'value_post_dconv' in cached_states

      # Update query in cache.
      updated_state.query = cached_states.query.at[time_step].set(
          new_query_proj)

      # Aggregate depth-wise convolution for keys and values at time step.
      new_query_proj = self.dconv_q.extend_step(
          theta.dconv_q, updated_state.query, axis=0, step=time_step)
      new_key_proj = self.dconv_k.extend_step(
          theta.dconv_k, extended_key, axis=0, step=time_step)
      new_value_proj = self.dconv_v.extend_step(
          theta.dconv_v, extended_value, axis=0, step=time_step)

      # Update queries, keys and values post dconv in cache.
      updated_state.query_post_dconv = cached_states.query_post_dconv.at[
          time_step].set(new_query_proj)
      updated_state.key_post_dconv = cached_states.key_post_dconv.at[
          time_step].set(new_key_proj)
      updated_state.value_post_dconv = cached_states.value_post_dconv.at[
          time_step].set(new_value_proj)

      # Add sharding annotations for all elements in the updated state.
      updated_state = jax.tree_map(self._shard_lbnh, updated_state)
      extended_key = updated_state.key_post_dconv
      extended_value = updated_state.value_post_dconv

    # Apply rotary position embeddings.
    # Paper: https://arxiv.org/abs/2104.09864.
    if p.use_rotary_position_emb:
      new_query_proj = self.rotary_position_emb.extend_step(
          theta.rotary_position_emb, new_query_proj, time_step)
      new_key_proj = self.rotary_position_emb.extend_step(
          theta.rotary_position_emb, new_key_proj, time_step)

      # Update key post rotary position embedding in the cache.
      updated_state.key_post_rotary_pos_emb = (
          cached_states.key_post_rotary_pos_emb.at[time_step].set(new_key_proj))

      # Add sharding annotations for all elements in the updated state.
      updated_state = jax.tree_map(self._shard_lbnh, updated_state)
      extended_key = updated_state.key_post_rotary_pos_emb

    encoded, atten_prob = self._dot_atten_one_step(theta, new_query_proj,
                                                   extended_key, extended_value,
                                                   atten_mask)
    # TODO(yonghui): return atten_probs back to the caller.
    del atten_prob
    # Post projection.
    encoded = self.post.fprop(theta.post, encoded)
    encoded = self._shard_bd(encoded)
    return updated_state, encoded


class CausalDepthwiseConv1D(base_layer.BaseLayer):
  """Causal depth-wise convolution applied to a 1-d sequence as in Primer.

  See https://arxiv.org/abs/2109.08668 for more details.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'kernel_size', 3, 'Kernel size for the causal depth-wise'
        'convolution on the 1-D sequence.')
    p.Define(
        'hidden_dims', 0, 'Dimensions of the convolution filter. It can be'
        'a list to signify if we convolve multiple dimensions from'
        'the end of the sequence. Alternatively, if just convolving over'
        'the last dimension, it can be a positive integer.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    assert p.name
    assert isinstance(p.hidden_dims, list) or isinstance(p.hidden_dims, int)
    assert p.kernel_size > 0
    if isinstance(p.hidden_dims, list):
      for dim in p.hidden_dims:
        assert dim > 0
    else:
      assert p.hidden_dims > 0

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    wp = p.weight_split_dims_mapping
    for i in range(p.kernel_size):
      if i == 0:
        p.params_init = py_utils.WeightInit.Constant(0.5)
      else:
        p.params_init = py_utils.WeightInit.Constant(0.5 / p.kernel_size)
      if isinstance(p.hidden_dims, list):
        shape = p.hidden_dims
      else:
        shape = [p.hidden_dims]
      self.create_variable(
          f'dconv_{i}',
          weight_params(
              shape=shape,
              init=p.params_init,
              dtype=p.dtype,
              device_mesh=p.device_mesh,
              tensor_split_dims_mapping=wp.wt))

  def fprop(self, theta: NestedMap, inputs: JTensor, axis: int) -> JTensor:
    """FProp applying depth-wise convolution on 1D sequence.

    Args:
      theta: NestedMap containing the filter weights to apply the depth-wise
        convolution.
      inputs: Input sequence of possible shapes: [B, L, D], [B, L, N, H] or [L,
        B, N, H] where the L represents the sequence length.
      axis: The axis which corresponds to the sequence dimension, i.e. the
        dimension corresponding to L. By default the axis is assumed to be 1.

    Returns:
      Output sequence after applying the depth-wise convolution on the sequence.
    """
    p = self.params
    outputs = inputs * theta.dconv_0
    for i in range(1, p.kernel_size):
      inputs = shift_1d(inputs, offset=1, axis=axis)
      outputs += inputs * getattr(theta, f'dconv_{i}')
    return outputs

  def extend_step(self, theta: NestedMap, inputs: JTensor, axis: int,
                  step: Union[int, JTensor]) -> JTensor:
    """extend_step applying depth-wise convolution on 1D sequence at a step.

    Args:
      theta: NestedMap containing the filter weights to apply the depth-wise
        convolution.
      inputs: Input sequence of possible shapes: [B, L, D], [B, L, N, H] or [L,
        B, N, H] where the L represents the sequence length.
      axis: The axis which corresponds to the sequence dimension, i.e. the
        dimension corresponding to L. By default the axis is assumed to be 1.
      step: Which step to perform the convolution for. This must be a valid
        non-negative index into the length dimension L.

    Returns:
      Output sequence at the step after applying the depth-wise convolution
      on the sequence.
    """
    p = self.params
    get_single_slice_at_index = functools.partial(
        jax.lax.dynamic_slice_in_dim, inputs, slice_size=1, axis=axis)
    outputs = get_single_slice_at_index(start_index=step)
    outputs *= theta.dconv_0
    use_cond = isinstance(step, JTensor)
    for i in range(1, p.kernel_size):
      if use_cond:
        prev_slice = jax.lax.cond(
            jnp.greater_equal(step - i, 0),
            get_single_slice_at_index,
            lambda _: jnp.zeros_like(outputs),
            operand=step - i)
      elif step >= i:
        prev_slice = get_single_slice_at_index(start_index=step - i)
      else:
        break
      outputs += prev_slice * getattr(theta, f'dconv_{i}')
    return jnp.squeeze(outputs, axis)
