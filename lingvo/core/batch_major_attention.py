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
"""Multi-headed attention layers for Transformer machine translation.

[1] Attention is all you need.
    https://arxiv.org/pdf/1706.03762.pdf Section 3.
[2] Rethinking Attention with Performers (FAVOR mechanism).
    https://arxiv.org/abs/2009.14794
"""

import bisect
import math
import string
from typing import Optional, Tuple, Union

from absl import logging
from lingvo import compat as tf
from lingvo.core import attention_util
from lingvo.core import base_layer
from lingvo.core import builder
from lingvo.core import computation_cost
from lingvo.core import conv_layers_builder as conv_layers
from lingvo.core import favor_attention as favor
from lingvo.core import gpipe
from lingvo.core import gshard_layers
from lingvo.core import gshard_utils
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import repeat_layer
from lingvo.core import scatter_update
from lingvo.core import symbolic
from lingvo.core import tshape
import numpy as np


def CausalSegmentMask(segment_ids, dtype):
  """Computes the masks which combines causal masking and segment masks.

  Args:
    segment_ids: a tensor of shape [b, slen], the segment that each token
      belongs to.
    dtype: tf dtype.

  Returns:
    A tensor of shape [b, 1, slen, slen].
  """

  assert dtype.is_floating
  # of shape [b, t, t].
  segment_mask = tf.cast(
      tf.not_equal(
          tf.expand_dims(segment_ids, 2), tf.expand_dims(segment_ids, 1)),
      dtype=dtype)
  slen = tf.shape(segment_ids)[1]
  causal_mask = 1 - tf.linalg.band_part(
      tf.ones([slen, slen], dtype=dtype), -1, 0)
  causal_mask = tf.expand_dims(causal_mask, 0)
  combined_mask = tf.cast(tf.greater(causal_mask + segment_mask, 0.5), dtype)
  min_value = GetDtypeMin(dtype)
  return tf.expand_dims(combined_mask * min_value, 1)


def CausalPadding(slen, dtype=tf.float32):
  return 1 - tf.linalg.band_part(tf.ones([slen, slen], dtype=dtype), -1, 0)


def CrossAttentionPaddingWithTimestamp(timestamp, source_paddings, left_context,
                                       right_context):
  """Create per_step_padding for cross-attention with timestamp.

  In this cross-attention, the target (query) sequence provides an extra
  timestamp sequence. This timestamp sequence is of the same length as the
  target sequence. Each time stamp specifies, for each target position, the
  'central' position in the source (key) sequence it attends to. A window of
  the source sequence around the central position, containing left_context-1
  positions to the left, the central position itself, and right_context
  positions to the right are used for computing the attention probabilities
  and the resulting context vector.

  To give an simple example, let query=[a0, a1, a2, a3, a4] have length 5, and
  key B=[b0, b1, b2, b3] have length 4, and timestamp=[0, 1, 1, 2, 3] contains
  indices of B (note that len(timestamp)=len(A)). With left_context=1 and
  right_context=1, the padding matrix shall be (rows indicates query, column
  indicates key)::

        b0  b1  b2  b3
    a0   0   0   1   1
    a1   0   0   0   1
    a2   0   0   0   1
    a3   1   0   0   0
    a4   1   1   0   0

  Args:
    timestamp: [batch, target_seq_len]
    source_paddings: [batch, source_seq_len]
    left_context: left context size.
    right_context: right context size.

  Returns:
    paddings of shape [b, t, s], where out_paddings[i] gives the padding for
    cross-attention between target[i] and source[i].
  """

  b, t = py_utils.GetShape(timestamp)
  _, s = py_utils.GetShape(source_paddings)

  # Verify that timestamp contains valid indices for source.
  timestamp = py_utils.with_dependencies([
      py_utils.assert_equal(
          tf.reduce_all(tf.math.greater_equal(timestamp, 0)), True),
      py_utils.assert_equal(tf.reduce_all(tf.math.less(timestamp, s)), True)
  ], timestamp)

  # indices and timestamp_comp are of shape [b, t, s].
  source_indices = tf.tile(tf.reshape(tf.range(s), [1, 1, s]), [b, t, 1])
  timestamp_comp = tf.tile(tf.expand_dims(timestamp, -1), [1, 1, s])
  # Check if each index is within the range [central-L+1, central+R).
  index_mask = tf.logical_and(
      tf.greater(source_indices, timestamp_comp - left_context),
      tf.less_equal(source_indices, timestamp_comp + right_context))

  length_mask = tf.tile(
      tf.reshape(tf.cast(1 - source_paddings, tf.bool), [b, 1, s]), [1, t, 1])
  final_mask = tf.logical_and(index_mask, length_mask)
  return 1 - tf.cast(final_mask, tf.float32)


def GetDtypeMin(dtype=tf.float32):
  return tf.constant(-0.7 * dtype.max, dtype=dtype)


def SegmentMask(segment_id,
                source_segment_id,
                dtype=tf.float32,
                apply_dtype_min=True):
  """Calculates a segment mask for attention.

  Args:
    segment_id: [B, T]
    source_segment_id: [B, S]
    dtype: data type of generated mask.
    apply_dtype_min: Outputs a 0/1 padding mask if set to False. This is needed
      for GPipe layers to avoid nan issues.

  Returns:
    segment_mask: [B, 1, T, S]: A mask that is ready to
    be added to [B, N, T, S] attention logits. if apply_dtype_min is False,
    outputs a 0/1 padding mask instead.
  """
  if segment_id is None or source_segment_id is None:
    return None
  # Compute [B, T, S] = [B, T, 1] != [B, 1, S]
  ret = tf.cast(
      tf.not_equal(
          tf.expand_dims(segment_id, 2), tf.expand_dims(source_segment_id, 1)),
      dtype=dtype)
  if apply_dtype_min:
    ret *= GetDtypeMin(ret.dtype)
  # [B, T, S] -> [B, 1, T, S]
  return tf.expand_dims(ret, axis=1)


class PerDimScaleLayer(base_layer.BaseLayer):
  """A layer to scale individual dims of the input."""

  @classmethod
  def Params(cls):
    """Params for `PerDimScaleLayer`."""
    p = super().Params()
    p.Define('dim', 0, 'Number of individual dims .')
    return p

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    pc = py_utils.WeightParams(
        shape=[p.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('per_dim_scale', pc)

  def FProp(self, theta, inputs):
    """Return theta.scale * inputs.

    Args:
      theta: weights defined in this layer.
      inputs: A tensor with shape [..., p.dim].

    Returns:
      outpus: A tensor with shape [..., p.dim].
    """
    p = self.params
    with tf.name_scope(p.name):
      dim = symbolic.ToStatic(p.dim)
      expected_shape = tf.concat([py_utils.GetShape(inputs)[:-1], [dim]],
                                 axis=0)
      inputs = py_utils.HasShape(inputs, expected_shape)

      # 1.0/tf.nn.softplus(0.0) = 1.442695041. Hard code this number so that we
      # can avoid unnecessary XLA op fusion mess on TPU.
      r_softplus_0 = 1.442695041
      if isinstance(dim, int) or isinstance(dim, float):
        scale = tf.constant(r_softplus_0 / np.sqrt(dim), dtype=inputs.dtype)
      else:
        scale = tf.cast(
            tf.math.rsqrt(tf.cast(dim, tf.float32)) * r_softplus_0,
            dtype=inputs.dtype)

      scale *= tf.nn.softplus(theta.per_dim_scale)
      return inputs * scale

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(
        flops=inputs.num_elements() * 5, out_shapes=(inputs,))


class MultiHeadedProjectionLayer(quant_utils.QuantizableLayer):
  """Layer that computes multi heads projection.

    This layer is expected to be used within MultiHeadedAttention below.
  """

  @classmethod
  def Params(cls):
    """Params for MultiHeadedProjectionLayer."""
    p = super().Params()
    p.Define('input_dim', 0, 'Input dimension.')
    p.Define('num_heads', 0, 'Number of heads.')
    p.Define('dim_per_head', 0, 'Size of each head.')
    p.Define(
        'is_output_projection', False,
        'Whether it is out projection or not. If False, we use '
        '"...D,DNH->...NH" for query,key,value projection. Otherwise we use '
        '"...NH,DNH->...D" for output projection.')
    p.Define(
        'make_output_proj_no_op', False, 'If True no output projection is '
        'applied. This should be set with is_output_projection True and will '
        'raise an error otherwise. This does not effect input projections. '
        'This is useful in combining different types of attention heads where'
        'mixing is done after getting all the different attention outputs.')
    p.Define('use_bias', True, 'If to add bias in projection.')
    return p

  def __init__(self, params):
    """Constructs a MultiHeadedProjectionLayer object."""
    super().__init__(params)
    p = self.params
    feature_axis = 0 if p.is_output_projection else (-2, -1)
    self.CreateAqtWeight(
        'w',
        shape=[p.input_dim, p.num_heads, p.dim_per_head],
        feature_axis=feature_axis,
        legacy_aqt_w_name='aqt_w')

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    if p.device_mesh is not None:
      assert p.weight_split_dims_mapping is not None, self.path
    if p.make_output_proj_no_op and not p.is_output_projection:
      raise ValueError('make_output_proj_no_op must be used with output '
                       'projection set to True.')
    if p.make_output_proj_no_op:
      return
    pc = py_utils.WeightParams(
        shape=[p.input_dim, p.num_heads, p.dim_per_head],
        init=p.params_init,
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=p.weight_split_dims_mapping,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('w', pc)
    if p.use_bias:
      if p.is_output_projection:
        if p.device_mesh is not None:
          bias_split_dims_mapping = [p.weight_split_dims_mapping[0]]
        else:
          bias_split_dims_mapping = None
        pc_bias = py_utils.WeightParams(
            shape=[p.input_dim],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=bias_split_dims_mapping,
            collections=[self.__class__.__name__ + '_vars'])
      else:
        if p.device_mesh is not None:
          bias_split_dims_mapping = [
              p.weight_split_dims_mapping[1], p.weight_split_dims_mapping[2]
          ]
        else:
          bias_split_dims_mapping = None
        pc_bias = py_utils.WeightParams(
            shape=[p.num_heads, p.dim_per_head],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=bias_split_dims_mapping,
            collections=[self.__class__.__name__ + '_vars'])
      self.CreateVariable('b', pc_bias)

  def FProp(self, theta, inputs):
    """Computes the multi headed projection for inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A tensor of shape [..., num_heads, dim_per_head] or [...,
        hidden_size].

    Returns:
      The projected tensor with shape [..., hidden_size] or
      [..., num_heads, dim_per_head].
    """

    # Because tf.einsum is not fully optimized unless all the dimensions are
    # fully specified, we have to avoid using '...' for batch dimensions in the
    # equation in tf.einsum for optimized performance. This is only feasible
    # when the rank of the tensor is known.
    # Sort the available symbols to avoid nondeterminism.
    eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('DHN')))
    shape = py_utils.GetShape(inputs)
    rank = None if isinstance(shape, tf.Tensor) else len(shape)

    p = self.params
    with tf.name_scope(p.name):
      inputs = self._CastToFPropDtype(inputs)
      if p.make_output_proj_no_op:
        return inputs

      if p.is_output_projection:
        expected_shape = tf.concat(
            [py_utils.GetShape(inputs)[:-2], [p.num_heads, p.dim_per_head]],
            axis=0)
        inputs = py_utils.HasShape(inputs, expected_shape)
        batch_eqn = eqn_sym[:(rank - 2)] if rank else '...'
        eqn = f'{batch_eqn}NH,DNH->{batch_eqn}D'
        out_feature_axis = 0
      else:
        expected_shape = tf.concat(
            [py_utils.GetShape(inputs)[:-1], [p.input_dim]], axis=0)
        inputs = py_utils.HasShape(inputs, expected_shape)
        batch_eqn = eqn_sym[:(rank - 1)] if rank else '...'
        eqn = f'{batch_eqn}D,DNH->{batch_eqn}NH'
        out_feature_axis = (-2, -1)
      theta = theta.Transform(lambda x: tf.cast(x, py_utils.FPropDtype(p)))
      inputs, w = self.ToAqtInputs(
          'w', act=inputs, weight=theta.w, w_feature_axis=out_feature_axis)
      ret = tf.einsum(eqn, inputs, w)
      ret = self.FromAqtMatmul('w', ret)
      if p.use_bias:
        ret += theta.b
      return ret


# TODO(shibow/wangtao) remove this after b/174094694 is done.
class ReshapedMultiHeadedProjectionLayer(MultiHeadedProjectionLayer):
  """MultiHeadedProjectionLayer with model dim D reshaped as Md."""

  def FProp(self, theta, inputs):
    """Computes the multi headed projection for inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A tensor of shape [batch_size, time_steps, num_heads,
        dim_per_head] or [batch_size, time_steps, dim_reshape_segments,
        hidden_size // dim_reshape_segments].

    Returns:
      The projected tensor with shape [batch_size, time_steps,
      dim_reshape_segments, hidden_size // dim_reshape_segments] or
      [batch_size, time_steps, num_heads, dim_per_head].
    """
    p = self.params
    assert p.device_mesh is not None
    assert p.device_mesh.ndim >= 2
    with tf.name_scope(p.name):
      inputs = self._CastToFPropDtype(inputs)
      if p.make_output_proj_no_op:
        return inputs
      theta.w = gshard_utils.ReshapeDim(theta.w, 0, p.device_mesh.shape[1])
      if p.is_output_projection:
        inputs = py_utils.HasShape(
            inputs, [-1, -1, p.num_heads,
                     symbolic.ToStatic(p.dim_per_head)])
        ret = tf.einsum('BTNH,MdNH->BTMd', inputs, theta.w)
      else:
        ret = tf.einsum('BTMd,MdNH->BTNH', inputs, theta.w)
      if p.use_bias:
        if p.is_output_projection:
          theta.b = gshard_utils.ReshapeDim(theta.b, 0, p.device_mesh.shape[1])
        ret += theta.b
      return ret


class MultiHeadedAttention(quant_utils.QuantizableLayer):
  """Dot-product attention with multiple attention heads.

  This implementation heavily uses einsum (wrapped in py_utils.Einsum) to be
  efficient on TPUs.  We use the following capital letters to denote certain
  tensor parameters.

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    D = model dimension
    N = number of attention heads
    H = dimensions of each attention head.

  The algorithm is sketched as follows. Each intermediate tensor or weight
  tensor is annotated with its shape. E.g., Wq, the weight tensor for query's
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
  def Params(cls):
    """Params for _MultiHeadedAttention."""
    p = super().Params()
    p.Define(
        'input_dim', 0,
        'An integer or a dict of integer values as number of input nodes. If '
        'input_dim is a dict, keys must be key, value and query.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    p.Define('num_heads', 1, 'Num of attention heads.')
    p.Define(
        'dim_per_head', None, 'Hidden dim of each attention head. If None, '
        'defaults to p.hidden_dim // p.num_heads')
    p.Define('dropout_tpl', layers.DropoutLayer.Params(),
             'Params for dropout layer.')
    p.Define(
        'enable_value_proj', True, 'Whether value v is pre-projected '
        ' before self attention or not.')
    p.Define('enable_query_scale', True, 'Enable scaling of query vector.')
    p.Define(
        'enable_per_dim_scale', True,
        'Whether using per_dim_scale or scaling by a constant factor. '
        'Only applied when enable_query_scale == True.')
    p.Define('atten_dropout_prob', 0.0,
             'Probability at which we apply dropout to the attention weights.')
    p.Define('proj_tpl', MultiHeadedProjectionLayer.Params(), 'Params for '
             'projection layer.')
    p.Define('packed_input', False, 'Whether there is packed input.')
    p.Define('use_bias', True, 'Whether to use bias for projection layers.')
    p.Define(
        'enable_scaling_code_motion', False, 'Move scalings from the side '
        'of T^2 to the side of T for better performance. This may result '
        'in model quality drops when using bf16 for some models due to '
        'different XLA fusion decisions.')
    p.Define(
        'atten_extra_logit', None, 'Extra logit for attention softmax.'
        'Notice None and 0 are different.')
    p.Define(
        'atten_logit_cap', 0.0, 'Cap the absolute values of logits by '
        'tanh. Enabled when a positive value is specified. May not be '
        'supported by a subclass.')
    # SPMD partition related params.
    #
    # d - model_dim
    # n - num_heads
    # h - attention_dim_per_heads
    # b - batch_size
    # l - seq_len
    p.activation_split_dims_mapping = hyperparams.Params()
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

  def __init__(self, params):
    """Constructs a _MultiHeadedAttention object."""
    super().__init__(params)
    p = self.params
    assert p.input_dim, f'input_dim is {p.input_dim}'
    assert p.hidden_dim, f'hidden_dim is {p.hidden_dim}'
    assert p.num_heads > 0, f'num_heads is {p.num_heads}'
    # if proj_tpl does not have dim_per_head set, set it
    if p.proj_tpl.dim_per_head == 0:
      p.proj_tpl.dim_per_head = self.dim_per_head

    if p.device_mesh is not None:
      assert p.weight_split_dims_mapping is not None
      assert p.activation_split_dims_mapping is not None

    def ProjectInput(input_dim):
      return p.proj_tpl.Copy().Set(
          input_dim=input_dim,
          num_heads=p.num_heads,
          use_bias=p.use_bias,
          device_mesh=p.device_mesh,
          weight_split_dims_mapping=p.weight_split_dims_mapping,
          make_output_proj_no_op=False)

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

    self.CreateChild('key', ProjectInput(key_input_dim))
    self.CreateChild('query', ProjectInput(query_input_dim))
    if p.enable_value_proj:
      assert value_input_dim, f'value_input_dim is {value_input_dim}'
      self.CreateChild('value', ProjectInput(value_input_dim))
    if p.enable_query_scale and p.enable_per_dim_scale:
      self.CreateChild(
          'per_dim_scale',
          PerDimScaleLayer.Params().Set(dim=p.proj_tpl.dim_per_head))
    self.CreateChild('atten_dropout',
                     p.dropout_tpl.Set(keep_prob=1.0 - p.atten_dropout_prob))
    # Setting is_output_projection=True to set the projection direction
    # from hidden dim to input dim. Output projection follows query_input_dim.
    self.CreateChild(
        'post',
        p.proj_tpl.Copy().Set(
            input_dim=query_input_dim,
            num_heads=p.num_heads,
            is_output_projection=True,
            use_bias=p.use_bias,
            device_mesh=p.device_mesh,
            weight_split_dims_mapping=p.weight_split_dims_mapping))

  @property
  def dim_per_head(self):
    """Returns the dimension per attention head."""
    p = self.params
    return p.dim_per_head or p.hidden_dim // p.num_heads

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

  def _AttenLogits(self, theta, query, key):
    """Computes attention logits.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query: A Tensor of shape [B, T, N, H]
      key: A Tensor of shape [B, T, N, H]

    Returns:
      A Tensor of shape [B, N, T, S]
    """
    query, key = self.ToAqtActActInputs(query, key)
    logits = attention_util.AttenLogits(query, key)
    logits = self.FromAqtActActMatmul(logits)
    return self._CapLogits(logits)

  def _AttenLogitsOneStep(self, theta, query, key, time_step):
    """Attention logits for one single target (query) step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, N, H].
      key:      [S, B, N, H] or [S, B, N*H/128, 128].
      time_step: Current time step.

    Returns:
      A Tensor of shape [S, B, N]
    """
    s, b, _, _ = py_utils.GetShape(key, 4)
    _, n, h = py_utils.GetShape(query, 3)
    # [s, b, n]
    key = tf.reshape(key, [s, b, n, h])
    query, key = self.ToAqtActActInputs(query, key)
    logits = tf.einsum('BNH,SBNH->SBN', query, key)
    logits = self.FromAqtActActMatmul(logits)
    return self._CapLogits(logits)

  def AttenProbs(self,
                 theta,
                 query,
                 key,
                 paddings,
                 segment_mask,
                 per_step_padding=None):
    """Compute attention probability.

    Note: We can currently pass a mask through both `segment_mask` and
    `per_step_padding`. `segment_mask` was initially added to deal with packed
    inputs, while `per_step_padding` was added to deal with causal or
    non-uniform padding. Ideally, we should merge these two arguments into a
    single `mask`-like argument, which can be used simultaneously for both
    purposes. This requires to propagate the changes accordingly in all
    downstream methods and layers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, T, N, H].
      key:      [B, S, N, H].
      paddings: [B, S].
      segment_mask: [B, 1, T, S]: A mask that is applied to prevent attention
        between different segments. This is already been converted into large
        negative logits. Only applied if packed_input = True.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, T, S] if
        not None.

    Returns:
      probs: [B, N, T, S].
      probs_sum: [B, N, T, 1].
    """
    p = self.params

    key = py_utils.HasRank(key, 4)
    b, s, n, h = py_utils.GetShape(key, 4)
    query = py_utils.HasShape(query, [b, -1, n, h])
    t = py_utils.GetShape(query)[1]
    if segment_mask is not None and self.params.packed_input:
      segment_mask = py_utils.HasShape(segment_mask, [b, 1, t, s])

    with tf.name_scope('logits'):
      # Keep softmax computation in float32 otherwise the low precision can
      # can lead to worse quality.
      logits = tf.cast(self._AttenLogits(theta, query, key), tf.float32)

    # Apply segment mask.
    if self.params.packed_input and segment_mask is not None:
      # Paddings have been included in segment_mask.
      padded_logits = logits + tf.cast(segment_mask, tf.float32)
    else:
      # Exclude padding frames.
      paddings = py_utils.HasShape(paddings, [b, s])
      paddings = tf.tile(tf.reshape(paddings, [b, 1, 1, s]), [1, n, t, 1])
      if per_step_padding is not None:
        per_step_padding = tf.tile(
            tf.expand_dims(per_step_padding, 1), [1, n, 1, 1])
        paddings += per_step_padding

      padded_logits = py_utils.ApplyPadding(paddings > 0.0, logits,
                                            GetDtypeMin(logits.dtype))

    if self.params.enable_scaling_code_motion:
      # Split the softmax into two parts. Do the 1st part here; the 2nd part
      # (scaling) is moved after _AttenContext for better performance.
      probs = padded_logits - tf.stop_gradient(
          tf.reduce_max(padded_logits, -1, True))
      probs = tf.exp(probs)
      probs_sum = tf.reduce_sum(probs, -1, True)
      probs = tf.cast(probs, key.dtype)
      probs_sum = tf.cast(probs_sum, key.dtype)
    else:
      probs = tf.cast(
          py_utils.Softmax(padded_logits, extra_logit=p.atten_extra_logit),
          key.dtype)
      probs_sum = None

    probs = py_utils.HasShape(probs, [b, n, t, s])
    return probs, probs_sum

  def _AttenContext(self, theta, probs, value):
    probs, value = self.ToAqtActActInputs(
        probs,
        value,
        act_lhs_distribution='positive',
        act_rhs_distribution='symmetric')
    encoded = attention_util.AttenContext(probs, value)
    return self.FromAqtActActMatmul(encoded)

  def _AttenContextOneStep(self, theta, probs, value, time_step, h):
    s, b, _, _ = py_utils.GetShape(value, 4)
    _, _, n = py_utils.GetShape(probs, 3)
    value = tf.reshape(value, [s, b, n, h])
    probs, value = self.ToAqtActActInputs(
        probs,
        value,
        act_lhs_distribution='positive',
        act_rhs_distribution='symmetric')
    encoded = tf.einsum('SBN,SBNH->BNH', probs, value)
    return self.FromAqtActActMatmul(encoded)

  def _DotAtten(self,
                theta,
                query,
                key,
                value,
                paddings,
                segment_mask,
                per_step_padding=None):
    """Main attention function.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, T, N, H].
      key:      [B, S, N, H].
      value:    [B, S, N, H].
      paddings: [B, S].
      segment_mask: [B, 1, T, S]: A mask that is applied to prevent attention
        between different segments. This is already been converted into large
        negative logits. Only applied if packed_input = True.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, T, S] if
        not None.

    Returns:
      encoded: [B, T, N, H].
      atten_probs: [B, N, T, S].
    """
    p = self.params
    # Scale the query projection.
    if p.enable_query_scale:
      if p.enable_per_dim_scale:
        query = self.per_dim_scale.FProp(theta.per_dim_scale, query)
      else:
        query *= (p.hidden_dim // p.num_heads)**-0.5

    # Compute prob with shape [batch, heads, target_time, source_time].
    with tf.name_scope('probs'):
      probs, probs_sum = self.AttenProbs(theta, query, key, paddings,
                                         segment_mask, per_step_padding)
      # Apply dropout to probs.
      probs = self.atten_dropout.FProp(theta.atten_dropout, probs)

    # Compute the attention context vector.
    with tf.name_scope('ctx'):
      encoded = self._AttenContext(theta, probs, value)
      if p.enable_scaling_code_motion:
        # The 2nd part of the softmax --- scaling.
        encoded = encoded / tf.transpose(probs_sum, [0, 2, 1, 3])

    encoded = gshard_utils.MeshSplit(encoded, p.device_mesh,
                                     p.activation_split_dims_mapping.blnh)
    return encoded, probs

  def _DotAttenOneStep(self,
                       theta,
                       query,
                       key,
                       value,
                       paddings,
                       segment_mask,
                       per_step_padding=None,
                       time_step=None,
                       use_short_seq_opt=False):
    """Dot attention function for queries with 1 time step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, 1, N, H].
      key:      [S, B, N, H] or [S, B, N*H/128, 128].
      value:    [S, B, N, H] or [S, B, N*H/128, 128].
      paddings: [B, S].
      segment_mask: [B, 1, T, S]: A mask that is applied to prevent attention
        between different segments. This is already been converted into large
        negative logits. Only applied if packed_input = True.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, 1, S] if
        not None.
      time_step: Current time step.
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      encoded: [B, 1, N, H].
    """
    p = self.params
    # Scale the query projection.
    if p.enable_query_scale:
      if p.enable_per_dim_scale:
        query = self.per_dim_scale.FProp(theta.per_dim_scale, query)
      else:
        query *= (p.hidden_dim // p.num_heads)**-0.5

    key = py_utils.HasRank(key, 4)

    b, t, n, h = py_utils.GetShape(query, 4)
    s, b, _, _ = py_utils.GetShape(key, 4)
    paddings = py_utils.HasShape(paddings, [b, s])
    assert t == 1, query

    if per_step_padding is not None:
      paddings += tf.squeeze(per_step_padding, 1)

    query = tf.reshape(query, [b, n, h])
    pad = tf.reshape(
        tf.tile(tf.expand_dims(tf.transpose(paddings), 2), [1, 1, n]), [s, -1])

    def _LongSeq():
      """For long sequence, directly apply to the entire tensor with padding."""
      logits = self._AttenLogitsOneStep(theta, query, key, time_step)

      logits = tf.reshape(logits, [s, -1])
      padded_logits = py_utils.ApplyPadding(pad > 0.0, logits,
                                            GetDtypeMin(logits.dtype))
      probs = py_utils.Softmax(
          padded_logits, axis=0, extra_logit=p.atten_extra_logit)
      probs = tf.reshape(probs, [s, b, n])

      encoded = self._AttenContextOneStep(theta, probs, value, time_step, h)
      return tf.expand_dims(encoded, 1)

    def _ShortSeq():
      """For short sequence, using while loop for early exit."""

      def _AttenStep(o, k, q, ts):
        """Computes logits for attention prob for one step.

        Args:
          o: the output logits of shape [S, B*N]
          k: cached key of shape [S, B, N*H/128, 8]
          q: query of shape [B, N, H]
          ts: a scala tensor to represent time_step

        Returns:
          Updated logits and time steps.
        """
        ot = tf.reshape(
            tf.reduce_sum(tf.reshape(tf.gather(k, ts), [-1, n, h]) * q, -1),
            [-1])
        return scatter_update.Update(o, ts, ot), k, q, ts + 1

      # Prefix with 'quant_' to avoid reassigning variables captured via closure
      quant_key, quant_query = self.ToAqtActActInputs(key, query)
      logits, _, _, _ = tf.while_loop(
          lambda _o, _k, _q, ts: ts <= time_step,
          _AttenStep,
          loop_vars=(
              tf.Empty([s, b * n], query.dtype, init=True),
              quant_key,
              quant_query,
              tf.zeros([], tf.int32),
          ))
      logits = self.FromAqtActActMatmul(logits)
      logits = self._CapLogits(logits)

      padded_logits = py_utils.ApplyPadding(pad > 0.0, logits,
                                            GetDtypeMin(logits.dtype))
      probs = py_utils.Softmax(
          padded_logits, axis=0, extra_logit=p.atten_extra_logit)

      def _DotStep(o, p, v, ts):
        """Computes encoded activation.

        Args:
          o: the output activation of shape [B, N, H]
          p: probability of shape [S, B*N]
          v: cached value of shape [S, B, N*H/128, 8]
          ts: a scala tensor to represent time_step

        Returns:
          Updated output and time steps.
        """
        return o + tf.reshape(tf.gather(p, ts), [-1, n, 1]) * tf.reshape(
            tf.gather(v, ts), [-1, n, h]), p, v, ts + 1

      encoded, _, _, _ = tf.while_loop(
          lambda o, p, v, ts: ts <= time_step,
          _DotStep,
          loop_vars=(tf.zeros([b, n, h],
                              probs.dtype), probs, value, tf.zeros([],
                                                                   tf.int32)))
      return tf.expand_dims(encoded, 1)

    return _ShortSeq() if use_short_seq_opt else _LongSeq()

  def FProp(self,
            theta,
            query_vec,
            key_vec,
            value_vec,
            paddings,
            segment_mask=None,
            per_step_padding=None):
    """Computes the value vector given the current query output.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [B, T, D].
      key_vec:   [B, S, D].
      value_vec: [B, S, D].
      paddings:  [B, S].
      segment_mask: [B, 1, T, S]. A mask only applied if packed_input=True.
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

    # Project inputs to key, value and query, respectively has shape
    # [B, S, N, H], [B, S, N, H], and [B, T, N, H].
    query_proj = self.query.FProp(theta.query, query_vec)
    key_proj = self.key.FProp(theta.key, key_vec)
    if p.enable_value_proj:
      value_proj = self.value.FProp(theta.value, value_vec)
    else:
      with tf.name_scope('value'):
        h = p.num_heads
        _, _, d = py_utils.GetShape(value_vec, 3)
        dh = self.dim_per_head
        # TODO(b/119531146): Reshape is inefficient here. Use one-hot matmul
        # avoids the data formatting. Change this back to reshape once XLA
        # has optimized reshape performance.
        rhs = tf.reshape(
            tf.one_hot(tf.range(d) // dh, h, dtype=value_vec.dtype),
            [d, h, 1]) * tf.reshape(
                tf.one_hot(tf.range(d) % dh, dh, dtype=value_vec.dtype),
                [d, 1, dh])

        value_vec, rhs = self.ToAqtActActInputs(value_vec, rhs)
        value_proj = tf.einsum('BTD,DNH->BTNH', value_vec, rhs)
        value_proj = self.FromAqtActActMatmul(value_proj)

    query_proj = gshard_utils.MeshSplit(query_proj, p.device_mesh,
                                        p.activation_split_dims_mapping.blnh)
    key_proj = gshard_utils.MeshSplit(key_proj, p.device_mesh,
                                      p.activation_split_dims_mapping.blnh)
    value_proj = gshard_utils.MeshSplit(value_proj, p.device_mesh,
                                        p.activation_split_dims_mapping.blnh)

    if p.packed_input and not self.do_eval:
      assert segment_mask is not None
    encoded, atten_probs = self._DotAtten(theta, query_proj, key_proj,
                                          value_proj, paddings, segment_mask,
                                          per_step_padding)
    # Post projection
    encoded = self.post.FProp(theta.post, encoded)

    # Shard the output
    encoded = gshard_utils.MeshSplit(encoded, p.device_mesh,
                                     p.activation_split_dims_mapping.bld)
    return encoded, atten_probs

  def InitStates(self, theta, target_batch_size, target_max_length):
    """Initializes the decoding states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      target_batch_size: The target batch size B.
      target_max_length: The target maximum length T.

    Returns:
      key:   [T, B, N, H].
      value: [T, B, N, H].
    """
    p = self.params
    num_heads = p.num_heads
    dim_per_head = self.dim_per_head
    # empty() is not supported for bfloat16 on CPU.
    dtype = py_utils.FPropDtype(p)
    if dtype == tf.bfloat16 and not py_utils.use_tpu():
      dtype = tf.float32
    # TODO(shafey): Determine if we want to make the cached shape 128 to
    # avoid padding and more efficient interpolation in beamsearch.
    return py_utils.NestedMap(
        key=tf.Empty(
            shape=(target_max_length, target_batch_size, num_heads,
                   dim_per_head),
            dtype=dtype,
            init=True),
        value=tf.Empty(
            shape=(target_max_length, target_batch_size, num_heads,
                   dim_per_head),
            dtype=dtype,
            init=True))

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 paddings,
                 segment_mask,
                 per_step_padding,
                 time_step,
                 use_short_seq_opt=False):
    """Computes the value vector given the query of the current step.

    This function is used by autoregressive decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:        [B, 1, D].
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. key   - [T, B,
        N, H]. value - [T, B, N, H].
      paddings:         [B, T], or None if there is no padding.
      segment_mask:     [B, 1, T, S] or None.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, 1, T] if
        not None.
      time_step: A scalar or tensor with [B], current decode step, 0-based. if
        it's a scalar, all the time step are the same decode step. if it's a
        tensor, it represents current decode step for each sample.
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      encoded:           [B, 1, D].
      updated_key_vec:   [T, B, N, H].
      updated_value_vec: [T, B, N, H].

    Raises:
      ValueError: If value projection is disabled.
    """
    p = self.params
    if not p.enable_value_proj:
      raise ValueError('Value projection must be enabled for Transformer '
                       'machine translation.')

    time_step = tf.convert_to_tensor(time_step)
    synced_time_step = (time_step.shape.ndims == 0)
    t, b, n, h = py_utils.GetShape(cached_states.key, 4)

    # Project inputs to key, value and query. Each has shape [B, 1, N, H].
    new_key_proj = self.key.FProp(theta.key, query_vec)
    new_value_proj = self.value.FProp(theta.value, query_vec)
    query_proj = self.query.FProp(theta.query, query_vec)

    # Using a if condition, in case it's more efficient to update the same index
    new_key_proj = tf.cast(
        tf.reshape(new_key_proj, [b, n, h]), dtype=cached_states.key.dtype)
    new_value_proj = tf.cast(
        tf.reshape(new_value_proj, [b, n, h]), dtype=cached_states.value.dtype)
    if synced_time_step:
      # The extended_key and extended_value have shape [T, B, N, H].
      extended_key = scatter_update.Update(cached_states.key, time_step,
                                           new_key_proj)
      extended_value = scatter_update.Update(cached_states.value, time_step,
                                             new_value_proj)
    else:
      # The extended_key and extended_value have shape [T, B, N, H].
      selected_indices = tf.range(b) + time_step * b
      extended_key = scatter_update.Update(
          tf.reshape(cached_states.key, [-1, n, h]), selected_indices,
          new_key_proj)
      extended_value = scatter_update.Update(
          tf.reshape(cached_states.value, [-1, n, h]), selected_indices,
          new_value_proj)
      extended_key = tf.reshape(extended_key, [t, b, n, h])
      extended_value = tf.reshape(extended_value, [t, b, n, h])
    updated_state = py_utils.NestedMap(key=extended_key, value=extended_value)

    if paddings is None:
      paddings = tf.zeros([b, t], dtype=query_vec.dtype)

    encoded = self._DotAttenOneStep(
        theta,
        query_proj,
        self._CastToFPropDtype(extended_key),
        self._CastToFPropDtype(extended_value),
        paddings,
        segment_mask,
        per_step_padding,
        time_step=time_step,
        use_short_seq_opt=use_short_seq_opt)

    # Post projection.
    encoded = self.post.FProp(theta.post, encoded)
    return encoded, updated_state

  @classmethod
  def FPropMeta(cls, p, *args):
    # args[0]: [b, t, d], args[1]: [b, s, d], args[2]: [b, s, d],
    # args[3]: [b, s], args[4]: [b, t, s] if not None
    args = tuple(py_utils.Flatten(args))
    py_utils.CheckShapes(args)
    b, t, d = args[0]
    s = args[3][1]
    n = p.num_heads
    # O(b * t * s * d) computation for self-attention and there are four
    # projection layers, two of which has O(b * t * d^2), the other two has
    # O(b * s * d^2). Each multiple-sum took 2 flops. Approximately
    # self_attention took 15 flops per element since softmax is expensive.
    flops = 15 * b * t * s * d + 2 * 2 * (b * t * d * d + b * s * d * d)
    return py_utils.NestedMap(flops=flops, out_shapes=(args[0], (b, n, t, s)))


# TODO(shibow/wangtao) remove this after b/174094694 is done.
class ReshapedMultiHeadedAttention(MultiHeadedAttention):
  """MultiHeadedAttention with model dim D reshaped as Md."""

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 paddings,
                 segment_mask,
                 per_step_padding,
                 time_step,
                 use_short_seq_opt=False):
    """Computes the value vector given the query of the current step.

    This function is used by autoregressive decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:        [B, 1, D].
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. key   - [T, B,
        N, H]. value - [T, B, N, H].
      paddings:         [B, T], or None if there is no padding.
      segment_mask:     [B, 1, T, S] or None.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, 1, T] if
        not None.
      time_step: A scalar or tensor with [B], current decode step, 0-based. if
        it's a scalar, all the time step are the same decode step. if it's a
        tensor, it represents current decode step for each sample.
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      encoded:           [B, 1, D].
      updated_key_vec:   [T, B, N, H].
      updated_value_vec: [T, B, N, H].

    Raises:
      ValueError: If value projection is disabled.
    """
    p = self.params
    with tf.name_scope(p.name):
      query_vec = gshard_utils.ReshapeDim(query_vec, 2, p.device_mesh.shape[1])
      encoded, updated_states = super().ExtendStep(
          theta,
          query_vec,
          cached_states,
          paddings,
          segment_mask,
          per_step_padding,
          time_step,
          use_short_seq_opt=use_short_seq_opt)
      encoded_shape = py_utils.GetShape(encoded, 2)
      shape = encoded_shape + [-1]
      encoded = tf.reshape(encoded, shape)
      return encoded, updated_states


class MultiHeadedFavorAttention(MultiHeadedAttention):
  """Performer multiheaded attention."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_random_features', 384,
             'Number of random projection features for performer.')
    p.Define('attention_type', 'softmax',
             'relu|softmax, performer kernel transformation methods')
    p.Define('redraw', False,
             'Whether kernel features should be redrawn (N/A if not random).')
    return p

  def _DotAtten(self,
                theta,
                query,
                key,
                value,
                paddings,
                segment_mask,
                per_step_padding=None):
    """Main FAVOR attention function from Rethinking Attention with Performers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, T, N, H].
      key:      [B, S, N, H].
      value:    [B, S, N, H].
      paddings: [B, S].
      segment_mask: [B, 1, T, S]: A mask that is applied to prevent attention
        between different segments. This is already been converted into large
        negative logits. Only applied if packed_input = True.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, T, S] if
        not None.

    Returns:
      encoded: [B, T, N, H].
      atten_probs: None for FAVOR attention.
    """
    # TODO(kchoro): Add segment_mask support when FAVOR mechanism is applied
    # and use causal FAVOR from favor_attention library when per_step_padding is
    # not None.
    p = self.params
    assert p.device_mesh is None, 'GShard mesh splits not supported.'
    assert not p.packed_input, 'Packed input not supported.'
    # Scale the query projection.
    if p.enable_query_scale and p.enable_per_dim_scale:
      query = self.per_dim_scale.FProp(theta.per_dim_scale, query)

    if p.attention_type == 'relu':
      kernel_transformation = favor.relu_kernel_transformation
      encoded = favor.favor_attention(query, key, value, paddings,
                                      kernel_transformation, False)
    elif p.attention_type == 'softmax':
      kernel_transformation = favor.softmax_kernel_transformation
      # TODO(kchoro): Add the option of redrawing projection matrices. This
      # improves in several applications.
      projection_matrix = favor.create_projection_matrix(
          p.num_random_features, query.shape[-1], None if p.redraw else 0)
      encoded = favor.favor_attention(query, key, value, paddings,
                                      kernel_transformation, False,
                                      projection_matrix)
    elif p.attention_type == 'cossim':
      # TODO(kchoro): Add paddings to the cossim variant.
      projection_matrix = favor.create_projection_matrix(
          p.num_random_features, query.shape[-1], None if p.redraw else 0)
      key_prime = favor.cossim_kernel_transformation(key, False,
                                                     projection_matrix, 0.0,
                                                     p.num_random_features)
      query_prime = favor.cossim_kernel_transformation(query, True,
                                                       projection_matrix, 0.0,
                                                       p.num_random_features)
      attention_scores = tf.einsum('BXHD,BYHD->BXYH', query_prime, key_prime)
      attention_scores = tf.nn.softmax(attention_scores, axis=2)
      encoded = tf.einsum('BXYH,BYHD->BXHD', attention_scores, value)
    else:
      logging.info(
          'FAVOR attention type: %s is not supported,returning query tensor.',
          p.attention_type)
      return query, None

    return encoded, None


class MultiHeadedAttentionXL(MultiHeadedAttention):
  """Transformer-XL multiheaded attention with relative positional embedding.

  https://arxiv.org/pdf/1901.02860.pdf section 3.3.

  Notice this is only intended for self attention.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('rel_pos_emb_dim', None,
             'Dimension of relative positional embedding.')
    p.Define('skip_term_b', False,
             'If True, skip term_b in the paper section 3.3.')
    p.Define('pos_atten_logits_tpl',
             attention_util.PositionalAttenLogits.Params(),
             'Params for the positional attention logits.')
    return p

  def __init__(self, params):
    """Constructs a MultiHeadedAttentionXL object."""
    super().__init__(params)
    params = self.params

    assert not params.packed_input, 'Packed input not implemented yet.'

    if params.rel_pos_emb_dim is None or params.rel_pos_emb_dim <= 0:
      raise ValueError('Invalid rel_pos_emb_dim: %s' % params.rel_pos_emb_dim)

    emb_params = layers.PositionalEmbeddingLayer.Params().Set(
        embedding_dim=params.rel_pos_emb_dim)
    self.CreateChild('pos_emb', emb_params)

    # Projection layer for relative position encoding
    dim_per_head = params.hidden_dim // params.num_heads
    pos_proj_tpl = params.proj_tpl.Copy().Set(
        input_dim=params.rel_pos_emb_dim,
        num_heads=params.num_heads,
        dim_per_head=dim_per_head,
        use_bias=False)
    self.CreateChild('pos_proj', pos_proj_tpl)
    self.CreateChild('pos_atten_logits', params.pos_atten_logits_tpl)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    params = self.params

    dim_per_head = params.hidden_dim // params.num_heads
    u_pc = py_utils.WeightParams(
        shape=[params.num_heads, dim_per_head],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=params.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    v_pc = py_utils.WeightParams(
        shape=[params.num_heads, dim_per_head],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=params.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    self.CreateVariable('u', u_pc)
    self.CreateVariable('v', v_pc)

  def _AttenLogits(self, theta, query, key):
    b, _, n, h = py_utils.GetShape(key, 4)
    t = py_utils.GetShape(query)[1]

    # This layer only supports self attention.
    key = py_utils.HasShape(key, [b, t, n, h])

    # [1, 2T - 1]
    pos = tf.expand_dims(tf.range(-(t - 1), t, name='relative_pos'), 0)
    sin_emb = self.pos_emb.FPropWithPosition(theta.pos_emb, pos)
    # [1, 2T - 1, N, H]
    sin_emb = self.pos_proj.FProp(theta.pos_proj, sin_emb)
    # [2T - 1, N, H]
    sin_emb = tf.squeeze(sin_emb, 0)

    logits = self.pos_atten_logits.AttenLogitsXL(
        query,
        key,
        abs_pos_emb=sin_emb,
        content_bias=theta.u,
        positional_bias=theta.v,
        skip_term_b=self.params.skip_term_b)

    return logits

  def _AttenLogitsOneStep(self, theta, query, key, time_step):
    """Attention logits for one single target (query) step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, N, H].
      key:      [S, B, N, H] or [S, B, N*H/128, 128].
      time_step: Current time step. if it's a scalar, all the time step are the
        same decode step. if it's a tensor, it represents current decode step
        for each sample.

    Returns:
      A Tensor of shape [S, B, N]
    """
    p = self.params
    synced_time_step = (time_step.shape.ndims == 0)
    s, b, _, _ = py_utils.GetShape(key, 4)

    # Transformer_XL relative attention.
    if time_step is None:
      raise ValueError('`time_step` can not be None when using relative '
                       'position encoding in attention.')

    if synced_time_step:
      # [1, s]
      position = tf.expand_dims(time_step - tf.range(s), 0)
    else:
      # [b, s]
      position = (
          tf.expand_dims(time_step, -1) -
          tf.tile(tf.expand_dims(tf.range(s), 0), [b, 1]))
    # [1 or b, s, emb_dim]
    sin_emb = self.pos_emb.FPropWithPosition(theta.pos_emb, position)
    # [1 or b, s, n, h]
    sin_emb = self.pos_proj.FProp(theta.pos_proj, sin_emb)
    if synced_time_step:
      # [s, n, h]
      sin_emb = tf.squeeze(sin_emb, 0)

    logits = self.pos_atten_logits.AttenLogitsXLOneStep(
        query,
        key,
        abs_pos_emb=sin_emb,
        content_bias=theta.u,
        positional_bias=theta.v,
        skip_term_b=p.skip_term_b)

    return logits

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 paddings,
                 segment_mask,
                 per_step_padding,
                 time_step,
                 use_short_seq_opt=False):
    # TODO(jamesqin): support use_short_seq_opt for TransformerXL attention.
    assert not use_short_seq_opt
    return super().ExtendStep(theta, query_vec, cached_states, paddings,
                              segment_mask, per_step_padding, time_step,
                              use_short_seq_opt)


class MultiHeadedAttentionRPE(MultiHeadedAttention):
  """Multiheaded attention with relative positional embedding ...

  See https://arxiv.org/pdf/1803.02155.pdf.

  Notice this is only intended for self attention.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('rel_pos_emb_dim', None,
             'Dimension of relative positional embedding.')
    p.Define('rel_pos_radius', None,
             'Relative distance is clipped to [-radius, radius].')
    p.Define('skip_value_emb', False, 'If skipping value positional embedding.')
    p.Define(
        'use_global_emb', True,
        'If using global relative positional embedding. Only effective if '
        '`rel_pos_emb_tpl` is not None.')
    p.Define('pos_atten_logits_tpl',
             attention_util.PositionalAttenLogits.Params(),
             'Params for the positional attention logits.')
    return p

  def __init__(self, params):
    """Constructs a MultiHeadedAttentionRPE object."""
    super().__init__(params)
    params = self.params

    assert not params.packed_input, 'Packed input not implemented yet.'

    if not params.rel_pos_radius:
      raise ValueError('Invalid rel_pos_radius: %s' % params.rel_pos_radius)

    if params.rel_pos_emb_dim is None:
      rel_pos_emb_dim = params.hidden_dim
    else:
      rel_pos_emb_dim = params.rel_pos_emb_dim

    rel_pos_emb_tpl = layers.RelativePositionalEmbeddingLayer.Params().Set(
        radius=params.rel_pos_radius, dim=rel_pos_emb_dim)
    if rel_pos_emb_dim != params.hidden_dim:
      # Projection layer for relative position encoding
      dim_per_head = params.hidden_dim // params.num_heads
      pos_proj_tpl = params.proj_tpl.Copy().Set(
          input_dim=rel_pos_emb_dim,
          num_heads=params.num_heads,
          dim_per_head=dim_per_head,
          use_bias=False)
    else:
      pos_proj_tpl = None

    self.CreateChild('key_emb', rel_pos_emb_tpl)
    # Add projection layer if rel_pos_emb_dim is different from hidden_dim.
    if pos_proj_tpl is not None:
      self.CreateChild('key_pos_proj', pos_proj_tpl)
    if not params.skip_value_emb:
      self.CreateChild('value_emb', rel_pos_emb_tpl)
      if pos_proj_tpl is not None:
        self.CreateChild('value_pos_proj', pos_proj_tpl)
    self.CreateChild('pos_atten_logits', params.pos_atten_logits_tpl)

  def _CreateChildrenVariables(self):
    with tf.variable_scope(
        self.params.name,
        reuse=tf.AUTO_REUSE if self.params.use_global_emb else False):
      for child in ['key_emb', 'key_pos_proj', 'value_emb', 'value_pos_proj']:
        if child in self.children:
          self.children[child].InstantiateVariables()
    super()._CreateChildrenVariables()

  def _RelativePositionValueEmb(self, theta, key):
    """Gets relative positional value embedding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      key: The attention key, a tensor of shape [batch, seqlen, dim]

    Returns:
      Relative positional embedding, a Tensor of shape
      [tgt_time=seqlen, src_time=seqlen, num_heads, attention_dim]
    """
    emb_layer = self.value_emb
    emb_theta = theta.value_emb

    seqlen = py_utils.GetShape(key)[1]
    src_time_indices = tf.tile(tf.expand_dims(tf.range(seqlen), 0), [seqlen, 1])
    tgt_time_indices = tf.tile(
        tf.expand_dims(tf.range(seqlen), -1), [1, seqlen])

    # [tgt_time=T, src_time=T, num_heads x hidden_dim]
    pos_emb = emb_layer.FProp(emb_theta, src_time_indices - tgt_time_indices)

    params = self.params
    num_heads = self.params.num_heads
    tgt_time, src_time, _ = py_utils.GetShape(pos_emb)

    pos_proj_layer = 'value_pos_proj'
    if hasattr(self, pos_proj_layer):
      return getattr(self, pos_proj_layer).FProp(
          getattr(theta, pos_proj_layer), pos_emb)
    else:
      return tf.reshape(
          pos_emb,
          [tgt_time, src_time, num_heads, params.hidden_dim // num_heads])

  def _AttenLogits(self, theta, query, key):
    # TODO(jamesqin): optimize it.
    b, _, n, h = py_utils.GetShape(key, 4)
    t = py_utils.GetShape(query)[1]

    # This layer only supports self attention.
    key = py_utils.HasShape(key, [b, t, n, h])

    # [1, 2T - 1]
    pos = tf.expand_dims(tf.range(-(t - 1), t), 0)
    # [1, 2T - 1, rel_pos_emb_dim]
    abs_emb = self.key_emb.FProp(theta.key_emb, pos)
    if hasattr(self, 'key_pos_proj'):
      # [1, 2T - 1, N, H]
      abs_emb = self.key_pos_proj.FProp(theta.key_pos_proj, abs_emb)
      # [2T - 1, N, H]
      abs_emb = tf.squeeze(abs_emb, 0)
    else:
      abs_emb = tf.reshape(abs_emb, [2 * t - 1, n, h])

    return self.pos_atten_logits.AttenLogitsRPE(query, key, abs_emb)

  def _AttenLogitsOneStep(self, theta, query, key, time_step):
    """Attention logits for one single target (query) step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, N, H].
      key:      [S, B, N, H] or [S, B, N*H/128, 128].
      time_step: Current time step.

    Returns:
      A Tensor of shape [S, B, N]
    """
    s, _, _, _ = py_utils.GetShape(key, 4)
    _, n, h = py_utils.GetShape(query, 3)

    # Transformer_XL relative attention.
    if time_step is None:
      raise ValueError('`time_step` can not be None when using relative '
                       'position encoding in attention.')
    # Gets positional embedding.
    # [1, S]
    rel_dists = tf.expand_dims(time_step - tf.range(s), 0)
    # [1, S, rel_pos_emb_dim]
    abs_emb = self.key_emb.FPropDefaultTheta(rel_dists)
    if hasattr(self, 'key_pos_proj'):
      # [1, S, N, H]
      abs_emb = self.key_pos_proj.FProp(theta.key_pos_proj, abs_emb)
      # [S, 1, N, H]
      abs_emb = tf.transpose(abs_emb, [1, 0, 2, 3])
    else:
      abs_emb = tf.reshape(abs_emb, [s, 1, n, h])

    return self.pos_atten_logits.AttenLogitsRPEOneStep(query, key, abs_emb)

  def _AttenContext(self, theta, probs, value):
    # TODO(jamesqin): optimize it.
    encoded = tf.einsum('BNij,BjNH->BiNH', probs, value)

    if not self.params.skip_value_emb:
      encoded += tf.einsum('BNij,ijNH->BiNH', probs,
                           self._RelativePositionValueEmb(theta, value))
    return encoded

  def _AttenContextOneStep(self, theta, probs, value, time_step, h):
    s, b, _, _ = py_utils.GetShape(value, 4)
    _, _, n = py_utils.GetShape(probs, 3)

    logits = tf.einsum('SBN,SBNH->BNH', probs, tf.reshape(value, [s, b, n, h]))

    if not self.params.skip_value_emb:
      # [1, S]
      rel_dists = tf.expand_dims(time_step - tf.range(s), 0)
      # [1, S, rel_pos_emb_dim]
      pos_emb = self.value_emb.FProp(theta.value_emb, rel_dists)
      if hasattr(self, 'value_pos_proj'):
        # [1, S, N, H]
        pos_emb = self.value_pos_proj.FProp(theta.value_pos_proj, pos_emb)
        pos_emb = tf.squeeze(pos_emb, 0)
      else:
        pos_emb = tf.reshape(pos_emb, [s, n, h])
      logits += tf.einsum('SBN,SNH->BNH', probs, pos_emb)
    return logits

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 paddings,
                 segment_mask,
                 per_step_padding,
                 time_step,
                 use_short_seq_opt=False):
    # TODO(jamesqin): support use_short_seq_opt.
    assert not use_short_seq_opt
    return super().ExtendStep(theta, query_vec, cached_states, paddings,
                              segment_mask, per_step_padding, time_step,
                              use_short_seq_opt)

  @classmethod
  def FPropMeta(cls, p, *args):
    return NotImplementedError()


class LocalSelfAttention(MultiHeadedAttention):
  """Dot-product self attention using a sliding window.

  We use the following capital letters to denote certain
  tensor parameters.

    B = batch size
    S=T= length of the key/value (source) and query (target)
    D = model dimension
    N = number of attention heads
    H = dimensions of each attention head
    W = block size
    L = left context size, including left L-1 positions and self
    R = right context size
    F = L + R = context size of one position.
    C = L + R + W - 1 = context size of a block of W positions.
    U = ceiling(T/W).

  For each position, its attention range includes from the left
  L-1 tokens before it (up to the beginning of the sequence),
  the self, and the right R tokens after it (up to the end of the
  sequence). This is not affected by the block size.

  Causality is enabled when right context size R=0.

  The key difference to base class is on calculating logits:
    Base class:
      1)  Compute the full S x T attention.
      2)  Apply a S x T mask to enforce local attention window.
    This implementation:
      1)  Compute a W x C attention for each of the U blocks. Where the i-th
      block has query[W*i:W*(i+1)] and key[W*(i-1)-L-1:W*(i+1)+R].
      2)  Apply a W x C mask for each block.

  Effectively, we reduce both time and space complexities for computing the
  sliding window attention from O(S * T) to O(S * C). In practice we observe
  reduced HBM usage on TPU but no speed gains.

  Note: Cross attention is not supported. As a result in speech models this
  class can only be used for encoder.

  TODO(weihan): add masking based local attention to the base class.

  """

  @classmethod
  def Params(cls):
    """Params for LocalSelfAttention."""
    p = super().Params()
    p.Define(
        'block_size', None, 'Size of a processing block, if unset, default to '
        'max(1, left_context-1).')
    p.Define(
        'left_context', None, 'Number of left positions to attend '
        '(including current position).')
    p.Define('right_context', 0, 'Number of right positions to attend.')
    p.Define(
        'force_consistent_probs_shape', False,
        'Bool, whether to force the attention_probs tensor returned from '
        'FProp() to have shape [B N T S] to be consistent with the MHA '
        'parent class. Default returns a custom rank-5 tensor with '
        'shape [B, N, U, W, C].')

    # The following are for streaming inference only.
    p.Define(
        'inference_step_max_length', None, 'Max inference step length '
        '(query_vec length). Used for efficient sunn inference on tpu. In case '
        'inference seq length is not static, set to None or negative, and a '
        'less optimized algorithm is used.')
    p.Define(
        'use_3d_recurrent_state', False,
        'If True, recurrent state for streaming inference is [B, T, N*H] '
        'instead of [B, T, N, H]. This is for performance optimization '
        'and does not change math. Only effective if inference_step_max_length '
        'is not None and > 0.')
    p.Define(
        'minimize_state_size', False,
        'If True, the recurrent state is a history of the layer inputs instead '
        'of a history of the keys/values. Only supported when '
        'right_context==0.')
    return p

  def __init__(self, params):
    """Constructs a LocalSelfAttention object."""
    super().__init__(params)

    p = self.params
    assert p.left_context >= 1, 'Left context should be at least one.'
    assert not p.packed_input, 'Packed input not implemented yet.'
    if p.block_size is None:
      p.block_size = max(1, p.left_context - 1)
      tf.logging.warning('block_size not set, use default value {}'.format(
          p.block_size))

    assert not p.packed_input, 'Packed input not implemented yet.'

  def _AttenLogits(self, theta, query, key):
    return tf.einsum('BUTNH,BUSNH->BNUTS', query, key)

  def _StreamAttenLogits(self, theta, query_proj, key):
    """Compute the dot products of a set of queries and a set of keys."""
    # [B, Q, N, T]
    return tf.einsum('BQNH,BTNH->BQNT', query_proj, key)

  def AttenProbs(self,
                 theta,
                 query,
                 key,
                 paddings,
                 segment_mask,
                 per_step_padding=None):
    """Compute attention probability.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, T, N, H].
      key:      [B, S=T, N, H].
      paddings: [B, T].
      segment_mask: [B, 1, T, S] not used right now.
      per_step_padding: Not used.

    Returns:
      probs: [B, N, U, W, C]
      probs_sum: [B, N, U, W, 1].
    """
    del per_step_padding
    p = self.params
    key = py_utils.HasRank(key, 4)
    b, t, n, h = py_utils.GetShape(key, 4)
    paddings = py_utils.HasShape(paddings, [b, t])
    query = py_utils.HasShape(query, [b, t, n, h])

    # -> [B, U, C, N, H]
    key_block_context = attention_util.ExtractBlockContext(
        key,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context)
    _, u, c, _, _ = py_utils.GetShape(key_block_context)

    # -> [B, U, W, N, H]
    query_blocks = attention_util.ConvertToBlocks(
        query, block_size=p.block_size)
    _, _, w, _, _ = py_utils.GetShape(query_blocks)

    # -> [B, U, C]
    mask = 1. - paddings
    mask_block_context = attention_util.ExtractBlockContext(
        mask,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context,
        padding_val=0)

    # -> [B, N, U, W, C]
    mask = tf.tile(
        tf.reshape(mask_block_context, [b, 1, u, 1, c]), [1, n, 1, w, 1])

    # Make local causal mask.
    # -> [U, W, C]
    local_causal_mask = attention_util.MakeLocalMask(
        seq_len=t,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context,
        dtype=mask.dtype)
    mask = mask * local_causal_mask
    paddings = 1. - mask

    # -> [B, N, U, W, C]
    logits = self._AttenLogits(theta, query_blocks, key_block_context)

    padded_logits = py_utils.ApplyPadding(
        paddings, logits, GetDtypeMin(logits.dtype), use_select=False)

    if p.enable_scaling_code_motion:
      # Split the softmax into two parts. Do the 1st part here; the 2nd part
      # (scaling) is moved after _AttenContext for better performance.
      probs = padded_logits - tf.stop_gradient(
          tf.reduce_max(padded_logits, -1, True))
      probs = tf.cast(tf.exp(probs), key.dtype)
      probs_sum = tf.reduce_sum(probs, -1, True)
    else:
      probs = tf.cast(
          py_utils.Softmax(padded_logits, extra_logit=p.atten_extra_logit),
          key.dtype)
      probs_sum = None

    return probs, probs_sum

  def _AttenContext(self, theta, probs, value):
    """Computes the local attention context vector.

    Args:
     theta: Layer theta: NestedMap.
     probs: Local-self-MultiHeaded Attention probabilities: [B, N, U, W, C].
     value: Input value vector: [B, S=T, N, H].

    Returns:
     encoded: Attention context vector: [B, T, N, H].
    """
    p = self.params
    # -> [B, U, C, N, H]
    value_block_context = attention_util.ExtractBlockContext(
        value,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context)

    # Compute the attention context vector.
    # -> [B, U, W, N, H]
    encoded = tf.einsum('BNUWC,BUCNH->BUWNH', probs, value_block_context)
    b, u, w, n, h = py_utils.GetShape(encoded)
    encoded = tf.reshape(encoded, [b, u * w, n, h])
    # Remove the extra time padding introduced by converting to blocks.
    # Note: t0 works presently only for self-attention.
    # For cross-atten, needs query[1] which'll be different.
    t0 = py_utils.GetShape(value)[1]
    encoded = encoded[:, :t0, ...]
    return encoded

  def FProp(self,
            theta,
            query_vec,
            key_vec,
            value_vec,
            paddings,
            segment_mask=None,
            per_step_padding=None):
    """Computes the value vector given the current query output.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [B, T, D].
      key_vec:   [B, S, D] with S == T (self-attention).
      value_vec: [B, S, D] with S == T (self-attention).
      paddings:  [B, S] with S == T (self-attention).
      segment_mask: [B, 1, T, S]. A mask only applied if packed_input=True.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, T, T] if
        not None.

    Returns:
      encoded: [B, T, D].
      probs: [B, N, U, W, C].

    Raises:
      ValueError: If value projection is disabled.
    """
    b, t, d = py_utils.GetShape(query_vec, 3)
    # LocalSelfAttention doesn't support cross-attention at the moment.
    # Verify T == S, for query and value vector.
    value_vec = py_utils.HasShape(value_vec, [b, t, d])
    key_vec = py_utils.HasShape(key_vec, [b, t, d])
    paddings = py_utils.HasShape(paddings, [b, t])
    encoded, probs = super().FProp(
        theta,
        query_vec,
        key_vec,
        value_vec,
        paddings,
        segment_mask=segment_mask,
        per_step_padding=per_step_padding)
    p = self.params
    if not p.force_consistent_probs_shape:
      return encoded, probs

    # We turn 'probs' into shape [B, N, T, S] before turning it.
    # probs has shape [B N U W C].
    _, n, u, w, _ = py_utils.GetShape(probs, 5)
    # shape [B N W U C]
    probs = tf.transpose(probs, [0, 1, 3, 2, 4])
    # Maximum length needed to keep track of probs along the T axis.
    m = t + p.left_context - 1 + p.right_context
    # shape [B N W U M+W], where M = (L-1) + T + R
    probs = py_utils.PadOrTrimTo(probs, [b, n, w, u, m + w])
    probs = tf.reshape(probs, [b, n, w, u * (m + w)])
    # Now each row is shifted by W from its previous row. This recovers
    # the true position of the C axis into the now expanded T axis.
    probs = tf.reshape(probs[:, :, :, :u * m], [b, n, w, u, m])
    # Shape [B N U W M]
    probs = tf.transpose(probs, [0, 1, 3, 2, 4])
    # Shape [B N U W T]
    probs = probs[:, :, :, :, p.left_context - 1:m - p.right_context]
    probs = tf.reshape(probs, [b, n, w * u, t])
    # Truncate to shape [B N T T]
    probs = probs[:, :, :t, :]
    return encoded, probs

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 paddings,
                 segment_mask=None,
                 per_step_padding=None,
                 time_step=None,
                 use_short_seq_opt=False):
    """Computes the value vector given the query of the current step.

    This function is used by autoregressive decoding, as opposed to
    StreamStep which is for single step self attention.

    Note: When the context window size is much smaller than target sequence
    length, to make it run more efficent, T below can be just the window size.
    Then, time_step should be the relative decode step and not bigger than T.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:        [B, 1, D].
      cached_states:   A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. key   - [T, B,
        N, H]. value - [T, B, N, H].
      paddings:         [B, T], or None if there is no padding.
      segment_mask:     [B, 1, T, S] or None. Not used right now.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, 1, T] if
        not None. Not used right now.
      time_step: A scalar, the current decode step, 0-based.
      use_short_seq_opt: A bool, whether using short sequence optimization. Not
        supported right now.

    Returns:
      encoded:           [B, 1, D].
      updated_key_vec:   [T, B, N, H].
      updated_value_vec: [T, B, N, H].

    Raises:
      ValueError: If right_context is non-zero.
      NotImplementedError: If use_short_seq_opt is true.
    """
    p = self.params
    if p.right_context != 0:
      raise ValueError(
          'Right context must be zero for autoregressive decoding.')
    if use_short_seq_opt:
      raise NotImplementedError('use_short_seq_opt is not supported yet.')

    # Make local causal paddings, which have shape [B, T].
    t, b, _, _ = py_utils.GetShape(cached_states.key, 4)
    if paddings is None:
      paddings = tf.zeros([b, t], dtype=query_vec.dtype)
    position_diff = tf.tile(tf.range(t)[tf.newaxis, :], [b, 1]) - time_step
    valid_atten = tf.math.logical_and(position_diff > -p.left_context,
                                      position_diff <= 0)
    local_causal_padding = tf.cast(
        tf.math.logical_not(valid_atten), dtype=query_vec.dtype)
    paddings += local_causal_padding

    return super().ExtendStep(theta, query_vec, cached_states, paddings,
                              segment_mask, per_step_padding, time_step,
                              use_short_seq_opt)

  def zero_state(self, batch_size):
    """Returns the initial state given the batch size."""
    p = self.params
    if (p.inference_step_max_length is not None and
        p.inference_step_max_length > 0 and not p.right_context):
      if p.minimize_state_size:
        return self._zero_state_static_length_inputs(batch_size)
      else:
        return self._zero_state_static_length_key_value(batch_size)
    else:
      return self._zero_state_dynamic_length(batch_size)

  def _zero_state_static_length_inputs(self, batch_size):
    """Returns the initial state given the batch size.

    Args:
      batch_size: the batch size.

    Returns:
      A `.NestedMap` object containing

      - inputs: [B, p.inference_step_max_length + p.left_context - 1, D]
      - masks: [B, p.inference_step_max_length + p.left_context-1]. A tf.bool
        Tensor where Falses are masked out positions.
      - circular_tail: [B, 1], currently only exists if
        use_3d_recurrent_state is True, the tail pointer to key, value and
        paddings circular buffers.
        Value range is [0, p.inference_step_max_length + p.left_context - 1).
    """
    p = self.params
    assert p.enable_value_proj, 'Value projection must be enabled.'
    assert p.right_context == 0, 'StreamStep does not support look-ahead'

    context_len = p.inference_step_max_length + p.left_context - 1
    inputs = tf.zeros([batch_size, context_len, p.input_dim],
                      py_utils.FPropDtype(p))
    # At the beginning, all positions are masked out.
    masks = tf.zeros([batch_size, context_len], tf.bool)
    state0 = py_utils.NestedMap(inputs=inputs, masks=masks)
    if p.use_3d_recurrent_state:
      state0.circular_tail = tf.zeros([batch_size, 1], tf.int32)
    return state0

  def _zero_state_static_length_key_value(self, batch_size):
    """Returns the initial state given the batch size.

    Args:
      batch_size: the batch size.

    Returns:
      A `.NestedMap` object containing

      - key: [B, p.inference_step_max_length + p.left_context - 1, N, H] or
        [..., N * H] if p.use_3d_recurrent_state.
      - value: [B, p.inference_step_max_length + p.left_context - 1, N, H] or
        [..., N * H] if p.use_3d_recurrent_state.
      - masks: [B, p.inference_step_max_length + p.left_context-1]. A tf.bool
        Tensor where Falses are masked out positions.
      - circular_tail: [B, 1], currently only effective if
        use_3d_recurrent_state is True, the tail pointer to key, value and
        paddings circular buffers.
        Value range is [0, p.inference_step_max_length + p.left_context - 1).
    """
    p = self.params
    assert p.enable_value_proj, 'Value projection must be enabled.'
    assert p.right_context == 0, 'StreamStep does not support look-ahead'

    context_len = p.inference_step_max_length + p.left_context - 1
    if not p.use_3d_recurrent_state:
      key_state = tf.zeros(
          [batch_size, context_len, p.num_heads, p.hidden_dim // p.num_heads],
          py_utils.FPropDtype(p))
    else:
      key_state = tf.zeros([batch_size, context_len, p.hidden_dim],
                           py_utils.FPropDtype(p))
    value_state = tf.zeros_like(key_state, py_utils.FPropDtype(p))
    # At the beginning, all positions are masked out.
    masks = tf.zeros([batch_size, context_len], tf.bool)
    state0 = py_utils.NestedMap(key=key_state, value=value_state, masks=masks)
    if p.use_3d_recurrent_state:
      state0.circular_tail = tf.zeros([batch_size, 1], tf.int32)
    return state0

  def _zero_state_dynamic_length(self, batch_size):
    """Returns the initial state given the batch size.

    Args:
      batch_size: the batch size.

    Returns:
      A `.NestedMap` object containing

      context_len = p.left_context - 1 + p.right_context
      - key:   [B, context_len, N, H].
      - value: [B, context_len, N, H].
      - masks: [B, context_len]. A Tensor where Falses are masked out positions.
      - query: (only if p.right_context > 0) [B, p.right_context, N, H].
      - out_masks : (only if p.right_context> 0): [B, p.right_context].
    """
    p = self.params
    assert p.enable_value_proj, 'Value projection must be enabled.'

    dtype = py_utils.FPropDtype(p)
    context_len = p.left_context - 1 + p.right_context
    per_head_dim = p.hidden_dim // p.num_heads

    key_state = tf.zeros([batch_size, context_len, p.num_heads, per_head_dim],
                         dtype)
    value_state = tf.zeros_like(key_state, dtype)
    # At the beginning, all positions are masked out.
    masks = tf.zeros([batch_size, context_len], tf.bool)
    state0 = py_utils.NestedMap(key=key_state, value=value_state, masks=masks)
    if p.right_context > 0:
      state0.query = tf.zeros(
          [batch_size, p.right_context, p.num_heads, per_head_dim], dtype)
      state0.out_masks = tf.zeros([batch_size, p.right_context], tf.bool)
      # This is used only if the caller of the layer uses skip_connection in
      # the layer's client code.
      state0.skip_conn_input = tf.zeros(
          [batch_size, p.right_context, p.hidden_dim], dtype)
    return state0

  def IsInferenceStepStatic(self):
    p = self.params
    return p.inference_step_max_length is not None and p.inference_step_max_length > 0

  def StreamStep(self, theta, inputs, paddings, state0):
    """Computes the value vector given the query of the current step.

    This differs from ExtendStep() which requires key/value seq lengths being
    known in advance.

    Args:
      theta: A NestedMap of layer params.
      inputs: An input vector of shape [B, Q, D].
      paddings: A 0/1 valued tensor of shape [B, Q].
      state0: A NestedMap of the same structure as returned by zero_state().

    Returns:
      output: Output of the given query vector with shape [B, Q, D].
      padding: the same as input paddings.
      state1: Updated state of the same structure as state0.
    """
    p = self.params
    assert p.enable_value_proj, 'Value projection must be enabled.'

    with tf.name_scope(f'{p.name}/StreamStep'):
      if self.IsInferenceStepStatic():
        assert p.right_context == 0, (
            'StreamStep() does not yet support look ahead with '
            'inference_step_max_length set.')
        return self._StreamStepStaticLength(theta, inputs, paddings, state0)
      else:
        return self._StreamStepDynamicLength(theta, inputs, paddings, state0)

  def StreamStepAddSkipConnection(self, input_to_add, output, state0, state1):
    p = self.params
    if not p.right_context:
      return input_to_add + output, state1

    seqlen = py_utils.GetShape(output)[1]
    output = py_utils.HasShape(output, py_utils.GetShape(input_to_add))
    concat_input_to_add = tf.concat([state0.skip_conn_input, input_to_add],
                                    axis=1)
    final_output = output + concat_input_to_add[:, :seqlen]
    state1.skip_conn_input = concat_input_to_add[:, seqlen:]
    return final_output, state1

  def _StreamStepDimensions(self, inputs):
    """Returns dimensions commonly used in StreamStep methods.

    Args:
      inputs: The query_vec parameter of StreamStep.

    Returns:
      A NestedMap containing n=num_heads, h=head_dimension,
      s=state_length, b=batch_size, q=num_queries_in_query_vec.
    """
    p = self.params
    n, h = p.num_heads, p.hidden_dim // p.num_heads
    s = p.inference_step_max_length + p.left_context - 1
    b, q = py_utils.GetShape(inputs, 2)
    return py_utils.NestedMap(n=n, h=h, s=s, b=b, q=q)

  def _StreamStepStaticComputeKeyValueMinimal(self, theta, indices, inputs,
                                              state0):
    """Computes key/value tensors in minimize_state_size mode.

    Args:
      theta: The theta NestedMap for this layer.
      indices: Locations to store new recurrent state in the circular buffer
        when in 3d mode.
      inputs: [B, Q, D]: The inputs for this step, note that Q>=1.
      state0: The recurrent state.

    Returns:
      key: [B, S, N, H]: Queries projected into key space.
      value: [B, S, N, H]: Queries projected into value space.
      state1: Updated recurrent state.
    """
    p = self.params
    dims = self._StreamStepDimensions(inputs)
    state0.query = py_utils.HasShape(state0.inputs,
                                     [dims.b, dims.s, p.hidden_dim])

    with tf.name_scope('next_inputs'):
      # [B, S, Q]
      if p.use_3d_recurrent_state:
        new_inputs = tf.tensor_scatter_nd_update(state0.inputs, indices, inputs)
      else:
        new_inputs = tf.concat([state0.inputs, inputs], axis=1)[:, -dims.s:, :]

    # [B, S, N, H]: Project input vectors into key space.
    key = self.key.FProp(theta.key, new_inputs)

    # [B, S, N, H]: Project input vectors into value space.
    value = self.value.FProp(theta.value, new_inputs)

    state1 = py_utils.NestedMap(inputs=new_inputs)
    return key, value, state1

  def _StreamStepStaticComputeKeyValue3d(self, theta, indices, inputs, state0):
    """Computes key/value tensors in use_3d_recurrent_state mode.

    This mode treats state like a circular buffer, and uses scatter_nd_update
    to update that buffer. This in-place update may be cheaper than using
    tf.concat.

    (Don't use this method when in minimize_state_size mode, you want the
    _StreamStepStaticComputeKeyValueMinimal even if you're using
    use_3d_recurrent_state mode)

    Args:
      theta: The theta NestedMap for this layer.
      indices: Locations to store new recurrent state in the circular buffer
        when in 3d mode.
      inputs: [B, Q, D]: The inputs for this step, note that Q>=1.
      state0: The recurrent state.

    Returns:
      key: [B, S, N, H]: Queries projected into key space.
      value: [B, S, N, H]: Queries projected into value space.
      state1: Updated recurrent state.
    """
    dims = self._StreamStepDimensions(inputs)

    state0.key = py_utils.HasShape(state0.key,
                                   [dims.b, dims.s, dims.n * dims.h])
    state0.value = py_utils.HasShape(state0.value,
                                     py_utils.GetShape(state0.key))

    def get_next_state(recur_state, inputs):  # pylint:disable=invalid-name
      next_state = tf.tensor_scatter_nd_update(recur_state, indices, inputs)
      # [B, S, N, H]
      outputs = tf.reshape(next_state, [dims.b, dims.s, dims.n, dims.h])
      return outputs, next_state

    # [B, Q, N * H]
    incr_key = tf.einsum(
        'DH,BTD->BTH',
        tf.reshape(theta.key.w, [self.key.params.input_dim, dims.n * dims.h]),
        inputs) + tf.reshape(theta.key.b, [-1])
    # [B, Q, N * H]
    incr_value = tf.einsum(
        'DH,BTD->BTH',
        tf.reshape(theta.value.w,
                   [self.value.params.input_dim, dims.n * dims.h]),
        inputs) + tf.reshape(theta.value.b, [-1])

    # [B, S, N, H], [B, S, N * H]
    key, next_key = get_next_state(state0.key, incr_key)
    # [B, S, N, H], [B, S, N * H]
    value, next_value = get_next_state(state0.value, incr_value)

    state1 = py_utils.NestedMap(key=next_key, value=next_value)
    return key, value, state1

  def _StreamStepStaticComputeKeyValueClassic(self, theta, inputs, state0):
    """Computes key/value tensors in classic mode.

    In this mode we store projected key/value tensors from previous
    steps in the state, and create new key/value tensors using tf.concat.

    Use this mode if memory transfer is cheap and computation is expensive.
    (Although 3D mode may also be interesting in that case)

    Args:
      theta: The theta NestedMap for this layer.
      inputs: [B, Q, D]: The query for this step, note that Q>=1.
      state0: The recurrent state.

    Returns:
      key: [B, S, N, H]: Queries projected into key space.
      value: [B, S, N, H]: Queries projected into value space.
      state1: Updated recurrent state.
    """
    dims = self._StreamStepDimensions(inputs)
    state0.key = py_utils.HasShape(state0.key, [dims.b, dims.s, dims.n, dims.h])
    state0.value = py_utils.HasShape(state0.value,
                                     py_utils.GetShape(state0.key))

    # [B, Q, N, H]
    incr_key = self.key.FProp(theta.key, inputs)
    # [B, Q, N, H]
    incr_value = self.value.FProp(theta.value, inputs)

    # [B, S, N, H]
    key = tf.concat([state0.key, incr_key], axis=1)[:, -dims.s:, :, :]
    # [B, S, N, H]
    value = tf.concat([state0.value, incr_value], axis=1)[:, -dims.s:, :, :]

    state1 = py_utils.NestedMap(key=key, value=value)
    return key, value, state1

  def _StreamStepStaticComputeKeyValue(self, theta, inputs, paddings, state0):
    """Computes key/value tensors.

    This method combines the new input for this step with previous
    state to generate key/value tensors for attention computation.

    Args:
      theta: The theta NestedMap for this layer.
      inputs: [B, Q, D]: The input for this step, note that Q>=1.
      paddings: A 0/1 valued tensor of shape [B, Q].
      state0: The recurrent state.

    Returns:
      key: [B, S, N, H]: Queries projected into key space.
      value: [B, S, N, H]: Queries projected into value space.
      state1: Updated recurrent state.
    """
    p = self.params
    dims = self._StreamStepDimensions(inputs)

    indices = None
    if p.use_3d_recurrent_state:
      # The following computes locations to update in the circular buffer.
      # [b, 1]
      rows = tf.expand_dims(tf.range(dims.b), -1)
      # [b, q]
      rows = tf.tile(rows, [1, dims.q])
      # [1, q]
      cols = tf.expand_dims(tf.range(dims.q), 0)
      # [b, q]
      cols = tf.tile(cols, [dims.b, 1])
      # [b, q]
      cols = tf.math.floormod(cols + state0.circular_tail, dims.s)
      # [b, q, 2]
      indices = tf.stack([rows, cols], axis=-1)

    if p.minimize_state_size:
      key, value, state1 = self._StreamStepStaticComputeKeyValueMinimal(
          theta, indices, inputs, state0)
    elif p.use_3d_recurrent_state:
      key, value, state1 = self._StreamStepStaticComputeKeyValue3d(
          theta, indices, inputs, state0)
    else:
      key, value, state1 = self._StreamStepStaticComputeKeyValueClassic(
          theta, inputs, state0)

    # paddings
    # [B, S]. 1s are masked positions.
    input_masks = tf.logical_not(tf.cast(paddings, tf.bool))
    if p.use_3d_recurrent_state:
      new_masks = tf.tensor_scatter_nd_update(state0.masks, indices,
                                              input_masks)
    else:
      new_masks = tf.concat([state0.masks, input_masks], axis=1)[:, -dims.s:]

    # [B, 1]
    if p.use_3d_recurrent_state:
      state1.circular_tail = tf.math.floormod(state0.circular_tail + dims.q,
                                              dims.s)
    state1.masks = new_masks
    return key, value, state1

  def _StreamStepStaticLength(self, theta, query_vec, paddings, state0):
    """query_vec length is staticly known."""
    p = self.params
    dims = self._StreamStepDimensions(query_vec)
    h, s, b, q = dims.h, dims.s, dims.b, dims.q
    assert query_vec.shape[1] is not None, 'query_vec.shape[1] must be static.'
    assert q <= p.inference_step_max_length, (
        f'q: {q} should be less than p.inference_step_max_length: '
        f'{p.inference_step_max_length}')

    query_vec = py_utils.HasShape(query_vec, [-1, -1, p.input_dim])
    paddings = py_utils.HasShape(paddings, [b, q])

    with tf.name_scope('static_length'):
      # query projection.
      # [B, Q, N, H]
      query_proj = self.query.FProp(theta.query, query_vec)
      if p.enable_query_scale:
        if p.enable_per_dim_scale:
          query_proj = self.per_dim_scale.FProp(theta.per_dim_scale, query_proj)
        else:
          query_proj *= h**-0.5

      key, value, state1 = self._StreamStepStaticComputeKeyValue(
          theta, query_vec, paddings, state0)

      # [B, Q, N, T]
      logits = self._StreamAttenLogits(theta, query_proj, key)

      with tf.name_scope('compute_padding'):
        # Generate local atten mask.
        # [Q, 1]
        rows = tf.expand_dims(tf.range(q), -1)
        # [1, S]
        cols = tf.expand_dims(tf.range(s), 0)
        # 1s are masked positions.
        # [Q, S]
        distance = tf.math.floormod(cols - rows, s)
        if p.use_3d_recurrent_state:
          # [B, 1]
          head = tf.math.floormod(state0.circular_tail - (p.left_context - 1),
                                  s)
          # [B, Q, S]
          shifted_distance = tf.math.floormod(
              tf.expand_dims(distance, 0) - tf.expand_dims(head, -1), s)
        else:
          # [Q, S]
          shifted_distance = distance - (p.inference_step_max_length - q)
        # [B, Q, S] or [Q, S]
        local_atten_per_step_masks = tf.logical_and(
            shifted_distance <= p.left_context - 1, shifted_distance >= 0)
        # [1, Q, S] or [B, Q, S]
        if py_utils.GetRank(local_atten_per_step_masks) < 3:
          local_atten_per_step_masks = tf.expand_dims(
              local_atten_per_step_masks, 0)
        # [B, 1, S]
        expanded_state_masks = tf.expand_dims(state1.masks, 1)

        # [B, Q, S]
        final_masks = tf.logical_and(expanded_state_masks,
                                     local_atten_per_step_masks)
        # [B, Q, 1, S]
        final_masks = tf.expand_dims(final_masks, axis=2)

      # [B, Q, N, S]
      logits = py_utils.ApplyPadding(
          tf.logical_not(final_masks),
          logits,
          GetDtypeMin(logits.dtype),
          use_select=False)
      # [B, Q, N, S]
      posteriors = py_utils.Softmax(
          logits, axis=-1, extra_logit=p.atten_extra_logit)
      # [B, Q, N, H]
      output = tf.einsum('BQNS,BSNH->BQNH', posteriors, value)

      # Post projection.
      # [B, Q, D]
      output = self.post.FProp(theta.post, output)
      return output, paddings, state1

  def _StreamStepDynamicLength(self, theta, query_vec, paddings, state0):
    """query_vec length is dynamic."""
    p = self.params
    # Sanity checks.
    b, q = py_utils.GetShape(query_vec, 2)
    h = p.hidden_dim // p.num_heads
    context_len = p.left_context - 1 + p.right_context

    query_vec = py_utils.HasShape(query_vec, [-1, -1, p.input_dim])
    paddings = py_utils.HasShape(paddings, [b, q])

    with tf.name_scope('dynamic_length'):
      # query projection.
      # [B, Q, N, H]
      query_proj = self.query.FProp(theta.query, query_vec)
      if p.enable_query_scale:
        if p.enable_per_dim_scale:
          query_proj = self.per_dim_scale.FProp(theta.per_dim_scale, query_proj)
        else:
          query_proj *= h**-0.5

      input_masks = tf.logical_not(tf.cast(paddings, tf.bool))
      if p.right_context == 0:
        # [B, Q, N, H]
        query = query_proj
        out_masks = input_masks
        out_paddings = paddings
      else:
        # [B, R + Q, N, H]
        concat_query = tf.concat([state0.query, query_proj], axis=1)
        # [B, Q, N, H]
        query = concat_query[:, :q]
        concat_out_masks = tf.concat([state0.out_masks, input_masks], axis=1)
        out_masks = concat_out_masks[:, :q]
        out_paddings = tf.cast(tf.logical_not(out_masks), paddings.dtype)

      # key, value, mask.
      # [B, T, N, H].
      key = tf.concat(
          [state0.key, self.key.FProp(theta.key, query_vec)],
          axis=1,
          name='concat_key')
      # [B, T, N, H]
      value = tf.concat(
          [state0.value, self.value.FProp(theta.value, query_vec)],
          axis=1,
          name='concat_value')
      # [B, T]
      state_masks = tf.concat([state0.masks, input_masks],
                              axis=1,
                              name='concat_masks')

      # [B, Q, N, T]
      logits = self._StreamAttenLogits(theta, query, key)

      with tf.name_scope('compute_padding'):
        # Generate local atten mask.
        # [Q, 1]
        # Assuming the current query index starts from 0
        query_indices = tf.expand_dims(
            tf.range(-p.right_context, -p.right_context + q), -1)
        # [1, T]
        target_indices = tf.expand_dims(tf.range(-context_len, q), 0)
        # 1s are masked positions.
        # [Q, T]
        distance = query_indices - target_indices
        local_atten_per_step_masks = tf.logical_and(
            distance <= p.left_context - 1, distance >= -p.right_context)
        # [1, Q, T]
        local_atten_per_step_masks = tf.expand_dims(local_atten_per_step_masks,
                                                    0)
        # [B, 1, T]
        expanded_state_masks = tf.expand_dims(state_masks, 1)

        # [B, Q, T]
        final_masks = tf.logical_and(expanded_state_masks,
                                     local_atten_per_step_masks)
        # [B, Q, 1, T]
        final_masks = tf.expand_dims(final_masks, axis=2)

      # [B, Q, N, T]
      logits = py_utils.ApplyPadding(
          tf.logical_not(final_masks), logits, GetDtypeMin(logits.dtype))

      # [B, Q, N, T]
      posteriors = py_utils.Softmax(
          logits, axis=-1, extra_logit=p.atten_extra_logit)
      # [B, Q, N, H]
      output = tf.einsum('BQNT,BTNH->BQNH', posteriors, value)

      # Post projection.
      # [B, Q, D]
      output = self.post.FProp(theta.post, output)

      state1 = py_utils.NestedMap(
          key=key[:, q:, :, :],
          value=value[:, q:, :, :],
          masks=state_masks[:, q:])
      if p.right_context > 0:
        state1.query = concat_query[:, q:]
        state1.out_masks = concat_out_masks[:, q:]
      return output, out_paddings, state1

  @classmethod
  def FPropMeta(cls, p, *args):
    raise NotImplementedError()


class LocalSelfAttentionXL(LocalSelfAttention):
  """Local causal version of transformer-xl self attention."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('rel_pos_emb_dim', None,
             'Dimension of relative positional embedding.')
    p.Define('skip_term_b', False,
             'If True, skip term_b in the paper section 3.3.')
    return p

  def __init__(self, params):
    """Constructs a LocalSelfAttentionXL object."""
    super().__init__(params)
    params = self.params
    if params.rel_pos_emb_dim is None or params.rel_pos_emb_dim <= 0:
      raise ValueError('Invalid rel_pos_emb_dim: %s' % params.rel_pos_emb_dim)

    if params.use_3d_recurrent_state:
      # Rel pos emb relies on the shape of query and key.
      raise ValueError('Rel pos emb does not support 3d recurrent state.')

    emb_params = layers.PositionalEmbeddingLayer.Params().Set(
        embedding_dim=params.rel_pos_emb_dim)
    self.CreateChild('pos_emb', emb_params)

    # Projection layer for relative position encoding
    dim_per_head = params.hidden_dim // params.num_heads
    pos_proj_tpl = params.proj_tpl.Copy().Set(
        input_dim=params.rel_pos_emb_dim,
        num_heads=params.num_heads,
        dim_per_head=dim_per_head,
        use_bias=False)
    self.CreateChild('pos_proj', pos_proj_tpl)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    params = self.params

    dim_per_head = params.hidden_dim // params.num_heads
    u_pc = py_utils.WeightParams(
        shape=[params.num_heads, dim_per_head],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=params.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    v_pc = py_utils.WeightParams(
        shape=[params.num_heads, dim_per_head],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=params.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    self.CreateVariable('u', u_pc)
    self.CreateVariable('v', v_pc)

  def _AttenLogits(self, theta, query, key):
    b, u, w, _, _ = py_utils.GetShape(query)
    _, _, c, _, _ = py_utils.GetShape(key)
    n = self.params.num_heads
    l = self.params.left_context
    r = self.params.right_context
    f = l + r
    # term a and c
    term_ac = tf.einsum('BUWNH,BUCNH->BNUWC', query + theta.u, key)

    # term b and d
    # [1, F]
    pos = tf.expand_dims(tf.range(l - 1, -r - 1, -1), 0)
    sin_emb = self.pos_emb.FPropWithPosition(theta.pos_emb, pos)
    # [1, F, N, H]
    sin_emb = self.pos_proj.FProp(theta.pos_proj, sin_emb)
    # [F, N, H]
    sin_emb = tf.squeeze(sin_emb, 0)

    p = self.params
    if not p.skip_term_b:
      # [B, N, U, W, F]
      term_bd = tf.einsum('BUWNH,FNH->BNUWF', query + theta.v, sin_emb)

      # Perform relative shift in order to get [B, N, U, W, C]
      # Pads the input to [B, N, U, C, C+1]
      term_bd = tf.pad(term_bd,
                       ((0, 0), (0, 0), (0, 0), (0, c - w), (0, c + 1 - f)))

      # Reshapes to [B, N, U, C+1, C]. Note the output last dim is 1-smaller
      # than the input, which "pushses" one element off to the next row for each
      # row. The accumulated effect is row_i is right-shifted i steps (i>=0).
      term_bd = tf.reshape(term_bd, [b, n, u, c + 1, c])

      # Keeps useful slices. [B, N, U, W, C]
      term_bd = tf.slice(term_bd, [0, 0, 0, 0, 0], [-1, -1, -1, w, -1])
    else:
      # [N, F]
      term_d = tf.einsum('NH,FNH->NF', theta.v, sin_emb)
      # [N, W, F]
      term_d = tf.tile(tf.expand_dims(term_d, 1), [1, w, 1])
      # [N, C, C+1]
      term_d = tf.pad(term_d, ((0, 0), (0, c - w), (0, c + 1 - f)))
      # [N, C+1, C]
      term_d = tf.reshape(term_d, [n, c + 1, c])
      # Keeps useful slices. [N, W, C]
      term_d = tf.slice(term_d, [0, 0, 0], [-1, w, -1])
      term_bd = tf.reshape(term_d, [1, n, 1, w, c])
    return term_ac + term_bd

  def _StreamAttenLogits(self, theta, query, key):
    # BQNH -> BUQNH
    query = tf.expand_dims(query, 1)
    # BTNH -> BUTNH
    key = tf.expand_dims(key, 1)
    logits = self._AttenLogits(theta, query, key)
    # BNUQT -> BNQT -> BQNT
    return tf.transpose(tf.squeeze(logits, 2), [0, 2, 1, 3])

  def _AttenLogitsOneStep(self, theta, query, key, time_step):
    """Attention logits for one single target (query) step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, N, H].
      key:      [S, B, N, H] or [S, B, N*H/128, 128].
      time_step: Current time step.

    Returns:
      A Tensor of shape [S, B, N]
    """
    p = self.params
    s, b, _, _ = py_utils.GetShape(key, 4)
    n = p.num_heads
    h = p.hidden_dim // n

    # Transformer_XL relative attention.
    if time_step is None:
      raise ValueError('`time_step` can not be None when using relative '
                       'position encoding in attention.')
    # term a and c.
    logits = tf.einsum('BNH,SBNH->SBN', query + theta.u,
                       tf.reshape(key, [s, b, n, h]))
    position = tf.expand_dims(time_step - tf.range(s), 0)
    # [1, s, emb_dim]
    sin_emb = self.pos_emb.FPropWithPosition(theta.pos_emb, position)
    sin_emb = self.pos_proj.FProp(theta.pos_proj, sin_emb)
    # [s, n, h]
    sin_emb = tf.squeeze(sin_emb, 0)

    # term b an d.
    if not p.skip_term_b:
      logits += tf.einsum('BNH,SNH->SBN', query + theta.v, sin_emb)
    else:
      logits += tf.expand_dims(tf.einsum('NH,SNH->SN', theta.v, sin_emb), 1)
    return logits

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 paddings,
                 segment_mask=None,
                 per_step_padding=None,
                 time_step=None,
                 use_short_seq_opt=False):
    raise NotImplementedError

  def StreamStep(self, theta, query_vec, paddings, state0):
    """Computes the value vector given the query of the current step.

    Note: Rel pos emb relies on the shape of key. It expects the seq length
    of key is 'q.length + left - 1'. See '_AttenLogits()'.
    'p.inference_step_max_length' must be same to 'q.length', when
    'p.inference_step_max_length > 0'.

    Args:
      theta: A NestedMap of layer params.
      query_vec: A query vector of shape [B, Q, D].
      paddings: A 0/1 valued tensor of shape [B, Q].
      state0: A NestedMap of the same structure as returned by zero_state().

    Returns:
      output: Output of the given query vector with shape [B, Q, D].
      padding: the same as input paddings.
      state1: Updated state of the same structure as state0.
    """
    if self.IsInferenceStepStatic():
      p = self.params
      _, q = py_utils.GetShape(query_vec, 2)
      # Rel pos emb expects the seq length of key is `q.length + left - 1`.
      assert q == p.inference_step_max_length, (
          'inference_step_max_length must be same to the seq length of query.')
    return super().StreamStep(theta, query_vec, paddings, state0)


class RoutingAttention(MultiHeadedAttention):
  """"Implements a sparse attention based on k-means clustering.

  This is used in the routing transformer https://arxiv.org/pdf/2003.05997.

  This version of multi-headed attention differs from the full attention
  in that it uses k-means clusterting to cluster the queries and keys first,
  and each query only attend to a subset of keys that are close to the centroid
  closest to that query. As Euclidean distance is used to determine closeness,
  we layer normalize queries and keys first so that closeness lead to a larger
  dot product.

  TODO(zhouwk) This class is missing the following features:
    * propagate clustering loss;
    * supporting packed inputs;
    * support attention dropout;
    * support relative position encoding;

  We use the following capital letters to denote shape parameters:
    B = batch size
    S = length of the source sequence
    T = length of the target sequence
    N = number of attention heads
    H = dimensions of each attention head
    D = model dimension

    K = number of clusters
    W = attention window
  """

  @classmethod
  def Params(cls):
    """Params."""
    p = super().Params()
    p.Define(
        'num_clusters', 0, 'Number of clusters, typically around the square'
        ' root of the sequence length.')
    p.Define('attention_window', 0, 'The number of keys each query attends to.')
    p.Define('clustering', attention_util.KMeansClusteringForAtten.Params(),
             'The params for a clustering layer.')
    p.Define(
        'causal_masking', False,
        'Whether causal masking is enabled. When set, a query at position idx '
        'is only allowed to attend to keys/values at positions <= idx.')
    p.Define(
        'fast_path', True,
        'Whether to use a more efficient implementation. The fast path is '
        'significantly faster by grouping queries when determining which '
        'values to attend to (which might leave out some queries or duplicate '
        'others); fast_path=False computes this per each query.')
    p.Define(
        'query_group_size_factor', 1.2,
        'Only used when p.fast_path=True. When grouping queries, we make the '
        'group size larger by this multiplier to not leave out any queries due '
        'to potential cluster imbalance.')
    return p

  def __init__(self, params):
    """Constructs an instance of RoutingAttention."""
    super().__init__(params)
    p = self.params
    assert p.num_clusters
    assert p.attention_window
    assert not p.packed_input

    clustering_p = p.clustering
    clustering_p.num_clusters = p.num_clusters
    clustering_p.num_heads = p.num_heads
    clustering_p.dim_per_head = p.dim_per_head or p.hidden_dim // p.num_heads
    # We normalize manually prior so that we can reuse the same normalized
    # query/key to compute attention probs later.
    clustering_p.apply_layer_norm = False
    self.CreateChild('clustering', clustering_p)

  def _DotAtten(self,
                theta,
                query,
                key,
                value,
                paddings,
                segment_mask=None,
                per_step_padding=None,
                query_paddings=None):
    """Computes the attention.

    Each query selects 'p.attention_window' number of keys to attend to. First
    we find the closest centroid to that query, and we only allow that query to
    attend to the 'p.attention_window' closest keys to that centroid.

    In order to use K-means, this implementation applies layer normalization
    to both the queries and the keys, and uses the normalized results to compute
    attention weights.

    When 'p.attention_window' is the source length, this should evalue to the
    full attention (using layer normalized queries and keys).

    The caller should pass in the paddings for both 'key' and 'query' because
    during training, when we update the clustering we need to know the paddings
    for both. (For the inference path only 'key_paddings' is useful.)

    Args:
      theta: A `.NestedMap` of the values of this layer's weights.
      query: [B, T, N, H].
      key:   [B, S, N, H].
      value: [B, S, N, H].
      paddings:   [B, S], paddings for key.
      segment_mask: must be None.
      per_step_padding: must be None. Please use p.causal_masking.
      query_paddings: [B, T], or None.

    Returns:
      encoded: [B, T, N, H].
      atten_probs: [B, N, T, S].
    """
    p = self.params
    if segment_mask is not None or per_step_padding is not None:
      raise ValueError('Requires segment_mask=None and per_step_padding=None.')
    key_paddings = paddings
    b, t = py_utils.GetShape(query, 2)
    if query_paddings is None:
      query_paddings = tf.zeros([b, t], dtype=key_paddings.dtype)

    is_self_attention = (query is key)
    # Whether to update the centroids. Only do this during training.
    update = not self.do_eval

    query = attention_util.KMeansClusteringForAtten.LayerNorm(query)
    # [B, T, N, K]
    q_dists, _ = self.clustering.FProp(
        theta.clustering, query, query_paddings, update=update)

    if is_self_attention:
      key = query
      k_dists = q_dists
    else:
      key = attention_util.KMeansClusteringForAtten.LayerNorm(key)
      # [B, S, N, K]
      k_dists, _ = self.clustering.FProp(
          theta.clustering, key, key_paddings, update=update)
    if p.fast_path:
      encoded, probs = self._DotAttenFastPath(theta, query, key, value, q_dists,
                                              k_dists, query_paddings,
                                              key_paddings)
    else:
      encoded, probs = self._DotAttenSlowPath(theta, query, key, value, q_dists,
                                              k_dists, query_paddings,
                                              key_paddings)
    # 'probs' has shape [B, T, N, S]
    atten_probs = tf.transpose(probs, perm=[0, 2, 1, 3])
    return encoded, atten_probs

  def InitStates(self, theta, target_batch_size, target_max_length):
    """Initialize 'states' with .key, .value, and .key_dists."""
    p = self.params
    states = super().InitStates(theta, target_batch_size, target_max_length)
    states.key_dists = tf.Empty(
        shape=(target_max_length, target_batch_size, p.num_heads,
               p.num_clusters),
        dtype=py_utils.FPropDtype(p),
        init=True)
    return states

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 paddings,
                 time_step,
                 segment_mask=None,
                 per_step_padding=None,
                 use_short_seq_opt=False):
    """Computes the value vector given the query of the current step.

    This function is used by autoregressive decoding. Used for self-attention
    (hence S=T) with p.causal_masking is True.

    We compute the key/value/key_dists at `time_step` and cache the updated
    full length results in `cache_states` to reduce duplicate computation.

    p.fast_path is ignored (as if p.fast_path=False) as at each step we only
    compute for query of length 1.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:         [B, 1, D].
      cached_states:     A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. It contains .key
        and .value with shape [T, B, N, H], and .key_dists with  shape [T, B, N,
        K]. Note that they are all time-major.
      paddings:          [B, T], or None if there is no padding.
      time_step:         Scalar, the current decode step, 0-based.
      segment_mask:      must be None.
      per_step_padding:  must be None. We obey causal masking.
      use_short_seq_opt: must be False.

    Returns:
      encoded:           [B, 1, D].
      updated_states:    `.NestedMap` with .key, .value, .key_dists.

    Raises:
      ValueError: If value projection is disabled.
    """
    p = self.params
    if not p.enable_value_proj:
      raise ValueError('Value projection must be enabled: '
                       'set p.enable_value_proj = True.')
    if not p.causal_masking:
      raise ValueError('p.causal_masking must be true.')
    if segment_mask is not None or per_step_padding is not None:
      raise ValueError('Requires segment_mask=None and per_step_padding=None.')
    if use_short_seq_opt:
      raise ValueError('Requires use_short_seq_opt=False.')
    if time_step is None:
      raise ValueError('Requires valid time_step, not None.')

    t, b, n, h = py_utils.GetShape(cached_states.key, 4)

    # Project inputs to key, value and query. Each has shape [B, 1, N, H].
    key_proj = self.key.FProp(theta.key, query_vec)
    value_proj = self.value.FProp(theta.value, query_vec)
    query_proj = self.query.FProp(theta.query, query_vec)

    query_proj = attention_util.KMeansClusteringForAtten.LayerNorm(query_proj)
    key_proj = attention_util.KMeansClusteringForAtten.LayerNorm(key_proj)
    # [B, 1, N, K]
    k_dists, _ = self.clustering.FProp(theta.clustering, key_proj)

    # The updated_key and extended_value have shape [T, B, N, H].
    updated_key = scatter_update.Update(cached_states.key, time_step,
                                        tf.reshape(key_proj, [b, n, h]))
    updated_value = scatter_update.Update(cached_states.value, time_step,
                                          tf.reshape(value_proj, [b, n, h]))
    # Shape [T, B, N, K]
    updated_key_dists = scatter_update.Update(
        cached_states.key_dists, time_step,
        tf.reshape(k_dists, [b, n, p.num_clusters]))
    updated_states = py_utils.NestedMap(
        key=updated_key, value=updated_value, key_dists=updated_key_dists)

    if paddings is None:
      paddings = tf.zeros([b, t], dtype=query_vec.dtype)
    # Apply causal padding. Shape [B, T]
    paddings = tf.where(
        tf.greater(
            tf.tile(tf.range(t)[None, :], [b, 1]), tf.fill([b, t], time_step)),
        tf.ones_like(paddings), paddings)
    query_paddings = tf.zeros([b, 1], dtype=paddings.dtype)

    encoded = self._DotAttenOneStep(
        theta,
        query_proj,
        updated_states,
        query_paddings=query_paddings,
        key_paddings=paddings,
        time_step=time_step)
    # Post projection.
    encoded = self.post.FProp(theta.post, encoded)
    return encoded, updated_states

  def _DotAttenOneStep(self, theta, query, states, query_paddings, key_paddings,
                       time_step):
    """Dot attention function for queries with 1 time step.

    Called from ExtendStep(). Used for self-attention with p.causal_masking
    is True.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, 1, N, H], already normalized.
      states:   .key and .value with shape [T, B, N, H], .key_dists with shape
        [T, B, N, K]. .key is normalized.
      query_paddings: [B, 1].
      key_paddings: [B, T].
      time_step: Scalar, the current decode step, 0-based.

    Returns:
      encoded: [B, 1, N, H].
    """
    p = self.params
    # [B, 1, N, K]
    q_dists, _ = self.clustering.FProp(theta.clustering, query)
    # [B, T, N, K]
    k_dists = tf.transpose(states.key_dists, [1, 0, 2, 3])

    very_large_dists = tf.ones_like(k_dists) * tf.constant(
        0.1 * k_dists.dtype.max, dtype=k_dists.dtype)
    paddings_tiled = tf.tile(key_paddings[:, :, None, None],
                             [1, 1, p.num_heads, p.num_clusters])
    k_dists = tf.where(paddings_tiled > 0.0, very_large_dists, k_dists)

    key = tf.transpose(states.key, [1, 0, 2, 3])
    value = tf.transpose(states.value, [1, 0, 2, 3])
    encoded, _ = self._DotAttenSlowPath(
        theta,
        query,
        key,
        value,
        q_dists,
        k_dists,
        query_paddings,
        key_paddings,
        query_relative_position_shift=time_step)
    return encoded

  def _DotAttenSlowPath(self,
                        theta,
                        query,
                        key,
                        value,
                        q_dists,
                        k_dists,
                        query_paddings,
                        key_paddings,
                        query_relative_position_shift=0):
    """Computes the attention via the slow path.

    This implementation selects, on a per query basis, p.attention_window
    number of keys/values to attend to.

    Args:
      theta: A `.NestedMap` of the values of this layer's weights.
      query: [B, T, N, H], already normalized.
      key:   [B, S, N, H], already normalized.
      value: [B, S, N, H].
      q_dists: [B, T, N, K].
      k_dists: [B, S, N, K].
      query_paddings: [B, T].
      key_paddings:   [B, S].
      query_relative_position_shift: scalar. The position (relative to key[0])
        of query[0]. This impacts relative position encoding (not yet
        implemented) and causal masking.

    Returns:
      encoded: [B, T, N, H].
      atten_probs: [B, T, N, S].
    """
    p = self.params

    # [B, N, K, S]
    # If key is padded in a position, 'k_dists' is inf which ensures
    # that we consider all non-padded keys even if some padded keys
    # might appear closer.
    k_dists = tf.transpose(k_dists, [0, 2, 3, 1])

    # [B, N, K, W], for each centroid, the indices of closest key vecs.
    # It's okay if W is so larger such that a padded index is included,
    # because below in attention_util.ComputeSparseAttention() correctly
    # handles 'paddings'.
    _, closest_indices = tf.math.top_k(-k_dists, p.attention_window)
    # [B, T, N, K], one hot encoded closest centroid for each query vec.
    nearest_one_hot = tf.one_hot(
        tf.math.argmin(q_dists, axis=-1),
        p.num_clusters,
        dtype=closest_indices.dtype)

    # For each query vec, we allow it to attend to those keys that are the
    # W closest to its centroid, where W is the attention window.
    sparsity_indices = tf.einsum('BTNK, BNKW -> BTNW', nearest_one_hot,
                                 closest_indices)
    if p.causal_masking:
      batch_size, q_length, num_heads = py_utils.GetShape(query, 3)
      query_positions = tf.range(q_length) + query_relative_position_shift
      # [B, T, N, W] where the T dimension is range(T)
      query_positions = tf.tile(query_positions[None, :, None, None],
                                [batch_size, 1, num_heads, p.attention_window])
      masked_indices = -tf.ones_like(sparsity_indices)
      # Replace key positions in the future with -1 to indicate masking.
      #
      # Note that this is done after selecting top_k from 'k_dists', so for
      # example if all the closest keys are in the future, we waste
      # p.attention_window on padded keys when in theory we could have attended
      # to further away keys that are not in the future (in order to achieve
      # that we need to pick top_k from 'k_dists' differently for each query).
      sparsity_indices = tf.where(
          tf.math.greater(sparsity_indices, query_positions), masked_indices,
          sparsity_indices)

    return attention_util.ComputeSparseAttention(query, key, value,
                                                 sparsity_indices, key_paddings)

  def _DotAttenFastPath(self, theta, query, key, value, q_dists, k_dists,
                        query_paddings, key_paddings):
    """Computes the attention via the fast path.

    This implementation compute groups of queries, and for each group,
    selects a set of p.attention_window number of keys/values that each
    query in that group all attend to.

    There is no guarantee a query uniquely belong to a single group, although
    via clustering this should likely be the case. When a query belong to
    multiple groups, the attention is averaged post softmax; when a query
    does not belong to any group, the attention result is zero.

    Args:
      theta: A `.NestedMap` of the values of this layer's weights.
      query: [B, T, N, H], already normalized.
      key:   [B, S, N, H], already normalized.
      value: [B, S, N, H].
      q_dists: [B, T, N, K].
      k_dists: [B, S, N, K].
      query_paddings: [B, T].
      key_paddings:   [B, S].

    Returns:
      encoded: [B, T, N, H].
      atten_probs: [B, T, N, S]. Note, N * S * T space complexity here.
    """
    p = self.params
    # [B, N, K, S]
    # If key is padded in a position, 'k_dists' is inf which ensures
    # that we consider all non-padded keys even if some padded keys
    # might appear closer.
    k_dists = tf.transpose(k_dists, [0, 2, 3, 1])

    # [B, N, K, W], for each centroid, the indices of closest key vecs.
    # closest_k may include padded positions.
    _, closest_k = tf.math.top_k(-k_dists, p.attention_window)

    q_length = py_utils.GetShape(query, 2)[1]
    k_length = py_utils.GetShape(key, 2)[1]
    q_cluster_size = tf.cast(
        p.query_group_size_factor / p.num_clusters *
        tf.cast(q_length, py_utils.FPropDtype(p)), tf.int32)
    # Of shape [B, N, K, T]
    q_dists = tf.transpose(q_dists, [0, 2, 3, 1])
    # closest_q of shape [B, N, K, V], where V = q_cluster_size
    # closest_q may include padded positions.
    _, closest_q = tf.math.top_k(-q_dists, q_cluster_size)

    def gather(v, indx):
      """Gathers values from v.

      Args:
        v: A tensor of shape [B, T, N, D]
        indx: A tensor of shape [B, N, K, W]

      Returns:
        A value of shape [B, N, K, W, D]
      """
      # pylint: disable=invalid-name
      B, _, N, _ = py_utils.GetShape(v, 4)
      _, _, K, W = py_utils.GetShape(indx, 4)
      # pylint: enable=invalid-name
      batch_idx = tf.range(B)[:, None, None, None, None]
      batch_idx = tf.tile(batch_idx, [1, N, K, W, 1])
      seq_idx = indx[:, :, :, :, None]
      head_idx = tf.range(N)[None, :, None, None, None]
      head_idx = tf.tile(head_idx, [B, 1, K, W, 1])
      gather_idx = tf.concat([batch_idx, seq_idx, head_idx], 4)
      return tf.gather_nd(v, gather_idx)

    # c_ shorts for clustered.
    _, _, num_heads, dim_per_head = py_utils.GetShape(query, 4)
    # of shape [B, N, K, V, D]
    c_query = gather(query, closest_q)
    # of shape [B, N, K, W, D]
    c_key, c_value = tf.split(
        gather(tf.concat([key, value], -1), closest_k), 2, -1)
    # of shape [B, N, K, W, 1]
    c_key_paddings = gather(
        # [B, T, N, 1]
        tf.tile(key_paddings[:, :, None, None], [1, 1, num_heads, 1]),
        closest_k)
    # of shape [B, N, K, V, W]
    is_key_padded = tf.tile(
        tf.transpose(c_key_paddings, [0, 1, 2, 4, 3]),
        [1, 1, 1, q_cluster_size, 1]) > 0.5
    if p.causal_masking:
      # both position matrices of shape [B, N, K, V, W]
      c_query_positions = tf.tile(closest_q[:, :, :, :, None],
                                  [1, 1, 1, 1, p.attention_window])
      c_key_positions = tf.tile(closest_k[:, :, :, None, :],
                                [1, 1, 1, q_cluster_size, 1])
      # We pad the logit for future key positions relative to each query
      is_key_padded = tf.math.logical_or(
          is_key_padded, tf.math.greater(c_key_positions, c_query_positions))

    logits = tf.einsum('BNKVD,BNKWD->BNKVW', c_query, c_key)
    logits *= tf.math.rsqrt(tf.cast(dim_per_head, py_utils.FPropDtype(p)))

    padded_logits = py_utils.ApplyPadding(is_key_padded, logits,
                                          GetDtypeMin(logits.dtype))

    c_atten_probs = tf.nn.softmax(padded_logits)
    c_outputs = tf.einsum('BNKWD,BNKVW->BNKVD', c_value, c_atten_probs)

    def scatter(v, indx, seq_len):
      """Scatters v according to indx.

      Args:
        v: A tensor of shape [B, N, K, V, D].
        indx: A tensor of shape [B, N, K, V].
        seq_len: sequence length of the output.

      Returns:
        output: A tensor of shape [B, T, N, D], where T = seq_len.
      """
      # Need to scatter outputs back to the original shape.
      # pylint: disable=invalid-name
      B, N, K, V, D = py_utils.GetShape(v, 5)
      # pylint: enable=invalid-name
      # [B, N, K, V, 1]
      batch_idx = tf.tile(
          tf.range(B)[:, None, None, None, None], [1, N, K, V, 1])
      # [B, N, K, V, 1]
      seq_idx = indx[:, :, :, :, None]
      # [B, N, K, V, 1]
      head_idx = tf.tile(
          tf.range(N)[None, :, None, None, None], [B, 1, K, V, 1])
      scatter_idx = tf.concat([batch_idx, seq_idx, head_idx], 4)
      scattered = tf.scatter_nd(
          scatter_idx, tf.concat([v, tf.ones_like(v[:, :, :, :, :1])], -1),
          [B, seq_len, N, D + 1])
      # We need to normaliz as one query vector may appear in multiple clusters.
      scattered, den = tf.split(scattered, [v.shape.as_list()[-1], 1], -1)
      # den = tf.squeeze(den, -1)
      out = scattered / tf.maximum(tf.constant(1.0, dtype=den.dtype),
                                   den)  # [:, :, :, None])
      return out

    def scatter_atten_prob(c_atten_probs, closest_k, closest_q, k_length,
                           q_length):
      """Scatters c_atten_probs.

      Args:
        c_atten_probs: A tensor of shape [B, N, K, V, W].
        closest_k: A tensor of shape [B, N, K, W].
        closest_q: A tensor of shape [B, N, K, V].
        k_length: Length of the key vectors.
        q_length: Length of the query vectors.

      Returns:
        output: A tensor of shape [B, q_length, N, k_length].
      """
      # Need to scatter outputs back to the original shape.
      # pylint: disable=invalid-name
      B, N, K, V, W = py_utils.GetShape(c_atten_probs, 5)
      # pylint: enable=invalid-name
      # [B, N, K, V, W, 1]
      batch_idx = tf.tile(
          tf.range(B)[:, None, None, None, None, None], [1, N, K, V, W, 1])
      # [B, N, K, V, W, 1]
      k_idx = tf.tile(closest_k[:, :, :, None, :, None], [1, 1, 1, V, 1, 1])
      q_idx = tf.tile(closest_q[:, :, :, :, None, None], [1, 1, 1, 1, W, 1])
      head_idx = tf.tile(
          tf.range(N)[None, :, None, None, None, None], [B, 1, K, V, W, 1])
      scatter_idx = tf.concat([batch_idx, q_idx, head_idx, k_idx], 5)
      scattered_prob = tf.scatter_nd(scatter_idx, c_atten_probs,
                                     [B, q_length, N, k_length])

      # We need to normalize the attention prob as one query vector may appear
      # in multiple clusters.
      # [B, N, K, V, 3]
      times_idx = tf.concat([batch_idx, q_idx, head_idx], 5)[:, :, :, :, 0, :]
      # [B, q_length, N]
      times = tf.scatter_nd(
          times_idx, tf.cast(tf.ones_like(closest_q), scattered_prob.dtype),
          [B, q_length, N])
      times = tf.maximum(tf.cast(1.0, times.dtype), times[:, :, :, None])
      out = scattered_prob / times
      return out

    out = scatter(c_outputs, closest_q, q_length)
    out_prob = scatter_atten_prob(c_atten_probs, closest_k, closest_q, k_length,
                                  q_length)
    return out, out_prob


class MultiSourceAttention(base_layer.BaseLayer):
  """Batch major attention with multiple source sub-attentions.

  It attends to multiple sources and uses one query as input to generates a
  combined attention context. The dimension of the combined context vector is a
  sum of all source context vectors. Each source attention has its separate
  params and is associated with a source key.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('source_atten_tpls', None,
             'A list of (source_key, attention_param) pairs.')
    p.Define('input_dim', 0, 'Default key dimension.')
    p.Define('hidden_dim', 0, 'Default hidden dimension.')
    p.Define(
        'primary_source_key', 'source_0', 'Key for the primary source '
        'whose attention probabilities will be used as an output.')
    p.Define('atten_merger_tpl', None,
             'Params to specify how to merge source attention vectors.')
    return p

  def __init__(self, params):
    """Constructs an MultiSourceAttention object."""
    super().__init__(params)
    p = self.params
    assert p.primary_source_key in [
        x for x, _ in p.source_atten_tpls
    ], ('Source attention must have the primary source key.')
    for source_key, atten_p in p.source_atten_tpls:
      if isinstance(atten_p, list):
        child_p_list = []
        for atten in atten_p:
          child_p = atten.Copy()
          if child_p.hidden_dim <= 0:
            child_p.hidden_dim = p.hidden_dim
          if child_p.input_dim <= 0:
            child_p.input_dim = p.input_dim
          child_p_list.append(child_p)
          self.CreateChildren('atten_%s' % source_key, child_p_list)
      else:
        child_p = atten_p.Copy()
        if child_p.hidden_dim <= 0:
          child_p.hidden_dim = p.hidden_dim
        if child_p.input_dim <= 0:
          child_p.input_dim = p.input_dim
        self.CreateChild('atten_%s' % source_key, child_p)

    # Initialize source context vector merging layer.
    merger_p = p.atten_merger_tpl.Copy()
    merger_p.name = 'atten_merger'
    merger_p.source_dim = p.input_dim
    merger_p.query_dim = p.input_dim
    self.CreateChild('atten_merger', merger_p)

  def FProp(self,
            theta,
            query_vec,
            key_vec,
            value_vec,
            paddings,
            segment_mask=None,
            per_step_padding=None):
    p = self.params
    with tf.name_scope(self.params.name):
      result_map = py_utils.NestedMap()
      for source_key, _ in p.source_atten_tpls:
        result_map[source_key] = (
            self.children['atten_%s' % source_key].FProp(
                theta.get('atten_%s' % source_key), query_vec,
                key_vec[source_key], value_vec[source_key],
                paddings[source_key],
                segment_mask[source_key] if segment_mask else None,
                per_step_padding))
      return self._CombineContext(theta, result_map, query_vec)

  def _CombineContext(self, theta, enc_map, query_vec):
    encs = enc_map.Flatten()
    combined_enc = (
        self.atten_merger.FProp(theta.atten_merger, [enc for enc, _ in encs],
                                query_vec))
    # Return atten_probs of the primary source.
    return combined_enc, enc_map[self.params.primary_source_key][1]

  def AttenProbs(self,
                 theta,
                 query,
                 key,
                 paddings,
                 segment_mask,
                 per_step_padding=None):
    primary_source_key = self.params.primary_source_key
    child_name = 'atten_%s' % primary_source_key
    return self.children[child_name].AttenProbs(
        theta.get(child_name), query, key[primary_source_key],
        paddings[primary_source_key],
        segment_mask[primary_source_key] if segment_mask else None,
        per_step_padding)


class TransformerAttentionLayer(base_layer.BaseLayer):
  """Multiheaded attention sub-layer in Transformer layer.

  Input is first normalized using Layer Normalization. Output of layer
  normalization is processed using multi-headed attention. And finally, the
  output of the attention layer is combined with the residual connection.
  This module also allows mixing different attention heads by making num_heads
  and atten_tpl into lists of the same size, specifying the distribution of
  heads for each attention type.

  This layer will be used in the following two scenarios:

  1. Multi-Headed Self-Attention, where attention keys, values (source_vecs) and
     queries come from the same previous layer output.
  2. Masked Multi-Headed Self-Attention, where attention keys, values and
     queries all come from the same previous layer output, but rightward
     activations are masked to prevent information flow from future. This is the
     use case for Transformer decoder self-attention layers. Can be activated by
     setting is_masked flag of this layer.
  3. Multi-Headed Cross-Attention, where attention keys and values
     (source_vecs) are coming from a different source (output of the encoder),
     and queries coming from the previous layer outputs (decoder).
  4. Mixture of different heads, for example 2 LocalSelfAttention heads
     and 3 RoutingAttention heads can be specified by setting num_heads = [2, 3]
     and atten_tpl = [LocalSelfAttention, RoutingAttention].

  We use the same capital letters to denote certain tensor parameters as
  MultiHeadedAttention class.

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    D = model dimension
    N = number of attention heads
    H = dimensions of each attention head.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the transformer input.')
    p.Define('hidden_dim', 0, 'Dimension of the attention hidden dim.')
    p.Define(
        'num_heads', 8, 'Number of attention heads. This can be a list in'
        'case of mixture of attention types')
    p.Define(
        'is_masked', False,
        'If set, uses causal non local multiheaded attention.'
        'This option is not valid when atten_tpl is LocalSelfAttention '
        'or its subclass(es).')
    p.Define(
        'atten_dropout_prob', 0.0,
        'Probability at which we apply dropout to the attention probs. '
        'This practically drops memory values at random positions.')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, y) = (x + dropout(y)).')
    p.Define('pre_layer_norm', True, 'Pre or post layer norm.')
    p.Define(
        'add_unnormalized_input', True,
        'If set, uses unnormalized input in the residual add. It is'
        'applicable only if pre_layer_norm is True.')
    p.Define('add_skip_connection', True,
             'If True, add input (or normalized input) to the output.')
    p.Define('ln_tpl', layers.LayerNorm.Params(),
             'Layer norm default params. No layernorm if set to None.')
    p.Define(
        'atten_tpl', MultiHeadedAttention.Params(),
        'Multi-Headed Dot-Product Attention default params. This can be'
        'a list in the case of mixture of attentions, must be of same size'
        'as num_heads')
    p.Define(
        'dropout_tpl', layers.DropoutLayer.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define('residual_droppath_prob', 0.0,
             'Probability at which we drop the entire residual path.')
    return p

  @classmethod
  def CommonParams(
      cls,
      input_dim,
      num_heads,
      is_masked=False,
      use_relative_atten=False,
      relative_pos_emb_dim=None,
      local_context=None,
      left_context=None,
      right_context=None,
      dropout_prob=0.,
  ):
    # pylint: disable=g-doc-args
    """Returns a hyperparam for the most representative cases.

    CommonParams is not expected to be extended to an omnipotent/generic builder
    method. Specific use cases should take the return value of it and apply
    further customization. It should be kept lean and only extended cautiously
    for very common cases.
    """
    # pylint: enable=g-doc-args
    if not use_relative_atten:
      assert not relative_pos_emb_dim
    else:
      relative_pos_emb_dim = relative_pos_emb_dim or input_dim

    if local_context:
      assert not left_context and not right_context, (
          'local_context and (left_context, right_context) can not be set '
          'at the same time.')
      left_context = local_context + 1  # include 'self' position.
      right_context = local_context

    p = cls.Params().Set(
        input_dim=input_dim,
        num_heads=num_heads,
        is_masked=is_masked,
        atten_dropout_prob=dropout_prob,
        residual_dropout_prob=dropout_prob)

    is_local = left_context or right_context
    if is_local:
      atten_cls = (
          LocalSelfAttentionXL if use_relative_atten else LocalSelfAttention)
    else:
      atten_cls = (
          MultiHeadedAttentionXL
          if use_relative_atten else MultiHeadedAttention)
    p.atten_tpl = atten_cls.Params()
    if use_relative_atten:
      p.atten_tpl.rel_pos_emb_dim = relative_pos_emb_dim
    if is_local:
      p.atten_tpl.Set(left_context=left_context, right_context=right_context)
    return p

  @classmethod
  def SetFPropDtype(cls, p, fprop_dtype):
    p.fprop_dtype = fprop_dtype
    return p

  def _InitAttentionParams(self, atten_tpl):
    """Returns an initialized transformer attention parameters."""
    p = self.params

    if isinstance(p.num_heads, list) != isinstance(atten_tpl, list):
      raise ValueError('p.num_heads and p.atten_tpl should both be lists '
                       f'or both scalars for {p.name} num_heads={p.num_heads}.')
    if isinstance(p.num_heads, list) and (len(p.num_heads) != len(atten_tpl)):
      raise ValueError('num_heads and atten_tpl should both be lists '
                       'of the equal sizes: '
                       f'{len(p.num_heads)} vs {len(atten_tpl)}')

    def _SetCommonParams(params, name, num_heads):
      params.name = name
      params.input_dim = p.input_dim
      params.hidden_dim = p.hidden_dim
      params.num_heads = num_heads
      params.atten_dropout_prob = p.atten_dropout_prob
      if isinstance(p.num_heads, list):
        params.proj_tpl.make_output_proj_no_op = True
        # Each dim per head is now divided among all heads
        dim_per_head = p.hidden_dim // sum(p.num_heads)
        params.proj_tpl.dim_per_head = dim_per_head
      return params

    if isinstance(p.num_heads, list):
      params_list = []
      for i in range(len(atten_tpl)):
        params = atten_tpl[i].Copy()
        params = _SetCommonParams(params, 'mixed_atten_{}'.format(i),
                                  p.num_heads[i])
        params_list.append(params)
      params = params_list
    else:
      params = atten_tpl.Copy()
      params = _SetCommonParams(params, 'multihead_atten', p.num_heads)
    return params

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if isinstance(p.input_dim, dict):
      query_input_dim = p.input_dim['query']
    else:
      query_input_dim = p.input_dim
    if not p.hidden_dim:
      p.hidden_dim = query_input_dim

    # Initialize attention.
    def _LocalAttentionError(params):
      if p.is_masked and issubclass(params.cls, LocalSelfAttention):
        tf.logging.warning('\'is_masked\' is not effective when used with '
                           'LocalSelfAttention and its subclass(es).')

    params = self._InitAttentionParams(p.atten_tpl)
    if isinstance(params, list):
      for i in range(len(params)):
        _LocalAttentionError(params[i])
      self.CreateChildren('atten', params)
      # Create head mixing variable
      dim_per_head = p.hidden_dim // sum(p.num_heads)
      # For the projection merging layer, parameter settings from the first
      # attention template in the list is used.
      self.CreateChild(
          'w_mix_heads', p.atten_tpl[0].proj_tpl.Copy().Set(
              input_dim=p.input_dim,
              num_heads=sum(p.num_heads),
              dim_per_head=dim_per_head,
              is_output_projection=True,
              make_output_proj_no_op=False,
              use_bias=p.atten_tpl[0].use_bias,
              device_mesh=p.atten_tpl[0].device_mesh,
              weight_split_dims_mapping=p.atten_tpl[0].weight_split_dims_mapping
          ))
    else:
      _LocalAttentionError(params)
      self.CreateChild('atten', params)

    # Initialize attention layer normalization.
    if p.ln_tpl:
      params = p.ln_tpl.Copy()
      params.name = 'atten_ln'
      params.input_dim = query_input_dim
      self.CreateChild('layer_norm', params)

    # Initialize residual dropout.
    dropout_tpl = p.dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.CreateChild('residual_dropout', dropout_tpl)
    # Initialize droppath layer if needed.
    if p.residual_droppath_prob > 0:
      droppath_p = layers_with_attention.StochasticResidualLayer.Params().Set(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.CreateChild('residual_droppath', droppath_p)
    self._prefix_state_target_max_length = None

  def FProp(self,
            theta,
            query_vec,
            source_vecs,
            paddings,
            per_step_padding_override=None,
            segment_mask=None):
    """Compute the result of Transformer attention layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:   [B, T, D].
      source_vecs: [B, S, D] (cross_attention) or None (self-attention).
      paddings:    [B, S].
      per_step_padding_override: [B, T, T] for self attention or [B, T, S] for
        cross attention.
      segment_mask: [B, 1, T, S].

    Returns:
      output: [B, T, D].
      atten_probs: [B, N, T, S].
    """
    p = self.params

    (query_vec, source_vecs, paddings, per_step_padding_override,
     segment_mask) = self._CastToFPropDtype(
         (query_vec, source_vecs, paddings, per_step_padding_override,
          segment_mask))

    b, t, _ = py_utils.GetShape(query_vec, 3)
    unnormalized_query_vec = query_vec

    # Layer normalization.
    if p.pre_layer_norm and p.ln_tpl:
      query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)
      query_vec = self._CastToFPropDtype(query_vec)

    # For self-attention: keys = queries.
    if source_vecs is None:
      source_vecs = query_vec

    # Generates mask, with shape [b, t, t].
    if per_step_padding_override is None:
      if p.is_masked and segment_mask is None:
        # causal padding.
        per_step_padding = tf.tile(
            tf.expand_dims(CausalPadding(t, dtype=query_vec.dtype), 0),
            [b, 1, 1])
      else:
        per_step_padding = None
    else:
      per_step_padding = per_step_padding_override

    # Multiheaded attention.
    def _AttenFProp(atten, theta):
      return atten.FProp(
          theta,
          query_vec,  # query
          source_vecs,  # key
          source_vecs,  # value
          paddings,
          segment_mask=segment_mask,
          per_step_padding=per_step_padding)

    with tf.name_scope('atten'):
      if isinstance(self.atten, list):
        ctx_vec_list = []
        atten_probs_list = []
        for i in range(len(self.atten)):
          ctx_vec, atten_probs = _AttenFProp(self.atten[i], theta.atten[i])
          ctx_vec_list.append(ctx_vec)
          atten_probs_list.append(atten_probs)
        # Concat all the outputs together
        ctx_vec = tf.concat(ctx_vec_list, axis=2)
        # ctx_vec has shape [B, T, N, H] due to identity projection
        ctx_vec = self.w_mix_heads.FProp(theta.w_mix_heads, ctx_vec)
        atten_probs = tf.concat(atten_probs_list, axis=1)
      else:
        ctx_vec, atten_probs = _AttenFProp(self.atten, theta.atten)

    # Residual connection.
    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    if p.add_skip_connection:
      if p.residual_droppath_prob:
        ctx_vec = self.residual_droppath.FProp(
            theta.residual_droppath,
            input_to_add,
            ctx_vec,
        )
      else:
        ctx_vec += input_to_add
    if not p.pre_layer_norm and p.ln_tpl:
      ctx_vec = self.layer_norm.FProp(theta.layer_norm, ctx_vec)
      ctx_vec = self._CastToFPropDtype(ctx_vec)
    return ctx_vec, atten_probs

  def InitStates(self, theta, target_batch_size, target_max_length):
    # Because we memoize `target_max_length`, we do not support
    # interleaving different InitStates() and ExtendStep() sequences.
    if (self._prefix_state_target_max_length and
        self._prefix_state_target_max_length != target_max_length):
      raise ValueError(
          'InitStates() cannot be called twice with different values '
          f'for target_max_length values: now {target_max_length} vs '
          f'before {self._prefix_state_target_max_length}')
    self._prefix_state_target_max_length = target_max_length
    if isinstance(self.atten, list):
      return py_utils.NestedMap(atten=[
          a.InitStates(a_theta, target_batch_size, target_max_length)
          for a, a_theta in zip(self.atten, theta)
      ])
    return self.atten.InitStates(theta.atten, target_batch_size,
                                 target_max_length)

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 time_step,
                 use_short_seq_opt=False,
                 per_step_padding=None,
                 segment_mask=None):
    """Compute the result and update cached states for the current step.

    This function is used by autoregressive decoding. This function knows the
    length of full sequence, thus it is different from StreamStep.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [B, 1, D]
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. key   - [T, B,
        N, H]. value - [T, B, N, H].
      time_step: A scalar or tensor with [B], current decode step, 0-based. if
        it's a scalar, all the time step are the same decode step. if it's a
        tensor, it represents current decode step for each sample.
      use_short_seq_opt: A bool, whether using short sequence optimization.
      per_step_padding: optional customized padding for this step - [B, 1, T].
      segment_mask: optional segment_mask for this step - [B, 1, T].
        segment_mask has already been converted to large negative number.

    Returns:
      cur_output: [B, 1, D]
      updated_states: A `.NestedMap` object containing the updated states.
      key   - [T, B, N, H].
      value - [T, B, N, H].

    Raises:
      ValueError: If not used as masked/causal self-attention.
    """
    p = self.params
    query_vec = self._CastToFPropDtype(query_vec)
    if not p.is_masked and per_step_padding is None:
      raise ValueError(
          'ExtendStep should be used only by masked/causal self-attention.')
    if isinstance(self.atten, list):
      t = py_utils.GetShape(cached_states.atten[0].key, 1)[0]
    elif 'key' in cached_states:
      t = py_utils.GetShape(cached_states.key, 1)[0]
    else:
      # Ideally we only need this branch, but many unit tests do not call
      # InitState().
      if not self._prefix_state_target_max_length:
        raise ValueError('Must call InitState() before ExtendStep()')
      t = self._prefix_state_target_max_length
    b = py_utils.GetShape(query_vec, 1)[0]

    unnormalized_query_vec = query_vec
    time_step = tf.convert_to_tensor(time_step)

    if time_step.shape.ndims == 0:
      batch_time_step = tf.tile(tf.reshape(time_step, [-1]), [b])
    else:
      batch_time_step = time_step

    if per_step_padding is None:
      # Generates mask, with shape [b, 1, t].
      zero_padding = tf.zeros([b, t], dtype=query_vec.dtype)
      # [b, t]
      per_step_padding = tf.where(
          tf.less(
              tf.tile(tf.expand_dims(tf.range(t), 0), [b, 1]),
              tf.expand_dims(batch_time_step + 1, -1)), zero_padding,
          tf.ones_like(zero_padding, dtype=query_vec.dtype))
      # [b, 1, t]
      per_step_padding = tf.expand_dims(per_step_padding, 1)
    per_step_padding = py_utils.HasShape(per_step_padding, [b, 1, t])

    if segment_mask is not None:
      segment_mask = py_utils.HasShape(segment_mask, [b, 1, t])
      segment_padding = tf.where(
          tf.less(segment_mask, 0.0), tf.ones_like(segment_mask),
          tf.zeros_like(segment_mask))
      # Adjust per_step_padding to also take into account segment_padding.
      per_step_padding = tf.minimum(per_step_padding + segment_padding, 1.0)

    # Layer normalization.
    if p.ln_tpl:
      query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)
      query_vec = self._CastToFPropDtype(query_vec)

    # Multiheaded masked/causal self-attention.
    def _AttenExtendStep(atten, theta, cached_states):
      return atten.ExtendStep(theta, query_vec, cached_states, None, None,
                              per_step_padding, time_step, use_short_seq_opt)

    if isinstance(self.atten, list):
      updated_states = py_utils.NestedMap(atten=[])
      ctx_vec_list = []
      for i in range(len(self.atten)):
        ctx_vec, updated_atten_states = _AttenExtendStep(
            self.atten[i], theta.atten[i], cached_states.atten[i])
        ctx_vec_list.append(ctx_vec)
        updated_states.atten.append(updated_atten_states)
      # Concat all attention heads together
      ctx_vec = tf.concat(ctx_vec_list, axis=2)
      # ctx_vec has shape [B, T, N, H] due to identity projection
      ctx_vec = self.w_mix_heads.FProp(theta.w_mix_heads, ctx_vec)
    else:
      ctx_vec, updated_states = _AttenExtendStep(self.atten, theta.atten,
                                                 cached_states)

    # Residual connection.
    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    if p.add_skip_connection:
      ctx_vec += input_to_add
    return ctx_vec, updated_states

  def zero_state(self, batch_size=1):
    """Returns the initial state given the batch size.

    Args:
      batch_size: the batch size.

    Returns:
      state: The initial state for streaming inference.
    """
    return py_utils.NestedMap(atten=self.atten.zero_state(batch_size))

  def StreamStep(self, theta, query_vec, paddings, state0):
    """Computes the value vector given the query of the current step.

    Args:
      theta: a NestedMap with layer weights.
      query_vec: A query vector of shape [B, T, D].
      paddings: A 0/1 valued tensor of shape [B, T].
      state0: A `.NestedMap` of the same structure as returned by zero_state().

    Returns:
      output: Output of the given query vector with shape [B, T, D].
      padding: the same as input paddings.
      state: updated state.
    """
    p = self.params
    assert p.is_masked
    with tf.name_scope(f'{p.name}/StreamStep'):
      query_vec, paddings = self._CastToFPropDtype((query_vec, paddings))
      unnormalized_query_vec = query_vec

      if p.ln_tpl:
        query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)
        query_vec = self._CastToFPropDtype(query_vec)

      output, paddings, atten_state1 = self.atten.StreamStep(
          theta.atten, query_vec, paddings, state0.atten)

      output = self.residual_dropout.FProp(theta.residual_dropout, output)

      # Residual connection.
      input_to_add = (
          unnormalized_query_vec if p.add_unnormalized_input else query_vec)

      if p.add_skip_connection:
        output, atten_state1 = self.atten.StreamStepAddSkipConnection(
            input_to_add, output, state0.atten, atten_state1)
      return output, paddings, py_utils.NestedMap(atten=atten_state1)


class TransformerMultiSourceAttentionLayer(TransformerAttentionLayer):
  """Batch major multi-source multi-headed attention.

  Only supports scenarios 3 described by comments on TransformerAttentionLayer:

  3. Multi-source multi-headed cross-attention, where attention keys and values
     (source_vecs) are coming from different sources (one of them is usually
     the outputs of the encoder), and queries coming from the previous layer
     outputs (decoder). Specifically, attention keys and values are NestedMaps
     containing encodings of different sources. This corresponds to a
     multi-source decoder-to-encoder attention mechanism, i.e., decoder attends
     to encoder outputs and other sources.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_source', 0, 'Number of sources to attend to.')
    p.Define(
        'primary_source_index', 0, 'Index of the primary source whose '
        'attention probs will be returned.')
    p.Define('multi_source_atten', MultiSourceAttention.Params(),
             'Multi-source attention params.')
    # Only used for case 3.
    p.is_masked = False
    return p

  def _InitAttentionParams(self, atten_tpl):
    """Returns an initialized multi-source transformer attention parameters."""
    p = self.params
    source_atten_tpls = []
    # Set up each source attention.
    for i in range(p.num_source):
      src_key = 'source_%d' % i
      src_atten = atten_tpl.Copy()
      src_atten = super()._InitAttentionParams(src_atten)
      if isinstance(src_atten, list):
        raise ValueError(
            'TransformerMultiSourceAttentionLayer does not support '
            'num_heads > 1.')
      src_atten.name = 'multihead_atten_%s' % src_key
      source_atten_tpls.append((src_key, src_atten))

    # Initialize multi-source attention.
    msa = p.multi_source_atten.Copy()
    msa.name = 'multi_source_atten'
    msa.input_dim = p.input_dim
    msa.hidden_dim = p.hidden_dim
    msa.source_atten_tpls = source_atten_tpls
    msa.primary_source_key = 'source_%d' % p.primary_source_index
    return msa


class TransformerLayer(base_layer.BaseLayer):
  """Transformer layer with multiheaded attention.

  Applies self-attention followed by a cross-attention and feed forward layer.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the transformer block input.')
    p.Define('output_dim', 0, 'Dimension of the transformer block output.')
    p.Define('num_heads', None, 'Num of heads in self attention.')
    p.Define('has_aux_atten', False,
             'If set, introduces a second attention layer')
    p.Define('mask_self_atten', False, 'If True, use masked self-attention.')
    p.Define('tr_atten_tpl',
             TransformerAttentionLayer.Params().Set(),
             'Transformer Attention Layer params.')
    p.Define(
        'tr_self_atten_tpl', None,
        'Attention template for self attention. If unset, use tr_atten_tpl.')
    p.Define(
        'tr_fflayer_tpl',
        layers_with_attention.TransformerFeedForwardLayer.Params().Set(
            hidden_dim=2048), 'Transformer Feed-Forward Layer params.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define('compute_flops', False,
             'If True adds computation cost for the layer FLOPs')
    return p

  @classmethod
  def SetCanonicalShardingParams(cls,
                                 params,
                                 reshape_dim=False,
                                 shard_atten_w=True):
    """Set up canonical SPMD sharding params.

    The topology is required to written as 2D. For 1D sharding, the topology is
    expected to be written as [1, num_partitions]. The split_dims_mappings
    specify how weights and activations are sharded in the corresponding layers.
    fflayer has two projection layers(df/blf and fd/bld), so the
    split_dims_mapping has higher rank to represent both projection layers one
    after another.
    For 1D sharding, better performance can be obtained by sharding activations
    on batch dim, so bld is set to [1, -1, -1] and blnh to None (will be auto
    propagated).
    For 2D sharding, typical sharding is [0, -1, 1, -1] for blnh and [0, -1, 1]
    for bld. If ReshapeDim trick is applied to model dim to remove the data
    formatting overheads, the bld sharding annotation needs to be adapted as
    [0, -1, 1, -1].

    Args:
      params: params of TransformerLayer.
      reshape_dim: A bool, whether to reshape model dim.
      shard_atten_w: A bool, whether to shard attention weight.
    """
    # Weights
    params.tr_atten_tpl.atten_tpl.weight_split_dims_mapping = [
        0, 1, -1
    ] if shard_atten_w else [-1, -1, -1]
    params.tr_fflayer_tpl.fflayer_tpl.weight_split_dims_mapping_list = [[0, 1],
                                                                        [1, 0]]
    # Activations
    params.tr_atten_tpl.atten_tpl.activation_split_dims_mapping.blnh = None
    bld_split = [1, -1, -1]
    blf_split = [0, -1, 1]
    sharding_2d = (
        params.device_mesh.shape[0] != 1 and params.device_mesh.shape[1] != 1)
    if sharding_2d:
      params.tr_atten_tpl.atten_tpl.activation_split_dims_mapping.blnh = [
          0, -1, 1, -1
      ]
      bld_split = ([0, -1, 1, -1] if reshape_dim else [0, -1, 1])
    params.tr_atten_tpl.atten_tpl.activation_split_dims_mapping.bld = bld_split
    params.tr_fflayer_tpl.fflayer_tpl.activation_split_dims_mapping_list = [
        blf_split, bld_split
    ]

  @classmethod
  def SetFPropDtype(cls, p, fprop_dtype):
    p.fprop_dtype = fprop_dtype
    for sub_p in (p.tr_atten_tpl, p.tr_self_atten_tpl, p.tr_fflayer_tpl):
      if sub_p is not None:
        sub_p.cls.SetFPropDtype(sub_p, fprop_dtype)
    return p

  @classmethod
  def SetReshapedLayers(cls, params):

    def _CopyParams(old_params, new_params):
      old_params_dict = dict(old_params.IterParams())
      del old_params_dict['cls']
      new_params.Set(**old_params_dict)

    old_tr_fflayer_p = params.tr_fflayer_tpl
    old_tr_fflayer_ln_p = params.tr_fflayer_tpl.ln_tpl
    old_tr_atten_ln_p = params.tr_atten_tpl.ln_tpl
    old_tr_atten_atten_p = params.tr_atten_tpl.atten_tpl
    old_tr_atten_atten_proj_p = params.tr_atten_tpl.atten_tpl.proj_tpl

    params.tr_fflayer_tpl = (
        layers_with_attention.ReshapedTransformerFeedForwardLayer.Params())
    _CopyParams(old_tr_fflayer_p, params.tr_fflayer_tpl)
    params.tr_fflayer_tpl.ln_tpl = layers.ReshapedLayerNorm.Params()
    _CopyParams(old_tr_fflayer_ln_p, params.tr_fflayer_tpl.ln_tpl)
    params.tr_atten_tpl.ln_tpl = layers.ReshapedLayerNorm.Params()
    _CopyParams(old_tr_atten_ln_p, params.tr_atten_tpl.ln_tpl)
    params.tr_atten_tpl.atten_tpl = ReshapedMultiHeadedAttention.Params()
    _CopyParams(old_tr_atten_atten_p, params.tr_atten_tpl.atten_tpl)
    params.tr_atten_tpl.atten_tpl.proj_tpl = (
        ReshapedMultiHeadedProjectionLayer.Params())
    _CopyParams(old_tr_atten_atten_proj_p,
                params.tr_atten_tpl.atten_tpl.proj_tpl)

  @classmethod
  def CommonParams(cls,
                   input_dim,
                   atten_num_heads,
                   atten_is_relative=False,
                   atten_local_context=None,
                   atten_left_context=None,
                   atten_right_context=None,
                   has_aux_atten=False,
                   mask_self_atten=False,
                   fflayer_hidden_dim=None,
                   fflayer_output_dim=None,
                   dropout_prob=0.):
    # pylint: disable=g-doc-args
    """Returns a hyperparam for the most representative cases.

    CommonParams is not expected to be extended to an omnipotent/generic builder
    method. Specific use cases should take the return value of it and apply
    further customization. It should be kept lean and only extended cautiously
    for very common cases.
    """
    # pylint: enable=g-doc-args
    output_dim = fflayer_output_dim or input_dim
    fflayer_hidden_dim = fflayer_hidden_dim or 4 * input_dim
    # TODO(jamesqin): check how mask_self_atten work with local atten.
    p = cls.Params().Set(
        name='transformer_layer',
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=atten_num_heads,
        has_aux_atten=has_aux_atten,
        mask_self_atten=mask_self_atten)
    p.tr_self_atten_tpl = TransformerAttentionLayer.CommonParams(
        input_dim,
        atten_num_heads,
        use_relative_atten=atten_is_relative,
        is_masked=mask_self_atten,
        local_context=atten_local_context,
        left_context=atten_left_context,
        right_context=atten_right_context,
        dropout_prob=dropout_prob)
    p.tr_fflayer_tpl.Set(
        hidden_dim=fflayer_hidden_dim,
        residual_dropout_prob=dropout_prob,
        relu_dropout_prob=dropout_prob)
    return p

  @classmethod
  def SetNumInputNodes(cls, p, num_input_nodes):
    p.input_dim = num_input_nodes

  @classmethod
  def NumOutputNodes(cls, p):
    return p.output_dim

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    # Initialize masked multi-headed self-attention
    if p.tr_self_atten_tpl is not None:
      self_atten_tpl = p.tr_self_atten_tpl
    else:
      self_atten_tpl = p.tr_atten_tpl
    params = self_atten_tpl.Copy()
    params.name = 'multihead_self_atten'
    params.input_dim = p.input_dim
    params.is_masked = p.mask_self_atten
    if p.num_heads and not isinstance(params.num_heads, list):
      params.num_heads = p.num_heads
    if isinstance(params.atten_tpl, list):
      for atten in params.atten_tpl:
        atten.packed_input = p.packed_input
    else:
      params.atten_tpl.packed_input = p.packed_input
    self.CreateChild('self_atten', params)

    if p.has_aux_atten:
      # Initialize multi-headed cross-attention
      params = p.tr_atten_tpl.Copy()
      params.name = 'multihead_cross_atten'
      params.input_dim = p.input_dim
      if p.num_heads and not isinstance(params.num_heads, list):
        params.num_heads = p.num_heads
      if isinstance(params.atten_tpl, list):
        for atten in params.atten_tpl:
          atten.packed_input = p.packed_input
      else:
        params.atten_tpl.packed_input = p.packed_input
      self.CreateChild('cross_atten', params)

    # Initialize feed-forward layer
    params = p.tr_fflayer_tpl.Copy()
    params.name = 'tr_fflayer'
    params.input_dim = p.input_dim
    params.output_dim = p.output_dim
    self.CreateChild('fflayer', params)

  def _GetSourceBatchSize(self, aux_vec):
    return py_utils.GetShape(aux_vec, 2)[0]

  def _GetSourceLength(self, aux_vec):
    return py_utils.GetShape(aux_vec, 2)[1]

  def FProp(self,
            theta,
            query_vec,
            paddings,
            aux_vec=None,
            aux_paddings=None,
            per_step_padding_override=None,
            aux_per_step_padding_override=None,
            segment_mask=None,
            aux_segment_mask=None):
    """Transformer decoder layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:    [target_batch, target_time, dim].
      paddings:     [target_batch, target_time].
      aux_vec:      [source_batch, source_time, dim].
      aux_paddings: [source_batch, source_time].
      per_step_padding_override: [target_batch, target_time, target_time].
      aux_per_step_padding_override: [target_batch, target_time, source_time].
      segment_mask:     [target_batch, 1, target_time, target_time].
      aux_segment_mask: [source_batch, 1, target_time, source_time].
        target_batch can be a multiple of source_batch, where samples in
        target_batch are arranged in the order of [m, source_batch] where m =
        target_batch / source_batch.

    Returns:
      The fflayer output with shape [target_batch, target_time, dim].
      atten_probs: A NestedMap with keys `self_atten` <float>[B, N, T, T], and
      `aux_atten` (optional): <float>[B, N, T, S].
    """
    p = self.params
    if p.compute_flops:
      assert not p.has_aux_atten, (
          'Current FLOPs computation does not include auxiliary attention')
      computation_cost.Add(
          self, 'flops',
          TransformerFlops(
              tf.shape(query_vec), p.tr_atten_tpl.num_heads,
              symbolic.EvalExpr(symbolic.TENSOR_VALUES,
                                p.tr_fflayer_tpl.hidden_dim),
              symbolic.EvalExpr(symbolic.TENSOR_VALUES,
                                p.tr_atten_tpl.hidden_dim),
              symbolic.EvalExpr(symbolic.TENSOR_VALUES, p.input_dim)))
    # First the self-attention layer.
    if p.packed_input:
      assert segment_mask is not None, ('Need to specify segment_mask '
                                        'for packed input.')
      if p.has_aux_atten:
        assert aux_segment_mask is not None, ('Need to specify aux_segment_mask'
                                              'for packed input.')

    with tf.name_scope('self_atten'):
      atten_vec, self_atten_probs = self.self_atten.FProp(
          theta.self_atten,
          query_vec,
          None,
          paddings,
          segment_mask=segment_mask,
          per_step_padding_override=per_step_padding_override)
      atten_probs = py_utils.NestedMap(self_atten=self_atten_probs)

    if p.has_aux_atten:
      with tf.name_scope('aux_atten'):
        # Next the cross-attention layer.
        target_batch, target_time, dim = py_utils.GetShape(query_vec, 3)

        source_batch = self._GetSourceBatchSize(aux_vec)
        source_time = self._GetSourceLength(aux_vec)

        atten_vec = tf.reshape(atten_vec, [-1, source_batch, target_time, dim])
        atten_vec = tf.reshape(
            tf.transpose(atten_vec, [1, 0, 2, 3]), [source_batch, -1, dim])
        atten_vec, aux_atten_probs = self.cross_atten.FProp(
            theta.cross_atten,
            atten_vec,
            aux_vec,
            aux_paddings,
            segment_mask=aux_segment_mask,
            per_step_padding_override=aux_per_step_padding_override)
        num_heads = py_utils.GetShape(aux_atten_probs)[1]
        aux_atten_probs = tf.reshape(
            aux_atten_probs,
            [source_batch, -1, num_heads, target_time, source_time])
        aux_atten_probs = tf.transpose(aux_atten_probs, [1, 0, 2, 3, 4])
        aux_atten_probs = tf.reshape(
            aux_atten_probs,
            [target_batch, num_heads, target_time, source_time])
        atten_vec = tf.reshape(atten_vec, [source_batch, -1, target_time, dim])
        atten_vec = tf.transpose(atten_vec, [1, 0, 2, 3])
        atten_vec = tf.reshape(atten_vec, [target_batch, target_time, dim])
        atten_probs.aux_atten = aux_atten_probs

    # Finally the feed-forward layer.
    with tf.name_scope('fflayer'):
      return self.fflayer.FProp(theta.fflayer, atten_vec, paddings), atten_probs

  def InitStates(self, theta, target_batch_size, target_max_length):
    return self.self_atten.InitStates(theta.self_atten, target_batch_size,
                                      target_max_length)

  def ExtendStep(self,
                 theta,
                 query_vec,
                 aux_vec,
                 aux_paddings,
                 cached_states,
                 time_step,
                 use_short_seq_opt=False,
                 per_step_padding=None,
                 *,
                 compute_atten_probs=False,
                 segment_mask=None,
                 aux_segment_mask=None):
    """Transformer decoder layer, extend one step in autoregressive decoding.

    query_vec and aux_* may have different batch sizes, e.g., during a beam
    search. target_batch must be a multiple of source_batch and
    query_vec[i * batch_multiplier + j] corresponds to aux_vec[i], where
    batch_multiplier = target_batch / source_batch, 0 <= i < source_batch,
    0 <= j < batch_multiplier.

    WARNING: note the DIFFERENCE between FProp and ExtendStep:
    FProp:      target_batch = [batch_multiplier, batch]
    ExtendStep: target_batch = [batch, batch_multiplier]

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:    [target_batch, 1, dim].
      aux_vec:      [source_batch, source_time, dim]
      aux_paddings: [source_batch, source_time]
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. key   -
        [target_time, target_batch, num_heads, dim_per_head]. value -
        [target_time, target_batch, num_heads, dim_per_head].
      time_step: A scalar, the current decode step, 0-based.
      use_short_seq_opt: A bool, whether using short sequence optimization.
      per_step_padding: optional customized padding for this step -
        [target_batch, 1, target_time].
      compute_atten_probs: A bool, whether attention probabilities should be
        computed. If false, returns None for atten_probs.
      segment_mask: if not None, per step segment mask for this time step, of
        shape [target_batch, 1, target_time].
      aux_segment_mask: if not None, aux_segment_mask for this time step, of
        shape [target_batch, 1, source_time]

    Returns:
      (cur_output, atten_probs, updated_states)
      * cur_output: [target_batch, 1, dim]
      * atten_probs: [target_batch, num_heads, target_time=1, source_time] if
      compute_atten_probs is True and p.has_aux_atten=True. None otherwise.
      * updated_states: A `.NestedMap` object containing the updated states.
      key   - [target_time, target_batch, num_heads, dim_per_head].
      value - [target_time, target_batch, num_heads, dim_per_head].
    """
    target_batch, _, dim = py_utils.GetShape(query_vec, 3)
    query_vec = py_utils.HasShape(query_vec, [target_batch, 1, dim])

    # First the self-attention layer.
    atten_vec, updated_states = self.self_atten.ExtendStep(
        theta.self_atten,
        query_vec,
        cached_states,
        time_step,
        use_short_seq_opt,
        per_step_padding,
        segment_mask=segment_mask)

    atten_vec = py_utils.HasShape(atten_vec, [target_batch, 1, dim])
    cross_atten_probs = None
    if self.params.has_aux_atten:
      source_batch = self._GetSourceBatchSize(aux_vec)
      source_length = self._GetSourceLength(aux_vec)
      batch_multiplier = target_batch // source_batch
      # Next the cross-attention layer.
      if aux_segment_mask is not None:
        # change this into [b, 1, 1, src_len]
        aux_segment_mask = tf.expand_dims(aux_segment_mask, 2)
      atten_vec = tf.reshape(atten_vec, [source_batch, -1, dim])
      atten_vec, aux_atten_probs = self.cross_atten.FProp(
          theta.cross_atten,
          atten_vec,
          aux_vec,
          aux_paddings,
          segment_mask=aux_segment_mask)

      atten_vec = tf.reshape(atten_vec, [target_batch, 1, -1])
      if compute_atten_probs:
        cross_atten_probs = py_utils.HasShape(
            aux_atten_probs,
            # [source_batch, num_heads, batch_multiplier, source_length].
            [source_batch, -1, batch_multiplier, source_length])
        _, num_heads, _, _ = py_utils.GetShape(cross_atten_probs)
        # [source_batch, batch_multiplier, num_heads, source_length].
        cross_atten_probs = tf.transpose(cross_atten_probs, [0, 2, 1, 3])
        # Reshape to [target_batch, num_heads, 1, source_length].
        cross_atten_probs = tf.reshape(
            cross_atten_probs, [target_batch, num_heads, 1, source_length])

    # Finally the feed-forward layer.
    cur_output = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([target_batch, 1], dtype=atten_vec.dtype))
    return cur_output, cross_atten_probs, updated_states


# TODO(garrettaxel): Distribute the computation to downstream layers.
def TransformerFlops(inputs, num_heads, ff_dim, atten_dim, model_dim):
  """Compute FLOPs for Transformer layer without auxiliary attention.

    Attention Layer FLOPs (N = num attention heads, H = dim per head):
      q, k, v projections, incl bias: 3 x 'BTD,DNH->BTNH' -> 6*N*H*D*B*T
      logits: 'BTNH,BDNH->BNTD' -> (2*H-1)*N*B*T^2
      softmax: 5 ops per element in BNTD -> 5*N*D*B*T
      context: 'BNTD,BDNH->BTNH' -> (2*T-1)*N*H*B*T
      output proj: 'BTNH,DNH->BTD' -> (2*N-1)*(2*H-1)*D*B*T

    2 residuals FLOPs: 2*D*B*T
    1 FF layer FLOPs: 4*ff_hidden*D*B*T

  Args:
    inputs:    Input dimensions to the layer, [Batch, Time, Dim].
    num_heads: Number of attention heads for layer.
    ff_dim:    Feedforward hidden dimension.
    atten_dim: Attention hidden dimension.
    model_dim: Dimension of the model.

  Returns:
    Total FLOPs of the transformer layer.
  """
  f = tf.cast(ff_dim, tf.int64)
  a = tf.cast(atten_dim, tf.int64)
  n = tf.cast(num_heads, tf.int64)
  d = tf.cast(model_dim, tf.int64)
  h = tf.cast(a / n, tf.int64)  # dim per head
  inputs = tf.cast(inputs, tf.int64)
  b, t = inputs[0], inputs[1]
  multi_head_atten_flops = (6 * a * d + n * t * (2 * h - 1) + a * (2 * t - 1) +
                            5 * n * d + d * (2 * h - 1) * (2 * n - 1))
  residual_flops = 2 * d
  ff_flops = 4 * f * d
  return (multi_head_atten_flops + residual_flops + ff_flops) * b * t


class MultiSourceTransformerLayer(TransformerLayer):
  """Multi-source transformer layer with multiheaded attention.

  Multi-source attention is used for cross attention.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_source', 0, 'Number of sources to attend to.')
    p.Define(
        'primary_source_index', 0, 'Index for the primary source '
        'whose attention probabilities will be used as an output.')
    return p

  def __init__(self, params):
    assert issubclass(params.tr_atten_tpl.cls,
                      TransformerMultiSourceAttentionLayer)
    # Set up multi-source attention layer
    cross_atten_p = params.tr_atten_tpl
    cross_atten_p.num_source = params.num_source
    cross_atten_p.primary_source_index = params.primary_source_index
    assert params.tr_self_atten_tpl
    super().__init__(params)

  @property
  def primary_source_key(self):
    return 'source_%d' % self.params.primary_source_index

  def _GetSourceBatchSize(self, aux_vec):
    return py_utils.GetShape(aux_vec[self.primary_source_key], 2)[0]

  def _GetSourceLength(self, aux_vec):
    return py_utils.GetShape(aux_vec[self.primary_source_key], 2)[1]


# mt_attention_layer.MultiHeadedAttentionXL
ATTEN_TRANSFORMER_XL = 'transformer_xl'
# mt_attention_layer.MultiHeadedAttentionRPE
ATTEN_RPE = 'rpe'


def UseRelativeAttentionInTransformerLayer(transformer_params,
                                           rel_pos_emb_dim,
                                           atten_type=ATTEN_TRANSFORMER_XL):
  """Uses transformer-xl attention for self attention of a transformer layer.

  Args:
    transformer_params: A mt_attention_layer.TransformerLayer.Params() object.
    rel_pos_emb_dim: (int) Relative positional embedding dim to be set.
    atten_type: (string) Attention type. Supported:
      - 'transformer_xl': mt_attention_layer.MultiHeadedAttentionXL
      - 'rpe': mt_attention_layer.MultiHeadedAttentionRPE

  Returns:
    A mt_attention_layer.TransformerLayer.Params() object with relative pos emb.
  """
  if not issubclass(transformer_params.cls, TransformerLayer):
    raise ValueError('Unsupported input transformer layer: %s' %
                     transformer_params.cls)

  if atten_type not in (ATTEN_TRANSFORMER_XL, ATTEN_RPE):
    raise ValueError('Relative attention type: %s unsupported' % atten_type)

  # Gets multiheaded attention tpl from self attention config in transformer.
  trans_params_copy = transformer_params.Copy()
  if trans_params_copy.tr_self_atten_tpl is None:
    trans_params_copy.tr_self_atten_tpl = trans_params_copy.tr_atten_tpl.Copy()
  atten_tpl = trans_params_copy.tr_self_atten_tpl.atten_tpl

  # If already using relative attention class.
  if atten_tpl.cls in (MultiHeadedAttentionRPE, MultiHeadedAttentionXL,
                       LocalSelfAttentionXL):
    atten_tpl.rel_pos_emb_dim = rel_pos_emb_dim
    return trans_params_copy

  if atten_type == ATTEN_TRANSFORMER_XL:
    if atten_tpl.cls == MultiHeadedAttention:
      rel_atten_tpl = MultiHeadedAttentionXL.Params()
    elif atten_tpl.cls == LocalSelfAttention:
      rel_atten_tpl = (LocalSelfAttentionXL.Params())
    else:
      raise ValueError('Unsupported attention: %s' % atten_tpl.cls)
  elif atten_type == ATTEN_RPE:
    rel_atten_tpl = MultiHeadedAttentionRPE.Params()

  rel_atten_tpl = hyperparams.CopyFieldsTo(atten_tpl, rel_atten_tpl)
  rel_atten_tpl.rel_pos_emb_dim = rel_pos_emb_dim

  trans_params_copy.tr_self_atten_tpl.atten_tpl = rel_atten_tpl
  return trans_params_copy


def ClearRelativeAttentionInTransformerLayer(transformer_params):
  """Removes relative position attention in the transformer layer.

  Args:
    transformer_params: A mt_attention_layer.TransformerLayer param.

  Returns:
    A mt_attention_layer.TransformerLayer param without relative attention.
  """
  if not issubclass(transformer_params.cls, TransformerLayer):
    raise ValueError('Unsupported input transformer layer: %s' %
                     transformer_params.cls)
  trans_params_copy = transformer_params.Copy()
  if trans_params_copy.tr_self_atten_tpl is None:
    trans_params_copy.tr_self_atten_tpl = trans_params_copy.tr_atten_tpl.Copy()
  attention_tpl = trans_params_copy.tr_self_atten_tpl.atten_tpl
  if attention_tpl.cls == MultiHeadedAttentionXL:
    new_attention_tpl = MultiHeadedAttention.Params()
  elif attention_tpl.cls == (LocalSelfAttentionXL):
    new_attention_tpl = LocalSelfAttention.Params()
  else:
    raise ValueError('Unsupported attention params: %s' % attention_tpl.cls)

  new_attention_tpl = hyperparams.CopyFieldsTo(
      attention_tpl,
      new_attention_tpl,
      skip=['rel_pos_emb_dim', 'skip_term_b', 'pos_atten_logits_tpl'])
  trans_params_copy.tr_self_atten_tpl.atten_tpl = new_attention_tpl
  return trans_params_copy


class TransformerDecoderLayer(TransformerLayer):
  """Transformer decoder layer with multiheaded attention."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.has_aux_atten = True
    p.mask_self_atten = True
    return p


class MultiSourceTransformerDecoderLayer(MultiSourceTransformerLayer):
  """Multi-source transformer decoder layer with multiheaded attention."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.has_aux_atten = True
    p.mask_self_atten = True
    return p


class RepeatedTransformerLayer(repeat_layer.GenericRepeatLayer):
  """A stack of uniform TransformerLayer's as a RepeatLayer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.body = TransformerLayer.Params()
    p.Define(
        'atten_prob_aggregation', None,
        'None: return attention probabilities for each layer separately. '
        'mean: return the mean attention probabilities across layers.')
    return p

  def FProp(self, theta, query_vec, **kwargs):
    p = self.params
    assert not p.body.tr_atten_tpl.atten_dropout_prob
    assert not p.body.tr_atten_tpl.residual_dropout_prob
    assert not p.body.tr_fflayer_tpl.relu_dropout_prob
    assert not p.body.tr_fflayer_tpl.residual_dropout_prob
    with tf.name_scope(p.name):
      # iterative: query_vec
      # common_input: **kwargs
      # layerwise_output: atten_probs

      def _Fn(theta, *, common_input, layerwise_input, iterative):
        del layerwise_input
        layer_out, layer_atten_probs = self._body.FProp(
            theta.body, query_vec=iterative.query_vec, **common_input)
        return py_utils.NestedMap(
            iterative=py_utils.NestedMap(query_vec=layer_out),
            layerwise_output=py_utils.NestedMap(atten_probs=layer_atten_probs))

      repeat_results = self._Repeat(
          theta,
          _Fn,
          common_input=py_utils.NestedMap(kwargs),
          layerwise_inputs=py_utils.NestedMap(),
          iterative_input_0=py_utils.NestedMap(query_vec=query_vec))
      atten_probs = repeat_results.layerwise.atten_probs
      assert p.atten_prob_aggregation in (None, 'mean')
      if p.atten_prob_aggregation == 'mean':
        tf.logging.info('atten_probs=%s', atten_probs)
        atten_probs = tf.nest.map_structure(lambda x: tf.reduce_mean(x, axis=0),
                                            atten_probs)
      return repeat_results.iterative.query_vec, atten_probs

  def InitStates(self, theta, *args, **kwargs):
    # common_input: *args, **kwargs
    # layerwise_output: states

    def _Fn(theta, *, common_input, layerwise_input, iterative):
      del layerwise_input
      del iterative
      states = self._body.InitStates(theta.body, *common_input.args,
                                     **common_input.kwargs)
      return py_utils.NestedMap(
          iterative=py_utils.NestedMap(), layerwise_output=states)

    return self._Repeat(
        theta,
        _Fn,
        common_input=py_utils.NestedMap(args=list(args),
                                        kwargs=kwargs)).layerwise

  def ExtendStep(self, theta, query_vec, *, cached_states, **kwargs):
    p = self.params
    with tf.name_scope(p.name):
      # iterative: query_vec
      # common_input: **kwargs
      # layerwise_input: cached_states
      # layerwise_output: updated_state, atten_probs

      def _Fn(theta, *, common_input, layerwise_input, iterative):
        layer_out, layer_atten_probs, updated_states = self._body.ExtendStep(
            theta.body,
            query_vec=iterative.query_vec,
            cached_states=layerwise_input.cached_states,
            **common_input)
        return py_utils.NestedMap(
            iterative=py_utils.NestedMap(query_vec=layer_out),
            layerwise_output=py_utils.NestedMap(
                atten_probs=layer_atten_probs, updated_states=updated_states))

      repeat_results = self._Repeat(
          theta,
          _Fn,
          common_input=py_utils.NestedMap(kwargs),
          layerwise_inputs=py_utils.NestedMap(cached_states=cached_states),
          iterative_input_0=py_utils.NestedMap(query_vec=query_vec))
      atten_probs = repeat_results.layerwise.atten_probs
      assert p.atten_prob_aggregation in (None, 'mean')
      if p.atten_prob_aggregation == 'mean':
        atten_probs = tf.nest.map_structure(lambda x: tf.reduce_mean(x, axis=0),
                                            atten_probs)
      return (repeat_results.iterative.query_vec, atten_probs,
              repeat_results.layerwise.updated_states)


class StackedTransformerLayers(base_layer.BaseLayer):
  """A stack of Batch-Major Transformer layers."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('has_aux_atten', False,
             'If set, introduces a second attention layer')
    p.Define('mask_self_atten', False, 'If True, use masked self-attention.')
    p.Define('num_layers', 0, 'Num of layers in this stack.')
    p.Define(
        'mdl_dim', None,
        'Model dimension in Transformer layers. If None, one must set model '
        'dimension directly in transformer_layer_params_tpl. If set, p.mdl_dim '
        'will override transformer_layer_params_tpl.')
    p.Define(
        'hidden_dim', None,
        'The hidden layer dimension in Transformer layers. If None, one must '
        'set model dimension directly in transformer_layer_params_tpl. If set, '
        'p.hidden_dim will override transformer_layer_params_tpl.')
    p.Define(
        'num_atten_heads', None,
        'Num of attention heads. If None, one must set num_atten_heads '
        'directly in transformer_layer_params_tpl. If set, p.num_atten_heads '
        'will override transformer_layer_params_tpl.')
    p.Define(
        'dropout_prob', None,
        'Apply dropout at this prob at various places. If None, dropout values '
        'in transformer_layer_params_tpl will be used. If set, p.dropout_prob '
        'will override transformer_layer_params_tpl.')
    p.Define('add_unnormalized_input', True,
             'If set, uses unnormalized input in the residual add.')
    p.Define(
        'transformer_layer_params_tpl', TransformerLayer.Params(),
        'A template of TransformerLayer.params, can be a list of params '
        'of length equal to the num_layers or a factor of num_layers.'
        'For a factor, the params are tiled as [a, a, ..., b, b,...,].')
    p.Define('final_layer_norm', False,
             'If true, apply layer normalization to the final output.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm. ')
    p.Define(
        'layernorm_tpl', layers.LayerNorm.Params(), 'Template for the '
        'LayerNorm layers. use_fused_layernorm param above overrides the '
        'layernorm_tpl.use_fused_layernorm for compatibility.')
    p.Define(
        'splits', None, 'None or a list of layer indices. If None, all layers '
        'are placed on the same and only one partition. Else, len(splits) is '
        'the number of partitions the stack is sliced into. layer_i is placed '
        'on the kth partition (0-based) where split[k] < i <= split[k+1].')
    # MOE related params.
    p.Define('moe_layer_tpl',
             layers_with_attention.TransformerShardedMoeLayer.Params(),
             'Template configuration for the moe feedforward layer.')
    p.Define('num_experts', 0, 'Total number of experts.')
    p.Define('num_groups', 1, 'Num of groups for dispatching.')
    p.Define(
        'min_group_size', None,
        'If not None, num_groups will be adjusted so that there will be at '
        'least min_group_size tokens in each group.')
    p.Define('moe_layers', [], 'The list of MoE layer indices, e.g. [0, 2, 4].')
    return p

  def __init__(self, params):
    if params.splits:
      assert all(x <= params.num_layers - 1 for x in params.splits)
      # Assert p.splits is strictly monotonically increasing.
      assert sorted(list(set(params.splits))) == params.splits
    super().__init__(params)
    p = self.params

    assert p.num_layers > 0
    assert p.mdl_dim is None or p.mdl_dim > 0
    assert p.hidden_dim is None or p.hidden_dim > 0
    if not isinstance(p.num_atten_heads, list):
      assert p.num_atten_heads is None or p.num_atten_heads > 0
    else:
      for num_heads in p.num_atten_heads:
        assert num_heads > 0
    assert p.dropout_prob is None or 0.0 <= p.dropout_prob < 1.0

    if isinstance(p.transformer_layer_params_tpl, list):
      if p.num_layers % len(p.transformer_layer_params_tpl):
        raise ValueError('num_layers should be divisible by '
                         'transformer_layer_params_tpl')

    def _MoeLayerParams(ff_p):
      """Convert a Feedforward layer into an MOE layer."""
      assert issubclass(ff_p.cls,
                        layers_with_attention.TransformerFeedForwardLayer)
      assert p.num_experts > 0
      moe_p = p.moe_layer_tpl.Copy()
      # Copy over the base params.
      base_layer.BaseLayer.CopyBaseParams(ff_p, moe_p)
      # Set other params.
      moe_p.name = ff_p.name
      moe_p.input_dim = ff_p.input_dim
      moe_p.output_dim = ff_p.output_dim
      moe_p.hidden_dim = ff_p.hidden_dim
      moe_p.activation = ff_p.activation
      moe_p.residual_dropout_prob = ff_p.residual_dropout_prob
      moe_p.relu_dropout_prob = ff_p.relu_dropout_prob
      moe_p.dropout_tpl = ff_p.residual_dropout_tpl.Copy()
      moe_p.num_groups = p.num_groups
      moe_p.min_group_size = p.min_group_size
      moe_p.num_experts = p.num_experts
      # weight_split_dims_mapping and activation_split_dims_mapping should have
      # been set through p.moe_layer_tpl params.
      return moe_p

    def _LayerParams(ii):
      """Construct ii-th layer params."""
      if isinstance(p.transformer_layer_params_tpl, list):
        factor = p.num_layers // len(p.transformer_layer_params_tpl)
        i = ii // factor
        p_ii = p.transformer_layer_params_tpl[i].Copy()
      else:
        p_ii = p.transformer_layer_params_tpl.Copy()
      p_ii.name = 'layer_%d' % ii
      p_ii.has_aux_atten = p.has_aux_atten
      p_ii.mask_self_atten = p.mask_self_atten
      p_ii.input_dim = p.mdl_dim or p_ii.input_dim
      p_ii.output_dim = p.mdl_dim or p_ii.output_dim
      p_ii.packed_input = p.packed_input
      if (not isinstance(p_ii.tr_atten_tpl.num_heads, list) and
          p.num_atten_heads is not None):
        p_ii.tr_atten_tpl.num_heads = p.num_atten_heads
      if p.dropout_prob is not None:
        p_ii.tr_atten_tpl.atten_dropout_prob = p.dropout_prob
        p_ii.tr_atten_tpl.residual_dropout_prob = p.dropout_prob
        p_ii.tr_fflayer_tpl.residual_dropout_prob = p.dropout_prob
        p_ii.tr_fflayer_tpl.relu_dropout_prob = p.dropout_prob
      if p.hidden_dim is not None:
        p_ii.tr_fflayer_tpl.hidden_dim = p.hidden_dim
      p_ii.tr_atten_tpl.add_unnormalized_input = p.add_unnormalized_input
      if ii in p.moe_layers:
        p_ii.tr_fflayer_tpl = _MoeLayerParams(p_ii.tr_fflayer_tpl)
      return p_ii

    layer_params = [_LayerParams(ii) for ii in range(p.num_layers)]

    self.CreateChildren('x_layers', layer_params)

    if p.final_layer_norm:
      assert p.mdl_dim
      final_ln_p = p.layernorm_tpl.Copy().Set(
          input_dim=p.mdl_dim, use_fused_layernorm=p.use_fused_layernorm)
      self.CreateChild('final_ln', final_ln_p)

  @classmethod
  def GetSplitForLayer(cls, buckets, layer_index):
    assert layer_index <= buckets[-1], (
        f'layer_index:{layer_index} > buckets[-1]:{buckets[-1]}')
    #  Return index of the smallest element greater than or equal to layer_index
    return bisect.bisect_left(buckets, layer_index)

  def _GetDeviceOfLayer(self, layer_idx):
    """Get the device for a given layer index based on our params."""
    if not self.params.splits:
      return None
    return self.cluster.WorkerDeviceInModelSplit(
        self.GetSplitForLayer(self.params.splits, layer_idx))

  def FProp(self,
            theta,
            query_vec,
            paddings=None,
            aux_vec=None,
            aux_paddings=None,
            per_step_padding_override=None,
            segment_mask=None,
            aux_segment_mask=None,
            collect_per_layer_output=False):
    """Stacked Transformer layer.

    Args:
      theta: A `NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:      [batch, target_time, dim].
      paddings:       [batch, target_time].
      aux_vec:        [batch, source_time, dim].
      aux_paddings:   [batch, source_time].
      per_step_padding_override: A mask used by decoder self-attention to
        prevent information flow from future (causal padding). It has shape
        [batch, target_time, target_time] if not None.
      segment_mask:     [batch, 1, target_time, target_time]
      aux_segment_mask: [batch, 1, target_time, source_time]
      collect_per_layer_output: A Python bool, if to return per-xformer-layer
        output.

    Returns:
      (context, (optional)paddings, (optional)all_layer_outputs):
        - context vector:   [batch, target_time, dim]
        - paddings:         [batch, target_time], returned only if input
          paddings is not None.
        - all_layer_outputs: A list of xformer layer outputs, each has the same
          shape as context_vector.
    """
    p = self.params
    x_out = query_vec
    has_paddings = False if paddings is None else True
    if paddings is None:
      batch_size, seq_length, _ = py_utils.GetShape(query_vec)
      paddings = tf.zeros((batch_size, seq_length), dtype=query_vec.dtype)

    all_outs = []
    with tf.name_scope(p.name):
      for i in range(p.num_layers):
        x_in = x_out
        with tf.device(self._GetDeviceOfLayer(i)):
          x_out, _ = self.x_layers[i].FProp(
              theta.x_layers[i],
              x_in,
              paddings,
              aux_vec,
              aux_paddings,
              per_step_padding_override=per_step_padding_override,
              segment_mask=segment_mask,
              aux_segment_mask=aux_segment_mask)
          all_outs.append(x_out)
    if p.final_layer_norm:
      # Place on the last device.
      with tf.device(self._GetDeviceOfLayer(p.num_layers - 1)):
        x_out = self.final_ln.FProp(theta.final_ln, x_out)

    res = [x_out]
    if has_paddings:
      res += [paddings]
    if collect_per_layer_output:
      res += [all_outs]

    if len(res) == 1:
      return res[0]
    return tuple(res)

  def InitStates(self, theta, *args, **kwargs):
    return py_utils.NestedMap(x_layers=[
        layer.InitStates(layer_theta, *args, **kwargs)
        for layer, layer_theta in zip(self.x_layers, theta.x_layers)
    ])

  def ExtendStep(self,
                 theta,
                 query_vec,
                 aux_vec,
                 aux_paddings,
                 cached_states,
                 time_step,
                 use_short_seq_opt=False,
                 **kwargs):
    """Transformer decoder layer, extend one step in autoregressive decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:    [target_batch, 1, dim].
      aux_vec:      [source_batch, source_time, dim]
      aux_paddings: [source_batch, source_time]
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding.
        cached_states.x_layers is a list corresponding to self.x_layers, where
        each element is a NestedMap with attention keys and values:
        - key: [target_time, target_batch, num_heads, dim_per_head].
        - value: [target_time, target_batch, num_heads, dim_per_head].
      time_step: A scalar, the current decode step, 0-based.
      use_short_seq_opt: A bool, whether using short sequence optimization.
      **kwargs: additional kwargs for TransformerLayer.ExtendStep.

    Returns:
      cur_output: The last decoder layer output of shape [target_batch, 1, dim].
      updated_states: A `.NestedMap` object containing the updated states.
      updated_states.x_layers is a list corresponding to self.x_layers, where
      each element is a NestedMap with attention keys and values:

      - key: [target_time, target_batch, num_heads, dim_per_head].
      - value: [target_time, target_batch, num_heads, dim_per_head].
    """
    p = self.params
    with tf.name_scope(p.name):
      updated_states = py_utils.NestedMap(x_layers=[])
      decoder_input = query_vec
      for layer, layer_theta, layer_states in zip(self.x_layers, theta.x_layers,
                                                  cached_states.x_layers):
        decoder_output, _, updated_layer_states = layer.ExtendStep(
            layer_theta, decoder_input, aux_vec, aux_paddings, layer_states,
            time_step, use_short_seq_opt, **kwargs)
        updated_states.x_layers.append(updated_layer_states)
        decoder_input = decoder_output

      if p.final_layer_norm:
        decoder_output = self.final_ln.FProp(theta.final_ln, decoder_output)
    return decoder_output, updated_states


class PipelinedTransformerLayers(base_layer.BaseLayer):
  """Pipelined layers of StackedTransformerLayers."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('pipeline_stage', StackedTransformerLayers.Params(),
             'The layer params of each stage.')
    p.Define('num_pipeline_stages', None, 'Number of pipeline stages.')
    p.Define('num_pipeline_microbatches', None,
             'Number of pipeline microbatches.')
    p.Define('pipeline_microbatch_size', None,
             'Size of each pipeline microbatch.')
    p.Define('shard_stages_1d', False,
             'Whether to use 1D sharding on pipeline stages.')
    p.Define('final_layer_norm', False,
             'Whether to add a layer norm after all stages.')
    p.Define(
        'final_ln_at_each_stage', False,
        'If True, a layer norm will be added to the end of each stage instead '
        'of the end of the whole pipeline. This is to support legacy '
        'behaviors. Set to False for all new use cases.')
    p.Define(
        'circular_repeat', 1,
        'If > 1, it enables circular pipeline, and this is the number of '
        'repeats for each stage.')
    p.Define(
        'pipeline_stage_mesh_dim', None,
        'The mesh dimension to shard the pipeline stage dimension. Set '
        'this only when shard_stages_1d is False.')
    p.Define('unroll', 'never', 'Unroll the layers: never, eval_only, always.')
    return p

  class WrappedStageClass:
    """Wrapper of the stage to fix argument order for pipelining."""

    def __init__(self, stage):
      self._stage = stage

    def __getattr__(self, attr):
      return getattr(self._stage, attr)

    def ExtendStep(self, theta, query_vec, cached_states, **kwargs):
      # LayerwiseShardablePipelinedLayer requires the outputs to have the same
      # structure as the positional arguments, which only include query_vec and
      # cached_states.
      return self._stage.ExtendStep(
          theta, query_vec, cached_states=cached_states, **kwargs)

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.num_pipeline_stages > 0
    # Use deterministic droupout in pipelined layers.
    stage_params = p.pipeline_stage.Copy()
    layer_params = stage_params.transformer_layer_params_tpl
    layer_params.tr_atten_tpl.dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    layer_params.tr_atten_tpl.atten_tpl.dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    if layer_params.tr_self_atten_tpl is not None:
      layer_params.tr_self_atten_tpl.dropout_tpl = (
          layers.DeterministicDropoutLayer.Params())
      layer_params.tr_self_atten_tpl.atten_tpl.dropout_tpl = (
          layers.DeterministicDropoutLayer.Params())
    layer_params.tr_fflayer_tpl.residual_dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    layer_params.tr_fflayer_tpl.fflayer_tpl.dropout = (
        layers.DeterministicDropoutLayer.Params())
    stage_params.final_layer_norm = p.final_ln_at_each_stage
    pipeline_params = gshard_layers.LayerwiseShardablePipelinedLayer.Params(
    ).Set(
        name=p.name,
        num_stages=p.num_pipeline_stages,
        single_stage_body=stage_params,
        num_microbatches=p.num_pipeline_microbatches,
        microbatch_size=p.pipeline_microbatch_size,
        shard_stages_1d=p.shard_stages_1d,
        per_stage_vars=False,
        circular_repeat=p.circular_repeat,
        pipeline_stage_mesh_dim=p.pipeline_stage_mesh_dim,
        unroll=p.unroll)
    self.CreateChild('pipeline', pipeline_params)
    self.pipeline.body = self.WrappedStageClass(self.pipeline.body)

    if p.final_layer_norm and not p.final_ln_at_each_stage:
      # Create the final LN layer.
      assert p.pipeline_stage.mdl_dim
      final_ln_p = p.pipeline_stage.layernorm_tpl.Copy().Set(
          input_dim=p.pipeline_stage.mdl_dim,
          use_fused_layernorm=p.pipeline_stage.use_fused_layernorm)
      self.CreateChild('final_ln', final_ln_p)

    if not p.final_ln_at_each_stage:
      p.fprop_dtype = layer_params.fprop_dtype
    else:
      p.fprop_dtype = p.pipeline_stage.layernorm_tpl.fprop_dtype

  def FProp(self, theta, *args, **kwargs):
    p = self.params
    args = self._CastToFPropDtype(args)
    out = self.pipeline.FProp(theta.pipeline, *args, **kwargs)
    if not (p.final_layer_norm and not p.final_ln_at_each_stage):
      return out

    has_paddings = isinstance(out, tuple) and len(out) == 2
    if has_paddings:
      x_out, padding = out
      return self.final_ln.FProp(theta.final_ln, x_out), padding
    else:
      return self.final_ln.FProp(theta.final_ln, out)

  def InitStates(self, theta, *args, **kwargs):
    p = self.params
    per_stage, _ = self.pipeline.BodyFPropNoMicrobatching(
        theta.pipeline.body, 'InitStates', *args, **kwargs)

    def _TransposeStageBatch(x):
      """Microbatches the state and applying padding."""
      # [num_stages, t, b, ...]
      shape = py_utils.GetShape(x)
      if p.num_pipeline_microbatches is not None:
        assert shape[2] % p.num_pipeline_microbatches == 0
        mb = p.num_pipeline_microbatches
      else:
        assert shape[2] % p.pipeline_microbatch_size == 0
        mb = shape[2] // p.pipeline_microbatch_size
      # [num_stages, t, mb_size, mb, ...]
      x = tf.reshape(x, shape[:2] + [shape[2] // mb, mb] + shape[3:])
      # [mb, num_stages, t, mb_size, ...]
      perm = [3, 0, 1, 2] + [i + 4 for i in range(len(shape) - 3)]
      x = tf.transpose(x, perm)
      return self.pipeline.PadMicrobatches(x)

    return tf.nest.map_structure(_TransposeStageBatch, per_stage)

  def ExtendStep(self,
                 theta,
                 query_vec,
                 aux_vec,
                 aux_paddings,
                 cached_states,
                 time_step,
                 use_short_seq_opt=False,
                 **kwargs):
    p = self.params
    query_vec = self._CastToFPropDtype(query_vec)
    # Fix argument order for LayerwiseShardablePipelinedLayer.
    decoder_output, updated_states = self.pipeline.FPropFn(
        theta.pipeline,
        'ExtendStep',
        query_vec,
        padded_per_stage_states=[cached_states],
        kwargs_no_batch={'time_step': time_step},
        aux_vec=aux_vec,
        aux_paddings=aux_paddings,
        use_short_seq_opt=use_short_seq_opt,
        **kwargs)

    if p.final_layer_norm and not p.final_ln_at_each_stage:
      decoder_output = self.final_ln.FProp(theta.final_ln, decoder_output)
    return decoder_output, updated_states


class TransformerFeedForwardLayerWithTaskId(
    layers_with_attention.TransformerFeedForwardLayer):
  """TransformerFeedForwardLayer with optional task_id input args."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('use_task_ids', False,
             'If set, introduces a second attention layer')
    return p

  def FProp(self, theta, inputs, paddings, task_id=None):
    """Feed-forward, residual and layer-norm.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: [batch, time, dim].
      paddings: [batch, time]
      task_id: optional task_id with shape [batch]

    Returns:
      tensor of the same shape with inputs
    """
    p = self.params
    if p.use_task_ids:
      if task_id is None:
        raise ValueError('Must pass task_id if use_task_ids.')
    inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
    if hasattr(self, 'res_proj_layer'):
      inputs = self.res_proj_layer.FProp(theta.res_proj_layer, inputs)
    expanded_paddings = tf.expand_dims(paddings, -1)
    fflayer_args = [inputs_normalized, expanded_paddings]
    fflayer_args += [task_id] if p.use_task_ids else []
    h = inputs + self.residual_dropout.FProp(
        theta.residual_dropout, self.fflayer.FProp(theta.fflayer, *
                                                   fflayer_args))
    return h


class GPipeBatchMajorTransformerLayer(TransformerLayer):
  """GPipe compatible batch majortransformer layer.

  To be used with the new GPipeBatchMajorStack.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('ln_tpl', layers.LayerNorm.Params(),
             'Layer norm default params. No layernorm if set to None.')
    p.Define('output_layer_norm', False,
             'Whether to layer normalize the output of the layer.')
    return p

  def __init__(self, params):
    # Initialize output layer norm.
    super().__init__(params)
    p = self.params
    if p.output_layer_norm:
      params = p.ln_tpl.Copy()
      params.name = 'output_ln'
      params.input_dim = p.input_dim
      self.CreateChild('layer_norm', params)

  def FProp(self, theta, source_vecs, source_paddings, target_vecs,
            target_paddings, encoder_self_atten_segment_mask,
            decoder_self_atten_segment_mask, decoder_cross_atten_segment_mask):
    p = self.params
    with tf.name_scope(p.name):
      if p.has_aux_atten:  # Decoder FProp
        sa_mask, ca_mask = None, None
        if p.packed_input:
          # This computation doesn't behave nicely when outside
          # recurrent.Recurrent resulting in nans for splits > 1
          min_val = GetDtypeMin(decoder_self_atten_segment_mask.dtype)
          # Operator overloading with * produces type-errors when running on
          # borg with splits > 1.
          sa_mask = tf.math.multiply(min_val, decoder_self_atten_segment_mask)
          ca_mask = tf.math.multiply(min_val, decoder_cross_atten_segment_mask)
        atten_vec, _ = self.self_atten.FProp(
            theta.self_atten,
            target_vecs,
            None,
            target_paddings,
            segment_mask=sa_mask)
        atten_vec, _ = self.cross_atten.FProp(
            theta.cross_atten,
            atten_vec,
            source_vecs,
            source_paddings,
            segment_mask=ca_mask)
        atten_vec = self.fflayer.FProp(theta.fflayer, atten_vec,
                                       target_paddings)
        atten_vec.set_shape(target_vecs.shape)
        if p.output_layer_norm:
          atten_vec = self.layer_norm.FProp(theta.layer_norm, atten_vec)
        return (source_vecs, source_paddings, atten_vec, target_paddings,
                encoder_self_atten_segment_mask,
                decoder_self_atten_segment_mask,
                decoder_cross_atten_segment_mask)

      # Encoder FProp
      sa_mask = None
      if p.packed_input:
        min_val = GetDtypeMin(encoder_self_atten_segment_mask.dtype)
        sa_mask = tf.math.multiply(min_val, encoder_self_atten_segment_mask)
      atten_vec, _ = self.self_atten.FProp(
          theta.self_atten,
          source_vecs,
          None,
          source_paddings,
          segment_mask=sa_mask)
      atten_vec = self.fflayer.FProp(theta.fflayer, atten_vec, source_paddings)
      atten_vec.set_shape(source_vecs.shape)
      if p.output_layer_norm:
        atten_vec = self.layer_norm.FProp(theta.layer_norm, atten_vec)

      return (atten_vec, source_paddings, target_vecs, target_paddings,
              encoder_self_atten_segment_mask, decoder_self_atten_segment_mask,
              decoder_cross_atten_segment_mask)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    py_utils.CheckShapes((inputs,))
    flops_per_element = 5
    source_batch, src_time, dim = inputs
    flops = flops_per_element * src_time * src_time * source_batch * dim
    args = args if isinstance(args, tuple) else (args,)
    return py_utils.NestedMap(flops=flops, out_shapes=(inputs,) + args)

  @classmethod
  def SetupDeterministicDropout(cls, params):
    """Replaced dropout layers in transformer with deterministic ones."""
    params.tr_atten_tpl.dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    params.tr_atten_tpl.atten_tpl.dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    params.tr_fflayer_tpl.residual_dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    params.tr_fflayer_tpl.fflayer_tpl.dropout = (
        layers.DeterministicDropoutLayer.Params())
    return params

  def ExtendStep(self,
                 theta,
                 query_vec,
                 aux_vec,
                 aux_paddings,
                 cached_states,
                 time_step,
                 use_short_seq_opt=False):
    """Transformer decoder layer, extend one step in autoregressive decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:    [target_batch, 1, dim].
      aux_vec:      [source_batch, source_time, dim]
      aux_paddings: [source_batch, source_time]
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. key   -
        [target_time, target_batch, num_heads, dim_per_head]. value -
        [target_time, target_batch, num_heads, dim_per_head].
      time_step: A scalar, the current decode step, 0-based.
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      cur_output: [target_batch, 1, dim]
      updated_states: A `.NestedMap` object containing the updated states.
      key   - [target_time, target_batch, num_heads, dim_per_head].
      value - [target_time, target_batch, num_heads, dim_per_head].
    """
    target_batch, _, dim = py_utils.GetShape(query_vec, 3)
    source_batch = py_utils.GetShape(aux_vec)[0]

    # First the self-attention layer.
    atten_vec, updated_states = self.self_atten.ExtendStep(
        theta.self_atten, query_vec, cached_states, time_step,
        use_short_seq_opt)

    # Next the cross-attention layer.
    if self.params.has_aux_atten:  # Decoder FProp
      atten_vec = tf.reshape(atten_vec, [source_batch, -1, dim])
      atten_vec, cross_atten_probs = self.cross_atten.FProp(
          theta.cross_atten, atten_vec, aux_vec, aux_paddings)
      atten_vec = tf.reshape(atten_vec, [target_batch, 1, -1])
    else:
      cross_atten_probs = None

    # Finally the feed-forward layer.
    cur_output = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([target_batch, 1], dtype=atten_vec.dtype))

    if self.params.output_layer_norm:
      cur_output = self.layer_norm.FProp(theta.layer_norm, cur_output)
    return cur_output, cross_atten_probs, updated_states


class ResidualAddLayer(base_layer.BaseLayer):
  """A layer to add inputs with residual weight."""

  @classmethod
  def Params(cls):
    """Params for `ResidualAddLayer`."""
    p = super().Params()
    p.Define('residual_weight', 1.0, 'Residual weight.')
    p.Define(
        'apply_residual', True, 'If set False, input is not added, decay '
        'to Id layer if residual_weight is 1.')
    return p

  def FProp(self, theta, x, y):
    """Return combined inputs.

    Args:
      theta: weights defined in this layer.
      x: input tensor.
      y: input tensor to apply weight to.

    Returns:
      Added tensors.
    """
    p = self.params
    if p.apply_residual:
      return x + p.residual_weight * y
    else:
      return p.residual_weight * y

  @classmethod
  def FPropMeta(cls, p, x, y):
    py_utils.CheckShapes((x, y))
    return py_utils.NestedMap(flops=x.num_elements() * 2, out_shapes=(x,))


class PaddingLayer(base_layer.BaseLayer):
  """A layer that applies paddings to the inputs."""

  def FProp(self, theta, inputs, paddings):
    """Return combined inputs.

    Args:
      theta: weights defined in this layer.
      inputs: input tensor.
      paddings: paddings tensor, should be of shape tf.shape(inputs)[:-1].

    Returns:
      Tensor with paddings applied.
    """
    paddings = tf.expand_dims(paddings, -1)
    if inputs.shape.ndims is not None and paddings.shape.ndims is not None:
      for _ in range(py_utils.GetRank(inputs) - py_utils.GetRank(paddings)):
        paddings = tf.expand_dims(paddings, -1)
    return py_utils.ApplyPadding(paddings, inputs)

  @classmethod
  def FPropMeta(cls, p, inputs, paddings):
    py_utils.CheckShapes((inputs, paddings))
    return py_utils.NestedMap(
        flops=max(inputs.num_elements(), paddings.num_elements()) * 2,
        out_shapes=(inputs,))


class StrideLayer(base_layer.BaseLayer):
  """A layer that does stride."""

  @classmethod
  def Params(cls):
    """Params for `StrideLayer`."""
    p = super().Params()
    p.Define(
        'stride', 0, 'To use every k-th token, set the stride to k. When '
        'stride == 0, only returns the first token of the input. When '
        'stride == 1, returns every token in the input.')
    p.Define(
        'first_n', None, 'only considers the first N tokens for the '
        'output. We use [:first_n:stride] to select the output tokens. If '
        'first_n is None, this flag is a no-op. If stride is positive, the'
        ' output sequence length is "(first_n-1) // stride + 1". If stride'
        ' is 0, first_n has to be None or 1. first_n ca not be 0. If '
        'first_n <= stride, only the first token is used.')
    p.Define('axis', 1, 'The axis to apply striding.')
    return p

  def FProp(self, theta, x):
    """Applies stride to the inputs.

    Args:
      theta: weights defined in this layer.
      x: input tensor, [..., time, ...]. Stride is applied to the time dim as
        given by p.axis.

    Returns:
      Strided tensor, with the stride applied to the time dim in x.
    """
    p = self.params
    assert p.first_n is None or p.first_n > 0
    assert p.axis in (1, 2, 3)

    stride, first_n = p.stride, p.first_n  # x[:None:k] == x[::k]
    if p.stride == 0:
      assert p.first_n is None or p.first_n == 1
      first_n = 1  # x[:k:1]  == x[:k]
      stride = 1

    if p.axis == 1:
      return x[:, :first_n:stride]
    elif p.axis == 2:
      return x[:, :, :first_n:stride]
    elif p.axis == 3:
      return x[:, :, :, :first_n:stride]
    else:
      return None

  @classmethod
  def FPropMeta(cls, p, x):
    assert p.first_n is None or p.first_n > 0
    assert p.axis in (1, 2, 3)
    py_utils.CheckShapes((x,))
    stride, first_n = p.stride, p.first_n
    if stride == 0:
      stride, first_n = 1, 1
    if first_n is None:
      first_n = x[p.axis]

    out_seq_len = (first_n - 1) // stride + 1
    return py_utils.NestedMap(
        flops=1,
        out_shapes=(tshape.Shape(x[0:p.axis] + [out_seq_len] +
                                 x[p.axis + 1:]),))


class FunnelPoolingLayer(StrideLayer):
  """A layer that does pooling in Funnel-Transformer.

    https://arxiv.org/pdf/2006.03236.pdf section 2.2. for query-only pooling and
    section A.1 for begin_intact & trunc_seq.
  """

  @classmethod
  def Params(cls):
    """Params for `FunnelPoolingLayer`."""
    p = super().Params()
    p.Define(
        'begin_intact', 0,
        'Number of starting tokens which we do not apply pooling to, i.e. '
        'y = concat([x[:, :begin_intact], pool(x[:, begin_intact:])])')
    p.Define(
        'trunc_seq', True,
        'Truncate ending tokens of the sequence when `begin_intact > 0` for '
        'TPU efficiency.')
    p.Define('pool_window', None, 'Size of the pooling window.')
    p.Define('pooling_type', 'AVG', 'Pooling type: MAX|AVG')
    p.Define(
        'padding_algorithm', 'SAME',
        'Padding algorithm. See the "returns" section of '
        '`tf.nn.convolution` for details. '
        'Roughly, VALID = NO_PADDING and SAME (default) = PAD INPUT')
    # TODO(zihangd): remove this option after verifying this change does not
    # harm existing results.
    p.Define(
        'exclude_pad_effect', True,
        'Ignore the padded values when applying MAX|AVG pooling to recover the '
        'case where there is no padding at all. Specifically, for MAX pooling, '
        'values of padding positions are set to dtype.min. For AVG pooling, '
        'the padding positions are set to 0 and then the token count in the '
        'corresponding window is reduced accordingly.'
        'This is a temporary option for back compatibility as the old '
        'implementation does not consider this.')
    return p

  def FProp(
      self,
      theta: py_utils.NestedMap,
      inputs: tf.Tensor,
      paddings: Optional[tf.Tensor] = None,
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Applies pooling to the inputs.

    Args:
      theta: weights defined in this layer.
      inputs: input tensor of shape [batch, time, dim], where the pooling is
        applied to the time dim.
      paddings: None or padding tensor of shape [batch, time]. If not None, the
        striding will be applied to the time dim.

    Returns:
      An (output, paddings) tensor tuple if paddings is not None, else just
      output, with the pooling/striding applied to the time dimension.
    """
    # Notes:
    # - The output shape is [batch, out_len, dim], where:
    #     inp_len = x.shape[2] - p.first_n
    #     if begin_intact == 0:
    #       out_len = inp_len / stride
    #     else:
    #       if not trunc_seq:
    #         out_len = begin_intact + (inp_len - begin_intact) / stride
    #       else:
    #         pooled_time = begin_intact + pool(
    #             inp_len - begin_intact - num_trunc, stride)
    # - How to compute `num_trunc`:
    #     Truncate last tokens of x[:, begin_intact:] such that `len_a == len_b`
    #       len_a = inp_len / stride (if begin_intact = 0)
    #       len_b = begin_intact + (inp_len - begin_intact - num_trunc) / stride
    #           (if begin_intact > 0 and trunc_seq)
    #     Solve the equality `len_a == len_b`, we get:
    #       num_trunc = stride * begin_intact - begin_intact

    p = self.params
    assert p.first_n is None or p.first_n > 0
    assert p.stride >= 0
    if p.axis != 1:
      raise ValueError('FunnelPoolingLayer only supports axis = 1 but got '
                       '%d' % (p.axis))

    # stride == 0
    if p.stride == 0:
      assert p.first_n is None or p.first_n == 1
      pooled_tensor = inputs[:, :1]
      pooled_paddings = paddings[:, :1] if paddings is not None else None
      if pooled_paddings is not None:
        return pooled_tensor, pooled_paddings
      return pooled_tensor

    if p.first_n:
      inputs = inputs[:, :p.first_n]
      paddings = paddings[:, :p.first_n] if paddings is not None else None

    # stride == 1
    if p.stride == 1:
      if paddings is not None:
        return inputs, paddings
      return inputs

    # stride > 1
    if p.begin_intact > 0:
      intact_inputs = inputs[:, :p.begin_intact]
      intact_paddings = (
          paddings[:, :p.begin_intact] if paddings is not None else None)
      if p.trunc_seq:
        num_trunc = p.begin_intact * p.stride - p.begin_intact
        inputs = inputs[:, p.begin_intact:-num_trunc]
        paddings = (
            paddings[:, p.begin_intact:-num_trunc]
            if paddings is not None else None)
      else:
        inputs = inputs[:, p.begin_intact:]
        paddings = (
            paddings[:, p.begin_intact:] if paddings is not None else None)

    if paddings is not None and p.exclude_pad_effect:
      if p.pooling_type == 'MAX':
        # Fill dtype.min in padded positions.
        min_value = tf.ones_like(inputs) * p.dtype.min
        inputs = py_utils.ApplyPadding(paddings[..., tf.newaxis], inputs,
                                       min_value)
      elif p.pooling_type == 'AVG':
        # Fill 0 in padded positions.
        inputs = py_utils.ApplyPadding(paddings[..., tf.newaxis], inputs,
                                       tf.zeros_like(inputs))

    pool_window = p.pool_window or p.stride
    pooled_tensor = tf.nn.pool(
        inputs,
        window_shape=[pool_window],
        pooling_type=p.pooling_type,
        strides=[p.stride],
        padding=p.padding_algorithm)

    if (paddings is not None and p.pooling_type == 'AVG' and
        p.exclude_pad_effect):
      # Count the fraction of non-padding elements inside each pooling window.
      in_mask = tf.cast(1.0 - paddings, dtype=pooled_tensor.dtype)
      non_padding_ratio = tf.nn.pool(
          in_mask[:, :, tf.newaxis],
          window_shape=[pool_window],
          pooling_type='AVG',
          strides=[p.stride],
          padding=p.padding_algorithm)
      # Divide by non-padding ratios to eliminate the effect of padded values.
      non_padding_ratio = tf.broadcast_to(non_padding_ratio,
                                          tf.shape(pooled_tensor))
      pooled_tensor = py_utils.DivideNoNan(pooled_tensor, non_padding_ratio)

    pooled_paddings = (
        paddings[:, ::p.stride] if paddings is not None else None)

    if p.begin_intact > 0:
      pooled_tensor = tf.concat([intact_inputs, pooled_tensor],
                                axis=1,
                                name='concat_intact_pooled_tensor')
      if pooled_paddings is not None:
        pooled_paddings = tf.concat([intact_paddings, pooled_paddings],
                                    axis=1,
                                    name='concat_intact_pooled_paddings')

    # Set padding values to 0. If not set, in the case of max pooling, padded
    # values will be dtype.min, which can cause numerical instability.
    if pooled_paddings is not None:
      pooled_tensor *= tf.cast(
          tf.expand_dims(1.0 - pooled_paddings, -1), pooled_tensor.dtype)
      return pooled_tensor, pooled_paddings
    return pooled_tensor

  @classmethod
  def FPropMeta(cls, p, x, paddings=None):
    """See base class."""
    assert p.first_n is None or p.first_n > 0
    assert p.axis in (1, 2, 3)
    py_utils.CheckShapes((x,))
    stride, first_n = p.stride, p.first_n
    if stride == 0:
      stride, first_n = 1, 1
    if first_n is None:
      first_n = x[p.axis]

    out_seq_len = (first_n - 1) // stride + 1
    out_x_shape = tshape.Shape(x[0:p.axis] + [out_seq_len] + x[p.axis + 1:])
    if paddings is None:
      return py_utils.NestedMap(flops=1, out_shapes=(out_x_shape,))

    out_paddings = tshape.Shape(paddings[0:p.axis] + [out_seq_len])
    return py_utils.NestedMap(flops=1, out_shapes=(out_x_shape, out_paddings))


class FunnelUpsampleLayer(base_layer.BaseLayer):
  """A layer that does upsampling in Funnel-Transformer."""

  @classmethod
  def Params(cls):
    """Params for `FunnelUpsampleLayer`."""
    p = super().Params()
    p.Define(
        'hidden_dim', 0,
        'The static size of 3rd dimension of both the input and output, '
        'which is usually the same as the Transformer hidden dimension '
        'when used together. This will be used for the shape of weights '
        'for DECONV upsampling.')
    p.Define('upsample_rate', 1,
             'The length multiplier for the upsampled sequence.')
    p.Define(
        'begin_intact', 0,
        'Number of starting tokens which we do not upsample. This value '
        'should be the same as the `begin_intact` used in the Funnel '
        'pooling layers to ensure correctness.')
    p.Define(
        'trunc_seq', True,
        'Truncate sequence for efficiency. This is only effective when '
        '`begin_intact > 0`')
    p.Define('upsample_type', 'REPEAT', 'upsample type: REPEAT|DECONV')
    return p

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    if p.upsample_type == 'DECONV':
      pc = py_utils.WeightParams(
          shape=[p.hidden_dim, p.upsample_rate, p.hidden_dim],
          init=py_utils.WeightInit.Gaussian(1.0 / math.sqrt(p.hidden_dim)),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      self.CreateVariable('weight', pc)

  def FProp(self, theta, x):
    """Upsample to the inputs.

    Args:
      theta: weights defined in this layer.
      x: input tensor, [batch, time, dim] upsampling is applied to the time dim.

    Returns:
      Upsampled tensor, with the upsampling applied to the second dim in x.
    """
    p = self.params
    if x.shape.ndims != 3:
      raise ValueError('FunnelUpsampleLayer expects input to be rank 3, but '
                       'got %d' % (x.shape.ndims))
    if p.upsample_type not in ['REPEAT', 'DECONV']:
      raise ValueError('Only supports upsample_type REPEAT and DECONV, but '
                       'got %s' % (p.upsample_type))

    assert isinstance(p.upsample_rate, int)
    if p.upsample_rate == 1:
      return x

    if p.begin_intact > 0:
      intact = x[:, :p.begin_intact]
      hid = x[:, p.begin_intact:]
    else:
      hid = x

    if p.upsample_type == 'REPEAT':
      upsampled = tf.repeat(hid, repeats=p.upsample_rate, axis=1)
    elif p.upsample_type == 'DECONV':
      upsampled = tf.einsum('BLD,DNH->BLNH', hid, theta.weight)
      upsampled = tf.reshape(
          upsampled,
          [hid.shape[0], p.upsample_rate * hid.shape[1], p.hidden_dim])

    if p.begin_intact > 0:
      sep_len = 1
      if p.trunc_seq:
        num_pad = p.begin_intact * p.upsample_rate - p.begin_intact
        upsampled = tf.pad(upsampled, [[0, 0], [0, num_pad], [0, 0]])
      else:
        upsampled = upsampled[:, :-sep_len]
      upsampled = tf.concat([intact, upsampled],
                            axis=1,
                            name='concat_upsampled')

    return upsampled


class MeshSplitLayer(base_layer.BaseLayer):
  """A layer that applies SPMD MeshSplit annotation to a tensor."""

  @classmethod
  def Params(cls):
    """Params for `MeshSplitLayer`."""
    p = super().Params()
    p.Define(
        'tensor_split_dims_mapping', None,
        'A list of integers that map each tensor axis to the device mesh axis '
        'along which it is sharded.')
    return p

  def FProp(self, theta, x):
    """Returns x with SPMD MeshSplit annotation.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      x: Tensor to annotate.

    Returns:
      The tensor with annotation applied.
    """
    p = self.params
    return gshard_utils.MeshSplit(x, p.device_mesh, p.tensor_split_dims_mapping)

  @classmethod
  def FPropMeta(cls, p, x):
    return py_utils.NestedMap(flops=0, out_shapes=(x,))


# pyformat: disable
class Builder(builder.Base):
  """Builder for self-attention layers."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('model_dim', 4, 'Model dim of this layer.')
    p.Define('num_heads', 1, 'Number of heads in the atten layer.')
    p.Define('ff_hidden_dim', 4, 'Hidden dim of the feedforward layer')
    p.Define('attention_hidden_dim', None,
             'Hidden dim of the attention layer.')
    p.Define('residual_dropout_prob', 0,
             'Dropout prob to the output of each sub-layer before it is added '
             'to the sub-layer input.')
    p.Define('ff_activation_fn', 'RELU',
             'Activation function in Feedforward layer.')
    p.Define('ff_residual_weight', 1.0, 'Weight given to F(x) in the residual '
             'connection: y = x + ff_residual_weight * F(x), in Feedforward '
             'layer.')
    p.Define('ff_apply_residual', True, 'If true,  '
             'y = x + ff_residual_weight * F(x) in Feedforward layer, else '
             'y = ff_residual_weight * F(x). This is experimental and could be '
             'removed in the future. See b/174568214.')
    p.Define('atten_apply_residual', True, 'If true,  '
             'y = x + F(x) in attention layer, else y = F(x). This is '
             'experimental and could be removed in the future. See '
             'b/174568214.')
    p.Define('relu_dropout_prob', 0,
             'Probability at which we apply dropout to the hidden layer of '
             'feed-forward network.')
    p.Define('atten_dropout_prob', 0,
             'Probability at which we apply dropout to the attention layer')
    p.Define('selfatten_add_unnormalized_input', True,
             'Whether to use unnormalized input in the residual add.')
    p.Define('selfatten_enable_value_proj', True,
             'Whether value v is pre-projected before self attention or not.')
    p.Define('conv_activation', 'RELU',
             'Activation function for convolution layer in Builder.')
    p.Define('num_splits', 1,
             'Number of model parallelism splits.')
    p.Define('num_micro_batches', 1,
             'Number of spatial partition along the batch dimension. '
             'When num_micro_batches > 1, the effective batch size of the '
             'intermediate activation is batch_size // num_micro_batches.'
             'This allows models to try larger batch size which might improve '
             'model quality')
    p.Define('glu_with_tanh', False,
             'If the Gated Linear Unit should apply tanh on the activation '
             'input.')
    p.Define('packed_input', False,
             'Whether to support packed input')
    p.Define('enable_query_scale', True, 'Enable scaling of query vector.')
    p.Define('enable_per_dim_scale', True,
             'Whether using per_dim_scale or scaling by a constant factor. '
             'Only applied when enable_query_scale == True.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm. ')
    p.Define('layernorm_tpl', layers.LayerNorm.Params(), 'Template for the '
             'LayerNorm layers. use_fused_layernorm param above overrides the '
             'layernorm_tpl.use_fused_layernorm for compatibility.')
    p.Define('use_bias', True, 'Whether to use bias for projection layer.')
    p.Define('norm_layer_tpl', None,
             'If specified, the normalization layer template.')
    p.Define(
        'enable_scaling_code_motion', False, 'Move scalings from the side '
        'of T^2 to the side of T for better performance. This may result '
        'in model quality drops when using bf16 for some models due to '
        'different XLA fusion decisions.')
    p.Define('funnel_pool_tpl', FunnelPoolingLayer.Params(),
             'Template for the Funnel Pooling layer.')
    p.Define('survival_prob', 1.0,
             'Survival probability for the residual branch.')
    # SPMD partition related params.
    #
    # d - model_dim
    # n - num_heads
    # h - attention_dim_per_heads
    # f - ff_hidden_dim
    # b - batch_size
    # l - seq_len
    p.weight_split_dims_mapping = hyperparams.Params()
    wp = p.weight_split_dims_mapping
    wp.Define('dnh', None, 'Mesh split for attention DNH weight with the shape '
              'of [model_dim, num_heads, dim_per_head].')
    wp.Define('df', None,
              'Mesh split for dense input weight with the shape of '
              '[model_dim, ff_hidden_dim].')
    wp.Define('fd', None,
              'Mesh split for dense output weight with the shape of '
              '[ff_hidden_dim, model_dim].')
    p.activation_split_dims_mapping = hyperparams.Params()
    ap = p.activation_split_dims_mapping
    ap.Define('blnh', None,
              'Mesh split for query, key, value, and encoded tensors with the '
              'shape of [batch_size, seq_len, num_heads, dim_per_head].')
    ap.Define('bld', None,
              'Mesh split for FeedForward layer input/output with the shape of '
              '[batch_size, seq_len, model_dim].')
    ap.Define('blf', None,
              'Mesh split for FeedForward layer hidden activations with the '
              'shape of [batch_size, seq_len, ff_hidden_dim].')
    return p

  @classmethod
  def SetCanonicalShardingParams(cls, params):
    """Set up canonical SPMD sharding params."""
    assert params.device_mesh.ndim >= 2
    wp = params.weight_split_dims_mapping
    wp.dnh = [0, 1, -1]
    wp.df = [0, 1]
    wp.fd = [1, 0]
    ap = params.activation_split_dims_mapping
    ap.blnh = None
    ap.bld = [1, -1, -1]
    ap.blf = [0, -1, 1]

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.num_splits > 1 or p.num_micro_batches > 1:
      assert p.deterministic_dropout

  def _Dropout(self, name, drop_prob):
    """Returns a DropoutLayer Params."""
    return super()._Dropout(name, keep_prob=1.0 - drop_prob)

  def _Add(self, name, residual_weight=1.0, apply_residual=True):
    if self.params.survival_prob < 1.0:
      assert apply_residual, ('survival_prob < 1.0 is not compatible with '
                              'apply_residual = False')
      return layers_with_attention.StochasticResidualLayer.Params().Set(
          name=name,
          residual_weight=residual_weight,
          survival_prob=self.params.survival_prob)
    else:
      return ResidualAddLayer.Params().Set(name=name,
                                           residual_weight=residual_weight,
                                           apply_residual=apply_residual)

  def _DefaultLN(self, name):
    """Layer norm with default params."""
    p = self.params
    return p.layernorm_tpl.Copy().Set(
        name=name,
        input_dim=p.model_dim,
        use_fused_layernorm=p.use_fused_layernorm,
        fprop_dtype=p.fprop_dtype)

  def _ExpandDims(self, name):
    return self._Fn(name,
                    fn=lambda x: tf.expand_dims(x, 2),
                    fn_out=lambda x: tshape.Shape(x[0:2] + [1] + x[2:]),
                    fn_flops=lambda x: 1)

  def _Squeeze(self, name):
    return self._Fn(name,
                    fn=lambda x: tf.squeeze(x, 2),
                    fn_out=lambda x: tshape.Shape(x[0:2] + x[3:]),
                    fn_flops=lambda x: 1)

  def _Glu(self, name):

    def _GLUFn(inputs):
      gated_inputs, act_inputs = tf.split(inputs, 2, axis=-1)
      return act_inputs * tf.sigmoid(gated_inputs)

    def _GatedTanhFn(inputs):
      gated_inputs, act_inputs = tf.split(inputs, 2, axis=-1)
      return tf.tanh(act_inputs) * tf.sigmoid(gated_inputs)

    fn = _GatedTanhFn if self.params.glu_with_tanh else _GLUFn

    return self._Fn(name,
                    fn=fn,
                    fn_out=lambda x: tshape.Shape(x[:-1] + [x[-1] / 2]),
                    fn_flops=lambda x: 15 * x.size)

  def _Pad(self, name):
    return PaddingLayer.Params().Set(name=name)

  def MeshSplit(self, name, tensor_split_dims_mapping):
    return MeshSplitLayer.Params().Set(
        name=name, device_mesh=self.params.device_mesh,
        tensor_split_dims_mapping=tensor_split_dims_mapping)

  def _MultiHeadedAtten(self, name, num_heads=None):
    """Returns a MultiHeadedAttention params."""
    p = self.params
    if num_heads is None:
      num_heads = p.num_heads
    atten_p = MultiHeadedAttention.Params().Set(
        name=name,
        input_dim=p.model_dim,
        hidden_dim=p.attention_hidden_dim or p.model_dim,
        num_heads=num_heads,
        atten_dropout_prob=p.atten_dropout_prob,
        enable_value_proj=p.selfatten_enable_value_proj,
        enable_query_scale=p.enable_query_scale,
        enable_per_dim_scale=p.enable_per_dim_scale,
        packed_input=p.packed_input,
        fprop_dtype=p.fprop_dtype,
        use_bias=p.use_bias,
        enable_scaling_code_motion=p.enable_scaling_code_motion,
        device_mesh=p.device_mesh,
        weight_split_dims_mapping=p.weight_split_dims_mapping.dnh)
    atten_ap = atten_p.activation_split_dims_mapping
    atten_ap.blnh = p.activation_split_dims_mapping.blnh
    atten_ap.bld = p.activation_split_dims_mapping.bld
    if p.deterministic_dropout:
      atten_p.dropout_tpl = layers.DeterministicDropoutLayer.Params()
    return atten_p

  def GatedGeluFeedforward(self, name, is_causal=False, ff_hidden_dim=None):
    return self.GatedFeedforward(
        name, is_causal, ff_hidden_dim,
        activation_fn=lambda x: tf.nn.gelu(x, approximate=True))

  def GatedFeedforward(self, name, is_causal=False, ff_hidden_dim=None,
                       activation_fn=tf.nn.relu):
    del is_causal
    p = self.params
    if ff_hidden_dim is None:
      ff_hidden_dim = p.ff_hidden_dim

    def GatedFn(x, y):
      return tf.math.multiply(activation_fn(x), y)

    sub_list = [
        ('i.vec->after_gelu', self._Graph(
            'feedforward', ['x'], ['y'],
            ('x->x1', self._DefaultLN('ln')),
            ('x1->h0', self._Linear('wi0', p.model_dim, ff_hidden_dim)),
            ('x1->h1', self._Linear('wi1', p.model_dim, ff_hidden_dim)),
            ('h0,h1->h', self._Fn('gelu', fn=GatedFn, fn_out=lambda x, y: x)),
            ('h->h_dropout', self._Dropout('dropout', p.relu_dropout_prob)),
            ('h_dropout->y', self._Linear('wo', ff_hidden_dim, p.model_dim)))),
        ('after_gelu->y', self._Dropout('dropout', p.residual_dropout_prob)),
        ('i.vec,y->added',
         self._Add('add', p.ff_residual_weight, p.ff_apply_residual)),
        ('added,i.paddings->o.vec', self._Pad('pad')),
        ('i.paddings->o.paddings', self._Id('id')),
    ]

    if p.packed_input:
      sub_list.append(('i.segment_mask->o.segment_mask',
                       self._Id('segment_mask')))

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list)

  def Feedforward(self, name, is_causal=False, ff_hidden_dim=None,
                  qdomain=None):
    del is_causal
    p = self.params
    if ff_hidden_dim is None:
      ff_hidden_dim = p.ff_hidden_dim
    if p.device_mesh is not None:
      assert p.device_mesh.ndim >= 2
      assert p.weight_split_dims_mapping.df is not None
      assert p.weight_split_dims_mapping.fd is not None
    bias_f_split = ([p.weight_split_dims_mapping.df[1]]
                    if p.weight_split_dims_mapping.df is not None else None)
    bias_d_split = ([p.weight_split_dims_mapping.fd[1]]
                    if p.weight_split_dims_mapping.fd is not None else None)
    sub_list = [
        ('i.vec->after_feedforward',
         self._Seq(
             'feedforward',
             self._DefaultLN('ln'),  # LN with default params.
             self._Linear('linear01', p.model_dim, ff_hidden_dim,
                          device_mesh=p.device_mesh,
                          weight_split_dims_mapping=(
                              p.weight_split_dims_mapping.df),
                          qdomain=qdomain),
             self.MeshSplit('split01', p.activation_split_dims_mapping.blf),
             self._Bias('bias01', ff_hidden_dim,
                        device_mesh=p.device_mesh,
                        weight_split_dims_mapping=bias_f_split),
             self._Activation('act', p.ff_activation_fn),
             self._Dropout('relu_dropout', p.relu_dropout_prob),
             self._Linear('linear02', ff_hidden_dim, p.model_dim,
                          device_mesh=p.device_mesh,
                          weight_split_dims_mapping=(
                              p.weight_split_dims_mapping.fd),
                          qdomain=qdomain),
             self.MeshSplit('split02', p.activation_split_dims_mapping.bld),
             self._Bias('bias02', p.model_dim,
                        device_mesh=p.device_mesh,
                        weight_split_dims_mapping=bias_d_split),
             self._Dropout('dropout', p.residual_dropout_prob))),
        ('i.vec,after_feedforward->added',
         self._Add('add', p.ff_residual_weight, p.ff_apply_residual)),
        ('added,i.paddings->o.vec', self._Pad('pad')),
        ('i.paddings->o.paddings', self._Id('id')),
    ]

    if p.packed_input:
      sub_list.append(('i.segment_mask->o.segment_mask',
                       self._Id('segment_mask')))

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list)

  def _MaybeSplit(self, name, blocks):
    p = self.params
    if p.num_splits == 1 and p.num_micro_batches == 1:
      return None

    num_layers = len(blocks)
    assert num_layers >= p.num_splits
    layers_per_split = (num_layers - 1) // p.num_splits + 1
    cells = []
    while blocks:
      head, blocks = blocks[:layers_per_split], blocks[layers_per_split:]
      cells.append(self._Seq('cell_{}'.format(len(cells)), *head))
    assert len(cells) == p.num_splits

    return gpipe.PipeliningLayer.Params().Set(
        name=name,
        cell_tpl=cells,
        nested_map_fprop=True,
        num_micro_batches=p.num_micro_batches)

  def _DepthwiseConv2D(self, name, filter_size, is_causal=False, qdomain=None):
    """A depthwise convolution block for lightweight conv."""
    p = self.params
    conv_builder_params = conv_layers.Builder.Params()
    if p.norm_layer_tpl:
      conv_builder_params.norm_layer_tpl = p.norm_layer_tpl
    conv_builder = conv_builder_params.Instantiate()
    return conv_builder.DepthwiseConv2D(
        name=name,
        in_dim=p.model_dim,
        depth_multiplier=1,
        filter_shape=[filter_size, 1],
        stride=(1, 1),
        dilation=(1, 1),
        activation=p.conv_activation,
        is_causal=is_causal)

  def _NormalizedDepthwiseConv2D(self, name, kernel_size, is_causal=False,
                                 qdomain=None):
    """A depthwise convolution block for lightweight conv."""
    p = self.params
    conv_builder_params = conv_layers.Builder.Params()
    conv_builder = conv_builder_params.Instantiate()
    return conv_builder.NormalizedDepthwiseConv2D(
        name=name,
        kernel_size=kernel_size,
        num_heads=p.num_heads,
        in_dim=p.model_dim,
        dropconnect_prob=p.atten_dropout_prob,
        deterministic_dropout=p.deterministic_dropout,
        is_causal=is_causal,
        qdomain=qdomain)

  def LConv(self,
            name,
            kernel_size,
            is_causal=False,
            convolution_fn=None,
            linear_qdomain=None,
            conv_qdomain=None):
    """[DEPRECATED] A lightweight convolution block as described in.

    Use conv_layers_builder.LConv() instead.

    https://arxiv.org/abs/1901.10430
    Corresponding PyTorch Implementation (L587):
    https://github.com/pytorch/fairseq/blob/v0.6.2/fairseq/models/lightconv.py


    This block can be used as an alternative to self-attention block.

    Args:
      name: name of the params
      kernel_size: kernel size used in the conv layer.
      is_causal: is causal padding or not.
      convolution_fn: Convolution to apply, default _NormalizedDepthwiseConv2D.
      linear_qdomain: The QDomain to apply to the linear layers.
      conv_qdomain: The QDomain to pass to convolution_fn.

    Returns:
      A LightWeightConvLayerBlock layer params.
    """
    p = self.params
    if convolution_fn is None:
      convolution_fn = getattr(self, '_NormalizedDepthwiseConv2D')

    sub_list = [
        ('i.vec->pre_conv',
         self._Seq(
             'pre_conv',
             self._DefaultLN('ln'),
             self._Linear('linear', p.model_dim, p.model_dim * 2, qdomain=linear_qdomain),
             self._Bias('bias', p.model_dim * 2),
             self._Glu('glu'),
             self._ExpandDims('expand'))),
        ('pre_conv,i.paddings->post_conv,o.paddings',
         convolution_fn('conv', kernel_size, is_causal, qdomain=conv_qdomain)),
        ('post_conv->after_dropout',
         self._Seq(
             'post_conv',
             self._Squeeze('squeeze'),
             self._Linear('linear', p.model_dim, p.model_dim, qdomain=linear_qdomain),
             self._Bias('bias', p.model_dim),
             self._Dropout('dropout', p.residual_dropout_prob))),
        ('i.vec,after_dropout->o.vec', self._Add('add')),
    ]
    if p.packed_input:
      sub_list.append(('i.segment_mask->o.segment_mask', self._Id('segment_mask')))

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list
    )

  def LconvBlock(self, name, kernel_size, is_causal,
                 convolution_fn):
    """A lightweight conv block followed by a feedforward one."""
    return self._Seq(
        name,
        self.LConv(
            name='lconv',
            kernel_size=kernel_size,
            is_causal=is_causal,
            convolution_fn=convolution_fn),
        self.Feedforward('ff', is_causal))

  def Seq(self, name, *subs):
    """Returns a stack of sequential layers."""
    return self._Seq(name, *subs)

  def LConvStack(self, name, kernel_sizes, is_causal=False):
    """Returns a stack of LConv layers with kernel size in kernel_sizes."""
    blocks = []
    for i, kernel_size in enumerate(kernel_sizes):
      blocks.append(
          self.LconvBlock(
              name='block_{}'.format(i),
              kernel_size=kernel_size,
              is_causal=is_causal,
              convolution_fn=None))
    return self._MaybeSplit(name, blocks) or self._Seq(name, *blocks)

  def _Stride(self, name, stride, first_n=None, axis=1):
    """Strides the input sequence.

    Args:
      name: name of this layer.
      stride: To use every k-th token, set the stride to k. When stride == 0,
        only returns the first token of the input. When stride == 1, returns
        every token in the input.
      first_n: only considers the first N tokens for the output. We use
        [:first_n:stride] to select the output tokens. If first_n is None, this
        flag is a no-op. If stride is positive, the output sequence length is
        "(first_n-1) // stride + 1". If stride is 0, first_n has to be None or
        1. first_n can't be 0. If first_n <= stride, only the first token is
        used.
      axis: along which axis to apply striding.

    Returns:
      A layer params that does stride.
    """
    return StrideLayer.Params().Set(stride=stride, first_n=first_n, axis=axis, name=name)

  def _StridedAttention(self, name, stride=1, first_n=None, num_heads=None):
    """Computes self attention with optional stride.

    Args:
      name: name of this layer.
      stride: If omitted, the default is 1: use every token in the query. To use
        every k-th token, set the stride to k. When set to 0, only use the first
        token of the query. When packed_input is true, need to make sure that
        each segment has length divisible by stride.
      first_n: only considers the first N tokens for the output. We use
        [:first_n:stride] to select the output tokens. If first_n is None, this
        flag is a no-op. If stride is positive, the output sequence length is
        "(first_n-1) // stride + 1". If stride is 0, first_n has to be None or
        1. first_n can't be 0. If first_n <= stride, only the first token is
        used.
      num_heads: the number of heads.

    Returns:
      A self attention layer params.
    """
    p = self.params
    input_to_add = ('i.vec'
                    if p.selfatten_add_unnormalized_input else 'after_ln')

    attention_inputs = 'strided_query,after_ln,after_ln,i.paddings'
    sub_list = []
    if p.packed_input:
      if stride > 1:
        # TODO(huangyp): Make sure striding won't cross segment boundaries.
        tf.logging.warning('Each segment in the packed input should has length '
                           'divisible by stride.')
      sub_list += [
          ('i.segment_mask->strided_segment_mask',
           self._Stride('segment_mask_query_stride', stride, first_n, axis=2)),
      ]
      attention_inputs += ',strided_segment_mask'

    if num_heads is None:
      num_heads = p.num_heads

    sub_list += [
        ('i.vec->after_ln',
         self._DefaultLN('LN')),
        ('after_ln->strided_query',
         self._Stride('query_after_stride', stride, first_n)),
        ('{}->after_att,prob'.format(attention_inputs),
         self._MultiHeadedAtten('atten', num_heads)),
        ('after_att->after_dropout',
         self._Dropout('dropout', p.residual_dropout_prob)),
        ('{}->strided_input'.format(input_to_add),
         self._Stride('before_add', stride, first_n)),
        ('strided_input,after_dropout->o.vec',
         self._Add('add')),
        ('i.paddings->o.paddings',
         self._Stride('padding_after_Stride', stride, first_n)),
    ]
    if p.packed_input:
      sub_list += [
          ('strided_segment_mask->o.segment_mask',
           self._Stride('segment_mask_context_stride', stride, first_n, axis=3)),
      ]

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list)

  def _Pool(self, name, stride, first_n=None):
    """Performs pooling on the input sequence.

    Args:
      name: name of this layer.
      stride: To pool every k token, set the stride to k. When stride == 1,
        returns every token in the input. When stride == 0, only returns the
        first token of the input without perform any pooling.
      first_n: only considers the first N tokens for the output. We only pool
        [:first_n] input tokens. If first_n is None, this flag is a no-op.

    Returns:
      A layer params that does stride.
    """
    p = self.params
    return p.funnel_pool_tpl.Copy().Set(
        stride=stride,
        first_n=first_n,
        name=name)

  def _FunnelAttention(self, name, stride=1, first_n=None, num_heads=None):
    """Computes self attention with optional stride.

    Args:
      name: name of this layer.
      stride: If omitted, the default is 1: use every token in the query. To
        pool every k-th token, set the stride to k. When set to 0, only use the
        first token of the query. When packed_input is true, need to make sure
        that each segment has length divisible by stride.
      first_n: only considers the first N tokens for the output.If first_n is
        None, this flag is a no-op. If stride is positive, the output sequence
        length is  "(first_n-1) // stride + 1". If stride is 0, first_n has to
        be None or 1. first_n can't be 0. If first_n <= stride, only the first
        token is used.
      num_heads: the number of heads.

    Returns:
      A self attention layer params.
    """
    p = self.params

    if p.selfatten_add_unnormalized_input:
      shortcut_sub = ('i.vec->strided_input',
                      self._Pool('shortcut_after_pooling', stride, first_n))
    else:
      # Reuse 'strided_query' to avoid doing the pooling twice
      shortcut_sub = ('strided_query->strided_input',
                      self._Id('shortcut_after_pooling'))

    # Note that only query vectors are pooled and key/value vectors are kept
    # the same (not pooled) to allow attention to retain more information
    attention_inputs = 'strided_query,after_ln,after_ln,i.paddings'

    if num_heads is None:
      num_heads = p.num_heads
    sub_list = []
    if p.packed_input:
      if stride > 1:
        assert p.funnel_pool_tpl.begin_intact == 0
        # TODO(huangyp): Make sure striding won't cross segment boundaries.
        tf.logging.warning('Each segment in the packed input should has length '
                           'divisible by stride.')
      sub_list += [
          ('i.segment_mask->strided_segment_mask',
           self._Stride('segment_mask_query_stride', stride, first_n, axis=2)),
      ]
      attention_inputs += ',strided_segment_mask'
    sub_list += [
        ('i.vec->after_ln',
         self._DefaultLN('LN')),
        ('after_ln,i.paddings->strided_query,o.paddings',
         self._Pool('query_after_pooling', stride, first_n)),
        ('{}->after_att,prob'.format(attention_inputs),
         self._MultiHeadedAtten('atten', num_heads)),
        ('after_att->after_dropout',
         self._Dropout('dropout', p.residual_dropout_prob)),
        shortcut_sub,
        ('strided_input,after_dropout->o.vec',
         self._Add('add')),
    ]
    if p.packed_input:
      sub_list += [
          ('strided_segment_mask->o.segment_mask',
           self._Stride('segment_mask_context_stride', stride, first_n, axis=3)),
      ]

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list)

  def FunnelEncoderLayer(self, name, stride=1, first_n=None,
                         ff_hidden_dim=None, num_heads=None,
                         ff_gated_fn=None,
                         num_ffns=1):
    """(inputs, paddings) -> (encoded, paddings).

    Args:
      name: the string name of the encoder layer params.
      stride: To pool every k tokens, set the stride to k. When stride == 0,
        only returns the first token of the input. When stride == 1, returns
        every token in the input.
      first_n: only considers the first N tokens for the output. We use
        pool([:first_n]) to select the output tokens. If first_n is None, this
        flag is a no-op. If stride is positive, the output sequence length is
        "(first_n-1) // stride + 1". If stride is 0, first_n has to be None or
        1. first_n can't be 0. If first_n <= stride, only the first token is
        used.
      ff_hidden_dim: The feed forward layer's hidden dimension. If specified,
        this will override p.ff_hidden_dim.
      num_heads: The number of heads for the multi-head attention module. If
        specified, this will override p.num_heads.
      ff_gated_fn: Activation function for gated feedforward layer, if None, use
        standard feedforward layer. Current supported options: None, 'silu',
        'gelu', and callable.
      num_ffns: number of ffn layers.

    Returns:
      A transformer encoder layer params that supports optional stride.
    """
    p = self.params
    if ff_hidden_dim is None:
      ff_hidden_dim = p.ff_hidden_dim
    if num_heads is None:
      num_heads = p.num_heads

    if ff_gated_fn is None:
      ff_layer = self.Feedforward('ff', ff_hidden_dim=ff_hidden_dim)
    elif ff_gated_fn == 'silu':
      ff_layer = self.GatedFeedforward('ff', ff_hidden_dim=ff_hidden_dim,
                                       activation_fn=tf.nn.silu)
    elif ff_gated_fn == 'gelu':
      ff_layer = self.GatedGeluFeedforward('ff', ff_hidden_dim=ff_hidden_dim)
    elif callable(ff_gated_fn):
      ff_layer = self.GatedFeedforward('ff', ff_hidden_dim=ff_hidden_dim,
                                       activation_fn=ff_gated_fn)
    else:
      raise ValueError('Unsupported ff_gated_fn {}.'.format(ff_gated_fn))

    s_layers = [self._FunnelAttention('self_atten', stride=stride,
                                      first_n=first_n, num_heads=num_heads),
                ff_layer]
    if num_ffns > 1:
      for ffn_id in range(1, num_ffns):
        s_layers.append(ff_layer.Copy().Set(name='ff%d' % ffn_id))
    return self._Seq(name, self._Seq('block', *s_layers))

  def TransformerEncoderLayer(self, name, stride=1, first_n=None,
                              ff_hidden_dim=None, num_heads=None):
    """(inputs, paddings) -> (encoded, paddings).

    Args:
      name: the string name of the encoder layer params.
      stride: To use every k-th token, set the stride to k. When stride == 0,
        only returns the first token of the input. When stride == 1, returns
        every token in the input.
      first_n: only considers the first N tokens for the output. We use
        [:first_n:stride] to select the output tokens. If first_n is None, this
        flag is a no-op. If stride is positive, the output sequence length is
        "(first_n-1) // stride + 1". If stride is 0, first_n has to be None or
        1. first_n can't be 0. If first_n <= stride, only the first token is
        used.
      ff_hidden_dim: The feed forward layer's hidden dimension. If specified,
        this will override p.ff_hidden_dim.
      num_heads: The number of heads for the multi-head attention module. If
        specified, this will override p.num_heads.

    Returns:
      A transformer encoder layer params that supports optional stride.
    """
    p = self.params
    if ff_hidden_dim is None:
      ff_hidden_dim = p.ff_hidden_dim
    if num_heads is None:
      num_heads = p.num_heads
    return self._Seq(name, self._Seq(
        'block',
        self._StridedAttention('self_atten', stride=stride,
                               first_n=first_n, num_heads=num_heads),
        self.Feedforward('ff', ff_hidden_dim=ff_hidden_dim)))

  def Stack(self, name, blocks):
    """Returns a stack of sequential layers."""
    return self._MaybeSplit(name, blocks) or self._Seq(name, *blocks)

  def TransformerEncoderStack(self, name, num_layers=1):
    """Returns a stack of num_layers self-attention layers."""
    blocks = [
        self.TransformerEncoderLayer(name='iter_{:0>3d}'.format(d))
        for d in range(num_layers)
    ]
    return self.Stack(name, blocks)
# pyformat: enable


class PerformerBuilder(Builder):
  """Builder for performer models. GShard mesh splits not supported."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_random_features', 384,
             'Number of random projection features for performer.')
    p.Define('attention_type', 'softmax',
             'relu|softmax, performer kernel transformation methods')
    p.Define('redraw', False,
             'Whether kernel features should be redrawn (N/A if not random).')
    return p

  def _MultiHeadedAtten(self, name, num_heads=None):
    """Returns a MultiHeadedAttention params."""
    p = self.params
    if num_heads is None:
      num_heads = p.num_heads
    assert not p.packed_input, 'Packed input not supported.'
    atten_p = MultiHeadedFavorAttention.Params().Set(
        name=name,
        input_dim=p.model_dim,
        hidden_dim=p.attention_hidden_dim or p.model_dim,
        num_heads=num_heads,
        atten_dropout_prob=p.atten_dropout_prob,
        enable_value_proj=p.selfatten_enable_value_proj,
        enable_query_scale=p.enable_query_scale,
        enable_per_dim_scale=p.enable_per_dim_scale,
        packed_input=False,
        fprop_dtype=p.fprop_dtype,
        use_bias=p.use_bias,
        num_random_features=p.num_random_features,
        attention_type=p.attention_type,
        redraw=p.redraw)
    if p.deterministic_dropout:
      atten_p.dropout_tpl = layers.DeterministicDropoutLayer.Params()
    return atten_p


class LmBuilder(Builder):
  """Langange model builder with causal padding."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dtype', tf.float32, 'Datatype to use.')
    return p

  def _Var(self, name, weights):
    return gshard_layers.VarLayer.Params().Set(name=name, weights=weights)

  def _ShardedVar(self, name, weights, mesh_split):
    sharded_weights = []
    for k, v in weights:
      sharded_weights.append((k,
                              gshard_layers.ShardedWeightParams(
                                  shape=v.shape,
                                  init=v.init,
                                  dtype=v.dtype,
                                  collections=v.collections,
                                  tensor_split_dims_mapping=mesh_split)))
    return gshard_layers.ShardedVarLayer.Params().Set(
        name=name,
        weights=sharded_weights,
        device_mesh=self.params.device_mesh,
        fprop_dtype=self.params.fprop_dtype)

  def _LinearWeight(self, name, input_dim, output_dim, mesh_split):
    return self._ShardedVar(
        name=name,
        weights=[('w',
                  py_utils.WeightParams(
                      shape=[input_dim, output_dim],
                      init=py_utils.WeightInit.Uniform((3. / input_dim)**0.5),
                      dtype=self.params.dtype))],
        mesh_split=mesh_split)

  def _Linear(self, name, input_dim, output_dim, mesh_split, qdomain=None):
    if qdomain is not None:
      raise NotImplementedError(
          'Quantization support is not implemented for LmBuilder._Linear.')
    return self._Graph(
        name,
        ['inputs'],
        ['outputs'],
        ('->w', self._LinearWeight('w', input_dim, output_dim, mesh_split)),
        ('inputs,w->outputs',
         self._Fn(
             'linear',
             fn=lambda inputs, w: tf.einsum('BLI,IO->BLO', inputs, w))),
    )

  def _BiasWeight(self, name, dim):
    return self._Var(
        name=name,
        weights=[('b',
                  py_utils.WeightParams(
                      shape=[dim],
                      init=py_utils.WeightInit.Constant(0.0),
                      dtype=self.params.dtype))])

  def _Bias(self, name, dim):
    return self._Graph(
        name,
        ['inputs'],
        ['outputs'],
        ('->b', self._BiasWeight('b', dim)),
        ('inputs,b->outputs', self._Fn('bias',
                                       fn=lambda inputs, b: inputs + b)),
    )

  def Feedforward(self, name):
    p = self.params

    ff_list = [
        self._DefaultLN('ln'),
        self._Linear('linear01', p.model_dim, p.ff_hidden_dim,
                     p.weight_split_dims_mapping.df)
    ]
    if p.use_bias:
      ff_list.append(self._Bias('bias01', p.ff_hidden_dim))
    ff_list.append(
        self.MeshSplit('hidden_split', p.activation_split_dims_mapping.blf))
    ff_list += [
        self._Activation('act', p.ff_activation_fn),
        self._Dropout('relu_dropout', p.relu_dropout_prob),
        self._Linear('linear02', p.ff_hidden_dim, p.model_dim,
                     p.weight_split_dims_mapping.fd)
    ]
    if p.use_bias:
      ff_list.append(self._Bias('bias02', p.model_dim))
    ff_list.append(
        self.MeshSplit('output_split', p.activation_split_dims_mapping.bld))
    ff_list.append(self._Dropout('dropout', p.residual_dropout_prob))

    sub_list = [
        ('i.vec->split_i',
         self.MeshSplit('input_split', p.activation_split_dims_mapping.bld)),
        ('split_i->after_feedforward', self._Seq('feedforward', *ff_list)),
        ('split_i,after_feedforward->added',
         self._Add('add', p.ff_residual_weight)),
        ('added,i.paddings->o.vec', self._Pad('pad')),
        ('i.paddings->o.paddings', self._Id('id')),
    ]

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings}
        ['o'],  # output NestedMap with {vec, paddings}
        *sub_list)

  def _Attention(self, name, is_causal=True):
    """Computes self attention with optional stride.

    Args:
      name: name of this layer.
      is_causal: If true, add cause per_step padding to the attention layer.

    Returns:
      A self attention layer params.
    """
    p = self.params
    tr_atten_p = TransformerAttentionLayer.Params().Set(
        name='transformer_atten',
        input_dim=p.model_dim,
        hidden_dim=p.attention_hidden_dim or p.model_dim,
        is_masked=is_causal,
        num_heads=p.num_heads,
        residual_dropout_prob=p.residual_dropout_prob,
        atten_dropout_prob=p.atten_dropout_prob,
        fprop_dtype=p.fprop_dtype,
        add_unnormalized_input=p.selfatten_add_unnormalized_input,
    )
    tr_atten_p.atten_tpl.use_bias = p.use_bias
    tr_atten_p.atten_tpl.enable_value_proj = p.selfatten_enable_value_proj
    tr_atten_p.atten_tpl.enable_query_scale = p.enable_query_scale
    tr_atten_p.atten_tpl.enable_per_dim_scale = p.enable_per_dim_scale
    tr_atten_p.atten_tpl.device_mesh = p.device_mesh
    tr_atten_p.atten_tpl.weight_split_dims_mapping = (
        p.weight_split_dims_mapping.dnh)
    tr_atten_p.atten_tpl.activation_split_dims_mapping.blnh = (
        p.activation_split_dims_mapping.blnh)
    tr_atten_p.atten_tpl.activation_split_dims_mapping.bld = (
        p.activation_split_dims_mapping.bld)
    if p.deterministic_dropout:
      tr_atten_p.dropout_tpl = layers.DeterministicDropoutLayer.Params()
      tr_atten_p.atten_p.dropout_tpl = layers.DeterministicDropoutLayer.Params()

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings}
        ['o'],  # output NestedMap with {vec, paddings}
        ('i.vec->split_i',
         self.MeshSplit('input_split', p.activation_split_dims_mapping.bld)),
        ('split_i,split_i,i.paddings->o.vec,unused_prob', tr_atten_p),
        ('i.paddings->o.paddings', self._Id('id')))

  def TransformerEncoderLayer(self, name, is_causal=True):
    """(inputs, paddings) -> (encoded, paddings).

    Args:
      name: the string name of the encoder layer params.
      is_causal: If true, add cause per_step padding to the attention layer.

    Returns:
      A transformer encoder layer params that supports optional stride.
    """
    # Hack to be compatible with ckpt generated by self._rep
    return self._Seq(
        name,
        self._Seq('block', self._Attention('self_atten', is_causal=is_causal),
                  self.Feedforward('ff')))
