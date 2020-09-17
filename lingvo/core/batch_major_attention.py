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
"""

import bisect
from lingvo import compat as tf
from lingvo.core import attention_util
from lingvo.core import base_layer
from lingvo.core import builder
from lingvo.core import computation_cost
from lingvo.core import conv_layers_builder as conv_layers
from lingvo.core import gpipe
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import moe_layers
from lingvo.core import py_utils
from lingvo.core import symbolic
from lingvo.core import tshape
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops

# pylint: enable=g-direct-tensorflow-import


def CausalPadding(slen, dtype=tf.float32):
  return 1 - tf.linalg.band_part(tf.ones([slen, slen], dtype=dtype), -1, 0)


def GetDtypeMin(dtype=tf.float32):
  return tf.constant(-0.7, dtype=dtype) * dtype.max


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
      inputs: 4D tensor with shape [..., p.dim]

    Returns:
      outpus: 4D tensor with shape [..., p.dim]
    """
    p = self.params
    dim = symbolic.ToStatic(p.dim)
    inputs = py_utils.HasShape(inputs, [-1, -1, -1, dim])
    scale = tf.math.rsqrt(tf.cast(dim, inputs.dtype))
    scale *= tf.nn.softplus(theta.per_dim_scale) / tf.nn.softplus(
        tf.constant(0.0, dtype=inputs.dtype))
    return inputs * scale

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(
        flops=inputs.num_elements() * 5, out_shapes=(inputs,))


class MultiHeadedProjectionLayer(base_layer.BaseLayer):
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
        '"BTD,DNH->BTNH" for query,key,value projection. Otherwise we use '
        '"BTNH,DNH->BTD" for output projection.')
    p.Define('use_bias', True, 'If to add bias in projection.')
    p.Define('xla_num_partitions', None, 'Number of SPMD partitions.')
    return p

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    pc = py_utils.WeightParams(
        shape=[p.input_dim, p.num_heads, p.dim_per_head],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('w', pc)
    if p.use_bias:
      if p.is_output_projection:
        pc_bias = py_utils.WeightParams(
            shape=[p.input_dim],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
      else:
        pc_bias = py_utils.WeightParams(
            shape=[p.num_heads, p.dim_per_head],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
      self.CreateVariable('b', pc_bias)

  def FProp(self, theta, inputs):
    """Computes the multi headed projection for inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A tensor of shape [batch_size, time_steps, num_heads,
        dim_per_head] or [batch_size, time_steps, hidden_size].

    Returns:
      The projected tensor with shape [[batch_size, time_steps, hidden_size] or
      [batch_size, time_steps, num_heads, dim_per_head].
    """
    p = self.params
    if p.xla_num_partitions:
      theta.w = moe_layers.Split(
          theta.w, 1, p.xla_num_partitions, use_sharding_op=True)
    if p.is_output_projection:
      inputs = py_utils.HasShape(
          inputs, [-1, -1, p.num_heads,
                   symbolic.ToStatic(p.dim_per_head)])
      ret = tf.einsum('BTNH,DNH->BTD', inputs, theta.w)
    else:
      inputs = py_utils.HasShape(
          inputs, [-1, -1, symbolic.ToStatic(p.input_dim)])
      ret = tf.einsum('BTD,DNH->BTNH', inputs, theta.w)
    if p.use_bias:
      if p.xla_num_partitions and not p.is_output_projection:
        theta.b = moe_layers.Split(
            theta.b, 0, p.xla_num_partitions, use_sharding_op=True)
      ret += theta.b
    return ret


class MultiHeadedAttention(base_layer.BaseLayer):
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
    p.Define('input_dim', 0, 'Number of key nodes.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    p.Define('num_heads', 1, 'Num of attention heads.')
    p.Define('dropout_tpl', layers.DropoutLayer.Params(),
             'Params for dropout layer.')
    p.Define(
        'enable_value_proj', True, 'Whether value v is pre-projected '
        ' before self attention or not.')
    p.Define('enable_per_dim_scale', True,
             'Whether using per_dim_scale or scaling by a constant factor.')
    p.Define('atten_dropout_prob', 0.0,
             'Probability at which we apply dropout to the attention weights.')
    p.Define('proj_tpl', MultiHeadedProjectionLayer.Params(), 'Params for '
             'projection layer.')
    p.Define('packed_input', False, 'Whether there is packed input.')
    p.Define('use_bias', True, 'Whether to use bias for projection layers.')
    p.Define('xla_num_partitions', None, 'Number of SPMD partitions.')
    p.Define(
        'enable_scaling_code_motion', False, 'Move scalings from the side '
        'of T^2 to the side of T for better performance. This may result '
        'in model quality drops when using bf16 for some models due to '
        'different XLA fusion decisions.')
    p.Define(
        'atten_extra_logit', None, 'Extra logit for attention softmax.'
        'Notice None and 0 are different.')
    return p

  def __init__(self, params):
    """Constructs a _MultiHeadedAttention object."""
    super().__init__(params)
    p = self.params
    assert p.input_dim, 'input_dim is {}'.format(p.input_dim)
    assert p.hidden_dim, 'hidden_dim is {}'.format(p.hidden_dim)
    assert (symbolic.IsExpr(p.hidden_dim) or p.hidden_dim % p.num_heads == 0), (
        f'hidden_dim: {p.hidden_dim} is not a multiple of num_heads: '
        f'{p.num_heads}.')
    dim_per_head = p.hidden_dim // p.num_heads

    def ProjectInput():
      return p.proj_tpl.Copy().Set(
          input_dim=p.input_dim,
          num_heads=p.num_heads,
          dim_per_head=dim_per_head,
          use_bias=p.use_bias,
          xla_num_partitions=p.xla_num_partitions)

    self.CreateChild('key', ProjectInput())
    self.CreateChild('query', ProjectInput())
    if p.enable_value_proj:
      self.CreateChild('value', ProjectInput())
    if p.enable_per_dim_scale:
      self.CreateChild('per_dim_scale',
                       PerDimScaleLayer.Params().Set(dim=dim_per_head))
    self.CreateChild('atten_dropout',
                     p.dropout_tpl.Set(keep_prob=1.0 - p.atten_dropout_prob))
    # Setting is_output_projection=True to set the projection direction
    # from hidden dim to input dim.
    self.CreateChild(
        'post',
        p.proj_tpl.Copy().Set(
            input_dim=p.input_dim,
            num_heads=p.num_heads,
            dim_per_head=dim_per_head,
            is_output_projection=True,
            use_bias=p.use_bias,
            xla_num_partitions=p.xla_num_partitions))

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
    return attention_util.AttenLogits(query, key)

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

    # [s, b, n]
    return tf.einsum('BNH,SBNH->SBN', query, tf.reshape(key, [s, b, n, h]))

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

      very_negative_logits = (
          tf.ones_like(logits) * logits.dtype.max *
          tf.constant(-0.7, dtype=logits.dtype))
      padded_logits = tf.where(paddings > 0.0, very_negative_logits, logits)

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
    return attention_util.AttenContext(probs, value)

  def _AttenContextOneStep(self, theta, probs, value, time_step):
    s, b, _, _ = py_utils.GetShape(value, 4)
    n = self.params.num_heads
    h = self.params.hidden_dim // n

    return tf.einsum('SBN,SBNH->BNH', probs, tf.reshape(value, [s, b, n, h]))

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
        # The 2nd part of the softamx --- scaling.
        encoded = encoded / tf.transpose(probs_sum, [0, 2, 1, 3])

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
    if p.enable_per_dim_scale:
      query = self.per_dim_scale.FProp(theta.per_dim_scale, query)
    else:
      query *= (p.hidden_dim // p.num_heads)**-0.5

    key = py_utils.HasRank(key, 4)

    b, t, n, h = py_utils.GetShape(query, 4)
    s, b, _, _ = py_utils.GetShape(key, 4)
    paddings = py_utils.HasShape(paddings, [b, s])
    assert t == 1

    if per_step_padding is not None:
      paddings += tf.squeeze(per_step_padding, 1)

    query = tf.reshape(query, [b, n, h])
    pad = tf.reshape(
        tf.tile(tf.expand_dims(tf.transpose(paddings), 2), [1, 1, n]), [s, -1])
    very_negative_logits = (
        tf.ones_like(pad) * query.dtype.max *
        tf.constant(-0.7, dtype=query.dtype))

    def _LongSeq():
      """For long sequence, directly apply to the entire tensor with padding."""
      logits = self._AttenLogitsOneStep(theta, query, key, time_step)

      logits = tf.reshape(logits, [s, -1])
      padded_logits = tf.where(pad > 0.0, very_negative_logits, logits)
      probs = py_utils.Softmax(
          padded_logits, axis=0, extra_logit=p.atten_extra_logit)
      probs = tf.reshape(probs, [s, b, n])

      encoded = self._AttenContextOneStep(theta, probs, value, time_step)
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
        return inplace_ops.alias_inplace_update(o, ts, ot), k, q, ts + 1

      logits, _, _, _ = tf.while_loop(
          lambda _o, _k, _q, ts: ts <= time_step,
          _AttenStep,
          loop_vars=(inplace_ops.empty([s, b * n], query.dtype,
                                       init=True), key, query,
                     tf.zeros([], tf.int32)))

      padded_logits = tf.where(pad > 0.0, very_negative_logits, logits)
      probs = py_utils.Softmax(
          padded_logits, axis=0, extra_logit=p.atten_extra_logit)

      def _DotStep(o, p, v, ts):
        """Computes encoded activation.

        Args:
          o: the output activation of shape [B, N, H]
          p: probabiliy of shape [S, B*N]
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
      h = p.num_heads
      _, _, d = py_utils.GetShape(value_vec, 3)
      dh = d // h
      # TODO(b/119531146): Reshape is inefficient here. Use one-hot matmul
      # avoids the data formatting. Change this back to reshape once XLA
      # has optimized reshape performance.
      rhs = tf.reshape(
          tf.one_hot(tf.range(d) // dh, h, dtype=value_vec.dtype),
          [d, h, 1]) * tf.reshape(
              tf.one_hot(tf.range(d) % dh, dh, dtype=value_vec.dtype),
              [d, 1, dh])
      value_proj = tf.einsum('BTD,DNH->BTNH', value_vec, rhs)

    if p.packed_input and not self.do_eval:
      assert segment_mask is not None
    encoded, atten_probs = self._DotAtten(theta, query_proj, key_proj,
                                          value_proj, paddings, segment_mask,
                                          per_step_padding)
    # Post projection
    encoded = self.post.FProp(theta.post, encoded)
    return encoded, atten_probs

  def InitStates(self, theta, target_batch_size, target_max_length):
    p = self.params
    num_heads = p.num_heads
    atten_dim = p.hidden_dim
    if not atten_dim:  # Check for Pathways as atten_tpl.hidden_dim is not set.
      atten_dim = p.input_dim
    dim_per_head = atten_dim // num_heads
    # TODO(shafey): Determine if we want to make the cached shape 128 to
    # avoid padding and more efficient interpolation in beamsearch.
    return py_utils.NestedMap(
        key=inplace_ops.empty(
            shape=(target_max_length, target_batch_size, num_heads,
                   dim_per_head),
            dtype=py_utils.FPropDtype(p),
            init=True),
        value=inplace_ops.empty(
            shape=(target_max_length, target_batch_size, num_heads,
                   dim_per_head),
            dtype=py_utils.FPropDtype(p),
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
      time_step: A scalar or tensor with [B], current decode step, 0-based.
        if it's a scalar, all the time step are the same decode step.
        if it's a tensor, it represents current decode step for each sample.
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

    # Using a if condtion, in case it's more efficient to update the same index.
    if synced_time_step:
      # The extended_key and extended_value have shape [T, B, N, H].
      extended_key = inplace_ops.alias_inplace_update(
          cached_states.key, time_step, tf.reshape(new_key_proj, [b, n, h]))
      extended_value = inplace_ops.alias_inplace_update(
          cached_states.value, time_step, tf.reshape(new_value_proj, [b, n, h]))
    else:
      # The extended_key and extended_value have shape [T, B, N, H].
      selected_indices = tf.range(b) + time_step * b
      extended_key = inplace_ops.alias_inplace_update(
          tf.reshape(cached_states.key, [-1, n, h]), selected_indices,
          tf.reshape(new_key_proj, [b, n, h]))
      extended_value = inplace_ops.alias_inplace_update(
          tf.reshape(cached_states.value, [-1, n, h]), selected_indices,
          tf.reshape(new_value_proj, [b, n, h]))
      extended_key = tf.reshape(extended_key, [t, b, n, h])
      extended_value = tf.reshape(extended_value, [t, b, n, h])
    updated_state = py_utils.NestedMap(key=extended_key, value=extended_value)

    if paddings is None:
      paddings = tf.zeros([b, t], dtype=query_vec.dtype)

    encoded = self._DotAttenOneStep(
        theta,
        query_proj,
        extended_key,
        extended_value,
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
    return p

  def __init__(self, params):
    """Constructs a MultiHeadedAttentionXL object."""
    super().__init__(params)
    params = self.params

    assert not params.packed_input, 'Packed input not implemented yet.'

    if params.rel_pos_emb_dim is None or params.rel_pos_emb_dim <= 0:
      raise ValueError('Invalide rel_pos_emb_dim: %s' % params.rel_pos_emb_dim)

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

    logits = attention_util.AttenLogitsTransformerXL(query, key, sin_emb,
                                                     theta.u, theta.v,
                                                     self.params.skip_term_b)
    return logits

  def _AttenLogitsOneStep(self, theta, query, key, time_step):
    """Attention logits for one single target (query) step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, N, H].
      key:      [S, B, N, H] or [S, B, N*H/128, 128].
      time_step: Current time step.
        if it's a scalar, all the time step are the same decode step.
        if it's a tensor, it represents current decode step for each sample.

    Returns:
      A Tensor of shape [S, B, N]
    """
    p = self.params
    synced_time_step = (time_step.shape.ndims == 0)
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
    if synced_time_step:
      position = tf.expand_dims(time_step - tf.range(s), 0)
    else:
      # [b, s]
      position = (
          tf.expand_dims(time_step, -1) -
          tf.tile(tf.expand_dims(tf.range(s), 0), [b, 1]))
    # [b, s, emb_dim]
    sin_emb = self.pos_emb.FPropWithPosition(theta.pos_emb, position)
    # [b, s, n, h]
    sin_emb = self.pos_proj.FProp(theta.pos_proj, sin_emb)
    if synced_time_step:
      # [s, n, h]
      sin_emb = tf.squeeze(sin_emb, 0)
      # term b an d.
      if not p.skip_term_b:
        logits += tf.einsum('BNH,SNH->SBN', query + theta.v, sin_emb)
      else:
        logits += tf.expand_dims(tf.einsum('NH,SNH->SN', theta.v, sin_emb), 1)
    else:
      # term b an d.
      if not p.skip_term_b:
        logits += tf.einsum('BNH,BSNH->SBN', query + theta.v, sin_emb)
      else:
        logits += tf.einsum('NH,BSNH->BSN', theta.v, sin_emb)
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
    # TODO(jamesqin): support use_short_seq_opt for TransofrmerXL attention.
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
      [tgt_time=seqlen, src_time=seqlen, num_heads, attenion_dim]
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

    return attention_util.AttenLogitsRPE(query, key, abs_emb)

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
    # Gets positional embedding.
    # [1, S]
    rel_dists = tf.expand_dims(time_step - tf.range(s), 0)
    # [1, S, rel_pos_emb_dim]
    pos_emb = self.key_emb.FPropDefaultTheta(rel_dists)
    if hasattr(self, 'key_pos_proj'):
      # [1, S, N, H]
      pos_emb = self.key_pos_proj.FProp(theta.key_pos_proj, pos_emb)
      # [S, 1, N, H]
      pos_emb = tf.transpose(pos_emb, [1, 0, 2, 3])
    else:
      pos_emb = tf.reshape(pos_emb, [s, 1, n, h])
    return tf.einsum('BNH,SBNH->SBN', query,
                     tf.reshape(key, [s, b, n, h]) + pos_emb)

  def _AttenContext(self, theta, probs, value):
    # TODO(jamesqin): optimize it.
    encoded = tf.einsum('BNij,BjNH->BiNH', probs, value)

    if not self.params.skip_value_emb:
      encoded += tf.einsum('BNij,ijNH->BiNH', probs,
                           self._RelativePositionValueEmb(theta, value))
    return encoded

  def _AttenContextOneStep(self, theta, probs, value, time_step):
    s, b, _, _ = py_utils.GetShape(value, 4)
    n = self.params.num_heads
    h = self.params.hidden_dim // n

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
  """Dot-product causal self attention using a sliding window.

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
      probs: [B, U, N, W, 2 * W]
      probs_sum: [B, U, N, W, 1].
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
    paddings_block_context = attention_util.ExtractBlockContext(
        paddings,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context,
        padding_val=1)

    # -> [B, N, U, W, C]
    paddings = tf.tile(
        tf.reshape(paddings_block_context, [b, 1, u, 1, c]), [1, n, 1, w, 1])

    # Make local causal paddings.
    # -> [U, W, C]
    local_causal_padding = attention_util.MakeCausalPadding(
        seq_len=t,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context,
        dtype=paddings.dtype)
    paddings += local_causal_padding

    # -> [B, N, U, W, C]
    logits = self._AttenLogits(theta, query_blocks, key_block_context)

    very_negative_logits = (
        tf.ones_like(logits) * logits.dtype.max *
        tf.constant(-0.7, dtype=logits.dtype))
    padded_logits = tf.where(paddings > 0.0, very_negative_logits, logits)

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
     probs: Local-self-MultiHeaded Attention probablities: [B, N, U, W, C].
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
      atten_probs: [B, N, T, S].

    Raises:
      ValueError: If value projection is disabled.
    """
    b, t, d = py_utils.GetShape(query_vec, 3)
    # LocalSelfAttention doesn't support cross-attention at the moment.
    # Verify T == S, for query and value vector.
    value_vec = py_utils.HasShape(value_vec, [b, t, d])
    key_vec = py_utils.HasShape(key_vec, [b, t, d])
    paddings = py_utils.HasShape(paddings, [b, t])
    return super().FProp(
        theta,
        query_vec,
        key_vec,
        value_vec,
        paddings,
        segment_mask=segment_mask,
        per_step_padding=per_step_padding)

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

    This function is used by autoregressive decoding. This function knows the
    length of full sequence, thus it is different from StreamingExtendStep.

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
    local_causal_padding = 1.0 - tf.cast(valid_atten, dtype=query_vec.dtype)
    paddings += local_causal_padding

    return super().ExtendStep(theta, query_vec, cached_states, paddings,
                              segment_mask, per_step_padding, time_step,
                              use_short_seq_opt)

  def zero_state(self, batch_size=1):
    """Returns the initial state given the batch size.

    Args:
      batch_size: the batch size.

    Returns:
      state: The initial state for streaming inference.
    """
    p = self.params
    assert p.enable_value_proj, 'Value projection must be enabled.'
    assert p.right_context == 0, ('StreamingExtendStep does not support look '
                                  'ahead')
    key_state = tf.zeros(
        shape=[
            p.left_context, batch_size, p.num_heads, p.hidden_dim // p.num_heads
        ],
        dtype=tf.float32)
    value_state = tf.zeros(
        shape=[
            p.left_context, batch_size, p.num_heads, p.hidden_dim // p.num_heads
        ],
        dtype=tf.float32)
    state = py_utils.NestedMap(key=key_state, value=value_state)
    return state

  def StreamingExtendStep(self, query_vec, state, time_step):
    """Computes the value vector given the query of the current step.

    This function doesn't know the length of full sequence, thus it is
    different from ExtendStep.

    Args:
      query_vec: A query vector of shape [B, 1, D].
      state: A `.NestedMap` object containing tensors {key, value} which are
        results of previous attentions. key, value are of shape [T, B, N, H]
        where T is the state size of this layer.
      time_step: A tensor of shape [1] and type tf.int32. Note, we can not use
        scalar tensor here because TfLiteConverter doesn't have good support of
        it (b/138865275).

    Returns:
      output: Output of the given query vector with shape [B, 1, D].
      state: updated state.
    """
    p = self.params
    assert p.enable_value_proj, 'Value projection must be enabled.'
    assert p.right_context == 0, ('StreamingExtendStep does not support look '
                                  'ahead')
    query_vec = py_utils.with_dependencies([
        py_utils.assert_shape_match(tf.shape(query_vec), [-1, 1, p.input_dim])
    ], query_vec)
    state.key = py_utils.with_dependencies([
        py_utils.assert_shape_match(
            tf.shape(state.key),
            [p.left_context, -1, p.num_heads, p.hidden_dim // p.num_heads])
    ], state.key)
    state.value = py_utils.with_dependencies([
        py_utils.assert_shape_match(
            tf.shape(state.value),
            [p.left_context, -1, p.num_heads, p.hidden_dim // p.num_heads])
    ], state.value)

    t, b, n, h = py_utils.GetShape(state.key, 4)  # t: context window size

    # Computes key, value projection and updates state.
    new_key_proj = self.key.FProp(self.theta.key, query_vec)  # [B, 1, N, H]
    new_key_proj = tf.reshape(new_key_proj, [1, b, n, h])
    new_value_proj = self.key.FProp(self.theta.value, query_vec)  # [B, 1, N, H]
    new_value_proj = tf.reshape(new_value_proj, [1, b, n, h])
    state.key = tf.concat([state.key[1:, :, :, :], new_key_proj], axis=0)
    state.value = tf.concat([state.value[1:, :, :, :], new_value_proj], axis=0)

    # For a time step less than the context window size, the time dimension of
    # input of logits computation is equal to the time step (not a full context
    # window).
    t = tf.math.minimum(time_step[0] + 1, t)
    key_input = state.key[-t:, :, :, :]
    value_input = state.value[-t:, :, :, :]

    # Computes query projection.
    query_proj = self.query.FProp(self.theta.query, query_vec)  # [B, 1, N, H]

    # Scales the query projection.
    if p.enable_per_dim_scale:
      query_proj = self.per_dim_scale.FProp(self.theta.per_dim_scale,
                                            query_proj)
    else:
      query_proj *= h**-0.5
    query_proj = tf.reshape(query_proj, [b, n, h])

    # Computes attention outputs.
    # TODO(wildstone): Replaces the einsum ops used below with mat mul to get
    # rid of TfLite Flex ops.
    logits = self._AttenLogitsOneStep(self.theta, query_proj, key_input,
                                      t - 1)  # [T, B, N]
    logits = tf.reshape(logits, [t, -1])
    posteriors = py_utils.Softmax(
        logits, axis=0, extra_logit=p.atten_extra_logit)
    posteriors = tf.reshape(posteriors, [t, b, n])
    output = tf.einsum('TBN, TBNH->BNH', posteriors, value_input)

    # Post projection.
    output = tf.expand_dims(output, 1)
    output = self.post.FProp(self.theta.post, output)
    return output, state

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
    term_ac = tf.einsum('BUTNH,BUSNH->BNUTS', query + theta.u, key)

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


class RoutingAttention(MultiHeadedAttention):
  """"Implements a sparse attention based on k-means clustering.

  This is used in the routing transformer https://arxiv.org/pdf/2003.05997.

  This verison of multi-headed attention differs from the full attention
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
    * support using local attention on some heads.

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
        'signanificantly faster by grouping queries when determining which '
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
    clustering_p.dim_per_head = p.hidden_dim // p.num_heads
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
      atten_probs: [B, T, N, S].
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
      return self._DotAttenFastPath(theta, query, key, value, q_dists, k_dists,
                                    query_paddings, key_paddings)
    else:
      return self._DotAttenSlowPath(theta, query, key, value, q_dists, k_dists,
                                    query_paddings, key_paddings)

  def InitStates(self, theta, target_batch_size, target_max_length):
    """Initialize 'states' with .key, .value, and .key_dists."""
    p = self.params
    states = super().InitStates(theta, target_batch_size, target_max_length)
    states.key_dists = inplace_ops.empty(
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
    updated_key = inplace_ops.alias_inplace_update(
        cached_states.key, time_step, tf.reshape(key_proj, [b, n, h]))
    updated_value = inplace_ops.alias_inplace_update(
        cached_states.value, time_step, tf.reshape(value_proj, [b, n, h]))
    # Shape [T, B, N, K]
    updated_key_dists = inplace_ops.alias_inplace_update(
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
        0.1, dtype=k_dists.dtype) * k_dists.dtype.max
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
    assert isinstance(q_length, int)
    assert isinstance(k_length, int)
    q_cluster_size = int(p.query_group_size_factor * q_length / p.num_clusters)
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
    logits *= tf.math.rsqrt(tf.cast(dim_per_head, p.dtype))

    very_negative_logits = (
        tf.ones_like(logits) * logits.dtype.max *
        tf.constant(-0.7, dtype=logits.dtype))
    padded_logits = tf.where(is_key_padded, very_negative_logits, logits)

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
      times = tf.maximum(1.0, times[:, :, :, None])
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
    p.Define('input_dim', 0, 'Dimension of the transformer block input.')
    p.Define('hidden_dim', 0, 'Dimension of the attention hidden dim.')
    p.Define('num_heads', 8, 'Number of attention heads.')
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
    p.Define('add_unnormalized_input', True,
             'If set, uses unnormalized input in the residual add.')
    p.Define('add_skip_connection', True,
             'If True, add input (or normalized input) to the output.')
    p.Define('ln_tpl', layers.LayerNorm.Params(),
             'Layer norm default params. No layernorm if set to None.')
    p.Define('atten_tpl',
             MultiHeadedAttention.Params().Set(),
             'Multi-Headed Dot-Product Attention default params')
    p.Define(
        'dropout_tpl', layers.DropoutLayer.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    return p

  @classmethod
  def CommonParams(cls,
                   input_dim,
                   num_heads,
                   is_masked=False,
                   use_relative_atten=False,
                   relative_pos_emb_dim=None,
                   local_context=None,
                   left_context=None,
                   right_context=None,
                   dropout_prob=0.):
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

  def _InitAttentionParams(self, atten_tpl):
    """Returns an initialized transformer attention parameters."""
    p = self.params
    params = atten_tpl.Copy()
    params.name = 'multihead_atten'
    params.input_dim = p.input_dim
    params.hidden_dim = p.hidden_dim
    params.num_heads = p.num_heads
    params.atten_dropout_prob = p.atten_dropout_prob
    return params

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if not p.hidden_dim:
      p.hidden_dim = p.input_dim

    # Initialize attention.
    params = self._InitAttentionParams(p.atten_tpl)
    if p.is_masked and issubclass(params.cls, LocalSelfAttention):
      tf.logging.warn('\'is_masked\' is not effective when used with '
                      'LocalSelfAttention and its subclass(es).')
    self.CreateChild('atten', params)

    # Initialize attention layer normalization.
    if p.ln_tpl:
      params = p.ln_tpl.Copy()
      params.name = 'atten_ln'
      params.input_dim = p.input_dim
      self.CreateChild('layer_norm', params)

    # Initialize residual dropout.
    dropout_tpl = p.dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.CreateChild('residual_dropout', dropout_tpl)

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
    b, t, _ = py_utils.GetShape(query_vec, 3)
    unnormalized_query_vec = query_vec

    # Layer normalization.
    if p.ln_tpl:
      query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)

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
    with tf.name_scope('atten'):
      ctx_vec, atten_probs = self.atten.FProp(
          theta.atten,
          query_vec,  # query
          source_vecs,  # key
          source_vecs,  # value
          paddings,
          segment_mask=segment_mask,
          per_step_padding=per_step_padding)

    # Residual connection.
    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    if p.add_skip_connection:
      ctx_vec += input_to_add
    return ctx_vec, atten_probs

  def InitStates(self, theta, target_batch_size, target_max_length):
    return self.atten.InitStates(theta.atten, target_batch_size,
                                 target_max_length)

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 time_step,
                 use_short_seq_opt=False):
    """Compute the result and update cached states for the current step.

    This function is used by autoregressive decoding. This function knows the
    length of full sequence, thus it is different from StreamingExtendStep.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [B, 1, D]
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. key   - [T, B,
        N, H]. value - [T, B, N, H].
      time_step: A scalar or tensor with [B], current decode step, 0-based.
        if it's a scalar, all the time step are the same decode step.
        if it's a tensor, it represents current decode step for each sample.
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      cur_output: [B, 1, D]
      updated_states: A `.NestedMap` object containing the updated states.
      key   - [T, B, N, H].
      value - [T, B, N, H].

    Raises:
      ValueError: If not used as masked/causal self-attention.
    """
    p = self.params
    if not p.is_masked:
      raise ValueError(
          'ExtendStep should be used only by masked/causal self-attention.')

    t, b, _, _ = py_utils.GetShape(cached_states.key, 4)
    unnormalized_query_vec = query_vec
    time_step = tf.convert_to_tensor(time_step)

    if time_step.shape.ndims == 0:
      batch_time_step = tf.tile(tf.reshape(time_step, [-1]), [b])
    else:
      batch_time_step = time_step

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

    # Layer normalization.
    if p.ln_tpl:
      query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)

    # Multiheaded masked/causal self-attention.
    ctx_vec, updated_states = self.atten.ExtendStep(theta.atten, query_vec,
                                                    cached_states, None, None,
                                                    per_step_padding, time_step,
                                                    use_short_seq_opt)

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
    return self.atten.zero_state(batch_size)

  def StreamingExtendStep(self, query_vec, state, time_step):
    """Computes the value vector given the query of the current step.

    Args:
      query_vec: A query vector of shape [B, 1, D].
      state: A `.NestedMap` object containing tensors {key, value} which are
        results of previous attentions. key, value are of shape [T, B, N, H]
        where T is the context size of this layer.
      time_step: A tensor of shape [1] and type tf.int32. Note, we can not use
        scalar tensor here because TfLiteConverter doesn't have good support of
        it (b/138865275).

    Returns:
      output: Output of the given query vector with shape [B, 1, D].
      state: updated state.
    """
    assert isinstance(self.atten, LocalSelfAttention) or isinstance(
        self.atten, LocalSelfAttentionXL)

    p = self.params
    unnormalized_query_vec = query_vec
    if p.ln_tpl:
      query_vec = self.layer_norm.FProp(self.theta.layer_norm, query_vec)
    output, state = self.atten.StreamingExtendStep(query_vec, state, time_step)

    # Residual connection.
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    if p.add_skip_connection:
      output += input_to_add
    return output, state


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
    if p.num_heads:
      params.num_heads = p.num_heads
    params.atten_tpl.packed_input = p.packed_input
    self.CreateChild('self_atten', params)

    if p.has_aux_atten:
      # Initialize multi-headed cross-attention
      params = p.tr_atten_tpl.Copy()
      params.name = 'multihead_cross_atten'
      params.input_dim = p.input_dim
      if p.num_heads:
        params.num_heads = p.num_heads
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
      segment_mask:     [target_batch, 1, target_time, target_time]
      aux_segment_mask: [source_batch, 1, target_time, source_time]

    target_batch can be a multiple of source_batch, where samples in
    target_batch are arranged in the order of [m, source_batch] where m =
    target_batch / source_batch.

    Returns:
      The fflayer output with shape [target_batch, target_time, dim].
      atten_probs: [B, N, T, S].
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
      assert aux_segment_mask is not None, ('Need to specify aux_segment_mask '
                                            'for packed input.')
    with tf.name_scope('self_atten'):
      atten_vec, atten_probs = self.self_atten.FProp(
          theta.self_atten,
          query_vec,
          None,
          paddings,
          segment_mask=segment_mask,
          per_step_padding_override=per_step_padding_override)

    if p.has_aux_atten:
      with tf.name_scope('aux_atten'):
        # Next the cross-attention layer.
        target_batch, target_time, dim = py_utils.GetShape(query_vec, 3)

        source_batch = self._GetSourceBatchSize(aux_vec)
        source_time = self._GetSourceLength(aux_vec)

        atten_vec = tf.reshape(atten_vec, [-1, source_batch, target_time, dim])
        atten_vec = tf.reshape(
            tf.transpose(atten_vec, [1, 0, 2, 3]), [source_batch, -1, dim])
        atten_vec, atten_probs = self.cross_atten.FProp(
            theta.cross_atten,
            atten_vec,
            aux_vec,
            aux_paddings,
            segment_mask=aux_segment_mask)
        num_heads = py_utils.GetShape(atten_probs)[1]
        atten_probs = tf.reshape(
            atten_probs,
            [source_batch, -1, num_heads, target_time, source_time])
        atten_probs = tf.transpose(atten_probs, [1, 0, 2, 3, 4])
        atten_probs = tf.reshape(
            atten_probs, [target_batch, num_heads, target_time, source_time])
        atten_vec = tf.reshape(atten_vec, [source_batch, -1, target_time, dim])
        atten_vec = tf.transpose(atten_vec, [1, 0, 2, 3])
        atten_vec = tf.reshape(atten_vec, [target_batch, target_time, dim])

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

    # First the self-attention layer.
    atten_vec, updated_states = self.self_atten.ExtendStep(
        theta.self_atten, query_vec, cached_states, time_step,
        use_short_seq_opt)
    if self.params.has_aux_atten:
      source_batch = self._GetSourceBatchSize(aux_vec)
      # Next the cross-attention layer.
      atten_vec = tf.reshape(atten_vec, [source_batch, -1, dim])
      atten_vec, _ = self.cross_atten.FProp(theta.cross_atten, atten_vec,
                                            aux_vec, aux_paddings)
      atten_vec = tf.reshape(atten_vec, [target_batch, 1, -1])

    # Finally the feed-forward layer.
    cur_output = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([target_batch, 1], dtype=atten_vec.dtype))
    return cur_output, updated_states


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
      attention_tpl, new_attention_tpl, skip=['rel_pos_emb_dim', 'skip_term_b'])
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


class StackedTransformerLayers(base_layer.BaseLayer):
  """A stack of Batch-Major Transformer layers."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('has_aux_atten', False,
             'If set, introduces a second attention layer')
    p.Define('mask_self_atten', False, 'If True, use masked self-attention.')
    p.Define('num_layers', 0, 'Num of layers in this stack.')
    p.Define('mdl_dim', 0, 'Model dimension in Transformer layers.')
    p.Define('hidden_dim', 0,
             'The hidden layer dimension in Transformer layers.')
    p.Define('num_atten_heads', 0, 'Num of attention heads.')
    p.Define('dropout_prob', 0.0,
             'Apply dropout at this prob at various places.')
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
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm.')
    p.Define(
        'splits', None, 'None or a list of layer indices. If None, all layers '
        'are placed on the same and only one partition. Else, len(splits) is '
        'the number of partitions the stack is sliced into. layer_i is placed '
        'on the kth partition (0-based) where split[k] < i <= split[k+1].')
    return p

  def __init__(self, params):
    if not params.splits:
      params.splits = [params.num_layers - 1]
    else:
      assert all(x <= params.num_layers - 1 for x in params.splits)
      # Assert p.splits is strictly monotonically increasing.
      assert sorted(list(set(params.splits))) == params.splits
    super().__init__(params)
    p = self.params

    assert p.num_layers > 0
    assert p.mdl_dim > 0
    assert p.hidden_dim > 0
    assert p.num_atten_heads > 0
    assert 0.0 <= p.dropout_prob < 1.0

    if isinstance(p.transformer_layer_params_tpl, list):
      if p.num_layers % len(p.transformer_layer_params_tpl):
        raise ValueError('num_layers should be divisible by '
                         'transformer_layer_params_tpl')

    def _LayerParams(ii):
      """Construct ii-th layer params."""
      if isinstance(p.transformer_layer_params_tpl, list):
        i = ii // len(p.transformer_layer_params_tpl)
        p_ii = p.transformer_layer_params_tpl[i].Copy()
      else:
        p_ii = p.transformer_layer_params_tpl.Copy()
      p_ii.name = 'layer_%d' % ii
      p_ii.has_aux_atten = p.has_aux_atten
      p_ii.mask_self_atten = p.mask_self_atten
      p_ii.input_dim = p.mdl_dim
      p_ii.output_dim = p.mdl_dim
      p_ii.packed_input = p.packed_input
      p_ii.tr_atten_tpl.num_heads = p.num_atten_heads
      p_ii.tr_atten_tpl.atten_dropout_prob = p.dropout_prob
      p_ii.tr_atten_tpl.residual_dropout_prob = p.dropout_prob
      p_ii.tr_atten_tpl.add_unnormalized_input = p.add_unnormalized_input
      p_ii.tr_fflayer_tpl.hidden_dim = p.hidden_dim
      p_ii.tr_fflayer_tpl.residual_dropout_prob = p.dropout_prob
      p_ii.tr_fflayer_tpl.relu_dropout_prob = p.dropout_prob
      return p_ii

    layer_params = [_LayerParams(ii) for ii in range(p.num_layers)]

    self.CreateChildren('x_layers', layer_params)

    if p.final_layer_norm:
      final_ln_p = layers.LayerNorm.Params().Set(
          input_dim=p.mdl_dim, use_fused_layernorm=p.use_fused_layernorm)
      self.CreateChild('final_ln', final_ln_p)

  @classmethod
  def GetSplitForLayer(cls, buckets, layer_index):
    assert layer_index <= buckets[-1], (
        f'layer_index:{layer_index} > buckets[-1]:{buckets[-1]}')
    #  Return index of the smallest element greater than or equal to layer_index
    return bisect.bisect_left(buckets, layer_index)

  def FProp(self,
            theta,
            query_vec,
            paddings,
            aux_vec=None,
            aux_paddings=None,
            segment_mask=None,
            aux_segment_mask=None):
    """Stacked Transformer layer.

    Args:
      theta: A `NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec:      [batch, target_time, dim].
      paddings:       [batch, target_time].
      aux_vec:        [batch, source_time, dim].
      aux_paddings:   [batch, source_time].
      segment_mask:     [batch, 1, target_time, target_time]
      aux_segment_mask: [batch, 1, target_time, source_time]

    Returns:
      (context, paddings), where the context vector has shape [batch,
      target_time, dim].
    """
    p = self.params
    x_out = query_vec
    cluster = self.cluster

    with tf.name_scope(p.name):
      for i in range(p.num_layers):
        x_in = x_out
        with tf.device(
            cluster.WorkerDeviceInModelSplit(
                self.GetSplitForLayer(self.params.splits, i))):
          x_out, _ = self.x_layers[i].FProp(theta.x_layers[i], x_in, paddings,
                                            aux_vec, aux_paddings, segment_mask,
                                            aux_segment_mask)
    if p.final_layer_norm:
      # Place on the last device.
      with tf.device(
          cluster.WorkerDeviceInModelSplit(
              self.GetSplitForLayer(self.params.splits, p.num_layers - 1))):
        x_out = self.final_ln.FProp(theta.final_ln, x_out)
    return x_out, paddings

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
                 use_short_seq_opt=False):
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
        "key"   - [target_time, target_batch, num_heads, dim_per_head].
        "value" - [target_time, target_batch, num_heads, dim_per_head].
      time_step: A scalar, the current decode step, 0-based.
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      cur_output: The last decoder layer output of shape [target_batch, 1, dim].
      updated_states: A `.NestedMap` object containing the updated states.
      updated_states.x_layers is a list corresponding to self.x_layers, where
      each element is a NestedMap with attention keys and values:
      "key"   - [target_time, target_batch, num_heads, dim_per_head].
      "value" - [target_time, target_batch, num_heads, dim_per_head].
    """
    p = self.params
    with tf.name_scope(p.name):
      updated_states = py_utils.NestedMap(x_layers=[])
      decoder_input = query_vec
      for layer, layer_theta, layer_states in zip(self.x_layers, theta.x_layers,
                                                  cached_states.x_layers):
        decoder_output, updated_layer_states = layer.ExtendStep(
            layer_theta, decoder_input, aux_vec, aux_paddings, layer_states,
            time_step, use_short_seq_opt)
        updated_states.x_layers.append(updated_layer_states)
        decoder_input = decoder_output
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
        theta.residual_dropout, self.fflayer.FProp(theta.fflayer,
                                                   *fflayer_args))
    return h


# TODO(ankurbpn,huangyp): Remove this layer.
class GPipeTransformerLayer(TransformerLayer):
  """GPipe compatible transformer layer.

  DEPRECATED: This layer and its use in GPipeTransformerStack is
  deprecated. Consider using the new GPipeBatchMajorTransformerStack instead.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.tr_fflayer_tpl = TransformerFeedForwardLayerWithTaskId.Params()
    return p

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            target_vecs,
            target_paddings,
            source_segment_id,
            target_segment_id,
            transparent_acc,
            transparent_acc_helper,
            source_task_id=None,
            target_task_id=None):
    p = self.params
    with tf.name_scope(p.name):
      if p.has_aux_atten:  # Decoder FProp
        seg_mask = SegmentMask(target_segment_id, target_segment_id)
        aux_seg_mask = SegmentMask(target_segment_id, source_segment_id)
        atten_vec, _ = self.self_atten.FProp(
            theta.self_atten,
            target_vecs,
            None,
            target_paddings,
            segment_mask=seg_mask)
        atten_vec, _ = self.cross_atten.FProp(
            theta.cross_atten,
            atten_vec,
            source_vecs,
            source_paddings,
            segment_mask=aux_seg_mask)
        atten_vec = self.fflayer.FProp(theta.fflayer, atten_vec,
                                       target_paddings, target_task_id)
        atten_vec.set_shape(target_vecs.shape)
        return (source_vecs, source_paddings, atten_vec, target_paddings,
                source_segment_id, target_segment_id, transparent_acc,
                transparent_acc_helper, source_task_id, target_task_id)
      # Encoder FProp
      seg_mask = SegmentMask(source_segment_id, source_segment_id)
      atten_vec, _ = self.self_atten.FProp(
          theta.self_atten,
          source_vecs,
          None,
          source_paddings,
          segment_mask=seg_mask)
      atten_vec = self.fflayer.FProp(theta.fflayer, atten_vec, source_paddings,
                                     source_task_id)
      atten_vec.set_shape(source_vecs.shape)

      return (atten_vec, source_paddings, target_vecs, target_paddings,
              source_segment_id, target_segment_id, transparent_acc,
              transparent_acc_helper, source_task_id, target_task_id)

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
                 task_id=None,
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
      task_id: [batch_size]: the input task_id meta information.
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
    atten_vec = tf.reshape(atten_vec, [source_batch, -1, dim])
    atten_vec, _ = self.cross_atten.FProp(theta.cross_atten, atten_vec, aux_vec,
                                          aux_paddings)
    atten_vec = tf.reshape(atten_vec, [target_batch, 1, -1])

    # Finally the feed-forward layer.
    cur_output = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([target_batch, 1], dtype=atten_vec.dtype), task_id)
    return cur_output, updated_states


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
    atten_vec = tf.reshape(atten_vec, [source_batch, -1, dim])
    atten_vec, _ = self.cross_atten.FProp(theta.cross_atten, atten_vec, aux_vec,
                                          aux_paddings)
    atten_vec = tf.reshape(atten_vec, [target_batch, 1, -1])

    # Finally the feed-forward layer.
    cur_output = self.fflayer.FProp(
        theta.fflayer, atten_vec,
        tf.zeros([target_batch, 1], dtype=atten_vec.dtype))

    if self.params.output_layer_norm:
      cur_output = self.layer_norm.FProp(theta.layer_norm, cur_output)
    return cur_output, updated_states


class ResidualAddLayer(base_layer.BaseLayer):
  """A layer to add inputs with residual weight."""

  @classmethod
  def Params(cls):
    """Params for `ResidualAddLayer`."""
    p = super().Params()
    p.Define('residual_weight', 1.0, 'Residual weight.')
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
    return x + p.residual_weight * y

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
    return py_utils.ApplyPadding(tf.expand_dims(paddings, -1), inputs)

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
    return p

  def FProp(self, theta, x):
    """Applies stride to the inputs.

    Args:
      theta: weights defined in this layer.
      x: input tensor, [batch, time, ...]. Stride is applied to the time dim.

    Returns:
      Strided tensor, with the stride applied to the second dim in x.
    """
    p = self.params
    assert p.first_n is None or p.first_n > 0
    if p.stride == 0:
      assert p.first_n is None or p.first_n == 1
      return tf.expand_dims(x[:, 0], 1)

    if p.first_n:
      return x[:, :p.first_n:p.stride]

    if p.stride == 1:
      return x

    return x[:, ::p.stride]

  @classmethod
  def FPropMeta(cls, p, x):
    py_utils.CheckShapes((x,))
    if p.stride == 0:
      return py_utils.NestedMap(
          flops=1, out_shapes=(tshape.Shape(x[0:1] + [1] + x[2:]),))

    if p.first_n:
      # out_seq_len is 1 if first_n is 1 ~ stride and is 2 if it's stride+1 ~
      # 2*stride...
      out_seq_len = (p.first_n - 1) // p.stride + 1
      return py_utils.NestedMap(
          flops=1, out_shapes=(tshape.Shape(x[0:1] + [out_seq_len] + x[2:]),))

    if p.stride == 1:
      return py_utils.NestedMap(flops=0, out_shapes=(x,))

    return py_utils.NestedMap(
        flops=1, out_shapes=(tshape.Shape(x[0:1] + x[1] // p.stride + x[2:]),))


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
    p.Define('enable_per_dim_scale', True,
             'Whether using per_dim_scale or scaling by a constant factor.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm.')
    p.Define('use_bias', True, 'Whether to use bias for projection layer.')
    p.Define('norm_layer_tpl', None,
             'If specified, the normalization layer template.')
    p.Define(
        'enable_scaling_code_motion', False, 'Move scalings from the side '
        'of T^2 to the side of T for better performance. This may result '
        'in model quality drops when using bf16 for some models due to '
        'different XLA fusion decisions.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.num_splits > 1 or p.num_micro_batches > 1:
      assert p.deterministic_dropout

  def _Dropout(self, name, drop_prob):
    """Returns a DropoutLayer Params."""
    return super()._Dropout(name, keep_prob=1.0 - drop_prob)

  def _Add(self, name, residual_weight=1.0):
    return ResidualAddLayer.Params().Set(residual_weight=residual_weight)

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
        enable_per_dim_scale=p.enable_per_dim_scale,
        packed_input=p.packed_input,
        fprop_dtype=p.fprop_dtype,
        use_bias=p.use_bias,
        enable_scaling_code_motion=p.enable_scaling_code_motion,
    )
    if p.deterministic_dropout:
      atten_p.dropout_tpl = layers.DeterministicDropoutLayer.Params()
    return atten_p

  def Feedforward(self, name, is_causal=False, ff_hidden_dim=None):
    del is_causal
    p = self.params
    if ff_hidden_dim is None:
      ff_hidden_dim = p.ff_hidden_dim
    sub_list = [
        ('i.vec->after_feedforward',
         self._Seq(
             'feedforward',
             self._LN('ln', p.model_dim,
                      use_fused_layernorm=p.use_fused_layernorm),
             self._Linear('linear01', p.model_dim, ff_hidden_dim),
             self._Bias('bias01', ff_hidden_dim),
             self._Activation('act', p.ff_activation_fn),
             self._Dropout('relu_dropout', p.relu_dropout_prob),
             self._Linear('linear02', ff_hidden_dim, p.model_dim),
             self._Bias('bias02', p.model_dim),
             self._Dropout('dropout', p.residual_dropout_prob))),
        ('i.vec,after_feedforward->added', self._Add('add', p.ff_residual_weight)),
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

  def _DepthwiseConv2D(self, name, filter_size, is_causal=False):
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

  def _NormalizedDepthwiseConv2D(self, name, kernel_size, is_causal=False):
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
        is_causal=is_causal)

  def LConv(self,
            name,
            kernel_size,
            is_causal=False,
            convolution_fn=None):
    """[DEPRECATED] A lightweight convolution block as described in.

    Use conv_layer_builder.LConv() instead.

    https://arxiv.org/abs/1901.10430
    Corresponding PyTorch Implementation (L587):
    https://github.com/pytorch/fairseq/blob/v0.6.2/fairseq/models/lightconv.py


    This block can be used as an alternative to self-attention block.

    Args:
      name: name of the params
      kernel_size: kernel size used in the conv layer.
      is_causal: is causal padding or not.
      convolution_fn: Convolution to apply, default _NormalizedDepthwiseConv2D.

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
             self._LN('ln', p.model_dim,
                      use_fused_layernorm=p.use_fused_layernorm),
             self._Linear('linear', p.model_dim, p.model_dim * 2),
             self._Bias('bias', p.model_dim * 2),
             self._Glu('glu'),
             self._ExpandDims('expand'))),
        ('pre_conv,i.paddings->post_conv,o.paddings',
         convolution_fn('conv', kernel_size, is_causal)),
        ('post_conv->after_dropout',
         self._Seq(
             'post_conv',
             self._Squeeze('squeeze'),
             self._Linear('linear', p.model_dim, p.model_dim),
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

  def _Stride(self, name, stride, first_n=None):
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

    Returns:
      A layer params that does stride.
    """
    return StrideLayer.Params().Set(stride=stride, first_n=first_n, name=name)

  def _StridedAttention(self, name, stride=1, first_n=None, num_heads=None):
    """Computes self attention with optional stride.

    Args:
      name: name of this layer.
      stride: If omitted, the default is 1: use every token in the query. To use
        every k-th token, set the stride to k. When set to 0, only use the first
        token of the query.
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
    if p.packed_input:
      attention_inputs += ',i.segment_mask'

    if num_heads is None:
      num_heads = p.num_heads

    sub_list = [
        ('i.vec->after_ln',
         self._LN('LN', p.model_dim,
                  use_fused_layernorm=p.use_fused_layernorm)),
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
      sub_list.append(('i.segment_mask->o.segment_mask', self._Id('segment_mask')))

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list)

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


class LmBuilder(Builder):
  """Langange model builder with causal padding."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('xla_num_partitions', None, 'Number of SPMD partitions.')
    p.Define('dtype', tf.float32, 'Datatype to use.')
    return p

  def _ShardedVar(self, name, weights, split_dim):
    return moe_layers.ShardedVarLayer.Params().Set(
        name=name,
        weights=weights,
        split_dimension=split_dim,
        fprop_dtype=self.params.fprop_dtype,
        num_devices=self.params.xla_num_partitions)

  def _LinearWeight(self, name, input_dim, output_dim, split_dim):
    return self._ShardedVar(
        name=name,
        weights=[('w',
                  py_utils.WeightParams(
                      shape=[input_dim, output_dim],
                      init=py_utils.WeightInit.Uniform((3. / input_dim)**0.5),
                      dtype=self.params.dtype))],
        split_dim=split_dim)

  def _Linear(self, name, input_dim, output_dim, split_dim=0):
    return self._Graph(
        name,
        ['inputs'],
        ['outputs'],
        ('->w', self._LinearWeight('w', input_dim, output_dim, split_dim)),
        ('inputs,w->outputs',
         self._Fn(
             'linear',
             fn=lambda inputs, w: tf.einsum('BLI,IO->BLO', inputs, w))),
    )

  def _BiasWeight(self, name, dim):
    return self._ShardedVar(
        name=name,
        weights=[('b',
                  py_utils.WeightParams(
                      shape=[dim],
                      init=py_utils.WeightInit.Constant(0.0),
                      dtype=self.params.dtype))],
        split_dim=0)

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
        self._LN('ln', p.model_dim, use_fused_layernorm=p.use_fused_layernorm),
        self._Linear('linear01', p.model_dim, p.ff_hidden_dim, split_dim=1)
    ]
    if p.use_bias:
      ff_list.append(self._Bias('bias01', p.ff_hidden_dim))
    ff_list += [
        self._Activation('act', p.ff_activation_fn),
        self._Dropout('relu_dropout', p.relu_dropout_prob),
        self._Linear('linear02', p.ff_hidden_dim, p.model_dim, split_dim=0)
    ]
    if p.use_bias:
      ff_list.append(self._Bias('bias02', p.model_dim))
    ff_list.append(self._Dropout('dropout', p.residual_dropout_prob))

    sub_list = [
        ('i.vec->after_feedforward', self._Seq('feedforward', *ff_list)),
        ('i.vec,after_feedforward->added', self._Add('add',
                                                     p.ff_residual_weight)),
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
    tr_atten_p.atten_tpl.enable_per_dim_scale = p.enable_per_dim_scale
    tr_atten_p.atten_tpl.xla_num_partitions = p.xla_num_partitions
    if p.deterministic_dropout:
      tr_atten_p.dropout_tpl = layers.DeterministicDropoutLayer.Params()
      tr_atten_p.atten_p.dropout_tpl = layers.DeterministicDropoutLayer.Params()

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings}
        ['o'],  # output NestedMap with {vec, paddings}
        ('i.vec,i.vec,i.paddings->o.vec,unused_prob', tr_atten_p),
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
