# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import builder
from lingvo.core import conv_layers_builder as conv_layers
from lingvo.core import gpipe
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import relative_atten_util
from lingvo.core import symbolic
from lingvo.core import tshape
from six.moves import range
from tensorflow.python.ops import inplace_ops  # pylint: disable=g-direct-tensorflow-import


def CausalPadding(slen, dtype=tf.float32):
  return 1 - tf.linalg.band_part(tf.ones([slen, slen], dtype=dtype), -1, 0)


def SegmentMask(segment_id, source_segment_id, dtype=tf.float32):
  """Calculates a segment mask for attention.

  Args:
    segment_id: [B, T]
    source_segment_id: [B, S]
    dtype: data type of generated mask.

  Returns:
    segment_mask: [B, 1, T, S]: A mask that is ready to
    be added to [B, N, T, S] attention logits.
  """
  if segment_id is None or source_segment_id is None:
    return None
  # Compute [B, T, S] = [B, T, 1] != [B, 1, S]
  ret = tf.cast(
      tf.not_equal(
          tf.expand_dims(segment_id, 2), tf.expand_dims(source_segment_id, 1)),
      dtype=dtype)
  ret *= ret.dtype.max * -0.7
  # [B, T, S] -> [B, 1, T, S]
  return tf.expand_dims(ret, axis=1)


class PerDimScaleLayer(base_layer.BaseLayer):
  """A layer to scale individual dims of the input."""

  @classmethod
  def Params(cls):
    """Params for `PerDimScaleLayer`."""
    p = super(PerDimScaleLayer, cls).Params()
    p.Define('dim', 0, 'Number of individual dims .')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a PerDimScaleLayer object."""
    super(PerDimScaleLayer, self).__init__(params)
    p = self.params
    assert p.name
    with tf.variable_scope(p.name):
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
    p = super(MultiHeadedProjectionLayer, cls).Params()
    p.Define('input_dim', 0, 'Input dimension.')
    p.Define('num_heads', 0, 'Number of heads.')
    p.Define('dim_per_head', 0, 'Size of each head.')
    p.Define(
        'is_output_projection', False,
        'Whether it is out projection or not. If False, we use '
        '"BTD,DNH->BTNH" for query,key,value projection. Otherwise we use '
        '"BTNH,DNH->BTD" for output projection.')
    p.Define('use_bias', True, 'If to add bias in projection.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MultiHeadedProjectionLayer, self).__init__(params)
    p = self.params
    assert p.name
    pc = py_utils.WeightParams(
        shape=[p.input_dim, p.num_heads, p.dim_per_head],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
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

    with tf.variable_scope(p.name):
      self.CreateVariable('w', pc)
      if p.use_bias:
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
    p = super(MultiHeadedAttention, cls).Params()
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
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a _MultiHeadedAttention object."""
    super(MultiHeadedAttention, self).__init__(params)
    p = self.params
    assert p.input_dim, 'input_dim is {}'.format(p.input_dim)
    assert p.hidden_dim, 'hidden_dim is {}'.format(p.hidden_dim)
    assert symbolic.IsExpr(
        p.hidden_dim
    ) or p.hidden_dim % p.num_heads == 0, 'hidden_dim: %s, num_heads: %s' % (
        p.hidden_dim, p.num_heads)
    dim_per_head = p.hidden_dim // p.num_heads

    with tf.variable_scope(p.name):

      def ProjectInput():
        return p.proj_tpl.Copy().Set(
            input_dim=p.input_dim,
            num_heads=p.num_heads,
            dim_per_head=dim_per_head)

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
              is_output_projection=True))

  def _AttenLogits(self, theta, query, key, per_step_padding):
    """Computes attention logits.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query: A Tensor of shape [B, T, N, H]
      key: A Tensor of shape [B, T, N, H]
      per_step_padding: A Tensor of shape [B, N, T, S] or None.

    Returns:
      A Tensor of shape [B, N, T, S]
    """
    return tf.einsum('BTNH,BSNH->BNTS', query, key)

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
      segment_mask: [B, 1, T, S]: A mask that is applied to prevent
        attention between different segments. This is already been
        converted into large negative logits. Only applied if
        packed_input = True.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, T, S] if
        not None.

    Returns:
      logits: [B, N, T, S].
    """
    key = py_utils.HasRank(key, 4)
    b, s, n, h = py_utils.GetShape(key, 4)
    query = py_utils.HasShape(query, [b, -1, n, h])
    t = py_utils.GetShape(query)[1]
    if segment_mask is not None and self.params.packed_input:
      segment_mask = py_utils.HasShape(segment_mask, [b, 1, t, s])

    logits = self._AttenLogits(theta, query, key, per_step_padding)

    # Apply segment mask.
    if self.params.packed_input and segment_mask is not None:
      # Paddings have been included in segment_mask.
      padded_logits = logits + segment_mask
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

    probs = tf.nn.softmax(padded_logits)

    probs = py_utils.HasShape(probs, [b, n, t, s])
    return probs

  def _AttenContext(self, theta, probs, value):
    return tf.einsum('BNTS,BSNH->BTNH', probs, value)

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
      segment_mask: [B, 1, T, S]: A mask that is applied to prevent
        attention between different segments. This is already been
        converted into large negative logits. Only applied if
        packed_input = True.

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
    probs = self.AttenProbs(theta, query, key, paddings, segment_mask,
                            per_step_padding)

    # Apply dropout to probs.
    probs = self.atten_dropout.FProp(theta.atten_dropout, probs)

    # Compute the attention context vector.
    encoded = self._AttenContext(theta, probs, value)
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
      segment_mask: [B, 1, T, S]: A mask that is applied to prevent
        attention between different segments. This is already been
        converted into large negative logits. Only applied if
        packed_input = True.
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
      probs = tf.nn.softmax(padded_logits, axis=0)
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
      probs = tf.nn.softmax(padded_logits, axis=0)

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

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_key_vec,
                 cached_value_vec,
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
      cached_key_vec:   [T, B, N, H].
      cached_value_vec: [T, B, N, H].
      paddings:         [B, T], or None if there is no padding.
      segment_mask:     [B, 1, T, S] or None.
      per_step_padding: A mask used by decoder self-attention to prevent
        information flow from future (causal padding). It has shape [B, 1, T] if
        not None.
      time_step: A scalar, the current decode step, 0-based.
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

    new_key_vec = query_vec
    new_value_vec = query_vec
    t, b, n, h = py_utils.GetShape(cached_key_vec, 4)

    # Project inputs to key, value and query. Each has shape [B, 1, N, H].
    new_key_proj = self.key.FProp(theta.key, new_key_vec)
    new_value_proj = self.value.FProp(theta.value, new_value_vec)
    query_proj = self.query.FProp(theta.query, query_vec)

    # The extended_key and extended_value have shape [T, B, N, H].
    cached_key_vec = inplace_ops.alias_inplace_update(
        cached_key_vec, time_step, tf.reshape(new_key_proj, [b, n, h]))
    cached_value_vec = inplace_ops.alias_inplace_update(
        cached_value_vec, time_step, tf.reshape(new_value_proj, [b, n, h]))
    extended_key = cached_key_vec
    extended_value = cached_value_vec

    if paddings is None:
      paddings = tf.zeros([b, t], dtype=new_key_vec.dtype)

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
    return encoded, cached_key_vec, cached_value_vec

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
    p = super(MultiHeadedAttentionXL, cls).Params()
    p.Define('rel_pos_emb_dim', None,
             'Dimension of relative positional embedding.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a MultiHeadedAttentionXL object."""
    super(MultiHeadedAttentionXL, self).__init__(params)
    params = self.params

    assert not params.packed_input, 'Packed input not implemented yet.'

    if params.rel_pos_emb_dim is None or params.rel_pos_emb_dim <= 0:
      raise ValueError('Invalide rel_pos_emb_dim: %s' % params.rel_pos_emb_dim)

    with tf.variable_scope(params.name):

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

  def _AttenLogits(self, theta, query, key, per_step_padding):
    b, _, n, h = py_utils.GetShape(key, 4)
    t = py_utils.GetShape(query)[1]

    # This layer only supports self attention.
    key = py_utils.HasShape(key, [b, t, n, h])

    if per_step_padding is None:
      is_causal_padding = False
    else:
      causal_padding = tf.tile(
          tf.reshape(CausalPadding(t), [1, t, t]), [b, 1, 1])
      is_causal_padding = tf.reduce_all(
          tf.equal(
              tf.cast(per_step_padding, dtype=tf.int32),
              tf.cast(causal_padding, dtype=tf.int32)))
    # [1, 2T - 1]
    pos = tf.expand_dims(tf.range(-(t - 1), t, name='relative_pos'), 0)
    sin_emb = self.pos_emb.FPropWithPosition(theta.pos_emb, pos)
    # [1, 2T - 1, N, H]
    sin_emb = self.pos_proj.FProp(theta.pos_proj, sin_emb)
    # [2T - 1, N, H]
    sin_emb = tf.squeeze(sin_emb, 0)

    logits = relative_atten_util.AttenLogitsTransformerXL(
        query, key, sin_emb, theta.u, theta.v, is_causal_padding)
    return logits

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
    logits += tf.einsum('BNH,SNH->SBN', query + theta.v, sin_emb)
    return logits

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_key_vec,
                 cached_value_vec,
                 paddings,
                 segment_mask,
                 per_step_padding,
                 time_step,
                 use_short_seq_opt=False):
    # TODO(jamesqin): support use_short_seq_opt for TransofrmerXL attention.
    assert not use_short_seq_opt
    return super(MultiHeadedAttentionXL,
                 self).ExtendStep(theta, query_vec, cached_key_vec,
                                  cached_value_vec, paddings, segment_mask,
                                  per_step_padding, time_step,
                                  use_short_seq_opt)


class MultiHeadedAttentionRPE(MultiHeadedAttention):
  """Multiheaded attention with relative positional embedding ...

  See https://arxiv.org/pdf/1803.02155.pdf.

  Notice this is only intended for self attention.
  """

  @classmethod
  def Params(cls):
    p = super(MultiHeadedAttentionRPE, cls).Params()
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

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a MultiHeadedAttentionRPE object."""
    super(MultiHeadedAttentionRPE, self).__init__(params)
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

    with tf.variable_scope(
        params.name, reuse=tf.AUTO_REUSE if params.use_global_emb else False):
      self.CreateChild('key_emb', rel_pos_emb_tpl)
      # Add projection layer if rel_pos_emb_dim is different from hidden_dim.
      if pos_proj_tpl is not None:
        self.CreateChild('key_pos_proj', pos_proj_tpl)
      if not params.skip_value_emb:
        self.CreateChild('value_emb', rel_pos_emb_tpl)
        if pos_proj_tpl is not None:
          self.CreateChild('value_pos_proj', pos_proj_tpl)

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

  def _AttenLogits(self, theta, query, key, per_step_padding):
    # TODO(jamesqin): optimize it.
    b, _, n, h = py_utils.GetShape(key, 4)
    t = py_utils.GetShape(query)[1]

    # This layer only supports self attention.
    key = py_utils.HasShape(key, [b, t, n, h])

    if per_step_padding is None:
      is_causal_padding = False
    else:
      causal_padding = tf.tile(
          tf.reshape(CausalPadding(t), [1, t, t]), [b, 1, 1])
      is_causal_padding = tf.reduce_all(
          tf.equal(
              tf.cast(per_step_padding, dtype=tf.int32),
              tf.cast(causal_padding, dtype=tf.int32)))

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

    return relative_atten_util.AttenLogitsRPE(query, key, abs_emb,
                                              is_causal_padding)

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
                 cached_key_vec,
                 cached_value_vec,
                 paddings,
                 segment_mask,
                 per_step_padding,
                 time_step,
                 use_short_seq_opt=False):
    # TODO(jamesqin): support use_short_seq_opt.
    assert not use_short_seq_opt
    return super(MultiHeadedAttentionRPE,
                 self).ExtendStep(theta, query_vec, cached_key_vec,
                                  cached_value_vec, paddings, segment_mask,
                                  per_step_padding, time_step,
                                  use_short_seq_opt)

  @classmethod
  def FPropMeta(cls, p, *args):
    return NotImplementedError()


class LocalCausalSelfAttention(MultiHeadedAttention):
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
    """Params for LocalCausalSelfAttention."""
    p = super(LocalCausalSelfAttention, cls).Params()
    p.Define(
        'block_size', None, 'Size of a processing block, if unset, default to '
        'max(1, left_context-1).')
    p.Define(
        'left_context', None, 'Number of left positions to attend '
        '(including current position).')
    p.Define('right_context', 0, 'Number of right positions to attend.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a LocalCausalSelfAttention object."""
    super(LocalCausalSelfAttention, self).__init__(params)

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
                 unused_per_step_padding=None):
    """Compute attention probability.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query:    [B, T, N, H].
      key:      [B, S=T, N, H].
      paddings: [B, T].
      segment_mask: [B, 1, T, S] not used right now.
      unused_per_step_padding: Not used.

    Returns:
      logits: [B, U, N, W, 2 * W]
    """
    p = self.params
    key = py_utils.HasRank(key, 4)
    b, t, n, h = py_utils.GetShape(key, 4)
    paddings = py_utils.HasShape(paddings, [b, t])
    query = py_utils.HasShape(query, [b, t, n, h])

    # -> [B, U, C, N, H]
    key_block_context = relative_atten_util.ExtractBlockContext(
        key,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context)
    _, u, c, _, _ = py_utils.GetShape(key_block_context)

    # -> [B, U, W, N, H]
    query_blocks = relative_atten_util.ConvertToBlocks(
        query, block_size=p.block_size)
    _, _, w, _, _ = py_utils.GetShape(query_blocks)

    # -> [B, U, C]
    paddings_block_context = relative_atten_util.ExtractBlockContext(
        paddings,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context,
        padding_val=1)

    # -> [B, N, U, W, C]
    paddings = tf.tile(
        tf.reshape(paddings_block_context, [b, 1, u, 1, c]), [1, n, 1, w, 1])

    # Make local casual paddings.
    # -> [U, W, C]
    local_causal_padding = relative_atten_util.MakeCausalPadding(
        seq_len=t,
        block_size=p.block_size,
        left_context=p.left_context,
        right_context=p.right_context)
    paddings += local_causal_padding

    # -> [B, N, U, W, C]
    logits = self._AttenLogits(theta, query_blocks, key_block_context)

    very_negative_logits = (
        tf.ones_like(logits) * logits.dtype.max *
        tf.constant(-0.7, dtype=logits.dtype))
    padded_logits = tf.where(paddings > 0.0, very_negative_logits, logits)

    probs = tf.nn.softmax(padded_logits)
    return probs

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
      key:      [B, S=T, N, H].
      value:    [B, S=T, N, H].
      paddings: [B, S=T].
      segment_mask: [B, 1, S=T, S=T].
      per_step_padding: A mask of shape [B, T, S=T] if not None.

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
    t0 = py_utils.GetShape(query)[1]

    # -> [B, N, U, W, C]
    probs = self.AttenProbs(theta, query, key, paddings, segment_mask,
                            per_step_padding)

    # Apply dropout to probs.
    probs = self.atten_dropout.FProp(theta.atten_dropout, probs)

    # -> [B, U, C, N, H]
    value_block_context = relative_atten_util.ExtractBlockContext(
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
    encoded = encoded[:, :t0, ...]
    return encoded, probs

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_key_vec,
                 cached_value_vec,
                 paddings,
                 segment_mask,
                 per_step_padding,
                 time_step,
                 use_short_seq_opt=False):
    raise NotImplementedError()

  @classmethod
  def FPropMeta(cls, p, *args):
    raise NotImplementedError()


class LocalCausalSelfAttentionXL(LocalCausalSelfAttention):
  """Local causal version of transformer-xl self attention."""

  @classmethod
  def Params(cls):
    p = super(LocalCausalSelfAttentionXL, cls).Params()
    p.Define('rel_pos_emb_dim', None,
             'Dimension of relative positional embedding.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a LocalCausalSelfAttentionXL object."""
    super(LocalCausalSelfAttentionXL, self).__init__(params)
    params = self.params
    if params.rel_pos_emb_dim is None or params.rel_pos_emb_dim <= 0:
      raise ValueError('Invalide rel_pos_emb_dim: %s' % params.rel_pos_emb_dim)

    with tf.variable_scope(params.name):

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
    return term_ac + term_bd


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
    p = super(TransformerAttentionLayer, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the transformer block input.')
    p.Define('hidden_dim', 0, 'Dimension of the attention hidden dim.')
    p.Define('num_heads', 8, 'Number of attention heads.')
    p.Define('is_masked', False, 'If set, uses masked MultiHededAttention.')
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define('atten_tpl',
             MultiHeadedAttention.Params().Set(),
             'Multi-Headed Dot-Product Attention default params')
    p.Define(
        'dropout_tpl', layers.DropoutLayer.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
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
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerAttentionLayer, self).__init__(params)
    p = self.params

    if not p.hidden_dim:
      p.hidden_dim = p.input_dim

    with tf.variable_scope(p.name):
      # Initialize multiheaded attention.
      params = p.atten_tpl.Copy()
      params.name = 'multihead_atten'
      params.input_dim = p.input_dim
      params.hidden_dim = p.hidden_dim
      params.num_heads = p.num_heads
      params.atten_dropout_prob = p.atten_dropout_prob
      self.CreateChild('atten', params)

      # Initialize attention layer normalization.
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
      per_step_padding_override: [B, T, T] for self attention or
                                 [B, T, S] for cross attention.
      segment_mask: [B, 1, T, S].

    Returns:
      output: [B, T, D].
      atten_probs: [B, N, T, S].
    """
    p = self.params
    b, t, _ = py_utils.GetShape(query_vec, 3)
    unnormalized_query_vec = query_vec

    # Layer normalization.
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

  def ExtendStep(self,
                 theta,
                 query_vec,
                 cached_states,
                 time_step,
                 use_short_seq_opt=False):
    """Compute the result and update cached states for the current step.

    This function is used by autoregressive decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      query_vec: [B, 1, D]
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for fast decoding. key   - [T, B,
        N, H]. value - [T, B, N, H].
      time_step: A scalar, the current decode step, 0-based.
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

    # Generates mask, with shape [b, 1, t].
    zero_padding = tf.fill([t], tf.constant(0.0, dtype=query_vec.dtype))
    per_step_padding = tf.where(
        tf.less(tf.range(t), tf.fill([t], time_step + 1)), zero_padding,
        tf.ones_like(zero_padding, dtype=query_vec.dtype))
    per_step_padding = tf.tile(tf.expand_dims(per_step_padding, axis=0), [b, 1])
    per_step_padding = tf.expand_dims(per_step_padding, 1)

    # Layer normalization.
    query_vec = self.layer_norm.FProp(theta.layer_norm, query_vec)

    # Multiheaded masked/causal self-attention.
    ctx_vec, updated_key_vec, updated_value_vec = self.atten.ExtendStep(
        theta.atten, query_vec, cached_states.key, cached_states.value, None,
        None, per_step_padding, time_step, use_short_seq_opt)
    updated_states = py_utils.NestedMap(
        key=updated_key_vec, value=updated_value_vec)

    # Residual connection.
    ctx_vec = self.residual_dropout.FProp(theta.residual_dropout, ctx_vec)
    input_to_add = (
        unnormalized_query_vec if p.add_unnormalized_input else query_vec)
    if p.add_skip_connection:
      ctx_vec += input_to_add
    return ctx_vec, updated_states


class TransformerLayer(base_layer.BaseLayer):
  """Transformer layer with multiheaded attention.

  Applies self-attention followed by a cross-attention and feed forward layer.
  """

  @classmethod
  def Params(cls):
    p = super(TransformerLayer, cls).Params()
    p.Define('has_aux_atten', False,
             'If set, introduces a second attention layer')
    p.Define('mask_self_atten', False, 'If True, use masked self-attention.')
    p.Define('input_dim', 0, 'Dimension of the transformer block input.')
    p.Define('output_dim', 0, 'Dimension of the transformer block output.')
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
    return p

  @classmethod
  def SetNumInputNodes(cls, p, num_input_nodes):
    p.input_dim = num_input_nodes

  @classmethod
  def NumOutputNodes(cls, p):
    return p.output_dim

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerLayer, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      # Initialize masked multi-headed self-attention
      if p.tr_self_atten_tpl is not None:
        self_atten_tpl = p.tr_self_atten_tpl
      else:
        self_atten_tpl = p.tr_atten_tpl
      params = self_atten_tpl.Copy()
      params.name = 'multihead_self_atten'
      params.input_dim = p.input_dim
      params.is_masked = p.mask_self_atten
      params.atten_tpl.packed_input = p.packed_input
      self.CreateChild('self_atten', params)

      if p.has_aux_atten:
        # Initialize multi-headed cross-attention
        params = p.tr_atten_tpl.Copy()
        params.name = 'multihead_cross_atten'
        params.input_dim = p.input_dim
        params.atten_tpl.packed_input = p.packed_input
        self.CreateChild('cross_atten', params)

      # Initialize feed-forward layer
      params = p.tr_fflayer_tpl.Copy()
      params.name = 'tr_fflayer'
      params.input_dim = p.input_dim
      params.output_dim = p.output_dim
      self.CreateChild('fflayer', params)

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
        source_batch, source_time = py_utils.GetShape(aux_vec, 2)
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
    p = self.params
    num_heads = p.tr_atten_tpl.num_heads
    atten_dim = p.tr_self_atten_tpl.hidden_dim if p.tr_self_atten_tpl else p.tr_atten_tpl.hidden_dim
    if not atten_dim:  # Check for Pathways as atten_tpl.hidden_dim is not set.
      atten_dim = p.input_dim
    dim_per_head = atten_dim // num_heads
    # TODO(shafey): Determine if we want to make the cached shape 128 to
    # avoid padding and more efficient interpolation in beamsearch.
    return py_utils.NestedMap(
        key=tf.zeros(
            shape=(target_max_length, target_batch_size, num_heads,
                   dim_per_head),
            dtype=tf.float32),
        value=tf.zeros(
            shape=(target_max_length, target_batch_size, num_heads,
                   dim_per_head),
            dtype=tf.float32))

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
    return cur_output, updated_states


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
                       LocalCausalSelfAttentionXL):
    atten_tpl.rel_pos_emb_dim = rel_pos_emb_dim
    return trans_params_copy

  if atten_type == ATTEN_TRANSFORMER_XL:
    if atten_tpl.cls == MultiHeadedAttention:
      rel_atten_tpl = MultiHeadedAttentionXL.Params()
    elif atten_tpl.cls == LocalCausalSelfAttention:
      rel_atten_tpl = (LocalCausalSelfAttentionXL.Params())
    else:
      raise ValueError('Unsupported attention: %s' % atten_tpl.cls)
  elif atten_type == ATTEN_RPE:
    rel_atten_tpl = MultiHeadedAttentionRPE.Params()

  rel_atten_tpl = hyperparams.CopyParamsTo(
      atten_tpl, rel_atten_tpl, skip=['cls'])
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
  elif attention_tpl.cls == (LocalCausalSelfAttentionXL):
    new_attention_tpl = LocalCausalSelfAttention.Params()
  else:
    raise ValueError('Unsupported attention params: %s' % attention_tpl.cls)

  new_attention_tpl = hyperparams.CopyParamsTo(
      attention_tpl, new_attention_tpl, skip=['cls', 'rel_pos_emb_dim'])
  trans_params_copy.tr_self_atten_tpl.atten_tpl = new_attention_tpl
  return trans_params_copy


class TransformerDecoderLayer(TransformerLayer):
  """Transformer decoder layer with multiheaded attention."""

  @classmethod
  def Params(cls):
    p = super(TransformerDecoderLayer, cls).Params()
    p.has_aux_atten = True
    p.mask_self_atten = True
    return p


class StackedTransformerLayers(base_layer.BaseLayer):
  """A stack of Batch-Major Transformer layers."""

  @classmethod
  def Params(cls):
    p = super(StackedTransformerLayers, cls).Params()
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
    p.Define('transformer_layer_params_tpl', TransformerLayer.Params(),
             'A template of TransformerLayer.params.')
    p.Define('final_layer_norm', False,
             'If true, apply layer normalization to the final output.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(StackedTransformerLayers, self).__init__(params)
    p = self.params

    assert p.num_layers > 0
    assert p.mdl_dim > 0
    assert p.hidden_dim > 0
    assert p.num_atten_heads > 0
    assert 0.0 <= p.dropout_prob < 1.0

    def _LayerParams(ii):
      """Construct ii-th layer params."""
      p_ii = p.transformer_layer_params_tpl.Copy()
      p.name = 'layer_%d' % ii
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

    with tf.variable_scope(p.name):
      self.CreateChildren('x_layers', layer_params)

    if p.final_layer_norm:
      final_ln_p = layers.LayerNorm.Params().Set(
          input_dim=p.mdl_dim, use_fused_layernorm=p.use_fused_layernorm)
      self.CreateChild('final_ln', final_ln_p)

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
      The attention context vector with shape [batch, target_time, dim].
    """
    p = self.params
    x_out = query_vec
    with tf.name_scope(p.name):
      for i in range(p.num_layers):
        x_in = x_out
        x_out, _ = self.x_layers[i].FProp(theta.x_layers[i], x_in, paddings,
                                          aux_vec, aux_paddings, segment_mask,
                                          aux_segment_mask)
    if p.final_layer_norm:
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
    p = super(TransformerFeedForwardLayerWithTaskId, cls).Params()
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


class GPipeTransformerLayer(TransformerLayer):
  """GPipe compatible transformer layer."""

  @classmethod
  def Params(cls):
    p = super(GPipeTransformerLayer, cls).Params()
    p.tr_fflayer_tpl = TransformerFeedForwardLayerWithTaskId.Params()
    return p

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            target_vecs,
            target_paddings,
            source_segment_mask,
            target_segment_mask,
            transparent_acc,
            transparent_acc_helper,
            source_task_id=None,
            target_task_id=None):
    p = self.params
    with tf.name_scope(p.name):
      if p.has_aux_atten:  # Decoder FProp
        atten_vec, _ = self.self_atten.FProp(
            theta.self_atten,
            target_vecs,
            None,
            target_paddings,
            segment_mask=target_segment_mask)
        atten_vec, _ = self.cross_atten.FProp(
            theta.cross_atten,
            atten_vec,
            source_vecs,
            source_paddings,
            segment_mask=source_segment_mask)
        atten_vec = self.fflayer.FProp(theta.fflayer, atten_vec,
                                       target_paddings, target_task_id)
        atten_vec.set_shape(target_vecs.shape)
        return (source_vecs, source_paddings, atten_vec, target_paddings,
                source_segment_mask, target_segment_mask, transparent_acc,
                transparent_acc_helper, source_task_id, target_task_id)
      # Encoder FProp
      atten_vec, _ = self.self_atten.FProp(
          theta.self_atten,
          source_vecs,
          None,
          source_paddings,
          segment_mask=source_segment_mask)
      atten_vec = self.fflayer.FProp(theta.fflayer, atten_vec, source_paddings,
                                     source_task_id)
      atten_vec.set_shape(source_vecs.shape)

      return (atten_vec, source_paddings, target_vecs, target_paddings,
              source_segment_mask, target_segment_mask, transparent_acc,
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


# pyformat: disable
class Builder(builder.Base):
  """Builder for self-attention layers."""

  @classmethod
  def Params(cls):
    p = super(Builder, cls).Params()
    p.Define('model_dim', 4, 'Model dim of this layer.')
    p.Define('num_heads', 1, 'Number of heads in the atten layer.')
    p.Define('ff_hidden_dim', 4, 'Hidden dim of the feedforward layer')
    p.Define('residual_dropout_prob', 0,
             'Dropout prob to the output of each sub-layer before it is added '
             'to the sub-layer input.')
    p.Define('ff_activation_fn', tf.nn.relu,
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
    return p

  def __init__(self, params):
    super(Builder, self).__init__(params)
    p = self.params
    if p.num_splits > 1 or p.num_micro_batches > 1:
      assert p.deterministic_dropout

  def _Dropout(self, name, drop_prob):
    """Returns a DropoutLayer Params."""
    return super(Builder, self)._Dropout(name, keep_prob=1.0 - drop_prob)

  def _Add(self, name, residual_weight=1.0):
    return self._Fn(name, fn=lambda x, y: x + residual_weight * y,
                    fn_out=lambda x, y: x)

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
    return self._Fn(
        name,
        fn=lambda x, p: py_utils.ApplyPadding(tf.expand_dims(p, -1), x),
        fn_out=lambda x, p: x,
        fn_flops=lambda x, p: 2 * max(x.size, p.size))

  def _MultiHeadedAtten(self, name):
    """Returns a MultiHeadedAttention params."""
    p = self.params
    atten_p = MultiHeadedAttention.Params().Set(
        name=name,
        input_dim=p.model_dim,
        hidden_dim=p.model_dim,
        num_heads=p.num_heads,
        atten_dropout_prob=p.atten_dropout_prob,
        enable_value_proj=p.selfatten_enable_value_proj,
        enable_per_dim_scale=p.enable_per_dim_scale,
        packed_input=p.packed_input,
        fprop_dtype=p.fprop_dtype
    )
    if p.deterministic_dropout:
      atten_p.dropout_tpl = layers.DeterministicDropoutLayer.Params()
    return atten_p

  def Feedforward(self, name, is_causal=False):
    del is_causal
    p = self.params

    sub_list = [
        ('i.vec->after_feedforward',
         self._Seq(
             'feedforward',
             self._LN('ln', p.model_dim,
                      use_fused_layernorm=p.use_fused_layernorm),
             self._Linear('linear01', p.model_dim, p.ff_hidden_dim),
             self._Bias('bias01', p.ff_hidden_dim),
             self._Activation('act', p.ff_activation_fn),
             self._Dropout('relu_dropout', p.relu_dropout_prob),
             self._Linear('linear02', p.ff_hidden_dim, p.model_dim),
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
    """A lightweight convolution block as described in.

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

  def _Stride(self, name, stride):
    """Strides the input sequence.

    Args:
      name: name of this layer.
      stride: To use every k-th token, set the stride to k. When stride == 0,
        only returns the first token of the input. When stride == 1, returns
        every token in the input.

    Returns:
      A layer params that does stride.
    """
    if stride == 1:
      return self._Id(name)
    if stride == 0:
      return self._Fn(
          name=name,
          fn=lambda x: tf.expand_dims(x[:, 0], 1),
          fn_out=lambda x: tshape.Shape(x[0:1] + [1] + x[2:]),
          fn_flops=lambda x: 1)
    return self._Fn(
        name=name,
        fn=lambda x: x[:, ::stride],
        fn_out=lambda x: tshape.Shape(x[0:1] + x[1] // stride + x[2:]),
        fn_flops=lambda x: 1)

  def _StridedAttention(self, name, stride=1):
    """Computes self attention with optional stride.

    Args:
      name: name of this layer.
      stride: If omitted, the default is 1: use every token in the query. To use
        every k-th token, set the stride to k. When set to 0, only use the first
        token of the query.

    Returns:
      A self attention layer params.
    """
    p = self.params
    input_to_add = ('i.vec'
                    if p.selfatten_add_unnormalized_input else 'after_ln')

    attention_inputs = 'strided_query,after_ln,after_ln,i.paddings'
    if p.packed_input:
      attention_inputs += ',i.segment_mask'

    sub_list = [
        ('i.vec->after_ln',
         self._LN('LN', p.model_dim,
                  use_fused_layernorm=p.use_fused_layernorm)),
        ('after_ln->strided_query',
         self._Stride('query_after_stride', stride)),
        ('{}->after_att,prob'.format(attention_inputs),
         self._MultiHeadedAtten('atten')),
        ('after_att->after_dropout',
         self._Dropout('dropout', p.residual_dropout_prob)),
        ('{}->strided_input'.format(input_to_add),
         self._Stride('before_add', stride)),
        ('strided_input,after_dropout->o.vec',
         self._Add('add')),
        ('i.paddings->o.paddings',
         self._Stride('padding_after_Stride', stride)),
    ]
    if p.packed_input:
      sub_list.append(('i.segment_mask->o.segment_mask', self._Id('segment_mask')))

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings, segment_mask}
        ['o'],  # output NestedMap with {vec, paddings, segment_mask}
        *sub_list)

  def TransformerEncoderLayer(self, name, stride=1):
    """(inputs, paddings) -> (encoded, paddings).

    Args:
      name: the string name of the encoder layer params.
      stride: To use every k-th token, set the stride to k. When stride == 0,
        only returns the first token of the input. When stride == 1, returns
        every token in the input.

    Returns:
      A transformer encoder layer params that supports optional stride.
    """
    # Hack to be compatible with ckpt generated by self._rep
    return self._Seq(name, self._Seq(
        'block',
        self._StridedAttention('self_atten', stride=stride),
        self.Feedforward('ff')))

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
