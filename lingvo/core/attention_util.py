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
"""Attention related utils, e.g. relative positional embeddings."""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import summary_utils


def ConvertToBlocks(x, block_size, padding_val=0.0):
  """Turns a sequence to non overlapping blocks.

  Args:
    x: a tensor of [batch, time, ...].
    block_size: int. Number of time frames in a block.
    padding_val: float. value on the padded frames.

  Returns:
    A tensor of [batch, num_blocks, block_size, ...], with necessary paddings,
    where output[:, i, ...] are x[:, i*block_size:(i+1)*block_size, ...].
  """
  shape = py_utils.GetShape(x)
  b = shape[0]
  t = shape[1]
  if block_size < 1:
    raise ValueError('block_size must be at least 1, got {}'.format(block_size))
  w = block_size
  # Pad t to be a multiply of w.
  num_blocks = (t + w - 1) // w
  pad_to_length = num_blocks * w
  padded = py_utils.PadSequenceDimension(x, pad_to_length, padding_val)
  reshaped = tf.reshape(padded, [b, num_blocks, w] + shape[2:])
  return reshaped


def ExtractBlockContext(x,
                        block_size,
                        left_context,
                        right_context,
                        padding_val=0.0):
  """Extracts temporal context for every block.

  Args:
    x: a tensor of [batch, time, ...].
    block_size: int. Number of time frames in a block.
    left_context: int. Left context size.
    right_context: int. Right context size.
    padding_val: float. value on the padded frames.

  Returns:
    A tensor of [batch, num_blocks, context_size, ...], with necessary paddings,
    where context_size = block_size + (left_context - 1) + right_context,
    and output[:, i, ...] are x[:, start-left_context+1:end+right_context, ...],
    start = i * block_size, end = (i + 1) * block_size.
  """
  if block_size < 1:
    raise ValueError('block_size must be at least 1, got {}'.format(block_size))
  if left_context < 1 or left_context > block_size + 1:
    raise ValueError(
        'left_context must be at least 1 and at most block_size + 1 = {}, '
        'got {}'.format(block_size + 1, left_context))
  if right_context < 0 or right_context > block_size:
    raise ValueError(
        'right_context must be at least 0 and at most block_size = {}, '
        'got {}'.format(block_size, right_context))

  block = ConvertToBlocks(x, block_size, padding_val)
  concat_list = [block]

  if left_context > 1:
    if block_size == left_context - 1:
      left_block = tf.roll(block, shift=1, axis=1)
    else:
      x_shift = tf.roll(x, shift=left_context - 1, axis=1)
      x_shift_block = ConvertToBlocks(x_shift, block_size, padding_val)
      left_block = x_shift_block[:, :, :left_context - 1:, ...]
    concat_list = [left_block] + concat_list

  if right_context > 0:
    if block_size == right_context:
      right_block = tf.roll(block, shift=-1, axis=1)
    else:
      x_shift = tf.roll(x, shift=-right_context, axis=1)
      x_shift_block = ConvertToBlocks(x_shift, block_size, padding_val)
      right_block = x_shift_block[:, :, -right_context:, ...]
    concat_list += [right_block]

  return tf.concat(concat_list, axis=2)


def MakeLocalMask(seq_len,
                  block_size,
                  left_context,
                  right_context,
                  dtype=tf.float32):
  """Makes the mask tensor for a full sequence.

  The returned mask reflects the given context sizes, where position i
  attends to tokens in the range [i - (left_context-1), i + right_context].

  Args:
    seq_len: int or scalar int tensor. Sequence length.
    block_size: int. Number of time frames in a block.
    left_context: int. Left context size.
    right_context: int. Right context size.
    dtype: tf.dtype, default is tf.float32.

  Returns:
    A tensor of [num_blocks, block_size, context_size] taking values in {0, 1},
    where context_size = block_size + (left_context - 1) + right_context.
    Element b, i, j is zero if in the b-th block, the i-th frame can access
    the j-th frame in the context.
  """
  seq_len = py_utils.with_dependencies([
      py_utils.assert_greater_equal(
          seq_len, 1, message='seq_len must be at least 1')
  ], seq_len)

  num_blocks = (seq_len + block_size - 1) // block_size
  context_size = block_size + (left_context - 1) + right_context

  # [num_blocks, block_size]: source positions in the original sequence.
  src_positions = tf.reshape(
      tf.range(num_blocks * block_size), [num_blocks, block_size])
  # [num_blocks,]: source positions at the start of each block.
  block_start_positions = tf.range(0, num_blocks * block_size, block_size)
  # [context_size]:  positions relative to the block start.
  relative_context_positions = tf.range(context_size) - (left_context - 1)

  # [num_blocks, context_size]: target positions in the original sequence.
  tgt_positions = (
      block_start_positions[:, tf.newaxis] +
      relative_context_positions[tf.newaxis, :])
  # [num_blocks, block_size, context_size]: position differences between source-
  # target pairs.
  position_diff = src_positions[:, :, tf.newaxis] - tgt_positions[:,
                                                                  tf.newaxis, :]
  # [num_blocks, block_size, context_size]: if attention is allowed between
  # source-target pairs.
  valid_atten = tf.math.logical_and(-right_context <= position_diff,
                                    position_diff < left_context)

  # [num_blocks, block_size]: if the source position is valid, not padded.
  valid_src = src_positions < seq_len
  # [num_blocks, context_size]: if the target position is valid, not padded.
  valid_tgt = tf.math.logical_and(0 <= tgt_positions, tgt_positions < seq_len)

  valid_atten &= tf.math.logical_and(valid_src[:, :, tf.newaxis],
                                     valid_tgt[:, tf.newaxis, :])

  return tf.cast(valid_atten, dtype=dtype)


def RelShift(x):
  """Performs relative shift on 4D tensor (first 2 axis are batching dims).

  Given input of shape [?, ?, W, W], this does "relative shifting" for the
  last two dims, s.t. output[b, n, i, j] = 0 if i > j else input[b, n, i, j-i]

  Args:
    x: A Tensor of shape [?, ?, W, W]

  Returns:
    A Tensor of the same shape as input with its content shifted (as described
    above).
  """
  b, n, w, _ = py_utils.GetShape(x)
  x = py_utils.HasShape(x, [-1, -1, w, w])
  x = tf.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1)))
  x = tf.reshape(x, [b, n, w + 1, w])
  x = x[:, :, :w, :]
  return x


def AttenLogits(query, key):
  """Computes attention logits.

  Args:
    query: A Tensor of shape [B, T, N, H]
    key: A Tensor of shape [B, T, N, H]

  Returns:
    A Tensor of shape [B, N, T, S]
  """
  return tf.einsum('BTNH,BSNH->BNTS', query, key)


def AttenContext(probs, value):
  """Computes the attention context vector based on per-head probs and value.

  Args:
    probs: [B, N, T, S].
    value: [B, S, N, H].

  Returns:
    encoded: [B, T, N, H].
  """
  return tf.einsum('BNTS,BSNH->BTNH', probs, value)


class PositionalAttenLogits(quant_utils.QuantizableLayer):
  """Implementation of the positional attention logit computation from ...

  - 'Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context'
    https://arxiv.org/pdf/1901.02860.pdf section 3.3
  - 'Self-Attention with Relative Position Representations'
    https://arxiv.org/pdf/1803.02155.pdf section 3
  """

  def RelPositionBias(self, content, abs_pos_emb, skip_term_b=False):
    """Compute relative position bias.

    This is a subroutine used by variants of self-attentions with relative
    positional embedding.

    output[b][n][i][j] = content[b][i][n] x abs_pos_emb[i-j+T-1][n]

    Padding should be masked by the caller of this function.

    B: batch size
    T: sequence length
    N: num of attention heads.
    H: per-head attention dimension.

    Args:
      tensors of the following shapes:
      content:         [N, H] if skip_term_b else [B, T, N, H]
      abs_pos_emb:     [2T - 1, N, H], the absolute positional embedding.
        abs_pos_emb[i] is the emb of relative distance i - (T-1).
      skip_term_b:     If to skip term_b in section 3.3 equation.

    Returns:
      The attention logits tensor. [N, T, T] if skip_term_b else [B, N, T, T].
    """
    if not skip_term_b:
      b, t, n, h = py_utils.GetShape(content)
      l = 2 * t - 1
      abs_pos_emb = py_utils.HasShape(abs_pos_emb, [l, n, h])
    else:
      n, h = py_utils.GetShape(content)
      l = py_utils.GetShape(abs_pos_emb)[0]
      t = (l + 1) // 2

    if not skip_term_b:
      # [B, N, T, L=2T-1]
      content, abs_pos_emb = self.ToAqtActActInputs(content, abs_pos_emb)
      term_bd = tf.einsum('BTNH,LNH->BNTL', content, abs_pos_emb)
      term_bd = self.FromAqtActActMatmul(term_bd)

      term_bd = tf.reshape(term_bd, [b, n, t * l], name='flatten')
      # [B, N, T * (L + 1)].
      term_bd = tf.pad(term_bd, ((0, 0), (0, 0), (0, t)))
      # [B, N, T, L + 1].
      term_bd = tf.reshape(term_bd, [b, n, t, l + 1], name='restore')
      return term_bd[:, :, :, t - 1::-1]
    else:
      # [N, L=2T-1]
      content, abs_pos_emb = self.ToAqtActActInputs(content, abs_pos_emb)
      term_d = tf.einsum('NH,LNH->NL', content, abs_pos_emb)
      term_d = self.FromAqtActActMatmul(term_d)

      # [N, T, L]
      term_d = tf.tile(tf.expand_dims(term_d, axis=1), [1, t, 1], name='tile')
      term_d = tf.reshape(term_d, [n, t * l])
      # [N, T * (L + 1)].
      term_d = tf.pad(term_d, ((0, 0), (0, t)))
      # [N, T, L + 1].
      term_d = tf.reshape(term_d, [n, t, l + 1], name='restore')
      return term_d[:, :, t - 1::-1]

  def _ValidateBiases(self, content_bias, positional_bias, n, h):
    if content_bias is not None:
      content_bias = py_utils.HasShape(content_bias, [n, h])
    else:
      content_bias = tf.constant(0, dtype=py_utils.FPropDtype(self.params))
    if positional_bias is not None:
      positional_bias = py_utils.HasShape(positional_bias, [n, h])
    else:
      positional_bias = tf.constant(0, dtype=py_utils.FPropDtype(self.params))
    return content_bias, positional_bias

  def _AttenLogits(self,
                   query,
                   key,
                   abs_pos_emb,
                   content_bias=None,
                   positional_bias=None,
                   skip_term_b=False):
    # pyformat: disable  (b/189357810)
    """Attention logits from TransformerXL and Self Attention RPE.

    Padding should be masked by the caller of this function.

    B: batch size
    T: sequence length
    N: num of attention heads.
    H: per-head attention dimension.

    Args:
      tensors of the following shapes:
      query:             [B, T, N, H]
      key:               [B, T, N, H]
      abs_pos_emb:     [2T - 1, N, H]. The sinusoid positional embedding from
        'Attention Is All You Need' (https://arxiv.org/abs/1706.03762).
        abs_pos_emb[i] is the emb of relative distance i - (T-1).
      content_bias:    [N, H] or None
      positional_bias: [N, H] or None
      skip_term_b:     If to skip term_b in section 3.3 equation.

    Returns:
      The attention logits tensor. [B, N, T, T]
    """
    # pyformat: enable
    b, t, n, h = py_utils.GetShape(query)
    key = py_utils.HasShape(key, [b, t, n, h])
    content_bias, positional_bias = self._ValidateBiases(
        content_bias, positional_bias, n, h)

    # [B, N, T, S=T]
    with tf.name_scope('term_ac'):
      content = query + content_bias
      content, key = self.ToAqtActActInputs(content, key)
      term_ac = tf.einsum('BTNH,BSNH->BNTS', content, key)
      term_ac = self.FromAqtActActMatmul(term_ac)
    with tf.name_scope('term_bd'):
      if skip_term_b:
        content = positional_bias
      else:
        content = query + positional_bias
      term_bd = self.RelPositionBias(content, abs_pos_emb, skip_term_b)
    return term_ac + term_bd

  def AttenLogitsXL(self, query, key, abs_pos_emb, content_bias,
                    positional_bias, skip_term_b):
    # pyformat: disable  (b/189357810)
    """Attention logits from Transformer-XL.

    Transformer-XL(https://arxiv.org/pdf/1901.02860.pdf, section 3.3) version of
    self attention with relative position embedding.

    Padding should be masked by the caller of this function.

    B: batch size
    T: sequence length
    N: num of attention heads.
    H: per-head attention dimension.

    Args:
      tensors of the following shapes:
      query:             [B, T, N, H]
      key:               [B, T, N, H]
      abs_pos_emb:     [2T - 1, N, H]. The sinusoid positional embedding from
        'Attention Is All You Need' (https://arxiv.org/abs/1706.03762).
        abs_pos_emb[i] is the emb of relative distance i - (T-1).
      content_bias:    [N, H] or None
      positional_bias: [N, H] or None
      skip_term_b:     If to skip term_b in section 3.3 equation.

    Returns:
      The attention logits tensor. [B, N, T, T]
    """
    # pyformat: enable
    return self._AttenLogits(query, key, abs_pos_emb, content_bias,
                             positional_bias, skip_term_b)

  def AttenLogitsRPE(self, query, key, abs_pos_emb):
    """Attention logits for Relative Position Representations.

    https://arxiv.org/pdf/1803.02155.pdf with trainable rel position emb.

    Padding should be masked by the caller of this function.

    B: batch size
    T: sequence length
    N: num of attention heads.
    H: per-head attention dimension.

    Args:
      tensors of the following shapes:
      query:           [B, T, N, H]
      key:             [B, T, N, H]
      abs_pos_emb:   [2T - 1, N, H]. The trainable embdding. abs_pos_emb[i] is
        the emb of relative distance i - (T-1).

    Returns:
      The attention logits tensor. [B, N, T, T]
    """
    return self._AttenLogits(query, key, abs_pos_emb)

  def AttenLogitsXLOneStep(self, query, key, abs_pos_emb, content_bias,
                           positional_bias, skip_term_b):
    """Transformer-XL attention logits for one single target (query) step.

    B: batch size
    S: sequence length
    N: num of attention heads.
    H: per-head attention dimension.

    Args:
      query:          [B, N, H].
      key:         [S, B, N, H] or [S, B, N*H/128, 128].
      abs_pos_emb: [B, S, N, H] or [S, N, H]
      content_bias:      [N, H] or None
      positional_bias:   [N, H] or None
      skip_term_b: If to skip term_b in section 3.3 equation of the
        TransformerXL paper.

    Returns:
      A Tensor of shape [S, B, N]
    """
    s, b, _, _ = py_utils.GetShape(key, 4)
    _, n, h = py_utils.GetShape(query, 3)
    key = tf.reshape(key, [s, b, n, h])
    content_bias, positional_bias = self._ValidateBiases(
        content_bias, positional_bias, n, h)

    # Term a and c.
    content = query + content_bias
    content, key = self.ToAqtActActInputs(content, key)
    term_ac = tf.einsum('BNH,SBNH->SBN', content, key)
    term_ac = self.FromAqtActActMatmul(term_ac)

    # Term b an d.
    synced_time_step = abs_pos_emb.shape.ndims == 3
    if skip_term_b:
      content = positional_bias
    else:
      content = query + positional_bias
    content, abs_pos_emb = self.ToAqtActActInputs(content, abs_pos_emb)
    if not skip_term_b:
      if synced_time_step:
        term_bd = tf.einsum('BNH,SNH->SBN', content, abs_pos_emb)
      else:
        term_bd = tf.einsum('BNH,BSNH->SBN', content, abs_pos_emb)
    else:
      if synced_time_step:
        term_bd = tf.einsum('NH,SNH->SN', content, abs_pos_emb)
      else:
        term_bd = tf.einsum('NH,BSNH->SBN', content, abs_pos_emb)
    term_bd = self.FromAqtActActMatmul(term_bd)
    # Reshape the output after dequantizing.
    if skip_term_b and synced_time_step:
      term_bd = tf.expand_dims(term_bd, 1)

    return term_ac + term_bd

  def AttenLogitsRPEOneStep(self, query, key, abs_pos_emb):
    """RPE attention logits for one single target (query) step.

    B: batch size
    S: sequence length
    N: num of attention heads.
    H: per-head attention dimension.

    Args:
      query:          [B, N, H].
      key:         [S, B, N, H] or [S, B, N*H/128, 128].
      abs_pos_emb: [S, 1, N, H]

    Returns:
      A Tensor of shape [S, B, N]
    """
    s, b, _, _ = py_utils.GetShape(key, 4)
    _, n, h = py_utils.GetShape(query, 3)
    key = tf.reshape(key, [s, b, n, h])

    key_emb = key + abs_pos_emb
    query, key_emb = self.ToAqtActActInputs(query, key_emb)
    logits = tf.einsum('BNH,SBNH->SBN', query, key_emb)
    return self.FromAqtActActMatmul(logits)


class KMeansClusteringForAtten(base_layer.BaseLayer):
  """Implements k-means clustering with mini-batch updates.

  This is used in the implementation of https://arxiv.org/pdf/2003.05997.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of clusters
  """

  @classmethod
  def Params(cls):
    """Params."""
    p = super().Params()
    p.Define(
        'num_clusters', 0, 'Number of clusters, typically around the square'
        ' root of the sequence length.')
    p.Define('num_heads', 1, 'Num of attention heads.')
    p.Define('dim_per_head', 0, 'Dimensions of each attention head.')
    p.Define('decay', 0.999, 'The decay with which to update centroids.')
    p.Define('epsilon', 1e-6, 'Tiny value to guard against divide by 0.')
    p.Define(
        'apply_layer_norm', True, 'Whether to apply LayerNorm() on the '
        'inputs first. If unset, caller must normalize first.')
    return p

  def __init__(self, params):
    """Constructs an instance which tracks its own set of centroids."""
    super().__init__(params)
    p = self.params
    assert p.num_clusters
    assert p.dim_per_head

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    # The per-head centroids. Shape [N, K, H].
    means = py_utils.WeightParams(
        shape=[p.num_heads, p.num_clusters, p.dim_per_head],
        init=py_utils.WeightInit.Gaussian(),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('means', means)

  @classmethod
  def LayerNorm(cls, x, epsilon=1e-6):
    """Performs layer normalization on the last dimension of 'x'.

    This differs from layers.LayerNorm in that it fixes both scale and bias at
    0.

    Args:
      x: An input tensor to be normalized.
      epsilon: Tiny value used to guard against rsqrt of 0.

    Returns:
      'x' with its last dimension normalized.
    """
    counts, means_ss, variance_ss, _, = tf.nn.sufficient_statistics(
        x, axes=[-1], keepdims=True)
    mean, variance = tf.nn.normalize_moments(counts, means_ss, variance_ss,
                                             None)
    return (x - mean) * tf.math.rsqrt(variance + epsilon)

  def FProp(self, theta, x, paddings=None, update=False):
    """Computes distances of the given input 'x' to all centroids.

    This implementation applies layer normalization on 'x' internally first,
    and the returned 'dists' is computed using the normalized 'x'.

    Args:
      theta: A `.NestedMap` of weights' values of this layer.
      x: A tensor of shape [B, L, N, H].
      paddings: If not None, a tensor of shape [B, L].
      update: bool, whether to update centroids using x.

    Returns:
      dists: "distances" of the given input 'x' to all centroids.
             Shape [B, L, N, K].
      k_means_loss: the average squared Euclidean distances to the closest
                    centroid, a scalar.
    """
    p = self.params
    if paddings is None:
      paddings = tf.zeros_like(x[:, :, 0, 0])
    # Shape [B, L, 1, 1]
    paddings_4d = paddings[:, :, None, None]

    if p.apply_layer_norm:
      x = KMeansClusteringForAtten.LayerNorm(x, p.epsilon)

    # 'x' is normalized (but theta.means is not), we use negative dot product to
    # approximate the Euclidean distance here.
    dists = -tf.einsum('BLNH, NKH -> BLNK', x, theta.means)

    # For padded positions we update the distances to very large numbers.
    very_large_dists = tf.ones_like(dists) * tf.constant(
        0.1, dtype=dists.dtype) * dists.dtype.max
    paddings_tiled = tf.tile(paddings_4d, [1, 1, p.num_heads, p.num_clusters])
    dists = tf.where(paddings_tiled > 0.0, very_large_dists, dists)

    # Shape [B, L, N, K], the same as 'dists' above.
    nearest_one_hot = tf.one_hot(
        tf.math.argmin(dists, axis=-1),
        p.num_clusters,
        dtype=py_utils.FPropDtype(p))
    # Same shape as the input 'x'.
    nearest_centroid = tf.einsum('BLNK, NKH -> BLNH', nearest_one_hot,
                                 theta.means)
    diff = tf.math.squared_difference(x, tf.stop_gradient(nearest_centroid))
    diff = py_utils.ApplyPadding(paddings_4d, diff)
    diff = tf.math.reduce_mean(diff, axis=2)

    # The commitment loss which when back proped against encourages the 'x'
    # values to commit to their chosen centroids.
    k_means_loss = tf.math.reduce_sum(diff) / tf.math.reduce_sum(1.0 - paddings)
    summary_utils.scalar('k_means/squared_distance_loss', k_means_loss)

    # TODO(zhouwk): investigate normalizing theta.means after each update.
    means_norm = tf.norm(theta.means)
    summary_utils.scalar('k_means/centroid_l2_norm/min',
                         tf.math.reduce_min(means_norm))
    summary_utils.scalar('k_means/centroid_l2_norm/mean',
                         tf.math.reduce_mean(means_norm))

    if not update:
      return dists, k_means_loss

    # To update the centroids (self.vars.means), we apply gradient descent on
    # the mini-batch of input 'x', which yields the following:
    #   new_centroid = centroid + (1 - decay) * (x_mean - centroid)
    # where x_mean is the average over all the input vectors closest to this
    # centroid.
    #
    # Note that this approach is equivalent with backprop via
    #    loss = tf.math.reduce_mean(
    #        tf.math.squared_difference(tf.stop_gradient(x), nearest_centroid)))
    # , except that here the learning rate is independently set via 'decay'.

    # Ensure that the padded positions are not used to update the centroids.
    nearest_one_hot = py_utils.ApplyPadding(paddings_4d, nearest_one_hot)

    # Sum away batch and sequence length dimensions to get per cluster count.
    # Shape: [N, K]
    per_cluster_count = tf.reduce_sum(nearest_one_hot, axis=[0, 1])
    summary_utils.histogram('k_means/per_cluster_vec_count', per_cluster_count)

    # Sum of the input 'x' per each closest centroid.
    sum_x = tf.einsum('BLNK, BLNH -> NKH', nearest_one_hot, x)

    if py_utils.use_tpu():
      per_cluster_count = tf.tpu.cross_replica_sum(per_cluster_count)
      sum_x = tf.tpu.cross_replica_sum(sum_x)

    # If per_cluster_count for a cluster is 0, then 'nearest_one_hot' in that
    # cluster's position will always be 0, hence 'sum_x' in that dimension will
    # be 0.
    new_means = sum_x / tf.maximum(
        tf.constant(1.0, dtype=per_cluster_count.dtype),
        tf.expand_dims(per_cluster_count, axis=-1))

    # We use exponential moving average. TODO(zhouwk): investigate smooth this
    # over an exponentially moving averaged per cluster count.
    #
    # Note that we intentionally do not normalize the means after this update
    # as empirically this works better.
    update_means_diff = tf.cast((1.0 - p.decay) * (new_means - theta.means),
                                self.vars.means.dtype)
    return py_utils.with_dependencies(
        [tf.assign_add(self.vars.means, update_means_diff)],
        dists), k_means_loss


def ComputeSparseAttention(q, k, v, sparsity_indices, paddings=None):
  """Computes attention according to a sparsity pattern.

  We use the following capital letters to denote shape parameters:
    B = batch size
    S = length of the source sequence
    T = length of the target sequence
    N = number of attention heads
    H = dimensions of each attention head
    K = number of clusters
    W = attention window (K <= S)

  The 'sparsity_indices' is a tensor of integral type where the last dimension
  contains W indices (W is the attention window) for each corresponding position
  along S in 'k' that the query is allowed to attend to.

  For example, if sparsity_indices[batch_idx, target time step, head_idx] =
  [1, 7, 8], it means that token in the query attends to values with indices
  1, 7, and 8, and the attention window here is 3.

  The valid values in 'sparsity_indices' are [-1, S-1]. Note that the value -1
  is reserved to mean paddings, distinct from the value (S-1).

  For example, if W=S and 'sparsity_indices' contains range(S) on the last
  dimension, this degenerates to the original full attention.

  We require that 'sparsity_indices' does not contain duplicates (except for -1
  to indicate paddings), but we do not require 'sparsity_indices' to be sorted.

  Note that this implementation is flexible and generic but is not optimized for
  time or space complexity. Please consider grouping queries that attend to the
  same subset of values first for efficiency.

  Args:
    q: (projected) queries, [B, T, N, H];
    k: (projected) keys, [B, S, N, H];
    v: (projected) values, [B, S, N, H];
    sparsity_indices: [B, T, N, W], where W is the attention window;
    paddings: paddings for keys, [B, S] if not None.

  Returns:
    output: the encoded output, [B, T, N, H].
    atten_probs: the attention weights, [B, T, N, S].
  """
  q = tf.convert_to_tensor(q)
  k = tf.convert_to_tensor(k)
  v = tf.convert_to_tensor(v)
  sparsity_indices = tf.convert_to_tensor(sparsity_indices)

  k = py_utils.HasRank(k, 4)
  _, source_length, _, dim_per_head = py_utils.GetShape(k, 4)
  sparsity_indices = py_utils.HasRank(sparsity_indices, 4)
  batch_size, target_length, num_heads, attention_window = py_utils.GetShape(
      sparsity_indices, 4)
  py_utils.assert_less_equal(
      attention_window, source_length,
      'The provided sparsity_indices has attention window '
      ' > source length. This is likely an error.')

  # To prepare for gathering the relevant vectors from 'k', we prepare
  # gather_idx of shape [B, T, N, W, 3] where the last dimension corresponds to
  # slices in 'k' indexed by (batch index, source time step, head index),
  # where the source length index comes from the original W dimension in
  # 'sparsity_indices'.
  seq_idx = tf.expand_dims(sparsity_indices, axis=-1)
  # Overwrite the paddings -1 with valid gather indices (zeros). We will
  # fix the logits with -inf in these positions later.
  seq_idx = tf.where(seq_idx < 0, tf.zeros_like(seq_idx), seq_idx)
  batch_idx = tf.reshape(
      tf.range(0, batch_size, dtype=sparsity_indices.dtype),
      [batch_size, 1, 1, 1, 1])
  batch_idx = tf.tile(batch_idx,
                      [1, target_length, num_heads, attention_window, 1])
  head_idx = tf.reshape(
      tf.range(0, num_heads, dtype=sparsity_indices.dtype),
      [1, 1, num_heads, 1, 1])
  head_idx = tf.tile(head_idx,
                     [batch_size, target_length, 1, attention_window, 1])
  # [B, T, N, W, 3], where last dimension is (batch index, source length index,
  # head index).
  gather_idx = tf.concat([batch_idx, seq_idx, head_idx], axis=-1)

  # Both the gathered k and v have shape [B, T, N, W, H]
  k = tf.gather_nd(k, gather_idx)
  v = tf.gather_nd(v, gather_idx)

  if paddings is None:
    paddings = tf.zeros([batch_size, source_length])
  paddings = tf.convert_to_tensor(paddings)
  paddings = tf.expand_dims(paddings, axis=-1)
  # [B, S, N]
  paddings = tf.tile(paddings, [1, 1, num_heads])
  # [B, T, N, W]
  paddings = tf.gather_nd(paddings, gather_idx)

  logits = tf.einsum('BTNH, BTNWH -> BTNW', q, k)
  logits *= tf.math.rsqrt(tf.cast(dim_per_head, q.dtype))

  very_negative_logits = (
      tf.ones_like(logits) * logits.dtype.max *
      tf.constant(-0.7, dtype=logits.dtype))
  padded_logits = tf.where(
      tf.math.logical_or(sparsity_indices < 0, paddings > 0.0),
      very_negative_logits, logits)

  # [B, T, N, W]
  atten_probs = tf.nn.softmax(padded_logits, name='attention_weights')
  atten_probs = tf.where(sparsity_indices < 0, tf.zeros_like(logits),
                         atten_probs)
  output = tf.einsum('BTNW, BTNWH -> BTNH', atten_probs, v)

  # Scatter 'atten_probs' back into the original source length.
  # [B, T, N, W, 1]
  batch_idx = tf.tile(
      tf.range(batch_size)[:, None, None, None, None],
      [1, target_length, num_heads, attention_window, 1])
  # [B, T, N, W, 1]
  target_seq_idx = tf.tile(
      tf.range(target_length)[None, :, None, None, None],
      [batch_size, 1, num_heads, attention_window, 1])
  # [B, T, N, W, 1]
  head_idx = tf.tile(
      tf.range(num_heads)[None, None, :, None, None],
      [batch_size, target_length, 1, attention_window, 1])
  # seq_idx: [B, T, N, W, 1]
  # [B, T, N, W, 4]
  scatter_idx = tf.concat([batch_idx, target_seq_idx, head_idx, seq_idx], -1)
  # [B, T, N, S]
  scattered_probs = tf.scatter_nd(
      scatter_idx, atten_probs,
      [batch_size, target_length, num_heads, source_length])
  return output, scattered_probs
