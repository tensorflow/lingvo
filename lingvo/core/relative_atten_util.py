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
"""Utils for relative positional embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import compat as tf

from lingvo.core import py_utils


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
  b, t = shape[:2]
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


def MakeCausalPadding(seq_len, block_size, left_context, right_context):
  """Makes the causal padding tensor for a full sequence.

  Args:
    seq_len: int or scalar int tensor. Sequence length.
    block_size: int. Number of time frames in a block.
    left_context: int. Left context size.
    right_context: int. Right context size.

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

  padding = 1.0 - tf.cast(valid_atten, dtype=tf.float32)

  return padding


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


def _RelPositionBias(query, abs_pos_emb):
  """Computes relative position bias for general cases."""
  _, t, n, h = py_utils.GetShape(query)
  abs_pos_emb = py_utils.HasShape(abs_pos_emb, [2 * t - 1, n, h])

  # abs_pos_emb is [-(T-1), -(T-2), ... 0, 1, 2, ... T-1]
  # Change to [T-1, T-2, ... 0, -1, -2, ... -(T-2), -(T-1)]
  abs_pos_emb = tf.reverse(abs_pos_emb, [0])

  # [B, N, T, L=2T-1]
  term_bd = tf.einsum('BTNH,LNH->BNTL', query, abs_pos_emb)

  # Convert to [B, N, T, T]
  # part1
  term_bd_left = term_bd[:, :, :, :t]
  term_bd_left = tf.reverse(term_bd_left, [2, 3])
  term_bd_left = RelShift(term_bd_left)
  # [B, N, T, T]
  term_bd_left = tf.reverse(term_bd_left, [2, 3])
  # part 2
  term_bd_right = term_bd[:, :, :, t - 1:]
  # [B, N, T, T]
  term_bd_right = RelShift(term_bd_right)
  # [lower triangle]
  mask = tf.linalg.band_part(tf.ones_like(term_bd_right), -1, 0)

  # stitching togather
  return tf.where(mask > 0, term_bd_left, term_bd_right)


def _RelPositionBiasCausal(query, abs_pos_emb):
  """Computes relative position bias for causal self attention."""
  _, t, n, h = py_utils.GetShape(query)

  abs_pos_emb = py_utils.HasShape(abs_pos_emb, [2 * t - 1, n, h])

  # abs_pos_emb is [-(T-1), -(T-2), ... 0, 1, 2, ... T-1]
  # Retain only half and change order to [T-1, T-2, ... 0]
  # [T, N, H]
  abs_pos_emb = tf.reverse(abs_pos_emb, [0])[:t]

  # [B, N, T, L=T]
  term_bd = tf.einsum('BTNH,LNH->BNTL', query, abs_pos_emb)

  # Perform shifting.
  term_bd = tf.reverse(term_bd, [2, 3])
  term_bd = RelShift(term_bd)
  return tf.reverse(term_bd, [2, 3])


def RelPositionBias(content, abs_pos_emb, is_causal):
  """Compute relative position bias.

  This is a subroutine used by variants of self-attentions with relative
  positional embedding.

  B: batch size
  T: sequence length
  N: num of attention heads.
  H: per-head attention dimension.

  output[b][n][i][j] = content[b][i][n] x abs_pos_emb[i-j+T-1][n]

  Notice padding is supposed to be masked by the caller of this function.

  Args:
    tensors of the following shapes:
    content:         [B, T, N, H]
    abs_pos_emb:     [2T - 1, N, H], the absolute positional embedding.
      abs_pos_emb[i] is the emb of relative distance i - (T-1).
    is_causal: A Python bool or a scalar bool Tensor. True for causal self
      attention.

  Returns:
    The attention logits tensor. [B, N, T, T]
  """
  if not isinstance(is_causal, tf.Tensor):
    fn = (_RelPositionBiasCausal if is_causal else _RelPositionBias)
    res = fn(content, abs_pos_emb)
  else:
    res = tf.cond(is_causal,
                  lambda: _RelPositionBiasCausal(content, abs_pos_emb),
                  lambda: _RelPositionBias(content, abs_pos_emb))
  return res


def _AttenLogits(query,
                 key,
                 abs_pos_emb,
                 content_bias=None,
                 positional_bias=None,
                 is_causal=False):
  """Attention logits from ...

  Transformer-XL(https://arxiv.org/pdf/1901.02860.pdf, section 3.3) version of
  self attention with relative position embedding.

  Notice padding is supposed to be masked by the caller of this function.

  B: batch size
  T: sequence length
  N: num of attention heads.
  H: per-head attention dimension.

  Args:
    tensors of the following shapes:
    query:           [B, T, N, H]
    key:             [B, T, N, H]
    abs_pos_emb:     [2T - 1, N, H]. The sinusoid positional embedding from
    https://arxiv.org/abs/1706.03762. abs_pos_emb[i] is the emb of relative
    distance i - (T-1).
    content_bias:    [N, H] or None
    positional_bias: [N, H] or None
    is_causal: A Python bool or a scalar bool Tensor. True for causal self
    attention.

  Returns:
    The attention logits tensor. [B, N, T, T]
  """
  b, t, n, h = py_utils.GetShape(query)

  key = py_utils.HasShape(key, [b, t, n, h])
  if content_bias is not None:
    content_bias = py_utils.HasShape(content_bias, [n, h])
  else:
    content_bias = 0
  if positional_bias is not None:
    positional_bias = py_utils.HasShape(positional_bias, [n, h])
  else:
    positional_bias = 0

  # [B, N, T, S=T]
  term_ac = tf.einsum('BTNH,BSNH->BNTS', query + content_bias, key)
  term_bd = RelPositionBias(query + positional_bias, abs_pos_emb, is_causal)
  return term_ac + term_bd


def AttenLogitsTransformerXL(query,
                             key,
                             abs_pos_emb,
                             content_bias,
                             positional_bias,
                             is_causal=False):
  """Attention logits from ...

  Transformer-XL(https://arxiv.org/pdf/1901.02860.pdf, section 3.3) version of
  self attention with relative position embedding.

  Notice padding is supposed to be masked by the caller of this function.

  B: batch size
  T: sequence length
  N: num of attention heads.
  H: per-head attention dimension.

  Args:
    tensors of the following shapes:
    query:           [B, T, N, H]
    key:             [B, T, N, H]
    abs_pos_emb:     [2T - 1, N, H]. The sinusoid positional embedding from
    https://arxiv.org/abs/1706.03762. abs_pos_emb[i] is the emb of relative
    distance i - (T-1).
    content_bias:    [N, H]
    positional_bias: [N, H]
    is_causal: A Python bool or a scalar bool Tensor. True for causal self
    attention.

  Returns:
    The attention logits tensor. [B, N, T, T]
  """
  return _AttenLogits(query, key, abs_pos_emb, content_bias, positional_bias,
                      is_causal)


def AttenLogitsRPE(query, key, abs_pos_emb, is_causal):
  """Attention logits from ...

  https://arxiv.org/pdf/1803.02155.pdf with trainable rel position emb.

  Notice padding is supposed to be masked by the caller of this function.

  B: batch size
  T: sequence length
  N: num of attention heads.
  H: per-head attention dimension.

  Args:
    tensors of the following shapes:
    query:           [B, T, N, H]
    key:             [B, T, N, H]
    abs_pos_emb:     [2T - 1, N, H]. The trainable embdding. abs_pos_emb[i] is
      the emb of relative distance i - (T-1).
    is_causal: A Python bool or a scalar bool Tensor. True for causal self
      attention.

  Returns:
    The attention logits tensor. [B, N, T, T]
  """
  return _AttenLogits(query, key, abs_pos_emb, is_causal=is_causal)
