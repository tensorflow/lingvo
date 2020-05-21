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
"""Tests for relative_atten_util."""


from absl.testing import parameterized

from lingvo import compat as tf
from lingvo.core import relative_atten_util
from lingvo.core import test_utils

import numpy as np

FLAGS = tf.flags.FLAGS


def OracleAttentionLogits(query, key, abs_pos_emb, content_bias,
                          positional_bias, is_causal):
  """Computes expected attention logits using non-vectorized approach."""
  batch, seqlen, num_heads, _ = query.shape
  tgtlen, srclen = seqlen, seqlen

  logits = np.zeros((batch, num_heads, tgtlen, srclen))

  very_negative = np.finfo(logits.dtype).min * 0.7

  for b in range(batch):
    for n in range(num_heads):
      for i in range(tgtlen):
        for j in range(srclen):
          if is_causal and j > i:
            logits[b][n][i][j] = very_negative
            continue
          # Non-causal case.
          offset = seqlen - 1
          pos_emb = abs_pos_emb[i - j + offset]
          logits[b][n][i][j] = (
              np.dot(query[b][i][n], key[b][j][n]) +
              np.dot(query[b][i][n], pos_emb[n]))
          if content_bias is not None:
            logits[b][n][i][j] += np.dot(content_bias[n], key[b][j][n])
          if positional_bias is not None:
            logits[b][n][i][j] += np.dot(positional_bias[n], pos_emb[n])
  return logits


class TransformerXLRelativeAttentionTest(test_utils.TestCase,
                                         parameterized.TestCase):

  def setUp(self):
    super(TransformerXLRelativeAttentionTest, self).setUp()
    self.input_dim = 32
    self.num_heads = 4
    self.batch = 4
    self.seqlen = 16

  def _GetTestInputs(self):
    np.random.seed(FLAGS.test_random_seed)
    query = 3 * np.random.rand(self.batch, self.seqlen, self.num_heads,
                               self.input_dim).astype(np.float32)
    key = 5 * np.random.rand(self.batch, self.seqlen, self.num_heads,
                             self.input_dim).astype(np.float32)
    abs_pos_emb = 7 * np.random.rand(2 * self.seqlen - 1, self.num_heads,
                                     self.input_dim).astype(np.float32)
    content_bias = 11 * np.random.rand(self.num_heads, self.input_dim).astype(
        np.float32)
    positional_bias = 13 * np.random.rand(self.num_heads,
                                          self.input_dim).astype(np.float32)
    return query, key, abs_pos_emb, content_bias, positional_bias

  def _Compare(self, expected, actual, is_causal):
    if is_causal:
      # 0s are masked positions
      mask = np.tril(np.ones(
          (self.seqlen, self.seqlen)), -1) + np.eye(self.seqlen, self.seqlen)
      expected = expected * mask
      actual = actual * mask
    self.assertAllClose(expected, actual)

  @parameterized.named_parameters(('Basic', False), ('Causal', True))
  def testTransformerXL(self, is_causal):
    (query, key, abs_pos_emb, content_bias,
     positional_bias) = self._GetTestInputs()
    expected = OracleAttentionLogits(query, key, abs_pos_emb, content_bias,
                                     positional_bias, is_causal)
    actual_t = relative_atten_util.AttenLogitsTransformerXL(
        query, key, abs_pos_emb, content_bias, positional_bias, is_causal)
    with self.session() as sess:
      actual = sess.run(actual_t)
    self._Compare(expected, actual, is_causal)

  @parameterized.named_parameters(('Basic', False), ('Causal', True))
  def testRPE(self, is_causal):
    (query, key, abs_pos_emb, _, _) = self._GetTestInputs()
    expected = OracleAttentionLogits(query, key, abs_pos_emb, None, None,
                                     is_causal)
    actual_t = relative_atten_util.AttenLogitsRPE(query, key, abs_pos_emb,
                                                  is_causal)
    with self.session() as sess:
      actual = sess.run(actual_t)
    self._Compare(expected, actual, is_causal)


class BlockUtilsTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_block', 7),
      ('one_frame_block', 1),
      ('two_frame_blocks', 2),
  )
  def testConvertToBlocks(self, block_size):
    x_val = np.random.random([2, 6, 2, 3, 4])
    with self.session() as sess:
      x = tf.convert_to_tensor(x_val, tf.float32)
      x_blocks = relative_atten_util.ConvertToBlocks(x, block_size)
      x_blocks_val = sess.run(x_blocks)
    # Check shape.
    batch_size = x_val.shape[0]
    other_dims = x_val.shape[2:]
    num_blocks = int(np.ceil(x_val.shape[1] / float(block_size)))
    expected_shape = (batch_size, num_blocks, block_size) + other_dims
    self.assertAllEqual(expected_shape, x_blocks_val.shape)

    # Check values.
    x_recover = x_blocks_val.reshape((x_blocks_val.shape[0], -1) +
                                     x_blocks_val.shape[3:])
    x_recover = x_recover[:, :x_val.shape[1], ...]
    self.assertAllClose(x_val, x_recover)

  @parameterized.named_parameters(
      ('single_block', 7, 2, 1),
      ('single_frame_context', 1, 1, 0),
      ('other_case_1', 3, 4, 1),
      ('other_case_2', 4, 2, 4),
  )
  def testExtractBlockContext(self, block_size, left_context, right_context):
    x_val = np.random.random([2, 6, 2, 3, 4])
    with self.session() as sess:
      x = tf.convert_to_tensor(x_val, tf.float32)
      x_context = relative_atten_util.ExtractBlockContext(
          x, block_size, left_context, right_context)
      x_context_val = sess.run(x_context)
    # Check shape.
    batch_size = x_val.shape[0]
    other_dims = x_val.shape[2:]
    num_blocks = int(np.ceil(x_val.shape[1] / float(block_size)))
    context_size = block_size + left_context - 1 + right_context
    expected_shape = (batch_size, num_blocks, context_size) + other_dims
    self.assertAllEqual(expected_shape, x_context_val.shape)

    # Check values block by block.
    for block_idx in range(num_blocks):
      context_start = block_idx * block_size - left_context + 1
      context_end = (block_idx + 1) * block_size + right_context
      slice_start = max(0, context_start)
      slice_end = min(x_val.shape[1], context_end)
      expected_val = x_val[:, slice_start:slice_end, ...]
      actual_val = x_context_val[:, block_idx, ...]
      # remove paddings
      front_padding = slice_start - context_start
      back_padding = context_end - slice_end
      actual_val = actual_val[:, front_padding:context_size - back_padding, ...]
      self.assertAllClose(expected_val, actual_val)

  def _getReferenceCausalPadding(self, seq_len, block_size, left_context,
                                 right_context):
    num_blocks = int(np.ceil(seq_len / float(block_size)))
    context_size = block_size + left_context - 1 + right_context
    padding = np.ones((num_blocks, block_size, context_size))

    for i in range(num_blocks):
      for j in range(block_size):
        actual_src_pos = j + i * block_size
        if actual_src_pos < seq_len:
          for k in range(context_size):
            actual_tgt_pos = k + i * block_size - (left_context - 1)
            if 0 <= actual_tgt_pos and actual_tgt_pos < seq_len:
              diff = actual_src_pos - actual_tgt_pos
              if -right_context <= diff and diff < left_context:
                padding[i, j, k] = 0

    return padding

  @parameterized.named_parameters(
      ('single_block', 6, 9, 2, 1),
      ('single_frame_block', 6, 1, 2, 1),
      ('single_frame_context', 6, 1, 1, 0),
      ('other_case_1', 6, 3, 4, 1),
      ('other_case_2', 6, 4, 2, 4),
  )
  def testMakeCausalPadding(self, seq_len, block_size, left_context,
                            right_context):
    with self.session() as sess:
      seq_len_t = tf.convert_to_tensor(seq_len)
      padding = relative_atten_util.MakeCausalPadding(seq_len_t, block_size,
                                                      left_context,
                                                      right_context)
      padding_val = sess.run(padding)

    ref_padding = self._getReferenceCausalPadding(seq_len, block_size,
                                                  left_context, right_context)
    self.assertAllEqual(ref_padding, padding_val)


if __name__ == '__main__':
  tf.test.main()
