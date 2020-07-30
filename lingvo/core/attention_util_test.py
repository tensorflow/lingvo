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
"""Tests for attention_util."""

from absl.testing import parameterized

from lingvo import compat as tf
from lingvo.core import attention_util
from lingvo.core import test_utils

import numpy as np

FLAGS = tf.flags.FLAGS


class RelPositionBiasTest(test_utils.TestCase, parameterized.TestCase):

  def testBasic(self):
    with self.session():
      t = 3
      # [BTNH].
      content = tf.linalg.diag(tf.ones([t]))[None, :, None, :]
      # [LNH].
      abs_pos_emb = tf.reshape(
          tf.range(t * (2 * t - 1), dtype=tf.float32), [2 * t - 1, 1, t])
      tf.logging.info('content=%s abs_pos_emb=%s', content.eval(),
                      abs_pos_emb.eval())
      self.assertAllClose([[[[6., 3., 0.], [10., 7., 4.], [14., 11., 8.]]]],
                          attention_util.RelPositionBias(content,
                                                         abs_pos_emb).eval())


def OracleAttentionLogits(query,
                          key,
                          abs_pos_emb,
                          content_bias,
                          positional_bias,
                          skip_term_b=False):
  """Computes expected attention logits using non-vectorized approach."""
  batch, seqlen, num_heads, _ = query.shape
  tgtlen, srclen = seqlen, seqlen

  logits = np.zeros((batch, num_heads, tgtlen, srclen))

  for b in range(batch):
    for n in range(num_heads):
      for i in range(tgtlen):
        for j in range(srclen):
          offset = seqlen - 1
          pos_emb = abs_pos_emb[i - j + offset]
          logits[b][n][i][j] = np.dot(query[b][i][n], key[b][j][n])
          if not skip_term_b:
            logits[b][n][i][j] += np.dot(query[b][i][n], pos_emb[n])
          if content_bias is not None:
            logits[b][n][i][j] += np.dot(content_bias[n], key[b][j][n])
          if positional_bias is not None:
            logits[b][n][i][j] += np.dot(positional_bias[n], pos_emb[n])
  return logits


class TransformerXLRelativeAttentionTest(test_utils.TestCase,
                                         parameterized.TestCase):

  def setUp(self):
    super().setUp()
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

  @parameterized.named_parameters(
      ('Base', False),
      ('Lite', True),
  )
  def testTransformerXL(self, skip_term_b):
    (query, key, abs_pos_emb, content_bias,
     positional_bias) = self._GetTestInputs()
    expected = OracleAttentionLogits(query, key, abs_pos_emb, content_bias,
                                     positional_bias, skip_term_b)
    actual_t = attention_util.AttenLogitsTransformerXL(query, key, abs_pos_emb,
                                                       content_bias,
                                                       positional_bias,
                                                       skip_term_b)
    with self.session() as sess:
      actual = sess.run(actual_t)
    self.assertAllClose(expected, actual)

  def testRPE(self):
    (query, key, abs_pos_emb, _, _) = self._GetTestInputs()
    expected = OracleAttentionLogits(query, key, abs_pos_emb, None, None)
    actual_t = attention_util.AttenLogitsRPE(query, key, abs_pos_emb)
    with self.session() as sess:
      actual = sess.run(actual_t)
    self.assertAllClose(expected, actual)


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
      x_blocks = attention_util.ConvertToBlocks(x, block_size)
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
      x_context = attention_util.ExtractBlockContext(x, block_size,
                                                     left_context,
                                                     right_context)
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
      padding = attention_util.MakeCausalPadding(seq_len_t, block_size,
                                                 left_context, right_context)
      padding_val = sess.run(padding)

    ref_padding = self._getReferenceCausalPadding(seq_len, block_size,
                                                  left_context, right_context)
    self.assertAllEqual(ref_padding, padding_val)


class KMeansClusteringForAttenTest(test_utils.TestCase):

  def testFProp(self):
    p = attention_util.KMeansClusteringForAtten.Params()
    p.name = 'k_means'
    p.num_clusters = 2
    p.dim_per_head = 4
    p.num_heads = 3
    batch_size = 5
    seq_length = 6
    x = np.random.rand(batch_size, seq_length, p.num_heads,
                       p.dim_per_head).astype(np.float32)
    k_means = p.Instantiate()

    with self.session():
      dists, loss = k_means.FProp(k_means.theta, x, update=True)
      self.evaluate(tf.global_variables_initializer())
      dists, loss = self.evaluate([dists, loss])
      self.assertEqual(dists.shape,
                       (batch_size, seq_length, p.num_heads, p.num_clusters))
      self.assertEqual(loss.shape, ())

  def testFPropFixedInput(self):
    p = attention_util.KMeansClusteringForAtten.Params()
    p.name = 'k_means'
    p.num_clusters = 3
    p.dim_per_head = 6
    p.num_heads = 4
    p.decay = 0.5
    k_means = p.Instantiate()
    batch_size = 2
    seq_length = 5

    with self.session():
      x = np.random.rand(batch_size, seq_length, p.num_heads,
                         p.dim_per_head).astype(np.float32)
      self.evaluate(tf.global_variables_initializer())
      fixed_loss = None
      for _ in range(10):
        dists, loss = k_means.FProp(k_means.theta, x, update=False)
        dists, loss = self.evaluate([dists, loss])
        if not fixed_loss:
          fixed_loss = loss
        else:
          # If we do not update, the loss remain fixed.
          self.assertEqual(loss, fixed_loss)
      prev_loss = fixed_loss
      self.evaluate(k_means.FProp(k_means.theta, x, update=True))
      for _ in range(5):
        dists, loss = k_means.FProp(k_means.theta, x, update=True)
        _, loss = self.evaluate([dists, loss])
        # If we update the centroids, the loss should strictly decrease.
        self.assertGreater(prev_loss - loss, 1e-5)
        prev_loss = loss

  def testFPropClustering(self):
    p = attention_util.KMeansClusteringForAtten.Params()
    p.name = 'k_means'
    p.num_clusters = 2
    p.dim_per_head = 3
    p.num_heads = 2
    p.decay = 0.8
    k_means = p.Instantiate()
    batch_size = 3
    seq_length = 3

    with self.session() as sess:
      self.evaluate(tf.global_variables_initializer())

      def _GenInput():
        # We randomly generate inputs such that head 0 is clustered
        # around (±1/√2, ±1/√2, ∓√2), while head 1 is clustered around
        # (∓-√2, ±1/√2, ±1/√2).
        noise = 0.05 * np.random.rand(batch_size, seq_length, p.num_heads,
                                      p.dim_per_head).astype(np.float32)
        x1 = np.random.binomial(1, 0.5, [batch_size, seq_length, 1, 1]) * 2 - 1
        x1 = np.tile(
            np.array([1., 1., -1.], dtype=np.float32),
            [batch_size, seq_length, 1, 1]) * x1
        x2 = np.random.binomial(1, 0.5, [batch_size, seq_length, 1, 1]) * 2 - 1
        x2 = np.tile(
            np.array([-1., 1., 1.], dtype=np.float32),
            [batch_size, seq_length, 1, 1]) * x2
        x = np.concatenate([x1, x2], axis=2) + noise
        return x.astype(np.float32)

      for _ in range(25):
        _, loss = sess.run(
            k_means.FProp(k_means.theta, _GenInput(), update=True))
      final_means = k_means.theta.means.eval()
    # We assert that the centroids are close to the true centers.
    self.assertAllClose(
        np.abs(final_means), [[[0.71, 0.71, 1.41], [0.71, 0.71, 1.41]],
                              [[1.41, 0.71, 0.71], [1.41, 0.71, 0.71]]],
        rtol=0.03,
        atol=0.03)
    self.assertLess(loss, 0.005)

  def testFPropPadding(self):
    p = attention_util.KMeansClusteringForAtten.Params()
    p.name = 'k_means'
    p.num_clusters = 2
    p.dim_per_head = 3
    p.num_heads = 1
    p.decay = 0.7
    k_means = p.Instantiate()
    batch_size = 3
    seq_length = 5

    with self.session() as sess:
      self.evaluate(tf.global_variables_initializer())

      def _GenInput():
        # We randomly generate inputs such that inputs are clustered
        # around (±1/√2, ±1/√2, ∓√2) or (∓-√2, ±1/√2, ±1/√2) with one of them
        # hidden by padding.
        paddings = np.random.binomial(1, 0.5, [batch_size, seq_length]).astype(
            np.float32)
        x = np.expand_dims(np.expand_dims(paddings, axis=-1), axis=-1)
        # When padding is 0, we generate (∓1, 1, ±1); when padding is 1, we
        # generate (±1, 1, ∓1).
        x = np.concatenate([2 * x - 1, np.ones_like(x), 1 - 2 * x], axis=-1)
        x *= np.random.binomial(1, 0.5, [batch_size, seq_length, 1, 1]) * 2 - 1
        return x, paddings

      for _ in range(30):
        x, paddings = _GenInput()
        self.assertEqual(x.shape,
                         (batch_size, seq_length, p.num_heads, p.dim_per_head))
        self.assertEqual(paddings.shape, (batch_size, seq_length))
        _, loss1 = sess.run(
            k_means.FProp(k_means.theta, x, paddings, update=True))
      means1 = k_means.theta.means.eval()

      # We reverse the padding to hide the other half.
      for _ in range(40):
        x, paddings = _GenInput()
        _, loss2 = sess.run(
            k_means.FProp(k_means.theta, x, 1.0 - paddings, update=True))
      means2 = k_means.theta.means.eval()

      # We compute the loss using the previous input centering on
      # different centroids. The squared distance should be 3.
      _, loss3 = sess.run(
          k_means.FProp(k_means.theta, x, paddings, update=False))

    self.assertAllClose(
        np.abs(means1), [[[1.41, 0.71, 0.71], [1.41, 0.71, 0.71]]],
        rtol=0.03,
        atol=0.03)
    self.assertLess(loss1, 1e-5)
    self.assertAllClose(
        np.abs(means2), [[[0.71, 0.71, 1.41], [0.71, 0.71, 1.41]]],
        rtol=0.03,
        atol=0.03)
    self.assertLess(loss2, 1e-5)

    self.assertAllClose(loss3, 3.0, 1e-4, 1e-4)

  def testFPropClusteringEmptyCluster(self):
    p = attention_util.KMeansClusteringForAtten.Params()
    p.name = 'k_means'
    p.num_clusters = 10
    p.dim_per_head = 3
    p.num_heads = 1
    p.decay = 0.8
    k_means = p.Instantiate()
    batch_size = 3
    seq_length = 4

    # All of our inputs are (1, 1, -1)
    x = np.ones([batch_size, seq_length, 1, 1], dtype=np.float32)
    x = np.concatenate([x, x, -x], axis=-1)
    with self.session() as sess:
      self.evaluate(tf.global_variables_initializer())
      for _ in range(30):
        dists, loss = sess.run(k_means.FProp(k_means.theta, x, update=True))
      means = k_means.theta.means.eval()
    idx = np.argmin(dists, axis=-1)
    idx_1 = idx[0, 0, 0]
    # We assert that 'dists' achieves minimum all at the same cluster.
    self.assertAllEqual(idx,
                        np.array(idx_1 * np.ones([batch_size, seq_length, 1])))
    # We assert that at this index the centroid is close to (1/√2, 1/√2, -√2).
    means = np.squeeze(means[:, idx_1, :])
    self.assertAllClose(means, [0.71, 0.71, -1.41], rtol=0.03, atol=0.03)
    self.assertLess(loss, 1e-4)


if __name__ == '__main__':
  tf.test.main()
