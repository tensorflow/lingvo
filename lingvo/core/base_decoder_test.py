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
"""Tests for base_decoder."""

from lingvo import compat as tf
from lingvo.core import base_decoder
from lingvo.core import test_utils

import numpy as np


class BaseDecoderTest(test_utils.TestCase):

  def testKeepTopP(self):
    sorted_probs = tf.constant([
        [0.3, 0.2, 0.1],
        [0.3, 0.2, 0.1],
        [0.3, 0.2, 0.1],
        [0.3, 0.2, 0.1],
    ])
    sorted_log_probs = tf.math.log(sorted_probs)
    p = tf.constant([1.0, 0.5, 0.4, 0.0])

    with self.session(use_gpu=False) as sess:
      filtered_sorted_log_probs = base_decoder._KeepTopP(sorted_log_probs, p)
      result = sess.run(filtered_sorted_log_probs)

      filtered = np.exp(base_decoder.LARGE_NEGATIVE_NUMBER)
      expected_probs = [
          [0.3, 0.2, 0.1],  # p = 1.0
          [0.3, 0.2, filtered],  # p = 0.5
          [0.3, 0.2, filtered],  # p = 0.4
          [0.3, filtered, filtered],  # p = 0.0 (first element is always kept)
      ]
      self.assertAllClose(expected_probs, np.exp(result))

  def testBatchScatter(self):
    batch_size = 3
    n = 7
    default_tensor = np.zeros([batch_size, n], dtype=np.float32)
    indices = tf.constant([
        [0, 2],
        [1, 6],
        [3, 1],
    ])
    values = tf.constant([
        [7, 9],
        [6, 8],
        [5, 3],
    ], dtype=tf.float32)

    with self.session(use_gpu=False) as sess:
      result = sess.run(
          base_decoder._BatchScatter(default_tensor, indices, values))

      # Expectation: (i,indices[i][j])-element of `default_tensor` has been
      # updated to values[i][j].
      expected = [
          [7, 0, 9, 0, 0, 0, 0],
          [0, 6, 0, 0, 0, 0, 8],
          [0, 3, 0, 5, 0, 0, 0],
      ]
      self.assertAllClose(expected, result)

  def testBatchLookup(self):
    table_keys = tf.constant([
        [27, 63, 72],
        [11, 58, 13],
    ])
    table_values = tf.constant([
        [77, 88, 99],
        [11, 22, 33],
    ])
    keys = [
        [63],  # matches table_keys[0][1]
        [13],  # matches table_keys[1][2]
    ]

    with self.session(use_gpu=False) as sess:
      result = sess.run(
          base_decoder._BatchLookup(keys, table_keys, table_values))
      expected = [
          [88],  # table_values[0][1]
          [33],  # table_values[1][2]
      ]
      self.assertAllClose(expected, result)

  def testBatchSampleGumbel(self):
    batch_seed = tf.constant([
        # seed = 0
        0,
        0,
        0,
        0,
        # seed = 1
        1,
        1,
        1,
        1,
    ])

    src_ids_pattern_a = [0, 1, 2, 3, 4]
    src_ids_pattern_b = [7, 2, 5, 8, 1, 3]
    # src_ids with alternating Pattern A and B. The three-digit elements
    # correspond to paddings, which shouldn't affect the results.
    src_ids = tf.constant([
        # seed = 0
        src_ids_pattern_a + [111, 111, 111],  # Pattern A.
        src_ids_pattern_b + [222, 222],  # Pattern B.
        src_ids_pattern_a + [333, 333, 333],  # Pattern A.
        src_ids_pattern_b + [444, 444],  # Pattern B.
        # seed = 1
        src_ids_pattern_a + [555, 555, 555],  # Pattern A.
        src_ids_pattern_b + [666, 666],  # Pattern B.
        src_ids_pattern_a + [777, 777, 777],  # Pattern A.
        src_ids_pattern_b + [888, 888],  # Pattern B.
    ])
    src_paddings = tf.cast(src_ids >= 100, tf.float32)

    shape = [2, 3]
    with self.session(use_gpu=False) as sess:
      # Test time_step=0 and time_step=1.
      result_time_0 = sess.run(
          base_decoder._BatchSampleGumbel(batch_seed, 0, src_ids, src_paddings,
                                          shape, tf.float32))
      result_time_1 = sess.run(
          base_decoder._BatchSampleGumbel(batch_seed, 1, src_ids, src_paddings,
                                          shape, tf.float32))

      # time_step=0, seed=0, src_ids="Pattern A"
      expected_time_0_seed_0_src_ids_a = [[-0.8168449, 1.2931604, -1.1208376],
                                          [0.5848354, 2.465711, -0.21484822]]
      # time_step=0, seed=0, src_ids="Pattern B"
      expected_time_0_seed_0_src_ids_b = [[-0.2371598, -1.5676343, -0.5775582],
                                          [0.16091877, 0.4489609, -0.5339234]]
      # time_step=0, seed=1, src_ids="Pattern A"
      expected_time_0_seed_1_src_ids_a = [[1.8465917, -0.38307178, 2.9526122],
                                          [3.151248, 0.4383798, -1.0428888]]
      # time_step=0, seed=1, src_ids="Pattern B"
      expected_time_0_seed_1_src_ids_b = [[0.2466429, 0.8616346, -0.43274638],
                                          [0.03198622, -0.16034941, 0.5214975]]
      # Expectation: #0-#2, #1-#3, #4-#6, #5-#7 items are the same.
      self.assertAllClose(
          result_time_0,
          [
              # seed = 0
              expected_time_0_seed_0_src_ids_a,
              expected_time_0_seed_0_src_ids_b,
              expected_time_0_seed_0_src_ids_a,
              expected_time_0_seed_0_src_ids_b,
              # seed = 1
              expected_time_0_seed_1_src_ids_a,
              expected_time_0_seed_1_src_ids_b,
              expected_time_0_seed_1_src_ids_a,
              expected_time_0_seed_1_src_ids_b,
          ])

      # time_step=1, seed=0, src_ids="Pattern A"
      expected_time_1_seed_0_src_ids_a = [[1.7323364, -1.0477272, 0.01648759],
                                          [0.11775304, -1.1763966, 0.46151116]]
      # time_step=1, seed=0, src_ids="Pattern B"
      expected_time_1_seed_0_src_ids_b = [[-1.0528497, 1.0473464, 2.8928924],
                                          [-1.2340496, 2.741108, -0.8313949]]
      # time_step=1, seed=1, src_ids="Pattern A"
      expected_time_1_seed_1_src_ids_a = [[-0.80684525, -0.3300631, -0.4008211],
                                          [1.1481359, -0.5537453, 1.6473633]]
      # time_step=1, seed=1, src_ids="Pattern B"
      expected_time_1_seed_1_src_ids_b = [[-0.635313, 0.6546277, -1.1632406],
                                          [-0.0557764, 0.60781074, -0.4813231]]
      # Expectation: #0-#2, #1-#3, #4-#6, #5-#7 items are the same.
      self.assertAllClose(
          result_time_1,
          [
              # seed = 0
              expected_time_1_seed_0_src_ids_a,
              expected_time_1_seed_0_src_ids_b,
              expected_time_1_seed_0_src_ids_a,
              expected_time_1_seed_0_src_ids_b,
              # seed = 1
              expected_time_1_seed_1_src_ids_a,
              expected_time_1_seed_1_src_ids_b,
              expected_time_1_seed_1_src_ids_a,
              expected_time_1_seed_1_src_ids_b,
          ])

  def testBatchSampleGumbel_SameResultsForSameSrcIdsWithDifferentPaddings(self):
    batch_seed = tf.constant([0, 1])
    shape = [2, 3]

    # src_ids1 and src_ids2 represent the same ID sequences though the number of
    # paddings are different.
    x = 999  # Padding.
    src_ids1 = tf.constant([
        [2, 5, 8, 1, x],  # seed = 0
        [3, 8, x, x, x],  # seed = 1
    ])
    src_ids2 = tf.constant([
        [2, 5, 8, 1, x, x, x, x],  # seed = 0
        [3, 8, x, x, x, x, x, x],  # seed = 1
    ])
    src_paddings1 = tf.cast(tf.equal(src_ids1, x), tf.float32)
    src_paddings2 = tf.cast(tf.equal(src_ids2, x), tf.float32)

    with self.session(use_gpu=False) as sess:
      # Run twice with the same batch_seed and time_step (= 0).
      result1 = sess.run(
          base_decoder._BatchSampleGumbel(batch_seed, 0, src_ids1,
                                          src_paddings1, shape, tf.float32))
      result2 = sess.run(
          base_decoder._BatchSampleGumbel(batch_seed, 0, src_ids2,
                                          src_paddings2, shape, tf.float32))

      # Expectation: Both runs have the same result, because src_ids1 and
      # src_ids2 represent the same ID sequence.
      self.assertAllClose(result1, result2)

  def testSampleGumbelWithMax(self):
    batch_size = 2
    num_hyps_per_beam = 5
    k = 7
    tgt_batch = num_hyps_per_beam * batch_size
    phi = np.random.rand(tgt_batch, k)
    target_max = np.random.rand(tgt_batch, 1)
    batch_seed = tf.constant([0, 1])
    time_step = 0

    vocab_size = 10
    src_len = 11
    src_ids = np.random.randint(
        low=1, high=vocab_size, size=[batch_size, src_len], dtype=np.int32)
    src_paddings = np.zeros([batch_size, src_len])  # No padding.

    with self.session(use_gpu=False) as sess:
      result = sess.run(
          base_decoder._SampleGumbelWithMax(phi, target_max, batch_seed,
                                            time_step, src_ids, src_paddings))

      # The maximum values of the samples must be equal to `target_max`.
      self.assertAllClose(target_max, np.max(result, axis=1, keepdims=True))


if __name__ == '__main__':
  test_utils.main()
