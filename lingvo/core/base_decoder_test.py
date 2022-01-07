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
    batch_seed = tf.constant([0, 1, 1, 0])
    shape = [2, 3]
    with self.session(use_gpu=False) as sess:
      # Test time_step=0 and time_step=1.
      result0 = sess.run(
          base_decoder._BatchSampleGumbel(batch_seed, 0, shape, tf.float32))
      result1 = sess.run(
          base_decoder._BatchSampleGumbel(batch_seed, 1, shape, tf.float32))

      # The results should be the same if the seeds are the same.
      expected0 = [
          # seed=0
          [[-0.4420052, 1.055191, 1.500295], [1.3457806, 0.05632289,
                                              1.9743681]],
          # seed=1
          [[2.0436666, 0.25286624, 0.67267746],
           [0.06724094, -0.3174366, 0.20917037]],
          # seed=1
          [[2.0436666, 0.25286624, 0.67267746],
           [0.06724094, -0.3174366, 0.20917037]],
          # seed=0
          [[-0.4420052, 1.055191, 1.500295], [1.3457806, 0.05632289,
                                              1.9743681]],
      ]
      expected1 = [
          # seed=0
          [[-0.91589576, 0.11277056, 0.06884012],
           [-0.14081508, -0.05630323, 1.1648132]],
          # seed=1
          [[-0.84155184, -0.67374235, 0.2399045],
           [-0.805948, -0.45244274, -0.162385]],
          # seed=1
          [[-0.84155184, -0.67374235, 0.2399045],
           [-0.805948, -0.45244274, -0.162385]],
          # seed=0
          [[-0.91589576, 0.11277056, 0.06884012],
           [-0.14081508, -0.05630323, 1.1648132]],
      ]
      self.assertAllClose(expected0, result0)
      self.assertAllClose(expected1, result1)

  def testSampleGumbelWithMax(self):
    batch_size = 2
    num_hyps_per_beam = 5
    k = 7
    tgt_batch = num_hyps_per_beam * batch_size
    phi = np.random.rand(tgt_batch, k)
    target_max = np.random.rand(tgt_batch, 1)
    batch_seed = tf.constant([0, 1])
    time_step = 0

    with self.session(use_gpu=False) as sess:
      result = sess.run(
          base_decoder._SampleGumbelWithMax(phi, target_max, batch_seed,
                                            time_step))

      # The maximum values of the samples must be equal to `target_max`.
      self.assertAllClose(target_max, np.max(result, axis=1, keepdims=True))


if __name__ == '__main__':
  tf.test.main()
