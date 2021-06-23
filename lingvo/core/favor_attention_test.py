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
"""Tests for FAVOR attention."""

import math
from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import favor_attention as favor
from lingvo.core import test_utils

import numpy as np


class FAVORTest(test_utils.TestCase, parameterized.TestCase):

  def test_softmax_noncausal_attention_block_output(self):
    batch_size = 1
    length = 2
    num_heads = 1
    dim = 8
    num_random_features = 1000
    query = tf.random.normal([batch_size, length, num_heads, dim])
    key = tf.random.normal([batch_size, length, num_heads, dim])
    value = tf.random.normal([batch_size, length, num_heads, dim])
    kernel_transformation = favor.softmax_kernel_transformation
    projection_matrix = favor.create_projection_matrix(num_random_features, dim)
    query = tf.cast(query, tf.float64)
    key = tf.cast(key, tf.float64)
    value = tf.cast(value, tf.float64)
    projection_matrix = tf.cast(projection_matrix, tf.float64)
    attention_block_output = favor.favor_attention(query, key, value, None,
                                                   kernel_transformation, False,
                                                   projection_matrix)

    query = tf.multiply(query, 1.0 / math.sqrt(float(dim)))
    attention_scores = tf.einsum("BXHD,BYHD->BXYH", query, key)
    attention_scores = tf.nn.softmax(attention_scores, axis=2)
    exact_attention_block_output = tf.einsum("BXYH,BYHD->BXHD",
                                             attention_scores, value)
    max_error = 0.5
    with self.session(use_gpu=False) as sess:
      favor_output, groundtruth_output = sess.run(
          [exact_attention_block_output, attention_block_output])
      error = np.max(
          np.abs((groundtruth_output - favor_output) / groundtruth_output))
      self.assertLess(error, max_error)

  def test_cossim_noncausal_attention_block_output(self):
    batch_size = 1
    length = 2
    num_heads = 1
    dim = 8
    num_random_features = 1000
    query = tf.random.normal([batch_size, length, num_heads, dim])
    key = tf.random.normal([batch_size, length, num_heads, dim])
    value = tf.random.normal([batch_size, length, num_heads, dim])
    projection_matrix = favor.create_projection_matrix(num_random_features, dim)
    query = tf.cast(query, tf.float64)
    key = tf.cast(key, tf.float64)
    value = tf.cast(value, tf.float64)

    key_prime = favor.cossim_kernel_transformation(key, False,
                                                   projection_matrix, 0.0,
                                                   num_random_features)
    query_prime = favor.cossim_kernel_transformation(query, True,
                                                     projection_matrix, 0.0,
                                                     num_random_features)
    attention_scores = tf.einsum("BXHD,BYHD->BXYH", query_prime, key_prime)
    attention_scores = tf.nn.softmax(attention_scores, axis=2)
    attention_block_output = tf.einsum("BXYH,BYHD->BXHD", attention_scores,
                                       value)

    query = tf.math.l2_normalize(query, axis=[-1])
    key = tf.math.l2_normalize(key, axis=[-1])
    query = tf.multiply(query, math.sqrt(float(dim)))
    attention_scores = tf.einsum("BXHD,BYHD->BXYH", query, key)
    attention_scores = tf.nn.softmax(attention_scores, axis=2)
    exact_attention_block_output = tf.einsum("BXYH,BYHD->BXHD",
                                             attention_scores, value)
    max_error = 0.5
    with self.session(use_gpu=False) as sess:
      favor_output, groundtruth_output = sess.run(
          [exact_attention_block_output, attention_block_output])
      error = np.max(
          np.abs((groundtruth_output - favor_output) / groundtruth_output))
      self.assertLess(error, max_error)

  def test_chunked_causal_attention(self):
    batch_size = 1
    length = 128
    num_heads = 1
    dim = 16
    query = tf.random.normal([batch_size, length, num_heads, dim])
    key = tf.random.normal([batch_size, length, num_heads, dim])
    value = tf.random.normal([batch_size, length, num_heads, dim])
    query = tf.cast(query, tf.float64)
    key = tf.cast(key, tf.float64)
    value = tf.cast(value, tf.float64)
    attention_block_output = favor.favor_attention(
        query, key, value, None, favor.relu_kernel_transformation, True)

    chunked_attention_block_output = favor.favor_attention(
        query,
        key,
        value,
        None,
        favor.relu_kernel_transformation,
        True,
        use_chunked_causal=True)

    max_error = 0.0001
    with self.session(use_gpu=False) as sess:
      chunked_output, groundtruth_output = sess.run(
          [chunked_attention_block_output, attention_block_output])
      error = np.max(np.abs(groundtruth_output - chunked_output))
      self.assertLess(error, max_error)


if __name__ == "__main__":
  tf.test.main()
