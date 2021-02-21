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
"""Tests for lingvo.core.entmax."""

from lingvo import compat as tf
from lingvo.core import entmax
from lingvo.core import test_utils


class EntmaxTest(test_utils.TestCase):

  # All expected values are generated based on official implementation.
  def test_entmax_support_generate_right_probability(self):
    inputs = tf.constant([[0.5, 1.0, 2.0]] * 3)
    expected_prob = tf.constant([[0.02328045, 0.16207013, 0.8146494]] * 3)
    entmax_prob = entmax.entmax_support(inputs, 1.5, -1)
    with self.session(use_gpu=False) as sess:
      output = sess.run(entmax_prob)
      self.assertAllClose(expected_prob, output)

  def test_entmax_loss_generate_right_loss(self):
    inputs = tf.constant([[[0.5, 1.0, 2.0]] * 3], dtype='bfloat16')
    labels = tf.constant([[0, 1, 2]])
    # Convert to the matrix with given depth, e.g. the vocabulary size.
    labels = tf.one_hot(labels, depth=3)
    expected_loss = tf.constant([[1.5642307, 1.0642307, 0.06423065]],
                                dtype='bfloat16')
    entmax_loss_val = entmax.entmax_loss(labels, inputs, alpha=1.5)
    with self.session(use_gpu=False) as sess:
      output = sess.run(entmax_loss_val)
      self.assertAllClose(expected_loss, output)

  def test_entmax_loss_generate_right_gradient(self):
    inputs = tf.constant([[0.5, 1.0, 2.0]] * 3)
    labels = tf.constant([0, 1, 2])
    expected_loss_gradient = tf.constant([[[-0.97671956, 0.16207013, 0.8146494],
                                           [0.02328045, -0.83792984, 0.8146494],
                                           [0.02328045, 0.16207013,
                                            -0.1853506]]])
    # Convert to the matrix with given depth, e.g. the vocabulary size.
    labels = tf.one_hot(labels, depth=3)
    expected_loss = tf.constant(2.692692)
    entmax_loss_val = tf.reduce_sum(entmax.entmax_loss(labels, inputs, 1.5))
    entmax_loss_gradient_val = tf.gradients(entmax_loss_val, inputs)

    with self.session(use_gpu=False) as sess:
      loss_output = sess.run(entmax_loss_val)
      gradient_output = sess.run(entmax_loss_gradient_val)
      self.assertAllClose(expected_loss, loss_output)
      self.assertAllClose(expected_loss_gradient, gradient_output)


if __name__ == '__main__':
  tf.test.main()
