# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for spectrum augmenter layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo.core import spectrum_augmenter
from lingvo.core import test_utils
import numpy as np
from six.moves import range
import tensorflow as tf


class SpectrumAugmenterTest(test_utils.TestCase):

  def testSpectrumAugmenterWithTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.compat.v1.set_random_seed(127)
      batch_size = 5
      inputs = tf.ones([batch_size, 10, 2, 2], dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, i + 3]),
                       tf.ones([1, 8 - i])], axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_frames = 10
      p.random_seed = 12345
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterBasedDynamicTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.compat.v1.set_random_seed(127)
      batch_size = 5
      inputs = tf.ones([batch_size, 10, 2, 2], dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, i + 3]),
                       tf.ones([1, 8 - i])], axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_ratio = 0.4
      p.time_mask_count = 100
      p.use_dynamic_time_mask_max_frames = True
      p.random_seed = 12345
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithFrequencyMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.compat.v1.set_random_seed(1234)
      inputs = tf.ones([3, 5, 4, 2], dtype=tf.float32)
      paddings = tf.zeros([3, 5])
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 6
      p.time_mask_max_frames = 0
      p.random_seed = 12345
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[1., 1.], [0., 0.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [0., 0.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [0., 0.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [0., 0.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [0., 0.], [1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]]],
                                  [[[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterUnstacking(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.compat.v1.set_random_seed(1234)
      inputs = tf.ones([3, 5, 4, 2], dtype=tf.float32)
      paddings = tf.zeros([3, 5])
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.unstack = True
      p.stack_height = 2
      p.freq_mask_max_bins = 6
      p.time_mask_max_frames = 1
      p.random_seed = 12345
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[0., 0.], [1., 1.], [0., 0.], [1., 1.]],
                                   [[0., 0.], [1., 1.], [0., 0.], [1., 1.]],
                                   [[0., 0.], [1., 1.], [0., 0.], [1., 1.]],
                                   [[0., 0.], [1., 1.], [0., 0.], [1., 1.]],
                                   [[0., 0.], [1., 1.], [0., 0.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]]],
                                  [[[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)


if __name__ == '__main__':
  tf.test.main()
