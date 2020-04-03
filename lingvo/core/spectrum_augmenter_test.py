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

import lingvo.compat as tf
from lingvo.core import spectrum_augmenter
from lingvo.core import test_utils
import numpy as np
from six.moves import range


class SpectrumAugmenterTest(test_utils.TestCase):

  def testSpectrumAugmenterWithTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(127)
      batch_size = 5
      inputs = tf.ones([batch_size, 20, 2, 2], dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, i + 12]),
                       tf.ones([1, 8 - i])], axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_frames = 5
      p.time_mask_count = 2
      p.time_mask_max_ratio = 1.0
      p.random_seed = 23456
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterDynamicSizeTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(127)
      batch_size = 3
      inputs = tf.ones([batch_size, 20, 2, 2], dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, 8 * i + 3]),
                       tf.ones([1, 17 - 8 * i])],
                      axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_ratio = 0.4
      p.time_mask_count = 1
      p.use_dynamic_time_mask_max_frames = True
      p.random_seed = 12345
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterDynamicMultiplicityTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(127)
      batch_size = 4
      inputs = tf.ones([batch_size, 22, 2, 2], dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, 5 * i + 5]),
                       tf.ones([1, 16 - 5 * i])],
                      axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_frames = 5
      p.time_mask_count = 10
      p.time_masks_per_frame = 0.2
      p.random_seed = 67890
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterDynamicSizeAndMultiplicityTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(127)
      batch_size = 4
      inputs = tf.ones([batch_size, 22, 2, 2], dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, 5 * i + 5]),
                       tf.ones([1, 16 - 5 * i])],
                      axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_frames = 5
      p.time_mask_count = 10
      p.time_masks_per_frame = 0.2
      p.time_mask_max_ratio = 0.4
      p.use_dynamic_time_mask_max_frames = True
      p.random_seed = 67890
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
                                  [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithFrequencyMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(1234)
      inputs = tf.ones([3, 5, 10, 1], dtype=tf.float32)
      paddings = tf.zeros([3, 5])
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 6
      p.freq_mask_count = 2
      p.time_mask_max_frames = 0
      p.random_seed = 34567
      specaug_layer = p.Instantiate()
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      expected_output = np.array(
          [[[[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]],
            [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]],
            [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]],
            [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]],
            [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]]],
           [[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [1.]]],
           [[[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]],
            [[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]],
            [[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]],
            [[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]],
            [[1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [1.]]]])
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWarpMatrixConstructor(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      inputs = tf.broadcast_to(tf.cast(tf.range(10), dtype=tf.float32), (4, 10))
      origin = tf.cast([2, 4, 4, 5], dtype=tf.float32)
      destination = tf.cast([3, 2, 6, 8], dtype=tf.float32)
      choose_range = tf.cast([4, 8, 8, 10], dtype=tf.float32)
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      specaug_layer = p.Instantiate()
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      expected_output = np.array(
          [[0.0000000, 0.6666667, 1.3333333, 2.0000000, 4.0000000,
            5.0000000, 6.0000000, 7.0000000, 8.0000000, 9.0000000],
           [0.0000000, 2.0000000, 4.0000000, 4.6666667, 5.3333333,
            6.0000000, 6.6666667, 7.3333333, 8.0000000, 9.0000000],
           [0.0000000, 0.6666667, 1.3333333, 2.0000000, 2.6666667,
            3.3333333, 4.0000000, 6.0000000, 8.0000000, 9.0000000],
           [0.0000000, 0.6250000, 1.2500000, 1.8750000, 2.5000000,
            3.1250000, 3.7500000, 4.3750000, 5.0000000, 7.5000000]])
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      warp_matrix = specaug_layer._ConstructWarpMatrix(
          batch_size=4,
          matrix_size=10,
          origin=origin,
          destination=destination,
          choose_range=choose_range,
          dtype=tf.float32)
      outputs = tf.einsum('bij,bj->bi', warp_matrix, inputs)
      actual_layer_output = sess.run(outputs)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithTimeWarping(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(1234)
      inputs = tf.broadcast_to(tf.cast(tf.range(10), dtype=tf.float32), (3, 10))
      inputs = tf.expand_dims(tf.expand_dims(inputs, -1), -1)
      paddings = []
      for i in range(3):
        paddings.append(
            tf.concat([tf.zeros([1, i + 7]),
                       tf.ones([1, 3 - i])], axis=1))
      paddings = tf.concat(paddings, axis=0)
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_frames = 0
      p.time_warp_max_frames = 8
      p.time_warp_max_ratio = 1.0
      p.time_warp_bound = 'static'
      p.random_seed = 34567
      specaug_layer = p.Instantiate()
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      expected_output = np.array(
          [[[[0.0000000]], [[0.6666667]], [[1.3333334]], [[2.0000000]],
            [[2.6666667]], [[3.3333335]], [[4.0000000]], [[7.0000000]],
            [[8.0000000]], [[9.0000000]]],
           [[[0.0000000]], [[3.0000000]], [[6.0000000]], [[6.3333334]],
            [[6.6666665]], [[7.0000000]], [[7.3333334]], [[7.6666667]],
            [[8.0000000]], [[9.0000000]]],
           [[[0.0000000]], [[0.5000000]], [[1.0000000]], [[1.5000000]],
            [[2.0000000]], [[3.4000000]], [[4.8000000]], [[6.2000000]],
            [[7.6000000]], [[9.0000000]]]])
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithDynamicTimeWarping(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(1234)
      inputs = tf.broadcast_to(tf.cast(tf.range(10), dtype=tf.float32), (3, 10))
      inputs = tf.expand_dims(tf.expand_dims(inputs, -1), -1)
      paddings = []
      for i in range(3):
        paddings.append(
            tf.concat([tf.zeros([1, 2 * i + 5]),
                       tf.ones([1, 5 - 2 * i])],
                      axis=1))
      paddings = tf.concat(paddings, axis=0)
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_frames = 0
      p.time_warp_max_ratio = 0.5
      p.time_warp_bound = 'dynamic'
      p.random_seed = 34567
      specaug_layer = p.Instantiate()
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      expected_output = np.array(
          [[[[0.0000000]], [[1.0000000]], [[2.0000000]], [[3.0000000]],
            [[4.0000000]], [[5.0000000]], [[6.0000000]], [[7.0000000]],
            [[8.0000000]], [[9.0000000]]],
           [[[0.0000000]], [[0.8333333]], [[1.6666666]], [[2.5000000]],
            [[3.3333333]], [[4.1666665]], [[5.0000000]], [[7.0000000]],
            [[8.0000000]], [[9.0000000]]],
           [[[0.0000000]], [[2.0000000]], [[2.8750000]], [[3.7500000]],
            [[4.6250000]], [[5.5000000]], [[6.3750000]], [[7.2500000]],
            [[8.1250000]], [[9.0000000]]]])
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterUnstacking(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(1234)
      inputs = tf.ones([3, 5, 10, 1], dtype=tf.float32)
      paddings = tf.zeros([3, 5])
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.unstack = True
      p.stack_height = 2
      p.freq_mask_max_bins = 5
      p.time_mask_max_frames = 8
      p.random_seed = 12345
      specaug_layer = p.Instantiate()
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      expected_output = np.array(
          [[[[1.], [1.], [0.], [1.], [1.], [1.], [1.], [0.], [1.], [1.]],
            [[1.], [1.], [0.], [1.], [1.], [0.], [0.], [0.], [0.], [0.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
            [[1.], [1.], [0.], [1.], [1.], [1.], [1.], [0.], [1.], [1.]],
            [[1.], [1.], [0.], [1.], [1.], [1.], [1.], [0.], [1.], [1.]]],
           [[[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]],
            [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]],
            [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]],
            [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]],
            [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]],
           [[[1.], [1.], [0.], [1.], [1.], [1.], [1.], [0.], [1.], [1.]],
            [[1.], [1.], [0.], [1.], [1.], [1.], [1.], [0.], [1.], [1.]],
            [[1.], [1.], [0.], [1.], [1.], [0.], [0.], [0.], [0.], [0.]],
            [[0.], [0.], [0.], [0.], [0.], [1.], [1.], [0.], [1.], [1.]],
            [[1.], [1.], [0.], [1.], [1.], [1.], [1.], [0.], [1.], [1.]]]])
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithPerDomainPolicyFreqMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(1234)
      inputs = tf.ones([6, 5, 4, 2], dtype=tf.float32)
      input_domain_ids = tf.constant(
          [[1] * 5, [2] * 5, [0] * 5, [2] * 5, [0] * 5, [1] * 5],
          dtype=tf.float32)
      paddings = tf.zeros([3, 5])
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.domain_ids = [0, 1, 2]
      p.freq_mask_max_bins = [0, 3, 8]
      p.time_mask_max_frames = 0
      p.random_seed = 1234
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[0., 0.], [0., 0.], [1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.], [1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.], [1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.], [1., 1.], [1., 1.]],
                                   [[0., 0.], [0., 0.], [1., 1.], [1., 1.]]],
                                  [[[1., 1.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[1., 1.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[1., 1.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[1., 1.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[1., 1.], [0., 0.], [0., 0.], [0., 0.]]],
                                  [[[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]]],
                                  [[[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]],
                                  [[[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]],
                                   [[1., 1.], [1., 1.], [1., 1.], [1., 1.]]],
                                  [[[1., 1.], [0., 0.], [0., 0.], [1., 1.]],
                                   [[1., 1.], [0., 0.], [0., 0.], [1., 1.]],
                                   [[1., 1.], [0., 0.], [0., 0.], [1., 1.]],
                                   [[1., 1.], [0., 0.], [0., 0.], [1., 1.]],
                                   [[1., 1.], [0., 0.], [0., 0.], [1., 1.]]]])
      h, _ = specaug_layer.FPropDefaultTheta(
          inputs, paddings, domain_ids=input_domain_ids)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterNoisify(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(127)
      batch_size = 2
      inputs = tf.ones([batch_size, 20, 2, 2], dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, 8 * i + 3]),
                       tf.ones([1, 17 - 8 * i])],
                      axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_ratio = 0.4
      p.time_mask_count = 1
      p.use_dynamic_time_mask_max_frames = True
      p.use_noise = True
      p.gaussian_noise = False
      p.random_seed = 12345
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[-0.00113627, -0.00113627],
                                    [0.08975883, 0.08975883]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]]],
                                  [[[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[0.09341543, 0.09341543],
                                    [-0.11914382, -0.11914382]],
                                   [[0.04238122, 0.04238122],
                                    [0.115249, 0.115249]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterGaussianNoisify(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(127)
      batch_size = 2
      inputs = tf.ones([batch_size, 20, 2, 2], dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, 8 * i + 3]),
                       tf.ones([1, 17 - 8 * i])],
                      axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_ratio = 0.4
      p.time_mask_count = 1
      p.use_dynamic_time_mask_max_frames = True
      p.use_noise = True
      p.gaussian_noise = True
      p.random_seed = 12345
      specaug_layer = p.Instantiate()
      expected_output = np.array([[[[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[-0.00798237, -0.00798237],
                                    [0.6305642, 0.6305642]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]]],
                                  [[[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[0.6562522, 0.6562522],
                                    [-0.83699656, -0.83699656]],
                                   [[0.29773206, 0.29773206],
                                    [0.8096351, 0.8096351]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]],
                                   [[1.00000000, 1.00000000],
                                    [1.00000000, 1.00000000]]]])
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithStatelessRandomOps(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      batch_size = 5
      inputs1 = tf.random_uniform(
          shape=[batch_size, 20, 2, 2], minval=0, maxval=1, dtype=tf.float32)
      inputs2 = tf.random_uniform(
          shape=[batch_size, 20, 2, 2], minval=0, maxval=1, dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, i + 12]),
                       tf.ones([1, 8 - i])], axis=1))
      paddings = tf.concat(paddings, axis=0)

      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_count = 1
      p.freq_mask_max_bins = 1
      p.time_mask_max_frames = 5
      p.time_mask_count = 2
      p.time_mask_max_ratio = 1.0
      p.use_input_dependent_random_seed = True
      specaug_layer = p.Instantiate()
      h1, _ = specaug_layer.FPropDefaultTheta(inputs1, paddings)
      h2, _ = specaug_layer.FPropDefaultTheta(inputs2, paddings)
      actual_layer_output1, actual_layer_output2 = sess.run([h1, h2])
      self.assertAllEqual(
          np.shape(actual_layer_output1), np.array([5, 20, 2, 2]))
      self.assertNotAllEqual(actual_layer_output1, actual_layer_output2)

  def testSpectrumAugmenterWithCalibration(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(1234)
      inputs = tf.ones([2, 5, 10, 1], dtype=tf.float32)
      paddings = tf.zeros([2, 5])
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 6
      p.freq_mask_count = 2
      p.time_mask_max_frames = 0
      p.random_seed = 34567
      p.use_calibration = True
      specaug_layer = p.Instantiate()
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      expected_output = np.array(
          [[[[2.5], [2.5], [2.5], [0.], [0.], [0.], [0.], [0.], [0.], [2.5]],
            [[2.5], [2.5], [2.5], [0.], [0.], [0.], [0.], [0.], [0.], [2.5]],
            [[2.5], [2.5], [2.5], [0.], [0.], [0.], [0.], [0.], [0.], [2.5]],
            [[2.5], [2.5], [2.5], [0.], [0.], [0.], [0.], [0.], [0.], [2.5]],
            [[2.5], [2.5], [2.5], [0.], [0.], [0.], [0.], [0.], [0.], [2.5]]],
           [[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [5.], [5.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [5.], [5.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [5.], [5.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [5.], [5.]],
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [5.], [5.]]]])
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = sess.run(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)


if __name__ == '__main__':
  tf.test.main()
