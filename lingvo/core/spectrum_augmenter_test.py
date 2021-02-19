# Lint as: python3
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

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import spectrum_augmenter
from lingvo.core import test_utils
import numpy as np


class SpectrumAugmenterTest(test_utils.TestCase, parameterized.TestCase):

  def testSpectrumAugmenterWithTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(127)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterDynamicSizeTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(127)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterDynamicMultiplicityTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(127)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterDynamicSizeAndMultiplicityTimeMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(127)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithFrequencyMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWarpMatrixConstructor(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
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
      actual_layer_output = self.evaluate(outputs)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithFreqWarping(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
      inputs = tf.broadcast_to(
          tf.cast(tf.range(8), dtype=tf.float32), (5, 1, 8))
      inputs = tf.expand_dims(inputs, -1)
      paddings = tf.zeros([3, 2])
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_mask_max_bins = 0
      p.time_mask_max_frames = 0
      p.freq_warp_max_bins = 4
      p.time_warp_max_frames = 0
      p.random_seed = 345678
      specaug_layer = p.Instantiate()
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      expected_output = np.array(
          [[[0.0, 4.0, 4.5714283, 5.142857, 5.714286, 6.285714, 6.8571434,
             3.999998]],
           [[0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 5.3333335, 6.6666665]],
           [[0.0, 0.6666667, 1.3333334, 2.0, 3.2, 4.4, 5.6000004, 6.8]],
           [[0.0, 1.3333334, 2.6666667, 4.0, 4.8, 5.6000004, 6.3999996,
             5.5999947]],
           [[0.0, 2.0, 2.857143, 3.7142859, 4.571429, 5.4285717, 6.2857146,
             5.999997]]])
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = self.evaluate(tf.squeeze(h, -1))
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithTimeWarping(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithDynamicTimeWarping(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  @parameterized.named_parameters(('Base', 0), ('WarmUp', 2))
  def testSpectrumAugmenterWithFreqNoise(self, warmup_steps):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
      inputs = tf.broadcast_to(
          tf.cast(tf.range(8), dtype=tf.float32), (5, 1, 8))
      inputs = tf.expand_dims(inputs, -1)
      paddings = tf.zeros([3, 2])
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      p.freq_noise_max_stddev = 0.1
      p.freq_mask_max_bins = 0
      p.time_mask_max_frames = 0
      p.freq_warp_max_bins = 0
      p.time_warp_max_frames = 0
      p.freq_noise_warmup_steps = warmup_steps
      p.random_seed = 345678
      specaug_layer = p.Instantiate()
      has_warmup_steps = warmup_steps != 0
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      if has_warmup_steps:
        expected_output = np.array(
            [[[0.      , 1.015837, 1.938253, 3.044222, 3.972092,
               5.028179, 6.046204, 7.000574]],
             [[0.      , 0.945328, 2.011299, 3.141700, 3.898298,
               4.969969, 6.153418, 6.860528]],
             [[0.      , 1.000181, 2.014447, 2.984220, 3.929903,
               5.016436, 6.002871, 6.995663]],
             [[0.      , 1.022656, 2.062904, 2.882567, 4.150898,
               4.98871 , 5.857354, 6.989822]],
             [[0.      , 1.002751, 2.009730, 3.000584, 3.984322,
               4.995187, 6.032375, 7.020331 ]]])
      else:
        expected_output = np.array(
            [[[0.      , 1.031674, 1.876506, 3.088444, 3.944183,
               5.056358, 6.092408, 7.001149]],
             [[0.      , 0.890657, 2.022598, 3.283400, 3.796596,
               4.939937, 6.306836, 6.721056]],
             [[0.      , 1.000362, 2.028894, 2.968441, 3.859807,
               5.032872, 6.005741, 6.991328]],
             [[0.      , 1.045312, 2.125809, 2.765134, 4.301796,
               4.97742 , 5.714708, 6.979644]],
             [[0.      , 1.005502, 2.019461, 3.001168, 3.968645,
               4.990373, 6.064750, 7.040662]]])
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      global_step = py_utils.GetGlobalStep()
      update_global_step = tf.assign(global_step, global_step + 1)
      with tf.control_dependencies([update_global_step]):
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
      actual_layer_output = self.evaluate(tf.squeeze(h, -1))
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterUnstacking(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithPerDomainPolicyFreqMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterNoisify(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(127)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterGaussianNoisify(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(127)
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
      actual_layer_output = self.evaluate(h)
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testSpectrumAugmenterWithStatelessRandomOps(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      batch_size = 5
      inputs1 = tf.random.uniform(
          shape=[batch_size, 20, 2, 2], minval=0, maxval=1, dtype=tf.float32)
      inputs2 = tf.random.uniform(
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
      actual_layer_output1, actual_layer_output2 = self.evaluate([h1, h2])
      self.assertAllEqual(
          np.shape(actual_layer_output1), np.array([5, 20, 2, 2]))
      self.assertNotAllEqual(actual_layer_output1, actual_layer_output2)

  def testAugmentWeight(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = spectrum_augmenter.SpectrumAugmenter.Params()
      p.name = 'specAug_layers'
      warmup_steps = 5
      p.freq_noise_warmup_steps = warmup_steps
      specaug_layer = p.Instantiate()
      global_step = py_utils.GetGlobalStep()
      update_global_step = global_step
      augment_weights = []
      for _ in range(7):
        with tf.control_dependencies([update_global_step]):
          weight = specaug_layer.augment_weight
          augment_weights += [weight]
        with tf.control_dependencies([weight]):
          update_global_step = tf.assign(global_step, global_step + 1)

      expected_augment_weights = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0])
      actual_augment_weights = self.evaluate(augment_weights)
      print(actual_augment_weights)
      self.assertAllClose(actual_augment_weights, expected_augment_weights)


if __name__ == '__main__':
  tf.test.main()
