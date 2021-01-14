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

import lingvo.compat as tf
from lingvo.core import spectrum_augmenter
from lingvo.core import spectrum_augmenter_on_device
from lingvo.core import test_utils
import numpy as np


class SpectrumAugmenterTest(test_utils.TestCase):

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
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 0
        p.time_mask_max_frames = 5
        p.time_mask_count = 2
        p.time_mask_max_ratio = 1.0
        p.random_seed = 23456
        specaug_layer = p.Instantiate()

        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

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
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 0
        p.time_mask_max_ratio = 0.4
        p.time_mask_count = 1
        p.use_dynamic_time_mask_max_frames = True
        p.random_seed = 12345
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

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
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 0
        p.time_mask_max_frames = 5
        p.time_mask_count = 10
        p.time_masks_per_frame = 0.2
        p.random_seed = 67890
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

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
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 0
        p.time_mask_max_frames = 5
        p.time_mask_count = 10
        p.time_masks_per_frame = 0.2
        p.time_mask_max_ratio = 0.4
        p.use_dynamic_time_mask_max_frames = True
        p.random_seed = 67890
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

  def testSpectrumAugmenterWithFrequencyMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
      inputs = tf.ones([3, 5, 10, 1], dtype=tf.float32)
      paddings = tf.zeros([3, 5])
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 6
        p.freq_mask_count = 2
        p.time_mask_max_frames = 0
        p.random_seed = 34567
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

  def testSpectrumAugmenterWarpMatrixConstructor(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      inputs = tf.broadcast_to(tf.cast(tf.range(10), dtype=tf.float32), (4, 10))
      origin = tf.cast([2, 4, 4, 5], dtype=tf.float32)
      destination = tf.cast([3, 2, 6, 8], dtype=tf.float32)
      choose_range = tf.cast([4, 8, 8, 10], dtype=tf.float32)
      outputs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        specaug_layer = p.Instantiate()
        warp_matrix = specaug_layer._ConstructWarpMatrix(
            batch_size=4,
            matrix_size=10,
            origin=origin,
            destination=destination,
            choose_range=choose_range,
            dtype=tf.float32)
        output = tf.einsum('bij,bj->bi', warp_matrix, inputs)
        outputs.append(output)
      layer_output, layer_output_on_device = self.evaluate(outputs)
      self.assertAllClose(layer_output, layer_output_on_device)

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
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 0
        p.time_mask_max_frames = 0
        p.time_warp_max_frames = 8
        p.time_warp_max_ratio = 1.0
        p.time_warp_bound = 'static'
        p.random_seed = 34567
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

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
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 0
        p.time_mask_max_frames = 0
        p.time_warp_max_ratio = 0.5
        p.time_warp_bound = 'dynamic'
        p.random_seed = 34567
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

  def testSpectrumAugmenterUnstacking(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
      inputs = tf.ones([3, 5, 10, 1], dtype=tf.float32)
      paddings = tf.zeros([3, 5])
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.unstack = True
        p.stack_height = 2
        p.freq_mask_max_bins = 5
        p.time_mask_max_frames = 8
        p.random_seed = 12345
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

  def testSpectrumAugmenterWithPerDomainPolicyFreqMask(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(1234)
      inputs = tf.ones([6, 5, 4, 2], dtype=tf.float32)
      input_domain_ids = tf.constant(
          [[1] * 5, [2] * 5, [0] * 5, [2] * 5, [0] * 5, [1] * 5],
          dtype=tf.float32)
      paddings = tf.zeros([3, 5])
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.domain_ids = [0, 1, 2]
        p.freq_mask_max_bins = [0, 3, 8]
        p.time_mask_max_frames = 0
        p.random_seed = 1234
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(
            inputs, paddings, domain_ids=input_domain_ids)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

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
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 0
        p.time_mask_max_ratio = 0.4
        p.time_mask_count = 1
        p.use_dynamic_time_mask_max_frames = True
        p.use_noise = True
        p.gaussian_noise = False
        p.random_seed = 12345
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

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
      hs = []
      for p in [
          spectrum_augmenter.SpectrumAugmenter.Params(),
          spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      ]:
        p.name = 'specAug_layers'
        p.freq_mask_max_bins = 0
        p.time_mask_max_ratio = 0.4
        p.time_mask_count = 1
        p.use_dynamic_time_mask_max_frames = True
        p.use_noise = True
        p.gaussian_noise = True
        p.random_seed = 12345
        specaug_layer = p.Instantiate()
        h, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
        hs.append(h)
      layer_output, layer_output_on_device = self.evaluate(hs)
      self.assertAllClose(layer_output, layer_output_on_device)

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

      p = spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
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

  def testGraphContainsOnDeviceOps(self):
    """Checks that einsum and stateful random ops are not used on-device."""
    model_graph = tf.Graph()
    with model_graph.as_default():
      batch_size = 5
      inputs = tf.random.stateless_uniform(
          shape=[batch_size, 20, 2, 2],
          minval=0,
          maxval=1,
          seed=tf.constant([123, 123]),
          dtype=tf.float32)
      paddings = []
      for i in range(batch_size):
        paddings.append(
            tf.concat([tf.zeros([1, i + 12]),
                       tf.ones([1, 8 - i])], axis=1))
      paddings = tf.concat(paddings, axis=0)
      p = spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      p.name = 'specAug_layers'
      p.freq_mask_count = 1
      p.freq_mask_max_bins = 1
      p.time_mask_max_frames = 5
      p.time_mask_count = 2
      p.use_noise = True
      p.gaussian_noise = True
      p.time_mask_max_ratio = 1.0
      p.use_input_dependent_random_seed = True
      specaug_layer = p.Instantiate()
      _, _ = specaug_layer.FPropDefaultTheta(inputs, paddings)
    # A list of ops that are not compatible with on-device training.
    unsupported_on_device_nodes = [
        'RandomUniform', 'RandomStandardNormal', 'Einsum'
    ]
    for node in model_graph.as_graph_def().node:
      self.assertNotIn(node.op, unsupported_on_device_nodes)

  def testEinsumReplacementBBmBm(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      a = tf.random.uniform(shape=[20], minval=0, maxval=1, dtype=tf.float32)
      b = tf.random.uniform(
          shape=[20, 10], minval=0, maxval=1, dtype=tf.float32)
      einsum = tf.einsum('b,bm->bm', a, b)
      p = spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      p.name = 'specAug_layers'
      specaug_layer = p.Instantiate()
      replacement = specaug_layer.EinsumBBmBm(a, b)
      einsum, replacement = self.evaluate([einsum, replacement])
      self.assertAllClose(einsum, replacement)

  def testEinsumReplacementBxycByBxyc(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      a = tf.random.uniform(
          shape=[20, 5, 7, 4], minval=0, maxval=1, dtype=tf.float32)
      b = tf.random.uniform(shape=[20, 7], minval=0, maxval=1, dtype=tf.float32)
      einsum = tf.einsum('bxyc,by->bxyc', a, b)
      p = spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      p.name = 'specAug_layers'
      specaug_layer = p.Instantiate()
      replacement = specaug_layer.EinsumBxycByBxyc(a, b)
      einsum, replacement = self.evaluate([einsum, replacement])
      self.assertAllClose(einsum, replacement)

  def testEinsumReplacementBxycBxBxyc(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      a = tf.random.uniform(
          shape=[20, 5, 7, 4], minval=0, maxval=1, dtype=tf.float32)
      b = tf.random.uniform(shape=[20, 5], minval=0, maxval=1, dtype=tf.float32)
      einsum = tf.einsum('bxyc,bx->bxyc', a, b)
      p = spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      p.name = 'specAug_layers'
      specaug_layer = p.Instantiate()
      replacement = specaug_layer.EinsumBxycBxBxyc(a, b)
      einsum, replacement = self.evaluate([einsum, replacement])
      self.assertAllClose(einsum, replacement)

  def testEinsumReplacementBxyBxBxy(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      a = tf.random.uniform(
          shape=[20, 7, 4], minval=0, maxval=1, dtype=tf.float32)
      b = tf.random.uniform(shape=[20, 7], minval=0, maxval=1, dtype=tf.float32)
      einsum = tf.einsum('bxy,bx->bxy', a, b)
      p = spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      p.name = 'specAug_layers'
      specaug_layer = p.Instantiate()
      replacement = specaug_layer.EinsumBxyBxBxy(a, b)
      einsum, replacement = self.evaluate([einsum, replacement])
      self.assertAllClose(einsum, replacement)

  def testEinsumReplacementBxycBzxBzyc(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      a = tf.random.uniform(
          shape=[20, 7, 4, 3], minval=0, maxval=1, dtype=tf.float32)
      b = tf.random.uniform(
          shape=[20, 5, 7], minval=0, maxval=1, dtype=tf.float32)
      einsum = tf.einsum('bxyc,bzx->bzyc', a, b)
      p = spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      p.name = 'specAug_layers'
      specaug_layer = p.Instantiate()
      replacement = specaug_layer.EinsumBxycBzxBzyc(a, b)
      einsum, replacement = self.evaluate([einsum, replacement])
      self.assertAllClose(einsum, replacement)

  def testEinsumReplacementBxycBzyBxzc(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      a = tf.random.uniform(
          shape=[20, 7, 4, 3], minval=0, maxval=1, dtype=tf.float32)
      b = tf.random.uniform(
          shape=[20, 5, 4], minval=0, maxval=1, dtype=tf.float32)
      einsum = tf.einsum('bxyc,bzy->bxzc', a, b)
      p = spectrum_augmenter_on_device.SpectrumAugmenterOnDevice.Params()
      p.name = 'specAug_layers'
      specaug_layer = p.Instantiate()
      replacement = specaug_layer.EinsumBxycBzyBxzc(a, b)
      einsum, replacement = self.evaluate([einsum, replacement])
      self.assertAllClose(einsum, replacement)


if __name__ == '__main__':
  tf.test.main()
