# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Insertion Framework tests."""

import lingvo.compat as tf
from lingvo.core import insertion
from lingvo.core import test_utils
import numpy as np


class SequenceTest(test_utils.TestCase):

  def testSequenceTrimLastToken(self):
    x = np.asarray([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                   np.int32)
    x_paddings = np.asarray(
        [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]], np.float32)

    with self.session():
      x_trimmed, x_trimmed_paddings = insertion.SequenceTrimLastToken(
          tf.convert_to_tensor(x), tf.convert_to_tensor(x_paddings))

      x_trimmed, x_trimmed_paddings = self.evaluate(
          [x_trimmed, x_trimmed_paddings])

      # `x_trimmed_gold` is the same as `x` w/ last token removed.
      # `x_trimmed_paddings_gold` is the corresponding paddings.
      x_trimmed_gold = np.asarray(
          [[1, 2, 3, 0], [1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], np.int32)
      x_trimmed_paddings_gold = np.asarray(
          [[0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], np.float32)

      self.assertAllEqual(x_trimmed, x_trimmed_gold)
      self.assertAllEqual(x_trimmed_paddings, x_trimmed_paddings_gold)

  def testSequenceAppendToken(self):
    x = np.asarray([[1, 2, 3, 0], [1, 2, 3, 4], [0, 0, 0, 0], [1, 0, 0, 0]],
                   np.int32)
    x_paddings = np.asarray(
        [[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]], np.float32)

    with self.session():
      x_appended, x_appended_paddings = insertion.SequenceAppendToken(
          tf.convert_to_tensor(x), tf.convert_to_tensor(x_paddings), 10)

      x_appended, x_appended_paddings = self.evaluate([
          tf.convert_to_tensor(x_appended),
          tf.convert_to_tensor(x_appended_paddings)
      ])

      # `x_appended_gold` is the same as `x` w/ token `10` appended.
      # `x_appended_paddings_gold` is the corresponding paddings.
      x_appended_gold = np.asarray(
          [[1, 2, 3, 10], [1, 2, 3, 10], [10, 0, 0, 0], [1, 10, 0, 0]],
          np.int32)
      x_appended_paddings_gold = np.asarray(
          [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1]], np.float32)

      self.assertAllEqual(x_appended, x_appended_gold)
      self.assertAllEqual(x_appended_paddings, x_appended_paddings_gold)

  def testSequenceAppendTokenExtend(self):
    x = np.asarray([[1, 2, 3, 0], [1, 2, 3, 4], [0, 0, 0, 0], [1, 0, 0, 0]],
                   np.int32)
    x_paddings = np.asarray(
        [[0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1]], np.int32)

    with self.session():
      x_appended, x_appended_paddings = insertion.SequenceAppendToken(
          tf.convert_to_tensor(x), tf.convert_to_tensor(x_paddings), 10, True)

      x_appended, x_appended_paddings = self.evaluate(
          [x_appended, x_appended_paddings])

      # `x_appended_gold` is the same as `x` w/ token `10` appended, we also
      # test for the condition of extend=True which requires +1 dim in the
      # time dimension.
      # `x_appended_paddings_gold` is the corresponding paddings.
      x_appended_gold = np.asarray([[1, 2, 3, 10, 0], [1, 2, 3, 4, 10],
                                    [10, 0, 0, 0, 0], [1, 10, 0, 0, 0]],
                                   np.int32)
      x_appended_paddings_gold = np.asarray(
          [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1]],
          np.int32)

      self.assertAllEqual(x_appended, x_appended_gold)
      self.assertAllEqual(x_appended_paddings, x_appended_paddings_gold)

  def testSequenceConcat(self):
    x = np.asarray([[1, 2, 3, 0], [1, 2, 3, 4], [0, 0, 0, 0], [1, 0, 0, 0]],
                   np.int32)
    x_paddings = np.asarray(
        [[0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1]], np.float32)

    y = np.asarray(
        [[10, 20, 30, 0], [10, 20, 30, 40], [0, 0, 0, 0], [10, 0, 0, 0]],
        np.int32)
    y_paddings = np.asarray(
        [[0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1]], np.float32)

    with self.session():
      xy, xy_paddings = insertion.SequenceConcat(
          tf.convert_to_tensor(x), tf.convert_to_tensor(x_paddings),
          tf.convert_to_tensor(y), tf.convert_to_tensor(y_paddings), 999)

      xy, xy_paddings = self.evaluate(
          [tf.convert_to_tensor(xy),
           tf.convert_to_tensor(xy_paddings)])

      # `xy_gold` is `x` and `y` concatenated.
      # `xy_paddings_gold` is the corresponding paddings.
      xy_gold = np.asarray(
          [[1, 2, 3, 10, 20, 30, 999, 999], [1, 2, 3, 4, 10, 20, 30, 40],
           [999, 999, 999, 999, 999, 999, 999, 999],
           [1, 10, 999, 999, 999, 999, 999, 999]], np.int32)
      xy_paddings_gold = np.asarray(
          [[0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]], np.float32)

      self.assertAllEqual(xy, xy_gold)
      self.assertAllEqual(xy_paddings, xy_paddings_gold)


class SymbolInsertionLayerTest(test_utils.TestCase):

  def testGetValidCanvasUnderUniformRollinPolicy(self):
    with self.session(use_gpu=True):
      params = insertion.SymbolInsertionLayer.Params()
      params.name = 'insertion'
      params.rollin_policy = 'oracle'
      params.oracle_policy = 'uniform'

      insertion_layer = insertion.SymbolInsertionLayer(params)

      batch_size = 4
      time_dim = 10

      inputs = tf.tile(tf.expand_dims(tf.range(time_dim), 0), [batch_size, 1])
      spec = insertion_layer.FProp(None, inputs, force_sample_last_token=False)

      canvas, canvas_indices, canvas_paddings = self.evaluate(
          [spec.canvas, spec.canvas_indices, spec.canvas_paddings])

      for b in range(batch_size):
        length = np.sum(1 - canvas_paddings[b, :]).astype(np.int32)
        self.assertAllEqual(canvas[b, :length], canvas_indices[b, :length])
        # Test the invalid slots.
        self.assertAllEqual(canvas[b, length:],
                            [0] * (canvas.shape[1] - length))
        self.assertAllEqual(canvas_indices[b, length:],
                            [time_dim - 1] * (canvas.shape[1] - length))

  def testMaxCanvasSizeUnderUniformRollinPolicy(self):
    """Tests for valid canvas size."""
    with self.session(use_gpu=True):
      params = insertion.SymbolInsertionLayer.Params()
      params.name = 'insertion'
      params.rollin_policy = 'oracle'
      params.oracle_policy = 'uniform'

      insertion_layer = insertion.SymbolInsertionLayer(params)

      batch_size = 4
      time_dim = 10

      inputs = tf.tile(tf.expand_dims(tf.range(time_dim), 0), [batch_size, 1])
      inputs_len = tf.random.uniform([batch_size], 0, time_dim, tf.int32)
      paddings = 1 - tf.sequence_mask(inputs_len, time_dim, tf.int32)
      spec = insertion_layer.FProp(
          None, inputs, paddings, force_sample_last_token=False)

      canvas_with_max_length = False
      for _ in range(1000):
        canvas_max_len, canvas, canvas_paddings = self.evaluate(
            [inputs_len, spec.canvas, spec.canvas_paddings])

        for b in range(batch_size):
          max_len = canvas_max_len[b]
          length = np.sum(1 - canvas_paddings[b, :]).astype(np.int32)
          canvas_with_max_length |= length == max_len
          self.assertLessEqual(length, max_len)
          # Invalid entries of canvas should be 0.
          self.assertAllEqual(canvas[b, length:],
                              [0] * (canvas.shape[1] - length))

      # With high probability, there should be at least one canvas that is
      # of the same size as the maximum canvas size.
      self.assertEqual(canvas_with_max_length, True)

  def testContiguousCanvasUnderUniformRollinPolicy(self):
    """Tests for valid canvas size."""
    with self.session(use_gpu=True):
      params = insertion.SymbolInsertionLayer.Params()
      params.name = 'insertion'
      params.rollin_policy = 'oracle'
      params.oracle_policy = 'uniform'

      insertion_layer = insertion.SymbolInsertionLayer(params)

      batch_size = 4
      time_dim = 10

      inputs = tf.tile(
          tf.expand_dims(tf.range(time_dim), 0) + 100, [batch_size, 1])
      inputs_len = tf.random.uniform([batch_size], 0, time_dim, tf.int32)
      paddings = 1 - tf.sequence_mask(inputs_len, time_dim, tf.int32)
      spec = insertion_layer.FProp(
          None, inputs, paddings, force_sample_last_token=False)

      for _ in range(1000):
        canvas, canvas_paddings = self.evaluate(
            [spec.canvas, spec.canvas_paddings])

        for b in range(batch_size):
          length = np.sum(1 - canvas_paddings[b, :]).astype(np.int32)
          # Check for valid part of the canvas and padding.
          for l in range(length):
            self.assertEqual(canvas_paddings[b, l], 0)
            self.assertNotEqual(canvas[b, l], 0)
          # Check for invalid part of the canvas and padding.
          for l in range(length, canvas.shape[1]):
            self.assertEqual(canvas_paddings[b, l], 1)
            self.assertEqual(canvas[b, l], 0)

  def testGetValidCanvasAndTargetsUnderUniformOraclePolicyWithoutForcedSample(
      self):
    """Tests for canvas+targets under uniform (rollin+oracle) policy."""
    with self.session(use_gpu=True):
      params = insertion.SymbolInsertionLayer.Params()
      params.name = 'insertion'
      params.rollin_policy = 'oracle'
      params.oracle_policy = 'uniform'
      params.random_seed = 12345

      insertion_layer = params.Instantiate()

      x = np.asarray(
          [[10, 11, 12, 13, 14, 15, 16], [10, 11, 12, 13, 14, 15, 16],
           [10, 0, 0, 0, 0, 0, 0], [10, 11, 12, 13, 14, 15, 0]], np.int32)
      x_paddings = np.asarray([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1]],
                              np.float32)

      spec = insertion_layer.FProp(
          None,
          tf.convert_to_tensor(x),
          tf.convert_to_tensor(x_paddings),
          force_sample_last_token=False)

      (canvas, canvas_paddings, target_indices,
       target_weights) = self.evaluate([
           spec.canvas, spec.canvas_paddings, spec.target_indices,
           spec.target_weights
       ])

      canvas_gold = np.asarray([[10, 12, 13, 15, 16], [14, 0, 0, 0, 0],
                                [10, 0, 0, 0, 0], [11, 12, 13, 0, 0]], np.int32)
      canvas_paddings_gold = np.asarray(
          [[0., 0., 0., 0., 0.], [0., 1., 1., 1., 1.], [0., 1., 1., 1., 1.],
           [0., 0., 0., 1., 1.]], np.float32)
      target_indices_gold = np.asarray(
          [[0, 0, 1], [0, 1, 11], [0, 1, 1], [0, 2, 1], [0, 3, 14], [0, 3, 1],
           [0, 4, 1], [1, 0, 10], [1, 0, 11], [1, 0, 12], [1, 0, 13], [1, 0, 1],
           [1, 1, 15], [1, 1, 1], [2, 0, 1], [3, 0, 10], [3, 0, 1], [3, 1, 1],
           [3, 2, 1], [3, 3, 14], [3, 3, 15]], np.int32)
      target_weights_gold = np.asarray([1, 1, 0, 1, 1, 0, 1] +
                                       [1, 1, 1, 1, 0, 1, 0] + [1] +
                                       [1, 0, 1, 1, 1, 1], np.float32)
      target_weights_gold = np.reshape(target_weights_gold,
                                       [target_weights_gold.shape[0], 1])

      self.assertAllEqual(canvas, canvas_gold)
      self.assertAllEqual(canvas_paddings, canvas_paddings_gold)
      self.assertAllEqual(target_indices, target_indices_gold)
      self.assertAllEqual(target_weights, target_weights_gold)

  def testGetValidCanvasAndTargetsUnderUniformOraclePolicyForcedSample(self):
    """Tests for canvas+targets under uniform (rollin+oracle) policy."""
    with self.session(use_gpu=True):
      params = insertion.SymbolInsertionLayer.Params()
      params.name = 'insertion'
      params.rollin_policy = 'oracle'
      params.oracle_policy = 'uniform'
      params.random_seed = 12345

      insertion_layer = insertion.SymbolInsertionLayer(params)

      x = np.asarray(
          [[10, 11, 12, 13, 14, 15, 16, 1], [10, 11, 12, 13, 14, 15, 16, 1],
           [10, 1, 0, 0, 0, 0, 0, 0], [10, 11, 12, 13, 14, 15, 1, 0]], np.int32)
      x_paddings = np.asarray(
          [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)

      spec = insertion_layer.FProp(None, tf.convert_to_tensor(x),
                                   tf.convert_to_tensor(x_paddings))

      (canvas, canvas_indices, canvas_paddings, target_indices,
       target_weights) = self.evaluate([
           spec.canvas, spec.canvas_indices, spec.canvas_paddings,
           spec.target_indices, spec.target_weights
       ])

      canvas_gold = np.asarray([[10, 12, 13, 15, 16, 1], [13, 1, 0, 0, 0, 0],
                                [10, 1, 0, 0, 0, 0], [10, 12, 14, 1, 0, 0]],
                               np.int32)
      canvas_indices_gold = np.asarray([[0, 2, 3, 5, 6, 7], [3, 7, 7, 7, 7, 7],
                                        [0, 1, 7, 7, 7, 7], [0, 2, 4, 6, 7, 7]],
                                       np.int32)
      canvas_paddings_gold = np.asarray(
          [[0., 0., 0., 0., 0., 0.], [0., 0., 1., 1., 1., 1.],
           [0., 0., 1., 1., 1., 1.], [0., 0., 0., 0., 1., 1.]], np.float32)
      target_indices_gold = np.asarray(
          [[0, 0, 1], [0, 1, 11], [0, 1, 1], [0, 2, 1], [0, 3, 14], [0, 3, 1],
           [0, 4, 1], [0, 5, 1], [1, 0, 10], [1, 0, 11], [1, 0, 12], [1, 0, 1],
           [1, 1, 14], [1, 1, 15], [1, 1, 16], [1, 1, 1], [2, 0, 1], [2, 1, 1],
           [3, 0, 1], [3, 1, 11], [3, 1, 1], [3, 2, 13], [3, 2, 1], [3, 3, 15],
           [3, 3, 1]], np.int32)
      target_weights_gold = np.asarray([1, 1, 0, 1, 1, 0, 1, 1] +
                                       [1, 1, 1, 0, 1, 1, 1, 0] + [1, 1] +
                                       [1, 1, 0, 1, 0, 1, 0], np.float32)
      target_weights_gold = np.reshape(target_weights_gold,
                                       [target_weights_gold.shape[0], 1])

      self.assertAllEqual(canvas, canvas_gold)
      self.assertAllEqual(canvas_indices, canvas_indices_gold)
      self.assertAllEqual(canvas_paddings, canvas_paddings_gold)
      self.assertAllEqual(target_indices, target_indices_gold)
      self.assertAllEqual(target_weights, target_weights_gold)


if __name__ == '__main__':
  tf.test.main()
