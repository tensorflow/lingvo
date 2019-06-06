# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow as tf

from lingvo.core import insertion
from lingvo.core import test_utils


class SymbolInsertionLayerTest(test_utils.TestCase):

  def testGetValidCanvasUnderUniformRollinPolicy(self):
    with self.session(use_gpu=True) as sess:
      params = insertion.SymbolInsertionLayer.Params()
      params.name = 'insertion'
      params.rollin_policy = 'oracle'
      params.oracle_policy = 'uniform'

      insertion_layer = insertion.SymbolInsertionLayer(params)

      batch_size = 4
      time_dim = 10

      inputs = tf.tile(tf.expand_dims(tf.range(time_dim), 0), [batch_size, 1])
      spec = insertion_layer.FProp(None, inputs)

      canvas, canvas_indices, canvas_paddings = sess.run(
          [spec.canvas, spec.canvas_indices, spec.canvas_paddings])

      for b in range(batch_size):
        length = np.sum(1 - canvas_paddings[b, :])
        self.assertAllEqual(canvas[b, :length], canvas_indices[b, :length])
        self.assertAllEqual(canvas[b, length:],
                            [time_dim - 1] * (canvas.shape[1] - length))

  def testGetValidCanvasAndTargetsUnderUniformOraclePolicy(self):
    """Tests for valid canvas+targets under uniform (rollin+oracle) policy."""
    with self.session(use_gpu=True) as sess:
      params = insertion.SymbolInsertionLayer.Params()
      params.name = 'insertion'
      params.rollin_policy = 'oracle'
      params.oracle_policy = 'uniform'

      insertion_layer = insertion.SymbolInsertionLayer(params)

      batch_size = 8
      time_dim = 10

      inputs = np.tile(np.arange(time_dim, dtype=np.int32), [batch_size, 1])
      inputs_len = np.random.randint(0, time_dim, [batch_size], np.int32)
      inputs_len[0] = 0
      inputs_len[1] = 1
      inputs_len[2] = time_dim - 1
      inputs_len[3] = time_dim - 2
      inputs_paddings = 1 - tf.sequence_mask(inputs_len, time_dim, tf.int32)
      spec = insertion_layer.FProp(None, inputs, inputs_paddings)

      (canvas_indices, canvas_paddings, target_indices,
       target_weights) = sess.run([
           spec.canvas_indices, spec.canvas_paddings, spec.target_indices,
           spec.target_weights
       ])

      target_index = 0
      for b in range(batch_size):
        canvas_length = np.sum(1 - canvas_paddings[b, :])

        canvas_index = 0

        # Loop through the original `inputs` length.
        for l in range(inputs_len[b]):
          if (canvas_index < canvas_length and
              canvas_indices[b, canvas_index] == l):
            canvas_index += 1
          elif target_indices[target_index, 2] == l:
            self.assertEqual(target_indices[target_index, 0], b,
                             'Mismatch in batch index.')
            self.assertEqual(target_indices[target_index, 1], canvas_index,
                             'Mismatch in slot index.')
            self.assertEqual(target_indices[target_index, 2], inputs[b, l],
                             'Mismatch in content index.')
            target_index += 1
          else:
            raise ValueError(
                'Mismatch between canvas_indices and target_indices!\n%s\n%s' %
                (str(canvas_indices), str(target_indices)))

      # Make sure we consume all the targets.
      self.assertEqual(target_index, target_indices.shape[0])
      self.assertAllEqual(target_weights,
                          np.ones([target_indices.shape[0]], np.int32))


if __name__ == '__main__':
  tf.test.main()
