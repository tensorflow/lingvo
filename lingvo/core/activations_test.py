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
"""Tests for layers."""

import lingvo.compat as tf
from lingvo.core import activations
from lingvo.core import test_utils
import numpy as np


class ActivationsTest(test_utils.TestCase):

  def testGeluActivation(self):
    with self.session(use_gpu=True):
      inputs = tf.constant(
          np.linspace(-10.0, 10.0, num=21, dtype='float32'), dtype=tf.float32)
      grads_gelu = tf.gradients(tf.nn.gelu(inputs), inputs)
      grads_relu = tf.gradients(tf.nn.relu(inputs), inputs)

      self.assertEqual(0.0,
                       tf.nn.gelu(tf.constant(-10.0, dtype='float32')).eval())
      self.assertEqual(0.0,
                       tf.nn.gelu(tf.constant(0.0, dtype='float32')).eval())
      self.assertEqual(10.0,
                       tf.nn.gelu(tf.constant(10.0, dtype='float32')).eval())
      actual_grads_gelu = grads_gelu[0].eval()
      actual_grads_relu = grads_relu[0].eval()

      self.assertAllClose(actual_grads_gelu[-5:], actual_grads_relu[-5:])
      self.assertAllClose(actual_grads_gelu[:5], actual_grads_relu[:5])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_grads_gelu = [
          -7.69459925e-22, -9.25176121e-18, -4.04182472e-14, -6.39430453e-11,
          -3.64552299e-08, -7.13557529e-06, -5.03641320e-04, -1.19456425e-02,
          -8.52318183e-02, -8.33154917e-02,  5.00000000e-01,  1.08331549e+00,
          1.08523178e+00,  1.01194561e+00,  1.00050366e+00,  1.00000715e+00,
          1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
          1.00000000e+00]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_grads_gelu, actual_grads_gelu)

  def testSquaredReluActivation(self):
    with self.session(use_gpu=True):
      inputs = tf.constant(
          np.linspace(-5.0, 5.0, num=11, dtype='float32'), dtype=tf.float32)
      act_fn = activations.GetFn('SQUARED_RELU')

      for inp, out in [(-10.0, 0.0), (0.0, 0.0), (2.0, 4.0)]:
        self.assertEqual(out, act_fn(tf.constant(inp, dtype='float32')).eval())
      grads_squared_relu = tf.gradients(act_fn(inputs), inputs)
      grads_squared_relu = grads_squared_relu[0].eval()

      self.assertAllClose([0., 0., 0., 0., 0., 0., 2., 4., 6., 8., 10.],
                          grads_squared_relu)


if __name__ == '__main__':
  tf.test.main()
