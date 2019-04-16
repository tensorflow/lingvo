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
"""Tests for lingvo recurrent on gpu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from lingvo.core import py_utils
from lingvo.core import recurrent
from lingvo.core import test_utils


class RecurrentGpuTest(test_utils.TestCase):

  def testRecurrent(self):
    """Run on GPU locally with --test_output=streamed --config=cuda."""

    def Sum(theta, state, inputs):
      next_state = py_utils.NestedMap()
      v = tf.reduce_sum(tf.one_hot(inputs.one_hot, depth=2) * theta.x, axis=0)
      next_state.sum = state.sum + v
      return next_state, py_utils.NestedMap()

    with self.session(use_gpu=True) as sess:

      theta = py_utils.NestedMap()
      theta.x = tf.constant([-1.0, 2.0])
      state = py_utils.NestedMap()
      state.sum = tf.constant(0.0)
      inputs = py_utils.NestedMap()
      inputs.one_hot = tf.constant([0, 1, 1], dtype=tf.int32)

      # sum = -1 + 2 + 2
      ret = recurrent.Recurrent(theta, state, inputs, Sum)

      acc, state = sess.run(ret)
      self.assertAllClose(acc.sum, [-1., 1., 3.])
      self.assertAllClose(state.sum, 3.)

      y = ret[1].sum
      dx, d_inputs = tf.gradients(ys=[y], xs=[theta.x, inputs.one_hot])
      tf.logging.info('d(inputs) = %s', d_inputs)
      dx_val = sess.run(dx)
      self.assertAllClose(dx_val, [1, 2])


if __name__ == '__main__':
  tf.test.main()
