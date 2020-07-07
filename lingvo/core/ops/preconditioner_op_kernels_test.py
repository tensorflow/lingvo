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
"""Tests for preconditioner driver."""

import lingvo.compat as tf
from lingvo.core import ops
import numpy as np


class PreconditionerTest(tf.test.TestCase):

  def inverse_pth_root(self, input_t, exponent, epsilon=1e-12):
    input_t_f64 = tf.cast(input_t, tf.float64)
    s, u, v = tf.linalg.svd(
        input_t_f64 +
        tf.eye(tf.shape(input_t_f64)[0], dtype=tf.float64) * epsilon,
        full_matrices=True)
    val = tf.matmul(
        tf.matmul(
            u,
            tf.linalg.tensor_diag(
                tf.pow(tf.maximum(s, epsilon), tf.cast(exponent, tf.float64)))),
        tf.transpose(v))
    return tf.cast(val, tf.float32), tf.reduce_max(tf.abs(u - v))

  def inverse_pth_root_graph(self, epsilon=1e-12):
    graph = tf.Graph()
    with graph.as_default():
      exponent_t = tf.placeholder(dtype=tf.float32, name='exponent', shape=None)
      input_t = tf.placeholder(dtype=tf.float32, name='input', shape=None)
      output, diff = self.inverse_pth_root(input_t, exponent_t, epsilon)
      tf.identity(output, 'output')
      tf.identity(tf.cast(diff, tf.float32), 'diff')
    return graph.as_graph_def().SerializeToString()

  def testPreconditioning(self):
    preconditioner_compute_graphdef = self.inverse_pth_root_graph()
    with tf.Session():
      global_step = tf.train.get_or_create_global_step()
      self.evaluate(tf.global_variables_initializer())
      rand_input_1_t = np.random.rand(4, 4)
      rand_input_2_t = np.random.rand(4, 4)
      exponents = [-0.25, -0.25]
      symmetric_input_1_t = np.dot(rand_input_1_t, rand_input_1_t.transpose())
      symmetric_input_2_t = np.dot(rand_input_2_t, rand_input_2_t.transpose())
      outputs, statuses = ops.get_preconditioners(
          [tf.shape(symmetric_input_1_t),
           tf.shape(symmetric_input_2_t)],
          keys=['a', 'b'],
          preconditioner_compute_graphdef=preconditioner_compute_graphdef)
      self.assertFalse(any(self.evaluate(statuses)))
      preconditioner = ops.compute_preconditioners(
          [symmetric_input_1_t, symmetric_input_2_t],
          exponents,
          tf.cast(global_step, tf.int32),
          keys=['a', 'b'],
          sync=True,
          preconditioner_compute_graphdef=preconditioner_compute_graphdef)
      self.assertAllClose(outputs[0].eval(), np.zeros((4, 4)), atol=1e-4)
      self.assertAllClose(outputs[1].eval(), np.zeros((4, 4)), atol=1e-4)
      preconditioner.run()
      self.assertTrue(any(self.evaluate(statuses)))
      expected_output_1_t = self.inverse_pth_root(symmetric_input_1_t,
                                                  exponents[0])
      expected_output_2_t = self.inverse_pth_root(symmetric_input_2_t,
                                                  exponents[1])
      outputs_np = self.evaluate(outputs)
      self.assertAllClose(
          outputs_np[0], expected_output_1_t[0].eval(), atol=1e-1)
      self.assertAllClose(
          outputs_np[1], expected_output_2_t[0].eval(), atol=1e-1)


if __name__ == '__main__':
  tf.test.main()
