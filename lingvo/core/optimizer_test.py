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
"""Tests for optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from lingvo.core import layers
from lingvo.core import optimizer
from lingvo.core import py_utils


class OptimizerTest(tf.test.TestCase):

  def testAccumulator(self):
    # testAccumulator compares
    #   - explicit averaging of independently computed var_grads1 and
    #     var_grads2,
    #   - Accumulator(SGD) optimizer effectively doing this over 2 steps.
    np.random.seed(12345)
    np_input1 = np.random.normal(0.1, 0.5, [2, 4, 3])
    np.random.seed(12346)
    np_input2 = np.random.normal(0.1, 0.5, [2, 4, 3])

    g1 = tf.Graph()
    with g1.as_default():
      tf.set_random_seed(123456)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.dtype = tf.float64
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.01, 123456)
      params.is_eval = False
      params.batch_norm = False
      proj_layer = layers.ProjectionLayer(params)
      inputs1 = tf.placeholder(shape=[2, 4, 3], dtype=tf.float64)
      in_padding1 = tf.zeros([2, 4, 1], dtype=tf.float64)
      inputs2 = tf.placeholder(shape=[2, 4, 3], dtype=tf.float64)
      in_padding2 = tf.zeros([2, 4, 1], dtype=tf.float64)
      output1 = proj_layer.FPropDefaultTheta(inputs1, in_padding1)
      output2 = proj_layer.FPropDefaultTheta(inputs2, in_padding2)
      loss1 = tf.reduce_sum(output1)
      loss2 = tf.reduce_sum(output2)
      var_grads1 = py_utils.ComputeGradients(loss1, proj_layer.vars)
      var_grads2 = py_utils.ComputeGradients(loss2, proj_layer.vars)
      op = optimizer.SGD.Params().Set(add_summary=False)
      opt = op.cls(op)
      lr = 1e-1
      var_update_op = tf.group(
          opt.Apply(lr, py_utils.ApplyGradMultiplier(var_grads1, 1. / 2.)),
          opt.Apply(lr, py_utils.ApplyGradMultiplier(var_grads2, 1. / 2.)))
      init_op = tf.global_variables_initializer()

    with self.test_session(use_gpu=True, graph=g1) as sess:
      sess.run(init_op)
      print(sess.run(proj_layer.vars))
      sess.run(
          [loss1, loss2, var_grads1, var_grads2, var_update_op],
          feed_dict={
              inputs1: np_input1,
              inputs2: np_input2,
          })
      expected = sess.run(proj_layer.vars.Flatten())
      print(expected)

    g2 = tf.Graph()
    with g2.as_default():
      tf.set_random_seed(123456)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.dtype = tf.float64
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.01, 123456)
      params.is_eval = False
      params.batch_norm = False
      proj_layer = layers.ProjectionLayer(params)
      in_padding1 = tf.zeros([2, 4, 1], dtype=tf.float64)
      inputs1 = tf.placeholder(shape=[2, 4, 3], dtype=tf.float64)
      output1 = proj_layer.FPropDefaultTheta(inputs1, in_padding1)
      loss = tf.reduce_sum(output1)
      var_grads = py_utils.ComputeGradients(loss, proj_layer.vars)
      op = optimizer.Accumulator.Params().Set(
          accum_steps=2,
          dtype=tf.float64,
          optimizer_tpl=optimizer.SGD.Params().Set(add_summary=False))
      opt = op.cls(op)
      lr = 1e-1
      var_update_op = opt.Apply(lr, var_grads)
      init_op = tf.global_variables_initializer()
      increment_global_step_op = tf.assign_add(py_utils.GetOrCreateGlobalStep(),
                                               1)

    with self.test_session(use_gpu=True, graph=g2) as sess:
      sess.run(init_op)
      print(sess.run(proj_layer.vars))
      sess.run(
          [loss, var_grads, var_update_op], feed_dict={
              inputs1: np_input1,
          })
      sess.run(increment_global_step_op)
      sess.run(
          [loss, var_grads, var_update_op], feed_dict={
              inputs1: np_input2,
          })
      actual = sess.run(proj_layer.vars.Flatten())
      print(actual)

    self.assertAllClose(actual, expected)


if __name__ == '__main__':
  tf.test.main()
