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
from lingvo.core import test_utils


class OptimizerTest(test_utils.TestCase):

  def testAccumulator(self):
    # testAccumulator compares
    #   - explicit averaging of independently computed var_grads1 and
    #     var_grads2,
    #   - Accumulator(SGD) optimizer effectively doing this over 2 steps.
    np.random.seed(12345)
    np_input1 = np.random.normal(0.1, 0.5, [2, 4, 3])
    np.random.seed(12346)
    np_input2 = np.random.normal(0.1, 0.5, [2, 4, 3])

    with self.session(use_gpu=True, graph=tf.Graph()) as sess:
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
      op = optimizer.SGD.Params()
      opt = op.cls(op)
      lr = 1e-1
      with tf.control_dependencies([loss1, loss2]):
        var_update_op1 = opt.Apply(
            lr, py_utils.ApplyGradMultiplier(var_grads1, 1. / 2.))
        with tf.control_dependencies([var_update_op1]):
          var_update_op2 = opt.Apply(
              lr, py_utils.ApplyGradMultiplier(var_grads2, 1. / 2.))

      sess.run(tf.global_variables_initializer())
      vars1 = sess.run(proj_layer.vars.Flatten())
      loss1_1, grads1_1, loss1_2, grads1_2 = sess.run(
          [loss1, var_grads1, loss2, var_grads2],
          feed_dict={
              inputs1: np_input1,
              inputs2: np_input2,
          })
      sess.run(
          [var_update_op2], feed_dict={
              inputs1: np_input1,
              inputs2: np_input2,
          })
      vars1_1 = sess.run(proj_layer.vars.Flatten())

    with self.session(use_gpu=True, graph=tf.Graph()) as sess:
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
          accum_steps=2, dtype=tf.float64, optimizer_tpl=optimizer.SGD.Params())
      opt = op.cls(op)
      lr = 1e-1
      var_update_op = opt.Apply(lr, var_grads)
      global_step = py_utils.GetGlobalStep()
      increment_global_step_op = tf.assign_add(global_step, 1)

      sess.run(tf.global_variables_initializer())
      vars2, global_step = sess.run([proj_layer.vars.Flatten(), global_step])
      loss2_1, grads2_1 = sess.run(
          [loss, var_grads], feed_dict={
              inputs1: np_input1,
          })
      loss2_2, grads2_2 = sess.run(
          [loss, var_grads], feed_dict={
              inputs1: np_input2,
          })
      acc_0 = sess.run(
          [v for v in tf.global_variables() if 'grad_accumulator' in v.name])[0]
      sess.run(
          [var_update_op], feed_dict={
              inputs1: np_input1,
          })
      acc_1 = sess.run(
          [v for v in tf.global_variables() if 'grad_accumulator' in v.name])[0]
      vars2_intermediate = sess.run(proj_layer.vars.Flatten())
      sess.run(increment_global_step_op)
      sess.run(
          [var_update_op], feed_dict={
              inputs1: np_input2,
          })
      acc_2 = sess.run(
          [v for v in tf.global_variables() if 'grad_accumulator' in v.name])[0]
      vars2_1 = sess.run(proj_layer.vars.Flatten())

    self.assertAllClose(vars1, vars2)

    self.assertAllClose(acc_0, np.zeros_like(acc_0))
    self.assertAllClose(acc_1, grads2_1['w'][1])
    self.assertAllClose(acc_2, np.zeros_like(acc_0))

    self.assertAllClose(loss1_1, loss2_1)
    self.assertAllClose(loss1_2, loss2_2)
    self.assertAllClose(grads1_1, grads2_1)
    self.assertAllClose(grads1_2, grads2_2)

    self.assertAllClose(vars1, vars2_intermediate)

    self.assertAllClose(vars2[0], grads2_1['w'][0])
    self.assertAllClose(vars2[0], grads2_2['w'][0])

    self.assertAllClose(
        vars1[0] - 0.5 * lr * (grads1_1['w'][1] + grads1_2['w'][1]), vars1_1[0])

    self.assertAllClose(
        vars2[0] - 0.5 * lr * (grads2_1['w'][1] + grads2_2['w'][1]), vars2_1[0])

    self.assertAllClose(vars2, vars2_intermediate)
    self.assertAllClose(vars1_1, vars2_1)


if __name__ == '__main__':
  tf.test.main()
