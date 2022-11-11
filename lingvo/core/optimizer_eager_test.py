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

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import layers
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np


class OptimizerTest(test_utils.TestCase):

  @tf.function
  def _FpropBprop(self, fc_layer, opt):
    inputs = tf.zeros(shape=[2, 4, 3], dtype=tf.float64)
    output = fc_layer.FPropDefaultTheta(inputs)
    loss = tf.reduce_sum(output)
    var_grads = py_utils.ComputeGradients(loss, fc_layer.vars)
    # Name becomes meaningless in Eager mode. Here we just check whether
    # errors get raised.
    update_op = opt.Apply(1e-1, var_grads)
    self.assertIn('composite_optimizer_train_op', update_op.name)

  def testCompositeOptimizerName(self):
    adam_op = optimizer.Adam.Params()
    rmsprop_op = optimizer.RMSProp.Params()
    adam_rmsprop_opt = optimizer.CompositeOptimizer.Params().Set(
        optimizer_map={
            'fc/w': (adam_op, 1.),
            'fc/b': (rmsprop_op, 1.),
            'default_optimizer': (adam_op, 1.)
        }).Instantiate()

    params = layers.FCLayer.Params()
    params.name = 'fc'
    params.dtype = tf.float64
    params.input_dim = 3
    params.output_dim = 2
    params.batch_norm = False
    fc_layer = layers.FCLayer(params)

    self._FpropBprop(fc_layer, adam_rmsprop_opt)

  def testCompositeOptimizerRaises(self):
    sgd_op = optimizer.SGD.Params()
    adagrad_op = optimizer.Adagrad.Params()
    overlapping_comp_opt = optimizer.CompositeOptimizer.Params().Set(
        optimizer_map={
            'fc/w': (sgd_op, 1.),
            '.': (adagrad_op, 1.),
            'default_optimizer': (adagrad_op, 1.)
        }).Instantiate()

    params = layers.FCLayer.Params()
    params.name = 'fc'
    params.dtype = tf.float64
    params.input_dim = 3
    params.output_dim = 2
    params.batch_norm = False
    fc_layer = layers.FCLayer(params)

    with self.assertRaisesRegex(
        Exception,
        'Variable fc/w/var:0 is matched 2 times by regex',
    ):
      self._FpropBprop(fc_layer, overlapping_comp_opt)

  def testAccumulator(self):
    # testAccumulator compares
    #   - explicit averaging of independently computed var_grads1 and
    #     var_grads2,
    #   - Accumulator(SGD) optimizer effectively doing this over 2 steps.
    np.random.seed(12345)
    np_input1 = np.random.normal(0.1, 0.5, [2, 4, 3])
    np.random.seed(12346)
    np_input2 = np.random.normal(0.1, 0.5, [2, 4, 3])

    tf.random.set_seed(123456)
    params = layers.ProjectionLayer.Params()
    params.name = 'proj'
    params.dtype = tf.float64
    params.input_dim = 3
    params.output_dim = 2
    params.params_init = py_utils.WeightInit.Gaussian(0.01, 123456)

    params.batch_norm = False
    proj_layer = layers.ProjectionLayer(params)
    inputs1 = np_input1
    in_padding1 = tf.zeros([2, 4, 1], dtype=tf.float64)
    inputs2 = np_input2
    in_padding2 = tf.zeros([2, 4, 1], dtype=tf.float64)

    op = optimizer.SGD.Params()
    opt = op.Instantiate()
    # Get `snapshots` of the variables
    vars1 = [v.read_value() for v in proj_layer.vars.Flatten()]

    lr = lambda: 1e-1

    @tf.function
    def _Apply1(proj_layer, opt):
      output1 = proj_layer.FPropDefaultTheta(inputs1, in_padding1)
      output2 = proj_layer.FPropDefaultTheta(inputs2, in_padding2)
      loss1 = tf.reduce_sum(output1)
      loss2 = tf.reduce_sum(output2)
      var_grads1 = py_utils.ComputeGradients(loss1, proj_layer.vars)
      var_grads2 = py_utils.ComputeGradients(loss2, proj_layer.vars)

      _ = opt.Apply(lr, py_utils.ApplyGradMultiplier(var_grads1, 1. / 2.))
      _ = opt.Apply(lr, py_utils.ApplyGradMultiplier(var_grads2, 1. / 2.))

      vars1_1 = proj_layer.vars.Flatten()

      grads1_1 = var_grads1.Transform(tuple)
      grads1_2 = var_grads2.Transform(tuple)

      return vars1_1, grads1_1, grads1_2

    vars1_1, grads1_1, grads1_2 = _Apply1(proj_layer, opt)

    tf.random.set_seed(123456)
    params = layers.ProjectionLayer.Params()
    params.name = 'proj2'
    params.dtype = tf.float64
    params.input_dim = 3
    params.output_dim = 2
    params.params_init = py_utils.WeightInit.Gaussian(0.01, 123456)

    params.batch_norm = False
    proj_layer = layers.ProjectionLayer(params)
    in_padding1 = tf.zeros([2, 4, 1], dtype=tf.float64)

    op = optimizer.Accumulator.Params().Set(
        accum_steps=2, dtype=tf.float64, optimizer_tpl=optimizer.SGD.Params())
    opt = op.Instantiate()
    # Get `snapshots` of the variables
    vars2 = [v.read_value() for v in proj_layer.vars.Flatten()]

    @tf.function(autograph=False)
    def _Apply2(proj_layer, opt):
      inputs1 = np_input1
      output1 = proj_layer.FPropDefaultTheta(inputs1, in_padding1)
      loss2_1 = tf.reduce_sum(output1)
      var_grads2_1 = py_utils.ComputeGradients(loss2_1, proj_layer.vars)
      grads2_1 = var_grads2_1.Transform(tuple)

      inputs1 = np_input2
      output1 = proj_layer.FPropDefaultTheta(inputs1, in_padding1)
      loss2_2 = tf.reduce_sum(output1)
      var_grads2_2 = py_utils.ComputeGradients(loss2_2, proj_layer.vars)
      grads2_2 = var_grads2_2.Transform(tuple)

      with cluster_factory.ForTestingWorker(add_summary=True):
        _ = opt.Apply(lr, var_grads2_1)

      # Get `snapshots` of the intermediate variables
      vars2_intermediate = [v.read_value() for v in proj_layer.vars.Flatten()]
      tf.assign_add(py_utils.GetOrCreateGlobalStepVar(), 1)

      with cluster_factory.ForTestingWorker(add_summary=True):
        _ = opt.Apply(lr, var_grads2_2)

      vars2_1 = proj_layer.vars.Flatten()

      return vars2_intermediate, vars2_1, grads2_1, grads2_2

    vars2_intermediate, vars2_1, grads2_1, grads2_2 = _Apply2(proj_layer, opt)
    # Unlike Graph mode, grads2_1['w'][0]/grads2_2['w'][0] returned from
    # `tf.function` are variables after updates. As a result we cannot compare
    # them with e.g. `vars1`.

    self.assertAllClose(vars1, vars2)

    self.assertAllClose(grads1_1, grads2_1)
    self.assertAllClose(grads1_2, grads2_2)

    self.assertAllClose(vars1, vars2_intermediate)

    lr = lr()
    self.assertAllClose(
        vars1[0] - 0.5 * lr * (grads1_1['w'][1] + grads1_2['w'][1]), vars1_1[0])
    self.assertAllClose(
        vars2[0] - 0.5 * lr * (grads2_1['w'][1] + grads2_2['w'][1]), vars2_1[0])

    self.assertAllClose(vars2, vars2_intermediate)
    self.assertAllClose(vars1_1, vars2_1)
    # TODO(jiaweix): Add checks for the event files from tf.summary
    # once we migrate summary_utils to TF2


if __name__ == '__main__':
  py_utils.SetEagerMode(True)
  tf.test.main()
