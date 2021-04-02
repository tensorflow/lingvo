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
"""Tests for lingvo.core.learner."""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import gradient_combiner
from lingvo.core import layers
from lingvo.core import learner
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import test_utils


class TestLayer(base_layer.BaseLayer):

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    pc = py_utils.WeightParams(
        shape=[],
        init=py_utils.WeightInit.Constant(0),
        dtype=self.params.dtype,
        collections=self._VariableCollections())
    self.CreateVariable('hello', pc)
    self.CreateVariable('world', pc)
    self.CreateVariable('moon', pc)
    self.CreateVariable('mars', pc, trainable=False)

  def Loss(self, theta):
    return self.MainLoss(theta) + self.AuxLoss(theta)

  def MainLoss(self, theta):
    return theta.hello

  def AuxLoss(self, theta):
    return -2 * theta.world


class TestSGD(optimizer.SGD):
  """Test optimizer which has its own variables."""

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    pc = py_utils.WeightParams(
        shape=[],
        init=py_utils.WeightInit.Constant(0),
        dtype=self.params.dtype,
        collections=self._VariableCollections())
    self.CreateVariable('ext', pc)


class GradientSum(gradient_combiner.GradientCombiner):

  def Combine(self, vmap, losses_and_grads):
    """Computes the sum of gradients on the variables."""

    def GradSum(v, *gs):
      tf.logging.info('GradSum: %s: %s', v, gs)
      if all(g is None for g in gs):
        return None
      return tf.add_n([g for g in gs if g is not None])

    grads = [l_and_g.grads for l_and_g in losses_and_grads.values()]
    tf.logging.info('grads: %s', grads)
    return tf.nest.map_structure(GradSum, vmap, *grads), {}


class LearnerTest(test_utils.TestCase):

  def testBasic(self):
    layer = TestLayer.Params().Set(name='test').Instantiate()
    learner_p = learner.Learner.Params().Set(
        name='learner', learning_rate=.1, optimizer=optimizer.SGD.Params())
    var_grads, updated_vars, _ = self._testLearner(layer, learner_p)
    tf.logging.info('var_grads=%s, updated_vars=%s', var_grads, updated_vars)
    self.assertAllClose(var_grads, {'hello': (0., 1.), 'world': (0., -2.)})
    self.assertAllClose(updated_vars, {
        'hello': -0.1,
        'world': 0.2,
        'moon': 0.,
        'mars': 0.
    })

  def testMultiLoss(self):
    layer = TestLayer.Params().Set(name='test').Instantiate()
    learner_p = learner.Learner.Params().Set(
        name='learner', learning_rate=.1, optimizer=optimizer.SGD.Params())
    learner_p.loss_name = ('main_loss', 'aux_loss')
    learner_p.gradient_combiner = GradientSum.Params()
    var_grads, updated_vars, _ = self._testLearner(layer, learner_p)
    self.assertAllClose(var_grads, {'hello': (0., 1.), 'world': (0., -2.)})
    self.assertAllClose(updated_vars, {
        'hello': -0.1,
        'world': 0.2,
        'moon': 0.,
        'mars': 0.
    })

  def testBPropVariableFilter(self):
    layer = TestLayer.Params().Set(name='test').Instantiate()
    learner_p = learner.Learner.Params().Set(
        name='learner',
        learning_rate=.1,
        optimizer=optimizer.SGD.Params(),
        bprop_variable_filter='ello')
    var_grads, updated_vars, eval_metrics = self._testLearner(layer, learner_p)
    # Only 'hello' is updated.
    self.assertAllClose(var_grads, {'hello': (0., 1.)})
    self.assertAllClose(updated_vars, {
        'hello': -0.1,
        'world': 0.,
        'moon': 0.,
        'mars': 0.
    })
    self.assertIn('grad_scale_all', eval_metrics)

  def testBPropVariableExclusion(self):
    layer = TestLayer.Params().Set(name='test').Instantiate()
    learner_p = learner.Learner.Params().Set(
        name='learner',
        learning_rate=.1,
        optimizer=optimizer.SGD.Params(),
        bprop_variable_filter='o',
        bprop_variable_exclusion='ello')
    var_grads, updated_vars, _ = self._testLearner(layer, learner_p)
    # Only 'world' is updated.
    self.assertAllClose(var_grads, {'world': (0., -2.)})
    self.assertAllClose(updated_vars, {
        'hello': 0.,
        'world': 0.2,
        'moon': 0.,
        'mars': 0.
    })

  def testMultiLearner(self):
    layer = TestLayer.Params().Set(name='test').Instantiate()
    # Set optimizer, lr_schedule and grad_norm_tracker whice have their own
    # variables.
    learner1_p = learner.Learner.Params().Set(
        name='learner1',
        learning_rate=.1,
        optimizer=TestSGD.Params(),
        lr_schedule=schedule.DevBasedSchedule.Params(),
        grad_norm_tracker=layers.GradNormTracker.Params())
    var_grads1, updated_vars1, _ = self._testLearner(layer, learner1_p)
    learner2_p = learner1_p.Copy().Set(name='learner2')
    var_grads2, updated_vars2, _ = self._testLearner(layer, learner2_p)
    self.assertAllClose(var_grads1, var_grads2)
    self.assertAllClose(updated_vars1, updated_vars2)

  def _testLearner(self, layer, learner_p):
    tf.train.get_or_create_global_step()  # needed for lr_schedule
    lrnr = learner_p.Instantiate()
    if isinstance(learner_p.loss_name, (list, tuple)):
      main_loss = layer.MainLoss(layer.theta)
      aux_loss = layer.AuxLoss(layer.theta)
      metrics = {'main_loss': (main_loss, 1.), 'aux_loss': (aux_loss, 1.)}
      expected_losses = [main_loss, aux_loss]
    else:
      loss = layer.Loss(layer.theta)
      metrics = {learner_p.name: (loss, 1.)}
      expected_losses = [loss]
    losses, update_op, eval_metrics = lrnr.Apply(metrics, layer.vars)
    self.assertAllEqual(losses, expected_losses)
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      var_grads = self.evaluate(lrnr.GetVarGrads().Transform(tuple))
      update_op.run()
      updated_vars = self.evaluate(layer.vars)
      return var_grads, updated_vars, eval_metrics


if __name__ == '__main__':
  tf.test.main()
