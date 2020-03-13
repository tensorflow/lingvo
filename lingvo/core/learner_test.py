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
"""Tests for lingvo.core.learner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import learner
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import test_utils


class TestLayer(base_layer.BaseLayer):

  @base_layer.initializer
  def __init__(self, params):
    super(TestLayer, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      pc = py_utils.WeightParams(
          shape=[],
          init=py_utils.WeightInit.Constant(0),
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('hello', pc)
      self.CreateVariable('world', pc)

  def Loss(self, theta):
    return theta.hello + -2 * theta.world


class LearnerTest(test_utils.TestCase):

  def testBasic(self):
    learner_p = learner.Learner.Params().Set(
        name='learner', learning_rate=.1, optimizer=optimizer.SGD.Params())
    var_grads, updated_vars, _ = self._testLearner(learner_p)
    self.assertAllClose(var_grads, {'hello': (0., 1.), 'world': (0., -2.)})
    self.assertAllClose(updated_vars, {'hello': -0.1, 'world': 0.2})

  def testBPropVariableFilter(self):
    learner_p = learner.Learner.Params().Set(
        name='learner',
        learning_rate=.1,
        optimizer=optimizer.SGD.Params(),
        bprop_variable_filter='ello')
    var_grads, updated_vars, eval_metrics = self._testLearner(learner_p)
    # Only 'hello' is updated.
    self.assertAllClose(var_grads, {'hello': (0., 1.)})
    self.assertAllClose(updated_vars, {'hello': -0.1, 'world': 0.})
    self.assertIn('grad_scale_all', eval_metrics)

  def testBPropVariableExclusion(self):
    learner_p = learner.Learner.Params().Set(
        name='learner',
        learning_rate=.1,
        optimizer=optimizer.SGD.Params(),
        bprop_variable_filter='o',
        bprop_variable_exclusion='ello')
    var_grads, updated_vars, _ = self._testLearner(learner_p)
    # Only 'world' is updated.
    self.assertAllClose(var_grads, {'world': (0., -2.)})
    self.assertAllClose(updated_vars, {'hello': 0., 'world': 0.2})

  def _testLearner(self, learner_p):
    tf.train.get_or_create_global_step()  # needed for lr_schedule
    lrnr = learner_p.Instantiate()
    layer = TestLayer.Params().Set(name='test').Instantiate()
    loss = layer.Loss(layer.theta)
    update_op, eval_metrics = lrnr.Apply(loss, layer.vars)
    with self.session() as sess:
      tf.global_variables_initializer().run()
      var_grads = sess.run(lrnr.GetVarGrads().Transform(tuple))
      update_op.run()
      updated_vars = sess.run(layer.vars)
      return var_grads, updated_vars, eval_metrics


if __name__ == '__main__':
  tf.test.main()
