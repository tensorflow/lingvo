# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for checkpointer."""

import os
from absl.testing import parameterized
from lingvo import model_registry
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import checkpointer
from lingvo.core import learner
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


def _get_checkpoint_keys(save_path):
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  return set(shapes.keys())


class SimpleInputGenerator(base_input_generator.BaseInputGenerator):

  def GetPreprocessedInputBatch(self):
    return py_utils.NestedMap(x=tf.constant([1]))


class LinearModel(base_model.BaseTask):
  """A basic linear model."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'linear_model'
    return p

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    w = py_utils.WeightParams(
        shape=[3],
        init=py_utils.WeightInit.Gaussian(scale=1.0, seed=123456),
        dtype=p.dtype)
    b = py_utils.WeightParams(
        shape=[],
        init=py_utils.WeightInit.Gaussian(scale=1.0, seed=234567),
        dtype=p.dtype)
    self.CreateVariable('w', w)
    self.CreateVariable('b', b)

  def FPropTower(self, theta, unused_input_batch):
    return py_utils.NestedMap(
        loss=(tf.reduce_sum(theta.w) + theta.b, 1.0),
        loss2=(tf.reduce_sum(theta.w) - theta.b, 1.0)), py_utils.NestedMap()


@model_registry.RegisterSingleTaskModel
class LinearModelParams(base_model_params.SingleTaskModelParams):

  def Train(self):
    return SimpleInputGenerator.Params()

  def Task(self):
    p = LinearModel.Params()
    p.train.learner = [
        learner.Learner.Params().Set(
            name='loss', optimizer=optimizer.Adam.Params().Set(name='Adam')),
        learner.Learner.Params().Set(
            name='loss2', optimizer=optimizer.Adam.Params().Set(name='Adam2'))
    ]
    return p


class EagerCheckpointerTest(test_utils.TestCase, parameterized.TestCase):

  def testEagerMultiLearnerCheckpointCompatibility(self):
    self.assertTrue(tf.executing_eagerly())
    cfg = model_registry.GetParams('test.LinearModelParams', 'Train')

    eager_logdir = os.path.join(self.get_temp_dir(), 'eager')
    mdl = cfg.Instantiate()
    with py_utils.GradientTape(persistent=True):
      mdl.ConstructFPropBPropGraph()
    checkpointer.EagerCheckpointerV1(eager_logdir, mdl).Save(gsteps=0)
    eager_keys = _get_checkpoint_keys(
        os.path.join(eager_logdir, 'ckpt_V1', 'ckpt-0'))

    py_utils.SetEagerMode(False)
    self.assertFalse(tf.executing_eagerly())
    graph_logdir = os.path.join(self.get_temp_dir(), 'graph')
    os.mkdir(graph_logdir)
    with self.session(graph=tf.Graph()) as sess:
      mdl = cfg.Instantiate()
      for lrn in mdl.GetTask().learners:
        lrn.optimizer.params.clear_variable_scope = False
      mdl.ConstructFPropBPropGraph()
      sess.run(tf.global_variables_initializer())
      checkpointer.Checkpointer(graph_logdir, mdl).Save(sess)
    graph_keys = _get_checkpoint_keys(os.path.join(graph_logdir, 'ckpt'))

    self.assertEqual(eager_keys, graph_keys)


if __name__ == '__main__':
  py_utils.SetEagerMode(True)
  tf.test.main()
