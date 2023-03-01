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
from lingvo.core import cluster_factory
from lingvo.core import learner
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS
test_utils.DisableEagerAdapter()  # Tests below runs in both Session/eager mode.


def _GetCheckpointKeys(save_path):
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
class LinearModelParamsWithV1Adam(base_model_params.SingleTaskModelParams):

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
    p.train.ema_decay = 0.999
    return p


@model_registry.RegisterSingleTaskModel
class LinearModelParamsWithV2Adam(base_model_params.SingleTaskModelParams):

  def Train(self):
    return SimpleInputGenerator.Params()

  def Task(self):
    p = LinearModel.Params()
    p.train.learner = [
        learner.Learner.Params().Set(
            name='loss', optimizer=optimizer.AdamV2.Params().Set(name='Adam')
        ),
        learner.Learner.Params().Set(
            name='loss2', optimizer=optimizer.AdamV2.Params().Set(name='Adam2')
        ),
    ]
    p.train.ema_decay = 0.999
    return p


# We cannnot use `variables_to_restore()` becase it will create new EMA
# variables if they don't exist. Here we just want existing EMA variables.
def _GetModelEMAVariablePairs(model, ema):
  res = {}
  for v in model.variables:
    shadow_v = ema.average(v)
    if shadow_v is not None:
      res[v.ref()] = shadow_v

  return res


class EagerCheckpointerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      'test.LinearModelParamsWithV1Adam', 'test.LinearModelParamsWithV2Adam'
  )
  def testEagerEMACheckpointCompatibility(self, model_name):
    self.assertTrue(tf.executing_eagerly())
    cfg = model_registry.GetParams(model_name, 'Train')
    # Use non-zero learning rate so that the weights are updated
    cfg.task.train.learner[0].learning_rate = 0.1
    cfg.task.train.learner[1].learning_rate = 0.1

    eager_v1_logdir = os.path.join(self.get_temp_dir(), 'eager_v1')
    eager_v2_logdir = os.path.join(self.get_temp_dir(), 'eager_v2')
    eager_v2_async_logdir = os.path.join(self.get_temp_dir(), 'eager_v2_async')
    mdl = cfg.Instantiate()

    @tf.function(autograph=False)
    def _Update():
      with py_utils.GradientTape(persistent=True):
        mdl.ConstructFPropBPropGraph()

    # Step 1
    _Update()
    # Save V1 checkpoints at step 1.
    ckpt_v1 = checkpointer.EagerCheckpointerV1(eager_v1_logdir, mdl)
    ckpt_v1.Save(gsteps=1)

    ema = mdl.ema
    model_to_ema_map = _GetModelEMAVariablePairs(mdl, ema)
    model_to_ema_map_snapshot_step1 = {
        k: v.value() for k, v in model_to_ema_map.items()
    }

    # Step 2
    _Update()
    # Save V2 checkpoints at step 2.
    ckpt_v2 = checkpointer.EagerCheckpointerV2(eager_v2_logdir, mdl)
    ckpt_v2.Save(gsteps=2)

    model_to_ema_map = _GetModelEMAVariablePairs(mdl, ema)
    model_to_ema_map_snapshot_step2 = {
        k: v.value() for k, v in model_to_ema_map.items()
    }

    # Step 3
    _Update()
    # Save V2 checkpoints at step 3.
    ckpt_v2_async = checkpointer.EagerCheckpointerV2(
        eager_v2_async_logdir, mdl, experimental_enable_async_checkpoint=True
    )
    ckpt_v2_async.Save(gsteps=3)

    model_to_ema_map = _GetModelEMAVariablePairs(mdl, ema)
    model_to_ema_map_snapshot_step3 = {
        k: v.value() for k, v in model_to_ema_map.items()
    }
    ckpt_v2_async.Sync()

    with cluster_factory.SetEval(True):
      # Restores variables to values saved in `eager_v1_logdir`
      ckpt_v1.Restore()
      # Verify that the EMA variables from V1 checkpoints at step 1 successfully
      # load the model EMA variables to self.theta.
      for v, t in zip(mdl.vars.Flatten(), mdl.theta.Flatten()):
        if v.ref() in model_to_ema_map_snapshot_step1:
          self.assertAllEqual(t, model_to_ema_map_snapshot_step1[v.ref()])

      # Restores variables to values saved in `eager_v2_logdir`
      ckpt_v2.Restore()
      # Verify that the EMA variables from V1 checkpoints at step 2 successfully
      # load the model EMA variables to self.theta.
      for v, t in zip(mdl.vars.Flatten(), mdl.theta.Flatten()):
        if v.ref() in model_to_ema_map_snapshot_step2:
          self.assertAllEqual(t, model_to_ema_map_snapshot_step2[v.ref()])

      # Restores variables to values saved in `eager_v2_async_logdir`
      ckpt_v2_async.Restore()
      # Verify that the EMA variables from V1 checkpoints at step 3 successfully
      # load the model EMA variables to self.theta.
      for v, t in zip(mdl.vars.Flatten(), mdl.theta.Flatten()):
        if v.ref() in model_to_ema_map_snapshot_step3:
          self.assertAllEqual(t, model_to_ema_map_snapshot_step3[v.ref()])

  def testEagerMultiLearnerCheckpointCompatibility(self):
    self.assertTrue(tf.executing_eagerly())
    cfg = model_registry.GetParams('test.LinearModelParamsWithV1Adam', 'Train')
    mdl = cfg.Instantiate()
    # Disable async checkpointing.
    cfg.task.train.async_checkpointing = False
    cfg.train.async_checkpointing = False
    with py_utils.GradientTape(persistent=True):
      mdl.ConstructFPropBPropGraph()

    eager_v1_logdir = os.path.join(self.get_temp_dir(), 'eager_v1')
    eager_v2_logdir = os.path.join(self.get_temp_dir(), 'eager_v2')
    checkpointer.EagerCheckpointerV1(eager_v1_logdir, mdl).Save(gsteps=0)
    checkpointer.EagerCheckpointerV2(eager_v2_logdir, mdl).Save(gsteps=0)
    eager_v1_keys = _GetCheckpointKeys(
        os.path.join(eager_v1_logdir, 'ckpt-00000000'))
    eager_v2_keys = _GetCheckpointKeys(
        os.path.join(eager_v2_logdir, 'ckpt_V2', 'ckpt-0'))
    # Expecting two more variables in V2 checkpoints:
    # _CHECKPOINTABLE_OBJECT_GRAPH
    # save_counter
    self.assertEqual(len(eager_v1_keys) + 2, len(eager_v2_keys))  # pylint:disable=g-generic-assert

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
      checkpointer.Checkpointer(graph_logdir, mdl).Save(sess, gsteps=0)
    graph_keys = _GetCheckpointKeys(os.path.join(graph_logdir, 'ckpt-00000000'))
    self.assertEqual(eager_v1_keys, graph_keys)

  def testEagerCheckpointConsumptionCheck(self):
    self.assertTrue(tf.executing_eagerly())
    cfg = model_registry.GetParams('test.LinearModelParamsWithV1Adam', 'Train')

    eager_v1_logdir = os.path.join(self.get_temp_dir(), 'eager_v1')
    eager_v2_logdir = os.path.join(self.get_temp_dir(), 'eager_v2')
    mdl = cfg.Instantiate()

    @tf.function(autograph=False)
    def _Update():
      with py_utils.GradientTape(persistent=True):
        mdl.ConstructFPropBPropGraph()

    # Step 1
    _Update()
    # Save V1 checkpoint.
    ckpt_v1 = checkpointer.EagerCheckpointerV1(eager_v1_logdir, mdl)
    ckpt_v1.Save(gsteps=1)
    # Save V2 checkpoint.
    ckpt_v2 = checkpointer.EagerCheckpointerV2(eager_v2_logdir, mdl)
    ckpt_v2.Save(gsteps=1)

    mdl_eval = cfg.Instantiate()
    ckpt_eval_v1 = checkpointer.EagerCheckpointerV1(eager_v1_logdir, mdl_eval)
    ckpt_eval_v2 = checkpointer.EagerCheckpointerV2(eager_v2_logdir, mdl_eval)
    ckpt_eval_v2_2 = checkpointer.EagerCheckpointerV2(
        eager_v2_logdir, mdl_eval, check_loading_status=False)
    # Because `mdl_eval` does not have any vars related to training,
    # We expect `EagerCheckpointerV2.Restore` to fail in its matching checks.
    # `EagerCheckpointerV1.Restore`, however, does not have this feature.
    with cluster_factory.SetEval(True):
      # Restores variables to values saved in `eager_v1_logdir`
      ckpt_eval_v1.Restore()

    with cluster_factory.SetEval(True):
      # Restores variables to values saved in `eager_v2_logdir`
      with self.assertRaisesRegex(AssertionError, 'Unresolved object'):
        ckpt_eval_v2.Restore()
      ckpt_eval_v2_2.Restore()


if __name__ == '__main__':
  test_utils.main()
