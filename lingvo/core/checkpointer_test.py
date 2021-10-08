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
"""Tests for checkpointer."""

import os
from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import py_utils
from lingvo.core import test_utils


FLAGS = tf.flags.FLAGS


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


class CheckpointerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('tf_train_saver', False),
      ('custom_saver', True),
  )
  def testSaveRestore(self, use_custom_saver):
    FLAGS.use_custom_saver = use_custom_saver
    train_dir = os.path.join(self.get_temp_dir(), 'testSaveRestore')
    os.mkdir(train_dir)
    p = base_model.SingleTaskModel.Params(LinearModel.Params())
    p.input = base_input_generator.BaseInputGenerator.Params()

    final_global_step = 10
    expected_w = [0.38615, 2.975221, -0.852826]
    initial_b = 1.418741
    final_b = 1234

    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      self.evaluate(tf.global_variables_initializer())
      w, b = self.evaluate([model.GetTask().vars.w, model.GetTask().vars.b])
      self.assertAllClose(expected_w, w)
      self.assertAlmostEqual(initial_b, b, places=5)

      saver = checkpointer.Checkpointer(train_dir, model)
      self.evaluate(
          tf.assign(py_utils.GetOrCreateGlobalStepVar(), final_global_step))
      self.evaluate(tf.assign(model.GetTask().vars.b, final_b))
      saver.Save(sess, model.global_step)

      w, b = self.evaluate([model.GetTask().vars.w, model.GetTask().vars.b])
      self.assertAllClose(expected_w, w)
      self.assertEqual(final_b, b)

    self.assertTrue(
        os.path.isfile(
            os.path.join(train_dir, 'ckpt-%08d.index' % final_global_step)))

    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      saver = checkpointer.Checkpointer(train_dir, model)
      saver.RestoreIfNeeded(sess)

      w, b, global_step = self.evaluate(
          [model.GetTask().vars.w,
           model.GetTask().vars.b, model.global_step])
      self.assertAllClose(expected_w, w)
      self.assertEqual(final_b, b)
      self.assertEqual(final_global_step, global_step)

      # Restore from checkpoint will always work, even though vars are already
      # initialized.
      saver.Restore(sess)

  def testRestoreWithGlobalStepAlreadyInitialized(self):
    train_dir = os.path.join(self.get_temp_dir(),
                             'testRestoreWithGlobalStepAlreadyInitialized')
    os.mkdir(train_dir)
    p = base_model.SingleTaskModel.Params(LinearModel.Params())
    p.input = base_input_generator.BaseInputGenerator.Params()

    with self.session(graph=tf.Graph()) as sess:
      global_step = tf.compat.v1.train.get_or_create_global_step()
      self.evaluate(tf.global_variables_initializer())

      model = p.Instantiate()
      saver = checkpointer.Checkpointer(train_dir, model)

      with self.assertRaises(tf.errors.FailedPreconditionError):
        self.evaluate([model.GetTask().vars.w, model.GetTask().vars.b])

      saver.RestoreIfNeeded(sess)
      w, b, global_step = self.evaluate(
          [model.GetTask().vars.w,
           model.GetTask().vars.b, model.global_step])
      self.assertAllClose([0.38615, 2.975221, -0.852826], w)
      self.assertAlmostEqual(1.418741, b, places=5)
      self.assertEqual(0, global_step)

    self.assertFalse(
        os.path.isfile(os.path.join(train_dir, 'ckpt-00000000.index')))

  def testRestore(self):
    train_dir = os.path.join(self.get_temp_dir(), 'testRestore')
    os.mkdir(train_dir)
    p = base_model.SingleTaskModel.Params(LinearModel.Params())
    p.input = base_input_generator.BaseInputGenerator.Params()

    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      saver = checkpointer.Checkpointer(train_dir, model)

      with self.assertRaises(tf.errors.FailedPreconditionError):
        self.evaluate([model.GetTask().vars.w, model.GetTask().vars.b])

      saver.RestoreIfNeeded(sess)
      w, b, global_step = self.evaluate(
          [model.GetTask().vars.w,
           model.GetTask().vars.b, model.global_step])
      self.assertAllClose([0.38615, 2.975221, -0.852826], w)
      self.assertAlmostEqual(1.418741, b, places=5)
      self.assertEqual(0, global_step)

      with self.assertRaises(AssertionError):
        # When initializing from scratch, variables are expected to not already
        # be initialized.
        saver.Restore(sess)

      # Unless force_reinitialize is used.
      saver.Restore(sess, force_reinitialize=True)

  def testRestoreWithoutCheckpointInitializesVars(self):
    train_dir = os.path.join(self.get_temp_dir(),
                             'testRestoreWithoutCheckpointInitializesVars')
    os.mkdir(train_dir)
    p = base_model.SingleTaskModel.Params(LinearModel.Params())
    p.input = base_input_generator.BaseInputGenerator.Params()

    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      saver = checkpointer.Checkpointer(train_dir, model)

      with self.assertRaises(tf.errors.FailedPreconditionError):
        self.evaluate([model.GetTask().vars.w, model.GetTask().vars.b])

      saver.RestoreIfNeeded(sess)
      w, b, global_step = self.evaluate(
          [model.GetTask().vars.w,
           model.GetTask().vars.b, model.global_step])
      self.assertAllClose([0.38615, 2.975221, -0.852826], w)
      self.assertAlmostEqual(1.418741, b, places=5)
      self.assertEqual(0, global_step)

    self.assertFalse(
        os.path.isfile(os.path.join(train_dir, 'ckpt-00000000.index')))

  def testInitRulesDirectory(self):
    train_dir = os.path.join(self.get_temp_dir(), 'testInitRulesDirectory')
    os.mkdir(train_dir)
    p = base_model.SingleTaskModel.Params(LinearModel.Params())
    p.input = base_input_generator.BaseInputGenerator.Params()
    b1 = 1234
    g1 = 10
    b2 = 12345
    g2 = 100

    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      self.evaluate(tf.global_variables_initializer())
      saver = checkpointer.Checkpointer(train_dir, model)
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), g1))
      self.evaluate(tf.assign(model.GetTask().vars.b, b1))
      saver.Save(sess, model.global_step)
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), g2))
      self.evaluate(tf.assign(model.GetTask().vars.b, b2))
      saver.Save(sess, model.global_step)

    train_dir_2 = os.path.join(self.get_temp_dir(), 'testInitRulesDirectory_2')

    # Set init_checkpoint_rules to only restore b from a specific ckpt
    # the first one, not the latest one.
    rules = [('(.*)', '%s')]
    spec_dir = os.path.join(train_dir, 'ckpt-00000010')
    p.train.init_from_checkpoint_rules = {spec_dir: (rules, [])}
    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      saver = checkpointer.Checkpointer(train_dir_2, model)
      saver.RestoreIfNeeded(sess)
      new_b = self.evaluate(model.GetTask().vars.b)
      self.assertEqual(b1, new_b)

    # Set init_checkpoint_rules to restore all from the latest checkpoint
    # by specifying just the original train directory, not a specific
    # checkpoint.
    rules = [('(.*)', '%s')]
    p.train.init_from_checkpoint_rules = {train_dir: (rules, [])}
    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      saver = checkpointer.Checkpointer(train_dir_2, model)
      saver.RestoreIfNeeded(sess)
      new_b = self.evaluate(model.GetTask().vars.b)
      self.assertEqual(b2, new_b)

  def testSaveOnly(self):
    train_dir = os.path.join(self.get_temp_dir(), 'testSaveOnly')
    os.mkdir(train_dir)
    p = base_model.SingleTaskModel.Params(LinearModel.Params())
    p.input = base_input_generator.BaseInputGenerator.Params()

    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      self.evaluate(tf.global_variables_initializer())
      saver = checkpointer.Checkpointer(train_dir, model, save_only=True)
      saver.Save(sess, model.global_step)
      with self.assertRaises(AssertionError):
        saver.RestoreIfNeeded(sess)

    self.assertTrue(
        os.path.isfile(os.path.join(train_dir, 'ckpt-00000000.index')))


if __name__ == '__main__':
  tf.test.main()
