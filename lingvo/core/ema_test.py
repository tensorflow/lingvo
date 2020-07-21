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
"""Tests for ExponentialMovingAverage support in Lingvo."""

import os
import lingvo.compat as tf
from lingvo.core import base_decoder
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np


class TestTask(base_model.BaseTask):
  """A task with a single 'encoder' child layer."""

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('encoder', p.encoder)


class EmaTest(test_utils.TestCase):

  @classmethod
  def TestParams(cls, encoder_params):
    p = TestTask.Params()
    p.name = 'base_mdl'
    p.input = base_input_generator.BaseSequenceInputGenerator.Params()
    p.encoder = encoder_params
    p.decoder = base_decoder.BaseDecoder.Params()
    return p

  def testBatchNormLayer(self):
    p = base_model.SingleTaskModel.Params()
    p.task = self.TestParams(layers.BatchNormLayer.Params().Set(dim=1))
    p.task.train.ema_decay = 0.9
    p.task.train.ema_decay_moving_vars = True
    model = p.Instantiate()
    self.assertIsNotNone(model.ema)
    task = model._task
    task._train_op = tf.no_op()
    task.ApplyExponentialMovingAverage(model.ema)

    layer = task.encoder
    self.assertLen(layer.vars, 4)
    for var in layer.vars.Flatten():
      self.assertIsNotNone(model.ema.average(var), msg=var.name)
    beta = layer.vars.beta
    mean = layer.vars.moving_mean

    global_step = 100
    beta_1 = np.asarray([.2])
    mean_1 = np.asarray([.03])
    beta_1_ema = beta_1 * .1
    mean_1_ema = mean_1 * .1
    with self.session() as sess:
      # Test EMA values.
      self.evaluate(tf.global_variables_initializer())
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), global_step))
      self.evaluate(tf.assign(beta, beta_1))
      self.evaluate(tf.assign(mean, mean_1))
      self.evaluate(task._post_train_ops)

      self.assertAllClose([beta_1, beta_1_ema, mean_1, mean_1_ema],
                          self.evaluate([
                              beta,
                              model.ema.average(beta), mean,
                              model.ema.average(mean)
                          ]))

      # Test checkpointer.
      train_dir = os.path.join(self.get_temp_dir(), 'testSaveRestore')
      os.mkdir(train_dir)
      saver = checkpointer.Checkpointer(train_dir, model)
      saver.Save(sess, model.global_step)

      self.assertTrue(
          os.path.isfile(
              os.path.join(train_dir, 'ckpt-%08d.index' % global_step)))

    # Restore from ckpt in training mode.
    with self.session(graph=tf.Graph()) as sess:
      model = p.Instantiate()
      self.assertIsNotNone(model.ema)
      task = model._task
      task._train_op = tf.no_op()
      task.ApplyExponentialMovingAverage(model.ema)
      layer = task.encoder
      for var in layer.vars.Flatten():
        self.assertIsNotNone(model.ema.average(var), msg=var.name)
      beta = layer.vars.beta
      mean = layer.vars.moving_mean

      saver = checkpointer.Checkpointer(train_dir, model)
      saver.RestoreIfNeeded(sess)

      self.assertAllClose([beta_1, beta_1_ema, mean_1, mean_1_ema],
                          self.evaluate([
                              beta,
                              model.ema.average(beta), mean,
                              model.ema.average(mean)
                          ]))

    # Restore from ckpt in eval mode.
    with self.session(graph=tf.Graph()) as sess, self.SetEval(True):
      model = p.Instantiate()
      self.assertIsNotNone(model.ema)
      task = model._task
      # task._train_op = tf.no_op()
      # task.ApplyExponentialMovingAverage(model.ema)
      layer = task.encoder
      # for var in layer.vars.Flatten():
      #   self.assertIsNotNone(model.ema.average(var), msg=var.name)
      beta = layer.vars.beta
      mean = layer.vars.moving_mean

      saver = checkpointer.Checkpointer(train_dir, model)
      saver.RestoreIfNeeded(sess)

      # Both beta and mean should use the EMA value.
      self.assertAllClose([beta_1_ema, mean_1_ema], self.evaluate([beta, mean]))


if __name__ == '__main__':
  tf.test.main()
