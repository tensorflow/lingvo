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
from lingvo.core import schedule
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
    task = self.TestParams(layers.BatchNormLayer.Params().Set(dim=1))
    task.train.ema_decay = 0.9
    task.train.ema_decay_moving_vars = True
    p = base_model.SingleTaskModel.Params(task)
    model = p.Instantiate()
    self.assertIsNotNone(model.ema)
    task = model._task

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
      self.evaluate(task.ApplyExponentialMovingAverage())

      self.assertAllClose([beta_1, beta_1_ema, mean_1, mean_1_ema],
                          self.evaluate([
                              beta,
                              model.ema.average(beta), mean,
                              model.ema.average(mean)
                          ]))

      # Test checkpointer.
      train_dir = os.path.join(self.get_temp_dir(), 'testSaveRestore')
      os.mkdir(train_dir)
      if py_utils.IsEagerMode():
        saver = checkpointer.EagerCheckpointerV1(train_dir, model)
        sess = None
      else:
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
      layer = task.encoder
      for var in layer.vars.Flatten():
        self.assertIsNotNone(model.ema.average(var), msg=var.name)
      beta = layer.vars.beta
      mean = layer.vars.moving_mean

      if py_utils.IsEagerMode():
        saver = checkpointer.EagerCheckpointerV1(train_dir, model)
        sess = None
      else:
        saver = checkpointer.Checkpointer(train_dir, model)
      saver.Restore(sess)

      self.assertAllClose([beta_1, beta_1_ema, mean_1, mean_1_ema],
                          self.evaluate([
                              beta,
                              model.ema.average(beta), mean,
                              model.ema.average(mean)
                          ]))

      # In train, var and theta are same (except for VN).
      theta_beta = layer.theta.beta
      theta_mean = layer.theta.moving_mean
      self.assertAllClose([beta_1, mean_1],
                          self.evaluate([theta_beta, theta_mean]))

    # Restore from ckpt in eval mode.
    # To use EMA variables as theta. See BaseLayer._InternalGetTheta.
    with self.session(graph=tf.Graph()) as sess, self.SetEval(True):
      model = p.Instantiate()
      self.assertIsNotNone(model.ema)
      task = model._task
      layer = task.encoder
      for var in layer.vars.Flatten():
        self.assertIsNotNone(model.ema.average(var), msg=var.name)
      beta = layer.vars.beta
      mean = layer.vars.moving_mean
      # layer.theta is EMA variable.
      beta_ema = layer.theta.beta
      mean_ema = layer.theta.moving_mean

      if py_utils.IsEagerMode():
        saver = checkpointer.EagerCheckpointerV1(train_dir, model)
        sess = None
      else:
        saver = checkpointer.Checkpointer(train_dir, model)
      saver.Restore(sess)

      self.assertAllClose([beta_1, beta_1_ema, mean_1, mean_1_ema],
                          self.evaluate([beta, beta_ema, mean, mean_ema]))

  def testEmaSchedule(self):
    task = self.TestParams(layers.BatchNormLayer.Params().Set(dim=1))
    task.train.ema_decay = 0
    # Note: EMA = decay * EMA + (1 - decay) * var
    ema_off = 1.0  # ema keeps constant.
    ema_is_var = 0.0  # ema copys var value.
    task.train.ema_schedule = schedule.PiecewiseConstantSchedule.Params().Set(
        boundaries=[99, 199], values=[ema_off, 0.9, ema_is_var])
    task.train.ema_decay_moving_vars = True
    p = base_model.SingleTaskModel.Params(task)
    model = p.Instantiate()
    self.assertIsNotNone(model.ema)
    self.assertIsNotNone(model.ema_decay)
    task = model._task

    layer = task.encoder
    self.assertLen(layer.vars, 4)
    for var in layer.vars.Flatten():
      self.assertIsNotNone(model.ema.average(var), msg=var.name)
    beta = layer.vars.beta
    mean = layer.vars.moving_mean

    beta_0 = np.asarray([0.])
    mean_0 = np.asarray([0.])
    beta_1 = np.asarray([.2])
    mean_1 = np.asarray([.03])
    beta_1_ema = beta_1 * .1
    mean_1_ema = mean_1 * .1
    # Check EMA decay schedul in Train.
    with self.session():
      # Test EMA values.
      self.evaluate(tf.global_variables_initializer())
      # var is initialized as 0, and EMA assigns the var value.
      self.assertAllClose([beta_0, beta_0, mean_0, mean_0],
                          self.evaluate([
                              beta,
                              model.ema.average(beta), mean,
                              model.ema.average(mean)
                          ]))

      # At step=1, ema_decay=1.0 by ema_schedule. EMA update is off.
      global_step = 1
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), global_step))
      self.evaluate(tf.assign(beta, beta_1))
      self.evaluate(tf.assign(mean, mean_1))
      ema_op = task.ApplyExponentialMovingAverage()
      self.evaluate(ema_op)
      self.assertAllClose([beta_1, beta_0, mean_1, mean_0],
                          self.evaluate([
                              beta,
                              model.ema.average(beta), mean,
                              model.ema.average(mean)
                          ]))

      # At step=100, ema_decay=0.9 by ema_schedule.
      global_step = 100
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), global_step))
      if py_utils.IsEagerMode():
        ema_op = task.ApplyExponentialMovingAverage()
      else:
        self.evaluate(ema_op)
      self.assertAllClose([beta_1, beta_1_ema, mean_1, mean_1_ema],
                          self.evaluate([
                              beta,
                              model.ema.average(beta), mean,
                              model.ema.average(mean)
                          ]))

      # At step=200, ema_decay=0.0 by ema_schedule. EMA copies var value.
      global_step = 200
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), global_step))
      if py_utils.IsEagerMode():
        ema_op = task.ApplyExponentialMovingAverage()
      else:
        self.evaluate(ema_op)
      self.assertAllClose([beta_1, beta_1, mean_1, mean_1],
                          self.evaluate([
                              beta,
                              model.ema.average(beta), mean,
                              model.ema.average(mean)
                          ]))

if __name__ == '__main__':
  test_utils.main()
