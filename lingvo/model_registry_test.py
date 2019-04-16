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
"""Tests for model_registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo import model_registry
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


@model_registry.RegisterSingleTaskModel
class DummyModel(base_model_params.SingleTaskModelParams):

  @classmethod
  def Train(cls):
    p = base_input_generator.BaseInputGenerator.Params()
    p.name = 'Train'
    return p

  @classmethod
  def Dev(cls):
    p = base_input_generator.BaseInputGenerator.Params()
    p.name = 'Dev'
    return p

  @classmethod
  def Test(cls):
    p = base_input_generator.BaseInputGenerator.Params()
    p.name = 'Test'
    return p

  @classmethod
  def Task(cls):
    p = base_model.BaseTask.Params()
    p.name = 'DummyModel'
    return p


class ModelRegistryTest(test_utils.TestCase):

  def setUp(self):
    FLAGS.model_params_override = ''

  def testGetClass(self):

    mp = model_registry.GetClass('test.DummyModel')
    self.assertEqual('Train', mp.Train().name)
    self.assertEqual('Dev', mp.Dev().name)
    self.assertEqual('Test', mp.Test().name)
    self.assertIsNotNone(mp.Task())
    self.assertIsNotNone(mp.Model())

    with self.assertRaises(LookupError):
      # Not yet registered.
      model_registry.GetClass('something.does.not.exist')

  def testGetParams(self):
    cfg = model_registry.GetParams('test.DummyModel', 'Test')
    self.assertIsNotNone(cfg)
    self.assertEqual(DummyModel.Test(), cfg.input)
    cfg.input = None
    # Registered version adds model source info but direct does not.
    cfg.model = None
    self.assertEqual(DummyModel.Model(), cfg)

    with self.assertRaises(LookupError):
      # Not yet registered.
      cfg = model_registry.GetParams('something.does.not.exist', 'Test')

    with self.assertRaises(AttributeError):
      cfg = model_registry.GetParams('test.DummyModel', 'UnknownDataset')

  def testGetParamsCanOverrideWithFlags(self):
    cfg = model_registry.GetParams('test.DummyModel', 'Train')

    FLAGS.model_params_override = (
        'train.max_steps: 10;  train.ema_decay: 0.9\n'
        'train.init_from_checkpoint_rules : {"ckpt": (["abc", "def"], [])}\n')
    cfg2 = model_registry.GetParams('test.DummyModel', 'Train')

    self.assertNotEqual(cfg.train.max_steps, 10)
    self.assertEqual(cfg2.train.max_steps, 10)
    self.assertNotEqual(cfg.train.ema_decay, 0.9)
    self.assertEqual(cfg2.train.ema_decay, 0.9)
    self.assertNotEqual(cfg.train.init_from_checkpoint_rules,
                        {'ckpt': (['abc', 'def'], [])})
    self.assertEqual(cfg2.train.init_from_checkpoint_rules,
                     {'ckpt': (['abc', 'def'], [])})

  def testGetParamsCanOverrideWithFlagsRaises(self):
    FLAGS.model_params_override = 'task.SOME_UNKNOWN_PARAM : 10'
    with self.assertRaises(AttributeError):
      _ = model_registry.GetParams('test.DummyModel', 'Train')

  def testGetParamsCanOverrideInputParamsWithFlags(self):
    cfg = model_registry.GetParams('test.DummyModel', 'Train')
    FLAGS.model_params_override = 'input.num_samples: 100'

    cfg2 = model_registry.GetParams('test.DummyModel', 'Train')
    self.assertNotEqual(cfg.input.num_samples, 100)
    self.assertEqual(cfg2.input.num_samples, 100)

  def testModelParamsIncludeSourceInfo(self):
    path = 'lingvo/model_registry_test.py'
    # NOTE: Only the registered version has source info.
    self.assertIn(path,
                  model_registry.GetParams('test.DummyModel', 'Test').model)

  def testDoubleRegister(self):

    def CreateDuplicate():
      # pylint: disable=unused-variable
      # pylint: disable=function-redefined
      @model_registry.RegisterSingleTaskModel
      class DummyDupl(DummyModel):
        pass

      @model_registry.RegisterSingleTaskModel
      class DummyDupl(DummyModel):
        pass

      # pylint: enable=unused-variable
      # pylint: enable=function-redefined

    with self.assertRaises(ValueError):
      CreateDuplicate()


if __name__ == '__main__':
  tf.test.main()
