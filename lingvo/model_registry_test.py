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
"""Tests for model_registry."""
from absl.testing import flagsaver
from absl.testing import parameterized

from lingvo import model_registry
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import program
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


@model_registry.RegisterSingleTaskModel
class DummyModel(base_model_params.SingleTaskModelParams):

  def Train(self):
    p = base_input_generator.BaseInputGenerator.Params()
    p.name = 'Train'
    return p

  def Dev(self):
    p = base_input_generator.BaseInputGenerator.Params()
    p.name = 'Dev'
    return p

  def Test(self):
    p = base_input_generator.BaseInputGenerator.Params()
    p.name = 'Test'
    return p

  def Task(self):
    p = base_model.BaseTask.Params()
    p.name = 'DummyModel'
    return p

  def Dataset(self):
    p = base_input_generator.BaseInputGenerator.Params()
    p.name = 'Dataset'
    return p

  def Task_Dataset(self):
    p = self.Task()
    p.name = 'DatasetSpecificTask'
    return p

  def ProgramSchedule(self):
    p = program.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=1000,
        eval_dataset_names=['Dev', 'Test'],
        eval_steps_per_loop=1,
        decode_steps_per_loop=1)
    p.train_executions_per_eval = 0
    return p


@model_registry.RegisterSingleTaskModel
class DummyModelWithInitRules(DummyModel):

  def Task(self):
    p = super().Task()
    p.train.init_from_checkpoint_rules = {
        '/ckpt/path': ([('abc', 'def')], []),
    }
    return p


class ModelRegistryTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    FLAGS.model_params_override = ''

  def testGetClass(self):

    mp_cls = model_registry.GetClass('test.DummyModel')
    mp = mp_cls()
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
    self.assertEqual(DummyModel().Test(), cfg.input)
    cfg.input = None
    # Registered version adds model source info but direct does not.
    cfg.model = None
    self.assertEqual(DummyModel().Model(), cfg)

    cfg = model_registry.GetParams('test.DummyModel', 'Dataset')
    self.assertIsNotNone(cfg)
    self.assertEqual(DummyModel().Task_Dataset(), cfg.task)

    with self.assertRaises(LookupError):
      # Not yet registered.
      cfg = model_registry.GetParams('something.does.not.exist', 'Test')

    with self.assertRaises(base_model_params.DatasetError):
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

  def testGetParamsOverrideWithInitCheckpointPath(self):
    # Without override, default value is None.
    cfg = model_registry.GetParams('test.DummyModel', 'Train')
    self.assertIsNone(cfg.task.train.init_from_checkpoint_override)
    # Override ckpt path from empty to flag.
    FLAGS.model_params_override = (
        'task.train.init_from_checkpoint_override:/new/ckpt/path')
    cfg1 = model_registry.GetParams('test.DummyModel', 'Train')
    self.assertEqual(cfg1.task.train.init_from_checkpoint_override,
                     '/new/ckpt/path')
    # Unset checkpoint path.
    FLAGS.model_params_override = ('task.train.init_from_checkpoint_override:')
    cfg2 = model_registry.GetParams('test.DummyModelWithInitRules', 'Train')
    self.assertEqual(cfg2.task.train.init_from_checkpoint_override, '')

  def testGetParamsCanOverrideWithFlagsRaises(self):
    FLAGS.model_params_override = 'task.SOME_UNKNOWN_PARAM : 10'
    with self.assertRaises(AttributeError):
      _ = model_registry.GetParams('test.DummyModel', 'Train')

  def testGetParamsCanOverrideWithFlagsBadSyntax(self):
    FLAGS.model_params_override = 'task.SOME_UNKNOWN_PARAM=10'
    with self.assertRaises(ValueError):
      _ = model_registry.GetParams('test.DummyModel', 'Train')

  def testGetParamsCanOverrideInputParamsWithFlags(self):
    cfg = model_registry.GetParams('test.DummyModel', 'Train')
    FLAGS.model_params_override = 'input.num_samples: 100'

    cfg2 = model_registry.GetParams('test.DummyModel', 'Train')
    self.assertNotEqual(cfg.input.num_samples, 100)
    self.assertEqual(cfg2.input.num_samples, 100)

  def _CheckProgramParams(self, eval_programs, expt_eval_dev, expt_eval_test,
                          expt_decode_dev, expt_decode_test):
    eval_dev, eval_test, decode_dev, decode_test = 0, 0, 0, 0
    for eval_program in eval_programs:
      if eval_program.dataset_name == 'Dev':
        if issubclass(eval_program.cls, program.EvalProgram):
          self.assertEqual(eval_program.name, 'eval_tpu')
          eval_dev += 1
        elif issubclass(eval_program.cls, program.DecodeProgram):
          self.assertEqual(eval_program.name, 'decode_tpu')
          decode_dev += 1
      elif eval_program.dataset_name == 'Test':
        if issubclass(eval_program.cls, program.EvalProgram):
          self.assertEqual(eval_program.name, 'eval_tpu')
          eval_test += 1
        elif issubclass(eval_program.cls, program.DecodeProgram):
          self.assertEqual(eval_program.name, 'decode_tpu')
          decode_test += 1
    self.assertEqual(eval_dev, expt_eval_dev)
    self.assertEqual(eval_test, expt_eval_test)
    self.assertEqual(decode_dev, expt_decode_dev)
    self.assertEqual(decode_test, expt_decode_test)

  @parameterized.named_parameters(
      ('Basic',),
      ('DevOnly', 'Dev', 0, 3, -1),
      ('OverrideExecutions', None, 1, None, None),
      ('DecodeOnly', None, None, 0, None),
      ('EvalOnly', None, None, None, 0),
  )
  def testProgramSchedule(self,
                          dataset_list_override=None,
                          train_executions_per_eval_override=None,
                          eval_steps_per_loop_override=None,
                          decode_steps_per_loop_override=None):
    with flagsaver.flagsaver(
        executor_datasets_to_eval=dataset_list_override,
        executor_train_executions_per_eval=train_executions_per_eval_override,
        executor_eval_steps_per_loop=eval_steps_per_loop_override,
        executor_decode_steps_per_loop=decode_steps_per_loop_override):
      ps_params = model_registry.GetProgramSchedule('test.DummyModel')

      if dataset_list_override is not None:
        self.assertAllEqual(ps_params.dataset_names,
                            dataset_list_override.split(';'))
      else:
        self.assertAllEqual(ps_params.dataset_names, ['Dev', 'Test'])

      if train_executions_per_eval_override is not None:
        self.assertEqual(ps_params.train_executions_per_eval,
                         train_executions_per_eval_override)
      else:
        self.assertEqual(ps_params.train_executions_per_eval, 0)

      # Assume only Dev and Test are avaiable eval datasets.
      eval_dev, eval_test, decode_dev, decode_test = 0, 0, 0, 0
      if dataset_list_override is None or 'Dev' in dataset_list_override:
        if eval_steps_per_loop_override != 0:
          eval_dev += 1
        if decode_steps_per_loop_override != 0:
          decode_dev += 1
      if dataset_list_override is None or 'Test' in dataset_list_override:
        if eval_steps_per_loop_override != 0:
          eval_test += 1
        if decode_steps_per_loop_override != 0:
          decode_test += 1
      self.assertLen(ps_params.eval_programs,
                     eval_dev + decode_dev + eval_test + decode_test)
      self._CheckProgramParams(ps_params.eval_programs, eval_dev, eval_test,
                               decode_dev, decode_test)

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
