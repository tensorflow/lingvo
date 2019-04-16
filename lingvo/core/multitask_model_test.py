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
"""Tests for multitask_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import hyperparams
from lingvo.core import multitask_model
from lingvo.core import py_utils
from lingvo.core import test_utils


class MultiTaskModelTest(test_utils.TestCase):

  class _TestTask(base_model.BaseTask):

    @classmethod
    def Params(cls):
      p = super(MultiTaskModelTest._TestTask, cls).Params()
      p.name = 'test_task'
      p.encoder = base_layer.BaseLayer.Params()
      p.encoder.name = 'enc'
      p.decoder = base_layer.BaseLayer.Params()
      p.decoder.name = 'dec'
      return p

    @base_layer.initializer
    def __init__(self, params):
      super(MultiTaskModelTest._TestTask, self).__init__(params)
      p = self.params
      if p.encoder:
        self.CreateChild('encoder', p.encoder)
      if p.decoder:
        self.CreateChild('decoder', p.decoder)

  def testSharedEncoderModel(self):
    p = multitask_model.SharedEncoderModel.Params()
    p.name = 'test'
    p.encoder_to_share = 'p0'

    p0 = MultiTaskModelTest._TestTask.Params()
    p1 = MultiTaskModelTest._TestTask.Params()
    p1.encoder = None

    p.input = base_model_params.MultiTaskModelParams.Train()
    p.input.Define('p0', base_input_generator.BaseInputGenerator.Params(), '')
    p.input.Define('p1', base_input_generator.BaseInputGenerator.Params(), '')
    p.task_params = hyperparams.Params()
    p.task_params.Define('p0', p0, '')
    p.task_params.Define('p1', p1, '')
    p.task_probs = hyperparams.Params()
    p.task_probs.Define('p0', 0.5, '')
    p.task_probs.Define('p1', 0.5, '')

    model = p.cls(p)
    self.assertEqual(model.p0.encoder, model.p1.encoder)

  def testSharedEncoderDecoderModel(self):
    p = multitask_model.SharedEncoderDecoderModel.Params()
    p.name = 'test'
    p.encoder_to_share = 'p0'
    p.decoder_to_share = 'p0'

    p0 = MultiTaskModelTest._TestTask.Params()
    p1 = MultiTaskModelTest._TestTask.Params()
    p1.encoder = None
    p1.decoder = None

    p.input = base_model_params.MultiTaskModelParams.Train()
    p.input.Define('p0', base_input_generator.BaseInputGenerator.Params(), '')
    p.input.Define('p1', base_input_generator.BaseInputGenerator.Params(), '')
    p.task_params = hyperparams.Params()
    p.task_params.Define('p0', p0, '')
    p.task_params.Define('p1', p1, '')
    p.task_probs = hyperparams.Params()
    p.task_probs.Define('p0', 0.5, '')
    p.task_probs.Define('p1', 0.5, '')

    model = p.cls(p)
    self.assertEqual(model.p0.encoder, model.p1.encoder)
    self.assertEqual(model.p0.decoder, model.p1.decoder)

  class _TestTaskWithVars(base_model.BaseTask):

    @classmethod
    def Params(cls):
      p = super(MultiTaskModelTest._TestTaskWithVars, cls).Params()
      p.name = 'test_task'
      return p

    @base_layer.initializer
    def __init__(self, params):
      super(MultiTaskModelTest._TestTaskWithVars, self).__init__(params)
      pc = py_utils.WeightParams(shape=[10, 10], dtype=tf.float32)
      self.CreateVariable('weight', pc)

  def testRegExSharedVariableModel(self):
    p = multitask_model.RegExSharedVariableModel.Params()
    p.name = 'test'
    p.variable_renaming_rules = [('p./(.*)', 'shared/%s')]

    p0 = MultiTaskModelTest._TestTaskWithVars.Params()
    p1 = MultiTaskModelTest._TestTaskWithVars.Params()

    p.input = base_model_params.MultiTaskModelParams.Train()
    p.input.Define('p0', base_input_generator.BaseInputGenerator.Params(), '')
    p.input.Define('p1', base_input_generator.BaseInputGenerator.Params(), '')

    p.task_params = hyperparams.Params()
    p.task_params.Define('p0', p0, '')
    p.task_params.Define('p1', p1, '')
    p.task_probs = hyperparams.Params()

    p.task_probs.Define('p0', 0.5, '')
    p.task_probs.Define('p1', 0.5, '')

    model = p.cls(p)
    all_vars = model.vars
    self.assertEqual('shared/weight/var:0', all_vars.p0.weight.name)
    self.assertEqual('shared/weight/var:0', all_vars.p1.weight.name)


if __name__ == '__main__':
  tf.test.main()
