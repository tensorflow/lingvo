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
"""Helper for models_test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import base_model
from lingvo.core import base_input_generator


class BaseModelsTest(tf.test.TestCase):
  """Base model test class which does not define any test methods of its own."""

  def _testOneModelParams(self, registry, name):
    cls = registry.GetClass(name)
    p = cls.Model()
    self.assertTrue(issubclass(p.cls, base_model.BaseModel))
    self.assertTrue(p.model is not None)
    for dataset in ('Train', 'Dev', 'Test'):
      input_p = cls.GetDatasetParams(dataset)
      if issubclass(p.cls, base_model.SingleTaskModel):
        self.assertTrue(
            issubclass(input_p.cls, base_input_generator.BaseInputGenerator),
            'Error in %s' % dataset)
        if (dataset != 'Train') and issubclass(
            input_p.cls, base_input_generator.BaseSequenceInputGenerator) and (
                input_p.num_samples != 0):
          self.assertEquals(
              input_p.num_batcher_threads, 1,
              'num_batcher_threads too large in %s. Decoder '
              'or eval runs over this set might not span '
              'exactly one epoch.' % dataset)
      else:
        self.assertTrue(issubclass(p.cls, base_model.MultiTaskModel))
        for _, v in input_p.IterParams():
          self.assertTrue(
              issubclass(v.cls, base_input_generator.BaseInputGenerator),
              'Error in %s' % dataset)

  @classmethod
  def CreateTestMethodsForAllRegisteredModels(cls, registry):
    """Programmatically defines test methods for each registered model."""
    model_names = registry.GetAllRegisteredClasses().keys()
    for model_name in sorted(model_names):

      def test(self, name=model_name):
        self._testOneModelParams(registry, name)  # pylint: disable=protected-access

      setattr(cls, 'testModelParams_%s' % model_name, test)
