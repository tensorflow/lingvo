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
"""Tests for models."""

from lingvo import model_imports  # pylint: disable=unused-import
from lingvo import model_registry
# Import DummyModel
from lingvo import model_registry_test  # pylint: disable=unused-import
from lingvo import models_test_helper
import lingvo.compat as tf
from lingvo.core import base_model


class ModelsTest(models_test_helper.BaseModelsTest):

  def testGetModelParams(self):
    name = 'test.DummyModel'
    p = model_registry.GetParams(name, 'Train')
    self.assertTrue(issubclass(p.cls, base_model.SingleTaskModel))


ModelsTest.CreateTestMethodsForAllRegisteredModels(model_registry)


if __name__ == '__main__':
  tf.test.main()
