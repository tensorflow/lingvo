# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import re

from absl.testing import absltest
from lingvo.jax import base_input
from lingvo.jax import base_task


class ModelsTestHelper(absltest.TestCase):
  """Helper class for testing model configurations."""

  _PREFIX = 'lingvo.'

  def _test_one_model_params(self, registry, name):
    """Performs basic checks on a model configuration."""
    if name.startswith(self._PREFIX):
      name = name[len(self._PREFIX):]

    model_params = registry.get_model(name)()

    task_p = model_params.task()
    task = task_p.Instantiate()
    self.assertIsInstance(task, base_task.BaseTask)

    dataset_splits = model_params.datasets()
    # Registered model configurations must have at least a dataset split.
    self.assertNotEmpty(dataset_splits)
    for s in dataset_splits:
      self.assertIsInstance(s, base_input.BaseInputParams)
      # Note: Creating the input generator may require data access.

  @classmethod
  def create_test_methods_for_all_registered_models(cls,
                                                    registry,
                                                    task_regexes=None,
                                                    exclude_regexes=None):
    """Programmatically defines test methods for each registered model."""
    task_regexes = task_regexes or []
    exclude_regexes = exclude_regexes or []
    model_names = list(registry.get_all_registered_models().keys())
    print(f'Creating tests for {task_regexes}, excluding {exclude_regexes}')
    valid_models = []
    for model_name in sorted(model_names):
      if not any([re.search(regex, model_name) for regex in task_regexes]):
        print(f'Skipping tests for registered model {model_name}')
        continue
      if any([re.search(regex, model_name) for regex in exclude_regexes]):
        print(f'Explicitly excluding tests for registered model {model_name}')
        continue
      valid_models.append(model_name)

      def _test(self, name=model_name):
        self._test_one_model_params(registry, name)  # pylint: disable=protected-access

      setattr(cls, 'test_model_params_%s' % model_name.replace('.', '_'), _test)
    print(f'Created {len(valid_models)} tests: {valid_models}')
