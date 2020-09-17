# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for dataset information."""
import inspect

import lingvo.compat as tf


class DatasetFunctionError(TypeError):
  pass


def GetDatasets(cls, warn_on_error=False):
  """Returns the list of dataset functions (e.g. Train, Dev, ...)."""
  datasets = []
  for name, _ in inspect.getmembers(
      cls, lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
    if name not in ['GetDatasetParams', 'Model', 'Task', 'ProgramSchedule'
                   ] and not name.startswith('_'):
      # Datasets are assumed to have no required positional arguments.
      # Check this assumption for all arguments beyond the first (which will be
      # 'self' or 'cls').
      args = list(inspect.signature(getattr(cls, name)).parameters.values())[1:]
      positional_arguments = [p.name for p in args if p.default == p.empty]
      if positional_arguments:
        message = (f'Found a public function {name} in {cls.__name__} with '
                   f'required positional arguments: {positional_arguments}.')
        if warn_on_error:
          tf.logging.warning(message)
        else:
          raise DatasetFunctionError(message)
      else:
        datasets += [name]
  return datasets
