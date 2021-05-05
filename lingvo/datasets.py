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


def GetDatasets(cls, warn_on_error=True):
  """Returns the list of dataset functions (e.g., Train, Dev, ...).

  All public functions apart from 'GetDatasetParams', 'Model', 'Task',
  'ProgramSchedule' are treated as datasets.  Dataset functions should not
  have any required positional arguments.

  Args:
    cls: A class variable or instance variable.  This function expects to be
      called on classes that can be used as model tasks e.g. via
      model_registry.RegisterSingleTaskModel.
    warn_on_error: When a class contains public methods that cannot be used as a
      dataset, if True, logs a warning, if False, raises a DatasetFunctionError.

  Returns:
    A list of strings containing names of valid dataset functions for cls.

  Raises:
    DatasetFunctionError: if the cls contains public methods that cannot be used
      as datasets, and warn_on_error is False.
  """

  mdl_params = None
  if inspect.isclass(cls):
    try:
      mdl_params = cls()
    except TypeError:  # Capture cls construction error
      pass
  else:
    mdl_params = cls

  if mdl_params:
    try:
      all_datasets = mdl_params.GetAllDatasetParams()
      return sorted(list(all_datasets.keys()))
    except NotImplementedError:
      pass

  datasets = []
  for name, _ in inspect.getmembers(cls, inspect.isroutine):
    if name not in [
        'GetAllDatasetParams', 'GetDatasetParams', 'Model', 'Task',
        'ProgramSchedule'
    ] and not name.startswith('_'):
      # Datasets are assumed to have no required positional arguments.
      fn = getattr(cls, name)
      args = list(inspect.signature(fn).parameters.values())
      if inspect.isclass(cls) and not inspect.ismethod(fn):
        # Methods obtained from inspecting a class includes a 'self' first
        # argument that should be ignored. That is because they are not bound.
        # Methods obtained from inspecting an instance, or classmethods obtained
        # from inspecting a class are bound and inspect.ismethod() returns True.
        args = args[1:]
      positional_arguments = [p.name for p in args if p.default == p.empty]
      if positional_arguments:
        if inspect.isclass(cls):
          class_name = cls.__name__
        else:
          class_name = cls.__class__.__name__
        message = (f'Found a public function {name} in {class_name} with '
                   f'required positional arguments: {positional_arguments}.')
        if warn_on_error:
          tf.logging.warning(message)
        else:
          raise DatasetFunctionError(message)
      else:
        datasets += [name]
  return datasets
