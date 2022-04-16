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

import ast
import inspect
import os
from typing import Any, List
from absl import logging

# List of member functions that are not dataset functions.
NON_DATASET_MEMBERS = [
    'GetAllDatasetParams', 'GetDatasetParams', 'Model', 'Search', 'Task',
    'ProgramSchedule', 'UpdateParamsFromSpec', 'CreateDynamicDatasetMethods'
]


class DatasetFunctionError(TypeError):
  pass


class GetAllDatasetParamsNotImplementedError(NotImplementedError):
  pass


def GetDatasets(cls: Any, warn_on_error: bool = True) -> List[str]:
  """Returns the list of dataset functions (e.g., Train, Dev, ...).

  All public functions apart from `NON_DATASET_MEMBERS` are treated as datasets.
  Dataset functions should not have any required positional arguments.

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
    except GetAllDatasetParamsNotImplementedError:
      pass

  datasets = []
  for name, _ in inspect.getmembers(cls, inspect.isroutine):
    if name not in NON_DATASET_MEMBERS and not name.startswith('_'):
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
          logging.warning(message)
        else:
          raise DatasetFunctionError(message)
      else:
        datasets += [name]
  return datasets


def GetDatasetsAst(base_dir: str, model: str) -> List[str]:
  """Gets datasets but without importing any code by using ast.

  Useful when running from python interpreter without bazel build.

  Args:
    base_dir: Base directory to search in.
    model: The model string.

  Returns:
    A list of strings containing names of valid dataset functions for cls.
    May not be accurate.

  Raises:
    Exception: if anything goes wrong.
  """
  parts = model.split('.')
  model_name = parts[-1]
  parts = parts[:-1]
  module = os.path.join(base_dir, '/'.join(parts)) + '.py'
  for i in range(1, len(parts)):
    # Insert params somewhere in the middle of parts.
    test = os.path.join(base_dir, '/'.join(parts[:i]), 'params', '/'.join(
        parts[i:])) + '.py'
    if os.path.exists(test):
      module = test
      break

  with open(os.path.join(module), 'r') as f:
    tree = ast.parse(f.read())

  class DatasetsVisitor(ast.NodeVisitor):
    """NodeVisitor for collecting datasets for a model."""

    def __init__(self):
      self.datasets = set()
      self._imports = {}

    def visit_Import(self, node):  # pylint: disable=invalid-name
      """Visit a 'import symbol [as alias]' definition."""
      for alias in node.names:
        self._imports[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node):  # pylint: disable=invalid-name
      """Visit a 'from module import symbol [as alias]' definition."""
      for alias in node.names:
        self._imports[alias.asname or alias.name] = (
            node.module + '.' + alias.name)

    def visit_ClassDef(self, node):  # pylint: disable=invalid-name
      """Visit a class definition."""
      if node.name == model_name:
        for base in node.bases:
          if isinstance(base, ast.Name):
            # A superclass in the same file.
            self.datasets |= set(
                GetDatasetsAst(base_dir, '.'.join(parts + [base.id])))
          elif isinstance(base, ast.Attribute):
            # A superclass in a different file.
            if base.value.id == 'base_model_params':
              continue
            self.datasets |= set(
                GetDatasetsAst(
                    base_dir,
                    '.'.join([self._imports[base.value.id], base.attr])))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):  # pylint: disable=invalid-name
      """Visit a function definition."""
      if node.name == 'GetAllDatasetParams':
        # It may be possible to parse the ast for GetAllDatasetParams to find
        # the dictionary keys, but this gets significantly harder when super()
        # calls need to be taken into consideration.
        raise NotImplementedError(
            'GetDatasetsAst does not support models using GetAllDatasetParams.')
      elif (node.name not in NON_DATASET_MEMBERS and
            not node.name.startswith('_')):
        self.datasets.add(node.name)

  visitor = DatasetsVisitor()
  visitor.visit(tree)
  return list(visitor.datasets)
