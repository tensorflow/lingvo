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
"""Global import for model hyper-parameters.

Using this module any ModelParams can be accessed via GetParams.
"""

import importlib
import re
import sys


def _Import(name):
  """Imports the python module of the given name."""
  print('model_imports.py: Importing %s' % name, file=sys.stderr)
  try:
    importlib.import_module(name)
    return True
  except ModuleNotFoundError as e:
    missing_module = re.match("No module named '(.*?)'", e.msg).group(1)
    if not name.startswith(missing_module):
      raise
  return False


def _InsertParams(module):
  """Try inserting 'params' everywhere in the module."""
  left = []
  right = module.split('.')
  while right:
    left.append(right.pop(0))
    yield '.'.join(left + ['params'] + right)


_TASK_ROOT = 'lingvo.tasks'


# LINT.IfChange(task_dirs)
_TASK_DIRS = (
    'asr',
    # 'car',  # TODO(b/179168646): Reenable car models.
    'image',
    'lm',
    'milan',
    'mt',
    'punctuator',
)
# LINT.ThenChange(tasks/BUILD:task_dirs)


def ImportAllParams(task_root=_TASK_ROOT,
                    task_dirs=_TASK_DIRS,
                    require_success=False):
  """Import all ModelParams to add to the global registry."""
  success = False
  for task in task_dirs:
    # By our code repository convention, there is a params.py under the task's
    # params directory. params.py imports _all_ modules that may registers a
    # model param.
    success = _Import('{}.{}.params.params'.format(task_root, task)) or success
  if require_success and not success:
    raise LookupError('Could not import any task params. Make sure task params '
                      'are linked into the binary.')
  return success


def ImportParams(model_name,
                 task_root=_TASK_ROOT,
                 require_success=True):
  """Attempts to only import the files that may contain the model."""
  # 'model_name' follows <task>.<path>.<class name>
  if '.' not in model_name:
    raise ValueError('Invalid model name %s' % model_name)
  if model_name.startswith('test.'):
    # Test models don't need external imports.
    return True
  model_module = model_name.rpartition('.')[0]
  # Try importing the module directly, in case it's a local import.
  success = _Import(model_module)
  # Try all locations of inserting params.
  for module_with_params in _InsertParams(model_module):
    success = _Import(module_with_params) or success

  # Try built-in tasks imports.
  task_model_module = f'{task_root}.{model_module}'
  success = _Import(task_model_module) or success
  # Try all locations of inserting params.
  for module_with_params in _InsertParams(task_model_module):
    success = _Import(module_with_params) or success

  if require_success and not success:
    raise LookupError(
        f'Could not find any valid import paths for module {model_module}. '
        f'Make sure the relevant params files are linked into the binary.')
  return success
