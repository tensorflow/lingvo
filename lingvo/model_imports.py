# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import sys


def _Import(name):
  """Imports the python module of the given name."""
  print('model_imports.py: Importing %s' % name, file=sys.stderr)
  try:
    importlib.import_module(name)
    print('model_imports.py: Imported %s' % name, file=sys.stderr)
    return True
  except ImportError as e:
    # It is expected that some imports may be missing.
    print(
        'model_imports.py: Could not import %s: %s' % (name, e),
        file=sys.stderr)
  return False


_TASK_ROOT = 'lingvo.tasks'


# LINT.IfChange(task_dirs)
_TASK_DIRS = (
    'asr',
    'car',
    'image',
    'lm',
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
    success = success or _Import('{}.{}.params.params'.format(task_root, task))
  if require_success and not success:
    raise LookupError('Could not import any task params. Make sure task params '
                      'are linked into the binary.')
  return success


def ImportParams(model_name,
                 task_root=_TASK_ROOT,
                 task_dirs=_TASK_DIRS,
                 require_success=True):
  """Attempts to only import the files that may contain the model."""
  # 'model_name' follows <task>.<path>.<class name>
  if '.' not in model_name:
    raise ValueError('Invalid model name %s' % model_name)
  model_module = model_name.rpartition('.')[0]
  # Try importing the module directly, in case it's a local import.
  success = _Import(model_module)

  # Try built-in tasks imports.
  for task in sorted(task_dirs):
    if model_module.startswith(task + '.'):
      path = model_module[len(task) + 1:]
      success = success or _Import('{}.{}.params.{}'.format(
          task_root, task, path))

  if require_success and not success:
    raise LookupError(
        'Could not find any valid import paths for module %s. Check the logs '
        'above to see if there were errors importing the module, and make sure '
        'the relevant params files are linked into the binary.' % model_module)
  return success
