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
import re
import lingvo.compat as tf
import six


def _Import(task_root, name):
  """Imports the python module of the given name."""
  tf.logging.info('Importing %s', name)
  try:
    importlib.import_module(name)
    tf.logging.info('Imported %s', name)
  except ImportError as e:
    errmsg = str(e)
    if six.PY2:
      match_str = 'No module named.*'
    else:
      match_str = 'No module named.*%s' % task_root
    if re.match(match_str, errmsg):
      # Expected that some imports may be missing.
      tf.logging.info('Expected error importing %s: %s', name, errmsg)
    else:
      tf.logging.info('Unexpected error importing %s: %s', name, errmsg)
      raise


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


def ImportAllParams(task_root=_TASK_ROOT, task_dirs=_TASK_DIRS):
  # Import all ModelParams to ensure that they are added to the global registry.
  for task in task_dirs:
    # By our code repository convention, there is a params.py under the task's
    # params directory. params.py imports _all_ modules that may registers a
    # model param.
    _Import(task_root, '{}.{}.params.params'.format(task_root, task))


def ImportParams(model_name, task_root=_TASK_ROOT, task_dirs=_TASK_DIRS):
  # Import precisely the params/.*py file that may defines the model.
  for task in task_dirs:
    if model_name.startswith(task + '.'):
      # 'model_name' follows <task>.<path>.<class name>
      path = model_name[:model_name.rfind('.')][len(task) + 1:]
      _Import(task_root, '{}.{}.params.{}'.format(task_root, task, path))
