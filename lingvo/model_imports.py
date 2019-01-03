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
import six
import tensorflow as tf

from lingvo import model_registry

_TASK_ROOT = 'lingvo.tasks'

# LINT.IfChange(task_dirs)
_TASK_DIRS = [
    'asr',
    'image',
    'lm',
    'mt',
    'punctuator',
]
# LINT.ThenChange(tasks/BUILD:task_dirs)

# Import all ModelParams to ensure that they are added to the global registry.
for task_name in _TASK_DIRS:
  name = '%s.%s.params' % (_TASK_ROOT, task_name)
  tf.logging.info('Importing %s', name)
  try:
    importlib.import_module(name)
  except ImportError as e:
    errmsg = str(e)
    if six.PY2:
      match_str = 'No module named.*params'
    else:
      match_str = 'No module named.*%s' % _TASK_ROOT
    if re.match(match_str, errmsg):
      # Expected that some imports may be missing.
      tf.logging.info('Expected error importing %s: %s', task_name, errmsg)
    else:
      tf.logging.info('Unexpected error importing %s: %s', task_name, errmsg)
      raise
