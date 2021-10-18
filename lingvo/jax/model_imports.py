# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Global import for all supported Lingvo Jax model hyper-parameters."""

from typing import Any

from lingvo import model_imports as lingvo_model_imports

_TASK_ROOT = 'lingvo.jax.tasks'
_TASK_DIRS = ('lm', 'test')


def ImportAllParams(require_success: bool = True) -> None:
  """Imports all ModelParams to add them to the global registry.

  Because the BUILD rule may selectively depend on a subset of task params,
  we use a try-except to guard around every import.

  Args:
    require_success: Whether we require the underlying lingvo ImportAllParams()
      function to succeed or not.
  """
  success = lingvo_model_imports.ImportAllParams(
      _TASK_ROOT, _TASK_DIRS, require_success=False)
  if require_success and not success:
    raise ValueError('Could not import any task params. Make sure task params '
                     'are linked into the binary.')


def ImportParams(model_name: str, require_success: bool = True) -> Any:
  """Attempts to only import files that may contain the model."""
  return lingvo_model_imports.ImportParams(
      model_name, _TASK_ROOT, require_success=require_success)
