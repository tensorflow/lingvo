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
"""A registry of models.

Models are typically named as '<task>.<module>.<model>'.
"""

from typing import Optional

from absl import logging
from lingvo.jax import base_model_params

BaseModelParamsT = base_model_params.BaseModelParamsT

_MODEL_PREFIX = 'lingvo'


def _ModelClassKey(model_class: BaseModelParamsT) -> str:
  """Retrieves a model key from the model class."""
  path = model_class.__module__ + '.' + model_class.__name__
  # Removes model_registry from `...lingvo.jax.model_registry.`.
  prefix = _ModelClassKey.__module__.replace('.model_registry', '.')
  return path.replace(prefix, '').replace('tasks.', '').replace('params.', '')


class _ModelRegistryHelper:
  """Helper class encapsulating a global registry keyed by model name."""

  # Global variable for the model registry
  _registry = {}

  @classmethod
  def _ClassPathPrefix(cls):
    """Prefixes for model names registered by this module."""
    return _MODEL_PREFIX

  @classmethod
  def RegisterModel(cls, model_class: BaseModelParamsT) -> BaseModelParamsT:
    """Registers a model class in the global registry."""
    key = cls._ClassPathPrefix() + '.' + _ModelClassKey(model_class)
    if key in cls._registry:
      raise ValueError(f'Model `{key}` already registed.')
    logging.info('Registering model %s as %s', model_class, key)
    cls._registry[key] = model_class
    return model_class

  @classmethod
  def GetModel(cls, key: str) -> Optional[BaseModelParamsT]:
    """Retrieves a model from the global registry from the input key."""
    key = cls._ClassPathPrefix() + '.' + key
    if key not in cls._registry:
      for k in cls._registry:
        logging.info('Known model: %s', k)
    return cls._registry.get(key)


RegisterModel = _ModelRegistryHelper.RegisterModel
GetModel = _ModelRegistryHelper.GetModel
