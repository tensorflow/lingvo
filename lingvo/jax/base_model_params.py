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
"""BaseModelParams class definition."""

import abc
from typing import List, Type, TypeVar

from lingvo.jax import py_utils

Params = py_utils.Params
InstantiableParams = py_utils.InstantiableParams
_BaseModelParamsT = TypeVar('_BaseModelParamsT', bound='BaseModelParams')
BaseModelParamsT = Type[_BaseModelParamsT]


class DatasetParams(Params):
  """Encapsulates the parameters for a dataset split."""

  def __init__(self) -> None:
    """Constructor."""
    super().__init__()
    self.Define('name', None, 'Name of this dataset.')
    self.Define('is_training', False,
                'Whether or not this dataset is used for model traning.')
    self.Define('input_gen_params', None,
                'Params for instantiating an input generator.')


class BaseModelParams(metaclass=abc.ABCMeta):
  """Encapsulates the parameters for a model."""

  @abc.abstractmethod
  def Datasets(self) -> List[DatasetParams]:
    """Returns the list of dataset parameters."""

  @abc.abstractmethod
  def Task(self) -> InstantiableParams:
    """Returns the task parameters."""
