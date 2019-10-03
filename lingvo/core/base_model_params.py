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
"""BaseModelParams class definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import hyperparams


class DatasetError(Exception):
  """Dataset error exception class."""
  pass


class _BaseModelParams(object):
  """Base class for storing model Params for a single experiment."""

  def GetDatasetParams(self, dataset):
    """Convenience function that returns the param for the given dataset name.

    Args:
      dataset: A python string. Typically, 'Dev', 'Test', etc.

    Returns:
      If there is a `cls.${dataset}` method defined, call that method to
      generate a hyperparam for the input data.

    Raises:
      DatasetError: if there is not a `${dataset}` method defined under `cls`.
    """
    try:
      f = getattr(self, dataset)
    except AttributeError as e:
      raise DatasetError(str(e))
    return f()


class SingleTaskModelParams(_BaseModelParams):
  """Model Params for a `.SingleTaskModel`."""

  def Train(self):
    """Returns Params for the training dataset."""
    return base_input_generator.BaseSequenceInputGenerator.Params().Set(
        name='Train')

  def Dev(self):
    """Returns Params for the development dataset."""
    return base_input_generator.BaseSequenceInputGenerator.Params().Set(
        name='Dev')

  def Test(self):
    """Returns Params for the testing dataset."""
    return base_input_generator.BaseSequenceInputGenerator.Params().Set(
        name='Test')

  def Task(self):
    """Returns task params."""
    raise NotImplementedError('Abstract method')

  def Model(self):
    """Returns model params.

    Emulates structure of `MultiTaskModelParams`.
    """
    return base_model.SingleTaskModel.Params(self.Task())

  def ProgramSchedule(self):
    """Returns a schedule for the Executor."""
    raise NotImplementedError('Abstract method')


class MultiTaskModelParams(_BaseModelParams):
  """Model Params for a `.MultiTaskModel`."""

  def Train(self):
    """Returns Params for the training dataset."""
    return hyperparams.Params()

  def Dev(self):
    """Returns Params for the development dataset."""
    return hyperparams.Params()

  def Test(self):
    """Returns Params for the testing dataset."""
    return hyperparams.Params()

  def Model(self):
    """Returns model params."""
    raise NotImplementedError('Abstract method')

  def ProgramSchedule(self):
    """Returns a schedule for the Executor."""
    raise NotImplementedError('Abstract method')
