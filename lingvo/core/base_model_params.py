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


class _BaseModelParams(object):
  """Base class for storing model Params for a single experiment."""

  @classmethod
  def GetDatasetParams(cls, dataset):
    """Convenience function that returns the param for the given dataset name.

    Args:
      dataset: A python string. Typically, 'Dev', 'Test', etc.

    Returns:
      If there is a `cls.${dataset}` method defined, call that method to
      generate a hyperparam for the input data.

    Raises:
      AttributeError: if there is not a `${dataset}` method defined under `cls`.
    """
    f = getattr(cls, dataset)
    return f()


class SingleTaskModelParams(_BaseModelParams):
  """Model Params for a `.SingleTaskModel`."""

  @classmethod
  def Train(cls):
    """Returns Params for the training dataset."""
    return base_input_generator.BaseSequenceInputGenerator.Params().Set(
        name='Train')

  @classmethod
  def Dev(cls):
    """Returns Params for the development dataset."""
    return base_input_generator.BaseSequenceInputGenerator.Params().Set(
        name='Dev')

  @classmethod
  def Test(cls):
    """Returns Params for the testing dataset."""
    return base_input_generator.BaseSequenceInputGenerator.Params().Set(
        name='Test')

  @classmethod
  def Task(cls):
    """Returns task params."""
    raise NotImplementedError('Abstract method')

  @classmethod
  def Model(cls):
    """Returns model params.

    Emulates structure of `MultiTaskModelParams`.
    """
    return base_model.SingleTaskModel.Params(cls.Task())


class MultiTaskModelParams(_BaseModelParams):
  """Model Params for a `.MultiTaskModel`."""

  @classmethod
  def Train(cls):
    """Returns Params for the training dataset."""
    return hyperparams.Params()

  @classmethod
  def Dev(cls):
    """Returns Params for the development dataset."""
    return hyperparams.Params()

  @classmethod
  def Test(cls):
    """Returns Params for the testing dataset."""
    return hyperparams.Params()

  @classmethod
  def Model(cls):
    """Returns model params."""
    raise NotImplementedError('Abstract method')
