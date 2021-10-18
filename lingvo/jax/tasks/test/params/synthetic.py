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
"""Test model configuration using synthetic data."""

from lingvo.jax import base_model_params
from lingvo.jax import layers
from lingvo.jax import model_registry


@model_registry.RegisterModel
class SyntheticClassifier(base_model_params.BaseModelParams):
  # TODO(shafey): Implement a real test model.

  def Datasets(self):
    return []

  def Task(self):
    act_p = layers.ActivationLayer.Params()
    return act_p
