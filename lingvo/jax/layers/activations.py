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
"""Activation layers."""

import jax
import jax.numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


class ActivationLayer(base_layer.BaseLayer):
  """Activation layer that wraps popular activation functions."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'activation', 'RELU', 'Activation function to use. '
        'Options are RELU, RELU6, RELU^2, RELU^3, SIGMOID, TANH,'
        'GELU, SILU, NONE.')
    return p

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    del theta  # not used
    p = self.params
    if p.activation == 'RELU':
      outputs = jax.nn.relu(inputs)
    elif p.activation == 'RELU6':
      outputs = jax.nn.relu6(inputs)
    elif p.activation == 'RELU^2':
      outputs = jax.nn.relu(inputs)
      outputs = jnp.square(outputs)
    elif p.activation == 'RELU^3':
      outputs = jax.nn.relu(inputs)
      outputs *= jnp.square(outputs)
    elif p.activation == 'SIGMOID':
      outputs = jax.nn.sigmoid(inputs)
    elif p.activation == 'TANH':
      outputs = jax.nn.tanh(inputs)
    elif p.activation == 'GELU':
      outputs = jax.nn.gelu(inputs)
    elif p.activation == 'SILU':
      outputs = jax.nn.silu(inputs)
    else:  # 'NONE'
      outputs = inputs
    return outputs
