# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Activations layers."""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils


# Supported activation functions.
_ACTIVATIONS = {
    'RELU': tf.nn.relu,
    'RELU6': tf.nn.relu6,
    'SIGMOID': tf.sigmoid,
    'TANH': tf.tanh,
    'GELU': tf.nn.gelu,
    'GELU_APPROXIMATE': lambda x: tf.nn.gelu(x, approximate=True),
    'SWISH': tf.nn.swish,
    'SOFTPLUS': tf.nn.softplus,
    'NONE': tf.identity,
}

_ACTIVATIONS_FLOPS = {
    'NONE': 0,
    'RELU': 1,
    'RELU6': 1,
    # 1 / (1 + exp(-x))
    'SIGMOID': 4,  # neg, exp, add, div
    # (exp(2*x) - 1) / (exp(2*x) + 1)
    'TANH': 7,  # mul, exp, sub, mul, exp, add, div
    # Gelu is tough, let's assume it is
    # .5 * x * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    'GELU': 15,  # mul, mul, add, tanh, mul, mul, add, mul, mul
    # Or approximated as x * sigmoid(1.702 * x).
    'GELU_APPROXIMATE': 6,  # mul, sigmoid, mul
    # x * sigmoid(x)
    'SWISH': 5,  # sigmoid, mul
    # ln(1+exp(x))
    'SOFTPLUS': 3,  # exp, add, ln
}


def GetFn(activation_name):
  """Returns function corresponding to the activation name."""
  return _ACTIVATIONS[activation_name]


def GetFlops(activation_name):
  """Returns FLOPS corresponding to the activation name."""
  return _ACTIVATIONS_FLOPS[activation_name]


def IsSupported(activation_name):
  """Checks if the activation is supported."""
  return activation_name in _ACTIVATIONS


class ActivationLayer(base_layer.BaseLayer):
  """Activation layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('activation', 'RELU', 'Activation function to use.')
    return p

  def FProp(self, theta, inputs, paddings=None):
    """Applies activation fn.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The input tensor.
      paddings: The paddings tensor.

    Returns:
      If paddings is not None, an (output, paddings) tensor, else just the
      output with the same shape and type of inputs.
    """
    p = self.params
    if p.activation == 'NONE':
      ret = inputs
    else:
      with tf.name_scope(p.name):
        ret = GetFn(p.activation)(inputs)
    if paddings is None:
      return ret
    else:
      return ret, paddings

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(
        flops=inputs.num_elements() * GetFlops(p.activation),
        out_shapes=(inputs,))
