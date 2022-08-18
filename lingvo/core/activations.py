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

import numpy as np

# Supported activation functions.
_ACTIVATIONS = {
    'RELU':
        tf.nn.relu,
    'RELU6':
        tf.nn.relu6,
    'LEAKY_RELU':
        tf.nn.leaky_relu,
    'SIGMOID':
        tf.sigmoid,
    'TANH':
        tf.tanh,
    'GELU':
        tf.nn.gelu,
    'GELU_APPROXIMATE':
        lambda x: tf.nn.gelu(x, approximate=True),
    'GELU_RAW':
        lambda x: 0.5 * x * (  # pylint: disable=g-long-lambda
            1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))),
    'SWISH':
        tf.nn.swish,
    'SOFTPLUS':
        tf.nn.softplus,
    # Squared ReLU from the Primer paper: https://arxiv.org/abs/2109.08668
    'SQUARED_RELU':
        lambda x: tf.math.square(tf.nn.relu(x)),
    'SILU':
        tf.nn.silu,
    # GLU Variants: https://arxiv.org/abs/2002.05202
    'GLU':
        lambda x: GLUVariants(x, 'SIGMOID'),
    'BILINEAR_GLU':
        lambda x: GLUVariants(x, 'NONE'),
    'RELU_GLU':
        lambda x: GLUVariants(x, 'RELU'),
    'GELU_GLU':
        lambda x: GLUVariants(x, 'GELU'),
    'SWISH_GLU':
        lambda x: GLUVariants(x, 'SWISH'),
    'NONE':
        tf.identity,
}

_ACTIVATIONS_FLOPS = {
    'NONE': 0,
    'RELU': 1,
    'RELU6': 1,
    # ReLU(x) - 0.2 * ReLU(-x)
    # neg, relu, mul, sub, relu
    'LEAKY_RELU': 5,
    # 1 / (1 + exp(-x))
    'SIGMOID': 4,  # neg, exp, add, div
    # (exp(2*x) - 1) / (exp(2*x) + 1)
    'TANH': 7,  # mul, exp, sub, mul, exp, add, div
    # Gelu is tough, let's assume it is
    # .5 * x * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    'GELU': 15,  # mul, mul, add, tanh, mul, mul, add, mul, mul
    'GELU_RAW': 15,  # same as GELU
    # Or approximated as x * sigmoid(1.702 * x).
    'GELU_APPROXIMATE': 6,  # mul, sigmoid, mul
    # x * sigmoid(x)
    'SWISH': 5,  # sigmoid, mul
    # ln(1+exp(x))
    'SOFTPLUS': 3,  # exp, add, ln
    'SQUARED_RELU': 2,  # relu, mul
    'GLU': 5,  # SIGMOID, mul
    'BILINEAR_GLU': 1,  # NONE, mul
    'RELU_GLU': 2,  # RELU, mul
    'GELU_GLU': 16,  # GELU, mul
    'SWISH_GLU': 6,  # SWISH, mul
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


def DimMultiplier(activation_name):
  """Returns dimension multiplier for the activation."""
  assert IsSupported(activation_name)
  if activation_name.endswith('GLU'):
    return 2
  return 1


def GLUVariants(x, activation_name):
  """Returns function corresponding to GLU variants."""
  x1, x2 = tf.split(x, 2, axis=-1)
  return x1 * _ACTIVATIONS[activation_name](x2)


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
