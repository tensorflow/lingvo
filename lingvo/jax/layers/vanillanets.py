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
"""Vanilla (Skip-free, Batch-norm free) layers."""

import math

from jax import nn

from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import convolutions

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


def tailored_lrelu(negative_slope, x):
  return math.sqrt(2.0 / (1 + negative_slope**2)) * nn.leaky_relu(
      x, negative_slope=negative_slope)


class VanillaBlock(base_layer.BaseLayer):
  """Vanilla Block."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for the VanillaBlock."""
    p = super().Params()
    p.Define('input_dim', 0, 'Input dimension.')
    p.Define('output_dim', 0, 'Output dimension.')
    # We enable bias (which is disabled by default in Conv2D) as we remove
    # batch normalization from the network.
    p.Define(
        'conv_params',
        convolutions.Conv2D.Params().Set(
            bias=True,
            params_init=py_utils.WeightInit.ScaledDeltaOrthogonal(1.0)),
        'Which Conv block to use.')
    p.Define('kernel_size', 3, 'Kernel sizes of the block.')
    p.Define('stride', 1, 'Stride')
    p.Define('negative_slope', 0.4, 'Negative slope for leaky relu.')

    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    self._in_out_same_shape = (p.input_dim == p.output_dim and p.stride == 1)

    body = []
    # conv_in, reduce the hidden dims by 4
    body.append(p.conv_params.Copy().Set(
        name='conv_in',
        filter_shape=(1, 1, p.input_dim, p.output_dim // 4),
        filter_stride=(1, 1)))

    # conv_mid using the kernel size and stride provided
    body.append(p.conv_params.Copy().Set(
        name='conv_mid',
        filter_shape=(p.kernel_size, p.kernel_size, p.output_dim // 4,
                      p.output_dim // 4),
        filter_stride=(p.stride, p.stride)))

    # conv_out, expand back to hidden dim
    body.append(p.conv_params.Copy().Set(
        name='conv_out',
        filter_shape=(1, 1, p.output_dim // 4, p.output_dim),
        filter_stride=(1, 1)))
    self.create_children('body', body)

  def fprop(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Forward propagation of a VanillaBlock.

    Args:
      theta: A `.NestedMap` object containing variable values of this layer.
      inputs: A `.JTensor` as inputs of [B, H, W, D_in] also commonly known as
        NHWC format.

    Returns:
      A `.JTensor` as outputs of shape [B, H', W', D_out].
    """
    p = self.params
    outputs = inputs

    for i in range(len(self.body)):
      outputs = tailored_lrelu(p.negative_slope,
                               self.body[i].fprop(theta.body[i], outputs))
    return outputs
