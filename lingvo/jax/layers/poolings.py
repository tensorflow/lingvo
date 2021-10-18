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
"""Pooling layers."""

from typing import Optional, Tuple

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import numpy as np

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


class PoolingLayer(base_layer.BaseLayer):
  """Pooling layer, which by default performs max pooling."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'window_shape', [0, 0],
        'Window shape which determines the window sizes over which the pooling '
        'is computed. It is given as a Sequence of ints of size 2. Elements are '
        'in the order of height and width, and assumes inputs are in NHWC.')
    p.Define(
        'window_stride', [0, 0],
        'Window stride to use. Must be a pair of ints. The first int '
        'specifies the stride on the height dimension. The second int '
        'specifies the stride on the width dimension.')
    p.Define('pooling_type', 'MAX', 'Pooling type: MAX|AVG')
    p.Define('padding', 'SAME', 'SAME|VALID')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    if len(p.window_shape) != 2:
      raise ValueError('window_shape must be a sequence of length 2.')
    if len(p.window_stride) != 2:
      raise ValueError('window_stride must be a sequence of length 2.')
    if not all([w_shape > 0 for w_shape in p.window_shape]):
      raise ValueError('window_shape entries must be positive integers.')
    if not all([w_stride > 0 for w_stride in p.window_stride]):
      raise ValueError('window_stride entries must be positive integers.')
    if p.pooling_type not in ['MAX', 'AVG']:
      raise ValueError('pooling_type must be one of AVG or MAX.')
    if p.padding not in ['SAME', 'VALID']:
      raise ValueError('padding must be one of SAME or VALID.')

  def FProp(
      self,
      theta: NestedMap,
      inputs: JTensor,
      paddings: Optional[JTensor] = None,
  ) -> Tuple[JTensor, Optional[JTensor]]:
    """Applies pooling to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.
      paddings: The paddings tensor. It is expected to be of shape [B, H].
        Defaults to None, which means there are no paddings.

    Returns:
      An (output, paddings) tensor tuple.
    Raises:
      ValueError: If the input dtype is not one of integer or floating point.
    """
    p = self.params
    if np.issubdtype(inputs.dtype, np.inexact):
      dtype_min = -np.inf
    elif np.issubdtype(inputs.dtype, np.integer):
      dtype_min = np.iinfo(inputs.dtype).min
    else:
      raise ValueError('Unsupported dtype for inputs.')
    if p.pooling_type == 'MAX':
      init_value = dtype_min
      computation = jax.lax.max
      # If paddings are provided and pooling type is 'MAX', replace the pads
      # with negative infinity.
      if paddings is not None:
        # Fill negative infinity in padded positions.
        min_value = jnp.ones_like(inputs) * dtype_min
        compatible_paddings = paddings[..., jnp.newaxis, jnp.newaxis]
        inputs = jnp.where(compatible_paddings > 0, min_value, inputs)
    else:
      assert p.pooling_type == 'AVG'
      init_value = 0
      computation = jax.lax.add
    # The vars `window_shape` and `window_stride` are given only for [H, W].
    # Make it compatible with inputs of shape [N, H, W, C].
    window_shape = [1, p.window_shape[0], p.window_shape[1], 1]
    window_stride = [1, p.window_stride[0], p.window_stride[1], 1]
    out = jax.lax.reduce_window(
        inputs,
        init_value=init_value,
        computation=computation,
        window_dimensions=window_shape,
        window_strides=window_stride,
        padding=p.padding)
    # If average pooling, rescale outputs by the window size.
    if p.pooling_type == 'AVG':
      ones = jnp.ones((inputs.shape[1], inputs.shape[2]), dtype=inputs.dtype)
      window_sizes = jax.lax.reduce_window(
          ones,
          init_value=0,
          computation=jax.lax.add,
          window_dimensions=p.window_shape,
          window_strides=p.window_stride,
          padding=p.padding)
      out *= jnp.reciprocal(window_sizes[jnp.newaxis, ..., jnp.newaxis])
    if paddings is not None:
      if p.pooling_type == 'AVG':
        # Shape of paddings is [N, H]. Renormalize by count of non-padding items
        # in a window.
        non_padding_items = 1 - paddings
        non_padding_count = jax.lax.reduce_window(
            non_padding_items,
            init_value=0,
            computation=jax.lax.add,
            window_dimensions=(1, p.window_shape[0]),
            window_strides=(1, p.window_stride[0]),
            padding=p.padding)
        non_pad_window_sizes = jax.lax.reduce_window(
            jnp.ones((inputs.shape[1]), dtype=inputs.dtype),
            init_value=0,
            computation=jax.lax.add,
            window_dimensions=(p.window_shape[0],),
            window_strides=(p.window_stride[0],),
            padding=p.padding)
        # Do a safe division, where if denominator is 0, return 0.
        # This is because some `non_padding_window_sizes` may be 0, if an
        # entire window is full of PADs.
        non_padding_count = non_padding_count[..., jnp.newaxis, jnp.newaxis]
        out *= jnp.where(non_padding_count, jnp.reciprocal(non_padding_count),
                         0)
        out *= non_pad_window_sizes[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
      # Compute the output paddings.
      if p.window_stride[0] > 1 or p.padding == 'VALID':
        # Output paddings are simply max-pooled since they are 0/1.
        paddings = jax.lax.reduce_window(
            paddings,
            init_value=dtype_min,
            computation=jax.lax.max,
            window_dimensions=(1, p.window_shape[0]),
            window_strides=(1, p.window_stride[0]),
            padding=p.padding)
      # Apply the paddings back to the output.
      # Note that here we shouldn't multiply the output by (1 - paddings)
      # Since the output may contain - np.inf, and -np.inf * 0 = nan.
      non_padding_mask = (1 - paddings[..., jnp.newaxis, jnp.newaxis])
      out = jnp.where(non_padding_mask > 0, out, 0)
    return out, paddings


class GlobalPoolingLayer(base_layer.BaseLayer):
  """Performs a simple global pooling over the input.

  Raises:
    ValueError if `pooling_dims` is not a list or if any of their entries is
    negative.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('pooling_type', 'AVG', 'Pooling type, can be MAX|AVG')
    p.Define('pooling_dims', None, 'A list of dims to perform pooling over.')
    p.Define('keepdims', False,
             'If True, keep dimension of inputs after pooling.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    if p.pooling_type not in ['MAX', 'AVG']:
      raise ValueError('pooling_type must be one of AVG or MAX.')
    if p.pooling_dims is None:
      raise ValueError('pooling_dims must be set as a list.')
    else:
      if not all([p_dims >= 0 for p_dims in p.pooling_dims]):
        raise ValueError('pooling_dims must be non-negative integers.')

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Applies global spatial pooling to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: An input tensor.

    Returns:
      Output tensor with global pooling applied.
    """
    p = self.params
    if p.pooling_type == 'MAX':
      outputs = jnp.max(inputs, p.pooling_dims, keepdims=p.keepdims)
    elif p.pooling_type == 'AVG':
      outputs = jnp.mean(inputs, p.pooling_dims, keepdims=p.keepdims)
    return outputs
