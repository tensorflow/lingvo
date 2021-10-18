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
"""Stochastic layers."""

import numbers
from typing import List

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


class DropoutLayer(base_layer.BaseLayer):
  """Apply dropout during training."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('keep_prob', 1.0, 'Keep probability.')
    # noise_shape is unknown when building layer params.
    p.Define(
        'noise_shape', None, 'A 1-D list of type `int32`, representing '
        'the shape for randomly generated keep/drop flags.')
    p.Define(
        'noise_shape_broadcast_dims', None,
        'A list of dimension where the noise shape is broadcasted. For '
        'example, noise_shape = [n, h, w, 1] when '
        'noise_shape_broadcast_dims=[-1] ')
    # We typically want to replace dropout by expectation during eval.
    # However, in certain cases E(f(x)) != f(E(x)), and replacing dropout by its
    # expectation during eval leads to worse quality.
    p.Define('dropout_at_eval', False,
             'Whether or not to also perform dropout at eval time.')
    return p

  def _Dropout(self, theta: NestedMap, inputs: JTensor,
               noise_shape: List[int]) -> JTensor:
    p = self.params
    if noise_shape is None:
      noise_shape = inputs.shape
    prng_seed = base_layer.NextPrngKey()
    keep_prob = p.keep_prob
    assert keep_prob > 0.0
    random_nums = keep_prob + jax.random.uniform(
        prng_seed, noise_shape, inputs.dtype, minval=0.0, maxval=1.0)
    binary_mask = jnp.floor(random_nums)
    return inputs * binary_mask / keep_prob

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Applies dropout to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs JTensor.

    Returns:
      inputs with dropout applied at training time.
    """
    p = self.params
    if not self.do_eval or p.dropout_at_eval:
      if isinstance(p.keep_prob, numbers.Real) and p.keep_prob == 1.0:
        return inputs
      if p.noise_shape_broadcast_dims:
        noise_shape = p.noise_shape or inputs.shape
        for dim in p.noise_shape_broadcast_dims:
          if dim >= len(noise_shape):
            raise ValueError('Invalid broadcasted dim {}'.format(dim))
          noise_shape[dim] = 1
      else:
        noise_shape = p.noise_shape
      ret = self._Dropout(theta, inputs, noise_shape)
      return ret
    else:
      return inputs


class StochasticResidualLayer(base_layer.BaseLayer):
  """Stochastic residual layer that randomly drops the residual branch."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for `StochasticResidualLayer`."""
    p = super().Params()
    p.Define(
        'residual_weight', 1.0, 'Residual weight with which to add the '
        'reisdual back to the input.')
    p.Define('survival_prob', 1.0,
             'Survival probability of the residual branch while dropping out.')
    return p

  def _DropConnect(self, inputs: JTensor) -> JTensor:
    """Drops the entire residual layer with given survival probability.

    Args:
      inputs: input `.JTensor` which is on the residual branch which is dropped.

    Returns:
      Dropped out inputs.
    """
    if self.do_eval:
      return inputs

    # Compute tensor.
    prng_key = base_layer.NextPrngKey()
    batch_size = inputs.shape[0]
    shape = [batch_size] + [1] * (len(inputs.shape) - 1)
    random_tensor = self.params.survival_prob + jax.random.uniform(
        prng_key, shape, dtype=inputs.dtype)
    binary_tensor = jnp.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no additional compute is
    # needed at test time.
    output = inputs / self.params.survival_prob * binary_tensor
    return output

  def FProp(self, theta: NestedMap, inputs: JTensor,
            residual: JTensor) -> JTensor:
    """Returns inputs + residual with stochastic dropout.

    Args:
      theta: A `.NestedMap` of weights defined in this layer.
      inputs: input `.JTensor`.
      residual: residual `.JTensor` which is added to input with dropout.

    Returns:
      Output `.JTensor` which is residual added to inputs with dropout.
    """
    return inputs + self.params.residual_weight * self._DropConnect(residual)
