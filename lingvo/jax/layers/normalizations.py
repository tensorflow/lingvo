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
"""Normalization layers."""

from typing import List, Optional, Tuple

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
weight_params = py_utils.weight_params

InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


def compute_moments(
    inputs: JTensor,
    padding: JTensor,
    reduce_over_dims: List[int],
    enable_cross_replica_sum_on_tpu: bool = False,
    keepdims: bool = False,
) -> Tuple[JTensor, JTensor]:
  """Computes mean and variance over the valid data points in inputs."""
  assert inputs.ndim == padding.ndim
  rank = inputs.ndim
  assert all([0 <= dim < rank for dim in reduce_over_dims])
  mask = 1.0 - padding
  sum_v = jnp.sum(inputs * mask, axis=reduce_over_dims, keepdims=keepdims)
  count_v = jnp.sum(
      jnp.ones_like(inputs) * mask, axis=reduce_over_dims, keepdims=keepdims)

  if enable_cross_replica_sum_on_tpu:
    # TODO(shafey, yonghui): Fetch axis_name from globals.
    sum_v = jax.lax.psum(sum_v, axis_name='batch')
    count_v = jax.lax.psum(count_v, axis_name='batch')

  count_v = jnp.maximum(count_v, 1.0)
  mean = sum_v / count_v
  sum_vv = jnp.sum(
      (inputs - mean) * (inputs - mean) * mask,
      axis=reduce_over_dims,
      keepdims=keepdims)

  if enable_cross_replica_sum_on_tpu:
    # TODO(shafey, yonghui): Fetch axis_name from globals.
    sum_vv = jax.lax.psum(sum_vv, axis_name='batch')

  variance = sum_vv / count_v
  return mean, variance


class BatchNorm(base_layer.BaseLayer):
  """Batch normalization layer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define(
        'decay', 0.999,
        'Decay in updating the mean and variance moving average used in'
        ' batch normalization.')
    p.Define(
        'enable_cross_replica_sum_on_tpu', False,
        'If true, computes global mean and variance across all replicas.'
        'Only effective for tpu.')
    p.Define(
        'use_moving_avg_in_training', False,
        'If True, use global moving avg (mean, variance) during training'
        ' to avoid mismatch between train and eval, which then'
        ' essentially acts as an adaptive normalization step. When this is'
        ' set to True, it also disables the use of beta and gamma variables.')
    p.Define('set_padded_output_to_zero', True,
             'If True, sets the padded outputs to zero.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    self._epsilon = 0.001
    self._decay = p.decay

  def _get_weight_shape(self) -> JTensor:
    return [self.params.dim]

  def create_layer_variables(self) -> None:
    p = self.params

    beta_pc = weight_params(
        shape=self._get_weight_shape(),
        init=WeightInit.Constant(0.0),
        dtype=p.dtype)
    self.create_variable('beta', beta_pc)

    # gamma = theta.gamma + 1.0
    gamma_pc = weight_params(
        shape=self._get_weight_shape(),
        init=WeightInit.Constant(0.0),
        dtype=p.dtype)
    self.create_variable('gamma', gamma_pc)

    mva = weight_params(
        shape=[p.dim],
        init=WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[base_layer.REQUIRES_MEAN_SYNC])
    self.create_variable('moving_mean', mva, trainable=False)

    mvv = weight_params(
        shape=[p.dim],
        init=WeightInit.Constant(1.0),
        dtype=p.dtype,
        collections=[base_layer.REQUIRES_MEAN_SYNC])
    self.create_variable('moving_variance', mvv, trainable=False)

  def _get_default_paddings(self, inputs: JTensor) -> JTensor:
    """Gets the default paddings for an input."""
    in_shape = list(inputs.shape)
    assert len(in_shape) > 1
    in_shape[-1] = 1
    return jnp.zeros(in_shape, dtype=inputs.dtype)

  def _get_beta_gamma(self, theta: NestedMap) -> Tuple[JTensor, JTensor]:
    p = self.params
    if p.use_moving_avg_in_training:
      beta = 0.0
      gamma = 1.0
    else:
      beta = theta.beta
      gamma = theta.gamma + 1.0
    return beta, gamma

  def compute_and_update_moments(
      self, theta: NestedMap, inputs: JTensor,
      paddings: JTensor) -> Tuple[JTensor, JTensor, JTensor, JTensor]:
    """Computes moments and updates state.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs JTensor.  Shaped [..., dim].
      paddings: The paddings JTensor.  Shaped [..., 1], with the same rank as
        the input JTensor.

    Returns:
      Tuple of (mean, variance, beta, gamma).
    """
    p = self.params
    if self.do_eval:
      # The mean and variance used for normalization.
      norm_mean, norm_variance = theta.moving_mean, theta.moving_variance
      base_layer.add_summary('moving_mean', theta.moving_mean)
      base_layer.add_summary('moving_variance', theta.moving_variance)
    else:
      rank = inputs.ndim
      reduce_over_dims = list(range(0, rank - 1))
      mean, variance = compute_moments(
          inputs,
          paddings,
          reduce_over_dims,
          enable_cross_replica_sum_on_tpu=p.enable_cross_replica_sum_on_tpu,
          keepdims=True)

      new_moving_mean = theta.moving_mean * p.decay + mean * (1.0 - p.decay)
      self.forward_update_var('moving_mean', new_moving_mean)
      new_moving_variance = (
          theta.moving_variance * p.decay + variance * (1.0 - p.decay))
      self.forward_update_var('moving_variance', new_moving_variance)

      # Add some summaries for visualization.
      base_layer.add_summary('mean', mean)
      base_layer.add_summary('variance', variance)
      base_layer.add_summary('moving_mean', theta.moving_mean)
      base_layer.add_summary('moving_variance', theta.moving_variance)
      if p.use_moving_avg_in_training:
        # Use the global statistics for normalization.
        norm_mean = theta.moving_mean
        norm_variance = theta.moving_variance
      else:
        # Use the batch statistics for normalization.
        norm_mean = mean
        norm_variance = variance

    beta, gamma = self._get_beta_gamma(theta)
    return norm_mean, norm_variance, beta, gamma

  def fprop(self,
            theta: NestedMap,
            inputs: JTensor,
            paddings: Optional[JTensor] = None) -> JTensor:
    """Apply batch normalization.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs JTensor.  Shaped [..., dim].
      paddings: The paddings JTensor.  Shaped [..., 1].

    Returns:
      Output after applying batch normalization, with the same shape as
        'inputs'.
    """
    p = self.params
    inputs, paddings = self._cast_to_fprop_dtype((inputs, paddings))
    if paddings is None:
      paddings = self._get_default_paddings(inputs)

    assert inputs.ndim == paddings.ndim
    assert paddings.shape[-1] == 1

    norm_mean, norm_variance, beta, gamma = self.compute_and_update_moments(
        theta, inputs, paddings)

    inv = gamma / jnp.sqrt(norm_variance + self._epsilon)
    bn_output = (inputs - norm_mean) * inv + beta

    if p.set_padded_output_to_zero:
      bn_output *= 1.0 - paddings

    return bn_output


class LayerNorm(base_layer.BaseLayer):
  """Layer normalization."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input to the network.')
    p.Define('epsilon', 1e-6, 'Tiny value to guard rsqrt.')
    p.Define('scale', True, 'Whether to use a learned scaling.')
    p.Define('bias', True, 'Whether to use bias.')
    return p

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    wp = p.weight_split_dims_mapping
    wp_scale = wp.wt
    if p.device_mesh is not None and wp.wt is None:
      # Simply replicate the weights.
      wp_scale = [-1]
    if p.scale:
      self.create_variable(
          'scale',
          weight_params(
              shape=[p.input_dims],
              init=WeightInit.Constant(0.0),
              dtype=p.dtype,
              device_mesh=p.device_mesh,
              tensor_split_dims_mapping=wp_scale))
    if p.bias:
      wp_bias = wp_scale  # bias should use the same sharding as scale.
      self.create_variable(
          'bias',
          weight_params(
              shape=[p.input_dims],
              init=WeightInit.Constant(0.0),
              dtype=p.dtype,
              device_mesh=p.device_mesh,
              tensor_split_dims_mapping=wp_bias))

  def fprop(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Apply layer norm to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Layer normalized input.
    """
    p = self.params
    mean = jnp.mean(inputs, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=[-1], keepdims=True)
    normed_inputs = (inputs - mean) * jax.lax.rsqrt(var + self.params.epsilon)
    if p.scale:
      normed_inputs *= (1 + theta.scale)
    if p.bias:
      normed_inputs += theta.bias
    return normed_inputs
