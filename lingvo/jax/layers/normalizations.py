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
from lingvo.jax import asserts
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
    cumulative_axis: Optional[int] = None,
    enable_cross_replica_sum_on_tpu: bool = False,
    keepdims: bool = False,
) -> Tuple[JTensor, JTensor]:
  """Computes mean and variance over the valid data points in inputs.

  Args:
    inputs: The inputs JTensor.
    padding: The paddings JTensor.
    reduce_over_dims: A sequence of ints for dimensions to reduce `inputs` over.
    cumulative_axis: An optional int for axis to compute a cumulative sum. If
      none, there will be no cumulative sum applied.
    enable_cross_replica_sum_on_tpu: A boolean indicating whether to use an
      all-reduce sum over the 'batch' axis.
    keepdims: A boolean indicating whether summations reduction axes should be
      left in the result as dimensions with size one.

  Returns:
    Tuple of (mean, variance).
  """
  asserts.eq(inputs.ndim, padding.ndim)
  rank = inputs.ndim
  for dim in reduce_over_dims:
    asserts.between(dim, 0, rank, left_strict=False, right_strict=True)
  mask = 1.0 - padding
  sum_v = jnp.sum(inputs * mask, axis=reduce_over_dims, keepdims=keepdims)
  count_v = jnp.sum(
      jnp.ones_like(inputs) * mask, axis=reduce_over_dims, keepdims=keepdims)
  if cumulative_axis is not None:
    sum_v = jnp.cumsum(sum_v, axis=cumulative_axis)
    count_v = jnp.cumsum(count_v, axis=cumulative_axis)

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
  if cumulative_axis is not None:
    sum_vv = jnp.cumsum(sum_vv, axis=cumulative_axis)

  if enable_cross_replica_sum_on_tpu:
    # TODO(shafey, yonghui): Fetch axis_name from globals.
    sum_vv = jax.lax.psum(sum_vv, axis_name='batch')

  variance = sum_vv / count_v
  return mean, variance


class BatchNorm(base_layer.BaseLayer):
  """Batch normalization layer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Returns the layer params with BatchNorm specific params."""
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
    """Initializes BatchNorm layer with default parameters."""
    super().__init__(params)
    p = self.params
    self._epsilon = 0.001
    self._decay = p.decay

  def _get_weight_shape(self) -> JTensor:
    return [self.params.dim]  # pytype: disable=bad-return-type  # jax-ndarray

  def create_layer_variables(self) -> None:
    """Creates batch normalization layer variables."""
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
    asserts.gt(len(in_shape), 1)
    in_shape[-1] = 1
    return jnp.zeros(in_shape, dtype=inputs.dtype)

  def _get_beta_gamma(self) -> Tuple[JTensor, JTensor]:
    p = self.params
    theta = self.local_theta()
    if p.use_moving_avg_in_training:
      beta = 0.0
      gamma = 1.0
    else:
      beta = theta.beta
      gamma = theta.gamma + 1.0
    return beta, gamma  # pytype: disable=bad-return-type  # jax-ndarray

  def compute_and_update_moments(
      self, inputs: JTensor,
      paddings: JTensor) -> Tuple[JTensor, JTensor, JTensor, JTensor]:
    """Computes moments and updates state.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: The paddings JTensor. Shaped [..., 1], with the same rank as the
        input JTensor.

    Returns:
      Tuple of (mean, variance, beta, gamma).
    """
    p = self.params
    theta = self.local_theta()
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
      self.update_var('moving_mean', new_moving_mean)
      new_moving_variance = (
          theta.moving_variance * p.decay + variance * (1.0 - p.decay))
      self.update_var('moving_variance', new_moving_variance)

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

    beta, gamma = self._get_beta_gamma()
    return norm_mean, norm_variance, beta, gamma

  def fprop(self,
            inputs: JTensor,
            paddings: Optional[JTensor] = None) -> JTensor:
    """Apply batch normalization.

    Args:
      inputs: The inputs JTensor. Shaped [..., dim].
      paddings: The paddings JTensor. Shaped [..., 1].

    Returns:
      Output after applying batch normalization, with the same shape as
        'inputs'.
    """
    p = self.params
    inputs, paddings = self._cast_to_fprop_dtype((inputs, paddings))
    if paddings is None:
      paddings = self._get_default_paddings(inputs)

    asserts.eq(inputs.ndim, paddings.ndim)
    asserts.eq(paddings.shape[-1], 1)

    norm_mean, norm_variance, beta, gamma = self.compute_and_update_moments(
        inputs, paddings)

    inv = gamma / jnp.sqrt(norm_variance + self._epsilon)
    bn_output = (inputs - norm_mean) * inv + beta

    if p.set_padded_output_to_zero:
      bn_output *= 1.0 - paddings

    return bn_output


class LayerNorm(base_layer.BaseLayer):
  """Layer normalization."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Returns the layer params with LayerNorm specific params."""
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input to the network.')
    p.Define('epsilon', 1e-6, 'Tiny value to guard rsqrt.')
    p.Define('scale', True, 'Whether to use a learned scaling.')
    p.Define('bias', True, 'Whether to use bias.')
    return p

  def create_layer_variables(self) -> None:
    """Creates layer normalization variables."""
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

  def fprop(self, inputs: JTensor) -> JTensor:
    """Apply layer norm to inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., input_dims].

    Returns:
      Layer normalized input.
    """
    p = self.params
    theta = self.local_theta()
    mean = jnp.mean(inputs, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=[-1], keepdims=True)
    normed_inputs = (inputs - mean) * jax.lax.rsqrt(var + self.params.epsilon)
    if p.scale:
      normed_inputs *= (1 + theta.scale)
    if p.bias:
      normed_inputs += theta.bias
    return normed_inputs


class RmsNorm(base_layer.BaseLayer):
  """RMS normalization: https://arxiv.org/abs/1910.07467."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Returns the layer params with RMS Norm specific params."""
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input to the network.')
    p.Define('epsilon', 1e-6, 'Tiny value to guard rsqrt.')
    p.Define(
        'direct_scale', True,
        'Whether to apply scale directly without a +1.0. Var is '
        'initialized to 1.0 instead when true. This makes the weight'
        ' compatible with the implementation in gshard/glam.')
    return p

  def create_layer_variables(self) -> None:
    """Creates RMS normalization variables."""
    super().create_layer_variables()
    p = self.params
    wp = p.weight_split_dims_mapping
    wp_scale = wp.wt
    if p.device_mesh is not None and wp.wt is None:
      # Simply replicate the weights.
      wp_scale = [-1]
    # Scale variable that scales the RMS norm output by (1 + scale).
    init_value = 1.0 if p.direct_scale else 0.0
    self.create_variable(
        'scale',
        weight_params(
            shape=[p.input_dims],
            init=WeightInit.Constant(init_value),
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=wp_scale))

  def fprop(self, inputs: JTensor) -> JTensor:
    """Apply RMS norm to inputs.

    Args:
      inputs: The inputs JTensor. Shaped [..., input_dims].

    Returns:
      RMS normalized input.
    """
    theta = self.local_theta()
    var = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)
    normed_inputs = inputs * jax.lax.rsqrt(var + self.params.epsilon)
    scale = theta.scale if self.params.direct_scale else 1 + theta.scale
    normed_inputs *= scale
    return normed_inputs


class GroupNorm(base_layer.BaseLayer):
  """Group normalization layer (https://arxiv.org/abs/1803.08494)."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Returns the layer params with GroupNorm specific params."""
    p = super().Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define('num_groups', 32, 'Number of groups for GroupNorm.')
    p.Define('min_group_size', 1, 'Minimum group size for GroupNorm')
    p.Define('cumulative', False, 'If true, only normalize by current and '
             'previous time steps.')
    p.Define(
        'enable_cross_replica_sum_on_tpu', False,
        'If true, computes global mean and variance across all replicas. '
        'Only effective for tpu.')
    p.Define('input_rank', 4, 'Rank of input. Only 3(BTD) and 4(NHWC) are '
             'supported.')
    p.Define('epsilon', 0.001, 'Epsilon.')
    return p

  def __init__(self, params):
    """Initializes GroupNorm layer and checks parameters."""
    super().__init__(params)
    p = self.params
    asserts.not_none(p.name)
    asserts.gt(p.num_groups, 0)
    asserts.gt(p.min_group_size, 0)
    asserts.le(p.min_group_size, p.dim)
    asserts.eq(p.dim % p.min_group_size, 0)

    if p.dim >= p.num_groups:
      asserts.eq(
          p.dim % p.num_groups,
          0,
          msg='p.dim({0}) is not dividable by p.num_groups({1})'.format(
              p.dim, p.num_groups))

    asserts.in_set(p.input_rank, (3, 4))

  def create_layer_variables(self) -> None:
    """Creates group normalization layer variables."""
    super().create_layer_variables()
    p = self.params
    shape = [1, 1, 1, p.dim] if p.input_rank == 4 else [1, 1, p.dim]
    pc = weight_params(
        shape,
        init=WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[base_layer.SKIP_LP_REGULARIZATION])

    self.create_variable('beta', pc)
    self.create_variable('gamma', pc)

  @property
  def group_size(self) -> int:
    p = self.params
    return max(p.dim // p.num_groups, p.min_group_size)

  @property
  def num_groups(self) -> int:
    p = self.params
    return p.dim // self.group_size

  def _normalize(self, grouped_inputs: JTensor, group_mean: JTensor,
                 group_variance: JTensor) -> JTensor:
    p = self.params
    theta = self.local_theta()
    moment_shape = list(grouped_inputs.shape)
    if p.input_rank == 4:
      moment_shape[2] = 1
    moment_shape[-1] = 1

    if not p.cumulative:
      # If not cumulative, the seqlen dimension is also reduced.
      moment_shape[1] = 1

    group_stddev_inv = jax.lax.rsqrt(group_variance + p.epsilon)

    grouped_inputs = (grouped_inputs - group_mean) * group_stddev_inv
    # Merges the last two dims.
    grouped_inputs = jnp.reshape(grouped_inputs,
                                 list(grouped_inputs.shape[:-2]) + [-1])

    # Note, The real gamma to use is 1 + gamma.
    outputs = grouped_inputs * (1 + theta.gamma) + theta.beta
    return outputs

  def fprop(self,
            inputs: JTensor,
            paddings: Optional[JTensor] = None) -> JTensor:
    """Applies group normalization.

    Args:
      inputs: The inputs JTensor. Shaped [batch_size, height, width, channel] if
        p.rank == 4, else [batch, height, channel].
      paddings: The paddings JTensor. Shaped [batch_size, height]. Intended to
        be used for sequence processing where `height` is `time`.

    Returns:
      Output after applying group normalization, with the same shape as
        'inputs'. Or an output, output_paddings pair if input paddings is not
        None.
    """
    p = self.params
    inputs, paddings = self._cast_to_fprop_dtype((inputs, paddings))
    asserts.eq(inputs.ndim, p.input_rank)

    x = jnp.reshape(
        inputs,
        list(inputs.shape[:-1]) + [self.num_groups, self.group_size])
    expanded_rank = p.input_rank + 1
    all_dims = list(range(expanded_rank))
    if paddings is None or not p.cumulative:
      # Skips batch and num_groups.
      reduce_over_dims = all_dims[1:-2] + all_dims[-1:]
    else:
      # Skips batch, seqlen and num_groups.
      reduce_over_dims = all_dims[2:-2] + all_dims[-1:]

    if paddings is None and not p.cumulative:
      group_mean = jnp.mean(x, axis=reduce_over_dims, keepdims=True)
      group_variance = jnp.mean(
          jnp.square(x - jax.lax.stop_gradient(group_mean)),
          axis=reduce_over_dims,
          keepdims=True)
    else:
      expanded_paddings = jnp.reshape(
          paddings,
          list(inputs.shape[:2]) + [1] * (expanded_rank - 2))
      group_mean, group_variance = compute_moments(
          x,
          expanded_paddings,
          reduce_over_dims,
          cumulative_axis=1,
          enable_cross_replica_sum_on_tpu=p.enable_cross_replica_sum_on_tpu,
          keepdims=True)

    outputs = self._normalize(x, group_mean, group_variance)

    if paddings is None:
      return outputs
    else:
      return outputs, paddings  # pytype: disable=bad-return-type  # jax-ndarray
