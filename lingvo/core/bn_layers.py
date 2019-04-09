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
# =============================================================================
"""Batch normalization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_function
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import summary_utils

_BN_FLOPS_PER_ELEMENT = 10


class AddingAccumulator(base_layer.Accumulator):
  """Accumulator for the sufficient statistics."""

  def __init__(self, shape, dtype):
    super(AddingAccumulator, self).__init__()
    self.dtype = dtype
    self.shape = shape

  def DefaultValue(self):
    """Returns the default value of the accumulator."""
    return tf.zeros(self.shape, dtype=self.dtype)

  def Update(self, value):
    """Adds value to the accumulator."""
    self.SetValue(self.GetValue() + tf.cast(value, self.dtype))


class BatchNormLayer(base_layer.BaseLayer):
  """Batch normalization layer."""

  @classmethod
  def Params(cls):
    p = super(BatchNormLayer, cls).Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define(
        'decay', 0.999,
        'Decay in updating the mean and variance moving average used in'
        ' batch normalization.')
    p.Define(
        'enable_cross_replica_sum_on_tpu', True,
        'If true, calls cross_replica_sum to the aggregate moving averages'
        ' across all replicas.')
    p.Define(
        'use_moving_avg_in_training', False,
        'If True, use global moving avg (mean, variance) during training'
        ' to avoid mismatch between train and eval, which then'
        ' essentially acts as an adaptive normalization step.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BatchNormLayer, self).__init__(params)
    p = self.params
    assert p.name

    pc = py_utils.WeightParams(
        shape=[p.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    with tf.variable_scope(p.name):
      if not p.use_moving_avg_in_training:
        self.CreateVariable('beta', pc)
        # Note, The real gamma to use is 1 + gamma.
        self.CreateVariable('gamma', pc, lambda x: 1.0 + x)

      # Two statistics.
      _, self._moving_mean = py_utils.CreateVariable(
          'moving_mean', pc, trainable=False)

      pc = py_utils.WeightParams(
          shape=[p.dim],
          init=py_utils.WeightInit.Constant(1.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      _, self._moving_variance = py_utils.CreateVariable(
          'moving_variance', pc, trainable=False)
    self._epsilon = 0.001
    self._decay = p.decay

  @property
  def epsilon(self):
    return self._epsilon

  @staticmethod
  def _Moments(inputs, mask, enable_cross_replica_sum_on_tpu=False):
    """Computes mean and variance over the valid data points in inputs."""
    inputs = py_utils.with_dependencies([
        py_utils.assert_equal(tf.rank(inputs), tf.rank(mask)),
        py_utils.assert_greater_equal(mask, tf.zeros_like(mask)),
    ], inputs)
    rank = tf.rank(mask)
    reduce_over_dims = tf.range(0, rank - 1)
    sum_v = tf.reduce_sum(inputs * tf.cast(mask, inputs.dtype),
                          reduce_over_dims)
    count_v = tf.reduce_sum(mask, reduce_over_dims)
    # Input shape is guaranteed to be a multiple of mask shape because the
    # inputs * mask op above was successfully broadcasted.
    mask_multiplier = tf.shape(inputs)[:-1] // tf.shape(mask)[:-1]
    count_v *= tf.cast(tf.reduce_prod(mask_multiplier), count_v.dtype)
    if py_utils.use_tpu() and enable_cross_replica_sum_on_tpu:
      sum_v = tf.contrib.tpu.cross_replica_sum(sum_v)
      count_v = tf.contrib.tpu.cross_replica_sum(count_v)

    count_v = tf.maximum(count_v, 1.0)
    mean = sum_v / count_v
    sum_vv = tf.reduce_sum((inputs - mean) * (inputs - mean) * mask,
                           reduce_over_dims)

    if py_utils.use_tpu() and enable_cross_replica_sum_on_tpu:
      sum_vv = tf.contrib.tpu.cross_replica_sum(sum_vv)

    variance = py_utils.with_dependencies([
        py_utils.assert_greater_equal(sum_vv, tf.zeros_like(sum_vv)),
    ], sum_vv / count_v)
    return mean, variance

  def _GetDefaultPaddings(self, inputs):
    """Gets the default paddings for an input."""
    return tf.zeros(
        tf.concat([tf.shape(inputs)[:-1], [1]], 0), dtype=inputs.dtype)

  def GetCurrentMoments(self, theta):
    """Gets the current computed moments, which should be applied at eval.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
    Returns:
      Tuple of (mean, variance, beta, gamma).
    """
    p = self.params
    if p.use_moving_avg_in_training:
      return self._moving_mean, self._moving_variance, 0.0, 1.0
    else:
      return self._moving_mean, self._moving_variance, theta.beta, theta.gamma

  def ComputeAndUpdateMoments(self, theta, inputs, paddings=None):
    """Computes moments and updates state.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., dim].
      paddings: The paddings tensor.  Shaped [..., 1], with the same rank as the
        input tensor.

    Returns:
      Tuple of (mean, variance, beta, gamma).
    """
    p = self.params
    if paddings is None:
      paddings = self._GetDefaultPaddings(inputs)
    inputs = py_utils.with_dependencies([
        py_utils.assert_shape_match([tf.shape(inputs)[-1]], [p.dim]),
        py_utils.assert_shape_match([tf.shape(paddings)[-1]], [1]),
    ], inputs)
    with tf.name_scope(p.name):
      if p.is_eval:
        # The mean and variance used for normalization.
        norm_mean, norm_variance = self._moving_mean, self._moving_variance
      else:
        mean, variance = self._Moments(inputs, 1.0 - paddings,
                                       p.enable_cross_replica_sum_on_tpu)

        py_utils.UpdateBatchNormVars(self._moving_mean, mean, self._decay)
        py_utils.UpdateBatchNormVars(self._moving_variance, variance,
                                     self._decay)
        # Add some summaries for visualization.
        summary_utils.histogram('%s_mean' % p.name, tf.cast(mean, tf.float32))
        summary_utils.histogram('%s_variance' % p.name,
                                tf.cast(variance, tf.float32))
        summary_utils.histogram('%s_moving_mean' % p.name,
                                tf.cast(self._moving_mean, tf.float32))
        summary_utils.histogram('%s_moving_variance' % p.name,
                                tf.cast(self._moving_variance, tf.float32))
        summary_utils.histogram('%s_mean_diff' % p.name,
                                tf.cast(mean - self._moving_mean, tf.float32))
        summary_utils.histogram(
            '%s_variance_diff' % p.name,
            tf.cast(variance - self._moving_variance, tf.float32))
        if p.use_moving_avg_in_training:
          # Use the global statistics for normalization.
          # Control dependencies on mean and variance make sure
          # moving_mean and variance will be updated for every training step.
          norm_mean = py_utils.with_dependencies([mean], self._moving_mean)
          norm_variance = py_utils.with_dependencies([variance],
                                                     self._moving_variance)
        else:
          # Use the batch statistics for normalization.
          norm_mean = mean
          norm_variance = variance

      norm_mean = py_utils.CheckNumerics(
          norm_mean, 'mean of %s failed numeric check' % p.name)
      norm_variance = py_utils.CheckNumerics(
          norm_variance, 'variance of %s failed numeric check' % p.name)

      if p.use_moving_avg_in_training:
        beta = 0.0
        gamma = 1.0
      else:
        beta = theta.beta
        gamma = theta.gamma
      return norm_mean, norm_variance, beta, gamma

  def FProp(self, theta, inputs, paddings=None):
    """Apply batch normalization.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., dim].
      paddings: The paddings tensor.  Shaped [..., 1], with the same rank as the
        input tensor.

    Returns:
      Output after applying batch normalization, with the same shape as
      'inputs'.
    """
    p = self.params
    if paddings is None:
      paddings = self._GetDefaultPaddings(inputs)
    with tf.name_scope(p.name):
      norm_mean, norm_variance, beta, gamma = self.ComputeAndUpdateMoments(
          theta, inputs, paddings)
      with tf.control_dependencies([
          py_utils.assert_greater_equal(norm_variance,
                                        tf.zeros_like(norm_variance)),
          py_utils.assert_shape_match([p.dim], tf.shape(norm_mean)),
          py_utils.assert_shape_match([p.dim], tf.shape(norm_variance)),
      ]):
        bn_output = tf.nn.batch_normalization(inputs, norm_mean, norm_variance,
                                              beta, gamma, self._epsilon)
      bn_output *= 1.0 - paddings
      return bn_output

  @classmethod
  def FPropMeta(cls, p, inputs, padding=None):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(
        flops=inputs.num_elements() * _BN_FLOPS_PER_ELEMENT,
        out_shapes=(inputs,))


class BatchNormLayerNoPadding(base_layer.BaseLayer):
  """Batchnorm layer without padding."""

  @classmethod
  def Params(cls):
    """Parameters for BatchNormLayerNoPadding."""
    p = super(BatchNormLayerNoPadding, cls).Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define(
        'decay', 0.997,
        'Decay in updating the mean and variance moving average used in'
        ' batch normalization.')
    p.Define('epsilon', 0.001,
             'Small float added to variance to avoid dividing by zero.')
    p.Define(
        'bn_group_size', 1,
        'The number of shards participating in normalization when distributed'
        ' batchnorm is used. Only used for TPU.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BatchNormLayerNoPadding, self).__init__(params)
    p = self.params
    assert p.name, 'Name of BatchNormLayerNoPadding is not set.'
    assert p.dim > 0, 'dim should be positive'
    p.fprop_dtype = None
    # Accumulate bn sufficient stats over micro-batches.
    self.RegisterAccumulator('counts', AddingAccumulator([], p.dtype))
    self.RegisterAccumulator('mean_ss', AddingAccumulator([p.dim], p.dtype))
    self.RegisterAccumulator('variance_ss', AddingAccumulator([p.dim], p.dtype))

    # Skip L-P regularization for these variables.
    collections = [
        self.__class__.__name__ + '_vars', py_utils.SKIP_LP_REGULARIZATION
    ]
    pc = py_utils.WeightParams(
        shape=[p.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=collections)

    with tf.variable_scope(p.name):
      self.CreateVariable('beta', pc)
      # Note, The real gamma to use is 1 + gamma.
      self.CreateVariable('gamma', pc, lambda x: 1.0 + x)

      moving_collections = [
          'moving_vars', tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
          self.__class__.__name__ + '_vars'
      ]
      mva = py_utils.WeightParams(
          shape=[p.dim],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=moving_collections)
      # Two statistics computed from sufficient stats.
      self.CreateVariable('moving_mean', mva, trainable=False)
      mvv = py_utils.WeightParams(
          shape=[p.dim],
          init=py_utils.WeightInit.Constant(1.0),
          dtype=p.dtype,
          collections=moving_collections)
      self.CreateVariable('moving_variance', mvv, trainable=False)

  def PostTrainingStepUpdate(self, global_step):
    """Updates moving_mean, moving_variance after each training step."""
    p = self.params
    # Get sufficient stats that accumulates over microbatches.
    counts = self.accumulators.counts.GetValue()
    mean_ss = self.accumulators.mean_ss.GetValue()
    variance_ss = self.accumulators.variance_ss.GetValue()
    # Compute batch mean and batch variance from sufficient stats
    mean, variance = tf.nn.normalize_moments(counts, mean_ss, variance_ss, None)
    decay = tf.convert_to_tensor(1.0 - p.decay, p.dtype)
    # Update moving_mean, moving_variance from  batch mean and batch variance.
    with tf.name_scope(p.name) as scope:
      with tf.colocate_with(self.vars.moving_mean):
        mean_update = tf.assign_sub(
            self.vars.moving_mean,
            (self.vars.moving_mean - tf.cast(mean, p.dtype)) * decay,
            name='moving_mean_update')
      with tf.colocate_with(self.vars.moving_variance):
        var_update = tf.assign_sub(
            self.vars.moving_variance,
            (self.vars.moving_variance - tf.cast(variance, p.dtype)) * decay,
            name='moving_variance_update')
      py_utils.CheckNumerics(
          self.vars.moving_mean,
          'moving mean of {} failed numeric check'.format(scope))
      py_utils.CheckNumerics(
          self.vars.moving_variance,
          'moving variance of {} failed numeric check'.format(scope))
    self.accumulators.counts.Reset()
    self.accumulators.mean_ss.Reset()
    self.accumulators.variance_ss.Reset()
    return tf.group(mean_update, var_update)

  def _Moments(self, inputs, group_size):
    """Computes mean and variance over N,H,W dimensions in inputs."""
    counts, mean_ss, variance_ss, _, = tf.nn.sufficient_statistics(
        inputs, axes=[0, 1, 2], keep_dims=False)
    self.accumulators.counts.Update(counts)
    self.accumulators.mean_ss.Update(mean_ss)
    self.accumulators.variance_ss.Update(variance_ss)
    # Distributed batch norm that computes sufficient statistics from group_size
    # replicas. This is useful when batch_size_per_replica is too small to
    # compute reliable sufficient statistics.
    if py_utils.use_tpu() and group_size > 1:
      group_assignment = None
      num_shards = tpu_function.get_tpu_context().number_of_shards
      if num_shards is not None:
        if num_shards < group_size:
          raise ValueError('TPU shards={} less than bn_gropu_size={}.'.format(
              num_shards, group_size))
        if num_shards % group_size:
          raise ValueError(
              'TPU shards={} not divisible by bn_group_size={}.'.format(
                  num_shards, group_size))
        num_groups = num_shards // group_size
        group_assignment = []
        for g in range(num_groups):
          replica_ids = [g * group_size + i for i in range(group_size)]
          group_assignment.append(replica_ids)
        counts *= group_size
      mean_ss = tf.contrib.tpu.cross_replica_sum(mean_ss, group_assignment)
      variance_ss = tf.contrib.tpu.cross_replica_sum(variance_ss,
                                                     group_assignment)
    # At each micro-step, batch_mean and batch_variance are computed
    # to normalize inputs. But they are not used to update moving_mean and
    # moving_variance variables until the last micro batch.
    mean, variance = tf.nn.normalize_moments(counts, mean_ss, variance_ss, None)
    return mean, variance

  def FProp(self, theta, inputs):
    """Applies batch normalization.

    Using the implementation in github.com/
    tensorflow/tpu/blob/master/models/official/amoeba_net/network_utils.py#L550

    Args:
      theta: A nested map object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., dim].

    Returns:
      Output after applying batch normalization, with the same shape as
      'inputs'.
    """
    p = self.params
    inputs_dtype = inputs.dtype
    inputs = tf.cast(inputs, p.dtype)
    inputs = py_utils.with_dependencies(
        [py_utils.assert_shape_match([tf.shape(inputs)[-1]], [p.dim])], inputs)
    with tf.name_scope(p.name) as scope:
      if p.is_eval:
        outputs = tf.nn.batch_normalization(inputs, self.vars.moving_mean,
                                            self.vars.moving_variance,
                                            theta.beta, theta.gamma, p.epsilon)
      else:
        mean, variance = self._Moments(inputs, p.bn_group_size)
        mean = py_utils.CheckNumerics(
            mean, 'mean of {} failed numeric check'.format(scope))
        variance = py_utils.CheckNumerics(
            variance, 'variance of {} failed numeric check'.format(scope))
        outputs = tf.nn.batch_normalization(inputs, mean, variance, theta.beta,
                                            theta.gamma, p.epsilon)
      outputs.set_shape(inputs.get_shape())
      return tf.cast(outputs, inputs_dtype)

  @classmethod
  def FPropMeta(cls, p, inputs):
    """Returns metadata about the `FProp` computation for this layer."""
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(
        flops=inputs.num_elements() * _BN_FLOPS_PER_ELEMENT,
        out_shapes=(inputs,))
