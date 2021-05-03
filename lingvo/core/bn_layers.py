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
# =============================================================================
"""Batch normalization layers."""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import summary_utils

from tensorflow.python.ops import nn  # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_function  # pylint:disable=g-direct-tensorflow-import

_BN_FLOPS_PER_ELEMENT = 10


# TODO(rpang): move AddingAccumulator to a separate library.
class AddingAccumulator(base_layer.Accumulator):
  """Accumulator for the sufficient statistics."""

  def __init__(self, shape, dtype):
    super().__init__()
    self.dtype = dtype
    self.shape = shape

  def DefaultValue(self):
    """Returns the default value of the accumulator."""
    return tf.zeros(self.shape, dtype=self.dtype)

  def Update(self, value):
    """Adds value to the accumulator."""
    self.SetValue(self.GetValue() + tf.cast(value, self.dtype))


def ComputeMoments(inputs,
                   padding,
                   reduce_over_dims,
                   cumulative_axis=None,
                   enable_cross_replica_sum_on_tpu=False,
                   keepdims=False):
  """Computes mean and variance over the valid data points in inputs."""
  mask = 1.0 - padding
  inputs = py_utils.with_dependencies([
      py_utils.assert_equal(tf.rank(inputs), tf.rank(mask)),
      py_utils.assert_greater_equal(mask, tf.zeros_like(mask)),
  ], inputs)
  sum_v = tf.reduce_sum(
      inputs * tf.cast(mask, inputs.dtype), reduce_over_dims, keepdims=keepdims)
  count_v = tf.reduce_sum(mask, reduce_over_dims, keepdims=keepdims)

  if cumulative_axis is not None:
    sum_v = tf.math.cumsum(sum_v, axis=cumulative_axis)
    count_v = tf.math.cumsum(count_v, axis=cumulative_axis)
  # Input shape is guaranteed to be a multiple of mask shape because the
  # inputs * mask op above was successfully broadcasted.
  input_size_on_reduced_dims = tf.reduce_prod(
      tf.gather(tf.shape(inputs), reduce_over_dims))
  mask_size_on_reduced_dims = tf.reduce_prod(
      tf.gather(tf.shape(mask), reduce_over_dims))
  mask_multiplier = tf.math.truediv(input_size_on_reduced_dims,
                                    mask_size_on_reduced_dims)
  count_v *= tf.cast(mask_multiplier, count_v.dtype)
  if py_utils.use_tpu() and enable_cross_replica_sum_on_tpu:
    sum_v = tf.tpu.cross_replica_sum(sum_v)
    count_v = tf.tpu.cross_replica_sum(count_v)

  count_v = tf.maximum(count_v, 1.0)
  mean = sum_v / count_v
  sum_vv = tf.reduce_sum(
      (inputs - mean) * (inputs - mean) * mask,
      reduce_over_dims,
      keepdims=keepdims)
  if cumulative_axis is not None:
    sum_vv = tf.math.cumsum(sum_vv, axis=cumulative_axis)

  if py_utils.use_tpu() and enable_cross_replica_sum_on_tpu:
    sum_vv = tf.tpu.cross_replica_sum(sum_vv)

  variance = py_utils.with_dependencies([
      py_utils.assert_greater_equal(sum_vv, tf.zeros_like(sum_vv)),
  ], sum_vv / count_v)
  return mean, variance


class BatchNormLayer(base_layer.BaseLayer):
  """Batch normalization layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define(
        'decay', 0.999,
        'Decay in updating the mean and variance moving average used in'
        ' batch normalization.')
    p.Define(
        'enable_cross_replica_sum_on_tpu', True,
        'If true, computes global mean and variance across all replicas.'
        'Only effective for tpu.')
    p.Define(
        'use_moving_avg_in_training', False,
        'If True, use global moving avg (mean, variance) during training'
        ' to avoid mismatch between train and eval, which then'
        ' essentially acts as an adaptive normalization step.')
    p.Define(
        'freeze_bn_stats', False,
        'If True, uses moving avg (mean, variance) during both training and '
        'inference. It behaves like force_eval but the gamma/beta are still '
        'trained when do_eval is False. The moving mean/var can be set by '
        'loading pretrained checkpoints. A use case is training detectors '
        'based on an pretrained checkpoint while BN stats are frozen.')
    p.Define(
        'gamma_zero_init', False,
        'If True, initialize gamma to zeros according to the technique '
        'introduced in the tech report: https://arxiv.org/abs/1706.02677')
    p.Define(
        'gamma_one_init', False,
        'If True, explicitly initialize gamma to one without invoking '
        'theta_fn.')
    # TODO(rpang): remove this hparam, as it is replaced
    # by p.train.ema_decay_moving_vars.
    p.Define(
        'add_stats_to_moving_average_variables', None,
        'If True, adds (mean, variance) to the MOVING_AVERAGE_VARIABLES '
        'collection to be compatible with ema_decay. '
        'Recommendation: set to True for new models, and to False to maintain '
        'checkpoint compatibility.')
    p.Define('set_padded_output_to_zero', True,
             'If True, sets the padded outputs to zero.')
    p.Define(
        'use_fused_batch_norm_for_eval', False,
        'If True, uses tf.compat.v1.nn.fused_batch_norm instead of '
        'tf.nn.batch_normalization during eval. The fused version may be more '
        'efficient but it has more restrictions on the expected input shapes.'
        'The input tensor has to be rank 4, where the first dimension '
        'corresponds to the batch, and the last dimension corresponds to the '
        'features to normalize over. This usually corresponds to NHWC with '
        'image inputs. Note that fused_batch_norm wants to track its own '
        'mean and variance during training, so we are unable to use it '
        'for training since we want to have a custom mean and variance to '
        'support padding.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._epsilon = 0.001
    self._decay = p.decay

  def _GetWeightShape(self):
    return [self.params.dim]

  def _CreateLayerVariables(self):
    p = self.params

    beta_pc = py_utils.WeightParams(
        shape=self._GetWeightShape(),
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    gamma_pc = py_utils.WeightParams(
        shape=self._GetWeightShape(),
        init=py_utils.WeightInit.Constant(1.0)
        if p.gamma_one_init else py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    if not p.use_moving_avg_in_training:
      self.CreateVariable('beta', beta_pc)
      if p.gamma_zero_init or p.gamma_one_init:
        # initialization to BN gamma
        self.CreateVariable('gamma', gamma_pc)
      else:
        # Note, The real gamma to use is 1 + gamma.
        self.CreateVariable('gamma', gamma_pc, lambda x: 1.0 + x)

    # Two statistics.
    moving_collections = ['moving_vars', self.__class__.__name__ + '_vars']
    if p.add_stats_to_moving_average_variables:
      moving_collections += [tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    elif p.add_stats_to_moving_average_variables is None:
      # TODO(rpang): force all models to set this param explicitly.
      tf.logging.warning(
          'BatchNormLayer.add_stats_to_moving_average_variables should be '
          'set to True for new models, and to False explicitly for '
          'checkpoint compatibility.')
    # Add to the MOVING_AVERAGE_VARIABLES collection so that they are returned
    # by tf.moving_average_variables() and included in EMA variables if
    # ema_decay is enabled.
    mva = py_utils.WeightParams(
        shape=[p.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=moving_collections)
    self.CreateVariable(
        'moving_mean',
        mva,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

    mvv = py_utils.WeightParams(
        shape=[p.dim],
        init=py_utils.WeightInit.Constant(1.0),
        dtype=p.dtype,
        collections=moving_collections)
    self.CreateVariable(
        'moving_variance',
        mvv,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

  @property
  def epsilon(self):
    return self._epsilon

  def _GetDefaultPaddings(self, inputs):
    """Gets the default paddings for an input."""
    return tf.zeros(
        tf.concat([tf.shape(inputs)[:-1], [1]], 0), dtype=inputs.dtype)

  def _GetBetaGamma(self, theta, inputs, **kwargs):
    del inputs
    del kwargs
    p = self.params
    if p.use_moving_avg_in_training:
      beta = 0.0
      gamma = 1.0
    else:
      beta = theta.beta
      gamma = theta.gamma
    return beta, gamma

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
      return self.vars.moving_mean, self.vars.moving_variance, 0.0, 1.0
    else:
      return (self.vars.moving_mean, self.vars.moving_variance, theta.beta,
              theta.gamma)

  def ComputeAndUpdateMoments(self, theta, inputs, paddings=None, **kwargs):
    """Computes moments and updates state.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., dim].
      paddings: The paddings tensor.  Shaped [..., 1], with the same rank as the
        input tensor.
      **kwargs: Additional inputs.

    Returns:
      Tuple of (mean, variance, beta, gamma).
    """
    p = self.params
    if paddings is None:
      paddings = self._GetDefaultPaddings(inputs)
    inputs = py_utils.with_dependencies([
        py_utils.assert_shape_match([tf.shape(paddings)[-1]], [1]),
    ], inputs)
    with tf.name_scope(p.name):
      if self.do_eval or p.freeze_bn_stats:
        # The mean and variance used for normalization.
        norm_mean, norm_variance = (self.vars.moving_mean,
                                    self.vars.moving_variance)
      else:
        rank = tf.rank(paddings)
        reduce_over_dims = tf.range(0, rank - 1)
        mean, variance = ComputeMoments(inputs, paddings, reduce_over_dims,
                                        None, p.enable_cross_replica_sum_on_tpu)

        py_utils.UpdateBatchNormVars(self.vars.moving_mean, mean, self._decay)
        py_utils.UpdateBatchNormVars(self.vars.moving_variance, variance,
                                     self._decay)
        # Add some summaries for visualization.
        summary_utils.histogram('%s_mean' % p.name, tf.cast(mean, tf.float32))
        summary_utils.histogram('%s_variance' % p.name,
                                tf.cast(variance, tf.float32))
        summary_utils.histogram('%s_moving_mean' % p.name,
                                tf.cast(self.vars.moving_mean, tf.float32))
        summary_utils.histogram('%s_moving_variance' % p.name,
                                tf.cast(self.vars.moving_variance, tf.float32))
        summary_utils.histogram(
            '%s_mean_diff' % p.name,
            tf.cast(
                tf.cast(mean, self.vars.moving_mean.dtype.base_dtype) -
                self.vars.moving_mean, tf.float32))
        summary_utils.histogram(
            '%s_variance_diff' % p.name,
            tf.cast(
                tf.cast(variance, self.vars.moving_variance.dtype.base_dtype) -
                self.vars.moving_variance, tf.float32))
        if p.use_moving_avg_in_training:
          # Use the global statistics for normalization.
          # Control dependencies on mean and variance make sure
          # moving_mean and variance will be updated for every training step.
          norm_mean = py_utils.with_dependencies([mean], self.vars.moving_mean)
          norm_variance = py_utils.with_dependencies([variance],
                                                     self.vars.moving_variance)
        else:
          # Use the batch statistics for normalization.
          norm_mean = mean
          norm_variance = variance

      norm_mean = py_utils.CheckNumerics(
          norm_mean, 'mean of %s failed numeric check' % p.name)
      norm_variance = py_utils.CheckNumerics(
          norm_variance, 'variance of %s failed numeric check' % p.name)

      beta, gamma = self._GetBetaGamma(theta, inputs, **kwargs)
      return norm_mean, norm_variance, beta, gamma

  def _ComputeBN(self, inputs, paddings, gamma, beta, norm_mean, norm_variance):
    p = self.params
    with tf.control_dependencies([
        py_utils.assert_greater_equal(norm_variance,
                                      tf.zeros_like(norm_variance)),
        py_utils.assert_shape_match([tf.shape(inputs)[-1]],
                                    tf.shape(norm_mean)),
        py_utils.assert_shape_match([tf.shape(inputs)[-1]],
                                    tf.shape(norm_variance)),
    ]):
      if p.use_fused_batch_norm_for_eval and (self.do_eval or
                                              p.freeze_bn_stats):
        bn_output, _, _ = nn.fused_batch_norm(
            inputs,
            gamma,
            beta,
            norm_mean,
            norm_variance,
            self._epsilon,
            is_training=False)
      else:
        bn_output = tf.nn.batch_normalization(inputs, norm_mean, norm_variance,
                                              beta, gamma, self._epsilon)
      if p.set_padded_output_to_zero:
        bn_output *= 1.0 - paddings
    return bn_output

  def _MaybeExpandPaddings(self, inputs, paddings):
    # rank difference is at most one.
    rank_diff = tf.rank(inputs) - tf.rank(paddings)
    paddings = py_utils.with_dependencies([
        py_utils.assert_less_equal(rank_diff, 1),
        py_utils.assert_greater_equal(rank_diff, 0)
    ], paddings)

    # Pads [1] to the end of paddings.
    paddings = tf.reshape(
        paddings,
        tf.concat(
            [tf.shape(paddings), tf.tile([1], [rank_diff])], axis=0))
    return paddings

  def FProp(self, theta, inputs, paddings=None):
    """Apply batch normalization.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., dim].
      paddings: The paddings tensor.  Shaped [..., 1] or [...], the rank is
        either the same as inputs or tf.rank(inputs) - 1.

    Returns:
      Output after applying batch normalization, with the same shape as
      'inputs'.
    """
    inputs, paddings = self._CastToFPropDtype((inputs, paddings))

    if py_utils.testonly_skip_norm_layers():
      return inputs

    p = self.params
    if paddings is None:
      paddings = self._GetDefaultPaddings(inputs)

    # shape [..., 1]
    paddings = self._MaybeExpandPaddings(inputs, paddings)

    with tf.name_scope(p.name):
      norm_mean, norm_variance, beta, gamma = self.ComputeAndUpdateMoments(
          theta, inputs, paddings)

      return self._ComputeBN(inputs, paddings, gamma, beta, norm_mean,
                             norm_variance)

  @classmethod
  def FPropMeta(cls, p, inputs, padding=None):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(
        flops=inputs.num_elements() * _BN_FLOPS_PER_ELEMENT,
        out_shapes=(inputs,))


class CategoricalBN(BatchNormLayer):
  """Implements a categorical BN which is akin to ...

  https://arxiv.org/pdf/1809.11096.pdf

  Specifically, the moving stats are category-agnostic, while {beta, gamma} are
  category-aware.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('class_emb_dim', None, 'Dim of input class embedding.')

    p.use_moving_avg_in_training = False
    p.use_fused_batch_norm_for_eval = False
    p.add_stats_to_moving_average_variables = True
    return p

  def __init__(self, params):
    assert params.name
    assert not params.use_moving_avg_in_training
    assert not params.use_fused_batch_norm_for_eval
    assert params.add_stats_to_moving_average_variables
    super().__init__(params)

  def _GetWeightShape(self):
    return [self.params.class_emb_dim, self.params.dim]

  def _GetBetaGamma(self, theta, inputs, **kwargs):
    assert 'class_emb' in kwargs
    class_emb = kwargs['class_emb']

    # class_emb is a one-hot vector of shape [batch, class_emb_dim=num_classes].
    class_ids = tf.math.argmax(class_emb, axis=-1, output_type=tf.int32)
    # [batch, dim]
    # Not using matmul/einsum to avoid potential precision problem on TPU with
    # sparse inputs.
    beta = tf.gather(theta.beta, class_ids)
    gamma = tf.gather(theta.gamma, class_ids)

    # Extend to [batch, 1, ... 1, dim]
    batch = py_utils.GetShape(inputs)[0]
    to_shape = tf.concat(
        [[batch],
         tf.ones([py_utils.GetRank(inputs) - 2], tf.int32), [self.params.dim]],
        axis=0)
    beta = tf.reshape(beta, to_shape)
    gamma = tf.reshape(gamma, to_shape)
    return beta, gamma

  def FProp(self, theta, inputs, paddings, class_emb):
    """Apply batch normalization.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [batch, ..., dim].
      paddings: The paddings tensor.  Shaped [batch, ..., 1], with the same rank
        as the input tensor.
      class_emb: The conditioning inputs, Shaped [batch, emb_dim].

    Returns:
      Output after applying batch normalization, with the same shape as
      'inputs'.
    """
    if py_utils.testonly_skip_norm_layers():
      return inputs

    p = self.params
    batch = py_utils.GetShape(inputs)[0]
    class_emb = py_utils.HasShape(class_emb, [batch, p.class_emb_dim])
    if not py_utils.use_tpu():
      class_emb = py_utils.with_dependencies([
          py_utils.assert_less_equal(
              tf.cast(class_emb, tf.int32), 1, name='one_hot_assert1'),
          py_utils.assert_greater_equal(
              tf.cast(class_emb, tf.int32), 0, name='one_hot_assert2'),
          py_utils.assert_equal(
              tf.ones([batch], tf.int32),
              tf.cast(tf.reduce_sum(class_emb, -1), tf.int32),
              name='one_hot_assert3'),
      ], class_emb)

    with tf.name_scope(p.name):
      norm_mean, norm_variance, beta, gamma = self.ComputeAndUpdateMoments(
          theta, inputs, paddings=paddings, class_emb=class_emb)
      return self._ComputeBN(inputs, paddings, gamma, beta, norm_mean,
                             norm_variance)


class BatchNormLayerNoPadding(base_layer.BaseLayer):
  """Batchnorm layer without padding."""

  @classmethod
  def Params(cls):
    """Parameters for BatchNormLayerNoPadding."""
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name, 'Name of BatchNormLayerNoPadding is not set.'
    p.fprop_dtype = None

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params

    # Skip L-P regularization for these variables.
    collections = [
        self.__class__.__name__ + '_vars', py_utils.SKIP_LP_REGULARIZATION
    ]
    pc = py_utils.WeightParams(
        shape=[p.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=collections)

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

    # Accumulate bn sufficient stats over micro-batches.
    dim = self.vars.beta.shape[0]
    self.RegisterAccumulator('counts', AddingAccumulator([], p.dtype))
    self.RegisterAccumulator('mean_ss', AddingAccumulator([dim], p.dtype))
    self.RegisterAccumulator('variance_ss', AddingAccumulator([dim], p.dtype))

  def PostTrainingStepUpdate(self):
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
      with tf.ops.colocate_with(self.vars.moving_mean):
        mean_update = tf.assign_sub(
            self.vars.moving_mean,
            tf.where(
                tf.greater(counts, 0.5),
                (self.vars.moving_mean - tf.cast(mean, p.dtype)) * decay,
                tf.zeros_like(self.vars.moving_mean)),
            name='moving_mean_update')
      with tf.ops.colocate_with(self.vars.moving_variance):
        var_update = tf.assign_sub(
            self.vars.moving_variance,
            tf.where(
                tf.greater(counts, 0.5),
                (self.vars.moving_variance - tf.cast(variance, p.dtype)) *
                decay, tf.zeros_like(self.vars.moving_variance)),
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
        inputs, axes=[0, 1, 2], keepdims=False)
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
      mean_ss = tf.tpu.cross_replica_sum(mean_ss, group_assignment)
      variance_ss = tf.tpu.cross_replica_sum(variance_ss, group_assignment)
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
    if py_utils.testonly_skip_norm_layers():
      return inputs

    p = self.params
    inputs_dtype = inputs.dtype
    inputs = tf.cast(inputs, p.dtype)
    inputs = py_utils.with_dependencies([
        py_utils.assert_shape_match([tf.shape(inputs)[-1]], tf.shape(
            theta.beta))
    ], inputs)
    with tf.name_scope(p.name) as scope:
      if self.do_eval:
        outputs = tf.nn.batch_normalization(inputs, theta.moving_mean,
                                            theta.moving_variance,
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


class GroupNormLayer(base_layer.BaseLayer):
  """Group normalization layer(https://arxiv.org/abs/1803.08494)."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define('num_groups', 32, 'Number of groups for GroupNorm.')
    p.Define('min_group_size', 1, 'Minimum group size for GroupNorm')
    p.Define('cumulative', False, 'If true, only normalize by current and '
             'previous time steps.')
    p.Define(
        'enable_cross_replica_sum_on_tpu', False,
        'If true, computes global mean and variance across all replicas.'
        'Only effective for tpu.')
    p.Define('input_rank', 4, 'Rank of input. Only 3(BTD) and 4(NHWC) are '
             'supported.')
    p.Define('epsilon', 0.001, 'Epsilon.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.num_groups > 0
    assert p.min_group_size > 0
    assert p.dim % p.min_group_size == 0

    if p.dim >= p.num_groups:
      assert p.dim % p.num_groups == 0, ('p.dim({0}) is not dividable by '
                                         'p.num_groups({1})').format(
                                             p.dim, p.num_groups)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    assert p.input_rank == 3 or p.input_rank == 4

    collections = [
        self.__class__.__name__ + '_vars', py_utils.SKIP_LP_REGULARIZATION
    ]

    shape = [1, 1, 1, p.dim] if p.input_rank == 4 else [1, 1, p.dim]
    pc = py_utils.WeightParams(
        shape=shape,
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=collections)

    self.CreateVariable('beta', pc)
    self.CreateVariable('gamma', pc)

  @property
  def group_size(self):
    p = self.params
    assert p.min_group_size <= p.dim
    return max(p.dim // p.num_groups, p.min_group_size)

  @property
  def num_groups(self):
    p = self.params
    return p.dim // self.group_size

  def zero_state(self, batch_size):
    p = self.params
    num_groups = self.num_groups

    if not p.cumulative:
      return py_utils.NestedMap()

    if p.input_rank == 4:
      cache_shape = [batch_size, 1, 1, num_groups, 1]
    else:
      cache_shape = [batch_size, 1, num_groups, 1]
    cached_sum = tf.zeros(cache_shape, py_utils.FPropDtype(p))
    cached_count = tf.zeros(cache_shape, py_utils.FPropDtype(p))
    cached_var = tf.zeros(cache_shape, py_utils.FPropDtype(p))
    return py_utils.NestedMap(
        cached_sum=cached_sum, cached_count=cached_count, cached_var=cached_var)

  def _Normalize(self, theta, grouped_inputs, group_mean, group_variance):
    p = self.params
    group_mean = py_utils.CheckNumerics(
        group_mean, f'mean of {p.name} failed numeric check.')
    group_variance = py_utils.CheckNumerics(
        group_variance, f'variance of {p.name} failed numeric check.')

    input_shape = py_utils.GetShape(grouped_inputs)
    moment_shape = list(input_shape)
    if p.input_rank == 4:
      moment_shape[2] = 1
      moment_shape[-1] = 1
    else:
      moment_shape[-1] = 1
    if not p.cumulative:
      # If not cumulative, the seqlen dimension is also reduced.
      moment_shape[1] = 1

    group_mean = py_utils.HasShape(group_mean, moment_shape)
    group_variance = py_utils.HasShape(group_variance, moment_shape)
    group_variance = py_utils.with_dependencies([
        py_utils.assert_greater_equal(group_variance,
                                      tf.cast(0, group_variance.dtype))
    ], group_variance)

    if group_variance.dtype == tf.bfloat16:
      # tf.rsqrt is not implemented for bfloat16, hence we always cast into
      # tf.float32.
      group_stddev_inv = tf.cast(
          tf.math.rsqrt(tf.cast(group_variance + p.epsilon, tf.float32)),
          group_mean.dtype)
    else:
      group_stddev_inv = tf.math.rsqrt(group_variance + p.epsilon)

    grouped_inputs = (grouped_inputs - group_mean) * group_stddev_inv
    # Merges the last two dims.
    grouped_inputs = tf.reshape(grouped_inputs, input_shape[:-2] + [-1])

    # Note, The real gamma to use is 1 + gamma.
    outputs = grouped_inputs * (theta.gamma + 1) + theta.beta
    return outputs

  def FProp(self, theta, inputs, paddings=None):
    """Apply group normalization.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: The inputs tensor with shape [batch_size, height, width, channel].
        if p.rank == 4, else [batch, height, channel].
      paddings: The paddings tensor with shape [batch_size, height]. Intended to
        be used for sequence processing where `height` is `time`.

    Returns:
      A single tensor as the output after applying group normalization, with
      the same shape as 'inputs'. Or a output, output_paddings pair if input
      paddings is not None.
    """
    inputs, paddings = self._CastToFPropDtype((inputs, paddings))

    if py_utils.testonly_skip_norm_layers():
      if paddings is None:
        return inputs
      else:
        return inputs, paddings

    p = self.params
    inputs = py_utils.HasRank(inputs, p.input_rank)
    num_groups = self.num_groups

    input_shape = py_utils.GetShape(inputs)
    with tf.name_scope(p.name):
      x = tf.reshape(inputs, input_shape[:-1] + [num_groups, self.group_size])
      expanded_rank = p.input_rank + 1
      all_dims = list(range(expanded_rank))
      if paddings is None or not p.cumulative:
        # Skips batch and num_groups.
        reduce_over_dims = all_dims[1:-2] + all_dims[-1:]
      else:
        # Skips batch, seqlen and num_groups.
        reduce_over_dims = all_dims[2:-2] + all_dims[-1:]

      if paddings is None and not p.cumulative:
        # Fast path on tpu without reshape.
        counts, means_ss, variance_ss, _, = tf.nn.sufficient_statistics(
            x, axes=reduce_over_dims, keepdims=True)
        group_mean, group_variance = tf.nn.normalize_moments(
            counts, means_ss, variance_ss, None)
      else:
        expanded_paddings = tf.reshape(
            paddings, input_shape[:2] + [1] * (expanded_rank - 2))
        group_mean, group_variance = ComputeMoments(
            x,
            expanded_paddings,
            reduce_over_dims,
            cumulative_axis=1,
            enable_cross_replica_sum_on_tpu=p.enable_cross_replica_sum_on_tpu,
            keepdims=True)

      outputs = self._Normalize(theta, x, group_mean, group_variance)

      if paddings is None:
        return outputs
      else:
        return outputs, paddings

  def _StreamMoments(self, inputs, paddings, cached_sum, cached_count,
                     cached_var):
    """Computes mean and variance over the valid data points in inputs.

    Args:
      inputs: [B, T, F, N, G] or [B, T, N, G]
      paddings: [B, T, 1, 1, 1] or [B, T, 1, 1]
      cached_sum: [B, 1, 1, N, 1] or [B, 1, N, 1]
      cached_count: same shape as cached_sum.
      cached_var: same shape as cached_sum.

    Returns:
      mean: [B, T, 1, N, 1] or [B, T, N, 1]
      variance: same shape as mean.
      new_cached_sum: same shape as cached_sum.
      new_cached_count: same shape as cached_count.
    """
    tf.logging.vlog(1, 'inputs: %r', inputs)
    tf.logging.vlog(1, 'paddings: %r', paddings)
    tf.logging.vlog(1, 'cached_sum: %r', cached_sum)
    tf.logging.vlog(1, 'cached_count: %r', cached_count)

    mask = tf.cast(1.0 - paddings, inputs.dtype)
    inputs *= tf.cast(mask, inputs.dtype)

    input_rank = py_utils.GetRank(inputs)
    assert input_rank is not None, (f'inputs rank must be staic for '
                                    f'{repr(inputs)}')
    reduce_over_dims = list(range(input_rank))
    # Skip B, T, and N. Reduce {F,G} or just G.
    reduce_over_dims = reduce_over_dims[2:-2] + reduce_over_dims[-1:]
    tf.logging.vlog(1, 'reduce_over_dims: %s', reduce_over_dims)

    # [B, T, 1, N, 1] or [B, T, N, 1]
    sum_v = tf.reduce_sum(inputs, reduce_over_dims, keepdims=True)
    sum_v = tf.math.cumsum(sum_v, axis=1)
    sum_v += cached_sum

    # [B, T, 1, 1, 1] or [B, T, 1, 1]
    count_v = tf.reduce_sum(mask, reduce_over_dims, keepdims=True)
    count_v = tf.math.cumsum(count_v, axis=1)
    input_shape = py_utils.GetShape(inputs)
    if input_rank == 4:
      # F * G
      multiplier = input_shape[-1] * input_shape[-3]
    else:
      # G
      multiplier = input_shape[-1]
    count_v *= multiplier
    count_v += cached_count

    tf.logging.vlog(1, 'sum_v: %r', sum_v)
    tf.logging.vlog(1, 'count_v: %r', count_v)

    mean = sum_v / tf.maximum(count_v, 1.0)
    if py_utils.FLAGS.tflite_compatible:
      # TfLite doesn't support broadcasting with 5D tensors.
      inputs_shape = py_utils.GetShape(inputs)
      if len(inputs_shape) == 4:
        tiled_mean = tf.tile(mean, [1, 1, 1, inputs_shape[3]])
      else:
        tiled_mean = tf.tile(mean, [1, 1, inputs_shape[2], 1, inputs_shape[4]])
      sum_vv = tf.reduce_sum(
          tf.math.square(inputs - tiled_mean) * mask,
          reduce_over_dims,
          keepdims=True)
    else:
      sum_vv = tf.reduce_sum(
          (inputs - mean)**2 * mask, reduce_over_dims, keepdims=True)
    sum_vv = tf.math.cumsum(sum_vv, axis=1)
    sum_vv += cached_var

    cached_sum = sum_v[:, -1:]
    cached_count = count_v[:, -1:]
    cached_var = sum_vv[:, -1:]

    variance = py_utils.with_dependencies([
        py_utils.assert_greater_equal(sum_vv, tf.cast(0, sum_vv.dtype)),
    ], sum_vv / tf.maximum(count_v, 1.0))
    return mean, variance, cached_sum, cached_count, cached_var

  def StreamStep(self, theta, inputs, paddings, state0):
    if py_utils.testonly_skip_norm_layers():
      return inputs, paddings, state0

    p = self.params
    assert p.cumulative
    inputs = py_utils.HasRank(inputs, p.input_rank)

    group_size = self.group_size
    num_groups = self.num_groups
    tf.logging.vlog(1, 'group_size: %s', group_size)
    tf.logging.vlog(1, 'num_groups: %s', num_groups)

    input_shape = py_utils.GetShape(inputs)
    with tf.name_scope(f'{p.name}/StreamStep'):
      x = tf.reshape(inputs, input_shape[:-1] + [num_groups, group_size])
      expanded_rank = p.input_rank + 1
      expanded_paddings = tf.reshape(
          paddings, input_shape[:2] + [1] * (expanded_rank - 2))
      (group_mean, group_variance, cached_sum, cached_count,
       cached_var) = self._StreamMoments(x, expanded_paddings,
                                         state0.cached_sum, state0.cached_count,
                                         state0.cached_var)
      outputs = self._Normalize(theta, x, group_mean, group_variance)

      return outputs, paddings, py_utils.NestedMap(
          cached_sum=cached_sum,
          cached_count=cached_count,
          cached_var=cached_var)

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    flops_per_element = 10  # Approximately 10 flops per element.
    return py_utils.NestedMap(
        flops=inputs.num_elements() * flops_per_element, out_shapes=(inputs,))
