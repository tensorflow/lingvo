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
"""Module for all optimizers."""

import dataclasses
import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from lingvo.jax import gshard_utils
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import optax

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedParams = pytypes.NestedParams

# Initializes sharding spec for the optimizer state variables.
TransformInitPartitionSpecFn = Callable[[NestedParams],
                                        Union[NestedParams,
                                              Sequence[NestedParams]]]


# Extension of optax.GradientTransformation that supports spmd sharding and
# explicit annotation of sharding specs for the optimizer state variables.
@dataclasses.dataclass(frozen=True)
class ShardedGradientTransformation:
  """GradientTransformation that supports spmd."""
  # init_fn and update_fn are the same as in optax.GradientTransformation
  init: optax.TransformInitFn
  update: optax.TransformUpdateFn
  # Input is the sharding specs of the variables used in the forward
  # computation.  Output is the sharding specs of the optimizer state variables.
  #
  # Constraints: output from this function should be of identical structure as
  # that of the init() function.
  init_partition_spec: TransformInitPartitionSpecFn


# pylint: disable=invalid-name
def count_init_fn(_):
  """Common init_fn that initializes a count for global step."""
  return NestedMap(count=jnp.zeros([], jnp.int32))


def count_init_partition_spec_fn(var_params):
  """Init partition spec for only partitioning the count/step."""
  var_spec_flattened, _ = jax.tree_flatten(var_params)
  assert var_spec_flattened
  first_var = var_spec_flattened[0]
  assert isinstance(first_var, py_utils.Params)
  device_mesh = first_var.device_mesh
  return NestedMap(
      count=py_utils.WeightParams(
          shape=[],
          init=None,
          dtype=jnp.int32,
          collections=None,
          device_mesh=device_mesh,
          tensor_split_dims_mapping=[]))


def sharded_sgd(learning_rate_fn: optax.Schedule, momentum: Optional[float],
                nesterov: bool) -> ShardedGradientTransformation:
  """A canonical Stochastic Gradient Descent optimiser that supports spmd ...

  sharding.

  This implements stochastic gradient descent. It also includes support for
  momentum, and nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    momentum: (default `None`), the `decay` rate used by the momentum term, when
      it is set to `None`, then momentum is not used at all.
    nesterov (default `False`): whether nesterov momentum is used.

  Returns:
    A `ShardedGradientTransformation`.
  """
  # TODO(yonghui): support momentum.
  assert momentum is None
  del nesterov

  def update_fn(updates, state, params=None):
    del params
    step_size = -1.0 * learning_rate_fn(state.count)
    updates = jax.tree_map(lambda g: jnp.array(step_size, dtype=g.dtype) * g,
                           updates)
    updated_states = NestedMap(count=state.count + jnp.array(1, jnp.int32))
    return updates, updated_states

  return ShardedGradientTransformation(
      init=count_init_fn,
      update=update_fn,
      init_partition_spec=count_init_partition_spec_fn)


class _AdamOptState:

  def __init__(self, *, m, v):
    self.m = m
    self.v = v


class _ShardedAdamOptimizerHelper:
  """A helper class facilitates the creation of sharded_adam_optimizer."""

  def OptStateShardingSpec(self, var_params: py_utils.Params) -> _AdamOptState:
    """Returns optimizer sharding spec for one particular variable."""
    m_var_params = var_params.Copy()
    m_var_params.init = None
    v_var_params = var_params.Copy()
    v_var_params.init = None
    # m and v simply share the same sharding.
    return _AdamOptState(m=m_var_params, v=v_var_params)

  def InitOptState(self, var_params: py_utils.Params) -> _AdamOptState:
    """Returns optimizer state for one particular variable."""
    return _AdamOptState(
        m=jnp.zeros_like(var_params), v=jnp.zeros_like(var_params))

  def SanitizeValues(self, array: JTensor, replacement: float = 0.0):
    """Sanitizes NaN and Infinity values."""
    return jnp.nan_to_num(
        array, nan=replacement, posinf=replacement, neginf=replacement)

  def BiasCorrectedDecay(self, step: JTensor, decay: float) -> JTensor:
    """Incorporates bias correction into decay.

    Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
    derivation of the formulas below. With bias-corrected decay, we can simply
    do

    m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
    v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

    without further bias correction.

    Args:
      step: current step, 0-based.
      decay: the raw decay. As t -> infinity, bias corrected decay converges to
        this value.

    Returns:
      Bias corrected decay.
    """
    t = step.astype(jnp.float32) + 1.
    return decay * (1. - jnp.power(decay, t - 1.)) / (1. - jnp.power(decay, t))

  def UpdateMoments(self, step: JTensor, update: JTensor,
                    moments: _AdamOptState, beta1: float,
                    beta2: float) -> _AdamOptState:
    """Updates momentum values."""
    beta1_decay = self.BiasCorrectedDecay(step, beta1)
    beta2_decay = self.BiasCorrectedDecay(step, beta2)
    m = (1.0 - beta1_decay) * update + beta1_decay * moments.m
    v = (1.0 - beta2_decay) * (update**2) + beta2_decay * moments.v
    return _AdamOptState(m=m, v=v)

  def ClipUpdate(self, update: JTensor, clip_threshold: float) -> JTensor:
    mean_update = self.SanitizeValues(ReduceRms(update), 1.0)
    clip_threshold = jnp.array(clip_threshold, dtype=update.dtype)
    denom = jnp.maximum(1.0, mean_update / clip_threshold)
    return update / denom


def sharded_chain(
    *args: Union[optax.GradientTransformation, ShardedGradientTransformation]
) -> ShardedGradientTransformation:
  """Applies a list of (possibly sharded) chainable update transformations.

  Given a sequence of chainable transforms, `sharded_chain` returns an `init_fn`
  that constructs a `state` by concatenating the states of the individual
  transforms, and returns an `update_fn` which chains the update transformations
  feeding the appropriate state to each. In addition, it differs from the optax
  `chain` function, by also supporting ShardedGradientTransformation by chaining
  also the `init_partition_spec_fn`. If there are no
  ShardedGradientTransformations in the chain, the sharding specs will be
  None, meaning all the variables are replicated.

  Args:
    *args: a sequence of chainable GradientTransformations or
      ShardedGradientTransformations or a combination of both.

  Returns:
    A single chained ShardedGradientTransformation.
  """

  def init_fn(params):
    return tuple(fn.init(params) for fn in args)

  def update_fn(updates, state, params=None):
    if len(args) != len(state):
      raise ValueError('The number of updates and states has to be the same in '
                       'sharded chain.')

    new_state = []
    for s, fn in zip(state, args):
      updates, new_s = fn.update(updates, s, params)
      new_state.append(new_s)
    return updates, tuple(new_state)

  def init_partition_spec_fn(mdl_vars):
    partition_specs = []
    for fn in args:
      if isinstance(fn, ShardedGradientTransformation):
        nmap = fn.init_partition_spec(mdl_vars)
        partition_specs.append(nmap)
      else:
        # Replicate the states.
        partition_specs.append(None)
    return tuple(partition_specs)

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


def apply_l2_weight_decay(
    learning_rate_fn: optax.Schedule,
    l2_regularizer_weight: Optional[float] = 0.
) -> ShardedGradientTransformation:
  """Applies L2 weight decay.

  Args:
    learning_rate_fn: An optax schedule that infers the lr given the step.
    l2_regularizer_weight: Weight for L2 regularization.

  Returns:
    A ShardedGradientTransformation applying L2 weight decay.
  """

  def update_fn(updates, state, params):
    count = state.count
    lr_multiplier = learning_rate_fn(count)
    if l2_regularizer_weight:
      if params is None:
        raise ValueError('Params must not be empty when applying weight decay.')
      updates = jax.tree_multimap(
          lambda g, p: g - lr_multiplier * l2_regularizer_weight * p, updates,
          params)
    updated_state = NestedMap(count=count + 1)
    return updates, updated_state

  return ShardedGradientTransformation(
      init=count_init_fn,
      update=update_fn,
      init_partition_spec=count_init_partition_spec_fn)


def apply_l1_weight_decay(
    learning_rate_fn: optax.Schedule,
    l1_regularizer_weight: Optional[float] = 0.
) -> ShardedGradientTransformation:
  """Applies L1 weight decay.

  Args:
    learning_rate_fn: An optax schedule that infers the lr given the step.
    l1_regularizer_weight: Weight for L1 regularization.

  Returns:
    A ShardedGradientTransformation applying L1 weight decay.
  """

  def update_fn(updates, state, params):
    count = state.count
    lr_multiplier = learning_rate_fn(count)
    if l1_regularizer_weight:
      if params is None:
        raise ValueError('Params must not be empty when applying weight decay.')
      updates = jax.tree_multimap(
          lambda g, p: g - lr_multiplier * l1_regularizer_weight * jnp.sign(p),
          updates, params)
    updated_state = NestedMap(count=count + 1)
    return updates, updated_state

  return ShardedGradientTransformation(
      init=count_init_fn,
      update=update_fn,
      init_partition_spec=count_init_partition_spec_fn)


def sharded_adam(learning_rate_fn: optax.Schedule, beta1: float, beta2: float,
                 epsilon: float, epsilon_root: float, update_capping: float,
                 weight_decay: float) -> ShardedGradientTransformation:
  """Standard Adam optimizer that also supports sharding.

  This Adam optimizer supports optional update capping when update_capping is >
  0. Update capping can help stabilizing model learning, avoiding excessive
  updates when gradient variance estimate is stale (e.g. when data distribution
  suddenly shifts).

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    beta1: decay rate to track the first moment.
    beta2: decay rate to track the second moment.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
    epsilon_root: Small constant applied to the denominator inside of the square
      root to avoid dividing by zero when rescaling.
    update_capping: If > 0, cap mean update to at most this value.
    weight_decay: If > 0, weight decay to apply.

  Returns:
    A `ShardedGradientTransformation`.
  """
  helper = _ShardedAdamOptimizerHelper()

  def init_fn(mdl_vars):
    slot_vars = jax.tree_map(helper.InitOptState, mdl_vars)
    count = jnp.array(0, dtype=jnp.int32)
    return NestedMap(
        count=count,
        m=jax.tree_map(lambda x: x.m, slot_vars),
        v=jax.tree_map(lambda x: x.v, slot_vars))

  def init_partition_spec_fn(mdl_params):
    slot_vars = jax.tree_map(helper.OptStateShardingSpec, mdl_params)
    count = py_utils.WeightParams(
        shape=[], init=None, dtype=jnp.int32, collections=None)

    return NestedMap(
        count=count,
        m=jax.tree_map(lambda x: x.m, slot_vars),
        v=jax.tree_map(lambda x: x.v, slot_vars))

  def update_fn(updates, state, params=None):
    # Sanitize updates just in case.
    if weight_decay > 0:
      assert params is not None
    updates = jax.tree_multimap(helper.SanitizeValues, updates)
    count = state.count

    def _UpdateMomentum(g, m, v):
      return helper.UpdateMoments(count, g, _AdamOptState(m=m, v=v), beta1,
                                  beta2)

    updated_moments = jax.tree_multimap(_UpdateMomentum, updates, state.m,
                                        state.v)

    m = jax.tree_map(lambda x: x.m, updated_moments)
    v = jax.tree_map(lambda x: x.v, updated_moments)

    updates = jax.tree_multimap(
        lambda m, v: m / (jnp.sqrt(v + epsilon_root) + epsilon), m, v)

    if update_capping > 0:
      updates = jax.tree_map(lambda x: helper.ClipUpdate(x, update_capping),
                             updates)

    if weight_decay > 0:
      updates = jax.tree_multimap(lambda x, v: x + weight_decay * v, updates,
                                  params)

    step_size = -1.0 * learning_rate_fn(count)
    # Finally, fold in step size.
    updates = jax.tree_map(lambda x: step_size * x, updates)

    updated_states = NestedMap(count=count + 1, m=m, v=v)
    return updates, updated_states

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


# pylint: enable=invalid-name
class BaseOptimizer:
  """Base class for all optimizers."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Defines hyper-params for all optimizers."""
    p = InstantiableParams(cls)
    p.Define(
        'l2_regularizer_weight', None,
        'If not None, L2 regularization to apply to the model weights. '
        'Otherwise, disable L2 regularization.')
    p.Define(
        'l1_regularizer_weight', None,
        'If not None, L1 regularization to apply to the model weights. '
        'Otherwise, disable L1 regularization.')
    p.Define(
        'clip_gradient_norm_to_value', 0.0,
        'Clip gradient by global norm to this value. This is similar to '
        'the bahaviour of tf.clip_by_global_norm. If you are looking for '
        'tf.clip_by_norm refer to clip_gradient_single_norm_to_value. Note '
        'these are mutually exclusive.')
    p.Define(
        'clip_gradient_single_norm_to_value', 0.0,
        'Clip gradient by single tensor norm to this value. This is '
        'similar to the bahaviour of tf.clip_by_norm. Note this is mutually '
        'exclusive to using clip_gradient_norm_to_value.')
    p.Define('learning_rate', 0.0, 'learning rate to use.')
    p.Define('lr_schedule', None, 'Learning rate decay schedule.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    self._params = params.Copy()
    p = self._params
    self._lr_schedule = self._params.lr_schedule.Instantiate()
    # Should not mix L1 and L2 weight decay together.
    if p.l2_regularizer_weight and p.l1_regularizer_weight:
      raise ValueError('Should not mix L1 and L2 regularization together.')

  @property
  def params(self) -> InstantiableParams:
    return self._params

  def GetLearningRate(self, step_count: JTensor) -> JTensor:
    """Get the learning rate of this optimizer at a particular step."""
    return self._lr_schedule.Value(step_count) * self.params.learning_rate

  def GetGradTransformation(
      self
  ) -> Union[optax.GradientTransformation, ShardedGradientTransformation]:
    """Get the grad transformation corresponds to this optimizer config.

    This is the final gradient transformation that incorporates all
    transformations.

    Returns:
      an optax.GradientTransformation or ShardedGradientTransformation.
    """
    # TODO(yonghui): respect gradient clipping, etc transformations.
    p = self.params
    return sharded_chain(
        self._GetRawGradTransformation(self.GetLearningRate),
        apply_l1_weight_decay(
            self.GetLearningRate,
            l1_regularizer_weight=p.l1_regularizer_weight),
        apply_l2_weight_decay(
            self.GetLearningRate,
            l2_regularizer_weight=p.l2_regularizer_weight))

  def _GetRawGradTransformation(
      self, lr: optax.Schedule
  ) -> Union[optax.GradientTransformation, ShardedGradientTransformation]:
    """Get the raw optimizer transformation without taking into other ...

    transformations such l1/l2 regularization, gradient norm clipping, etc.

    Args:
      lr: an optax schedule.

    Returns:
      an optax.GradientTransformation or ShardedGradientTransformation.
    """
    raise NotImplementedError()


class SgdOptimizer(BaseOptimizer):
  """Canonical SGD optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'momentum', None,
        'Decay rate used by the momentum term. If set to None, momentum is not '
        'used.')
    p.Define('nesterov', False, 'Whether Nesterov momentum is used or not.')
    return p

  def _GetRawGradTransformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    p = self._params
    return optax.sgd(learning_rate=lr, momentum=p.momentum, nesterov=p.nesterov)


class ShardedSgdOptimizer(BaseOptimizer):
  """Sharded SGD optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'momentum', None,
        'Decay rate used by the momentum term. If set to None, momentum is not '
        'used.')
    p.Define('nesterov', False, 'Whether Nesterov momentum is used or not.')
    return p

  def _GetRawGradTransformation(
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    p = self._params
    return sharded_sgd(
        learning_rate_fn=lr, momentum=p.momentum, nesterov=p.nesterov)


class AdamOptimizer(BaseOptimizer):
  """Adam optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'beta1', 0.9,
        'Expenonential decay rate to track the first moment of past gradients.')
    p.Define(
        'beta2', 0.999,
        'Exponential decay rate to track the second moment of past gradients.')
    p.Define(
        'epsilon', 1e-6,
        'Small constant applied to the denominator outside of the square root '
        'to avoid dividing by zero when rescaling.')
    p.Define(
        'epsilon_root', 0.0,
        'Small constant applied to the denominator inside of the square root '
        'to avoid dividing by zero when rescaling.')
    # update clipping (as specified by clip_threshold) and weight decay features
    # are available only when sharded_adam is set to True.
    p.Define('clip_threshold', 1.0,
             'An optional float to clip raw adam updates to.')
    p.Define('weight_decay', 0.0, 'Decoupled weight decay to apply.')
    p.Define('sharded_adam', True, 'whether or not to use sharded_adam')
    return p

  @classmethod
  def ParamsA(cls) -> InstantiableParams:
    """Convenient method for a commonly used Adam config."""
    return cls.Params().Set(beta1=0.9, beta2=0.997, epsilon=1e-9)

  @classmethod
  def ParamsB(cls) -> InstantiableParams:
    """Convenient method for another commonly used Adam config."""
    return cls.Params().Set(beta1=0.9, beta2=0.98, epsilon=1e-9)

  def _GetRawGradTransformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    p = self._params
    if p.sharded_adam:
      logging.info('Using sharded_adam.')
      return sharded_adam(
          learning_rate_fn=lr,
          beta1=p.beta1,
          beta2=p.beta2,
          epsilon=p.epsilon,
          epsilon_root=p.epsilon_root,
          update_capping=p.clip_threshold,
          weight_decay=p.weight_decay)
    else:
      logging.info('Using optax.adam.')
      return optax.adam(
          learning_rate=lr,
          b1=p.beta1,
          b2=p.beta2,
          eps=p.epsilon,
          eps_root=p.epsilon_root)


class AdafactorOptimizer(BaseOptimizer):
  """Adafactor optimizer from Optax."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'min_dim_size_to_factor', 128,
        'Only factor the statistics if two array dimensions have at least '
        'this size.')
    p.Define('decay_rate', 0.8,
             'Controls second-moment exponential decay schedule.')
    p.Define(
        'decay_offset', 0.,
        'For finetuning, one may set this to the starting step number of the '
        'finetuning phase.')
    p.Define(
        'multiply_by_parameter_scale', True,
        'If True, then scale learning_rate by parameter norm. if False, '
        'provided learning_rate is absolute step size.')
    p.Define('clipping_threshold', 1.,
             'Optional value; if None, clipping disabled.')
    p.Define(
        'momentum', None,
        'Optional value between 0 and 1, enables momentum and uses extra '
        'memory if non-None! None by default.')
    p.Define('dtype_momentum', 'float32', 'dtype of momentum buffers.')
    p.Define('weight_decay_rate', None,
             'Optional rate at which to decay weights.')
    p.Define('eps', 1e-30,
             'Regularization constant for root mean squared gradient.')
    p.Define('factored', True,
             'Whether to use factored second-moment estimates.')
    return p

  def _GetRawGradTransformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    p = self._params
    return optax.adafactor(
        learning_rate=lr,
        min_dim_size_to_factor=p.min_dim_size_to_factor,
        decay_rate=p.decay_rate,
        decay_offset=p.decay_offset,
        multiply_by_parameter_scale=p.multiply_by_parameter_scale,
        clipping_threshold=p.clipping_threshold,
        momentum=p.momentum,
        dtype_momentum=getattr(jnp, p.dtype_momentum),
        weight_decay_rate=p.weight_decay_rate,
        eps=p.eps,
        factored=p.factored)


def ToQuantized(fvalue: JTensor,
                quantized_dtype: jnp.dtype) -> Tuple[JTensor, JTensor]:
  """Converts floating point values `fvalues` to quantized values.

  We use a very simple quantization scheme where the range is symmetric around
  0.0, and we simply map 0 to 0.0.

  Let x = bucket_size
  We map [-0.5x, 0.5x] to 0
         [-1.5x, -0.5x] to -1
         [0.5x, 1.5x] to 1
         and so on so forth.

  Some properties:
    a1, a2 = ToQuantized(x, quantized_dtype)
    b1 = ToFloat(a1, a2)
    c1, c2 = ToQuantized(b1, quantized_dtype)

    then a1 == c1, a2 == c2

  Args:
    fvalue: Values in floating point.
    quantized_dtype: Quantized dtype, can be either jnp.int8, or jnp.int16.

  Returns:
    A (quantized_values, bucket_size) 2-tuple.
    `quantized_values * bucket_size[jnp.newaxis, ...]` are the quantized
    values
    on the floating value axis.
  """
  float_dtype = fvalue.dtype
  if quantized_dtype == jnp.int8:
    # value -128 is not used.
    num_buckets = jnp.array(127.0, dtype=float_dtype)
  elif quantized_dtype == jnp.int16:
    # value -32768 is not used.
    num_buckets = jnp.array(32767.0, dtype=float_dtype)
  else:
    raise ValueError(f'Quantized dtype {quantized_dtype} not supported.')
  # max value is mapped to num_buckets

  # We first decide the scale.
  if fvalue.ndim < 1:
    raise ValueError(
        f'Input array {fvalue} must have a strictly positive number of '
        'dimensions.')

  max_abs = jnp.max(jnp.abs(fvalue), axis=0)
  bucket_size = max_abs / num_buckets
  bs_expanded = bucket_size[jnp.newaxis, ...]
  # To avoid divide by 0.0
  bs_nonzero = jnp.where(bs_expanded > 0.0, bs_expanded,
                         jnp.ones_like(bs_expanded))
  ratio = fvalue / bs_nonzero
  # We use rounding to remove bias.
  quantized = jnp.round(ratio)
  return quantized.astype(quantized_dtype), bucket_size


def ToFloat(quantized: JTensor, bucket_size: JTensor) -> JTensor:
  """Converts quantized values to float values.

  Args:
    quantized: Quantized values, of type either jnp.int8 or jnp.int16.
    bucket_size: The size of each bucket on the floating-point axis. bucket_size
      is of rank tf.rank(quantized) - 1. For example, if quantized is of shape
      [x, ...], bucket_size is of shape [...].

  Returns:
    Unquantized values of type bucket_size.dtype.
  """
  float_dtype = bucket_size.dtype
  bucket_size = bucket_size[jnp.newaxis, ...]
  return quantized.astype(float_dtype) * bucket_size


def AdafactorDecayRateAdam(beta2: float, step_counter: JTensor) -> JTensor:
  """Second-moment decay rate like Adam, subsuming the correction factor.

  Args:
    beta2: A floating point value between 0 and 1.
    step_counter: A scalar tensor keeping track of the number of steps
      performed.

  Returns:
    The decay rate as a scalar JTensor.
  """
  step = step_counter
  beta2 = jnp.array(beta2, dtype=jnp.float32)
  t = step + 1.
  return beta2 * (1. - jnp.power(beta2, t - 1.)) / (1. - jnp.power(beta2, t))


def AdafactorDecayRatePow(exponent: float, step_counter: JTensor) -> JTensor:
  """Second moment decay rate where memory-length grows as step_num^exponent.

  Args:
    exponent: A floating point value between 0 and 1.
    step_counter: A scalar tensor keeping track of the number of steps
      performed.

  Returns:
    The decay rate as a scalar JTensor.
  """
  step = step_counter
  exponent = jnp.array(exponent, dtype=jnp.float32)
  return 1. - jnp.power((step + 1.), -exponent)


def ReduceMean(array: JTensor) -> JTensor:
  """Computes the mean of `array` in a more numerically stable way.

  Args:
    array: Input array.

  Returns:
    The mean of the input array as a scalar array.
  """
  num_elements = array.size
  if num_elements > 1e8:
    # When x is too large, simple jnp.mean() can result in nan or inf values.
    # TODO(bf-jax): The following code snippet is consistent with the TensorFlow
    # implementation. This can be simplified into `jnp.mean(jnp.mean(x, -1))`.
    # Update to using mean() after verifying consistency.
    array_sum = jnp.sum(array, axis=-1)
    array_sum = jnp.sum(array_sum)
    return array_sum / jnp.array(num_elements, dtype=array_sum.dtype)
  else:
    return jnp.mean(array)


def ReduceRms(array: JTensor) -> JTensor:
  """Computes the RMS of `array` (in a numerically stable way).

  Args:
    array: Input array.

  Returns:
    The root mean square of the input array as a scalar array.
  """
  sq = jnp.square(array)
  sq_mean = ReduceMean(sq)
  return jnp.sqrt(sq_mean)


@dataclasses.dataclass(frozen=True)
class _ShardedAdafactorUpdateResult:
  """Structure containing per-variable info for Adafactor."""
  update: Optional[Any]
  m: Optional[Any]
  m_scale: Optional[Any]
  vr: Optional[Any]
  vc: Optional[Any]
  v: Optional[Any]


class ShardedAdafactorState(optax.OptState):
  """Overall state of the ShardedAdafactor optimizer."""
  count: JTensor
  m: Optional[NestedJTensor]
  m_scale: Optional[NestedJTensor]
  vr: Optional[NestedJTensor]
  vc: Optional[NestedJTensor]
  v: Optional[NestedJTensor]


class ShardedAdafactorHelper:
  """Helper class to implement optax-based sharded Adafactor."""

  def __init__(
      self,
      learning_rate_fn: optax.Schedule,
      weight_decay: Optional[float],
      layerwise_adaptation: bool,
      decay_method: str,
      decay_adam: float,
      decay_pow: float,
      beta1: float,
      clipping_threshold: float,
      factored: bool,
      epsilon1: float,
      quantized_dtype: jnp.dtype,
      # TODO(bf-jax) Update default value to True, once this is supported.
      respect_skip_lp_regularization: bool,
  ) -> None:
    """Constructor. See ShardedAdafactor() below."""
    self._learning_rate_fn = learning_rate_fn
    self._weight_decay = weight_decay
    self._layerwise_adaptation = layerwise_adaptation
    self._decay_method = decay_method
    self._decay_adam = decay_adam
    self._decay_pow = decay_pow
    self._beta1 = beta1
    self._clipping_threshold = clipping_threshold
    self._factored = factored
    self._epsilon1 = epsilon1
    self._quantized_dtype = quantized_dtype
    self._respect_skip_lp_regularization = respect_skip_lp_regularization

  def ShouldUseFactoredSecondMomentEstimate(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.

    Args:
      shape: a list of integers.

    Returns:
      A boolean.
    """
    return self._factored and len(shape) >= 2

  def ShouldStoreMomentumInQint(self, shape):
    """Should we store momentum as quantized integers.

    Based on the shape of the variable.

    Args:
      shape: a list of integers

    Returns:
      A boolean.
    """
    return len(shape) >= 1

  def ToState(self, count, result_tree):
    """Maps from a tree of (factored) values to separate trees of values."""
    return ShardedAdafactorState(
        count=count,
        m=jax.tree_map(lambda o: o.m, result_tree),
        m_scale=jax.tree_map(lambda o: o.m_scale, result_tree),
        vr=jax.tree_map(lambda o: o.vr, result_tree),
        vc=jax.tree_map(lambda o: o.vc, result_tree),
        v=jax.tree_map(lambda o: o.v, result_tree))

  def Init(self, param):
    """Initializes the optimizer state for a given param."""
    # The actually value that will be added to a variable for updating it.
    output_update = jnp.zeros((1,))
    output_m = jnp.zeros((1,))
    output_m_scale = jnp.zeros((1,))
    output_vr = jnp.zeros((1,))
    output_vc = jnp.zeros((1,))
    output_v = jnp.zeros((1,))
    shape = param.shape
    if self._beta1:
      if self._quantized_dtype == jnp.bfloat16:
        output_m = jnp.zeros(shape, dtype=jnp.bfloat16)
      elif self.ShouldStoreMomentumInQint(shape):
        output_m = jnp.zeros(shape, dtype=self._quantized_dtype)
        scale_shape = shape[1:]
        output_m_scale = jnp.zeros(scale_shape, dtype=jnp.float32)
      else:
        output_m = jnp.zeros(shape, dtype=jnp.float32)
    if self.ShouldUseFactoredSecondMomentEstimate(shape):
      output_vr = jnp.zeros(shape[:-1], dtype=jnp.float32)
      output_vc = jnp.zeros(shape[:-2] + shape[-1:], dtype=jnp.float32)
    else:
      output_v = jnp.zeros(shape, dtype=jnp.float32)
    return _ShardedAdafactorUpdateResult(
        update=output_update,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v)

  def InitPartitionSpec(self, var_param):
    """Initializes the partition spec for a given param."""
    output_update = py_utils.WeightParams((1,))
    output_m = py_utils.WeightParams((1,))
    output_m_scale = py_utils.WeightParams((1,))
    output_vr = py_utils.WeightParams((1,))
    output_vc = py_utils.WeightParams((1,))
    output_v = py_utils.WeightParams((1,))
    shape = var_param.shape
    tensor_split_dims_mapping = var_param.tensor_split_dims_mapping

    if var_param.repeat_prefix is not None:
      prefix_shape = var_param.repeat_prefix
      prefix_sharding = var_param.repeat_prefix_split_dims_mapping
      if prefix_sharding is None:
        prefix_sharding = [-1] * len(prefix_shape)
      shape = tuple(prefix_shape) + tuple(shape)
      if tensor_split_dims_mapping is not None:
        tensor_split_dims_mapping = (
            tuple(prefix_sharding) + tuple(tensor_split_dims_mapping))

    if tensor_split_dims_mapping is not None:
      assert len(tensor_split_dims_mapping) == len(shape)
      sharding_specified = True
    else:
      sharding_specified = False

    # TODO(yonghui): Fix me. For stacked weight (which is a stack of multiple
    # logical weights), we should be performing data aggregation on the right
    # axis, not always on the first two.
    if self._beta1:
      if self._quantized_dtype == jnp.bfloat16:
        output_m = py_utils.WeightParams(
            shape=shape,
            init=None,
            dtype=jnp.bfloat16,
            collections=None,
            device_mesh=var_param.device_mesh,
            tensor_split_dims_mapping=tensor_split_dims_mapping)
      elif self.ShouldStoreMomentumInQint(shape):
        output_m = py_utils.WeightParams(
            shape=shape,
            init=None,
            dtype=self._quantized_dtype,
            collections=None,
            device_mesh=var_param.device_mesh,
            tensor_split_dims_mapping=tensor_split_dims_mapping)
        scale_shape = shape[1:]
        m_scale_split_dims_mapping = tensor_split_dims_mapping
        # TODO(shafey): Fix logic for updating sharding annotations.
        if sharding_specified:
          m_scale_split_dims_mapping = gshard_utils.RemoveDim(
              0, tensor_split_dims_mapping)
        output_m_scale = py_utils.WeightParams(
            shape=scale_shape,
            init=None,
            dtype=jnp.float32,
            collections=None,
            device_mesh=var_param.device_mesh,
            tensor_split_dims_mapping=m_scale_split_dims_mapping)
      else:
        output_m = py_utils.WeightParams(
            shape=shape,
            init=None,
            dtype=jnp.float32,
            collections=None,
            device_mesh=var_param.device_mesh,
            tensor_split_dims_mapping=tensor_split_dims_mapping)
    if self.ShouldUseFactoredSecondMomentEstimate(shape):
      # TODO(shafey): Fix logic for updating sharding annotations.
      if sharding_specified:
        vr_split_dims_mapping = gshard_utils.RemoveDim(
            -1, tensor_split_dims_mapping)
        vc_split_dims_mapping = gshard_utils.RemoveDim(
            -2, tensor_split_dims_mapping)
      else:
        vr_split_dims_mapping = tensor_split_dims_mapping
        vc_split_dims_mapping = tensor_split_dims_mapping
      output_vr = py_utils.WeightParams(
          shape[:-1],
          init=None,
          dtype=jnp.float32,
          collections=None,
          device_mesh=var_param.device_mesh,
          tensor_split_dims_mapping=vr_split_dims_mapping)
      output_vc = py_utils.WeightParams(
          shape[:-2] + shape[-1:],
          init=None,
          dtype=jnp.float32,
          collections=None,
          device_mesh=var_param.device_mesh,
          tensor_split_dims_mapping=vc_split_dims_mapping)
    else:
      output_v = py_utils.WeightParams(
          shape=shape,
          init=None,
          dtype=var_param.dtype,
          collections=None,
          device_mesh=var_param.device_mesh,
          tensor_split_dims_mapping=tensor_split_dims_mapping)
    return _ShardedAdafactorUpdateResult(
        update=output_update,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v)

  def SanitizeValues(self, array, replacement=0.):
    """Sanitizes NaN and Infinity values."""
    return jnp.nan_to_num(
        array, nan=replacement, posinf=replacement, neginf=replacement)

  def ComputeVarAndSlotUpdate(self, count, grad, m, m_scale, vr, vc, v, param):
    """Computes the var and optimizer slots updates for a single variable."""
    # We can probably skip this step
    grad = self.SanitizeValues(grad)
    grad = grad.astype(jnp.float32)
    # Add epsilon1 as per Algorithm 4 of https://arxiv.org/pdf/1804.04235.pdf
    grad_squared = jnp.square(grad) + self._epsilon1
    grad_squared_mean = self.SanitizeValues(ReduceMean(grad_squared))
    if self._decay_method == 'adam':
      assert self._decay_adam > 0
      decay_rate = AdafactorDecayRateAdam(self._decay_adam, count)
    elif self._decay_method == 'pow':
      assert self._decay_pow > 0
      decay_rate = AdafactorDecayRatePow(self._decay_pow, count)
    else:
      raise ValueError(f'decay_method {self._decay_method} not supported.')

    learning_rate = self._learning_rate_fn(count)

    update_scale = learning_rate
    old_val = param
    # Q(yonghui): Can we remove the hack now?
    # HACK: Make things dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    decay_rate += grad_squared_mean * 1e-30
    update_scale += grad_squared_mean * 1e-30
    # END HACK
    mixing_rate = 1. - decay_rate
    shape = param.shape

    output_m = jnp.zeros((1,))
    output_m_scale = jnp.zeros((1,))
    output_vr = jnp.zeros((1,))
    output_vc = jnp.zeros((1,))
    output_v = jnp.zeros((1,))

    if self.ShouldUseFactoredSecondMomentEstimate(shape):
      # Q(shafey): Should we use the more numerically stable version
      # ReduceMean().
      grad_squared_row_mean = self.SanitizeValues(
          jnp.mean(grad_squared, axis=-1))
      grad_squared_col_mean = self.SanitizeValues(
          jnp.mean(grad_squared, axis=-2))
      new_vr = decay_rate * vr + mixing_rate * grad_squared_row_mean
      new_vc = decay_rate * vc + mixing_rate * grad_squared_col_mean
      output_vr = new_vr
      output_vc = new_vc
      long_term_mean = jnp.mean(new_vr, axis=-1, keepdims=True)
      r_factor = 1. / jnp.sqrt(new_vr / long_term_mean)
      c_factor = 1. / jnp.sqrt(new_vc)
      x = grad * jnp.expand_dims(r_factor, -1) * jnp.expand_dims(c_factor, -2)
    else:
      # v with sharding annotation.
      new_v = decay_rate * v + mixing_rate * grad_squared
      output_v = new_v
      x = grad / jnp.sqrt(new_v)
    if self._clipping_threshold is not None:
      clipping_denom = jnp.maximum(1., ReduceRms(x) / self._clipping_threshold)
      clipping_denom = self.SanitizeValues(clipping_denom, replacement=1.)
      x /= clipping_denom
    subtrahend = update_scale * x
    if self._beta1:
      if self._quantized_dtype == jnp.bfloat16:
        m = m.astype(jnp.float32)
      elif self.ShouldStoreMomentumInQint(shape):
        m_init_dtype = m.dtype
        m = ToFloat(m, m_scale)
      subtrahend = self._beta1 * m + (1. - self._beta1) * subtrahend
      subtrahend = self.SanitizeValues(subtrahend)
      if self._quantized_dtype == jnp.bfloat16:
        new_m = subtrahend.astype(jnp.bfloat16)
        output_m = new_m
      elif self.ShouldStoreMomentumInQint(shape):
        # Update the momentum values.
        new_m_val, new_m_scale = ToQuantized(subtrahend, m_init_dtype)
        output_m = new_m_val
        output_m_scale = new_m_scale
      else:
        output_m = subtrahend

    if self._weight_decay is not None:
      # Apply decoupled weight decay to be consistent with AdamW.
      weight_decay = self._weight_decay * learning_rate
      # TODO(bf-jax): Add support for skip weight decay
      subtrahend += weight_decay * old_val

    # TODO(bf-jax): Add support for layerwise adaptation

    return _ShardedAdafactorUpdateResult(
        update=-subtrahend,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v)


def ShardedAdafactor(
    learning_rate_fn: optax.Schedule,
    weight_decay: Optional[float] = None,
    layerwise_adaptation: bool = False,
    decay_method: str = '',
    decay_adam: float = 0.,
    decay_pow: float = 0.,
    beta1: float = 0.,
    clipping_threshold: float = 1.,
    factored: bool = True,
    epsilon1: float = 1e-30,
    quantized_dtype: jnp.dtype = jnp.int8,
    # TODO(bf-jax) Update default value to True, once this is supported.
    respect_skip_lp_regularization: bool = False
) -> ShardedGradientTransformation:
  """AdaFactor optimizer that supports SPMD sharding.

  Reference:
    Shazeer et al, 2018: https://arxiv.org/abs/1804.04235

  Adafactor is very similar to Adam (Kingma and Ba, 2019), the major
  differences being:

  1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
     parameters to maintain the second-moment estimator, instead of AB.
     This is advantageous on memory-limited systems.  In addition, beta1
     (momentum) is set to zero by default, saving an additional auxiliary
     parameter per weight.  Variables with >=3 dimensions are treated as
     collections of two-dimensional matrices - factorization is over the final
     two dimensions.

  2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
     gradient clipping.  This improves stability.

  3. Adafactor does not require an external "learning rate".  By default, it
     incorporates a relative-update-scale schedule, corresponding to
     inverse-square-root learning-rate-decay in Adam.  We hope this works well
     for most applications.

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    weight_decay: an optional float tensor as decoupled weight decay value.
    layerwise_adaptation: a boolean, whether or not to use layer-wise
      adaptive moments (LAMB): https://arxiv.org/abs/1904.00962.
    decay_method: a string, deciding how decay_rate should be computed.
      Permitted values are 'adam' and 'pow'.
    decay_adam: a float, decay if decay_method == 'adam'.
    decay_pow: a float, decay if decay_method == 'pow'.
    beta1: a float value between 0 and 1 for momentum.
    clipping_threshold: an optional float >= 1
    factored: a boolean, whether or not to use factored second order momentum.
    epsilon1: Regularization constant for squared gradient.
    quantized_dtype: type of the quantized input. Allowed options are jnp.int8,
      jnp.int16, and jnp.bfloat16. If jnp.bfloat16 is specified, accumulators
      are stored as bfloat16, instead of quantized integers.
    respect_skip_lp_regularization: whether or not to respect lingvo
      SKIP_LP_REGULARIZATION var collection that skips decoupled weight decay.

  Returns:
    A `ShardedGradientTransformation`.
  """
  # TODO(bf-jax): layerwise adaptation and skip regularization.
  assert not layerwise_adaptation
  assert not respect_skip_lp_regularization
  assert decay_adam >= 0
  assert decay_pow >= 0
  assert learning_rate_fn is not None
  assert decay_method == 'adam' or decay_method == 'pow', (
      f'decay_method: {decay_method} not supported. Supported methods are '
      '"pow", or "adam".')

  sharded_adafactor_helper = ShardedAdafactorHelper(
      learning_rate_fn=learning_rate_fn,
      weight_decay=weight_decay,
      layerwise_adaptation=layerwise_adaptation,
      decay_method=decay_method,
      decay_adam=decay_adam,
      decay_pow=decay_pow,
      beta1=beta1,
      clipping_threshold=clipping_threshold,
      factored=factored,
      epsilon1=epsilon1,
      quantized_dtype=quantized_dtype,
      respect_skip_lp_regularization=respect_skip_lp_regularization)

  def init_fn(params):  # pylint: disable=invalid-name
    """Initializes the optimizer's state."""
    return sharded_adafactor_helper.ToState(
        jnp.zeros([], jnp.int32),
        jax.tree_map(sharded_adafactor_helper.Init, params))

  def init_partition_spec_fn(var_params):  # pylint: disable=invalid-name
    var_spec_flattened, _ = jax.tree_flatten(var_params)
    assert var_spec_flattened
    first_var = var_spec_flattened[0]
    assert isinstance(first_var, py_utils.Params)
    device_mesh = first_var.device_mesh
    count = py_utils.WeightParams(
        shape=[],
        init=None,
        dtype=jnp.int32,
        collections=None,
        device_mesh=device_mesh,
        tensor_split_dims_mapping=[])
    return sharded_adafactor_helper.ToState(
        count,
        jax.tree_map(sharded_adafactor_helper.InitPartitionSpec, var_params))

  def update_fn(updates, state, params=None):  # pylint: disable=invalid-name
    if params is None:
      raise ValueError(
          'You are using a transformation that requires the current value of '
          'parameters, but you are not passing `params` when calling `update`.')

    compute_var_and_slot_update_fn = functools.partial(
        sharded_adafactor_helper.ComputeVarAndSlotUpdate, state.count)
    output = jax.tree_multimap(compute_var_and_slot_update_fn, updates, state.m,
                               state.m_scale, state.vr, state.vc, state.v,
                               params)
    updates = jax.tree_map(lambda o: o.update, output)
    count_plus_one = state.count + jnp.array(1, jnp.int32)
    updated_states = sharded_adafactor_helper.ToState(count_plus_one, output)
    return updates, updated_states

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


class ShardedAdafactorOptimizer(BaseOptimizer):
  """Sharded AdaFactor optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('weight_decay', None,
             'An optional float tensor as decoupled weight decay value.')
    p.Define(
        'layerwise_adaptation', False,
        'A boolean, whether or not to use layer-wise adaptive moments (LAMB): '
        'https://arxiv.org/abs/1904.00962.')
    p.Define(
        'decay_method', '',
        'A string, deciding how decay_rate should be computed. Permitted '
        'values are `adam` and `pow`.')
    p.Define('decay_adam', 0., 'A float, decay if decay_method == `adam`.')
    p.Define('decay_pow', 0., 'A float, decay if decay_method == `pow`.')
    p.Define('beta1', 0., 'A float value between 0 and 1 for the momentum.')
    p.Define('clipping_threshold', 1., 'An optional float >= 1.')
    p.Define(
        'factored', True,
        'A boolean, whether or not to use factored second order momentum.')
    p.Define('epsilon1', 1e-30, 'Regularization constant for squared gradient.')
    p.Define(
        'quantized_dtype', 'int8',
        'Type of the quantized input. Allowed options are jnp.int8, jnp.int16, '
        'and jnp.bfloat16. If jnp.bfloat16 is specified, accumulators are '
        'stored as bfloat16, instead of quantized integers.')
    p.Define(
        # TODO(bf-jax) Update default value to True, once this is supported.
        'respect_skip_lp_regularization',
        False,
        'Whether or not to respect lingvo SKIP_LP_REGULARIZATION var '
        'collection that skips decoupled weight decay.')
    return p

  def _GetRawGradTransformation(
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    p = self._params
    return ShardedAdafactor(
        learning_rate_fn=lr,
        weight_decay=p.weight_decay,
        layerwise_adaptation=p.layerwise_adaptation,
        decay_method=p.decay_method,
        decay_adam=p.decay_adam,
        decay_pow=p.decay_pow,
        beta1=p.beta1,
        clipping_threshold=p.clipping_threshold,
        factored=p.factored,
        epsilon1=p.epsilon1,
        quantized_dtype=getattr(jnp, p.quantized_dtype),
        respect_skip_lp_regularization=p.respect_skip_lp_regularization)
