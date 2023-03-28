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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express o.r implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for all optimizers."""

import dataclasses
import functools
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax import lax
from jax import numpy as jnp
from lingvo.jax import asserts
from lingvo.jax import base_layer
from lingvo.jax import gshard_utils
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import optax

from optax_shampoo import distributed_shampoo

# DistributedShampoo types
distributed_shampoo_optimizer = distributed_shampoo.distributed_shampoo
Preconditioner = distributed_shampoo.Preconditioner
QuantizedValue = distributed_shampoo.QuantizedValue
GraftingType = distributed_shampoo.GraftingType
ShardedShampooStats = distributed_shampoo.ShardedShampooStats
ShampooState = distributed_shampoo.ShampooState
LocalShardedParameterStats = distributed_shampoo.LocalShardedParameterStats
GlobalShardedParameterStats = distributed_shampoo.GlobalShardedParameterStats

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


GeneralGradientTransformation = Union[optax.GradientTransformation,
                                      ShardedGradientTransformation]


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
      count=py_utils.weight_params(
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


class _ShardedAdamHelper:
  """A helper class facilitates the creation of sharded_adam_optimizer."""

  def opt_state_sharding_spec(self,
                              var_params: py_utils.Params) -> _AdamOptState:
    """Returns optimizer sharding spec for one particular variable."""
    m_var_params = var_params.Copy()
    m_var_params.init = None
    v_var_params = var_params.Copy()
    v_var_params.init = None
    # m and v simply share the same sharding.
    return _AdamOptState(m=m_var_params, v=v_var_params)

  def init_opt_state(self, var_params: py_utils.Params) -> _AdamOptState:
    """Returns optimizer state for one particular variable."""
    return _AdamOptState(
        m=jnp.zeros_like(var_params), v=jnp.zeros_like(var_params))

  def sanitize_values(self, array: JTensor, replacement: float = 0.0):
    """Sanitizes NaN and Infinity values."""
    return jnp.nan_to_num(
        array, nan=replacement, posinf=replacement, neginf=replacement)

  def bias_corrected_decay(self, step: JTensor, decay: float) -> JTensor:
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

  def update_moments(self, step: JTensor, update: JTensor,
                     moments: _AdamOptState, beta1: float,
                     beta2: float) -> _AdamOptState:
    """Updates momentum values."""
    beta1_decay = self.bias_corrected_decay(step, beta1)
    beta2_decay = self.bias_corrected_decay(step, beta2)
    m = (1.0 - beta1_decay) * update + beta1_decay * moments.m
    v = (1.0 - beta2_decay) * (update**2) + beta2_decay * moments.v
    return _AdamOptState(m=m, v=v)

  def clip_update(self, update: JTensor, clip_threshold: float) -> JTensor:
    mean_update = self.sanitize_values(reduce_rms(update), 1.0)
    clip_threshold = jnp.array(clip_threshold, dtype=update.dtype)
    denom = jnp.maximum(1.0, mean_update / clip_threshold)
    return update / denom


def sharded_chain(
    *args: GeneralGradientTransformation) -> ShardedGradientTransformation:
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
      init_partition_spec = getattr(fn, 'init_partition_spec', None)
      if callable(init_partition_spec):
        nmap = init_partition_spec(mdl_vars)
        partition_specs.append(nmap)
      else:
        # Replicate the states.
        partition_specs.append(None)
    return tuple(partition_specs)

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


def sharded_masked(
    inner: GeneralGradientTransformation, mask: Union[NestedParams,
                                                      Callable[[NestedParams],
                                                               NestedParams]]
) -> GeneralGradientTransformation:
  """Mask updates so only some are transformed, the rest are passed through.

  This differs from the Optax version in that it supports sharding annotations.

  Args:
    inner: Inner transformation to mask.
    mask: a PyTree with same structure as (or a prefix of) the params PyTree, or
      a Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, ``True`` for leaves/subtrees you want to apply the
      transformation to, and ``False`` for those you want to skip.

  Returns:
    New ShardedGradientTransformation wrapping ``inner``.
  """

  def init_partition_spec_fn(mdl_vars):
    init_partition_spec = getattr(inner, 'init_partition_spec', None)
    if callable(init_partition_spec):
      return init_partition_spec(mdl_vars)

  grad_tx = optax.masked(inner, mask)
  if not hasattr(inner, 'init_partition_spec'):
    return grad_tx
  else:
    return ShardedGradientTransformation(
        init=grad_tx.init,
        update=grad_tx.update,
        init_partition_spec=init_partition_spec_fn)


def apply_lp_regularizer(
    learning_rate_fn: optax.Schedule,
    regularizer_weight: Optional[float] = 0.0,
    p: Optional[float] = 2.0,
    skip_lp_1d_vectors: Optional[bool] = False,
) -> ShardedGradientTransformation:
  """Applies Lp regularization by adjusting gradients.

  Note, lp regularizers add loss to final loss objective, while decoupled
  weight decay adds decay directly into weights. They are different especially
  when there are moment statistics in optimizers. A good reference can be found
  in: https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw

  Args:
    learning_rate_fn: An optax schedule that infers the lr given the step.
    regularizer_weight: Weight for L2 regularization.
    p: 1 or 2 as L1/L2 regularization.
    skip_lp_1d_vectors: If True, skip L1/L2 regularization for 1d vector vars.

  Returns:
    A ShardedGradientTransformation applying Lp regularizers.
  """
  # Adjust raw gradients directly.
  del learning_rate_fn

  asserts.in_set(p, [1.0, 2.0])

  # TODO(aurkor, yonghui): we need respect SKIP_LP_REGULARIZATION var collection
  # by propagating var names into ShardedGradientTransformation. Right now
  # disable all 1d vars from lp regularizers if skip_lp_1d_vectors is True.
  def skip_mask(var):
    if skip_lp_1d_vectors and var.ndim <= 1:
      return 0.0
    else:
      return 1.0

  def update_fn(updates, state, params):
    count = state.count
    if regularizer_weight:
      if params is None:
        raise ValueError('Params must not be empty when applying weight decay.')

      if p == 1.0:
        fn = lambda g, p: g + regularizer_weight * jnp.sign(p) * skip_mask(p)
      elif p == 2.0:
        fn = lambda g, p: g + regularizer_weight * p * skip_mask(p)

      updates = jax.tree_map(fn, updates, params)
    updated_state = NestedMap(count=count + 1)
    return updates, updated_state

  return ShardedGradientTransformation(
      init=count_init_fn,
      update=update_fn,
      init_partition_spec=count_init_partition_spec_fn)


def apply_decoupled_weight_decay(
    learning_rate_fn: optax.Schedule,
    regularizer_weight: Optional[float] = 0.0,
) -> ShardedGradientTransformation:
  """Applies decoupled weight decay on weights.

  Note, lp regularizers add loss to final loss objective, while decoupled
  weight decay adds decay directly into weights. They are different especially
  when there are moment statistics in optimizers. A good reference can be found
  in: https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw

  Args:
    learning_rate_fn: An optax schedule that infers the lr given the step.
    regularizer_weight: Weight for decoupled weight decay.

  Returns:
    A ShardedGradientTransformation applying weight decay.
  """

  # TODO(aurkor, yonghui): we need respect SKIP_LP_REGULARIZATION var collection
  # by propagating var names into ShardedGradientTransformation.

  def update_fn(updates, state, params):
    count = state.count
    lr = learning_rate_fn(count)
    if regularizer_weight:
      if params is None:
        raise ValueError('Params must not be empty when applying weight decay.')

      fn = lambda g, p: g - lr * regularizer_weight * p

      updates = jax.tree_map(fn, updates, params)
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
  helper = _ShardedAdamHelper()

  def init_fn(mdl_vars):
    slot_vars = jax.tree_map(helper.init_opt_state, mdl_vars)
    count = jnp.array(0, dtype=jnp.int32)
    return NestedMap(
        count=count,
        m=jax.tree_map(lambda x: x.m, slot_vars),
        v=jax.tree_map(lambda x: x.v, slot_vars))

  def init_partition_spec_fn(mdl_params):
    slot_vars = jax.tree_map(helper.opt_state_sharding_spec, mdl_params)
    count = py_utils.weight_params(
        shape=[], init=None, dtype=jnp.int32, collections=None)

    return NestedMap(
        count=count,
        m=jax.tree_map(lambda x: x.m, slot_vars),
        v=jax.tree_map(lambda x: x.v, slot_vars))

  def update_fn(updates, state, params=None):
    # Sanitize updates just in case.
    if weight_decay > 0:
      assert params is not None
    updates = jax.tree_map(helper.sanitize_values, updates)
    count = state.count

    def _update_momentum(g, m, v):
      return helper.update_moments(count, g, _AdamOptState(m=m, v=v), beta1,
                                   beta2)

    updated_moments = jax.tree_map(_update_momentum, updates, state.m,
                                        state.v)

    m = jax.tree_map(lambda x: x.m, updated_moments)
    v = jax.tree_map(lambda x: x.v, updated_moments)

    updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v + epsilon_root) + epsilon), m, v)

    if update_capping > 0:
      updates = jax.tree_map(lambda x: helper.clip_update(x, update_capping),
                             updates)

    if weight_decay > 0:
      updates = jax.tree_map(lambda x, v: x + weight_decay * v, updates,
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


def apply_ema_weights(decay: float,
                      debias: bool) -> optax.GradientTransformation:
  """Applies exponential moving average on weights.

  Note, this implementation averages the weight before optimization because
  trainable and non-trainable variables are handled separately. In such case
  the updates on non-trainable variables like bn stats are not available in
  updates.

  This differs from optax.ema which applies ema on gradients so it changes
  training process.

  ema = ema * decay + new_weight * (1.0 - decay)
  debias reduces the bias from the initialization as introduced in Section
  3 of https://arxiv.org/pdf/1412.6980.pdf

  Args:
    decay: A float number represents the weight on the moving average.
    debias: whether reduces the bias on the zero initialization

  Returns:
    A GradientTransformation applying ema.
  """

  def init_fn(params):
    return NestedMap(
        count=jnp.array(0, dtype=jnp.int32),
        ema=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError('Params required for the EMA')
    new_ema = jax.tree_map(
        lambda old_v, new_v: (1.0 - decay) * new_v + decay * old_v, state.ema,
        params)
    count_inc = state.count + jnp.array(1, jnp.int32)

    if debias:
      bias_correction = 1 - decay**count_inc
      new_ema = jax.tree_map(lambda v: v / (1e-8 + bias_correction), new_ema)
    return updates, NestedMap(count=count_inc, ema=new_ema)

  return optax.GradientTransformation(init=init_fn, update=update_fn)


class BaseOptimizer:
  """Base class for all optimizers."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
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
    # TODO(aurkor, yonghui): remove skip_lp_1d_vectors once we respect
    # SKIP_LP_REGULARIZATION var collection by propagating var names into
    # ShardedGradientTransformation.
    p.Define('skip_lp_1d_vectors', False,
             'If True, skip L1/L2 regularization for 1d vector vars.')
    p.Define(
        'decoupled_weight_decay', None,
        'If not None, (decoupled) weight decay to apply to the model weights. '
        'Otherwise, disable weight decay. Note, lp regularizers add loss to '
        'final loss objective, while decoupled weight decay adds decay '
        'directly into weights. They are different especially when there are '
        'moment statistics in optimizers. A good reference can be found in: '
        'https://www.fast.ai/2018/07/02/adam-weight-decay/#adamw')
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

    # Exponential moving average
    p.Define(
        'ema_decay', 0.0,
        'If > 0, enable ExponentialMovingAverage during training '
        'with the give decay. '
        'Must be < 1. Disabled if <= 0. ')
    p.Define(
        'ema_zero_debias', False,
        'If True, apply zero-debiasing as described in Section 3 '
        'of https://arxiv.org/pdf/1412.6980.pdf.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    self._params = params.Copy()
    p = self._params
    self._lr_schedule = self._params.lr_schedule.Instantiate()
    # Should not mix L1, L2 regularizer and weight decay together.
    if p.l2_regularizer_weight and p.l1_regularizer_weight:
      raise ValueError('Should not mix L1 and L2 regularization.')
    if (p.decoupled_weight_decay and
        (p.l2_regularizer_weight or p.l1_regularizer_weight)):
      raise ValueError(
          'Should not mix decoupled weight decay with L1 or L2 regularization.')

  @property
  def params(self) -> InstantiableParams:
    return self._params

  def get_learning_rate(self, step_count: JTensor) -> JTensor:
    """Get the learning rate of this optimizer at a particular step."""
    return self._lr_schedule.value(step_count) * self.params.learning_rate

  def get_grad_transformation(self) -> GeneralGradientTransformation:
    """Get the grad transformation corresponds to this optimizer config.

    This is the final gradient transformation that incorporates all
    transformations.

    Returns:
      an optax.GradientTransformation or ShardedGradientTransformation.
    """
    # TODO(yonghui): respect gradient clipping, etc transformations.
    p = self.params

    optax_list = [
        apply_lp_regularizer(
            self.get_learning_rate,
            regularizer_weight=p.l1_regularizer_weight,
            p=1.0,
            skip_lp_1d_vectors=p.skip_lp_1d_vectors,
        ),
        apply_lp_regularizer(
            self.get_learning_rate,
            regularizer_weight=p.l2_regularizer_weight,
            p=2.0,
            skip_lp_1d_vectors=p.skip_lp_1d_vectors,
        ),
        self._get_raw_grad_transformation(self.get_learning_rate),
        apply_decoupled_weight_decay(
            self.get_learning_rate,
            regularizer_weight=p.decoupled_weight_decay),
    ]
    if p.ema_decay > 0.0:
      # EMA adds new optimizer states that is not compatible
      asserts.lt(p.ema_decay, 1.)
      optax_list.append(
          apply_ema_weights(decay=p.ema_decay, debias=p.ema_zero_debias))
    return sharded_chain(*optax_list)

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> GeneralGradientTransformation:
    """Get the raw optimizer transformation without taking into other ...

    transformations such l1/l2 regularization, gradient norm clipping, etc.

    Args:
      lr: an optax schedule.

    Returns:
      an optax.GradientTransformation or ShardedGradientTransformation.
    """
    raise NotImplementedError()


class Sgd(BaseOptimizer):
  """Canonical SGD optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    p = super().Params()
    p.Define(
        'momentum', None,
        'Decay rate used by the momentum term. If set to None, momentum is not '
        'used.')
    p.Define('nesterov', False, 'Whether Nesterov momentum is used or not.')
    return p

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    p = self._params
    return optax.sgd(learning_rate=lr, momentum=p.momentum, nesterov=p.nesterov)


class ShardedSgd(BaseOptimizer):
  """Sharded SGD optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    p = super().Params()
    p.Define(
        'momentum', None,
        'Decay rate used by the momentum term. If set to None, momentum is not '
        'used.')
    p.Define('nesterov', False, 'Whether Nesterov momentum is used or not.')
    return p

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    p = self._params
    return sharded_sgd(
        learning_rate_fn=lr, momentum=p.momentum, nesterov=p.nesterov)


class Adam(BaseOptimizer):
  """Adam optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
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
  def ParamsA(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    """Convenient method for a commonly used Adam config."""
    return cls.Params().Set(beta1=0.9, beta2=0.997, epsilon=1e-9)

  @classmethod
  def ParamsB(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    """Convenient method for another commonly used Adam config."""
    return cls.Params().Set(beta1=0.9, beta2=0.98, epsilon=1e-9)

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> GeneralGradientTransformation:
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


class Adafactor(BaseOptimizer):
  """Adafactor optimizer from Optax."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
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

  def _get_raw_grad_transformation(
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


class DistributedShampoo(BaseOptimizer):
  """DistributedShampoo optimizer from Optax."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    p = super().Params()
    p.Define('block_size', 1024,
             'Size of the preconditioner (block size x block size).')
    p.Define('beta1', 0.9, 'Momentum parameter.')
    p.Define('beta2', 0.999, 'Second moment averaging parameter.')
    p.Define('matrix_epsilon', 1e-6,
             'Epsilon parameter as part of computing the inverse-pth roots.')
    p.Define('weight_decay', 0.0, 'Weight decay.')
    p.Define('start_preconditioning_step', 101,
             'Start preconditionining after N steps.')
    p.Define('preconditioning_compute_steps', 100,
             'How often to compute the inverse-pth roots.')
    p.Define('statistics_compute_steps', 1,
             'How often to compute the statistics.')
    p.Define('graft_type', GraftingType.ADAGRAD,
             'Type of Grafting. 1 for SGD, 2 for AdaGrad, 3 for RMSPROP .')
    p.Define('batch_axis_name', 'batch', 'Batch axis name for pmap.')
    p.Define('mesh_axis_names', None, 'Axis names for the mesh (used in pjit).')
    p.Define('num_devices_for_pjit', None,
             'Number of devices to parallelize over in pjit mode.')
    p.Define('nesterov', True, 'Use nesterov update for momentum.')
    p.Define('exponent_override', 0, 'Exponent override.')
    p.Define('moving_average_for_momentum', False,
             'Moving average for momentum.')
    p.Define('skip_preconditioning_dim_size_gt', 4096,
             'Skips preconditioning if any dim is greater than this value.')
    p.Define('clip_by_scaled_gradient_norm', None,
             'Clip by scaled gradient norm (if not None).')
    p.Define('best_effort_shape_interpretation', True,
             'Best effort shape interpretation to coalesce dimensions.')
    p.Define(
        'shard_optimizer_states', False,
        'Shard optimizer states, used by ShardedDistributedShampoo'
        ' (Do not explicitly set this).')
    p.Define(
        'statistics_partition_spec', None,
        'PartitionSpec used by ShardedDistributedShampoo'
        ' (Do not explicitly set this).')
    p.Define(
        'tensor_split_dims_mapping', [-1, 1, -1],
        'Sharding information for statistics and preconditioner matrices.')
    p.Define(
        'preconditioner_partition_spec', None,
        'PartitionSpec used by ShardedDistributedShampoo'
        ' (Do not explicitly set this).')
    p.Define('tensor_split_dims_mapping_for_inverse_pth_root', [1, -1, -1],
             'Sharding information for preconditioner matrices.')
    p.Define('best_effort_memory_usage_reduction', False,
             'Experimental mode: Best effort memory usage reduction.')
    return p

  @classmethod
  def ParamsImageClassification(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    """Common Shampoo config for Image Classification."""
    return cls.Params().Set(
        beta1=0.9,
        beta2=0.95,
        block_size=128,
        weight_decay=1e-4,
        nesterov=True,
        preconditioning_compute_steps=1,
        statistics_compute_steps=1,
        graft_type=GraftingType.SGD)

  @classmethod
  def ParamsLanguageModeling(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    """Common Shampoo config for Language Modeling."""
    return cls.Params().Set(
        block_size=1536,
        beta1=0.9,
        beta2=0.999,
        clip_gradient_norm_to_value=5.0,
        weight_decay=0.0,
        matrix_epsilon=1e-8,
        graft_type=GraftingType.RMSPROP_NORMALIZED,
        nesterov=False,
        exponent_override=0,
        start_preconditioning_step=51,
        preconditioning_compute_steps=50,
        skip_preconditioning_dim_size_gt=4096,
        moving_average_for_momentum=True,
        clip_by_scaled_gradient_norm=None)

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    p = self._params
    return distributed_shampoo_optimizer(
        learning_rate=lr,
        block_size=p.block_size,
        beta1=p.beta1,
        beta2=p.beta2,
        diagonal_epsilon=1e-10,
        matrix_epsilon=p.matrix_epsilon,
        weight_decay=p.weight_decay,
        start_preconditioning_step=p.start_preconditioning_step,
        preconditioning_compute_steps=p.preconditioning_compute_steps,
        statistics_compute_steps=p.statistics_compute_steps,
        best_effort_shape_interpretation=p.best_effort_shape_interpretation,
        graft_type=p.graft_type,
        nesterov=p.nesterov,
        exponent_override=p.exponent_override,
        batch_axis_name=p.batch_axis_name,
        num_devices_for_pjit=p.num_devices_for_pjit,
        statistics_partition_spec=p.statistics_partition_spec,
        preconditioner_partition_spec=p.preconditioner_partition_spec,
        shard_optimizer_states=p.shard_optimizer_states,
        inverse_failure_threshold=0.1,
        moving_average_for_momentum=p.moving_average_for_momentum,
        skip_preconditioning_dim_size_gt=p.skip_preconditioning_dim_size_gt,
        clip_by_scaled_gradient_norm=p.clip_by_scaled_gradient_norm,
        precision=lax.Precision.HIGHEST,
        best_effort_memory_usage_reduction=p.best_effort_memory_usage_reduction)


class ShardedDistributedShampoo(DistributedShampoo):
  """Sharded version of distributed shampoo for model parallel training."""

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    self._params.shard_optimizer_states = True
    self._params.statistics_partition_spec = jax.sharding.PartitionSpec(
        *self._sharded_axes(self._params.mesh_axis_names,
                            self._params.tensor_split_dims_mapping))
    self._params.preconditioner_partition_spec = jax.sharding.PartitionSpec(
        *self._sharded_axes(
            self._params.mesh_axis_names,
            self._params.tensor_split_dims_mapping_for_inverse_pth_root))

  def _sharded_axes(self, axes_names, tensor_split_dims_mapping):
    """Returns the axes to shard with."""
    axes = []
    if not tensor_split_dims_mapping:
      return [None]
    for tsdm in tensor_split_dims_mapping:
      if isinstance(tsdm, str):
        axes.append(tsdm)
      elif tsdm and tsdm != -1:
        axes.append(axes_names[tsdm])
      elif tsdm == -1 or not tsdm:
        axes.append(None)
    return tuple(axes)

  def init_partition_spec_fn(self, init_pspec, init_shapes_dtypes, axes_names,
                             params):
    """Annotates the PartitionSpec for optimizer states."""
    p = self._params
    param_pspec_flattened, _ = jax.tree_flatten(params)
    assert param_pspec_flattened
    first_param = param_pspec_flattened[0]
    assert isinstance(first_param, py_utils.Params)
    assert len(axes_names) == len(p.tensor_split_dims_mapping)
    device_mesh = first_param.device_mesh

    partition_spec_statistics = jax.sharding.PartitionSpec(
        *self._sharded_axes(axes_names, p.tensor_split_dims_mapping))

    def _pspec_from_weight_param(param):
      p = jax.sharding.PartitionSpec(
          *self._sharded_axes(axes_names, param.tensor_split_dims_mapping))
      return p

    partition_spec_params = jax.tree_map(_pspec_from_weight_param, params)
    shapes_and_dtypes = init_shapes_dtypes(params)
    partition_spec_opt_state = init_pspec(params, partition_spec_params,
                                          partition_spec_statistics)

    def _weight_param_from_pspec_shape_dtype(pspec, shapes_and_dtypes):
      if not pspec:
        if len(shapes_and_dtypes[0]) == 1:
          tensor_split_dims_mapping = [-1]
        else:
          tensor_split_dims_mapping = []
      else:
        tensor_split_dims_mapping = []
        if len(pspec) == 1 and not pspec[0]:
          if len(shapes_and_dtypes[0]) == 1:
            tensor_split_dims_mapping = [-1]
          else:
            tensor_split_dims_mapping = []
        else:
          tensor_split_dims_mapping = [
              axes_names.index(axis) if axis else -1 for axis in pspec
          ]
      assert len(shapes_and_dtypes[0]) == len(tensor_split_dims_mapping)
      return py_utils.weight_params(
          shape=shapes_and_dtypes[0],
          init=None,
          dtype=shapes_and_dtypes[1],
          collections=None,
          device_mesh=device_mesh,
          tensor_split_dims_mapping=tensor_split_dims_mapping)

    return jax.tree_map(_weight_param_from_pspec_shape_dtype,
                             partition_spec_opt_state, shapes_and_dtypes)

  def _get_raw_grad_transformation(  # pytype: disable=signature-mismatch  # overriding-return-type-checks
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    result = super()._get_raw_grad_transformation(lr)
    # TODO(rohananil): Refactor after PartitionSpec layering is finalized in
    # the JAX ecosystem.
    fns = result.init(None)  # pytype: disable=wrong-arg-types  # numpy-scalars

    def _wrapped_update_fn(grads, state, params):
      new_params, new_state = result.update(grads, state, params)
      local_stats = new_state.stats.local_stats  # pytype: disable=attribute-error  # numpy-scalars
      var_keys, _ = jax.tree_flatten(
          py_utils.extract_prefixed_keys_from_nested_map(local_stats))
      var_keys = [x for x in var_keys if 'inverse_pth_root_errors' in x]
      is_stats = lambda l: isinstance(l, (LocalShardedParameterStats))
      local_stats_flattened, _ = jax.tree_flatten(local_stats, is_stats)

      def add_summary(key, local_stat):
        num_statistics = len(local_stat.sizes)
        for i in range(num_statistics):
          value = local_stat.training_metrics.inverse_pth_root_errors[i]
          base_layer.add_summary(f'inverse_pth_root_errors/{key}_{i}', value)

      with base_layer.JaxContext.new_context():
        assert len(var_keys) == len(local_stats_flattened)
        for key, local_stat in zip(var_keys, local_stats_flattened):
          add_summary(key, local_stat)
      return new_params, new_state

    return ShardedGradientTransformation(
        init=fns.init_fn,
        update=_wrapped_update_fn,
        init_partition_spec=functools.partial(self.init_partition_spec_fn,
                                              fns.pspec_fn,
                                              fns.shape_and_dtype_fn,
                                              self.params.mesh_axis_names))


class Adagrad(BaseOptimizer):
  """Adagrad optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    p = super().Params()
    p.Define('initial_accumulator_value', 0.1,
             'Initial value of the accumulator.')
    p.Define(
        'epsilon', 1e-10,
        'Small constant applied to the denominator outside of the square root '
        'to avoid dividing by zero when rescaling.')
    return p

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> optax.GradientTransformation:
    p = self._params
    return optax.adagrad(
        learning_rate=lr,
        initial_accumulator_value=p.initial_accumulator_value,
        eps=p.epsilon)


def to_quantized(fvalue: JTensor,
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
    a1, a2 = to_quantized(x, quantized_dtype)
    b1 = to_float(a1, a2)
    c1, c2 = to_quantized(b1, quantized_dtype)

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


def to_float(quantized: JTensor, bucket_size: JTensor) -> JTensor:
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


def adafactor_decay_rate_adam(beta2: float, step_counter: JTensor) -> JTensor:
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


def adafactor_decay_rate_pow(exponent: float, step_counter: JTensor) -> JTensor:
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


def reduce_mean(array: JTensor) -> JTensor:
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


def reduce_rms(array: JTensor) -> JTensor:
  """Computes the RMS of `array` (in a numerically stable way).

  Args:
    array: Input array.

  Returns:
    The root mean square of the input array as a scalar array.
  """
  sq = jnp.square(array)
  sq_mean = reduce_mean(sq)
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


class ShardedAdafactorState(NamedTuple):
  """Overall state of the ShardedAdafactor optimizer."""
  count: JTensor
  m: Optional[NestedJTensor]
  m_scale: Optional[NestedJTensor]
  vr: Optional[NestedJTensor]
  vc: Optional[NestedJTensor]
  v: Optional[NestedJTensor]


class _ShardedAdafactorHelper:
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
      epsilon1_grad_sq_reg: float,
      quantized_dtype: jnp.dtype,
      # TODO(bf-jax) Update default value to True, once this is supported.
      respect_skip_lp_regularization: bool,
      per_var_learning_summary: bool,
      sort_factored_second_moment_dims: bool,
      min_dim_size_to_factor: int,
      multiply_by_parameter_scale: bool,
      epsilon2_param_scale_reg: float) -> None:
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
    self._epsilon1 = epsilon1_grad_sq_reg
    self._quantized_dtype = quantized_dtype
    self._respect_skip_lp_regularization = respect_skip_lp_regularization
    self._per_var_learning_summary = per_var_learning_summary
    self._sort_factored_second_moment_dims = sort_factored_second_moment_dims
    self._min_dim_size_to_factor = min_dim_size_to_factor
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    self._epsilon2 = epsilon2_param_scale_reg

  def should_use_factored_second_moment_estimate(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.

    Args:
      shape: a list of integers.

    Returns:
      A boolean.
    """
    return self.factored_second_moment_dims(shape) is not None

  def factored_second_moment_dims(self, shape):
    """Should we use a factored second moment estimator.

    We select largest and second largest var dims as row and colum dims.

    Default list of factored dims is -1, -2.

    Args:
      shape: a list of integers.

    Returns:
      either a list of 2 Dimension indices for row and col or None
    """
    if not self._factored:
      return None
    if len(shape) < 2:
      return None
    if not self._sort_factored_second_moment_dims:
      return len(shape) - 1, len(shape) - 2

    def largest_two_dim_indices():
      s = [(s, i) for i, s in enumerate(shape)]
      sorted_dims = sorted(s, key=lambda d: -d[0])
      return sorted_dims[0][1], sorted_dims[1][1]

    r_idx, c_idx = largest_two_dim_indices()
    if shape[c_idx] < self._min_dim_size_to_factor:
      return None
    return r_idx, c_idx

  def should_store_momentum_in_qint(self, shape):
    """Should we store momentum as quantized integers.

    Based on the shape of the variable.

    Args:
      shape: a list of integers

    Returns:
      A boolean.
    """
    return len(shape) >= 1

  def to_state(self, count, result_tree):
    """Maps from a tree of (factored) values to separate trees of values."""
    return ShardedAdafactorState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        count=count,
        m=jax.tree_map(lambda o: o.m, result_tree),
        m_scale=jax.tree_map(lambda o: o.m_scale, result_tree),
        vr=jax.tree_map(lambda o: o.vr, result_tree),
        vc=jax.tree_map(lambda o: o.vc, result_tree),
        v=jax.tree_map(lambda o: o.v, result_tree))

  def init(self, param):
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
      if (self._quantized_dtype == jnp.bfloat16 or
          self._quantized_dtype == jnp.float32):
        output_m = jnp.zeros(shape, dtype=self._quantized_dtype)
      elif self.should_store_momentum_in_qint(shape):
        output_m = jnp.zeros(shape, dtype=self._quantized_dtype)
        scale_shape = shape[1:]
        output_m_scale = jnp.zeros(scale_shape, dtype=jnp.float32)
      else:
        output_m = jnp.zeros(shape, dtype=jnp.float32)
    if self.should_use_factored_second_moment_estimate(shape):
      factored_dims = self.factored_second_moment_dims(shape)
      vr_axis, vc_axis = factored_dims
      output_vr_shape = list(shape).copy()
      del output_vr_shape[vr_axis]
      output_vc_shape = list(shape).copy()
      del output_vc_shape[vc_axis]
      output_vr = jnp.zeros(output_vr_shape, dtype=jnp.float32)
      output_vc = jnp.zeros(output_vc_shape, dtype=jnp.float32)
    else:
      output_v = jnp.zeros(shape, dtype=jnp.float32)
    return _ShardedAdafactorUpdateResult(
        update=output_update,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v)

  def init_partition_spec(self, var_param):
    """Initializes the partition spec for a given param."""
    output_update = py_utils.weight_params((1,))
    output_m = py_utils.weight_params((1,))
    output_m_scale = py_utils.weight_params((1,))
    output_vr = py_utils.weight_params((1,))
    output_vc = py_utils.weight_params((1,))
    output_v = py_utils.weight_params((1,))
    shape = var_param.shape
    tensor_split_dims_mapping = var_param.tensor_split_dims_mapping

    if var_param.repeat_prefix:
      raise ValueError(
          'ShardedAdafactor: repeat_prefix is not empty. Consider using '
          'get_transformations_with_vectorized_repeat_prefix to vectorize '
          'prefix dimensions.')

    if tensor_split_dims_mapping is not None:
      assert len(tensor_split_dims_mapping) == len(shape)
      sharding_specified = True
    else:
      sharding_specified = False

    if self._beta1:
      if self._quantized_dtype == jnp.bfloat16:
        output_m = py_utils.weight_params(
            shape=shape,
            init=None,
            dtype=jnp.bfloat16,
            collections=None,
            device_mesh=var_param.device_mesh,
            tensor_split_dims_mapping=tensor_split_dims_mapping)
      elif self.should_store_momentum_in_qint(shape):
        output_m = py_utils.weight_params(
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
          m_scale_split_dims_mapping = gshard_utils.remove_dim(
              0, tensor_split_dims_mapping)
        output_m_scale = py_utils.weight_params(
            shape=scale_shape,
            init=None,
            dtype=jnp.float32,
            collections=None,
            device_mesh=var_param.device_mesh,
            tensor_split_dims_mapping=m_scale_split_dims_mapping)
      else:
        output_m = py_utils.weight_params(
            shape=shape,
            init=None,
            dtype=jnp.float32,
            collections=None,
            device_mesh=var_param.device_mesh,
            tensor_split_dims_mapping=tensor_split_dims_mapping)
    if self.should_use_factored_second_moment_estimate(shape):
      factored_dims = self.factored_second_moment_dims(shape)
      vr_axis, vc_axis = factored_dims
      # TODO(shafey): Fix logic for updating sharding annotations.
      if sharding_specified:
        vr_split_dims_mapping = gshard_utils.remove_dim(
            vr_axis, tensor_split_dims_mapping)
        vc_split_dims_mapping = gshard_utils.remove_dim(
            vc_axis, tensor_split_dims_mapping)
      else:
        vr_split_dims_mapping = tensor_split_dims_mapping
        vc_split_dims_mapping = tensor_split_dims_mapping
      output_vr_shape = list(shape).copy()
      del output_vr_shape[vr_axis]
      output_vr = py_utils.weight_params(
          output_vr_shape,
          init=None,
          dtype=jnp.float32,
          collections=None,
          device_mesh=var_param.device_mesh,
          tensor_split_dims_mapping=vr_split_dims_mapping)
      output_vc_shape = list(shape).copy()
      del output_vc_shape[vc_axis]
      output_vc = py_utils.weight_params(
          output_vc_shape,
          init=None,
          dtype=jnp.float32,
          collections=None,
          device_mesh=var_param.device_mesh,
          tensor_split_dims_mapping=vc_split_dims_mapping)
    else:
      output_v = py_utils.weight_params(
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

  def sanitize_values(self, array, replacement=0.):
    """Sanitizes NaN and Infinity values."""
    return jnp.nan_to_num(
        array, nan=replacement, posinf=replacement, neginf=replacement)

  def parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.

    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.

    Instead of using the value, we could impute the scale from the shape,
    as initializers do.

    Args:
      var: a variable or Tensor.

    Returns:
      a Scalar
    """
    return jnp.maximum(reduce_rms(var), jnp.asarray(self._epsilon2, var.dtype))

  def compute_var_and_slot_update(self, count, grad, m, m_scale, vr, vc, v,
                                  param, var_name):
    """Computes the var and optimizer slots updates for a single variable."""
    # We can probably skip this step
    grad = self.sanitize_values(grad)
    grad = grad.astype(jnp.float32)
    # Add epsilon1_grad_sq_reg as per Algorithm 4
    # of https://arxiv.org/pdf/1804.04235.pdf
    grad_squared = jnp.square(grad) + self._epsilon1
    grad_squared_mean = self.sanitize_values(reduce_mean(grad_squared))
    if self._decay_method == 'adam':
      assert self._decay_adam > 0
      decay_rate = adafactor_decay_rate_adam(self._decay_adam, count)
    elif self._decay_method == 'pow':
      assert self._decay_pow > 0
      decay_rate = adafactor_decay_rate_pow(self._decay_pow, count)
    else:
      raise ValueError(f'decay_method {self._decay_method} not supported.')

    learning_rate = self._learning_rate_fn(count)

    update_scale = learning_rate
    old_val = param

    if self._multiply_by_parameter_scale:
      update_scale *= self.parameter_scale(old_val).astype(update_scale.dtype)  # pytype: disable=attribute-error  # numpy-scalars

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

    factored_second_moment_dims = self.factored_second_moment_dims(shape)
    if factored_second_moment_dims is not None:
      # Q(shafey): Should we use the more numerically stable version
      # reduce_mean().
      vr_axis, vc_axis = factored_second_moment_dims
      grad_squared_row_mean = self.sanitize_values(
          jnp.mean(grad_squared, axis=vr_axis))
      grad_squared_col_mean = self.sanitize_values(
          jnp.mean(grad_squared, axis=vc_axis))
      new_vr = decay_rate * vr + mixing_rate * grad_squared_row_mean
      new_vc = decay_rate * vc + mixing_rate * grad_squared_col_mean
      output_vr = new_vr
      output_vc = new_vc
      long_term_mean = jnp.mean(new_vr, axis=-1, keepdims=True)
      r_factor = 1. / jnp.sqrt(new_vr / long_term_mean)
      c_factor = 1. / jnp.sqrt(new_vc)
      x = grad * jnp.expand_dims(r_factor, vr_axis) * jnp.expand_dims(
          c_factor, vc_axis)
    else:
      # v with sharding annotation.
      new_v = decay_rate * v + mixing_rate * grad_squared
      output_v = new_v
      x = grad / jnp.sqrt(new_v)

    if self._per_var_learning_summary:
      # Add summary for this var.
      x_l2_scale = jnp.sqrt(reduce_mean(x * x))
      base_layer.add_summary(f'sharded_adafactor_learning/{var_name}',
                             x_l2_scale)

    if self._clipping_threshold is not None:
      clipping_denom = jnp.maximum(1., reduce_rms(x) / self._clipping_threshold)
      clipping_denom = self.sanitize_values(clipping_denom, replacement=1.)
      x /= clipping_denom
    subtrahend = update_scale * x
    if self._beta1:
      if self._quantized_dtype == jnp.bfloat16:
        m = m.astype(jnp.float32)
      elif self.should_store_momentum_in_qint(shape):
        m_init_dtype = m.dtype
        m = to_float(m, m_scale)
      subtrahend = self._beta1 * m + (1. - self._beta1) * subtrahend
      subtrahend = self.sanitize_values(subtrahend)
      if self._quantized_dtype == jnp.bfloat16:
        new_m = subtrahend.astype(jnp.bfloat16)
        output_m = new_m
      elif self.should_store_momentum_in_qint(shape):
        # Update the momentum values.
        new_m_val, new_m_scale = to_quantized(subtrahend, m_init_dtype)
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


def sharded_adafactor(
    learning_rate_fn: optax.Schedule,
    weight_decay: Optional[float] = None,
    layerwise_adaptation: bool = False,
    decay_method: str = '',
    decay_adam: float = 0.,
    decay_pow: float = 0.,
    beta1: float = 0.,
    clipping_threshold: float = 1.,
    factored: bool = True,
    epsilon1_grad_sq_reg: float = 1e-30,
    quantized_dtype: jnp.dtype = jnp.int8,
    # TODO(bf-jax) Update default value to True, once this is supported.
    respect_skip_lp_regularization: bool = False,
    per_var_learning_summary=False,
    sort_factored_second_moment_dims=False,
    # min_dim_size_to_factor is only used when
    # sort_factored_second_moment_dims=True.
    min_dim_size_to_factor: int = 128,
    multiply_by_parameter_scale: bool = False,
    epsilon2_param_scale_reg: float = 1e-3,
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
    epsilon1_grad_sq_reg: Regularization constant for squared gradient.
    quantized_dtype: type of the quantized input. Allowed options are jnp.int8,
      jnp.int16, jnp.bfloat16 and jnp.float32. If floating-point type is
      specified, accumulators are stored as such type, instead of quantized
      integers.
    respect_skip_lp_regularization: whether or not to respect lingvo
      SKIP_LP_REGULARIZATION var collection that skips decoupled weight decay.
    per_var_learning_summary: a bool, whether or not to export per-var learning
      summaries.
    sort_factored_second_moment_dims: a bool, whether to select dims to factor
      by size, for the factored second moment.
    min_dim_size_to_factor: an integer, only factor the statistics if two array
      dimensions have at least this size. NOTE: min_dim_size_to_factor is only
      used when sort_factored_second_moment_dims=True.
    multiply_by_parameter_scale: a boolean, if True, then scale learning_rate
      by parameter scale. if False provided learning_rate is absolute step
      size. NOTE: False by default.
    epsilon2_param_scale_reg: Regularization constant for parameter scale.
      Only used when multiply_by_parameter_scale is True.

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

  sharded_adafactor_helper = _ShardedAdafactorHelper(
      learning_rate_fn=learning_rate_fn,
      weight_decay=weight_decay,
      layerwise_adaptation=layerwise_adaptation,
      decay_method=decay_method,
      decay_adam=decay_adam,
      decay_pow=decay_pow,
      beta1=beta1,
      clipping_threshold=clipping_threshold,
      factored=factored,
      epsilon1_grad_sq_reg=epsilon1_grad_sq_reg,
      quantized_dtype=quantized_dtype,
      respect_skip_lp_regularization=respect_skip_lp_regularization,
      per_var_learning_summary=per_var_learning_summary,
      sort_factored_second_moment_dims=sort_factored_second_moment_dims,
      min_dim_size_to_factor=min_dim_size_to_factor,
      multiply_by_parameter_scale=multiply_by_parameter_scale,
      epsilon2_param_scale_reg=epsilon2_param_scale_reg)

  def init_fn(params):
    """Initializes the optimizer's state."""
    return sharded_adafactor_helper.to_state(
        jnp.zeros([], jnp.int32),
        jax.tree_map(sharded_adafactor_helper.init, params))

  def init_partition_spec_fn(var_params):
    var_spec_flattened, _ = jax.tree_flatten(var_params)
    assert var_spec_flattened
    first_var = var_spec_flattened[0]
    assert isinstance(first_var, py_utils.Params)
    device_mesh = first_var.device_mesh
    count = py_utils.weight_params(
        shape=[],
        init=None,
        dtype=jnp.int32,
        collections=None,
        device_mesh=device_mesh,
        tensor_split_dims_mapping=[])
    return sharded_adafactor_helper.to_state(
        count,
        jax.tree_map(sharded_adafactor_helper.init_partition_spec, var_params))

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError(
          'You are using a transformation that requires the current value of '
          'parameters, but you are not passing `params` when calling `update`.')

    compute_var_and_slot_update_fn = functools.partial(
        sharded_adafactor_helper.compute_var_and_slot_update, state.count)
    var_names = py_utils.extract_prefixed_keys_from_nested_map(updates)
    output = jax.tree_map(compute_var_and_slot_update_fn, updates, state.m,
                               state.m_scale, state.vr, state.vc, state.v,
                               params, var_names)
    updates = jax.tree_map(lambda o: o.update, output)
    count_plus_one = state.count + jnp.array(1, jnp.int32)
    updated_states = sharded_adafactor_helper.to_state(count_plus_one, output)
    return updates, updated_states

  return ShardedGradientTransformation(
      init=init_fn,
      update=update_fn,
      init_partition_spec=init_partition_spec_fn)


class ShardedAdafactor(BaseOptimizer):
  """Sharded AdaFactor optimizer."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
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
    p.Define('epsilon1_grad_sq_reg', 1e-30,
             'Regularization constant for squared gradient.')
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
    p.Define('per_var_learning_summary', False,
             'If True, output per var learning summary.')
    p.Define(
        'sort_factored_second_moment_dims', False,
        'If True, will select largest and second largest dims as row and '
        'column dims for factored second moment.')
    # Note this is uncommon: min_dim_size_to_factor does not affect
    # factorization in default case of sort_factored_second_moment_dims=False.
    # Does not match optax API.
    p.Define(
        'min_dim_size_to_factor', 128,
        'Only factor the statistics if two array dimensions have at least '
        'this size. NOTE: min_dim_size_to_factor threshold only applies when ')
    # NOTE this has uncommon: default value False.
    p.Define(
        'multiply_by_parameter_scale', False,
        'If True, then scale learning_rate by parameter norm. if False, '
        'provided learning_rate is absolute step size.')
    p.Define('epsilon2_param_scale_reg', 1e-3,
             'Regularization constant for parameter scale.')
    return p

  def _get_raw_grad_transformation(
      self, lr: optax.Schedule) -> ShardedGradientTransformation:
    p = self._params
    return sharded_adafactor(
        learning_rate_fn=lr,
        weight_decay=p.weight_decay,
        layerwise_adaptation=p.layerwise_adaptation,
        decay_method=p.decay_method,
        decay_adam=p.decay_adam,
        decay_pow=p.decay_pow,
        beta1=p.beta1,
        clipping_threshold=p.clipping_threshold,
        factored=p.factored,
        epsilon1_grad_sq_reg=p.epsilon1_grad_sq_reg,
        quantized_dtype=getattr(jnp, p.quantized_dtype),
        respect_skip_lp_regularization=p.respect_skip_lp_regularization,
        per_var_learning_summary=p.per_var_learning_summary,
        sort_factored_second_moment_dims=p.sort_factored_second_moment_dims,
        min_dim_size_to_factor=p.min_dim_size_to_factor,
        multiply_by_parameter_scale=p.multiply_by_parameter_scale,
        epsilon2_param_scale_reg=p.epsilon2_param_scale_reg)
