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
"""Learning rate schedule."""

import math

import jax
from jax import numpy as jnp
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import optax

JTensor = pytypes.JTensor
InstantiableParams = py_utils.InstantiableParams


class BaseSchedule:
  """Base class for all schedules."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = InstantiableParams(cls)
    return p

  def __init__(self, params: InstantiableParams) -> None:
    self._params = params.Copy()

  @property
  def params(self) -> InstantiableParams:
    return self._params

  def Value(self, count: JTensor) -> JTensor:
    """Returns the value of schedule at step 'count'.

    Args:
      count: a scalar uint32 array.

    Returns:
      A float32 value of the schedule at step 'count' as a scalar array.
    """
    raise NotImplementedError()


class ConstantSchedule(BaseSchedule):
  """A schedule whose value is a constant."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('value', 1., 'The constant value.')
    return p

  def Value(self, count: JTensor) -> JTensor:
    del count
    return jnp.array(self.params.value, dtype=jnp.float32)


class PolynomialSchedule(BaseSchedule):
  """Polynomial learning rate schedule.

  If x < x0, returns y0. If x >= x1, returns y1. Otherwise, interpolate with
  a polynomial between (x0, y0) and (x1, y1).
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('power', 1, 'Polynomial power.')
    p.Define('start', (0, 1.), '(x0, y0)')
    p.Define('limit', (1, 1.), '(x1, y1)')
    p.Define('origin', 'start',
             'Origin of the polynomial. Can be "start" or "limit".')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    if len(p.start) != 2:
      raise ValueError(f'{p.start} must be of length 2.')
    if len(p.limit) != 2:
      raise ValueError(f'{p.limit} must be of length 2.')
    x0, _ = p.start
    x1, _ = p.limit
    if x0 >= x1:
      raise ValueError(f'{x0} must be < {x1}')
    if p.origin not in {'start', 'limit'}:
      raise ValueError('Invalid parameter origin: %s' % p.origin)

  def Value(self, count: JTensor) -> JTensor:
    p = self.params
    x = jnp.array(count).astype(jnp.float32)
    x0, y0 = p.start
    x1, y1 = p.limit
    ratio = (x - x0) / (x1 - x0)
    if p.origin == 'start':
      f_x = ratio**p.power
    elif p.origin == 'limit':
      f_x = 1 - (1 - ratio)**p.power
    y = y0 + f_x * (y1 - y0)
    return jnp.where(x < x0, y0, jnp.where(x >= x1, y1, y))


class LinearSchedule(PolynomialSchedule):
  """Linear learning rate schedule.

  If x < x0, returns y0. If x >= x1, returns y1. Otherwise, interpolate
  linearly between (x0, y0) and (x1, y1).
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    return super().Params().Set(power=1)


class ExponentialSchedule(BaseSchedule):
  """Exponential learning rate schedule."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('start', (0, 1.), '(x0, y0)')
    p.Define('limit', (1, 0.5), '(x1, y1)')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    x0, y0 = p.start
    x1, y1 = p.limit
    assert x0 < x1, '%s must be < %s' % (x0, x1)
    assert y0 > 0, '%s must be > 0' % y0
    assert y1 > 0, '%s must be > 0' % y1
    self.linear = LinearSchedule.Params().Set(
        start=(x0, math.log(y0)), limit=(x1, math.log(y1))).Instantiate()

  def Value(self, count: JTensor) -> JTensor:
    return jnp.exp(self.linear.Value(count))


class PiecewiseConstantSchedule(BaseSchedule):
  """A schedule with piecewise constants rate decay."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('boundaries', None, 'Boundaries at which learning rate drops.')
    p.Define(
        'values', None,
        'Values in each interval. The number of values must be equal to the '
        'the number of boundaries plus 1.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    if p.boundaries is None or p.values is None:
      raise ValueError(
          'The parameters `boundaries` and `values` must not be None.')
    if len(p.values) != len(p.boundaries) + 1:
      raise ValueError(
          f'The number of values ({len(p.values)}) is expected to be equal '
          f'to the number of boundaries plus 1 ({len(p.boundaries) + 1}).')
    if sorted(p.boundaries) != list(p.boundaries):
      raise ValueError(f'The boundaries ({p.boundaries}) must be sorted.')

  def Value(self, count: JTensor) -> JTensor:
    p = self.params
    # Map the step/boundaries to jnp.float32.
    boundaries = [jnp.array(v, dtype=jnp.float32) for v in p.boundaries]
    values = [jnp.array(v, dtype=jnp.float32) for v in p.values]
    count = count.astype(jnp.float32)
    if not boundaries:
      assert len(values) == 1
      return values[0]
    v = 0
    for i, threshold in enumerate(boundaries):
      indicator = jnp.maximum(0., jnp.sign(threshold - count))
      v = jnp.where(v > 0, v, indicator * values[i])
    # Check if step is greater equal to the last value.
    indicator = jnp.maximum(0., jnp.sign(1 + count - boundaries[-1]))
    v = jnp.where(v > 0, v, indicator * values[-1])
    return v


class TransformerSchedule(BaseSchedule):
  """Inverse-decay learning rate until warmup_steps, then decay."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'warmup_steps', 4000,
        'Increase the learning rate linearly for the first warmup_steps '
        'training steps.')
    p.Define(
        'model_dim', 512,
        'Model dimension that applies to embedding layers and all Transformer '
        'layers.')
    p.Define('worker_replicas', 1, 'Number of worker replicas.')
    p.Define('decay_end', None,
             'Ends the learning rate decay at decay_end-th step.')
    return p

  def Value(self, count: JTensor) -> JTensor:
    """Returns the current learning rate decay."""
    p = self.params
    current_step = count.astype(jnp.float32)
    model_dim = jnp.array(p.model_dim, dtype=jnp.float32)
    warmup_steps = jnp.array(
        p.warmup_steps * p.worker_replicas, dtype=jnp.float32)
    if p.decay_end is not None:
      decay_end = jnp.array(p.decay_end, dtype=jnp.float32)
      current_step = jnp.where(current_step < decay_end, current_step,
                               decay_end)
    return model_dim**-0.5 * jnp.minimum(
        (current_step + 1) * warmup_steps**-1.5, (current_step + 1)**-0.5)


class SqrtDecaySchedule(BaseSchedule):
  """Square root decay learning rate after warmup_steps."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'warmup_steps', 10000, 'Increase the learning rate linearly for '
        'the first warmup_steps training steps.')
    p.Define('multiplier', 1., 'Multiplier.')
    p.Define('offset', 0., 'Offset.')
    return p

  def Value(self, count: JTensor) -> JTensor:
    """Returns the current learning rate decay."""
    p = self.params
    current_step = count.astype(jnp.float32)
    offset = jnp.array(p.offset, dtype=jnp.float32)
    warmup_steps = jnp.array(p.warmup_steps, dtype=jnp.float32)
    multiplier = jnp.array(p.multiplier, dtype=jnp.float32)
    return jax.lax.rsqrt(jnp.maximum(current_step - offset,
                                     warmup_steps)) * multiplier


class LinearRampupExponentialDecay(BaseSchedule):
  """Learning rate that first linearly ramps up to max and exponentially decays."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'warmup', 0,
        'Increases the learning rate linearly  before warmup * num_splits '
        'steps.')
    p.Define('decay_start', 0,
             'Starts the learning rate decay at decay_start-th step.')
    p.Define('decay_end', 0,
             'Ends the learning rate decay at decay_end-th step.')
    p.Define('min_ratio', 0.01, 'After decay_end, the multiplier stays at min.')
    p.Define('max', 0, 'The schedule is never larger than this value.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)

    p = self.params

    assert p.decay_start >= p.warmup, ('decay_start must greater than warmup.')
    assert p.decay_end >= p.decay_start, (
        'decay_end must be greater than decay_start')
    assert p.max > 0, 'Must set max.'

    # Offset the boundaries, since each schedule passed to
    # optax.join_schedules() will receive a step count indicating the number
    # of steps since the previous boundary transition.
    self._schedules = []
    self._boundaries = []
    if p.warmup > 0:
      self._schedules.append(LinearSchedule.Params().Set(
          start=(0, 0.0), limit=(p.warmup, p.max)).Instantiate())
      self._boundaries.append(p.warmup)
    if p.decay_start > p.warmup:
      self._schedules.append(LinearSchedule.Params().Set(
          start=(0, p.max),
          limit=(p.decay_start - p.warmup, p.max)).Instantiate())
      self._boundaries.append(p.decay_start)
    self._schedules.append(ExponentialSchedule.Params().Set(
        start=(0, p.max),
        limit=(p.decay_end - p.decay_start, p.max * p.min_ratio)).Instantiate())

  def Value(self, value: JTensor) -> JTensor:
    return jnp.array(
        optax.join_schedules([s.Value for s in self._schedules],
                             self._boundaries)(value), jnp.float32)


class LinearRampupPiecewiseConstantSchedule(BaseSchedule):
  """A learning rate schedule that does the following.

  1. The multiplier ramps up linearly from 0 to the peak(lrs[0]) at
     boundaries[0].
  2. After peak, the multiplier stays values[i] when step falls into
     [boundaries[i], boundaries[i+1]).
  3. When step is more than boundaries[-1], then the multiplier is values[-1].
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('boundaries', [], 'Boundaries at which learning rate changes.')
    p.Define(
        'values', [],
        'The learning rate values for the PiecewiseConstant schedule and if '
        'the step is between boundaries[i] and boundaries[i + 1] then '
        'values[i] is returned, except when it is linearly ramping up from to '
        'values[0].')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    assert len(p.boundaries) >= 1 and len(p.boundaries) == len(p.values)
    self.p0 = LinearSchedule.Params().Set(
        start=(0, 0.0), limit=(p.boundaries[0], p.values[0])).Instantiate()
    # Offset the boundaries, since each schedule passed to
    # optax.join_schedules() will receive a step count indicating the number
    # of steps since the previous boundary transition.
    boundaries_pc = [b - p.boundaries[0] for b in p.boundaries[1:]]
    self.p1 = PiecewiseConstantSchedule.Params().Set(
        boundaries=boundaries_pc, values=p.values).Instantiate()

  def Value(self, value: JTensor) -> JTensor:
    p = self.params
    return jnp.array(
        optax.join_schedules([self.p0.Value, self.p1.Value],
                             p.boundaries[:1])(value), jnp.float32)
