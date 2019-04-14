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
# ==============================================================================
"""Learning rate schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from tensorflow.python.framework import function
from lingvo.core import base_layer
from lingvo.core import early_stop
from lingvo.core import py_utils
from lingvo.core.ops import py_x_ops


class BaseLearningRateSchedule(base_layer.BaseLayer):
  """Base class for learning rate decay algorithms."""

  @classmethod
  def Params(cls):
    p = super(BaseLearningRateSchedule, cls).Params()
    p.name = 'LRSched'
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseLearningRateSchedule, self).__init__(params)

  def Value(self, current_step):
    """Returns the current learning rate schedule value.

    Args:
      current_step: The current global step value.

    Returns:
      Returns the current learning rate schedule value given the
      current training global step. Typically, the base learning rate
      multiplied by the returned schedule value is used as the
      effective learning rate.
    """
    return self.FProp(self.theta, current_step)


class ConstantOne(BaseLearningRateSchedule):
  """A lr schedule remains constant 1."""

  def FProp(self, theta, current_step):
    del theta, current_step
    return tf.constant(1.0, self.params.dtype)


class PiecewiseConstantLearningRateSchedule(BaseLearningRateSchedule):
  """Piecewise constants rate decay."""

  @classmethod
  def Params(cls):
    p = super(PiecewiseConstantLearningRateSchedule, cls).Params()
    p.Define('boundaries', None, 'Boundaries at which learning rate drops.')
    p.Define('values', None, 'Values in each interval.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(PiecewiseConstantLearningRateSchedule, self).__init__(params)

  def FProp(self, theta, current_step):
    p = self.params
    return py_utils.PiecewiseConstant(current_step, p.boundaries, p.values,
                                      p.dtype)


class ContinuousLearningRateSchedule(BaseLearningRateSchedule):
  """Continuous learning rate decay."""

  @classmethod
  def Params(cls):
    p = super(ContinuousLearningRateSchedule, cls).Params()
    p.Define('initial_value', 1.0, 'Initial decay value.')
    p.Define('start_step', 400000,
             'Starts to decay the learning rate from this step.')
    p.Define('half_life_steps', 100000,
             'Halve the learning rate every this many steps after start_step.')
    p.Define('min', 0.01, 'Minimum relative learning rate.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ContinuousLearningRateSchedule, self).__init__(params)
    p = self.params
    q = ExponentialLearningRateSchedule.Params().Set(
        start=(p.start_step, 1.0),
        limit=(p.start_step +
               p.half_life_steps * math.log(p.min) / math.log(0.5), p.min))
    self.CreateChild('exp', q)

  def FProp(self, theta, current_step):
    """Returns the current learning rate decay."""
    return self.params.initial_value * self.exp.Value(current_step)


class PolynomialLearningRateSchedule(BaseLearningRateSchedule):
  """Polynomial learning rates.

  If x < x0, returns y0. If x >= x1, returns y1. Otherwise,
  interpolate with a polynomial between (x0, y0) and (x1, y1).

  """

  @classmethod
  def Params(cls):
    p = super(PolynomialLearningRateSchedule, cls).Params()
    p.Define('power', 1, 'Polynomial power.')
    p.Define('start', (0, 1.), '(x0, y0)')
    p.Define('limit', (1, 1.), '(x1, y1)')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(PolynomialLearningRateSchedule, self).__init__(params)

    @function.Defun()
    def Polynomial(x):
      """Polynomial function of x."""
      p = self.params
      x0, y0 = p.start
      x1, y1 = p.limit

      assert x0 < x1, '%s must be < %s' % (x0, x1)

      x0 = tf.cast(x0, dtype=x.dtype)
      x1 = tf.cast(x1, dtype=x.dtype)
      y0 = tf.cast(y0, dtype=x.dtype)
      y1 = tf.cast(y1, dtype=x.dtype)

      f_x = ((x - x0) / (x1 - x0))**p.power
      y = y0 + f_x * (y1 - y0)
      return tf.where(x < x0, y0, tf.where(x >= x1, y1, y))

    self._polynomial = Polynomial

  def FProp(self, theta, current_step):
    return self._polynomial(tf.cast(current_step, dtype=self.params.dtype))


class LinearLearningRateSchedule(PolynomialLearningRateSchedule):
  """Linear learning rate schedule.

  If x < x0, returns y0. If x >= x1, returns y1. Otherwise,
  interpolate linearly between (x0, y0) and (x1, y1).

  """

  @classmethod
  def Params(cls):
    p = super(LinearLearningRateSchedule, cls).Params().Set(power=1)
    return p


class ExponentialLearningRateSchedule(BaseLearningRateSchedule):
  """Linear learning rate schedule.

  If x < x0, returns y0. If x >= x1, returns y1. Otherwise,
  interpolate exponentially between (x0, y0) and (x1, y1).
  """

  @classmethod
  def Params(cls):
    p = super(ExponentialLearningRateSchedule, cls).Params()
    p.Define('start', (0, 1.), '(x0, y0)')
    p.Define('limit', (1, 0.5), '(x1, y1)')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ExponentialLearningRateSchedule, self).__init__(params)
    p = self.params
    x0, y0 = p.start
    x1, y1 = p.limit
    assert x0 < x1, '%s must be < %s' % (x0, x1)
    assert y0 > 0, '%s must be > 0' % (y0)
    assert y1 > 0, '%s must be > 0' % (y1)

    self.CreateChild(
        'linear',
        LinearLearningRateSchedule.Params().Set(
            start=(x0, math.log(y0)), limit=(x1, math.log(y1))))

    @function.Defun()
    def Exp(x):
      return tf.exp(self.linear.Value(x))

    self._exp = Exp

  def FProp(self, theta, current_step):
    return self._exp(tf.cast(current_step, dtype=self.params.dtype))


class StepwiseExponentialSchedule(BaseLearningRateSchedule):
  """Exponential decay every N steps."""

  @classmethod
  def Params(cls):
    p = super(StepwiseExponentialSchedule, cls).Params()
    p.Define('decay', 0.99, 'Decay factor.')
    p.Define('num_steps_per_decay', 1000, 'Number of steps between decays.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(StepwiseExponentialSchedule, self).__init__(params)

  def FProp(self, theta, current_step):
    p = self.params
    num_decays = tf.floor(
        tf.div(tf.cast(current_step, tf.float32), float(p.num_steps_per_decay)))
    return tf.pow(p.decay, num_decays)


class CombinedMinimumLearningRateSchedule(BaseLearningRateSchedule):
  """Combine a few learning rate decay schedules and takes the min."""

  @classmethod
  def Params(cls):
    p = super(CombinedMinimumLearningRateSchedule, cls).Params()
    p.Define('schedules', [LinearLearningRateSchedule.Params()],
             'A list of learning rate schedule params.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(CombinedMinimumLearningRateSchedule, self).__init__(params)
    p = self.params
    self.CreateChildren('schedules', p.schedules)

    @function.Defun()
    def Combined(x):
      ys = [s.Value(x) for s in self.schedules]
      return tf.reduce_min(tf.stack(ys), axis=0)

    self._combined = Combined

  def FProp(self, theta, current_step):
    return self._combined(current_step)


class TransformerLearningRateSchedule(BaseLearningRateSchedule):
  """Inverse-decay learning rate until warmup_steps, then decay."""

  @classmethod
  def Params(cls):
    p = super(TransformerLearningRateSchedule, cls).Params()
    p.Define(
        'warmup_steps', 4000, 'Increase the learning rate linearly for '
        'the first warmup_steps training steps.')
    p.Define(
        'model_dim', 512, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    p.Define('worker_replicas', 1, 'Number of worker replicas.')
    p.Define('decay_end', None, 'Ends the learning rate decay at '
             'decay_end-th step.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerLearningRateSchedule, self).__init__(params)

  def FProp(self, theta, current_step):
    """Returns the current learning rate decay."""
    p = self.params
    current_step = tf.to_float(current_step)
    warmup_steps = tf.to_float(p.warmup_steps * p.worker_replicas)
    if p.decay_end is not None:
      current_step = tf.where(current_step < p.decay_end, current_step,
                              tf.to_float(p.decay_end))
    return p.model_dim**-0.5 * tf.minimum(
        (current_step + 1) * warmup_steps**-1.5, (current_step + 1)**-0.5)


class TransformerLearningRateScheduleNoWarmUp(BaseLearningRateSchedule):
  """Fixed learning rate until decay_start, then decay.

  This learning rate schedule is identical to TransformerLearningRateSchedule
  except in the warm-up phase, where this learning rate schedule uses a fixed
  learning rate (peak-learning rate of TransformerLearningRateSchedule) for the
  original warm-up phase.
  """

  @classmethod
  def Params(cls):
    p = super(TransformerLearningRateScheduleNoWarmUp, cls).Params()
    p.Define('decay_start', 4000, 'It is used to estimate peak-lr.')
    p.Define('decay_end', None, 'Ends the learning rate decay at '
             'decay_end-th step.')
    p.Define(
        'model_dim', 512, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    p.Define('worker_replicas', 1, 'Number of worker replicas.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerLearningRateScheduleNoWarmUp, self).__init__(params)
    tf.logging.info(
        'Peak lr: %f',
        (self.params.decay_start * self.params.worker_replicas)**-0.5)

  def FProp(self, theta, current_step):
    """Returns the current learning rate decay."""
    params = self.params
    warmup_steps = tf.to_float(params.decay_start * params.worker_replicas)
    current_step = tf.to_float(current_step)
    if params.decay_end is not None:
      current_step = tf.where(current_step < params.decay_end, current_step,
                              tf.to_float(params.decay_end))
    peak_learning_rate = (warmup_steps**-0.5)
    return (params.model_dim**-0.5) * tf.minimum(
        tf.minimum((current_step + 1),
                   (current_step + 1)**-0.5), peak_learning_rate)


class LinearRampupExponentialDecayScaledByNumSplitSchedule(
    BaseLearningRateSchedule):
  """A learning rate schedule that does the following.

  1. The peak learning rate multiplier is scaled by num splits,
     (often the same as #replicas during batch splitting synchronous
     training).
  2. The multiplier ramps up linearly from 1 to the peak initially.
  3. The multiplier stays constant until the exponential decay starts.
  4. The multiplier is capped at max.
  """

  @classmethod
  def Params(cls):
    p = super(LinearRampupExponentialDecayScaledByNumSplitSchedule,
              cls).Params()
    p.Define(
        'warmup', 300, 'Increases the learning rate linearly  '
        'before warmup * num_splits steps.')
    p.Define('warmup_init', 1.0, 'The initial value of the warm-up phase.')
    p.Define('decay_start', 70000, 'Starts the learning rate decay at '
             'decay_start-th step.')
    p.Define('decay_end', 100000, 'Ends the learning rate decay at '
             'decay_end-th step.')
    p.Define('min', 0.01, 'After decay_end, the multiplier stays at min.')
    p.Define(
        'max', 1e8, 'The schedule is never larger than this value. '
        'By default, 1e8 effectively means there is no cap.')
    p.Define(
        'num_splits', 0, 'Specifies the intended number of splits for the '
        'LR. Overrides num_splits_per_client if non-zero.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LinearRampupExponentialDecayScaledByNumSplitSchedule,
          self).__init__(params)

    p = self.params

    # We always compute lr schedule from the trainer's perspective.
    # Also note that this schedule makes sense to sync training only.
    if p.num_splits:
      splits = p.num_splits
    else:
      # Infer num_splits from cluster.
      cluster_params = self.cluster.params.Copy()
      cluster_params.task = 0
      assert cluster_params.mode == 'sync'
      cluster_params.job = 'trainer_client'
      my_cluster = cluster_params.cls(cluster_params)
      splits = my_cluster.num_splits_per_client

    warmup_end = p.warmup * splits
    decay_start = max(warmup_end + 1.0, p.decay_start / splits)
    peak = 1.0 * splits
    tf.logging.info('Peak lr: %f', peak)
    decay_end = max(decay_start + 1.0, p.decay_end / splits)
    schedules = [
        LinearLearningRateSchedule.Params().Set(
            start=(0., p.warmup_init), limit=(warmup_end, peak)),
        LinearLearningRateSchedule.Params().Set(
            start=(warmup_end, peak), limit=(decay_start, peak)),
        ExponentialLearningRateSchedule.Params().Set(
            start=(decay_start, peak), limit=(decay_end, p.min)),
        LinearLearningRateSchedule.Params().Set(
            start=(0, p.max), limit=(decay_end, p.max)),
    ]
    self.CreateChild(
        'combine',
        CombinedMinimumLearningRateSchedule.Params().Set(schedules=schedules))

  def FProp(self, theta, current_step):
    return self.combine.Value(current_step)


class LinearRampupPiecewiseConstantSchedule(BaseLearningRateSchedule):
  """A learning rate schedule that does the following.

  1. The learning rate is scaled by #split * lrs[i]
     (often #split is the same as #replicas during batch splitting synchronous
     training).
  2. The multiplier ramps up linearly from 0 to the peak(lrs[0]) at
     boundaries[0].
  3. After peak, the multiplier stays lrs[i] when step falls into
     [boundaries[i], boundaries[i+1])
  """

  @classmethod
  def Params(cls):
    p = super(LinearRampupPiecewiseConstantSchedule, cls).Params()
    p.Define('boundaries', [], 'Boundaries at which learning rate changes.')
    p.Define('lrs', [], 'A list of learning rate multiplers.')
    p.Define(
        'num_splits', 0, 'Specifies the intended number of num_splits for '
        'LR. Overrides num_splits if non-zero.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LinearRampupPiecewiseConstantSchedule, self).__init__(params)

    p = self.params
    assert len(p.boundaries) >= 2 and len(p.boundaries) == len(p.lrs)
    # We always compute lr schedule from the trainer's perspective.
    # Also note that this schedule makes sense to sync training only.
    if p.num_splits:
      splits = p.num_splits
    else:
      # Infer num_splits from cluster.
      cluster_params = self.cluster.params.Copy()
      cluster_params.task = 0
      assert cluster_params.mode == 'sync'
      cluster_params.job = 'trainer_client'
      my_cluster = cluster_params.cls(cluster_params)
      splits = my_cluster.num_splits_per_client

    assert splits >= 1
    splits = float(splits)
    boundaries = [step / splits for step in p.boundaries]
    lrs = [step * splits for step in p.lrs]

    tf.logging.info('splits: {}\n boundaries: {}\n lrs: {} '.format(
        splits, boundaries, lrs))

    schedules = [
        LinearLearningRateSchedule.Params().Set(
            start=(0., 0.), limit=(boundaries[0], lrs[0])),
        PiecewiseConstantLearningRateSchedule.Params().Set(
            boundaries=boundaries, values=[1e8] + lrs)
    ]
    self.CreateChild(
        'combine',
        CombinedMinimumLearningRateSchedule.Params().Set(schedules=schedules))

  def FProp(self, theta, current_step):
    return self.combine.Value(current_step)


class LinearRampupCosineSchedule(BaseLearningRateSchedule):
  """A cosine decaying learning rate schedule with a linear rampup phase."""

  @classmethod
  def Params(cls):
    p = super(LinearRampupCosineSchedule, cls).Params()
    p.Define('warmup_init', 0, 'The initial lr value of the warm-up phase.')
    p.Define('warmup_steps', 0, 'Number of warm up steps.')
    p.Define('initial_value', 1.0, 'Initial decay value.')
    p.Define('total_steps', 0, 'Number of steps to reach full decay.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LinearRampupCosineSchedule, self).__init__(params)
    p = self.params
    schedules = [
        LinearLearningRateSchedule.Params().Set(
            start=(0., p.warmup_init), limit=(p.warmup_steps, p.initial_value)),
        CosineSchedule.Params().Set(
            initial_value=p.initial_value, total_steps=p.total_steps),
    ]
    self.CreateChild(
        'combine',
        CombinedMinimumLearningRateSchedule.Params().Set(schedules=schedules))

  def FProp(self, theta, current_step):
    return self.combine.Value(current_step)


class DevBasedSchedule(BaseLearningRateSchedule):
  """Decay triggered by lack of improvement on the dev set.

  This reads a file containing a history of values of a selected metric versus
  global step (file is recorded by the evaler loop in the trainer). Decay
  depends on these variables:

    - best_step - step at which optimum metric value occurred in history file
    - last_step - last step recorded in history file
    - ref_step - most recent decay step or best_step
    - cur_factor - current multiplier on initial learning rate

  The decay algorithm is::

    ref_step = max(ref_step, best_step)
    if last_step - ref_step > window:
      cur_factor = max(cur_factor * decay, min_factor)
      ref_step = last_step
  """

  @classmethod
  def Params(cls):
    p = super(DevBasedSchedule, cls).Params()
    p.Define('metric_history', early_stop.MetricHistory.Params(),
             'Metric to monitor for stopping.')
    p.Define('tolerance', 0.0, 'Minimum significant difference in metric.')
    p.Define('window', 10000,
             'Steps since most recent decay or best_step before decaying.')
    p.Define('decay', 0.5,
             'Factor by which learning rate multiplier is decayed.')
    p.Define('min_factor', 0.01, 'Minimum learning rate multiplier.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(DevBasedSchedule, self).__init__(params)

    p = self.params

    with tf.variable_scope(p.name):
      wp = py_utils.WeightParams(
          shape=[],
          init=py_utils.WeightInit.Constant(1.0),
          collections=['DevBasedSchedule_vars'],
          dtype=tf.float32)
      _, self._cur_factor, = py_utils.CreateVariable(
          'cur_factor', wp, trainable=False)
      wp = py_utils.WeightParams(
          shape=[],
          init=py_utils.WeightInit.Constant(0),
          collections=['DevBasedSchedule_vars'],
          dtype=tf.int64)
      _, self._ref_step, = py_utils.CreateVariable(
          'ref_step', wp, trainable=False)

      self._metric_history = early_stop.MetricHistory(p.metric_history)
      self._best_step = py_x_ops.best_step(self._metric_history.hist_file,
                                           p.tolerance)

  def FProp(self, theta, current_step):
    p = self.params
    with tf.name_scope(p.name):

      steps = self._best_step
      best_step = steps[0]
      last_step = steps[1]

      ref_step = tf.maximum(self._ref_step, best_step)
      f = self._cur_factor

      # Decay if no improvement within window.
      new_factor = tf.where(last_step - ref_step < p.window, f,
                            tf.maximum(p.min_factor, f * p.decay))
      # Update ref_step if we decayed.
      new_step = tf.where(tf.equal(new_factor, f), ref_step, last_step)
      update_step = tf.assign(self._ref_step, new_step)
      with tf.control_dependencies([update_step]):
        return tf.assign(self._cur_factor, new_factor)


class CosineSchedule(BaseLearningRateSchedule):
  """Cosine learning rate decay.

  First proposed in https://arxiv.org/pdf/1608.03983.pdf, which only uses
  multiple cycles with angle from 0 to pi/2. Later people use only one cycle
  with angle from 0 to pi (e.g., https://arxiv.org/pdf/1711.09224.pdf), which
  is implemented here.

  where:
    angle = pi * min(1, current_step / total_steps)
    decay_gap = initial_value - final_value
    value = final_value + decay_gap * (1 + cosine(angle)) / 2
  """

  @classmethod
  def Params(cls):
    p = super(CosineSchedule, cls).Params()
    p.Define('initial_value', 1.0, 'Initial decay value.')
    p.Define('final_value', 0., 'Final decay value.')
    p.Define('total_steps', 0, 'Number of steps to reach full decay.')
    return p

  def FProp(self, theta, current_step):
    p = self.params
    assert p.total_steps > 0
    assert p.initial_value > p.final_value
    with tf.name_scope(p.name):
      decay_gap = p.initial_value - p.final_value
      return p.final_value + 0.5 * decay_gap * (1 + tf.cos(math.pi * tf.minimum(
          1.0,
          tf.cast(current_step, tf.float32) / p.total_steps)))


class PiecewiseSchedule(BaseLearningRateSchedule):
  """Piecewise schedule composed of sub-schedules."""

  @classmethod
  def Params(cls):
    p = super(PiecewiseSchedule, cls).Params()
    p.Define('boundaries', None, 'Boundaries between subschedules.')
    p.Define(
        'schedules', None, 'A list of sub-schedules. '
        'The length must be len(boundaries) + 1. '
        'schedules[i] starts at boundaries[i-1] (inclusive) and ends at '
        'boundaries[i] (exclusive). '
        'The *relative* step in each interval will be passed to the '
        'sub-schedule for FProp.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(PiecewiseSchedule, self).__init__(params)
    p = self.params
    prev_boundary = 0
    for boundary in p.boundaries:
      if boundary < prev_boundary:
        raise ValueError('Invalid boundary %s < %s' % (boundary, prev_boundary))
      prev_boundary = boundary
    if len(p.schedules) != len(p.boundaries) + 1:
      raise ValueError('len(schedules) != len(boundaries) + 1: %s vs %s' %
                       (len(p.schedules), len(p.boundaries)))
    self.CreateChildren('schedules', p.schedules)

  def FProp(self, theta, current_step):
    p = self.params
    current_step = tf.cast(current_step, tf.int64)
    interval_starts = [0] + p.boundaries
    values = []
    for interval_start, schedule, schedule_theta in zip(
        interval_starts, self.schedules, theta.schedules):
      relative_step = tf.maximum(
          tf.cast(0, current_step.dtype), current_step - interval_start)
      values.append(schedule.FProp(schedule_theta, relative_step))

    return py_utils.PiecewiseConstant(current_step, p.boundaries, values,
                                      values[0].dtype)
