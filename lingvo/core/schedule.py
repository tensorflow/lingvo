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
# ==============================================================================
"""Learning rate schedule."""

import math
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import early_stop
from lingvo.core import ops
from lingvo.core import py_utils


class BaseSchedule(base_layer.BaseLayer):
  """Base class for learning rate decay algorithms."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'LRSched'
    return p

  def __init__(self, params):
    super().__init__(params)
    self.SetVariableFree()

  def Value(self):
    """Returns the current learning rate schedule value.

    Returns:
      Returns the current learning rate schedule value given the
      current training global step. Typically, the base learning rate
      multiplied by the returned schedule value is used as the
      effective learning rate.
    """
    raise NotImplementedError()


class Constant(BaseSchedule):
  """A schedule that always returns a constant value."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('value', 1., 'The constant value.')
    return p

  def Value(self):
    return tf.constant(self.params.value, self.params.dtype)


class ConstantOne(Constant):
  """A lr schedule remains constant 1."""
  pass


class PiecewiseConstantSchedule(BaseSchedule):
  """Piecewise constants rate decay."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('boundaries', None, 'Boundaries at which learning rate drops.')
    p.Define('values', None, 'Values in each interval.')
    return p

  def Value(self):
    p = self.params
    return py_utils.PiecewiseConstant(py_utils.GetGlobalStep(), p.boundaries,
                                      p.values, p.dtype)


class ContinuousSchedule(BaseSchedule):
  """Continuous learning rate decay."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('initial_value', 1.0, 'Initial decay value.')
    p.Define('start_step', 400000,
             'Starts to decay the learning rate from this step.')
    p.Define('half_life_steps', 100000,
             'Halve the learning rate every this many steps after start_step.')
    p.Define('min', 0.01, 'Minimum relative learning rate.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    q = ExponentialSchedule.Params().Set(
        start=(p.start_step, 1.0),
        limit=(p.start_step +
               p.half_life_steps * math.log(p.min) / math.log(0.5), p.min))
    self.CreateChild('exp', q)

  def Value(self):
    """Returns the current learning rate decay."""
    return self.params.initial_value * self.exp.Value()


class PolynomialSchedule(BaseSchedule):
  """Polynomial learning rates.

  If x < x0, returns y0. If x >= x1, returns y1. Otherwise,
  interpolate with a polynomial between (x0, y0) and (x1, y1).

  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('power', 1, 'Polynomial power.')
    p.Define('start', (0, 1.), '(x0, y0)')
    p.Define('limit', (1, 1.), '(x1, y1)')
    p.Define('origin', 'start', 'Origin of the polynomial. Can be "start" or '
             '"limit".')
    return p

  def Value(self):
    p = self.params
    x = tf.cast(py_utils.GetGlobalStep(), dtype=p.dtype)
    x0, y0 = p.start
    x1, y1 = p.limit

    if x0 >= x1:
      raise ValueError(f'{x0} must be < {x1}')

    x0 = tf.cast(x0, dtype=x.dtype)
    x1 = tf.cast(x1, dtype=x.dtype)
    y0 = tf.cast(y0, dtype=x.dtype)
    y1 = tf.cast(y1, dtype=x.dtype)

    ratio = (x - x0) / (x1 - x0)
    if p.origin == 'start':
      f_x = ratio**p.power
    elif p.origin == 'limit':
      f_x = 1 - (1 - ratio)**p.power
    else:
      raise ValueError('Invalid parameter origin: %s' % p.origin)

    y = y0 + f_x * (y1 - y0)
    return tf.where(x < x0, y0, tf.where(x >= x1, y1, y))


class LinearSchedule(PolynomialSchedule):
  """Linear learning rate schedule.

  If x < x0, returns y0. If x >= x1, returns y1. Otherwise,
  interpolate linearly between (x0, y0) and (x1, y1).

  """

  @classmethod
  def Params(cls):
    return super().Params().Set(power=1)


class AnnealingSchedule(BaseSchedule):
  """Annealing schedule.

  y = max(pow(factor, step), lower_bound)
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('init', None, 'start value.')
    p.Define('lower_bound', None, 'lower bound value.')
    p.Define('factor', None, 'Anealing factor.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.init
    assert p.lower_bound
    assert p.factor and 0 < p.factor <= 1

  def Value(self):
    p = self.params
    x = tf.cast(py_utils.GetGlobalStep(), dtype=p.dtype)
    return tf.math.maximum(tf.pow(p.factor, x), p.lower_bound)


class ExponentialSchedule(BaseSchedule):
  """Linear learning rate schedule.

  If x < x0, returns y0. If x >= x1, returns y1. Otherwise,
  interpolate exponentially between (x0, y0) and (x1, y1).
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('start', (0, 1.), '(x0, y0)')
    p.Define('limit', (1, 0.5), '(x1, y1)')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    x0, y0 = p.start
    x1, y1 = p.limit
    assert x0 < x1, '%s must be < %s' % (x0, x1)
    assert y0 > 0, '%s must be > 0' % y0
    assert y1 > 0, '%s must be > 0' % y1

    self.CreateChild(
        'linear',
        LinearSchedule.Params().Set(
            start=(x0, math.log(y0)), limit=(x1, math.log(y1))))

  def Value(self):
    return tf.exp(self.linear.Value())


class StepwiseExponentialSchedule(BaseSchedule):
  """Exponential decay every N steps."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('decay', 0.99, 'Decay factor.')
    p.Define('num_steps_per_decay', 1000, 'Number of steps between decays.')
    return p

  def Value(self):
    p = self.params
    num_decays = tf.floor(
        tf.div(
            tf.cast(py_utils.GetGlobalStep(), tf.float32),
            float(p.num_steps_per_decay)))
    return tf.pow(p.decay, num_decays)


class CombinedMinimumSchedule(BaseSchedule):
  """Combine a few learning rate decay schedules and takes the min."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('schedules', [LinearSchedule.Params()],
             'A list of learning rate schedule params.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChildren('schedules', p.schedules)

  def Value(self):
    ys = [s.Value() for s in self.schedules]
    return tf.reduce_min(tf.stack(ys), axis=0)


class TransformerSchedule(BaseSchedule):
  """Inverse-decay learning rate until warmup_steps, then decay."""

  @classmethod
  def Params(cls):
    p = super().Params()
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

  def Value(self):
    """Returns the current learning rate decay."""
    p = self.params
    current_step = tf.cast(py_utils.GetGlobalStep(), tf.float32)
    warmup_steps = tf.cast(p.warmup_steps * p.worker_replicas, tf.float32)
    if p.decay_end is not None:
      current_step = tf.where(current_step < p.decay_end, current_step,
                              tf.cast(p.decay_end, tf.float32))
    return p.model_dim**-0.5 * tf.minimum(
        (current_step + 1) * warmup_steps**-1.5, (current_step + 1)**-0.5)


class TransformerMLPerfSchedule(BaseSchedule):
  """learning rate with linear warmup, then rsqrt decay."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'warmup_steps', 4000, 'Increase the learning rate linearly for '
        'the first warmup_steps training steps.')
    p.Define(
        'warmup_init_fraction', 0.,
        'Begin the first step of warm-up with this fraction of the peak LR. '
        'Set to 1 for constant peak LR during the entire warm-up period.')
    p.Define(
        'model_dim', 512, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    return p

  def Value(self):
    """Returns the current learning rate decay."""
    p = self.params
    current_step = tf.cast(py_utils.GetGlobalStep(), tf.float32)
    warmup_steps = tf.cast(p.warmup_steps, tf.float32)
    init_fraction = tf.cast(p.warmup_init_fraction, tf.float32)
    linear_warmup = tf.minimum(
        1.0, init_fraction + (1. - init_fraction) * current_step / warmup_steps)
    rsqrt_decay = tf.math.rsqrt(tf.maximum(current_step, warmup_steps))
    return p.model_dim**-0.5 * linear_warmup * rsqrt_decay


class TransformerScheduleNoWarmUp(BaseSchedule):
  """Fixed learning rate until decay_start, then decay.

  This learning rate schedule is identical to TransformerSchedule
  except in the warm-up phase, where this learning rate schedule uses a fixed
  learning rate (peak-learning rate of TransformerSchedule) for the
  original warm-up phase.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('decay_start', 4000, 'It is used to estimate peak-lr.')
    p.Define('decay_end', None, 'Ends the learning rate decay at '
             'decay_end-th step.')
    p.Define(
        'model_dim', 512, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    p.Define('worker_replicas', 1, 'Number of worker replicas.')
    return p

  def __init__(self, params):
    super().__init__(params)
    tf.logging.info('Peak lr: %f', (self.params.decay_start *
                                    self.params.worker_replicas)**-0.5)

  def Value(self):
    """Returns the current learning rate decay."""
    params = self.params
    warmup_steps = tf.cast(params.decay_start * params.worker_replicas,
                           tf.float32)
    current_step = tf.cast(py_utils.GetGlobalStep(), tf.float32)
    if params.decay_end is not None:
      current_step = tf.where(current_step < params.decay_end, current_step,
                              tf.cast(params.decay_end, tf.float32))
    peak_learning_rate = (warmup_steps**-0.5)
    return (params.model_dim**-0.5) * tf.minimum(
        tf.minimum((current_step + 1),
                   (current_step + 1)**-0.5), peak_learning_rate)


class LinearRampupExponentialDecayScaledByNumSplitSchedule(BaseSchedule):
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
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)

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
      my_cluster = cluster_params.Instantiate()
      splits = my_cluster.num_splits_per_client

    warmup_end = p.warmup * splits
    decay_start = max(warmup_end + 1.0, p.decay_start / splits)
    peak = 1.0 * splits
    tf.logging.info('Peak lr: %f', peak)
    decay_end = max(decay_start + 1.0, p.decay_end / splits)
    schedules = [
        LinearSchedule.Params().Set(
            start=(warmup_end, peak), limit=(decay_start, peak)),
        ExponentialSchedule.Params().Set(
            start=(decay_start, peak), limit=(decay_end, p.min)),
        LinearSchedule.Params().Set(start=(0, p.max), limit=(decay_end, p.max)),
    ]
    # Only include a warm up schedule if the warmup_end exceeds 0.0. Note that
    # linear schedules must have x1 > x0 strictly.
    if warmup_end > 0.0:
      schedules = [
          LinearSchedule.Params().Set(
              start=(0., p.warmup_init), limit=(warmup_end, peak))
      ] + schedules
    self.CreateChild('combine',
                     CombinedMinimumSchedule.Params().Set(schedules=schedules))

  def Value(self):
    return self.combine.Value()


class LinearRampupExponentialDecay(
    LinearRampupExponentialDecayScaledByNumSplitSchedule):
  """A schedule that does the following...

  1. Linearly ramps up from `p.warmup_init` to 1.0 initially;
  2. Stays at constant 1.0 until the decay starts;
  3. Exponential decays from 1.0 to `p.min`.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.num_splits = 1
    p.warmup_init = 0.0
    return p

  def __init__(self, params):
    assert params.num_splits == 1
    super().__init__(params)


class LinearRampupSqrtDecayByBatchSizeAndReplicas(BaseSchedule):
  """Linearly increase learning rate until warmup_examples, then sqrt decay.

  Same as the Transformer learning schedule, except that learning rate
  multiplier and time scales are adjusted according to the aggregate batch size
  (batch_size * num_replicas).

  The implied peak learning rate multiplier given by this schedule is:
      sqrt(batch_size * num_replicas / warmup_examples).

  Can only be used with synchronized trainers.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'warmup_examples', 256 * 2**20,
        'Increase the learning rate linearly for the first warmup_examples '
        'training examples, and then starts square-root decay.')
    p.Define('batch_size', None,
             'Norminal (per-replica) batch size. Must be provided.')
    p.Define(
        'num_replicas', None, 'Number of worker replicas. If None, '
        'determined automatically (and error if this fails).')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.batch_size > 0
    if p.num_replicas:
      self._num_replicas = p.num_replicas
    else:
      # Infer from cluster.
      cluster_params = self.cluster.params.Copy()
      cluster_params.task = 0
      assert cluster_params.mode == 'sync'
      cluster_params.job = 'trainer_client'
      my_cluster = cluster_params.Instantiate()
      self._num_replicas = my_cluster.num_splits_per_client
    assert self._num_replicas > 0

  def Value(self):
    """Returns the current learning rate decay."""
    p = self.params
    current_step = tf.cast(py_utils.GetGlobalStep(), tf.float32)
    warmup_steps = tf.cast(
        p.warmup_examples / (p.batch_size * self._num_replicas), tf.float32)
    return tf.minimum((current_step + 1) * warmup_steps**-1.5,
                      (current_step + 1)**-0.5)


class LinearRampupPiecewiseConstantSchedule(BaseSchedule):
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
    p = super().Params()
    p.Define('boundaries', [], 'Boundaries at which learning rate changes.')
    p.Define('lrs', [], 'A list of learning rate multiplers.')
    p.Define(
        'num_splits', 0, 'Specifies the intended number of num_splits for '
        'LR. Overrides num_splits if non-zero.')
    return p

  def __init__(self, params):
    super().__init__(params)

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
      my_cluster = cluster_params.Instantiate()
      splits = my_cluster.num_splits_per_client

    assert splits >= 1
    splits = float(splits)
    boundaries = [step / splits for step in p.boundaries]
    lrs = [step * splits for step in p.lrs]

    tf.logging.info('splits: {}\n boundaries: {}\n lrs: {} '.format(
        splits, boundaries, lrs))

    schedules = [
        LinearSchedule.Params().Set(
            start=(0., 0.), limit=(boundaries[0], lrs[0])),
        PiecewiseConstantSchedule.Params().Set(
            boundaries=boundaries, values=[1e8] + lrs)
    ]
    self.CreateChild('combine',
                     CombinedMinimumSchedule.Params().Set(schedules=schedules))

  def Value(self):
    return self.combine.Value()


class LinearRampupCosineSchedule(BaseSchedule):
  """A cosine decaying learning rate schedule with a linear rampup phase."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('warmup_init', 0, 'The initial lr value of the warm-up phase.')
    p.Define('warmup_steps', 0, 'Number of warm up steps.')
    p.Define('initial_value', 1.0, 'Initial decay value.')
    p.Define('final_value', 0., 'Final decay value.')
    p.Define('total_steps', 0, 'Number of steps to reach full decay.')
    p.Define(
        'num_splits', 1, 'Specifies the intended number of splits for the '
        'LR. Overrides num_splits_per_client if non-zero. Uses '
        'num_splits_per_client as num_splits if zero or negative.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    # We always compute lr schedule from the trainer's perspective.
    # Also note that this schedule makes sense to sync training only.
    if p.num_splits > 0:
      splits = p.num_splits
    else:
      # Infer num_splits from cluster.
      cluster_params = self.cluster.params.Copy()
      cluster_params.task = 0
      assert cluster_params.mode == 'sync'
      cluster_params.job = 'trainer_client'
      my_cluster = cluster_params.Instantiate()
      splits = my_cluster.num_splits_per_client

    schedules = [
        LinearSchedule.Params().Set(
            start=(0., p.warmup_init * splits),
            limit=(p.warmup_steps // splits, p.initial_value * splits)),
        CosineSchedule.Params().Set(
            initial_value=p.initial_value * splits,
            final_value=p.final_value * splits,
            total_steps=p.total_steps // splits),
    ]
    self.CreateChild('combine',
                     CombinedMinimumSchedule.Params().Set(schedules=schedules))

  def Value(self):
    return self.combine.Value()


class DevBasedSchedule(BaseSchedule):
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
    p = super().Params()
    p.Define('metric_history', early_stop.MetricHistory.Params(),
             'Metric to monitor for stopping.')
    p.Define('tolerance', 0.0, 'Minimum significant difference in metric.')
    p.Define('window', 10000,
             'Steps since most recent decay or best_step before decaying.')
    p.Define('decay', 0.5,
             'Factor by which learning rate multiplier is decayed.')
    p.Define('min_factor', 0.01, 'Minimum learning rate multiplier.')

    return p

  def __init__(self, params):
    super().__init__(params)
    self.SetVariableFree(False)
    p = self.params
    self._metric_history = early_stop.MetricHistory(p.metric_history)

  def _CreateLayerVariables(self):
    p = self.params
    wp = py_utils.WeightParams(
        shape=[],
        init=py_utils.WeightInit.Constant(1.0),
        collections=['DevBasedSchedule_vars'],
        dtype=tf.float32)
    self.CreateVariable('cur_factor', wp, trainable=False)
    wp = py_utils.WeightParams(
        shape=[],
        init=py_utils.WeightInit.Constant(0),
        collections=['DevBasedSchedule_vars'],
        dtype=tf.int64)
    self.CreateVariable('ref_step', wp, trainable=False)
    self._best_step = ops.best_step(self._metric_history.hist_file, p.tolerance)

  def Value(self):
    p = self.params
    with tf.name_scope(p.name):
      steps = self._best_step
      best_step = steps[0]
      last_step = steps[1]

      ref_step = tf.maximum(self.theta.ref_step, best_step)
      f = self.theta.cur_factor

      # Decay if no improvement within window.
      new_factor = tf.where(last_step - ref_step < p.window, f,
                            tf.maximum(p.min_factor, f * p.decay))
      # Update ref_step if we decayed.
      new_step = tf.where(tf.equal(new_factor, f), ref_step, last_step)
      update_step = tf.assign(self.theta.ref_step, new_step)
      with tf.control_dependencies([update_step]):
        return tf.assign(self.theta.cur_factor, new_factor)


class CosineSchedule(BaseSchedule):
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
    p = super().Params()
    p.Define('initial_value', 1.0, 'Initial decay value.')
    p.Define('final_value', 0., 'Final decay value.')
    p.Define('total_steps', 0, 'Number of steps to reach full decay.')
    return p

  def Value(self):
    p = self.params
    assert p.total_steps > 0
    with tf.name_scope(p.name):
      decay_gap = p.initial_value - p.final_value
      return p.final_value + 0.5 * decay_gap * (1 + tf.cos(math.pi * tf.minimum(
          1.0,
          tf.cast(py_utils.GetGlobalStep(), tf.float32) / p.total_steps)))


class PiecewiseSchedule(BaseSchedule):
  """Piecewise schedule composed of sub-schedules."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('boundaries', None, 'Boundaries between subschedules.')
    p.Define(
        'schedules', None, 'A list of sub-schedules. '
        'The length must be len(boundaries) + 1. '
        'schedules[i] starts at boundaries[i-1] (inclusive) and ends at '
        'boundaries[i] (exclusive). '
        'The *relative* step in each interval will be passed to the '
        'sub-schedule for Value.')
    return p

  def __init__(self, params):
    super().__init__(params)
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

  def Value(self):
    p = self.params
    current_step = tf.cast(py_utils.GetGlobalStep(), tf.int64)
    interval_starts = [0] + p.boundaries
    values = []
    for interval_start, schedule in zip(interval_starts, self.schedules):
      relative_step = tf.maximum(
          tf.cast(0, current_step.dtype),
          current_step - tf.cast(interval_start, current_step.dtype))
      with py_utils.GlobalStepContext(relative_step):
        values.append(schedule.Value())

    return py_utils.PiecewiseConstant(current_step, p.boundaries, values,
                                      values[0].dtype)


class SqrtDecay(BaseSchedule):
  """Sqrt decay schedule."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('warmup_steps', 10000, 'Number of warm up steps.')
    p.Define('multiplier', 1.0, 'Multiplier.')
    p.Define('offset', 0.0, 'Offset.')
    return p

  def Value(self):
    p = self.params
    step_num = tf.cast(py_utils.GetGlobalStep(), tf.float32)
    learning_rate = tf.math.rsqrt(
        tf.maximum(step_num - p.offset, p.warmup_steps))
    learning_rate *= p.multiplier
    return learning_rate


class CycleSchedule(BaseSchedule):
  """Piecewise schedule composed of sub-schedules in a cycle."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'schedules', None, 'A list of sub-schedules. Unlike PiecewiseSchedule, '
        'the absolute step is passed to the sub-schedule.')
    p.Define('steps', None, 'The number of steps to run each sub-schedule.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if len(p.schedules) != len(p.steps):
      raise ValueError('len(schedules) != len(steps): %s vs %s' %
                       (len(p.schedules), len(p.steps)))
    self.CreateChildren('schedules', p.schedules)
    boundaries = [0]
    for step in p.steps:
      boundaries.append(boundaries[-1] + step)
    self._period = boundaries[-1]
    self._boundaries = boundaries[1:-1]

  def Value(self):
    values = []
    for schedule in self.schedules:
      values.append(schedule.Value())
    relative_step = tf.math.mod(py_utils.GetGlobalStep(), self._period)
    return py_utils.PiecewiseConstant(relative_step, self._boundaries, values,
                                      values[0].dtype)
