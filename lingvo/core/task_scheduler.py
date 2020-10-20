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
"""Multi-task task sampling schedules."""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import early_stop
import numpy as np


class TaskScheduler(base_layer.BaseLayer):
  """Generic multi-task scheduler.

  Subclasses should override the `Sample` method to return a task string given
  a step. All of the task strings as well as additional hyperparameters needed
  by `Sample` should be exposed and stored in the params. `Sample` should also
  update `cur_probs`.
  """

  @classmethod
  def Params(cls):
    """Parameters for this task scheduler."""
    p = super().Params()
    p.name = 'task_scheduler'
    return p

  def __init__(self, params):
    super().__init__(params)
    self.cur_probs = None
    self.SetVariableFree()

  def Sample(self, current_step):
    raise NotImplementedError('Abstract method')


class AdaptiveScheduler(TaskScheduler):
  """Tasks with low scores will be sampled more often.

  Scores are expected to be non-negative. Larger scores are better."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('tasks', [], 'List of tasks')
    p.Define('expected', [], 'List of final expected scores')
    p.Define('mh_a', early_stop.MetricHistory.Params(), '')
    p.Define('mh_b', early_stop.MetricHistory.Params(), '')
    p.Define(
        'epsilon', 0.05, 'Regularizarion term. A large epsilon will lead'
        'to a more uniform task distribution.')
    p.Define('alpha', 1.0, 'Normalized task scores are raised to this power.')
    return p

  def __init__(self, params):
    super().__init__(params)
    if len(self.params.tasks) != 2 or len(self.params.expected) != 2:
      raise ValueError('Only two tasks are supported by this scheduler.')

    if self.params.epsilon < 0:
      raise ValueError('Epsilon should be positive.')

    self.tasks = self.params.tasks

    self.last_scores = [0.0] * 2

    self._metric_histories = [
        early_stop.MetricHistory(self.params.mh_a),
        early_stop.MetricHistory(self.params.mh_b)
    ]

  def getMetricHistories(self):
    # If too slow, consider another implementation.
    # TODO(sebjean) Time file reading and change behaviour if too long.
    for index, mh in enumerate(self._metric_histories):
      try:
        with tf.io.gfile.GFile(mh.hist_file) as f:
          lines = f.readlines()
      except tf.errors.NotFoundError:
        tf.logging.warning('File not found. '
                                'Expected at start of training only.')
        score, lines = 0.0, []
      if lines:
        try:
          score = lines[-1].split()[-1]
        except IndexError:
          tf.logging.warning(
              'IndexError. Your history file may be corrupted.')
          score = 0.0
      self.last_scores[index] = float(score)


class SimpleAdaptiveScheduler(AdaptiveScheduler):
  """Simple adaptive scheduler.

  A task with a normalized score of `s` is approximately weighted as `1 - s`.
  """

  def Sample(self, current_step):
    """Sample a task.

    The unnormalized probability of a task if given by
    1 + epsilon - min(1, score / expected)**alpha.

    Args:
        current_step: Unused.

    Returns:
        str, the name of the sampled task.
    """
    del current_step  # Unused

    self.getMetricHistories()

    alpha, eps = self.params.alpha, self.params.epsilon
    probs = [
        1 + eps - min(1, score / self.params.expected[index])**alpha
        for index, score in enumerate(self.last_scores)
    ]
    probs = tuple(probs / np.sum(probs))
    sampled_task = np.random.choice(self.params.tasks, p=probs)
    self.cur_probs = probs
    return sampled_task


class InverseRatioAdaptiveScheduler(AdaptiveScheduler):
  """Inverse ratio adaptive scheduler.

  Tasks are approximately weighed as the inverse of their normalized scores.
  """

  def Sample(self, current_step):
    """Sample a task.

    The unnormalized probability of a task if given by
    1 / (min(1, score / expected)**alpha + epsilon)

    Args:
        current_step: Unused.

    Returns:
        str, the name of the sampled task.
    """
    del current_step  # Unused

    self.getMetricHistories()

    alpha, eps = self.params.alpha, self.params.epsilon
    probs = [
        1.0 / (min(1, score / self.params.expected[index])**alpha + eps)
        for index, score in enumerate(self.last_scores)
    ]
    probs = tuple(probs / np.sum(probs))
    sampled_task = np.random.choice(self.params.tasks, p=probs)
    self.cur_probs = probs
    return sampled_task


class ShiftedExponentialScheduler(TaskScheduler):
  """The unnormalized score of each task follows a shifted exponential function.

  Generalizes the constant, exponential and sigmoid
  schedules described in "Scheduled Multi-Task Learning: From Syntax to
  Translation" (Kiperwasser and Ballesteros).
  https://arxiv.org/pdf/1804.08915.pdf
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'alpha', 0, 'Controls the rate at which the schedule changes. '
        'A large alpha will lead to fast convergence toward final values.')
    p.Define(
        'task_probs', [], 'List of 2-tuples (task, prob). For non-constant'
        'schedulers, prob is a tuple of the form (init_prob, final_prob).')
    return p

  def __init__(self, params):
    super().__init__(params)
    assert isinstance(self.params.task_probs, list)
    self.tasks = []
    self._descriptors = []

  def Sample(self, current_step):
    """Sample a task.

    Given an input [a, b] and a rate `alpha`, the unnormalized
    score of eack task is a + b * exp(-alpha * t).

    Args:
        current_step: int. Current time step.

    Returns:
        str, the name of the sampled task.
    """
    probs = [
        a + b * np.exp(-self.params.alpha * current_step)
        for a, b in self._descriptors
    ]
    probs = tuple(probs / np.sum(probs))
    sampled_task = np.random.choice(self.tasks, p=probs)
    self.cur_probs = probs
    return sampled_task


class ConstantScheduler(ShiftedExponentialScheduler):
  """Constant schedule. Tasks are sampled from a fixed probability distribution.
  """

  def __init__(self, params):
    super().__init__(params)

    for key, value in self.params.task_probs:
      self.tasks.append(key)
      self._descriptors.append((value, 0))


class ExponentialScheduler(ShiftedExponentialScheduler):
  """Exponential schedule.

  For a task with initial and final probabilities p_0 and p_1 respectively,
  its unnormalized score is given by
  `p_1 + (p_0 - p_1) * exp(-alpha * current_step)`.
  """

  def __init__(self, params):
    super().__init__(params)

    for key, value in self.params.task_probs:
      self.tasks.append(key)
      self._descriptors.append((value[1], value[0] - value[1]))


class SigmoidScheduler(ShiftedExponentialScheduler):
  """Sigmoid schedule.

  For a task with initial and final probabilities p_0 and p_1 respectively,
  its unnormalized score is given by
  `p_1 + (2 * p_0 - p_1) * exp(-alpha * current_step)`.
  """

  def __init__(self, params):
    super().__init__(params)

    for key, value in self.params.task_probs:
      self.tasks.append(key)
      self._descriptors.append((value[1], 2 * value[0] - value[1]))


class RoundRobinScheduler(TaskScheduler):
  """Deterministic sequential schedule."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('tasks', [], 'List of task names. No repetitions allowed.')
    return p

  def __init__(self, params):
    super().__init__(params)
    assert isinstance(self.params.tasks, list)
    self.tasks = sorted(self.params.tasks)
    self.n_tasks = len(self.tasks)
    self.cur_probs = [1. / self.n_tasks] * self.n_tasks  # For summary
    self.next_task_idx = 0

  def Sample(self, current_step):
    """Sample a task."""
    sampled_task = self.tasks[self.next_task_idx]
    self.next_task_idx = (self.next_task_idx + 1) % self.n_tasks
    return sampled_task


class SequentialScheduler(TaskScheduler):
  """Deterministic schedule that stays a fixed number of steps on each task."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'task_steps', [], 'List of tuples of (task_name, steps_for_task). Goes '
        'through list sequentially in the specified order, staying '
        'steps_for_task steps on task_name. On completing the schedule, '
        'remains on the final task for the rest of the time. Assumes '
        'p.task_global_step is False.')
    return p

  def __init__(self, params):
    super().__init__(params)
    assert isinstance(self.params.task_steps, list)
    assert self.params.task_steps
    self.task_steps = []
    for (name, steps) in self.params.task_steps:
      assert steps > 0
      if self.task_steps:
        self.task_steps.append((name, steps + self.task_steps[-1][1]))
      else:
        self.task_steps.append((name, steps))
    self.n_tasks = len(self.task_steps)
    self.task_idx = 0
    self.cur_probs = [1] + [0] * (self.n_tasks - 1)  # For summary

  def Sample(self, current_step):
    """Sample a task."""
    sampled_task, to_step = self.task_steps[self.task_idx]
    if current_step >= to_step and self.task_idx < self.n_tasks - 1:
      self.task_idx += 1
      sampled_task = self.task_steps[self.task_idx][0]
      self.cur_probs[self.task_idx - 1] = 0
      self.cur_probs[self.task_idx] = 1
    return sampled_task


class PieceWiseScheduler(TaskScheduler):
  """Piecewise scheduler using different scheduling strategies."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'schedule_steps', [], 'List of tuples of (schedule_class_params, '
        'number of steps to use this schedule class)')
    return p

  def __init__(self, params):
    super().__init__(params)
    assert isinstance(self.params.schedule_steps, list)
    self.schedule_steps = []
    self.schedule_params = []
    for (cls_params, steps) in self.params.schedule_steps:
      if self.schedule_steps:
        self.schedule_steps.append(steps + self.schedule_steps[-1])
      else:
        self.schedule_steps.append(steps)
      self.schedule_params.append(cls_params)

    self.CreateChildren('schedules', self.schedule_params)

    self.n_schedules = len(self.schedule_steps)
    self.schedule_idx = 0
    self.task_step_offset = 0
    self.cur_probs = self.schedules[0].cur_probs

  def Sample(self, current_step):
    """Sample a task."""

    to_step = self.schedule_steps[self.schedule_idx]

    if current_step >= to_step and self.schedule_idx < self.n_schedules - 1:
      self.task_step_offset = to_step
      self.schedule_idx += 1

    cur_schedule = self.schedules[self.schedule_idx]
    sampled_task = cur_schedule.Sample(current_step - self.task_step_offset)
    self.cur_probs = cur_schedule.cur_probs

    return sampled_task
