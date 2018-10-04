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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import early_stop


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
    p = super(TaskScheduler, cls).Params()
    p.name = 'task_scheduler'
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TaskScheduler, self).__init__(params)
    self.cur_probs = None

  def Sample(self, current_step):
    raise NotImplementedError('Abstract method')


class AdaptiveScheduler(TaskScheduler):
  """Tasks with low scores will be sampled more often.

  Scores are expected to be non-negative. Larger scores are better."""

  @classmethod
  def Params(cls):
    p = super(AdaptiveScheduler, cls).Params()
    p.Define('tasks', [], 'List of tasks')
    p.Define('expected', [], 'List of final expected scores')
    p.Define('mh_a', early_stop.MetricHistory.Params(), '')
    p.Define('mh_b', early_stop.MetricHistory.Params(), '')
    p.Define(
        'epsilon', 0.05, 'Regularizarion term. A large epsilon will lead'
        'to a more uniform task distribution.')
    p.Define('alpha', 1.0, 'Normalized task scores are raised to this power.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(AdaptiveScheduler, self).__init__(params)
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
        with tf.gfile.FastGFile(mh.hist_file) as f:
          lines = f.readlines()
      except tf.errors.NotFoundError:
        tf.logging.warning('File not found. '
                           'Expected at start of training only.')
        score, lines = 0.0, []
      if lines:
        try:
          score = lines[-1].split()[-1]
        except IndexError:
          tf.logging.warning('IndexError. Your history file may be corrupted.')
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
    p = super(ShiftedExponentialScheduler, cls).Params()
    p.Define(
        'alpha', 0, 'Controls the rate at which the schedule changes. '
        'A large alpha will lead to fast convergence toward final values.')
    p.Define(
        'task_probs', [], 'List of 2-tuples (task, prob). For non-constant'
        'schedulers, prob is a tuble of the form (init_prob, final_prob).')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ShiftedExponentialScheduler, self).__init__(params)
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

  @base_layer.initializer
  def __init__(self, params):
    super(ConstantScheduler, self).__init__(params)

    for key, value in self.params.task_probs:
      self.tasks.append(key)
      self._descriptors.append((value, 0))


class ExponentialScheduler(ShiftedExponentialScheduler):
  """Exponential schedule.

  For a task with initial and final probabilities p_0 and p_1 respectively,
  its unnormalized score is given by
  `p_1 + (p_0 - p_1) * exp(-alpha * current_step)`.
  """

  @base_layer.initializer
  def __init__(self, params):
    super(ExponentialScheduler, self).__init__(params)

    for key, value in self.params.task_probs:
      self.tasks.append(key)
      self._descriptors.append((value[1], value[0] - value[1]))


class SigmoidScheduler(ShiftedExponentialScheduler):
  """Sigmoid schedule.

  For a task with initial and final probabilities p_0 and p_1 respectively,
  its unnormalized score is given by
  `p_1 + (2 * p_0 - p_1) * exp(-alpha * current_step)`.
  """

  @base_layer.initializer
  def __init__(self, params):
    super(SigmoidScheduler, self).__init__(params)

    for key, value in self.params.task_probs:
      self.tasks.append(key)
      self._descriptors.append((value[1], 2 * value[0] - value[1]))


class RoundRobinScheduler(TaskScheduler):
  """Deterministic sequential schedule."""

  @classmethod
  def Params(cls):
    p = super(RoundRobinScheduler, cls).Params()
    p.Define('tasks', [], 'List of task names. No repetitions allowed.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RoundRobinScheduler, self).__init__(params)
    assert isinstance(self.params.tasks, list)
    self.tasks = sorted(self.params.tasks)
    self.n_tasks = len(self.tasks)
    self.cur_probs = [1. / self.n_tasks] * self.n_tasks  # For summary

  def Sample(self, current_step):
    """Sample a task."""
    sampled_task = self.tasks[current_step % self.n_tasks]
    return sampled_task
