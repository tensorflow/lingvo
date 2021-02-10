# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Learning rate schedule utility functions."""

import math

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core import schedule


def _GetTrainingStatistics(train_input_p):
  """Get training statistics, including total batch size and steps per epoch."""
  cluster = cluster_factory.Current()
  # E.g., this is 1 for a single GPU, 8 for a 2x2 TPU, 32 for a 4x4 TPU,
  # or 0 if no training job is launched.
  total_num_cores = cluster.total_worker_devices
  total_batch_size = max(train_input_p.batch_size * total_num_cores, 1)
  steps_per_epoch = float(train_input_p.num_samples) / total_batch_size
  tf.logging.info('#cores = %d batch size = %d steps/epoch = %d',
                  total_num_cores, total_batch_size, steps_per_epoch)
  return py_utils.NestedMap(
      total_num_cores=total_num_cores,
      total_batch_size=total_batch_size,
      steps_per_epoch=steps_per_epoch)


def _GetSteps(train_stats, warmup_epoch, start_epoch, total_epoch):
  """Convert epochs to steps based on train_stats."""
  warmup_steps = warmup_epoch * train_stats.steps_per_epoch
  start_steps = start_epoch * train_stats.steps_per_epoch
  total_steps = total_epoch * train_stats.steps_per_epoch
  tf.logging.info('warmup_steps = %d start_steps = %d total_steps = %d ',
                  warmup_steps, start_steps, total_steps)
  assert 0. <= warmup_steps <= start_steps <= total_steps
  return warmup_steps, start_steps, total_steps


def SetExponentialLR(train_p,
                     train_input_p,
                     exp_start_epoch,
                     total_epoch,
                     warmup_epoch=0,
                     limit_epoch=None,
                     multiplier_min=0.01,
                     warmup_init=0.):
  """Sets a linear rampup and exponential decay LR schedule on train_p.

  This is a wrapper around LinearRampupExponentialDecayScaledByNumSplitSchedule
  that sets the steps using epochs and the training statistics.

  Args:
    train_p: train parameters.
    train_input_p: The training set input parameters.
    exp_start_epoch: The start epoch of exponential annealing.
    total_epoch: Total number of epoch to train.
    warmup_epoch: Epoch for the warm up ramp to end at. Note that the learning
      rate will be fixed between the end of the warmup phase and the beginning
      of the exponential annealing phase.
    limit_epoch: Epoch to end exponential annealing. If None, this will be set
      to 0.95 * total_epoch, that is, the last 5% of training time will be at
      the minimum learning rate.
    multiplier_min: The multiplier minimum at the end of exponential decay.
    warmup_init: Initial value for the warmup phase. Note that warm up can be
      disabled by either setting warmup_init to 1 or setting warmup_epoch to 0.
  """

  # Determine steps based on the training statistics, since the number of steps
  # depends on the number of examples per step.
  train_stats = _GetTrainingStatistics(train_input_p)
  warmup_steps, exp_start_steps, total_steps = _GetSteps(
      train_stats, warmup_epoch, exp_start_epoch, total_epoch)

  if limit_epoch is None:
    limit_epoch = 0.95 * total_epoch
  limit_steps = limit_epoch * train_stats.steps_per_epoch
  tf.logging.info('limit_steps = %d', limit_steps)

  assert 0. <= warmup_steps <= exp_start_steps <= limit_steps <= total_steps

  # Ensure that warmup is disabled by also setting warmup_init to 1 if
  # warmup_epoch is set to 0.
  if warmup_epoch == 0.:
    warmup_init = 1.

  train_p.max_steps = math.ceil(total_epoch * train_stats.steps_per_epoch)
  train_p.lr_schedule = (
      schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params())
  train_p.lr_schedule.Set(
      warmup=warmup_steps,
      decay_start=exp_start_steps,
      decay_end=limit_steps,
      min=multiplier_min,
      warmup_init=warmup_init,
      # Set num_splits to 1 so that no further scaling is done.
      num_splits=1)


def SetCosineLR(train_p,
                train_input_p,
                total_epoch,
                warmup_epoch=0,
                warmup_init=0.):
  """Sets a linear rampup and cosine decay LR schedule on train_p.

  This is a wrapper around LinearRampupCosineSchedule that sets the steps using
  epochs and the training statistics.

  Note: LinearRampupCosineSchedule takes a minimum of the linear ramp and the
  cosine decay. It does not actually reach the peak of the ramp (at 1.0) before
  decaying.

  Args:
    train_p: train parameters.
    train_input_p: The training set input parameters.
    total_epoch: Total number of epoch to train.
    warmup_epoch: Epoch for the warm up ramp to end at. Note that the learning
      rate will transition directly to the cosine decay at the end of the ramp.
    warmup_init: Initial value for the warmup phase. Note that warm up can be
      disabled by either setting warmup_init to 1 or setting warmup_epoch to 0.
  """

  # Determine steps based on the training statistics, since the number of steps
  # depends on the number of examples per step.
  train_stats = _GetTrainingStatistics(train_input_p)
  warmup_steps, _, total_steps = _GetSteps(train_stats, warmup_epoch,
                                           warmup_epoch, total_epoch)

  # Ensure that warmup is disabled by also setting warmup_init to 1 if
  # warmup_epoch is set to 0.
  if warmup_epoch == 0.:
    warmup_init = 1.

  train_p.max_steps = math.ceil(total_epoch * train_stats.steps_per_epoch)
  train_p.lr_schedule = schedule.LinearRampupCosineSchedule.Params()
  train_p.lr_schedule.Set(
      warmup_steps=warmup_steps,
      warmup_init=warmup_init,
      total_steps=total_steps,
      # Set num_splits to 1 so that no further scaling is done.
      num_splits=1)
