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
"""Defines trials for parameter exploration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from lingvo.core import hyperparams


class Trial(object):
  """Base class for a trial."""

  @classmethod
  def Params(cls):
    """Default parameters for a trial."""
    p = hyperparams.Params()
    p.Define(
        'report_interval_seconds', 600,
        'Interval between reporting trial results and checking for early '
        'stopping.')
    p.Define('vizier_objective_metric_key', 'loss',
             'Which eval metric to use as the "objective value" for tuning.')
    p.Define(
        'report_during_training', False,
        'Whether to report objective metrics during the training process.')
    return p

  def __init__(self, params):
    self._params = params.Copy()
    self._next_report_time = time.time()

  @property
  def report_interval_seconds(self):
    return self._params.report_interval_seconds

  @property
  def objective_metric_key(self):
    return self._params.vizier_objective_metric_key

  def Name(self):
    raise NotImplementedError('Abstract method')

  def OverrideModelParams(self, model_params):
    """Modifies `model_params` according to trial params.

    Through this method a `Trial` may tweak model hyperparams (e.g., learning
    rate, shape, depth, or width of networks).

    Args:
      model_params: the original model hyperparams.

    Returns:
      The modified `model_params`.
    """
    raise NotImplementedError('Abstract method')

  def ShouldStop(self):
    """Returns whether the trial should stop."""
    raise NotImplementedError('Abstract method')

  def ReportDone(self, infeasible=False, infeasible_reason=''):
    """Report that the trial is completed."""
    raise NotImplementedError('Abstract method')

  def ShouldStopAndMaybeReport(self, global_step, metrics_dict):
    """Returns whether the trial should stop.

    Args:
      global_step: The global step counter.
      metrics_dict: If not None, contains the metric should be
        reported. If None, do nothing but returns whether the
        trial should stop.
    """
    if not metrics_dict or not self._params.report_during_training:
      return self.ShouldStop()
    if time.time() < self._next_report_time:
      return False
    self._next_report_time = time.time() + self.report_interval_seconds
    return self._DoReportTrainingProgress(global_step, metrics_dict)

  def _DoReportTrainingProgress(self, global_step, metrics_dict):
    raise NotImplementedError('Abstract method')

  def ReportEvalMeasure(self, global_step, metrics_dict, checkpoint_path):
    """Reports eval measurement and returns whether the trial should stop."""
    raise NotImplementedError('Abstract method')


class NoOpTrial(Trial):
  """A Trial implementation that does nothing."""

  def __init__(self):
    super(NoOpTrial, self).__init__(Trial.Params())

  def Name(self):
    return ''

  def OverrideModelParams(self, model_params):
    return model_params

  def ShouldStop(self):
    return False

  def ReportDone(self, infeasible=False, infeasible_reason=''):
    return False

  def ShouldStopAndMaybeReport(self, global_step, metrics_dict):
    del global_step, metrics_dict  # Unused
    return False

  def ReportEvalMeasure(self, global_step, metrics_dict, checkpoint_path):
    del global_step, metrics_dict, checkpoint_path  # Unused
    return False
