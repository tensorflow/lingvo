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
"""Early stopping based on dev-set performance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from lingvo.core import hyperparams
from lingvo.core.ops import py_x_ops


class MetricHistory(object):
  """Record given metric versus global_step history to a file."""

  @staticmethod
  def SetLogdirInMetricHistories(params, logdir):
    """Set the logdir member in all MetricHistory.params objects in params.

    Args:
      params: global model params.
      logdir: root dir for current run.

    Needs to be called by trainer once the logdir is known, but before
    MetricHistory objects are constructed.
    """
    for _, p in params.IterParams():
      if isinstance(p, hyperparams.Params):
        try:
          p.Get('name')
        except AttributeError:
          pass
        else:
          if p.name == 'MetricHistory':
            p.logdir = logdir
        MetricHistory.SetLogdirInMetricHistories(p, logdir)

  # Global map from jobname + metric key to class.
  _metric_histories_map = {}

  @staticmethod
  def _Key(jobname, metric):
    """Generate a key for _metric_histories_map."""
    return jobname + '.' + metric

  @classmethod
  def Params(cls):
    p = hyperparams.Params()
    p.Define('name', 'MetricHistory', 'Used by SetLogdirInMetricHistories.')
    p.Define('jobname', 'eval_dev', 'Job and dataset to which metric applies.')
    p.Define('metric', 'log_pplx', 'Metric to record.')
    p.Define(
        'minimize', True,
        'If True, training minimizes the metric. If False, training '
        'maximizes the metric.')
    p.Define('logdir', '', 'Root dir for BF logs.')
    p.Define(
        'tfevent_file', False, 'If True, read the metric from '
        'events.out.tfevents.* files in the job dir instead of '
        'maintaining a history file.')
    p.Define('local_filesystem', False,
             'Logdir is on local filesystem (needed for unit test).')
    return p

  def __init__(self, params):
    self.params = params.Copy()
    if params.tfevent_file:
      self._hist_file = os.path.join(params.logdir, params.jobname,
                                     'events.out.tfevents*')
    else:
      fname = params.metric + '.history.txt'
      self._hist_file = os.path.join(params.logdir, params.jobname, fname)
    self._metric_histories_map[self._Key(params.jobname, params.metric)] = self
    self._minimize = params.minimize
    self._metric = params.metric
    self._tfevent_file = params.tfevent_file

  @property
  def hist_file(self):
    return self._hist_file

  @property
  def minimize(self):
    return self._minimize

  @property
  def metric(self):
    return self._metric

  @property
  def tfevent_file(self):
    return self._tfevent_file

  @classmethod
  def ConditionalAppend(cls, jobname, metric, global_step, value):
    """Updates history file iff we are recording given metric and jobname."""
    key = cls._Key(jobname, metric)
    if key in cls._metric_histories_map:
      if cls._metric_histories_map[key].tfevent_file:
        return False
      cls._metric_histories_map[key].Append(global_step, value)
      return True
    else:
      return False

  def Append(self, global_step, value):
    """Updates history file with given record."""
    fname = self._hist_file
    if not self.params.local_filesystem:
      fname += '%r=3.2:sl=8M'
    with tf.gfile.FastGFile(fname, 'a') as f:
      f.write('%d %f\n' % (global_step, value))


class EarlyStop(object):
  """Early stopping based on dev-set performance.

  Factors out the steps needed to perform early stopping in the trainer when a
  selected metric hasn't improved for a given number of steps. If the window
  param is 0 this is guaranteed to be a no-op.
  """

  @classmethod
  def Params(cls):
    p = hyperparams.Params()
    p.Define('name', 'EarlyStop', '')
    p.Define('metric_history', MetricHistory.Params(), 'Metric history params.')
    p.Define(
        'tolerance', 0.0, 'Minimum significant difference in metric; '
        'useful if progress is asymptotic.')
    p.Define('window', 0, 'Maximum number of steps between best and current.')
    p.Define('verbose', True, 'Log early-stop checks.')
    return p

  def __init__(self, params):
    self.params = params.Copy()
    if self.params.window:
      self._metric_history = MetricHistory(self.params.metric_history)
    else:
      self._metric_history = None
    self._node = None
    self._best_step = 0
    self._last_step = 0

  @property
  def metric_history(self):
    return self._metric_history

  @property
  def best_step(self):
    return self._best_step

  @property
  def last_step(self):
    return self._last_step

  def FProp(self, theta):
    """Creates an op to determine the best step from the metric history file.

    Args:
      theta: Not currently used.
    Returns:
      The created op.

    This uses BestStepOp rather than reading the file directly from python in
    order to ensure compatibility with DevBasedSchedule for learning-rate decay.
    It is natural to use dev-based decay and early stopping together, for
    example decaying when dev-set perplexity hasn't improved for n steps, and
    stopping when it hasn't improved for 3n steps.
    """
    del theta  # not used
    if self.params.window:
      self._node = py_x_ops.best_step(
          self.metric_history.hist_file, self.params.tolerance,
          self.metric_history.minimize, self.metric_history.metric)
    else:
      self._node = None
    return self._node

  def Stop(self, session):
    """Returns true if stop criterion is met."""
    if self.params.window and self._node is not None:
      self._best_step, self._last_step = session.run(self._node)
      s = self._last_step - self._best_step > self.params.window
      if self.params.verbose:
        tf.logging.info('early stop check: best_step=%d, last_step=%d, stop=%d',
                        self._best_step, self._last_step, s)
      return s
    else:
      return False
