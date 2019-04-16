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
"""Tests for task_scheduler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import range
import tensorflow as tf
from lingvo.core import early_stop
from lingvo.core import task_scheduler
from lingvo.core import test_utils

_NUMPY_RANDOM_SEED = 9885784


class SchedulerTests(test_utils.TestCase):

  def _TestSchedulerHelper(self, schedule, global_step, count_a):
    np.random.seed(_NUMPY_RANDOM_SEED)

    task_counts = {'a': 0, 'b': 0}
    for _ in range(100):
      task = schedule.Sample(global_step)
      task_counts[task] += 1
    self.assertEqual(task_counts['a'], count_a)
    self.assertEqual(task_counts['b'], 100 - count_a)

  def testConstantScheduler(self):
    """Approximate expected probabilities: (a:0.8, b:0.2)."""
    p = task_scheduler.ConstantScheduler.Params()
    p.task_probs = [('a', 0.8), ('b', 0.2)]

    schedule = p.cls(p)

    self._TestSchedulerHelper(schedule, 0, 83)

  def testExponentialScheduler(self):
    """Test exponential scheduler.

    Approximate probabilities:
      t=0: (a:0, b:1)
      t=1e5: (a:0.63, b:0.37)
      t=1e10: (a:1, b:0)
    """
    p = task_scheduler.ExponentialScheduler.Params()
    p.alpha = 1e-5
    p.task_probs = [('a', (0, 1)), ('b', (1, 0))]

    schedule = p.cls(p)

    self._TestSchedulerHelper(schedule, global_step=0, count_a=0)
    self._TestSchedulerHelper(schedule, global_step=1e5, count_a=63)
    self._TestSchedulerHelper(schedule, global_step=1e10, count_a=100)

  def testSigmoidScheduler(self):
    """Test sigmoid scheduler.

    Approximate probabilities:
      t=0: (a:0.5, b:0.5)
      t=1e5: (a:0.73, b:0.27)
      t=1e10: (a:1, b:0)
    """
    p = task_scheduler.SigmoidScheduler.Params()
    p.alpha = 1e-5
    p.task_probs = [('a', (0.5, 1)), ('b', (0.5, 0))]

    schedule = p.cls(p)

    self._TestSchedulerHelper(schedule, global_step=0, count_a=54)
    self._TestSchedulerHelper(schedule, global_step=1e5, count_a=73)
    self._TestSchedulerHelper(schedule, global_step=1e10, count_a=100)

  def _setupTestAdaptiveScheduler(self, p):
    logdir = tf.test.get_temp_dir()
    tf.gfile.MkDir(os.path.join(logdir, 'decoder_dev_a'))
    tf.gfile.MkDir(os.path.join(logdir, 'decoder_dev_b'))

    early_stop.MetricHistory.SetLogdirInMetricHistories(p, logdir)

    p.epsilon = 0.05
    p.tasks = ['a', 'b']
    p.expected = [0.3, 0.5]

    mh_a = early_stop.MetricHistory.Params()
    mh_a.jobname = 'decoder_dev_a'
    mh_a.metric = 'corpus_bleu'
    mh_a.logdir = logdir
    mh_a.local_filesystem = True

    mh_b = early_stop.MetricHistory.Params()
    mh_b.jobname = 'decoder_dev_b'
    mh_b.metric = 'corpus_bleu'
    mh_b.logdir = logdir
    mh_b.local_filesystem = True

    p.mh_a = mh_a
    p.mh_b = mh_b

    schedule = p.cls(p)

    early_stop.MetricHistory.ConditionalAppend(mh_a.jobname, mh_a.metric, 1,
                                               0.05)
    early_stop.MetricHistory.ConditionalAppend(mh_b.jobname, mh_b.metric, 1,
                                               0.25)

    return schedule

  def testSimpleAdaptiveScheduler(self):
    """Test simple adaptive schedule.

    Probability of task a:
      (1.05 - 0.05/0.3) / ((1.05 - 0.05/0.3) + (1.05 - 0.25/0.5)) /approx 0.616
    """
    np.random.seed(_NUMPY_RANDOM_SEED)

    p = task_scheduler.SimpleAdaptiveScheduler.Params()
    schedule = self._setupTestAdaptiveScheduler(p)

    self._TestSchedulerHelper(schedule, 0, 63)

  def testInverseRatioAdaptiveScheduler(self):
    """Test simple adaptive schedule.

    Probability of task a:
      1.05/(13/60.) / (1.05/(13/60.) + 1.05/(11/20.)) /approx 0.717
    """
    np.random.seed(_NUMPY_RANDOM_SEED)

    p = task_scheduler.InverseRatioAdaptiveScheduler.Params()
    schedule = self._setupTestAdaptiveScheduler(p)
    self._TestSchedulerHelper(schedule, 0, 71)

  def testRoundRobinScheduler(self):
    """Test round-robin scheduler."""
    p = task_scheduler.RoundRobinScheduler.Params()
    p.tasks = ['a', 'b']

    schedule = p.cls(p)
    for global_step in range(20):
      task = schedule.Sample(global_step)
      if global_step % 2 == 0:
        self.assertEqual('a', task)
      else:
        self.assertEqual('b', task)


if __name__ == '__main__':
  tf.test.main()
