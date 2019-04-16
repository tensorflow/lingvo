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
"""Tests for early_stop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from lingvo.core import early_stop
from lingvo.core import hyperparams
from lingvo.core import test_helper
from lingvo.core import test_utils


class MetricHistoryTest(test_utils.TestCase):

  def setUp(self):
    early_stop.MetricHistory._metric_histories_map = {}

  def testSetLogdirInMetricHistories(self):
    p = hyperparams.Params()
    p.Define('name', 'testparams', 'test params')
    p.Define('logdir', None, 'dummy logdir')
    p.Define('mh1', early_stop.MetricHistory.Params(), 'history1')
    p2 = hyperparams.Params()
    p2.Define('mh2', early_stop.MetricHistory.Params(), 'history2')
    p.Define('subparams', p2, 'subparams')

    early_stop.MetricHistory.SetLogdirInMetricHistories(p, 'dir')

    self.assertEqual(p.mh1.logdir, 'dir')
    self.assertEqual(p.subparams.mh2.logdir, 'dir')
    self.assertNotEqual(p.logdir, 'dir')

  def testMetricHistoriesMapUniqueness(self):
    # pylint: disable=unused-variable
    p = early_stop.MetricHistory.Params()
    mh1 = early_stop.MetricHistory(p.Set(jobname='job1', metric='m1'))
    mh2 = early_stop.MetricHistory(p.Set(jobname='job2', metric='m2'))
    mh3 = early_stop.MetricHistory(p.Set(jobname='job1', metric='m1'))

    m = early_stop.MetricHistory._metric_histories_map
    self.assertEqual(len(m), 2)
    self.assertEqual(m[early_stop.MetricHistory._Key('job1', 'm1')], mh3)
    self.assertEqual(m[early_stop.MetricHistory._Key('job2', 'm2')], mh2)

  def testMetricHistoriesFiles(self):
    logdir = tf.test.get_temp_dir()
    tf.gfile.MkDir(os.path.join(logdir, 'job1'))
    tf.gfile.MkDir(os.path.join(logdir, 'job2'))

    p = early_stop.MetricHistory.Params().Set(logdir=logdir)
    mh1 = early_stop.MetricHistory(
        p.Set(jobname='job1', metric='m1', local_filesystem=True))
    mh2 = early_stop.MetricHistory(
        p.Set(jobname='job2', metric='m2', local_filesystem=True))

    early_stop.MetricHistory.ConditionalAppend('job1', 'm1', 1, 10.0)
    early_stop.MetricHistory.ConditionalAppend('job1', 'm2', 1, 10.0)
    early_stop.MetricHistory.ConditionalAppend('job2', 'm2', 1, 10.0)
    early_stop.MetricHistory.ConditionalAppend('job1', 'm1', 2, 5.0)

    self.assertTrue(tf.gfile.Exists(mh1.hist_file))
    self.assertTrue(tf.gfile.Exists(mh2.hist_file))
    with tf.gfile.FastGFile(mh1.hist_file) as f:
      lines = f.readlines()
      self.assertEqual(len(lines), 2)
      self.assertEqual(lines[0].rstrip(), '1 10.000000')
      self.assertEqual(lines[1].rstrip(), '2 5.000000')
    with tf.gfile.FastGFile(mh2.hist_file) as f:
      lines = f.readlines()
      self.assertEqual(len(lines), 1)
      self.assertEqual(lines[0].rstrip(), '1 10.000000')


class EarlyStopTest(test_utils.TestCase):

  def testEarlyStopDefaultIsNoOp(self):
    p = early_stop.EarlyStop.Params()
    es = early_stop.EarlyStop(p)
    es.FProp(None)
    mh = early_stop.MetricHistory
    a = mh.ConditionalAppend(es.params.metric_history.jobname,
                             es.params.metric_history.metric, 1, 10.0)
    s = es.Stop(None)

    self.assertFalse(a)
    self.assertFalse(s)
    self.assertIsNone(es._node)
    self.assertEqual(len(early_stop.MetricHistory._metric_histories_map), 0)

  def testEarlyStopping(self):
    logdir = tf.test.get_temp_dir()
    tf.gfile.MkDir(os.path.join(logdir, 'eval_dev'))

    p = early_stop.EarlyStop.Params()
    p.window = 2
    p.tolerance = 1.0
    p.metric_history.local_filesystem = True
    early_stop.MetricHistory.SetLogdirInMetricHistories(p, logdir)

    es = early_stop.EarlyStop(p)
    es.FProp(None)
    with self.session() as sess:
      jobname = es.metric_history.params.jobname
      metric = es.metric_history.params.metric
      mh = early_stop.MetricHistory

      mh.ConditionalAppend(jobname, metric, 1, 10.0)
      self.assertFalse(es.Stop(sess))
      self.assertEqual(es.best_step, 1)
      self.assertEqual(es.last_step, 1)

      mh.ConditionalAppend(jobname, metric, 2, 5.0)
      self.assertFalse(es.Stop(sess))
      self.assertEqual(es.best_step, 2)
      self.assertEqual(es.last_step, 2)

      mh.ConditionalAppend(jobname, metric, 3, 4.0)
      self.assertFalse(es.Stop(sess))
      self.assertEqual(es.best_step, 2)
      self.assertEqual(es.last_step, 3)

      mh.ConditionalAppend(jobname, metric, 5, 4.0)
      self.assertTrue(es.Stop(sess))
      self.assertEqual(es.best_step, 2)
      self.assertEqual(es.last_step, 5)

  def testEarlyStoppingAscendingMetric(self):
    logdir = tf.test.get_temp_dir()
    tf.gfile.MkDir(os.path.join(logdir, 'decoder_dev'))

    p = early_stop.EarlyStop.Params()
    p.window = 2
    p.tolerance = 1.0
    p.metric_history.local_filesystem = True
    p.metric_history.minimize = False
    p.metric_history.jobname = 'decoder_dev'
    p.metric_history.metric = 'canonical_bleu'
    early_stop.MetricHistory.SetLogdirInMetricHistories(p, logdir)

    es = early_stop.EarlyStop(p)
    es.FProp(None)
    with self.session() as sess:
      jobname = es.metric_history.params.jobname
      metric = es.metric_history.params.metric
      mh = early_stop.MetricHistory

      mh.ConditionalAppend(jobname, metric, 1, 0.0)
      self.assertFalse(es.Stop(sess))
      self.assertEqual(es.best_step, 1)
      self.assertEqual(es.last_step, 1)

      mh.ConditionalAppend(jobname, metric, 2, 1.0)
      self.assertFalse(es.Stop(sess))
      self.assertEqual(es.best_step, 1)
      self.assertEqual(es.last_step, 2)

      mh.ConditionalAppend(jobname, metric, 3, 2.5)
      self.assertFalse(es.Stop(sess))
      self.assertEqual(es.best_step, 3)
      self.assertEqual(es.last_step, 3)

      mh.ConditionalAppend(jobname, metric, 5, 2.0)
      self.assertFalse(es.Stop(sess))
      self.assertEqual(es.best_step, 3)
      self.assertEqual(es.last_step, 5)

      mh.ConditionalAppend(jobname, metric, 6, 1.0)
      self.assertTrue(es.Stop(sess))
      self.assertEqual(es.best_step, 3)
      self.assertEqual(es.last_step, 6)

  def testEarlyStoppingAscendingTfEvents(self):
    logdir = test_helper.test_src_dir_path('core/ops')
    p = early_stop.EarlyStop.Params()
    p.window = 1000
    p.tolerance = 0.0
    p.metric_history.local_filesystem = True
    p.metric_history.minimize = False
    p.metric_history.jobname = 'testdata'
    p.metric_history.metric = 'bleu/dev'
    p.metric_history.tfevent_file = True
    early_stop.MetricHistory.SetLogdirInMetricHistories(p, logdir)

    es = early_stop.EarlyStop(p)
    es.FProp(None)
    with self.session() as sess:
      self.assertTrue(es.Stop(sess))
      self.assertEqual(es.best_step, 102600)
      self.assertEqual(es.last_step, 185200)


if __name__ == '__main__':
  tf.test.main()
