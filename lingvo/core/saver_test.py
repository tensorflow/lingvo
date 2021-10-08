# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for saver."""
import os
import tempfile
import time

from lingvo import compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core import saver
from lingvo.core import test_utils
from lingvo.tasks.image.params import mnist
import numpy as np
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.platform import test
# pylint: enable=g-direct-tensorflow-import


class SaverTest(test_utils.TestCase):

  @staticmethod
  def _buildGraphAndSaver(logdir, keep_latest_n=5, keep_every_n_hours=None):
    g = tf.Graph()
    with g.as_default():
      p = mnist.LeNet5().Task()
      p.input = mnist.LeNet5().Train()
      with cluster_factory.ForTestingWorker(mode='sync', job='controller'):
        _ = p.Instantiate()
      gsv = py_utils.GetOrCreateGlobalStepVar()
      inc = gsv.assign_add(1)
      variables = tf.all_variables()
      sanity_checks = [([gsv], saver.InRange(0, 10))]
      for var in variables:
        sanity_checks.append(([var], saver.IsFinite()))
      sav = saver.Saver(
          logdir,
          variables,
          sanity_checks,
          keep_latest_n=keep_latest_n,
          keep_every_n_hours=keep_every_n_hours)
    return g, sav, inc

  @staticmethod
  def _checkpointIds(logdir):
    filenames = tf.io.gfile.glob('{}/*'.format(logdir))
    print('\n'.join(filenames))
    ckpt_ids = []
    for f in filenames:
      if f.endswith('.meta'):
        ckpt_id = saver.Saver.GetCheckpointId(f)
        ckpt_ids.append(ckpt_id)
    # Sort ascending.
    ckpt_ids.sort()
    return ckpt_ids

  def testBasic(self):
    logdir = tempfile.mkdtemp()
    # Create a dummy file that looks like a checkpoint that shouldn't
    # be touched.
    with tf.io.gfile.GFile(logdir + '/ckpt-foo', 'w') as f:
      f.write('contents')

    g, sav, inc = self._buildGraphAndSaver(logdir)
    with self.session(graph=g) as sess:
      # Creates a few checkpoints.
      sess.run(tf.global_variables_initializer())
      for _ in range(10):
        sess.run(inc)
        _ = sav.Save(sess)

      # Restore to the latest.
      sess.run(tf.global_variables_initializer())
      _ = sav.Restore(sess)

      # Restore to a specific checkpoint.
      sess.run(tf.global_variables_initializer())
      _ = sav.Restore(sess, checkpoint_id=6)

      # Increments global_step out of range, Save() fails.
      for _ in range(5):
        sess.run(inc)
      with self.assertRaises(tf.errors.AbortedError):
        _ = sav.Save(sess)

    filenames = tf.io.gfile.glob('{}/*'.format(logdir))
    filenames = [x[len(logdir) + 1:] for x in filenames]
    print('\n'.join(filenames))
    self.assertIn('checkpoint', filenames)

    meta_files = []
    for f in filenames:
      if f.endswith('.meta'):
        meta_files.append(f)
    # A .meta for each checkpoint.
    self.assertEqual(len(meta_files), 6)

    # 1 for checkpoint. 3 files per checkpoint. 5 good checkpoints, 1 bad.
    # 1 extra file contains the error message, and 1 dummy file
    self.assertEqual(len(filenames), 1 + (5 + 1) * 3 + 1 + 1)

  @test.mock.patch.object(saver, 'time')
  def testBothPolicies(self, mock_time):
    """Test indefinite retention policy and recent policy."""
    fake_time = time.time()
    mock_time.time.return_value = fake_time
    logdir = tempfile.mkdtemp()
    g, sav, inc = self._buildGraphAndSaver(
        logdir, keep_latest_n=2, keep_every_n_hours=5)
    with self.session(graph=g) as sess:
      # Creates a few checkpoints.
      sess.run(tf.global_variables_initializer())
      for _ in range(9):
        sess.run(inc)
        _ = sav.Save(sess)
        # Advance mock time one-ish hour.
        fake_time += 3601.0
        mock_time.time.return_value = fake_time

    ckpt_ids = self._checkpointIds(logdir)

    # 1,6 are kept due to indefinite policy
    # 8,9 due to recent policy.
    self.assertEqual([1, 6, 8, 9], ckpt_ids)

  def testRecentOnlyPreempt(self):
    """Test only recent retention policy when there's pre-emptions."""
    logdir = tempfile.mkdtemp()
    g, sav, inc = self._buildGraphAndSaver(
        logdir, keep_latest_n=5, keep_every_n_hours=None)
    with self.session(graph=g) as sess:
      # Creates a few checkpoints.
      sess.run(tf.global_variables_initializer())
      for _ in range(5):
        sess.run(inc)
        _ = sav.Save(sess)

      # Restore to the latest.
      sess.run(tf.global_variables_initializer())
      _ = sav.Restore(sess)

    # Simulate a pre-emption, create a brand new graph/saver.
    g, sav, inc = self._buildGraphAndSaver(
        logdir, keep_latest_n=5, keep_every_n_hours=None)
    with self.session(graph=g) as sess:
      # Creates a few checkpoints.
      sess.run(tf.global_variables_initializer())
      _ = sav.Restore(sess)
      for _ in range(4):
        sess.run(inc)
        _ = sav.Save(sess)

    ckpt_ids = self._checkpointIds(logdir)
    # Expect only the most recent 5.
    self.assertEqual([5, 6, 7, 8, 9], ckpt_ids)

  def testIndefinitePreempt(self):
    """Test indefinite retention policy when there's pre-emptions."""
    logdir = tempfile.mkdtemp()
    g, sav, inc = self._buildGraphAndSaver(
        logdir, keep_latest_n=0, keep_every_n_hours=1e-9)
    with self.session(graph=g) as sess:
      # Creates a few checkpoints.
      sess.run(tf.global_variables_initializer())
      for _ in range(5):
        sess.run(inc)
        _ = sav.Save(sess)

      # Restore to the latest.
      sess.run(tf.global_variables_initializer())
      _ = sav.Restore(sess)

    # Simulate a pre-emption, create a brand new graph/saver and run
    # a few steps.
    g, sav, inc = self._buildGraphAndSaver(
        logdir, keep_latest_n=1, keep_every_n_hours=1e-9)
    with self.session(graph=g) as sess:
      # Creates a few checkpoints.
      sess.run(tf.global_variables_initializer())
      _ = sav.Restore(sess)
      for _ in range(4):
        sess.run(inc)
        _ = sav.Save(sess)

    ckpt_ids = self._checkpointIds(logdir)

    # We expect all 9 checkpoints.
    self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8, 9], ckpt_ids)

  @test.mock.patch.object(saver, 'time')
  def testBothPoliciesPreempt(self, mock_time):
    """Test indefinite retention policy and recent policy."""
    fake_time = time.time()
    mock_time.time.return_value = fake_time
    logdir = tempfile.mkdtemp()
    g, sav, inc = self._buildGraphAndSaver(
        logdir, keep_latest_n=2, keep_every_n_hours=5)
    with self.session(graph=g) as sess:
      # Creates a few checkpoints.
      sess.run(tf.global_variables_initializer())
      for _ in range(6):
        sess.run(inc)
        _ = sav.Save(sess)
        # Advance mock time one-ish hour.
        fake_time += 3601.0
        mock_time.time.return_value = fake_time

    fake_time += 100000.0
    mock_time.time.return_value = fake_time

    # Simulate a pre-emption, create a brand new graph/saver and run
    # a few steps.
    g, sav, inc = self._buildGraphAndSaver(
        logdir, keep_latest_n=2, keep_every_n_hours=5)
    with self.session(graph=g) as sess:
      # Creates a few checkpoints.
      sess.run(tf.global_variables_initializer())
      _ = sav.Restore(sess)
      for _ in range(4):
        sess.run(inc)
        _ = sav.Save(sess)
        fake_time += 3601.0
        mock_time.time.return_value = fake_time

    ckpt_ids = self._checkpointIds(logdir)

    # 1,6,7 are kept due to indefinite policy
    # 9,10 due to recent policy.
    self.assertEqual([1, 6, 7, 9, 10], ckpt_ids)

  def testSingleCheckpoint(self):
    logdir = tempfile.mkdtemp()
    g = tf.Graph()
    with g.as_default():
      _ = py_utils.GetOrCreateGlobalStepVar()
      sav = saver.Saver(logdir, tf.all_variables(), [], keep_latest_n=1)
    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      _ = sav.Save(sess)

  def testWriteReadNpArrays(self):
    prefix = os.path.join(tempfile.mkdtemp(), 'nptest')
    nmap = py_utils.NestedMap()
    nmap.train = np.random.normal(size=(3, 3))
    nmap.test = np.random.normal(size=(1, 3))
    nmap.foo = py_utils.NestedMap()
    nmap.foo.bar = np.arange(10).astype(np.int32).reshape([2, 5])
    saver.WriteNpArrays(prefix, nmap)
    files = sorted(tf.io.gfile.glob(prefix + '*'))
    self.assertEqual(len(files), 2)
    self.assertEqual(files[0], prefix + '.data-00000-of-00001')
    self.assertEqual(files[1], prefix + '.index')
    read_nmap = saver.ReadNpArrays(prefix, nmap.Transform(lambda x: x.dtype))
    self.assertTrue(nmap.IsCompatible(read_nmap))
    self.assertAllEqual(nmap.train, read_nmap.train)
    self.assertAllEqual(nmap.test, read_nmap.test)
    self.assertAllEqual(nmap.foo.bar, read_nmap.foo.bar)


if __name__ == '__main__':
  tf.test.main()
