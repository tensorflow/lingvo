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
"""Tests for input_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

import numpy as np
from six.moves import range
import tensorflow as tf
from lingvo.core import test_utils
from lingvo.tasks.image import input_generator


class InputGeneratorTest(test_utils.TestCase):

  def setUp(self):
    self._tmpdir, self.data_path = input_generator.FakeMnistData()

  def tearDown(self):
    shutil.rmtree(self._tmpdir)

  def _trainInput(self):
    p = input_generator.MnistTrainInput.Params()
    p.ckpt = self.data_path
    p.batch_size = 100
    return p

  def _testInput(self):
    p = input_generator.MnistTestInput.Params()
    p.ckpt = self.data_path
    return p

  def testMnistTrain(self):
    p = self._trainInput()
    with self.session() as sess:
      inp = p.cls(p)
      inp_batch = inp.InputBatch()
      for _ in range(10):
        batch = sess.run(inp_batch)
        self.assertEqual(batch.data.shape, (100, 28, 28, 1))
        self.assertEqual(batch.data.dtype, np.float32)
        self.assertEqual(batch.label.shape, (100,))
        self.assertEqual(batch.label.dtype, np.float32)

  def testMnistTest(self):
    p = self._testInput()
    with self.session() as sess:
      inp = p.cls(p)
      inp_batch = inp.InputBatch()
      ids = []
      for _ in range(39):
        batch = sess.run(inp_batch)
        self.assertEqual(batch.data.shape, (256, 28, 28, 1))
        self.assertEqual(batch.data.dtype, np.float32)
        self.assertEqual(batch.label.shape, (256,))
        self.assertEqual(batch.label.dtype, np.float32)
        ids += batch.sample_ids.tolist()
      batch = sess.run(inp_batch)
      self.assertEqual(batch.data.shape, (256, 28, 28, 1))
      self.assertEqual(batch.data.dtype, np.float32)
      self.assertEqual(batch.label.shape, (256,))
      self.assertEqual(batch.label.dtype, np.float32)
      ids += batch.sample_ids.tolist()
      self.assertEqual(list(range(p.num_samples)),
                       sorted(ids))  # Exactly 1 epoch.

      # repeat=False. We should see OutOfRange error.
      with self.assertRaises(tf.errors.OutOfRangeError):
        _ = sess.run(inp_batch)

  def _GetIds(self, sess, p, sample_ids):
    """Goes through one epoch of inp and returns the sample ids."""
    iters = int((p.num_samples + p.batch_size - 1) / p.batch_size)
    ids = []
    for _ in range(iters):
      ids += sess.run(sample_ids).tolist()
    return ids

  def testMnistTrainRandomness(self):
    p = self._trainInput()
    with self.session() as sess:
      inp = p.cls(p)
      batch = inp.InputBatch()
      epoch0 = self._GetIds(sess, p, batch.sample_ids)
      epoch1 = self._GetIds(sess, p, batch.sample_ids)
      self.assertEqual(list(range(p.num_samples)), sorted(epoch0))
      self.assertEqual(list(range(p.num_samples)), sorted(epoch1))
      self.assertNotEqual(epoch0, epoch1)


if __name__ == '__main__':
  tf.test.main()
