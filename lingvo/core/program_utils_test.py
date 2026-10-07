# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for program_utils."""

import os

import lingvo.compat as tf
from lingvo.core import program_utils
from lingvo.core import test_utils


class DecodeStatusCacheTest(test_utils.TestCase):

  def _Summary(self, tag, value):
    return tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=value)])

  def testUpdateDatasetPreservesCheckpointAndDecodedDatasets(self):
    program_dir = self.create_tempdir().full_path
    cache = program_utils.DecodeStatusCache(program_dir)

    cache.UpdateCkpt('ckpt-123')
    cache.UpdateDataset('Dev', {'acc': self._Summary('acc', 0.5)})
    cache.UpdateDataset('Test', {'loss': self._Summary('loss', 1.25)})
    cache.UpdateDataset('Dev', {'acc': self._Summary('acc', 0.75)})

    reloaded_cache = program_utils.DecodeStatusCache(program_dir)

    self.assertEqual('ckpt-123', reloaded_cache.ckpt_key)
    self.assertEqual(['Dev', 'Test'], reloaded_cache.decoded_datasets)

    summaries = reloaded_cache.TryLoadCache('ckpt-123', 'Dev')
    self.assertIsNotNone(summaries)
    self.assertIn('acc', summaries)
    self.assertAlmostEqual(0.75, summaries['acc'].value[0].simple_value)
    self.assertEqual(['Dev', 'Test'], reloaded_cache.decoded_datasets)

    status_file = os.path.join(program_dir, 'decoded_datasets.txt')
    with tf.io.gfile.GFile(status_file, 'r') as f:
      self.assertEqual(['ckpt-123', 'Dev', 'Test'],
                       [line.strip() for line in f.readlines()])


if __name__ == '__main__':
  test_utils.main()
