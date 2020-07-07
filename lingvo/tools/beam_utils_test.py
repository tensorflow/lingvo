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
"""Tests for beam_utils."""

import apache_beam as beam
from lingvo import compat as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tools import beam_utils


class BeamUtilsTest(test_utils.TestCase):

  def testReaders(self):
    pattern = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.tfrecord')
    _ = beam_utils.GetReader(
        'tfrecord',
        pattern,
        value_coder=beam.coders.ProtoCoder(tf.train.Example))

    with self.assertRaises(ValueError):
      _ = beam_utils.GetReader(
          'unknown',
          '/tmp/foo',
          value_coder=beam.coders.ProtoCoder(tf.train.Example))

  def testWriters(self):
    _ = beam_utils.GetWriter(
        'tfrecord',
        '/tmp/foo@1',
        value_coder=beam.coders.ProtoCoder(tf.train.Example))

    with self.assertRaises(ValueError):
      _ = beam_utils.GetWriter(
          'unknown',
          '/tmp/foo@1',
          value_coder=beam.coders.ProtoCoder(tf.train.Example))

  def testGetPipelineRoot(self):
    with beam_utils.GetPipelineRoot() as root:
      _ = root | beam.Create([1, 2, 3]) | beam.Map(lambda x: x)

  def testGetEmitterFn(self):
    _ = beam_utils.GetEmitterFn('tfrecord')
    with self.assertRaises(ValueError):
      _ = beam_utils.GetEmitterFn('unknown')


if __name__ == '__main__':
  tf.test.main()
