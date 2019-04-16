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
"""Tests for base_input_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np

import tensorflow as tf
from lingvo.core import base_input_generator
from lingvo.core import test_utils


def _CreateFakeTFRecordFiles(record_count=10):
  tmpdir = tempfile.mkdtemp()
  data_path = os.path.join(tmpdir, 'fake.tfrecord')
  with tf.io.TFRecordWriter(data_path) as w:
    for _ in range(record_count):
      feature = {
          'audio':
              tf.train.Feature(
                  float_list=tf.train.FloatList(
                      value=np.random.uniform(-1.0, 1.0, 48000))),
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      w.write(example.SerializeToString())
  return tmpdir, data_path


class ToyInputGenerator(base_input_generator.BaseDataExampleInputGenerator):

  def GetFeatureSpec(self):
    return {'audio': tf.FixedLenFeature([48000], tf.float32)}


class BaseExampleInputGeneratorTest(test_utils.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def tearDown(self):
    if hasattr(self, '_tmpdir'):
      shutil.rmtree(self._tmpdir)

  def testTfRecordFile(self):
    p = ToyInputGenerator.Params()
    p.batch_size = 2
    self._tmpdir, p.input_files = _CreateFakeTFRecordFiles()
    p.dataset_type = tf.data.TFRecordDataset
    p.randomize_order = False
    p.parallel_readers = 1
    ig = p.cls(p)
    with self.session(graph=tf.get_default_graph()) as sess:
      inputs = ig.InputBatch()
      eval_inputs = sess.run(inputs)
      input_shapes = eval_inputs.Transform(lambda t: t.shape)
      self.assertEqual(input_shapes.audio, (2, 48000))

  def testTfRecordFileLargeBatch(self):
    p = ToyInputGenerator.Params()
    p.batch_size = 200
    self._tmpdir, p.input_files = _CreateFakeTFRecordFiles()
    p.dataset_type = tf.data.TFRecordDataset
    p.randomize_order = False
    p.parallel_readers = 1
    ig = p.cls(p)
    with self.session(graph=tf.get_default_graph()) as sess:
      inputs = ig.InputBatch()
      eval_inputs = sess.run(inputs)
      input_shapes = eval_inputs.Transform(lambda t: t.shape)
      self.assertEqual(input_shapes.audio, (200, 48000))


if __name__ == '__main__':
  tf.test.main()
