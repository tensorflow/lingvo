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
"""Tests for generic_input_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from six.moves import range

import tensorflow as tf
from lingvo.core.ops import py_x_ops


class GenericInputOpTest(tf.test.TestCase):

  def testBasic(self):
    # Generate a test file w/ 100 records.
    tmp = os.path.join(tf.test.get_temp_dir(), 'basic')
    with tf.python_io.TFRecordWriter(tmp) as w:
      for i in range(100):
        w.write('%08d' % i)

    g = tf.Graph()
    with g.as_default():

      # A simple string parsing routine. Just convert a string to a
      # number.
      def str_to_num(s):
        return np.array(float(s), dtype=np.float32)

      # A record processor written in TF graph.
      def _process(record):
        num, = tf.py_func(str_to_num, [record], [tf.float32])
        return record, tf.stack([num, tf.square(num)]), tf.to_int32(1)

      # Samples random records from the data files and processes them
      # to generate batches.
      strs, vals = py_x_ops.generic_input(
          file_pattern='tfrecord:' + tmp,
          file_random_seed=0,
          file_buffer_size=32,
          file_parallelism=4,
          bucket_upper_bound=[1],
          bucket_batch_limit=[8],
          processor=_process)

    with self.test_session(graph=g) as sess:
      record_seen = set()
      for i in range(100):
        ans_strs, ans_vals = sess.run([strs, vals])
        for s in ans_strs:
          record_seen.add(s)
        self.assertEqual(ans_strs.shape, (8,))
        self.assertEqual(ans_vals.shape, (8, 2))
        self.assertAllEqual(np.square(ans_vals[:, 0]), ans_vals[:, 1])
      for i in range(100):
        self.assertTrue('%08d' % i in record_seen)


if __name__ == '__main__':
  tf.test.main()
