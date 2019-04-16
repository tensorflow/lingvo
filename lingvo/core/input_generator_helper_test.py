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
"""Tests for input_generator_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from lingvo.core import input_generator_helper
from lingvo.core import test_utils


class InputGeneratorHelperTest(test_utils.TestCase):

  def testComputeSplitsLessThanNumSplits(self):
    with self.session(use_gpu=False) as sess:
      batch_size = tf.constant(2, dtype=tf.int32)
      num_splits = 4
      splits = input_generator_helper.ComputeSplits(batch_size, num_splits)
      expected = [1, 1, 0, 0]

      actual = sess.run(splits)
      self.assertAllEqual(actual, expected)

  def testComputeSplitsEven(self):
    with self.session(use_gpu=False) as sess:
      batch_size = tf.constant(32, dtype=tf.int32)
      num_splits = 4
      splits = input_generator_helper.ComputeSplits(batch_size, num_splits)
      expected = [8, 8, 8, 8]

      actual = sess.run(splits)
      self.assertAllEqual(actual, expected)

  def testComputeSplitsUnevenOne(self):
    with self.session(use_gpu=False) as sess:
      batch_size = tf.constant(31, dtype=tf.int32)
      num_splits = 4
      splits = input_generator_helper.ComputeSplits(batch_size, num_splits)
      expected = [8, 8, 8, 7]

      actual = sess.run(splits)
      self.assertAllEqual(actual, expected)

  def testComputeSplitsUnevenTwo(self):
    with self.session(use_gpu=False) as sess:
      batch_size = tf.constant(30, dtype=tf.int32)
      num_splits = 4
      splits = input_generator_helper.ComputeSplits(batch_size, num_splits)
      expected = [8, 8, 7, 7]

      actual = sess.run(splits)
      self.assertAllEqual(actual, expected)

  def testComputeSplitsUnevenThree(self):
    with self.session(use_gpu=False) as sess:
      batch_size = tf.constant(29, dtype=tf.int32)
      num_splits = 4
      splits = input_generator_helper.ComputeSplits(batch_size, num_splits)
      expected = [8, 7, 7, 7]

      actual = sess.run(splits)
      self.assertAllEqual(actual, expected)

  def _assertTupleOfListsEqual(self, actual, expected):
    self.assertEqual(len(actual), len(expected))
    for i in range(len(actual)):
      self.assertEqual(len(actual[i]), len(expected[i]))
      for j in range(len(actual[i])):
        self.assertAllEqual(actual[i][j], expected[i][j])

  def _assertListOfDictsEqual(self, actual, expected):
    self.assertEqual(len(actual), len(expected))
    for i in range(len(actual)):
      self.assertListEqual(list(actual[i].keys()), list(expected[i].keys()))
      for k in actual[i].keys():
        self.assertAllEqual(actual[i][k], expected[i][k])

  def testSplitTensorsLessThanNumSplits(self):
    t1 = tf.constant([[1, 2, 3, 4]])
    t2 = tf.constant([[5, 6, 7, 8]])
    t3 = tf.constant([[13, 14, 15, 16]])

    tensor_tuple = (t1, t2, t3)
    num_splits = 2
    splits = input_generator_helper.SplitTensors(tensor_tuple, num_splits)

    with self.session(use_gpu=False) as sess:
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'first dim of tensors in xs must be greater '
                                   'than num_splits'):
        sess.run(splits)

  def testSplitTensorsOne(self):
    with self.session(use_gpu=False) as sess:
      t1 = tf.constant([[1, 2, 3, 4], [4, 5, 6, 7]])
      t2 = tf.constant([[5, 6, 7, 8], [9, 10, 11, 12]])
      t3 = tf.constant([[13, 14, 15, 16], [14, 15, 16, 17]])

      tensor_tuple = (t1, t2, t3)
      num_splits = 1
      splits = input_generator_helper.SplitTensors(tensor_tuple, num_splits)
      expected = ([np.array([[1, 2, 3, 4], [4, 5, 6, 7]])],
                  [np.array([[5, 6, 7, 8], [9, 10, 11, 12]])],
                  [np.array([[13, 14, 15, 16], [14, 15, 16, 17]])])

      actual = sess.run(splits)
      self._assertTupleOfListsEqual(actual, expected)

  def testSplitTensorsEven(self):
    with self.session(use_gpu=False) as sess:
      t1 = tf.constant([[1, 2, 3, 4], [4, 5, 6, 7]])
      t2 = tf.constant([[5, 6, 7, 8], [9, 10, 11, 12]])
      t3 = tf.constant([[13, 14, 15, 16], [14, 15, 16, 17]])

      tensor_tuple = (t1, t2, t3)
      num_splits = 2
      splits = input_generator_helper.SplitTensors(tensor_tuple, num_splits)
      expected = ([np.array([[1, 2, 3, 4]]), np.array([[4, 5, 6, 7]])],
                  [np.array([[5, 6, 7, 8]]), np.array([[9, 10, 11, 12]])],
                  [np.array([[13, 14, 15, 16]]), np.array([[14, 15, 16, 17]])])

      actual = sess.run(splits)
      self._assertTupleOfListsEqual(actual, expected)

  def testSplitTensorsUneven(self):
    with self.session(use_gpu=False) as sess:
      t1 = tf.constant([[1], [4], [8]])
      t2 = tf.constant([[5], [9], [10]])
      t3 = tf.constant([[13], [14], [11]])

      tensor_tuple = (t1, t2, t3)
      num_splits = 2
      splits = input_generator_helper.SplitTensors(tensor_tuple, num_splits)
      expected = ([np.array([[1], [4]]), np.array([[8]])],
                  [np.array([[5], [9]]), np.array([[10]])],
                  [np.array([[13], [14]]), np.array([[11]])])

      actual = sess.run(splits)
      self._assertTupleOfListsEqual(actual, expected)

  def testSplitTensorsAssert(self):
    t1 = tf.constant([[1], [7], [8]])
    t2 = tf.constant([[5], [9], [10]])
    t3 = tf.constant([[13], [14]])

    tensor_tuple = (t1, t2, t3)
    num_splits = 2

    with self.assertRaisesRegexp(
        ValueError, 'can\'t split axis of size 2 into pieces of size \[2,1\]'):
      splits = input_generator_helper.SplitTensors(tensor_tuple, num_splits)

  def testSplitDictOfTensorsEven(self):
    with self.session(use_gpu=False) as sess:
      t1 = tf.constant([[1], [4], [8], [9]])
      t2 = tf.constant([[5], [9], [10], [12]])
      t3 = tf.constant([[13], [14], [11], [15]])

      tensor_dict = {'a': t1, 'b': t2, 'c': t3}
      num_splits = 2
      splits = input_generator_helper.SplitDictOfTensors(tensor_dict,
                                                         num_splits)
      expected = [{
          'a': np.array([[1], [4]]),
          'b': np.array([[5], [9]]),
          'c': np.array([[13], [14]])
      }, {
          'a': np.array([[8], [9]]),
          'b': np.array([[10], [12]]),
          'c': np.array([[11], [15]])
      }]

      actual = sess.run(splits)
      self._assertListOfDictsEqual(actual, expected)

  def testSplitDictOfTensorsUneven(self):
    with self.session(use_gpu=False) as sess:
      t1 = tf.constant([[1], [4], [8]])
      t2 = tf.constant([[5], [9], [10]])
      t3 = tf.constant([[13], [14], [11]])

      tensor_dict = {'a': t1, 'b': t2, 'c': t3}
      num_splits = 2
      splits = input_generator_helper.SplitDictOfTensors(tensor_dict,
                                                         num_splits)
      expected = [{
          'a': np.array([[1], [4]]),
          'b': np.array([[5], [9]]),
          'c': np.array([[13], [14]])
      }, {
          'a': np.array([[8]]),
          'b': np.array([[10]]),
          'c': np.array([[11]])
      }]

      actual = sess.run(splits)
      self._assertListOfDictsEqual(actual, expected)

  def testSplitDictOfTensorsAssert(self):
    t1 = tf.constant([[1], [7], [8]])
    t2 = tf.constant([[5], [9], [10]])
    t3 = tf.constant([[13], [14]])

    tensor_dict = {'a': t1, 'b': t2, 'c': t3}
    num_splits = 2

    with self.assertRaisesRegexp(
        ValueError, 'can\'t split axis of size 2 into pieces of size \[2,1\]'):
      splits = input_generator_helper.SplitDictOfTensors(tensor_dict,
                                                         num_splits)

if __name__ == '__main__':
  tf.test.main()
