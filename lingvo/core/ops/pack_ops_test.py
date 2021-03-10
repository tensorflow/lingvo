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
# ==============================================================================
"""Tests for pack_ops."""

import collections

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import ops
from lingvo.core import test_utils
import numpy as np

PackSequenceTestCase = collections.namedtuple('PackSequenceTestCase', [
    'src_actual_seq_len', 'tgt_actual_seq_len', 'packed_batch_size',
    'packed_src_seq_len', 'packed_tgt_seq_len', 'src_segment_ids',
    'src_segment_pos', 'src_indices_in_input', 'tgt_segment_ids',
    'tgt_segment_pos', 'tgt_indices_in_input'
])

# Set default to empty.
PackSequenceTestCase.__new__.__defaults__ = (None,) * len(
    PackSequenceTestCase._fields)

ApplyPackingTestCase = collections.namedtuple(
    'ApplyPackingTestCase',
    ['input', 'padding', 'segment_ids', 'indices_in_input', 'output'])

# Set default to empty.
ApplyPackingTestCase.__new__.__defaults__ = (None,) * len(
    ApplyPackingTestCase._fields)


def FindResultFromList(result, test_cases):
  """Whether a result is among a list of possible results.

  Args:
    result: A ops.PackSequences object that is the result of running
      ops.pack_sequences().
    test_cases: A list of PackSequenceTestCase.

  Returns:
    The index of first match found, or None for not found.
  """
  for idx, test_case in enumerate(test_cases):
    match = True
    for attr in [
        'src_segment_ids', 'src_segment_pos', 'src_indices_in_input',
        'tgt_segment_ids', 'tgt_segment_pos', 'tgt_indices_in_input'
    ]:
      if not np.array_equal(getattr(result, attr), getattr(test_case, attr)):
        match = False
        break
    if not match:
      continue
    return idx
  return None


class PackSequencesOpTest(test_utils.TestCase):

  def testPackSequences(self):
    test_cases = {
        'Basic':
            PackSequenceTestCase([1, 2, 1], [1, 2, 1], 2, 5, 5,
                                 [[1, 2, 0, 0, 0], [1, 1, 0, 0, 0]],
                                 [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
                                 [[0, 2, 0, 0, 0], [1, 1, 0, 0, 0]],
                                 [[1, 2, 0, 0, 0], [1, 1, 0, 0, 0]],
                                 [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
                                 [[0, 2, 0, 0, 0], [1, 1, 0, 0, 0]]),
        'SpreadFirstN':
            PackSequenceTestCase([3, 1, 2], [4, 2, 1], 2, 5, 5,
                                 [[1, 1, 1, 2, 2], [1, 0, 0, 0, 0]],
                                 [[0, 1, 2, 0, 1], [0, 0, 0, 0, 0]],
                                 [[0, 0, 0, 2, 2], [1, 0, 0, 0, 0]],
                                 [[1, 1, 1, 1, 2], [1, 1, 0, 0, 0]],
                                 [[0, 1, 2, 3, 0], [0, 1, 0, 0, 0]],
                                 [[0, 0, 0, 0, 2], [1, 1, 0, 0, 0]]),
        'DifferentSrcTgtLengths':
            PackSequenceTestCase([3, 2, 1], [4, 1, 5], 2, 4, 6,
                                 [[1, 1, 1, 0], [1, 1, 2, 0]],
                                 [[0, 1, 2, 0], [0, 1, 0, 0]],
                                 [[0, 0, 0, 0], [1, 1, 2, 0]],
                                 [[1, 1, 1, 1, 0, 0], [1, 2, 2, 2, 2, 2]],
                                 [[0, 1, 2, 3, 0, 0], [0, 0, 1, 2, 3, 4]],
                                 [[0, 0, 0, 0, 0, 0], [1, 2, 2, 2, 2, 2]]),
        'Padding':
            PackSequenceTestCase([1], [2], 3, 3, 3,
                                 [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
                                 [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        'DroppingInputsTooLong':
            PackSequenceTestCase([1, 3, 1], [4, 1, 2], 2, 2, 3,
                                 [[1, 0], [0, 0]], [[0, 0], [0, 0]],
                                 [[2, 0], [0, 0]], [[1, 1, 0], [0, 0, 0]],
                                 [[0, 1, 0], [0, 0, 0]],
                                 [[2, 2, 0], [0, 0, 0]]),
        'DroppingNonPositiveLengths':
            PackSequenceTestCase([1, 0, 1], [0, 1, 3], 2, 2, 3,
                                 [[1, 0], [0, 0]], [[0, 0], [0, 0]],
                                 [[2, 0], [0, 0]], [[1, 1, 1], [0, 0, 0]],
                                 [[0, 1, 2], [0, 0, 0]],
                                 [[2, 2, 2], [0, 0, 0]]),
        'PackedBatchSize0':
            PackSequenceTestCase(
                [3, 1, 2, 0, 1, 6, 2, 3, 4, 1, 1],
                [4, 2, 1, 1, 0, 2, 6, 1, 1, 4, 3], 0, 5, 5,
                [[1, 1, 1, 2, 2], [1, 2, 2, 2, 0], [1, 1, 1, 1, 2],
                 [1, 0, 0, 0, 0]], [[0, 1, 2, 0, 1], [0, 0, 1, 2, 0],
                                    [0, 1, 2, 3, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 2, 2], [1, 7, 7, 7, 0], [8, 8, 8, 8, 9],
                 [10, 0, 0, 0, 0]], [[1, 1, 1, 1, 2], [1, 1, 2, 0, 0],
                                     [1, 2, 2, 2, 2], [1, 1, 1, 0, 0]],
                [[0, 1, 2, 3, 0], [0, 1, 0, 0, 0], [0, 0, 1, 2, 3],
                 [0, 1, 2, 0, 0]], [[0, 0, 0, 0, 2], [1, 1, 7, 0, 0],
                                    [8, 9, 9, 9, 9], [10, 10, 10, 0, 0]])
    }
    for name, test in test_cases.items():
      with self.subTest(name=name), self.session() as sess:
        r = sess.run(
            ops.pack_sequences(
                tf.constant(test.src_actual_seq_len, tf.int32),
                tf.constant(test.tgt_actual_seq_len,
                            tf.int32), test.packed_batch_size,
                test.packed_src_seq_len, test.packed_tgt_seq_len))
        self.assertEqual(6, len(r), name)
        self.assertAllEqual(r[0], test.src_segment_ids, name)
        self.assertAllEqual(r[1], test.src_segment_pos, name)
        self.assertAllEqual(r[2], test.src_indices_in_input, name)
        self.assertAllEqual(r[3], test.tgt_segment_ids, name)
        self.assertAllEqual(r[4], test.tgt_segment_pos, name)
        self.assertAllEqual(r[5], test.tgt_indices_in_input, name)

  def testPackSequencesShapeUnknown(self):
    actual_seq_len = tf.compat.v1.placeholder(tf.int32, shape=None)
    with self.session() as sess:
      output = ops.pack_sequences(actual_seq_len, actual_seq_len, 2, 5, 5)
      r = sess.run(output, feed_dict={actual_seq_len: np.array([1, 2, 1])})
    self.assertEqual(6, len(r))
    self.assertAllEqual(r[0], [[1, 2, 0, 0, 0], [1, 1, 0, 0, 0]])
    self.assertAllEqual(r[1], [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    self.assertAllEqual(r[2], [[0, 2, 0, 0, 0], [1, 1, 0, 0, 0]])
    self.assertAllEqual(r[3], r[0])
    self.assertAllEqual(r[4], r[1])
    self.assertAllEqual(r[5], r[2])

  def testPackSequencesErrors(self):
    test_cases = {
        'actual_seq_len must be the same shape':
            PackSequenceTestCase([1, 1, 1], [1, 1], 2, 2, 2),
        'actual_seq_len must be a vector':
            PackSequenceTestCase([[1], [1]], [[1], [1]], 2, 2, 2)
    }
    for name, test in test_cases.items():
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError, name):
        with self.session() as sess:
          sess.run(
              ops.pack_sequences(test.src_actual_seq_len,
                                 test.tgt_actual_seq_len,
                                 test.packed_batch_size,
                                 test.packed_src_seq_len,
                                 test.packed_tgt_seq_len))

    # seq_len must be a scalar.
    test = PackSequenceTestCase([1, 1], [1, 1], 2, [2, 2], 2)
    with self.assertRaisesRegex(TypeError, 'Expected int'):
      with self.session() as sess:
        sess.run(
            ops.pack_sequences(test.src_actual_seq_len, test.tgt_actual_seq_len,
                               test.packed_batch_size, test.packed_src_seq_len,
                               test.packed_tgt_seq_len))

  def testDroppingInputsFixedSeed(self):
    # Packing 3 rows into 2, where we need to drop one row.
    inputs = [[2, 1, 2], [1, 2, 1], 2, 2, 2]
    test_cases = [
        # (seed, test_case)
        (
            45,
            # dropping the last row
            PackSequenceTestCase(*(
                inputs +
                [[[1, 1], [1, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]],
                 [[1, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 0], [1, 1]]]))),
        (
            (1 << 37) + 1,
            # dropping the second row
            PackSequenceTestCase(*(
                inputs +
                [[[1, 1], [1, 1]], [[0, 1], [0, 1]], [[0, 0], [2, 2]],
                 [[1, 0], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [2, 0]]]))),
    ]
    for seed, test in test_cases:
      with self.session() as sess:
        r = sess.run(
            ops.pack_sequences(
                tf.constant(test.src_actual_seq_len, tf.int32),
                tf.constant(test.tgt_actual_seq_len, tf.int32),
                test.packed_batch_size,
                test.packed_src_seq_len,
                test.packed_tgt_seq_len,
                seed=seed))
        name = 'test case with seed {}'.format(seed)
        self.assertEqual(6, len(r), name)
        self.assertAllEqual(r[0], test.src_segment_ids, name)
        self.assertAllEqual(r[1], test.src_segment_pos, name)
        self.assertAllEqual(r[2], test.src_indices_in_input, name)
        self.assertAllEqual(r[3], test.tgt_segment_ids, name)
        self.assertAllEqual(r[4], test.tgt_segment_pos, name)
        self.assertAllEqual(r[5], test.tgt_indices_in_input, name)

  def testDroppingInputsNonDeterministic(self):
    # Packing 3 rows into 1, where we need to drop two rows.
    inputs = [[2, 1, 2], [1, 2, 1], 1, 2, 2]
    test_cases = [
        # keeping only the first row
        PackSequenceTestCase(*(
            inputs +
            [[[1, 1]], [[0, 1]], [[0, 0]], [[1, 0]], [[0, 0]], [[0, 0]]])),
        # keeping only the second row
        PackSequenceTestCase(*(
            inputs +
            [[[1, 0]], [[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]], [[1, 1]]])),
        # keeping only the last row
        PackSequenceTestCase(*(
            inputs +
            [[[1, 1]], [[0, 1]], [[2, 2]], [[1, 0]], [[0, 0]], [[2, 0]]])),
    ]
    counts = [0] * 3
    with self.session() as sess:
      test = test_cases[0]
      for _ in range(100):
        r = sess.run(
            ops.pack_sequences(
                tf.constant(test.src_actual_seq_len, tf.int32),
                tf.constant(test.tgt_actual_seq_len,
                            tf.int32), test.packed_batch_size,
                test.packed_src_seq_len, test.packed_tgt_seq_len))
        match_idx = FindResultFromList(r, test_cases)
        self.assertIsNotNone(match_idx, '{} is not a valid result'.format(r))
        counts[match_idx] += 1
    # We test that all possible outcomes occur sufficiently often to ensure that
    # dropping is not biased.
    # The probability of this test failing due to chance is less than 1 in a
    # million runs, as scipy.stats.binom.cdf(10, 100, 0.3333) ~= 5e-8
    for idx, count in enumerate(counts):
      self.assertGreater(
          count, 10,
          'test case {} does not occur sufficiently often: {}'.format(
              idx, counts))


class PackSingleSequenceOpTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      # Output with batch size 1 containing all 3 inputs.
      ('Ex1Sequential', [1, 2, 1], 5, True, [[0, 1, 2]]),
      # The 1 is not combined with the final 4 because of sequential order.
      ('Ex2Sequential', [5, 4, 3, 2, 1, 5, 4], 5, True, [[0], [1], [2, 3], [4],
                                                         [5], [6]]),
      # Output with batch size 1 containing all 3 inputs.
      ('Ex1', [1, 2, 1], 5, False, [[0, 1, 2]]),
      # [0] [5] [1] [6] [2] open new buckets, the rest fit into existing ones.
      ('Ex2', [5, 4, 3, 2, 1, 5, 4], 5, False, [[0], [5], [1, 4], [6], [2, 3]]),
      ('Ex3', [5, 4, 3, 2, 1, 5, 4], 10, False, [[0, 5], [1, 3, 6], [2, 4]]),
  )
  def testPackSingleSequence(self, input_lengths, max_packed_length,
                             require_sequential_order, expected_packed_idxs):
    with self.session() as sess:
      np.random.seed(12345)
      segment_ids, indices_in_input = sess.run(
          ops.pack_single_sequence(
              input_lengths=input_lengths,
              max_packed_length=max_packed_length,
              require_sequential_order=require_sequential_order))
      self.assertLen(expected_packed_idxs, segment_ids.shape[0])

      # Test the output is compatible with apply_packing.
      inputs = []
      for i, length in enumerate(input_lengths):
        inputs.append(
            np.random.randint(100000, size=[length, 2, 2], dtype=np.int32))
      outputs = sess.run(
          ops.apply_packing(
              input=tf.stack([
                  tf.pad(x,
                         [[0, max_packed_length - x.shape[0]], [0, 0], [0, 0]])
                  for x in inputs
              ]),
              padding=0,
              segment_ids=segment_ids,
              indices_in_input=indices_in_input))

      for segment_id, idxs, output, expected_idxs in zip(
          segment_ids, indices_in_input, outputs, expected_packed_idxs):
        # Build the expected results from the provided expected_packed_idxs.
        expected_segment_ids = []
        expected_idxs_vec = []
        expected_outputs = []
        for i, idx in enumerate(expected_idxs):
          expected_segment_ids += [i + 1] * input_lengths[idx]
          expected_idxs_vec += [idx] * input_lengths[idx]
          expected_outputs.append(inputs[idx])
        expected_outputs = np.concatenate(expected_outputs)
        expected_packed_length = len(expected_outputs)
        self.assertLessEqual(expected_packed_length, max_packed_length)
        self.assertLen(expected_segment_ids, expected_packed_length)
        self.assertLen(expected_idxs_vec, expected_packed_length)

        # Check indices_in_input is non-decreasing.
        if expected_packed_length > 1:
          self.assertAllGreaterEqual(
              idxs[1:expected_packed_length] -
              idxs[:expected_packed_length - 1], 0)

        # Pad to max_packed_length.
        pad_len = max_packed_length - expected_packed_length
        expected_segment_ids += [0] * pad_len
        expected_idxs_vec += [-1] * pad_len
        expected_outputs = np.pad(
            expected_outputs, [(0, pad_len), (0, 0), (0, 0)], mode='constant')

        self.assertAllEqual(expected_idxs_vec, idxs)
        self.assertAllEqual(expected_segment_ids, segment_id)
        self.assertAllEqual(expected_outputs, output)

  def testInputLengthTooLong(self):
    with self.session() as sess:
      sess.run(
          ops.pack_single_sequence(input_lengths=[10], max_packed_length=10))
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'too long'):
        sess.run(
            ops.pack_single_sequence(input_lengths=[10], max_packed_length=5))


class ApplyPackingOpTest(test_utils.TestCase, parameterized.TestCase):

  def testApplyPacking(self):
    test_cases = {
        'Basic':
            ApplyPackingTestCase([[0, 1], [2, 3]], 0, [[1, 1], [1, 1]],
                                 [[1, 1], [0, 0]], [[2, 3], [0, 1]]),
        'Padding':
            ApplyPackingTestCase([[0, 1], [2, 3]], -1, [[1, 1, 0], [1, 1, 0]],
                                 [[1, 1, 0], [0, 0, 0]],
                                 [[2, 3, -1], [0, 1, -1]]),
        'Tiny':
            ApplyPackingTestCase([[0, 1], [2, 3]], 0, [[1], [1]], [[0], [1]],
                                 [[0], [2]]),
        '5x2 input to 2x4':
            ApplyPackingTestCase(
                np.reshape(np.arange(10),
                           [5, 2]), -1, [[1, 1, 2, 2], [0, 1, 0, 2]],
                [[2, 2, 3, 3], [0, 4, 0, 0]], [[4, 5, 6, 7], [-1, 8, -1, 0]]),
        '6x4 input to 3x4':
            ApplyPackingTestCase(
                np.reshape(np.arange(24), [6, 4]), -1,
                [[0, 0, 0, 1], [1, 1, 1, 1], [1, 2, 3, 4]],
                [[0, 0, 0, 0], [1, 1, 1, 1], [2, 3, 4, 5]],
                [[-1, -1, -1, 0], [4, 5, 6, 7], [8, 12, 16, 20]]),
        '6x4 input to 3x5':
            ApplyPackingTestCase(
                np.reshape(np.arange(24), [6, 4]), -1,
                [[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 2]],
                [[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 3]],
                [[-1, -1, -1, -1, -1], [-1, 4, 5, -1, -1], [-1, 0, 1, -1, 12]]),
        '100x4 input to 3x5':
            ApplyPackingTestCase(
                np.reshape(np.arange(400), [100, 4]), -1,
                [[1, 1, 1, 2, 2], [0, 1, 1, 1, 1], [2, 2, 0, 0, 3]],
                [[99, 99, 99, 1, 1], [0, 50, 50, 50, 50], [90, 90, 0, 0, 3]],
                [[396, 397, 398, 4, 5], [-1, 200, 201, 202, 203],
                 [360, 361, -1, -1, 12]]),
    }
    for name, test in test_cases.items():
      with self.session():
        output = ops.apply_packing(
            tf.constant(test.input, tf.int32),
            tf.constant(test.padding, tf.int32),
            tf.constant(test.segment_ids, tf.int32),
            tf.constant(test.indices_in_input, tf.int32)).eval()
        self.assertAllEqual(output, test.output, name)

  def testApplyPackingUnknownShape(self):
    x = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
    self.assertAllEqual(x.shape.as_list(), [None, None])
    with self.session():
      x_val = np.array([[0, 1], [2, 3]])
      output = ops.apply_packing(
          x, 0, tf.constant([[1, 1], [1, 1]], tf.int32),
          tf.constant([[1, 1], [0, 0]], tf.int32)).eval(feed_dict={x: x_val})
    self.assertAllEqual(output, [[2, 3], [0, 1]])

  def testApplyPackingTypes(self):
    test = ApplyPackingTestCase([[0, 1], [2, 3]], 99, [[1, 1, 0], [1, 1, 0]],
                                [[1, 1, 0], [0, 0, 0]],
                                [[2, 3, 99], [0, 1, 99]])
    for dtype in [
        tf.int32, tf.int64, tf.float32, tf.float64, tf.uint32, tf.uint64,
        tf.bfloat16
    ]:
      with self.session():
        output = ops.apply_packing(
            tf.constant(test.input, dtype), tf.constant(test.padding, dtype),
            tf.constant(test.segment_ids, tf.int32),
            tf.constant(test.indices_in_input, tf.int32)).eval()
        expected = tf.constant(test.output, dtype).eval()
        self.assertAllEqual(output, expected, dtype)

  def testApplyPackingStrings(self):
    test_cases = {
        'Basic':
            ApplyPackingTestCase(['a', 'b'], ',', [[1, 1]], [[1, 0]], [b'b,a']),
        'Repeated':
            ApplyPackingTestCase(['a', 'b'], ',', [[1, 1, 1]], [[1, 1, 1]],
                                 [b'b']),
        'Separator':
            ApplyPackingTestCase(['a', 'b', 'c', 'd'], '=', [[1, 1, 1, 0]],
                                 [[1, 0, 3, 2]], [b'b=a=d']),
        'MultiRows':
            ApplyPackingTestCase(['a', 'b', 'c', 'd'], ';',
                                 [[1, 1, 1, 0], [0, 1, 1, 1]],
                                 [[2, 2, 1, 0], [2, 0, 1, 1]],
                                 [b'c;b', b'a;b']),
        'SingleString':
            ApplyPackingTestCase(['a', 'b', 'c', 'd'], ',',
                                 [[0, 0, 1], [0, 1, 0]], [[0, 1, 2], [0, 1, 2]],
                                 [b'c', b'b']),
        'EmptyRow':
            ApplyPackingTestCase(['a', 'b', 'c', 'd'], ',',
                                 [[0, 0, 0], [1, 1, 1]], [[0, 1, 2], [0, 0, 2]],
                                 [b'', b'a,c']),
    }
    for name, test in test_cases.items():
      with self.session():
        output = ops.apply_packing(
            tf.constant(test.input, tf.string),
            tf.constant(test.padding, tf.string),
            tf.constant(test.segment_ids, tf.int32),
            tf.constant(test.indices_in_input, tf.int32)).eval()
        self.assertAllEqual(output, test.output, name)

  def testApplyPackingSum(self):
    test_cases = {
        'Basic':
            ApplyPackingTestCase(
                np.arange(10), 0, [[1, 1], [1, 1]], [[1, 1], [5, 5]], [1, 5]),
        'Padding':
            ApplyPackingTestCase(
                np.arange(10), 0, [[1, 1, 0], [1, 1, 0]],
                [[1, 1, 0], [3, 3, 0]], [1, 3]),
        'Tiny':
            ApplyPackingTestCase(
                np.arange(10), 0, [[1], [1]], [[3], [1]], [3, 1]),
        'Larger':
            ApplyPackingTestCase(
                np.arange(10), 0, [[1, 1, 2, 2], [0, 1, 2, 3], [0, 1, 1, 1]],
                [[2, 2, 3, 3], [9, 4, 5, 6], [9, 8, 8, 8]], [5, 15, 8]),
    }
    for name, test in test_cases.items():
      for dtype in [
          tf.int32, tf.int64, tf.float32, tf.float64, tf.uint32, tf.uint64
      ]:
        with self.session():
          output = ops.apply_packing(
              tf.constant(test.input, dtype), tf.constant(test.padding, dtype),
              tf.constant(test.segment_ids, tf.int32),
              tf.constant(test.indices_in_input, tf.int32)).eval()
          expected = tf.constant(test.output, dtype).eval()
          self.assertAllEqual(output, expected, f'{name} {dtype}')

  @parameterized.named_parameters(
      ('Test1', tf.errors.InvalidArgumentError, 'out of bound',
       ApplyPackingTestCase([[0, 1], [2, 3]], 0, [[1, 1, 1], [1, 0, 0]],
                            [[1, 1, 1], [0, 0, 0]])),
      ('Test2', tf.errors.InvalidArgumentError, 'out of bound ',
       ApplyPackingTestCase([[0, 1], [2, 3]], 0, [[1, 1], [1, 0]],
                            [[0, 0], [2, 0]])),
      ('Test3', tf.errors.InvalidArgumentError, 'out of bound.',
       ApplyPackingTestCase(['a', 'b'], ',', [[1, 2]], [[1, 2]])),
      ('Test4', ValueError, 'segment_ids and indices_in_input must be matrices',
       ApplyPackingTestCase([[0, 1], [2, 3]], 0, [1, 1], [0, 0])),
      ('Test5', ValueError,
       'segment_ids and indices_in_input must be matrices of the same shape',
       ApplyPackingTestCase([[0, 1], [2, 3]], 0, [[1, 1], [1, 0]],
                            [[0, 0], [0, 0], [0, 0]])),
      ('Test6', ValueError, 'padding must be a scalar',
       ApplyPackingTestCase([[0, 1], [2, 3]], [-1], [[1]], [[0]])),
  )
  def testApplyPackingErrors(self, expected_error_type, expected_error, test):
    with self.assertRaisesRegex(expected_error_type, expected_error):
      with self.session():
        ops.apply_packing(test.input, test.padding, test.segment_ids,
                          test.indices_in_input).eval()


if __name__ == '__main__':
  tf.test.main()
