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
    }
    for name, test in test_cases.items():
      with self.session() as sess:
        r = sess.run(
            ops.pack_sequences(
                tf.constant(test.src_actual_seq_len, tf.int32),
                tf.constant(test.tgt_actual_seq_len, tf.int32),
                tf.constant(test.packed_batch_size, tf.int32),
                tf.constant(test.packed_src_seq_len, tf.int32),
                tf.constant(test.packed_tgt_seq_len, tf.int32)))
        self.assertEqual(6, len(r), name)
        self.assertAllEqual(r[0], test.src_segment_ids, name)
        self.assertAllEqual(r[1], test.src_segment_pos, name)
        self.assertAllEqual(r[2], test.src_indices_in_input, name)
        self.assertAllEqual(r[3], test.tgt_segment_ids, name)
        self.assertAllEqual(r[4], test.tgt_segment_pos, name)
        self.assertAllEqual(r[5], test.tgt_indices_in_input, name)

  def testPackSequencesErrors(self):
    test_cases = {
        'actual_seq_len must be the same shape':
            PackSequenceTestCase([1, 1, 1], [1, 1], 2, 2, 2),
        'actual_seq_len must be a vector':
            PackSequenceTestCase([[1], [1]], [[1], [1]], 2, 2, 2),
        'seq_len must be a scalar':
            PackSequenceTestCase([1, 1], [1, 1], 2, [2, 2], 2),
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
                tf.constant(test.packed_batch_size, tf.int32),
                tf.constant(test.packed_src_seq_len, tf.int32),
                tf.constant(test.packed_tgt_seq_len, tf.int32),
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
                tf.constant(test.tgt_actual_seq_len, tf.int32),
                tf.constant(test.packed_batch_size, tf.int32),
                tf.constant(test.packed_src_seq_len, tf.int32),
                tf.constant(test.packed_tgt_seq_len, tf.int32)))
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


class ApplyPackingOpTest(test_utils.TestCase):

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

  def testApplyPackingTypes(self):
    test = ApplyPackingTestCase([[0, 1], [2, 3]], 99, [[1, 1, 0], [1, 1, 0]],
                                [[1, 1, 0], [0, 0, 0]],
                                [[2, 3, 99], [0, 1, 99]])
    for dtype in [
        tf.int32, tf.int64, tf.float32, tf.float64, tf.uint32, tf.uint64
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

  def testApplyPackingErrors(self):
    test_cases = {
        'out of bound':
            ApplyPackingTestCase([[0, 1], [2, 3]], 0, [[1, 1, 1], [1, 0, 0]],
                                 [[1, 1, 1], [0, 0, 0]]),
        'out of bound ':
            ApplyPackingTestCase([[0, 1], [2, 3]], 0, [[1, 1], [1, 0]],
                                 [[0, 0], [2, 0]]),
        'out of bound.':
            ApplyPackingTestCase(['a', 'b'], ',', [[1, 2]], [[1, 2]]),
        'segment_ids and indices_in_input must be matrices':
            ApplyPackingTestCase([[0, 1], [2, 3]], 0, [1, 1], [0, 0]),
        'segment_ids and indices_in_input must be matrices of the same shape':
            ApplyPackingTestCase([[0, 1], [2, 3]], 0, [[1, 1], [1, 0]],
                                 [[0, 0], [0, 0], [0, 0]]),
        'input must be a matrix':
            ApplyPackingTestCase([0, 1], 0, [[1]], [[0]]),
        'padding must be a scalar':
            ApplyPackingTestCase([[0, 1], [2, 3]], [-1], [[1]], [[0]]),
    }
    for name, test in test_cases.items():
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError, name):
        with self.session():
          ops.apply_packing(test.input, test.padding, test.segment_ids,
                            test.indices_in_input).eval()


if __name__ == '__main__':
  tf.test.main()
