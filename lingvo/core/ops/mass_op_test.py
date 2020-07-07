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
"""Tests for mass_op."""

import collections
from lingvo import compat as tf
from lingvo.core import ops
from lingvo.core import test_utils
import numpy as np

FLAGS = tf.flags.FLAGS

BOS = 1
EOS = 2

MassOutput = collections.namedtuple(
    'MassOutput', ['src_ids', 'tgt_ids', 'tgt_labels', 'tgt_weights'])

# Set default to empty.
MassOutput.__new__.__defaults__ = (None,) * len(MassOutput._fields)


def FindResultFromList(result, expected_results):
  """Find the given result from a list of expected results.

  Args:
    result: A MassOutput tuple, from running ops.mass().
    expected_results: A list of MassOutput.  The test asserts `result` is equal
      to at least one result from `expected_results`.

  Returns:
    The index of first match found, or None for not found.

  We use this when the specific output from ops.mass() is not stable across
  different platforms. Specifically, the implementation currently uses
  std::shuffle(), which have different implementations between libc++ and
  stdlibc++.
  """
  for idx, expected in enumerate(expected_results):
    match = True
    for attr in MassOutput._fields:
      if not np.array_equal(getattr(result, attr), getattr(expected, attr)):
        match = False
        break
    if match:
      return idx

  tf.logging.error('Found unexpected output from op.mass that fails to match'
                   ' any expected result.')
  for attr in MassOutput._fields:
    tf.logging.info('%s = %s', attr, np.array_repr(getattr(result, attr)))
  return None


class MassOpTest(test_utils.TestCase):

  def testFixedStart(self):
    ids = np.array(
        [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int32)
    weights = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32)
    actual_seq_len = np.array([14, 10, 14, 10], dtype=np.int32)

    g = tf.Graph()
    with g.as_default():
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
          ids,
          weights,
          actual_seq_len,
          mask_id=3,
          mask_ratio=0.5,
          mask_minlen=1,
          span_len=8,
          random_start_prob=0,
          keep_prob=0,
          rand_prob=0,
          mask_prob=1,
          mask_target=True,
          vocab_size=9)

      with self.session(graph=g) as sess:
        (src_ids, tgt_ids, tgt_labels,
         tgt_weights) = sess.run([src_ids, tgt_ids, tgt_labels, tgt_weights])
        self.assertAllEqual(
            src_ids,
            np.array([[
                3, 3, 3, 3, 3, 3, 3, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0,
                0
            ], [
                3, 3, 3, 3, 3, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [4, 5, 6, 7, 8, 9, 10, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0
               ], [4, 5, 6, 7, 8, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_ids,
            np.array([
                [BOS, 4, 5, 6, 7, 8, 9, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                [BOS, 4, 5, 6, 7, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    3, 3, 3, 3, 3, 3, 3, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0,
                    0, 0
                ],
                [3, 3, 3, 3, 3, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_labels,
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0,
                          0, 0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_weights,
            np.array(
                [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

  def testRandomStart(self):
    ids = np.array(
        [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int32)
    weights = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32)
    actual_seq_len = np.array([14, 10, 14, 10], dtype=np.int32)

    g = tf.Graph()
    with g.as_default():
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
          ids,
          weights,
          actual_seq_len,
          mask_id=3,
          mask_ratio=0.5,
          mask_minlen=1,
          span_len=100000,
          random_start_prob=1,
          keep_prob=0,
          rand_prob=0,
          mask_prob=1,
          mask_target=True,
          vocab_size=9)

      with self.session(graph=g) as sess:
        (src_ids, tgt_ids, tgt_labels, tgt_weights) = sess.run([
            src_ids,
            tgt_ids,
            tgt_labels,
            tgt_weights,
        ])
        result = MassOutput(src_ids, tgt_ids, tgt_labels, tgt_weights)
        expected_output1 = MassOutput(
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0
            ], [
                4, 3, 3, 3, 3, 3, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                4, 5, 6, 7, 3, 3, 3, 3, 3, 3, 3, 15, 16, EOS, 0, 0, 0, 0, 0, 0
            ], [4, 5, 3, 3, 3, 3, 3, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                     ],
                     dtype=np.int32),
            np.array([[
                3, 3, 3, 3, 3, 3, 3, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0,
                0
            ], [
                3, 4, 5, 6, 7, 8, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [3, 3, 3, 3, 7, 8, 9, 10, 11, 12, 13, 3, 3, 3, 0, 0, 0, 0, 0, 0
               ], [3, 3, 5, 6, 7, 8, 9, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0,
                          0, 0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        expected_output2 = MassOutput(
            np.array([[
                3, 3, 3, 3, 3, 3, 3, 11, 12, 13, 14, 15, 16, 2, 0, 0, 0, 0, 0, 0
            ], [
                3, 3, 3, 3, 3, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                4, 5, 6, 7, 3, 3, 3, 3, 3, 3, 3, 15, 16, 2, 0, 0, 0, 0, 0, 0
            ], [4, 5, 3, 3, 3, 3, 3, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                1, 4, 5, 6, 7, 8, 9, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0
            ], [
                1, 4, 5, 6, 7, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [3, 3, 3, 3, 7, 8, 9, 10, 11, 12, 13, 3, 3, 3, 0, 0, 0, 0, 0, 0
               ], [3, 3, 5, 6, 7, 8, 9, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0, 0, 0, 0,
                0
            ], [4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0,
                          0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        match_idx = FindResultFromList(result,
                                       [expected_output1, expected_output2])
        self.assertIsNotNone(match_idx,
                             '{} is not a valid result'.format(result))

  def testSegmented(self):
    ids = np.array(
        [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int32)
    weights = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32)
    actual_seq_len = np.array([14, 10, 14, 10], dtype=np.int32)

    g = tf.Graph()
    with g.as_default():
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
          ids,
          weights,
          actual_seq_len,
          mask_id=3,
          mask_ratio=0.5,
          mask_minlen=1,
          span_len=3,
          keep_prob=0,
          rand_prob=0,
          mask_prob=1,
          mask_target=True,
          vocab_size=9)

      with self.session(graph=g) as sess:
        (src_ids, tgt_ids, tgt_labels, tgt_weights) = sess.run([
            src_ids,
            tgt_ids,
            tgt_labels,
            tgt_weights,
        ])

        result = MassOutput(src_ids, tgt_ids, tgt_labels, tgt_weights)
        expected_output1 = MassOutput(
            np.array([[
                4, 3, 3, 3, 3, 3, 3, 11, 12, 13, 14, 15, 16, 3, 0, 0, 0, 0, 0, 0
            ], [
                4, 3, 3, 7, 8, 9, 10, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [4, 5, 6, 7, 8, 3, 3, 3, 12, 3, 14, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                      [
                          3, 3, 6, 3, 3, 3, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array([[
                3, 4, 5, 6, 7, 8, 9, 3, 3, 3, 3, 3, 3, 16, 0, 0, 0, 0, 0, 0
            ], [
                3, 4, 5, 3, 3, 3, 3, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                3, 3, 3, 3, 3, 8, 9, 10, 3, 12, 3, 14, 15, 16, 0, 0, 0, 0, 0, 0
            ], [BOS, 4, 3, 6, 7, 8, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0,
                          0, 0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        expected_output2 = MassOutput(
            np.array([[
                4, 5, 6, 3, 3, 3, 10, 3, 3, 3, 14, 15, 16, 3, 0, 0, 0, 0, 0, 0
            ], [
                3, 3, 6, 7, 8, 9, 10, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [4, 3, 3, 3, 8, 9, 10, 11, 12, 3, 3, 3, 16, 3, 0, 0, 0, 0, 0, 0
               ], [4, 5, 3, 3, 8, 9, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                3, 3, 3, 6, 7, 8, 3, 10, 11, 12, 3, 3, 3, 16, 0, 0, 0, 0, 0, 0
            ], [
                1, 4, 3, 3, 3, 3, 3, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                3, 4, 5, 6, 3, 3, 3, 3, 3, 12, 13, 14, 3, 16, 0, 0, 0, 0, 0, 0
            ], [3, 3, 5, 6, 3, 3, 9, 10, 11, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0, 0, 0, 0,
                0
            ], [4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0,
                          0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array([[
                0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0.
            ],
                      [
                          1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0.
                      ],
                      [
                          0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
                          1., 0., 0., 0., 0., 0., 0.
                      ],
                      [
                          0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0.
                      ]],
                     dtype=np.float32))
        match_idx = FindResultFromList(result,
                                       [expected_output1, expected_output2])
        self.assertIsNotNone(match_idx,
                             '{} is not a valid result'.format(result))

  def testNoMaskTarget(self):
    ids = np.array(
        [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int32)
    weights = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32)
    actual_seq_len = np.array([14, 10, 14, 10], dtype=np.int32)

    g = tf.Graph()
    with g.as_default():
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
          ids,
          weights,
          actual_seq_len,
          mask_id=3,
          mask_ratio=0.5,
          mask_minlen=1,
          span_len=100000,
          random_start_prob=0,
          keep_prob=0,
          rand_prob=0,
          mask_prob=1,
          mask_target=False,
          vocab_size=9)

      with self.session(graph=g) as sess:
        (src_ids, tgt_ids, tgt_labels, tgt_weights) = sess.run([
            src_ids,
            tgt_ids,
            tgt_labels,
            tgt_weights,
        ])

        self.assertAllEqual(
            src_ids,
            np.array([[
                3, 3, 3, 3, 3, 3, 3, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0,
                0
            ], [
                3, 3, 3, 3, 3, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [4, 5, 6, 7, 8, 9, 10, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0
               ], [4, 5, 6, 7, 8, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_ids,
            np.array([[
                1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0,
                0
            ], [
                BOS, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0,
                          0, 0, 0, 0
                      ],
                      [
                          1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0
                      ]],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_labels,
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0,
                          0, 0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_weights,
            np.array(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

  def testKeepOrRandMaskedTokens(self):
    ids = np.array(
        [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int32)
    weights = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32)
    actual_seq_len = np.array([14, 10, 14, 10], dtype=np.int32)

    g = tf.Graph()
    with g.as_default():
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
          ids,
          weights,
          actual_seq_len,
          mask_id=3,
          mask_ratio=0.5,
          mask_minlen=1,
          span_len=100000,
          random_start_prob=0,
          keep_prob=0.5,
          rand_prob=0.5,
          mask_prob=0,
          mask_target=True,
          vocab_size=9)

      with self.session(graph=g) as sess:
        (src_ids, tgt_ids, tgt_labels, tgt_weights) = sess.run([
            src_ids,
            tgt_ids,
            tgt_labels,
            tgt_weights,
        ])

        result = MassOutput(src_ids, tgt_ids, tgt_labels, tgt_weights)
        expected_output1 = MassOutput(
            np.array([[
                4, 7, 6, 9, 5, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 8, 9, 8, 12, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 7, 12, 5, 7, 15, 16, EOS, 0, 0, 0, 0, 0, 0
            ], [4, 5, 6, 7, 8, 9, 6, 11, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([
                [BOS, 4, 5, 6, 7, 8, 9, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                [3, 3, 3, 3, 3, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    3, 3, 3, 3, 3, 3, 3, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0,
                    0, 0
                ],
                [3, 3, 3, 3, 3, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0,
                          0, 0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        expected_output2 = MassOutput(
            np.array([[
                4, 8, 6, 7, 6, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0, 0, 0, 0,
                0
            ], [4, 5, 6, 7, 8, 4, 9, 4, 12, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
               ], [
                   4, 5, 6, 7, 8, 9, 10, 5, 7, 6, 4, 15, 16, 2, 0, 0, 0, 0, 0, 0
               ], [
                   4, 5, 6, 7, 8, 9, 6, 11, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
               ]],
                     dtype=np.int32),
            np.array([
                [1, 4, 5, 6, 7, 8, 9, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
                [3, 3, 3, 3, 3, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    3, 3, 3, 3, 3, 3, 3, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0,
                    0, 0
                ],
                [3, 3, 3, 3, 3, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0, 0, 0, 0,
                0
            ], [4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0,
                          0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        match_idx = FindResultFromList(result,
                                       [expected_output1, expected_output2])
        self.assertIsNotNone(match_idx,
                             '{} is not a valid result'.format(result))

  def testKeepMaskedTokens(self):
    ids = np.array(
        [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int32)
    weights = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32)
    actual_seq_len = np.array([14, 10, 14, 10], dtype=np.int32)

    g = tf.Graph()
    with g.as_default():
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
          ids,
          weights,
          actual_seq_len,
          mask_id=3,
          mask_ratio=0.5,
          mask_minlen=1,
          span_len=3,
          keep_prob=1,
          rand_prob=0,
          mask_prob=0,
          mask_target=True,
          vocab_size=9)

      with self.session(graph=g) as sess:
        (src_ids, tgt_ids, tgt_labels, tgt_weights) = sess.run([
            src_ids,
            tgt_ids,
            tgt_labels,
            tgt_weights,
        ])

        result = MassOutput(src_ids, tgt_ids, tgt_labels, tgt_weights)
        expected_output1 = MassOutput(
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0,
                          0, 0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array([[
                3, 4, 5, 6, 7, 8, 9, 3, 3, 3, 3, 3, 3, 16, 0, 0, 0, 0, 0, 0
            ], [
                3, 4, 5, 3, 3, 3, 3, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                3, 3, 3, 3, 3, 8, 9, 10, 3, 12, 3, 14, 15, 16, 0, 0, 0, 0, 0, 0
            ], [BOS, 4, 3, 6, 7, 8, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0,
                          0, 0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        expected_output2 = MassOutput(
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0, 0, 0, 0,
                0
            ], [4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0,
                          0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array([[
                3, 3, 3, 6, 7, 8, 3, 10, 11, 12, 3, 3, 3, 16, 0, 0, 0, 0, 0, 0
            ], [
                1, 4, 3, 3, 3, 3, 3, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                3, 4, 5, 6, 3, 3, 3, 3, 3, 12, 13, 14, 3, 16, 0, 0, 0, 0, 0, 0
            ], [3, 3, 5, 6, 3, 3, 9, 10, 11, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0, 0, 0, 0,
                0
            ], [4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0,
                          0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        match_idx = FindResultFromList(result,
                                       [expected_output1, expected_output2])
        self.assertIsNotNone(match_idx,
                             '{} is not a valid result'.format(result))

  def testSpanLen1(self):
    ids = np.array(
        [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0, 0, 0],
         [4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int32)
    weights = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32)
    actual_seq_len = np.array([14, 10, 14, 10], dtype=np.int32)

    g = tf.Graph()
    with g.as_default():
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
          ids,
          weights,
          actual_seq_len,
          mask_id=3,
          mask_ratio=0.5,
          mask_minlen=1,
          span_len=1,
          keep_prob=0,
          rand_prob=0,
          mask_prob=1,
          mask_target=True,
          vocab_size=9)

      with self.session(graph=g) as sess:
        (src_ids, tgt_ids, tgt_labels, tgt_weights) = sess.run([
            src_ids,
            tgt_ids,
            tgt_labels,
            tgt_weights,
        ])

        result = MassOutput(src_ids, tgt_ids, tgt_labels, tgt_weights)
        expected_output1 = MassOutput(
            np.array([
                [4, 3, 3, 7, 8, 9, 3, 3, 3, 13, 14, 3, 16, 3, 0, 0, 0, 0, 0, 0],
                [3, 3, 6, 3, 3, 9, 10, 11, 12, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, 5, 3, 3, 8, 9, 10, 3, 3, 13, 3, 15, 3, 3, 0, 0, 0, 0, 0, 0],
                [3, 5, 6, 3, 8, 3, 10, 3, 12, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
                     dtype=np.int32),
            np.array([[
                3, 4, 5, 3, 3, 3, 9, 10, 11, 3, 3, 14, 3, 16, 0, 0, 0, 0, 0, 0
            ], [
                1, 4, 3, 6, 7, 3, 3, 3, 3, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                3, 3, 5, 6, 3, 3, 3, 10, 11, 3, 13, 3, 15, 16, 0, 0, 0, 0, 0, 0
            ], [BOS, 3, 3, 6, 3, 8, 3, 10, 3, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                     ],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0, 0, 0, 0,
                0, 0
            ], [
                4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, EOS, 0,
                          0, 0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, EOS, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        expected_output2 = MassOutput(
            np.array([[
                4, 5, 6, 3, 3, 3, 10, 3, 12, 13, 3, 3, 16, 3, 0, 0, 0, 0, 0, 0
            ], [
                3, 5, 3, 7, 3, 9, 3, 3, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ], [
                3, 5, 3, 3, 8, 9, 3, 3, 12, 3, 3, 15, 16, 2, 0, 0, 0, 0, 0, 0
            ], [3, 3, 3, 7, 8, 9, 3, 11, 12, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.int32),
            np.array([[
                3, 3, 3, 6, 7, 8, 3, 10, 3, 3, 13, 14, 3, 16, 0, 0, 0, 0, 0, 0
            ], [1, 3, 5, 3, 7, 3, 9, 10, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
               ], [
                   1, 3, 5, 6, 3, 3, 9, 10, 3, 12, 13, 3, 3, 3, 0, 0, 0, 0, 0, 0
               ], [1, 4, 5, 3, 3, 3, 9, 3, 3, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                     ],
                     dtype=np.int32),
            np.array([[
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0, 0, 0, 0,
                0
            ], [4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 2, 0, 0,
                          0, 0, 0, 0
                      ],
                      [
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0
                      ]],
                     dtype=np.int32),
            np.array(
                [[0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.float32))

        match_idx = FindResultFromList(result,
                                       [expected_output1, expected_output2])
        self.assertIsNotNone(match_idx,
                             '{} is not a valid result'.format(result))

  def testZeroLengthSeq(self):
    ids = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                   dtype=np.int32)
    weights = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                       dtype=np.float32)
    actual_seq_len = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    g = tf.Graph()
    with g.as_default():
      (src_ids, tgt_ids, tgt_labels, tgt_weights) = ops.mass(
          ids,
          weights,
          actual_seq_len,
          mask_id=3,
          mask_ratio=0.5,
          mask_minlen=0,
          span_len=8,
          random_start_prob=0,
          keep_prob=0,
          rand_prob=0,
          mask_prob=1,
          mask_target=True,
          vocab_size=9)

      with self.session(graph=g) as sess:
        (src_ids, tgt_ids, tgt_labels,
         tgt_weights) = sess.run([src_ids, tgt_ids, tgt_labels, tgt_weights])

        self.assertAllEqual(
            src_ids,
            np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_ids,
            np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_labels,
            np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                     dtype=np.int32))
        self.assertAllEqual(
            tgt_weights,
            np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                     dtype=np.float32))


if __name__ == '__main__':
  tf.test.main()
