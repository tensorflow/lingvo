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
"""Tests for random_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf
from lingvo.core import test_utils
from lingvo.core.ops import py_x_ops

FLAGS = tf.flags.FLAGS


class RandomOpsTest(test_utils.TestCase):

  def testRandomPermutationSequenceRepeat(self):
    with self.session() as sess:
      out = py_x_ops.random_permutation_sequence(num=20, batch=7, repeat=True)

      remaining = list(range(20))
      for _ in range(10):
        # Each epoch takes exactly 3 steps.
        vals = sess.run(out).tolist() + sess.run(out).tolist() + sess.run(
            out).tolist()
        self.assertEqual(len(vals), 21)

        # Contains all the remaining values from previous epoch.
        for x in remaining:
          vals.remove(x)  # Raises exception if x is not in vals.

        # Remaining items have no duplicates.
        self.assertEqual(len(vals), len(set(vals)))

        remaining = list(set(range(20)) - set(vals))

  def testRandomPermutationSequenceNoRepeat(self):
    with self.session() as sess:
      out = py_x_ops.random_permutation_sequence(num=20, batch=7, repeat=False)

      # Each epoch takes exactly 3 steps.
      vals = sess.run(out).tolist() + sess.run(out).tolist() + sess.run(
          out).tolist()
      self.assertEqual(list(range(20)), sorted(vals))

      # repeat=False. We should see OutOfRange error.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(out)


if __name__ == '__main__':
  tf.test.main()
