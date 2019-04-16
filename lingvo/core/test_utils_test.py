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
"""Tests for test_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import test_utils


class TestUtilsTest(test_utils.TestCase):

  def testReplaceGoldenSingleFloat(self):
    old_line = '      CompareToGoldenSingleFloat(self, 1.489712, vs[0])\n'
    expected = '      CompareToGoldenSingleFloat(self, 1.000000, vs[0])\n'
    actual = test_utils.ReplaceGoldenSingleFloat(old_line, 1.0)
    self.assertEqual(expected, actual)

    old_line = ('test_utils.CompareToGoldenSingleFloat(self, -2.e-3, vs[0])'
                '  # pylint: disable=line-too-long\n')
    expected = ('test_utils.CompareToGoldenSingleFloat(self, 1.000000, vs[0])'
                '  # pylint: disable=line-too-long\n')
    actual = test_utils.ReplaceGoldenSingleFloat(old_line, 1.0)
    self.assertEqual(expected, actual)

  def CompareToGoldenSingleFloat(self, unused_v1, v2):
    return test_utils.ReplaceGoldenStackAnalysis(v2)

  def testReplaceGoldenStackAnalysis(self):
    v2 = 2.0
    result = TestUtilsTest.CompareToGoldenSingleFloat(self, 1.0, v2)
    self.assertTrue(result[0].endswith('test_utils_test.py'))
    old_line = ('    result = TestUtilsTest.CompareToGoldenSingleFloat('
                'self, 1.0, v2)\n')
    new_line = ('    result = TestUtilsTest.CompareToGoldenSingleFloat('
                'self, 2.000000, v2)\n')
    self.assertEqual(old_line, result[2])
    self.assertEqual(new_line, result[3])


if __name__ == '__main__':
  tf.test.main()
