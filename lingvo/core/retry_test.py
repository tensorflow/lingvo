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
"""Tests for retry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import retry
from lingvo.core import test_utils


class RetryTest(test_utils.TestCase):

  def testRetry(self):

    @retry.Retry(max_retries=5)
    def _TestFunc():
      _TestFunc.calls += 1
      raise ValueError()

    _TestFunc.calls = 0

    with self.assertRaises(ValueError):
      _TestFunc()
    self.assertEqual(_TestFunc.calls, 6)

  def testZeroMaxRetries(self):

    @retry.Retry(max_retries=0)
    def _TestFunc():
      _TestFunc.calls += 1
      raise ValueError()

    _TestFunc.calls = 0

    with self.assertRaises(ValueError):
      _TestFunc()
    self.assertEqual(_TestFunc.calls, 1)

  def testArgsAndReturnValue(self):

    @retry.Retry()
    def _TestFunc(val):
      if _TestFunc.calls > 2:
        return val
      else:
        _TestFunc.calls += 1
        raise ValueError()

    _TestFunc.calls = 0

    self.assertEqual(_TestFunc(2), 2)
    self.assertEqual(_TestFunc.calls, 3)

  def testSpecificExceptions(self):

    @retry.Retry(max_retries=1, retry_value=(ValueError, KeyError))
    def _TestFunc(e):
      _TestFunc.calls += 1
      raise e()

    _TestFunc.calls = 0
    with self.assertRaises(ValueError):
      _TestFunc(ValueError)
    self.assertEqual(_TestFunc.calls, 2)

    _TestFunc.calls = 0
    with self.assertRaises(KeyError):
      _TestFunc(KeyError)
    self.assertEqual(_TestFunc.calls, 2)

    _TestFunc.calls = 0
    with self.assertRaises(AttributeError):
      _TestFunc(AttributeError)
    self.assertEqual(_TestFunc.calls, 1)


if __name__ == '__main__':
  tf.test.main()
