# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lingvo.core.tshape."""

import lingvo.compat as tf
from lingvo.core import test_utils
from lingvo.core.tshape import Shape


class TshapeTest(test_utils.TestCase):

  def testShape(self):
    w1 = Shape([3, 3, 'd0', 'd1'])
    w2 = Shape([5, 5, w1[2], 'd2'])
    self.assertEqual(w2.ToTensorShape().as_list(), [5, 5, None, None])
    s2 = w2.Subs({w1[2]: 8, w1[3]: 16, w2[3]: 32})
    self.assertEqual(s2.ToTensorShape().as_list(), [5, 5, 8, 32])

    # __getitem__
    inner = w1[-2:]
    self.assertIsInstance(inner, Shape)

    # unpack
    d0, d1 = w1[-2:]
    self.assertEqual((d0 * d1).subs({d0: 3, d1: 5}), 15)

    # __add__
    self.assertEqual(str(w1 + w2), '[3, 3, _d0, _d1, 5, 5, _d0, _d2]')

    # __radd_
    self.assertEqual(str([7] + w1), '[7, 3, 3, _d0, _d1]')
    self.assertEqual(str(w1[-2:] + w2[-1:]), '[_d0, _d1, _d2]')


if __name__ == '__main__':
  tf.test.main()
