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
"""Tests for lingvo.core.symbolic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo.core import symbolic
from lingvo.core import test_utils
import sympy
import tensorflow as tf


class SymbolicTest(test_utils.TestCase):

  def testGetSymbol(self):
    x = symbolic.NewSymbol('x')
    self.assertIsInstance(x, sympy.Expr)

  def testEvalExpr(self):
    x = symbolic.NewSymbol('x')
    y = symbolic.NewSymbol('y')
    xy = x * y

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)

    with symbolic.SymbolToValueMap({x: 2, y: 3}):
      self.assertEqual(symbolic.EvalExpr(xy), 6)
      # The inner map overrides the outer map.
      with symbolic.SymbolToValueMap({x: a, y: b}):
        ab = symbolic.EvalExpr(xy)
      self.assertEqual(symbolic.EvalExpr(xy), 6)

    # EvalExpr can also evaluate a symbolic expression to a
    # Tensor.
    self.assertIsInstance(ab, tf.Tensor)
    with self.session() as sess:
      self.assertEqual(12, sess.run(ab, {a: 3, b: 4}))

    with self.assertRaises(Exception):
      # EvalExpr does not support partial evaluation.
      with symbolic.SymbolToValueMap({y: 3}):
        symbolic.EvalExpr(xy)


if __name__ == '__main__':
  tf.test.main()
