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

import lingvo.compat as tf
from lingvo.core import symbolic
from lingvo.core import test_utils
import sympy


class SymbolicTest(test_utils.TestCase):

  def testGetSymbol(self):
    x = symbolic.Symbol('x')
    self.assertIsInstance(x, sympy.Expr)

  def testEvalExpr(self):
    x = symbolic.Symbol('x')
    y = symbolic.Symbol('y')
    xy = x * y

    with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES, {x: 2, y: 3}):
      self.assertEqual(symbolic.EvalExpr(symbolic.STATIC_VALUES, xy), 6)
      # The inner map overrides the outer map.
      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES, {x: 5, y: 6}):
        self.assertEqual(symbolic.EvalExpr(symbolic.STATIC_VALUES, xy), 30)
      # Back to the outer map.
      self.assertEqual(symbolic.EvalExpr(symbolic.STATIC_VALUES, xy), 6)

    # EvalExpr can also evaluate a symbolic expression to a
    # Tensor.
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    with symbolic.SymbolToValueMap(symbolic.TENSOR_VALUES, {x: a, y: b}):
      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES, {x: 2, y: 3}):
        # Value maps of different types do not affect each other.
        self.assertEqual(symbolic.EvalExpr(symbolic.STATIC_VALUES, xy), 6)
        ab = symbolic.EvalExpr(symbolic.TENSOR_VALUES, xy)
        self.assertIsInstance(ab, tf.Tensor)
        with self.session() as sess:
          self.assertEqual(12, sess.run(ab, {a: 3, b: 4}))

      # EvalExpr supports partial evaluation.
      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES, {y: 3}):
        x3 = symbolic.EvalExpr(symbolic.STATIC_VALUES, xy)
        with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES, {x: 9}):
          self.assertEqual(27, symbolic.EvalExpr(symbolic.STATIC_VALUES, x3))


if __name__ == '__main__':
  tf.test.main()
