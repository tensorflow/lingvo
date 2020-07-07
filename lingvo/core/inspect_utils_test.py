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
"""Tests for inspect_utils."""

import lingvo.compat as tf
from lingvo.core import hyperparams
from lingvo.core import inspect_utils
from lingvo.core import test_utils


class InspectUtilsTest(test_utils.TestCase):

  def testBareFunction(self):

    def my_function(a, b):
      return a + 1, b + 2

    params = hyperparams.Params()
    inspect_utils.DefineParams(my_function, params)
    self.assertIn('a', params)
    self.assertIn('b', params)
    self.assertIsNone(params.a)
    self.assertIsNone(params.b)

    params.a = 5
    params.b = 6
    a1, b1 = inspect_utils.CallWithParams(my_function, params)
    self.assertEqual(a1, 5 + 1)
    self.assertEqual(b1, 6 + 2)

  def testFunctionWithDefaults(self):

    def my_function(a, b=3):
      return a + 1, b + 2

    params = hyperparams.Params()
    inspect_utils.DefineParams(my_function, params)
    self.assertIn('a', params)
    self.assertIn('b', params)
    self.assertIsNone(params.a)
    self.assertEqual(params.b, 3)

    params.a = 6
    a1, b1 = inspect_utils.CallWithParams(my_function, params)
    self.assertEqual(a1, 6 + 1)
    self.assertEqual(b1, 3 + 2)

  def testFunctionWithIgnore(self):

    def my_function(a, b=3, c=4):
      return a + 1, b + 2, c + 3

    params = hyperparams.Params()
    inspect_utils.DefineParams(my_function, params, ignore=['c'])
    self.assertIn('a', params)
    self.assertIn('b', params)
    self.assertNotIn('c', params)
    self.assertIsNone(params.a)
    self.assertEqual(params.b, 3)

    params.a = 6
    a1, b1, c1 = inspect_utils.CallWithParams(my_function, params, c=9)
    self.assertEqual(a1, 6 + 1)
    self.assertEqual(b1, 3 + 2)
    self.assertEqual(c1, 9 + 3)

  def testFunctionWithOverrides(self):

    def my_function(a, b=3):
      return a + 1, b + 2

    params = hyperparams.Params()
    inspect_utils.DefineParams(my_function, params)
    self.assertIn('a', params)
    self.assertIn('b', params)
    self.assertIsNone(params.a)
    self.assertEqual(params.b, 3)

    params.a = 6
    a1, b1 = inspect_utils.CallWithParams(my_function, params, a=7)
    self.assertEqual(a1, 7 + 1)
    self.assertEqual(b1, 3 + 2)

  def testFunctionWithVarArgs(self):

    def my_function(a, *args, b=3, **kwargs):
      del args
      del kwargs
      return a + 1, b + 2

    params = hyperparams.Params()
    inspect_utils.DefineParams(my_function, params)
    self.assertIn('a', params)
    self.assertNotIn('args', params)
    self.assertIn('b', params)
    self.assertNotIn('kwargs', params)
    self.assertIsNone(params.a)
    self.assertEqual(params.b, 3)

    params.a = 6
    a1, b1 = inspect_utils.CallWithParams(my_function, params)
    self.assertEqual(a1, 6 + 1)
    self.assertEqual(b1, 3 + 2)

  def testClassInit(self):

    class MyClass:

      def __init__(self, a, b=3):
        self.a = a
        self.b = b

    params = hyperparams.Params()
    inspect_utils.DefineParams(MyClass, params)
    self.assertIn('a', params)
    self.assertIn('b', params)
    self.assertIsNone(params.a)
    self.assertEqual(params.b, 3)

    params.a = 9
    params.b = 5
    obj = inspect_utils.CallWithParams(MyClass, params)
    self.assertEqual(obj.a, 9)
    self.assertEqual(obj.b, 5)

  # TODO(oday): Remove this test when the bug on Keras has been resolved.
  def testClassInit2(self):

    class MyClass:

      def __init__(self, a, b=3):
        self.a = a
        self.b = b

    params = hyperparams.Params()
    inspect_utils.DefineParams(MyClass.__init__, params, bound=True)
    self.assertNotIn('self', params)
    self.assertIn('a', params)
    self.assertIn('b', params)
    self.assertIsNone(params.a)
    self.assertEqual(params.b, 3)

    params.a = 9
    params.b = 5
    obj = inspect_utils.ConstructWithParams(MyClass, params)
    self.assertEqual(obj.a, 9)
    self.assertEqual(obj.b, 5)

  def testMethod(self):

    class MyClass:

      def __init__(self):
        self._s = 'a/b'

      def split(self, sep):
        return self._s.split(sep)

    params = hyperparams.Params()
    inspect_utils.DefineParams(MyClass.split, params, bound=True)
    self.assertNotIn('self', params)
    self.assertIn('sep', params)
    self.assertIsNone(params.sep)

    params.sep = '/'
    parts = inspect_utils.CallWithParams(MyClass().split, params)
    self.assertEqual(['a', 'b'], parts)


if __name__ == '__main__':
  tf.test.main()
