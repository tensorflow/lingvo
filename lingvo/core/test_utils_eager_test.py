# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import lingvo.compat as tf
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


class TestUtilsEagerTest(test_utils.TestCase):

  def testEagerSessionAdapterNoFunction(self):
    with self.session() as sess:
      self.assertIsInstance(sess, test_utils._EagerSessionAdaptor)
      a = tf.Variable(1.0)
      b = tf.constant(2.0)
      c = a + b
      sess.run(tf.global_variables_initializer())
      self.assertEqual(3.0, sess.run(c))

  def testEagerSessionAdapterWithFunctionNoPlaceholder(self):
    with self.session() as sess:
      self.assertIsInstance(sess, test_utils._EagerSessionAdaptor)

      a = tf.Variable(1.0)
      traced = False

      @test_utils.DefineAndTrace()
      def func():
        nonlocal traced
        b = tf.constant(2.0)
        c = a + b
        traced = True
        return c

      self.assertTrue(traced)
      sess.run(tf.global_variables_initializer())
      self.assertEqual(3.0, sess.run(func))

  def testEagerSessionAdapterWithFunctionAndPlaceholder(self):
    with self.session() as sess:
      self.assertIsInstance(sess, test_utils._EagerSessionAdaptor)
      a = tf.Variable(1.0)
      b = tf.placeholder(tf.float32)
      traced = False

      @test_utils.DefineAndTrace(b)
      def func(b):
        nonlocal traced
        c = a + b
        traced = True
        return c

      self.assertTrue(traced)
      sess.run(tf.global_variables_initializer())
      self.assertEqual(3.0, sess.run(func, feed_dict={b: 2.0}))


if __name__ == '__main__':
  test_utils.main()
