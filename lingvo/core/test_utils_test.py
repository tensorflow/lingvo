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

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


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

  def testEagerSessionAdapterNoFunction(self):
    with self.session() as sess:
      self.assertIsInstance(sess, tf.Session)
      a = tf.Variable(1.0)
      b = tf.constant(2.0)
      c = a + b
      sess.run(tf.global_variables_initializer())
      self.assertEqual(3.0, sess.run(c))

  def testEagerSessionAdapterWithFunctionNoPlaceholder(self):
    with self.session() as sess:
      self.assertIsInstance(sess, tf.Session)

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
      self.assertIsInstance(sess, tf.Session)
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

  def testEagerSessionAdapterWithFunctionAcceptingNMap(self):
    with self.session() as sess:
      self.assertIsInstance(sess, tf.Session)
      a = tf.Variable(1, dtype=tf.int32)
      nmap = py_utils.NestedMap(
          x=tf.placeholder(tf.int32), y=tf.placeholder(tf.int32))
      traced = False

      @test_utils.DefineAndTrace(nmap)
      def func(nmap):
        nonlocal traced
        c = py_utils.Transform(lambda t: t + a, nmap)
        traced = True
        return c

      self.assertTrue(traced)
      sess.run(tf.global_variables_initializer())
      self.assertEqual(
          py_utils.NestedMap(x=3, y=4),
          sess.run(func, feed_dict=dict(zip(py_utils.Flatten(nmap), (2, 3)))))

  @test_utils.SkipIfNonEager
  def testGetScalarSummaryValuesTF2(self):
    logdir = self.create_tempdir()
    writer = tf.summary.create_file_writer(logdir.full_path)

    with writer.as_default():
      for step in range(100):
        tf.summary.scalar('dummy_metric', step * 2, step=step)
        writer.flush()

    summaries = self.GetScalarSummaryValuesTF2(logdir.full_path)
    self.assertEqual(
        summaries, {'dummy_metric': {s: s * 2 for s in range(100)}}
    )


if __name__ == '__main__':
  test_utils.main()
