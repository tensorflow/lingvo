# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tpu_summary."""

from lingvo import compat as tf

from lingvo.core import test_utils
from lingvo.core import tpu_summary


class MockTransformer:

  def FProp(self, x, y):
    for i in range(3):
      with tf.name_scope('encoder%03d' % i):
        x = tf.identity(x)
        y = tf.identity(y)
        x = x + 1
        tpu_summary.scalar('x_mean', tf.reduce_mean(x))
        tpu_summary.scalar('y_mean', tf.reduce_mean(y))
    for i in range(3):
      with tf.name_scope('decoder%03d' % i):
        x = tf.identity(x)
        y = tf.identity(y)
        y = y + 1
        tpu_summary.scalar('x_mean', tf.reduce_mean(x))
        tpu_summary.scalar('y_mean', tf.reduce_mean(y))
    return x, y

  def BeamSearch(self, x, y, decoder_reduce_sum=False):
    for i in range(3):
      with tf.name_scope('encoder%03d' % i):
        x = tf.identity(x)
        y = tf.identity(y)
        x = x + 1
        tpu_summary.scalar('x_mean', tf.reduce_mean(x))
        tpu_summary.scalar('y_mean', tf.reduce_mean(y))

    def DecoderStep(x, y):
      for i in range(3):
        with tf.name_scope('decoder%03d' % i):
          x = tf.identity(x)
          y = tf.identity(y)
          y = y + 1
          if decoder_reduce_sum:
            tpu_summary.scalar(
                'x_mean', tf.reduce_mean(x), while_loop_reduce='sum')
            tpu_summary.scalar(
                'y_mean', tf.reduce_mean(y), while_loop_reduce='sum')
          else:
            tpu_summary.scalar('x_mean', tf.reduce_mean(x))
            tpu_summary.scalar('y_mean', tf.reduce_mean(y))
      return x, y

    def DecoderCond(x, y):
      del x, y
      return True

    (x, y) = tf.while_loop(
        cond=DecoderCond,
        body=DecoderStep,
        loop_vars=(x, y),
        maximum_iterations=10)
    return x, y


class TpuSummaryTest(test_utils.TestCase):

  def testNoContext(self):
    with self.session() as sess:
      model = MockTransformer()
      x = tf.constant(0, dtype=tf.float32)
      y = tf.constant(0, dtype=tf.int64)
      x, y = model.FProp(x, y)
      x, y = sess.run((x, y))
      self.assertEqual((3.0, 3), (x, y))

  def _CanonicalizeSummaryName(self, summaries):
    ret = dict()
    for k in summaries:
      ret[k.replace('/while', '')] = summaries[k]
    return ret

  def testMergeAll(self):
    with self.session() as sess:
      model = MockTransformer()
      x = tf.constant(0, dtype=tf.float32)
      y = tf.constant(0, dtype=tf.int64)
      with tpu_summary.context():
        x, y = model.FProp(x, y)
        summaries = tpu_summary.merge_all()
      x, y, summaries = sess.run((x, y, summaries))
      self.assertEqual((3.0, 3), (x, y))
      expected = {
          'x_mean/decoder000': 3.0,
          'x_mean/decoder001': 3.0,
          'x_mean/decoder002': 3.0,
          'x_mean/encoder000': 1.0,
          'x_mean/encoder001': 2.0,
          'x_mean/encoder002': 3.0,
          'y_mean/decoder000': 1,
          'y_mean/decoder001': 2,
          'y_mean/decoder002': 3,
          'y_mean/encoder000': 0,
          'y_mean/encoder001': 0,
          'y_mean/encoder002': 0,
      }
      self.assertEqual(expected, self._CanonicalizeSummaryName(summaries))

  def testWhileLoopNoMergeAll(self):
    with self.session() as sess:
      model = MockTransformer()
      x = tf.constant(0, dtype=tf.float32)
      y = tf.constant(0, dtype=tf.int64)
      with tpu_summary.context():
        x, y = model.BeamSearch(x, y)
      x, y = sess.run((x, y))
      self.assertEqual((3.0, 30), (x, y))

  def testWhileLoopNoRewrite(self):
    with self.session() as sess:
      model = MockTransformer()
      x = tf.constant(0, dtype=tf.float32)
      y = tf.constant(0, dtype=tf.int64)
      with tpu_summary.context():
        x, y = model.BeamSearch(x, y)
        # ValueError: Tensor decoder000/Mean:0 is not an element of this graph.
        with self.assertRaises(ValueError):
          summaries = tpu_summary.merge_all()
          x, y, summaries = sess.run((x, y, summaries))

  def testWhileLoopRewrite(self):
    with self.session() as sess:
      model = MockTransformer()
      x = tf.constant(0, dtype=tf.float32)
      y = tf.constant(0, dtype=tf.int64)
      with tpu_summary.context(rewrite_while_loop=True):
        x, y = model.BeamSearch(x, y)
        summaries = tpu_summary.merge_all()
      tf.logging.info('summaries=%r', summaries)
      x, y, summaries = sess.run((x, y, summaries))
      self.assertEqual((3.0, 30), (x, y))
      expected = {
          'x_mean/encoder000': 1.0,
          'x_mean/encoder001': 2.0,
          'x_mean/encoder002': 3.0,
          'y_mean/encoder000': 0,
          'y_mean/encoder001': 0,
          'y_mean/encoder002': 0,
          'x_mean/decoder000': 3.0,
          'x_mean/decoder001': 3.0,
          'x_mean/decoder002': 3.0,
          'y_mean/decoder000': 14.5,
          'y_mean/decoder001': 15.5,
          'y_mean/decoder002': 16.5,
      }
      self.assertEqual(expected, self._CanonicalizeSummaryName(summaries))

  def testWhileLoopRewriteMaxVarsLimit(self):
    with self.session() as sess:
      model = MockTransformer()
      x = tf.constant(0, dtype=tf.float32)
      y = tf.constant(0, dtype=tf.int64)
      with tpu_summary.context(rewrite_while_loop=True, max_loop_vars=2):
        x, y = model.BeamSearch(x, y)
        summaries = tpu_summary.merge_all()
      tf.logging.info('summaries=%r', summaries)
      x, y, summaries = sess.run((x, y, summaries))
      self.assertEqual((3.0, 30), (x, y))
      expected = {
          'x_mean/encoder000': 1.0,
          'x_mean/encoder001': 2.0,
          'x_mean/encoder002': 3.0,
          'y_mean/encoder000': 0,
          'y_mean/encoder001': 0,
          'y_mean/encoder002': 0,
          'x_mean/decoder000': 3.0,
          'y_mean/decoder000': 14.5,
      }
      self.assertEqual(expected, self._CanonicalizeSummaryName(summaries))

  def testWhileLoopReduceSum(self):
    with self.session() as sess:
      model = MockTransformer()
      x = tf.constant(0, dtype=tf.float32)
      y = tf.constant(0, dtype=tf.int64)
      with tpu_summary.context(rewrite_while_loop=True):
        x, y = model.BeamSearch(x, y, decoder_reduce_sum=True)
        summaries = tpu_summary.merge_all()
      tf.logging.info('summaries=%r', summaries)
      x, y, summaries = sess.run((x, y, summaries))
      self.assertEqual((3.0, 30), (x, y))
      expected = {
          'x_mean/encoder000': 1.0,
          'x_mean/encoder001': 2.0,
          'x_mean/encoder002': 3.0,
          'y_mean/encoder000': 0,
          'y_mean/encoder001': 0,
          'y_mean/encoder002': 0,
          'x_mean/decoder000': 30.0,
          'x_mean/decoder001': 30.0,
          'x_mean/decoder002': 30.0,
          'y_mean/decoder000': 145.0,
          'y_mean/decoder001': 155.0,
          'y_mean/decoder002': 165.0,
      }
      self.assertEqual(expected, self._CanonicalizeSummaryName(summaries))


if __name__ == '__main__':
  tf.test.main()
