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
"""Tests for best_step_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.core.ops import py_x_ops

FLAGS = tf.flags.FLAGS


class BestStepOp(test_utils.TestCase):

  def _HistFile(self):
    return test_helper.test_src_dir_path('core/ops/testdata/history.txt')

  def _BleuFile(self):
    return test_helper.test_src_dir_path('core/ops/testdata/history_bleu.txt')

  def _TfEventFile(self):
    return test_helper.test_src_dir_path(
        'core/ops/testdata/events.out.tfevents.test')

  def testTol0(self):
    g = tf.Graph()
    with g.as_default():
      output = py_x_ops.best_step(self._HistFile())
    with self.session(graph=g) as sess:
      best_step, last_step = sess.run(output)
      self.assertEqual(best_step, 42122)
      self.assertEqual(last_step, 42792)

  def testTolNon0(self):
    g = tf.Graph()
    with g.as_default():
      output = py_x_ops.best_step(self._HistFile(), 0.1)
    with self.session(graph=g) as sess:
      best_step, last_step = sess.run(output)
      self.assertEqual(best_step, 37553)
      self.assertEqual(last_step, 42792)

  def testNoFile(self):
    g = tf.Graph()
    with g.as_default():
      output = py_x_ops.best_step('')
    with self.session(graph=g) as sess:
      best_step, last_step = sess.run(output)
      self.assertEqual(best_step, 0)
      self.assertEqual(last_step, 0)

  def testAscendingValTol0(self):
    g = tf.Graph()
    with g.as_default():
      output = py_x_ops.best_step(self._BleuFile(), 0.0, False)
    with self.session(graph=g) as sess:
      best_step, last_step = sess.run(output)
      self.assertEqual(best_step, 41500)
      self.assertEqual(last_step, 46800)

  def testTfEventAscendingValTol0(self):
    g = tf.Graph()
    with g.as_default():
      output = py_x_ops.best_step(self._TfEventFile(), 0.0, False, 'bleu/dev')
    with self.session(graph=g) as sess:
      best_step, last_step = sess.run(output)
      self.assertEqual(best_step, 102600)
      self.assertEqual(last_step, 185200)


if __name__ == '__main__':
  tf.test.main()
