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
"""Helpers for unittests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import re

import numpy as np
from six.moves import range
import tensorflow as tf
from lingvo.core import py_utils

tf.flags.DEFINE_boolean(
    'update_goldens', False,
    'Update the goldens, rather than diffing against them.')

FLAGS = tf.flags.FLAGS


class TestCase(tf.test.TestCase):
  """TestCase that performs Lingvo-specific setup."""

  def setUp(self):
    super(TestCase, self).setUp()
    # Ensure the global_step variable is created in the default graph.
    py_utils.GetOrCreateGlobalStepVar()

  def _create_session(self, *args, **kwargs):
    sess = super(TestCase, self)._create_session(*args, **kwargs)
    with sess.graph.as_default():
      # Ensure the global_step variable is created in every new session.
      py_utils.GetOrCreateGlobalStepVar()
    return sess


def _ReplaceOneLineInFile(fpath, linenum, old, new):
  lines = []
  lines = open(fpath).readlines()
  assert lines[linenum] == old, (
      'Expected "%s" at line %d in file %s, but got "%s"' %
      (lines[linenum], linenum + 1, fpath, old))
  lines[linenum] = new
  with open(fpath, 'w') as f:
    for l in lines:
      f.write(l)


def ReplaceGoldenSingleFloat(old, float_value):
  m = re.match(
      r'(?P<prefix>.*)\bCompareToGoldenSingleFloat\('
      r'(?P<testobj>[^,]+), *'
      r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?, *'
      r'(?P<v2>.*)\)(?P<postfix>.*)\n', old)
  assert m
  return ('%sCompareToGoldenSingleFloat(%s, %f, %s)%s\n' %
          (m.group('prefix'), m.group('testobj'), float_value, m.group('v2'),
           m.group('postfix')))


def ReplaceGoldenStackAnalysis(new_float_value):
  """Analyze the stack trace to figure out how to update the golden value."""
  src_file_frame = None
  for frame in inspect.stack():
    if frame[-2] and 'CompareToGoldenSingleFloat' in frame[-2][0]:
      src_file_frame = frame
      break
  assert src_file_frame
  runfiles_pattern = 'runfiles/[^/]+/'
  fpath = src_file_frame[1].split(runfiles_pattern)[-1]
  line_num = src_file_frame[2] - 1
  old_line = src_file_frame[4][0]
  new_line = ReplaceGoldenSingleFloat(old_line, new_float_value)
  return fpath, line_num, old_line, new_line


def CompareToGoldenSingleFloat(testobj, v1, v2, *args, **kwargs):
  if not FLAGS.update_goldens:
    testobj.assertAllClose(v1, v2, *args, **kwargs)
  else:
    _ReplaceOneLineInFile(*ReplaceGoldenStackAnalysis(v2))


def PickEveryN(np_arr, step=1):
  """Flattens `np_arr` and keeps one value every step values."""
  return np_arr.flatten()[::step]


def ComputeNumericGradient(sess, y, x, delta=1e-4, step=1,
                           extra_feed_dict=None):
  """Compute the numeric gradient of y wrt to x.

  Args:
    sess: The TF session constructed with a graph containing x and y.
    y: A scalar TF Tensor in the graph constructed in sess.
    x: A TF Tensor in the graph constructed in sess.
    delta: Gradient checker's small perturbation of x[i].
    step: Only compute numerical gradients for a subset of x values.
      I.e. dy/dx[i] is computed if i % step == 0.
    extra_feed_dict: Additional feed_dict of tensors to keep fixed during the
      gradient checking.

  Returns:
    A Tensor of the same shape and dtype as x. If x[i] is not chosen
    to compute the numerical gradient dy/x[i], the corresponding
    value is set to 0.
  """

  x_data = sess.run(x)
  x_size = x_data.size
  x_shape = x_data.shape

  numeric_grad = np.zeros(x_size, dtype=x_data.dtype)

  for i in range(0, x_size, step):
    x_pos = x_data.copy()
    if x_size == 1:
      x_pos += delta
    else:
      x_pos.flat[i] += delta
    y_pos_feed_dict = extra_feed_dict or {}
    y_pos_feed_dict.update(dict([(x.name, x_pos)]))
    y_pos = sess.run(y, feed_dict=y_pos_feed_dict)
    x_neg = x_data.copy()
    if x_size == 1:
      x_neg -= delta
    else:
      x_neg.flat[i] -= delta
    y_neg_feed_dict = extra_feed_dict or {}
    y_neg_feed_dict.update(dict([(x.name, x_neg)]))
    y_neg = sess.run(y, feed_dict=y_neg_feed_dict)
    numeric_grad[i] = (y_pos - y_neg) / (2 * delta)
  return numeric_grad.reshape(x_shape)
