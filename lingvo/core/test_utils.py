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
"""Helpers for unittests."""

import contextlib
import inspect
import re
import sys

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
import numpy as np

tf.flags.DEFINE_bool('enable_eager_execution', False,
                     'Whether to enable eager execution.')

tf.flags.DEFINE_boolean(
    'update_goldens', False,
    'Update the goldens, rather than diffing against them.')

FLAGS = tf.flags.FLAGS

# Disable eager mode for tests by default.
py_utils.SetEagerMode(False)


class TestCase(tf.test.TestCase):
  """TestCase that performs Lingvo-specific setup."""

  def setUp(self):
    super().setUp()
    # Ensure the global_step variable is created in the default graph.
    py_utils.GetOrCreateGlobalStepVar()
    cluster = cluster_factory.SetRequireSequentialInputOrder(True)
    cluster.params.in_unit_test = True
    cluster.__enter__()

  def _create_session(self, *args, **kwargs):
    sess = super()._create_session(*args, **kwargs)
    with sess.graph.as_default():
      # Ensure the global_step variable is created in every new session.
      global_step = py_utils.GetOrCreateGlobalStepVar()
      sess.run(
          tf.cond(
              tf.is_variable_initialized(global_step), tf.no_op,
              lambda: tf.variables_initializer([global_step])))
    return sess

  @contextlib.contextmanager
  def session(self, *args, **kwargs):
    """Test session context manager."""
    with super().session(*args, **kwargs) as sess:
      with py_utils.UnitTestSessionScope(sess):
        yield sess

  def SetEval(self, mode):
    return cluster_factory.SetEval(mode=mode)


def _ReplaceOneLineInFile(fpath, linenum, old, new):
  """Replaces a line for the input file."""
  lines = []
  lines = open(fpath).readlines()
  assert lines[linenum] == old, (
      'Expected "%s" at line %d in file %s, but got "%s"' %
      (lines[linenum], linenum + 1, fpath, old))
  tf.logging.info('Replacing {}:{}.'.format(fpath, linenum))
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
  """Compare golden value with real value.

  When running the bazel tests with FLAGS.update_goldens to be True, this
  function automatically updates the golden value in the test file if there is a
  mismatch and the calling site of CompareToGoldenSingleFloat is a 1-liner. E.g.
  Code::

    test_utils.CompareToGoldenSingleFloat(self, 0.3232, input_batch.label)

  works but this will not::

    test_utils.CompareToGoldenSingleFloat(self,
                                          0.3232,
                                          input_batch.label)

  Args:
    testobj: A test object, such as tf.test.TestCase or test_utils.TestCase.
    v1: the golden value to compare against.
    v2: the returned value.
    *args: extra args
    **kwargs: extra args
  """
  if not FLAGS.update_goldens:
    testobj.assertAllClose(v1, v2, *args, **kwargs)
  else:
    _ReplaceOneLineInFile(*ReplaceGoldenStackAnalysis(v2))


def PickEveryN(np_arr, step=1):
  """Flattens `np_arr` and keeps one value every step values."""
  return np_arr.flatten()[::step]


def ComputeNumericGradient(sess,
                           y,
                           x,
                           delta=1e-4,
                           step=1,
                           extra_feed_dict=None):
  """Compute the numeric gradient of y wrt to x.

  Args:
    sess: The TF session constructed with a graph containing x and y.
    y: A scalar TF Tensor in the graph constructed in sess.
    x: A TF Tensor in the graph constructed in sess.
    delta: Gradient checker's small perturbation of x[i].
    step: Only compute numerical gradients for a subset of x values. I.e.
      dy/dx[i] is computed if i % step == 0.
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

  # For variables we need to issue an assignment operation in order to update
  # the value of the variable. This is because with resource variables x will be
  # pointing to the handle rather than its value.
  feed_dict = extra_feed_dict or {}
  ph = tf.placeholder(x_data.dtype, x_shape)
  x_assign = x.assign(ph) if isinstance(x, tf.Variable) else None

  for i in range(0, x_size, step):
    x_pos = x_data.copy()
    if x_size == 1:
      x_pos += delta
    else:
      x_pos.flat[i] += delta
    if x_assign is None:
      feed_dict.update(dict([(x, x_pos)]))
    else:
      sess.run(x_assign, feed_dict={ph: x_pos})
    y_pos = sess.run(y, feed_dict=feed_dict)

    x_neg = x_data.copy()
    if x_size == 1:
      x_neg -= delta
    else:
      x_neg.flat[i] -= delta
    if x_assign is None:
      feed_dict.update(dict([(x, x_neg)]))
    else:
      sess.run(x_assign, feed_dict={ph: x_neg})
    y_neg = sess.run(y, feed_dict=feed_dict)
    numeric_grad[i] = (y_pos - y_neg) / (2 * delta)

  # Restore the variable back to its original value to avoid breaking any
  # further test code that operates on the graph.
  if x_assign is not None:
    sess.run(x_assign, feed_dict={ph: x_data})

  return numeric_grad.reshape(x_shape)


def main(*args, **kwargs):
  FLAGS(sys.argv, known_only=True)
  py_utils.SetEagerMode(FLAGS.enable_eager_execution)
  FLAGS.unparse_flags()
  tf.test.main(*args, **kwargs)
