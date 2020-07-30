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
"""Tests for functional ops."""

from lingvo import compat as tf
from lingvo.core import ops
from lingvo.core import test_utils
import numpy as np


class FunctionalOpsTest(test_utils.TestCase):

  def testCachedCall(self):
    # A generator returns different values for each invocation.
    def Gen():
      for i in range(1, 1000):
        yield np.array([[0, i], [i, 0]]).astype(np.float32), np.array(
            [[i, 0], [0, -i]]).astype(np.float32)

    it = Gen()

    # Wraps gen() in a defun.
    @tf.function(autograph=False)
    def MyFn():
      return tf.py_func(lambda: next(it), [], [tf.float32, tf.float32])

    # A graph calls MyFn via CachedCall.
    g = tf.Graph()
    with g.as_default():
      fn = MyFn.get_concrete_function()
      fn.add_to_graph()
      u, v = ops.cached_call(fn, [tf.float32, tf.float32])

    with self.session(graph=g):
      for _ in range(10):
        x, y = self.evaluate([u, v])
        self.assertAllEqual(x, [[0, 1], [1, 0]])
        self.assertAllEqual(y, [[1, 0], [0, -1]])


if __name__ == '__main__':
  tf.test.main()
