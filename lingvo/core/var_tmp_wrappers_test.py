# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for var_tmp_wrappers."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core import var_tmp_wrappers


class VarTmpWrappersTest(test_utils.TestCase):

  def testVarWrapperTrackAssign(self):
    with tf.Graph().as_default():
      var = tf.get_variable('v0', shape=[8, 16], dtype=tf.float32)
      wrapper = var_tmp_wrappers.VarWrapperTrackAssign(var)
      ones = tf.ones_like(wrapper)
      a = wrapper.assign(ones)
      b = wrapper.assign_add(ones)
      c = wrapper.assign_sub(ones)
      self.assertSameElements(wrapper.previous_assigns(), [a, b, c])

  def testStackedVarWrapperWithManualSharding(self):
    with tf.Graph().as_default():
      var = tf.get_variable('v2', shape=[8, 16], dtype=tf.float32)
      wrapper = var_tmp_wrappers.StackedVarWrapperWithManualSharding(var)
      ones = tf.ones_like(wrapper)
      wrapper.assign(ones)
      wrapper.assign_add(ones)
      wrapper.assign_sub(ones)
      self.assertEqual(ones.shape, [16])


if __name__ == '__main__':
  py_utils.SetEagerMode(False)
  tf.test.main()
