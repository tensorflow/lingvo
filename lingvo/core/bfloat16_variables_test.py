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
"""Tests for bfloat16_variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from lingvo.core import bfloat16_variables
from lingvo.core import test_utils


class Bfloat16VariablesTest(test_utils.TestCase):

  def testBfloat16Reload(self):
    checkpoint_path = os.path.join(self.get_temp_dir(), "bfloat16_restore")

    # Create a resource variable of type tf.float32 and save them to disk.
    g_for_save_graph = tf.Graph()
    fl = 0.99
    with self.session(graph=g_for_save_graph) as sess:
      v0 = tf.Variable(fl, name="v0", dtype=tf.float32, use_resource=True)
      tf.global_variables_initializer().run()
      self.assertAlmostEqual(fl, v0.eval())

      saver = tf.train.Saver({
          "v0": v0,
      }, restore_sequentially=True)
      val = saver.save(sess, checkpoint_path)
      self.assertEqual(checkpoint_path, val)

    # Restore the variable as bfloat16.
    g_for_restore_graph = tf.Graph()
    with self.session(graph=g_for_restore_graph) as sess:
      v0 = tf.Variable(0.0, name="v0", dtype=tf.bfloat16, use_resource=True)
      tf.global_variables_initializer().run()
      self.assertAlmostEqual(0.0, v0.eval())
      saveable = bfloat16_variables.Bfloat16VariableSaveable(
          v0, tf.float32, "", "v0")
      saver = tf.train.Saver({"v0": saveable}, restore_sequentially=True)
      saver.restore(sess, checkpoint_path)
      self.assertAlmostEqual(fl, v0.eval())


if __name__ == "__main__":
  tf.test.main()
