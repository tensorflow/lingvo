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
"""Tests for compat.py."""

import lingvo.compat as tf
import tensorflow.compat.v1 as tf1  # TF_DIRECT_IMPORT
import tensorflow.compat.v2 as tf2  # TF_DIRECT_IMPORT
# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tf2 import enabled as tf2_enabled
# pylint:enable=g-direct-tensorflow-import


class CompatTest(tf.test.TestCase):

  def testSomeTFSymbols(self):
    self.assertIsNotNone(tf.logging)
    self.assertIsNotNone(tf.flags)

  def testDoesNotModifyTF2(self):
    modules_no_overwritten = [
        (tf2.data, tf1.data),
        (tf2.graph_util, tf1.graph_util),
        (tf2.image, tf1.image),
        (tf2.initializers, tf1.initializers),
        (tf2.io, tf1.io),
        (tf2.losses, tf1.losses),
        (tf2.metrics, tf1.metrics),
        (tf2.nn, tf1.nn),
        (tf2.random, tf1.random),
        (tf2.saved_model, tf1.saved_model),
        (tf2.strings, tf1.strings),
        (tf2.summary, tf1.summary),
        (tf2.test, tf1.test),
        (tf2.train, tf1.train),
    ]
    for modules in modules_no_overwritten:
      self.assertIsNot(modules[0], modules[1])

  def testTF2Enabled(self):
    self.assertTrue(tf2_enabled())
    self.assertEqual(2, tf._major_api_version)


if __name__ == '__main__':
  tf.test.main()
