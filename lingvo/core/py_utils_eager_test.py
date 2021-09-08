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
"""Tests for py_utils in eager mode."""

from lingvo import compat as tf
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np


class PyUtilsEagerTest(test_utils.TestCase):

  def testCheckNumerics(self):
    checked = py_utils.CheckNumerics(
        tf.convert_to_tensor([2.0, 3.0], tf.float32))
    self.assertListEqual([2.0, 3.0], checked.numpy().tolist())

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'NaN'):
      py_utils.CheckNumerics(
          tf.reduce_mean(tf.convert_to_tensor([], tf.float32)))

  def testBatchNormUpdatesWithUpdateUseGlobalStatsForTraining(self):
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayer.Params()
    params.name = 'bn'
    params.dim = 3
    params.use_moving_avg_in_training = True
    params.params_init = py_utils.WeightInit.Gaussian(0.1)

    bn_layer = layers.BatchNormLayer(params)
    in_padding1 = tf.zeros([2, 8, 1], dtype=tf.float32)
    bn_in1 = tf.constant(
        np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

    bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
    sig1 = tf.reduce_sum(bn_out)
    sig2 = tf.reduce_sum(bn_out * bn_out)

    # IMPORTANT: Keep these values consistent with the corresponding
    # test in layers_test.py
    self.assertAllClose(2.6575434, sig1, atol=1e-5)
    self.assertAllClose(15.473802, sig2)

    updates_collection = tf.get_collection(py_utils.BATCH_NORM_UPDATES)
    l1, l2 = py_utils.FindRelevantBatchNormUpdates(bn_out, updates_collection)
    self.assertEqual(l1, [])
    self.assertEqual(l2, [])


if __name__ == '__main__':
  py_utils.SetEagerMode()
  tf.test.main()
