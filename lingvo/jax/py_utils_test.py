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
"""Tests for Python utils."""

from absl.testing import absltest
import jax
from jax import test_util
from lingvo.jax import py_utils
import tensorflow.compat.v2 as tf


class PyUtilsTest(test_util.JaxTestCase):

  def test_reshard_empty_array(self):
    batch_size = 128
    empty_inputs = tf.ones(shape=(batch_size, 0))
    sharded_inputs = py_utils.Reshard(empty_inputs)
    # Check the shape of returned inputs.
    num_devices = jax.local_device_count()
    self.assertEqual(sharded_inputs.shape,
                     (num_devices, batch_size // num_devices, 0))

  def test_sync_global_devices(self):
    py_utils.SyncGlobalDevices('sync')

if __name__ == '__main__':
  absltest.main()
