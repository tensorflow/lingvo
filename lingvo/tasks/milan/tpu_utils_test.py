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
"""Tests for tpu_utils.py."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.milan import tpu_utils
import mock


class TpuUtilsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    # Pretend we're running on TPU.
    mock.patch.object(py_utils, 'use_tpu', return_value=True).start()
    self.addCleanup(mock.patch.stopall)

  def testCrossReplicaConcatInt64Emulation(self):
    """Verifies that ConcatenateAcrossReplicas() supports int64 inputs.

    Note this doesn't check general correctness of the function, just its
    ability to decompose int64s into int32s that can be handled by the
    underlying TPU op and to reconstruct the correct int64 output.
    """

    # Edge-case int64 inputs.
    inputs = [
        tf.int64.min, tf.int64.max, 1, -1, 2**32 + 1, -(2**32 + 1), 2**32 - 1,
        -(2**32 - 1)
    ]
    num_tpu_replicas = 4

    # Mock out the TPU op and replace it with the identity function.
    with mock.patch.object(
        tpu_utils.tf.raw_ops,
        'CollectivePermute',
        side_effect=lambda input, source_target_pairs: tf.identity(input)
    ) as mock_collective_permute:
      outputs = tpu_utils.ConcatenateAcrossReplicas(
          tf.constant(inputs, dtype=tf.int64), num_tpu_replicas)
      expected_outputs = inputs * num_tpu_replicas

    # Verify that int64s get translated into ops with int32 inputs.
    for _, call_kwargs in mock_collective_permute.call_args_list:
      self.assertEqual(tf.int32, call_kwargs['input'].dtype)

    self.assertEqual(outputs.dtype, tf.int64)
    with self.session() as sess:
      self.assertAllEqual(expected_outputs, sess.run(outputs))


if __name__ == '__main__':
  tf.test.main()
