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
"""Test lm input generator."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import test_util
from lingvo.core import test_helper
from lingvo.jax import py_utils
from lingvo.jax.tasks.lm import input_generator
import numpy as np
import tensorflow.compat.v2 as tf


class InputTest(test_util.JaxTestCase):

  # We use first id and seq length as fingerprint to identify each shard
  # has the right elements.
  def _get_first_id_and_lengths(self, batch):
    return batch.labels.numpy()[:, 1], np.sum(
        batch.segment_ids.numpy(), axis=1, dtype=np.int32)

  def test_full(self):
    p = input_generator.TFRecordBertInput.Params()
    # There are 10 examples in this test data file.
    p.input_file = test_helper.test_src_dir_path(
        'jax/tasks/lm/testdata/tfrecords')
    p.batch_size = 10
    p.read_as_eval_data = True

    inp = p.Instantiate()
    batch = inp.GetPreprocessedInputBatch()
    ids, lengths = self._get_first_id_and_lengths(batch)
    expected_ids = np.array(
        [2003, 1996, 1996, 2049, 3748, 1007, 4862, 1996, 2004, 2002],
        dtype=np.int32)
    expected_lengths = np.array([35, 239, 55, 56, 511, 511, 161, 43, 416, 511],
                                dtype=np.int32)
    self.assertArraysEqual(ids, expected_ids)
    self.assertArraysEqual(lengths, expected_lengths)

  @parameterized.parameters(True, False)
  def test_sharded(self, provide_data_size):
    p = input_generator.TFRecordBertInput.Params()
    # There are 10 examples in this test data file.
    p.input_file = test_helper.test_src_dir_path(
        'jax/tasks/lm/testdata/tfrecords')
    p.batch_size = 4
    p.read_as_eval_data = True
    p.eval_data_size = 10 if provide_data_size else 0
    sharded_inputs = [None] * 4
    for i in range(4):
      with py_utils.InfeedContextScope(infeed_host_index=i, num_infeed_hosts=4):
        sharded_inputs[i] = p.Instantiate()

    # This is the same as in test_full() above.
    expected_ids = np.array(
        [2003, 1996, 1996, 2049, 3748, 1007, 4862, 1996, 2004, 2002],
        dtype=np.int32)
    expected_lengths = np.array([35, 239, 55, 56, 511, 511, 161, 43, 416, 511],
                                dtype=np.int32)
    expected_ids = np.reshape(
        np.concatenate(
            [expected_ids, np.array([0] * 6, dtype=np.int32)], axis=0), [4, -1])
    expected_lengths = np.reshape(
        np.concatenate([expected_lengths,
                        np.array([0] * 6, dtype=np.int32)],
                       axis=0), [4, -1])

    for i in [1, 3, 2, 0]:
      # each shard would produce one batch, and then out of range.
      batch = sharded_inputs[i].GetPreprocessedInputBatch()
      ids, lengths = self._get_first_id_and_lengths(batch)
      self.assertArraysEqual(ids, expected_ids[i])
      self.assertArraysEqual(lengths, expected_lengths[i])

      with self.assertRaisesRegex(tf.errors.OutOfRangeError, 'End of sequence'):
        sharded_inputs[i].GetPreprocessedInputBatch()

  def test_shard_helper(self):
    dataset = tf.data.Dataset.range(16)
    with py_utils.InfeedContextScope(infeed_host_index=1, num_infeed_hosts=4):
      self.assertEqual(
          list(
              input_generator.TFRecordBertInput.ShardData(
                  dataset).as_numpy_iterator()), [1, 5, 9, 13])


if __name__ == '__main__':
  absltest.main()
