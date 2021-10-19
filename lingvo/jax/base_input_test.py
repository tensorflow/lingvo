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
"""Tests for base_input."""

import os
from absl import flags
from absl.testing import absltest
from jax import test_util
from lingvo.core import base_input_generator
from lingvo.core import generic_input
from lingvo.jax import base_input
from lingvo.jax import py_utils
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


class TestInput(base_input.BaseInput):

  def __init__(self, params):
    super().__init__(params)
    self._dataset = None
    self._iter = None

  def GetNext(self) -> py_utils.NestedMap:
    assert tf.compat.v1.executing_eagerly()
    if self._dataset is None:
      self._dataset = self._GetDataset()
    if self._iter is None:
      self._iter = iter(self._dataset)
    ret = self._iter.get_next()
    return tf.nest.map_structure(lambda x: x.numpy(), ret)

  def Reset(self):
    if self.params.reset_for_eval:
      self._iter = iter(self._dataset)

  def _ToNestedMap(self, x) -> py_utils.NestedMap:
    t = tf.ones(shape=[4], dtype=tf.int32) * tf.cast(x, dtype=tf.int32)
    return py_utils.NestedMap(data=t)

  def _GetDataset(self):
    p = self.params
    d = tf.data.Dataset.range(10)
    d = d.shard(p.num_infeed_hosts, p.infeed_host_index)
    d = d.shuffle(10, seed=p.input_random_seed).repeat(-1)
    if p.reset_for_eval:
      d = d.take(p.batch_size * 2)
    d = d.map(self._ToNestedMap)
    d = d.batch(p.batch_size)
    return d


class LingvoInput(base_input_generator.BaseInputGeneratorFromFiles):

  def _DataSourceFromFilePattern(self, file_pattern, input_source_weights=None):
    assert not tf.compat.v1.executing_eagerly()
    assert tf.compat.v1.executing_eagerly_outside_functions()

    def _process(source_id, record):
      del source_id
      num = tf.strings.to_number(record, tf.int32)
      return py_utils.NestedMap(num=num), 1

    inputs, _ = generic_input.GenericInput(
        processor=_process,
        file_pattern=file_pattern,
        file_random_seed=0,
        require_sequential_order=True,
        repeat_count=1,
        file_buffer_size=32,
        file_parallelism=1,
        bucket_upper_bound=[10],
        bucket_batch_limit=[2])
    return inputs


class InputTest(test_util.JaxTestCase):

  def test_lingvo_input(self):
    tmp = os.path.join(FLAGS.test_tmpdir, 'tmptest')
    batch_size = 2
    num_batches = 10
    num_data = batch_size * num_batches
    with tf.io.TFRecordWriter(tmp) as w:
      for i in range(num_data):
        w.write(('%04d' % i).encode('utf-8'))

    p = base_input.LingvoInputAdaptor.Params()
    p.input = LingvoInput.Params()
    p.input.file_pattern = 'tfrecord:' + tmp
    inp = p.Instantiate()
    for i in range(num_batches):
      batch = inp.GetNext()
      self.assertArraysEqual(
          np.array([2 * i, 2 * i + 1], dtype=np.int32), batch.num)
    with self.assertRaisesRegex(tf.errors.OutOfRangeError,
                                'SequentialRecordYielder reached 1 repeat'):
      inp.GetNext()

  def test_tfdata_input(self):
    p = TestInput.Params()
    p.num_infeed_hosts = 3
    p.input_random_seed = 345
    p.batch_size = 2
    train = [None] * p.num_infeed_hosts
    test = [None] * p.num_infeed_hosts
    for i in range(p.num_infeed_hosts):
      train_p = p.Copy().Set(infeed_host_index=i)
      test_p = train_p.Copy().Set(reset_for_eval=True)
      train[i] = train_p.Instantiate()
      test[i] = test_p.Instantiate()

    num_train_batches = 10
    for _ in range(num_train_batches):
      for i in range(p.num_infeed_hosts):
        batch = train[i].GetNext()
        self.assertTrue(np.all(batch.data % p.num_infeed_hosts == i))

    num_test_batches = 2
    for _ in range(num_test_batches):
      for i in range(p.num_infeed_hosts):
        batch = test[i].GetNext()
        self.assertTrue(np.all(batch.data % p.num_infeed_hosts == i))
    for i in range(p.num_infeed_hosts):
      with self.assertRaisesRegex(tf.errors.OutOfRangeError, 'End of sequence'):
        batch = test[i].GetNext()

    # input works again after Reset().
    for i in range(p.num_infeed_hosts):
      test[i].Reset()
      batch = test[i].GetNext()
      self.assertEqual(batch.data[0, 0] % p.num_infeed_hosts, i)


if __name__ == '__main__':
  absltest.main()
