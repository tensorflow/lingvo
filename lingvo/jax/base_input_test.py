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
from lingvo.core import base_input_generator
from lingvo.core import generic_input
from lingvo.core import py_utils as tf_py_utils
from lingvo.jax import base_input
from lingvo.jax import py_utils
from lingvo.jax import test_utils
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


class TestInput(base_input.BaseInput):

  def __init__(self, params):
    super().__init__(params)
    self._dataset = self._get_dataset()
    self._iter = iter(self._dataset)

  def get_next(self) -> py_utils.NestedMap:
    assert tf.compat.v1.executing_eagerly()
    ret = self._iter.get_next()
    return tf.nest.map_structure(lambda x: x.numpy(), ret)

  def reset(self):
    if self.params.reset_for_eval:
      self._iter = iter(self._dataset)

  def _to_nested_map(self, x) -> py_utils.NestedMap:
    t = tf.ones(shape=[4], dtype=tf.int32) * tf.cast(x, dtype=tf.int32)
    return py_utils.NestedMap(data=t)

  def _get_dataset(self):
    p = self.params
    d = tf.data.Dataset.range(10)
    d = d.shard(p.num_infeed_hosts, p.infeed_host_index)
    d = d.shuffle(10, seed=p.input_random_seed).repeat(-1)
    if p.reset_for_eval:
      d = d.take(p.batch_size * 2)
    d = d.map(self._to_nested_map)
    d = d.batch(p.batch_size)
    return d


class LingvoInput(base_input_generator.BaseInputGeneratorFromFiles):

  def _DataSourceFromFilePattern(self,
                                 file_pattern,
                                 input_source_weights=None,
                                 **extra_input_kwargs):
    assert not tf.compat.v1.executing_eagerly()
    assert tf.compat.v1.executing_eagerly_outside_functions()

    def _process(source_id, record):
      del source_id
      num = tf.strings.to_number(record, tf.int32)
      if not tf_py_utils.use_tpu():
        num = num * num
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


def _get_test_dataset(num: int) -> tf.data.Dataset:

  def to_map(i: int):
    return {'data': i}

  return tf.data.Dataset.range(num).map(to_map)


TestDataset = base_input_generator.DefineTFDataInput('TestDataset',
                                                     _get_test_dataset)


class TestDatasetOverride(TestDataset):

  def GetPreprocessedInputBatch(self) -> py_utils.NestedMap:
    batch = super().GetPreprocessedInputBatch()
    assert isinstance(batch, py_utils.NestedMap)
    batch.data2 = batch.data * 2 + 1
    return batch


class InputTest(test_utils.TestCase):

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
    p.input.file_random_seed = 0
    p.reset_for_eval = True
    inp = p.Instantiate()
    for i in range(num_batches):
      batch = inp.get_next()
      self.assertArraysEqual(
          np.array([2 * i, 2 * i + 1], dtype=np.int32), batch.num)
    with self.assertRaisesRegex(tf.errors.OutOfRangeError,
                                'SequentialRecordYielder reached 1 repeat'):
      inp.get_next()
    inp.reset()
    for i in range(num_batches):
      batch = inp.get_next()
      self.assertArraysEqual(
          np.array([2 * i, 2 * i + 1], dtype=np.int32), batch.num)
    del inp

    # Test that we can force a raise earlier manually.
    smaller_num_batches = 4
    p2 = p.Copy().Set(num_batches=smaller_num_batches)
    inp2 = p2.Instantiate()
    for i in range(smaller_num_batches):
      batch = inp2.get_next()
      self.assertArraysEqual(
          np.array([2 * i, 2 * i + 1], dtype=np.int32), batch.num)
    with self.assertRaisesRegex(tf.errors.OutOfRangeError,
                                f'num_batches exceeding {smaller_num_batches}'):
      inp2.get_next()
    inp2.reset()
    batch = inp2.get_next()
    self.assertArraysEqual(np.array([0, 1], dtype=np.int32), batch.num)

  def test_lingvo_input_change_batch_size(self):
    tmp = os.path.join(FLAGS.test_tmpdir, 'tmptest2')
    batch_size = 2
    num_batches = 6
    num_data = batch_size * num_batches
    with tf.io.TFRecordWriter(tmp) as w:
      for i in range(num_data):
        w.write(('%04d' % i).encode('utf-8'))

    p = base_input.LingvoInputAdaptorNewBatchSize.Params()
    p.input = LingvoInput.Params()
    p.input.file_pattern = 'tfrecord:' + tmp
    p.input.file_random_seed = 0
    p.batch_size = 1
    p.reset_for_eval = True
    inp = p.Instantiate()
    for i in range(num_batches * 2):
      batch = inp.get_next()
      self.assertArraysEqual(np.array([i], dtype=np.int32), batch.num)
    with self.assertRaises(tf.errors.OutOfRangeError):
      inp.get_next()

  def test_lingvo_tfdata_input(self):
    num_batches = 10
    input_p = TestDataset.Params()
    input_p.args.num = num_batches
    p = base_input.LingvoInputAdaptor.Params().Set(
        input=input_p, is_training=True, cluster_do_eval=True)
    inp = p.Instantiate()
    # When used for training data, cluster.do_eval is never set.
    # The input repeats the data indefinitely.
    for i in range(int(num_batches * 2.5)):
      x = inp.get_next()
      self.assertEqual(x.data, i % num_batches)
    # Resets the input to begin from the first element again.
    inp.reset()
    x = inp.get_next()
    self.assertEqual(x.data, 0)

  def test_lingvo_tfdata_input_eval(self):
    num_batches = 10
    input_p = TestDataset.Params()
    input_p.args.num = num_batches
    # We have two versions of the input, with different values for
    # cluster.do_eval.
    p_eval = base_input.LingvoInputAdaptor.Params().Set(
        input=input_p, is_training=False, cluster_do_eval=True)
    p_noeval = base_input.LingvoInputAdaptor.Params().Set(
        input=input_p, is_training=False, cluster_do_eval=False)
    inp_eval = p_eval.Instantiate()
    inp_noeval = p_noeval.Instantiate()
    for i in range(num_batches):
      self.assertEqual(inp_eval.get_next().data, i)
      self.assertEqual(inp_noeval.get_next().data, i)
    # When cluster.do_eval is set, the input exhausts one epoch and raises.
    with self.assertRaisesRegex(tf.errors.OutOfRangeError, 'End of sequence'):
      inp_eval.get_next()
    # When cluster.do_eval is not set (the default), the input repeats.
    self.assertEqual(inp_noeval.get_next().data, 0)
    # Resets the input to begin from the first element again.
    inp_eval.reset()
    self.assertEqual(inp_eval.get_next().data, 0)

  def test_lingvo_tfdata_override(self):
    num_batches = 10
    input_p = TestDatasetOverride.Params()
    input_p.args.num = num_batches
    p = base_input.LingvoInputAdaptor.Params().Set(
        input=input_p, is_training=True)
    inp = p.Instantiate()
    for i in range(int(num_batches * 2.5)):
      x = inp.get_next()
      self.assertEqual(x.data, i % num_batches)
      self.assertEqual(x.data2, (i % num_batches) * 2 + 1)
    inp.reset()
    x = inp.get_next()
    self.assertEqual(x.data, 0)
    self.assertEqual(x.data2, 1)

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
        batch = train[i].get_next()
        self.assertTrue(np.all(batch.data % p.num_infeed_hosts == i))

    num_test_batches = 2
    for _ in range(num_test_batches):
      for i in range(p.num_infeed_hosts):
        batch = test[i].get_next()
        self.assertTrue(np.all(batch.data % p.num_infeed_hosts == i))
    for i in range(p.num_infeed_hosts):
      with self.assertRaisesRegex(tf.errors.OutOfRangeError, 'End of sequence'):
        batch = test[i].get_next()

    # input works again after reset().
    for i in range(p.num_infeed_hosts):
      test[i].reset()
      batch = test[i].get_next()
      self.assertEqual(batch.data[0, 0] % p.num_infeed_hosts, i)

  def test_validate_batch_size(self):
    tmp = os.path.join(FLAGS.test_tmpdir, 'tmptest3')
    with tf.io.TFRecordWriter(tmp) as w:
      for i in range(12):
        w.write(('%04d' % i).encode('utf-8'))

    p = base_input.LingvoInputAdaptorNewBatchSize.Params()
    p.input = LingvoInput.Params().Set(
        file_pattern='tfrecord:' + tmp, file_random_seed=0)
    with self.assertRaisesRegex(ValueError, 'p.batch_size'):
      p.Instantiate()

    p2 = base_input.LingvoInputAdaptor.Params().Set(input=p.input)
    p2.batch_size = 2
    with self.assertRaisesRegex(ValueError, 'p.batch_size'):
      p2.Instantiate()

    p3 = TestInput.Params()
    with self.assertRaisesRegex(ValueError, 'p.batch_size'):
      p3.Instantiate()

if __name__ == '__main__':
  absltest.main()
