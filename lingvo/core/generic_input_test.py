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
"""Tests for generic_input_op."""

import collections
import os
import pickle
import unittest
from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import generic_input
from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np


def get_test_input(path, bucket_batch_limit=8, **kwargs):
  return generic_input.GenericInput(
      file_pattern='tfrecord:' + path,
      file_random_seed=0,
      file_buffer_size=32,
      file_parallelism=4,
      bucket_batch_limit=[bucket_batch_limit],
      **kwargs)


def run_basic_graph(use_nested_map,
                    bucket_fn=lambda x: 1,
                    bucket_batch_limit=8):
  # Generate a test file w/ 100 records.
  tmp = os.path.join(tf.test.get_temp_dir(), 'basic')
  with tf.python_io.TFRecordWriter(tmp) as w:
    for i in range(100):
      w.write(('%08d' % i).encode('utf-8'))

  # A simple string parsing routine. Just convert a string to a
  # number.
  def str_to_num(s):
    return np.array(float(s), dtype=np.float32)

  # A record processor written in TF graph.
  def _process(source_id, record):
    num, = tf.py_func(str_to_num, [record], [tf.float32])
    num = tf.stack([num, tf.square(num)])
    if use_nested_map:
      return py_utils.NestedMap(
          source_id=source_id, record=record, num=num), bucket_fn(num)
    else:
      return [source_id, record, num], bucket_fn(num)

  # Samples random records from the data files and processes them
  # to generate batches.
  inputs, _ = get_test_input(
      tmp,
      bucket_batch_limit=bucket_batch_limit,
      bucket_upper_bound=[1],
      processor=_process)
  if use_nested_map:
    return inputs
  else:
    src_ids, strs, vals = inputs
    return py_utils.NestedMap(source_id=src_ids, record=strs, num=vals)


class GenericInputOpTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('OutputList', False, 8),
                                  ('OutputNestedMap', True, 8),
                                  ('OutputNestedMap_Batch1', True, 1))
  def testBasic(self, use_nested_map, bucket_batch_limit):
    input_batch = run_basic_graph(
        use_nested_map=use_nested_map, bucket_batch_limit=bucket_batch_limit)
    with self.session():
      record_seen = set()
      for i in range(100):
        ans_input_batch = self.evaluate(input_batch)
        for s in ans_input_batch.record:
          record_seen.add(s)
        self.assertEqual(ans_input_batch.source_id.shape, (bucket_batch_limit,))
        self.assertEqual(ans_input_batch.record.shape, (bucket_batch_limit,))
        self.assertEqual(ans_input_batch.num.shape, (bucket_batch_limit, 2))
        ans_vals = ans_input_batch.num
        self.assertAllEqual(np.square(ans_vals[:, 0]), ans_vals[:, 1])
      for i in range(100):
        self.assertIn(('%08d' % i).encode('utf-8'), record_seen)

  def testPadding(self):
    # Generate a test file w/ 50 records of different lengths.
    tmp = os.path.join(tf.test.get_temp_dir(), 'basic')
    with tf.python_io.TFRecordWriter(tmp) as w:
      for n in range(1, 50):
        w.write(pickle.dumps(np.full([n, 3, 3], n, np.int32)))

    g = tf.Graph()
    with g.as_default():
      # A record processor written in TF graph.
      def _process(record):
        num = tf.py_func(pickle.loads, [record], tf.int32)
        bucket_key = tf.shape(num)[0]
        return [num, tf.transpose(num, [1, 0, 2])], bucket_key

      # Samples random records from the data files and processes them
      # to generate batches.
      (vals_t, transposed_vals_t), _ = get_test_input(
          tmp,
          bucket_upper_bound=[10],
          processor=_process,
          dynamic_padding_dimensions=[0, 1],
          dynamic_padding_constants=[0] * 2)

    with self.session(graph=g):
      for _ in range(10):
        vals, transposed_vals = self.evaluate([vals_t, transposed_vals_t])
        print(vals, np.transpose(transposed_vals, [0, 2, 1, 3]))
        self.assertEqual(vals.shape[0], 8)
        self.assertEqual(vals.shape[2], 3)
        self.assertEqual(vals.shape[3], 3)
        largest = np.amax(vals)
        self.assertLessEqual(largest, 10)
        self.assertEqual(vals.shape[1], largest)
        for j in range(8):
          n = vals[j, 0, 0, 0]
          self.assertTrue(np.all(vals[j, :n] == n))
          self.assertTrue(np.all(vals[j, n:] == 0))
        self.assertAllEqual(vals, np.transpose(transposed_vals, [0, 2, 1, 3]))

  def testDropRecordIfNegativeBucketKey(self):

    def bucket_fn(num):
      # Drops record if num[0] is odd.
      return tf.cond(
          tf.equal(tf.math.floormod(num[0], 2), 0), lambda: 1,
          lambda: -tf.cast(num[0], tf.int32))

    input_batch = run_basic_graph(use_nested_map=False, bucket_fn=bucket_fn)

    with self.session():
      record_seen = set()
      for i in range(100):
        ans_input_batch = self.evaluate(input_batch)
        for s in ans_input_batch.record:
          record_seen.add(s)
      for i in range(100):
        if i % 2 == 0:
          self.assertIn(('%08d' % i).encode('utf-8'), record_seen)
        else:
          self.assertNotIn(('%08d' % i).encode('utf-8'), record_seen)

  def testWithinBatchMixing(self):
    # Generate couple files.
    def generate_test_data(tag, cnt):
      tmp = os.path.join(tf.test.get_temp_dir(), tag)
      with tf.python_io.TFRecordWriter(tmp) as w:
        for i in range(cnt):
          w.write(('%s:%08d' % (tag, i)).encode('utf-8'))
      return tmp

    path1 = generate_test_data('input1', 100)
    path2 = generate_test_data('input2', 200)
    path3 = generate_test_data('input3', 10)

    g = tf.Graph()
    with g.as_default():
      # A record processor written in TF graph.
      def _process(source_id, record):
        return py_utils.NestedMap(source_id=source_id, record=record), 1

      # Samples random records from the data files and processes them
      # to generate batches.
      input_batch, buckets = generic_input.GenericInput(
          file_pattern=','.join(
              ['tfrecord:' + path1, 'tfrecord:' + path2, 'tfrecord:' + path3]),
          input_source_weights=[0.2, 0.3, 0.5],
          file_random_seed=0,
          file_buffer_size=32,
          file_parallelism=4,
          bucket_batch_limit=[8],
          bucket_upper_bound=[1],
          processor=_process)

    with self.session(graph=g):
      source_id_count = collections.defaultdict(int)
      tags_count = collections.defaultdict(int)
      total_count = 10000
      for _ in range(total_count):
        ans_input_batch, ans_buckets = self.evaluate([input_batch, buckets])
        for s in ans_input_batch.source_id:
          source_id_count[s] += 1
        for s in ans_input_batch.record:
          tags_count[s.split(b':')[0]] += 1
        self.assertEqual(ans_input_batch.source_id.shape, (8,))
        self.assertEqual(ans_input_batch.record.shape, (8,))
        self.assertAllEqual(ans_buckets, [1] * 8)
      self.assertEqual(sum(source_id_count.values()), total_count * 8)
      self.assertEqual(sum(tags_count.values()), total_count * 8)
      num_records = 8. * total_count
      self.assertAlmostEqual(
          tags_count[b'input1'] / num_records, 0.2, delta=0.01)
      self.assertAlmostEqual(
          tags_count[b'input2'] / num_records, 0.3, delta=0.01)
      self.assertAlmostEqual(
          tags_count[b'input3'] / num_records, 0.5, delta=0.01)
      self.assertAlmostEqual(source_id_count[0] / num_records, 0.2, delta=0.01)
      self.assertAlmostEqual(source_id_count[1] / num_records, 0.3, delta=0.01)
      self.assertAlmostEqual(source_id_count[2] / num_records, 0.5, delta=0.01)

  def testBoolDType(self):
    tmp = os.path.join(tf.test.get_temp_dir(), 'bool')
    with tf.python_io.TFRecordWriter(tmp) as w:
      for i in range(50):
        w.write(pickle.dumps(True if i % 2 == 0 else False))

    g = tf.Graph()
    with g.as_default():
      # A record processor written in TF graph.
      def _process(record):
        bucket_key = 1
        num, = tf.py_func(pickle.loads, [record], [tf.bool])
        return [num], bucket_key

      # Samples random records from the data files and processes them
      # to generate batches.
      inputs, _ = get_test_input(
          tmp, bucket_upper_bound=[1], processor=_process)

    with self.session(graph=g):
      for _ in range(10):
        inputs_vals = self.evaluate(inputs)[0]
        self.assertEqual(inputs_vals.dtype, bool)

  def testExtraArgs(self):

    def _parse_record(record):
      del record
      example = py_utils.NestedMap(t=tf.convert_to_tensor(0))
      bucketing_key = 1
      return example, bucketing_key

    def _parse_record_stateful(record):
      del record
      extra = tf.Variable(0)
      example = py_utils.NestedMap(t=extra.value())
      bucketing_key = 1
      return example, bucketing_key

    generic_input.GenericInput(
        _parse_record,
        file_pattern='',
        bucket_upper_bound=[1],
        bucket_batch_limit=[1])

    with self.assertRaisesRegex(AssertionError, 'is not pure: extra_args='):
      generic_input.GenericInput(
          _parse_record_stateful,
          file_pattern='',
          bucket_upper_bound=[1],
          bucket_batch_limit=[1])

  def testTfData(self):
    """Checks that GenericInput can be invoked from a tf.data.Dataset."""

    def _input_batch():
      return run_basic_graph(use_nested_map=True)

    # Trick to create dataset from tensor coming from custom op.
    dummy_dataset = tf.data.Dataset.from_tensors(0).repeat()
    dataset = dummy_dataset.map(lambda _: _input_batch())

    with self.session(use_gpu=False) as sess:
      it = tf.compat.v1.data.make_initializable_iterator(dataset)
      sess.run(it.initializer)
      batch = it.get_next()
      for _ in range(10):  # Read 10 batches.
        print(sess.run(batch))

  @unittest.skip('This test is expected to crash.')
  def testFatalErrors(self):
    tmp = os.path.join(tf.test.get_temp_dir(), 'fatal')
    with tf.python_io.TFRecordWriter(tmp) as w:
      for i in range(50):
        w.write(str((i % 2) * 2**33))

    def _parse_record(record):
      # tf.strings.to_number raises error on overflow.
      i = tf.strings.to_number(record, tf.int32)
      example = py_utils.NestedMap(record=i)
      bucketing_key = 1
      return example, bucketing_key

    with self.session():
      # Without specifying fatal_errors all records not 0 are skipped.
      input_batch, _ = generic_input.GenericInput(
          _parse_record,
          file_pattern=f'tfrecord:{tmp}',
          bucket_upper_bound=[1],
          bucket_batch_limit=[1])

      for i in range(25):
        ans_input_batch = self.evaluate(input_batch)
        self.assertEqual(ans_input_batch.record[0], 0)

      # With fatal_errors it dies instead.
      input_batch, _ = generic_input.GenericInput(
          _parse_record,
          file_pattern=f'tfrecord:{tmp}',
          bucket_upper_bound=[1],
          bucket_batch_limit=[1],
          fatal_errors=['StringToNumberOp could not correctly convert string:'])

      # NOTE: There is no way to catch LOG(FATAL) from python side, so running
      # this test will cause a crash.
      for i in range(10):
        self.evaluate(input_batch)

  @parameterized.named_parameters(('BatchOnce', 4, 1), ('BatchTwice', 5, 2))
  def testNestedGenericInput(self, inner_batch_limit, outer_batch_limit):
    # Generate records using an inner GenericInput, and post-process them using
    # an outer one.
    # Test that the generated records are complete and contain no duplicates

    def _process(record):
      del record
      # Construct the inner GenericInput.
      batch = run_basic_graph(
          use_nested_map=True, bucket_batch_limit=inner_batch_limit)
      batch.num += 1
      return batch, 1

    input_batch, _ = generic_input.GenericInput(
        file_pattern='iota:',
        processor=_process,
        bucket_upper_bound=[1],
        bucket_batch_limit=[outer_batch_limit])

    with self.session():
      global_batch = inner_batch_limit * outer_batch_limit
      record_seen = set()
      # Iterate the inputs for exactly one epoch.
      for i in range(100 // global_batch):
        ans_input_batch = self.evaluate(input_batch)
        for record_array in ans_input_batch.record:
          for s in record_array:
            # There should not be duplicates since GenericInput is stateful.
            assert s not in record_seen
            record_seen.add(s)
        self.assertEqual(ans_input_batch.source_id.shape,
                         (outer_batch_limit, inner_batch_limit))
        self.assertEqual(ans_input_batch.record.shape,
                         (outer_batch_limit, inner_batch_limit))
        self.assertEqual(ans_input_batch.num.shape,
                         (outer_batch_limit, inner_batch_limit, 2))
        ans_vals = ans_input_batch.num
        self.assertAllEqual(
            np.square(ans_vals[:, :, 0] - 1), ans_vals[:, :, 1] - 1)
      for i in range(100):
        self.assertIn(('%08d' % i).encode('utf-8'), record_seen)


class GenericInputOpBenchmark(tf.test.Benchmark):

  def benchmark_basic(self):
    input_batch = run_basic_graph(use_nested_map=True)
    with tf.Session() as sess:
      print(self.run_op_benchmark(sess, input_batch, min_iters=10))


if __name__ == '__main__':
  tf.test.main()
