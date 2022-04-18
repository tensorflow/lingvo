# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for generic_input_op in TF2 (pure Eager and tf.function)."""

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
  return generic_input.GenericInputV2Create(
      file_pattern='tfrecord:' + path,
      file_random_seed=0,
      file_buffer_size=32,
      file_parallelism=4,
      bucket_batch_limit=[bucket_batch_limit],
      **kwargs)


def setup_basic(use_nested_map, bucket_fn=lambda x: 1, bucket_batch_limit=8):
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

  resource, out_types, output_tmpl = get_test_input(
      tmp,
      bucket_batch_limit=bucket_batch_limit,
      bucket_upper_bound=[1],
      processor=_process)

  return resource, out_types, output_tmpl


def run_basic(resource, use_nested_map, out_types, output_tmpl):

  inputs = generic_input.GenericInputV2GetNext(resource, out_types, output_tmpl)
  if use_nested_map:
    return inputs[0]
  else:
    (src_ids, strs, vals), _ = inputs
    return py_utils.NestedMap(source_id=src_ids, record=strs, num=vals)


class GenericInputV2OpTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('OutputListEager', False, 8, False),
      ('OutputNestedMapEager', True, 8, False),
      ('OutputNestedMap_Batch1Eager', True, 1, False),
      ('OutputListFunc', False, 8, True),
      ('OutputNestedMapFunc', True, 8, True),
      ('OutputNestedMap_Batch1Func', True, 1, True))
  def testBasic(self, use_nested_map, bucket_batch_limit, use_tf_func):

    resource, out_types, output_tmpl = setup_basic(
        use_nested_map=use_nested_map, bucket_batch_limit=bucket_batch_limit)

    def _get_batch():
      return run_basic(
          resource,
          use_nested_map=use_nested_map,
          out_types=out_types,
          output_tmpl=output_tmpl)

    if use_tf_func:
      _get_batch = tf.function(autograph=False)(_get_batch)  # pylint: disable=invalid-name

    record_seen = set()
    # Iterate for 1 epoch
    for _ in range(100 // bucket_batch_limit + 1):
      ans_input_batch = _get_batch()
      for s in ans_input_batch.record:
        record_seen.add(s.numpy())
      self.assertEqual(ans_input_batch.source_id.shape, (bucket_batch_limit,))
      self.assertEqual(ans_input_batch.record.shape, (bucket_batch_limit,))
      self.assertEqual(ans_input_batch.num.shape, (bucket_batch_limit, 2))
      ans_vals = ans_input_batch.num
      self.assertAllEqual(np.square(ans_vals[:, 0]), ans_vals[:, 1])
    for i in range(100):
      self.assertIn(('%08d' % i).encode('utf-8'), record_seen)

  @parameterized.named_parameters(('Eager', False),
                                  ('Func', True))
  def testPadding(self, use_tf_func):
    # Generate a test file w/ 50 records of different lengths.
    tmp = os.path.join(tf.test.get_temp_dir(), 'basic')
    with tf.python_io.TFRecordWriter(tmp) as w:
      for n in range(1, 50):
        w.write(pickle.dumps(np.full([n, 3, 3], n, np.int32)))

    # A record processor written in TF graph.
    def _process(record):
      num = tf.py_func(pickle.loads, [record], tf.int32)
      bucket_key = tf.shape(num)[0]
      return [num, tf.transpose(num, [1, 0, 2])], bucket_key

    # Samples random records from the data files and processes them
    # to generate batches.
    resource, out_types, output_tmpl = get_test_input(
        tmp,
        bucket_upper_bound=[10],
        processor=_process,
        dynamic_padding_dimensions=[0, 1],
        dynamic_padding_constants=[0] * 2)

    def _get_batch():
      return generic_input.GenericInputV2GetNext(resource, out_types,
                                                 output_tmpl)

    if use_tf_func:
      _get_batch = tf.function(autograph=False)(_get_batch)  # pylint: disable=invalid-name

    for _ in range(10):
      (vals, transposed_vals), _ = _get_batch()
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

  @parameterized.named_parameters(
      ('Eager', False),
      ('Func', True))
  def testDropRecordIfNegativeBucketKey(self, use_tf_func):

    def bucket_fn(num):
      # Drops record if num[0] is odd.
      return tf.cond(
          tf.equal(tf.math.floormod(num[0], 2), 0), lambda: 1,
          lambda: -tf.cast(num[0], tf.int32))

    resource, out_types, output_tmpl = setup_basic(
        use_nested_map=False, bucket_fn=bucket_fn)

    def _get_batch():
      return run_basic(
          resource,
          use_nested_map=False,
          out_types=out_types,
          output_tmpl=output_tmpl)

    if use_tf_func:
      _get_batch = tf.function(autograph=False)(_get_batch)  # pylint: disable=invalid-name

    record_seen = set()
    for i in range(100):
      ans_input_batch = _get_batch()
      for s in ans_input_batch.record:
        record_seen.add(s.numpy())
    for i in range(100):
      if i % 2 == 0:
        self.assertIn(('%08d' % i).encode('utf-8'), record_seen)
      else:
        self.assertNotIn(('%08d' % i).encode('utf-8'), record_seen)

  @parameterized.named_parameters(('Eager', False),
                                  ('Func', True))
  def testWithinBatchMixing(self, use_tf_func):
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

    # A record processor written in TF graph.
    def _process(source_id, record):
      return py_utils.NestedMap(source_id=source_id, record=record), 1

    # Samples random records from the data files and processes them
    # to generate batches.
    resource, out_types, output_tmpl = generic_input.GenericInputV2Create(
        file_pattern=','.join(
            ['tfrecord:' + path1, 'tfrecord:' + path2, 'tfrecord:' + path3]),
        input_source_weights=[0.2, 0.3, 0.5],
        file_random_seed=0,
        file_buffer_size=32,
        file_parallelism=4,
        bucket_batch_limit=[8],
        bucket_upper_bound=[1],
        processor=_process)

    def _get_batch():
      return generic_input.GenericInputV2GetNext(resource, out_types,
                                                 output_tmpl)

    if use_tf_func:
      _get_batch = tf.function(autograph=False)(_get_batch)  # pylint: disable=invalid-name

    source_id_count = collections.defaultdict(int)
    tags_count = collections.defaultdict(int)
    total_count = 10000
    for _ in range(total_count):
      ans_input_batch, ans_buckets = _get_batch()
      for s in ans_input_batch.source_id:
        # We use `numpy()` to get Tensor's value
        source_id_count[s.numpy()] += 1
      for s in ans_input_batch.record:
        # We use `numpy()` to get Tensor's value
        tags_count[s.numpy().split(b':')[0]] += 1
      self.assertEqual(ans_input_batch.source_id.shape, (8,))
      self.assertEqual(ans_input_batch.record.shape, (8,))
      self.assertAllEqual(ans_buckets, [1] * 8)
    self.assertEqual(sum(source_id_count.values()), total_count * 8)
    self.assertEqual(sum(tags_count.values()), total_count * 8)
    num_records = 8. * total_count
    self.assertAlmostEqual(tags_count[b'input1'] / num_records, 0.2, delta=0.01)
    self.assertAlmostEqual(tags_count[b'input2'] / num_records, 0.3, delta=0.01)
    self.assertAlmostEqual(tags_count[b'input3'] / num_records, 0.5, delta=0.01)
    self.assertAlmostEqual(source_id_count[0] / num_records, 0.2, delta=0.01)
    self.assertAlmostEqual(source_id_count[1] / num_records, 0.3, delta=0.01)
    self.assertAlmostEqual(source_id_count[2] / num_records, 0.5, delta=0.01)

  @parameterized.named_parameters(('Eager', False),
                                  ('Func', True))
  def testBoolDType(self, use_tf_func):
    tmp = os.path.join(tf.test.get_temp_dir(), 'bool')
    with tf.python_io.TFRecordWriter(tmp) as w:
      for i in range(50):
        w.write(pickle.dumps(True if i % 2 == 0 else False))

    # A record processor written in TF graph.
    def _process(record):
      bucket_key = 1
      num, = tf.py_func(pickle.loads, [record], [tf.bool])
      return [num], bucket_key

    # Samples random records from the data files and processes them
    # to generate batches.
    resource, out_types, output_tmpl = get_test_input(
        tmp, bucket_upper_bound=[1], processor=_process)

    def _get_batch():
      return generic_input.GenericInputV2GetNext(resource, out_types,
                                                 output_tmpl)

    if use_tf_func:
      _get_batch = tf.function(autograph=False)(_get_batch)  # pylint: disable=invalid-name

    for _ in range(10):
      inputs_vals, _ = _get_batch()
      self.assertEqual(inputs_vals[0].dtype, bool)

  def testExtraArgs(self):

    def _parse_record_stateful(record):
      del record
      extra = tf.Variable(0)
      example = py_utils.NestedMap(t=extra.value())
      bucketing_key = 1
      return example, bucketing_key

    with self.assertRaisesRegex(AssertionError, 'is not pure: extra_args='):
      generic_input.GenericInputV2Create(
          _parse_record_stateful,
          file_pattern='',
          bucket_upper_bound=[1],
          bucket_batch_limit=[1])

  @parameterized.named_parameters(('Eager', False),
                                  ('Func', True))
  def testTfData(self, use_tf_func):
    """Checks that GenericInput can be invoked from a tf.data.Dataset."""

    resource, out_types, output_tmpl = setup_basic(use_nested_map=True)

    def _get_batch():
      return run_basic(
          resource,
          use_nested_map=False,
          out_types=out_types,
          output_tmpl=output_tmpl)

    if use_tf_func:
      _get_batch = tf.function(autograph=False)(_get_batch)  # pylint: disable=invalid-name

    # Trick to create dataset from tensor coming from custom op.
    dummy_dataset = tf.data.Dataset.from_tensors(0).repeat()
    dataset = dummy_dataset.map(lambda _: _get_batch())

    it = iter(dataset)
    for _ in range(10):  # Read 10 batches.
      print(it.get_next())

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

    # Without specifying fatal_errors all records not 0 are skipped.
    resource, out_types, output_tmpl = generic_input.GenericInputV2Create(
        _parse_record,
        file_pattern=f'tfrecord:{tmp}',
        bucket_upper_bound=[1],
        bucket_batch_limit=[1])

    for i in range(25):
      ans_input_batch, _ = generic_input.GenericInputV2GetNext(
          resource, out_types, output_tmpl)
      self.assertEqual(ans_input_batch.record[0], 0)

    # With fatal_errors it dies instead.
    resource, out_types, output_tmpl = generic_input.GenericInputV2Create(
        _parse_record,
        file_pattern=f'tfrecord:{tmp}',
        bucket_upper_bound=[1],
        bucket_batch_limit=[1],
        fatal_errors=['StringToNumberOp could not correctly convert string:'])

    # NOTE: There is no way to catch LOG(FATAL) from python side, so running
    # this test will cause a crash.
    for i in range(10):
      ans_input_batch, _ = generic_input.GenericInputV2GetNext(
          resource, out_types, output_tmpl)

  # TODO(b/223283717): Unfortunately, currently we do not support nested
  # GenericInputV2 op calls. Unlike GenericInput, the new implementation passes
  # a resource handle to the users, and each call to `GenericInputV2GetNext`
  # requires this resource handle explicitly. As a result, in the outer
  # `GenericInputV2Create` call, the process function will capture the resource.
  # Right now, this is forbidden in python. Otherwise, in C++ the processor will
  # keep failing because of a mismatch between the expected and received arg
  # numbers. We leave this as a TODO item as the models that use nested
  # GenericInput calls are very few.
  @unittest.skip('This test is expected to crash.')
  @parameterized.named_parameters(('BatchOnce', 4, 1), ('BatchTwice', 5, 2))
  def testNestedGenericInput(self, inner_batch_limit, outer_batch_limit):
    # Generate records using an inner GenericInput, and post-process them using
    # an outer one.
    # Test that the generated records are complete and contain no duplicates

    resource_inner, out_types_inner, output_tmpl_inner = setup_basic(
        use_nested_map=True, bucket_batch_limit=inner_batch_limit)

    def _process(record):
      del record
      # Construct the inner GenericInput.
      batch = run_basic(
          resource_inner,
          use_nested_map=True,
          out_types=out_types_inner,
          output_tmpl=output_tmpl_inner)
      batch.num += 1
      return batch, 1

    # The `AssertionError` error will be raised from `GenericInputV2Create()`
    (resource_outer, out_types_outer,
     output_tmpl_outer) = generic_input.GenericInputV2Create(
         file_pattern='iota:',
         processor=_process,
         bucket_upper_bound=[1],
         bucket_batch_limit=[outer_batch_limit])

    global_batch = inner_batch_limit * outer_batch_limit
    record_seen = set()
    # Iterate the inputs for exactly one epoch.
    for i in range(100 // global_batch):
      ans_input_batch, _ = generic_input.GenericInputV2GetNext(
          resource_outer, out_types_outer, output_tmpl_outer)
      for record_array in ans_input_batch.record:
        for s in record_array:
          # There should not be duplicates since GenericInput is stateful.
          assert s.numpy() not in record_seen
          record_seen.add(s.numpy())
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

  @parameterized.named_parameters(
      ('EagerMissingFlag', False, False), ('FuncMissingFlag', True, False),
      ('EagerMissingKey', False, True), ('FuncMissingKey', True, True))
  def testV2OpsErrorRaised(self, use_tf_func, set_allow_eager):
    # Generate a test file w/ 100 records.
    tmp = os.path.join(tf.test.get_temp_dir(), 'basic')
    with tf.python_io.TFRecordWriter(tmp) as w:
      for i in range(100):
        w.write(('%08d' % i).encode('utf-8'))

    # A simple string parsing routine. Just convert a string to a
    # number.
    def str_to_num(s):
      return np.array(float(s), dtype=np.float32)

    bucket_fn = lambda x: 1

    # A record processor written in TF graph.
    def _process(source_id, record):
      num, = tf.py_func(str_to_num, [record], [tf.float32])
      num = tf.stack([num, tf.square(num)])
      return py_utils.NestedMap(
          source_id=source_id, record=record, num=num), bucket_fn(num)

    if set_allow_eager:
      # Test unique keys must be provided to distinguish GenericInputV2 ops
      generic_input.SetAllowGenericInputV2InEager(True)
      err_regex = 'op requires a unique key'
    else:
      # Test flags must be set to enable GenericInputV2 ops in Eager mode
      generic_input.SetAllowGenericInputV2InEager(False)
      err_regex = 'please add keyword arg'

    with self.assertRaisesRegex(RuntimeError, err_regex):
      _ = generic_input.GenericInput(
          file_pattern='tfrecord:' + tmp,
          file_random_seed=0,
          file_buffer_size=32,
          file_parallelism=4,
          bucket_batch_limit=[8],
          bucket_upper_bound=[1],
          processor=_process)

  @parameterized.named_parameters(('Eager', False, 'foo'),
                                  ('Func', True, 'bar'))
  def testV2OpsGetCalledInEager(self, use_tf_func, mock_op_key):
    # Generate a test file w/ 100 records.
    tmp = os.path.join(tf.test.get_temp_dir(), 'basic')
    with tf.python_io.TFRecordWriter(tmp) as w:
      for i in range(100):
        w.write(('%08d' % i).encode('utf-8'))

    # A simple string parsing routine. Just convert a string to a
    # number.
    def str_to_num(s):
      return np.array(float(s), dtype=np.float32)

    bucket_fn = lambda x: 1

    # A record processor written in TF graph.
    def _process(source_id, record):
      num, = tf.py_func(str_to_num, [record], [tf.float32])
      num = tf.stack([num, tf.square(num)])
      return py_utils.NestedMap(
          source_id=source_id, record=record, num=num), bucket_fn(num)

    # pylint: disable=protected-access
    len_before = len(generic_input._GENERIC_CACHE_V2)
    _ = generic_input.GenericInput(
        file_pattern='tfrecord:' + tmp,
        file_random_seed=0,
        file_buffer_size=32,
        file_parallelism=4,
        bucket_batch_limit=[8],
        bucket_upper_bound=[1],
        processor=_process,
        generic_input_v2_key=mock_op_key)

    # pylint: disable=protected-access
    len_after = len(generic_input._GENERIC_CACHE_V2)
    self.assertEqual(len_after, len_before + 1)

if __name__ == '__main__':
  py_utils.SetEagerMode()
  tf.test.main()
