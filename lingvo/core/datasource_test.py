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
"""Tests for lingvo.core.datasource."""

import glob
import os
import random

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import cluster_factory
from lingvo.core import datasource
from lingvo.core import generic_input
from lingvo.core import py_utils
from lingvo.core import test_utils

import mock
import tensorflow_datasets as tfds


class TestInputGenerator(base_input_generator.TFDataSequenceInputGenerator):

  def _DataSourceFromFilePattern(self, file_pattern, input_source_weights=None):
    """Read and return input batch from a string file_pattern.

    Args:
      file_pattern: A string file pattern.
      input_source_weights: A list of float input source weights to control
        input example mix in the batch. The records will be sampled from inputs
        proportionally to these weights. Defaults to None which should be
        treated as an empty list.

    Returns:
      file_pattern.  This is the file_pattern that will be sent to InputOp along
      with input_source_weights and other parameters to generate a batch.
    """
    del input_source_weights  # Unused.
    return py_utils.NestedMap(data=tf.constant(file_pattern))

  def LoadDataset(self, file_pattern):
    file_pattern_glob = py_utils.ShardedFilePatternToGlob(file_pattern)
    filenames = sorted(tf.io.gfile.glob(file_pattern_glob))
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    def MakeExample(data):
      return py_utils.NestedMap(data=data, source_id=tf.constant(0))

    return dataset.map(MakeExample, deterministic=True)

  def _InputShape(self, name):
    if name in ('data', 'len'):
      return ()
    return super()._InputShape(name)

  def LoadFilePattern(self, dataset):

    def Load(file_pattern):
      patterns = file_pattern.split(b',')
      files = []
      for pattern in patterns:
        files.extend(glob.glob(pattern))
      return random.choice(files)

    def MapFn(example):
      example.data = tf.numpy_function(Load, inp=[example.data], Tout=tf.string)
      return example

    return dataset.map(MapFn, deterministic=True)

  def GetSequenceLength(self, example):
    return tf.strings.length(example.data)

  def SequenceLengthTransform(self, dataset):

    def AddSequenceLength(example):
      example.len = self.GetSequenceLength(example)
      return example

    return dataset.map(AddSequenceLength, deterministic=True)


class TestFileInputGenerator(base_input_generator.BaseInputGeneratorFromFiles):

  def _DataSourceFromFilePattern(self, file_pattern, input_source_weights=None):
    """Read and return input batch from a string file_pattern."""
    del input_source_weights  # Unused.

    def Process(source_id, record):
      del source_id  # Unused.
      [num] = tf.py_func(int, [record], [tf.int64])
      return py_utils.NestedMap(data=num), 1

    # Samples random records from the data files and processes them
    # to generate batches.
    inputs, _ = generic_input.GenericInput(
        processor=Process,
        file_pattern=file_pattern,
        file_random_seed=123,
        file_buffer_size=1,
        file_parallelism=1,
        bucket_batch_limit=[1],
        bucket_upper_bound=[1])
    return inputs


class ConstantTFDatasetSource(datasource.TFDatasetSource):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('output', py_utils.NestedMap(), 'The output.')
    return p

  def GetDataset(self):
    return tf.data.Dataset.from_tensors(self.params.output).repeat()


class DatasourceTest(test_utils.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Generate a test file w/ 1 number.
    cls.inputs = []
    for i in range(2):
      tmp = os.path.join(tf.test.get_temp_dir(), f'inputs.{i}')
      with tf.python_io.TFRecordWriter(tmp) as w:
        w.write(('%08d' % i).encode('utf-8'))
      cls.inputs.append(tmp)

  def testSimpleDataSourceSucceedsWithStringInput(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern='path_to_file')
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertEqual(ret.data, b'path_to_file')

  def testSimpleDataSourceSucceedsWithFileType(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern='pattern1,pattern2', file_type='tfrecord')
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertEqual(ret.data, b'tfrecord:pattern1,pattern2')

  def testSimpleDataSourceSucceedsWithListInput(self):
    files = ['file1', 'file2']
    ds_params = datasource.SimpleDataSource.Params().Set(file_pattern=files)
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with tf.Session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertEqual(ret.data, b'file1,file2')

  def testSimpleDataSourceSucceedsWithListFilesAndWeights(self):
    files = ['path_to_file1', 'path_to_file2']
    weights = [1, 4]
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern=files, weights=weights)
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertEqual(ret.data, b'path_to_file1,path_to_file2')
    self.assertCountEqual(ret.bprop_variable_filters, [''] * len(files))

  def testSimpleDataSourceFailsWithListTuplesFiles(self):
    # This legacy p.file_pattern behavior is only supported through
    # base_input_generator.
    files = [('file1', 1.0), ('file2', 2.0)]
    ds_params = datasource.SimpleDataSource.Params().Set(file_pattern=files)

    with self.assertRaises(ValueError):
      ds_params.Instantiate()

  def testSimpleDataSourceFileInputSucceedsWithStringInput(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern='tfrecord:' + ','.join(self.inputs))
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestFileInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      data = []
      for _ in range(2):
        data.append(self.evaluate(batch.data).tolist()[0])
      self.assertCountEqual(data, [0, 1])

  def testSimpleDataSourceFileInputSucceedsWithFileType(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern=','.join(self.inputs), file_type='tfrecord')
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestFileInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      data = []
      for _ in range(2):
        data.append(self.evaluate(batch.data).tolist()[0])
      self.assertCountEqual(data, [0, 1])

  def testSimpleDataSourceFileInputSucceedsWithListInput(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern=self.inputs, file_type='tfrecord')
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestFileInputGenerator.Params().Instantiate())
    with tf.Session():
      batch = ds.GetNext()
      data = []
      for _ in range(2):
        data.append(self.evaluate(batch.data).tolist()[0])
      self.assertCountEqual(data, [0, 1])

  def testSimpleDataSourceFileInputSucceedsWithListFilesAndWeights(self):
    weights = [1, 4]
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern=self.inputs, file_type='tfrecord', weights=weights)
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestFileInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      data = []
      for _ in range(2):
        data.append(self.evaluate(batch.data).tolist()[0])
      self.assertCountEqual(data, [0, 1])

  def testCrossBatchMixingDataSourceSucceedsWithListFilesAndWeights(self):
    files = ['path_to_file', 'path_to_file']
    datasources = [
        datasource.SimpleDataSource.Params().Set(file_pattern=f) for f in files
    ]
    weights = [1, 4]
    ds_params = datasource.CrossBatchMixingDataSource.Params().Set(
        sub=datasources, weights=weights)
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertCountEqual(
        sorted(ret.keys()),
        ['bprop_variable_filters', 'data', 'selected_bprop', 'source_selected'])
    # CrossBatchMixing operates on the python side of the tf op so a single
    # element will be returned by TestInputGenerator.Params().Instantiate()
    self.assertEqual(ret.data, b'path_to_file')
    self.assertCountEqual(ret.bprop_variable_filters, [''] * len(files))
    self.assertAllEqual(ret.selected_bprop.shape, [2])
    self.assertAllEqual(ret.source_selected.shape, [1, 2])

  def testCurriculumDataSourceSucceedsWithSimpleDataSource(self):
    sources = [
        datasource.SimpleDataSource.Params().Set(file_pattern='file1'),
        datasource.SimpleDataSource.Params().Set(file_pattern='file2'),
    ]
    ds_params = datasource.CurriculumDataSource.Params().Set(
        sub=sources, boundaries=[5])
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertEqual(ret.data, b'file1')
    self.assertCountEqual(ret.bprop_variable_filters, [''])

  def testCurriculumDataSourceTransitionsCorrectlyWithSimpleDataSource(self):
    sources = [
        datasource.SimpleDataSource.Params().Set(file_pattern='file1'),
        datasource.SimpleDataSource.Params().Set(file_pattern='file2'),
    ]
    boundary = 5
    ds_params = datasource.CurriculumDataSource.Params().Set(
        sub=sources, boundaries=[boundary])
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      # Advance the global step to the next curriculum stage
      global_step = py_utils.GetOrCreateGlobalStepVar()
      self.evaluate(tf.global_variables_initializer())
      set_global_step = tf.assign(global_step, boundary, name='advance_step')
      self.evaluate(set_global_step)

      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertEqual(ret.data, b'file2')
    self.assertCountEqual(ret.bprop_variable_filters, [''])

  def testCurriculumDataSourceTransitionsCorrectlyWithMixingDataSource(self):
    sources = [
        datasource.SimpleDataSource.Params().Set(
            file_pattern=['file1', 'file2'], weights=[1, 5]),
        datasource.SimpleDataSource.Params().Set(
            file_pattern=['file3', 'file4'], weights=[2, 3])
    ]
    boundary = 5
    ds_params = datasource.CurriculumDataSource.Params().Set(
        sub=sources, boundaries=[boundary])
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      # Advance the global step to the next curriculum stage
      global_step = py_utils.GetOrCreateGlobalStepVar()
      self.evaluate(tf.global_variables_initializer())
      set_global_step = tf.assign(global_step, boundary, name='advance_step')
      self.evaluate(set_global_step)

      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertEqual(ret.data, b'file3,file4')
    self.assertCountEqual(ret.bprop_variable_filters, [''])

  def testCurriculumDataSourceFailsWithBadBoundaries(self):
    sources = [
        datasource.SimpleDataSource.Params().Set(
            file_pattern=['file1', 'file2'], weights=[1, 5]),
        datasource.SimpleDataSource.Params().Set(
            file_pattern=['file3', 'file4'], weights=[2, 3])
    ]
    ds_params = datasource.CurriculumDataSource.Params().Set(
        sub=sources, boundaries=[10, 5])
    with self.assertRaises(ValueError):
      ds_params.Instantiate()

  def testPrefixDataSourceSucceedsWithDirectory(self):
    ds_params = datasource.PrefixedDataSource.Params().Set(
        file_pattern='filename-*.tfrecord', file_pattern_prefix='/dir/')
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertEqual(ret.data, b'/dir/filename-*.tfrecord')

  def testPrefixDataSourceSucceedsWithMultiplePatterns(self):
    ds_params = datasource.PrefixedDataSource.Params().Set(
        file_pattern='filename-*.tfrecord,other/file/pattern/*',
        file_pattern_prefix='/dir/')
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertEqual(ret.data,
                     b'/dir/filename-*.tfrecord,/dir/other/file/pattern/*')

  def testPrefixDataSourceSucceedsWithGcsBucket(self):
    ds_params = datasource.PrefixedDataSource.Params().Set(
        file_pattern='filename-*.tfrecord',
        file_pattern_prefix='gs://bucket/dir')
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertEqual(ret.data, b'gs://bucket/dir/filename-*.tfrecord')

  def testPrefixDataSourceSucceedsWithFileType(self):
    ds_params = datasource.PrefixedDataSource.Params().Set(
        file_pattern='filename-*.tfrecord',
        file_type='tfrecord',
        file_pattern_prefix='dir')
    ds = ds_params.Instantiate()

    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      ret = ds.GetMeta()
      ret.data = self.evaluate(batch.data)

    self.assertEqual(ret.data, b'tfrecord:dir/filename-*.tfrecord')


class TFDatasetSourceTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    files = sorted([
        'file_1', 'file_2', 'file_3', 'longfile_1', 'longfile_2', 'longerfile_1'
    ])
    self.tmpdir = self.create_test_files(self.id(), files)
    self.files = [os.path.join(self.tmpdir, file).encode() for file in files]

  def create_test_files(self, test_name, files):
    tmpdir = os.path.join(self.get_temp_dir(), test_name)
    os.mkdir(tmpdir)
    for file in files:
      os.mknod(os.path.join(tmpdir, file))
    return tmpdir

  def testTFDatasetFnInput(self):
    ds_params = datasource.TFDatasetFnInput.Params().Set(
        load_fn='LoadDataset',
        kwargs=dict(file_pattern=os.path.join(self.tmpdir, '*file_*')),
        shuffle_buffer_size=100)
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    files = []
    with self.session(), cluster_factory.SetRequireSequentialInputOrder(False):
      batch = ds.GetNext()
      for _ in range(len(self.files) * 5):
        file, source_id = self.evaluate([batch.data, batch.source_id])
        self.assertEqual(0, source_id)
        self.assertIn(file, self.files)
        files.append(file)
    self.assertEqual(set(files), set(self.files))
    # Should not be produced in deterministic order.
    self.assertNotAllEqual(self.files * 5, files)

  def testTFDatasetFnInput_ShuffleBufferMustBeSet(self):

    def CreateDatasource(**kwargs):
      ds_params = datasource.TFDatasetFnInput.Params().Set(
          load_fn='LoadDataset',
          kwargs=dict(file_pattern=os.path.join(self.tmpdir, '*file_*')),
          **kwargs)
      ds = ds_params.Instantiate()
      ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
      ds.GetNext()

    with cluster_factory.SetRequireSequentialInputOrder(False):
      with self.assertRaisesRegex(ValueError,
                                  'shuffle_buffer_size must be set.'):
        CreateDatasource()

    # Setting shuffle_buffer_size works.
    with cluster_factory.SetRequireSequentialInputOrder(False):
      CreateDatasource(shuffle_buffer_size=1)

    # Setting require_sequential_input_order works.
    with cluster_factory.SetRequireSequentialInputOrder(True):
      CreateDatasource()

    # Sanity check that params are not persisting between calls.
    with cluster_factory.SetRequireSequentialInputOrder(False):
      with self.assertRaisesRegex(ValueError,
                                  'shuffle_buffer_size must be set.'):
        CreateDatasource()

  def testTFDatasetFnInput_RequireSequentialOrder(self):
    ds_params = datasource.TFDatasetFnInput.Params().Set(
        load_fn='LoadDataset',
        kwargs=dict(file_pattern=os.path.join(self.tmpdir, '*file_*')))
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session() as sess:
      batch = ds.GetNext()
      for expected_file in self.files:
        file, source_id = self.evaluate([batch.data, batch.source_id])
        self.assertEqual(0, source_id)
        self.assertEqual(expected_file, file)
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(batch)
      ds.Reset(sess)
      for expected_file in self.files:
        file, source_id = self.evaluate([batch.data, batch.source_id])
        self.assertEqual(0, source_id)
        self.assertEqual(expected_file, file)
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(batch)

  def testCustomTFDatasetTransform(self):
    ds_params = datasource.TFDatasetFnInput.Params().Set(
        load_fn='LoadDataset',
        kwargs=dict(file_pattern=os.path.join(self.tmpdir, '*file_*')))
    ds_params = datasource.CustomTFDatasetTransform.Params().Set(
        sub=ds_params, fn='SequenceLengthTransform')
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session():
      batch = ds.GetNext()
      for expected_file in self.files:
        file, length = self.evaluate([batch.data, batch.len])
        self.assertEqual(expected_file, file)
        self.assertEqual(len(expected_file), length)

  def testRepeatableCustomTFDatasetTransform(self):
    ds_params = datasource.RepeatableCustomTFDatasetTransform.Params().Set(
        sub=datasource.TFDatasetSource.Params())
    ds = ds_params.Instantiate()
    ds._input_generator = mock.Mock()
    # Test repeat_steps (3). Note that this ignores repeat_with_sentinel.
    ds._input_generator.params = py_utils.NestedMap({
        'use_per_host_infeed': False,
        'repeat_steps': 3,
        'repeat_with_sentinel': True
    })
    dataset = tf.data.Dataset.range(5)
    with self.session(), mock.patch.object(
        ds, 'GetDataset', return_value=dataset, autospec=True):
      elem = ds.GetNext()
      for idx in range(6):
        value = self.evaluate(elem)
        self.assertEqual(value, idx % 3)

    # Test repeat_with_sentinel using sentinel_key and sentinel_value.
    ds_params = datasource.RepeatableCustomTFDatasetTransform.Params().Set(
        sub=datasource.TFDatasetSource.Params(),
        sentinel_key='positive_key',
        sentinel_value=-1)
    ds = ds_params.Instantiate()
    ds._input_generator = mock.Mock()
    ds._input_generator.params = py_utils.NestedMap({
        'use_per_host_infeed': False,
        'repeat_steps': None,  # default.
        'repeat_with_sentinel': True
    })
    dataset = tf.data.Dataset.range(3)

    def _ConvertToNestedMap(x):
      return py_utils.NestedMap({'data': x, 'positive_key': 1})

    dataset = dataset.map(_ConvertToNestedMap)
    with self.session(), mock.patch.object(
        ds, 'GetDataset', return_value=dataset, autospec=True):
      batch = ds.GetNext()
      for idx in range(3):
        nested_map = self.evaluate(batch)
        self.assertEqual(nested_map.data, idx)
        self.assertEqual(nested_map.positive_key, 1)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(batch)

  def testTFDatasetAdaptor(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern=os.path.join(self.tmpdir, '*file_*'))
    ds_params = datasource.CustomTFDatasetTransform.Params().Set(
        sub=ds_params, fn='LoadFilePattern')
    ds_params = datasource.CustomTFDatasetTransform.Params().Set(
        sub=ds_params, fn='SequenceLengthTransform')
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    seen = set()
    with self.session():
      batch = ds.GetNext()
      for _ in range(30):
        file, length = self.evaluate([batch.data, batch.len])
        self.assertEqual(len(file), length)
        seen.add(file)
    self.assertEqual(seen, set(self.files))

  def testTFDatasetBatchBySequenceLength(self):
    ds_params = datasource.TFDatasetFnInput.Params().Set(
        load_fn='LoadDataset',
        kwargs=dict(file_pattern=os.path.join(self.tmpdir, '*file_*')),
        shuffle_buffer_size=100)
    ds_params = datasource.TFDatasetBatchBySequenceLength.Params().Set(
        sub=ds_params,
        seqlen_fn='GetSequenceLength',
        input_shape_fn='_InputShape',
        input_padding_fn='_InputPaddingValue',
        bucket_upper_bound=[
            len(os.path.join(self.tmpdir, 'file_1')),
            len(os.path.join(self.tmpdir, 'longfile_1'))
        ],
        bucket_batch_limit=[8, 8])
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
    with self.session(), cluster_factory.SetRequireSequentialInputOrder(False):
      batch = ds.GetNext()
      seen = set()
      for _ in range(20):
        files = self.evaluate(batch.data)
        self.assertEqual(len(files), 8)
        seen.update(files)
        basenames = [os.path.basename(file) for file in files]
        # Batch contains different files of the same length.
        self.assertGreater(len(set(basenames)), 1)
        # But everything in the batch is the same length.
        self.assertLen(set([len(basename) for basename in basenames]), 1)
      # Longer than bucket_upper_bound[-1] is filtered out.
      longerfile = os.path.join(self.tmpdir, 'longerfile_1').encode()
      self.assertEqual(set(seen), set(self.files) - set([longerfile]))

  def testTFDatasetMixer(self):
    ds1 = datasource.SimpleDataSource.Params().Set(
        file_pattern=os.path.join(self.tmpdir, '*file_*'))
    ds1 = datasource.CustomTFDatasetTransform.Params().Set(
        sub=ds1, fn='LoadFilePattern')

    ds2 = datasource.TFDatasetFnInput.Params().Set(
        load_fn='LoadDataset',
        kwargs=dict(file_pattern=os.path.join(self.tmpdir, '*file_*')))

    with self.subTest(name='DS1Only'), self.session(graph=tf.Graph()):
      ds_params = datasource.TFDatasetMixer.Params().Set(
          sub=[ds1, ds2], weights=[1.0, 0.0])
      ds = ds_params.Instantiate()
      ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
      batch = ds.GetNext()
      seen = set()
      for _ in range(30):
        file, source_id = self.evaluate([batch.data, batch.source_id])
        self.assertEqual(0, source_id)
        seen.add(file)
      self.assertEqual(seen, set(self.files))

    with self.subTest(name='DS2Only'), self.session(graph=tf.Graph()) as sess:
      ds_params = datasource.TFDatasetMixer.Params().Set(
          sub=[ds1, ds2], weights=[0.0, 1.0])
      ds = ds_params.Instantiate()
      ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
      batch = ds.GetNext()
      seen = set()
      for _ in self.files:
        file, source_id = self.evaluate([batch.data, batch.source_id])
        self.assertEqual(1, source_id)
        seen.add(file)
      self.assertEqual(seen, set(self.files))
      ds.Reset(sess)
      seen = set()
      for _ in self.files:
        file, source_id = self.evaluate([batch.data, batch.source_id])
        self.assertEqual(1, source_id)
        seen.add(file)
      self.assertEqual(seen, set(self.files))
      # Cannot run another self.evaluate(batch) here, as ds2 is exhausted but
      # ds1 has 0 prob so the op just hangs.

    with self.subTest(name='Mixed'), self.session(graph=tf.Graph()) as sess:
      ds_params = datasource.TFDatasetMixer.Params().Set(
          sub=[ds1, ds2], weights=[0.5, 0.5])
      ds = ds_params.Instantiate()
      ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
      batch = ds.GetNext()
      seen = [set(), set()]
      while seen[0] != set(self.files) or seen[1] != set(self.files):
        file, source_id = self.evaluate([batch.data, batch.source_id])
        seen[source_id].add(file)
      # Now ds2 is exhausted.
      for _ in range(100):
        source_id = self.evaluate(batch.source_id)
        self.assertEqual(0, source_id)
      ds.Reset(sess)
      while True:
        file, source_id = self.evaluate([batch.data, batch.source_id])
        if source_id == 1:
          # ds2 should have been reset.
          break

    with self.subTest(name='MixedDS2'), self.session(graph=tf.Graph()):
      ds_params = datasource.TFDatasetMixer.Params().Set(
          sub=[ds2, ds2], weights=[0.5, 0.5])
      ds = ds_params.Instantiate()
      ds.SetInputGenerator(TestInputGenerator.Params().Instantiate())
      batch = ds.GetNext()
      files = []
      for _ in range(len(self.files) * 2):
        files.append(self.evaluate(batch.data))
      self.assertAllEqual(sorted(files), sorted(self.files * 2))
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(batch)

  def testTFDatasetMixerBroadcast(self):
    a = tf.zeros([1], dtype=tf.float32)
    b = tf.zeros([2], dtype=tf.int32)
    c = tf.zeros([3], dtype=tf.string)
    ds1 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(a=a, b=b))
    ds2 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(a=a, c=c))
    ds3 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(b=b, c=c))

    ds_params = datasource.TFDatasetMixer.Params().Set(
        sub=[ds1, ds2, ds3], broadcast_dataset_structures=True)

    with self.session():
      ds = ds_params.Instantiate()
      batch = self.evaluate(ds.GetNext())

    for key in ['a', 'b', 'c']:
      self.assertIn(key, batch.keys())

  def testTFDatasetMixerBroadcast_ShapeMismatch(self):
    a = tf.zeros([1], dtype=tf.float32)
    b = tf.zeros([2], dtype=tf.int32)
    c = tf.zeros([3], dtype=tf.string)
    ds1 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(a=a, b=b))
    ds2 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(a=tf.zeros([1, 1], dtype=tf.float32), c=c))
    ds3 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(b=b, c=c))

    ds_params = datasource.TFDatasetMixer.Params().Set(
        sub=[ds1, ds2, ds3], broadcast_dataset_structures=True)

    with self.session():
      ds = ds_params.Instantiate()
      with self.assertRaisesRegex(ValueError, 'Incompatible dataset specs'):
        self.evaluate(ds.GetNext())

  def testTFDatasetMixerBroadcast_DTypeMismatch(self):
    a = tf.zeros([1], dtype=tf.float32)
    b = tf.zeros([2], dtype=tf.int32)
    c = tf.zeros([3], dtype=tf.string)
    ds1 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(a=a, b=b))
    ds2 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(a=tf.zeros([1], dtype=tf.float64), c=c))
    ds3 = ConstantTFDatasetSource.Params().Set(
        output=py_utils.NestedMap(b=b, c=c))

    ds_params = datasource.TFDatasetMixer.Params().Set(
        sub=[ds1, ds2, ds3], broadcast_dataset_structures=True)

    with self.session():
      ds = ds_params.Instantiate()
      with self.assertRaisesRegex(ValueError, 'Incompatible dataset specs'):
        self.evaluate(ds.GetNext())


class TFDSMnistInputGenerator(base_input_generator.BaseInputGenerator):

  def LoadTFDSDataset(self, info, features_dict):
    example = py_utils.NestedMap.FromNestedDict(features_dict)
    example.num_classes = info.features['label'].num_classes
    return example


class TFDSInputTest(test_utils.TestCase):

  def testTFDSInput(self):
    ds_params = datasource.TFDSInput.Params().Set(
        dataset='mnist', split='train[:10]')
    ds = ds_params.Instantiate()
    with self.session():
      with tfds.testing.mock_data(num_examples=10):
        batch = ds.GetNext()
      for _ in range(10):
        res = self.evaluate(batch)
        self.assertAllEqual((28, 28, 1), res['image'].shape)
        self.assertLess(res['label'], 10)
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(batch)

  def testTFDSInputLoadFn(self):
    ds_params = datasource.TFDSInput.Params().Set(
        dataset='mnist', split='train[:10]', load_fn='LoadTFDSDataset')
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TFDSMnistInputGenerator.Params().Instantiate())
    with self.session():
      with tfds.testing.mock_data(num_examples=10):
        batch = ds.GetNext()
      for _ in range(10):
        res = self.evaluate(batch)
        self.assertAllEqual((28, 28, 1), res.image.shape)
        self.assertLess(res.label, 10)
        self.assertEqual(res.num_classes, 10)
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(batch)


if __name__ == '__main__':
  tf.test.main()
