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


import lingvo.compat as tf

from lingvo.core import datasource
from lingvo.core import py_utils
from lingvo.core import test_utils


def _MockDataSourceFromFilePattern(file_pattern, input_source_weights=None):
  """Read and return input batch from a string file_pattern.

  Args:
    file_pattern: A string file pattern.
    input_source_weights: A list of float input source weights to control input
      example mix in the batch. The records will be sampled from inputs
      proportionally to these weights. Defaults to None which should be treated
      as an empty list.

  Returns:
    file_pattern.  This is the file_pattern that will be sent to InputOp along
    with input_source_weights and other parameters to generate a batch.
  """
  del input_source_weights  # Unused.
  return tf.constant([file_pattern])


class DatasourceTest(test_utils.TestCase):

  def testSimpleDataSourceSucceedsWithStringInput(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern='path_to_file')
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)
    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertAllEqual(ret.data, [[b'path_to_file']])

  def testSimpleDataSourceSucceedsWithFileType(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern='pattern1,pattern2', file_type='tfrecord')
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)
    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertAllEqual(ret.data, [[b'tfrecord:pattern1,pattern2']])

  def testSimpleDataSourceFailsWithListInput(self):
    files = ['file1', 'file2']
    ds_params = datasource.SimpleDataSource.Params().Set(file_pattern=files)

    ds = ds_params.Instantiate()
    with self.assertRaises(ValueError):
      ds.BuildDataSource(_MockDataSourceFromFilePattern)

  def testChainingDataSourceSucceedsWithListInput(self):
    files = ['path_to_file1', 'path_to_file2']
    ds_params = datasource.ChainingDataSource.Params().Set(file_patterns=files)
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)

    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertAllEqual(ret.data, [[b'path_to_file1,path_to_file2']])
    self.assertCountEqual(ret.bprop_variable_filters, [''] * len(files))

  def testChainingDataSourceFailsWithWeightedTupleListInput(self):
    files = [('file1', 1.0), ('file2', 2.0)]
    ds_params = datasource.ChainingDataSource.Params().Set(file_patterns=files)
    ds = ds_params.Instantiate()

    with self.assertRaises(ValueError):
      ds.BuildDataSource(_MockDataSourceFromFilePattern)

  def testWithinBatchMixingDataSourceSucceedsWithListFilesAndWeights(self):
    files = ['path_to_file1', 'path_to_file2']
    weights = [1, 4]
    ds_params = datasource.WithinBatchMixingDataSource.Params().Set(
        file_patterns=files, weights=weights)
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)

    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertAllEqual(ret.data, [[b'path_to_file1,path_to_file2']])
    self.assertCountEqual(ret.bprop_variable_filters, [''] * len(files))

  # TODO(b/139345706) should fail when the p.file_pattern behavior is deprecated
  def testWithinBatchMixingDataSourceFailsWithListTuplesFiles(self):
    # This legacy p.file_pattern behavior is only supported through
    # base_input_generator.
    files = [('file1', 1.0), ('file2', 2.0)]
    ds_params = datasource.WithinBatchMixingDataSource.Params().Set(
        file_patterns=files)
    ds = ds_params.Instantiate()

    with self.assertRaises(ValueError):
      ds.BuildDataSource(_MockDataSourceFromFilePattern)

  def testCrossBatchMixingDataSourceSucceedsWithListFilesAndWeights(self):
    files = ['path_to_file', 'path_to_file']
    weights = [1, 4]
    ds_params = datasource.CrossBatchMixingDataSource.Params().Set(
        file_patterns=files, weights=weights)
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)

    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertCountEqual(
        sorted(ret.keys()),
        ['bprop_variable_filters', 'data', 'selected_bprop', 'source_selected'])
    # CrossBatchMixing operates on the python side of the tf op so a single
    # element will be returned by _MockDataSourceFromFilePattern
    self.assertAllEqual(ret.data, [[b'path_to_file']])
    self.assertCountEqual(ret.bprop_variable_filters, [''] * len(files))
    self.assertAllEqual(ret.selected_bprop.shape, [2])
    self.assertAllEqual(ret.source_selected.shape, [1, 2])

  def testCrossBatchMixingDataSourceFailsWithListTuplesFiles(self):
    # This legacy p.file_pattern behavior is only supported through
    # base_input_generator.
    files = [('file1', 1.0), ('file2', 2.0)]
    ds_params = datasource.CrossBatchMixingDataSource.Params().Set(
        file_patterns=files)
    ds = ds_params.Instantiate()

    with self.assertRaises(ValueError):
      ds.BuildDataSource(_MockDataSourceFromFilePattern)

  def testCurriculumDataSourceSucceedsWithSimpleDataSource(self):
    sources = [
        datasource.SimpleDataSource.Params().Set(file_pattern='file1'),
        datasource.SimpleDataSource.Params().Set(file_pattern='file2'),
    ]
    ds_params = datasource.CurriculumDataSource.Params().Set(
        datasource_params=sources, boundaries=[5])
    ds = ds_params.Instantiate()

    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)
    with tf.Session():
      self.evaluate(tf.global_variables_initializer())
      ret.data = self.evaluate([ret.data])

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertAllEqual(ret.data, [[b'file1']])
    self.assertCountEqual(ret.bprop_variable_filters, [''])

  def testCurriculumDataSourceTransitionsCorrectlyWithSimpleDataSource(self):
    sources = [
        datasource.SimpleDataSource.Params().Set(file_pattern='file1'),
        datasource.SimpleDataSource.Params().Set(file_pattern='file2'),
    ]
    boundary = 5
    ds_params = datasource.CurriculumDataSource.Params().Set(
        datasource_params=sources, boundaries=[boundary])
    ds = ds_params.Instantiate()

    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)
    with tf.Session():
      # Advance the global step to the next curriculum stage
      global_step = py_utils.GetOrCreateGlobalStepVar()
      self.evaluate(tf.global_variables_initializer())
      set_global_step = tf.assign(global_step, boundary, name='advance_step')
      self.evaluate(set_global_step)

      ret.data = self.evaluate([ret.data])

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertAllEqual(ret.data, [[b'file2']])
    self.assertCountEqual(ret.bprop_variable_filters, [''])

  def testCurriculumDataSourceTransitionsCorrectlyWithMixingDataSource(self):
    sources = [
        datasource.WithinBatchMixingDataSource.Params().Set(
            file_patterns=['file1', 'file2'], weights=[1, 5]),
        datasource.WithinBatchMixingDataSource.Params().Set(
            file_patterns=['file3', 'file4'], weights=[2, 3])
    ]
    boundary = 5
    ds_params = datasource.CurriculumDataSource.Params().Set(
        datasource_params=sources, boundaries=[boundary])
    ds = ds_params.Instantiate()

    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)
    with tf.Session():
      # Advance the global step to the next curriculum stage
      global_step = py_utils.GetOrCreateGlobalStepVar()
      self.evaluate(tf.global_variables_initializer())
      set_global_step = tf.assign(global_step, boundary, name='advance_step')
      self.evaluate(set_global_step)

      ret.data = self.evaluate([ret.data])

    self.assertCountEqual(
        sorted(ret.keys()), ['bprop_variable_filters', 'data'])
    self.assertAllEqual(ret.data, [[b'file3,file4']])
    self.assertCountEqual(ret.bprop_variable_filters, [''])

  def testCurriculumDataSourceFailsWithBadBoundaries(self):
    sources = [
        datasource.WithinBatchMixingDataSource.Params().Set(
            file_patterns=['file1', 'file2'], weights=[1, 5]),
        datasource.WithinBatchMixingDataSource.Params().Set(
            file_patterns=['file3', 'file4'], weights=[2, 3])
    ]
    ds_params = datasource.CurriculumDataSource.Params().Set(
        datasource_params=sources, boundaries=[10, 5])
    ds = ds_params.Instantiate()
    with self.assertRaises(ValueError):
      ds.BuildDataSource(_MockDataSourceFromFilePattern)

  def testPrefixDataSourceSucceedsWithDirectory(self):
    ds_params = datasource.PrefixedDataSource.Params().Set(
        file_pattern='filename-*.tfrecord',
        file_type=None,
        file_pattern_prefix='/dir/')
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)

    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertAllEqual(ret.data, [[b'/dir/filename-*.tfrecord']])

  def testPrefixDataSourceSucceedsWithMultiplePatterns(self):
    ds_params = datasource.PrefixedDataSource.Params().Set(
        file_pattern='filename-*.tfrecord,other/file/pattern/*',
        file_type=None,
        file_pattern_prefix='/dir/')
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)

    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertAllEqual(
        ret.data, [[b'/dir/filename-*.tfrecord,/dir/other/file/pattern/*']])

  def testPrefixDataSourceSucceedsWithGcsBucket(self):
    ds_params = datasource.PrefixedDataSource.Params().Set(
        file_pattern='filename-*.tfrecord',
        file_type=None,
        file_pattern_prefix='gs://bucket/dir')
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)

    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertAllEqual(ret.data, [[b'gs://bucket/dir/filename-*.tfrecord']])

  def testPrefixDataSourceSucceedsWithFileType(self):
    ds_params = datasource.PrefixedDataSource.Params().Set(
        file_pattern='filename-*.tfrecord',
        file_type='tfrecord',
        file_pattern_prefix='dir')
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)

    with tf.Session():
      ret.data = self.evaluate([ret.data])

    self.assertAllEqual(ret.data, [[b'tfrecord:dir/filename-*.tfrecord']])


if __name__ == '__main__':
  tf.test.main()
