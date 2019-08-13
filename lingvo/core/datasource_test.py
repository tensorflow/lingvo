# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf

from lingvo.core import datasource


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


class DatasourceTest(tf.test.TestCase):

  def testSimpleDataSourceSucceedsWithStringInput(self):
    ds_params = datasource.SimpleDataSource.Params().Set(
        file_pattern='path_to_file')
    ds = ds_params.Instantiate()
    ret = ds.BuildDataSource(_MockDataSourceFromFilePattern)
    with tf.Session() as sess:
      ret.data = sess.run([ret.data])

    self.assertCountEqual(sorted(ret.keys()), ['data'])
    self.assertAllEqual(ret.data, [[b'path_to_file']])

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

    with tf.Session() as sess:
      ret.data = sess.run([ret.data])

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

    with tf.Session() as sess:
      ret.data = sess.run([ret.data])

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

    with tf.Session() as sess:
      ret.data = sess.run([ret.data])

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


if __name__ == '__main__':
  tf.test.main()
