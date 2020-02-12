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
"""DataSources describe how files should be used to provide data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils

import six
from six.moves import range


class DataSource(base_layer.BaseLayer):
  """A base class for file based Data Sources."""

  @classmethod
  def Params(cls):
    return super(DataSource, cls).Params().Set(name='datasource')

  def BuildDataSource(self, data_source_from_file_pattern_fn):
    """Builds a data source.

    Subclasses implement this.

    Args:
      data_source_from_file_pattern_fn: a function that takes file_pattern and
        input_source_weights as arguments and returns an input batch from a
        string file_pattern.

    Returns:
      A NestedMap containing

      - data: (Required) a tuple of tf.Tensor or `.NestedMap` of tf.Tensor same
          as ``BaseInputGeneratorFromFiles._DataSourceFromFilePattern(
          file_pattern, input_source_weights=None)``
      - source_selected: (Optional) a tensor of size
          [batch_size, number of datasources]
      - selected_bprop: (Optional) a tensor of size [number of data sources]
      - bprop_variable_filters: (Optional) containing a list of bprop_variable
          filters for each source.
    """
    raise NotImplementedError()


class SimpleDataSource(DataSource):
  """A simple file based data source."""

  @classmethod
  def Params(cls):
    p = super(SimpleDataSource, cls).Params()
    # TODO(b/139345706): move filetype prefix (eg tfrecord:) into its own param
    # and clean up existing usages.
    p.Define(
        'file_pattern', '', 'A single file pattern string which can '
        'contain a single file pattern, or a comma separated list of patterns.'
        'Samples from each file with unspecified likelihood, though in practice'
        ' this will be roughly equal per file.  To explicitly '
        'describe the mixture weights of different file patterns use '
        'WithinBatchMixingDataSource or CrossBatchMixingDataSource')
    p.Define('file_type', '', 'A file type, such as `tfrecord`.')

    return p

  def BuildDataSource(self, data_source_from_file_pattern_fn):
    """Builds a simple, unweighted Data Source.

    Args:
      data_source_from_file_pattern_fn: a function that takes file_pattern as an
        argument and returns an input batch.

    Returns:
      A NestedMap containing `data`, which is a tuple of tf.Tensor or
      `.NestedMap` of tf.Tensor.
    """
    p = self.params
    if not isinstance(p.file_pattern, six.string_types):
      raise ValueError('SimpleDataSource expects p.file_pattern to be a string.'
                       ' To use multiple files use a comma separated string, '
                       'e.g. \', \'.join(list_of_file_patterns)')

    if p.file_type:
      file_pattern = '{}:{}'.format(p.file_type, p.file_pattern)
    else:
      file_pattern = p.file_pattern

    ret = py_utils.NestedMap()
    ret.data = data_source_from_file_pattern_fn(file_pattern)
    ret.bprop_variable_filters = ['']
    return ret


class ChainingDataSource(DataSource):
  """A data source that reads each file_pattern in sequence."""

  @classmethod
  def Params(cls):
    p = super(ChainingDataSource, cls).Params()
    # TODO(b/139345706): This can probably be a list of DataSource params
    # instead of a list of file_patterns to be more generic.
    p.Define(
        'file_patterns', [], 'A list of file pattern strings which are read '
        'from in sequence. Commas cannot be used in individual file_patterns.')
    return p

  def BuildDataSource(self, data_source_from_file_pattern_fn):
    """Builds a Chaining Data Source.

    Args:
      data_source_from_file_pattern_fn: a function that takes file_pattern as an
        argument and returns an input batch.

    Returns:
      A NestedMap containing `data`, which is a tuple of tf.Tensor or
      `.NestedMap` of tf.Tensor.

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params
    if not isinstance(p.file_patterns, list):
      raise ValueError('Expected a list, got %s' % (p.file_patterns,))
    if not all(isinstance(x, six.string_types) for x in p.file_patterns):
      # Chaining doesn't work with weights or backprop filters, i.e. when
      # file_pattern param contains a list of
      # <file_pattern, weight, [bprop_variable_filter]> tuples.
      raise ValueError('Expected a list of strings, got %s' %
                       (p.file_patterns,))

    for file_pattern in p.file_patterns:
      if ',' in file_pattern:
        raise ValueError('Can not use commas in file_pattern when chaining '
                         'is used. file_pattern: %s' % (file_pattern,))
    ret = py_utils.NestedMap()
    ret.data = data_source_from_file_pattern_fn(','.join(p.file_patterns))
    ret.bprop_variable_filters = [''] * len(p.file_patterns)
    return ret


class WithinBatchMixingDataSource(DataSource):
  """Mixes records from different sources into the same batch."""

  @classmethod
  def Params(cls):
    p = super(WithinBatchMixingDataSource, cls).Params()
    # TODO(b/139345706): This can probably be a list of DataSource params
    # instead of a list of file_patterns to be more generic.
    p.Define(
        'file_patterns', [], 'A list of file pattern strings which are read '
        'from in sequence. Commas cannot be used in individual file_patterns. ')
    p.Define('weights', [], 'A list of weights for each file pattern')
    return p

  def BuildDataSource(self, data_source_from_file_pattern_fn):
    """Read and return input batch from p.file_patterns list weighted by p.weights.

    Examples in the batch will be mixed together from different file_pattern
    source proportionally to the weights.

    Args:
      data_source_from_file_pattern_fn: a function that takes file_pattern and
        input_source_weights as arguments and returns an input batch from a
        string file_pattern.

    Returns:
      A NestedMap containing: data: a tuple of tf.Tensor or `.NestedMap` of
      tf.Tensor

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params
    if not isinstance(p.file_patterns, list):
      raise ValueError('Expected a list, got %s' % (p.file_patterns,))
    if not isinstance(p.weights, list):
      raise ValueError('Expected a list, got %s' % (p.weights,))
    if len(p.file_patterns) != len(p.weights):
      raise ValueError(
          'Expected p.file_patterns and p.weights to be the same length. '
          'Found %d file_patterns, and %d weights' %
          (len(p.file_patterns), len(p.weights)))
    # TODO(rosenberg) confirm that weights are numeric
    if not all(isinstance(x, six.string_types) for x in p.file_patterns):
      raise ValueError('Expected all elements of p.file_patterns to be strings')

    file_patterns = p.file_patterns
    weights = p.weights
    for file_pattern in file_patterns:
      if ',' in file_pattern:
        raise ValueError('Can not use commas in file_pattern when within-batch '
                         'mixing is used. file_pattern: %s' % (file_pattern,))
    ret = py_utils.NestedMap()
    ret.data = data_source_from_file_pattern_fn(
        ','.join(file_patterns), input_source_weights=weights)
    ret.bprop_variable_filters = [''] * len(file_patterns)
    return ret


class CrossBatchMixingDataSource(DataSource):
  """Mixes batches from different sources, each batch from only one source."""

  @classmethod
  def Params(cls):
    p = super(CrossBatchMixingDataSource, cls).Params()
    # TODO(b/139345706): This can probably be a list of DataSource params
    # instead of a list of file_patterns to be more generic.
    p.Define(
        'file_patterns', [], 'A list of file pattern strings which are read '
        'from in sequence. Commas cannot be used in individual file_patterns. ')
    p.Define('weights', [], 'A list of weights for each file pattern')
    p.Define(
        'bprop_variable_filters', [], 'An optional list of '
        'bprop_variariable_filters for each file_pattern.  If not empty, '
        'expected to have the same length as file_pattern and weights')
    return p

  def BuildDataSource(self, data_source_from_file_pattern_fn):
    """Read and return input batch from a p.file_pattern list.

    `p.file_patterns` is a list of file patterns, `p.weights` contains
    weights for each file pattern.  If provided `p.bprop_variable_filters`
    includes a bprop_variable_filter for each file pattern.

    Args:
      data_source_from_file_pattern_fn: a function that takes file_pattern as an
        argument and returns an input batch.

    Returns:
      A NestedMap containing:
        data: a tuple of tf.Tensor or `.NestedMap` of tf.Tensor
        source_selected: a tensor of size [batch_size, number of data sources]
        selected_bprop: a tensor of size [number of data sources]
        bprop_variable_filters: containing a list of bprop_variable filters for
        each source

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params

    def _MakeDataSourceFromFilePatternFunc(data_source_from_file_pattern_fn,
                                           file_pattern):
      # It's important to invoke self._DataSourceFromFilePattern() inside the
      # lambda to make sure that the record is drawn from data source
      # only if it will be used. Weights are handled by MixByWeight, not the
      # data_source_from_file_pattern_fn.
      return lambda: data_source_from_file_pattern_fn(file_pattern)

    if len(p.weights) != len(p.file_patterns):
      raise ValueError(
          'Expected p.file_patterns and p.weights to be the same length. '
          'Found %d file_patterns, and %d weights' %
          (len(p.file_patterns), len(p.weights)))
    if not all(isinstance(x, six.string_types) for x in p.file_patterns):
      raise ValueError('Expected all elements of p.file_patterns to be strings')

    # TODO(rosenberg) replace this with functools.partial
    inputs = [
        _MakeDataSourceFromFilePatternFunc(data_source_from_file_pattern_fn,
                                           file_pattern)
        for file_pattern in p.file_patterns
    ]
    weights = p.weights
    if not p.bprop_variable_filters:
      bprop_variable_filters = [''] * len(inputs)
    else:
      bprop_variable_filters = p.bprop_variable_filters

    data_source, selected_bprop = py_utils.MixByWeight(
        inputs, weights, seed=p.random_seed)
    # TODO(neerajgaur): Remove _bprop_onehot and change code that uses it to
    # use source_selected from input_batch.
    batch_size = py_utils.GetShape(tf.nest.flatten(data_source)[0])[0]
    ret = py_utils.NestedMap()
    ret.data = data_source
    ret.bprop_variable_filters = bprop_variable_filters
    ret.selected_bprop = selected_bprop
    ret.source_selected = tf.tile(
        tf.expand_dims(selected_bprop, 0), [batch_size, 1])
    return ret


class CurriculumDataSource(DataSource):
  """A data source that reads different DataSources in stages.

  Supports multiple stages of training, where each stage yields batches for
  based on global step boundaries. The contents per stage are defined by nested
  DataSources.

  Boundaries are defined by the training global step.

  Currently bprop_variable_filters are not_supported.
  # TODO(rosenberg) support bprop_variable_filter within CurriculumDataSource
  The issue here is that by conditioning on tf.Variable global_step, the
  bprop_variable_filter from the selected DataSource is itself a tf.Variable.
  Other bprop_variable_filters are python strings.
  """

  @classmethod
  def Params(cls):
    p = super(CurriculumDataSource, cls).Params()
    p.Define(
        'datasource_params', [], 'A list of DataSource Params which define '
        'the DataSource curriculum.')
    p.Define(
        'boundaries', [], 'A list of global step thresholds determining when '
        'to move from one training stage to another.')
    p.Define(
        'bprop_variable_filters', [''], 'A list of bprop_variable_filters to '
        'apply during training.  NOTE: these are constant across all stages.'
        'Changing variable filters per stage is not supported.')
    return p

  def BuildDataSource(self, data_source_from_file_pattern_fn):
    """Read and return input batch.

    Args:
      data_source_from_file_pattern_fn: a function to read and return input
        batch from a string file_pattern

    Returns:
      A NestedMap containing:
        data: a tuple of tf.Tensor or `.NestedMap` of tf.Tensor

    Raises:
      ValueError: inconsistent sizes between boundaries and datasource_params,
      specification of unsupported datasources, or out of order boundaries.
    """
    p = self.params

    if len(p.datasource_params) != len(p.boundaries) + 1:
      raise ValueError(
          'Expected p.datasource_params to have one more entry than '
          'p.boundaries. Found %d datasource_params, and %d boundaries' %
          (len(p.datasource_params), len(p.boundaries)))

    for ds_p in p.datasource_params:
      if 'bprop_variable_filters' in ds_p:
        if any(filter for filter in ds_p.bprop_variable_filters):
          raise ValueError('CurriculumDataSource does not support distinct '
                           'bprop_variable_filters per stage.')

    for idx in range(len(p.boundaries) - 1):
      if p.boundaries[idx] > p.boundaries[idx + 1]:
        raise ValueError('Expected p.boundaries to monotonically increase, but '
                         'found %d > %d at position %d' %
                         (p.boundaries[idx], p.boundaries[idx + 1], idx))

    global_step = py_utils.GetOrCreateGlobalStepVar()
    datasources = [ds_p.Instantiate() for ds_p in p.datasource_params]

    def GetDatasourceFn(idx):

      def DatasourceFn():
        datasource = datasources[idx].BuildDataSource(
            data_source_from_file_pattern_fn)
        datasource.pop('bprop_variable_filters', None)
        return datasource

      return DatasourceFn

    cases = []
    for idx in range(len(p.boundaries)):
      cases.append(
          (tf.less(global_step,
                   tf.constant(p.boundaries[idx],
                               dtype=global_step.dtype)), GetDatasourceFn(idx)))

    ret = tf.case(cases, default=GetDatasourceFn(-1))
    ret.bprop_variable_filters = p.bprop_variable_filters
    return ret


class PrefixedDataSource(SimpleDataSource):
  """Prepends path prefix to file patterns."""

  @classmethod
  def Params(cls):
    p = super(PrefixedDataSource, cls).Params()
    p.Define(
        'file_pattern_prefix', '',
        'Prefix to add to file_pattern, eg. a base directory that contains '
        'dataset files.')

    return p

  def __init__(self, params):
    super(PrefixedDataSource, self).__init__(params)

    p = self.params

    patterns = p.file_pattern.split(',')
    p.file_pattern = ','.join(
        os.path.join(p.file_pattern_prefix, pattern) for pattern in patterns)
