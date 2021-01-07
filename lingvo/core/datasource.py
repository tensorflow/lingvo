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
"""DataSources describe how files should be used to provide data."""

import os

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils


class DataSource(base_layer.BaseLayer):
  """A base class for file based Data Sources."""

  @classmethod
  def Params(cls):
    return super().Params().Set(name='datasource')

  def __init__(self, params):
    super().__init__(params)
    self.SetVariableFree()

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
  """A simple file based data source.

  This uses the Lingvo input pipeline consisting of input_common.h and
  record_yielder/record_batcher.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'file_pattern', '', 'A string or list of file pattern strings. '
        'If it is a comma separated string, it will be interpreted as a list. '
        'If a list is provided elements may not contain commas.')
    # TODO(b/139345706): move filetype prefix (eg tfrecord:) into its own param
    # and clean up existing usages.
    p.Define('file_type', '', 'A string or list of file types, eg. `tfrecord`.')
    p.Define(
        'weights', None,
        'A list of weights for each file pattern for within-batch mixing. If '
        'not specified, a default implementation is used (roughly uniform).')
    p.Define(
        'bprop_variable_filters', None, 'An optional list of '
        'bprop_variariable_filters for each file_pattern.  If not empty, '
        'expected to have the same length as weights.')
    return p

  def BuildDataSource(self, data_source_from_file_pattern_fn):
    """Read and return input batch from p.sub weighted by p.weights.

    Examples in the batch will be mixed together from different sources
    proportional to the weights.

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
    if p.weights:
      if not isinstance(p.file_pattern, list):
        raise ValueError('Expected a list, got %s' % p.file_pattern)
      if not isinstance(p.weights, list):
        raise ValueError('Expected a list, got %s' % p.weights)
      if len(p.file_pattern) != len(p.weights):
        raise ValueError(
            'Expected p.file_pattern and p.weights to be the same length. '
            'Found %d file_pattern, and %d weights' %
            (len(p.file_pattern), len(p.weights)))
      # TODO(rosenberg) confirm that weights are numeric

    if isinstance(p.file_pattern, str):
      p.file_pattern = p.file_pattern.split(',')
      if p.file_type:
        p.file_type = [p.file_type] * len(p.file_pattern)
    else:
      if not all(isinstance(x, str) for x in p.file_pattern):
        raise ValueError(
            'Expected all elements of p.file_pattern to be strings.' +
            str(p.file_pattern))
      for x in p.file_pattern:
        if ',' in x:
          raise ValueError('List file_pattern %s should not contain commas.' %
                           p.file_pattern)

    file_patterns = p.file_pattern
    if p.file_type:
      file_patterns = [f'{t}:{p}' for t, p in zip(p.file_type, file_patterns)]

    ret = py_utils.NestedMap()
    if p.weights:
      # Within-batch mixing.
      ret.data = data_source_from_file_pattern_fn(
          ','.join(file_patterns), input_source_weights=p.weights)
    else:
      # Default.
      ret.data = data_source_from_file_pattern_fn(','.join(file_patterns))

    if not p.bprop_variable_filters:
      ret.bprop_variable_filters = [''] * len(file_patterns)
    else:
      ret.bprop_variable_filters = p.bprop_variable_filters
    return ret


class CrossBatchMixingDataSource(DataSource):
  """Mixes batches from different sources, each batch from only one source."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('sub', None, 'A list of datasources to mix.')
    p.Define('weights', None, 'A list of weights for each datasource.')
    p.Define(
        'bprop_variable_filters', None, 'An optional list of '
        'bprop_variariable_filters for each datasource. If not empty, '
        'expected to have the same length as sub and weights')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateChildren('sub', self.params.sub)

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

    if len(p.weights) != len(p.sub):
      raise ValueError('Expected p.sub and p.weights to be the same length. '
                       'Found %d sub, and %d weights' %
                       (len(p.sub), len(p.weights)))

    def GetDatasourceFn(sub):

      def DatasourceFn():
        datasource = sub.BuildDataSource(data_source_from_file_pattern_fn)
        return datasource.data

      return DatasourceFn

    inputs = [GetDatasourceFn(sub) for sub in self.sub]
    if not p.bprop_variable_filters:
      bprop_variable_filters = [''] * len(inputs)
    else:
      bprop_variable_filters = p.bprop_variable_filters

    data_source, selected_bprop = py_utils.MixByWeight(
        inputs, p.weights, seed=p.random_seed)
    # TODO(neerajgaur): Remove _bprop_onehot and change code that uses it to
    # use source_selected from input_batch.
    batch_size = py_utils.GetShape(py_utils.Flatten(data_source)[0])[0]
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
    p = super().Params()
    p.Define('sub', None,
             'A list of DataSource Params which define the curriculum.')
    p.Define(
        'boundaries', None, 'A list of global step thresholds determining when '
        'to move from one training stage to another.')
    p.Define(
        'bprop_variable_filters', None, 'A list of bprop_variable_filters to '
        'apply during training.  NOTE: these are constant across all stages.'
        'Changing variable filters per stage is not supported.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateChildren('sub', self.params.sub)

  def BuildDataSource(self, data_source_from_file_pattern_fn):
    """Read and return input batch.

    Args:
      data_source_from_file_pattern_fn: a function to read and return input
        batch from a string file_pattern

    Returns:
      A NestedMap containing:
        data: a tuple of tf.Tensor or `.NestedMap` of tf.Tensor

    Raises:
      ValueError: inconsistent sizes between boundaries and sub, specification
      of unsupported datasources, or out of order boundaries.
    """
    p = self.params

    if len(p.sub) != len(p.boundaries) + 1:
      raise ValueError('Expected p.sub to have one more entry than '
                       'p.boundaries. Found %d sub, and %d boundaries' %
                       (len(p.sub), len(p.boundaries)))

    for ds_p in p.sub:
      if 'bprop_variable_filters' in ds_p and ds_p.bprop_variable_filters:
        if any(filter for filter in ds_p.bprop_variable_filters):
          raise ValueError('CurriculumDataSource does not support distinct '
                           'bprop_variable_filters per stage.')

    for idx in range(len(p.boundaries) - 1):
      if p.boundaries[idx] > p.boundaries[idx + 1]:
        raise ValueError('Expected p.boundaries to monotonically increase, but '
                         'found %d > %d at position %d' %
                         (p.boundaries[idx], p.boundaries[idx + 1], idx))

    global_step = py_utils.GetGlobalStep()

    def GetDatasourceFn(idx):

      def DatasourceFn():
        datasource = self.sub[idx].BuildDataSource(
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
    ret.bprop_variable_filters = p.bprop_variable_filters or ['']
    return ret


class PrefixedDataSource(SimpleDataSource):
  """Prepends path prefix to file patterns."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'file_pattern_prefix', '',
        'Prefix to add to file_pattern, eg. a base directory that contains '
        'dataset files.')

    return p

  def __init__(self, params):
    if isinstance(params.file_pattern, str):
      params.file_pattern = params.file_pattern.split(',')
      if params.file_type:
        params.file_type = [params.file_type] * len(params.file_pattern)
    prefixed = []
    for file_pattern in params.file_pattern:
      patterns = file_pattern.split(',')
      prefixed.append(','.join(
          os.path.join(params.file_pattern_prefix, pattern)
          for pattern in patterns))
    params.file_pattern = prefixed
    super().__init__(params)
