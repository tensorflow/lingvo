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
"""DataSources describe how data should be produced.

There are two broad types of DataSources:
1) A source that reads some type of resource to produce tensors;
2) A metasource/transformation that modifies the output of other datasources.

The InputGenerator instance is passed in to each DataSource. A DataSource
subclass may attempt to call specific methods or access specific attributes on
the InputGenerator instance, and it is up to the user to ensure such
requirements are met.
"""

import functools
import os
import uuid

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import batch_utils
from lingvo.core import cluster
from lingvo.core import py_utils


class DataSource(base_layer.BaseLayer):
  """A base class for data sources."""

  @classmethod
  def Params(cls):
    return super().Params().Set(name='datasource')

  def __init__(self, params):
    super().__init__(params)
    self._input_generator = None

  def SetInputGenerator(self, input_generator):
    """Initialize this datasource.

    Args:
      input_generator: A reference to the external input_generator.
    """
    self._input_generator = input_generator

    for child in self._children_list:
      if isinstance(child, DataSource):
        child.SetInputGenerator(input_generator)

  def Initialize(self, sess=None):
    for child in self._children_list:
      if isinstance(child, DataSource):
        child.Initialize(sess)

  def Reset(self, sess=None):
    for child in self._children_list:
      if isinstance(child, DataSource):
        child.Reset(sess)

  def GetNext(self):
    """Override this method to return the next element from the datasource.

    Returns:
      A `.NestedMap` containing the next element from the datasource. Depending
      on the datasource this may be a single tensor or a batch of tensors.
    """
    raise NotImplementedError()

  def GetMeta(self):
    """Gets metadata for the batch."""
    return py_utils.NestedMap()


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
    p.Define('file_type', '', 'A string file types, eg. `tfrecord`.')
    p.Define(
        'weights', None,
        'A list of weights for each file pattern for within-batch mixing. If '
        'not specified, a default implementation is used (roughly uniform).')
    p.Define(
        'source_id_offset', 0,
        'All source_id values from this source will be offset by this value.')
    p.Define(
        'bprop_variable_filters', None, 'An optional list of '
        'bprop_variariable_filters for each file_pattern.  If not empty, '
        'expected to have the same length as weights.')
    return p

  def __init__(self, params):
    super().__init__(params)
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
    else:
      if not all(isinstance(x, str) for x in p.file_pattern):
        raise ValueError(
            'Expected all elements of p.file_pattern to be strings.' +
            str(p.file_pattern))
      for x in p.file_pattern:
        if ',' in x:
          raise ValueError('List file_pattern %s should not contain commas.' %
                           p.file_pattern)

  def GetNext(self):
    """Return input batch from p.file_patterns list weighted by p.weights.

    Examples in the batch will be mixed together from different file_pattern
    source proportionally to the weights.

    Returns:
      An input batch.
    """
    p = self.params
    file_patterns = ','.join(p.file_pattern)
    if p.file_type:
      file_patterns = f'{p.file_type}:{file_patterns}'

    extra_args = dict()
    if p.source_id_offset != 0:
      extra_args['input_source_id_offset'] = p.source_id_offset

    # In TF1 mode, the python method `GetNext` only gets called once for each
    # DataSource object during graph construction.
    # In TF2 mode, however, `GetNext` can be called many times. We must specify
    # keys to uniquely identify its `GenericInputV2` resource. This
    # ensures that the resource is properly reused.
    if py_utils.IsEagerMode():
      # The current DataSource object is used as the key to GenericInputV2 ops.
      extra_args['generic_input_v2_key'] = self

    if p.weights:
      # Within-batch mixing.
      batch = self._input_generator._DataSourceFromFilePattern(  # pylint: disable=protected-access
          file_patterns,
          input_source_weights=p.weights,
          **extra_args)
    else:
      # Default.
      batch = self._input_generator._DataSourceFromFilePattern(  # pylint: disable=protected-access
          file_patterns, **extra_args)

    return batch

  def GetMeta(self):
    p = self.params

    ret = py_utils.NestedMap()
    if not p.bprop_variable_filters:
      ret.bprop_variable_filters = [''] * len(p.file_pattern)
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
    p = self.params

    if len(p.weights) != len(p.sub):
      raise ValueError('Expected p.sub and p.weights to be the same length. '
                       'Found %d sub, and %d weights' %
                       (len(p.sub), len(p.weights)))

    self.CreateChildren('sub', p.sub)

  def GetNext(self):
    p = self.params
    inputs = [sub.GetNext for sub in self.sub]

    data_source, self._selected_bprop = py_utils.MixByWeight(
        inputs, p.weights, seed=p.random_seed)
    # TODO(neerajgaur): Remove _bprop_onehot and change code that uses it to
    # use source_selected from input_batch.
    tensor = py_utils.Flatten(data_source)[0]
    if py_utils.IsEagerMode():
      self._batch_size = tf.cond(
          tf.rank(tensor) == 0,
          lambda: 1,
          lambda: tf.shape(tensor)[0],
      )
    else:
      shape = py_utils.GetShape(tensor)
      self._batch_size = shape[0] if shape != [] else 1  # pylint: disable=g-explicit-bool-comparison
    return data_source

  def GetMeta(self):
    p = self.params
    ret = py_utils.NestedMap()

    if not p.bprop_variable_filters:
      ret.bprop_variable_filters = [''] * len(p.sub)
    else:
      ret.bprop_variable_filters = p.bprop_variable_filters
    ret.selected_bprop = self._selected_bprop
    ret.source_selected = tf.tile(
        tf.expand_dims(self._selected_bprop, 0), [self._batch_size, 1])
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

    self.CreateChildren('sub', p.sub)

  def GetNext(self):
    p = self.params

    global_step = py_utils.GetGlobalStep()

    cases = []
    for idx in range(len(p.boundaries)):
      cases.append(
          (tf.less(global_step,
                   tf.constant(p.boundaries[idx], dtype=global_step.dtype)),
           self.sub[idx].GetNext))

    return tf.case(cases, default=self.sub[-1].GetNext)

  def GetMeta(self):
    return py_utils.NestedMap(
        bprop_variable_filters=self.params.bprop_variable_filters or [''])


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
    prefixed = []
    for file_pattern in params.file_pattern:
      patterns = file_pattern.split(',')
      prefixed.append(','.join(
          os.path.join(params.file_pattern_prefix, pattern)
          for pattern in patterns))
    params.file_pattern = prefixed
    super().__init__(params)


class TFDatasetSource(DataSource):
  """Base DataSource class based on tf.data.Dataset."""

  def __init__(self, params):
    super().__init__(params)
    self._dataset = {}
    self._iterator = {}

  @property
  def num_hosts(self):
    if (self._input_generator and
        self._input_generator.params.use_per_host_infeed):
      return max(self.cluster.num_tpu_hosts, 1)
    return 1

  @property
  def host_id(self):
    if self.num_hosts > 1:
      return cluster.GetInfeedContext().infeed_host_index
    return 0

  @property
  def _map_args(self):
    """Arguments for calls to dataset.map() or similar functions."""
    return {
        'num_parallel_calls':
            1 if self.cluster.in_unit_test else tf.data.experimental.AUTOTUNE,
        'deterministic':
            self.cluster.require_sequential_input_order
    }

  def GetDataset(self):
    """Override to return a tf.data.Dataset containing a NestedMap."""
    raise NotImplementedError()

  def _InitIterator(self):
    if self.host_id in self._dataset:
      return

    with py_utils.GlobalStepContext(None):
      # Hide global_step tensor from being captured by dataset function.
      ds = self.GetDataset()
    options = tf.data.Options()
    options.experimental_deterministic = bool(self.cluster.in_unit_test)
    ds = ds.with_options(options)
    self._dataset[self.host_id] = ds
    if tf.executing_eagerly_outside_functions():
      it = iter(ds)
    else:
      it = tf.data.make_initializable_iterator(ds)
    self._iterator[self.host_id] = it

  def Initialize(self, sess=None):
    if not tf.executing_eagerly_outside_functions():
      sess.run([it.initializer for it in self._iterator.values()])
    super().Initialize(sess)

  def Reset(self, sess=None):
    if tf.executing_eagerly_outside_functions():
      self._iterator = {key: iter(ds) for key, ds in self._dataset.items()}
    else:
      sess.run([it.initializer for it in self._iterator.values()])
    super().Reset(sess)

  def GetNext(self):
    """Returns the next element from the dataset."""
    # Use `init_scope()` to ensure that the datasets and iterators are created
    # outside of the function-building graph. This ensures that these creation
    # operations are not repeated in subsequent `tf.function` calls.
    with tf.init_scope():
      self._InitIterator()
    if py_utils.GetUnitTestSession():
      self.Initialize(py_utils.GetUnitTestSession())
    return self._iterator[self.host_id].get_next()


class TFDatasetAdaptor(TFDatasetSource):
  """Converts a DataSource into a TFDatasetSource."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('sub', None, 'A DataSource to adapt.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateChild('sub', self.params.sub)

  def GetDataset(self):
    return tf.data.Dataset.from_tensors(0).repeat().map(
        lambda _: self.sub.GetNext(), **self._map_args)


class TFDatasetTransform(TFDatasetSource):
  """Transforms the output of a child TFDatasetSource."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'sub', None, 'A DatasetSource to adapt. '
        'If it is not a TFDatasetSource, TFDatasetAdaptor will be used.')
    return p

  def __init__(self, params):
    super().__init__(params)
    ds = self.params.sub
    if not issubclass(ds.cls, TFDatasetSource):
      ds = TFDatasetAdaptor.Params().Set(sub=ds)
    self.CreateChild('sub', ds)

  def GetDataset(self):
    return self.Transform(self.sub.GetDataset())

  def Transform(self, dataset):
    """Returns a transformed tf.data.Dataset."""
    raise NotImplementedError()


class CustomTFDatasetTransform(TFDatasetTransform):
  """Transforms using a custom method of the input generator."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'fn', '', 'The method name to call. '
        'It must accept and return a tf.data.Dataset.')
    p.Define('kwargs', None, 'A dict with keyword arguments to pass to fn.')
    return p

  def Transform(self, dataset):
    """Returns a transformed tf.data.Dataset."""
    fn = getattr(self._input_generator, self.params.fn)
    kwargs = self.params.kwargs or {}
    return fn(dataset, **kwargs)


class RepeatableTFDatasetTransform(TFDatasetTransform):
  """A custom transform w/ repeatable dataset.

  The repeat options for the dataset are configured by the Generic input
  generator (could be TF data based) that creates/owns this datasource.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'sentinel_key', 'bucket_keys', 'The key in a dataset batch whose '
        'values are overwritten in the dummy sentinel batch. The key must '
        'already exist in each batch (NestedMap) of the dataset.')
    p.Define(
        'sentinel_value', -1, 'The value to overwrite for each element in '
        'the sentinel_key field in the sentinel dummy batch. This value '
        'should be impossible for the sentinel_key in real data.')
    return p

  def GetDataset(self):
    p = self.params
    self._repeat_steps = getattr(self._input_generator.params, 'repeat_steps',
                                 None)
    self._repeat_with_sentinel = getattr(self._input_generator.params,
                                         'repeat_with_sentinel', None)
    with py_utils.GlobalStepContext(None):
      # Hide global_step tensor from being captured by dataset function.
      ds = super().GetDataset()
    if self._repeat_steps:
      tf.logging.info('Repeating dataset every %d steps.', self._repeat_steps)
      ds = ds.take(self._repeat_steps).repeat()
    elif self._repeat_with_sentinel:
      tf.logging.info('Attaching sentinel to end of dataset and repeat.')
      # Dataset should contain batches of type NestedMap.
      sentinel_batch = ds.element_spec.Transform(
          lambda x: tf.zeros(x.shape, dtype=x.dtype))
      # Fill the dummy sentinel batch's sentinel_key tensor with sentinel_value.
      sentinel_batch[p.sentinel_key] = tf.fill(
          sentinel_batch[p.sentinel_key].shape, p.sentinel_value)
      tf.logging.info('attaching sentinel %r', sentinel_batch[p.sentinel_key])
      tf.logging.info('sentinel type %r', sentinel_batch[p.sentinel_key].dtype)
      ds = ds.concatenate(tf.data.Dataset.from_tensors(sentinel_batch)).repeat()
    return ds

  def GetNext(self):
    """Override of the root's GetNext to support checking repeat sentinel."""
    batch = super().GetNext()
    if self._repeat_with_sentinel and not self._repeat_steps:
      assert_op = tf.debugging.assert_none_equal(
          batch[self.params.sentinel_key],
          tf.constant(self.params.sentinel_value),
          summarize=1,
          message='REPEAT_SENTINEL_')
      tf.logging.info('sentinel constant dtype %r',
                      tf.constant(self.params.sentinel_value))
      with tf.control_dependencies([assert_op]):
        # This identity transform will throw tf.errors.InvalidArgumentError
        # if assert_op fails (sentinel_key takes on sentinel_value).
        batch = batch.Transform(tf.identity)
    return batch

  def Transform(self, dataset):
    return dataset


class TFDatasetFnInput(TFDatasetSource):
  """Loads a TFDataset using a function."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'load_fn', 'LoadDataset',
        'An input_generator method name to call to load data. It must accept '
        'p.args and return a tf.data.Dataset.')
    p.Define('kwargs', None,
             'A dict with keyword arguments to pass to load_fn.')
    p.Define('shuffle_buffer_size', None,
             'Number of records buffered for random shuffling.')
    return p

  def __init__(self, params):
    super().__init__(params)

    if (not self.params.shuffle_buffer_size and
        not self.cluster.require_sequential_input_order):
      raise ValueError('shuffle_buffer_size must be set.')

  def GetDataset(self):
    p = self.params
    fn = getattr(self._input_generator, p.load_fn)
    kwargs = p.kwargs or {}
    dataset = fn(**kwargs)

    if not self.cluster.require_sequential_input_order:
      dataset = dataset.shuffle(
          p.shuffle_buffer_size, reshuffle_each_iteration=True)
    if not self.do_eval:
      dataset = dataset.repeat()
    return dataset


class TFDatasetBatchBySequenceLength(TFDatasetTransform):
  """Batches examples without a leading batch dimension by length buckets.

  Examples in the dataset are assumed to be NestedMaps.

  This will create an element 'bucket_keys' in each example containing the
  result of running p.seqlen_fn on the example.

  On TPU, the final partial batch will be dropped. Note that this only applies
  to finite datasets. During training, where dataset.repeat() is usually called,
  there will be no such thing as a final partial batch.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'seqlen_fn', 'GetSequenceLength',
        'Name of input_generator function that takes an example and returns '
        'its sequence length.')
    p.Define(
        'input_shape_fn', '_InputShape',
        'Name of input_generator function that takes a tensor name and returns '
        'its shape.')
    p.Define(
        'input_padding_fn', '_InputPaddingValue',
        'Name of input_generator function that takes a tensor name and '
        'tensorspec and returns the value to pad with.')
    p.Define(
        'bucket_upper_bound', [], 'Bucketing scheme. Required to be a '
        'sorted list of integers. Examples that exceed bucket_upper_bound[-1] '
        'will be filtered out.')
    p.Define(
        'bucket_batch_limit', [], 'Desired per-split batch size per bucket. '
        'Must be the same length as bucket_upper_bound.')
    return p

  def Transform(self, dataset):
    """Batches a dataset containing NestedMaps of tensors."""
    p = self.params

    seqlen_fn = getattr(self._input_generator, p.seqlen_fn)

    def SetBucketKeys(example):
      example.bucket_keys = seqlen_fn(example)
      return example

    dataset = dataset.map(SetBucketKeys, **self._map_args)

    dataset = dataset.filter(
        lambda x: x.bucket_keys <= p.bucket_upper_bound[-1])

    dataset_structure = py_utils.NestedMap.FromNestedDict(
        tf.data.experimental.get_structure(dataset))

    input_shape_fn = getattr(self._input_generator, p.input_shape_fn)
    padded_shapes = dataset_structure.TransformWithKey(
        lambda k, _: tf.TensorShape(input_shape_fn(k)))
    input_padding_fn = getattr(self._input_generator, p.input_padding_fn)
    padding_values = dataset_structure.TransformWithKey(input_padding_fn)

    dataset_structure.VLog(0, 'dataset_structure:')
    padded_shapes.VLog(0, 'padded_shapes:')

    bucket_batch_limit = [
        batch_utils.scale_split_to_infeed(
            b, self._input_generator.params.use_per_host_infeed)
        for b in p.bucket_batch_limit
    ]
    dataset = dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(
            lambda x: x.bucket_keys,
            # Upper-bound for bucket_by_sequence_length is exclusive, so add 1
            [x + 1 for x in p.bucket_upper_bound],
            bucket_batch_limit + [1],
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            pad_to_bucket_boundary=True,
            drop_remainder=py_utils.use_tpu()))

    # Set static shapes if possible.
    if self.cluster.require_sequential_input_order:
      # When require_sequential_input_order is True the input is not repeated so
      # only one epoch is available, thus the last batch may be a smaller size.
      pass
    elif min(bucket_batch_limit) == max(bucket_batch_limit):
      b = bucket_batch_limit[0]

      def SetShape(element):
        for t in element.Flatten():
          t.set_shape((b,) + t.shape[1:])
        return element

      dataset = dataset.map(SetShape, **self._map_args)
    elif py_utils.use_tpu():
      raise ValueError('TPU requires constant batch sizes.')

    return dataset


class TFDatasetPrefetch(TFDatasetTransform):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('buffer_size', 1, 'Prefetch buffer size.')
    return p

  def Transform(self, dataset):
    return dataset.prefetch(self.params.buffer_size)


class TFDatasetMixer(TFDatasetSource):
  """Mixes multiple TFDatasetSource with provided weights."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('sub', None, 'A list of TFDatasetSource to mix.')
    p.Define('weights', None, 'A list of weights for each datasource.')
    p.Define(
        'broadcast_dataset_structures', False,
        'Attempt to make the output dataset structures compatible. '
        'For example, if one of the datasources has a key none of the '
        'other datasources have, add that key to the other datasources '
        'with tf.zeros() with a compatible shape. If a fully defined '
        'shape cannot be determined, unknown dimensions will have size 1, '
        'with the assumption that this will be consolidated later, e.g. '
        'with a call to dataset.padded_batch().')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateChildren('sub', list(self.params.sub))

  def GetDataset(self):
    p = self.params
    datasets = [sub.GetDataset() for sub in self.sub]

    def SetSourceId(i, element):
      element.source_id = i
      return element

    for i in range(len(datasets)):
      datasets[i] = datasets[i].map(
          functools.partial(SetSourceId, i), **self._map_args)

    if len(datasets) == 1:
      return datasets[0]

    if p.broadcast_dataset_structures:
      expected_structure = {}
      for dataset in datasets:
        for key, value in tf.data.experimental.get_structure(dataset).items():
          if not isinstance(value, tf.TensorSpec):
            raise ValueError(
                'broadcast_dataset_structures only supports flat structures.')
          if key not in expected_structure:
            expected_structure[key] = value
          else:
            if not expected_structure[key].is_compatible_with(value):
              raise ValueError(f'Incompatible dataset specs for key {key}: '
                               f'{expected_structure[key]} vs {value}.')
            expected_structure[key] = (
                expected_structure[key].most_specific_compatible_type(value))

      tf.logging.info('broadcast_dataset_structures: expected_structure')
      for key, value in expected_structure.items():
        tf.logging.info(f'{key}: {value}')

      def BroadcastStructure(element):
        for key, value in expected_structure.items():
          if key not in element:
            # Replace None dims with 1 and hope it will be broadcasted correctly
            # (eg. via padded_batch()) later -- it should lead to an error later
            # on in the pipeline if this assumption is incorrect.
            shape = [1 if x is None else x for x in value.shape.as_list()]
            element[key] = tf.zeros(shape, dtype=value.dtype)
        return element

      for i in range(len(datasets)):
        datasets[i] = datasets[i].map(BroadcastStructure, **self._map_args)

    return tf.data.experimental.sample_from_datasets(datasets, p.weights,
                                                     p.random_seed or None)


def GetTFDataServiceDataSet(job_name,
                            tf_data_service_address,
                            bucket_upper_bound,
                            num_hosts,
                            host_id,
                            processing_mode=None,
                            dataset=None,
                            dataset_id=None,
                            element_spec=None):
  """Register and get a dataset from TF Data Service.

  Must provide an already registered dataset_id, or a dataset to be registered.

  Args:
    job_name: name of the TF data service job (uuid str).
    tf_data_service_address: tf_data_service_address (str).
    bucket_upper_bound: The bucket upper bounds of datasource. (tensor/array)
    num_hosts: Number of hosts (e.g. depends on use_per_host_infeed). (int)
    host_id: The datasource's host_id. (int)
    processing_mode: mode for TF data service. Optional.
    dataset: a TF data Dataset to be registered. Optional.
    dataset_id: the ID of an already registered TF data Dataset. Optional.
    element_spec: the element_spec of the TF data Dataset. Optional.

  Returns:
    A TF data Dataset registered with the TF Data Service, its dataset_id, and
    element_spec.

  Raises:
    ValueError if both dataet and dataset_id are not set.
  """
  if dataset_id is None:
    if dataset is None:
      raise ValueError('Either a dataset or dataset_id must be provided.')
    dataset_id = tf.data.experimental.service.register_dataset(
        service=tf_data_service_address, dataset=dataset)
    element_spec = dataset.element_spec
  if bucket_upper_bound and num_hosts > 1:
    # Batch is bucketed by sequence length. Use round-robin order.
    consumer_index = host_id
    num_consumers = num_hosts
  else:
    consumer_index = None
    num_consumers = None
  dataset = tf.data.experimental.service.from_dataset_id(
      processing_mode=(processing_mode or
                       tf.data.experimental.service.ShardingPolicy.OFF),
      service=tf_data_service_address,
      dataset_id=dataset_id,
      element_spec=element_spec,
      job_name=job_name,
      consumer_index=consumer_index,
      num_consumers=num_consumers)
  return dataset, dataset_id, element_spec


class TFDataServiceSource(TFDatasetTransform):
  """Obtains input using remote tf.data service, potentially in batches."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('bucket_upper_bound', [], 'Bucketing scheme. Required to be'
             'a sorted list of integers.')
    p.Define('processing_mode', tf.data.experimental.service.ShardingPolicy.OFF,
             'Processing mode for TF data service.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self._job_name = str(uuid.uuid4())
    self._dataset_id = None
    self._element_spec = None

  def SetInputGenerator(self, input_generator):
    super().SetInputGenerator(input_generator)
    if self.params.bucket_upper_bound and self.num_hosts > 1:
      if self._input_generator.params.tpu_infeed_parallelism != 1:
        tf.logging.warning('Bucket-synchronized input from the tf.data service '
                           'requires setting tpu_infeed_parallelism to 1.')
        # Hacky: relies on SetInputGenerator called in input_generator.__init__,
        # before p.tpu_infeed_parallelism is used.
        self._input_generator.params.tpu_infeed_parallelism = 1
    if self._input_generator.params.use_per_host_infeed:
      tf.logging.warning(
          'When using tf.data service, it is usually better to set '
          'use_per_host_infeed=False unless the global batch is unable to fit '
          'in memory of a single machine.')

  def GetDataset(self):
    p = self.params
    if self._dataset_id is None:
      dataset = self.sub.GetDataset()

      if p.bucket_upper_bound and self.num_hosts > 1:
        # Batch is bucketed by sequence length. Group num_hosts batches
        # in each window.
        def KeyFunc(batch):
          key = tf.reduce_min(batch.bucket_keys)
          idx = tf.reduce_sum(
              tf.cast(tf.greater(key, p.bucket_upper_bound), tf.int32))
          return tf.constant(p.bucket_upper_bound, dtype=tf.int64)[idx]

        dataset = dataset.apply(
            tf.data.experimental.group_by_window(
                key_func=KeyFunc,
                reduce_func=lambda _, x: tf.data.Dataset.from_tensors(x),
                window_size=self.num_hosts))
        dataset = dataset.flat_map(lambda x: x)

      self._dataset_id = tf.data.experimental.service.register_dataset(
          service=self.cluster.tf_data_service_address, dataset=dataset)
      self._element_spec = dataset.element_spec

    dataset, _, _ = GetTFDataServiceDataSet(
        self._job_name,
        self.cluster.tf_data_service_address,
        p.bucket_upper_bound,
        self.num_hosts,
        self.host_id,
        processing_mode=p.processing_mode,
        dataset_id=self._dataset_id,
        element_spec=self._element_spec)
    return dataset

  def Reset(self, sess=None):
    # TFDataServiceSource should not be used for eval/decode, as it does not
    # have at-most-once guarantees for ShardingPolicy.OFF mode.
    raise ValueError('TFDataServiceSource does not support reset.')
