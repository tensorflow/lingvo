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

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import batch_utils
from lingvo.core import cluster
from lingvo.core import py_utils

import tensorflow_datasets as tfds


class DataSource(base_layer.BaseLayer):
  """A base class for data sources."""

  @classmethod
  def Params(cls):
    return super().Params().Set(name='datasource')

  def __init__(self, params):
    super().__init__(params)
    self.SetVariableFree()
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

  def Initialize(self, sess):
    for child in self._children_list:
      if isinstance(child, DataSource):
        child.Initialize(sess)

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

    if p.weights:
      # Within-batch mixing.
      batch = self._input_generator._DataSourceFromFilePattern(  # pylint: disable=protected-access
          file_patterns,
          input_source_weights=p.weights)
    else:
      # Default.
      batch = self._input_generator._DataSourceFromFilePattern(  # pylint: disable=protected-access
          file_patterns)

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
    shape = py_utils.GetShape(py_utils.Flatten(data_source)[0])
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

  def GetDataset(self):
    """Override to return a tf.data.Dataset containing a NestedMap."""
    raise NotImplementedError()

  def _InitIterator(self):
    if self.host_id in self._dataset:
      return

    with py_utils.GlobalStepContext(None):
      # Hide global_step tensor from being captured by dataset function.
      ds = self.GetDataset()
    ds.options().experimental_deterministic = False
    self._dataset[self.host_id] = ds
    if tf.executing_eagerly():
      it = iter(ds)
    else:
      it = tf.data.make_initializable_iterator(ds)
    self._iterator[self.host_id] = it

  def Initialize(self, sess):
    self.Reset(sess)
    super().Initialize(sess)

  def Reset(self, sess):
    if self._dataset:
      if tf.executing_eagerly():
        self._iterator = {key: iter(ds) for key, ds in self._dataset.items()}
      else:
        sess.run([it.initializer for it in self._iterator.values()])

  def GetNext(self):
    """Returns the next element from the dataset."""
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
        lambda _: self.sub.GetNext())


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
    return p

  def Transform(self, dataset):
    """Returns a transformed tf.data.Dataset."""
    return getattr(self._input_generator, self.params.fn)(dataset)


class TFDatasetFnInput(TFDatasetSource):
  """Loads a TFDataset using a function."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'load_fn', 'LoadDataset',
        'An input_generator method name to call to load data. It must accept '
        'p.args and return a tf.data.Dataset.')
    p.Define('args', None,
             'Arguments to pass to load_fn. If a list, it will be expanded.')
    p.Define('shuffle_buffer_size', None,
             'Number of records buffered for random shuffling.')
    p.Define(
        'require_sequential_order', False,
        'Whether elements need to be produced in sequential order. '
        'Disables randomization.')
    return p

  def __init__(self, params):
    super().__init__(params)

    if (not self.params.shuffle_buffer_size and
        not self.params.require_sequential_order):
      raise ValueError('shuffle_buffer_size must be set.')

  def GetDataset(self):
    p = self.params
    fn = getattr(self._input_generator, p.load_fn)
    if p.args is None:
      dataset = fn()
    elif isinstance(p.args, list):
      dataset = fn(*p.args)
    else:
      dataset = fn(p.args)

    require_sequential_order = p.require_sequential_order or self.do_eval
    if not require_sequential_order:
      dataset = dataset.shuffle(
          p.shuffle_buffer_size, reshuffle_each_iteration=True)
      dataset = dataset.repeat()
    return dataset


class TFDSInput(TFDatasetSource):
  """Load tfds datasets."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dataset', None, 'The tfds dataset to load.')
    p.Define(
        'split', None,
        'The split to load. See https://www.tensorflow.org/datasets/splits.')
    p.Define(
        'load_fn', 'LoadTFDSDataset',
        'An input_generator method name to call to load data. It must accept '
        '(info, features_dict) and return a NestedMap.')
    p.Define('shuffle_buffer_size', 10000,
             'Number of records buffered for random shuffling.')
    p.Define(
        'require_sequential_order', False,
        'Whether elements need to be produced in sequential order. '
        'Disables randomization.')
    return p

  def GetDataset(self):
    p = self.params
    if not p.dataset or not p.split:
      raise ValueError('A dataset and split must be specified.')

    require_sequential_order = p.require_sequential_order or self.do_eval
    dataset, info = tfds.load(
        p.dataset,
        split=p.split,
        download=True,  # download dataset locally.
        shuffle_files=not require_sequential_order,
        with_info=True)

    dataset = dataset.map(
        functools.partial(getattr(self._input_generator, p.load_fn), info),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=require_sequential_order)

    if not require_sequential_order:
      dataset = dataset.shuffle(p.shuffle_buffer_size)
      dataset = dataset.repeat()
    return dataset


class TFDatasetBatchBySequenceLength(TFDatasetTransform):
  """Batches examples without a leading batch dimension by length buckets."""

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
    p.Define('bucket_upper_bound', [], 'Bucketing scheme. Required to be'
             'a sorted list of integers.')
    p.Define(
        'bucket_batch_limit', [], 'Desired per-split batch size per bucket. '
        'Must be the same length as bucket_upper_bound.')
    p.Define(
        'require_sequential_order', False,
        'Whether elements need to be produced in sequential order. '
        'Disables randomization.')
    return p

  def Transform(self, dataset):
    """Batches a dataset containing NestedMaps of tensors."""
    p = self.params

    require_sequential_order = p.require_sequential_order or self.do_eval
    seqlen_fn = getattr(self._input_generator, p.seqlen_fn)

    def SetBucketKeys(example):
      example.bucket_keys = seqlen_fn(example)
      return example

    dataset = dataset.map(
        SetBucketKeys,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=require_sequential_order)

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
            # TODO(jeffreyzhao): There is a off-by-one bug with the upper bound
            # boundary check, so add 2 instead. Remove when fixed.
            [x + 2 for x in p.bucket_upper_bound],
            bucket_batch_limit + [1],
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            pad_to_bucket_boundary=True,
            drop_remainder=py_utils.use_tpu()))

    if py_utils.use_tpu():
      # Set static shapes for TPU.
      if min(bucket_batch_limit) != max(bucket_batch_limit):
        raise ValueError('TPU requires constant batch sizes.')
      else:
        b = bucket_batch_limit[0]

        def SetShape(element):
          for t in element.Flatten():
            t.set_shape((b,) + t.shape[1:])
          return element

        dataset = dataset.map(
            SetShape,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=require_sequential_order)

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
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateChildren('sub', self.params.sub)

  def GetDataset(self):
    p = self.params
    datasets = [sub.GetDataset() for sub in self.sub]

    def SetSourceId(i, element):
      element.source_id = i
      return element

    for i in range(len(datasets)):
      datasets[i] = datasets[i].map(
          functools.partial(SetSourceId, i),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return tf.data.experimental.sample_from_datasets(datasets, p.weights,
                                                     p.random_seed or None)


class TFDataServiceSource(TFDatasetTransform):
  """Obtains input using remote tf.data service, potentially in batches."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('bucket_upper_bound', [], 'Bucketing scheme. Required to be'
             'a sorted list of integers.')
    return p

  def Transform(self, dataset):
    p = self.params
    if p.bucket_upper_bound and self.num_hosts > 1:
      # Batch is bucketed by sequence length. split into num_hosts batches
      # and pull from the service in round-robin style.
      if self._input_generator.params.tpu_infeed_parallelism != 1:
        raise ValueError('Bucket-synchronized input from the tf.data service '
                         'requires tpu_infeed_parallelism == 1.')

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
      dataset = dataset.apply(
          tf.data.experimental.service.distribute(
              job_name=cluster.GetProcessUUID(),
              processing_mode='parallel_epochs',
              service=self.cluster.tf_data_service_address,
              consumer_index=self.host_id,
              num_consumers=self.num_hosts))
    else:
      dataset = dataset.apply(
          tf.data.experimental.service.distribute(
              job_name=cluster.GetProcessUUID(),
              processing_mode='parallel_epochs',
              service=self.cluster.tf_data_service_address))

    return dataset
