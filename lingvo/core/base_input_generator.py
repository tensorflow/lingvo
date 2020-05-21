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
"""Input generators.

There are three types of batch sizes:

* Device split batch size: Defined by Params() and is the batch size
  on each device/TPU core. BaseInputGenerator.params.batch_size and
  BaseSequenceInputGenerator.params.bucket_batch_limit specify per-split batch
  size.

* GlobalBatchSize: number of examples in a global batch.

* InfeedBatchSize: global_batch_size // num_infeed_hosts, where
  num_infeed_hosts is cluster.num_tpu_hosts if using per-host infeed with TPU,
  otherwise num_infeed_hosts is 1.

TODO(rpang): Deal with on packed_inputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import datasource
from lingvo.core import hyperparams
from lingvo.core import input_generator_helper as ig_helper
from lingvo.core import ops
from lingvo.core import py_utils
from lingvo.core import tokenizers
import six
from six.moves import map
from six.moves import range
from six.moves import zip

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import io_ops
from tensorflow.python.tpu import tpu_embedding as tpu_embedding_lib
from tensorflow.python.tpu import tpu_feed
# pylint: enable=g-direct-tensorflow-import

DEFAULT_TOKENIZER_KEY = 'default'


class BaseInputGenerator(base_layer.BaseLayer):
  """The abstract base input generator."""

  @classmethod
  def Params(cls):
    """Defaults params for input generators."""
    p = super(BaseInputGenerator, cls).Params()
    p.name = 'input'
    p.Define(
        'batch_size', 0, 'Batch size for a device split. This will be '
        'scaled to match the accelarator hardware topology.')
    p.Define(
        'num_samples', 0,
        'If non-zero, the dataset contains these many samples. '
        'For test/eval dataset, if we want the test/evel job evaluate '
        'the whole dataset, this param must be set precisely. Otherwise, '
        'this param is optional.')

    # TPU related infeed tuning.
    p.Define('use_per_host_infeed', False,
             'Whether run infeed op on each host.')
    p.Define('tpu_infeed_parallelism', 1,
             'Uses these many python threads to drive infeed concurrently.')
    p.Define('use_partitioned_infeed_queue', False, 'Use partitioned infeed')
    p.Define('num_partitions', None, 'Num partitions')

    p.Define('remote', hyperparams.Params(),
             'Params to configure remote input policy.')
    pp = p.remote
    pp.Define(
        'shardable_batch', True,
        'True if and only if this input generates simple batches whose 1st '
        'dimension of every tensor in a batch is the batch dimension, and '
        'other dimensions are always the same.')
    pp.Define(
        'max_inflights_per_target', 32, 'The maximum number of '
        'concurrent inflight remote input fetches per remote target.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseInputGenerator, self).__init__(params)
    # parameter to tell the bprop one hot for all the files.
    # TODO(ankurbpn): Initialize when using sources from mixed record yielders.
    self._bprop_onehot = tf.constant([1], dtype=tf.float32)
    # Each entry is a regular expression specifying the set of variables
    # to bprop per data source.
    self._bprop_variable_filters = ['']
    # For TPU enqueue ops, we do not use graph collections, instead, we rely
    # on this member variable. This is especially useful for
    # executor-driven multiple programs, as we need more fine-grained
    # access to drive the infeed for a specific program, rather than
    # a single global collection across the graph.
    self._tpu_infeed_op = None
    # A list of InfeedQueues.
    self._tpu_queues = []

  def CommonInputOpArgs(self):
    """Common input params."""
    return {}

  def GetBpropVariableFilters(self):
    return self._bprop_variable_filters

  def GetInputSourceOneHot(self):
    """Get the current bprop type of the input generator batch."""
    return self._bprop_onehot

  def GlobalBatchSize(self):
    """Returns the total batch size (for stats), int or dynamic int tensor."""
    p = self.params
    global_batch_size = self.InfeedBatchSize()
    cluster = self.cluster
    if p.use_per_host_infeed and cluster.num_tpu_hosts > 0:
      if not py_utils.use_tpu():
        raise ValueError('Scaling to TPU hosts without TPUs. {}'.format(
            cluster.num_tpu_hosts))
      global_batch_size *= cluster.num_tpu_hosts
    tf.logging.info('GlobalBatchSize {}'.format(global_batch_size))
    return global_batch_size

  def InfeedBatchSize(self):
    """Returns the batch size of the input batch: int or dynamic int tensor."""
    p = self.params
    cluster = self.cluster
    # Here we do not call self.GlobalBatchSize() since it can be overridden,
    # e.g., when packed inputs are used.
    batch_per_input = p.batch_size * cluster.num_splits_per_client
    # If use_per_host_infeed, each input op is only responsible
    # for generating a subset of the whole batch.
    if p.use_per_host_infeed and cluster.num_tpu_hosts > 0:
      tf.logging.info('batch_size %d cluster.num_tpu_hosts %d',
                           batch_per_input, cluster.num_tpu_hosts)
      batch_per_input //= cluster.num_tpu_hosts
    tf.logging.info('batch_per_input: %d', batch_per_input)
    return batch_per_input

  def _InputBatch(self):
    """The current input batch, not preprocessed.

    This is meant to be overridden by subclasses, but not called directly.
    Callers should use `GetPreprocessedInputBatch()`.

    Returns:
      A `.NestedMap` of input tensors. Each tensor's dim-0 must be the same
      and denotes the batch dimension.
    """
    raise NotImplementedError('Abstract method')

  def _PreprocessInputBatch(self, batch):
    """Preprocesses input batch from _InputBatch.

    Args:
      batch: A NestedMap containing input tensors in the format returned by
        _InputBatch.

    Returns:
      A NestedMap containing preprocessed inputs to feed to the model.
    """
    return batch

  def GetPreprocessedInputBatch(self):
    """Returns a NestedMap containing a preprocessed batch of inputs.

    These are the actual inputs fed to the model.

    Subclasses generally should not override this function directly. Instead,
    override _InputBatch and maybe _PreprocessInputBatch.
    """
    return self._PreprocessInputBatch(self._InputBatch())

  @property
  def tpu_number_of_shards(self):
    p = self.params
    cluster = self.cluster
    num_tpu_hosts = cluster.num_tpu_hosts
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1
    shards = (cluster.total_worker_devices //
              num_infeed_hosts) // cluster.num_devices_per_split
    return shards

  def CreateTpuEnqueueOps(self):
    """Create the host-side enqueue ops.

    This should be called in an outer non-TPU context.
    """
    assert not self._tpu_queues, ('CreateTpuEnqueueOps should only be called '
                                  'once.')
    self._tpu_queues = []
    p = self.params
    cluster = self.cluster
    num_tpu_hosts = cluster.num_tpu_hosts
    num_cores_per_host = cluster.total_worker_devices // num_tpu_hosts
    tf.logging.info(
        'CreateTpuEnqueueOps num_splits_per_client={} '
        'num_devices_per_split={} num_tpu_hosts={} use_per_host_infeed={}'
        .format(cluster.num_splits_per_client, cluster.num_devices_per_split,
                num_tpu_hosts, p.use_per_host_infeed))

    assert num_tpu_hosts > 0, ('num_tpu_hosts: %d' % num_tpu_hosts)
    if (cluster.num_devices_per_split > num_cores_per_host and
        p.use_per_host_infeed):
      tf.logging.fatal(
          'Doesn\'t support per host infeed mode when '
          'num_devices_per_split({}) > num_cores_per_host({})'.format(
              cluster.num_devices_per_split, num_cores_per_host))
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1

    shards = (cluster.total_worker_devices //
              num_infeed_hosts) // cluster.num_devices_per_split
    tf.logging.info('shards {}'.format(shards))

    input_ops_list = []
    tpu_embedding_collection = tf.get_collection(py_utils.TPU_EMBEDDING)
    tpu_embedding = (
        tpu_embedding_collection[0] if tpu_embedding_collection else None)

    if num_tpu_hosts > 1 and tpu_embedding is not None:
      if not p.use_per_host_infeed:
        tf.logging.fatal(
            'TPU Embedding must be used with per_host_infeed with multiple '
            'TPU host topologies.')

    tpu_emb_input_keys = (
        list(tpu_embedding.feature_to_config_dict.keys())
        if tpu_embedding is not None else [])
    tf.logging.info('tpu_emb_input_keys: %r', tpu_emb_input_keys)
    tf.logging.info('num_infeed_hosts: %d', num_infeed_hosts)

    for task_id in range(num_infeed_hosts):
      host_device = '/task:{}/device:CPU:0'.format(task_id)
      with tf.device(host_device):
        self._batch = self.GetPreprocessedInputBatch()
        if isinstance(self._batch, py_utils.NestedMap):
          # Hack: bucket_keys and xxx.bucket_keys are not needed on TPU.
          # Note that when MultiTaskData is used, bucket_keys will be at the
          # second level of the dictionary.
          self._batch = self._batch.FilterKeyVal(
              lambda k, _: not k.endswith('bucket_keys'))
        tf.logging.info('host_device: %s, batch: %r', host_device, self._batch)

        for k, x in self._batch.FlattenItems():
          assert x.shape.is_fully_defined(), (
              'Shape must be fully defined: %s: %s' % (k, x))
          # TODO(cwhipkey): if it's a string (or other type not supported on
          # TPU), drop it from feeding and on the other end add in an op that
          # fails if used.
        shapes = self._batch.Transform(lambda x: x.shape).Flatten()
        dtypes = self._batch.Transform(lambda x: x.dtype).Flatten()

        tf.logging.info('host_device: %s infeed shapes: %r', host_device,
                        shapes)
        tf.logging.info('host_device: %s infeed dtypes: %r', host_device,
                        dtypes)

        if p.use_partitioned_infeed_queue:
          device_assignment = py_utils.GetTpuDeviceAssignment()

          host_device = device_assignment.host_device(
              replica=0, job=tf.flags.FLAGS.tf_master)
          host_id = int(host_device.split('/task:')[1].split('/device:')[0])
          tf.logging.info('host_id: {} host_device: {}'.format(
              host_id, host_device))
          q = tpu_feed._PartitionedInfeedQueue(  # pylint: disable=protected-access
              number_of_tuple_elements=len(dtypes),
              device_assignment=device_assignment,
              host_id=host_id,
              input_partition_dims=[
                  [p.num_partitions] + [1] * (len(s) - 1) for s in shapes
              ],
              tuple_types=dtypes,
              tuple_shapes=shapes)
        else:
          q = tpu_feed.InfeedQueue(tuple_types=dtypes, tuple_shapes=shapes)
          assert shards is not None
          q.set_number_of_shards(shards)

        self._tpu_queues.append(q)

        if p.use_partitioned_infeed_queue:
          input_ops = q.generate_enqueue_ops([self._batch.Flatten()])
        elif p.use_per_host_infeed:
          # TODO(ylc/zhifengc): Add this to a policy module and test it.
          def TPUOrdinalFunction(shard_index_in_host):
            device_assignment = py_utils.GetTpuDeviceAssignment()
            if device_assignment:
              # We put both enqueue/dequeue ops at core 0 in each replica.
              replica = device_assignment.lookup_replicas(
                  task_id, 0)[shard_index_in_host]  # pylint: disable=cell-var-from-loop
              return device_assignment.tpu_ordinal(replica=replica)
            else:
              return shard_index_in_host

          input_ops = q.split_inputs_and_generate_enqueue_ops(
              self._batch.Flatten(),
              placement_function=lambda x: host_device,  # pylint: disable=cell-var-from-loop
              tpu_ordinal_function=TPUOrdinalFunction)
        else:
          input_ops = q.split_inputs_and_generate_enqueue_ops(
              self._batch.Flatten(),
              device_assignment=py_utils.GetTpuDeviceAssignment())
        input_ops_list += input_ops

    tf.logging.info('input_ops_list %s', input_ops_list)
    self._tpu_infeed_op = [tf.group(*input_ops_list)]

  def TpuDequeueBatch(self):
    """Create TPU dequeue ops.

    This should only be called within a TPU context.

    Returns:
    - A NestedMap of the input batch.
    """
    assert self._tpu_queues, 'CreateTpuEnqueueOps must be called first.'
    with tf.device(tf.tpu.core(0)):
      # Note that the dequeue_tuple op on the TPU core
      # only cares about the shape/types being dequeued
      # which is why this is hard-coded to the first Queue.
      tensors = self._tpu_queues[0].generate_dequeue_op()
    return self._batch.Pack(tensors)

  def CreateTpuEmbeddingEnqueueOps(self):
    """Creates the TpuEmbedding enqueue ops on the host.

    Note that this must be called after the instantiation of the
    monolithic TPUEmbeddingLayer.
    """
    p = self.params
    cluster = self.cluster
    num_tpu_hosts = cluster.num_tpu_hosts
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1

    tpu_embedding_collection = tf.get_collection(py_utils.TPU_EMBEDDING)
    tpu_embedding = (
        tpu_embedding_collection[0] if tpu_embedding_collection else None)

    enqueue_ops = []

    if num_tpu_hosts > 1 and tpu_embedding is not None:
      if not p.use_per_host_infeed:
        tf.logging.fatal(
            'TPU Embedding must be used with per_host_infeed with multiple '
            'TPU host topologies.')
    tpu_emb_input_keys = (
        list(tpu_embedding.feature_to_config_dict.keys())
        if tpu_embedding is not None else [])
    tf.logging.info('tpu_emb_input_keys: %r', tpu_emb_input_keys)
    if not tpu_embedding:
      return

    for task_id in range(num_infeed_hosts):
      host_device = '/task:{}/device:CPU:0'.format(task_id)
      with tf.device(host_device):
        if isinstance(self._batch, py_utils.NestedMap):
          # Hack: bucket_keys and xxx.bucket_keys are not needed on TPU.
          # Note that when MultiTaskData is used, bucket_keys will be at the
          # second level of the dictionary.
          self._batch = self._batch.FilterKeyVal(
              lambda k, _: not k.endswith('bucket_keys'))
        tf.logging.info('host_device: %s, batch: %r', host_device, self._batch)

        enqueue_dict_per_core = [
            {} for _ in range(tpu_embedding.num_cores_per_host)
        ]
        num_cores_per_host = tpu_embedding.num_cores_per_host
        for key in tpu_emb_input_keys:
          feat = self._batch[key]
          tpu_emb_feat_splitted = tf.split(feat, num_cores_per_host)
          for core, split in enumerate(tpu_emb_feat_splitted):
            # Dense to sparse. Note the assumption of a padding id.
            sample_indices = tf.where(tf.not_equal(split, -1))
            embedding_indices = tf.gather_nd(split, sample_indices)
            enqueue_data = tpu_embedding_lib.EnqueueData(
                embedding_indices, sample_indices)
            enqueue_dict_per_core[core][key] = enqueue_data
        enqueue_ops += tpu_embedding.generate_enqueue_ops(enqueue_dict_per_core)
    self._tpu_infeed_op.append(tf.group(*enqueue_ops))

  @property
  def tpu_infeed_op(self):
    if self._tpu_infeed_op is not None:
      return self._tpu_infeed_op
    else:
      raise ValueError('TPU infeed op not set. Call CreateTpuEnqueueOps first.')

  def SplitInputBatch(self, num_splits):
    """Splits the current InputBatch into num_splits ways.

    Args:
      num_splits: The number of splits.

    Returns:
      A list of `.NestedMap`. Each `.NestedMap` represents the input
      tensors in one split.
    """
    assert num_splits >= 1

    batch = self.GetPreprocessedInputBatch()
    if num_splits == 1:
      # Special case. No split is needed.
      return [batch]

    assert not py_utils.use_tpu()
    field_split = ig_helper.SplitTensors(batch.Flatten(), num_splits)
    num_fields = len(field_split)
    ret = []
    for j in range(num_splits):
      split_flatten = [field_split[i][j] for i in range(num_fields)]
      split = batch.Pack(split_flatten)
      ret += [split]
    return ret


class BaseInputGeneratorFromFiles(BaseInputGenerator):
  """Base class for input generators that reads from files."""

  @classmethod
  def Params(cls):
    """Defaults params for input generators."""
    p = super(BaseInputGeneratorFromFiles, cls).Params()
    p.Define(
        # NOTE: file_pattern is deprecated.  New params should use
        # file_datasource instead.
        # TODO(b/139345706) remove file_pattern parameter
        'file_pattern',
        '',
        'A single file pattern string, a list of file pattern strings or a list'
        ' of <file_pattern, weight> pairs or a list of  <file_pattern, weight, '
        'bprop_variable_filter> tuples. Some of the cases may not be supported '
        'Depending on the value of use_within_batch_mixing and use_chaining.'
        'In the later 2 cases, probablistic samples are from the inputs '
        'proportional to their weights. Typically, values are binary '
        'protocol buffers containing train/eval samples. Keys are not used.')
    p.Define(
        'file_datasource', None, 'A DataSource describing the file sources '
        'including any weights and bprop_variable_filters required.')
    p.Define('file_random_seed', 301,
             'Random seed for shuffling the input data.')
    p.Define(
        'file_buffer_size', 10000,
        'How many records are buffered for random shuffling. This param '
        'affects how much RAM a train/test job needs. E.g., if an average '
        'record is about 500KB, the buffer needs 5GB ram.')
    p.Define(
        'file_buffer_size_in_seconds', 0,
        'If non-zero, keep enough records in the buffer to handle N seconds '
        'worth of demand. E.g., if the training job is reading 1000 records '
        'per second and this parameter is set to 10, the buffer is resized '
        'to contain 10000 records. This parameter is useful when reading from '
        'many data sources at different speeds, as it automatically tunes the '
        'size of buffers to fit demand. The file_buffer_size parameter is an '
        'upper bound to the buffer size.')
    p.Define('file_parallelism', 16, 'How many files to read concurrently.')
    p.Define(
        'bucket_adjust_every_n', 0, 'If non-zero, optimize the values of '
        'bucket_upper_bound except the last one after every N records '
        'based on the current input length distribution.')
    p.Define(
        'flush_every_n', 0, 'If non-zero, flushes all batches buffered '
        'so far every these many records are yielded.')
    p.Define('num_batcher_threads', 1, 'Number of threads to use for input '
             'record batcher.')
    p.Define(
        'require_sequential_order', False,
        'If true, the input op is required to process the file glob as '
        'well as the contents of each file in a deterministic sequential order.'
        ' This is intended for unit tests. Setting this automatically disables '
        'file_random_seed, file_buffer_size, file_parallelism, '
        'num_batcher_threads, and requires a single file_pattern.')
    p.Define(
        'repeat_count', -1,
        'Number of repetitions of a dataset before throwing OutOfRange error '
        'when using require_sequential_order. Must only be set if '
        'require_sequential_order is True.')
    # TODO(b/139345706) when file_pattern is deleted use_within_batch_mixing
    # will be specified by passing a WithinBatchMixingDataSource to
    # p.file_datasource and this param should be deleted as well.
    p.Define(
        'use_within_batch_mixing', False, 'Whether to mix records from '
        'different input sources within batch or across batches (the '
        'default option). This option only takes effect when file_pattern'
        ' is a list of file patterns with weights.')
    # TODO(b/139345706) when file_pattern is deleted use_chaining
    # will be specified by passing a ChainingDataSource to
    # p.file_datasource and this param should be deleted as well.
    p.Define(
        'use_chaining', False, 'Whether to output records from '
        'different input sources one after another, i.e., first all records '
        'from a first input source, then all records from a second one, etc. '
        'use_chaining does not guarantee that records from subsequent input '
        'sources are placed in separate input batches. '
        'This option only takes effect when file_pattern is a list of file '
        'patterns.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseInputGeneratorFromFiles, self).__init__(params)
    p = self.params
    if p.use_per_host_infeed and p.file_random_seed != 0:
      raise ValueError('file_random_seed needs to be 0 when '
                       'use_per_host_infeed == True.')

    assert not (p.file_pattern and p.file_datasource
               ), 'Only one of file_pattern and data_source can be specified'

    # TODO(b/139345706) remove support for file_pattern
    if not p.file_datasource:
      if isinstance(p.file_pattern, six.string_types):
        p.file_datasource = datasource.SimpleDataSource.Params().Set(
            file_pattern=p.file_pattern)
      elif isinstance(p.file_pattern, (list, tuple)):
        if all([isinstance(x, six.string_types) for x in p.file_pattern]):
          # While this violates the documentation and intended use, there are
          # subclasses that have used a tuple of strings, rather than a list of
          # string, weight tuples.  Rather than treating lists and tuples
          # differently, support both here until p.file_pattern is removed.
          p.file_datasource = datasource.SimpleDataSource.Params().Set(
              file_pattern=','.join(p.file_pattern))
        elif p.use_within_batch_mixing:
          assert not p.use_chaining, "Can't both use chaining and mixing"

          if max(list(map(len, p.file_pattern))) >= 3:
            # Within batch mixing doesn't work with backprop filters, i.e. when
            # file_pattern param contains a list of
            # <file_pattern, weight, [bprop_variable_filter]> tuples.
            raise ValueError('Expected a list of pairs, got %s' %
                             (p.file_pattern,))

          file_patterns, weights = (list(x) for x in zip(*p.file_pattern))

          p.file_datasource = datasource.WithinBatchMixingDataSource.Params(
          ).Set(
              file_patterns=file_patterns, weights=weights)
        elif p.use_chaining:
          p.file_datasource = datasource.ChainingDataSource.Params().Set(
              file_patterns=p.file_pattern)
        else:
          # Otherwise fall back to MixByWeight-based approach.
          file_patterns = []
          weights = []
          bprop_variable_filters = []
          for input_entry in p.file_pattern:
            if isinstance(input_entry, six.string_types):
              raise ValueError(
                  'Should explicitly specify weights, got string: %s' %
                  (input_entry,))
            file_pattern, weight = input_entry[:2]
            file_patterns.append(file_pattern)
            weights.append(weight)
            bprop_variable_filter = input_entry[2] if len(
                input_entry) > 2 else ''
            bprop_variable_filters.append(bprop_variable_filter)
          p.file_datasource = datasource.CrossBatchMixingDataSource.Params(
          ).Set(
              file_patterns=file_patterns,
              weights=weights,
              bprop_variable_filters=bprop_variable_filters)
      else:
        raise ValueError('Cannot parse p.file_pattern into a datasource.')

    self.CreateChild('datasource', p.file_datasource)

  def CommonInputOpArgs(self):
    """Common input params."""
    p = self.params
    args = super(BaseInputGeneratorFromFiles, self).CommonInputOpArgs()
    if p.file_datasource and issubclass(p.file_datasource.cls,
                                        datasource.ChainingDataSource):
      # If a user provides a ChainingDataSource make sure that the
      # param is set correctly when passed to the InputOp
      p.use_chaining = True
    args.update({
        'file_random_seed': p.file_random_seed,
        'file_buffer_size': p.file_buffer_size,
        'file_parallelism': p.file_parallelism,
        'file_buffer_size_in_seconds': p.file_buffer_size_in_seconds,
        'bucket_adjust_every_n': p.bucket_adjust_every_n,
        'flush_every_n': p.flush_every_n,
        'num_threads': p.num_batcher_threads,
        'require_sequential_order': p.require_sequential_order,
        'repeat_count': p.repeat_count,
        'use_chaining': p.use_chaining,
    })
    args.update(self._InputOpBucketingArgs())
    return args

  def _InputOpBucketingArgs(self):
    return {
        'bucket_upper_bound': [1000000000],
        'bucket_batch_limit': [self.InfeedBatchSize()],
    }

  # TODO(b/139345706): After p.file_pattern is deleted, the following functions
  # _DataSourceFromFilePattern, _BuildDataSourceWithMetadata, _BuildDataSource
  # can be deleted and functionality moved to using the DataSource directly.
  def _DataSourceFromFilePattern(self, file_pattern, input_source_weights=None):
    """Read and return input batch from a string file_pattern.

    Args:
      file_pattern: A string file pattern.
      input_source_weights: A list of float input source weights to control
        input example mix in the batch. The records will be sampled from inputs
        proportionally to these weights. Defaults to None which should be
        treated as an empty list.

    Returns:
      A tuple of tf.Tensors where the tensors which contain the input data and
      have major dimension same as size of input batch.
    """
    raise NotImplementedError()

  def _BuildDataSourceWithMetadata(self):
    """Read and return input batch from `p.file_pattern`.

    `p.file_pattern` may be a string file_pattern or a
    list of (file_pattern, weight, [bprop_variable_filter]) tuples.
    bprop_variable_filter is optional. When bprop_variable_filter is used,
    batches will always contain the examples from the same source. Otherwise,
    examples from different sources may be mixed together.

    Returns:
      A `.NestedMap` containing

      - data: a tuple of tf.Tensor or `.NestedMap` of
        tf.Tensor same as `self._DataSourceFromFilePattern()`
      - source_selected: a tensor of size [batch_size, number of data sources]
        or None.
      - selected_bprop: a tensor of size [number of data sources] or None.
      - bprop_variable_filters: a list of bprop_variable filters for each source
        or None.

    Raises:
      ValueError: If file_datasource is not set
    """
    p = self.params
    if not p.file_datasource and p.file_pattern:
      # This is a workaround for subclasses which have defined
      # their own data source-like functionality.
      tf.logging.info(
          'Creating data source-like output from class %s using '
          'file_pattern %s', self, p.file_pattern)
      ret = py_utils.NestedMap()
      ret.data = self._DataSourceFromFilePattern(p.file_pattern)
    else:
      tf.logging.info(
          'Building data source %s with params %s and '
          'file_pattern %s', self.datasource, self.datasource.params,
          p.file_pattern)
      ret = self.datasource.BuildDataSource(self._DataSourceFromFilePattern)
    if 'selected_bprop' in ret:
      self._bprop_onehot = ret.selected_bprop
    if 'bprop_variable_filters' in ret:
      self._bprop_variable_filters = ret.bprop_variable_filters
    if 'source_selected' not in ret:
      ret.source_selected = None
    return ret

  def _BuildDataSource(self):
    """Read and return input batch from `p.file_pattern`.

    Same as _BuildDataSourceWithMetadata but does not return any metadata.

    Returns:
      A tuple of tf.Tensor or `.NestedMap` of tf.Tensor same as
      `self._DataSourceFromFilePattern()`.

    Raises:
      ValueError: If unknown token type.
    """
    return self._BuildDataSourceWithMetadata()['data']


class BaseSequenceInputGenerator(BaseInputGeneratorFromFiles):
  """The basic sequence input generator."""

  @classmethod
  def Params(cls):
    """Defaults params for sequence input generators."""
    p = super(BaseSequenceInputGenerator, cls).Params()
    p.Delete('batch_size')
    p.remote.shardable_batch = False

    # How input should be bucketized.
    p.Define(
        'bucket_upper_bound', [2560], 'Bucketing scheme. Required to be'
        'a sorted list of integers. Examples that are longer than all bucket'
        'upper bounds are skipped.')
    p.Define(
        'bucket_batch_limit', [8],
        'Desired per-split batch size per bucket. Scaled in '
        'infeed_bucket_batch_size to the infeed size.'
        'Must be the same length as bucket_upper_bound.')
    p.Define('source_max_length', None,
             'The maximum length of the source sequence.')
    p.Define('target_max_length', 300,
             'The maximum length of the target sequence.')
    p.Define('pad_to_max_seq_length', False,
             'If True, input tensors will be padded to max_length.')
    p.Define('tokenizer', tokenizers.AsciiTokenizer.Params(),
             'Tokenizer params.')
    p.Define(
        'tokenizer_dict', {},
        'If multiple tokenizers are required, they can be accessed through '
        'this dict via a key.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseSequenceInputGenerator, self).__init__(params)

    p = self.params

    if p.tokenizer:
      assert DEFAULT_TOKENIZER_KEY not in p.tokenizer_dict
      p.tokenizer_dict[DEFAULT_TOKENIZER_KEY] = p.tokenizer

    self.tokenizer_dict = {}
    for k, p in six.iteritems(p.tokenizer_dict):
      if p:
        name = '_tokenizer_' + k
        self.CreateChild(name, p)
        self.tokenizer_dict[k] = self.children[name]
      else:
        self.tokenizer_dict[k] = None

    if DEFAULT_TOKENIZER_KEY in self.tokenizer_dict:
      self.tokenizer = self.tokenizer_dict[DEFAULT_TOKENIZER_KEY]

  @property  # Adjust batch size according to the cluster spec.
  def infeed_bucket_batch_limit(self):
    """Returns the bucket batch limit for one infeed host."""
    p = self.params
    cluster = self.cluster
    infeed_bucket_batch_limit = [
        b * cluster.num_splits_per_client for b in p.bucket_batch_limit
    ]
    if p.use_per_host_infeed and cluster.num_tpu_hosts > 0:
      if not py_utils.use_tpu():
        raise ValueError('Scaling to TPU hosts without TPUs. {}'.format(
            cluster.num_tpu_hosts))
      tf.logging.info(
          'scaling infeed_bucket_batch_limit num_tpu_hosts={}'.format(
              cluster.num_tpu_hosts))
      infeed_bucket_batch_limit = [
          x // cluster.num_tpu_hosts for x in infeed_bucket_batch_limit
      ]
    tf.logging.info(
        'infeed_bucket_batch_limit={} num_splits_per_client={} bucket_batch_limit={}'
        .format(infeed_bucket_batch_limit, cluster.num_splits_per_client,
                p.bucket_batch_limit))
    return infeed_bucket_batch_limit

  def InfeedBatchSize(self):
    """Returns the batch size of one infeed pipeline.

    Override in subclass to provide dynamically shaped infeed batch size.

    If use_per_host_infeed is False then there is only one infeed pipeline and
    then the GlobalBatchSize() and the InfeedBatchSize() is the same.
    """
    buckets = self.infeed_bucket_batch_limit
    if any(x != buckets[0] for x in buckets):
      tf.logging.warning(
          'Using max bucket batch limit but not all limits are '
          'the same {}'.format(buckets))
    infeed_size = max(buckets)
    tf.logging.info('InfeedBatchSize: %d', infeed_size)
    return infeed_size

  def _InputOpBucketingArgs(self):
    p = self.params
    bucket_batch_limit = self.infeed_bucket_batch_limit
    tf.logging.info('infeed_bucket_batch_limit %r', bucket_batch_limit)
    return {
        'bucket_upper_bound': p.bucket_upper_bound,
        'bucket_batch_limit': bucket_batch_limit,
    }

  def StringsToIds(self,
                   strs,
                   is_source=False,
                   external_max_length=None,
                   external_append_eos=None,
                   key=None,
                   languages=None):
    """Tokenize strs into vocab ids.

    Args:
      strs: A vector of strings.
      is_source: A bool to indicate whether to use `source_max_length` to pad
        'strs'.
      external_max_length: An int providing the max_length for strs.
      external_append_eos: Bool or None. If None, will be ignored and
        `params.append_eos` will be used. If bool, will determine if an eos
        symbol will be added to tokens.
      key: A string key in case the model has multiple tokenizers.
      languages: A vector of str with the same length as `strs`.

    Returns:
      A tuple (ids, labels, paddings) with the same shape [batch, maxlen].

      - ids[i, j] is the input token id of i-th sample for j-th step.
      - labels[i, j] is the target token id of i-th sample for j-th step.
      - paddings[i, j] is 1 iff i-th sample's j-th step is padded.

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params

    if external_max_length is not None:
      maxlen = external_max_length
    elif is_source:
      maxlen = p.source_max_length
    else:
      maxlen = p.target_max_length

    key = key or DEFAULT_TOKENIZER_KEY
    return self.tokenizer_dict[key].StringsToIds(
        strs, maxlen, external_append_eos, languages=languages)

  def IdsToStrings(self, ids, lens, key=None):
    """Converts ids back to strings.

    Args:
      ids: A matrix of shape [batch, seqlen]. ids[i, :] is the i-th sample's
        ids.
      lens: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th sample. Only the first lens[i] tokens in ids[i, :] are valid tokens
        for the i-th sequence.
      key: A string key in case the model has multiple tokenizers.

    Returns:
      sequences - A vector of shape [batch]. The converted string sequence.

    Raises:
      ValueError: If unknown token type.
    """
    key = key or DEFAULT_TOKENIZER_KEY
    return self.tokenizer_dict[key].IdsToStrings(ids, lens)

  def Cast(self, v):
    """Cast tensor dtype to fprop_dtype."""
    if not v.dtype.is_floating:
      return v
    return tf.cast(v, py_utils.FPropDtype(self.params))


class BaseTinyDatasetInput(BaseInputGenerator):
  """Input generator for tiny dataset which are stored in tf checkpoint.

      | Input batch (b: batch size, h: height, w: width, d: depth):
      |   raw: Samples. [b, h, w, d].
      |   data: Preprocessed samples. [b, h, w, d].
      |   label: Labels. [b].
      |   weight: [b]. weight[i] is 1.0 if i-th sample is considered to
      |     be a real example. Otherwise, weight[i] is 0.0.
  """

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super(BaseTinyDatasetInput, cls).Params()
    p.Define('ckpt', None, 'A TensorFlow checkpoint.')
    p.Define('data', 'x_train', 'The tensor name in the ckpt.')
    p.Define('data_dtype', tf.uint8, 'The tensor dtype in the ckpt.')
    p.Define(
        'data_shape', (0, 0, 0), 'A tuple of ints. E.g., a tiny image '
        'has the shape (height, weight, depth).')
    p.Define('label', 'y_train', 'The tensor name in the ckpt.')
    p.Define('label_dtype', tf.uint8, 'The tensor dtype in the ckpt.')
    p.Define('repeat', True, 'If true, goes through the dataset repeatedly.')
    p.use_per_host_infeed = True
    return p

  def _InputBatch(self):
    p = self.params

    @tf.function
    def ReadData():
      x, y = io_ops.restore_v2(p.ckpt, [p.data, p.label], [''] * 2,
                               [p.data_dtype, p.label_dtype])
      # Always convert to float32.
      return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # Loads data and label into memory and keep it around.
    data, label = ops.cached_call(
        f=ReadData.get_concrete_function(), T=[tf.float32, tf.float32])
    b, shape = self.InfeedBatchSize(), list(p.data_shape)
    data = tf.reshape(data, [-1] + shape)
    label = tf.reshape(label, [-1])
    label = py_utils.HasShape(label, [tf.shape(data)[0]])
    sample_ids = ops.random_permutation_sequence(
        num=p.num_samples,
        batch=b,
        repeat=p.repeat,
        seed=p.random_seed if p.random_seed else 0)
    n = tf.shape(sample_ids)[0]
    raw = py_utils.PadOrTrimTo(tf.gather(data, sample_ids), [b] + shape)
    ret = py_utils.NestedMap(
        raw=raw,
        data=self._Preprocess(raw),
        label=py_utils.PadOrTrimTo(tf.gather(label, sample_ids), [b]),
        weight=py_utils.PadOrTrimTo(tf.ones([n], dtype=tf.float32), [b]))
    if not py_utils.use_tpu():
      ret['sample_ids'] = sample_ids
    return ret

  def _Preprocess(self, raw):
    return raw


class BaseDataExampleInputGenerator(BaseInputGenerator):
  """Base class for input generators that read Feature protos via tf.data."""

  @classmethod
  def Params(cls):
    p = super(BaseDataExampleInputGenerator, cls).Params()
    p.Define('input_files', None, 'Delimited glob of input files.')
    p.Define(
        'dataset_type', None,
        'A dataset class constructor such as tf.data.TFRecordDataset. '
        'The class constructor must take a list of filenames and produce an '
        'object that extends tf.data.Dataset.')
    p.Define('randomize_order', True, 'Whether to randomize the order.')
    p.Define('parallel_readers', 1, 'Number of parallel reader threads.')
    p.Define('num_examples', -1, 'Number of examples (-1 for unlimited).')
    p.Define(
        'num_epochs', -1,
        'Number of passes through the data to make (-1 for unlimited).'
        '`tf.errors.OutOfRangeError` is thrown after the limit is reached.')
    p.Define('randomize_shuffle_size', 500,
             'Size of the random shuffle buffer.')
    p.remote.shardable_batch = False
    return p

  def __init__(self, params):
    super(BaseDataExampleInputGenerator, self).__init__(params)
    p = params
    assert p.input_files, (
        'input_files is required for a tf.data example input generator')
    assert p.dataset_type, (
        'dataset_type is required for a tf.data example input generator')

  def GetFeatureSpec(self):
    """Subclasses must implement and return a feature spec.

    Returns:
      NestedMap of features compatible with tf.io.parse_example. Default
      implementation returns an empty dict.
    """
    return {}

  def GetPreprocessedInputBatch(self):
    p = self.params

    def ParseAndProcess(*cols):
      """Parses a Tensorflow example into features."""
      # Assume either one or two column input. If one, then the record is
      # assumed to be that column. If 2, then it is assumed to be a KV store
      # and the record is the second column.
      assert len(cols) in [
          1, 2
      ], ('BaseExampleInputGenerator supports one or two column input')
      record = cols[-1]
      feature_spec = self.GetFeatureSpec()
      features = py_utils.NestedMap(tf.io.parse_example(record, feature_spec))
      return self._PreprocessInputBatch(features)

    dataset_factory = p.dataset_type
    dataset = (
        tf.data.Dataset.list_files(
            p.input_files, shuffle=bool(p.randomize_order)).apply(
                tf.data.experimental.parallel_interleave(
                    dataset_factory,
                    cycle_length=p.parallel_readers,
                    sloppy=p.randomize_order)))

    if p.randomize_order:
      dataset = dataset.shuffle(p.randomize_shuffle_size)
    dataset = dataset.take(p.num_examples)
    dataset = dataset.repeat(p.num_epochs)
    dataset = dataset.batch(self.InfeedBatchSize(), drop_remainder=True)
    dataset = dataset.map(
        ParseAndProcess, num_parallel_calls=p.parallel_readers)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_one_shot_iterator()
    input_batch = iterator.get_next()
    return input_batch
