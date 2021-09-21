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

import inspect

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import batch_utils
from lingvo.core import cluster
from lingvo.core import cluster_factory
from lingvo.core import datasource
from lingvo.core import hyperparams
from lingvo.core import input_generator_helper as ig_helper
from lingvo.core import inspect_utils
from lingvo.core import ops
from lingvo.core import py_utils
from lingvo.core import tokenizers
from lingvo.core import tpu_embedding_layers

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import io_ops
from tensorflow.python.tpu import tpu_embedding as tpu_embedding_lib
from tensorflow.python.tpu import tpu_feed
# pylint: enable=g-direct-tensorflow-import

DEFAULT_TOKENIZER_KEY = 'default'
INPUT_DATA_STATS_SUMMARIES_COLLECTION = 'INPUT_DATA_STATS_SUMMARIES'


class BaseInputGenerator(base_layer.BaseLayer):
  """The abstract base input generator."""

  @classmethod
  def DefineInfeedParams(cls, p):
    # TPU related infeed tuning.
    # Supported use cases:
    #
    # Data parallelism (num_partitions=None)
    #  - single host (use_per_host_infeed=False, tpu_infeed_parallelism=1))
    #  - multi host (use_per_host_infeed=False, tpu_infeed_parallelism>1)
    #  - per host (use_per_host_infeed=True)
    #    - unsharded inputs (_InputBatch returns a single NestedMap)
    #    - sharded inputs (_InputBatch returns a list containing
    #      tpu_number_of_shards NestedMaps)
    # Model parallelism (num_partitions>1 where)
    #  - non-partitioned infeed (use_partitioned_infeed_queue=False):
    #    - Only first partition gets infeed (e.g. manual partition)
    #      - single host (use_per_host_infeed=False)
    #      - per host (use_per_host_infeed=True)
    #    - All partitions gets data parallel infeed (e.g. MoE)
    #      - single host not supported
    #      - per host (use_per_host_infeed=True, use_per_core_infeed=True)
    #        num_partitions should be set to number of partitions per replica
    #  - partitioned infeed (use_partitioned_infeed_queue=True)
    #    - single host (use_per_host_infeed=False)
    #    - per host (use_per_host_infeed=True)
    #        num_partitions should be set to number of partitions per replica
    #        and all partitions should exist on a single host
    p.Define('use_per_host_infeed', False,
             'Whether run infeed op on each host.')
    p.Define('use_per_core_infeed', False,
             'Whether to shard the infeed per TPU core instead of per replica')
    p.Define('tpu_infeed_parallelism', 1,
             'Uses these many python threads to drive infeed concurrently.')
    p.Define('use_partitioned_infeed_queue', False, 'Use partitioned infeed')
    p.Define(
        'num_partitions', None,
        'Number of partitions to split the model graph into. Used with '
        'model parallelism. When >1, it specifies the number of devices '
        'used to place one replica of the model graph nodes.')

  @classmethod
  def Params(cls):
    """Defaults params for input generators."""
    p = super().Params()
    p.name = 'input'
    p.Define(
        'file_datasource', None,
        'The DataSource that produces input batches for this input generator.')
    p.Define(
        'batch_size', 0, 'Batch size for a device split. This will be '
        'scaled to match the accelarator hardware topology.')
    p.Define(
        'num_samples', 0,
        'If non-zero, the dataset contains these many samples. '
        'For test/eval dataset, if we want the test/evel job evaluate '
        'the whole dataset, this param must be set precisely. Otherwise, '
        'this param is optional.')
    p.Define('resettable', False,
             'If True, the input generator must implement Reset().')
    # For an input generator to support samples_per_summary == 0 to indicate
    # using the entire dataset, it must (1) be resettable, and (2) throws
    # tf.errors.OutOfRangeError when reading a batch beyond an epoch.
    p.Define(
        'eval_samples_per_summary', None, 'If not None, overrides '
        'task_p.eval.samples_per_summary directly. Allowed to be 0, which '
        'means to use the entire dataset.')
    p.Define(
        'decoder_samples_per_summary', None, 'If not None, overrides '
        'task_p.eval.decoder_samples_per_summary directly. Allowed to be 0, '
        'which means to use the entire dataset.')
    p.Define(
        'filter_sparse_tensors', False,
        'If true, filter out SparseTensors in input_batch before enqueuing '
        'onto TPU.')
    cls.DefineInfeedParams(p)

    p.Define('remote', hyperparams.Params(),
             'Params to configure remote input policy.')
    p.remote.Define(
        'max_inflights_per_target', 32, 'The maximum number of '
        'concurrent inflight remote input fetches per remote target.')

    p.Define(
        'input_stats_summary_interval_steps', 10,
        'Number of steps in between logging of TF scalar summaries for '
        'training related input data stats.')

    p.Define(
        'tpu_embedding_mode', 'train',
        'The mode used to enqueue TPU embedding ids. Valid values are: {'
        'None: no TPU embedding enqueue ops will be generated; '
        '"inference": enqueue ops will be generated, but backprop will be '
        'disabled (i.e. no gradient will be generated and the embedding '
        'tables are freezed); '
        '"train": both enqueue ops and gradient will be generated when '
        'do_eval is False, otherwise fallback to "inference" mode; }.')

    return p

  def __init__(self, params):
    super().__init__(params)
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

    # Set to true in GetPreprocessedInputBatch() (and thus _InputBatch())
    self._in_get_processed_input_batch = False

    # Merged TF scalar summaries for training related input data stats.
    self._merged_input_data_summary_op = None

    # Tensorboard layout for charts displaying input data stats.
    self._input_data_summary_layout = None

    assert self.params.tpu_embedding_mode in [None, 'train', 'inference']
    self._tpu_embedding_mode = self.params.tpu_embedding_mode
    if self._tpu_embedding_mode == 'train' and self.do_eval:
      self._tpu_embedding_mode = 'inference'  # Always disable backprop in eval.

    if self.parent:
      # Set the TPU embedding mode for the task. This need to happen in __init__
      # so that the mode is available when the bprop graph is built (note that
      # CreateTpuEmbeddingEnqueueOps() is called *after* building bprop graph).
      tpu_embedding_collection = (
          tpu_embedding_layers.TpuEmbeddingCollection.Get())
      tpu_embedding_collection.SetTaskMode(
          py_utils.TaskCallScopeName(self.parent), self._tpu_embedding_mode)

    self.CreateDatasource()

  def CreateDatasource(self):
    if self.params.file_datasource:
      self.CreateChild('datasource', self.params.file_datasource)
      self.datasource.SetInputGenerator(self)

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
    # Uses `InfeedBatchSize()` instead of calculating it from `p.batch_size`
    # because the behavior would be overridden by subclasses.
    global_batch_size = batch_utils.scale_infeed_to_global(
        self.InfeedBatchSize(), self.params.use_per_host_infeed)
    tf.logging.info('GlobalBatchSize {}'.format(global_batch_size))
    return global_batch_size

  def InfeedBatchSize(self):
    """Returns the batch size of the input batch: int or dynamic int tensor."""
    batch_per_input = batch_utils.scale_split_to_infeed(
        self.params.batch_size, self.params.use_per_host_infeed)
    tf.logging.info('batch_per_input: %d', batch_per_input)
    return batch_per_input

  def Initialize(self, sess=None):
    """Initialize using a session."""
    if 'datasource' in self.children:
      self.datasource.Initialize(sess)

  def _InputBatch(self):
    """The current input batch, not preprocessed.

    This is meant to be overridden by subclasses, but not called directly.
    Callers should use `GetPreprocessedInputBatch()`.

    Returns:
      A NestedMap (or list of NestedMaps when using TPU sharded infeed) of
      input tensors.
    """
    raise NotImplementedError('Abstract method')

  def _PreprocessInputBatch(self, batch):
    """Preprocesses input batch from _InputBatch.

    Args:
      batch: A NestedMap (or list of NestedMaps when using TPU sharded infeed)
        containing input tensors in the format returned by _InputBatch.

    Returns:
      A NestedMap containing preprocessed inputs to feed to the model.
    """
    return batch

  def GetPreprocessedInputBatch(self):
    """Returns preprocessed batch of inputs.

    These are the actual inputs fed to the model.

    Subclasses generally should not override this function directly. Instead,
    override _InputBatch and maybe _PreprocessInputBatch.
    """
    self._in_get_processed_input_batch = True
    # TODO(b/139345706): Use self.datasource.GetNext() for all datasource.
    if ('datasource' in self.children and
        isinstance(self.datasource, datasource.TFDatasetSource)):
      if self.cluster.input_targets:
        raise ValueError(
            'TFDatasetSource subclassed DataSources do not support using '
            'train_input_replica. Try tf_data_service_replicas instead.')
      # pylint: disable=protected-access
      if ((self._InputBatch.__func__ is not BaseInputGenerator._InputBatch and
           self._InputBatch.__func__
           is not BaseInputGeneratorFromFiles._InputBatch) or
          self._PreprocessInputBatch.__func__
          is not BaseInputGenerator._PreprocessInputBatch):
        # pylint: enable=protected-access
        # If you hit this error trying to run with --tf_data_service_replicas,
        # try to refactor your input generator by moving all the code inside
        # _InputBatch and _PreprocessInputBatch to _DataSourceFromFilePattern.
        raise ValueError(
            'Batches obtained through p.file_datasource do not go through '
            'self._InputBatch() or self._PreprocessInputBatch(). To reduce the '
            'potential of mistakes, this error is raised when either of those '
            'functions have been overridden.')
      batch = self.datasource.GetNext()
    else:
      batch = self._PreprocessInputBatch(self._InputBatch())
    self._in_get_processed_input_batch = False

    if py_utils.GetUnitTestSession():
      self.Initialize(py_utils.GetUnitTestSession())
    return batch

  @property
  def tpu_number_of_shards(self):
    """Number of shards to split the input batch into."""
    p = self.params
    num_tpu_hosts = self.cluster.num_tpu_hosts
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1
    shards = (self.cluster.total_worker_devices // num_infeed_hosts)
    if p.use_partitioned_infeed_queue or not p.use_per_core_infeed:
      shards = shards // self.cluster.num_devices_per_split
    return shards

  def CreateTpuEnqueueOps(self, job_name=None):
    """Create the host-side enqueue ops.

    This should be called in an outer non-TPU context.
    Args:
      job_name: the name of the job on which the enqueue operations run.
    """
    if not py_utils.IsEagerMode():
      assert not self._tpu_queues, (
          'CreateTpuEnqueueOps should only be called once.')
    self._tpu_queues = []
    self._per_host_batches = []
    self._per_host_emb_batches = []
    # A list of lists, where the [i][j] element is the j-th passthrought batch
    # of the i-th task. Each task will have more than one passthrought batch iff
    # sharded infeed is used.
    self._per_host_passthrough_batches = []
    p = self.params
    num_tpu_hosts = self.cluster.num_tpu_hosts
    num_cores_per_host = self.cluster.total_worker_devices // num_tpu_hosts
    tf.logging.info(
        'CreateTpuEnqueueOps num_splits_per_client={} '
        'num_devices_per_split={} num_tpu_hosts={} use_per_host_infeed={}'
        .format(self.cluster.num_splits_per_client,
                self.cluster.num_devices_per_split, num_tpu_hosts,
                p.use_per_host_infeed))

    assert num_tpu_hosts > 0, ('num_tpu_hosts: %d' % num_tpu_hosts)
    if p.use_per_core_infeed:
      if (not p.use_per_host_infeed) or p.use_partitioned_infeed_queue:
        raise ValueError('use_per_core_infeed need to have use_per_host_infeed '
                         'but not use_partitioned_infeed_queue.')
      if p.num_partitions is None or p.num_partitions <= 1:
        raise ValueError('use_per_core_infeed needs num_partitions > 1.')
    if (self.cluster.num_devices_per_split > num_cores_per_host and
        (p.use_per_host_infeed and not p.use_per_core_infeed)):
      tf.logging.fatal('Doesn\'t support per host infeed mode when '
                       'num_devices_per_split({}) > num_cores_per_host({}).'
                       'Each host must be able to accommodate >= 1 split when '
                       'using per_host_infeed.'.format(
                           self.cluster.num_devices_per_split,
                           num_cores_per_host))

    shards = self.tpu_number_of_shards
    tf.logging.info('shards {}'.format(shards))

    input_ops_list = []
    cpu_passthrough_keys = self.GetCpuPassthroughKeys()

    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1
    tf.logging.info('num_infeed_hosts: %d', num_infeed_hosts)
    host_devices = self.cluster.ListDevices(self.cluster.job_spec).flatten()
    if p.use_per_host_infeed and num_infeed_hosts != len(host_devices):
      raise ValueError(
          f'Configuration mismatch, number of infeed hosts {num_infeed_hosts} '
          f'does not match available devices {host_devices}.')
    for task_id in range(num_infeed_hosts):
      host_device = host_devices[task_id]
      if cpu_passthrough_keys and (
          '/task:{}/device:CPU:0'.format(task_id) not in host_device):
        raise ValueError(
            f'CPU passthrough configuration mismatch, device {host_device} '
            f'does not match task id {task_id}.')
      with tf.device(host_device), cluster.InfeedContextScope(
          infeed_host_index=task_id, num_infeed_hosts=num_infeed_hosts):
        batch = self.GetPreprocessedInputBatch()
        if not isinstance(batch, (list, tuple)):
          batch = [batch]

        cur_passthrough_batches = []
        for i in range(len(batch)):
          b = batch[i]
          assert isinstance(b, py_utils.NestedMap)
          # Hack: bucket_keys and xxx.bucket_keys are not needed on TPU.
          # Note that when MultiTaskData is used, bucket_keys will be at the
          # second level of the dictionary.
          b = b.FilterKeyVal(lambda k, _: not k.endswith('bucket_keys'))

          # Split out any keys that are meant for CPU passthrough only.
          cur_passthrough_batches.append(
              b.FilterKeyVal(lambda k, _: k in cpu_passthrough_keys))
          b = b.FilterKeyVal(lambda k, _: k not in cpu_passthrough_keys)
          batch[i] = b
          if i > 0:
            # If the input batch is already sharded, check that the shards are
            # compatible with each other.
            assert py_utils.IsCompatible(b, batch[0])
        self._per_host_passthrough_batches.append(cur_passthrough_batches)
        tf.logging.info('CPU passthrough keys: %s', cpu_passthrough_keys)

        if p.filter_sparse_tensors:
          # Make a copy of this host's input batch, then filter out any
          # SparseTensor features. This way, SparseTensor features are not fed
          # into the TPU InfeedQueue (and only to TPUEmbedding).
          # TODO(jeffreyzhao): Hack, come up with better solution.
          # Ideally we would like users to override
          # CreateTpuEmbeddingEnqueueOps() to modify the input batch
          # and remove fields they don't want to enqueue onto TPU.
          # However, the TPUEmbedding singleton and TPU embedding enqueue ops
          # are currently constructed after CreateTpuEnqueueOps() is called.
          emb_batch = []
          new_batch = []
          for i, b in enumerate(batch):
            emb_batch.append(
                b.Filter(lambda v: isinstance(v, tf.sparse.SparseTensor)))
            new_batch.append(
                b.Filter(lambda v: not isinstance(v, tf.sparse.SparseTensor)))
          self._per_host_emb_batches.append(emb_batch)
          batch = new_batch

        self._batch_nm_types = batch[0]
        tf.logging.info('host_device: %s, batch: %r', host_device, batch)
        self._per_host_batches.append(batch)

        for b in batch:
          for k, x in b.FlattenItems():
            assert x.shape.is_fully_defined(), (
                'Shape must be fully defined: %s: %s' % (k, x))
          # TODO(cwhipkey): if it's a string (or other type not supported on
          # TPU), drop it from feeding and on the other end add in an op that
          # fails if used.
        shapes = batch[0].Transform(lambda x: x.shape).Flatten()
        dtypes = batch[0].Transform(lambda x: x.dtype).Flatten()

        tf.logging.info('host_device: %s infeed shapes: %r', host_device,
                        shapes)
        tf.logging.info('host_device: %s infeed dtypes: %r', host_device,
                        dtypes)

        if p.use_partitioned_infeed_queue:
          device_assignment = py_utils.GetTpuDeviceAssignment(job_name)

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
          if p.use_per_core_infeed:
            q = tpu_feed.InfeedQueue(
                tuple_types=dtypes,
                tuple_shapes=shapes,
                number_of_partitions=p.num_partitions)
          elif len(batch) > 1:
            # When the input batch is sharded, the unsharded dtypes and shapes
            # will be determined later by the generate_enqueue_ops() call.
            q = tpu_feed.InfeedQueue(
                number_of_tuple_elements=len(batch[0].Flatten()))
          else:
            q = tpu_feed.InfeedQueue(tuple_types=dtypes, tuple_shapes=shapes)
          assert shards is not None
          q.set_number_of_shards(shards)

        self._tpu_queues.append(q)

        if p.use_partitioned_infeed_queue:
          assert len(batch) == 1
          input_ops = q.generate_enqueue_ops([batch[0].Flatten()])
        elif p.use_per_host_infeed:
          # TODO(ylc/zhifengc): Add this to a policy module and test it.
          def TPUOrdinalFunction(shard_index_in_host):
            if p.use_per_core_infeed:
              return shard_index_in_host
            device_assignment = py_utils.GetTpuDeviceAssignment()
            if device_assignment:
              # We put both enqueue/dequeue ops at core 0 in each replica.
              replica = device_assignment.lookup_replicas(
                  task_id, 0)[shard_index_in_host]  # pylint: disable=cell-var-from-loop
              return device_assignment.tpu_ordinal(replica=replica)
            else:
              return shard_index_in_host

          if len(batch) > 1:
            # In this case, the `shard_index_in_host` argument of
            # `TPUOrdinalFunction` is the index of a sharded batch in the
            # `batch` list.
            input_ops = q.generate_enqueue_ops(
                [b.Flatten() for b in batch],
                placement_function=lambda x: host_device,  # pylint: disable=cell-var-from-loop
                tpu_ordinal_function=TPUOrdinalFunction)
          else:
            input_ops = q.split_inputs_and_generate_enqueue_ops(
                batch[0].Flatten(),
                placement_function=lambda x: host_device,  # pylint: disable=cell-var-from-loop
                tpu_ordinal_function=TPUOrdinalFunction)
        else:
          assert len(batch) == 1
          input_ops = q.split_inputs_and_generate_enqueue_ops(
              batch[0].Flatten(),
              device_assignment=py_utils.GetTpuDeviceAssignment(job_name))
        input_ops_list += input_ops

    tf.logging.info('input_ops_list %s', input_ops_list)
    grouped_infeed_op = tf.group(*input_ops_list)
    self._tpu_infeed_op = []
    for _ in range(p.tpu_infeed_parallelism):
      self._tpu_infeed_op.append(grouped_infeed_op)

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
    return self._batch_nm_types.Pack(tensors)

  def CreateTpuEmbeddingEnqueueOps(self):
    """Creates the TpuEmbedding enqueue ops on all hosts.

    Note that this must be called after the instantiation of the
    monolithic TPUEmbeddingLayer.
    """
    p = self.params
    if self._tpu_embedding_mode is None:
      return

    tpu_embedding_collection = tpu_embedding_layers.TpuEmbeddingCollection.Get()
    tpu_embedding = tpu_embedding_collection.tpu_embedding
    if not tpu_embedding:
      return

    num_tpu_hosts = self.cluster.num_tpu_hosts
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1
    input_batches = (
        self._per_host_emb_batches
        if p.filter_sparse_tensors else self._per_host_batches)
    assert len(input_batches) == num_infeed_hosts

    enqueue_ops = []
    if num_tpu_hosts > 1 and not p.use_per_host_infeed:
      batch = input_batches[0]
      assert len(batch) == 1, "Tpu Embedding doesn't support sharded inputs."
      with tf.device('/task:0/device:CPU:0'):
        batch = self.PreprocessTpuEmbeddingInputBatch(batch[0])
        # When not using per-host infeed, we use `self.tpu_number_of_shards`
        # when splitting the inputs, so `num_tpu_hosts` is taken into account.
        all_enqueue_data = self._GetTpuEmbeddingEnqueueData(
            tpu_embedding, batch, self.tpu_number_of_shards)

        # Translate replica index to (host_device, tpu_ordinal). The mechanism
        # need to be the same as the one for other tpu infeed, so that the same
        # split of tpu-embedding and non-tpu-embedding inputs are sent to the
        # same core. See CreateTpuEnqueueOps() for more details.
        num_cores_per_host = tpu_embedding.num_cores_per_host
        enqueue_data_per_host = {}
        device_assignment = py_utils.GetTpuDeviceAssignment()
        for replica_index, per_replica_data in enumerate(all_enqueue_data):
          host_device = device_assignment.host_device(replica=replica_index)
          core = device_assignment.tpu_ordinal(replica=replica_index)
          assert core < num_cores_per_host
          if host_device not in enqueue_data_per_host:
            enqueue_data_per_host[host_device] = [None] * num_cores_per_host
          assert enqueue_data_per_host[host_device][core] is None
          enqueue_data_per_host[host_device][core] = per_replica_data
        assert len(enqueue_data_per_host) == num_tpu_hosts

      for host_device, src_enqueue_data in enqueue_data_per_host.items():
        with tf.device(host_device):
          # TF's `TPUEmbedding` colocates the enqueue ops with the input
          # tensors, so we add a tf.identity here to ensure they are copied to
          # `host_device` before generating the enqueue ops.
          dst_enqueue_data = [
              {} for _ in range(tpu_embedding.num_cores_per_host)
          ]
          # src_enqueue_data is a list of dicts, one for each core.
          for i, data_dict in enumerate(src_enqueue_data):
            assert data_dict, src_enqueue_data
            for key, data in data_dict.items():
              dst_enqueue_data[i][key] = tpu_embedding_lib.EnqueueData(
                  embedding_indices=tf.identity(data.embedding_indices),
                  sample_indices=tf.identity(data.sample_indices)
                  if data.sample_indices is not None else None,
                  aggregation_weights=tf.identity(data.aggregation_weights)
                  if data.aggregation_weights is not None else None)
          tf.logging.info('host_device: %s, enqueue_data: %r', host_device,
                          dst_enqueue_data)
          enqueue_ops += tpu_embedding.generate_enqueue_ops(
              dst_enqueue_data, mode_override=self._tpu_embedding_mode)
    else:
      assert tpu_embedding.num_cores_per_host == self.tpu_number_of_shards
      for task_id in range(num_tpu_hosts):
        host_device = '/task:{}/device:CPU:0'.format(task_id)
        batch = input_batches[task_id]
        assert len(batch) == 1, "Tpu Embedding doesn't support sharded inputs."
        with tf.device(host_device):
          batch = self.PreprocessTpuEmbeddingInputBatch(batch[0])
          tf.logging.info('host_device: %s, batch: %r', host_device, batch)
          enqueue_data = self._GetTpuEmbeddingEnqueueData(
              tpu_embedding, batch, tpu_embedding.num_cores_per_host)
          enqueue_ops += tpu_embedding.generate_enqueue_ops(
              enqueue_data, mode_override=self._tpu_embedding_mode)

    self._tpu_infeed_op.append(tf.group(*enqueue_ops))

  def _GetTpuEmbeddingEnqueueData(self, tpu_embedding, input_batch, num_splits):
    """Get a list of per-core TPU embedding enqueue data.

    Args:
      tpu_embedding: The monolithic TpuEmbedding object.
      input_batch: The input batch used to generate the enqueue data.
      num_splits: The number of shards to split the inputs into in order to get
        per-core inputs, before generating enqueue data.

    Returns:
      A list of `num_splits` enqueue elements, where each element is a dict of
      feature_name -> `tpu_embedding_lib.EnqueueData`.
    """
    assert isinstance(input_batch, py_utils.NestedMap)
    tpu_emb_input_keys = list(tpu_embedding.feature_to_config_dict.keys())
    tf.logging.info('tpu_emb_input_keys: %r', tpu_emb_input_keys)
    enqueue_data = [{} for _ in range(num_splits)]

    # Get enqueue data for each replica.
    for key in tpu_emb_input_keys:
      feat = input_batch.GetItem(key)
      if isinstance(feat, tf.sparse.SparseTensor):
        tpu_emb_feat_splitted = tf.sparse.split(feat, num_splits, axis=0)
        for i, split in enumerate(tpu_emb_feat_splitted):
          enqueue_data[i][key] = (
              tpu_embedding_lib.EnqueueData.from_sparse_tensor(split))
      else:
        tpu_emb_feat_splitted = tf.split(feat, num_splits)
        for i, split in enumerate(tpu_emb_feat_splitted):
          # Dense to sparse. Note the assumption of a padding id.
          sample_indices = tf.where(tf.not_equal(split, -1))
          embedding_indices = tf.gather_nd(split, sample_indices)
          enqueue_data[i][key] = tpu_embedding_lib.EnqueueData(
              embedding_indices, sample_indices)
    return enqueue_data

  def PreprocessTpuEmbeddingInputBatch(self, input_batch):
    """Hook to manipulate the TPU embedding input batch.

    Used by CreateTpuEmbeddingEnqueueOps(). Override this method in input
    generators to preprocess the TPU embedding inputs before using them to
    generate enqueue ops.

    Args:
      input_batch: The input batch to process.

    Returns:
      The preprocessed TPU embedding input batch.
    """
    return input_batch

  def GetCpuPassthroughKeys(self):
    """Return a list of keys from the input to skip sending to the device.

    When running on TPU, a user may want to avoid sending some inputs to the
    device; either the type is not supported (e.g., string), or the input will
    not be processed on the device at all.  However, these items may be still
    useful to passthrough to the "output", e.g., for decoding purposes.

    This function should return a list of keys from InputBatch() that should not
    be sent to the TPU, but can be combined with the outputs of Decode() before
    passing to PostProcessDecodeOut().

    Returns:
      A list of keys from the input to filter from being sent to the device,
        which may be combined with the output of Decode() prior to
        PostProcessDecodeOut().
    """
    return []

  def CreateCpuPassthroughEnqueueOps(self):
    """Creates enqueue ops to pass through CPU inputs to the output."""
    p = self.params
    num_tpu_hosts = self.cluster.num_tpu_hosts
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1

    cpu_passthrough_keys = self.GetCpuPassthroughKeys()
    if not cpu_passthrough_keys:
      return

    # There is one enqueue op per host.
    self._host_queues = {}
    enqueue_ops = []

    assert len(self._per_host_batches) == num_infeed_hosts
    for task_id in range(num_infeed_hosts):
      host_device = '/task:{}/device:CPU:0'.format(task_id)
      batch = self._per_host_passthrough_batches[task_id]
      assert isinstance(batch, list)
      with tf.device(host_device):
        self._cpu_nm_types = batch[0] if len(batch) == 1 else batch
        tf.logging.info('host_device CPU passthrough types: %s, batch: %r',
                        host_device, batch)
        cpu_dtypes = py_utils.Flatten(
            py_utils.Transform(lambda x: x.dtype, batch))
        # NOTE: we use a large capacity queue under the assumption that the size
        # of these tensors will be generally smaller than that sent to the TPU,
        # and that the TPU queue will likely fill up before the host queue,
        # blocking further enqueues.
        host_queue = tf.queue.FIFOQueue(capacity=10000, dtypes=cpu_dtypes)
        self._host_queues[task_id] = host_queue
        enqueue_ops += [host_queue.enqueue(py_utils.Flatten(batch))]
    self._tpu_infeed_op.append(tf.group(*enqueue_ops))

  def DequeueCpuPassthrough(self, concat=True):
    """Create CPU dequeue ops.

    Args:
      concat: Whether to concat the passthrough batches for each host into one
        batch.

    Returns:
      None if there are no CPU passthrough values. Otherwise, a NestedMap of the
      CPU passthrough input batch if `concat`, or a list of NestedMaps (one for
      each host) if not `concat`.
    """
    cpu_passthrough_keys = self.GetCpuPassthroughKeys()
    if not cpu_passthrough_keys:
      return None

    p = self.params
    num_tpu_hosts = self.cluster.num_tpu_hosts
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1
    tensor_list = []
    for task_id in range(num_infeed_hosts):
      with tf.device('/task:{}/device:CPU:0'.format(task_id)):
        tensors = self._host_queues[task_id].dequeue()
        # Make list if only one tensor.
        if not isinstance(tensors, list):
          tensors = [tensors]
        tensor_list.append(tensors)

    # TODO(laigd): consider moving the concat logic out to make the API simpler.
    if concat:
      with tf.device('/task:0/device:CPU:0'):
        # Transpose to get per-dequeue-element tuples, then concat.
        result = list(map(lambda xs: tf.concat(xs, axis=0), zip(*tensor_list)))
        return py_utils.Pack(self._cpu_nm_types, result)

    # Return a list of batches, one per host.
    return [py_utils.Pack(self._cpu_nm_types, xs) for xs in tensor_list]

  @property
  def tpu_infeed_op(self):
    if self._tpu_infeed_op is not None:
      return self._tpu_infeed_op
    else:
      raise ValueError('TPU infeed op not set. Call CreateTpuEnqueueOps first.')

  @property
  def merged_input_data_summary_op(self):
    return self._merged_input_data_summary_op

  @property
  def input_data_summary_layout(self):
    return self._input_data_summary_layout

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

  def Reset(self, sess=None):
    """Reset the input-generator.

    Override so that the input_generator reproduces examples as if from a fresh
    instantiation.

    Args:
      sess: A tensorflow session.
    """
    raise NotImplementedError()

  @property
  def _map_args(self):
    """Default args for tf.data.DataSet.map()."""
    return {
        'num_parallel_calls':
            1 if self.cluster.in_unit_test else tf.data.experimental.AUTOTUNE,
        'deterministic':
            self.cluster.require_sequential_input_order
    }


def FilePatternToDataSource(p):
  """Helper to turn p.file_pattern (deprecated) into p.file_datasource."""
  if isinstance(p.file_pattern, str):
    ds = datasource.SimpleDataSource.Params().Set(file_pattern=p.file_pattern)
  elif isinstance(p.file_pattern, (list, tuple)):
    if all([isinstance(x, str) for x in p.file_pattern]):
      # While this violates the documentation and intended use, there are
      # subclasses that have used a tuple of strings, rather than a list of
      # string, weight tuples.  Rather than treating lists and tuples
      # differently, support both here until p.file_pattern is removed.
      ds = datasource.SimpleDataSource.Params().Set(
          file_pattern=list(p.file_pattern))
    elif p.use_within_batch_mixing:
      if max(list(map(len, p.file_pattern))) >= 3:
        # Within batch mixing doesn't work with backprop filters, i.e. when
        # file_pattern param contains a list of
        # <file_pattern, weight, [bprop_variable_filter]> tuples.
        raise ValueError('Expected a list of pairs, got %s' % p.file_pattern)

      file_patterns, weights = (list(x) for x in zip(*p.file_pattern))

      ds = datasource.SimpleDataSource.Params().Set(
          file_pattern=file_patterns, weights=weights)
    else:
      # Otherwise fall back to MixByWeight-based approach.
      datasources = []
      weights = []
      bprop_variable_filters = []
      for source_id, input_entry in enumerate(p.file_pattern):
        if isinstance(input_entry, str):
          raise ValueError('Should explicitly specify weights, got string: %s' %
                           input_entry)
        file_pattern, weight = input_entry[:2]
        datasources.append(
            datasource.SimpleDataSource.Params().Set(file_pattern=file_pattern))

        # This is essentially a bug fix, but we only enable it based on this
        #  param to maintain backward compatibility.
        if not p.all_zero_source_id_without_within_batch_mixing:
          # SimpleDataSource will output source_id=0. We use source_id_offset
          # to correct this.
          datasources[-1].Set(source_id_offset=source_id)

        weights.append(weight)
        bprop_variable_filter = input_entry[2] if len(input_entry) > 2 else ''
        bprop_variable_filters.append(bprop_variable_filter)
      ds = datasource.CrossBatchMixingDataSource.Params().Set(
          sub=datasources,
          weights=weights,
          bprop_variable_filters=bprop_variable_filters)
  else:
    raise ValueError('Cannot parse p.file_pattern into a datasource.')

  cluster_cur = cluster_factory.Current()
  if cluster_cur.tf_data_service_address and not cluster_cur.do_eval:
    bucket_upper_bound = None
    if 'bucket_upper_bound' in p:
      bucket_upper_bound = p.bucket_upper_bound
    ds = datasource.TFDataServiceSource.Params().Set(
        sub=ds, bucket_upper_bound=bucket_upper_bound)
    ds = datasource.TFDatasetPrefetch.Params().Set(sub=ds)

  return ds


class BaseInputGeneratorFromFiles(BaseInputGenerator):
  """Base class for input generators that reads from files.

  Subclasses should implement _DataSourceFromFilePattern.
  """

  @classmethod
  def Params(cls):
    """Defaults params for input generators."""
    p = super().Params()
    p.Define(
        # NOTE: file_pattern is deprecated.  New params should use
        # file_datasource instead.
        # TODO(b/139345706) remove file_pattern parameter
        'file_pattern',
        '',
        'A single file pattern string, a list of file pattern strings or a list'
        ' of <file_pattern, weight> pairs or a list of  <file_pattern, weight, '
        'bprop_variable_filter> tuples. Some of the cases may not be supported '
        'with use_within_batch_mixing, where probablistic samples are from the '
        'inputs proportional to their weights. Typically, values are binary '
        'protocol buffers containing train/eval samples. Keys are not used.')
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
        'flush_every_n', 0, 'If non-zero, flushes all batches buffered '
        'so far every these many records are yielded.')
    p.Define('num_batcher_threads', 1, 'Number of threads to use for input '
             'record batcher.')
    p.Define(
        'repeat_count', -1,
        'Number of repetitions of a dataset before throwing OutOfRange error '
        'when using require_sequential_input_order. Must only be set if '
        'cluster.require_sequential_input_order is True.')
    # TODO(b/139345706) when file_pattern is deleted use_within_batch_mixing
    # will be specified by setting weights in SimpleDataSource in
    # p.file_datasource and this param should be deleted as well.
    p.Define(
        'use_within_batch_mixing', False, 'Whether to mix records from '
        'different input sources within batch or across batches (the '
        'default option). This option only takes effect when file_pattern'
        ' is a list of file patterns with weights. Note: without mixing, all'
        ' source_id values for records will be set to 0 unless '
        'all_zero_source_id_without_within_batch_mixing is set to False.')
    p.Define(
        'all_zero_source_id_without_within_batch_mixing', True,
        'When set (by default) and use_within_batch_mixing is false, all '
        'record.source_id values returned will be 0. This is most likely '
        'undesired behavior, but enables backwards compatibility with previous '
        'work. Only classes that have _DataSourceFromFilePattern take a '
        'input_source_id_offset argument can handle this flag being False.')

    return p

  def __init__(self, params):
    if params.use_per_host_infeed and params.file_random_seed != 0:
      raise ValueError('file_random_seed needs to be 0 when '
                       'use_per_host_infeed == True.')

    super().__init__(params)

  def CreateDatasource(self):
    p = self.params
    assert not (
        p.file_pattern and p.file_datasource
    ), 'Only one of file_pattern and file_datasource can be specified'

    if not p.file_datasource:
      p.file_datasource = FilePatternToDataSource(p)
      # TODO(b/139345706) remove support for file_pattern
      # p.file_pattern = ''

    super().CreateDatasource()

  def CommonInputOpArgs(self):
    """Common input params."""
    p = self.params

    args = super().CommonInputOpArgs()
    num_input_replicas = 1
    input_replica_id = 0
    infeed_context = cluster.GetInfeedContext()
    if infeed_context:
      num_input_replicas = infeed_context.num_infeed_hosts
      input_replica_id = infeed_context.infeed_host_index
      tf.logging.info('input_replica_id=%s/%s', input_replica_id,
                      num_input_replicas)
    # Legacy behavior for Lingvo input ops: require_sequential_order defaults to
    # False for eval jobs. Note that this value is different from
    # self.cluster.require_sequential_input_order.
    require_sequential_order = bool(
        self.cluster.params.require_sequential_input_order)
    args.update({
        'file_random_seed': p.file_random_seed,
        'file_buffer_size': p.file_buffer_size,
        'file_parallelism': p.file_parallelism,
        'file_buffer_size_in_seconds': p.file_buffer_size_in_seconds,
        'flush_every_n': p.flush_every_n,
        'num_threads': p.num_batcher_threads,
        'require_sequential_order': require_sequential_order,
        'repeat_count': p.repeat_count,
        'num_input_replicas': num_input_replicas,
        'input_replica_id': input_replica_id,
    })
    args.update(self._InputOpBucketingArgs())
    return args

  def _InputOpBucketingArgs(self):
    return {
        'bucket_upper_bound': [1000000],
        'bucket_batch_limit': [self.InfeedBatchSize()],
        'bucket_adjust_every_n': 0,
    }

  def _InputBatch(self):
    return self._BuildDataSource()

  # TODO(b/139345706): After p.file_pattern is deleted, the following functions
  # _DataSourceFromFilePattern, _BuildDataSourceWithMetadata, _BuildDataSource
  # can be deleted and functionality moved to using the DataSource directly.
  def _DataSourceFromFilePattern(self,
                                 file_pattern,
                                 input_source_weights=None,
                                 input_source_id_offset=0):
    """Return a NestedMap containing an input batch from a string file_pattern.

    Subclasses should implement this function.

    Args:
      file_pattern: A string file pattern.
      input_source_weights: A list of float input source weights to control
        input example mix in the batch. The records will be sampled from inputs
        proportionally to these weights. Defaults to None which should be
        treated as an empty list.
      input_source_id_offset: All source_ids returned from datasource will be
        offset by this value.

    Returns:
      A `.NestedMap` of tf.Tensors containing a batch of input data with shapes
      [batch, ...].
    """
    return py_utils.NestedMap(x=tf.zeros([1]))

  def _BuildDataSourceWithMetadata(self):
    """Read and return input batch from `p.file_pattern`.

    `p.file_pattern` may be a string file_pattern or a
    list of (file_pattern, weight, [bprop_variable_filter]) tuples.
    bprop_variable_filter is optional. When bprop_variable_filter is used,
    batches will always contain the examples from the same source. Otherwise,
    examples from different sources may be mixed together.

    Returns:
      A `.NestedMap` containing

      - data: `.NestedMap` of tf.Tensor as in `_DataSourceFromFilePattern()`.
      - source_selected: optional tensor of size [batch_size, #datasources].
      - selected_bprop: optional tensor of size [#datasources].
      - bprop_variable_filters: optional list of filters for each source.

    Raises:
      ValueError: If file_datasource is not set
    """
    p = self.params
    if p.use_per_host_infeed and not self._in_get_processed_input_batch:
      raise ValueError(
          'This input generator does not support p.use_per_host_infeed. '
          'Please set it to False, or move the call to self._BuildDataSource() '
          'from self.__init__() to self._InputBatch() for batches to be '
          'correctly replicated per host.')
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
      batch = self.datasource.GetNext()
      ret = self.datasource.GetMeta()
      ret.data = batch
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
      A `.NestedMap` of tf.Tensor as in `self._DataSourceFromFilePattern()`.

    Raises:
      ValueError: If unknown token type.
    """
    return self._BuildDataSourceWithMetadata()['data']


class BaseSequenceInputGenerator(BaseInputGeneratorFromFiles):
  """The basic sequence input generator.

  Subclasses should implement _DataSourceFromFilePattern defined in
  BaseInputGeneratorFromFiles.
  """

  @classmethod
  def Params(cls):
    """Defaults params for sequence input generators."""
    p = super().Params()
    p.Delete('batch_size')

    # How input should be bucketized.
    p.Define(
        'bucket_upper_bound', [2560], 'Bucketing scheme. Required to be'
        'a sorted list of integers. Examples that are longer than all bucket'
        'upper bounds are skipped.')
    p.Define(
        'bucket_batch_limit', [8],
        'Desired per-split batch size per bucket. Scaled in '
        'infeed_bucket_batch_limit to the infeed size.'
        'Must be the same length as bucket_upper_bound.')
    p.Define(
        'bucket_adjust_every_n', 0, 'If non-zero, optimize the values of '
        'bucket_upper_bound except the last one after every N records '
        'based on the current input length distribution.')
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

  def __init__(self, params):
    super().__init__(params)

    p = self.params

    if p.tokenizer:
      assert DEFAULT_TOKENIZER_KEY not in p.tokenizer_dict
      p.tokenizer_dict[DEFAULT_TOKENIZER_KEY] = p.tokenizer

    self.tokenizer_dict = {}
    for k, p in p.tokenizer_dict.items():
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
    infeed_bucket_batch_limit = [
        batch_utils.scale_split_to_infeed(b, p.use_per_host_infeed)
        for b in p.bucket_batch_limit
    ]
    tf.logging.info(
        'infeed_bucket_batch_limit={} num_splits_per_client={} bucket_batch_limit={}'
        .format(infeed_bucket_batch_limit, self.cluster.num_splits_per_client,
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
      tf.logging.warning('Using max bucket batch limit but not all limits are '
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
        'bucket_adjust_every_n': p.bucket_adjust_every_n,
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

      Usually ids[i, 0] == SOS, ids[i, j+1] == labels[i, j], and labels[i, :]
      ends with EOS. That is, `ids` and `labels` are inputs and ground-truth
      labels for step-by-step teacher-forcing training, respectively.

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

  def StringsToIdsWithOffsets(self,
                              strs,
                              is_source=False,
                              external_max_length=None,
                              external_append_eos=None,
                              key=None,
                              languages=None):
    """Tokenize strs into vocab ids, and also return byte-level offsets.

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
      - start_offset[i, j] is the byte-level offset of the start of the j-th id
          in the i-th original string
      - end_offset[i, j] is the byte-level offset of the end of the j-th id
          in the i-th original string

      Usually ids[i, 0] == SOS, ids[i, j+1] == labels[i, j], and labels[i, :]
      ends with EOS. That is, `ids` and `labels` are inputs and ground-truth
      labels for step-by-step teacher-forcing training, respectively.

    Raises:
      ValueError: If unknown token type.
      Exception: If the specified tokenizer does not support offsets.
    """
    p = self.params

    if external_max_length is not None:
      maxlen = external_max_length
    elif is_source:
      maxlen = p.source_max_length
    else:
      maxlen = p.target_max_length

    key = key or DEFAULT_TOKENIZER_KEY
    return self.tokenizer_dict[key].StringsToIdsWithOffsets(
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
    p = super().Params()
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


class TFDataSequenceInputGenerator(BaseSequenceInputGenerator):
  """tf.data input pipeline for sequences.

  Inherits params from BaseSequenceInputGenerator so this can be a drop-in
  replacement for existing input generators inheriting from
  BaseSequenceInputGenerator. However, many params may be ignored / unused.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('prefetch_buffer_size', 1, 'Local prefetch buffer size.')
    p.resettable = True
    return p

  def CreateDatasource(self):
    p = self.params
    if not p.file_datasource:
      # Convert p.file_pattern into p.file_datasource.
      ds = self.ConvertFilePatternToDataSource(p, p.file_pattern)
      p.file_pattern = ''
    else:
      ds = p.file_datasource

    ds = datasource.CustomTFDatasetTransform.Params().Set(
        sub=ds, fn='TakeEvalSamples')
    ds = datasource.TFDatasetBatchBySequenceLength.Params().Set(
        sub=ds,
        seqlen_fn='GetSequenceLength',
        input_shape_fn='_InputShape',
        input_padding_fn='_InputPaddingValue',
        bucket_upper_bound=p.bucket_upper_bound,
        bucket_batch_limit=p.bucket_batch_limit)
    if self.cluster.tf_data_service_address and not self.cluster.do_eval:
      ds = datasource.TFDataServiceSource.Params().Set(
          sub=ds, bucket_upper_bound=p.bucket_upper_bound)
    ds = datasource.TFDatasetPrefetch.Params().Set(
        sub=ds, buffer_size=p.prefetch_buffer_size)

    p.file_datasource = ds
    super().CreateDatasource()

  @classmethod
  def ConvertFilePatternToDataSource(cls, p, file_pattern):
    if isinstance(file_pattern, str):
      file_patterns = file_pattern.split(',')
      weights = None
    else:
      if all([isinstance(x, str) for x in file_pattern]):
        file_patterns = file_pattern
        weights = None
      elif all([isinstance(x, tuple) for x in file_pattern]):
        file_patterns, weights = zip(*file_pattern)
      else:
        raise ValueError(
            f'file_pattern must be all strings or all tuples, but got: '
            f'{file_pattern}.')
    for fp in file_patterns:
      if ',' in fp:
        raise ValueError(f'file_pattern should not contain comma: {fp}')

    ds = []
    for fp in file_patterns:
      ds.append(datasource.TFDatasetFnInput.Params().Set(
          load_fn='LoadDataset',
          kwargs=dict(file_pattern=fp),
          shuffle_buffer_size=p.file_buffer_size))
    if len(ds) > 1:
      if not p.use_within_batch_mixing:
        raise ValueError(
            'Only p.use_within_batch_mixing is supported with multiple '
            'file_patterns.')
      ds = [datasource.TFDatasetMixer.Params().Set(sub=ds, weights=weights)]
    ds = datasource.CustomTFDatasetTransform.Params().Set(
        sub=ds[0], fn='ProcessDataset')
    return ds

  def Reset(self, sess=None):
    self.datasource.Reset(sess)

  def GetPreprocessedInputBatch(self):
    return self.datasource.GetNext()

  def LoadDataset(self, file_pattern):
    """Load a dataset from file.

    Args:
      file_pattern: the path to the file to load.

    Returns:
      A tf.data.Dataset() whose elements represent a single training sample
      without a leading batch dim.
    """
    raise NotImplementedError()

  def TakeEvalSamples(self, dataset):
    p = self.params
    if self.do_eval and p.num_samples > 0:
      dataset = dataset.take(p.num_samples)
    return dataset

  def ProcessDataset(self, dataset):
    """Processes a dataset returned by LoadDataset.

    Args:
      dataset: A dataset returned by LoadDataset.

    Returns:
      A processed dataset containing NestedMaps of Tensors without a leading
      batch dimension.
    """
    raise NotImplementedError()

  def GetSequenceLength(self, example):
    """Returns sequence length for the example NestedMap from the dataset.

    Args:
      example: A NestedMap containing an input example. Tensors in the example
        do not have a leading batch dimension.

    Returns:
      An integer sequence length for the example.
    """
    raise NotImplementedError()

  def _InputShape(self, key):
    """Returns the final shape of the tensor corresponding to key as a tuple.

    The shape should not include a leading batch dimension.

    Args:
      key: The NestedMap key to return shape for.
    """
    if key in ('source_id', 'bucket_keys'):
      return ()

    raise ValueError('Unexpected key %s' % key)

  def _InputPaddingValue(self, key, tensorspec):
    """Returns the value to pad the tensor corresponding to key with."""
    if key.endswith('_paddings'):
      return tf.ones([], dtype=tensorspec.dtype)
    else:
      return tf.zeros([], dtype=tensorspec.dtype)


class BaseDataExampleInputGenerator(BaseInputGenerator):
  """Base class for input generators that read Feature protos via tf.data."""

  @classmethod
  def Params(cls):
    p = super().Params()
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
    return p

  def __init__(self, params):
    super().__init__(params)
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

  def _AdditionalPreprocessInputBatch(self, batch):
    """Additionally preprocesses input batch from iterator.get_next().

    Args:
      batch: A NestedMap (or list of NestedMaps when using TPU sharded infeed)
        containing input tensors in the format returned by
        _PreprocessInputBatch.

    Returns:
      A NestedMap containing additionally preprocessed inputs to feed to the
      model.
    """
    return batch

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
    return self._AdditionalPreprocessInputBatch(input_batch)


def DefineTFDataInput(name,
                      func,
                      ignore_args=None,
                      map_args=None,
                      base_class=BaseInputGenerator):
  """Defines a new InputGenerator class from given tf.data pipeline.

  This function allows users to utilize existing tf.data pipelines which are
  defined externally, without making binding boilerplates.
  The generated InputGenerator behaves like a one-shot iterator of the given
  pipeline. If the iterator is designed to be repeated, the returned
  InputGenerator will work similarly.
  This function generates `Params` automatically by analysing the given
  pipeline's signature so that the behavior of the pipeline can be saved into
  `Params`.
  This function defines the InputGenerator class on the caller's module. To
  avoid any confusion, the returned class have to be stored in the module-level
  symbol with the same identifier with given `name`.

  Example:
    >>> # A tf.data pipeline which returns a dict of Tensors.
    >>> def my_dataset(begin=0, end=10):
    ...   ds = tf.data.Dataset.from_tensor_slices(tf.range(begin, end))
    ...   return ds.map(lambda x: {'value': x})

    >>> # Defines the InputGenerator class for my_dataset.
    >>> MyInput = DefineTFDataInput('MyInput', my_dataset)

    >>> # Obtains Params of MyInput.
    >>> p = MyInput.Params()
    >>> assert p.args.begin == 0
    >>> assert p.args.end == 10

    >>> # Instantiates the InputGenerator from Params.
    >>> ig = p.Instantiate()
    >>> assert isinstance(ig, MyInput)

    >>> # Obtains the data tensors.

    >>> # In TFv1:
    >>> data = ig.GetPreprocessedInputBatch()
    >>> with tf.Session() as sess:
    ...   values = sess.run(data)  # {'value': 0}
    ...   values = sess.run(data)  # {'value': 1}
    ...   values = sess.run(data)  # {'value': 2}

    >>> # In TFv2:
    >>> values = ig.GetPreprocessedInputBatch()  # {'value': 0}
    >>> values = ig.GetPreprocessedInputBatch()  # {'value': 1}
    >>> values = ig.GetPreprocessedInputBatch()  # {'value': 2}

  Args:
    name: A string, representing the name of the new InputGenerator class.
    func: A callable to be analysed to generate the new InputGenerator. The
      return value of `func` must be a single `tf.data.Dataset` which yields a
      dict or its subclasses. The signature (parameter list) of `func` must have
      all explicit parameters needed to configure the pipeline. `*args` and
      `**kwargs` parameters would be ignored from defining `Params`.
    ignore_args: A collection of strings, representing the set of parameter
      names to be ignored from defining `Params`.
    map_args: A {str: str} dict, representing mappings from existing fields in
      `Params()` to `func`'s parameter. These mappings can be used to propagate
      some particular Lingvo-specific options defined by others (typically by
      super classes: `BaseInputGenerator` or `BaseLayer`) to the given function.
      Each entry in the dict represents a `{func_param: layer_param}` pair such
      that the `Params().layer_param` field will be mapped to the parameter
      `func_param` of `func`. `func_param` won't be added into `Params().args`
      to avoid duplicated definitions about the same parameters.
    base_class: A class name to inherit from, default is BaseInputGenerator.

  Returns:
    A new InputGenerator class that invokes `func` internally. The `Params()`
    method of the returned class makes a new Params containing the `args` field
    representing the parameters of `func`. The `GetPreprocessedInputBatch()`
    method returns a `py_utils.NestedMap` representing the same dict of the
    obtained data from the dataset.
  """
  ignore_args = set(ignore_args if ignore_args is not None else ())
  map_args = dict(map_args if map_args is not None else {})

  # Defines the class first as it will be required to call `super()`.
  generated_cls = type(name, (base_class,), {})

  @classmethod
  def _Params(cls):
    """Generates Params to configure the InputGenerator.

    This function analyses the signature of the given callable `func` and
    defines corresponding fields into `Params` to the obtained function
    parameters.

    Returns:
      An `InstantiableParams` object representing the InputGenerator. It has the
      `args` field which contains the set of parameters of `func`.
    """
    # Keys in `map_args` will also be ignored.
    actual_ignore_args = ignore_args | set(map_args.keys())

    p = super(generated_cls, cls).Params()

    # Introduces a new group `args` to avoid confusion between `func`'s
    # parameters and existing params defined by super classes.
    # TODO(oday): For better UX, consider removing this nested field and add
    # `func`s parameters to `p` directly. We need to make sure that there are no
    # side effects by integrating `func`'s parameters and follows:
    # - BaseInputGenerator.Params()
    # - BaseLayer.Params()
    # - InstantiableParams.cls
    p.Define('args', hyperparams.Params(), 'Parameter list of the pipeline.')
    inspect_utils.DefineParams(func, p.args, actual_ignore_args)
    ds = datasource.TFDatasetFnInput.Params().Set(
        load_fn='GetDataset', shuffle_buffer_size=1)
    cluster_cur = cluster_factory.Current()
    if cluster_cur.tf_data_service_address and not cluster_cur.do_eval:
      ds = datasource.TFDataServiceSource.Params().Set(sub=ds)
      ds = datasource.TFDatasetPrefetch.Params().Set(sub=ds)
    p.file_datasource = ds
    return p

  def _GetDataset(self):
    p = self.params
    overrides = {k: p.Get(v) for k, v in map_args.items()}
    dataset = inspect_utils.CallWithParams(func, p.args, **overrides)
    assert isinstance(dataset, (tf.tf1.data.Dataset, tf.tf2.data.Dataset)), (
        'DefineTFDataInput must take a callable which returns a '
        '`tf.data.Dataset`. The given callable `%s` returned `%s`' %
        (func, dataset))
    return dataset

  def _GetPreprocessedInputBatch(self):
    """Generates data tensors by invoking the pipeline."""
    # TFv1: Returns Tensors which will be determined by Session.run().
    # TFv2: Returns Tensors with actual values.
    data = self.datasource.GetNext()

    # Converts dict to NestedMap to maintain consistency with existing
    # functionalities in base_input_generator.
    # TODO(oday): Consider mitigating this restriction.
    assert isinstance(data, dict), (
        'DefineTFDataInput accepts only datasets that returns a dict or its '
        'subclasses.')
    if not isinstance(data, py_utils.NestedMap):
      data = py_utils.NestedMap.FromNestedDict(data)

    return data

  # Overrides member methods.
  generated_cls.Params = _Params
  generated_cls.GetDataset = _GetDataset
  generated_cls.GetPreprocessedInputBatch = _GetPreprocessedInputBatch

  # Sets __module__ to the caller's module name for pickling and restoring from
  # Params to work.
  # See also the namedtuple's implementation for details.
  module = inspect.stack()[1].frame.f_globals.get('__name__', '__main__')
  generated_cls.__module__ = module

  return generated_cls
