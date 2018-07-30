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
"""Input generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.framework import function
from tensorflow.python.ops import io_ops
from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import input_generator_helper as ig_helper
from lingvo.core import py_utils
from lingvo.core.ops import py_x_ops


class BaseInputGenerator(base_layer.LayerBase):
  """The base input generator."""

  @classmethod
  def Params(cls):
    """Defaults params for input generators."""
    p = super(BaseInputGenerator, cls).Params()
    p.name = 'input'
    p.Define('random_seed', None, 'Random seed used for input sampling.')
    p.Define(
        'num_samples', 0,
        'If non-zero, the dataset contains these many samples. '
        'For test/eval dataset, if we want the test/evel job evaluate '
        'the whole dataset, this param must be set precisely. Otherwise, '
        'this param is optional.')

    # TPU related infeed tuning.
    p.Define('use_per_host_infeed', False,
             'Whether run infeed op on each host.')
    p.Define('tpu_infeed_parallism', 1,
             'Uses these many python threads to drive infeed concurrently.')
    return p

  def __init__(self, params):
    super(BaseInputGenerator, self).__init__(params)
    self._sample_ids = None
    self._input_batch_size = None
    self._made_tpu_infeed = False

  def InputBatchSize(self):
    """Returns the batch size for a single step."""
    assert self._input_batch_size is not None, (
        'No input batch size is defined.')
    assert isinstance(
        self._input_batch_size,
        tf.Tensor), ('self._input_batch_size must be a tf.Tensor in the graph.')
    return self._input_batch_size

  def BatchSizePerInput(self, batch_size):
    """Returns the expected batch size for one single input batch."""
    p = self.params
    cluster = cluster_factory.Current()

    # If use_per_host_infeed, each input op is only responsible
    # for generating a subset of the whole batch.
    batch_per_input = batch_size * cluster.num_splits_per_client
    if p.use_per_host_infeed and cluster.num_tpu_hosts > 0:
      tf.logging.info('batch_size %d cluster.num_tpu_hosts %d', batch_per_input,
                      cluster.num_tpu_hosts)
      batch_per_input //= cluster.num_tpu_hosts
    tf.logging.info('batch_per_input: %d', batch_per_input)
    return batch_per_input

  def SampleIds(self):
    """Returns a string vector tensor.

    The i-th element in the vector is the id for the i-th sample in
    the batch.

    Returns:
      The sample ids, a 1D string Tensor, or None.
    """
    return self._sample_ids

  def InputBatch(self):
    """The current input batch, not preprocessed.

    This is meant to be overridden by subclasses, but not called directly.
    Callers should use GetPreprocessedInputBatch().

    Returns:
      A NestedMap of input tensors. Each tensor's dim-0 must be the same
      and denotes the batch dimension.
    """
    raise NotImplementedError('Abstract method')

  def GetPreprocessedInputBatch(self):
    return self.PreprocessInputBatch(self.InputBatch())

  def PreprocessInputBatch(self, batch):
    # Can be overridden by subclasses.
    return batch

  def CreateTpuFeeds(self):
    """Creates the TPU infeed queue from preprocessed batch."""
    p = self.params
    cluster = cluster_factory.Current()
    num_tpu_hosts = cluster.num_tpu_hosts
    assert num_tpu_hosts > 0, ('num_tpu_hosts: %d' % num_tpu_hosts)
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1

    with py_utils.outside_all_rewrites():
      assert py_utils.use_tpu()
      assert not self._made_tpu_infeed

      shards = tpu_function.get_tpu_context(
      ).number_of_shards // num_infeed_hosts
      input_ops_list = []
      queues = []
      first_batch = None
      for task_id in range(num_infeed_hosts):
        host_device = '/task:{}/device:CPU:0'.format(task_id)
        with tf.device(host_device):
          batch = self.GetPreprocessedInputBatch()
          if first_batch is None:
            first_batch = batch
          flat_batch = batch.FlattenItems()

          shapes, types = [], []
          for k, x in flat_batch:
            assert x.shape.is_fully_defined(), (
                'Shape must be fully defined: %s: %s' % (k, x))
            # TODO(cwhipkey): if it's a string (or other type not supported on
            # TPU), drop it from feeding and on the other end add in an op that
            # fails if used.
            shapes.append(x.shape)
            types.append(x.dtype)
          q = tf.contrib.tpu.InfeedQueue(tuple_types=types, tuple_shapes=shapes)
          queues.append(q)
          assert shards is not None
          q.set_number_of_shards(shards)

          if p.use_per_host_infeed:

            # TODO(ylc/zhifengc): Add this to a policy module and test it.
            def _tpu_ordinal_function(shard_index_in_host):
              device_assignment = py_utils.GetTpuDeviceAssignment()
              if device_assignment:
                # We put both enqueue/dequeue ops at core 0 in each replica.
                replica = device_assignment.lookup_replicas(
                    task_id, (0, 0, 0))[shard_index_in_host]  # pylint: disable=cell-var-from-loop
                return device_assignment.tpu_ordinal(replica=replica)
              else:
                return shard_index_in_host

            input_ops = q.split_inputs_and_generate_enqueue_ops(
                [v for _, v in flat_batch],
                placement_function=lambda x: host_device,  # pylint: disable=cell-var-from-loop
                tpu_ordinal_function=_tpu_ordinal_function)
          else:
            input_ops = q.split_inputs_and_generate_enqueue_ops(
                [v for _, v in flat_batch],
                device_assignment=py_utils.GetTpuDeviceAssignment())

          input_ops_list += input_ops
      tf.logging.info('input_ops_list %s', input_ops_list)
      tpu_infeed_op = tf.group(*input_ops_list)
    self._made_tpu_infeed = True
    # Let trainer.py use multiple threads to drive the infeed op.
    for _ in range(p.tpu_infeed_parallism):
      tf.add_to_collection(py_utils.ENQUEUE_OPS, tpu_infeed_op)

    with tf.device(tf.contrib.tpu.core(0)):
      tensors = queues[0].generate_dequeue_op()
    return first_batch.Pack(tensors)

  def SplitInputBatch(self, num_splits):
    """Splits the current InputBatch into num_splits ways.

    Args:
      num_splits: The number of splits.

    Returns:
      A list of NestedMaps. Each NestedMap represents the input
      tensors in one split.
    """
    assert num_splits >= 1

    batch = self.InputBatch()
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
        'file_pattern', '', 'file pattern. Typically, values are binary '
        'protocol buffers containing train/eval samples. keys are not used.')
    p.Define('file_random_seed', 301,
             'Random seed for shuffling the input data.')
    p.Define(
        'file_buffer_size', 10000,
        'How many records are buffered for random shuffling. This param '
        'affects how much RAM a train/test job needs. E.g., if an average '
        'record is about 500KB, the buffer needs 5GB ram.')
    p.Define('file_parallelism', 16, 'How many files to read concurrently.')
    p.Define('file_shuffle_shift_ratio', 0, 'DEPRECATED.')
    p.Define(
        'flush_every_n', 0, 'If non-zero, flushes all batches buffered '
        'so far every these many records are yielded.')
    p.Define('num_batcher_threads', 1, 'Number of threads to use for input '
             'record batcher.')
    return p


class BaseSequenceInputGenerator(BaseInputGeneratorFromFiles):
  """The basic sequence input generator."""

  UNK = 0  # <unk>
  SOS = 1  # <s>
  EOS = 2  # </s>

  @classmethod
  def Params(cls):
    """Defaults params for sequence input generators."""
    p = super(BaseSequenceInputGenerator, cls).Params()
    # How input should be bucketized.
    p.Define('bucket_upper_bound', [2560], 'Bucketing scheme.')
    p.Define('bucket_batch_limit', [8], 'For each bucket, desired batch size.')
    p.Define('vocab_size', 64, 'The size of the vocabuary.')
    p.Define(
        'append_eos', True, 'Whether to append </s> at the end and treat '
        'it as a non-padded label.')
    p.Define(
        'source_max_token', None,
        'The maximum number of tokens for source labels. Speech model '
        'needs the source to have a different padded length than the '
        'target.')
    p.Define('target_max_token', 300,
             'The maximum number of tokens for target labels.')
    p.Define('pad_to_max_seq_length', False,
             'If True, input tensors will be padded to p.target_max_token.')
    # TODO(ciprianchelba): there should be a check in __init__ that the ids
    # below are consistent with the ones assigned by the vocabulary.
    p.Define('target_sos_id', 1, 'Target start of sequence id.')
    p.Define('target_eos_id', 2, 'Target end of sequence id.')
    p.Define('target_unk_id', 3, 'Target end of sequence id.')
    return p

  def __init__(self, params):
    super(BaseSequenceInputGenerator, self).__init__(params)
    BaseSequenceInputGenerator.UNK = params.target_unk_id  # <UNK>
    BaseSequenceInputGenerator.SOS = params.target_sos_id  # <S>
    BaseSequenceInputGenerator.EOS = params.target_eos_id  # </S>

    # Adjust batch size according to the cluster spec.
    cluster = cluster_factory.Current()
    self._scaled_bucket_batch_limit = [
        b * cluster.num_splits_per_client
        for b in self.params.bucket_batch_limit
    ]

  @property
  def scaled_bucket_batch_limit(self):
    return self._scaled_bucket_batch_limit

  def CommonInputOpArgs(self):
    p = self.params
    cluster = cluster_factory.Current()
    bucket_batch_limit = self.scaled_bucket_batch_limit
    if p.use_per_host_infeed and cluster.num_tpu_hosts > 0:
      bucket_batch_limit = [
          x // cluster.num_tpu_hosts for x in bucket_batch_limit
      ]
      tf.logging.info('bucket_batch_limit', str(bucket_batch_limit))
    return {
        'file_pattern':
            p.file_pattern,
        'file_random_seed':
            p.file_random_seed,
        'file_shuffle_shift_ratio': (
            cluster.task * 1.0 / max(1, cluster.num_replicas)),
        'file_buffer_size':
            p.file_buffer_size,
        'file_parallelism':
            p.file_parallelism,
        'flush_every_n':
            p.flush_every_n,
        'bucket_upper_bound':
            p.bucket_upper_bound,
        'bucket_batch_limit':
            bucket_batch_limit,
        'num_threads':
            p.num_batcher_threads
    }

  def StringsToIds(self,
                   strs,
                   is_source=False,
                   key=None,
                   external_max_token=None,
                   external_append_eos=None):
    """Tokenize strs into vocab ids.

    Args:
      strs: A vector of strings.
      is_source: A bool to indicate whether to use source_max_token to pad
        'strs'.
      key: A string key in case the model has multiple vocabularies.
      external_max_token: An int providing the max_token for str. This is used
        if we have an extra set of transcripts (i.e., bias_transcripts) and
        the max_tokens is different than that of transcripts.
      external_append_eos: Bool or None. If None, will be ignored and
        params.append_eos will be used. If bool, will determine if an eos
        symbol will be added to tokens.

    Returns:
      (ids, labels, paddings): Tensors with the same shape [batch, maxlen].
      ids[i, j] is the input token id of i-th sample for j-th step.
      labels[i, j] is the target token id of i-th sample for j-th step.
      paddings[i, j] is 1 iff i-th sample's j-th step is padded.

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params

    if external_max_token:
      maxlen = external_max_token
    elif is_source:
      maxlen = p.source_max_token
    else:
      maxlen = p.target_max_token

    if external_append_eos is None:
      append_eos = p.append_eos
    else:
      append_eos = external_append_eos

    return py_x_ops.label_to_token_id(
        strs, append_eos=append_eos, maxlen=maxlen)

  def IdsToStrings(self, ids, lens, key=None):
    """Converts ids back to strings.

    Args:
      ids: A matrix of shape [batch, seqlen]. ids[i, :] is the i-th sample's
        ids.
      lens: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th sample. Only the first lens[i] tokens in ids[i, :] are valid
        tokens for the i-th sequence.
      key: A string key in case the model has multiple vocabularies.

    Returns:
      sequences: A vector of shape [batch]. The converted string sequence.

    Raises:
      ValueError: If unknown token type.
    """
    return py_x_ops.id_to_token(ids, lens)

  @property
  def sos_id(self):
    return BaseSequenceInputGenerator.SOS

  @property
  def eos_id(self):
    return BaseSequenceInputGenerator.EOS

  @property
  def unk_id(self):
    return BaseSequenceInputGenerator.UNK


class BaseTinyDatasetInput(BaseInputGenerator):
  """Input generator for tiny dataset which are stored in tf checkpoint.

  Input batch (b: batch size, h: height, w: width, d: depth):
    raw: Samples. [b, h, w, d].
    data: Preprocessed samples. [b, h, w, d].
    label: Labels. [b].
    weight: [b]. weight[i] is 1.0 if i-th sample is considered to
      be a real example. Otherwise, weight[i] is 0.0.
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
    p.Define('batch_size', 0, 'Batch size.')
    p.Define('repeat', True, 'If true, goes through the dataset repeatedly.')
    p.use_per_host_infeed = True
    return p

  def __init__(self, params):
    super(BaseTinyDatasetInput, self).__init__(params)

    # Adjust batch size according to the cluster spec.
    self._input_batch_size = tf.constant(
        self.params.batch_size *
        cluster_factory.Current().num_splits_per_client)

  def InputBatch(self):
    p = self.params
    batch_per_input = self.BatchSizePerInput(p.batch_size)

    @function.Defun()
    def ReadData():
      x, y = io_ops.restore_v2(p.ckpt, [p.data, p.label], [''] * 2,
                               [p.data_dtype, p.label_dtype])
      # Always convert to float32.
      return tf.to_float(x), tf.to_float(y)

    # Loads data and label into memory and keep it around.
    data, label = py_x_ops.cached_call(f=ReadData, T=[tf.float32, tf.float32])
    b, shape = batch_per_input, list(p.data_shape)
    data = tf.reshape(data, [-1] + shape)
    label = tf.reshape(label, [-1])
    label = py_utils.HasShape(label, [tf.shape(data)[0]])
    sample_ids = py_x_ops.random_permutation_sequence(
        num=p.num_samples,
        batch=b,
        repeat=p.repeat,
        seed=p.random_seed if p.random_seed else 0)
    self._sample_ids = sample_ids

    n = tf.shape(sample_ids)[0]
    raw = py_utils.PadOrTrimTo(tf.gather(data, sample_ids), [b] + shape)
    return py_utils.NestedMap(
        raw=raw,
        data=self._Preprocess(raw),
        label=py_utils.PadOrTrimTo(tf.gather(label, sample_ids), [b]),
        weight=py_utils.PadOrTrimTo(tf.ones([n], dtype=tf.float32), [b]))

  def _Preprocess(self, raw):
    return raw
