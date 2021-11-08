# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Language model input generator."""

from absl import logging
from lingvo.core import base_input_generator
from lingvo.core import layers
from lingvo.core import ops
from lingvo.jax import base_input
from lingvo.jax import py_utils
import tensorflow.compat.v2 as tf

InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap


class TFRecordBertInput(base_input.BaseInput):
  """Input generator reading TFRecords of ids for MLPerf eval."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    # https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#tfrecord-features
    p.Define('input_file', None, 'String, path of an input file.')
    p.Define('max_sequence_length', 512,
             'Maximum number of tokens to be present in a single example.')
    p.Define('max_predictions_per_seq', 76,
             'Maximum number of tokens that can be masked per example.')
    p.Define('eos_token_id', 102, 'id for EOS token.')
    p.Define('eval_data_size', 10000,
             'The number of examples in the eval data. Set to 0 for unknown.')
    p.Define('file_buffer_size', 10000,
             'How many records are buffered for random shuffling.')
    p.Define('enable_packing', False,
             'Whether to pack multiple documents on the same row.')
    p.Define(
        'prepacking_batch_size', 1 << 14,
        'Only used when p.enable_packing is set. Batch size before packing. '
        'Note that this does not affect post-packing batch size but may '
        'have a minor effect on how tight the packed output is.')
    p.Define(
        'remask', False,
        'Whether to re-apply the masking on-the-fly. Should only be used on '
        'the training data.')
    p.Define('mlm_augmenter', layers.MaskedLmDataAugmenter.Params(),
             'params for masking. Only used when p.remask=True.')
    p.Define('num_samples', -1, 'For accounting purposes only.')
    return p

  def __init__(self, p: InstantiableParams) -> None:
    if not p.is_training:
      p.reset_for_eval = True
      p.enable_packing = False
      p.remask = False
    if isinstance(p.input_file, str):
      p.input_file = [p.input_file]
    super().__init__(p)

    if p.remask:
      mlm_p = p.mlm_augmenter.Copy()
      mlm_p.name = 'mlm_augmenter'
      mlm_p.dtype = tf.float32
      mlm_p.fprop_dtype = tf.float32
      logging.info('mlm_p=%s', mlm_p.ToText())
      self.mlm = mlm_p.Instantiate()

    self._dataset = self._gen_dataset()
    self._iterator = iter(self._dataset)

  def get_next(self) -> NestedMap:
    """Returns a batch with .labels, .masked_ids, and .masked_pos."""
    ret = self._iterator.get_next()
    return tf.nest.map_structure(lambda x: x.numpy(), ret)

  def reset(self) -> None:
    self._iterator = iter(self._dataset)

  def _parse_record(self, record) -> NestedMap:
    """Reads and parses a single record."""
    p = self.params
    name_to_features = {
        'input_ids':
            tf.io.FixedLenFeature([p.max_sequence_length], tf.int64),
        'input_mask':
            tf.io.FixedLenFeature([p.max_sequence_length], tf.int64),
        'masked_lm_positions':
            tf.io.FixedLenFeature([p.max_predictions_per_seq], tf.int64),
        'masked_lm_ids':
            tf.io.FixedLenFeature([p.max_predictions_per_seq], tf.int64),
        'masked_lm_weights':
            tf.io.FixedLenFeature([p.max_predictions_per_seq], tf.float32),
    }
    example = tf.io.parse_single_example(record, name_to_features)
    mask_length = tf.cast(
        tf.reduce_sum(example['masked_lm_weights']), dtype=tf.int32)
    masked_lm_positions = tf.slice(example['masked_lm_positions'], [0],
                                   [mask_length])
    masked_lm_ids = tf.cast(
        tf.slice(example['masked_lm_ids'], [0], [mask_length]), dtype=tf.int32)
    ret = py_utils.NestedMap()
    ret.masked_ids = tf.cast(example['input_ids'], dtype=tf.int32)
    # Get back non-masked, original ids.
    ret.labels = tf.tensor_scatter_nd_update(
        tensor=ret.masked_ids,
        indices=tf.reshape(masked_lm_positions, [-1, 1]),
        updates=masked_lm_ids)
    ret.masked_pos = tf.tensor_scatter_nd_update(
        tensor=tf.zeros_like(ret.masked_ids, dtype=tf.float32),
        indices=tf.reshape(masked_lm_positions, [-1, 1]),
        updates=tf.ones_like(masked_lm_ids, dtype=tf.float32))
    ret.segment_ids = tf.cast(example['input_mask'], dtype=tf.float32)

    first_eos_idx = tf.where(tf.math.equal(ret.labels, p.eos_token_id))[0][0]

    def remove_first_eos(x):
      # We remove the element at position `first_eos_idx`, and pad with 0
      # to keep length unchanged.
      zero = tf.constant(0, shape=(1,), dtype=x.dtype)
      return tf.concat([x[:first_eos_idx], x[first_eos_idx + 1:], zero], axis=0)

    ret = ret.Transform(remove_first_eos)
    ret.paddings = 1.0 - ret.segment_ids
    pos = tf.cast(tf.range(p.max_sequence_length), dtype=tf.float32)
    ret.segment_pos = tf.cast(ret.segment_ids * pos, dtype=tf.int32)

    if p.remask:
      new_masked_ids, new_masked_pos = self.mlm.FProp(None, ret.labels,
                                                      ret.paddings)
      ret.masked_ids = new_masked_ids
      ret.masked_pos = new_masked_pos
    return ret

  def _all_paddings_batch(self) -> NestedMap:
    p = self.params
    shape = [p.batch_size, p.max_sequence_length]
    ret = py_utils.NestedMap()
    ret.labels = tf.zeros(shape, dtype=tf.int32)
    ret.masked_ids = ret.labels
    ret.segment_pos = ret.labels
    ret.masked_pos = tf.zeros(shape, dtype=tf.float32)
    ret.segment_ids = ret.masked_pos
    ret.paddings = 1.0 - ret.segment_ids
    return ret

  def _pad_to_even_length(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    p = self.params
    n = p.num_infeed_hosts
    if n <= 1:
      return dataset
    # pad with all paddings batch so that the total number of elements in
    # `dataset` can be evenly divided by n.
    if p.eval_data_size < 1:
      # dataset.cardinality() returns unknown, so we first materialize all
      # data.
      total_batches = len(list(dataset.as_numpy_iterator()))
    else:
      total_batches = (p.eval_data_size + p.batch_size - 1) // p.batch_size
    if total_batches % n == 0:
      return dataset
    per_host_batches = (total_batches + n - 1) // n
    num_pad_batches = per_host_batches * n - total_batches
    pad_batches = tf.data.Dataset.from_tensors(
        self._all_paddings_batch()).repeat(num_pad_batches)
    return dataset.concatenate(pad_batches)

  def _pad_to_batch_size(self, batch: NestedMap) -> NestedMap:

    def pad(key, t):
      constant_v = 0
      if t.dtype.is_floating and key.endswith('.paddings'):
        constant_v = 1.0
      need = self.params.batch_size - (t.shape[0] or tf.shape(t)[0])
      padded = tf.pad(t, [[0, need], [0, 0]], 'CONSTANT', constant_v)
      return padded

    return batch.TransformWithKey(pad)

  def _ensure_shape(self, batch: NestedMap) -> NestedMap:
    p = self.params

    def ensure(x):
      x = tf.ensure_shape(x, [p.batch_size, p.max_sequence_length])
      return x

    return batch.Transform(ensure)

  def _gen_dataset(self) -> tf.data.Dataset:
    p = self.params
    file_patterns = list(
        map(py_utils.sharded_file_pattern_to_glob, p.input_file))
    files = tf.data.Dataset.list_files(file_patterns, shuffle=False)
    if p.is_training:
      # For training data, each host will only use a non-overlapping subset
      # of the training files.
      # This logic is specific to the mlperf training data, which has exactly
      # 1024 shards. Other implementations might opt to shard after reading
      # all input files, in which case one must not shuffle before sharding.
      num_files = len(list(files.as_numpy_iterator()))
      if num_files % p.num_infeed_hosts != 0:
        raise ValueError(
            'Input files sharding not supported: we require the number of files'
            f' {num_files} to evenly divide num_infeed_hosts='
            f'{p.num_infeed_hosts} so we can shard at file level.')
      files = files.shard(
          num_shards=p.num_infeed_hosts, index=p.infeed_host_index)
      logging.info('Reading input from files: %s',
                   b', '.join(list(files.as_numpy_iterator())))

    shuffle = p.is_training
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE if shuffle else 1,
        num_parallel_calls=tf.data.AUTOTUNE if shuffle else 1)
    if shuffle:
      dataset = dataset.shuffle(p.file_buffer_size, seed=p.input_random_seed)
    dataset = dataset.repeat(-1 if shuffle else 1)
    dataset = dataset.map(self._parse_record)

    if p.enable_packing:
      dataset = dataset.batch(
          p.prepacking_batch_size,
          drop_remainder=True,
          num_parallel_calls=tf.data.AUTOTUNE).map(
              self._pack,
              num_parallel_calls=tf.data.AUTOTUNE).unbatch().shuffle(
                  p.file_buffer_size, seed=p.input_random_seed)

    dataset = dataset.batch(batch_size=p.batch_size, drop_remainder=shuffle)
    if not shuffle:
      dataset = dataset.map(self._pad_to_batch_size)

    if not p.is_training:
      # For the eval data, each infeed host will only see a non-overlapping
      # shard of the data, since eval data is always read sequentially.
      # We need to ensure that all hosts see an equal number of batches.
      dataset = self._pad_to_even_length(dataset)
      dataset = dataset.shard(
          num_shards=p.num_infeed_hosts, index=p.infeed_host_index)

    dataset = dataset.map(self._ensure_shape)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

  def _pack(self, batch_in: NestedMap) -> NestedMap:
    """Packs a given batch, which changes the batch size."""

    actual_seq_len = tf.math.reduce_sum(
        tf.cast(batch_in.segment_ids, tf.int32), axis=1)
    (segment_ids, segment_pos, indices_in_input, _, _, _) = ops.pack_sequences(
        actual_seq_len,
        actual_seq_len,
        packed_batch_size=0,
        packed_src_seq_len=self.params.max_sequence_length,
        packed_tgt_seq_len=self.params.max_sequence_length)

    def apply_packing(x):
      return ops.apply_packing(x, 0, segment_ids, indices_in_input)

    batch_out = batch_in.DeepCopy()
    batch_out = batch_out.Transform(apply_packing)
    batch_out.paddings = ops.apply_packing(batch_in.paddings, 1, segment_ids,
                                           indices_in_input)
    batch_out.segment_ids = tf.cast(segment_ids, tf.float32)
    batch_out.segment_pos = segment_pos

    return batch_out


class SyntheticLmData(base_input_generator.BaseInputGenerator):
  """Generated synthetic data with packed_input lm formats."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Defaults params for input generators."""
    p = super().Params()
    p.Define('seq_len', 0, 'Number of tokens in one example')
    return p

  def _InputBatch(self):
    p = self.params
    targets = tf.ones([p.batch_size, p.seq_len], dtype=tf.int32)
    input_batch = py_utils.NestedMap()
    input_batch.ids = targets  # equivalent to tf.roll(targets, 1, axis=1)
    input_batch.paddings = tf.zeros_like(targets)
    input_batch.weights = tf.ones_like(targets)
    input_batch.labels = targets
    # segment_id = 0 meant padded tokens
    # e.g., if we have two segments packed into one sentence with paddings
    # segment_ids = 1, 1, 1, 1, 2, 2, 2, 2, 0, 0
    # segment_pos = 0, 1, 2, 3, 0, 1, 2, 3, 0, 0
    input_batch.segment_ids = targets
    input_batch.segment_pos = tf.tile(
        tf.range(0, p.seq_len)[tf.newaxis, :], [p.batch_size, 1])
    return input_batch
