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
from lingvo.jax import py_utils
import tensorflow.compat.v2 as tf

InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap


class TFRecordBertInput(base_input_generator.BaseInputGenerator):
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
    p.Define(
        'read_as_eval_data', False,
        'Whether to read the input as eval data: it disables features '
        'like data shuffling and repeat.')
    p.Define('eval_data_size', 10000,
             'The number of examples in the eval data. Set to 0 for unknown.')
    p.Define('file_buffer_size', 1000000,
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
    return p

  def __init__(self, p: InstantiableParams) -> None:
    if p.read_as_eval_data:
      p.resettable = True
      p.enable_packing = False
      p.remask = False
    if isinstance(p.input_file, str):
      p.input_file = [p.input_file]
    super().__init__(p)

    if p.remask:
      mlm_p = p.mlm_augmenter.Copy()
      mlm_p.dtype = tf.float32
      mlm_p.fprop_dtype = tf.float32
      logging.info('mlm_p=%s', mlm_p.ToText())
      self.CreateChild('mlm', mlm_p)

    self._dataset = self._Dataset()
    self._iterator = iter(self._dataset)

  def _ParseRecord(self, record) -> NestedMap:
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

    def _RemoveFirstEos(x):
      # We remove the element at position `first_eos_idx`, and pad with 0
      # to keep length unchanged.
      zero = tf.constant(0, shape=(1,), dtype=x.dtype)
      return tf.concat([x[:first_eos_idx], x[first_eos_idx + 1:], zero], axis=0)

    ret = ret.Transform(_RemoveFirstEos)
    ret.paddings = 1.0 - ret.segment_ids
    pos = tf.cast(tf.range(p.max_sequence_length), dtype=tf.float32)
    ret.segment_pos = tf.cast(ret.segment_ids * pos, dtype=tf.int32)

    if p.remask:
      new_masked_ids, new_masked_pos = self.mlm.FProp(None, ret.labels,
                                                      ret.paddings)
      ret.masked_ids = new_masked_ids
      ret.masked_pos = new_masked_pos
    return ret

  def _AllPaddingsBatch(self) -> NestedMap:
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

  def _PadToEvenLength(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    p = self.params
    infeed_ctx = py_utils.GetInfeedContext()
    n = infeed_ctx.num_infeed_hosts
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
        self._AllPaddingsBatch()).repeat(num_pad_batches)
    return dataset.concatenate(pad_batches)

  def _PadToBatchSize(self, batch: NestedMap) -> NestedMap:

    def Pad(key, t):
      constant_v = 0
      if t.dtype.is_floating and key.endswith('.paddings'):
        constant_v = 1.0
      need = self.params.batch_size - (t.shape[0] or tf.shape(t)[0])
      padded = tf.pad(t, [[0, need], [0, 0]], 'CONSTANT', constant_v)
      return padded

    return batch.TransformWithKey(Pad)

  def _EnsureShape(self, batch: NestedMap) -> NestedMap:
    p = self.params

    def _Cast(x):
      x = tf.ensure_shape(x, [p.batch_size, p.max_sequence_length])
      if x.dtype.is_floating:
        x = tf.cast(x, py_utils.FPropDtype(p))
      return x

    return batch.Transform(_Cast)

  @classmethod
  def ShardData(cls, dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Helper method to shard data by filtering per each infeed host."""
    infeed_ctx = py_utils.GetInfeedContext()
    if infeed_ctx.num_infeed_hosts > 1:
      shard_ids = tf.data.Dataset.range(infeed_ctx.num_infeed_hosts).repeat()
      dataset = tf.data.Dataset.zip((shard_ids, dataset))
      dataset = dataset.filter(
          lambda id, _: id == infeed_ctx.infeed_host_index).map(lambda _, x: x)
    return dataset

  def _Dataset(self) -> tf.data.Dataset:
    p = self.params
    file_patterns = list(map(py_utils.ShardedFilePatternToGlob, p.input_file))
    files = tf.data.Dataset.list_files(file_patterns, shuffle=False)
    if not p.read_as_eval_data:
      # For training data, each host will only use a non-overlapping subset
      # of the training files. The caller should make sure the files are sharded
      # evenly and divisible by the number of infeed hosts.
      files = self.ShardData(files)
      logging.info('Reading input from files: %s',
                   b', '.join(list(files.as_numpy_iterator())))

    shuffle = (not p.read_as_eval_data)
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE if shuffle else 1,
        num_parallel_calls=tf.data.AUTOTUNE if shuffle else 1)
    if shuffle:
      dataset = dataset.shuffle(p.file_buffer_size)
    dataset = dataset.repeat(-1 if shuffle else 1)
    dataset = dataset.map(self._ParseRecord)

    if p.enable_packing:
      dataset = dataset.batch(
          p.prepacking_batch_size,
          drop_remainder=True,
          num_parallel_calls=tf.data.AUTOTUNE).map(
              self._Pack,
              num_parallel_calls=tf.data.AUTOTUNE).unbatch().shuffle(
                  p.file_buffer_size)

    dataset = dataset.batch(batch_size=p.batch_size, drop_remainder=shuffle)
    if not shuffle:
      dataset = dataset.map(self._PadToBatchSize)

    if p.read_as_eval_data:
      # For the eval data, each infeed host will only see a non-overlapping
      # shard of the data, since eval data is always read sequentially.
      # We need to ensure that all hosts see an equal number of batches.
      dataset = self._PadToEvenLength(dataset)
      dataset = self.ShardData(dataset)

    dataset = dataset.map(self._EnsureShape)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

  def _InputBatch(self):
    """Returns a batch with .labels, .masked_ids, and .masked_pos."""
    # NOTE: tf.executing_eagerly() is False here.
    return self._iterator.get_next()

  def _Pack(self, batch_in: NestedMap) -> NestedMap:
    """Packs a given batch, which changes the batch size."""

    actual_seq_len = tf.math.reduce_sum(
        tf.cast(batch_in.segment_ids, tf.int32), axis=1)
    (segment_ids, segment_pos, indices_in_input, _, _, _) = ops.pack_sequences(
        actual_seq_len,
        actual_seq_len,
        packed_batch_size=0,
        packed_src_seq_len=self.params.max_sequence_length,
        packed_tgt_seq_len=self.params.max_sequence_length)

    def ApplyPacking(x):
      return ops.apply_packing(x, 0, segment_ids, indices_in_input)

    batch_out = batch_in.DeepCopy()
    batch_out = batch_out.Transform(ApplyPacking)
    batch_out.paddings = ops.apply_packing(batch_in.paddings, 1, segment_ids,
                                           indices_in_input)
    batch_out.segment_ids = tf.cast(segment_ids, tf.float32)
    batch_out.segment_pos = segment_pos

    return batch_out

  def Reset(self, session=None) -> None:
    pass
