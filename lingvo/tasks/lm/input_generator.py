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
"""Language model input generator."""

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import generic_input
from lingvo.core import ops
from lingvo.core import py_utils
from lingvo.core import summary_utils
from lingvo.core import tokenizers
from lingvo.tasks.lm import tokenizer as lm_tokenizer


class LmInput(base_input_generator.BaseSequenceInputGenerator):
  """Reads tokenized plain text input such as from lm1b."""

  @classmethod
  def Params(cls):
    """Defaults params for `LmInput`."""
    p = super().Params()
    p.Define('fixed_input_shape', False, 'Fixed input shape or not.')
    p.tokenizer = tokenizers.AsciiTokenizer.Params()
    return p

  def __init__(self, params):
    params.pad_to_max_seq_length = True
    params.fixed_input_shape = params.fixed_input_shape or py_utils.use_tpu()
    super().__init__(params)

  def _DataSourceFromFilePattern(self, file_pattern, **extra_input_kwargs):

    def ReadInput(line):
      word_count = tf.size(tf.strings.split([line]))
      strlen = tf.size(tf.strings.split([line], ''))
      return [line, word_count], strlen

    features, bucket_keys = generic_input.GenericInput(
        file_pattern=file_pattern,
        processor=ReadInput,
        **self.CommonInputOpArgs())

    return self.BuildInputBatch(
        batch_size=self.InfeedBatchSize(),
        features_list=features,
        bucket_keys=bucket_keys)

  def BuildInputBatch(self, batch_size, features_list, bucket_keys=None):
    """Builds an input batch.

    Args:
      batch_size: batch size to use, defaults to infeed batch size.
      features_list: Use this list to build the batch.
      bucket_keys: If None, bucket_keys[i] is the bucketing key of the i-th
        sample.

    Returns:
      py_utils.NestedMap with feature names as keys and tensors as values.
    """
    p = self.params
    ret = py_utils.NestedMap()
    text, ret.word_count = features_list
    ret.bucket_keys = bucket_keys
    ret.ids, ret.labels, ret.paddings = self.StringsToIds(text)
    tf.summary.histogram('examples/sequence_length',
                         tf.reduce_sum(1.0 - ret.paddings, axis=1))
    ret.weights = 1.0 - ret.paddings
    if p.fixed_input_shape:
      if py_utils.use_tpu():
        # When flush_every_n is on, at end of each epoch, our input
        # generator can generate a batch smaller than
        # bucket_batch_limit
        assert not p.flush_every_n, 'flush_every_n is not allowed on TPU.'
        assert min(self.infeed_bucket_batch_limit) == max(
            self.infeed_bucket_batch_limit)
        bs = min(self.infeed_bucket_batch_limit)
      else:
        bs = max(self.infeed_bucket_batch_limit)

      def SetShape(x):
        x.set_shape([bs, p.target_max_length])

      SetShape(ret.ids)
      SetShape(ret.labels)
      SetShape(ret.paddings)
      SetShape(ret.weights)
      ret.word_count.set_shape([bs])
    return ret


class PackedTextInputGenerator(base_input_generator.BaseSequenceInputGenerator):
  """Input generator reading tf.Examples of texts with packing."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('enable_packing', False,
             'Whether to pack multiple documents on the same row.')
    p.Define('shuffle', True, 'Whether to randomly shuffle.')
    p.Define(
        'prepacking_batch_size', 1 << 14,
        'Only used when p.enable_packing is set. The batch size before we pack. '
        'Note that this does not affect post-packing batch size but may '
        'have a minor effect on how tight the packed output is.')
    p.Define(
        'dataset_type', None,
        'A dataset class constructor such as tf.data.TFRecordDataset. '
        'The class constructor must take a list of filenames and produce an '
        'object that extends tf.data.Dataset.')
    p.tokenizer = lm_tokenizer.BertTokenizer.Params()
    p.source_max_length = 512
    p.pad_to_max_seq_length = True
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.source_max_length
    if self.do_eval:
      p.shuffle = False
    if isinstance(p.file_pattern, str):
      p.file_pattern = [p.file_pattern]

  def _ProcessSingleExample(self, key, record):
    """Parses a serialized tf.Example record."""
    p = self.params
    del key
    features = tf.io.parse_single_example(
        serialized=record,
        features={
            'text':
                tf.io.FixedLenFeature((), dtype=tf.string, default_value=''),
        })
    ids, labels, paddings = self.StringsToIds(
        [features['text']], external_max_length=p.source_max_length)
    ret = py_utils.NestedMap(ids=ids, labels=labels, paddings=paddings)
    ret.weights = 1.0 - ret.paddings
    return ret

  def _DatasetOfExamples(self):
    """Returns tf.data.Dataset of tf.Examples from p.file_pattern."""
    p = self.params
    file_patterns = p.file_pattern
    weights = None
    if all([isinstance(x, tuple) for x in p.file_pattern]):
      if self.do_eval:
        raise ValueError('Sampling with weights not support for eval data.')
      file_patterns, weights = zip(*p.file_pattern)

    file_patterns = list(map(py_utils.ShardedFilePatternToGlob, file_patterns))
    tf.logging.info(f'Mixing files {file_patterns} with weights {weights}.')

    def _Load(file_pattern):
      dataset = tf.data.Dataset.list_files(
          file_pattern, shuffle=p.shuffle).interleave(
              p.dataset_type,
              cycle_length=tf.data.AUTOTUNE if p.shuffle else 1,
              num_parallel_calls=tf.data.AUTOTUNE)
      if p.shuffle:
        dataset = dataset.shuffle(p.file_buffer_size)
      dataset = dataset.repeat(1 if self.do_eval else -1)
      return dataset

    if weights is None or len(weights) <= 1:
      return _Load(file_patterns)

    return tf.data.experimental.sample_from_datasets(
        [_Load(f) for f in file_patterns], weights)

  def _GetPaddingValues(self, add_segment_fields=False):
    ret = py_utils.NestedMap(ids=0, labels=0, weights=0.0, paddings=1.0)
    if add_segment_fields:
      ret.segment_ids = 0.0
      ret.segment_pos = 0
    return ret

  def _InputBatch(self):
    """Returns tf.data.Dataset of unbatched NestedMap."""
    p = self.params
    dataset = self._DatasetOfExamples()
    dataset = dataset.map(
        self._ProcessSingleExample,
        num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    dataset = dataset.take(p.num_samples if p.num_samples > 0 else -1)

    if p.enable_packing:
      dataset = dataset.batch(
          p.prepacking_batch_size,
          drop_remainder=True,
          num_parallel_calls=tf.data.AUTOTUNE if p.shuffle else 1).map(
              self._Pack, num_parallel_calls=tf.data.AUTOTUNE).unbatch()

    dataset = dataset.batch(
        self.InfeedBatchSize(),
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE if p.shuffle else 1,
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    iterator = tf.data.make_initializable_iterator(dataset)
    self._initializer = iterator.initializer
    batch = iterator.get_next()

    if not hasattr(batch, 'segment_ids'):
      # Supply segment_ids and segment_pos with no packing.
      batch.segment_ids = 1.0 - batch.paddings
      segpos = tf.cast(tf.range(p.source_max_length), dtype=tf.float32)
      batch.segment_pos = tf.cast(batch.segment_ids * segpos, dtype=tf.int32)

    def ShapeAndCast(x):
      x = tf.ensure_shape(x, (self.InfeedBatchSize(), p.source_max_length))
      if x.dtype.is_floating:
        x = tf.cast(x, py_utils.FPropDtype(p))
      return x

    batch = batch.Transform(ShapeAndCast)

    num_samples = tf.math.reduce_max(batch.segment_ids, axis=1)
    summary_utils.scalar('examples/num_packed_samples',
                         tf.reduce_sum(num_samples))
    return batch

  def _Pack(self, batch_in):
    """Packs a given batch, which changes the batch size."""
    assert not hasattr(batch_in, 'segment_ids')
    assert not hasattr(batch_in, 'segment_pos')

    actual_seq_len = tf.math.reduce_sum(
        tf.cast(batch_in.weights, tf.int32), axis=1)
    (segment_ids, segment_pos, indices_in_input, _, _, _) = ops.pack_sequences(
        actual_seq_len,
        actual_seq_len,
        packed_batch_size=0,
        packed_src_seq_len=self.params.source_max_length,
        packed_tgt_seq_len=self.params.source_max_length)

    def ApplyPacking(x):
      return ops.apply_packing(x, 0, segment_ids, indices_in_input)

    batch_out = batch_in.DeepCopy()
    batch_out = batch_out.Transform(ApplyPacking)
    batch_out.paddings = ops.apply_packing(batch_in.paddings, 1, segment_ids,
                                           indices_in_input)
    batch_out.segment_ids = tf.cast(segment_ids, tf.float32)
    batch_out.segment_pos = segment_pos

    return batch_out

  def Reset(self, tf_session):
    tf_session.run(self._initializer)

  def Initialize(self, sess):
    sess.run(self._initializer)


class TFRecordBertInput(base_input_generator.BaseInputGenerator):
  """Input generator reading TFRecords of ids for MLPerf eval."""

  @classmethod
  def Params(cls):
    p = super().Params()
    # https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#tfrecord-features
    p.Define('input_file', None, 'String, path of an input file.')
    p.Define('max_sequence_length', 512,
             'Maximum number of tokens to be present in a single example.')
    p.Define('max_predictions_per_seq', 76,
             'Maximum number of tokens that can be masked per example.')
    p.Define('eos_token_id', 102, 'id for EOS token.')
    p.Define('shuffle', False, 'Whether to randomly shuffle.')
    p.Define('file_buffer_size', 10000000,
             'How many records are buffered for random shuffling.')
    p.Define('enable_packing', False,
             'Whether to pack multiple documents on the same row.')
    p.Define(
        'prepacking_batch_size', 1 << 14,
        'Only used when p.enable_packing is set. Batch size before packing. '
        'Note that this does not affect post-packing batch size but may '
        'have a minor effect on how tight the packed output is.')
    p.Define(
        'remove_mask', False,
        'Whether to remove masking to force masking on the fly. '
        'Should only be used on the training data.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    p.resettable = True
    if self.do_eval:
      p.shuffle = False
      p.enable_packing = False
    if isinstance(p.input_file, str):
      p.input_file = [p.input_file]

  def _ParseRecord(self, record):
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
    ret.ids = tf.tensor_scatter_nd_update(
        tensor=ret.masked_ids,
        indices=tf.reshape(masked_lm_positions, [-1, 1]),
        updates=masked_lm_ids)
    ret.masked_pos = tf.tensor_scatter_nd_update(
        tensor=tf.zeros_like(ret.masked_ids, dtype=tf.float32),
        indices=tf.reshape(masked_lm_positions, [-1, 1]),
        updates=tf.ones_like(masked_lm_ids, dtype=tf.float32))
    ret.segment_ids = tf.cast(example['input_mask'], dtype=tf.float32)

    first_eos_idx = tf.where(tf.math.equal(ret.ids, p.eos_token_id))[0][0]

    def _RemoveFirstEos(x):
      # We remove the element at position `first_eos_idx`, and pad with 0
      # to keep length unchanged.
      zero = tf.constant(0, shape=(1,), dtype=x.dtype)
      return tf.concat([x[:first_eos_idx], x[first_eos_idx + 1:], zero], axis=0)

    ret = ret.Transform(_RemoveFirstEos)
    ret.paddings = 1.0 - ret.segment_ids
    pos = tf.cast(tf.range(p.max_sequence_length), dtype=tf.float32)
    ret.segment_pos = tf.cast(ret.segment_ids * pos, dtype=tf.int32)

    if p.remove_mask:
      del ret.masked_pos
      del ret.masked_ids
    return ret

  def _PadToBatchSize(self, batch):

    def Pad(key, t):
      constant_v = 0
      if t.dtype.is_floating and key.endswith('.paddings'):
        constant_v = 1.0
      need = self.params.batch_size - py_utils.GetShape(t)[0]
      padded = tf.pad(t, [[0, need], [0, 0]], 'CONSTANT', constant_v)
      return padded

    return batch.TransformWithKey(Pad)

  def _InputBatch(self):
    """Returns a batch with .ids, .masked_ids, and .masked_pos."""
    p = self.params
    file_patterns = list(map(py_utils.ShardedFilePatternToGlob, p.input_file))
    dataset = tf.data.Dataset.list_files(
        file_patterns, shuffle=p.shuffle).interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.AUTOTUNE if p.shuffle else 1,
            num_parallel_calls=tf.data.AUTOTUNE if p.shuffle else 1)
    if p.shuffle:
      dataset = dataset.shuffle(p.file_buffer_size)
    dataset = dataset.repeat(1 if self.do_eval else -1)
    dataset = dataset.map(self._ParseRecord)

    if p.enable_packing:
      dataset = dataset.batch(
          p.prepacking_batch_size,
          drop_remainder=True,
          num_parallel_calls=tf.data.AUTOTUNE).map(
              self._Pack, num_parallel_calls=tf.data.AUTOTUNE).unbatch()

    dataset = dataset.batch(batch_size=p.batch_size, drop_remainder=p.shuffle)
    dataset = dataset.map(self._PadToBatchSize)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    iterator = tf.data.make_initializable_iterator(dataset)
    self._initializer = iterator.initializer
    batch = iterator.get_next()

    def Cast(x):
      x = tf.ensure_shape(x, [p.batch_size, p.max_sequence_length])
      if x.dtype.is_floating:
        x = tf.cast(x, py_utils.FPropDtype(p))
      return x

    return batch.Transform(Cast)

  def _Pack(self, batch_in):
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

  def Reset(self, tf_session):
    tf_session.run(self._initializer)

  def Initialize(self, sess):
    sess.run(self._initializer)
