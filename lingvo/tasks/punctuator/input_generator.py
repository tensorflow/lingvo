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
"""Punctuator input generator."""

import string
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import datasource
from lingvo.core import py_utils
from lingvo.core import tokenizers


class TextLines(datasource.TFDatasetSource):
  """Returns a tf.data.Dataset containing lines from a text file."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('file_pattern', None, 'A file pattern string.')
    p.Define('shuffle_buffer_size', 10000,
             'Number of records buffered for random shuffling.')
    return p

  def GetDataset(self):
    p = self.params
    if not p.file_pattern:
      raise ValueError('A file pattern must be provided.')
    file_pattern_glob = py_utils.ShardedFilePatternToGlob(p.file_pattern)
    dataset = tf.data.Dataset.list_files(
        file_pattern_glob,
        shuffle=not self.cluster.require_sequential_input_order)
    dataset = tf.data.TextLineDataset(
        dataset,
        num_parallel_reads=(1 if self.cluster.in_unit_test else
                            tf.data.experimental.AUTOTUNE))

    if not self.cluster.require_sequential_input_order:
      dataset = dataset.shuffle(
          p.shuffle_buffer_size, reshuffle_each_iteration=True)
      dataset = dataset.repeat()
    return dataset


class PunctuatorInput(base_input_generator.BaseInputGenerator):
  """Reads text line by line and processes them for the punctuator task.

  Input batches are NestedMaps containing:

  - src.ids: int32 source word-piece ids of shape [batch, p.source_max_length].
  - src.paddings: float32 paddings of shape [batch, p.source_max_length] where
    paddings == 0.0 if the position is part of the input and 1.0 if the position
    is padding.
  - tgt.ids: int32 target word-piece ids of shape [batch, p.target_max_length].
  - tgt.labels = int32 target label word-piece ids of shape
    [batch, p.target_max_length]. The difference between labels and ids is that
    ids include the sos (start of sequence) token in the front and labels
    include the eos (end of sequence) token in the end.
  - tgt.paddings: float32 paddings of shape [batch, p.target_max_length].
  - tgt.weights: float32 weights of shape [batch, p.target_max_length]. Weights
    are generally 1.0 - paddings and are used for loss computation.
  - bucket_keys: int32 value used for bucketing. This is set automatically by
    TFDatasetBatchBySequenceLength, and is for debugging purposes.
  """

  @classmethod
  def Params(cls):
    """Defaults params for PunctuatorInput."""
    p = super().Params()
    p.file_datasource = TextLines.Params()
    p.Define('tokenizer', tokenizers.WpmTokenizer.Params(), 'Tokenizer params.')
    p.Define('source_max_length', None,
             'The maximum length of the source sequence.')
    p.Define('target_max_length', None,
             'The maximum length of the target sequence.')
    p.Define(
        'bucket_upper_bound', [], 'Bucketing scheme. Required to be'
        'a sorted list of integers. Examples that are longer than all bucket'
        'upper bounds are skipped.')
    p.Define(
        'bucket_batch_limit', [], 'Desired per-split batch size per bucket. '
        'Must be the same length as bucket_upper_bound.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('tokenizer', p.tokenizer)

  def CreateDatasource(self):
    p = self.params
    ds = p.file_datasource
    ds = datasource.CustomTFDatasetTransform.Params().Set(sub=ds, fn='_Process')
    # See documentation for TFDatasetBatchBySequenceLength for the specifics of
    # how the bucketing process works.
    ds = datasource.TFDatasetBatchBySequenceLength.Params().Set(
        sub=ds,
        seqlen_fn='_GetSequenceLength',
        input_shape_fn='_InputShape',
        input_padding_fn='_InputPaddingValue',
        bucket_upper_bound=p.bucket_upper_bound,
        bucket_batch_limit=p.bucket_batch_limit)
    p.file_datasource = ds
    super().CreateDatasource()

  def _Process(self, dataset):
    """Processes the dataset containing individual lines."""
    return dataset.map(self._ProcessLine, **self._map_args)

  def _ProcessLine(self, line):
    """A single-text-line processor.

    Gets a string tensor representing a line of text that have been read from
    the input file, and splits it to graphemes (characters).
    We use original characters as the target labels, and the lowercased and
    punctuation-removed characters as the source labels.

    Args:
      line: a 1D string tensor.

    Returns:
      A NestedMap containing the processed example.
    """
    p = self.params
    # Tokenize the input into integer ids.
    # tgt_ids has the start-of-sentence token prepended, and tgt_labels has the
    # end-of-sentence token appended.
    tgt_ids, tgt_labels, tgt_paddings = self.tokenizer.StringsToIds(
        tf.convert_to_tensor([line]), p.target_max_length)
    # Because StringsToIds requires a vector but _ProcessLine is called for
    # individual lines, we need to manually remove the batch dimension.
    tgt_ids = tgt_ids[0]
    tgt_labels = tgt_labels[0]
    tgt_paddings = tgt_paddings[0]

    # This normalization function produces the "source" text from which the
    # Punctuator task is trained to reproduce the original "target" text.
    def Normalize(line):
      # Lowercase and remove punctuation.
      line = line.lower().translate(None, string.punctuation.encode('utf-8'))
      # Convert multiple consecutive spaces to a single one.
      line = b' '.join(line.split())
      return line

    normalized_line = tf.py_func(Normalize, [line], tf.string, stateful=False)
    _, src_labels, src_paddings = self.tokenizer.StringsToIds(
        tf.convert_to_tensor([normalized_line]), p.source_max_length)
    # Because StringsToIds requires a vector but _ProcessLine is called for
    # individual lines, we need to manually remove the batch dimension.
    src_labels = src_labels[0]
    src_paddings = src_paddings[0]
    # The model expects the source without a start-of-sentence token.
    src_ids = src_labels
    tgt_weights = 1.0 - tgt_paddings

    ret = py_utils.NestedMap()

    ret.src = py_utils.NestedMap()
    ret.src.ids = tf.cast(src_ids, dtype=tf.int32)
    ret.src.paddings = src_paddings

    ret.tgt = py_utils.NestedMap()
    ret.tgt.ids = tgt_ids
    ret.tgt.labels = tf.cast(tgt_labels, dtype=tf.int32)
    ret.tgt.weights = tgt_weights
    ret.tgt.paddings = tgt_paddings
    return ret

  def _GetSequenceLength(self, example):
    """Returns sequence length for the example NestedMap from the dataset.

    This function is used by the TFDatasetBatchBySequenceLength DataSource to
    obtain the key used for bucketing. Bucketing separates examples into
    groups before batching, such that each batch contains only examples within a
    certain length.

    Args:
      example: A NestedMap containing an input example. Tensors in the example
        do not have a leading batch dimension.

    Returns:
      An integer sequence length for the example.
    """
    return tf.cast(
        tf.round(
            tf.maximum(
                tf.reduce_sum(1.0 - example.src.paddings),
                tf.reduce_sum(1.0 - example.tgt.paddings))), tf.int32)

  def _InputShape(self, key):
    """Returns the final shape of the tensor corresponding to key as a tuple.

    The shape should not include a leading batch dimension.

    This function is used by the TFDatasetBatchBySequenceLength DataSource to
    specify the shape for each key in an example. Because sequence examples are
    of different lengths, they need to be padded to a common shape for batching.

    Args:
      key: The NestedMap key to return shape for.
    """
    p = self.params
    if key == 'bucket_keys':
      return ()
    if key.startswith('src.'):
      return [p.source_max_length]
    if key.startswith('tgt.'):
      return [p.target_max_length]
    raise ValueError('Unexpected key %s' % key)

  def _InputPaddingValue(self, key, tensorspec):
    """Returns a scalar value to pad the tensor corresponding to key with.

    This function is used by the TFDatasetBatchBySequenceLength DataSource to
    specify the value used for padding.

    Args:
      key: The NestedMap key to return padding value for.
      tensorspec: a tf.TensorSpec describing the tensor to be padded.
    """
    if key.endswith('_paddings'):
      return tf.ones([], dtype=tensorspec.dtype)
    else:
      return tf.zeros([], dtype=tensorspec.dtype)
