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
"""Speech recognition input generator."""

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import generic_input
from lingvo.core import py_utils

from tensorflow.python.ops import inplace_ops  # pylint:disable=g-direct-tensorflow-import


class AsrInput(base_input_generator.BaseSequenceInputGenerator):
  """Input generator for ASR."""

  @classmethod
  def Params(cls):
    """Defaults params for AsrInput."""
    p = super().Params()
    p.Define('frame_size', 40, 'The number of coefficients in each frame.')
    p.Define('append_eos_frame', True, 'Append an all-zero frame.')
    p.source_max_length = 3000
    return p

  def _DataSourceFromFilePattern(self, file_pattern):

    def Proc(record):
      """Parses a serialized tf.Example record."""
      features = [
          ('uttid', tf.io.VarLenFeature(tf.string)),
          ('transcript', tf.io.VarLenFeature(tf.string)),
          ('frames', tf.io.VarLenFeature(tf.float32)),
      ]
      example = tf.io.parse_single_example(record, dict(features))
      fval = {k: v.values for k, v in example.items()}
      # Reshape the flattened vector into its original time-major
      # representation.
      fval['frames'] = tf.reshape(
          fval['frames'], shape=[-1, self.params.frame_size])
      # Input duration determines the bucket.
      bucket_key = tf.cast(tf.shape(fval['frames'])[0], tf.int32)
      if self.params.append_eos_frame:
        bucket_key += 1
      tgt_ids, tgt_labels, tgt_paddings = self.StringsToIds(fval['transcript'])
      src_paddings = tf.zeros([tf.shape(fval['frames'])[0]], dtype=tf.float32)
      return [
          fval['uttid'], tgt_ids, tgt_labels, tgt_paddings, fval['frames'],
          src_paddings
      ], bucket_key

    features, bucket_keys = generic_input.GenericInput(
        file_pattern=file_pattern,
        processor=Proc,
        dynamic_padding_dimensions=[0] * 6,
        dynamic_padding_constants=[0] * 5 + [1],
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

    batch = py_utils.NestedMap()
    batch.bucket_keys = bucket_keys

    (utt_ids, tgt_ids, tgt_labels, tgt_paddings, src_frames,
     src_paddings) = features_list

    if not py_utils.use_tpu():
      batch.sample_ids = utt_ids

    src_frames, src_paddings = self._MaybePadSourceInputs(
        src_frames, src_paddings)

    # We expect src_inputs to be of shape
    # [batch_size, num_frames, feature_dim, channels].
    src_frames = tf.expand_dims(src_frames, axis=-1)

    # Convert target ids, labels, paddings, and weights from shape [batch_size,
    # 1, num_frames] to [batch_size, num_frames]
    tgt_ids = tf.squeeze(tgt_ids, axis=1)
    tgt_labels = tf.squeeze(tgt_labels, axis=1)
    tgt_paddings = tf.squeeze(tgt_paddings, axis=1)

    if p.pad_to_max_seq_length:
      assert p.source_max_length
      assert p.target_max_length

      if all(x == p.bucket_batch_limit[0] for x in p.bucket_batch_limit):
        # Set the input batch size as an int rather than a tensor.
        src_frames_shape = (self.InfeedBatchSize(), p.source_max_length,
                            p.frame_size, 1)
        src_paddings_shape = (self.InfeedBatchSize(), p.source_max_length)
        tgt_shape = (self.InfeedBatchSize(), p.target_max_length)
      else:
        tf.logging.warning(
            'Could not set static input shape since not all bucket batch sizes '
            'are the same:', p.bucket_batch_limit)
        src_frames_shape = None
        src_paddings_shape = None
        tgt_shape = None

      src_frames = py_utils.PadSequenceDimension(
          src_frames, p.source_max_length, 0, shape=src_frames_shape)
      src_paddings = py_utils.PadSequenceDimension(
          src_paddings, p.source_max_length, 1, shape=src_paddings_shape)
      tgt_ids = py_utils.PadSequenceDimension(
          tgt_ids, p.target_max_length, 0, shape=tgt_shape)
      tgt_labels = py_utils.PadSequenceDimension(
          tgt_labels, p.target_max_length, 0, shape=tgt_shape)
      tgt_paddings = py_utils.PadSequenceDimension(
          tgt_paddings, p.target_max_length, 1, shape=tgt_shape)

    batch.src = py_utils.NestedMap(src_inputs=src_frames, paddings=src_paddings)
    batch.tgt = py_utils.NestedMap(
        ids=tgt_ids,
        labels=tgt_labels,
        paddings=tgt_paddings,
        weights=1.0 - tgt_paddings)

    return batch

  def _MaybePadSourceInputs(self, src_inputs, src_paddings):
    p = self.params
    if not p.append_eos_frame:
      return src_inputs, src_paddings

    per_src_len = tf.reduce_sum(1 - src_paddings, 1)
    per_src_len += 1
    max_src_len = tf.reduce_max(per_src_len)
    input_shape = tf.shape(src_inputs)
    input_len = tf.maximum(input_shape[1], tf.cast(max_src_len, tf.int32))
    pad_steps = input_len - input_shape[1]
    src_inputs = tf.concat([
        src_inputs,
        tf.zeros(
            inplace_ops.inplace_update(input_shape, 1, pad_steps),
            src_inputs.dtype)
    ], 1)
    src_paddings = 1 - tf.sequence_mask(
        tf.reshape(per_src_len, [input_shape[0]]), tf.reshape(input_len, []),
        src_paddings.dtype)
    return src_inputs, src_paddings
