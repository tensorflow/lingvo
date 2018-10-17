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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tensorflow.python.ops import inplace_ops
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core.ops import py_x_ops


class AsrInput(base_input_generator.BaseSequenceInputGenerator):
  """Input generator for ASR."""

  @classmethod
  def Params(cls):
    """Defaults params for AsrInput."""
    p = super(AsrInput, cls).Params()
    p.Define('frame_size', 40, 'The number of coefficients in each frame.')
    p.Define('append_eos_frame', True, 'Append an all-zero frame.')
    p.source_max_length = 3000
    return p

  def _DataSourceFromFilePattern(self, file_pattern):

    def Proc(record):
      """Parses a serialized tf.Example record."""
      features = [
          ('uttid', tf.VarLenFeature(tf.string)),
          ('transcript', tf.VarLenFeature(tf.string)),
          ('frames', tf.VarLenFeature(tf.float32)),
      ]
      example = tf.parse_single_example(record, dict(features))
      fval = {k: v.values for k, v in six.iteritems(example)}
      # Reshape the flattened vector into its original time-major
      # representation.
      fval['frames'] = tf.reshape(
          fval['frames'], shape=[-1, self.params.frame_size])
      # Input duration determines the bucket.
      bucket_key = tf.to_int32(tf.shape(fval['frames'])[0])
      if self.params.append_eos_frame:
        bucket_key += 1
      tgt_ids, tgt_labels, tgt_paddings = self.StringsToIds(fval['transcript'])
      src_paddings = tf.zeros([tf.shape(fval['frames'])[0]], dtype=tf.float32)
      return fval['uttid'], tgt_ids, tgt_labels, tgt_paddings, fval[
          'frames'], src_paddings, bucket_key

    return py_x_ops.generic_input(
        file_pattern=file_pattern,
        processor=Proc,
        dynamic_padding_dimensions=[0] * 6,
        dynamic_padding_constants=[0] * 5 + [1],
        **self.CommonInputOpArgs())

  def _MaybePadSourceInputs(self, src_inputs, src_paddings):
    p = self.params
    if not p.append_eos_frame:
      return src_inputs, src_paddings

    per_src_len = tf.reduce_sum(1 - src_paddings, 1)
    per_src_len += 1
    max_src_len = tf.reduce_max(per_src_len)
    input_shape = tf.shape(src_inputs)
    input_len = tf.maximum(input_shape[1], tf.to_int32(max_src_len))
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

  @base_layer.initializer
  def __init__(self, params):
    super(AsrInput, self).__init__(params)
    p = self.params

    (utt_ids, tgt_ids, tgt_labels, tgt_paddings, src_frames,
     src_paddings) = self._BuildDataSource()

    self._input_batch_size = tf.shape(utt_ids)[0]
    self._sample_ids = utt_ids

    src_frames, src_paddings = self._MaybePadSourceInputs(
        src_frames, src_paddings)

    # We expect src_inputs to be of shape
    # [batch_size, num_frames, feature_dim, channels].
    src_frames = tf.expand_dims(src_frames, dim=-1)

    # Convert target ids, labels, paddings, and weights from shape [batch_size,
    # 1, num_frames] to [batch_size, num_frames]
    tgt_ids = tf.squeeze(tgt_ids, axis=1)
    tgt_labels = tf.squeeze(tgt_labels, axis=1)
    tgt_paddings = tf.squeeze(tgt_paddings, axis=1)

    tgt = py_utils.NestedMap(
        ids=tgt_ids,
        labels=tgt_labels,
        paddings=tgt_paddings,
        weights=1.0 - tgt_paddings)
    src = py_utils.NestedMap(src_inputs=src_frames, paddings=src_paddings)

    self._tgt = tgt
    self._src = src

  def InputBatch(self):
    batch = py_utils.NestedMap()

    batch.src = self._src
    batch.tgt = self._tgt
    if not py_utils.use_tpu():
      batch.sample_ids = self._sample_ids

    return batch
