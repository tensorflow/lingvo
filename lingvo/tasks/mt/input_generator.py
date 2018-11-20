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
"""Machine translation input generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import tokenizers
from lingvo.core.ops import py_x_ops


class NmtInput(base_input_generator.BaseSequenceInputGenerator):
  """Generator for NMT."""

  @classmethod
  def Params(cls):
    """Defaults params for `NmtInput`."""
    p = super(NmtInput, cls).Params()
    p.Define(
        'natural_order_model', True,
        'Whether the model consuming the input is a natural order model. Input '
        'is generated in natural order if True. Input is generated in reversed '
        'order if False. The value should be consistent with the underlying '
        'model. Set to True if training or using a natural order model, '
        'otherwise set to False.')
    p.tokenizer = tokenizers.VocabFileTokenizer.Params()
    p.source_max_length = 300
    return p

  def _DataSourceFromFilePattern(self, file_pattern):

    def Proc(record):
      """Parses a serialized tf.Example record."""
      outputs = [
          ('source_id', tf.VarLenFeature(tf.int64)),
          ('source_padding', tf.VarLenFeature(tf.float32)),
          ('target_id', tf.VarLenFeature(tf.int64)),
          ('target_padding', tf.VarLenFeature(tf.float32)),
          ('target_label', tf.VarLenFeature(tf.int64)),
          ('target_weight', tf.VarLenFeature(tf.float32)),
      ]
      features = tf.parse_single_example(record, dict(outputs))
      for k, v in six.iteritems(features):
        features[k] = v.values
      bucket_key = tf.to_int32(
          tf.maximum(
              tf.reduce_sum(1.0 - features['source_padding']),
              tf.reduce_sum(1.0 - features['target_padding'])))
      return [features[k] for k, _ in outputs] + [bucket_key]

    return py_x_ops.generic_input(
        file_pattern=file_pattern,
        processor=Proc,
        dynamic_padding_dimensions=[0] * 6,
        dynamic_padding_constants=[0, 1, 0, 1, 0, 0],
        **self.CommonInputOpArgs())

  @base_layer.initializer
  def __init__(self, params):
    super(NmtInput, self).__init__(params)
    p = self.params

    self.natural_order_model = p.natural_order_model

    (self._src_ids, self._src_paddings, self._tgt_ids, self._tgt_paddings,
     self._tgt_labels, self._tgt_weights) = self._BuildDataSource()

    if p.pad_to_max_seq_length:
      assert p.source_max_length

      if min(self.scaled_bucket_batch_limit) == max(
          self.scaled_bucket_batch_limit):
        source_shape = [
            min(self.scaled_bucket_batch_limit), p.source_max_length
        ]
        target_shape = [
            min(self.scaled_bucket_batch_limit), p.target_max_length
        ]
      else:
        source_shape = None
        target_shape = None
      self._src_ids = py_utils.PadSequenceDimension(
          self._src_ids, p.source_max_length, 0, source_shape)
      self._src_paddings = py_utils.PadSequenceDimension(
          self._src_paddings, p.source_max_length, 1, source_shape)
      self._tgt_ids = py_utils.PadSequenceDimension(
          self._tgt_ids, p.target_max_length, 0, target_shape)
      self._tgt_paddings = py_utils.PadSequenceDimension(
          self._tgt_paddings, p.target_max_length, 1, target_shape)
      self._tgt_labels = py_utils.PadSequenceDimension(
          self._tgt_labels, p.target_max_length, 0, target_shape)
      self._tgt_weights = py_utils.PadSequenceDimension(
          self._tgt_weights, p.target_max_length, 0, target_shape)

    # TODO(zhifengc): come up more meaningful training sample ids here.
    self._input_batch_size = tf.shape(self._src_ids)[0]
    self._sample_ids = tf.range(0, self._input_batch_size, 1)

  def InputBatch(self):
    ret = py_utils.NestedMap()

    ret.src = py_utils.NestedMap()
    ret.src.ids = tf.cast(self._src_ids, dtype=tf.int32)
    ret.src.paddings = self._src_paddings

    ret.tgt = py_utils.NestedMap()
    ret.tgt.ids = self._tgt_ids
    ret.tgt.labels = tf.cast(self._tgt_labels, dtype=tf.int32)
    ret.tgt.weights = self._tgt_weights
    ret.tgt.paddings = self._tgt_paddings

    if (self.params.fprop_dtype is None or
        self.params.dtype == self.params.fprop_dtype):
      return ret

    def _Cast(v):
      if not v.dtype.is_floating:
        return v
      return tf.cast(v, self.params.fprop_dtype)

    return ret.Transform(_Cast)
