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
"""Machine translation input generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import generic_input
from lingvo.core import py_utils
from lingvo.core import tokenizers
import six


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
      bucket_key = tf.cast(
          tf.maximum(
              tf.reduce_sum(1.0 - features['source_padding']),
              tf.reduce_sum(1.0 - features['target_padding'])), tf.int32)
      return [features[k] for k, _ in outputs], bucket_key

    return generic_input.GenericInput(
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
     self._tgt_labels,
     self._tgt_weights), self._bucket_keys = self._BuildDataSource()

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

    ret.bucket_keys = self._bucket_keys

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


class MlPerfInput(base_input_generator.BaseSequenceInputGenerator):
  """Generator for MLPerf TFRecords."""

  @classmethod
  def Params(cls):
    """Default params for `MlPerfInput`."""
    p = super(MlPerfInput, cls).Params()

    p.Define('natural_order_model', True, '')
    p.Define(
        'sos_id', 0, 'Start of sentence id'
        'Note in the MLPerf encoding, this is actually <PAD>, however we can '
        'make use of it since we never actually use <PAD>.')

    p.Define(
        'packed_input', False,
        'If True, then we also consume {inputs,targets}_{position,segementation}'
    )
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MlPerfInput, self).__init__(params)
    p = self.params

    self.natural_order_model = p.natural_order_model

    if not p.packed_input:
      (self._src_ids, self._src_paddings, self._tgt_ids, self._tgt_paddings,
       self._tgt_labels,
       self._tgt_weights), self._bucket_keys = self._BuildDataSource()
    else:
      (
          self._src_ids,
          self._src_paddings,
          self._tgt_ids,
          self._tgt_paddings,
          self._tgt_labels,
          self._tgt_weights,
          self._src_seg_pos,
          self._src_seg_ids,
          self._tgt_seg_pos,
          self._tgt_seg_ids,
      ), self._bucket_keys = self._BuildDataSource()

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
      self._src_ids = py_utils.PadSequenceDimension(self._src_ids,
                                                    p.source_max_length, 0,
                                                    source_shape)
      self._src_paddings = py_utils.PadSequenceDimension(
          self._src_paddings, p.source_max_length, 1, source_shape)
      self._tgt_ids = py_utils.PadSequenceDimension(self._tgt_ids,
                                                    p.target_max_length, 0,
                                                    target_shape)
      self._tgt_paddings = py_utils.PadSequenceDimension(
          self._tgt_paddings, p.target_max_length, 1, target_shape)
      self._tgt_labels = py_utils.PadSequenceDimension(self._tgt_labels,
                                                       p.target_max_length, 0,
                                                       target_shape)
      self._tgt_weights = py_utils.PadSequenceDimension(self._tgt_weights,
                                                        p.target_max_length, 0,
                                                        target_shape)

      if p.packed_input:
        self._src_seg_ids = py_utils.PadSequenceDimension(
            self._src_seg_ids, p.source_max_length, 0, source_shape)
        self._src_seg_pos = py_utils.PadSequenceDimension(
            self._src_seg_pos, p.source_max_length, 0, source_shape)
        self._tgt_seg_ids = py_utils.PadSequenceDimension(
            self._tgt_seg_ids, p.target_max_length, 0, target_shape)
        self._tgt_seg_pos = py_utils.PadSequenceDimension(
            self._tgt_seg_pos, p.target_max_length, 0, target_shape)

    self._input_batch_size = tf.shape(self._src_ids)[0]
    self._sample_ids = tf.range(0, self._input_batch_size, 1)

  def _DataSourceFromFilePattern(self, file_pattern):
    p = self._params

    def _DerivePaddingsAndIds(src_ids, tgt_labels):
      """tgt_ids is tgt_labels shifted right by one, with a SOS ID prepended."""
      tgt_ids = tf.concat([[p.sos_id], tgt_labels[:-1]], axis=0)
      src_paddings = tf.zeros(tf.shape(src_ids), dtype=tf.float32)
      tgt_paddings = tf.zeros(tf.shape(tgt_ids), dtype=tf.float32)
      tgt_weights = tf.ones(tf.shape(tgt_ids), dtype=tf.float32)

      bucket_key = tf.cast(
          tf.maximum(
              tf.reduce_sum(1.0 - src_paddings),
              tf.reduce_sum(1.0 - tgt_paddings)), tf.int32)

      return src_paddings, tgt_ids, tgt_paddings, tgt_weights, bucket_key

    def _ProcPacked(record):
      """TFExample -> Tensors for PackedInput."""
      outputs = [
          ('inputs', tf.VarLenFeature(tf.int64)),
          ('targets', tf.VarLenFeature(tf.int64)),
          ('inputs_segmentation', tf.VarLenFeature(tf.int64)),
          ('inputs_position', tf.VarLenFeature(tf.int64)),
          ('targets_segmentation', tf.VarLenFeature(tf.int64)),
          ('targets_position', tf.VarLenFeature(tf.int64)),
      ]

      features = tf.parse_single_example(record, dict(outputs))
      for k, v in six.iteritems(features):
        features[k] = v.values

      src_ids = features['inputs']
      tgt_labels = features['targets']

      src_pos = features['inputs_position']
      src_seg = features['inputs_segmentation']

      tgt_pos = features['targets_position']
      tgt_seg = features['targets_segmentation']

      src_paddings, tgt_ids, tgt_paddings, tgt_weights, bucket_key = _DerivePaddingsAndIds(
          src_ids, tgt_labels)
      return [
          src_ids,
          src_paddings,
          tgt_ids,
          tgt_paddings,
          tgt_labels,
          tgt_weights,
          src_pos,
          src_seg,
          tgt_pos,
          tgt_seg,
      ], bucket_key

    def _Proc(record):
      """Parses a serialized tf.Example record."""
      outputs = [
          ('inputs', tf.VarLenFeature(tf.int64)),
          ('targets', tf.VarLenFeature(tf.int64)),
      ]
      features = tf.parse_single_example(record, dict(outputs))
      for k, v in six.iteritems(features):
        features[k] = v.values

      src_ids = features['inputs']
      tgt_labels = features['targets']

      src_paddings, tgt_ids, tgt_paddings, tgt_weights, bucket_key = _DerivePaddingsAndIds(
          src_ids, tgt_labels)
      return [
          src_ids, src_paddings, tgt_ids, tgt_paddings, tgt_labels, tgt_weights
      ], bucket_key

    if not p.packed_input:
      return generic_input.GenericInput(
          file_pattern=file_pattern,
          processor=_Proc,
          dynamic_padding_dimensions=[0] * 6,
          dynamic_padding_constants=[0, 1, 0, 1, 0, 0],
          **self.CommonInputOpArgs())
    else:
      return generic_input.GenericInput(
          file_pattern=file_pattern,
          processor=_ProcPacked,
          dynamic_padding_dimensions=[0] * 10,
          dynamic_padding_constants=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
          **self.CommonInputOpArgs())

  def InputBatch(self):
    ret = py_utils.NestedMap()
    ret.bucket_keys = self._bucket_keys

    ret.src = py_utils.NestedMap()
    ret.src.ids = tf.cast(self._src_ids, dtype=tf.int32)
    ret.src.paddings = self._src_paddings

    ret.tgt = py_utils.NestedMap()
    ret.tgt.ids = self._tgt_ids
    ret.tgt.labels = tf.cast(self._tgt_labels, dtype=tf.int32)
    ret.tgt.weights = self._tgt_weights
    ret.tgt.paddings = self._tgt_paddings

    if self.params.packed_input:
      ret.src.segment_pos = self._src_seg_pos
      ret.src.segment_ids = self._src_seg_ids

      ret.tgt.segment_pos = self._tgt_seg_pos
      ret.tgt.segment_ids = self._tgt_seg_ids

    if (self.params.fprop_dtype is None or
        self.params.dtype == self.params.fprop_dtype):
      return ret

    def _Cast(v):
      if not v.dtype.is_floating:
        return v
      return tf.cast(v, self.params.fprop_dtype)

    return ret.Transform(_Cast)
