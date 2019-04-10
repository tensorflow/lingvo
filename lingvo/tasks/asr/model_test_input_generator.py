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
"""Simple input generator used for ASR model tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import base_input_generator
from lingvo.core import py_utils


class TestInputGenerator(base_input_generator.BaseSequenceInputGenerator):
  """A simple InputGenerator that delegate requests to another obj."""

  @classmethod
  def Params(cls):
    p = super(TestInputGenerator, cls).Params()
    p.random_seed = 20349582
    p.Define('feature_dims', 240, 'Feature dims')
    p.Define('num_channels', 3, 'Data are preprocessed into these many '
             'channels per timestep. E.g., feature_dims=240 is actually '
             '[80, 3], i.e., 3 channels, each with a 40-dim feature vector.')
    p.Define('source_shape', [2, 10, 8, 3], 'source shape.')
    p.Define('target_shape', [2, 5], 'targets shape.')
    p.Define('fixed_target_labels', None,
             'If not None, use these as the targets instead of generating '
             'random targets and set target padding to 0.  Must have same '
             'shape as target_shape.')
    p.Define('cur_iter_in_seed', True, 'use current iter value in seed '
             'computation.')
    p.Define('integer_source_max', None, 'Generate integers as source values '
             'with this value as an upper bound.')
    p.Define(
        'float_source_max', None, 'Generate floats as source values '
        'with this value as an upper bound.')
    p.Define('for_mt', False, 'True if this is for mt models; '
             'this affects some parts of batch generation')
    p.Define('target_key', '', 'If non-empty, targets will be specified in '
             'batch.additional_tgts[target_key] instead of batch.tgt.')
    p.Define('target_key_target_shape', [2, 5], 'Shape of the targets stored '
             'batch.additional_tgts[target_key].')
    p.Define('set_tgt_and_additional_tgts', False, 'If true, '
             'both batch.tgt and batch.additional_tgts[target_key] will '
             'be set. target_key_target_shape must be specified.')
    p.Define('target_language', 'ENGLISH',
             'The target language. Both language name (e.g. "ENGLISH") and '
             'language code (e.g. "zh-CN") are acceptted.')
    p.Define('align_label_with_frame', False,
             'Whether to generate label-frame alignments.')
    p.Define(
        'bprop_filters', [], 'If set, simulates a multi source'
        'input and sets filters for each source i.e the first filter'
        'corresponds to the first source etc. The number of sources is set'
        'to the length of this param.')
    p.Define(
        'number_sources', None, 'Integer which specifies the number of'
        'sources. Cannot be used along with bprop_filters.')
    p.Define(
        'source_selected', None, 'Integer which specifies the index of the'
        'source selected. Corresponds to the data source that would be'
        'sampled by the input_generator when given multiple file_patterns.'
        'This has an effect only when number_sources is set and greater than 1.'
        'Can use either constant values or a tensor like'
        'tf.mod(tf.train.get_or_create_global_step(), num_sources)')
    p.Define('target_transcript', 'dummy_transcript',
             'Text to use for transcript.')
    return p

  def __init__(self, params):
    super(TestInputGenerator, self).__init__(params)
    p = self.params
    self._bprop_variable_filters = ['']
    self._bprop_onehot = tf.constant([1], dtype=tf.float32)
    if p.target_key and not p.target_key_target_shape:
      raise ValueError('target_key_target_shape must be set when '
                       'target_key (%s) is not empty.' % p.target_key)
    if (p.set_tgt_and_additional_tgts and
        (p.target_key_target_shape[0] != p.target_shape[0])):
      raise ValueError('The first dimension of target_key_target_shape (%d) '
                       'should match the first dimension of target_shape '
                       '(%d) when both have to be set.' %
                       (p.target_key_target_shape[0], p.target_shape[0]))
    self._cur_iter = 0
    if p.bprop_filters and p.number_sources:
      raise ValueError(
          'Number of sources will be set to length of bprop_filters, the param'
          'number_sources should not be used when bprop_filters is set.')
    number_sources = p.number_sources
    if p.bprop_filters:
      self._bprop_variable_filters = p.bprop_filters
      number_sources = len(p.bprop_filters)
    if number_sources and number_sources > 1:
      self._bprop_onehot = tf.one_hot(
          p.source_selected, number_sources, dtype=tf.float32)

  def _check_paddings(self, paddings):
    with tf.name_scope('check_paddings'):
      unpacked_paddings = tf.unstack(paddings)

      non_decr = []
      for t in unpacked_paddings:
        non_d = tf.is_non_decreasing(t)
        non_decr.append(non_d)
      all_non_decr = tf.stack(non_decr)

      paddings = py_utils.with_dependencies([
          tf.assert_equal(
              tf.reduce_any(tf.equal(paddings, 0.0)),
              True,
              message='must have at least one zero value.'),
          tf.assert_equal(all_non_decr, True, message='must be non-decreasing')
      ], paddings)
      return paddings

  def GetBpropParams(self):
    return self._bprop_params

  def GetBpropType(self):
    """Get the current bprop type of the input generator batch."""
    return self._bprop_onehot

  def SampleIds(self):
    p = self.params
    if p.cur_iter_in_seed:
      random_seed = p.random_seed * 2000 * self._cur_iter
    else:
      random_seed = p.random_seed * 2000
    return tf.as_string(tf.random_uniform(p.target_shape[:1], seed=random_seed))

  def _Sources(self):
    p = self.params
    if p.cur_iter_in_seed:
      self._cur_iter += 1

    if p.integer_source_max:
      inputs = tf.random_uniform(
          p.source_shape,
          maxval=p.integer_source_max,
          dtype=tf.int32,
          seed=p.random_seed + 1000 * self._cur_iter)
    elif p.float_source_max:
      inputs = tf.random_uniform(
          p.source_shape,
          maxval=p.float_source_max,
          seed=p.random_seed + 1000 * self._cur_iter)
    else:
      inputs = tf.random_normal(
          p.source_shape, seed=p.random_seed + 1000 * self._cur_iter)

    paddings = tf.cast(
        tf.cumsum(
            tf.random_uniform(
                p.source_shape[:2], seed=p.random_seed + 1001 * self._cur_iter),
            axis=1) > 0.5 * p.source_shape[1], tf.float32)

    paddings = self._check_paddings(paddings)

    return inputs, paddings

  def _Targets(self, target_shape):
    p = self.params
    if p.cur_iter_in_seed:
      self._cur_iter += 1
    random_seed = p.random_seed * 2000 * self._cur_iter
    tids = tf.cast(
        tf.random_uniform(target_shape, seed=random_seed) *
        p.tokenizer.vocab_size, tf.int32)
    if p.fixed_target_labels is None:
      tlabels = tf.cast(
          tf.random_uniform(target_shape, seed=random_seed + 1) *
          p.tokenizer.vocab_size, tf.int32)
      tpaddings = tf.cast(
          tf.cumsum(
              tf.random_uniform(
                  target_shape[:2], seed=p.random_seed + 1001 * self._cur_iter),
              axis=1) > 0.4 * target_shape[1], tf.float32)
      tpaddings = self._check_paddings(tpaddings)
    else:
      tlabels = p.fixed_target_labels
      assert tlabels.shape_as_list() == target_shape
      tpaddings = tf.constant(0.0, shape=target_shape)
    tweights = 1.0 - tpaddings
    d = {
        'ids': tids,
        'labels': tlabels,
        'weights': tweights,
        'paddings': tpaddings
    }
    if not p.for_mt:
      d['transcripts'] = tf.constant(
          p.target_transcript, shape=[target_shape[0]])
    if p.align_label_with_frame:
      source_len = p.source_shape[1]
      d['alignments'] = tf.cast(
          tf.random_uniform(target_shape, seed=p.random_seed) * source_len,
          tf.int32)
    return d

  def GlobalBatchSize(self):
    p = self.params
    return tf.constant(p.target_shape[0])

  def InputBatch(self):
    p = self.params
    ret = py_utils.NestedMap()

    ret.src = py_utils.NestedMap()
    input_name = 'ids' if p.for_mt else 'src_inputs'
    ret.src[input_name], ret.src.paddings = self._Sources()

    # Set tgts only when needed: If target_key is specified, and both tgt and
    # additional_tgts are not needed, we only set additional_tgts. This is
    # useful when testing a model that solely uses additional_tgts instead
    # of tgt.
    if not p.target_key or p.set_tgt_and_additional_tgts:
      ret.tgt = py_utils.NestedMap(self._Targets(p.target_shape))
    else:
      ret.tgt = None
    if p.target_key:
      ret.additional_tgts = py_utils.NestedMap()
      ret.additional_tgts[p.target_key] = py_utils.NestedMap(
          self._Targets(p.target_key_target_shape))
    ret.sample_ids = self.SampleIds()

    # Cast floating point tensors to the fprop dtype (default: float32).
    def _CastFloats(v):
      if v is None:
        return None
      return tf.cast(v, py_utils.FPropDtype(p)) if v.dtype.is_floating else v

    return ret.Transform(_CastFloats)

  def _GetSourceInputsAndLabels(self, data_source):
    p = self.params
    src_inputs, src_paddings, labels = data_source
    # The data are laid out in the channel-major order. In order to move channel
    # to the last dimension, a tf.transpose of the data is needed.
    src_inputs = tf.transpose(
        tf.reshape(
            src_inputs,
            tf.concat([tf.shape(src_inputs)[:-1], [p.num_channels, -1]], 0)),
        [0, 1, 3, 2])
    return src_inputs, src_paddings, labels

  def SetBpropType(self):
    """Get the current bprop type of the input generator batch."""
    self._bprop_index = tf.one_hot(1, 2)
