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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import base_input_generator
from lingvo.core import py_utils
from lingvo.core import tokenizers
from lingvo.core.ops import py_x_ops


class LmInput(base_input_generator.BaseSequenceInputGenerator):
  """Reads tokenized plain text input such as from lm1b."""

  @classmethod
  def Params(cls):
    """Defaults params for `LmInput`."""
    p = super(LmInput, cls).Params()
    p.Define('fixed_input_shape', False, 'Fixed input shape or not.')
    p.tokenizer = tokenizers.AsciiTokenizer.Params()
    return p

  def __init__(self, params):
    params.pad_to_max_seq_length = True
    super(LmInput, self).__init__(params)
    p = self.params
    p.fixed_input_shape = p.fixed_input_shape or py_utils.use_tpu()

    text, self._word_count = self._BuildDataSource()
    self._ids, self._labels, self._paddings = self.StringsToIds(text)
    self._input_batch_size = tf.shape(self._ids)[0]
    tf.summary.histogram('examples/sequence_length',
                         tf.reduce_sum(1.0 - self._paddings, axis=1))
    self._weights = 1.0 - self._paddings
    if p.fixed_input_shape:
      if py_utils.use_tpu():
        # When flush_every_n is on, at end of each epoch, our input
        # generator can generate a batch smaller than
        # bucket_batch_limit
        assert not p.flush_every_n, 'flush_every_n is not allowed on TPU.'
        assert min(self.scaled_bucket_batch_limit) == max(
            self.scaled_bucket_batch_limit)
        bs = min(self.scaled_bucket_batch_limit)
      else:
        bs = max(self.scaled_bucket_batch_limit)

      def SetShape(x):
        x.set_shape([bs, p.target_max_length])

      SetShape(self._ids)
      SetShape(self._labels)
      SetShape(self._paddings)
      SetShape(self._weights)
      self._word_count.set_shape([bs])

  def _DataSourceFromFilePattern(self, file_pattern):

    def ReadInput(line):
      word_count = tf.size(tf.strings.split([line]))
      strlen = tf.size(tf.strings.split([line], ''))
      return line, word_count, strlen

    return py_x_ops.generic_input(
        file_pattern=file_pattern,
        processor=ReadInput,
        **self.CommonInputOpArgs())

  def InputBatch(self):
    ret = py_utils.NestedMap()
    ret.ids = self._ids
    ret.labels = self._labels
    ret.paddings = self._paddings
    ret.weights = self._weights
    ret.word_count = self._word_count
    return ret
