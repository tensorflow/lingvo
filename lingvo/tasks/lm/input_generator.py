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
# ==============================================================================
"""Language model input generator."""

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import generic_input
from lingvo.core import py_utils
from lingvo.core import tokenizers


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

  def _DataSourceFromFilePattern(self, file_pattern):

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
