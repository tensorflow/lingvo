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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
import tensorflow as tf

from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import tokenizers
from lingvo.core.ops import py_x_ops


class PunctuatorInput(base_input_generator.BaseSequenceInputGenerator):
  """Reads text line by line and processes them for the punctuator task."""

  @classmethod
  def Params(cls):
    """Defaults params for PunctuatorInput."""
    p = super(PunctuatorInput, cls).Params()
    p.tokenizer = tokenizers.WpmTokenizer.Params()
    return p

  def _ProcessLine(self, line):
    """A single-text-line processor.

    Gets a string tensor representing a line of text that have been read from
    the input file, and splits it to graphemes (characters).
    We use original characters as the target labels, and the lowercased and
    punctuation-removed characters as the source labels.

    Args:
      line: a 1D string tensor.

    Returns:
      A list of tensors, in the expected order by __init__.
    """
    # Tokenize the input into integer ids.
    # tgt_ids has the start-of-sentence token prepended, and tgt_labels has the
    # end-of-sentence token appended.
    tgt_ids, tgt_labels, tgt_paddings = self.StringsToIds(
        tf.convert_to_tensor([line]))

    def Normalize(line):
      # Lowercase and remove punctuation.
      line = line.lower().translate(None, string.punctuation)
      # Convert multiple consecutive spaces to a single one.
      line = ' '.join(line.split())
      return line

    normalized_line = tf.py_func(Normalize, [line], tf.string, stateful=False)
    _, src_labels, src_paddings = self.StringsToIds(
        tf.convert_to_tensor([normalized_line]), is_source=True)
    # The model expects the source without a start-of-sentence token.
    src_ids = src_labels

    # Compute the length for bucketing.
    bucket_key = tf.to_int32(
        tf.maximum(
            tf.reduce_sum(1.0 - src_paddings),
            tf.reduce_sum(1.0 - tgt_paddings)))
    tgt_weights = 1.0 - tgt_paddings

    # Return tensors in an order consistent with __init__.
    out_tensors = [
        src_ids, src_paddings, tgt_ids, tgt_paddings, tgt_labels, tgt_weights
    ]
    return [tf.squeeze(t, axis=0) for t in out_tensors] + [bucket_key]

  def _DataSourceFromFilePattern(self, file_pattern):
    """Create the input processing op.

    Args:
      file_pattern: The file pattern to use as input.

    Returns:
      an operation that when executed, calls `_ProcessLine` on a line read
    from `file_pattern`.
    """
    return py_x_ops.generic_input(
        file_pattern=file_pattern,
        processor=self._ProcessLine,
        # Pad dimension 0 to the same length.
        dynamic_padding_dimensions=[0] * 6,
        # The constant values to use for padding each of the outputs.
        dynamic_padding_constants=[0, 1, 0, 1, 0, 0],
        **self.CommonInputOpArgs())

  @base_layer.initializer
  def __init__(self, params):
    super(PunctuatorInput, self).__init__(params)

    # Build the input processing graph.
    (self._src_ids, self._src_paddings, self._tgt_ids, self._tgt_paddings,
     self._tgt_labels, self._tgt_weights) = self._BuildDataSource()

    self._input_batch_size = tf.shape(self._src_ids)[0]
    self._sample_ids = tf.range(0, self._input_batch_size, 1)

  def InputBatch(self):
    """Returns a single batch as a `.NestedMap` to be passed to the model."""
    ret = py_utils.NestedMap()

    ret.src = py_utils.NestedMap()
    ret.src.ids = tf.cast(self._src_ids, dtype=tf.int32)
    ret.src.paddings = self._src_paddings

    ret.tgt = py_utils.NestedMap()
    ret.tgt.ids = self._tgt_ids
    ret.tgt.labels = tf.cast(self._tgt_labels, dtype=tf.int32)
    ret.tgt.weights = self._tgt_weights
    ret.tgt.paddings = self._tgt_paddings

    return ret
