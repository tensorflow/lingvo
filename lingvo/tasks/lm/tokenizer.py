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
"""Tokenizers."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import tokenizers

import tensorflow_text as tf_text


class BertTokenizer(tokenizers.BaseTokenizer):
  """A WordPiece tokenizer used by BERT."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('vocab_path', None, 'Path to the vocab file.')
    p.Define(
        'prepend_sos', False,
        'Whether to always prepend SOS token. This makes ids and target '
        'returned identical and effective length longer by 1.')
    p.target_unk_id = 100
    p.target_sos_id = 101
    p.target_eos_id = 102
    return p

  def __init__(self, params):
    super().__init__(params)
    self._tokenizer = tf_text.BertTokenizer(
        params.vocab_path,
        lower_case=True,
        max_bytes_per_word=200,
        token_out_type=tf.int32,
    )

  def _StringsToIdsImpl(self, strs, max_length, append_eos, languages):
    del languages
    p = self.params
    if append_eos is None:
      append_eos = p.append_eos

    batch_size = py_utils.GetShape(strs)[0]
    token_ids_ta = tf.TensorArray(tf.int32, batch_size)
    target_ids_ta = tf.TensorArray(tf.int32, batch_size)
    paddings_ta = tf.TensorArray(tf.float32, batch_size)

    def _TokenizeOneSentence(i, text, token_ids_ta, target_ids_ta, paddings_ta):
      """Tokenizes a single sentence."""
      if tf.is_tensor(i):
        text_i = tf.gather(text, i)
      else:
        text_i = text[i]
      ids = self._tokenizer.tokenize(text_i).merge_dims(0, -1)
      ids.set_shape([None])

      if append_eos:
        ids = tf.concat([ids, [self.eos_id]], axis=0)
      sos_ids = tf.concat([[self.sos_id], ids], axis=0)
      if p.prepend_sos:
        ids = sos_ids

      # This truncates after the EOS is added, so some sentences might
      # not have EOS at the end.
      token_ids_ta = token_ids_ta.write(
          i, py_utils.PadOrTrimTo(sos_ids, [max_length], 0))
      target_ids_ta = target_ids_ta.write(
          i, py_utils.PadOrTrimTo(ids, [max_length], 0))
      paddings_ta = paddings_ta.write(
          i,
          py_utils.PadOrTrimTo(
              tf.zeros_like(ids, dtype=tf.float32), [max_length], 1.))

      return i + 1, strs, token_ids_ta, target_ids_ta, paddings_ta

    _, _, token_ids_ta, target_ids_ta, paddings_ta = tf.while_loop(
        lambda i, *_: i < batch_size,
        _TokenizeOneSentence,
        loop_vars=(tf.constant(0, tf.int32), strs, token_ids_ta, target_ids_ta,
                   paddings_ta),
        parallel_iterations=30,
        back_prop=False)

    token_ids = token_ids_ta.stack()
    target_ids = target_ids_ta.stack()
    paddings = paddings_ta.stack()

    if not p.pad_to_max_length:
      maxlen = tf.cast(
          tf.round(tf.reduce_max(tf.reduce_sum(1.0 - paddings, axis=1))),
          tf.int32)
      token_ids = token_ids[:, :maxlen]
      target_ids = target_ids[:, :maxlen]
      paddings = paddings[:, :maxlen]

    return token_ids, target_ids, paddings

  def IdsToStrings(self, ids, lens):
    """Takes int32 token ids and returns approximate detokenized strings."""
    ids = py_utils.with_dependencies([py_utils.assert_same_dim0([ids, lens])],
                                     ids)

    def _ProcessRow(inputs):
      length = inputs[1]
      ids = tf.reshape(inputs[0][:length], [1, -1])
      tokens = self._tokenizer.detokenize(ids)
      return tf.strings.reduce_join(tokens.flat_values, separator=' ')

    return tf.map_fn(
        _ProcessRow, (ids, lens),
        dtype=tf.string,
        parallel_iterations=30,
        back_prop=False)
