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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core.ops import py_x_ops
from lingvo.core import wpm_encoder


class BaseTokenizer(base_layer.BaseLayer):
  """The base tokenizer."""

  @classmethod
  def Params(cls):
    """Defaults params for tokenizers."""
    p = super(BaseTokenizer, cls).Params()
    p.name = 'tokenizer'
    p.Define('vocab_size', 64, 'The size of the vocabuary.')
    p.Define(
        'append_eos', True, 'Whether to append </s> at the end and treat '
        'it as a non-padded label.')
    p.Define('pad_to_max_length', True,
             'If True, output ids will be padded to max_length.')
    # TODO(ciprianchelba): there should be a check in __init__ that the ids
    # below are consistent with the ones assigned by the vocabulary.
    p.Define('target_unk_id', 0, 'Target unknown token id.')
    p.Define('target_sos_id', 1, 'Target start of sequence id.')
    p.Define('target_eos_id', 2, 'Target end of sequence id.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseTokenizer, self).__init__(params)
    p = self.params

    self.sos_id = p.target_sos_id  # <S>
    self.eos_id = p.target_eos_id  # </S>
    self.unk_id = p.target_unk_id  # <UNK>

  def StringsToIds(self,
                   strs,
                   max_length,
                   external_append_eos=None,
                   languages=None):
    """Tokenize strs into vocab ids.

    Args:
      strs: A vector of strings.
      max_length: An int providing the max_length for strs.
      external_append_eos: Bool or None. If None, will be ignored and
        `params.append_eos` will be used. If bool, will determine if an eos
        symbol will be added to tokens.
      languages: A vector of strings with the same length as `strs`.

    Returns:
      A tuple (ids, labels, paddings) with the same shape [batch, maxlen].

      - ids[i, j] is the input token id of i-th sample for j-th step.
      - labels[i, j] is the target token id of i-th sample for j-th step.
      - paddings[i, j] is 1 iff i-th sample's j-th step is padded.

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params

    if external_append_eos is None:
      append_eos = p.append_eos
    else:
      append_eos = external_append_eos

    return self._StringsToIdsImpl(strs, max_length, append_eos, languages)

  def _StringsToIdsImpl(self, strs, max_length, append_eos, languages):
    raise NotImplementedError('Abstract method.')

  def IdsToStrings(self, ids, lens, languages=None):
    """Converts ids back to strings.

    Args:
      ids: A matrix of shape [batch, seqlen]. ids[i, :] is the i-th sample's
        ids.
      lens: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th sample. Only the first lens[i] tokens in ids[i, :] are valid tokens
        for the i-th sequence.
      languages: A vector of strings of shape [batch].

    Returns:
      sequences - A vector of shape [batch]. The converted string sequence.

    Raises:
      ValueError: If unknown token type.
    """
    raise NotImplementedError('Abstract method.')


class AsciiTokenizer(BaseTokenizer):
  """A simple grapheme tokenizer.

  Maps a small vocabulary of character tokens for (lower case) letters, digits,
  and punctuation symbols.
  """

  def _StringsToIdsImpl(self, strs, max_length, append_eos, languages):
    p = self.params
    return py_x_ops.ascii_to_token_id(
        strs,
        maxlen=max_length,
        pad_to_maxlen=p.pad_to_max_length,
        append_eos=append_eos)

  def IdsToStrings(self, ids, lens):
    return py_x_ops.id_to_ascii(ids, lens)


class VocabFileTokenizer(BaseTokenizer):
  """Tokenizers that use vocab files for look-up."""

  @classmethod
  def Params(cls):
    p = super(VocabFileTokenizer, cls).Params()
    p.Define('token_vocab_filepath', None,
             'If set, specifies a filepath to the token vocab file.')
    p.Define('ngram_vocab_filepath', None,
             'If set, specifies a filepath to the Ngram vocab file.')
    p.Define('ngram_separator', '',
             'string separator to use when joining ngrams.')
    p.Define('tokens_delimiter', ' ',
             'The delimiter to split a string to tokens with.')
    p.Define(
        'load_token_ids_from_vocab', True,
        'Whether token ids are present in vocab (i.e. vocab contains two '
        'colums, one for IDs and one for words).  If false, line numbers '
        'are used.')
    return p

  @property
  def _vocab_file_params(self):
    return ['token_vocab_filepath', 'ngram_vocab_filepath']

  def _CheckParams(self):
    p = self.params
    num_params_specified = sum(
        [getattr(p, x) is not None for x in self._vocab_file_params])
    if num_params_specified != 1:
      raise ValueError('Exactly one vocab file should be specified!')

  def _StringsToIdsImpl(self, strs, max_length, append_eos, languages):
    self._CheckParams()
    p = self.params

    if p.token_vocab_filepath:
      return py_x_ops.str_to_vocab_tokens(
          strs,
          maxlen=max_length,
          pad_to_maxlen=p.pad_to_max_length,
          append_eos=append_eos,
          vocab_filepath=p.token_vocab_filepath,
          load_token_ids_from_vocab=p.load_token_ids_from_vocab,
          delimiter=p.tokens_delimiter)
    elif p.ngram_vocab_filepath:
      raise NotImplementedError('ngram vocab StringsToIds is not supported.')

  def IdsToStrings(self, ids, lens):
    self._CheckParams()
    p = self.params
    if p.token_vocab_filepath:
      ngram_vocab_filepath = p.token_vocab_filepath
      ngram_separator = p.tokens_delimiter
    elif p.ngram_vocab_filepath:
      ngram_vocab_filepath = p.ngram_vocab_filepath
      ngram_separator = p.ngram_separator

    return py_x_ops.ngram_id_to_token(
        token_ids=ids,
        seq_lengths=lens,
        ngram_vocab_filepath=ngram_vocab_filepath,
        ngram_separator=ngram_separator)


class BpeTokenizer(BaseTokenizer):
  """Tokenizers that use BPE vocab files and word to id lists for look-up."""

  @classmethod
  def Params(cls):
    p = super(BpeTokenizer, cls).Params()
    p.Define('codes_filepath', None,
             'Specifies a filepath to the list of bpe codes vocab file.')
    p.Define('words_to_ids_filepath', None,
             'Specifies a filepath to the word bpe vocab file.')
    return p

  def _StringsToIdsImpl(self, strs, max_length, append_eos, languages):
    p = self.params

    return py_x_ops.bpe_words_to_ids(
        strs,
        maxlen=max_length,
        append_eos=append_eos,
        tokenization_filepath=p.words_to_ids_filepath)

  def IdsToStrings(self, ids, lens):
    p = self.params

    return py_x_ops.bpe_ids_to_words(
        token_ids=ids, seq_lengths=lens, vocab_filepath=p.codes_filepath)


class WpmTokenizer(BaseTokenizer):
  """Tokenizer for word-piece models."""

  @classmethod
  def Params(cls):
    p = super(WpmTokenizer, cls).Params()
    p.Define(
        'vocab_filepath', None,
        'Specifies a filepath to the WPM vocab. The vocab is sorted by '
        'descending merge score.')
    p.Define(
        'merge_prob', 1.,
        'Probability of merging WPMs. If less than 1, then decomposition '
        'of words into wordpieces will no longer be deterministic, and '
        'result in longer ID sequences. At 0, it will be graphemes.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(WpmTokenizer, self).__init__(params)
    p = self.params
    self._wpm_encoder = wpm_encoder.WpmEncoder(p.vocab_filepath, p.merge_prob)
    assert p.target_unk_id == self._wpm_encoder.unk_id
    assert p.target_sos_id == self._wpm_encoder.sentence_start_id
    assert p.target_eos_id == self._wpm_encoder.sentence_end_id

  def _StringsToIdsImpl(self, strs, max_length, append_eos, languages):
    """Takes a tensor of strings and returns id/padding tensors.

    This generates `token_ids`, `target_ids`, and `paddings` in the format that
    is expected for tokenizers. This performs padding to a fixed length and
    appends the end-of-sentence token as appropriate.

    Args:
      strs: a string Tensor.
      max_length: a python integer. The second dimension of the returned arrays.
        All sequences are padded or truncated to that length.
      append_eos: a python bool. See `BaseTokenizer` for explanation.
      languages: A vector of strings with the same length as `strs`.

    Returns:
      token_ids: a tensor of sequences of WPM ids starting with SOS. Sequences
        always end with EOS unless the sequence exceeds the maximum length.
        Always padded with EOS.
      target_ids: a tensor of sequences of WPM ids not starting with SOS
        but ending with EOS. Always padded with EOS.
      paddings: a tensor of floats indicating, at each position, whether
        the corresponding position is padded.
    """
    p = self.params
    if append_eos is None:
      append_eos = p.append_eos

    batch_size = py_utils.GetShape(strs)[0]
    token_ids_ta = tf.TensorArray(tf.int32, batch_size)
    target_ids_ta = tf.TensorArray(tf.int32, batch_size)
    paddings_ta = tf.TensorArray(tf.float32, batch_size)

    def _TokenizeOneSentence(i, strs, token_ids_ta, target_ids_ta, paddings_ta):
      """Tokenizes a single sentence."""
      ids, _ = self._wpm_encoder.Encode(strs[i])

      if append_eos:
        ids = tf.concat([ids, [self.eos_id]], axis=0)

      # This truncates after the eos is added, so some sentences might
      # not have </s> at the end.
      token_ids_ta = token_ids_ta.write(
          i,
          py_utils.PadOrTrimTo(
              tf.concat([[self.sos_id], ids], axis=0), [max_length],
              self.eos_id))
      target_ids_ta = target_ids_ta.write(
          i, py_utils.PadOrTrimTo(ids, [max_length], self.eos_id))
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
      maxlen = tf.to_int32(tf.reduce_max(tf.reduce_sum(1.0 - paddings, axis=1)))
      token_ids = token_ids[:, :maxlen]
      target_ids = target_ids[:, :maxlen]
      paddings = paddings[:, :maxlen]

    return token_ids, target_ids, paddings

  def IdsToStrings(self, ids, lens):
    """Takes integer matrices and returns vectors of strings."""
    ids = py_utils.with_dependencies([py_utils.assert_same_dim0([ids, lens])],
                                     ids)
    return tf.map_fn(
        lambda inputs: self._wpm_encoder.Decode(inputs[0][:inputs[1]]),
        (ids, lens),
        dtype=tf.string,
        parallel_iterations=30,
        back_prop=False)
