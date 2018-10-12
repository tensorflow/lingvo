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

  def StringsToIds(self, strs, max_length, external_append_eos=None):
    """Tokenize strs into vocab ids.

    Args:
      strs: A vector of strings.
      max_length: An int providing the max_length for strs.
      external_append_eos: Bool or None. If None, will be ignored and
        `params.append_eos` will be used. If bool, will determine if an eos
        symbol will be added to tokens.

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

    ids, labels, paddings = self._StringsToIdsImpl(strs, max_length, append_eos)
    if py_utils.use_tpu():
      batch_size = strs.shape[0]
      ids.set_shape([batch_size, max_length])
      labels.set_shape([batch_size, max_length])
      paddings.set_shape([batch_size, max_length])
    return ids, labels, paddings

  def _StringsToIdsImpl(self, strs, max_length, append_eos):
    raise NotImplementedError('Abstract method.')

  def IdsToStrings(self, ids, lens):
    """Converts ids back to strings.

    Args:
      ids: A matrix of shape [batch, seqlen]. ids[i, :] is the i-th sample's
        ids.
      lens: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th sample. Only the first lens[i] tokens in ids[i, :] are valid
        tokens for the i-th sequence.

    Returns:
      sequences - A vector of shape [batch]. The converted string sequence.

    Raises:
      ValueError: If unknown token type.
    """
    raise NotImplementedError('Abstract method.')


class SimpleTokenizer(BaseTokenizer):
  """A simple grapheme tokenizer.

  Maps a small vocabulary of character tokens for (lower case) letters, digits,
  and punctuation symbols.
  """

  def _StringsToIdsImpl(self, strs, max_length, append_eos):
    return py_x_ops.label_to_token_id(
        strs, maxlen=max_length, append_eos=append_eos)

  def IdsToStrings(self, ids, lens):
    return py_x_ops.id_to_token(ids, lens)


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

  def _StringsToIdsImpl(self, strs, max_length, append_eos):
    self._CheckParams()
    p = self.params

    if p.token_vocab_filepath:
      return py_x_ops.str_to_vocab_tokens(
          strs,
          maxlen=max_length,
          append_eos=append_eos,
          vocab_filepath=p.token_vocab_filepath,
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

  def _StringsToIdsImpl(self, strs, max_length, append_eos):
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
    p.Define('lowercase', False, 'Lowercase all text as a preprocessing step.')
    return p

  BOW_STR = b'\xe2\x96\x81'.decode('utf-8')

  @base_layer.initializer
  def __init__(self, params):
    super(WpmTokenizer, self).__init__(params)
    self._wpm_encoder = wpm_encoder.WpmEncoder(params.vocab_filepath)
    p = self.params
    assert p.target_unk_id == self._wpm_encoder.unk_id
    assert p.target_sos_id == self._wpm_encoder.sentence_start_id
    assert p.target_eos_id == self._wpm_encoder.sentence_end_id

  def _PostProcessIds(self, parsed_token_ids, max_length, append_eos):
    """Post-process token ids.

    This generates `token_ids`, `target_ids`, and `paddings` in the format that
    is expected for tokenizers. This performs padding to a fixed length and
    appends the end-of-sentence token as appropriate.

    Args:
      parsed_token_ids: a list of vectors of token ids. The vectors have
        variable length.
      max_length: a python integer. The maximum length of the returned arrays.
        That is, only up to the first `max_length` tokens will be used,
        including the `max_length`-th unless we append EOS.
      append_eos: a python bool. See `BaseTokenizer` for explanation.

    Returns:
      token_ids: a tensor of sequences of WPM ids starting with SOS. Sequences
        always end with EOS unless the sequence exceeds the maximum length.
        Always padded with EOS.
      target_ids: a tensor of sequences of WPM ids not starting with SOS
        but ending with EOS. Always padded with EOS.
      paddings: a tensor of floats indicating, at each position, whether
        the corresponding position is padded.
    """
    if append_eos is None:
      append_eos = self.params.append_eos
    eos_len = 1 if append_eos else 0
    max_seqlen = np.minimum(
        np.amax([len(ids) for ids in parsed_token_ids]) + eos_len, max_length)
    token_ids, target_ids, paddings = [], [], []
    for ids in parsed_token_ids:
      ids = list(ids)  # It was a tuple.
      if append_eos:
        ids += [self.eos_id]
      # This truncates after the eos is added, so some sentences might
      # not have </s> at the end.
      ids = ids[:np.minimum(max_seqlen, len(ids))]
      seq_len = len(ids)
      pad_len = max_seqlen - seq_len
      token_ids += [[self.sos_id] + ids + [self.eos_id] * pad_len]
      target_ids += [ids + [self.eos_id] * pad_len]
      paddings += [[0.] * seq_len + [1.] * pad_len]
    token_ids = np.matrix(token_ids)
    target_ids = np.matrix(target_ids)
    paddings = np.matrix(paddings, dtype=np.float32)
    return [token_ids, target_ids, paddings]

  def _WpmEncode(self, batch_strs, max_length, append_eos):
    """Takes a batch of python strings and returns id/padding numpy matrices."""
    token_ids = []
    for sentence in batch_strs:
      if sentence:
        text = self.BOW_STR + sentence.replace(' ', self.BOW_STR)
        encoded_ids = zip(*self._wpm_encoder.EncodeToStringAndIds(text))[1]
        token_ids += [encoded_ids]
      else:
        token_ids += [[]]
    return self._PostProcessIds(token_ids, max_length, append_eos)

  def _StringsToIdsImpl(self, strs, max_length, append_eos):
    """Takes a tensor of strings and returns id/padding tensors."""
    if self.params.lowercase:
      strs = [s.lower() for s in strs]
    return tf.py_func(self._WpmEncode, [strs, max_length, append_eos],
                      [tf.int64, tf.int64, tf.float32])

  def _WpmDecode(self, ids, lengths):
    """Takes numpy integer matrices and returns vectors of strings."""
    assert len(ids) == len(lengths)
    strs = []
    for k in range(len(ids)):
      i = ids[k][:lengths[k]]
      wpm_str = self._wpm_encoder.Decode(i)
      txt = ''.join(wpm_str).replace(self.BOW_STR, ' ')
      # Remove the first space that's inserted in front of each word.
      txt = txt.lstrip(' ')
      strs += [txt]
    return [strs]

  def IdsToStrings(self, ids, lens):
    """Takes id tensors and returns vector of `tf.string`s."""
    return tf.py_func(self._WpmDecode, [ids, lens], tf.string)
