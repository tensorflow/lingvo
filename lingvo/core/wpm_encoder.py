# -*- coding: utf-8 -*-
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
"""Encode using wordpiece models.

Implements the segmentation algorithm described in the last paragraph of
p. 5150, in the following publication:

M. Schuster and K. Nakajima, "Japanese and Korean voice
search," 2012 IEEE International Conference on Acoustics,
Speech and Signal Processing, 2012

https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

from lingvo.core.ops import py_x_ops

# Must be a large ID.
NO_TOKEN = 1 << 31 - 1
NO_TOKEN_STRING = '<unk>'

SENTENCE_START_STRING = '<s>'
SENTENCE_END_STRING = '</s>'

BOW_STR = '▁'


class WpmEncoder(object):

  def __init__(self, wpm_filepath, merge_prob=1.):
    """Create a WPM encoder.

    Args:
      wpm_filepath: a path to the file containing the vocabulary.
      merge_prob: the probability of merging tokens while encoding.
    """
    # Load vocabulary file.
    self._pieces = []
    with tf.gfile.Open(wpm_filepath, 'r') as f:
      for line in f.readlines():
        line = line.decode('utf-8')
        piece = line.strip().split('\t')[0]
        self._pieces.append(piece)
    self._merge_prob = merge_prob

  def _TokenToString(self, token):
    return py_x_ops.vocab_id_to_token(token, vocab=self._pieces)

  def _StringToToken(self, tokstr):
    return tf.where(
        py_x_ops.token_in_vocab(tokstr, vocab=self._pieces),
        py_x_ops.vocab_token_to_id(tokstr, vocab=self._pieces),
        tf.broadcast_to(NO_TOKEN, tf.shape(tokstr)))

  def _MergeTokens(self, tokens):
    return self._StringToToken(
        self._TokenToString(tokens[0]) + self._TokenToString(tokens[1]))

  def _EncodeToIds(self, word):
    # Below:
    #   * a token is a wordpiece ID.
    #   * the tokens array will be merged in-place.
    #   * the candidates array is an array of size len(tokens) - 1.
    #     It contains the token for the merged wordpiece, if it exists,
    #     -1 otherwise. For instance, candidate[3] = id(token[3] + token[4]).
    # First, split into basic UTF-8 characters (letters).
    chars = tf.strings.unicode_split(word, 'UTF-8')
    tokens = self._StringToToken(chars)
    tokens = tf.where(
        tf.equal(tokens, NO_TOKEN),
        # Unseen character.
        tf.broadcast_to(self.unk_id, tf.shape(tokens)),
        tokens)
    # Create initial candidate list.
    candidates = tf.map_fn(
        self._MergeTokens, (tokens[:-1], tokens[1:]), dtype=tokens.dtype)

    def _ShouldMerge(unused_tokens, candidates):
      """Merge until not possible, or we abort early according to merge_prob."""
      return tf.logical_and(
          tf.reduce_any(tf.not_equal(candidates, NO_TOKEN)),
          tf.random.uniform([]) < self._merge_prob)

    def _MergeOneToken(tokens, i):
      return tf.expand_dims(
          self._MergeTokens((tokens[i], tokens[i + 1])), axis=-1)

    def _MergeCandidates(tokens, candidates):
      """Merge in the reverse binary tree."""
      best_id = tf.argmin(candidates, output_type=tf.int32)
      # Perform the merge at position best_id.
      tokens = tf.concat(
          [tokens[:best_id], [candidates[best_id]], tokens[best_id + 2:]],
          axis=0)
      # Recompute the merge candidates.
      # Only the neighbors of best_id need to be recomputed.
      empty = tf.zeros([0], dtype=candidates.dtype)

      def _MergeLeft():
        return tf.concat(
            [candidates[:best_id - 1],
             _MergeOneToken(tokens, best_id - 1)],
            axis=0)

      left_candidates = tf.cond(tf.equal(best_id, 0), lambda: empty, _MergeLeft)

      def _MergeRight():
        return tf.concat(
            [_MergeOneToken(tokens, best_id), candidates[best_id + 2:]], axis=0)

      right_candidates = tf.cond(
          tf.greater_equal(best_id,
                           tf.size(tokens) - 1), lambda: empty, _MergeRight)

      candidates = tf.concat([left_candidates, right_candidates], axis=0)
      return tokens, candidates

    return tf.while_loop(
        _ShouldMerge,
        _MergeCandidates, (tokens, candidates),
        parallel_iterations=1,
        back_prop=False)[0]

  def Encode(self, text):
    """Converts string `text` to integer ids and the encoded string.

    Encoding includes prefixing the beginning-of-word token to each word.

    Returns:
      ids: the encoded integer ids.
      tokens: the encoded string.
    """
    words = tf.sparse.to_dense(tf.strings.split([text]), default_value='')[0]
    num_words = tf.size(words)
    ids_ta = tf.TensorArray(tf.int32, 0, dynamic_size=True)

    def _WordsToIds(i, words, ids_ta):
      encoded_ids = self._EncodeToIds(BOW_STR + words[i])
      ids_ta = ids_ta.scatter(
          tf.range(ids_ta.size(),
                   ids_ta.size() + tf.size(encoded_ids)), encoded_ids)
      return i + 1, words, ids_ta

    _, _, ids_ta = tf.while_loop(
        lambda i, *_: i < num_words,
        _WordsToIds,
        loop_vars=(tf.constant(0, tf.int32), words, ids_ta),
        parallel_iterations=30,
        back_prop=False)

    ids = ids_ta.stack()
    return ids, self._TokenToString(ids)

  def Decode(self, ids):
    txt = tf.strings.reduce_join(self._TokenToString(ids))
    txt = tf.strings.regex_replace(txt, BOW_STR, ' ')
    # Note that this strips spaces from the end of the input as well.
    # We assume no inputs rely on the existence of trailing whitespace.
    txt = tf.strings.strip(txt)
    return txt

  @property
  def sentence_start_id(self):
    return self._pieces.index(SENTENCE_START_STRING)

  @property
  def sentence_start_string(self):
    return SENTENCE_START_STRING

  @property
  def sentence_end_id(self):
    return self._pieces.index(SENTENCE_END_STRING)

  @property
  def sentence_end_string(self):
    return SENTENCE_END_STRING

  @property
  def unk_id(self):
    return self._pieces.index(NO_TOKEN_STRING)
