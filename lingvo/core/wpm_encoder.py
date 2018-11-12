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

import numpy as np
import sys

import tensorflow as tf

# Must be a large ID.
NO_TOKEN = sys.maxsize
NO_TOKEN_STRING = '<unk>'

SENTENCE_START_STRING = '<s>'
SENTENCE_END_STRING = '</s>'


class WpmEncoder(object):

  def __init__(self, wpm_filepath, merge_prob=1.):
    """Create a WPM encoder.

    Args:
      wpm_filepath: a path to the file containing the vocabulary.
      merge_prob: the probability of merging tokens while encoding.
    """
    # Load vocabulary file.
    self._piece2id = {}
    self._pieces = []
    with tf.gfile.Open(wpm_filepath, 'r') as f:
      pid = 0
      for line in f.readlines():
        line = line.decode('utf-8')
        piece = line.strip().split('\t')[0]
        tf.logging.vlog(6, 'voc: %s -> %d', piece, pid)
        self._piece2id[piece] = pid
        pid += 1
        self._pieces += [piece]
    assert self._StringToToken(NO_TOKEN_STRING) != NO_TOKEN
    self._merge_prob = merge_prob

  def _TokenToString(self, token, safe=False):
    if token == NO_TOKEN:
      if safe:
        return NO_TOKEN_STRING
      else:
        assert token != NO_TOKEN
    return self._pieces[token]

  def _TokensToString(self, tokens, sep=' ', safe=False):
    return sep.join([self._TokenToString(t, safe) for t in tokens])

  def _StringToToken(self, tokstr):
    if tokstr in self._piece2id:
      return self._piece2id[tokstr]
    return NO_TOKEN

  def _MergeTokens(self, token1, token2):
    return self._StringToToken(
        self._TokenToString(token1) + self._TokenToString(token2))

  def _IsAllNull(self, candidates):
    return len(candidates) == candidates.count(NO_TOKEN)

  def _ShouldMerge(self):
    return np.random.uniform() < self._merge_prob

  def _EncodeToIds(self, word):
    if not isinstance(word, unicode):
      word = word.decode('utf-8')
    # First, match entire words. That will work to shortcut exception words
    # such as <S>, even if there is no path to merge hierarchically. To wit,
    # in this case, either '<S' or 'S>' must be present.
    if word in self._piece2id:
      return [self._piece2id[word]]
    # Henceforth,
    #   * a token is a wordpiece ID.
    #   * the tokens array will be merged in-place.
    #   * the candidates array is an array of size len(tokens) - 1.
    #     It contains the token for the merged wordpiece, if it exists,
    #     -1 otherwise. For instance, candidate[3] = id(token[3] + token[4]).
    # First, split into basic UTF-8 characters (letters).
    tokens = [self._StringToToken(letter) for letter in word]
    for t in range(len(tokens)):
      if tokens[t] == NO_TOKEN:
        # Unseen character.
        tokens[t] = self._StringToToken(NO_TOKEN_STRING)
    # Create initial candidate list.
    candidates = []
    for i in range(len(tokens) - 1):
      merged = self._MergeTokens(tokens[i], tokens[i + 1])
      candidates += [merged]
    # Merge in the reverse binary tree until no more merges are possible, or
    # we decide to abort the process early, according to merge_prob.
    while not self._IsAllNull(candidates) and self._ShouldMerge():
      best_id = np.argmin(candidates)
      # Perform the merge at position best_id.
      tokens = tokens[:best_id] + [candidates[best_id]] + tokens[best_id + 2:]
      # Recompute the merge candidates to the right:
      if best_id < len(tokens) - 1:
        candidates[best_id] = self._MergeTokens(tokens[best_id],
                                                tokens[best_id + 1])
      else:
        candidates.pop()
      # Recompute the merge candidates to the left:
      if best_id > 0:
        candidates[best_id - 1] = self._MergeTokens(tokens[best_id - 1],
                                                    tokens[best_id])
      candidates = candidates[:best_id + 1] + candidates[best_id + 2:]
      assert len(tokens) == len(candidates) + 1
    return tokens

  def EncodeWord(self, word):
    return [self._TokenToString(t) for t in self._EncodeToIds(word)]

  def EncodeToStringAndIds(self, word):
    return [(self._TokenToString(t), t) for t in self._EncodeToIds(word)]

  def Encode(self, text):
    """Assumes that the text is utf-8 decoded."""
    words = text.split(' ')
    encoded_words = []
    for w in words:
      encoded_words += self.EncodeWord(w)
    return ' '.join(encoded_words)

  def Decode(self, ids):
    return [self._TokenToString(i) for i in ids]

  @property
  def sentence_start_id(self):
    return self._piece2id[SENTENCE_START_STRING]

  @property
  def sentence_start_string(self):
    return SENTENCE_START_STRING

  @property
  def sentence_end_id(self):
    return self._piece2id[SENTENCE_END_STRING]

  @property
  def sentence_end_string(self):
    return SENTENCE_END_STRING

  @property
  def unk_id(self):
    return self._piece2id[NO_TOKEN_STRING]
