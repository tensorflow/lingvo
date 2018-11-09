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
"""Common utilities for ASR decoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import six

import tensorflow as tf

from lingvo.core import py_utils


def SetRnnCellNodes(decoder_params, rnn_cell_params):
  rnn_cell_params.num_output_nodes = decoder_params.rnn_cell_dim
  if decoder_params.rnn_cell_hidden_dim > 0:
    if not hasattr(rnn_cell_params, 'num_hidden_nodes'):
      raise ValueError(
          'num_hidden_nodes not supported by the RNNCell: %s' % rnn_cell_params)
    rnn_cell_params.num_hidden_nodes = decoder_params.rnn_cell_hidden_dim


def Tokenize(string):
  """Returns a list containing non-empty tokens from the given string."""
  if not isinstance(string, six.text_type):
    string = string.decode('utf-8')
  return string.split()


def ComputeWer(hyps, refs):
  """Computes word errors in hypotheses relative to reference transcripts.

  Args:
    hyps: Hypotheses, represented as string tensors of shape [N].
    refs: References, represented as string tensors of shape [N].

  Returns:
    An int64 tensor, word_errs, of size [N, 2] where word_errs[i, 0] corresponds
    to the number of word errors in hyps[i] relative to refs[i]; word_errs[i, 1]
    corresponds to the number of words in refs[i].
  """

  def _NormalizeWhitespace(s):
    return tf.regex_replace(tf.strings.strip(s), r'\s+', ' ')

  hyps = _NormalizeWhitespace(hyps)
  refs = _NormalizeWhitespace(refs)

  hyps = py_utils.HasRank(hyps, 1)
  refs = py_utils.HasRank(refs, 1)
  hyps = py_utils.HasShape(hyps, tf.shape(refs))

  word_errors = tf.to_int64(
      tf.edit_distance(
          tf.string_split(hyps), tf.string_split(refs), normalize=False))

  # Count number of spaces in reference, and increment by 1 to get total number
  # of words.
  ref_words = tf.to_int64(
      tf.strings.length(tf.regex_replace(refs, '[^ ]', '')) + 1)
  # Set number of words to 0 if the reference was empty.
  ref_words = tf.where(
      tf.equal(refs, ''), tf.zeros_like(ref_words, tf.int64), ref_words)

  return tf.concat(
      [tf.expand_dims(word_errors, -1),
       tf.expand_dims(ref_words, -1)], axis=1)


def EditDistance(ref_str, hyp_str):
  """Computes Levenshtein edit distance between reference and hypotheses.

  Args:
    ref_str:   A string of the ref sentence.
    hyp_str:   A string of one actual hyp.

  Returns:
    ins:         number of insertions.
    subs:        number of substitutions.
    del:         number of deletions.
    total:       total difference length.
  """

  class ErrorStats(object):
    """Class to keep track of error counts.

    """

    def __init__(self, ins, dels, subs, tot):
      self.ins, self.dels, self.subs, self.total_cost = ins, dels, subs, tot

    def __repr__(self):
      return 'ErrorStats(ins=%d, dels=%d, subs=%d, tot=%d)' % (
          self.ins, self.dels, self.subs, self.total_cost)

  # temp sequence to remember error type and stats.
  e, cur_e = [], []
  lst_ref = Tokenize(ref_str)
  for i in range(len(lst_ref) + 1):
    e.append(ErrorStats(0, i, 0, i))
    cur_e.append(ErrorStats(0, 0, 0, 0))

  lst_hyp = Tokenize(hyp_str)
  for hyp_index in range(1, len(lst_hyp) + 1):
    cur_e[0] = copy.copy(e[0])
    cur_e[0].ins += 1
    cur_e[0].total_cost += 1

    for ref_index in range(1, len(lst_ref) + 1):
      ins_err = e[ref_index].total_cost + 1
      del_err = cur_e[ref_index - 1].total_cost + 1
      sub_err = e[ref_index - 1].total_cost
      if lst_hyp[hyp_index - 1] != lst_ref[ref_index - 1]:
        sub_err += 1

      if sub_err < ins_err and sub_err < del_err:
        cur_e[ref_index] = copy.copy(e[ref_index - 1])
        if lst_hyp[hyp_index - 1] != lst_ref[ref_index - 1]:
          cur_e[ref_index].subs += 1
        cur_e[ref_index].total_cost = sub_err
      elif del_err < ins_err:
        cur_e[ref_index] = copy.copy(cur_e[ref_index - 1])
        cur_e[ref_index].total_cost = del_err
        cur_e[ref_index].dels += 1
      else:
        cur_e[ref_index] = copy.copy(e[ref_index])
        cur_e[ref_index].total_cost = ins_err
        cur_e[ref_index].ins += 1

    for i in range(len(e)):
      e[i] = copy.copy(cur_e[i])

  return e[-1].ins, e[-1].subs, e[-1].dels, e[-1].total_cost


def EditDistanceInIds(ref_ids, hyp_ids):
  ref_ids = ['%d' % x for x in ref_ids]
  hyp_ids = ['%d' % x for x in hyp_ids]
  ref_str = ' '.join(ref_ids)
  hyp_str = ' '.join(hyp_ids)
  return EditDistance(ref_str, hyp_str)


def FilterEpsilon(string):
  """Filters out <epsilon> tokens from the given string."""
  return ' '.join(Tokenize(string.replace('<epsilon>', ' ')))


def FilterNoise(string):
  """Filters out <noise> tokens from the given string."""
  return ' '.join(t for t in Tokenize(string) if t != '<noise>')
