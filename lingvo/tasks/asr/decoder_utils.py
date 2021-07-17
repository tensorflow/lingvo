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
"""Common utilities for ASR decoders."""

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import symbolic
import six
from lingvo.tasks.asr import levenshtein_distance


def _IsSymbolOrPositive(dim):
  return symbolic.IsSymbol(dim) or dim > 0


def SetRnnCellNodes(decoder_params, rnn_cell_params):
  if _IsSymbolOrPositive(decoder_params.rnn_cell_dim):
    rnn_cell_params.num_output_nodes = decoder_params.rnn_cell_dim
  if _IsSymbolOrPositive(decoder_params.rnn_cell_hidden_dim):
    if not hasattr(rnn_cell_params, 'num_hidden_nodes'):
      raise ValueError(
          'num_hidden_nodes not supported by the RNNCell: %s' % rnn_cell_params)
    rnn_cell_params.num_hidden_nodes = decoder_params.rnn_cell_hidden_dim


def Tokenize(string):
  """Returns a list containing non-empty tokens from the given string."""
  if not isinstance(string, str):
    string = six.ensure_text(string, 'utf-8')
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
    return tf.strings.regex_replace(tf.strings.strip(s), r'\s+', ' ')

  hyps = _NormalizeWhitespace(hyps)
  refs = _NormalizeWhitespace(refs)

  hyps = py_utils.HasRank(hyps, 1)
  refs = py_utils.HasRank(refs, 1)
  hyps = py_utils.HasShape(hyps, tf.shape(refs))

  word_errors = tf.cast(
      tf.edit_distance(
          tf.string_split(hyps), tf.string_split(refs), normalize=False),
      tf.int64)

  # Count number of spaces in reference, and increment by 1 to get total number
  # of words.
  ref_words = tf.cast(
      tf.strings.length(tf.strings.regex_replace(refs, '[^ ]', '')) + 1,
      tf.int64)
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
    (ins, subs, del, total):

    - ins:         number of insertions.
    - subs:        number of substitutions.
    - del:         number of deletions.
    - total:       total difference length.
  """
  results = levenshtein_distance.LevenshteinDistance(
      Tokenize(ref_str), Tokenize(hyp_str))
  return results.insertions, results.subs, results.deletions, results.total


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
