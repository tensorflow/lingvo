# Lint as: python2, python3
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
"""Helper classes for computing scores."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import six
from six.moves import range


def _ToUnicode(line):
  return six.ensure_text(line, 'utf-8')


def _Tokenize(string):
  return _ToUnicode(string).split()


def NGrams(lst, order):
  """Generator that yields all n-grams of the given order present in lst."""
  return (lst[i:i + order] for i in range(len(lst) - order + 1))


class Unsegmenter(object):
  """Un-segments (merges) segmented strings.

  Used to retain back the original surface form of strings that are encoded
  using byte-pair-encoding (BPE), word-piece-models (WPM) or
  sentence-piece-models (SPM).
  """

  _BPE_SEPARATOR = _ToUnicode('@@ ')
  _WPM_SEPARATOR = _ToUnicode('\xe2\x96\x81')  # Same for SPM.

  def __init__(self, separator_type=None):
    self._separator_type = separator_type

  def _UnsegmentWpm(self, line):
    return _ToUnicode(line).replace(' ', '').replace(self._WPM_SEPARATOR,
                                                     ' ').strip()

  def _UnsegmentBpe(self, line):
    return _ToUnicode(line).replace(self._BPE_SEPARATOR, '').strip()

  def __call__(self, line):
    if self._separator_type == 'bpe':
      return self._UnsegmentBpe(line)
    elif self._separator_type in ['wpm', 'spm']:
      return self._UnsegmentWpm(line)
    else:
      return line


class BleuScorer(object):
  """Scorer to compute BLEU scores to measure translation quality.

  The BLEU score is the geometric average precision of all token n-grams of
  order 1 to max_ngram across all sentences.

  Successive calls to AddSentence() accumulate statistics which are converted to
  an overall score on calls to ComputeOverallScore().

  Example usage:
  >>> scorer = BleuScorer(max_ngram=4)
  >>> scorer.AddSentence("hyp matches ref str", "hyp matches ref str")
  >>> scorer.AddSentence("almost right", "almost write")
  >>> print(scorer.ComputeOverallScore())
  0.6687...
  """

  def __init__(self, max_ngram=4, separator_type=None):
    self._max_ngram = max_ngram
    self._hyp_ngram_matches = [0 for _ in range(max_ngram)]
    self._hyp_ngram_counts = [0 for _ in range(max_ngram)]
    self._num_ref_tokens = 0
    self._num_hyp_tokens = 0
    self._unsegmenter = Unsegmenter(separator_type)

  @property
  def unsegmenter(self):
    return self._unsegmenter

  def AddSentence(self, ref_str, hyp_str):
    """Accumulates ngram statistics for the given ref and hyp string pair."""
    ref_tokens = tuple(_Tokenize(self._unsegmenter(ref_str)))
    self._num_ref_tokens += len(ref_tokens)
    hyp_tokens = tuple(_Tokenize(self._unsegmenter(hyp_str)))
    self._num_hyp_tokens += len(hyp_tokens)
    for order_idx in range(self._max_ngram):
      ref_counts = collections.Counter(NGrams(ref_tokens, order_idx + 1))
      hyp_matches = collections.Counter()
      hyp_count = 0
      for x in NGrams(hyp_tokens, order_idx + 1):
        hyp_count += 1
        count = ref_counts[x]
        if count:
          # Clip hyp_matches so ngrams that are repeated more frequently in hyp
          # than ref are not double counted.
          hyp_matches[x] = min(hyp_matches[x] + 1, count)
      self._hyp_ngram_matches[order_idx] += sum(six.itervalues(hyp_matches))
      self._hyp_ngram_counts[order_idx] += hyp_count

  def ComputeOverallScore(self):
    """Computes overall BLEU score from the statistics accumulated so far."""
    score = 0.0
    num_nonzero_orders = 0
    for order_idx in range(self._max_ngram):
      matches = self._hyp_ngram_matches[order_idx]
      total = self._hyp_ngram_counts[order_idx]
      if matches > 0.0 and total > 0.0:
        score += math.log(matches / total)
        num_nonzero_orders += 1
    if not num_nonzero_orders:
      return 0.0
    precision = math.exp(score / num_nonzero_orders)

    brevity_penalty = 1.0
    if self._num_hyp_tokens < self._num_ref_tokens:
      brevity_penalty = math.exp(1 - self._num_ref_tokens/self._num_hyp_tokens)
    return brevity_penalty * precision
