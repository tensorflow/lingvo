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
"""The MLPerf reference implementation of BLEU."""

import collections
import math
import re
import sys
import unicodedata

import lingvo.compat as tf
from lingvo.core import metrics
import numpy as np
import six


def is_unicode(s):
  if isinstance(s, str):
    return True
  return False


def to_unicode(s, ignore_errors=False):
  if is_unicode(s):
    return s
  error_mode = "ignore" if ignore_errors else "strict"
  return s.decode("utf-8", errors=error_mode)


def native_to_unicode(s):
  if is_unicode(s):
    return s
  try:
    return to_unicode(s)
  except UnicodeDecodeError:
    res = to_unicode(s, ignore_errors=True)
    return res


def _get_ngrams(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this methods.

  Returns:
    The Counter containing all n-grams up to max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each reference
      should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation should
      be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)

    overlap = dict((ngram, min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) -
                                1] += translation_ngram_counts[ngram]
  precisions = [0] * max_order
  smooth = 1.0
  for i in range(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    if not reference_length:
      bp = 1.0
    else:
      ratio = translation_length / reference_length
      if ratio <= 0.0:
        bp = 0.0
      elif ratio >= 1.0:
        bp = 1.0
      else:
        bp = math.exp(1 - 1. / ratio)
  bleu = geo_mean * bp
  return np.float32(bleu)


def bleu_score(predictions, labels, **unused_kwargs):
  """BLEU score computation between labels and predictions.

  An approximate BLEU scoring method since we do not glue word pieces or
  decode the ids and tokenize the output. By default, we use ngram order of 4
  and use brevity penalty. Also, this does not have beam search.

  Args:
    predictions: tensor, model predictions
    labels: tensor, gold output.

  Returns:
    bleu: int, approx bleu score
  """
  outputs = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int32)
  # Convert the outputs and labels to a [batch_size, input_length] tensor.
  outputs = tf.squeeze(outputs, axis=[-1, -2])
  labels = tf.squeeze(labels, axis=[-1, -2])

  bleu = tf.py_func(compute_bleu, (labels, outputs), tf.float32)
  return bleu, tf.constant(1.0)


class UnicodeRegex:
  """Ad-hoc hack to recognize all punctuation and symbols."""

  def __init__(self):
    punctuation = self.property_chars("P")
    self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
    self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
    self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

  def property_chars(self, prefix):
    return "".join(
        six.unichr(x)
        for x in range(sys.maxunicode)
        if unicodedata.category(six.unichr(x)).startswith(prefix))


uregex = UnicodeRegex()


def bleu_tokenize(string):
  """Tokenize a string following the official BLEU implementation.

  Args:
    string: the input string

  Returns:
    a list of tokens
  """
  string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
  string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
  string = uregex.symbol_re.sub(r" \1 ", string)
  return string.split()


def bleu_wrapper(ref_lines, hyp_lines, case_sensitive=False):
  """Compute BLEU for two files (reference and hypothesis translation)."""
  assert len(ref_lines) == len(hyp_lines), ("{} != {}".format(
      len(ref_lines), len(hyp_lines)))
  ref_lines = [native_to_unicode(x) for x in ref_lines]
  hyp_lines = [native_to_unicode(x) for x in hyp_lines]
  if not case_sensitive:
    ref_lines = [x.lower() for x in ref_lines]
    hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
  return compute_bleu(ref_tokens, hyp_tokens)


class MlPerfBleuMetric(metrics.BaseMetric):
  """Use the MLPerf reference impelmentation."""

  def __init__(self, **kwargs):
    self._ref_lines = []
    self._hyp_lines = []

  def Update(self, ref_str, hyp_str):
    self._ref_lines.append(ref_str)
    self._hyp_lines.append(hyp_str)

  @property
  def value(self):
    return bleu_wrapper(self._ref_lines, self._hyp_lines)
