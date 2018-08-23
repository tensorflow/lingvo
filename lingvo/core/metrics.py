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
"""Helper classes for computing performance metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf

from lingvo.core import scorers


def CreateScalarSummary(name, simple_value):
  return tf.Summary(
      value=[tf.Summary.Value(tag=name, simple_value=simple_value)])


class BaseMetric(object):
  """Base class for aggregating statistics to compute a performance metric."""

  def Update(self, *args, **kwargs):
    """Updates this metric (e.g. accumulates statistics) from the arguments."""
    pass

  @property
  def value(self):
    """Current value of this metric."""
    return None

  def Summary(self, name):
    """Converts the current state of this metric to a tf.Summary.

    Args:
      name: A string to use as the summary value tag.

    Returns:
      A tf.Summary proto.
    """
    return CreateScalarSummary(name, self.value)


class AverageMetric(BaseMetric):
  """Class to compute a weighted (arithmetic) average value metric."""

  def __init__(self):
    self._total_value = 0.0
    self._total_weight = 0.0

  def Update(self, value, weight=1.0):
    if weight < 0.0:
      raise ValueError('weight must be non-negative.  Got: %f' % weight)
    self._total_value += value * weight
    self._total_weight += weight

  # We may want both a getter and a setter method for total_value and
  # total_weight, respectively.
  def GetTotalValue(self):
    return self._total_value

  def SetTotalValue(self, val):
    self._total_value = val

  total_value = property(GetTotalValue, SetTotalValue)

  def GetTotalWeight(self):
    return self._total_weight

  def SetTotalWeight(self, val):
    self._total_weight = val

  total_weight = property(GetTotalWeight, SetTotalWeight)

  @property
  def value(self):
    return self._total_value/self._total_weight if self._total_weight > 0 else 0


class F1Metric(BaseMetric):
  """Class to compute F1 metrics."""

  def __init__(self):
    self._true_pos = 0.0
    self._false_pos = 0.0
    self._false_neg = 0.0

  def UpdateTruePositive(self, count=1.0):
    self._true_pos += count

  def UpdateFalsePositive(self, count=1.0):
    self._false_pos += count

  def UpdateFalseNegative(self, count=1.0):
    self._false_neg += count

  @property
  def value(self):
    if (self._true_pos + self._false_pos) > 0:
      precision = self._true_pos / (self._true_pos + self._false_pos)
    else:
      precision = 0.0
    if (self._true_pos + self._false_neg) > 0:
      recall = self._true_pos / (self._true_pos + self._false_neg)
    else:
      recall = 0.0
    if (precision + recall) > 0:
      return 2.0 * precision * recall / (precision + recall)
    else:
      return 0.0


class CorpusBleuMetric(BaseMetric):
  """Metric class to compute the corpus-level BLEU score."""

  def __init__(self, **kwargs):
    self._scorer = scorers.BleuScorer(**kwargs)

  def Update(self, ref_str, hyp_str):
    self._scorer.AddSentence(ref_str, hyp_str)

  @property
  def unsegmenter(self):
    return self._scorer.unsegmenter

  @property
  def value(self):
    return self._scorer.ComputeOverallScore()
