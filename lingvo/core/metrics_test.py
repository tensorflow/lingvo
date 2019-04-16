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
"""Tests for metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf
from lingvo.core import metrics
from lingvo.core import test_utils


class MetricsTest(test_utils.TestCase):

  def testAverageMetric(self):
    m = metrics.AverageMetric()
    m.Update(1.0)
    m.Update(2.0, 10.0)

    self.assertEqual(1.0 + 2.0*10.0, m.total_value)
    expected_average = (1.0 + 2.0*10.0) / (1.0 + 10.0)
    self.assertEqual(expected_average, m.value)

    name = 'metric_name'
    self.assertEqual(
        tf.Summary(value=[tf.Summary.Value(tag=name,
                                           simple_value=expected_average)]),
        m.Summary(name))

    # Calling m.Summary() does not reset statistics.
    m.Update(1.0)
    self.assertEqual(1.0 + 2.0*10.0 + 1.0, m.total_value)

  def testF1Metric(self):
    m = metrics.F1Metric()
    m.UpdateTruePositive(count=2.0)
    m.UpdateFalsePositive()
    m.UpdateFalseNegative()

    precision = 2.0 / 3.0
    recall = 2.0 / 3.0
    expected_f1 = 2 * precision * recall / (precision + recall)
    self.assertAlmostEqual(expected_f1, m.value)

    name = 'my_f1_metric'
    self.assertEqual(
        tf.Summary(value=[tf.Summary.Value(tag=name,
                                           simple_value=expected_f1)]),
        m.Summary(name))

  def testCorpusBleuMetric(self):
    m = metrics.CorpusBleuMetric()
    m.Update('a b c d', 'a b c d')
    m.Update('a b c', 'a b c')

    self.assertEqual(1.0, m.value)

    name = 'corpus_bleu'
    self.assertEqual(
        tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=1.0)]),
        m.Summary(name))


if __name__ == '__main__':
  tf.test.main()
