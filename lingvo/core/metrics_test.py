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

import lingvo.compat as tf
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np


class MetricsTest(test_utils.TestCase):

  def testAverageMetric(self):
    m = metrics.AverageMetric()
    m.Update(1.0)
    m.Update(2.0, 10.0)

    self.assertEqual(1.0 + 2.0 * 10.0, m.total_value)
    expected_average = (1.0 + 2.0 * 10.0) / (1.0 + 10.0)
    self.assertEqual(expected_average, m.value)

    name = 'metric_name'
    self.assertEqual(
        tf.Summary(
            value=[tf.Summary.Value(tag=name, simple_value=expected_average)]),
        m.Summary(name))

    # Calling m.Summary() does not reset statistics.
    m.Update(1.0)
    self.assertEqual(1.0 + 2.0 * 10.0 + 1.0, m.total_value)

  def testUniqueAverageMetric(self):
    m = metrics.UniqueAverageMetric()
    m.Update('a', 1.0)
    m.Update('b', 2.0, 10.0)
    # Different value for 'a' than the previous one.
    m.Update('a', 2.0)

    with self.assertRaises(ValueError):
      # Error raised during value, so that we can collect all
      # of the keys in the error reporting.
      _ = m.value

    m = metrics.UniqueAverageMetric()
    m.Update('a', 1.0)
    m.Update('b', 2.0, 10.0)
    # Duplicate update is ignored.
    m.Update('a', 1.0)

    self.assertEqual(1.0 + 2.0 * 10.0, m.total_value)

    expected_average = (1.0 + 2.0 * 10.0) / (1.0 + 10.0)
    self.assertEqual(expected_average, m.value)

    name = 'metric_name'
    self.assertEqual(
        tf.Summary(value=[
            tf.Summary.Value(tag=name, simple_value=expected_average),
            tf.Summary.Value(tag=name + '/total_count', simple_value=2),
            tf.Summary.Value(tag=name + '/total_value', simple_value=21.),
            tf.Summary.Value(tag=name + '/total_weight', simple_value=11.),
        ]), m.Summary(name))

    m = metrics.UniqueAverageMetric(mismatch_is_error=False)
    m.Update('a', 1.0)
    m.Update('a', 2.0)
    self.assertEqual(1.0, m.value)

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
        tf.Summary(
            value=[tf.Summary.Value(tag=name, simple_value=expected_f1)]),
        m.Summary(name))

  def testMCCMetric(self):
    m = metrics.MCCMetric()
    m.UpdateTruePositive(count=2.0)
    m.UpdateTrueNegative(count=2.0)
    m.UpdateFalsePositive()
    m.UpdateFalseNegative()

    expected_mcc = 1 / 3
    self.assertAlmostEqual(expected_mcc, m.value)

    name = 'my_mcc_metric'
    self.assertEqual(
        tf.Summary(
            value=[tf.Summary.Value(tag=name, simple_value=expected_mcc)]),
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

  def testCorrelationMetric(self):
    m = metrics.CorrelationMetric()
    m.Update([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    m.Update([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    m.Update([1.0, 2.0, 3.0], [0.3, 0.2, 0.1])
    # (1 + 1 + -1) / 3
    self.assertAlmostEqual(1.0 / 3, m.value)

  def testAverageKeyedCorrelationMetric(self):
    m = metrics.AverageKeyedCorrelationMetric()
    m.Update('k1', [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    m.Update('k2', [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    m.Update('k2', [1.0, 2.0, 3.0], [0.3, 0.2, 0.1])
    # (1 / 1 + (1 + -1) / 2) / 2
    self.assertAlmostEqual(0.5, m.value)
    m.Update('k3', [1.0], [0.3])
    # k3 does not affect result
    self.assertAlmostEqual(0.5, m.value)

  def testAverageKeyedCorrelationMetricEmptyInput(self):
    m = metrics.AverageKeyedCorrelationMetric()
    m.Update('k1', [], [])
    m.Update('k2', [], [])
    self.assertEqual(0.0, m.value)

  def testSamplingMetric(self):

    class TestSamplingMetric(metrics.SamplingMetric):

      def _CreateSummary(self, name):
        ret = tf.Summary()
        for sample in self.samples:
          value = sample.value
          ret.value.add(tag=name, simple_value=value)
        return ret

    np.random.seed(1337)
    p = TestSamplingMetric.Params()
    p.num_samples = 2
    m = p.Instantiate()
    m.Update(py_utils.NestedMap(value=1))
    summary = m.Summary('test')
    self.assertEqual(1, summary.value[0].simple_value)

    # Calling Summary() clears out history.
    #
    # Add five total updates.
    m.Update(py_utils.NestedMap(value=1))
    m.Update(py_utils.NestedMap(value=2))
    m.Update(py_utils.NestedMap(value=3))
    m.Update(py_utils.NestedMap(value=4))
    m.Update(py_utils.NestedMap(value=5))
    summary = m.Summary('test')
    # Reservoir sampling will sample values 5 and 3 to remain with the current
    # seed.
    self.assertEqual(2, len(summary.value))
    self.assertEqual(5, summary.value[0].simple_value)
    self.assertEqual(3, summary.value[1].simple_value)

  def testAUCMetric(self):
    if not metrics.HAS_SKLEARN:
      self.skipTest('sklearn is not installed.')
    m = metrics.AUCMetric()
    m.Update(label=[1, 1], prob=[0.1, 0.2], weight=[1.0, 1.0])
    # No meaningful AUC yet, since all(labels==1) and function needs 2 types of
    # labels.
    self.assertEqual(0.0, m.value)

    m.Update(label=[0, 0], prob=[0.1, 0.2], weight=[1.0, 1.0])
    self.assertEqual(0.5, m.value)

  def testPrecisionAtRecall(self):
    if not metrics.HAS_SKLEARN:
      self.skipTest('sklearn is not installed.')
    m = metrics.PrecisionAtRecall(recall_threshold=0.5)
    m.Update(label=[1, 1], prob=[0.15, 0.25], weight=[1.0, 1.0])
    m.Update(label=[0, 0], prob=[0.1, 0.2], weight=[1.0, 1.0])
    self.assertEqual(1.0, m.value)

    m = metrics.PrecisionAtRecall(recall_threshold=1.0)
    m.Update(label=[1, 1], prob=[0.15, 0.25], weight=[1.0, 1.0])
    m.Update(label=[0, 0], prob=[0.1, 0.2], weight=[1.0, 1.0])
    self.assertEqual(2.0 / 3, m.value)

  def testRecallAtPrecision(self):
    if not metrics.HAS_SKLEARN:
      self.skipTest('sklearn is not installed.')
    m = metrics.RecallAtPrecision(precision_threshold=0.5)
    m.Update(label=[1, 1], prob=[0.15, 0.25], weight=[1.0, 1.0])
    m.Update(label=[0, 0], prob=[0.1, 0.2], weight=[1.0, 1.0])
    self.assertEqual(1.0, m.value)

    m = metrics.RecallAtPrecision(precision_threshold=1.0)
    m.Update(label=[1, 1], prob=[0.15, 0.25], weight=[1.0, 1.0])
    m.Update(label=[0, 0], prob=[0.1, 0.2], weight=[1.0, 1.0])
    self.assertAlmostEqual(0.5, m.value)

  def testMultiClassAUCMetric(self):
    if not metrics.HAS_SKLEARN:
      self.skipTest('sklearn is not installed.')
    m = metrics.MultiClassAUCMetric(num_classes=3)
    class_labels = [[1, 1], [1, 1], [1, 1]]
    class_probs = [[0.1, 0.2], [0.7, 0.9], [0.4, 0.6]]
    m.Update(class_labels, class_probs)
    # No meaningful AUC yet, since all classes have the same label and AUCMetric
    # requires 2 types of labels to properly compute AUC.
    for auc_metric in m._class_auc_metrics:
      self.assertEqual(0.0, auc_metric.value)
    self.assertEqual(0.0, m.value)

    class_labels = [[0, 0], [0, 0], [0, 1]]
    class_probs = [[0.1, 0.2], [0.1, 0.15], [0.5, 0.9]]
    m.Update(class_labels, class_probs)

    # Verify each class's AUC value.
    self.assertEqual(0.5, m._class_auc_metrics[0].value)
    self.assertEqual(1.0, m._class_auc_metrics[1].value)
    self.assertAllClose(2 / 3, m._class_auc_metrics[2].value)

    # Verify average AUC value.
    self.assertAllClose(0.722222222, m.value)

  def testGroupPairAUCMetric(self):
    if not metrics.HAS_SKLEARN:
      self.skipTest('sklearn is not installed.')
    pair_m = metrics.AUCMetric()
    group_m = metrics.GroupPairAUCMetric()
    group_ids = [0, 0, 0, 1, 1, 1, 2, 2]
    target = np.random.rand(8).tolist()
    logits = np.random.rand(8).tolist()
    weight = [1.0] * 8
    group_m.UpdateRaw(
        group_ids=group_ids, target=target, logits=logits, weight=weight)

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    left, right = 0, 0
    while right < len(group_ids):
      while right < len(group_ids) and group_ids[right] == group_ids[left]:
        right += 1
      for i in range(left, right):
        for j in range(i + 1, right):
          if group_ids[i] == group_ids[j] and target[i] != target[j]:
            pair_m.Update(
                label=[1 if target[i] > target[j] else 0],
                prob=[sigmoid(logits[i] - logits[j])],
                weight=[min(1.0, weight[i] + weight[j])])
      left = right
    self.assertEqual(pair_m.value, group_m.value)

if __name__ == '__main__':
  test_utils.main()
