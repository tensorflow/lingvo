# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for calibration_processing."""

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import calibration_processing
from lingvo.tasks.car import kitti_metadata

import numpy as np


class CalibrationProcessingTest(test_utils.TestCase):

  def testExpectedCalibrationError(self):
    confidence = np.array([0.05, 0.15, 0.2, 0.3])
    accuracies = np.array([0.25, 0.15, 0.2, 0.3])
    num_examples = np.array([1000, 10, 10, 10])
    ece = calibration_processing.ExpectedCalibrationError(
        confidence, accuracies, num_examples, min_confidence=0.1)
    self.assertNear(ece, 0.0, 1e-4)
    ece = calibration_processing.ExpectedCalibrationError(
        confidence, accuracies, num_examples)
    self.assertNear(ece, 0.1942, 1e-4)

  def testCalibrationCurveEqualScoresAndHits(self):
    # Test calibration curve data when scores are equal to hits.
    # This is an edge case where scores are either 0 or 1.
    num_bins = 2
    scores = np.array([0, 1, 0])
    hits = scores
    mean_predicted_accuracies, mean_empirical_accuracies, num_examples = \
        calibration_processing.CalibrationCurve(scores, hits, num_bins)
    expected_mean_predicted_accuracies = np.array([0.0, 1.0])
    self.assertAllEqual(expected_mean_predicted_accuracies,
                        mean_predicted_accuracies)
    expected_mean_empirical_accuracies = np.array([0, 1])
    self.assertAllEqual(expected_mean_empirical_accuracies,
                        mean_empirical_accuracies)
    expected_num_examples = np.zeros(shape=num_bins)
    expected_num_examples[0] = 2
    expected_num_examples[num_bins - 1] = 1
    self.assertAllEqual(expected_num_examples, num_examples)
    ece = calibration_processing.ExpectedCalibrationError(
        mean_predicted_accuracies, mean_empirical_accuracies, num_examples)
    self.assertEqual(ece, 0.0)

  def testCalibrationCurvePerfectCalibration(self):
    # Test calibration curve data when empirical accuracy corresponds to mean
    # predicted accuracy.
    num_bins = 2
    scores = np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75])
    hits = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    mean_predicted_accuracies, mean_empirical_accuracies, num_examples = \
        calibration_processing.CalibrationCurve(scores, hits, num_bins)
    expected_mean_predicted_accuracies = np.array([0.25, 0.75])
    self.assertAllEqual(expected_mean_predicted_accuracies,
                        mean_predicted_accuracies)
    expected_mean_empirical_accuracies = np.array([0.25, 0.75])
    self.assertAllEqual(expected_mean_empirical_accuracies,
                        mean_empirical_accuracies)
    expected_num_examples = np.zeros(shape=num_bins)
    expected_num_examples[0] = 4
    expected_num_examples[num_bins - 1] = 4
    self.assertAllEqual(expected_num_examples, num_examples)
    ece = calibration_processing.ExpectedCalibrationError(
        mean_predicted_accuracies, mean_empirical_accuracies, num_examples)
    self.assertEqual(ece, 0.0)

  def testAllDataInOneBin(self):
    # Test calibration curve data when all data is in one bin.
    num_bins = 2
    scores = np.array([1, 1, 1])
    hits = np.array([0, 0, 0])
    mean_predicted_accuracies, mean_empirical_accuracies, num_examples = \
        calibration_processing.CalibrationCurve(scores, hits, num_bins)
    expected_mean_predicted_accuracies = np.array([0.25, 1.0])
    self.assertAllEqual(expected_mean_predicted_accuracies,
                        mean_predicted_accuracies)
    expected_mean_empirical_accuracies = np.array([0, 0])
    self.assertAllEqual(expected_mean_empirical_accuracies,
                        mean_empirical_accuracies)
    expected_num_examples = np.zeros(shape=num_bins)
    expected_num_examples[0] = 0
    expected_num_examples[num_bins - 1] = 3
    self.assertAllEqual(expected_num_examples, num_examples)
    ece = calibration_processing.ExpectedCalibrationError(
        mean_predicted_accuracies, mean_empirical_accuracies, num_examples)
    self.assertEqual(ece, 1.0)

  def testEmptyBins(self):
    # Test calibration curve data when there are no examples.
    num_bins = 2
    scores = np.array([])
    hits = np.array([])
    mean_predicted_accuracies, mean_empirical_accuracies, num_examples = \
        calibration_processing.CalibrationCurve(scores, hits, num_bins)
    expected_mean_predicted_accuracies = np.array([0.25, 0.75])
    self.assertAllEqual(expected_mean_predicted_accuracies,
                        mean_predicted_accuracies)
    expected_mean_empirical_accuracies = np.array([0, 0])
    self.assertAllEqual(expected_mean_empirical_accuracies,
                        mean_empirical_accuracies)
    expected_num_examples = np.zeros(shape=num_bins)
    self.assertAllEqual(expected_num_examples, num_examples)
    ece = calibration_processing.ExpectedCalibrationError(
        mean_predicted_accuracies, mean_empirical_accuracies, num_examples)
    self.assertEqual(ece, 0.0)

  def testCalibrationCalculator(self):
    # End to end test for the calibration calculator.
    metadata = kitti_metadata.KITTIMetadata()
    calculator = calibration_processing.CalibrationCalculator(metadata)
    scores_and_hits = np.array([[0.3, 1], [0.5, 1], [0.7, 1]])
    metrics = {}
    metrics['calibrations'] = [{'calibrations': scores_and_hits}]
    calculator.Calculate(metrics)
    summaries = calculator.Summary('Test')
    self.assertEqual(len(summaries), 2)
    ece_summary = summaries[1]
    self.assertEqual(0.5, ece_summary.value[0].simple_value)


if __name__ == '__main__':
  tf.test.main()
