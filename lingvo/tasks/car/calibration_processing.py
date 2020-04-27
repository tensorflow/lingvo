# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Library for calculating calibration on a prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
from lingvo.core import plot
import numpy as np
from six.moves import range


def ExpectedCalibrationError(confidence,
                             empirical_accuracy,
                             num_examples,
                             min_confidence=None):
  """Calculate the expected calibration error.

  Args:
    confidence: 1-D np.array of float32 binned confidence scores with one number
      per bin
    empirical_accuracy: 1-D np.array of float32 binned empirical accuracies with
      one number per bin
    num_examples: 1-D np.array of int for the number of examples within a bin.
    min_confidence: float32 of minimum confidence score to use in the
      calculation. If None, no filtering is applied.

  Returns:
    float32 of expected calibration error
  """
  assert confidence.shape[0] == empirical_accuracy.shape[0]
  assert empirical_accuracy.shape[0] == num_examples.shape[0]

  ece = np.abs(empirical_accuracy - confidence) * num_examples

  if min_confidence:
    bin_indices = np.where(confidence > min_confidence)
    ece = ece[bin_indices]
    num_examples = num_examples[bin_indices]

  ece = np.sum(ece)
  total_num_examples = np.sum(num_examples)
  if total_num_examples != 0:
    ece /= total_num_examples
  else:
    ece = 0.0
  return ece


def CalibrationCurve(scores, hits, num_bins):
  """Compute data for calibration reliability diagrams.

  Args:
    scores: 1-D np.array of float32 confidence scores
    hits: 1-D np.array of int32 (either 0 or 1) indicating whether predicted
      label matches the ground truth label
    num_bins: int for the number of calibration bins

  Returns:
    A tuple containing:
      - mean_predicted_accuracies: np.array of mean predicted accuracy for each
          bin
      - mean_empirical_accuracies: np.array of mean empirical accuracy for each
          bin
      - num_examples: np.array of the number of examples in each bin
  """
  mean_predicted_accuracies = []
  mean_empirical_accuracies = []
  num_examples = []

  # Bin the hits and scores based on the scores.
  edges = np.linspace(0.0, 1.0, num_bins + 1)
  bin_indices = np.digitize(scores, edges, right=True)
  # Put examples with score equal to 0 in bin 1 because we will skip bin 0.
  bin_indices = np.where(scores == 0.0, 1, bin_indices)

  for j in range(num_bins + 1):
    if j == 0:
      continue
    indices = np.where(bin_indices == j)[0]
    # pylint: disable=g-explicit-length-test
    if len(indices) > 0:
      mean_predicted_accuracy = np.mean(scores[indices])
      mean_empirical_accuracy = np.mean(hits[indices])
      num_example = len(indices)
    else:
      mean_predicted_accuracy = (edges[j - 1] + edges[j]) / 2.0
      mean_empirical_accuracy = 0.0
      num_example = 0
    # pylint: enable=g-explicit-length-test

    mean_predicted_accuracies.append(mean_predicted_accuracy)
    mean_empirical_accuracies.append(mean_empirical_accuracy)
    num_examples.append(num_example)

  mean_predicted_accuracies = np.array(mean_predicted_accuracies)
  mean_empirical_accuracies = np.array(mean_empirical_accuracies)
  num_examples = np.array(num_examples)
  return mean_predicted_accuracies, mean_empirical_accuracies, num_examples


class CalibrationCalculator(object):
  """Base class for calculating calibration on a prediction."""

  def __init__(self, metadata):
    self._metadata = metadata
    self._num_calibration_bins = self._metadata.NumberOfCalibrationBins()
    self._calibration_by_class = None
    self._classnames = self._metadata.ClassNames()
    self._classids = self._metadata.EvalClassIndices()

  def Calculate(self, metrics):
    """Calculate metrics for calibration.

    Args:
      metrics: A dict. Each entry in the dict is a list of C (number of classes)
        dicts containing mapping from metric names to individual results.
      Individual entries may be the following items:
      - scalars: A list of C (number of classes) dicts mapping metric names to
        scalar values.
      - curves: A list of C dicts mapping metrics names to np.float32 arrays of
        shape [NumberOfPrecisionRecallPoints()+1, 2]. In the last dimension, 0
        indexes precision and 1 indexes recall.
      - calibrations: A list of C dicts mapping metrics names to np.float32
        arrays of shape [number of predictions, 2]. The first column is the
        predicted probabilty and the second column is 0 or 1 indicating that the
        prediction matched a ground truth item.

    Returns:
      nothing
    """
    if 'calibrations' not in metrics:
      tf.logging.info(
          'CalibrationProcessing invoked but no metrics available '
          'for calculating calibration.')
      return

    self._calibration_by_class = {}
    for i, c in enumerate(metrics['calibrations']):
      classid = self._classids[i]
      classname = self._classnames[classid]

      if np.all(np.isnan(c['calibrations'])) or c['calibrations'].size == 0:
        tf.logging.info(
            'Skipping %s for calibration calculation because no '
            'output provided.' % classname)
        continue
      tf.logging.info('Calculating calibration for %s: %d items.' %
                           (classname, len(c['calibrations'])))

      # Ensure that all counts are greater then zero and less then or equal
      # to 1.0 to guarantee that all scores are counted.
      scores_and_hits = np.clip(c['calibrations'], 1e-10, 1.0)
      scores = scores_and_hits[:, 0]
      hits = scores_and_hits[:, 1]
      curve_data = CalibrationCurve(scores, hits, self._num_calibration_bins)
      self._calibration_by_class[classname] = np.array(curve_data[0:3])
      tf.logging.info('Finished calculating calibration for %s.' %
                           classname)

  def Summary(self, name):
    """Generate tf summaries for calibration.

    Args:
      name: str, name of summary.

    Returns:
      list of tf.Summary
    """
    summaries = []
    for class_id in self._metadata.EvalClassIndices():
      classname = self._metadata.ClassNames()[class_id]
      tag_str = '{}/{}/calibration'.format(name, classname)

      if classname not in self._calibration_by_class:
        continue

      # Extract the data.
      mean_predicted_accuracy = self._calibration_by_class[classname][0, :]
      mean_empirical_accuracy = self._calibration_by_class[classname][1, :]
      num_examples_per_bin = self._calibration_by_class[classname][-1, :]
      total_examples = np.sum(num_examples_per_bin)
      legend = ['%s (%d)' % (classname, total_examples)]

      def _CalibrationSetter(fig, axes):
        """Configure the plot for calibration."""
        ticks = np.arange(0, 1.05, 0.1)
        axes.grid(b=False)
        axes.set_xlabel('Predicted accuracy')
        axes.set_xticks(ticks)
        axes.set_ylabel('Empirical accuracy')
        axes.set_yticks(ticks)
        axes.legend(legend, numpoints=1)  # pylint: disable=cell-var-from-loop
        fig.tight_layout()

      calibration_curve_summary = plot.Curve(
          name=tag_str,
          figsize=(10, 8),
          xs=mean_predicted_accuracy,
          ys=mean_empirical_accuracy,
          setter=_CalibrationSetter,
          marker='.',
          markersize=14,
          linestyle='-',
          linewidth=2,
          alpha=0.5)

      ece = ExpectedCalibrationError(mean_predicted_accuracy,
                                     mean_empirical_accuracy,
                                     num_examples_per_bin)

      ece_summary = tf.Summary(value=[
          tf.Summary.Value(
              tag='{}/{}/calibration_ece'.format(name, classname),
              simple_value=ece)
      ])
      summaries.extend([calibration_curve_summary, ece_summary])
    return summaries
