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
import numpy as np
from six.moves import range


class CalibrationCalculator(object):
  """Base class for calculating calibration on a prediction."""

  def __init__(self, metadata):
    self._metadata = metadata
    self._num_calibration_bins = self._metadata.NumberOfCalibrationBins()
    self._calibration_by_class = None
    self._classnames = self._metadata.ClassNames()
    self._classids = self._metadata.EvalClassIndices()

  def Calculate(self, metrics):
    """Calculate metrics for calibration."""
    if 'calibrations' not in metrics:
      tf.logging.info('CalibrationProcessing invoked but no metrics available '
                      'for calculating calibration.')
      return

    calibration_by_class = {}
    for i, c in enumerate(metrics['calibrations']):
      classid = self._classids[i]
      classname = self._classnames[classid]

      if np.all(np.isnan(c['calibrations'])) or c['calibrations'].size == 0:
        tf.logging.info('Skipping %s for calibration calculation because no '
                        'output provided.' % classname)
        continue
      tf.logging.info('Calculating calibration for %s: %d items.' %
                      (classname, len(c['calibrations'])))
      calibration_by_class[classname] = []

      # Bin the hits and scores based on the scores.
      edges = np.linspace(0.0, 1.0, self._num_calibration_bins + 1)

      # Ensure that all counts are greater then zero and less then or equal
      # to 1.0 to guarantee that all scores are counted.
      scores_and_hits = np.clip(c['calibrations'], 1e-10, 1.0)
      bin_indices = np.digitize(scores_and_hits[:, 0], edges, right=True)

      for j in range(self._num_calibration_bins + 1):
        if j == 0:
          continue
        indices = np.where(bin_indices == j)[0]
        mean_predicted_accuracy = (edges[j - 1] + edges[j]) / 2.0
        mean_empirical_accuracy = np.mean(scores_and_hits[indices, 1])
        num_example = len(indices)
        calibration_by_class[classname].append(
            [mean_predicted_accuracy, mean_empirical_accuracy, num_example])

      calibration_by_class[classname] = np.array(
          calibration_by_class[classname])
      tf.logging.info('Finished calculating calibration for %s.' % classname)
      self._calibration_by_class = calibration_by_class

  def Summary(self, name):
    """Generate an image summary for the calibration curve."""
    # TODO(shlens, rofls): Implement me.
    del name
    return
