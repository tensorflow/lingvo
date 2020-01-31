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
"""Metadata for KITTI dataset employed in evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo.tasks.car import evaluation_metadata


class KITTIMetadata(evaluation_metadata.EvaluationMetadata):
  """Metadata describing the KITTI dataset used for evaluation."""

  def __init__(self):
    super(KITTIMetadata, self).__init__('kitti')

  def ClassNames(self):
    """Returns a list of human-interpretable strings."""
    return [
        'Background', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
        'Cyclist', 'Tram', 'Misc', 'DontCare'
    ]

  def DifficultyLevels(self):
    """Dictionary of difficulty level strings to int32 indices."""
    return {'hard': 1, 'moderate': 2, 'easy': 3}

  def IoUThresholds(self):
    return {
        'Car': 0.7,
        'Pedestrian': 0.5,
        'Cyclist': 0.5,
    }

  def IgnoreClassIndices(self):
    """Dictionary of int32 indices for the classes that should be ignored."""
    # A detection that matches with a groundtruth bbox of any neighbor class
    # will not be considered as false positive in eval.
    return {
        self.ClassNames().index('Car'): [self.ClassNames().index('Van')],
        self.ClassNames().index('Pedestrian'): [
            self.ClassNames().index('Person_sitting')
        ],
    }

  def NumberOfPrecisionRecallPoints(self):
    """Number of points on the precision-recall curve."""
    return 41

  def MaximumDistance(self):
    """Maximum empirically observed Euclidean distance in world coordinates."""
    return 80.0

  def DistanceBinWidth(self):
    """The width of each bin for Euclidean distance in world coordinates."""
    return 10.0

  def MaximumNumberOfPoints(self):
    """Maximum empirically observed number of points in bounding box."""
    return 3000.0

  def NumberOfPointsBins(self):
    """Number of logarithmically space bins for examining points."""
    return 20

  def NumberOfRotationBins(self):
    """Number of linear spaced rotations to bin."""
    return 10

  def NumberOfCalibrationBins(self):
    """Number of linear spaced calibration bins."""
    return 15

  def MinHeight2D(self):
    """Minimum height of detections to be evaluated."""
    return {'hard': 25, 'moderate': 25, 'easy': 40}
