# Lint as: python3
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
"""Base class specifying metadata used in evaluation."""

import numpy as np


class EvaluationMetadata:
  """Interface for defining metadata of dataset."""

  def __init__(self, name):
    self.name = name

  def ClassNames(self):
    """Returns a list of human-interpretable strings."""
    raise NotImplementedError()

  def LabelMap(self):
    """Return a label map of int -> str for each class."""
    return dict(zip(range(self.NumClasses()), self.ClassNames()))

  def NumClasses(self):
    """Total number of classes for the dataset."""
    return len(self.ClassNames())

  def DifficultyLevels(self):
    """Dictionary of difficulty level strings to int32 indices."""
    raise NotImplementedError()

  def EvalClassIndices(self):
    """List of int32 indices for the classes that should be evaled."""
    eval_classes = sorted(self.IoUThresholds().keys())
    return [self.ClassNames().index(name) for name in eval_classes]

  def IoUThresholds(self):
    """Dictionary of IoU thresholds for every evaluated class.

    The keys of the dictionary are used to compute EvalClassIndices().
    """
    raise NotImplementedError()

  def IgnoreClassIndices(self):
    """List of int32 indices for the classes that should be ignored.

    A detection that matches with a groundtruth bbox of any neighbor class will
    not be considered as false positive in eval.
    """
    raise NotImplementedError()

  def NumberOfPrecisionRecallPoints(self):
    """Number of points on the precision-recall curve."""
    raise NotImplementedError()

  def MaximumDistance(self):
    """Maximum empirically observed Euclidean distance in world coordinates."""
    raise NotImplementedError()

  def DistanceBinWidth(self):
    """The width of each bin for Euclidean distance in world coordinates."""
    raise NotImplementedError()

  def MaximumNumberOfPoints(self):
    """Maximum empirically observed number of points in bounding box."""
    raise NotImplementedError()

  def NumberOfPointsBins(self):
    """Number of logarithmically space bins for examining points."""
    raise NotImplementedError()

  def MaximumRotation(self):
    """Maximum rotation angle in world coordinates."""
    return np.pi

  def NumberOfRotationBins(self):
    """Number of linear spaced rotations to bin."""
    raise NotImplementedError()

  def MinHeight2D(self):
    """Minimum height of detections to be evaluated.

    Returns:
      A dictionary of difficulty level strings to ints that are height
    thresholds for each level.
    """
    return NotImplementedError()

  def RecallAtPrecision(self):
    """Report the recall at a given precision level.."""
    return [0.50, 0.95]
