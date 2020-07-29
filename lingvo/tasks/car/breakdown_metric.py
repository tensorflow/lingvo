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
"""Library for calculating precision recall conditioned on a variate.

ByDistance: Calculate precision recall based on distance in world coordinates.
ByRotation: Calculate precision recall based on rotation in world coordinates.
ByNumPoints: Calculate maximum recall based on number of points in bounding box.
ByDifficulty: Calculate precision recall based on difficulty.
"""

from lingvo import compat as tf
from lingvo.core import hyperparams
from lingvo.core import plot
from lingvo.tasks.car import calibration_processing as calibration
import numpy as np


def _FindRecallAtGivenPrecision(precision_recall, precision_level):
  """Find the recall at given precision level.

  Args:
    precision_recall: np.array of shape [n, m, 2] where n is the number of
      classes, m is then number of values in the curve, 2 indexes between
      precision [0] and recall [1]. np.float32.
    precision_level: float32. Selected precision level between (0, 1).
      Typically, this may be 0.95.

  Returns:
    recall: np.array of shape [n] consisting of recall for all classes
      where the values are 0.0 if a given precision is never achieved.
  """
  # The method for computing precision-recall inserts precision = 0.0
  # when a particular recall value has not been achieved. The maximum
  # recall value is therefore the highest recall value when the associated
  # precision > 0.
  assert len(precision_recall.shape) == 3, 'Invalid precision recall curve.'
  assert precision_recall.shape[-1] == 2, 'Invalid precision recall curve.'
  assert precision_level > 0.0, 'Precision must be greater then 0.'
  assert precision_level < 1.0, 'Precision must be less then 1.'
  num_classes = precision_recall.shape[0]

  recall = np.zeros(shape=(num_classes), dtype=np.float32)
  for i in range(num_classes):
    precisions = precision_recall[i, :, 0]
    recalls = precision_recall[i, :, 1]
    indices_at_precision_level = np.flatnonzero(precisions >= precision_level)
    if indices_at_precision_level.size > 0:
      recall[i] = np.max(recalls[indices_at_precision_level])
  return recall


def _FindMaximumRecall(precision_recall):
  """Find the maximum recall in all precision recall curves.

  Args:
    precision_recall: np.array of shape [n, m, 2] where n is the number of
      classes, m is then number of values in the curve, 2 indexes between
      precision [0] and recall [1]. np.float32.

  Returns:
    max_recall: np.array of shape [n] consisting of max recall for all classes
      where the values are 0.0 if objects are found.
  """
  # The method for computing precision-recall inserts precision = 0.0
  # when a particular recall value has not been achieved. The maximum
  # recall value is therefore the highest recall value when the associated
  # precision > 0.
  assert len(precision_recall.shape) == 3, 'Invalid precision recall curve.'
  assert precision_recall.shape[-1] == 2, 'Invalid precision recall curve.'
  num_classes = precision_recall.shape[0]

  max_recall = np.zeros(shape=(num_classes), dtype=np.float32)

  valid_precisions = precision_recall[:, :, 0] > 0.0
  for i in range(num_classes):
    valid_precisions_indices = valid_precisions[i, :]
    if np.any(valid_precisions_indices):
      max_recall[i] = np.max(precision_recall[i, valid_precisions_indices, 1])
  return max_recall


class BreakdownMetric:
  """Base class for calculating precision recall conditioned on a variate."""

  @classmethod
  def Params(cls):
    p = hyperparams.Params()
    p.Define(
        'metadata', None,
        'Class obeying EvaluationMetadata interface consisting of '
        'parameters specifying the details of the evaluation.')
    return p

  def __init__(self, p):
    self.params = p
    assert p.metadata is not None
    self._histogram = np.zeros(
        shape=(self.NumBinsOfHistogram(), p.metadata.NumClasses()),
        dtype=np.int32)
    self._values = np.zeros(
        shape=(self.NumBinsOfHistogram(), 1), dtype=np.float32)
    self._cumulative_distribution = {}
    for l in range(p.metadata.NumClasses()):
      self._cumulative_distribution[l] = []
    self._average_precisions = {}
    self._precision_recall = {}
    self._calibration = {}
    self._classnames = p.metadata.ClassNames()
    self._classids = p.metadata.EvalClassIndices()

    # We only want to calculate calibration for the lowest difficulty setting
    # which will include *all* data. Including all data is due to the way
    # _GetData() works in the kitti_ap_metric.
    difficulties = p.metadata.DifficultyLevels()
    lowest_difficulty_str = min(difficulties, key=lambda k: difficulties[k])
    self._calibration[lowest_difficulty_str] = (
        calibration.CalibrationCalculator(p.metadata))

  def NumBinsOfHistogram(self):
    """Returns int32 of number of bins in histogram."""
    return NotImplementedError()

  def ComputeMetrics(self, compute_metrics_fn):
    """Compute precision-recall analysis conditioned on particular metric.

    Args:
      compute_metrics_fn: Function that that calculates precision-recall metrics
        and accepts named arguments for conditioning. Typically, this would be
        APMetrics._ComputeFinalMetrics().

    Returns:
       nothing
    """
    del compute_metrics_fn
    return NotImplementedError()

  def GenerateSummaries(self, name):
    """Generate list of image summaries plotting precision-recall analysis.

    Args:
      name: string providing scope

    Returns:
      list of image summaries
    """
    del name
    return NotImplementedError()

  def Discretize(self, values):
    """Discretize statistics into integer values.

    Args:
      values: 1-D np.array of variate to be discretized.

    Returns:
      1-D np.array of int32 ranging within [0, cls.NumOfBinsOfHistogram()]
    """
    del values
    return NotImplementedError()

  def _AccumulateHistogram(self, statistics=None, labels=None):
    """Accumulate histogram of binned statistic by label.

    Args:
      statistics: int32 np.array of shape [K, 1] of binned statistic
      labels: int32 np.array of shape [K, 1] of labels

    Returns:
      nothing
    """
    assert np.issubdtype(statistics.dtype, np.integer)
    if not statistics.size:
      return
    p = self.params
    assert np.max(statistics) < self._histogram.shape[0], (
        'Histogram shape too small %d vs %d' %
        (np.max(statistics), self._histogram.shape[0]))
    for l in range(p.metadata.NumClasses()):
      indices = np.where(labels == l)[0]
      for s in statistics[indices]:
        self._histogram[s, l] += 1

  def _AccumulateCumulative(self, statistics=None, labels=None):
    """Accumulate cumulative of real-valued statistic by label.

    Args:
      statistics: float32 np.array of shape [K, 1] of statistic
      labels: int32 np.array of shape [K, 1] of labels

    Returns:
      nothing
    """
    p = self.params
    for l in range(p.metadata.NumClasses()):
      indices = np.where(labels == l)[0]
      if indices.size:
        self._cumulative_distribution[l].extend(statistics[indices].tolist())

  def AccumulateCumulative(self, result):
    """Accumulate cumulative of real-valued statistic by label.

    Args:
      result: A NestedMap with these fields:
        .labels: [N]. Groundtruth boxes' labels.
        .bboxes: [N, 7]. Groundtruth boxes coordinates.
        .difficulties: [N]. Groundtruth boxes difficulties.
        .num_points: [N]. Number of laser points in bounding boxes.

    Returns:
      nothing
    """
    pass


def ByName(breakdown_metric_name):
  """Return a BreakdownMetric class by name."""
  breakdown_mapping = {
      'distance': ByDistance,
      'num_points': ByNumPoints,
      'rotation': ByRotation,
      'difficulty': ByDifficulty
  }
  if breakdown_metric_name not in breakdown_mapping:
    raise ValueError('Invalid breakdown name: %s, valid names are %s' %
                     (breakdown_metric_name, list(breakdown_mapping.keys())))
  return breakdown_mapping[breakdown_metric_name]


class ByDistance(BreakdownMetric):
  """Calculate average precision as function of distance."""

  def NumBinsOfHistogram(self):
    p = self.params
    return int(
        np.rint(p.metadata.MaximumDistance() / p.metadata.DistanceBinWidth()))

  @classmethod
  def _CalculateEuclideanDistanceFromOrigin(cls, bboxes):
    """Calculate the Euclidean distance from the origin for each bounding box.

    Note that the LabelsExtractor originally returns groundtruth_bboxes of
    shape [N, 7] where N is the number of bounding boxes. The last axis is
    ordered [x, y, z, w, h, d, phi]. Hence, the Euclidean distance to the origin
    is the L2 norm of the first 3 entries.

    Args:
      bboxes: [N, 7] np.float of N bounding boxes. See details above.

    Returns:
      np.array [N] of Euclidean distances.
    """
    positions = bboxes[:, :3]
    # Note that we employ axis=1 to compute the norm over [x, y, z] for each
    # of N bounding boxes.
    return np.linalg.norm(positions, axis=1, keepdims=False)

  def Discretize(self, bboxes):
    p = self.params
    distances = self._CalculateEuclideanDistanceFromOrigin(bboxes)
    distances_binned = np.digitize(
        distances,
        np.arange(0.0, p.metadata.MaximumDistance(),
                  p.metadata.DistanceBinWidth()))
    # index == 0 corresponds to distances outside less than 0.0. Since this is
    # not possible, we discard this possibility and make the output 0 indexed to
    # match the behavior of np.histogram().
    assert np.all(distances_binned > 0.0), 'Euclidean distance is negative.'
    distances_binned -= 1
    return distances_binned

  def AccumulateHistogram(self, result):
    distances = self.Discretize(result.bboxes)
    self._AccumulateHistogram(statistics=distances, labels=result.labels)

  def ComputeMetrics(self, compute_metrics_fn):
    tf.logging.info('Calculating by distance: start')
    p = self.params
    for d in range(self.NumBinsOfHistogram()):
      value_at_histogram = (
          d * p.metadata.DistanceBinWidth() +
          p.metadata.DistanceBinWidth() / 2.0)
      self._values[d] = value_at_histogram

      metrics = compute_metrics_fn(distance=d)
      curves = metrics['curves']
      self._precision_recall[d] = np.array([c['pr'] for c in curves])
    assert len(self._values) == len(list(self._precision_recall.keys()))
    tf.logging.info('Calculating by distance: finished')

  def GenerateSummaries(self, name):
    """Generate an image summary for AP versus distance by class."""
    p = self.params

    legend_names = []
    for j in p.metadata.EvalClassIndices():
      legend_names.append(p.metadata.ClassNames()[j])

    num_distances = self._values.shape[0]
    ys = np.zeros(
        shape=(num_distances, len(p.metadata.EvalClassIndices())),
        dtype=np.float32)

    for dist in self._precision_recall:
      ys[dist, :] = _FindMaximumRecall(self._precision_recall[dist])

    def _Setter(fig, axes):
      """Configure the plot for mAP versus distance."""
      axes.grid(b=True)
      fontsize = 14
      for i, j in enumerate(p.metadata.EvalClassIndices()):
        for d, x in enumerate(self._values):
          h = self._histogram[d][j]
          y = min(ys[d][i] + 0.03, 1.0)
          if h > 0:
            text_label = '{} {}s'.format(h, legend_names[i].lower()[:3])
            axes.text(x, y, text_label, fontdict={'fontsize': fontsize - 2})

      axes.set_xlabel('distance (world coordinates)', fontsize=fontsize)
      axes.set_xticks(
          np.arange(
              0.0,
              p.metadata.MaximumDistance() + p.metadata.DistanceBinWidth(),
              p.metadata.DistanceBinWidth()))
      axes.set_ylabel('maximum recall', fontsize=fontsize)
      axes.set_ylim([-0.01, 1.05])
      axes.set_yticks(np.arange(0.0, 1.05, 0.1))
      axes.legend([name.lower() for name in legend_names], numpoints=1, loc=3)
      fig.tight_layout()

    tag_str = '{}/recall_by_distance'.format(name)
    image_summary = plot.Curve(
        name=tag_str,
        figsize=(10, 8),
        xs=self._values,
        ys=ys,
        setter=_Setter,
        marker='.',
        markersize=14,
        linestyle='-',
        linewidth=2,
        alpha=0.5)
    return [image_summary]


class ByNumPoints(BreakdownMetric):
  """Calculate average precision as function of the number of points."""

  def NumBinsOfHistogram(self):
    return self.params.metadata.NumberOfPointsBins()

  def _LogSpacedBinEdgesofPoints(self):
    p = self.params
    return np.logspace(
        np.log10(1.0), np.log10(p.metadata.MaximumNumberOfPoints()),
        p.metadata.NumberOfPointsBins() + 1)

  def Discretize(self, num_points):
    num_points_binned = np.digitize(num_points,
                                    self._LogSpacedBinEdgesofPoints())
    # index == 0 corresponds to boxes with 0 points. Because we plot everything
    # logarithmically, this is a pain in the buttocks. For simplicity, we merely
    # accumulate the boxes with 0 points into the first bin.
    num_points_binned[num_points_binned == 0] = 1
    num_bins = len(self._LogSpacedBinEdgesofPoints())
    # index == len(self._LogSpacedBinEdgesofPoints()) corresponds to
    # points with to points outside of the range of the last edge. We map
    # these points back to the final bucket for simplicity.
    num_points_binned[num_points_binned == num_bins] -= 1
    # There is an inconsistency between how np.digitize() and np.histogram()
    # index their bins and this is due to the fact that index == 0 is reserved
    # for examples less than the minimum bin edge.
    num_points_binned -= 1
    return num_points_binned

  def AccumulateHistogram(self, result):
    num_points = self.Discretize(result.num_points)
    self._AccumulateHistogram(statistics=num_points, labels=result.labels)

  def AccumulateCumulative(self, result):
    self._AccumulateCumulative(
        statistics=result.num_points, labels=result.labels)

  def ComputeMetrics(self, compute_metrics_fn):
    tf.logging.info('Calculating by number of points: start')
    # Note that we skip the last edge as the number of edges is one greater
    # then the number of bins.
    self._values = self._LogSpacedBinEdgesofPoints()[:-1]
    for n, _ in enumerate(self._values):
      metrics = compute_metrics_fn(num_points=n)
      curves = metrics['curves']
      self._precision_recall[n] = np.array([c['pr'] for c in curves])
    assert len(self._values) == len(list(self._precision_recall.keys()))
    tf.logging.info('Calculating by number of points: finished')

  def GenerateSummaries(self, name):
    """Generate an image summary for max recall by number of points by class."""
    image_summaries = self._GenerateCumulativeSummaries(name)
    p = self.params

    legend_names = []
    for j in p.metadata.EvalClassIndices():
      legend_names.append(p.metadata.ClassNames()[j])

    num_points_bins = self._values.shape[0]
    ys = np.zeros(
        shape=(num_points_bins, len(p.metadata.EvalClassIndices())),
        dtype=np.float32)

    for num_points in self._precision_recall:
      ys[num_points, :] = _FindMaximumRecall(self._precision_recall[num_points])

    def _Setter(fig, axes):
      """Configure the plot for max recall versus number of points."""
      axes.grid(b=True)
      fontsize = 14
      for i, j in enumerate(p.metadata.EvalClassIndices()):
        for n, x in enumerate(self._values):
          h = self._histogram[n][j]
          y = min(ys[n][i] + 0.03, 1.0)
          if h > 0:
            text_label = '{} {}s'.format(h, legend_names[i].lower()[:3])
            axes.text(x, y, text_label, fontdict={'fontsize': fontsize - 2})

      axes.set_xlabel('number of points', fontsize=fontsize)
      axes.set_xticks(self._values)
      axes.set_ylabel('maximum recall', fontsize=fontsize)
      axes.set_ylim([-0.01, 1.05])
      axes.set_xlim([(1.0 / 1.3) * self._values[0], 1.3 * self._values[-1]])
      axes.set_yticks(np.arange(0.0, 1.05, 0.1))
      axes.set_xscale('log')
      axes.legend([name.lower() for name in legend_names],
                  numpoints=1,
                  loc='upper left')
      fig.tight_layout()

    tag_str = '{}/recall_by_num_points'.format(name)
    image_summary = plot.Curve(
        name=tag_str,
        figsize=(10, 8),
        xs=self._values,
        ys=ys,
        setter=_Setter,
        marker='.',
        markersize=14,
        linestyle='-',
        linewidth=2,
        alpha=0.5)
    image_summaries.append(image_summary)
    return image_summaries

  def _GenerateCumulativeSummaries(self, name):
    """Generate an image summary for CDF of a variate."""
    xs = []
    ys = []
    num_zeros = []
    legend_names = []
    min_value = 5.0
    p = self.params

    for i, j in enumerate(p.metadata.EvalClassIndices()):
      legend_names.append(p.metadata.ClassNames()[j])
      if len(self._cumulative_distribution[j]) > min_value:
        self._cumulative_distribution[j].sort()
        x = np.array(self._cumulative_distribution[j])
        nonzeros = np.flatnonzero(x)
        cdf = np.arange(x.size).astype(np.float) / x.size
        xs.append(x)
        ys.append(cdf)
        num_zeros.append(x.size - nonzeros.size)
      else:
        xs.append(None)
        ys.append(None)
        num_zeros.append(None)

    image_summaries = []
    for i, j in enumerate(p.metadata.EvalClassIndices()):
      classname = p.metadata.ClassNames()[j]

      def _Setter(fig, axes):
        """Configure the plot for CDF of the variate."""
        axes.grid(b=False)
        fontsize = 14

        axes.set_ylim([0, 1.05])
        axes.set_xlim([1.0, 11500])
        axes.set_ylabel('cumulative distribution', fontsize=fontsize)
        axes.set_xlabel('number of points', fontsize=fontsize)
        axes.set_xscale('log')
        legend_text = '{} {}s ({} contain zero points)'.format(
            xs[i].size,
            p.metadata.ClassNames()[j].lower(), num_zeros[i])
        axes.legend({legend_text}, loc='upper left')
        fig.tight_layout()

      if xs[i] is not None:
        tag_str = '{}/{}/cdf_of_num_points'.format(name, classname)
        image_summary = plot.Curve(
            name=tag_str,
            figsize=(10, 8),
            xs=xs[i],
            ys=ys[i],
            setter=_Setter,
            marker='',
            linestyle='-',
            linewidth=2,
            alpha=0.5)
        image_summaries.append(image_summary)
    return image_summaries


class ByRotation(BreakdownMetric):
  """Calculate average precision as function of rotation."""

  def NumBinsOfHistogram(self):
    return self.params.metadata.NumberOfRotationBins()

  def _CalculateRotation(self, bboxes):
    """Calculate rotation angle mod between (0, 2 * pi) for each box.

    Args:
      bboxes: [N, 7] np.float of N bounding boxes. See details above.

    Returns:
      np.array [N] of rotation angles in radians.
    """
    if not bboxes.size:
      return np.empty_like(bboxes)
    p = self.params
    # Although groundtruth is constrained to be in [-pi, pi], predictions are
    # unbounded. We map all predictions to their equivalent value in [-pi, pi].
    rotations = np.copy(bboxes[:, -1])
    rotations += np.pi
    rotations = np.mod(rotations, 2.0 * np.pi)
    rotations -= np.pi
    # Now we remove ambiguity in 180 degree rotations as measured by our IOU
    # calculations by mapping everything to [0, pi] range.
    rotations = np.where(rotations > 0.0, rotations, rotations + np.pi)
    # Floating numerical issues can surface occasionally particularly within
    # subsequent binning. The clipping makes these operations reliable.
    epsilon = 1e-5
    rotations = np.clip(rotations, epsilon,
                        p.metadata.MaximumRotation() - epsilon)
    return rotations

  def Discretize(self, bboxes):
    rotations = self._CalculateRotation(bboxes)
    p = self.params
    bin_width = (
        p.metadata.MaximumRotation() / float(self.NumBinsOfHistogram()))
    # TODO(shlens): Consider merging the entries with -1 and 0 bin index
    # because rotation is circular.
    rotations_binned = np.digitize(
        rotations, np.arange(0.0, p.metadata.MaximumRotation(), bin_width))
    # index == 0 corresponds to distances outside less than 0.0. Since this is
    # not possible, we discard this possibility and make the output 0 indexed to
    # match the behavior of np.histogram().
    assert np.all(rotations_binned > 0.0), ('Rotation is negative: %s' %
                                            rotations_binned)
    rotations_binned -= 1
    return rotations_binned

  def AccumulateHistogram(self, result):
    rotations = self.Discretize(result.bboxes)
    self._AccumulateHistogram(statistics=rotations, labels=result.labels)

  def ComputeMetrics(self, compute_metrics_fn):
    tf.logging.info('Calculating by rotation: start')
    p = self.params
    self._values = np.zeros(
        shape=(self.NumBinsOfHistogram(), 1), dtype=np.float32)
    bin_width = (
        p.metadata.MaximumRotation() / float(self.NumBinsOfHistogram()))
    for r in range(self.NumBinsOfHistogram()):
      # Calculate the center of the histogram bin.
      value_at_histogram = r * bin_width + bin_width / 2.0
      self._values[r] = value_at_histogram
      metrics = compute_metrics_fn(rotation=r)
      curves = metrics['curves']
      self._precision_recall[r] = np.array([c['pr'] for c in curves])
    assert len(self._values) == len(list(self._precision_recall.keys()))
    tf.logging.info('Calculating by rotation: finished')

  def GenerateSummaries(self, name):
    """Generate an image summary for AP versus rotation by class."""
    p = self.params
    legend_names = []
    for j in p.metadata.EvalClassIndices():
      legend_names.append(p.metadata.ClassNames()[j])

    num_rotations = self._values.shape[0]
    rotation_in_degrees = self._values * 180.0 / np.pi
    ys = np.zeros(
        shape=(num_rotations, len(p.metadata.EvalClassIndices())),
        dtype=np.float32)

    for rotation in self._precision_recall:
      ys[rotation, :] = _FindMaximumRecall(self._precision_recall[rotation])

    def _Setter(fig, axes):
      """Configure the plot for max recall versus distance."""
      axes.grid(b=True)
      fontsize = 14
      for i, j in enumerate(p.metadata.EvalClassIndices()):
        for r, x in enumerate(rotation_in_degrees):
          h = self._histogram[r][j]
          y = min(ys[r][i] + 0.03, 1.0)
          text_label = '{} {}s'.format(h, legend_names[i].lower()[:3])
          axes.text(x, y, text_label, fontdict={'fontsize': fontsize - 2})

      axes.set_xlabel('rotation (degrees)', fontsize=fontsize)
      bin_width = (
          p.metadata.MaximumRotation() / float(self.NumBinsOfHistogram()))
      axes.set_xticks(
          np.arange(0.0,
                    p.metadata.MaximumRotation() + bin_width, bin_width) *
          180.0 / np.pi)
      axes.set_ylabel('maximum recall', fontsize=fontsize)
      axes.set_ylim([-0.01, 1.05])
      axes.set_yticks(np.arange(0.0, 1.05, 0.1))
      axes.set_xlim([0.0, 180.0])
      axes.legend([name.lower() for name in legend_names],
                  numpoints=1,
                  loc='upper right')
      fig.tight_layout()

    tag_str = '{}/recall_by_rotation'.format(name)
    image_summary = plot.Curve(
        name=tag_str,
        figsize=(10, 8),
        xs=rotation_in_degrees,
        ys=ys,
        setter=_Setter,
        marker='.',
        markersize=14,
        linestyle='-',
        linewidth=2,
        alpha=0.5)
    return [image_summary]


class ByDifficulty(BreakdownMetric):
  """Calculate average precision as function of difficulty."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('ap_key', 'ap', 'Metrics key for the AP value.')
    p.Define('pr_key', 'pr', 'Metrics key for the PR value.')
    return p

  def NumBinsOfHistogram(self):
    return len(self.params.metadata.DifficultyLevels()) + 1

  def Discretize(self, difficulties):
    return difficulties.astype(np.int32)

  def AccumulateHistogram(self, result):
    difficulties = self.Discretize(result.difficulties)
    self._AccumulateHistogram(statistics=difficulties, labels=result.labels)

  def ComputeMetrics(self, compute_metrics_fn):
    p = self.params
    tf.logging.info('Calculating by difficulty: start')
    for difficulty in self.params.metadata.DifficultyLevels():
      metrics = compute_metrics_fn(difficulty=difficulty)
      scalars = metrics['scalars']
      curves = metrics['curves']
      self._average_precisions[difficulty] = [s[p.ap_key] for s in scalars]
      self._precision_recall[difficulty] = np.array(
          [c[p.pr_key] for c in curves])

      # Only KITTI metrics contains calibration data,
      # And we only compute calibration for the lowest difficulty level
      # to include all data.
      if difficulty in self._calibration:
        self._calibration[difficulty].Calculate(metrics)

    tf.logging.info('Calculating by difficulty: finished')

  def GenerateSummaries(self, name):
    """Generate an image summary for PR by difficulty and for calibration.

    Args:
      name: str, name of summary.

    Returns:
      list of summaries
    """

    legend = {}
    p = self.params
    for class_id in p.metadata.EvalClassIndices():
      legend[class_id] = []
      for difficulty, i in p.metadata.DifficultyLevels().items():
        num_objects = self._histogram[i][class_id]
        legend[class_id].append('%s (%d)' % (difficulty, num_objects))

    summaries = []
    for i, j in enumerate(p.metadata.EvalClassIndices()):

      def _PRSetter(fig, axes):
        """Configure the plot for precision recall."""
        ticks = np.arange(0, 1.05, 0.1)
        axes.grid(b=False)
        axes.set_xlabel('Recall')
        axes.set_xticks(ticks)
        axes.set_ylabel('Precision')
        axes.set_yticks(ticks)
        # pylint: disable=undefined-loop-variable
        axes.legend(legend[j], numpoints=1)  # pylint: disable=cell-var-from-loop
        # pylint: enable=undefined-loop-variable
        fig.tight_layout()

      classname = p.metadata.ClassNames()[j]
      # Generate Precision-Recall curves.
      rs = []
      ps = []
      for difficulty in p.metadata.DifficultyLevels():
        ps += [self._precision_recall[difficulty][i][:, 0]]
        rs += [self._precision_recall[difficulty][i][:, 1]]
      tag_str = '{}/{}/PR'.format(name, classname)
      image_summary = plot.Curve(
          name=tag_str,
          figsize=(10, 8),
          xs=rs[0],
          ys=np.array(ps).T,
          setter=_PRSetter,
          marker='.',
          markersize=14,
          linestyle='-',
          linewidth=2,
          alpha=0.5)
      summaries.append(image_summary)

    for difficulty, c in self._calibration.items():
      # Note that we only generate a calibration for a single difficulty level.
      calibration_summaries = c.Summary(name)
      for calibration_summary in calibration_summaries:
        summaries.append(calibration_summary)

    # Generate scalar summaries for the various recalls for each difficulty.
    for difficulty in p.metadata.DifficultyLevels():
      max_recall = _FindMaximumRecall(self._precision_recall[difficulty])
      for i, j in enumerate(p.metadata.EvalClassIndices()):
        classname = p.metadata.ClassNames()[j]
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='{}/{}/max_recall_{}'.format(name, classname, difficulty),
                simple_value=max_recall[i])
        ])
        summaries.append(summary)

      for precision_level in p.metadata.RecallAtPrecision():
        recall_at_precision = _FindRecallAtGivenPrecision(
            self._precision_recall[difficulty], precision_level)
        for i, j in enumerate(p.metadata.EvalClassIndices()):
          classname = p.metadata.ClassNames()[j]
          summary = tf.Summary(value=[
              tf.Summary.Value(
                  tag='{}/{}/recall_{}_{}'.format(
                      name, classname, int(precision_level * 100), difficulty),
                  simple_value=recall_at_precision[i])
          ])
          summaries.append(summary)
    return summaries
