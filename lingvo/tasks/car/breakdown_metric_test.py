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
"""Tests for breakdown_metric."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import breakdown_metric
from lingvo.tasks.car import kitti_ap_metric
from lingvo.tasks.car import kitti_metadata
import numpy as np

FLAGS = tf.flags.FLAGS


class BreakdownMetricTest(test_utils.TestCase):

  def _GenerateRandomBBoxes(self, num_bboxes):
    xyz = np.random.uniform(low=-1.0, high=1.0, size=(num_bboxes, 3))
    dimension = np.random.uniform(low=-1, high=1.0, size=(num_bboxes, 3))
    rotation = np.random.uniform(low=-np.pi, high=np.pi, size=(num_bboxes, 1))
    bboxes = np.concatenate([xyz, dimension, rotation], axis=-1)
    return bboxes

  def _GenerateBBoxesAtDistanceAndRotation(self, num_boxes, distance, rotation):
    bboxes = np.zeros(shape=(num_boxes, 7))
    bboxes[:, -1] = rotation
    bboxes[:, 0] = distance
    return bboxes

  def _GenerateMetricsWithTestData(self, num_classes):
    metadata = kitti_metadata.KITTIMetadata()
    num_bins_of_distance = int(
        np.rint(metadata.MaximumDistance() / metadata.DistanceBinWidth()))
    num_bins_of_rotation = metadata.NumberOfRotationBins()
    num_bins_of_points = metadata.NumberOfPointsBins()

    # Generate ground truth bounding boxes with prescribed labels, distances,
    # rotations and number of points.
    expected_objects_at_distance = np.random.randint(
        low=0, high=8, size=(num_classes, num_bins_of_distance), dtype=np.int32)
    expected_objects_at_rotation = np.zeros(
        shape=(num_classes, num_bins_of_rotation), dtype=np.int32)
    # Note that we need preserve the same number of objects for each label.
    expected_objects_at_points = np.zeros(
        shape=(num_classes, num_bins_of_points), dtype=np.int32)
    prob = 1.0 / float(num_bins_of_points)
    for c in range(num_classes):
      num_objects_for_class = np.sum(expected_objects_at_distance[c, :])
      expected_objects_at_points[c, :] = np.random.multinomial(
          num_objects_for_class, pvals=num_bins_of_points * [prob])
    # Zero out the number of boxes in the background class.
    expected_objects_at_distance[0, :] = 0
    expected_objects_at_points[0, :] = 0
    expected_objects_at_rotation[0, :] = 0

    bboxes = []
    labels = []
    num_points = []
    bin_width = (
        metadata.MaximumRotation() / float(metadata.NumberOfRotationBins()))
    # Note that we always skip 'Background' class 0.
    for label in range(1, num_classes):
      for distance_index in range(num_bins_of_distance):
        distance = (
            distance_index * metadata.DistanceBinWidth() +
            metadata.DistanceBinWidth() / 2.0)
        num_box = expected_objects_at_distance[label, distance_index]

        if num_box > 0:
          rotation_index = np.random.randint(num_bins_of_rotation)
          expected_objects_at_rotation[label, rotation_index] += num_box
          rotation = rotation_index * bin_width + bin_width / 2.0

          bboxes.append(
              self._GenerateBBoxesAtDistanceAndRotation(num_box, distance,
                                                        rotation))
          labels.append(label * np.ones(shape=[num_box], dtype=np.int32))

      point_bin_edges = np.logspace(
          np.log10(1.0), np.log10(metadata.MaximumNumberOfPoints()),
          metadata.NumberOfPointsBins() + 1)
      for point_index in range(num_bins_of_points):
        num_box = expected_objects_at_points[label, point_index]
        for _ in range(num_box):
          points = (point_bin_edges[point_index] +
                    point_bin_edges[point_index + 1]) / 2.0
          num_points.append([points])

    bboxes = np.concatenate(bboxes)
    labels = np.concatenate(labels)
    num_points = np.concatenate(num_points)

    # Generate dummy predictions as placeholders for the API.
    num_predictions = 9
    prediction_scores = np.random.uniform(size=[num_classes, num_predictions])
    prediction_bboxes = self._GenerateRandomBBoxes(
        num_predictions * num_classes).reshape(
            (num_classes, num_predictions, 7))

    # Update the metrics.
    metric_names = ['rotation', 'num_points', 'distance']
    ap_params = kitti_ap_metric.KITTIAPMetrics.Params(metadata).Set(
        breakdown_metrics=metric_names)
    metrics = ap_params.Instantiate()
    metrics.Update(
        'dummy_image1',
        py_utils.NestedMap(
            groundtruth_labels=labels,
            groundtruth_bboxes=bboxes,
            groundtruth_difficulties=np.ones(shape=(bboxes.shape[0])),
            groundtruth_num_points=num_points,
            detection_scores=prediction_scores,
            detection_boxes=prediction_bboxes,
            detection_heights_in_pixels=np.ones(
                shape=prediction_bboxes.shape[0:2]) * 100))

    return py_utils.NestedMap(
        metrics=metrics,
        expected_objects_at_distance=expected_objects_at_distance,
        expected_objects_at_points=expected_objects_at_points,
        expected_objects_at_rotation=expected_objects_at_rotation)

  def testLoadBoundingBoxes(self):
    # Test if all of the groundtruth data loads correctly for each label
    # when no distance is specified.
    metadata = kitti_metadata.KITTIMetadata()
    num_classes = len(metadata.ClassNames())
    test_data = self._GenerateMetricsWithTestData(num_classes)

    expected_num_objects = np.sum(
        test_data.expected_objects_at_distance, axis=1)

    # Note that we always skip 'Background' class 0.
    for label in range(1, num_classes):
      data = test_data.metrics._LoadBoundingBoxes(
          'groundtruth', label, distance=None)

      if expected_num_objects[label] == 0:
        self.assertIsNone(data)
      else:
        self.assertEqual(expected_num_objects[label], len(data.boxes))
        self.assertEqual(expected_num_objects[label], len(data.imgids))
        self.assertEqual(expected_num_objects[label], len(data.scores))
        self.assertEqual(expected_num_objects[label], len(data.difficulties))

        self.assertAllEqual(
            np.ones(shape=[expected_num_objects[label]]), data.scores)
        self.assertAllEqual(
            np.zeros(shape=[expected_num_objects[label]]), data.imgids)

  def testLoadBoundingBoxesDifficulty(self):
    metadata = kitti_metadata.KITTIMetadata()
    num_classes = len(metadata.ClassNames())
    test_data = self._GenerateMetricsWithTestData(num_classes)

    expected_num_objects = np.sum(
        test_data.expected_objects_at_distance, axis=1)

    difficulty_metric = test_data.metrics._breakdown_metrics['difficulty']

    # Test if difficulties are properly accumulated.
    for d in metadata.DifficultyLevels().values():
      if d == 1:
        self.assertAllEqual(expected_num_objects,
                            difficulty_metric._histogram[d, :])
      else:
        self.assertAllEqual(
            np.zeros_like(expected_num_objects),
            difficulty_metric._histogram[d, :])

  def testLoadBoundingBoxesDistance(self):
    # Test if all of the groundtruth data loads correctly for each label
    # when distance is specified.
    metadata = kitti_metadata.KITTIMetadata()
    num_classes = len(metadata.ClassNames())
    test_data = self._GenerateMetricsWithTestData(num_classes)
    num_bins_of_distance = int(
        np.rint(metadata.MaximumDistance() / metadata.DistanceBinWidth()))

    distance_metric = test_data.metrics._breakdown_metrics['distance']

    # Test if all of the groundtruth data loads correctly for each label
    # when no distance is specified.
    self.assertAllEqual(test_data.expected_objects_at_distance,
                        np.transpose(distance_metric._histogram))

    # Note that we always skip 'Background' class 0.
    for label in range(1, num_classes):
      for distance in range(num_bins_of_distance):
        data = test_data.metrics._LoadBoundingBoxes(
            'groundtruth', label, distance=distance)

        if test_data.expected_objects_at_distance[label, distance] == 0:
          self.assertIsNone(data)
        else:
          self.assertEqual(
              test_data.expected_objects_at_distance[label, distance],
              len(data.boxes))
          self.assertEqual(
              test_data.expected_objects_at_distance[label, distance],
              len(data.imgids))
          self.assertEqual(
              test_data.expected_objects_at_distance[label, distance],
              len(data.scores))
          self.assertEqual(
              test_data.expected_objects_at_distance[label, distance],
              len(data.difficulties))

          self.assertAllEqual(
              np.ones(shape=[
                  test_data.expected_objects_at_distance[label, distance]
              ]), data.scores)
          self.assertAllEqual(
              np.zeros(shape=[
                  test_data.expected_objects_at_distance[label, distance]
              ]), data.imgids)

  def testLoadBoundingBoxesNumPoints(self):
    # Test if all of the groundtruth data loads correctly for each label
    # when number of points is specified.
    metadata = kitti_metadata.KITTIMetadata()
    num_classes = len(metadata.ClassNames())
    test_data = self._GenerateMetricsWithTestData(num_classes)
    num_bins_of_points = metadata.NumberOfPointsBins()

    num_points_metric = test_data.metrics._breakdown_metrics['num_points']

    self.assertAllEqual(test_data.expected_objects_at_points,
                        np.transpose(num_points_metric._histogram))

    # Note that we always skip 'Background' class 0.
    for label in range(1, num_classes):
      for num_points in range(num_bins_of_points):
        data = test_data.metrics._LoadBoundingBoxes(
            'groundtruth', label, num_points=num_points)

        if test_data.expected_objects_at_points[label, num_points] == 0:
          self.assertIsNone(data)
        else:
          # Skip the first bin because it is a special case.
          if num_points == 0:
            continue
          self.assertEqual(
              test_data.expected_objects_at_points[label, num_points],
              len(data.boxes))
          self.assertEqual(
              test_data.expected_objects_at_points[label, num_points],
              len(data.imgids))
          self.assertEqual(
              test_data.expected_objects_at_points[label, num_points],
              len(data.scores))
          self.assertEqual(
              test_data.expected_objects_at_points[label, num_points],
              len(data.difficulties))

          self.assertAllEqual(
              np.ones(shape=[
                  test_data.expected_objects_at_points[label, num_points]
              ]), data.scores)
          self.assertAllEqual(
              np.zeros(shape=[
                  test_data.expected_objects_at_points[label, num_points]
              ]), data.imgids)

  def testLoadBoundingBoxesRotation(self):
    # Test if all of the groundtruth data loads correctly for each label
    # when rotation is specified.
    metadata = kitti_metadata.KITTIMetadata()
    num_classes = len(metadata.ClassNames())
    test_data = self._GenerateMetricsWithTestData(num_classes)
    num_bins_of_rotation = metadata.NumberOfRotationBins()

    rotation_metric = test_data.metrics._breakdown_metrics['rotation']

    # Test if all of the groundtruth data loads correctly for each label
    # when no distance is specified.
    self.assertAllEqual(test_data.expected_objects_at_rotation,
                        np.transpose(rotation_metric._histogram))

    # Note that we always skip 'Background' class 0.
    for label in range(1, num_classes):
      for rotation in range(num_bins_of_rotation):
        data = test_data.metrics._LoadBoundingBoxes(
            'groundtruth', label, rotation=rotation)

        if test_data.expected_objects_at_rotation[label, rotation] == 0:
          self.assertIsNone(data)
        else:
          self.assertEqual(
              test_data.expected_objects_at_rotation[label, rotation],
              len(data.boxes))
          self.assertEqual(
              test_data.expected_objects_at_rotation[label, rotation],
              len(data.imgids))
          self.assertEqual(
              test_data.expected_objects_at_rotation[label, rotation],
              len(data.scores))
          self.assertEqual(
              test_data.expected_objects_at_rotation[label, rotation],
              len(data.difficulties))

          self.assertAllEqual(
              np.ones(shape=[
                  test_data.expected_objects_at_rotation[label, rotation]
              ]), data.scores)
          self.assertAllEqual(
              np.zeros(shape=[
                  test_data.expected_objects_at_rotation[label, rotation]
              ]), data.imgids)

  def testAccumulateHistogram(self):
    metadata = kitti_metadata.KITTIMetadata()
    num_per_class = np.arange(metadata.NumClasses()) + 1
    statistics = [
        1 * np.ones(shape=(np.sum(num_per_class)), dtype=np.int32),
        2 * np.ones(shape=(np.sum(2 * num_per_class)), dtype=np.int32)
    ]
    statistics = np.concatenate(statistics)

    labels = []
    for i, n in enumerate(num_per_class):
      labels.extend([i] * n)
    for i, n in enumerate(num_per_class):
      labels.extend([i] * 2 * n)
    labels = np.array(labels)
    assert len(statistics) == len(labels)

    metrics_params = breakdown_metric.BreakdownMetric.Params().Set(
        metadata=metadata)
    test_breakdown_metric = breakdown_metric.ByDifficulty(metrics_params)
    test_breakdown_metric._AccumulateHistogram(
        statistics=statistics, labels=labels)

    for class_index, n in enumerate(num_per_class):
      self.assertEqual(n, test_breakdown_metric._histogram[1, class_index])
      self.assertEqual(2 * n, test_breakdown_metric._histogram[2, class_index])

  def testByName(self):
    metric_class = breakdown_metric.ByName('difficulty')
    self.assertEqual(metric_class, breakdown_metric.ByDifficulty)
    with self.assertRaises(ValueError):
      breakdown_metric.ByName('undefined')

  def testFindMaximumRecall(self):
    # The shape of the precision_recall_curves is [n, m, 2] where n is the
    # number of classes, m is then number of values in the curve, 2 indexes
    # between precision [0] and recall [1].
    car = np.transpose(
        np.array(
            [[0.9, 0.7, 0.5, 0.1, 0.0, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]],
            dtype=np.float32))
    ped = np.transpose(
        np.array(
            [[0.9, 0.7, 0.5, 0.0, 0.0, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]],
            dtype=np.float32))
    cyc = np.transpose(
        np.array(
            [[0.9, 0.7, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]],
            dtype=np.float32))
    foo = np.transpose(
        np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]],
            dtype=np.float32))
    precision_recall_curves = np.stack([car, ped, cyc, foo])
    max_recall = breakdown_metric._FindMaximumRecall(precision_recall_curves)
    self.assertAllEqual([4], max_recall.shape)
    self.assertNear(0.9, max_recall[0], 1e-7)
    self.assertNear(0.5, max_recall[1], 1e-7)
    self.assertNear(0.2, max_recall[2], 1e-7)
    self.assertNear(0.0, max_recall[3], 1e-7)

  def testFindRecallAtGivenPrecision(self):
    # The shape of the precision_recall_curves is [n, m, 2] where n is the
    # number of classes, m is then number of values in the curve, 2 indexes
    # between precision [0] and recall [1].
    car = np.transpose(
        np.array(
            [[0.9, 0.7, 0.5, 0.1, 0.0, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]],
            dtype=np.float32))
    ped = np.transpose(
        np.array(
            [[0.9, 0.7, 0.5, 0.0, 0.0, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]],
            dtype=np.float32))
    cyc = np.transpose(
        np.array(
            [[0.9, 0.7, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]],
            dtype=np.float32))
    foo = np.transpose(
        np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]],
            dtype=np.float32))
    precision_recall_curves = np.stack([car, ped, cyc, foo])
    precision_level = 0.5
    recall = breakdown_metric._FindRecallAtGivenPrecision(
        precision_recall_curves, precision_level)
    self.assertAllEqual([4], recall.shape)
    self.assertNear(0.5, recall[0], 1e-7)
    self.assertNear(0.5, recall[1], 1e-7)
    self.assertNear(0.2, recall[2], 1e-7)
    self.assertNear(0.0, recall[3], 1e-7)


if __name__ == '__main__':
  tf.test.main()
