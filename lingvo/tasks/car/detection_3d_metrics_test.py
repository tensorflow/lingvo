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
"""Tests for detection_3d_metrics."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import detection_3d_metrics
from lingvo.tasks.car import transform_util
import numpy as np


class Detection3dMetricsTest(test_utils.TestCase):

  def testTopDownVisualizationMetric(self):
    top_down_transform = transform_util.MakeCarToImageTransform(
        pixels_per_meter=32.,
        image_ref_x=512.,
        image_ref_y=1408.,
        flip_axes=True)
    metric = detection_3d_metrics.TopDownVisualizationMetric(top_down_transform)

    batch_size = 4
    num_preds = 10
    num_gt = 12
    num_points = 128

    visualization_labels = np.random.randint(0, 2, (batch_size, num_preds))
    predicted_bboxes = np.random.rand(batch_size, num_preds, 5)
    visualization_weights = np.abs(np.random.rand(batch_size, num_preds))

    labels = np.random.randint(0, 2, (batch_size, num_gt))
    gt_bboxes_2d = np.random.rand(batch_size, num_gt, 5)
    gt_bboxes_2d_weights = np.abs(np.random.rand(batch_size, num_gt))
    difficulties = np.random.randint(0, 3, (batch_size, num_gt))

    points_xyz = np.random.rand(batch_size, num_points, 3)
    points_padding = np.random.randint(0, 2, (batch_size, num_points))
    source_ids = np.full([batch_size], '012346')

    metric.Update(
        py_utils.NestedMap({
            'visualization_labels': visualization_labels,
            'predicted_bboxes': predicted_bboxes,
            'visualization_weights': visualization_weights,
            'labels': labels,
            'gt_bboxes_2d': gt_bboxes_2d,
            'gt_bboxes_2d_weights': gt_bboxes_2d_weights,
            'points_xyz': points_xyz,
            'points_padding': points_padding,
            'difficulties': difficulties,
            'source_ids': source_ids,
        }))

    _ = metric.Summary('test')

  def testCameraVisualization(self):
    metric = detection_3d_metrics.CameraVisualization()

    batch_size = 4
    num_preds = 10

    images = np.random.rand(batch_size, 512, 1024, 3)
    bbox_corners = np.random.rand(batch_size, num_preds, 8, 2)
    bbox_scores = np.random.rand(batch_size, num_preds)

    metric.Update(
        py_utils.NestedMap({
            'camera_images': images,
            'bbox_corners': bbox_corners,
            'bbox_scores': bbox_scores
        }))

    # Test that the metric runs.
    _ = metric.Summary('test')

  def testMesh(self):
    metric = detection_3d_metrics.WorldViewer()

    batch_size = 4
    num_points = 128
    points_xyz = np.random.rand(batch_size, num_points, 3) * 40.
    points_padding = np.random.randint(0, 2, (batch_size, num_points))

    metric.Update(
        py_utils.NestedMap({
            'points_xyz': points_xyz,
            'points_padding': points_padding,
        }))

    # Test that the metric runs.
    _ = metric.Summary('test')


if __name__ == '__main__':
  tf.test.main()
