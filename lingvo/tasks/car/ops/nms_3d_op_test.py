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

import time
import unittest
from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import ops
import numpy as np


class Nms3dOpTest(test_utils.TestCase):

  def _GetData(self):
    # Assignments and IoU scores derived from externally calculated test cases
    # created with shapely.
    bboxes = tf.constant(
        [
            [10.35, 8.429, -1.003, 3.7, 1.64, 1.49, 1.582],
            [10.35, 8.429, -1.003, 3.7, 1.64, 1.49, 0.0
            ],  # Box 0 rotated ~90 deg
            [11.5, 8.429, -1.003, 3.7, 1.64, 1.49, 1.0],  # Rotated to overlap
            [13.01, 8.149, -0.953, 4.02, 1.55, 1.52, 1.592],
            [13.51, 8.39, -1.0, 4.02, 1.55, 1.52, 1.592],  # Slight translation
            [13.51, 8.39, -1.0, 1.0, 1.0, 1.52, 1.592],  # Smaller box
            [13.51, 8.39, -1.0, 1.0, 1.0, 1.52, 1.9],  # Smaller box
        ],
        dtype=tf.float32)

    scores = tf.constant([
        [0.9, 0.09, 0.0],
        [0.88, 0.109, 0.011],
        [0.5, 0.01, 0.49],
        [0.8, 0.1, 0.1],
        [0.79, 0.12, 0.19],
        [0.2, 0.79, 0.11],
        [0.1, 0.9, 0.0],
    ],
                         dtype=tf.float32)
    return bboxes, scores

  def _TestNMSOp(self, bboxes_3d, class_scores, nms_iou_threshold,
                 score_threshold, max_boxes_per_class, expected_indices):
    with self.session():
      bbox_indices, bbox_scores, valid_mask = ops.non_max_suppression_3d(
          bboxes_3d,
          class_scores,
          nms_iou_threshold=nms_iou_threshold,
          score_threshold=score_threshold,
          max_boxes_per_class=max_boxes_per_class)
      bbox_idx, scores, mask = self.evaluate(
          [bbox_indices, bbox_scores, valid_mask])

      num_classes = len(expected_indices)
      expected_shape = (num_classes, max_boxes_per_class)
      self.assertEqual(bbox_idx.shape, expected_shape)
      self.assertEqual(scores.shape, expected_shape)
      self.assertEqual(mask.shape, expected_shape)

      total_expected_valid_boxes = sum([len(exp) for exp in expected_indices])
      self.assertEqual(mask.sum(), total_expected_valid_boxes)

      for cls_idx in range(num_classes):
        cls_mask = mask[cls_idx, :].astype(np.bool)
        self.assertEqual(cls_mask.sum(), len(expected_indices[cls_idx]))
        self.assertAllEqual(bbox_idx[cls_idx, cls_mask],
                            expected_indices[cls_idx])

  def testMultiClassNMS(self):
    bboxes_3d, class_scores = self._GetData()
    expected_indices = [[0, 3], [6], [2]]
    self._TestNMSOp(
        bboxes_3d,
        class_scores,
        nms_iou_threshold=[0.1, 0.1, 0.1],
        score_threshold=[0.3, 0.3, 0.3],
        max_boxes_per_class=5,
        expected_indices=expected_indices)

  def testLowerScoreThreshold(self):
    bboxes_3d, class_scores = self._GetData()
    # Lower threshold means more boxes are included.
    expected_indices = [[0, 3], [6, 1], [2, 4]]
    self._TestNMSOp(
        bboxes_3d,
        class_scores,
        nms_iou_threshold=[0.1, 0.1, 0.1],
        score_threshold=[0.01, 0.01, 0.01],
        max_boxes_per_class=5,
        expected_indices=expected_indices)

  def testHighIoUThreshold(self):
    bboxes_3d, class_scores = self._GetData()
    expected_indices = [[0, 1, 3, 4, 2, 5, 6], [6, 5, 4, 1, 3, 0, 2],
                        [2, 4, 5, 3, 1]]
    # Increase IoU Threshold and max number of boxes so
    # all non-zero score boxes are returned.
    self._TestNMSOp(
        bboxes_3d,
        class_scores,
        nms_iou_threshold=[0.999, 0.999, 0.999],
        score_threshold=[0.01, 0.01, 0.01],
        max_boxes_per_class=10,
        expected_indices=expected_indices)

  def testOneClassVsMultiClass(self):
    # Check running on all 3 classes versus each independently.
    bboxes_3d, class_scores = self._GetData()
    num_classes = 3
    max_boxes_per_class = 5
    with self.session():
      bbox_indices, bbox_scores, valid_mask = ops.non_max_suppression_3d(
          bboxes_3d,
          class_scores,
          nms_iou_threshold=[0.1, 0.1, 0.1],
          score_threshold=[0.3, 0.3, 0.3],
          max_boxes_per_class=max_boxes_per_class)
      multiclass_indices, multiclass_scores, multiclass_valid_mask = self.evaluate(
          [bbox_indices, bbox_scores, valid_mask])
      self.assertEqual(multiclass_indices.shape,
                       (num_classes, max_boxes_per_class))
      self.assertEqual(multiclass_scores.shape,
                       (num_classes, max_boxes_per_class))
      self.assertEqual(multiclass_valid_mask.shape,
                       (num_classes, max_boxes_per_class))

      # For each class, get results for just that class and compare.
      for cls_idx in range(num_classes):
        bbox_idx, bbox_scores, valid_mask = ops.non_max_suppression_3d(
            bboxes_3d,
            class_scores[:, cls_idx:cls_idx + 1],
            nms_iou_threshold=[0.1],
            score_threshold=[0.3],
            max_boxes_per_class=max_boxes_per_class)
        per_class_indices, per_class_scores, per_class_valid_mask = self.evaluate(
            [bbox_idx, bbox_scores, valid_mask])

        self.assertEqual(per_class_indices.shape, (1, max_boxes_per_class))
        self.assertEqual(per_class_scores.shape, (1, max_boxes_per_class))
        self.assertEqual(per_class_valid_mask.shape, (1, max_boxes_per_class))

        per_class_mask = per_class_valid_mask[0, :].astype(np.bool)
        multiclass_mask = multiclass_valid_mask[cls_idx, :].astype(np.bool)
        self.assertAllEqual(per_class_indices[0, per_class_mask],
                            multiclass_indices[cls_idx, multiclass_mask])
        self.assertAllEqual(per_class_scores[0, per_class_mask],
                            multiclass_scores[cls_idx, multiclass_mask])

  @unittest.skip('Speed benchmark')
  def testSpeed(self):
    num_bboxes_list = [500, 1000, 10000]
    num_classes_list = [3, 10, 25]

    for num_bboxes in num_bboxes_list:
      for num_classes in num_classes_list:
        bboxes_3d = tf.random.uniform((num_bboxes, 7),
                                      minval=0.1,
                                      maxval=2,
                                      dtype=tf.float32)
        # Make half zero so we can see behavior with very low values that
        # will get filtered out quickly.
        class_scores = tf.concat([
            tf.random.uniform((num_bboxes // 2, num_classes),
                              minval=0,
                              maxval=1,
                              dtype=tf.float32),
            tf.zeros((num_bboxes // 2, num_classes), dtype=tf.float32)
        ],
                                 axis=0)

        with self.session():
          outputs = ops.non_max_suppression_3d(
              bboxes_3d,
              class_scores,
              max_boxes_per_class=1000,
              nms_iou_threshold=[0.1] * num_classes,
              score_threshold=[0.3] * num_classes)

          timings = []
          for _ in range(10):
            start = time.time()
            _ = self.evaluate(outputs)
            end = time.time()
            timings.append(end - start)
          avg = sum(timings) / len(timings)
          print('[{},{},{},{},{}]'.format(num_bboxes, num_classes, min(timings),
                                          avg, max(timings)))


if __name__ == '__main__':
  tf.test.main()
