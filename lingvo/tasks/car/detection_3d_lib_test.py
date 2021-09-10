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
"""Tests for detection_3d_lib."""

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import detection_3d_lib
import numpy as np


class Utils3DTest(test_utils.TestCase):

  def testScaledHuberLoss(self):
    utils_3d = detection_3d_lib.Utils3D()
    labels = tf.constant([1, 2, 3], dtype=tf.float32)
    # Predictions are less than delta, exactly at delta, and more than delta,
    # respectively.
    predictions = tf.constant([1.4, 1.2, 4.0], dtype=tf.float32)
    delta = 0.8
    expected_loss = [
        1. / delta * 0.5 * (0.4)**2,
        0.5 * delta,
        1.0 - 0.5 * delta,
    ]
    loss = utils_3d.ScaledHuberLoss(labels, predictions, delta=delta)
    with self.session():
      actual_loss = self.evaluate(loss)
      self.assertAllClose(actual_loss, expected_loss)

  def testCornerLoss(self):
    utils_3d = detection_3d_lib.Utils3D()
    gt_bboxes = tf.constant([[[[0., 0., 0., 1., 1., 1., 0.],
                               [0., 0., 0., 1., 1., 1., 0.],
                               [0., 0., 0., 1., 1., 1., 0.],
                               [0., 0., 0., 1., 1., 1., 0.],
                               [0., 0., 0., 1., 1., 1., 0.]]]])
    predicted_bboxes = tf.constant([[[
        [0., 0., 0., 1., 1., 1., 0.],  # Same as GT
        [0., 0., 0., 1., 1., 1., np.pi],  # Opposite heading
        [0., 0., 0., 1., 1., 1., np.pi / 2.],  # 90-deg rotation
        [1., 1., 1., 1., 1., 1., 0],  # Different center
        [0., 0., 0., 2., 2., 2., 0],  # Different size
    ]]])
    loss = utils_3d.CornerLoss(gt_bboxes, predicted_bboxes)
    with self.session():
      actual_loss = self.evaluate(loss)
      self.assertEqual(actual_loss.shape, (1, 1, 5))

  def testCornerLossAsym(self):
    utils_3d = detection_3d_lib.Utils3D()
    gt_bboxes = tf.constant([[[[0., 0., 0., 1., 1., 1., 0.],
                               [0., 0., 0., 1., 1., 1., 0.]]]])
    predicted_bboxes = tf.constant([[[
        [0., 0., 0., 1., 1., 1., 0.],  # Same as GT
        [0., 0., 0., 1., 1., 1., np.pi],  # Opposite heading
    ]]])
    expected_loss = [[[
        0.,
        8,
    ]]]
    loss = utils_3d.CornerLoss(gt_bboxes, predicted_bboxes, symmetric=False)
    with self.session():
      actual_loss = self.evaluate(loss)
      print(actual_loss)
      self.assertAllClose(actual_loss, expected_loss)

  def testCreateDenseCoordinates(self):
    utils_3d = detection_3d_lib.Utils3D()
    one_dim = utils_3d.CreateDenseCoordinates([(0.5, 1.5, 3)])
    with self.session():
      actual_one_dim = self.evaluate(one_dim)
      self.assertAllEqual(actual_one_dim, [[0.5], [1.0], [1.5]])

    two_by_two = utils_3d.CreateDenseCoordinates([(0, 1, 2), (1, 2, 2)])
    with self.session():
      actual_two_by_two = self.evaluate(two_by_two)
      self.assertAllEqual(actual_two_by_two, [[0, 1], [0, 2], [1, 1], [1, 2]])

    three_dims = utils_3d.CreateDenseCoordinates([(0, 1, 5), (1, 2, 5),
                                                  (0, 10, 5)])
    self.assertAllEqual(three_dims.shape, [5 * 5 * 5, 3])

  def testCreateDenseCoordinatesCenterInCell(self):
    utils_3d = detection_3d_lib.Utils3D()
    one_dim = utils_3d.CreateDenseCoordinates([(0., 3., 3)],
                                              center_in_cell=True)
    with self.session():
      actual_one_dim = self.evaluate(one_dim)
      self.assertAllEqual(actual_one_dim, [[0.5], [1.5], [2.5]])

    two_by_two = utils_3d.CreateDenseCoordinates([(0, 1, 2), (1, 2, 2)],
                                                 center_in_cell=True)
    with self.session():
      actual_two_by_two = self.evaluate(two_by_two)
      self.assertAllEqual(
          actual_two_by_two,
          [[0.25, 1.25], [0.25, 1.75], [0.75, 1.25], [0.75, 1.75]])

  def testMakeAnchorBoxesWithoutRotation(self):
    utils_3d = detection_3d_lib.Utils3D()
    anchor_bboxes = utils_3d.MakeAnchorBoxes(
        anchor_centers=tf.constant([[0, 0, 0], [1, 1, 1]], dtype=tf.float32),
        anchor_box_dimensions=tf.constant([[1, 2, 3], [3, 4, 5]],
                                          dtype=tf.float32),
        anchor_box_offsets=tf.constant([[0, 0, 0], [1, 1, 1]],
                                       dtype=tf.float32),
        anchor_box_rotations=None)
    with self.session():
      actual_anchor_bboxes = self.evaluate(anchor_bboxes)
      self.assertAllEqual(actual_anchor_bboxes,
                          [[[0, 0, 0, 1, 2, 3, 0], [1, 1, 1, 3, 4, 5, 0]],
                           [[1, 1, 1, 1, 2, 3, 0], [2, 2, 2, 3, 4, 5, 0]]])

  def testMakeAnchorBoxesWithRotation(self):
    utils_3d = detection_3d_lib.Utils3D()
    anchor_bboxes = utils_3d.MakeAnchorBoxes(
        anchor_centers=tf.constant([[0, 0, 0], [1, 1, 1]], dtype=tf.float32),
        anchor_box_dimensions=tf.constant([[1, 2, 3], [3, 4, 5]],
                                          dtype=tf.float32),
        anchor_box_offsets=tf.constant([[0, 0, 0], [1, 1, 1]],
                                       dtype=tf.float32),
        anchor_box_rotations=tf.constant([0, 0.5]))
    with self.session():
      actual_anchor_bboxes = self.evaluate(anchor_bboxes)
      self.assertAllEqual(actual_anchor_bboxes,
                          [[[0, 0, 0, 1, 2, 3, 0], [1, 1, 1, 3, 4, 5, 0.5]],
                           [[1, 1, 1, 1, 2, 3, 0], [2, 2, 2, 3, 4, 5, 0.5]]])

  def testAssignAnchors(self):
    utils_3d = detection_3d_lib.Utils3D()
    anchor_bboxes = tf.constant(
        [
            [0, 1, 1, 2, 2, 2, 0],  # Ignored
            [-1, 1, 1, 2, 2, 2, 0],  # Background
            [0.9, 1, 1, 2, 2, 2, 0],  # Foreground
            [5, 5, 5, 1, 1, 2, 0],  # Force matched to foreground
        ],
        dtype=tf.float32)

    # Second gt box should be forced match, third one should be ignored.
    gt_bboxes = tf.constant([[1, 1, 1, 2, 2, 2, 0], [5, 5, 5, 2, 2, 2, 0],
                             [10, 10, 10, 2, 2, 2, 0]],
                            dtype=tf.float32)
    gt_bboxes_labels = tf.constant([1, 2, 3])
    gt_bboxes_mask = tf.constant([1, 1, 1])

    assigned_anchors = utils_3d.AssignAnchors(
        anchor_bboxes,
        gt_bboxes,
        gt_bboxes_labels,
        gt_bboxes_mask,
        foreground_assignment_threshold=0.5,
        background_assignment_threshold=0.25)
    with self.session():
      actual_assigned_anchors, gt_bboxes = self.evaluate(
          (assigned_anchors, gt_bboxes))

      self.assertAllEqual(actual_assigned_anchors.assigned_gt_idx,
                          [-1, -1, 0, 1])
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_labels,
                          [0, 0, 1, 2])
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_bbox, [
          [0, 0, 0, 1, 1, 1, 0],
          [0, 0, 0, 1, 1, 1, 0],
          [1, 1, 1, 2, 2, 2, 0],
          [5, 5, 5, 2, 2, 2, 0],
      ])

      self.assertAllEqual(actual_assigned_anchors.assigned_cls_mask,
                          [0, 1, 1, 1])
      self.assertAllEqual(actual_assigned_anchors.assigned_reg_mask,
                          [0, 0, 1, 1])

      self.assertAllEqual(
          actual_assigned_anchors.assigned_gt_similarity_score.shape, [4])

  def testAssignAnchorsWithoutForceMatch(self):
    utils_3d = detection_3d_lib.Utils3D()
    anchor_bboxes = tf.constant(
        [
            [0, 1, 1, 2, 2, 2, 0],  # Ignored
            [-1, 1, 1, 2, 2, 2, 0],  # Background
            [0.9, 1, 1, 2, 2, 2, 0],  # Foreground
            [5, 5, 5, 1, 1, 2, 0],  # Background, since no force match
        ],
        dtype=tf.float32)

    # Second gt box should be forced match, third one should be ignored.
    gt_bboxes = tf.constant([[1, 1, 1, 2, 2, 2, 0], [5, 5, 5, 2, 2, 2, 0],
                             [10, 10, 10, 2, 2, 2, 0]],
                            dtype=tf.float32)
    gt_bboxes_labels = tf.constant([1, 2, 3])
    gt_bboxes_mask = tf.constant([1, 1, 1])

    assigned_anchors = utils_3d.AssignAnchors(
        anchor_bboxes,
        gt_bboxes,
        gt_bboxes_labels,
        gt_bboxes_mask,
        foreground_assignment_threshold=0.5,
        background_assignment_threshold=0.25,
        force_match=False)
    with self.session():
      actual_assigned_anchors, gt_bboxes = self.evaluate(
          (assigned_anchors, gt_bboxes))

      self.assertAllEqual(actual_assigned_anchors.assigned_gt_idx,
                          [-1, -1, 0, -1])
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_labels,
                          [0, 0, 1, 0])
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_bbox, [
          [0, 0, 0, 1, 1, 1, 0],
          [0, 0, 0, 1, 1, 1, 0],
          [1, 1, 1, 2, 2, 2, 0],
          [0, 0, 0, 1, 1, 1, 0],
      ])

      self.assertAllEqual(actual_assigned_anchors.assigned_cls_mask,
                          [0, 1, 1, 1])
      self.assertAllEqual(actual_assigned_anchors.assigned_reg_mask,
                          [0, 0, 1, 0])

      self.assertAllEqual(
          actual_assigned_anchors.assigned_gt_similarity_score.shape, [4])

  def testAssignAnchorsWithPadding(self):
    utils_3d = detection_3d_lib.Utils3D()
    anchor_bboxes = tf.constant([[0, 0, 0, 1, 2, 3, 0], [1, 1, 1, 3, 4, 5, 0.5],
                                 [1, 1, 1, 1, 2, 3, 0], [2, 2, 2, 3, 4, 5,
                                                         0.5]])
    gt_bboxes = anchor_bboxes + 0.05
    gt_bboxes_labels = tf.constant([1, 2, 3, 4])
    gt_bboxes_mask = tf.constant([1, 1, 0, 0])

    assigned_anchors = utils_3d.AssignAnchors(anchor_bboxes, gt_bboxes,
                                              gt_bboxes_labels, gt_bboxes_mask)
    with self.session():
      actual_assigned_anchors, gt_bboxes = self.evaluate(
          (assigned_anchors, gt_bboxes))

      # Last two boxes are padded, thus not assigned.
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_idx,
                          [0, 1, -1, -1])
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_labels,
                          [1, 2, 0, 0])
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_bbox[0:2, :],
                          gt_bboxes[0:2, :])

      # 2nd and 3rd should match dummy bbox.
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_bbox[2, :],
                          [0, 0, 0, 1, 1, 1, 0])
      self.assertAllEqual(actual_assigned_anchors.assigned_gt_bbox[3, :],
                          [0, 0, 0, 1, 1, 1, 0])

      # First two are foreground, last two are background.
      self.assertAllEqual(actual_assigned_anchors.assigned_cls_mask,
                          [1, 1, 1, 1])
      self.assertAllEqual(actual_assigned_anchors.assigned_reg_mask,
                          [1, 1, 0, 0])

      self.assertAllEqual(
          actual_assigned_anchors.assigned_gt_similarity_score.shape, [4])

  def testLocalizationResiduals(self):
    utils_3d = detection_3d_lib.Utils3D()

    anchor_bboxes = tf.constant([[1, 2, 3, 4, 3, 6, 0]], dtype=tf.float32)
    gt_bboxes = tf.constant([[2, 22, 303, 4, 9, 12, 0.5]], dtype=tf.float32)

    # diagonal_xy = 5 [since sqrt(3^2 + 4^2) = 5]
    expected_residuals = np.asarray([[
        1. / 5,
        20. / 5,
        300. / 6,
        0.,
        np.log(9. / 3.),
        np.log(12. / 6.),
        0.5,
    ]])
    residuals = utils_3d.LocalizationResiduals(anchor_bboxes, gt_bboxes)

    with self.session():
      actual_residuals = self.evaluate(residuals)
      self.assertAllClose(actual_residuals, expected_residuals)

  def testResidualsToBBoxes(self):
    utils_3d = detection_3d_lib.Utils3D()

    anchor_bboxes = tf.constant([[1, 2, 3, 4, 3, 6, 0]], dtype=tf.float32)
    expected_predicted_bboxes = np.asarray([[2, 22, 303, 4, 9, 12, 0.5]])

    residuals = tf.constant([[
        1. / 5, 20. / 5, 300. / 6, 0.,
        np.log(9. / 3.),
        np.log(12. / 6.),
        0.5,
    ]], dtype=tf.float32)  # pyformat: disable
    predicted_bboxes = utils_3d.ResidualsToBBoxes(anchor_bboxes, residuals)

    with self.session():
      actual_predicted_bboxes = self.evaluate(predicted_bboxes)
      self.assertAllClose(actual_predicted_bboxes, expected_predicted_bboxes)

  def testResidualsToBBoxesNegPiToPi(self):
    utils_3d = detection_3d_lib.Utils3D()

    anchor_bboxes = tf.constant(
        [[1, 2, 3, 4, 3, 6, 0.2], [1, 2, 3, 4, 3, 6, -0.2]], dtype=tf.float32)
    expected_predicted_bboxes = np.asarray(
        [[2, 22, 303, 4, 9, 12, -np.pi + 0.2],
         [2, 22, 303, 4, 9, 12, np.pi - 0.2]])

    residuals = tf.constant([
        [1. / 5, 20. / 5, 300. / 6, 0.,
         np.log(9. / 3.), np.log(12. / 6.), np.pi],
        [1. / 5, 20. / 5, 300. / 6, 0.,
         np.log(9. / 3.), np.log(12. / 6.), -np.pi]
    ], dtype=tf.float32)  # pyformat: disable
    predicted_bboxes = utils_3d.ResidualsToBBoxes(
        anchor_bboxes, residuals, min_angle_rad=-np.pi, max_angle_rad=np.pi)

    with self.session():
      actual_predicted_bboxes = self.evaluate(predicted_bboxes)
      self.assertAllClose(actual_predicted_bboxes, expected_predicted_bboxes)

  def testZeroResiduals(self):
    utils_3d = detection_3d_lib.Utils3D()

    anchor_bboxes = tf.constant([[1, 2, 3, 4, 3, 6, 0]], dtype=tf.float32)
    expected_predicted_bboxes = np.asarray([[1, 2, 3, 4, 3, 6, 0]])

    residuals = tf.zeros((1, 7))
    predicted_bboxes = utils_3d.ResidualsToBBoxes(anchor_bboxes, residuals)

    with self.session():
      actual_predicted_bboxes = self.evaluate(predicted_bboxes)
      self.assertAllClose(actual_predicted_bboxes, expected_predicted_bboxes)

  def testResidualsToBBoxPhiFloorMod(self):
    utils_3d = detection_3d_lib.Utils3D()

    anchor_bboxes = tf.constant([[1, 2, 3, 4, 3, 6, np.pi]], dtype=tf.float32)

    # We expected the returned phi value to be floormod w.r.t. pi.
    expected_predicted_bboxes = np.asarray([[1, 2, 3, 4, 3, 6, 1.]])

    residuals = tf.constant([[0, 0, 0, 0, 0, 0, 1.0]], dtype=tf.float32)
    predicted_bboxes = utils_3d.ResidualsToBBoxes(
        anchor_bboxes, residuals, min_angle_rad=0.0)

    with self.session():
      actual_predicted_bboxes = self.evaluate(predicted_bboxes)
      self.assertAllClose(actual_predicted_bboxes, expected_predicted_bboxes)

  def testNMSIndices(self):
    utils_3d = detection_3d_lib.Utils3D()

    # Create three anchor boxes, two largely overlapping and one
    # not overlapping with either.
    #
    # Set a batch size of 1 and use the Batched version to test
    # both functions.
    anchor_bboxes = tf.constant(
        [[[1, 2, 3, 4, 3, 6, 0.], [1, 2, 2, 4, 3, 6, 0.],
          [10, 20, 30, 4, 3, 6, 0.]]],
        dtype=tf.float32)

    # Treat them all as high scores.
    scores = tf.constant([[0.7, 0.8, 0.6]])

    with self.session():
      nms_indices, valid_mask = utils_3d.BatchedNMSIndices(
          anchor_bboxes, scores)
      indices, mask = self.evaluate([nms_indices, valid_mask])
      # One box is filtered out.
      self.assertEqual(2, np.sum(mask))
      # The two boxes that remain are the second one (because of its higher
      # score) and the last one (which overlaps with nothing).
      self.assertAllEqual([[1, 2, 0]], indices)

      # Flip the scores; expect the first box to be chosen instead.
      # Change the last box's threshold to be 0.0, so that the
      # default setting for the score threshold filters it out too.
      scores_2 = tf.constant([[0.8, 0.7, 0.0]])
      nms_indices, valid_mask = utils_3d.BatchedNMSIndices(
          anchor_bboxes, scores_2)
      indices, mask = self.evaluate([nms_indices, valid_mask])
      self.assertEqual(1, np.sum(mask))
      self.assertAllEqual([[0, 0, 0]], indices)

  def testOrientedNMSIndices(self):
    utils_3d = detection_3d_lib.Utils3D()

    # Assignments and IoU scores calculated offline.
    bboxes_data = tf.constant(
        [[
            [10.35, 8.429, -1.003, 3.7, 1.64, 1.49, 1.582],
            [10.35, 8.429, -1.003, 3.7, 1.64, 1.49, 0.0],  # box 0 rotated
            [11.5, 8.429, -1.003, 3.7, 1.64, 1.49, 1.0],  # Rotated to overlap
            [13.01, 8.149, -0.953, 4.02, 1.55, 1.52, 1.592],
            [13.51, 8.39, -1.0, 4.02, 1.55, 1.52, 1.592],  # Slight translation
            [13.51, 8.39, -1.0, 1.0, 1.0, 1.52, 1.592],  # Smaller box
            [13.51, 8.39, -1.0, 1.0, 1.0, 1.52, 1.9],  # Smaller box
        ]],
        dtype=tf.float32)

    # Notes on the data:
    # Lets say we have 3 classes and a thresh of 0.1
    # Keep box [0, 3] for class 0
    # Keep box [6] only for class 1
    # Keep box [2] for class 2
    scores_data = tf.constant([[
        [0.9, 0.1, 0.0],
        [0.89, 0.1, 0.01],
        [0.5, 0.01, 0.49],
        [0.8, 0.1, 0.1],
        [0.79, 0.11, 0.2],
        [0.2, 0.8, 0.1],
        [0.1, 0.9, 0.0],
    ]],
                              dtype=tf.float32)

    with self.session():
      outputs = utils_3d.BatchedOrientedNMSIndices(
          bboxes_data,
          scores_data,
          nms_iou_threshold=0.1,
          score_threshold=0.3,
          max_boxes_per_class=5)
      indices, scores, valid_mask = self.evaluate(outputs)

      class_masks = [
          valid_mask[0, cls_idx, :].astype(np.bool) for cls_idx in range(3)
      ]
      # Check the correct number of valid results per class
      self.assertEqual(class_masks[0].sum(), 2)
      self.assertEqual(class_masks[1].sum(), 1)
      self.assertEqual(class_masks[2].sum(), 1)

      # Check the results for each class
      self.assertAllEqual(indices[0, 0, class_masks[0]], [0, 3])
      self.assertAllClose(scores[0, 0, class_masks[0]], [0.9, 0.8])

      self.assertAllEqual(indices[0, 1, class_masks[1]], [6])
      self.assertAllClose(scores[0, 1, class_masks[1]], [0.9])

      self.assertAllEqual(indices[0, 2, class_masks[2]], [2])
      self.assertAllClose(scores[0, 2, class_masks[2]], [0.49])

      # Use a list of score thresholds instead
      outputs = utils_3d.BatchedOrientedNMSIndices(
          bboxes_data,
          scores_data,
          nms_iou_threshold=[0.1, 0.1, 0.1],
          score_threshold=[0.899, 0.5, 0.3],
          max_boxes_per_class=5)
      indices, scores, valid_mask = self.evaluate(outputs)

      class_masks = [
          valid_mask[0, cls_idx, :].astype(np.bool) for cls_idx in range(3)
      ]
      # Check the correct number of valid results per class
      self.assertEqual(class_masks[0].sum(), 1)
      self.assertEqual(class_masks[1].sum(), 1)
      self.assertEqual(class_masks[2].sum(), 1)

      # Check the results for each class
      self.assertAllEqual(indices[0, 0, class_masks[0]], [0])
      self.assertAllClose(scores[0, 0, class_masks[0]], [0.9])

      self.assertAllEqual(indices[0, 1, class_masks[1]], [6])
      self.assertAllClose(scores[0, 1, class_masks[1]], [0.9])

      self.assertAllEqual(indices[0, 2, class_masks[2]], [2])
      self.assertAllClose(scores[0, 2, class_masks[2]], [0.49])

  def testRandomPadOrTrimToTrim(self):
    points = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                          [10., 11., 12.]])
    features = tf.constant([[100.], [200.], [300.], [400.]])

    points, features = detection_3d_lib.RandomPadOrTrimTo([points, features],
                                                          2,
                                                          seed=123)[0]
    with self.session():
      points_np, features_np = self.evaluate([points, features])
      # Slicing choose a random 2 points.
      self.assertAllClose([[1., 2., 3.], [10., 11., 12.]], points_np)
      self.assertAllClose([[100.], [400.]], features_np)

  def testRandomPadOrTrimToPad(self):
    points = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                          [10., 11., 12.]])
    features = tf.constant([[100.], [200.], [300.], [400.]])

    points, features = detection_3d_lib.RandomPadOrTrimTo([points, features],
                                                          10,
                                                          seed=123)[0]
    with self.session():
      points_np, features_np = self.evaluate([points, features])
      # Padding repeats a random set of points.
      self.assertAllClose([[1., 2., 3.], [1., 2., 3.], [10., 11., 12.],
                           [7., 8., 9.], [7., 8., 9.], [4., 5., 6.]],
                          points_np[4:])
      self.assertAllClose([[100.], [100.], [400.], [300.], [300.], [200.]],
                          features_np[4:])

  def testRandomPadOrTrimToEmpty(self):
    points = tf.constant([[1., 2., 3.]])
    features = tf.constant([[100.]])
    points, features = detection_3d_lib.RandomPadOrTrimTo(
        [points[0:0], features[0:0]], 10, seed=123)[0]
    with self.session():
      points_np, features_np = self.evaluate([points, features])
      self.assertAllClose(points_np, np.zeros(shape=(10, 3)))
      self.assertAllClose(features_np, np.zeros(shape=(10, 1)))

  def testCornersToImagePlane(self):
    utils_3d = detection_3d_lib.Utils3D()
    batch = 4
    num_boxes = 50

    corners = tf.random.uniform([batch, num_boxes, 8, 3])
    velo_to_image_plane = tf.random.uniform([batch, 3, 4])
    corners_to_image_plane = utils_3d.CornersToImagePlane(
        corners, velo_to_image_plane)
    self.assertEqual([batch, num_boxes, 8, 2], corners_to_image_plane.shape)


if __name__ == '__main__':
  tf.test.main()
