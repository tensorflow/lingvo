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
"""Library of useful for functions for working with 3D object detection."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.car import geometry
from lingvo.tasks.car import ops
import numpy as np


class Utils3D:
  """Helper routines for 3D detection problems.

  One common method to do 3D anchor box assignment is to match anchors to
  ground-truth bboxes. First, we generate proposal anchors at given priors
  (center locations, dimension prior, offset prior) to tile the input space.
  After tiling the input space, each anchor can be assigned to a ground-truth
  bbox by measuring IOU and taking a threshold. Note that a ground-truth bbox
  may be assigned to multiple anchors - this is expected, and is managed at
  inference time by non-max suppression.

  Note: This implementation is designed to be used at input generation, and
  does not support a batch dimension.

  The following functions in this utility class helps with that:

    - CreateDenseCoordinates: Makes it easy to create a dense grid of
      coordinates that would usually correspond to center locations.

    - MakeAnchorBoxes: Given a list of center coordinates, dimension priors, and
      offset priors, this function creates actual anchor bbox parameters at each
      coordinate. More than one box can be at each center.

    - IOUAxisAlignedBoxes: This function computes the IOU between two lists of
      boxes.

    - AssignAnchors: This function assigns each anchor a ground-truth bbox. Note
      that one ground-truth bbox can be assigned to multiple anchors. The model
      is expected to regress the residuals each anchor and it's corresponding
      ground-truth bbox parameters.
  """

  def ScaledHuberLoss(self, labels, predictions, weights=1.0, delta=1.0):
    r"""Scaled Huber (SmoothL1) Loss.

    This function wraps tf.losses.huber_loss to rescale it by 1 / delta, and
    uses Reduction.NONE.

    This scaling results in the following formulation instead::

      (1/d) * 0.5 * x^2       if \|x\| <= d
      \|x\| - 0.5 * d         if \|x\| > d

    where x is labels - predictions.

    Hence, delta changes where the quadratic bowl is, but does not change the
    overall shape of the loss outside of delta.

    Args:
      labels: The ground truth output tensor, same dimensions as 'predictions'.
      predictions: The predicted outputs.
      weights: Optional Tensor whose rank is either 0, or the same rank as
        labels, and must be broadcastable to labels (i.e., all dimensions must
        be either 1, or the same as the corresponding losses dimension).
      delta: float, the point where the huber loss function changes from a
        quadratic to linear.

    Returns:
      Weighted loss float Tensor. This has the same shape as labels.
    """
    return (1. / delta) * tf.losses.huber_loss(
        labels,
        predictions,
        weights=weights,
        delta=delta,
        reduction=tf.losses.Reduction.NONE,
        loss_collection=None)

  def CornerLoss(self, gt_bboxes, predicted_bboxes, symmetric=True):
    """Corner regularization loss.

    This function computes the corner loss, an alternative regression loss
    for box residuals. This was used in the Frustum-PointNets paper [1].

    We compute the predicted bboxes (all 8 corners) and compute a SmoothedL1
    loss between the corners of the predicted boxes and ground truth. Hence,
    this loss can help encourage the model to maximize the IoU of the
    predictions.

    [1] Frustum PointNets for 3D Object Detection from RGB-D Data
        https://arxiv.org/pdf/1711.08488.pdf

    Args:
      gt_bboxes: tf.float32 of shape [..., 7] which contains (x, y, z, dx, dy,
        dz, phi), corresponding to ground truth bbox parameters.
      predicted_bboxes: tf.float32 of same shape as gt_bboxes containing
        predicted bbox parameters.
      symmetric: boolean.  If True, computes the minimum of the corner loss
        with respect to both the gt box and the gt box rotated 180 degrees.

    Returns:
      tf.float32 Tensor of shape [...] where each entry contains the corner loss
      for the corresponding bbox.
    """
    bbox_shape = py_utils.GetShape(gt_bboxes)
    batch_size = bbox_shape[0]

    gt_bboxes = tf.reshape(gt_bboxes, [batch_size, -1, 7])
    predicted_bboxes = tf.reshape(predicted_bboxes, [batch_size, -1, 7])

    gt_corners = geometry.BBoxCorners(gt_bboxes)
    predicted_corners = geometry.BBoxCorners(predicted_bboxes)
    huber_loss = self.ScaledHuberLoss(gt_corners, predicted_corners)
    huber_loss = tf.reduce_sum(huber_loss, axis=[-2, -1])

    if symmetric:
      # Compute the loss assuming the ground truth is flipped 180, and
      # take the minimum of the two losses.
      rot = tf.constant([[[0., 0., 0., 0., 0., 0., np.pi]]], dtype=tf.float32)
      rotated_gt_bboxes = gt_bboxes + rot
      rotated_gt_corners = geometry.BBoxCorners(rotated_gt_bboxes)
      rotated_huber_loss = self.ScaledHuberLoss(
          labels=rotated_gt_corners, predictions=predicted_corners)
      rotated_huber_loss = tf.reduce_sum(rotated_huber_loss, axis=[-2, -1])
      huber_loss = tf.minimum(huber_loss, rotated_huber_loss)

    huber_loss = tf.reshape(huber_loss, bbox_shape[:-1])
    return huber_loss

  def CreateDenseCoordinates(self, ranges, center_in_cell=False):
    """Create a matrix of coordinate locations corresponding to a dense grid.

    Example: To create (x, y) coordinates corresponding over a 10x10 grid with
    step sizes 1, call ``CreateDenseCoordinates([(1, 10, 10), (1, 10, 10)])``.

    Args:
      ranges: A list of 3-tuples, each tuple is expected to contain (min, max,
        num_steps). Each list element corresponds to one dimesion. Each tuple
        will be passed into np.linspace to create the values for a single
        dimension.
      center_in_cell: Whether to center the each location in the grid cell
        center; defaults to False.

    Returns:
      tf.float32 tensor of shape [total_points, len(ranges)], where
      total_points = product of all num_steps.

    """
    total_points = int(np.prod([r_steps for _, _, r_steps in ranges]))
    cycle_steps = total_points
    stack_coordinates = []

    for r_start, r_stop, r_steps in ranges:
      if center_in_cell:
        # Compute the size of each grid cell, and then start from the first cell
        # but in the center location.
        cell_size = float(r_stop - r_start) / r_steps
        half_size = cell_size / 2.
        r_start += half_size
        r_stop -= half_size
      values = tf.linspace(
          tf.cast(r_start, tf.float32), tf.cast(r_stop, tf.float32),
          tf.cast(r_steps, tf.int32))
      cycle_steps //= r_steps
      gather_idx = (tf.range(total_points) // cycle_steps) % r_steps
      stack_coordinates.append(tf.gather(values, gather_idx))

    return tf.stack(stack_coordinates, axis=1)

  def MakeAnchorBoxes(self,
                      anchor_centers,
                      anchor_box_dimensions,
                      anchor_box_offsets,
                      anchor_box_rotations=None):
    """Create anchor boxes from centers, dimensions, offsets.

    Args:
      anchor_centers: [A, dims] tensor. Center locations to generate boxes at.
      anchor_box_dimensions: [B, dims] tensor corresponding to dimensions of
        each box. The inner-most dimension of this tensor must match
        anchor_centers.
      anchor_box_offsets: [B, dims] tensor corresponding to offsets of each box.
      anchor_box_rotations: [B] tensor corresponding to rotation of each box. If
        None, rotation will be set to 0.

    Returns:
      A [num_anchors_center, num_boxes_per_center, 2 * dims + 1] tensor. Usually
      dims=3 for 3D, where ``[..., :dims]`` corresponds to location,
      ``[..., dims:2*dims]`` corresponds to dimensions, and ``[..., -1]``
      corresponds to rotation.
    """
    num_centers, dims = py_utils.GetShape(anchor_centers)
    num_box_per_center = py_utils.GetShape(anchor_box_dimensions, 1)[0]

    # Offset the centers by the box offsets
    anchor_centers = tf.reshape(anchor_centers, [num_centers, 1, dims])
    anchor_box_offsets = tf.reshape(anchor_box_offsets,
                                    [1, num_box_per_center, dims])
    anchor_box_centers = anchor_centers + anchor_box_offsets
    anchor_box_centers = py_utils.HasShape(
        anchor_box_centers, [num_centers, num_box_per_center, dims])

    # Concat the dimensions and rotation parameters
    anchor_box_dimensions = tf.tile(
        tf.expand_dims(anchor_box_dimensions, 0), [num_centers, 1, 1])
    if anchor_box_rotations is None:
      anchor_box_rotations = tf.zeros([num_centers, num_box_per_center, 1])
    else:
      anchor_box_rotations = tf.tile(
          tf.reshape(anchor_box_rotations, [1, num_box_per_center, 1]),
          [num_centers, 1, 1])

    anchor_bboxes = tf.concat(
        [anchor_box_centers, anchor_box_dimensions, anchor_box_rotations],
        axis=-1)

    return anchor_bboxes

  def IOU2DRotatedBoxes(self, bboxes_u, bboxes_v):
    """Computes IoU between every pair of bboxes with headings.

    This function ignores the z dimension, which is not usually considered
    during anchor assignment.

    Args:
      bboxes_u: tf.float32. [U, dims]. [..., :7] are (x, y, z, dx, dy, dz, r).
      bboxes_v: tf.float32. [V, dims]. [..., :7] are (x, y, z, dx, dy, dz, r).

    Returns:
      tf.float32 tensor with shape [U, V], where [i, j] is IoU between
        i-th bbox of bboxes_u and j-th bbox of bboxes_v.
    """

    def _IgnoreZCoordinate(bboxes):
      """Set z center to 0, and z dimension to 1."""
      num_bboxes = py_utils.GetShape(bboxes, 1)[0]
      return tf.stack([
          bboxes[:, 0], bboxes[:, 1], tf.zeros((num_bboxes,)),
          bboxes[:, 3], bboxes[:, 4], tf.ones((num_bboxes,)),
          bboxes[:, 6]
      ], axis=1)   # pyformat: disable

    bboxes_u = _IgnoreZCoordinate(bboxes_u)
    bboxes_v = _IgnoreZCoordinate(bboxes_v)
    return ops.pairwise_iou3d(bboxes_u, bboxes_v)

  def AssignAnchors(self,
                    anchor_bboxes,
                    gt_bboxes,
                    gt_bboxes_labels,
                    gt_bboxes_mask,
                    foreground_assignment_threshold=0.5,
                    background_assignment_threshold=0.35,
                    background_class_id=0,
                    force_match=True,
                    similarity_fn=None):
    """Assigns anchors to bboxes using a similarity function (SSD-based).

    Each anchor box is assigned to the top matching ground truth box.
    Ground truth boxes can be assigned to multiple anchor boxes.

    Assignments can result in 3 outcomes:

      - Positive assignment (if score >= foreground_assignment_threshold):
        assigned_gt_labels will reflect the assigned box label and
        assigned_cls_mask will be set to 1.0
      - Background assignment (if score <= background_assignment_threshold):
        assigned_gt_labels will be background_class_id and assigned_cls_mask
        will be set to 1.0
      - Ignore assignment (otherwise):
        assigned_gt_labels will be background_class_id and assigned_cls_mask
        will be set to 0.0

    The detection loss function would usually:

      - Use assigned_cls_mask for weighting the classification loss. The mask
        is set such that the loss applies to foreground and background
        assignments only - ignored anchors will be set to 0.
      - Use assigned_reg_mask for weighting the regression loss. The mask is set
        such that the loss applies to foreground assignments only.

    The thresholds (foreground_assignment_threshold and
    background_assignment_threshold) should be tuned per dataset.

    TODO(jngiam): Consider having a separate threshold for regression boxes; a
    separate threshold is used in PointRCNN.

    Args:
      anchor_bboxes: tf.float32. [A, 7], where [..., :] corresponds to box
        parameters (x, y, z, dx, dy, dz, r).
      gt_bboxes: tf.float32. [G, 7], where [..., :] corresponds to ground truth
        box parameters (x, y, z, dx, dy, dz, r).
      gt_bboxes_labels: tensor with shape [G]. Ground truth labels for each
        bounding box.
      gt_bboxes_mask: tensor with shape [G]. Mask for ground truth boxes, 1 iff
        the gt_bbox is a real bbox.
      foreground_assignment_threshold: Similarity score threshold for assigning
        foreground bounding boxes; scores need to be >=
        foreground_assignment_threshold to be assigned to foreground.
      background_assignment_threshold: Similarity score threshold for assigning
        background bounding boxes; scores need to be <=
        background_assignment_threshold to be assigned to background.
      background_class_id: class id to be assigned to anchors_gt_class if no
        anchor boxes match.
      force_match: Boolean specifying if force matching is enabled. If
        force matching is enabled, then matched anchors which are also the
        highest scoring with a ground-truth box are considered foreground
        matches as long as their similarity score > 0.
      similarity_fn: Function that computes the a similarity score (e.g., IOU)
        between pairs of bounding boxes. This function should take in two
        tensors corresponding to anchor and ground-truth bboxes, and return a
        matrix [A, G] with the similarity score between each pair of bboxes. The
        score must be non-negative, with greater scores representing more
        similar. The fore/background_assignment_thresholds will be applied to
        this score to determine if the an anchor is foreground, background or
        ignored. If set to None, the function will default to IOU2DRotatedBoxes.

    Returns:
      NestedMap with the following keys

      - assigned_gt_idx: shape [A] index corresponding to the index of the
        assigned ground truth box. Anchors not assigned to a ground truth box
        will have the index set to -1.
      - assigned_gt_bbox: shape [A, 7] bbox parameters assigned to each anchor.
      - assigned_gt_similarity_score: shape [A] (iou) score between the anchor
        and the gt bbox.
      - assigned_gt_labels: shape [A] label assigned to bbox.
      - assigned_cls_mask: shape [A] mask for classification loss per anchor.
        This should be 1.0 if the anchor has a foreground or background
        assignment; otherwise, it will be assigned to 0.0.
      - assigned_reg_mask: shape [A] mask for regression loss per anchor.
        This should be 1.0 if the anchor has a foreground assignment;
        otherwise, it will be assigned to 0.0.
        Note: background anchors do not have regression targets.
    """
    if similarity_fn is None:
      similarity_fn = self.IOU2DRotatedBoxes

    # Shape validation.
    anchor_bboxes = py_utils.HasShape(anchor_bboxes, [-1, 7])
    num_anchor_bboxes, _ = py_utils.GetShape(anchor_bboxes, 2)
    gt_bboxes = py_utils.HasShape(gt_bboxes, [-1, 7])
    num_gt_bboxes, _ = py_utils.GetShape(gt_bboxes, 2)

    # Compute similarity score and reduce max by anchors and by ground-truth.
    similarity_score = similarity_fn(anchor_bboxes, gt_bboxes)
    similarity_score = py_utils.HasShape(similarity_score,
                                         [num_anchor_bboxes, num_gt_bboxes])

    # Reduce over ground-truth boxes, so we have the max score per anchor.
    anchor_max_score = tf.reduce_max(similarity_score, axis=1)
    anchor_max_idx = tf.argmax(similarity_score, axis=1)

    if force_match:
      # Reduce over anchors, so we have the max score per ground truth box.
      gt_max_score = tf.reduce_max(similarity_score, axis=0, keepdims=True)

      # Force matches occur when the top matching gt bbox for an anchor is the
      # top matching anchor for the gt bbox. When force matching, we match
      # these boxes as long as their similarity score exceeds 0.
      force_matches = (
          tf.equal(similarity_score, gt_max_score)
          & tf.equal(similarity_score, anchor_max_score[..., tf.newaxis])
          & tf.greater(similarity_score, 0.)
          & tf.cast(gt_bboxes_mask[tf.newaxis, ...], tf.bool))
      force_match_indicator = tf.reduce_any(force_matches, axis=1)
      force_match_idx = tf.argmax(tf.cast(force_matches, tf.int32), axis=1)

      # In assigning foreground/background anchors later, force_match_indicator
      # is used to determine which anchors are force foreground, and the index
      # assigned will be taken from anchor_max_idx.

      # Force matchers must also be the max scoring gt bbox per anchor.
      # We overwrite anchor_max_idx to ensure that the right match is done.
      anchor_max_idx = tf.where(force_match_indicator, force_match_idx,
                                anchor_max_idx)

    # Ensure that max score boxes are not padded boxes by setting score to 0
    # for boxes that are padded.
    gathered_mask = tf.array_ops.batch_gather(gt_bboxes_mask, anchor_max_idx)
    anchor_max_score = tf.where(
        tf.equal(gathered_mask, 1), anchor_max_score,
        tf.zeros_like(anchor_max_score))

    # Boolean tensors corresponding to whether an anchor is background or
    # foreground based on thresholding.
    background_anchors = tf.less_equal(anchor_max_score,
                                       background_assignment_threshold)
    foreground_anchors = tf.greater_equal(anchor_max_score,
                                          foreground_assignment_threshold)
    if force_match:
      # Background anchors are below threshold and not force matches.
      background_anchors &= ~force_match_indicator
      # Foreground anchors are above thresholds or force matches.
      foreground_anchors |= force_match_indicator

    # Add dummy background bbox to gt_boxes to facilitate batch gather.
    dummy_bbox = tf.constant([[0, 0, 0, 1, 1, 1, 0]], dtype=tf.float32)

    # Since we are concatenating the dummy bbox, the index corresponds to the
    # number of boxes.
    dummy_bbox_idx = py_utils.GetShape(gt_bboxes, 1)[0]
    dummy_bbox_idx = tf.cast(dummy_bbox_idx, tf.int64)

    gt_bboxes = tf.concat([gt_bboxes, dummy_bbox], axis=0)
    gt_bboxes_labels = tf.concat([gt_bboxes_labels, [background_class_id]],
                                 axis=0)

    # Gather indices so that all foreground boxes are gathered from gt_bboxes,
    # while all background and ignore boxes gather the dummy_bbox.
    anchor_gather_idx = tf.where(foreground_anchors, anchor_max_idx,
                                 tf.ones_like(anchor_max_idx) * dummy_bbox_idx)

    # Gather the bboxes and weights.
    assigned_gt_bbox = tf.array_ops.batch_gather(gt_bboxes, anchor_gather_idx)
    assigned_gt_labels = tf.array_ops.batch_gather(gt_bboxes_labels,
                                                   anchor_gather_idx)

    # Set masks for classification and regression losses.
    assigned_cls_mask = tf.cast(background_anchors | foreground_anchors,
                                tf.float32)
    assigned_reg_mask = tf.cast(foreground_anchors, tf.float32)

    # Set assigned_gt_idx such that dummy boxes have idx = -1.
    assigned_gt_idx = tf.where(
        tf.equal(anchor_gather_idx, dummy_bbox_idx),
        tf.ones_like(anchor_gather_idx) * -1, anchor_gather_idx)
    assigned_gt_idx = tf.cast(assigned_gt_idx, tf.int32)

    return py_utils.NestedMap(
        assigned_gt_idx=assigned_gt_idx,
        assigned_gt_bbox=assigned_gt_bbox,
        assigned_gt_similarity_score=anchor_max_score,
        assigned_gt_labels=assigned_gt_labels,
        assigned_cls_mask=assigned_cls_mask,
        assigned_reg_mask=assigned_reg_mask)

  def LocalizationResiduals(self, anchor_bboxes, assigned_gt_bboxes):
    """Computes the anchor residuals for every bbox.

    For a given bbox, compute residuals in the following way:

      Let ``anchor_bbox = (x_a, y_a, z_a, dx_a, dy_a, dz_a, phi_a)``
      and ``assigned_gt_bbox = (x_gt, y_gt, z_gt, dx_gt, dy_gt, dz_gt, phi_gt)``

      Define ``diagonal_xy = sqrt(dx_a^2 + dy_a^2)``

      Then the corresponding residuals are given by::

        x_residual = (x_gt - x_a) / (diagonal_xy)
        y_residual = (y_gt - y_a) / (diagonal_xy)
        z_residual = (z_gt - z_a) / (dz_a)

        dx_residual = log(dx_gt / dx_a)
        dy_residual = log(dy_gt / dy_a)
        dz_residual = log(dz_gt / dz_a)

        phi_residual = phi_gt - phi_a

      The normalization for x and y residuals by the diagonal was first
      proposed by [1]. Intuitively, this reflects that objects can usually
      move freely in the x-y plane, including diagonally. On the other hand,
      moving in the z-axis (up and down) can be considered orthogonal to x-y.

      For phi_residual, one way to frame the loss is with
      SmoothL1(sine(phi_residual - phi_predicted)).
      The use of sine to wrap the phi residual was proposed by [2]. This
      stems from the observation that bboxes at phi and phi + pi are the same
      bbox, fully overlapping in 3D space, except that the direction is
      different. Note that the use of sine makes this residual invariant to
      direction when a symmetric loss like SmoothL1 is used. In
      ResidualsToBBoxes, we ensure that the phi predicted is between [0, pi).

    The Huber (SmoothL1) loss can then be applied to the delta between these
    target residuals and the model predicted residuals.

    [1] VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
        https://arxiv.org/abs/1711.06396

    [2] SECOND: Sparsely Embedded Convolutional Detection
        https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf

    Args:
      anchor_bboxes: tf.float32. where [..., :7] contains (x, y, z, dx, dy, dz,
        phi), corresponding to each anchor bbox parameters.
      assigned_gt_bboxes: tf.float32 of the same shape as anchor_bboxes
        containing the corresponding assigned ground-truth bboxes.

    Returns:
      A tf.float32 tensor of the same shape as anchor_bboxes with target
      residuals for every corresponding bbox.
    """
    anchor_bboxes_shape = py_utils.GetShape(anchor_bboxes)
    anchor_bboxes = py_utils.with_dependencies(
        [py_utils.assert_equal(anchor_bboxes_shape[-1], 7)], anchor_bboxes)
    assigned_gt_bboxes = py_utils.HasShape(assigned_gt_bboxes,
                                           anchor_bboxes_shape)

    x_a, y_a, z_a, dx_a, dy_a, dz_a, phi_a = tf.unstack(
        anchor_bboxes, num=7, axis=-1)
    x_gt, y_gt, z_gt, dx_gt, dy_gt, dz_gt, phi_gt = tf.unstack(
        assigned_gt_bboxes, num=7, axis=-1)

    diagonal_xy = tf.sqrt(tf.square(dx_a) + tf.square(dy_a))

    # The anchor dimensions is usually a hard-coded param given to the input
    # generator and should not be 0. We use CheckNumerics to ensure that is the
    # case.
    x_residual = py_utils.CheckNumerics((x_gt - x_a) / diagonal_xy)
    y_residual = py_utils.CheckNumerics((y_gt - y_a) / diagonal_xy)
    z_residual = py_utils.CheckNumerics((z_gt - z_a) / dz_a)

    dx_residual = py_utils.CheckNumerics(tf.math.log(dx_gt / dx_a))
    dy_residual = py_utils.CheckNumerics(tf.math.log(dy_gt / dy_a))
    dz_residual = py_utils.CheckNumerics(tf.math.log(dz_gt / dz_a))

    phi_residual = phi_gt - phi_a

    return tf.stack([
        x_residual, y_residual, z_residual,
        dx_residual, dy_residual, dz_residual,
        phi_residual,
    ], axis=-1)  # pyformat: disable

  def ResidualsToBBoxes(self,
                        anchor_bboxes,
                        residuals,
                        min_angle_rad=-np.pi,
                        max_angle_rad=np.pi):
    r"""Converts anchor_boxes and residuals to predicted bboxes.

    This converts predicted residuals into bboxes using the following formulae::

      x_predicted = x_a + x_residual * diagonal_xy
      y_predicted = y_a + y_residual * diagonal_xy
      z_predicted = z_a + z_residual * dz_a

      dx_predicted = dx_a * exp(dx_residual)
      dy_predicted = dy_a * exp(dy_residual)
      dz_predicted = dz_a * exp(dz_residual)

      # Adding the residual, and bounding it between
      # [min_angle_rad, max_angle_rad]
      phi_predicted = NormalizeAngleRad(phi_a + phi_residual,
                                        min_angle_rad, max_angle_rad)

    These equations follow from those in LocalizationResiduals, where we solve
    for the \*_gt variables.

    Args:
      anchor_bboxes: tf.float32. where [..., :7] contains (x, y, z, dx, dy, dz,
        phi), corresponding to each anchor bbox parameters.
      residuals: tf.float32 of the same shape as anchor_bboxes containing
        predicted residuals at each anchor location.
      min_angle_rad: Scalar with the minimum angle allowed (before wrapping)
        in radians.
      max_angle_rad: Scalar with the maximum angle allowed (before wrapping)
        in radians. This value usually should be pi.

    Returns:
      A tf.float32 tensor of the same shape as anchor_bboxes with predicted
      bboxes.
    """
    anchor_bboxes_shape = py_utils.GetShape(anchor_bboxes)
    anchor_bboxes = py_utils.with_dependencies(
        [py_utils.assert_equal(anchor_bboxes_shape[-1], 7)], anchor_bboxes)
    residuals = py_utils.HasShape(residuals, anchor_bboxes_shape)

    x_a, y_a, z_a, dx_a, dy_a, dz_a, phi_a = tf.unstack(
        anchor_bboxes, num=7, axis=-1)
    (x_residual, y_residual, z_residual, dx_residual, dy_residual, dz_residual,
     phi_residual) = tf.unstack(
         residuals, num=7, axis=-1)

    diagonal_xy = tf.sqrt(tf.square(dx_a) + tf.square(dy_a))

    x_predicted = x_a + x_residual * diagonal_xy
    y_predicted = y_a + y_residual * diagonal_xy
    z_predicted = z_a + z_residual * dz_a

    dx_predicted = dx_a * tf.exp(dx_residual)
    dy_predicted = dy_a * tf.exp(dy_residual)
    dz_predicted = dz_a * tf.exp(dz_residual)

    # We bound the angle between [min_angle_rad, max_angle_rad], which should
    # be passed in depending on the heading handling in the calling model.
    # If the model uses a sine(delta_phi) transformation in the loss, then it
    # cannot distinguish direction and a [0, np.pi]
    # [min_angle_rad, max_angle_rad] should be used.
    # If there is a heading encoding that is directional, most likely you
    # should use a [-np.pi, np.pi] [min_angle_rad, max_angle_rad].
    phi_predicted = phi_a + phi_residual
    phi_predicted = geometry.WrapAngleRad(phi_predicted, min_angle_rad,
                                          max_angle_rad)

    return tf.stack([
        x_predicted, y_predicted, z_predicted,
        dx_predicted, dy_predicted, dz_predicted,
        phi_predicted,
    ], axis=-1)  # pyformat: disable

  def NMSIndices(self,
                 bboxes,
                 scores,
                 max_output_size,
                 nms_iou_threshold=0.3,
                 score_threshold=0.01):
    """Apply NMS to a series of 3d bounding boxes in 7-DOF format.

    Args:
      bboxes: A [num_boxes, 7] floating point Tensor of bounding boxes in [x, y,
        z, dx, dy, dz, phi] format.
      scores: A [num_boxes] floating point Tensor containing box
        scores.
      max_output_size: Maximum number of boxes to predict per input.
      nms_iou_threshold: IoU threshold to use when determining whether two boxes
        overlap for purposes of suppression.
      score_threshold: The score threshold passed to NMS that allows NMS to
        quickly ignore irrelevant boxes.

    Returns:
      The NMS indices and the mask of the padded indices.
    """
    bboxes = py_utils.HasShape(bboxes, [-1, 7])

    # Extract x, y, w, h, then convert to extrema.
    #
    # Note that we drop the rotation angle because we don't have an NMS
    # operation that takes rotation into account.
    bboxes_2d = tf.stack(
        [bboxes[:, 0], bboxes[:, 1], bboxes[:, 3], bboxes[:, 4]], axis=-1)
    bboxes_extrema = geometry.XYWHToBBoxes(bboxes_2d)

    # Compute NMS with padding; we use the padded version so this function can
    # be used in a map_fn.  This function returns the scalar number of boxes
    # for each example.
    #
    # We use an IoU threshold of 0.3 since our anchor boxes have rotations
    # that make the default IoU threshold of 0.5 possibly too high.
    nms_index_padded, num_valid = tf.image.non_max_suppression_padded(
        bboxes_extrema,
        scores,
        iou_threshold=nms_iou_threshold,
        max_output_size=max_output_size,
        score_threshold=score_threshold,
        pad_to_max_output_size=True)

    # Return the mask of valid indices instead of just a scalar number.
    mask = tf.concat(
        [tf.ones([num_valid]),
         tf.zeros([max_output_size - num_valid])], axis=0)

    nms_index_padded = tf.where(mask > 0, nms_index_padded,
                                tf.zeros_like(nms_index_padded))
    return nms_index_padded, mask

  def BatchedNMSIndices(self,
                        bboxes,
                        scores,
                        nms_iou_threshold=0.3,
                        score_threshold=0.01,
                        max_num_boxes=None):
    """Batched version of NMSIndices.

    Args:
      bboxes: A [batch_size, num_boxes, 7] floating point Tensor of bounding
        boxes in [x, y, z, dx, dy, dz, phi] format.
      scores: A [batch_size, num_boxes, num_classes] floating point Tensor
        containing box scores.
      nms_iou_threshold: IoU threshold to use when determining whether two boxes
        overlap for purposes of suppression.
      score_threshold: The score threshold passed to NMS that allows NMS to
        quickly ignore irrelevant boxes.
      max_num_boxes: The maximum number of boxes per example to emit. If None,
        this value is set to num_boxes from the shape of bboxes.

    Returns:
      The NMS indices and the mask of the padded indices for each example
      in the batch.
    """
    batch_size, num_boxes = py_utils.GetShape(bboxes, 2)

    if max_num_boxes is not None:
      max_output_size = max_num_boxes
    else:
      max_output_size = num_boxes

    output_shape = [batch_size, max_output_size]

    def NMSBody(args):
      bbox, score = args
      return self.NMSIndices(bbox, score, max_output_size, nms_iou_threshold,
                             score_threshold)

    nms_indices, valid_mask = tf.map_fn(
        fn=NMSBody,
        elems=(bboxes, scores),
        dtype=(tf.int32, tf.float32),
        back_prop=False)

    nms_indices = py_utils.PadOrTrimTo(nms_indices, output_shape)
    return nms_indices, valid_mask

  def BatchedOrientedNMSIndices(self, bboxes, scores, nms_iou_threshold,
                                score_threshold, max_boxes_per_class):
    """Runs batched version of a Per-Class 3D (7-DOF) Non Max Suppression.

    All outputs have shape [batch_size, num_classes, max_boxes_per_class].

    Args:
      bboxes: A [batch_size, num_boxes, 7] floating point Tensor of bounding
        boxes in [x, y, z, dx, dy, dz, phi] format.
      scores: A [batch_size, num_boxes, num_classes] floating point Tensor
        containing box scores.
      nms_iou_threshold: Either a float or a list of floats of len num_classes
        with the IoU threshold to use when determining whether two boxes overlap
        for purposes of suppression.
      score_threshold: Either a float or a list of floats of len num_classes
        with the score threshold that allows NMS to quickly ignore boxes.
      max_boxes_per_class: An integer scalar with the maximum number of boxes
        per example to emit per class.

    Returns:
      A tuple of 3 tensors:

      - bbox_indices: An int32 Tensor with the indices of the chosen boxes.
        Values are in sort order until the class_idx switches.
      - bbox_scores: A float32 Tensor with the score for each box.
      - valid_mask: A float32 Tensor with 1/0 values indicating the validity of
        each box. 1 indicates valid, and 0 invalid.
    """
    bboxes = py_utils.HasShape(bboxes, [-1, -1, 7])
    batch_size, num_boxes = py_utils.GetShape(bboxes, 2)
    scores = py_utils.HasShape(scores, [batch_size, num_boxes, -1])
    _, _, num_classes = py_utils.GetShape(scores)

    # Force the thresholds to be tensors of len num_classes
    nms_iou_threshold = tf.broadcast_to(
        tf.convert_to_tensor(nms_iou_threshold), [num_classes])
    score_threshold = tf.broadcast_to(
        tf.convert_to_tensor(score_threshold), [num_classes])

    def NMSBody(args):
      per_sample_bboxes, per_sample_scores = args
      indices, scores, mask = ops.non_max_suppression_3d(
          per_sample_bboxes,
          per_sample_scores,
          nms_iou_threshold=nms_iou_threshold,
          score_threshold=score_threshold,
          max_boxes_per_class=max_boxes_per_class)
      return indices, scores, mask

    bbox_indices, bbox_scores, valid_mask = tf.map_fn(
        fn=NMSBody,
        elems=(bboxes, scores),
        dtype=(tf.int32, tf.float32, tf.float32),
        back_prop=False)

    output_shape = [batch_size, num_classes, max_boxes_per_class]
    bbox_indices = py_utils.PadOrTrimTo(bbox_indices, output_shape)
    bbox_scores = py_utils.PadOrTrimTo(bbox_scores, output_shape)
    valid_mask = py_utils.PadOrTrimTo(valid_mask, output_shape)
    return bbox_indices, bbox_scores, valid_mask

  def CornersToImagePlane(self, corners, velo_to_image_plane):
    """Project 3d box corners to the image plane.

    Args:
      corners: A [batch, num_boxes, 8, 3] floating point tensor containing the 8
        corners points for each 3d bounding box.
      velo_to_image_plane: A [batch, 3, 4] batch set of projection matrices from
        velo xyz to image plane xy. After multiplication, you need to divide by
        last coordinate to recover 2D pixel locations.

    Returns:
      A [batch, num_boxes, 8, 2] floating point Tensor containing the 3D
      bounding box corners projected to the image plane.
    """
    batch_size, num_boxes, _, _ = py_utils.GetShape(corners, 4)

    def CornersToPlaneBody(args):
      """Body of function to convert each bounding box to the image plane."""
      (corners, velo_to_image_plane) = args
      # corners[i] is [num_boxes, 8, 3]: flatten the points in this batch and do
      # the conversion in one call.
      bbox_corners = tf.reshape(corners, [-1, 3])
      image_plane_corners = geometry.PointsToImagePlane(bbox_corners,
                                                        velo_to_image_plane)
      image_plane_corners = tf.reshape(image_plane_corners, [-1, 8, 2])
      return image_plane_corners

    corners_in_image_plane = tf.map_fn(
        fn=CornersToPlaneBody,
        elems=(corners, velo_to_image_plane),
        dtype=tf.float32,
        back_prop=False)

    corners_in_image_plane = py_utils.HasShape(corners_in_image_plane,
                                               [batch_size, num_boxes, 8, 2])
    return corners_in_image_plane


def RandomPadOrTrimTo(tensor_list, num_points_out, seed=None):
  """Pads or Trims a list of Tensors on the major dimension.

  Slices if there are more points, or pads if not enough.

  In this implementation:
    Padded points are random duplications of real points.
    Sliced points are a random subset of the real points.

  Args:
    tensor_list: A list of tf.Tensor objects to pad or trim along first dim. All
      tensors are expected to have the same first dimension.
    num_points_out: An int for the requested number of points to trim/pad to.
    seed: Random seed to use for random generators.

  Returns:
    A tuple of output_tensors and a padding indicator.

    - output_tensors: A list of padded or trimmed versions of our tensor_list
      input tensors, all with the same first dimension.
    - padding: A tf.float32 tf.Tensor of shape [num_points_out] with 0 if the
      point is real, 1 if it is padded.
  """
  actual_num = tf.shape(tensor_list[0])[0]
  point_idx = tf.range(num_points_out, dtype=tf.int32)
  padding_tensor = tf.where(point_idx < actual_num,
                            tf.zeros([num_points_out], dtype=tf.float32),
                            tf.ones([num_points_out], dtype=tf.float32))

  def _Slicing():
    # Choose a random set of indices.
    indices = tf.range(actual_num)
    indices = tf.random.shuffle(indices, seed=seed)[:num_points_out]
    return [tf.gather(t, indices, axis=0) for t in tensor_list]

  def _Padding():
    indices = tf.random.uniform([num_points_out - actual_num],
                                minval=0,
                                maxval=actual_num,
                                dtype=tf.int32,
                                seed=seed)
    padded = []
    for t in tensor_list:
      padded.append(tf.concat([t, tf.gather(t, indices, axis=0)], axis=0))
    return padded

  def _PadZeros():
    padded = []
    for t in tensor_list:
      shape = tf.concat([[num_points_out], tf.shape(t)[1:]], axis=0)
      padded.append(tf.zeros(shape=shape, dtype=t.dtype))
    return padded

  data = tf.cond(
      actual_num > num_points_out,
      _Slicing, lambda: tf.cond(tf.equal(actual_num, 0), _PadZeros, _Padding))
  return (data, padding_tensor)
