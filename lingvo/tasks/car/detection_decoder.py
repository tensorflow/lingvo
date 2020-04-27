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
"""Functions to help with decoding detector model outputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.car import detection_3d_lib


def DecodeWithNMS(predicted_bboxes,
                  classification_scores,
                  nms_iou_threshold,
                  score_threshold,
                  max_boxes_per_class=None,
                  use_oriented_per_class_nms=False):
  """Perform NMS on predicted bounding boxes / associated logits.

  Args:
    predicted_bboxes: [batch_size, num_boxes, 7] float Tensor containing
      predicted bounding box coordinates.
    classification_scores: [batch_size, num_boxes, num_classes] float Tensor
      containing predicted classification scores for each box.
    nms_iou_threshold: IoU threshold to use when determining whether two boxes
      overlap for purposes of suppression. Either a float or a list of len
      num_classes.
    score_threshold: The score threshold passed to NMS that allows NMS to
      quickly ignore irrelevant boxes. Either a float or a list of len
      num_classes.
    max_boxes_per_class: The maximum number of boxes per example to emit.
        If None, this value is set to num_boxes from the shape of
        predicted_bboxes.
    use_oriented_per_class_nms: Whether to use the oriented per class NMS
      or treat everything as one class and having no orientation.

  Returns:
    predicted_bboxes: Filtered bboxes after NMS of shape
      [batch_size, num_classes, max_boxes_per_class, 7].
    bbox_scores: A float32 Tensor with the score for each box of shape
      [batch_size, num_classes, max_boxes_per_class].
    valid_mask: A float32 Tensor with 1/0 values indicating the validity of
      each box. 1 indicates valid, and 0 invalid. Tensor of shape
      [batch_size, num_classes, max_boxes_per_class].
  """
  if use_oriented_per_class_nms:
    nms_fn = _MultiClassOrientedDecodeWithNMS
  else:
    nms_fn = _SingleClassDecodeWithNMS

  return nms_fn(
      predicted_bboxes=predicted_bboxes,
      classification_scores=classification_scores,
      nms_iou_threshold=nms_iou_threshold,
      score_threshold=score_threshold,
      max_boxes_per_class=max_boxes_per_class)


def _MultiClassOrientedDecodeWithNMS(predicted_bboxes,
                                     classification_scores,
                                     nms_iou_threshold,
                                     score_threshold,
                                     max_boxes_per_class=None):
  """Perform Oriented Per Class NMS on predicted bounding boxes / logits.

  Args:
    predicted_bboxes: [batch_size, num_boxes, 7] float Tensor containing
      predicted bounding box coordinates.
    classification_scores: [batch_size, num_boxes, num_classes] float Tensor
      containing predicted classification scores for each box.
    nms_iou_threshold: IoU threshold to use when determining whether two boxes
      overlap for purposes of suppression. Either a float or a list of len
      num_classes.
    score_threshold: The score threshold passed to NMS that allows NMS to
      quickly ignore irrelevant boxes. Either a float or a list of len
      num_classes. It is strongly recommended that the score for non-active
      classes (like background) be set to 1 so they are discarded.
    max_boxes_per_class: The maximum number of boxes per example to emit. If
      None, this value is set to num_boxes from the shape of predicted_bboxes.

  Returns:
    predicted_bboxes: Filtered bboxes after NMS of shape
      [batch_size, num_classes, max_boxes_per_class, 7].
    bbox_scores: A float32 Tensor with the score for each box of shape
      [batch_size, num_classes, max_boxes_per_class].
    valid_mask: A float32 Tensor with 1/0 values indicating the validity of
      each box. 1 indicates valid, and 0 invalid. Tensor of shape
      [batch_size, num_classes, max_boxes_per_class].
  """
  utils_3d = detection_3d_lib.Utils3D()
  predicted_bboxes = py_utils.HasShape(predicted_bboxes, [-1, -1, 7])
  batch_size, num_predicted_boxes, _ = py_utils.GetShape(predicted_bboxes)
  classification_scores = py_utils.HasShape(
      classification_scores, [batch_size, num_predicted_boxes, -1])
  _, _, num_classes = py_utils.GetShape(classification_scores)

  if max_boxes_per_class is None:
    max_boxes_per_class = num_predicted_boxes

  # Compute NMS for every sample in the batch.
  bbox_indices, bbox_scores, valid_mask = utils_3d.BatchedOrientedNMSIndices(
      predicted_bboxes,
      classification_scores,
      nms_iou_threshold=nms_iou_threshold,
      score_threshold=score_threshold,
      max_boxes_per_class=max_boxes_per_class)

  # TODO(bencaine): Consider optimizing away the tf.tile or make upstream
  # changes to make predicted boxes include a class dimension.
  # Get the original box for each index selected by NMS.
  predicted_bboxes = tf.tile(predicted_bboxes[:, tf.newaxis, :, :],
                             [1, num_classes, 1, 1])
  predicted_bboxes = tf.array_ops.batch_gather(predicted_bboxes, bbox_indices)
  return predicted_bboxes, bbox_scores, valid_mask


def _SingleClassDecodeWithNMS(predicted_bboxes,
                              classification_scores,
                              nms_iou_threshold,
                              score_threshold,
                              max_boxes_per_class=None):
  """Perform NMS on predicted bounding boxes / associated logits.

  Args:
    predicted_bboxes: [batch_size, num_boxes, 7] float Tensor containing
      predicted bounding box coordinates.
    classification_scores: [batch_size, num_boxes, num_classes] float Tensor
      containing predicted classification scores for each box.
    nms_iou_threshold: IoU threshold to use when determining whether two boxes
      overlap for purposes of suppression.
    score_threshold: The score threshold passed to NMS that allows NMS to
      quickly ignore irrelevant boxes.
    max_boxes_per_class: The maximum number of boxes per example to emit. If
      None, this value is set to num_boxes from the shape of predicted_bboxes.

  Returns:
    predicted_bboxes: Filtered bboxes after NMS of shape
      [batch_size, num_classes, max_boxes_per_class, 7].
    bbox_scores: A float32 Tensor with the score for each box of shape
      [batch_size, num_classes, max_boxes_per_class].
    valid_mask: A float32 Tensor with 1/0 values indicating the validity of
      each box. 1 indicates valid, and 0 invalid. Tensor of shape
      [batch_size, num_classes, max_boxes_per_class].
  """
  utils_3d = detection_3d_lib.Utils3D()
  predicted_bboxes = py_utils.HasShape(predicted_bboxes, [-1, -1, 7])
  batch_size, num_predicted_boxes, _ = py_utils.GetShape(predicted_bboxes)
  classification_scores = py_utils.HasShape(
      classification_scores, [batch_size, num_predicted_boxes, -1])
  _, _, num_classes = py_utils.GetShape(classification_scores)

  if not isinstance(nms_iou_threshold, float):
    raise ValueError('Single class NMS only supports a scalar '
                     '`nms_iou_threshold`.')
  if not isinstance(score_threshold, float):
    raise ValueError('Single class NMS only supports a scalar '
                     '`score_threshold`.')

  if max_boxes_per_class is None:
    max_boxes_per_class = num_predicted_boxes

  # TODO(jngiam): Change to be per-class bboxes, and hence, per-class NMS, and
  # per-class thresholding.
  # [batch, num_predicted_boxes]
  nms_scores = tf.reduce_max(classification_scores, axis=-1)

  # Compute the most likely label by computing the highest class score from
  # the output of the sigmoid.
  likely_labels = tf.argmax(classification_scores, axis=-1)

  # When background is the most likely class for the box, mask out the scores
  # of that box from NMS scoring so the background boxes don't dominate the
  # NMS.
  nms_scores *= tf.cast(likely_labels > 0, tf.float32)

  # Compute NMS for every sample in the batch.
  nms_indices, valid_mask = utils_3d.BatchedNMSIndices(
      predicted_bboxes,
      nms_scores,
      nms_iou_threshold=nms_iou_threshold,
      score_threshold=score_threshold,
      max_num_boxes=max_boxes_per_class)

  # Reorder the box data and logits according to NMS scoring.
  predicted_bboxes = tf.array_ops.batch_gather(predicted_bboxes, nms_indices)
  classification_scores = tf.array_ops.batch_gather(classification_scores,
                                                    nms_indices)

  # Now reformat the output of NMS to match the format of the
  # MultiClassOrientedDecodeWithNMS, which outputs a per class NMS result.
  # This takes the leading shape of
  # [batch_size, num_classes, max_boxes_per_class] for all outputs, which
  # means since this NMS is not class specific we need to tile the outputs
  # num_classes times or reorder the data such that its [batch, num_classes].
  predicted_bboxes = tf.tile(predicted_bboxes[:, tf.newaxis, :, :],
                             [1, num_classes, 1, 1])
  classification_scores = tf.transpose(classification_scores, (0, 2, 1))
  classification_scores = py_utils.HasShape(
      classification_scores, [batch_size, num_classes, max_boxes_per_class])
  valid_mask = tf.tile(valid_mask[:, tf.newaxis, :], [1, num_classes, 1])
  return predicted_bboxes, classification_scores, valid_mask
