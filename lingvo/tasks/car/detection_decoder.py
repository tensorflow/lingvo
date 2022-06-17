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
    bbox_indices: Indices of the boxes selected after NMS. Tensor of shape
      [batch_size, num_classes, max_boxes_per_class] if per class NMS is used.
      If single class NMS, this will be of shape [batch_size,
      max_boxes_per_class].
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
    bbox_indices: Indices of the boxes selected after NMS. Tensor of shape
      [batch_size, num_classes, max_boxes_per_class].
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
  return bbox_indices, predicted_bboxes, bbox_scores, valid_mask


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
    nms_indices: Indices of the boxes selected after NMS. Tensor of shape
      [batch_size, num_classes, max_boxes_per_class].
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
  return nms_indices, predicted_bboxes, classification_scores, valid_mask


def HeatMapNMS(heat_map_scores, kernel_size, max_num_objects, score_threshold):
  """Extract top k peaks of heat map from heat_map_scores.

  Args:
    heat_map_scores: A [batch, gx, gy, nms_cls] float Tensor with values ranging
      from (0, 1) indicating the likelihood of that object being a center.
    kernel_size: A list of integers specifying the max pooling kernel size to
      use on the heat_map_scores input.
    max_num_objects: Maximum number of peaks to extract from the heat map.
    score_threshold: Floating point value in [0., 1.] specifying the minimum
      score that will be considered a peak in the peak_heat_map output.

  Returns:
    A Nested map containing:

    - top_k_indices: A [batch_size, nms_cls, max_num_objects, 4]
      Tensor indicating which indices are the top heat map scores for
      each sample;

    - peak_heat_map is a [batch_size, gx, gy, 1] floating
      point Tensor visualizing the top peaks that pass the
      p.top_k_score_threshold score filter where the values are either
      1.0 for a peak and 0.0 if not a peak.
  """
  bs, grid_height, grid_width, nms_cls = py_utils.GetShape(heat_map_scores, 4)
  # Keep the peaks using maxpooling / masking.
  max_scores = tf.nn.max_pool(
      heat_map_scores, ksize=kernel_size, strides=[1, 1, 1, 1], padding='SAME')
  # Creates a mask only returning the heat map pixels that were the
  # max of their pooled neighborhood.
  peak_mask = tf.cast(tf.equal(heat_map_scores, max_scores), tf.float32)

  # Apply the mask on the original heat map scores to get the "pooled"
  # score peaks.
  filtered_heat_map = heat_map_scores * peak_mask

  # bs, nms_cls, grid_height, grid_width.
  t_filtered_heat_map = tf.transpose(filtered_heat_map, [0, 3, 1, 2])
  flattened_t_heat_map = tf.reshape(t_filtered_heat_map, [bs, nms_cls, -1])
  # Compute the indices of the top K for every sample and every class.
  # bs, nms_cls, max_num_objects
  _, batch_top_k_indices = tf.nn.top_k(
      flattened_t_heat_map, k=max_num_objects, sorted=True)
  # bs, max_num_objects, nms_cls
  batch_top_k_indices = tf.transpose(batch_top_k_indices, [0, 2, 1])
  # Convert batch of top_k indices into a single vector of indices
  # for use with unravel_index
  batch_index = tf.range(bs) * grid_height * grid_width * nms_cls
  flattened_indices = (
      tf.reshape(batch_index, [bs, 1, 1]) + batch_top_k_indices * nms_cls +
      tf.reshape(tf.range(nms_cls), [1, 1, nms_cls]))
  flattened_indices = tf.reshape(flattened_indices, [-1])
  top_k_indices = tf.unravel_index(flattened_indices,
                                   tf.shape(filtered_heat_map))
  top_k_indices = py_utils.HasShape(
      tf.transpose(top_k_indices), [bs * max_num_objects * nms_cls, 4])

  # Filter top k to include only those indices that pass
  # a score threshold to produce the peak heat map for visualization.
  top_k_scores = tf.reshape(tf.gather_nd(heat_map_scores, top_k_indices), [-1])

  ones = tf.ones(tf.shape(top_k_indices)[0])
  ones_mask = tf.cast(tf.greater(top_k_scores, score_threshold), ones.dtype)
  # If the score is greater than threshold (ones_mask), its filtered_value is 1.
  # Otherwise, (1 - ones_mask), its filtered_value is 0.
  filtered_values = ones * ones_mask + (1 - ones) * (1 - ones_mask)

  ret = py_utils.NestedMap()
  ret.peak_heat_map = tf.scatter_nd(top_k_indices, filtered_values,
                                    tf.shape(filtered_heat_map))

  # Also return the raw top_k_indices.
  ret.top_k_indices = top_k_indices

  return ret


def DecodeWithMaxPoolNMS(predicted_bboxes,
                         classification_scores,
                         score_threshold,
                         heatmap_shape,
                         kernel_size,
                         max_boxes_per_class=None,
                         use_oriented_per_class_nms=False):
  """Perform MaxPoolNMS on predicted bounding boxes / associated logits.

  This method is illustrated in: https://arxiv.org/pdf/1904.07850.pdf.

  Args:
    predicted_bboxes: [batch_size, num_boxes, 7] float Tensor containing
      predicted bounding box coordinates.
    classification_scores: [batch_size, num_boxes, num_classes] float Tensor
      containing predicted classification scores for each box.
    score_threshold: The score threshold passed to NMS that allows NMS to
      quickly ignore irrelevant boxes. It has to be a list of len num_classes.
    heatmap_shape: The shape of the classification_score and predicted_bboxes
      prediction. At the beginning of this method, we have to reshape these two
      tensors according to the shape. In a pillars model, this shape should be
      [bs, nx, ny]
    kernel_size: The kernel_size used for max-pool the classification heatmap.
    max_boxes_per_class: The maximum number of boxes per example to emit. If
      None, this value is set to num_boxes from the shape of predicted_bboxes.
    use_oriented_per_class_nms: Whether to use the oriented per class NMS or
      treat everything as one class and having no orientation.

  Returns:
    bbox_indices: Indices of the boxes selected after NMS. Tensor of shape
      [batch_size, num_classes, max_boxes_per_class] if per class NMS is used.
      If single class NMS, this will be of shape [batch_size,
      max_boxes_per_class].
    predicted_bboxes: Filtered bboxes after NMS of shape
      [batch_size, num_classes, max_boxes_per_class, 7].
    bbox_scores: A float32 Tensor with the score for each box of shape
      [batch_size, num_classes, max_boxes_per_class].
    valid_mask: A float32 Tensor with 1/0 values indicating the validity of
      each box. 1 indicates valid, and 0 invalid. Tensor of shape
      [batch_size, num_classes, max_boxes_per_class].
  """
  # First reshape the predicted_bboxes and classification_scores to
  # target shape.
  bs, nx, ny = heatmap_shape
  bs, num_boxes, num_classes = py_utils.GetShape(classification_scores, 3)
  predicted_bboxes = py_utils.HasShape(predicted_bboxes, [bs, num_boxes, 7])

  classification_scores = tf.reshape(classification_scores,
                                     heatmap_shape + [-1, num_classes])
  predicted_bboxes = tf.reshape(predicted_bboxes, heatmap_shape + [-1, 7])

  # Find the prediction on z-axis with the highest confidence.
  nz = py_utils.GetShape(predicted_bboxes)[-2]
  valid_idx = tf.argmax(classification_scores, axis=-2)
  valid_idx = py_utils.HasShape(valid_idx, [bs, nx, ny, num_classes])
  classification_scores = tf.reduce_max(classification_scores, axis=-2)
  classification_scores = py_utils.HasShape(classification_scores,
                                            [bs, nx, ny, num_classes])

  predicted_bboxes = tf.array_ops.batch_gather(predicted_bboxes, valid_idx)
  predicted_bboxes = py_utils.HasShape(predicted_bboxes,
                                       [bs, nx, ny, num_classes, 7])

  score_threshold = tf.convert_to_tensor(score_threshold, dtype=tf.float32)
  score_threshold = py_utils.HasShape(score_threshold, [num_classes])

  if use_oriented_per_class_nms:
    nms_scores = classification_scores
    nms_num_classes = num_classes
  else:
    likely_labels = tf.argmax(classification_scores, axis=-1)[..., tf.newaxis]
    nms_scores = tf.reduce_max(classification_scores, axis=-1, keepdims=True)
    nms_num_classes = 1

    # When background is the most likely class for the box, mask out the scores
    # of that box from NMS scoring so the background boxes don't dominate the
    # NMS.
    nms_scores *= tf.cast(likely_labels > 0, tf.float32)

  if max_boxes_per_class is None:
    max_boxes_per_class = num_boxes

  ret = HeatMapNMS(nms_scores, kernel_size, max_boxes_per_class, 0.)
  top_k_indices = ret.top_k_indices
  top_k_indices = py_utils.HasShape(
      top_k_indices, [bs * max_boxes_per_class * nms_num_classes, 4])
  top_k_indices = tf.reshape(top_k_indices,
                             [bs, max_boxes_per_class, nms_num_classes, 4])

  if not use_oriented_per_class_nms:
    top_k_indices = tf.tile(top_k_indices, [1, 1, num_classes, 1])
    pad_top_k_indices = tf.reshape(
        tf.range(num_classes), [1, 1, num_classes, 1])
    pad_top_k_indices = tf.tile(pad_top_k_indices,
                                [bs, max_boxes_per_class, 1, 1])
    top_k_indices = tf.concat([top_k_indices[..., :3], pad_top_k_indices],
                              axis=-1)

  top_k_indices = tf.transpose(top_k_indices, [0, 2, 1, 3])
  top_k_indices = py_utils.HasShape(top_k_indices,
                                    [bs, num_classes, max_boxes_per_class, 4])
  predicted_scores = tf.gather_nd(classification_scores, top_k_indices)
  predicted_bboxes = tf.gather_nd(predicted_bboxes, top_k_indices)

  r_score_threshold = tf.reshape(score_threshold, [1, num_classes, 1])
  predicted_mask = tf.greater(predicted_scores, r_score_threshold)
  predicted_mask = tf.cast(predicted_mask, tf.float32)

  valid_idx = tf.cast(tf.gather_nd(valid_idx, top_k_indices), tf.int32)
  valid_idx = py_utils.HasShape(valid_idx,
                                [bs, num_classes, max_boxes_per_class])
  top_k_indices = tf.concat([
      top_k_indices[..., :3], valid_idx[..., tf.newaxis], top_k_indices[..., 3:]
  ], axis=-1)  # pyformat: disable

  # Flatten top_k_indices.
  # After flatten, the corresponding elements can be gathered from original
  # arrays by tf.gather(x, top_k_indices, batch_dim=1).
  top_k_indices = (
      top_k_indices[..., 1] * ny * nz + top_k_indices[..., 2] * nz +
      top_k_indices[..., 3])

  return top_k_indices, predicted_bboxes, predicted_scores, predicted_mask
