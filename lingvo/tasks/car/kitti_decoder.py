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
"""Base models for point-cloud based detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.tasks.car import base_decoder
from lingvo.tasks.car import detection_3d_lib
from lingvo.tasks.car import detection_3d_metrics
from lingvo.tasks.car import geometry
from lingvo.tasks.car import kitti_ap_metric
from lingvo.tasks.car import kitti_metadata
from lingvo.tasks.car import transform_util
import numpy as np
from six.moves import range


class KITTIDecoder(base_decoder.BaseDecoder):
  """A decoder to use for decoding a detector model on KITTI.

  This class implements the basic Decoder metrics for KITTI to provide
  visualizations and AP calculations.
  """

  @classmethod
  def Params(cls):
    p = super(KITTIDecoder, cls).Params()
    p.Define(
        'filter_predictions_outside_frustum', False,
        'If true, predictions whose bounding box center is outside of the '
        'image frustum are dropped.')
    p.Define(
        'truncation_threshold', 0.0,
        'Specifies how much of a bounding box can be truncated '
        'by the edge of the image frustum and still be kept. A value of 0.0 '
        'means that we only drop predictions whose 2d bounding box '
        'falls entirely outside the image frustum.  A value of 1.0 means '
        'we drop predictions where *any* portion of the bounding box falls '
        'outside the frustum.')
    p.ap_metric = kitti_ap_metric.KITTIAPMetrics.Params(
        kitti_metadata.KITTIMetadata())
    return p

  def CreateDecoderMetrics(self):
    """Decoder metrics for KITTI."""
    p = self.params

    kitti_metric_p = p.ap_metric.Copy().Set(cls=kitti_ap_metric.KITTIAPMetrics)
    apm = kitti_metric_p.Instantiate()
    class_names = apm.metadata.ClassNames()

    # Convert the list of class names to a dictionary mapping class_id -> name.
    class_id_to_name = dict(enumerate(class_names))

    top_down_transform = transform_util.MakeCarToImageTransform(
        pixels_per_meter=32.,
        image_ref_x=512.,
        image_ref_y=1408.,
        flip_axes=True)

    decoder_metrics = py_utils.NestedMap({
        'top_down_visualization':
            (detection_3d_metrics.TopDownVisualizationMetric(
                top_down_transform,
                image_height=1536,
                image_width=1024,
                class_id_to_name=class_id_to_name)),
        'num_samples_in_batch': metrics.AverageMetric(),
        'kitti_AP_v2': apm,
    })

    decoder_metrics.mesh = detection_3d_metrics.WorldViewer()

    if p.summarize_boxes_on_image:
      decoder_metrics.camera_visualization = (
          detection_3d_metrics.CameraVisualization(
              bbox_score_threshold=p.visualization_classification_threshold))

    return decoder_metrics

  def _CreateFrustumMask(self, bbox_corners_image, bbox2d_corners_image_clipped,
                         image_height, image_width):
    """Creates a box mask for boxes whose projections fall outside of image."""
    p = self.params
    batch_size, num_boxes = py_utils.GetShape(bbox_corners_image, 2)
    if not p.filter_predictions_outside_frustum:
      return tf.ones(shape=(batch_size, num_boxes), dtype=tf.float32)

    def _MinMax(bbox_corners):
      """Computes the min and max over corners."""
      bbox_min = tf.reduce_min(bbox_corners, axis=-1)
      bbox_max = tf.reduce_max(bbox_corners, axis=-1)
      bbox_min = py_utils.HasShape(bbox_min, [batch_size, num_boxes])
      bbox_max = py_utils.HasShape(bbox_max, [batch_size, num_boxes])
      return bbox_min, bbox_max

    bbox_min_x, bbox_max_x = _MinMax(bbox_corners_image[:, :, :, 0])
    bbox_min_y, bbox_max_y = _MinMax(bbox_corners_image[:, :, :, 1])

    # Compute the fraction of the clipped 2d image projection and the
    # full 2d image projection.  We simply need to divide the area
    # of each cropped box by the area of the full box to get the
    # overlap fraction.
    original_area = (bbox_max_x - bbox_min_x) * (bbox_max_y - bbox_min_y)
    bbox_clipped_x_min = bbox2d_corners_image_clipped[..., 0]
    bbox_clipped_y_min = bbox2d_corners_image_clipped[..., 1]
    bbox_clipped_x_max = bbox2d_corners_image_clipped[..., 2]
    bbox_clipped_y_max = bbox2d_corners_image_clipped[..., 3]
    clipped_area = (bbox_clipped_x_max - bbox_clipped_x_min) * (
        bbox_clipped_y_max - bbox_clipped_y_min)
    fraction = clipped_area / original_area

    frustum_mask = (fraction > p.truncation_threshold)
    frustum_mask = py_utils.HasShape(frustum_mask, [batch_size, num_boxes])
    frustum_mask = tf.cast(frustum_mask, tf.float32)
    return frustum_mask

  def _BBox2DImage(self, bbox_corners_image, input_images):
    """Compute [xmin, ymin, xmax, ymax] 2D bounding boxes from corners."""
    # Clip the boundaries of the bounding box to the image width/height.
    bci_x = bbox_corners_image[..., 0:1]
    image_width = tf.broadcast_to(
        input_images.width[..., tf.newaxis, tf.newaxis], tf.shape(bci_x))
    bci_x = tf.clip_by_value(bci_x, 0.0, tf.cast(image_width, tf.float32))

    bci_y = bbox_corners_image[..., 1:2]
    image_height = tf.broadcast_to(
        input_images.height[..., tf.newaxis, tf.newaxis], tf.shape(bci_y))
    bci_y = tf.clip_by_value(bci_y, 0.0, tf.cast(image_height, tf.float32))

    bbox_corners_image_clipped = tf.concat([bci_x, bci_y], axis=-1)

    # Compute the [xmin, ymin, xmax, ymax] bounding boxes from [batch,
    # num_boxes, 8, 2] extrema.
    min_vals = tf.math.reduce_min(bbox_corners_image_clipped, axis=2)
    max_vals = tf.math.reduce_max(bbox_corners_image_clipped, axis=2)
    bbox2d_corners_image = tf.concat([min_vals, max_vals], axis=2)
    return bbox2d_corners_image

  def ProcessOutputs(self, input_batch, model_outputs):
    """Produce additional decoder outputs for KITTI.

    Args:
      input_batch: A .NestedMap of the inputs to the model.
      model_outputs: A .NestedMap of the outputs of the model, including::
        - per_class_predicted_bboxes: [batch, num_classes, num_boxes, 7] float
          Tensor with per class 3D (7 DOF) bounding boxes.
        - per_class_predicted_bbox_scores: [batch, num_classes, num_boxes] float
          Tensor with per class, per box scores.
        - per_class_valid_mask: [batch, num_classes, num_boxes] masking Tensor
          indicating which boxes were still kept after NMS for each class.

    Returns:
      A NestedMap of additional decoder outputs needed for
      PostProcessDecodeOut.
    """
    p = self.params
    per_class_predicted_bboxes = model_outputs.per_class_predicted_bboxes
    batch_size, num_classes, num_boxes, _ = py_utils.GetShape(
        per_class_predicted_bboxes)
    flattened_num_boxes = num_classes * num_boxes

    input_labels = input_batch.decoder_copy.labels
    input_lasers = input_batch.decoder_copy.lasers
    input_images = input_batch.decoder_copy.images

    with tf.device('/cpu:0'):
      # Convert the predicted bounding box points to their corners
      # and then project them to the image plane.
      #
      # This output can be used to:
      #
      # A) Visualize bounding boxes (2d or 3d) on the camera image.
      #
      # B) Compute the height of the predicted boxes to filter 'too small' boxes
      #    as is done in the KITTI eval.
      predicted_bboxes = tf.reshape(per_class_predicted_bboxes,
                                    [batch_size, flattened_num_boxes, 7])
      bbox_corners = geometry.BBoxCorners(predicted_bboxes)
      bbox_corners = py_utils.HasShape(bbox_corners,
                                       [batch_size, flattened_num_boxes, 8, 3])
      utils_3d = detection_3d_lib.Utils3D()
      bbox_corners_image = utils_3d.CornersToImagePlane(
          bbox_corners, input_images.velo_to_image_plane)
      bbox_corners_image = py_utils.HasShape(
          bbox_corners_image, [batch_size, flattened_num_boxes, 8, 2])

      # Clip the bounding box corners so they remain within
      # the image coordinates.
      bbox2d_corners_image_clipped = self._BBox2DImage(bbox_corners_image,
                                                       input_images)
      bbox2d_corners_image_clipped = py_utils.HasShape(
          bbox2d_corners_image_clipped, [batch_size, flattened_num_boxes, 4])

      # Compute the frustum mask to filter out bounding boxes that
      # are 'outside the frustum'.
      frustum_mask = self._CreateFrustumMask(bbox_corners_image,
                                             bbox2d_corners_image_clipped,
                                             input_images.height,
                                             input_images.width)

      # Reshape all of these back to [batch_size, num_classes, num_boxes, ...]
      bbox_corners_image = tf.reshape(
          bbox_corners_image, [batch_size, num_classes, num_boxes, 8, 2])

      bbox2d_corners_image_clipped = tf.reshape(
          bbox2d_corners_image_clipped, [batch_size, num_classes, num_boxes, 4])
      frustum_mask = tf.reshape(frustum_mask,
                                [batch_size, num_classes, num_boxes])

    ret = py_utils.NestedMap({
        # For mAP eval
        'source_ids': input_labels.source_id,
        'difficulties': input_labels.difficulties,
        'num_points_in_bboxes': input_batch.labels.bboxes_3d_num_points,
        # For exporting.
        'velo_to_image_plane': input_images.velo_to_image_plane,
        'velo_to_camera': input_images.velo_to_camera,
        # Predictions.
        'bbox_corners_image': bbox_corners_image,
        'bbox2d_corners_image': bbox2d_corners_image_clipped,
        'frustum_mask': frustum_mask,
        # Ground truth.
        'bboxes_3d': input_labels.bboxes_3d,
        'bboxes_3d_mask': input_labels.bboxes_3d_mask,
        'unfiltered_bboxes_3d_mask': input_labels.unfiltered_bboxes_3d_mask,
        'labels': input_labels.labels,
    })

    laser_sample = self._SampleLaserForVisualization(
        input_lasers.points_xyz, input_lasers.points_padding)
    ret.update(laser_sample)

    if p.summarize_boxes_on_image:
      ret.camera_images = input_images.image
    return ret

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    """Post-processes the decoder outputs."""
    p = self.params
    # Update num_samples_in_batch.
    batch_size, num_classes, num_boxes, _ = (
        dec_out_dict.per_class_predicted_bboxes.shape)
    dec_metrics_dict.num_samples_in_batch.Update(batch_size)

    # Apply frustum mask to predicted box outputs.
    masked_pred_bbox_scores = (
        dec_out_dict.per_class_predicted_bbox_scores *
        dec_out_dict.frustum_mask)
    visualization_weights = (
        dec_out_dict.visualization_weights * dec_out_dict.frustum_mask)

    # TODO(bencaine): Add class base colors to camera image projection.
    if p.summarize_boxes_on_image:
      # Update the camera visualization.
      flattened_bbox_corners = np.reshape(dec_out_dict.bbox_corners_image,
                                          [batch_size, -1, 8, 2])
      flattened_visualization_weights = np.reshape(visualization_weights,
                                                   [batch_size, -1])
      dec_metrics_dict.camera_visualization.Update(
          py_utils.NestedMap({
              'camera_images': dec_out_dict.camera_images,
              # TODO(vrv): Use 2D output instead of 3D output.
              'bbox_corners': flattened_bbox_corners,
              'bbox_scores': flattened_visualization_weights,
          }))

    # Update decoder output by removing z-coordinate, thus reshaping the bboxes
    # to [batch, num_bboxes, 5] to be compatible with
    # TopDownVisualizationMetric.

    # Indices corresponding to the 2D bbox parameters (x, y, dx, dy, phi).
    bbox_2d_idx = np.asarray([1, 1, 0, 1, 1, 0, 1], dtype=np.bool)
    bboxes_2d = dec_out_dict.bboxes_3d[..., bbox_2d_idx]
    predicted_bboxes = dec_out_dict.per_class_predicted_bboxes[..., bbox_2d_idx]

    if dec_out_dict.points_sampled:
      tf.logging.info('Updating sample for top down visualization')
      dec_metrics_dict.mesh.Update(
          py_utils.NestedMap({
              'points_xyz': dec_out_dict.points_xyz,
              'points_padding': dec_out_dict.points_padding,
          }))

      # Flatten our predictions/scores to match the API of the visualization
      # The last dimension of flattened_bboxes is 5 due to the mask
      # above using bbox_2d_idx.
      flattened_bboxes = np.reshape(predicted_bboxes,
                                    [batch_size, num_classes * num_boxes, 5])
      flattened_visualization_weights = np.reshape(
          visualization_weights, [batch_size, num_classes * num_boxes])
      # Create a label id mask for now to maintain compatibility.
      # TODO(bencaine): Refactor visualizations to reflect new structure.
      flattened_visualization_labels = np.tile(
          np.arange(0, num_classes)[np.newaxis, :, np.newaxis],
          [batch_size, 1, num_boxes])
      flattened_visualization_labels = np.reshape(
          flattened_visualization_labels, [batch_size, num_classes * num_boxes])

      dec_metrics_dict.top_down_visualization.Update(
          py_utils.NestedMap({
              'visualization_labels': flattened_visualization_labels,
              'predicted_bboxes': flattened_bboxes,
              'visualization_weights': flattened_visualization_weights,
              'points_xyz': dec_out_dict.points_xyz,
              'points_padding': dec_out_dict.points_padding,
              'gt_bboxes_2d': bboxes_2d,
              'gt_bboxes_2d_weights': dec_out_dict.bboxes_3d_mask,
              'labels': dec_out_dict.labels,
              'difficulties': dec_out_dict.difficulties,
              'source_ids': dec_out_dict.source_ids,
          }))

    # Update KITTI AP metrics.
    for batch_idx in range(batch_size):
      # Use class scores since it's masked
      pred_bboxes = dec_out_dict.per_class_predicted_bboxes[batch_idx]
      pred_bbox_scores = masked_pred_bbox_scores[batch_idx]

      # Compute height in 2D perspective view, that will be used for filtering
      # outputs downstream.
      pred_bbox2d_corners_image = dec_out_dict.bbox2d_corners_image[batch_idx]
      # We assume y2 >= y1, so pred_heights_image will always be >= 0.
      pred_heights_image = (
          pred_bbox2d_corners_image[:, :, 3] -
          pred_bbox2d_corners_image[:, :, 1])

      gt_mask = dec_out_dict.unfiltered_bboxes_3d_mask[batch_idx].astype(bool)
      gt_labels = dec_out_dict.labels[batch_idx][gt_mask]
      gt_bboxes = dec_out_dict.bboxes_3d[batch_idx][gt_mask]
      gt_difficulties = dec_out_dict.difficulties[batch_idx][gt_mask]
      gt_num_points = dec_out_dict.num_points_in_bboxes[batch_idx][gt_mask]

      for metric_class in [dec_metrics_dict.kitti_AP_v2]:
        metric_class.Update(
            dec_out_dict.source_ids[batch_idx],
            py_utils.NestedMap(
                groundtruth_labels=gt_labels,
                groundtruth_bboxes=gt_bboxes,
                groundtruth_difficulties=gt_difficulties,
                groundtruth_num_points=gt_num_points,
                detection_scores=pred_bbox_scores,
                detection_boxes=pred_bboxes,
                detection_heights_in_pixels=pred_heights_image,
            ))

    # Returned values are saved in model_dir/decode. We can offline convert
    # them into KITTI's format.
    # velo_to_image_plane and velo_to_camera are needed for converting 3D bboxes
    # back to the camera coordinate as required by KITTI's format.
    #
    output_to_save = []
    for i in range(batch_size):
      gt_save_mask = dec_out_dict.unfiltered_bboxes_3d_mask[i].astype(bool)
      pd_save_mask = dec_out_dict.per_class_valid_mask[i] > 0
      # Create a class id matrix we can then mask out
      # since when you use the boolean mask in numpy it collapses the matrix
      # from [num_classes, num_bboxes,...] to just [num_valid_bbboxes].
      class_ids = np.tile(np.arange(num_classes)[:, np.newaxis], [1, num_boxes])

      saved_results = py_utils.NestedMap(
          img_id=dec_out_dict.source_ids[i],
          bboxes=dec_out_dict.per_class_predicted_bboxes[i][pd_save_mask],
          scores=masked_pred_bbox_scores[i][pd_save_mask],
          class_ids=class_ids[pd_save_mask],
          gt_labels=dec_out_dict.labels[i][gt_save_mask],
          gt_bboxes=dec_out_dict.bboxes_3d[i][gt_save_mask],
          gt_difficulties=dec_out_dict.difficulties[i][gt_save_mask],
          velo_to_image_plane=dec_out_dict.velo_to_image_plane[i],
          velo_to_camera=dec_out_dict.velo_to_camera[i],
          bboxes_2d=dec_out_dict.bbox2d_corners_image[i][pd_save_mask],
      )

      serialized = self.SaveTensors(saved_results)
      output_to_save += [(dec_out_dict.source_ids[i], serialized)]
    return output_to_save
