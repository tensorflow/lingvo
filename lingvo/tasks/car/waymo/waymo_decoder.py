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
"""Base models for point-cloud based detection."""

from lingvo import compat as tf
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.tasks.car import base_decoder
from lingvo.tasks.car import detection_3d_metrics
from lingvo.tasks.car import transform_util
from lingvo.tasks.car.waymo import waymo_ap_metric
from lingvo.tasks.car.waymo import waymo_metadata
import numpy as np


class WaymoOpenDatasetDecoder(base_decoder.BaseDecoder):
  """A decoder to use for decoding a detector model on Waymo."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'draw_visualizations', False, 'Boolean for whether to draw '
        'visualizations. This is independent of laser_sampling_rate.')
    p.ap_metric = waymo_ap_metric.WaymoAPMetrics.Params(
        waymo_metadata.WaymoMetadata())
    p.Define(
        'extra_ap_metrics', {},
        'Dictionary of extra AP metrics to run in the decoder. The key'
        'is the name of the metric and the value is a sub-class of '
        'APMetric')
    p.Define(
        'save_residuals', False,
        'If True, this expects the residuals and ground-truth to be available '
        'in the decoder output dictionary, and it will save it to the decoder '
        'output file. See decode_include_residuals in PointDetectorBase '
        'for details.')
    return p

  def CreateDecoderMetrics(self):
    """Decoder metrics for WaymoOpenDataset."""
    p = self.params

    waymo_metric_p = p.ap_metric.Copy().Set(cls=waymo_ap_metric.WaymoAPMetrics)
    waymo_metrics = waymo_metric_p.Instantiate()
    class_names = waymo_metrics.metadata.ClassNames()

    # TODO(bencaine,vrv): There's some code smell with this ap_metrics params
    # usage. We create local copies of the params to then instantiate them.
    # Failing to do this risks users editing the params after construction of
    # the object, making each object method call have the potential for side
    # effects.
    # Create a new dictionary with copies of the params converted to objects
    # so we can then add these to the decoder metrics.
    extra_ap_metrics = {}
    for k, metric_p in p.extra_ap_metrics.items():
      extra_ap_metrics[k] = metric_p.Instantiate()

    waymo_metric_bev_p = waymo_metric_p.Copy()
    waymo_metric_bev_p.box_type = '2d'
    waymo_metrics_bev = waymo_metric_bev_p.Instantiate()
    # Convert the list of class names to a dictionary mapping class_id -> name.
    class_id_to_name = dict(enumerate(class_names))

    # TODO(vrv): This uses the same top down transform as for KITTI;
    # re-visit these settings since detections can happen all around
    # the car.
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
        'waymo_metrics': waymo_metrics,
        'waymo_metrics_bev': waymo_metrics_bev,
    })
    self._update_metrics_class_keys = ['waymo_metrics_bev', 'waymo_metrics']
    for k, metric in extra_ap_metrics.items():
      decoder_metrics[k] = metric
      self._update_metrics_class_keys.append(k)

    decoder_metrics.mesh = detection_3d_metrics.WorldViewer()
    return decoder_metrics

  def ProcessOutputs(self, input_batch, model_outputs):
    """Produce additional decoder outputs for WaymoOpenDataset.

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
    del model_outputs
    p = self.params
    input_labels = input_batch.labels
    input_metadata = input_batch.metadata
    source_ids = tf.strings.join([
        input_metadata.run_segment,
        tf.as_string(input_metadata.run_start_offset)
    ],
                                 separator='_')
    ret = py_utils.NestedMap({
        'num_points_in_bboxes': input_batch.labels.bboxes_3d_num_points,
        # Ground truth.
        'bboxes_3d': input_labels.bboxes_3d,
        'bboxes_3d_mask': input_labels.bboxes_3d_mask,
        'labels': input_labels.labels,
        'label_ids': input_labels.label_ids,
        'speed': input_labels.speed,
        'acceleration': input_labels.acceleration,
        # Fill the following in.
        'source_ids': source_ids,
        'difficulties': input_labels.single_frame_detection_difficulties,
        'unfiltered_bboxes_3d_mask': input_labels.unfiltered_bboxes_3d_mask,
        'run_segment': input_metadata.run_segment,
        'run_start_offset': input_metadata.run_start_offset,
        'pose': input_metadata.pose,
    })
    if p.draw_visualizations:
      laser_sample = self._SampleLaserForVisualization(
          input_batch.lasers.points_xyz, input_batch.lasers.points_padding)
      ret.update(laser_sample)
    return ret

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    """Post-processes the decoder outputs."""
    p = self.params
    # Update num_samples_in_batch.
    batch_size, num_classes, num_boxes, _ = (
        dec_out_dict.per_class_predicted_bboxes.shape)
    dec_metrics_dict.num_samples_in_batch.Update(batch_size)

    # Update decoder output by removing z-coordinate, thus reshaping the bboxes
    # to [batch, num_bboxes, 5] to be compatible with
    # TopDownVisualizationMetric.

    # Indices corresponding to the 2D bbox parameters (x, y, dx, dy, phi).
    bbox_2d_idx = np.asarray([1, 1, 0, 1, 1, 0, 1], dtype=np.bool)
    bboxes_2d = dec_out_dict.bboxes_3d[..., bbox_2d_idx]
    predicted_bboxes = dec_out_dict.per_class_predicted_bboxes[..., bbox_2d_idx]

    if p.draw_visualizations and dec_out_dict.points_sampled:
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
          dec_out_dict.visualization_weights,
          [batch_size, num_classes * num_boxes])

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

    # Update AP metrics.

    # Skip zeroth step decoding.
    if dec_out_dict.global_step == 0:
      return None

    # TODO(bencaine/vrv): Refactor to unify Waymo code and KITTI

    # Returned values are saved in model_dir/decode_* directories.
    output_to_save = []

    for batch_idx in range(batch_size):
      pred_bboxes = dec_out_dict.per_class_predicted_bboxes[batch_idx]
      pred_bbox_scores = dec_out_dict.per_class_predicted_bbox_scores[batch_idx]

      # The current API expects a 'height' matrix to be passed for filtering
      # detections based on height.  This is a KITTI-ism that we need to remove,
      # but for now we just give a height of 1.  The MinHeight metadata function
      # for non-KITTI datasets should have a threshold lower than this value.
      heights = np.ones((num_classes, num_boxes)).astype(np.float32)

      gt_mask = dec_out_dict.bboxes_3d_mask[batch_idx].astype(bool)
      gt_labels = dec_out_dict.labels[batch_idx][gt_mask]
      gt_bboxes = dec_out_dict.bboxes_3d[batch_idx][gt_mask]
      gt_difficulties = dec_out_dict.difficulties[batch_idx][gt_mask]
      gt_num_points = dec_out_dict.num_points_in_bboxes[batch_idx][gt_mask]
      # Note that this is not used in the KITTI evaluation.
      gt_speed = dec_out_dict.speed[batch_idx][gt_mask]

      # TODO(shlens): Update me
      for metric_key in self._update_metrics_class_keys:
        metric_cls = dec_metrics_dict[metric_key]
        metric_cls.Update(
            dec_out_dict.source_ids[batch_idx],
            py_utils.NestedMap(
                groundtruth_labels=gt_labels,
                groundtruth_bboxes=gt_bboxes,
                groundtruth_difficulties=gt_difficulties,
                groundtruth_num_points=gt_num_points,
                groundtruth_speed=gt_speed,
                detection_scores=pred_bbox_scores,
                detection_boxes=pred_bboxes,
                detection_heights_in_pixels=heights,
            ))

      # We still want to save all ground truth (even if it was filtered
      # in some way) so we use the unfiltered_bboxes_3d_mask here.
      gt_save_mask = dec_out_dict.unfiltered_bboxes_3d_mask[batch_idx].astype(
          bool)
      pd_save_mask = dec_out_dict.per_class_valid_mask[batch_idx] > 0
      class_ids = np.tile(np.arange(num_classes)[:, np.newaxis], [1, num_boxes])

      saved_results = py_utils.NestedMap(
          pose=dec_out_dict.pose[batch_idx],
          frame_id=dec_out_dict.source_ids[batch_idx],
          bboxes=pred_bboxes[pd_save_mask],
          scores=pred_bbox_scores[pd_save_mask],
          gt_labels=dec_out_dict.labels[batch_idx][gt_save_mask],
          gt_label_ids=dec_out_dict.label_ids[batch_idx][gt_save_mask],
          gt_speed=dec_out_dict.speed[batch_idx][gt_save_mask],
          gt_acceleration=dec_out_dict.acceleration[batch_idx][gt_save_mask],
          class_ids=class_ids[pd_save_mask],
          gt_bboxes=dec_out_dict.bboxes_3d[batch_idx][gt_save_mask],
          gt_difficulties=dec_out_dict.difficulties[batch_idx][gt_save_mask],
      )

      if p.save_residuals:
        # The leading shapes of these tensors should match bboxes and scores.
        # These are the underlying tensors that can are used to compute score
        # and bboxes.
        saved_results.update({
            'bboxes_gt_residuals':
                dec_out_dict.per_class_gt_residuals[batch_idx][pd_save_mask],
            'bboxes_gt_labels':
                dec_out_dict.per_class_gt_labels[batch_idx][pd_save_mask],
            'bboxes_residuals':
                dec_out_dict.per_class_residuals[batch_idx][pd_save_mask],
            'bboxes_logits':
                dec_out_dict.per_class_logits[batch_idx][pd_save_mask],
            'bboxes_anchor_boxes':
                dec_out_dict.per_class_anchor_boxes[batch_idx][pd_save_mask],
        })

      serialized = self.SaveTensors(saved_results)
      output_to_save += [(dec_out_dict.source_ids[batch_idx], serialized)]
    return output_to_save
