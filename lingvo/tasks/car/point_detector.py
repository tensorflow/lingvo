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
from lingvo.core import base_model
from lingvo.core import py_utils
from lingvo.tasks.car import detection_3d_lib
from lingvo.tasks.car import detection_decoder

from lingvo.tasks.car import kitti_decoder
import numpy as np


class PointDetectorBase(base_model.BaseTask):
  """Base class for implementing point-based detectors.

  Subclasses should implement _BBoxesAndLogits() to compute the bounding box and
  scores given an input batch, and specify an appropriate decoder
  implementation.
  """

  @classmethod
  def Params(cls, num_classes):
    p = super().Params()
    # We choose a high number of boxes per example by default to bound overall
    # runtime but not so low that we end up missing real boxes from complicated
    # scenes.
    p.Define('num_classes', num_classes,
             'The number of classes, including the background class.')
    p.Define(
        'max_nms_boxes', 1024,
        'Maximum number of boxes per example to emit from non-max-suppression.')
    p.Define(
        'nms_iou_threshold', 0.3,
        'NMS IoU threshold for suppressing overlapping boxes. '
        'Can either be a float or a list of len num_classes.')
    p.Define(
        'nms_score_threshold', 0.01, 'NMS threshold for scores. '
        'Can either be a float or a list of len num_classes. '
        'It is recommended that this be 1 for all non-active classes '
        'like background.')
    p.Define(
        'visualization_classification_threshold', 0.25,
        'Classification score threshold for determining if a prediction '
        'is positive for the purposes of visualizations.')
    p.Define('output_decoder', kitti_decoder.KITTIDecoder.Params(),
             'Implementation of decoder.')
    p.Define(
        'use_oriented_per_class_nms', False,
        'Whether to use oriented per class nms or single class non-oriented.')
    p.Define(
        'inference_batch_size', None,
        'If specified, hardcodes the inference batch size to this value. '
        'Useful mostly for computing the FLOPS of a model so that the shape is '
        'fully defined.')
    p.Define(
        'decode_include_residuals', False,
        'If True, includes the residuals and ground truth anchors in the '
        'decoder output dictionary. This can be helpful for downstream '
        'analysis.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._utils_3d = detection_3d_lib.Utils3D()
    self.CreateChild('output_decoder', p.output_decoder)

  def CreateDecoderMetrics(self):
    """Create decoder metrics."""
    return self.output_decoder.CreateDecoderMetrics()

  def _BBoxesAndLogits(self, input_batch, predictions):
    """Fetch and return the bounding boxes and logits from an input.

    Args:
      input_batch: The input batch from which to produce boxes and logits.
      predictions: The output dictionary of ComputePredictions.

    Returns:
      A .NestedMap containing

      - predicted_bboxes: A [batch_size, num_boxes, 7] floating point Tensor.
      - classification_logits: A [batch_size, num_boxes, num_classes] floating
        point Tensor.
    """
    raise NotImplementedError('_BoxesAndLogits method not implemented.')

  def _Placeholders(self):
    """Return a NestedMap of placeholders to fill in for inference.

    Runs the configured input pipeline to generate the expected shapes and types
    of the inputs.

    Returns:
      A NestedMap of placeholders matching the input structure of
       the inference model.
    """
    p = self.params
    with tf.Graph().as_default():
      inputs = self.params.input.Instantiate()

    # Turn those inputs into placeholders.
    placeholders = []
    for input_shape, dtype in zip(inputs.Shape().Flatten(),
                                  inputs.DType().Flatten()):
      batched_input_shape = [p.inference_batch_size] + input_shape.as_list()
      placeholders.append(tf.placeholder(dtype, batched_input_shape))

    result = inputs.DType().Pack(placeholders)
    return result

  def _BBoxDimensionErrors(self,
                           gt_bboxes,
                           pred_bboxes,
                           regression_weights,
                           epsilon=1e-6):
    """Calculates the errors per bounding box dimension for assigned anchors.

    Args:
      gt_bboxes: float Tensor of shape [..., 7] with the ground truth bounding
        box for each anchor.
      pred_bboxes: float Tensor of shape [..., 7] with the predicted bounding
        box for each anchor.
      regression_weights: float Tensor with 0/1 weights indicating whether the
        anchor had a positive assignment with same base shape as `gt_bboxes` and
        `pred_bboxes` excluding the last dimension.
      epsilon: A float epsilon for the denominiator of our MaskedAverage.

    Returns:
      A metrics dict with mean bounding box errors for all positive assigned
      anchor locations.
    """
    if py_utils.GetShape(gt_bboxes)[-1] != 7:
      raise ValueError('`gt_bboxes` last dimension should be 7.')
    if py_utils.GetShape(pred_bboxes)[-1] != 7:
      raise ValueError('`pred_bboxes` last dimension should be 7.')
    batch_size = py_utils.GetShape(pred_bboxes)[0]
    # Get the leading dims for later (the -1 is to exclude the last dim).
    leading_dims = len(py_utils.GetShape(pred_bboxes)) - 1

    sum_regression_weights = tf.reduce_sum(regression_weights) + epsilon

    def _MaskedAverage(value, axis=None):
      return (tf.reduce_sum(value * regression_weights, axis=axis) /
              sum_regression_weights)

    center_error = tf.linalg.norm(
        gt_bboxes[..., :3] - pred_bboxes[..., :3], axis=-1, keepdims=True)
    mean_center_error = _MaskedAverage(center_error)

    # Dimension error as shape [3] so we can get separate height, width, length
    mean_dimension_error = _MaskedAverage(
        gt_bboxes[..., 3:6] - pred_bboxes[..., 3:6],
        axis=list(range(leading_dims)))

    # Angular error in degrees
    mean_angular_error_rad = _MaskedAverage(gt_bboxes[..., 6:] -
                                            pred_bboxes[..., 6:])
    mean_angular_error_deg = mean_angular_error_rad * (180 / np.pi)

    return {
        'error/center_distance': (mean_center_error, batch_size),
        'error/length': (mean_dimension_error[0], batch_size),
        'error/width': (mean_dimension_error[1], batch_size),
        'error/height': (mean_dimension_error[2], batch_size),
        'error/rotation_deg': (mean_angular_error_deg, batch_size),
    }

  def Inference(self):
    """Builds the inference graph.

    Default subgraph should return:

      predicted_bboxes: A [batch_size, num_boxes, 7] float Tensor.

      classification_scores: A [batch_size, num_boxes, num_classes] float
      Tensor.

    Returns:
      A dictionary whose values are a tuple of fetches and feeds.
    """
    p = self.params
    subgraphs = {}
    with tf.name_scope('inference'):
      input_placeholders = self._Placeholders()
      predictions = self.ComputePredictions(self.theta, input_placeholders)
      bboxes_and_logits = self._BBoxesAndLogits(input_placeholders, predictions)
      predicted_bboxes = bboxes_and_logits.predicted_bboxes
      classification_logits = bboxes_and_logits.classification_logits
      classification_scores = tf.sigmoid(classification_logits)

      _, per_cls_bboxes, per_cls_bbox_scores, per_cls_valid_mask = (
          detection_decoder.DecodeWithNMS(
              predicted_bboxes,
              classification_scores,
              nms_iou_threshold=p.nms_iou_threshold,
              score_threshold=p.nms_score_threshold,
              max_boxes_per_class=p.max_nms_boxes,
              use_oriented_per_class_nms=p.use_oriented_per_class_nms))
      per_cls_bbox_scores *= per_cls_valid_mask

      # TODO(vrv): Fix the inference graph for KITTI, since we need
      # to apply frustum clipping.  This requires customizing the
      # inference placeholders for each model.
      fetches = {
          'per_class_predicted_bboxes': per_cls_bboxes,
          'per_class_predicted_bbox_scores': per_cls_bbox_scores,
          'per_class_valid_mask': per_cls_valid_mask
      }
      subgraphs['default'] = fetches, dict(input_placeholders.FlattenItems())
    return subgraphs

  # TODO(bencaine): Reduce code duplication between Inference/Decode.
  def Decode(self, input_batch):
    """Decode an input batch, computing predicted bboxes from residuals."""
    p = self.params

    predictions = self.ComputePredictions(self.theta, input_batch)
    bboxes_and_logits = self._BBoxesAndLogits(input_batch, predictions)
    predicted_bboxes = bboxes_and_logits.predicted_bboxes
    batch_size, num_bboxes, _ = py_utils.GetShape(predicted_bboxes, 3)
    classification_logits = bboxes_and_logits.classification_logits
    classification_logits = py_utils.HasShape(
        classification_logits, [batch_size, num_bboxes, p.num_classes])

    classification_scores = tf.sigmoid(classification_logits)

    _, per_example_dict = self.ComputeLoss(self.theta, predictions, input_batch)
    if 'score_scaler' in per_example_dict:
      classification_scores *= per_example_dict['score_scaler']

    with tf.device('/cpu:0'):
      # Decode the predicted bboxes, performing NMS.
      per_cls_idxs, per_cls_bboxes, per_cls_bbox_scores, per_cls_valid_mask = (
          detection_decoder.DecodeWithNMS(
              predicted_bboxes,
              classification_scores,
              nms_iou_threshold=p.nms_iou_threshold,
              score_threshold=p.nms_score_threshold,
              max_boxes_per_class=p.max_nms_boxes,
              use_oriented_per_class_nms=p.use_oriented_per_class_nms))

      # per_cls_valid_mask is [batch, num_classes, num_boxes] Tensor that
      # indicates which boxes were selected by NMS. Each example will have a
      # different number of chosen bboxes, so the mask is present to allow us
      # to keep the boxes as a batched dense Tensor.
      #
      # We mask the scores by the per_cls_valid_mask so that none of these boxes
      # will be interpreted as valid.
      per_cls_bbox_scores *= per_cls_valid_mask
      visualization_weights = py_utils.HasShape(
          per_cls_bbox_scores, [batch_size, p.num_classes, p.max_nms_boxes])

      # For top down visualization, filter boxes whose scores are not above the
      # visualization threshold.
      visualization_weights = tf.where(
          tf.greater_equal(visualization_weights,
                           p.visualization_classification_threshold),
          visualization_weights, tf.zeros_like(visualization_weights))

    model_outputs = py_utils.NestedMap()
    model_outputs.per_class_predicted_bboxes = per_cls_bboxes
    model_outputs.per_class_predicted_bbox_scores = per_cls_bbox_scores
    model_outputs.per_class_valid_mask = per_cls_valid_mask

    decoder_outputs = py_utils.NestedMap({
        'per_class_predicted_bboxes': per_cls_bboxes,
        'per_class_predicted_bbox_scores': per_cls_bbox_scores,
        'per_class_valid_mask': per_cls_valid_mask,
        'visualization_weights': visualization_weights,
    })

    if p.decode_include_residuals:
      # Including the residuals in the decoder output makes it possible to save
      # the outputs for further analysis. Note that we ensure that the outputs
      # match the per-class NMS output format of [batch, num_classes, ...].
      def _ReshapeGather(tensor):
        """Reshapes tensor and then gathers using the nms indices."""
        tensor = tf.gather(
            tf.reshape(tensor, [batch_size, num_bboxes, -1]),
            per_cls_idxs,
            batch_dims=1)
        if not p.use_oriented_per_class_nms:
          # Tile so that the data fits the expected per class shape of
          # [batch_size, num_classes, ...]. When *not* using oriented NMS, the
          # num_classes dimension will be missing since the indices will not
          # have it.
          tensor = tf.tile(tensor[:, tf.newaxis, :, :],
                           [1, p.num_classes, 1, 1])
        return tensor

      decoder_outputs.update({
          'per_class_gt_residuals':
              _ReshapeGather(input_batch.anchor_localization_residuals),
          'per_class_gt_labels':
              _ReshapeGather(input_batch.assigned_gt_labels),
          'per_class_residuals':
              _ReshapeGather(predictions.residuals),
          'per_class_logits':
              _ReshapeGather(predictions.classification_logits),
          'per_class_anchor_boxes':
              _ReshapeGather(input_batch.anchor_bboxes),
      })

    decoder_outputs.update(
        self.output_decoder.ProcessOutputs(input_batch, model_outputs))

    # Produce global step as an output (which is the step
    # of the checkpoint being decoded.)
    decoder_outputs.global_step = py_utils.GetGlobalStep()

    return decoder_outputs

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    return self.output_decoder.PostProcessDecodeOut(dec_out_dict,
                                                    dec_metrics_dict)

  def DecodeFinalize(self, decode_finalize_args):
    decode_out_path = decode_finalize_args.decode_out_path
    decode_out = decode_finalize_args.decode_out

    if not decode_out:
      return

    # Write out a tf record file for all values in decode_out.
    with tf.io.TFRecordWriter(decode_out_path) as f:
      for _, v in decode_out:
        f.write(v)
