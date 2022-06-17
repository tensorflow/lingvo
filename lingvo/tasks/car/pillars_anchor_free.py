# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""AnchorFree Pillars models."""

import enum
import functools

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.tasks.car import detection_3d_lib
from lingvo.tasks.car import detection_decoder
from lingvo.tasks.car import pillars
from lingvo.tasks.car import point_detector
import tensorflow_probability as tfp


class NMSDecoderType(enum.Enum):
  NMS_DECODER = 0
  HEATMAP_NMS_DECODER = 1
  NO_NMS_DECODER = 2


class ClassLossFN(enum.Enum):
  SIGMOID_LOSS = 0
  FOCAL_SIGMOID_LOSS = 1


def HeatMapNMS(heat_map_scores, kernel_size):
  """Extract top k peaks of heat map from heat_map_scores.

  Args:
    heat_map_scores: A [batch, gx, gy, 1] float Tensor with values ranging from
      (0, 1) indicating the likelihood of that object being a center.
    kernel_size: A list of integers specifying the max pooling kernel size to
      use on the heat_map_scores input.

  Returns:
    peak_heat_map: A [batch_size, gx, gy, 1] float32 Tensor with values
      range [0, 1]. The values of suppressed pillars are 0.
  """
  # Keep the peaks using maxpooling / masking.
  max_scores = tf.nn.max_pool(
      heat_map_scores, ksize=kernel_size, strides=[1, 1, 1, 1], padding='SAME')
  # Creates a mask only returning the heat map pixels that were the
  # max of their pooled neighborhood.
  peak_heat_map = tf.cast(tf.equal(heat_map_scores, max_scores), tf.float32)
  peak_heat_map = heat_map_scores * peak_heat_map

  return peak_heat_map


################################################################################
# AnchorFREE model.
################################################################################
class _LossInterface(base_layer.BaseLayer):
  """Interface for uncertainty losses.

  Sub-classes are expected to specify num_params_per_prediction, which
  corresponds to the number of predicted value per prediction. For example,
  if a loss requires predicting both the mean and variance, then there'll be
  two params per prediction.

  This value is helpful so that the parent model builder can use it to build
  an output layer of the appropriate shape.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'num_params_per_prediction', None,
        'Number of parameters per prediction. Sub-classes should specify '
        'this, and it should be treated as a property of the class.')
    return p

  def MeanPrediction(self, theta, prediction_tensors):
    """Returns the mean prediction."""
    raise NotImplementedError()

  def FProp(self, theta, prediction_tensors, labels):
    """Computes the loss between the predictions and labels."""
    raise NotImplementedError()


class HuberLoss(_LossInterface):
  """Compute huber loss."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.num_params_per_prediction = 1

    p.Define('delta', 1 / (3.**2), 'The delta for smooth-L1 loss.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self._utils = detection_3d_lib.Utils3D()

  def MeanPrediction(self, theta, prediction_tensors):
    """Returns the mean prediction."""
    return prediction_tensors

  def FProp(self, theta, prediction_tensors, labels, transform_fn=None):
    """Computes the loss between the predictions and labels."""
    p = self.params
    predictions = self.MeanPrediction(theta, prediction_tensors)

    if transform_fn is not None:
      predictions, labels = transform_fn(predictions, labels)

    loss = self._utils.ScaledHuberLoss(
        predictions=predictions, labels=labels, delta=p.delta)
    return loss


class LaplaceKL(_LossInterface):
  """Compute the KL-divergence loss."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.num_params_per_prediction = 2
    p.Define(
        'targets_scale', 0.003, 'A float tensor representing the scale '
        'parameter for the target distribution. This will be used if no '
        'targets_scale is explicitly passed to FProp.')
    p.Define(
        'epsilon', 1e-3, 'Epsilon for numerical stability. Also the '
        'minimum scale for the predictions distribution.')
    return p

  def _SplitPredictionParams(self, prediction_tensors):
    """Split prediction_tensors into predictions and pre_scale."""
    predictions, pre_scale = tf.split(prediction_tensors, 2, axis=-1)
    return predictions, pre_scale

  def MeanPrediction(self, theta, prediction_tensors):
    """Returns the mean prediction logit."""
    predictions, _ = self._SplitPredictionParams(prediction_tensors)
    return predictions

  def Scale(self, theta, prediction_tensors):
    p = self.params
    _, pre_scale = self._SplitPredictionParams(prediction_tensors)
    scale = tf.nn.softplus(pre_scale) + p.epsilon
    return scale

  def FProp(self,
            theta,
            prediction_tensors,
            labels,
            targets_scale=None,
            transform_fn=None):
    """Computes the loss between the predictions and labels."""
    p = self.params
    if targets_scale is None:
      targets_scale = p.targets_scale

    predictions, pre_scale = self._SplitPredictionParams(prediction_tensors)
    scale = tf.nn.softplus(pre_scale) + p.epsilon
    if transform_fn is not None:
      predictions, labels = transform_fn(predictions, labels)
    predicted_dist = tfp.distributions.Laplace(predictions, scale)
    labels_dist = tfp.distributions.Laplace(labels, targets_scale)
    return labels_dist.kl_divergence(predicted_dist)


class AnchorFreePillarsBase(point_detector.PointDetectorBase):
  """AnchorFree PointPillars model."""

  NUM_OUTPUT_CHANNELS = 128

  @classmethod
  def Params(cls,
             grid_size_z=1,
             num_classes=1,
             num_laser_features=3,
             angle_bin_num=12):
    p = super().Params(num_classes=num_classes)
    p.Define('grid_size_z', grid_size_z, 'The grid size along the z-axis.')
    p.Define('num_laser_features', num_laser_features,
             'The number of (non-xyz) laser features of the input.')
    p.Define('angle_bin_num', angle_bin_num,
             'The number of bin for angle prediction.')
    p.Define('input_featurizer',
             pillars.PointsToGridFeaturizer.Params(num_laser_features),
             'Point cloud feature extractor.')

    builder = pillars.Builder()
    p.Define('backbone', builder.Backbone(cls.NUM_OUTPUT_CHANNELS),
             'Dense features pyramid.')
    # Backbone() concatenates 3 different scales of features.
    idims = 3 * cls.NUM_OUTPUT_CHANNELS
    class_odims = grid_size_z * num_classes
    # A regression head is composed of 3 parts, residual prediction (3)
    # dimension prediction (3) and angle prediction (angle_bin_num * 2)
    reg_odims = grid_size_z * (6 + angle_bin_num + angle_bin_num)

    p.Define('class_detector', builder.Detector('class', idims, class_odims),
             'Dense features to class logits.')

    p.Define('centerness_detector',
             builder.Detector('centerness', idims, class_odims),
             'Dense features to centerness logits.')

    p.Define('regression_detector', builder.Detector('reg', idims, reg_odims),
             'Dense features to regression logits.')

    p.Define('classification_loss_fn', ClassLossFN.FOCAL_SIGMOID_LOSS,
             'The classification loss function.')
    p.Define('focal_loss_alpha', 0.25, 'The alpha parameter in focal loss '
             '(see paper eq. 4).')
    p.Define('focal_loss_gamma', 2.0, 'The gamma parameter in focal loss '
             '(see paper eq. 4).')

    p.Define('location_loss', HuberLoss.Params(),
             'Regression loss params.')
    p.Define('dimensions_loss', HuberLoss.Params(),
             'Dimensions loss params.')

    p.Define(
        'localization_loss_weight', 2.0,
        'Localization loss weight factor between localization and '
        'class loss contributions.')
    p.Define(
        'classification_loss_weight', 1.0,
        'Classification loss weight factor between localization and '
        'class loss contributions.')
    p.Define('centerness_loss_weight', 0.0, 'Centerness loss weight factor.')

    p.Define(
        'location_loss_weight', 1.0,
        'Weight multiplier for contribution of location loss '
        'to full localization/regression loss')
    p.Define(
        'dimensions_loss_weight', None,
        'Weight multiplier for contribution of dimensions loss '
        'to full localization/regression loss. If it is None,'
        'the weight equals to location_loss_weight')
    p.Define(
        'rotation_loss_weight', 1.0,
        'Weight multiplier for contribution of rotation loss '
        'to full localization/regression loss')
    p.Define(
        'corner_loss_weight', 1.0,
        'Weight multiplier for contribution of corner loss'
        'to full localization/regression loss')

    p.Define('loss_norm_type',
             pillars.LossNormType.NORM_BY_NUM_POSITIVES,
             'Normalization function for class and regularization weights.')

    p.Define(
        'nms_decoder_type', NMSDecoderType.NMS_DECODER,
        'The decoder type for post-processing: 1. Original NMS decoder. '
        '2. HeatMapNMS decoder provided by centernet - '
        'https://arxiv.org/pdf/1904.07850.pdf')
    p.Define('heatmap_nms_kernel_size', [1, 3, 3, 1],
             'The kernel size of MaxPooling used in HeatMapNMS decoder.')

    p.Define(
        'per_class_loss_weight', [0.] + [1.] * (num_classes - 1),
        'A list with a float value per class with a multiple to multiply '
        'the classification losses of that classes anchors by. Note that '
        'the background class is always class 0, and should be assigned a '
        'weight of 0.')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._utils = detection_3d_lib.Utils3D()

    if p.dimensions_loss_weight is None:
      p.dimensions_loss_weight = p.location_loss_weight
    if len(p.per_class_loss_weight) != p.num_classes:
      raise ValueError('`Need `per_class_loss_weight` to be of len equal '
                       'to the number of classes.')
    if p.per_class_loss_weight[0] != 0.0:
      raise ValueError('Background class should be assigned 0 weight. '
                       'per_class_loss_weight={}'.format(
                           str(p.per_class_loss_weight)))
    self.CreateChild('location_loss', p.location_loss)
    self.CreateChild('dimensions_loss', p.dimensions_loss)

    self.CreateChild('input_featurizer', p.input_featurizer)
    self.CreateChild('backbone', p.backbone)
    self.CreateChild('class_detector', p.class_detector)
    self.CreateChild('regression_detector', p.regression_detector)
    if p.centerness_loss_weight > 0:
      self.CreateChild('centerness_detector', p.centerness_detector)

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions for `input_batch`.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      A `.NestedMap` contains
        logits - [b, -1, 7 + num_classes]
        (optional) centerness_logits - [b, -1, num_classes]. Return when
          p.centerness_loss_weight > 0.
    """
    p = self.params
    input_batch.Transform(lambda x: (x.shape, x.shape.num_elements())).VLog(
        0, 'input_batch shapes: ')

    # Make pillars representation from input_batch.
    dense_features = self.input_featurizer.FProp(theta.input_featurizer,
                                                 input_batch)

    # Backbone
    tf.logging.vlog(1, 'dense_features.shape = %s', dense_features.shape)
    act = self.backbone.FProp(theta.backbone, dense_features)
    tf.logging.vlog(1, 'act.shape = %s', act.shape)

    # Convert the output of the backbone into class logits and regression
    # residuals using two different layers.
    class_detection = self.class_detector.FProp(theta.class_detector, act)
    reg_detection = self.regression_detector.FProp(theta.regression_detector,
                                                   act)
    bs, nx, ny, _ = py_utils.GetShape(class_detection, 4)

    predicted_classification_logits = tf.reshape(
        class_detection, [bs, nx * ny * p.grid_size_z, p.num_classes])
    num_residual_dims = (3 * p.location_loss.num_params_per_prediction +
                         3 * p.dimensions_loss.num_params_per_prediction +
                         p.angle_bin_num * 2)
    predicted_residuals = tf.reshape(
        reg_detection, [bs, nx * ny * p.grid_size_z, num_residual_dims])

    points = py_utils.HasShape(input_batch.anchor_centers,
                               [bs, nx, ny, p.grid_size_z, 3])
    points = tf.reshape(points, [bs, -1, 3])

    ret = py_utils.NestedMap({
        'points': points,
        'base_feature': act,
        'residuals': predicted_residuals,
        'classification_logits': predicted_classification_logits,
    })

    if p.centerness_loss_weight > 0:
      predicted_centerness_logits = self.centerness_detector.FProp(
          theta.centerness_detector, act)
      predicted_centerness_logits = tf.reshape(
          predicted_centerness_logits,
          [bs, nx * ny * p.grid_size_z, p.num_classes])
      ret.update({
          'centerness_logits': predicted_centerness_logits,
      })

    return ret

  def _ComputeClassificationLoss(self, predicted_class_logits,
                                 assigned_gt_labels, class_weights, loss_fn):
    """Compute classification loss for the given predictions.

    It is a wrapper function for applying classification loss on different type
    of loss_fn.

    Args:
      predicted_class_logits: A float32 tensor with shape [bs, -1, num_classes],
        indicating the classification confidence predictions.
      assigned_gt_labels: A int32 tensor with shape [bs, -1, num_classes],
        indicating the assigned label of each pillar within the point cloud.
      class_weights: Per-class weights to use in loss computation.
      loss_fn: The function for calculating loss. FocalLoss or
        SigmoidCrossEntropy.

    Returns:
      Classification loss. A float value. The total classification loss for
        all pillars within the point cloud.

    """
    p = self.params

    predicted_class_logits = py_utils.HasShape(predicted_class_logits,
                                               [-1, -1, p.num_classes])
    bs, npillars, ncls = py_utils.GetShape(predicted_class_logits, 3)
    class_weights = py_utils.HasShape(class_weights, [bs, npillars, ncls])
    assigned_gt_labels = py_utils.HasShape(assigned_gt_labels,
                                           [bs, npillars, ncls])

    class_loss = loss_fn(
        logits=predicted_class_logits, labels=assigned_gt_labels)
    class_loss *= class_weights
    class_loss_sum = tf.reduce_sum(class_loss)

    return class_loss_sum

  def _ComputeRegressionLoss(self, points, predicted_residuals,
                             target_predictions, reg_weights):
    """Compute the anchor-free regression loss.

    The regression loss is composed of center-residual loss,
      dimension residual loss, angle-bin-loss and corner loss.

    Args:
      points: A float32 tensor with shape [bs, -1, 3], indicating the
        coordinates of points.
      predicted_residuals: A float32 tensor with shape [bs, -1, 3 + 3 +
        p.angle_bin_num * 2], indicating the location residual, dimension and
        angle predictions.
      target_predictions: A float32 tensor with shape [bs, -1, 7], containing
        the target predictions after encoding method.
      reg_weights: Per-point weights to use in loss computation.

    Returns:
      A `.NestedMap` object:
       - reg_loc_and_dims_loss_sum: A float value indicating the total location
           and dimension losses for all valid pillars within the point cloud.
       - angle_loss_sum: A float value indicating the orientation losses for all
           pillars.
       - reg_corner_loss_sum: A float value.
       - decoded_bboxes: A dictionary containing the assigned_gt_box,
           decoded_gt_bbox for all pillars.
    """
    p = self.params
    num_residual_dims = (3 * p.location_loss.num_params_per_prediction +
                         3 * p.dimensions_loss.num_params_per_prediction +
                         p.angle_bin_num * 2)
    predicted_residuals = py_utils.HasShape(predicted_residuals,
                                            [-1, -1, num_residual_dims])

    # predicted_loc_residuals_scale and predicted_dim_residuals_scale have
    # a shape of [..., 0] when p.location_loss.num_params_per_prediction == 1
    # and p.dimensions_loss.num_params_per_prediction == 1. When they have a
    # shape of [..., 0], there is no effect when concatted in the next step.
    (predicted_loc_dim_residuals, predicted_angle_bin, predicted_angle_res,
     predicted_loc_residuals_scale, predicted_dim_residuals_scale) = (
         tf.split(
             predicted_residuals, [
                 6, p.angle_bin_num, p.angle_bin_num, 3 *
                 (p.location_loss.num_params_per_prediction - 1), 3 *
                 (p.dimensions_loss.num_params_per_prediction - 1)
             ],
             axis=-1))
    predicted_loc_residuals, predicted_dim_residuals = (
        tf.split(predicted_loc_dim_residuals, [3, 3], axis=-1))
    predicted_loc_tensors = tf.concat(
        [predicted_loc_residuals, predicted_loc_residuals_scale], axis=-1)
    predicted_dim_tensors = tf.concat(
        [predicted_dim_residuals, predicted_dim_residuals_scale], axis=-1)

    bs, npillars = py_utils.GetShape(predicted_residuals, 2)

    reg_weights = py_utils.HasShape(reg_weights, [bs, npillars, 1])

    # --- Localization loss ---
    target_predictions = py_utils.HasShape(target_predictions,
                                           [bs, npillars, 7])

    # Location and dimensions loss.

    reg_loc_loss = self.location_loss(predicted_loc_tensors,
                                      target_predictions[..., :3])
    reg_loc_loss *= reg_weights
    reg_loc_loss_sum = tf.reduce_sum(reg_loc_loss)

    reg_dim_loss = self.dimensions_loss(predicted_dim_tensors,
                                        target_predictions[..., 3:6])
    reg_dim_loss *= reg_weights
    reg_dim_loss_sum = tf.reduce_sum(reg_dim_loss)
    reg_dim_loss_sum *= p.dimensions_loss_weight
    reg_dim_loss_sum /= p.location_loss_weight
    reg_loc_and_dims_loss_sum = reg_loc_loss_sum + reg_dim_loss_sum

    # --- Angle loss ---
    # Angle classification loss
    target_groundtruth_angle_bin, target_groundtruth_angle_res = (
        self._utils.AngleToBin(target_predictions, p.angle_bin_num))
    reg_angle_bin_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_groundtruth_angle_bin, logits=predicted_angle_bin)
    # Angle residual regression loss
    target_groundtruth_angle_bin_onehot = tf.one_hot(
        target_groundtruth_angle_bin,
        depth=p.angle_bin_num,
        axis=-1,
        dtype=tf.float32)
    target_groundtruth_angle_bin_onehot = py_utils.HasShape(
        target_groundtruth_angle_bin_onehot, [bs, npillars, p.angle_bin_num])
    target_groundtruth_angle_res = (
        target_groundtruth_angle_res[..., tf.newaxis] *
        target_groundtruth_angle_bin_onehot)
    target_groundtruth_angle_res = py_utils.HasShape(
        target_groundtruth_angle_res, [bs, npillars, p.angle_bin_num])
    reg_angle_res_loss = self._utils.ScaledHuberLoss(
        predictions=tf.reduce_sum(
            predicted_angle_res * target_groundtruth_angle_bin_onehot, axis=-1),
        labels=tf.reduce_sum(target_groundtruth_angle_res, axis=-1),
        delta=1 / (3.**2))
    angle_loss = reg_angle_bin_loss + reg_angle_res_loss
    angle_loss = tf.expand_dims(angle_loss, axis=-1)
    angle_loss *= reg_weights
    angle_loss_sum = tf.reduce_sum(angle_loss)

    # --- Corner loss ---
    points = py_utils.HasShape(points, [bs, npillars, 3])
    gt_bboxes = self._utils.ResidualsToBBoxesAnchorFree(
        points, target_predictions[..., :6],
        target_groundtruth_angle_bin_onehot, target_groundtruth_angle_res)
    predicted_bboxes_for_corner_loss = self._utils.ResidualsToBBoxesAnchorFree(
        points, predicted_loc_dim_residuals,
        target_groundtruth_angle_bin_onehot, predicted_angle_res)
    predicted_bboxes = self._utils.ResidualsToBBoxesAnchorFree(
        points, predicted_loc_dim_residuals, predicted_angle_bin,
        predicted_angle_res)
    decoded_bboxes = {
        'gt_bboxes': gt_bboxes,
        'predicted_bboxes_for_corner_loss': predicted_bboxes_for_corner_loss,
        'predicted_bboxes': predicted_bboxes
    }
    reg_corner_loss = self._utils.CornerLoss(
        gt_bboxes=gt_bboxes,
        predicted_bboxes=predicted_bboxes_for_corner_loss,
        symmetric=False)
    reg_corner_loss = tf.expand_dims(reg_corner_loss, axis=-1)
    reg_corner_loss *= reg_weights
    reg_corner_loss_sum = tf.reduce_sum(reg_corner_loss)

    ret = py_utils.NestedMap({
        'reg_loc_and_dims_loss_sum': reg_loc_and_dims_loss_sum,
        'angle_loss_sum': angle_loss_sum,
        'reg_corner_loss_sum': reg_corner_loss_sum,
        'decoded_bboxes': decoded_bboxes,
    })

    return ret

  def GenerateTarget(self, predictions, input_batch):
    """Compute the regression and classification target for each prediction.

    Args:
      predictions: The output of `ComputePredictions`, contains: logits - [b,
        nx, ny, nz, na, 7 + num_classes]. na is the number of anchor
        boxes per cell. [..., :7] are (dx, dy, dz, dw, dl, dh, dt).
      input_batch: The input batch from which we accesses the groundtruth.

    Returns:
      A `.NestedMap` object:
        1. points: A float32 tensor with shape [bs, -1, 3] indicating the
          left pillars for predicting boundingboxes.
        2. residuals: A float32 tensor with shape [bs, -1, -1] indicating the
          residual prediction for decoding boundingboxes.
        3. classification_logits: A float32 tensor with shape [bs, -1,
          num_classes] indicating the confidence of specific class.
        4. class_weights: A float32 tensor with shape [bs, -1]
          indicating the classification loss weight for each prediction.
        5. assigned_gt_labels: A float32 tensor with shape [bs, -1]
          , indicating the classification label of each prediction.
        6. reg_weights: A float32 tensor with shape [bs, -1] indicating the
          regression loss weight for each prediction.
        7. assigned_gt_bboxes: A float32 tensor with shape [bs, -1, 7],
          indicating the target gt_box for each prediction.
        8. target_predictions: A float32 tensor with shape [bs, -1, 7],
          indicating the target residuals after encoding.
        9. (optional) assigned_gt_centerness: A float32 tensor with shape
          [bs, -1], indicating the centerness label of each
          prediction. It will return when p.centerness_loss_weight > 0.
        10. (optional) centerness_logits: A float32 tensor with shape [bs, -1,
          num_classes] indicating the predicted centerness score for each
          class. It will return when p.centerness_loss_weight > 0.
    """
    p = self.params

    num_residual_dims = (3 * p.location_loss.num_params_per_prediction +
                         3 * p.dimensions_loss.num_params_per_prediction +
                         p.angle_bin_num * 2)
    predicted_residuals = py_utils.HasShape(predictions.residuals,
                                            [-1, -1, num_residual_dims])
    bs, npillars = py_utils.GetShape(predicted_residuals, 2)
    predicted_class_logits = py_utils.HasShape(
        predictions.classification_logits, [bs, npillars, p.num_classes])

    class_weights = input_batch.assigned_cls_mask
    class_weights = py_utils.HasShape(class_weights,
                                      [bs, -1, -1, p.grid_size_z])
    _, nx, ny, nz = py_utils.GetShape(class_weights, 4)
    class_weights = tf.reshape(class_weights, [bs, npillars])

    assigned_gt_labels = py_utils.HasShape(input_batch.assigned_gt_labels,
                                           [bs, nx, ny, nz])
    assigned_gt_labels = tf.reshape(assigned_gt_labels, [bs, npillars])

    reg_weights = input_batch.assigned_reg_mask
    reg_weights = py_utils.HasShape(reg_weights,
                                    [bs, nx, ny, nz, p.num_classes])
    reg_weights = tf.reduce_sum(reg_weights, axis=-1)
    reg_weights = tf.reshape(reg_weights, [bs, npillars])

    assigned_gt_bboxes = py_utils.HasShape(input_batch.assigned_gt_bbox,
                                           [bs, nx, ny, nz, 7])
    assigned_gt_bboxes = tf.reshape(assigned_gt_bboxes, [bs, npillars, 7])

    target_predictions = py_utils.HasShape(input_batch.target_predictions,
                                           [bs, nx, ny, nz, 7])
    target_predictions = tf.reshape(target_predictions, [bs, npillars, 7])

    points = py_utils.HasShape(predictions.points, [bs, npillars, 3])

    ret = py_utils.NestedMap({
        'points': points,
        'residuals': predicted_residuals,
        'classification_logits': predicted_class_logits,
        'class_weights': class_weights,
        'assigned_gt_labels': assigned_gt_labels,
        'reg_weights': reg_weights,
        'assigned_gt_bboxes': assigned_gt_bboxes,
        'target_predictions': target_predictions,
    })

    if p.centerness_loss_weight > 0:
      assigned_gt_centerness = py_utils.HasShape(
          input_batch.assigned_gt_center_ness, [bs, nx, ny, nz])
      assigned_gt_centerness = tf.reshape(assigned_gt_centerness,
                                          [bs, npillars])
      predicted_centerness_logits = py_utils.HasShape(
          predictions.centerness_logits, [bs, npillars, p.num_classes])
      ret.update({
          'centerness_logits': predicted_centerness_logits,
          'assigned_gt_centerness': assigned_gt_centerness,
      })

    return ret

  def ComputeLoss(self, theta, predictions, input_batch):
    """Computes loss and other metrics for the given predictions.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      predictions: The output of `ComputePredictions`, contains: logits - [b,
        nx, ny, nz, na, 7 + num_classes]. na is the number of anchor
        boxes per cell. [..., :7] are (dx, dy, dz, dw, dl, dh, dt).
      input_batch: The input batch from which we accesses the groundtruth.

    Returns:
      Two dicts defined as BaseTask.ComputeLoss.
    """
    p = self.params

    assigned_labels = self.GenerateTarget(predictions, input_batch)

    num_residual_dims = (3 * p.location_loss.num_params_per_prediction +
                         3 * p.dimensions_loss.num_params_per_prediction +
                         p.angle_bin_num * 2)
    predicted_residuals = py_utils.HasShape(assigned_labels.residuals,
                                            [-1, -1, num_residual_dims])
    bs, npillars = py_utils.GetShape(predicted_residuals, 2)
    predicted_class_logits = py_utils.HasShape(
        assigned_labels.classification_logits, [bs, npillars, p.num_classes])

    # Compute class and regression weights.
    # --- class weights ---
    class_weights = assigned_labels.class_weights
    per_class_loss_weight = tf.constant([[[p.per_class_loss_weight]]],
                                        dtype=tf.float32)
    per_class_loss_weight = tf.reshape(per_class_loss_weight,
                                       [1, 1, p.num_classes])
    class_weights = class_weights[..., tf.newaxis] * per_class_loss_weight
    # --- regression weights ---
    reg_weights = assigned_labels.reg_weights
    reg_weights = tf.reshape(reg_weights, [bs, npillars, 1])

    if p.loss_norm_type == pillars.LossNormType.NORM_BY_NUM_POSITIVES:
      # Sum to get the number of foreground anchors for each example.
      loss_normalization = tf.reduce_sum(reg_weights, axis=[1, 2])
      loss_normalization = tf.maximum(loss_normalization,
                                      tf.ones_like(loss_normalization))
      # Reshape for broadcasting.
      loss_normalization = tf.reshape(loss_normalization, [bs, 1, 1])

      class_weights /= loss_normalization
      reg_weights /= loss_normalization

    # Classification loss.
    assigned_gt_labels = tf.one_hot(
        assigned_labels.assigned_gt_labels, p.num_classes, dtype=tf.float32)
    if p.classification_loss_fn == ClassLossFN.FOCAL_SIGMOID_LOSS:
      loss_fn = functools.partial(
          py_utils.SigmoidCrossEntropyFocalLoss,
          alpha=p.focal_loss_alpha,
          gamma=p.focal_loss_gamma)
    elif p.classification_loss_fn == ClassLossFN.SIGMOID_LOSS:
      loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    class_loss_sum = self._ComputeClassificationLoss(
        predicted_class_logits=predicted_class_logits,
        assigned_gt_labels=assigned_gt_labels,
        class_weights=class_weights,
        loss_fn=loss_fn)

    reg_loss = self._ComputeRegressionLoss(
        points=assigned_labels.points,
        predicted_residuals=predicted_residuals,
        target_predictions=assigned_labels.target_predictions,
        reg_weights=reg_weights)
    reg_loc_and_dims_loss_sum = reg_loss.reg_loc_and_dims_loss_sum
    angle_loss_sum = reg_loss.angle_loss_sum
    reg_corner_loss_sum = reg_loss.reg_corner_loss_sum
    decoded_bboxes = reg_loss.decoded_bboxes

    # Num. predictions.
    preds = tf.cast(bs, class_loss_sum.dtype)

    # Normalize all of the components by batch size.
    reg_loc_and_dims_loss = reg_loc_and_dims_loss_sum / preds
    angle_loss = angle_loss_sum / preds
    reg_corner_loss = reg_corner_loss_sum / preds
    class_loss = class_loss_sum / preds

    # Compute total localization regression loss.
    reg_loss = (
        p.location_loss_weight * reg_loc_and_dims_loss +
        p.rotation_loss_weight * angle_loss +
        p.corner_loss_weight * reg_corner_loss)

    # Apply weights to normalized class losses.
    loss = (
        class_loss * p.classification_loss_weight +
        reg_loss * p.localization_loss_weight)

    metrics_dict = {
        'loss': (loss, preds),
        'loss/class': (class_loss, preds),
        'loss/reg': (reg_loss, preds),
        'loss/reg/rot': (angle_loss, preds),
        'loss/reg/loc': (reg_loc_and_dims_loss, preds),
        'loss/reg/corner': (reg_corner_loss, preds),
    }

    # Calculate dimension errors
    gt_bboxes = decoded_bboxes['gt_bboxes']
    predicted_bboxes = decoded_bboxes['predicted_bboxes_for_corner_loss']
    dimension_errors_dict = self._BBoxDimensionErrors(gt_bboxes,
                                                      predicted_bboxes,
                                                      reg_weights)
    metrics_dict.update(dimension_errors_dict)

    per_example_dict = dict()
    for k, v in assigned_labels.items():
      per_example_dict[k] = v

    if p.centerness_loss_weight > 0:
      predicted_centerness_logits = py_utils.HasShape(
          assigned_labels.centerness_logits, [bs, npillars, p.num_classes])
      assigned_gt_centerness = py_utils.HasShape(
          assigned_labels.assigned_gt_centerness, [bs, npillars])
      assigned_gt_centerness = (
          assigned_gt_centerness[..., tf.newaxis] * assigned_gt_labels)
      centerness_loss_sum = self._ComputeClassificationLoss(
          predicted_class_logits=predicted_centerness_logits,
          assigned_gt_labels=assigned_gt_centerness,
          class_weights=class_weights,
          loss_fn=tf.nn.sigmoid_cross_entropy_with_logits)
      centerness_loss = centerness_loss_sum / preds
      loss += centerness_loss * p.centerness_loss_weight
      metrics_dict.update({
          'loss': (loss, preds),
          'loss/centerness': (centerness_loss, preds),
      })

    return metrics_dict, per_example_dict

  def _BBoxesAndLogits(self, input_batch, predictions):
    """Decode an input batch, computing predicted bboxes from residuals."""
    p = self.params

    num_residual_dims = (3 * p.location_loss.num_params_per_prediction +
                         3 * p.dimensions_loss.num_params_per_prediction +
                         p.angle_bin_num * 2)
    predicted_residuals = py_utils.HasShape(predictions.residuals,
                                            [-1, -1, num_residual_dims])
    predicted_residuals = predicted_residuals[..., :6 + p.angle_bin_num * 2]
    bs, npillars = py_utils.GetShape(predicted_residuals, 2)
    predicted_loc_dim_residuals, predicted_angle_bin, predicted_angle_res = (
        tf.split(
            predicted_residuals, [6, p.angle_bin_num, p.angle_bin_num],
            axis=-1))

    points = py_utils.HasShape(predictions.points, [bs, npillars, 3])
    predicted_bboxes = self._utils.ResidualsToBBoxesAnchorFree(
        points, predicted_loc_dim_residuals, predicted_angle_bin,
        predicted_angle_res)

    predicted_bboxes = py_utils.HasShape(predicted_bboxes, [bs, npillars, 7])

    classification_logits = py_utils.HasShape(predictions.classification_logits,
                                              [bs, npillars, p.num_classes])
    classification_scores = tf.sigmoid(classification_logits)

    if p.centerness_loss_weight > 0:
      centerness_logits = py_utils.HasShape(predictions.centerness_logits,
                                            [bs, npillars, p.num_classes])
      centerness_logits = tf.sigmoid(centerness_logits)
      classification_scores *= centerness_logits

    return py_utils.NestedMap({
        'predicted_bboxes': predicted_bboxes,
        'classification_scores': classification_scores,
    })

  def _NoPostProcessDecoder(self, predicted_bboxes, classification_scores,
                            score_threshold):
    """Cast the predictions to PostProcessing format without any conversion.

    This function is useful for end-to-end HeatMapNMS.
    Args:
      predicted_bboxes: A float32 tensor with shape [bs, -1, 7], indicating the
        decoded boundingbox prediction.
      classification_scores: A float32 tensor with shape [bs, -1, num_classes].
      score_threshold: A float32 tensor with shape [num_classes].

    Returns:
      per_cls_idx: A int32 tensor with shape [bs, num_classes, -1]. Used for
        gathering to cast a tensor from [bs, -1, ...] to
        [bs, num_classes, -1, ...].
      per_cls_bboxes: A float32 tensor with shape [bs, num_classes, -1, 7].
      per_cls_bbox_scores: A float32 tensor with shape [bs, num_classes, -1].
      per_cls_valid_mask: A float32 tensor with shape [bs, num_classes, -1],
        indicating whether a prediction is valid or not.
    """
    p = self.params

    predicted_bboxes = py_utils.HasShape(predicted_bboxes, [-1, -1, 7])
    bs, npillars = py_utils.GetShape(predicted_bboxes, 2)

    per_cls_idx = tf.reshape(tf.range(npillars), [1, 1, npillars])
    per_cls_idx = tf.tile(per_cls_idx, [bs, p.num_classes, 1])

    per_cls_bboxes = tf.gather(predicted_bboxes, per_cls_idx, batch_dims=1)
    per_cls_bbox_scores = tf.transpose(classification_scores, [0, 2, 1])

    nms_score_threshold = tf.convert_to_tensor(
        p.nms_score_threshold, dtype=tf.float32)
    per_cls_valid_mask = tf.greater_equal(
        per_cls_bbox_scores, nms_score_threshold[tf.newaxis, ..., tf.newaxis])
    per_cls_valid_mask = tf.cast(per_cls_valid_mask, tf.float32)

    return per_cls_idx, per_cls_bboxes, per_cls_bbox_scores, per_cls_valid_mask

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
      classification_scores = bboxes_and_logits.classification_scores

      if p.nms_decoder_type == NMSDecoderType.NMS_DECODER:
        decode_fn = functools.partial(
            detection_decoder.DecodeWithNMS,
            nms_iou_threshold=p.nms_iou_threshold,
            max_boxes_per_class=p.max_nms_boxes,
            use_oriented_per_class_nms=p.use_oriented_per_class_nms)
      elif p.nms_decoder_type == NMSDecoderType.HEATMAP_NMS_DECODER:
        points = py_utils.HasShape(input_placeholders.anchor_centers,
                                   [-1, -1, -1, -1, 3])
        bs, nx, ny = py_utils.GetShape(points, 3)
        decode_fn = functools.partial(
            detection_decoder.DecodeWithMaxPoolNMS,
            heatmap_shape=[bs, nx, ny],
            kernel_size=p.heatmap_nms_kernel_size,
            max_boxes_per_class=p.max_nms_boxes,
            use_oriented_per_class_nms=p.use_oriented_per_class_nms)
      elif p.nms_decoder_type == NMSDecoderType.NO_NMS_DECODER:
        decode_fn = self._NoPostProcessDecoder

      _, per_cls_bboxes, per_cls_bbox_scores, per_cls_valid_mask = (
          decode_fn(
              predicted_bboxes,
              classification_scores,
              score_threshold=p.nms_score_threshold))
      per_cls_bbox_scores *= per_cls_valid_mask

      fetches = {
          'per_class_predicted_bboxes': per_cls_bboxes,
          'per_class_predicted_bbox_scores': per_cls_bbox_scores,
          'per_class_valid_mask': per_cls_valid_mask
      }
      subgraphs['default'] = fetches, dict(input_placeholders.FlattenItems())
    return subgraphs

  def Decode(self, input_batch):
    """Decode an input batch, computing predicted bboxes from residuals."""
    p = self.params

    predictions = self.ComputePredictions(self.theta, input_batch)
    bboxes_and_logits = self._BBoxesAndLogits(input_batch, predictions)
    predicted_bboxes = bboxes_and_logits.predicted_bboxes
    batch_size, num_bboxes, _ = py_utils.GetShape(predicted_bboxes, 3)
    classification_scores = bboxes_and_logits.classification_scores
    classification_scores = py_utils.HasShape(
        classification_scores, [batch_size, num_bboxes, p.num_classes])

    _, per_example_dict = self.ComputeLoss(self.theta, predictions, input_batch)
    if 'score_scaler' in per_example_dict:
      classification_scores *= per_example_dict['score_scaler']

    with tf.device('/cpu:0'):
      # Decode the predicted bboxes, performing NMS.
      if p.nms_decoder_type == NMSDecoderType.NMS_DECODER:
        decode_fn = functools.partial(
            detection_decoder.DecodeWithNMS,
            nms_iou_threshold=p.nms_iou_threshold,
            max_boxes_per_class=p.max_nms_boxes,
            use_oriented_per_class_nms=p.use_oriented_per_class_nms)
      elif p.nms_decoder_type == NMSDecoderType.HEATMAP_NMS_DECODER:
        points = py_utils.HasShape(input_batch.anchor_centers,
                                   [-1, -1, -1, -1, 3])
        bs, nx, ny = py_utils.GetShape(points, 3)
        decode_fn = functools.partial(
            detection_decoder.DecodeWithMaxPoolNMS,
            heatmap_shape=[bs, nx, ny],
            kernel_size=p.heatmap_nms_kernel_size,
            max_boxes_per_class=p.max_nms_boxes,
            use_oriented_per_class_nms=p.use_oriented_per_class_nms)
      elif p.nms_decoder_type == NMSDecoderType.NO_NMS_DECODER:
        decode_fn = self._NoPostProcessDecoder
      per_cls_idxs, per_cls_bboxes, per_cls_bbox_scores, per_cls_valid_mask = (
          decode_fn(
              predicted_bboxes,
              classification_scores,
              score_threshold=p.nms_score_threshold))

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
              _ReshapeGather(per_example_dict['target_predictions']),
          'per_class_gt_labels':
              _ReshapeGather(per_example_dict['assigned_gt_labels']),
          'per_class_gt_bboxes':
              _ReshapeGather(per_example_dict['assigned_gt_bboxes']),
          'per_class_residuals':
              _ReshapeGather(per_example_dict['residuals']),
          'per_class_logits':
              _ReshapeGather(per_example_dict['classification_logits']),
          'per_class_points':
              _ReshapeGather(per_example_dict['points']),
      })

    decoder_outputs.update(
        self.output_decoder.ProcessOutputs(input_batch, model_outputs))

    # Produce global step as an output (which is the step
    # of the checkpoint being decoded.)
    decoder_outputs.global_step = py_utils.GetGlobalStep()

    return decoder_outputs
