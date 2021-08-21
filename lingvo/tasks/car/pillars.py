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
r"""PointPillars implementation.

[1] PointPillars. https://arxiv.org/abs/1812.05784
"""

import enum
import functools

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.tasks.car import builder_lib
from lingvo.tasks.car import detection_3d_lib
from lingvo.tasks.car import geometry
from lingvo.tasks.car import point_detector
import numpy as np


def SparseToDense(grid_shape, locations, feats):
  """Converts a sparse representation back to the dense grid.

  Args:
    grid_shape: (nx, ny, nz). The shape of the grid.
    locations: [b, p, 3]. Locations of the pillars.
    feats: [b, p, fdims]. Extracted features for pillars.

  Returns:
    grid_feats of shape [b, nx, ny, nz * fdims].
  """
  nx, ny, nz = grid_shape
  b, p, _ = py_utils.GetShape(locations, 3)
  feats = py_utils.HasShape(feats, [b, p, -1])
  _, _, fdims = py_utils.GetShape(feats, 3)
  indices = tf.concat(
      [tf.tile(tf.range(b)[:, tf.newaxis, tf.newaxis], [1, p, 1]), locations],
      axis=2)
  grid = tf.scatter_nd(indices, feats, [b, nx, ny, nz, fdims])
  return tf.reshape(grid, [b, nx, ny, nz * fdims])


class PointsToGridFeaturizer(base_layer.BaseLayer):
  """Layer for processing points to grid outputs."""

  @classmethod
  def Params(cls, num_laser_features, num_output_features=64):
    p = super().Params()
    p.Define('num_laser_features', num_laser_features,
             'The number of (non-xyz) laser features of the input.')

    builder = Builder()
    total_num_laser_features = 9 + num_laser_features
    p.Define(
        'featurizer',
        builder.Featurizer('feat',
                           [total_num_laser_features, num_output_features]),
        'Point cloud feature extractor.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('featurizer', p.featurizer)

  def FProp(self, theta, input_batch):
    # pyformat: disable
    """Compute features for the pillars and convert them back to a dense grid.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` object containing input tensors. Following
        keys are required:

        - grid_num_points: Integer tensor with shape [batch size, nx, ny, nz],
          where nx, ny, nz corresponds to the grid sizes (i.e., number of voxels
          in each axis dimension).
        - pillar_points: Float tensor with shape [batch size, num_pillars,
          num_points_per_pillar, 3 + num_laser_features]
        - pillar_centers: Float tensor with shape [batch size, num_pillars,
          num_points_per_pillar, 3]
        - pillar_locations: Float tensor with shape [batch size, num_pillars, 3]

    Returns:
      The dense features with shape [b, nx, ny, nz * fdims].
    """
    # pyformat: enable
    p = self.params
    bs, nx, ny, nz = py_utils.GetShape(input_batch.grid_num_points, 4)
    # Process points to concatenate a set of fixed features (e.g.,
    # add means, centers, normalize points to means).
    num_features = 3 + p.num_laser_features
    pillar_points = py_utils.HasShape(input_batch.pillar_points,
                                      [bs, -1, -1, num_features])
    _, npillars, npoints, _ = py_utils.GetShape(pillar_points, 4)
    pillar_xyz = pillar_points[..., :3]

    # Compute number of points per pillar and prepare for broadcasting.
    pillar_num_points = tf.gather_nd(
        input_batch.grid_num_points, input_batch.pillar_locations, batch_dims=1)
    pillar_num_points = pillar_num_points[..., tf.newaxis, tf.newaxis]

    # Compute mean by computing sum and dividing by number of points. Clip the
    # denominator by 1.0 to gracefully handle empty pillars.
    pillar_sum = tf.reduce_sum(pillar_xyz, axis=2, keepdims=True)
    pillar_means = pillar_sum / tf.maximum(
        tf.cast(pillar_num_points, tf.float32), 1.0)

    pillar_feats = pillar_points[..., 3:]
    pillar_centers = py_utils.HasShape(input_batch.pillar_centers,
                                       [bs, -1, 1, 3])
    pillar_concat = tf.concat(
        axis=3,
        values=[
            pillar_xyz - pillar_means, pillar_feats,
            tf.tile(pillar_means, [1, 1, npoints, 1]),
            tf.tile(pillar_centers, [1, 1, npoints, 1])
        ])
    # Featurize pillars.
    pillar_features = self.featurizer.FProp(theta.featurizer, pillar_concat)

    # Convert back to the dense grid.
    pillar_locations = py_utils.HasShape(input_batch.pillar_locations,
                                         [bs, npillars, 3])
    dense_features = SparseToDense(
        grid_shape=(nx, ny, nz),
        locations=pillar_locations,
        feats=pillar_features)
    return dense_features


# pyformat: disable
class Builder(builder_lib.ModelBuilderBase):
  """Builder for the Pillars model."""

  def __init__(self):
    super().__init__()
    self.conv_init_method = builder_lib.KaimingUniformFanInRelu
    self.linear_params_init = py_utils.WeightInit.KaimingUniformFanInRelu()
    self.bn_params_init = py_utils.WeightInit.UniformPositive()

  def Featurizer(self, name, dims):
    return self._Seq(
        name,
        self._MLP('mlp', dims),
        self._Max('max'))

  def _Deconv(self, name, filter_shape, stride):
    return layers.DeconvLayer.Params().Set(
        name=name,
        filter_shape=filter_shape,
        filter_stride=(stride, stride))

  def _Block(self, name, stride, repeats, idims, odims, activation=None):
    """[1]. Sec 2.2."""
    return self._Seq(
        name,
        self._Conv('c3x3', (3, 3, idims, odims), stride, activation=activation),
        self._Rep(
            'rep',
            repeats,
            self._Conv('c3x3', (3, 3, odims, odims), activation=activation)),
        self._Fetch('final'))

  def _TopDown(self, name, strides=(2, 2, 2), channel_multiplier=1,
               activation=None):
    """[1]. Sec 2.2."""
    if len(strides) != 3:
      raise ValueError('`strides` expected to be list/tuple of len 3.')

    return self._Seq(
        name,
        self._Block('b0', strides[0], 3, channel_multiplier * 64,
                    channel_multiplier * 64, activation),
        self._Block('b1', strides[1], 5, channel_multiplier * 64,
                    channel_multiplier * 128, activation),
        self._Block('b2', strides[2], 5, channel_multiplier * 128,
                    channel_multiplier * 256, activation))

  def _Upsample(self, name, stride, idims, odims, activation=None):
    """[1]. Sec 2.2."""
    # Match the kernel size to the stride in order to ensure that the output
    # activation map has no holes and to minimize any checkerboard artifacts.
    # TODO(shlens): Consider replacing this in the future with a bilinear
    # interpolation followed by a 3x3 convolution.
    kernel = stride
    return self._Seq(
        name,
        self._Deconv('deconv', (kernel, kernel, odims, idims), stride),
        self._BN('bn', odims),
        self._Activation('activation', activation))

  def Contract(self, down_strides=(2, 2, 2), channel_multiplier=1,
               activation=None):
    """Contracting part of [1] Sec 2.2."""
    return self._Branch(
        'branch',
        self._TopDown('topdown', strides=down_strides,
                      channel_multiplier=channel_multiplier,
                      activation=activation),
        ['b1.final', 'b0.final'])

  def Expand(self, odims, channel_multiplier=1, activation=None):
    """Expanding part of [1] Sec 2.2."""
    # Note that the resulting output will be 3*odims
    return self._Concat(
        'concat',
        self._Seq(
            'b2',
            self._ArgIdx('idx', [0]),
            self._Upsample('ups', 4, channel_multiplier * 256, odims, activation)),
        self._Seq(
            'b1',
            self._ArgIdx('idx', [1]),
            self._Upsample('ups', 2, channel_multiplier * 128, odims,
                           activation)),
        self._Seq(
            'b0',
            self._ArgIdx('idx', [2]),
            self._Upsample('ups', 1, channel_multiplier * 64, odims,
                           activation)))

  def Backbone(self, odims, down_strides=(2, 2, 2), channel_multiplier=1,
               activation=None):
    """[1]. Sec 2.2."""
    # We assume (H, W) are multiple of 8. So that we can concat
    # multiple-scale feature maps together after upsample.
    return self._Seq(
        'backbone',
        self.Contract(down_strides, channel_multiplier=channel_multiplier,
                      activation=activation),
        self.Expand(odims, channel_multiplier=channel_multiplier,
                    activation=activation))

  def Detector(self, name, idims, odims, conv_init_method=None,
               bias_params_init=None):
    # Implemented according to VoxelNet
    # https://arxiv.org/pdf/1711.06396.pdf
    # May add more Conv2D layers before predictor for better performance.
    return self._Seq(
        name,
        self._ConvPlain('predict', (3, 3, idims, odims),
                        conv_init_method=conv_init_method),
        self._Bias('predict_bias', odims, bias_params_init))

# pyformat: enable


class LossNormType(enum.Enum):
  NO_NORM = 0
  NORM_BY_NUM_POSITIVES = 1


class ModelV1(point_detector.PointDetectorBase):
  """PointPillars model.

  Base class implements common Decoder functions, though they can be
  overridden if desired.
  """

  NUM_OUTPUT_CHANNELS = 128

  @classmethod
  def Params(cls,
             grid_size_z=1,
             num_anchors=2,
             num_classes=1,
             num_laser_features=1):
    p = super().Params(num_classes=num_classes)
    p.Define('grid_size_z', grid_size_z, 'The grid size along the z-axis.')
    p.Define('num_anchors', num_anchors, 'The number of anchor boxes.')
    p.Define('num_laser_features', num_laser_features,
             'The number of (non-xyz) laser features of the input.')
    p.Define('input_featurizer',
             PointsToGridFeaturizer.Params(num_laser_features),
             'Point cloud feature extractor.')

    builder = Builder()
    p.Define('backbone', builder.Backbone(cls.NUM_OUTPUT_CHANNELS),
             'Dense features pyramid.')
    # Backbone() concatenates 3 different scales of features.
    idims = 3 * cls.NUM_OUTPUT_CHANNELS
    # 7: predicted (dx, dy, dz, dw, dl, dh, dt).
    class_odims = grid_size_z * num_anchors * num_classes
    reg_odims = grid_size_z * num_anchors * 7
    rot_odims = grid_size_z * num_anchors * 2
    # Although theoretically a single conv layer can generate both the
    # regression and classification logits, we try to implement the paper
    # faithfully, which uses two different layers.
    p.Define('class_detector', builder.Detector('class', idims, class_odims),
             'Dense features to class logits.')
    p.Define('regression_detector', builder.Detector('reg', idims, reg_odims),
             'Dense features to regression logits.')
    p.Define('direction_classifier', builder.Detector('dir', idims, rot_odims),
             'Dense features to rotation direction classifier.')
    # We disable the direction classifier by default since it has
    # weird discontinous optimization objectives around the threshold
    # and it doesn't improve mAP.
    p.Define(
        'direction_classifier_weight', 0.0,
        'If > 0, adds a direction classifier to the model and adds '
        'to the total loss with this weight.')
    p.Define(
        'direction_aware_rot_loss', False, 'If True, changes the heading loss '
        'from sin(theta_delta) to WrapAngleRad(theta_delta), which makes the '
        'model produce headings between [-pi to pi].')

    p.Define(
        'squash_rotation_predictions', False,
        'Apply tanh squashing to rotation predictions to ensure outputs '
        'are between (-pi, pi).')
    p.Define('focal_loss_alpha', 0.25, 'The alpha parameter in focal loss '
             '(see paper eq. 4).')
    p.Define('focal_loss_gamma', 2.0, 'The gamma parameter in focal loss '
             '(see paper eq. 4).')
    p.Define(
        'localization_loss_weight', 2.0,
        'Localization loss weight factor between localization and '
        'class loss contributions.')
    p.Define(
        'classification_loss_weight', 1.0,
        'Classification loss weight factor between localization and '
        'class loss contributions.')
    p.Define(
        'location_loss_weight', 1.0,
        'Weight multiplier for contribution of location loss '
        'to full localization/regression loss')
    p.Define(
        'dimension_loss_weight', 1.0,
        'Weight multiplier for contribution of dimension loss '
        'to full localization/regression loss')
    p.Define(
        'rotation_loss_weight', 1.0,
        'Weight multiplier for contribution of rotation loss '
        'to full localization/regression loss')

    p.Define('loss_norm_type', LossNormType.NORM_BY_NUM_POSITIVES,
             'Normalization function for class and regularization weights.')

    p.Define('oracle_location', False,
             'If true, the model predicts the ground truth for location.')
    p.Define('oracle_dimension', False,
             'If true, the model predicts the ground truth for dimension.')
    p.Define('oracle_rotation', False,
             'If true, the model predicts the ground truth for rotation.')

    tp = p.train
    tp.learning_rate = 0.001
    tp.optimizer = optimizer.Momentum.Params().Set(alpha=0.9)
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._utils = detection_3d_lib.Utils3D()

    self.CreateChild('input_featurizer', p.input_featurizer)
    self.CreateChild('backbone', p.backbone)
    self.CreateChild('class_detector', p.class_detector)
    self.CreateChild('regression_detector', p.regression_detector)
    if p.direction_classifier_weight > 0.0:
      self.CreateChild('direction_classifier', p.direction_classifier)

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions for `input_batch`.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      A `.NestedMap` contains
        logits - [b, nx, ny, nz, na, 7 + num_classes]
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
        class_detection,
        [bs, nx, ny, p.grid_size_z, p.num_anchors, p.num_classes])
    predicted_residuals = tf.reshape(
        reg_detection, [bs, nx, ny, p.grid_size_z, p.num_anchors, 7])

    if p.squash_rotation_predictions:
      predicted_rotations = predicted_residuals[..., 6:]
      predicted_rotations = np.pi * tf.tanh(predicted_rotations)
      predicted_residuals = tf.concat(
          [predicted_residuals[..., :6], predicted_rotations], axis=-1)

    if p.oracle_location or p.oracle_dimension or p.oracle_rotation:
      gt_residuals = py_utils.HasShape(
          input_batch.anchor_localization_residuals,
          [bs, nx, ny, p.grid_size_z, p.num_anchors, 7])

      # Replace the predicted components with the ground truth if needed.
      if p.oracle_location:
        location = gt_residuals[..., 0:3]
      else:
        location = predicted_residuals[..., 0:3]

      if p.oracle_dimension:
        dimension = gt_residuals[..., 3:6]
      else:
        dimension = predicted_residuals[..., 3:6]

      if p.oracle_rotation:
        rotation = gt_residuals[..., 6:]
      else:
        rotation = predicted_residuals[..., 6:]
      predicted_residuals = tf.concat([location, dimension, rotation], axis=-1)

    ret = py_utils.NestedMap({
        'residuals': predicted_residuals,
        'classification_logits': predicted_classification_logits,
    })

    if p.direction_classifier_weight > 0.0:
      predicted_dir = self.direction_classifier.FProp(
          theta.direction_classifier, act)
      predicted_dir = tf.reshape(predicted_dir,
                                 [bs, nx, ny, p.grid_size_z, p.num_anchors, 2])
      ret.predicted_dir = predicted_dir

    return ret

  def _ComputeClassificationLoss(self, predictions, input_batch, class_weights):
    """Compute classification loss for the given predictions.

    Args:
      predictions: The output of `ComputePredictions`, contains: logits - [b,
        nx, ny, nz, na, 7 + num_classes]. na is the number of anchor
        boxes per cell. [..., :7] are (dx, dy, dz, dw, dl, dh, dt).
      input_batch: The input batch from which we accesses the groundtruth.
      class_weights: Per-class weights to use in loss computation.

    Returns:
      Classification loss.

    """
    p = self.params
    predicted_class_logits = py_utils.HasShape(
        predictions.classification_logits,
        [-1, -1, -1, -1, p.num_anchors, p.num_classes])
    bs, nx, ny, nz, na, _ = py_utils.GetShape(predicted_class_logits, 6)
    assigned_gt_labels = py_utils.HasShape(input_batch.assigned_gt_labels,
                                           [bs, nx, ny, nz, na])
    class_loss = py_utils.SigmoidCrossEntropyFocalLoss(
        logits=predicted_class_logits,
        labels=tf.one_hot(assigned_gt_labels, p.num_classes),
        alpha=p.focal_loss_alpha,
        gamma=p.focal_loss_gamma)
    class_loss *= class_weights[..., tf.newaxis]
    class_loss_sum = tf.reduce_sum(class_loss)
    return class_loss_sum

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
    predicted_residuals = py_utils.HasShape(predictions.residuals,
                                            [-1, -1, -1, -1, p.num_anchors, 7])
    predicted_class_logits = py_utils.HasShape(
        predictions.classification_logits,
        [-1, -1, -1, -1, p.num_anchors, p.num_classes])
    bs, nx, ny, nz, na, _ = py_utils.GetShape(predicted_class_logits, 6)

    # Compute class and regression weights.
    class_weights = input_batch.assigned_cls_mask
    class_weights = py_utils.HasShape(class_weights, [bs, nx, ny, nz, na])
    reg_weights = input_batch.assigned_reg_mask
    reg_weights = py_utils.HasShape(reg_weights, [bs, nx, ny, nz, na])
    reg_weights = tf.expand_dims(reg_weights, -1)

    if p.loss_norm_type == LossNormType.NORM_BY_NUM_POSITIVES:
      # Compute number of positive anchors per example.
      foreground_mask = py_utils.HasShape(input_batch.assigned_reg_mask,
                                          [bs, nx, ny, nz, na])
      # Sum to get the number of foreground anchors for each example.
      loss_normalization = tf.reduce_sum(foreground_mask, axis=[1, 2, 3, 4])
      loss_normalization = tf.maximum(loss_normalization,
                                      tf.ones_like(loss_normalization))
      # Reshape for broadcasting.
      loss_normalization = tf.reshape(loss_normalization, [bs, 1, 1, 1, 1, 1])

      class_weights /= loss_normalization
      reg_weights /= loss_normalization

    # Classification loss.
    class_loss_sum = self._ComputeClassificationLoss(predictions, input_batch,
                                                     class_weights)

    # Regression loss.
    anchor_localization_residuals = py_utils.HasShape(
        input_batch.anchor_localization_residuals, [bs, nx, ny, nz, na, 7])

    # Location and dimensions loss.
    reg_loc_and_dims_loss = self._utils.ScaledHuberLoss(
        predictions=py_utils.HasShape(predicted_residuals[..., :6],
                                      [bs, nx, ny, nz, na, 6]),
        labels=anchor_localization_residuals[..., :6],
        delta=1 / (3.**2))

    # Rotation loss is computed on a transform on rot_delta. For a direction
    # aware loss, we simply wrap the angles to -pi to pi; for a loss that is
    # symmetric to direction (i.e., rotating by pi), we use a sin transform.
    rot_delta_transform = tf.sin
    if p.direction_aware_rot_loss:
      rot_delta_transform = functools.partial(
          geometry.WrapAngleRad, min_val=-np.pi, max_val=np.pi)

    rot_delta = (
        predicted_residuals[..., 6:] - anchor_localization_residuals[..., 6:])
    reg_rot_loss = self._utils.ScaledHuberLoss(
        predictions=rot_delta_transform(rot_delta),
        labels=tf.zeros_like(rot_delta),
        delta=1 / (3.**2))

    # Direction loss
    if p.direction_classifier_weight > 0.0:
      # The target rotations are in the assigned_gt_bbox tensor,
      # which already has assigned a gt bounding box to every anchor.
      rot_target = input_batch.assigned_gt_bbox[..., 6]
      # If rotation is > 0, the class is 1, else it is 0.
      rot_dir = tf.cast(rot_target > 0., tf.int32)

      # Compute one-hot labels as a target.
      rot_dir_onehot = tf.one_hot(rot_dir, 2)

      # Manually handle loss reduction.
      dir_loss = tf.losses.softmax_cross_entropy(
          onehot_labels=rot_dir_onehot,
          logits=predictions.predicted_dir,
          weights=tf.squeeze(reg_weights, axis=-1),
          reduction=tf.losses.Reduction.NONE)
      # Reduce across all dimensions (we'll divide by the batch size below).
      dir_loss_sum = tf.reduce_sum(dir_loss)
    else:
      dir_loss_sum = 0.0

    # Compute loss contribution from location and dimension separately.
    reg_loc_loss = reg_loc_and_dims_loss[..., :3] * reg_weights
    reg_loc_loss_sum = tf.reduce_sum(reg_loc_loss)

    reg_dim_loss = reg_loc_and_dims_loss[..., 3:6] * reg_weights
    reg_dim_loss_sum = tf.reduce_sum(reg_dim_loss)

    # Compute rotation loss contribution.
    reg_rot_loss *= reg_weights
    reg_rot_loss_sum = tf.reduce_sum(reg_rot_loss)

    # Num. predictions.
    # TODO(zhifengc): Consider other normalization factors. E.g., # of bboxes.
    preds = tf.cast(bs, class_loss_sum.dtype)

    # Normalize all of the components by batch size.
    reg_loc_loss = reg_loc_loss_sum / preds
    reg_dim_loss = reg_dim_loss_sum / preds
    reg_rot_loss = reg_rot_loss_sum / preds
    class_loss = class_loss_sum / preds
    dir_loss = dir_loss_sum / preds

    # Compute total localization regression loss.
    reg_loss = (
        p.location_loss_weight * reg_loc_loss +
        p.dimension_loss_weight * reg_dim_loss +
        p.rotation_loss_weight * reg_rot_loss)

    # Apply weights to normalized class losses.
    loss = (
        class_loss * p.classification_loss_weight +
        reg_loss * p.localization_loss_weight +
        dir_loss * p.direction_classifier_weight)

    metrics_dict = {
        'loss': (loss, preds),
        'loss/class': (class_loss, preds),
        'loss/reg': (reg_loss, preds),
        'loss/reg/rot': (reg_rot_loss, preds),
        'loss/reg/loc': (reg_loc_loss, preds),
        'loss/reg/dim': (reg_dim_loss, preds),
        'loss/dir': (dir_loss, preds),
    }

    # Calculate dimension errors
    min_angle_rad = -np.pi if p.direction_aware_rot_loss else 0
    gt_bboxes = self._utils_3d.ResidualsToBBoxes(
        input_batch.anchor_bboxes,
        anchor_localization_residuals,
        min_angle_rad=min_angle_rad,
        max_angle_rad=np.pi)
    predicted_bboxes = self._utils_3d.ResidualsToBBoxes(
        input_batch.anchor_bboxes,
        predicted_residuals,
        min_angle_rad=min_angle_rad,
        max_angle_rad=np.pi)
    dimension_errors_dict = self._BBoxDimensionErrors(gt_bboxes,
                                                      predicted_bboxes,
                                                      reg_weights)
    metrics_dict.update(dimension_errors_dict)

    per_example_dict = {
        'residuals': predicted_residuals,
        'classification_logits': predicted_class_logits,
    }

    return metrics_dict, per_example_dict

  def _BBoxesAndLogits(self, input_batch, predictions):
    """Decode an input batch, computing predicted bboxes from residuals."""
    p = self.params

    # Decode residuals.
    min_angle_rad = -np.pi if p.direction_aware_rot_loss else 0
    predicted_bboxes = self._utils.ResidualsToBBoxes(
        input_batch.anchor_bboxes,
        predictions.residuals,
        min_angle_rad=min_angle_rad,
        max_angle_rad=np.pi)

    # predicted_bboxes is a [batch, nx, ny, nz, na, 7] Tensor.
    batch_size, nx, ny, nz, na, _ = py_utils.GetShape(predicted_bboxes, 6)
    num_boxes = nx * ny * nz * na

    # Reshape to [batch_size, num_boxes, 7]
    predicted_bboxes = tf.reshape(predicted_bboxes, [batch_size, num_boxes, 7])

    classification_logits = tf.reshape(predictions.classification_logits,
                                       [batch_size, num_boxes, -1])

    return py_utils.NestedMap({
        'predicted_bboxes': predicted_bboxes,
        'classification_logits': classification_logits
    })
