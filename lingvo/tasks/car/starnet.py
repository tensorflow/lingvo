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
"""StarNet: A sparse, targeted detection network.

Point clouds and anchors are sampled from the raw points, at centers determined
by a sampler. Each center can have multiple anchors bboxes. Each center's
point cloud is featurized and then used to regress and classify the associated
anchors bboxes.

The featurizer could be range from a simple MLPMax network, to more complex
point-based models such as PointNet++, PointCNN, PCNN, etc.

In V1, anchor_bboxes shared the same featurization around each cell
(featurized_cell), but they had different regression/classification networks
for each offset/dimension/rotation that produced predictions.

In V2, we bring the featurization closer to each anchor_bbox, by featurizing
at each anchor_bbox (offset) location instead. Note that anchor_bboxes at the
same offset, but with different rotation/dimension priors will have the same
featurization.
"""

import enum
import functools
from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.car import builder_lib
from lingvo.tasks.car import detection_3d_lib
from lingvo.tasks.car import geometry
from lingvo.tasks.car import point_detector
import numpy as np


class Builder(builder_lib.ModelBuilderBase):
  """Builder for StarNet Model."""

  def MLPMaxFeaturizer(self, dims):
    """MLP followed by Max Featurizer."""
    return self._Seq('feat', self._GetValue('get_value', 'features'),
                     self._MLP('mlp', dims), self._Max('max'))

  def PaddedMLPMaxFeaturizer(self, idims, dims, use_bn=True):
    """MLP followed by Max on padded points."""

    # TODO(bencaine): Add simple builder test
    # Avoid batch norm in the first layer since our FC layers do BN-FC-Relu
    # So the first op on the raw data would be BN, which is weird for absolute
    # xyz values.
    return self._Seq('feat',
                     self._FeaturesFC('input_fc', idims, dims[0], use_bn=False),
                     self._FeaturesMLP('mlp', dims, use_bn=use_bn),
                     self._PaddedMax('max'))

  def FC(self, name, idims, odims, use_bn=True, activation_fn=tf.nn.relu):
    """Fully-connected layer."""
    return self._FC(name, idims, odims, use_bn, activation_fn)

  def Linear(self, name, idims, odims, params_init=None):
    """Linear layer for predicting residuals and classification logits."""
    return self._Linear(name, idims, odims, params_init)

  def Bias(self, name, dims, params_init=None):
    """Bias layer with optional initialization."""
    return self._Bias(name, dims, params_init)

  def LinearWithBias(self,
                     name,
                     idims,
                     odims,
                     linear_params_init=None,
                     bias_params_init=None):
    """Linear with bias layer with optional initialization."""
    return self._Seq(name,
                     self.Linear('linear', idims, odims, linear_params_init),
                     self.Bias('bias', odims, bias_params_init))

  def Atten(self,
            name,
            depth,
            dims,
            hdims,
            heads,
            odims,
            keep_prob=1.0,
            linear_params_init=None,
            bias_params_init=None):
    """Stacked self-attention followed with a projection."""
    return self._Seq(
        name,
        self._SelfAttenStack('attens', depth, dims, hdims, heads, keep_prob),
        self.LinearWithBias('proj', dims, odims, linear_params_init,
                            bias_params_init))

  def GINFeaturizer(self, name, fc_dims, mlp_dims, num_laser_features=1):
    """GIN-based Featurizer."""
    total_input_features = 3 + num_laser_features
    return self._Seq(
        name,
        # Dropping the cell center xyz prevents the model from over-fitting
        # to the absolute coordinates.
        self._SeqToKey(
            'drop_cell_center_xyz', 'features',
            self._GetValue('get_features', 'features'),
            self._ApplyFn('drop_cell_center_xyz', fn=lambda t: t[..., 3:])),
        self._FeaturesFC('fc0', total_input_features, fc_dims),
        self._GIN(
            'gin',
            mlp_dims,
            aggregate_sub=self._PaddedMax('p_max'),
            readout_sub=self._PaddedMean('p_mean'),
            combine_method='concat'))  # pyformat: disable

  def GINFeaturizerV2(self,
                      name,
                      fc_dims,
                      mlp_dims,
                      num_laser_features=1,
                      fc_use_bn=True):
    """GIN-based Featurizer for Model V2."""
    total_input_features = 3 + num_laser_features
    return self._Seq(
        name,
        self._FeaturesFC(
            'fc0', total_input_features, fc_dims, use_bn=fc_use_bn),
        self._GIN(
            'gin',
            mlp_dims,
            aggregate_sub=self._PaddedMax('p_max'),
            readout_sub=self._PaddedMean('p_mean'),
            combine_method='concat'))  # pyformat: disable

  def ZerosCellFeaturizer(self, name, dims):
    """Produces features with 0 values, used for disabling cell featurizer."""

    def _ZerosFeature(nested_data):
      # Points has shape [batch_size, num_anchors, points_per_anchor, 3].
      # The featurizer should produce features for each anchor.
      shape = py_utils.GetShape(nested_data.points)
      return tf.zeros(shape[:-2] + [dims])

    return self._ApplyFn(name, fn=_ZerosFeature)


class LossNormType(enum.Enum):
  NO_NORM = 0
  NORM_BY_NUM_POS_PER_CENTER = 1


class ModelBase(point_detector.PointDetectorBase):
  """StarNet Detection Model.

  This model expects that input_batch contains:
    cell_center_xyz: [N, C, 3]
    cell_points_xyz: [N, C, P, 3]
    cell_feature: [N, C, P, 1]
    anchor_bboxes: [N, C, B, 7]
    anchor_localization_residuals: [N, C, B, 7]
    assigned_gt_labels: [N, C, B]
    assigned_cls_mask: [N, C, B]

  where:
    N - batch size
    C - num centers
    P - num points per center
    B - num anchor bboxes per center

  The centers for the anchor_bboxes should match those of cell_center_xyz.
  Specifically, num_anchor_bboxes_per_center should match that of the
  corresponding input generator.

  Base class implements common Decoder functions, though they can be
  overridden if desired.

  Sub-classes are expected to implement ComputePredictions.
  """

  @classmethod
  def Params(cls,
             num_classes,
             num_anchor_bboxes_per_center,
             num_laser_features=1):
    p = super().Params(num_classes=num_classes)
    p.Define(
        'num_anchor_bboxes_per_center', num_anchor_bboxes_per_center,
        'The number of anchor bboxes per center. This should match that '
        'of the corresponding input generator.')

    p.Define('focal_loss_alpha', 0.25,
             'Alpha parameter for focal loss for classification.')
    p.Define('focal_loss_gamma', 2.0,
             'Gamma parameter for focal loss for classification.')

    p.Define('huber_loss_delta', 1. / (3.**2),
             'delta threshold for scaled huber loss.')
    p.Define('loss_weight_localization', 2.0,
             'Weighting factor for localization loss.')
    p.Define('loss_weight_classification', 1.0,
             'Weighting factor for classification loss.')

    p.Define('loss_norm_type', None,
             'Normalization type for class and regularization weights.')

    p.Define(
        'squash_rotation_predictions', False,
        'Apply tanh squashing to rotation predictions to ensure outputs '
        'are between (-pi, pi).')
    p.Define(
        'direction_aware_rot_loss', False, 'If True, changes the heading loss '
        'from sin(theta_delta) to WrapAngleRad(theta_delta), which makes the '
        'model produce headings between [-pi to pi].')

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
    p.Define(
        'corner_loss_weight', 0.0,
        'Weight multiplier for contribution of corner loss '
        'to full localization/regression loss')
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

    if len(p.per_class_loss_weight) != p.num_classes:
      raise ValueError('`Need `per_class_loss_weight` to be of len equal '
                       'to the number of classes.')
    if p.per_class_loss_weight[0] != 0.0:
      raise ValueError('Background class should be assigned 0 weight. '
                       'per_class_loss_weight={}'.format(
                           str(p.per_class_loss_weight)))

  def ComputeLoss(self, theta, predictions, input_batch):
    """Compute loss for the sparse detector model v1.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      predictions: A `.NestedMap` object containing residuals and
        classification_logits.
      input_batch: A `.NestedMap` expected to contain cell_center_xyz,
        cell_points_xyz, cell_feature, anchor_bboxes,
        anchor_localization_residuals, assigned_gt_labels, and
        assigned_cls_mask. See class doc string for details.

    Returns:
      Two dicts:

      - A dict containing str keys and (metric, weight) pairs as values, where
        one of the keys is expected to be 'loss'.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    p = self.params

    batch_size, num_centers = py_utils.GetShape(input_batch.cell_center_xyz, 2)

    # Assert shapes of inputs.
    anchor_bboxes = py_utils.HasShape(
        input_batch.anchor_bboxes,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center, 7])
    anchor_localization_residuals = py_utils.HasShape(
        input_batch.anchor_localization_residuals,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center, 7])
    predicted_residuals = py_utils.HasShape(
        predictions.residuals,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center, 7])

    assigned_gt_labels = py_utils.HasShape(
        input_batch.assigned_gt_labels,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center])
    predicted_classification_logits = py_utils.HasShape(
        predictions.classification_logits, [
            batch_size, num_centers, p.num_anchor_bboxes_per_center,
            p.num_classes
        ])

    # assigned_cls_mask is for weighting the classification loss.
    # Ignored targets will have their mask = 0; this happens when their IOU is
    # not high enough to be a foreground object and not low enough to be
    # background.
    class_weights = py_utils.HasShape(
        input_batch.assigned_cls_mask,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center])
    class_weights = tf.reshape(
        class_weights,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center, 1])

    # Broadcast per class loss weights. For each anchor, there are num_classes
    # prediction heads, we weight the outputs of these heads by the per class
    # loss weights.
    per_class_loss_weight = tf.constant([[[p.per_class_loss_weight]]],
                                        dtype=tf.float32)
    per_class_loss_weight = py_utils.HasShape(per_class_loss_weight,
                                              [1, 1, 1, p.num_classes])
    class_weights *= per_class_loss_weight
    class_weights = py_utils.HasShape(class_weights, [
        batch_size, num_centers, p.num_anchor_bboxes_per_center, p.num_classes
    ])

    # We use assigned_reg_mask for masking the regression loss.
    # Only foreground objects will have assigned_reg_mask = 1.
    reg_weights = py_utils.HasShape(
        input_batch.assigned_reg_mask,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center])
    reg_weights = tf.reshape(
        reg_weights,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center, 1])

    if p.loss_norm_type == LossNormType.NORM_BY_NUM_POS_PER_CENTER:
      # Compute number of positive anchors per example.
      foreground_mask = py_utils.HasShape(
          input_batch.assigned_reg_mask,
          [batch_size, num_centers, p.num_anchor_bboxes_per_center])

      # Sum to get the number of foreground anchors for each example.
      loss_normalization = tf.reduce_sum(foreground_mask, axis=2)
      loss_normalization = tf.maximum(loss_normalization,
                                      tf.ones_like(loss_normalization))

      # Reshape for broadcasting.
      loss_normalization = tf.reshape(loss_normalization,
                                      [batch_size, num_centers, 1, 1])

      # Normalize so that the loss is independent of # centers.
      loss_normalization *= num_centers
      class_weights /= loss_normalization
      reg_weights /= loss_normalization

    classification_loss = py_utils.SigmoidCrossEntropyFocalLoss(
        logits=predicted_classification_logits,
        labels=tf.one_hot(assigned_gt_labels, p.num_classes),
        alpha=p.focal_loss_alpha,
        gamma=p.focal_loss_gamma)

    # Apply mask.
    classification_loss *= class_weights

    # TODO(jngiam): Consider normalizing by num_foreground_anchors for each
    # example instead. This would match the 1/N_positive normalization in
    # point pillars.

    # Reduce sum over centers, boxes and classes.
    classification_loss = tf.reduce_sum(classification_loss, axis=[1, 2, 3])

    # Reduce mean over batch.
    classification_loss = tf.reduce_mean(classification_loss)

    # Localization regression loss with Huber loss (SmoothL1).
    regression_loc_and_dims_loss = self._utils_3d.ScaledHuberLoss(
        labels=anchor_localization_residuals[..., :6],
        predictions=predicted_residuals[..., :6],
        delta=p.huber_loss_delta)

    # Rotation loss is computed on a transform on rotation_delta. For a
    # direction aware loss, we simply wrap the angles to -pi to pi; for a loss
    # that is symmetric to direction (i.e., rotating by pi), we use a sin
    # transform.
    rotation_delta_transform = tf.sin
    if p.direction_aware_rot_loss:
      rotation_delta_transform = functools.partial(
          geometry.WrapAngleRad, min_val=-np.pi, max_val=np.pi)
    rotation_delta = (
        predicted_residuals[..., 6:] - anchor_localization_residuals[..., 6:])
    regression_rotation_loss = self._utils_3d.ScaledHuberLoss(
        labels=tf.zeros_like(rotation_delta),
        predictions=rotation_delta_transform(rotation_delta),
        delta=p.huber_loss_delta)

    reg_loc_loss = regression_loc_and_dims_loss[..., :3]
    reg_dim_loss = regression_loc_and_dims_loss[..., 3:6]

    gt_bboxes = self._utils_3d.ResidualsToBBoxes(
        anchor_bboxes,
        anchor_localization_residuals,
        min_angle_rad=-np.pi,
        max_angle_rad=np.pi)
    predicted_bboxes = self._utils_3d.ResidualsToBBoxes(
        anchor_bboxes,
        predicted_residuals,
        min_angle_rad=-np.pi,
        max_angle_rad=np.pi)

    # Apply mask to individual losses.
    #
    # And then reduce sum over centers, boxes, residuals, and batch
    # and divide by the batch_size.
    regression_rotation_loss *= reg_weights
    reg_rot_loss = tf.reduce_sum(regression_rotation_loss) / batch_size

    reg_loc_loss *= reg_weights
    reg_loc_loss = tf.reduce_sum(reg_loc_loss) / batch_size

    reg_dim_loss *= reg_weights
    reg_dim_loss = tf.reduce_sum(reg_dim_loss) / batch_size

    # Do not create corner loss graph if weight is 0.0
    # TODO(bcyang): Remove condition after fixing corner loss NaN issue
    if p.corner_loss_weight != 0.0:
      reg_corner_loss = self._utils_3d.CornerLoss(
          gt_bboxes=gt_bboxes, predicted_bboxes=predicted_bboxes)
      reg_corner_loss = tf.expand_dims(reg_corner_loss, axis=-1)

      reg_corner_loss *= reg_weights
      reg_corner_loss = tf.reduce_sum(reg_corner_loss) / batch_size
    else:
      reg_corner_loss = 0.0

    # Sum components of regression loss.
    regression_loss = (
        p.location_loss_weight * reg_loc_loss +
        p.dimension_loss_weight * reg_dim_loss +
        p.rotation_loss_weight * reg_rot_loss +
        p.corner_loss_weight * reg_corner_loss)

    # Compute total loss.
    total_loss = (
        p.loss_weight_localization * regression_loss +
        p.loss_weight_classification * classification_loss)

    metrics_dict = {
        'loss': (total_loss, batch_size),
        'loss/regression': (regression_loss, batch_size),
        'loss/regression/loc': (reg_loc_loss, batch_size),
        'loss/regression/dim': (reg_dim_loss, batch_size),
        'loss/regression/rot': (reg_rot_loss, batch_size),
        'loss/regression/corner': (reg_corner_loss, batch_size),
        'loss/classification': (classification_loss, batch_size),
    }

    # Calculate dimension errors
    dimension_errors_dict = self._BBoxDimensionErrors(gt_bboxes,
                                                      predicted_bboxes,
                                                      reg_weights)
    metrics_dict.update(dimension_errors_dict)

    per_example_dict = {
        'residuals': predicted_residuals,
        'classification_logits': predicted_classification_logits,
        'predicted_bboxes': predicted_bboxes,
        'gt_bboxes': gt_bboxes,
        'reg_weights': reg_weights,
    }

    return metrics_dict, per_example_dict

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions for `input_batch`.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` expected to contain cell_center_xyz,
        cell_points_xyz, cell_feature, anchor_bboxes,
        anchor_localization_residuals, assigned_gt_labels, and
        assigned_cls_mask. See class doc string for details.

    Returns:
      A `.NestedMap` object containing residuals and classification_logits.
    """
    raise NotImplementedError('Abstract method: %s' % type(self))

  def _BBoxesAndLogits(self, input_batch, predictions):
    """Decode an input batch, computing predicted bboxes from residuals."""
    batch_size, num_centers, num_predictions_per_center = py_utils.GetShape(
        predictions.residuals, 3)

    # Decode residuals.
    predicted_bboxes = self._utils_3d.ResidualsToBBoxes(
        input_batch.anchor_bboxes,
        predictions.residuals,
        min_angle_rad=-np.pi,
        max_angle_rad=np.pi)

    # Reshape to [batch_size, num_centers * num_predictions_per_center, ...]
    num_predicted_boxes = num_centers * num_predictions_per_center
    predicted_bboxes = tf.reshape(predicted_bboxes,
                                  [batch_size, num_predicted_boxes, -1])

    classification_logits = tf.reshape(predictions.classification_logits,
                                       [batch_size, num_predicted_boxes, -1])

    return py_utils.NestedMap({
        'predicted_bboxes': predicted_bboxes,
        'classification_logits': classification_logits,
    })


class ModelV1(ModelBase):
  """StarNet Model V1.

  In this model, each center is first featurized into a single feature vector,
  which is then used to predict the residuals for all the bboxes at that center.
  Concretely, each featurized center needs to make B * (7 + num_classes)
  predictions.

  Effectively, anchor_bboxes shared the same featurization around each cell
  (featurized_cell), but have different regression/classification networks
  for each offset/dimension/rotation that produced predictions.
  """

  @classmethod
  def Params(cls,
             num_classes,
             num_anchor_bboxes_per_center,
             num_laser_features=1):

    p = super().Params(
        num_classes=num_classes,
        num_anchor_bboxes_per_center=num_anchor_bboxes_per_center,
        num_laser_features=num_laser_features)

    builder = Builder()
    final_feature_dims = 1024
    # The first 7 here corresponds to the concatenation of the following
    # features:
    #   x, y, z of the center.
    #   x, y, z of each point relative to the center.
    #   features of the point.
    num_features = 6 + num_laser_features
    p.Define(
        'cell_featurizer',
        builder.MLPMaxFeaturizer([num_features, 32, 256, final_feature_dims]),
        'Point cloud feature extractor.')

    p.Define(
        'localization_regressor',
        builder.Linear('localization', final_feature_dims,
                       num_anchor_bboxes_per_center * 7),
        'Localization residual regressor.')
    p.Define(
        'classifier',
        builder.Linear('classification', final_feature_dims,
                       num_anchor_bboxes_per_center * num_classes),
        'Classification layer (producing logits).')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('cell_featurizer', p.cell_featurizer)
    self.CreateChild('localization_regressor', p.localization_regressor)
    self.CreateChild('classifier', p.classifier)

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions for `input_batch`.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` expected to contain cell_center_xyz,
        cell_points_xyz, cell_feature, anchor_bboxes,
        anchor_localization_residuals, assigned_gt_labels, and
        assigned_cls_mask. See class doc string for details.

    Returns:
      A `.NestedMap` object containing residuals and classification_logits.
    """
    p = self.params
    input_batch.Transform(lambda x: (x.shape, x.shape.num_elements())).VLog(
        1, 'input_batch shapes: ')

    cell_feature = py_utils.HasRank(input_batch.cell_feature, 4)
    batch_size, num_centers, num_points_per_cell = py_utils.GetShape(
        cell_feature, 3)

    cell_points_xyz = py_utils.HasShape(
        input_batch.cell_points_xyz,
        [batch_size, num_centers, num_points_per_cell, 3])
    cell_center_xyz = py_utils.HasShape(input_batch.cell_center_xyz,
                                        [batch_size, num_centers, 3])

    cell_points_padding = py_utils.HasShape(
        input_batch.cell_points_padding,
        [batch_size, num_centers, num_points_per_cell])

    # TODO(jngiam): Make concat_feature computation a layer or configureable.
    cell_center_xyz = tf.reshape(cell_center_xyz,
                                 [batch_size, num_centers, 1, 3])
    centered_cell_points_xyz = cell_points_xyz - cell_center_xyz
    concat_feature = tf.concat([
        tf.tile(cell_center_xyz, [1, 1, num_points_per_cell, 1]),
        centered_cell_points_xyz, cell_feature
    ], axis=-1)  # pyformat: disable

    # Featurize point clouds at each center.
    point_input = py_utils.NestedMap({
        'points': centered_cell_points_xyz,
        'features': concat_feature,
        'padding': cell_points_padding,
    })
    featurized_cell = self.cell_featurizer.FProp(theta.cell_featurizer,
                                                 point_input)
    featurized_cell = py_utils.HasShape(featurized_cell,
                                        [batch_size, num_centers, -1])

    # Predict localization residuals.
    predicted_residuals = self.localization_regressor.FProp(
        theta.localization_regressor, featurized_cell)
    predicted_residuals = tf.reshape(
        predicted_residuals,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center, 7])

    if p.squash_rotation_predictions:
      predicted_rotations = predicted_residuals[..., 6:]
      predicted_rotations = np.pi * tf.tanh(predicted_rotations)
      predicted_residuals = tf.concat(
          [predicted_residuals[..., :6], predicted_rotations], axis=-1)

    # Predict object classification at each bbox.
    predicted_classification_logits = self.classifier.FProp(
        theta.classifier, featurized_cell)
    predicted_classification_logits = tf.reshape(
        predicted_classification_logits, [
            batch_size, num_centers, p.num_anchor_bboxes_per_center,
            p.num_classes
        ])

    return py_utils.NestedMap({
        'residuals': predicted_residuals,
        'classification_logits': predicted_classification_logits,
    })


class ModelV2(ModelBase):
  """StarNet Model V2.

  This model is similar to V1 except that featurizations are computed at the
  location of each anchor_bbox instead of each cell center.

  In V2, we don't share the featurization among anchor_bboxes in the same cell.
  Instead, the model featurizes at each anchor_bbox (offset) location.
  Note that anchor_bboxes at the same offset, but with different
  rotation/dimension priors will have the same featurization.

  Note: This model makes assumptions about the ordering of anchor_bboxes, and
  assumes that offsets correspond to the 'outer dimensions'; see
  input_generator._AnchorBoxSettings.GenerateAnchorSettings for details.

  Note that the different rotation/dimension settings will have their own
  classification/regression heads - and these are now shared across all offsets.

  In summary, we want the featurizer to be location (xyz coordinate) specific,
  and at each location, we want to have rotation/dimension specific regressors.
  This makes it more natural to leverage featurizers that produce features
  at these anchor-offset specific locations. For example, one could apply a
  PCNN/PointConv model on the entire point cloud (instead of each cell) and
  produce features at given anchor locations.
  """

  @classmethod
  def Params(cls,
             num_classes,
             num_anchor_bboxes_offsets,
             num_anchor_bboxes_rotations,
             num_anchor_bboxes_dimensions,
             num_laser_features=1):
    num_anchor_bboxes_per_center = (
        num_anchor_bboxes_offsets * num_anchor_bboxes_rotations *
        num_anchor_bboxes_dimensions)
    p = super().Params(num_classes, num_anchor_bboxes_per_center,
                       num_laser_features)

    # Good defaults from V1 tuning.
    p.loss_norm_type = LossNormType.NORM_BY_NUM_POS_PER_CENTER

    # V2 specifics below.
    p.Define(
        'num_anchor_bboxes_offsets', num_anchor_bboxes_offsets,
        'The number of anchor bboxes offsets per center. '
        'This should match that of the corresponding input generator.')
    p.Define(
        'num_anchor_bboxes_rotations', num_anchor_bboxes_rotations,
        'The number of anchor bboxes rotations per center. '
        'This should match that of the corresponding input generator.')
    p.Define(
        'num_anchor_bboxes_dimensions', num_anchor_bboxes_dimensions,
        'The number of anchor bboxes dimensions per center. '
        'This should match that of the corresponding input generator.')

    builder = Builder()
    builder.linear_params_init = py_utils.WeightInit.KaimingUniformFanInRelu()

    # In Model V2, the inputs to the cell featurizer only include the relative
    # coordinates of each point concatenated with the point's features.
    # GINFeaturizerV2 takes that into account and expects input_dims=4.
    num_gin_layers = 5
    gin_hidden_dims = 128
    gin_layers = [
        [gin_hidden_dims * 2, gin_hidden_dims * 4, gin_hidden_dims]
    ] * num_gin_layers  # pyformat: disable

    # Models can specify different cell featurizers as long as cell_feature_dims
    # is also set correctly.
    p.Define(
        'cell_feature_dims', gin_hidden_dims * (num_gin_layers + 1),
        'Dimensions of the features produced by the cell featurizer. '
        'This should match the output of the cell_featurizer.')
    p.Define(
        'cell_featurizer',
        builder.GINFeaturizerV2(
            'feat',
            gin_hidden_dims,
            gin_layers,
            num_laser_features=num_laser_features),
        'Point cloud local cell feature extractor.')

    # The cell_feature_projector layer projects the featurized cells to each
    # offset location. This is mainly to keep compatibility with V1. Note that
    # this may be removed once we find a better global featurizer.
    # A FC layer will automatically be created at init that projects the
    # cell features to each anchor location with this dims.
    p.Define(
        'anchor_projected_feature_dims', 128, 'Dimensions of projected '
        'features from cell featurizer. A FC layer will be used to '
        'project the cell features.')

    p.Define('oracle_location', False,
             'If true, the model predicts the ground truth for location.')
    p.Define('oracle_dimension', False,
             'If true, the model predicts the ground truth for dimension.')
    p.Define('oracle_rotation', False,
             'If true, the model predicts the ground truth for rotation.')
    p.Define(
        'oracle_classification', False,
        'If true, the model predicts the ground truth for classification.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    builder = Builder()
    builder.linear_params_init = py_utils.WeightInit.KaimingUniformFanInRelu()

    num_anchors_per_offset = (
        p.num_anchor_bboxes_rotations * p.num_anchor_bboxes_dimensions)

    # Create params for fixed sub layers.
    cell_feature_projector_p = builder.FC(
        'cell_feature_projector', p.cell_feature_dims,
        p.num_anchor_bboxes_offsets * p.anchor_projected_feature_dims)

    localization_regressor_p = builder.Linear('localization_regressor',
                                              p.anchor_projected_feature_dims,
                                              num_anchors_per_offset * 7,
                                              py_utils.WeightInit.Constant(0.))

    classifier_p = builder.LinearWithBias(
        'classifier',
        p.anchor_projected_feature_dims,
        num_anchors_per_offset * p.num_classes,
        bias_params_init=py_utils.WeightInit.Constant(-4.595))

    self.CreateChild('cell_featurizer', p.cell_featurizer)
    self.CreateChild('cell_feature_projector', cell_feature_projector_p)
    self.CreateChild('localization_regressor', localization_regressor_p)
    self.CreateChild('classifier', classifier_p)

  def _CellFeaturizer(self, theta, input_batch):
    """Featurizes each center location."""
    # Validate Shapes
    cell_feature = py_utils.HasRank(input_batch.cell_feature, 4)
    batch_size, num_centers, num_points_per_cell = py_utils.GetShape(
        cell_feature, 3)

    cell_points_xyz = py_utils.HasShape(
        input_batch.cell_points_xyz,
        [batch_size, num_centers, num_points_per_cell, 3])
    cell_center_xyz = py_utils.HasShape(input_batch.cell_center_xyz,
                                        [batch_size, num_centers, 3])

    cell_points_padding = py_utils.HasShape(
        input_batch.cell_points_padding,
        [batch_size, num_centers, num_points_per_cell])

    # Center each cell
    cell_center_xyz = tf.reshape(cell_center_xyz,
                                 [batch_size, num_centers, 1, 3])
    centered_cell_points_xyz = cell_points_xyz - cell_center_xyz
    concat_feature = tf.concat([
        centered_cell_points_xyz, cell_feature
    ], axis=-1)  # pyformat: disable

    # Featurize point clouds at each center.
    point_input = py_utils.NestedMap({
        'points': centered_cell_points_xyz,
        'features': concat_feature,
        'padding': cell_points_padding,
    })
    featurized_cell = self.cell_featurizer.FProp(theta.cell_featurizer,
                                                 point_input)
    featurized_cell = py_utils.HasShape(featurized_cell,
                                        [batch_size, num_centers, -1])
    return featurized_cell

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions for `input_batch`.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` expected to contain lasers.points_xyz,
        lasers.points_feature, lasers.points_padding, cell_center_xyz,
        cell_points_xyz, cell_feature, anchor_bboxes,
        anchor_localization_residuals, assigned_gt_labels, and
        assigned_cls_mask. See class doc string for details.

    Returns:
      A `.NestedMap` object containing residuals and classification_logits.
    """
    p = self.params
    input_batch.Transform(lambda x: (x.shape, x.shape.num_elements())).VLog(
        1, 'input_batch shapes: ')
    cell_feature = py_utils.HasRank(input_batch.cell_feature, 4)
    batch_size, num_centers = py_utils.GetShape(cell_feature, 2)

    featurized_cell = self._CellFeaturizer(theta, input_batch)

    # Project each featurized_cell features to each bbox per center.
    featurized_anchors = self.cell_feature_projector.FProp(
        theta.cell_feature_projector, featurized_cell)

    # Reshape output so that we have features per offset.
    featurized_anchors = tf.reshape(
        featurized_anchors,
        [batch_size, num_centers, p.num_anchor_bboxes_offsets, -1])

    # Predict localization residuals.
    predicted_residuals = self.localization_regressor.FProp(
        theta.localization_regressor, featurized_anchors)
    predicted_residuals = tf.reshape(
        predicted_residuals,
        [batch_size, num_centers, p.num_anchor_bboxes_per_center, 7])

    if any([p.oracle_location, p.oracle_dimension, p.oracle_rotation]):
      gt_residuals = py_utils.HasShape(
          input_batch.anchor_localization_residuals,
          [batch_size, num_centers, p.num_anchor_bboxes_per_center, 7])
      residuals = []
      if p.oracle_location:
        residuals.append(gt_residuals[..., 0:3])
      else:
        residuals.append(predicted_residuals[..., 0:3])

      if p.oracle_dimension:
        residuals.append(gt_residuals[..., 3:6])
      else:
        residuals.append(predicted_residuals[..., 3:6])

      if p.oracle_rotation:
        residuals.append(gt_residuals[..., 6:])
      else:
        residuals.append(predicted_residuals[..., 6:])
      predicted_residuals = tf.concat(residuals, axis=-1)

    if p.squash_rotation_predictions:
      predicted_rotations = predicted_residuals[..., 6:]
      predicted_rotations = np.pi * tf.tanh(predicted_rotations)
      predicted_residuals = tf.concat(
          [predicted_residuals[..., :6], predicted_rotations], axis=-1)

    # Predict object classification at each bbox.
    predicted_classification_logits = self.classifier.FProp(
        theta.classifier, featurized_anchors)
    predicted_classification_logits = tf.reshape(
        predicted_classification_logits, [
            batch_size, num_centers, p.num_anchor_bboxes_per_center,
            p.num_classes
        ])

    if p.oracle_classification:
      assigned_gt_labels = py_utils.HasShape(
          input_batch.assigned_gt_labels,
          [batch_size, num_centers, p.num_anchor_bboxes_per_center])
      predicted_classification_logits = tf.one_hot(assigned_gt_labels,
                                                   p.num_classes)

    return py_utils.NestedMap({
        'residuals': predicted_residuals,
        'classification_logits': predicted_classification_logits,
    })
