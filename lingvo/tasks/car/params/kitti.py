# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train models on KITTI data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import cluster_factory
from lingvo.core import datasource
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.tasks.car import input_preprocessors
from lingvo.tasks.car import kitti_input_generator
from lingvo.tasks.car import lr_util
from lingvo.tasks.car import starnet
import numpy as np


# Set $KITTI_DIR to the base path of where all the KITTI files can be found.
#
# E.g., 'gs://your-bucket/kitti/3d'
_KITTI_BASE = os.environ.get('KITTI_DIR', 'FILL-ME-IN')


# Specifications for the different dataset splits.
def KITTITrainSpec(params):
  p = params.Copy()
  p.file_datasource.base_datasource.file_pattern = (
      'kitti_object_3dop_train.tfrecord-*-of-00100')
  p.num_samples = 3712
  return p


def KITTIValSpec(params):
  p = params.Copy()
  p.file_datasource.base_datasource.file_pattern = (
      'kitti_object_3dop_val.tfrecord-*-of-00100')
  p.num_samples = 3769
  return p


def KITTITestSpec(params):
  p = params.Copy()
  p.file_datasource.base_datasource.file_pattern = (
      'kitti_object_test.tfrecord-*-of-00100')
  p.num_samples = 7518
  return p


class KITTITrain(kitti_input_generator.KITTILaser):
  """KITTI train set with raw laser data."""

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super(KITTITrain, cls).Params()
    return KITTITrainSpec(p)


class KITTIValidation(kitti_input_generator.KITTILaser):
  """KITTI validation set with raw laser data."""

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super(KITTIValidation, cls).Params()
    return KITTIValSpec(p)


class KITTITest(kitti_input_generator.KITTILaser):
  """KITTI test set with raw laser data."""

  @classmethod
  def Params(cls):
    p = super(KITTITest, cls).Params()
    return KITTITestSpec(p)


class KITTIGridTrain(kitti_input_generator.KITTIGrid):
  """KITTI train set with grid laser data."""

  @classmethod
  def Params(cls):
    p = super(KITTIGridTrain, cls).Params()
    return KITTITrainSpec(p)


class KITTIGridValidation(kitti_input_generator.KITTIGrid):
  """KITTI validation set with grid laser data."""

  @classmethod
  def Params(cls):
    p = super(KITTIGridValidation, cls).Params()
    return KITTIValSpec(p)


class KITTIGridTest(kitti_input_generator.KITTIGrid):
  """KITTI validation set with grid laser data."""

  @classmethod
  def Params(cls):
    p = super(KITTIGridTest, cls).Params()
    return KITTITestSpec(p)


class KITTISparseLaserTrain(kitti_input_generator.KITTISparseLaser):
  """KITTI train set with sparse laser data."""

  @classmethod
  def Params(cls):
    p = super(KITTISparseLaserTrain, cls).Params()
    return KITTITrainSpec(p)


class KITTISparseLaserValidation(kitti_input_generator.KITTISparseLaser):
  """KITTI validation set with sparse laser data."""

  @classmethod
  def Params(cls):
    p = super(KITTISparseLaserValidation, cls).Params()
    return KITTIValSpec(p)


class KITTISparseLaserTest(kitti_input_generator.KITTISparseLaser):
  """KITTI test set with sparse laser data."""

  @classmethod
  def Params(cls):
    p = super(KITTISparseLaserTest, cls).Params()
    return KITTITestSpec(p)


def _MaybeRemove(values, key):
  """Remove the entry 'key' from 'values' if present."""
  if key in values:
    values.remove(key)


def AddLaserAndCamera(params):
  """Adds laser and camera extractors."""
  cluster = cluster_factory.Current()
  job = cluster.job
  if job != 'decoder':
    return params

  extractor_params = dict(params.extractors.IterParams()).values()
  extractor_classes = [p.cls for p in extractor_params]

  # Add images if not present.
  if kitti_input_generator.KITTIImageExtractor not in extractor_classes:
    params.extractors.Define('images',
                             kitti_input_generator.KITTIImageExtractor.Params(),
                             '')

  # Add raw lasers if not present.
  if kitti_input_generator.KITTILaserExtractor not in extractor_classes:
    labels = None
    for p in extractor_params:
      if p.cls == kitti_input_generator.KITTILabelExtractor:
        labels = p
    if labels is None:
      labels = kitti_input_generator.KITTILabelExtractor.Params()
    params.extractors.Define(
        'lasers', kitti_input_generator.KITTILaserExtractor.Params(labels), '')

  return params


################################################################################
# StarNet
################################################################################
@model_registry.RegisterSingleTaskModel
class StarNetCarsBase(base_model_params.SingleTaskModelParams):
  """StarNet model for cars."""
  RUN_LOCALLY = False
  NUM_ANCHOR_BBOX_OFFSETS = 25
  NUM_ANCHOR_BBOX_ROTATIONS = 4
  NUM_ANCHOR_BBOX_DIMENSIONS = 1
  FOREGROUND_ASSIGNMENT_THRESHOLD = 0.6
  BACKGROUND_ASSIGNMENT_THRESHOLD = 0.45
  INCLUDED_CLASSES = ['Car']

  class AnchorBoxSettings(input_preprocessors.SparseCarV1AnchorBoxSettings):
    ROTATIONS = [0, np.pi / 2, 3. * np.pi / 4, np.pi / 4]

  def _configure_input(self, p):
    """Base function managing the delegation of job specific input configs."""
    self._configure_generic_input(p)
    cluster = cluster_factory.Current()
    job = cluster.job
    if job.startswith('trainer'):
      self._configure_trainer_input(p)
    elif job.startswith('decoder'):
      self._configure_decoder_input(p)
    elif job.startswith('evaler'):
      self._configure_evaler_input(p)
    else:
      tf.logging.info('There are no input configuration changes to for '
                      'job {}.'.format(job))
    if self.RUN_LOCALLY:
      p.num_batcher_threads = 1
      p.file_buffer_size = 1
      p.file_parallelism = 1

  def _configure_generic_input(self, p):
    """Update input_config `p` for all jobs."""
    p.file_datasource.file_pattern_prefix = _KITTI_BASE

    # Perform frustum dropping before ground removal (keep_xyz_range).
    p.preprocessors.Define(
        'remove_out_of_frustum',
        (input_preprocessors.KITTIDropPointsOutOfFrustum.Params()), '')
    p.preprocessors_order.insert(
        p.preprocessors_order.index('keep_xyz_range'), 'remove_out_of_frustum')

    # Approximate ground removal.
    p.preprocessors.keep_xyz_range.keep_z_range = (-1.35, np.inf)

    # Max num points can be smaller since we have dropped points out of frustum.
    p.preprocessors.pad_lasers.max_num_points = 32768

    # TODO(jngiam): Analyze if these settings are optimal.
    p.preprocessors.select_centers.num_cell_centers = 256
    p.preprocessors.gather_features.num_points_per_cell = 512
    p.preprocessors.gather_features.sample_neighbors_uniformly = True
    p.preprocessors.gather_features.max_distance = 3.0

    p.preprocessors.assign_anchors.foreground_assignment_threshold = (
        self.FOREGROUND_ASSIGNMENT_THRESHOLD)
    p.preprocessors.assign_anchors.background_assignment_threshold = (
        self.BACKGROUND_ASSIGNMENT_THRESHOLD)

    # Apply car anchor box settings.
    tile_anchors_p = p.preprocessors.tile_anchors
    self.AnchorBoxSettings.Update(p.preprocessors.tile_anchors)
    num_anchor_configs = (
        self.NUM_ANCHOR_BBOX_OFFSETS * self.NUM_ANCHOR_BBOX_ROTATIONS *
        self.NUM_ANCHOR_BBOX_DIMENSIONS)

    assert len(tile_anchors_p.anchor_box_dimensions) == num_anchor_configs
    assert len(tile_anchors_p.anchor_box_rotations) == num_anchor_configs
    assert len(tile_anchors_p.anchor_box_offsets) == num_anchor_configs

    # Filter label extractor for anchors and visualization.
    if 'labels' in p.extractors:
      filtered_labels = [
          kitti_input_generator.KITTILabelExtractor.KITTI_CLASS_NAMES.index(
              class_name) for class_name in self.INCLUDED_CLASSES
      ]
      p.extractors.labels.filter_labels = filtered_labels
    p = AddLaserAndCamera(p)

  def _configure_trainer_input(self, p):
    """Update input_config `p` for jobs running training."""
    # TODO(bencaine): Change the default in input_generator to be False
    # and only set this true in _configure_decoder_input
    p.extractors.images.decode_image = False
    _MaybeRemove(p.preprocessors_order, 'count_points')
    _MaybeRemove(p.preprocessors_order, 'viz_copy')
    p.preprocessors.Define(
        'rot_box', (input_preprocessors.RandomBBoxTransform.Params().Set(
            max_rotation=np.pi / 20.)), '')
    p.preprocessors.Define('random_flip',
                           input_preprocessors.RandomFlipY.Params(), '')
    p.preprocessors.Define(
        'global_rot',
        (input_preprocessors.RandomWorldRotationAboutZAxis.Params().Set(
            max_rotation=np.pi / 4.)), '')
    p.preprocessors.Define(
        'world_scaling',
        (input_preprocessors.WorldScaling.Params().Set(scaling=[0.95, 1.05])),
        '')

    # Do per object transforms, then random flip, then global rotation, then
    # global scaling.
    preprocessor_order = [
        'rot_box', 'random_flip', 'global_rot', 'world_scaling'
    ]
    insert_index = p.preprocessors_order.index('select_centers')
    p.preprocessors_order = (
        p.preprocessors_order[:insert_index] + preprocessor_order +
        p.preprocessors_order[insert_index:])

    # Add ground truth augmenter to before all preprocessors.
    allowed_label_ids = [
        kitti_input_generator.KITTILabelExtractor.KITTI_CLASS_NAMES.index(
            class_name) for class_name in self.INCLUDED_CLASSES
    ]
    groundtruth_db = datasource.PrefixedDataSourceWrapper.Params()
    groundtruth_db.file_pattern_prefix = _KITTI_BASE
    groundtruth_db.base_datasource = datasource.SimpleDataSource.Params()
    groundtruth_db.base_datasource.file_pattern = (
        'kitti_train_object_cls.tfrecord-*-of-00100')

    p.preprocessors.Define(
        'bbox_aug', (input_preprocessors.GroundTruthAugmentor.Params().Set(
            groundtruth_database=groundtruth_db,
            num_db_objects=19700,
            filter_min_points=5,
            max_augmented_bboxes=15,
            label_filter=allowed_label_ids,
        )), '')
    p.preprocessors_order = ['bbox_aug'] + p.preprocessors_order

    p.preprocessors.Define('frustum_dropout',
                           (input_preprocessors.FrustumDropout.Params().Set(
                               theta_width=0.03, phi_width=0.0)), '')
    p.preprocessors_order.insert(
        p.preprocessors_order.index('gather_features'), 'frustum_dropout')

    p.batch_size = 2
    p.file_parallelism = 64
    p.num_batcher_threads = 64

  def _configure_decoder_input(self, p):
    """Update input_config `p` for jobs running decoding."""
    p.batch_size = 4
    p.file_parallelism = 8
    p.num_batcher_threads = 8
    p.file_buffer_size = 500

  def _configure_evaler_input(self, p):
    """Update input_config `p` for jobs running evaluation."""
    # TODO(bencaine): Change the default in input_generator to be False
    # and only set this true in _configure_decoder_input
    p.extractors.images.decode_image = False
    _MaybeRemove(p.preprocessors_order, 'count_points')
    _MaybeRemove(p.preprocessors_order, 'viz_copy')
    p.batch_size = 4
    p.file_parallelism = 8
    p.num_batcher_threads = 8
    p.file_buffer_size = 500

  def Train(self):
    p = KITTISparseLaserTrain.Params()
    self._configure_input(p)
    return p

  def Test(self):
    p = KITTISparseLaserTest.Params()
    self._configure_input(p)
    return p

  def Dev(self):
    p = KITTISparseLaserValidation.Params()
    self._configure_input(p)
    return p

  def Task(self):
    num_classes = len(
        kitti_input_generator.KITTILabelExtractor.KITTI_CLASS_NAMES)
    p = starnet.ModelV2.Params(
        num_classes,
        num_anchor_bboxes_offsets=self.NUM_ANCHOR_BBOX_OFFSETS,
        num_anchor_bboxes_rotations=self.NUM_ANCHOR_BBOX_ROTATIONS,
        num_anchor_bboxes_dimensions=self.NUM_ANCHOR_BBOX_DIMENSIONS)

    p.name = 'sparse_detector'

    tp = p.train
    tp.optimizer = optimizer.Adam.Params()
    tp.clip_gradient_norm_to_value = 5

    ep = p.eval
    # Evaluate the whole dataset.
    ep.samples_per_summary = 0

    # To be tuned.
    p.train.l2_regularizer_weight = 1e-4

    # Adapted from V1 tuning.
    tp.ema_decay = 0.99
    tp.learning_rate = 0.001
    lr_util.SetExponentialLR(
        train_p=tp,
        train_input_p=self.Train(),
        exp_start_epoch=150,
        total_epoch=650)

    p.dimension_loss_weight = .3
    p.location_loss_weight = 3.
    p.loss_weight_classification = 1.
    p.loss_weight_localization = 3.
    p.rotation_loss_weight = 0.3

    return p


@model_registry.RegisterSingleTaskModel
class StarNetCarModel0701(StarNetCarsBase):
  """StarNet Car model trained on KITTI."""

  class AnchorBoxSettings(input_preprocessors.SparseCarV1AnchorBoxSettings):
    CENTER_X_OFFSETS = np.linspace(-1.294, 1.294, 5)
    CENTER_Y_OFFSETS = np.linspace(-1.294, 1.294, 5)

  def _configure_generic_input(self, p):
    super(StarNetCarModel0701, self)._configure_generic_input(p)
    # For selecting centers, drop points out of frustum and do approximate
    # ground removal.
    p.preprocessors.select_centers.features_preparation_layers = [
        input_preprocessors.KITTIDropPointsOutOfFrustum.Params(),
        input_preprocessors.DropLaserPointsOutOfRange.Params().Set(
            keep_z_range=(-1., np.inf)),
    ]

    # Remove frustum dropping from original preprocessors.
    p.preprocessors_order.remove('remove_out_of_frustum')

    # Keep all points in front of the car for featurizing, do not remove ground.
    p.preprocessors.keep_xyz_range.keep_x_range = (0., np.inf)
    p.preprocessors.keep_xyz_range.keep_y_range = (-40., 40.)
    p.preprocessors.keep_xyz_range.keep_z_range = (-np.inf, np.inf)
    p.preprocessors.pad_lasers.max_num_points = 72000

    p.preprocessors.select_centers.sampling_method = 'farthest_point'

    p.preprocessors.select_centers.num_cell_centers = 768

    p.preprocessors.gather_features.max_distance = 3.75

    p.preprocessors.assign_anchors.foreground_assignment_threshold = 0.567087
    # Disable ignore class, by setting background threshold > foreground.
    p.preprocessors.assign_anchors.background_assignment_threshold = 1.0

    p.preprocessors.select_centers.features_preparation_layers = [
        input_preprocessors.KITTIDropPointsOutOfFrustum.Params(),
        input_preprocessors.DropLaserPointsOutOfRange.Params().Set(
            keep_z_range=(-1.4, np.inf)),
    ]

  def _configure_trainer_input(self, p):
    super(StarNetCarModel0701, self)._configure_trainer_input(p)

    p.preprocessors.Define(
        'global_loc_noise',
        (input_preprocessors.GlobalTranslateNoise.Params().Set(
            noise_std=[0., 0., 0.35])), '')
    p.preprocessors_order.insert(
        p.preprocessors_order.index('world_scaling') + 1, 'global_loc_noise')

  def Task(self):
    p = super(StarNetCarModel0701, self).Task()

    # Builder configuration.
    builder = starnet.Builder()
    builder.linear_params_init = py_utils.WeightInit.KaimingUniformFanInRelu()
    gin_layer_sizes = [32, 256, 512, 256, 256, 128]
    num_laser_features = 1
    gin_layers = [
        # Each layer should expect as input - 2 * dims of the last layer's
        # output. We assume a middle layer that's the size of 2 * dim_out.
        [dim_in * 2, dim_out * 2, dim_out]
        for (dim_in, dim_out) in zip(gin_layer_sizes[:-1], gin_layer_sizes[1:])
    ]
    p.cell_feature_dims = sum(gin_layer_sizes)
    p.cell_featurizer = builder.GINFeaturizerV2(
        name='feat',
        fc_dims=gin_layer_sizes[0],
        mlp_dims=gin_layers,
        num_laser_features=num_laser_features,
        fc_use_bn=False)
    p.anchor_projected_feature_dims = 512

    # Loss and training params
    p.train.learning_rate = 0.001 / 2.  # Divide by batch size.
    p.focal_loss_alpha = 0.2
    p.focal_loss_gamma = 3.0
    class_name_to_idx = kitti_input_generator.KITTILabelExtractor.KITTI_CLASS_NAMES
    num_classes = len(class_name_to_idx)
    p.per_class_loss_weight = [0.] * num_classes
    p.per_class_loss_weight[class_name_to_idx.index('Car')] = 1.

    # Decoding / NMS params.
    p.use_oriented_per_class_nms = True
    p.max_nms_boxes = 512
    p.nms_iou_threshold = [0.0] * num_classes
    p.nms_iou_threshold[class_name_to_idx.index('Car')] = 0.0831011
    p.nms_score_threshold = [1.0] * num_classes
    p.nms_score_threshold[class_name_to_idx.index('Car')] = 0.321310
    p.output_decoder.truncation_threshold = 0.65
    p.output_decoder.filter_predictions_outside_frustum = True
    return p


@model_registry.RegisterSingleTaskModel
class StarNetPedCycModel0704(StarNetCarsBase):
  """StarNet Ped/Cyc model trained on KITTI."""

  INCLUDED_CLASSES = ['Pedestrian', 'Cyclist']

  FOREGROUND_ASSIGNMENT_THRESHOLD = 0.48
  # Any value > FOREGROUND is equivalent.
  BACKGROUND_ASSIGNMENT_THRESHOLD = 0.80

  NUM_ANCHOR_BBOX_OFFSETS = 9
  NUM_ANCHOR_BBOX_ROTATIONS = 4
  NUM_ANCHOR_BBOX_DIMENSIONS = 3

  class AnchorBoxSettings(input_preprocessors.SparseCarV1AnchorBoxSettings):
    # PointPillars priors for pedestrian/cyclists.
    DIMENSION_PRIORS = [(0.6, 0.8, 1.7), (0.6, 0.6, 1.2), (0.6, 1.76, 1.73)]
    ROTATIONS = [0, np.pi / 2, 3. * np.pi / 4, np.pi / 4]
    CENTER_X_OFFSETS = np.linspace(-0.31, 0.31, 3)
    CENTER_Y_OFFSETS = np.linspace(-0.31, 0.31, 3)
    CENTER_Z_OFFSETS = [-0.6]

  def _configure_generic_input(self, p):
    super(StarNetPedCycModel0704, self)._configure_generic_input(p)
    # For selecting centers, drop points out of frustum and do approximate
    # ground removal.
    p.preprocessors.select_centers.features_preparation_layers = [
        input_preprocessors.KITTIDropPointsOutOfFrustum.Params(),
        input_preprocessors.DropLaserPointsOutOfRange.Params().Set(
            keep_z_range=(-1., np.inf)),
    ]

    # Remove frustum dropping from original preprocessors.
    p.preprocessors_order.remove('remove_out_of_frustum')

    # Keep all points in front of the car for featurizing, do not remove ground.
    p.preprocessors.keep_xyz_range.keep_x_range = (0., 48.0)
    p.preprocessors.keep_xyz_range.keep_y_range = (-20., 20.)
    p.preprocessors.keep_xyz_range.keep_z_range = (-np.inf, np.inf)
    p.preprocessors.pad_lasers.max_num_points = 72000
    p.preprocessors.select_centers.sampling_method = 'farthest_point'
    p.preprocessors.select_centers.num_cell_centers = 512
    p.preprocessors.select_centers.features_preparation_layers = [
        input_preprocessors.KITTIDropPointsOutOfFrustum.Params(),
        input_preprocessors.DropLaserPointsOutOfRange.Params().Set(
            keep_z_range=(-1.4, np.inf)),
    ]

    p.preprocessors.gather_features.max_distance = 2.55

  def _configure_trainer_input(self, p):
    super(StarNetPedCycModel0704, self)._configure_trainer_input(p)

    allowed_label_ids = [
        kitti_input_generator.KITTILabelExtractor.KITTI_CLASS_NAMES.index(
            class_name) for class_name in self.INCLUDED_CLASSES
    ]
    p.preprocessors.bbox_aug.Set(
        num_db_objects=19700,
        filter_min_difficulty=2,
        filter_min_points=7,
        max_augmented_bboxes=2,
        max_num_points_per_bbox=1558,
        label_filter=allowed_label_ids,
    )
    p.batch_size = 2

  def _configure_decoder_input(self, p):
    """Update input_config `p` for jobs running decoding."""
    super(StarNetPedCycModel0704, self)._configure_decoder_input(p)
    p.batch_size = 4

  def _configure_evaler_input(self, p):
    """Update input_config `p` for jobs running evaluation."""
    super(StarNetPedCycModel0704, self)._configure_evaler_input(p)
    p.batch_size = 4

  def Task(self):
    p = super(StarNetPedCycModel0704, self).Task()
    p.train.learning_rate = 7e-4

    builder = starnet.Builder()
    builder.linear_params_init = py_utils.WeightInit.KaimingUniformFanInRelu()
    gin_layer_sizes = [32, 256, 512, 256, 256, 128]
    num_laser_features = 1
    gin_layers = [
        # Each layer should expect as input - 2 * dims of the last layer's
        # output. We assume a middle layer that's the size of 2 * dim_out.
        [dim_in * 2, dim_out * 2, dim_out]
        for (dim_in, dim_out) in zip(gin_layer_sizes[:-1], gin_layer_sizes[1:])
    ]
    p.cell_feature_dims = sum(gin_layer_sizes)
    # Disable BN on first layer
    p.cell_featurizer = builder.GINFeaturizerV2(
        'feat',
        gin_layer_sizes[0],
        gin_layers,
        num_laser_features,
        fc_use_bn=False)
    p.anchor_projected_feature_dims = 512

    class_name_to_idx = kitti_input_generator.KITTILabelExtractor.KITTI_CLASS_NAMES
    num_classes = len(class_name_to_idx)
    p.per_class_loss_weight = [0.] * num_classes
    p.per_class_loss_weight[class_name_to_idx.index('Pedestrian')] = 3.5
    p.per_class_loss_weight[class_name_to_idx.index('Cyclist')] = 3.25

    p.focal_loss_alpha = 0.9
    p.focal_loss_gamma = 1.25

    p.use_oriented_per_class_nms = True
    p.max_nms_boxes = 1024
    p.nms_iou_threshold = [0.0] * num_classes
    p.nms_iou_threshold[class_name_to_idx.index('Cyclist')] = 0.49
    p.nms_iou_threshold[class_name_to_idx.index('Pedestrian')] = 0.32

    p.nms_score_threshold = [1.0] * num_classes
    p.nms_score_threshold[class_name_to_idx.index('Cyclist')] = 0.11
    p.nms_score_threshold[class_name_to_idx.index('Pedestrian')] = 0.23

    p.output_decoder.filter_predictions_outside_frustum = True
    p.output_decoder.truncation_threshold = 0.65
    # Equally weight pedestrian and cyclist moderate classes.
    p.output_decoder.ap_metric.metric_weights = {
        'easy': np.array([0.0, 0.0, 0.0]),
        'moderate': np.array([0.0, 1.0, 1.0]),
        'hard': np.array([0.0, 0.0, 0.0])
    }

    return p
