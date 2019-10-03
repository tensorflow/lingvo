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
"""Train models on the waymo open dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import cluster_factory
from lingvo.core import hyperparams
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.tasks.car import input_preprocessors
from lingvo.tasks.car import lr_util
from lingvo.tasks.car import starnet
from lingvo.tasks.car.waymo import waymo_decoder
from lingvo.tasks.car.waymo import waymo_metadata
from lingvo.tasks.car.waymo import waymo_open_input_generator
import numpy as np

# Set $WAYMO_DIR to the base path of where all the WAYMO files can be found.
_WAYMO_BASE = os.environ.get('WAYMO_DIR', 'FILL-ME-IN')


def WaymoTrainSpec(params):
  """Training set."""
  p = params.Copy()
  p.file_datasource.base_datasource.file_pattern = 'train.tfr-*-of-01000'
  p.num_samples = 158361
  return p


def WaymoMiniTrainSpec(params):
  """Training shards used for decoding."""
  p = params.Copy()
  p.file_datasource.base_datasource.file_pattern = 'train.tfr-000[0-2]?-of-01000'
  p.num_samples = 4773
  return p


def WaymoValSpec(params):
  """Validation set.

  Validation contains no overlapping run segments with training.

  Args:
    params: A Waymo open dataset input params.

  Returns:
    An updated Waymo open dataset params with the validation spec.
  """
  p = params.Copy()
  p.file_datasource.base_datasource.file_pattern = 'valid.tfr-*-of-01000'
  p.num_samples = 40077
  return p


def WaymoMinivalSpec(params):
  """Miniature validation set (10% of full validation set)."""
  p = params.Copy()
  p.file_datasource.base_datasource.file_pattern = 'valid.tfr-000??-of-01000'
  p.num_samples = 4109
  return p


def WaymoTestSpec(params):
  # For now, using validation set, since test set is not available.
  return WaymoValSpec(params)


class WaymoSparseLaserTrain(waymo_open_input_generator.WaymoSparseLaser):
  """Waymo train set with sparse laser data."""

  @classmethod
  def Params(cls):
    p = super(WaymoSparseLaserTrain, cls).Params()
    return WaymoTrainSpec(p)


class WaymoSparseLaserValidation(waymo_open_input_generator.WaymoSparseLaser):
  """Waymo validation set with sparse laser data."""

  @classmethod
  def Params(cls):
    p = super(WaymoSparseLaserValidation, cls).Params()
    return WaymoValSpec(p)


class WaymoSparseLaserMinival(waymo_open_input_generator.WaymoSparseLaser):
  """Waymo mini validation set with sparse laser data."""

  @classmethod
  def Params(cls):
    p = super(WaymoSparseLaserMinival, cls).Params()
    return WaymoMinivalSpec(p)


class WaymoSparseLaserTest(waymo_open_input_generator.WaymoSparseLaser):
  """Waymo test set with sparse laser data. Has no groundtruth labels."""

  @classmethod
  def Params(cls):
    p = super(WaymoSparseLaserTest, cls).Params()
    return WaymoTestSpec(p)


def _FilterKeepLabels(params, label_names):
  """Keep only label names in 'label_names' from input."""
  metadata = waymo_metadata.WaymoMetadata()
  filtered_labels = [
      metadata.ClassNames().index(label_name) for label_name in label_names
  ]
  params.extractors.labels.filter_labels = filtered_labels


@model_registry.RegisterSingleTaskModel
class StarNetBase(base_model_params.SingleTaskModelParams):
  """StarNet model for running on Waymo.

  This the base model, please refer to specialized vehicles, pedestrians, etc.
  models below to get an appropriate model to train.
  """
  RUN_LOCALLY = False
  NUM_ANCHOR_BBOX_OFFSETS = 25
  NUM_ANCHOR_BBOX_ROTATIONS = 4
  NUM_ANCHOR_BBOX_DIMENSIONS = 1

  # Ground truth filtering
  GT_MIN_NUM_POINTS = 5

  # Architecture Params
  NUM_GIN_LAYERS = 5
  GIN_HIDDEN_DIMS = 64
  NUM_POINTS_PER_CELL = 512

  class AnchorBoxSettings(input_preprocessors.SparseCarV1AnchorBoxSettings):
    ROTATIONS = [0, np.pi / 2, 3. * np.pi / 4, np.pi / 4]

  def _configure_input(self, p, split):
    p.file_datasource.file_pattern_prefix = _WAYMO_BASE

    job_type = cluster_factory.Current().job

    max_num_points = int(64 * 2650 * 1.5)
    p.preprocessors = hyperparams.Params()
    p.preprocessors.Define('filter_nlz_points',
                           waymo_open_input_generator.FilterNLZPoints.Params(),
                           '')
    p.preprocessors.Define(
        'filter_groundtruth',
        input_preprocessors.FilterGroundTruthByNumPoints.Params(), '')
    p.preprocessors.Define('viz_copy',
                           input_preprocessors.CreateDecoderCopy.Params(), '')
    p.preprocessors.Define('select_centers',
                           input_preprocessors.SparseCenterSelector.Params(),
                           '')
    p.preprocessors.Define(
        'gather_features',
        input_preprocessors.SparseCellGatherFeatures.Params(), '')
    p.preprocessors.Define('tile_anchors',
                           input_preprocessors.TileAnchorBBoxes.Params(), '')
    p.preprocessors.Define('assign_anchors',
                           input_preprocessors.AnchorAssignment.Params(), '')
    p.preprocessors.Define(
        'pad_lasers',
        input_preprocessors.PadLaserFeatures.Params().Set(
            max_num_points=max_num_points), '')

    p.preprocessors.viz_copy.pad_lasers.max_num_points = max_num_points
    p.preprocessors.filter_groundtruth.min_num_points = self.GT_MIN_NUM_POINTS

    p.preprocessors.select_centers.num_cell_centers = 1024
    p.preprocessors.gather_features.num_points_per_cell = self.NUM_POINTS_PER_CELL
    p.preprocessors.gather_features.sample_neighbors_uniformly = True
    p.preprocessors.gather_features.max_distance = 2.75

    p.preprocessors.assign_anchors.foreground_assignment_threshold = 0.6
    p.preprocessors.assign_anchors.background_assignment_threshold = 0.45

    p.preprocessors_order = [
        'filter_nlz_points',
        'filter_groundtruth',
        'viz_copy',
        'select_centers',
        'gather_features',
        'tile_anchors',
        'assign_anchors',
        'pad_lasers',
    ]

    # Apply car anchor box settings.
    tile_anchors_p = p.preprocessors.tile_anchors
    self.AnchorBoxSettings.Update(p.preprocessors.tile_anchors)
    num_anchor_configs = self.AnchorBoxSettings.NumAnchors()

    assert len(tile_anchors_p.anchor_box_dimensions) == num_anchor_configs
    assert len(tile_anchors_p.anchor_box_rotations) == num_anchor_configs
    assert len(tile_anchors_p.anchor_box_offsets) == num_anchor_configs

    # If this is not the decoder job (e.g., this is trainer), turn off
    # image decoding, do not count points, and do not make visualization copies.
    if job_type != 'decoder':
      p.preprocessors_order.remove('viz_copy')
      # Do not need laser points during training for current V2 model. This
      # reduces amount of data sent over during training.
      p.preprocessors.pad_lasers.max_num_points = 0

    p.file_buffer_size = 32
    p.file_parallelism = 8
    p.num_batcher_threads = 8
    if self.RUN_LOCALLY:
      p.num_batcher_threads = 1
      p.file_buffer_size = 1
      p.file_parallelism = 1

    if job_type.startswith('trainer'):
      p.batch_size = 2
    else:
      p.batch_size = 4
      p.file_buffer_size = 64
      p.file_parallelism = 16
      p.num_batcher_threads = 16
    return p

  def Train(self):
    p = waymo_open_input_generator.WaymoSparseLaser.Params()
    p = WaymoTrainSpec(p)
    p = self._configure_input(p, 'Train')
    return p

  def Minitrain(self):
    p = self.Train()
    p = WaymoMiniTrainSpec(p)
    return p

  def Test(self):
    p = waymo_open_input_generator.WaymoSparseLaser.Params()
    p = WaymoTestSpec(p)
    p = self._configure_input(p, 'Test')
    return p

  def Dev(self):
    p = waymo_open_input_generator.WaymoSparseLaser.Params()
    p = WaymoValSpec(p)
    p = self._configure_input(p, 'Dev')
    return p

  def Minidev(self):
    p = self.Dev()
    p = WaymoMinivalSpec(p)
    return p

  def Task(self):
    metadata = waymo_metadata.WaymoMetadata()
    num_classes = len(metadata.ClassNames())
    p = starnet.ModelV2.Params(
        num_classes,
        num_anchor_bboxes_offsets=self.NUM_ANCHOR_BBOX_OFFSETS,
        num_anchor_bboxes_rotations=self.NUM_ANCHOR_BBOX_ROTATIONS,
        num_anchor_bboxes_dimensions=self.NUM_ANCHOR_BBOX_DIMENSIONS,
        num_laser_features=3)

    # Update the Point Cloud Featurizer architecture
    starnet_builder = starnet.Builder()
    starnet_builder.linear_params_init = (
        py_utils.WeightInit.KaimingUniformFanInRelu())

    gin_layers = [
        [self.GIN_HIDDEN_DIMS*2, self.GIN_HIDDEN_DIMS*4, self.GIN_HIDDEN_DIMS]
    ] * self.NUM_GIN_LAYERS  # pyformat: disable

    p.cell_featurizer = starnet_builder.GINFeaturizerV2(
        'feat',
        num_laser_features=3,
        fc_dims=self.GIN_HIDDEN_DIMS,
        mlp_dims=gin_layers,
        fc_use_bn=False)
    p.cell_feature_dims = self.GIN_HIDDEN_DIMS * (self.NUM_GIN_LAYERS + 1)

    p.output_decoder = waymo_decoder.WaymoOpenDatasetDecoder.Params()
    p.max_nms_boxes = 512
    p.use_oriented_per_class_nms = True

    # Note: Sub-classes need to set nms_iou_threshold and nms_score_threshold
    # appropriately.
    p.nms_iou_threshold = [0.0] * num_classes

    # TODO(jngiam): 1.1 for untrained classes is needed to avoid an issue
    # with boxutils error.
    p.nms_score_threshold = [1.1] * num_classes

    p.name = 'starnet'
    tp = p.train
    tp.optimizer = optimizer.Adam.Params()
    tp.clip_gradient_norm_to_value = 5

    ep = p.eval

    # Train set uses a smaller decoding set, so we can
    # safely eval over the entire input.
    ep.samples_per_summary = 0

    # To be tuned.
    p.train.l2_regularizer_weight = 1e-8

    cluster = cluster_factory.Current()
    train_cluster_p = cluster.params.Copy()
    train_cluster_p.job = 'trainer_client'
    train_cluster_p.mode = 'sync'

    # When running a decoding only job, there are no trainer workers, so we set
    # worker replicas to 1 as a dummy value.
    if train_cluster_p.worker.replicas <= 0:
      train_cluster_p.worker.replicas = 1

    # Set learning rate and schedule.
    with cluster_factory.Cluster(train_cluster_p):
      train_input_p = self.Train()

    # Adapted from V1 tuning.
    tp.ema_decay = 0.99
    tp.learning_rate = 0.001
    lr_util.SetExponentialLR(
        train_p=tp,
        train_input_p=train_input_p,
        exp_start_epoch=5,
        total_epoch=75)

    p.dimension_loss_weight = .3
    p.location_loss_weight = 3.
    p.loss_weight_classification = 1.
    p.loss_weight_localization = 3.
    p.rotation_loss_weight = 0.3

    return p


@model_registry.RegisterSingleTaskModel
class StarNetVehicle(StarNetBase):
  """StarNet model for cars, running on Waymo."""
  GT_MIN_NUM_POINTS = 5

  class AnchorBoxSettings(input_preprocessors.SparseCarV1AnchorBoxSettings):
    DIMENSION_PRIORS = [(4.725808, 2.079292, 1.768998)]
    ROTATIONS = [0, np.pi / 2, 3. * np.pi / 4, np.pi / 4]
    CENTER_X_OFFSETS = np.linspace(-1.294, 1.294, 5)
    CENTER_Y_OFFSETS = np.linspace(-1.294, 1.294, 5)
    CENTER_Z_OFFSETS = [0.819622]

  def _configure_input(self, p, split):
    p = super(StarNetVehicle, self)._configure_input(p, split)

    # Select points in approx z range, set using 10 and 90 percentile from
    # train data.
    p.preprocessors.select_centers.features_preparation_layers = [
        input_preprocessors.DropLaserPointsOutOfRange.Params().Set(
            keep_z_range=(0.197431, 2.046459)),
    ]

    _FilterKeepLabels(p, ['Vehicle'])
    return p

  def Task(self):
    p = super(StarNetVehicle, self).Task()
    p.train.l2_regularizer_weight = 3e-5

    class_names = waymo_metadata.WaymoMetadata().ClassNames()
    num_classes = len(class_names)

    p.per_class_loss_weight = [0.] * num_classes
    p.per_class_loss_weight[class_names.index('Vehicle')] = 1.0

    # See WaymoMetadata.EvalClassIndices for correspondences for metric_weights.
    p.output_decoder.ap_metric.metric_weights = {
        'default': np.array([1.0, 0.0, 0.0]),
    }
    p.nms_iou_threshold[class_names.index('Vehicle')] = 0.03
    p.nms_score_threshold[class_names.index('Vehicle')] = 0.31
    return p


@model_registry.RegisterSingleTaskModel
class StarNetPed(StarNetBase):
  """StarNet model for pedestrians, running on Waymo."""
  NUM_ANCHOR_BBOX_ROTATIONS = 2
  GT_MIN_NUM_POINTS = 5

  class AnchorBoxSettings(input_preprocessors.SparseCarV1AnchorBoxSettings):
    DIMENSION_PRIORS = [(0.901340, 0.857218, 1.712443)]
    ROTATIONS = [0, np.pi / 2]
    CENTER_X_OFFSETS = np.linspace(-0.6, 0.6, 5)
    CENTER_Y_OFFSETS = np.linspace(-0.6, 0.6, 5)
    CENTER_Z_OFFSETS = [0.819622]

  def _configure_input(self, p, split):
    p = super(StarNetPed, self)._configure_input(p, split)

    # Select points in approx z range, set using 10 and 90 percentile from
    # train data.
    p.preprocessors.select_centers.features_preparation_layers = [
        input_preprocessors.DropLaserPointsOutOfRange.Params().Set(
            keep_z_range=(0.09522381, 1.720825)),
    ]

    _FilterKeepLabels(p, ['Pedestrian'])
    return p

  def Task(self):
    p = super(StarNetPed, self).Task()
    p.train.l2_regularizer_weight = 3e-6

    class_names = waymo_metadata.WaymoMetadata().ClassNames()
    num_classes = len(class_names)

    p.per_class_loss_weight = [0.] * num_classes
    p.per_class_loss_weight[class_names.index('Pedestrian')] = 1.0

    # See WaymoMetadata.EvalClassIndices for correspondences for metric_weights.
    p.output_decoder.ap_metric.metric_weights = {
        'default': np.array([0.0, 1.0, 0.0]),
    }
    p.nms_iou_threshold[class_names.index('Pedestrian')] = 0.46
    p.nms_score_threshold[class_names.index('Pedestrian')] = 0.01

    return p


@model_registry.RegisterSingleTaskModel
class StarNetPedFused(StarNetPed):
  """StarNet Pedestrian model with SparseSampler fused input preprocessor."""

  def _configure_input(self, p, split):
    p = super(StarNetPedFused, self)._configure_input(p, split)

    job_type = cluster_factory.Current().job

    if job_type.startswith('trainer') and not self.RUN_LOCALLY:
      p.file_buffer_size = 48
      p.file_parallelism = 48
      p.num_batcher_threads = 48

    # Fuses select_centers and gather_features into one sampler.
    p.preprocessors.Define(
        'sampler',
        input_preprocessors.SparseSampler.Params().Set(
            center_selector='farthest',
            neighbor_sampler='uniform',
            num_centers=p.preprocessors.select_centers.num_cell_centers,
            keep_z_range=(0.09522381, 1.720825),
            num_neighbors=p.preprocessors.gather_features.num_points_per_cell,
            max_distance=2.75), '')
    p.preprocessors.Delete('select_centers')
    p.preprocessors.Delete('gather_features')
    p.preprocessors_order.remove('select_centers')
    p.preprocessors_order[p.preprocessors_order.index(
        'gather_features')] = 'sampler'
    return p
