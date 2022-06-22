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
"""Train deepfusion models on Waymo."""

import os

from lingvo import model_registry
from lingvo.core import activations
from lingvo.core import base_model_params
from lingvo.core import cluster_factory
from lingvo.core import hyperparams
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.tasks.car import deep_fusion
from lingvo.tasks.car import input_preprocessors
from lingvo.tasks.car import kitti_ap_metric
from lingvo.tasks.car import lr_util
from lingvo.tasks.car import pillars
from lingvo.tasks.car import pillars_anchor_free
from lingvo.tasks.car.params import waymo as waymo_params
from lingvo.tasks.car.waymo import waymo_decoder
from lingvo.tasks.car.waymo import waymo_metadata
from lingvo.tasks.car.waymo import waymo_open_input_generator
import numpy as np

# Set $WAYMO_DIR to the base path of where all the WAYMO files can be found.
_WAYMO_BASE = os.environ.get('WAYMO_DIR', 'FILL-ME-IN')


################## DeepFusion #############################
def _FilterKeepLabels(params, label_names):
  """Keep only label names in 'label_names' from input."""
  metadata = waymo_metadata.WaymoMetadata()
  filtered_labels = [
      metadata.ClassNames().index(label_name) for label_name in label_names
  ]
  params.extractors.labels.filter_labels = filtered_labels


def _NestedMapToParams(nmap):
  p = hyperparams.Params()
  for k, v in nmap.FlattenItems():
    p.Define(k, v, '')
  return p


def AddKITTIMetric(params):
  """Append the KITTI evaluation metrics to the list metrics run."""
  p = params.Copy()
  p.output_decoder.extra_ap_metrics = {
      # We use the configuration for the Waymo dataset for evaluating
      # with the KITTI evaluation code.
      'kitti':
          kitti_ap_metric.KITTIAPMetrics.Params(waymo_metadata.WaymoMetadata())
  }
  return p


def TrainerInputParams(train_params_fn):
  """Returns input params called under the context of a trainer.

  Args:
    train_params_fn: A callable that returns a Params() input object.

  Returns:
    An input params called in the context of a trainer.
  """
  cluster = cluster_factory.Current()
  train_cluster_p = cluster.params.Copy()
  train_cluster_p.job = 'trainer_client'
  train_cluster_p.mode = 'sync'

  # When running a decoding only job, the job is configured so that there are no
  # worker replicas.
  #
  # This prevents us from fetching the training parameters (an assert triggers
  # if you try to fetch the training params with 0 workers), so we set worker
  # replicas to 1 as a dummy value.
  if train_cluster_p.worker.replicas <= 0:
    train_cluster_p.worker.replicas = 1

  with cluster_factory.Cluster(train_cluster_p):
    train_input_p = train_params_fn()

  return train_input_p


def AddPreprocessor(input_p,
                    name,
                    new_preprocessor_p,
                    insert_before=None,
                    insert_after=None):
  """Add a new preprocessor before an existing preprocessor.

  Args:
    input_p: The input params.
    name: A string with the name of the new preprocessor.
    new_preprocessor_p: The params of the new preprocessor.
    insert_before: A string with which preprocessor to insert the new
      preprocessor before. Defaults to None. Must specify either this or
      `insert_after`.
    insert_after: A string with which preprocessor to insert the new
      preprocessor before. Defaults to None. Must specify either this or
      `insert_before`.

  Returns:
    input_p: The input preprocessor with the new added preprocessor.
  """
  if insert_before and insert_before not in input_p.preprocessors_order:
    raise ValueError('`insert_before` preprocessor `{}` not found in '
                     'preprocessors_order.'.format(insert_before))
  if insert_after and insert_after not in input_p.preprocessors_order:
    raise ValueError('`insert_after` preprocessor `{}` not found in '
                     'preprocessors_order.'.format(insert_after))
  if insert_before is None and insert_after is None:
    raise ValueError('Must either specify `insert_before` or `insert_after`.')
  if insert_before is not None and insert_after is not None:
    raise ValueError('Please only provide `insert_before` or `insert_after` '
                     'not both.')

  input_p.preprocessors.Define(name, new_preprocessor_p, '')
  if insert_before:
    insert_index = input_p.preprocessors_order.index(insert_before)
  else:
    insert_index = input_p.preprocessors_order.index(insert_after) + 1
  input_p.preprocessors_order = (
      input_p.preprocessors_order[:insert_index] + [name] +
      input_p.preprocessors_order[insert_index:])
  return input_p


@model_registry.RegisterSingleTaskModel
class AnchorFreePillarsModelV1Base(base_model_params.SingleTaskModelParams):
  """Base model using point pillars featurization on a point cloud grid.

  This the base model, please refer to specialized vehicles, pedestrians, etc.
  models below to get an appropriate model to train.
  """
  RUN_LOCALLY = 'BORG_TASK_HANDLE' not in os.environ
  NUM_PILLARS = 16000

  NUM_LASER_FEATURES = 3
  ANGLE_BIN_NUM = 12

  FRAME_OFFSETS = None
  FRAME_DROPOUT_RATE = 0.
  CAMERA_INPUT = False

  GRID_SETTINGS = input_preprocessors.MakeGridSettings(
      grid_x_range=(-85.00, 85.00),
      grid_y_range=(-85.00, 85.00),
      grid_z_range=(-3, 3),
      grid_x=512,
      grid_y=512,
      grid_z=1)

  TASK_CLS = pillars_anchor_free.AnchorFreePillarsBase

  METADATA = waymo_metadata.WaymoMetadata()
  CLASS_NAMES = METADATA.ClassNames()
  NUM_CLASSES = len(CLASS_NAMES)

  def _configure_input_single_frame(self, p, split):
    p.file_datasource.file_pattern_prefix = _WAYMO_BASE

    job_type = cluster_factory.Current().job
    max_num_points = int(64 * 2560 * 1.5)

    p.preprocessors = _NestedMapToParams(
        py_utils.NestedMap(
            filter_nlz_points=waymo_open_input_generator.FilterNLZPoints.Params(
            ),
            filter_groundtruth=(
                input_preprocessors.FilterGroundTruthByDifficulty.Params()),
            viz_copy=input_preprocessors.CreateDecoderCopy.Params(),
            points_to_grid=input_preprocessors.PointsToGrid.Params().Set(
                normalize_td_labels=False),
            grid_to_pillars=input_preprocessors.GridToPillars.Params().Set(
                num_pillars=self.NUM_PILLARS),
            grid_anchor_centers=input_preprocessors.GridAnchorCenters.Params(),
            assign_points=input_preprocessors.PointAssignment.Params(
                num_classes=self.NUM_CLASSES),
            pad_lasers=input_preprocessors.PadLaserFeatures.Params().Set(
                max_num_points=max_num_points),
        ))

    p.preprocessors.viz_copy.pad_lasers.max_num_points = max_num_points

    p.preprocessors_order = [
        'filter_nlz_points',
        'filter_groundtruth',
        'viz_copy',
        'points_to_grid',
        'grid_to_pillars',
        'grid_anchor_centers',
        'assign_points',
        'pad_lasers',
    ]

    # Only train on LEVEL_1, and evaluate on LEVEL_2 or lower.
    if job_type.startswith('trainer'):
      p.preprocessors.filter_groundtruth.difficulty_threshold = 1
    else:
      p.preprocessors.filter_groundtruth.difficulty_threshold = 2

    self.GRID_SETTINGS().UpdateGridParams(p.preprocessors.points_to_grid)
    self.GRID_SETTINGS().UpdateAnchorGridParams(
        p.preprocessors.grid_anchor_centers)

    # If this is not the decoder job (e.g., this is trainer), do not
    # count points and do not make visualization copies.
    if job_type != 'decoder':
      p.preprocessors_order.remove('viz_copy')

    if job_type.startswith('trainer'):
      p.batch_size = 2
    else:
      p.batch_size = 4

    if self.RUN_LOCALLY:
      p.num_batcher_threads = 1
      p.file_buffer_size = 1
      p.file_parallelism = 1
    else:
      p.num_batcher_threads = 16
      p.file_buffer_size = 32
      p.file_parallelism = 32

    return p

  def _configure_input(self, p, split):
    if self.FRAME_OFFSETS:
      raise NotImplementedError
    else:
      return self._configure_input_single_frame(p, split)

  def Train(self):
    p = waymo_open_input_generator.WaymoSparseLaser.Params()
    p = waymo_params.WaymoTrainSpec(p)
    p = self._configure_input(p, 'Train')
    return p

  def Minitrain(self):
    p = self.Train()
    p = waymo_params.WaymoMiniTrainSpec(p)
    return p

  def Test(self):
    p = waymo_open_input_generator.WaymoSparseLaser.Params()
    p = waymo_params.WaymoTestSpec(p)
    p = self._configure_input(p, 'Test')
    return p

  def Dev(self):
    p = waymo_open_input_generator.WaymoSparseLaser.Params()
    p = waymo_params.WaymoValSpec(p)
    p = self._configure_input(p, 'Dev')
    return p

  def Minidev(self):
    p = self.Dev()
    p = waymo_params.WaymoMinivalSpec(p)
    return p

  def Task(self):
    # Number of classes can be fetched from input.
    p = self.TASK_CLS.Params(
        grid_size_z=self.GRID_SETTINGS().GRID_Z,
        num_classes=self.NUM_CLASSES,
        num_laser_features=self.NUM_LASER_FEATURES,
        angle_bin_num=self.ANGLE_BIN_NUM)
    p.name = 'anchor_free_point_pillars_waymo'

    p.output_decoder = waymo_decoder.WaymoOpenDatasetDecoder.Params()

    p.max_nms_boxes = 512
    p.use_oriented_per_class_nms = True

    # Note: Sub-classes need to set nms_iou_threshold and nms_score_threshold
    # appropriately.
    p.nms_iou_threshold = [0.0] * self.NUM_CLASSES

    # TODO(jngiam): 1.1 for untrained classes is needed to avoid an issue
    # with boxutils error.
    p.nms_score_threshold = [1.1] * self.NUM_CLASSES

    p.nms_iou_threshold[self.CLASS_NAMES.index('Vehicle')] = 0.5
    p.nms_score_threshold[self.CLASS_NAMES.index('Vehicle')] = 0.05

    ep = p.eval
    # Train set uses a smaller decoding set, so we can
    # safely eval over the entire input.
    ep.samples_per_summary = 0

    tp = p.train
    tp.optimizer = optimizer.Adam.Params()
    tp.clip_gradient_norm_to_value = 50

    # To be tuned.
    p.train.l2_regularizer_weight = 1e-4

    # Adapted from V1 tuning.
    tp.ema_decay = 0.99
    # TODO(b/148537111): consider setting this to True.
    tp.ema_decay_moving_vars = False

    train_input_p = TrainerInputParams(self.Train)

    # Get number of parallel processing cores for the worker.
    # This is 8 for a 2x2 TPU, or 1 for a single GPU.
    cluster = cluster_factory.Current()
    total_num_cores = cluster.total_worker_devices
    total_batch_size = train_input_p.batch_size * total_num_cores

    # Set learning rate and schedule.
    tp.learning_rate = 1e-4 * total_batch_size / 2

    # Train for 75 epochs.
    lr_util.SetExponentialLR(
        train_p=tp,
        train_input_p=train_input_p,
        exp_start_epoch=5,
        total_epoch=75)
    return p


@model_registry.RegisterSingleTaskModel
class AnchorFreePillarsModelV1Ped(AnchorFreePillarsModelV1Base):
  """Pedestrian model w/ 512x512, 32k pillars, 1 stride in backbone."""

  BLOCK0_STRIDE = 1
  NUM_PILLARS = 32000
  USE_BASIC_DATA_AUG = False
  VALID_CLASS_NAMES = ['Pedestrian']

  def _configure_input(self, p, split):
    job_type = cluster_factory.Current().job
    p = super()._configure_input(p, split)

    if job_type.startswith('trainer') and self.USE_BASIC_DATA_AUG:
      p.preprocessors.Define('global_loc_noise',
                             input_preprocessors.GlobalTranslateNoise.Params(),
                             '')
      p.preprocessors.Define(
          'rot_box',
          input_preprocessors.RandomBBoxTransform.Params().Set(
              max_rotation=np.pi / 20.), '')
      p.preprocessors.Define('random_flip',
                             input_preprocessors.RandomFlipY.Params(), '')
      p.preprocessors.Define(
          'world_scaling',
          (input_preprocessors.WorldScaling.Params().Set(scaling=[0.95, 1.05])),
          '')
      p.preprocessors_order = [
          'rot_box', 'random_flip', 'world_scaling', 'global_loc_noise'
      ] + p.preprocessors_order

    # Overwrites previous filtering. For multi frame model, this is already done
    # when preparing the input (See ConfigurePillarsSquashedSequenceInputs).
    if not self.FRAME_OFFSETS:
      _FilterKeepLabels(p, self.VALID_CLASS_NAMES)
    self.GRID_SETTINGS().UpdateAnchorGridParams(
        p.preprocessors.grid_anchor_centers, output_stride=self.BLOCK0_STRIDE)
    return p

  def Task(self):
    p = super().Task()
    pillars_builder = pillars.Builder()
    p.backbone = pillars_builder.Backbone(
        odims=self.TASK_CLS.NUM_OUTPUT_CHANNELS,
        down_strides=(self.BLOCK0_STRIDE, 2, 2))

    # Initialize the class detector's bias term to be negative in line
    # with focal losses paper (so predictions init as background).
    p.class_detector = pillars_builder.Detector(
        name='class',
        idims=(3 * self.TASK_CLS.NUM_OUTPUT_CHANNELS),
        odims=(p.grid_size_z * p.num_classes),
        bias_params_init=py_utils.WeightInit.Constant(-4.595))

    # Normalization hurts the training in some steps:
    # pc_046 is without normalization, pc_047 is with.
    p.loss_norm_type = pillars.LossNormType.NO_NORM

    metadata = waymo_metadata.WaymoMetadata()
    num_classes = len(metadata.ClassNames())
    p.use_oriented_per_class_nms = True
    p.max_nms_boxes = 512
    p.nms_iou_threshold = [0.0] * num_classes
    p.nms_iou_threshold[metadata.ClassNames().index('Pedestrian')] = 0.46

    p.nms_score_threshold = [1.1] * num_classes
    p.nms_score_threshold[metadata.ClassNames().index('Pedestrian')] = 0.01

    p.per_class_loss_weight = [0.] * num_classes
    p.per_class_loss_weight[metadata.ClassNames().index('Pedestrian')] = 1.0

    # Add the KITTI evaluation metric to the Waymo Open Dataset in order to
    # perform calibration analysis.
    p = AddKITTIMetric(p)
    return p


@model_registry.RegisterSingleTaskModel
class AnchorFreePillarsModelV1PedDV(AnchorFreePillarsModelV1Ped):
  """V1 Pedestrian with Dynamic voxelization."""

  def _configure_input(self, p, split):
    p = super()._configure_input(p, split)
    if self.FRAME_OFFSETS:
      assert 'points_to_grid' not in p.preprocessors_order
      assert 'grid_to_pillars' not in p.preprocessors_order
    else:
      p.preprocessors_order.remove('points_to_grid')
      p.preprocessors_order.remove('grid_to_pillars')
    return p

  def Task(self):
    p = super().Task()
    p.input_featurizer = pillars.DynamicVoxelizationFeaturizer.Params(
        p.num_laser_features)
    # Update input_featurizer settings by reference.
    self.GRID_SETTINGS().UpdateGridParams(p.input_featurizer)
    return p


@model_registry.RegisterSingleTaskModel
class AnchorFreePillarsModelV1PedAug(AnchorFreePillarsModelV1PedDV):
  """V1 Ped model w/ dynamic voxelization and aug.

  highest L1 mAP: 65.9
  """
  APPLY_DATA_AUG = True

  def _configure_input(self, p, split):
    p = super()._configure_input(p, split)
    job_type = cluster_factory.Current().job
    if job_type.startswith('trainer') and self.APPLY_DATA_AUG:
      p = AddPreprocessor(
          p,
          'random_flip',
          input_preprocessors.RandomFlipY.Params().Set(flip_probability=0.25),
          insert_before='grid_anchor_centers')
      p = AddPreprocessor(
          p,
          'global_rot',
          input_preprocessors.RandomWorldRotationAboutZAxis.Params().Set(
              max_rotation=np.pi / 4.),
          insert_before='grid_anchor_centers')
    return p

  def Task(self):
    p = super().Task()
    p.output_decoder.ap_metric.waymo_breakdown_metrics = ['RANGE', 'VELOCITY']

    return p


@model_registry.RegisterSingleTaskModel
class AnchorFreePillarsModelV1PedCenterNess(AnchorFreePillarsModelV1PedAug):
  """Add center-ness loss in the pedestrian model.

  highest L1 mAP: 69.5
  """

  def _configure_input(self, p, split):
    p = super()._configure_input(p, split)
    job_type = cluster_factory.Current().job
    if job_type.startswith('trainer'):
      p.preprocessors.assign_points.extra_label_range = [0.0, 1.0]

    return p

  def Task(self):
    p = super().Task()
    p.centerness_loss_weight = 1.0
    return p


@model_registry.RegisterSingleTaskModel
class AnchorFreePillarsModelV1PedCenterNessRelated(
    AnchorFreePillarsModelV1PedCenterNess):
  """Times center-ness label with regression mask.

  That is, if a pillar has higher center-ness label, it will also have higher
  regression loss weight.

  highest L1 mAP: 71
  """

  def _configure_input(self, p, split):
    p = super()._configure_input(p, split)
    job_type = cluster_factory.Current().job
    if job_type.startswith('trainer'):
      # Use center-ness label to reweight regression loss.
      p.preprocessors.assign_points.extra_label_related_reg_mask = True

    return p


@model_registry.RegisterSingleTaskModel
class AnchorFreePillarsModelV1VehicleCenterNess(
    AnchorFreePillarsModelV1PedCenterNessRelated):
  """This is the model of refactoring anchor-free pillars.

  highest L1 mAP: 65.22
  """

  GRID_SETTINGS = input_preprocessors.MakeGridSettings(
      grid_x_range=(-76.8, 76.8),
      grid_y_range=(-76.8, 76.8),
      grid_z_range=(-3, 3),
      grid_x=512,
      grid_y=512,
      grid_z=1)

  VALID_CLASS_NAMES = ['Vehicle']

  def Task(self):
    p = super().Task()

    metadata = waymo_metadata.WaymoMetadata()
    class_names = metadata.ClassNames()
    num_classes = len(class_names)

    p.nms_iou_threshold = [0.0] * num_classes
    p.nms_score_threshold = [1.1] * num_classes
    p.per_class_loss_weight = [0.] * num_classes
    p.nms_iou_threshold[class_names.index('Vehicle')] = 0.2
    p.nms_score_threshold[class_names.index('Vehicle')] = 0.001
    p.per_class_loss_weight[metadata.ClassNames().index('Vehicle')] = 1.0

    p.max_nms_boxes = 256

    p.train.l2_regularizer_weight = 0.0

    p.centerness_loss_weight = 0.0

    return p


@model_registry.RegisterSingleTaskModel
class CenterPointImprovedVehicle(AnchorFreePillarsModelV1VehicleCenterNess):
  """CenterPoint Vehicle model with improved implementation.

  Following parameters are tuned to achieve better performance: nms parameters,
  weight decay, gradient clip, data augmentation, training schedule, activation
  function, featurizer, backbone channels, EMA, and train with LEVEL_2
  difficulty data.

  highest L1 mAP: 76.45
  """

  VALID_CLASS_NAMES = ['Vehicle']
  NMS_IOU_THRESHOLD = 0.8
  NMS_SCORE_THRESHOLD = 0.01
  L2_REGULARIZER_WEIGHT = 0
  PER_CLASS_LOSS_WEIGHT = 1.0
  CLIP_GRADIENT_NORM_TO_VALUE = 5

  APPLY_DATA_AUG = False
  APPLY_STRONG_DATA_AUG = True
  MAX_ROTATION = 3.14159

  EPOCH = 60
  WARM_UP_EPOCH = 3
  LEARN_RATE = 3e-4

  NUM_OUTPUT_FEATURES = 256
  MLP_DIMS = [256, 256, 512]
  ACTIVATION = 'SWISH'

  def _apply_strong_data_aug(self, p, insert_before='grid_anchor_centers'):
    """Apply strong data augmentations to an input params."""
    # Global Rotation.
    p = AddPreprocessor(
        p,
        'random_apply_global_rot',
        input_preprocessors.RandomApplyPreprocessor.Params().Set(
            prob=0.74,
            choice_save_prefix='inverse_aug.global_rot',
            subprocessor=(
                input_preprocessors.RandomWorldRotationAboutZAxis.Params().Set(
                    max_rotation=0.41,
                    include_world_rot_z=False,
                    rot_save_key='inverse_aug.global_rot.rot'))),
        insert_before=insert_before)

    # World Scaling.
    p = AddPreprocessor(
        p,
        'world_scaling',
        input_preprocessors.WorldScaling.Params().Set(
            scaling_save_key='inverse_aug.world_scaling.scaling',
            scaling=[0.95, 1.05]),
        insert_before=insert_before)

    # Global Translation.
    p = AddPreprocessor(
        p,
        'global_loc_noise',
        input_preprocessors.GlobalTranslateNoise.Params().Set(
            noise_save_key='inverse_aug.global_loc_noise.noise',
            noise_std=[0., 0., 0.35]),
        insert_before=insert_before)

    # Random Flip.
    p = AddPreprocessor(
        p,
        'random_flip',
        input_preprocessors.RandomFlipY.Params().Set(
            flip_save_key='inverse_aug.random_flip.flip', flip_probability=0.5),
        insert_before=insert_before)

    # Frustum Dropout.
    p = AddPreprocessor(
        p,
        'random_apply_frustum_dropout',
        input_preprocessors.RandomApplyPreprocessor.Params().Set(
            prob=0.3575,
            subprocessor=(input_preprocessors.FrustumDropout.Params().Set(
                theta_width=0.08,
                phi_width=1.07,
                distance=9.46,
                drop_type='union',
                keep_prob=0.44))),
        insert_before=insert_before)

    # Frustum Noise.
    p = AddPreprocessor(
        p,
        'random_apply_frustum_noise',
        input_preprocessors.FrustumNoise.Params().Set(
            theta_width=0.03, phi_width=0.0),
        insert_before=insert_before)

    # Random Dropout.
    p = AddPreprocessor(
        p,
        'random_apply_random_dropout',
        input_preprocessors.RandomDropLaserPoints.Params(),
        insert_before=insert_before)
    return p

  def _configure_input(self, p, split):
    p = super()._configure_input(p, split)
    # change batch size to 1 for comparing with MultiModal Models.
    p.batch_size = 1

    # set training data
    job_type = cluster_factory.Current().job
    if job_type.startswith('trainer') and self.APPLY_STRONG_DATA_AUG:
      assert not self.APPLY_DATA_AUG
      p = self._apply_strong_data_aug(p)
      p.preprocessors.random_apply_global_rot.subprocessor.max_rotation = self.MAX_ROTATION
      p.preprocessors.filter_groundtruth.difficulty_threshold = 2

    return p

  def Task(self):
    p = super().Task()

    # Set class specific parameters (nms, loss weight, l2 loss, gradient clip).
    metadata = waymo_metadata.WaymoMetadata()
    class_names = metadata.ClassNames()
    num_classes = len(class_names)

    nms_iou_threshold = self.NMS_IOU_THRESHOLD if isinstance(
        self.NMS_IOU_THRESHOLD,
        list) else [self.NMS_IOU_THRESHOLD] * len(self.VALID_CLASS_NAMES)
    nms_score_threshold = self.NMS_SCORE_THRESHOLD if isinstance(
        self.NMS_SCORE_THRESHOLD,
        list) else [self.NMS_SCORE_THRESHOLD] * len(self.VALID_CLASS_NAMES)
    per_class_loss_weight = self.PER_CLASS_LOSS_WEIGHT if isinstance(
        self.PER_CLASS_LOSS_WEIGHT,
        list) else [self.PER_CLASS_LOSS_WEIGHT] * len(self.VALID_CLASS_NAMES)

    p.nms_iou_threshold = [0.0] * num_classes
    p.nms_score_threshold = [1.1] * num_classes
    p.per_class_loss_weight = [0.] * num_classes
    for class_idx, class_name in enumerate(self.VALID_CLASS_NAMES):
      p.nms_iou_threshold[class_names.index(
          class_name)] = nms_iou_threshold[class_idx]
      p.nms_score_threshold[class_names.index(
          class_name)] = nms_score_threshold[class_idx]
      p.per_class_loss_weight[metadata.ClassNames().index(
          class_name)] = per_class_loss_weight[class_idx]

    p.train.l2_regularizer_weight = self.L2_REGULARIZER_WEIGHT
    if self.CLIP_GRADIENT_NORM_TO_VALUE:
      tp = p.train
      tp.clip_gradient_norm_to_value = self.CLIP_GRADIENT_NORM_TO_VALUE

    # Set architecture.
    pillars_builder = pillars.Builder()
    pillars_builder.activation_fn = activations.GetFn(self.ACTIVATION)
    point_encoder = p.input_featurizer.point_encoder.Instantiate()
    encoding_size = point_encoder.NumEncodingFeatures(p.num_laser_features)
    p.input_featurizer.featurizer = pillars_builder.MLPFeaturizer(
        'feat', [encoding_size] + self.MLP_DIMS + [self.NUM_OUTPUT_FEATURES],
        activation_fn=self.ACTIVATION)
    backbone_channel_multiplier = self.NUM_OUTPUT_FEATURES // 64
    p.backbone = pillars_builder.Backbone(
        odims=self.TASK_CLS.NUM_OUTPUT_CHANNELS,
        down_strides=(self.BLOCK0_STRIDE, 2, 2),
        channel_multiplier=backbone_channel_multiplier,
        activation=self.ACTIVATION)

    # Get number of parallel processing cores for the worker.
    # This is 8 for a 2x2 TPU, or 1 for a single GPU.
    tp = p.train
    train_input_p = TrainerInputParams(self.Train)
    cluster = cluster_factory.Current()
    total_num_cores = cluster.total_worker_devices
    total_batch_size = train_input_p.batch_size * total_num_cores
    # Set learning rate and schedule.
    tp.learning_rate = self.LEARN_RATE * total_batch_size / 2
    # TODO(ywli): currently the warmup phase is linear rampup,
    # but in DeepFusion, Cosine Rampup is used.
    # see function SetOneCycleLR from cl/419866459
    # lingvo/tasks/car/lr_util.py
    lr_util.SetCosineLR(
        train_p=tp,
        train_input_p=train_input_p,
        total_epoch=self.EPOCH,
        warmup_epoch=self.WARM_UP_EPOCH,
        warmup_init=0.1)

    # Set EMA parameters.
    tp = p.train
    tp.ema_decay = 0.9999
    tp.ema_decay_moving_vars = True

    return p


@model_registry.RegisterSingleTaskModel
class CenterPointImprovedPedestrian(CenterPointImprovedVehicle):
  """CenterPoint Pedestrian model with improved implementation.

  Following parameters are tuned to achieve better performance: nms parameters,
  weight decay, gradient clip, data augmentation, training schedule, activation
  function, featurizer, backbone channels, EMA, and train with LEVEL_2
  difficulty data.

  highest L1 mAP: 80.36
  """
  MAX_ROTATION = 2.0944
  NMS_IOU_THRESHOLD = 0.3
  VALID_CLASS_NAMES = ['Pedestrian']
  L2_REGULARIZER_WEIGHT = 1e-4
  CLIP_GRADIENT_NORM_TO_VALUE = 50


@model_registry.RegisterSingleTaskModel
class UncertaintyCenterPointPed(CenterPointImprovedPedestrian):
  """CenterPoint Pedestrian model with uncertainty loss.

  The uncertainty (from https://arxiv.org/abs/1910.11375) is applied to the
  location loss and dimensions loss.

  highest L1 mAP: 81.49
  """

  def Task(self):
    p = super().Task()
    p.location_loss = pillars_anchor_free.LaplaceKL.Params().Set()
    p.dimensions_loss = pillars_anchor_free.LaplaceKL.Params().Set(
        targets_scale=0.001)
    p.dimensions_loss_weight = 0.3

    pillars_builder = pillars.Builder()
    p.regression_detector = pillars_builder.Detector(
        name='reg',
        idims=(3 * self.TASK_CLS.NUM_OUTPUT_CHANNELS),
        odims=(p.grid_size_z *
               (3 * p.location_loss.num_params_per_prediction +
                3 * p.dimensions_loss.num_params_per_prediction +
                p.angle_bin_num + p.angle_bin_num)),
        conv_init_method=py_utils.WeightInit.Constant(0.0))
    return p


@model_registry.RegisterSingleTaskModel
class DeepFusionCenterPointPed(UncertaintyCenterPointPed):
  """DeepFusion CenterPoint Pedstrain Model.

  A late-stage deep feature level fusion, with InverseAug and LearnableAlign, to
  improve the quality of alignment among multimodal features. For more details,
  see https://arxiv.org/pdf/2203.08195.pdf.
  """
  CAMERA_INPUT = True

  def _apply_inverse_aug(self, p, insert_after='create_cell_center_xyz'):
    """Apply inverse augmentations to an input params."""
    # Random Flip.
    p = AddPreprocessor(
        p,
        'inverse_random_flip',
        input_preprocessors.InverseRandomFlipY.Params().Set(
            flip_save_key='inverse_aug.random_flip.flip',
            points_keys=['cell_center_xyz']),
        insert_after=insert_after)

    # Global Translation.
    p = AddPreprocessor(
        p,
        'inverse_global_loc_noise',
        input_preprocessors.InverseGlobalTranslateNoise.Params().Set(
            noise_save_key='inverse_aug.global_loc_noise.noise',
            points_keys=['cell_center_xyz']),
        insert_after='inverse_random_flip')

    # World Scaling.
    p = AddPreprocessor(
        p,
        'inverse_world_scaling',
        input_preprocessors.InverseWorldScaling.Params().Set(
            scaling_save_key='inverse_aug.world_scaling.scaling',
            points_keys=['cell_center_xyz']),
        insert_after='inverse_global_loc_noise')

    # Global Rotation.
    p = AddPreprocessor(
        p,
        'inverse_random_apply_global_rot',
        input_preprocessors.InverseRandomApplyPreprocessor.Params().Set(
            choice_save_prefix='inverse_aug.global_rot',
            subprocessor=(input_preprocessors
                          .InverseRandomWorldRotationAboutZAxis.Params().Set(
                              rot_save_key='inverse_aug.global_rot.rot',
                              points_keys=['cell_center_xyz']))),
        insert_after='inverse_world_scaling')
    # Check if the order of InverseAug is correct.
    assert (p.preprocessors_order.index('random_apply_global_rot') <
            p.preprocessors_order.index('world_scaling'))
    assert (p.preprocessors_order.index('world_scaling') <
            p.preprocessors_order.index('global_loc_noise'))
    assert (p.preprocessors_order.index('global_loc_noise') <
            p.preprocessors_order.index('random_flip'))

    return p

  def _configure_input(self, p, split):
    p = super()._configure_input(p, split)

    if not self.FRAME_OFFSETS:
      # Add images when it is a single frame model.
      # When it is a multi frame model, the images are automitically added when
      # setting CAMERA_INPUT = True
      assert self.CAMERA_INPUT
      p.extractors.Define(
          'images', waymo_open_input_generator.WaymoImageExtractor.Params(), '')
    p.preprocessors.Define(
        'create_cell_center_xyz',
        input_preprocessors.CopyFeatures.Params().Set(
            source_key='lasers.points_xyz', target_key='cell_center_xyz'), '')
    p.preprocessors.Define(
        'cell_center_to_camera',
        waymo_open_input_generator.CellCenterToBestCamera.Params().Set(
            camera_names=[]), '')
    p.preprocessors.Define(
        'resize_images',
        waymo_open_input_generator.RescaleResizeImages.Params().Set(
            rescale=True,
            resize_ratio=0.3125,
            projected_points_keys=[
                'cell_center_projected.points_in_best_camera'
            ]), '')

    p.preprocessors_order += [
        'create_cell_center_xyz', 'cell_center_to_camera', 'resize_images'
    ]

    job_type = cluster_factory.Current().job
    if job_type.startswith('trainer') and self.APPLY_STRONG_DATA_AUG:
      p = self._apply_inverse_aug(p)
    return p

  def Task(self):
    p = super().Task()
    image_builder = deep_fusion.ImageFeatureExtractorBuilder()
    learnable_align_builder = deep_fusion.LearnableAlignBuilder(
        lidar_channels=self.NUM_OUTPUT_FEATURES)

    assert p.input_featurizer.cls == pillars.DynamicVoxelizationFeaturizer
    p.input_featurizer.Set(return_dynamic_voxels=True)

    p.input_featurizer = deep_fusion.MultiModalFeaturizer.Params().Set(
        pointcloud_featurizer=p.input_featurizer,
        image_featurizer=image_builder.ImageFeatureExtractor('image_tower'),
        fusion=learnable_align_builder.Fusion('fusion'),
        camera_feature_aligner=deep_fusion.DeepFusionAligner.Params().Set(
            q_embedding=learnable_align_builder.LidarEmbedding('q_embedding'),
            k_embedding=learnable_align_builder.ImageEmbedding('k_embedding'),
            v_embedding=learnable_align_builder.ImageEmbedding('v_embedding'),
            attn_dropout=learnable_align_builder.Dropout('attn_dropout'),
            learnable_align_fc=learnable_align_builder.FC('learnable_align_fc'),
        ))
    return p
