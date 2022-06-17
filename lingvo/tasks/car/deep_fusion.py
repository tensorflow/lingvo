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
r"""DeepFusion implementation.

[1] DeepFusion. https://arxiv.org/abs/2203.08195
"""

import functools

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.tasks.car import car_lib
from lingvo.tasks.car import pillars

import numpy as np


class ImageFeatureExtractorBuilder(pillars.Builder):
  """A very simple image feature extractor."""

  def ImageFeatureExtractor(self, name):
    """A simple UNet image feature extractor."""
    return self._Seq(name, self._Conv('c3x3', (3, 3, 3, 64)),
                     self.Backbone(64, down_strides=(1, 2, 2)))


class LearnableAlignBuilder(pillars.Builder):
  """LearnableAlign for multi-modality fusion.

  A cross-attention based Fusion op for LiDAR-RGB fusion.
  See Section 3.3 of DeepFusion paper for details.
  https://arxiv.org/pdf/2203.08195.pdf
  """

  def __init__(self, lidar_channels=64, image_channels=192, qkv_channels=128):
    super().__init__()
    self.lidar_channels = lidar_channels
    self.image_channels = image_channels
    self.qkv_channels = qkv_channels

  def Fusion(self, name):
    """A simple fusion module. The archtecture is a fully connected layer."""
    return self._FC(name, self.image_channels + self.lidar_channels,
                    self.lidar_channels)

  def LidarEmbedding(self, name):
    idims = self.lidar_channels
    odims = self.qkv_channels
    return self._FC(name, idims, odims, activation_fn=tf.identity)

  def ImageEmbedding(self, name):
    idims = self.image_channels
    odims = self.qkv_channels
    return self._FC(name, idims, odims, activation_fn=tf.identity)

  def Dropout(self, name, keep_prob=0.7):
    return self._Dropout(name, keep_prob)

  def FC(self, name):
    return self._FC(name, self.qkv_channels, odims=self.image_channels)


class MultiModalFeaturizer(base_layer.BaseLayer):
  """Layer for using dynamic voxelization, and RGB image to create pillars."""

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.image_featurizer
    assert p.pointcloud_featurizer
    assert p.fusion
    assert p.camera_feature_aligner
    self.CreateChild('image_featurizer', p.image_featurizer)
    self.CreateChild('pointcloud_featurizer', p.pointcloud_featurizer)
    self.CreateChild('fusion', p.fusion)
    self.CreateChild('camera_feature_aligner', p.camera_feature_aligner)

  @classmethod
  def Params(cls, *arg, **kwargs):
    p = super().Params(*arg, **kwargs)
    p.Define(
        'camera_names', [], 'Cameras to compute image features for. '
        'If specified, this is expected to correspond to the same order '
        'specified in the corresponding input preprocessor. If this is empty, '
        'then projections for *all* cameras present in the input_batch will '
        'be computed and the order will follow the sorted keys of '
        'input_batch.images.')
    p.Define('image_featurizer', None,
             'Params for a layer that featurizes the images.')
    p.Define('pointcloud_featurizer', None,
             'Params for a layer that featurizes the point cloud.')
    p.Define(
        'fusion', None, 'Params for a layer that fuses the features from '
        'the image and laser towers.')
    p.Define('camera_feature_aligner', SinglePointAligner.Params(),
             'Params for camera feature alignment.')
    return p

  def FProp(self, theta, input_batch):
    """Compute features for the pillars and convert them back to a dense grid.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` object containing input tensors. Required keys
        are `lasers.points_xyz`, `lasers.points_padding`,
        `lasers.points_feature`, `images`, and `cell_center_projected`.

    Returns:
      The dense features with shape [b, nx, ny, nz * fdims] and additionally,
    """
    p = self.params
    # Get the feature from PointCloud.
    if not self.pointcloud_featurizer.params.return_dynamic_voxels:
      assert isinstance(self.camera_feature_aligner, SinglePointAligner)
      assert not isinstance(self.camera_feature_aligner, MultiPointsAligner)
      assert not isinstance(self.camera_feature_aligner, DeepFusionAligner)

      pointcloud_featurized_cell = self.pointcloud_featurizer.FProp(
          theta.pointcloud_featurizer, input_batch)
    else:
      (pointcloud_featurized_cell, dynamic_voxels,
       _) = self.pointcloud_featurizer.FProp(theta.pointcloud_featurizer,
                                             input_batch)

    # Get the feature from RGB Image.
    all_cameras = car_lib.StackCameraImages(input_batch.images, p.camera_names)
    image_features = self.image_featurizer.FProp(theta.image_featurizer,
                                                 all_cameras)

    # Align image features to point cloud features.
    feat_ratio = car_lib.ComputeFeatureRatio(
        list(input_batch.images.values())[0].image, image_features)

    # Construct camera_feature_aligner for different Aligners.
    camera_feature_aligner = self.camera_feature_aligner
    if isinstance(self.camera_feature_aligner, MultiPointsAligner):
      camera_feature_aligner = functools.partial(
          camera_feature_aligner, dynamic_voxels=dynamic_voxels)
    if isinstance(self.camera_feature_aligner, DeepFusionAligner):
      camera_feature_aligner = functools.partial(
          camera_feature_aligner,
          dynamic_voxels=dynamic_voxels,
          featurized_cell=pointcloud_featurized_cell)

    image_featurized_cell = camera_feature_aligner(
        image_features,
        feat_ratio,
        points_projected=input_batch.cell_center_projected)

    image_featurized_cell = tf.reshape(
        image_featurized_cell,
        shape=py_utils.GetShape(pointcloud_featurized_cell)[:-1] +
        py_utils.GetShape(image_featurized_cell)[-1:])

    # Fuse them together.
    featurized_cell = tf.concat(
        [pointcloud_featurized_cell, image_featurized_cell], axis=-1)
    fused_cell = self.fusion.FProp(theta.fusion, featurized_cell)
    return fused_cell


class SinglePointAligner(base_layer.BaseLayer):
  """Align image features to point cloud features.

  For each point cloud feature (i.e., each pillar), we only extract ONE single
  corresponding image feature.
  """

  def FProp(self, theta, image_features, feat_ratio, points_projected):
    """Align image features to point cloud features.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      image_features: A float tensor with shape [batch_size, num_cameras, H, W,
        C] containing the features extracted from backbone network.
      feat_ratio: A float for indicating the ratio between the feature map size
        and original image size, assuming that assuming the height and width are
        scaled with the same ratio.
      points_projected: NestedMap of cameras_idx, points_in_best_camera, and
        mask.

    Returns:
      image_features_cell: The image feature aligned with the point cloud
      feature.
    """

    # Adjust col/row based on feature ratio.
    col, row = tf.unstack(
        points_projected.points_in_best_camera[..., :2], num=2, axis=-1)
    col *= feat_ratio
    row *= feat_ratio

    # TODO(jngiam): Consider doing local averaging or rounding as well.
    col = tf.cast(col, tf.int32)
    row = tf.cast(row, tf.int32)

    # Assemble indices to gather features from each camera, with the first
    # index being the camera index.
    gather_indices = tf.concat([
        points_projected.cameras_idx[..., tf.newaxis],
        row[..., tf.newaxis],
        col[..., tf.newaxis],
    ], axis=-1)  # pyformat: disable

    # Indices may be out of bounds, set to 0 by multiplying against mask to
    # avoid gather errors.
    gather_indices *= tf.cast(points_projected.mask[..., tf.newaxis], tf.int32)

    batch_size, = py_utils.GetShape(gather_indices, 1)
    _, feat_h, feat_w, feat_c = py_utils.GetShape(image_features)
    image_features = tf.reshape(image_features,
                                [batch_size, -1, feat_h, feat_w, feat_c])
    image_features_cell = tf.gather_nd(
        image_features, gather_indices, batch_dims=1)
    image_features_cell = image_features_cell * points_projected.mask[
        ..., tf.newaxis]
    return image_features_cell


class MultiPointsAligner(SinglePointAligner):
  """Align image features to point cloud features.

  For each point cloud feature (i.e., each pillar), we extract Multiple
  corresponding image features, and then equally average these image features.
  """

  def FProp(self, theta, image_features, feat_ratio, points_projected,
            dynamic_voxels):
    """Align image features to point cloud features.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      image_features: A float tensor with shape [batch_size, num_cameras, H, W,
        C] containing the features extracted from backbone network.
      feat_ratio: A float for indicating the ratio between the feature map size
        and original image size, assuming that assuming the height and width are
        scaled with the same ratio.
      points_projected: NestedMap of cameras_idx, points_in_best_camera, and
        mask.
      dynamic_voxels: A NestedMap corresponding to the output of running
        DynamicVoxelization on points_xyz.

    Returns:
      image_features_cell: The image feature aligned with the point cloud
      feature.
    """

    image_features_cell = super().FProp(theta, image_features, feat_ratio,
                                        points_projected)
    image_features_cell_sum = car_lib.BatchedUnsortedSegmentSum(
        image_features_cell,
        dynamic_voxels.indices,
        dynamic_voxels.num_voxels,
        batched_padding=dynamic_voxels.padding)

    cell_center_projected_mask_sum = car_lib.BatchedUnsortedSegmentSum(
        points_projected.mask[..., tf.newaxis],
        dynamic_voxels.indices,
        dynamic_voxels.num_voxels,
        batched_padding=dynamic_voxels.padding)

    zero_mask = tf.equal(cell_center_projected_mask_sum, 0.)
    cell_center_projected_mask_sum = tf.where_v2(
        zero_mask, tf.ones_like(cell_center_projected_mask_sum),
        cell_center_projected_mask_sum)
    image_features_cell_sum = tf.where_v2(
        zero_mask, tf.zeros_like(image_features_cell_sum),
        image_features_cell_sum)

    image_features_cell = image_features_cell_sum / cell_center_projected_mask_sum
    return image_features_cell


class DeepFusionAligner(SinglePointAligner):
  """Align image features to point cloud features.

  For each point cloud feature (i.e., each pillar), we extract Multiple
  corresponding image features, and then weighted sum these image features.
  The weights come from a cross-attention module.
  """

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    for param in [
        p.q_embedding, p.k_embedding, p.v_embedding, p.attn_dropout,
        p.learnable_align_fc
    ]:
      assert param
      self.CreateChild(param.name, param)

  @classmethod
  def Params(cls, *arg, **kwargs):
    p = super().Params(*arg, **kwargs)
    # See DeepFusion (https://arxiv.org/pdf/2203.08195.pdf) Section 3.3
    # paragraph LearnableAlign and Figure 1 for details.
    p.Define('q_embedding', None, 'The embedding function for Q^l.')
    p.Define('k_embedding', None, 'The embedding function for K^c.')
    p.Define('v_embedding', None, 'The embedding function for V^c.')
    p.Define('learnable_align_fc', None,
             'The fully connected layer in the LearnableAlign module')
    p.Define(
        'attn_dropout', None, 'Attention Dropout Module. See'
        'DeepFusion (https://arxiv.org/pdf/2203.08195.pdf)'
        'Section 4.1 paragraph LearnableAlign for details.')
    return p

  def FProp(self, theta, image_features, feat_ratio, points_projected,
            dynamic_voxels, featurized_cell):
    """Align image features to point cloud features.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      image_features: A float tensor with shape [batch_size, num_cameras, H, W,
        C] containing the features extracted from backbone network.
      feat_ratio: A float for indicating the ratio between the feature map size
        and original image size, assuming that assuming the height and width are
        scaled with the same ratio.
      points_projected: NestedMap of cameras_idx, points_in_best_camera, and
        mask.
      dynamic_voxels: A NestedMap corresponding to the output of running
        DynamicVoxelization on points_xyz.
      featurized_cell: A float tensor with shape [batch_size, pseudo_image_H,
        pseudo_image_W, C] containing the lidar feature extracted from backbone
        lidar feature extractor.

    Returns:
      image_features_cell: The image feature aligned with the point cloud
      feature.
    """
    image_features_cell = super().FProp(theta, image_features, feat_ratio,
                                        points_projected)
    # Compute (single-head) cross attention weights.
    # To illustrate the following tensor shape, we define B as batch size, N as
    # the maximum number of 3D points, and C (or C') as the number of channels.
    featurized_cell_shape = py_utils.GetShape(featurized_cell)
    flatten_featurized_cell = tf.reshape(featurized_cell, [
        featurized_cell_shape[0], featurized_cell_shape[1] *
        featurized_cell_shape[2], featurized_cell_shape[3]
    ])  # with shape [B, pseudo_image_H * pseudo_image_W, C]
    featurized_cell4attention = tf.gather(
        flatten_featurized_cell, dynamic_voxels.indices,
        batch_dims=1)  # with shape [B, N, C]

    q = self.q_embedding(featurized_cell4attention)  # with shape [B, N, C']
    k = self.k_embedding(image_features_cell)  # with shape [B, N, C']
    v = self.v_embedding(image_features_cell)  # with shape [B, N, C']

    affinity = tf.einsum('bnc,bnc->bn', q, k) / tf.sqrt(
        tf.cast(q.shape[-1], tf.float32))  # with shape [B, N]
    invalid_mask = (points_projected.mask * (1 - dynamic_voxels.padding)) < 0.5
    affinity = tf.where_v2(invalid_mask, -tf.ones_like(affinity) * np.inf,
                           affinity)

    # Do softmax on affinity to compute attention weights.
    max_affinity = car_lib.BatchedUnsortedSegmentMax(affinity,
                                                     dynamic_voxels.indices,
                                                     dynamic_voxels.num_voxels)
    max_affinity = tf.gather(max_affinity, dynamic_voxels.indices, batch_dims=1)
    e_affinity = tf.exp(affinity - max_affinity)
    e_affinity_sum = car_lib.BatchedUnsortedSegmentSum(
        e_affinity,
        dynamic_voxels.indices,
        dynamic_voxels.num_voxels,
        batched_padding=tf.cast(invalid_mask, tf.float32))
    e_affinity_sum = tf.gather(
        e_affinity_sum, dynamic_voxels.indices, batch_dims=1)
    weights = (e_affinity * (1. - tf.cast(invalid_mask, tf.float32))) / (
        e_affinity_sum + 1e-3 * tf.cast(invalid_mask, tf.float32))
    weights = self.attn_dropout(weights)

    retrieved_output_flatten = tf.einsum('bn,bnc->bnc', weights, v)
    retrieved_output_flatten = self.learnable_align_fc(retrieved_output_flatten)
    image_features_cell = car_lib.BatchedUnsortedSegmentSum(
        retrieved_output_flatten,
        dynamic_voxels.indices,
        dynamic_voxels.num_voxels,
        batched_padding=tf.cast(invalid_mask, tf.float32))
    return image_features_cell
