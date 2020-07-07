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
"""PointNet architecture.

PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
  https://arxiv.org/abs/1612.00593
"""

from lingvo import compat as tf
from lingvo.core import builder_layers
from lingvo.core import py_utils
from lingvo.tasks.car import builder_lib
import numpy as np


class PointNet(builder_lib.ModelBuilderBase):
  """Builder for PointNet layers."""

  def _ConcatWithBranch(self, name, tile, *subs):
    """Concat x with subs(x)."""

    def Merge(xs):
      act_self = xs[0][0]
      act_branch = xs[1][0]
      if tile:
        num_pts = tf.shape(act_self)[1]
        act_branch = tf.tile(tf.expand_dims(act_branch, 1), [1, num_pts, 1])
      return (tf.concat([act_self, act_branch], axis=-1),)

    return builder_layers.ParallelLayer.Params().Set(
        name=name,
        sub=[self._Seq('id'), self._Seq('branch', *subs)],
        merge=Merge)

  def _ConcatWithOnehot(self, name):
    """Make an onehot from shape category id and concat with point features.

    See original Pointnet paper for justification.

    Args:
      name: Name of the layer.

    Returns:
      Params for a layer that updates the NestedMap features value with
      the input features + a one hot encoding of the category id.
    """

    def _ConcatOnehotFn(input_data):
      """Concat the input features with a onehot version of the label ids."""
      features = input_data.features
      label = input_data.label
      num_pts = tf.shape(features)[1]
      label_one_hot = tf.one_hot(tf.cast(label, tf.int32), depth=16)
      label_one_hot = tf.tile(tf.expand_dims(label_one_hot, 1), [1, num_pts, 1])
      input_data.features = tf.concat([features, label_one_hot], axis=-1)
      return input_data

    return self._ApplyFn(name, fn=_ConcatOnehotFn)

  def _TNet(self, name, idims):
    return self._Matmul(
        name,
        self._Seq('iden'),
        self._Seq(
            'tran',
            self._MLP('mlp0', [idims, 64, 128, 1024]),
            self._Max('max'),
            self._MLP('mlp1', [1024, 512, 256]),
            # This layer is initialized as an identity transformation.
            self._Linear('linear', 256, idims * idims)
            .Set(params_init=py_utils.WeightInit.Constant(scale=0)),
            self._Bias('bias', idims * idims).Set(
                params_init=py_utils.WeightInit.Constant(
                    scale=np.eye(idims).reshape([-1]).tolist())),
            self._Reshape('reshape', [-1, idims, idims]),
            self._Fetch('transform_matrix')
        ))

  def Classifier(self,
                 name='pointnet',
                 keep_prob=0.7,
                 input_dims=3,
                 feature_dims=256):
    """PointNet architecture for classification."""
    # pyformat: disable
    p = self._Seq(
        name,
        self._SeqOnFeatures(
            'point_features',
            self._TNet('inp_trans', input_dims),
            self._Fetch('inp_transformed'),
            self._MLP('feat', [input_dims, 64, 64]),
            self._TNet('feat_trans', 64),
            self._MLP('mlp1', [64, 64, 128, 1024]),
            self._Fetch('pre_maxpool')),
        self._PaddedMax('max'),
        self._Seq(
            'cls_features',
            self._Fetch('feature'),
            self._FC('fc0', 1024, 512),
            self._Dropout('dropout', keep_prob),
            self._FC('fc1', 512, feature_dims)))
    # pyformat: enable

    p.Define('output_dim', feature_dims, 'Final output dimension.')
    return p

  def Segmentation(self, name='pointnet_segmentation'):
    """PointNet archetecture for segmentation."""
    # dropout is NOT used in the vanilla setting from the paper.
    # pyformat: disable
    main_tower = self._Seq(
        'main_tower',
        self._GetValue('get_features', builder_lib.FEATURES_KEY),
        self._TNet('inp_trans', 3),
        self._MLP('feat', [3, 64, 64]),
        self._TNet('feat_trans', 64),
        self._ConcatWithBranch(
            'concat1',
            True,
            self._MLP('mlp1', [64, 64, 128, 1024]),
            self._Max('max'),
            self._Fetch('feature')))
    segmentation_tower = self._Seq(
        name,
        main_tower,
        self._Fetch('point_feature'),
        self._MLP('final_mlp', [1088, 512, 256, 128]))
    # pyformat: enable
    p = segmentation_tower
    p.Define('output_dim', 128, 'Final output dimension.')
    return p

  def SegmentationShapeNet(self, name='pointnet_shapenet', keep_prob=0.8):
    """The modified network for part segmentation in ShapeNet."""
    main_tower = self._SeqOnFeatures(
        'main_tower_seq', self._TNet('inp_trans', 3), self._FC('fc1', 3, 64),
        self._ConcatWithBranch(
            'concat1', False, self._FC('fc2', 64, 128),
            self._ConcatWithBranch(
                'concat2', False, self._FC('fc3', 128, 128),
                self._ConcatWithBranch(
                    'concat3', False, self._TNet('fea_trans', 128),
                    self._ConcatWithBranch(
                        'concat4', False, self._FC('fc4', 128, 512),
                        self._ConcatWithBranch('concat5', True,
                                               self._FC('fc5', 512, 2048),
                                               self._Max('max')))))))
    p = self._Seq(name, main_tower, self._ConcatWithOnehot('concat_onehot'),
                  self._GetValue('get_features', builder_lib.FEATURES_KEY),
                  self._FC('final_fc1', 3024, 256),
                  self._Dropout('dropout1', keep_prob),
                  self._FC('final_fc2', 256, 256),
                  self._Dropout('dropout2', keep_prob),
                  self._FC('final_fc3', 256, 128))
    # pyformat: enabled
    p.Define('output_dim', 128, 'Final output dimension.')
    return p


class PointNetPP(builder_lib.ModelBuilderBase):
  """Builder for PointNet++ Model."""

  def _SetAbstractionWithMLPMax(self,
                                name,
                                mlp_dims,
                                num_samples,
                                group_size,
                                ball_radius,
                                sample_neighbors_uniformly=True):
    feature_extraction_sub = self._Seq(
        'mlpmax',
        self._ConcatPointsToFeatures('concat_feat_points'),
        self._FeaturesMLP('sa_MLP', mlp_dims),
        self._PaddedMax('sa_max_pool'))  # pyformat: disable
    return self._SetAbstraction(
        name=name,
        feature_extraction_sub=feature_extraction_sub,
        num_samples=num_samples,
        group_size=group_size,
        ball_radius=ball_radius,
        sample_neighbors_uniformly=sample_neighbors_uniformly)

  def _ModelNet40Featurizer(self, input_dims):
    return self._Seq(
        'sa_featurizer',
        self._SetAbstractionWithMLPMax(
            name='sa0',
            num_samples=512,
            ball_radius=0.2,
            group_size=32,
            sample_neighbors_uniformly=True,
            mlp_dims=[input_dims + 3, 64, 64, 128]),
        self._SetAbstractionWithMLPMax(
            name='sa1',
            num_samples=128,
            ball_radius=0.4,
            group_size=32,
            sample_neighbors_uniformly=True,
            mlp_dims=[128 + 3, 128, 128, 256]),
        self._ConcatPointsToFeatures('concat_feat_points'),
        self._FeaturesMLP('sa_MLP', [256 + 3, 256, 512, 1024]),
        self._PaddedMax('sa_max_pool'))

  def Classifier(self,
                 name='pointnetpp',
                 input_dims=3,
                 feature_dims=256,
                 keep_prob=0.8,
                 num_points=1024):
    """PointNet++ architecture for modelnet40 classification."""

    featurizer_sub = self._ModelNet40Featurizer(input_dims)
    p = self._Seq(
        name,
        featurizer_sub,
        self._FC('fc0', 1024, 512),
        self._Dropout('dropout0', keep_prob),
        self._FC('fc1', 512, feature_dims),
        self._Dropout('dropout1', keep_prob))  # pyformat: disable
    p.Define('output_dim', feature_dims, 'Final output dimension.')

    return p
