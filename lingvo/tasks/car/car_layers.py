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
"""Common layers for car models.

These are usually sub-classes of base_layer.BaseLayer used by builder_lib.
"""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.tasks.car import car_lib


class SamplingAndGroupingLayer(base_layer.BaseLayer):
  """Sampling and Grouping layer (based on PointNet++)."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'num_samples', 0,
        'Number of points to be sampled. Each sampled point will be '
        'returned with its corresponding group of neighbors.')
    p.Define('ball_radius', 0,
             'The distance around each sampled point to obtain neighbours.')
    p.Define('group_size', 0, 'Number neighbours for each sampled point.')
    p.Define(
        'sample_neighbors_uniformly', True,
        'Whether to sample neighbors uniformly within the ball radius. '
        'If False, this will pick the nearest neighbors by distance.')
    return p

  def FProp(self, theta, input_data):
    """Apply projection to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      input_data: A NestedMap object containing 'points', 'features', 'padding'
        Tensors, all of type tf.float32.
        'points': Shape [N, P1, 3]
        'features': Shape [N, P1, F]
        'padding': Shape [N, P1] where 0 indicates real, 1 indicates padded.

    Returns:
      A NestedMap consisting of the following two NestedMaps,
        grouped_points: consists of the grouped points, features and padding.
        query_points: consists of the sampled points and padding.
    """

    p = self.params
    features = input_data.features
    n, p1, c = py_utils.GetShape(features)
    points = py_utils.HasShape(input_data.points, [n, p1, 3])
    padding = py_utils.HasShape(input_data.padding, [n, p1])

    # Sampling
    sampled_idx, _ = car_lib.FarthestPointSampler(
        points, padding, num_sampled_points=p.num_samples)
    query_points = car_lib.MatmulGather(points, tf.expand_dims(sampled_idx, -1))
    query_points = tf.squeeze(query_points, -2)

    # Grouping
    grouped_idx, grouped_padding = car_lib.NeighborhoodIndices(
        points,
        query_points,
        p.group_size,
        points_padding=padding,
        max_distance=p.ball_radius,
        sample_neighbors_uniformly=p.sample_neighbors_uniformly)
    grouped_points = car_lib.MatmulGather(points, grouped_idx)
    # Normalize the grouped points based on the location of the query point.
    grouped_points -= tf.expand_dims(query_points, -2)
    grouped_features = car_lib.MatmulGather(features, grouped_idx)

    # Get the padding for the query points.
    query_padding = tf.array_ops.batch_gather(padding, sampled_idx)

    # Verify the shapes of output tensors.
    query_points = py_utils.HasShape(query_points, [n, p.num_samples, 3])
    query_padding = py_utils.HasShape(query_padding, [n, p.num_samples])
    grouped_features = py_utils.HasShape(grouped_features,
                                         [n, p.num_samples, p.group_size, c])
    grouped_padding = py_utils.HasShape(grouped_padding,
                                        [n, p.num_samples, p.group_size])

    output_grouped_points = py_utils.NestedMap(
        points=grouped_points,
        features=grouped_features,
        padding=grouped_padding)
    output_query = py_utils.NestedMap(
        points=query_points, padding=query_padding)
    output_map = py_utils.NestedMap({
        'grouped_points': output_grouped_points,
        'query_points': output_query
    })
    return output_map


class PointEncoder(base_layer.BaseLayer):
  """Layer to encode points based on points, dynamic voxels, and voxel stats."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('include_xyz', False, 'Add point coordinates as feature.')
    p.Define('include_xyz_norm_by_centroid', True,
             'Add point coordinates normalized by the centroid as feature.')
    p.Define(
        'include_xyz_norm_by_voxel_center', False,
        'Add point coordinates normalized by the voxel center as feature.')
    p.Define('include_features', True, 'Add point features as feature.')
    p.Define('include_centroid', True, 'Add the voxel centroids as feature.')
    p.Define('include_centers', True, 'Add the voxel centers as feature.')
    p.Define('include_covariance', False, 'Add the covariance as feature.')
    return p

  def NumEncodingFeatures(self, num_laser_features):
    """Computes the dimension of the augmented points based on the params."""
    p = self.params
    features_dim = {
        'include_features': num_laser_features,
        'include_xyz': 3,
        'include_xyz_norm_by_centroid': 3,
        'include_xyz_norm_by_voxel_center': 3,
        'include_centroid': 3,
        'include_centers': 3,
        'include_covariance': 9,
    }
    return sum(dim for name, dim in features_dim.items() if p.Get(name))

  def FProp(self, unused_theta, points_xyz, dynamic_voxels,
            dynamic_voxels_stats, points_feature):
    """Compute the encoding for each point.

    Args:
      unused_theta: The variables for this task.
      points_xyz: a floating point tf.Tensor with shape [batch_size, num_points,
        3], corresponding to xyz location per point.
      dynamic_voxels: the NestedMap object containing dynamic voxel tensors
        returned by the corresponding DynamicVoxelization operation. Required
        keys are 'centers', 'indices', and 'padding'.
      dynamic_voxels_stats: the NestedMap object containing the dynamic voxel
        statistic tensors returned by the corresponding ynamicVoxelization
        operation. Reuired keys are 'centered_xyz' and 'centroids'.
      points_feature: a floating point tf.Tensor with shape [batch_size,
        num_points, num_laser_features], corresponding to the features per
        point.

    Returns:
      The encoded points as tf.Tensor of shape
      [batch_size, num_points, encoding_size].
    """
    p = self.params
    features = {
        'include_xyz_norm_by_centroid': dynamic_voxels_stats.centered_xyz,
        'include_features': points_feature,
        'include_centroid': dynamic_voxels_stats.centroids,
        'include_centers': dynamic_voxels.centers,
        'include_xyz_norm_by_voxel_center': points_xyz - dynamic_voxels.centers,
        'include_xyz': points_xyz,
        'include_covariance': dynamic_voxels_stats.covariance,
    }
    # The order of the first four features corresponds to the initial default
    # for backwards compatibility.
    feature_order = [
        'include_xyz_norm_by_centroid', 'include_features', 'include_centroid',
        'include_centers', 'include_xyz_norm_by_voxel_center', 'include_xyz',
        'include_covariance'
    ]
    encoding = [features[name] for name in feature_order if p.Get(name)]
    return tf.concat(encoding, axis=-1)


class DynamicVoxelization(base_layer.BaseLayer):
  """Layer for using dynamic voxelization on a set of points."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('grid_size', (40, 40, 1), 'Grid size along x,y,z axis.')
    p.Define('grid_range_x', (0, 40), 'The X-axis Range covered by the grid')
    p.Define('grid_range_y', (-40, 40), 'The Y-axis Range covered by the grid')
    p.Define('grid_range_z', (-3, 3), 'The Z-axis Range covered by the grid')
    p.Define(
        'min_points_per_voxel', 1, 'Minimum number of points for a voxel '
        'to be considered active. Inactive voxels are set to 0.')
    p.Define('point_encoder', PointEncoder.Params(),
             'Layer to encoder the points.')
    p.Define(
        'featurizer', None, 'Featurizer on each voxel, this should be '
        'Params for a layer that expects as input a tensor of shape '
        '[..., num_features] and produces a tensor of shape '
        '[..., num_output_features]. num_features will be the encoding dims.')
    p.Define(
        'aggregation_ops', ('max',), 'Tuple of one or multiple '
        'aggregation operations to use on the featurized voxels. This op '
        'supports max, mean, and sum.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.point_encoder
    assert p.featurizer
    if p.min_points_per_voxel < 1:
      raise ValueError('min_points_per_voxel must be >= 1, '
                       'min_points_per_voxel={}'.format(p.min_points_per_voxel))
    if not set(p.aggregation_ops).issubset({'max', 'mean', 'sum'}):
      raise ValueError('p.aggregation_ops must be a subset of (max, mean, sum) '
                       'but is {}'.format(p.aggregation_ops))
    self.CreateChild('featurizer', p.featurizer)
    self.CreateChild('point_encoder', p.point_encoder)

  def _VoxelizeAndEncodePoints(self, theta, points_xyz, points_feature,
                               points_padding):
    p = self.params
    dynamic_voxels = car_lib.DynamicVoxelization(points_xyz, points_padding,
                                                 p.grid_size, p.grid_range_x,
                                                 p.grid_range_y, p.grid_range_z)

    dynamic_voxels_stats = car_lib.DynamicVoxelStatistics(
        points_xyz, dynamic_voxels)

    # Filter voxels that have too few points by marking all the points in those
    # voxels as padded points. If min points is set to 1, empty voxels will
    # be handled gracefully later and set to 0.
    if p.min_points_per_voxel > 1:
      dynamic_voxels.padding = tf.maximum(
          dynamic_voxels.padding,
          tf.cast(
              dynamic_voxels_stats.voxel_point_count < p.min_points_per_voxel,
              dtype=tf.float32))
    encoded_points = self.point_encoder.FProp(theta.point_encoder, points_xyz,
                                              dynamic_voxels,
                                              dynamic_voxels_stats,
                                              points_feature)

    # Save the encoded points for use with decoding.
    dynamic_voxels.encoded_points = encoded_points

    # Featurize each point.
    featurized_points = self.featurizer.FProp(
        theta.featurizer,
        py_utils.NestedMap(
            features=encoded_points, padding=dynamic_voxels.padding))

    return dynamic_voxels, dynamic_voxels_stats, featurized_points

  def _ComputeVoxelFeatures(self, dynamic_voxels, featurized_points):
    p = self.params
    aggregation_ops = {
        'max': car_lib.BatchedUnsortedSegmentMax,
        'mean': car_lib.BatchedUnsortedSegmentMean,
        'sum': car_lib.BatchedUnsortedSegmentSum
    }
    voxel_features = []
    for op_name in p.aggregation_ops:
      aggregation_fn = aggregation_ops[op_name]
      voxel_features.append(
          aggregation_fn(
              featurized_points,
              dynamic_voxels.indices,
              dynamic_voxels.num_voxels,
              batched_padding=dynamic_voxels.padding))
    voxel_features = tf.concat(voxel_features, axis=-1)

    # Handle empty voxels by setting them to zero.
    voxel_features = tf.where_v2(voxel_features <= voxel_features.dtype.min, 0.,
                                 voxel_features)

    return voxel_features

  def FProp(self, theta, points_xyz, points_feature, points_padding):
    """Compute features for the voxels and convert them back to a dense grid.

    Points are dynamically mapped to voxel locations, where for each voxel
    location. Each point is then augmented with additional information such as
    the distance of the point from the voxel center, the mean of the points
    assigned, etc. We use a MLP (featurizer) to compute new features for each
    point. Finally, each voxel is summarized using a max over points for
    each feature.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      points_xyz: a floating point tf.Tensor with shape [batch_size, num_points,
        3], corresponding to xyz location per point.
      points_feature: a floating point tf.Tensor with shape [batch_size,
        num_points, num_laser_features], corresponding to the features per
        point.
      points_padding: a floating point tf.Tensor with shape [batch_size,
        num_points] corresponding to a padding tensor per point, where 1 (True)
        represents that the point is padded.

    Returns:
      - A voxelization of the points with shape [batch_size, nx, ny, nz,
        num_output_features], where (nx, ny, nz) correspond to elements of
        grid_size,
      - the dynamic voxel statistics, a NestedMap returned by
        car_lib.DynamicVoxelStatistics,
      - the dynamic voxels, a NestedMap returned by
        car_lib.DynamicVoxelization.
    """
    p = self.params
    batch_size = py_utils.GetShape(points_xyz)[0]
    nx, ny, nz = p.grid_size

    dynamic_voxels, dynamic_voxels_stats, featurized_points = (
        self._VoxelizeAndEncodePoints(theta, points_xyz, points_feature,
                                      points_padding))

    voxel_features = self._ComputeVoxelFeatures(dynamic_voxels,
                                                featurized_points)

    num_output_features = py_utils.GetShape(voxel_features)[-1]
    voxel_features = tf.reshape(voxel_features,
                                [batch_size, nx, ny, nz, num_output_features])
    return voxel_features, dynamic_voxels, dynamic_voxels_stats
