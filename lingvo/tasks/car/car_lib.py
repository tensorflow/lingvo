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
"""Library of functions on tensors for car layers, builders, and models."""


import functools
# pylint:enable=g-direct-tensorflow-import
import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.tasks.car import geometry
import numpy as np


def SquaredDistanceMatrix(pa, pb, mem_optimized=False):
  """Compute pair-wise squared distances.

  Expanded version (faster but potentially numerically unstable):
    distance = pa^2 - 2*pa*pb + pb^2

  Non-expanded version (slow but numerically stable):
    distance = (pa - pb)^2

  Args:
    pa: tensor of shape [N, P1, dims]
    pb: tensor of shape [N, P2, dims]
    mem_optimized: Whether to use the memory-optimized expanded formulation.
      Defaults to False. If enabled, the expanded version is used that
      may have numerical issues.

  Returns:
    tensor of shape [N, P1, P2]
  """

  def _ExpandedSquaredDistanceMatrix(pa, pb):
    squared_pa = tf.reduce_sum(tf.square(pa), axis=2, keepdims=True)
    squared_pb = tf.transpose(
        tf.reduce_sum(tf.square(pb), axis=2, keepdims=True), perm=[0, 2, 1])
    # We have observed that entries may < 0. when using the expanded version.
    # The max operation guards that from being possible.
    return tf.maximum(
        squared_pa - 2 * tf.matmul(pa, pb, transpose_b=True) + squared_pb, 0.0)

  def _NonExpandedSquaredDistanceMatrix(pa, pb):
    diff = tf.expand_dims(pa, axis=2) - tf.expand_dims(pb, axis=1)
    squared_diff = tf.square(diff)
    squared_dis = tf.reduce_sum(squared_diff, axis=3)
    return squared_dis

  if mem_optimized:
    return _ExpandedSquaredDistanceMatrix(pa, pb)
  else:
    return _NonExpandedSquaredDistanceMatrix(pa, pb)


def NeighborSquaredDistanceMatrix(points, neighbor_points):
  """Compute the squared distance matrix between points and their neighbors.

  Args:
    points: A float tf.Tensor of shape [N, P1, 3] with point positions.
    neighbor_points: A float tf.Tensor fo shape [N, P1, K, 3] with neighbor
      positions.

  Returns:
    Squared distance matrix between points and their K nearest neighbors
    as a float tf.Tensor of shape [N, P1, K].
  """
  points = py_utils.HasShape(points, [-1, -1, 3])
  n, p1 = py_utils.GetShape(points, 2)
  neighbor_points = py_utils.HasShape(neighbor_points, [n, p1, -1, 3])
  _, _, k = py_utils.GetShape(neighbor_points, 3)

  sq_diff = tf.square(neighbor_points - tf.reshape(points, [n, p1, 1, 3]))
  sq_dist = tf.reduce_sum(sq_diff, axis=3)
  return py_utils.HasShape(sq_dist, [n, p1, k])


def KnnIndices(points, query_points, k, valid_num=None, max_distance=None):
  """k-nearest neighbors of query_points in points.

  The caller should ensure that points[i, :valid_num[i], :] are the non-padding
  points.

  Padding is returned alongside indices. Non-padded points are guaranteed to
  be unique (non-repeated) points from original non-padded points.

  Padded points arise due to either a lack of points (k exceeds valid_num)
  or points are too far away (exceeds max distance).

  TODO(weihan,jngiam): For backwards compatibility with PointCNN, if there are
  fewer than k points to select (possibly because of valid_num), the points
  selected will first consist of those in the non-padded points, and
  then those from the padded points. This assumes that the padded points are
  duplications of the original points. PointCNN should be updated to respect
  padding.

  The auxiliary input 'valid_num' marks the number of non-padding points in each
  sample. This is needed because we randomly duplicated points to make the input
  fix-sized, we want search for k-NN in non-padding points first otherwise the
  result may degenerate to be k-duplications of the query point itself.

  Args:
    points: tensor of shape [N, P1, dims].
    query_points: tensor of shape [N, P2, dims]
    k: Integer.
    valid_num: tensor of shape [N,]
    max_distance: float representing the maximum distance that each neighbor can
      be. If there are no points within the distance, then the closest point is
      returned (regardless of distance). If this is set to None, then
      max_distance is not used.

  Returns:
    A pair of tensors:

    - indices: tensor of shape [N, P2, k].
    - padding: tensor of shape [N, P2 ,k] where 1 represents a padded point, and
      0 represents an unpadded (real) point.

  """
  p1 = tf.shape(points)[1]
  padding = None
  if valid_num is not None:
    padding = tf.greater_equal(tf.range(p1), tf.expand_dims(
        valid_num, -1))  # [N, P1], False/True padding
  return NeighborhoodIndices(points, query_points, k, padding, max_distance)


def NeighborhoodIndices(points,
                        query_points,
                        k,
                        points_padding=None,
                        max_distance=None,
                        sample_neighbors_uniformly=False):
  """Get indices to k-neighbors of query_points in points.

  Padding is returned along-side indices. Non-padded points are guaranteed to
  be unique (non-repeated) points from original non-padded points.

  Padded points arise due to either a lack of points (k exceeds the number
  of original non-padded points) or points are too far away (exceeds max
  distance).

  Note: Padded point indices may refer to padded points from the original, or
  may be duplicates of the closest point.

  TODO(weihan,jngiam): PointCNN implementation makes an assumption that padded
  points are repeated points from the original points. This behavior is
  maintained here, but we should update PointCNN to respect indices paddings.

  Args:
    points: tensor of shape [N, P1, dims].
    query_points: tensor of shape [N, P2, dims]
    k: Integer.
    points_padding: optional tensor of shape [N, P1] containing True/1.0 iff the
      point is a padded point. if None, then all points are considered real
      points.
    max_distance: float representing the maximum distance that each neighbor can
      be. If there are no points within the distance, then the closest point is
      returned (regardless of distance). If this is set to None, then no
      filtering by distance is performed.
    sample_neighbors_uniformly: boolean specifying whether to sample neighbors
      uniformly if they are within max distance.

  Returns:
    A pair of tensors:

    - indices: tensor of shape [N, P2, k].
    - padding: tensor of shape [N, P2, k] where 1 represents a padded point, and
      0 represents an unpadded (real) point.

  """
  n, p1 = py_utils.GetShape(points, 2)
  query_points = py_utils.HasShape(query_points, [n, -1, -1])
  _, p2 = py_utils.GetShape(query_points, 2)

  # Compute pair-wise squared distances.
  # Note that dist_mat contains the squared distance (without sqrt). Thus, when
  # using max_distance, we will need to square max_distance to make sure it's
  # in the same units.
  dist_mat = SquaredDistanceMatrix(query_points, points)
  dist_mat = py_utils.HasShape(dist_mat, [n, p2, p1])

  # Add a large scalar to the distances for padded points.
  # dist_mat[i, j, k] will be:
  #   if k < valid_num[i]: distance between points[i, k] and query_points[i, j]
  #   otherwise:           a large scalar added to dist_mat[i, j, k]
  if points_padding is not None:
    points_padding = tf.cast(tf.expand_dims(points_padding, 1), tf.float32)
    points_padding = py_utils.HasShape(points_padding, [n, 1, p1])
    large_scalar = tf.reduce_max(dist_mat) + 1
    dist_mat += points_padding * large_scalar

  # To perform sampling neighbors uniformly efficiently, we set all neighbors
  # that are within the distance threshold to have distances be drawn uniformly
  # at random. Using top_k with this enables selecting a random set quickly
  # without replacement.
  if sample_neighbors_uniformly:
    if max_distance is not None:
      mask_by_distance = tf.less_equal(dist_mat, max_distance**2)
      dist_mat = tf.where(
          mask_by_distance,
          tf.square(max_distance) * tf.random.uniform(tf.shape(dist_mat)),
          dist_mat)
    else:
      raise ValueError('Uniform sampling requires specifying max_distance.')

  top_k_dist, indices = tf.nn.top_k(-dist_mat, k=k, sorted=True)  # N x P2 x K

  # Set padding using top_k_dist; padded points will have distance exceeding
  # the large_scalar.
  if points_padding is not None:
    paddings = tf.greater_equal(-top_k_dist, large_scalar)
  else:
    paddings = tf.zeros_like(top_k_dist, dtype=tf.bool)

  # Filter by max_distances by setting all indices that exceed the max_distance
  # to the closest point.
  if max_distance is not None:
    # Mask is true for points that are further than max_distance.
    mask_by_distance = tf.greater(-top_k_dist, tf.square(max_distance))
    closest_idx = tf.tile(indices[:, :, :1], [1, 1, k])
    indices = tf.where(mask_by_distance, closest_idx, indices)
    paddings |= mask_by_distance

  indices = tf.reshape(indices, [n, p2, k])
  paddings = tf.cast(paddings, tf.float32)

  return indices, paddings


def MatmulGather(source, indices):
  """Drop in replacement for tf.gather_nd() optimized for speed on TPU.

  TODO(weihan): tf.gather_nd() is supposed to be implemented in the same way
  on TPU. Investigate why it's much slower.

  Args:
    source: tensor of shape [N, P1, C]
    indices: tensor of shape [N, P2, K]

  Returns:
    tensor of shape [N, P2, K, C]
  """
  source = py_utils.HasRank(source, 3)
  n, p1, c = py_utils.GetShape(source)
  indices = py_utils.HasShape(indices, [n, -1, -1])
  _, p2, k = py_utils.GetShape(indices)

  onehot = tf.one_hot(indices, depth=p1)  # N x P2 x K x P1
  reshaped = tf.reshape(onehot, [n, -1, p1])  # N x (P2 x K) x P1
  target = tf.matmul(reshaped, source)  # N x (P2 x K) x C
  return tf.reshape(target, [n, p2, k, c])


def FarthestPointSampler(points,
                         padding,
                         num_sampled_points,
                         precomputed_squared_distance=None,
                         num_seeded_points=0,
                         random_seed=None):
  """Samples num_sampled_points from points using farthest point sampling.

  Algorithm:
  1. Start by selecting a random point and adding to a selected set.
  2. For all remaining points, find the furthest point from those selected.
  3. Add furthest point to selected.
  4. Repeat 2-3 until num_sampled_points are selected.

  More details at https://en.wikipedia.org/wiki/Farthest-first_traversal

  This output of this function can be used with tf.array_ops.batch_gather to
  extract the desired points, for example:
  tf.array_ops.batch_gather(points, sampled_idx)

  Args:
    points: floating point tf.Tensor of shape [N, P1, dims]
    padding: A floating point tf.Tensor of shape [N, P1] with 0 if the point is
      real, and 1 otherwise.
    num_sampled_points: integer number of points to sample.
    precomputed_squared_distance: optional tf.Tensor of shape [N, P1, P1] of
      distances between each point. if None, distances will be computed on the
      fly.
    num_seeded_points: If num_seeded_points > 0, then the first
      num_seeded_points in points are considered to be seeded in the FPS
      sampling. Note that we assume that these points are *not* padded, and do
      not check padding when seeding them.
    random_seed: optional integer random seed to use with all the random ops.

  Returns:
    A tuple of tf.Tensors (sampled_idx, closest_idx) of types
    (tf.int32, tf.int32).

    sampled_idx is of shape [N, num_sampled_points] representing the indices
    selected using the sampler. This will have range of [0, P1].

    closest_idx is of shape [N, P1] representing the indices of the closest
    sampled points for each input point. closest_idx is used in PCNN as part of
    the pooling operation: each point is assigned to the closest sampled point
    and a max is taken over them. This will have a range of [0, P2] with the
    index of the closest sampled point that remains.
  """
  points = py_utils.HasRank(points, 3)
  batch_size, num_points, dims = py_utils.GetShape(points, 3)

  points = py_utils.with_dependencies(
      [py_utils.assert_greater_equal(num_points, num_sampled_points)], points)

  # Add a tiny bit of noise to the distance matrix or points so all
  # points are unique. This will also ensure true repeated points
  # like padded points are only selected after all valid points are selected.
  if precomputed_squared_distance is not None:
    precomputed_squared_distance = py_utils.HasShape(
        precomputed_squared_distance, [batch_size, num_points, num_points])
    precomputed_squared_distance += tf.random.uniform(
        (batch_size, num_points, 1),
        minval=1e-6,
        maxval=1e-5,
        dtype=tf.float32,
        seed=random_seed)
  else:
    points += tf.random.uniform((batch_size, num_points, dims),
                                minval=1e-6,
                                maxval=1e-5,
                                dtype=tf.float32,
                                seed=random_seed)

  # TensorArray to store the sampled indices in the loop.
  sampled_idx = tf.TensorArray(tf.int32, num_sampled_points)

  # Initialize distance_to_selected to inf for all points.
  distance_to_selected = float('inf') * tf.ones((batch_size, num_points))

  # For tracking the index to the closest selected point.
  closest_idx = tf.zeros((batch_size, num_points), dtype=tf.int32)

  # Current loop index counter.
  curr_idx = tf.constant(0, dtype=tf.int32)

  # Get number of valid points (1 is padded, so num_points - num_padded).
  num_valid_points = tf.cast(
      tf.cast(num_points, dtype=tf.float32) - tf.reduce_sum(padding, axis=1),
      dtype=tf.int32)

  def _BodyFn(curr_idx, distance_to_selected, sampled_idx, closest_idx):
    """Loop body for farthest point sampler."""

    def _GetRandomRealPoint():
      """Select the first point.

      For the first point, we want any random real (non padded) point, so we
      create a random values per point, and then set all padded ones to
      some large value (more than the maxval). We then take the min per batch
      element to get the first points.

      Returns:
        Tensor containing the index of a random point selected for each example
        in the batch.
      """
      random_values = tf.random.uniform((batch_size, num_points),
                                        minval=0,
                                        maxval=1,
                                        dtype=tf.float32,
                                        seed=random_seed)
      random_values = tf.where(
          tf.equal(padding, 0.0), random_values, padding * 10)
      return tf.argmin(random_values, axis=1, output_type=tf.int32)

    def _GetFurthestPoint():
      """Get point that is furthest from those already selected.

      We also bias the sampling towards real points by setting the distance
      to padded points negative until we are out of real points.

      Returns:
        Tensor containing the index of the next farthest point selected for each
        example in the batch.
      """
      # Set padded points distance to negative so they aren't selected.
      padding_masked_distance_to_selected = tf.where(
          tf.equal(padding, 0.0), distance_to_selected, -1.0 * tf.ones(
              (batch_size, num_points), dtype=tf.float32))
      # But only do this when we still have valid points left.
      padding_masked_distance_to_selected = tf.where(
          tf.less(curr_idx, num_valid_points),
          padding_masked_distance_to_selected, distance_to_selected)
      return tf.argmax(
          padding_masked_distance_to_selected, axis=-1, output_type=tf.int32)

    def _GetSeededPoint():
      """Select a seeded point.

      Seeded points are assumed to be at the beginning of the original points.

      Returns:
        Tensor containing the index of the next seeded point to select for each
        example in the batch.
      """
      return tf.ones((batch_size,), dtype=tf.int32) * curr_idx

    # Select indices for this loop iteration.
    def _Seeded():
      return tf.cond(
          tf.less(curr_idx, num_seeded_points), _GetSeededPoint,
          _GetFurthestPoint)

    def _Real():
      return tf.cond(
          tf.equal(curr_idx, 0), _GetRandomRealPoint, _GetFurthestPoint)

    new_selected = tf.cond(tf.greater(num_seeded_points, 0), _Seeded, _Real)
    sampled_idx = sampled_idx.write(curr_idx, new_selected)

    # Extract the distance to the latest point selected to update
    # distance_to_selected.
    new_selected_gather_idx = tf.stack([tf.range(batch_size), new_selected],
                                       axis=1)
    if precomputed_squared_distance is not None:
      new_distance = tf.gather_nd(precomputed_squared_distance,
                                  new_selected_gather_idx)
    else:
      new_points = tf.reshape(
          tf.gather_nd(points, new_selected_gather_idx), [batch_size, 1, dims])
      new_distance = tf.reshape(
          SquaredDistanceMatrix(points, new_points), [batch_size, num_points])

    is_newly_closest = tf.less(new_distance, distance_to_selected)
    distance_to_selected = tf.minimum(distance_to_selected, new_distance)

    # Track the index to the closest selected point.
    new_selected_tiled = tf.tile([[curr_idx]], [batch_size, num_points])
    closest_idx = tf.cond(
        tf.equal(curr_idx, 0),
        # At the first loop iteration, the init points are the closest.
        lambda: new_selected_tiled,
        # Otherwise, update with the new points based on the distances.
        lambda: tf.where(is_newly_closest, new_selected_tiled, closest_idx))
    return curr_idx + 1, distance_to_selected, sampled_idx, closest_idx

  _, _, sampled_idx, closest_idx = tf.while_loop(
      lambda curr_idx, *args: tf.less(curr_idx, num_sampled_points),
      _BodyFn,
      loop_vars=(curr_idx, distance_to_selected, sampled_idx, closest_idx),
      back_prop=False,
      maximum_iterations=num_sampled_points)

  sampled_idx = sampled_idx.stack()  # num_sampled_points x n
  sampled_idx = tf.transpose(sampled_idx, [1, 0])

  if isinstance(batch_size, int) and isinstance(num_sampled_points, int):
    sampled_idx.set_shape((batch_size, num_sampled_points))

  return sampled_idx, closest_idx


# TODO(bencaine): This was moved so that we can make this more generic in the
# future and provide min/avg/max pooling with one function.
def MaxPool3D(points, point_features, pooling_idx, closest_idx):
  """Apply max pooling to a point cloud with computed sampling indices.

  sampled_idx and closest_idx are the outputs of a sampler such as
  FurthestPointSampler.

  The pooling operation results in a point cloud with fewer points, where the
  pooled points are specified by pooling_idx. Each element of pooling_idx
  contains an integer in the range [0, P1) containing the index of the point in
  points/points_features.

  Max pooling is performed by assigning each point to its closest pooled point,
  and then taking a max over the features of points assigned. We assume that
  this mapping is provided by closest_idx, where each element should contain
  an integer in the range [0, P2) containing the index of the pooled point that
  each point is assigned to.

  Note: This logic for pooling assumes that there will be at least
  one value > 0 per sampled region for each feature, otherwise it will return 0.
  Additionally, it does a reduce over a masked version of the features, so
  mean and min would not work without a change in the logic.

  Args:
    points: a floating point tf.Tensor with shape [N, P1, 3]
    point_features: a floating point tf.Tensor with shape [N, P1, C]
    pooling_idx: A tf.int32 tf.Tensor of shape [N, P2] with the index of which
      points we want to keep. Each value should be in the range [0, P1].
    closest_idx: A tf.int32 tf.Tensor of shape [N, P1] representing which
      sampled point is closest to each original point. Each value should be in
      the range of [0, P2].

  Returns:
    A tuple of tf.Tensors (pooled_points, pooled_features).

    pooled_points has shape [N, P2, 3] representing the locations of each
    selected point. P2 corresponds to num_pooled_points.

    pooled_features has shape [N, P2, C] representing the pooled features at
    each point.
  """
  batch_size, num_points = py_utils.GetShape(points, 2)
  point_features = py_utils.HasShape(point_features,
                                     [batch_size, num_points, -1])
  pooling_idx = py_utils.HasShape(pooling_idx, [batch_size, -1])
  _, num_output_points = py_utils.GetShape(pooling_idx)
  _, _, feature_dims = py_utils.GetShape(point_features, 3)

  # Gather new point locations.
  pooled_points = tf.array_ops.batch_gather(points, pooling_idx)

  mask = tf.one_hot(closest_idx, num_output_points)  # [N, P1, P2]
  mask = tf.transpose(mask, [2, 0, 1])  # [P2, N, P1]

  def _PartialPoolFeaturesFn(partial_mask):
    partial_mask = tf.tile(
        tf.reshape(partial_mask, [batch_size, num_points, 1]),
        [1, 1, feature_dims])
    # Note: This method of pooling assumes there will be a value > 0
    # And will only work with max under this condition.
    return tf.reduce_max(partial_mask * point_features, axis=1)

  # Performing a map_fn over the pooled points is more memory efficient.
  pooled_point_features = tf.map_fn(_PartialPoolFeaturesFn, mask)  # [P2, N, P1]
  pooled_point_features = tf.transpose(pooled_point_features, [1, 0, 2])

  return pooled_points, pooled_point_features


def SegmentPool3D(points,
                  point_features,
                  pooling_idx,
                  closest_idx,
                  pooling_method='max'):
  """Performs {min/max/average} pooling over a pointcloud given indices.

  This should be functionally identical when using max to the above
  MaxPool3D function, except it turns out to be much more memory efficient
  on a TPU, and supports min/max/mean.

  Args:
    points: A float tf.Tensor of shape [N, P1, 3] with point locations.
    point_features: A float tf.Tensor of shape [N, P1, C] with point features.
    pooling_idx: A tf.int32 tf.Tensor of shape [N, P2] with the index of which
      points we want to keep. Each value should be in the range [0, P1].
    closest_idx: A tf.int32 tf.Tensor of shape [N, P1] representing which
      sampled point is closest to each original point. Each value should be in
      the range of [0, P2].
    pooling_method: A string for which pooling function to use. Should be one of
      {'min', 'max', 'mean'}.

  Returns:
    pooled_points: A float tf.Tensor of shape [N, P2, 3] with the pooled
      point locations.
    pooled_features: A float tf.Tensor of shape [N, P2, C] with the pooled
      features.
  Raises:
    ValueError: If pooling_method is not one of {min/max/mean}.
  """
  segment_pooling_functions = {
      'min': tf.math.unsorted_segment_min,
      'max': tf.math.unsorted_segment_max,
      'mean': tf.math.unsorted_segment_mean
  }

  if pooling_method not in segment_pooling_functions:
    raise ValueError('`pooling_method` must be one of {}.'.format(
        list(segment_pooling_functions.keys())))
  segment_fn = segment_pooling_functions[pooling_method]

  points = py_utils.HasShape(points, [-1, -1, 3])
  n, p1 = py_utils.GetShape(points, 2)
  point_features = py_utils.HasShape(point_features, [n, p1, -1])
  _, _, c = py_utils.GetShape(point_features)
  pooling_idx = py_utils.HasShape(pooling_idx, [n, -1])
  _, p2 = py_utils.GetShape(pooling_idx)
  closest_idx = py_utils.HasShape(closest_idx, [n, p1])

  # Subselect our output points
  pooled_points = tf.array_ops.batch_gather(points, pooling_idx)

  # Loop over batch dimension of our features/indices, as unsorted_segment_X
  # does not currently support a batch dimension.
  def _LoopFn(args):
    example_features, example_closest_idx = args
    return segment_fn(example_features, example_closest_idx, num_segments=p2)

  pooled_features = tf.map_fn(
      fn=_LoopFn, elems=(point_features, closest_idx), dtype=tf.float32)

  return (py_utils.HasShape(pooled_points, [n, p2, 3]),
          py_utils.HasShape(pooled_features, [n, p2, c]))


def WhereBroadcast(conditional, true_result, false_result):
  """Perform a tf.where, but with a conditional that's not the full rank.

  Args:
    conditional: A boolean tf.Tensor whose shape is a prefix of
      true_result.shape. Additional dimensions will be added to the end of this
      tensor, and it will be broadcasted to match the shape of true_result.
    true_result: The tensor values to return if True. This tensor should have
      the same dtype and shape as false_result.
    false_result: The tensor values to return if False. This tensor should have
      the same dtype and shape as true_result.

  Returns:
    Result of the where clause with shape and type the same as true_result
    and false_result.

  Raises:
    ValueError: If the conditional has higher rank than the result.
  """
  conditional_shape = py_utils.GetShape(conditional)
  result_shape = py_utils.GetShape(true_result)
  false_result = py_utils.HasShape(false_result, result_shape)
  conditional_rank = len(conditional_shape)
  result_rank = len(result_shape)

  if conditional_rank > result_rank:
    raise ValueError('Conditional should be rank <= result tensors.')
  elif conditional_rank < result_rank:
    # Expand dimensions to match result, and then broadcast.
    conditional = tf.reshape(
        conditional, conditional_shape + [1] * (result_rank - conditional_rank))
    conditional = tf.broadcast_to(conditional, result_shape)

  return tf.where(conditional, true_result, false_result)


def DynamicVoxelStatistics(points_xyz, dynamic_voxels):
  """Computes first and second order voxel statistics for each point.

  After dynamic voxelization, we can further compute statistics for each voxel,
  such as the mean of points xyz coordinates assigned to each voxel (centroids),
  the covariance (second order moment) of the points assigned to each voxel, and
  also the points locations normalized to their centroids.

  This function computes these three statistics and maps it back to each point
  so that we can use them as additional features per point. After calling this
  function, one can create a new feature representation for each point by
  concatenating the features here with the original features.

  References:
    https://arxiv.org/pdf/1711.06396.pdf Figure 3.
    https://arxiv.org/pdf/1812.05784.pdf Figure 2.

  Example usage:

    dynamic_voxels = DynamicVoxelization(
        points_xyz, grid_size, grid_range_x, grid_range_y, grid_range_z)
    dynamic_voxel_statistics = DynamicVoxelStatistics(
        points_xyz, dynamic_voxels)

    new_points_features = tf.concat([
        points_feature,
        dynamic_voxel_statistics.centered_xyz,
        dynamic_voxel_statistics.centroids,
        dynamic_voxel_statistics.covariance,
        dynamic_voxel.centers,
    ], axis=-1)

  Args:
    points_xyz: A float tf.Tensor of shape [batch_size, num_points, 3]
      representing point positions in (x, y, z) dimensions.
    dynamic_voxels: A NestedMap corresponding to the output of running
      DynamicVoxelization on points_xyz.

  Returns:
    A NestedMap with keys:
      centered_xyz: A float tf.Tensor of shape [batch_size, num_points, 3]
        containing voxel-centered (x, y, z) coordinate for each point.
      centroids: A float tf.Tensor of shape [batch_size, num_points, 3]
        containing the mean for all points xyz assigned to the same voxel that
        the corresponding point is assigned to.
      covariance: A float tf.Tensor of shape [batch_size, num_points, 9]
        containing the covariance for all points xyz assigned to the same voxel
        that the corresponding point is assigned to.
      voxel_point_count: A float tf.Tensor of shape [batch_size, num_points]
        with the number of valid points inside each voxel per point.
      points_per_voxel: A float tf.Tensor of shape [batch_size, num_voxels]
        with the number of valid points inside each voxel.
      voxel_centroids: A float tf.Tensor of shape [batch_size, num_voxels, 3]
        containing the mean for all points xyz assigned to the that voxel.
  """
  batch_size, num_points, _ = py_utils.GetShape(points_xyz)

  # Compute centroids of each voxel.
  voxel_centroids = BatchedUnsortedSegmentMean(
      points_xyz,
      dynamic_voxels.indices,
      dynamic_voxels.num_voxels,
      batched_padding=dynamic_voxels.padding)

  # Map the centroid back to each point.
  point_centroids = tf.array_ops.batch_gather(voxel_centroids,
                                              dynamic_voxels.indices)

  # Normalize the points so that they have origin at their voxel centroid.
  points_xyz -= point_centroids

  # Count number of points assigned to each voxel.
  points_per_voxel = BatchedUnsortedSegmentSum(
      tf.ones((batch_size, num_points), dtype=tf.int32),
      dynamic_voxels.indices,
      dynamic_voxels.num_voxels,
      batched_padding=dynamic_voxels.padding)
  voxel_point_count = tf.array_ops.batch_gather(points_per_voxel,
                                                dynamic_voxels.indices)

  # Compute second order moment at each point, and them sum over the voxel to
  # obtain the second order moment for each voxel.
  points_outer_prod = (
      points_xyz[..., :, tf.newaxis] * points_xyz[..., tf.newaxis, :])
  points_outer_prod = tf.reshape(points_outer_prod, [batch_size, num_points, 9])
  voxel_covariance = BatchedUnsortedSegmentMean(
      points_outer_prod,
      dynamic_voxels.indices,
      dynamic_voxels.num_voxels,
      batched_padding=dynamic_voxels.padding)
  points_covariance = tf.array_ops.batch_gather(voxel_covariance,
                                                dynamic_voxels.indices)

  dynamic_voxel_statistics = py_utils.NestedMap(
      centroids=point_centroids,
      centered_xyz=points_xyz,
      covariance=points_covariance,
      voxel_point_count=voxel_point_count,
      points_per_voxel=points_per_voxel,
      voxel_centroids=voxel_centroids)

  return dynamic_voxel_statistics


def DynamicVoxelization(points_xyz, points_padding, grid_size, grid_range_x,
                        grid_range_y, grid_range_z):
  """Computes dynamic voxelization mappings for each point.

  When using dynamic voxelization, we do not explicitly create the voxelization
  tensors. Instead, we store indices that map to the voxels and use unsorted
  segment aggregation functions to compute values for each voxel. This function
  computes the mappings needed for using unsorted segment functions.

  Args:
    points_xyz: A float tf.Tensor of shape [batch_size, num_points, 3]
      representing point positions in (x, y, z) dimensions.
    points_padding: A float tf.Tensor of shape [batch_size, num_points]
      containing 1.0 if the corresponding point is a padded point, and 0.0 if it
      is a real point.
    grid_size: A list of 3 integers describing the grid size along the x, y, and
      z dimensions. This corresponds to the number of voxels in each dimension.
    grid_range_x: A 2-tuple of floats containing the range (in real world
      coordinates) for the x dimension to be voxelized.
    grid_range_y: A 2-tuple of floats containing the range (in real world
      coordinates) for the y dimension to be voxelized.
    grid_range_z: A 2-tuple of floats containing the range (in real world
      coordinates) for the z dimension to be voxelized.

  Returns:
    A NestedMap with keys:
      coords: An int tf.Tensor of shape [batch_size, num_points, 3] with
        voxel coordinates in x, y, z for each point. Note that voxel_coords may
        be have out of bounds values for points falling outside the range. This
        can be determined by the mask below.
      centers: A float tf.Tensor of shape [batch_size, num_points, 3] with
        real world coordinates of the voxel center for each point.
      indices: An int tf.Tensor of shape [batch_size, num_points] with raveled
        indices where we have flattened x, y, z into one dimension. This can
        be used in conjunction with batched unsorted segment functions to
        perform computations. Note that indices are set so that they are always
        valid: indices that are out of range are set to 0. This makes it easy
        for us to use indices with segment and gather functions.
      padding: A float tf.Tensor of shape [batch_size, num_points], where 1.0
        indicates that the corresponding voxel_coords is padded (out of bounds).
      num_voxels: The total number of voxels.
  """
  batch_size, num_points, _ = py_utils.GetShape(points_xyz)

  # Compute the size of each voxel cell.
  num_voxels = np.prod(grid_size)
  grid_size_x, grid_size_y, grid_size_z = grid_size
  grid_cell_sizes = [
      float(grid_range_x[1] - grid_range_x[0]) / grid_size_x,
      float(grid_range_y[1] - grid_range_y[0]) / grid_size_y,
      float(grid_range_z[1] - grid_range_z[0]) / grid_size_z,
  ]

  # Reposition points_xyz so that 0 aligns with the start of each range.
  grid_offset = tf.cast([grid_range_x[0], grid_range_y[0], grid_range_z[0]],
                        dtype=tf.float32)
  points_xyz -= grid_offset

  # Compute the voxel coords for points according to point position.
  voxel_coords = tf.cast(points_xyz // grid_cell_sizes, dtype=tf.int32)

  # Compute a valid mask based on the voxel coords. Any coordinate out of range
  # results in the point considered to be a padded point.
  voxel_padding = tf.equal(points_padding, 1.0) | tf.reduce_any(
      (voxel_coords >= grid_size) | (voxel_coords < [0, 0, 0]), axis=-1)

  # Ravel only on the coordinates, excluding the batch dimension.
  voxel_indices = RavelIndex(
      tf.reshape(voxel_coords, [batch_size * num_points, 3]), grid_size)
  voxel_indices = tf.reshape(voxel_indices, [batch_size, num_points])

  # Set invalid voxels to have 0 for their index. Note that when using the
  # batched unsorted segment functions, you should pass the padding tensor
  # so that it does the appropriate handling of out of bounds points. By setting
  # the indices here to 0, we make it easy to use batch_gather to map the voxel
  # values back to each point; points that are out of bounds will get the value
  # of the voxel at the 0 index.
  voxel_indices = WhereBroadcast(voxel_padding, tf.zeros_like(voxel_indices),
                                 voxel_indices)

  voxel_padding = tf.cast(voxel_padding, dtype=tf.float32)

  # Compute the voxel centers real-world coordinate for each point. We add 0.5
  # to ensure that we are computing at the center.
  voxel_centers = ((0.5 + tf.cast(voxel_coords, dtype=tf.float32)) *
                   tf.cast(grid_cell_sizes, dtype=tf.float32) + grid_offset)

  dynamic_voxels = py_utils.NestedMap(
      coords=voxel_coords,
      centers=voxel_centers,
      indices=voxel_indices,
      padding=voxel_padding,
      num_voxels=num_voxels)

  return dynamic_voxels


def RavelIndex(coords, dims):
  """Converts a coordinate arrays into an array of flat indices.

  This function does the opposite conversion to tf.unravel_index.

  Args:
    coords: An int tf.Tensor of shape [N, D] with coordinates in each dimension
      for each row.
    dims: A Tensor. Must have the same type as indices. An 1-D int Tensor. The
      shape of the array to use for unraveling indices.

  Returns:
    An int tf.Tensor of shape [N] with voxel indices.
  """
  _, num_dims = py_utils.GetShape(coords)
  dims = py_utils.HasShape(dims, [num_dims])
  multiplier = tf.math.cumprod(dims, exclusive=True, reverse=True)
  indices = tf.reduce_sum(coords * multiplier, axis=1)
  return indices


def _BatchedUnsortedSegmentFn(batched_data,
                              batched_segment_ids,
                              num_segments,
                              unsorted_segment_fn,
                              batched_padding=None,
                              name=None):
  """Calls an unsorted segment function on a batch of data.

  This assumes that batched_data and batched_segment_ids have a leading batch
  dimension that match. The num_segments must be the same for all examples
  in the batch.

  Each example usually has segment_ids that are the same as other examples, but
  we want them to be processed separately. For example, when performing
  dynamic voxelization, we want to map each point to a voxel, where this mapping
  has voxels that correspond to segment_ids. While the ids are the same across
  the voxel coordinate system, we want to keep each example separate.

  Args:
    batched_data: a tensor with a batch dimension matching that of
      batched_segment_ids.
    batched_segment_ids: an integer tensor whose shape is a prefix of
      batched_data.shape. Each element contains the segment id for the
      corresponding slice in batched_data. Entries with negative ids are
      ignored.
    num_segments: an integer tensor. This corresponds to the maximum number of
      possible segments, which may be greater than the actual number of
      segments. This ensures that the shapes are known after this operation.
    unsorted_segment_fn: an unsorted segment function, e.g.,
      tf.math.unsorted_segment_max, unsorted_segment_mean, unsorted_segment_min,
      etc.
    batched_padding: an optional float tensor whose shape is a prefix of
      batched_segment_ids.shape. A value of 1.0 means that the corresponding
      slice of batched_data is padded, and 0.0 means it is real. Padded slices
      will be ignored by setting their segment_id to -1, which causes the
      unsorted segment functions to ignore them.
    name: A name for the operation.

  Returns:
    a tensor containing the result of running unsorted_segment_fn on each
    example separately.
  """
  batch_size = py_utils.GetShape(batched_data)[0]
  batched_segment_shape = py_utils.GetShape(batched_segment_ids)

  # Unsorted segment functions drop elements for which the id is negative.
  ignore_element = batched_segment_ids < 0
  # Padded elements should have segment_ids set to -1, so that they are ignored.
  if batched_padding is not None:
    padding_shape = py_utils.GetShape(batched_padding)
    rank_diff = len(padding_shape) - len(py_utils.GetShape(ignore_element))
    broadcast_shape = padding_shape + [1] * rank_diff
    batched_padding = tf.reshape(batched_padding, broadcast_shape)
    ignore_element |= tf.equal(batched_padding, 1.0)

  # Convert segment_id -> batch_idx * num_segments + segment_id so that each
  # batch is placed in a different range of segment ids.
  segment_id_start = tf.range(0, batch_size, dtype=batched_segment_ids.dtype)
  segment_id_start *= num_segments

  # Broadcast and add.
  segment_id_start = tf.reshape(segment_id_start,
                                [-1] + [1] * (len(batched_segment_shape) - 1))
  batched_segment_ids += segment_id_start
  # Set negative or padded indices to -1.
  batched_segment_ids = tf.where_v2(ignore_element,
                                    tf.constant(-1, batched_segment_ids.dtype),
                                    batched_segment_ids)

  batched_segment_output = unsorted_segment_fn(batched_data,
                                               batched_segment_ids,
                                               batch_size * num_segments, name)

  output_shape = py_utils.GetShape(batched_segment_output)

  # Reshape to recover batch dimension.
  batched_segment_output = tf.reshape(
      batched_segment_output, [batch_size, num_segments] + output_shape[1:])

  return batched_segment_output


# Public methods for batched unsorted segment functions.
# pylint: disable=invalid-name
BatchedUnsortedSegmentMax = functools.partial(
    _BatchedUnsortedSegmentFn, unsorted_segment_fn=tf.math.unsorted_segment_max)
BatchedUnsortedSegmentMin = functools.partial(
    _BatchedUnsortedSegmentFn, unsorted_segment_fn=tf.math.unsorted_segment_min)
BatchedUnsortedSegmentMean = functools.partial(
    _BatchedUnsortedSegmentFn,
    unsorted_segment_fn=tf.math.unsorted_segment_mean)
BatchedUnsortedSegmentSum = functools.partial(
    _BatchedUnsortedSegmentFn, unsorted_segment_fn=tf.math.unsorted_segment_sum)

# pylint: enable=invalid-name


################################################################################
# Anchor Free Model Helpers.
################################################################################
def LocalTransform(points, bboxes_3d):
  """Transform a point to the local coordinate of an bbox.

  This function can transform a point from global coordinate system to the
  local coordinate system of its assigned_gt_bbox. Here are two steps to
  implement this:
    - Transform the point to the center of the corresponding bbox.
    - Rotate the point by the -assigned_gt_phi(which is the orientation of its
      corresponding bbox in global coordinate).

  Args:
    points: A float tensor with shape [..., 3] indicating the XYZ coordinates of
      each input point.
    bboxes_3d: A float tensor with shape [..., 7] indicating the corresponding
      bounding boxes for each point.

  Returns:
    local_points: A float tensor with shape [..., 3] indicating the XYZ
      coordinates of each input point in local coordinate system of its
      corresponding bbox.
  """
  points_shape = py_utils.GetShape(points)

  phi = bboxes_3d[..., -1]
  # Rotate the whole point cloud so as to let the x-axis of point cloud
  # is aligned with the local x-axis of the bbox.
  rotation_matrix = geometry.BatchMakeRotationMatrix(phi)
  rotation_matrix = tf.reshape(rotation_matrix, points_shape[:-1] + [3, 3])

  local_points = points - bboxes_3d[..., :3]
  local_points = tf.matmul(
      tf.expand_dims(local_points, axis=-2), rotation_matrix)
  local_points = local_points[..., 0, :]
  return local_points


def GenerateCenternessLabel(points,
                            assigned_gt_bboxes,
                            centerness_range,
                            ignore_z=False,
                            epsilon=1e-6):
  """Compute the centerness label for each point.

  This loss is first proposed in FCOS (https://arxiv.org/abs/1904.01355).
  Intuitively, it can generate a heatmap, in which pixels closed to center
  obtains higher scores while pixels far from the center obtain lower scores.
  This can be used as an alternative heatmap generation method.

  Args:
    points: A float tensor with shape [..., 3] indicating the XYZ coordinates of
      each input point.
    assigned_gt_bboxes: A float tensor with shape [..., 7] indicating the
      assigned groundtruth bounding boxes for each point, so as to calculate the
      corresponding center-ness groundtruth.
    centerness_range: [0, 1] indicating the valid range of centerness label.
    ignore_z: a boolean. Whether consider z in centerness calculation process.
      For example, in PointPillars model, since there is no z-axis, the
      center-ness value among z-axis can be ignored. By default, it is set False
      to be utilized in Pillars model.
    epsilon: The minimum value for avoiding negative center-ness value. By
      default, it is 1e-6.

  Returns:
    centerness: [...] indicating the center-ness score for each
      point with the value between centerness_range.
  """
  points_shape = py_utils.GetShape(points)
  points = py_utils.with_dependencies(
      [py_utils.assert_equal(points_shape[-1], 3)], points)

  assigned_gt_bboxes = py_utils.HasShape(assigned_gt_bboxes,
                                         points_shape[:-1] + [7])

  # We transform each point to the local coordinates of the groundtruth
  # boudingbox that it is assigned to.
  local_points = LocalTransform(points, assigned_gt_bboxes)

  lx, ly, lz = tf.unstack(local_points, axis=-1)
  dx, dy, dz = tf.unstack(assigned_gt_bboxes[..., 3:-1], axis=-1)

  distance_front = 0.5 * dx - lx
  distance_back = lx + 0.5 * dx
  centerness_dx = (
      tf.minimum(distance_front, distance_back) /
      tf.maximum(distance_front, distance_back))
  valid_dx = tf.logical_and(
      tf.greater_equal(distance_front, 0.), tf.greater_equal(distance_back, 0.))

  distance_left = 0.5 * dy - ly
  distance_right = ly + 0.5 * dy
  centerness_dy = (
      tf.minimum(distance_left, distance_right) /
      tf.maximum(distance_left, distance_right))
  valid_dy = tf.logical_and(
      tf.greater_equal(distance_left, 0.), tf.greater_equal(distance_right, 0.))

  centerness = centerness_dx * centerness_dy
  valid = tf.logical_and(valid_dx, valid_dy)

  if not ignore_z:
    distance_top = 0.5 * dz - lz
    distance_bottom = lz + 0.5 * dz
    centerness_dz = (
        tf.minimum(distance_bottom, distance_top) /
        tf.maximum(distance_bottom, distance_top))
    valid_dz = tf.logical_and(
        tf.greater_equal(distance_top, 0.),
        tf.greater_equal(distance_bottom, 0.))
    valid = tf.logical_and(valid, valid_dz)

    centerness *= centerness_dz

  centerness = tf.maximum(centerness, epsilon)
  if ignore_z:
    centerness = tf.pow(centerness, 1 / 2.)
  else:
    centerness = tf.pow(centerness, 1 / 3.)

  min_threshold, max_threshold = centerness_range
  centerness_interval = max_threshold - min_threshold
  centerness *= centerness_interval
  centerness += min_threshold

  centerness = tf.where_v2(valid, centerness, 0.)
  return centerness


def ComputeFeatureRatio(images, image_features):
  """Compute Ratio between feature height and original RGB image height.

  Args:
    images: A float tensor with shape [batch_size, image height, image width, 3]
      indicating the RGB images.
    image_features: A float tensor with shape [batch_size, num_cameras, H, W, C]
      containing the features extracted from backbone network.

  Returns:
    feat_ratio: A float for indicating the ratio between feature map height
      (or width) and original image height (or width).
  """
  _, feat_height = py_utils.GetShape(image_features, 2)
  _, height = py_utils.GetShape(images, 2)
  feat_ratio = tf.cast(feat_height, tf.float32) / tf.cast(height, tf.float32)
  return feat_ratio


def StackCameraImages(images, camera_names=None):
  """Stack all camera images as a tensor.

  Args:
    images: A dict saves images for different cameras.
    camera_names: A string list that illustrates the name of all cameras.

  Returns:
    all_cameras: A float tensor with shape [batch_size * num_cameras,
      height, width, 3].
  """
  camera_names = camera_names or sorted(images.keys())
  # Stack and compute CNN features for all cameras. Note that we assume that
  # all cameras have the same resolution and thus can be stacked together.
  all_cameras = []
  num_cameras = len(camera_names)
  all_cameras = [images[camera_name].image for camera_name in camera_names]
  all_cameras = tf.stack(all_cameras, axis=1)
  all_cameras = py_utils.HasShape(all_cameras, [-1, -1, -1, -1, 3])
  batch_size, num_cameras, height, width = py_utils.GetShape(all_cameras, 4)
  all_cameras = tf.reshape(all_cameras,
                           [batch_size * num_cameras, height, width, 3])
  return all_cameras
