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
"""Library of functions on tensors for car layers, builders, and models."""


# pylint:enable=g-direct-tensorflow-import
import lingvo.compat as tf
from lingvo.core import py_utils


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
