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
"""Tests for car_lib."""

import lingvo.compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import car_lib
import numpy as np


class CarLibTest(test_utils.TestCase):

  def np_sq_dis(self, pa, pb):
    sq_dis = np.zeros((pa.shape[0], pa.shape[1], pb.shape[1]))
    for n in range(pa.shape[0]):
      for i in range(pa.shape[1]):
        for j in range(pb.shape[1]):
          sq_dis[n, i, j] = np.sum(np.square(pa[n, i] - pb[n, j]))
    return sq_dis

  def _np_sq_dis_neighbors(self, points, neighbor_idx):
    """Run the np square distance function and gather based on neighbor."""
    n, p1, k = neighbor_idx.shape
    sq_dist_all = self.np_sq_dis(points, points)
    sq_dist_neighbors = np.zeros((n, p1, k), dtype=np.float32)
    for b_i in range(n):
      for p_i in range(p1):
        for k_i in range(k):
          sq_dist_neighbors[b_i, p_i, k_i] = sq_dist_all[
              b_i, p_i, neighbor_idx[b_i, p_i, k_i]]
    return sq_dist_neighbors

  def testSquaredDistance(self):
    g = tf.Graph()
    with g.as_default():
      pa_tensor = tf.placeholder(tf.float32, shape=[None, None, 4])
      pb_tensor = tf.placeholder(tf.float32, shape=[None, None, 4])
      sq_dis_tensor = car_lib.SquaredDistanceMatrix(
          pa_tensor, pb_tensor, mem_optimized=False)
      sq_dis_tensor_mem_optimized = car_lib.SquaredDistanceMatrix(
          pa_tensor, pb_tensor, mem_optimized=True)

    num_trials = 3
    for _ in range(num_trials):
      rand_pa = np.random.randn(10, 5, 4)
      rand_pb = np.random.randn(10, 6, 4)
      np_sq_dis = self.np_sq_dis(rand_pa, rand_pb)

      with self.session(graph=g) as sess:
        sq_dis, sq_dis_mem_optimized = sess.run(
            [sq_dis_tensor, sq_dis_tensor_mem_optimized],
            feed_dict={
                pa_tensor: rand_pa,
                pb_tensor: rand_pb,
            })
        self.assertAllClose(sq_dis, np_sq_dis)
        self.assertAllClose(sq_dis_mem_optimized, np_sq_dis)

  def testNeighborSquaredDistance(self):
    n, p1, k = 2, 10, 3
    points = tf.random.uniform((n, p1, 3))
    neighbor_idx = tf.random.uniform((n, p1, k),
                                     minval=0,
                                     maxval=p1,
                                     dtype=tf.int32)
    neighbor_points = car_lib.MatmulGather(points, neighbor_idx)

    sq_dist_result = car_lib.NeighborSquaredDistanceMatrix(
        points, neighbor_points)

    with self.session():
      [np_points, np_neighbor_idx, np_sq_dist_result
      ] = self.evaluate([points, neighbor_idx, sq_dist_result])
      np_sq_dist_expected = self._np_sq_dis_neighbors(np_points,
                                                      np_neighbor_idx)
      self.assertAllClose(np_sq_dist_result, np_sq_dist_expected)

  def testKnnIndicesMaxDistance(self):
    points = tf.constant([[[1, 1, 1], [2, 2, 2], [4, 4, 4], [5, 5, 5]]],
                         dtype=tf.float32)
    query_points = tf.constant([[
        [1, 1, 1],
        [4, 4, 4],
    ]], dtype=tf.float32)
    valid_num = tf.constant([4], dtype=tf.int32)

    # Max distance so that one neighbor can be selected.
    max_distance = 2.0
    expected_1nn = np.array([[[0], [2]]], dtype=np.int32)
    expected_1nn_padding = np.array([[[0], [0]]], dtype=np.float32)
    expected_2nn = np.array([[[0, 1], [2, 3]]], dtype=np.int32)
    expected_2nn_padding = np.array([[[0, 0], [0, 0]]], dtype=np.float32)
    expected_3nn = np.array([[[0, 1, 0], [2, 3, 2]]], dtype=np.int32)
    expected_3nn_padding = np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32)
    with self.session():
      output_1nn, output_1nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 1, valid_num, max_distance))
      self.assertAllEqual(output_1nn, expected_1nn)
      self.assertAllEqual(output_1nn_padding, expected_1nn_padding)

      output_2nn, output_2nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 2, valid_num, max_distance))
      self.assertAllEqual(output_2nn, expected_2nn)
      self.assertAllEqual(output_2nn_padding, expected_2nn_padding)

      output_3nn, output_3nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 3, valid_num, max_distance))
      self.assertAllEqual(output_3nn, expected_3nn)
      self.assertAllEqual(output_3nn_padding, expected_3nn_padding)

    # Max distance so that only itself can be selected.
    max_distance = 0.1
    expected_1nn = np.array([[[0], [2]]], dtype=np.int32)
    expected_1nn_padding = np.array([[[0], [0]]], dtype=np.float32)
    expected_2nn = np.array([[[0, 0], [2, 2]]], dtype=np.int32)
    expected_2nn_padding = np.array([[[0, 1], [0, 1]]], dtype=np.float32)
    expected_3nn = np.array([[[0, 0, 0], [2, 2, 2]]], dtype=np.int32)
    expected_3nn_padding = np.array([[[0, 1, 1], [0, 1, 1]]], dtype=np.float32)
    with self.session():
      output_1nn, output_1nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 1, valid_num, max_distance))
      self.assertAllEqual(output_1nn, expected_1nn)
      self.assertAllEqual(output_1nn_padding, expected_1nn_padding)

      output_2nn, output_2nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 2, valid_num, max_distance))
      self.assertAllEqual(output_2nn, expected_2nn)
      self.assertAllEqual(output_2nn_padding, expected_2nn_padding)

      output_3nn, output_3nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 3, valid_num, max_distance))
      self.assertAllEqual(output_3nn, expected_3nn)
      self.assertAllEqual(output_3nn_padding, expected_3nn_padding)

  def testKnnIndices(self):
    points = tf.constant([[[1, 1], [2, 2], [4, 4], [5, 5]]], dtype=tf.float32)
    query_points = tf.constant([[
        [1, 1],
        [4, 4],
    ]], dtype=tf.float32)

    # Case 1: when all points are valid.
    valid_num = tf.constant([4], dtype=tf.int32)
    expected_1nn = np.array([[[0], [2]]], dtype=np.int32)
    expected_2nn = np.array([[[0, 1], [2, 3]]], dtype=np.int32)
    expected_3nn = np.array([[[0, 1, 2], [2, 3, 1]]], dtype=np.int32)
    with self.session():
      output_1nn, _ = self.evaluate(
          car_lib.KnnIndices(points, query_points, 1, valid_num))
      self.assertAllEqual(output_1nn, expected_1nn)

      output_2nn, _ = self.evaluate(
          car_lib.KnnIndices(points, query_points, 2, valid_num))
      self.assertAllEqual(output_2nn, expected_2nn)

      output_3nn, _ = self.evaluate(
          car_lib.KnnIndices(points, query_points, 3, valid_num))
      self.assertAllEqual(output_3nn, expected_3nn)

    # Case 2: not all points are valid.
    valid_num = tf.constant([2], dtype=tf.int32)
    expected_1nn = np.array([[[0], [1]]], dtype=np.int32)
    expected_1nn_padding = np.array([[[0], [0]]], dtype=np.float32)
    expected_2nn = np.array([[[0, 1], [1, 0]]], dtype=np.int32)
    expected_2nn_padding = np.array([[[0, 0], [0, 0]]], dtype=np.float32)
    expected_3nn = np.array([[[0, 1, 2], [1, 0, 2]]], dtype=np.int32)
    expected_3nn_padding = np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32)

    with self.session():
      output_1nn, output_1nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 1, valid_num))
      self.assertAllEqual(output_1nn, expected_1nn)
      self.assertAllEqual(output_1nn_padding, expected_1nn_padding)

      output_2nn, output_2nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 2, valid_num))
      self.assertAllEqual(output_2nn, expected_2nn)
      self.assertAllEqual(output_2nn_padding, expected_2nn_padding)

      output_3nn, output_3nn_padding = self.evaluate(
          car_lib.KnnIndices(points, query_points, 3, valid_num))
      self.assertAllEqual(output_3nn, expected_3nn)
      self.assertAllEqual(output_3nn_padding, expected_3nn_padding)

    # Case 3: explicit padding.
    padding = tf.constant([[1, 0, 1, 0]])
    expected_1nn = np.array([[[1], [3]]], dtype=np.int32)
    expected_1nn_padding = np.array([[[0], [0]]], dtype=np.float32)
    expected_2nn = np.array([[[1, 3], [3, 1]]], dtype=np.int32)
    expected_2nn_padding = np.array([[[0, 0], [0, 0]]], dtype=np.float32)
    expected_3nn = np.array([[[1, 3, 0], [3, 1, 2]]], dtype=np.int32)
    expected_3nn_padding = np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32)
    with self.session():
      output_1nn, output_1nn_padding = self.evaluate(
          car_lib.NeighborhoodIndices(points, query_points, 1, padding))
      self.assertAllEqual(output_1nn, expected_1nn)
      self.assertAllEqual(output_1nn_padding, expected_1nn_padding)

      output_2nn, output_2nn_padding = self.evaluate(
          car_lib.NeighborhoodIndices(points, query_points, 2, padding))
      self.assertAllEqual(output_2nn, expected_2nn)
      self.assertAllEqual(output_2nn_padding, expected_2nn_padding)

      output_3nn, output_3nn_padding = self.evaluate(
          car_lib.NeighborhoodIndices(points, query_points, 3, padding))
      self.assertAllEqual(output_3nn, expected_3nn)
      self.assertAllEqual(output_3nn_padding, expected_3nn_padding)

  def testNeighborhoodIndicesWithUniformSampling(self):
    points = tf.constant([[[1, 1], [2, 2], [4, 4], [5, 5]]], dtype=tf.float32)
    query_points = tf.constant([[
        [2, 2],
        [5, 5],
    ]], dtype=tf.float32)

    padding = tf.constant([[1, 0, 1, 0]])

    # With max_distance=1.1, only the nearest point will be returned (and
    # repeated).
    expected_3nn = np.array([[[1, 1, 1], [3, 3, 3]]], dtype=np.int32)
    expected_paddings = np.array([[[0, 1, 1], [0, 1, 1]]], dtype=np.float32)
    with self.session():
      output_3nn, paddings = self.evaluate(
          car_lib.NeighborhoodIndices(
              points,
              query_points,
              3,
              padding,
              max_distance=1.1,
              sample_neighbors_uniformly=True))
      self.assertAllEqual(output_3nn, expected_3nn)
      self.assertAllEqual(paddings, expected_paddings)

  def testNeighborhoodIndicesWithUniformSamplingRaisesIfNoMaxDistance(self):
    points = tf.constant([[[1, 1], [2, 2], [4, 4], [5, 5]]], dtype=tf.float32)
    query_points = tf.constant([[
        [1, 1],
        [4, 4],
    ]], dtype=tf.float32)
    padding = tf.constant([[1, 0, 1, 0]])
    with self.assertRaisesRegex(
        ValueError, r'.*Uniform sampling requires specifying max_distance.*'):
      car_lib.NeighborhoodIndices(
          points, query_points, 1, padding, sample_neighbors_uniformly=True)

  def testFarthestPointSamplerOnePoint(self):
    points = tf.constant([
        [[1, 1, 1, 1]],
        [[2, 2, 2, 2]],
    ], dtype=tf.float32)
    padding = tf.zeros((2, 1), dtype=tf.float32)
    selected_idx, _ = car_lib.FarthestPointSampler(points, padding, 1)
    with self.session():
      selected_idx = self.evaluate(selected_idx)
      self.assertAllEqual(selected_idx, [[0], [0]])

  def testFarthestPointSamplerInsufficientPoints(self):
    points = tf.constant([
        [[0, 1, 1]],
        [[2, 2, 2]],
    ], dtype=tf.float32)
    padding = tf.zeros((2, 1), dtype=tf.float32)
    with self.session():
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  r'.*Condition x >= y did not hold.*'):
        sampled_idx, closest_idx = car_lib.FarthestPointSampler(
            points, padding, 2)
        self.evaluate((sampled_idx, closest_idx))

  def testFarthestPointSamplerSelectMinMax(self):
    points = tf.constant([
        [[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
        [[0, 2, 3], [1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3], [5, 2, 3]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
    ], dtype=tf.float32)  # pyformat: disable
    padding = tf.zeros((4, 6), dtype=tf.float32)
    selected_idx, closest_idx = car_lib.FarthestPointSampler(points, padding, 2)
    with self.session():
      selected_idx, closest_idx = self.evaluate([selected_idx, closest_idx])
      for batch_idx in range(4):
        self.assertIn(
            selected_idx[batch_idx, 1], [0, 5],
            msg=('The second selected point must be one of the end '
                 'points corresponding to index 0 or 5.'))

      # Closest indices should either be 0 or 1 since we picked 2 points.
      self.assertTrue(
          np.all((closest_idx >= 0) & (closest_idx < 2)),
          msg='Closest index must be among selected indices.')

  def testFarthestPointSamplerSeeded(self):
    points = tf.constant([
        [[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
        [[0, 2, 3], [1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3], [5, 2, 3]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
    ], dtype=tf.float32)  # pyformat: disable
    padding = tf.zeros((4, 6), dtype=tf.float32)
    selected_idx, closest_idx = car_lib.FarthestPointSampler(
        points, padding, 3, num_seeded_points=2)
    with self.session():
      selected_idx, closest_idx = self.evaluate([selected_idx, closest_idx])
      # First two selected points are seeded.
      self.assertTrue(np.all(selected_idx[:, 0] == 0))
      self.assertTrue(np.all(selected_idx[:, 1] == 1))
      # Third point is the last point since it's farthest.
      self.assertTrue(np.all(selected_idx[:, 2] == 5))
      # Closest indices should either be 0, 1 or 2 since we picked 3 points.
      self.assertTrue(
          np.all((closest_idx >= 0) & (closest_idx < 3)),
          msg='Closest index must be among selected indices.')

  def testFarthestPointSamplerAllPoints(self):
    points = tf.constant([
        [[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
        [[0, 2, 3], [1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3], [5, 2, 3]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
    ], dtype=tf.float32)  # pyformat: disable
    padding = tf.zeros((4, 6), dtype=tf.float32)
    sampled_idx, closest_idx = car_lib.FarthestPointSampler(points, padding, 6)
    with self.session():
      sampled_idx, closest_idx = self.evaluate([sampled_idx, closest_idx])
      for batch_n in range(4):
        self.assertSetEqual(
            set(sampled_idx[batch_n, :]),
            set(np.arange(6)),
            msg='All points should be selected.')
        for point_idx in range(6):
          # For each selected point in sampled_idx, we verify that the
          # closest_idx assigned to that point matches itself. This is done by
          # finding the location of the point in sampled_idx (which is shuffled
          # during sampling). The value of the point in closest_idx should match
          # the index assigned to the point in sampled_idx.
          expected_closest_idx = None
          # location refers to the index where the point appears in sampled_idx.
          # This should be what closest_idx refers to.
          for location, sample_idx in enumerate(sampled_idx[batch_n, :]):
            if point_idx == sample_idx:
              expected_closest_idx = location
              break

          self.assertIsNotNone(expected_closest_idx,
                               'Point not found in sampled_idx result.')
          self.assertEqual(
              closest_idx[batch_n][point_idx],
              expected_closest_idx,
              msg='Closest index should be the point itself.')

  def testFarthestPointSamplerGatherPoints(self):
    points = tf.constant([
        [[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
        [[0, 2, 3], [1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3], [5, 2, 3]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
    ], dtype=tf.float32)  # pyformat: disable
    padding = tf.zeros((4, 6), dtype=tf.float32)
    n = 4
    num_points = 3
    selected_idx, _ = car_lib.FarthestPointSampler(points, padding, num_points)
    gather_indices = tf.stack([
        tf.tile(tf.expand_dims(tf.range(n), 1), [1, num_points]), selected_idx
    ],
                              axis=2)
    sampled_points = tf.gather_nd(points, gather_indices)
    with self.session():
      sampled_points = self.evaluate(sampled_points)
      self.assertEqual(sampled_points.shape, (n, num_points, 3))

  def testFarthestPointSamplerPadding(self):
    points = tf.constant([
        [[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
        [[0, 2, 3], [1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3], [5, 2, 3]],
        [[0, 2, 1], [1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 2, 1], [5, 2, 1]],
    ],
                         dtype=tf.float32)

    padding = tf.constant([[0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0],
                           [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1]],
                          dtype=tf.float32)

    np_expected_selected_idx = np.array(
        [[0, 1, 2, 3], [0, 1, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4]],
        dtype=np.int32)

    num_points = 4
    selected_idx, _ = car_lib.FarthestPointSampler(points, padding, num_points)

    with self.session():
      np_selected_idx = self.evaluate(selected_idx)
      np_selected_idx.sort(axis=1)
      self.assertAllEqual(np_selected_idx, np_expected_selected_idx)

  def _testPooling3D(self, pooling_fn):
    num_points_in = 100
    num_points_out = 10
    batch_size = 8
    num_features = 32
    points = tf.random.uniform(
        shape=(batch_size, num_points_in, 3),
        minval=-1,
        maxval=1,
        dtype=tf.float32)

    # Note: This max pooling impl is incorrect if the feature range is negative.
    features = tf.random.uniform(
        shape=(batch_size, num_points_in, num_features),
        minval=0,
        maxval=1,
        dtype=tf.float32)
    padding = tf.zeros((batch_size, num_points_in), dtype=tf.float32)

    pooling_idx, closest_idx = car_lib.FarthestPointSampler(
        points, padding, num_points_out)

    pooled_points, pooled_features = pooling_fn(
        points=points,
        point_features=features,
        pooling_idx=pooling_idx,
        closest_idx=closest_idx)

    with self.session():
      [
          np_points, np_features, np_pooling_idx, np_closest_idx,
          np_pooled_points, np_pooled_features
      ] = self.evaluate([
          points, features, pooling_idx, closest_idx, pooled_points,
          pooled_features
      ])

      for batch_n in range(batch_size):
        # Grab the selected pooling points from our sampler to compare to
        # the output of our pooling.
        expected_pooled_pts = np_points[batch_n, np_pooling_idx[batch_n, :], :]
        self.assertAllClose(expected_pooled_pts, np_pooled_points[batch_n, ...])

        np_batch_features = np_features[batch_n, ...]
        for idx in range(num_points_out):
          in_group = np_closest_idx[batch_n, :] == idx
          expected_max = np.max(np_batch_features[in_group], axis=0)
          self.assertAllClose(expected_max, np_pooled_features[batch_n, idx])

  def testMaxPool3D(self):
    return self._testPooling3D(car_lib.MaxPool3D)

  def testSegmentPool3D(self):
    return self._testPooling3D(car_lib.SegmentPool3D)


if __name__ == '__main__':
  tf.test.main()
