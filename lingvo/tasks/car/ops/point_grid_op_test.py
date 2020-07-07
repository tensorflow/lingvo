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
"""Tests for lingvo.tasks.car.ops.point_grid_op."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import ops
import numpy as np


class PointGridOpTest(test_utils.TestCase):

  def _testGrid(self, points, num_points_per_cell, grid_size, grid_range):
    points = np.array(points)  # p x (3 + d)
    with self.session(use_gpu=False):
      out_points, grid_centers, num_points = ops.point_to_grid(
          points,
          num_points_per_cell=num_points_per_cell,
          x_intervals=grid_size[0],
          y_intervals=grid_size[1],
          z_intervals=grid_size[2],
          x_range=grid_range[0],
          y_range=grid_range[1],
          z_range=grid_range[2])
      # Test shape
      out_points = py_utils.HasShape(
          out_points,
          list(grid_size) + [num_points_per_cell, -1])
      grid_centers = py_utils.HasShape(grid_centers, list(grid_size) + [3])
      num_points = py_utils.HasShape(num_points, grid_size)
      out_points, grid_centers, num_points = self.evaluate(
          [out_points, grid_centers, num_points])

      # Test points are in the right cell.
      boundaries = []
      for axis in range(3):
        boundaries += [
            np.linspace(
                start=grid_range[axis][0],
                stop=grid_range[axis][1],
                num=grid_size[axis] + 1,
            )
        ]
      boundaries = np.stack(np.meshgrid(*boundaries, indexing='ij'), axis=-1)

      # Cast num_points to the points_mask, indicating whether a point
      # within a pillar is a real point or not.
      points_index = np.arange(num_points_per_cell, dtype=num_points.dtype)
      points_index = np.reshape(points_index, [1, 1, 1, num_points_per_cell])
      real_points_mask = np.less(points_index, num_points[..., np.newaxis])
      real_points_mask = np.tile(
          np.expand_dims(real_points_mask, axis=-1), [1, 1, 1, 1, 3])

      # Check whether real points are within the right boundaries.
      valid_bound = np.logical_and(
          out_points[..., :3] >= boundaries[:-1, :-1, :-1, np.newaxis, ...],
          out_points[..., :3] < boundaries[1:, 1:, 1:, np.newaxis, ...])
      padded_points_mask = 1 - real_points_mask
      # Check if points are either in valid bounds or padded.
      valid = np.logical_or(valid_bound, padded_points_mask)
      self.assertTrue(np.all(valid))

      # Check that points are either zero (padded) or real.
      valid = np.logical_or(out_points[..., :3] == 0, real_points_mask)
      self.assertTrue(np.all(valid))

      counts = np.zeros(grid_size)

      def idx(v, l, u, k):
        if v < l or v >= u:
          return None
        return int((v - l) / (u - l) * k)

      for (x, y, z) in points:
        ix = idx(x, grid_range[0][0], grid_range[0][1], grid_size[0])
        iy = idx(y, grid_range[1][0], grid_range[1][1], grid_size[1])
        iz = idx(z, grid_range[2][0], grid_range[2][1], grid_size[2])
        if all(i is not None for i in (ix, iy, iz)):
          counts[ix, iy, iz] += 1
      counts = np.minimum(counts, num_points_per_cell)
      self.assertAllEqual(counts, num_points)
      print(np.min(counts), np.max(counts))

  def testSimpleGrid(self):
    points = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    self._testGrid(points, 1, [2, 2, 2], [[0, 2.1], [0, 2.1], [0, 2.1]])

  def testRandomPoints(self):
    np.random.seed(7483)
    points = np.random.random((100, 3))
    grid_size = (5, 4, 3)
    grid_range = ((0, 0.3), (-1, 1.4), (0, 1.0))
    num_points_per_cell = 3
    self._testGrid(points, num_points_per_cell, grid_size, grid_range)


class PointGridOpBenchmark(tf.test.Benchmark):

  def benchmarkSimpleGrid(self):
    points = np.random.uniform(0.0, 1.0, (200000, 3))
    num_points_per_cell = 100
    grid_size = [200, 200, 1]
    grid_range = [[0., 1.], [0., 1], [0, 1]]
    sess = tf.Session()
    out_points, _, _ = ops.point_to_grid(
        points,
        num_points_per_cell=num_points_per_cell,
        x_intervals=grid_size[0],
        y_intervals=grid_size[1],
        z_intervals=grid_size[2],
        x_range=grid_range[0],
        y_range=grid_range[1],
        z_range=grid_range[2])

    print(self.run_op_benchmark(sess, out_points, min_iters=1000))


if __name__ == '__main__':
  tf.test.main()
