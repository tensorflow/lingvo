# Lint as: python2, python3
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
"""Tests for geometry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import geometry
import numpy as np


class GeometryTest(test_utils.TestCase):

  def testXYWHBBoxesCentroid(self):
    with self.session():
      xywh = np.tile(
          np.array([[[10, 20, 8, 6], [0, 0, 0, 0], [-10, -20, 2.4, 3.6]]]),
          (3, 2, 1))
      bboxes = np.tile(
          np.array([[[17, 6, 23, 14], [0, 0, 0, 0], [-21.8, -11.2, -18.2,
                                                     -8.8]]]), (3, 2, 1))
      centroid = np.tile(np.array([[[10, 20], [0, 0], [-10, -20]]]), (3, 2, 1))
      print('shapes = ', xywh.shape, bboxes.shape, centroid.shape)
      self.assertAllClose(geometry.XYWHToBBoxes(xywh).eval(), bboxes)
      self.assertAllClose(geometry.BBoxesToXYWH(bboxes).eval(), xywh)
      self.assertAllClose(geometry.BBoxesCentroid(bboxes).eval(), centroid)

  def testPointsToImage(self):
    # From a KITTI example.
    velo_to_image_plane = tf.constant(
        [[6.09695409e+02, -7.21421597e+02, -1.25125855e+00, -1.23041806e+02],
         [1.80384202e+02, 7.64479802e+00, -7.19651474e+02, -1.01016688e+02],
         [9.99945389e-01, 1.24365378e-04, 1.04513030e-02, -2.69386912e-01]])

    points = tf.constant([[1.25120001e+01, -5.09700012e+00, -7.26999998e-01],
                          [1.27309999e+01, -5.21099997e+00, 1.85000002e-01],
                          [-100, -20, -30]])

    with self.session():
      points_image = geometry.PointsToImagePlane(points, velo_to_image_plane)
      result = points_image.eval()
      # First two points map to the image plane, the last one falls outside.
      expected = [[914.15222168, 215.81137085], [914.61260986, 162.28927612],
                  [463.57763672, -32.7820015]]
      self.assertAllClose(expected, result)

  def testReorderIndicesByPhi(self):
    # Picks the anchor point somewhere on the x-y plane.
    phi0 = 2 * np.pi / 3
    anchor = np.array([np.cos(phi0), np.sin(phi0)])

    # Generates points clock-wise relative to the anchor.
    pnts = []
    n, delta = 16, 0.01
    for i in range(n):
      phi = phi0 + 1e-6 + 2 * np.pi / n * i
      x, y = (1 + i) * np.cos(phi), (1 + i) * np.sin(phi)
      pnts += [[y - delta, x - delta, y + delta, x + delta]]
    pnts = np.array(pnts)
    labels = np.arange(n)

    # Randomly permutate the points.
    perm = np.random.permutation(n)
    pnts = pnts[perm]
    labels = labels[perm]

    # Uses ReorderIndicesByPhi to figure out how to reshuffle the points to
    # recover their original order.
    with self.session():
      indices = geometry.ReorderIndicesByPhi(anchor, pnts).eval()

    print('indices = ', indices)
    print('labels = ', labels[indices])
    self.assertAllEqual(labels[indices], np.flip(np.arange(n), 0))

  def testReorderIndicesByPhiEmpty(self):
    with self.session():
      indices = geometry.ReorderIndicesByPhi(
          tf.constant([1., 0.], tf.float32),
          tf.constant(np.zeros([0, 4]), tf.float32)).eval()

    self.assertEqual(indices.shape, (0,))

  def testDistanceBetweenCentroidsAndBBoxesFastAndFurious(self):
    # pyformat: disabled
    predicted = np.array([
        [1., 1., 2., 2.],  # Perfect.
        [2., 2., 2., 2.],  # w/h are perfect.
        [1., 1., 1., 1.],  # centroids are perfect.
        [0., 0., 0., 0.],  # /0 if not padded properly.
    ])
    groundtruth = np.array([
        [0., 0., 2., 2.],
        [0., 0., 2., 2.],
        [0., 0., 2., 2.],
        [0., 0., 0., 0.],
    ])
    masks = np.array([
        1.,
        1.,
        1.,
        0.,
    ])
    # pyformat: enabled
    with self.session():
      distance = geometry.DistanceBetweenCentroidsAndBBoxesFastAndFurious(
          predicted, groundtruth, masks)
      self.assertAllClose(
          distance.eval(),
          [0.0, 1.0 / 4, np.square(np.log(1. / 2)), 0.0])

  def testDistanceBetweenCentroids(self):
    # pyformat: disabled
    u = np.array([
        [1., 1., 2., 2.],  # Perfect.
        [2., 2., 2., 2.],  # w/h are perfect.
        [1., 1., 1., 1.],  # centroids are perfect.
        [0., 1., 2., 3.],  # does not matter.
    ])
    v = np.array([
        [1., 1., 2., 2.],  # Perfect.
        [1., 1., 2., 2.],  # w/h are perfect.
        [1., 1., 2., 2.],
        [1., 2., 3., 4.],
    ])
    masks = np.array([
        1.,
        1.,
        1.,
        0.,
    ])
    # pyformat: enabled
    with self.session():
      distance = geometry.DistanceBetweenCentroids(u, v, masks)
      self.assertAllClose(distance.eval(), [0.0, 1.0, 1.0, 0.0])

  def testCoordinateTransform(self):
    # This is a validated test case from a colab on a real scene.
    #
    # A single point [1, 1, 3].
    point = tf.constant([[[5736.94580078, 1264.85168457, 45.0271225]]],
                        dtype=tf.float32)
    # Replicate the point to test broadcasting behavior.
    replicated_points = tf.tile(point, [2, 4, 1])

    # Pose of the car (x, y, z, yaw, roll, pitch).
    #
    # We negate the translations so that the coordinates are translated
    # such that the car is at the origin.
    pose = tf.constant([
        -5728.77148438, -1264.42236328, -45.06399918, -3.10496902, 0.03288471,
        0.00115049
    ],
                       dtype=tf.float32)

    transformed_points = geometry.CoordinateTransform(replicated_points, pose)
    with self.session():
      result = transformed_points.eval()

      # We expect the point to be translated close to the car, and then rotated
      # mostly around the x-axis.
      expected = np.tile([[[-8.18451203, -0.13086951, -0.04200766]]], [2, 4, 1])

      self.assertAllClose(expected, result)

  def testIsOnLeftHandSideOrOn(self):
    v1 = tf.constant([[0., 0.]], dtype=tf.float32)
    v2 = tf.constant([[1., 0.]], dtype=tf.float32)
    p = tf.constant([[.5, .5], [-1., -3], [-1., 1.]], dtype=tf.float32)
    with self.session() as sess:
      actual = sess.run(geometry._IsOnLeftHandSideOrOn(p, v1, v2))
      self.assertAllEqual([[True], [False], [True]], actual)
      actual = sess.run(geometry._IsOnLeftHandSideOrOn(v1, v1, v2))
      self.assertAllEqual([[True]], actual)
      actual = sess.run(geometry._IsOnLeftHandSideOrOn(v2, v1, v2))
      self.assertAllEqual([[True]], actual)

  def testIsWithinBBox3D(self):
    num_points, num_bboxes = 19, 4
    # rotate the first box by pi / 2 so dim_x and dim_y are swapped.
    # The last box is a cube rotated by 45 degrees.
    bboxes = tf.constant([[1.0, 2.0, 3.0, 6.0, 0.4, 6.0, np.pi / 2],
                          [4.0, 5.0, 6.0, 7.0, 0.8, 7.0, 0.0],
                          [0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.0],
                          [-10., -10., -10., 3., 3., 3., np.pi / 4]],
                         dtype=tf.float32)
    points = tf.constant(
        [
            [1.0, 2.0, 3.0],  # box 0 (centroid)
            [0.8, 2.0, 3.0],  # box 0 (below x)
            [1.1, 2.0, 3.0],  # box 0 (above x)
            [1.3, 2.0, 3.0],  # box 0 (too far x)
            [0.7, 2.0, 3.0],  # box 0 (too far x)
            [4.0, 5.0, 6.0],  # box 1 (centroid)
            [4.0, 4.6, 6.0],  # box 1 (below y)
            [4.0, 5.4, 6.0],  # box 1 (above y)
            [4.0, 4.5, 6.0],  # box 1 (too far y)
            [4.0, 5.5, 6.0],  # box 1 (too far y)
            [0.4, 0.3, 0.2],  # box 2 (centroid)
            [0.4, 0.3, 0.1],  # box 2 (below z)
            [0.4, 0.3, 0.3],  # box 2 (above z)
            [0.4, 0.3, 0.0],  # box 2 (too far z)
            [0.4, 0.3, 0.4],  # box 2 (too far z)
            [5.0, 7.0, 8.0],  # none
            [1.0, 5.0, 3.6],  # box0, box1
            [-11.6, -10., -10.],  # box3 (rotated corner point).
            [-11.4, -11.4, -10.],  # not in box3, would be if not rotated.
        ],
        dtype=tf.float32)
    expected_is_inside = np.array([
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, True, False, False],
        [False, True, False, False],
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, True, False],
        [False, False, True, False],
        [False, False, True, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [True, True, False, False],
        [False, False, False, True],
        [False, False, False, False],
    ])
    assert points.shape[0] == num_points
    assert bboxes.shape[0] == num_bboxes
    assert expected_is_inside.shape[0] == num_points
    assert expected_is_inside.shape[1] == num_bboxes

    with self.session() as sess:
      is_inside = sess.run(geometry.IsWithinBBox3D(points, bboxes))
      self.assertAllEqual([num_points, num_bboxes], is_inside.shape)
      self.assertAllEqual(expected_is_inside, is_inside)

  def testIsWithinBBox(self):
    bbox = tf.constant([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]],
                       dtype=tf.float32)
    points = tf.constant(
        [[-.5, -.5], [.5, -.5], [1.5, -.5], [1.5, .5], [1.5, 1.5], [.5, 1.5],
         [-.5, 1.5], [-.5, .5], [1., 1.], [.5, .5]],
        dtype=tf.float32)
    with self.session() as sess:
      is_inside = sess.run(geometry.IsWithinBBox(points, bbox))
      expected = [[False]] * 8 + [[True]] * 2
      self.assertAllEqual(expected, is_inside)

  def testIsWithinRotatedBBox(self):
    bbox = tf.constant([[[.2, 0.], [1., .2], [.8, 1.], [0., .8]]],
                       dtype=tf.float32)
    points = tf.constant([[0., 0.], [1., 0], [1., 1.], [0., 1.], [.5, .5]],
                         dtype=tf.float32)
    with self.session() as sess:
      is_inside = sess.run(geometry.IsWithinBBox(points, bbox))
      expected = [[False]] * 4 + [[True]]
      self.assertAllEqual(expected, is_inside)

  def testIsCounterClockwiseDirection(self):
    points = tf.constant(
        [[[0., 0.], [0., 0.], [0., 1.]], [[0., 0.], [0., 1.], [1., 1.]],
         [[0., .8], [.2, 0.], [1., .2]], [[.2, .0], [1., .2], [.0, 1.]],
         [[1., 1.], [1., 0], [0., 0.]]],
        dtype=tf.float32)
    # points ~ [num points, v1/v2/v3, x/y].
    points = py_utils.HasShape(points, [-1, 3, 2])
    expected = [True, False, True, True, False]
    with self.session() as sess:
      dircheck = sess.run(
          geometry._IsCounterClockwiseDirection(
              points[:, 0, :], points[:, 1, :], points[:, 2, :]))
      self.assertAllEqual(expected, dircheck)

  def testBBoxCorners(self):
    # Create four bounding boxes, two identical in each batch.
    #
    # This tests both that the batching and number of box dimensions are handled
    # properly.
    bboxes = tf.constant([[[1, 2, 3, 4, 3, 6, 0.], [1, 2, 3, 4, 3, 6, 0.]],
                          [[1, 2, 3, 4, 3, 6, np.pi / 2.],
                           [1, 2, 3, 4, 3, 6, np.pi / 2.]]])
    corners = geometry.BBoxCorners(bboxes)
    with self.session() as sess:
      corners_np = sess.run(corners)
      self.assertEqual((2, 2, 8, 3), corners_np.shape)

      # Extrema of first two boxes are ([-1, 3], [0.5, 3.5], [0, 6])
      for i in [0, 1]:
        self.assertAllClose(-1, np.min(corners_np[0, i, :, 0]))
        self.assertAllClose(3, np.max(corners_np[0, i, :, 0]))
        self.assertAllClose(0.5, np.min(corners_np[0, i, :, 1]))
        self.assertAllClose(3.5, np.max(corners_np[0, i, :, 1]))
        self.assertAllClose(0, np.min(corners_np[0, i, :, 2]))
        self.assertAllClose(6, np.max(corners_np[0, i, :, 2]))

      # Extrema of second two boxes is ([-0.5, 2.5], [0, 4], [0, 6])
      # because it's the first box rotated by 90 degrees.
      for i in [0, 1]:
        self.assertAllClose(-0.5, np.min(corners_np[1, i, :, 0]))
        self.assertAllClose(2.5, np.max(corners_np[1, i, :, 0]))
        self.assertAllClose(0, np.min(corners_np[1, i, :, 1]))
        self.assertAllClose(4, np.max(corners_np[1, i, :, 1]))
        self.assertAllClose(0, np.min(corners_np[1, i, :, 2]))
        self.assertAllClose(6, np.max(corners_np[1, i, :, 2]))

  def testSphericalCoordinatesTransform(self):
    np_xyz = np.random.randn(5, 6, 3)
    points_xyz = tf.constant(np_xyz, dtype=tf.float32)
    spherical_coordinates = geometry.SphericalCoordinatesTransform(points_xyz)

    with self.session() as sess:
      actual_spherical_coordinates = sess.run(spherical_coordinates)

    # Convert coordinates back to xyz to verify.
    dist = actual_spherical_coordinates[..., 0]
    theta = actual_spherical_coordinates[..., 1]
    phi = actual_spherical_coordinates[..., 2]

    x = dist * np.sin(theta) * np.cos(phi)
    y = dist * np.sin(theta) * np.sin(phi)
    z = dist * np.cos(theta)

    self.assertAllClose(x, np_xyz[..., 0])
    self.assertAllClose(y, np_xyz[..., 1])
    self.assertAllClose(z, np_xyz[..., 2])


if __name__ == '__main__':
  tf.test.main()
