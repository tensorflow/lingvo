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
"""Tests for kitti_data."""

from lingvo import compat as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.car import geometry
from lingvo.tasks.car.tools import kitti_data
import numpy as np


class KittiDataTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    self._label_file = test_helper.test_src_dir_path(
        'tasks/car/testdata/kitti_raw_label_testdata.txt')
    self._calib_file = test_helper.test_src_dir_path(
        'tasks/car/testdata/kitti_raw_calib_testdata.txt')

  def testLoadLabelFile(self):
    objects = kitti_data.LoadLabelFile(self._label_file)
    self.assertEqual(len(objects), 7)
    self.assertEqual(objects[1]['type'], 'Car')
    self.assertEqual(objects[1]['truncated'], 0.0)
    self.assertEqual(objects[1]['occluded'], 0)
    self.assertEqual(objects[1]['alpha'], 1.85)
    self.assertAllClose(objects[1]['bbox'], [387.63, 181.54, 423.81, 203.12])
    self.assertAllClose(objects[1]['dimensions'], [1.67, 1.87, 3.69])
    self.assertAllClose(objects[1]['location'], [-16.53, 2.39, 58.49])
    self.assertEqual(objects[1]['rotation_y'], 1.57)
    self.assertEqual(objects[1]['score'], -1.)

  def testLoadCalibrationFile(self):
    calib = kitti_data.LoadCalibrationFile(self._calib_file)

    self.assertAllEqual(calib['P0'].shape, [3, 4])
    self.assertAllEqual(calib['P1'].shape, [3, 4])
    self.assertAllEqual(calib['P2'].shape, [3, 4])
    self.assertAllEqual(calib['P3'].shape, [3, 4])

    self.assertAllEqual(calib['R0_rect'].shape, [4, 4])
    self.assertAllEqual(calib['R0_rect'][3, :], [0., 0., 0., 1.])
    self.assertAllEqual(calib['R0_rect'][:, 3], [0., 0., 0., 1.])

    self.assertAllEqual(calib['Tr_imu_to_velo'].shape, [4, 4])
    self.assertAllEqual(calib['Tr_imu_to_velo'][3, :], [0., 0., 0., 1.])
    self.assertAllEqual(calib['Tr_velo_to_cam'].shape, [4, 4])
    self.assertAllEqual(calib['Tr_velo_to_cam'][3, :], [0., 0., 0., 1.])

  def testVeloToCamAndCamToVeloAreInverses(self):
    calib = kitti_data.LoadCalibrationFile(self._calib_file)
    velo_to_cam = kitti_data.VeloToCameraTransformation(calib)
    cam_to_velo = kitti_data.CameraToVeloTransformation(calib)
    self.assertAllClose(cam_to_velo.dot(velo_to_cam), np.eye(4))

  def testAnnotateKITTIObjectsWithBBox3D(self):
    objects = kitti_data.LoadLabelFile(self._label_file)
    calib = kitti_data.LoadCalibrationFile(self._calib_file)
    objects = kitti_data.AnnotateKITTIObjectsWithBBox3D(objects, calib)
    for obj in objects:
      self.assertEqual(len(obj['bbox3d']), 7)

    # atol=0.01 corresponds to a 1cm tolerance.
    self.assertAllClose(
        objects[0]['bbox3d'][:3], [69.72, -0.45, 0.58], atol=0.01)
    self.assertAllClose(objects[0]['bbox3d'][3:],
                        [12.34, 2.63, 2.85, -0.01079633])
    self.assertAllEqual(objects[0]['has_3d_info'], True)

    # no 3D data
    self.assertAllEqual(objects[3]['has_3d_info'], False)

  def testKITTIObjToBBoxAndInverse(self):
    objects = kitti_data.LoadLabelFile(self._label_file)
    calib = kitti_data.LoadCalibrationFile(self._calib_file)
    for obj in objects:
      bbox3d = kitti_data._KITTIObjectToBBox3D(
          obj, kitti_data.CameraToVeloTransformation(calib))
      location, dimensions, rotation_y = kitti_data.BBox3DToKITTIObject(
          bbox3d, kitti_data.VeloToCameraTransformation(calib))
      self.assertAllClose(obj['location'], location)
      self.assertAllClose(obj['dimensions'], dimensions)
      self.assertAllClose(obj['rotation_y'], rotation_y)

  def testVeloToImagePlaneTransformation(self):
    objects = kitti_data.LoadLabelFile(self._label_file)
    calib = kitti_data.LoadCalibrationFile(self._calib_file)

    # Only apply to object 0.
    obj = objects[0]
    bbox3d = kitti_data._KITTIObjectToBBox3D(
        obj, kitti_data.CameraToVeloTransformation(calib))

    # Convert to corners in our canonical space.
    corners = geometry.BBoxCorners(tf.constant([[bbox3d]], dtype=tf.float32))
    with self.session():
      corners_np = self.evaluate(corners)
    corners_np = corners_np.reshape([8, 3])

    # Add homogenous coordinates.
    corners_np = np.concatenate([corners_np, np.ones((8, 1))], axis=-1)

    # Apply the velo to image plane transformation.
    velo_to_img = kitti_data.VeloToImagePlaneTransformation(calib)
    corners_np = np.dot(corners_np, velo_to_img.T)

    # Divide by the last coordinate to recover pixel locations.
    corners_np[:, 0] /= corners_np[:, 2]
    corners_np[:, 1] /= corners_np[:, 2]

    # Obtain 2D bbox.
    min_x = np.min(corners_np[:, 0])
    max_x = np.max(corners_np[:, 0])
    min_y = np.min(corners_np[:, 1])
    max_y = np.max(corners_np[:, 1])
    bbox = [min_x, min_y, max_x, max_y]  # left, top, right, bottom.

    # This should correspond to the GT bbox in obj['bbox'].
    # We use atol=0.1 here since they should close to the nearest pixel.
    self.assertAllClose(bbox, obj['bbox'], atol=0.1)


if __name__ == '__main__':
  tf.test.main()
