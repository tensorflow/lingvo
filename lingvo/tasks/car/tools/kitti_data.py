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
"""Library for parsing KITTI raw data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
import numpy as np


def LoadVeloBinFile(filepath):
  """Reads and parse raw KITTI velodyne binary file.

  Args:
    filepath: Path to a raw KITTI velodyne binary file.

  Returns:
    A dictionary with keys xyz and reflectance containing numpy arrays.
  """
  with tf.io.gfile.GFile(filepath, 'rb') as f:
    scan = np.frombuffer(f.read(), dtype=np.float32).reshape((-1, 4))
  xyz = scan[:, :3]
  reflectance = scan[:, 3:]
  return {
      'xyz': xyz,
      'reflectance': reflectance,
  }


def LoadLabelFile(filepath):
  """Reads and parse raw KITTI label file.

  The ordering of the arrays for bbox, dimensions, and location follows the
  order in the table below. We refer to the length (dx), width (dy), height (dz)
  for clarity.

  Each line in the label contains (per KITTI documentation):

  +--------+------------+------------------------------------------------------+
  | Values |    Name    |  Description                                         |
  +========+============+======================================================+
  |   1    |   type     | Describes the type of object: 'Car', 'Van', 'Truck', |
  |        |            | 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',   |
  |        |            | 'Misc' or 'DontCare'                                 |
  +--------+------------+------------------------------------------------------+
  |   1    | truncated  | Float from 0 (non-truncated) to 1 (truncated), where |
  |        |            | truncated refers to the object leaving image         |
  |        |            | boundaries.                                          |
  +--------+------------+------------------------------------------------------+
  |   1    |  occluded  | Integer (0,1,2,3) indicating occlusion state:        |
  |        |            | 0 = fully visible, 1 = partly occluded               |
  |        |            | 2 = largely occluded, 3 = unknown                    |
  +--------+------------+------------------------------------------------------+
  |   1    |   alpha    | Observation angle of object, ranging [-pi..pi]       |
  +--------+------------+------------------------------------------------------+
  |   4    |   bbox     | 2D bounding box of object in the image (0-based      |
  |        |            | index): left, top, right, bottom pixel coordinates.  |
  +--------+------------+------------------------------------------------------+
  |   3    | dimensions | 3D object dimensions: height, width, length (meters) |
  +--------+------------+------------------------------------------------------+
  |   3    |  location  | 3D object location x,y,z in camera coordinates       |
  |        |            | (in meters)                                          |
  +--------+------------+------------------------------------------------------+
  |   1    | rotation_y | Rotation ry around Y-axis in camera coordinates      |
  |        |            | [-pi..pi]                                            |
  +--------+------------+------------------------------------------------------+
  |   1    |   score    | Only for results: Float, indicating confidence in    |
  |        |            | detection, needed for p/r curves, higher is better.  |
  +--------+------------+------------------------------------------------------+

  Args:
    filepath: Path to a raw KITTI label file.

  Returns:
    A list of dictionary with keys corresponding to the name column above. type,
    truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score.
    Note that the order of the floats in bbox, dimensions, and location
    correspond to that in the doc-string above.
  """
  objects = []
  with tf.io.gfile.GFile(filepath, 'r') as f:
    for line in f:
      line = line.strip()
      if not line:  # Skip empty lines
        continue
      line_splits = line.split(' ')
      if len(line_splits) not in [15, 16]:
        raise ValueError(
            'Found {} tokens in Line: "{}". Expects only 15/16 token'.format(
                len(line_splits), line))

      # If score does not exist, we append a -1 to indicate so.
      if len(line_splits) == 15:
        line_splits.append(-1.)

      (obj_type, truncated, occluded, alpha,
       bbox_left, bbox_top, bbox_right, bbox_bottom,
       height, width, length,
       cam_x, cam_y, cam_z,
       rotation_y, score) = line_splits  # pyformat: disable

      obj = {
          'type': obj_type,
          'truncated': float(truncated),
          'occluded': int(occluded),
          'alpha': float(alpha),
          'bbox': [
              float(x) for x in [bbox_left, bbox_top, bbox_right, bbox_bottom]
          ],
          'dimensions': [float(x) for x in [height, width, length]],
          'location': [float(x) for x in [cam_x, cam_y, cam_z]],
          'rotation_y': float(rotation_y),
          'score': float(score),
      }
      _ValidateLabeledObject(obj)
      objects.append(obj)

  return objects


def _ValidateLabeledObject(obj):
  """Validate that obj has expected values."""

  if obj['type'] not in [
      'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
      'Misc', 'DontCare'
  ]:
    raise ValueError('Invalid type: %s' % obj['type'])

  if not ((obj['truncated'] == -1.0) or
          (obj['truncated'] >= 0.0 and obj['truncated'] <= 1.0)):
    raise ValueError('Invalid truncated value: %s' % obj['truncated'])

  if obj['occluded'] not in [-1, 0, 1, 2, 3]:
    raise ValueError('Invalid occluded value: %s' % obj['occluded'])

  if not (obj['alpha'] == -10. or
          (obj['alpha'] >= -np.pi and obj['alpha'] <= np.pi)):
    raise ValueError('Invalid alpha value: %s' % obj['alpha'])

  if not (obj['rotation_y'] == -10. or
          (obj['rotation_y'] >= -np.pi and obj['rotation_y'] <= np.pi)):
    raise ValueError('Invalid rotation_y value: %s' % obj['rotation_y'])

  return obj


def ParseCalibrationDict(raw_calib):
  """Parse transformation matrices in a raw KITTI calibration dictionary.

  Per the KITTI documentation:

  All matrices are stored row-major, i.e., the first values correspond
  to the first row. R0_rect contains a 3x3 matrix which you need to extend to
  a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
  Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
  in the same way.

  IMPORTANT: The coordinates in the camera coordinate system can be projected in
  the image by using the 3x4 projection matrix in the calib folder, where for
  the left color camera for which the images are provided, P2 must be used.

  Args:
    raw_calib: A dictionary of raw KITTI calibration values with keys P0, P1,
      P2, P3, R0_rect, Tr_imu_to_velo, and Tr_velo_to_cam containing flattened
      matrices of appropriate size.

  Returns:
    A dictionary with keys P0, P1, P2, P3, R0_rect, Tr_imu_to_velo,
    and Tr_velo_to_cam containing reshaped and extended matrices.
  """
  # The projection matrices are 3x4 matrices.
  calib = {}
  calib['P0'] = raw_calib['P0'].reshape([3, 4])
  calib['P1'] = raw_calib['P1'].reshape([3, 4])
  calib['P2'] = raw_calib['P2'].reshape([3, 4])
  calib['P3'] = raw_calib['P3'].reshape([3, 4])

  # R0_rect contains a 3x3 matrix which you need to extend to a 4x4 matrix by
  # adding a 1 as the bottom-right element and 0's elsewhere.
  extended_r0_rect = np.eye(4)
  extended_r0_rect[:3, :3] = raw_calib['R0_rect'].reshape([3, 3])
  calib['R0_rect'] = extended_r0_rect

  # Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
  # in the same way!
  extended_tr_imu_to_velo = np.eye(4)
  extended_tr_imu_to_velo[:3, :4] = raw_calib['Tr_imu_to_velo'].reshape([3, 4])
  calib['Tr_imu_to_velo'] = extended_tr_imu_to_velo

  extended_tr_velo_to_cam = np.eye(4)
  extended_tr_velo_to_cam[:3, :4] = raw_calib['Tr_velo_to_cam'].reshape([3, 4])
  calib['Tr_velo_to_cam'] = extended_tr_velo_to_cam

  return calib


def LoadCalibrationFile(filepath):
  """Read and parse a raw KITTI calibration file.

  Args:
    filepath: Path to a raw KITTI calibration file.

  Returns:
    A dictionary with keys P0, P1, P2, P3, R0_rect, Tr_imu_to_velo,
    and Tr_velo_to_cam containing reshaped and extended transformation
    matrices.
  """
  raw_calib = {}
  with tf.io.gfile.GFile(filepath, 'r') as f:
    for line in f:
      line = line.strip()
      if not line:  # Skip empty lines
        continue
      key, value = line.split(':', 1)
      raw_calib[key] = np.array([float(x) for x in value.split()])
  return ParseCalibrationDict(raw_calib)


def VeloToImagePlaneTransformation(calib):
  """Compute the transformation matrix from velo xyz to image plane xy.

  Per the KITTI documentation, to project a point from Velodyne coordinates into
  the left color image, you can use this formula:

     x = P2 * R0_rect * Tr_velo_to_cam * y

  After applying the transformation, you will need to divide the by the last
  coordinate to recover the 2D pixel locations.

  Args:
    calib: A calibration dictionary returned by LoadCalibrationFile.

  Returns:
    A numpy 3x4 transformation matrix.
  """
  return np.dot(calib['P2'], np.dot(calib['R0_rect'], calib['Tr_velo_to_cam']))


def VeloToCameraTransformation(calib):
  """Compute the transformation matrix from velo xyz to camera xyz.

  Per the KITTI documentation, to project a point from Velodyne coordinates into
  the left color image, you can use this formula:

     x = P2 * R0_rect * Tr_velo_to_cam * y

  NOTE: The above formula further projects the xyz point to the image plane
  using P2, which we do not apply in this function since we are working with
  xyz (3D coordinates).

  Args:
    calib: A calibration dictionary returned by LoadCalibrationFile.

  Returns:
    A numpy 4x4 transformation matrix.
  """
  return np.dot(calib['R0_rect'], calib['Tr_velo_to_cam'])


def CameraToVeloTransformation(calib):
  """Compute the transformation matrix from camera to velo.

  This is the inverse transformation of CameraToVeloTransformation.

  Args:
    calib: A calibartion dictionary returned by LoadCalibrationFile.

  Returns:
    A numpy 4x4 transformation matrix.
  """
  return np.linalg.pinv(VeloToCameraTransformation(calib))


def AnnotateKITTIObjectsWithBBox3D(objects, calib):
  """Add our canonical bboxes 3d format to KITTI objects.

  The annotated bboxes 3d are in the velodyne coordinate frame.

  Args:
    objects: A list of KITTI objects returned by LoadLabelFile.
    calib: A calibartion dictionary returned by LoadCalibrationFile.

  Returns:
    The original list of KITTI objects, where each object has new keys
    'has_3d_info' indicating if the object has valid 3D bounding box data, and
    'bbox3d' which corresponds to our canonical bboxes 3d format.
  """

  # All objects will share the same transformation matrix, which we compute
  # once here.
  transformation_matrix = CameraToVeloTransformation(calib)
  for obj in objects:
    obj['bbox3d'] = _KITTIObjectToBBox3D(obj, transformation_matrix)
    obj['has_3d_info'] = _KITTIObjectHas3DInfo(obj)
  return objects


def _KITTIObjectHas3DInfo(obj):
  """Check whether KITTI object has valid 3D bounding box information."""
  height, width, length = obj['dimensions']
  # KITTI raw data has -1 for all 3 dimensions when no 3D box info is present.
  return not (width == -1 or length == -1 or height == -1)


def _KITTIObjectToBBox3D(obj, cam_to_velo_transform):
  """Convert one object given the transformation matrix."""
  height, width, length = obj['dimensions']

  # Avoid transforming objects with invalid boxes.
  if not _KITTIObjectHas3DInfo(obj):
    return [-1000, -1000, -1000, -1, -1, -1, -10]

  velo_xyz = np.dot(cam_to_velo_transform, np.asarray(obj['location'] + [1.]))
  x, y, z = velo_xyz.tolist()[:3]

  # In the raw data, the z coordinate is at the bottom of the object, we need
  # to reposition z so that it's at the center of the object.
  z += height / 2.

  # Our velodyne bbox rotation goes the other direction and is rotated by
  # -np.pi/2. See http://www.cvlibs.net/datasets/kitti/setup.php.
  rot = -obj['rotation_y']
  rot -= np.pi / 2.
  bbox3d = [x, y, z] + [length, width, height] + [rot]

  return bbox3d


def BBox3DToKITTIObject(bbox3d, velo_to_cam_transform):
  """Convert one bbox3d into KITTI's location, dimension, and rotation_y."""
  x, y, z, length, width, height, rot = bbox3d

  # Avoid transforming objects with invalid boxes. See _KITTIObjectHas3DInfo.
  if width == -1 or length == -1 or height == -1:
    return [-1000, -1000, -1000], [-1, -1, -1], -10

  # Convert our velodyne bbox rotation back to camera. Reverse the direction and
  # rotate by np.pi/2. See http://www.cvlibs.net/datasets/kitti/setup.php.
  rotation_y = rot + np.pi / 2.
  rotation_y = -rotation_y
  rotation_y = np.mod(rotation_y, 2 * np.pi)
  rotation_y = np.where(rotation_y >= np.pi, rotation_y - 2 * np.pi, rotation_y)

  # Reposition z so that it is at the bottom of the object.
  if height > 0:
    z -= height / 2.

  camera_xyz = np.dot(velo_to_cam_transform, np.asarray([x, y, z, 1.]))
  location = camera_xyz.tolist()[:3]
  dimensions = height, width, length

  return location, dimensions, rotation_y
