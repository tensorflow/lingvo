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
"""Routines related to geometric operations on bboxes.

Bboxes and coordinates are always in the last dimension of a tensor.

2D BBoxes are represented by (ymin, xmin, ymax, xmax).

2D BBoxes can also be represented by (centroid x, centroid y, width, height).

2D coordinates are represented by (x, y).

3D coordinates are represented by (x, y, z).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import py_utils
import numpy as np
from tensorflow.python.ops import functional_ops


def _BroadcastMatmul(x, y):
  """Broadcast y and matmul with x.

  Args:
    x: A tensor of shape [..., b].
    y: A matrix of shape [b, c].

  Returns:
    Tensor: ``z[..., c]``, where ``z[i..., :] = matmul(x[i..., :], y)``
  """
  y = py_utils.HasRank(y, 2)
  x_reshaped = tf.reshape(x, [-1, tf.shape(x)[-1]])
  result = tf.matmul(x_reshaped, y)
  return tf.reshape(result,
                    tf.concat(
                        [tf.shape(x)[:-1], tf.shape(y)[-1:]], axis=0))


def _MakeRotationMatrix(yaw, roll, pitch):
  """Create a 3x3 rotation matrix from yaw, roll, pitch (angles in radians).

  Note: Yaw -> Z, Roll -> X, Pitch -> Y.

  Args:
    yaw: float tensor representing a yaw angle in radians.
    roll: float tensor representing a roll angle in radians.
    pitch: float tensor representing a pitch angle in radians.

  Returns:
    A [3, 3] tensor corresponding to a rotation matrix.
  """

  # pyformat: disable
  def _UnitX(angle):
    return tf.reshape([1., 0., 0.,
                       0., tf.cos(angle), -tf.sin(angle),
                       0., tf.sin(angle), tf.cos(angle)],
                      shape=[3, 3])

  def _UnitY(angle):
    return tf.reshape([tf.cos(angle), 0., tf.sin(angle),
                       0., 1., 0.,
                       -tf.sin(angle), 0., tf.cos(angle)],
                      shape=[3, 3])

  def _UnitZ(angle):
    return tf.reshape([tf.cos(angle), -tf.sin(angle), 0.,
                       tf.sin(angle), tf.cos(angle), 0.,
                       0., 0., 1.],
                      shape=[3, 3])
  # pyformat: enable
  return tf.matmul(tf.matmul(_UnitZ(yaw), _UnitX(roll)), _UnitY(pitch))


def CoordinateTransform(points, pose):
  """Translate 'points' to coordinates according to 'pose' vector.

  pose should contain 6 floating point values:
    translate_x, translate_y, translate_z: The translation to apply.
    yaw, roll, pitch: The rotation angles in radians.

  Args:
    points: Float shape [..., 3]: Points to transform to new coordinates.
    pose: Float shape [6]: [translate_x, translate_y, translate_z, yaw, roll,
      pitch]. The pose in the frame that 'points' comes from, and the defintion
      of the rotation and translation angles to apply to points.

  Returns:
    'points' transformed to the coordinates defined by 'pose'.
  """
  translate_x = pose[0]
  translate_y = pose[1]
  translate_z = pose[2]

  # Translate the points so the origin is the pose's center.
  translation = tf.reshape([translate_x, translate_y, translate_z], shape=[3])
  translated_points = points + translation

  # Compose the rotations along the three axes.
  #
  # Note: Yaw->Z, Roll->X, Pitch->Y.
  yaw, roll, pitch = pose[3], pose[4], pose[5]
  rotation_matrix = _MakeRotationMatrix(yaw, roll, pitch)

  # Finally, rotate the points about the pose's origin according to the
  # rotation matrix.
  rotated_points = _BroadcastMatmul(translated_points, rotation_matrix)
  return rotated_points


def TransformPoints(points, transforms):
  """Apply 4x4 transforms to a set of points.

  Args:
    points: A [..., num_points, 3] tensor of xyz point locations.
    transforms: A [..., 4, 4] tensor with the same leading shape as points.

  Returns:
    A tensor with the same shape as points, transformed respectively.
  """
  # Create homogeneous coordinates for points.
  points = tf.concat([points, tf.ones_like(points[..., :1])], axis=-1)

  # Apply transformations, and divide by last axis to project back to 3D-space.
  # Transpose the transforms since the transformation is usually expected to
  # be applied such that new_points = T * current_point.
  points = tf.matmul(points, transforms, transpose_b=True)
  points = points[..., :3] / points[..., 3:]

  return points


def WrapAngleRad(angles_rad, min_val=-np.pi, max_val=np.pi):
  """Wrap the value of `angles_rad` to the range [min_val, max_val]."""
  max_min_diff = max_val - min_val
  return min_val + tf.math.floormod(angles_rad + max_val, max_min_diff)


def TransformBBoxes3D(bboxes_3d, transforms):
  """Apply 4x4 transforms to 7 DOF bboxes (change center and rotation).

  Args:
    bboxes_3d: A [..., num_boxes, 7] tensor representing 3D bboxes.
    transforms: A [..., 4, 4] tensor with the same leading shape as bboxes_3d.
      These transforms are expected to only affect translation and rotation,
      while not scaling the data. This ensures that the bboxes have the same
      dimensions post transformation.

  Returns:
    A tensor with the same shape as bboxes_3d, with transforms applied to each
    bbox3d.
  """
  center_xyz = bboxes_3d[..., :3]
  dimensions = bboxes_3d[..., 3:6]
  rot = bboxes_3d[..., 6:]

  # Transform center and rotation, assuming that dimensions are not changed.
  center_xyz = TransformPoints(center_xyz, transforms)
  rot += tf.atan2(transforms[..., 1:2, 0:1], transforms[..., 0:1, 0:1])
  rot = WrapAngleRad(rot)

  return tf.concat([
      center_xyz,
      dimensions,
      rot,
  ], axis=-1)  # pyformat: disable


def XYWHToBBoxes(xywh):
  """Converts xywh to bboxes."""
  mtrx = tf.constant(
      np.array([
          # x    y    h    w
          [0.0, 1.0, 0.0, -.5],  # ymin
          [1.0, 0.0, -.5, 0.0],  # xmin
          [0.0, 1.0, 0.0, 0.5],  # ymax
          [1.0, 0.0, 0.5, 0.0],  # xmax
      ]).T,
      dtype=xywh.dtype)
  return _BroadcastMatmul(xywh, mtrx)


def PointsToImagePlane(points, velo_to_image_plane):
  """Converts 3D points to the image plane.

  Args:
    points: A [N, 3] Floating point tensor containing xyz points. Points are
      assumed to be in velo coordinates.
    velo_to_image_plane: A [3, 4] matrix from velo xyz to image plane xy. After
      multiplication, you need to divide by last coordinate to recover 2D pixel
      locations.

  Returns:
    A [N, 2] Floating point tensor containing points in the image plane.
  """
  points = py_utils.HasRank(points, 2)
  num_points = tf.shape(points)[0]
  points = py_utils.HasShape(points, [num_points, 3])
  velo_to_image_plane = py_utils.HasShape(velo_to_image_plane, [3, 4])

  # Add homogenous coordinates to points.
  points = tf.concat([points, tf.ones((num_points, 1))], axis=-1)

  # Perform projection and divide by last coordinate to recover 2D pixel
  # locations.
  points_image = tf.matmul(points, velo_to_image_plane, transpose_b=True)
  points_image = points_image[:, :2] / points_image[:, 2:3]

  points_image = py_utils.HasShape(points_image, [num_points, 2])
  return points_image


def BBoxesToXYWH(bboxes):
  """Converts bboxes to xywh."""
  mtrx = tf.constant(
      np.array([
          # ymin xmin ymax xmax
          [0.0, 0.5, 0.0, 0.5],  # x centroid
          [0.5, 0.0, 0.5, 0.0],  # y centroid
          [0.0, -1., 0.0, 1.0],  # width
          [-1., 0.0, 1.0, 0.0],  # height
      ]).T,
      dtype=bboxes.dtype)
  return _BroadcastMatmul(bboxes, mtrx)


def BBoxesCentroid(bboxes):
  """Returns the centroids of bboxes."""
  mtrx = tf.constant(
      np.array([
          # ymin xmin ymax xmax
          [0.0, 0.5, 0.0, 0.5],  # x centroid
          [0.5, 0.0, 0.5, 0.0],  # y centroid
      ]).T,
      dtype=bboxes.dtype)
  return _BroadcastMatmul(bboxes, mtrx)


def ReorderIndicesByPhi(anchor, bboxes):
  """Sort bboxes based their angles relative to the anchor point.

  Args:
    anchor: A vector of (x0, y0).
    bboxes: A matrix of shape [N, 4].

  Returns:
    A permutation of tf.range(n) which can be used to reshuffle bboxes to the
    sorted order. (e.g., tf.gather(bboxes, indices)).
  """

  @tf.Defun(anchor.dtype, bboxes.dtype)
  def _True(anchor, bboxes):
    """True branch when num of bboxes is non-zero."""
    n = tf.shape(bboxes)[0]
    centroid = BBoxesCentroid(bboxes)

    # Computed dot products between centroid and the anchor point.
    dot = tf.squeeze(tf.matmul(centroid, tf.expand_dims(anchor, 1)), axis=1)

    # Normalize dot to get the cosine of the angles.
    norm = tf.norm(anchor) * tf.norm(centroid, axis=1)
    cosine = tf.where(
        tf.greater(norm, 0), dot / norm, tf.zeros([n], norm.dtype))

    # Disambiguates the angle anchor--O--point is positive or negative by the
    # sign of cross products between angle and points.  tf.linalg.cross takes
    # 3-vector (x, y, z), so we set z to 0.  tf.linalg.cross does not support
    # broadcasting, so we tile anchor to shape [n, 3].
    cross = tf.linalg.cross(
        tf.tile(tf.pad(tf.expand_dims(anchor, 0), [[0, 0], [0, 1]]), [n, 1]),
        tf.pad(centroid, [[0, 0], [0, 1]]))

    # If the sign is positive, the points lie on the clockwise side of
    # O-->anchor. Hence, -1 - cosine moves the cosine values to [-2, 0].  If the
    # sign is negative, the points lie on the counter-clockwise side of
    # O-->anchor. 1 + cosine moves the cosine values to [0, 2].
    #
    # The car dataset shows that the points are scanned in the counter-clockwise
    # fashion. Therefore, top-k orders the points in the same order in which
    # bboxes appears in the spin.
    score = tf.where(tf.greater(cross, 0)[:, 2], -1 - cosine, 1 + cosine)

    _, indices = tf.nn.top_k(score, n, sorted=True)
    return indices

  @tf.Defun(anchor.dtype, bboxes.dtype)
  def _False(anchor, bboxes):
    del anchor, bboxes
    return tf.zeros([0], dtype=tf.int32)

  n = tf.shape(bboxes)[0]
  return functional_ops.If(tf.greater(n, 0), [anchor, bboxes], _True, _False)[0]


def _SmoothL1Norm(a):
  """Smoothed L1 norm."""
  # F&F paper formula (3).
  # http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
  return tf.where(tf.abs(a) < 1, 0.5 * tf.square(a), tf.abs(a) - 0.5)


def DistanceBetweenCentroidsAndBBoxesFastAndFurious(centroids, bboxes, masks):
  """Computes the distance between centroids and bboxes.

  The distance/loss is loosely following the 'Fast and Furious' paper by Luo et
  al., CVPR'18.  This is just one way of calculating the distances. We will
  probably develop other ways.

  Args:
    centroids: [..., 4]. x/y/w/h for bboxes.
    bboxes: [..., 4]. ymin/xmin/ymax/xmax for bboxes.
    masks: [...]. masks[i] == 1 means i-th entry (centroids[i] and bboxes[i])
      should be considered in the distance/loss calculation.

  Returns:
    A [...] tensor. i-th value is the distance measure of centroids[i] and
    bboxes[i].
  """
  x, y, w, h = tf.unstack(centroids, axis=-1, num=4)
  # "gt" suffix means 'ground truth'.
  x_gt, y_gt, w_gt, h_gt = tf.unstack(BBoxesToXYWH(bboxes), axis=-1, num=4)

  def Pos(x):
    return tf.maximum(tf.constant(1e-8, x.dtype), x)

  # The following terms are zeros when masks[i] is 0.
  l_x = py_utils.CheckNumerics(masks * (x - x_gt) / Pos(w_gt))
  l_y = py_utils.CheckNumerics(masks * (y - y_gt) / Pos(h_gt))
  s_w = py_utils.CheckNumerics(masks * tf.math.log(Pos(w) / Pos(w_gt)))
  s_h = py_utils.CheckNumerics(masks * tf.math.log(Pos(h) / Pos(h_gt)))
  return (_SmoothL1Norm(l_x) + _SmoothL1Norm(l_y) + _SmoothL1Norm(s_w) +
          _SmoothL1Norm(s_h))


def DistanceBetweenCentroids(u, v, masks):
  """Computes the distance between centroids.

  Args:
    u: [..., 4]. x/y/w/h for bboxes.
    v: [..., 4]. x/y/w/h for bboxes.
    masks: [...]. masks[i] == 1 means i-th entry (u[i] and v[i]) should be
      considered in the distance/loss calculation.

  Returns:
    A [...] tensor. i-th value is the distance measure of u[i] and v[i].
  """
  return masks * tf.reduce_sum(_SmoothL1Norm(u - v), axis=-1)


# TODO(zhifengc/drpng): Consider other possible loss formuation:
# E.g.,
#  (L1(u[x], v[x]) + L1(u[w], v[w]))*(L1(u[y], v[y]) + L1(u[h], v[h]))


def _IsOnLeftHandSideOrOn(point, v1, v2):
  """Checks if a point lays on a vector direction, or is to the left.

  Args:
    point: a tensor of shape [..., 2] of points to check.
    v1: a float tensor of shape [..., 2] of vertices.
    v2: a tensor of shape and type as v1. The second vertices.

  Returns:
    A tensor of booleans indicating whether each point is on the left
    of, or exactly on, the direction indicated by the vertices.
  """
  v1 = py_utils.HasShape(v1, tf.shape(v2))
  # Prepare for broadcast: All point operations are on the right,
  # and all v1/v2 operations are on the left. This is faster than left/right
  # under the assumption that we have more points than vertices.
  point_x = point[..., tf.newaxis, :, 0]
  point_y = point[..., tf.newaxis, :, 1]
  v1_x = v1[..., 0, tf.newaxis]
  v2_x = v2[..., 0, tf.newaxis]
  v1_y = v1[..., 1, tf.newaxis]
  v2_y = v2[..., 1, tf.newaxis]
  d1 = (point_y - v1_y) * (v2_x - v1_x)
  d2 = (point_x - v1_x) * (v2_y - v1_y)
  return d1 >= d2


def _IsCounterClockwiseDirection(v1, v2, v3):
  """Checks if the path from v1 to v3 via v2 is counter-clockwise.

  When v1 is equal to v2, or v2 equals v3, return true, by fiat. Tis will
  work when the v's are padded vectors.

  Args:
    v1: a float Tensor of shape [..., 2], indicating the starting point.
    v2: a Tensor of same type and shape as v1, indicating the via point.
    v3: a Tensor of same type and shape as v1, indicating the ending point.

  Returns:
    True for all directions such that v1 to v3 via v2 is a counter clockwise
  direction.
  """
  # Check if it's on the left hand side, strictly, and without broadcasting.
  v1 = py_utils.HasShape(v1, tf.shape(v2))
  v1 = py_utils.HasShape(v1, tf.shape(v3))
  v1_x, v1_y = v1[..., 0], v1[..., 1]
  v2_x, v2_y = v2[..., 0], v2[..., 1]
  v3_x, v3_y = v3[..., 0], v3[..., 1]
  d1 = (v3_y - v1_y) * (v2_x - v1_x)
  d2 = (v3_x - v1_x) * (v2_y - v1_y)
  return d1 >= d2


def IsWithinBBox(points, bbox):
  """Checks if points are within a 2-d bbox.

  The function returns true if points are strictly inside the box. It also
  returns true when the points are exactly on the box edges.

  Args:
    points: a float Tensor of shape [..., 2] of points to be tested. The last
      coordinates are (x, y).
    bbox: a float Tensor of shape [..., 4, 2] of bboxes. The last coordinates
      are the four corners of the bbox and (x, y). The corners are assumed to be
      given in counter-clockwise order.

  Returns:
    Tensor: If ``pshape = tf.shape(points)[:-1]`` and
    ``bshape = tf.shape(bbox)[:-2]``, returns a boolean tensor of shape
    ``tf.concat(pshape, bshape)``, where each element is true if the point is
    inside to the corresponding box.  If a point falls exactly on an edge of the
    bbox, it is also true.
  """
  bshape = py_utils.GetShape(bbox)[:-2]
  pshape = py_utils.GetShape(points)[:-1]
  bbox = py_utils.HasShape(bbox, tf.concat([bshape, [4, 2]], axis=0))
  points = py_utils.HasShape(points, tf.concat([pshape, [2]], axis=0))
  # Enumerate all 4 edges:
  v1, v2, v3, v4 = (bbox[..., 0, :], bbox[..., 1, :], bbox[..., 2, :],
                    bbox[..., 3, :])
  v1v2v3_check = tf.reduce_all(_IsCounterClockwiseDirection(v1, v2, v3))
  v2v3v4_check = tf.reduce_all(_IsCounterClockwiseDirection(v2, v3, v4))
  v4v1v2_check = tf.reduce_all(_IsCounterClockwiseDirection(v4, v1, v2))
  v3v4v1_check = tf.reduce_all(_IsCounterClockwiseDirection(v3, v4, v1))
  with tf.control_dependencies([
      py_utils.Assert(v1v2v3_check, [v1, v2, v3]),
      py_utils.Assert(v2v3v4_check, [v3, v3, v4]),
      py_utils.Assert(v4v1v2_check, [v4, v1, v2]),
      py_utils.Assert(v3v4v1_check, [v3, v4, v1])
  ]):
    is_inside = tf.math.logical_and(
        tf.math.logical_and(
            _IsOnLeftHandSideOrOn(points, v1, v2),
            _IsOnLeftHandSideOrOn(points, v2, v3)),
        tf.math.logical_and(
            _IsOnLeftHandSideOrOn(points, v3, v4),
            _IsOnLeftHandSideOrOn(points, v4, v1)))
  # Swap the last two dimensions.
  is_inside = tf.einsum('...ij->...ji', tf.cast(is_inside, tf.int32))
  return tf.cast(is_inside, tf.bool)


def BBoxCorners2D(bboxes):
  """Extract the corner points from a 5-DOF bbox representation.

  Args:
    bboxes: A [..., 5] floating point bounding box representation ([x, y, dx,
      dy, phi]).

  Returns:
    A [..., 4, 2] floating point Tensor containing
      the corner (x, y) points for every bounding box.
  """
  corners = tf.constant([
      [0.5, 0.5],
      [-0.5, 0.5],
      [-0.5, -0.5],
      [0.5, -0.5],
  ])

  leading_shape = py_utils.GetShape(bboxes)[:-1]

  # Extract location, dimension, and rotation.
  location = bboxes[..., :2]
  dimensions = bboxes[..., 2:4]
  phi_world = bboxes[..., 4]

  # Convert rotation_phis into rotation matrices along unit z.
  cos = tf.cos(phi_world)
  sin = tf.sin(phi_world)
  rotations_world = tf.reshape(
      tf.stack([cos, -sin, sin, cos], axis=-1), leading_shape + [2, 2])

  # Create axis-aligned corners from length/width/height.
  corners = tf.einsum('...i,ji->...ji', dimensions, corners)

  # Rotate the corners coordinates to the rotated world frame.
  corners = tf.einsum('...ij,...kj->...ki', rotations_world, corners)

  # Translate corners to the world location.
  corners = corners + tf.reshape(location, leading_shape + [1, 2])
  return corners


def BBoxCorners(bboxes):
  """Extract the corner points from a 7-DOF bbox representation.

  Args:
    bboxes: A [batch, num_boxes, 7] floating point bounding box representation
      ([x, y, z, dx, dy, dz, phi]).

  Returns:
    A [batch, num_boxes, 8, 3] floating point Tensor containing
      the corner (x, y, z) points for every bounding box.
  """
  # Code adapted from vale/soapbox codebase.
  #
  # Corners in normalized box frame (unit cube centered at origin).
  #
  # Dimensions is [length, width, height].
  corners = tf.constant([
      [0.5, 0.5, 0.5],  # top
      [-0.5, 0.5, 0.5],  # top
      [-0.5, -0.5, 0.5],  # top
      [0.5, -0.5, 0.5],  # top
      [0.5, 0.5, -0.5],  # bottom
      [-0.5, 0.5, -0.5],  # bottom
      [-0.5, -0.5, -0.5],  # bottom
      [0.5, -0.5, -0.5],  # bottom
  ])

  batch, nb, _ = py_utils.GetShape(bboxes, 3)

  # Extract location, dimension, and rotation.
  location = bboxes[:, :, :3]
  dimensions = bboxes[:, :, 3:6]
  phi_world = bboxes[:, :, 6]

  # Convert rotation_phis into rotation matrices along unit z.
  cos = tf.cos(phi_world)
  sin = tf.sin(phi_world)
  zero = tf.zeros_like(cos)
  one = tf.ones_like(cos)
  rotations_world = tf.reshape(
      tf.stack([cos, -sin, zero, sin, cos, zero, zero, zero, one], axis=2),
      [batch, nb, 3, 3])

  # Create axis-aligned corners from length/width/height.
  corners = tf.einsum('bni,ji->bnji', dimensions, corners)

  # Rotate the corners coordinates to the rotated world frame.
  corners = tf.einsum('bnij,bnkj->bnki', rotations_world, corners)

  # Translate corners to the world location.
  corners = corners + tf.reshape(location, (batch, nb, 1, 3))
  return corners


def IsWithinBBox3D(points_3d, bboxes_3d):
  """Checks if points are within a 3-d bbox.

  Args:
    points_3d: [num_points, 3] float32 Tensor specifying points in 3-d space as
      [x, y, z] coordinates.
    bboxes_3d: [num_bboxes, 7] float32 Tensor specifying a 3-d bboxes specified
      as [x, y, z, dx, dy, dz, phi] where x, y and z is the center of the box.

  Returns:
    boolean Tensor of shape [num_points, num_bboxes] indicating whether the
    points belong within each box.
  """
  points_3d = py_utils.HasRank(points_3d, 2)
  points_3d = py_utils.HasShape(points_3d, [-1, 3])
  num_points, _ = py_utils.GetShape(points_3d, 2)

  bboxes_3d = py_utils.HasRank(bboxes_3d, 2)
  bboxes_3d = py_utils.HasShape(bboxes_3d, [-1, 7])
  num_bboxes, _ = py_utils.GetShape(bboxes_3d, 2)

  # Compute the 3-D corners of the bounding boxes.
  bboxes_3d_b = tf.expand_dims(bboxes_3d, 0)
  bbox_corners = BBoxCorners(bboxes_3d_b)
  bbox_corners = py_utils.HasShape(bbox_corners, [1, -1, 8, 3])
  # First four points are the top of the bounding box.
  # Counter-clockwise arrangement of points specifying 2-d Euclidean box.
  #   (x0, y1) <--- (x1, y1)
  #                    ^
  #                    |
  #                    |
  #   (x0, y0) ---> (x1, y0)
  bboxes_2d_corners = bbox_corners[0, :, 0:4, 0:2]
  bboxes_2d_corners = py_utils.HasShape(bboxes_2d_corners, [-1, 4, 2])
  # Determine if points lie within 2-D (x, y) plane for all bounding boxes.
  points_2d = points_3d[:, :2]
  is_inside_2d = IsWithinBBox(points_2d, bboxes_2d_corners)
  is_inside_2d = py_utils.HasShape(is_inside_2d, [num_points, num_bboxes])

  # Determine if points lie with the z-dimension for all bounding boxes.
  [_, _, z, _, _, dz, _] = tf.split(bboxes_3d, 7, axis=-1)

  def _ComputeLimits(center, width):
    left = center - width / 2.0
    right = center + width / 2.0
    return left, right

  z0, z1 = _ComputeLimits(z, dz)
  z_points = tf.expand_dims(points_3d[:, 2], -1)

  is_inside_z = tf.math.logical_and(
      tf.less_equal(z_points, z1[tf.newaxis, :, 0]),
      tf.greater_equal(z_points, z0[tf.newaxis, :, 0]))
  is_inside_z = py_utils.HasShape(is_inside_z, [num_points, num_bboxes])

  return tf.math.logical_and(is_inside_z, is_inside_2d)


def SphericalCoordinatesTransform(points_xyz):
  """Converts points from xyz coordinates to spherical coordinates.

  https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
  for definitions of the transformations.

  Args:
    points_xyz: A floating point tensor with shape [..., 3], where the inner 3
      dimensions correspond to xyz coordinates.

  Returns:
    A floating point tensor with the same shape [..., 3], where the inner
    dimensions correspond to (dist, theta, phi), where phi corresponds to
    azimuth/yaw (rotation around z), and theta corresponds to pitch/inclination
    (rotation around y).
  """
  dist = tf.sqrt(tf.reduce_sum(tf.square(points_xyz), axis=-1))
  theta = tf.acos(points_xyz[..., 2] / tf.maximum(dist, 1e-7))
  # Note: tf.atan2 takes in (y, x).
  phi = tf.atan2(points_xyz[..., 1], points_xyz[..., 0])
  return tf.stack([dist, theta, phi], axis=-1)
