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
"""Utilities for performing 3D transformations on points."""

import copy

import numpy as np


class Box2D:
  """A representation of a 2D rotated bounding box.

  Box2D is based on conventions for 3D coordinate systems where y-x are
  switched.
  """

  def __init__(self, x, y, width, length, angle):
    """Initialize a Box2D.

    Args:
      x: center x coordinate of bounding box.
      y: center y coordinate of bounding box.
      width: Width of box in the x-dimension.
      length: Length of box in the y-dimension.
      angle: Angle in radians wrt to the direction of the longer axis.
    """
    self._center = np.array([x, y])
    self._width = width
    self._length = length
    self._angle = angle
    self._angle_v = np.array([np.cos(angle), np.sin(angle)])

    offset = self._angle_v * (length / 2.)
    self._start = self._center - offset
    self._end = self._center + offset
    # Compute the four corners of the rotated box.
    self._corners = self._ComputeCorners()

  @property
  def corners(self):
    """Returns a [4, 2] numpy matrix containing the four corner points."""
    return self._corners

  def _ComputeCorners(self):
    """Compute the four corners of the bounding box."""
    if self._length > 0:
      perp_unit = np.array([-self._angle_v[1], self._angle_v[0]])
    else:
      perp_unit = np.array([0., 0.])
    w2 = perp_unit * (self._width / 2.)

    corner_1 = np.array([self._start[0] + w2[0], self._start[1] + w2[1]])
    corner_2 = np.array([self._end[0] + w2[0], self._end[1] + w2[1]])
    corner_3 = np.array([self._end[0] - w2[0], self._end[1] - w2[1]])
    corner_4 = np.array([self._start[0] - w2[0], self._start[1] - w2[1]])
    return np.array([corner_1, corner_2, corner_3, corner_4])

  def Extrema(self):
    """Returns the extrema of the bounding box."""
    ymin = np.min(self._corners[:, 1])
    xmin = np.min(self._corners[:, 0])
    ymax = np.max(self._corners[:, 1])
    xmax = np.max(self._corners[:, 0])
    return ymin, xmin, ymax, xmax

  def Apply(self, transform):
    """Apply `transform` to the current box and return a new box."""
    # Transform corner points using the transform matrix.
    new_corners = []

    # TODO(vrv): vectorize
    for corner in self._corners:
      # Extend to z, w.
      corner = np.concatenate([corner, np.array([0., 1.])], axis=0)
      # Apply the transform.
      corner_adjusted = np.matmul(transform, corner)
      new_corners.append(corner_adjusted[0:2])

    new_corners = np.stack(new_corners, axis=0)

    # Compute the new center.
    ymin = np.min(new_corners[:, 1])
    xmin = np.min(new_corners[:, 0])
    ymax = np.max(new_corners[:, 1])
    xmax = np.max(new_corners[:, 0])
    center_x = xmin + (xmax - xmin) / 2.
    center_y = ymin + (ymax - ymin) / 2.

    # Compute the new width and length.
    scale_transform = CopyTransform(transform)[0:2, 0:2]
    w_l = np.array([self._width, self._length])
    new_wl = np.abs(np.matmul(scale_transform, w_l))
    new_width = new_wl[0]
    new_length = new_wl[1]

    # Compute the transformed heading.
    transformed_heading = TransformHeading(transform, self._angle)
    return Box2D(center_x, center_y, new_width, new_length, transformed_heading)

  def AsNumpy(self):
    """Return the 5DOF (xywhh) representation as a numpy array."""
    return np.array([
        self._center[0], self._center[1], self._width, self._length, self._angle
    ])


def TransformHeading(transform, heading):
  """Compute 'heading' given transform.

  The heading provided as input is assumed to be in the original coordinate
  space.  When the coordinate space undergoes a transformation (e.g., with
  CarToImageTransform), the heading in the new coordinate space must be
  recomputed.

  We compute this by deriving the formula for the angle of transformed unit
  vector defined by 'heading'.

  Args:
    transform: 4x4 numpy matrix used to convert from car to image coordinates.
    heading: Floating point scalar heading.

  Returns:
    Heading in the transformed coordinate system.
  """
  x1, y1 = np.cos(heading), np.sin(heading)

  # Transform the unit ray.
  unit_ray = np.array([x1, y1, 0.0, 1.0])
  transform_no_shift = CopyTransform(transform)
  transform_no_shift[0, 3] = 0
  transform_no_shift[1, 3] = 0
  transformed_ray = np.matmul(transform_no_shift, unit_ray)
  x2, y2 = transformed_ray[0:2]

  # Use arctan2 to compute the new rotation angle; note that arctan2 takes 'y'
  # and then 'x'.
  new_heading = np.arctan2(y2, x2)
  return new_heading


def TransformPoint(transform, x, y, z):
  """Transform an x, y, z point given the 4x4 `transform`."""
  result = np.matmul(transform, np.array([x, y, z, 1.]))
  return result[0], result[1], result[2]


def CopyTransform(transform):
  """Return a copy of `transform`."""
  return copy.copy(transform)


def MakeCarToImageTransform(pixels_per_meter, image_ref_x, image_ref_y,
                            flip_axes):
  """Creates a 4x4 numpy matrix for car to top down image coordinates.

  Args:
    pixels_per_meter: Number of pixels that represent a meter in top down view.
    image_ref_x: Number of pixels to shift the car in the x direction.
    image_ref_y: Number of pixels to shift the car in the y direction.
    flip_axes: Boolean indicating whether the x/y axes should be flipped[ during
      the transform.

  Returns:
    A 4x4 matrix transform.
  """
  ppm1 = 0. if flip_axes else pixels_per_meter
  ppm2 = -pixels_per_meter if flip_axes else 0.
  # pyformat: disable
  car_to_image_transform = np.array([
      [ppm1, ppm2, 0., image_ref_x],
      [ppm2, ppm1, 0., image_ref_y],
      [0., 0., 1., 0.],
      [0., 0., 0., 1.]])
  # pyformat: enable
  return car_to_image_transform
