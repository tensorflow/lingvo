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
"""Tests for transform_util."""

import lingvo.compat as tf
from lingvo.tasks.car import transform_util
import numpy as np


class TransformUtilTest(tf.test.TestCase):
  """Tests for methods in transform_util."""

  def testMakeCarToImageTransformFlipAxesTrue(self):
    transform = transform_util.MakeCarToImageTransform(
        pixels_per_meter=10.0, image_ref_x=250, image_ref_y=750, flip_axes=True)
    # pyformat: disable
    self.assertAllClose(
        np.matrix([[0., -10., 0., 250.,],
                   [-10., 0., 0., 750.,],
                   [0., 0., 1., 0.,],
                   [0., 0., 0., 1.,]]),
        transform)
    # pyformat: enable

  def testMakeCarToImageTransformFlipAxesFalse(self):
    transform = transform_util.MakeCarToImageTransform(
        pixels_per_meter=10.0,
        image_ref_x=250,
        image_ref_y=750,
        flip_axes=False)
    # pyformat: disable
    self.assertAllClose(
        np.matrix([[10., 0., 0., 250.,],
                   [0., 10., 0., 750.,],
                   [0., 0., 1., 0.,],
                   [0., 0., 0., 1.,]]),
        transform)
    # pyformat: enable

  def testTransformPoint(self):
    transform = transform_util.MakeCarToImageTransform(
        pixels_per_meter=10.0,
        image_ref_x=250,
        image_ref_y=750,
        flip_axes=False)
    tx, ty, tz = transform_util.TransformPoint(transform, 0.0, 1.0, 0.0)
    # X gets translated.
    self.assertEqual(250., tx)
    # Y gets translated and scaled by pixels_per_meter.
    self.assertEqual(760., ty)
    self.assertEqual(0., tz)

  def testCopyTransform(self):
    # Same transform as above.
    transform = transform_util.MakeCarToImageTransform(
        pixels_per_meter=10.0,
        image_ref_x=250,
        image_ref_y=750,
        flip_axes=False)
    # Test that copying the transform yields the same result.
    copy_transform = transform_util.CopyTransform(transform)
    tx, ty, tz = transform_util.TransformPoint(copy_transform, 0.0, 1.0, 0.0)
    self.assertEqual(250., tx)
    self.assertEqual(760., ty)
    self.assertEqual(0., tz)

  def testBox2DCornerAxisAligned(self):
    # Center at 1., 1., width of 1. and length of 2.
    #
    # The heading of 0. indicates the heading of the 'long'
    # side of the box, so it is longer in the positive y direction.
    box = transform_util.Box2D(1.0, 1.0, 2., 1., 0.)
    self.assertAllClose(box.corners,
                        [[0.5, 2.], [1.5, 2.], [1.5, 0.], [0.5, 0.]])

  def testBox2DCornerRotated(self):
    # Like above but rotated 90 degrees; the order is important.
    box = transform_util.Box2D(1.0, 1.0, 2., 1., np.pi / 2.)
    self.assertAllClose(box.corners,
                        [[0., 0.5], [0., 1.5], [2., 1.5], [2., 0.5]])

  def testBox2DTransform(self):
    # Take the box from above and apply a car-image transform.
    box = transform_util.Box2D(1.0, 1.0, 2., 1., 0.)
    transform = transform_util.MakeCarToImageTransform(
        pixels_per_meter=10.0, image_ref_x=250, image_ref_y=750, flip_axes=True)
    new_box = box.Apply(transform)

    # The center flips across the x=y axis to -1, -1.  After the scaling and
    # translation, the box should be centered (240, 740).  Because the box flips
    # across the axis, the width and length get flipped from 2, 1 to [10, 20].
    #
    # The flip axes should cause the heading to go from 0. to -pi/2.
    self.assertAllClose([240., 740., 10., 20., -np.pi / 2.], new_box.AsNumpy())

    # Check ymin/xmin/ymax/xmax: the rectangle is now longer in the y-dimension
    # than the x-dimension.
    self.assertAllClose((730., 235., 750., 245.), new_box.Extrema())

  def testTransformHeading(self):
    transform = transform_util.MakeCarToImageTransform(
        pixels_per_meter=1.0, image_ref_x=123, image_ref_y=455, flip_axes=True)

    # Ray (0, 1): 90 degrees becomes -180 degrees.
    self.assertAllClose(-np.pi,
                        transform_util.TransformHeading(transform, np.pi / 2.))

    # Ray (1, 0): 0 degrees becomes -90
    self.assertAllClose(-np.pi / 2.,
                        transform_util.TransformHeading(transform, 0.))

    # (-1, 0) becomes (0, 1) or np.pi / 2.
    self.assertAllClose(np.pi / 2.,
                        transform_util.TransformHeading(transform, np.pi))

    # (0, -1) becomes (1, 0) or 0
    self.assertAllClose(0.,
                        transform_util.TransformHeading(transform, 1.5 * np.pi))


if __name__ == '__main__':
  tf.test.main()
