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
"""Tests for lingvo.tasks.car.ops.sampling_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import ops

import numpy as np


class SamplingOpsTest(parameterized.TestCase, test_utils.TestCase):

  @parameterized.named_parameters([
      ('uniform_uniform', 'uniform', 'uniform'),
      ('uniform_closest', 'uniform', 'closest'),
      ('farthest_uniform', 'farthest', 'uniform'),
      ('farthest_closest', 'farthest', 'closest'),
  ])
  def testBasic(self, cmethod, nmethod):
    n, m, k = 10000, 128, 128
    g = tf.Graph()
    with g.as_default():
      points = tf.random.uniform(shape=(n, 3))
      center_padding, center, indices, padding = ops.sample_points(
          points=points,
          center_selector=cmethod,
          neighbor_sampler=nmethod,
          num_centers=m,
          center_z_min=-np.inf,
          center_z_max=np.inf,
          num_neighbors=k,
          max_distance=0.25)

    # Ensure shapes are known at graph construction.
    self.assertListEqual(center_padding.shape.as_list(), [m])
    self.assertListEqual(center.shape.as_list(), [m])
    self.assertListEqual(indices.shape.as_list(), [m, k])
    self.assertListEqual(padding.shape.as_list(), [m, k])

    with self.session(graph=g) as sess:
      cp, c, i, p = sess.run([center_padding, center, indices, padding])

    # Very basic validity checking.
    self.assertEqual(cp.shape, (m,))
    self.assertTrue(np.all(cp == 0.0))
    self.assertEqual(c.shape, (m,))
    self.assertTrue(np.all(np.logical_and(0 <= c, c < n)))
    self.assertEqual(i.shape, (m, k))
    self.assertTrue(np.all(np.logical_and(0 <= i, i < n)))
    self.assertEqual(p.shape, (m, k))
    self.assertTrue(np.all(np.logical_or(0. == p, 1. == p)))

  @parameterized.named_parameters([
      ('uniform_uniform', 'uniform', 'uniform'),
      ('uniform_closest', 'uniform', 'closest'),
      ('farthest_uniform', 'farthest', 'uniform'),
      ('farthest_closest', 'farthest', 'closest'),
  ])
  def testZFilter(self, cmethod, nmethod):
    n, m, k = 10000, 128, 128
    g = tf.Graph()
    with g.as_default():
      points = tf.random.uniform(shape=(n, 3))
      center_padding, center, indices, padding = ops.sample_points(
          points=points,
          center_selector=cmethod,
          neighbor_sampler=nmethod,
          num_centers=m,
          center_z_min=0.25,
          center_z_max=0.75,
          num_neighbors=k,
          max_distance=0.25)

    # Ensure shapes are known at graph construction.
    self.assertListEqual(center_padding.shape.as_list(), [m])
    self.assertListEqual(center.shape.as_list(), [m])
    self.assertListEqual(indices.shape.as_list(), [m, k])
    self.assertListEqual(padding.shape.as_list(), [m, k])

    with self.session(graph=g) as sess:
      c1, p1 = sess.run([center, points])
      c2, p2 = sess.run([center, points])

    # With extremely high probability, sampling centers twice should be
    # different.
    self.assertGreater(np.setdiff1d(c1, c2).size, 0)

    # Centers should be filtered by z range.
    self.assertTrue((0.25 <= p1[c1, 2]).all())
    self.assertTrue((p1[c1, 2] <= 0.75).all())
    self.assertTrue((0.25 <= p2[c2, 2]).all())
    self.assertTrue((p2[c2, 2] <= 0.75).all())

  @parameterized.named_parameters([
      ('uniform_uniform', 'uniform', 'uniform'),
      ('uniform_closest', 'uniform', 'closest'),
      ('farthest_uniform', 'farthest', 'uniform'),
      ('farthest_closest', 'farthest', 'closest'),
  ])
  def testSampleFewerCentersThanPoints(self, cmethod, nmethod):
    n, m, k = 100, 128, 8
    g = tf.Graph()
    with g.as_default():
      points = tf.random.uniform(shape=(n, 3))
      center_padding, center, indices, padding = ops.sample_points(
          points=points,
          center_selector=cmethod,
          neighbor_sampler=nmethod,
          num_centers=m,
          center_z_min=-np.inf,
          center_z_max=np.inf,
          num_neighbors=k,
          max_distance=0.25)

    # Ensure shapes are known at graph construction.
    self.assertListEqual(center_padding.shape.as_list(), [m])
    self.assertListEqual(center.shape.as_list(), [m])
    self.assertListEqual(indices.shape.as_list(), [m, k])
    self.assertListEqual(padding.shape.as_list(), [m, k])

    with self.session(graph=g) as sess:
      p, c = sess.run([center_padding, center])

    self.assertAllEqual(p[:n], np.zeros([n]))
    self.assertAllEqual(p[n:], np.ones([m - n]))
    tf.logging.info('c[:n]=%s', c[:n])
    self.assertAllEqual(np.sort(c[:n]), np.arange(n))
    self.assertAllEqual(c[n:], np.zeros([m - n]))


if __name__ == '__main__':
  tf.test.main()
