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
"""Tests for lingvo.tasks.car.ops.sampling_ops."""

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import ops

import numpy as np


class SamplingOpsTest(parameterized.TestCase, test_utils.TestCase):

  @parameterized.named_parameters([
      ('uniform_uniform', 'uniform', 'uniform', 'auto'),
      ('uniform_closest', 'uniform', 'closest', 'auto'),
      ('farthest_uniform', 'farthest', 'uniform', 'auto'),
      ('farthest_closest', 'farthest', 'closest', 'auto'),
      ('farthest_closest_hash', 'farthest', 'closest', 'hash'),
  ])
  def testBasic(self, cmethod, nmethod, nalgo):
    b, n, m, k = 3, 10000, 128, 128
    g = tf.Graph()
    with g.as_default():
      points = tf.random.uniform(shape=(b, n, 3))
      points_padding = tf.zeros(shape=(b, n))
      center, center_padding, indices, indices_padding = ops.sample_points(
          points=points,
          points_padding=points_padding,
          num_seeded_points=0,
          center_selector=cmethod,
          neighbor_sampler=nmethod,
          neighbor_algorithm=nalgo,
          num_centers=m,
          center_z_min=-np.inf,
          center_z_max=np.inf,
          num_neighbors=k,
          max_distance=0.25)

    # Ensure shapes are known at graph construction.
    self.assertListEqual(center.shape.as_list(), [b, m])
    self.assertListEqual(center_padding.shape.as_list(), [b, m])
    self.assertListEqual(indices.shape.as_list(), [b, m, k])
    self.assertListEqual(indices_padding.shape.as_list(), [b, m, k])

    with self.session(graph=g):
      cp, c, i, p = self.evaluate(
          [center_padding, center, indices, indices_padding])

    # Very basic validity checking.
    self.assertEqual(cp.shape, (b, m))
    self.assertTrue(np.all(cp == 0.0))
    self.assertEqual(c.shape, (b, m))
    self.assertTrue(np.all(np.logical_and(0 <= c, c < n)))
    self.assertEqual(i.shape, (b, m, k))
    self.assertTrue(np.all(np.logical_and(0 <= i, i < n)))
    self.assertEqual(p.shape, (b, m, k))
    self.assertTrue(np.all(np.logical_or(0. == p, 1. == p)))

  @parameterized.named_parameters([
      ('uniform_uniform', 'uniform', 'uniform', 'auto'),
      ('uniform_closest', 'uniform', 'closest', 'auto'),
      ('farthest_uniform', 'farthest', 'uniform', 'auto'),
      ('farthest_closest', 'farthest', 'closest', 'auto'),
      ('farthest_closest_hash', 'farthest', 'closest', 'hash'),
  ])
  def testZFilter(self, cmethod, nmethod, nalgo):
    b, n, m, k = 1, 10000, 128, 128
    g = tf.Graph()
    with g.as_default():
      points = tf.random.uniform(shape=(b, n, 3))
      points_padding = tf.zeros(shape=(b, n))
      center, center_padding, indices, indices_padding = ops.sample_points(
          points=points,
          points_padding=points_padding,
          num_seeded_points=0,
          center_selector=cmethod,
          neighbor_sampler=nmethod,
          num_centers=m,
          center_z_min=0.25,
          center_z_max=0.75,
          num_neighbors=k,
          max_distance=0.25)

    # Ensure shapes are known at graph construction.
    self.assertListEqual(center.shape.as_list(), [b, m])
    self.assertListEqual(center_padding.shape.as_list(), [b, m])
    self.assertListEqual(indices.shape.as_list(), [b, m, k])
    self.assertListEqual(indices_padding.shape.as_list(), [b, m, k])

    with self.session(graph=g):
      c1, p1 = self.evaluate([center, points])
      c2, p2 = self.evaluate([center, points])

    # With extremely high probability, sampling centers twice should be
    # different.
    self.assertGreater(np.setdiff1d(c1, c2).size, 0)

    # Centers should be filtered by z range.
    self.assertTrue((0.25 <= p1[0, c1[0], 2]).all())
    self.assertTrue((p1[0, c1[0], 2] <= 0.75).all())
    self.assertTrue((0.25 <= p2[0, c2[0], 2]).all())
    self.assertTrue((p2[0, c2[0], 2] <= 0.75).all())

  @parameterized.named_parameters([
      ('uniform_uniform', 'uniform', 'uniform'),
      ('uniform_closest', 'uniform', 'closest'),
      ('farthest_uniform', 'farthest', 'uniform'),
      ('farthest_closest', 'farthest', 'closest'),
  ])
  def testSampleFewerCentersThanPoints(self, cmethod, nmethod):
    b, n, m, k = 1, 100, 128, 8
    g = tf.Graph()
    with g.as_default():
      points = tf.random.uniform(shape=(b, n, 3))
      points_padding = tf.zeros(shape=(b, n))
      center, center_padding, indices, indices_padding = ops.sample_points(
          points=points,
          points_padding=points_padding,
          num_seeded_points=0,
          center_selector=cmethod,
          neighbor_sampler=nmethod,
          num_centers=m,
          center_z_min=-np.inf,
          center_z_max=np.inf,
          num_neighbors=k,
          max_distance=0.25)

    # Ensure shapes are known at graph construction.
    self.assertListEqual(center.shape.as_list(), [b, m])
    self.assertListEqual(center_padding.shape.as_list(), [b, m])
    self.assertListEqual(indices.shape.as_list(), [b, m, k])
    self.assertListEqual(indices_padding.shape.as_list(), [b, m, k])

    with self.session(graph=g):
      p, c = self.evaluate([center_padding, center])

    self.assertAllEqual(p[0, :n], np.zeros([n]))
    self.assertAllEqual(p[0, n:], np.ones([m - n]))
    tf.logging.info('c[:n]=%s', c[0, :n])
    self.assertAllEqual(np.sort(c[0, :n]), np.arange(n))
    self.assertAllEqual(c[0, n:], np.zeros([m - n]))


if __name__ == '__main__':
  tf.test.main()
