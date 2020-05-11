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
"""Tests for car_layers."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import car_layers


class CarLayersTest(test_utils.TestCase):

  def _testNestedOutShape(self, p, input_shape, expected_shape):
    batch_size, num_points, _ = input_shape
    g = tf.Graph()
    with g.as_default():
      net = p.Instantiate()
      input_data = py_utils.NestedMap(
          points=tf.random.uniform(input_shape[:-1] + (3,)),
          features=tf.random.uniform(input_shape),
          padding=tf.zeros((batch_size, num_points), dtype=tf.float32),
          label=tf.random.uniform((batch_size,),
                                  minval=0,
                                  maxval=16,
                                  dtype=tf.int32))
      result = net.FPropDefaultTheta(input_data)
    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      np_result = self.evaluate(result)
    grouped_points_result = np_result.grouped_points
    self.assertEqual(grouped_points_result.features.shape,
                     expected_shape.grouped_points.features)
    self.assertEqual(grouped_points_result.points.shape,
                     expected_shape.grouped_points.points)
    self.assertEqual(grouped_points_result.padding.shape,
                     expected_shape.grouped_points.padding)

    query_points_result = np_result.query_points
    self.assertEqual(query_points_result.points.shape,
                     expected_shape.query_points.points)
    self.assertEqual(query_points_result.padding.shape,
                     expected_shape.query_points.padding)

  def testSamplingAndGrouping(self):
    for num_points in [1024, 256]:
      for input_dims in [3, 6, 9]:
        for group_size in [32, 64]:
          p = car_layers.SamplingAndGroupingLayer.Params().Set(
              name='SampleGroupTest',
              num_samples=256,
              ball_radius=0.2,
              group_size=group_size,
              sample_neighbors_uniformly=True)
          grouped_points_shape = py_utils.NestedMap(
              features=(8, 256, group_size, input_dims),
              points=(8, 256, group_size, 3),
              padding=(8, 256, group_size))
          query_points_shape = py_utils.NestedMap(
              points=(8, 256, 3), padding=(8, 256))
          expected_shape = py_utils.NestedMap({
              'grouped_points': grouped_points_shape,
              'query_points': query_points_shape
          })
          self._testNestedOutShape(p, (8, num_points, input_dims),
                                   expected_shape)


if __name__ == '__main__':
  tf.test.main()
