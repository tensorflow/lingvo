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
"""Tests for pointnet."""

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import pointnet


class PointNetTest(test_utils.TestCase, parameterized.TestCase):

  def _testOutShape(self, p, input_shape, expected_shape):
    batch_size, num_points, _ = input_shape
    g = tf.Graph()
    with g.as_default():
      net = p.Instantiate()
      input_data = py_utils.NestedMap(
          points=tf.random.uniform((batch_size, num_points, 3)),
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
    self.assertEqual(np_result.shape, expected_shape)

  @parameterized.parameters((128, 3), (128, 9), (256, 3))
  def testPointNetClassifier(self, feature_dims, input_dims):
    p = pointnet.PointNet().Classifier(
        input_dims=input_dims, feature_dims=feature_dims)
    # Network should produce a global feature of feature_dims.
    self.assertEqual(p.output_dim, feature_dims)
    self._testOutShape(p, (8, 128, input_dims), (8, feature_dims))

  def testPointNetSegmentation(self):
    p = pointnet.PointNet().Segmentation()
    # Network takes batch_size=8 input and produce 128-dim pointwise feature.
    self.assertEqual(p.output_dim, 128)
    self._testOutShape(p, (8, 100, 3), (8, 100, 128))

  def testPointNetSegmentationShapeNet(self):
    p = pointnet.PointNet().SegmentationShapeNet()
    self.assertEqual(p.output_dim, 128)
    self._testOutShape(p, (8, 2000, 3), (8, 2000, 128))

  @parameterized.parameters((128, 3), (128, 9), (256, 3))
  def testPointNetPPClassifier(self, feature_dims, input_dims):
    p = pointnet.PointNetPP().Classifier(
        input_dims=input_dims, feature_dims=feature_dims)
    # Network should produce a global feature of feature_dims.
    self.assertEqual(p.output_dim, feature_dims)
    self._testOutShape(p, (8, 1024, input_dims), (8, feature_dims))


if __name__ == '__main__':
  tf.test.main()
