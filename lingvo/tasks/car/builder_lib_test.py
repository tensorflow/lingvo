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
"""Tests for builder_lib."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import builder_lib
import numpy as np


class BuilderLibTest(test_utils.TestCase):

  def _TestFProp(self, p, in_shape, expected_out_shape):
    g = tf.Graph()
    with g.as_default():
      l = p.Instantiate()
      x = tf.random.normal(shape=in_shape)
      y = l.FPropDefaultTheta(x)
      if isinstance(y, (list, tuple)):
        self.assertEqual(len(y), 1)
        y = y[0]
    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      val = self.evaluate(y)
    self.assertEqual(val.shape, expected_out_shape)

  def testLN(self):
    self._TestFProp(builder_lib.ModelBuilderBase()._LN('ln', 128),
                    (4, 100, 128), (4, 100, 128))

  def testProject(self):
    self._TestFProp(builder_lib.ModelBuilderBase()._Project('p', 64, 128),
                    (4, 100, 64), (4, 100, 128))

  def testAdd(self):
    b = builder_lib.ModelBuilderBase()
    p = b._Add('p', b._Seq('l'), b._Seq('r'))
    self._TestFProp(p, (4, 100, 64), (4, 100, 64))

  def testAttenFF(self):
    b = builder_lib.ModelBuilderBase()
    p = b._AttenFF('p', 128, 256, 0.8)
    self._TestFProp(p, (4, 100, 128), (4, 100, 128))

  def testAttenSelf(self):
    b = builder_lib.ModelBuilderBase()
    p = b._AttenSelf('p', 128, 256, 4, 0.8)
    self._TestFProp(p, (4, 100, 128), (4, 100, 128))

  def testAtten(self):
    b = builder_lib.ModelBuilderBase()
    p = b._Atten('p', 128, 256, 4, 0.8)
    self._TestFProp(p, (4, 100, 128), (4, 100, 128))

  def testSelfAttenStack(self):
    b = builder_lib.ModelBuilderBase()
    p = b._SelfAttenStack('p', 6, 128, 256, 4, 0.8)
    self._TestFProp(p, (4, 100, 128), (4, 100, 128))

  def testGetValue(self):
    b = builder_lib.ModelBuilderBase()
    p = b._GetValue('p', 'test_key')
    l = p.Instantiate()
    x = py_utils.NestedMap({
        'test_key': tf.constant(2.0),
        'ignore_key': tf.constant(1.0)
    })
    y = l.FPropDefaultTheta(x)
    with self.session():
      actual_y = self.evaluate(y)
      self.assertEqual(actual_y, 2.0)

  def testGetValueDefault(self):
    b = builder_lib.ModelBuilderBase()
    p = b._GetValue('p', 'not_a_key', default=tf.constant(3.0))
    l = p.Instantiate()
    x = py_utils.NestedMap({
        'test_key': tf.constant(2.0),
        'ignore_key': tf.constant(1.0)
    })
    y = l.FPropDefaultTheta(x)
    with self.session():
      actual_y = self.evaluate(y)
      self.assertEqual(actual_y, 3.0)

  def testSeqToKeyFeaturesOnly(self):
    b = builder_lib.ModelBuilderBase()
    p = b._SeqToKey('p', 'features', b._GetValue('get_features', 'features'),
                    b._Sigmoid('sigmoid'))
    l = p.Instantiate()
    x = py_utils.NestedMap({
        'points': tf.constant(3.0),
        'features': tf.constant(0.0),
        'padding': tf.constant(1.0),
    })
    y = l.FPropDefaultTheta(x)
    with self.session():
      actual_y = self.evaluate(y)
      self.assertDictEqual(
          actual_y, {
              'points': np.asarray([3.0]),
              'features': np.asarray([0.5]),
              'padding': np.asarray([1.0]),
          })

  def testSeqToKeyFeaturesRaisesIfNotNestedMap(self):
    b = builder_lib.ModelBuilderBase()
    p = b._SeqToKey('p', 'features', b._GetValue('get_features', 'features'),
                    b._Sigmoid('sigmoid'))
    l = p.Instantiate()
    x = tf.constant(1.0)
    with self.assertRaisesRegex(ValueError,
                                r'Input not a `NestedMap`. Is a .*'):
      l.FPropDefaultTheta(x)

  def testSeqToKey(self):
    b = builder_lib.ModelBuilderBase()
    # Update features with sigmoid(padding).
    p = b._SeqToKey('p', 'features', b._GetValue('get_padding', 'padding'),
                    b._Sigmoid('sigmoid'))
    l = p.Instantiate()
    x = py_utils.NestedMap({
        'points': tf.constant(3.0),
        'features': tf.constant(5.0),
        'padding': tf.constant(0.0),
        'extra_key': tf.constant(1.0),
    })
    y = l.FPropDefaultTheta(x)
    with self.session():
      actual_y = self.evaluate(y)
      self.assertDictEqual(
          actual_y, {
              'points': np.asarray([3.0]),
              'features': np.asarray([0.5]),
              'padding': np.asarray([0.0]),
              'extra_key': np.asarray([1.0]),
          })

  def _getNestedMapTestData(self):
    np_x = py_utils.NestedMap({
        'points':
            np.random.rand(3, 5, 3),
        'features':
            np.random.rand(3, 5, 5),
        'padding':
            np.asarray([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    })
    x = py_utils.NestedMap({
        'points': tf.constant(np_x.points, dtype=tf.float32),
        'features': tf.constant(np_x.features, dtype=tf.float32),
        'padding': tf.constant(np_x.padding, dtype=tf.float32),
    })
    return np_x, x

  def testPaddedMax(self):
    b = builder_lib.ModelBuilderBase()
    p = b._PaddedMax('p')
    l = p.Instantiate()

    np_x, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    expected_y = np.stack([
        # First example should take max over the first two points.
        np.amax(np_x.features[0, :2, :], axis=0),
        # Second example should take max over only the first points.
        np_x.features[1, 0, :],
        # Third example should be all zeros, since all points are padded.
        np.zeros_like(np_x.features[2, 0, :]),
    ], axis=0)  # pyformat: disable
    with self.session():
      actual_y = self.evaluate(y)
      self.assertAllClose(actual_y, expected_y)

  def testPaddedMaxNestedOutput(self):
    b = builder_lib.ModelBuilderBase()
    p = b._PaddedMax('p', nested_output=True)
    l = p.Instantiate()

    np_x, x = self._getNestedMapTestData()
    out = l.FPropDefaultTheta(x)

    expected_y = np.stack([
        # First example should take max over the first two points.
        np.amax(np_x.features[0, :2, :], axis=0),
        # Second example should take max over only the first points.
        np_x.features[1, 0, :],
        # Third example should be all zeros, since all points are padded.
        np.zeros_like(np_x.features[2, 0, :]),
    ], axis=0)  # pyformat: disable
    with self.session():
      actual_out = self.evaluate(out)
      self.assertAllClose(actual_out.features, expected_y)
      self.assertAllEqual(actual_out.padding, np.asarray([0., 0., 1.]))

  def testPaddedMean(self):
    b = builder_lib.ModelBuilderBase()
    p = b._PaddedMean('p')
    l = p.Instantiate()

    np_x, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    expected_y = np.stack([
        # First example should take mean over the first two points.
        np.mean(np_x.features[0, :2, :], axis=0),
        # Second example should take mean over only the first points.
        np_x.features[1, 0, :],
        # Third example should be all zeros, since all points are padded.
        np.zeros_like(np_x.features[2, 0, :]),
    ], axis=0)  # pyformat: disable
    with self.session():
      actual_y = self.evaluate(y)
      self.assertAllClose(actual_y, expected_y)

  def testPaddedMeanGrad(self):
    b = builder_lib.ModelBuilderBase()
    p = b._Seq('seq', b._FeaturesFC('fc', 5, 10), b._PaddedMean('p'))
    l = p.Instantiate()

    _, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)
    loss = tf.reduce_sum(y)

    all_vars = tf.trainable_variables()
    grads = tf.gradients(loss, all_vars)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      np_grads = self.evaluate(grads)
      for np_grad in np_grads:
        self.assertTrue(np.all(np.isfinite(np_grad)))

  def testPaddedSum(self):
    b = builder_lib.ModelBuilderBase()
    p = b._PaddedSum('p')
    l = p.Instantiate()

    np_x, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    expected_y = np.stack([
        # First example should take sum over the first two points.
        np.sum(np_x.features[0, :2, :], axis=0),
        # Second example should take sum over only the first points.
        np_x.features[1, 0, :],
        # Third example should be all zeros, since all points are padded.
        np.zeros_like(np_x.features[2, 0, :]),
    ], axis=0)  # pyformat: disable
    with self.session():
      actual_y = self.evaluate(y)
      self.assertAllClose(actual_y, expected_y)

  def testFeaturesFC(self):
    b = builder_lib.ModelBuilderBase()
    p = b._FeaturesFC('p', idims=5, odims=6, use_bn=True)
    l = p.Instantiate()

    np_x, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      # Points and padding should be equal, and features should be transformed.
      self.assertAllClose(actual_y.points, np_x.points)
      self.assertAllEqual(actual_y.padding, np_x.padding)
      self.assertAllEqual(actual_y.features.shape, (3, 5, 6))
      self.assertTrue(np.all(np.isfinite(actual_y.features)))

  def testFeaturesFCNoBatchNorm(self):
    b = builder_lib.ModelBuilderBase()
    p = b._FeaturesFC('p', idims=5, odims=6, use_bn=False)
    l = p.Instantiate()

    np_x, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      # Points and padding should be equal, and features should be transformed.
      self.assertAllClose(actual_y.points, np_x.points)
      self.assertAllEqual(actual_y.padding, np_x.padding)
      self.assertAllEqual(actual_y.features.shape, (3, 5, 6))
      self.assertTrue(np.all(np.isfinite(actual_y.features)))

  def testFeaturesFCBNAfterLinear(self):
    b = builder_lib.ModelBuilderBase()
    b.fc_bn_after_linear = True
    p = b._FeaturesFC('p', idims=5, odims=6, use_bn=True)
    l = p.Instantiate()

    np_x, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      # Points and padding should be equal, and features should be transformed.
      self.assertAllClose(actual_y.points, np_x.points)
      self.assertAllEqual(actual_y.padding, np_x.padding)
      self.assertAllEqual(actual_y.features.shape, (3, 5, 6))
      self.assertTrue(np.all(np.isfinite(actual_y.features)))

  def testCondFC(self):
    b = builder_lib.ModelBuilderBase()
    p = b._CondFC('p', idims=10, adims=8, odims=12)
    l = p.Instantiate()
    y = l.FPropDefaultTheta(
        tf.random.uniform((3, 4, 5, 10)), tf.random.uniform((3, 4, 1, 8)))
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      self.assertAllEqual(actual_y.shape, (3, 4, 5, 12))
      self.assertTrue(np.all(np.isfinite(actual_y)))

  def testGINWithCondFC(self):
    b = builder_lib.ModelBuilderBase()
    p = b._GIN(
        'p', [[5, 2, 6], [6, 8, 7]],
        b._PaddedMax('p_max'),
        b._PaddedMax('p_max'),
        combine_method='cond_fc')
    l = p.Instantiate()

    _, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      # We expect the output to have all the features concatenated together.
      self.assertAllEqual(actual_y.shape, (3, 5 + 6 + 7))
      self.assertTrue(np.all(np.isfinite(actual_y)))

  def testGINIntermediateLayer(self):
    b = builder_lib.ModelBuilderBase()
    p = b._GINIntermediateLayer('p', [5, 3, 6], b._PaddedSum('sum'))
    l = p.Instantiate()

    np_x, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      # Points and padding should be equal, and features should be transformed.
      self.assertAllClose(actual_y.points, np_x.points)
      self.assertAllEqual(actual_y.padding, np_x.padding)
      self.assertAllEqual(actual_y.features.shape, (3, 5, 6))
      self.assertTrue(np.all(np.isfinite(actual_y.features)))

  def testGINLayer(self):
    b = builder_lib.ModelBuilderBase()
    p = b._GIN('p', [[5, 2, 6], [6, 8, 7]], b._PaddedMax('p_max'),
               b._PaddedMax('p_max'))
    l = p.Instantiate()

    _, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      # We expect the output to have all the features concatenated together.
      self.assertAllEqual(actual_y.shape, (3, 5 + 6 + 7))
      self.assertTrue(np.all(np.isfinite(actual_y)))

  def testGINLayerRaisesUnexpectedCombiner(self):
    b = builder_lib.ModelBuilderBase()
    with self.assertRaisesRegex(ValueError, 'Unexpected combine method: .*'):
      b._GIN(
          'p', [[5, 2, 6], [6, 8, 7]],
          b._PaddedMax('p_max'),
          b._PaddedMax('p_max'),
          combine_method='unknown')

  def testGINLayerRaisesDimsMismatch(self):
    b = builder_lib.ModelBuilderBase()
    with self.assertRaisesRegex(ValueError, 'mlp_dims do not match .*'):
      b._GIN(
          'p', [[5, 2, 6], [6, 8, 7]],
          b._PaddedMax('p_max'),
          b._PaddedMax('p_max'),
          combine_method='concat')

  def testBroadcastConcat(self):
    b = builder_lib.ModelBuilderBase()
    p = b._BroadcastConcat('p', b._ArgIdx('arg0', [0]), b._ArgIdx('arg1', [1]))
    l = p.Instantiate()

    # 2 x 1 x 3
    x1 = np.asarray([[[1, 2, 3]], [[4, 5, 6]]])
    self.assertEqual(x1.shape, (2, 1, 3))
    # 1 x 4 x 2
    x2 = np.asarray([[[7, 8], [9, 10], [11, 12], [13, 14]]])
    self.assertEqual(x2.shape, (1, 4, 2))

    y = l.FPropDefaultTheta(tf.constant(x1), tf.constant(x2))

    expected_y = np.concatenate([
        np.broadcast_to(x1, [2, 4, 3]),
        np.broadcast_to(x2, [2, 4, 2]),
    ], axis=-1)  # pyformat: disable
    with self.session():
      actual_y = self.evaluate(y)
      # x1 will be broadcasted to [2, 4, 3]
      # x2 will be broadcasted to [2, 4, 2]
      # Concatenation on axis=-1 should result in a tensor of shape [2, 4, 5]
      self.assertEqual(actual_y.shape, (2, 4, 5))
      self.assertAllEqual(actual_y, expected_y)

  def testGINLayerWithConcat(self):
    b = builder_lib.ModelBuilderBase()
    p = b._GIN(
        # The first dim in each MLP should be double the last output as we are
        # using concat combiner.
        'p',
        [[10, 2, 6], [12, 8, 7]],
        b._PaddedMax('p_max'),
        b._PaddedMax('p_max'),
        combine_method='concat')
    l = p.Instantiate()

    _, x = self._getNestedMapTestData()
    y = l.FPropDefaultTheta(x)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      # We expect the output to have all the features concatenated together.
      self.assertAllEqual(actual_y.shape, (3, 5 + 6 + 7))
      self.assertTrue(np.all(np.isfinite(actual_y)))

  def testResidualBlock(self):
    b = builder_lib.ModelBuilderBase()
    p_square_stride = b._ResidualBlock('residual_block', (3, 3, 4, 8), (2, 2),
                                       2)
    self._TestFProp(p_square_stride, (4, 100, 100, 4), (4, 50, 50, 8))
    p_nonsquare_stride = b._ResidualBlock('residual_block', (3, 3, 4, 8),
                                          (2, 1), 2)
    self._TestFProp(p_nonsquare_stride, (4, 100, 100, 4), (4, 50, 100, 8))

  def testPointConvParametricConvShapes(self):
    batch_size, num_groups, points_per_group, num_in_channels = 4, 5, 6, 7
    num_out_channels = 8
    b = builder_lib.ModelBuilderBase()
    p = b._PointConvParametricConv('test', [3, 4, 9], num_in_channels,
                                   num_out_channels)
    l = p.Instantiate()
    x = py_utils.NestedMap(
        points=tf.random.uniform((batch_size, num_groups, points_per_group, 3),
                                 dtype=tf.float32),
        features=tf.random.uniform(
            (batch_size, num_groups, points_per_group, num_in_channels),
            dtype=tf.float32),
        padding=tf.cast(
            tf.random.uniform((batch_size, num_groups, points_per_group),
                              minval=0,
                              maxval=2,
                              dtype=tf.int32), tf.float32))
    y = l.FPropDefaultTheta(x)

    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
      self.assertAllEqual(actual_y.shape,
                          (batch_size, num_groups, num_out_channels))

  def testActivationReluString(self):
    b = builder_lib.ModelBuilderBase()
    x = tf.constant([2, 5, 1, -3])
    p = b._Activation('RELU', activation_fn_or_name='RELU')
    l = p.Instantiate()
    y = l.FPropDefaultTheta(x)
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
    self.assertAllEqual(actual_y, [2, 5, 1, 0])

  def testActivationReluFn(self):
    b = builder_lib.ModelBuilderBase()
    x = tf.constant([2, 5, 1, -3])
    p = b._Activation('RELU', activation_fn_or_name=tf.nn.relu)
    l = p.Instantiate()
    y = l.FPropDefaultTheta(x)
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
    self.assertAllEqual(actual_y, [2, 5, 1, 0])

  def testActivationIdentityFn(self):
    b = builder_lib.ModelBuilderBase()
    x = tf.constant([2, 5, 1, -3])
    p = b._Activation('Identity', activation_fn_or_name=tf.identity)
    l = p.Instantiate()
    y = l.FPropDefaultTheta(x)
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
    self.assertAllEqual(actual_y, [2, 5, 1, -3])

  def testActivationDefault(self):
    b = builder_lib.ModelBuilderBase()
    x = tf.constant([2, 5, 1, -3])
    # Default activation is relu
    p = b._Activation('Default', activation_fn_or_name=None)
    l = p.Instantiate()
    y = l.FPropDefaultTheta(x)
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      actual_y = self.evaluate(y)
    self.assertAllEqual(actual_y, [2, 5, 1, 0])


if __name__ == '__main__':
  tf.test.main()
