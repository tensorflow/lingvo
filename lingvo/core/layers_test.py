# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.python.framework import ops
from lingvo.core import gpipe
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import test_utils


class ActivationsTest(test_utils.TestCase):

  def testGeluActivation(self):
    with self.session(use_gpu=True):
      inputs = tf.constant(
          np.linspace(-10.0, 10.0, num=21, dtype='float32'), dtype=tf.float32)
      grads_gelu = tf.gradients(layers.Gelu(inputs), inputs)
      grads_relu = tf.gradients(tf.nn.relu(inputs), inputs)

      self.assertEqual(0.0,
                       layers.Gelu(tf.constant(-10.0, dtype='float32')).eval())
      self.assertEqual(0.0,
                       layers.Gelu(tf.constant(0.0, dtype='float32')).eval())
      self.assertEqual(10.0,
                       layers.Gelu(tf.constant(10.0, dtype='float32')).eval())
      actual_grads_gelu = grads_gelu[0].eval()
      actual_grads_relu = grads_relu[0].eval()

      self.assertAllClose(actual_grads_gelu[-5:], actual_grads_relu[-5:])
      self.assertAllClose(actual_grads_gelu[:5], actual_grads_relu[:5])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_grads_gelu = [
          -7.69459925e-22, -9.25176121e-18, -4.04182472e-14, -6.39430453e-11,
          -3.64552299e-08, -7.13557529e-06, -5.03641320e-04, -1.19456425e-02,
          -8.52318183e-02, -8.33154917e-02,  5.00000000e-01,  1.08331549e+00,
          1.08523178e+00,  1.01194561e+00,  1.00050366e+00,  1.00000715e+00,
          1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
          1.00000000e+00]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_grads_gelu, actual_grads_gelu)


class BatchNormLayerTest(test_utils.TestCase):

  def testBatchNormLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn'
      params.dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      layers.BatchNormLayer(params)
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'bn/beta/var:0', 'bn/gamma/var:0', 'bn/moving_mean/var:0',
          'bn/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)

  def testBatchNormLayerMoments(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)

      in_padding1 = tf.zeros([2, 2, 8, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 2, 8, 2]), dtype=tf.float32)
      mean1, var1 = layers.BatchNormLayer._Moments(bn_in1, 1.0 - in_padding1)
      mean2, var2 = tf.nn.moments(bn_in1, [0, 1, 2])

      in_padding2 = tf.ones([2, 2, 8, 1], dtype=tf.float32)
      bn_in2 = tf.constant(
          np.random.normal(-0.3, 1.0, [2, 2, 8, 2]), dtype=tf.float32)
      in_padding3 = tf.concat([in_padding1, in_padding2], 1)
      bn_in3 = tf.concat([bn_in1, bn_in2], 1)
      mean3, var3 = layers.BatchNormLayer._Moments(bn_in3, 1.0 - in_padding3)
      mean4, var4 = tf.nn.moments(bn_in3, [0, 1, 2])

      mean_diff = tf.reduce_sum(tf.square(mean3 - mean4))
      var_diff = tf.reduce_sum(tf.square(var3 - var4))

      tf.global_variables_initializer().run()

      self.assertAllClose(mean2.eval(), mean1.eval())
      self.assertAllClose(var2.eval(), var1.eval())
      self.assertAllClose(mean3.eval(), mean1.eval())
      self.assertAllClose(var3.eval(), var1.eval())
      # Since tf.nn.moments() doesn't support padding, it is expected to produce
      # different results than our own implementation (of moments).
      self.assertAllClose(0.095987, mean_diff.eval())
      self.assertAllClose(0.364456, var_diff.eval())

  def testBatchNormLayerFProp(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn'
      params.dim = 3
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False

      bn_layer = layers.BatchNormLayer(params)
      in_padding1 = tf.zeros([2, 8, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      tf.global_variables_initializer().run()
      self.assertAllClose(0.0, sig1.eval(), atol=1e-5)
      self.assertAllClose(47.8371887, sig2.eval())

  def testBatchNormLayerFPropUseGlobalStatsForTraining(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn'
      params.dim = 3
      params.use_moving_avg_in_training = True
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False

      bn_layer = layers.BatchNormLayer(params)
      in_padding1 = tf.zeros([2, 8, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      tf.global_variables_initializer().run()
      self.assertAllClose(2.6593573, sig1.eval(), atol=1e-5)
      self.assertAllClose(15.464208, sig2.eval())

  def testBatchNormLayerMomentsForConv(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)

      in_padding1 = tf.zeros([2, 8, 1, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 4, 3]), dtype=tf.float32)
      mean1, var1 = layers.BatchNormLayer._Moments(bn_in1, 1.0 - in_padding1)
      mean2, var2 = tf.nn.moments(bn_in1, [0, 1, 2])

      in_padding2 = tf.ones([2, 8, 1, 1], dtype=tf.float32)
      bn_in2 = tf.constant(
          np.random.normal(-0.3, 1.0, [2, 8, 4, 3]), dtype=tf.float32)
      in_padding3 = tf.concat([in_padding1, in_padding2], 1)
      bn_in3 = tf.concat([bn_in1, bn_in2], 1)
      mean3, var3 = layers.BatchNormLayer._Moments(bn_in3, 1.0 - in_padding3)
      mean4, var4 = tf.nn.moments(bn_in3, [0, 1, 2])

      mean_diff = tf.reduce_sum(tf.square(mean3 - mean4))
      var_diff = tf.reduce_sum(tf.square(var3 - var4))

      tf.global_variables_initializer().run()

      self.assertAllClose(mean2.eval(), mean1.eval())
      self.assertAllClose(var2.eval(), var1.eval())
      self.assertAllClose(mean3.eval(), mean1.eval())
      self.assertAllClose(var3.eval(), var1.eval())
      self.assertAllClose(0.1726295, mean_diff.eval())
      self.assertAllClose(0.5592572093009949, var_diff.eval())

  def testBatchNormLayerFPropForConv(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn_conv'
      params.dim = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False

      bn_layer = layers.BatchNormLayer(params)
      in_padding1 = tf.zeros([2, 8, 1, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 4, 32]), dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      tf.global_variables_initializer().run()
      self.assertAllClose(0.0, sig1.eval(), atol=1e-4)
      self.assertAllClose(2039.398681, sig2.eval())


class ConvLayerTest(test_utils.TestCase):
  """Tests conv layers.

  Note that there are multiple subclasses of BaseConv2DLayer and most cases
  are tested via the concrete Conv2DLayer. Other tests are done against
  other subclasses to cover key differences.
  """

  def testConv2DLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.Conv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      layers.Conv2DLayer(params)
      conv_vars = tf.get_collection('Conv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/w/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'conv/beta/var:0', 'conv/gamma/var:0', 'conv/moving_mean/var:0',
          'conv/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)

  def testDepthwiseConv2DLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      layers.DepthwiseConv2DLayer(params)
      conv_vars = tf.get_collection('DepthwiseConv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/w/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'conv/beta/var:0', 'conv/gamma/var:0', 'conv/moving_mean/var:0',
          'conv/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)

  def testSeparableConv2DLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.SeparableConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      params.cls(params)
      # Vars for the outer conv layer.
      conv_vars = tf.get_collection('SeparableConv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/w/var:0']
      self.assertSetEqual(set(expected_var_names), set(conv_var_names))
      # Vars for the inner depthwise layer.
      conv_vars = tf.get_collection('DepthwiseConv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/depthwise_conv/w/var:0']
      self.assertSetEqual(set(expected_var_names), set(conv_var_names))
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          # Outer conv batchnorm.
          'conv/beta/var:0',
          'conv/gamma/var:0',
          'conv/moving_mean/var:0',
          'conv/moving_variance/var:0',
          # Inner depthwise batchnorm.
          'conv/depthwise_conv/beta/var:0',
          'conv/depthwise_conv/gamma/var:0',
          'conv/depthwise_conv/moving_mean/var:0',
          'conv/depthwise_conv/moving_variance/var:0',
      ]
      self.assertSetEqual(set(expected_var_names), set(bn_var_names))

  def testConv2DLayerWithBiasConstruction(self):
    """Tests Conv2DLayer with only bias and without batch normalization."""
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.Conv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      params.bias = True
      params.batch_norm = False
      layers.Conv2DLayer(params)
      conv_vars = tf.get_collection('Conv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      # Has both 'w' and 'b'.
      expected_var_names = ['conv/w/var:0', 'conv/b/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      # No BatchNorm variables.
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = []
      self.assertEqual(expected_var_names, bn_var_names)

  def testDepthwiseConv2DLayerWithBiasConstruction(self):
    """Tests DepthwiseConv2D with only bias and without batch normalization."""
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      params.bias = True
      params.batch_norm = False
      layers.DepthwiseConv2DLayer(params)
      conv_vars = tf.get_collection('DepthwiseConv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      # Has both 'w' and 'b'.
      expected_var_names = ['conv/w/var:0', 'conv/b/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      # No BatchNorm variables.
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = []
      self.assertEqual(expected_var_names, bn_var_names)

  def testConv2DLayerOutShape(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.Conv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = layers.Conv2DLayer(params)
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 5, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 10, 5, 32])

  def testDepthwiseConv2DLayerOutShape(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = layers.DepthwiseConv2DLayer(params)
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 5, 96])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 10, 5, 96])

  def testSeparableConv2DLayerOutShape(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.SeparableConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = params.cls(params)
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 5, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 10, 5, 32])

  def testConv2DLayerWithDilationOutShape(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.Conv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [1, 1]
      params.dilation_rate = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = layers.Conv2DLayer(params)
      # dilation_rate does not change output shape.
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 10, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 20, 10, 32])

  def testDepthwiseConv2DLayerWithDilationOutShape(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [1, 1]
      params.dilation_rate = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = layers.DepthwiseConv2DLayer(params)
      # dilation_rate does not change output shape.
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 10, 96])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 20, 10, 96])

  def testSeparableConv2DLayerWithDilationOutShape(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.SeparableConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [1, 1]
      params.dilation_rate = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = params.cls(params)
      # dilation_rate does not change output shape.
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 10, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 20, 10, 32])

  def testConvPoolComputeOutPadding(self):
    with self.session(use_gpu=True):
      in_padding = tf.constant(
          [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
          dtype=tf.float32)
      out_padding = layers._ComputeConvOutputPadding(in_padding, 2, 2)
      expected_out_padding = [[1, 1, 0, 0, 0, 1, 1, 0],
                              [1, 1, 0, 0, 0, 1, 1, 0]]

      tf.global_variables_initializer().run()
      self.assertAllClose(expected_out_padding, out_padding.eval().tolist())

  def testConvPoolComputeOutPaddingUnevenStride(self):
    with self.session(use_gpu=True):
      in_padding = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
                               dtype=tf.float32)
      out_padding = layers._ComputeConvOutputPadding(in_padding, 3, 3)
      expected_out_padding = [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]

      tf.global_variables_initializer().run()
      self.assertAllClose(expected_out_padding, out_padding.eval().tolist())

  def _checkConvLayerShapes(self,
                            input_shape,
                            filter_shape,
                            filter_stride,
                            dilation_rate=None,
                            depth_multiplier=None,
                            params_builder=layers.Conv2DLayer.Params):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = params_builder()
      params.name = 'conv'
      params.filter_shape = filter_shape
      params.filter_stride = filter_stride
      if dilation_rate:
        params.dilation_rate = dilation_rate
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      if depth_multiplier is not None:
        params.depth_multiplier = depth_multiplier
      conv_layer = params.cls(params)

      inp = tf.random_uniform(input_shape)
      inp_pad = tf.floor(0.5 + tf.random_uniform(input_shape[:2]))
      out, out_pad = conv_layer.FPropDefaultTheta(inp, inp_pad)

    with self.session(use_gpu=True, graph=g) as sess:
      tf.global_variables_initializer().run()
      out, out_pad = sess.run([out, out_pad])
      print(out.shape, out_pad.shape)
      # We expect conv_layer.OutShape can compute the actual output shape.
      self.assertAllEqual(out.shape, conv_layer.OutShape(inp.shape.as_list()))
      # We expect out_pad.shape matches the 1st 2 dimensions of out.
      self.assertAllEqual(out.shape[:2], out_pad.shape)

  def testConv2DLayerOutputShapes(self):
    self._checkConvLayerShapes([2, 4, 4, 3], [3, 3, 3, 32], [1, 1])
    self._checkConvLayerShapes([2, 4, 4, 3], [3, 3, 3, 32], [2, 2])
    self._checkConvLayerShapes([2, 10, 4, 3], [3, 3, 3, 32], [3, 3])

    self._checkConvLayerShapes([2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
                               dilation_rate=[2, 2])
    self._checkConvLayerShapes([2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
                               dilation_rate=[3, 3])

  def testDepthwiseConv2DLayerOutputShapes(self):
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [1, 1],
        params_builder=layers.DepthwiseConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [2, 2],
        params_builder=layers.DepthwiseConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [3, 3],
        params_builder=layers.DepthwiseConv2DLayer.Params)

    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
        dilation_rate=[2, 2],
        params_builder=layers.DepthwiseConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
        dilation_rate=[3, 3],
        params_builder=layers.DepthwiseConv2DLayer.Params)

  def testSeparableConv2DLayerOutputShapes(self):
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [1, 1],
        params_builder=layers.SeparableConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [2, 2],
        params_builder=layers.SeparableConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [3, 3],
        params_builder=layers.SeparableConv2DLayer.Params)
    # Dilations.
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
        dilation_rate=[2, 2],
        params_builder=layers.SeparableConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
        dilation_rate=[3, 3],
        params_builder=layers.SeparableConv2DLayer.Params)
    # Depth multiplier.
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [1, 1],
        params_builder=layers.SeparableConv2DLayer.Params,
        depth_multiplier=2)
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [2, 2],
        params_builder=layers.SeparableConv2DLayer.Params,
        depth_multiplier=6)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [3, 3],
        params_builder=layers.SeparableConv2DLayer.Params,
        depth_multiplier=12)

  def _evalConvLayerFProp(self,
                          params_builder=layers.Conv2DLayer.Params,
                          batch_norm=True,
                          weight_norm=False,
                          bias=False,
                          activation='RELU',
                          conv_last=False,
                          strides=(2, 2),
                          dilation_rate=(1, 1),
                          bn_fold_weights=False,
                          is_eval=False,
                          quantized=False):
    self._ClearCachedSession()
    tf.reset_default_graph()
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = params_builder()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.filter_stride = strides
      params.dilation_rate = dilation_rate
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.conv_last = conv_last
      params.batch_norm = batch_norm
      params.bn_fold_weights = bn_fold_weights
      params.weight_norm = weight_norm
      params.bias = bias
      params.activation = activation
      params.is_eval = is_eval

      if quantized:
        params.qdomain.default = quant_utils.PassiveAsymQDomain.Params()

      conv_layer = params.cls(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)

      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      output2, _ = conv_layer.FPropDefaultTheta(inputs1)
      tf.global_variables_initializer().run()
      v1, v2 = sess.run([output1, output2])
      self.assertAllClose(v1, v2)
      return v1

  def testConv2DLayerFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.36669245,  0.91488785],
          [ 0.07532132,  0.        ]],
         [[ 0.34952009,  0.        ],
          [ 1.91783941,  0.        ]]],
        [[[ 0.28304493,  0.        ],
          [ 0.        ,  0.        ]],
         [[ 0.        ,  0.86575812],
          [ 0.        ,  1.60203481]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp()
    print('actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testDepthwiseConv2DLayerFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.93514717,  0.35602099,  0.        ,  0.51261222,  0.        ,
            1.4310323 ],
          [ 0.        ,  0.        ,  0.49176404,  0.        ,  1.01494753,
            0.51337928]],
         [[ 0.62087697,  0.34572476,  0.        ,  0.19352221,  0.47142431,
            0.        ],
          [ 0.81119895,  1.00890303,  0.90471351,  0.        ,  1.22736526,
            0.        ]]],
        [[[ 0.        ,  0.        ,  0.48927376,  0.        ,  0.74019426,
            0.        ],
          [ 0.        ,  0.        ,  1.49952257,  0.        ,  0.        ,
            0.        ]],
         [[ 0.29156703,  0.        ,  0.        ,  1.14509106,  0.        ,
            0.74238932],
          [ 0.91312039,  1.39783907,  0.        ,  1.47650909,  0.        ,
            0.37969294]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        params_builder=layers.DepthwiseConv2DLayer.Params)
    print('actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testSeparableConv2DLayerFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 =[
        [[[ 0.39866772,  0.        ],
          [ 1.36471784,  0.        ]],
         [[ 0.        ,  0.        ],
          [ 0.        ,  0.        ]]],
        [[[ 1.15356529,  0.1036691 ],
          [ 0.12865055,  0.61244327]],
         [[ 0.03609803,  1.81620765],
          [ 0.        ,  0.23052886]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        params_builder=layers.SeparableConv2DLayer.Params)
    print('actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testConv2DLayerWithDilationFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.        ,  0.48857123],
          [ 1.07320869,  0.        ],
          [ 0.        ,  0.1550007 ],
          [ 0.        ,  1.59097648]],
         [[ 0.        ,  0.        ],
          [ 0.20024362,  0.        ],
          [ 0.        ,  0.64265913],
          [ 1.52903616,  0.        ]],
         [[ 0.099805  ,  0.        ],
          [ 0.        ,  0.61720949],
          [ 1.31608474,  0.        ],
          [ 0.        ,  0.        ]],
         [[ 0.0175612 ,  0.        ],
          [ 0.        ,  0.17234094],
          [ 0.21719536,  0.        ],
          [ 1.68514931,  0.        ]]],
        [[[ 1.45240796,  0.        ],
          [ 0.        ,  0.        ],
          [ 0.72675145,  1.971596  ],
          [ 0.        ,  0.01062769]],
         [[ 0.        ,  1.70299017],
          [ 1.36936104,  1.29897082],
          [ 1.40132439,  1.74345171],
          [ 0.02585058,  0.29061913]],
         [[ 0.        ,  0.        ],
          [ 0.32962656,  0.05025356],
          [ 0.        ,  0.        ],
          [ 0.        ,  0.        ]],
         [[ 0.97244394,  0.        ],
          [ 0.23401484,  0.5722279 ],
          [ 0.        ,  0.40940297],
          [ 0.        ,  0.52711827]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(strides=[1, 1], dilation_rate=[2, 2])
    print('testConvLayerWithDilationFProp actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testSeparableConv2DLayerWithDilationFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.21535617,  0.86965537],
          [ 2.11499524,  1.2463783 ],
          [ 0.        ,  0.39275286],
          [ 0.        ,  0.        ]],
         [[ 1.12706482,  1.37450278],
          [ 0.        ,  0.        ],
          [ 0.        ,  0.        ],
          [ 1.2390101 ,  0.22932449]],
         [[ 0.        ,  0.        ],
          [ 0.15051894,  1.32616639],
          [ 0.        ,  0.        ],
          [ 0.72912866,  0.47753802]],
         [[ 0.91655868,  0.        ],
          [ 0.88526261,  0.26690534],
          [ 0.        ,  0.26084688],
          [ 0.42923039,  0.        ]]],
        [[[ 0.82440329,  0.        ],
          [ 0.49015623,  0.52662987],
          [ 0.        ,  0.        ],
          [ 0.35344127,  0.        ]],
         [[ 0.        ,  0.        ],
          [ 0.        ,  0.        ],
          [ 0.43848675,  0.        ],
          [ 0.        ,  1.21124518]],
         [[ 1.1026746 ,  1.39578998],
          [ 0.        ,  0.        ],
          [ 0.34652925,  0.        ],
          [ 0.        ,  1.26868236]],
         [[ 0.91519427,  0.09030763],
          [ 0.        ,  0.59271163],
          [ 0.        ,  0.54207176],
          [ 0.        ,  0.        ]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        strides=[1, 1],
        dilation_rate=[2, 2],
        params_builder=layers.SeparableConv2DLayer.Params)
    print('testConvLayerWithDilationFProp actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testConv2DLayerConvFirstVsLastFProp(self):
    """Compare results of conv first vs. last."""
    # ... with batch_norm and activation disabled.
    self.assertAllClose(
        self._evalConvLayerFProp(
            batch_norm=False, activation='NONE', conv_last=False),
        self._evalConvLayerFProp(
            batch_norm=False, activation='NONE', conv_last=True))

  def testConv2DLayerFPropConvLast(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.22165056,  0.20731729],
          [ 0.09577402, -0.15359652]],
         [[ 0.07151584,  0.03027298],
          [ 0.05370769,  0.0143405 ]]],
        [[[-0.08854639,  0.06143938],
          [-0.37708873,  0.00889082]],
         [[-0.58154356,  0.30798748],
          [-0.37575331,  0.54729235]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(conv_last=True)
    print(['ConvLast actual = ', np.array_repr(actual)])
    self.assertAllClose(expected_output1, actual)

  def testConv2DLayerConvWithBias(self):
    """Compare results with bias vs. with neither batch_norm nor bias."""
    # Results should match since bias is initialized to be 0.
    self.assertAllClose(
        self._evalConvLayerFProp(batch_norm=False, bias=False),
        self._evalConvLayerFProp(batch_norm=False, bias=True))

  def testConv2DLayerWeightNormFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [
        [[[ 0.37172362, 0.92405349],
          [ 0.07635488, 0.]],
         [[ 0.35431579, 0.],
          [ 1.94415355, 0.]]],
        [[[ 0.28692839, 0.],
          [ 0.        , 0.]],
         [[ 0.        , 0.87443149],
          [ 0.        , 1.61808443]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(weight_norm=True)
    print('actual1 = ', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testDepthwiseConv2DLayerWeightNormFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [
        [[[ 0.97023201,  0.37429881,  0.        ,  0.53157473,  0.        ,
            1.60764372],
          [ 0.        ,  0.        ,  0.50401598,  0.        ,  1.07683432,
            0.57673818]],
         [[ 0.644171  ,  0.36347377,  0.        ,  0.20068097,  0.50016963,
            0.        ],
          [ 0.8416335 ,  1.06069875,  0.92725372,  0.        ,  1.30220449,
            0.        ]]],
        [[[ 0.        ,  0.        ,  0.50146359,  0.        ,  0.78532791,
            0.        ],
          [ 0.        ,  0.        ,  1.53688192,  0.        ,  0.        ,
            0.        ]],
         [[ 0.302506  ,  0.        ,  0.        ,  1.18745029,  0.        ,
            0.83401161],
          [ 0.94737887,  1.46960247,  0.        ,  1.53112805,  0.        ,
            0.42655289]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        weight_norm=True, params_builder=layers.DepthwiseConv2DLayer.Params)
    print('actual1 = ', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testSeparableConv2DLayerWeightNormFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [
        [[[ 0.41837293,  0.        ],
          [ 1.39592457,  0.        ]],
         [[ 0.        ,  0.        ],
          [ 0.        ,  0.        ]]],
        [[[ 1.20513153,  0.11938372],
          [ 0.1284119 ,  0.6927582 ]],
         [[ 0.0227453 ,  2.05591369],
          [ 0.        ,  0.26530063]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        weight_norm=True, params_builder=layers.SeparableConv2DLayer.Params)
    print('actual1 = ', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testConv2DLayerFoldedBatchNormFProp(self):
    actual_unfolded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=False)
    actual_folded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=True)
    print('testConvLayerFoldedBatchNormFProp folded = ',
          np.array_repr(actual_folded))
    print('testConvLayerFoldedBatchNormFProp unfolded = ',
          np.array_repr(actual_unfolded))
    self.assertAllClose(actual_folded, actual_unfolded)

  def testDepthwiseConv2DLayerFoldedBatchNormFProp(self):
    actual_unfolded = self._evalConvLayerFProp(
        batch_norm=True,
        bn_fold_weights=False,
        params_builder=layers.DepthwiseConv2DLayer.Params)
    actual_folded = self._evalConvLayerFProp(
        batch_norm=True,
        bn_fold_weights=True,
        params_builder=layers.DepthwiseConv2DLayer.Params)
    print('testDepthwiseConvLayerFoldedBatchNormFProp folded = ',
          np.array_repr(actual_folded))
    print('testDepthwiseConvLayerFoldedBatchNormFProp unfolded = ',
          np.array_repr(actual_unfolded))
    self.assertAllClose(actual_folded, actual_unfolded)

  def testSeparableConv2DLayerFoldedBatchNormFProp(self):
    actual_unfolded = self._evalConvLayerFProp(
        batch_norm=True,
        bn_fold_weights=False,
        params_builder=layers.SeparableConv2DLayer.Params)
    actual_folded = self._evalConvLayerFProp(
        batch_norm=True,
        bn_fold_weights=True,
        params_builder=layers.SeparableConv2DLayer.Params)
    print('testSeparableConvLayerFoldedBatchNormFProp folded = ',
          np.array_repr(actual_folded))
    print('testSeparableConvLayerFoldedBatchNormFProp unfolded = ',
          np.array_repr(actual_unfolded))
    self.assertAllClose(actual_folded, actual_unfolded)

  def testConvLayerFoldedBatchNormFPropEval(self):
    actual_unfolded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=False, is_eval=True)
    actual_folded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=True, is_eval=True)
    print('testConvLayerFoldedBatchNormFPropEval folded = ',
          np.array_repr(actual_folded))
    print('testConvLayerFoldedBatchNormFPropEval unfolded = ',
          np.array_repr(actual_unfolded))
    self.assertAllClose(actual_folded, actual_unfolded)

  def testConv2DLayerNoPadding(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(24332)
      p = layers.Conv2DLayerNoPadding.Params().Set(
          name='test', filter_shape=(3, 3, 3, 5), filter_stride=(2, 2))
      l = p.cls(p)
      x = tf.random_normal(shape=[17, 64, 64, 3])
      y = l.FPropDefaultTheta(x)

    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      y_val = sess.run(y)

    self.assertEqual(y_val.shape, (17, 32, 32, 5))

  def testConvLayerFoldedBatchNormFPropQuantized(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [
       [[[ 0.36997819,  0.91361964],
         [ 0.07550576,  0.        ]],

        [[ 0.35487702,  0.        ],
         [ 1.92539668,  0.        ]]],
       [[[ 0.27937129,  0.        ],
         [ 0.        ,  0.        ]],

        [[ 0.        ,  0.86831617],
         [ 0.        ,  1.59317136]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace

    actual_folded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=True, quantized=True)
    print('testConvLayerFoldedBatchNormFPropQuantized folded = ',
          np.array_repr(actual_folded))
    self.assertAllClose(actual_folded, expected_output)

  def testCausalConvLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [2, 1, 3, 2]
      params.filter_stride = [1, 1]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      params.causal_convolution = True
      params.activation = 'NONE'
      params.batch_norm = False

      conv_layer = layers.ConvLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3, 3]), dtype=tf.float32)
      # Change the input for the last two steps.
      inputs2 = tf.concat([inputs1[:, :2, :, :], inputs1[:, 2:, :, :] + 0.5], 1)

      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      output2, _ = conv_layer.FPropDefaultTheta(inputs2, in_padding1)
      tf.global_variables_initializer().run()
      v1, v2 = sess.run([output1, output2])
      tf.logging.info('CausalConv output: %s', np.array_repr(v1))
      # pylint: disable=bad-whitespace,bad-continuation,line-too-long
      self.assertAllClose(v1, [
          [[[-0.01093466,  0.00369835],
            [ 0.03474921,  0.01418608],
            [ 0.01887876, -0.00763734]],
           [[-0.06922598, -0.04526342],
            [-0.02428233,  0.02042499],
            [-0.04504267, -0.01260209]],
           [[-0.14253227, -0.11353028],
            [-0.09067881,  0.03742362],
            [ 0.01281691,  0.00644186]],
           [[-0.06524619, -0.0555004 ],
            [-0.18850081, -0.05325979],
            [ 0.04960757,  0.05512709]]],
          [[[-0.01077277,  0.03013588],
            [ 0.00325067, -0.0223705 ],
            [-0.00895232,  0.03310337]],
           [[ 0.03113075, -0.02388876],
            [ 0.03238059,  0.00590346],
            [ 0.12839797, -0.02194144]],
           [[-0.09115655, -0.06798521],
            [-0.09801255, -0.01440183],
            [-0.04321899,  0.00340509]],
           [[-0.089603  , -0.07257183],
            [-0.04469771, -0.0389927 ],
            [-0.01747611,  0.00903451]]]
      ])  # pyformat: disable
      # pylint: enable=bad-whitespace,bad-continuation,line-too-long
      self.assertAllClose(v1[:, :2, :, :], v2[:, :2, :, :])
      with self.assertRaises(AssertionError):
        self.assertAllClose(v1[:, 2:, :, :], v2[:, 2:, :, :])

  def testConvLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False

      conv_layer = layers.ConvLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      loss = tf.reduce_sum(output1)

      all_vars = tf.trainable_variables()
      self.assertEqual(3, len(all_vars))

      grads = tf.gradients(loss, all_vars)
      tf.global_variables_initializer().run()
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-02, atol=1e-02)

  def testConvLayerFPropTanh(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.activation = 'TANH'
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False

      conv_layer = layers.ConvLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)

      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      tf.global_variables_initializer().run()

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_output1 = [
          [[[ 0.35109526,  0.72346997],
            [ 0.0751792 , -0.84315312]],
           [[ 0.33594984, -0.18976833],
            [ 0.95773894, -0.28015777]]],
          [[[ 0.27572086, -0.26577294],
            [-0.38503852, -0.88501388]],
           [[-0.92332661,  0.69921255],
            [-0.75103623,  0.9219743 ]]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      actual = output1.eval()
      print(['actual = ', actual])
      self.assertAllClose(expected_output1, actual)

  def testConvSetLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvSetLayer.Params()
      params.name = 'conv_set'
      params.filter_shapes = [[3, 3, 3, 32], [8, 5, 3, 64]]
      params.cnn_tpl.filter_stride = [2, 2]
      params.cnn_tpl.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.cnn_tpl.is_eval = False
      layers.ConvSetLayer(params)

  def _evalConvSetLayerFProp(self,
                             batch_norm=True,
                             bn_fold_weights=False,
                             weight_norm=False,
                             bias=False,
                             activation='RELU',
                             conv_last=False,
                             strides=(2, 2),
                             dilation_rate=(1, 1),
                             quantized=False,
                             dump_graphdef=False):
    self._ClearCachedSession()
    ops.reset_default_graph()
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvSetLayer.Params()
      params.name = 'conv_set'
      params.filter_shapes = [[2, 2, 6, 1], [3, 5, 6, 3]]
      params.cnn_tpl.filter_stride = strides
      params.cnn_tpl.dilation_rate = dilation_rate
      params.cnn_tpl.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.cnn_tpl.conv_last = conv_last
      params.cnn_tpl.batch_norm = batch_norm
      params.cnn_tpl.bn_fold_weights = bn_fold_weights
      params.cnn_tpl.weight_norm = weight_norm
      params.cnn_tpl.bias = bias
      params.cnn_tpl.activation = activation
      params.cnn_tpl.is_eval = False
      if quantized:
        params.qdomain.default = quant_utils.PassiveAsymQDomain.Params()

      conv_set_layer = layers.ConvSetLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 6]), dtype=tf.float32)

      output1, _ = conv_set_layer.FPropDefaultTheta(inputs1, in_padding1)
      tf.global_variables_initializer().run()

      if dump_graphdef:
        print('ConvSet GraphDef:', sess.graph.as_graph_def())
        assert False, 'Disable "dump_graphdef" before submit'

      return output1.eval()

  def testConvSetLayerFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace,bad-continuation
    expected_output1 = [
        [[[ 1.04307961,  0.        ,  1.27613628,  0.        ],
        [ 0.          ,  0.        ,  0.        ,  1.21081829 ]],
        [[ 0.         ,  0.18475296,  0.        ,  0.        ],
        [ 1.34087086  ,  2.2726357 ,  0.        ,  0.         ]]],
        [[[ 0.        ,  0.25231963,  0.        ,  0.       ],
        [ 1.13677704  ,  0.        ,  0.996117  ,  1.836285   ]],
        [[ 0.         ,  0.        ,  1.04101253,  0.        ],
        [ 0.12628449  ,  0.37599814,  0.3134549 ,  0.51208746 ]]]
    ]
    # pyformat: enable
    # pylint: enable=bad-whitespace,bad-continuation
    actual = self._evalConvSetLayerFProp()
    print(['actual = ', np.array_repr(actual)])
    self.assertAllClose(expected_output1, actual)

  def testConvSetLayerFPropQuantized(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace,bad-continuation
    expected_output1 = [
        [[[ 1.04016984,  0.        ,  1.28103447,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  1.20986581]],
         [[ 0.        ,  0.18681753,  0.        ,  0.        ],
          [ 1.35328221,  2.26849842,  0.        ,  0.        ]]],
        [[[ 0.        ,  0.24909003,  0.        ,  0.        ],
          [ 1.14100266,  0.        ,  0.98746401,  1.83259094]],
         [[ 0.        ,  0.        ,  1.04084051,  0.        ],
          [ 0.12736773,  0.38253111,  0.32025862,  0.5159722 ]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace,bad-continuation
    actual = self._evalConvSetLayerFProp(bn_fold_weights=True, quantized=True)
    # Note that we don't have many ways to verify in a unit test that the
    # quant nodes were added properly; however, if their placement changes,
    # it will very likely perturb the golden values above. If digging deeper,
    # add 'dump_graphdef=True' to the above call and inspect the graphdef:
    # There should be one layer of fake_quant* nodes before the ConcatV2.
    print('actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  # TODO(yonghui): more test for convolution layer


class PoolingLayerTest(test_utils.TestCase):

  def testPoolLayerFProp(self):
    with self.session(use_gpu=True):
      params = layers.PoolingLayer.Params()
      params.name = 'pool'
      params.window_shape = [3, 3]
      params.window_stride = [1, 2]
      params.is_eval = False

      pool_layer = layers.PoolingLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.arange(96, dtype='float32').reshape([2, 4, 4, 3]),
          dtype=tf.float32)

      output1, _ = pool_layer.FPropDefaultTheta(inputs1, in_padding1)
      tf.global_variables_initializer().run()
      print([np.array_repr(output1.eval())])
      # pyformat: disable
      expected_output1 = [
          [[[18., 19., 20.],
            [21., 22., 23.]],
           [[30., 31., 32.],
            [33., 34., 35.]],
           [[42., 43., 44.],
            [45., 46., 47.]],
           [[42., 43., 44.],
            [45., 46., 47.]]],
          [[[66., 67., 68.],
            [69., 70., 71.]],
           [[78., 79., 80.],
            [81., 82., 83.]],
           [[90., 91., 92.],
            [93., 94., 95.]],
           [[90., 91., 92.],
            [93., 94., 95.]]]]
      # pyformat: enable
      self.assertAllClose(expected_output1, output1.eval())

  def testPoolLayerMoreShapes(self):
    with self.session(use_gpu=True):
      for window_shape, window_stride in [
          [[3, 3], [1, 2]],
          [[2, 2], [1, 2]],
          [[3, 4], [1, 3]],
      ]:
        params = layers.PoolingLayer.Params()
        params.name = 'pool'
        params.window_shape = window_shape
        params.window_stride = window_stride
        params.is_eval = False

        pool_layer = layers.PoolingLayer(params)
        in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
        inputs1 = tf.constant(
            np.arange(96, dtype='float32').reshape([2, 4, 4, 3]),
            dtype=tf.float32)

        output1, _ = pool_layer.FPropDefaultTheta(inputs1, in_padding1)

        output2 = tf.nn.max_pool(inputs1, [1] + params.window_shape + [1],
                                 [1] + params.window_stride + [1], 'SAME')

        predicted_out_shape = pool_layer.OutShape(inputs1.shape.as_list())

        tf.global_variables_initializer().run()
        output1_v = output1.eval()
        self.assertAllClose(output2.eval(), output1_v)
        self.assertAllClose(predicted_out_shape, output1_v.shape)


class ProjectionLayerTest(test_utils.TestCase):

  def testProjectionLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.input_dim = 2
      params.output_dim = 3
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      layers.ProjectionLayer(params)
      proj_vars = tf.get_collection('ProjectionLayer_vars')
      proj_var_names = [x.name for x in proj_vars]
      self.assertEqual(['proj/w/var:0'], proj_var_names)
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'proj/beta/var:0', 'proj/gamma/var:0', 'proj/moving_mean/var:0',
          'proj/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)

  def _evalProjectionLayer(self,
                           reshape_to_2d=False,
                           batch_norm=True,
                           weight_norm=False,
                           activation='RELU',
                           affine_last=False,
                           input_dim=3,
                           output_dim=2,
                           quantized=False,
                           has_bias=False,
                           bn_fold_weights=None,
                           expect_bn_fold_weights=None,
                           is_eval=False,
                           layer_callback=None):
    self._ClearCachedSession()
    tf.reset_default_graph()
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.input_dim = input_dim
      params.output_dim = output_dim
      params.has_bias = has_bias
      if has_bias:
        params.bias_init = 5.0
      params.activation = activation
      params.batch_norm = batch_norm
      params.weight_norm = weight_norm
      params.affine_last = affine_last
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.bn_fold_weights = bn_fold_weights
      if quantized:
        cc_schedule = quant_utils.FakeQuantizationSchedule.Params().Set(
            clip_end_step=1, quant_start_step=1)
        qdomain_default = quant_utils.SymmetricScheduledClipQDomain.Params(
        ).Set(cc_schedule=cc_schedule.Copy())
        params.qdomain.default = qdomain_default.Copy()
      params.is_eval = is_eval

      in_padding = tf.zeros([2, 4, 1], dtype=tf.float32)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float32)
      if reshape_to_2d:
        in_padding = tf.reshape(in_padding, [-1, 1])
        inputs = tf.reshape(inputs, [-1, 3])

      proj_layer = layers.ProjectionLayer(params)
      if layer_callback:
        layer_callback(proj_layer)
      if expect_bn_fold_weights is not None:
        self.assertEqual(expect_bn_fold_weights, proj_layer._is_bn_folded)

      output = proj_layer.FPropDefaultTheta(inputs, in_padding)
      tf.global_variables_initializer().run()
      if quantized:
        # Put it in the fully quantized range.
        sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 5))
      return output.eval()

  def testProjectionLayerFProp(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[ 0.        ,  0.33779466],
         [ 0.4527415 ,  0.99911398],
         [ 0.44320837,  0.        ],
         [ 0.        ,  0.04557215]],
        [[ 0.69273949,  0.        ],
         [ 0.30908319,  0.        ],
         [ 0.        ,  0.        ],
         [ 0.        ,  1.54578114]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    for reshape_to_2d in (False, True):
      actual = self._evalProjectionLayer(
          reshape_to_2d=reshape_to_2d, expect_bn_fold_weights=False)
      if reshape_to_2d:
        expected_output = np.reshape(np.array(expected_output), (-1, 2))
      tf.logging.info('expected = %s', expected_output)
      tf.logging.info('actual = %s', np.array_repr(actual))
      self.assertAllClose(expected_output, actual)

  def testProjectionLayerFPropWithBias(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[ 4.98987579,  5.03493643],
         [ 5.01192808,  5.0917592 ],
         [ 5.01156807,  4.99741936],
         [ 4.96849394,  5.00982761]],
        [[ 5.02098131,  4.98014927],
         [ 5.00650883,  4.87676954],
         [ 4.98995209,  4.91770315],
         [ 4.95948696,  5.138731  ]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # Tested without batch_norm because batch_norm will mostly cancel out the
    # affect of bias.
    actual = self._evalProjectionLayer(
        has_bias=True,
        batch_norm=False,
        expect_bn_fold_weights=False,
        activation='RELU6')
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testProjectionLayerExplicitFolding(self):
    unfolded = self._evalProjectionLayer(
        bn_fold_weights=False, expect_bn_fold_weights=False)
    folded = self._evalProjectionLayer(
        bn_fold_weights=True, expect_bn_fold_weights=True)
    tf.logging.info('unfolded = %s', np.array_repr(unfolded))
    tf.logging.info('folded = %s', np.array_repr(folded))
    self.assertAllClose(folded, unfolded)

  def testProjectionLayerExplicitFoldingEval(self):
    unfolded = self._evalProjectionLayer(
        bn_fold_weights=False, expect_bn_fold_weights=False, is_eval=True)
    folded = self._evalProjectionLayer(
        bn_fold_weights=True, expect_bn_fold_weights=True, is_eval=True)
    tf.logging.info('unfolded = %s', np.array_repr(unfolded))
    tf.logging.info('folded = %s', np.array_repr(folded))
    self.assertAllClose(folded, unfolded)

  def testProjectionLayerExplicitFoldingNoBatchNorm(self):
    unfolded = self._evalProjectionLayer(
        batch_norm=False, bn_fold_weights=False, expect_bn_fold_weights=False)
    # Note that weight folding will report as disabled because batch norm is
    # disabled.
    folded = self._evalProjectionLayer(
        batch_norm=False, bn_fold_weights=True, expect_bn_fold_weights=False)
    tf.logging.info('unfolded = %s', np.array_repr(unfolded))
    tf.logging.info('folded = %s', np.array_repr(folded))
    self.assertAllClose(folded, unfolded)

  def testProjectionLayerExplicitFoldingWithWeightNorm(self):
    unfolded = self._evalProjectionLayer(
        weight_norm=True, bn_fold_weights=False, expect_bn_fold_weights=False)
    folded = self._evalProjectionLayer(
        weight_norm=True, bn_fold_weights=True, expect_bn_fold_weights=True)
    tf.logging.info('unfolded = %s', np.array_repr(unfolded))
    tf.logging.info('folded = %s', np.array_repr(folded))
    self.assertAllClose(folded, unfolded)

  def testProjectionLayerWeightNorm(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[ 0.        ,  0.36285588],
         [ 0.82909501,  1.07323885],
         [ 0.81163716,  0.        ],
         [ 0.        ,  0.04895319]],
        [[ 1.26859784,  0.        ],
         [ 0.56601691,  0.        ],
         [ 0.        ,  0.        ],
         [ 0.        ,  1.66046333]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    for reshape_to_2d in (False, True):
      actual = self._evalProjectionLayer(
          reshape_to_2d=reshape_to_2d, weight_norm=True)
      if reshape_to_2d:
        expected_output = np.reshape(np.array(expected_output), (-1, 2))
      tf.logging.info('expected = %s', expected_output)
      tf.logging.info('actual = %s', np.array_repr(actual))
      self.assertAllClose(expected_output, actual)

  def testProjectionLayerAffineFirstVsLastFProp(self):
    """Compare results of affine first vs. last."""
    # ... with batch_norm and activation disabled.
    self.assertAllClose(
        self._evalProjectionLayer(
            batch_norm=False, activation='NONE', affine_last=False),
        self._evalProjectionLayer(
            batch_norm=False, activation='NONE', affine_last=True))

  def testProjectionLayerAffineLastFProp(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output1 = [
        [[ 0.        ,  0.        ],
         [ 0.03410175,  0.04741348],
         [ 0.02665393, -0.02072855],
         [-0.01116518, -0.06280501]],
        [[ 0.04615254, -0.03589247],
         [-0.00376316, -0.0464084 ],
         [-0.01111402, -0.13706152],
         [-0.02596203,  0.16340451]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalProjectionLayer(affine_last=True)
    print(['actual = ', np.array_repr(actual)])
    self.assertAllClose(expected_output1, actual)

  def testProjectionLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.dtype = tf.float64
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.is_eval = False

      proj_layer = layers.ProjectionLayer(params)
      in_padding1 = tf.zeros([2, 4, 1], dtype=tf.float64)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float64)
      output1 = proj_layer.FPropDefaultTheta(inputs1, in_padding1)
      loss = tf.reduce_sum(output1)

      all_vars = tf.trainable_variables()
      self.assertEqual(3, len(all_vars))

      grads = tf.gradients(loss, all_vars)
      tf.global_variables_initializer().run()
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v, 1e-6)
          for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-06, atol=1e-06)

  def testProjectionLayerFPropQuantizedWithUnfusedActivation(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[-0.1328125,  0.3125   ],
         [ 0.421875 ,  0.734375 ],
         [ 0.421875 , -0.109375 ],
         [-0.6015625,  0.0078125]],
        [[ 0.6015625, -0.3046875],
         [ 0.3046875, -0.7578125],
         [-0.125    , -0.7578125],
         [-0.734375 ,  0.7578125]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    def CheckLayer(proj_layer):
      # Should not error because this qtensor is defined.
      proj_layer.QTensor('activation', tf.convert_to_tensor(0.))
      # The intermediate tensor should be defined.
      proj_layer.QTensor('affine_matmul', tf.convert_to_tensor(0.))

    # When quantization enabled, batchnorm folding should auto enable.
    # TANH is unfused.
    actual = self._evalProjectionLayer(
        activation='TANH',
        quantized=True,
        expect_bn_fold_weights=True,
        layer_callback=CheckLayer)
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testProjectionLayerFPropQuantizedWithFusedActivation(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[ 0.       ,  0.3203125],
         [ 0.453125 ,  0.9375   ],
         [ 0.4453125,  0.       ],
         [ 0.       ,  0.0078125]],
        [[ 0.6953125,  0.       ],
         [ 0.3125   ,  0.       ],
         [ 0.       ,  0.       ],
         [ 0.       ,  0.9921875]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    def CheckLayer(proj_layer):
      # Should not error because this qtensor is defined.
      proj_layer.QTensor('activation', tf.convert_to_tensor(0.))
      with self.assertRaises(AssertionError):
        # The intermediate tensor should *not* be quantized.
        proj_layer.QTensor('affine_matmul', tf.convert_to_tensor(0.))

    # When quantization enabled, batchnorm folding should auto enable.
    # RELU6 is fused.
    actual = self._evalProjectionLayer(
        activation='RELU6',
        quantized=True,
        expect_bn_fold_weights=True,
        layer_callback=CheckLayer)
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testProjectionLayerFPropQuantizedOnlyMatmul(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[-0.0078125,  0.0390625],
         [ 0.0078125,  0.09375  ],
         [ 0.0078125,  0.       ],
         [-0.03125  ,  0.015625 ]],
        [[ 0.015625 , -0.015625 ],
         [ 0.0078125, -0.125    ],
         [-0.0078125, -0.078125 ],
         [-0.0390625,  0.1484375]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    def CheckLayer(proj_layer):
      # Should not error because this qtensor is defined.
      proj_layer.QTensor('affine_matmul', tf.convert_to_tensor(0.))

    actual = self._evalProjectionLayer(
        activation='NONE',
        quantized=True,
        batch_norm=False,
        expect_bn_fold_weights=False,
        layer_callback=CheckLayer)
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testProjectionLayerFPropQuantizedOnlyMatmulBias(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    # Saturated because of the out of range bias.
    expected_output = [[[0.9921875, 0.9921875], [0.9921875, 0.9921875],
                        [0.9921875, 0.9921875], [0.9921875, 0.9921875]],
                       [[0.9921875, 0.9921875], [0.9921875, 0.9921875],
                        [0.9921875, 0.9921875], [0.9921875, 0.9921875]]]

    # pyformat: enable
    # pylint: enable=bad-whitespace
    def CheckLayer(proj_layer):
      # Should not error because this qtensor is defined.
      proj_layer.QTensor('affine_matmul', tf.convert_to_tensor(0.))

    actual = self._evalProjectionLayer(
        activation='NONE',
        quantized=True,
        has_bias=True,
        batch_norm=False,
        expect_bn_fold_weights=False,
        layer_callback=CheckLayer)
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testFCLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.FCLayer.Params()
      params.name = 'fc'
      params.input_dim = 2
      params.output_dim = 3
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      layers.FCLayer(params)
      proj_vars = tf.get_collection('FCLayer_vars')
      proj_var_names = [x.name for x in proj_vars]
      expected_var_names = ['fc/w/var:0', 'fc/b/var:0']
      self.assertEqual(expected_var_names, proj_var_names)

  def testFCLayerFProp(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.FCLayer.Params()
      params.name = 'fc'
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      proj_layer = layers.FCLayer(params)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float32)

      output = proj_layer.FPropDefaultTheta(inputs)
      tf.global_variables_initializer().run()

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ 0.        ,  0.04883499],
           [ 0.17094055,  0.        ],
           [ 0.09287541,  0.        ],
           [ 0.        ,  0.19471419]],
          [[ 0.15290432,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.10548697],
           [ 0.        ,  0.22610095]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      actual = output.eval()
      print(['actual = ', np.array_repr(actual)])
      self.assertAllClose(expected_output, actual)

  def testFCLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.FCLayer.Params()
      params.name = 'fc'
      params.dtype = tf.float64
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.01)

      proj_layer = layers.FCLayer(params)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float64)
      output = proj_layer.FPropDefaultTheta(inputs)
      loss = tf.reduce_sum(output)

      all_vars = tf.trainable_variables()
      self.assertEqual(2, len(all_vars))

      grads = tf.gradients(loss, all_vars)
      tf.global_variables_initializer().run()
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v, 1e-6)
          for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-06, atol=1e-06)


class EmbeddingLayerTest(test_utils.TestCase):

  def testEmbeddingLayer(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      params = layers.EmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.max_num_shards = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = layers.EmbeddingLayer(params)
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      tf.global_variables_initializer().run()
      test_utils.CompareToGoldenSingleFloat(self, 0.234941, embs_sum.eval())

  def testCheckedIds(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      params = layers.EmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 16
      params.embedding_dim = 128
      params.max_num_shards = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = layers.EmbeddingLayer(params)

      neg_ids = tf.constant([[-1]])
      neg_embs = emb_layer.EmbLookupDefaultTheta(neg_ids)
      oov_ids = tf.constant([[params.vocab_size]])
      oov_embs = emb_layer.EmbLookupDefaultTheta(oov_ids)
      tf.global_variables_initializer().run()

      with self.assertRaises(tf.errors.InvalidArgumentError):
        neg_embs.eval()
      with self.assertRaises(tf.errors.InvalidArgumentError):
        oov_embs.eval()

  def testEmbeddingLayerScaling(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      params = layers.EmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.max_num_shards = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.scale_sqrt_depth = True
      emb_layer = layers.EmbeddingLayer(params)
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      tf.global_variables_initializer().run()
      self.assertAllClose(0.23494134843349457 * params.embedding_dim**0.5,
                          sess.run(embs_sum))

  def testEmbeddingLayerWithVN(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      params = layers.EmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.max_num_shards = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.01, seed=398847392)
      params.vn.global_vn = True
      params.vn.per_step_vn = False
      params.vn.scale = 0.5
      params.vn.seed = 398847392
      emb_layer = layers.EmbeddingLayer(params)
      self.assertEqual(len(emb_layer.vars.Flatten()), 4)
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      tf.global_variables_initializer().run()
      test_utils.CompareToGoldenSingleFloat(self, -6.807296, embs_sum.eval())

  def _testSimpleEmbeddingLayer(self, use_matmul, use_3d_weight_tensor,
                                fprop_mode):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(398847392)
      params = layers.SimpleEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 8000
      params.embedding_dim = 128
      params.use_matmul = use_matmul
      params.fprop_mode = fprop_mode
      params.use_3d_weight_tensor = use_3d_weight_tensor
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False

      emb_layer = layers.SimpleEmbeddingLayer(params)
      expected_fprop_mode = fprop_mode
      if expected_fprop_mode is None:
        expected_fprop_mode = 'matmul' if use_matmul else 'gather'
      self.assertEqual(emb_layer._fprop_mode, expected_fprop_mode)

      emb_matrix = emb_layer.vars.wm
      ids = tf.constant([[89], [100]])
      outputs = emb_layer.EmbLookupDefaultTheta(ids)
      fast_outputs = emb_layer.EmbLookupDefaultThetaOnCpu(ids)

    with self.session(use_gpu=True, graph=g) as sess:
      tf.global_variables_initializer().run()
      emb_matrix_val, ids_val, outputs_val, fast_outputs_val = sess.run(
          [emb_matrix, ids, outputs, fast_outputs])
      self.assertEqual(emb_matrix_val.shape, (8000, 128))
      self.assertEqual(ids_val.shape, (2, 1))
      self.assertEqual(outputs_val.shape, (2, 1, 128))
      self.assertAllClose(emb_matrix_val[89, :], outputs_val[0, 0, :])
      self.assertAllClose(emb_matrix_val[100, :], outputs_val[1, 0, :])

      self.assertEqual(fast_outputs_val.shape, (2, 1, 128))
      self.assertAllClose(emb_matrix_val[89, :], fast_outputs_val[0, 0, :])
      self.assertAllClose(emb_matrix_val[100, :], fast_outputs_val[1, 0, :])

  def testSimpleEmbeddingLayerForLoop(self):
    self._testSimpleEmbeddingLayer(False, True, None)

  def testSimpleEmbeddingLayerForLoop2D(self):
    self._testSimpleEmbeddingLayer(False, False, None)

  def testSimpleEmbeddingLayerMatmul(self):
    self._testSimpleEmbeddingLayer(True, False, None)

  def testSimpleEmbeddingLayerGather(self):
    self._testSimpleEmbeddingLayer(False, False, 'gather')

  def testSimpleEmbeddingLayerMasked(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(398847392)
      params = layers.SimpleEmbeddingLayer.Params()
      params.name = 'emd'
      params.dtype = tf.float32
      params.vocab_size = 10
      params.embedding_dim = 5
      params.fprop_mode = 'gather'
      params.use_3d_weight_tensor = False
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.apply_pruning = True

      emb_layer = layers.SimpleEmbeddingLayer(params)
      emb_matrix = emb_layer.vars.wm
      ids = tf.constant([[1], [2]])
      outputs = emb_layer.EmbLookupDefaultTheta(ids)

      self.assertTrue('wm' in emb_layer.vars.wm.name)
      self.assertTrue('mask' in emb_layer.vars.mask.name)
      self.assertTrue('threshold' in emb_layer.vars.threshold.name)

      self.assertEqual(emb_layer.theta.wm.get_shape(), tf.TensorShape([10, 5]))
      self.assertEqual(emb_layer.theta.mask.get_shape(), tf.TensorShape([10,
                                                                         5]))
      self.assertEqual(emb_layer.theta.threshold.get_shape(),
                       tf.TensorShape([]))

      embedding_var_count = 1
      wts = tf.get_collection('SimpleEmbeddingLayer_vars')
      self.assertEqual(embedding_var_count, len(wts))

      embedding_mask_count = 1
      masks = tf.get_collection('masks')
      self.assertEqual(embedding_mask_count, len(masks))

      emebdding_threshold_count = 1
      threshold = tf.get_collection('thresholds')
      self.assertEqual(emebdding_threshold_count, len(threshold))

    with self.session(use_gpu=False, graph=g) as sess:
      tf.global_variables_initializer().run()
      emb_matrix_val, _, outputs_val = sess.run([emb_matrix, ids, outputs])

      self.assertAllClose(emb_matrix_val[1:3], outputs_val[:, 0, :])

  def _testSimpleEmbeddingLayerGrad(self, use_matmul, use_3d_weight_tensor):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(398847392)
      params = layers.SimpleEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 8000
      params.embedding_dim = 128
      params.use_matmul = use_matmul
      params.use_3d_weight_tensor = use_3d_weight_tensor
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = layers.SimpleEmbeddingLayer(params)
      ids = tf.constant([89, 100, 89, 89])
      embs = emb_layer.EmbLookupDefaultTheta(ids) * tf.constant([[0.1], [0.2],
                                                                 [0.3], [0.4]])
      embs_sum = tf.reduce_sum(embs)
      emb_weight = emb_layer.vars.wm
      emb_grad, = tf.gradients(ys=[embs_sum], xs=[emb_weight])
    with self.session(use_gpu=True, graph=g) as sess:
      tf.global_variables_initializer().run()
      emb_grad_val = sess.run(emb_grad)

    if not use_matmul:
      # tf.embedding_lookup's gradient is a sparse representation.
      # For testing, we convert it to a dense representation.
      o_grad_matrix = np.zeros((8000, 128))
      for i in range(emb_grad_val.indices.shape[0]):
        o_grad_matrix[emb_grad_val.indices[i], :] += emb_grad_val.values[i, :]
      emb_grad_val = o_grad_matrix

    expected_emb_grad = np.zeros(shape=(8000, 128))
    expected_emb_grad[89, :] = 0.8
    expected_emb_grad[100, :] = 0.2
    self.assertAllClose(expected_emb_grad, emb_grad_val)

  def testSimpleEmbeddingLayerGradForLoop(self):
    self._testSimpleEmbeddingLayerGrad(False, True)

  def testSimpleEmbeddingLayerGradForLoop2D(self):
    self._testSimpleEmbeddingLayerGrad(False, False)

  def testSimpleEmbeddingLayerGradMatmul(self):
    self._testSimpleEmbeddingLayerGrad(True, False)

  def testCompareEmbeddingLayers(self):
    classes = 8000
    dims = 128
    g = tf.Graph()
    with g.as_default():
      ids = tf.placeholder(tf.int32)

      def CreateSimple():
        tf.set_random_seed(398847392)
        p = layers.SimpleEmbeddingLayer.Params()
        p.name = 'emb'
        p.dtype = tf.float32
        p.vocab_size = classes
        p.embedding_dim = dims
        p.params_init = py_utils.WeightInit.Gaussian(0.01)
        p.vn.global_vn = False
        p.vn.per_step_vn = False
        return layers.SimpleEmbeddingLayer(p)

      simple = CreateSimple()
      simple_outs = simple.EmbLookupDefaultTheta(ids)
      simple_grad = tf.gradients(simple_outs, simple.vars.wm)[0]

      def CreateOriginal():
        tf.set_random_seed(398847392)
        p = layers.EmbeddingLayer.Params()
        p.name = 'emb'
        p.dtype = tf.float32
        p.vocab_size = classes
        p.embedding_dim = dims
        p.max_num_shards = 1
        p.params_init = py_utils.WeightInit.Gaussian(0.01)
        p.vn.global_vn = False
        p.vn.per_step_vn = False
        return layers.EmbeddingLayer(p)

      original = CreateOriginal()
      weight = tf.identity(simple.vars.wm)
      theta = py_utils.NestedMap()
      theta.wm = [weight]
      original_outs = original.EmbLookup(theta, ids)
      original_grad = tf.gradients(original_outs, weight)[0]

    ids_val = np.random.randint(0, high=classes, size=(4000,))
    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      s_outs, s_grad, o_outs, o_grad = sess.run(
          [simple_outs, simple_grad, original_outs, original_grad],
          feed_dict={ids: ids_val})
      self.assertAllClose(s_outs, o_outs)
      self.assertAllClose(s_grad, o_grad)

  def testPositionalEmbeddingLayer(self):
    with self.session(use_gpu=False) as sess:
      p = layers.PositionalEmbeddingLayer.Params()
      p.name = 'position_emb'
      p.min_timescale = 1
      p.max_timescale = 7
      p.embedding_dim = 4
      seq_length = 11

      pos_emb_layer = layers.PositionalEmbeddingLayer(p)
      position_embs = pos_emb_layer.FPropDefaultTheta(seq_length)
      actual_position_embs, = sess.run([position_embs])

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032],
          [-0.7568025 ,  0.54083425, -0.65364361,  0.84112918],
          [-0.95892417,  0.65507787,  0.28366217,  0.75556135],
          [-0.27941549,  0.75597537,  0.96017027,  0.65460002],
          [ 0.65698659,  0.84147096,  0.7539022 ,  0.54030228],
          [ 0.98935831,  0.90982294, -0.14550003,  0.41499668],
          [ 0.41211855,  0.9596386 , -0.91113025,  0.28123617],
          [-0.54402113,  0.98990309, -0.83907151,  0.14174587]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print('expected_position_embs:', expected_output)
      print('actual_position_embs:', actual_position_embs)
      self.assertAllClose(actual_position_embs, expected_output)

  def testPositionalEmbeddingLayerWithPosition(self):
    with self.session(use_gpu=False) as sess:
      p = layers.PositionalEmbeddingLayer.Params()
      p.name = 'position_emb'
      p.min_timescale = 1
      p.max_timescale = 7
      p.embedding_dim = 4
      pos_tensor = tf.constant(
          np.asarray([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3],
                      [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 0]]),
          dtype=tf.int32)

      pos_emb_layer = layers.PositionalEmbeddingLayer(p)
      position_embs = pos_emb_layer.FPropWithPosition(pos_emb_layer.theta,
                                                      pos_tensor)
      actual_position_embs, = sess.run([position_embs])

      # pylint: disable=bad-whitespace,bad-continuation
      # pyformat: disable
      expected_output = [
          [[ 0.        ,  0.        ,  1.        ,  1.       ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032],
          [-0.7568025 ,  0.54083425, -0.65364361,  0.84112918],
          [-0.95892417,  0.65507787,  0.28366217,  0.75556135],
          [-0.27941549,  0.75597537,  0.96017027,  0.65460002],
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032]],
          [[ 0.        ,  0.        ,  1.        ,  1.       ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032],
          [-0.7568025 ,  0.54083425, -0.65364361,  0.84112918],
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.        ,  0.        ,  1.        ,  1.        ]]
      ]
      # pyformat: enable
      # pylint: enable=bad-whitespace,bad-continuation
      print('expected_position_embs:', expected_output)
      print('actual_position_embs:', actual_position_embs)
      self.assertAllClose(actual_position_embs, expected_output)

  def testPositionalEmbeddingLayerWithScaling(self):
    with self.session(use_gpu=False) as sess:
      p = layers.PositionalEmbeddingLayer.Params()
      p.name = 'position_emb'
      p.min_timescale = 1
      p.max_timescale = 7
      p.embedding_dim = 4
      p.trainable_scaling = True
      p.trainable_scaling_init = 1.0 / np.sqrt(p.embedding_dim)
      seq_length = 11

      pos_emb_layer = layers.PositionalEmbeddingLayer(p)
      position_embs = pos_emb_layer.FPropDefaultTheta(seq_length)
      tf.global_variables_initializer().run()
      actual_position_embs, = sess.run([position_embs])

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032],
          [-0.7568025 ,  0.54083425, -0.65364361,  0.84112918],
          [-0.95892417,  0.65507787,  0.28366217,  0.75556135],
          [-0.27941549,  0.75597537,  0.96017027,  0.65460002],
          [ 0.65698659,  0.84147096,  0.7539022 ,  0.54030228],
          [ 0.98935831,  0.90982294, -0.14550003,  0.41499668],
          [ 0.41211855,  0.9596386 , -0.91113025,  0.28123617],
          [-0.54402113,  0.98990309, -0.83907151,  0.14174587]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_output / np.sqrt(p.embedding_dim),
                          actual_position_embs)


class SoftmaxLayerTest(test_utils.TestCase):

  def _RunSimpleFullSoftmax(self,
                            num_shards=1,
                            chunk_size=0,
                            inputs=None,
                            class_ids=None,
                            class_weights=None,
                            class_probabilities=None,
                            num_samples=0,
                            default_qdomain=None,
                            training_step=-1,
                            seed=None,
                            dtype=tf.float32,
                            fprop_dtype=None,
                            apply_pruning=False):
    if fprop_dtype is None:
      fprop_dtype = dtype
    with self.session(use_gpu=True, graph=tf.Graph()) as sess:
      if seed is not None:
        tf.set_random_seed(seed)
      if class_ids is None:
        class_ids = tf.constant([[1], [5], [10]], dtype=tf.int32)
      else:
        class_ids = tf.constant(class_ids)
      if class_weights is None:
        class_weights = tf.constant([1.0, 0.4, 0.8], dtype=fprop_dtype)
      else:
        class_weights = tf.constant(class_weights)
      np.random.seed(12345)
      if inputs is None:
        inputs = [tf.constant(np.random.rand(3, 10), dtype=fprop_dtype)]
      else:
        inputs = [tf.constant(inputs, dtype=fprop_dtype)]

      params = layers.SimpleFullSoftmax.Params()
      params.dtype = dtype
      params.fprop_dtype = fprop_dtype
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.num_shards = num_shards
      params.chunk_size = chunk_size
      params.apply_pruning = apply_pruning
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.random_seed = 12345678

      if default_qdomain is not None:
        params.qdomain.default = default_qdomain

      if num_samples > 0:
        # Turn on sampled soft-max; the asserts need to hold for it to be used.
        params.num_sampled = num_samples
        assert class_probabilities is None
        assert chunk_size == 0
        assert params.is_eval is not True

      params.vn.global_vn = False
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.FProp(
          softmax.theta,
          inputs,
          class_weights=class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)

      all_vars = tf.get_collection('SimpleFullSoftmax_vars')
      expected_var_names = []
      for i in range(num_shards):
        expected_var_names.append(u'softmax/weight_%d/var:0' % i)
        expected_var_names.append(u'softmax/bias_%d/var:0' % i)

      all_var_names = [v.name for v in all_vars]
      self.assertEqual(sorted(expected_var_names), sorted(all_var_names))

      tf.global_variables_initializer().run()
      if training_step >= 0:
        sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), training_step))
      return sess.run(xent_loss)

  def testSimpleFullSoftmaxMasked(self):
    num_shards = 2
    apply_pruning = True
    params = layers.SimpleFullSoftmax.Params()
    params.name = 'softmax'
    params.dtype = tf.float32
    params.input_dim = 10
    params.num_classes = 32
    params.fprop_dtype = tf.float32
    params.num_shards = num_shards
    params.apply_pruning = apply_pruning
    params.random_seed = 12345678
    softmax_layer = layers.SimpleFullSoftmax(params)

    self.assertTrue('weight_0' in softmax_layer.vars.weight_0.name)
    self.assertTrue('weight_1' in softmax_layer.vars.weight_1.name)
    self.assertTrue('mask_0' in softmax_layer.vars.mask_0.name)
    self.assertTrue('mask_1' in softmax_layer.vars.mask_1.name)
    self.assertTrue('threshold_0' in softmax_layer.vars.threshold_0.name)
    self.assertTrue('threshold_1' in softmax_layer.vars.threshold_1.name)

    self.assertEqual(softmax_layer.theta.weight_0.get_shape(),
                     tf.TensorShape([10, 16]))
    self.assertEqual(softmax_layer.theta.weight_1.get_shape(),
                     tf.TensorShape([10, 16]))
    self.assertEqual(softmax_layer.theta.mask_0.get_shape(),
                     tf.TensorShape([10, 16]))
    self.assertEqual(softmax_layer.theta.mask_1.get_shape(),
                     tf.TensorShape([10, 16]))
    self.assertEqual(softmax_layer.theta.threshold_0.get_shape(),
                     tf.TensorShape([]))
    self.assertEqual(softmax_layer.theta.threshold_0.get_shape(),
                     tf.TensorShape([]))

    softmax_var_count = 4  # 2 each for weights and biases (we have 2 shards)
    wts = tf.get_collection('SimpleFullSoftmax_vars')
    self.assertEqual(softmax_var_count, len(wts))

    softmax_mask_count = 2
    masks = tf.get_collection('masks')
    self.assertEqual(softmax_mask_count, len(masks))

    softmax_threshold_count = 2
    threshold = tf.get_collection('thresholds')
    self.assertEqual(softmax_threshold_count, len(threshold))

    # Sampled and Masked
    xent_loss = self._RunSimpleFullSoftmax(
        num_samples=32, seed=12345, apply_pruning=True)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.681571, 1e-5)
    self.assertNear(log_perplexity, 3.946169, 1e-5)

    # Sharded and Masked
    xent_loss = self._RunSimpleFullSoftmax(num_shards=2, apply_pruning=True)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 6.14888, 1e-5)
    self.assertNear(log_perplexity, 2.79495, 1e-5)

    # Non_2D and Masked
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_ids=np.random.randint(32, size=(4, 3)),
        apply_pruning=True)
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_probabilities=np.random.uniform(size=(4, 3, 32)),
        apply_pruning=True)
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    # Chunked and Masked
    for chunk_size in (0, 1, 2, 3, 4, 5):
      print('chunk_size = ', chunk_size)
      xent_output = self._RunSimpleFullSoftmax(
          chunk_size=chunk_size, apply_pruning=True)
      loss = xent_output.total_xent
      log_perplexity = xent_output.avg_xent
      print('xent_output ', xent_output)
      print('xent_output.per_example_argmax.dtype ',
            xent_output.per_example_argmax.dtype)
      self.assertAllClose(loss, 6.22425)
      self.assertAllClose(log_perplexity, 2.82920)
      self.assertAllEqual(xent_output.per_example_argmax,
                          np.argmax(xent_output.logits, axis=1))

  def testSimpleFullSoftmax_Sampled(self):
    xent_loss = self._RunSimpleFullSoftmax(num_samples=32, seed=12345)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.681571, 1e-5)
    self.assertNear(log_perplexity, 3.946169, 1e-5)

  def testSimpleFullSoftmax_SampledAndSharded(self):
    xent_loss = self._RunSimpleFullSoftmax(
        num_shards=4, num_samples=32, seed=12345)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.510439, 1e-5)
    self.assertNear(log_perplexity, 3.868381, 1e-5)

  def testSimpleFullSoftmax_Non2D(self):
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_ids=np.random.randint(32, size=(4, 3)))
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_probabilities=np.random.uniform(size=(4, 3, 32)))
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

  def _testSimpleFullSoftmax_Basic_Helper(self, dtype, fprop_dtype):
    xent_loss = self._RunSimpleFullSoftmax(dtype=dtype, fprop_dtype=fprop_dtype)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    err = 1e-5
    if fprop_dtype == tf.float16 or fprop_dtype == tf.bfloat16:
      err = 1e-2
    self.assertNear(loss, 6.22425, err=err)
    self.assertNear(log_perplexity, 2.8292, err=err)
    self.assertAllEqual(xent_loss.per_example_argmax,
                        np.argmax(xent_loss.logits, axis=1))

  def testSimpleFullSoftmax_Basic_Float32(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float32)

  def testSimpleFullSoftmax_Basic_Float32Float16(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float16)

  def testSimpleFullSoftmax_Sharded(self):
    xent_loss = self._RunSimpleFullSoftmax(2)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    self.assertNear(loss, 6.14888, 1e-5)
    self.assertNear(log_perplexity, 2.79495, 1e-5)

  def testSimpleFullSoftmax_Chunked(self):
    for chunk_size in (0, 1, 2, 3, 4, 5):
      print('chunk_size = ', chunk_size)
      xent_output = self._RunSimpleFullSoftmax(chunk_size=chunk_size)
      loss = xent_output.total_xent
      log_perplexity = xent_output.avg_xent
      print('xent_output ', xent_output)
      print('xent_output.per_example_argmax.dtype ',
            xent_output.per_example_argmax.dtype)
      self.assertAllClose(loss, 6.22425)
      self.assertAllClose(log_perplexity, 2.82920)
      self.assertAllEqual(xent_output.per_example_argmax,
                          np.argmax(xent_output.logits, axis=1))

  def testSimpleFullSoftmax_Basic_Distributions(self):
    with self.session(use_gpu=False) as sess:
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = False
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs,
          class_weights=class_weights,
          class_probabilities=tf.one_hot(class_ids, params.num_classes))
      tf.global_variables_initializer().run()
      loss = sess.run(xent_loss.total_xent)
      log_perplexity = sess.run(xent_loss.avg_xent)
      print(['loss', loss])
      print(['log_perplexity', log_perplexity])
      self.assertNear(loss, 6.22425, 1e-5)
      self.assertNear(log_perplexity, 2.8292, 1e-5)

  def testSimpleFullSoftmax_GlobalVN(self):
    with self.session(use_gpu=False) as sess:
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = True
      params.vn.seed = 23456
      params.vn.scale = 1.0
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs, class_weights=class_weights, class_ids=class_ids)
      tf.global_variables_initializer().run()
      loss = sess.run(xent_loss.total_xent)
      log_perplexity = sess.run(xent_loss.avg_xent)
      print(['testSimpleFullSoftmax_GlobalVN loss', loss])
      print(['testSimpleFullSoftmax_GlobalVN log_perplexity', log_perplexity])
      self.assertNear(loss, 19.9612, 1e-4)
      self.assertNear(log_perplexity, 3.46426, 1e-4)

  def testSimpleFullSoftmax_PerStepVN(self):
    with self.session(use_gpu=False) as sess:
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = False
      params.vn.per_step_vn = True
      params.vn.seed = 23456
      params.vn.scale = 1.0
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs, class_weights=class_weights, class_ids=class_ids)
      tf.global_variables_initializer().run()
      loss = sess.run(xent_loss.total_xent)
      log_perplexity = sess.run(xent_loss.avg_xent)
      print(['testShardedFullSoftmax_PerStepVN loss', loss])
      print(['testShardedFullSoftmax_PerStepVN log_perplexity', log_perplexity])
      self.assertNear(loss, 19.9612, 1e-4)
      self.assertNear(log_perplexity, 3.46426, 1e-4)

  def testSimpleFullSoftmax_FakeQuantized(self):
    default_qdomain = quant_utils.SymmetricScheduledClipQDomain.Params()
    default_qdomain.cc_schedule = quant_utils.FakeQuantizationSchedule.Params(
    ).Set(
        clip_start_step=0, clip_end_step=2, quant_start_step=2)
    xent_loss = self._RunSimpleFullSoftmax(
        default_qdomain=default_qdomain, training_step=5)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    self.assertNear(loss, 6.285590, 1e-5)
    self.assertNear(log_perplexity, 2.857086, 1e-5)

  def _RunSimpleFullSoftmaxGradientChecker(self, batch_size, num_classes,
                                           chunk_size, num_shards):
    for (dtype, use_gpu, tolerance) in [(tf.float32, True, 1e-2),
                                        (tf.float64, False, 1e-6)]:
      tf.logging.info('dtype %s tolerance %g', dtype, tolerance)
      with self.session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
        input_dim = 10
        np.random.seed(12345)
        class_ids = tf.constant(
            np.random.randint(num_classes, size=(batch_size, 1)),
            dtype=tf.int32)
        class_weights = tf.constant(np.random.rand(batch_size), dtype=dtype)
        inputs = [
            tf.constant(np.random.rand(batch_size, input_dim), dtype=dtype)
        ]

        params = layers.SimpleFullSoftmax.Params()
        params.name = 'softmax'
        params.dtype = dtype
        params.input_dim = input_dim
        params.num_classes = num_classes
        params.num_shards = num_shards
        params.chunk_size = chunk_size
        params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
        params.vn.global_vn = False
        softmax = layers.SimpleFullSoftmax(params)
        xent_loss = softmax.XentLoss(
            inputs, class_weights=class_weights, class_ids=class_ids)
        softmax_vars = softmax.vars.Flatten()
        # Now add the backward graph.
        grads = tf.gradients(xent_loss.total_xent, softmax_vars)

        tf.global_variables_initializer().run()
        assert len(softmax_vars) == len(grads)
        for x, grad_x in zip(softmax_vars, grads):
          grad_symbolic = sess.run(grad_x)
          grad_numeric = test_utils.ComputeNumericGradient(
              sess, xent_loss.total_xent, x)
          self.assertAllClose(
              grad_symbolic, grad_numeric, atol=tolerance, rtol=tolerance)

  def testSimpleFullSoftmaxGradientChecker(self):
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 0, 1)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 0, 2)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 2, 2)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 5, 2)


class SoftmaxLayerLogitsTest(test_utils.TestCase):
  """Testing SoftmaxLayer.Logits()."""

  def _Logits(self, params, batch_size=2, seq_length=None):
    with self.session(use_gpu=True, graph=tf.Graph()):
      np.random.seed(12345)
      tf.set_random_seed(1234)

      params.name = 'softmax'
      if not params.input_dim:
        params.input_dim = 3
      if not params.num_classes:
        params.num_classes = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      softmax = params.cls(params)

      input_dim = params.input_dim
      if seq_length:
        inputs = np.random.rand(batch_size, seq_length, input_dim)
      else:
        inputs = np.random.rand(batch_size, input_dim)
      inputs = tf.constant(inputs, dtype=py_utils.FPropDtype(params))
      logits = softmax.Logits(softmax.theta, inputs)

      if seq_length:
        logits = py_utils.HasShape(logits,
                                   [batch_size, seq_length, params.num_classes])
      else:
        logits = py_utils.HasShape(logits, [batch_size, params.num_classes])
      tf.global_variables_initializer().run()
      return logits.eval()

  def testConvSoftmaxLogits(self):
    params = layers.ConvSoftmax.Params()
    self.assertAllClose([[0.52536774, -0.17598523, 0.38314393, -0.36068222],
                         [0.75792629, -0.18001975, 0.42298675, -0.35423514]],
                        self._Logits(params))

  def testSimpleFullSoftmax(self):
    params = layers.SimpleFullSoftmax.Params()
    self.assertAllClose([[0.52536774, -0.17598523, 0.38314393, -0.36068222],
                         [0.75792629, -0.18001975, 0.42298675, -0.35423514]],
                        self._Logits(params))

  def testConvSoftmaxLogitsWith3DInputs(self):
    params = layers.ConvSoftmax.Params()
    logits = self._Logits(params, seq_length=5)
    self.assertAllClose(6.9934864, np.sum(logits))


class FeedForwardNetTest(test_utils.TestCase):

  def testFeedForwardNetConstruction(self):
    with self.session(use_gpu=False):
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=True,
          activation='TANH',
          params_init=py_utils.WeightInit.Uniform(1.0))
      p.dropout.keep_prob = 0.5
      proj_l = p.cls(p)
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)

      p = layers.FeedForwardNet.Params().Set(
          name='ffn2',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=True,
          activation='TANH',
          params_init=py_utils.WeightInit.Uniform(1.0))
      p.dropout = [
          layers.DropoutLayer.Params().Set(keep_prob=0.5),
          layers.DropoutLayer.Params().Set(keep_prob=0.9)
      ]
      proj_l = p.cls(p)
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)

      p = layers.FeedForwardNet.Params().Set(
          name='ffn3',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=[True, False],
          activation=['TANH', 'RELU'],
          params_init=py_utils.WeightInit.Uniform(1.0))
      p.dropout = [
          layers.DropoutLayer.Params().Set(keep_prob=0.5),
          layers.DropoutLayer.Params().Set(keep_prob=0.9)
      ]
      proj_l = p.cls(p)
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)

  def testFeedForwardNet(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=False,
          activation=['RELU', 'NONE'])
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.cls(p)

      p1 = layers.ProjectionLayer.Params().Set(
          name='p1',
          input_dim=10,
          output_dim=20,
          activation='RELU',
          batch_norm=False)
      p1.params_init = params_init
      p1_l = p1.cls(p1)

      p2 = layers.ProjectionLayer.Params().Set(
          name='p2',
          input_dim=20,
          output_dim=30,
          activation='NONE',
          batch_norm=False)
      p2.params_init = params_init
      p2_l = p2.cls(p2)

      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out1 = feedforward_net.FPropDefaultTheta(a)

      out2 = p2_l.FPropDefaultTheta(p1_l.FPropDefaultTheta(a))

      tf.global_variables_initializer().run()
      out1_v, out2_v = sess.run([out1, out2])
      self.assertAllClose(out1_v, out2_v)

  def testFeedForwardNetQuantized(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)

      cc_schedule = quant_utils.FakeQuantizationSchedule.Params().Set(
          clip_start_step=1,
          clip_end_step=2,
          quant_start_step=2,
          start_cap=8.0,
          end_cap=2.0)
      proj_qdomain = quant_utils.SymmetricScheduledClipQDomain.Params().Set(
          cc_schedule=cc_schedule)

      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=False,
          activation=['RELU', 'NONE'])
      p.qdomain.default = proj_qdomain.Copy()
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.cls(p)

      p1 = layers.ProjectionLayer.Params().Set(
          name='p1',
          input_dim=10,
          output_dim=20,
          activation='RELU',
          batch_norm=False)
      p1.qdomain.default = proj_qdomain.Copy()
      p1.params_init = params_init
      p1_l = p1.cls(p1)

      p2 = layers.ProjectionLayer.Params().Set(
          name='p2',
          input_dim=20,
          output_dim=30,
          activation='NONE',
          batch_norm=False)
      p2.params_init = params_init
      p2.qdomain.default = proj_qdomain.Copy()
      p2_l = p2.cls(p2)

      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out1 = feedforward_net.FPropDefaultTheta(a)
      out2 = p2_l.FPropDefaultTheta(p1_l.FPropDefaultTheta(a))

      tf.global_variables_initializer().run()

      sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 5))
      out1_v, out2_v = sess.run([out1, out2])
      self.assertAllClose(out1_v, out2_v)

  def testFeedForwardNetBnFolded(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=True,
          bn_fold_weights=True,
          activation=['RELU', 'NONE'])
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.cls(p)

      p1 = layers.ProjectionLayer.Params().Set(
          name='p1',
          input_dim=10,
          output_dim=20,
          activation='RELU',
          batch_norm=True,
          bn_fold_weights=True)
      p1.params_init = params_init
      p1_l = p1.cls(p1)

      p2 = layers.ProjectionLayer.Params().Set(
          name='p2',
          input_dim=20,
          output_dim=30,
          activation='NONE',
          batch_norm=True,
          bn_fold_weights=True)
      p2.params_init = params_init
      p2_l = p2.cls(p2)

      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out1 = feedforward_net.FPropDefaultTheta(a)

      out2 = p2_l.FPropDefaultTheta(p1_l.FPropDefaultTheta(a))

      tf.global_variables_initializer().run()
      out1_v, out2_v = sess.run([out1, out2])
      self.assertAllClose(out1_v, out2_v)

  def testFeedForwardNetSmokeTest(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          activation=['RELU', 'NONE'])
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.cls(p)
      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out = tf.reduce_sum(feedforward_net.FPropDefaultTheta(a))
      out_abs = tf.reduce_sum(tf.abs(feedforward_net.FPropDefaultTheta(a)))

      tf.global_variables_initializer().run()
      # pyformat: disable
      test_utils.CompareToGoldenSingleFloat(self, 8.190775, out.eval(), atol=1e-5)  # pylint: disable=line-too-long
      # pyformat: enable
      test_utils.CompareToGoldenSingleFloat(self, 36.773586, out_abs.eval())

  def testDropoutLayerTrain(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      p = layers.DropoutLayer.Params()
      p.keep_prob = 0.5
      p.random_seed = 1234
      p.name = 'dropout'

      dl = p.cls(p)

      x = tf.random_normal([10, 10, 10, 3])
      xd = dl.FPropDefaultTheta(x)
      x, xd = sess.run([x, xd])
      self.assertGreater((xd == 0).mean(), 0.3)
      self.assertLess((xd == 0).mean(), 0.7)
      self.assertAllClose(xd[xd != 0], x[xd != 0] / p.keep_prob)

  def testDropoutLayerEval(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      p = layers.DropoutLayer.Params()
      p.keep_prob = 0.5
      p.random_seed = 1234
      p.name = 'dropout'
      p.is_eval = True

      dl = p.cls(p)

      x = tf.random_normal([10, 10, 10, 3])
      xd = dl.FPropDefaultTheta(x)

      x, xd = sess.run([x, xd])

      self.assertAllEqual(xd, x)


class AddingAccumulatorTest(test_utils.TestCase):
  """Test for AddingAccumulator."""

  def testAddingAccumulator(self):
    with self.session():
      layer_p = layers.IdentityLayer.Params()
      layer_p.name = 'test'
      layer = layer_p.cls(layer_p)

      layer.RegisterAccumulator('acc1', layers.AddingAccumulator([],
                                                                 tf.float32))

      # Initial value.
      self.assertEqual(0.0, layer.accumulators.acc1.GetValue().eval())

      # Update/merge.
      layer.accumulators.acc1.Update(1.0)
      layer.accumulators.acc1.Update(1.0)
      self.assertEqual(2.0, layer.accumulators.acc1.GetValue().eval())

      # Reset.
      layer.accumulators.Transform(lambda acc: acc.Reset())
      self.assertEqual(0.0, layer.accumulators.acc1.GetValue().eval())


class BatchNormLayerNoPaddingTest(test_utils.TestCase, parameterized.TestCase):

  def testBatchNormLayerNoPaddingConstruction(self):
    tf.set_random_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayerNoPadding.Params()
    params.name = 'bn'
    params.dim = 2
    params.params_init = py_utils.WeightInit.Gaussian(0.1)
    params.is_eval = False
    layers.BatchNormLayerNoPadding(params)
    bn_vars = tf.get_collection('BatchNormLayerNoPadding_vars')
    bn_var_names = [x.name for x in bn_vars]
    expected_var_names = [
        'bn/beta/var:0', 'bn/gamma/var:0', 'bn/moving_mean/var:0',
        'bn/moving_variance/var:0'
    ]
    self.assertEqual(expected_var_names, bn_var_names)

  @parameterized.named_parameters({
      'testcase_name': '_eval',
      'is_eval': True,
  }, {
      'testcase_name': '_train',
      'is_eval': False,
  })
  def testBatchNormLayerNoPaddingFProp(self, is_eval):
    tf.set_random_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayerNoPadding.Params()
    params.name = 'bn'
    params.dim = 3
    params.params_init = py_utils.WeightInit.Gaussian(0.1)
    params.is_eval = is_eval

    bn_layer = layers.BatchNormLayerNoPadding(params)
    bn_in1 = tf.constant(
        np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

    bn_out = bn_layer.FPropDefaultTheta(bn_in1)
    sig1 = tf.reduce_sum(bn_out)
    sig2 = tf.reduce_sum(bn_out * bn_out)
    expected_sig1 = 2.6593573 if is_eval else 0
    expected_sig2 = 15.4642076 if is_eval else 47.850193
    with self.session(use_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(expected_sig1, sig1.eval(), atol=1e-5)
      self.assertAllClose(expected_sig2, sig2.eval(), atol=1e-5)

  def testBatchNormLayerNoPaddingFPropUseGlobalStatsForTraining(self):
    tf.set_random_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayerNoPadding.Params()
    params.name = 'bn'
    params.dim = 3
    params.params_init = py_utils.WeightInit.Gaussian(0.1)
    params.is_eval = False

    bn_layer = layers.BatchNormLayerNoPadding(params)
    bn_in1 = tf.constant(
        np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

    bn_out = bn_layer.FPropDefaultTheta(bn_in1)
    sig1 = tf.reduce_sum(bn_out)
    sig2 = tf.reduce_sum(bn_out * bn_out)
    with self.session(use_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(1.19209289551e-06, sig1.eval(), atol=1e-5)
      self.assertAllClose(47.8501930237, sig2.eval(), atol=1e-5)

  def testBatchNormLayerNoPaddingFPropForConv(self):
    tf.set_random_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayerNoPadding.Params()
    params.name = 'bn_conv'
    params.dim = 32
    params.params_init = py_utils.WeightInit.Gaussian(0.1)
    params.is_eval = False

    bn_layer = layers.BatchNormLayerNoPadding(params)
    bn_in1 = tf.constant(
        np.random.normal(0.1, 0.5, [2, 8, 4, 32]), dtype=tf.float32)

    bn_out = bn_layer.FPropDefaultTheta(bn_in1)
    sig1 = tf.reduce_sum(bn_out)
    sig2 = tf.reduce_sum(bn_out * bn_out)
    with self.session(use_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(0.0, sig1.eval(), atol=1e-4)
      self.assertAllClose(2039.398681, sig2.eval())

  def _BuildDummyStackedBNLayer(self, splits):
    num_micro_batches = 8
    if splits == 0:
      endpoint = layers.BatchNormLayerNoPadding.Params().Set(
          decay=0.997, name='bn', dim=1)
    else:
      cell_tpl = []
      for split in range(splits):
        nets_to_split = [
            layers.BatchNormLayerNoPadding.Params().Set(
                decay=0.997, name='bn_{}'.format(split), dim=1),
        ]
        split_layer = gpipe.FeatureExtractionLayer.Params().Set(
            name='split_{}'.format(split), sub=nets_to_split)
        cell_tpl.append(split_layer)
      endpoint = gpipe.PipeliningLayer.Params().Set(
          name='pipeline',
          num_micro_batches=num_micro_batches,
          cell_tpl=cell_tpl,
          before_tpl=[])
    layer = endpoint.cls(endpoint)
    return layer

  @parameterized.named_parameters({
      'testcase_name': '_baseline',
      'splits': 0,
  }, {
      'testcase_name': '_two_splits',
      'splits': 2,
  }, {
      'testcase_name': '_four_splits',
      'splits': 4,
  })
  def testBatchNormLayerNoPaddingAccumulators(self, splits):
    batch_size = 1024
    with self.session(graph=tf.Graph()) as sess:
      # Construct a network where loss = w * x + b
      global_step = py_utils.GetGlobalStep()
      inputs = tf.concat([
          tf.ones([batch_size // 2, 1, 1, 1]),
          tf.zeros([batch_size // 2, 1, 1, 1])
      ],
                         axis=0)
      net = self._BuildDummyStackedBNLayer(splits)
      logits = net.FPropDefaultTheta(inputs)
      loss = tf.reduce_mean(logits)
      grads = tf.gradients(loss, tf.trainable_variables())
      # Check the accumulator values
      counts = []
      means = []
      variances = []
      for i in range(splits):
        l = net.children['split_{}'.format(i)].children['bn_{}'.format(i)]
        counts.append(l.accumulators.counts.GetValue())
        means.append(l.accumulators.mean_ss.GetValue())
        variances.append(l.accumulators.variance_ss.GetValue())
      if splits == 0:
        counts.append(net.accumulators.counts.GetValue())
        means.append(net.accumulators.mean_ss.GetValue())
        variances.append(net.accumulators.variance_ss.GetValue())
      post_training_step_updates = net.PostTrainingStepUpdate(global_step)

      sess.run(tf.global_variables_initializer())
      _, count_vals, mean_vals, var_vals = sess.run(
          [grads, counts, means, variances])

      self.assertSameElements(count_vals, {batch_size})

      self.assertEqual(batch_size // 2, mean_vals[0])
      if len(mean_vals) > 1:
        self.assertSameElements(mean_vals[1:], {0})

      self.assertEqual(batch_size // 2, var_vals[0])
      if len(var_vals) > 1:
        self.assertSameElements(var_vals[1:], {0})
      sess.run(post_training_step_updates)
      moving_vars = sess.run(tf.get_collection('moving_vars'))
    self.assertEqual(0.0015, moving_vars[0])
    self.assertNear(0.997750, moving_vars[1], err=1.0e-6)


class LayerNormTest(test_utils.TestCase):

  def testLayerNormFProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.LayerNorm.Params()
      p.name = 'ln'
      p.input_dim = 3
      layer_norm = layers.LayerNorm(p)
      npy_input = np.random.normal(1.0, 0.5,
                                   [2, 4, 4, p.input_dim]).astype('float32')
      inputs = tf.constant(npy_input, dtype=tf.float32)
      output = layer_norm.FPropDefaultTheta(inputs)

      tf.global_variables_initializer().run()
      sym_output = sess.run(output)

      # Mean should be zero and variance should be close to one.
      self.assertNear(0.0, sym_output.sum(), 1e-5)
      self.assertNear(1.0, np.var(sym_output), 1e-4)

      # Compare with numpy.
      mean = npy_input.mean(-1, keepdims=True)
      variance = np.mean(np.square(npy_input - mean), -1, keepdims=True)
      npy_output = (npy_input - mean) / np.sqrt(variance + p.epsilon)
      self.assertAllClose(sym_output, npy_output)

  def testLayerNormBProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.LayerNorm.Params()
      p.name = 'ln'
      p.input_dim = 3
      layer_norm = layers.LayerNorm(p)

      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, p.input_dim]), dtype=tf.float32)
      output = layer_norm.FPropDefaultTheta(inputs)
      loss = tf.reduce_sum(output)

      all_vars = tf.trainable_variables()
      self.assertEqual(2, len(all_vars))

      grads = tf.gradients(loss, all_vars)
      tf.global_variables_initializer().run()
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-02, atol=1e-02)


class DeterministicDropoutTest(test_utils.TestCase, parameterized.TestCase):

  def testDeterministicDropoutLayer(self):
    params = layers.DeterministicDropoutLayer.Params().Set(keep_prob=0.7)
    params.name = 'drop'
    dropout = layers.DeterministicDropoutLayer(params)

    x = tf.ones([4, 6], dtype=tf.float32)
    x_expected = np.array([
        [1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 1, 1],
    ]) / 0.7

    with self.session():
      tf.assign(py_utils.GetGlobalStep(), 1234).eval()
      py_utils.ResetStepSeed(seed=5678)
      x_val = dropout.FPropDefaultTheta(x).eval()
      self.assertAllClose(x_expected, x_val)
      self.assertEqual(5679, py_utils.GetStepSeed().eval())

      # Different step seed gives different result.
      x_val = dropout.FPropDefaultTheta(x).eval()
      self.assertNotAllClose(x_expected, x_val)

      # Different global step gives different result
      tf.assign(py_utils.GetGlobalStep(), 1235).eval()
      py_utils.ResetStepSeed(seed=5678)
      x_val = dropout.FPropDefaultTheta(x).eval()
      self.assertNotAllClose(x_expected, x_val)

      # The same seeds in the same session is consistent.
      tf.assign(py_utils.GetGlobalStep(), 1234).eval()
      py_utils.ResetStepSeed(seed=5678)
      x_val = dropout.FPropDefaultTheta(x).eval()
      self.assertAllClose(x_expected, x_val)

    # The same seeds in a different session is consistent.
    with self.session():
      tf.assign(py_utils.GetGlobalStep(), 1234).eval()
      py_utils.ResetStepSeed(seed=5678)
      x_val = dropout.FPropDefaultTheta(x).eval()
      self.assertAllClose(x_expected, x_val)

  def testNoiseShapeBroadcastDims(self):
    params = layers.DeterministicDropoutLayer.Params().Set(
        keep_prob=0.7, noise_shape_broadcast_dims=[-1])
    params.name = 'drop'
    dropout = layers.DeterministicDropoutLayer(params)

    x = tf.ones([4, 6])
    x_expected = np.array([
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]) / 0.7

    with self.session():
      tf.assign(py_utils.GetGlobalStep(), 1234).eval()
      self.assertEqual(1234, dropout.theta.global_step.eval())
      py_utils.ResetStepSeed(seed=5678)
      x_val = dropout.FPropDefaultTheta(x).eval()
      self.assertEqual(5679, py_utils.GetStepSeed().eval())
    self.assertAllClose(x_expected, x_val)

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'splits': 1,
          'num_micro_batches': 1
      },
      {
          'testcase_name': 'OneSplitTwoMicroBatches',
          'splits': 1,
          'num_micro_batches': 2
      },
      {
          'testcase_name': 'TwoSplitsOneMicroBatch',
          'splits': 2,
          'num_micro_batches': 1
      },
      {
          'testcase_name': 'TwoSplitsTwoMicroBatches',
          'splits': 2,
          'num_micro_batches': 2
      },
  )
  def testDropoutInRecurrent(self, splits=1, num_micro_batches=1):
    """Test to verify the drop mask used in fprop and bprop is identical."""
    assert splits in [1, 2, 4]
    with self.session() as sess:
      tf.set_random_seed(12345)
      num_layers = 4
      # Build a model with 4 dropout layers.
      blocks = []
      for l in range(num_layers):
        blocks.append(layers.DeterministicDropoutLayer.Params().Set(
            name='dropout_{}'.format(l), keep_prob=0.7))
      # Divide the model into splits partitions.
      cell_tpl = []
      blocks_per_split = num_layers // splits
      for i in range(splits):
        sub = blocks[i * blocks_per_split:(i + 1) * blocks_per_split]
        cell_tpl.append(gpipe.FeatureExtractionLayer.Params().Set(
            name='cell_{}'.format(i), sub=sub))
      # Parallelize partitions using pipeline.
      p = gpipe.PipeliningLayer.Params().Set(
          name='pipeline',
          num_micro_batches=num_micro_batches,
          cell_tpl=cell_tpl)
      # Fake input
      x = tf.ones([2, 3])
      # Construct weights.
      w = tf.get_variable(
          'w', shape=[2, 3], initializer=tf.constant_initializer([[1] * 3] * 2))
      mdl = p.cls(p)
      y = mdl.FPropDefaultTheta(x * w)
      # Construct loss function such that gradients = final activation.
      loss = tf.reduce_sum(y)
      grads = py_utils.ComputeGradients(loss, py_utils.NestedMap(w=w))
      tf.global_variables_initializer().run()
      y_val = sess.run(y)
      grads_val = sess.run(grads)['w'][1]
      self.assertAllClose(y_val, grads_val)


class GradNormTrackerTest(test_utils.TestCase):

  def testGradNormTracker(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.GradNormTracker.Params().Set(
          name='grad_norm_tracker', clip_threshold=3.0)
      grad_norm_tracker = p.cls(p)
      grad_norm = tf.placeholder(tf.float32)
      grad_norm_clip = grad_norm_tracker.FPropDefaultTheta(grad_norm)

      tf.global_variables_initializer().run()

      random_normal = np.exp(np.random.normal(5.0, 1.0, size=10000))
      # We are expected to reject 16% of the outliers.
      outliers = np.exp(np.random.normal(7.0, 1.0, size=100))
      total_rejections = 0
      for i in range(100):
        for j in range(100):
          sess.run([grad_norm_clip], {grad_norm: random_normal[i * 100 + j]})
        clip = sess.run([grad_norm_clip], {grad_norm: outliers[i]})[0]
        if clip == 0.0:
          total_rejections += 1
      # Q(yonghui): Why is total_rejections not deterministic?
      print('total_rejections', total_rejections)
      self.assertGreater(total_rejections, 5)

  def testGradNormTrackerClipCapMin(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.GradNormTracker.Params().Set(
          name='grad_norm_tracker',
          clip_threshold=3.0,
          grad_norm_clip_cap_min=math.exp(10.0))
      grad_norm_tracker = p.cls(p)
      grad_norm = tf.placeholder(tf.float32)
      grad_norm_clip = grad_norm_tracker.FPropDefaultTheta(grad_norm)

      tf.global_variables_initializer().run()

      random_normal = np.exp(np.random.normal(5.0, 1.0, size=10000))
      # We expect no outliers being rejected due to the grad_norm_clip_cap_min.
      outliers = np.exp(np.random.normal(7.0, 1.0, size=100))
      total_rejections = 0
      for i in range(100):
        for j in range(100):
          sess.run([grad_norm_clip], {grad_norm: random_normal[i * 100 + j]})
        clip = sess.run([grad_norm_clip], {grad_norm: outliers[i]})[0]
        if clip == 0.0:
          total_rejections += 1
      print('total_rejections', total_rejections)
      self.assertEqual(total_rejections, 0)

  def testGradNormTrackerHasNan(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.GradNormTracker.Params().Set(
          name='grad_norm_tracker', clip_threshold=3.0)
      grad_norm_tracker = p.cls(p)
      grad_norm = tf.placeholder(tf.float32)
      has_nan = tf.cast(tf.ones([]), dtype=tf.bool)
      grad_norm_clip = grad_norm_tracker.FPropDefaultTheta(grad_norm, has_nan)

      tf.global_variables_initializer().run()

      random_normal = np.exp(np.random.normal(5.0, 1.0, size=10000))
      outliers = np.exp(np.random.normal(7.0, 1.0, size=100))
      total_rejections = 0
      for i in range(100):
        for j in range(100):
          sess.run([grad_norm_clip], {grad_norm: random_normal[i * 100 + j]})
        clip = sess.run([grad_norm_clip], {grad_norm: outliers[i]})[0]
        if clip == 0.0:
          total_rejections += 1
      self.assertEqual(total_rejections, 100)


class HighwaySkipLayerTest(test_utils.TestCase):

  def testHighwaySkipLayerConstruction(self):
    with self.session(use_gpu=False):
      p = layers.HighwaySkipLayer.Params().Set(
          name='gffn',
          input_dim=10,
          carry_bias_init=1.0,
          couple_carry_transform_gates=True,
          batch_norm=False,
          params_init=py_utils.WeightInit.Uniform(1.0))
      proj_l = p.cls(p)
      a = tf.constant(1.0, shape=[20, 10])
      b = tf.constant(-2.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a, b)

  def testHighwaySkipLayerCarryGate(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.HighwaySkipLayer.Params().Set(
          name='gffn',
          input_dim=10,
          carry_bias_init=1000.0,
          couple_carry_transform_gates=True,
          batch_norm=False,
          params_init=py_utils.WeightInit.Uniform(1.0))
      proj_l = p.cls(p)
      a = tf.constant(1.0, shape=[20, 10])
      b = tf.constant(-2.0, shape=[20, 10])
      out = proj_l.FPropDefaultTheta(a, b)
      tf.global_variables_initializer().run()
      a, out = sess.run([a, out])
      self.assertAllClose(a, out)


class UniformLabelSmootherTest(test_utils.TestCase):

  def testUniformLabelSmoother(self):
    with self.session(use_gpu=False):
      params = layers.UniformLabelSmoother.Params()
      params.name = 'uls'
      params.num_classes = 5
      params.uncertainty = 0.1

      smooth_layer = layers.UniformLabelSmoother(params)
      target_labels = tf.constant([[0, 1, 2, 3, 3, 3, 4]], dtype=tf.int32)
      target_ids = tf.constant([[0, 0, 1, 2, 3, 3, 3]], dtype=tf.int32)
      target_paddings = tf.zeros(tf.shape(target_ids))
      output = smooth_layer.FPropDefaultTheta(target_paddings, target_labels,
                                              target_ids)
      tf.global_variables_initializer().run()
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [[
          [0.89999998,  0.025     ,  0.025     ,  0.025     ,  0.025     ],
          [0.025     ,  0.89999998,  0.025     ,  0.025     ,  0.025     ],
          [0.025     ,  0.025     ,  0.89999998,  0.025     ,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.89999998,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.89999998,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.89999998,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.025     ,  0.89999998]
      ]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      output_v = output.eval()
      self.assertAllClose(expected_output, output_v, atol=1e-2, rtol=1e-2)
      self.assertAllClose(np.ones(output_v.shape[:-1]), output_v.sum(-1))

  def testUniformLabelSmootherLargerToken(self):
    with self.session(use_gpu=False):
      params = layers.UniformLabelSmoother.Params()
      params.name = 'uls'
      params.num_classes = 5
      params.uncertainty = 0.1
      params.uncertainty_larger = 0.2
      params.token_id_uncertainty_larger = 4

      smooth_layer = layers.UniformLabelSmoother(params)
      target_labels = tf.constant([[0, 1, 2, 3, 3, 3, 3]], dtype=tf.int32)
      target_ids = tf.constant([[0, 0, 1, 2, 4, 4, 4]], dtype=tf.int32)
      target_paddings = tf.zeros(tf.shape(target_ids))
      output = smooth_layer.FPropDefaultTheta(target_paddings, target_labels,
                                              target_ids)
      tf.global_variables_initializer().run()
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [[
          [0.89999998,  0.025     ,  0.025     ,  0.025     ,  0.025     ],
          [0.025     ,  0.89999998,  0.025     ,  0.025     ,  0.025     ],
          [0.025     ,  0.025     ,  0.89999998,  0.025     ,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.89999998,  0.025     ],
          [0.05      ,  0.05      ,  0.05      ,  0.80000001,  0.05      ],
          [0.05      ,  0.05      ,  0.05      ,  0.80000001,  0.05      ],
          [0.05      ,  0.05      ,  0.05      ,  0.80000001,  0.05      ]
      ]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      output_v = output.eval()
      self.assertAllClose(expected_output, output_v, atol=1e-2, rtol=1e-2)
      self.assertAllClose(np.ones(output_v.shape[:-1]), output_v.sum(-1))


class WeightedSumLayerTest(test_utils.TestCase):

  def testWeightedSumLayer(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 2
      n_sources = 3
      ctxs = [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
              [[3.0, 4.0, 5.0, 6.0], [6.0, 7.0, 8.0, 9.0]],
              [[4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 1.0, 2.0]]]
      p = layers.WeightedSumLayer.Params()
      p.name = 'transparent_layer'
      p.num_sources = n_sources
      p.random_seed = 505837249
      merger = p.cls(p)

      ctxs = [tf.expand_dims(i, 2) for i in ctxs]
      ctx = tf.squeeze(merger.FProp(merger.theta, ctxs), 2)
      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [[ 2.66666675,  3.66666675,  4.66666698,  5.66666698],
                      [ 5.0,         6.0,         4.33333349,  5.33333349]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testWeightedSumLayerGlobalWeightAndMinimalProb(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 2
      n_sources = 3
      ctxs = [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
              [[3.0, 4.0, 5.0, 6.0], [6.0, 7.0, 8.0, 9.0]],
              [[4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 1.0, 2.0]]]
      p = layers.WeightedSumLayer.Params()
      p.name = 'transparent_layer'
      p.num_sources = n_sources
      p.random_seed = 505837249
      p.minimal_prob = 0.01
      p.global_weight_scale = 10.0
      merger = p.cls(p)

      ctxs = [tf.expand_dims(i, 2) for i in ctxs]
      ctx = tf.squeeze(merger.FProp(merger.theta, ctxs), 2)
      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [[ 2.66666675,  3.66666675,  4.66666698,  5.66666698],
                      [ 5.0,         6.0,         4.33333349,  5.33333349]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)


class GatedAverageLayerTest(test_utils.TestCase):

  def testGatedAverageLayer(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 2
      num_inputs = 3

      inp_1 = np.asarray([[0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      inp_2 = np.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      inp_3 = np.asarray([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      p = layers.GatedAverageLayer.Params()
      p.name = 'gated_avg_layer'
      p.num_inputs = num_inputs
      p.num_nodes = depth
      p.random_seed = 505837249
      g_avg = p.cls(p)

      avg = g_avg.FProp(g_avg.theta, [inp_1, inp_2, inp_3])
      tf.global_variables_initializer().run()
      actual_avg = sess.run(avg)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_avg = [
          [ 0.13070658,  0.13070658,  0.13070658,  0.13070658],
          [ -1.0, -1.0, 1.0 , 1.0]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_avg.shape, (batch, depth))
      self.assertAllClose(expected_avg, actual_avg, rtol=1e-05, atol=1e-05)


class LHUCLayerTest(test_utils.TestCase):

  def testLHUCLayer(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 2

      inp = np.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
                       dtype=np.float32)
      p = layers.LHUCLayer.Params()
      p.name = 'lhuc_layer'
      p.input_dim = depth
      p.random_seed = 505837249
      lhuc = p.cls(p)

      lhuc = lhuc.FProp(lhuc.theta, inp)
      tf.global_variables_initializer().run()
      actual_avg = sess.run(lhuc)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_avg = [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_avg.shape, (batch, depth))
      self.assertAllClose(expected_avg, actual_avg, rtol=1e-05, atol=1e-05)


class ResidualAdapterLayerTest(test_utils.TestCase):

  def testResidualAdapterLayer(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 2

      inp = np.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
                       dtype=np.float32)
      p = layers.ResidualAdapterLayer.Params()
      p.name = 'resadap_layer'
      p.input_dim = depth
      p.bottleneck_dim = 2
      p.random_seed = 505837249
      resadap = p.cls(p)

      resadap = resadap.FProp(resadap.theta, inp)
      tf.global_variables_initializer().run()
      actual_avg = sess.run(resadap)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_avg = [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_avg.shape, (batch, depth))
      self.assertAllClose(expected_avg, actual_avg, rtol=1e-05, atol=1e-05)


class GluLayerTest(test_utils.TestCase):

  def testGlu(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      inputs = tf.random_normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers.GluLayer.Params()
      p.name = 'glu_layers'
      p.input_dim = 3
      glu_layer = layers.GluLayer(p)

      h = glu_layer.FPropDefaultTheta(inputs, paddings)
      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(h)
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ -1.84272185e-01,  -3.82728219e-01,   8.69752645e-01],
           [  4.42533880e-01,   1.51665461e+00,   3.26201534e+00]],
          [[ -7.06624031e-01,  -6.52632236e-01,   1.22156203e+00],
           [  1.66484845e+00,   5.98078966e-01,   1.14039946e+00]],
          [[  3.26439053e-01,   2.47359693e-01,  -1.14889514e+00],
           [  7.71084905e-01,   1.07083774e+00,   1.74589559e-01]],
          [[  5.70576251e-01,   7.95466423e-01,  -4.07778949e-01],
           [ -8.71581078e-01,  -5.38501918e-01,  -2.50373930e-01]],
          [[ -3.88817638e-01,   5.84501982e-01,  -6.60797715e-01],
           [ -1.34579837e+00,  -2.18637614e-03,   1.55258143e+00]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testGluWithoutResidual(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      inputs = tf.random_normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers.GluLayer.Params()
      p.name = 'glu_layers'
      p.input_dim = 3
      p.output_dim = 4
      p.apply_residual = False
      glu_layer = layers.GluLayer(p)

      h = glu_layer.FPropDefaultTheta(inputs, paddings)
      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(h)
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ 0.2498899 ,  0.        ,  0.62683833,  0.        ],
           [ 0.34115699,  0.        ,  0.38020864,  0.        ]],
          [[ 0.3014423 ,  0.        ,  0.59274423,  0.        ],
           [ 0.        ,  0.35897657,  0.2908403 ,  0.03678071]],
          [[ 0.        ,  0.78786391,  0.        ,  0.38839644],
           [ 0.        ,  0.44012907,  0.        ,  0.41553062]],
          [[ 0.        ,  0.61838603,  0.        ,  0.41521466],
           [ 0.34117079,  0.        ,  0.0372162 ,  0.        ]],
          [[ 0.        ,  0.        ,  0.        ,  0.28136203],
           [ 0.34413674,  0.        ,  0.30943182,  0.        ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)


if __name__ == '__main__':
  tf.test.main()
