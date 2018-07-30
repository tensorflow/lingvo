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

import numpy as np
import tensorflow as tf

from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import test_utils


class BatchNormLayerTest(tf.test.TestCase):

  def testBatchNormLayerConstruction(self):
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
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


class ConvLayerTest(tf.test.TestCase):

  def testConvLayerConstruction(self):
    with self.test_session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      layers.ConvLayer(params)
      conv_vars = tf.get_collection('ConvLayer_vars')
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

  def testConvLayerWithBiasConstruction(self):
    """Tests ConvLayer with only bias and without batch normalization."""
    with self.test_session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      params.bias = True
      params.batch_norm = False
      layers.ConvLayer(params)
      conv_vars = tf.get_collection('ConvLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      # Has both 'w' and 'b'.
      expected_var_names = ['conv/w/var:0', 'conv/b/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      # No BatchNorm variables.
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = []
      self.assertEqual(expected_var_names, bn_var_names)

  def testConvLayerOutShape(self):
    with self.test_session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = layers.ConvLayer(params)
      in_shape = tf.TensorShape([None, None, 10, 3])
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape.as_list(), [None, None, 5, 32])
      in_shape = tf.TensorShape([None, 20, 10, 3])
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape.as_list(), [None, 10, 5, 32])

  def testConvLayerWithDilationOutShape(self):
    with self.test_session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [1, 1]
      params.dilation_rate = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = layers.ConvLayer(params)
      # dilation_rate does not change output shape.
      in_shape = tf.TensorShape([None, None, 10, 3])
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape.as_list(), [None, None, 10, 32])
      in_shape = tf.TensorShape([None, 20, 10, 3])
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape.as_list(), [None, 20, 10, 32])

  def testConvPoolComputeOutPadding(self):
    with self.test_session(use_gpu=True):
      in_padding = tf.constant(
          [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
          dtype=tf.float32)
      out_padding = layers._ComputeOutputPadding(in_padding, 2)
      expected_out_padding = [[1, 1, 0, 0, 0, 1, 1, 0],
                              [1, 1, 0, 0, 0, 1, 1, 0]]

      tf.global_variables_initializer().run()
      self.assertAllClose(expected_out_padding, out_padding.eval().tolist())

  def testConvPoolComputeOutPaddingUnevenStride(self):
    with self.test_session(use_gpu=True):
      in_padding = tf.constant(
          [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1
          ], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
          dtype=tf.float32)
      out_padding = layers._ComputeOutputPadding(in_padding, 3)
      expected_out_padding = [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]

      tf.global_variables_initializer().run()
      self.assertAllClose(expected_out_padding, out_padding.eval().tolist())

  def _checkConvLayerShapes(self,
                            input_shape,
                            filter_shape,
                            filter_stride,
                            dilation_rate=None):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = filter_shape
      params.filter_stride = filter_stride
      if dilation_rate:
        params.dilation_rate = dilation_rate
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      conv_layer = layers.ConvLayer(params)

      inp = tf.random_uniform(input_shape)
      inp_pad = tf.floor(0.5 + tf.random_uniform(input_shape[:2]))
      out, out_pad = conv_layer.FPropDefaultTheta(inp, inp_pad)

    with self.test_session(use_gpu=True, graph=g) as sess:
      tf.global_variables_initializer().run()
      out, out_pad = sess.run([out, out_pad])
      print(out.shape, out_pad.shape)
      # We expect conv_layer.OutShape can compute the actually output's shape.
      self.assertEqual(out.shape, conv_layer.OutShape(inp.get_shape()))
      # We expect out_pad.shape matches the 1st 2 dimensions of out.
      self.assertEqual(out.shape[:2], out_pad.shape)

  def testConvLayerOutputShapes(self):
    self._checkConvLayerShapes([2, 4, 4, 3], [3, 3, 3, 32], [1, 1])
    self._checkConvLayerShapes([2, 4, 4, 3], [3, 3, 3, 32], [2, 2])
    self._checkConvLayerShapes([2, 10, 4, 3], [3, 3, 3, 32], [3, 3])

    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1], dilation_rate=[2, 2])
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1], dilation_rate=[3, 3])

  def _evalConvLayerFProp(self,
                          batch_norm=True,
                          weight_norm=False,
                          bias=False,
                          activation='RELU',
                          conv_last=False,
                          strides=(2, 2),
                          dilation_rate=(1, 1)):
    self._ClearCachedSession()
    tf.reset_default_graph()
    with self.test_session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.filter_stride = strides
      params.dilation_rate = dilation_rate
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.conv_last = conv_last
      params.batch_norm = batch_norm
      params.weight_norm = weight_norm
      params.bias = bias
      params.activation = activation
      params.is_eval = False

      conv_layer = layers.ConvLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)

      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      output2, _ = conv_layer.FPropDefaultTheta(inputs1)
      tf.global_variables_initializer().run()
      v1, v2 = sess.run([output1, output2])
      self.assertAllClose(v1, v2)
      return v1

  def testConvLayerFProp(self):
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
    print(['actual = ', np.array_repr(actual)])
    self.assertAllClose(expected_output1, actual)

  def testConvLayerWithDilationFProp(self):
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
    print(['testConvLayerWithDilationFProp actual = ', np.array_repr(actual)])
    self.assertAllClose(expected_output1, actual)

  def testConvLayerConvFirstVsLastFProp(self):
    """Compare results of conv first vs. last."""
    # ... with batch_norm and activation disabled.
    self.assertAllClose(
        self._evalConvLayerFProp(
            batch_norm=False, activation='NONE', conv_last=False),
        self._evalConvLayerFProp(
            batch_norm=False, activation='NONE', conv_last=True))

  def testConvLayerFPropConvLast(self):
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

  def testConvLayerConvWithBias(self):
    """Compare results with bias vs. with neither batch_norm nor bias."""
    # Results should match since bias is initialized to be 0.
    self.assertAllClose(
        self._evalConvLayerFProp(batch_norm=False, bias=False),
        self._evalConvLayerFProp(batch_norm=False, bias=True))

  def testConvLayerWeightNormFProp(self):
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
    print(['actual1 = ', np.array_repr(actual)])
    self.assertAllClose(expected_output, actual)

  def testCausalConvLayerFProp(self):
    with self.test_session(use_gpu=True) as sess:
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
    with self.test_session(use_gpu=True) as sess:
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
    with self.test_session(use_gpu=True):
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

  # TODO(yonghui): more test for convolution layer


class PoolingLayerTest(tf.test.TestCase):

  def testPoolLayerFProp(self):
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
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

        predicted_out_shape = pool_layer.OutShape(inputs1.shape)

        tf.global_variables_initializer().run()
        output1_v = output1.eval()
        self.assertAllClose(output2.eval(), output1_v)
        self.assertAllClose(predicted_out_shape.as_list(), output1_v.shape)


class ProjectionLayerTest(tf.test.TestCase):

  def testProjectionLayerConstruction(self):
    with self.test_session(use_gpu=True):
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
                           quantized=False):
    self._ClearCachedSession()
    tf.reset_default_graph()
    with self.test_session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.input_dim = input_dim
      params.output_dim = output_dim
      # Disable both activation and batch_norm.
      params.activation = activation
      params.batch_norm = batch_norm
      params.weight_norm = weight_norm
      params.affine_last = affine_last
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      if quantized:
        params.cc_schedule = quant_utils.FakeQuantizationSchedule.Params().Set(
            clip_end_step=1, quant_start_step=1)
      params.is_eval = False

      in_padding = tf.zeros([2, 4, 1], dtype=tf.float32)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float32)
      if reshape_to_2d:
        in_padding = tf.reshape(in_padding, [-1, 1])
        inputs = tf.reshape(inputs, [-1, 3])

      proj_layer = layers.ProjectionLayer(params)
      output = proj_layer.FPropDefaultTheta(inputs, in_padding)
      tf.global_variables_initializer().run()
      if quantized:
        # Put it in the fully quantized range.
        sess.run([proj_layer.PostTrainingStepUpdate(5)])
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
      actual = self._evalProjectionLayer(reshape_to_2d=reshape_to_2d)
      if reshape_to_2d:
        expected_output = np.reshape(np.array(expected_output), (-1, 2))
      tf.logging.info('expected = %s', expected_output)
      tf.logging.info('actual = %s', np.array_repr(actual))
      self.assertAllClose(expected_output, actual)

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
    with self.test_session(use_gpu=True) as sess:
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

  def testProjectionLayerFPropQuantized(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[-0.109375 ,  0.296875 ],
         [ 0.390625 ,  0.765625 ],
         [ 0.4140625, -0.1015625],
         [-0.6015625,  0.046875 ]],
        [[ 0.5859375, -0.28125  ],
         [ 0.34375  , -0.90625  ],
         [-0.1171875, -0.765625 ],
         [-0.7421875,  0.9140625]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    for reshape_to_2d in (False, True):
      # Note: Generally, quantized projections prefer a TANH activation.
      # We only test that here.
      actual = self._evalProjectionLayer(
          reshape_to_2d=reshape_to_2d, activation='TANH', quantized=True)
      if reshape_to_2d:
        expected_output = np.reshape(np.array(expected_output), (-1, 2))
      tf.logging.info('expected = %s', expected_output)
      tf.logging.info('actual = %s', np.array_repr(actual))
      self.assertAllClose(expected_output, actual)

  def testFCLayerConstruction(self):
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True) as sess:
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


class SoftmaxLayerTest(tf.test.TestCase):

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
                            fprop_dtype=None):
    if fprop_dtype is None:
      fprop_dtype = dtype
    with self.test_session(use_gpu=True, graph=tf.Graph()) as sess:
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
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)

      if default_qdomain is not None:
        params.qdomain.default = default_qdomain

      if num_samples > 0:
        # Turn on sampled soft-max; the asserts need to hold for it to be used.
        params.num_sampled = num_samples
        assert class_probabilities is None
        assert chunk_size is 0
        assert params.is_eval is not True

      params.vn.global_vn = False
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.FProp(
          py_utils.MaybeCastTheta(softmax.theta, params),
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
        step_op = softmax.PostTrainingStepUpdate(training_step)
        if step_op:
          sess.run([step_op])
      return sess.run(xent_loss)

  def testSimpleFullSoftmax_Sampled(self):
    xent_loss = self._RunSimpleFullSoftmax(num_samples=32, seed=12345)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.654818, 1e-5)
    self.assertNear(log_perplexity, 3.934008, 1e-5)

  def testSimpleFullSoftmax_SampledAndSharded(self):
    xent_loss = self._RunSimpleFullSoftmax(
        num_shards=4, num_samples=32, seed=12345)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.545459, 1e-5)
    self.assertNear(log_perplexity, 3.884299, 1e-5)

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
    with self.test_session(use_gpu=False) as sess:
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
    with self.test_session(use_gpu=False) as sess:
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
    with self.test_session(use_gpu=False) as sess:
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
    default_qdomain = quant_utils.SymetricScheduledClipQDomain.Params()
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
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
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


class FeedForwardNetTest(tf.test.TestCase):

  def testFeedForwardNetConstruction(self):
    with self.test_session(use_gpu=False):
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          dropout_prob=0.5,
          hidden_layer_dims=[20, 30],
          batch_norm=True,
          activation='TANH',
          params_init=py_utils.WeightInit.Uniform(1.0))
      proj_l = p.cls(p)
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)

      p = layers.FeedForwardNet.Params().Set(
          name='ffn2',
          input_dim=10,
          dropout_prob=[0.5, 0.1],
          hidden_layer_dims=[20, 30],
          batch_norm=True,
          activation='TANH',
          params_init=py_utils.WeightInit.Uniform(1.0))
      proj_l = p.cls(p)
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)

      p = layers.FeedForwardNet.Params().Set(
          name='ffn3',
          input_dim=10,
          dropout_prob=[0.5, 0.1],
          hidden_layer_dims=[20, 30],
          batch_norm=[True, False],
          activation=['TANH', 'RELU'],
          params_init=py_utils.WeightInit.Uniform(1.0))
      proj_l = p.cls(p)
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)

  def testFeedForwardNet(self):
    with self.test_session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          dropout_prob=0.0,
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

  def testFeedForwardNetWithSkipConnections(self):
    with self.test_session(use_gpu=False) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=20,
          hidden_layer_dims=[10, 10, 20, 25],
          dropout_prob=0.0,
          batch_norm=True,
          activation=['RELU', 'RELU', 'RELU', 'RELU'],
          skip_connections=[None, 'ResNet', 'DenseNet', 'DenseNet'])
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.cls(p)

      p1 = layers.ProjectionLayer.Params().Set(
          name='p1',
          input_dim=20,
          output_dim=10,
          activation='NONE',
          batch_norm=False)
      p1.params_init = params_init
      p1_l = p1.cls(p1)
      p_bn1 = layers.BatchNormLayer.Params().Set(
          name='p1', dim=10, params_init=params_init)
      bn1_l = p_bn1.cls(p_bn1)

      p2 = layers.ProjectionLayer.Params().Set(
          name='p2',
          input_dim=10,
          output_dim=10,
          activation='NONE',
          batch_norm=False)
      p2.params_init = params_init
      p2_l = p2.cls(p2)
      p_bn2 = layers.BatchNormLayer.Params().Set(
          name='p2', dim=10, params_init=params_init)
      bn2_l = p_bn2.cls(p_bn2)

      p3 = layers.ProjectionLayer.Params().Set(
          name='p3',
          input_dim=10,
          output_dim=10,
          activation='RELU',
          batch_norm=True)
      p3.params_init = params_init
      p3_l = p3.cls(p3)

      p4 = layers.ProjectionLayer.Params().Set(
          name='p4',
          input_dim=20,
          output_dim=5,
          activation='RELU',
          batch_norm=True)
      p4.params_init = params_init
      p4_l = p4.cls(p4)

      a = tf.constant(np.random.rand(5, 20), dtype=tf.float32)
      out1 = feedforward_net.FPropDefaultTheta(a)
      # skip = None
      l1_proj_out = p1_l.FPropDefaultTheta(a)
      l1_out = tf.nn.relu(bn1_l.FPropDefaultTheta(l1_proj_out))
      # skip = ResNet
      l2_proj_out = tf.add(l1_proj_out, p2_l.FPropDefaultTheta(l1_out))
      l2_out = tf.nn.relu(bn2_l.FPropDefaultTheta(l2_proj_out))
      # skip = ResNet
      l3_out = tf.concat([l2_out, p3_l.FPropDefaultTheta(l2_out)], axis=-1)
      # skip = DenseNet
      out2 = tf.concat([l3_out, p4_l.FPropDefaultTheta(l3_out)], axis=-1)

      tf.global_variables_initializer().run()
      out1_v, out2_v = sess.run([out1, out2])
      self.assertAllClose(out1_v, out2_v)

  def testFeedForwardNetSmokeTest(self):
    with self.test_session(use_gpu=False):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          dropout_prob=0.0,
          batch_norm=True,
          activation=['RELU', 'NONE'])
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.cls(p)
      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out = tf.reduce_sum(feedforward_net.FPropDefaultTheta(a))
      out_abs = tf.reduce_sum(tf.abs(feedforward_net.FPropDefaultTheta(a)))

      tf.global_variables_initializer().run()
      # pyformat: disable
      test_utils.CompareToGoldenSingleFloat(self, -0.000002, out.eval(), atol=1e-5)  # pylint: disable=line-too-long
      # pyformat: enable
      test_utils.CompareToGoldenSingleFloat(self, 126.990379, out_abs.eval())

  def testDropoutLayerTrain(self):
    with self.test_session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      p = layers.DropoutLayer.Params()
      p.keep_prob = 0.5
      p.seed = 1234
      p.name = 'dropout'

      dl = p.cls(p)

      x = tf.random_normal([10, 10, 10, 3])
      xd = dl.FPropDefaultTheta(x)
      x, xd = sess.run([x, xd])
      self.assertGreater((xd == 0).mean(), 0.3)
      self.assertLess((xd == 0).mean(), 0.7)
      self.assertAllClose(xd[xd != 0], x[xd != 0] / p.keep_prob)

  def testDropoutLayerEval(self):
    with self.test_session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      p = layers.DropoutLayer.Params()
      p.keep_prob = 0.5
      p.seed = 1234
      p.name = 'dropout'
      p.is_eval = True

      dl = p.cls(p)

      x = tf.random_normal([10, 10, 10, 3])
      xd = dl.FPropDefaultTheta(x)

      x, xd = sess.run([x, xd])

      self.assertAllEqual(xd, x)


class LayerNormTest(tf.test.TestCase):

  def testLayerNormFProp(self):
    with self.test_session(use_gpu=True) as sess:
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
    with self.test_session(use_gpu=True) as sess:
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


if __name__ == '__main__':
  tf.test.main()
