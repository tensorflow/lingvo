# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for conv layers builder."""


from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import conv_layers_builder
from lingvo.core import layers
from lingvo.core import test_utils
import numpy as np


class ConvPaddedLayersTest(test_utils.TestCase):

  def _ConvTestHelper(self, dilation, stride, activation, batch_norm,
                      weight_norm, in_dim, out_dim, filter_shape, conv_last,
                      causal_conv):
    with self.session(use_gpu=True) as sess:
      p1 = layers.Conv2DLayer.Params().Set(
          name='conv_2d01',
          filter_shape=filter_shape + [in_dim, out_dim],
          filter_stride=stride,
          dilation_rate=dilation,
          activation=activation,
          batch_norm=batch_norm,
          weight_norm=weight_norm,
          bias=not batch_norm,
          conv_last=conv_last,
          causal_convolution=causal_conv)
      builder_params = conv_layers_builder.Builder.Params().Set(
          use_bn=batch_norm, weight_norm=weight_norm)
      p2 = builder_params.Instantiate().Conv2D(
          'conv_2d02',
          in_dim,
          out_dim,
          filter_shape,
          stride=stride,
          dilation=dilation,
          activation=activation,
          conv_last=conv_last,
          is_causal=causal_conv)

      l1 = p1.Instantiate()
      l2 = p2.Instantiate()

      conv_in = tf.constant(np.random.normal(size=[4, 5, 6, 3]), tf.float32)
      conv_pad = np.full([4, 5], 0.0)
      conv_pad[2, 3] = 1.0
      conv_pad[2, 4] = 1.0
      conv_pad = tf.constant(conv_pad, tf.float32)
      conv_out1, out1_padding = l1.FProp(l1.theta, conv_in, conv_pad)
      conv_out2, out2_padding = l2.FProp(l2.theta, conv_in, conv_pad)

      tf.logging.info(l1.theta)
      tf.logging.info(l2.theta)
      l1_num_vars = l1.theta.Flatten()
      l2_num_var2 = l2.theta.Flatten()
      if len(l1_num_vars) != len(l2_num_var2):
        tf.logging.info(
            'Mismatched number of vars: l1: %d vars, l2: %d vars',
            len(l1_num_vars), len(l2_num_var2))

      w1 = l1.theta.w
      w2 = l2.theta.conv_2d.w
      # b1 = l1.theta.b
      # b2 = l2.theta.bn_or_bias.b

      tf.global_variables_initializer().run()
      v1, p1 = sess.run([conv_out1, out1_padding])
      w1_v = sess.run([w1])[0]
      v2, p2 = sess.run([conv_out2, out2_padding], feed_dict={w2: w1_v})

      self.assertAllClose(v1, v2)
      self.assertAllClose(p1, p2)

  def testConvBasic(self):
    dilation = [1, 1]
    stride = [2, 3]
    activation = 'NONE'
    batch_norm = False
    weight_norm = False
    in_dim = 3
    out_dim = 3
    filter_shape = [2, 2]
    conv_last = False
    causal_conv = False
    self._ConvTestHelper(dilation, stride, activation, batch_norm, weight_norm,
                         in_dim, out_dim, filter_shape, conv_last, causal_conv)

  def testConvBnWnTanh(self):
    dilation = [1, 1]
    stride = [2, 3]
    activation = 'TANH'
    batch_norm = True
    weight_norm = True
    in_dim = 3
    out_dim = 3
    filter_shape = [2, 2]
    conv_last = False
    causal_conv = False
    self._ConvTestHelper(dilation, stride, activation, batch_norm, weight_norm,
                         in_dim, out_dim, filter_shape, conv_last, causal_conv)

  def testConvLastWnTanh(self):
    dilation = [1, 1]
    stride = [2, 3]
    activation = 'TANH'
    batch_norm = False
    weight_norm = True
    in_dim = 3
    out_dim = 3
    filter_shape = [2, 2]
    conv_last = True
    causal_conv = False
    self._ConvTestHelper(dilation, stride, activation, batch_norm, weight_norm,
                         in_dim, out_dim, filter_shape, conv_last, causal_conv)

  def testConvLastCausal(self):
    dilation = [1, 1]
    stride = [2, 3]
    activation = 'TANH'
    batch_norm = True
    weight_norm = True
    in_dim = 3
    out_dim = 3
    filter_shape = [2, 1]
    conv_last = True
    causal_conv = True
    self._ConvTestHelper(dilation, stride, activation, batch_norm, weight_norm,
                         in_dim, out_dim, filter_shape, conv_last, causal_conv)

  def _DepthwiseConvTestHelper(self, dilation, stride, activation, batch_norm,
                               weight_norm, in_dim, depth_multiplier,
                               filter_shape, conv_last, causal_conv):
    with self.session(use_gpu=True) as sess:
      p1 = layers.DepthwiseConv2DLayer.Params().Set(
          name='conv_2d01',
          filter_shape=filter_shape + [in_dim, depth_multiplier],
          filter_stride=stride,
          dilation_rate=dilation,
          activation=activation,
          batch_norm=batch_norm,
          weight_norm=weight_norm,
          bias=not batch_norm,
          conv_last=conv_last,
          causal_convolution=causal_conv)
      builder_params = conv_layers_builder.Builder.Params().Set(
          use_bn=batch_norm, weight_norm=weight_norm)
      p2 = builder_params.Instantiate().DepthwiseConv2D(
          'conv_2d02',
          in_dim,
          depth_multiplier,
          filter_shape,
          stride=stride,
          activation=activation,
          dilation=dilation,
          conv_last=conv_last,
          is_causal=causal_conv)

      l1 = p1.Instantiate()
      l2 = p2.Instantiate()

      conv_in = tf.constant(np.random.normal(size=[4, 5, 6, 3]), tf.float32)
      conv_pad = np.full([4, 5], 0.0)
      conv_pad[2, 3] = 1.0
      conv_pad[2, 4] = 1.0
      conv_pad = tf.constant(conv_pad, tf.float32)
      conv_out1, out1_padding = l1.FProp(l1.theta, conv_in, conv_pad)
      conv_out2, out2_padding = l2.FProp(l2.theta, conv_in, conv_pad)

      tf.logging.info(l1.theta)
      tf.logging.info(l2.theta)
      l1_num_vars = l1.theta.Flatten()
      l2_num_var2 = l2.theta.Flatten()
      if len(l1_num_vars) != len(l2_num_var2):
        tf.logging.info(
            'Mismatched number of vars: l1: %d vars, l2: %d vars',
            len(l1_num_vars), len(l2_num_var2))

      w1 = l1.theta.w
      w2 = l2.theta.conv_2d.w
      # b1 = l1.theta.b
      # b2 = l2.theta.bn_or_bias.b

      tf.global_variables_initializer().run()
      v1, p1 = sess.run([conv_out1, out1_padding])
      w1_v = sess.run([w1])[0]
      v2, p2 = sess.run([conv_out2, out2_padding], feed_dict={w2: w1_v})

      self.assertAllClose(v1, v2)
      self.assertAllClose(p1, p2)

  def testDepthConvBasic(self):
    dilation = [1, 1]
    stride = [2, 2]
    activation = 'NONE'
    batch_norm = False
    weight_norm = False
    in_dim = 3
    depth_multiplier = 2
    filter_shape = [2, 2]
    conv_last = False
    causal_conv = False
    self._DepthwiseConvTestHelper(dilation, stride, activation, batch_norm,
                                  weight_norm, in_dim, depth_multiplier,
                                  filter_shape, conv_last, causal_conv)

  def testDepthConvBnWnTanh(self):
    dilation = [1, 1]
    stride = [2, 2]
    activation = 'TANH'
    batch_norm = True
    weight_norm = True
    in_dim = 3
    depth_multiplier = 3
    filter_shape = [2, 2]
    conv_last = False
    causal_conv = False
    self._DepthwiseConvTestHelper(dilation, stride, activation, batch_norm,
                                  weight_norm, in_dim, depth_multiplier,
                                  filter_shape, conv_last, causal_conv)

  def testDepthConvLastWnTanh(self):
    dilation = [1, 1]
    stride = [2, 2]
    activation = 'TANH'
    batch_norm = False
    weight_norm = True
    in_dim = 3
    depth_multiplier = 3
    filter_shape = [2, 2]
    conv_last = True
    causal_conv = False
    self._DepthwiseConvTestHelper(dilation, stride, activation, batch_norm,
                                  weight_norm, in_dim, depth_multiplier,
                                  filter_shape, conv_last, causal_conv)

  def testDepthConvLastCausal(self):
    dilation = [1, 1]
    stride = [2, 2]
    activation = 'TANH'
    batch_norm = True
    weight_norm = True
    in_dim = 3
    depth_multiplier = 3
    filter_shape = [2, 1]
    conv_last = True
    causal_conv = True
    self._DepthwiseConvTestHelper(dilation, stride, activation, batch_norm,
                                  weight_norm, in_dim, depth_multiplier,
                                  filter_shape, conv_last, causal_conv)

  def _SeparableConvTestHelper(self, dilation, stride, activation, batch_norm,
                               weight_norm, in_dim, depth_multiplier, out_dim,
                               filter_shape, conv_last, causal_conv,
                               assert_equality=True):
    with self.session(use_gpu=True) as sess:
      p1 = layers.SeparableConv2DLayer.Params().Set(
          name='conv_2d01',
          filter_shape=filter_shape + [in_dim, out_dim],
          depth_multiplier=depth_multiplier,
          filter_stride=stride,
          dilation_rate=dilation,
          activation=activation,
          batch_norm=batch_norm,
          weight_norm=weight_norm,
          bias=not batch_norm,
          conv_last=conv_last,
          causal_convolution=causal_conv)
      builder_params = conv_layers_builder.Builder.Params().Set(
          use_bn=batch_norm, weight_norm=weight_norm)
      p2 = builder_params.Instantiate().SeparableConv2D(
          'conv_2d02',
          in_dim,
          out_dim,
          depth_multiplier,
          filter_shape,
          stride=stride,
          activation=activation,
          dilation=dilation,
          conv_last=conv_last,
          is_causal=causal_conv)

      l1 = p1.Instantiate()
      l2 = p2.Instantiate()

      conv_in = tf.constant(np.random.normal(size=[4, 5, 6, 3]), tf.float32)
      conv_pad = np.full([4, 5], 0.0)
      conv_pad[2, 3] = 1.0
      conv_pad[2, 4] = 1.0
      conv_pad = tf.constant(conv_pad, tf.float32)
      conv_out1, out1_padding = l1.FProp(l1.theta, conv_in, conv_pad)
      conv_out2, out2_padding = l2.FProp(l2.theta, conv_in, conv_pad)

      tf.logging.info(l1.theta)
      tf.logging.info(l2.theta)
      l1_num_vars = l1.theta.Flatten()
      l2_num_var2 = l2.theta.Flatten()
      if len(l1_num_vars) != len(l2_num_var2):
        tf.logging.info(
            'Mismatched number of vars: l1: %d vars, l2: %d vars',
            len(l1_num_vars), len(l2_num_var2))

      pointwise_conv_w1 = l1.theta.w
      depth_conv_w1 = l1.theta.depthwise_conv.w
      pointwise_conv_w2 = l2.theta.conv_1x1.w
      depth_conv_w2 = l2.theta.conv_2d.w
      # b1 = l1.theta.b
      # b2 = l2.theta.bn_or_bias.b
      tf.global_variables_initializer().run()
      v1, p1 = sess.run([conv_out1, out1_padding])
      p_w1_v, d_w1_v = sess.run([pointwise_conv_w1, depth_conv_w1])
      v2, p2 = sess.run([conv_out2, out2_padding],
                        feed_dict={
                            pointwise_conv_w2: p_w1_v,
                            depth_conv_w2: d_w1_v
                        })

    if assert_equality:
      self.assertAllClose(v1, v2)
      self.assertAllClose(p1, p2)

  def testSeparableConv2DLayerBasic(self):
    dilation = [1, 1]
    stride = [2, 2]
    activation = 'NONE'
    batch_norm = False
    weight_norm = False
    in_dim = 3
    depth_multiplier = 3
    out_dim = 2
    filter_shape = [2, 2]
    conv_last = False
    causal_conv = False
    self._SeparableConvTestHelper(dilation, stride, activation, batch_norm,
                                  weight_norm, in_dim, depth_multiplier,
                                  out_dim, filter_shape, conv_last, causal_conv)

  def testSeparableConvWnWnTanh(self):
    dilation = [1, 1]
    stride = [2, 2]
    activation = 'TANH'
    batch_norm = False
    weight_norm = True
    in_dim = 3
    depth_multiplier = 3
    out_dim = 2
    filter_shape = [2, 1]
    conv_last = False
    causal_conv = True
    self._SeparableConvTestHelper(dilation, stride, activation, batch_norm,
                                  weight_norm, in_dim, depth_multiplier,
                                  out_dim, filter_shape, conv_last, causal_conv)

  def testSeparableConvLastBnWnTanh(self):
    dilation = [1, 1]
    stride = [2, 2]
    activation = 'TANH'
    batch_norm = True
    weight_norm = True
    in_dim = 3
    depth_multiplier = 3
    out_dim = 2
    filter_shape = [2, 1]
    conv_last = True
    causal_conv = True
    # New implementation is not equivallent to the old.
    self._SeparableConvTestHelper(dilation, stride, activation, batch_norm,
                                  weight_norm, in_dim, depth_multiplier,
                                  out_dim, filter_shape, conv_last, causal_conv,
                                  assert_equality=False)


class CausalPoolingLayerTest(test_utils.TestCase, parameterized.TestCase):
  """Tests for CausalPoolingLayer."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'max_pooling',
          'pooling_type': 'MAX',
          'left_context': 2,
          'inputs': np.array([-2, 0, 2, 4, 0, 0]),
          'input_paddings': np.array([0, 0, 0, 0, 1, 1]),
          'expected_output': np.array([-2, 0, 2, 4, 0, 0]),
          'expected_output_padding': np.array([0, 0, 0, 0, 1, 1]),
      }, {
          'testcase_name': 'avg_pooling',
          'pooling_type': 'AVG',
          'left_context': 2,
          'inputs': np.array([-2, 0, 2, 4, 0, 0]),
          'input_paddings': np.array([0, 0, 0, 0, 1, 1]),
          'expected_output': np.array([-2, -1, 1, 3, 0, 0]),
          'expected_output_padding': np.array([0, 0, 0, 0, 1, 1]),
      }, {
          'testcase_name': 'max_pooling_large_window',
          'pooling_type': 'MAX',
          'left_context': 10,
          'inputs': np.array([-2, 0, 2, 4, 0, 0]),
          'input_paddings': np.array([0, 0, 0, 0, 1, 1]),
          'expected_output': np.array([-2, 0, 2, 4, 0, 0]),
          'expected_output_padding': np.array([0, 0, 0, 0, 1, 1]),
      }, {
          'testcase_name': 'avg_pooling_large_window',
          'pooling_type': 'AVG',
          'left_context': 10,
          'inputs': np.array([-2, 0, 2, 4, 0, 0]),
          'input_paddings': np.array([0, 0, 0, 0, 1, 1]),
          'expected_output': np.array([-2, -1, 0, 1, 0, 0]),
          'expected_output_padding': np.array([0, 0, 0, 0, 1, 1]),
      })
  def testSimpleCase(self, pooling_type, left_context, inputs, input_paddings,
                     expected_output, expected_output_padding):
    inputs = inputs[np.newaxis, :, np.newaxis, np.newaxis]
    input_paddings = input_paddings[np.newaxis, :]
    param = conv_layers_builder.CausalPoolingLayer.Params().Set(
        name='test_layer', pooling_type=pooling_type, left_context=left_context)
    pooling_layer = param.Instantiate()
    with self.session(use_gpu=True) as sess:
      inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
      input_paddings = tf.convert_to_tensor(input_paddings, dtype=tf.float32)
      output, output_paddings = pooling_layer.FPropDefaultTheta(
          inputs, input_paddings)
      tf.global_variables_initializer().run()
      output_val, output_paddings_val = sess.run([output, output_paddings])

    self.assertAllClose(expected_output, output_val.flatten())
    self.assertAllEqual(expected_output_padding, output_paddings_val.flatten())


if __name__ == '__main__':
  tf.test.main()
