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
"""Tests for lingvo.core.conv_layers_with_time_padding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lingvo.core import conv_layers_with_time_padding
from lingvo.core import py_utils
from lingvo.core import test_utils


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
      params = conv_layers_with_time_padding.Conv2DLayerWithPadding.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.is_eval = False
      _ = params.cls(params)
      conv_vars = tf.get_collection('Conv2DLayerWithPadding_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/w/var:0']
      self.assertEqual(expected_var_names, conv_var_names)

  def testConv2DLayerOutShape(self):
    with self.session(use_gpu=True):
      tf.set_random_seed(398847392)
      np.random.seed(12345)
      params = conv_layers_with_time_padding.Conv2DLayerWithPadding.Params()
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

  def testConv2DLayerWithPaddingFProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)

      params = conv_layers_with_time_padding.Conv2DLayerWithPadding.Params()
      params.weight_norm = True
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      conv_layer = params.cls(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      out_sum = tf.reduce_sum(output)
      out_sum_squared = tf.reduce_sum(output * output)
      tf.global_variables_initializer().run()
      v1, v2 = sess.run([out_sum, out_sum_squared])
      tf.logging.info('actual = %f, %f', v1, v2)
      self.assertAllClose([-0.293671, 4.198602], [v1, v2])

  def testCausalConv2DLayerWithPaddingFProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)

      params = (
          conv_layers_with_time_padding.CausalConv2DLayerWithPadding.Params())
      params.weight_norm = True
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [2, 1, 3, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      conv_layer = params.cls(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      tf.global_variables_initializer().run()
      out_sum = tf.reduce_sum(output)
      out_sum_squared = tf.reduce_sum(output * output)
      tf.global_variables_initializer().run()
      v1, v2 = sess.run([out_sum, out_sum_squared])
      tf.logging.info('actual = %f, %f', v1, v2)
      self.assertAllClose([-3.584711, 3.324082], [v1, v2])

  def testDepthwiseConv2DLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)

      params = conv_layers_with_time_padding.DepthwiseConv2DLayer.Params()
      params.weight_norm = True
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      conv_layer = params.cls(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      tf.global_variables_initializer().run()
      out_sum = tf.reduce_sum(output)
      out_sum_squared = tf.reduce_sum(output * output)
      tf.global_variables_initializer().run()
      v1, v2 = sess.run([out_sum, out_sum_squared])
      tf.logging.info('actual = %f, %f', v1, v2)
      self.assertAllClose([-1.455162, 6.813269], [v1, v2])

  def testCausalDepthwiseConv2DLayer(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(398847392)
      np.random.seed(12345)

      params = conv_layers_with_time_padding.CausalDepthwiseConv2DLayer.Params()
      params.weight_norm = True
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [2, 1, 3, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = params.cls(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      tf.global_variables_initializer().run()
      tf.global_variables_initializer().run()
      out_sum = tf.reduce_sum(output)
      out_sum_squared = tf.reduce_sum(output * output)
      tf.global_variables_initializer().run()
      v1, v2 = sess.run([out_sum, out_sum_squared])
      tf.logging.info('actual = %f, %f', v1, v2)
      self.assertAllClose([-2.031689, 7.911201], [v1, v2])

  def testActivationLayer(self):
    with self.session(use_gpu=True) as sess:
      p = conv_layers_with_time_padding.ActivationLayer.Params()
      p.name = 'act'
      l = p.cls(p)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      in_padding = tf.zeros([2, 4], dtype=tf.float32)
      out, out_padding = l.FProp(l.theta, inputs, in_padding)
      tf.global_variables_initializer().run()
      v1, v2 = sess.run([out, out_padding])
      print(v1, v2)

  def _testNormalizedDepthwiseConv2DHelper(self,
                                           is_causal=False,
                                           dropconnect_prob=0):
    if is_causal:
      conv_cls = (
          conv_layers_with_time_padding.CausalNormalizedDepthwiseConv2DLayer)
    else:
      conv_cls = conv_layers_with_time_padding.NormalizedDepthwiseConv2DLayer
    tf.set_random_seed(398847392)
    np.random.seed(12345)
    params = conv_cls.Params().Set(
        name='conv',
        weight_tiling_factor=2,
        filter_shape=[3, 1, 2, 1],
        dropconnect_prob=dropconnect_prob,
        deterministic_dropout=True)
    conv_layer = conv_cls(params)
    in_padding = tf.zeros([2, 4], dtype=tf.float32)
    inputs = tf.constant(
        np.random.normal(0.1, 0.5, [2, 4, 1, 4]), dtype=tf.float32)
    output, _ = conv_layer.FPropDefaultTheta(inputs, in_padding)
    return output

  def testNormalizedDepthwiseConv2DLayerFProp(self):
    expected_output = [[0.91136134, 1.25781929, 1.76708317, 0.9021343],
                       [0.52296412, 0.7703352, 0.65711987, 0.23177178]]
    with self.session(use_gpu=True) as sess:
      output = self._testNormalizedDepthwiseConv2DHelper()
      output_sum = tf.squeeze(tf.reduce_sum(output, -1))
      tf.global_variables_initializer().run()
      output_sum_val = sess.run(output_sum)
    self.assertAllClose(expected_output, output_sum_val)

  def testCausalNormalizedDepthwiseConv2DLayerFProp(self):
    expected_output = [[0.00819603, 0.91136134, 1.25781929, 1.76708317],
                       [-0.07673456, 0.52296412, 0.7703352, 0.65711987]]
    with self.session(use_gpu=True) as sess:
      output = self._testNormalizedDepthwiseConv2DHelper(is_causal=True)
      output_sum = tf.squeeze(tf.reduce_sum(output, -1))
      tf.global_variables_initializer().run()
      output_sum_val = sess.run(output_sum)
    self.assertAllClose(expected_output, output_sum_val)

  def testNormalizedDepthwiseConv2DLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      output = self._testNormalizedDepthwiseConv2DHelper(dropconnect_prob=0.1)
      loss = tf.reduce_sum(output)
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      tf.global_variables_initializer().run()
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]
      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-02, atol=1e-02)

  def testCausualNormalizedDepthwiseConv2DLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      output = self._testNormalizedDepthwiseConv2DHelper(
          is_causal=True, dropconnect_prob=0.1)
      loss = tf.reduce_sum(output)
      all_vars = tf.trainable_variables()
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
