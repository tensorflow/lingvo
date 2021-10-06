# Lint as: python3
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
"""Tests for lingvo.core.conv_layers."""

from absl.testing import flagsaver
from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import conv_layers_with_time_padding as conv_layers
from lingvo.core import py_utils
from lingvo.core import stream_step_test_base
from lingvo.core import test_utils
from lingvo.core import tshape
import numpy as np


class ConvLayerTest(parameterized.TestCase, test_utils.TestCase):
  """Tests conv layers.

  Note that there are multiple subclasses of BaseConv2DLayer and most cases
  are tested via the concrete Conv2DLayer. Other tests are done against
  other subclasses to cover key differences.
  """

  def testConv2DLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = conv_layers.Conv2DLayerWithPadding.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      _ = params.Instantiate()
      conv_vars = tf.get_collection('Conv2DLayerWithPadding_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/w/var:0']
      self.assertEqual(expected_var_names, conv_var_names)

  def testConv2DLayerWithPaddingOutputChannels(self):
    with self.session():
      params = conv_layers.Conv2DLayerWithPadding.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      actual_output_channels = params.cls.OutputChannels(params)
      self.assertEqual(32, actual_output_channels)

  @parameterized.parameters(False, True)
  def testConv2DLayerOutShape(self, v2_padding):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = conv_layers.Conv2DLayerWithPadding.Params()
      params.v2_padding = v2_padding
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      conv_layer = params.Instantiate()
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 5, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 10, 5, 32])

  # ComputeConvOutputPadding is broken for strided convolutions. Below we mark
  # cases that are in correct with "Bug". testComputeConvOutputPaddingV2 below
  # has the same test cases but without the bugs.
  @parameterized.parameters(
      ([0, 0, 0, 1], [0, 0, 0, 1], 1, 'SAME'),
      ([0, 0, 0, 0], [0, 0], 2, 'SAME'),
      ([0, 0, 0, 1], [0, 1], 2, 'SAME'),
      ([0, 0, 1, 1], [1, 1], 2, 'SAME'),  # Bug
      ([0, 0, 0, 0, 0], [0, 0, 1], 2, 'SAME'),  # Bug
      ([0, 0, 0, 0, 1], [0, 1, 1], 2, 'SAME'),  # Bug
      ([0, 0, 0, 1, 1], [0, 1, 1], 2, 'SAME'),  # Bug
      ([0, 0, 1, 1, 1], [1, 1, 1], 2, 'SAME'),  # Bug
      ([0, 0, 0, 0, 0, 0], [0, 0, 0], 2, 'SAME'),
      ([0, 0, 0, 0, 0, 1], [0, 0, 1], 2, 'SAME'),
      ([0, 0, 0, 0, 1, 1], [0, 1, 1], 2, 'SAME'),  # Bug
      ([0, 0, 0, 1, 1, 1], [0, 1, 1], 2, 'SAME'),
      ([0, 0, 1, 1, 1, 1], [1, 1, 1], 2, 'SAME'),
      ([0, 0, 0, 0], [0, 0, 0, 0], 1, 'VALID'),  # Bug
      ([0, 0, 0, 1], [0, 0, 0, 1], 1, 'VALID'),  # Bug
      ([0, 0, 0, 0], [0], 2, 'VALID'),
      ([0, 0, 0, 1], [0], 2, 'VALID'),
      ([0, 0, 1, 1], [1], 2, 'VALID'),
      ([0, 0, 0, 0, 0], [0, 0], 2, 'VALID'),
      ([0, 0, 0, 0, 1], [0, 1], 2, 'VALID'),
      ([0, 0, 0, 1, 1], [0, 1], 2, 'VALID'),
      ([0, 0, 1, 1, 1], [1, 1], 2, 'VALID'),
      ([0, 0, 0, 0, 0, 0], [0, 0], 2, 'VALID'),
      ([0, 0, 0, 0, 0, 1], [0, 0], 2, 'VALID'),
      ([0, 0, 0, 0, 1, 1], [0, 1], 2, 'VALID'),
      ([0, 0, 0, 1, 1, 1], [0, 1], 2, 'VALID'),
      ([0, 0, 1, 1, 1, 1], [1, 1], 2, 'VALID'),
  )
  def testComputeConvOutputPadding(self, padding, expected_padding, stride,
                                   padding_algorithm):
    """Tests padding behavior. There are multiple bugs in the implementation."""
    padding = tf.constant([padding], tf.float32)
    expected_padding = tf.constant([expected_padding], tf.float32)
    with self.session(use_gpu=True):
      conv_padding = conv_layers.ComputeConvOutputPadding(
          padding, window=3, stride=stride, padding_algorithm=padding_algorithm)
      self.evaluate(tf.global_variables_initializer())
      conv_padding = py_utils.Debug(conv_padding)
      conv_padding = self.evaluate(conv_padding)
      tf.logging.info('expected_padding {expected_padding}')
      self.assertAllClose(expected_padding, conv_padding)

  @parameterized.parameters(
      ([0, 0, 0, 1], [0, 0, 0, 1], 1, 'SAME'),
      ([0, 0, 0, 0], [0, 0], 2, 'SAME'),
      ([0, 0, 0, 1], [0, 0], 2, 'SAME'),
      ([0, 0, 1, 1], [0, 1], 2, 'SAME'),
      ([0, 0, 0, 0, 0], [0, 0, 0], 2, 'SAME'),
      ([0, 0, 0, 0, 1], [0, 0, 1], 2, 'SAME'),
      ([0, 0, 0, 1, 1], [0, 0, 1], 2, 'SAME'),
      ([0, 0, 1, 1, 1], [0, 1, 1], 2, 'SAME'),
      ([0, 0, 0, 0, 0, 0], [0, 0, 0], 2, 'SAME'),
      ([0, 0, 0, 0, 0, 1], [0, 0, 0], 2, 'SAME'),
      ([0, 0, 0, 0, 1, 1], [0, 0, 1], 2, 'SAME'),
      ([0, 0, 0, 1, 1, 1], [0, 0, 1], 2, 'SAME'),
      ([0, 0, 1, 1, 1, 1], [0, 1, 1], 2, 'SAME'),
      ([0, 0, 0, 0], [0, 0], 1, 'VALID'),
      ([0, 0, 0, 1], [0, 1], 1, 'VALID'),
      ([0, 0, 0, 0], [0], 2, 'VALID'),
      ([0, 0, 0, 1], [0], 2, 'VALID'),
      ([0, 0, 1, 1], [1], 2, 'VALID'),
      ([0, 0, 0, 0, 0], [0, 0], 2, 'VALID'),
      ([0, 0, 0, 0, 1], [0, 1], 2, 'VALID'),
      ([0, 0, 0, 1, 1], [0, 1], 2, 'VALID'),
      ([0, 0, 1, 1, 1], [1, 1], 2, 'VALID'),
      ([0, 0, 0, 0, 0, 0], [0, 0], 2, 'VALID'),
      ([0, 0, 0, 0, 0, 1], [0, 0], 2, 'VALID'),
      ([0, 0, 0, 0, 1, 1], [0, 1], 2, 'VALID'),
      ([0, 0, 0, 1, 1, 1], [0, 1], 2, 'VALID'),
      ([0, 0, 1, 1, 1, 1], [1, 1], 2, 'VALID'),
  )
  def testComputeConvOutputPaddingV2(self, padding, expected_padding, stride,
                                     padding_algorithm):
    """Test Convolution padding computation."""
    padding = tf.constant([padding], tf.float32)
    expected_padding = tf.constant([expected_padding], tf.float32)
    with self.session(use_gpu=True):
      conv_padding = conv_layers._ComputeConvOutputPaddingV2(
          padding, window=3, stride=stride, padding_algorithm=padding_algorithm)
      self.evaluate(tf.global_variables_initializer())
      conv_padding = py_utils.Debug(conv_padding)
      conv_padding = self.evaluate(conv_padding)
      tf.logging.info('expected_padding {expected_padding}')
      self.assertAllClose(expected_padding, conv_padding)

  @parameterized.parameters(5, 6)
  def testConv2DLayerStridedWithPaddingFProp(self, seq_len):
    """Check strided convs get the same values for different length dim."""
    # TODO(isaace): THIS TEST SHOWS THAT THERE IS A BUG IN THE CODE.
    with self.session(use_gpu=True):
      batch_size = 3
      expected_seq_len = 3

      params = conv_layers.Conv2DLayerWithPadding.Params()
      params.weight_norm = False
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [3, 3, 1, 1]
      params.params_init = py_utils.WeightInit.Constant(1.0)
      conv_layer = params.Instantiate()

      # Set up the padding for the sequence length. (starting at 5).
      in_padding = tf.constant([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1],
          [0, 0, 0, 1, 1],
      ], tf.float32)
      in_padding = tf.pad(
          in_padding, [[0, 0], [0, seq_len - 5]], constant_values=1.0)

      inputs = 1.0 + tf.tile(
          tf.reshape(tf.range(seq_len, dtype=tf.float32), [1, seq_len, 1, 1]),
          [batch_size, 1, 3, 1])
      inputs = py_utils.ApplyPadding(
          tf.reshape(in_padding, [batch_size, seq_len, 1, 1]), inputs)

      inputs = py_utils.Debug(inputs)

      output, out_padding = conv_layer.FPropDefaultTheta(inputs, in_padding)

      output = py_utils.Debug(output)
      out_padding = py_utils.Debug(out_padding)

      self.evaluate(tf.global_variables_initializer())
      output, out_padding = self.evaluate([output, out_padding])

      self.assertEqual((batch_size, expected_seq_len, 2, 1), output.shape)
      self.assertAllClose([
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 1],
      ], out_padding)

      # This here shows a bug in the implementation; the output should be the
      # same. Also there are bugs with the output not having the correct
      # padding.
      if seq_len == 5:
        self.assertAllClose([
            [[[6], [6]], [[18], [18]], [[18], [18]]],
            [[[6], [6]], [[18], [18]], [[8], [8]]],
            [[[6], [6]], [[10], [10]], [[0], [0]]],
        ], output)
      elif seq_len == 6:
        self.assertAllClose([
            [[[12], [12]], [[24], [24]], [[10], [10]]],
            [[[12], [12]], [[14], [14]], [[0], [0]]],
            [[[12], [12]], [[6], [6]], [[0], [0]]],
        ], output)
      else:
        raise ValueError('Test does not handle length {seq_len}')

  @parameterized.parameters(5, 6)
  def testConv2DLayerStridedWithPaddingFPropV2(self, seq_len):
    """Check strided convs get the same values for different seq_len."""
    with self.session(use_gpu=True):
      batch_size = 3
      expected_seq_len = 3

      params = conv_layers.Conv2DLayerWithPadding.Params()
      params.v2_padding = True
      params.weight_norm = False
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [3, 3, 1, 1]
      params.params_init = py_utils.WeightInit.Constant(1.0)
      conv_layer = params.Instantiate()

      # Set up the padding for the sequence length. (starting at 5).
      in_padding = tf.constant([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1],
          [0, 0, 0, 1, 1],
      ], tf.float32)
      in_padding = tf.pad(
          in_padding, [[0, 0], [0, seq_len - 5]], constant_values=1.0)

      inputs = 1.0 + tf.tile(
          tf.reshape(tf.range(seq_len, dtype=tf.float32), [1, seq_len, 1, 1]),
          [batch_size, 1, 3, 1])
      inputs = py_utils.ApplyPadding(
          tf.reshape(in_padding, [batch_size, seq_len, 1, 1]), inputs)

      inputs = py_utils.Debug(inputs)

      output, out_padding = conv_layer.FPropDefaultTheta(inputs, in_padding)

      output = py_utils.Debug(output)
      out_padding = py_utils.Debug(out_padding)

      self.evaluate(tf.global_variables_initializer())
      output, out_padding = self.evaluate([output, out_padding])

      self.assertEqual((batch_size, expected_seq_len, 2, 1), output.shape)
      self.assertAllClose([
          [0, 0, 0],
          [0, 0, 1],
          [0, 0, 1],
      ], out_padding)

      # Explanation of some computations (0s are padded)
      # 6 = (1*0 + 1*0 + 1*0) + (1*0 + 1*1 + 1*1) + (1*0 + 1*2 + 1*2)
      # 18 = (1*1 + 1*1 + 1*0) + (1*3 + 1*3 + 1*0) + (1*5 + 1*5 + 1*0)
      self.assertAllClose(
          [
              [[[6], [6]], [[18], [18]], [[18], [18]]],
              [[[6], [6]], [[18], [18]], [[8], [8]]],  # NOTE: Not padded.
              [[[6], [6]], [[10], [10]], [[0], [0]]],
          ],
          output)

  def testConv2DLayerWithPaddingFPropRandom(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)

      params = conv_layers.Conv2DLayerWithPadding.Params()
      params.weight_norm = True
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      conv_layer = params.Instantiate()
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      out_sum = tf.reduce_sum(output)
      out_sum_squared = tf.reduce_sum(output * output)
      self.evaluate(tf.global_variables_initializer())
      v1, v2 = self.evaluate([out_sum, out_sum_squared])
      tf.logging.info('actual = %f, %f', v1, v2)
      self.assertAllClose([-0.293671, 4.198602], [v1, v2])

  @parameterized.parameters(5, 6)
  def testCausalConv2DLayerStridedWithPaddingFProp(self, seq_len):
    """Check strided convs get the same values for different length dim."""
    # TODO(isaace): THIS TEST SHOWS THAT THERE IS A BUG WITH PADDING
    with self.session(use_gpu=True):
      batch_size = 5
      expected_seq_len = 3

      params = conv_layers.CausalConv2DLayerWithPadding.Params()
      params.weight_norm = False
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [3, 1, 1, 1]
      params.params_init = py_utils.WeightInit.Constant(1.0)
      conv_layer = params.Instantiate()

      # Set up the padding for the sequence length. (starting at 5).
      in_padding = tf.constant([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1],
          [0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1],
          [0, 1, 1, 1, 1],
      ], tf.float32)
      in_padding = tf.pad(
          in_padding, [[0, 0], [0, seq_len - 5]], constant_values=1.0)

      inputs = 1.0 + tf.tile(
          tf.reshape(tf.range(seq_len, dtype=tf.float32), [1, seq_len, 1, 1]),
          [batch_size, 1, 3, 1])
      inputs = py_utils.ApplyPadding(
          tf.reshape(in_padding, [batch_size, seq_len, 1, 1]), inputs)

      inputs = py_utils.Debug(inputs)

      output, out_padding = conv_layer.FPropDefaultTheta(inputs, in_padding)

      output = py_utils.Debug(output)
      out_padding = py_utils.Debug(out_padding)

      self.evaluate(tf.global_variables_initializer())
      output, out_padding = self.evaluate([output, out_padding])

      self.assertEqual((batch_size, expected_seq_len, 2, 1), output.shape)
      self.assertAllClose([
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 1],
          [0, 1, 1],
          [1, 1, 1],
      ], out_padding)

      # NOTE: There is a bug in the output not being padded correctly.
      self.assertAllClose([
          [[[1], [1]], [[6], [6]], [[12], [12]]],
          [[[1], [1]], [[6], [6]], [[7], [7]]],
          [[[1], [1]], [[6], [6]], [[3], [3]]],
          [[[1], [1]], [[3], [3]], [[0], [0]]],
          [[[1], [1]], [[1], [1]], [[0], [0]]],
      ], output)

  @parameterized.parameters(5, 6)
  def testCausalConv2DLayerStridedWithPaddingFPropV2(self, seq_len):
    """Check strided convs get the same values for different length dim."""
    with self.session(use_gpu=True):
      batch_size = 5
      expected_seq_len = 3

      params = conv_layers.CausalConv2DLayerWithPadding.Params()
      params.v2_padding = True
      params.weight_norm = False
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [3, 1, 1, 1]
      params.params_init = py_utils.WeightInit.Constant(1.0)
      conv_layer = params.Instantiate()

      # Set up the padding for the sequence length. (starting at 5).
      in_padding = tf.constant([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1],
          [0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1],
          [0, 1, 1, 1, 1],
      ], tf.float32)
      in_padding = tf.pad(
          in_padding, [[0, 0], [0, seq_len - 5]], constant_values=1.0)

      inputs = 1.0 + tf.tile(
          tf.reshape(tf.range(seq_len, dtype=tf.float32), [1, seq_len, 1, 1]),
          [batch_size, 1, 3, 1])
      inputs = py_utils.ApplyPadding(
          tf.reshape(in_padding, [batch_size, seq_len, 1, 1]), inputs)

      inputs = py_utils.Debug(inputs)

      output, out_padding = conv_layer.FPropDefaultTheta(inputs, in_padding)

      output = py_utils.Debug(output)
      out_padding = py_utils.Debug(out_padding)

      self.evaluate(tf.global_variables_initializer())
      output, out_padding = self.evaluate([output, out_padding])

      self.assertEqual((batch_size, expected_seq_len, 2, 1), output.shape)
      self.assertAllClose([
          [0, 0, 0],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 1],
          [0, 1, 1],
      ], out_padding)

      self.assertAllClose(
          [
              [[[1], [1]], [[6], [6]], [[12], [12]]],
              [[[1], [1]], [[6], [6]], [[7], [7]]],
              [[[1], [1]], [[6], [6]], [[3], [3]]],  # NOTE: not padded.
              [[[1], [1]], [[3], [3]], [[0], [0]]],
              [[[1], [1]], [[1], [1]], [[0], [0]]],
          ],
          output)

  def testCausalConv2DLayerWithPaddingFPropRandom(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)

      params = (conv_layers.CausalConv2DLayerWithPadding.Params())
      params.weight_norm = True
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [2, 1, 3, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      conv_layer = params.Instantiate()
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      self.evaluate(tf.global_variables_initializer())
      out_sum = tf.reduce_sum(output)
      out_sum_squared = tf.reduce_sum(output * output)
      self.evaluate(tf.global_variables_initializer())
      v1, v2 = self.evaluate([out_sum, out_sum_squared])
      tf.logging.info('actual = %f, %f', v1, v2)
      self.assertAllClose([-3.584711, 3.324082], [v1, v2])

  def testDepthwiseConv2DLayerOutputChannels(self):
    with self.session():
      params = conv_layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.bias = True
      actual_output_channels = params.cls.OutputChannels(params)
      self.assertEqual(6, actual_output_channels)

  def testDepthwiseConv2DLayerFProp(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)

      params = conv_layers.DepthwiseConv2DLayer.Params()
      params.weight_norm = True
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      conv_layer = params.Instantiate()
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      self.evaluate(tf.global_variables_initializer())
      out_sum = tf.reduce_sum(output)
      out_sum_squared = tf.reduce_sum(output * output)
      self.evaluate(tf.global_variables_initializer())
      v1, v2 = self.evaluate([out_sum, out_sum_squared])
      tf.logging.info('actual = %f, %f', v1, v2)
      self.assertAllClose([-1.455162, 6.813269], [v1, v2])

  def testCausalDepthwiseConv2DLayer(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)

      params = conv_layers.CausalDepthwiseConv2DLayer.Params()
      params.weight_norm = True
      params.filter_stride = [2, 2]
      params.name = 'conv'
      params.filter_shape = [2, 1, 3, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = params.Instantiate()
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      self.evaluate(tf.global_variables_initializer())
      self.evaluate(tf.global_variables_initializer())
      out_sum = tf.reduce_sum(output)
      out_sum_squared = tf.reduce_sum(output * output)
      self.evaluate(tf.global_variables_initializer())
      v1, v2 = self.evaluate([out_sum, out_sum_squared])
      tf.logging.info('actual = %f, %f', v1, v2)
      self.assertAllClose([-2.031689, 7.911201], [v1, v2])

  def testActivationLayer(self):
    with self.session(use_gpu=True):
      p = conv_layers.ActivationLayer.Params()
      p.name = 'act'
      l = p.Instantiate()
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      in_padding = tf.zeros([2, 4], dtype=tf.float32)
      out, out_padding = l.FProp(l.theta, inputs, in_padding)
      self.evaluate(tf.global_variables_initializer())
      v1, v2 = self.evaluate([out, out_padding])
      print(v1, v2)

  def _testNormalizedDepthwiseConv2DHelper(self,
                                           is_causal=False,
                                           dropconnect_prob=0):
    if is_causal:
      conv_cls = (conv_layers.CausalNormalizedDepthwiseConv2DLayer)
    else:
      conv_cls = conv_layers.NormalizedDepthwiseConv2DLayer
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    params = conv_cls.Params().Set(
        name='conv',
        weight_tiling_factor=2,
        filter_shape=[3, 1, 2, 1],
        dropconnect_prob=dropconnect_prob,
        deterministic_dropout=True)
    conv_layer = params.Instantiate()
    in_padding = tf.zeros([2, 4], dtype=tf.float32)
    inputs = tf.constant(
        np.random.normal(0.1, 0.5, [2, 4, 1, 4]), dtype=tf.float32)
    output, _ = conv_layer.FPropDefaultTheta(inputs, in_padding)
    return output

  def testNormalizedDepthwiseConv2DLayerOutputChannels(self):
    with self.session():
      params = (conv_layers.NormalizedDepthwiseConv2DLayer.Params())
      params.name = 'conv'
      params.filter_shape = [3, 1, 2, 1]
      params.weight_tiling_factor = 2
      actual_output_channels = params.cls.OutputChannels(params)
      self.assertEqual(4, actual_output_channels)

  def testNormalizedDepthwiseConv2DLayerFPropMeta(self):
    params = (conv_layers.NormalizedDepthwiseConv2DLayer.Params())
    params.name = 'conv'
    params.filter_shape = [3, 1, 2, 1]
    params.weight_tiling_factor = 2
    batch, time, frequency, in_channel = 2, 4, 1, 4
    output_channels = 4
    inputs_shape = tshape.Shape([batch, time, frequency, in_channel])
    paddings_shape = tshape.Shape([batch, time])
    with self.session():
      out = params.cls.FPropMeta(params, inputs_shape, paddings_shape)
      expected_flops = batch * time * frequency * params.filter_shape[
          0] * output_channels * 5
      self.assertEqual(expected_flops, out.flops)
      out_shapes = out.out_shapes
      self.assertEqual(out_shapes[0].ToTensorShape().as_list(),
                       [batch, time, frequency, output_channels])
      self.assertEqual(out_shapes[1].ToTensorShape().as_list(), [batch, time])

  def testNormalizedDepthwiseConv2DLayerFProp(self):
    expected_output = [[0.91136134, 1.25781929, 1.76708317, 0.9021343],
                       [0.52296412, 0.7703352, 0.65711987, 0.23177178]]
    with self.session(use_gpu=True):
      output = self._testNormalizedDepthwiseConv2DHelper()
      output_sum = tf.squeeze(tf.reduce_sum(output, -1))
      self.evaluate(tf.global_variables_initializer())
      output_sum_val = self.evaluate(output_sum)
    self.assertAllClose(expected_output, output_sum_val)

  def testCausalNormalizedDepthwiseConv2DLayerFProp(self):
    expected_output = [[0.00819603, 0.91136134, 1.25781929, 1.76708317],
                       [-0.07673456, 0.52296412, 0.7703352, 0.65711987]]
    with self.session(use_gpu=True):
      output = self._testNormalizedDepthwiseConv2DHelper(is_causal=True)
      output_sum = tf.squeeze(tf.reduce_sum(output, -1))
      self.evaluate(tf.global_variables_initializer())
      output_sum_val = self.evaluate(output_sum)
    self.assertAllClose(expected_output, output_sum_val)

  def testNormalizedDepthwiseConv2DLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      output = self._testNormalizedDepthwiseConv2DHelper(dropconnect_prob=0.1)
      loss = tf.reduce_sum(output)
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      self.evaluate(tf.global_variables_initializer())
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
      self.evaluate(tf.global_variables_initializer())
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]
      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-02, atol=1e-02)


class CausalDepthwiseConv2DLayerStreamStepTest(
    stream_step_test_base.StreamStepTestBase):

  @property
  def input_rank(self):
    return 4

  def _GetParams(self, **kwargs):
    channel = kwargs['input_dim']
    channel_multiplier = kwargs['channel_multiplier']
    kernel = kwargs['kernel']
    bias = kwargs['bias']
    p = conv_layers.CausalDepthwiseConv2DLayer.Params().Set(
        name='conv',
        filter_stride=[1, 1],
        filter_shape=[kernel, 1, channel, channel_multiplier],
        params_init=py_utils.WeightInit.Gaussian(0.1),
        bias=bias,
        bias_init=py_utils.WeightInit.Gaussian(0.1))
    return p

  def _FProp(self, layer, inputs, paddings):
    return layer.FProp(layer.theta, inputs, paddings)

  def _GetFPropOutput(self, fprop_out):
    return fprop_out[0]

  @parameterized.named_parameters(
      ('Basic',),
      ('BasicS2', False, 2),
      ('BasicBias', False, 1, True),
      ('SkipNorm', True),
      ('SkipNormS4', True, 4),
  )
  def testCommon(self, testonly_skip_norm_layers=False, stride=1, bias=False):
    kwargs = dict(
        input_dim=3, kernel=5, stride=stride, channel_multiplier=1, bias=bias)
    with flagsaver.flagsaver(
        testonly_skip_norm_layers=testonly_skip_norm_layers):
      self._TestStreamStepHelper(**kwargs)

  @parameterized.named_parameters(
      ('Basic',),
      ('S2', 2),
      ('S4', 4),
      ('SkipNormS4', 4, True),
  )
  def testLeadingPaddings(self, stride=1, skip_norm=False):
    with flagsaver.flagsaver(testonly_skip_norm_layers=skip_norm):
      self._TestLeadingPaddingsHelper(stride)

  def _TestLeadingPaddingsHelper(self, stride=1):
    """Tests leading paddings case, useful for local atten with right ctx."""
    batch, max_seqlen, channel = 2, 16, 2
    kernel, channel_multiplier = 3, 2

    p = conv_layers.CausalDepthwiseConv2DLayer.Params().Set(
        name='conv',
        filter_stride=[1, 1],
        filter_shape=[kernel, 1, channel, channel_multiplier],
        params_init=py_utils.WeightInit.Gaussian(0.1))

    l = p.Instantiate()
    init_op = tf.global_variables_initializer()

    np.random.seed(None)
    inputs = np.random.normal(0.1, 0.5, [batch, max_seqlen, 1, channel]).astype(
        np.float32)
    print(f'np.sum(inputs): {np.sum(inputs)}')
    inputs_t = tf.convert_to_tensor(inputs)

    # The upperbound is always max_seqlen-1, so the batch is always padded.
    seqlen = np.random.randint(
        low=1, high=max_seqlen, size=(batch,), dtype=np.int32)
    print(f'seqlen: {seqlen}')
    paddings = py_utils.PaddingsFromLengths(
        tf.convert_to_tensor(seqlen), max_seqlen)

    shift_inputs = np.array(inputs)
    for i in range(batch):
      shift_inputs[i] = np.roll(shift_inputs[i], max_seqlen - seqlen[i], axis=0)
    shift_inputs_t = tf.convert_to_tensor(shift_inputs)

    # Has the same number of tokens as paddings per example
    leading_paddings = 1 - py_utils.PaddingsFromLengths(
        max_seqlen - tf.convert_to_tensor(seqlen), max_seqlen)

    def expand_pad(pad):  # pylint:disable=invalid-name
      return py_utils.AppendDims(pad, 2)

    def stream(l, inputs, paddings):  # pylint:disable=invalid-name
      state = l.zero_state(batch)
      all_outs = []
      for i in range(max_seqlen // stride):
        step_inputs = inputs[:, stride * i:stride * (i + 1)]
        step_paddings = paddings[:, stride * i:stride * (i + 1)]
        output, _, state = l.StreamStep(l.theta, step_inputs, step_paddings,
                                        state)
        all_outs.append(output)
      all_outs = tf.concat(all_outs, axis=1)
      return all_outs * (1. - expand_pad(paddings))

    base_outs = stream(l, inputs_t, paddings)
    actual_outs = stream(l, shift_inputs_t, leading_paddings)

    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      expected, actual = sess.run([base_outs, actual_outs])
      for i in range(batch):
        actual[i] = np.roll(actual[i], -(max_seqlen - seqlen[i]), axis=0)
      print(f'expected: {repr(expected)}')
      print(f'actual: {repr(actual)}')
      print(f'np.sum(np.abs(expected)): {np.sum(np.abs(expected))}')
      print(f'np.sum(np.abs(actual)): {np.sum(np.abs(actual))}')
      self.assertAllClose(expected, actual)


class CausalConv2DLayerStreamStepTest(stream_step_test_base.StreamStepTestBase):

  @property
  def input_rank(self):
    return 4

  def _GetParams(self, **kwargs):
    channel = kwargs['input_dim']
    kernel = kwargs['kernel']
    bias = kwargs['bias']
    p = conv_layers.CausalConv2DLayerWithPadding.Params().Set(
        name='conv',
        filter_stride=[1, 1],
        filter_shape=[kernel, 1, channel, channel],
        params_init=py_utils.WeightInit.Gaussian(0.1),
        bias=bias,
        bias_init=py_utils.WeightInit.Gaussian(0.1))
    return p

  def _FProp(self, layer, inputs, paddings):
    return layer.FProp(layer.theta, inputs, paddings)

  def _GetFPropOutput(self, fprop_out):
    return fprop_out[0]

  @parameterized.named_parameters(
      ('Basic',),
      ('BasicS2', False, 2),
      ('BasicBias', False, 1, True),
      ('SkipNorm', True),
      ('SkipNormS4', True, 4),
  )
  def testCommon(self, testonly_skip_norm_layers=False, stride=1, bias=False):
    kwargs = dict(input_dim=3, kernel=5, stride=stride, bias=bias)
    with flagsaver.flagsaver(
        testonly_skip_norm_layers=testonly_skip_norm_layers):
      self._TestStreamStepHelper(**kwargs)


class GlobalPoolingLayerTest(test_utils.TestCase):
  """Tests for GlobalPoolingLayer."""

  def _testHelper(self,
                  pooling_type,
                  inputs,
                  input_paddings,
                  expected_output,
                  expected_output_padding,
                  feed_dict=None):
    param = conv_layers.GlobalPoolingLayer.Params().Set(
        name='test_layer', pooling_type=pooling_type)
    pooling_layer = param.Instantiate()
    with self.session(use_gpu=True) as sess:
      inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
      input_paddings = None if input_paddings is None else tf.convert_to_tensor(
          input_paddings, dtype=tf.float32)
      output, output_paddings = pooling_layer.FPropDefaultTheta(
          inputs, input_paddings)
      self.evaluate(tf.global_variables_initializer())
      if input_paddings is None:
        self.assertIsNone(output_paddings)
        output_val = sess.run(output, feed_dict=feed_dict)
      else:
        output_val, output_paddings_val = sess.run([output, output_paddings],
                                                   feed_dict=feed_dict)

    self.assertAllClose(expected_output, output_val)
    if input_paddings is not None:
      self.assertAllEqual(expected_output_padding, output_paddings_val)

  def testPooling(self):
    inputs = np.random.random([3, 5, 2, 4]) - 0.5
    expected_avg_output = np.mean(inputs, axis=(1, 2), keepdims=True)
    expected_max_output = np.amax(inputs, axis=(1, 2), keepdims=True)
    self._testHelper('AVG', inputs, None, expected_avg_output, None)
    self._testHelper('MAX', inputs, None, expected_max_output, None)

  def testPoolingWithPadding(self):
    inputs = np.random.random([4, 3, 2, 4]) - 0.5
    paddings = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
    expected_paddings = np.array([[0], [0], [0], [1]])
    expected_avg_output = np.array([
        np.mean(inputs[0][:3], axis=(0, 1), keepdims=True),
        np.mean(inputs[1][:2], axis=(0, 1), keepdims=True),
        np.mean(inputs[2][:1], axis=(0, 1), keepdims=True),
        np.zeros((1, 1, 4))
    ])
    expected_max_output = np.array([
        np.amax(inputs[0][:3], axis=(0, 1), keepdims=True),
        np.amax(inputs[1][:2], axis=(0, 1), keepdims=True),
        np.amax(inputs[2][:1], axis=(0, 1), keepdims=True),
        np.zeros((1, 1, 4))
    ])

    self._testHelper('AVG', inputs, paddings, expected_avg_output,
                     expected_paddings)
    self._testHelper('MAX', inputs, paddings, expected_max_output,
                     expected_paddings)

  def testPoolingWithUnknowShapeInput(self):
    """Tests GlobalPooling layer with unknown shape tensor."""

    def remove_shape(tensor):
      shape = tf.placeholder(tf.int32, name='removed_shape')
      return tf.reshape(tensor, shape)

    g = tf.Graph()
    with g.as_default(), tf.Session(graph=g) as _:
      tf.random.set_seed(24332)
      input_shape = [3, 5, 2, 4]
      inputs = np.random.random(input_shape) - 0.5
      expected_avg_output = np.mean(inputs, axis=(1, 2), keepdims=True)
      input_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
      # initial shape is [3, 5, 2, 4]
      self.assertEqual(py_utils.GetShape(input_tensor), input_shape)
      # remove shape using a tf Defun and verify dynamic tensor shape.
      input_tensor = remove_shape(input_tensor)
      self.assertIsInstance(py_utils.GetShape(input_tensor), tf.Tensor)
      self.assertIsNone(input_tensor.shape.rank)
      self._testHelper(
          'AVG',
          input_tensor,
          None,
          expected_avg_output,
          None,
          feed_dict={'removed_shape:0': input_shape})


if __name__ == '__main__':
  tf.test.main()
