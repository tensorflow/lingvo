# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lingvo Jax linear layers."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import layers as lingvo_layers
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax.layers import linears
import numpy as np
import tensorflow.compat.v2 as tf

to_np = test_utils.to_np
to_tf_nmap = test_utils.to_tf_nmap


class LinearsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters(('RELU'), ('TANH'), ('RELU6'), ('SIGMOID'),
                            ('NONE'))
  def test_feedforward_layer(self, activation):
    p = linears.FeedForward.Params().Set(
        name='jax_ffn', input_dims=3, output_dims=20, activation=activation)
    ffn = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffn.instantiate_variables(prng_key)
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    outputs = test_utils.apply(ffn, initial_vars, ffn.fprop, inputs)
    logging.info('initial_vars in ffn = %s', initial_vars)
    # Test whether tf projection layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap()
    tf_initial_vars.w = initial_vars.linear.w
    tf_initial_vars.b = initial_vars.bias.b
    tf_initial_vars = to_tf_nmap(tf_initial_vars)
    tf_p = lingvo_layers.ProjectionLayer.Params().Set(
        name='tf_ffn',
        input_dim=p.input_dims,
        output_dim=p.output_dims,
        batch_norm=False,
        has_bias=True,
        activation=activation)
    tf_ffn = tf_p.Instantiate()
    tf_output = tf_ffn.FProp(tf_initial_vars,
                             tf.constant(inputs, dtype=tf.float32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)

  @parameterized.parameters(('RELU'), ('TANH'), ('RELU6'), ('SIGMOID'),
                            ('NONE'))
  def test_feedforward_layer_no_bias(self, activation):
    p = linears.FeedForward.Params().Set(
        name='jax_ffn',
        input_dims=3,
        output_dims=20,
        has_bias=False,
        activation=activation)
    ffn = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffn.instantiate_variables(prng_key)
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    outputs = test_utils.apply(ffn, initial_vars, ffn.fprop, inputs)
    logging.info('initial_vars in ffn = %s', initial_vars)
    # Test whether tf projection layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap()
    tf_initial_vars.w = initial_vars.linear.w
    tf_initial_vars = to_tf_nmap(tf_initial_vars)
    tf_p = lingvo_layers.ProjectionLayer.Params().Set(
        name='tf_ffn',
        input_dim=p.input_dims,
        output_dim=p.output_dims,
        batch_norm=False,
        has_bias=False,
        activation=activation)
    tf_ffn = tf_p.Instantiate()
    tf_output = tf_ffn.FProp(tf_initial_vars,
                             tf.constant(inputs, dtype=tf.float32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-6)


class StackingOverTimeLayerTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.named_parameters(
      {
          'testcase_name': 'pad_with_left_frame',
          'pad_with_left_frame': True
      },
      {
          'testcase_name': 'pad_with_zeros',
          'pad_with_left_frame': False
      },
  )
  def testStackingOverTimeFProp(self, pad_with_left_frame):
    params = linears.StackingOverTime.Params()
    params.name = 'stackingOverTime'
    params.left_context = 2
    params.right_context = 0
    params.stride = 2
    params.pad_with_left_frame = pad_with_left_frame

    stacker = linears.StackingOverTime(params)
    stacker_vars = None
    self.assertEqual(stacker.window_size, 3)

    inputs = jnp.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                        [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0], [0, 0]]],
                       dtype=jnp.float32)
    paddings = jnp.array(
        [[[0], [0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1], [1]]],
        dtype=jnp.float32)

    outputs, output_paddings = test_utils.apply(stacker, stacker_vars,
                                                stacker.fprop, inputs, paddings)
    print(f'{outputs}')
    if pad_with_left_frame:
      expected_outputs = jnp.array([
          [[1, 1, 1, 1, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
          [[7, 7, 7, 7, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
      ],
                                   dtype=jnp.float32)
    else:
      expected_outputs = jnp.array([
          [[0, 0, 0, 0, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
          [[0, 0, 0, 0, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
      ],
                                   dtype=jnp.float32)
    self.assertAllClose(expected_outputs, outputs)

    expected_output_paddings = jnp.array([[[0], [0], [0]], [[0], [0], [1]]],
                                         dtype=jnp.float32)
    self.assertAllClose(expected_output_paddings, output_paddings)

  @parameterized.named_parameters(
      {
          'testcase_name': 'pad_with_right_frame',
          'pad_with_right_frame': True
      },
      {
          'testcase_name': 'pad_with_zeros',
          'pad_with_right_frame': False
      },
  )
  def testStackingOverTimePadWithRightFrameFProp(self, pad_with_right_frame):
    params = linears.StackingOverTime.Params()
    params.name = 'stackingOverTime'
    params.left_context = 0
    params.right_context = 1
    params.stride = 2
    params.pad_with_right_frame = pad_with_right_frame

    stacker = linears.StackingOverTime(params)
    stacker_vars = None
    self.assertEqual(stacker.window_size, 2)

    # input shape [2, 5, 2]
    inputs = jnp.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                        [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0]]],
                       dtype=jnp.float32)
    paddings = jnp.array([[[0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1]]],
                         dtype=jnp.float32)
    outputs, output_paddings = test_utils.apply(stacker, stacker_vars,
                                                stacker.fprop, inputs, paddings)
    print(f'{outputs}')

    if pad_with_right_frame:
      # output shape [2, 3, 4]
      # [5, 5] is duplication of the last input frame.
      expected_outputs = jnp.array([
          [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 5, 5]],
          [[7, 7, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
      ],
                                   dtype=jnp.float32)
    else:
      expected_outputs = jnp.array([
          [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 0, 0]],
          [[7, 7, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
      ],
                                   dtype=jnp.float32)

    self.assertAllClose(expected_outputs, outputs)

    expected_output_paddings = jnp.array([[[0], [0], [0]], [[0], [1], [1]]],
                                         dtype=jnp.float32)
    self.assertAllClose(expected_output_paddings, output_paddings)

  def testStackingOverTimeFPropReduceMaxPadding(self):
    params = linears.StackingOverTime.Params()
    params.name = 'stackingOverTime'
    params.left_context = 2
    params.right_context = 0
    params.stride = 2
    params.padding_reduce_option = 'reduce_max'

    stacker = linears.StackingOverTime(params)
    stacker_vars = None
    self.assertEqual(stacker.window_size, 3)

    inputs = jnp.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                        [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0], [0, 0]]],
                       dtype=jnp.float32)
    paddings = jnp.array(
        [[[0], [0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1], [1]]],
        dtype=jnp.float32)

    outputs, output_paddings = test_utils.apply(stacker, stacker_vars,
                                                stacker.fprop, inputs, paddings)
    print(f'{outputs}')
    expected_outputs = jnp.array([
        [[0, 0, 0, 0, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
        [[0, 0, 0, 0, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
    ],
                                 dtype=jnp.float32)

    self.assertAllClose(expected_outputs, outputs)

    expected_output_paddings = jnp.array([[[1], [0], [0]], [[1], [1], [1]]],
                                         dtype=jnp.float32)
    self.assertAllClose(expected_output_paddings, output_paddings)

  def testStackingOverTimeFProp2(self):
    params = linears.StackingOverTime.Params()
    params.name = 'stackingOverTime'
    params.left_context = 0
    params.right_context = 1
    params.stride = 2

    stacker = linears.StackingOverTime(params)
    stacker_vars = None
    self.assertEqual(stacker.window_size, 2)

    inputs = np.random.normal(size=[2, 21, 16])
    # poor man's tf.sequence_mask in np.
    mask = np.zeros([2, 21]).astype(np.float32)
    mask[0, :9] = 1.
    mask[1, :14] = 1.

    paddings = 1.0 - mask
    paddings = jnp.expand_dims(paddings, -1)
    outputs, output_paddings = test_utils.apply(stacker, stacker_vars,
                                                stacker.fprop, inputs, paddings)

    # length
    self.assertAllClose(
        np.array([5, 7], dtype=np.float32), np.sum(1.0 - output_paddings,
                                                   (1, 2)))
    # input and output sums are equal
    self.assertAllClose(np.sum(inputs, (1, 2)), np.sum(outputs, (1, 2)))

  def testStackingOverTimeIdentityFProp(self):
    params = linears.StackingOverTime.Params()
    params.name = 'stackingOverTime'
    params.left_context = 0
    params.right_context = 0
    params.stride = 1

    stacker = linears.StackingOverTime(params)
    stacker_vars = None
    self.assertEqual(stacker.window_size, 1)
    inputs = jnp.array([[[1], [2], [3], [4], [5]]], dtype=jnp.float32)
    paddings = jnp.zeros([1, 5, 1], dtype=jnp.float32)

    outputs, output_paddings = test_utils.apply(stacker, stacker_vars,
                                                stacker.fprop, inputs, paddings)
    print(f'{outputs}')
    expected_outputs = jnp.array([[[1], [2], [3], [4], [5]]], dtype=jnp.float32)
    self.assertAllClose(expected_outputs, outputs)
    expected_output_paddings = jnp.array([[[0], [0], [0], [0], [0]]],
                                         dtype=jnp.float32)
    self.assertAllClose(expected_output_paddings, output_paddings)

  def _testUnstack(self, inputs, **kwargs):
    params = linears.StackingOverTime.Params().Set(
        name='stackingOverTime', **kwargs)

    stacker = params.Instantiate()
    stacker_vars = None
    stacked, _ = test_utils.apply(stacker, stacker_vars, stacker.fprop, inputs)
    unstacked = test_utils.apply(stacker, stacker_vars, stacker.unstack,
                                 stacked)
    print(f'{unstacked}')

    batch, input_length, depth = inputs.shape
    stacked_length = stacked.shape[1]
    stride = stacker.params.stride
    right_context = stacker.params.right_context

    self.assertAllClose(
        unstacked.shape,
        [batch, (stacked_length - 1) * stride + right_context + 1, depth])
    if right_context + 1 >= stride:
      self.assertGreaterEqual(unstacked.shape[1], input_length)
      self.assertAllClose(inputs, unstacked[:, :input_length])
    else:
      self.assertLessEqual(unstacked.shape[1], input_length)
      # The final up to stride - right_context - 1 values are missing.
      self.assertLessEqual(input_length - unstacked.shape[1],
                           stride - right_context - 1)
      self.assertAllClose(inputs[:, :unstacked.shape[1]], unstacked)

  def testStackingOverTimeUnstack(self):
    batch_size = 2
    length = 7
    depth = 3
    inputs = jnp.reshape(
        jnp.arange(batch_size * length * depth, dtype=jnp.float32),
        [batch_size, length, depth])
    self._testUnstack(inputs, left_context=2, stride=1)
    with self.assertRaises(ValueError):
      self._testUnstack(inputs, stride=2)
    self._testUnstack(inputs, stride=2, right_context=3)
    self._testUnstack(inputs, left_context=2, stride=3)
    self._testUnstack(inputs, stride=4, right_context=3)
    self._testUnstack(inputs, stride=4, left_context=1, right_context=2)

if __name__ == '__main__':
  absltest.main()
