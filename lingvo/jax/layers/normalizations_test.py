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
"""Tests for lingvo Jax normalization layers."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.core import bn_layers
from lingvo.core import layers as lingvo_layers
from lingvo.jax import base_layer
from lingvo.jax import test_utils
from lingvo.jax.layers import normalizations
import numpy as np
import tensorflow.compat.v2 as tf

to_np = test_utils.to_np


def _JaxToTfDtype(jax_dtype):
  if jax_dtype == jnp.bfloat16:
    return tf.bfloat16
  elif jax_dtype == jax.dtypes.float0:
    return tf.float32
  else:
    return tf.dtypes.as_dtype(jax_dtype)


class NormalizationsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def test_momentum(self):
    inputs = np.random.normal(1.5, 2.0, [2, 200, 8])
    paddings = np.zeros([2, 200, 1])
    paddings[1, 1, 0] = 1.0
    reduce_over_dims = [0, 1]
    keepdims = True

    jax_mean, jax_variance = normalizations.compute_moments(
        inputs, paddings, reduce_over_dims=reduce_over_dims, keepdims=keepdims)

    tf_mean, tf_variance = bn_layers.ComputeMoments(
        inputs, paddings, reduce_over_dims=reduce_over_dims, keepdims=keepdims)

    logging.info('jax_mean: %s', jax_mean)
    logging.info('jax_variance: %s', jax_variance)
    logging.info('tf_mean: %s', tf_mean)
    logging.info('tf_variance: %s', tf_variance)

    self.assertAllClose(to_np(jax_mean), to_np(tf_mean))
    self.assertAllClose(to_np(jax_variance), to_np(tf_variance))

  def test_bn01(self):
    test_layer_p = normalizations.BatchNorm.Params().Set(
        name='bn', decay=0.8, dim=8)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=1234)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.instantiate_variables(init_key)
    logging.info('initial_vars: %s', initial_vars)

    inputs = np.random.normal(1.5, 2.0, [2, 200, 8])
    paddings = np.zeros([2, 200, 1])
    paddings[1, 1, 0] = 1.0
    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    # comp function is fully functional.
    @jax.jit
    def Comp(theta, prng_key, global_step, inputs, paddings):
      with base_layer.JaxContext.new_context(
          global_step=global_step, prng_key=prng_key) as jax_context:
        jax_context.bind(layer, layer.vars_to_flax_vars(theta),
                         [base_layer.SCOPE_VARS])
        # Mix in global steps so that prng seed depends on a global step.
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.reset_prng_key(per_step_prng_key, global_step)
        output = layer.fprop(inputs, paddings)
        forward_updated_theta = layer.updated_vars

        def UpdateParam(old, new):
          if new is not None:
            return new
          else:
            return old

        # Get the new variables.
        new_theta = tf.nest.map_structure(UpdateParam, theta,
                                          forward_updated_theta)
        # Fetch summaries.
        summaries = base_layer.all_summaries()

        return new_theta, output, summaries

    new_vars, output1, summaries = Comp(initial_vars, compute_key, global_step,
                                        inputs, paddings)

    tf.nest.assert_same_structure(
        summaries, {
            'bn.fprop/moving_mean_scalar': None,
            'bn.fprop/variance_scalar': None,
            'bn.fprop/mean_scalar': None,
            'bn.fprop/moving_variance_scalar': None
        })

    logging.info('new_vars: %s', new_vars)
    logging.info('output1: %s', output1)
    logging.info('summaries: %s', summaries)

    expected_moving_mean = (
        initial_vars.moving_mean * 0.8 +
        0.2 * summaries['bn.fprop/mean_scalar'])
    expected_moving_variance = (
        initial_vars.moving_variance * 0.8 +
        0.2 * summaries['bn.fprop/variance_scalar'])

    self.assertAllClose(
        to_np(expected_moving_mean), to_np(new_vars.moving_mean))
    self.assertAllClose(
        to_np(expected_moving_variance), to_np(new_vars.moving_variance))

  def test_bn02(self):
    test_layer_p = normalizations.BatchNorm.Params().Set(
        name='bn', decay=0.8, dim=1)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.instantiate_variables(init_key)
    initial_vars.beta = jnp.array([0.7])
    initial_vars.gamma = jnp.array([1.8])
    logging.info('initial_vars: %s', initial_vars)

    inputs = np.random.normal(1.5, 2.0, [2, 200, 1])
    paddings = np.zeros([2, 200, 1])
    paddings[1, 1, 0] = 1.0
    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    # comp function is fully functional.
    def Comp(theta, prng_key, global_step, inputs, paddings):
      with base_layer.JaxContext.new_context(
          global_step=global_step, prng_key=prng_key) as jax_context:
        jax_context.bind(layer, layer.vars_to_flax_vars(theta),
                         [base_layer.SCOPE_VARS])
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.reset_prng_key(per_step_prng_key, global_step)
        output = layer.fprop(inputs, paddings)
        forward_updated_theta = layer.updated_vars

        def UpdateParam(old, new):
          if new is not None:
            return new
          else:
            return old

        # Get the new variables.
        new_theta = tf.nest.map_structure(UpdateParam, theta,
                                          forward_updated_theta)
        # Fetch summaries.
        summaries = base_layer.all_summaries()

        return new_theta, output, summaries

    _, jax_output, _ = Comp(initial_vars, compute_key, global_step, inputs,
                            paddings)

    logging.info('jax_output: %s', jax_output)

    tf_initial_vars = initial_vars.Transform(to_np)

    # Now run TF based computation.
    tf_layer_p = bn_layers.BatchNormLayer.Params().Set(
        name='bn', dim=1, decay=8)
    tf_layer = tf_layer_p.Instantiate()
    tf_output = tf_layer.FProp(tf_initial_vars, inputs, paddings)
    logging.info('tf_output: %s', tf_output)
    self.assertAllClose(to_np(jax_output), to_np(tf_output))

  @parameterized.parameters((0.0, 0.0), (0.5, 0.), (0.0, 0.5), (0.5, 0.5),
                            (0.5, 1.0))
  def test_layer_norm(self, scale, bias):
    p = normalizations.LayerNorm.Params().Set(name='jax_ln', input_dims=3)
    layer_norm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer_norm.instantiate_variables(init_key)
    initial_vars.scale = scale
    initial_vars.bias = bias
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    outputs = test_utils.apply(layer_norm, initial_vars, layer_norm.fprop,
                               inputs)
    # Now test whether tf layer norm returns same output
    tf_p = lingvo_layers.LayerNorm.Params().Set(
        name='tf_ln', input_dim=p.input_dims)
    tf_layer_norm = tf_p.Instantiate()
    tf_output = tf_layer_norm.FProp(initial_vars,
                                    tf.constant(inputs, dtype=tf.float32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    self.assertAllClose(bias, np_outputs.mean(), atol=1e-3)
    self.assertAllClose((1.0 + scale)**2, np.var(np_outputs), atol=5e-3)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=6e-5)

  @parameterized.parameters((0.0,), (0.5,))
  def test_rms_norm(self, scale):
    input_dims = 3
    p = normalizations.RmsNorm.Params().Set(
        name='jax_rmsn', input_dims=input_dims, direct_scale=False)
    rms_norm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = rms_norm.instantiate_variables(init_key)
    initial_vars.scale = scale
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    outputs = test_utils.apply(rms_norm, initial_vars, rms_norm.fprop, inputs)
    # Now test whether tf RMS norm returns same output.
    tf_p = lingvo_layers.LayerNorm.Params().Set(
        name='tf_rmsn', input_dim=p.input_dims, bias=False, center=False)
    tf_layer_norm = tf_p.Instantiate()
    tf_output = tf_layer_norm.FProp(initial_vars,
                                    tf.constant(inputs, dtype=tf.float32))
    np_outputs = to_np(outputs)
    tf_np_outputs = to_np(tf_output)
    np_norms = np.linalg.norm(np_outputs / np.sqrt(float(input_dims)), axis=-1)
    self.assertAllClose(
        (1.0 + scale) * np.ones_like(np_norms), np_norms, atol=5e-3)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=6e-5)

  @parameterized.named_parameters(
      ('_epsilon_1e-3', 4, 2, False, 4, 1e-3, [2, 2, 2, 4], jnp.float32, None,
       jnp.float32),
      ('_epsilon_1e-6', 4, 2, False, 4, 1e-6, [2, 2, 2, 4], jnp.float32, None,
       jnp.float32),
      ('_f32_input_f32_fprop', 4, 2, False, 4, 1e-3, [2, 2, 2, 4], jnp.float32,
       [[0, 0], [0, 1]], jnp.float32),
      ('_bf16_input_f32_fprop', 4, 2, False, 4, 1e-3, [2, 2, 2, 4],
       jnp.bfloat16, [[0, 0], [0, 1]], jnp.float32),
      ('_f32_input_bf16_fprop', 4, 2, False, 4, 1e-3, [2, 2, 2, 4], jnp.float32,
       [[0, 0], [0, 1]], jnp.bfloat16),
      ('_bf16_input_bf16_fprop', 4, 2, False, 4, 1e-3, [2, 2, 2, 4],
       jnp.bfloat16, [[0, 0], [0, 1]], jnp.bfloat16),
      ('_3d_input', 4, 2, False, 3, 1e-3, [2, 2, 4], jnp.float32,
       [[0, 0], [0, 1]], jnp.float32),
      ('_4d_input_cumulative_mode', 2, 2, True, 4, 1e-3, [2, 4, 1, 2],
       jnp.float32, [[0, 0, 0, 0], [0, 0, 0, 0]], jnp.float32),
      ('_3d_input_cumulative_mode', 2, 2, True, 3, 1e-3, [2, 4, 2], jnp.float32,
       [[0, 0, 0, 0], [0, 0, 0, 0]], jnp.float32),
  )
  def test_group_norm(self, dim, num_groups, cumulative, input_rank, epsilon,
                      input_shape, input_dtype, paddings, fprop_dtype):
    p = normalizations.GroupNorm.Params().Set(
        name='jax_gn',
        dim=dim,
        num_groups=num_groups,
        cumulative=cumulative,
        input_rank=input_rank,
        epsilon=epsilon,
        fprop_dtype=fprop_dtype)
    group_norm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = group_norm.instantiate_variables(init_key)
    npy_input = np.random.normal(1.0, 0.5, input_shape).astype(np.float32)
    inputs = jnp.asarray(npy_input, dtype=input_dtype)
    if paddings is None:
      output = test_utils.apply(
          group_norm, initial_vars, group_norm.fprop, inputs, paddings=None)
    else:
      output, output_paddings = test_utils.apply(
          group_norm,
          initial_vars,
          group_norm.fprop,
          inputs,
          paddings=jnp.asarray(paddings, dtype=input_dtype))

    # Now test whether tf layer norm returns same output.
    tf_p = bn_layers.GroupNormLayer.Params().Set(
        name='tf_gn',
        dim=dim,
        num_groups=num_groups,
        cumulative=cumulative,
        input_rank=input_rank,
        epsilon=epsilon,
        fprop_dtype=_JaxToTfDtype(fprop_dtype))
    tf_group_norm = tf_p.Instantiate()
    tf_inputs = tf.constant(inputs, dtype=_JaxToTfDtype(input_dtype))
    if paddings is None:
      tf_output = tf_group_norm.FProp(initial_vars, tf_inputs, paddings=None)
    else:
      tf_output, tf_output_paddings = tf_group_norm.FProp(
          initial_vars,
          tf_inputs,
          paddings=tf.convert_to_tensor(
              paddings, dtype=_JaxToTfDtype(input_dtype)))

    self.assertAllClose(to_np(tf_output), to_np(output))
    if paddings is not None:
      self.assertAllClose(to_np(tf_output_paddings), to_np(output_paddings))


if __name__ == '__main__':
  absltest.main()
