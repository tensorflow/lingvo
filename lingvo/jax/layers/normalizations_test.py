# Lint as: python3
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
from jax import test_util
from lingvo.core import bn_layers
from lingvo.core import layers as lingvo_layers
from lingvo.jax import base_layer
from lingvo.jax import test_utils
from lingvo.jax.layers import normalizations
import numpy as np
import tensorflow.compat.v2 as tf

ToNp = test_utils.ToNp


class NormalizationsTest(test_util.JaxTestCase):

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

    jax_mean, jax_variance = normalizations.ComputeMoments(
        inputs, paddings, reduce_over_dims=reduce_over_dims, keepdims=keepdims)

    tf_mean, tf_variance = bn_layers.ComputeMoments(
        inputs, paddings, reduce_over_dims=reduce_over_dims, keepdims=keepdims)

    logging.info('jax_mean: %s', jax_mean)
    logging.info('jax_variance: %s', jax_variance)
    logging.info('tf_mean: %s', tf_mean)
    logging.info('tf_variance: %s', tf_variance)

    self.assertAllClose(ToNp(jax_mean), ToNp(tf_mean))
    self.assertAllClose(ToNp(jax_variance), ToNp(tf_variance))

  def test_bn01(self):
    test_layer_p = normalizations.BatchNormLayer.Params().Set(
        name='bn', decay=0.8, dim=8)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=1234)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)
    logging.info('initial_vars: %s', initial_vars)

    inputs = np.random.normal(1.5, 2.0, [2, 200, 8])
    paddings = np.zeros([2, 200, 1])
    paddings[1, 1, 0] = 1.0
    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    # comp function is fully functional.
    @jax.jit
    def Comp(theta, prng_key, global_step, inputs, paddings):
      with base_layer.JaxContext.NewContext():
        # Mix in global steps so that prng seed depends on a global step.
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.ResetPrngKey(per_step_prng_key, global_step)
        layer.PrepareFProp()
        output = layer.FProp(theta, inputs, paddings)
        forward_updated_theta = layer.forward_updated_vars

        def UpdateParam(old, new):
          if new is not None:
            return new
          else:
            return old

        # Get the new variables.
        new_theta = tf.nest.map_structure(UpdateParam, theta,
                                          forward_updated_theta)
        # Fetch summaries.
        summaries = base_layer.AllSummaries()

        return new_theta, output, summaries

    new_vars, output1, summaries = Comp(initial_vars, compute_key, global_step,
                                        inputs, paddings)

    tf.nest.assert_same_structure(
        summaries, {
            'bn.FProp/moving_mean': None,
            'bn.FProp/variance': None,
            'bn.FProp/mean': None,
            'bn.FProp/moving_variance': None
        })

    logging.info('new_vars: %s', new_vars)
    logging.info('output1: %s', output1)
    logging.info('summaries: %s', summaries)

    expected_moving_mean = (
        initial_vars.moving_mean * 0.8 + 0.2 * summaries['bn.FProp/mean'])
    expected_moving_variance = (
        initial_vars.moving_variance * 0.8 +
        0.2 * summaries['bn.FProp/variance'])

    self.assertAllClose(ToNp(expected_moving_mean), ToNp(new_vars.moving_mean))
    self.assertAllClose(
        ToNp(expected_moving_variance), ToNp(new_vars.moving_variance))

  def test_bn02(self):
    test_layer_p = normalizations.BatchNormLayer.Params().Set(
        name='bn', decay=0.8, dim=1)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)
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
      with base_layer.JaxContext.NewContext():
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.ResetPrngKey(per_step_prng_key, global_step)
        layer.PrepareFProp()
        output = layer.FProp(theta, inputs, paddings)
        forward_updated_theta = layer.forward_updated_vars

        def UpdateParam(old, new):
          if new is not None:
            return new
          else:
            return old

        # Get the new variables.
        new_theta = tf.nest.map_structure(UpdateParam, theta,
                                          forward_updated_theta)
        # Fetch summaries.
        summaries = base_layer.AllSummaries()

        return new_theta, output, summaries

    _, jax_output, _ = Comp(initial_vars, compute_key, global_step, inputs,
                            paddings)

    logging.info('jax_output: %s', jax_output)

    tf_initial_vars = initial_vars.Transform(ToNp)

    # Now run TF based computation.
    tf_layer_p = bn_layers.BatchNormLayer.Params().Set(
        name='bn', dim=1, decay=8)
    tf_layer = tf_layer_p.Instantiate()
    tf_output = tf_layer.FProp(tf_initial_vars, inputs, paddings)
    logging.info('tf_output: %s', tf_output)
    self.assertAllClose(ToNp(jax_output), ToNp(tf_output))

  @parameterized.parameters((0.0, 0.0), (0.5, 0.), (0.0, 0.5), (0.5, 0.5),
                            (0.5, 1.0))
  def test_layer_norm(self, scale, bias):
    p = normalizations.LayerNorm.Params().Set(name='jax_ln', input_dims=3)
    layer_norm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer_norm.InstantiateVariables(init_key)
    initial_vars.scale = scale
    initial_vars.bias = bias
    npy_input = np.random.normal(1.0, 0.5,
                                 [10, 10, 10, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_input)
    outputs = layer_norm.FProp(initial_vars, inputs)
    # Now test whether tf layer norm returns same output
    tf_p = lingvo_layers.LayerNorm.Params().Set(
        name='tf_ln', input_dim=p.input_dims)
    tf_layer_norm = tf_p.Instantiate()
    tf_output = tf_layer_norm.FProp(initial_vars,
                                    tf.constant(inputs, dtype=tf.float32))
    np_outputs = ToNp(outputs)
    tf_np_outputs = ToNp(tf_output)
    self.assertAllClose(bias, np_outputs.mean(), atol=1e-3)
    self.assertAllClose((1.0 + scale)**2, np.var(np_outputs), atol=5e-3)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=6e-5)


if __name__ == '__main__':
  absltest.main()
