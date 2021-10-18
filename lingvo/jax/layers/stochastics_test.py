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
"""Tests for lingvo Jax stochastic layers."""

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax import test_util
from lingvo.jax import base_layer
from lingvo.jax import test_utils
from lingvo.jax.layers import stochastics

ToNp = test_utils.ToNp


class StochaticsTest(test_util.JaxTestCase):

  def test_dropout_layer01(self):
    test_layer_p = stochastics.DropoutLayer.Params().Set(
        name='dropout', keep_prob=0.8)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=12346)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)
    logging.info('initial_vars: %s', initial_vars)

    inputs = jnp.ones([10, 1000], dtype=jnp.bfloat16)
    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    def Comp(theta, prng_key, global_step, inputs):
      with base_layer.JaxContext.NewContext():
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.ResetPrngKey(per_step_prng_key, global_step)
        output1 = layer.FProp(theta, inputs)
        output2 = layer.FProp(theta, inputs)
        return output1, output2

    output1, output2 = Comp(initial_vars, compute_key, global_step, inputs)

    out1_sum = jnp.sum(output1)
    out2_sum = jnp.sum(output2)
    out1_nonzero = jnp.sum(output1 > 0.0)
    out2_nonzero = jnp.sum(output2 > 0.0)

    logging.info('out1_sum: %s', out1_sum)
    logging.info('out2_sum: %s', out2_sum)
    logging.info('out1_nonzero: %s', out1_nonzero)
    logging.info('out2_nonzero: %s', out2_nonzero)

    self.assertEqual(9920.0, out1_sum)
    self.assertEqual(10048.0, out2_sum)
    self.assertEqual(7944.0, out1_nonzero)
    self.assertEqual(8029.0, out2_nonzero)

  def test_dropout_layer_02(self):
    test_layer_p = stochastics.DropoutLayer.Params().Set(
        name='dropout',
        keep_prob=0.8,
        noise_shape=[10, 6, 8],
        noise_shape_broadcast_dims=[2])
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=12346)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)
    logging.info('initial_vars: %s', initial_vars)

    inputs = jnp.ones([2, 10, 6, 8], dtype=jnp.bfloat16)
    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    def Comp(theta, prng_key, global_step, inputs):
      with base_layer.JaxContext.NewContext():
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.ResetPrngKey(per_step_prng_key, global_step)
        layer.PrepareFProp()
        output1 = layer.FProp(theta, inputs)
        return output1

    output1 = Comp(initial_vars, compute_key, global_step, inputs)

    out1_sum = jnp.sum(output1)
    out1_nonzero = jnp.sum(output1 > 0.0)

    logging.info('out1_sum: %s', out1_sum)
    logging.info('out1_nonzero: %s', out1_nonzero)

    self.assertEqual(980, out1_sum)
    self.assertEqual(784, out1_nonzero)


if __name__ == '__main__':
  absltest.main()
