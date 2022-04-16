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
"""Tests for lingvo Jax vit model."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
from lingvo.jax import test_utils
from lingvo.jax.layers import vit

import numpy as np


class VitTest(test_utils.TestCase):

  def test_vit_entry_layers(self):
    batch_size, height, width = 3, 48, 48
    patch_size, hidden_dim = 12, 24

    p_entry = vit.VitEntryLayers.Params().Set(
        name='entry',
        image_size=height,
        patch_size=patch_size,
        dim_per_patch=hidden_dim,
        image_channels=3,
        pos_emb_dropout_prob=0.1)
    entry = p_entry.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = entry.instantiate_variables(prng_key)

    inputs_np = np.random.normal(size=[batch_size, height, width, 3])
    inputs = jnp.asarray(inputs_np)

    features = test_utils.apply(entry, initial_vars, entry.fprop, inputs)

    self.assertEqual(features.shape,
                     (batch_size, height * width // patch_size**2, hidden_dim))

  def test_vit_transformer_layers(self):
    batch_size, num_tokens, input_dims, hidden_dims = 3, 8, 12, 48
    num_heads, num_layers = 4, 2
    residual_dropout_prob, activation_dropout_prob = 0.2, 0.2
    atten_dropout_prob = 0.2
    atten_logit_cap = 50.0

    p_middle = vit.VitTransformerLayers.Params().Set(
        name='middle',
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        num_heads=num_heads,
        num_layers=num_layers,
        atten_logit_cap=atten_logit_cap,
        residual_dropout_prob=residual_dropout_prob,
        activation_dropout_prob=activation_dropout_prob,
        atten_dropout_prob=atten_dropout_prob)

    middle = p_middle.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = middle.instantiate_variables(prng_key)

    inputs_np = np.random.normal(size=[batch_size, num_tokens, input_dims])
    inputs = jnp.asarray(inputs_np)

    features = test_utils.apply(middle, initial_vars, middle.fprop, inputs)

    self.assertEqual(features.shape, (batch_size, num_tokens, input_dims))

  def test_vit_exit_layers(self):
    batch_size, num_tokens, input_dims = 3, 8, 12
    output_dropout_prob = 0.1

    p_exit = vit.VitExitLayers.Params().Set(
        name='exit',
        hidden_dim=input_dims,
        output_dim=input_dims,
        output_dropout_prob=output_dropout_prob)

    exit_module = p_exit.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = exit_module.instantiate_variables(prng_key)

    inputs_np = np.random.normal(size=[batch_size, num_tokens, input_dims])
    inputs = jnp.asarray(inputs_np)

    features = test_utils.apply(exit_module, initial_vars, exit_module.fprop,
                                inputs)

    self.assertEqual(features.shape, (batch_size, input_dims))

  def _vit_params(self):
    image_size, patch_size = 48, 6
    hidden_dims, mlp_dims = 12, 48
    num_heads, num_tfm_layers = 4, 2
    residual_dropout_prob, activation_dropout_prob = 0.1, 0.1
    atten_dropout_prob, output_dropout_prob = 0.1, 0.3
    pos_emb_dropout_prob = 0.05
    atten_logit_cap = 50.0

    return vit.VisionTransformer.Params().Set(
        name='vit',
        hidden_dim=hidden_dims,
        mlp_dim=mlp_dims,
        patch_size=patch_size,
        num_heads=num_heads,
        num_tfm_layers=num_tfm_layers,
        image_size=image_size,
        activation_dropout_prob=activation_dropout_prob,
        atten_dropout_prob=atten_dropout_prob,
        pos_emb_dropout_prob=pos_emb_dropout_prob,
        residual_dropout_prob=residual_dropout_prob,
        output_dropout_prob=output_dropout_prob,
        atten_logit_cap=atten_logit_cap)

  def test_vit(self):
    batch_size = 3

    p_vit = self._vit_params()
    vit_model = p_vit.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = vit_model.instantiate_variables(prng_key)

    inputs_np = np.random.normal(
        size=[batch_size, p_vit.image_size, p_vit.image_size, 3])
    inputs = jnp.asarray(inputs_np)

    features = test_utils.apply(vit_model, initial_vars, vit_model.fprop,
                                inputs)

    self.assertEqual(features.shape, (batch_size, p_vit.hidden_dim))

  def testVitSkipExitLayers(self):
    batch_size = 3

    p_vit = self._vit_params().Set(exit_layers_tpl=None)
    vit_model = p_vit.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = vit_model.instantiate_variables(prng_key)

    inputs_np = np.random.normal(
        size=[batch_size, p_vit.image_size, p_vit.image_size, 3])
    inputs = jnp.asarray(inputs_np)

    features = test_utils.apply(vit_model, initial_vars, vit_model.fprop,
                                inputs)

    patch_count = p_vit.image_size // p_vit.patch_size
    self.assertEqual(features.shape,
                     (batch_size, patch_count**2, p_vit.hidden_dim))


if __name__ == '__main__':
  absltest.main()
