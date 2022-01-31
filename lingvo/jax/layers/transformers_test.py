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
"""Tests for lingvo Jax transformer layers."""

import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import test_util
from lingvo.core import batch_major_attention
from lingvo.core import gshard_builder
from lingvo.core import layers_with_attention
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax.layers import attentions
from lingvo.jax.layers import embedding_softmax
from lingvo.jax.layers import ngrammer
from lingvo.jax.layers import transformers
import numpy as np
import tensorflow.compat.v2 as tf


@test_util.with_config(jax_numpy_rank_promotion='allow')
class TransformersTest(test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_transformer_layer(self, mask_self_attention, packed_input,
                             cross_attention):
    p = transformers.Transformer.Params().Set(
        name='jax_transformer_layer',
        input_dims=32,
        hidden_dims=128,
        num_heads=8,
        mask_self_attention=mask_self_attention,
        packed_input=packed_input,
        cross_attention=cross_attention)
    seq_len = np.random.randint(10, 32)
    batch_size = 10
    transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_layer.instantiate_variables(prng_key)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    causal_mask = None
    segment_mask = None
    tf_segment_mask = None
    attention_mask = attentions.convert_paddings_to_mask(paddings)
    if mask_self_attention:
      causal_mask = attentions.causal_mask(inputs)
      attention_mask = jnp.minimum(attention_mask, causal_mask)
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)
      attention_mask = jnp.minimum(attention_mask, segment_mask)
      if mask_self_attention:
        tf_segment_mask = batch_major_attention.CausalSegmentMask(
            segment_ids, tf.float32)
      else:
        tf_segment_mask = batch_major_attention.SegmentMask(
            segment_ids, segment_ids)

    cross_inputs = None
    cross_attention_mask = None
    tf_cross_inputs = None
    tf_cross_paddings = None
    tf_cross_segment_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 128)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5, [batch_size, cross_seq_len, p.input_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      tf_cross_inputs = tf.constant(npy_cross_inputs, dtype=tf.float32)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      cross_attention_mask = attentions.convert_paddings_to_mask(cross_paddings)
      tf_cross_paddings = tf.constant(npy_cross_paddings, dtype=tf.float32)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.segment_mask(
            segment_ids, source_segment_ids, dtype=np.float32)
        cross_attention_mask = jnp.minimum(cross_attention_mask,
                                           cross_segment_mask)
        tf_cross_segment_mask = batch_major_attention.SegmentMask(
            segment_ids, source_segment_ids)

    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs, _ = transformer_layer.fprop(
          initial_vars,
          inputs,
          paddings,
          attention_mask=attention_mask,
          cross_inputs=cross_inputs,
          cross_attention_mask=cross_attention_mask)
    logging.info('initial_vars in transformer layer = %s', initial_vars)

    # Test whether tf Transformer layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = test_utils.replace_jax_attention_vars_to_tf(
        initial_vars, cross_attention)
    tf_initial_vars = test_utils.to_tf_nmap(tf_initial_vars)
    logging.info('tf_initial_vars in transformer layer = %s', initial_vars)
    tf_p = batch_major_attention.TransformerLayer.Params().Set(
        name='tf_transformer_layer',
        input_dim=p.input_dims,
        num_heads=p.num_heads,
        mask_self_atten=mask_self_attention,
        packed_input=packed_input,
        has_aux_atten=cross_attention)
    tf_p.tr_fflayer_tpl.hidden_dim = p.hidden_dims
    tf_p.tr_fflayer_tpl.fflayer_tpl.batch_norm = False
    tf_p.tr_fflayer_tpl.fflayer_tpl.has_bias = True
    tf_transformer_layer = tf_p.Instantiate()
    tf_output, _ = tf_transformer_layer.FProp(
        tf_initial_vars,
        tf.constant(npy_inputs, dtype=tf.float32),
        paddings=test_utils.to_tf_nmap(npy_paddings),
        segment_mask=tf_segment_mask,
        aux_vec=tf_cross_inputs,
        aux_paddings=tf_cross_paddings,
        aux_segment_mask=test_utils.to_tf_nmap(tf_cross_segment_mask))
    np_outputs = test_utils.to_np(outputs)
    tf_np_outputs = test_utils.to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-5)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=4)))
  def test_transformer_layer_extendstep(self, packed_input, cross_attention,
                                        dconv_qkv, use_rotary_position_emb):
    p = transformers.Transformer.Params().Set(
        name='jax_transformer_layer',
        input_dims=8,
        hidden_dims=32,
        num_heads=4,
        mask_self_attention=True,
        packed_input=packed_input,
        cross_attention=cross_attention)
    p.tr_atten_tpl.dconv_qkv = dconv_qkv
    p.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    if cross_attention:
      p.cross_atten_tpl = p.tr_atten_tpl.Copy()
      # Cross attention should not have depth-wise convolution.
      p.cross_atten_tpl.dconv_qkv = False
      # Cross attention should not have rotary position embedding.
      p.cross_atten_tpl.use_rotary_position_emb = False

    p.tr_atten_tpl.dconv_kernel_size = 2
    seq_len = 4
    batch_size = 4
    transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_layer.instantiate_variables(prng_key)
    initial_states = transformer_layer.init_states(initial_vars, batch_size,
                                                   seq_len)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    # npy_paddings = np.zeros([batch_size, seq_len])
    paddings = jnp.asarray(npy_paddings)
    attention_mask = attentions.convert_paddings_to_mask(paddings)
    segment_mask = None
    causal_mask = attentions.causal_mask(inputs)
    attention_mask = jnp.minimum(causal_mask, attention_mask)
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)
      attention_mask = jnp.minimum(attention_mask, segment_mask)
    cross_inputs = None
    cross_paddings = None
    cross_attention_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 32)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5, [batch_size, cross_seq_len, p.input_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      cross_attention_mask = attentions.convert_paddings_to_mask(cross_paddings)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.segment_mask(
            segment_ids, source_segment_ids, dtype=np.float32)
        cross_attention_mask = jnp.minimum(cross_attention_mask,
                                           cross_segment_mask)

    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      fprop_outputs, _ = transformer_layer.fprop(
          initial_vars,
          inputs,
          paddings,
          attention_mask=attention_mask,
          cross_inputs=cross_inputs,
          cross_attention_mask=cross_attention_mask)
      decoder_outputs = jnp.zeros(shape=[seq_len, batch_size, p.input_dims])
      atten_states = initial_states
      for t in range(seq_len):
        attention_mask_t = attention_mask[:, :, t, :]
        cross_attention_mask_t = cross_attention_mask
        if cross_attention:
          cross_attention_mask_t = cross_attention_mask[:, :, t, :]
          cross_attention_mask_t = np.expand_dims(
              cross_attention_mask_t, axis=2)
        atten_states, encoded = transformer_layer.extend_step(
            initial_vars,
            atten_states,
            inputs=inputs[:, t, :],
            time_step=t,
            attention_mask=attention_mask_t,
            cross_inputs=cross_inputs,
            cross_attention_mask=cross_attention_mask_t)
        decoder_outputs = decoder_outputs.at[t].set(encoded)

    decoder_out_transposed = jnp.transpose(decoder_outputs, [1, 0, 2])
    logging.info('initial_vars in transformer layer = %s', initial_vars)
    np_fprop_outputs = test_utils.to_np(fprop_outputs)
    np_decoder_outputs = test_utils.to_np(decoder_out_transposed)
    self.assertAllClose(np_fprop_outputs, np_decoder_outputs, atol=1e-5)

  @parameterized.parameters(True, False)
  def test_transformer_layer_cross_attention_ln(self, packed_input):
    p = transformers.Transformer.Params().Set(
        name='jax_transformer_layer',
        input_dims=8,
        hidden_dims=32,
        num_heads=4,
        mask_self_attention=True,
        packed_input=packed_input,
        cross_attention=True)
    seq_len = 5
    batch_size = 4
    transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_layer.instantiate_variables(prng_key)
    # Change the self attention initial vars.
    initial_vars.layer_norm.scale = 0.5
    initial_vars.layer_norm.bias = 5.0
    # Change the cross attention initial vars.
    initial_vars.cross_layer_norm.scale = 15
    initial_vars.cross_layer_norm.bias = 1.5
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    attention_mask = attentions.convert_paddings_to_mask(paddings)
    causal_mask = attentions.causal_mask(inputs)
    attention_mask = jnp.minimum(causal_mask, attention_mask)
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)
      attention_mask = jnp.minimum(attention_mask, segment_mask)
    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      inputs_normalized = transformer_layer.layer_norm.fprop(
          initial_vars.layer_norm, inputs)
      # Compute self-attention, key/value vectors are the input itself
      atten_output, _ = transformer_layer.self_attention.fprop(
          initial_vars.self_attention,
          inputs_normalized,
          inputs_normalized,
          inputs_normalized,
          atten_mask=attention_mask)
      # Residual dropout and connection.
      atten_output = transformer_layer.residual_dropout.fprop(
          initial_vars.residual_dropout, atten_output)
      atten_output += inputs
      # Normalize atten outputs using cross attention.
      atten_output_normalized = transformer_layer.cross_layer_norm.fprop(
          initial_vars.cross_layer_norm, atten_output)
      inputs_normalized = test_utils.to_np(inputs_normalized)
      atten_output_normalized = test_utils.to_np(atten_output_normalized)
    self.assertAllClose(
        initial_vars.layer_norm.bias, inputs_normalized.mean(), atol=1e-3)
    self.assertAllClose(
        (1.0 + initial_vars.layer_norm.scale)**2,
        np.var(inputs_normalized),
        atol=5e-3)
    self.assertAllClose(
        initial_vars.cross_layer_norm.bias,
        atten_output_normalized.mean(),
        atol=1e-3)
    self.assertAllClose(
        (1.0 + initial_vars.cross_layer_norm.scale)**2,
        np.var(atten_output_normalized),
        atol=5e-3)

  def test_transformer_layer_cross_attention_dconv_value_error(self):
    p = transformers.Transformer.Params().Set(
        name='jax_transformer_layer',
        input_dims=8,
        hidden_dims=32,
        num_heads=4,
        cross_attention=True,
        mask_self_attention=True)
    # Enable cross attention.
    p.cross_atten_tpl = p.tr_atten_tpl.Copy()
    # Enable depth-wise convolution.
    p.cross_atten_tpl.dconv_qkv = True
    with self.assertRaises(ValueError):
      p.Instantiate()

  def test_transformer_layer_cross_attention_pos_emb_value_error(self):
    p = transformers.Transformer.Params().Set(
        name='jax_transformer_layer',
        input_dims=8,
        hidden_dims=32,
        num_heads=4,
        cross_attention=True,
        mask_self_attention=True)
    # Enable cross attention.
    p.cross_atten_tpl = p.tr_atten_tpl.Copy()
    # Enable rotary position embedding.
    p.cross_atten_tpl.use_rotary_position_emb = True
    with self.assertRaises(ValueError):
      p.Instantiate()

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_transformer_moe_dense_layer(self, mask_self_attention, packed_input,
                                       cross_attention):
    # Comparing scan over blocks of layers and regular loop
    block_p = transformers.StackedTransformer.Params().Set(
        name='transformer_block',
        enable_while_loop=True,
        num_layers=0,
        num_blocks=1,
        num_layers_per_block=2,
        model_dims=3,
        hidden_dims=6,
        num_heads=1,
        mask_self_attention=mask_self_attention,
        packed_input=packed_input,
        cross_attention=cross_attention,
        num_experts=4,
        num_groups=1,
        moe_layers=[0])
    stack_p = transformers.StackedTransformer.Params().Set(
        name='transformer_stack',
        enable_while_loop=False,
        num_layers=2,  # moe + dense
        model_dims=block_p.model_dims,
        hidden_dims=block_p.hidden_dims,
        num_heads=block_p.num_heads,
        mask_self_attention=block_p.mask_self_attention,
        packed_input=block_p.packed_input,
        cross_attention=block_p.cross_attention,
        num_experts=block_p.num_experts,
        num_groups=block_p.num_groups,
        moe_layers=[0])

    moe_p = stack_p.moe_layer_tpl
    moe_p.expert_capacity_dim = 2
    moe_p.expert_capacity_factor = 0

    moe_p = block_p.moe_layer_tpl
    moe_p.expert_capacity_dim = 2
    moe_p.expert_capacity_factor = 0

    transformer_block = block_p.Instantiate()
    transformer_stack = stack_p.Instantiate()

    seq_len = 4
    batch_size = 3
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_block.instantiate_variables(prng_key)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, block_p.model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    segment_mask = None
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)

    cross_inputs = None
    cross_paddings = None
    cross_segment_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 64)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5,
          [batch_size, cross_seq_len, block_p.model_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.segment_mask(
            segment_ids, source_segment_ids, dtype=np.float32)

    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      block_outputs = transformer_block.fprop(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      stack_outputs = transformer_stack.fprop(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
    block_np_outputs = test_utils.to_np(block_outputs)
    stack_np_outputs = test_utils.to_np(stack_outputs)
    self.assertAllClose(stack_np_outputs, block_np_outputs, atol=1e-5)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_stacked_transformer_layer(self, mask_self_attention, packed_input,
                                     cross_attention):
    p = transformers.StackedTransformer.Params().Set(
        name='jax_stacked_transformer_layer',
        model_dims=16,
        hidden_dims=64,
        num_heads=8,
        mask_self_attention=mask_self_attention,
        num_layers=4,
        packed_input=packed_input,
        cross_attention=cross_attention)
    seq_len = np.random.randint(10, 32)
    batch_size = 10
    stacked_transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = stacked_transformer_layer.instantiate_variables(prng_key)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    segment_mask = None
    tf_segment_mask = None
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)
      if mask_self_attention:
        tf_segment_mask = batch_major_attention.CausalSegmentMask(
            segment_ids, tf.float32)
      else:
        tf_segment_mask = batch_major_attention.SegmentMask(
            segment_ids, segment_ids)

    cross_inputs = None
    cross_paddings = None
    cross_segment_mask = None
    tf_cross_inputs = None
    tf_cross_paddings = None
    tf_cross_segment_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 64)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5, [batch_size, cross_seq_len, p.model_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      tf_cross_inputs = tf.constant(npy_cross_inputs, dtype=tf.float32)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      tf_cross_paddings = tf.constant(npy_cross_paddings, dtype=tf.float32)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.segment_mask(
            segment_ids, source_segment_ids, dtype=np.float32)
        tf_cross_segment_mask = batch_major_attention.SegmentMask(
            segment_ids, source_segment_ids)

    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs = stacked_transformer_layer.fprop(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
    logging.info('initial_vars in transformer layer = %s', initial_vars)

    # Test whether tf Transformer layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = py_utils.NestedMap()
    tf_initial_vars.x_layers = []
    for jax_initial_vars in initial_vars.x_layers:
      tf_layer_vars = test_utils.replace_jax_attention_vars_to_tf(
          jax_initial_vars, cross_attention)
      tf_initial_vars.x_layers.append(tf_layer_vars)
    tf_initial_vars = test_utils.to_tf_nmap(tf_initial_vars)
    logging.info('tf_initial_vars in transformer layer = %s', initial_vars)
    tf_p = batch_major_attention.StackedTransformerLayers.Params().Set(
        name='tf_transformer_layer',
        mdl_dim=p.model_dims,
        hidden_dim=p.hidden_dims,
        num_atten_heads=p.num_heads,
        mask_self_atten=mask_self_attention,
        num_layers=p.num_layers,
        packed_input=packed_input,
        has_aux_atten=cross_attention)
    tf_p.transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.batch_norm = (
        False)
    tf_p.transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.has_bias = True
    tf_stacked_transformer_layer = tf_p.Instantiate()
    tf_output, _ = tf_stacked_transformer_layer.FProp(
        tf_initial_vars,
        test_utils.to_tf_nmap(npy_inputs),
        paddings=test_utils.to_tf_nmap(npy_paddings),
        segment_mask=test_utils.to_tf_nmap(tf_segment_mask),
        aux_vec=test_utils.to_tf_nmap(tf_cross_inputs),
        aux_paddings=test_utils.to_tf_nmap(tf_cross_paddings),
        aux_segment_mask=test_utils.to_tf_nmap(tf_cross_segment_mask))
    np_outputs = test_utils.to_np(outputs)
    tf_np_outputs = test_utils.to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-5)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_repeated_stacked_xformer_layer(self, mask_self_attention,
                                          packed_input, cross_attention):
    model_dims = 16
    p1 = transformers.StackedTransformer.Params().Set(
        name='jax_stacked_transformer_layer',
        model_dims=model_dims,
        hidden_dims=64,
        num_heads=8,
        mask_self_attention=mask_self_attention,
        num_layers=4,
        packed_input=packed_input,
        cross_attention=cross_attention)
    p1_one_layer = p1.Copy()
    p1_one_layer.num_layers = 1
    p2 = transformers.StackedTransformerRepeated.Params().Set(
        name='jax_stacked_transformer_layer_repeated',
        block=p1_one_layer,
        x_times=p1.num_layers)
    seq_len = np.random.randint(10, 32)
    batch_size = 10
    stacked_transformer_layer = p1.Instantiate()
    repeated_transformer_layer = p2.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)

    initial_vars = stacked_transformer_layer.instantiate_variables(prng_key)
    repeated_transformer_layer.instantiate_variable_configs()

    def _StackVars(*args):
      args = [x[jnp.newaxis, :] for x in args]
      return jnp.vstack(args)

    stacked_vars = tf.nest.map_structure(_StackVars, *initial_vars.x_layers)
    repeated_vars = py_utils.NestedMap(
        repeat=py_utils.NestedMap(
            sub=py_utils.NestedMap(x_layers=[stacked_vars])))

    tf.nest.assert_same_structure(
        repeated_vars,
        repeated_transformer_layer.instantiate_variables(prng_key))

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    segment_mask = None
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)

    cross_inputs = None
    cross_paddings = None
    cross_segment_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 64)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5, [batch_size, cross_seq_len, model_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.segment_mask(
            segment_ids, source_segment_ids, dtype=np.float32)

    with base_layer.JaxContext.new_context(
        prng_key=jax.random.PRNGKey(seed=1234),
        global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs = stacked_transformer_layer.fprop(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      outputs_repeated = repeated_transformer_layer.fprop(
          repeated_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      self.assertAllClose(outputs, outputs_repeated, atol=1e-5)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=5)))
  def test_stacked_transformer_layer_extendstep(self, packed_input,
                                                cross_attention, combine_qkv,
                                                dconv_qkv,
                                                use_rotary_position_emb):
    if cross_attention and combine_qkv:
      self.skipTest('combine_qkv optimization only works for self-attention.')
    layer_params = transformers.StackedTransformer.Params()

    num_layers = 2
    model_dims = 8
    p = layer_params.Set(
        name='jax_transformer_layer',
        model_dims=model_dims,
        hidden_dims=32,
        num_heads=2,
        mask_self_attention=True,
        packed_input=packed_input,
        cross_attention=cross_attention,
        num_layers=num_layers)
    p.transformer_layer_params_tpl.tr_atten_tpl.combine_qkv = combine_qkv
    p.transformer_layer_params_tpl.tr_atten_tpl.dconv_qkv = dconv_qkv
    p.transformer_layer_params_tpl.tr_atten_tpl.use_rotary_position_emb = (
        use_rotary_position_emb)
    if cross_attention:
      p.transformer_layer_params_tpl.cross_atten_tpl = (
          p.transformer_layer_params_tpl.tr_atten_tpl.Copy())
      # Cross attention should not have depth-wise convolution.
      p.transformer_layer_params_tpl.cross_atten_tpl.dconv_qkv = False
      # Cross attention should not have rotary position embedding.
      p.transformer_layer_params_tpl.cross_atten_tpl.use_rotary_position_emb = (
          False)

    p_copy = p.Copy()
    p_copy.num_layers = 1
    p = transformers.StackedTransformerRepeated.Params()
    p.name = 'jax_transformer_repeated_layer'
    p.block = p_copy
    p.x_times = num_layers

    seq_len = 4
    batch_size = 4
    stacked_transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = stacked_transformer_layer.instantiate_variables(prng_key)
    initial_states = stacked_transformer_layer.init_states(
        initial_vars, batch_size, seq_len)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    attention_mask = attentions.convert_paddings_to_mask(paddings)
    segment_mask = None
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)

    cross_inputs = None
    cross_paddings = None
    cross_segment_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 32)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5, [batch_size, cross_seq_len, model_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.segment_mask(
            segment_ids, source_segment_ids, dtype=np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    global_step = jnp.array(0, dtype=jnp.uint64)
    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=global_step):
      fprop_outputs = stacked_transformer_layer.fprop(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      decoder_outputs = jnp.zeros(shape=[seq_len, batch_size, model_dims])
      atten_states = initial_states
      for t in range(seq_len):
        segment_mask_t = attention_mask[:, :, t, :]
        cross_segment_mask_t = cross_segment_mask
        if segment_mask is not None:
          segment_mask_t = jnp.minimum(segment_mask_t, segment_mask[:, :, t, :])
        if cross_segment_mask is not None:
          cross_segment_mask_t = cross_segment_mask[:, :, t, :]
        atten_states, encoded = stacked_transformer_layer.extend_step(
            initial_vars,
            atten_states,
            inputs=inputs[:, t, :],
            time_step=t,
            segment_mask=segment_mask_t,
            cross_inputs=cross_inputs,
            cross_paddings=cross_paddings,
            cross_segment_mask=cross_segment_mask_t)
        decoder_outputs = decoder_outputs.at[t].set(encoded)

    decoder_out_transposed = jnp.transpose(decoder_outputs, [1, 0, 2])
    # TODO(lepikhin): remove noisy test logging
    # logging.info('initial_vars in transformer layer = %s', initial_vars)
    np_fprop_outputs = test_utils.to_np(fprop_outputs)
    np_decoder_outputs = test_utils.to_np(decoder_out_transposed)
    self.assertAllClose(np_fprop_outputs, np_decoder_outputs, atol=1e-5)

  @parameterized.parameters((True, True), (False, True), (True, False),
                            (False, False))
  def test_stacked_transformer_layer_while_loop(self, packed_input,
                                                cross_attention):
    num_layers = 2
    p1 = transformers.StackedTransformer.Params().Set(
        name='jax_transformer_layer',
        model_dims=8,
        hidden_dims=32,
        num_heads=2,
        mask_self_attention=True,
        packed_input=packed_input,
        cross_attention=cross_attention,
        num_layers=num_layers,
        enable_while_loop=False)
    p2 = transformers.StackedTransformer.Params().Set(
        name='jax_transformer_layer',
        model_dims=8,
        hidden_dims=32,
        num_heads=2,
        mask_self_attention=True,
        packed_input=packed_input,
        cross_attention=cross_attention,
        num_layers=num_layers,
        enable_while_loop=True)
    seq_len = 5
    batch_size = 4
    layer1 = p1.Instantiate()
    layer2 = p2.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = layer1.instantiate_variables(prng_key)
    layer2.instantiate_variable_configs()

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p1.model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    segment_mask = None
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)

    cross_inputs = None
    cross_paddings = None
    cross_segment_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 32)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5,
          [batch_size, cross_seq_len, p1.model_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.segment_mask(
            segment_ids, source_segment_ids, dtype=np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    global_step = jnp.array(0, dtype=jnp.uint64)
    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=global_step):
      fprop_outputs_1 = layer1.fprop(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      fprop_outputs_2 = layer2.fprop(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      all_summaries = base_layer.all_summaries()
      for key, value in all_summaries.items():
        logging.info('summary: %s', f'key:{key}, value:{value}')

    np_fprop_outputs_1 = test_utils.to_np(fprop_outputs_1)
    np_fprop_outputs_2 = test_utils.to_np(fprop_outputs_2)
    logging.info('np_fprop_outputs_1: %s', np_fprop_outputs_1)
    logging.info('np_fprop_outputs_2: %s', np_fprop_outputs_2)
    self.assertAllClose(np_fprop_outputs_1, np_fprop_outputs_2)

  @parameterized.parameters([True, False])
  def test_transformer_bert(self, trainable_position_emb):
    seq_len = 512
    if trainable_position_emb:
      position_emb_tpl = embedding_softmax.TrainablePositionalEmbedding.Params()
      position_emb_tpl.max_seq_length = seq_len
    else:
      position_emb_tpl = embedding_softmax.PositionalEmbedding.Params()
    p = transformers.TransformerLm.Params().Set(
        name='bert_lm',
        model_dims=32,
        vocab_size=52,
        position_emb_tpl=position_emb_tpl)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = 32
    stacked_transformer_tpl.hidden_dims = 4 * 32
    stacked_transformer_tpl.num_heads = 4
    stacked_transformer_tpl.num_layers = 1
    p.softmax_tpl.scale_sqrt_depth = True
    batch_size = 8
    bert_lm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = bert_lm.instantiate_variables(prng_key)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1234), [batch_size, seq_len], 0, 51)
    input_paddings = jnp.zeros([batch_size, seq_len])
    input_weights = jnp.ones([batch_size, seq_len])
    input_segment_ids = jnp.ones([batch_size, seq_len])
    input_segment_pos = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], [batch_size, 1])

    labels = py_utils.NestedMap()
    labels.class_ids = input_ids
    labels.class_weights = input_weights

    with base_layer.JaxContext.new_context(
        prng_key=jax.random.PRNGKey(seed=1234),
        global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs = bert_lm.fprop(
          initial_vars,
          input_ids,
          input_paddings,
          labels=labels,
          segment_ids=input_segment_ids,
          segment_pos=input_segment_pos)
      logging.info('outputs: %s', outputs)

  @parameterized.parameters('RELU', 'SILU', 'GATED_SILU')
  def test_transformer_feedforward(self, activation_function):
    p = transformers.TransformerFeedForward.Params().Set(
        name='ffwd',
        input_dims=8,
        hidden_dims=32,
        activation=activation_function)
    batch_size = 8
    seq_len = 512
    ffwd = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffwd.instantiate_variables(prng_key)

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.zeros([batch_size, seq_len], dtype=np.float32)
    input_paddings = jnp.asarray(npy_paddings)

    with base_layer.JaxContext.new_context(
        prng_key=jax.random.PRNGKey(seed=1234),
        global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs = ffwd.fprop(initial_vars, inputs, input_paddings)
      logging.info('outputs: %s', outputs)

    if activation_function.startswith('GATED_'):
      # Default lingvo layers_with_attention.TransformerFeedForwardLayer does
      # not support gating.
      return

    # Test whether Tensorflow TransformerFeedForwardLayer returns the same
    # output. Modify `initial_vars` to use TF compatible params.
    tf_initial_vars = test_utils.replace_jax_transformer_ffwd_vars_to_tf(
        initial_vars)
    tf_initial_vars = test_utils.to_tf_nmap(tf_initial_vars)
    logging.info('tf_initial_vars in transformer feedforward layer = %s',
                 initial_vars)
    tf_p = layers_with_attention.TransformerFeedForwardLayer.Params().Set(
        name='tf_ffwd',
        input_dim=p.input_dims,
        hidden_dim=p.hidden_dims,
        activation=p.activation)
    tf_ffwd = tf_p.Instantiate()
    tf_output = tf_ffwd.FProp(
        tf_initial_vars,
        tf.constant(npy_inputs, dtype=tf.float32),
        paddings=test_utils.to_tf_nmap(npy_paddings))
    np_outputs = test_utils.to_np(outputs)
    tf_np_outputs = test_utils.to_np(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-5)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_ngrammer_lm_extendstep(self, use_vq_ngrams, use_rotary_position_emb,
                                  share_embedding_and_softmax):
    vocab_size = 8
    num_layers = 2
    num_heads = 2
    dim_per_head = 8
    ngram_emb_dim = 4
    if use_vq_ngrams:
      ngrammer_params = ngrammer.VQNgrammer.Params().Set(
          ngram_vocab_size=64,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
    else:
      ngrammer_params = ngrammer.Ngrammer.Params().Set(
          ngram_vocab_size=64,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
    p = transformers.TransformerLm.Params().Set(
        name='jax_ngrammer_layer',
        model_dims=num_heads * dim_per_head,
        masked_lm=False,
        packed_input=False,
        ngrammer_tpl=ngrammer_params,
        vocab_size=vocab_size)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 4 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    if not share_embedding_and_softmax:
      p.separate_embedding_tpl = embedding_softmax.SingleShardEmbedding.Params()
      p.softmax_tpl = embedding_softmax.SingleShardFullSoftmax.Params()
    # Rotary position embedding.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    seq_len = 4
    batch_size = 2
    transformer_lm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_lm.instantiate_variables(prng_key)
    initial_states = transformer_lm.init_states(initial_vars, batch_size,
                                                seq_len)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)
    with base_layer.JaxContext.new_context(
        params=context_params,
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)):
      fprop_outputs = transformer_lm.fprop(initial_vars, inputs,
                                           jnp.zeros_like(inputs))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(seq_len):
        if t > 0:
          inputs_prefix = inputs[:, t - 1:t + 1]
        else:
          inputs_prefix = inputs[:, t]
        cached_states, xent_output = transformer_lm.extend_step(
            initial_vars, cached_states, inputs_prefix)
        self.assertAllClose(logits[:, t, :], xent_output.logits)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=2)))
  def test_primer_lm_extendstep(self, use_rotary_position_emb,
                                share_embedding_and_softmax):
    vocab_size = 8
    num_layers = 2
    num_heads = 2
    dim_per_head = 4
    dconv_kernel_size = 3
    p = transformers.TransformerLm.Params().Set(
        name='jax_primer_layer',
        model_dims=num_heads * dim_per_head,
        masked_lm=False,
        packed_input=False,
        vocab_size=vocab_size)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 2 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    if not share_embedding_and_softmax:
      p.separate_embedding_tpl = embedding_softmax.SingleShardEmbedding.Params()
      p.softmax_tpl = embedding_softmax.SingleShardFullSoftmax.Params()
    seq_len = 4
    batch_size = 3
    # Turn on dconv as in Primer.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.dconv_qkv = True
    # Rotary position embedding.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.dconv_kernel_size = dconv_kernel_size
    params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    transformer_lm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_lm.instantiate_variables(prng_key)
    initial_states = transformer_lm.init_states(initial_vars, batch_size,
                                                seq_len)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)
    with base_layer.JaxContext.new_context(
        params=context_params,
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)):
      fprop_outputs = transformer_lm.fprop(initial_vars, inputs,
                                           jnp.zeros_like(inputs))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(seq_len):
        cached_states, xent_output = transformer_lm.extend_step(
            initial_vars, cached_states, inputs[:, t])
        self.assertAllClose(logits[:, t, :], xent_output.logits)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_ngrammer_primer_lm_extendstep(self, use_vq_ngrams,
                                         use_rotary_position_emb,
                                         share_embedding_and_softmax):
    vocab_size = 8
    num_layers = 2
    num_heads = 2
    dim_per_head = 8
    ngram_emb_dim = 4
    dconv_kernel_size = 3
    if use_vq_ngrams:
      ngrammer_params = ngrammer.VQNgrammer.Params().Set(
          ngram_vocab_size=64,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
    else:
      ngrammer_params = ngrammer.Ngrammer.Params().Set(
          ngram_vocab_size=64,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
    p = transformers.TransformerLm.Params().Set(
        name='jax_ngrammer_layer',
        model_dims=num_heads * dim_per_head,
        masked_lm=False,
        packed_input=False,
        ngrammer_tpl=ngrammer_params,
        vocab_size=vocab_size)
    stacked_transformer_tpl = p.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = num_heads * dim_per_head
    stacked_transformer_tpl.hidden_dims = 4 * num_heads * dim_per_head
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = num_layers
    if not share_embedding_and_softmax:
      p.separate_embedding_tpl = embedding_softmax.SingleShardEmbedding.Params()
      p.softmax_tpl = embedding_softmax.SingleShardFullSoftmax.Params()
    seq_len = 4
    batch_size = 2
    # Turn on dconv as in Primer.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.dconv_qkv = True
    params.tr_atten_tpl.dconv_kernel_size = dconv_kernel_size
    # Rotary position embedding.
    params = p.stacked_transformer_tpl.transformer_layer_params_tpl
    params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    transformer_lm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_lm.instantiate_variables(prng_key)
    initial_states = transformer_lm.init_states(initial_vars, batch_size,
                                                seq_len)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)
    with base_layer.JaxContext.new_context(
        params=context_params,
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)):
      fprop_outputs = transformer_lm.fprop(initial_vars, inputs,
                                           jnp.zeros_like(inputs))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(seq_len):
        if t > 0:
          inputs_prefix = inputs[:, t - 1:t + 1]
        else:
          inputs_prefix = inputs[:, t]
        cached_states, xent_output = transformer_lm.extend_step(
            initial_vars, cached_states, inputs_prefix)
        self.assertAllClose(logits[:, t, :], xent_output.logits)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=8)))
  def test_transformer_encoder_decoder_extendstep(
      self, use_encoder_ngrams, use_decoder_ngrams, use_encoder_vq_ngrams,
      use_decoder_vq_ngrams, use_rotary_position_emb,
      separate_encoder_embedding, separate_decoder_embedding,
      use_stacked_transformer_repeated):
    vocab_size = 4
    num_layers = 2
    num_heads = 2
    dim_per_head = 4
    ngram_emb_dim = 2
    encoder_ngrammer_params = None
    decoder_ngrammer_params = None
    if use_encoder_vq_ngrams:
      encoder_ngrammer_params = ngrammer.VQNgrammer.Params().Set(
          ngram_vocab_size=8,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
    if use_encoder_ngrams:
      encoder_ngrammer_params = ngrammer.Ngrammer.Params().Set(
          ngram_vocab_size=16,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
    if use_decoder_vq_ngrams:
      decoder_ngrammer_params = ngrammer.VQNgrammer.Params().Set(
          ngram_vocab_size=8,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          num_clusters=2,
          dim_per_head=dim_per_head)
    if use_decoder_ngrams:
      decoder_ngrammer_params = ngrammer.Ngrammer.Params().Set(
          ngram_vocab_size=16,
          unigram_vocab_size=vocab_size,
          ngram_emb_dim=ngram_emb_dim,
          num_heads=num_heads,
          concat_ngrams=True,
          dim_per_head=dim_per_head)
    p = transformers.TransformerEncoderDecoder.Params().Set(
        name='jax_transformer_encoder_decoder',
        model_dims=num_heads * dim_per_head,
        decoder_ngrammer_tpl=decoder_ngrammer_params,
        encoder_ngrammer_tpl=encoder_ngrammer_params)

    # Encoder stack.
    if use_stacked_transformer_repeated:
      block_param = transformers.StackedTransformer.Params().Set(
          num_layers=num_layers,
          num_heads=num_heads,
          model_dims=num_heads * dim_per_head,
          hidden_dims=num_heads * dim_per_head,
          mask_self_attention=False,
          fold_padding_with_segment_mask=True)
      p.encoder_stacked_transformer_tpl = (
          transformers.StackedTransformerRepeated.Params().Set(
              block=block_param, x_times=1))
    else:
      p.encoder_stacked_transformer_tpl = (
          transformers.StackedTransformer.Params().Set(
              model_dims=num_heads * dim_per_head,
              hidden_dims=num_heads * dim_per_head,
              num_heads=num_heads,
              num_layers=num_layers,
              mask_self_attention=False,
              fold_padding_with_segment_mask=True))

    # Decoder stack.
    if use_stacked_transformer_repeated:
      block_param = transformers.StackedTransformer.Params().Set(
          num_layers=num_layers,
          num_heads=num_heads,
          model_dims=num_heads * dim_per_head,
          hidden_dims=num_heads * dim_per_head,
          mask_self_attention=True,
          fold_padding_with_segment_mask=True)
      p.decoder_stacked_transformer_tpl = (
          transformers.StackedTransformerRepeated.Params().Set(
              block=block_param, x_times=1))
    else:
      p.decoder_stacked_transformer_tpl = (
          transformers.StackedTransformer.Params().Set(
              model_dims=num_heads * dim_per_head,
              hidden_dims=num_heads * dim_per_head,
              num_heads=num_heads,
              num_layers=num_layers,
              mask_self_attention=True,
              fold_padding_with_segment_mask=True))

    if separate_encoder_embedding:
      p.encoder_embedding_tpl = (
          embedding_softmax.SingleShardEmbedding.Params().Set(
              vocab_size=vocab_size, embedding_dims=num_heads * dim_per_head))

    if separate_decoder_embedding:
      p.decoder_embedding_tpl = (
          embedding_softmax.SingleShardEmbedding.Params().Set(
              vocab_size=vocab_size, embedding_dims=num_heads * dim_per_head))

    # Softmax params.
    if separate_decoder_embedding:
      p.softmax_tpl = embedding_softmax.SingleShardFullSoftmax.Params().Set(
          input_dims=num_heads * dim_per_head, num_classes=vocab_size)
    else:
      p.softmax_tpl = (
          embedding_softmax.SingleShardSharedEmbeddingSoftmax.Params().Set(
              input_dims=num_heads * dim_per_head, num_classes=vocab_size))

    # Rotary position embedding.
    if use_rotary_position_emb:
      if use_stacked_transformer_repeated:
        params = p.encoder_stacked_transformer_tpl.block
      else:
        params = p.encoder_stacked_transformer_tpl
      params = params.transformer_layer_params_tpl
      params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
      if use_stacked_transformer_repeated:
        params = p.decoder_stacked_transformer_tpl.block
      else:
        params = p.decoder_stacked_transformer_tpl
      params = params.transformer_layer_params_tpl
      params.tr_atten_tpl.use_rotary_position_emb = use_rotary_position_emb
    p.position_emb_tpl = None

    seq_len = 4
    batch_size = 1
    transformer_enc_dec = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_enc_dec.instantiate_variables(prng_key)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    npy_input_paddings = np.random.randint(0, 2, size=(batch_size, seq_len))
    npy_targets = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    input_paddings = jnp.asarray(npy_input_paddings)
    targets = jnp.asarray(npy_targets)
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)
    with base_layer.JaxContext.new_context(
        params=context_params,
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)):
      initial_states = transformer_enc_dec.init_states(initial_vars, inputs,
                                                       input_paddings,
                                                       batch_size, seq_len)
      fprop_outputs = transformer_enc_dec.fprop(initial_vars, inputs,
                                                input_paddings, targets,
                                                jnp.zeros_like(targets))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(seq_len):
        targets_prefix = targets[:, t]
        if use_decoder_ngrams or use_decoder_vq_ngrams:
          if t > 0:
            targets_prefix = targets[:, t - 1:t + 1]
        cached_states, xent_output = transformer_enc_dec.extend_step(
            initial_vars, cached_states, targets_prefix)
        self.assertAllClose(logits[:, t, :], xent_output.logits, atol=2e-6)

  @parameterized.parameters(['pre', 'primer_hybrid'])
  def test_transformer_layer_norm_policies(self, norm_policy):
    p = transformers.Transformer.Params().Set(
        name='jax_transformer_layer',
        input_dims=32,
        hidden_dims=128,
        num_heads=8,
        mask_self_attention=True,
        packed_input=True,
        cross_attention=False,
        norm_policy=norm_policy)
    seq_len = np.random.randint(10, 32)
    batch_size = 10
    transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_layer.instantiate_variables(prng_key)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    attention_mask = attentions.convert_paddings_to_mask(paddings)
    causal_mask = attentions.causal_mask(inputs)
    attention_mask = jnp.minimum(attention_mask, causal_mask)
    segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
    segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)
    attention_mask = jnp.minimum(attention_mask, segment_mask)

    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs, _ = transformer_layer.fprop(
          initial_vars, inputs, paddings, attention_mask=attention_mask)
    logging.info('initial_vars in transformer layer = %s', initial_vars)

    np_outputs = test_utils.to_np(outputs)
    # Plumbing test.
    self.assertAllClose(np_outputs, np_outputs, atol=1e-5)

  @parameterized.parameters([True, False])
  def test_transformer_relative_bias(self, use_relative_bias):
    p = transformers.Transformer.Params().Set(
        name='jax_transformer_layer',
        input_dims=32,
        hidden_dims=128,
        num_heads=8,
        mask_self_attention=True,
        packed_input=True,
        cross_attention=False)
    seq_len = np.random.randint(10, 32)
    batch_size = 10
    if use_relative_bias:
      p.tr_atten_tpl.relative_bias_tpl = attentions.RelativeBias.Params().Set(
          relative_attention_num_buckets=2, relative_attention_max_distance=8)
    transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_layer.instantiate_variables(prng_key)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    attention_mask = attentions.convert_paddings_to_mask(paddings)
    causal_mask = attentions.causal_mask(inputs)
    attention_mask = jnp.minimum(attention_mask, causal_mask)
    segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
    segment_mask = attentions.segment_mask(segment_ids, dtype=np.float32)
    attention_mask = jnp.minimum(attention_mask, segment_mask)

    if use_relative_bias:
      segment_pos = np.random.randint(0, seq_len,
                                      [batch_size, seq_len]).astype('int32')
      segment_pos = jnp.asarray(segment_pos)
    else:
      segment_pos = None

    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs, _ = transformer_layer.fprop(
          initial_vars,
          inputs,
          paddings,
          attention_mask=attention_mask,
          segment_pos=segment_pos)
    logging.info('initial_vars in transformer layer = %s', initial_vars)

    np_outputs = test_utils.to_np(outputs)
    logging.info('np_outputs: %s', np_outputs)
    if use_relative_bias:
      self.assertAlmostEqual(np_outputs[0, 0, 1], 0.79015386, places=5)
      self.assertAlmostEqual(np_outputs[0, 1, 0], 0.48336178, places=5)
    # Plumbing test.
    self.assertAllClose(np_outputs, np_outputs, atol=1e-5)

  def testGlamUniTransformer(self):
    batch = 2
    length = 3
    d_model = 6
    num_heads = 2
    vocab_size = 16
    ff_dim = 8
    c_dim = 3
    e_dim = 2
    num_layers = 4
    # Build jax layer
    jax_p = transformers.TransformerLm.GLaMUniTransformerParams(
        name='model',
        vocab_size=vocab_size,
        num_transformer_layers=num_layers,
        moe=True,
        model_dim=d_model,
        ff_dim=ff_dim,
        moe_hidden_dim=ff_dim,
        attention_num_heads=num_heads,
        attention_key_value_dim=d_model // num_heads,
        attention_extra_logit=0.0,
        moe_load_balance_loss_weight=0.01,
        z_loss_weight=1e-4,
        c_dim=c_dim,
        e_dim=e_dim)
    assert jax_p.packed_input
    jax_layer = jax_p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=42)
    jax_vars = jax_layer.instantiate_variables(prng_key)

    builder_p = gshard_builder.DenseBuilder.Params().Set(
        num_groups=1,
        second_expert_policy='all',
        relative_attention_type='bias',
        model_dim=d_model,
        attention_key_value_dim=d_model // num_heads,
        attention_num_heads=num_heads,
        attention_combine_dims=True,
        c_dim=c_dim,
        capacity_factor=None,
        attention_extra_logit=0.0,
        e_dim=e_dim,
        moe_hidden_dim=ff_dim,
        ff_dim=ff_dim)
    tf_layer = gshard_builder.UniTransformer.Params().Set(
        name='model',
        num_transformer_layers=num_layers,
        builder=builder_p,
        vocab_size=vocab_size,
        sequence_length=length,
        label_smoothing=0,
        aux_loss_coef=0.01,
        z_loss=1e-4,
        use_tgt_labels_size_as_loss_denominator=False,
        positional_embedding=False,
        gated_gelu=True,
        moe=True).Instantiate()

    # Build Jax Inputs
    np.random.seed(42)
    npy_ids = np.random.randint(0, vocab_size - 1, [batch, length])
    jax_ids = jnp.asarray(npy_ids)
    npy_paddings = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)

    jax_paddings = jnp.asarray(npy_paddings)
    npy_segment_ids = np.array([[1, 2, 0], [1, 1, 0]], dtype=np.int32)
    npy_segment_pos = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int32)
    npy_labels = np.roll(npy_ids, -1, axis=1)
    jax_labels = jnp.asarray(npy_labels)
    jax_seg_ids = jnp.asarray(npy_segment_ids)
    jax_seg_pos = jnp.asarray(npy_segment_pos)
    jax_label_weighs = jnp.asarray([[1, 1, 0], [1, 1, 0]])

    # Build TF Inputs
    tf_tgt_inputs = py_utils.NestedMap(
        ids=tf.convert_to_tensor(npy_ids, dtype=tf.int32),
        labels=tf.convert_to_tensor(npy_labels, dtype=tf.int32),
        segment_ids=tf.convert_to_tensor(npy_segment_ids, dtype=tf.int32),
        segment_pos=tf.convert_to_tensor(npy_segment_pos, dtype=tf.int32))
    tf_inputs = py_utils.NestedMap(tgt=tf_tgt_inputs)

    with base_layer.JaxContext.new_context(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):

      # Compute jax outputs
      jax_outputs = jax_layer.fprop(
          jax_vars,
          jax_ids,
          jax_paddings,
          labels=py_utils.NestedMap(
              class_ids=jax_labels,
              class_weights=jax_label_weighs,
          ),
          segment_ids=jax_seg_ids,
          segment_pos=jax_seg_pos)

      # Copy jax vars to tf ones.
      tf_theta = tf_layer.theta.DeepCopy()

      # GShardBuilder softmax weight use self.vars rather than theta.
      tf_layer.vars.dec_emb.w.embedding.assign(jax_vars.softmax.embedding.w)
      tf_theta.dec_emb.w.embedding = jax_vars.softmax.embedding.w
      tf_theta.dec.final_layer_norm.w.scale = jax_vars.final_ln.scale
      jax_layer_0_var = tf.nest.map_structure(
          lambda v: jnp.squeeze(jnp.split(v, 2)[0], axis=0),
          jax_vars.transformer.repeat.sub.x_layers[0])
      tf_theta.dec.layer_000.ln.w.scale = jax_layer_0_var.layer_norm.scale
      jax_atten_var = jax_layer_0_var.self_attention
      tf_atten_var = tf_theta.dec.layer_000.dec_self_attention
      tf_atten_var.w.wk = jax_atten_var.key.w
      tf_atten_var.w.wq = jax_atten_var.query.w
      tf_atten_var.w.wv = jax_atten_var.value.w
      tf_atten_var.w.wo = jax_atten_var.post.w
      tf_atten_var.wrb.wrb = jax_atten_var.relative_bias.wrb

      jax_moe_var = jax_layer_0_var.ff_layer
      tf_theta.dec.layer_001.ln.w.scale = jax_moe_var.layer_norm.scale
      tf_theta.dec.layer_001.moe.ffw.top_2_gating.w = jax_moe_var.gate
      tf_theta.dec.layer_001.moe.moe.wi = jax_moe_var.wi_0
      tf_theta.dec.layer_001.moe.moe.wo = jax_moe_var.wo_0

      jax_layer_1_var = tf.nest.map_structure(
          lambda v: jnp.squeeze(jnp.split(v, 2)[0], axis=0),
          jax_vars.transformer.repeat.sub.x_layers[1])
      tf_theta.dec.layer_002.ln.w.scale = jax_layer_1_var.layer_norm.scale
      jax_atten_var = jax_layer_1_var.self_attention
      tf_atten_var = tf_theta.dec.layer_002.dec_self_attention
      tf_atten_var.w.wk = jax_atten_var.key.w
      tf_atten_var.w.wq = jax_atten_var.query.w
      tf_atten_var.w.wv = jax_atten_var.value.w
      tf_atten_var.w.wo = jax_atten_var.post.w
      tf_atten_var.wrb.wrb = jax_atten_var.relative_bias.wrb

      jax_ffn_var = jax_layer_1_var.ff_layer
      tf_ffn_var = tf_theta.dec.layer_003.dense_relu_dense
      tf_ffn_var.w.wi_0 = jax_ffn_var.ffn_layer1_gate.linear.w
      tf_ffn_var.w.wi_1 = jax_ffn_var.ffn_layer1.linear.w
      tf_ffn_var.w.wo = jax_ffn_var.ffn_layer2.linear.w
      tf_theta.dec.layer_003.ln.w.scale = jax_ffn_var.layer_norm.scale

      jax_layer_2_var = tf.nest.map_structure(
          lambda v: jnp.squeeze(jnp.split(v, 2)[1], axis=0),
          jax_vars.transformer.repeat.sub.x_layers[0])
      tf_theta.dec.layer_004.ln.w.scale = jax_layer_2_var.layer_norm.scale
      jax_atten_var = jax_layer_2_var.self_attention
      tf_atten_var = tf_theta.dec.layer_004.dec_self_attention
      tf_atten_var.w.wk = jax_atten_var.key.w
      tf_atten_var.w.wq = jax_atten_var.query.w
      tf_atten_var.w.wv = jax_atten_var.value.w
      tf_atten_var.w.wo = jax_atten_var.post.w
      tf_atten_var.wrb.wrb = jax_atten_var.relative_bias.wrb

      jax_moe_var = jax_layer_2_var.ff_layer
      tf_theta.dec.layer_005.ln.w.scale = jax_moe_var.layer_norm.scale
      tf_theta.dec.layer_005.moe.ffw.top_2_gating.w = jax_moe_var.gate
      tf_theta.dec.layer_005.moe.moe.wi = jax_moe_var.wi_0
      tf_theta.dec.layer_005.moe.moe.wo = jax_moe_var.wo_0

      jax_layer_3_var = tf.nest.map_structure(
          lambda v: jnp.squeeze(jnp.split(v, 2)[1], axis=0),
          jax_vars.transformer.repeat.sub.x_layers[1])
      tf_theta.dec.layer_006.ln.w.scale = jax_layer_3_var.layer_norm.scale
      jax_atten_var = jax_layer_3_var.self_attention
      tf_atten_var = tf_theta.dec.layer_006.dec_self_attention
      tf_atten_var.w.wk = jax_atten_var.key.w
      tf_atten_var.w.wq = jax_atten_var.query.w
      tf_atten_var.w.wv = jax_atten_var.value.w
      tf_atten_var.w.wo = jax_atten_var.post.w
      tf_atten_var.wrb.wrb = jax_atten_var.relative_bias.wrb

      jax_ffn_var = jax_layer_3_var.ff_layer
      tf_ffn_var = tf_theta.dec.layer_007.dense_relu_dense
      tf_ffn_var.w.wi_0 = jax_ffn_var.ffn_layer1_gate.linear.w
      tf_ffn_var.w.wi_1 = jax_ffn_var.ffn_layer1.linear.w
      tf_ffn_var.w.wo = jax_ffn_var.ffn_layer2.linear.w
      tf_theta.dec.layer_007.ln.w.scale = jax_ffn_var.layer_norm.scale

      tf_theta = test_utils.to_tf_nmap(tf_theta)

      # Compute TF outputs
      tf_out, _ = tf_layer.FProp(tf_theta, tf_inputs)
      self.assertAllClose(
          test_utils.to_np(jax_outputs.avg_xent),
          test_utils.to_np(tf_out['mean_xent'][0]))
      self.assertAllClose(
          test_utils.to_np(jax_outputs.total_loss),
          test_utils.to_np(tf_out['loss'][0]))

if __name__ == '__main__':
  absltest.main()
