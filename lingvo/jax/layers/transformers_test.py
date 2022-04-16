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
from lingvo.core import batch_major_attention
from lingvo.core import layers_with_attention
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax.layers import attentions
from lingvo.jax.layers import transformers
import numpy as np
import tensorflow.compat.v2 as tf


class TransformersTest(test_utils.TestCase):

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

    outputs, _ = test_utils.apply(
        transformer_layer,
        initial_vars,
        transformer_layer.fprop,
        inputs,
        paddings,
        context_p=None,
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
    initial_states = transformer_layer.init_states(batch_size, seq_len)
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
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_layer,
                       transformer_layer.vars_to_flax_vars(initial_vars))
      fprop_outputs, _ = transformer_layer.fprop(
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
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_layer,
                       transformer_layer.vars_to_flax_vars(initial_vars))
      inputs_normalized = transformer_layer.layer_norm.fprop(inputs)
      # Compute self-attention, key/value vectors are the input itself
      atten_output, _ = transformer_layer.self_attention.fprop(
          inputs_normalized,
          inputs_normalized,
          inputs_normalized,
          atten_mask=attention_mask)
      # Residual dropout and connection.
      atten_output = transformer_layer.residual_dropout.fprop(atten_output)
      atten_output += inputs
      # Normalize atten outputs using cross attention.
      atten_output_normalized = transformer_layer.cross_layer_norm.fprop(
          atten_output)
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
        num_layers=2,
        model_dims=3,
        hidden_dims=6,
        num_heads=1,
        mask_self_attention=mask_self_attention,
        packed_input=packed_input,
        cross_attention=cross_attention,
        num_experts=4,
        num_groups=1,
        moe_layers=[0])

    block_p_repeated = transformers.StackedTransformerRepeated.Params().Set(
        name='stacked_transformer_layer_repeated',
        block=block_p.Copy(),
        x_times=1)

    stack_p = transformers.StackedTransformer.Params().Set(
        name='transformer_stack',
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

    transformer_block = block_p_repeated.Instantiate()
    transformer_stack = stack_p.Instantiate()

    seq_len = 4
    batch_size = 3
    prng_key = jax.random.PRNGKey(seed=123)
    block_initial_vars = transformer_block.instantiate_variables(prng_key)
    stack_initial_vars = transformer_stack.instantiate_variables(prng_key)
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

    block_outputs = test_utils.apply(
        transformer_block,
        block_initial_vars,
        transformer_block.fprop,
        inputs,
        paddings,
        segment_mask=segment_mask,
        cross_inputs=cross_inputs,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask)

    stack_outputs = test_utils.apply(
        transformer_stack,
        stack_initial_vars,
        transformer_stack.fprop,
        inputs,
        paddings,
        segment_mask=segment_mask,
        cross_inputs=cross_inputs,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask)
    _ = test_utils.to_np(block_outputs)
    _ = test_utils.to_np(stack_outputs)

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

    # test conversion between vars and flax vars.
    pax_vars = stacked_transformer_layer.vars
    flax_vars = stacked_transformer_layer.flax_vars
    tf.nest.assert_same_structure(
        pax_vars, stacked_transformer_layer.flax_vars_to_vars(flax_vars))
    tf.nest.assert_same_structure(
        flax_vars, stacked_transformer_layer.vars_to_flax_vars(pax_vars))

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

    outputs = test_utils.apply(
        stacked_transformer_layer,
        initial_vars,
        stacked_transformer_layer.fprop,
        inputs,
        paddings,
        context_p=None,
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

    def _stack_vars(*args):
      args = [x[jnp.newaxis, :] for x in args]
      return jnp.vstack(args)

    stacked_vars = tf.nest.map_structure(_stack_vars, *initial_vars.x_layers)
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

    outputs = test_utils.apply(
        stacked_transformer_layer,
        initial_vars,
        stacked_transformer_layer.fprop,
        inputs,
        paddings,
        context_p=None,
        segment_mask=segment_mask,
        cross_inputs=cross_inputs,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask)

    outputs_repeated = test_utils.apply(
        repeated_transformer_layer,
        repeated_vars,
        repeated_transformer_layer.fprop,
        inputs,
        paddings,
        context_p=None,
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
        prng_key=prng_key, global_step=global_step) as jax_context:
      jax_context.bind(
          stacked_transformer_layer,
          stacked_transformer_layer.vars_to_flax_vars(initial_vars))
      fprop_outputs = stacked_transformer_layer.fprop(
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      decoder_outputs = jnp.zeros(shape=[seq_len, batch_size, model_dims])
      initial_states = stacked_transformer_layer.init_states(
          batch_size, seq_len)
      atten_states = initial_states
      for t in range(seq_len):
        segment_mask_t = attention_mask[:, :, t, :]
        cross_segment_mask_t = cross_segment_mask
        if segment_mask is not None:
          segment_mask_t = jnp.minimum(segment_mask_t, segment_mask[:, :, t, :])
        if cross_segment_mask is not None:
          cross_segment_mask_t = cross_segment_mask[:, :, t, :]
        atten_states, encoded = stacked_transformer_layer.extend_step(
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
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(ffwd, ffwd.vars_to_flax_vars(initial_vars))
      outputs = ffwd.fprop(inputs, input_paddings)
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
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_layer,
                       transformer_layer.vars_to_flax_vars(initial_vars))
      outputs, _ = transformer_layer.fprop(
          inputs, paddings, attention_mask=attention_mask)
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
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_layer,
                       transformer_layer.vars_to_flax_vars(initial_vars))
      outputs, _ = transformer_layer.fprop(
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


if __name__ == '__main__':
  absltest.main()
