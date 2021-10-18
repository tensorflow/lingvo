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
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax.layers import attentions
from lingvo.jax.layers import transformers
import numpy as np
import tensorflow.compat.v2 as tf


class TransformersTest(test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_transformer_layer(self, mask_self_attention, packed_input,
                             cross_attention):
    p = transformers.TransformerLayer.Params().Set(
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
    initial_vars = transformer_layer.InstantiateVariables(prng_key)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    causal_mask = None
    segment_mask = None
    tf_segment_mask = None
    attention_mask = attentions.ConvertPaddingsToMask(paddings)
    if mask_self_attention:
      causal_mask = attentions.CausalMask(inputs)
      attention_mask = jnp.minimum(attention_mask, causal_mask)
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.SegmentMask(segment_ids, dtype=np.float32)
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
      cross_attention_mask = attentions.ConvertPaddingsToMask(cross_paddings)
      tf_cross_paddings = tf.constant(npy_cross_paddings, dtype=tf.float32)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.SegmentMask(
            segment_ids, source_segment_ids, dtype=np.float32)
        cross_attention_mask = jnp.minimum(cross_attention_mask,
                                           cross_segment_mask)
        tf_cross_segment_mask = batch_major_attention.SegmentMask(
            segment_ids, source_segment_ids)

    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs, _ = transformer_layer.FProp(
          initial_vars,
          inputs,
          paddings,
          attention_mask=attention_mask,
          cross_inputs=cross_inputs,
          cross_attention_mask=cross_attention_mask)
    logging.info('initial_vars in transformer layer = %s', initial_vars)

    # Test whether tf Transformer layer returns same output
    # Modify initial_vars to use TF compatible params
    tf_initial_vars = test_utils.ReplaceJaxAttentionVarsToTf(
        initial_vars, cross_attention)
    tf_initial_vars = test_utils.ToTfNmap(tf_initial_vars)
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
        paddings=test_utils.ToTfNmap(npy_paddings),
        segment_mask=tf_segment_mask,
        aux_vec=tf_cross_inputs,
        aux_paddings=tf_cross_paddings,
        aux_segment_mask=test_utils.ToTfNmap(tf_cross_segment_mask))
    np_outputs = test_utils.ToNp(outputs)
    tf_np_outputs = test_utils.ToNp(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-5)

  @parameterized.parameters((True, True), (False, True), (True, False),
                            (False, False))
  def test_transformer_layer_extendstep(self, packed_input, cross_attention):
    p = transformers.TransformerLayer.Params().Set(
        name='jax_transformer_layer',
        input_dims=8,
        hidden_dims=32,
        num_heads=4,
        mask_self_attention=True,
        packed_input=packed_input,
        cross_attention=cross_attention)
    seq_len = 5
    batch_size = 4
    transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_layer.InstantiateVariables(prng_key)
    initial_states = transformer_layer.InitStates(initial_vars, batch_size,
                                                  seq_len)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    attention_mask = attentions.ConvertPaddingsToMask(paddings)
    segment_mask = None
    causal_mask = attentions.CausalMask(inputs)
    attention_mask = jnp.minimum(causal_mask, attention_mask)
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.SegmentMask(segment_ids, dtype=np.float32)
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
      cross_attention_mask = attentions.ConvertPaddingsToMask(cross_paddings)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.SegmentMask(
            segment_ids, source_segment_ids, dtype=np.float32)
        cross_attention_mask = jnp.minimum(cross_attention_mask,
                                           cross_segment_mask)

    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      fprop_outputs, _ = transformer_layer.FProp(
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
        atten_states, encoded = transformer_layer.ExtendStep(
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
    np_fprop_outputs = test_utils.ToNp(fprop_outputs)
    np_decoder_outputs = test_utils.ToNp(decoder_out_transposed)
    self.assertAllClose(np_fprop_outputs, np_decoder_outputs, atol=1e-5)

  @parameterized.parameters((True, True, True), (True, False, True),
                            (True, True, False), (False, True, True),
                            (True, False, False), (False, True, False),
                            (False, False, True), (False, False, False))
  def test_stacked_transformer_layer(self, mask_self_attention, packed_input,
                                     cross_attention):
    p = transformers.StackedTransformerLayers.Params().Set(
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
    initial_vars = stacked_transformer_layer.InstantiateVariables(prng_key)
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
      segment_mask = attentions.SegmentMask(segment_ids, dtype=np.float32)
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
        cross_segment_mask = attentions.SegmentMask(
            segment_ids, source_segment_ids, dtype=np.float32)
        tf_cross_segment_mask = batch_major_attention.SegmentMask(
            segment_ids, source_segment_ids)

    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs = stacked_transformer_layer.FProp(
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
      tf_layer_vars = test_utils.ReplaceJaxAttentionVarsToTf(
          jax_initial_vars, cross_attention)
      tf_initial_vars.x_layers.append(tf_layer_vars)
    tf_initial_vars = test_utils.ToTfNmap(tf_initial_vars)
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
        test_utils.ToTfNmap(npy_inputs),
        paddings=test_utils.ToTfNmap(npy_paddings),
        segment_mask=test_utils.ToTfNmap(tf_segment_mask),
        aux_vec=test_utils.ToTfNmap(tf_cross_inputs),
        aux_paddings=test_utils.ToTfNmap(tf_cross_paddings),
        aux_segment_mask=test_utils.ToTfNmap(tf_cross_segment_mask))
    np_outputs = test_utils.ToNp(outputs)
    tf_np_outputs = test_utils.ToNp(tf_output)
    self.assertAllClose(tf_np_outputs, np_outputs, atol=1e-5)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=3)))
  def test_repeated_stacked_xformer_layer(self, mask_self_attention,
                                          packed_input, cross_attention):
    model_dims = 16
    p1 = transformers.StackedTransformerLayers.Params().Set(
        name='jax_stacked_transformer_layer',
        model_dims=model_dims,
        hidden_dims=64,
        num_heads=8,
        mask_self_attention=mask_self_attention,
        num_layers=4,
        packed_input=packed_input,
        cross_attention=cross_attention)
    p2 = transformers.StackedTransformerLayersRepeated.Params().Set(
        name='jax_stacked_transformer_layer_repeated',
        model_dims=model_dims,
        hidden_dims=64,
        num_heads=8,
        mask_self_attention=mask_self_attention,
        num_layers=4,
        packed_input=packed_input,
        cross_attention=cross_attention)
    seq_len = np.random.randint(10, 32)
    batch_size = 10
    stacked_transformer_layer = p1.Instantiate()
    repeated_transformer_layer = p2.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = stacked_transformer_layer.InstantiateVariables(prng_key)
    repeated_transformer_layer.InstantiateVariableConfigs()

    def _StackVars(*args):
      args = [x[jnp.newaxis, :] for x in args]
      return jnp.vstack(args)

    stacked_vars = py_utils.NestedMap(
        repeat=py_utils.NestedMap(
            sub=tf.nest.map_structure(_StackVars, *initial_vars.x_layers)))

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    segment_mask = None
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.SegmentMask(segment_ids, dtype=np.float32)

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
        cross_segment_mask = attentions.SegmentMask(
            segment_ids, source_segment_ids, dtype=np.float32)

    with base_layer.JaxContext.NewContext(
        prng_key=jax.random.PRNGKey(seed=1234),
        global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs = stacked_transformer_layer.FProp(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      outputs_repeated = repeated_transformer_layer.FProp(
          stacked_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      self.assertAllClose(outputs, outputs_repeated)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=5)))
  def test_stacked_transformer_layer_extendstep(self, packed_input,
                                                cross_attention,
                                                enable_while_loop,
                                                use_repeat_layer, combine_qkv):
    if cross_attention and combine_qkv:
      self.skipTest('combine_qkv optimization only works for self-attention')
    if use_repeat_layer:
      layer_params = transformers.StackedTransformerLayersRepeated.Params()
    else:
      layer_params = transformers.StackedTransformerLayers.Params()

    p = layer_params.Set(
        name='jax_transformer_layer',
        model_dims=8,
        hidden_dims=32,
        num_heads=2,
        mask_self_attention=True,
        packed_input=packed_input,
        cross_attention=cross_attention,
        num_layers=2,
        enable_while_loop=enable_while_loop)
    p.transformer_layer_params_tpl.tr_atten_tpl.combine_qkv = combine_qkv
    seq_len = 5
    batch_size = 4
    stacked_transformer_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = stacked_transformer_layer.InstantiateVariables(prng_key)
    initial_states = stacked_transformer_layer.InitStates(
        initial_vars, batch_size, seq_len)
    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    attention_mask = attentions.ConvertPaddingsToMask(paddings)
    segment_mask = None
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.SegmentMask(segment_ids, dtype=np.float32)

    cross_inputs = None
    cross_paddings = None
    cross_segment_mask = None
    if cross_attention:
      cross_seq_len = np.random.randint(10, 32)
      npy_cross_inputs = np.random.normal(
          1.0, 0.5, [batch_size, cross_seq_len, p.model_dims]).astype('float32')
      cross_inputs = jnp.asarray(npy_cross_inputs)
      npy_cross_paddings = np.random.randint(
          0, 1, [batch_size, cross_seq_len]).astype('float32')
      cross_paddings = jnp.asarray(npy_cross_paddings)
      if packed_input:
        source_segment_ids = np.random.random_integers(
            0, 2, [batch_size, cross_seq_len])
        cross_segment_mask = attentions.SegmentMask(
            segment_ids, source_segment_ids, dtype=np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    global_step = jnp.array(0, dtype=jnp.uint64)
    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=global_step):
      fprop_outputs = stacked_transformer_layer.FProp(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      decoder_outputs = jnp.zeros(shape=[seq_len, batch_size, p.model_dims])
      atten_states = initial_states
      for t in range(seq_len):
        segment_mask_t = attention_mask[:, :, t, :]
        cross_segment_mask_t = cross_segment_mask
        if segment_mask is not None:
          segment_mask_t = jnp.minimum(segment_mask_t, segment_mask[:, :, t, :])
        if cross_segment_mask is not None:
          cross_segment_mask_t = cross_segment_mask[:, :, t, :]
        atten_states, encoded = stacked_transformer_layer.ExtendStep(
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
    logging.info('initial_vars in transformer layer = %s', initial_vars)
    np_fprop_outputs = test_utils.ToNp(fprop_outputs)
    np_decoder_outputs = test_utils.ToNp(decoder_out_transposed)
    self.assertAllClose(np_fprop_outputs, np_decoder_outputs, atol=1e-5)

  @parameterized.parameters((True, True), (False, True), (True, False),
                            (False, False))
  def test_stacked_transformer_layer_while_loop(self, packed_input,
                                                cross_attention):
    num_layers = 2
    p1 = transformers.StackedTransformerLayers.Params().Set(
        name='jax_transformer_layer',
        model_dims=8,
        hidden_dims=32,
        num_heads=2,
        mask_self_attention=True,
        packed_input=packed_input,
        cross_attention=cross_attention,
        num_layers=num_layers,
        enable_while_loop=False)
    p2 = transformers.StackedTransformerLayers.Params().Set(
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
    initial_vars = layer1.InstantiateVariables(prng_key)
    layer2.InstantiateVariableConfigs()

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p1.model_dims]).astype('float32')
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.random.randint(0, 1,
                                     [batch_size, seq_len]).astype('float32')
    paddings = jnp.asarray(npy_paddings)
    segment_mask = None
    if packed_input:
      segment_ids = np.random.random_integers(0, 2, [batch_size, seq_len])
      segment_mask = attentions.SegmentMask(segment_ids, dtype=np.float32)

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
        cross_segment_mask = attentions.SegmentMask(
            segment_ids, source_segment_ids, dtype=np.float32)

    prng_key = jax.random.PRNGKey(seed=123)
    global_step = jnp.array(0, dtype=jnp.uint64)
    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=global_step):
      fprop_outputs_1 = layer1.FProp(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)
      fprop_outputs_2 = layer2.FProp(
          initial_vars,
          inputs,
          paddings,
          segment_mask=segment_mask,
          cross_inputs=cross_inputs,
          cross_paddings=cross_paddings,
          cross_segment_mask=cross_segment_mask)

    np_fprop_outputs_1 = test_utils.ToNp(fprop_outputs_1)
    np_fprop_outputs_2 = test_utils.ToNp(fprop_outputs_2)
    logging.info('np_fprop_outputs_1: %s', np_fprop_outputs_1)
    logging.info('np_fprop_outputs_2: %s', np_fprop_outputs_2)
    self.assertAllClose(np_fprop_outputs_1, np_fprop_outputs_2)

  def test_transformer_bert(self):
    """Test JAX and TF transformer on PTB."""
    p = transformers.TransformerLm.Params().Set(
        name='bert_lm',
        model_dims=32,
        hidden_dims=4 * 32,
        num_heads=4,
        num_layers=1,
        vocab_size=52)
    p.softmax_tpl.scale_sqrt_depth = True
    batch_size = 8
    seq_len = 512
    bert_lm = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = bert_lm.InstantiateVariables(prng_key)
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

    with base_layer.JaxContext.NewContext(
        prng_key=jax.random.PRNGKey(seed=1234),
        global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs = bert_lm.FProp(
          initial_vars,
          input_ids,
          input_paddings,
          labels=labels,
          segment_ids=input_segment_ids,
          segment_pos=input_segment_pos)
      logging.info('outputs: %s', outputs)

  @parameterized.parameters(('RELU',), ('GATED_SILU',))
  def test_gated_ffwd(self, activation_function):
    """Test JAX and TF transformer on PTB."""
    p = transformers.TransformerFeedForwardLayer.Params().Set(
        name='ffwd',
        input_dims=8,
        hidden_dims=32,
        activation=activation_function)
    batch_size = 8
    seq_len = 512
    ffwd = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = ffwd.InstantiateVariables(prng_key)
    inputs = jax.random.normal(
        jax.random.PRNGKey(1234), [batch_size, seq_len, 8])
    input_paddings = jnp.zeros([batch_size, seq_len])

    with base_layer.JaxContext.NewContext(
        prng_key=jax.random.PRNGKey(seed=1234),
        global_step=jnp.array(0, dtype=jnp.uint32)):
      outputs = ffwd.FProp(initial_vars, inputs, input_paddings)
      logging.info('outputs: %s', outputs)


if __name__ == '__main__':
  absltest.main()
