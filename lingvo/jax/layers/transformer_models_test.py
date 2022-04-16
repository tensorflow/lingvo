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
from lingvo.core import gshard_builder
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax.layers import embedding_softmax
from lingvo.jax.layers import ngrammer
from lingvo.jax.layers import transformer_models
from lingvo.jax.layers import transformers
import numpy as np
import tensorflow.compat.v2 as tf


class TransformerModelsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.parameters([True, False])
  def test_transformer_bert(self, trainable_position_emb):
    seq_len = 512
    if trainable_position_emb:
      position_emb_tpl = embedding_softmax.TrainablePositionalEmbedding.Params()
      position_emb_tpl.max_seq_length = seq_len
    else:
      position_emb_tpl = embedding_softmax.PositionalEmbedding.Params()
    p = transformer_models.TransformerLm.Params().Set(
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
    outputs = test_utils.apply(
        bert_lm,
        initial_vars,
        bert_lm.fprop,
        input_ids,
        input_paddings,
        labels=labels,
        segment_ids=input_segment_ids,
        segment_pos=input_segment_pos)
    logging.info('outputs: %s', outputs)

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
    p = transformer_models.TransformerLm.Params().Set(
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
    initial_states = transformer_lm.init_states(batch_size, seq_len)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)
    with base_layer.JaxContext.new_context(
        params=context_params,
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_lm,
                       transformer_lm.vars_to_flax_vars(initial_vars))
      fprop_outputs = transformer_lm.fprop(inputs, jnp.zeros_like(inputs))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(seq_len):
        if t > 0:
          inputs_prefix = inputs[:, t - 1:t + 1]
        else:
          inputs_prefix = inputs[:, t]
        cached_states, xent_output = transformer_lm.extend_step(
            cached_states, inputs_prefix)
        self.assertAllClose(logits[:, t, :], xent_output.logits)

  @parameterized.parameters(*list(itertools.product([True, False], repeat=2)))
  def test_primer_lm_extendstep(self, use_rotary_position_emb,
                                share_embedding_and_softmax):
    vocab_size = 8
    num_layers = 2
    num_heads = 2
    dim_per_head = 4
    dconv_kernel_size = 3
    p = transformer_models.TransformerLm.Params().Set(
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
    initial_states = transformer_lm.init_states(batch_size, seq_len)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)
    with base_layer.JaxContext.new_context(
        params=context_params,
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_lm,
                       transformer_lm.vars_to_flax_vars(initial_vars))
      fprop_outputs = transformer_lm.fprop(inputs, jnp.zeros_like(inputs))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(seq_len):
        cached_states, xent_output = transformer_lm.extend_step(
            cached_states, inputs[:, t])
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
    p = transformer_models.TransformerLm.Params().Set(
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
    initial_states = transformer_lm.init_states(batch_size, seq_len)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch_size, seq_len)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)
    with base_layer.JaxContext.new_context(
        params=context_params,
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_lm,
                       transformer_lm.vars_to_flax_vars(initial_vars))
      fprop_outputs = transformer_lm.fprop(inputs, jnp.zeros_like(inputs))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(seq_len):
        if t > 0:
          inputs_prefix = inputs[:, t - 1:t + 1]
        else:
          inputs_prefix = inputs[:, t]
        cached_states, xent_output = transformer_lm.extend_step(
            cached_states, inputs_prefix)
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
    p = transformer_models.TransformerEncoderDecoder.Params().Set(
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
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_enc_dec,
                       transformer_enc_dec.vars_to_flax_vars(initial_vars))
      initial_states = transformer_enc_dec.init_states(inputs, input_paddings,
                                                       batch_size, seq_len)
      fprop_outputs = transformer_enc_dec.fprop(inputs, input_paddings, targets,
                                                jnp.zeros_like(targets))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(seq_len):
        targets_prefix = targets[:, t]
        if use_decoder_ngrams or use_decoder_vq_ngrams:
          if t > 0:
            targets_prefix = targets[:, t - 1:t + 1]
        cached_states, xent_output = transformer_enc_dec.extend_step(
            cached_states, targets_prefix)
        self.assertAllClose(logits[:, t, :], xent_output.logits, atol=2e-6)

  def test_glam_unitransformer(self):
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
    jax_p = transformer_models.TransformerLm.GLaMUniTransformerParams(
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
        use_tgt_labels_size_as_loss_denominator=True,
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
        use_tgt_labels_size_as_loss_denominator=True,
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

    # Compute jax outputs
    jax_outputs = test_utils.apply(
        jax_layer,
        jax_vars,
        jax_layer.fprop,
        jax_ids,
        jax_paddings,
        context_p=None,
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
        test_utils.to_np(jax_outputs.total_loss),
        test_utils.to_np(tf_out['loss'][0]))

  @parameterized.parameters([True, False])
  def test_glam_unitransformer_extendstep(self, moe):
    batch = 1
    length = 3
    d_model = 6
    num_heads = 2
    vocab_size = 16
    ff_dim = 8
    c_dim = 3
    e_dim = 4
    num_layers = 4
    # Build jax layer
    transformer_lm = transformer_models.TransformerLm.GLaMUniTransformerParams(
        name='model',
        vocab_size=vocab_size,
        num_transformer_layers=num_layers,
        moe=moe,
        model_dim=d_model,
        ff_dim=ff_dim,
        moe_hidden_dim=ff_dim,
        attention_num_heads=num_heads,
        attention_key_value_dim=d_model // num_heads,
        attention_extra_logit=0.0,
        use_tgt_labels_size_as_loss_denominator=True,
        moe_load_balance_loss_weight=0.01,
        num_groups=1,
        z_loss_weight=1e-4,
        c_dim=c_dim,
        e_dim=e_dim).Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    initial_vars = transformer_lm.instantiate_variables(prng_key)
    npy_inputs = np.random.randint(
        vocab_size, size=(batch, length)).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)
    with base_layer.JaxContext.new_context(
        params=context_params,
        prng_key=prng_key,
        global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
      jax_context.bind(transformer_lm,
                       transformer_lm.vars_to_flax_vars(initial_vars))
      initial_states = transformer_lm.init_states(batch, length)
      fprop_outputs = transformer_lm.fprop(inputs, jnp.zeros_like(inputs))
      logits = fprop_outputs.logits
      cached_states = initial_states
      for t in range(length):
        cached_states, xent_output = transformer_lm.extend_step(
            cached_states, inputs[:, t])
        self.assertAllClose(logits[:, t, :], xent_output.logits, atol=1e-5,
                            rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
