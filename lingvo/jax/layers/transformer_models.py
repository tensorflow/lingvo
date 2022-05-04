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
"""Transformer-related layers."""

from typing import Any, Optional, Tuple

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import attentions
from lingvo.jax.layers import embedding_softmax
from lingvo.jax.layers import normalizations
from lingvo.jax.layers import transformers

NestedMap = py_utils.NestedMap

InstantiableParams = py_utils.InstantiableParams
AuxLossContext = py_utils.AuxLossContext
JTensor = pytypes.JTensor


class TransformerLm(base_layer.BaseLayer):
  """Packed Transformer LM with position embedding and shared softmax layer.

  This folds the padding with the segment mask when the inputs are not packed.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Parameterization of this model."""
    p = super().Params()
    p.Define('position_emb_tpl', embedding_softmax.PositionalEmbedding.Params(),
             'The Positional Embedding layer params.')
    p.Define('model_dims', 0, 'Model dimension in Transformer layers.')
    p.Define('stacked_transformer_tpl',
             transformers.StackedTransformer.Params(),
             'StackedTransformer params tpl for the TransformerLm.')
    p.Define(
        'softmax_tpl',
        embedding_softmax.SingleShardSharedEmbeddingSoftmax.Params(),
        'The softmax layer params. By default the softmax layer is of type '
        'SingleSharedEmbeddingSoftmax so the softmax and embedding lookup '
        'share parameters in this case.')
    p.Define('vocab_size', 0, 'Size of the vocabulary for LM.')
    p.Define('packed_input', False, 'Whether the inputs are packed.')
    p.Define('masked_lm', False, 'Whether this is BERT style masked LM.')
    p.Define(
        'ngrammer_tpl', None,
        'Params for the Ngrammer layer. This param is shared between'
        'the Ngrammer layer as well as the VQNgrammer layer. If this is None'
        'then the Ngrammer layer is not used.')
    p.Define(
        'separate_embedding_tpl', None,
        'Optional separate embedding lookup layer params. By default this is '
        'None since the softmax and embedding lookup share parameters, however '
        'if we wish to separate the parameters of embedding lookup and softmax '
        'then we can set this param.')
    p.Define('final_ln_tpl', normalizations.LayerNorm.Params(),
             'Layer norm params.')
    return p

  @classmethod
  def GLaMUniTransformerParams(cls,
                               vocab_size,
                               model_dim,
                               ff_dim,
                               attention_num_heads,
                               attention_key_value_dim,
                               num_transformer_layers,
                               name='transformer',
                               moe=False,
                               moe_hidden_dim=None,
                               ffn_activation='GATED_GELU',
                               attention_extra_logit=0.0,
                               relative_attention_num_buckets=32,
                               relative_attention_max_distance=128,
                               num_groups=1,
                               c_dim=None,
                               capacity_factor=0.0,
                               e_dim=None,
                               use_tgt_labels_size_as_loss_denominator=True,
                               moe_load_balance_loss_weight=0.01,
                               z_loss_weight=1e-4,
                               combine_qkv=False):
    """Common setup for GLaM Decoder-only Transformer Model.

    This function sets up configs for both MoE and dense GLaM models.
    The MoE block consists of two transformer layer with the feedforward
    sublayer of the first one replaced by a MoE layer. The dense block consists
    of a transformer. The transformer layer used by GLam differs from the
    standard transformer in these configs:

    1) The feedforward sublayer used gated gleu so there are two wi and one wo.
    2) No bias in all projections and embeddings.
    3) Use no bias RMS norm for the layer norm.
    4) Use relative attention bias.
    5) Add an optional z-loss to stablize final softmax logits.

    Args:
      vocab_size: Size of the vocabulary for LM.
      model_dim: model dimension.
      ff_dim: hidden dimension of feed-forward inner layer.
      attention_num_heads: number of attention heads.
      attention_key_value_dim: key value dimension of attention inner layer.
      num_transformer_layers: Number of transformer layers.
      name: Name of the this layer
      moe: If this is a moe block or not.
      moe_hidden_dim: hidden dimension of MoE layer.
      ffn_activation: Activation function used in the ffn layer.
      attention_extra_logit: Extra logit for attention softmax.
      relative_attention_num_buckets: Relative attention num buckets
      relative_attention_max_distance: Max relative distance.
      num_groups: Total number of groups for token dispatching in MoE layer.
      c_dim: Expert capacity.
      capacity_factor: This is the ratio between max allowed examples per expert
        over the average number of examples per expert assuming routing is
        completely uniform.
      e_dim: Number of experts.
      use_tgt_labels_size_as_loss_denominator: False to use total number of
        non-padding tokens instead of fixed tgt_labels tensor size.
      moe_load_balance_loss_weight: Weight of the aux loss for MoE layers.
      z_loss_weight: additional loss term to stablize the final softmax logit.
      combine_qkv: if combined qkv projection layer is used.

    Returns:
      A Params object to set up a StackedTransformer.
    """
    p = cls.Params()
    p.name = name
    p.packed_input = True
    p.model_dims = model_dim
    p.vocab_size = vocab_size
    p.position_emb_tpl = None

    p.final_ln_tpl = normalizations.RmsNorm.Params().Set(
        name='rms_norm', input_dims=model_dim)

    p.softmax_tpl = (
        embedding_softmax.GShardSharedEmbeddingSoftmax.Params().Set(
            name='emb',
            input_dims=model_dim,
            num_classes=vocab_size,
            z_loss_weight=z_loss_weight))
    p.softmax_tpl.use_tgt_labels_size_as_loss_denominator = (
        use_tgt_labels_size_as_loss_denominator)
    glam_p = transformers.StackedTransformer.GLaMParams(
        model_dim=model_dim,
        ff_dim=ff_dim,
        attention_num_heads=attention_num_heads,
        attention_key_value_dim=attention_key_value_dim,
        name='decoder_block',
        moe=moe,
        moe_hidden_dim=moe_hidden_dim,
        ffn_activation=ffn_activation,
        mask_self_attention=True,
        cross_attention=False,
        attention_extra_logit=attention_extra_logit,
        relative_attention_num_buckets=relative_attention_num_buckets,
        relative_attention_max_distance=relative_attention_max_distance,
        moe_load_balance_loss_weight=moe_load_balance_loss_weight,
        num_groups=num_groups,
        c_dim=c_dim,
        capacity_factor=capacity_factor,
        e_dim=e_dim,
        combine_qkv=combine_qkv)

    p.stacked_transformer_tpl = transformers.StackedTransformerRepeated.Params()
    num_blocks = num_transformer_layers // 2 if moe else num_transformer_layers
    p.stacked_transformer_tpl.Set(
        name='decoder', block=glam_p, x_times=num_blocks)

    return p

  @classmethod
  def set_sharding_params_v1(cls,
                             lm_p,
                             *,
                             replica_axis,
                             data_axis,
                             mdl_axis,
                             device_ids_mesh,
                             mesh_axis_names,
                             mode='train'):
    """Set Canonical sharding params.

    Args:
      lm_p: A params of this class.
      replica_axis: A string or int of the model replica axis name.
      data_axis: A string or int of the data axis name.
      mdl_axis: A string or int of the mdl axis name.
      device_ids_mesh: A numpy array of device ids.
      mesh_axis_names: A list of length len(device_ids_mesh.shape). Each element
        of the list is the name of the corresponding device axis.
      mode: The mode this model will be used in. Can be either 'train' or
        'decode'.

    Returns:
      Params with sharding annotations added.
    """
    # In the following, model weights are layed out on the [data_axis, mdl_axis]
    # 2d mesh. Model weights are always replicated over the replica_axis mesh
    # axis.
    #
    # The batch axis of the activations are always sharded over the combination
    # of (replica_axis, data_axis).
    lm_p.device_mesh = device_ids_mesh
    lm_p.mesh_axis_names = mesh_axis_names
    # TODO(zhangqiaorjc): Remove once scan no longer needs explicit weight
    # sharding annotations.
    lm_p.stacked_transformer_tpl.device_mesh = device_ids_mesh
    lm_p.stacked_transformer_tpl.mesh_axis_names = mesh_axis_names

    # We assume activation batch is split on both replica_axis and data_axis.
    batch_split = (replica_axis, data_axis)

    if lm_p.position_emb_tpl is not None:
      pos_emb_p = lm_p.position_emb_tpl
      pos_emb_p.weight_split_dims_mapping.wt = [data_axis, mdl_axis]

    # NGrammer embedding table is currently replicated.
    # TODO(aurkor): Explore different sharding configs for the table.
    # n-gram table is of shape [ngram_vocab_size, embedding_dims].
    if lm_p.ngrammer_tpl is not None:
      ngrammer_p = lm_p.ngrammer_tpl
      ngrammer_p.weight_split_dims_mapping.wt = [mdl_axis, data_axis]

    softmax_p = lm_p.softmax_tpl
    if softmax_p.cls == embedding_softmax.GShardSharedEmbeddingSoftmax:
      # Softmax weight is of shape [vocab_size, input_dim].
      softmax_p.weight_split_dims_mapping.wt = [mdl_axis, data_axis]
    else:
      # Softmax weight is of shape [input_dim, vocab_size].
      softmax_p.weight_split_dims_mapping.wt = [data_axis, mdl_axis]
      softmax_p.lookup_style = 'matmul'
    if mode == 'train':
      # During training, softmax output is 3d.
      softmax_p.activation_split_dims_mapping.out = [
          batch_split, None, mdl_axis
      ]
    elif mode == 'decode':
      # During decoding, output from softmax is 2d.
      softmax_p.activation_split_dims_mapping.out = [batch_split, mdl_axis]
    else:
      raise NotImplementedError(f'mode {mode} not supported.')

    softmax_p.activation_split_dims_mapping.emb_out_split_dims_mapping = [
        batch_split, None, mdl_axis
    ]

    if lm_p.stacked_transformer_tpl.cls == transformers.PipelinedTransformer:
      stacked_transformer_tpl = lm_p.stacked_transformer_tpl.pipeline_stage
    else:
      stacked_transformer_tpl = lm_p.stacked_transformer_tpl

    if stacked_transformer_tpl.cls == transformers.StackedTransformer:
      xformer_p = stacked_transformer_tpl.transformer_layer_params_tpl
    elif stacked_transformer_tpl.cls == transformers.StackedTransformerRepeated:
      xformer_p = stacked_transformer_tpl.block.transformer_layer_params_tpl
    else:
      assert False, f'{stacked_transformer_tpl.cls} not supported.'

    xformer_p.tr_atten_tpl.activation_split_dims_mapping.blnh = [
        batch_split, None, mdl_axis, None
    ]
    xformer_p.tr_atten_tpl.activation_split_dims_mapping.bld = [
        batch_split, None, mdl_axis
    ]
    # Attention project weight matrix is of shape [data_dim, num_heads,
    # dim_per_head].
    xformer_p.tr_atten_tpl.weight_split_dims_mapping.proj = [
        data_axis, mdl_axis, None
    ]
    # Sharding for depth-wise conv weights. Depth-wise conv weights are of shape
    # [num_heads, dim_per_head].
    xformer_p.tr_atten_tpl.weight_split_dims_mapping.dconv = [mdl_axis, None]

    ffw_wp = xformer_p.tr_fflayer_tpl.weight_split_dims_mapping
    ffw_ap = xformer_p.tr_fflayer_tpl.activation_split_dims_mapping
    ffw_wp.ffn0 = [data_axis, mdl_axis]
    ffw_wp.ffn1 = [mdl_axis, data_axis]
    if mode == 'train':
      ffw_ap.ffn0 = [batch_split, None, mdl_axis]
      ffw_ap.ffn1 = [batch_split, None, mdl_axis]
    elif mode == 'decode':
      # For decoding, we need to change them to [data_axis, mdl_axis] to match
      # the shape of the input/output to/from the feedforward layers.
      ffw_ap.ffn0 = [batch_split, mdl_axis]
      ffw_ap.ffn1 = [batch_split, mdl_axis]
    else:
      raise NotImplementedError(f'mode {mode} not supported.')

    # MoE
    # Following GShard sharding settings for large 2D sharded models.
    #
    # TODO(lepikhin): Provide better reference.
    #   lingvo/core/gshard_builder.py and
    # specifically MoE splits
    #   emh_split=[0, -1, 1],
    #   ehm_split=[0, 1, -1],
    #   egcm_split=[0, -1, -1, 1],
    #   gecm_split=[0, -1, -1, 1],
    #   gsec_split=[0, -1, -1, -1],
    # for mesh with 2 dimensions.
    if stacked_transformer_tpl.cls == transformers.StackedTransformer:
      moe_p = stacked_transformer_tpl.moe_layer_tpl
    elif stacked_transformer_tpl.cls == transformers.StackedTransformerRepeated:
      moe_p = stacked_transformer_tpl.block.moe_layer_tpl
    else:
      assert False, f'{stacked_transformer_tpl.cls} not supported.'
    # Weights
    moe_wp = moe_p.weight_split_dims_mapping
    # TODO(lepikhin): RET_CHECK with [data_axis, None] http://b/209481545
    moe_wp.me = [None, None]  # replicated
    moe_wp.emh = [data_axis, None, mdl_axis]
    moe_wp.ehm = [data_axis, mdl_axis, None]
    # Activations
    moe_ap = moe_p.activation_split_dims_mapping
    moe_ap.gsm = [data_axis, None, mdl_axis]
    moe_ap.gs = [data_axis, None]
    moe_ap.gsec = [data_axis, None, None, None]  # dispatch and combine tensors
    moe_ap.egcm = [data_axis, None, None, mdl_axis]
    moe_ap.egch = [data_axis, None, None, mdl_axis]
    moe_ap.gecm = [data_axis, None, None, mdl_axis]

    return lm_p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructor."""
    super().__init__(params)
    p = self.params

    # Optional positional embedding layer.
    if p.position_emb_tpl is not None:
      pos_params = p.position_emb_tpl.Copy()
      pos_params.embedding_dims = p.model_dims
      self.create_child('position_emb', pos_params)

    # Optional separate embedding layer.
    if p.separate_embedding_tpl is not None:
      emb_params = p.separate_embedding_tpl.Copy()
      emb_params.embedding_dims = p.model_dims
      emb_params.vocab_size = p.vocab_size
      self.create_child('embedding_lookup', emb_params)

    # Ngrammer layer.
    if p.ngrammer_tpl is not None:
      self.create_child('ngrammer', p.ngrammer_tpl)

    # Transformer layers
    xformer_params = p.stacked_transformer_tpl
    if xformer_params.cls == transformers.PipelinedTransformer:
      xformer_params = xformer_params.pipeline_stage
    if xformer_params.cls == transformers.StackedTransformerRepeated:
      xformer_params = xformer_params.block
    if xformer_params.cls != transformers.StackedTransformer:
      assert False, f'{xformer_params.cls} not supported.'
    assert (xformer_params.model_dims == 0 or
            xformer_params.model_dims == p.model_dims)
    xformer_params.model_dims = p.model_dims
    if p.masked_lm:
      xformer_params.mask_self_attention = False
    else:
      xformer_params.mask_self_attention = True
    xformer_params.packed_input = p.packed_input
    xformer_params.fold_padding_with_segment_mask = True

    self.create_child('transformer', p.stacked_transformer_tpl)

    # Final layer norm
    if p.final_ln_tpl is not None:
      ln_params = p.final_ln_tpl.Set(input_dims=p.model_dims)
      self.create_child('final_ln', ln_params)

    # Final softmax
    softmax_params = p.softmax_tpl.Copy()
    softmax_params.input_dims = p.model_dims
    softmax_params.num_classes = p.vocab_size
    self.create_child('softmax', softmax_params)

  def init_states(self, *args: Any, **kwargs: Any) -> NestedMap:
    """Initialize the cache for the autoregressive decoding.

    Args:
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      A `.NestedMap` corresponding to the cache.
    """
    return NestedMap(
        step=jnp.array(0, dtype=jnp.uint32),
        transformer=self.transformer.init_states(*args, **kwargs))

  def compute_loss(self,
                   activations: JTensor,
                   labels: Optional[NestedMap] = None) -> NestedMap:
    """Computes cross entropy loss.

    Args:
      activations: Output of last layer of shape [B, T, D].
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [B, T] containing weights for each target word.
        class_ids, a JTensor with shape [B, T] of int32 dtype containing the
        target class labels. class_probabilities, a JTensor with shape [B, T, V]
        of float values indicating class-membership probabilities.

    Returns:
      Returns xent_output, where `xent_output` is a `.NestedMap` as defined by
      `SoftmaxLayer`'s return. In addition, per_sequence_xent is added which
      equal to the sum of xent loss for tokens in a sequence.
    """
    if labels is None:
      logits = self.softmax.get_logits(inputs=activations)
      xent_output = NestedMap(logits=logits)
      xent_output.log_probs = jax.nn.log_softmax(logits)
      xent_output.probs = jax.nn.softmax(xent_output.logits)
    else:
      class_ids = None
      class_probabilities = None
      if 'class_ids' in labels:
        class_ids = labels.class_ids[:, :, jnp.newaxis]
      if 'class_probabilities' in labels:
        class_probabilities = labels.class_probabilities
      class_weights = labels.class_weights[:, :, jnp.newaxis]
      xent_output = self.softmax.fprop(
          activations,
          class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)
      per_token_xent = xent_output.per_example_xent * labels.class_weights
      xent_output.per_token_xent = per_token_xent
      xent_output.per_sequence_xent = jnp.sum(per_token_xent, -1)

      # Compute aux_loss and add to avg_xent.
      if AuxLossContext.Current() and AuxLossContext.Current().aux_losses:
        aux_loss_tensors = AuxLossContext.Current().aux_losses
        assert isinstance(aux_loss_tensors, list)
        aux_loss = sum(aux_loss_tensors)
      else:
        aux_loss = 0.0
      if not isinstance(aux_loss, jnp.ndarray):
        aux_loss = jnp.array(aux_loss, dtype=self.fprop_dtype)
      xent_output.aux_loss = aux_loss
      # This is the loss to minimize.
      xent_output.total_loss = xent_output.avg_xent + xent_output.aux_loss
    return xent_output

  def fprop(self,
            inputs: JTensor,
            paddings: JTensor,
            labels: Optional[NestedMap] = None,
            segment_ids: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> NestedMap:
    """Computes xent loss given the language model inputs.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, T].
      paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding.
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [batch, seqlen] containing weights for each target
        word. class_ids, a JTensor with shape [B, T] of int32 dtype containing
        the target class labels. class_probabilities, a JTensor with shape [B,
        T, V] of float values indicating class-membership probabilities.
      segment_ids: A JTensor of shape [B, T]. The segment that each token
        belongs to.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.

    Returns:
      Returns xent_output, where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return. In
      addition, per_sequence_xent is added which equal to the sum of xent loss
      for tokens in a sequence.
    """
    p = self.params
    # reentrant=True, to enable scan-local context override.
    with py_utils.AuxLossContext(reentrant=True) as aux_loss_ctx:
      assert aux_loss_ctx is not None
      # Get the input embeddings.
      if self.params.separate_embedding_tpl is not None:
        input_emb = self.embedding_lookup.fprop(inputs)
      else:
        input_emb = self.softmax.emb_lookup(inputs)
      batch, seq_length = inputs.shape

      if segment_ids is None:
        assert segment_pos is None
        # Fold the paddings with the segment mask
        segment_ids = jnp.asarray(1 - paddings, jnp.int32)
        segment_pos = jnp.tile(
            jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1])

      # Add NGrammer to the source embeddings.
      if p.ngrammer_tpl is not None:
        input_emb = self.ngrammer.fprop(
            input_ids=inputs,
            input_embs=input_emb,
            paddings=paddings,
            segment_pos=segment_pos)

      if p.position_emb_tpl is not None:
        position_emb = self.position_emb.fprop(
            seq_length=seq_length, position=segment_pos)
        inputs = input_emb + position_emb
      else:
        inputs = input_emb

      if p.masked_lm:
        segment_mask = attentions.segment_mask(segment_ids, segment_ids,
                                               inputs.dtype)
      else:
        segment_mask = attentions.causal_segment_mask(segment_ids, inputs.dtype)

      output = self.transformer.fprop(
          inputs, paddings, segment_mask=segment_mask, segment_pos=segment_pos)

      # Final layer norm
      if p.final_ln_tpl is not None:
        output = self.final_ln.fprop(output)
      return self.compute_loss(output, labels)

  def extend_step(
      self,
      cached_states: NestedMap,
      inputs: JTensor,
  ) -> Tuple[NestedMap, NestedMap]:
    """Autoregressive cached decoding of Transformer LM.

    Args:
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for cached decoding.
        cached_states.transformer.x_layers is a list corresponding to
        self.transformer.x_layers with key - [T, B, N, H]. value - [T, B, N, H].
        cached_states.step corresponds to the current time step being decoded.
      inputs: Target sequence of shape [B] or [B, P] corresponding to target
        sequence at index time_step. Note that the shape [B, P] corresponds to a
        prefix which is useful for decoding in some special architectures such
        as Primer or Ngrammer.

    Returns:
      cached_states: A `.NestedMap` object containing the updated states. The
        cached_states.step is incremented to the next time step, and
        cached_states.transformer is updated with the keys and values of the
        current time step.
      xent_output: A `.NestedMap` object containing the log probabilities and
        probabilities.
    """
    p = self.params
    # Extend step should only be called with causal LM.
    assert not p.masked_lm

    if len(inputs.shape) == 1:
      inputs = inputs[:, jnp.newaxis]

    # Get the input embeddings.
    if self.params.separate_embedding_tpl is not None:
      input_emb = self.embedding_lookup.fprop(inputs)
    else:
      input_emb = self.softmax.emb_lookup(inputs)
    time_step = cached_states.step

    # Add Ngrammer layer if applicable.
    if p.ngrammer_tpl is not None:
      input_emb = self.ngrammer.fprop(
          inputs, input_emb, paddings=None, segment_pos=None)
      inputs = inputs[:, -1][:, jnp.newaxis]
      input_emb = input_emb[:, -1, :][:, jnp.newaxis, :]

    if p.position_emb_tpl is not None:
      # During autoregressive decoding inputs are not packed.
      segment_pos = jnp.zeros((inputs.shape[0], 1)) + time_step
      position_emb = self.position_emb.fprop(seq_length=1, position=segment_pos)

      inputs = input_emb + position_emb
    else:
      inputs = input_emb

    updated_cache, outputs = self.transformer.extend_step(
        cached_states.transformer, inputs[:, 0, :], time_step=time_step)
    cached_states.transformer = updated_cache
    cached_states.step += 1
    if p.final_ln_tpl is not None:
      outputs = self.final_ln.fprop(outputs)
    xent_output = self.compute_loss(outputs)
    return cached_states, xent_output


class TransformerEncoderDecoder(base_layer.BaseLayer):
  """Transformer encoder/decoder class.

  This uses the param `encoder_stacked_transformer_tpl` to set the configuration
  for the encoder stack, and the param `decoder_stacked_transformer_tpl` to set
  the configuration for the decoder stack.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Parameterization of the Transformer encoder-decoder model."""
    p = super().Params()
    p.Define('position_emb_tpl', embedding_softmax.PositionalEmbedding.Params(),
             'The Positional Embedding layer params for encoder and decoder.')
    p.Define(
        'encoder_stacked_transformer_tpl', None,
        'StackedTransformer params tpl for the encoder. This must be set with '
        'a value that is not None at initialization time.')
    p.Define(
        'encoder_ngrammer_tpl', None,
        'Optional params for the Ngrammer layer for the encoder. This param is '
        'shared between the Ngrammer layer as well as the VQNgrammer layer. If '
        'this is None then the Ngrammer layer is not used.')
    p.Define(
        'encoder_embedding_tpl', None,
        'Optional separate embedding layer for the source ids. By default this '
        'is set to None, so the inputs and targets share the same set of '
        'embeddings.')
    p.Define(
        'decoder_stacked_transformer_tpl', None,
        'StackedTransformer params tpl for the decoder. This must be set with '
        'a value that is not None at initialization time.')
    p.Define(
        'decoder_ngrammer_tpl', None,
        'Optional params for the Ngrammer layer for the decoder. This param is '
        'shared between the Ngrammer layer as well as the VQNgrammer layer. If '
        'this is None then the Ngrammer layer is not used.')
    p.Define(
        'decoder_embedding_tpl', None,
        'Optional separate embedding layer for the target ids. By default this '
        'is set to None, so the embedding parameters are shared with the '
        'softmax layer.')
    p.Define(
        'model_dims', 0, 'Model dimension of the Transformer layers. This '
        'must match the model dimension of the encoder stack and the '
        'decoder stack, as well as the embedding and softmax dimensions.')
    p.Define(
        'softmax_tpl',
        embedding_softmax.SingleShardSharedEmbeddingSoftmax.Params(),
        'The softmax layer params. By default the softmax layer is of type '
        'SingleSharedEmbeddingSoftmax so the softmax and embedding lookup '
        'share parameters in this case.')
    p.Define('packed_input', False, 'Whether the inputs are packed.')
    return p

  def __init__(self, params):
    # This will create a decoder (LM) with key transformer.
    super().__init__(params)
    p = self.params

    def set_model_dims_and_packing(stacked_transformer_tpl, model_dims,
                                   packed_input):
      if stacked_transformer_tpl.cls == transformers.StackedTransformer:
        assert (stacked_transformer_tpl.model_dims == 0 or
                stacked_transformer_tpl.model_dims == model_dims)
        stacked_transformer_tpl.model_dims = model_dims
        stacked_transformer_tpl.packed_input = packed_input
      elif stacked_transformer_tpl.cls == transformers.StackedTransformerRepeated:
        assert (stacked_transformer_tpl.block.model_dims == 0 or
                stacked_transformer_tpl.block.model_dims == model_dims)
        stacked_transformer_tpl.block.model_dims = model_dims
        stacked_transformer_tpl.block.packed_input = packed_input
      else:
        assert False, f'{stacked_transformer_tpl.cls} not supported.'

    # Create position embeddings.
    if p.position_emb_tpl is not None:
      assert (p.position_emb_tpl.embedding_dims == 0 or
              p.position_emb_tpl.embedding_dims == p.model_dims)
      p.position_emb_tpl.embedding_dims = p.model_dims
      self.create_child('position_emb', p.position_emb_tpl)

    # Create the encoder.
    if p.encoder_stacked_transformer_tpl is None:
      raise ValueError(
          'Encoder stack must be specified for TransformerEncoderDecoder.')

    # Use the user specified StackedTransformer for the encoder, assuming
    # everything is set up appropriately.
    encoder_params = p.encoder_stacked_transformer_tpl.Copy()
    set_model_dims_and_packing(encoder_params, p.model_dims, p.packed_input)
    # Assert that encoder is not masked.
    if encoder_params.cls == transformers.StackedTransformer:
      mask_self_attention = encoder_params.mask_self_attention
    elif encoder_params.cls == transformers.StackedTransformerRepeated:
      mask_self_attention = encoder_params.block.mask_self_attention
    else:
      raise ValueError('Unknown encoder stack.')

    if mask_self_attention:
      raise ValueError(
          'Encoder attention should be un-masked in TransformerEncoderDecoder.')
    self.create_child('encoder', encoder_params)

    # Optional separate embedding layer for source ids.
    if p.encoder_embedding_tpl is not None:
      encoder_embedding_params = p.encoder_embedding_tpl.Copy()
      assert (encoder_embedding_params.embedding_dims == 0 or
              encoder_embedding_params.embedding_dims == p.model_dims)
      encoder_embedding_params.embedding_dims = p.model_dims
      self.create_child('encoder_embedding_lookup', encoder_embedding_params)

    # Optional NGrammer layer for the encoder.
    # Paper: https://openreview.net/forum?id=GxjCYmQAody
    if p.encoder_ngrammer_tpl is not None:
      self.create_child('encoder_ngrammer', p.encoder_ngrammer_tpl)

    # Encoder output layer norm.
    encoder_ln_params = normalizations.LayerNorm.Params().Set(
        input_dims=p.model_dims)
    self.create_child('encoder_ln', encoder_ln_params)

    # Create the decoder.
    if p.decoder_stacked_transformer_tpl is None:
      raise ValueError(
          'Decoder stack must be specified for TransformerEncoderDecoder.')

    # Use the user specified StackedTransformer for the decoder, assuming
    # everything is set up appropriately.
    decoder_params = p.decoder_stacked_transformer_tpl
    set_model_dims_and_packing(decoder_params, p.model_dims, p.packed_input)
    # Assert that decoder is masked.
    # Assert that encoder is not masked.
    if decoder_params.cls == transformers.StackedTransformer:
      mask_self_attention = decoder_params.mask_self_attention
    elif decoder_params.cls == transformers.StackedTransformerRepeated:
      mask_self_attention = decoder_params.block.mask_self_attention
    else:
      raise ValueError('Unknown decoder stack.')

    if not mask_self_attention:
      raise ValueError(
          'Decoder attention should be masked in TransformerEncoderDecoder.')
    self.create_child('decoder', decoder_params)

    # Optional separate embedding layer for target ids.
    if p.decoder_embedding_tpl is not None:
      decoder_embedding_params = p.decoder_embedding_tpl.Copy()
      assert (decoder_embedding_params.embedding_dims == 0 or
              decoder_embedding_params.embedding_dims == p.model_dims)
      decoder_embedding_params.embedding_dims = p.model_dims
      self.create_child('decoder_embedding_lookup', decoder_embedding_params)

    # Optional NGrammer layer for the decoder.
    # Paper: https://openreview.net/forum?id=GxjCYmQAody
    if p.decoder_ngrammer_tpl is not None:
      self.create_child('decoder_ngrammer', p.decoder_ngrammer_tpl)

    # Decoder output layer norm.
    decoder_ln_params = normalizations.LayerNorm.Params().Set(
        input_dims=p.model_dims)
    self.create_child('decoder_ln', decoder_ln_params)

    # Final softmax.
    softmax_params = p.softmax_tpl.Copy()
    assert (softmax_params.input_dims == 0 or
            softmax_params.input_dims == p.model_dims)
    softmax_params.input_dims = p.model_dims
    self.create_child('softmax', softmax_params)

  def _encode(self,
              inputs: JTensor,
              input_paddings: JTensor,
              input_segment_ids: Optional[JTensor] = None,
              input_segment_pos: Optional[JTensor] = None) -> JTensor:
    """Apply the Transformer encoder to the source sequence.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, S].
      input_paddings: A 0/1 JTensor of shape [B, S] with 1 denoting padding
        correspdonding to the input sequence.
      input_segment_ids: A JTensor of shape [B,S]. The segment that each input
        token belongs to.
      input_segment_pos: A JTensor of shape [B, S]. The position of each input
        token within a segment.

    Returns:
      The encoded sequence after applying the Transformer encoder.
    """
    p = self.params
    batch, seq_length = inputs.shape
    if p.encoder_embedding_tpl is not None:
      # Encoder has its own embedding lookup table for source ids.
      input_emb = self.encoder_embedding_lookup.fprop(inputs)
    elif p.decoder_embedding_tpl is not None:
      # Encoder shares the same embedding as the target ids.
      # The embedding lookup for target ids is separate from the softmax.
      input_emb = self.decoder_embedding_lookup.fprop(inputs)
    else:
      # Encoder and decoder share the softmax and embedding params.
      input_emb = self.softmax.emb_lookup(inputs)

    if input_segment_ids is None:
      assert input_segment_pos is None
      # Fold the paddings with the segment mask.
      input_segment_ids = jnp.asarray(1 - input_paddings, jnp.int32)
      input_segment_pos = jnp.tile(
          jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1])
    assert input_segment_ids is not None
    assert input_segment_pos is not None

    # Add NGrammer to the source embeddings.
    if p.encoder_ngrammer_tpl is not None:
      input_emb = self.encoder_ngrammer.fprop(
          input_ids=inputs,
          input_embs=input_emb,
          paddings=input_paddings,
          segment_pos=input_segment_pos)

    if p.position_emb_tpl is not None:
      position_emb = self.position_emb.fprop(
          seq_length=seq_length, position=input_segment_pos)
      input_emb += position_emb

    inputs_segment_mask = attentions.segment_mask(
        input_segment_ids, dtype=input_emb.dtype)
    encoder_output = self.encoder.fprop(
        input_emb, input_paddings, segment_mask=inputs_segment_mask)

    # Final layer norm for encoder output.
    encoder_output = self.encoder_ln.fprop(encoder_output)
    return encoder_output

  def compute_loss(self,
                   activations: JTensor,
                   labels: Optional[NestedMap] = None) -> NestedMap:
    """Computes cross entropy loss.

    Args:
      activations: Output of last layer of shape [B, T, D].
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [B, T] containing weights for each target word.
        class_ids, a JTensor with shape [B, T] of int32 dtype containing the
        target class labels. class_probabilities, a JTensor with shape [B, T, V]
        of float values indicating class-membership probabilities.

    Returns:
      Returns xent_output, where `xent_output` is a `.NestedMap` as defined by
      `SoftmaxLayer`'s return. In addition, per_sequence_xent is added which
      equal to the sum of xent loss for tokens in a sequence.
    """
    if labels is None:
      logits = self.softmax.get_logits(inputs=activations)
      xent_output = NestedMap(logits=logits)
      xent_output.log_probs = jax.nn.log_softmax(logits)
      xent_output.probs = jax.nn.softmax(xent_output.logits)
    else:
      class_ids = None
      class_probabilities = None
      if 'class_ids' in labels:
        class_ids = labels.class_ids[:, :, jnp.newaxis]
      if 'class_probabilities' in labels:
        class_probabilities = labels.class_probabilities
      class_weights = labels.class_weights[:, :, jnp.newaxis]
      xent_output = self.softmax.fprop(
          activations,
          class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)
      per_token_xent = xent_output.per_example_xent * labels.class_weights
      xent_output.per_token_xent = per_token_xent
      xent_output.per_sequence_xent = jnp.sum(per_token_xent, -1)

      # Compute aux_loss and add to avg_xent.
      if AuxLossContext.Current() and AuxLossContext.Current().aux_losses:
        aux_loss_tensors = AuxLossContext.Current().aux_losses
        assert isinstance(aux_loss_tensors, list)
        aux_loss = sum(aux_loss_tensors)
      else:
        aux_loss = 0.0
      if not isinstance(aux_loss, jnp.ndarray):
        aux_loss = jnp.array(aux_loss, dtype=self.fprop_dtype)
      xent_output.aux_loss = aux_loss
      # This is the loss to minimize.
      xent_output.total_loss = xent_output.avg_xent + xent_output.aux_loss
    return xent_output

  def fprop(
      self,
      inputs: JTensor,
      input_paddings: JTensor,
      targets: JTensor,
      target_paddings: JTensor,
      labels: Optional[NestedMap] = None,
      input_segment_ids: Optional[JTensor] = None,
      input_segment_pos: Optional[JTensor] = None,
      target_segment_ids: Optional[JTensor] = None,
      target_segment_pos: Optional[JTensor] = None,
  ) -> NestedMap:
    """Computes xent loss given the sequence model inputs.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, S].
      input_paddings: A 0/1 JTensor of shape [B, S] with 1 denoting padding
        correspdonding to the input sequence.
      targets: Target ids. An int32 JTensor of shape [B, T].
      target_paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding
        corresponding to the target sequence.
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [batch, seqlen] containing weights for each target
        word. class_ids, a JTensor with shape [B, T] of int32 dtype containing
        the target class labels. class_probabilities, a JTensor with shape [B,
        T, V] of float values indicating class-membership probabilities.
      input_segment_ids: A JTensor of shape [B,S]. The segment that each input
        token belongs to.
      input_segment_pos: A JTensor of shape [B, S]. The position of each input
        token within a segment.
      target_segment_ids: A JTensor of shape [B,T]. The segment that each target
        token belongs to.
      target_segment_pos: A JTensor of shape [B, T]. The position of each target
        token within a segment.

    Returns:
      Returns xent_output, where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return. In
      addition, per_sequence_xent is added which equal to the sum of xent loss
      for tokens in a sequence.
    """
    # Get the input embeddings.
    p = self.params
    batch, seq_length = inputs.shape
    _, target_seq_length = targets.shape

    encoder_output = self._encode(inputs, input_paddings, input_segment_ids,
                                  input_segment_pos)

    if p.decoder_embedding_tpl is not None:
      # Targets have separate embedding params.
      target_emb = self.decoder_embedding_lookup.fprop(targets)
    else:
      # Embedding parameters are shared with targets and softmax.
      target_emb = self.softmax.emb_lookup(targets)

    if p.decoder_ngrammer_tpl is not None:
      target_emb = self.decoder_ngrammer.fprop(
          input_ids=targets,
          input_embs=target_emb,
          paddings=target_paddings,
          segment_pos=target_segment_pos)

    if p.position_emb_tpl is not None:
      targets_position_emb = self.position_emb.fprop(
          seq_length=target_seq_length, position=target_segment_pos)
      target_emb += targets_position_emb

    if input_segment_ids is None:
      assert input_segment_pos is None
      # Fold the paddings with the segment mask.
      input_segment_ids = jnp.asarray(1 - input_paddings, jnp.int32)
      input_segment_pos = jnp.tile(
          jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1])

    if target_segment_ids is None:
      assert target_segment_pos is None
      # Fold the paddings with the segment mask.
      target_segment_ids = jnp.asarray(1 - target_paddings, jnp.int32)
      target_segment_pos = jnp.tile(
          jnp.arange(target_seq_length, dtype=jnp.int32)[None, :], [batch, 1])

    # Cross attention.
    cross_segment_mask = attentions.segment_mask(target_segment_ids,
                                                 input_segment_ids,
                                                 target_emb.dtype)
    target_segment_mask = attentions.causal_segment_mask(
        target_segment_ids, target_emb.dtype)
    output = self.decoder.fprop(
        target_emb,
        target_paddings,
        target_segment_mask,
        cross_inputs=encoder_output,
        cross_paddings=input_paddings,
        cross_segment_mask=cross_segment_mask)

    # Final layer norm for decoder.
    output = self.decoder_ln.fprop(output)

    return self.compute_loss(output, labels)

  def init_states(self, inputs: JTensor, input_paddings: JTensor, *args: Any,
                  **kwargs: Any) -> NestedMap:
    """Initialize the cache for autoregressive decoding.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, S].
      input_paddings: A 0/1 JTensor of shape [B, S] with 1 denoting padding
        correspdonding to the input sequence.
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      A `.NestedMap` corresponding to the cache.
    """
    cache = NestedMap(
        step=jnp.array(0, dtype=jnp.uint32),
        decoder=self.decoder.init_states(*args, **kwargs))
    encoder_output = self._encode(
        inputs, input_paddings, input_segment_ids=None, input_segment_pos=None)
    cache.encoder_output = encoder_output
    cache.input_paddings = input_paddings
    return cache

  def extend_step(self, cached_states: NestedMap,
                  targets: JTensor) -> Tuple[NestedMap, NestedMap]:
    """Autoregressive cached decoding of the Transformer encoder decoder.

    Args:
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for cached decoding.
        cached_states.transformer.x_layers is a list corresponding to
        self.transformer.x_layers with key - [T, B, N, H]. value - [T, B, N, H].
        cached_states.step corresponds to the current time step being decoded.
        In addition, for the TransformerEncoderDecoder class, we also cache the
        output of the encoder since that need not be computed afresh for every
        time step.
      targets: Target sequence of shape [B] or [B, P] corresponding to target
        sequence at index time_step. Note that the shape [B, P] corresponds to a
        prefix which is useful for decoding in some special architectures such
        as Primer or Ngrammer.

    Returns:
      cached_states: A `.NestedMap` object containing the updated states. The
        cached_states.step is incremented to the next time step, and
        cached_states.transformer is updated with the keys and values of the
        current time step.
      xent_output: A `.NestedMap` object containing the log probabilities and
        probabilities.
    """
    p = self.params
    # Fetch encoder output from the cache.
    encoder_output = cached_states.encoder_output
    input_paddings = cached_states.input_paddings

    # During autoregressive decoding inputs and targets are not packed.
    if len(targets.shape) == 1:
      targets = targets[:, jnp.newaxis]

    if p.decoder_embedding_tpl is not None:
      # Targets have separate embedding params.
      target_emb = self.decoder_embedding_lookup.fprop(targets)
    else:
      # Embedding parameters are shared with targets and softmax.
      target_emb = self.softmax.emb_lookup(targets)

    time_step = cached_states.step
    if p.decoder_ngrammer_tpl is not None:
      target_emb = self.decoder_ngrammer.fprop(
          targets, target_emb, paddings=None, segment_pos=None)

    targets = targets[:, -1][:, jnp.newaxis]
    target_emb = target_emb[:, -1, :][:, jnp.newaxis, :]

    # Add position embeddings to target ids.
    if p.position_emb_tpl is not None:
      # During autoregressive decoding inputs are not packed.
      segment_pos = jnp.zeros((targets.shape[0], 1)) + time_step
      target_position_emb = self.position_emb.fprop(
          seq_length=1, position=segment_pos)
      target_emb += target_position_emb

    updated_cache, outputs = self.decoder.extend_step(
        cached_states.decoder,
        target_emb[:, 0, :],
        time_step=time_step,
        cross_inputs=encoder_output,
        cross_paddings=input_paddings)
    cached_states.decoder = updated_cache
    cached_states.step += 1
    outputs = self.decoder_ln.fprop(outputs)
    xent_output = self.compute_loss(outputs)
    return cached_states, xent_output
