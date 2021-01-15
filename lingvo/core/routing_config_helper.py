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
"""Stacked Transformer setup template for Routing Transformer.

See https://arxiv.org/abs/2003.05997 for the paper.
"""
from lingvo.core import batch_major_attention


def SetupRoutingTransformerEncoder(model_dim,
                                   num_layers,
                                   num_heads,
                                   left_context,
                                   right_context,
                                   block_size,
                                   num_routing_layers,
                                   num_routing_heads,
                                   num_clusters,
                                   attention_window,
                                   atten_dropout_prob=0.,
                                   relu_dropout_prob=0.,
                                   residual_dropout_prob=0.):
  """A lightweight wrapper for Routing Transformer encoder stack.

  Args:
   model_dim: specifies dimension of transformer layers, token embeddings, and
     positional embeddings as well context vectors (attention values).
   num_layers: number of transformer layers.
   num_heads: number of attention heads.
   left_context: amount of left context in local attention.
   right_context: amount of right context in local attention.
   block_size: size of block in local attention.
   num_routing_layers: number of routing attention layers.
   num_routing_heads: number of routing attention heads.
   num_clusters: number of clusters for routing attention.
   attention_window: attention window for routing attention.
   atten_dropout_prob: used in attention layer.
   relu_dropout_prob: used in transformer feedforward layer.
   residual_dropout_prob: used in transformer feedforward and attention layer.

  Returns:
    Routing Transformer encoder stack.
  """
  transformer_stack = batch_major_attention.StackedTransformerLayers.Params()
  transformer_stack.Set(
      name='routing_transformer_stack',
      mdl_dim=model_dim,
      num_layers=num_layers,
      hidden_dim=model_dim,
      num_atten_heads=num_heads,
      dropout_prob=relu_dropout_prob,
      add_unnormalized_input=False,
      use_fused_layernorm=False,
      fprop_dtype=None)

  local_params = batch_major_attention.TransformerLayer.Params()
  local_params.tr_atten_tpl.Set(
      atten_dropout_prob=relu_dropout_prob,
      residual_dropout_prob=residual_dropout_prob,
      num_heads=num_heads)
  local_params.tr_fflayer_tpl.Set(
      relu_dropout_prob=relu_dropout_prob, activation='RELU')
  local_params.tr_atten_tpl.ln_tpl.Set(use_fused_layernorm=False)
  local_params.tr_fflayer_tpl.ln_tpl.Set(use_fused_layernorm=False)
  local_params.tr_fflayer_tpl.fflayer_tpl.projection.Set(use_einsum=False)
  local_params.tr_atten_tpl.atten_tpl = (
      batch_major_attention.LocalSelfAttention.Params().Set(
          left_context=left_context,
          right_context=right_context,
          block_size=block_size,
          enable_per_dim_scale=True,
          force_consistent_probs_shape=True,
          use_bias=True))
  routing_params = batch_major_attention.TransformerLayer.Params()
  routing_params.tr_atten_tpl.Set(
      atten_dropout_prob=atten_dropout_prob,
      residual_dropout_prob=residual_dropout_prob,
      num_heads=[num_heads - num_routing_heads, num_routing_heads])
  routing_params.tr_fflayer_tpl.Set(
      relu_dropout_prob=relu_dropout_prob, activation='RELU')
  routing_params.tr_atten_tpl.ln_tpl.Set(use_fused_layernorm=False)
  routing_params.tr_fflayer_tpl.ln_tpl.Set(use_fused_layernorm=False)
  routing_params.tr_fflayer_tpl.fflayer_tpl.projection.Set(use_einsum=False)
  routing_params.tr_atten_tpl.atten_tpl = [
      batch_major_attention.LocalSelfAttention.Params().Set(
          left_context=left_context,
          right_context=right_context,
          block_size=block_size,
          enable_per_dim_scale=True,
          force_consistent_probs_shape=True,
          use_bias=True),
      batch_major_attention.RoutingAttention.Params().Set(
          dim_per_head=model_dim // num_heads,
          num_clusters=num_clusters,
          attention_window=attention_window,
          query_group_size_factor=1.0,
          causal_masking=False,
          fast_path=True,
          enable_per_dim_scale=True,
          use_bias=True)
  ]
  params_list = ([local_params] * (num_layers - num_routing_layers) +
                 [routing_params] * num_routing_layers)
  transformer_stack.transformer_layer_params_tpl = params_list
  return transformer_stack
