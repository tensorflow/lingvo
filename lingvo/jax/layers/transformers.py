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
"""Transformer-related layers."""

from typing import Any, Optional, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from lingvo.jax import base_layer
from lingvo.jax import gshard_utils
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import activations as activations_lib
from lingvo.jax.layers import attentions
from lingvo.jax.layers import embedding_softmax
from lingvo.jax.layers import linears
from lingvo.jax.layers import normalizations
from lingvo.jax.layers import recurrent
from lingvo.jax.layers import repeats
from lingvo.jax.layers import stats
from lingvo.jax.layers import stochastics
import numpy as np
import tensorflow.compat.v2 as tf

CreateLayerVariableStatus = base_layer.CreateLayerVariableStatus

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
weight_params = py_utils.weight_params

InstantiableParams = py_utils.InstantiableParams
AuxLossContext = py_utils.AuxLossContext
JTensor = pytypes.JTensor


def compute_attention_masks_for_fprop(
    inputs: JTensor,
    paddings: Optional[JTensor] = None,
    causal_attention: Optional[bool] = False,
    segment_mask: Optional[JTensor] = None,
    cross_inputs: Optional[JTensor] = None,
    cross_paddings: Optional[JTensor] = None,
    cross_segment_mask: Optional[JTensor] = None,
    fold_padding_with_segment_mask: Optional[bool] = False,
) -> Tuple[JTensor, Union[JTensor, None]]:
  """Computes attention mask from paddings, segment masks etc for fprop.

  Args:
    inputs: Input sequence JTensor of shape [B, T, H].
    paddings: Input paddings JTensor of shape [B, T] (optional). Note that one
      of paddings or segment_mask must be provided.
    causal_attention: Boolean to apply causal masking (optional).
    segment_mask: Segment mask JTensor for packed input of shape [B, 1, T, T]
      ready to add to logits (optional).
    cross_inputs: Output JTensor of the encoder, to be used for cross attention,
      of shape [B, S, H].
    cross_paddings: Paddings JTensor for cross atention of shape [B, S].
    cross_segment_mask: Segment mask JTensor for encoder-decoder in packed input
      case of shape [B, 1, T, S].
    fold_padding_with_segment_mask: If True then segment mask is supposed to
      include the padding mask as well, i.e. treating PADs as one sequence and
      non-PADs as another.

  Returns:
    attention_mask: Attention mask JTensor ready to add to logits for self
      attention of shape [B/1, 1, T/1, T].
    cross_attention_mask: Attention mask ready to add to logits for cross
      attention of shape [B/1, 1, T/1, S]. This will be None if cross_inputs
      are None.
  """
  if fold_padding_with_segment_mask:
    # In this case a separate padding mask is not needed, it is assumed
    # folded with segment mask.
    assert segment_mask is not None
    attention_mask = segment_mask
  else:
    # Paddings must be provided to create the attention mask
    assert paddings is not None
    # Get paddings mask to [B, 1, 1, T]
    attention_mask = attentions.convert_paddings_to_mask(paddings, inputs.dtype)

    # Additional segment_mask may also be provided in this case
    if segment_mask is not None:
      attention_mask = jnp.minimum(attention_mask, segment_mask)

  # Causal mask of shape [1, 1, T, T]
  if causal_attention:
    causal_mask = attentions.causal_mask(inputs)
    attention_mask = jnp.minimum(attention_mask, causal_mask)

  # Compute cross attention mask if applicable
  cross_attention_mask = None
  if cross_inputs is not None:
    assert cross_paddings is not None

    # Compute paddings
    cross_attention_mask = attentions.convert_paddings_to_mask(
        cross_paddings, dtype=cross_inputs.dtype)

    # Packed inputs
    if cross_segment_mask is not None:
      cross_attention_mask = jnp.minimum(cross_attention_mask,
                                         cross_segment_mask)
  return attention_mask, cross_attention_mask


def compute_attention_masks_for_extend_step(
    time_step: JTensor,
    seq_len: int,
    segment_mask: Optional[JTensor] = None,
    cross_inputs: Optional[JTensor] = None,
    cross_paddings: Optional[JTensor] = None,
    cross_segment_mask: Optional[JTensor] = None
) -> Tuple[JTensor, Union[JTensor, None]]:
  """Computes attention mask from paddings, segment masks etc for extend_step.

  Args:
    time_step: Time step for which to generate causal mask.
    seq_len: Sequence length for generating causal mask.
    segment_mask: if not None, per step segment mask JTensor for this time step,
      of shape [B, 1, T].
    cross_inputs: Source sequence JTensor of shape [B, S, D].
    cross_paddings: Source paddings JTensor of shape [B, S].
    cross_segment_mask: if not None, cross_segment_mask JTensor for this time
      step, of shape [B, 1, S].

  Returns:
    attention_mask: Attention mask JTensor ready to add to logits for self
      attention of shape [B/1, 1, 1, T].
    cross_attention_mask: Attention mask JTensor ready to add to logits for
      cross attention of shape [B/1, 1, 1, S]. This will be None if
      cross_inputs are None.
  """
  # Create a broadcast friendly version of time step of shape [1, 1]
  batch_time_step = jnp.asarray(time_step, dtype=jnp.uint32)
  batch_time_step = jnp.reshape(batch_time_step, [1, 1])

  # Create causal padding by masking out any index > time_step.
  # [1, T], 0 for non-pad and 1 for pad.
  causal_padding = jnp.greater(
      jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step)

  # Create attention mask from padding of shape [1/B, 1, T]
  attention_mask = jnp.squeeze(
      attentions.convert_paddings_to_mask(causal_padding), axis=1)

  # Include segment mask, has shape [B, 1, T]
  if segment_mask is not None:
    attention_mask = jnp.minimum(attention_mask, segment_mask)

  # Compute cross attention mask if applicable
  cross_attention_mask = None
  if cross_inputs is not None:
    assert cross_paddings is not None

    # Compute paddings mask [B, 1, 1, S]
    cross_attention_mask = attentions.convert_paddings_to_mask(
        cross_paddings, dtype=cross_inputs.dtype)

    # Cross segment mask may be overloaded
    if cross_segment_mask is not None:
      # [B, 1, S] -> [B, 1, 1, S]
      cross_segment_mask = jnp.expand_dims(cross_segment_mask, axis=1)
      cross_attention_mask = jnp.minimum(cross_attention_mask,
                                         cross_segment_mask)
  return attention_mask, cross_attention_mask


class TransformerFeedForward(base_layer.BaseLayer):
  """Transformer feedforward layer with residual connection and dropout."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define(
        'output_dims', 0, 'Depth of the output.'
        'If unset or output_dims == input_dims,'
        'there is no residual projection layer.'
        'Otherwise, add a residual projection layer '
        'followed by batch normalization.')
    p.Define('hidden_dims', 0, 'Hidden dimension of FFN')
    p.Define('has_bias', True, 'Adds bias weights to Feedforward or not.')
    p.Define(
        'apply_padding_first', False,
        'Apply padding to inputs before everything else or not. For '
        'example, it is better to apply padding before batch norm.')
    p.Define(
        'activation', 'RELU', 'Activation function to use.'
        'Options are RELU, RELU6, RELU^2, RELU^3, SIGMOID, TANH, GELU,'
        'GATED_GELU, GATED_SILU, NONE.')
    p.Define('fflayer_tpl', linears.FeedForward.Params(),
             'Feedforward layer params')
    p.Define('ln_tpl', normalizations.LayerNorm.Params(),
             'Layer norm params, other options include RmsNorm as well.')
    p.Define('residual_dropout_prob', 0., 'Residual dropout')
    p.Define(
        'relu_dropout_tpl', stochastics.Dropout.Params(),
        'Relu dropout params template. keep_prop will be reset to '
        '(1.0 - relu_dropout_prob).')
    p.Define('relu_dropout_prob', 0., 'FFN dropout')
    p.Define(
        'residual_dropout_tpl', stochastics.Dropout.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define('add_skip_connection', True, 'Whether to add residual connection')
    p.Define(
        'residual_weight', 1.0, 'Weight of the residual connection. '
        'Output = fn(x) * residual_weight + x.')
    p.Define(
        'norm_policy', 'pre',
        'Policy for applying normaliztion wrt. transformations. '
        'Options are: '
        '(1) "pre", applied before transformation.'
        '(2) "primer_hybrid", applied before and after transformation.')
    p.Define(
        'internal_gshard_variance_scaling_fan_in_init', False,
        'Feedforward weight init follows uniform distribution with'
        ' bound = 1.0 / sqrt(3 / dim_0).')
    p.weight_split_dims_mapping = py_utils.Params()
    wp = p.weight_split_dims_mapping
    wp.Define('ffn0', None,
              'Weight-split dims mapping for the first ffw network.')
    wp.Define('ffn1', None,
              'Weight-split dims mapping for the second ffw network.')

    p.activation_split_dims_mapping = py_utils.Params()
    ap = p.activation_split_dims_mapping
    ap.Define('ffn0', None,
              'Activation-split dims mapping for the first ffw network.')
    ap.Define('ffn1', None,
              'Activation-split dims mapping for the second ffw network.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    if p.output_dims == 0:
      # Make it compatible with previous implementation
      p.output_dims = p.input_dims

    if p.output_dims != p.input_dims:
      self.create_child(
          'res_proj',
          linears.Linear.Params().Set(
              input_dims=p.input_dims,
              output_dims=p.output_dims,
          ))
      self.create_child(
          'res_proj_norm',
          normalizations.BatchNorm.Params().Set(dim=p.output_dims))

    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping
    # Create Layer Norm
    if p.norm_policy == 'primer_hybrid':
      ln_p = p.ln_tpl.Copy()
      ln_p.input_dims = p.input_dims
      self.create_child('pre_layer_norm', ln_p)
      self.create_child('post_layer_norm', ln_p)
    elif p.norm_policy == 'pre':
      ln_p = p.ln_tpl.Copy()
      ln_p.name = 'fflayer_ln'
      ln_p.input_dims = p.input_dims
      self.create_child('layer_norm', ln_p)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % p.norm_policy)

    if p.activation.startswith('GATED_'):
      activation = 'NONE'
      gate_activation = p.activation[len('GATED_'):]
      self._is_ffn1_gated = True
    else:
      activation = p.activation
      gate_activation = None
      self._is_ffn1_gated = False

    # Create the first Feedforward layer mapping to hidden dims
    ffn1_p = p.fflayer_tpl.Copy()
    ffn1_p.name = 'ffn_layer1'
    ffn1_p.input_dims = p.input_dims
    ffn1_p.has_bias = p.has_bias
    ffn1_p.activation = activation
    ffn1_p.output_dims = p.hidden_dims
    ffn1_p.weight_split_dims_mapping.wt = wp.ffn0
    ffn1_p.activation_split_dims_mapping.out = ap.ffn0
    if p.internal_gshard_variance_scaling_fan_in_init:
      scale = (1. / p.input_dims)**0.5 * (3.0**0.5)
      ffn1_p.linear_tpl.params_init = WeightInit.Uniform(scale)
    self.create_child('ffn_layer1', ffn1_p)

    if self._is_ffn1_gated:
      # This is a gated ffw network, corresponding to gshard_builder's wi0
      gate_p = p.fflayer_tpl.Copy()
      gate_p.name = 'ffn_layer1_gate'
      gate_p.input_dims = p.input_dims
      gate_p.has_bias = p.has_bias
      gate_p.activation = gate_activation
      gate_p.output_dims = p.hidden_dims
      gate_p.weight_split_dims_mapping.wt = wp.ffn0
      gate_p.activation_split_dims_mapping.out = ap.ffn0
      if p.internal_gshard_variance_scaling_fan_in_init:
        scale = (1. / p.input_dims)**0.5 * (3.0**0.5)
        gate_p.linear_tpl.params_init = WeightInit.Uniform(scale)
      self.create_child('ffn_layer1_gate', gate_p)

    # Create RELU dropout layer
    relu_dropout_p = p.relu_dropout_tpl.Copy()
    relu_dropout_p.keep_prob = 1.0 - p.relu_dropout_prob
    self.create_child('relu_dropout', relu_dropout_p)

    # Create the second Feedforward layer mapping to input dims
    ffn2_p = p.fflayer_tpl.Copy()
    ffn2_p.name = 'ffn_layer2'
    ffn2_p.input_dims = p.hidden_dims
    ffn2_p.has_bias = p.has_bias
    ffn2_p.activation = 'NONE'
    ffn2_p.output_dims = p.output_dims
    ffn2_p.weight_split_dims_mapping.wt = wp.ffn1
    ffn2_p.activation_split_dims_mapping.out = ap.ffn1
    if p.internal_gshard_variance_scaling_fan_in_init:
      scale = (1. / p.hidden_dims)**0.5 * (3.0**0.5)
      ffn2_p.linear_tpl.params_init = WeightInit.Uniform(scale)
    self.create_child('ffn_layer2', ffn2_p)

    # Create residual dropout layer
    residual_dropout_p = p.residual_dropout_tpl.Copy()
    residual_dropout_p.keep_prob = 1.0 - p.residual_dropout_prob
    self.create_child('residual_dropout', residual_dropout_p)

  def fprop(self,
            theta: NestedMap,
            inputs: JTensor,
            paddings: Optional[JTensor] = None) -> JTensor:
    p = self.params
    # Expand paddings to last dim if not None to have shape [batch, time, 1]
    if paddings is not None:
      paddings = jnp.expand_dims(paddings, axis=-1)

    if p.apply_padding_first and paddings is not None:
      inputs *= (1.0 - paddings)

    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm.fprop(theta.pre_layer_norm,
                                                    inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm.fprop(theta.layer_norm, inputs)
    else:
      inputs_normalized = inputs

    # Apply first FFN layer
    if self._is_ffn1_gated:
      # theta.ffn_layer1_gate corresponds to gshard_builder's wi0
      gate_value = self.ffn_layer1_gate.fprop(theta.ffn_layer1_gate,
                                              inputs_normalized)
      # theta.ffn_layer1 corresponds to gshard_builder's wi1
      projected_inputs = gate_value * self.ffn_layer1.fprop(
          theta.ffn_layer1, inputs_normalized)
    else:
      projected_inputs = self.ffn_layer1.fprop(theta.ffn_layer1,
                                               inputs_normalized)
      projected_inputs = checkpoint_name(projected_inputs, 'ffn1')

    # Apply paddings if not None
    if not p.apply_padding_first and paddings is not None:
      projected_inputs *= (1.0 - paddings)

    # Apply RELU dropout
    projected_inputs = self.relu_dropout.fprop(theta.relu_dropout,
                                               projected_inputs)

    # Apply second FFN layer
    projected_inputs = self.ffn_layer2.fprop(theta.ffn_layer2, projected_inputs)
    projected_inputs = checkpoint_name(projected_inputs, 'ffn2')

    # Apply paddings if not None
    if not p.apply_padding_first and paddings is not None:
      projected_inputs *= (1.0 - paddings)

    # Apply Primer normalization before dropout.
    if p.norm_policy == 'primer_hybrid':
      projected_inputs = self.post_layer_norm.fprop(theta.post_layer_norm,
                                                    projected_inputs)

    # Apply residual dropout
    projected_inputs = self.residual_dropout.fprop(theta.residual_dropout,
                                                   projected_inputs)

    if hasattr(self, 'res_proj'):
      inputs = self.res_proj_norm.fprop(
          theta.res_proj_norm, self.res_proj.fprop(theta.res_proj, inputs))

    # Apply skip connection
    if p.add_skip_connection:
      projected_inputs = projected_inputs * p.residual_weight + inputs

    return projected_inputs


class TransformerFeedForwardMoe(base_layer.BaseLayer):
  """A sharded MoE Layer.

  This is a drop-in replacement of the transformer feedforward layer. It is a
  composite of the following sub-layers.

  ln_inputs = ln(inputs)
  moe_output = moe(ln_inputs)
  drop_output = dropout(moe_output)
  output = inputs + drop_output
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Dimension of the layer input.')
    p.Define('hidden_dims', 0, 'Dimension of the hidden layer.')
    p.Define(
        'apply_padding_first', False,
        'Apply padding to inputs before everything else or not. For '
        'example, it is better to apply padding before batch norm.')
    # NOTE(yonghui): layer-norm as used in gshard doesn't have bias and assume
    # the mean is 0.0. See gshard_builder._LN for more details.
    p.Define('ln_tpl', normalizations.LayerNorm.Params(),
             'Layer norm default params.')
    p.Define(
        'activation', 'RELU', 'Activation function to use.'
        'Options are RELU, RELU6, SIGMOID, TANH, NONE.')
    p.Define(
        'relu_dropout_tpl', stochastics.Dropout.Params(),
        'Relu dropout params template. keep_prop will be reset to '
        '(1.0 - relu_dropout_prob).')
    p.Define(
        'relu_dropout_prob', 0.0,
        'Probability at which we apply dropout to the hidden layer of '
        'feedforward network.')
    p.Define(
        'residual_dropout_tpl', stochastics.Dropout.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, y) = (x + dropout(y)).')
    p.Define('add_skip_connection', True,
             'If True, add skip_connection from input to output.')
    p.Define(
        'residual_weight', 1.0, 'Weight applied on residual connection.'
        'Final output is residual_weight * residual_fn(x) + x.'
        'Only in effect when add_skip_connection is True.')
    p.Define(
        'norm_policy', 'pre',
        'Policy for applying normaliztion wrt. transformations. '
        'Options are: '
        '(1) "pre", applied before transformation.'
        '(2) "primer_hybrid", applied before and after transformation.')
    p.Define('residual_droppath_prob', 0.0,
             'Probability at which we drop the entire residual path.')
    p.Define('num_experts', 0, 'Total number of experts in this layer.')
    p.Define(
        'num_groups', 0,
        'Total number of groups for dispatching. num_groups typically'
        'should be the same as num devices')
    p.Define(
        'min_group_size', None,
        'If not None, num_groups will be adjusted so that there will be '
        'at least min_group_size tokens in each group.')
    p.Define(
        'expert_capacity_dim', 0, 'Internal. Exact expert capacity. '
        'Setting non-zero expert_capacity_factor is a preferred way.')
    p.Define(
        'expert_capacity_factor', 1.0,
        'Expert capacity_factor. This is the ratio between max allowed '
        'examples per expert over the average number of examples per '
        'expert assuming routing is completely uniform.')
    p.Define(
        'expert_weight_shards', 1,
        'Shard each expert params into this many number of shards to '
        'reduce the size of individual weight params.')
    p.Define('second_expert_policy', 'all',
             'How to pick second expert: all, sampling or random.')
    p.Define('internal_gshard_variance_scaling_fan_in_init', True,
             'Internal. Do not use. To study MoE layer init.')
    p.Define('moe_load_balance_loss_weight', 1.0,
             'Weight for the load balancing loss of the MoE layer')
    # SPMD partition related params.
    # M - model_dim, for both inputs and outputs
    # E - experts dim
    # G - groups dim
    # C - experts capacity dim
    # H - hidden dim
    # S - sequence dim
    p.weight_split_dims_mapping = py_utils.Params()
    wp = p.weight_split_dims_mapping
    wp.Define(
        'me', None, 'Sharding for the gating network weight, of shape '
        '[input_dims, num_experts].')
    wp.Define(
        'emh', None, 'Sharding of the first projection matrix that maps '
        'from input to hidden dim, of shape '
        '[num_experts, input_dims, hidden_dims].')
    wp.Define(
        'ehm', None, 'Sharding of the second projection matrix that maps '
        'from hidden to output dim, of shape '
        '[num_experts, hidden_dims, output_dims].')

    p.activation_split_dims_mapping = py_utils.Params()
    ap = p.activation_split_dims_mapping
    ap.Define('gsm', None, 'Sharding of the gsm tensors.')
    ap.Define('gs', None, 'Sharding of the gs tensors.')
    ap.Define('gsec', None, 'Sharding of the gsec tensors.')
    ap.Define('egcm', None, 'Sharding of the egcm tensors.')
    ap.Define('egch', None, 'Sharding of the egch tensors.')
    ap.Define('gecm', None, 'Sharding of the gecm tensors.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dims
    assert p.hidden_dims

    assert (p.expert_capacity_factor or p.expert_capacity_dim)
    assert p.num_experts > 0
    assert p.num_groups > 0
    assert p.expert_weight_shards == 1, (
        f'[Deprecated] Should be removed {p.expert_weight_shards} != 1')

    if p.norm_policy == 'primer_hybrid':
      params = p.ln_tpl.Copy()
      params.input_dims = p.input_dims
      self.create_child('pre_layer_norm', params)
      self.create_child('post_layer_norm', params)
    elif p.norm_policy == 'pre':
      params = p.ln_tpl.Copy()
      params.name = 'layer_norm'
      params.input_dims = p.input_dims
      self.create_child('layer_norm', params)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % p.norm_policy)

    dropout_tpl = p.residual_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.create_child('residual_dropout', dropout_tpl)

    dropout_tpl = p.relu_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.relu_dropout_prob)
    self.create_child('relu_dropout', dropout_tpl)

    if p.residual_droppath_prob > 0:
      assert p.add_skip_connection
      droppath_p = stochastics.StochasticResidual.Params().Set(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.create_child('residual_droppath', droppath_p)

    act_p = activations_lib.Activation.Params().Set(activation=p.activation)
    self.create_child('activation', act_p)

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    # Assume output_dims == input_dims
    output_dims = p.input_dims

    # First create the gating network.
    wp = p.weight_split_dims_mapping
    gate_init = None  # default xavier init
    if p.internal_gshard_variance_scaling_fan_in_init:
      # TODO(lepikhin): this init is related with Adafactor settings, study
      stddev = (1.0 / p.input_dims)**0.5
      gate_scale = stddev * 3.0**0.5
      gate_init = WeightInit.Uniform(gate_scale)
    gate_pc = weight_params(
        shape=[p.input_dims, p.num_experts],
        init=gate_init,
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.me)
    for l in gate_pc.ToText().split('\n'):
      logging.debug('moe gate weight_params %s', l)
    self.create_variable('gate', gate_pc)

    # Next create the expert network.
    # Params initialization follows gshard_builder.py.
    # emh tensor typically mesh-shard on first dim and last dim. Hence, here we
    # split the tensor manually into multiple tensors on the second dim.
    emh_shape = [
        p.num_experts, p.input_dims // p.expert_weight_shards, p.hidden_dims
    ]
    wi_init = None
    if p.internal_gshard_variance_scaling_fan_in_init:
      stddev = (1.0 / p.input_dims)**0.5
      wi_init_scale = stddev * 3.0**0.5
      wi_init = WeightInit.Uniform(wi_init_scale)
    wi_pc = weight_params(
        shape=emh_shape,
        init=wi_init,
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.emh)
    for l in wi_pc.ToText().split('\n'):
      logging.debug('moe wi weight_params %s', l)
    for ii in range(p.expert_weight_shards):
      self.create_variable('wi_%d' % ii, wi_pc)

    # EHM Tensor (output transformation after RELU)
    # ehm tensor typically shard on the first dim and the second dim. Here we
    # manually split the tensor on the last dim into multiple tensors.
    ehm_shape = [
        p.num_experts, p.hidden_dims, output_dims // p.expert_weight_shards
    ]
    wo_init = None
    if p.internal_gshard_variance_scaling_fan_in_init:
      wi_init = None
      stddev = (1.0 / p.hidden_dims)**0.5
      wo_init_scale = stddev * 3.0**0.5
      wo_init = WeightInit.Uniform(wo_init_scale)
    wo_pc = weight_params(
        shape=ehm_shape,
        init=wo_init,
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.ehm)
    for l in wo_pc.ToText().split('\n'):
      logging.debug('moe wo weight_params %s', l)
    for ii in range(p.expert_weight_shards):
      self.create_variable('wo_%d' % ii, wo_pc)

  # TODO(zhangqiaorjc): Allow paddings to be optional?
  def fprop(self, theta: NestedMap, inputs: JTensor,
            paddings: JTensor) -> JTensor:
    """Layer-norm, route, feed-forward, combine, residual.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: [batch, seq_len, model].
      paddings: [batch, seq_len].

    Returns:
      Tensor of the same shape as inputs.
    """
    p = self.params
    # Assume output_dims == input_dims
    output_dims = p.input_dims

    # Consistent with gshard implementation.
    if p.apply_padding_first:
      inputs *= (1.0 - jnp.expand_dims(paddings, axis=-1))

    # TODO(zhangqiaorjc): Handle input of shape [batch, seq_len, g, model/g]?
    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm.fprop(theta.pre_layer_norm,
                                                    inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm.fprop(theta.layer_norm, inputs)
    else:
      inputs_normalized = inputs

    assert len(inputs_normalized.shape) == 3
    bs, s_len, m_dim = inputs_normalized.shape
    assert len(paddings.shape) == 2
    assert paddings.shape == (bs, s_len)

    num_groups = p.num_groups
    assert num_groups
    if (p.min_group_size is not None and
        bs * s_len / num_groups < p.min_group_size):
      num_groups = (bs * s_len + p.min_group_size - 1) // p.min_group_size
      logging.info('num_groups adjusted to %s.', num_groups)
    assert (bs * s_len) % num_groups == 0
    g_len = (bs * s_len) // num_groups
    reshaped_inputs = inputs_normalized.reshape([num_groups, g_len, m_dim])
    reshaped_paddings = paddings.reshape([num_groups, g_len])

    # Sharding annotation.
    ap = p.activation_split_dims_mapping

    def split(t_in, sharding):
      return base_layer.maybe_shard(t_in, sharding, p.mesh_axis_names)

    reshaped_inputs = split(reshaped_inputs, ap.gsm)
    reshaped_paddings = split(reshaped_paddings, ap.gs)

    fprop_dtype = py_utils.fprop_dtype(p)
    logits = jnp.einsum('gsm,me->gse', reshaped_inputs, theta.gate)

    # Here and below, we assume num devices equals num groups.
    # TODO(yonghui): Expose some of the options below through params.
    # NOTE(yonghui): The following code might break during beam search decode
    # due to much smaller group size.
    # TODO(yonghui): Avoid explicitly casting everything to fp32 once
    # top2_gating_on_logits is stable in low-precision mode.
    # TODO(lepikhin): Validate stability. mask_dtype=np.int32 and
    # logits.astype(np.float32) should generally be sufficient.
    gating = gshard_utils.top2_gating_on_logits(
        paddings=reshaped_paddings.astype(fprop_dtype),
        logits=logits.astype(jnp.float32),
        experts_dim=p.num_experts,
        expert_capacity_dim=p.expert_capacity_dim,
        fprop_dtype=fprop_dtype,
        prng_key=base_layer.next_prng_key(),
        second_expert_policy=p.second_expert_policy,
        second_expert_threshold=0.0,
        # legacy_mtf_behavior=True doesn't normalize gates when one expert is
        # being dropped. This is more appropriate for routing decisions like
        # 'random'.
        legacy_mtf_behavior=True,
        # *2.0 because we choose top-2 experts per example
        capacity_factor=2.0 * p.expert_capacity_factor,
        mask_dtype=jnp.int32)

    aux_loss, combine_tensor, dispatch_tensor, summary = gating
    over_capacity_1_ratio, over_capacity_2_ratio = summary
    base_layer.add_summary('over_capacity_1_ratio', over_capacity_1_ratio)
    base_layer.add_summary('over_capacity_2_ratio', over_capacity_2_ratio)

    if fprop_dtype != np.float32:
      aux_loss = aux_loss.astype(fprop_dtype)
      combine_tensor = combine_tensor.astype(fprop_dtype)
      dispatch_tensor = dispatch_tensor.astype(fprop_dtype)

    # both tensors have shape [g, s, e, c]
    combine_tensor = split(combine_tensor, ap.gsec)
    dispatch_tensor = split(dispatch_tensor, ap.gsec)

    theta_wis = []
    theta_wos = []
    for ii in range(p.expert_weight_shards):
      theta_wis.append(theta.get('wi_%d' % ii))
      theta_wos.append(theta.get('wo_%d' % ii))

    # Concatenate theta_wis and theta_wos
    # since each sub-theta_wi has shape
    # (p.num_experts, p.input_dims // p.expert_weight_shards, p.hidden_dims)
    # and each sub-theta_wo has shape
    # (p.num_experts, p.hidden_dims, output_dims // p.expert_weight_shards)
    if len(theta_wis) == 1:
      theta_wi = theta_wis[0]
    else:
      # new shape: (p.num_experts, p.input_dims, p.hidden_dims)
      theta_wi = jnp.concatenate(theta_wis, 1)

    if len(theta_wos) == 1:
      theta_wo = theta_wos[0]
    else:
      # new shape: (p.num_experts, p.hidden_dims, output_dims)
      theta_wo = jnp.concatenate(theta_wos, 2)

    expert_inputs = jnp.einsum('gsec,gsm->egcm', dispatch_tensor,
                               reshaped_inputs)
    expert_inputs = split(expert_inputs, ap.egcm)

    hidden = jnp.einsum('egcm,emh->egch', expert_inputs, theta_wi)
    hidden = split(hidden, ap.egch)

    # Activation function.
    hidden = self.activation.fprop(theta.activation, hidden)
    # Dropout.
    hidden = self.relu_dropout.fprop(theta.relu_dropout, hidden)
    # Output.
    expert_output = jnp.einsum('egch,ehm->egcm', hidden, theta_wo)
    expert_output = split(expert_output, ap.egcm)
    # Now transpose and reshard.
    transposed_expert_output = jnp.einsum('egcm->gecm', expert_output)
    transposed_expert_output = split(transposed_expert_output, ap.gecm)
    combined_output = jnp.einsum('gecm,gsec->gsm', transposed_expert_output,
                                 combine_tensor)
    combined_output = split(combined_output, ap.gsm)

    combined_output = combined_output.reshape((bs, s_len, output_dims))
    # Apply padding.
    combined_output *= (1.0 - jnp.expand_dims(paddings, -1)).astype(fprop_dtype)
    # Primer normalization before dropout.
    if p.norm_policy == 'primer_hybrid':
      combined_output = self.post_layer_norm.fprop(theta.post_layer_norm,
                                                   combined_output)
    # Residual dropout.
    after_residual = self.residual_dropout.fprop(theta.residual_dropout,
                                                 combined_output)
    if p.add_skip_connection:
      if p.residual_droppath_prob:
        out = self.residual_droppath.fprop(theta.residual_droppath, inputs,
                                           after_residual)
      else:
        out = inputs + after_residual * p.residual_weight

    # Add loss to a global collection. We don't return the loss to the caller
    # to avoid the change of the api here.
    assert p.moe_load_balance_loss_weight, (
        'p.moe_load_balance_loss_weight > 0 when there is an aux '
        'load balancing loss in MoE layers.')
    aux_loss *= p.moe_load_balance_loss_weight
    base_layer.add_summary('aux_moe_load_balance_loss_weight', aux_loss)
    aux_loss_ctx = py_utils.AuxLossContext.Current()
    if aux_loss_ctx is not None:
      aux_loss_ctx.AddLoss(aux_loss)

    return out


class Transformer(base_layer.BaseLayer):
  """Transformer layer with multi-headed attention."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Dimension of the transformer block input.')
    p.Define('hidden_dims', 0, 'Hidden dimension of FFN layer.')
    p.Define('num_heads', None, 'Num of heads in self attention.')
    p.Define(
        'dim_per_head', None, 'Dimension of each attention head. If None then '
        'dim_per_head == hidden_dim // num_heads.')
    p.Define(
        'dropout_tpl', stochastics.Dropout.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define('atten_dropout_prob', 0.0,
             'Probability at which we apply dropout to the attention weights.')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, y) = (x + dropout(y)).')
    p.Define('relu_dropout_prob', 0.0,
             'Probability at which we apply dropout to the FFN layers.')
    p.Define('mask_self_attention', False, 'If True, use causal mask.')
    p.Define('cross_attention', False, 'If True, perform cross'
             'encoder-decoder attention.')
    p.Define(
        'cross_atten_tpl', None, 'Optional cross attention params template'
        'that can be set when cross attention is enabled. If cross'
        'attention is enabled and this is set to None, then cross'
        'attention params will be inherited from tr_atten_tpl.')
    p.Define('ln_tpl', normalizations.LayerNorm.Params(), 'Layer norm params.')
    p.Define(
        'norm_policy', 'pre',
        'Policy for applying normaliztion wrt. transformations. '
        'Options are: '
        '(1) "pre", applied before transformation.'
        '(2) "primer_hybrid", applied before and after transformation.')
    p.Define('tr_atten_tpl', attentions.DotProductAttention.Params(),
             'DotProductAttention Layer params.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define('tr_fflayer_tpl', TransformerFeedForward.Params(),
             'Transformer Feed-Forward Layer params.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    # Initialize Layer Norm
    if p.norm_policy == 'primer_hybrid':
      params = p.ln_tpl.Copy()
      params.input_dims = p.input_dims
      self.create_child('pre_layer_norm', params)
      self.create_child('post_layer_norm', params)
    elif p.norm_policy == 'pre':
      params = p.ln_tpl.Copy()
      params.name = 'layer_norm'
      params.input_dims = p.input_dims
      self.create_child('layer_norm', params)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % p.norm_policy)

    # Initialize multi-headed self-attention
    params = p.tr_atten_tpl.Copy()
    params.name = 'multihead_self_atten'
    params.input_dim = p.input_dims
    params.hidden_dim = p.input_dims
    params.num_heads = p.num_heads
    params.dim_per_head = p.dim_per_head
    params.atten_dropout_prob = p.atten_dropout_prob
    self.create_child('self_attention', params)

    # Initialize residual dropout.
    params = p.dropout_tpl.Copy()
    params.keep_prob = (1.0 - p.residual_dropout_prob)
    self.create_child('residual_dropout', params)

    # Initialize multi-headed cross-attention and layer norm.
    if p.cross_attention:
      params = p.ln_tpl.Copy()
      params.name = 'cross_layer_norm'
      params.input_dims = p.input_dims
      self.create_child('cross_layer_norm', params)

      if p.cross_atten_tpl is not None:
        params = p.cross_atten_tpl.Copy()
      else:
        params = p.tr_atten_tpl.Copy()
      params.name = 'multihead_cross_atten'
      params.input_dim = p.input_dims
      params.hidden_dim = p.input_dims
      params.num_heads = p.num_heads
      params.dim_per_head = p.dim_per_head
      params.atten_dropout_prob = p.atten_dropout_prob
      # Note that cross attention should not use any position embeddings.
      if params.use_rotary_position_emb:
        raise ValueError('Rotary position embedding should not be enabled for '
                         'cross attention.')
      # Note that cross attention should not use depth-wise convolution.
      if params.dconv_qkv:
        raise ValueError('Depth-wise convolution should not be enabled for '
                         'cross attention.')
      self.create_child('cross_attention', params)

    # Initialize feed-forward layer
    if p.tr_fflayer_tpl:
      params = p.tr_fflayer_tpl.Copy()
      params.name = 'tr_fflayer'
      params.input_dims = p.input_dims
      params.hidden_dims = p.hidden_dims
      params.relu_dropout_prob = p.relu_dropout_prob
      params.residual_dropout_prob = p.residual_dropout_prob
      params.norm_policy = p.norm_policy
      self.create_child('ff_layer', params)

  def init_states(self, theta: NestedMap, target_batch_size: int,
                  target_max_length: int) -> NestedMap:
    """Initialize the cache for the Transformer layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      target_batch_size: Batch size for the target.
      target_max_length: The length to decode the target.

    Returns:
      Initialized cache for decoding.
    """
    return self.self_attention.init_states(theta.self_attention,
                                           target_batch_size, target_max_length)

  def fprop(
      self,
      theta: NestedMap,
      inputs: JTensor,
      paddings: JTensor,
      attention_mask: JTensor,
      cross_inputs: Optional[JTensor] = None,
      cross_attention_mask: Optional[JTensor] = None,
      segment_pos: Optional[JTensor] = None,
  ) -> Tuple[JTensor, JTensor]:
    """Transformer decoder layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T] (only used in FFN layer).
      attention_mask: Self attention mask ready to add to the logits. It can be
        of shape [B/1, 1, T/1, T] which is broadcast compatible with the self
        attention matrix of shape [B, N, T, T]. This is assumed to have combined
        paddings, causal masking as well as segment maskings.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_attention_mask: Cross attention mask ready to add to the logits. It
        can be of shape [B/1, 1, T/1, S] which is broadcast compatible with the
        cross attention matrix of shape [B, N, T, T]. This is assumed to have
        combined paddings as well as segment maskings.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.

    Returns:
      The fflayer output with shape [B, T, D].
      atten_probs: A NestedMap with keys `self_atten` <float>[B, N, T, T].
    """
    # Layer normalize input
    p = self.params

    inputs_stats = stats.compute_stats(inputs, jnp.expand_dims(paddings, -1))
    base_layer.add_summary('xformer_input_mean', inputs_stats.mean_v)
    base_layer.add_summary('xformer_input_std', inputs_stats.std_v)
    base_layer.add_summary('xformer_input_abs_max', inputs_stats.max_v)

    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm.fprop(theta.pre_layer_norm,
                                                    inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm.fprop(theta.layer_norm, inputs)
    else:
      inputs_normalized = inputs

    # Compute self-attention, key/value vectors are the input itself
    atten_output, self_atten_probs = self.self_attention.fprop(
        theta.self_attention,
        inputs_normalized,
        inputs_normalized,
        inputs_normalized,
        atten_mask=attention_mask,
        query_segment_pos=segment_pos)
    atten_probs = NestedMap(self_atten=self_atten_probs)
    if p.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm.fprop(theta.post_layer_norm,
                                                atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout.fprop(theta.residual_dropout,
                                               atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if self.params.cross_attention:
      assert cross_inputs is not None
      assert cross_attention_mask is not None
      # TODO(davidso): integrate primer_hybrid normalization with
      #.               cross_attention.
      assert p.norm_policy != 'primer_hybrid'

      cross_atten_output, cross_atten_probs = self.cross_attention.fprop(
          theta.cross_attention,
          self.cross_layer_norm.fprop(theta.cross_layer_norm, atten_output),
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)
      atten_probs.cross_atten = cross_atten_probs

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout.fprop(theta.residual_dropout,
                                                       cross_atten_output)
      atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer.fprop(
        theta.ff_layer, atten_output, paddings=paddings)
    return output, atten_probs

  def extend_step(
      self,
      theta: NestedMap,
      cached_states: NestedMap,
      inputs: JTensor,
      *,
      time_step: JTensor,
      attention_mask: JTensor,
      cross_inputs: Optional[JTensor] = None,
      cross_attention_mask: Optional[JTensor] = None
  ) -> Tuple[JTensor, NestedMap]:
    """Transformer decoder layer, autoregressive cached decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for cached decoding. key   - [T, B,
        N, H]. value - [T, B, N, H].
      inputs: Target sequence of shape [B, D] corresponding to target sequence
        at index time_step.
      time_step: A scalar, the current decode step, 0-based.
      attention_mask: per step attention mask for this time step, of shape [B,
        1, T]. This combines causal mask with any segment mask if applicable.
      cross_inputs: Source sequence - [B, S, H].
      cross_attention_mask: if not None, cross_segment_mask for this time step,
        of shape [B, 1, 1, S]. This combines padding mask with any segment mask
        if applicable.

    Returns:
      (updated_states, cur_output)
      * updated_states: A `.NestedMap` object containing the updated states.
      * cur_output: [B, D]
      key   - [T, B, N, H].
      value - [T, B, N, H].
    """
    if not self.params.mask_self_attention:
      raise ValueError('extend_step should only be called with causal masking.')

    p = self.params
    # Layer normalize input
    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm.fprop(theta.pre_layer_norm,
                                                    inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm.fprop(theta.layer_norm, inputs)

    # Self-attention layer.
    updated_states, atten_output = self.self_attention.extend_step(
        theta.self_attention,
        cached_states,
        inputs_normalized,
        atten_mask=attention_mask,
        time_step=time_step)
    if p.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm.fprop(theta.post_layer_norm,
                                                atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout.fprop(theta.residual_dropout,
                                               atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if self.params.cross_attention:
      assert cross_inputs is not None
      assert cross_attention_mask is not None

      atten_output_normalized = self.cross_layer_norm.fprop(
          theta.cross_layer_norm, jnp.expand_dims(atten_output, axis=1))
      cross_atten_output, _ = self.cross_attention.fprop(
          theta.cross_attention,
          atten_output_normalized,
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout.fprop(theta.residual_dropout,
                                                       cross_atten_output)
      # Squeeze sequence dim
      cross_atten_output = jnp.squeeze(cross_atten_output, axis=1)
      atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer.fprop(theta.ff_layer, atten_output)
    return updated_states, output


class StackedTransformer(base_layer.BaseLayer):
  """A stack of Transformer layers."""

  @classmethod
  def GLaMParams(cls,
                 model_dim,
                 ff_dim,
                 attention_num_heads,
                 attention_key_value_dim,
                 name='transformer',
                 moe=False,
                 moe_hidden_dim=None,
                 ffn_activation='GATED_GELU',
                 mask_self_attention=True,
                 cross_attention=False,
                 attention_extra_logit=0.0,
                 relative_attention_num_buckets=32,
                 relative_attention_max_distance=128,
                 moe_load_balance_loss_weight=0.01,
                 num_groups=1,
                 c_dim=None,
                 capacity_factor=0.0,
                 e_dim=None):
    """Common setup for GLaM Transformer layers.

    This function setups a transformer block for both MoE and dense GLaM models.
    The MoE block consists of two transformer layer with the feedforward
    sublayer of the first one replaced by a MoE layer. The dense block consists
    of a transformer. The transformer layer used by GLam differs from the
    standard transformer in these configs:
    1) The feedforward sublayer used gated gleu so there are two wi and one wo.
    2) No bias in all projections.
    3) Use no bias RMS norm for the layer norm.
    4) Use relative attention bias

    Args:
      model_dim: model dimension.
      ff_dim: hidden dimension of feed-forward inner layer.
      attention_num_heads: number of attention heads.
      attention_key_value_dim: key value dimension of attention inner layer.
      name: Name of the this layer
      moe: If this is a moe block or not.
      moe_hidden_dim: hidden dimension of MoE layer.
      ffn_activation: Activation function used in the ffn layer.
      mask_self_attention: Use masked self-attention.
      cross_attention: If set, use cross encoder-decoder attention layer.
      attention_extra_logit: Extra logit for attention softmax.
      relative_attention_num_buckets: Relative attention num buckets
      relative_attention_max_distance: Max relative distance.
      moe_load_balance_loss_weight: Weight of load balancing loss in MoE layers.
      num_groups: Total number of groups for token dispatching in MoE layer.
      c_dim: Expert capacity.
      capacity_factor: This is the ratio between max allowed examples per expert
        over the average number of examples per expert assuming routing is
        completely uniform.
      e_dim: Number of experts.

    Returns:
      A Params object to set up a StackedTransformer.
    """

    p = cls.Params()
    p.name = name
    p.enable_while_loop = True
    p.packed_input = True
    p.num_layers_per_block = 2 if moe else 1
    p.num_blocks = 1
    p.moe_layers = [0] if moe else None
    p.model_dims = model_dim
    p.hidden_dims = ff_dim
    p.num_heads = attention_num_heads
    p.dim_per_head = attention_key_value_dim
    p.num_experts = e_dim
    p.num_groups = num_groups
    p.mask_self_attention = mask_self_attention
    p.cross_attention = cross_attention
    # Attention setup
    p.transformer_layer_params_tpl.ln_tpl = normalizations.RmsNorm.Params()
    p.transformer_layer_params_tpl.ln_tpl.direct_scale = True
    tr_atten_tpl = p.transformer_layer_params_tpl.tr_atten_tpl
    assert tr_atten_tpl.cls == attentions.DotProductAttention
    tr_atten_tpl.attention_extra_logit = attention_extra_logit
    tr_atten_tpl.use_bias = False
    tr_atten_tpl.internal_gshard_gaussian_init = True
    tr_atten_tpl.internal_enable_per_dim_scale = False
    tr_atten_tpl.relative_bias_tpl = attentions.RelativeBias.Params().Set(
        relative_attention_num_buckets=relative_attention_num_buckets,
        relative_attention_max_distance=relative_attention_max_distance)
    tr_atten_tpl.output_proj_use_nhd_shape = True
    # Non-MoE ffn setup
    ff_tpl = p.transformer_layer_params_tpl.tr_fflayer_tpl
    assert ff_tpl.cls == TransformerFeedForward
    ff_tpl.input_dims = model_dim
    ff_tpl.hidden_dims = ff_dim
    ff_tpl.has_bias = False
    ff_tpl.apply_padding_first = True
    ff_tpl.ln_tpl = normalizations.RmsNorm.Params()
    ff_tpl.ln_tpl.direct_scale = True
    ff_tpl.add_skip_connection = True
    ff_tpl.activation = ffn_activation
    ff_tpl.internal_gshard_variance_scaling_fan_in_init = True
    # MoE ffn setup
    moe_p = p.moe_layer_tpl
    assert moe_p.cls == TransformerFeedForwardMoe
    moe_p.input_dims = model_dim
    moe_p.hidden_dims = ff_dim
    moe_p.ln_tpl = normalizations.RmsNorm.Params()
    moe_p.ln_tpl.direct_scale = True
    moe_p.num_experts = e_dim
    moe_p.num_groups = num_groups
    moe_p.expert_capacity_dim = c_dim
    moe_p.expert_capacity_factor = capacity_factor
    moe_p.internal_gshard_variance_scaling_fan_in_init = True
    moe_p.moe_load_balance_loss_weight = moe_load_balance_loss_weight
    return p

  @staticmethod
  def DefineParams(p):
    p.Define('cross_attention', False,
             'If set, introduces cross encoder-decoder attention layer.')
    p.Define('mask_self_attention', False, 'Use masked self-attention.')
    p.Define('num_layers', 0, 'Num of layers in this stack.')
    # TODO(lepikhin): consider adding explicit block scope for blocks of layers
    # so checkpoint has
    # transformer/block_{0...num_blocks-1}/layer_{0...num_layers_per_block-1}
    # p.Define('block_scope', False, 'Explicit block scope.')
    #
    # You must specify:
    #   p.num_layers or
    #   p.num_blocks and p.num_layers_per_block,
    # so that p.num_layers == p.num_blocks * p.num_layers_per_block.
    p.Define('num_blocks', None, 'Number of blocks.')
    p.Define('num_layers_per_block', None, 'Block size.')
    p.Define('model_dims', 0, 'Model dimension in Transformer layers.')
    p.Define('hidden_dims', 0,
             'The hidden layer dimension of FFN in Transformer layers.')
    p.Define('num_heads', 0, 'Number of attention heads.')
    p.Define(
        'dim_per_head', None, 'Dimension of each attention head. If None then '
        'dim_per_head == hidden_dim // num_heads.')
    p.Define('dropout_prob', 0.0,
             'Apply dropout at this prob at various places.')
    p.Define(
        'transformer_layer_params_tpl', Transformer.Params(),
        'A template of Transformer.params, can be a list of params '
        'of length equal to the num_layers or a factor of num_layers.'
        'For a factor, the params are tiled as [a, a, ..., b, b,...,].')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define(
        'fold_padding_with_segment_mask', False, 'If True then segment'
        'mask is supposed to include the padding mask as well, i.e.'
        'treating PADs as one sequence and non-PADs as another.')
    p.Define(
        'enable_while_loop', False,
        'Whether or not to use a while loop to unroll the transformer layer'
        ' stack. Potential benefits: 1) reduce xla compilation time. '
        ' 2) improve hbm usage due to explicit rematerialization.')
    p.Define(
        'checkpoint_policy', recurrent.AutodiffCheckpointType.SAVE_NOTHING,
        'How to checkpoint residuals for BProp: save nothing, dot only or '
        'dot with no batch dimensions.')
    # MoE related params.
    p.Define('moe_layer_tpl', TransformerFeedForwardMoe.Params(),
             'Template configuration for the moe feedforward layer.')
    p.Define('num_experts', 0, 'Total number of experts.')
    p.Define('num_groups', 1, 'Num of groups for dispathcing.')
    p.Define(
        'min_group_size', None,
        'If not None, num_groups will be adjusted so that there will be '
        'at least min_group_size tokens in each group.')
    p.Define('moe_layers', [], 'List of MoE layer indices, e.g. [0, 2, 4].')
    return p

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p = StackedTransformer.DefineParams(p)
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    if p.num_blocks:
      assert p.num_layers_per_block > 0
      assert not p.num_layers
      p.num_layers = p.num_blocks * p.num_layers_per_block
    else:
      assert p.num_layers > 0
      p.num_blocks = p.num_layers
      p.num_layers_per_block = 1
    assert p.model_dims > 0
    assert p.hidden_dims > 0
    assert p.num_heads > 0
    assert 0.0 <= p.dropout_prob < 1.0

    def _layer_params(i):
      """Construct i-th layer params."""
      p_i = p.transformer_layer_params_tpl.Copy()
      p_i.name = f'layer_{i}'
      p_i.cross_attention = p.cross_attention
      p_i.mask_self_attention = p.mask_self_attention
      p_i.num_heads = p.num_heads
      p_i.dim_per_head = p.dim_per_head
      p_i.input_dims = p.model_dims
      p_i.packed_input = p.packed_input
      p_i.atten_dropout_prob = p.dropout_prob
      p_i.residual_dropout_prob = p.dropout_prob
      p_i.relu_dropout_prob = p.dropout_prob
      p_i.hidden_dims = p.hidden_dims
      if i in p.moe_layers:
        assert p.num_experts > 0
        moe_p = p.moe_layer_tpl.Copy()
        moe_p.num_experts = p.num_experts
        moe_p.num_groups = p.num_groups
        moe_p.min_group_size = p.min_group_size
        p_i.tr_fflayer_tpl = moe_p
      return p_i

    layer_params = [_layer_params(i) for i in range(p.num_layers)]
    self.create_children('x_layers', layer_params)

  def init_states(self, theta: NestedMap, *args: Any,
                  **kwargs: Any) -> NestedMap:
    """Initialize the cache for the StackedTransformer layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      Initialized cache for decoding.
    """
    return NestedMap(x_layers=[
        layer.init_states(layer_theta, *args, **kwargs)
        for layer, layer_theta in zip(self.x_layers, theta.x_layers)
    ])

  def fprop(self,
            theta: NestedMap,
            inputs: JTensor,
            paddings: JTensor,
            segment_mask: Optional[JTensor] = None,
            cross_inputs: Optional[JTensor] = None,
            cross_paddings: Optional[JTensor] = None,
            cross_segment_mask: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> JTensor:
    """Stacked Transformer layer.

    Args:
      theta: A `NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      segment_mask: Segment mask for packed input of shape [B, 1, T, T] ready to
        add to logits.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_paddings: Paddings for cross atention of shape [B, S].
      cross_segment_mask: Segment mask for encoder-decoder in packed input case
        of shape [B, 1, T, S].
      segment_pos: Segment pos for packed input of shape [B, T].

    Returns:
      Output vector with shape [B, T, D].
    """
    p = self.params
    x_out = inputs
    if p.packed_input:
      assert segment_mask is not None

    if p.cross_attention:
      assert cross_inputs is not None
      assert cross_paddings is not None
      if p.packed_input:
        assert cross_segment_mask is not None

    attention_mask, cross_attention_mask = compute_attention_masks_for_fprop(
        inputs,
        paddings,
        p.mask_self_attention,
        segment_mask,
        cross_inputs,
        cross_paddings,
        cross_segment_mask,
        fold_padding_with_segment_mask=p.fold_padding_with_segment_mask)

    if p.enable_while_loop:

      num_blocks = p.num_layers // p.num_layers_per_block

      def _stack_vars(*args):
        args = [x[jnp.newaxis, :] for x in args]
        return jnp.vstack(args)

      # We stack variables independently for each layer in the block, e.g.
      # for each index i from 0 to num_layers_per_block - 1, in the
      # trivial case num_layers_per_block=1 it's equivalent to
      #
      # stacked_vars = py_utils.NestedMap(layer_000=tf.nest.map_structure(
      #     _stack_vars, *theta.x_layers))
      #
      stacked_vars = []
      for i in range(p.num_layers_per_block):
        x_layers_i = theta.x_layers[i::p.num_layers_per_block]
        stacked_vars_i = tf.nest.map_structure(_stack_vars, *x_layers_i)
        stacked_vars.append(stacked_vars_i)

      def _key(i):
        # NestedMap requires string keys
        return 'layer_%03d' % i

      stacked_vars = py_utils.NestedMap(
          {_key(i): v for i, v in enumerate(stacked_vars)})
      # aux_loss is a cumulative aux_loss, we need ys to have per-layer
      # aux_loss increments
      carry = py_utils.NestedMap(
          x_in=x_out, aux_loss=jnp.asarray(0., dtype=self.fprop_dtype))

      # TODO(lepikhin): generalize this function to be a more generic one
      def _scan_fn(carry, block_vars):
        # TODO(b/199950567): Sharding propagation does not seem to handle scan
        # boundary well. We need to annotate all parameters from within the scan
        # body even though we already pass them to pjit invocation outside the
        # scan at the top level once. Consider removing after bug fix.
        def annotate_var_sharding_constraint(x, weight_param):
          partition_spec = base_layer.var_partition_specs(
              weight_param,
              device_mesh=p.device_mesh,
              device_axis_names=p.mesh_axis_names)
          return base_layer.with_sharding_constraint(x, partition_spec)

        aux_loss = carry.aux_loss
        with py_utils.AuxLossContext(reentrant=True) as al_ctx:
          assert al_ctx is not None
          x_out = carry.x_in
          for i in range(p.num_layers_per_block):
            layer_vars_i = block_vars[_key(i)]
            if p.device_mesh is not None:
              assert p.mesh_axis_names is not None
              layer_vars_i = tf.nest.map_structure(
                  annotate_var_sharding_constraint, layer_vars_i,
                  self.x_layers[i].vars)
            x_out, _ = self.x_layers[i].fprop(
                layer_vars_i,
                x_out,
                paddings,
                attention_mask,
                cross_inputs,
                cross_attention_mask,
                segment_pos=segment_pos,
            )
          if al_ctx.aux_losses:
            assert isinstance(al_ctx.aux_losses, list)
            aux_loss_inc = sum(al_ctx.aux_losses).astype(self.fprop_dtype)
            base_layer.add_summary('aux_loss_inc', aux_loss_inc)
            aux_loss = aux_loss + aux_loss_inc

        return py_utils.NestedMap(
            x_in=x_out, aux_loss=aux_loss), py_utils.NestedMap()

      carry_final, _, summaries = recurrent.scan(
          carry,
          stacked_vars,
          _scan_fn,
          root_layer=self,
          checkpoint_policy=p.checkpoint_policy)

      # Now unpack summaries to the out-context.
      # Q(yonghui): Shall we move summary handling to be within recurrent.scan?
      for summary_key, summary_value in summaries.items():
        # unstack summary_value
        logging.info((summary_key, summary_value))
        assert summary_value.shape[0] == num_blocks
        unstacked_values = jnp.split(summary_value, num_blocks)
        # Q(yonghui): shall we keep the summaries packed instead?
        for i, v in enumerate(unstacked_values):
          # TODO(yonghui): Here we assume summaries are all scalar values.
          base_layer.add_summary(f'{summary_key}/{i}', v)

      x_out = carry_final.x_in
      aux_loss_ctx = py_utils.AuxLossContext.Current()
      # Scan can not have sideeffects so we have to capture side effect
      # "aux_loss" in the moe layer and propagate it explicitly.
      if aux_loss_ctx is not None:
        aux_loss_ctx.AddLoss(carry_final.aux_loss)
    else:
      for i in range(p.num_layers):
        x_in = x_out
        x_out, _ = self.x_layers[i].fprop(
            theta.x_layers[i],
            x_in,
            paddings,
            attention_mask,
            cross_inputs,
            cross_attention_mask,
            segment_pos=segment_pos)
    return x_out

  def extend_step(
      self,
      theta: NestedMap,
      cached_states: NestedMap,
      inputs: JTensor,
      *,
      time_step: JTensor,
      segment_mask: Optional[JTensor] = None,
      cross_inputs: Optional[JTensor] = None,
      cross_paddings: Optional[JTensor] = None,
      cross_segment_mask: Optional[JTensor] = None
  ) -> Tuple[JTensor, NestedMap]:
    """Transformer stacked decoder layers, autoregressive cached decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for cached decoding.
        cached_states.x_layers is a list corresponding to self.x_layers with key
        - [T, B, N, H]. value - [T, B, N, H].
      inputs: Target sequence of shape [B, D] corresponding to target sequence
        at index time_step.
      time_step: A scalar, the current decode step, 0-based.
      segment_mask: if not None, per step segment mask for this time step, of
        shape [B, 1, T].
      cross_inputs: Source sequence - [B, S, D].
      cross_paddings: Source paddings - [B, S].
      cross_segment_mask: if not None, cross_segment_mask for this time step, of
        shape [B, 1, S].

    Returns:
      updated_states: A `.NestedMap` object containing the updated states.
      updated_states.x_layers is a list corresponding to self.x_layers, where
      each element is a NestedMap with attention keys and values:
      cur_output: The last decoder layer output of shape [B, D].

      - key - [T, B, N, H].
      - value - [T, B, N, H].
    """
    p = self.params
    if not self.params.mask_self_attention:
      raise ValueError('extend_step should only be used with masked attention')

    if 'key' in cached_states.x_layers[0]:
      key = cached_states.x_layers[0].key
      max_t = key.shape[0]
    else:
      raise ValueError('Must call init_states before extend_step')

    if p.cross_attention:
      assert cross_inputs is not None
      assert cross_paddings is not None

    attention_mask, cross_attention_mask = (
        compute_attention_masks_for_extend_step(time_step, max_t, segment_mask,
                                                cross_inputs, cross_paddings,
                                                cross_segment_mask))

    updated_states = NestedMap(x_layers=[])
    decoder_input = inputs
    for layer, layer_theta, layer_states in zip(self.x_layers, theta.x_layers,
                                                cached_states.x_layers):
      updated_layer_states, decoder_output = layer.extend_step(
          layer_theta,
          layer_states,
          decoder_input,
          time_step=time_step,
          attention_mask=attention_mask,
          cross_inputs=cross_inputs,
          cross_attention_mask=cross_attention_mask)
      updated_states.x_layers.append(updated_layer_states)
      decoder_input = decoder_output
    return updated_states, decoder_output


class StackedTransformerRepeated(base_layer.BaseLayer):
  """A StackedTransformer implemented using the generic Repeat."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'block', StackedTransformer.Params(),
        'The params of a block. A block can be a single transformer layer,'
        ' multiple layers, or a dense layer followed by a sparse layer, '
        ' a.k.a. MOE block.')
    p.Define('x_times', 0, 'Num times to repeat a block.')
    p.Define(
        'checkpoint_policy', recurrent.AutodiffCheckpointType.SAVE_NOTHING,
        'How to checkpoint residuals for BProp: save nothing, dot only or '
        'dot with no batch dimensions.')
    wp = p.weight_split_dims_mapping
    wp.Define('block', None, 'How the list of blocks should be sharded.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    wp = p.weight_split_dims_mapping

    repeat_l_params = repeats.Repeat.Params().Set(
        sub=p.block,
        x_times=p.x_times,
        checkpoint_policy=p.checkpoint_policy,
        unpack_summaries=True)
    repeat_l_params.weight_split_dims_mapping.sub = wp.block

    self.create_child('repeat', repeat_l_params)

  def fprop(self,
            theta: NestedMap,
            inputs: JTensor,
            paddings: JTensor,
            segment_mask: Optional[JTensor] = None,
            cross_inputs: Optional[JTensor] = None,
            cross_paddings: Optional[JTensor] = None,
            cross_segment_mask: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> JTensor:
    """Stacked Transformer layer.

    Args:
      theta: A `NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      segment_mask: Segment mask for packed input of shape [B, 1, T, T] ready to
        add to logits.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_paddings: Paddings for cross atention of shape [B, S].
      cross_segment_mask: Segment mask for encoder-decoder in packed input case
        of shape [B, 1, T, S].
      segment_pos: Segment position of shape [B, T].

    Returns:
      Output vector with shape [B, T, D].
    """

    def sub_fprop_fn(sub, theta, inputs, *args: Any, **kwargs: Any):
      with py_utils.AuxLossContext(reentrant=True) as al_ctx:
        assert al_ctx is not None
        # sub is expected to be an instance of StackedTransformer
        assert isinstance(sub, StackedTransformer)
        out = sub.fprop(theta, inputs, *args, **kwargs)

        if al_ctx.aux_losses:
          assert isinstance(al_ctx.aux_losses, list)
          aux_loss_inc = sum(al_ctx.aux_losses).astype(self.fprop_dtype)
          base_layer.add_summary('aux_loss_inc', aux_loss_inc)
        else:
          aux_loss_inc = jnp.array(0.0, dtype=self.fprop_dtype)

        return out, NestedMap(aux_loss=aux_loss_inc)

    out, stacked_extra = self.repeat.fprop(sub_fprop_fn, theta.repeat, inputs,
                                           paddings, segment_mask, cross_inputs,
                                           cross_paddings, cross_segment_mask,
                                           segment_pos)
    aux_loss = jnp.sum(stacked_extra.aux_loss)
    aux_loss_ctx = py_utils.AuxLossContext.Current()
    # Scan can not have sideeffects so we have to capture side effect
    # "aux_loss" in the moe layer and propagate it explicitly.
    if aux_loss_ctx is not None:
      aux_loss_ctx.AddLoss(aux_loss)
    return out

  def init_states(self, theta: NestedMap, *args: Any,
                  **kwargs: Any) -> NestedMap:
    """Initialize the cache for the StackedTransformerRepeated layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      Initialized cache for decoding.
    """

    def init_fn(block, theta, *args, **kwargs):
      assert isinstance(block, StackedTransformer)
      return block.init_states(theta, *args, **kwargs)

    return self.repeat.init_states(init_fn, theta.repeat, *args, **kwargs)

  def extend_step(
      self,
      theta: NestedMap,
      cached_states: NestedMap,
      inputs: JTensor,
      *,
      time_step: JTensor,
      segment_mask: Optional[JTensor] = None,
      cross_inputs: Optional[JTensor] = None,
      cross_paddings: Optional[JTensor] = None,
      cross_segment_mask: Optional[JTensor] = None
  ) -> Tuple[JTensor, NestedMap]:
    """Transformer stacked decoder layers, autoregressive cached decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      cached_states: A `.NestedMap` object containing tensors which are the
        results of previous attentions, used for cached decoding.
        cached_states.x_layers is a list corresponding to self.x_layers with key
        - [T, B, N, H]. value - [T, B, N, H].
      inputs: Target sequence of shape [B, D] corresponding to target sequence
        at index time_step.
      time_step: A scalar, the current decode step, 0-based.
      segment_mask: if not None, per step segment mask for this time step, of
        shape [B, 1, T].
      cross_inputs: Source sequence - [B, S, D].
      cross_paddings: Source paddings - [B, S].
      cross_segment_mask: if not None, cross_segment_mask for this time step, of
        shape [B, 1, S].

    Returns:
      updated_states: A `.NestedMap` object containing the updated states.
      updated_states.x_layers is a list corresponding to self.x_layers, where
      each element is a NestedMap with attention keys and values:
      cur_output: The last decoder layer output of shape [B, D].

      - key - [T, B, N, H].
      - value - [T, B, N, H].
    """

    def extend_fn(sub, theta, cached_states, step_inputs, *args, **kwargs):
      assert isinstance(sub, StackedTransformer)
      return sub.extend_step(theta, cached_states, step_inputs, *args, **kwargs)

    return self.repeat.extend_step(
        extend_fn,
        theta.repeat,
        cached_states,
        inputs,
        time_step=time_step,
        segment_mask=segment_mask,
        cross_inputs=cross_inputs,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask)


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
    p.Define('stacked_transformer_tpl', StackedTransformer.Params(),
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
                               moe_load_balance_loss_weight=0.01,
                               z_loss_weight=1e-4):
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
      moe_load_balance_loss_weight: Weight of the aux loss for MoE layers.
      z_loss_weight: additional loss term to stablize the final softmax logit.

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
        embedding_softmax.GShardSharedEmebeddingSoftmax.Params().Set(
            name='emb',
            input_dims=model_dim,
            num_classes=vocab_size,
            z_loss_weight=z_loss_weight))

    glam_p = StackedTransformer.GLaMParams(
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
        e_dim=e_dim)

    p.stacked_transformer_tpl = StackedTransformerRepeated.Params()
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
    if softmax_p.cls == embedding_softmax.GShardSharedEmebeddingSoftmax:
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

    if lm_p.stacked_transformer_tpl.cls == StackedTransformer:
      xformer_p = lm_p.stacked_transformer_tpl.transformer_layer_params_tpl
    elif lm_p.stacked_transformer_tpl.cls == StackedTransformerRepeated:
      xformer_p = (
          lm_p.stacked_transformer_tpl.block.transformer_layer_params_tpl)
    else:
      assert False, f'{lm_p.stacked_transformer_tpl.cls} not supported.'

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
    if lm_p.stacked_transformer_tpl.cls == StackedTransformer:
      moe_p = lm_p.stacked_transformer_tpl.moe_layer_tpl
    elif lm_p.stacked_transformer_tpl.cls == StackedTransformerRepeated:
      moe_p = lm_p.stacked_transformer_tpl.block.moe_layer_tpl
    else:
      assert False, f'{lm_p.stacked_transformer_tpl.cls} not supported.'
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
    if p.stacked_transformer_tpl.cls == StackedTransformer:
      xformer_params = p.stacked_transformer_tpl
    elif p.stacked_transformer_tpl.cls == StackedTransformerRepeated:
      xformer_params = p.stacked_transformer_tpl.block
    else:
      assert False, f'{p.stacked_transformer_tpl.cls} not supported.'
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

  def init_states(self, theta: NestedMap, *args: Any,
                  **kwargs: Any) -> NestedMap:
    """Initialize the cache for the autoregressive decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      A `.NestedMap` corresponding to the cache.
    """
    return NestedMap(
        step=jnp.array(0, dtype=jnp.uint32),
        transformer=self.transformer.init_states(theta.transformer, *args,
                                                 **kwargs))

  def compute_loss(self,
                   theta: NestedMap,
                   activations: JTensor,
                   labels: Optional[NestedMap] = None) -> NestedMap:
    """Computes cross entropy loss.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
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
      logits = self.softmax.get_logits(theta=theta.softmax, inputs=activations)
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
          theta.softmax,
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
            theta: NestedMap,
            inputs: JTensor,
            paddings: JTensor,
            labels: Optional[NestedMap] = None,
            segment_ids: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> NestedMap:
    """Computes xent loss given the language model inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
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
        input_emb = self.embedding_lookup.fprop(theta.embedding_lookup, inputs)
      else:
        input_emb = self.softmax.emb_lookup(theta.softmax, inputs)
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
            theta.ngrammer,
            input_ids=inputs,
            input_embs=input_emb,
            paddings=paddings,
            segment_pos=segment_pos)

      if p.position_emb_tpl is not None:
        position_emb = self.position_emb.fprop(
            theta.position_emb, seq_length=seq_length, position=segment_pos)
        inputs = input_emb + position_emb
      else:
        inputs = input_emb

      if p.masked_lm:
        segment_mask = attentions.segment_mask(segment_ids, segment_ids,
                                               inputs.dtype)
      else:
        segment_mask = attentions.causal_segment_mask(segment_ids, inputs.dtype)

      output = self.transformer.fprop(
          theta.transformer,
          inputs,
          paddings,
          segment_mask=segment_mask,
          segment_pos=segment_pos)

      # Final layer norm
      if p.final_ln_tpl is not None:
        output = self.final_ln.fprop(theta.final_ln, output)
      return self.compute_loss(theta, output, labels)

  def extend_step(
      self,
      theta: NestedMap,
      cached_states: NestedMap,
      inputs: JTensor,
  ) -> Tuple[NestedMap, NestedMap]:
    """Autoregressive cached decoding of Transformer LM.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
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
      input_emb = self.embedding_lookup.fprop(theta.embedding_lookup, inputs)
    else:
      input_emb = self.softmax.emb_lookup(theta.softmax, inputs)
    time_step = cached_states.step

    # Add Ngrammer layer if applicable.
    if p.ngrammer_tpl is not None:
      input_emb = self.ngrammer.fprop(
          theta.ngrammer, inputs, input_emb, paddings=None, segment_pos=None)
      inputs = inputs[:, -1][:, jnp.newaxis]
      input_emb = input_emb[:, -1, :][:, jnp.newaxis, :]

    if p.position_emb_tpl is not None:
      # During autoregressive decoding inputs are not packed.
      segment_pos = jnp.zeros((inputs.shape[0], 1)) + time_step
      position_emb = self.position_emb.fprop(
          theta.position_emb, seq_length=1, position=segment_pos)

      inputs = input_emb + position_emb
    else:
      inputs = input_emb

    updated_cache, outputs = self.transformer.extend_step(
        theta.transformer,
        cached_states.transformer,
        inputs[:, 0, :],
        time_step=time_step)
    cached_states.transformer = updated_cache
    cached_states.step += 1
    if p.final_ln_tpl is not None:
      outputs = self.final_ln.fprop(theta.final_ln, outputs)
    xent_output = self.compute_loss(theta, outputs)
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
      if stacked_transformer_tpl.cls == StackedTransformer:
        assert (stacked_transformer_tpl.model_dims == 0 or
                stacked_transformer_tpl.model_dims == model_dims)
        stacked_transformer_tpl.model_dims = model_dims
        stacked_transformer_tpl.packed_input = packed_input
      elif stacked_transformer_tpl.cls == StackedTransformerRepeated:
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
    if encoder_params.cls == StackedTransformer:
      mask_self_attention = encoder_params.mask_self_attention
    elif encoder_params.cls == StackedTransformerRepeated:
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
    if decoder_params.cls == StackedTransformer:
      mask_self_attention = decoder_params.mask_self_attention
    elif decoder_params.cls == StackedTransformerRepeated:
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
              theta: NestedMap,
              inputs: JTensor,
              input_paddings: JTensor,
              input_segment_ids: Optional[JTensor] = None,
              input_segment_pos: Optional[JTensor] = None) -> JTensor:
    """Apply the Transformer encoder to the source sequence.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
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
      input_emb = self.encoder_embedding_lookup.fprop(
          theta.encoder_embedding_lookup, inputs)
    elif p.decoder_embedding_tpl is not None:
      # Encoder shares the same embedding as the target ids.
      # The embedding lookup for target ids is separate from the softmax.
      input_emb = self.decoder_embedding_lookup.fprop(
          theta.decoder_embedding_lookup, inputs)
    else:
      # Encoder and decoder share the softmax and embedding params.
      input_emb = self.softmax.emb_lookup(theta.softmax, inputs)

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
          theta.encoder_ngrammer,
          input_ids=inputs,
          input_embs=input_emb,
          paddings=input_paddings,
          segment_pos=input_segment_pos)

    if p.position_emb_tpl is not None:
      position_emb = self.position_emb.fprop(
          theta.position_emb, seq_length=seq_length, position=input_segment_pos)
      input_emb += position_emb

    inputs_segment_mask = attentions.segment_mask(
        input_segment_ids, dtype=input_emb.dtype)
    encoder_output = self.encoder.fprop(
        theta.encoder,
        input_emb,
        input_paddings,
        segment_mask=inputs_segment_mask)

    # Final layer norm for encoder output.
    encoder_output = self.encoder_ln.fprop(theta.encoder_ln, encoder_output)
    return encoder_output

  def compute_loss(self,
                   theta: NestedMap,
                   activations: JTensor,
                   labels: Optional[NestedMap] = None) -> NestedMap:
    """Computes cross entropy loss.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
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
      logits = self.softmax.get_logits(theta=theta.softmax, inputs=activations)
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
          theta.softmax,
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
      theta: NestedMap,
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
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
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

    encoder_output = self._encode(theta, inputs, input_paddings,
                                  input_segment_ids, input_segment_pos)

    if p.decoder_embedding_tpl is not None:
      # Targets have separate embedding params.
      target_emb = self.decoder_embedding_lookup.fprop(
          theta.decoder_embedding_lookup, targets)
    else:
      # Embedding parameters are shared with targets and softmax.
      target_emb = self.softmax.emb_lookup(theta.softmax, targets)

    if p.decoder_ngrammer_tpl is not None:
      target_emb = self.decoder_ngrammer.fprop(
          theta.decoder_ngrammer,
          input_ids=targets,
          input_embs=target_emb,
          paddings=target_paddings,
          segment_pos=target_segment_pos)

    if p.position_emb_tpl is not None:
      targets_position_emb = self.position_emb.fprop(
          theta.position_emb,
          seq_length=target_seq_length,
          position=target_segment_pos)
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
        theta.decoder,
        target_emb,
        target_paddings,
        target_segment_mask,
        cross_inputs=encoder_output,
        cross_paddings=input_paddings,
        cross_segment_mask=cross_segment_mask)

    # Final layer norm for decoder.
    output = self.decoder_ln.fprop(theta.decoder_ln, output)

    return self.compute_loss(theta, output, labels)

  def init_states(self, theta: NestedMap, inputs: JTensor,
                  input_paddings: JTensor, *args: Any,
                  **kwargs: Any) -> NestedMap:
    """Initialize the cache for autoregressive decoding.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
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
        decoder=self.decoder.init_states(theta.decoder, *args, **kwargs))
    encoder_output = self._encode(
        theta,
        inputs,
        input_paddings,
        input_segment_ids=None,
        input_segment_pos=None)
    cache.encoder_output = encoder_output
    cache.input_paddings = input_paddings
    return cache

  def extend_step(self, theta: NestedMap, cached_states: NestedMap,
                  targets: JTensor) -> Tuple[NestedMap, NestedMap]:
    """Autoregressive cached decoding of the Transformer encoder decoder.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
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
      target_emb = self.decoder_embedding_lookup.fprop(
          theta.decoder_embedding_lookup, targets)
    else:
      # Embedding parameters are shared with targets and softmax.
      target_emb = self.softmax.emb_lookup(theta.softmax, targets)

    time_step = cached_states.step
    if p.decoder_ngrammer_tpl is not None:
      target_emb = self.decoder_ngrammer.fprop(
          theta.decoder_ngrammer,
          targets,
          target_emb,
          paddings=None,
          segment_pos=None)

    targets = targets[:, -1][:, jnp.newaxis]
    target_emb = target_emb[:, -1, :][:, jnp.newaxis, :]

    # Add position embeddings to target ids.
    if p.position_emb_tpl is not None:
      # During autoregressive decoding inputs are not packed.
      segment_pos = jnp.zeros((targets.shape[0], 1)) + time_step
      target_position_emb = self.position_emb.fprop(
          theta.position_emb, seq_length=1, position=segment_pos)
      target_emb += target_position_emb

    updated_cache, outputs = self.decoder.extend_step(
        theta.decoder,
        cached_states.decoder,
        target_emb[:, 0, :],
        time_step=time_step,
        cross_inputs=encoder_output,
        cross_paddings=input_paddings)
    cached_states.decoder = updated_cache
    cached_states.step += 1
    outputs = self.decoder_ln.fprop(theta.decoder_ln, outputs)
    xent_output = self.compute_loss(theta, outputs)
    return cached_states, xent_output
