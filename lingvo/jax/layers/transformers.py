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
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from lingvo.jax import base_layer
from lingvo.jax import gshard_utils
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import activations as activations_lib
from lingvo.jax.layers import attentions
from lingvo.jax.layers import linears
from lingvo.jax.layers import normalizations
from lingvo.jax.layers import pipeline
from lingvo.jax.layers import recurrent
from lingvo.jax.layers import repeats
from lingvo.jax.layers import stats
from lingvo.jax.layers import stochastics
import numpy as np

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
weight_params = py_utils.weight_params

InstantiableParams = py_utils.InstantiableParams
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
    p.Define('residual_droppath_prob', 0.0,
             'Probability at which we drop the entire residual path.')
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

    if p.residual_droppath_prob > 0:
      assert p.add_skip_connection
      droppath_p = stochastics.StochasticResidual.Params().Set(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.create_child('residual_droppath', droppath_p)

  def fprop(self,
            inputs: JTensor,
            paddings: Optional[JTensor] = None) -> JTensor:
    p = self.params
    # Expand paddings to last dim if not None to have shape [batch, time, 1]
    if paddings is not None:
      paddings = jnp.expand_dims(paddings, axis=-1)

    if p.apply_padding_first and paddings is not None:
      inputs *= (1.0 - paddings)

    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm.fprop(inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm.fprop(inputs)
    else:
      inputs_normalized = inputs

    # Apply first FFN layer
    if self._is_ffn1_gated:
      # theta.ffn_layer1_gate corresponds to gshard_builder's wi0
      gate_value = self.ffn_layer1_gate.fprop(inputs_normalized)
      # theta.ffn_layer1 corresponds to gshard_builder's wi1
      projected_inputs = gate_value * self.ffn_layer1.fprop(inputs_normalized)
    else:
      projected_inputs = self.ffn_layer1.fprop(inputs_normalized)
      projected_inputs = checkpoint_name(projected_inputs, 'ffn1')

    # Apply paddings if not None
    if not p.apply_padding_first and paddings is not None:
      projected_inputs *= (1.0 - paddings)

    # Apply RELU dropout
    projected_inputs = self.relu_dropout.fprop(projected_inputs)

    # Apply second FFN layer
    projected_inputs = self.ffn_layer2.fprop(projected_inputs)
    projected_inputs = checkpoint_name(projected_inputs, 'ffn2')

    # Apply paddings if not None
    if not p.apply_padding_first and paddings is not None:
      projected_inputs *= (1.0 - paddings)

    # Apply Primer normalization before dropout.
    if p.norm_policy == 'primer_hybrid':
      projected_inputs = self.post_layer_norm.fprop(projected_inputs)

    # Apply residual dropout
    projected_inputs = self.residual_dropout.fprop(projected_inputs)

    if hasattr(self, 'res_proj'):
      inputs = self.res_proj_norm.fprop(self.res_proj.fprop(inputs))

    # Apply skip connection
    if p.add_skip_connection:
      if p.residual_droppath_prob:
        projected_inputs = self.residual_droppath.fprop(inputs,
                                                        projected_inputs)
      else:
        projected_inputs = inputs + projected_inputs * p.residual_weight

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

  def fprop(self, inputs: JTensor, paddings: JTensor = None) -> JTensor:  # pytype: disable=annotation-type-mismatch  # jax-ndarray
    """Layer-norm, route, feed-forward, combine, residual.

    Args:
      inputs: [batch, seq_len, model].
      paddings: [batch, seq_len], optional when called by extend_step.

    Returns:
      Tensor of the same shape as inputs.
    """
    p = self.params
    theta = self.local_theta()
    # Assume output_dims == input_dims
    output_dims = p.input_dims
    fprop_dtype = self.fprop_dtype

    # Consistent with gshard implementation.
    if p.apply_padding_first and paddings is not None:
      inputs *= (1.0 - jnp.expand_dims(paddings, axis=-1))

    # TODO(zhangqiaorjc): Handle input of shape [batch, seq_len, g, model/g]?
    if p.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm.fprop(inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm.fprop(inputs)
    else:
      inputs_normalized = inputs

    assert len(inputs_normalized.shape) in [2, 3]
    token_shape = inputs_normalized.shape[:-1]
    num_tokens = np.prod(token_shape)
    m_dim = inputs_normalized.shape[-1]
    if paddings is not None:
      assert paddings.shape == token_shape

    num_groups = p.num_groups
    assert num_groups
    if (p.min_group_size is not None and
        num_tokens / num_groups < p.min_group_size):
      num_groups = (num_tokens + p.min_group_size - 1) // p.min_group_size
      logging.info('num_groups adjusted to %s.', num_groups)
    assert num_tokens % num_groups == 0
    g_len = num_tokens // num_groups

    # Sharding annotation.
    ap = p.activation_split_dims_mapping

    def split(t_in, sharding):
      return base_layer.maybe_shard(t_in, sharding, p.mesh_axis_names)

    reshaped_inputs = inputs_normalized.reshape([num_groups, g_len, m_dim])
    reshaped_inputs = split(reshaped_inputs, ap.gsm)
    if paddings is not None:
      reshaped_paddings = paddings.reshape([num_groups, g_len])
      reshaped_paddings = split(reshaped_paddings, ap.gs)
      reshaped_paddings = reshaped_paddings.astype(fprop_dtype)
    else:
      reshaped_paddings = None

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
        paddings=reshaped_paddings,
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
    hidden = self.activation.fprop(hidden)
    # Dropout.
    hidden = self.relu_dropout.fprop(hidden)
    # Output.
    expert_output = jnp.einsum('egch,ehm->egcm', hidden, theta_wo)
    expert_output = split(expert_output, ap.egcm)
    # Now transpose and reshard.
    transposed_expert_output = jnp.einsum('egcm->gecm', expert_output)
    transposed_expert_output = split(transposed_expert_output, ap.gecm)
    combined_output = jnp.einsum('gecm,gsec->gsm', transposed_expert_output,
                                 combine_tensor)
    combined_output = split(combined_output, ap.gsm)

    combined_output = combined_output.reshape(token_shape + (output_dims,))
    # Apply padding.
    if paddings is not None:
      combined_output *= (1.0 -
                          jnp.expand_dims(paddings, -1)).astype(fprop_dtype)
    # Primer normalization before dropout.
    if p.norm_policy == 'primer_hybrid':
      combined_output = self.post_layer_norm.fprop(combined_output)
    # Residual dropout.
    after_residual = self.residual_dropout.fprop(combined_output)
    if p.add_skip_connection:
      if p.residual_droppath_prob:
        out = self.residual_droppath.fprop(inputs, after_residual)
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
    p.Define('residual_droppath_prob', 0.0,
             'Probability at which we drop the entire residual path.')
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

    # Initialize residual droppath
    if p.residual_droppath_prob > 0:
      droppath_p = stochastics.StochasticResidual.Params().Set(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.create_child('residual_droppath', droppath_p)

    # Initialize feed-forward layer
    if p.tr_fflayer_tpl:
      params = p.tr_fflayer_tpl.Copy()
      params.name = 'tr_fflayer'
      params.input_dims = p.input_dims
      params.hidden_dims = p.hidden_dims
      params.relu_dropout_prob = p.relu_dropout_prob
      params.residual_dropout_prob = p.residual_dropout_prob
      params.residual_droppath_prob = p.residual_droppath_prob
      params.norm_policy = p.norm_policy
      self.create_child('ff_layer', params)

  def init_states(self, target_batch_size: int,
                  target_max_length: int) -> NestedMap:
    """Initialize the cache for the Transformer layer.

    Args:
      target_batch_size: Batch size for the target.
      target_max_length: The length to decode the target.

    Returns:
      Initialized cache for decoding.
    """
    return self.self_attention.init_states(target_batch_size, target_max_length)

  def fprop(
      self,
      inputs: JTensor,
      paddings: JTensor,
      attention_mask: JTensor,
      cross_inputs: Optional[JTensor] = None,
      cross_attention_mask: Optional[JTensor] = None,
      segment_pos: Optional[JTensor] = None,
  ) -> Tuple[JTensor, JTensor]:
    """Transformer decoder layer.

    Args:
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
      inputs_normalized = self.pre_layer_norm.fprop(inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm.fprop(inputs)
    else:
      inputs_normalized = inputs

    # Compute self-attention, key/value vectors are the input itself
    atten_output, self_atten_probs = self.self_attention.fprop(
        inputs_normalized,
        inputs_normalized,
        inputs_normalized,
        atten_mask=attention_mask,
        query_segment_pos=segment_pos)
    atten_probs = NestedMap(self_atten=self_atten_probs)
    if p.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm.fprop(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout.fprop(atten_output)

    # Apply skip connection
    if p.residual_droppath_prob > 0.0:
      atten_output = self.residual_droppath.fprop(inputs, atten_output)
    else:
      atten_output += inputs

    # Apply cross attention if applicable
    if self.params.cross_attention:
      assert cross_inputs is not None
      assert cross_attention_mask is not None
      # TODO(davidso): integrate primer_hybrid normalization with
      #.               cross_attention.
      assert p.norm_policy != 'primer_hybrid'

      cross_atten_output, cross_atten_probs = self.cross_attention.fprop(
          self.cross_layer_norm.fprop(atten_output),
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)
      atten_probs.cross_atten = cross_atten_probs

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout.fprop(cross_atten_output)

      if p.residual_droppath_prob > 0.0:
        atten_output = self.residual_droppath.fprop(atten_output,
                                                    cross_atten_output)
      else:
        atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer.fprop(atten_output, paddings=paddings)
    return output, atten_probs  # pytype: disable=bad-return-type  # jax-ndarray

  def extend_step(
      self,
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
      inputs_normalized = self.pre_layer_norm.fprop(inputs)
    elif p.norm_policy == 'pre':
      inputs_normalized = self.layer_norm.fprop(inputs)

    # Self-attention layer.
    updated_states, atten_output = self.self_attention.extend_step(
        cached_states,
        inputs_normalized,
        atten_mask=attention_mask,
        time_step=time_step)
    if p.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm.fprop(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout.fprop(atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if self.params.cross_attention:
      assert cross_inputs is not None
      assert cross_attention_mask is not None

      atten_output_normalized = self.cross_layer_norm.fprop(
          jnp.expand_dims(atten_output, axis=1))
      cross_atten_output, _ = self.cross_attention.fprop(
          atten_output_normalized,
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout.fprop(cross_atten_output)
      # Squeeze sequence dim
      cross_atten_output = jnp.squeeze(cross_atten_output, axis=1)
      atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer.fprop(atten_output)
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
                 e_dim=None,
                 combine_qkv=False):
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
      combine_qkv: if combined qkv projection layer is used.

    Returns:
      A Params object to set up a StackedTransformer.
    """

    p = cls.Params()
    p.name = name
    p.packed_input = True
    p.moe_layers = [0] if moe else None
    p.num_layers = 2 if moe else 1
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
    assert tr_atten_tpl.proj_tpl.cls == attentions.AttentionProjection
    tr_atten_tpl.proj_tpl.attention_combine_dims = True
    tr_atten_tpl.relative_bias_tpl = attentions.RelativeBias.Params().Set(
        relative_attention_num_buckets=relative_attention_num_buckets,
        relative_attention_max_distance=relative_attention_max_distance)
    tr_atten_tpl.output_proj_use_nhd_shape = True
    if combine_qkv:
      tr_atten_tpl.combine_qkv = True
      tr_atten_tpl.combined_qkv_proj_tpl.use_bias = False
      tr_atten_tpl.combined_qkv_proj_tpl.attention_combine_dims = True
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

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('cross_attention', False,
             'If set, introduces cross encoder-decoder attention layer.')
    p.Define('mask_self_attention', False, 'Use masked self-attention.')
    p.Define('num_layers', 0, 'Num of layers in this stack.')
    p.Define('model_dims', 0, 'Model dimension in Transformer layers.')
    p.Define('hidden_dims', 0,
             'The hidden layer dimension of FFN in Transformer layers.')
    p.Define('num_heads', 0, 'Number of attention heads.')
    p.Define(
        'dim_per_head', None, 'Dimension of each attention head. If None then '
        'dim_per_head == hidden_dim // num_heads.')
    p.Define('dropout_prob', 0.0,
             'Apply dropout at this prob at various places.')
    p.Define('residual_droppath_prob', 0.0,
             'Probability at which we drop the entire residual path.')
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

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    assert p.num_layers > 0
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

      if p.residual_droppath_prob > 0.0:
        p_i.residual_droppath_prob = (
            p.residual_droppath_prob * i / max(1, p.num_layers))

      if p.moe_layers and i in p.moe_layers:
        assert p.num_experts > 0
        moe_p = p.moe_layer_tpl.Copy()
        moe_p.num_experts = p.num_experts
        moe_p.num_groups = p.num_groups
        moe_p.min_group_size = p.min_group_size
        p_i.tr_fflayer_tpl = moe_p
      return p_i

    layer_params = [_layer_params(i) for i in range(p.num_layers)]
    self.create_children('x_layers', layer_params)

  def init_states(self, *args: Any, **kwargs: Any) -> NestedMap:
    """Initialize the cache for the StackedTransformer layer.

    Args:
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      Initialized cache for decoding.
    """
    return NestedMap(x_layers=[
        layer.init_states(*args, **kwargs) for layer in self.x_layers
    ])

  def fprop(self,
            inputs: JTensor,
            paddings: JTensor,
            segment_mask: Optional[JTensor] = None,
            cross_inputs: Optional[JTensor] = None,
            cross_paddings: Optional[JTensor] = None,
            cross_segment_mask: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> JTensor:
    """Stacked Transformer layer.

    Args:
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

    for i in range(p.num_layers):
      x_in = x_out
      x_out, _ = self.x_layers[i].fprop(
          x_in,
          paddings,
          attention_mask,
          cross_inputs,
          cross_attention_mask,
          segment_pos=segment_pos)
    return x_out

  def extend_step(
      self,
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
    for layer, layer_states in zip(self.x_layers, cached_states.x_layers):
      updated_layer_states, decoder_output = layer.extend_step(
          layer_states,
          decoder_input,
          time_step=time_step,
          attention_mask=attention_mask,
          cross_inputs=cross_inputs,
          cross_attention_mask=cross_attention_mask)
      updated_states.x_layers.append(updated_layer_states)
      decoder_input = decoder_output
    return updated_states, decoder_output  # pytype: disable=bad-return-type  # jax-ndarray


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
            inputs: JTensor,
            paddings: JTensor,
            segment_mask: Optional[JTensor] = None,
            cross_inputs: Optional[JTensor] = None,
            cross_paddings: Optional[JTensor] = None,
            cross_segment_mask: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> JTensor:
    """Stacked Transformer layer.

    Args:
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

    def sub_fprop_fn(sub, inputs, *args: Any, **kwargs: Any):
      with py_utils.AuxLossContext(reentrant=True) as al_ctx:
        assert al_ctx is not None
        # sub is expected to be an instance of StackedTransformer
        assert isinstance(sub, StackedTransformer)
        out = sub.fprop(inputs, *args, **kwargs)

        if al_ctx.aux_losses:
          assert isinstance(al_ctx.aux_losses, list)
          aux_loss_inc = sum(al_ctx.aux_losses).astype(self.fprop_dtype)
          base_layer.add_summary('aux_loss_inc', aux_loss_inc)
        else:
          aux_loss_inc = jnp.array(0.0, dtype=self.fprop_dtype)

        return out, NestedMap(aux_loss=aux_loss_inc)

    out, stacked_extra = self.repeat.fprop(sub_fprop_fn, inputs, paddings,
                                           segment_mask, cross_inputs,
                                           cross_paddings, cross_segment_mask,
                                           segment_pos)
    aux_loss = jnp.sum(stacked_extra.aux_loss)
    aux_loss_ctx = py_utils.AuxLossContext.Current()
    # Scan can not have sideeffects so we have to capture side effect
    # "aux_loss" in the moe layer and propagate it explicitly.
    if aux_loss_ctx is not None:
      aux_loss_ctx.AddLoss(aux_loss)
    return out

  def init_states(self, *args: Any, **kwargs: Any) -> NestedMap:
    """Initialize the cache for the StackedTransformerRepeated layer.

    Args:
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      Initialized cache for decoding.
    """

    def init_fn(block, *args, **kwargs):
      assert isinstance(block, StackedTransformer)
      return block.init_states(*args, **kwargs)

    return self.repeat.init_states(init_fn, *args, **kwargs)

  def extend_step(
      self,
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

    def extend_fn(sub, cached_states, step_inputs, *args, **kwargs):
      assert isinstance(sub, StackedTransformer)
      return sub.extend_step(cached_states, step_inputs, *args, **kwargs)

    return self.repeat.extend_step(
        extend_fn,
        cached_states,
        inputs,
        time_step=time_step,
        segment_mask=segment_mask,
        cross_inputs=cross_inputs,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask)


class PipelinedTransformer(base_layer.BaseLayer):
  """A pipelined Transformer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('pipeline_stage', StackedTransformerRepeated.Params(),
             'The layer params of each stage.')
    p.Define('num_pipeline_stages', None, 'Number of pipeline stages.')
    p.Define('num_pipeline_microbatches', None,
             'Number of pipeline microbatches.')
    p.Define('pipeline_microbatch_size', None,
             'Size of each pipeline microbatch.')
    wp = p.weight_split_dims_mapping
    wp.Define('stages', [-1], 'How the num_stages dimension should be sharded.')
    ap = p.activation_split_dims_mapping
    ap.Define('final_out', None, 'How the final output should be sharded.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    assert p.num_pipeline_stages > 0

    stage_params = p.pipeline_stage.Copy()
    # Must use deterministic dropout in pipelined layers.
    pipeline_params = pipeline.LayerwiseShardablePipelined.Params().Set(
        name=p.name,
        num_stages=p.num_pipeline_stages,
        single_stage_body=stage_params,
        num_microbatches=p.num_pipeline_microbatches,
        microbatch_size=p.pipeline_microbatch_size,
        unpack_summaries=True)
    pipeline_params.weight_split_dims_mapping.stages = (
        p.weight_split_dims_mapping.stages)
    self.create_child('pipeline', pipeline_params)

  def fprop(self,
            inputs: JTensor,
            paddings: JTensor,
            segment_mask: Optional[JTensor] = None,
            cross_inputs: Optional[JTensor] = None,
            cross_paddings: Optional[JTensor] = None,
            cross_segment_mask: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> JTensor:
    """Pipelined Transformer layer.

    Args:
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
    p = self.params
    if p.pipeline_stage.cls == StackedTransformer:
      xformer_layer_p = p.pipeline_stage.transformer_layer_params_tpl
    else:
      assert p.pipeline_stage.cls == StackedTransformerRepeated
      xformer_layer_p = p.pipeline_stage.block.transformer_layer_params_tpl
    bld_mapping = xformer_layer_p.tr_atten_tpl.activation_split_dims_mapping.bld
    # Annotate the inputs before the pipeline to prevent unexpected propagation
    # from earlier layers.
    inputs = base_layer.maybe_shard(inputs, bld_mapping, p.mesh_axis_names)
    if bld_mapping is not None:
      # Annotate other inputs.
      paddings = base_layer.maybe_shard(paddings, bld_mapping[:-1],
                                        p.mesh_axis_names)
      # For cross inputs, we only specify the batch dim sharding.
      if segment_mask is not None:
        segment_mask = base_layer.maybe_shard(
            segment_mask, [bld_mapping[0], -1, -1, -1],
            p.mesh_axis_names,
            unconstrained_dims=[1, 2, 3])
      if cross_inputs is not None:
        cross_inputs = base_layer.maybe_shard(
            cross_inputs, [bld_mapping[0], -1, -1],
            p.mesh_axis_names,
            unconstrained_dims=[1, 2])
      if cross_paddings is not None:
        cross_paddings = base_layer.maybe_shard(
            cross_paddings, [bld_mapping[0], -1],
            p.mesh_axis_names,
            unconstrained_dims=[1])
      if cross_segment_mask is not None:
        cross_segment_mask = base_layer.maybe_shard(
            cross_segment_mask, [bld_mapping[0], -1, -1, -1],
            p.mesh_axis_names,
            unconstrained_dims=[1, 2, 3])
      if segment_pos is not None:
        segment_pos = base_layer.maybe_shard(segment_pos, bld_mapping[:-1],
                                             p.mesh_axis_names)
    outputs = self.pipeline.fprop(
        inputs,
        paddings,
        segment_mask=segment_mask,
        cross_inputs=cross_inputs,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask,
        segment_pos=segment_pos)
    # Annotate the output to match input sharding.
    outputs = base_layer.maybe_shard(outputs, bld_mapping, p.mesh_axis_names)
    # Re-annotate the final output.
    outputs = base_layer.maybe_shard(outputs,
                                     p.activation_split_dims_mapping.final_out,
                                     p.mesh_axis_names)
    return outputs

  def init_states(self, *args, **kwargs) -> NestedMap:
    raise NotImplementedError(type(self))

  def extend_step(
      self,
      cached_states: NestedMap,
      inputs: JTensor,
      *,
      time_step: JTensor,
      segment_mask: Optional[JTensor] = None,
      cross_inputs: Optional[JTensor] = None,
      cross_paddings: Optional[JTensor] = None,
      cross_segment_mask: Optional[JTensor] = None
  ) -> Tuple[JTensor, NestedMap]:
    raise NotImplementedError(type(self))
