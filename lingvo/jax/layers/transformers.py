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
from lingvo.jax.layers import stochastics
import numpy as np
import tensorflow.compat.v2 as tf

CreateLayerVariablesStatus = base_layer.CreateLayerVariablesStatus

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
WeightParams = py_utils.WeightParams

InstantiableParams = py_utils.InstantiableParams
AuxLossContext = py_utils.AuxLossContext
JTensor = pytypes.JTensor


def ComputeAttentionMasksForFProp(
    inputs: JTensor,
    paddings: Optional[JTensor] = None,
    causal_attention: Optional[bool] = False,
    segment_mask: Optional[JTensor] = None,
    cross_inputs: Optional[JTensor] = None,
    cross_paddings: Optional[JTensor] = None,
    cross_segment_mask: Optional[JTensor] = None,
    fold_padding_with_segment_mask: Optional[bool] = False,
) -> Tuple[JTensor, Union[JTensor, None]]:
  """Computes attention mask from paddings, segment masks etc for FProp.

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
    attention_mask = attentions.ConvertPaddingsToMask(paddings, inputs.dtype)

    # Additional segment_mask may also be provided in this case
    if segment_mask is not None:
      attention_mask = jnp.minimum(attention_mask, segment_mask)

  # Causal mask of shape [1, 1, T, T]
  if causal_attention:
    causal_mask = attentions.CausalMask(inputs)
    attention_mask = jnp.minimum(attention_mask, causal_mask)

  # Compute cross attention mask if applicable
  cross_attention_mask = None
  if cross_inputs is not None:
    assert cross_paddings is not None

    # Compute paddings
    cross_attention_mask = attentions.ConvertPaddingsToMask(
        cross_paddings, dtype=cross_inputs.dtype)

    # Packed inputs
    if cross_segment_mask is not None:
      cross_attention_mask = jnp.minimum(cross_attention_mask,
                                         cross_segment_mask)
  return attention_mask, cross_attention_mask


def ComputeAttentionMasksForExtendStep(
    time_step: JTensor,
    seq_len: int,
    segment_mask: Optional[JTensor] = None,
    cross_inputs: Optional[JTensor] = None,
    cross_paddings: Optional[JTensor] = None,
    cross_segment_mask: Optional[JTensor] = None
) -> Tuple[JTensor, Union[JTensor, None]]:
  """Computes attention mask from paddings, segment masks etc for ExtendStep.

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
      attentions.ConvertPaddingsToMask(causal_padding), axis=1)

  # Include segment mask, has shape [B, 1, T]
  if segment_mask is not None:
    attention_mask = jnp.minimum(attention_mask, segment_mask)

  # Compute cross attention mask if applicable
  cross_attention_mask = None
  if cross_inputs is not None:
    assert cross_paddings is not None

    # Compute paddings mask [B, 1, 1, S]
    cross_attention_mask = attentions.ConvertPaddingsToMask(
        cross_paddings, dtype=cross_inputs.dtype)

    # Cross segment mask may be overloaded
    if cross_segment_mask is not None:
      # [B, 1, S] -> [B, 1, 1, S]
      cross_segment_mask = jnp.expand_dims(cross_segment_mask, axis=1)
      cross_attention_mask = jnp.minimum(cross_attention_mask,
                                         cross_segment_mask)
  return attention_mask, cross_attention_mask


class TransformerFeedForwardLayer(base_layer.BaseLayer):
  """Transformer feedforward layer with residual connection and dropout."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('hidden_dims', 0, 'Hidden dimension of FFN')
    p.Define(
        'activation', 'RELU', 'Activation function to use.'
        'Options are RELU, RELU6, RELU^2, RELU^3, SIGMOID, TANH, GELU,'
        'GATED_SILU, NONE.')
    p.Define('fflayer_tpl', linears.FeedForwardLayer.Params(),
             'Feedforward layer params')
    p.Define('ln_tpl', normalizations.LayerNorm.Params(), 'Layer norm params')
    p.Define('residual_dropout_prob', 0., 'Residual dropout')
    p.Define(
        'relu_dropout_tpl', stochastics.DropoutLayer.Params(),
        'Relu dropout params template. keep_prop will be reset to '
        '(1.0 - relu_dropout_prob).')
    p.Define('relu_dropout_prob', 0., 'FFN dropout')
    p.Define(
        'residual_dropout_tpl', stochastics.DropoutLayer.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')
    p.Define('add_skip_connection', True, 'Whether to add residual connection')
    p.Define('pre_layer_norm', True, 'Whether to apply LN pre or post')
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
    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping
    # Create Layer Norm
    ln_p = p.ln_tpl.Copy()
    ln_p.name = 'fflayer_ln'
    ln_p.input_dims = p.input_dims
    self.CreateChild('layer_norm', ln_p)

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
    ffn1_p.activation = activation
    ffn1_p.output_dims = p.hidden_dims
    ffn1_p.weight_split_dims_mapping.wt = wp.ffn0
    ffn1_p.activation_split_dims_mapping.out = ap.ffn0
    self.CreateChild('ffn_layer1', ffn1_p)

    if self._is_ffn1_gated:
      # This is a gated ffw network.
      gate_p = p.fflayer_tpl.Copy()
      gate_p.name = 'ffn_layer1_gate'
      gate_p.input_dims = p.input_dims
      gate_p.activation = gate_activation
      gate_p.output_dims = p.hidden_dims
      gate_p.weight_split_dims_mapping.wt = wp.ffn0
      gate_p.activation_split_dims_mapping.out = ap.ffn0
      self.CreateChild('ffn_layer1_gate', gate_p)

    # Create RELU dropout layer
    relu_dropout_p = p.relu_dropout_tpl.Copy()
    relu_dropout_p.keep_prob = 1.0 - p.relu_dropout_prob
    self.CreateChild('relu_dropout', relu_dropout_p)

    # Create the second Feedforward layer mapping to input dims
    ffn2_p = p.fflayer_tpl.Copy()
    ffn2_p.name = 'ffn_layer2'
    ffn2_p.input_dims = p.hidden_dims
    ffn2_p.activation = 'NONE'
    ffn2_p.output_dims = p.input_dims
    ffn2_p.weight_split_dims_mapping.wt = wp.ffn1
    ffn2_p.activation_split_dims_mapping.out = ap.ffn1
    self.CreateChild('ffn_layer2', ffn2_p)

    # Create residual dropout layer
    residual_dropout_p = p.residual_dropout_tpl.Copy()
    residual_dropout_p.keep_prob = 1.0 - p.residual_dropout_prob
    self.CreateChild('residual_dropout', residual_dropout_p)

  def FProp(self,
            theta: NestedMap,
            inputs: JTensor,
            paddings: Optional[JTensor] = None) -> JTensor:
    p = self.params
    if p.pre_layer_norm:
      inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
    else:
      inputs_normalized = inputs

    # Expand paddings to last dim if not None to have shape [batch, time, 1]
    if paddings is not None:
      paddings = jnp.expand_dims(paddings, axis=-1)

    # Apply first FFN layer
    if self._is_ffn1_gated:
      gate_value = self.ffn_layer1_gate.FProp(theta.ffn_layer1_gate,
                                              inputs_normalized)
      projected_inputs = gate_value * self.ffn_layer1.FProp(
          theta.ffn_layer1, inputs_normalized)
    else:
      projected_inputs = self.ffn_layer1.FProp(theta.ffn_layer1,
                                               inputs_normalized)
      projected_inputs = checkpoint_name(projected_inputs, 'ffn1')

    # Apply paddings if not None
    if paddings is not None:
      projected_inputs *= (1.0 - paddings)

    # Apply RELU dropout
    projected_inputs = self.relu_dropout.FProp(theta.relu_dropout,
                                               projected_inputs)

    # Apply second FFN layer
    projected_inputs = self.ffn_layer2.FProp(theta.ffn_layer2, projected_inputs)
    projected_inputs = checkpoint_name(projected_inputs, 'ffn2')

    # Apply paddings if not None
    if paddings is not None:
      projected_inputs *= (1.0 - paddings)

    # Apply residual dropout
    projected_inputs = self.residual_dropout.FProp(theta.residual_dropout,
                                                   projected_inputs)

    # Apply skip connection
    if p.add_skip_connection:
      projected_inputs += inputs

    # Apply Layer norm if not applied
    if not p.pre_layer_norm:
      projected_inputs = self.layer_norm.FProp(theta.layer_norm,
                                               projected_inputs)
    return projected_inputs


class TransformerShardedMoeLayer(base_layer.BaseLayer):
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
    # NOTE(yonghui): layer-norm as used in gshard doesn't have bias and assume
    # the mean is 0.0. See gshard_builder._LN for more details.
    p.Define('ln_tpl', normalizations.LayerNorm.Params(),
             'Layer norm default params.')
    p.Define(
        'activation', 'RELU', 'Activation function to use.'
        'Options are RELU, RELU6, SIGMOID, TANH, NONE.')
    p.Define(
        'relu_dropout_tpl', stochastics.DropoutLayer.Params(),
        'Relu dropout params template. keep_prop will be reset to '
        '(1.0 - relu_dropout_prob).')
    p.Define(
        'relu_dropout_prob', 0.0,
        'Probability at which we apply dropout to the hidden layer of '
        'feedforward network.')
    p.Define(
        'residual_dropout_tpl', stochastics.DropoutLayer.Params(),
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
    p.Define('pre_layer_norm', True, 'Pre or post layer norm.')
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
        'expert_capacity_factor', 1.5,
        'Expert capacity_factor. This should be set to a value greater '
        'than or equal to 1.0. This is the ratio between max allowed '
        'examples per expert over the average number of examples per '
        'expert assuming routing is completely uniform.')
    p.Define(
        'expert_weight_shards', 1,
        'Shard each expert params into this many number of shards to '
        'reduce the size of individual weight params.')
    p.Define('second_expert_policy', 'all',
             'How to pick second expert: all, sampling or random.')

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

    assert p.expert_capacity_factor >= 1.0
    assert p.num_experts > 0
    assert p.num_groups > 0
    assert p.expert_weight_shards > 0

    params = p.ln_tpl.Copy()
    params.name = 'layer_norm'
    params.input_dims = p.input_dims
    self.CreateChild('layer_norm', params)

    dropout_tpl = p.residual_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.residual_dropout_prob)
    self.CreateChild('residual_dropout', dropout_tpl)

    dropout_tpl = p.relu_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.relu_dropout_prob)
    self.CreateChild('relu_dropout', dropout_tpl)

    if p.residual_droppath_prob > 0:
      assert p.add_skip_connection
      droppath_p = stochastics.StochasticResidualLayer.Params().Set(
          name='residual_droppath',
          survival_prob=1.0 - p.residual_droppath_prob)
      self.CreateChild('residual_droppath', droppath_p)

    act_p = activations_lib.ActivationLayer.Params().Set(
        activation=p.activation)
    self.CreateChild('activation', act_p)

  def CreateLayerVariables(self) -> None:
    super().CreateLayerVariables()
    p = self.params
    # Assume output_dims == input_dims
    output_dims = p.input_dims

    # First create the gating network.
    wp = p.weight_split_dims_mapping
    stddev = (1.0 / p.input_dims)**0.5
    gate_scale = stddev * 3.0**0.5
    gate_pc = WeightParams(
        shape=[p.input_dims, p.num_experts],
        init=WeightInit.Uniform(gate_scale),
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.me)
    self.CreateVariable('gate', gate_pc)

    # Next create the expert network.
    # Params initialization follows gshard_builder.py.
    # emh tensor typically mesh-shard on first dim and last dim. Hence, here we
    # split the tensor manually into multiple tensors on the second dim.
    emh_shape = [
        p.num_experts, p.input_dims // p.expert_weight_shards, p.hidden_dims
    ]
    stddev = (1.0 / p.input_dims)**0.5
    wi_init_scale = stddev * 3.0**0.5
    wi_pc = WeightParams(
        shape=emh_shape,
        init=WeightInit.Uniform(wi_init_scale),
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.emh)

    for ii in range(p.expert_weight_shards):
      self.CreateVariable('wi_%d' % ii, wi_pc)

    # EHM Tensor (output transformation after RELU)
    # ehm tensor typically shard on the first dim and the second dim. Here we
    # manually split the tensor on the last dim into multiple tensors.
    ehm_shape = [
        p.num_experts, p.hidden_dims, output_dims // p.expert_weight_shards
    ]
    stddev = (1.0 / p.hidden_dims)**0.5
    wo_init_scale = stddev * 3.0**0.5
    wo_pc = WeightParams(
        shape=ehm_shape,
        init=WeightInit.Uniform(wo_init_scale),
        dtype=p.dtype,
        device_mesh=p.device_mesh,
        tensor_split_dims_mapping=wp.ehm)

    for ii in range(p.expert_weight_shards):
      self.CreateVariable('wo_%d' % ii, wo_pc)

    # TODO(zhangqiaorjc): Possibly add bias variable.

  # TODO(zhangqiaorjc): Allow paddings to be optional?
  def FProp(self, theta: NestedMap, inputs: JTensor,
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

    # TODO(zhangqiaorjc): Handle input of shape [batch, seq_len, g, model/g]?
    if p.pre_layer_norm:
      inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
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

    def Split(t_in, sharding):
      return base_layer.MaybeShard(t_in, sharding, p.mesh_axis_names)

    reshaped_inputs = Split(reshaped_inputs, ap.gsm)
    reshaped_paddings = Split(reshaped_paddings, ap.gs)

    fprop_dtype = py_utils.FPropDtype(p)
    logits = jnp.einsum('gsm,me->gse', reshaped_inputs, theta.gate)

    # Here and below, we assume num devices equals num groups.
    # TODO(yonghui): Expose some of the options below through params.
    # NOTE(yonghui): The following code might break during beam search decode
    # due to much smaller group size.
    # TODO(yonghui): Avoid explicitly casting everything to fp32 once
    # Top2GatingOnLogits is stable in low-precision mode.
    aux_loss, combine_tensor, dispatch_tensor = gshard_utils.Top2GatingOnLogits(
        paddings=reshaped_paddings.astype(np.float32),
        logits=logits.astype(np.float32),
        experts_dim=p.num_experts,
        expert_capacity_dim=0,  # automatically decided.
        fprop_dtype=np.float32,
        prng_key=base_layer.NextPrngKey(),
        second_expert_policy=p.second_expert_policy,
        second_expert_threshold=0.0,
        # legacy_mtf_behavior=True doesn't normalize gates when one expert is
        # being dropped. This is more appropriate for routing decisions like
        # 'random'.
        legacy_mtf_behavior=True,
        # *2.0 because we choose top-2 experts per example
        capacity_factor=2.0 * p.expert_capacity_factor)

    if fprop_dtype != np.float32:
      aux_loss = aux_loss.astype(fprop_dtype)
      combine_tensor = combine_tensor.astype(fprop_dtype)
      dispatch_tensor = dispatch_tensor.astype(fprop_dtype)

    # both tensors have shape [g, s, e, c]
    combine_tensor = Split(combine_tensor, ap.gsec)
    dispatch_tensor = Split(dispatch_tensor, ap.gsec)

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
    expert_inputs = Split(expert_inputs, ap.egcm)

    hidden = jnp.einsum('egcm,emh->egch', expert_inputs, theta_wi)
    hidden = Split(hidden, ap.egch)

    # Activation function.
    hidden = self.activation.FProp(theta.activation, hidden)
    # Dropout.
    hidden = self.relu_dropout.FProp(theta.relu_dropout, hidden)
    # Output.
    expert_output = jnp.einsum('egch,ehm->egcm', hidden, theta_wo)
    expert_output = Split(expert_output, ap.egcm)
    # Now transpose and reshard.
    transposed_expert_output = jnp.einsum('egcm->gecm', expert_output)
    transposed_expert_output = Split(transposed_expert_output, ap.gecm)
    combined_output = jnp.einsum('gecm,gsec->gsm', transposed_expert_output,
                                 combine_tensor)
    combined_output = Split(combined_output, ap.gsm)

    combined_output = combined_output.reshape((bs, s_len, output_dims))
    # Apply padding.
    combined_output *= (1.0 - jnp.expand_dims(paddings, -1)).astype(fprop_dtype)
    # Residual dropout.
    after_residual = self.residual_dropout.FProp(theta.residual_dropout,
                                                 combined_output)
    if p.add_skip_connection:
      if p.residual_droppath_prob:
        out = self.residual_droppath.FProp(theta.residual_droppath, inputs,
                                           after_residual)
      else:
        out = inputs + after_residual * p.residual_weight

    if not p.pre_layer_norm:
      out = self.layer_norm.FProp(theta.layer_norm, out)

    # Add loss to a global collection. We don't return the loss to the caller
    # to avoid the change of the api here.
    aux_loss_ctx = py_utils.AuxLossContext.Current()
    if aux_loss_ctx is not None:
      aux_loss_ctx.AddLoss(aux_loss)

    return out


class TransformerLayer(base_layer.BaseLayer):
  """Transformer layer with multi-headed attention."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Dimension of the transformer block input.')
    p.Define('hidden_dims', 0, 'Hidden dimension of FFN layer.')
    p.Define('num_heads', None, 'Num of heads in self attention.')
    p.Define(
        'dropout_tpl', stochastics.DropoutLayer.Params(),
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
    p.Define('ln_tpl', normalizations.LayerNorm.Params(), 'Layer norm params.')
    p.Define('tr_atten_tpl',
             attentions.MultiHeadedAttention.Params().Set(),
             'Attention Layer params.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    p.Define('tr_fflayer_tpl', TransformerFeedForwardLayer.Params(),
             'Transformer Feed-Forward Layer params.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    # Initialize Layer Norm
    params = p.ln_tpl.Copy()
    params.name = 'layer_norm'
    params.input_dims = p.input_dims
    self.CreateChild('layer_norm', params)

    # Initialize multi-headed self-attention
    params = p.tr_atten_tpl.Copy()
    params.name = 'multihead_self_atten'
    params.input_dim = p.input_dims
    params.hidden_dim = p.input_dims
    params.num_heads = p.num_heads
    params.atten_dropout_prob = p.atten_dropout_prob
    self.CreateChild('self_attention', params)

    # Initialize residual dropout.
    params = p.dropout_tpl.Copy()
    params.keep_prob = (1.0 - p.residual_dropout_prob)
    self.CreateChild('residual_dropout', params)

    # Initialize multi-headed cross-attention
    if p.cross_attention:
      params = p.tr_atten_tpl.Copy()
      params.name = 'multihead_self_atten'
      params.input_dim = p.input_dims
      params.hidden_dim = p.input_dims
      params.num_heads = p.num_heads
      params.atten_dropout_prob = p.atten_dropout_prob
      self.CreateChild('cross_attention', params)

    # Initialize feed-forward layer
    params = p.tr_fflayer_tpl.Copy()
    params.name = 'tr_fflayer'
    params.input_dims = p.input_dims
    params.hidden_dims = p.hidden_dims
    params.relu_dropout_prob = p.relu_dropout_prob
    params.residual_dropout_prob = p.residual_dropout_prob
    self.CreateChild('ff_layer', params)

  def InitStates(self, theta: NestedMap, target_batch_size: int,
                 target_max_length: int) -> NestedMap:
    return self.self_attention.InitStates(theta.self_attention,
                                          target_batch_size, target_max_length)

  def FProp(
      self,
      theta: NestedMap,
      inputs: JTensor,
      paddings: JTensor,
      attention_mask: JTensor,
      cross_inputs: Optional[JTensor] = None,
      cross_attention_mask: Optional[JTensor] = None
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

    Returns:
      The fflayer output with shape [B, T, D].
      atten_probs: A NestedMap with keys `self_atten` <float>[B, N, T, T].
    """
    # Layer normalize input
    inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)

    # Compute self-attention, key/value vectors are the input itself
    atten_output, self_atten_probs = self.self_attention.FProp(
        theta.self_attention,
        inputs_normalized,
        inputs_normalized,
        inputs_normalized,
        atten_mask=attention_mask)
    atten_probs = NestedMap(self_atten=self_atten_probs)
    # Residual dropout and connection
    atten_output = self.residual_dropout.FProp(theta.residual_dropout,
                                               atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if self.params.cross_attention:
      assert cross_inputs is not None
      assert cross_attention_mask is not None

      cross_atten_output, cross_atten_probs = self.cross_attention.FProp(
          theta.cross_attention,
          self.layer_norm.FProp(theta.layer_norm, atten_output),
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)
      atten_probs.cross_atten = cross_atten_probs

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout.FProp(theta.residual_dropout,
                                                       cross_atten_output)
      atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer.FProp(
        theta.ff_layer, atten_output, paddings=paddings)
    return output, atten_probs

  def ExtendStep(
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
      raise ValueError('ExtendStep should only be called with causal masking.')

    # Layer normalize input
    inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)

    # Self-attention layer.
    updated_states, atten_output = self.self_attention.ExtendStep(
        theta.self_attention,
        cached_states,
        inputs_normalized,
        atten_mask=attention_mask,
        time_step=time_step)

    # Residual dropout and connection
    atten_output = self.residual_dropout.FProp(theta.residual_dropout,
                                               atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if self.params.cross_attention:
      assert cross_inputs is not None
      assert cross_attention_mask is not None

      atten_output_normalized = self.layer_norm.FProp(
          theta.layer_norm, jnp.expand_dims(atten_output, axis=1))
      cross_atten_output, _ = self.cross_attention.FProp(
          theta.cross_attention,
          atten_output_normalized,
          cross_inputs,
          cross_inputs,
          atten_mask=cross_attention_mask)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout.FProp(theta.residual_dropout,
                                                       cross_atten_output)
      # Squeeze sequence dim
      cross_atten_output = jnp.squeeze(cross_atten_output, axis=1)
      atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer.FProp(theta.ff_layer, atten_output)
    return updated_states, output


class StackedTransformerLayers(base_layer.BaseLayer):
  """A stack of Transformer layers."""

  @staticmethod
  def DefineParams(p):
    p.Define('cross_attention', False,
             'If set, introduces cross encoder-decoder attention layer.')
    p.Define('mask_self_attention', False, 'Use masked self-attention.')
    p.Define('num_layers', 0, 'Num of layers in this stack.')
    p.Define('model_dims', 0, 'Model dimension in Transformer layers.')
    p.Define('hidden_dims', 0,
             'The hidden layer dimension of FFN in Transformer layers.')
    p.Define('num_heads', 0, 'Number of attention heads.')
    p.Define('dropout_prob', 0.0,
             'Apply dropout at this prob at various places.')
    p.Define(
        'transformer_layer_params_tpl', TransformerLayer.Params(),
        'A template of TransformerLayer.params, can be a list of params '
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
    p.Define('moe_layer_tpl', TransformerShardedMoeLayer.Params(),
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
    p = StackedTransformerLayers.DefineParams(p)
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    assert p.num_layers > 0
    assert p.model_dims > 0
    assert p.hidden_dims > 0
    assert p.num_heads > 0
    assert 0.0 <= p.dropout_prob < 1.0

    def _MoeLayerParams(ff_p):
      """Convert a TransformerFeedforwardLayer to a MoE Layer."""
      assert issubclass(ff_p.cls, TransformerFeedForwardLayer)
      p = self.params
      assert p.num_experts > 0
      moe_p = p.moe_layer_tpl.Copy()
      # Copy over the base params.
      base_layer.BaseLayer.CopyBaseParams(ff_p, moe_p)
      # Copy over othe params.
      moe_p.name = ff_p.name
      moe_p.input_dims = ff_p.input_dims
      moe_p.hidden_dims = ff_p.hidden_dims
      moe_p.ln_tpl = ff_p.ln_tpl.Copy()
      moe_p.activation = ff_p.activation
      moe_p.relu_dropout_tpl = ff_p.relu_dropout_tpl.Copy()
      moe_p.relu_dropout_prob = ff_p.relu_dropout_prob
      moe_p.residual_dropout_tpl = ff_p.residual_dropout_tpl.Copy()
      moe_p.residual_dropout_prob = ff_p.residual_dropout_prob
      moe_p.add_skip_connection = ff_p.add_skip_connection
      moe_p.pre_layer_norm = ff_p.pre_layer_norm
      moe_p.num_experts = p.num_experts
      moe_p.num_groups = p.num_groups
      moe_p.min_group_size = p.min_group_size
      return moe_p

    def _LayerParams(i):
      """Construct i-th layer params."""
      p_i = p.transformer_layer_params_tpl.Copy()
      p_i.name = f'layer_{i}'
      p_i.cross_attention = p.cross_attention
      p_i.mask_self_attention = p.mask_self_attention
      p_i.num_heads = p.num_heads
      p_i.input_dims = p.model_dims
      p_i.packed_input = p.packed_input
      p_i.atten_dropout_prob = p.dropout_prob
      p_i.residual_dropout_prob = p.dropout_prob
      p_i.relu_dropout_prob = p.dropout_prob
      p_i.hidden_dims = p.hidden_dims
      if i in p.moe_layers:
        moe_p = _MoeLayerParams(p_i.tr_fflayer_tpl)
        p_i.tr_fflayer_tpl = moe_p
      return p_i

    layer_params = [_LayerParams(i) for i in range(p.num_layers)]
    self.CreateChildren('x_layers', layer_params)

  def InitStates(self, theta: NestedMap, *args: Any,
                 **kwargs: Any) -> NestedMap:
    return NestedMap(x_layers=[
        layer.InitStates(layer_theta, *args, **kwargs)
        for layer, layer_theta in zip(self.x_layers, theta.x_layers)
    ])

  def FProp(self,
            theta: NestedMap,
            inputs: JTensor,
            paddings: JTensor,
            segment_mask: Optional[JTensor] = None,
            cross_inputs: Optional[JTensor] = None,
            cross_paddings: Optional[JTensor] = None,
            cross_segment_mask: Optional[JTensor] = None) -> JTensor:
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

    attention_mask, cross_attention_mask = ComputeAttentionMasksForFProp(
        inputs,
        paddings,
        p.mask_self_attention,
        segment_mask,
        cross_inputs,
        cross_paddings,
        cross_segment_mask,
        fold_padding_with_segment_mask=p.fold_padding_with_segment_mask)

    if p.enable_while_loop:

      def _StackVars(*args):
        args = [x[jnp.newaxis, :] for x in args]
        return jnp.vstack(args)

      stacked_vars = tf.nest.map_structure(_StackVars, *theta.x_layers)
      carry = py_utils.NestedMap(x_in=x_out)

      def _ScanFn(carry, layer_vars):
        # TODO(b/199950567): Sharding propagation does not seem to handle scan
        # boundary well. We need to annotate all parameters from within the scan
        # body even though we already pass them to pjit invocation outside the
        # scan at the top level once. Consider removing after bug fix.
        def AnnotateVarShardingConstraint(x, weight_param):
          partition_spec = base_layer.VarPartitionSpecs(
              weight_param,
              device_mesh=p.device_mesh,
              device_axis_names=p.mesh_axis_names)
          return base_layer.WithShardingConstraint(x, partition_spec)

        if p.device_mesh is not None:
          assert p.mesh_axis_names is not None
          layer_vars = tf.nest.map_structure(AnnotateVarShardingConstraint,
                                             layer_vars, self.x_layers[0].vars)
        x_out, _ = self.x_layers[0].FProp(layer_vars, carry.x_in, paddings,
                                          attention_mask, cross_inputs,
                                          cross_attention_mask)
        return py_utils.NestedMap(x_in=x_out), py_utils.NestedMap()

      carry_final, _ = recurrent.scan(
          carry,
          stacked_vars,
          _ScanFn,
          root_layer=self,
          checkpoint_policy=p.checkpoint_policy)
      x_out = carry_final.x_in
    else:
      for i in range(p.num_layers):
        x_in = x_out
        x_out, _ = self.x_layers[i].FProp(theta.x_layers[i], x_in, paddings,
                                          attention_mask, cross_inputs,
                                          cross_attention_mask)
    return x_out

  def ExtendStep(
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
      raise ValueError('ExtendStep should only be used with masked attention')

    if 'key' in cached_states.x_layers[0]:
      key = cached_states.x_layers[0].key
      max_t = key.shape[0]
    else:
      raise ValueError('Must call InitStates before ExtendStep')

    if p.cross_attention:
      assert cross_inputs is not None
      assert cross_paddings is not None

    attention_mask, cross_attention_mask = ComputeAttentionMasksForExtendStep(
        time_step, max_t, segment_mask, cross_inputs, cross_paddings,
        cross_segment_mask)

    updated_states = NestedMap(x_layers=[])
    decoder_input = inputs
    for layer, layer_theta, layer_states in zip(self.x_layers, theta.x_layers,
                                                cached_states.x_layers):
      updated_layer_states, decoder_output = layer.ExtendStep(
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


class StackedTransformerLayersRepeated(base_layer.BaseLayer):
  """A StackedTransformerLayer implemented using the generic RepeatLayer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    # Share the same params as the StackedTransformerLayers.
    p = StackedTransformerLayers.DefineParams(p)
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    assert p.num_layers > 0
    assert p.model_dims > 0
    assert p.hidden_dims > 0
    assert p.num_heads > 0
    assert 0.0 <= p.dropout_prob < 1.0

    def _SubParams():
      """Construct i-th layer params."""
      sub_p = p.transformer_layer_params_tpl.Copy()
      sub_p.name = 'sub'
      sub_p.cross_attention = p.cross_attention
      sub_p.mask_self_attention = p.mask_self_attention
      sub_p.num_heads = p.num_heads
      sub_p.input_dims = p.model_dims
      sub_p.packed_input = p.packed_input
      sub_p.atten_dropout_prob = p.dropout_prob
      sub_p.residual_dropout_prob = p.dropout_prob
      sub_p.relu_dropout_prob = p.dropout_prob
      sub_p.hidden_dims = p.hidden_dims
      return sub_p

    repeat_l_params = repeats.RepeatLayer.Params().Set(
        sub=_SubParams(), x_times=p.num_layers)

    self.CreateChild('repeat', repeat_l_params)

  def FProp(self,
            theta: NestedMap,
            inputs: JTensor,
            paddings: JTensor,
            segment_mask: Optional[JTensor] = None,
            cross_inputs: Optional[JTensor] = None,
            cross_paddings: Optional[JTensor] = None,
            cross_segment_mask: Optional[JTensor] = None) -> JTensor:
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

    attention_mask, cross_attention_mask = ComputeAttentionMasksForFProp(
        inputs,
        paddings,
        p.mask_self_attention,
        segment_mask,
        cross_inputs,
        cross_paddings,
        cross_segment_mask,
        fold_padding_with_segment_mask=p.fold_padding_with_segment_mask)

    x_out = self.repeat.FProp(theta.repeat, inputs, paddings, attention_mask,
                              cross_inputs, cross_attention_mask)
    return x_out

  def InitStates(self, theta: NestedMap, *args: Any,
                 **kwargs: Any) -> NestedMap:
    return self.repeat.InitStates(theta.repeat, *args, **kwargs)

  def ExtendStep(
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
      raise ValueError('ExtendStep should only be used with masked attention')

    if 'key' in cached_states:
      key = cached_states.key
      # key is of shape [num_layers, max_seq_length, batch_size, ...].
      max_t = key.shape[1]
    else:
      raise ValueError('Must call InitStates before ExtendStep')

    if p.cross_attention:
      assert cross_inputs is not None
      assert cross_paddings is not None

    attention_mask, cross_attention_mask = ComputeAttentionMasksForExtendStep(
        time_step, max_t, segment_mask, cross_inputs, cross_paddings,
        cross_segment_mask)

    updated_states, dec_out = self.repeat.ExtendStep(
        theta.repeat,
        cached_states,
        inputs,
        time_step=time_step,
        attention_mask=attention_mask,
        cross_inputs=cross_inputs,
        cross_attention_mask=cross_attention_mask)

    return updated_states, dec_out


class TransformerLm(base_layer.BaseLayer):
  """Packed Transformer LM with position embedding and shared softmax layer.

  This folds the padding with the segment mask when the inputs are not packed.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Parameterization of this model."""
    p = super().Params()
    p.Define('position_emb_tpl',
             embedding_softmax.PositionalEmbeddingLayer.Params(),
             'The Positional Embedding layer params.')
    p.Define('model_dims', 0, 'Model dimension in Transformer layers.')
    p.Define('hidden_dims', 0,
             'The hidden layer dimension of FFN in Transformer layers.')
    p.Define('num_layers', 0, 'The number of transformer layers.')
    p.Define('num_heads', 0,
             'The number of attention heads in transformer layers.')
    p.Define('stacked_transformer_tpl', StackedTransformerLayers.Params(),
             'StackedTransformerLayer params tpl.')
    p.Define(
        'softmax_tpl',
        embedding_softmax.SingleShardSharedEmbeddingSoftmax.Params(),
        'The softmax layer params. The softmax and embedding lookup'
        'share parameters in this case.')
    p.Define('vocab_size', 0, 'Size of the vocabulary for LM.')
    p.Define('packed_input', False, 'Whether the inputs are packed.')
    p.Define('aux_loss_weight', 0.0, 'Weight of the aux loss for MoE layers.')
    p.Define('masked_lm', False, 'Whether this is BERT style masked LM.')
    return p

  @classmethod
  def SetShardingParamsV1(cls,
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
    # Softmax weight is of shape [input_dim, vocab_size].
    softmax_p = lm_p.softmax_tpl
    softmax_p.weight_split_dims_mapping.wt = [data_axis, mdl_axis]
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
    softmax_p.lookup_style = 'matmul'
    xformer_p = lm_p.stacked_transformer_tpl.transformer_layer_params_tpl
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
    # TODO(zhangqiaorjc): Set weight_split_dims_mapping and
    # activation_split_dims_mapping for MoE layer.
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

    return lm_p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructor."""
    super().__init__(params)
    p = self.params

    # Positional embedding layer.
    params = p.position_emb_tpl.Copy()
    params.embedding_dims = p.model_dims
    self.CreateChild('position_emb', params)

    # Transformer layers
    params = p.stacked_transformer_tpl.Copy()
    params.num_layers = p.num_layers
    params.num_heads = p.num_heads
    params.model_dims = p.model_dims
    params.hidden_dims = p.hidden_dims
    if p.masked_lm:
      params.mask_self_attention = False
    else:
      params.mask_self_attention = True
    params.packed_input = p.packed_input
    params.fold_padding_with_segment_mask = True
    self.CreateChild('transformer', params)

    # Final layer norm
    params = normalizations.LayerNorm.Params().Set(input_dims=p.model_dims)
    self.CreateChild('final_ln', params)

    # Final softmax
    params = p.softmax_tpl.Copy()
    params.input_dims = p.model_dims
    params.num_classes = p.vocab_size
    self.CreateChild('softmax', params)

  def InitStates(self, theta: NestedMap, *args: Any,
                 **kwargs: Any) -> NestedMap:
    return NestedMap(
        step=jnp.array(0, dtype=jnp.uint32),
        transformer=self.transformer.InitStates(theta.transformer, *args,
                                                **kwargs))

  def ComputeLoss(self,
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
      logits = self.softmax.GetLogits(theta=theta.softmax, inputs=activations)
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
      xent_output = self.softmax.FProp(
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
        p = self.params
        if p.aux_loss_weight == 0.0:
          logging.warn('p.aux_loss_weight == 0 when there is aux_loss')
        aux_loss = p.aux_loss_weight * sum(aux_loss_tensors)
      else:
        aux_loss = 0.0
      if not isinstance(aux_loss, jnp.ndarray):
        aux_loss = jnp.array(aux_loss, dtype=self.fprop_dtype)
      xent_output.aux_loss = aux_loss
      # This is the loss to minimize.
      xent_output.total_loss = xent_output.avg_xent + xent_output.aux_loss
    return xent_output

  def FProp(self,
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
    with py_utils.AuxLossContext() as aux_loss_ctx:
      assert aux_loss_ctx is not None
      input_emb = self.softmax.EmbLookup(theta.softmax, inputs)
      batch, seq_length = inputs.shape
      if segment_ids is None:
        assert segment_pos is None
        # Fold the paddings with the segment mask
        segment_ids = jnp.asarray(1 - paddings, jnp.int32)
        segment_pos = jnp.tile(
            jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1])
      position_emb = self.position_emb.FProp(
          theta.position_emb, seq_length=seq_length, position=segment_pos)
      inputs = input_emb + position_emb
      if p.masked_lm:
        segment_mask = attentions.SegmentMask(segment_ids, segment_ids,
                                              inputs.dtype)
      else:
        segment_mask = attentions.CausalSegmentMask(segment_ids, inputs.dtype)
      output = self.transformer.FProp(
          theta.transformer, inputs, paddings, segment_mask=segment_mask)
      # Final layer norm
      output = self.final_ln.FProp(theta.final_ln, output)

      return self.ComputeLoss(theta, output, labels)

  def ExtendStep(
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
      inputs: Target sequence of shape [B] corresponding to target sequence at
        index time_step.

    Returns:
      cached_states: A `.NestedMap` object containing the updated states. The
        cached_states.step is incremented to the next time step, and
        cached_states.transformer is updated with the keys and values of the
        current time step.
      xent_output: A `.NestedMap` object containing the log probabilities and
        probabilities.
    """
    input_emb = self.softmax.EmbLookup(theta.softmax, inputs[:, jnp.newaxis])
    # During autoregressive decoding inputs are not packed
    time_step = cached_states.step
    segment_pos = jnp.zeros((inputs.shape[0], 1)) + time_step
    position_emb = self.position_emb.FProp(
        theta.position_emb, seq_length=1, position=segment_pos)
    inputs = input_emb + position_emb
    updated_cache, outputs = self.transformer.ExtendStep(
        theta.transformer,
        cached_states.transformer,
        inputs[:, 0, :],
        time_step=time_step)
    cached_states.transformer = updated_cache
    cached_states.step += 1
    outputs = self.final_ln.FProp(theta.final_ln, outputs)
    xent_output = self.ComputeLoss(theta, outputs)
    return cached_states, xent_output
