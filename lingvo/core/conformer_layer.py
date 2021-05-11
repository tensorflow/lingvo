# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Conformer layers as in https://arxiv.org/abs/2005.08100."""

from lingvo import compat as tf

from lingvo.core import activations
from lingvo.core import base_layer
from lingvo.core import batch_major_attention as attention_lib
from lingvo.core import bn_layers
from lingvo.core import conv_layers_with_time_padding
from lingvo.core import gshard_builder
from lingvo.core import gshard_utils
from lingvo.core import hyperparams as hparams_lib
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import recurrent


class LConvLayer(base_layer.BaseLayer):
  r"""Lightweight conv layer.

  architecture::

    input
    /   \
    |   ln(.)                   # input_dim
    |   fflayer(.)              # 2 * input_dim
    |    |
    |   glu(.)                  # input_dim
    |   depthwise_conv_1d(.)
    |   norm(.)
    |   act(.)
    |    |
    |   fflayer(.)
    |   dropout(.)
    \   /
      +
      |
    output
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', None, 'Input and (in fact,) output dimension.')
    p.Define('kernel_size', None, 'Kernel size of 1d deptwise conv.')
    p.Define('conv_activation', 'SWISH', 'Activation after normalization.')
    p.Define(
        'is_causal', False, 'Whether this is a causal layer.'
        'If set to true, use '
        'conv_layers_with_time_padding.CausalDepthwiseConv2DLayer for '
        '`depthwise_conv_tpl`.')
    p.Define(
        'glu_activation', 'NONE',
        'Activation in GLU. Check lingvo.core.activations._ACTIVATIONS for '
        'other options.')
    p.Define('dropout_prob', 0., 'Dropout probability.')

    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Input layer norm template.')
    p.Define('linear_start_tpl', layers.FCLayer.Params(), 'Linear start layer.')
    p.Define(
        'depthwise_conv_tpl',
        conv_layers_with_time_padding.DepthwiseConv2DLayer.Params(),
        'Depthwise conv template. For causal layer, use '
        'conv_layers_with_time_padding.CausalDepthwiseConv2DLayer.')
    p.Define('conv_norm_layer_tpl', bn_layers.BatchNormLayer.Params(),
             'Normalization layer after conv.')
    p.Define('linear_end_tpl', layers.FCLayer.Params(), 'Linear end layer.')
    p.Define('dropout_tpl', layers.DropoutLayer.Params(),
             'Residual dropout layer.')
    p.Define(
        'split_act_gated_linear_start', False,
        'Separate act and gated linear start to remove data formatting '
        'overheads')
    p.linear_start_tpl.Set(activation='NONE', has_bias=True)
    p.linear_end_tpl.Set(activation='NONE', has_bias=True)
    # SPMD partition related params.
    #
    # d - model_dim
    # f - ff_hidden_dim (here ff_hidden_dim has the same size as model_dim)
    # h - height
    # w - width
    # i - in_channels
    # m - channel_multiplier
    # b - batch_size
    # l - seq_len
    p.weight_split_dims_mapping = hparams_lib.Params()
    wp = p.weight_split_dims_mapping
    wp.Define(
        'df', None,
        'Mesh split for lconv linear start weight with the shape of '
        '[model_dim, ff_hidden_dim], the default hidden_dim is the same as '
        'the model_dim.')
    wp.Define(
        'hwim', None,
        'Mesh split for lconv depthwise conv weight with the shape of '
        '[height, width, in_channels, channel_multiplier]. Width and '
        'channel_multiplier are both 1 for the common use case.')
    wp.Define(
        'fd', None, 'Mesh split for lconv linear end weight with the shape of '
        '[ff_hidden_dim, model_dim], the default hidden_dim is the same as '
        'the model_dim.')
    p.activation_split_dims_mapping = hparams_lib.Params()
    ap = p.activation_split_dims_mapping
    ap.Define(
        'blf', None, 'Mesh split for lconv linear start activation and lconv '
        'depthwise conv after normalization with the shape of '
        '[batch_size, seq_len, ff_hidden_dim], the default hidden_dim is the '
        'same as model_dim.')
    ap.Define(
        'bld', None,
        'Mesh split for lconv linear end activation with the shape of '
        '[batch_size, seq_len, model_dim].')
    return p

  @classmethod
  def SetCanonicalShardingParams(cls, params):
    """Set up canonical SPMD sharding params."""
    assert params.device_mesh.ndim >= 2
    wp = params.weight_split_dims_mapping
    wp.df = [0, 1]
    wp.hwim = [-1, -1, 1, -1]
    # TODO(shibow/rpang): understand the effects of fd sharding, especially why
    # [-1, -1] performs better when bld is [0, -1, -1].
    wp.fd = [1, 0]
    ap = params.activation_split_dims_mapping
    ap.blf = [0, -1, 1]
    ap.bld = [1, -1, -1]

  @classmethod
  def CommonParams(cls,
                   input_dim=None,
                   kernel_size=None,
                   is_causal=False,
                   conv_activation='SWISH',
                   dropout_prob=0.):
    p = cls.Params().Set(
        input_dim=input_dim,
        is_causal=is_causal,
        kernel_size=kernel_size,
        conv_activation=conv_activation,
        dropout_prob=dropout_prob)
    if is_causal:
      p.depthwise_conv_tpl = (
          conv_layers_with_time_padding.CausalDepthwiseConv2DLayer.Params())
    return p

  @classmethod
  def SetFPropDtype(cls, p, fprop_dtype):
    p.fprop_dtype = fprop_dtype
    if fprop_dtype == tf.bfloat16 and not py_utils.use_tpu():
      # Depthwise conv supports bfloat16 only on TPUs.
      p.depthwise_conv_tpl.fprop_dtype = tf.float32
      if issubclass(p.conv_norm_layer_tpl.cls, bn_layers.BatchNormLayer):
        # Batch norm does not support bfloat16 on TPUs.
        p.conv_norm_layer_tpl.fprop_dtype = tf.float32
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    ln_p = p.ln_tpl.Copy().Set(name='ln', input_dim=p.input_dim)
    self.CreateChild('ln', ln_p)

    if p.split_act_gated_linear_start:
      linear_start_act_p = p.linear_start_tpl.Copy().Set(
          input_dim=p.input_dim,
          output_dim=p.input_dim,
          device_mesh=p.device_mesh,
          weight_split_dims_mapping=p.weight_split_dims_mapping.df,
          activation_split_dims_mapping=p.activation_split_dims_mapping.blf)
      linear_start_gated_p = p.linear_start_tpl.Copy().Set(
          input_dim=p.input_dim,
          output_dim=p.input_dim,
          device_mesh=p.device_mesh,
          weight_split_dims_mapping=p.weight_split_dims_mapping.df,
          activation_split_dims_mapping=p.activation_split_dims_mapping.blf)
      self.CreateChild('linear_start_act', linear_start_act_p)
      self.CreateChild('linear_start_gated', linear_start_gated_p)
    else:
      linear_start_p = p.linear_start_tpl.Copy().Set(
          name='linear_start',
          input_dim=p.input_dim,
          output_dim=2 * p.input_dim)
      self.CreateChild('linear_start', linear_start_p)

    linear_end_p = p.linear_end_tpl.Copy().Set(
        name='linear_end',
        input_dim=p.input_dim,
        output_dim=p.input_dim,
        device_mesh=p.device_mesh,
        weight_split_dims_mapping=p.weight_split_dims_mapping.fd,
        activation_split_dims_mapping=p.activation_split_dims_mapping.bld)
    self.CreateChild('linear_end', linear_end_p)

    if p.conv_norm_layer_tpl.cls == layers.LayerNorm:
      norm_p = p.conv_norm_layer_tpl.Copy().Set(
          name='norm_layer', input_dim=p.input_dim)
    else:
      norm_p = p.conv_norm_layer_tpl.Copy().Set(
          name='norm_layer', dim=p.input_dim)
    if p.conv_norm_layer_tpl.cls == bn_layers.GroupNormLayer:
      norm_p.cumulative = p.is_causal
    self.CreateChild('norm', norm_p)

    if (p.is_causal and p.depthwise_conv_tpl.cls ==
        conv_layers_with_time_padding.DepthwiseConv2DLayer):
      # If causal, switch to causal depthwise conv.
      depthwise_conv_p = (
          conv_layers_with_time_padding.CausalDepthwiseConv2DLayer.Params())
      hparams_lib.CopyFieldsTo(p.depthwise_conv_tpl, depthwise_conv_p)
    else:
      depthwise_conv_p = p.depthwise_conv_tpl.Copy()
    # 1d depthwise conv with channel_mulitplier = 1
    depthwise_conv_p.Set(
        name='depthwise_conv',
        filter_shape=(p.kernel_size, 1, p.input_dim, 1),
        filter_stride=(1, 1))
    self.CreateChild('depthwise_conv1d', depthwise_conv_p)

    dropout_p = p.dropout_tpl.Copy().Set(
        name='dropout', keep_prob=1. - p.dropout_prob)
    self.CreateChild('dropout', dropout_p)

  def _GLU(self, gated_inputs, act_inputs):
    p = self.params
    return self._ApplyActivation(act_inputs,
                                 p.glu_activation) * tf.sigmoid(gated_inputs)

  def _ApplyActivation(self, inputs, act_name):
    if act_name == 'NONE':
      return inputs
    return activations.GetFn(act_name)(inputs)

  def _Normalize(self, theta, inputs, paddings):
    """Applies normalization.

    Args:
      theta: A NestedMap of layer params.
      inputs: [b, t, 1, d].
      paddings: [b, t].

    Returns:
      A Tensor of shape [b, t, d].
    """
    if isinstance(self.norm, bn_layers.GroupNormLayer):
      assert self.norm.params.input_rank == 4
      inputs, _ = self.norm.FProp(theta.norm, inputs, paddings)
      # [b, t, d]
      inputs = tf.squeeze(inputs, 2)
    else:
      # [b, t, 1, d] -> [b, t, d]
      inputs = tf.squeeze(inputs, 2)
      if isinstance(self.norm, bn_layers.BatchNormLayer):
        inputs = self.norm.FProp(theta.norm, inputs, paddings)
      elif isinstance(self.norm, layers.LayerNorm):
        inputs = self.norm.FProp(theta.norm, inputs)
      else:
        raise NotImplementedError(
            'Only bn_layers.{BatchNormLayer,GroupNormLayer}, layers.LayerNorm '
            'are supported.')
    return self._CastToFPropDtype(inputs)

  def FProp(self, theta, inputs, paddings):
    """Builds FProp graph.

    Args:
      theta: A NestedMap of Tensors, see base class.
      inputs: A Tensor of shape [batch, seqlen, dim0].
      paddings: A Tensor of shape [batch, seqlen].

    Returns:
      output: A Tensor of shape [batch, seqlen, dim0].
      out_paddings: A Tensor of shape [batch, seqlen].
    """

    p = self.params
    with tf.name_scope(p.name):
      inputs, paddings = self._CastToFPropDtype((inputs, paddings))
      unnormalized_inputs = inputs

      inputs = self.ln.FProp(theta.ln, inputs)
      inputs = self._CastToFPropDtype(inputs)
      if p.split_act_gated_linear_start:
        act_inputs = self.linear_start_act.FProp(theta.linear_start_act, inputs)
        gated_inputs = self.linear_start_gated.FProp(theta.linear_start_gated,
                                                     inputs)
      else:
        inputs = self.linear_start.FProp(theta.linear_start, inputs)
        gated_inputs, act_inputs = tf.split(inputs, 2, axis=-1)
      inputs = self._GLU(gated_inputs, act_inputs)

      # TODO(jamesqin): inroduce depthwise conv2d with 3d inputs.
      # [b, t, d] --> [b, t, 1, d]
      inputs = tf.expand_dims(inputs, 2)
      adapted_blf_dims_mapping = None
      if p.activation_split_dims_mapping.blf is not None:
        adapted_blf_dims_mapping = p.activation_split_dims_mapping.blf.copy()
        adapted_blf_dims_mapping.insert(2, -1)
      inputs = gshard_utils.MeshSplit(inputs, p.device_mesh,
                                      adapted_blf_dims_mapping)
      theta.depthwise_conv1d.w = gshard_utils.MeshSplit(
          theta.depthwise_conv1d.w, p.device_mesh,
          p.weight_split_dims_mapping.hwim)
      if inputs.dtype == tf.bfloat16 and not py_utils.use_tpu():
        # Depthwise conv doesn't support bfloat32 on CPU.
        inputs = tf.cast(inputs, tf.float32)
        paddings = tf.cast(paddings, tf.float32)
      inputs, paddings = self.depthwise_conv1d.FProp(theta.depthwise_conv1d,
                                                     inputs, paddings)
      inputs, paddings = self._CastToFPropDtype((inputs, paddings))

      inputs = gshard_utils.MeshSplit(inputs, p.device_mesh,
                                      adapted_blf_dims_mapping)
      inputs = self._Normalize(theta, inputs, paddings)
      inputs = gshard_utils.MeshSplit(inputs, p.device_mesh,
                                      p.activation_split_dims_mapping.blf)

      inputs = self._ApplyActivation(inputs, p.conv_activation)

      inputs = self.linear_end.FProp(theta.linear_end, inputs)
      inputs = self.dropout.FProp(theta.dropout, inputs)

      output = inputs + unnormalized_inputs
      return output, paddings

  def zero_state(self, batch_size):
    p = self.params
    with tf.name_scope('zero_state'):
      if p.is_causal:
        with tf.name_scope('depthwise_conv1d'):
          res = py_utils.NestedMap(
              conv_state=self.depthwise_conv1d.zero_state(batch_size))
        if hasattr(self.norm, 'zero_state'):
          with tf.name_scope('norm'):
            res.norm_state = self.norm.zero_state(batch_size)
        return res
      else:
        # If not causal, depthwise_conv1d does not have zero_state().
        return py_utils.NestedMap()

  def _NormalizeStep(self, theta, inputs, paddings, state0, state1):
    if hasattr(self.norm, 'StreamStep'):
      # TODO(jamesqin): support 3d inputs.
      # At present it's guaranteed GroupNorm.
      assert (isinstance(self.norm, bn_layers.GroupNormLayer) and
              self.norm.params.input_rank == 4)
      inputs, paddings, norm_state1 = self.norm.StreamStep(
          theta.norm, inputs, paddings, state0.norm_state)
      # [b, t, d]
      inputs = tf.squeeze(inputs, 2)
      state1.norm_state = norm_state1
    else:
      # [b, t, 1, d] -> [b, t, d]
      inputs = tf.squeeze(inputs, 2)
      if isinstance(self.norm, layers.LayerNorm):
        inputs = self.norm.FProp(theta.norm, inputs)
      else:
        raise NotImplementedError(
            'Only bn_layers.GroupNormLayer, layers.LayerNorm are supported.')
    # [b, t, d]
    return inputs, paddings

  def StreamStep(self, theta, inputs, paddings, state0):
    """Streams t steps.

    Args:
      theta: A NestedMap of layer params.
      inputs: [b, t, d].
      paddings: A 0/1 valued tensor of shape [b, t].
      state0: A NestedMap of tensors of the same struct as returned by
        zero_state().

    Returns:
      outputs: A NestedMap of tensors consisting:
      padding: the same as input paddings.
      state1: A NestedMap of tensors of the same struct as state0.
    """
    p = self.params
    assert p.is_causal

    state1 = py_utils.NestedMap()
    with tf.name_scope(f'{p.name}/StreamStep'):
      unnormalized_inputs = inputs

      inputs = self.ln.FProp(theta.ln, inputs)
      if p.split_act_gated_linear_start:
        act_inputs = self.linear_start_act.FProp(theta.linear_start_act, inputs)
        gated_inputs = self.linear_start_gated.FProp(theta.linear_start_gated,
                                                     inputs)
      else:
        inputs = self.linear_start.FProp(theta.linear_start, inputs)
        gated_inputs, act_inputs = tf.split(inputs, 2, axis=-1)
      inputs = self._GLU(gated_inputs, act_inputs)

      # TODO(jamesqin): inroduce depthwise conv2d with 3d inputs.
      # TODO(jamesqin): optimize DepthwiseConv1D.StreamStep()
      # [b, t, d] --> [b, t, 1, d]
      inputs = tf.expand_dims(inputs, 2)
      # [b, t, 1, d]
      inputs, paddings, conv_state1 = self.depthwise_conv1d.StreamStep(
          theta.depthwise_conv1d, inputs, paddings, state0.conv_state)
      state1.conv_state = conv_state1
      # [b, t, d]
      inputs, paddings = self._NormalizeStep(theta, inputs, paddings, state0,
                                             state1)

      inputs = self._ApplyActivation(inputs, p.conv_activation)

      inputs = self.linear_end.FProp(theta.linear_end, inputs)
      inputs = self.dropout.FProp(theta.dropout, inputs)

      output = inputs + unnormalized_inputs
      return output, paddings, state1


def _AttenCtxIsSet(atten_context):
  return atten_context is not None and atten_context >= 0


def _GShardMoELayerParams(num_devices, num_groups, num_experts,
                          per_expert_capacity_dim):
  return gshard_builder.MoEBuilder.Params().Set(
      num_devices=num_devices,
      num_groups=num_groups,
      e_dim=num_experts,
      c_dim=per_expert_capacity_dim)


class ConformerLayer(base_layer.BaseLayer):
  """Conformer layer as in https://arxiv.org/abs/2005.08100.

    Canonical version (with default params.)
      x = x + 1/2 * FFN(x)
      x = x + MHSA(x)
      x = x + Lconv(x)
      x = x + 1/2 * FFN(x)
      y = ln(x)

    Optionally one can change the order of MHSA and conv.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', None, 'Input dimension.')
    p.Define(
        'is_causal', False, 'If use causal lconv and MHSA layer.'
        'Notice atten_right_context must be not be infinite(None) if is_causal '
        'is True. It is important to always set is_causal for streaming case, '
        'and not expect to infer from atten_{left,right}_context.')
    # atten layer
    # TODO(rpang): consider removing the attention hparams since they overlap
    # with p.trans_atten_tpl.
    p.Define('atten_num_heads', None,
             'Num of heads in multi-head self-attention.')
    p.Define(
        'layer_order', 'mhsa_before_conv',
        'Only mhsa, conv, mhsa_before_conv or conv_before_mhsa are '
        'supported.')

    # lconv layer
    p.Define('kernel_size', None, 'Kernel size of 1d lightweight conv.')

    # fflayer
    p.Define('fflayer_hidden_dim', None,
             'Hidden dim of the fflayers (start and end).')
    p.Define('fflayer_activation', 'SWISH', 'fflayer activation.')
    p.Define('fflayer_residual_weight', 0.5, 'fflayer residual weight.')
    p.Define('dropout_prob', None, 'Signature dropout prob of inner componets.')

    # tpl
    p.Define(
        'fflayer_start_tpl',
        layers_with_attention.TransformerFeedForwardLayer.Params(),
        'Layer params for Feed forward layer at the beginning. Supports '
        'using gshard_builder.MoEBuilder.Params() as well wherein the '
        'MoE() will be used. If set to None, this layer is excluded.')
    p.Define('trans_atten_tpl',
             attention_lib.TransformerAttentionLayer.Params(),
             'Self attention layer params.')
    p.Define(
        'lconv_tpl', LConvLayer.Params(),
        'Convolution module params. If set to None, this layer is excluded.')
    p.Define(
        'fflayer_end_tpl',
        layers_with_attention.TransformerFeedForwardLayer.Params(),
        'Layer params for Feed forward layer at the end. Supports using '
        'gshard_builder.MoEBuilder.Params() as well wherein the MoE() '
        'will be used.')
    p.Define(
        'fflayer_weight_sharing', False,
        'If True, will ignore `fflayer_end_tpl`, and will make the fflayer_end '
        'layer as a weight-shared copy of the fflayer_start layer.')
    p.Define('final_ln_tpl', layers.LayerNorm.Params(), 'Final layer norm.')
    # https://b/167460492#comment16
    p.Define(
        'remat', False, 'If to rematerialize the layer. If true, '
        'intermediate tensors are not saved in FProp().')
    return p

  @classmethod
  def CommonParams(cls,
                   *,
                   input_dim=None,
                   is_causal=False,
                   atten_num_heads=None,
                   atten_local_context=None,
                   atten_left_context=None,
                   atten_right_context=None,
                   use_relative_atten=True,
                   kernel_size=None,
                   fflayer_hidden_dim=None,
                   fflayer_activation='SWISH',
                   fflayer_residual_weight=0.5,
                   layer_order='mhsa_before_conv',
                   dropout_prob=0.,
                   conv_norm_layer_tpl=None,
                   fprop_dtype=None,
                   use_moe_in_fflayer_start=False,
                   use_moe_in_fflayer_end=False,
                   moe_num_partitions=None,
                   moe_num_experts=None,
                   moe_num_groups=None,
                   moe_per_capacity_dim=None,
                   fflayer_start_tpl=None,
                   fflayer_end_tpl=None,
                   trans_atten_tpl=None,
                   lconv_tpl=None):
    assert all([input_dim, fflayer_hidden_dim])
    if layer_order != 'conv':
      assert atten_num_heads
    if layer_order == 'mhsa':
      assert not any([kernel_size, conv_norm_layer_tpl, lconv_tpl])
    else:
      assert kernel_size

    if _AttenCtxIsSet(atten_local_context):
      assert not _AttenCtxIsSet(atten_left_context) and not _AttenCtxIsSet(
          atten_right_context
      ), ('atten_local_context and atten_{left,right}_context can not be set'
          'at the same time.')
      atten_left_context = atten_local_context + 1  # including self position.
      atten_right_context = atten_local_context

    if is_causal and trans_atten_tpl is None:
      # None is different from 0, the former is 'infinite'.
      assert atten_right_context is not None, (
          'is_causal is not compatible with infinite atten_right_context '
          '(None).')

    p = cls.Params().Set(
        input_dim=input_dim,
        atten_num_heads=atten_num_heads,
        fflayer_hidden_dim=fflayer_hidden_dim,
        fflayer_activation=fflayer_activation,
        fflayer_residual_weight=fflayer_residual_weight,
        kernel_size=kernel_size,
        is_causal=is_causal,
        layer_order=layer_order,
        dropout_prob=dropout_prob)
    # Set the two feed forward modules.
    if fflayer_end_tpl is not None:
      p.fflayer_end_tpl = fflayer_end_tpl
    if fflayer_start_tpl is not None:
      p.fflayer_start_tpl = fflayer_start_tpl
    # Set the MHSA module.
    if trans_atten_tpl is not None:
      assert atten_left_context is None
      assert atten_right_context is None
      assert use_relative_atten is None
      p.trans_atten_tpl = trans_atten_tpl
    else:
      atten_tpl = cls._ConfigSelfAttenContext(
          atten_left_context,
          atten_right_context,
          use_relative_atten=use_relative_atten,
          relative_pos_emb_dim=input_dim)
      p.trans_atten_tpl = attention_lib.TransformerAttentionLayer.Params().Set(
          atten_tpl=atten_tpl)
    # Set the convolution module.
    if lconv_tpl is not None:
      p.lconv_tpl = lconv_tpl
    if conv_norm_layer_tpl is not None:
      p.lconv_tpl.conv_norm_layer_tpl = conv_norm_layer_tpl
    if fprop_dtype is not None:
      p.cls.SetFPropDtype(p, fprop_dtype)
    if use_moe_in_fflayer_start:
      p.cls.SetMoEFFLayerStartParams(p, moe_num_partitions, moe_num_experts,
                                     moe_num_groups, moe_per_capacity_dim)
    if use_moe_in_fflayer_end:
      p.cls.SetMoEFFLayerEndParams(p, moe_num_partitions, moe_num_experts,
                                   moe_num_groups, moe_per_capacity_dim)
    return p

  @classmethod
  def _ConfigSelfAttenContext(cls, atten_left_context, atten_right_context, *,
                              use_relative_atten, relative_pos_emb_dim):
    # TODO(jamesqin): add an attention factory in batch_major_attention.
    if not _AttenCtxIsSet(atten_left_context) and not _AttenCtxIsSet(
        atten_right_context):
      # No atten context set, each position attends to all positions.
      atten_type = 'global' if not use_relative_atten else 'global_relative'
    elif not _AttenCtxIsSet(atten_left_context) and atten_right_context == 0:
      # Left context is infinite, right context is 0.
      assert not use_relative_atten, (
          'Relative attention isn\'t supported for causal attention.')
      atten_type = 'global_causal'
    else:
      atten_type = 'local_relative' if use_relative_atten else 'local'

    if atten_type == 'global_relative':
      atten_tpl = (
          attention_lib.MultiHeadedAttentionXL.Params().Set(
              rel_pos_emb_dim=relative_pos_emb_dim))
    elif atten_type == 'local_relative':
      atten_tpl = attention_lib.LocalSelfAttentionXL.Params().Set(
          left_context=atten_left_context,
          right_context=atten_right_context,
          rel_pos_emb_dim=relative_pos_emb_dim)
    elif atten_type == 'local':
      atten_tpl = attention_lib.LocalSelfAttention.Params().Set(
          left_context=atten_left_context, right_context=atten_right_context)
    else:
      # No op for 'global' atten
      assert atten_type in ('global', 'global_causal'), (
          f'Unknown atten_type {atten_type}')
      atten_tpl = attention_lib.MultiHeadedAttention.Params()
    return atten_tpl

  @classmethod
  def SetFPropDtype(cls, p, fprop_dtype):
    p.fprop_dtype = fprop_dtype
    for sub_p in (p.lconv_tpl, p.trans_atten_tpl, p.fflayer_start_tpl,
                  p.fflayer_end_tpl):
      if sub_p is not None:
        sub_p.cls.SetFPropDtype(sub_p, fprop_dtype)
    return p

  @classmethod
  def SetMoEFFLayerStartParams(cls, params, num_devices, num_experts,
                               num_groups, per_expert_capacity_dim):
    """Updates params setting MoE as feed-forward layer."""
    params.fflayer_start_tpl = _GShardMoELayerParams(num_devices, num_groups,
                                                     num_experts,
                                                     per_expert_capacity_dim)

  @classmethod
  def SetMoEFFLayerEndParams(cls, params, num_devices, num_experts, num_groups,
                             per_expert_capacity_dim):
    """Updates params setting MoE as feed-forward layer."""
    params.fflayer_end_tpl = _GShardMoELayerParams(num_devices, num_groups,
                                                   num_experts,
                                                   per_expert_capacity_dim)

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.layer_order in [
        'mhsa', 'conv', 'mhsa_before_conv', 'conv_before_mhsa'
    ]
    if p.layer_order == 'mhsa':
      assert not self.has_lconv, 'mhsa must not have a lconv block.'

    if self.has_fflayer_start:
      fflayer_start_p, is_moe_layer = self._ConfigFFLayerOrMoEParams(
          p.fflayer_start_tpl, 'fflayer_start')
      if is_moe_layer:
        self.CreateChild('fflayer_start_moe', fflayer_start_p)
      else:
        self.CreateChild('fflayer_start', fflayer_start_p)

    if not p.fflayer_weight_sharing:
      fflayer_end_p, is_moe_layer = self._ConfigFFLayerOrMoEParams(
          p.fflayer_end_tpl, 'fflayer_end')
      if is_moe_layer:
        self.CreateChild('fflayer_end_moe', fflayer_end_p)
      else:
        self.CreateChild('fflayer_end', fflayer_end_p)
    else:
      if is_moe_layer:
        self.AddChild('fflayer_end_moe', self.fflayer_start_moe)
      else:
        self.AddChild('fflayer_end', self.fflayer_start)

    # For local MHSA, is_masked is ignored, thus it's safe to set is_masked
    # based on p.is_causal, for global and local MHSA cases.
    if self.has_mhsa:
      trans_atten_p = p.trans_atten_tpl.Copy().Set(
          input_dim=p.input_dim,
          num_heads=p.atten_num_heads,
          is_masked=p.is_causal,
          atten_dropout_prob=p.dropout_prob,
          residual_dropout_prob=p.dropout_prob)
      if tf.logging.vlog_is_on(2):
        for line in trans_atten_p.atten_tpl.ToText().split('\n'):
          tf.logging.info('ConformerLayer.atten_tpl: %s', line)
      self.CreateChild('trans_atten', trans_atten_p)

    if self.has_lconv:
      lconv_p = p.lconv_tpl.Copy().Set(
          input_dim=p.input_dim,
          kernel_size=p.kernel_size,
          is_causal=p.is_causal)
      self.CreateChild('lconv', lconv_p)

    ln_p = p.final_ln_tpl.Copy().Set(name='final_ln', input_dim=p.input_dim)
    self.CreateChild('final_ln', ln_p)

  # lconv and fflayer_start have the special treatment, which can be absent,
  # because Transformer doesn't have those.
  @property
  def has_lconv(self):
    return bool(self.params.lconv_tpl)

  @property
  def has_mhsa(self):
    return bool('mhsa' in self.params.layer_order)

  @property
  def has_fflayer_start(self):
    return bool(self.params.fflayer_start_tpl)

  def _ConfigFFLayerOrMoEParams(self, fflayer_tpl, name_prefix):
    """Configures fflayer_tpl params to create Feed-forward layer or MoE params.

    Args:
      fflayer_tpl: Input Feedforward/MoE params to be initialized.
      name_prefix: Layer name prefix to be added in case of creating MoE layer.

    Returns:
      fflayer_p: Configured Feedforward/MoE layer params to be initialized.
      is_moe_layer_p: Whether returned `fflayer_p` params of form subclass:
      gshard_builder.MoEBuilder.
    """
    p = self.params
    if (issubclass(fflayer_tpl.cls,
                   layers_with_attention.TransformerFeedForwardLayer)):
      fflayer_p = fflayer_tpl.Copy().Set(
          input_dim=p.input_dim,
          hidden_dim=p.fflayer_hidden_dim,
          activation=p.fflayer_activation,
          residual_weight=p.fflayer_residual_weight,
          residual_dropout_prob=p.dropout_prob,
          relu_dropout_prob=p.dropout_prob)
      return fflayer_p, False
    elif issubclass(fflayer_tpl.cls, gshard_builder.MoEBuilder):
      moe_builder_p = fflayer_tpl.Copy().Set(
          model_dim=p.input_dim,
          dropout_rate=p.dropout_prob,
          moe_hidden_dim=p.fflayer_hidden_dim,
          moe_activation=p.fflayer_activation)
      if moe_builder_p.num_devices is None:
        raise ValueError('num_devices must be specified for MoEBuilder.')
      is_moe_layer = True
      name = name_prefix + '_moe'
      moe_p = moe_builder_p.Instantiate().MoE(name)
      moe_builder = moe_builder_p.Instantiate()
      moe_p = moe_builder.EncoderLayer(
          name,
          moe_builder.MoE(name),
          residual_weight=p.fflayer_residual_weight)
      return moe_p, is_moe_layer
    else:
      raise ValueError('p.fflayer_tpl must be either '
                       'TransformerFeedForwardLayer or MoEBuilder.')

  def _SelfAtten(self, theta, inputs, paddings):
    inputs, _ = self.trans_atten.FProp(
        theta.trans_atten,
        query_vec=inputs,
        source_vecs=None,
        paddings=paddings)
    return inputs, paddings

  def _LConv(self, theta, inputs, paddings):
    assert self.has_lconv and self.params.layer_order != 'mhsa', (
        'mhsa does not have a lconv block.')
    inputs, paddings = self.lconv.FProp(theta.lconv, inputs, paddings)
    return inputs, paddings

  def _MoeOrFFLayer(self, theta, fflayer_name, in_nmap):
    """FProp for MoE or Feed forward layer.

    Args:
      theta: Layer theta: A NestedMap of Tensors.
      fflayer_name: Child FFLayer name as created in __init__.
        For example: 'fflayer_end'. This assumes the moe_layer if created would
        have the convention as (`fflayer_name` + `_moe`).
      in_nmap: Nested Map containing the following:

        * inputs: A Tensor of shape [batch, seqlen, dim0].
        * paddings: A Tensor of shape [batch, seqlen].
        * moe_aux_loss: [None] Optional aux loss if present in input batch.

    Returns:
     out_nmap: A NestedMap of output tensors:

       * features: Tensor of shape [batch, seqlen, dim0].
       * paddings: A Tensor of shape [batch, seqlen].
       * aux_loss: [Optional] Scalar tensor. Output moe auxiliary loss with
         input aux loss added.

    """
    out_nmap = in_nmap.copy()
    if fflayer_name in self.children:
      outputs = self.children[fflayer_name].FProp(
          theta.GetItem(fflayer_name), in_nmap.features, in_nmap.paddings)
      out_nmap.features = outputs
      return out_nmap
    else:
      moe_fflayer_name = fflayer_name + '_moe'
      if moe_fflayer_name not in self.children:
        raise AssertionError(
            '{} child layer not present.'.format(moe_fflayer_name))
      if moe_fflayer_name not in theta:
        raise AssertionError(
            '{} layer theta not present.'.format(moe_fflayer_name))
      # 0 - padded positions and 1 - non-padded positions.
      segment_ids = tf.cast(1. - in_nmap.paddings, tf.int32)
      segment_pos = tf.zeros_like(segment_ids)  # not used but required by MoE.
      moe_in = py_utils.NestedMap(
          vec=in_nmap.features, segment_id=segment_ids, segment_pos=segment_pos)
      moe_out = self.children[moe_fflayer_name].FProp(
          theta.GetItem(moe_fflayer_name), moe_in)
      out_nmap.features = moe_out.vec
      aux_loss = moe_out.aux_loss
      if 'aux_loss' in in_nmap:
        aux_loss += in_nmap.aux_loss
      # Add 'aux_loss' in out_nmap.
      out_nmap.aux_loss = aux_loss
      return out_nmap

  def _FProp(self, theta, in_nmap):
    p = self.params

    with tf.name_scope(p.name):
      inputs = in_nmap.features
      paddings = in_nmap.paddings
      inputs, paddings = self._CastToFPropDtype((inputs, paddings))
      out_nmap = py_utils.NestedMap()
      if self.has_fflayer_start:
        in_nmap = self._MoeOrFFLayer(theta, 'fflayer_start', in_nmap)
      inputs = in_nmap.features
      if p.layer_order == 'mhsa':
        inputs, paddings = self._SelfAtten(theta, inputs, paddings)
      elif p.layer_order == 'conv':
        inputs, paddings = self._LConv(theta, inputs, paddings)
      elif p.layer_order == 'mhsa_before_conv':
        inputs, paddings = self._SelfAtten(theta, inputs, paddings)
        inputs, paddings = self._LConv(theta, inputs, paddings)
      else:
        assert p.layer_order == 'conv_before_mhsa'
        inputs, paddings = self._LConv(theta, inputs, paddings)
        inputs, paddings = self._SelfAtten(theta, inputs, paddings)
      in_nmap.features = inputs
      in_nmap.paddings = paddings
      in_nmap = self._MoeOrFFLayer(theta, 'fflayer_end', in_nmap)
      inputs = in_nmap.features
      if 'aux_loss' in in_nmap:
        out_nmap.aux_loss = in_nmap.aux_loss
      inputs = self.final_ln.FProp(theta.final_ln, inputs)
      inputs, paddings = self._CastToFPropDtype((inputs, paddings))
      out_nmap.features = inputs
      out_nmap.paddings = paddings
      return out_nmap

  def FProp(self, theta, in_nmap):
    p = self.params
    if not p.remat:
      return self._FProp(theta, in_nmap)

    def CellFn(theta, state0, unused_inputs):
      out_nmap = self._FProp(theta, state0)
      return out_nmap, py_utils.NestedMap()

    _, state1 = recurrent.Recurrent(
        theta=theta,
        state0=in_nmap,
        inputs=py_utils.NestedMap(
            inputs=tf.zeros([1, 0])),  # A dummy input of shape [T, ?].
        cell_fn=CellFn,
        allow_implicit_capture=p.allow_implicit_capture)

    return state1

  def zero_state(self, batch_size):
    if self.params.is_causal:
      lconv_state = py_utils.NestedMap()
      atten_state = py_utils.NestedMap()
      if self.has_lconv:
        with tf.name_scope('lconv'):
          lconv_state = self.lconv.zero_state(batch_size)
      if self.has_mhsa:
        with tf.name_scope('atten'):
          atten_state = self.trans_atten.zero_state(batch_size)
      return py_utils.NestedMap(
          lconv_state=lconv_state, atten_state=atten_state)
    else:
      return py_utils.NestedMap()

  def StreamStep(self, theta, inputs, paddings, state0):
    """Streams t steps.

    Args:
      theta: A NestedMap of read-only layer params.
      inputs: A tensor of shape [b, t, d].
      paddings: A 0/1 valued tensor of shape [b, t].
      state0: A NestedMap of tensors of the same struct as returned by
        zero_state().

    Returns:
      outputs:A tensor of shape [b, t, d].
      padding: the same as input paddings.
      state1: A NestedMap of tensors of the same struct as state0.
    """
    p = self.params
    assert p.is_causal
    assert not p.remat

    with tf.name_scope(f'{p.name}/StreamStep'):
      in_nmap = py_utils.NestedMap(features=inputs, paddings=paddings)
      if self.has_fflayer_start:
        in_nmap = self._MoeOrFFLayer(theta, 'fflayer_start', in_nmap)
      inputs = in_nmap.features

      if p.layer_order == 'mhsa':
        inputs, paddings, atten_state1 = self.trans_atten.StreamStep(
            theta.trans_atten, inputs, paddings, state0.atten_state)
      elif p.layer_order == 'conv':
        inputs, paddings, lconv_state1 = self.lconv.StreamStep(
            theta.lconv, inputs, paddings, state0.lconv_state)
      elif p.layer_order == 'mhsa_before_conv':
        inputs, paddings, atten_state1 = self.trans_atten.StreamStep(
            theta.trans_atten, inputs, paddings, state0.atten_state)
        inputs, paddings, lconv_state1 = self.lconv.StreamStep(
            theta.lconv, inputs, paddings, state0.lconv_state)
      else:
        assert p.layer_order == 'conv_before_mhsa'
        inputs, paddings, lconv_state1 = self.lconv.StreamStep(
            theta.lconv, inputs, paddings, state0.lconv_state)
        inputs, paddings, atten_state1 = self.trans_atten.StreamStep(
            theta.trans_atten, inputs, paddings, state0.atten_state)
      if not self.has_lconv:
        lconv_state1 = py_utils.NestedMap()
      if not self.has_mhsa:
        atten_state1 = py_utils.NestedMap()

      in_nmap.features = inputs
      in_nmap.paddings = paddings
      in_nmap = self._MoeOrFFLayer(theta, 'fflayer_end', in_nmap)
      inputs = in_nmap.features
      outputs = self.final_ln.FProp(theta.final_ln, inputs)

      state1 = py_utils.NestedMap(
          lconv_state=lconv_state1, atten_state=atten_state1)
      return outputs, paddings, state1


def ApplyGshard(conformer_tpl,
                device_mesh=None,
                proj_w_split_list=None,
                proj_activation_split_list=None,
                atten_dnh_w_split=None,
                atten_blnh_activation_split=None,
                atten_bld_activation_split=None,
                lconv_df_w_split=None,
                lconv_hwim_w_split=None,
                lconv_fd_w_split=None,
                lconv_blf_activation_split=None,
                lconv_bld_activation_split=None):
  """Applies gshard on conformer params.

  Args:
    conformer_tpl: A NestedMap of conformer Params.
    device_mesh: A numpy.ndarray specifying the device mesh on which the
      computation is sharded.
    proj_w_split_list: A list of mesh split specifying how weights are sharded
      for fflayer.
    proj_activation_split_list: A list of mesh split specifying how activations
      are sharded for fflayer.
    atten_dnh_w_split: Mesh split of the attention projection weight with the
      shape of [model_dim, num_heads, dim_per_head].
    atten_blnh_activation_split: Mesh split of the attention activation with
      shape of [batch, seq_len, num_heads, dim_per_head].
    atten_bld_activation_split: Mesh split of the attention activation with
      shape of [batch, seq_len, model_dim].
    lconv_df_w_split: Mesh split of the weights in lconv with the shape of
      [model_dim, ff_hidden_dim].
    lconv_hwim_w_split: Mesh split of the depthwise conv weight in lconv with
      the shape of [height, width, in_channels, channel_multiplier].
    lconv_fd_w_split: Mesh split of the weights in lconv with the shape of
      [ff_hidden_dim, model_dim].
    lconv_blf_activation_split: Mesh split of the activations in lconv with the
      shape of [batch, seq_len, ff_hidden_dim].
    lconv_bld_activation_split: Mesh split of the activations in lconv with the
      shape of [batch, seq_len, model_dim].

  Returns:
    The updated conformer_tpl.
  """
  # Not all attention class supports gshard. If not, errors would be throw here.
  conformer_tpl.trans_atten_tpl.atten_tpl.device_mesh = device_mesh
  conformer_tpl.trans_atten_tpl.atten_tpl.weight_split_dims_mapping = (
      atten_dnh_w_split)
  conformer_tpl.trans_atten_tpl.atten_tpl.proj_tpl.weight_split_dims_mapping = (
      atten_dnh_w_split)
  conformer_tpl.trans_atten_tpl.atten_tpl.activation_split_dims_mapping.blnh = (
      atten_blnh_activation_split)
  conformer_tpl.trans_atten_tpl.atten_tpl.activation_split_dims_mapping.bld = (
      atten_bld_activation_split)
  # TODO(jamesqin): support residual_proj xla sharding too.
  conformer_tpl.fflayer_start_tpl.fflayer_tpl.Set(
      device_mesh=device_mesh,
      weight_split_dims_mapping_list=proj_w_split_list,
      activation_split_dims_mapping_list=proj_activation_split_list)
  conformer_tpl.fflayer_end_tpl.fflayer_tpl.Set(
      device_mesh=device_mesh,
      weight_split_dims_mapping_list=proj_w_split_list,
      activation_split_dims_mapping_list=proj_activation_split_list)
  conformer_tpl.lconv_tpl.Set(
      split_act_gated_linear_start=True, device_mesh=device_mesh)
  lconv_w_split = conformer_tpl.lconv_tpl.weight_split_dims_mapping
  lconv_w_split.df = lconv_df_w_split
  lconv_w_split.hwim = lconv_hwim_w_split
  lconv_w_split.fd = lconv_fd_w_split
  lconv_activation_split = conformer_tpl.lconv_tpl.activation_split_dims_mapping
  lconv_activation_split.blf = lconv_blf_activation_split
  lconv_activation_split.bld = lconv_bld_activation_split
  return conformer_tpl
