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
        'glu_activation', 'NONE',
        'Activation in GLU. Check lingvo.core.activations._ACTIVATIONS for '
        'other options.')
    p.Define('dropout_prob', 0., 'Dropout probability.')

    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Input layer norm template.')
    p.Define('linear_start_tpl', layers.FCLayer.Params(), 'Linear start layer.')
    p.Define('depthwise_conv_tpl',
             conv_layers_with_time_padding.DepthwiseConv2DLayer.Params(),
             'Depthwise conv template.')
    p.Define('conv_norm_layer_tpl', bn_layers.BatchNormLayer.Params(),
             'Normalization layer after conv.')
    p.Define('linear_end_tpl', layers.FCLayer.Params(), 'Linear end layer.')
    p.Define('dropout_tpl', layers.DropoutLayer.Params(),
             'Residual dropout layer.')

    p.linear_start_tpl.Set(activation='NONE', has_bias=True)
    p.linear_end_tpl.Set(activation='NONE', has_bias=True)
    return p

  @classmethod
  def CommonParams(cls,
                   input_dim=None,
                   kernel_size=None,
                   conv_activation='SWISH',
                   dropout_prob=0.):
    p = cls.Params().Set(
        input_dim=input_dim,
        kernel_size=kernel_size,
        conv_activation=conv_activation,
        dropout_prob=dropout_prob)
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    ln_p = p.ln_tpl.Copy().Set(name='ln', input_dim=p.input_dim)
    self.CreateChild('ln', ln_p)

    linear_start_p = p.linear_start_tpl.Copy().Set(
        name='linear_start', input_dim=p.input_dim, output_dim=2 * p.input_dim)
    linear_end_p = p.linear_end_tpl.Copy().Set(
        name='linear_end', input_dim=p.input_dim, output_dim=p.input_dim)
    self.CreateChild('linear_start', linear_start_p)
    self.CreateChild('linear_end', linear_end_p)

    norm_p = p.conv_norm_layer_tpl.Copy().Set(
        name='norm_layer', dim=p.input_dim)
    self.CreateChild('norm', norm_p)

    # 1d depthwise conv with channel_mulitplier = 1
    depthwise_conv_p = p.depthwise_conv_tpl.Copy().Set(
        name='depthwise_conv',
        filter_shape=(p.kernel_size, 1, p.input_dim, 1),
        filter_stride=(1, 1))
    self.CreateChild('depthwise_conv1d', depthwise_conv_p)

    dropout_p = p.dropout_tpl.Copy().Set(
        name='dropout', keep_prob=1. - p.dropout_prob)
    self.CreateChild('dropout', dropout_p)

  def _GLU(self, inputs):
    p = self.params
    gated_inputs, act_inputs = tf.split(inputs, 2, axis=-1)
    return self._ApplyActivation(act_inputs,
                                 p.glu_activation) * tf.sigmoid(gated_inputs)

  def _ApplyActivation(self, inputs, act_name):
    if act_name == 'NONE':
      return inputs
    return activations.GetFn(act_name)(inputs)

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
      unnormalized_inputs = inputs

      inputs = self.ln.FProp(theta.ln, inputs)
      inputs = self.linear_start.FProp(theta.linear_start, inputs)

      inputs = self._GLU(inputs)

      # [b, t, d] --> [b, t, 1, d]
      inputs = tf.expand_dims(inputs, 2)
      inputs, paddings = self.depthwise_conv1d.FProp(theta.depthwise_conv1d,
                                                     inputs, paddings)
      # normalize on 4d inputs. sometimes normalization layer reshapes inputs,
      # so there's no hurry to squeeze the input back, which adds extra overhead
      # on tpu.
      # TODO(jamesqin): add paddings in the call, for causal case.
      inputs = self.norm.FProp(theta.norm, inputs)
      inputs = tf.squeeze(inputs, 2)

      inputs = self._ApplyActivation(inputs, p.conv_activation)

      inputs = self.linear_end.FProp(theta.linear_end, inputs)
      inputs = self.dropout.FProp(theta.dropout, inputs)

      output = inputs + unnormalized_inputs
      return output, paddings


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
    # atten layer
    p.Define('atten_num_heads', None,
             'Num of heads in multi-head self-attention.')
    p.Define(
        'atten_left_context', None, 'Local self attention left context.'
        'If None, infinite left context.')
    p.Define(
        'atten_right_context', None, 'Local self attention right context.'
        'If None, infinite right context.')
    p.Define('use_relative_atten', True, 'If using relative attention.')
    p.Define(
        'relative_pos_emb_dim', None,
        'If use_relative_atten, sets the relative pos embedding dim.'
        'Default is the same as input_dim.')
    p.Define('layer_order', 'mhsa_before_conv',
             'Only mhsa_before_conv or conv_before_mhsa are supported.')

    # lconv layer
    p.Define('kernel_size', None, 'Kernel size of 1d lightweight conv.')

    # fflayer
    p.Define('fflayer_hidden_dim', None,
             'Hidden dim of the fflayers (start and end).')
    p.Define('fflayer_activation', 'SWISH', 'fflayer activation.')
    p.Define('fflayer_residual_weight', 0.5, 'fflayer residual weight.')
    p.Define('dropout_prob', None, 'Signature dropout prob of inner componets.')

    # tpl
    p.Define('fflayer_start_tpl',
             layers_with_attention.TransformerFeedForwardLayer.Params(),
             'Layer params for Feed forward layer at the beginning.')
    p.Define('trans_atten_tpl',
             attention_lib.TransformerAttentionLayer.Params(),
             'Self attention layer params.')
    p.Define('lconv_tpl', LConvLayer.Params(), 'Convolution module params.')
    p.Define('fflayer_end_tpl',
             layers_with_attention.TransformerFeedForwardLayer.Params(),
             'Layer params for Feed forward layer at the end.')
    p.Define('final_ln_tpl', layers.LayerNorm.Params(), 'Final layer norm.')
    # https://b.corp.google.com/issues/167460492#comment16
    p.Define(
        'remat', False, 'If to rematerialize the layer. If true, '
        'intermediate tensors are not saved in FProp().')
    return p

  @classmethod
  def CommonParams(cls,
                   input_dim=None,
                   atten_num_heads=None,
                   atten_local_context=None,
                   atten_left_context=None,
                   atten_right_context=None,
                   use_relative_atten=True,
                   kernel_size=None,
                   fflayer_hidden_dim=None,
                   fflayer_activation='SWISH',
                   layer_order='mhsa_before_conv',
                   dropout_prob=0.):
    assert all([input_dim, atten_num_heads, kernel_size, fflayer_hidden_dim])

    if atten_local_context:
      assert not any([atten_left_context, atten_right_context]), (
          'atten_local_context and atten_{left,right}_context can not be set'
          'at the same time.')
      atten_left_context = atten_local_context + 1  # including self position.
      atten_right_context = atten_local_context

    p = cls.Params().Set(
        input_dim=input_dim,
        atten_num_heads=atten_num_heads,
        atten_left_context=atten_left_context,
        atten_right_context=atten_right_context,
        use_relative_atten=use_relative_atten,
        fflayer_hidden_dim=fflayer_hidden_dim,
        fflayer_activation=fflayer_activation,
        kernel_size=kernel_size,
        layer_order=layer_order,
        dropout_prob=0.)
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.layer_order in ['mhsa_before_conv', 'conv_before_mhsa']

    fflayer_start_p = p.fflayer_start_tpl.Copy().Set(
        input_dim=p.input_dim,
        hidden_dim=p.fflayer_hidden_dim,
        activation='SWISH',
        residual_weight=p.fflayer_residual_weight,
        residual_dropout_prob=p.dropout_prob,
        relu_dropout_prob=p.dropout_prob)
    self.CreateChild('fflayer_start', fflayer_start_p)

    fflayer_end_p = p.fflayer_end_tpl.Copy().Set(
        input_dim=p.input_dim,
        hidden_dim=p.fflayer_hidden_dim,
        activation='SWISH',
        residual_weight=p.fflayer_residual_weight,
        residual_dropout_prob=p.dropout_prob,
        relu_dropout_prob=p.dropout_prob)
    self.CreateChild('fflayer_end', fflayer_end_p)

    trans_atten_p = p.trans_atten_tpl.Copy().Set(
        input_dim=p.input_dim,
        num_heads=p.atten_num_heads,
        atten_dropout_prob=p.dropout_prob,
        residual_dropout_prob=p.dropout_prob)
    self._ConfigSelfAttenParams(trans_atten_p)
    self.CreateChild('trans_atten', trans_atten_p)

    lconv_p = p.lconv_tpl.Copy().Set(
        input_dim=p.input_dim, kernel_size=p.kernel_size)
    self.CreateChild('lconv', lconv_p)

    ln_p = p.final_ln_tpl.Copy().Set(name='final_ln', input_dim=p.input_dim)
    self.CreateChild('final_ln', ln_p)

  def _ConfigSelfAttenParams(self, trans_atten_p):
    p = self.params
    if not p.relative_pos_emb_dim:
      p.relative_pos_emb_dim = p.input_dim
    if not any([p.atten_left_context, p.atten_right_context]):
      atten_type = 'global' if not p.use_relative_atten else 'global_relative'
    elif p.atten_left_context is None and p.atten_right_context == 0:
      assert not p.use_relative_atten, (
          'Relative attention isn\'t supported for causal attention.')
      atten_type = 'causal'
    else:
      atten_type = 'local_relative' if p.use_relative_atten else 'local'

    if atten_type == 'causal':
      trans_atten_p.is_masked = True
    elif atten_type == 'global_relative':
      trans_atten_p.atten_tpl = (
          attention_lib.MultiHeadedAttentionXL.Params().Set(
              rel_pos_emb_dim=p.relative_pos_emb_dim))
    elif atten_type == 'local_relative':
      trans_atten_p.atten_tpl = attention_lib.LocalSelfAttentionXL.Params().Set(
          left_context=p.atten_left_context,
          right_context=p.atten_right_context,
          rel_pos_emb_dim=p.relative_pos_emb_dim)
    elif atten_type == 'local':
      trans_atten_p.atten_tpl = attention_lib.LocalSelfAttention.Params().Set(
          left_context=p.atten_left_context,
          right_context=p.atten_right_context)
    # No op for 'global' atten

  def _SelfAtten(self, theta, inputs, paddings):
    inputs, _ = self.trans_atten.FProp(
        theta.trans_atten,
        query_vec=inputs,
        source_vecs=None,
        paddings=paddings)
    return inputs, paddings

  def _LConv(self, theta, inputs, paddings):
    inputs, paddings = self.lconv.FProp(theta.lconv, inputs, paddings)
    return inputs, paddings

  def _FProp(self, theta, inputs, paddings):
    p = self.params

    with tf.name_scope(p.name):
      inputs = self.fflayer_start.FProp(theta.fflayer_start, inputs, paddings)
      if p.layer_order == 'mhsa_before_conv':
        inputs, paddings = self._SelfAtten(theta, inputs, paddings)
        inputs, paddings = self._LConv(theta, inputs, paddings)
      else:
        assert p.layer_order == 'conv_before_mhsa'
        inputs, paddings = self._LConv(theta, inputs, paddings)
        inputs, paddings = self._SelfAtten(theta, inputs, paddings)
      inputs = self.fflayer_end.FProp(theta.fflayer_end, inputs, paddings)

      inputs = self.final_ln.FProp(theta.final_ln, inputs)
      return inputs, paddings

  def FProp(self, theta, inputs, paddings):
    p = self.params
    if not p.remat:
      return self._FProp(theta, inputs, paddings)

    def CellFn(theta, state0, unused_inputs):
      outs, out_paddings = self._FProp(theta, state0.inputs, state0.paddings)
      return py_utils.NestedMap(
          inputs=outs, paddings=out_paddings), py_utils.NestedMap()

    state0 = py_utils.NestedMap(inputs=inputs, paddings=paddings)
    _, state1 = recurrent.Recurrent(
        theta=theta,
        state0=state0,
        inputs=py_utils.NestedMap(
            inputs=tf.zeros([1, 0])),  # A dummy input of shape [T, ?].
        cell_fn=CellFn,
        allow_implicit_capture=p.allow_implicit_capture)

    return state1.inputs, state1.paddings
