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
"""Conformer-related layers."""

from typing import Dict, Tuple

from lingvo.jax import asserts
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax.layers import attentions
from lingvo.jax.layers import convolutions
from lingvo.jax.layers import normalizations
from lingvo.jax.layers import stochastics
from lingvo.jax.layers import transformers

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor
InstantiableParams = py_utils.InstantiableParams


class ResidualNormWrapper(base_layer.BaseLayer):
  """This is a wrapper layer used in the conformer.

  This wrapper requires kwargs rather than in positional form.

  Example: wrapper.fprop(theta.wrapper, inputs=inputs, paddings=paddings)

  It takes a layer as input and adds normalization and residual connection.

  For the normalization, we can apply it before the layer (pre_layer_norm=True)

  or after the layer (pre_layer_norm=False).

  For the residual connection, we can specify the residual weight.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'residual_weight', 1.0, 'Weight of the residual connection.'
        'Output = fn(x) * residual_weight + x * input_weight.')
    p.Define(
        'input_weight', 1.0, 'Weight of the input connection.'
        'Output = fn(x) * residual_weight + x * input_weight.')
    p.Define('core_tpl', None, 'Template of core layer.')
    p.Define('norm_tpl', normalizations.LayerNorm.Params(),
             'Normalization params.')
    p.Define('pre_layer_norm', True,
             'Whether to apply norm before or after the layer.')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, y) = (x + dropout(y)).')
    p.Define(
        'residual_dropout_tpl', stochastics.Dropout.Params(),
        'Residual dropout params template. keep_prop will be reset to '
        '(1.0 - residual_dropout_prob).')

    p.Define(
        'input_adaptor_fn', None,
        'Input adaptor used to connect between normalized input and inputs to the core_layer'
    )
    p.Define(
        'residual_adaptor_fn', None,
        'Residual adaptor used to connect between core_layer output and input')
    p.Define('norm_adaptor_fn', None,
             'Normalization adaptor used to collect tensor from input dicts')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    asserts.not_none(p.core_tpl)
    self.create_child('core', p.core_tpl)

    asserts.in_set(p.norm_tpl.cls,
                   [normalizations.LayerNorm, normalizations.BatchNorm])
    self.create_child('norm', p.norm_tpl)

    # Initialize residual dropout.
    params = p.residual_dropout_tpl.Copy()
    params.keep_prob = (1.0 - p.residual_dropout_prob)
    self.create_child('residual_dropout', params)

    if p.input_adaptor_fn:
      self.create_child('input_adaptor', p.input_adaptor_fn)
    if p.residual_adaptor_fn:
      self.create_child('residual_adaptor', p.residual_adaptor_fn)
    if p.norm_adaptor_fn:
      self.create_child('norm_adaptor', p.norm_adaptor_fn)

  @property
  def has_input_adaptor(self) -> bool:
    return hasattr(self, 'input_adaptor')

  @property
  def has_residual_adaptor(self) -> bool:
    return hasattr(self, 'residual_adaptor')

  @property
  def has_norm_adaptor(self) -> bool:
    return hasattr(self, 'norm_adaptor')

  def fprop(self, theta: NestedMap, **kwargs) -> JTensor:
    p = self.params

    if p.pre_layer_norm:
      if self.has_norm_adaptor:
        unnormalized_inputs = self.norm_adaptor.fprop(**kwargs)
      else:
        asserts.in_set('inputs', list(kwargs.keys()))
        unnormalized_inputs = kwargs['inputs']

      inputs_normalized = self.norm.fprop(theta.norm, unnormalized_inputs)
      kwargs['inputs'] = inputs_normalized
    else:
      asserts.in_set('inputs', list(kwargs.keys()))
      unnormalized_inputs = kwargs['inputs']

    if self.has_input_adaptor:
      result = self.core.fprop(theta.core,
                               **(self.input_adaptor.fprop(**kwargs)))
    else:
      result = self.core.fprop(theta.core, **kwargs)

    if not p.pre_layer_norm:
      if self.has_norm_adaptor:
        result = self.norm.fprop(theta.norm, self.norm_adaptor.fprop(result))
      else:
        result = self.norm.fprop(theta.norm, result)

    if self.has_residual_adaptor and p.pre_layer_norm:
      result = self.residual_dropout.fprop(
          theta.residual_dropout, self.residual_adaptor.fprop(result)
      ) * p.residual_weight + unnormalized_inputs * p.input_weight
    else:
      result = self.residual_dropout.fprop(
          result) * p.residual_weight + unnormalized_inputs * p.input_weight
    return result


class SelfAttentionInputAdaptor(base_layer.BaseLayer):
  """This adaptor generates core_layer arguments from input dictionary."""

  def fprop(self, **kwargs) -> Dict[str, JTensor]:
    return NestedMap(
        query_vec=kwargs['inputs'],
        key_vec=kwargs['inputs'],
        value_vec=kwargs['inputs'],
        atten_mask=attentions.convert_paddings_to_mask(
            kwargs['paddings'], kwargs['inputs'].dtype)).ToNestedDict()


class SelfAttentionResidualAdaptor(base_layer.BaseLayer):
  """This adaptor select tensor from core_layer results for residual."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    return p

  def fprop(self, inputs: Tuple[JTensor]) -> JTensor:
    return inputs[0]


class SelfAttentionNormAdaptor(base_layer.BaseLayer):
  """If post_norm, this adaptor handles results from core_layer.

    If pre_norm, this adaptor handles inputs from the wrapper.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    return p

  def fprop(self, **kwargs) -> JTensor:
    return kwargs['inputs']


class Conformer(base_layer.BaseLayer):
  """Conformer layer as in https://arxiv.org/abs/2005.08100.

    Canonical version (with default params.)
      x = x + 1/2 * FFN(x)
      x = x + MHSA(x)
      x = x + Lconv(x)
      x = x + 1/2 * FFN(x)
      y = ln(x)

    Residual connections are implemented inside each individual block:
      FFN, MHSA, LConv.
    Optionally one can change the order of MHSA and conv.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    # TODO(nanxinchen): add causal support
    p.Define('input_dims', None, 'Input dimension.')
    p.Define('model_dims', 512, 'Encoder model dimension.')
    p.Define('kernel_size', 32, 'Conv kernel size.')

    p.Define('ff_activation', 'SWISH',
             'Activation function used in the feedforward network.')

    p.Define('ff_residual_weight', 0.5, 'Residual weight used in the fflayer.')

    p.Define(
        'ffn_dim_multiplier', 4,
        'Feed forward hidden dimension will be ffn_dim_multiplier * model_dims.'
    )
    p.Define('atten_num_heads', 8, 'Number of attention heads.')
    p.Define(
        'layer_order', 'mhsa_before_conv',
        'Only mhsa, conv, mhsa_before_conv or conv_before_mhsa are '
        'supported.')
    p.Define('dropout_prob', None, 'Dropout prob of inner components.')

    # fflayer tpl
    p.Define(
        'fflayer_start_tpl', transformers.TransformerFeedForward.Params(),
        'Layer params for Feed forward layer at the beginning.'
        'If set to None, this layer is excluded.')
    p.Define(
        'trans_atten_tpl',
        ResidualNormWrapper.Params().Set(
            core_tpl=attentions.DotProductAttention.Params(),
            input_adaptor_fn=SelfAttentionInputAdaptor.Params(),
            norm_adaptor_fn=SelfAttentionNormAdaptor.Params(),
            residual_adaptor_fn=SelfAttentionResidualAdaptor.Params()),
        'Self attention layer params.')
    p.Define(
        'lconv_tpl', convolutions.LightConv1D.Params(),
        'Convolution module params. If set to None, this layer is excluded.')
    p.Define(
        'fflayer_end_tpl', transformers.TransformerFeedForward.Params(),
        'Layer params for Feed forward layer at the end.'
        'If set to None, this layer is excluded.')
    p.Define(
        'fflayer_weight_sharing', False,
        'If True, will ignore `fflayer_end_tpl`, and will make the fflayer_end '
        'layer as a weight-shared copy of the fflayer_start layer.')
    p.Define('final_ln_tpl', normalizations.LayerNorm.Params(),
             'Final layer norm.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    asserts.in_set(p.layer_order,
                   ['mhsa', 'conv', 'mhsa_before_conv', 'conv_before_mhsa'])

    if p.fflayer_start_tpl:
      if p.input_dims == p.model_dims:
        fflayer_start_p = p.fflayer_start_tpl.Copy().Set(
            name='fflayer_start',
            activation=p.ff_activation,
            input_dims=p.input_dims,
            hidden_dims=p.model_dims * p.ffn_dim_multiplier,
            residual_weight=p.ff_residual_weight,
        )
      else:
        # Need to add another projection layer in fflayer
        fflayer_start_p = p.fflayer_start_tpl.Copy().Set(
            name='fflayer_start',
            activation=p.ff_activation,
            input_dims=p.input_dims,
            projection_dims=p.model_dims,
            hidden_dims=p.model_dims * p.ffn_dim_multiplier,
            residual_weight=p.ff_residual_weight,
        )
      self.create_child(fflayer_start_p.name, fflayer_start_p)

    if p.fflayer_end_tpl:
      fflayer_end_p = p.fflayer_end_tpl.Copy().Set(
          name='fflayer_end',
          activation=p.ff_activation,
          input_dims=p.model_dims,
          hidden_dims=p.model_dims * p.ffn_dim_multiplier,
          residual_weight=p.ff_residual_weight,
      )
      if not p.fflayer_weight_sharing:
        self.create_child(fflayer_end_p.name, fflayer_end_p)
      else:
        asserts.not_none(p.fflayer_start_tpl)

    if 'mhsa' in p.layer_order:
      trans_atten_p = p.trans_atten_tpl.Copy().Set(
          residual_dropout_prob=p.dropout_prob,
          core_tpl=p.trans_atten_tpl.core_tpl.Copy().Set(
              input_dim=p.model_dims,
              hidden_dim=p.model_dims,
              atten_dropout_prob=p.dropout_prob,
              num_heads=p.atten_num_heads))
      if p.trans_atten_tpl.norm_tpl.cls == normalizations.LayerNorm:
        trans_atten_p.norm_tpl = trans_atten_p.norm_tpl.Copy().Set(
            input_dims=p.model_dims)
      else:
        trans_atten_p.norm_tpl = trans_atten_p.norm_tpl.Copy().Set(
            dim=p.model_dims)
      self.create_child('trans_atten', trans_atten_p)

    if 'conv' in p.layer_order:
      lconv_p = p.lconv_tpl.Copy().Set(
          input_dims=p.model_dims, kernel_size=p.kernel_size)
      self.create_child('lconv', lconv_p)

    ln_p = p.final_ln_tpl.Copy().Set(name='final_ln', input_dims=p.model_dims)
    self.create_child('final_ln', ln_p)

  @property
  def has_fflayer_start(self) -> bool:
    return hasattr(self, 'fflayer_start')

  @property
  def has_fflayer_end(self) -> bool:
    return hasattr(self, 'fflayer_end')

  def fprop(self, theta: NestedMap, inputs: JTensor,
            paddings: JTensor) -> JTensor:
    """Conformer layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T] (only used in FFN layer).

    Returns:
      The conformer output with shape [B, T, D].
    """
    p = self.params

    if self.has_fflayer_start:
      inputs = self.fflayer_start.fprop(theta.fflayer_start, inputs, paddings)

    if p.layer_order == 'mhsa':
      inputs = self.trans_atten.fprop(
          theta.trans_atten, inputs=inputs, paddings=paddings)
    elif p.layer_order == 'conv':
      inputs = self.lconv.fprop(theta.lconv, inputs, paddings)
    elif p.layer_order == 'mhsa_before_conv':
      inputs = self.trans_atten.fprop(
          theta.trans_atten, inputs=inputs, paddings=paddings)
      inputs = self.lconv.fprop(theta.lconv, inputs, paddings)
    else:
      assert p.layer_order == 'conv_before_mhsa'
      inputs = self.lconv.fprop(theta.lconv, inputs, paddings)
      inputs = self.trans_atten.fprop(
          theta.trans_atten, inputs=inputs, paddings=paddings)

    if self.has_fflayer_end:
      inputs = self.fflayer_end.fprop(theta.fflayer_end, inputs, paddings)
    elif p.fflayer_weight_sharing:
      # With the weight sharing, we apply fflayer_start again
      inputs = self.fflayer_start.fprop(theta.fflayer_start, inputs, paddings)

    inputs = self.final_ln.fprop(theta.final_ln, inputs)
    return inputs
