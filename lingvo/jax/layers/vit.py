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
"""Layers for Vision Transformer (Vit).

The following notations are used through this file:
B = batch size
H = height
W = width
P = patch size
C = number of channels
N = number of tokens
D = hidden dims
"""

import einops
import jax.numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import embedding_softmax
from lingvo.jax.layers import linears
from lingvo.jax.layers import normalizations
from lingvo.jax.layers import poolings
from lingvo.jax.layers import stochastics
from lingvo.jax.layers import transformers

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


def image_to_patch(img: JTensor, patch_size: int) -> JTensor:
  """Convert image to patches.

  Args:
    img: JTensor, [B, H, W, C] ,
    patch_size: integer, dimension of a square patch.

  Returns:
    batched_img: [B, (H * W / P^2), P^2 * C].
  """

  if len(img.shape) < 4:
    raise ValueError('Image should be formatted as 4D [B, H, W, C]')
  height, width, channels = img.shape[1:]

  if height % patch_size != 0 or width % patch_size != 0:
    raise ValueError(
        'Image height and width should be multiples of patch_size.')

  row_blocks = height // patch_size
  column_blocks = width // patch_size

  img = einops.rearrange(
      img,
      '... (m p)(n q) c->...(m n)(p q c)',
      m=row_blocks,
      n=column_blocks,
      p=patch_size,
      q=patch_size,
      c=channels)
  return img


class VitEntryLayers(base_layer.BaseLayer):
  """Entry block of ViT.

  It performs the following operations:
    - patchifying the input
    - linear projection
    - adding positional embedding
    - adding potential dropouts
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('image_size', 224, 'Height/width of a square image.')
    p.Define('patch_size', 0, 'Height/width of a square patch')
    p.Define('dim_per_patch', 0,
             'Number of channels per patch after pachifying.')
    p.Define('image_channels', 3, 'Number of channels of the input image.')
    p.Define(
        'pos_emb_dropout_prob', 0.0,
        'Probability to apply dropout on the learnable positional'
        'embedding.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    p_patch_projection = linears.FeedForward.Params().Set(
        name='proj',
        input_dims=p.patch_size**2 * p.image_channels,
        output_dims=p.dim_per_patch,
        activation='NONE')
    self.create_child('patch_projection', p_patch_projection)

    num_patches = (p.image_size // p.patch_size)**2

    # TODO(zhangzd): Support cls-token and other heads.

    p_emb = embedding_softmax.TrainablePositionalEmbedding.Params().Set(
        name='emb',
        max_seq_length=num_patches,
        embedding_dims=p.dim_per_patch,
        params_init=py_utils.WeightInit.Gaussian(scale=0.02))
    self.create_child('pos_emb', p_emb)

    if p.pos_emb_dropout_prob > 0.0:
      p_dropout = stochastics.Dropout.Params().Set(
          name='dropout', keep_prob=1.0 - p.pos_emb_dropout_prob)
      self.create_child('dropout', p_dropout)

  def fprop(self, inputs: JTensor) -> JTensor:
    """Applies the vit entry operations to the input image.

    Args:
      inputs: Input image tensor of shape [B, H, W, 3].

    Returns:
      Output tensor of shape [B, N, D].
    """
    p = self.params
    features = image_to_patch(inputs, p.patch_size)
    features = self.patch_projection.fprop(features)

    num_patches = (p.image_size // p.patch_size)**2
    features = features + self.pos_emb.fprop(seq_length=num_patches)
    if self.params.pos_emb_dropout_prob > 0.0:
      features = self.dropout.fprop(features)

    return features


class VitTransformerLayers(base_layer.BaseLayer):
  """Middle block of ViT consisting of a number of transformer layers."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Number of channels of the input tensor.')
    p.Define('hidden_dims', 0,
             'Hidden dims of the FFN layers in the transformer.')
    p.Define('num_heads', 0,
             'Number of heads of the multi-headed self-attention.')
    p.Define(
        'residual_dropout_prob', 0.0,
        'Probability at which we apply dropout to the residual layers, '
        'such that, residual(x, y) = (x + dropout(y)).')
    p.Define('activation_dropout_prob', 0.0,
             'Probability at which we apply dropout to the FFN layers.')
    p.Define('atten_dropout_prob', 0.0,
             'Probability at which we apply dropout to the attention weights.')
    p.Define('stochastic_depth_dropout_prob', 0.0,
             'A float as the stochastic depth dropout probability.')
    p.Define('ff_activation', 'GELU', 'Activation used in the FFN layers.')
    p.Define('num_layers', 12, 'Number of transformer layers.')
    p.Define('atten_logit_cap', 0.0,
             'Attention logit cap in the self-attention.')

    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    if p.activation_dropout_prob != p.atten_dropout_prob or (
        p.activation_dropout_prob != p.residual_dropout_prob):
      raise ValueError(
          'This implementation is based on StackedTransformer, '
          'which only supports same dropout prob applied everhwere.')

    p_stacked_tfm = transformers.StackedTransformer.Params().Set(
        model_dims=p.input_dims,
        hidden_dims=p.hidden_dims,
        num_heads=p.num_heads,
        mask_self_attention=False,
        cross_attention=False,
        packed_input=False,
        num_layers=p.num_layers,
        dropout_prob=p.atten_dropout_prob,
        residual_droppath_prob=p.stochastic_depth_dropout_prob,
    )
    p_tfm = p_stacked_tfm.transformer_layer_params_tpl
    p_tfm.norm_policy = 'pre'
    p_tfm.tr_fflayer_tpl.activation = p.ff_activation
    p_tfm.tr_atten_tpl.atten_logit_cap = p.atten_logit_cap

    self.create_child('tfms', p_stacked_tfm)

  def fprop(self, inputs: JTensor) -> JTensor:
    """Applying transformers sequentially.

    Args:
      inputs: Input tensor of shape [B, N, D].

    Returns:
      Output tensor of shape [B, N, D].
    """
    paddings = jnp.zeros(inputs.shape[:2])
    inputs = self.tfms.fprop(inputs, paddings)

    return inputs


class VitExitLayers(base_layer.BaseLayer):
  """Exit block of ViT.

  It consists of layer norm, pooling, projection and dropout.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('hidden_dim', 0, 'Number of channels of the input tensor.')
    p.Define('output_dim', 0, 'Number of channels of the output tensor.')
    p.Define('output_dropout_prob', 0.0,
             'Probability to apply dropout on the output tensor.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    p_ln = normalizations.LayerNorm.Params().Set(
        name='ln', input_dims=p.hidden_dim)
    self.create_child('ln', p_ln)

    p_pooling = poolings.GlobalPooling.Params().Set(
        pooling_type='MAX', pooling_dims=[1], keepdims=False)
    self.create_child('pooling', p_pooling)

    p_fc_tanh = linears.FeedForward.Params().Set(
        input_dims=p.hidden_dim, output_dims=p.output_dim, activation='TANH')
    self.create_child('fc_tanh', p_fc_tanh)

    if p.output_dropout_prob > 0.0:
      p_dropout = stochastics.Dropout.Params().Set(keep_prob=1.0 -
                                                   p.output_dropout_prob)
      self.create_child('dropout', p_dropout)

  def fprop(self, inputs: JTensor) -> JTensor:
    """FProp function.

    Args:
      inputs: Input tensor of shape [B, N, D].

    Returns:
      Output tensor of shape [B, D].
    """
    inputs = self.ln.fprop(inputs)
    inputs = self.pooling.fprop(inputs)
    inputs = self.fc_tanh.fprop(inputs)
    if self.params.output_dropout_prob > 0.0:
      inputs = self.dropout.fprop(inputs)
    return inputs


# TODO(zhangzd): Add support for stochastic drop-path
# TODO(zhangzd): Fix the potential discrepency on per_dim_scale.
class VisionTransformer(base_layer.BaseLayer):
  """Vision transformer model."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()

    # Architecture related parameters
    p.Define('hidden_dim', 768,
             'An integer specifying hidden dimension of transformers.')
    p.Define('num_tfm_layers', 12,
             'An integer specifying number of transformers.')
    p.Define(
        'num_heads', 12,
        'An integer specifying number of attention heads in transformers.')
    p.Define('mlp_dim', 3072,
             'An integer specifying mlp dimension of transformers.')
    p.Define('patch_size', 16,
             'An integer specifying the size of patch for ViT.')
    p.Define('ff_activation', 'GELU',
             'Activation in the feed forward of the tfm.')

    # Input specification
    p.Define('image_size', 224,
             'An integer specifying the size of input image.')
    p.Define('image_channel_size', 3,
             'An integer specifying the number of channels of input image.')

    # Drop out related parameters
    p.Define('activation_dropout_prob', 0.0,
             'A float as the activation dropout probability in transformers.')
    p.Define('atten_dropout_prob', 0.0,
             'A float as the attention dropout probability in transformers.')
    p.Define('pos_emb_dropout_prob', 0.0,
             'A float as the dropout probability after positional embedding.')
    p.Define('residual_dropout_prob', 0.0,
             'A float as the residual dropout probability in transformers.')
    p.Define('output_dropout_prob', 0.0,
             'A float as the output dropout probability.')
    p.Define('stochastic_depth_dropout_prob', 0.0,
             'A float as the stochastic depth dropout probability.')

    # Architecture details
    p.Define(
        'atten_logit_cap', 0,
        'If > 0, applies cap * tf.math.tanh(logits / cap) to attention '
        'logits to improve training stabilities.')
    p.Define('exit_layers_tpl', VitExitLayers.Params(), 'Exit block of ViT.')

    return p

  @classmethod
  def ParamsViTBase(cls, dropout: bool) -> InstantiableParams:
    """Returns commonly used ViTBase hyperparams.

    Args:
      dropout: A bool. If true, enables default dropouts.

    Returns:
      A hyperparams.Params() instance as commonly used config.
    """
    params = cls.Params().Set(
        hidden_dim=768,
        num_tfm_layers=12,
        num_heads=12,
        mlp_dim=3072,
        patch_size=16,
    )
    if dropout:
      params.Set(
          activation_dropout_prob=0.1,
          # Diffrent from Vit-tf, since Vit-jax does NOT support different
          # dropout probs at different places.
          atten_dropout_prob=0.1,
          pos_emb_dropout_prob=0.1,
          residual_dropout_prob=0.1,
      )
    else:
      params.Set(
          activation_dropout_prob=0.0,
          atten_dropout_prob=0.0,
          pos_emb_dropout_prob=0.0,
          residual_dropout_prob=0.0,
      )
    return params

  @classmethod
  def ParamsViTMedium(cls, dropout: bool) -> InstantiableParams:
    return cls.ParamsViTBase(dropout).Set(
        hidden_dim=1024,
        num_tfm_layers=12,
        num_heads=32,
        mlp_dim=8192,
        patch_size=16,
    )

  @classmethod
  def ParamsViTLarge(cls, dropout: bool) -> InstantiableParams:
    return cls.ParamsViTBase(dropout).Set(
        hidden_dim=1024,
        num_tfm_layers=24,
        num_heads=16,
        mlp_dim=4096,
        patch_size=16,
    )

  @classmethod
  def ParamsViTHuge(cls, dropout: bool) -> InstantiableParams:
    return cls.ParamsViTBase(dropout).Set(
        hidden_dim=1280,
        num_tfm_layers=32,
        num_heads=16,
        mlp_dim=5120,
        patch_size=14,
    )

  @classmethod
  def ParamsViTGiant(cls, dropout: bool) -> InstantiableParams:
    return cls.ParamsViTBase(dropout).Set(
        hidden_dim=2048,
        num_tfm_layers=48,
        num_heads=16,
        mlp_dim=8192,
        patch_size=14,
    )

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    p_entry = VitEntryLayers.Params().Set(
        name='entry',
        image_size=p.image_size,
        patch_size=p.patch_size,
        dim_per_patch=p.hidden_dim,
        image_channels=p.image_channel_size,
        pos_emb_dropout_prob=p.pos_emb_dropout_prob)
    self.create_child('entry_stack', p_entry)

    p_tfm = VitTransformerLayers.Params().Set(
        name='transformers',
        input_dims=p.hidden_dim,
        hidden_dims=p.mlp_dim,
        num_heads=p.num_heads,
        num_layers=p.num_tfm_layers,
        residual_dropout_prob=p.residual_dropout_prob,
        activation_dropout_prob=p.activation_dropout_prob,
        atten_dropout_prob=p.activation_dropout_prob,
        stochastic_depth_dropout_prob=p.stochastic_depth_dropout_prob,
        ff_activation=p.ff_activation,
        atten_logit_cap=p.atten_logit_cap)
    self.create_child('transformers_stack', p_tfm)

    if p.exit_layers_tpl is not None:
      p_exit = p.exit_layers_tpl.Copy().Set(
          name='exit',
          hidden_dim=p.hidden_dim,
          output_dim=p.hidden_dim,
          output_dropout_prob=p.output_dropout_prob,
      )
      self.create_child('exit_stack', p_exit)

  def fprop(self, inputs: JTensor) -> JTensor:
    """Applies the Vit model to the inputs.

    Args:
      inputs: Input image tensor of shape [B, H, W, 3] (H == W).

    Returns:
      Output tensor of shape [B, D] or [B, N, D].
    """
    features = self.entry_stack.fprop(inputs)  # [B, N, D]
    features = self.transformers_stack.fprop(features)  # [B, N, D]
    if 'exit_stack' in self.children:
      features = self.exit_stack.fprop(features)  # [B, D]
    return features
