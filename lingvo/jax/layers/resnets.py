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
"""ResNet layers."""

from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import activations
from lingvo.jax.layers import convolutions
from lingvo.jax.layers import poolings
from lingvo.jax.layers import stochastics

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


class ResNetBlock(base_layer.BaseLayer):
  """ResNet Block as in https://arxiv.org/abs/1512.03385."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for the ResNetBlock."""
    p = super().Params()
    p.Define('input_dim', 0, 'Input dimension.')
    p.Define('output_dim', 0, 'Output dimension.')
    p.Define('conv_params', convolutions.ConvBNAct.Params(),
             'Which Conv block to use.')
    p.Define('kernel_size', 3, 'Kernel sizes of the block.')
    p.Define('stride', 1, 'Stride')
    p.Define(
        'activation', 'RELU', 'Activation function to use.'
        'Options are RELU, RELU6, SIGMOID, TANH, GELU, NONE.')
    p.Define('residual_droppath_prob', 0.0, 'Probability for residual path.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    self._in_out_same_shape = (p.input_dim == p.output_dim and p.stride == 1)

    body = []
    # conv_in, reduce the hidden dims by 4
    body.append(p.conv_params.Copy().Set(
        name='conv_in',
        filter_shape=(1, 1, p.input_dim, p.output_dim // 4),
        filter_stride=(1, 1),
        activation=p.activation))

    # conv_mid using the kernel size and stride provided
    body.append(p.conv_params.Copy().Set(
        name='conv_mid',
        filter_shape=(p.kernel_size, p.kernel_size, p.output_dim // 4,
                      p.output_dim // 4),
        filter_stride=(p.stride, p.stride),
        activation=p.activation))

    # conv_out, expand back to hidden dim
    body.append(p.conv_params.Copy().Set(
        name='conv_out',
        filter_shape=(1, 1, p.output_dim // 4, p.output_dim),
        filter_stride=(1, 1),
        activation='NONE'))
    self.CreateChildren('body', body)

    # projection with 1x1 if input dim and output dim are not the same
    if not self._in_out_same_shape:
      shortcut = p.conv_params.Copy().Set(
          name='shortcut',
          filter_shape=(1, 1, p.input_dim, p.output_dim),
          filter_stride=(p.stride, p.stride),
          activation='NONE')
      self.CreateChild('shortcut', shortcut)

    # Initialize droppath layer
    if p.residual_droppath_prob > 0:
      droppath_p = stochastics.StochasticResidualLayer.Params().Set(
          survival_prob=1.0 - p.residual_droppath_prob)
      self.CreateChild('residual_droppath', droppath_p)

    post_activation = activations.ActivationLayer.Params().Set(
        name='post_activation', activation=p.activation)
    self.CreateChild('postact', post_activation)

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Forward propagation of a ResNetBlock.

    Args:
      theta: A `.NestedMap` object containing variable values of this layer.
      inputs: A `.JTensor` as inputs of [B, H, W, D_in] also commonly known as
        NHWC format.

    Returns:
      A `.JTensor` as outputs of shape [B, H', W', D_out].
    """
    p = self.params
    outputs = inputs

    # body
    for i in range(len(self.body)):
      outputs = self.body[i].FProp(theta.body[i], outputs)

    # projection
    if not self._in_out_same_shape:
      inputs = self.shortcut.FProp(theta.shortcut, inputs)

    # residual
    if p.residual_droppath_prob:
      outputs = self.residual_droppath.FProp(
          theta.residual_droppath,
          inputs,
          outputs,
      )
    else:
      outputs += inputs

    # post activation
    outputs = self.postact.FProp(theta.postact, outputs)
    return outputs


class ResNet(base_layer.BaseLayer):
  """Resnet model with default params matching Resnet-50.

  Additionally, params are also provided for Resnet-101 and Resnet-152.
  See https://arxiv.org/abs/1512.03385 for more details.

  Raises:
    ValueError if length of `strides`, `channels`, `blocks` and `kernels` do
    not match.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'conv_params', convolutions.ConvBNAct.Params(),
        'A layer params template specifying Conv-BN-Activation template '
        'used by the ResNet model.')
    p.Define(
        'block_params', ResNetBlock.Params(),
        'A layer params template specifying Convolution Block used in '
        'each stage. We use the same ResNetBlock tpl in all stages '
        '(4 stages in total) in ResNet.')
    p.Define(
        'strides', [1, 2, 2, 2],
        'A list of integers specifying the stride for each stage. A stage '
        'is defined as a stack of Convolution Blocks that share same '
        'type, channels and kernels. The stride is always applied only at '
        'the beginning of each stage, while within that stage, all other '
        'strides are set to 1 (no stride).')
    p.Define(
        'channels', [256, 512, 1024, 2048],
        'A list of integers specifying the number of channels at '
        'different stages. The first channel is usually 4x the input dim.')
    p.Define(
        'blocks', [3, 4, 6, 3],
        'A list of integers specifying the number of blocks at different '
        'stages.')
    p.Define(
        'kernels', [3, 3, 3, 3],
        'A list of integers specifying the number of kernel sizes at '
        'different stages.')
    p.Define(
        'entryflow_conv_kernel', (7, 7),
        'A tuple of two integers as the kernel size of entryflow convolution.')
    p.Define('entryflow_conv_stride', (2, 2),
             'A tuple of two integers as the stride of entryflow convolution.')
    p.Define(
        'output_spatial_pooling_params', poolings.GlobalPoolingLayer.Params(),
        'A layer params template specifying spatial pooling before output '
        'If None, spatial pooling is not added.')
    return p

  @classmethod
  def ParamsResNet5(cls) -> InstantiableParams:
    """Returns ResNet5 hyperparams for testing purposes."""
    return cls.Params().Set(strides=[1], channels=[16], blocks=[1], kernels=[1])

  @classmethod
  def ParamsResNet50(cls) -> InstantiableParams:
    """Returns commonly used ResNet50 hyperparams."""
    return cls.Params()

  @classmethod
  def ParamsResNet101(cls) -> InstantiableParams:
    """Returns commonly used ResNet101 hyperparams."""
    return cls.Params().Set(blocks=[3, 4, 23, 3])

  @classmethod
  def ParamsResNet152(cls) -> InstantiableParams:
    """Returns commonly used ResNet152 hyperparams."""
    return cls.Params().Set(blocks=[3, 8, 36, 3])

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    num_stages = len(p.strides)
    if num_stages != len(p.channels):
      raise ValueError(
          f'num_stages {num_stages} != channels {len(p.channels)}.')
    if num_stages != len(p.blocks):
      raise ValueError(f'num_stages {num_stages} != blocks {len(p.blocks)}.')
    if num_stages != len(p.kernels):
      raise ValueError(f'num_stages {num_stages} != kernels {len(p.kernels)}.')

    # Set the convolution type used in the Resnet block.
    if hasattr(p.block_params, 'conv_params'):
      _ = p.block_params.Set(conv_params=p.conv_params)

    # Create the entryflow convolution layer.
    input_dim = p.channels[0] // 4
    entryflow_conv_params = p.conv_params.Copy()
    entryflow_conv_params.filter_shape = (p.entryflow_conv_kernel[0],
                                          p.entryflow_conv_kernel[1], 3,
                                          input_dim)
    entryflow_conv_params.filter_stride = p.entryflow_conv_stride
    self.CreateChild('entryflow_conv', entryflow_conv_params)

    # Create the entryflow max pooling layer.
    maxpool_params = poolings.PoolingLayer.Params().Set(
        name='entryflow_maxpool',
        window_shape=(3, 3),
        window_stride=(2, 2),
        pooling_type='MAX')
    self.CreateChild('entryflow_maxpool', maxpool_params)

    # Create the chain of ResNet blocks.
    for stage_id, (channel, num_blocks, kernel, stride) in enumerate(
        zip(p.channels, p.blocks, p.kernels, p.strides)):
      for block_id in range(num_blocks):
        name = f'stage_{stage_id}_block_{block_id}'
        output_dim = channel
        block_p = p.block_params.Copy().Set(
            name=name,
            kernel_size=kernel,
            input_dim=input_dim,
            output_dim=output_dim,
            stride=1 if block_id != 0 else stride,
        )
        self.CreateChild(name, block_p)
        input_dim = output_dim

    # Add optional spatial global pooling.
    if p.output_spatial_pooling_params is not None:
      self.CreateChild('output_spatial_pooling',
                       p.output_spatial_pooling_params)

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Applies the ResNet model to the inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Input image tensor of shape [B, H, W, 3].

    Returns:
      Output tensor of ResNet of shape [B, H, W, D] where D is the last channel
      dimension. If `output_spatial_pooling_params` is not None, then the
      output will be of a shape that depends on which dims are pooled. For e.g.,
      if the pooling dims are [1, 2], then output shape will be [B, D].
    """
    p = self.params

    # Apply the entryflow conv.
    outputs = self.entryflow_conv.FProp(theta.entryflow_conv, inputs)

    # Apply the entryflow maxpooling layer.
    outputs, _ = self.entryflow_maxpool.FProp(theta.entryflow_maxpool, outputs)

    # Apply the ResNet blocks.
    for stage_id, num_blocks in enumerate(p.blocks):
      for block_id in range(num_blocks):
        block_name = f'stage_{stage_id}_block_{block_id}'
        outputs = getattr(self, block_name).FProp(theta[block_name], outputs)

    # Apply optional spatial global pooling.
    if p.output_spatial_pooling_params is not None:
      outputs = self.output_spatial_pooling.FProp(theta.output_spatial_pooling,
                                                  outputs)
    return outputs
