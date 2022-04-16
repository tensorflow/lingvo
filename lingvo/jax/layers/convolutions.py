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
"""Convolutional layers."""

from typing import Tuple

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import activations
from lingvo.jax.layers import linears
from lingvo.jax.layers import normalizations
from lingvo.jax.layers import stochastics

NestedMap = py_utils.NestedMap
weight_params = py_utils.weight_params

InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


class Conv2D(base_layer.BaseLayer):
  """Conv2D with support of SAME/VALID paddings."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'filter_shape', (0, 0, 0, 0),
        'Filter shape. Must be a sequence of length 4. Elements are in'
        ' the order of height (time), width (frequency), in_channel,'
        ' out_channel. ')
    p.Define(
        'filter_stride', (0, 0),
        'Filter stride to use. Must be a pair of ints. The first int'
        ' specifies the stride on the height dimension. The second int'
        ' specifies the stride on the width dimension.')
    p.Define(
        'dilations', (1, 1), ' An optional list of ints. Defaults to [1, 1]. '
        '1-D tensor of length 2. The dilation factor for each dimension '
        'of input. If set to k > 1, there will be k-1 skipped cells '
        'between each filter element on that dimension.')
    p.Define('bias', False, 'Whether or not to apply a bias before activation.')
    p.Define('bias_init', py_utils.WeightInit.Constant(0.0),
             'Bias initializer to use if bias is to be applied.')
    p.Define('padding', 'SAME', 'SAME|VALID')
    p.Define(
        'tf_equivalent_padding', False,
        ' Whether to make it equivalent to tf. By default'
        'we apply extra padding that is different than tf conv when stride > 1.'
        'This is mainly used for multimodal which leads to better accuracy')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.padding in ['SAME', 'VALID']
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert len(p.dilations) == 2
    assert all(x > 0 for x in p.filter_stride)

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    wp = p.weight_split_dims_mapping
    self.create_variable(
        'w',
        weight_params(
            shape=p.filter_shape,
            init=p.params_init,
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=wp.wt))
    if p.bias:
      self.create_variable(
          'b',
          weight_params(
              shape=[p.filter_shape[-1]], dtype=p.dtype, init=p.bias_init))

  def fprop(self, inputs: JTensor) -> JTensor:
    """FProp that supports strided, dilated convolution, depthwise convolution.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    p = self.params
    theta = self.local_theta()
    # Check if the feature_group_count is compatible with the inputs and filter
    # For more information see XLA docs on ConvWithGeneralPadding below
    # https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
    # feature group count is D_in // filter input dim
    feature_group_count = inputs.shape[3] // p.filter_shape[2]
    # filter output dim must be a multiple of feature group count
    assert p.filter_shape[3] % feature_group_count == 0
    if not p.tf_equivalent_padding:
      if p.padding == 'SAME':
        pad_height_total = p.filter_shape[0] - 1
        pad_height_beg = pad_height_total // 2
        pad_height_end = pad_height_total - pad_height_beg
        pad_width_total = p.filter_shape[1] - 1
        pad_width_beg = pad_width_total // 2
        pad_width_end = pad_width_total - pad_width_beg
      else:
        assert p.padding == 'VALID', p.padding
        pad_height_beg = 0
        pad_height_end = 0
        pad_width_beg = 0
        pad_width_end = 0
      padding = [(pad_height_beg, pad_height_end),
                 (pad_width_beg, pad_width_end)]
    else:
      padding = p.padding
    # The `dimension_numbers=('NHWC', 'HWIO', 'NHWC')` is to be consistent
    # with tf.conv2d, see e.g., see
    # https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L622
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=theta.w,
        window_strides=p.filter_stride,
        padding=padding,
        rhs_dilation=p.dilations,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=feature_group_count)
    if p.bias:
      outputs += jnp.reshape(theta.b, (1,) * (outputs.ndim - 1) + (-1,))
    return outputs


class ConvBNAct(Conv2D):
  """A block of conv-bn-activation layers used for image encoders.

  By default, we use cross-replica sum on TPUs.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('batch_norm', True, 'Whether or not to apply batch norm.')
    p.Define('bn_decay', 0.9, 'Decay in updating the mean and variance.')
    p.Define(
        'bn_cross_replica_sum_on_tpu', True,
        'If true, computes global mean and variance across all replicas.'
        'Only effective for tpu.')
    p.Define(
        'activation', 'RELU', 'Activation function to use.'
        'Options are RELU, RELU6, SIGMOID, TANH, GELU, NONE.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    if p.batch_norm:
      bn = normalizations.BatchNorm.Params().Set(
          name='bn',
          dim=p.filter_shape[3],
          decay=p.bn_decay,
          use_moving_avg_in_training=False,
          enable_cross_replica_sum_on_tpu=p.bn_cross_replica_sum_on_tpu,
      )
      self.create_child('bn', bn)
    act_p = activations.Activation.Params().Set(activation=p.activation)
    self.create_child('activation', act_p)

  def fprop(self, inputs: JTensor) -> JTensor:
    """Forward prop which applies conv-bn-activation.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    p = self.params
    outputs = super().fprop(inputs)
    if p.batch_norm:
      outputs = self.bn.fprop(outputs)
    outputs = self.activation.fprop(outputs)
    return outputs

  def fprop_with_padding(self, inputs: JTensor,
                         paddings: JTensor) -> Tuple[JTensor, JTensor]:
    """Forward prop with time paddings.

    Args:
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.
      paddings: Input sequence of shape [B, H], where H is the time dimension.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
      Output padding after applying convolutions.
    """
    p = self.params

    outputs = self.fprop(inputs)

    if p.filter_stride[0] == 1 and p.padding == 'SAME':
      return outputs, paddings
    if p.padding == 'SAME':
      input_length = paddings.shape[1]
      stride = p.filter_stride[0]

      pad_len = (input_length + stride - 1) // stride * stride - input_length
      out_padding = jax.lax.conv_general_dilated(
          lhs=paddings[:, :, None],
          rhs=jnp.ones([1, 1, 1]),
          window_strides=p.filter_stride[:1],
          padding=[(0, pad_len)],
          rhs_dilation=p.dilations[:1],
          dimension_numbers=('NHC', 'HIO', 'NHC'))
      out_padding = jnp.squeeze(out_padding, axis=-1)
    else:

      def rolling_window(arr: JTensor, window: int, stride: int):
        idx = jnp.arange(0, arr.shape[1] - window + 1,
                         stride)[:, None] + jnp.arange(window)[None, :]
        return arr[:, idx]

      window = p.filter_shape[0]
      stride = p.filter_stride[0]
      out_padding = rolling_window(paddings, window, stride)
      out_padding = out_padding.min(axis=-1, keepdims=False)
    return outputs, out_padding


# TODO(nanxinchen): add Depthwise Conv2D support
class DepthwiseConv1D(base_layer.BaseLayer):
  """Depthwise 1D convolution based on lax implementation."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'filter_shape', (0, 0, 0),
        'Filter shape. Must be a sequence of length 3. Elements are in'
        ' the order of kernel_size, in_channels, channel_multipliers. ')
    p.Define('bias', False, 'Whether or not to apply a bias before activation.')
    p.Define('bias_init', py_utils.WeightInit.Constant(0.0),
             'Bias initializer to use if bias is to be applied.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    assert len(self.params.filter_shape) == 3

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    self.create_variable(
        'w',
        weight_params(
            shape=[p.filter_shape[0], 1, p.filter_shape[1] * p.filter_shape[2]],
            dtype=p.dtype,
            init=p.params_init))
    if p.bias:
      self.create_variable(
          'b', weight_params(shape=[p.dim], dtype=p.dtype, init=p.bias_init))

  def fprop(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    """Depthwise convolution layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The depthwise conv output with shape [B, T, H].
    """
    p = self.params
    theta = self.local_theta()

    # Applying padding.
    inputs = inputs * (1.0 - jnp.expand_dims(paddings, axis=-1))

    dn = jax.lax.conv_dimension_numbers(inputs.shape, theta.w.shape,
                                        ('NHC', 'HIO', 'NHC'))

    out = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=theta.w,
        window_strides=(1,),
        padding='SAME',
        lhs_dilation=(1,),
        rhs_dilation=(1,),
        dimension_numbers=dn,
        feature_group_count=p.filter_shape[1])
    if p.bias:
      out = out + theta.b
    return out


class LightConv1D(base_layer.BaseLayer):
  """Lightweight conv layer.

  architecture::

  input-ln()-ff()-glu()-depthwise_conv1d()-norm()-act()-ff()-dropout()-+-output
    |__________________________________________________________________|

  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', None, 'Input and (in fact,) output dimension.')
    p.Define('kernel_size', None, 'Kernel size of 1d deptwise conv.')
    p.Define('conv_activation', 'SWISH', 'Activation after normalization.')
    p.Define('dropout_prob', 0., 'Dropout probability.')

    p.Define('ln_tpl', normalizations.LayerNorm.Params(),
             'Input layer norm template.')
    p.Define('linear_start_tpl', linears.FeedForward.Params(),
             'Linear start layer.')

    # TODO(nanxinchen): add causal support
    p.Define('depthwise_conv_tpl', DepthwiseConv1D.Params(),
             'Depthwise conv template.')
    p.Define('conv_norm_layer_tpl', normalizations.BatchNorm.Params(),
             'Normalization layer after conv.')

    # TODO(nanxinchen): add SPMD partitioning support
    p.Define('linear_end_tpl', linears.FeedForward.Params(),
             'Linear end layer.')
    p.Define('dropout_tpl', stochastics.Dropout.Params(),
             'Residual dropout layer.')
    p.linear_start_tpl.Set(activation='NONE')
    p.linear_end_tpl.Set(activation='NONE')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    ln_p = p.ln_tpl.Copy().Set(name='ln', input_dims=p.input_dims)
    self.create_child('ln', ln_p)

    linear_start_act_p = p.linear_start_tpl.Copy().Set(
        input_dims=p.input_dims, output_dims=p.input_dims)
    linear_start_gated_p = p.linear_start_tpl.Copy().Set(
        input_dims=p.input_dims, output_dims=p.input_dims)
    self.create_child('linear_start_act', linear_start_act_p)
    self.create_child('linear_start_gated', linear_start_gated_p)

    depthwise_conv_p = p.depthwise_conv_tpl.Set(
        name='depthwise_conv', filter_shape=(p.kernel_size, p.input_dims, 1))
    self.create_child('depthwise_conv1d', depthwise_conv_p)

    if p.conv_norm_layer_tpl.cls == normalizations.LayerNorm:
      norm_p = p.conv_norm_layer_tpl.Copy().Set(
          name='norm_layer', input_dims=p.input_dims)
    else:
      norm_p = p.conv_norm_layer_tpl.Copy().Set(
          name='norm_layer', dim=p.input_dims)
    self.create_child('conv_norm', norm_p)

    self.create_child(
        'conv_activation',
        activations.Activation.Params().Set(activation=p.conv_activation))

    linear_end_p = p.linear_end_tpl.Copy().Set(
        name='linear_end', input_dims=p.input_dims, output_dims=p.input_dims)
    self.create_child('linear_end', linear_end_p)

    dropout_p = p.dropout_tpl.Copy().Set(
        name='dropout', keep_prob=1. - p.dropout_prob)
    self.create_child('dropout', dropout_p)

  def fprop(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    """Lightweight conv layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T].

    Returns:
      The lconv output with shape [B, T, H].
    """
    unnormalized_inputs = inputs

    inputs = self.ln.fprop(inputs)
    act_inputs = self.linear_start_act.fprop(inputs)
    gated_inputs = self.linear_start_gated.fprop(inputs)
    inputs = act_inputs * jax.nn.sigmoid(gated_inputs)

    inputs = self.depthwise_conv1d.fprop(inputs, paddings)

    inputs = self.conv_norm.fprop(inputs, jnp.expand_dims(paddings, -1))

    inputs = self.conv_activation.fprop(inputs)

    inputs = self.linear_end.fprop(inputs)
    inputs = self.dropout.fprop(inputs)

    output = inputs + unnormalized_inputs
    return output
