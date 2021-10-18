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
"""Convolutional layers."""

import jax
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import activations
from lingvo.jax.layers import normalizations

NestedMap = py_utils.NestedMap
WeightParams = py_utils.WeightParams

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
    p.Define('padding', 'SAME', 'SAME|VALID')
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

  def CreateLayerVariables(self) -> None:
    super().CreateLayerVariables()
    p = self.params
    wp = p.weight_split_dims_mapping
    self.CreateVariable(
        'w',
        WeightParams(
            shape=p.filter_shape,
            init=p.params_init,
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=wp.wt))

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """FProp that supports strided, dilated convolution, depthwise convolution.

    Args:
      theta: NestedMap containing the filter weights of shape [F_h, F_w, D_in,
        D_out]. Optionally for depthwise separable convolutions the kernel can
        be of shape [F_h, F_w, D_in//f, D_out], where f is the
        feature_group_count. Note that in this case D_out must be a multiple of
        feature_group_count.
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    p = self.params
    # Check if the feature_group_count is compatible with the inputs and filter
    # For more information see XLA docs on ConvWithGeneralPadding below
    # https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
    # feature group count is D_in // filter input dim
    feature_group_count = inputs.shape[3] // p.filter_shape[2]
    # filter output dim must be a multiple of feature group count
    assert p.filter_shape[3] % feature_group_count == 0
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
    # The `dimension_numbers=('NHWC', 'HWIO', 'NHWC')` is to be consistent
    # with tf.conv2d, see e.g., see
    # https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L622
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=theta.w,
        window_strides=p.filter_stride,
        padding=[(pad_height_beg, pad_height_end),
                 (pad_width_beg, pad_width_end)],
        rhs_dilation=p.dilations,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=feature_group_count)
    return outputs


class ConvBNAct(Conv2D):
  """A block of conv-bn-activation layers used for image encoders."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('batch_norm', True, 'Whether or not to apply batch norm.')
    p.Define('bn_decay', 0.9, 'Decay in updating the mean and variance.')
    p.Define(
        'bn_use_moving_avg_in_training', False,
        'If True, uses moving avg (mean, variance) during both training '
        'and inference.')
    p.Define(
        'activation', 'RELU', 'Activation function to use.'
        'Options are RELU, RELU6, SIGMOID, TANH, GELU, NONE.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    if p.batch_norm:
      bn = normalizations.BatchNormLayer.Params().Set(
          name='bn',
          dim=p.filter_shape[3],
          decay=p.bn_decay,
          use_moving_avg_in_training=p.bn_use_moving_avg_in_training)
      self.CreateChild('bn', bn)
    act_p = activations.ActivationLayer.Params().Set(activation=p.activation)
    self.CreateChild('activation', act_p)

  def FProp(self, theta: NestedMap, inputs: JTensor) -> JTensor:
    """Forward prop which applies conv-bn-activation.

    Args:
      theta: NestedMap containing the filter weights of shape [F_h, F_w, D_in,
        D_out]. Optionally for depthwise separable convolutions the kernel can
        be of shape [F_h, F_w, D_in//f, D_out], where f is the
        feature_group_count. Note that in this case D_out must be a multiple of
        feature_group_count.
      inputs: Input sequence of shape [B, H, W, D_in], also known more popularly
        as NHWC format.

    Returns:
      Output sequence after applying convolutions of shape [B, H', W', D_out].
      Note that if the padding is SAME and there is no dilation and striding,
      then H' = H and W' = W.
    """
    p = self.params
    outputs = super().FProp(theta, inputs)
    if p.batch_norm:
      outputs = self.bn.FProp(theta.bn, outputs)
    outputs = self.activation.FProp(theta.activation, outputs)
    return outputs
