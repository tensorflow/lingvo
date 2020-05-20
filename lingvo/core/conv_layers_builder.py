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
"""2D conv layers that are expected to be used with sequence inputs."""
import math
from absl import flags

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import builder
from lingvo.core import builder_layers
from lingvo.core import conv_layers_with_time_padding
from lingvo.core import py_utils

FLAGS = flags.FLAGS

Conv2DLayerWithPadding = conv_layers_with_time_padding.Conv2DLayerWithPadding
CausalConv2DLayerWithPadding = conv_layers_with_time_padding.CausalConv2DLayerWithPadding
DepthwiseConv2DLayer = conv_layers_with_time_padding.DepthwiseConv2DLayer
CausalDepthwiseConv2DLayer = conv_layers_with_time_padding.CausalDepthwiseConv2DLayer
ConvBatchNormLayer = conv_layers_with_time_padding.ConvBatchNormLayer
ActivationLayer = conv_layers_with_time_padding.ActivationLayer
PaddingLayer = conv_layers_with_time_padding.PaddingLayer
NormalizedDepthwiseConv2DLayer = conv_layers_with_time_padding.NormalizedDepthwiseConv2DLayer
CausalNormalizedDepthwiseConv2DLayer = conv_layers_with_time_padding.CausalNormalizedDepthwiseConv2DLayer
GlobalPoolingLayer = conv_layers_with_time_padding.GlobalPoolingLayer


class BiasLayer(builder_layers.BiasLayer):

  def FProp(self, theta, inputs, paddings):
    bias_added = super(BiasLayer, self).FProp(theta, inputs)
    return bias_added, paddings


class CausalPoolingLayer(base_layer.BaseLayer):
  """Pooling layer with causal dependency on the time axis."""

  @classmethod
  def Params(cls):
    p = super(CausalPoolingLayer, cls).Params()
    p.Define('pooling_type', 'AVG', 'Pooling type: MAX|AVG')
    p.Define(
        'left_context', None, 'Number of frames to the left in the pooling'
        'window (including the current frame).')
    return p

  def FProp(self, theta, inputs, paddings):
    """Applies causal pooling to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch, time,
        frequency, channel]. The time dimension corresponds to the height
        dimension as in images and the frequency dimension corresponds to the
        width dimension as in images.
      paddings: The paddings tensor. It is expected to be of shape [batch,
        time].

    Returns:
      outputs, out_paddings pair.
       - outputs: has the same shape as inputs.
       - out_paddings: has the same tshape as paddings.
    """

    p = self.params
    if p.left_context is None:
      raise ValueError('left_context must be set.')
    window_size = p.left_context
    left_pad_size = window_size - 1
    inputs = tf.pad(inputs, [[0, 0], [left_pad_size, 0], [0, 0], [0, 0]])

    out_feature = tf.nn.pool(
        inputs,
        window_shape=(window_size, 1),
        pooling_type=p.pooling_type,
        padding='VALID')

    out_feature *= 1.0 - paddings[..., tf.newaxis, tf.newaxis]
    return out_feature, paddings


class Builder(builder.Base):
  """Builder patterns for commonly used conv layers."""

  @classmethod
  def Params(cls):
    p = super(Builder, cls).Params()
    p.Define('use_bn', True, 'Add additional bn layers to conv layers or not.')
    p.Define('weight_norm', False, 'Add weight norm for kernel weights or not.')
    return p

  def _Bias(self, name, dims):
    """Bias layer. The bias is added to the last dimension of the input."""
    return BiasLayer.Params().Set(name=name, dims=dims)

  def _BN(self, name, dims, decay=0.999):
    return ConvBatchNormLayer.Params().Set(name=name, dim=dims, decay=decay)

  def _BiasOrBN(self, name, dims):
    if self.params.use_bn:
      return self._BN(name, dims)
    else:
      return self._Bias(name, dims)

  def _MaybeBN(self, name, dims):
    if self.params.use_bn:
      return self._BN(name, dims)
    else:
      return self._Id(name)

  def _Activation(self, name, activation):
    return ActivationLayer.Params().Set(name=name, activation=activation)

  def _Padding(self, name):
    return PaddingLayer.Params().Set(name=name)

  def _RawConv2D(self,
                 name,
                 in_dim,
                 out_dim,
                 filter_shape,
                 stride,
                 dilation,
                 is_causal):
    if is_causal:
      conv_cls = CausalConv2DLayerWithPadding
    else:
      conv_cls = Conv2DLayerWithPadding
    return conv_cls.Params().Set(
        name=name,
        filter_shape=filter_shape + [in_dim, out_dim],
        filter_stride=stride,
        dilation_rate=dilation,
        weight_norm=self.params.weight_norm)

  def _RawDepthwiseConv2D(self,
                          name,
                          in_dim,
                          depth_multiplier,
                          filter_shape,
                          stride,
                          dilation,
                          is_causal):
    if is_causal:
      conv_cls = CausalDepthwiseConv2DLayer
    else:
      conv_cls = DepthwiseConv2DLayer
    return conv_cls.Params().Set(
        name=name,
        filter_shape=filter_shape + [in_dim, depth_multiplier],
        filter_stride=stride,
        dilation_rate=dilation,
        weight_norm=self.params.weight_norm)

  def _GlobalPooling(self, name, pooling_type):
    return GlobalPoolingLayer.Params().Set(name=name, pooling_type=pooling_type)

  def Conv2D(self,
             name,
             in_dim,
             out_dim,
             filter_shape,
             stride=None,
             dilation=None,
             activation='RELU',
             conv_last=False,
             is_causal=False):
    if stride is None:
      stride = [1, 1]
    if dilation is None:
      dilation = [1, 1]
    if conv_last:
      layers_in_sequence = [
          self._MaybeBN('bn', in_dim),
          self._Activation('act', activation),
          self._RawConv2D('conv_2d', in_dim, out_dim, filter_shape, stride,
                          dilation, is_causal),
          self._Bias('bias', out_dim),
          self._Padding('pad')
      ]
    else:
      layers_in_sequence = [
          self._RawConv2D('conv_2d', in_dim, out_dim, filter_shape, stride,
                          dilation, is_causal),
          self._BiasOrBN('bn_or_bias', out_dim),
          self._Activation('act', activation),
          self._Padding('pad')
      ]
    return self._Seq(name, *layers_in_sequence)

  def DepthwiseConv2D(self,
                      name,
                      in_dim,
                      depth_multiplier,
                      filter_shape,
                      stride=None,
                      dilation=None,
                      activation='RELU',
                      conv_last=False,
                      is_causal=False):
    if stride is None:
      stride = [1, 1]
    if dilation is None:
      dilation = [1, 1]
    if conv_last:
      layers_in_sequence = [
          self._MaybeBN('bn', in_dim),
          self._Activation('act', activation),
          self._RawDepthwiseConv2D('conv_2d', in_dim, depth_multiplier,
                                   filter_shape, stride, dilation, is_causal),
          self._Bias('bias', in_dim * depth_multiplier),
          self._Padding('pad')
      ]
    else:
      layers_in_sequence = [
          self._RawDepthwiseConv2D('conv_2d', in_dim, depth_multiplier,
                                   filter_shape, stride, dilation, is_causal),
          self._BiasOrBN('bn_or_bias', in_dim * depth_multiplier),
          self._Activation('act', activation),
          self._Padding('pad')
      ]
    return self._Seq(name, *layers_in_sequence)

  def SeparableConv2D(self,
                      name,
                      in_dim,
                      out_dim,
                      depth_multiplier,
                      filter_shape,
                      stride=None,
                      dilation=None,
                      activation='RELU',
                      conv_last=False,
                      is_causal=False):
    if stride is None:
      stride = [1, 1]
    if dilation is None:
      dilation = [1, 1]
    if conv_last:
      layers_in_sequence = [
          self._MaybeBN('bn', in_dim),
          self._Activation('act', activation),
          self._RawDepthwiseConv2D('conv_2d', in_dim, depth_multiplier,
                                   filter_shape, stride, dilation, is_causal),
          # No need to add a padding layer here as subsequent conv layer always
          # properly zeros out padded nodes.
          self._RawConv2D('conv_1x1', in_dim * depth_multiplier, out_dim,
                          filter_shape=[1, 1], stride=[1, 1], dilation=[1, 1],
                          is_causal=False),
          self._Bias('bias', out_dim),
          self._Padding('pad')
      ]
    else:
      layers_in_sequence = [
          self._RawDepthwiseConv2D('conv_2d', in_dim, depth_multiplier,
                                   filter_shape, stride, dilation, is_causal),
          # No need to add a padding layer here as subsequent conv layer always
          # properly zeros out padded nodes.
          self._RawConv2D('conv_1x1', in_dim * depth_multiplier, out_dim,
                          filter_shape=[1, 1], stride=[1, 1], dilation=[1, 1],
                          is_causal=False),
          self._BiasOrBN('bn_or_bias', out_dim),
          self._Activation('act', activation),
          self._Padding('pad')
      ]
    return self._Seq(name, *layers_in_sequence)

  def NormalizedDepthwiseConv2D(self,
                                name,
                                kernel_size,
                                num_heads,
                                in_dim,
                                dropconnect_prob=0,
                                deterministic_dropout=False,
                                is_causal=False):
    if is_causal:
      conv_cls = CausalNormalizedDepthwiseConv2DLayer
    else:
      conv_cls = NormalizedDepthwiseConv2DLayer
    return conv_cls.Params().Set(
        name=name,
        filter_shape=[kernel_size, 1, num_heads, 1],
        weight_tiling_factor=in_dim // num_heads,
        deterministic_dropout=deterministic_dropout,
        params_init=py_utils.WeightInit.TruncatedGaussian(
            scale=math.sqrt(2.6 / kernel_size)),  # Fan-out initialization.
        dropconnect_prob=dropconnect_prob)
