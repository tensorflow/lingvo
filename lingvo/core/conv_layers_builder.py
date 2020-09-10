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
from lingvo.core import activations
from lingvo.core import base_layer
from lingvo.core import builder
from lingvo.core import builder_layers
from lingvo.core import conv_layers_with_time_padding
from lingvo.core import py_utils
from lingvo.core import tshape

FLAGS = flags.FLAGS

Conv2DLayerWithPadding = conv_layers_with_time_padding.Conv2DLayerWithPadding
CausalConv2DLayerWithPadding = conv_layers_with_time_padding.CausalConv2DLayerWithPadding
DepthwiseConv2DLayer = conv_layers_with_time_padding.DepthwiseConv2DLayer
CausalDepthwiseConv2DLayer = conv_layers_with_time_padding.CausalDepthwiseConv2DLayer
ConvBatchNormLayer = conv_layers_with_time_padding.ConvBatchNormLayer
ConvCategoricalBN = conv_layers_with_time_padding.ConvCategoricalBN
ActivationLayer = activations.ActivationLayer
PaddingLayer = conv_layers_with_time_padding.PaddingLayer
NormalizedDepthwiseConv2DLayer = conv_layers_with_time_padding.NormalizedDepthwiseConv2DLayer
CausalNormalizedDepthwiseConv2DLayer = conv_layers_with_time_padding.CausalNormalizedDepthwiseConv2DLayer
GlobalPoolingLayer = conv_layers_with_time_padding.GlobalPoolingLayer


class BiasLayer(builder_layers.BiasLayer):

  def FProp(self, theta, inputs, paddings):
    bias_added = super().FProp(theta, inputs)
    return bias_added, paddings


class CausalPoolingLayer(base_layer.BaseLayer):
  """Pooling layer with causal dependency on the time axis."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('pooling_type', 'AVG', 'Pooling type: MAX|AVG')
    p.Define(
        'left_context', None, 'Number of frames to the left in the pooling'
        'window (including the current frame). A special value "-1" means '
        'using all left frames')
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
    if p.left_context == -1:
      if p.pooling_type == 'AVG':
        cumulative_sum = tf.math.cumsum(inputs, axis=1)
        cumulative_count = 1.0 + tf.range(
            py_utils.GetShape(inputs)[1], dtype=p.dtype)
        cumulative_mean = cumulative_sum / cumulative_count[tf.newaxis, :,
                                                            tf.newaxis,
                                                            tf.newaxis]
        cumulative_mean *= 1.0 - paddings[..., tf.newaxis, tf.newaxis]
        return cumulative_mean, paddings
      else:
        raise NotImplementedError('Cumulative max pooling not implemented.')

    window_size = p.left_context
    left_pad_size = window_size - 1
    large_negative = p.dtype.max * tf.constant(-0.7, dtype=p.dtype)
    # For max pooling, use a large negative padding value such that the max
    # element is almost always from a non-padding position.
    pad_value = 0 if p.pooling_type == 'AVG' else large_negative
    inputs = tf.pad(
        inputs, [[0, 0], [left_pad_size, 0], [0, 0], [0, 0]],
        constant_values=pad_value)

    out_feature = tf.nn.pool(
        inputs,
        window_shape=(window_size, 1),
        pooling_type=p.pooling_type,
        padding='VALID')

    if p.pooling_type == 'AVG':
      # Count the fraction of non-padding elements inside each pooling window.
      max_seq_len = py_utils.GetShape(paddings)[1]
      num_non_padded_elements = tf.range(1, 1 + max_seq_len, dtype=p.dtype)
      num_non_padded_elements = tf.minimum(num_non_padded_elements,
                                           tf.cast(window_size, p.dtype))
      non_padded_ratio = num_non_padded_elements / tf.cast(window_size, p.dtype)
      # Divide by non-padding ratios to eliminate the effect of padded zeros.
      out_feature *= tf.math.reciprocal_no_nan(non_padded_ratio[tf.newaxis, :,
                                                                tf.newaxis,
                                                                tf.newaxis])
    out_feature *= 1.0 - paddings[..., tf.newaxis, tf.newaxis]
    return out_feature, paddings


class Builder(builder.Base):
  """Builder patterns for commonly used conv layers."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('norm_layer_tpl',
             ConvBatchNormLayer.Params().Set(decay=0.999),
             'If specified, the normalization layer template.')
    p.Define('weight_norm', False, 'Add weight norm for kernel weights or not.')
    p.Define(
        'v2_padding', False, 'Prefer setting to True. The default '
        'implementation is incorrect for strided convolutions.')

    return p

  def _BiasNoPadding(self, name, dims):
    return super()._Bias(name, dims)

  def _Bias(self, name, dims):
    """Bias layer. The bias is added to the last dimension of the input."""
    return BiasLayer.Params().Set(name=name, dims=dims)

  def _Norm(self, name, dims):
    return self.params.norm_layer_tpl.Copy().Set(name=name, dim=dims)

  def _NormOrBias(self, name, dims):
    if self.params.norm_layer_tpl:
      return self._Norm(name, dims)
    else:
      return self._Bias(name, dims)

  def _MaybeNorm(self, name, dims):
    if self.params.norm_layer_tpl:
      return self._Norm(name, dims)
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
        weight_norm=self.params.weight_norm,
        v2_padding=self.params.v2_padding)

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
        weight_norm=self.params.weight_norm,
        v2_padding=self.params.v2_padding)

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
          self._MaybeNorm('bn', in_dim),
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
          self._NormOrBias('bn_or_bias', out_dim),
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
          self._MaybeNorm('bn', in_dim),
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
          self._NormOrBias('bn_or_bias', in_dim * depth_multiplier),
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
          self._MaybeNorm('bn', in_dim),
          self._Activation('act', activation),
          self._RawDepthwiseConv2D('conv_2d', in_dim, depth_multiplier,
                                   filter_shape, stride, dilation, is_causal),
          # No need to add a padding layer here as subsequent conv layer always
          # properly zeros out padded nodes.
          self._RawConv2D(
              'conv_1x1',
              in_dim * depth_multiplier,
              out_dim,
              filter_shape=[1, 1],
              stride=[1, 1],
              dilation=[1, 1],
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
          self._RawConv2D(
              'conv_1x1',
              in_dim * depth_multiplier,
              out_dim,
              filter_shape=[1, 1],
              stride=[1, 1],
              dilation=[1, 1],
              is_causal=False),
          self._NormOrBias('bn_or_bias', out_dim),
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
        dropconnect_prob=dropconnect_prob,
        v2_padding=self.params.v2_padding)

  def _Add(self, name, residual_weight=1.0):
    return self._Fn(
        name, fn=lambda x, y: x + residual_weight * y, fn_out=lambda x, y: x)

  def _ExpandDims(self, name):
    return self._Fn(
        name,
        fn=lambda x: tf.expand_dims(x, 2),
        fn_out=lambda x: tshape.Shape(x[0:2] + [1] + x[2:]),
        fn_flops=lambda x: 1)

  def _Squeeze(self, name):
    return self._Fn(
        name,
        fn=lambda x: tf.squeeze(x, 2),
        fn_out=lambda x: tshape.Shape(x[0:2] + x[3:]),
        fn_flops=lambda x: 1)

  def _Glu(self, name, glu_with_tanh):

    def _GLUFn(inputs):
      gated_inputs, act_inputs = tf.split(inputs, 2, axis=-1)
      return act_inputs * tf.sigmoid(gated_inputs)

    def _GatedTanhFn(inputs):
      gated_inputs, act_inputs = tf.split(inputs, 2, axis=-1)
      return tf.tanh(act_inputs) * tf.sigmoid(gated_inputs)

    fn = _GatedTanhFn if glu_with_tanh else _GLUFn

    return self._Fn(
        name,
        fn=fn,
        fn_out=lambda x: tshape.Shape(x[:-1] + [x[-1] / 2]),
        fn_flops=lambda x: 15 * x.size)

  def _LConvCommon(self,
                   name,
                   input_dim,
                   kernel_size,
                   activation='RELU',
                   is_causal=False,
                   glu_with_tanh=False,
                   residual_dropout_prob=0):
    # pyformat: disable
    return py_utils.NestedMap(
        pre_conv=('i.vec->pre_conv',
                  self._Seq(
                      'pre_conv',
                      self._LN('ln', input_dim),
                      self._Linear('linear', input_dim, input_dim * 2),
                      self._BiasNoPadding('bias', input_dim * 2),
                      self._Glu('glu', glu_with_tanh),
                      self._ExpandDims('expand'))),
        post_conv=('post_conv->after_dropout',
                   self._Seq(
                       'post_conv',
                       self._Squeeze('squeeze'),
                       self._Linear('linear', input_dim, input_dim),
                       self._BiasNoPadding('bias', input_dim),
                       self._Dropout(
                           'dropout', keep_prob=1 - residual_dropout_prob))),
        residual_add=('i.vec,after_dropout->o.vec', self._Add('add'))
        )
    # pyformat: enable

  def LConv(self,
            name,
            input_dim,
            kernel_size,
            activation='RELU',
            is_causal=False,
            glu_with_tanh=False,
            residual_dropout_prob=0):
    """A lightweight convolution block as described in ...

    https://arxiv.org/abs/1901.10430

    Reference PyTorch Implementation (L587):
    https://github.com/pytorch/fairseq/blob/v0.6.2/fairseq/models/lightconv.py

    Args:
      name: name of the params
      input_dim: Input dimension.
      kernel_size: kernel size used in the conv layer.
      activation: A string, activation function used by the inner conv block.
      is_causal: is causal padding or not.
      glu_with_tanh: if the Gated Linear Unit should apply tanh on the
        activation input.
      residual_dropout_prob: Residual dropout prob.

    Returns:
      A GraphLayer params with a FProp() function of signature
        f(inputs, paddings) -> outputs, out_paddings
    """
    sub_nmap = self._LConvCommon(
        name,
        input_dim,
        kernel_size,
        activation=activation,
        is_causal=is_causal,
        residual_dropout_prob=residual_dropout_prob)
    conv_graph = ('pre_conv,i.paddings->post_conv,o.paddings',
                  self.DepthwiseConv2D(
                      name,
                      in_dim=input_dim,
                      depth_multiplier=1,
                      filter_shape=[kernel_size, 1],
                      activation=activation,
                      is_causal=is_causal))
    sub_list = [
        sub_nmap.pre_conv, conv_graph, sub_nmap.post_conv, sub_nmap.residual_add
    ]

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings}
        ['o'],  # output NestedMap with {vec, paddings}
        *sub_list)

  def NormalizedLConv(self,
                      name,
                      input_dim,
                      kernel_size,
                      num_heads,
                      activation='RELU',
                      is_causal=False,
                      glu_with_tanh=False,
                      residual_dropout_prob=0,
                      dropconnect_prob=0):
    """A lightweight convolution block as described in ...

    https://arxiv.org/abs/2004.11886

    Args:
      name: name of the params
      input_dim: Input dimension.
      kernel_size: kernel size used in the conv layer.
      num_heads: Num of heads.
      activation: A string, activation function used by the inner conv block.
      is_causal: is causal padding or not.
      glu_with_tanh: if the Gated Linear Unit should apply tanh on the
        activation input.
      residual_dropout_prob: Residual dropout prob.
      dropconnect_prob: attention dropout prob.

    Returns:
      A GraphLayer params with a FProp() function of signature
        f(inputs, paddings) -> outputs, out_paddings
    """
    sub_nmap = self._LConvCommon(
        name,
        input_dim,
        kernel_size,
        activation=activation,
        is_causal=is_causal,
        residual_dropout_prob=residual_dropout_prob)
    conv_graph = ('pre_conv,i.paddings->post_conv,o.paddings',
                  self.NormalizedDepthwiseConv2D(
                      name,
                      kernel_size=kernel_size,
                      num_heads=num_heads,
                      in_dim=input_dim,
                      dropconnect_prob=dropconnect_prob,
                      deterministic_dropout=self.params.deterministic_dropout,
                      is_causal=is_causal))
    sub_list = [
        sub_nmap.pre_conv, conv_graph, sub_nmap.post_conv, sub_nmap.residual_add
    ]

    return self._Graph(
        name,
        ['i'],  # input NestedMap with {vec, paddings}
        ['o'],  # output NestedMap with {vec, paddings}
        *sub_list)
