# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers
import numpy as np
import six
from six.moves import range
import tensorflow as tf

from tensorflow.python.framework import function
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import inplace_ops
from lingvo.core import base_layer
from lingvo.core import bn_layers
from lingvo.core import conv_layers_with_time_padding
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import recurrent
from lingvo.core import summary_utils


def Gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (
      1.0 + tf.erf(input_tensor / tf.cast(tf.sqrt(2.0), input_tensor.dtype)))
  return input_tensor * cdf


# Supported activation functions.
_ACTIVATIONS = {
    'RELU': tf.nn.relu,
    'RELU6': tf.nn.relu6,
    'SIGMOID': tf.sigmoid,
    'TANH': tf.tanh,
    'GELU': Gelu,
    'SWISH': tf.nn.swish,
}

# A subset of activation functions are supported by TFLite as fused activation
# functions with a preceding matmul or conv. If this is the case, then they
# require special treatment for quantization.
_TFLITE_FUSED_ACTIVATION_NAMES = (
    'RELU',
    'RELU6',
)

LOG_SCALE_CLAMP_BOUND = 20.0


class IdentityLayer(base_layer.BaseLayer):
  """Identity layer, adds name and propagates its input."""

  @base_layer.initializer
  def __init__(self, params):
    super(IdentityLayer, self).__init__(params)

  def FProp(self, theta, inputs, *args):
    """Identity mapping.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., input_dim].
      *args: Arguments to be ignored.

    Returns:
      Tensor with the same shape and type of inputs.
    """
    p = self.params
    return tf.identity(inputs, name=p.name)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(flops=0, out_shapes=(inputs,))


# TODO(yonghui/jonathanasdf): Remove the forwarded links.
_ComputeConvOutputShape = conv_layers_with_time_padding.ComputeConvOutputShape
_ComputeConvOutputPadding = (
    conv_layers_with_time_padding.ComputeConvOutputPadding)
BatchNormLayer = bn_layers.BatchNormLayer
BatchNormLayerNoPadding = bn_layers.BatchNormLayerNoPadding
AddingAccumulator = bn_layers.AddingAccumulator


class BaseConv2DLayer(quant_utils.QuantizableLayer):
  """Base class for 2D convolution layers.

  Has support for optional batch-normalization, activation and sequence
  padding.
  """

  @classmethod
  def Params(cls):
    p = super(BaseConv2DLayer, cls).Params()
    p.Define(
        'filter_shape', (0, 0, 0, 0),
        'Filter shape. Must be a sequence of length 4. Elements are in'
        ' the order of height (time), width (frequency), in_channel,'
        ' out_channel. When causal_convolution is True, filter_shape[1]'
        ' is the actual number of trained weights in the time dimension'
        ' of the kernel.')
    p.Define(
        'filter_stride', (0, 0),
        'Filter stride to use. Must be a pair of ints. The first int'
        ' specifies the stride on the time dimension. The second int'
        ' specifies the stride on the frequency dimension.')
    p.Define(
        'dilation_rate', (1, 1),
        'If > 1, dilation rate for atrous convolution. '
        'Must be a pair of ints. '
        'The first int specifies the dilation rate on the time dimension. '
        'The second int specifies the dilation rate on the frequency '
        'dimension. '
        'If any value of dilation_rate is > 1, then all values of strides '
        'must be 1.')
    p.Define(
        'activation', 'RELU',
        'Activation function to use. Options are RELU, RELU6, SIGMOID, '
        'TANH, NONE.')
    p.Define('bias', False, 'Whether or not to apply a bias before activation.')
    p.Define('batch_norm', True, 'Whether or not to apply batch norm.')
    p.Define(
        'bn_decay', 0.999,
        'Decay in updating the mean and variance moving average used in'
        ' batch normalization.')
    p.Define(
        'bn_fold_weights', None,
        'Fold the batch norm parameters into the convolution weights at '
        'eval/inference time as per https://arxiv.org/pdf/1712.05877.pdf. '
        'Requires that batch_norm be True and is incompatible with some other '
        'parameters (conv_last=True).')
    p.Define(
        'causal_convolution', False,
        'If true, conv layer output only depends on time steps in'
        ' the past.')
    p.Define(
        'conv_last', False,
        'If true, apply the convolution transformation as the last step, '
        'i.e., first apply batch normalization on the input, followed '
        'by activation, and finally the convolution. '
        'Otherwise, apply convolution first, followed by batch '
        'normalization and activation. Not compatible with bn_fold_weights '
        'or quantization.')
    p.Define(
        'weight_norm', False,
        'If true, apply weight normalization to weights as proposed by'
        ' Salimans and Kingma, 2016: https://arxiv.org/abs/1602.07868')
    p.Define(
        'disable_activation_quantization', False,
        'Disables the quantization tracking/clamping for the output '
        'activation. This is most often used in conjunction with a concat '
        'layer which needs to have a merged set of statistics.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseConv2DLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert len(p.dilation_rate) == 2
    assert all(x > 0 for x in p.filter_shape)
    assert all(x > 0 for x in p.filter_stride)
    assert all(x > 0 for x in p.dilation_rate)
    if any(x > 1 for x in p.dilation_rate):
      assert all(x == 1 for x in p.filter_stride)
    # Bias is not needed with batch_norm=True.
    if p.batch_norm:
      assert not p.bias
    assert (p.activation == 'NONE' or p.activation in _ACTIVATIONS)
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)
      if p.bias:
        self.CreateVariable(
            'b',
            py_utils.WeightParams(
                shape=[self.output_channels],
                init=py_utils.WeightInit.Constant(0.0),
                dtype=p.dtype,
                collections=[self.__class__.__name__ + '_vars']))
      if p.weight_norm:
        self.CreateVariable(
            'g',
            py_utils.WeightParams(
                shape=self.filter_output_shape,
                init=py_utils.WeightInit.Constant(0.0),
                dtype=p.dtype,
                collections=[self.__class__.__name__ + '_vars']))

    if not p.disable_activation_quantization:
      self.TrackQTensor('activation')
      if (p.activation not in _TFLITE_FUSED_ACTIVATION_NAMES and
          p.activation != 'NONE'):
        self.TrackQTensor('pre_activation')
    if p.batch_norm:
      # batch normalization dimension is number of input channels
      # (filter_shape[2]) if we apply batch_norm on input and convolution
      # in the end, number of output channels otherwise.
      bn_dim = p.filter_shape[2] if p.conv_last else self.output_channels
      bn_params = BatchNormLayer.Params().Set(
          dim=bn_dim, decay=p.bn_decay, name=p.name, params_init=p.params_init)
      self.CreateChild('bn', bn_params)

    if self._is_bn_folded:
      assert p.batch_norm, 'bn_fold_weights requires batch_norm = True'
      assert not p.conv_last, 'bn_fold_weights requires conv_last = False'

    # TODO(yonghui): implement the variational noise logic.

  @property
  def output_channels(self):
    """The number of output channels for this conv layer."""
    # Normal convolution filter shape is [..., out_channels].
    p = self.params
    return p.filter_shape[-1]

  @property
  def filter_output_shape(self):
    """Final dims of the filter corresponding to the output channels.

    Returns:
      A one (standard conv) or two (depthwise conv) element shape representing
      the final dimensions of the filter weights that are output channel
      specific for this layer. This shape is needed for any arithmetic that
      needs to convert between a linear list of filter weights and the
      arrangement in the actual filter.
    """
    # Standard convolution has all output channels in the last dim.
    p = self.params
    return [p.filter_shape[-1]]

  @property
  def _is_bn_folded(self):
    """Whether batchnorm folded weights are effectively enabled."""
    p = self.params
    if not p.batch_norm:
      return False
    return (p.bn_fold_weights or
            (p.bn_fold_weights is None and p.qdomain.default is not None))

  def _EvaluateConvKernel(self, inputs, filter_w, strides, dilation_rate,
                          padding_algorithm, data_format):
    """Evaluates the lower level convolution kernel.

    Args:
      inputs: As to tf.nn.convolution.
      filter_w: As to tf.nn.depthwise_conv2d.
      strides: As to tf.nn.convolution.
      dilation_rate: As to tf.nn.convolution.
      padding_algorithm: As to tf.nn.convolution (padding argument).
      data_format: As to tf.nn.convolution.

    Returns:
      Convolution kernel output.
    """
    raise NotImplementedError()

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    p = self.params
    return _ComputeConvOutputShape(in_shape, p.filter_stride[0],
                                   p.filter_stride[1], self.output_channels)

  def _GetWeights(self,
                  theta,
                  convolution_lambda,
                  folded_bn_padding,
                  cast_dtype=None):
    """Gets a dictionary of weights and biases for the convolution.

    This is necessary for some operating modes where the weights are fused
    with batch normalization differently for training vs eval.

    Args:
      theta: A `.NestedMap` object containing underlying weights values of this
        layer and its children layers.
      convolution_lambda: Lambda which takes the convolution weights and runs
        the convolution.
      folded_bn_padding: Padding to apply to folded batch normalization moment
        computation (or None for no padding).
      cast_dtype: If not None, cast weights to the given dtype.

    Returns:
      Tuple of (filter, biases).
    """
    p = self.params

    # Original weights.
    filter_w = theta.w
    filter_output_shape = self.filter_output_shape
    # TODO(miachen): remove casting once tf.nn.conv2d supports tf.float64.
    if cast_dtype:
      filter_w = tf.cast(filter_w, tf.float32)
    if p.weight_norm:
      if len(filter_output_shape) == 1:
        # Normalize along the last dim (standard conv).
        filter_w = tf.nn.l2_normalize(filter_w, [0, 1, 2]) * tf.reshape(
            (theta.g + 1.0), [1, 1, 1, p.filter_shape[-1]])
      elif len(filter_output_shape) == 2:
        # Normalize along the last two dimensions (depthwise conv).
        filter_w = tf.nn.l2_normalize(filter_w, [0, 1]) * tf.reshape(
            (theta.g + 1.0), [1, 1] + filter_output_shape)
      else:
        assert False, 'Unsupported weight norm filter shape'

    # Original bias.
    if p.bias:
      b = theta.b
    else:
      b = tf.zeros([self.output_channels], dtype=filter_w.dtype)

    # Pass-through if weights are not folded with batch normalization.
    if not self._is_bn_folded:
      return filter_w, b

    # If batch norm is fused with weights, then compute the weights as from
    # figure C.8 of https://arxiv.org/pdf/1712.05877.pdf for training and
    # figure C.6 for eval.
    if p.is_eval:
      # Gets current moments without updating.
      mean, variance, beta, gamma = self.bn.GetCurrentMoments(theta.bn)
    else:
      # Updates moments based on a trial run of the convolution.
      raw_conv_output = convolution_lambda(filter_w)
      mean, variance, beta, gamma = self.bn.ComputeAndUpdateMoments(
          theta.bn, raw_conv_output, folded_bn_padding)

    # Fold weights and bias. Note that this layer's bias is not used (not
    # applicable for batch norm case).
    sigma_recip = tf.rsqrt(variance + self.bn.epsilon)
    scale_correction = gamma * sigma_recip
    # Normal conv will have all weights in the last dim
    # ([_, _, _, output_channels]), which matches the 1D layout from
    # batch norm. Depthwise uses the last two dims so reshape
    # ([_, _, in_c, c_multiplier]).
    scale_correction = tf.reshape(scale_correction, filter_output_shape)
    filter_w = filter_w * scale_correction
    b = (beta - (gamma * mean * sigma_recip))
    return filter_w, b

  def _ApplyConv(self, theta, inputs, folded_bn_padding=None):
    p = self.params
    strides = [p.filter_stride[0], p.filter_stride[1]]
    dtype = inputs.dtype
    cast_dtype = None
    if dtype != tf.float32:
      cast_dtype = tf.float32
      inputs = tf.cast(inputs, cast_dtype)

    padding_algorithm = 'SAME'
    if p.causal_convolution:
      assert p.filter_shape[1] == 1, 'Only 1D causal convolutions supported.'
      # Use VALID padding and shift the inputs to the right to ensure that the
      # first output only depends on the first input and so on. The output is
      # the same size as the input, as if the convolution used SAME padding.
      padding_algorithm = 'VALID'
      # The effective spatial filter width for dilated convolutions is
      # (kernel_width - 1) * dilation_rate + 1 as according to
      # https://www.tensorflow.org/api_docs/python/tf/nn/convolution.
      causal_pad_size = (p.filter_shape[0] - 1) * p.dilation_rate[0]
      inputs = tf.pad(inputs, [[0, 0], [causal_pad_size, 0], [0, 0], [0, 0]])

    # Lambda for computing the actual convolution.
    def ComputeRawConvolution(filter_w):
      return self._EvaluateConvKernel(
          inputs,
          filter_w=filter_w,
          strides=strides,
          dilation_rate=p.dilation_rate,
          data_format='NHWC',
          padding_algorithm=padding_algorithm)

    filter_w, b = self._GetWeights(
        theta, ComputeRawConvolution, folded_bn_padding, cast_dtype=cast_dtype)

    # TODO(miachen): remove casting once tf.nn.conv2d supports tf.float64.
    assert inputs.dtype == filter_w.dtype

    filter_w = self.QWeight(filter_w)
    out = ComputeRawConvolution(filter_w)

    # Note that we always apply the bias (which may be zero) because some
    # normalization mechanisms do implicitly produce a bias.
    b = tf.cast(b, tf.float32)
    out = tf.nn.bias_add(out, b)

    if dtype != tf.float32:
      out = tf.cast(out, dtype)
    return out

  def FProp(self, theta, inputs, paddings=None):
    """Apply convolution to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch, time,
        frequency, channel]. The time dimension corresponds to the height
        dimension as in images and the frequency dimension corresponds to the
        width dimension as in images.
      paddings: The paddings tensor. If None, the inputs have no paddings in the
        sense of sequence training (e.g., in CNN models). Otherwise, it is
        expected to be of shape [batch, time].

    Returns:
      outputs, out_paddings pair.
    """
    p = self.params
    if paddings is None:
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(
              tf.shape(inputs), [-1, -1, -1, p.filter_shape[2]])
      ], inputs)
    else:
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(tf.shape(paddings), [-1, -1]),
          py_utils.assert_shape_match(
              tf.shape(inputs),
              tf.concat([tf.shape(paddings), [-1, p.filter_shape[2]]], 0))
      ], inputs)
      # Zeroing out padded inputs.
      qpadding = self.QRPadding(
          tf.expand_dims(tf.expand_dims(paddings, -1), -1))
      # Select based padding is required for quantized inference but is
      # causing regressions on other platforms. TODO: Remove use_select
      # attribute when root-caused/resolved.
      inputs = py_utils.ApplyPadding(
          qpadding,
          inputs,
          use_select=p.is_inference and p.qdomain.default is not None)

    with tf.name_scope(p.name):
      input_shape = tf.shape(inputs)

      if paddings is None:
        conv_padding = None
      else:
        # NOTE: this may be slightly inaccurate when p.dilation_rate[0] > 1.
        # But there's likely no real problems. Trying to set it gives an error:
        # pooling with SAME padding is not implemented for dilation_rate > 1.
        # NOTE: window=p.filter_stride[0] means output i will be padded if any
        # input in the stride between the two conv centers are padded.
        conv_padding = _ComputeConvOutputPadding(
            paddings, window=p.filter_stride[0], stride=p.filter_stride[0])

      if p.conv_last:
        out = self._ComputeConvLast(theta, inputs, paddings, conv_padding)
      else:
        out = self._Compute(theta, inputs, paddings, conv_padding)

      # Lastly zeroing out padded states.
      if conv_padding is not None:
        qpadding = self.QRPadding(
            tf.expand_dims(tf.expand_dims(conv_padding, -1), -1))
        # Select based padding is required for quantized inference but is
        # causing regressions on other platforms. TODO: Remove use_select
        # attribute when root-caused/resolved.
        out = py_utils.ApplyPadding(
            qpadding,
            out,
            use_select=p.is_inference and p.qdomain.default is not None)

      out = py_utils.HasShape(out, BaseConv2DLayer.OutShape(self, input_shape))
      return out, conv_padding

  def _Compute(self, theta, inputs, paddings, conv_padding):
    """Computes the forward prop (conv, bn, act)."""
    p = self.params

    bn_padding = conv_padding
    if bn_padding is None:
      bn_padding_expanded = None
    else:
      batch_time = tf.shape(bn_padding)
      batch_time_any_any = tf.concat([batch_time, [-1, -1]], 0)
      bn_padding_expanded = tf.reshape(bn_padding,
                                       tf.concat([batch_time, [1, 1]], 0))

    out = self._ApplyConv(theta, inputs, bn_padding_expanded)
    if bn_padding is not None:
      out = py_utils.with_dependencies([
          py_utils.assert_shape_match(batch_time, [-1, -1]),
          py_utils.assert_shape_match(tf.shape(out), batch_time_any_any)
      ], out)

    # Only apply batch norm if it was not folded into the weights.
    if p.batch_norm and not p.bn_fold_weights:
      out = self.bn.FProp(theta.bn, out, bn_padding_expanded)

    # Apply activation.
    if p.activation != 'NONE':
      if p.activation not in _TFLITE_FUSED_ACTIVATION_NAMES:
        out = self.QTensor('pre_activation', out)
      out = _ACTIVATIONS[p.activation](out)
    if not p.disable_activation_quantization:
      out = self.QTensor('activation', out)

    return out

  def _ComputeConvLast(self, theta, inputs, paddings, conv_padding):
    """Computes the forward prop in conv_last mode (bn, act, conv)."""
    p = self.params
    out = inputs
    out_padding = paddings

    if p.batch_norm:
      if out_padding is None:
        out_padding_expanded = None
      else:
        batch_time = tf.shape(out_padding)
        batch_time_any_any = tf.concat([batch_time, [-1, -1]], 0)
        out = py_utils.with_dependencies([
            py_utils.assert_shape_match(batch_time, [-1, -1]),
            py_utils.assert_shape_match(tf.shape(out), batch_time_any_any)
        ], out)
        out_padding_expanded = tf.reshape(out_padding,
                                          tf.concat([batch_time, [1, 1]], 0))
      out = self.bn.FProp(theta.bn, out, out_padding_expanded)

    if p.activation != 'NONE':
      out = _ACTIVATIONS[p.activation](out)

    out = self._ApplyConv(theta, out)

    return out


class Conv2DLayer(BaseConv2DLayer):
  """Convolution layer, with optional batch-normalization and activation."""

  def _EvaluateConvKernel(self, inputs, filter_w, strides, dilation_rate,
                          padding_algorithm, data_format):
    p = self.params
    return tf.nn.convolution(
        inputs,
        filter_w,
        strides=strides,
        dilation_rate=p.dilation_rate,
        data_format='NHWC',
        padding=padding_algorithm)


# Alias of Conv2DLayer (for compatibility with historical uses).
ConvLayer = Conv2DLayer


class DepthwiseConv2DLayer(BaseConv2DLayer):
  """Depthwise conv 2D layer.

  paper: https://arxiv.org/abs/1610.02357
  """

  @classmethod
  def Params(cls):
    p = super(DepthwiseConv2DLayer, cls).Params()
    # Redefine 'filter_shape' since the semantic of shape elements is different
    # from regular Conv2D.
    p.Delete('filter_shape')
    p.Define(
        'filter_shape', (0, 0, 0, 0),
        'Filter shape. Must be a sequence of length 4. Elements are in'
        ' the order of height (time), width (frequency), in_channel,'
        ' channel_multipliers. ')
    return p

  @property
  def output_channels(self):
    """The number of output channels for this conv layer."""
    p = self.params
    # Depthwise convolution filter shape is:
    #   [..., in_channels, channel_multiplier].
    return p.filter_shape[-2] * p.filter_shape[-1]

  @property
  def filter_output_shape(self):
    """Final dims of the filter corresponding to the output channels."""
    # Depthwise convolution uses the final two dims for output channels.
    p = self.params
    _, _, in_c, c_mul = p.filter_shape
    return [in_c, c_mul]

  def _EvaluateConvKernel(self, inputs, filter_w, strides, dilation_rate,
                          padding_algorithm, data_format):
    p = self.params
    return tf.nn.depthwise_conv2d(
        inputs,
        filter=filter_w,
        strides=[1, strides[0], strides[1], 1],
        rate=p.dilation_rate,
        data_format='NHWC',
        padding=padding_algorithm)


class SeparableConv2DLayer(Conv2DLayer):
  """Separable 2D convolution.

  This class aggregates a DepthwiseConv2DLayer that feeds in to the point
  wise convolution defined by this layer. Since the point wise convolution
  controls the output, this class is defined in terms of that and delegates
  to a depthwise sub-layer.

  The `filter_shape` parameter is rewritten on initialization from the form:
    (h, w, cin, cout)
  To:
    Depthwise filter: (h, w, cin, p.depth_multiplier)
    Pointwise filter (on this instance): (1, 1, cin * p.depth_multiplier, cout)

  This way, the layer is configured as if it were a normal 2D convolution
  but is internally reconfigured to be separable.

  paper: https://arxiv.org/abs/1610.02357
  """

  @classmethod
  def Params(cls):
    p = super(SeparableConv2DLayer, cls).Params()
    p.Define(
        'depth_multiplier', 1,
        'Number of depthwise convolution output channels per input channel. '
        'The total number of depthwise convolution output channels will be.'
        'equal to in_channel * depth_multiplier.')
    p.Define('depthwise_tpl',
             DepthwiseConv2DLayer.Params().Set(activation='NONE'),
             'Template for the depthwise conv sub-layer.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    # Rewrite the filter.
    params = params.Copy()
    h, w, cin, cout = params.filter_shape
    params.filter_shape = (1, 1, cin * params.depth_multiplier, cout)
    depthwise_filter_shape = (h, w, cin, params.depth_multiplier)

    # Dilation rate and stride go to the depthwise layer and reset ours.
    depthwise_filter_stride = params.filter_stride
    depthwise_dilation_rate = params.dilation_rate
    params.filter_stride = (1, 1)
    params.dilation_rate = (1, 1)

    super(SeparableConv2DLayer, self).__init__(params)
    p = self.params
    del params

    # Create the depthwise sub-layer.
    depthwise_params = p.depthwise_tpl.Copy().Set(
        filter_shape=depthwise_filter_shape,
        filter_stride=depthwise_filter_stride,
        dilation_rate=depthwise_dilation_rate,
        causal_convolution=p.causal_convolution,
        weight_norm=p.weight_norm,
        batch_norm=p.batch_norm,
        bn_decay=p.bn_decay,
        bn_fold_weights=p.bn_fold_weights)
    depthwise_params.qdomain.default = p.qdomain.default
    with tf.variable_scope(p.name):
      self.CreateChild('depthwise_conv', depthwise_params)

  def FProp(self, theta, inputs, paddings=None):
    inputs, paddings = self.depthwise_conv.FProp(theta.depthwise_conv, inputs,
                                                 paddings)
    return super(SeparableConv2DLayer, self).FProp(theta, inputs, paddings)

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    in_shape = self.depthwise_conv.OutShape(in_shape)
    return super(SeparableConv2DLayer, self).OutShape(in_shape)


class ProjectionLayer(quant_utils.QuantizableLayer):
  """Projection layer, with batch normalization and relu activation."""

  @classmethod
  def Params(cls):
    p = super(ProjectionLayer, cls).Params()
    p.Define('input_dim', 0, 'Depth of the input.')
    p.Define('output_dim', 0, 'Depth of the output.')
    p.Define(
        'activation', 'RELU',
        'Activation function to use. Options are RELU, RELU6, SIGMOID, '
        'TANH, NONE.')
    p.Define('batch_norm', True, 'Whether or not to apply batch norm.')
    p.Define('has_bias', False,
             'Whether or not to introduce the bias params to the layer.')
    p.Define('bias_init', 0.0, 'Initial value for the bias')
    p.Define(
        'affine_last', False,
        'If true, apply the affine transformation as the last step, i.e., '
        'first apply batch normalization on the input, followed '
        'by activation, and finally the affine transformation. '
        'Otherwise, apply affine transformation first, followed by batch '
        'normalization and activation.')
    p.Define(
        'weight_norm', False,
        'If true, apply weight normalization to weights as proposed by'
        ' Salimans and Kingma, 2016: https://arxiv.org/abs/1602.07868')
    p.Define(
        'bn_fold_weights', None,
        'Fold the batch norm parameters into the convolution weights at '
        'eval/inference time as per https://arxiv.org/pdf/1712.05877.pdf. '
        'Defaults to None which means that it will be disabled by default '
        'and enabled when quantized training is enabled. Not compatible with '
        'affine_last=True')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ProjectionLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim > 0
    assert p.output_dim > 0
    assert p.activation == 'NONE' or p.activation in _ACTIVATIONS
    if p.batch_norm and p.has_bias:
      tf.logging.warning(
          'Projection layer enables both batch_norm and has_bias. '
          'This is generally redundant/wasteful and may introduce '
          'accuracy problems in some inference scenarios.')
    if self._is_bn_folded:
      assert not p.affine_last, (
          'Folded batchnorm is not compatible with affine_last')
    w_pc = py_utils.WeightParams(
        shape=[p.input_dim, p.output_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    if p.has_bias:
      b_pc = py_utils.WeightParams(
          shape=[p.output_dim],
          init=py_utils.WeightInit.Constant(scale=p.bias_init),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
    if p.weight_norm:
      g_pc = py_utils.WeightParams(
          shape=[p.output_dim],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)
      if p.has_bias:
        self.CreateVariable('b', b_pc)
      if p.weight_norm:
        self.CreateVariable('g', g_pc)

    # Determine quantization needs based on whether fusing activation
    # or not.
    self._pre_activation_qt_name = None
    self._output_qt_name = ('activation'
                            if p.activation != 'NONE' else 'affine_matmul')
    if (p.activation != 'NONE' and
        p.activation not in _TFLITE_FUSED_ACTIVATION_NAMES):
      # Not a fused activation function.
      # Need a qtensor to track the pre-activation tensor. The name is
      # compatible with older checkpoints.
      self._pre_activation_qt_name = 'affine_matmul'
    self.TrackQTensor(self._output_qt_name)
    if self._pre_activation_qt_name:
      self.TrackQTensor(self._pre_activation_qt_name)

    if p.batch_norm:
      bn_params = BatchNormLayer.Params().Set(
          dim=p.input_dim if p.affine_last else p.output_dim,
          decay=0.999,
          name=p.name)
      self.CreateChild('bn', bn_params)
    # TODO(yonghui): implement the variational noise logic.

  @property
  def output_qt_name(self):
    """Name of QTensor used for the output value.

    Useful for grabbing the quantization of the output.

    Returns:
      String name of output qtensor.
    """
    return self._output_qt_name

  def FProp(self, theta, inputs, paddings=None):
    """Apply projection to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., input_dim].
      paddings: The paddings tensor.  Shaped [..., 1], where all but the last
        dimension match.

    Returns:
      Output after applying projection, and optionally batch normalization and
      relu non-linearity.
    """
    p = self.params
    with tf.name_scope(p.name):
      if paddings is None:
        paddings = tf.zeros(
            tf.concat([py_utils.GetShape(inputs)[:-1], [1]], axis=0),
            dtype=inputs.dtype)
      w, b = self._GetWeights(theta, inputs, paddings)
      w = self.QWeight(w)

      if p.affine_last:
        # Reversed computation. Does not handle folding.
        out = inputs
        if p.batch_norm:
          out = self.bn.FProp(theta.bn, out, paddings)
        if p.activation != 'NONE':
          if not p.is_inference:
            out = py_utils.CheckNumerics(out)
          out = _ACTIVATIONS[p.activation](out)
        out = self._ApplyProjectionKernel(w, b, out, with_activation=False)
      else:
        # Normal ordered projection.
        if self._is_bn_folded or not p.batch_norm:
          # Everything folded together. This is the only variant that supports
          # quantization.
          out = self._ApplyProjectionKernel(w, b, inputs, quant=True)
        else:
          # Projection kernel(no activation fn) -> BN -> Activation fn.
          out = self._ApplyProjectionKernel(w, b, inputs, with_activation=False)
          if p.batch_norm:
            out = self.bn.FProp(theta.bn, out, paddings)
          if p.activation != 'NONE':
            if not p.is_inference:
              out = py_utils.CheckNumerics(out)
            out = _ACTIVATIONS[p.activation](out)
      return py_utils.ApplyPadding(self.QRPadding(paddings), out)

  @property
  def _is_bn_folded(self):
    """Whether batchnorm folded weights are effectively enabled."""
    p = self.params
    if not p.batch_norm:
      return False
    return (p.bn_fold_weights or
            (p.bn_fold_weights is None and p.qdomain.default is not None))

  def _GetWeights(self, theta, inputs, paddings):
    """Gets the weights for the computation.

    Weights will always have weight_norm applied and may have batch_norm
    folded if enabled.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Inputs (needed for batchnorm folding).
      paddings: Paddings (needed for batchnorm folding).

    Returns:
      Tuple of (w, b) to use for the forward pass. b may be None if bias is
      disabled.
    """
    p = self.params
    w = theta.w
    b = theta.b if p.has_bias else None
    if p.weight_norm:
      w = tf.reshape((theta.g + 1.0) * tf.nn.l2_normalize(w, [0]),
                     [p.input_dim, p.output_dim])
    if not self._is_bn_folded:
      return w, b

    # If batch norm is fused with weights, then compute the weights as from
    # figure C.8 of https://arxiv.org/pdf/1712.05877.pdf for training and
    # figure C.6 for eval.
    if p.is_eval:
      # Gets current moments without updating.
      mean, variance, beta, gamma = self.bn.GetCurrentMoments(theta.bn)
    else:
      # Updates moments based on a trial run of the kernel (without activation
      # function).
      raw_output = self._ApplyProjectionKernel(
          w, b, inputs, with_activation=False)
      mean, variance, beta, gamma = self.bn.ComputeAndUpdateMoments(
          theta.bn, raw_output, paddings)

    # Fold weights and bias.
    sigma_recip = tf.rsqrt(variance + self.bn.epsilon)
    scale_correction = gamma * sigma_recip
    w = w * scale_correction
    b = beta - (gamma * mean * sigma_recip)
    return w, b

  def _ApplyProjectionKernel(self,
                             w,
                             b,
                             inputs,
                             with_activation=True,
                             quant=False,
                             bn=False):
    """Applies matmul/bias/activation in one step.

    Note that it is important that these three ops be computed in this way as
    downstream inference engines (esp. for quantized inference) can recognize
    and fuse them. For floating point, this is an optimization, but for
    quantization, it is required.

    Args:
      w: Weight matrix.
      b: Bias vector (or None).
      inputs: FProp inputs.
      with_activation: Whether to also compute the activation function.
      quant: Whether to apply quantization.
      bn: Apply batchnorm.

    Returns:
      Output tensor reshaped.
    """
    p = self.params
    out = py_utils.Matmul(tf.reshape(inputs, [-1, p.input_dim]), w)
    if b is not None:
      out += b  # NOTE: Bias on matmul is never quantized.
    if with_activation and p.activation != 'NONE':
      if self._pre_activation_qt_name:
        # Track quantization for unfused activation function.
        out = self.QTensor(self._pre_activation_qt_name, out)
      if not p.is_inference:
        out = py_utils.CheckNumerics(out)
      out = _ACTIVATIONS[p.activation](out)
    if quant:
      out = self.QTensor(self._output_qt_name, out)
    out = tf.reshape(
        out, tf.concat([py_utils.GetShape(inputs)[:-1], [p.output_dim]],
                       axis=0))
    return out


class FCLayer(ProjectionLayer):
  """Fully-connected layer (matmul + bias + optional activation)."""

  @classmethod
  def Params(cls):
    p = super(FCLayer, cls).Params()
    p.batch_norm = False
    p.has_bias = True
    return p


class PoolingLayer(quant_utils.QuantizableLayer):
  """Pooling layer, by default performs max-pooling.

  Quantization notes: Unlike the common pattern, the pooling layer inputs
  and output must be quantized to the same range, so it tracks both (vs
  just the output). The preceding layer must have its output quantization
  disabled.
  """

  @classmethod
  def Params(cls):
    p = super(PoolingLayer, cls).Params()
    p.Define(
        'window_shape', (0, 0),
        'Window shape. Must be a pair of ints. Elements are in'
        ' the order of height (time), width (frequency).')
    p.Define(
        'window_stride', (0, 0),
        'Window stride to use. Must be a pair of ints. The first int'
        ' specifies the stride on the time dimension. The second int'
        ' specifies the stride on the frequency dimension.')
    p.Define('pooling_type', 'MAX', 'Pooling type: MAX|AVG')
    p.Define(
        'padding_algorithm', 'SAME',
        'Padding algorithm. See the "returns" section of '
        '`tf.nn.convolution` for details. '
        'Roughly, VALID = NO_PADDING and SAME (default) = PAD INPUT')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(PoolingLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert len(p.window_shape) == 2
    assert len(p.window_stride) == 2
    assert all([x > 0 for x in p.window_shape])
    assert all([x > 0 for x in p.window_stride])
    assert p.pooling_type in ['MAX', 'AVG']
    self.TrackQTensor('output')

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    p = self.params
    return _ComputeConvOutputShape(
        in_shape,
        p.window_stride[0],
        p.window_stride[1],
        padding=p.padding_algorithm)

  def FProp(self, theta, inputs, paddings=None):
    """Apply pooling to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch, time,
        frequency, channel]. The time dimension corresponds to the height
        dimension as in images and the frequency dimension corresponds to the
        width dimension as in images.
      paddings: The paddings tensor. It is expected to be of shape [batch,
        time]. Defaults to None, which means there no paddings.

    Returns:
      outputs, out_paddings pair.
    """
    p = self.params
    stride = p.window_stride
    window = p.window_shape
    if paddings is not None:
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(tf.shape(paddings), [-1, -1]),
          py_utils.assert_shape_match(tf.shape(inputs)[:2], tf.shape(paddings))
      ], inputs)
    with tf.name_scope(p.name):
      if paddings is not None:
        out_padding = _ComputeConvOutputPadding(paddings, window[0], stride[0],
                                                p.padding_algorithm)
      else:
        out_padding = None
      inputs = self.QTensor('output', inputs)
      out = tf.nn.pool(
          inputs,
          window,
          p.pooling_type,
          strides=stride,
          padding=p.padding_algorithm,
          data_format='NHWC',
      )
      out = self.QTensor('output', out)
      if out_padding is not None:
        out *= tf.expand_dims(tf.expand_dims(1.0 - out_padding, -1), -1)
      return out, out_padding


class EmbeddingLayer(base_layer.BaseLayer):
  """Embedding layer."""

  @classmethod
  def Params(cls):
    p = super(EmbeddingLayer, cls).Params()
    p.Define('vocab_size', 0, 'Depth of the input.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define('max_num_shards', 0, 'Num param shards.')
    p.Define('on_ps', True, 'True if to perform the embedding lookup on ps.')
    p.Define(
        'scale_sqrt_depth', False, 'If set True, activations are scaled'
        ' with sqrt(embedding_dim) in EmbLookup.')
    return p

  # Min number of params per shard.
  MIN_PARAMS_PER_SHARD = 1024 * 256

  @base_layer.initializer
  def __init__(self, params):
    super(EmbeddingLayer, self).__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert p.embedding_dim > 0
    assert p.max_num_shards > 0
    assert p.name

    total_size = p.vocab_size * p.embedding_dim
    actual_shards = min(
        p.max_num_shards,
        int(math.ceil(float(total_size) / self.MIN_PARAMS_PER_SHARD)))
    self._ids_per_shard = int(math.ceil(float(p.vocab_size) / actual_shards))

    w_pc = py_utils.WeightParams(
        shape=[self._ids_per_shard, p.embedding_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    # EmbeddingLayer handles vars/theta differently from other layers
    # because when embedding shards are placed on ps, it's more
    # efficiently to do embedding looups on ps and sends the result
    # back to the worker.
    self._vars = []
    self._emb_shards = []
    with tf.variable_scope(p.name):
      for i in range(actual_shards):
        vi, vi_var = py_utils.CreateVariable('var_%d' % i, w_pc)
        self._vars.append(vi_var)
        if p.on_ps:
          v = vi_var
        else:
          v = tf.identity(vi)
        if p.fprop_dtype is not None and p.fprop_dtype != p.dtype:
          v = tf.cast(v, p.fprop_dtype)
        self._emb_shards.append(v)

    self._vars_map = py_utils.NestedMap(wm=self._vars)
    self._theta_map = py_utils.NestedMap(wm=self._emb_shards)

  @property
  def vars(self):
    return self._vars_map

  @property
  def theta(self):
    return self._theta_map

  def EmbLookupDefaultTheta(self, ids):
    return self.EmbLookup(self.theta, ids)

  def EmbLookup(self, theta, ids):
    """Looks up embedding vectors for ids.

    Args:
      theta: Named tuple with the weight matrix for the embedding.
      ids: A rank-N int32 tensor.

    Returns:
      A rank-(N+1) params.dtype tensor.
      embs[indices, :] is the embedding vector for ids[indices].
    """
    p = self.params
    ids = tf.convert_to_tensor(ids)
    ids = py_utils.with_dependencies(
        [py_utils.assert_between(ids, 0, p.vocab_size)], ids)
    embs = tf.nn.embedding_lookup(theta.wm, tf.reshape(ids, [-1]))
    if p.scale_sqrt_depth:
      embs *= p.embedding_dim**0.5
    if p.vn.global_vn or p.vn.per_step_vn:
      embs = py_utils.AddGlobalVN(p, embs)
    out_shape = tf.concat([tf.shape(ids), [p.embedding_dim]], 0)
    return tf.reshape(embs, out_shape)


class SimpleEmbeddingLayer(quant_utils.QuantizableLayer):
  """An embedding layer that is simple to compile (by XLA and Toco).

  The params use_matmul and use_gather control how the lookup is performed.
  If neither is True, then a loop is used to compute the embedding.

  This layer is "simple" in comparison to 'EmbeddingLayer' in that it does
  not shard the embeddings.
  """

  @classmethod
  def Params(cls):
    p = super(SimpleEmbeddingLayer, cls).Params()
    p.Define('vocab_size', 0,
             'Depth of the input. I.e., the number of classes.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define(
        'use_matmul', False, 'If True, use a matmul to implement '
        'the embedding lookup. Depending on vocab_size and #ids, '
        'e.g., when vocab_size is small, use_matmul can be more '
        'efficient. On the other hand, use_matmul creates a 0/1 '
        'sparse matrix and hence may use more memory than the '
        'final output.')
    p.Define(
        'fprop_mode', None, 'Sets the mode used for computing the fprop '
        '(different inference engines have different capabilities and this '
        'accomodates them). Can be "loop", "matmul" or "gather". If None, '
        'defaults to "matmul" if use_matmul or "loop" if false.')
    p.Define(
        'use_3d_weight_tensor', False, 'If True, and use_matmul is False,'
        'in TPU compatibility mode, we reshape the normal 2D weight'
        'tensor to [num_rows, embed_dim] to be '
        '[num_rows, embed_dim // 128, 128].')
    p.Define('apply_pruning', False,
             'Whether to prune the weights while training')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SimpleEmbeddingLayer, self).__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert p.embedding_dim > 0

    valid_fprop_modes = ['loop', 'matmul', 'gather']
    self._fprop_mode = p.fprop_mode
    if not self._fprop_mode:
      self._fprop_mode = 'matmul' if p.use_matmul else 'gather'
    assert self._fprop_mode in valid_fprop_modes, (
        'fprop_mode must be one of %r' % valid_fprop_modes)

    if py_utils.tpu_compat() and self._fprop_mode != 'matmul':
      if p.use_3d_weight_tensor:
        assert p.embedding_dim % 128 == 0
        emb_shape_suf = [p.embedding_dim // 128, 128]
      else:
        emb_shape_suf = [p.embedding_dim]
    else:
      emb_shape_suf = [p.embedding_dim]
    weight_shape = [p.vocab_size] + emb_shape_suf

    with tf.variable_scope(p.name):
      # Define weights
      pc = py_utils.WeightParams(
          shape=weight_shape,
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])

      if p.apply_pruning:
        mask_pc = py_utils.WeightParams(pc.shape,
                                        py_utils.WeightInit.Constant(1.0),
                                        p.dtype)
        threshold_pc = py_utils.WeightParams([],
                                             py_utils.WeightInit.Constant(0.0),
                                             tf.float32)
        self.CreateVariable('mask', mask_pc, theta_fn=None, trainable=False)
        self.CreateVariable(
            'threshold', threshold_pc, theta_fn=None, trainable=False)

        def MaskWeightFn(weight):
          return tf.multiply(
              self.AddGlobalVN(weight), self.vars.mask, 'masked_weights')

        self.CreateVariable('wm', pc, theta_fn=MaskWeightFn)
        py_utils.AddToPruningCollections(self.vars.wm, self.vars.mask,
                                         self.vars.threshold)

      else:
        self.CreateVariable('wm', pc)

    # flags passed to @function.Defun
    compiled = py_utils.use_xla()

    @function.Defun(p.dtype, tf.int32, p.dtype)
    def EmbBprop(embs, ids_vec, drets):
      """Embedding backprop.

      Effectively, it computes:
        num = size of ids_vec
        dembs = zeros_like(embs)
        for i in range(num):
          dembs[ids_vec[i], :] += drets[i, :]
        return dembs, zeros_like(ids_vec)

      Args:
        embs: The embedding matrix. Unused in the backprop.
        ids_vec: A vector of int32 embedding ids.
        drets: A matrix of size (size of ids_vec, embedding dims).

      Returns:
        dembs: A matrix of the same shape of embs. Gradients for embs.
        dids_vec: Zeros. Same shape as ids_vec.
      """
      del embs
      num = tf.shape(ids_vec)[0]
      dembs = inplace_ops.empty(weight_shape, p.dtype, init=True)
      if len(weight_shape) != 2:
        drets_shape = tf.shape(drets)
        drets = tf.reshape(drets, [drets_shape[0]] + emb_shape_suf)

      @function.Defun(tf.int32, tf.int32, p.dtype, p.dtype)
      def EmbBpropLoop(i, ids_vec, drets, dembs):
        # row_id = ids_vec[i]
        row_id = tf.gather(ids_vec, i)
        # row = drets[i]
        row = tf.reshape(tf.gather(drets, i), [1] + emb_shape_suf)
        # dembs[row_id] = row
        dembs = inplace_ops.alias_inplace_add(dembs, [row_id], row)
        return ids_vec, drets, dembs

      _, _, dembs = functional_ops.For(
          start=0,
          limit=num,
          delta=1,
          inputs=[ids_vec, drets, dembs],
          body=EmbBpropLoop,
          rewrite_with_while=compiled)
      return dembs, tf.zeros_like(ids_vec)

    @function.Defun(p.dtype, tf.int32, grad_func=EmbBprop)
    def EmbFprop(embs, ids_vec):
      """Embedding forward prop.

      Effectively, it computes:
        num = size of ids_vec
        rets = zeros([num, embedding dim])
        for i in range(num):
          rets[i, :] = embs[ids_vec[i], :]
        return rets

      Args:
        embs: The embedding matrix.
        ids_vec: A vector of int32 embedding ids.

      Returns:
        rets: The result of embedding lookups. A matrix of shape
          (num ids in ids_vec, embedding dims).
      """
      num = tf.shape(ids_vec)[0]
      rets = inplace_ops.empty([num] + emb_shape_suf, p.dtype)

      @function.Defun(tf.int32, p.dtype, tf.int32, p.dtype)
      def EmbFpropLoop(i, embs, ids_vec, rets):
        # row_id = ids_vec[i]
        row_id = tf.gather(ids_vec, i)
        # row = embs[row_id]
        row = tf.reshape(tf.gather(embs, row_id), [1] + emb_shape_suf)
        # rets[i] = row
        rets = inplace_ops.alias_inplace_update(rets, [i], row)
        return embs, ids_vec, rets

      _, _, rets = functional_ops.For(
          start=0,
          limit=num,
          delta=1,
          inputs=[embs, ids_vec, rets],
          body=EmbFpropLoop,
          rewrite_with_while=compiled)
      if len(weight_shape) > 2:
        rets = tf.reshape(rets, [num, p.embedding_dim])
      return rets

    def EmbMatmul(embs, ids_vec):
      # lhs[i, j] is True iff ids_vec[i] == j.
      lhs = tf.equal(
          tf.expand_dims(ids_vec, 1), tf.range(
              p.vocab_size, dtype=ids_vec.dtype))
      return tf.matmul(tf.cast(lhs, embs.dtype), embs)

    def EmbGather(embs, ids_vec):
      return tf.nn.embedding_lookup(embs, ids_vec)

    if self._fprop_mode == 'matmul':
      self._fprop = EmbMatmul
    elif self._fprop_mode == 'loop':
      self._fprop = EmbFprop
    elif self._fprop_mode == 'gather':
      self._fprop = EmbGather

  def EmbLookupDefaultTheta(self, ids):
    """Lookups embedding vectors for ids."""
    return self.FProp(self.theta, ids)

  def EmbLookup(self, theta, ids):
    return self.FProp(theta, ids)

  def EmbLookupDefaultThetaOnCpu(self, ids):
    """A faster path for CPU inference than the default gather."""
    p = self.params
    embs = tf.nn.embedding_lookup(self.theta.wm, tf.reshape(ids, [-1]))
    out_shape = tf.concat([tf.shape(ids), [p.embedding_dim]], 0)
    return tf.reshape(embs, out_shape)

  def FProp(self, theta, ids):
    """Lookups embedding vectors for ids.

    Args:
      theta: Named tuple collection of weights for the layer.
      ids: A rank-N int32 tensor.

    Returns:
      A rank-(N+1) params.dtype tensor.
      embs[indices, :] is the embedding vector for ids[indices].
    """
    p = self.params
    if not py_utils.use_xla():
      ids = py_utils.with_dependencies(
          [py_utils.assert_between(ids, 0, p.vocab_size)], ids)
    embs_result = self._fprop(self.QWeight(theta.wm), tf.reshape(ids, [-1]))
    if p.vn.global_vn or p.vn.per_step_vn:
      emb_noise = p.vn.scale * tf.random_normal(
          tf.shape(embs_result), stddev=1.0, dtype=embs_result.dtype)
      embs_result += emb_noise
    out_shape = tf.concat([tf.shape(ids), [p.embedding_dim]], 0)
    return tf.reshape(embs_result, out_shape)


class PositionalEmbeddingLayer(base_layer.BaseLayer):
  """Generates sinusoidals with respect to the position in time and dimension.

  Implements the positional embedding layer from 'Attention is All You Need',
  the Transformer Network.

  Code and comments are adapted from tensor2tensor/layers/common_attention.py
  """

  @classmethod
  def Params(cls):
    p = super(PositionalEmbeddingLayer, cls).Params()
    p.Define(
        'min_timescale', 1, 'Start of the geometric index.'
        'Determines the periodicity of the added signal.')
    p.Define(
        'max_timescale', 10000, 'End of the geometric index. '
        'Determines the frequency of the added signal.')
    p.Define('embedding_dim', 0, 'Dimension of the embedding to be generated.')
    p.Define(
        'trainable_scaling', False,
        'Introduces a trainable scaling parameter (a scalar) that'
        ' multiplies the positional embedding in FProp.')
    p.Define('trainable_scaling_init', 1.0,
             'Initial value of the scaling parameter.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(PositionalEmbeddingLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.min_timescale
    assert p.max_timescale
    assert p.embedding_dim % 2 == 0
    if p.trainable_scaling:
      pc = py_utils.WeightParams(
          shape=[1],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      with tf.variable_scope(p.name):
        self.CreateVariable('scale', pc)

  def _PosEmbeddingsFromPositions(self, theta, position):
    """Generates the positional embeddings given the position tensor.

    Factors out the common code from FProp and FPropWithPosition. Returns
    positional embeddings corresponding to the input position tensor.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      position: Position tensor of dtype float and shape [bs, seq_length] to
        generate positional embeddings.

    Returns:
      a Tensor of shape [bs, seq_length, embedding_dim].
    """
    p = self.params
    seq_length = tf.shape(position)[1]
    num_timescales = p.embedding_dim // 2
    log_timescale_increment = (
        math.log(float(p.max_timescale) / float(p.min_timescale)) /
        (tf.cast(num_timescales, py_utils.FPropDtype(p)) - 1))

    inv_timescales = p.min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), py_utils.FPropDtype(p)) *
        -log_timescale_increment)

    scaled_time = tf.expand_dims(position, 2) * tf.reshape(
        inv_timescales, [1, 1, -1])

    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
    signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(p.embedding_dim, -1)]])
    signal = tf.reshape(signal, [-1, seq_length, p.embedding_dim])
    if p.trainable_scaling:
      signal *= (p.trainable_scaling_init + theta.scale)
    return signal

  def FProp(self, theta, seq_length):
    """Generates a Tensor of sinusoids with different frequencies.

    Each channel (dimension) of the generated positionanl embedding Tensor
    corresponds to a sinusoid of different frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can
    be experessed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels (dimension) / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      seq_length: Sequence length of the embeddings to be generated

    Returns:
      a Tensor of shape [seq_length, embedding_dim].
    """
    p = self.params
    position = tf.reshape(
        tf.cast(tf.range(seq_length), py_utils.FPropDtype(p)), [1, seq_length])
    pos_emb = self._PosEmbeddingsFromPositions(theta, position)
    return tf.reshape(pos_emb, [seq_length, -1])

  def FPropWithPosition(self, theta, position_tensor):
    """Generates a Tensor of sinusoids with different frequencies.

    Uses the provided position tensor to generate positional embeddings. Refer
    to FProp description for details of sinusoidal positional embeddings.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      position_tensor: Position tensor of shape [bs, seq_length] to generate
        positional embeddings.

    Returns:
      a Tensor of shape [bs, seq_length, embedding_dim].
    """
    position = tf.cast(position_tensor, py_utils.FPropDtype(self.params))
    return self._PosEmbeddingsFromPositions(theta, position)


class SoftmaxLayer(quant_utils.QuantizableLayer):
  """Base class for softmax layers."""

  @classmethod
  def Params(cls):
    """Params for SoftmaxLayer."""
    p = super(SoftmaxLayer, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the input.')
    p.Define('num_classes', 0, 'Total number of target classes.')
    p.Define(
        'logits_abs_max', None, 'If not None, logits are clipped to be within'
        ' [-logits_abs_max, logits_abs_max]. This can be a scalar'
        ' or a scalar tensor. Applies back pressure at training time; ignored'
        ' for inference.')
    p.Define(
        'chunk_size', 0, 'If non-zero, computes the per example '
        'xent by small chunks along the batch dimension.')
    return p

  def Logits(self, **unused):
    """Returns the logits computed before the softmax."""
    raise NotImplementedError('GetLogits is not implemented.')

  def XentLoss(self, *args, **kwargs):
    """Computes cross entropy."""
    return self.FProp(self.theta, *args, **kwargs)

  def _FProp2D(self,
               theta,
               inputs,
               class_weights,
               class_ids=None,
               class_probabilities=None):
    """Specialized FProp for matrix inputs."""
    raise NotImplementedError(
        'Subclasses of SoftmaxLayer must implement _FProp2D')

  def FProp(self,
            theta,
            inputs,
            class_weights,
            class_ids=None,
            class_probabilities=None):
    """Computes logit, cross entropy etc.

    This function can both work with class_ids, or probability distributions
    over classes. Exactly one of class_ids or class_probabilities must be
    provided.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: a list of a single tensor, or a single tensor with the shape [...,
        input_dim].
      class_weights: a tensor with shape [...] containing the weights for each
        target word.
      class_ids: a tensor with shape [..., 1] of int32 dtype containing the
        target class labels.
      class_probabilities: a tensor with shape [..., num_classes] of float
        values indicating class-membership probabilities.

    Returns:
      A `.NestedMap` containing the following fields

      - logits: with shape [..., num_classes]. Unnormalized softmax's logits.
      - per_example_argmax: with shape [...]. argmax of i-th example.
      - per_example_xent: with shape [...]. Cross entropy between i-th example's
        prediction and its label.
      - per_example_weight: with shape [...]. class_weights casted to
        this layer's dtype.
      - total_xent: A scalar. The sum of per_example_weight * per_example_xent.
      - total_weight: A scalar. The sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
    """
    p = self.params

    # Consolidate list/single value into a list.
    if not isinstance(inputs, list):
      inputs = [inputs]

    # If inputs are matrices already, delegate to _FProp2D.
    if inputs[0].shape.ndims == 2:
      return self._FProp2D(theta, inputs, class_weights, class_ids,
                           class_probabilities)

    # Remembers the original shape[1:-1].
    shape_mid = tf.shape(inputs[0])[1:-1]

    # Reshape inputs to matrices, labels to vectors, etc.
    inputs = [tf.reshape(x, [-1, p.input_dim]) for x in inputs]
    class_weights = tf.reshape(class_weights, [-1])
    if class_ids is not None:
      class_ids = tf.reshape(class_ids, [-1, 1])
    if class_probabilities is not None:
      class_probabilities = tf.reshape(class_probabilities, [-1, p.num_classes])

    # Delegates to _FProp2D.
    xent_loss = self._FProp2D(theta, inputs, class_weights, class_ids,
                              class_probabilities)

    # Reshapes xent_loss fields according to the inputs' shape.
    xent_loss.logits = tf.reshape(
        xent_loss.logits, tf.concat([[-1], shape_mid, [p.num_classes]], axis=0))
    per_example_shape = tf.concat([[-1], shape_mid], axis=0)
    xent_loss.per_example_argmax = tf.reshape(xent_loss.per_example_argmax,
                                              per_example_shape)
    xent_loss.per_example_xent = tf.reshape(xent_loss.per_example_xent,
                                            per_example_shape)
    xent_loss.per_example_weight = tf.reshape(xent_loss.per_example_weight,
                                              per_example_shape)
    return xent_loss


class SimpleFullSoftmax(SoftmaxLayer):
  """A somewhat simple softmax layer."""

  @classmethod
  def Params(cls):
    """Params for SimpleFullSoftmax."""
    p = super(SimpleFullSoftmax, cls).Params()
    p.Define(
        'num_sampled', 0, 'Number of samples to use for the sampled soft-max. '
        'Default value of 0 means no sampling is done; if set to > 0 then '
        'training will use sampled soft-max when both chunk_size == 0 and '
        'FProp is called with class_probabilities=None.')
    p.Define(
        'num_shards', 1,
        'Number of shards to split params into. num_shards should'
        ' divide num_classes.')
    p.Define('apply_pruning', False,
             'Whether to prune the weights while training')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a SimpleFullSoftmax layer."""
    super(SimpleFullSoftmax, self).__init__(params)
    p = self.params
    assert p.name
    # We shard params across the class dimension.
    assert p.num_classes % p.num_shards == 0
    num_classes_per_shard = p.num_classes // p.num_shards
    # When using sampled soft-max we'd rather work with weights of
    # shape=[num_classes_per_shard, p.input_dim] to avoid an expensive transpose
    # op before computing the sampled_softmax_loss.
    self._transpose_weight_params = False
    self._weights_shard_shape = [p.input_dim, num_classes_per_shard]
    if p.num_sampled:
      self._transpose_weight_params = True
      self._weights_shard_shape = [num_classes_per_shard, p.input_dim]
    self.TrackQTensor('inputs', 'logits')

    with tf.variable_scope(p.name):
      pc = py_utils.WeightParams(
          shape=self._weights_shard_shape,
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])

      if p.apply_pruning:
        mask_pc = py_utils.WeightParams(pc.shape,
                                        py_utils.WeightInit.Constant(1.0),
                                        p.dtype)
        threshold_pc = py_utils.WeightParams([],
                                             py_utils.WeightInit.Constant(0.0),
                                             tf.float32)

      for i in range(p.num_shards):
        weights_var_name = 'weight_%d' % i
        if p.apply_pruning:
          mask_var_name = 'mask_%d' % i
          threshold_var_name = 'threshold_%d' % i
          self.CreateVariable(
              mask_var_name, mask_pc, theta_fn=None, trainable=False)
          self.CreateVariable(
              threshold_var_name, threshold_pc, theta_fn=None, trainable=False)

          def MaskWeightFn(weight):
            return tf.multiply(
                self.AddGlobalVN(weight), getattr(self.vars, mask_var_name),
                'masked_weights')

          self.CreateVariable(weights_var_name, pc, theta_fn=MaskWeightFn)
          py_utils.AddToPruningCollections(
              getattr(self.vars, weights_var_name),
              getattr(self.vars, mask_var_name),
              getattr(self.vars, threshold_var_name))

        else:
          self.CreateVariable(weights_var_name, pc, self.AddGlobalVN)

      pc = py_utils.WeightParams(
          shape=[num_classes_per_shard],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      for i in range(p.num_shards):
        self.CreateVariable('bias_%d' % i, pc, self.AddGlobalVN)

  def _GetInputs(self, inputs):
    if isinstance(inputs, list):
      assert len(inputs) == 1
      return inputs[0]
    return inputs

  def _ConcatWeights(self, theta):
    p = self.params
    # Add per-step noise if configured so.
    concat_axis = 1
    if self._transpose_weight_params:
      concat_axis = 0
    weights = [
        self.QWeight(theta['weight_%d' % i]) for i in range(p.num_shards)
    ]
    biases = [self.QWeight(theta['bias_%d' % i]) for i in range(p.num_shards)]
    new_theta = theta.copy()
    new_theta.wm = py_utils.AddPerStepVN(p, tf.concat(
        weights, axis=concat_axis))
    new_theta.bias = py_utils.AddPerStepVN(p, tf.concat(biases, axis=0))
    return new_theta

  def _LogitsUsingConcatenatedWeights(self, theta, inputs):
    p = self.params
    inputs = self.QTensor('inputs', inputs)
    wm = self.QWeight(theta.wm)
    bias = self.QWeight(theta.bias)

    # x * w + b
    # Note that theta.wm and theta.bias are transformed to concated/clipped
    # by caller.
    logits = tf.nn.bias_add(
        py_utils.Matmul(inputs, wm, transpose_b=self._transpose_weight_params),
        bias)

    # Clip logits by range.
    # Note that this is generally not used in conjunction with quantization and
    # shouldn't be needed at inference time as the quantized matmul above will
    # take care of clipping naturally based on the data type and qparams.
    abs_max = p.logits_abs_max
    if abs_max is not None and not p.is_inference:
      abs_min = -abs_max  # pylint: disable=invalid-unary-operand-type
      logits = py_utils.clip_by_value(logits, abs_min, abs_max)

    return self.QTensor('logits', logits)

  def Logits(self, theta, inputs):
    """Returns the logits computed before the softmax.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: a list of a single tensor, or a single tensor with the shape [N,
        input_dim].

    Returns:
      logits [batch, num_classes]
    """
    return self._LogitsUsingConcatenatedWeights(
        self._ConcatWeights(theta), self._GetInputs(inputs))

  def _XentLossByChunk(self, theta, activation, class_ids):
    """Computes per-example xent loss between activation and class_ids."""
    p = self.params

    # We reshape activation from a matrix to a 3-D tensor (a sequence
    # of matrices), where the 2nd dimenion is p.chunk_size.  Because
    # the batch dimenion may not be multiple of p.chunk_size, we pad
    # zeros.
    activation = py_utils.HasRank(activation, 2)
    batch, input_dim = tf.unstack(tf.shape(activation))
    dim0, dim1 = (batch + p.chunk_size - 1) // p.chunk_size, p.chunk_size
    pad = dim0 * dim1 - batch
    padded_activation = tf.concat(
        [activation,
         tf.zeros([pad, input_dim], dtype=activation.dtype)],
        axis=0)
    class_ids = py_utils.HasShape(class_ids, [batch, 1])
    padded_class_ids = tf.concat(
        [class_ids, tf.zeros([pad, 1], dtype=class_ids.dtype)], axis=0)

    if py_utils.use_tpu():
      id_dtype = tf.int32
    else:
      id_dtype = tf.int64
    padded_class_ids = tf.cast(padded_class_ids, id_dtype)

    # For each chunk, we compute logits of padded_activation[i, :, :],
    # and its xent loss with padded_class_ids[i, :].
    def ChunkFn(theta, state0, inputs):
      del state0
      activation, class_ids = inputs.activation, inputs.class_ids
      logits = self._LogitsUsingConcatenatedWeights(theta, activation)
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=class_ids)
      amax = tf.stop_gradient(py_utils.ArgMax(logits))
      return py_utils.NestedMap(xent=xent, amax=amax), py_utils.NestedMap()

    acc, _ = recurrent.Recurrent(
        theta=self._ConcatWeights(theta),
        state0=py_utils.NestedMap(
            xent=tf.zeros([p.chunk_size], dtype=p.dtype),
            amax=tf.zeros([p.chunk_size], dtype=id_dtype)),
        inputs=py_utils.NestedMap(
            activation=tf.reshape(padded_activation, [dim0, dim1, input_dim]),
            class_ids=tf.reshape(padded_class_ids, [dim0, dim1])),
        cell_fn=ChunkFn)

    # acc.xent has the shape [dim0, dim1]. acc.xent[i, :] are
    # per-example xent loss for examples in the i-th chunk.  We
    # reshape acc.xent to a vector and slice the first 'batch' values.
    def GetBatch(x):
      return tf.reshape(x, [-1])[:batch]

    return GetBatch(acc.xent), GetBatch(acc.amax)

  def _FProp2D(self,
               theta,
               inputs,
               class_weights,
               class_ids=None,
               class_probabilities=None):
    """Computes xent loss and log-prob logit."""
    p = self.params
    inputs = self._GetInputs(inputs)
    logits = self.Logits(theta, inputs)

    if class_probabilities is not None:
      per_example_xent = tf.nn.softmax_cross_entropy_with_logits(
          labels=class_probabilities, logits=logits)
      per_example_argmax = py_utils.ArgMax(logits)
    elif p.chunk_size:
      class_ids = py_utils.HasShape(class_ids, [-1, 1])
      per_example_xent, per_example_argmax = self._XentLossByChunk(
          theta, inputs, class_ids)
    elif p.num_sampled == 0 or p.is_eval:
      assert class_ids is not None
      assert logits is not None
      tf.logging.vlog(
          0, 'Using sparse_softmax_cross_entropy_with_logits() in '
          'SimpleFullSoftmax::_FProp2D logits_shape=%r',
          py_utils.GetShape(logits))
      per_example_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.reshape(class_ids, [-1]), logits=logits)
      per_example_argmax = py_utils.ArgMax(logits)
    else:  # Use sampled soft-max in training mode with p.num_sampled set.
      assert p.num_sampled > 0
      tf.logging.vlog(
          0, 'Using sampled_softmax_loss(..., num_sampled=%d, '
          'num_classes=%d) in SimpleFullSoftmax::_FProp2D', p.num_sampled,
          p.num_classes)
      per_example_xent = tf.nn.sampled_softmax_loss(
          weights=[theta['weight_%d' % i] for i in range(p.num_shards)],
          biases=tf.concat([theta['bias_%d' % i] for i in range(p.num_shards)],
                           axis=0),
          labels=tf.reshape(class_ids, [-1, 1]),
          inputs=self._GetInputs(inputs),
          num_sampled=p.num_sampled,
          num_classes=p.num_classes,
          partition_strategy='div',
          seed=p.random_seed)
      # Avoid computing logits; per_example_argmax is going to be always right.
      per_example_argmax = tf.identity(class_ids)

    label_weights = tf.reshape(
        tf.cast(class_weights, py_utils.FPropDtype(p)), [-1])
    total_xent = tf.reduce_sum(per_example_xent * label_weights)
    total_weights = tf.reduce_sum(label_weights)

    return py_utils.NestedMap(
        logits=logits,
        log_probs=tf.nn.log_softmax(logits),
        per_example_argmax=per_example_argmax,
        per_example_xent=per_example_xent,
        per_example_weight=label_weights,
        total_xent=total_xent,
        total_weight=total_weights,
        avg_xent=total_xent / total_weights)


class ConvSoftmax(quant_utils.QuantizableLayer):
  """A softmax implementation based on 1x1 convolution.

  On TPU this is much more memory efficient than MatMul after reshaping logits
  to a matrix.
  """

  @classmethod
  def Params(cls):
    """Params for SoftmaxLayer."""
    p = super(ConvSoftmax, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the input.')
    p.Define('hidden_dim', 0, 'Dimension of the hidden layer.')
    p.Define('num_classes', 0, 'Total number of target classes.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a SimpleFullSoftmax layer."""
    super(ConvSoftmax, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      if p.hidden_dim:
        w_proj_pc = py_utils.WeightParams(
            shape=(1, p.input_dim, p.hidden_dim),
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('w_proj', w_proj_pc)
      w_pc = py_utils.WeightParams(
          shape=(1, p.hidden_dim or p.input_dim, p.num_classes),
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      self.CreateVariable('w', w_pc)
      self.CreateVariable(
          'b',
          py_utils.WeightParams(
              shape=[p.num_classes],
              init=py_utils.WeightInit.Constant(0.0),
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']))

  def Logits(self, theta, inputs):
    p = self.params
    with tf.name_scope(p.name):
      if inputs.shape.ndims == 2:
        # [batch, time, depth]
        x = inputs[:, tf.newaxis, :]
      else:
        x = py_utils.HasShape(inputs, [-1, -1, -1])
      if p.hidden_dim:
        x = tf.nn.conv1d(x, theta.w_proj, 1, 'VALID')
      logits = tf.nn.bias_add(tf.nn.conv1d(x, theta.w, 1, 'VALID'), theta.b)
      if inputs.shape.ndims == 2:
        return logits[:, 0, :]
      else:
        return logits


class FeedForwardNet(quant_utils.QuantizableLayer):
  """A simple multiple layer feedforward network.

  This class represents a stack of fully connected feedforward network. Each
  layer in the network can be configured for whether or not to have batch-norm
  applied to its output, its activation function, whether or not to apply
  dropout to post-activation output.
  """

  @classmethod
  def Params(cls):
    p = super(FeedForwardNet, cls).Params()
    p.Define('input_dim', 0, 'Depth of the input to the network.')
    p.Define('hidden_layer_dims', [], 'Depth of the hidden layer outputs.')
    p.Define(
        'dropout', DropoutLayer.Params(),
        'Dropout layer params. Can be a single params or a tuple/list of params'
        ' having the same length as the number of layers.')
    p.Define(
        'batch_norm', False,
        'Whether or not to apply BN to hidden layer output. '
        'This can be a single bool or a tuple/list of bools having the'
        ' same length as the number of layers.')
    p.Define(
        'activation', 'RELU',
        'The activation function to use. Can be a single string, or a'
        ' tuple/list of strings having the same length as the number'
        ' of layers.')
    p.Define('skip_connections', None, 'Must be None.')
    p.Define(
        'bn_fold_weights', None, 'Force folding the batch normalization '
        'weights in the projection layer.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(FeedForwardNet, self).__init__(params)
    p = self.params
    assert p.name

    assert p.skip_connections is None
    batch_norm = p.batch_norm
    num_layers = len(p.hidden_layer_dims)
    if isinstance(batch_norm, (list, tuple)):
      assert len(batch_norm) == num_layers
    else:
      batch_norm = [batch_norm] * num_layers
    activation = p.activation
    if isinstance(activation, six.string_types):
      activation = [activation] * num_layers
    else:
      assert len(activation) == num_layers
    params_dropout_layers = p.dropout
    if isinstance(params_dropout_layers, (list, tuple)):
      assert len(params_dropout_layers) == num_layers
    else:
      params_dropout_layers = [params_dropout_layers] * num_layers

    with tf.variable_scope(p.name):
      # Residual connections work better in the form of:
      #   y = x + Affine(Activation(BatchNorm(x)))
      params_fc_layers = []
      in_dim = p.input_dim
      for i in range(num_layers):
        out_dim = p.hidden_layer_dims[i]
        proj_out_dim = out_dim
        name = '%s_%d' % (p.name, i)
        params_i = ProjectionLayer.Params().Set(
            batch_norm=batch_norm[i],
            has_bias=(not batch_norm[i]),
            activation=activation[i],
            input_dim=in_dim,
            output_dim=proj_out_dim,
            bn_fold_weights=p.bn_fold_weights,
            name=name)
        params_fc_layers.append(params_i)
        in_dim = out_dim

        if p.qdomain.default is not None:
          params_i.qdomain.default = p.qdomain.default.Copy()

      self.CreateChildren('fc', params_fc_layers)
      self.CreateChildren('dropout', params_dropout_layers)

  def FProp(self, theta, inputs, paddings=None):
    p = self.params
    num_layers = len(self.fc)

    in_dim, layer_in = p.input_dim, inputs
    for i in range(num_layers):
      layer_in = py_utils.with_dependencies(
          [py_utils.assert_shape_match([tf.shape(layer_in)[-1]], [in_dim])],
          layer_in)
      out_dim = p.hidden_layer_dims[i]
      layer_out = self.fc[i].FProp(theta.fc[i], layer_in, paddings)
      layer_out = self.dropout[i].FProp(theta.dropout[i], layer_out)
      layer_in = layer_out
      in_dim = out_dim
    return layer_in


class DropoutLayer(base_layer.BaseLayer):
  """Apply dropout during trainig."""

  @classmethod
  def Params(cls):
    p = super(DropoutLayer, cls).Params()
    p.Define('keep_prob', 1.0, 'Keep probability.')
    # noise_shape is unknown when building layer params.
    p.Define(
        'noise_shape', None, 'A 1-D `Tensor` of type `int32`, representing'
        ' the shape for randomly generated keep/drop flags.')
    p.Define(
        'noise_shape_broadcast_dims', None,
        'A list of dimension where the noise shape is broadcasted. For '
        'example, noise_shape = [n, h, w, 1] when '
        'noise_shape_broadcast_dims=[-1] ')
    # We typically want to replace dropout by expectation during eval.
    # However, in certain cases E(f(x)) != f(E(x)), and replacing dropout by its
    # expectation during eval leads to worse quality.
    p.Define('dropout_at_eval', False,
             'Whether or not to also perform dropout at eval time.')
    return p

  def _Dropout(self, theta, inputs, noise_shape):
    return tf.nn.dropout(
        inputs,
        keep_prob=self.params.keep_prob,
        noise_shape=noise_shape,
        seed=self.params.random_seed)

  def FProp(self, theta, inputs):
    """Apply dropout to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.

    Returns:
      inputs with dropout applied at training time.
    """
    p = self.params
    if not p.is_eval or p.dropout_at_eval:
      if isinstance(p.keep_prob, numbers.Real) and p.keep_prob == 1.0:
        return inputs
      if p.noise_shape_broadcast_dims:
        noise_shape = p.noise_shape or py_utils.GetShape(inputs)
        for dim in p.noise_shape_broadcast_dims:
          if dim >= len(noise_shape):
            raise ValueError('Invalid broadcasted dim {}'.format(dim))
          noise_shape[dim] = 1
      else:
        noise_shape = p.noise_shape
      ret = self._Dropout(theta, inputs, noise_shape)
      ret.set_shape(inputs.get_shape())
      return ret
    else:
      return inputs

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    py_utils.CheckShapes((inputs,))
    flops_per_element = 10  # Approximately 10 flops per element.
    return py_utils.NestedMap(
        flops=inputs.num_elements() * flops_per_element, out_shapes=(inputs,))


class DeterministicDropoutLayer(DropoutLayer):
  """Apply dropout during trainig."""

  def _Dropout(self, theta, inputs, noise_shape):
    return py_utils.DeterministicDropout(
        inputs,
        keep_prob=self.params.keep_prob,
        seeds=py_utils.GenerateStepSeedPair(self.params, theta.global_step),
        noise_shape=noise_shape)


class LayerNorm(base_layer.BaseLayer):
  """Layer normalization.

  Implements layer normalization:
  https://arxiv.org/abs/1607.06450
  """

  @classmethod
  def Params(cls):
    p = super(LayerNorm, cls).Params()
    p.Define('input_dim', 0, 'Depth of the input to the network.')
    p.Define('epsilon', 1e-6, 'Tiny value to guard rsqrt.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LayerNorm, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim > 0
    pc = py_utils.WeightParams(
        shape=[p.input_dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('bias', pc)
      self.CreateVariable('scale', pc)

  def FProp(self, theta, inputs):
    """Applies normalization over the last dimension (layer).

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A tensor of shape [..., hidden_dim].

    Returns:
      tensor of the same shape with inputs
    """
    p = self.params
    inputs = py_utils.with_dependencies(
        [py_utils.assert_equal(tf.shape(inputs)[-1], p.input_dim)], inputs)

    @function.Defun(
        *[py_utils.FPropDtype(p)] * 3,
        noinline=not py_utils.use_tpu(),
        shape_func=lambda op: [op.inputs[0].shape])
    def Normalize(x, scale, bias):
      """Normalize `x` w/ `scale` and `bias` gain/shift."""
      x_shape = py_utils.GetShape(x)
      inner_dim = x_shape[-1]
      x_reshaped = tf.reshape(x, [-1, inner_dim])
      mean = tf.reduce_mean(x_reshaped, axis=[1], keepdims=True)
      variance = tf.reduce_mean(
          tf.square(x_reshaped - mean), axis=[1], keepdims=True)
      x_norm = (x_reshaped - mean) * tf.rsqrt(variance + p.epsilon)
      x_norm = tf.reshape(x_norm, x_shape)
      return x_norm * (1.0 + scale) + bias

    return Normalize(inputs, theta.scale, theta.bias)


class ConvSetLayer(quant_utils.QuantizableLayer):
  """Set of Convolutions with different filter sizes in a single layer.

    Applies a set of convolutions with different filter shapes to the inputs and
    returns the concatenated outputs.
  """

  @classmethod
  def Params(cls):
    p = super(ConvSetLayer, cls).Params()
    p.Define('cnn_tpl',
             ConvLayer.Params().Set(filter_stride=(1, 1)),
             'Conv layer template for the set of conv layers.')
    p.Define(
        'filter_shapes', [(0, 0, 0, 0)],
        'Must be a list of sequences of 4. Elements are in order of height'
        ' (time), width (frequency), in_channel, out_channel')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ConvSetLayer, self).__init__(params)
    p = self.params
    assert p.name

    filter_set = set()
    input_shape = None
    # Asserting kernel sizes are different and input sizes are the same.
    for filter_shape in p.filter_shapes:
      key = '%d_%d' % (filter_shape[0], filter_shape[1])
      assert key not in filter_set
      filter_set.add(key)
      if input_shape is None:
        input_shape = filter_shape[2]
      assert input_shape == filter_shape[2]

    # The same QTensor is used for all inputs to the concat.
    self.TrackQTensor('activation')

    params_conv_set = []
    with tf.variable_scope(p.name):
      for filter_shape in p.filter_shapes:
        conv_p = p.cnn_tpl.Copy()
        conv_p.name = '%d_%d' % (filter_shape[0], filter_shape[1])
        # Important: combined quantization will be done pre-concat versus
        # by each layer on its output. Otherwise, inherit quantization params
        # from this layer.
        if p.qdomain.default is not None:
          conv_p.qdomain.default = p.qdomain.default.Copy()
        conv_p.disable_activation_quantization = True
        conv_p.filter_shape = filter_shape
        params_conv_set.append(conv_p)
      self.CreateChildren('conv_set', params_conv_set)

  def FProp(self, theta, inputs, paddings):
    """Apply all convolution sets to inputs and concatenate outputs.

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
      A tuple (out, output_paddings).

      - out: output tensor. Expected to be of shape [batch, time_mod,
        frequency_mod, out_channel_1 + out_channel_2 ...] where time_mod and
        frequency_mod depend on the conv layer strides and out_channel_i is
        the output channel size of the i-th conv layer in the set.
      - output_paddings: Modified paddings generated within `ConvLayer.FProp`.
        Expected to be of the shape [batch, time_mod].
    """
    p = self.params
    inputs = py_utils.with_dependencies([
        py_utils.assert_shape_match(tf.shape(paddings), [-1, -1]),
        py_utils.assert_shape_match(
            tf.shape(inputs),
            tf.concat([tf.shape(paddings), [-1, p.filter_shapes[0][2]]], 0))
    ], inputs)

    conv_outputs = []
    output_paddings = None
    # output_padding should be same for all filters for the same stride.
    for i, conv_i in enumerate(self.conv_set):
      conv_i_output, conv_i_padding = conv_i.FProp(theta.conv_set[i], inputs,
                                                   paddings)
      if output_paddings is None:
        output_paddings = conv_i_padding
      conv_outputs.append(conv_i_output)

    # Track for quantization.
    conv_outputs = [self.QTensor('activation', t) for t in conv_outputs]

    out = tf.concat(conv_outputs, -1)
    return out, output_paddings


class LocalizedLabelSmoother(base_layer.BaseLayer):
  """Smooths labels given as class ids.

  Implements the smoothing from https://arxiv.org/abs/1612.02695. Instead of
  1-hot class ids the model is trained to predict a distribution over classes
  that includes the correct class label and with a small probability the labels
  of tokens that appear nearby in time in the ground truth. This typically acts
  as a strong regularizer.

  """

  @classmethod
  def Params(cls):
    p = super(LocalizedLabelSmoother, cls).Params()
    p.Define('num_classes', 0, 'Number of classes')
    p.Define(
        'offsets', [], 'Offset (over time) for smoothing. At time T the '
        'smoothed target is class[T] + sum_i weights[i]*class[T+offset[i]]')
    p.Define('weights', [], 'Weight of the smoothing at corresponding offset')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LocalizedLabelSmoother, self).__init__(params)
    p = self.params
    assert p.num_classes > 0
    assert len(p.offsets) == len(p.weights)
    assert p.name

  def FProp(self, theta, target_paddings, target_labels, target_ids):
    """Convert class_ids to 1hot and smooth by neighborhood.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      target_paddings: float32 matrix [bs, seq_len]
      target_labels: int32 matrix [bs, seq_len]. This stores the target label
        output at each decoder step as generated by the speech input generator
        input_batch.tgt.labels
      target_ids: int32 matrix [bs, seq_len]. This stores the target_id that is
        fed to the decoder, as generated by the speech input generator
        input_batch.tgt.ids

    Returns:
      A tensor [bs, seq_len, num_classes] denoting a smoothed distribution over
      num_classes.
    """
    del target_ids  # Unused.
    p = self.params
    class_probabilities = tf.one_hot(
        target_labels, p.num_classes, dtype=py_utils.FPropDtype(p))

    # Start list keeping the scaled class-probabilities at different offsets.
    output_distributions = [class_probabilities]
    seq_len = tf.shape(class_probabilities)[1]
    # If offsets < 0 we force a future output_act to be like a past token.
    # If offsets > 0 we force a past output_act to be like a future token.
    min_offset = np.min(p.offsets + [0])
    max_offset = np.max(p.offsets + [0])
    class_probabilities = tf.pad(class_probabilities,
                                 [[0, 0], [-min_offset, max_offset], [0, 0]])
    # Shift the weights to the left by one location - we don't make the
    # EOS more probable.
    class_weights = tf.pad(1.0 - target_paddings[:, 1:],
                           [[0, 0], [-min_offset, max_offset + 1]])
    class_weights = tf.expand_dims(class_weights, 2)

    for offset, weight in zip(p.offsets, p.weights):
      offset_in_padded = offset - min_offset
      output_distributions.append(
          class_probabilities[:, offset_in_padded:offset_in_padded + seq_len, :]
          * class_weights[:, offset_in_padded:offset_in_padded + seq_len, :] *
          weight)
    output_distributions = tf.add_n(output_distributions)
    output_distributions /= tf.reduce_sum(
        output_distributions, axis=-1, keepdims=True)
    return output_distributions


class UniformLabelSmoother(base_layer.BaseLayer):
  """Smooths labels given as class ids and confidence.

  Implements the smoothing from https://arxiv.org/abs/1512.00567. Correct class
  label confidence is dropped by eps and all the other classes are increased
  by eps/num_classes.
  """

  @classmethod
  def Params(cls):
    p = super(UniformLabelSmoother, cls).Params()
    p.Define('num_classes', 0, 'Number of classes')
    p.Define('uncertainty', 0.1, 'Uncertainty of correct label, eps.')
    p.Define(
        'uncertainty_larger', 0.1,
        'Apply a larger uncertainty to specific tokens, as specified '
        'by token_from_target_ids.')
    p.Define('token_id_uncertainty_larger', None, 'Id of token from target_ids '
             'to apply uncertainty_larger to.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(UniformLabelSmoother, self).__init__(params)
    p = self.params
    assert p.num_classes > 0
    assert 0.0 <= p.uncertainty < 1.0
    assert p.token_id_uncertainty_larger is None or (
        p.token_id_uncertainty_larger >= 0)
    assert p.name

  def FProp(self, theta, target_paddings, target_labels, target_ids):
    """Convert target_labels to 1hot and smooth uniformly.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      target_paddings: float32 matrix [bs, seq_len]
      target_labels: int32 matrix [bs, seq_len]. This stores the target label
        output at each decoder step as generated by the speech input generator
        input_batch.tgt.labels
      target_ids: int32 matrix [bs, seq_len]. This stores the target_id that is
        fed to the decoder, as generated by the speech input generator
        input_batch.tgt.ids

    Returns:
      A tensor of float32 [bs, seq_len, num_classes] denoting a smoothed
      distribution over num_classes.
    """
    del target_paddings  # Unused by FProp.
    p = self.params

    low_confidence = p.uncertainty / tf.to_float(p.num_classes - 1)
    high_confidence = (1.0 - p.uncertainty)

    smooth_targets = tf.one_hot(
        tf.cast(target_labels, tf.int32),
        depth=p.num_classes,
        on_value=high_confidence,
        off_value=low_confidence)
    if p.token_id_uncertainty_larger is not None:
      assert target_ids is not None
      low_confidence_larger = p.uncertainty_larger / tf.to_float(p.num_classes -
                                                                 1)
      high_confidence_larger = (1.0 - p.uncertainty_larger)
      smooth_targets_larger = tf.one_hot(
          tf.cast(target_labels, tf.int32),
          depth=p.num_classes,
          on_value=high_confidence_larger,
          off_value=low_confidence_larger)
      should_smooth_larger = tf.tile(
          tf.expand_dims(
              tf.equal(target_ids, p.token_id_uncertainty_larger), -1),
          multiples=[1, 1, p.num_classes])
      smooth_targets = tf.where(should_smooth_larger, smooth_targets_larger,
                                smooth_targets)
    return smooth_targets


class HighwaySkipLayer(base_layer.BaseLayer):
  """A highway skip layer.

  This class represents a highway skip layer, which takes multiple
  inputs (from different layers of the network) and gates them.
  This returns C(x)x + T(x)h, initially biasing C to be open.
  For some discussion about initialization please see:
  Section 2.2 in [Srivastava, 2015]: https://arxiv.org/pdf/1505.00387v2.pdf
  """

  @classmethod
  def Params(cls):
    p = super(HighwaySkipLayer, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the input to the network.')
    p.Define(
        'batch_norm', False,
        'Whether or not to apply BN to the highway skip layer output. '
        'Note this is only a single bool.')
    p.Define('carry_bias_init', 1.0, 'carry gates bias initialization')
    p.Define('couple_carry_transform_gates', False,
             'Boolean on whether to couple the transform and carry gates.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(HighwaySkipLayer, self).__init__(params)
    p = self.params
    assert p.name
    self._carry_gate = None
    self._transform_gate = None
    with tf.variable_scope(p.name):
      carry_gate_params = ProjectionLayer.Params().Set(
          batch_norm=p.batch_norm,
          has_bias=True,
          activation='SIGMOID',
          input_dim=p.input_dim,
          output_dim=p.input_dim,
          bias_init=p.carry_bias_init,
          name='%s_carry_gate' % p.name)
      self.CreateChild('carry_gate', carry_gate_params)

      if not p.couple_carry_transform_gates:
        transform_gate_params = ProjectionLayer.Params().Set(
            batch_norm=p.batch_norm,
            has_bias=True,
            activation='SIGMOID',
            input_dim=p.input_dim,
            output_dim=p.input_dim,
            bias_init=-p.carry_bias_init,
            name='%s_transform_gate' % p.name)
        self.CreateChild('transform_gate', transform_gate_params)

  def FProp(self, theta, x, transformed_x, paddings=None):
    """Fprop for Highway Skip layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      x: feature at the lower layer.
      transformed_x: transformation of x at a higher layer.
      paddings: padding applied to the features.

    Returns:
      layer_out - activations after forward propagation.
    """
    p = self.params
    assert self.carry_gate is not None
    carry = self.carry_gate.FProp(theta.carry_gate, x, paddings)
    if p.couple_carry_transform_gates:
      transform = 1 - carry
    else:
      assert self.transform_gate is not None
      transform = self.transform_gate.FProp(theta.transform_gate, x, paddings)
    layer_out = x * carry + transformed_x * transform
    return layer_out


class GradNormTracker(base_layer.BaseLayer):
  """A helper class to keep track of gradient norm stats."""

  @classmethod
  def Params(cls):
    p = super(GradNormTracker, cls).Params()
    p.Define('decay', 0.995,
             'Decay in updating the moving avgs in grad norm stats')
    p.Define('grad_norm_lower_cap', 1e-2, 'The minimal gradient norm value.')
    p.Define(
        'clip_threshold', 4.0,
        'Distance threshold at which gradients are clipped to 0.0.'
        ' Distance is measured in the number of standard deviations a'
        ' given gradient norm is from the mean gradient norm. The'
        ' default value of 3.0 means we are throwing away roughly'
        ' 0.15% of steps.')
    p.Define(
        'grad_norm_clip_cap_min', 0.0,
        'We stop clipping if grad norm is already smaller than this'
        ' value.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GradNormTracker, self).__init__(params)
    p = self.params
    assert p.name

    with tf.variable_scope(p.name):
      pc = py_utils.WeightParams(
          shape=[],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=tf.float32,
          collections=[self.__class__.__name__ + '_vars'])
      _, self._log_mean = py_utils.CreateVariable(
          'log_mean', pc, trainable=False)
      _, self._log_mean_squared = py_utils.CreateVariable(
          'log_mean_squared', pc, trainable=False)
      _, self._total_weight = py_utils.CreateVariable(
          'total_weight', pc, trainable=False)
      _, self._total_rejections = py_utils.CreateVariable(
          'total_rejections', pc, trainable=False)
      self._decay = p.decay

  def FProp(self, theta, grad_norm, has_nan=None):
    """Update gradient norm moving avgs, and returns whether or not ...

    to clip gradients to 0.0. If the current batch has NaN grads, does not
    update the moving avgs and forces to clip the gradients to 0.0.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      grad_norm: A float scalar tensor.
      has_nan: A boolean scalar tensor to indicate if the current batch has nan.

    Returns:
      A scalar float tensor with value of either 1.0 or 0.0. The value of 0.0
      means the gradient norm is excessively large or contains NaN, and the step
      should be aborted completely.
    """
    p = self.params
    with tf.name_scope(p.name):
      grad_norm = tf.maximum(grad_norm, p.grad_norm_lower_cap)

      total_weight = self._total_weight
      log_mean = self._log_mean
      log_mean_squared = self._log_mean_squared

      # Exponentially decayed moving avg of log(grad_norm) mean.
      mean = log_mean / tf.maximum(total_weight, 1e-6)
      # Exponentially decayed moving avg of log(grad_norm) variance.
      var = (log_mean_squared / tf.maximum(total_weight, 1e-6)) - mean * mean
      std = tf.sqrt(tf.maximum(var, 1e-6))

      summary_utils.scalar('log_grad_norm_mean', mean)
      summary_utils.scalar('log_grad_norm_std', std)
      summary_utils.scalar('clip_ratio_threshold',
                           tf.exp(std * p.clip_threshold))
      summary_utils.scalar('clip_threshold',
                           tf.exp(mean + std * p.clip_threshold) - 1.0)
      summary_utils.scalar('total_rejections', self._total_rejections)

      log_grad_norm = tf.log(grad_norm + 1.0)
      log_grad_norm_cap = mean + std * p.clip_threshold
      log_grad_norm_cap_min = tf.log(p.grad_norm_clip_cap_min + 1.0)
      log_grad_norm_cap = tf.maximum(log_grad_norm_cap, log_grad_norm_cap_min)

      def UpdateExpMovingAvg(ref_var, val, ignore):
        if ignore is not None:
          delta = tf.where(ignore, tf.zeros([]),
                           (1.0 - p.decay) * (val - ref_var))
        else:
          delta = (1.0 - p.decay) * (val - ref_var)
        return tf.assign_add(ref_var, delta)

      # We trigger when total_weight is at least half of max weight or the
      # current batch contains NaNs.
      trigger = tf.logical_and(log_grad_norm > log_grad_norm_cap,
                               total_weight > 0.75)
      if has_nan is not None:
        trigger = tf.logical_or(trigger, has_nan)

      log_grad_norm_capped = tf.minimum(log_grad_norm, log_grad_norm_cap)

      update_moving_avg = tf.group(
          UpdateExpMovingAvg(log_mean, log_grad_norm_capped, has_nan),
          UpdateExpMovingAvg(log_mean_squared,
                             log_grad_norm_capped * log_grad_norm_capped,
                             has_nan),
          UpdateExpMovingAvg(total_weight, tf.constant(1.0), has_nan),
          tf.assign_add(self._total_rejections, tf.cast(trigger, tf.float32)))

      return py_utils.with_dependencies([update_moving_avg],
                                        1.0 - tf.cast(trigger, tf.float32))


class WeightedSumLayer(base_layer.BaseLayer):
  """Returns the weighted sum of a list of input tensors."""

  @classmethod
  def Params(cls):
    """Params for this MergerLayer class."""
    p = super(WeightedSumLayer, cls).Params()
    p.Define('num_sources', 0, 'Number of input sources to combine.')
    p.Define('weighted_merger_dropout_prob', 0.1,
             'Applies dropout to the weights.')
    p.Define(
        'weighted_merger_softmax', True, 'If set, applies a softmax '
        'layer on top of the weights for normalization.')
    p.Define('global_weight_scale', 1.0, 'A global scale put on weights.')
    p.Define('minimal_prob', 0.0, 'The minimal weight for each component.')
    p.Define('add_weight_summaries', False, 'If set, creates summaries for the '
             'sum weights.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(WeightedSumLayer, self).__init__(params)
    p = self.params
    if not p.name:
      raise ValueError('Layer must have a specified name!')

    assert p.num_sources > 0, ('Must specify num_sources > 0.')
    params_init = py_utils.WeightInit.Constant(0.0)
    # Weights to be learned.
    pw = py_utils.WeightParams(
        shape=[p.num_sources],
        init=params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('sum_weight', pw)

    if p.weighted_merger_dropout_prob > 0.0:
      dropout_tpl = DropoutLayer.Params()
      dropout_tpl.keep_prob = (1.0 - p.weighted_merger_dropout_prob)
      self.CreateChild('weighted_merger_dropout', dropout_tpl)
    else:
      self.CreateChild('weighted_merger_dropout', IdentityLayer.Params())

  def FProp(self, theta, inputs):
    """Combines the list of input tensors into a single tensor.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A list of tensors of shape [time, batch, hidden_dim]

    Returns:
      A tensor of the same shape with input tensors.
    """
    p = self.params
    n_sources = len(inputs)

    if n_sources == 1:
      return inputs[0]

    # Weighted sum of all sources, all dims must match.
    # For weighted_sum, assume input is a list of rank 3 tensors
    inputs = tf.stack(inputs)
    inputs = py_utils.HasRank(inputs, 4)

    # The constant factor is just meant to support the non-normalized scenario.
    # If softmax is applied, this factor will cancel out.
    w = theta.sum_weight * p.global_weight_scale + (1 / p.num_sources)
    w = self.weighted_merger_dropout.FProp(theta.weighted_merger_dropout, w)

    if p.weighted_merger_softmax:
      residual_weights = p.minimal_prob * p.num_sources
      assert residual_weights >= 0.0
      assert residual_weights < 1.0
      w = tf.nn.softmax(w, axis=0) * (1.0 - residual_weights) + p.minimal_prob

    if p.add_weight_summaries:
      for i in range(p.num_sources):
        summary_utils.scalar(p.name + 'weight_%d' % i, w[i])
    w = tf.reshape(w, [p.num_sources, 1, 1, 1])
    output = tf.reduce_sum(inputs * w, axis=0)

    return output


class GatedAverageLayer(base_layer.BaseLayer):
  """Gated combination of n input vectors.

  Given n inputs, x_1 ... x_n. First learns a gate g in a single layer.
  Returns g_1 * x_1 + ... g_n * x_n.
  """

  @classmethod
  def Params(cls):
    p = super(GatedAverageLayer, cls).Params()
    p.Define('num_nodes', 0, 'Number of nodes in each input vector.')
    p.Define('num_inputs', 0, 'Number of input vectors to combine.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes GatedAverageLayer."""
    super(GatedAverageLayer, self).__init__(params)
    p = self.params

    assert p.num_nodes > 0, 'Number of dimensions should be greater than 0.'
    assert p.num_inputs > 0, 'Number of inputs should be greater than 0.'

    in_size = p.num_inputs * p.num_nodes

    with tf.variable_scope(p.name):
      # Weight matrix for scalar gates
      gm_pc = py_utils.WeightParams(
          shape=[in_size, p.num_inputs],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('gm', gm_pc)

  def FProp(self, theta, inputs):
    """Gates, then merges a list of n input vectors.

    Args:
      theta: gm (gate matrix)
      inputs: List of inputs, each of shape [..., num_nodes]

    Returns:
      a gated output vector [..., num_nodes]
    """
    p = self.params
    assert len(inputs) == p.num_inputs, 'Number of inputs should match params.'

    for i, inp in enumerate(inputs):
      inputs[i] = py_utils.with_dependencies([
          py_utils.assert_shape_match([tf.shape(inp)[-1]], [p.num_nodes]),
          py_utils.assert_shape_match(tf.shape(inp), tf.shape(inputs[0])),
      ], inp)

    input_shape = tf.shape(inputs[0])

    reshaped_inputs = [tf.reshape(inp, [-1, p.num_nodes]) for inp in inputs]
    concat_inputs = tf.concat(reshaped_inputs, axis=1)

    xmg = tf.nn.softmax(py_utils.Matmul(concat_inputs, theta.gm))
    xmg = tf.expand_dims(xmg, 2)
    inputs = tf.reshape(concat_inputs, [-1, p.num_inputs, p.num_nodes])
    gated_sum = tf.reduce_sum(xmg * inputs, axis=1)

    return tf.reshape(gated_sum, input_shape)


class LHUCLayer(base_layer.BaseLayer):
  """`Learning Hidden Unit Contribution (LHUC)` layer.

  This paper proposes to use LHUC layer for NMT adaptation:
      http://aclweb.org/anthology/N18-2080

  During base model training, LHUC layer is fixed to 1.0 (no-op in
  multiplication). During adaptation, only LHUC layer is trained, and all other
  parameters in the model are frozen.
  """

  @classmethod
  def Params(cls):
    p = super(LHUCLayer, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the input and output.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LHUCLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim > 0

    pc = py_utils.WeightParams(
        shape=[p.input_dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=self._VariableCollections())
    with tf.variable_scope(p.name):
      self.CreateVariable('w', pc)

  def FProp(self, theta, inp):
    """Add learnt gate for adaptation."""
    out = 2.0 * tf.sigmoid(theta.w) * inp
    return out


class ResidualAdapterLayer(base_layer.BaseLayer):
  """Residual Adapter layer for NLP tasks.

  This paper proposes using residual adapters for fine-tuning new tasks on BERT.
  https://arxiv.org/pdf/1902.00751.pdf

  During adaptation, residual adapter layers can be added to a pre-trained
  model and trained, while all other parameters are frozen.
  In terms of operations, the layer is identical to a vanilla Transformer
  feedforward layer. Separate implementation is meant to distinguish function.
  """

  @classmethod
  def Params(cls):
    p = super(ResidualAdapterLayer, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the input to the adapter.')
    p.Define('bottleneck_dim', 0, 'Dimension of the feedforward inner layer.')
    p.Define('ln_tpl', LayerNorm.Params(), 'Layer norm default params.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ResidualAdapterLayer, self).__init__(params)
    p = self.params
    assert p.name

    with tf.variable_scope(p.name):
      bottleneck_params = FeedForwardNet.Params().Set(
          name='bottleneck',
          activation=['RELU', 'NONE'],
          input_dim=p.input_dim,
          hidden_layer_dims=[p.bottleneck_dim, p.input_dim])
      self.CreateChild('bottleneck', bottleneck_params)

      params = p.ln_tpl.Copy()
      params.name = 'adapter_ln'
      params.input_dim = p.input_dim
      self.CreateChild('layer_norm', params)

  def FProp(self, theta, x, paddings=None):
    """Fprop for Residual Adapter.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      x: [..., input_dim].
      paddings: padding applied to the features.

    Returns:
      layer_out - [..., input_dim].
    """
    normalized_x = self.layer_norm.FProp(theta.layer_norm, x)
    bottleneck_x = self.bottleneck.FProp(theta.bottleneck, normalized_x,
                                         paddings)
    return x + bottleneck_x


class Conv2DLayerNoPadding(base_layer.BaseLayer):
  """2-D Convolution layer w/o padding.

  TODO(laurenzo): Dedup in favor of SeparableConv2DLayer where possible.
  """

  @classmethod
  def Params(cls):
    p = super(Conv2DLayerNoPadding, cls).Params()
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

  @base_layer.initializer
  def __init__(self, params):
    super(Conv2DLayerNoPadding, self).__init__(params)
    p = self.params
    assert p.name
    assert p.padding in ['SAME', 'VALID']
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert len(p.dilations) == 2
    assert all(x > 0 for x in p.filter_shape)
    assert all(x > 0 for x in p.filter_stride)
    self._CreateConvVariables()

  def _CreateConvVariables(self):
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)

  def FProp(self, theta, x):
    """Apply convolution to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      x: The inputs tensor. It is expected to be of shape [batch, height, width,
        channel].

    Returns:
      Convolution output.
    """
    p = self.params
    with tf.name_scope(p.name):
      return tf.nn.conv2d(
          input=x,
          filter=theta.w,
          strides=[1, p.filter_stride[0], p.filter_stride[1], 1],
          padding=p.padding,
          dilations=[1, p.dilations[0], p.dilations[1], 1],
          data_format='NHWC')

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    b, h, w, c = inputs.as_list()
    fh, fw, ic, oc = p.filter_shape
    assert ic == c
    sh, sw = p.filter_stride
    if p.padding == 'SAME':
      oh = int(np.ceil(float(h) / float(sh)))
      ow = int(np.ceil(float(w) / float(sw)))
    else:
      oh = int(np.ceil(float(h - fh + 1) / float(sh)))
      ow = int(np.ceil(float(w - fw + 1) / float(sw)))
    flops = b * oh * ow * fh * fw * ic * oc * 2  # mul/add counts as 2 flop.
    outputs = tf.TensorShape([b, oh, ow, oc])

    return py_utils.NestedMap(flops=flops, out_shapes=(outputs,))


class FetchLayer(base_layer.BaseLayer):
  """A layer facilitating fetching activations and their gradients."""

  @base_layer.initializer
  def __init__(self, params):
    super(FetchLayer, self).__init__(params)
    assert self.params.name
    self._activations = None
    self._gradients = None

  @classmethod
  def FPropMeta(cls, params, *args):
    return py_utils.NestedMap(flops=0, out_shapes=args)

  def _ReturnSingleValueOrList(self, lst):
    assert lst is not None
    assert isinstance(lst, list)
    return lst if len(lst) > 1 else lst[0]

  @property
  def activation(self):
    return self._ReturnSingleValueOrList(self._activations)

  @property
  def gradient(self):
    return self._ReturnSingleValueOrList(self._gradients)

  def FProp(self, theta, *args):
    del theta
    num = len(args)
    self._activations = [None] * num
    self._gradients = [None] * num

    for i, v in enumerate(args):

      def ShapeFunc(op):
        return [op.inputs[0].shape]

      def FetchBak(op, dy, index=i):
        del op
        self._gradients[index] = dy
        return dy

      @function.Defun(v.dtype, shape_func=ShapeFunc, python_grad_func=FetchBak)
      def FetchFwd(x):
        return x

      self._activations[i] = FetchFwd(v)

    return tuple(self._activations) if num > 1 else self._activations[0]


class GluLayer(base_layer.BaseLayer):
  """Gated Linear Unit.

  See https://arxiv.org/abs/1612.08083 for more details.
  """

  @classmethod
  def Params(cls):
    p = super(GluLayer, cls).Params()
    p.Define('input_dim', 0, 'Dimension of the layer input.')
    p.Define('output_dim', 0, 'Dimension of the layer output.')
    p.Define('ln_tpl', LayerNorm.Params(), 'Layer norm default params.')
    p.Define('dense_tpl', FCLayer.Params().Set(), 'Fully connected layer.')
    p.Define(
        'activation', 'RELU',
        'Non-linearity applied after the dense layer in the value branch.')
    p.Define('dropout_tpl', DropoutLayer.Params(), 'Dropout applied to output.')
    p.Define('apply_residual', True, 'Whether or not to add inputs to outputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GluLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim

    if p.output_dim:
      output_dim = p.output_dim
    else:
      output_dim = p.input_dim

    if p.apply_residual:
      assert output_dim == p.input_dim

    with tf.variable_scope(p.name):
      # Initialize value feed-forward layer.
      params = p.dense_tpl.Copy()
      params.name = 'value_layer'
      params.input_dim = p.input_dim
      params.activation = p.activation
      params.output_dim = output_dim
      self.CreateChild('value_layer', params)

      # Initialize gate feed-forward layer.
      params = p.dense_tpl.Copy()
      params.name = 'gate_layer'
      params.input_dim = p.input_dim
      params.activation = 'SIGMOID'
      params.output_dim = output_dim
      self.CreateChild('gate_layer', params)

      # Initialize layer norm.
      params = p.ln_tpl.Copy()
      params.name = 'layer_norm'
      params.input_dim = p.input_dim
      self.CreateChild('layer_norm', params)

      # Initialize dropout.
      dropout_tpl = p.dropout_tpl.Copy()
      self.CreateChild('dropout', dropout_tpl)

  def FProp(self, theta, inputs, paddings):
    inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
    values = self.value_layer.FProp(theta.value_layer, inputs_normalized,
                                    tf.expand_dims(paddings, -1))
    gates = self.gate_layer.FProp(theta.gate_layer, inputs_normalized,
                                  tf.expand_dims(paddings, -1))
    glu_output = values * gates
    glu_output = self.dropout.FProp(theta.dropout, glu_output)
    if self.params.apply_residual:
      return inputs + glu_output
    return glu_output
