# Lint as: python3
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

import copy
import functools
import math
import numbers
from typing import Optional, Tuple, Union

import lingvo.compat as tf
from lingvo.core import activations
from lingvo.core import base_layer
from lingvo.core import bn_layers
from lingvo.core import builder_layers
from lingvo.core import computation_cost
from lingvo.core import conv_layers_with_time_padding
from lingvo.core import gshard_utils
from lingvo.core import pruning_utils
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import recurrent
from lingvo.core import schedule
from lingvo.core import summary_utils
from lingvo.core import symbolic
from lingvo.core import tshape
import numpy as np
import sympy

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops
# pylint:enable=g-direct-tensorflow-import


class DeconvLayer(base_layer.BaseLayer):
  """Deconv (transposed conv2d) layer.

  DeconvLayer is different from ConvTransposeLayer in that
  DeconvLayer does not support padding and biasing. Hence,
  it's simpler and more basic than ConvTransposeLayer.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'filter_shape', (0, 0, 0, 0),
        'Filter shape. Must be a sequence of length 4. Elements are in'
        ' the order of height, width, out_channel, in_channel.')
    p.Define(
        'filter_stride', (0, 0),
        'Filter stride to use. Must be a pair of ints. The first int'
        ' specifies the stride on the height dimension. The second int'
        ' specifies the stride on the width dimension.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert all(x > 0 for x in p.filter_shape)
    assert all(x > 0 for x in p.filter_stride)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('w', w_pc)

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    p = self.params
    t_stride = p.filter_stride[0]
    f_stride = p.filter_stride[1]
    return tf.stack([
        in_shape[0], in_shape[1] * t_stride, in_shape[2] * f_stride,
        p.filter_shape[2]
    ])

  def _ApplyConv(self, theta, inputs):
    p = self.params
    w = theta.w
    strides = [1, p.filter_stride[0], p.filter_stride[1], 1]
    # TODO(miachen): remove casting once tf.nn.conv2d supports tf.float64.
    assert inputs.dtype == w.dtype
    dtype = inputs.dtype
    if dtype != tf.float32:
      inputs = tf.cast(inputs, tf.float32)
      w = tf.cast(w, tf.float32)
    # TODO(zhifengc): Try some better way to do Deconv. Search for
    # "resize-convolution".
    out = tf.nn.conv2d_transpose(
        inputs,
        w,
        output_shape=self.OutShape(tf.shape(inputs)),
        strides=strides,
        padding='SAME')
    if dtype != tf.float32:
      out = tf.cast(out, dtype)
    return py_utils.HasShape(out, [-1, -1, -1, p.filter_shape[2]])

  def FProp(self, theta, inputs):
    """Apply deconvolution to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch, height,
        width, channel].

    Returns:
      outputs. outputs is expected to have shape [batch, height * height_stride,
      width * width_stride, out_channel].
    """
    p = self.params
    inputs = py_utils.HasShape(inputs, [-1, -1, -1, p.filter_shape[3]])
    return self._ApplyConv(theta, inputs)


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

  def FProp(self, theta, inputs, *args):
    """Identity mapping.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The input tensor or the input NestedMap.
      *args: Arguments to be ignored.

    Returns:
      Tensor with the same shape and type of inputs.
    """
    p = self.params
    with tf.name_scope(p.name):
      return tf.nest.map_structure(tf.identity, inputs)

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
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert len(p.dilation_rate) == 2
    assert all(x > 0 for x in p.filter_stride)
    assert all(x > 0 for x in p.dilation_rate)
    if any(x > 1 for x in p.dilation_rate):
      assert all(x == 1 for x in p.filter_stride)
    # Bias is not needed with batch_norm=True.
    if p.batch_norm:
      assert not p.bias
    assert (p.activation == 'NONE' or activations.IsSupported(p.activation))

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

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
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

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    if self.params.batch_norm:
      self.bn.InstantiateVariables()
    super()._CreateChildrenVariables()

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

  @classmethod
  def OutputShape(cls, params, in_shape):
    return _ComputeConvOutputShape(in_shape, params.filter_stride[0],
                                   params.filter_stride[1],
                                   params.filter_shape[-1])

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
      b = tf.zeros([symbolic.ToStatic(self.output_channels)],
                   dtype=filter_w.dtype)

    # Pass-through if weights are not folded with batch normalization.
    if not self._is_bn_folded:
      return filter_w, b

    # If batch norm is fused with weights, then compute the weights as from
    # figure C.8 of https://arxiv.org/pdf/1712.05877.pdf for training and
    # figure C.6 for eval.
    if self.do_eval:
      # Gets current moments without updating.
      mean, variance, beta, gamma = self.bn.GetCurrentMoments(theta.bn)
    else:
      # Updates moments based on a trial run of the convolution.
      raw_conv_output = convolution_lambda(filter_w)
      mean, variance, beta, gamma = self.bn.ComputeAndUpdateMoments(
          theta.bn, raw_conv_output, folded_bn_padding)

    # Fold weights and bias. Note that this layer's bias is not used (not
    # applicable for batch norm case).
    sigma_recip = tf.math.rsqrt(variance + self.bn.epsilon)
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
      # Causal convolution is only applied in time (height) dimension.
      # Use VALID padding and shift the inputs to the right to ensure that the
      # first output only depends on the first input and so on. The output is
      # the same size as the input, as if the convolution used SAME padding.
      padding_algorithm = 'VALID'
      # The effective spatial filter size for dilated convolutions is
      # (kernel - 1) * dilation_rate + 1 as according to
      # https://www.tensorflow.org/api_docs/python/tf/nn/convolution.
      causal_pad_size = (p.filter_shape[0] - 1) * p.dilation_rate[0]

      # Apply padding in width dimension to mimic SAME padding.
      # Using the similar logic as above to produce the same number of output
      # as if SAME padding is used.
      width_pad_size = (p.filter_shape[1] - 1) * p.dilation_rate[1]

      # The amount of padding on the left is tricky. If stride > 1, total
      # padding required for SAME padding would be:
      #   pad = ceil(input_size / stride - 1) * stride + eff_kernel - input_size
      # where eff_kernel = (kernel - 1) * dilation_rate + 1
      # TensorFlow also pads more on the right / bottom side if total padding
      # required is an odd number, so pad_left = pad // 2
      # Therefore pad_left could depend on input size, which might be dynamic.
      # Here we only handle two special cases where 1) stride = 1, then
      #   pad_left = (eff_kernel - 1) // 2
      # and 2) kernel = 1, then
      #   pad_left = 0
      if p.filter_stride[1] > 1 and p.filter_shape[1] > 1:
        raise ValueError('Causal convolution only supports width stride = 1 '
                         'or filter width = 1.')
      width_pad_left = max(0, width_pad_size - 1) // 2
      width_pad_right = width_pad_size - width_pad_left
      inputs = tf.pad(inputs, [[0, 0], [causal_pad_size, 0],
                               [width_pad_left, width_pad_right], [0, 0]])

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

      out = py_utils.HasShape(
          out, symbolic.ToStatic(BaseConv2DLayer.OutShape(self, input_shape)))
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
      out = activations.GetFn(p.activation)(out)
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
      out = activations.GetFn(p.activation)(out)

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
        dilations=p.dilation_rate,
        data_format='NHWC',
        padding=padding_algorithm)


class ConvNN2DLayer(BaseConv2DLayer):
  """Convolution layer, based on tf.nn.conv2d instead of tf.nn.convolution.

  tf.nn.convolution is using a different implementation on atrous convolutions,
  by wrapping the actual convolution with space_to_batch and batch_to_space.
  This implementation is not supported in tflite conversion, hence we need
  a different layer for using atrous convolutions.
  """

  def _EvaluateConvKernel(self, inputs, filter_w, strides, dilation_rate,
                          padding_algorithm, data_format):
    p = self.params
    return tf.nn.conv2d(
        inputs,
        filter_w,
        strides=strides,
        dilations=p.dilation_rate,
        data_format='NHWC',
        padding='SAME')


# Alias of Conv2DLayer (for compatibility with historical uses).
ConvLayer = Conv2DLayer


class DepthwiseConv2DLayer(BaseConv2DLayer):
  """Depthwise conv 2D layer.

  paper: https://arxiv.org/abs/1610.02357
  """

  @classmethod
  def Params(cls):
    p = super().Params()
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
        dilations=p.dilation_rate,
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
    p = super().Params()
    p.Define(
        'depth_multiplier', 1,
        'Number of depthwise convolution output channels per input channel. '
        'The total number of depthwise convolution output channels will be.'
        'equal to in_channel * depth_multiplier.')
    p.Define('depthwise_tpl',
             DepthwiseConv2DLayer.Params().Set(activation='NONE'),
             'Template for the depthwise conv sub-layer.')
    return p

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

    super().__init__(params)
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
    self.CreateChild('depthwise_conv', depthwise_params)

  def FProp(self, theta, inputs, paddings=None):
    inputs, paddings = self.depthwise_conv.FProp(theta.depthwise_conv, inputs,
                                                 paddings)
    return super().FProp(theta, inputs, paddings)

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    in_shape = self.depthwise_conv.OutShape(in_shape)
    return super().OutShape(in_shape)


class ProjectionLayer(quant_utils.QuantizableLayer):
  """Projection layer, with batch normalization and relu activation."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Depth of the input.')
    p.Define('output_dim', 0, 'Depth of the output.')
    p.Define(
        'activation', 'RELU',
        'Activation function to use. Options are RELU, RELU6, SIGMOID, '
        'TANH, NONE.')
    p.Define('batch_norm', None, 'Whether or not to apply batch norm.')
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
    p.Define('bn_params',
             BatchNormLayer.Params().Set(decay=0.999),
             'Default params for batch norm layer.')
    p.Define('apply_pruning', False,
             'Whether to prune the weights while training')
    p.Define(
        'pruning_hparams_dict', None, 'Pruning related hyperparameters. A dict '
        'with hyperparameter: value pairs. See google-research.model_pruning.')
    p.Define(
        'use_einsum', True, 'Whether to use tf.einsum for optimizing '
        'computations. When this is set to False, this causes an increase in '
        'TPU memory usage (b/158336491).  When this is set to True, it might '
        ' cause problems with model quantization for on device inference '
        '(b/146421936)')
    p.Define(
        'use_blocked_matmul', False, 'Whether to use blocked matrix '
        'multiplications. This allows for weight updates to be paralellized '
        'across the cores for Shampoo optimizer.')
    p.Define('block_dim', 1024, 'Dimension of the block')
    # Non-default quantization behaviour for weights.
    p.qdomain.Define('weight', None, 'Quantization domain for the weights.')

    p.Define('xla_num_partitions', None,
             'Obsolete. Kept for backwards compatibility.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert symbolic.EvalExpr(symbolic.STATIC_VALUES, p.input_dim) > 0
    assert symbolic.EvalExpr(symbolic.STATIC_VALUES, p.output_dim) > 0
    assert p.activation == 'NONE' or activations.IsSupported(p.activation)
    assert p.xla_num_partitions is None

    if p.batch_norm is None:
      raise RuntimeError(
          'ProjectionLayer.batch_norm not set explicitly for %s' % self.path)
    if p.batch_norm and p.has_bias:
      tf.logging.warning(
          'Projection layer enables both batch_norm and has_bias. '
          'This is generally redundant/wasteful and may introduce '
          'accuracy problems in some inference scenarios.')
    if self._is_bn_folded:
      assert not p.use_blocked_matmul, (
          'bn_fold_weights requires use_blocked_matmul = False')
      assert not p.affine_last, (
          'Folded batchnorm is not compatible with affine_last')

    if p.use_einsum:
      assert not p.use_blocked_matmul, (
          'use_einsum requires use_blocked_matmul = False')

    if p.device_mesh is not None:
      assert not p.use_blocked_matmul, (
          'Enabling xla_sharding requires use_blocked_matmul = False')
      assert p.weight_split_dims_mapping is not None, self.path
      assert len(p.weight_split_dims_mapping) == 2

    if p.batch_norm:
      bn_params = p.bn_params.Copy()
      bn_params.name = p.name
      bn_params.dim = p.input_dim if p.affine_last else p.output_dim

      self.CreateChild('bn', bn_params)
    # TODO(yonghui): implement the variational noise logic.
    self.CreateAqtWeight(
        'projection_aqt', shape=[p.input_dim, p.output_dim], feature_axis=-1)

    if p.pruning_hparams_dict:
      self.compression_op = None
    # only apply compression on tall matrices (input_dim > output_dim)
    self.apply_compression = pruning_utils.ApplyCompression(p) and (
        p.input_dim > p.output_dim)

  def _GetBlockedMatMulInputOutputMultipliers(self):
    """Get number of input and output blocks."""
    p = self.params
    # Number of input and output blocks.
    w_im = p.input_dim // p.block_dim
    w_om = p.output_dim // p.block_dim
    # Add padding if input_dim / output_dim is not divisible by block_dim.
    if p.input_dim % p.block_dim != 0:
      w_im += 1
    if p.output_dim % p.block_dim != 0:
      w_om += 1
    return w_im, w_om

  def _GetBlockedWeightMatrix(self, w):
    p = self.params
    # w is 3D Tensor of shape [i * o, block_dim, block_dim] such that
    # i * block_dim = num_inputs (modulo padding).
    # j * block_dim = num_outputs
    #
    # To efficiently apply forward prop, we transpose and reshape w into
    # shape [i * block_dim, o, block_dim]
    w_im, w_om = self._GetBlockedMatMulInputOutputMultipliers()
    block_dim = p.block_dim
    w_4d = tf.reshape(w, [w_im, w_om, block_dim, block_dim])
    # Transpose to [i, block_dim, o, block_dim].
    w_4d_t = tf.transpose(w_4d, [0, 2, 1, 3])
    w = tf.reshape(w_4d_t, [w_im * block_dim, w_om, block_dim])
    # Slice out padding from the weight matrix.
    if p.input_dim % p.block_dim != 0:
      w = tf.slice(w, [0, 0, 0], [p.input_dim, w_om, block_dim])
    return w

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    if p.use_blocked_matmul:
      w_im, w_om = self._GetBlockedMatMulInputOutputMultipliers()
      w_pc = py_utils.WeightParams(
          shape=[w_im * w_om, p.block_dim, p.block_dim],
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
    else:
      w_pc = py_utils.WeightParams(
          shape=[p.input_dim, p.output_dim],
          init=p.params_init,
          dtype=p.dtype,
          device_mesh=p.device_mesh,
          tensor_split_dims_mapping=p.weight_split_dims_mapping,
          collections=[self.__class__.__name__ + '_vars'])

    if p.apply_pruning:
      mask_w_pc = py_utils.WeightParams(w_pc.shape,
                                        py_utils.WeightInit.Constant(1.0),
                                        p.dtype)
      threshold_w_pc = py_utils.WeightParams([],
                                             py_utils.WeightInit.Constant(0.0),
                                             tf.float32)
    if p.has_bias:
      if p.device_mesh is not None:
        bias_split_dims_mapping = [p.weight_split_dims_mapping[1]]
      else:
        bias_split_dims_mapping = None
      b_pc = py_utils.WeightParams(
          shape=[p.output_dim],
          init=py_utils.WeightInit.Constant(scale=p.bias_init),
          dtype=p.dtype,
          device_mesh=p.device_mesh,
          tensor_split_dims_mapping=bias_split_dims_mapping,
          collections=[self.__class__.__name__ + '_vars'])
    if p.weight_norm:
      g_pc = py_utils.WeightParams(
          shape=[p.output_dim],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])

    weights_var_name = 'w'
    if p.apply_pruning:
      mask_var_name = 'mask'
      threshold_var_name = 'threshold'
      self.CreateVariable(
          mask_var_name, mask_w_pc, theta_fn=None, trainable=False)
      self.CreateVariable(
          threshold_var_name, threshold_w_pc, theta_fn=None, trainable=False)

      def MaskWeightFn(weight):
        return tf.multiply(
            self.AddVN(weight), getattr(self.vars, mask_var_name), 'masked_w')

      self.CreateVariable(weights_var_name, w_pc, theta_fn=MaskWeightFn)
      pruning_utils.AddToPruningCollections(
          getattr(self.vars, weights_var_name), getattr(self.vars,
                                                        mask_var_name),
          getattr(self.vars, threshold_var_name))
    else:
      self.CreateVariable(weights_var_name, w_pc)

    if pruning_utils.ApplyCompression(p):
      pruning_utils.PruningOp.ApplyPruning(p.pruning_hparams_dict, self,
                                           weights_var_name, w_pc, p.dtype,
                                           p.name)
      self.compression_op = pruning_utils.PruningOp.GetLastCompressionOp()

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

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    if self.params.batch_norm:
      self.bn.InstantiateVariables()
    super()._CreateChildrenVariables()

  @classmethod
  def NumOutputNodes(cls, p):
    return p.output_dim

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
      inputs, paddings = self._CastToFPropDtype((inputs, paddings))
      if paddings is None:
        paddings = tf.zeros(
            tf.concat([py_utils.GetShape(inputs)[:-1], [1]], axis=0),
            dtype=inputs.dtype)
      w, b = self._GetWeights(theta, inputs, paddings)
      if pruning_utils.ApplyCompression(p):
        if p.pruning_hparams_dict[
            'compression_option'] == 9 and self.apply_compression:
          # compression_option 9 corresponds to input compression
          # redirect w to point to c
          w = theta.c_matrix_tfvar
      w = self.QWeight(w)

      if p.affine_last:
        # Reversed computation. Does not handle folding.
        out = inputs
        if p.batch_norm:
          out = self.bn.FProp(theta.bn, out, paddings)
        if p.activation != 'NONE':
          if not p.is_inference:
            out = py_utils.CheckNumerics(out)
          out = activations.GetFn(p.activation)(out)
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
            out = activations.GetFn(p.activation)(out)
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
    if p.use_blocked_matmul:
      w = self._GetBlockedWeightMatrix(w)
      if p.weight_norm:
        w = tf.nn.l2_normalize(w, 0)
    else:
      if p.weight_norm:
        w = tf.reshape((theta.g + 1.0) * tf.nn.l2_normalize(w, [0]),
                       py_utils.ToStaticShape([p.input_dim, p.output_dim]))

    if not self._is_bn_folded:
      return w, b

    # If batch norm is fused with weights, then compute the weights as from
    # figure C.8 of https://arxiv.org/pdf/1712.05877.pdf for training and
    # figure C.6 for eval.
    if self.do_eval:
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
    sigma_recip = tf.math.rsqrt(variance + self.bn.epsilon)
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

    if not p.use_blocked_matmul:
      inputs, w = self.ToAqtInputs(
          'projection_aqt', act=inputs, weight=w, w_feature_axis=-1)
      if p.use_einsum:
        if self.apply_compression:
          out = pruning_utils.PruningOp.GetProjectLastDim(
              inputs, w, p.input_dim, p.output_dim, self)
        else:
          out = py_utils.ProjectLastDim(inputs, w, p.input_dim, p.output_dim)
      else:
        out = py_utils.Matmul(
            tf.reshape(inputs, py_utils.ToStaticShape([-1, p.input_dim])), w)
      out = self.FromAqtMatmul('projection_aqt', out)
    else:
      x = tf.reshape(inputs, py_utils.ToStaticShape([-1, p.input_dim]))
      # TODO(shivaniagrawal): There are the following dimmensions: bn, nmk, the
      # the correct thing to do here might be scaling on every m and every k,
      # while we are doing every k only.
      x, w = self.ToAqtInputs(
          'projection_aqt', act=x, weight=w, w_feature_axis=-1)
      out = tf.einsum('bn,nmk->bmk', x, w)
      out = self.FromAqtMatmul('projection_aqt', out)
      # Create an output layer [b, num_outputs].
      bsz = py_utils.GetShape(out)[0]
      out = tf.reshape(out, [bsz, -1])
      if p.output_dim % p.block_dim != 0:
        out_shape = [bsz, p.output_dim]
        out = tf.slice(out, [0, 0], out_shape)

    if b is not None:
      out += b  # NOTE: Bias on matmul is never quantized.
    out = gshard_utils.MeshSplit(out, p.device_mesh,
                                 p.activation_split_dims_mapping)
    return self._ApplyActivationFunction(out, inputs, with_activation, quant)

  def _ApplyActivationFunction(self,
                               out,
                               inputs,
                               with_activation=True,
                               quant=False):
    """Applies the activation function in one step.

    Args:
      out: The result of applying the weight matrix (and bias) to the inputs.
      inputs: FProp inputs.
      with_activation: Whether to also compute the activation function.
      quant: Whether to apply quantization.

    Returns:
      Output tensor reshaped.
    """
    p = self.params
    if with_activation and p.activation != 'NONE':
      if self._pre_activation_qt_name:
        # Track quantization for unfused activation function.
        out = self.QTensor(self._pre_activation_qt_name, out)
      if not p.is_inference:
        out = py_utils.CheckNumerics(out)
      out = activations.GetFn(p.activation)(out)
    if quant:
      out = self.QTensor(self._output_qt_name, out)
    if not p.use_einsum:
      out = tf.reshape(
          out,
          tf.concat([
              py_utils.GetShape(inputs)[:-1],
              py_utils.ToStaticShape([p.output_dim])
          ],
                    axis=0))
    return out

  @classmethod
  def FPropMeta(cls, p, inputs, paddings=None):
    py_utils.CheckShapes((inputs,))
    assert inputs[-1] == p.input_dim
    flops = 0
    in_dim = inputs[-1]
    other_dims = inputs.num_elements() / in_dim
    # matmuls.
    flops += other_dims * p.input_dim * p.output_dim * 2
    # activations.
    flops += other_dims * p.output_dim * activations.GetFlops(p.activation)
    if p.has_bias:
      flops += p.output_dim
    out_shape = tshape.Shape(inputs[:-1] + [p.output_dim])
    if p.batch_norm:
      bn_meta = p.bn_params.cls.FPropMeta(
          p.bn_params.Copy().Set(dim=p.output_dim), out_shape)
      flops += bn_meta.flops
    if p.weight_norm:
      # l2 normalize + element-wise multiply.
      flops += 2 * p.input_dim + 2 * p.input_dim * p.output_dim + 2

    return py_utils.NestedMap(flops=flops, out_shapes=(out_shape,))


class FCLayer(ProjectionLayer):
  """Fully-connected layer (matmul + bias + optional activation)."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.batch_norm = False
    p.has_bias = True
    return p


class FeedForwardNet(quant_utils.QuantizableLayer):
  """A simple multiple layer feedforward network.

  This class represents a stack of fully connected feedforward network. Each
  layer in the network can be configured for whether or not to have batch-norm
  applied to its output, its activation function, whether or not to apply
  dropout to post-activation output.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Depth of the input to the network.')
    p.Define('hidden_layer_dims', [], 'Depth of the hidden layer outputs.')
    p.Define(
        'projection', ProjectionLayer.Params(),
        'Projection layer params. A single parameter that will be shared by'
        'all layers, or a list of params matching the number of layers.')
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
    p.Define(
        'has_bias', None, 'Whether or not to use bias for projection layers.'
        'This can be a None, single bool or a tuple/list of bools having the '
        'same length as the number of layers. If None, the has_bias is set to '
        'True whenever batch_norm is False for each projection layer.')
    p.Define(
        'weight_norm', False,
        'Whether or not to apply weight normalization to weights. This can be '
        'a single bool or a tuple/list of bools having the same length as the '
        'number of layers.')
    p.Define('skip_connections', None, 'Must be None.')
    p.Define(
        'bn_fold_weights', None, 'Force folding the batch normalization '
        'weights in the projection layer.')
    # TODO(rpang): retire weight_split_dims_mapping_list and
    # activation_split_dims_mapping_list. Use
    # {weight,activation}_split_dims_mapping (defined in BaseLayer) instead.
    p.Define('weight_split_dims_mapping_list', None,
             'A list of weight_split_dims_mapping for each sub-layer.')
    p.Define('activation_split_dims_mapping_list', None,
             'A list of activation_split_dims_mapping for each sub-layer.')
    # Non-default quantization behaviour for the weights.
    p.qdomain.Define('weight', None, 'Quantization domain for the weights.')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert symbolic.ToStatic(p.input_dim) > 0
    assert all(symbolic.ToStatic(x) > 0 for x in p.hidden_layer_dims)

    assert p.skip_connections is None
    batch_norm = p.batch_norm
    num_layers = len(p.hidden_layer_dims)
    if isinstance(batch_norm, (list, tuple)):
      assert len(batch_norm) == num_layers
    else:
      batch_norm = [batch_norm] * num_layers
    weight_norm = p.weight_norm
    if isinstance(weight_norm, (list, tuple)):
      assert len(weight_norm) == num_layers
    else:
      weight_norm = [weight_norm] * num_layers

    activation = p.activation
    if isinstance(activation, str):
      activation = [activation] * num_layers
    else:
      assert len(activation) == num_layers
    has_bias = p.has_bias
    if isinstance(has_bias, (list, tuple)):
      assert len(has_bias) == num_layers
    else:
      has_bias = [has_bias] * num_layers
    # Set has_bias to (not batch_norm) if None.
    for i in range(num_layers):
      if has_bias[i] is None:
        has_bias[i] = (not batch_norm[i])

    params_proj_layers = p.projection
    if isinstance(params_proj_layers, (list, tuple)):
      assert len(params_proj_layers) == num_layers
    else:
      params_proj_layers = [params_proj_layers] * num_layers

    params_dropout_layers = p.dropout
    if isinstance(params_dropout_layers, (list, tuple)):
      assert len(params_dropout_layers) == num_layers
    else:
      params_dropout_layers = [params_dropout_layers] * num_layers

    if p.device_mesh is not None:
      weight_split_dims_mapping_list = p.weight_split_dims_mapping_list
      activation_split_dims_mapping_list = p.activation_split_dims_mapping_list
      if activation_split_dims_mapping_list is None:
        activation_split_dims_mapping_list = [None] * num_layers
    else:
      weight_split_dims_mapping_list = [None] * num_layers
      activation_split_dims_mapping_list = [None] * num_layers
    assert len(weight_split_dims_mapping_list) == num_layers
    assert len(activation_split_dims_mapping_list) == num_layers

    # Residual connections work better in the form of:
    #   y = x + Affine(Activation(BatchNorm(x)))
    params_fc_layers = []
    in_dim = p.input_dim
    for i in range(num_layers):
      out_dim = p.hidden_layer_dims[i]
      proj_out_dim = out_dim
      name = '%s_%d' % (p.name, i)
      params_i = params_proj_layers[i].Copy()

      if 'dense_tpl' in params_i:
        dense_params = params_i.dense_tpl
      else:
        dense_params = params_i
      dense_params.Set(
          batch_norm=batch_norm[i],
          weight_norm=weight_norm[i],
          has_bias=has_bias[i],
          bn_fold_weights=p.bn_fold_weights,
          device_mesh=p.device_mesh,
          weight_split_dims_mapping=weight_split_dims_mapping_list[i],
          activation_split_dims_mapping=activation_split_dims_mapping_list[i])
      params_i.Set(
          input_dim=in_dim,
          output_dim=proj_out_dim,
          activation=activation[i],
          name=name)
      params_fc_layers.append(params_i)
      in_dim = out_dim

      if p.qdomain.default is not None:
        params_i.qdomain.default = p.qdomain.default.Copy()
      if p.qdomain.weight is not None:
        params_i.qdomain.weight = p.qdomain.weight.Copy()

    self.CreateChildren('fc', params_fc_layers)
    self.CreateChildren('dropout', params_dropout_layers)

  @property
  def output_dim(self):
    """Returns output dimension of the FeedForwardNet."""
    return self.params.hidden_layer_dims[-1]

  def FPropAllLayers(self, theta, inputs, paddings=None):
    """FProp, returns all layers including the input and output layers."""
    p = self.params
    num_layers = len(self.fc)
    in_dim, layer_in = p.input_dim, inputs
    all_layers = [layer_in]

    for i in range(num_layers):
      layer_in = py_utils.with_dependencies([
          py_utils.assert_shape_match([tf.shape(layer_in)[-1]],
                                      [symbolic.ToStatic(in_dim)])
      ], layer_in)
      out_dim = p.hidden_layer_dims[i]
      layer_out = self.fc[i].FProp(theta.fc[i], layer_in, paddings)
      layer_out = self.dropout[i].FProp(theta.dropout[i], layer_out)
      all_layers.append(layer_out)
      layer_in = layer_out
      in_dim = out_dim
    return all_layers

  def FProp(self, theta, inputs, paddings=None):
    """Computes the output of the feed-forward network.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor.  Shaped [..., input_dim].
      paddings: The paddings tensor.  Shaped [..., 1], where all but the last
        dimension match.

    Returns:
      Output after applying all layers.  Shaped [..., p.hidden_layer_dims[-1]].
    """
    return self.FPropAllLayers(theta, inputs, paddings)[-1]

  @classmethod
  def FPropMeta(cls, p, inputs, paddings=None):
    py_utils.CheckShapes((inputs,))
    assert inputs[-1] == p.input_dim
    flops = 0
    with tf.Graph().as_default():  # throw-away graph.
      instance = p.Instantiate()
      for fc in instance.fc:
        proj_params = fc.params
        proj_shape = tshape.Shape(inputs[:-1] + [proj_params.input_dim])
        proj_meta = proj_params.cls.FPropMeta(proj_params, proj_shape)
        flops += proj_meta.flops
    out_shape = tshape.Shape(inputs[:-1] + [p.hidden_layer_dims[-1]])
    return py_utils.NestedMap(flops=flops, out_shapes=(out_shape,))


class StackingOverTime(base_layer.BaseLayer):
  """Stacking applied along the time axis.

     At each time step of an input sequence, elements are stacked over the
     window of ('left_context' + 1 + 'right_context') steps around the current
     time step. Zeros will be padded to the left or right of the sequence for
     elements around the boundaries. Finally the stacked outputs are emitted
     once every 'stride' steps.

     E.g. if an input sequence is: [4], [1], [9], [3], [5], [2], [8]
     left_context = 1, right_context = 1, stride = 3,
     then the output sequence would be: [0, 4, 1], [9, 3, 5], [2, 8, 0]

     Note that this layer only performs tensor transformation, so there are no
     learnable parameters.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('left_context', 0,
             'Number of time steps to stack on the left to the central step.')
    p.Define('right_context', 0,
             'Number of time steps to stack on the right to the central step.')
    p.Define('stride', 1, 'The stride for emitting the stacked output.')
    p.Define('pad_with_left_frame', False,
             'Whether to use the left frame for padding instead of 0s.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.left_context >= 0
    assert p.right_context >= 0
    assert p.stride >= 1

  @property
  def window_size(self):
    """Returns the stacking window size.

    The output dimension will be window_size * the input dimension.

    Returns:
      Window size.
    """
    p = self.params
    return p.left_context + p.right_context + 1

  def _ApplyStack(self, inputs, pad_value=0.0):
    """The core function to apply the stacking to inputs.

    Args:
      inputs: [batch, time, depth].
      pad_value: the padding value for left/right context.

    Returns:
      [batch, ceil(time / stride), depth * stacking_window_length] tensor.
    """
    p = self.params
    if p.left_context == 0 and p.right_context == 0:
      out = inputs
    else:
      inputs_max_len = py_utils.GetShape(inputs, 3)[1]
      if p.pad_with_left_frame:
        left_pad = tf.repeat(inputs[:, :1, :], repeats=p.left_context, axis=1)
        inputs = tf.concat([left_pad, inputs], axis=1)
        inputs = tf.pad(
            inputs, [[0, 0], [0, p.right_context], [0, 0]],
            constant_values=pad_value)
      else:
        # Add zero paddings to the left and right of the input sequence.
        inputs = tf.pad(
            inputs, [[0, 0], [p.left_context, p.right_context], [0, 0]],
            constant_values=pad_value)

      # Make window_size() copies of the padded sequence with the original
      # sequence length, where each copy is offset by 1 time step.
      pieces = []
      for i in range(self.window_size):
        pieces.append(inputs[:, i:i + inputs_max_len])
      # Apply stacking.
      out = tf.concat(pieces, 2)

    # Apply striding.
    out = out[:, ::p.stride]
    return out

  def FProp(self, inputs, paddings=None):
    """Apply the stacking to inputs along the time axis.

    Args:
      inputs: The inputs tensor. It is expected to be of shape [batch, time,
        feature].
      paddings: The paddings tensor. It is expected to be of shape [batch, time,
        1], where all but the last dimension match inputs. Each value is 0 or 1
        indicating whether a time step of a sequence is padded in the inputs to
        reach the max length in the batch.

    Returns:
      (outputs, out_paddings) pair.
        outputs is of shape [batch, ceil(time / stride), feature * stacking].
        out_paddings is of shape [batch, ceil(time / stride), 1]. out_paddings
        will be 0 if any of the corresponding input padding is 0.
    """
    if paddings is None:
      paddings = tf.zeros(
          tf.concat([py_utils.GetShape(inputs)[:-1], [1]], 0),
          dtype=inputs.dtype)
    inputs = py_utils.with_dependencies(
        [
            # Checks the inputs shape has 3 dimensions.
            py_utils.assert_shape_match(tf.shape(inputs), [-1, -1, -1]),
            # Checks the paddings shape has 3 dimensions, and the last one is 1.
            py_utils.assert_shape_match(tf.shape(paddings), [-1, -1, 1]),
            # Checks the first two dimensions of inputs and paddings match.
            py_utils.assert_shape_match(
                tf.shape(inputs)[:-1],
                tf.shape(paddings)[:-1])
        ],
        inputs)
    p = self.params
    with tf.name_scope(p.name):
      outputs = self._ApplyStack(inputs)

      # Stack the padding values with the same context and stride parameters.
      # Then take the minimum padding values within each stacking window, since
      # an output time step becomes a padded one only if all of the underlying
      # stacked steps are padded ones.
      out_paddings = self._ApplyStack(paddings, pad_value=1)
      out_paddings = tf.reduce_min(out_paddings, axis=2, keepdims=True)

      return outputs, out_paddings

  def Unstack(self, stacked):
    """Inverts stacking over time.

    Given 'stacked' outputs from this StackingOverTime layer,

      stacked, _ = this_layer.FProp(inputs),

    this method attempts to reconstruct the original 'inputs'.

    If stride > window_size, the original input cannot be recovered, and a
    ValueError is raised.

    Otherwise, if right_context + 1 >= stride, this method returns a Tensor that
    is identical to 'inputs' but potentially longer due to paddings.

    If right_context + 1 < stride, this method returns a Tensor that may be up
    to ```stride - right_context - 1``` frames shorter than the original input,
    but identical in the frames that are returned. e.g.::

      left_context = 2, right_context = 1, stride = 4
      input sequence:     1 2 3 4 5 6 7 8
      after padding:  0 0 1 2 3 4 5 6 7 8 0
      windows:
        [0 0 (1) 2] 3 4 5 6 7 8 0
         0 0 1 2 [3 4 (5) 6] 7 8 0
      stacked:
        [[0 0 1 2], [3 4 5 6]]
      unstacked:
        [1 2 3 4 5 6], which is 4 - 1 - 1 = 2 (stride - right_context - 1)
        frames shorter than the original input.

    `Unstack()` can be used to project the outputs of downstream layers back to
    the shape of the original unstacked inputs. For example::

        inputs = ...  # [batch, length, input_dim]
        # [batch, ceil(length / stride), rnn_dim]
        rnn_out = rnn.FProp(stacking.FProp(inputs)[0])
        # [batch, length, rnn_dim]
        back_projected_rnn_out = py_utils.PadOrTrimTo(
            stacking.Unstack(tf.tile(rnn_out, [1, 1, stacking.window_size])),
            py_utils.GetShape(inputs))

    Note this method does not take or return a separate padding tensor. The
    caller is responsible for knowing which of outputs are padding (e.g. based
    on the padding of the original FProp inputs).

    Args:
      stacked: Tensor of shape [batch, time, window_size * feature_dim], assumed
        to be the output of `FProp`.

    Returns:
      The reconstructed input Tensor, with shape
      [batch, (frames - 1) * stride + right_context + 1, feature_dim].

    Raises:
      ValueError: if stride > window_size.
    """
    p = self.params
    if p.stride > self.window_size:
      raise ValueError(
          "Can't invert StackingOverTime with stride (%d) > window_size (%d)" %
          (p.stride, self.window_size))

    # Reshape to allow indexing individual frames within each stacked window.
    batch_size, stacked_length, _ = py_utils.GetShape(stacked, 3)
    stacked = tf.reshape(stacked,
                         [batch_size, stacked_length, self.window_size, -1])

    # Compute the index of the window and frame in 'stacked' where each frame of
    # the original input is located, and extract them with tf.gather_nd.
    # First compute for all except the last window, since these elements have
    # the potential of being looked up from the next window.
    input_indices = tf.range(0, (stacked_length - 1) * p.stride)
    mod = input_indices % p.stride
    in_next_window = tf.cast(tf.greater(mod, p.right_context), tf.int32)
    window_index = input_indices // p.stride + in_next_window
    frame_index = p.left_context + mod - p.stride * in_next_window
    # Now handle the last window explicitly and concatenate onto the existing
    # window_index/frame_index tensors.
    last_window_length = p.right_context + 1
    window_index = tf.concat(
        [window_index,
         tf.fill([last_window_length], stacked_length - 1)],
        axis=0)
    frame_index = tf.concat(
        [frame_index, p.left_context + tf.range(last_window_length)], axis=0)
    # Stack the indices for tf.gather_nd.
    window_and_frame_indices = tf.stack([window_index, frame_index], axis=1)
    window_and_frame_indices = tf.tile(
        tf.expand_dims(window_and_frame_indices, 0), [batch_size, 1, 1])
    return tf.gather_nd(stacked, window_and_frame_indices, batch_dims=1)


class PoolingLayer(quant_utils.QuantizableLayer):
  """Pooling layer, by default performs max-pooling.

  Quantization notes: Unlike the common pattern, the pooling layer inputs
  and output must be quantized to the same range, so it tracks both (vs
  just the output). The preceding layer must have its output quantization
  disabled.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert len(p.window_shape) == 2
    assert len(p.window_stride) == 2
    assert all([x > 0 for x in p.window_shape])
    assert all([x > 0 for x in p.window_stride])
    assert p.pooling_type in ['MAX', 'AVG']

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    self.TrackQTensor('output')

  @classmethod
  def OutputShape(cls, params, in_shape):
    p = params
    return _ComputeConvOutputShape(
        in_shape,
        p.window_stride[0],
        p.window_stride[1],
        padding=p.padding_algorithm)

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    return self.OutputShape(self.params, in_shape)

  def FProp(
      self,
      theta: py_utils.NestedMap,
      inputs: tf.Tensor,
      paddings: Optional[tf.Tensor] = None,
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
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
      An (output, paddings) tensor tuple if paddings is not None, else just
      output tensor.
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
        if p.pooling_type == 'MAX':
          # Fill dtype.min in padded positions.
          min_value = tf.ones_like(inputs) * p.dtype.min
          inputs = py_utils.ApplyPadding(paddings[..., tf.newaxis, tf.newaxis],
                                         inputs, min_value)
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
      if paddings is not None and p.pooling_type == 'AVG':
        # Count the fraction of non-padding elements inside each pooling window.
        in_mask = 1.0 - paddings
        non_padding_ratio = tf.nn.pool(
            in_mask[:, :, tf.newaxis],
            window_shape=(p.window_shape[0],),
            pooling_type='AVG',
            strides=(p.window_stride[0],),
            padding=p.padding_algorithm)
        # Divide by non-padding ratios to eliminate the effect of padded values.
        out *= tf.math.reciprocal_no_nan(non_padding_ratio)[..., tf.newaxis]

      out = self.QTensor('output', out)
      if out_padding is not None:
        out *= tf.expand_dims(tf.expand_dims(1.0 - out_padding, -1), -1)
        return out, out_padding
      return out


class BlurPoolLayer(base_layer.BaseLayer):
  """BlurPool from https://arxiv.org/pdf/1904.11486.pdf.

  This layer blurs the input with a fixed filter and performs subsampling
  afterwards. Only supports 2x1 or 2x2 spatial reduction.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('blur_filter', 'B5', 'One of [R2, T3, B5]; the fixed blur filter.')
    p.Define('subsample_type', '1D', 'Choose between [1D, 2D] subsampling.')
    p.Define('input_channels', None, 'Number of input channels.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.blur_filter in ['R2', 'T3', 'B5']
    assert p.subsample_type in ['1D', '2D']
    assert p.input_channels

    filter_dict = {
        'B5': np.array([1, 4, 6, 4, 1], dtype=np.float32),
        'T3': np.array([1, 2, 1], dtype=np.float32),
        'R2': np.array([1, 1], dtype=np.float32)
    }
    base_filter = filter_dict[p.blur_filter]

    if p.subsample_type == '2D':
      base_filter = base_filter[:, np.newaxis] * base_filter[np.newaxis, :]
    else:
      base_filter = base_filter[:, np.newaxis]
    base_filter /= base_filter.sum()

    self._blur_filter = np.tile(base_filter[..., np.newaxis, np.newaxis],
                                (1, 1, p.input_channels, 1))
    conv_params = DepthwiseConv2DLayer.Params().Set(
        activation='NONE',
        batch_norm=False,
        filter_stride=(1, 1),
        filter_shape=self._blur_filter.shape)

    self.CreateChild('blur_conv', conv_params)

  def FProp(
      self,
      theta: py_utils.NestedMap,
      inputs: tf.Tensor,
      paddings: Optional[tf.Tensor] = None,
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Apply blur pooling.

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
      An (output, paddings) tensor tuple if paddings is not None, else just
      output tensor.
    """
    p = self.params
    if paddings is not None:
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(tf.shape(paddings), [-1, -1]),
          py_utils.assert_shape_match(tf.shape(inputs)[:2], tf.shape(paddings))
      ], inputs)
    # blur
    theta_cp = copy.copy(theta.blur_conv)
    theta_cp.w = tf.convert_to_tensor(self._blur_filter, dtype=p.dtype)
    out, out_padding = self.blur_conv.FProp(theta_cp, inputs, paddings)

    # b/142399320
    # Use stride in blur conv for subsampling once non-square stride gets
    # supported.
    if p.subsample_type == '2D':
      out = out[:, ::2, ::2, :]
    else:
      out = out[:, ::2, :, :]

    if out_padding is not None:
      out_padding = _ComputeConvOutputPadding(
          out_padding, window=2, stride=2, padding_algorithm='SAME')
      out *= (1.0 - out_padding)[..., tf.newaxis, tf.newaxis]
      return out, out_padding
    return out


class SingleShardEmbeddingLayer(base_layer.BaseLayer):
  """Embedding layer that is not sharded.

  This embedding layer is expected to be replicated over all compute devices
  (e.g. tpu cores). It is intended to support small to medium embedding tables
  (< 50k) only.

  This is intended to be a unification of EmbeddingLayer and
  SimpleEmbeddingLayer (and cleanup of both). It is targeting the most common
  use-case we have in speech/nmt/tts/deeprank. Currently we often first
  configure a model using EmbeddingLayer, and then call ChangeToSimpleEmbedding
  to switch to SimpleEmbedding where  we lose some configuration (e.g.
  scale_by_sqrt_dim).

  TODO(lingvo): Implement the matmul option which should be more efficient for
  small vocabs (e.g. < 1k vocab).
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('vocab_size', 0, 'Num tokens in vocab.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define(
        'scale_sqrt_depth', False, 'If set True, activations are scaled'
        ' with sqrt(embedding_dim) in EmbLookup.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert p.embedding_dim > 0
    assert p.name

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=[p.vocab_size, p.embedding_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('emb_var', w_pc)

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
    ids = py_utils.with_dependencies([
        py_utils.assert_between(
            ids, 0, p.vocab_size, name='vocab_id_validation')
    ], ids)
    embs = tf.nn.embedding_lookup(theta.emb_var, tf.reshape(ids, [-1]))
    if p.scale_sqrt_depth:
      embs *= p.embedding_dim**0.5
    embs = py_utils.AddVN(p, embs)
    out_shape = tf.concat([tf.shape(ids), [p.embedding_dim]], 0)
    return tf.reshape(embs, out_shape)

  def FProp(self, theta, ids):
    return self.EmbLookup(theta, ids)


class EmbeddingLayer(base_layer.BaseLayer):
  """Embedding layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert p.embedding_dim > 0
    assert p.max_num_shards > 0
    assert p.name

    total_size = p.vocab_size * p.embedding_dim
    self._actual_shards = min(
        p.max_num_shards,
        int(math.ceil(float(total_size) / self.MIN_PARAMS_PER_SHARD)))
    self._ids_per_shard = int(
        math.ceil(float(p.vocab_size) / self._actual_shards))

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=[self._ids_per_shard, p.embedding_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    # EmbeddingLayer handles vars/theta differently from other layers
    # because when embedding shards are placed on ps, it's more
    # efficiently to do embedding lookups on ps and sends the result
    # back to the worker.
    emb_vars = []
    emb_shards = []
    for i in range(self._actual_shards):
      var_name = 'var_%d' % i
      self.CreateVariable(var_name, w_pc)
      emb_vars.append(self.vars[var_name])
      # NOTE: self.theta[var_name] has transformations such as variational noise
      # applied via theta_fn in self.CreateVariable. For embedding layer we
      # apply variational noise explicitly in EmbLookup, so we do not use
      # self.theta[var_name] here.
      v = self.vars[var_name]
      if not p.on_ps:
        v = tf.identity(v)
      if p.fprop_dtype is not None and p.fprop_dtype != p.dtype:
        v = tf.cast(v, p.fprop_dtype)
      emb_shards.append(v)
      # Remove from _private_vars / _private_thetas to be added later as wm.
      del self._private_vars[var_name]
      del self._private_theta[var_name]
    self._private_vars['wm'] = emb_vars
    self._private_theta['wm'] = emb_shards

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
    ids = py_utils.with_dependencies([
        py_utils.assert_between(
            ids, 0, p.vocab_size, name='vocab_id_validation')
    ], ids)
    embs = tf.nn.embedding_lookup(theta.wm, tf.reshape(ids, [-1]))
    if p.scale_sqrt_depth:
      embs *= p.embedding_dim**0.5
    with tf.name_scope('vn'):
      embs = py_utils.AddVN(p, embs)
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
    p = super().Params()
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
    p.Define(
        'scale_sqrt_depth', False, 'If set True, activations are scaled'
        ' with sqrt(embedding_dim) in EmbLookup.')
    p.Define(
        'pruning_hparams_dict', None, 'Pruning related hyperparameters. A dict '
        'with hyperparameter: value pairs. See google-research.model_pruning.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert symbolic.ToStatic(p.embedding_dim) > 0

    valid_fprop_modes = ['loop', 'matmul', 'gather']
    self._fprop_mode = p.fprop_mode
    if not self._fprop_mode:
      self._fprop_mode = 'matmul' if p.use_matmul else 'gather'
    assert self._fprop_mode in valid_fprop_modes, (
        'fprop_mode must be one of %r' % valid_fprop_modes)

    _, weight_shape = self._GetWeightShape()
    self.CreateAqtWeight('emb_aqt', shape=weight_shape, feature_axis=-1)
    if p.pruning_hparams_dict:
      self.compression_op = None
    self.apply_compression = pruning_utils.ApplyCompression(p)

  def _FpropImpl(self, embs, ids_vec):
    """The embedding lookup implementation."""
    p = self.params
    emb_shape_suf, weight_shape = self._GetWeightShape()

    def EmbBprop(xs, ys, dys):
      """Embedding backprop.

      Effectively, it computes:
        num = size of xs.ids_vec
        dembs = zeros_like(xs.embs)
        for i in range(num):
          dembs[xs.ids_vec[i], :] += dys[i, :]
        return dembs, zeros_like(xs.ids_vec)

      Args:
        xs: A NestedMap containing:
          - embs: The embedding matrix. Unused in the backprop.
          - ids_vec: A vector of int32 embedding ids.
        ys: Required by py_utils._DefineDefun, not used here.
        dys: A matrix of size (size of xs.ids_vec, embedding dims).

      Returns:
        A NestedMap containing:

          - embs: A matrix of the same shape of xs.embs. Gradients for xs.embs.
          - ids_vec: Zeros. Same shape as xs.ids_vec.
      """
      del ys
      num = tf.shape(xs.ids_vec)[0]
      dembs = inplace_ops.empty(weight_shape, py_utils.FPropDtype(p), init=True)
      if len(weight_shape) != 2:
        dys_shape = tf.shape(dys)
        dys = tf.reshape(dys, [dys_shape[0]] + emb_shape_suf)

      def EmbBpropLoop(i, state):
        # row_id = state.ids_vec[i]
        row_id = tf.gather(state.ids_vec, i)
        # row = state.drets[i]
        row = tf.reshape(tf.gather(state.drets, i), [1] + emb_shape_suf)
        # state.dembs[row_id] = row
        state.dembs = inplace_ops.alias_inplace_add(state.dembs, [row_id], row)
        return state

      dembs = py_utils.ForLoop(
          body=EmbBpropLoop,
          start=0,
          limit=num,
          delta=1,
          loop_state=py_utils.NestedMap(
              ids_vec=xs.ids_vec, drets=dys, dembs=dembs)).dembs

      if p.scale_sqrt_depth:
        dembs *= p.embedding_dim**0.5

      return py_utils.NestedMap(embs=dembs, ids_vec=tf.zeros_like(ids_vec))

    def EmbFprop(xs):
      """Embedding forward prop.

      Effectively, it computes:
        num = size of xs.ids_vec
        rets = zeros([num, embedding dim])
        for i in range(num):
          rets[i, :] = xs.embs[xs.ids_vec[i], :]
        return rets

      Args:
        xs: A NestedMap containing:
          - embs: The embedding matrix.
          - ids_vec: A vector of int32 embedding ids.

      Returns:
        The result of embedding lookups. A matrix of shape
        [num ids in xs.ids_vec, embedding dims].
      """
      num = tf.shape(xs.ids_vec)[0]
      rets = inplace_ops.empty([num] + emb_shape_suf, py_utils.FPropDtype(p))

      def EmbFpropLoop(i, state):
        # row_id = state.ids_vec[i]
        row_id = tf.gather(state.ids_vec, i)
        # row = state.embs[row_id]
        row = tf.reshape(tf.gather(state.embs, row_id), [1] + emb_shape_suf)
        # state.rets[i] = row
        state.rets = inplace_ops.alias_inplace_update(state.rets, [i], row)
        return state

      rets = py_utils.ForLoop(
          body=EmbFpropLoop,
          start=0,
          limit=num,
          delta=1,
          loop_state=py_utils.NestedMap(
              embs=xs.embs, ids_vec=xs.ids_vec, rets=rets)).rets
      if len(weight_shape) > 2:
        rets = tf.reshape(rets, [num, symbolic.ToStatic(p.embedding_dim)])
      return rets

    def EmbMatmul(xs):
      """Lookups embedding vectors by doing Matmul with one-hot vector."""
      # lhs[i, j] is True iff xs.ids_vec[i] == j.
      lhs = tf.equal(
          tf.expand_dims(xs.ids_vec, 1),
          tf.range(p.vocab_size, dtype=xs.ids_vec.dtype))
      return tf.matmul(tf.cast(lhs, xs.embs.dtype), xs.embs)

    def EmbGather(xs):
      """Lookups embedding vectors."""
      if not self.do_eval:
        # If tf.gather is used, the gradient for the wm will be represented as
        # IndexedSlices which is sparse. tf.tpu.cross_replica_sum turns
        # IndexedSlices into a dense tensor with undefined first dimension.
        # This may cause issues on TPU so instead we just wrap this with
        # tf.identity which allows tf.tpu.cross_replica_sum to properly compute
        # the first dim.
        return tf.nn.embedding_lookup(tf.identity(xs.embs), xs.ids_vec)
      else:
        # The above fix tf.tpu_cross_replica_sum causes issues
        # on inference graphs in which the EmbeddingLayer is on the host
        # as the tf.identity prevents ResourceGather from being used.
        return tf.nn.embedding_lookup(xs.embs, xs.ids_vec)

    xs = py_utils.NestedMap(embs=embs, ids_vec=ids_vec)
    if self._fprop_mode == 'matmul':
      return py_utils.CallDefun(EmbMatmul, xs)
    elif self._fprop_mode == 'loop':
      return py_utils.CallDefun(
          EmbFprop, xs, bak=EmbBprop, bak_as_function=True)
    elif self._fprop_mode == 'gather':
      return EmbGather(xs)

  def _GetWeightShape(self):
    p = self.params
    if py_utils.tpu_compat() and self._fprop_mode != 'matmul':
      if p.use_3d_weight_tensor:
        assert symbolic.ToStatic(p.embedding_dim) % 128 == 0
        emb_shape_suf = [symbolic.ToStatic(p.embedding_dim) // 128, 128]
      else:
        emb_shape_suf = [symbolic.ToStatic(p.embedding_dim)]
    else:
      emb_shape_suf = [symbolic.ToStatic(p.embedding_dim)]
    weight_shape = [p.vocab_size] + emb_shape_suf
    return emb_shape_suf, weight_shape

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    _, weight_shape = self._GetWeightShape()

    # Define weights
    pc = py_utils.WeightParams(
        shape=weight_shape,
        init=p.params_init,
        dtype=p.dtype,
        tensor_split_dims_mapping=p.weight_split_dims_mapping,
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
        return tf.multiply(self.AddVN(weight), self.vars.mask, 'masked_weights')

      self.CreateVariable('wm', pc, theta_fn=MaskWeightFn)
      pruning_utils.AddToPruningCollections(self.vars.wm, self.vars.mask,
                                            self.vars.threshold)
    else:
      self.CreateVariable('wm', pc, theta_fn=None)

    if pruning_utils.ApplyCompression(p):
      pruning_utils.PruningOp.ApplyPruning(p.pruning_hparams_dict, self, 'wm',
                                           pc, p.dtype, p.name)
      self.compression_op = pruning_utils.PruningOp.GetLastCompressionOp()

  def EmbLookupDefaultTheta(self, ids):
    """Lookups embedding vectors for ids."""
    return self.FProp(self.theta, ids)

  def EmbLookup(self, theta, ids):
    return self.FProp(theta, ids)

  def EmbLookupDefaultThetaOnCpu(self, ids):
    """A faster path for CPU inference than the default gather."""
    p = self.params
    embs = tf.nn.embedding_lookup(self.theta.wm, tf.reshape(ids, [-1]))
    out_shape = tf.concat([tf.shape(ids), [symbolic.ToStatic(p.embedding_dim)]],
                          0)
    if p.scale_sqrt_depth:
      embs *= p.embedding_dim**0.5
    return tf.reshape(embs, out_shape)

  def _FlatFProp(self, theta, ids):
    """Lookups embedding vectors for ids.

    Args:
      theta: Named tuple collection of weights for the layer.
      ids: A rank-N int32 tensor.

    Returns:
      A tuple of the flattened inputs to the embedding lookup, and a tensor that
      is ready to be reshaped into the final shape in FProp.
    """
    if not isinstance(ids, tf.Tensor):
      tf.logging.warning('ids should be a tf.Tensor!')
      ids = tf.convert_to_tensor(ids, tf.int32)
    elif ids.dtype != tf.int32:
      tf.logging.warning('ids should be tf.int32, but is %s!', ids.dtype)
      ids = tf.cast(ids, tf.int32)
    p = self.params
    ids = py_utils.with_dependencies([
        py_utils.assert_between(
            ids, 0, p.vocab_size, name='vocab_id_validation')
    ], ids)
    flat_ids = tf.reshape(ids, [-1])
    wm = self.QWeight(theta.wm)
    wm = self.ToAqtWeight('emb_aqt', wm, feature_axis=-1)
    if self.apply_compression:
      embs_result = pruning_utils.PruningOp.GetEmbeddingLookupResult(
          theta, flat_ids, self._fprop_mode, self)
    else:
      embs_result = self._FpropImpl(wm, flat_ids)

    embs_result = self.FromAqtWeight('emb_aqt', embs_result)
    with tf.name_scope('vn'):
      embs_result = py_utils.AddVN(p, embs_result)

    if p.scale_sqrt_depth:
      embs_result *= p.embedding_dim**0.5
    return flat_ids, embs_result

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
    _, embs_result = self._FlatFProp(theta, ids)
    out_shape = tf.concat(
        [tf.shape(ids), [symbolic.ToStatic(self.params.embedding_dim)]], 0)
    embs_result = tf.reshape(embs_result, out_shape)
    embs_result = gshard_utils.MeshSplit(embs_result, p.device_mesh,
                                         p.activation_split_dims_mapping)
    return embs_result


class EinsumEmbeddingLayer(base_layer.BaseLayer):
  """An embedding layer that uses einsum to avoid reshaping."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('vocab_size', 0,
             'Depth of the input. I.e., the number of classes.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define(
        'scale_sqrt_depth', False, 'If set True, activations are scaled'
        ' with sqrt(embedding_dim) in EmbLookup.')
    p.params_init = py_utils.WeightInit.Uniform(1.)
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert symbolic.ToStatic(p.embedding_dim) > 0

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params

    # Define weights
    pc = py_utils.WeightParams(
        shape=[p.vocab_size, symbolic.ToStatic(p.embedding_dim)],
        init=p.params_init,
        dtype=p.dtype,
        tensor_split_dims_mapping=p.weight_split_dims_mapping,
        collections=[self.__class__.__name__ + '_vars'])
    # Apply VN on theta.wm so that this layer can be used within a recurrent
    # loop.
    self.CreateVariable('wm', pc, theta_fn=self.AddVN)

  def EmbLookup(self, theta, ids):
    return self.FProp(theta, ids)

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
    # Emulate tf.nn.embedding_lookup(theta.wm, ids) with tf.einsum.
    embs_result = py_utils.ProjectLastDim(
        tf.one_hot(ids, p.vocab_size),
        theta.wm,
        input_dim=p.vocab_size,
        output_dim=p.embedding_dim)
    if p.scale_sqrt_depth:
      embs_result *= p.embedding_dim**0.5
    embs_result = gshard_utils.MeshSplit(embs_result, p.device_mesh,
                                         p.activation_split_dims_mapping)
    return embs_result


class OneHotEmbeddingLayer(base_layer.BaseLayer):
  """Generates one-hot embeddings with uncertainties."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('vocab_size', 0,
             'Depth of the input. I.e., the number of classes.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define('uncertainty', 0.0, 'Uncertainty of the correct ID.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.vocab_size > 1
    assert p.embedding_dim == p.vocab_size

  def EmbLookupDefaultTheta(self, ids):
    """Lookups embedding vectors for ids."""
    return self.FProp(self.theta, ids)

  def EmbLookup(self, theta, ids):
    return self.FProp(theta, ids)

  def FProp(self, theta, ids):
    """Lookups embedding vectors for ids.

    Args:
      theta: Named tuple collection of weights for the layer.
      ids: A rank-N int32 tensor.

    Returns:
      A rank-(N+1) params.dtype tensor.
      embs[indices, :] is the embedding vector for ids[indices].
    """
    del theta
    p = self.params
    ids = py_utils.with_dependencies([
        py_utils.assert_between(
            ids, 0, p.vocab_size, name='vocab_id_validation')
    ], ids)
    low_confidence = p.uncertainty / tf.cast(p.vocab_size - 1, tf.float32)
    high_confidence = 1.0 - p.uncertainty
    embs_result = tf.one_hot(
        ids,
        depth=p.vocab_size,
        on_value=high_confidence,
        off_value=low_confidence)
    if p.fprop_dtype is not None:
      embs_result = tf.cast(embs_result, p.fprop_dtype)
    return embs_result


class PositionalEmbeddingLayer(base_layer.BaseLayer):
  """Generates sinusoidals with respect to the position in time and dimension.

  Implements the positional embedding layer from 'Attention is All You Need',
  the Transformer Network.

  Code and comments are adapted from tensor2tensor/layers/common_attention.py
  """

  @classmethod
  def Params(cls):
    p = super().Params()
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
    p.Define(
        'frequency_scaling', False,
        'Introduces a trainable frequency scaling parameter (a scalar) that'
        ' multiplies the frequency of the sinusoids.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.min_timescale
    assert p.max_timescale
    assert p.embedding_dim % 2 == 0

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    if p.trainable_scaling:
      pc = py_utils.WeightParams(
          shape=[1],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      self.CreateVariable('scale', pc)
    if p.frequency_scaling:
      pc = py_utils.WeightParams(
          shape=[1],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      self.CreateVariable('freq_scale', pc)

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
    seq_length = py_utils.GetShape(position)[1]
    num_timescales = p.embedding_dim // 2
    log_timescale_increment = (
        math.log(float(p.max_timescale) / float(p.min_timescale)) / tf.maximum(
            tf.cast(1.0, py_utils.FPropDtype(p)),
            tf.cast(num_timescales, py_utils.FPropDtype(p)) - 1))

    inv_timescales = p.min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), py_utils.FPropDtype(p)) *
        -log_timescale_increment)

    scaled_time = tf.expand_dims(position, 2) * tf.reshape(
        inv_timescales, [1, 1, -1])

    if p.frequency_scaling:
      scaled_time *= (1.0 + theta.freq_scale)

    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
    signal = tf.pad(
        signal, [[0, 0], [0, 0], [0, tf.math.floormod(p.embedding_dim, -1)]])
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


class RelativePositionalEmbeddingLayer(base_layer.BaseLayer):
  """Relative positional embedding.

  Section 3.2 of https://arxiv.org/pdf/1803.02155.pdf
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'radius', None,
        'Radius of the relative window size. Distance are clipped to '
        '[-radius, radius].')
    p.Define('dim', None, 'Dimension of embedding.')
    return p

  def __init__(self, params):
    super().__init__(params)
    params = self.params
    if not isinstance(params.radius, numbers.Integral) or params.radius <= 0:
      raise ValueError('params.radius must be a positive int, but is %s' %
                       params.radius)
    if not isinstance(params.dim, numbers.Integral) or params.dim <= 0:
      raise ValueError('params.dim must be a positive int, but is %s' %
                       params.radius)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    pc = py_utils.WeightParams(
        shape=[2 * self.params.radius + 1, self.params.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=self.params.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('w', pc)

  def FProp(self, theta, relative_distance):
    """Computes relative positional embedding.

    Args:
      theta: A NestedMap of Tensors of layer weights.
      relative_distance: A Tensor.

    Returns:
      A Tensor of shape relative_distance.shape + [params.dim]
    """
    params = self.params
    clipped_indices = tf.clip_by_value(relative_distance, -params.radius,
                                       params.radius)
    # Right-shift indices to make them all non-negative.
    calibrated_indices = clipped_indices + params.radius
    return tf.gather_nd(theta.w, tf.expand_dims(calibrated_indices, -1))


class SinusoidalPositionalEmbeddingLayer(base_layer.BaseLayer):
  """Generates sinusoidals with respect to the position in time and dimension.

  Implements the a variant of the positional embedding layer from 'Attention is
  All You Need', the Transformer Network that doesn't require tuning of the
  max_timescale/min_timescale. See this blog post and Ron's colab.
  https://kazemnejad.com/blog/transformer_architecture_positional_encoding
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('embedding_dim', 0, 'Dimension of the embedding to be generated.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.embedding_dim % 2 != 0:
      raise ValueError('embedding_dim needs to be even.')

  def FProp(self, theta, seq_length):
    """Generates a Tensor of sinusoids with different frequencies.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      seq_length: Sequence length of the embeddings to be generated

    Returns:
      a Tensor of shape [seq_length, embedding_dim].
    """
    p = self.params
    positions = tf.cast(tf.range(seq_length), py_utils.FPropDtype(p))
    num_timescales = p.embedding_dim // 2
    freq = tf.range(
        1, num_timescales + 1,
        dtype=py_utils.FPropDtype(p)) * (2 * math.pi / seq_length)
    scaled_pos = tf.matmul(positions[:, tf.newaxis], freq[tf.newaxis, :])
    sincos = tf.concat([tf.sin(scaled_pos), tf.cos(scaled_pos)], axis=-1)
    return tf.reshape(sincos, [seq_length, -1])


class SoftmaxLayer(quant_utils.QuantizableLayer):
  """Base class for softmax layers."""

  @classmethod
  def Params(cls):
    """Params for SoftmaxLayer."""
    p = super().Params()
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
    p.qdomain.Define('logits', None, 'Quantization domain for logits.')
    p.qdomain.Define('weight', None, 'Quantization domain for the weights.')
    return p

  @property
  def wm_transposed(self):
    """Whether wm (as returned by DenseWeights) is transposed."""
    return False

  def DenseWeights(self, theta):
    """Returns a NestedMap containing dense weights for 'wm'/'bias'."""
    raise NotImplementedError(
        f'DenseWeights is not implemented: {self.params.cls}.')

  def Logits(self, **unused):
    """Returns the logits computed before the softmax."""
    raise NotImplementedError(
        f'GetLogits is not implemented: {self.params.cls}.')

  def XentLossFromLogits(self, **unused):
    """Returns the Xent loss from pre-computed logits."""
    raise NotImplementedError(
        f'XentLossFromLogits is not implemented: {self.params.cls}.')

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
        f'Subclasses of SoftmaxLayer must implement _FProp2D: {self.params.cls}'
    )

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
    inputs = [
        tf.reshape(x, py_utils.ToStaticShape([-1, p.input_dim])) for x in inputs
    ]
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
    p = super().Params()
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
    p.Define(
        'pruning_hparams_dict', None,
        'Pruning related hyperparameters. A dict with hyperparameter: value'
        'pairs. See google-research.model_pruning.')
    p.Define(
        'use_num_classes_major_weight', False,
        'Whether to use num_classes as major dimension for weight params. '
        'This shows performance benefit especially when sharing embedding '
        'and softmax. By removing the transpose before gather, it allows '
        'better XLA fusions and optimizations.')

    p.Define(
        'use_bias', True, 'Whether or not to use a bias variable.'
        'Not using bias is not compatible with sampled softmax '
        '(num_sampled > 0).')
    p.Define('bias_init', 0, 'Weight initialization constant for bias.')
    return p

  def __init__(self, params):
    """Constructs a SimpleFullSoftmax layer."""
    super().__init__(params)
    p = self.params
    assert p.name

    # We shard params across the class dimension.
    assert p.num_classes % p.num_shards == 0
    if not p.use_bias:
      assert p.num_sampled == 0, 'Sampled softmax requires bias.'

    if p.num_shards == 1:
      self.CreateAqtWeight(
          'softmax_aqt', shape=[p.input_dim, p.num_classes], feature_axis=-1)
    self.compression_ops = []

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params

    num_classes_per_shard = p.num_classes // p.num_shards
    # When using sampled soft-max we'd rather work with weights of
    # shape=[num_classes_per_shard, p.input_dim] to avoid an expensive transpose
    # op before computing the sampled_softmax_loss.
    self._transpose_weight_params = False
    weights_shard_shape = [p.input_dim, num_classes_per_shard]
    weight_split_dims_mapping = p.weight_split_dims_mapping
    bias_split_dims_mapping = (None if weight_split_dims_mapping is None else
                               weight_split_dims_mapping[-1:])
    if p.num_sampled or p.use_num_classes_major_weight:
      self._transpose_weight_params = True
      weights_shard_shape = [num_classes_per_shard, p.input_dim]
      if weight_split_dims_mapping is not None:
        weight_split_dims_mapping = weight_split_dims_mapping[::-1]

    pc = py_utils.WeightParams(
        shape=weights_shard_shape,
        init=p.params_init,
        dtype=p.dtype,
        tensor_split_dims_mapping=weight_split_dims_mapping,
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

        def MaskWeightFn(mask_var_name, weight):
          return tf.multiply(
              self.AddVN(weight), getattr(self.vars, mask_var_name),
              'masked_weights')

        self.CreateVariable(
            weights_var_name,
            pc,
            theta_fn=functools.partial(MaskWeightFn, mask_var_name))
        pruning_utils.AddToPruningCollections(
            getattr(self.vars, weights_var_name),
            getattr(self.vars, mask_var_name),
            getattr(self.vars, threshold_var_name))

      else:
        self.CreateVariable(weights_var_name, pc, self.AddVN)
        if pruning_utils.ApplyCompression(p):
          # matrix compression path. call ApplyPruning to setup compression op
          pruning_utils.PruningOp.ApplyPruning(p.pruning_hparams_dict, self,
                                               weights_var_name, pc, p.dtype,
                                               p.name)
          self.compression_ops.append(
              pruning_utils.PruningOp.GetLastCompressionOp())

    pc = py_utils.WeightParams(
        shape=[num_classes_per_shard],
        init=py_utils.WeightInit.Constant(scale=p.bias_init),
        dtype=p.dtype,
        tensor_split_dims_mapping=bias_split_dims_mapping,
        collections=[self.__class__.__name__ + '_vars'])
    if p.use_bias:
      for i in range(p.num_shards):
        self.CreateVariable('bias_%d' % i, pc, self.AddVN)

    self.TrackQTensor('inputs')
    self.TrackQTensor('logits', domain='logits')

  def _GetInputs(self, inputs):
    if isinstance(inputs, list):
      assert len(inputs) == 1
      return inputs[0]
    return inputs

  @property
  def wm_transposed(self):
    return self._transpose_weight_params

  def DenseWeights(self, theta):
    p = self.params
    # Add per-step noise if configured so.
    concat_axis = 1
    if self._transpose_weight_params:
      concat_axis = 0
    weights = [
        self.QWeight(theta['weight_%d' % i]) for i in range(p.num_shards)
    ]
    new_theta = py_utils.NestedMap()
    if p.use_bias:
      biases = [self.QWeight(theta['bias_%d' % i]) for i in range(p.num_shards)]
      new_theta.bias = py_utils.AddVN(
          p, tf.concat(biases, axis=0), per_step=True)
    if p.num_shards == 1:
      new_theta.wm = py_utils.AddVN(p, weights[0], per_step=True)
    else:
      new_theta.wm = py_utils.AddVN(
          p, tf.concat(weights, axis=concat_axis), per_step=True)
    return new_theta

  def _LogitsUsingConcatenatedWeightsHelper(self, theta, inputs):
    p = self.params
    inputs = self.QTensor('inputs', inputs)
    wm = self.QWeight(theta.wm)
    if p.num_shards == 1:
      if self._transpose_weight_params:
        # TODO(shivaniagrawal): having two transpose is expensive, we should
        # optimize this by allowing feature axis to other that last axis.
        # For this particular case num_classes is the first dimension, transpose
        # of weight would make it last dimension; we scale on the axis
        # corresponding to num_classes.
        inputs, wm = self.ToAqtInputs(
            'softmax_aqt',
            act=inputs,
            weight=tf.transpose(wm),
            w_feature_axis=-1)
        wm = tf.transpose(wm)
      else:
        inputs, wm = self.ToAqtInputs(
            'softmax_aqt', act=inputs, weight=wm, w_feature_axis=-1)

      if pruning_utils.ApplyCompression(p):
        # compression path. call GetMatmulResult.
        # inputs and wm are both rank 2. using GetMatmulResult
        logits = pruning_utils.PruningOp.GetMatmulResult(
            inputs, wm, self, transpose_b=self._transpose_weight_params)
      else:
        logits = py_utils.Matmul(
            inputs, wm, transpose_b=self._transpose_weight_params)

      # We used weight's output_dimension, i.e. p.num_classes as feature axis
      # while quantizing weight.
      logits = self.FromAqtMatmul('softmax_aqt', logits)

    else:
      logits = py_utils.Matmul(
          inputs, wm, transpose_b=self._transpose_weight_params)

    if p.use_bias:
      bias = self.QWeight(theta.bias)

      # x * w + b
      # Note that theta.wm and theta.bias are transformed to concated/clipped
      # by caller.
      logits = tf.nn.bias_add(logits, bias)

    # Clip logits by range.
    # Note that this is generally not used in conjunction with quantization and
    # shouldn't be needed at inference time as the quantized matmul above will
    # take care of clipping naturally based on the data type and qparams.
    abs_max = p.logits_abs_max
    if abs_max is not None and not p.is_inference:
      abs_min = -abs_max  # pylint: disable=invalid-unary-operand-type
      logits = py_utils.clip_by_value(logits, abs_min, abs_max)
    return logits

  def _LogitsUsingConcatenatedWeights(self, theta, inputs):
    logits = self._LogitsUsingConcatenatedWeightsHelper(theta, inputs)
    return self.QTensor('logits', logits)

  def SimpleLogits(self, theta, inputs):
    """Returns the simple logits computed before the softmax.

    Compared to the Logits function, this one has only weights, no bias for the
    linear projection.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A tensor with the shape [N, input_dim].

    Returns:
      logits: [N, num_classes]
    """
    inputs = self.QTensor('inputs', inputs)
    theta = self.DenseWeights(theta)
    wm = self.QWeight(theta.wm)
    logits = py_utils.Matmul(
        inputs, wm, transpose_b=self._transpose_weight_params)

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
        self.DenseWeights(theta), self._GetInputs(inputs))

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
        theta=self.DenseWeights(theta),
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
      per_example_xent, per_example_argmax = self.XentLossFromLogits(
          theta, logits, class_weights, class_ids, class_probabilities)
    elif p.chunk_size:
      class_ids = py_utils.HasShape(class_ids, [-1, 1])
      per_example_xent, per_example_argmax = self._XentLossByChunk(
          theta, inputs, class_ids)
    elif p.num_sampled == 0 or self.do_eval:
      per_example_xent, per_example_argmax = self.XentLossFromLogits(
          theta, logits, class_weights, class_ids, class_probabilities)
    else:  # Use sampled soft-max in training mode with p.num_sampled set.
      assert p.num_sampled > 0
      assert p.use_bias
      tf.logging.vlog(
          0, 'Using sampled_softmax_loss(..., num_sampled=%d, '
          'num_classes=%d) in SimpleFullSoftmax::_FProp2D', p.num_sampled,
          p.num_classes)
      # tf.nn.sampled_softmax_loss will call tf.embedding_lookup. And when
      # tf.embedding_lookup is used, the gradient for the weights will be
      # represented as IndexedSlices which is sparse. tf.tpu.cross_replica_sum
      # turns IndexedSlices into a dense tensor with undefined first dimension.
      # This may cause issues on TPU so instead we just wrap this with
      # tf.identity which allows tf.tpu.cross_replica_sum to properly compute
      # the first dim.
      per_example_xent = tf.nn.sampled_softmax_loss(
          weights=[
              tf.identity(theta['weight_%d' % i]) for i in range(p.num_shards)
          ],
          biases=tf.concat([theta['bias_%d' % i] for i in range(p.num_shards)],
                           axis=0),
          labels=tf.reshape(class_ids, [-1, 1]),
          inputs=self._GetInputs(inputs),
          num_sampled=p.num_sampled,
          num_classes=p.num_classes,
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

  def XentLossFromLogits(self,
                         theta,
                         logits,
                         class_weights,
                         class_ids=None,
                         class_probabilities=None):
    """Computes cross-entropy, argmax etc. from logits."""
    p = self.params
    assert logits is not None
    per_example_argmax = py_utils.ArgMax(logits)
    if class_probabilities is not None:
      per_example_xent = tf.nn.softmax_cross_entropy_with_logits(
          labels=class_probabilities, logits=logits)
    elif p.num_sampled == 0 or self.do_eval:
      assert class_ids is not None
      tf.logging.vlog(
          0, 'Using sparse_softmax_cross_entropy_with_logits() in '
          'SimpleFullSoftmax::_FProp2D logits_shape=%r',
          py_utils.GetShape(logits))
      per_example_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.reshape(class_ids, [-1]), logits=logits)
    else:
      raise ValueError(
          'This set of arguments is not supported for XentLossFromLogits.')
    return per_example_xent, per_example_argmax


class FocalFullSoftmax(SimpleFullSoftmax):
  """An extended softmax layer with focal loss.

  Focal loss: https://arxiv.org/abs/1708.02002, Eq (3) and (4).
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'focal_loss_alpha', None,
        'The weighting factor alpha with shape [#classes] for focal loss.')
    p.Define('focal_loss_gamma', None,
             'The modulating factor scalar gamma for focal loss.')
    return p

  def XentLossFromLogits(self,
                         theta,
                         logits,
                         class_weights,
                         class_ids=None,
                         class_probabilities=None):
    """Computes cross-entropy, argmax etc. from logits."""
    p = self.params
    assert logits is not None
    per_example_argmax = py_utils.ArgMax(logits)
    if class_ids is not None:
      class_ids = tf.reshape(class_ids, [-1])
    per_example_xent = py_utils.SoftmaxCrossEntropyFocalLoss(
        logits=logits,
        label_ids=class_ids,
        label_probs=class_probabilities,
        alpha=p.focal_loss_alpha,
        gamma=p.focal_loss_gamma)
    return per_example_xent, per_example_argmax


class EinsumSoftmax(base_layer.BaseLayer):
  """A simple softmax layer implemented with Einsum to avoid reshape ops."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the input.')
    p.Define('num_classes', 0, 'Total number of target classes.')
    p.Define(
        'focal_loss_alpha', None,
        'The weighting factor alpha with shape [#classes] for focal loss.')
    p.Define('focal_loss_gamma', None,
             'The modulating factor scalar gamma for focal loss.')
    p.Define('use_bias', True, 'Whether or not to use a bias variable.')
    return p

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params

    weight_split_dims_mapping = p.weight_split_dims_mapping
    bias_split_dims_mapping = (None if weight_split_dims_mapping is None else
                               weight_split_dims_mapping[-1:])
    w_pc = py_utils.WeightParams(
        shape=(p.input_dim, p.num_classes),
        init=p.params_init,
        dtype=p.dtype,
        tensor_split_dims_mapping=weight_split_dims_mapping,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('w', w_pc)
    if p.use_bias:
      self.CreateVariable(
          'b',
          py_utils.WeightParams(
              shape=[p.num_classes],
              init=py_utils.WeightInit.Constant(0.0),
              dtype=p.dtype,
              tensor_split_dims_mapping=bias_split_dims_mapping,
              collections=[self.__class__.__name__ + '_vars']))

  @property
  def wm_transposed(self):
    """Whether wm (as returned by DenseWeights) is transposed."""
    return False

  def DenseWeights(self, theta):
    ret = py_utils.NestedMap()
    ret.wm = theta.w
    if self.params.use_bias:
      ret.bias = theta.b
    return ret

  def Logits(self, theta, inputs):
    """Returns the logits computed before the softmax.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: a single tensor with the shape [..., input_dim].

    Returns:
      logits [..., num_classes].
    """
    p = self.params
    inputs = self._CastToFPropDtype(inputs)
    if inputs.shape is not None and inputs.shape.rank < 26:
      # A common path.
      s = ''.join([chr(x) for x in range(97, 123)])  # abc...xyz
      r = inputs.shape.rank
      logits = tf.einsum('{0}y,yz->{0}z'.format(s[:r - 1]), inputs, theta.w)
    else:
      logits = tf.einsum('...d,dv->...v', inputs, theta.w)
    logits = gshard_utils.MeshSplit(
        logits,
        p.device_mesh,
        tensor_split_dims_mapping=p.activation_split_dims_mapping)
    if p.use_bias:
      logits = tf.nn.bias_add(logits, theta.b)
    return logits

  def XentLossFromLogits(self,
                         theta,
                         logits,
                         class_weights,
                         class_ids=None,
                         class_probabilities=None):
    """Computes cross-entropy, argmax etc. from logits."""
    p = self.params
    assert logits is not None
    per_example_argmax = py_utils.ArgMax(logits)
    per_example_xent = py_utils.SoftmaxCrossEntropyFocalLoss(
        logits=logits,
        label_ids=class_ids,
        label_probs=class_probabilities,
        alpha=p.focal_loss_alpha,
        gamma=p.focal_loss_gamma)
    return per_example_xent, per_example_argmax

  def FProp(self, theta, inputs, class_weights, *args, **kwargs):
    logits = self.Logits(theta, inputs)
    per_example_xent, per_example_argmax = self.XentLossFromLogits(
        theta, logits, class_weights, *args, **kwargs)
    return py_utils.NestedMap(
        per_example_xent=per_example_xent,
        per_example_argmax=per_example_argmax)


class SharedSoftmaxLayer(base_layer.BaseLayer):
  """Shared softmax layer for decoder embedding/softmax matrix.

  This implements weight tying, where the softmax weights are the transpose of
  the embedding matrix.
  """

  @classmethod
  def Params(cls):
    """Params for SharedSoftmaxLayer."""
    p = super().Params()
    p.Define('softmax', SimpleFullSoftmax.Params(), 'Softmax params.')
    p.Define('input_dim', 0,
             'Dimension of the input. Overrides softmax.input_dim.')
    p.Define('num_classes', 0,
             'Total number of target classes. Overrides softmax.num_classes.')
    p.Define(
        'chunk_size', 0,
        'If non-zero, computes the per example xent by small chunks along '
        'the batch dimension. Overrides softmax.num_classes.')
    # Embedding params.
    p.Define(
        'scale_sqrt_depth', False, 'If set True, activations are scaled'
        ' with sqrt(input_dim) in EmbLookup.')
    p.Define(
        'embedding_dim', 0, 'Set to be compatible with embedding layer, '
        ' and it is equivalent to input_dim')
    p.Define(
        'vocab_size', 0, 'Set to be compatible with embedding layer, and '
        'it is equivalent to num_classes')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    softmax_params = p.softmax.Copy().Set(name=p.name)
    if p.input_dim:
      softmax_params.input_dim = p.input_dim
    if p.num_classes:
      softmax_params.num_classes = p.num_classes
    if p.chunk_size:
      softmax_params.chunk_size = p.chunk_size
    if not p.vocab_size:
      p.vocab_size = softmax_params.num_classes
    if p.vocab_size != softmax_params.num_classes:
      raise ValueError('SharedSoftmaxLayer vocab_size must equal num_classes.')
    if p.scale_sqrt_depth and softmax_params.input_dim == 0:
      tf.logging.warning(
          'Input_dim is not set for scaled embedding! Outputs will be 0s!')
    self.CreateChild('softmax', softmax_params)

  def _CreateChildrenVariables(self):
    # Backwards compatibility: 'softmax' should be created outside of
    # tf.variable_scope(p.name).
    self.softmax.InstantiateVariables()
    super()._CreateChildrenVariables()

  def Logits(self, theta, *args, **kwargs):
    return self.softmax.Logits(theta.softmax, *args, **kwargs)

  def XentLossFromLogits(self, theta, *args, **kwargs):
    return self.softmax.XentLossFromLogits(theta.softmax, *args, **kwargs)

  def FProp(self, theta, *args, **kwargs):
    return self.softmax.FProp(theta.softmax, *args, **kwargs)

  def EmbLookup(self, theta, ids):
    p = self.params
    ids = py_utils.with_dependencies([
        py_utils.assert_between(
            ids,
            0,
            p.vocab_size,
            summarize=100000,
            message='{}:class_id_validation'.format(p.cls))
    ], ids)

    wm = self.softmax.DenseWeights(theta.softmax).wm
    if not self.softmax.wm_transposed:
      wm = tf.transpose(wm)
    embs_result = tf.gather(wm, ids)

    if p.scale_sqrt_depth:
      assert self.softmax.params.input_dim > 0
      embs_result *= self.softmax.params.input_dim**0.5

    return embs_result


class SingleShardFullSoftmax(SoftmaxLayer):
  """Full softmax layer."""

  def __init__(self, params):
    """Constructs a SingleShardFullSoftmax layer."""
    super().__init__(params)
    p = self.params
    assert p.name
    if p.device_mesh is not None:
      assert p.weight_split_dims_mapping is not None
      assert len(p.weight_split_dims_mapping) == 2
    linear_p = builder_layers.LinearLayer.Params().Set(
        name='linear',
        input_dims=p.input_dim,
        output_dims=p.num_classes,
        device_mesh=p.device_mesh,
        weight_split_dims_mapping=p.weight_split_dims_mapping)
    self.CreateChild('linear', linear_p)
    if p.device_mesh is not None:
      bias_split_dims_mapping = [p.weight_split_dims_mapping[1]]
    else:
      bias_split_dims_mapping = None
    bias_p = builder_layers.BiasLayer.Params().Set(
        name='bias',
        dims=p.num_classes,
        device_mesh=p.device_mesh,
        weight_split_dims_mapping=bias_split_dims_mapping)
    self.CreateChild('bias', bias_p)

  def DenseWeights(self, theta):
    return py_utils.NestedMap(wm=theta.linear.w, bias=theta.bias.b)

  def Logits(self, theta, inputs):
    """Returns the logits computed before the softmax.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A single tensor with shape [..., input_dim].

    Returns:
      logits [..., num_classes]
    """
    p = self.params
    if isinstance(inputs, (list, tuple)):
      assert len(inputs) == 1
      inputs = inputs[0]
    after_proj = self.linear.FProp(theta.linear, inputs)
    logits = self.bias.FProp(theta.bias, after_proj)
    # Clip logits by range.
    # Note that this is generally not used in conjunction with quantization and
    # shouldn't be needed at inference time as the quantized matmul above will
    # take care of clipping naturally based on the data type and qparams.
    abs_max = p.logits_abs_max
    if abs_max is not None and not p.is_inference:
      abs_min = -abs_max  # pylint: disable=invalid-unary-operand-type
      logits = py_utils.clip_by_value(logits, abs_min, abs_max)
    return logits

  def XentLossFromLogits(self,
                         theta,
                         logits,
                         class_ids=None,
                         class_probabilities=None):
    """Computes cross-entropy, argmax etc. from logits."""
    assert logits is not None
    if class_probabilities is not None:
      per_example_xent = tf.nn.softmax_cross_entropy_with_logits(
          labels=class_probabilities, logits=logits)
      per_example_argmax = tf.stop_gradient(py_utils.ArgMax(logits))
    else:
      assert class_ids is not None
      fpdtype = logits.dtype
      if fpdtype == tf.bfloat16:
        # This is needed in order to workaround the limitation that
        # tf.nn.sparse_softmax_cross_entropy_with_logits is not implemented for
        # bf16 on cpu.
        logits = tf.cast(logits, tf.float32)
      per_example_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=class_ids, logits=logits)
      if fpdtype == tf.bfloat16:
        per_example_xent = tf.cast(per_example_xent, fpdtype)

      per_example_argmax = tf.stop_gradient(py_utils.ArgMax(logits))
    return per_example_xent, per_example_argmax

  def XentLossByChunk(self, theta, activation, class_ids, class_probabilities):
    """Computes per-example xent loss."""
    p = self.params

    act_orig_shape = tf.shape(activation)
    batch_size = act_orig_shape[0]
    chunk_size = p.chunk_size
    num_chunks = batch_size // chunk_size

    num_chunks = py_utils.with_dependencies([
        py_utils.assert_equal(
            0,
            tf.math.floormod(batch_size, chunk_size),
            summarize=2,
            message='assert_equal')
    ], num_chunks)

    def ReshapeX(x):
      if x is None:
        return None
      x_shape = tf.shape(x)
      new_shape = tf.concat([[num_chunks, chunk_size], x_shape[1:]], 0)
      return tf.reshape(x, new_shape)

    activation = ReshapeX(activation)
    class_ids = ReshapeX(class_ids)
    class_probabilities = ReshapeX(class_probabilities)

    # For each chunk, we compute logits of activation[i, :, :],
    # and its xent loss with class_ids[i, :].
    def ChunkFn(theta, state0, inputs):
      del state0
      activation = inputs.activation
      class_ids = inputs.get('class_ids', None)
      class_probabilities = inputs.get('class_probabilities', None)
      logits = self.Logits(theta, activation)
      per_example_xent, per_example_argmax = self.XentLossFromLogits(
          theta, logits, class_ids, class_probabilities)
      return py_utils.NestedMap(
          xent=per_example_xent, amax=per_example_argmax), py_utils.NestedMap()

    inputs_nmap = py_utils.NestedMap(activation=activation)
    if class_ids is not None:
      inputs_nmap.class_ids = class_ids
    if class_probabilities is not None:
      inputs_nmap.class_probabilities = class_probabilities

    xent_state0 = tf.zeros(tf.shape(activation)[1:-1], dtype=p.dtype)
    argmax_out_dtype = tf.int32 if py_utils.use_tpu() else tf.int64
    amax_state0 = tf.zeros(tf.shape(activation)[1:-1], dtype=argmax_out_dtype)

    acc, _ = recurrent.Recurrent(
        theta=theta,
        state0=py_utils.NestedMap(xent=xent_state0, amax=amax_state0),
        inputs=inputs_nmap,
        cell_fn=ChunkFn)

    # acc.xent has the shape [dim0, dim1]. acc.xent[i, :] are
    # per-example xent loss for examples in the i-th chunk.  We
    # reshape acc.xent to a vector and slice the first 'batch' values.
    def GetBatch(x):
      return tf.reshape(x, act_orig_shape[:-1])

    return GetBatch(acc.xent), GetBatch(acc.amax)

  def FProp(self,
            theta,
            inputs,
            class_weights,
            class_ids=None,
            class_probabilities=None):
    """Computes logits, cross entropy etc.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: a single tensor with shape [..., input_dim].
      class_weights: a tensor with shape [..., 1] containing the weights for
        each target word.
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
    if isinstance(inputs, (list, tuple)):
      assert len(inputs) == 1
      inputs = inputs[0]

    inputs_shape = tf.shape(inputs)
    ids_shape = tf.concat([inputs_shape[:-1], [1]], 0)
    probs_shape = tf.concat([inputs_shape[:-1], [p.num_classes]], 0)

    class_weights = py_utils.HasShape(class_weights, ids_shape)
    class_weights = tf.squeeze(class_weights, -1)
    if class_ids is not None:
      class_ids = py_utils.HasShape(class_ids, ids_shape)
      class_ids = tf.squeeze(class_ids, -1)
    if class_probabilities is not None:
      class_probabilities = py_utils.HasShape(class_probabilities, probs_shape)

    if (not self.do_eval) and (p.chunk_size > 0):
      # Chunking.
      logits = None
      log_probs = None
      per_example_xent, per_example_argmax = self.XentLossByChunk(
          theta, inputs, class_ids, class_probabilities)
    else:
      logits = self.Logits(theta, inputs)
      log_probs = tf.nn.log_softmax(logits)
      per_example_xent, per_example_argmax = self.XentLossFromLogits(
          theta, logits, class_ids, class_probabilities)

    label_weights = tf.cast(class_weights, py_utils.FPropDtype(p))
    total_xent = tf.reduce_sum(per_example_xent * label_weights)
    total_weights = tf.reduce_sum(label_weights)
    output_nmap = py_utils.NestedMap(
        per_example_argmax=per_example_argmax,
        per_example_xent=per_example_xent,
        per_example_weight=label_weights,
        total_xent=total_xent,
        total_weight=total_weights,
        avg_xent=total_xent / (total_weights + 1e-6))
    if logits is not None:
      output_nmap.logits = logits
      output_nmap.log_probs = log_probs
    return output_nmap


class SingleShardSharedEmbeddingSoftmax(SingleShardFullSoftmax):
  """A shared softmax/embedding layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('vocab_size', 0, 'Num tokens in vocab.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define(
        'scale_sqrt_depth', False, 'If set True, activations are scaled'
        ' with sqrt(embedding_dim) in EmbLookup.')
    p.Define('emb_with_matmul', False, 'use one-hot vector to perform matmul.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.vocab_size == p.num_classes
    assert p.embedding_dim == p.input_dim

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
    ids = py_utils.with_dependencies([
        py_utils.assert_between(
            ids, 0, p.vocab_size, name='vocab_id_validation')
    ], ids)

    if p.emb_with_matmul:
      # [b, t, vocab_size]
      one_hot = tf.one_hot(ids, p.vocab_size, dtype=theta.linear.w.dtype)
      if one_hot.shape.is_fully_defined() and len(one_hot.shape.as_list()) == 3:
        embs = tf.einsum('blv,kv->blk', one_hot, theta.linear.w)
      else:
        embs = tf.einsum('kv,...v->...k', theta.linear.w, one_hot)
    else:
      # TODO(yonghui): Get rid of this extra copy (tf.transpose).
      emb_vars = tf.transpose(theta.linear.w)
      embs = tf.nn.embedding_lookup(emb_vars, tf.reshape(ids, [-1]))

    if p.scale_sqrt_depth:
      embs *= p.embedding_dim**0.5
    embs = py_utils.AddVN(p, embs)
    out_shape = tf.concat([tf.shape(ids), [p.embedding_dim]], 0)
    return tf.reshape(embs, out_shape)


class ConvSoftmax(quant_utils.QuantizableLayer):
  """A softmax implementation based on 1x1 convolution.

  On TPU this is much more memory efficient than MatMul after reshaping logits
  to a matrix.
  """

  @classmethod
  def Params(cls):
    """Params for SoftmaxLayer."""
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the input.')
    p.Define('hidden_dim', 0, 'Dimension of the hidden layer.')
    p.Define('num_classes', 0, 'Total number of target classes.')
    return p

  def _CreateLayerVariables(self):
    """Constructs a SimpleFullSoftmax layer."""
    super()._CreateLayerVariables()
    p = self.params
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


class DropoutLayer(base_layer.BaseLayer):
  """Apply dropout during trainig."""

  @classmethod
  def Params(cls):
    p = super().Params()
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
        rate=1 - self.params.keep_prob,
        noise_shape=noise_shape,
        seed=self.params.random_seed)

  @classmethod
  def NumOutputNodes(cls, p):
    # The layer does element-wise processing thus is input-shape agnostic.
    return

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
    if not self.do_eval or p.dropout_at_eval:
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
        seeds=py_utils.GenerateStepSeedPair(self.params),
        noise_shape=noise_shape)


class LayerNorm(base_layer.BaseLayer):
  """Layer normalization.

  Implements layer normalization:
  https://arxiv.org/abs/1607.06450
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Depth of the input to the network.')
    p.Define('epsilon', 1e-6, 'Tiny value to guard rsqrt.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm.')
    p.Define(
        'direct_scale', False, 'Whether to apply scale directly '
        'without a +1.0.  Var is initialized to 1.0 instead. This makes '
        'the layer weight-compatible with the implementation in '
        'contrib.layers.')
    p.Define('bias', True, 'Whether to use bias.')
    p.Define('center', True,
             'Whether to subtract the mean when computing variance.')
    p.Define('use_defun', True, 'Whether to use CallDefun for normalization.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim > 0, p.input_dim

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    if p.bias:
      pc = py_utils.WeightParams(
          shape=[p.input_dim],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'] +
          [py_utils.SKIP_LP_REGULARIZATION])
      self.CreateVariable('bias', pc)

    if p.direct_scale:
      scale_pc = py_utils.WeightParams(
          shape=[p.input_dim],
          init=py_utils.WeightInit.Constant(1.0),
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'] +
          [py_utils.SKIP_LP_REGULARIZATION])
    else:
      scale_pc = pc
    self.CreateVariable('scale', scale_pc)

  def _GetScaleAndBias(self, theta):
    if self.params.bias:
      bias = theta.bias
    else:
      bias = tf.zeros_like(theta.scale)
    return theta.scale, bias

  def FProp(self, theta, inputs):
    """Applies normalization over the last dimension (layer).

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A tensor of shape [..., hidden_dim].

    Returns:
      tensor of the same shape with inputs
    """
    if py_utils.testonly_skip_norm_layers():
      return inputs

    p = self.params
    with tf.name_scope(p.name):
      inputs = py_utils.with_dependencies(
          [py_utils.assert_equal(tf.shape(inputs)[-1], p.input_dim)], inputs)
      inputs = self._CastToFPropDtype(inputs)

      cur_scale, cur_bias = self._GetScaleAndBias(theta)

      if p.direct_scale:
        scale = cur_scale
      else:
        scale = 1.0 + cur_scale

      if p.use_fused_layernorm:
        if not p.center:
          raise ValueError('use_fused_layernorm does not support center=false.')
        counts, means_ss, variance_ss, _, = tf.nn.sufficient_statistics(
            inputs, axes=[-1], keepdims=True)
        mean, variance = tf.nn.normalize_moments(counts, means_ss, variance_ss,
                                                 None)
        # Adding a cast here. Sometimes, inputs/mean/variance/p.epsilon are in
        # float32 while scale and cur_bias are in bf16.
        inputs_norm = tf.cast(
            (inputs - mean) * tf.math.rsqrt(variance + p.epsilon),
            dtype=scale.dtype)
        return inputs_norm * scale + cur_bias

      def Normalize(xs):
        """Normalize `xs.x` w/ `xs.scale` and `xs.bias` gain/shift."""
        x_shape = py_utils.GetShape(xs.x)
        x_reshaped = tf.reshape(xs.x, [-1, x_shape[-1]])
        mean = tf.reduce_mean(x_reshaped, axis=[1], keepdims=True)
        if p.center:
          x_in = x_reshaped - mean
        else:
          x_in = x_reshaped
        if x_in.dtype == tf.bfloat16:
          # tf.rsqrt and SquaredDifference are not implemented for bfloat16,
          # hence we always cast into tf.float32.
          x_cast = tf.cast(x_in, tf.float32)
        else:
          x_cast = x_in
        variance = tf.reduce_mean(tf.square(x_cast), axis=[1], keepdims=True)
        x_norm_den_inv = tf.cast(
            tf.math.rsqrt(variance + p.epsilon), x_in.dtype)
        x_norm = x_in * x_norm_den_inv
        x_norm = tf.reshape(x_norm, x_shape)
        return x_norm * xs.scale + xs.bias

    if p.use_defun:
      return py_utils.CallDefun(
          Normalize, py_utils.NestedMap(x=inputs, scale=scale, bias=cur_bias))
    return Normalize(py_utils.NestedMap(x=inputs, scale=scale, bias=cur_bias))

  @classmethod
  def NumOutputNodes(cls, p):
    return p.input_dim

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(
        flops=inputs.num_elements() * 10, out_shapes=(inputs,))


# TODO(shibow/wangtao) remove this after b/174094694 is done.
class ReshapedLayerNorm(LayerNorm):
  """Customized LayerNorm with model dim D reshaped as Md."""

  def FProp(self, theta, inputs):
    """Applies normalization over the last two dimensions.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A tensor of shape [..., dim_reshape_segments, hidden_dim //
        dim_reshape_segments].

    Returns:
      tensor of the same shape with inputs.
    """
    p = self.params
    with tf.name_scope(p.name):
      inputs = self._CastToFPropDtype(inputs)

      cur_scale, cur_bias = self._GetScaleAndBias(theta)

      if p.direct_scale:
        scale = cur_scale
      else:
        scale = 1.0 + cur_scale

      axes = list(range(len(inputs.shape) - 2, len(inputs.shape)))
      counts, means_ss, variance_ss, _, = tf.nn.sufficient_statistics(
          inputs, axes=axes, keepdims=True)
      mean, variance = tf.nn.normalize_moments(counts, means_ss, variance_ss,
                                               None)
      scale = tf.reshape(scale, tf.shape(inputs)[-2:])
      cur_bias = tf.reshape(cur_bias, tf.shape(inputs)[-2:])
      # Adding a cast here. Sometimes, inputs/mean/variance/p.epsilon are in
      # float32 while scale and cur_bias are in bf16.
      inputs_norm = tf.cast(
          (inputs - mean) * tf.math.rsqrt(variance + p.epsilon),
          dtype=scale.dtype)
      return inputs_norm * scale + cur_bias


class CategoricalLayerNorm(LayerNorm):
  """Categorical layer normalization.

  Allow dynamic switch of normalization params based on given class_index.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_classes', 1,
             'Number of privatized copies of layer norm params.')
    return p

  def _BiasVarName(self, i):
    return 'bias_' + str(i)

  def _ScaleVarName(self, i):
    return 'scale_' + str(i)

  def _CreateLayerVariables(self):
    # Skip LayerNorm's _CreateLayerVariables() as bias and scale variables will
    # be created in this function.
    super(LayerNorm, self)._CreateLayerVariables()  # pylint: disable=bad-super-call
    p = self.params
    pc = py_utils.WeightParams(
        shape=[self.params.input_dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'] +
        [py_utils.SKIP_LP_REGULARIZATION])
    for i in range(p.num_classes):
      self.CreateVariable(self._BiasVarName(i), pc)
      self.CreateVariable(self._ScaleVarName(i), pc)

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert isinstance(p.num_classes, int)
    assert p.num_classes > 0
    self.AddExtraTheta('class_index', tf.constant(0, dtype=tf.int32))

  def _GetScaleAndBias(self, theta):
    p = self.params
    with tf.control_dependencies(
        [py_utils.assert_between(theta.class_index, 0, p.num_classes)]):
      biases = [theta[self._BiasVarName(i)] for i in range(p.num_classes)]
      cur_bias = tf.gather(biases, theta.class_index)
      scales = [theta[self._ScaleVarName(i)] for i in range(p.num_classes)]
      cur_scale = tf.gather(scales, theta.class_index)
      return cur_scale, cur_bias


class ConvSetLayer(quant_utils.QuantizableLayer):
  """Set of Convolutions with different filter sizes in a single layer.

    Applies a set of convolutions with different filter shapes to the inputs and
    returns the concatenated outputs.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('cnn_tpl',
             ConvLayer.Params().Set(filter_stride=(1, 1)),
             'Conv layer template for the set of conv layers.')
    p.Define(
        'filter_shapes', [(0, 0, 0, 0)],
        'Must be a list of sequences of 4. Elements are in order of height'
        ' (time), width (frequency), in_channel, out_channel')
    return p

  def __init__(self, params):
    super().__init__(params)
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

    params_conv_set = []
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

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    # The same QTensor is used for all inputs to the concat.
    self.TrackQTensor('activation')

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
    p = super().Params()
    p.Define('num_classes', 0, 'Number of classes')
    p.Define(
        'offsets', [], 'Offset (over time) for smoothing. At time T the '
        'smoothed target is class[T] + sum_i weights[i]*class[T+offset[i]]')
    p.Define('weights', [], 'Weight of the smoothing at corresponding offset')
    return p

  def __init__(self, params):
    super().__init__(params)
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
    p = super().Params()
    p.Define('num_classes', 0, 'Number of classes')
    p.Define('uncertainty', 0.1, 'Uncertainty of correct label, eps.')
    p.Define(
        'uncertainty_larger', 0.1,
        'Apply a larger uncertainty to specific tokens, as specified '
        'by token_from_target_ids.')
    p.Define('token_id_uncertainty_larger', None, 'Id of token from target_ids '
             'to apply uncertainty_larger to.')
    return p

  def __init__(self, params):
    super().__init__(params)
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

    low_confidence = p.uncertainty / tf.cast(p.num_classes - 1, tf.float32)
    high_confidence = (1.0 - p.uncertainty)

    smooth_targets = tf.one_hot(
        tf.cast(target_labels, tf.int32),
        depth=p.num_classes,
        on_value=high_confidence,
        off_value=low_confidence)
    if p.token_id_uncertainty_larger is not None:
      assert target_ids is not None
      low_confidence_larger = p.uncertainty_larger / tf.cast(
          p.num_classes - 1, tf.float32)
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
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the input to the network.')
    p.Define(
        'batch_norm', False,
        'Whether or not to apply BN to the highway skip layer output. '
        'Note this is only a single bool.')
    p.Define('carry_bias_init', 1.0, 'carry gates bias initialization')
    p.Define('couple_carry_transform_gates', False,
             'Boolean on whether to couple the transform and carry gates.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
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


class GatingLayer(base_layer.BaseLayer):
  """A gating layer.

  This class represents a gating layer, which takes 2 inputs of the same shape
  and gates them.

  The output is: carry * x + (1 - carry) * y where, carry is given by
  sigmoid(x @ w_1 + y @ w_2 + bias).

  This is different from the HighwaySkipLayer above in that carry is also a
  function of y (named transformed_x in HighwaySkipLayer).
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the input to the network.')
    p.Define('has_bias', False, 'Whether carry has a bias term.')
    p.Define('carry_bias_init', 0.0, 'carry gates bias initialization')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    carry_gate_params = ProjectionLayer.Params().Set(
        batch_norm=False,
        has_bias=p.has_bias,
        activation='SIGMOID',
        input_dim=p.input_dim * 2,
        output_dim=p.input_dim,
        bias_init=p.carry_bias_init,
        name='carry')
    self.CreateChild('carry_gate', carry_gate_params)

  def FProp(self, theta, x, y, paddings=None):
    """Fprop for the gating layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      x: An input feature, the last dimension must match p.input_dim.
      y: Another input feature. Must have the same shape as 'x'.
      paddings: padding applied to the features. When x and y have shape [...,
        input_dim], 'paddings', when specified, must have shaped [..., 1], where
        all but the last dimension match.

    Returns:
      layer_out - activations after forward propagation. Same shape as x and y.
    """
    y = py_utils.with_dependencies(
        [py_utils.assert_shape_match(tf.shape(x), tf.shape(y))], y)
    carry = self.carry_gate.FProp(theta.carry_gate, tf.concat([x, y], axis=-1),
                                  paddings)
    layer_out = x * carry + y * (1 - carry)
    return layer_out


class GradNormTracker(base_layer.BaseLayer):
  """A helper class to keep track of gradient norm stats."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('decay', 0.995,
             'Decay in updating the moving avgs in grad norm stats')
    p.Define('grad_norm_lower_cap', 1e-2, 'The minimal gradient norm value.')
    p.Define(
        'clip_threshold', 4.0,
        'Distance threshold at which gradients are clipped to 0.0.'
        ' Distance is measured in the number of standard deviations a'
        ' given gradient norm is from the mean gradient norm. The'
        ' default value of 4.0 means we are throwing away roughly'
        ' 0.15% of steps.')
    p.Define(
        'grad_norm_clip_cap_min', 0.0,
        'We stop clipping if grad norm is already smaller than this'
        ' value.')
    p.Define(
        'dry_run', False, 'If True, always return 1.0 in FProp() to signify '
        'no grad clipping suggested, in which case the class only collects '
        'stats and summaries.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self._decay = params.decay

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()

    pc = py_utils.WeightParams(
        shape=[],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=tf.float32,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('log_mean', pc, trainable=False)
    self.CreateVariable('log_mean_squared', pc, trainable=False)
    self.CreateVariable('total_weight', pc, trainable=False)
    self.CreateVariable('total_rejections', pc, trainable=False)

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

      # Exponentially decayed moving avg of log(grad_norm) mean.
      mean = theta.log_mean / tf.maximum(theta.total_weight, 1e-6)
      # Exponentially decayed moving avg of log(grad_norm) variance.
      var = ((theta.log_mean_squared / tf.maximum(theta.total_weight, 1e-6)) -
             mean * mean)
      std = tf.sqrt(tf.maximum(var, 1e-6))

      summary_utils.scalar('log_grad_norm_mean', mean)
      summary_utils.scalar('log_grad_norm_std', std)
      summary_utils.scalar('clip_ratio_threshold',
                           tf.exp(std * p.clip_threshold))
      summary_utils.scalar('clip_threshold',
                           tf.exp(mean + std * p.clip_threshold) - 1.0)
      summary_utils.scalar('total_rejections', theta.total_rejections)

      log_grad_norm = tf.math.log(grad_norm + 1.0)
      log_grad_norm_cap = tf.cast(mean + std * p.clip_threshold, tf.float32)
      log_grad_norm_cap_min = tf.math.log(p.grad_norm_clip_cap_min + 1.0)
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
      trigger = tf.math.logical_and(log_grad_norm > log_grad_norm_cap,
                                    theta.total_weight > 0.75)
      if has_nan is not None:
        trigger = tf.math.logical_or(trigger, has_nan)

      log_grad_norm_capped = tf.minimum(log_grad_norm, log_grad_norm_cap)

      update_moving_avg = tf.group(
          UpdateExpMovingAvg(self.vars.log_mean, log_grad_norm_capped, has_nan),
          UpdateExpMovingAvg(self.vars.log_mean_squared,
                             log_grad_norm_capped * log_grad_norm_capped,
                             has_nan),
          UpdateExpMovingAvg(self.vars.total_weight, tf.constant(1.0), has_nan),
          tf.assign_add(self.vars.total_rejections,
                        tf.cast(trigger, tf.float32)))

      return py_utils.with_dependencies([update_moving_avg],
                                        1.0 if p.dry_run else 1.0 -
                                        tf.cast(trigger, tf.float32))


class WeightedSumLayer(base_layer.BaseLayer):
  """Returns the weighted sum of a list of input tensors."""

  @classmethod
  def Params(cls):
    """Params for this MergerLayer class."""
    p = super().Params()
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if not p.name:
      raise ValueError('Layer must have a specified name!')

    assert p.num_sources > 0, ('Must specify num_sources > 0.')

    if p.weighted_merger_dropout_prob > 0.0:
      dropout_tpl = DropoutLayer.Params()
      dropout_tpl.keep_prob = (1.0 - p.weighted_merger_dropout_prob)
      self.CreateChild('weighted_merger_dropout', dropout_tpl)
    else:
      self.CreateChild('weighted_merger_dropout', IdentityLayer.Params())

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    params_init = py_utils.WeightInit.Constant(0.0)
    # Weights to be learned.
    pw = py_utils.WeightParams(
        shape=[p.num_sources],
        init=params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('sum_weight', pw)

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
    p = super().Params()
    p.Define('num_nodes', 0, 'Number of nodes in each input vector.')
    p.Define('num_inputs', 0, 'Number of input vectors to combine.')
    return p

  def __init__(self, params):
    """Initializes GatedAverageLayer."""
    super().__init__(params)
    p = self.params

    assert p.num_nodes > 0, 'Number of dimensions should be greater than 0.'
    assert p.num_inputs > 0, 'Number of inputs should be greater than 0.'

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    in_size = p.num_inputs * p.num_nodes

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
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the input and output.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim > 0

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    pc = py_utils.WeightParams(
        shape=[p.input_dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=self._VariableCollections())
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
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the input to the adapter.')
    p.Define('bottleneck_dim', 0, 'Dimension of the feedforward inner layer.')
    p.Define('ln_tpl', LayerNorm.Params(), 'Layer norm default params.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name

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


def Conv2DFlops(inputs, filter_shape, stride, padding):
  """Returns number of float operations (mult/adds) for a Conv2D op.

  Args:
    inputs: the input shape. Must have four elements.
    filter_shape: the convolution filter shape. Must have four elements.
    stride: the strides along height and width, respectively.
    padding: 'SAME' or 'VALID'.

  Returns:
    Number of multiplications and additions.
  """
  b, h, w = inputs[0], inputs[1], inputs[2]
  fh, fw, ic, oc = filter_shape
  sh, sw = stride

  def _CeilDiv(x, y):
    return tf.math.floordiv(x + y - 1, y)

  if padding == 'SAME':
    oh = _CeilDiv(h, sh)
    ow = _CeilDiv(w, sw)
  else:
    assert padding == 'VALID'
    oh = _CeilDiv(h - fh + 1, sh)
    ow = _CeilDiv(w - fw + 1, sw)
  # Mul/add counts as 2 flops.
  return (tf.cast(b * oh * ow, tf.int64) *
          tf.cast(fh * fw * ic * oc, tf.int64) * 2)


class Conv2DLayerNoPadding(base_layer.BaseLayer):
  """2-D Convolution layer w/o padding.

  TODO(laurenzo): Dedup in favor of SeparableConv2DLayer where possible.
  """

  @classmethod
  def Params(cls):
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.padding in ['SAME', 'VALID']
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert len(p.dilations) == 2
    assert all(x > 0 for x in p.filter_stride)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
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
      computation_cost.Add(
          self, 'flops',
          Conv2DFlops(
              tf.shape(x),
              filter_shape=symbolic.EvalExpr(symbolic.TENSOR_VALUES,
                                             p.filter_shape),
              stride=p.filter_stride,
              padding=p.padding))
      return tf.nn.conv2d(
          input=x,
          filters=theta.w,
          strides=[1, p.filter_stride[0], p.filter_stride[1], 1],
          padding=p.padding,
          dilations=[1, p.dilations[0], p.dilations[1], 1],
          data_format='NHWC')

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    b, h, w, c = inputs
    fh, fw, ic, oc = p.filter_shape
    assert ic == c
    sh, sw = p.filter_stride
    if p.padding == 'SAME':
      oh = sympy.ceiling(h / sh)
      ow = sympy.ceiling(w / sw)
    else:
      oh = sympy.ceiling((h - fh + 1) / sh)
      ow = sympy.ceiling((w - fw + 1) / sw)
    flops = b * oh * ow * fh * fw * ic * oc * 2  # mul/add counts as 2 flop.
    outputs = tshape.Shape([b, oh, ow, oc])
    return py_utils.NestedMap(flops=flops, out_shapes=(outputs,))


class FetchLayer(base_layer.BaseLayer):
  """A layer facilitating fetching activations and their gradients."""

  def __init__(self, params):
    super().__init__(params)
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

      def FetchBak(xs, ys, dys, index=i):
        del xs, ys
        self._gradients[index] = dys
        return dys

      def FetchFwd(x):
        return x

      self._activations[i] = py_utils.CallDefun(FetchFwd, v, bak=FetchBak)

    return tuple(self._activations) if num > 1 else self._activations[0]


class GluLayer(base_layer.BaseLayer):
  """Gated Linear Unit.

  See https://arxiv.org/abs/1612.08083 for more details.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Dimension of the layer input.')
    p.Define('output_dim', 0, 'Dimension of the layer output.')
    p.Define('ln_tpl', LayerNorm.Params(), 'Layer norm default params.')
    p.Define('dense_tpl', FCLayer.Params().Set(), 'Fully connected layer.')
    p.Define(
        'activation', 'NONE',
        'Non-linearity applied after the dense layer in the value branch.')
    p.Define('gate_activation', 'SIGMOID',
             'Non-linearity applied for the gating.')
    p.Define('dropout_tpl', DropoutLayer.Params(), 'Dropout applied to output.')
    p.Define('apply_residual', True, 'Whether or not to add inputs to outputs.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim

    if p.output_dim:
      output_dim = p.output_dim
    else:
      output_dim = p.input_dim

    if p.apply_residual:
      assert output_dim == p.input_dim

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
    params.activation = p.gate_activation
    params.output_dim = output_dim
    self.CreateChild('gate_layer', params)

    # Initialize layer norm.
    if p.ln_tpl:
      params = p.ln_tpl.Copy()
      params.name = 'layer_norm'
      params.input_dim = p.input_dim
      self.CreateChild('layer_norm', params)

    # Initialize dropout.
    dropout_tpl = p.dropout_tpl.Copy()
    self.CreateChild('dropout', dropout_tpl)

  def FProp(self, theta, inputs, paddings):
    if 'layer_norm' in self.children:
      inputs_normalized = self.layer_norm.FProp(theta.layer_norm, inputs)
    else:
      inputs_normalized = inputs
    if (paddings.shape.ndims is None or
        paddings.shape.ndims != inputs_normalized.shape.ndims):
      paddings = tf.expand_dims(paddings, -1)
    values = self.value_layer.FProp(theta.value_layer, inputs_normalized,
                                    paddings)
    gates = self.gate_layer.FProp(theta.gate_layer, inputs_normalized, paddings)
    glu_output = values * gates
    glu_output = self.dropout.FProp(theta.dropout, glu_output)
    if self.params.apply_residual:
      return inputs + glu_output
    return glu_output


class MultitaskAdapterBaseLayer(base_layer.BaseLayer):
  """Residual adapter layer for multilingual models.

  Residual adapters can be used to fine-tune a single model to multiple
  domains, tasks, or languages: https://arxiv.org/pdf/1902.00751.pdf

  Each adapter consists of a "down" projection to a smaller dimension followed
  by an "up" projection, the result of which is added back to the input
  activation.  The projection weights and biases are task-specific.

  Whereas ResidualAdapterLayer learns and applies the parameters for a single
  task, this layer learns and applies the parameters for multiple tasks so that
  we have a single model serving the different tasks. The parameters can be
  trained for all tasks at the same time, or in one-off per-task training jobs.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_tasks', 0, 'Number of tasks.')
    p.Define('input_dim', 0, 'Dimension of the input to the adapter.')
    p.Define('bottleneck_dim', 0, 'Dimension of the bottleneck.')
    p.Define('layer_norm_tpl', LayerNorm.Params(), 'Layer norm default params.')
    p.Define(
        'data_format', 'TBC', 'String(enum) specifying the input and output '
        'data format for this layer. Supported formats: '
        '"TBC": [time, batch, input_dim] and "BTC": [batch, time, input_dim].')
    p.Define('clip_task_ids', False,
             'If True, clips the given task ids to [0, p.num_tasks - 1].')
    return p


class MultitaskAdapterLayer(MultitaskAdapterBaseLayer):
  """MultitaskAdapterBaseLayer implemented with EmbeddingLayers."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'projection_params_init', None,
        'Weight initialization for up and down projections. Only used for '
        'weights, not biases.  If None, uses default weight init, which is '
        'typically Xavier with scale of 1.0.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    # Data format is either 'TBC' (time-major) or 'BTC' (batch-major).
    assert p.data_format in ('TBC', 'BTC')
    base_emb_params = EmbeddingLayer.Params().Set(
        vocab_size=p.num_tasks, max_num_shards=1)
    down_proj_w_params = base_emb_params.Copy()
    down_proj_w_params.Set(
        embedding_dim=p.input_dim * p.bottleneck_dim, name='down_proj_w')
    if p.projection_params_init:
      down_proj_w_params.params_init = p.projection_params_init
    down_proj_b_params = base_emb_params.Copy()
    down_proj_b_params.Set(embedding_dim=p.bottleneck_dim, name='down_proj_b')
    up_proj_w_params = base_emb_params.Copy()
    up_proj_w_params.Set(
        embedding_dim=p.bottleneck_dim * p.input_dim, name='up_proj_w')
    if p.projection_params_init:
      up_proj_w_params.params_init = p.projection_params_init
    up_proj_b_params = base_emb_params.Copy()
    up_proj_b_params.Set(embedding_dim=p.input_dim, name='up_proj_b')

    self.CreateChild('down_proj_w', down_proj_w_params)
    self.CreateChild('down_proj_b', down_proj_b_params)
    self.CreateChild('up_proj_w', up_proj_w_params)
    self.CreateChild('up_proj_b', up_proj_b_params)
    params = p.layer_norm_tpl.Copy()
    params.name = 'adapter_ln'
    params.input_dim = p.input_dim
    self.CreateChild('layer_norm', params)

  def FProp(self, theta, inputs, tasks):
    """Fprop for multitask adapter.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: A tensor containing the activations from the previous layer. For
        'TBC', the shape is [time, batch, input_dim] and for 'BTC', it's [batch,
        time, input_dim].
      tasks: An int32 tensor containing the task ID for each input.  If 'tasks'
        is of rank 2, we assume it to be of shape [time, batch] if 'BTC' and
        [batch, time] if 'TBC', indicating a different task for each timestep.
        In this case we look up adapter params for each timestep.  If 'tasks' is
        of rank 1, we assume it to be of shape [batch], indicating a single task
        for all timesteps of a sequence. This latter setup uses substantially
        less memory and is generally preferred.

    Returns:
      A tensor containing the adapted activations with shape
      [time, batch, input_dim] for 'TBC' and [batch, time, input_dim] for 'BTC'.
    """
    p = self.params
    inputs_shape = tf.shape(inputs)
    per_timestep_task = (tasks.shape.ndims == 2)
    batch_index = 1 if p.data_format == 'TBC' else 0
    time_index = 1 - batch_index
    inputs = py_utils.with_dependencies(
        [
            # Checks that inputs has 3 dimensions, last is hidden dim.
            py_utils.assert_shape_match(inputs_shape, [-1, -1, p.input_dim]),
            # Checks that inputs and tasks have same batch dimension.
            py_utils.assert_shape_match([inputs_shape[batch_index]], [
                tf.shape(tasks)[batch_index]
                if per_timestep_task else tf.shape(tasks)[0]
            ])
        ],
        inputs)
    if p.clip_task_ids:
      tasks = tf.clip_by_value(tasks, 0, p.num_tasks - 1)

    # To support different task for each timetstep, flatten inputs and
    # tasks.  Below, 'batch' now refers to flattened batch size, time * batch.
    if per_timestep_task:
      tasks = py_utils.with_dependencies(
          [
              # Checks that inputs and tasks have same time dimension.
              py_utils.assert_shape_match(inputs_shape[:1],
                                          tf.shape(tasks)[:1])
          ],
          tasks)
      tasks = tf.reshape(tasks, [-1])
      if p.data_format == 'TBC':
        inputs = tf.reshape(inputs, [1, -1, p.input_dim])
      else:
        inputs = tf.reshape(inputs, [-1, 1, p.input_dim])

    # Lookup all weights and biases
    # [batch] -> [batch, hidden * k] -> [batch, hidden, k]
    down_weights = tf.reshape(
        self.down_proj_w.EmbLookup(theta.down_proj_w, tasks),
        [-1, p.input_dim, p.bottleneck_dim])
    # [batch] -> [batch, k] -> [1, batch, k] if 'TBC' else [batch, 1, k]
    down_biases = tf.expand_dims(
        self.down_proj_b.EmbLookup(theta.down_proj_b, tasks), time_index)
    # [batch] -> [batch, k * hidden] -> [batch, k, hidden]
    up_weights = tf.reshape(
        self.up_proj_w.EmbLookup(theta.up_proj_w, tasks),
        [-1, p.bottleneck_dim, p.input_dim])
    # [batch] -> [batch, h] -> [1, batch, h] if 'TBC' else [batch, 1, h]
    up_biases = tf.expand_dims(
        self.up_proj_b.EmbLookup(theta.up_proj_b, tasks), time_index)

    # Layer norm -> down-projection -> non-linearity -> up-projection
    norm_inputs = self.layer_norm.FProp(theta.layer_norm, inputs)
    # If per_timestep_task, t = 1, b = time * batch.
    # Otherwise, t = time, b = batch.
    if p.data_format == 'TBC':
      down_projected = tf.einsum('tbh,bhk->tbk', norm_inputs, down_weights)
    else:
      down_projected = tf.einsum('bth,bhk->btk', norm_inputs, down_weights)
    down_projected += down_biases
    down_projected = tf.nn.relu(down_projected)
    if p.data_format == 'TBC':
      up_projected = tf.einsum('tbk,bkh->tbh', down_projected, up_weights)
    else:
      up_projected = tf.einsum('btk,bkh->bth', down_projected, up_weights)
    up_projected += up_biases
    output = inputs + up_projected

    # Unflatten output:
    #   for 'TBC': [1, time * batch, hidden] -> [time, batch, hidden]
    #   for 'BTC': [1, batch * time, hidden] -> [batch, time, hidden]
    if per_timestep_task:
      output = tf.reshape(output, inputs_shape)
    return output


class MultitaskAdapterEinsumLayer(MultitaskAdapterBaseLayer):
  """MultitaskAdapterBaseLayer implemented with Einsum.

  The embedding-based solution sometimes triggers b/175464137.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.data_format = 'BTC'
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.data_format == 'BTC'
    params = p.layer_norm_tpl.Copy()
    params.input_dim = p.input_dim
    self.CreateChild('layer_norm', params)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    down_w_pc = py_utils.WeightParams(
        shape=[p.num_tasks, p.input_dim, p.bottleneck_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('down_w', down_w_pc)
    down_b_pc = py_utils.WeightParams(
        shape=[p.num_tasks, p.bottleneck_dim],
        init=py_utils.WeightInit.Constant(0.),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('down_b', down_b_pc)
    up_w_pc = py_utils.WeightParams(
        shape=[p.num_tasks, p.bottleneck_dim, p.input_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('up_w', up_w_pc)
    up_b_pc = py_utils.WeightParams(
        shape=[p.num_tasks, p.input_dim],
        init=py_utils.WeightInit.Constant(0.),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('up_b', up_b_pc)

  def FProp(self, theta, inputs, tasks):
    """Fprop for multitask adapter.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: A tensor containing the activations from the previous layer.
        [batch, time, input_dim].
      tasks: An int32 tensor containing the task ID for each input. [batch].

    Returns:
      A tensor containing the adapted activations with the same shape as inputs.
    """
    p = self.params
    inputs = self._CastToFPropDtype(inputs)
    assert tasks.shape.ndims == 1
    if p.clip_task_ids:
      tasks = tf.clip_by_value(tasks, 0, p.num_tasks - 1)
    # [batch, num_tasks].
    tasks_onehot = tf.one_hot(tasks, p.num_tasks, axis=-1, dtype=inputs.dtype)

    # Einsum axis names:
    # b - batch
    # t - time
    # k - task
    # i - input_dim
    # n - bottleneck_dim

    # [batch, input_dim, bottleneck_dim].
    down_w = tf.einsum('bk,kin->bin', tasks_onehot, theta.down_w)
    # [batch, 1, bottleneck_dim].
    down_b = tf.einsum('bk,kn->bn', tasks_onehot, theta.down_b)[:, None, :]
    # [batch, bottleneck_dim, input_dim].
    up_w = tf.einsum('bk,kni->bni', tasks_onehot, theta.up_w)
    # [batch, 1, input_dim].
    up_b = tf.einsum('bk,ki->bi', tasks_onehot, theta.up_b)[:, None, :]

    # Layer norm -> down-projection -> non-linearity -> up-projection
    norm_inputs = self.layer_norm.FProp(theta.layer_norm, inputs)
    # [batch, time, bottleneck_dim].
    down_projected = tf.einsum('bti,bin->btn', norm_inputs, down_w) + down_b
    # ReLU.
    down_projected = tf.nn.relu(down_projected)
    # [batch, time, input_dim].
    up_projected = tf.einsum('btn,bni->bti', down_projected, up_w) + up_b
    # Residual.
    return inputs + up_projected


class CCTGatingNetwork(quant_utils.QuantizableLayer):
  """A gating network that is continous for training and discrete for eval.

  Based on the gating network from https://arxiv.org/abs/2002.07106.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Depth of the input to the network.')
    p.Define('hidden_layer_dim', 0, 'Depth of the hidden layer outputs.')
    p.Define('num_outputs', 0, 'Number of scalar gate outputs.')
    p.Define('noise_std', 1.0, 'Standard deviation for gating noise.')
    p.Define('noise_warmup_steps', 1.0, 'Steps to full noise.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    params = schedule.PolynomialSchedule.Params()
    params.start = (0, 0.0)
    params.limit = (p.noise_warmup_steps, p.noise_std)
    self.CreateChild('noise_std', params)

    params = FeedForwardNet.Params()
    params.name = 'gating_layer'
    params.input_dim = p.input_dim
    params.activation = ['RELU', 'NONE']
    params.hidden_layer_dims = [p.hidden_layer_dim, p.num_outputs]
    self.CreateChild('gatingfflayer', params)

  def FProp(self, theta, inputs, paddings=None):
    p = self.params
    p_c = self.gatingfflayer.FProp(theta.gatingfflayer, inputs, paddings)
    if self.do_eval:
      ones = tf.ones(tf.shape(p_c), py_utils.FPropDtype(p))
      zeros = tf.zeros(tf.shape(p_c), py_utils.FPropDtype(p))
      p_c = tf.where(
          tf.greater_equal(p_c, tf.constant(0.0, dtype=py_utils.FPropDtype(p))),
          ones, zeros)
    else:
      noise_std = self.noise_std.Value()
      noise = py_utils.DeterministicVN(
          p, py_utils.GenerateStepSeedPair(p), tf.shape(p_c), std=noise_std)
      p_c = tf.nn.sigmoid(p_c + noise)
    return p_c

  @classmethod
  def FPropMeta(cls, p, inputs, paddings=None):
    py_utils.CheckShapes((inputs,))
    assert inputs[-1] == p.input_dim
    flops = 0
    in_dim = inputs[-1]
    other_dims = inputs.num_elements() / in_dim
    flops = 5 * other_dims * in_dim * p.hidden_layer_dim
    flops = 5 * other_dims * p.num_outputs * p.hidden_layer_dim
    out_shape = tshape.Shape(inputs[:-1] + [symbolic.ToStatic(p.num_outputs)])
    return py_utils.NestedMap(flops=flops, out_shapes=(out_shape,))


class CondScaleShiftFFNLayer(base_layer.BaseLayer):
  """Feature Modulation layer.

  https://distill.pub/2018/feature-wise-transformations/
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 0, 'Depth of the input.')
    p.Define('output_dim', 0, 'Depth of the output.')
    p.Define('ffn', FeedForwardNet.Params(), 'Projection layer params')
    p.Define('scale_fn', 'NONE',
             'The activation function to use for scale output')
    p.Define('shift_fn', 'NONE',
             'The activation function to use for shift output')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name

    output_dim = p.output_dim * 2  # 1st split for shift, 2nd split for scale
    params_ffn = p.ffn.Copy().Set(
        input_dim=p.input_dim, name='{}_ffn'.format(p.name))
    params_fcout = FCLayer.Params().Copy().Set(
        input_dim=params_ffn.hidden_layer_dims[-1],
        output_dim=output_dim,
        activation='NONE',
        name='{}_fcout'.format(p.name))
    self.CreateChild('ffn', params_ffn)
    self.CreateChild('fcout', params_fcout)

  def FProp(self, theta, inputs, paddings=None):
    """Calculate scale shift and modify input.

    Args:
      theta: params.
      inputs: The input tensor. Shaped [..., input_dim].
      paddings: The input padding tensors.

    Returns:
      Output after calculating shift and scale (2 tensors).
      Shaped [..., output_dim].
    """
    p = self.params

    ffn_output = self.ffn.FProp(theta.ffn, inputs, paddings)
    fcout_output = self.fcout.FProp(theta.fcout, ffn_output, paddings)
    scale_output, shift_output = tf.split(
        fcout_output, num_or_size_splits=2, axis=-1)

    def OpWrapper(name, tensor):
      """Wrapper for retrieve tf operations."""
      if activations.IsSupported(name):
        op = activations.GetFn(name)
      else:
        if name == 'EXP':
          op = tf.exp
        elif name == 'NONE':
          op = tf.identity
        else:
          raise ValueError()
      return op(tensor)

    scale_output = OpWrapper(p.scale_fn, scale_output)
    shift_output = OpWrapper(p.shift_fn, shift_output)
    return scale_output, shift_output
