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
"""Common conv layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import bn_layers
from lingvo.core import py_utils


def ComputeConvOutputShape(in_shape,
                           t_stride,
                           f_stride,
                           outc=None,
                           padding='SAME'):
  """Computes output shape for convolution and pooling layers.

  If `in_shape` is a dynamic shape, the output will be Tensors, while if
  `in_shape` is a list of ints then the output will also be a list of ints.

  Args:
    in_shape: A length 4 Tensor or list representing the input shape.
    t_stride: The stride along the time dimension.
    f_stride: The stride along the frequency dimension.
    outc: The expected output channel. If None, will use the input channel.
    padding: 'SAME' or 'VALID'.

  Returns:
    The expected output shape.
  """
  # In the order of batch, time, frequency, channel
  n = in_shape[0]
  t = in_shape[1]
  f = in_shape[2]
  c = in_shape[3]
  # Last two dimensions has to be specified.
  assert f is not None and c is not None
  if padding == 'VALID':
    if t:
      t -= t_stride - 1
    f -= f_stride - 1
  ot = t
  if ot is not None:
    ot = (ot + t_stride - 1) // t_stride
  of = (f + f_stride - 1) // f_stride
  if outc is None:
    outc = c
  return [n, ot, of, outc]


def ComputeConvOutputPadding(paddings, window, stride,
                             padding_algorithm='SAME'):
  """Computes paddings for convolution and pooling output.

  out_padding[i] == 1 iff any in_padding corresponding to that output is 1.

  Args:
    paddings: The paddings tensor. It is expected to be of shape [batch, time].
    window: The size of the windows.
    stride: The time-stride between adjacent windows.
    padding_algorithm: 'SAME' or 'VALID'.

  Returns:
    out_padding, The new padding tensor of size [batch, ceil(time / stride)].
  """
  if stride == 1:
    return paddings

  # Pad so input_length divides stride.
  input_length = py_utils.GetShape(paddings)[1]
  pad_len = (input_length + stride - 1) // stride * stride - input_length
  paddings = tf.pad(paddings, [[0, 0], [0, pad_len]], constant_values=1.0)
  out_padding = tf.nn.pool(
      tf.expand_dims(paddings, -1),
      [window],
      'MAX',
      padding_algorithm,
      strides=[stride],
  )
  return tf.squeeze(out_padding, -1)


class BaseConv2DLayerWithPadding(base_layer.BaseLayer):
  """Base class for 2D convolution layers."""

  @classmethod
  def Params(cls):
    p = super(BaseConv2DLayerWithPadding, cls).Params()
    p.Define(
        'filter_shape', (0, 0, 0, 0),
        'Filter shape. Must be a sequence of length 4. Elements are in'
        ' the order of height (time), width (frequency), in_channel,'
        ' out_channel. For causal convolution, filter_shape[0]'
        ' is the actual number of trained weights in the time dimension'
        ' of the kernel.')
    p.Define(
        'filter_stride', (1, 1),
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
        'weight_norm', False,
        'If true, apply weight normalization to weights as proposed by'
        ' Salimans and Kingma, 2016: https://arxiv.org/abs/1602.07868')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseConv2DLayerWithPadding, self).__init__(params)
    p = self.params
    assert p.name
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert all(x > 0 for x in p.filter_shape)
    assert all(x > 0 for x in p.filter_stride)
    assert len(p.dilation_rate) == 2
    assert all(x > 0 for x in p.dilation_rate)
    # Dilation and stride can't be combined.
    if any(x > 1 for x in p.dilation_rate):
      assert all(x == 1 for x in p.filter_stride)

  @property
  def output_channels(self):
    """The number of output channels for this conv layer."""
    raise NotImplementedError()

  @property
  def input_channels(self):
    """The number of input channels for this conv layer."""
    return self.params.filter_shape[2]

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    p = self.params
    return ComputeConvOutputShape(in_shape, p.filter_stride[0],
                                  p.filter_stride[1], self.output_channels)

  def FProp(self, theta, inputs, paddings):
    """Apply convolution to inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch, time,
        frequency, channel]. The time dimension corresponds to the height
        dimension as in images and the frequency dimension corresponds to the
        width dimension as in images.
      paddings: The paddings tensor, expected to be of shape [batch, time].

    Returns:
      outputs, out_paddings pair.
    """
    p = self.params
    with tf.name_scope(p.name):
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(tf.shape(paddings), [-1, -1]),
          py_utils.assert_shape_match(
              tf.shape(inputs),
              tf.concat([tf.shape(paddings), [-1, self.input_channels]], 0))
      ], inputs)

      def _ApplyPadding(tensor_in, padding_in):
        padding_expanded = tf.expand_dims(tf.expand_dims(padding_in, -1), -1)
        return tensor_in * (1.0 - padding_expanded)

      # Zeroing out padded inputs.
      inputs = _ApplyPadding(inputs, paddings)

      # Evaluate the conv kernel on 'inputs'.
      out = self._EvaluateConvKernel(theta, inputs)

      # NOTE: this may be slightly inaccurate when p.dilation_rate[0] > 1.
      # But there's likely no real problems. Trying to set it gives an error:
      # pooling with SAME padding is not implemented for dilation_rate > 1.
      # NOTE: we use window=p.filter_stride[0] to be compatible with legacy
      # implementation.  Consider updating it to be the actual shape.
      conv_padding = ComputeConvOutputPadding(
          paddings, window=p.filter_stride[0], stride=p.filter_stride[0])
      # Assuming padded nodes will be properly zero-ed out if necessary by
      # sub-sequent layers.
      # out = _ApplyPadding(out, conv_padding)
      out = py_utils.HasShape(out, self.OutShape(tf.shape(inputs)))
      return out, conv_padding

  def _EvaluateConvKernel(self, theta, conv_input):
    """Evaluate the convolution kernel on input 'conv_input'."""
    raise NotImplementedError


class Conv2DLayerWithPadding(BaseConv2DLayerWithPadding):
  """Conv2D layer."""

  @base_layer.initializer
  def __init__(self, params):
    super(Conv2DLayerWithPadding, self).__init__(params)
    p = self.params
    assert p.name
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)
      if p.weight_norm:
        self.CreateVariable(
            'g',
            py_utils.WeightParams(
                shape=[p.filter_shape[-1]],
                init=py_utils.WeightInit.Constant(0.0),
                dtype=p.dtype,
                collections=[self.__class__.__name__ + '_vars']))

  @property
  def output_channels(self):
    """The number of output channels for this conv layer."""
    p = self.params
    return p.filter_shape[-1]

  def _GetWeight(self, theta):
    p = self.params
    if p.weight_norm:
      # Normalize along the last dim (standard conv).
      filter_w = tf.nn.l2_normalize(theta.w, [0, 1, 2]) * tf.reshape(
          (theta.g + 1.0), [1, 1, 1, p.filter_shape[-1]])
    else:
      filter_w = theta.w
    return filter_w

  def _EvaluateConvKernel(self, theta, inputs):
    """Apply convolution to inputs."""
    p = self.params
    filter_w = self._GetWeight(theta)
    return tf.nn.convolution(
        inputs,
        filter_w,
        strides=p.filter_stride,
        dilation_rate=p.dilation_rate,
        data_format='NHWC',
        padding='SAME')


class CausalConv2DLayerWithPadding(Conv2DLayerWithPadding):
  """2D conv layer with causal dependency on the time axis."""

  @base_layer.initializer
  def __init__(self, params):
    super(CausalConv2DLayerWithPadding, self).__init__(params)
    p = self.params
    assert p.filter_shape[1] == 1, 'Only 1d causal convolution is supported.'

  def _EvaluateConvKernel(self, theta, inputs):
    """Apply convolution to inputs."""
    p = self.params
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

    filter_w = self._GetWeight(theta)
    return tf.nn.convolution(
        inputs,
        filter_w,
        strides=p.filter_stride,
        dilation_rate=p.dilation_rate,
        data_format='NHWC',
        padding=padding_algorithm)


class DepthwiseConv2DLayer(BaseConv2DLayerWithPadding):
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

  @base_layer.initializer
  def __init__(self, params):
    super(DepthwiseConv2DLayer, self).__init__(params)
    p = self.params
    assert p.name
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)
      if p.weight_norm:
        self.CreateVariable(
            'g',
            py_utils.WeightParams(
                shape=[p.filter_shape[2], p.filter_shape[3]],
                init=py_utils.WeightInit.Constant(0.0),
                dtype=p.dtype,
                collections=[self.__class__.__name__ + '_vars']))

  @property
  def output_channels(self):
    """The number of output channels for this conv layer."""
    p = self.params
    # Depthwise convolution filter shape is:
    #   [..., in_channels, channel_multiplier].
    return p.filter_shape[2] * p.filter_shape[3]

  def _GetWeight(self, theta):
    p = self.params
    if p.weight_norm:
      # Normalize along the last two dims.
      filter_w = tf.nn.l2_normalize(theta.w, [0, 1]) * tf.reshape(
          (theta.g + 1.0), [1, 1, p.filter_shape[2], p.filter_shape[3]])
    else:
      filter_w = theta.w
    return filter_w

  def _EvaluateConvKernel(self, theta, inputs):
    """Apply convolution to inputs."""
    p = self.params
    filter_w = self._GetWeight(theta)
    return tf.nn.depthwise_conv2d(
        inputs,
        filter_w,
        strides=[1, p.filter_stride[0], p.filter_stride[1], 1],
        rate=p.dilation_rate,
        data_format='NHWC',
        padding='SAME')


class CausalDepthwiseConv2DLayer(DepthwiseConv2DLayer):
  """Depthwise conv layer with causal dependency on the time axis."""

  @base_layer.initializer
  def __init__(self, params):
    super(CausalDepthwiseConv2DLayer, self).__init__(params)
    p = self.params
    assert p.filter_shape[1] == 1, 'Only 1d causal convolution is supported.'

  def _EvaluateConvKernel(self, theta, inputs):
    """Apply convolution to inputs."""
    p = self.params
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
    filter_w = self._GetWeight(theta)
    return tf.nn.depthwise_conv2d(
        inputs,
        filter_w,
        strides=[1, p.filter_stride[0], p.filter_stride[1], 1],
        rate=p.dilation_rate,
        data_format='NHWC',
        padding=padding_algorithm)


class NormalizedDepthwiseConv2DLayer(DepthwiseConv2DLayer):
  """DepthwiseConv2DLayer where weights are normalized over the time dim.

  https://arxiv.org/abs/1901.10430
  """

  @classmethod
  def Params(cls):
    p = super(NormalizedDepthwiseConv2DLayer, cls).Params()
    p.Define('dropconnect_prob', 0.0,
             'Prob at which DropConnect regularization is performed.')
    p.Define('deterministic_dropout', False, 'Use determnisitc dropout or not.')
    p.Define('temperature', 1.0,
             'Temperature for the softmax normalization of the weights.')
    p.Define('weight_tiling_factor', 1,
             'Number of times weights are tiled over the input channels.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(NormalizedDepthwiseConv2DLayer, self).__init__(params)
    p = self.params
    assert p.filter_shape[1] == 1, 'Only 1d convolution is supported.'
    assert p.temperature > 0.0, 'Absolute zero temperature is not possible.'

  @property
  def output_channels(self):
    """The number of output channels for this conv layer."""
    p = self.params
    # Depthwise convolution filter shape is:
    # [kernel_size, 1, in_channels, channel_multiplier].
    return p.filter_shape[2] * p.filter_shape[3] * p.weight_tiling_factor

  @property
  def input_channels(self):
    """The number of output channels for this conv layer."""
    p = self.params
    return p.filter_shape[2] * p.weight_tiling_factor

  def _GetWeight(self, theta):
    p = self.params
    filter_w = theta.w
    filter_w.set_shape(p.filter_shape)

    # First normalize filter_w over the temporal dimension here.
    filter_w = tf.nn.softmax(filter_w / p.temperature, axis=0)

    # Add dropconnect on the weights for regularization.
    if p.dropconnect_prob > 0.0 and not p.is_eval:
      if p.deterministic_dropout:
        filter_w = py_utils.DeterministicDropout(
            filter_w, 1.0 - p.dropconnect_prob,
            py_utils.GenerateStepSeedPair(p, theta.global_step))
      else:
        filter_w = tf.nn.dropout(
            filter_w, 1.0 - p.dropconnect_prob, seed=p.random_seed)

    # Tie the parameters of every subsequent number of weight_tiling_factor
    # channels.
    filter_w = tf.tile(filter_w, [1, 1, p.weight_tiling_factor, 1])
    return filter_w


class CausalNormalizedDepthwiseConv2DLayer(NormalizedDepthwiseConv2DLayer):
  """Depthwise conv layer with causal dependency on the time axis."""

  def _EvaluateConvKernel(self, theta, inputs):
    """Apply convolution to inputs."""
    # Same as CausalDepthwiseConv2DLayer.
    p = self.params
    assert p.filter_shape[1] == 1, 'Only 1D causal convolutions supported.'
    padding_algorithm = 'VALID'
    causal_pad_size = (p.filter_shape[0] - 1) * p.dilation_rate[0]
    inputs = tf.pad(inputs, [[0, 0], [causal_pad_size, 0], [0, 0], [0, 0]])
    filter_w = self._GetWeight(theta)
    return tf.nn.depthwise_conv2d(
        inputs,
        filter_w,
        strides=[1, p.filter_stride[0], p.filter_stride[1], 1],
        rate=p.dilation_rate,
        data_format='NHWC',
        padding=padding_algorithm)


class ConvBatchNormLayer(bn_layers.BatchNormLayer):
  """A wrapper around regular BatchNormLayer that pass around the ...

  paddings layers.
  """

  def FProp(self, theta, inputs, paddings):
    paddings_expanded = tf.expand_dims(tf.expand_dims(paddings, -1), -1)
    bned = super(ConvBatchNormLayer, self).FProp(
        theta, inputs, paddings_expanded)
    return bned, paddings


# Supported activation functions.
_ACTIVATIONS = {
    'RELU': tf.nn.relu,
    'RELU6': tf.nn.relu6,
    'SIGMOID': tf.sigmoid,
    'TANH': tf.tanh,
    'SWISH': tf.nn.swish,
    'NONE': tf.identity,
}


class ActivationLayer(base_layer.BaseLayer):
  """Applies activation function to the inputs."""

  @classmethod
  def Params(cls):
    p = super(ActivationLayer, cls).Params()
    p.Define('activation', 'RELU',
             'The activation function to apply')
    return p

  def FProp(self, theta, inputs, paddings):
    p = self.params
    out = _ACTIVATIONS[p.activation](inputs)
    return out, paddings


class PaddingLayer(base_layer.BaseLayer):
  """Zeros out padded positions."""

  def FProp(self, theta, inputs, paddings):
    paddings_expanded = tf.expand_dims(tf.expand_dims(paddings, -1), -1)
    return inputs * (1.0 - paddings_expanded), paddings
