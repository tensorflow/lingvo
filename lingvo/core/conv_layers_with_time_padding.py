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
"""Common conv layers.

WARNING: Strided convolutions are buggy. Consider using v2_padding=True.
"""

import copy

import lingvo.compat as tf
from lingvo.core import activations
from lingvo.core import base_layer
from lingvo.core import bn_layers
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import symbolic
from lingvo.core import tshape

ActivationLayer = activations.ActivationLayer


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


def ComputeConvOutputPadding(paddings,
                             window,
                             stride,
                             padding_algorithm='SAME',
                             v2_padding=False):
  """Computes paddings for convolution and pooling output.

  WARNING: This implementation is buggy prefer using ComputeConvOutputPaddingV2.

  out_padding[i] == 1 iff any in_padding corresponding to that output is 1.

  Args:
    paddings: The paddings tensor. It is expected to be of shape [batch, time].
    window: The size of the windows.
    stride: The time-stride between adjacent windows.
    padding_algorithm: 'SAME' or 'VALID'.
    v2_padding: Prefer setting to True. The default implementation is buggy for
      strided convolutions.

  Returns:
    out_padding, The new padding tensor of size [batch, ceil(time / stride)].
  """
  if v2_padding:
    return _ComputeConvOutputPaddingV2(paddings, window, stride,
                                       padding_algorithm)

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
      padding=padding_algorithm,
      strides=[stride],
  )
  return tf.squeeze(out_padding, -1)


def ComputeExplicitPaddingForCausalConv(filter_shape, dilation_rate):
  """Computes the explicit paddings for causal convolutions.

  Args:
    filter_shape: a sequence of length 4. Elements are in the order of height
      (time), width (frequency), in_channel, out_channel.
    dilation_rate: a pair of int: dilations on height and width axises.

  Returns:
    explicit_padding: a list of pairs in the form of :
      [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
  """
  assert filter_shape[1] == 1, 'Only 1D causal convolutions supported.'
  # Use VALID padding and shift the inputs to the right to ensure that the
  # first output only depends on the first input and so on. The output is
  # the same size as the input, as if the convolution used SAME padding.
  # The effective spatial filter width for dilated convolutions is
  # (kernel_width - 1) * dilation_rate + 1 as according to
  # https://www.tensorflow.org/api_docs/python/tf/nn/convolution.
  causal_pad_size = (filter_shape[0] - 1) * dilation_rate[0]
  explicit_padding = [[0, 0], [causal_pad_size, 0], [0, 0], [0, 0]]
  return explicit_padding


def _ComputeConvOutputPaddingV2(paddings,
                                window,
                                stride,
                                padding_algorithm='SAME'):
  """Computes paddings for convolution and pooling output.

  - If padding_algorithm='SAME': out_padding[i] == 0 if the in_padding
    corresponding to that output is 0. This prevents the output from shrinking
    unnecessarily when striding.
  - If padding algorithm='VALID': out_padding[i] == 1 iff any in_padding
    corresponding to that output is 1.

  Args:
    paddings: The paddings tensor. It is expected to be of shape [batch, time].
    window: The size of the windows.
    stride: The time-stride between adjacent windows.
    padding_algorithm: 'SAME' or 'VALID'.

  Returns:
    out_padding, The new padding tensor of size [batch, ceil(time / stride)].
  """
  if stride == 1 and padding_algorithm == 'SAME':
    return paddings

  paddings, slice_len = _PadForLengthCompatibleStridesV2(
      paddings, stride, padding_algorithm, 1.0)

  expanded_paddings = tf.expand_dims(paddings, -1)

  if padding_algorithm == 'SAME':
    # Using a strided conv1d of size 1x1 we find all non-padded positions for
    # the specified stride.
    out_paddings = tf.nn.conv1d(
        expanded_paddings,
        filters=tf.ones([1, 1, 1], paddings.dtype),
        stride=stride,
        padding='SAME',
        name='padding_conv')
  elif padding_algorithm == 'VALID':
    out_paddings = tf.nn.pool(
        expanded_paddings, [window],
        'MAX',
        padding=padding_algorithm,
        strides=[stride])
  out_paddings = tf.squeeze(out_paddings, -1)
  if stride > 1:
    slice_end = py_utils.GetShape(out_paddings)[1] - slice_len
    out_paddings = out_paddings[:, :slice_end]
  return out_paddings


def _PadForLengthCompatibleStridesV2(tensor, stride, padding_algorithm,
                                     constant_values):
  """Pads tensor to make strided convolutions start in the first position.

  Tensorflow strided convolutions and Lingvo paddings are incompatible.
  Strided convolutions always end at the last index of the length dimension.
  Therefore, the output of a Lingvo padded tensor depends on the length
  dimension. Here we remove this dependency by pre-padding the tensor so that
  the first convolution starts in the first position.

  Args:
    tensor: The tensor to prepare for convolution. [batch, time, ...].
    stride: The stride in the length dimension.
    padding_algorithm: 'SAME' or 'VALID'.
    constant_values: Value to pad 0. for data tensor and 1.0 for padding tensor.

  Returns:
    A tuple (tensor, padded_length) where tensor is the potentionally padded
    tensor and padded_length is the number paddings.
  """
  if padding_algorithm == 'VALID':
    return tensor, 0

  input_length = py_utils.GetShape(tensor)[1]
  pad_len = ((input_length // stride) + 1) * stride - 1 - input_length
  if pad_len == 0:
    return tensor, 0
  tensor = py_utils.PadSequenceDimension(tensor, input_length + pad_len,
                                         constant_values)
  return tensor, pad_len


class BaseConv2DLayerWithPadding(base_layer.BaseLayer):
  """Abstract base class for 2D convolution layers.

  WARNING: Strided convolutions are buggy. Prefer using v2_padding=True.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
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
    p.Define(
        'partial_conv', False, 'If true, rescale positions near sequence'
        'boundaries as proposed in https://arxiv.org/abs/1811.11718')
    p.Define(
        'v2_padding', False, 'Prefer setting to True. The default '
        'implementation is incorrect for strided convolutions.')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert all(x > 0 for x in p.filter_stride)
    assert len(p.dilation_rate) == 2
    assert all(x > 0 for x in p.dilation_rate)
    # Dilation and stride can't be combined.
    if any(x > 1 for x in p.dilation_rate):
      assert all(x == 1 for x in p.filter_stride)

  @classmethod
  def OutputChannels(cls, p):
    """The number of output channels for this conv layer."""
    raise NotImplementedError()

  @property
  def output_channels(self):
    return self.OutputChannels(self.params)

  @property
  def input_channels(self):
    """The number of input channels for this conv layer."""
    return self.params.filter_shape[2]

  @property
  def filter_stride(self):
    return self.params.filter_stride

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
              tf.concat([
                  tf.shape(paddings),
                  [-1, symbolic.ToStatic(self.input_channels)]
              ], 0))
      ], inputs)

      def _ApplyPadding(tensor_in, padding_in):
        padding_expanded = tf.expand_dims(tf.expand_dims(padding_in, -1), -1)
        return tensor_in * (1.0 - padding_expanded)

      # Zeroing out padded inputs.
      inputs = _ApplyPadding(inputs, paddings)

      # Apply conv on 'inputs'.
      if p.v2_padding:
        padded_inputs, slice_len = _PadForLengthCompatibleStridesV2(
            inputs, p.filter_stride[0], 'SAME', 0.)
        out = self._ApplyConv(theta, padded_inputs)
        if p.filter_stride[0] > 1:
          slice_end = py_utils.GetShape(out)[1] - slice_len
          out = out[:, :slice_end, :, :]
      else:
        out = self._ApplyConv(theta, inputs)

      if p.partial_conv:
        out = self._RescaleBoundary(out, paddings)
      # NOTE: this may be slightly inaccurate when p.dilation_rate[0] > 1.
      # But there's likely no real problems. Trying to set it gives an error:
      # pooling with SAME padding is not implemented for dilation_rate > 1.
      # implementation. Consider updating it to be the actual shape.
      if p.v2_padding:
        conv_padding = _ComputeConvOutputPaddingV2(
            paddings, window=p.filter_shape[0], stride=p.filter_stride[0])
      else:
        conv_padding = ComputeConvOutputPadding(
            paddings, window=p.filter_stride[0], stride=p.filter_stride[0])

      # Assuming padded nodes will be properly zero-ed out if necessary by
      # sub-sequent layers.
      # out = _ApplyPadding(out, conv_padding)
      out = py_utils.HasShape(
          out, symbolic.ToStatic(self.OutShape(tf.shape(inputs))))
      return out, conv_padding

  def _RescaleBoundary(self, out, in_paddings):
    # Rescale every output position by:
    #   (# input positions) / (# non-padding input positions)
    # where (# input positions) = filter_size.
    p = self.params
    in_mask = 1.0 - in_paddings

    # Compute the left and right implicity padding size used in 'SAME' mode.
    filter_t = p.filter_shape[0]
    effective_filter_size = (filter_t - 1) * p.dilation_rate[0] + 1
    left_pad_size = (effective_filter_size - 1) // 2
    right_pad_size = effective_filter_size // 2

    # Compute the rescaling factor.
    # This expanded tensor has 1 on all valid positions, 0 on all padded ones,
    # which include both explicit padding provided by 'in_padding', and implicit
    # padding on boundaries.
    in_mask_padded = tf.pad(in_mask, [[0, 0], [left_pad_size, right_pad_size]])
    # (# non-padding input positions) / (# input positions)
    factor_inverse = tf.nn.pool(
        in_mask_padded[:, :, tf.newaxis],
        window_shape=(filter_t,),
        pooling_type='AVG',
        strides=(p.filter_stride[0],),
        padding='VALID',
        dilations=(p.dilation_rate[0],))

    factor = tf.math.reciprocal_no_nan(factor_inverse)
    return out * factor[..., tf.newaxis]

  def _ApplyConv(self, theta, conv_input):
    return self._EvaluateConvKernel(theta, conv_input,
                                    self._MaybeCausalPadding())

  def _MaybeCausalPadding(self):
    """Returns the padding algorithm for tf.*conv2d api.

    The default value is 'SAME' for non-causal layers. Causal layers may
      override and return a explicit padding.
    """
    return 'SAME'

  def _EvaluateConvKernel(self, theta, conv_input, padding_algorithm):
    """Evaluate the convolution kernel on input 'conv_input'."""
    raise NotImplementedError


class Conv2DLayerWithPadding(BaseConv2DLayerWithPadding):
  """Conv2D layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('bias', False, 'Whether or not to apply a bias before activation.')
    return p

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('w', w_pc)
    if p.weight_norm:
      self.CreateVariable(
          'g',
          py_utils.WeightParams(
              shape=[p.filter_shape[-1]],
              init=py_utils.WeightInit.Constant(0.0),
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']))
    if p.bias:
      # NOTE(jiahuiyu): bias is subject to LP regularization in this version.
      self.CreateVariable(
          'b',
          py_utils.WeightParams(
              shape=[self.output_channels],
              init=py_utils.WeightInit.Constant(0.0),
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']))

  @classmethod
  def OutputChannels(cls, p):
    """The number of output channels for this conv layer."""
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

  def _ApplyConv(self, theta, conv_input):
    out = self._EvaluateConvKernel(theta, conv_input,
                                   self._MaybeCausalPadding())
    p = self.params
    if p.bias:
      out = tf.nn.bias_add(out, theta.b)
    return out

  def _EvaluateConvKernel(self, theta, inputs, padding_algorithm):
    """Apply convolution to inputs."""
    p = self.params
    filter_w = self._GetWeight(theta)
    return tf.nn.conv2d(
        inputs,
        filter_w,
        strides=p.filter_stride,
        dilations=p.dilation_rate,
        data_format='NHWC',
        padding=padding_algorithm)


class CausalConv2DLayerWithPadding(Conv2DLayerWithPadding):
  """2D conv layer with causal dependency on the time axis."""

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.filter_shape[1] == 1, 'Only 1d causal convolution is supported.'

  def _MaybeCausalPadding(self):
    p = self.params
    return ComputeExplicitPaddingForCausalConv(p.filter_shape, p.dilation_rate)

  def zero_state(self, batch_size):
    """Returns the initial state given the batch size.

    Args:
      batch_size: the batch size.

    Returns:
      state0: A NestedMap of tensors including:
        - context: A Tensor of shape [b, filter_shape[0]-1, 1, c].
    """
    p = self.params
    assert p.filter_shape[1] == 1, (
        'zero_state() only supports 1d causal convolution.')

    context = tf.zeros(
        shape=[batch_size] +
        [p.filter_shape[0] - 1, p.filter_shape[1], p.filter_shape[2]],
        dtype=py_utils.FPropDtype(p))
    return py_utils.NestedMap(context=context)

  def StreamStep(self, theta, inputs, paddings, state0):
    """Apply a singele step of convolution to input_tensor.

    Only supports 1d causal convolution. Doesn't support dilation.

    Args:
      theta: A NestedMap of layer params.
      inputs: A Tensor of shape [b, t, 1, c]
      paddings: A 0/1 valued tensor of shape [b, t].
      state0: A NestedMap of tensors of the same struct as returned by
        zero_state().

    Returns:
      outputs: A Tensor of shape [b, t, 1, c]
      padding: the same as input paddings.
      state1: A NestedMap of the same struct as input state
    """
    p = self.params
    assert p.filter_shape[1] == 1, (
        'StreamStep only supports 1d causal convolution.')
    assert all(stride == 1 for stride in p.filter_stride), (
        f'StreamStep doesn\'t support striding: {p.filter_stride}')
    assert p.dilation_rate == (1, 1), ('StreamStep doesn\'t support dilation')

    with tf.name_scope(p.name):
      inputs = py_utils.HasShape(inputs, [-1, -1, 1, p.filter_shape[2]])
      paddings = py_utils.HasShape(paddings, py_utils.GetShape(inputs)[:2])
      q = py_utils.GetShape(paddings)[1]

      concat_inputs = tf.concat(
          [state0.context, inputs * (1 - py_utils.AppendDims(paddings, 2))],
          axis=1)
      outputs = tf.nn.conv2d(
          concat_inputs,
          self._GetWeight(theta),
          strides=p.filter_stride,
          dilations=p.dilation_rate,
          data_format='NHWC',
          padding='VALID')
      new_context = concat_inputs[:, q:]
      return outputs, paddings, py_utils.NestedMap(context=new_context)


class DepthwiseConv2DLayer(BaseConv2DLayerWithPadding,
                           quant_utils.QuantizableLayer):
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
    p.Define('bias', False, 'Whether or not to apply a bias before activation.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateAqtWeight('w', self._GetWeightShape(), feature_axis=(2, 3))

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    self.CreateVariable('w', w_pc)
    if p.weight_norm:
      self.CreateVariable(
          'g',
          py_utils.WeightParams(
              shape=[p.filter_shape[2], p.filter_shape[3]],
              init=py_utils.WeightInit.Constant(0.0),
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']))
    if p.bias:
      # NOTE(jiahuiyu): bias is subject to LP regularization in this version.
      self.CreateVariable(
          'b',
          py_utils.WeightParams(
              shape=[self.output_channels],
              init=py_utils.WeightInit.Constant(0.0),
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']))

  @classmethod
  def OutputChannels(cls, p):
    """The number of output channels for this conv layer."""
    # Depthwise convolution filter shape is:
    #   [..., in_channels, channel_multiplier].
    return p.filter_shape[2] * p.filter_shape[3]

  def _GetWeight(self, theta):
    p = self.params
    if p.weight_norm:
      # Normalize along the feature dimensions.
      w_norm = tf.nn.l2_normalize(theta.w, [0, 1])
      g = tf.reshape((theta.g + 1.0), [1, 1, *p.filter_shape[-2:]])
      return w_norm * g
    else:
      return theta.w

  def _GetWeightShape(self):
    """The shape of the filter returned by _GetWeight."""
    return self.params.filter_shape

  def _ApplyConv(self, theta, conv_input):
    out = self._EvaluateConvKernel(theta, conv_input,
                                   self._MaybeCausalPadding())
    p = self.params
    if p.bias:
      out = tf.nn.bias_add(out, theta.b)
    return out

  def _EvaluateConvKernel(self, theta, inputs, padding_algorithm):
    """Apply convolution to inputs."""
    p = self.params
    filter_w = self._GetWeight(theta)
    inputs, filter_w = self.ToAqtConv(
        'w', inputs, filter_w, w_feature_axis=(2, 3))
    output = tf.nn.depthwise_conv2d(
        inputs,
        filter_w,
        strides=[1, p.filter_stride[0], p.filter_stride[1], 1],
        dilations=p.dilation_rate,
        data_format='NHWC',
        padding=padding_algorithm)
    return self.FromAqtConv('w', output, is_depthwise=True)


class CausalDepthwiseConv2DLayer(DepthwiseConv2DLayer):
  """Depthwise conv layer with causal dependency on the time axis."""

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.filter_shape[1] == 1, 'Only 1d causal convolution is supported.'

  def _MaybeCausalPadding(self):
    p = self.params
    return ComputeExplicitPaddingForCausalConv(p.filter_shape, p.dilation_rate)

  def zero_state(self, batch_size):
    """Returns the initial state given the batch size.

    Args:
      batch_size: the batch size.

    Returns:
      state0: A NestedMap of tensors including:
        - context: A Tensor of shape [b, filter_shape[0]-1, 1, c].
    """
    p = self.params
    assert p.filter_shape[1] == 1, (
        'zero_state() only supports 1d causal convolution.')

    context = tf.zeros(
        shape=[batch_size] +
        [p.filter_shape[0] - 1, p.filter_shape[1], p.filter_shape[2]],
        dtype=py_utils.FPropDtype(p))
    return py_utils.NestedMap(context=context)

  def StreamStep(self, theta, inputs, paddings, state0):
    """Apply a singele step of convolution to input_tensor.

    Only supports 1d causal convolution. Doesn't support dilation.

    Args:
      theta: A NestedMap of layer params.
      inputs: A Tensor of shape [b, t, 1, c]
      paddings: A 0/1 valued tensor of shape [b, t].
      state0: A NestedMap of tensors of the same struct as returned by
        zero_state().

    Returns:
      outputs: A Tensor of shape [b, t, 1, c * channel_multiplier]
      padding: the same as input paddings.
      state1: A NestedMap of the same struct as input state
    """
    p = self.params
    assert p.filter_shape[1] == 1, (
        'StreamStep only supports 1d causal convolution.')
    assert p.filter_stride[0] == 1, ('StreamStep doesn\'t support striding')
    assert p.dilation_rate == (1, 1), ('StreamStep doesn\'t support dilation')

    with tf.name_scope(p.name):
      inputs = py_utils.HasShape(inputs, [-1, -1, 1, p.filter_shape[2]])
      paddings = py_utils.HasShape(paddings, py_utils.GetShape(inputs)[:2])
      q = py_utils.GetShape(paddings)[1]

      padded_inputs = py_utils.ApplyPadding(
          py_utils.AppendDims(paddings, 2), inputs)

      concat_inputs = tf.concat([state0.context, padded_inputs], axis=1)
      outputs = tf.nn.depthwise_conv2d(
          concat_inputs,
          self._GetWeight(theta),
          strides=(1, 1, 1, 1),
          dilations=(1, 1),
          data_format='NHWC',
          padding='VALID')
      new_context = concat_inputs[:, q:]
      return outputs, paddings, py_utils.NestedMap(context=new_context)


class NormalizedDepthwiseConv2DLayer(DepthwiseConv2DLayer):
  """DepthwiseConv2DLayer where weights are normalized over the time dim.

  https://arxiv.org/abs/1901.10430
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dropconnect_prob', 0.0,
             'Prob at which DropConnect regularization is performed.')
    p.Define('deterministic_dropout', False, 'Use determnisitc dropout or not.')
    p.Define('temperature', 1.0,
             'Temperature for the softmax normalization of the weights.')
    p.Define('weight_tiling_factor', 1,
             'Number of times weights are tiled over the input channels.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.filter_shape[1] == 1, 'Only 1d convolution is supported.'
    assert p.temperature > 0.0, 'Absolute zero temperature is not possible.'

  @classmethod
  def OutputChannels(cls, p):
    """The number of output channels for this conv layer."""
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

    # First normalize filter_w over the temporal dimension here.
    filter_w = tf.nn.softmax(filter_w / p.temperature, axis=0)

    # Add dropconnect on the weights for regularization.
    if p.dropconnect_prob > 0.0 and not self.do_eval:
      if p.deterministic_dropout:
        filter_w = py_utils.DeterministicDropout(
            filter_w, 1.0 - p.dropconnect_prob,
            py_utils.GenerateStepSeedPair(p))
      else:
        filter_w = tf.nn.dropout(
            filter_w, rate=p.dropconnect_prob, seed=p.random_seed)

    # Tie the parameters of every subsequent number of weight_tiling_factor
    # channels.
    filter_w = tf.tile(filter_w, [1, 1, p.weight_tiling_factor, 1])
    return filter_w

  def _GetWeightShape(self):
    """The shape of the filter returned by _GetWeight."""
    p = self.params
    tiled_shape = copy.deepcopy(p.filter_shape)
    tiled_shape[2] *= p.weight_tiling_factor
    return tiled_shape

  @classmethod
  def FPropMeta(cls, p, inputs, paddings):
    py_utils.CheckShapes((inputs, paddings))
    b, t, f, _ = inputs
    assert f == 1
    oc = p.filter_shape[2] * p.filter_shape[3] * p.weight_tiling_factor
    outputs = tshape.Shape([b, t, f, oc])
    flops = b * t * f * p.filter_shape[0] * oc * 5
    return py_utils.NestedMap(flops=flops, out_shapes=(outputs, paddings))


class CausalNormalizedDepthwiseConv2DLayer(NormalizedDepthwiseConv2DLayer):
  """Depthwise conv layer with causal dependency on the time axis."""

  def _MaybeCausalPadding(self):
    p = self.params
    return ComputeExplicitPaddingForCausalConv(p.filter_shape, p.dilation_rate)


class ConvBatchNormLayer(bn_layers.BatchNormLayer):
  """A wrapper around regular BatchNormLayer that pass around the ...

  paddings layers.
  """

  def FProp(self, theta, inputs, paddings):
    paddings_expanded = tf.expand_dims(tf.expand_dims(paddings, -1), -1)
    bned = super().FProp(theta, inputs, paddings_expanded)
    return bned, paddings


class ConvCategoricalBN(bn_layers.CategoricalBN):
  """A wrapper around regular CategoricalBN that pass around the ...

  paddings layers.
  """

  def FProp(self, theta, inputs, paddings, class_emb):
    paddings_expanded = tf.expand_dims(tf.expand_dims(paddings, -1), -1)
    bned = super().FProp(theta, inputs, paddings_expanded, class_emb)
    return bned, paddings


class PaddingLayer(base_layer.BaseLayer):
  """Zeros out padded positions."""

  def FProp(self, theta, inputs, paddings):
    paddings_expanded = tf.expand_dims(tf.expand_dims(paddings, -1), -1)
    return inputs * (1.0 - paddings_expanded), paddings


class GlobalPoolingLayer(base_layer.BaseLayer):
  """Padding aware global pooling."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('pooling_type', 'MAX', 'Pooling type: MAX|AVG')
    return p

  def FProp(self, theta, inputs, paddings):
    """Apply global spatial pooling to inputs.

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
       - outputs: has shape [batch, 1, 1, channel].
       - out_paddings: None or has shape [batch, 1].
    """
    p = self.params
    assert p.pooling_type in ['MAX', 'AVG'], p.pooling_type
    b, t, f = py_utils.GetShape(inputs, ndims=3)

    if paddings is not None:
      paddings = py_utils.HasShape(paddings, [b, t])

    if paddings is not None:
      mask = 1.0 - paddings[..., tf.newaxis, tf.newaxis]
    else:
      mask = tf.ones([b, t, 1, 1], p.dtype)
    if p.pooling_type == 'AVG':
      global_sum = tf.reduce_sum(inputs * mask, axis=[1, 2], keepdims=True)
      f = tf.cast(tf.convert_to_tensor(f), p.dtype)
      count = f * tf.reduce_sum(mask, axis=[1, 2], keepdims=True)
      out_feature = global_sum / tf.maximum(1.0, count)
    elif p.pooling_type == 'MAX':
      large_negative = (
          tf.ones_like(inputs) * p.dtype.max * tf.constant(-0.7, dtype=p.dtype))
      padded_inputs = tf.where_v2(mask > 0.0, inputs, large_negative)
      out_feature = tf.reduce_max(padded_inputs, axis=[1, 2], keepdims=True)
    if paddings is None:
      out_paddings = None
    else:
      out_paddings = tf.reduce_min(paddings, axis=1, keepdims=True)
      out_feature *= 1.0 - out_paddings[..., tf.newaxis, tf.newaxis]
    return out_feature, out_paddings
