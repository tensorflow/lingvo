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

import six
import tensorflow as tf

from tensorflow.python.framework import function
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import recurrent
from lingvo.core import summary_utils

# Supported activation functions.
_ACTIVATIONS = {
    'RELU': tf.nn.relu,
    'RELU6': tf.nn.relu6,
    'SIGMOID': tf.sigmoid,
    'TANH': tf.tanh
}

LOG_SCALE_CLAMP_BOUND = 20.0


def FPropDtype(params):
  return params.fprop_dtype if params.fprop_dtype is not None else params.dtype


class IdentityLayer(base_layer.LayerBase):
  """Identity layer, adds name and propagates its input."""

  @base_layer.initializer
  def __init__(self, params):
    super(IdentityLayer, self).__init__(params)

  def FProp(self, theta, inputs, *args):
    """Identity mapping.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor.  Shaped [..., input_dim].
      *args: Arguments to be ignored.
    Returns:
      Tensor with the same shape and type of inputs.
    """
    p = self.params
    return tf.identity(inputs, name=p.name)


class BatchNormLayer(base_layer.LayerBase):
  """Batch normalization layer."""

  @classmethod
  def Params(cls):
    p = super(BatchNormLayer, cls).Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define(
        'decay', 0.999,
        'Decay in updating the mean and variance moving average used in'
        ' batch normalization.')
    p.Define(
        'enable_cross_replica_sum_on_tpu', True,
        'If true, calls cross_replica_sum to the aggregate moving averages'
        ' across all replicas.')
    p.Define(
        'use_moving_avg_in_training', False,
        'If True, use global moving avg (mean, variance) during training'
        ' to avoid mismatch between train and eval, which then'
        ' essentially acts as an adaptive normalization step.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BatchNormLayer, self).__init__(params)
    p = self.params
    assert p.name

    pc = py_utils.WeightParams(
        shape=[p.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    with tf.variable_scope(p.name):
      if not p.use_moving_avg_in_training:
        self.CreateVariable('beta', pc)
        # Note, The real gamma to use is 1 + gamma.
        self.CreateVariable('gamma', pc, lambda x: 1.0 + x)

      # Two statistics.
      _, self._moving_mean = py_utils.CreateVariable(
          'moving_mean', pc, trainable=False)
      pc.init.scale = 1.0
      _, self._moving_variance = py_utils.CreateVariable(
          'moving_variance', pc, trainable=False)
    self._epsilon = 0.001
    self._decay = p.decay

  @staticmethod
  def _Moments(inputs, mask, enable_cross_replica_sum_on_tpu=False):
    """Computes mean and variance over the valid data points in inputs."""
    assert inputs.dtype == mask.dtype
    inputs = py_utils.with_dependencies([
        py_utils.assert_equal(tf.rank(inputs), tf.rank(mask)),
        py_utils.assert_greater_equal(mask, tf.cast(0., mask.dtype)),
    ], inputs)
    rank = tf.rank(mask)
    reduce_over_dims = tf.range(0, rank - 1)
    sum_v = tf.reduce_sum(inputs * mask, reduce_over_dims)
    count_v = tf.reduce_sum(mask, reduce_over_dims)
    # Input shape is guaranteed to be a multiple of mask shape because the
    # inputs * mask op above was successfully broadcasted.
    mask_multiplier = tf.shape(inputs)[:-1] // tf.shape(mask)[:-1]
    count_v *= tf.cast(tf.reduce_prod(mask_multiplier), count_v.dtype)
    if py_utils.use_tpu() and enable_cross_replica_sum_on_tpu:
      sum_v = tf.contrib.tpu.cross_replica_sum(sum_v)
      count_v = tf.contrib.tpu.cross_replica_sum(count_v)

    count_v = tf.maximum(count_v, 1.0)
    mean = sum_v / count_v
    sum_vv = tf.reduce_sum((inputs - mean) * (inputs - mean) * mask,
                           reduce_over_dims)

    if py_utils.use_tpu() and enable_cross_replica_sum_on_tpu:
      sum_vv = tf.contrib.tpu.cross_replica_sum(sum_vv)

    variance = py_utils.with_dependencies([
        py_utils.assert_greater_equal(sum_vv, tf.cast(0., sum_vv.dtype)),
    ], sum_vv / count_v)
    return mean, variance

  def FProp(self, theta, inputs, paddings=None):
    """Apply batch normalization.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor.  Shaped [..., dim].
      paddings: The paddings tensor.  Shaped [..., 1], with the same rank as
        the input tensor.
    Returns:
      Output after applying batch normalization, with the same shape as
      'inputs'.
    """
    if paddings is None:
      paddings = tf.zeros(
          tf.concat([tf.shape(inputs)[:-1], [1]], 0), dtype=inputs.dtype)

    p = self.params
    inputs = py_utils.with_dependencies([
        py_utils.assert_shape_match([tf.shape(inputs)[-1]], [p.dim]),
        py_utils.assert_shape_match([tf.shape(paddings)[-1]], [1]),
    ], inputs)
    with tf.name_scope(p.name):
      if p.is_eval:
        # The mean and variance used for normalization.
        norm_mean, norm_variance = self._moving_mean, self._moving_variance
      else:
        mean, variance = self._Moments(inputs, 1.0 - paddings,
                                       p.enable_cross_replica_sum_on_tpu)

        py_utils.UpdateBatchNormVars(self._moving_mean, mean, self._decay)
        py_utils.UpdateBatchNormVars(self._moving_variance, variance,
                                     self._decay)
        # Add some summaries for visualization.
        summary_utils.histogram(p, '%s_mean' % p.name, tf.cast(
            mean, tf.float32))
        summary_utils.histogram(p, '%s_variance' % p.name,
                                tf.cast(variance, tf.float32))
        summary_utils.histogram(p, '%s_moving_mean' % p.name,
                                tf.cast(self._moving_mean, tf.float32))
        summary_utils.histogram(p, '%s_moving_variance' % p.name,
                                tf.cast(self._moving_variance, tf.float32))
        summary_utils.histogram(p, '%s_mean_diff' % p.name,
                                tf.cast(mean - self._moving_mean, tf.float32))
        summary_utils.histogram(
            p, '%s_variance_diff' % p.name,
            tf.cast(variance - self._moving_variance, tf.float32))
        if p.use_moving_avg_in_training:
          # Use the global statistics for normalization.
          # Control dependencies on mean and variance make sure
          # moving_mean and variance will be updated for every training step.
          norm_mean = py_utils.with_dependencies([mean], self._moving_mean)
          norm_variance = py_utils.with_dependencies([variance],
                                                     self._moving_variance)
        else:
          # Use the batch statistics for normalization.
          norm_mean = mean
          norm_variance = variance

      norm_mean = py_utils.CheckNumerics(
          norm_mean, 'mean of %s failed numeric check' % p.name)
      norm_variance = py_utils.CheckNumerics(
          norm_variance, 'variance of %s failed numeric check' % p.name)

      if p.use_moving_avg_in_training:
        beta = 0.0
        gamma = 1.0
      else:
        beta = theta.beta
        gamma = theta.gamma

      with tf.control_dependencies([
          py_utils.assert_greater_equal(norm_variance, tf.cast(0., p.dtype)),
          py_utils.assert_shape_match([p.dim], tf.shape(norm_mean)),
          py_utils.assert_shape_match([p.dim], tf.shape(norm_variance)),
      ]):
        bn_output = tf.nn.batch_normalization(inputs, norm_mean, norm_variance,
                                              beta, gamma, self._epsilon)
      bn_output *= 1.0 - paddings
      return bn_output


def _ComputeOutputPadding(in_padding, stride):
  """Computes paddings for convolution and pooling output.

  Let the stride on the time axis be ts and the input sequence length be
  s_len, the output sequence length is s_len / ts. out_padding[i] == 1 iff any
  of in_padding[ts * i], ..., in_padding[ts * (i + 1) - 1] is 1.

  Args:
    in_padding: The paddings tensor. It is expected to be of shape [batch,
        time].
    stride: The time-stride between adjacent windows.

  Returns:
    out_padding: The new padding tensor of size [batch,
        ceil(time / stride)].
  """
  if stride == 1:
    return in_padding

  dtype = in_padding.dtype

  # in_padding is [b, t]
  b = tf.shape(in_padding)[0]
  t = tf.shape(in_padding)[1]

  # time dimension stride s.
  s = stride

  # We want to pad in_padding from [b, t] to [b, tt] so that tt
  # divides s.
  tt = ((t + s - 1) // s) * s
  extra_padding = tf.ones([b, tt - t], dtype)
  in_padding = tf.concat([in_padding, extra_padding], 1)

  in_padding = tf.reshape(in_padding, [b, -1, s])
  out_padding = tf.reduce_sum(in_padding, 2)

  # A timestep becomes a padded step as long as one of the underlying
  # t_stride steps is a padded one.
  out_padding = tf.cast(out_padding > 0.5, dtype)
  return out_padding


class ConvLayer(base_layer.LayerBase):
  """Convolution layer, with optional batch-normalization and activation."""

  @classmethod
  def Params(cls):
    p = super(ConvLayer, cls).Params()
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
        'causal_convolution', False,
        'If true, conv layer output only depends on time steps in'
        ' the past.')
    p.Define(
        'conv_last', False,
        'If true, apply the convolution transformation as the last step, '
        'i.e., first apply batch normalization on the input, followed '
        'by activation, and finally the convolution. '
        'Otherwise, apply convolution first, followed by batch '
        'normalization and activation.')
    p.Define(
        'weight_norm', False,
        'If true, apply weight normalization to weights as proposed by'
        ' Salimans and Kingma, 2016: https://arxiv.org/abs/1602.07868')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ConvLayer, self).__init__(params)
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
                shape=[p.filter_shape[-1]],
                init=py_utils.WeightInit.Constant(0.0),
                dtype=p.dtype,
                collections=[self.__class__.__name__ + '_vars']))
      if p.weight_norm:
        self.CreateVariable(
            'g',
            py_utils.WeightParams(
                shape=[p.filter_shape[-1]],
                init=py_utils.WeightInit.Constant(0.0),
                dtype=p.dtype,
                collections=[self.__class__.__name__ + '_vars']))

    if p.batch_norm:
      bn_params = BatchNormLayer.Params().Set(
          # batch normalization dimension is number of input channels if we
          # apply batch_norm on input and convolution in the end, number of
          # output channels otherwise.
          dim=p.filter_shape[2 if p.conv_last else 3],
          decay=p.bn_decay,
          name=p.name,
          params_init=p.params_init)
      self.CreateChild('bn', bn_params)
    # TODO(yonghui): implement the variational noise logic.

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    p = self.params
    assert isinstance(in_shape, tf.TensorShape)
    assert in_shape.ndims == 4
    # In the order of batch, time, frequency, channel
    n, t, f, c = in_shape.as_list()
    _, _, f_inc, f_outc = p.filter_shape
    # Last two dimensions has to be specified.
    assert f > 0 and c > 0
    assert c == f_inc
    t_stride = p.filter_stride[0]
    f_stride = p.filter_stride[1]
    ot = t
    if ot:
      ot = (ot + t_stride - 1) // t_stride
    of = (f + f_stride - 1) // f_stride
    oc = f_outc
    return tf.TensorShape([n, ot, of, oc])

  def _ApplyConv(self, theta, inputs):
    p = self.params
    w = theta.w
    strides = [p.filter_stride[0], p.filter_stride[1]]
    # TODO(miachen): remove casting once tf.nn.conv2d supports tf.float64.
    assert inputs.dtype == w.dtype
    dtype = inputs.dtype
    if dtype != tf.float32:
      inputs = tf.cast(inputs, tf.float32)
      w = tf.cast(w, tf.float32)
    if p.weight_norm:
      w = tf.nn.l2_normalize(w, [0, 1, 2]) * tf.reshape(
          (theta.g + 1.0), [1, 1, 1, p.filter_shape[-1]])

    conv_padding = 'SAME'
    if p.causal_convolution:
      assert p.filter_shape[1] == 1, 'Only 1D causal convolutions supported.'
      # Use VALID padding and shift the inputs to the right to ensure that the
      # first output only depends on the first input and so on. The output is
      # the same size as the input, as if the convolution used SAME padding.
      conv_padding = 'VALID'
      # The effective spatial filter width for dilated convolutions is
      # (kernel_width - 1) * dilation_rate + 1 as according to
      # https://www.tensorflow.org/api_docs/python/tf/nn/convolution.
      causal_pad_size = (p.filter_shape[0] - 1) * p.dilation_rate[0]
      inputs = tf.pad(inputs, [[0, 0], [causal_pad_size, 0], [0, 0], [0, 0]])
    out = tf.nn.convolution(
        inputs,
        w,
        strides=strides,
        dilation_rate=p.dilation_rate,
        data_format='NHWC',
        padding=conv_padding)
    if p.bias:
      b = tf.cast(theta.b, tf.float32)
      out = tf.nn.bias_add(out, b)
    if dtype != tf.float32:
      out = tf.cast(out, dtype)
    return py_utils.HasShape(out, [-1, -1, -1, p.filter_shape[3]])

  def FProp(self, theta, inputs, paddings=None):
    """Apply convolution to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch,
          time, frequency, channel]. The time dimension corresponds to
          the height dimension as in images and the frequency
          dimension corresponds to the width dimension as in images.
      paddings: The paddings tensor. If None, the inputs have no
        paddings in the sense of sequence training (e.g., in CNN
        models). Otherwise, it is expected to be of shape [batch,
        time].

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
    with tf.name_scope(p.name):
      if paddings is None:
        conv_padding = None
      else:
        conv_padding = _ComputeOutputPadding(paddings, p.filter_stride[0])
      if p.conv_last:
        out = inputs
        out_padding = paddings
      else:
        out = self._ApplyConv(theta, inputs)
        out_padding = conv_padding

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

      if p.conv_last:
        out = self._ApplyConv(theta, out)

      # Lastly zeroing out padded states.
      if conv_padding is not None:
        out *= tf.expand_dims(tf.expand_dims(1.0 - conv_padding, -1), -1)

      return out, conv_padding


class ProjectionLayer(base_layer.LayerBase):
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
    p.Define('cc_schedule', None, 'Clipping cap schedule.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ProjectionLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim > 0
    assert p.output_dim > 0
    assert p.activation == 'NONE' or p.activation in _ACTIVATIONS
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
      if p.cc_schedule is None:
        self.cc_schedule = None
      else:
        self.CreateChild('cc_schedule', p.cc_schedule)

    if p.batch_norm:
      bn_params = BatchNormLayer.Params().Set(
          dim=p.input_dim if p.affine_last else p.output_dim,
          decay=0.999,
          name=p.name)
      self.CreateChild('bn', bn_params)
    # TODO(yonghui): implement the variational noise logic.

  def FProp(self, theta, inputs, paddings=None):
    """Apply projection to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor.  Shaped [..., input_dim].
      paddings: The paddings tensor.  Shaped [..., 1], where all but the last
        dimension match.
    Returns:
      Output after applying projection, and optionally batch normalization and
      relu non-linearity.
    """
    if paddings is None:
      paddings = tf.zeros(
          tf.concat([tf.shape(inputs)[:-1], [1]], 0), dtype=inputs.dtype)
    p = self.params
    with tf.name_scope(p.name):
      if p.affine_last:
        out = inputs
      else:
        out = self._ApplyAffineTransformation(theta, inputs)
      if p.batch_norm:
        out = self.bn.FProp(theta.bn, out, paddings)
      if p.activation != 'NONE':
        out = py_utils.CheckNumerics(out)
        out = _ACTIVATIONS[p.activation](out)
      if p.affine_last:
        out = self._ApplyAffineTransformation(theta, out)
      out = self.ApplyClipping(theta, out)

      # lastly, zeroing out padded states.
      return py_utils.ApplyPadding(paddings, out)

  def _ApplyAffineTransformation(self, theta, inputs):
    p = self.params
    w = theta.w
    if p.weight_norm:
      w = tf.reshape((theta.g + 1.0) * tf.nn.l2_normalize(w, [0]),
                     [-1, p.output_dim])
    # Apply clipping after weight_norm. At inference time, weight normalization
    # reduces to a constant, and therefore, we want to make sure to apply
    # clipping after.
    w = self.ApplyClipping(theta, w)
    out = tf.reshape(
        py_utils.Matmul(tf.reshape(inputs, [-1, p.input_dim]), w),
        tf.concat([tf.shape(inputs)[:-1], [p.output_dim]], 0))
    if p.has_bias:
      b = self.ApplyClipping(theta, theta.b)
      out += b
    return out

  def ApplyClipping(self, theta, x):
    return (self.cc_schedule.ApplyClipping(theta.cc_schedule, x)
            if self.cc_schedule else x)


class FCLayer(ProjectionLayer):
  """Fully-connected layer (matmul + bias + optional activation)."""

  @classmethod
  def Params(cls):
    p = super(FCLayer, cls).Params()
    p.batch_norm = False
    p.has_bias = True
    return p


class PoolingLayer(base_layer.LayerBase):
  """Pooling layer, by default performs max-pooling."""

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

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    p = self.params
    assert isinstance(in_shape, tf.TensorShape)
    assert in_shape.ndims == 4
    # In the order of batch, time, frequency, channel
    n, t, f, c = in_shape.as_list()
    # Last two dimensions must be specified.
    assert f > 0 and c > 0
    t_stride = p.window_stride[0]
    f_stride = p.window_stride[1]
    ot = t
    if ot:
      ot = (ot + t_stride - 1) // t_stride
    of = (f + f_stride - 1) // f_stride
    oc = c
    return tf.TensorShape([n, ot, of, oc])

  def FProp(self, theta, inputs, paddings=None):
    """Apply pooling to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
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
        out_padding = _ComputeOutputPadding(paddings, p.window_stride[0])
      else:
        out_padding = None
      out = tf.nn.pool(
          inputs,
          window,
          p.pooling_type,
          strides=stride,
          padding='SAME',
          data_format='NHWC',
      )
      if out_padding is not None:
        out *= tf.expand_dims(tf.expand_dims(1.0 - out_padding, -1), -1)
      return out, out_padding


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
        ' or a scalar tensor.')
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
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a list of a single tensor, or a single tensor with the shape
        [..., input_dim].
      class_weights: a tensor with shape [...] containing the weights
          for each target word.
      class_ids: a tensor with shape [..., 1] of int32 dtype containing the
          target class labels.
      class_probabilities: a tensor with shape [..., num_classes] of
          float values indicating class-membership probabilities.

    Returns:
      A NestedMap containing the following fields:
        logits: with shape [..., num_classes]. Unnormalized softmax's logits.
        per_example_argmax: with shape [...]. argmax of i-th example.
        per_example_xent: with shape [...]. Cross entropy between i-th example's
          prediction and its label.
        per_example_weight: with shape [...]. class_weights casted to
          this layer's dtype.
        total_xent: A scalar. The sum of per_example_weight * per_example_xent.
        total_weight: A scalar. The sum of per_example_weight.
        avg_xent: A scalar. total_loss / total_weight.
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
      for i in range(p.num_shards):
        self.CreateVariable('weight_%d' % i, pc, self.AddGlobalVN)

      pc.shape = [num_classes_per_shard]
      pc.init.method = 'constant'
      pc.init.scale = 0.0
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
    # Note that this is generally not used in conjunction with quantization.
    abs_max = p.logits_abs_max
    if abs_max is not None:
      abs_min = -abs_max  # pylint: disable=invalid-unary-operand-type
      logits = tf.clip_by_value(logits, abs_min, abs_max)

    return self.QTensor('logits', logits)

  def Logits(self, theta, inputs):
    """Returns the logits computed before the softmax.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a list of a single tensor, or a single tensor with the shape
        [N, input_dim].

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
    elif p.num_sampled is 0 or p.is_eval:
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
          biases=tf.concat(
              [theta['bias_%d' % i] for i in range(p.num_shards)], axis=0),
          labels=tf.reshape(class_ids, [-1, 1]),
          inputs=self._GetInputs(inputs),
          num_sampled=p.num_sampled,
          num_classes=p.num_classes,
          partition_strategy='div')
      # Avoid computing logits; per_example_argmax is going to be always right.
      per_example_argmax = tf.identity(class_ids)

    label_weights = tf.reshape(tf.cast(class_weights, FPropDtype(p)), [-1])
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


class FeedForwardNet(base_layer.LayerBase):
  """"A simple multiple layer feedforward network.

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
        'dropout_prob', 0.0,
        'Probability at which we apply dropout to the hidden layer'
        ' outputs. This can be a single float, or a tuple/list of floats '
        ' having the same length as the number of layers.')
    p.Define(
        'batch_norm', False,
        'Whether or not to apply BN to hidden layer output. '
        'This can be a single bool or a tuple/list of bools having the '
        ' same length as the number of layers.')
    p.Define(
        'activation', 'RELU',
        'The activation function to use. Can be a single string, or a'
        ' tuple/list of strings having the same length as the number'
        ' of layers.')
    p.Define(
        'init', None, 'The initialization to use. Can be None, indicating the '
        'default initialization is used for every underlying '
        'ProjectionLayer; or a single WeightInit param, to be used for '
        'every layer; or a tuple/list of WeightInit params having the '
        'same length as the number of layers.')
    # We typically want to replace dropout by expectation during eval.
    # However, in certain cases E(f(x)) != f(E(x)), and replacing dropout by its
    # expectation during eval leads to worse quality.
    p.Define('dropout_at_eval', False,
             'Whether or not to also perform dropout at eval time.')
    p.Define('dropout_random_seed', None,
             'If not None, the random seed to use in tf.nn.dropout.')
    p.Define(
        'skip_connections', None,
        'This can be a single string or a tuple/list of strings, one per '
        'layer. '
        'If "ResNet", add a ResNet-style skip connections between input '
        'and output of the layer, requiring them to have the same depth. '
        'If "DenseNet", add a DenseNet-style skip connection between '
        'input and output of the layer.')
    return p

  _SKIP_CONNECTION_TYPES = ('ResNet', 'DenseNet')

  @base_layer.initializer
  def __init__(self, params):
    super(FeedForwardNet, self).__init__(params)
    p = self.params
    assert p.name

    batch_norm = p.batch_norm
    init = p.init
    num_layers = len(p.hidden_layer_dims)
    if isinstance(batch_norm, (list, tuple)):
      assert len(batch_norm) == num_layers
    else:
      batch_norm = [batch_norm] * num_layers
    if isinstance(init, (list, tuple)):
      assert len(init) == num_layers
    elif init is None:
      init = [p.params_init] * num_layers
    else:
      init = [init] * num_layers
    self._skip_connections = p.skip_connections
    if isinstance(self._skip_connections, (list, tuple)):
      assert len(self._skip_connections) == num_layers
    else:
      self._skip_connections = [p.skip_connections] * num_layers

    with tf.variable_scope(p.name):
      # Residual connections work better in the form of:
      #   y = x + Affine(Activation(BatchNorm(x)))
      params_fc_layers = []
      params_bn_layers = []
      in_dim = p.input_dim
      for i in range(num_layers):
        out_dim = p.hidden_layer_dims[i]
        proj_out_dim = out_dim
        if self._skip_connections[i] == 'ResNet':
          if out_dim != in_dim:
            # Disable ResNet.
            self._skip_connections[i] = 'NONE'
        elif self._skip_connections[i] == 'DenseNet':
          if out_dim > in_dim:
            proj_out_dim = out_dim - in_dim
          else:
            # Disable DenseNet.
            self._skip_connections[i] = 'NONE'
        name = '%s_%d' % (p.name, i)
        # We explicitly disable activation and batch_norm for ProjectLayer and
        # apply them separately to support skip connections in between.
        params_i = ProjectionLayer.Params().Set(
            batch_norm=False,
            has_bias=True,
            activation='NONE',
            input_dim=in_dim,
            output_dim=proj_out_dim,
            params_init=init[i],
            name=name)
        params_fc_layers.append(params_i)
        if batch_norm[i]:
          bn_params_i = BatchNormLayer.Params().Set(
              name=name, dim=proj_out_dim, params_init=init[i])
          params_bn_layers.append(bn_params_i)
        else:
          ident_params_i = IdentityLayer.Params().Set(name=name)
          params_bn_layers.append(ident_params_i)
        in_dim = out_dim

      self.CreateChildren('fc', params_fc_layers)
      self.CreateChildren('bn', params_bn_layers)

  def FProp(self, theta, inputs, paddings=None):
    p = self.params
    num_layers = len(self.fc)
    activation = p.activation
    if isinstance(activation, six.string_types):
      activation = [activation] * num_layers
    else:
      assert len(activation) == num_layers

    dropout_prob = p.dropout_prob
    if isinstance(dropout_prob, (list, tuple)):
      assert len(dropout_prob) == num_layers
    else:
      dropout_prob = [dropout_prob] * num_layers

    in_dim, layer_in = p.input_dim, inputs
    prev_proj_out = None
    for i in range(num_layers):
      layer_in = py_utils.with_dependencies(
          [py_utils.assert_shape_match([tf.shape(layer_in)[-1]], [in_dim])],
          layer_in)
      out_dim = p.hidden_layer_dims[i]
      layer_out = self.fc[i].FProp(theta.fc[i], layer_in, paddings)
      skip_connection = self._skip_connections[i]
      if skip_connection == 'ResNet' and prev_proj_out is not None:
        layer_out = tf.add(prev_proj_out, layer_out)
      prev_proj_out = layer_out
      layer_out = self.bn[i].FProp(theta.bn[i], layer_out, paddings)
      if activation[i] != 'NONE':
        layer_out = _ACTIVATIONS[activation[i]](layer_out)
      if dropout_prob[i] > 0.0 and (not p.is_eval or p.dropout_at_eval):
        layer_out = tf.nn.dropout(
            layer_out, 1.0 - dropout_prob[i], seed=p.dropout_random_seed)
      if skip_connection == 'DenseNet':
        layer_in = tf.concat([layer_in, layer_out], axis=-1)
      else:
        layer_in = layer_out
      in_dim = out_dim
    return layer_in


class DropoutLayer(base_layer.LayerBase):
  """Apply dropout during trainig."""

  @classmethod
  def Params(cls):
    p = super(DropoutLayer, cls).Params()
    p.Define('keep_prob', 1.0, 'Keep probability.')
    p.Define(
        'noise_shape', None, 'A 1-D `Tensor` of type `int32`, representing'
        ' the shape for randomly generated keep/drop flags.')
    p.Define('seed', None, 'Random seed')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(DropoutLayer, self).__init__(params)

  def FProp(self, theta, inputs):
    """Apply dropout to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor.
    Returns:
      inputs with dropout applied at training time.
    """
    p = self.params
    if p.keep_prob < 1.0 and not p.is_eval:
      return tf.nn.dropout(
          inputs, keep_prob=p.keep_prob, noise_shape=p.noise_shape, seed=p.seed)
    else:
      return inputs


class DeterministicDropoutLayer(base_layer.LayerBase):
  """Apply dropout during trainig."""

  @classmethod
  def Params(cls):
    p = super(DeterministicDropoutLayer, cls).Params()
    p.Define('keep_prob', 1.0, 'Keep probability.')
    p.Define('seed', None, 'Random seed')
    return p

  def FProp(self, theta, inputs):
    """Apply dropout to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor.
    Returns:
      inputs with dropout applied at training time.
    """
    p = self.params
    if p.keep_prob < 1.0 and not p.is_eval:
      return py_utils.DeterministicDropout(
          inputs,
          self.params.keep_prob,
          py_utils.GetOpSeedPair(op_seed=self.params.seed))
    else:
      return inputs


class LayerNorm(base_layer.LayerBase):
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
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: A tensor of shape [..., hidden_dim].
    Returns:
      tensor of the same shape with inputs
    """
    p = self.params
    inputs = py_utils.with_dependencies(
        [py_utils.assert_equal(tf.shape(inputs)[-1], p.input_dim)], inputs)

    @function.Defun(*[FPropDtype(p)] * 3, noinline=not py_utils.use_tpu())
    def Normalize(x, scale, bias):
      x_shape = tf.shape(x)
      inner_dim = x_shape[-1]
      x_reshaped = tf.reshape(x, [-1, inner_dim])
      mean = tf.reduce_mean(x_reshaped, axis=[1], keepdims=True)
      variance = tf.reduce_mean(
          tf.square(x_reshaped - mean), axis=[1], keepdims=True)
      x_norm = (x_reshaped - mean) * tf.rsqrt(variance + p.epsilon)
      x_norm = tf.reshape(x_norm, x_shape)
      return x_norm * (1.0 + scale) + bias

    return Normalize(inputs, theta.scale, theta.bias)
