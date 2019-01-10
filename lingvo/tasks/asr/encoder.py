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
"""Encoders for the speech model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.python.ops import inplace_ops
from lingvo.core import base_encoder
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import model_helper
from lingvo.core import plot
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers
from lingvo.core import summary_utils

ConvLSTMBlock = collections.namedtuple('ConvLSTMBlock', ('rnn', 'cnn'))


class AsrEncoder(base_encoder.BaseEncoder):
  """Speech encoder version 1."""

  @classmethod
  def Params(cls):
    """Configs for AsrEncoder."""
    p = super(AsrEncoder, cls).Params()
    p.Define('lstm_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('cnn_tpl', layers.ConvLayer.Params(),
             'Configs template for the conv layer.')
    p.Define('proj_tpl', layers.ProjectionLayer.Params(),
             'Configs template for the projection layer.')
    p.Define(
        'highway_skip', False,
        'If set, residual connections from different layers are gated. '
        'Will only be used if residual_start is enabled.')
    p.Define('highway_skip_tpl', layers.HighwaySkipLayer.Params(),
             'Configs template for the highway skip layer.')
    p.Define('conv_lstm_tpl', rnn_cell.ConvLSTMCell.Params(),
             'Configs template for ConvLSTMCell.')
    p.Define(
        'after_conv_lstm_cnn_tpl', layers.ConvLayer.Params(),
        'Configs template for the cnn layer immediately follow the'
        ' convlstm layer.')
    p.Define('conv_filter_shapes', None, 'Filter shapes for each conv layer.')
    p.Define('conv_filter_strides', None, 'Filter strides for each conv layer.')
    p.Define('input_shape', [None, None, None, None],
             'Shape of the input. This should a TensorShape with rank 4.')
    p.Define('lstm_cell_size', 256, 'LSTM cell size for the RNN layer.')
    p.Define('num_cnn_layers', 2, 'Number of conv layers to create.')
    p.Define('num_conv_lstm_layers', 1, 'Number of conv lstm layers to create.')
    p.Define('num_lstm_layers', 3, 'Number of rnn layers to create')
    p.Define('project_lstm_output', True,
             'Include projection layer after each encoder LSTM layer.')
    p.Define('pad_steps', 6,
             'Extra zero-padded timesteps to add to the input sequence. ')
    p.Define(
        'residual_start', 0, 'Start residual connections from this lstm layer. '
        'Disabled if 0 or greater than num_lstm_layers.')
    p.Define('residual_stride', 1,
             'Number of lstm layers to skip per residual connection.')
    p.Define(
        'bidi_rnn_type', 'func', 'Options: func, native_cudnn. '
        'func: BidirectionalFRNN, '
        'native_cudnn: BidirectionalNativeCuDNNLSTM.')
    p.Define(
        'extra_per_layer_outputs', False,
        'Whether to output the encoding result from each encoder layer besides '
        'the regular final output. The corresponding extra outputs are keyed '
        'by "${layer_type}_${layer_index}" in the encoder output NestedMap, '
        'where layer_type is one of: "conv", "conv_lstm" and "rnn".')

    # TODO(yonghui): Maybe move those configs to a separate file.
    # Set some reasonable default values.
    #
    # NOTE(yonghui): The default config below assumes the following encoder
    # architecture:
    #
    #   cnn/batch-norm/relu ->
    #   cnn/batch-norm/relu ->
    #   bidirectional conv-lstm ->
    #   cnn/batch-norm/relu
    #   bidirectional lstm ->
    #   projection/batch-norm/relu ->
    #   bidirectional lstm ->
    #   projection/batch-norm/relu ->
    #   bidirectional lstm
    #
    # Default config for the rnn layer.
    p.lstm_tpl.params_init = py_utils.WeightInit.Uniform(0.1)

    # Default config for the convolution layer.
    p.input_shape = [None, None, 80, 3]
    p.conv_filter_shapes = [(3, 3, 3, 32), (3, 3, 32, 32)]
    p.conv_filter_strides = [(2, 2), (2, 2)]
    p.cnn_tpl.params_init = py_utils.WeightInit.TruncatedGaussian(0.1)
    # TODO(yonghui): Disable variational noise logic.
    # NOTE(yonghui): Fortunately, variational noise logic is currently not
    # implemented for ConvLayer yet (as of sep 22, 2016).

    # Default config for the projection layer.
    p.proj_tpl.params_init = py_utils.WeightInit.TruncatedGaussian(0.1)
    # TODO(yonghui): Disable variational noise logic.
    # NOTE(yonghui): Fortunately, variational noise logic is currently not
    # implemented for ProjectionLayer yet (as of sep 22, 2016).

    p.conv_lstm_tpl.filter_shape = [1, 3]  # height (time), width (frequency)
    p.conv_lstm_tpl.inputs_shape = [None, None, None, None]
    p.conv_lstm_tpl.cell_shape = [None, None, None, None]
    p.conv_lstm_tpl.params_init = py_utils.WeightInit.TruncatedGaussian(0.1)
    p.after_conv_lstm_cnn_tpl.filter_shape = [3, 3, None, None]
    p.after_conv_lstm_cnn_tpl.params_init = (
        py_utils.WeightInit.TruncatedGaussian(0.1))
    p.after_conv_lstm_cnn_tpl.filter_stride = [1, 1]
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(AsrEncoder, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed inputs are not yet supported for '
                                'AsrEncoder.')
    name = p.name

    with tf.variable_scope(name):
      # First create the conv layers.

      assert p.num_cnn_layers == len(p.conv_filter_shapes)
      assert p.num_cnn_layers == len(p.conv_filter_strides)
      params_conv_layers = []
      for i in range(p.num_cnn_layers):
        conv_p = p.cnn_tpl.Copy()
        conv_p.name = 'conv_L%d' % (i)
        conv_p.filter_shape = p.conv_filter_shapes[i]
        conv_p.filter_stride = p.conv_filter_strides[i]
        conv_p.is_eval = p.is_eval
        params_conv_layers.append(conv_p)
      self.CreateChildren('conv', params_conv_layers)

      conv_output_shape = p.input_shape
      for i in range(p.num_cnn_layers):
        conv_output_shape = self.conv[i].OutShape(conv_output_shape)
      assert len(conv_output_shape) == 4  # batch, height, width, channel.

      params_conv_lstm_rnn = []
      params_conv_lstm_cnn = []
      for i in range(p.num_conv_lstm_layers):
        # NOTE(yonghui): We assume that output from ConvLSTMBlock has the same
        # shape as its input.
        _, _, width, in_channel = conv_output_shape
        f_conv_lstm_p = p.conv_lstm_tpl.Copy()
        f_conv_lstm_p.name = 'f_conv_lstm_%d' % (i)
        f_conv_lstm_p.inputs_shape = [None, 1, width, in_channel]
        f_conv_lstm_p.cell_shape = [None, 1, width, in_channel]
        b_conv_lstm_p = f_conv_lstm_p.Copy()
        b_conv_lstm_p.name = 'b_conv_lstm_%d' % (i)
        conv_lstm_rnn_p = self.CreateConvLstmLayerParams()
        conv_lstm_rnn_p.name = 'conv_lstm_rnn'
        conv_lstm_rnn_p.fwd = f_conv_lstm_p
        conv_lstm_rnn_p.bak = b_conv_lstm_p
        params_conv_lstm_rnn.append(conv_lstm_rnn_p)
        cnn_p = p.after_conv_lstm_cnn_tpl.Copy()
        cnn_p.name = 'conv_lstm_cnn_%d' % (i)
        cnn_p.filter_shape[2] = 2 * in_channel
        cnn_p.filter_shape[3] = in_channel
        params_conv_lstm_cnn.append(cnn_p)
        # TODO(yonghui): Refactor ConvLSTMBlock into a layer.
      self.CreateChildren('conv_lstm_rnn', params_conv_lstm_rnn)
      self.CreateChildren('conv_lstm_cnn', params_conv_lstm_cnn)

      (self._first_lstm_input_dim,
       self._first_lstm_input_dim_pad) = self.FirstLstmLayerInputDimAndPadding(
           conv_output_shape, pad_to_multiple=16)

      # Now create all the rnn layers and projection layers.
      # TODO(yonghui): take care of device placement.
      params_rnn_layers = []
      params_proj_layers = []
      params_highway_skip_layers = []
      for i in range(p.num_lstm_layers):
        if i == 0:
          input_dim = self._first_lstm_input_dim
        else:
          input_dim = 2 * p.lstm_cell_size
        forward_p = p.lstm_tpl.Copy()
        forward_p.name = 'fwd_rnn_L%d' % (i)
        forward_p.num_input_nodes = input_dim
        forward_p.num_output_nodes = p.lstm_cell_size
        backward_p = forward_p.Copy()
        backward_p.name = 'bak_rnn_L%d' % (i)
        rnn_p = self.CreateBidirectionalRNNParams(forward_p, backward_p)
        rnn_p.name = 'brnn_L%d' % (i)
        params_rnn_layers.append(rnn_p)

        if p.project_lstm_output and (i < p.num_lstm_layers - 1):
          proj_p = p.proj_tpl.Copy()
          proj_p.input_dim = 2 * p.lstm_cell_size
          proj_p.output_dim = 2 * p.lstm_cell_size
          proj_p.name = 'proj_L%d' % (i)
          proj_p.is_eval = p.is_eval
          params_proj_layers.append(proj_p)

        # add the skip layers
        residual_index = i - p.residual_start + 1
        if p.residual_start > 0 and residual_index >= 0 and p.highway_skip:
          highway_skip = p.highway_skip_tpl.Copy()
          highway_skip.name = 'enc_hwskip_%d' % len(params_highway_skip_layers)
          highway_skip.input_dim = 2 * p.lstm_cell_size
          params_highway_skip_layers.append(highway_skip)
      self.CreateChildren('rnn', params_rnn_layers)
      self.CreateChildren('proj', params_proj_layers)
      self.CreateChildren('highway_skip', params_highway_skip_layers)

  @property
  def _use_functional(self):
    return True

  def CreateBidirectionalRNNParams(self, forward_p, backward_p):
    return model_helper.CreateBidirectionalRNNParams(self.params, forward_p,
                                                     backward_p)

  def CreateConvLstmLayerParams(self):
    return rnn_layers.BidirectionalFRNN.Params()

  def FirstLstmLayerInputDimAndPadding(self,
                                       conv_output_shape,
                                       pad_to_multiple=16):
    lstm_input_shape = conv_output_shape
    # Makes sure the lstm input dims is multiple of 16 (alignment
    # requirement from FRNN).
    first_lstm_input_dim_unpadded = lstm_input_shape[2] * lstm_input_shape[3]

    if self._use_functional and (first_lstm_input_dim_unpadded % pad_to_multiple
                                 != 0):
      first_lstm_input_dim = int(
          (first_lstm_input_dim_unpadded + pad_to_multiple - 1) /
          pad_to_multiple) * pad_to_multiple
    else:
      first_lstm_input_dim = first_lstm_input_dim_unpadded

    first_lstm_input_dim_padding = (
        first_lstm_input_dim - first_lstm_input_dim_unpadded)
    return first_lstm_input_dim, first_lstm_input_dim_padding

  @property
  def supports_streaming(self):
    return False

  def zero_state(self, batch_size):
    return py_utils.NestedMap()

  def FProp(self, theta, batch, state0=None):
    """Encodes source as represented by 'inputs' and 'paddings'.

    Args:
      theta: A NestedMap object containing weights' values of this
        layer and its children layers.
      batch: A NestedMap with fields:

        - src_inputs - The inputs tensor. It is expected to be of shape [batch,
          time, feature_dim, channels].
        - paddings - The paddings tensor. It is expected to be of shape [batch,
          time].
      state0: Recurrent input state. Not supported/ignored by this encoder.

    Returns:
      A NestedMap containing:

      - 'encoded': a feature tensor of shape [time, batch, depth]
      - 'padding': a 0/1 tensor of shape [time, batch]
      - 'state': the updated recurrent state
      - '${layer_type}_${layer_index}': The per-layer encoder output. Each one
        is a NestedMap containing 'encoded' and 'padding' similar to regular
        final outputs, except that 'encoded' from conv or conv_lstm layers are
        of shape [time, batch, depth, channels].
    """
    p = self.params
    inputs, paddings = batch.src_inputs, batch.paddings
    outputs = py_utils.NestedMap()
    with tf.name_scope(p.name):
      # Add a few extra padded timesteps at the end. This is for ensuring the
      # correctness of the conv-layers at the edges.
      if p.pad_steps > 0:
        # inplace_update() is not supported by TPU for now. Since we have done
        # padding on the input_generator, we may avoid this additional padding.
        assert not py_utils.use_tpu()
        inputs_pad = tf.zeros(
            inplace_ops.inplace_update(tf.shape(inputs), 1, p.pad_steps),
            inputs.dtype)
        paddings_pad = tf.ones(
            inplace_ops.inplace_update(tf.shape(paddings), 1, p.pad_steps),
            paddings.dtype)
        inputs = tf.concat([inputs, inputs_pad], 1, name='inputs')
        paddings = tf.concat([paddings, paddings_pad], 1)

      def ReshapeForPlot(tensor, padding, name):
        """Transposes and flattens channels to [batch, dim, seq_len] shape."""
        # Flatten any dimensions beyond the third into the third.
        batch_size = tf.shape(tensor)[0]
        max_len = tf.shape(tensor)[1]
        plot_tensor = tf.reshape(tensor, [batch_size, max_len, -1])
        plot_tensor = tf.transpose(plot_tensor, [0, 2, 1], name=name)
        return (plot_tensor, summary_utils.SequenceLength(padding))

      plots = [
          ReshapeForPlot(
              tf.transpose(inputs, [0, 1, 3, 2]), paddings, 'inputs')
      ]

      conv_out = inputs
      out_padding = paddings
      for i, conv_layer in enumerate(self.conv):
        conv_out, out_padding = conv_layer.FProp(theta.conv[i], conv_out,
                                                 out_padding)
        if p.extra_per_layer_outputs:
          conv_out *= (1.0 - out_padding[:, :, tf.newaxis, tf.newaxis])
          outputs['conv_%d' % i] = py_utils.NestedMap(
              encoded=tf.transpose(conv_out, [1, 0, 2, 3]),  # to [t, b, d, c]
              padding=tf.transpose(out_padding))
        plots.append(
            ReshapeForPlot(
                tf.transpose(conv_out, [0, 1, 3, 2]), out_padding,
                'conv_%d_out' % i))

      def TransposeFirstTwoDims(t):
        first_dim = tf.shape(t)[0]
        second_dim = tf.shape(t)[1]
        t_new = tf.transpose(
            tf.reshape(t, [first_dim, second_dim, -1]), [1, 0, 2])
        t_shape_new = tf.concat([[second_dim], [first_dim], tf.shape(t)[2:]], 0)
        return tf.reshape(t_new, t_shape_new)

      # Now the conv-lstm part.
      conv_lstm_out = conv_out
      conv_lstm_out_padding = out_padding
      for i, (rnn, cnn) in enumerate(
          zip(self.conv_lstm_rnn, self.conv_lstm_cnn)):
        conv_lstm_in = conv_lstm_out
        # Move time dimension to be the first.
        conv_lstm_in = TransposeFirstTwoDims(conv_lstm_in)
        conv_lstm_in = tf.expand_dims(conv_lstm_in, 2)
        conv_lstm_in_padding = tf.expand_dims(
            tf.transpose(conv_lstm_out_padding), 2)
        lstm_out = rnn.FProp(theta.conv_lstm_rnn[i], conv_lstm_in,
                             conv_lstm_in_padding)
        # Move time dimension to be the second.
        cnn_in = TransposeFirstTwoDims(lstm_out)
        cnn_in = tf.squeeze(cnn_in, 2)
        cnn_in_padding = conv_lstm_out_padding
        cnn_out, cnn_out_padding = cnn.FProp(theta.conv_lstm_cnn[i], cnn_in,
                                             cnn_in_padding)
        conv_lstm_out, conv_lstm_out_padding = cnn_out, cnn_out_padding
        if p.extra_per_layer_outputs:
          conv_lstm_out *= (
              1.0 - conv_lstm_out_padding[:, :, tf.newaxis, tf.newaxis])
          outputs['conv_lstm_%d' % i] = py_utils.NestedMap(
              encoded=tf.transpose(conv_lstm_out,
                                   [1, 0, 2, 3]),  # to [t, b, d, c]
              padding=tf.transpose(conv_lstm_out_padding))
        plots.append(
            ReshapeForPlot(conv_lstm_out, conv_lstm_out_padding,
                           'conv_lstm_%d_out' % i))

      # Need to do a reshape before starting the rnn layers.
      conv_lstm_out = py_utils.HasRank(conv_lstm_out, 4)
      conv_lstm_out_shape = tf.shape(conv_lstm_out)
      new_shape = tf.concat([conv_lstm_out_shape[:2], [-1]], 0)
      conv_lstm_out = tf.reshape(conv_lstm_out, new_shape)
      if self._first_lstm_input_dim_pad:
        conv_lstm_out = tf.pad(
            conv_lstm_out,
            [[0, 0], [0, 0], [0, self._first_lstm_input_dim_pad]])

      conv_lstm_out = py_utils.HasShape(conv_lstm_out,
                                        [-1, -1, self._first_lstm_input_dim])

      # Transpose to move the time dimension to be the first.
      rnn_in = tf.transpose(conv_lstm_out, [1, 0, 2])
      rnn_padding = tf.expand_dims(tf.transpose(conv_lstm_out_padding), 2)
      # rnn_in is of shape [time, batch, depth]
      # rnn_padding is of shape [time, batch, 1]

      # Now the rnn layers.
      num_skips = 0
      for i in range(p.num_lstm_layers):
        rnn_out = self.rnn[i].FProp(theta.rnn[i], rnn_in, rnn_padding)
        residual_index = i - p.residual_start + 1
        if p.residual_start > 0 and residual_index >= 0:
          if residual_index % p.residual_stride == 0:
            residual_in = rnn_in
          if residual_index % p.residual_stride == p.residual_stride - 1:
            # Highway skip connection.
            if p.highway_skip:
              rnn_out = self.highway_skip[num_skips].FProp(
                  theta.highway_skip[num_skips], residual_in, rnn_out)
              num_skips += 1
            else:
              # Residual skip connection.
              rnn_out += py_utils.HasShape(residual_in, tf.shape(rnn_out))
        if p.project_lstm_output and (i < p.num_lstm_layers - 1):
          # Projection layers.
          rnn_out = self.proj[i].FProp(theta.proj[i], rnn_out, rnn_padding)
        if i == p.num_lstm_layers - 1:
          rnn_out *= (1.0 - rnn_padding)
        if p.extra_per_layer_outputs:
          rnn_out *= (1.0 - rnn_padding)
          outputs['rnn_%d' % i] = py_utils.NestedMap(
              encoded=rnn_out, padding=tf.squeeze(rnn_padding, [2]))
        plots.append(
            ReshapeForPlot(
                tf.transpose(rnn_out, [1, 0, 2]),
                tf.transpose(rnn_padding, [1, 0, 2]), 'rnn_%d_out' % i))
        rnn_in = rnn_out
      final_out = rnn_in

      if self.cluster.add_summary:
        fig = plot.MatplotlibFigureSummary(
            'encoder_example', figsize=(8, len(plots) * 3.5))

        # Order layers from bottom to top.
        plots.reverse()
        for tensor, seq_len in plots:
          fig.AddSubplot(
              [tensor, seq_len],
              summary_utils.TrimPaddingAndPlotSequence,
              title=tensor.name,
              xlabel='Time')
        fig.Finalize()

      outputs['encoded'] = final_out
      outputs['padding'] = tf.squeeze(rnn_padding, [2])
      outputs['state'] = py_utils.NestedMap()
      return outputs
