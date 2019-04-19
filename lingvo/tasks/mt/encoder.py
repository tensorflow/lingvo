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
"""Encoders for the machine translation model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


from six.moves import range
import tensorflow as tf

from lingvo.core import base_encoder
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import model_helper
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import summary_utils
from lingvo.tasks.mt import layers as mt_layers

tf.flags.DEFINE_bool('transformer_encoder_truncates_inputs', False,
                     'Whether TransformerEncoder truncates inputs to max len.')


class MTEncoderV1(base_encoder.BaseEncoder):
  """Machine translation encoder version 1."""

  @classmethod
  def Params(cls):
    """Configs for `MTEncoderV1`."""
    p = super(MTEncoderV1, cls).Params()
    p.Define('emb', layers.EmbeddingLayer.Params(), 'Embedding layer params.')
    p.Define('lstm_tpl',
             rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('lstm_tpl_uni', None,
             'Override configs template for the unidirectional RNN layers.')
    p.Define('lstm_tpl_bidi', None,
             'Override configs template for the bidirectional RNN layer.')
    p.Define('lstm_cell_size', 1024, 'LSTM cell size for the RNN layer.')
    p.Define('num_lstm_layers', 8, 'Number of rnn layers to create')
    p.Define('dropout_prob', 0.0, 'Prob at which we do dropout.')
    p.Define('unidi_rnn_type', 'func', 'Options: func, native_cudnn. '
             'func: FRNN, native_cudnn: CuDNNLSTM.')
    p.Define(
        'bidi_rnn_type', 'func', 'Options: func, native_cudnn. '
        'func: BidirectionalFRNN, '
        ' native_cudnn: BidirectionalNativeCuDNNLSTM.')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')

    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    default_params_init = py_utils.WeightInit.Uniform(0.04)

    # Default config for the embedding.
    p.emb.vn = disable_vn
    p.emb.vocab_size = 32000
    p.emb.embedding_dim = 1024
    p.emb.max_num_shards = 16
    p.emb.params_init = default_params_init

    for tpl in [p.lstm_tpl, p.lstm_tpl_uni, p.lstm_tpl_bidi]:
      if tpl is not None:
        tpl.vn = disable_vn
        tpl.params_init = default_params_init
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTEncoderV1, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed inputs are not yet supported for '
                                'MTEncoderV1.')

    with tf.variable_scope(p.name):
      if p.cc_schedule is not None:
        self.CreateChild('cc_schedule', p.cc_schedule)

      self.CreateChild('emb', p.emb)

      rnn_layers_params = []

      # L0 is a bi-directional lstm.

      # L0's forward lstm cell
      if p.lstm_tpl_bidi is None:
        params = p.lstm_tpl.Copy()
      else:
        params = p.lstm_tpl_bidi.Copy()
      params.name = 'L0_rnn_fwd'
      params.num_input_nodes = p.emb.embedding_dim
      params.num_output_nodes = p.lstm_cell_size
      forward_lstm = params

      # L0's backward lstm cell
      params = params.Copy()
      params.name = 'L0_rnn_bak'
      backward_lstm = params

      # L0 layer.
      params = model_helper.CreateBidirectionalRNNParams(
          self.params, forward_lstm, backward_lstm)
      params.name = 'L0'
      rnn_layers_params.append(params)

      # The latter layers are all uni-directional lstm.
      input_size = 2 * p.lstm_cell_size
      for i in range(1, p.num_lstm_layers):
        # Forward lstm cell.
        if p.lstm_tpl_uni is None:
          cell = p.lstm_tpl.Copy()
        else:
          cell = p.lstm_tpl_uni.Copy()
        cell.name = 'L%d_rnn' % i
        cell.num_input_nodes = input_size
        cell.num_output_nodes = p.lstm_cell_size
        # Forward lstm layer.
        params = model_helper.CreateUnidirectionalRNNParams(self.params, cell)
        params.name = 'L%d' % i
        rnn_layers_params.append(params)
        input_size = p.lstm_cell_size

      self.CreateChildren('rnn', rnn_layers_params)

      dropout_p = layers.DropoutLayer.Params().Set(
          name='dropout_layer',
          keep_prob=1.0 - p.dropout_prob,
          random_seed=p.random_seed + 84828474 if p.random_seed else None)
      self.CreateChild('dropout', dropout_p)

  def ApplyClipping(self, theta, x):
    p = self.params
    if not p.cc_schedule:
      return x
    cap = tf.cast(self.cc_schedule.GetState(theta.cc_schedule), x.dtype)
    return tf.clip_by_value(x, -cap, cap)

  def FProp(self, theta, input_batch):
    """Encodes source as represented by `inputs` and `paddings`.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: A `.NestedMap` with fields:
        - ids: The inputs tensor. It is expected to be of shape [batch, time].
        - paddings: The paddings tensor. Expected shape [batch, time].

    Returns:
      A NestedMap containing:
        - encoded: The encoded features, a tensor of shape [time, batch, depth]
        - padding: of shape [time, batch]
        - segment_id: [time, batch] if packed inputs are supported by the model
            (and all layers), or None otherwise.
    """
    p = self.params
    src_segment_id = None
    with tf.name_scope(p.name):
      # Now the rnn layers.
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(tf.shape(input_batch.ids), [-1, -1]),
          py_utils.assert_shape_match(
              tf.shape(input_batch.ids), tf.shape(input_batch.paddings))
      ], tf.transpose(input_batch.ids))
      paddings = tf.expand_dims(tf.transpose(input_batch.paddings), 2)
      xs = self.emb.EmbLookup(theta.emb, inputs)
      xs = self.ApplyClipping(theta, xs)
      self._emb_out = xs
      ps = paddings
      # When cc_schedule is specified, make sure lstm_tpl is QuantizedLSTMCell
      # with the same cc_schedule so that the RNN layer output is within
      # clipping range.
      xs = self.rnn[0].FProp(theta.rnn[0], xs, ps)
      xs = self.dropout.FProp(theta.dropout, xs)
      for i in range(1, p.num_lstm_layers):
        layer = self.rnn[i]
        ys, _ = layer.FProp(theta.rnn[i], xs, ps)
        ys = self.dropout.FProp(theta.dropout, ys)
        if hasattr(layer.params, 'cell'):
          layer_params = layer.params.cell
        else:
          layer_params = layer.params
        if layer_params.num_input_nodes == layer_params.num_output_nodes:
          xs += ys  # Residual skip
          xs = self.ApplyClipping(theta, xs)
        else:
          # When cc_schedule is specified, make sure lstm_tpl is
          # QuantizedLSTMCell with the same cc_schedule so that the RNN layer
          # output is within clipping range.
          xs = ys
      return py_utils.NestedMap(
          encoded=xs, padding=tf.squeeze(ps, [2]), segment_id=src_segment_id)


class MTEncoderUniRNN(base_encoder.BaseEncoder):
  """MT encoder that consists of a stack of uni-directional RNN layers."""

  @classmethod
  def Params(cls):
    """Configs for `MTEncoderUniRNN`."""
    p = super(MTEncoderUniRNN, cls).Params()
    p.Define('emb', layers.EmbeddingLayer.Params(), 'Embedding layer params.')
    p.Define('lstm_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('lstm_cell_size', 512, 'LSTM cell size for the RNN layer.')
    p.Define('num_lstm_layers', 8, 'Number of rnn layers to create')
    p.Define('dropout_prob', 0.0, 'Prob at which we do dropout.')
    p.Define('residual_start', 2,
             'Layer at which we start residual connections.')
    p.Define(
        'unidi_rnn_type', 'func', 'Options: func, native_cudnn. '
        'func: FRNN, native_cudnn: CuDNNLSTM.')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')

    p.Define('is_transparent', False,
             'If set, outputs a merger of layer outputs.')
    p.Define(
        'transparent_merger_tpl',
        layers.WeightedSumLayer.Params().Set(add_weight_summaries=True),
        'Merger op for layer outputs.')

    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    default_params_init = py_utils.WeightInit.Uniform(0.04)

    # Default config for the embedding.
    p.emb.vn = disable_vn
    p.emb.vocab_size = 32000
    p.emb.embedding_dim = 1024
    p.emb.max_num_shards = 16
    p.emb.params_init = default_params_init

    p.lstm_tpl.vn = disable_vn
    p.lstm_tpl.params_init = default_params_init
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTEncoderUniRNN, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed inputs are not yet supported for '
                                'MTEncoderUniRNN.')

    with tf.variable_scope(p.name):
      if p.cc_schedule is None:
        self.cc_schedule = None
      else:
        self.CreateChild('cc_schedule', p.cc_schedule)

      self.CreateChild('emb', p.emb)

      rnn_layers_params = []

      num_input_nodes = p.emb.embedding_dim
      for i in range(p.num_lstm_layers):
        cell = p.lstm_tpl.Copy()
        cell.name = 'L%d_rnn' % i
        cell.num_input_nodes = num_input_nodes
        cell.num_output_nodes = p.lstm_cell_size
        params = model_helper.CreateUnidirectionalRNNParams(self.params, cell)
        params.name = 'L%d' % i
        rnn_layers_params.append(params)
        num_input_nodes = cell.num_output_nodes

      self.CreateChildren('rnn', rnn_layers_params)

      dropout_p = layers.DropoutLayer.Params().Set(
          name='dropout_layer',
          keep_prob=1.0 - p.dropout_prob,
          random_seed=p.random_seed + 827366448 if p.random_seed else None)
      self.CreateChild('dropout', dropout_p)

      if p.is_transparent:
        transparent_params = p.transparent_merger_tpl.Copy()
        transparent_params.name = 'transparent'
        transparent_params.num_sources = p.num_lstm_layers
        self.CreateChild('transparent_merger', transparent_params)

  def ApplyClipping(self, theta, x):
    if self.cc_schedule:
      return self.cc_schedule.ApplyClipping(theta.cc_schedule, x)
    else:
      return x

  def FProp(self, theta, input_batch):
    p = self.params
    src_segment_id = None
    with tf.name_scope(p.name):
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(tf.shape(input_batch.ids), [-1, -1]),
          py_utils.assert_shape_match(
              tf.shape(input_batch.ids), tf.shape(input_batch.paddings))
      ], tf.transpose(input_batch.ids))
      paddings = tf.expand_dims(tf.transpose(input_batch.paddings), 2)
      xs = self.emb.EmbLookup(theta.emb, inputs)
      xs = self.ApplyClipping(theta, xs)
      summary_utils.histogram('input_emb', xs)
      xs = self.dropout.FProp(theta.dropout, xs)
      ps = paddings
      # Now the rnn layers.
      outputs_list = []
      for i in range(0, p.num_lstm_layers):
        layer = self.rnn[i]
        ys, _ = layer.FProp(theta.rnn[i], xs, ps)
        ys = self.dropout.FProp(theta.dropout, ys)
        if i >= p.residual_start:
          xs += ys  # Residual skip
          xs = self.ApplyClipping(theta, xs)
        else:
          xs = ys
        outputs_list.append(xs)
        summary_utils.histogram('layer_out_%s' % i, xs)

      if p.is_transparent:
        xs = self.transparent_merger.FProp(theta.transparent_merger,
                                           outputs_list)

      return py_utils.NestedMap(
          encoded=xs, padding=tf.squeeze(ps, [2]), segment_id=src_segment_id)


class MTEncoderBiRNN(base_encoder.BaseEncoder):
  """MT encoder that consists of a stack of bi-directional RNN layers."""

  @classmethod
  def Params(cls):
    """Configs for `MTEncoderBiRNN`."""
    p = super(MTEncoderBiRNN, cls).Params()
    p.Define('emb', layers.EmbeddingLayer.Params(), 'Embedding layer params.')
    p.Define('lstm_tpl',
             rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('proj_tpl', layers.ProjectionLayer.Params(),
             'Configs template for the projection layer.')
    p.Define('lstm_cell_size', 512, 'LSTM cell size for the RNN layer.')
    p.Define('num_lstm_layers', 8, 'Number of rnn layers to create')
    p.Define('dropout_prob', 0.0, 'Prob at which we do dropout.')
    p.Define('residual_start', 2,
             'Layer at which we start residual connections.')
    p.Define('encoder_out_dim', 1024, 'Depth of the encoder output.')
    p.Define(
        'bidi_rnn_type', 'func', 'Options: func, native_cudnn. '
        'func: BidirectionalFRNN, '
        ' native_cudnn: BidirectionalNativeCuDNNLSTM.')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')

    p.Define('is_transparent', False,
             'If set, outputs a merger of layer outputs.')
    p.Define(
        'transparent_merger_tpl',
        layers.WeightedSumLayer.Params().Set(add_weight_summaries=True),
        'Merger op for layer outputs.')

    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    default_params_init = py_utils.WeightInit.Uniform(0.04)

    # Default config for the embedding.
    p.emb.vn = disable_vn
    p.emb.vocab_size = 32000
    p.emb.embedding_dim = 1024
    p.emb.max_num_shards = 16
    p.emb.params_init = default_params_init

    p.lstm_tpl.vn = disable_vn
    p.lstm_tpl.params_init = default_params_init
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTEncoderBiRNN, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      if p.cc_schedule is None:
        self.cc_schedule = None
      else:
        self.CreateChild('cc_schedule', p.cc_schedule)

      self.CreateChild('emb', p.emb)

      rnn_layers_params = []

      for i in range(p.num_lstm_layers):
        params = p.lstm_tpl.Copy()
        params.name = 'L%d_rnn_fwd' % i
        if i == 0:
          params.num_input_nodes = p.emb.embedding_dim
        else:
          params.num_input_nodes = 2 * p.lstm_cell_size
        params.num_output_nodes = p.lstm_cell_size
        params.reset_cell_state = p.packed_input
        forward_lstm = params

        params = params.Copy()
        params.name = 'L%d_rnn_bak' % i
        params.reset_cell_state = p.packed_input
        backward_lstm = params

        params = model_helper.CreateBidirectionalRNNParams(
            self.params, forward_lstm, backward_lstm)
        params.packed_input = p.packed_input
        params.name = 'L%d' % i
        rnn_layers_params.append(params)

      self.CreateChildren('rnn', rnn_layers_params)

      if p.lstm_cell_size * 2 != p.encoder_out_dim:
        # Project the encoder output to the desired dim.
        proj_p = p.proj_tpl.Copy().Set(
            name='proj',
            batch_norm=False,
            input_dim=p.lstm_cell_size * 2,
            output_dim=p.encoder_out_dim)
        if p.cc_schedule is not None:
          proj_p.has_bias = False
          proj_p.activation = 'TANH'
        else:
          proj_p.has_bias = True
          proj_p.activation = 'NONE'
        self.CreateChild('final_proj', proj_p)

      dropout_p = layers.DropoutLayer.Params().Set(
          name='dropout_layer',
          keep_prob=1.0 - p.dropout_prob,
          random_seed=p.random_seed + 827366448 if p.random_seed else None)
      self.CreateChild('dropout', dropout_p)

      if p.is_transparent:
        transparent_params = p.transparent_merger_tpl.Copy()
        transparent_params.name = 'transparent'
        transparent_params.num_sources = p.num_lstm_layers
        self.CreateChild('transparent_merger', transparent_params)

  def ApplyClipping(self, theta, x):
    if self.cc_schedule:
      return self.cc_schedule.ApplyClipping(theta.cc_schedule, x)
    else:
      return x

  def FProp(self, theta, input_batch):
    p = self.params
    with tf.name_scope(p.name):
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(tf.shape(input_batch.ids), [-1, -1]),
          py_utils.assert_shape_match(
              tf.shape(input_batch.ids), tf.shape(input_batch.paddings))
      ], tf.transpose(input_batch.ids))
      paddings = tf.expand_dims(tf.transpose(input_batch.paddings), 2)
      if p.packed_input:
        src_segment_id = tf.expand_dims(
            tf.transpose(input_batch.segment_ids), 2)
      else:
        src_segment_id = None
      xs = self.emb.EmbLookup(theta.emb, inputs)
      xs = self.ApplyClipping(theta, xs)
      summary_utils.histogram('input_emb', xs)
      xs = self.dropout.FProp(theta.dropout, xs)
      ps = paddings
      # Now the rnn layers.
      outputs_list = []
      for i in range(0, p.num_lstm_layers):
        layer = self.rnn[i]
        ys = layer.FProp(theta.rnn[i], xs, ps, segment_id=src_segment_id)
        ys = self.dropout.FProp(theta.dropout, ys)
        if i >= p.residual_start:
          xs += ys  # Residual skip
          xs = self.ApplyClipping(theta, xs)
        else:
          xs = ys
        outputs_list.append(xs)
        summary_utils.histogram('layer_out_%s' % i, xs)

      if p.is_transparent:
        xs = self.transparent_merger.FProp(theta.transparent_merger,
                                           outputs_list)

      if p.lstm_cell_size * 2 != p.encoder_out_dim:
        # Project to the right depth.
        xs = self.final_proj.FProp(theta.final_proj, xs, ps)
        summary_utils.histogram('final_proj_out', xs)

      if src_segment_id is not None:
        src_segment_id = tf.squeeze(src_segment_id, [2])

      return py_utils.NestedMap(
          encoded=xs, padding=tf.squeeze(ps, [2]), segment_id=src_segment_id)


class TransformerEncoder(base_encoder.BaseEncoder):
  """Transformer stack with sinusoidal positional embeddings and attention.

  Implements the encoder of 'Attention is All You Need':
  https://arxiv.org/abs/1706.03762.
  """

  @classmethod
  def Params(cls):
    """Configs for `TransformerEncoder`."""
    p = super(TransformerEncoder, cls).Params()

    # Embedding related
    p.Define('token_emb',
             layers.EmbeddingLayer.Params().Set(
                 vocab_size=32000,
                 embedding_dim=1024,
                 max_num_shards=16,
                 params_init=py_utils.WeightInit.Gaussian(
                     1.0 / math.sqrt(1024)),
                 scale_sqrt_depth=True), 'Embedding layer params.')

    # Positional embedding related
    p.Define(
        'position_emb',
        layers.PositionalEmbeddingLayer.Params().Set(embedding_dim=1024),
        'Positional Embedding layer params.')

    p.Define('model_dim', 1024, 'Characteristic depth (dimension).')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')

    p.Define('transformer_stack', mt_layers.TransformerStack.Params(),
             'TransformerStack layer params.')

    p.transformer_stack.num_transformer_layers = 6
    p.transformer_stack.transformer_tpl.tr_atten_tpl.num_attention_heads = 8
    p.transformer_stack.transformer_tpl.tr_fflayer_tpl.hidden_dim = 8192
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerEncoder, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      assert p.token_emb.embedding_dim == p.position_emb.embedding_dim
      p.transformer_stack.Set(
          model_dim=p.model_dim, packed_input=p.packed_input)
      if p.model_dim != p.token_emb.embedding_dim:
        tf.logging.warning('token_emb.embedding_dim != model_dim (%s vs. %s), '
                           'creating a projection!')
        proj_p = layers.ProjectionLayer.Params().Copy()
        proj_p.name = 'emb_proj'
        proj_p.input_dim = p.token_emb.embedding_dim
        proj_p.output_dim = p.model_dim
        self.CreateChild('emb_proj', proj_p)

      # Token embeddings
      p.token_emb.dtype = p.dtype
      self.CreateChild('token_emb', p.token_emb)

      # Positional embeddings
      self.CreateChild('position_emb', p.position_emb)

      dropout_tpl = layers.DropoutLayer.Params()
      dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
      self.CreateChild('input_dropout', dropout_tpl)

    p.transformer_stack.name = p.name
    self.CreateChild('transformer_stack', p.transformer_stack)

  def FProp(self, theta, input_batch):
    """Embeds source ids and transforms with TransformerStack.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      input_batch: A `.NestedMap` with fields:

        - ids: The inputs tensor. It is expected to be of shape [batch, time].
        - paddings: The paddings tensor. Expected shape [batch, time].

    Returns:
      A NestedMap containing:
        - encoded: The encoded features, either a tensor of shape [time, batch,
            depth], or a list of tensors if is_transparent is set in
            transformer_stack.
        - padding: of shape [time, batch]
        - segment_id: [time, batch] if packed inputs are supported by the model
            (and all layers), or None otherwise.
    """
    p = self.params
    with tf.name_scope(p.name):
      src_segment_id = None
      src_segment_pos = None
      input_ids = py_utils.with_dependencies([
          py_utils.assert_shape_match(
              tf.shape(input_batch.ids), tf.shape(input_batch.paddings)),
          py_utils.assert_equal(tf.rank(input_batch.ids), 2)
      ], input_batch.ids)

      if (not py_utils.use_tpu() and
          tf.flags.FLAGS.transformer_encoder_truncates_inputs):
        max_seq_length = tf.cast(
            tf.reduce_max(tf.reduce_sum(1.0 - input_batch.paddings, 1)),
            tf.int32)
        paddings = py_utils.with_dependencies([
            py_utils.assert_equal(
                tf.constant(True, tf.bool),
                tf.reduce_all(input_batch.paddings[:, max_seq_length:] > 0.5))
        ], input_batch.paddings)
        input_ids = input_ids[:, :max_seq_length]
        paddings = paddings[:, :max_seq_length]
        if p.packed_input:
          src_segment_id = input_batch.segment_ids[:, :max_seq_length]
          src_segment_pos = input_batch.segment_pos[:, :max_seq_length]
      else:
        paddings = input_batch.paddings
        if p.packed_input:
          src_segment_id = input_batch.segment_ids
          src_segment_pos = input_batch.segment_pos

      max_time = tf.shape(input_ids)[1]

      # Input token embeddings + positional embeddings
      input_embs = self.token_emb.EmbLookup(theta.token_emb,
                                            tf.reshape(input_ids, [-1]))
      input_embs = tf.reshape(input_embs,
                              [-1, max_time, p.token_emb.embedding_dim])
      if p.packed_input:
        position_embs = self.position_emb.FPropWithPosition(
            theta.position_emb, src_segment_pos)
      else:
        position_embs = self.position_emb.FProp(theta.position_emb, max_time)
        position_embs = tf.reshape(position_embs,
                                   [1, max_time, p.token_emb.embedding_dim])
      input_embs += position_embs

      if p.model_dim != p.token_emb.embedding_dim:
        input_embs = self.emb_proj.FProp(theta.emb_proj, input_embs)

      paddings = tf.transpose(paddings)
      if p.packed_input:
        src_segment_id = tf.transpose(src_segment_id)
      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

      # [time, batch, dim]
      transformer_input = tf.transpose(input_embs, [1, 0, 2])

    encoded, padding, segment_id = self.transformer_stack.FProp(
        theta.transformer_stack, transformer_input, paddings, src_segment_id)
    return py_utils.NestedMap(
        encoded=encoded, padding=padding, segment_id=segment_id)
