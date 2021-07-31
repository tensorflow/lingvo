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
"""Encoders for the machine translation model.
"""

import math
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import batch_major_attention
from lingvo.core import layers
from lingvo.core import model_helper
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import summary_utils
from lingvo.tasks.mt import layers as mt_layers

tf.flags.DEFINE_bool('transformer_encoder_truncates_inputs', False,
                     'Whether TransformerEncoder truncates inputs to max len.')
FLAGS = tf.flags.FLAGS


class MTEncoderV1(base_layer.BaseLayer):
  """Machine translation encoder version 1."""

  @classmethod
  def Params(cls):
    """Configs for `MTEncoderV1`."""
    p = super().Params()
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
    p.Define('unidi_rnn_type', 'func', 'Options: func. ' 'func: FRNN.')
    p.Define('bidi_rnn_type', 'func', 'Options: func. '
             'func: BidirectionalFRNN. ')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')
    p.Define(
        'packed_input', False, 'If True, encoder and all layers support '
        'multiple examples in a single sequence.')

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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed inputs are not yet supported for '
                                'MTEncoderV1.')

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
    params = model_helper.CreateBidirectionalRNNParams(self.params,
                                                       forward_lstm,
                                                       backward_lstm)
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


class MTEncoderUniRNN(base_layer.BaseLayer):
  """MT encoder that consists of a stack of uni-directional RNN layers."""

  @classmethod
  def Params(cls):
    """Configs for `MTEncoderUniRNN`."""
    p = super().Params()
    p.Define('emb', layers.EmbeddingLayer.Params(), 'Embedding layer params.')
    p.Define('lstm_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('lstm_cell_size', 512, 'LSTM cell size for the RNN layer.')
    p.Define('num_lstm_layers', 8, 'Number of rnn layers to create')
    p.Define('dropout_prob', 0.0, 'Prob at which we do dropout.')
    p.Define('residual_start', 2,
             'Layer at which we start residual connections.')
    p.Define('unidi_rnn_type', 'func', 'Options: func. ' 'func: FRNN.')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')

    p.Define('is_transparent', False,
             'If set, outputs a merger of layer outputs.')
    p.Define(
        'transparent_merger_tpl',
        layers.WeightedSumLayer.Params().Set(add_weight_summaries=True),
        'Merger op for layer outputs.')
    p.Define(
        'packed_input', False, 'If True, encoder and all layers support '
        'multiple examples in a single sequence.')

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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed inputs are not yet supported for '
                                'MTEncoderUniRNN.')

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

  def zero_state(self, theta, batch_size):
    return py_utils.NestedMap(rnn=[
        self.rnn[i].zero_state(theta.rnn[i], batch_size)
        for i in range(len(self.rnn))
    ])

  def FProp(self, theta, input_batch, state0=None):
    p = self.params
    src_segment_id = None
    with tf.name_scope(p.name):
      # Reshape to [t, b]
      inputs = py_utils.with_dependencies([
          py_utils.assert_shape_match(tf.shape(input_batch.ids), [-1, -1]),
          py_utils.assert_shape_match(
              tf.shape(input_batch.ids), tf.shape(input_batch.paddings))
      ], tf.transpose(input_batch.ids))
      paddings = tf.expand_dims(tf.transpose(input_batch.paddings), 2)

      # Setup streaming states.
      if not state0:
        state0 = self.zero_state(theta, tf.shape(inputs)[1])
      state1 = py_utils.NestedMap(rnn=[None] * p.num_lstm_layers)

      xs = self.emb.EmbLookup(theta.emb, inputs)
      xs = self.ApplyClipping(theta, xs)
      summary_utils.histogram('input_emb', xs)
      xs = self.dropout.FProp(theta.dropout, xs)
      ps = paddings
      # Now the rnn layers.
      outputs_list = []
      for i in range(0, p.num_lstm_layers):
        layer = self.rnn[i]
        ys, state1.rnn[i] = layer.FProp(
            theta.rnn[i], xs, ps, state0=state0.rnn[i])
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
          encoded=xs,
          padding=tf.squeeze(ps, [2]),
          segment_id=src_segment_id,
          state=state1)


class MTEncoderBiRNN(base_layer.BaseLayer):
  """MT encoder that consists of a stack of bi-directional RNN layers."""

  @classmethod
  def Params(cls):
    """Configs for `MTEncoderBiRNN`."""
    p = super().Params()
    p.Define('emb', layers.EmbeddingLayer.Params(), 'Embedding layer params.')
    p.Define('shared_emb', None, 'Embedding shared with decoder.')
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
    p.Define('bidi_rnn_type', 'func', 'Options: func. '
             'func: BidirectionalFRNN. ')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')

    p.Define('is_transparent', False,
             'If set, outputs a merger of layer outputs.')
    p.Define(
        'transparent_merger_tpl',
        layers.WeightedSumLayer.Params().Set(add_weight_summaries=True),
        'Merger op for layer outputs.')
    p.Define(
        'packed_input', False, 'If True, encoder and all layers support '
        'multiple examples in a single sequence.')

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

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if p.cc_schedule is None:
      self.cc_schedule = None
    else:
      self.CreateChild('cc_schedule', p.cc_schedule)

    if p.shared_emb:
      # Naming this 'softmax' to match the name of the same component in the
      # decoder. Variable names need to be the same in order to be reused.
      self.CreateChild('softmax', p.shared_emb)
    else:
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

  def _ComputeInputs(self, theta, xs, input_batch):
    if self.params.shared_emb:
      xs = self.softmax.EmbLookup(theta.softmax, xs)
    else:
      xs = self.emb.EmbLookup(theta.emb, xs)
    xs = self.ApplyClipping(theta, xs)
    xs = self.dropout.FProp(theta.dropout, xs)
    return xs

  def _CreateChildrenVariables(self):
    if self.params.shared_emb:
      with tf.variable_scope('shared_emb', reuse=tf.AUTO_REUSE):
        self.softmax.InstantiateVariables()
    super()._CreateChildrenVariables()

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
      xs = self._ComputeInputs(theta, inputs, input_batch)
      summary_utils.histogram('input_emb', xs)
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


class MTEncoderBiRNNPrecomputedEmbedding(MTEncoderBiRNN):
  """A variant of MTEncoderBiRNN where the RNN input is consumed directly...

  instead of looked up in an embedding layer from one-hot vectors.
  """

  def _ComputeInputs(self, theta, xs, input_batch):
    return input_batch.embeddings


class TransformerEncoder(base_layer.BaseLayer):
  """Transformer stack with sinusoidal positional embeddings and attention.

  Implements the encoder of 'Attention is All You Need':
  https://arxiv.org/abs/1706.03762.
  """

  @classmethod
  def Params(cls):
    """Configs for `TransformerEncoder`."""
    p = super().Params()

    # Embedding related
    p.Define('token_emb',
             layers.EmbeddingLayer.Params().Set(
                 vocab_size=32000,
                 embedding_dim=1024,
                 max_num_shards=16,
                 params_init=py_utils.WeightInit.Gaussian(
                     1.0 / math.sqrt(1024)),
                 scale_sqrt_depth=True), 'Embedding layer params.')

    p.Define('shared_emb', None, 'Embedding shared with Decoder.')

    # Positional embedding related
    p.Define(
        'position_emb',
        layers.PositionalEmbeddingLayer.Params().Set(embedding_dim=1024),
        'Positional Embedding layer params.')

    # TODO(miachen): Extend this to more general logic of adding multiple
    # embedding fields.
    # Task embedding related
    p.Define('task_emb', None, 'Task embedding layer params.')

    p.Define('model_dim', 1024, 'Characteristic depth (dimension).')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')

    p.Define('transformer_stack', mt_layers.TransformerStack.Params(),
             'TransformerStack layer params.')
    p.Define(
        'packed_input', False, 'If True, encoder and all layers support '
        'multiple examples in a single sequence.')

    p.Define(
        'emb_projection_tpl', layers.ProjectionLayer.Params(),
        'Template for embedding projection layer. The token embeddings '
        'are projected to match the `model_dim`, if they are different.')

    # MASS pretraining related (https://github.com/microsoft/MASS).
    p.Define(
        'apply_source_mask', False, 'If True, apply source mask '
        '(corresponding to those masked words) to encoder states.')
    p.Define(
        'source_mask_id', 0, 'Id for masked words in source inputs. '
        'Only needed when p.apply_source_mask is True.')
    p.Define('ln_input', None, 'Whether to use input ln')

    p.transformer_stack.num_transformer_layers = 6
    p.transformer_stack.transformer_tpl.tr_atten_tpl.num_attention_heads = 8
    p.transformer_stack.transformer_tpl.tr_fflayer_tpl.hidden_dim = 8192

    p.emb_projection_tpl.batch_norm = True
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if p.shared_emb:
      # Naming this 'softmax' to match the name of the same component in the
      # decoder. Variable names need to be the same in order to be reused.
      self.CreateChild('softmax', p.shared_emb)

    assert p.token_emb.embedding_dim == p.position_emb.embedding_dim
    p.transformer_stack.Set(model_dim=p.model_dim, packed_input=p.packed_input)
    if p.model_dim != p.token_emb.embedding_dim:
      tf.logging.warning(
          f'token_emb.embedding_dim != model_dim ({p.token_emb.embedding_dim} '
          f'vs. {p.model_dim}), creating a projection!')
      proj_p = p.emb_projection_tpl.Copy()
      proj_p.name = 'emb_proj'
      proj_p.input_dim = p.token_emb.embedding_dim
      proj_p.output_dim = p.model_dim
      self.CreateChild('emb_proj', proj_p)

    # Token embeddings
    if not p.shared_emb:
      p.token_emb.dtype = p.dtype
      self.CreateChild('token_emb', p.token_emb)

    # Positional embeddings
    self.CreateChild('position_emb', p.position_emb)

    # Task embeddings.
    if p.task_emb:
      assert p.task_emb.embedding_dim == p.token_emb.embedding_dim
      self.CreateChild('task_emb', p.task_emb)

    dropout_tpl = layers.DropoutLayer.Params()
    dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
    self.CreateChild('input_dropout', dropout_tpl)

    p.transformer_stack.name = p.name
    self.CreateChild('transformer_stack', p.transformer_stack)

    if p.ln_input:
      params = p.transformer_stack.ln_tpl.Copy()
      params.name = 'enc_ln_input'
      params.input_dim = p.model_dim
      self.CreateChild('layer_norm_input', params)

  def _CreateChildrenVariables(self):
    if self.params.shared_emb:
      with tf.variable_scope('shared_emb', reuse=tf.AUTO_REUSE):
        self.softmax.InstantiateVariables()
    self.transformer_stack.InstantiateVariables()
    super()._CreateChildrenVariables()

  def FProp(self, theta, input_batch):
    """Embeds source ids and transforms with TransformerStack.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      input_batch: A `.NestedMap` with fields:

        - ids: The inputs tensor. It is expected to be of shape [batch, time].
        - paddings: The paddings tensor. Expected shape [batch, time].
        - task_ids: If p.task_emb is provided, must contain per-token task
            ids of shape [batch, time].

    Returns:
      A NestedMap containing

      - encoded: The encoded features, either a tensor of shape
        [time, batch, depth], or a list of tensors if is_transparent is set in
        transformer_stack.
      - padding: of shape [time, batch]
      - segment_id: [time, batch] if packed inputs are supported by the model
        (and all layers), or None otherwise.
      - embedded_inputs: [time, batch, depth] embedded inputs tokens without
        positional encodings.
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
          FLAGS.transformer_encoder_truncates_inputs):
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
      if not p.shared_emb:
        input_embs = self.token_emb.EmbLookup(theta.token_emb,
                                              tf.reshape(input_ids, [-1]))
      else:
        input_embs = self.softmax.EmbLookup(theta.softmax,
                                            tf.reshape(input_ids, [-1]))

      input_embs = tf.reshape(input_embs,
                              [-1, max_time, p.token_emb.embedding_dim])
      # [time, batch, dim]
      orig_input_embs = tf.transpose(input_embs, [1, 0, 2])

      if p.packed_input:
        position_embs = self.position_emb.FPropWithPosition(
            theta.position_emb, src_segment_pos)
      else:
        position_embs = self.position_emb.FProp(theta.position_emb, max_time)
        position_embs = tf.reshape(position_embs,
                                   [1, max_time, p.token_emb.embedding_dim])
      input_embs += position_embs

      if p.ln_input:
        input_embs = self.layer_norm_input.FProp(theta.layer_norm_input,
                                                 input_embs)

      if p.task_emb:
        input_embs += self.task_emb.EmbLookup(theta.task_emb,
                                              input_batch.task_ids)

      summary_utils.histogram('input_embs', input_embs)
      if p.model_dim != p.token_emb.embedding_dim:
        input_embs = self.emb_proj.FProp(theta.emb_proj, input_embs)
        summary_utils.histogram('emb_proj', input_embs)

      paddings = tf.cast(tf.transpose(paddings), py_utils.FPropDtype(p))
      if p.packed_input:
        src_segment_id = tf.transpose(src_segment_id)
      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

      # [time, batch, dim]
      transformer_input = tf.transpose(input_embs, [1, 0, 2])

    if not self.do_eval and p.apply_source_mask:
      # Augment padding for masked source word positions.
      dtype = paddings.dtype
      source_mask = tf.where(
          tf.equal(input_ids, p.source_mask_id),
          tf.ones_like(input_ids, dtype=dtype),
          tf.zeros_like(input_ids, dtype=dtype))
      # Make sure padding is between 0 and 1.
      paddings = tf.clip_by_value(paddings + tf.transpose(source_mask), 0.0,
                                  1.0)

    encoded, padding, segment_id = self.transformer_stack.FProp(
        theta.transformer_stack, transformer_input, paddings, src_segment_id)
    return py_utils.NestedMap(
        encoded=encoded,
        padding=padding,
        segment_id=segment_id,
        embedded_inputs=orig_input_embs)

  def FPropFullSequence(self, theta, ids, paddings):
    return self.FProp(theta, py_utils.NestedMap(ids=ids,
                                                paddings=paddings))['encoded']


class TransformerBatchMajorEncoder(base_layer.BaseLayer):
  """Transformer encoder with batch major implementation.

  This encoder first applies dropout to the input embeddings,
  then returns encoded output produced by p.transformer_stack using
  self_attention_layer builder.

  Example definition for a stack of 6 transformer layers:

  builder_params = self_attention_layer.Builder.Params().Set(
      model_dim=model_dim,
      ff_hidden_dim=ff_hidden_dim,
      num_heads=num_heads,
      selfatten_add_unnormalized_input=False,
      selfatten_enable_value_proj=True)

  p.transformer_stack = builder_params.Instantiate().TransformerStack(
      'transformer_stack', 6)

  Implements the encoder of 'Attention is All You Need':
  https://arxiv.org/abs/1706.03762.
  """

  @classmethod
  def Params(cls):
    p = super().Params()

    # Default config for the token embedding.
    p.Define(
        'token_emb',
        layers.EmbeddingLayer.Params().Set(
            vocab_size=32000,
            embedding_dim=1024,
            max_num_shards=16,
            params_init=py_utils.WeightInit.Gaussian(1.0 / math.sqrt(1024)),
            scale_sqrt_depth=True), 'Embedding layer params.')

    p.Define('shared_emb', None, 'Embedding shared with Decoder.')

    # Default config for the position embedding.
    p.Define('position_emb',
             layers.PositionalEmbeddingLayer.Params().Set(embedding_dim=1024),
             'Positional Embedding layer params.')

    p.Define('model_dim', 1024, 'Characteristic depth (dimension).')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define('input_dropout_tpl', layers.DropoutLayer.Params(),
             'Input dropout layer params.')
    p.Define('transformer_stack', None, 'TransformerStack layer params.')
    p.Define(
        'packed_input', False, 'If True, encoder and all layers support '
        'multiple examples in a single sequence.')
    p.Define('final_layer_norm', False,
             'Whether or not to apply the final layer normalization.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm.')
    p.Define(
        'output_data_format', 'TBC', 'The data format of output features: '
        'TBC for [time, batch, feature_dim], '
        'BTC for [batch, time, feature_dim].')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    assert p.output_data_format in ('TBC', 'BTC')

    if p.shared_emb:
      self.CreateChild('softmax', p.shared_emb)

    p.token_emb.dtype = p.dtype
    if not p.shared_emb:
      self.CreateChild('token_emb', p.token_emb)
    self.CreateChild('position_emb', p.position_emb)

    dropout_tpl = p.input_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
    self.CreateChild('input_dropout', dropout_tpl)

    if p.transformer_stack:
      self.CreateChild('transformer_stack', p.transformer_stack)

    if p.final_layer_norm:
      layer_norm_p = layers.LayerNorm.Params().Set(
          name='final_ln',
          input_dim=p.model_dim,
          use_fused_layernorm=p.use_fused_layernorm,
          fprop_dtype=p.input_dropout_tpl.fprop_dtype)
      self.CreateChild('final_ln', layer_norm_p)

  def _CreateChildrenVariables(self):
    if self.params.shared_emb:
      with tf.variable_scope('shared_emb', reuse=tf.AUTO_REUSE):
        self.softmax.InstantiateVariables()
    super()._CreateChildrenVariables()

  def FProp(self, theta, input_batch):
    """Embeds source ids and transforms with TransformerStack.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: A `.NestedMap` object containing: ids - The inputs tensor of
        shape [batch, time]. paddings - The ids' paddings of shape [batch,
        time].

    Returns:
      A '.NestedMap' object containing:
        encoded - The encoded features of shape [time, batch, dim] or [batch,
          time, dim], depending p.output_data_format.
        padding - The encoded features' padding of shape [time, batch] or
          [batch, time].
        segment_id - The segmentation of packed inputs of shape [time, batch] or
          [batch, time] if it is supported by the model, or None otherwise.
        embedded_inputs - The embedded inputs tokens without positional
          encodings of shape [time, batch, dim] or [batch, time, dim].
    """

    p = self.params
    with tf.name_scope(p.name):
      # [batch, time]
      input_ids = input_batch.ids
      # [batch, time]
      paddings = input_batch.paddings

      # [batch, time]
      segment_ids = input_batch.segment_ids if p.packed_input else None

      batch = py_utils.GetShape(input_ids)[0]
      time = py_utils.GetShape(input_ids)[1]

      # Embedding layer.
      # [batch, time, dim]
      if not p.shared_emb:
        input_embs = self.token_emb.EmbLookup(theta.token_emb, input_ids)
      else:
        input_embs = self.softmax.EmbLookup(theta.softmax, input_ids)
      orig_input_embs = input_embs

      # [1, time, dim]
      if p.packed_input:
        positions = input_batch.segment_pos
        position_embs = tf.expand_dims(
            self.position_emb.FPropWithPosition(theta.position_emb, positions),
            0)
      else:
        position_embs = tf.expand_dims(
            self.position_emb.FProp(theta.position_emb, time), 0)

      # [batch, time, dim]
      input_embs += position_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        paddings = tf.cast(paddings, p.input_dropout_tpl.fprop_dtype)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      # [batch, time, dim]
      transformer_input = input_embs
      # Explicitly set the input shape of Transformer layers, to avoid
      # unknown shape error occurred to tf.einsum on nonTPU devices.
      transformer_input = tf.reshape(transformer_input,
                                     [batch, time, p.model_dim])

      # Compute self-attention segment mask once.
      if p.packed_input:
        segment_mask = batch_major_attention.SegmentMask(
            segment_ids, segment_ids, dtype=transformer_input.dtype)
      else:
        segment_mask = tf.zeros([batch, 1, time, time])

      shape = py_utils.GetShape(transformer_input)
      batch_size = shape[0]
      seq_len = shape[1]
      paddings = tf.reshape(paddings, [batch_size, seq_len])
      encoded, padding = self.transformer_stack.FProp(theta.transformer_stack,
                                                      transformer_input,
                                                      paddings, segment_mask)

      if p.final_layer_norm:
        encoded = self.final_ln.FProp(theta.final_ln, encoded)

      seq_lengths = tf.cast(tf.reduce_sum(1. - padding, axis=1), tf.int32)

      if p.output_data_format == 'TBC':
        encoded = tf.transpose(encoded, [1, 0, 2])  # [time, batch, dim]
        padding = tf.transpose(padding)  # [time, batch]
        segment_ids = tf.transpose(segment_ids) if p.packed_input else None
        orig_input_embs = tf.transpose(orig_input_embs, [1, 0, 2])

      return py_utils.NestedMap(
          encoded=encoded,
          padding=padding,
          seq_lengths=seq_lengths,  # used by beam_search_helper.
          segment_id=segment_ids,
          embedded_inputs=orig_input_embs)


class TransformerXEncoder(TransformerEncoder):
  """Transformer Encoder to Interpolate two Sentences.

  This encoder can be used to combine input embeddings of two sentencs with a
  pre-defined interpolation vector (lambdas in FProp).
  """

  def FProp(self, theta, input_batch, interpolation_batch=None, lambdas=None):
    # pyformat: disable
    """Interpolates source ids in input_batch and interpolation_batch.

    Refer to Eq. (4) in paper https://arxiv.org/abs/2106.04060.
    It is a standard Transformer Encoder if interpolation_batch != None.

    Args:
      theta: A `.NestedMap` object containing weights values of this layer and
        its children layers.
      input_batch: A `.NestedMap` with fields:

        - ids: The inputs tensor. It is expected to be of shape [batch, time].
        - paddings: The paddings tensor. Expected shape [batch, time].
        - task_ids: If p.task_emb is provided, must contain per-token task ids
          of shape [batch, time].
      interpolation_batch: A `.NestedMap` with fields:

        - ids: The inputs tensor. It is expected to be of shape [batch, time].
        - paddings: The paddings tensor. Expected shape [batch, time].
        - task_ids: If p.task_emb is provided, must contain per-token task ids
          of shape [batch, time].
        - embs: Embeddings of ids.
      lambdas: A pair of tensors to combine embeddings of ids in input_batch and
        interpolation_batch.

    Returns:
      A NestedMap of

        - encoded: The encoded features, either a tensor of shape
          [time, batch, depth], or a list of tensors if is_transparent is set in
          transformer_stack.
        - padding: of shape [time, batch]
        - segment_id: [time, batch] if packed inputs are supported by the model
          (and all layers), or None otherwise.
        - embedded_inputs: [time, batch, depth] embedded inputs tokens without
          positional encodings.
    """
    # pyformat: enable

    p = self.params
    with tf.name_scope(p.name):
      src_segment_id = None
      src_segment_pos = None
      input_ids = py_utils.with_dependencies([
          py_utils.assert_shape_match(
              tf.shape(input_batch.ids), tf.shape(input_batch.paddings)),
          py_utils.assert_equal(tf.rank(input_batch.ids), 2)
      ], input_batch.ids)

      max_seq_length = None
      if (not py_utils.use_tpu() and
          FLAGS.transformer_encoder_truncates_inputs):
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
      if not p.shared_emb:
        input_embs = self.token_emb.EmbLookup(theta.token_emb,
                                              tf.reshape(input_ids, [-1]))
      else:
        input_embs = self.softmax.EmbLookup(theta.softmax,
                                            tf.reshape(input_ids, [-1]))

      if interpolation_batch is not None:
        other_input_ids = interpolation_batch.ids
        if not p.shared_emb:
          other_input_embs = self.token_emb.EmbLookup(
              theta.token_emb, tf.reshape(other_input_ids, [-1]))
        else:
          other_input_embs = self.softmax.EmbLookup(
              theta.softmax, tf.reshape(other_input_ids, [-1]))
        lambdas = [tf.expand_dims(a, -1) for a in lambdas]
        if 'embs' in input_batch and input_batch.embs is not None:
          input_embs = input_batch.embs
        if 'embs' in interpolation_batch and interpolation_batch.embs is not None:
          other_input_embs = interpolation_batch.embs
        else:
          input_embs = tf.reshape(
              input_embs,
              [-1, tf.shape(input_ids)[1], p.token_emb.embedding_dim])
          other_input_embs = tf.reshape(
              other_input_embs,
              [-1, tf.shape(other_input_ids)[1], p.token_emb.embedding_dim])
        input_embs = lambdas[0] * input_embs + lambdas[1] * other_input_embs
        paddings = paddings + interpolation_batch.paddings - 1.0
        paddings = tf.clip_by_value(paddings, 0.0, 1.0)

      input_embs = tf.reshape(input_embs,
                              [-1, max_time, p.token_emb.embedding_dim])

      orig_input_embs = input_embs
      if p.task_emb:
        if interpolation_batch is None:
          input_embs += self.task_emb.EmbLookup(theta.task_emb,
                                                input_batch.task_ids)
        else:
          task_embs = self.task_emb.EmbLookup(theta.task_emb,
                                              input_batch.task_ids)
          other_task_embs = self.task_emb.EmbLookup(
              theta.task_emb, interpolation_batch.task_ids)
          task_embs = lambdas[0] * task_embs + lambdas[1] * other_task_embs
          input_embs += task_embs

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

      paddings = tf.cast(tf.transpose(paddings), py_utils.FPropDtype(p))
      if p.packed_input:
        src_segment_id = tf.transpose(src_segment_id)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

      # [time, batch, dim]
      transformer_input = tf.transpose(input_embs, [1, 0, 2])

    if not self.do_eval and p.apply_source_mask:
      # Augment padding for masked source word positions.
      dtype = paddings.dtype
      source_mask = tf.where(
          tf.equal(input_ids, p.source_mask_id),
          tf.ones_like(input_ids, dtype=dtype),
          tf.zeros_like(input_ids, dtype=dtype))
      # Make sure padding is between 0 and 1.
      paddings = tf.clip_by_value(paddings + tf.transpose(source_mask), 0.0,
                                  1.0)

    encoded, padding, segment_id = self.transformer_stack.FProp(
        theta.transformer_stack, transformer_input, paddings, src_segment_id)

    return py_utils.NestedMap(
        encoded=encoded,
        padding=padding,
        segment_id=segment_id,
        embedded_inputs=orig_input_embs)
