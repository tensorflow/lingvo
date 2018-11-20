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
"""Machine translation decoder.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.python.framework import function

from lingvo.core import attention
from lingvo.core import base_decoder
from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import model_helper
from lingvo.core import plot
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers
from lingvo.core import summary_utils


@function.Defun()
def AssertIdShape(expected_ids_shape_pattern, ids_shape, *args):
  dependencies = [
      py_utils.assert_shape_match(ids_shape, expected_ids_shape_pattern)
  ] + [py_utils.assert_shape_match(ids_shape, x_shape) for x_shape in args]
  return py_utils.with_dependencies(dependencies, ids_shape)


class MTBaseDecoder(base_decoder.BaseBeamSearchDecoder):
  """Base class for Lingvo MT decoders."""

  @classmethod
  def Params(cls):
    p = super(MTBaseDecoder, cls).Params()
    p.Define('label_smoothing', None, 'Label smoothing class.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(), 'Softmax params.')
    p.Define(
        'per_word_avg_loss', False, 'Compute loss averaged per word. If False '
        'loss is computed averaged per sequence.')
    p.Define('unidi_rnn_type', 'func', 'Options: func, native_cudnn. '
             'func: FRNN, native_cudnn: CuDNNLSTM.')
    p.Define('feed_attention_context_vec_to_softmax', False,
             'Whether to concatenate attention context vector to rnn output'
             ' before softmax.')

    # Default config for the softmax part.
    p.softmax.num_classes = 32000  # 32k
    p.softmax.num_shards = 8

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTBaseDecoder, self).__init__(params)
    p = self.params
    if p.label_smoothing is not None:
      p.label_smoothing.name = 'smoother'
      p.label_smoothing.num_classes = p.softmax.num_classes
      self.CreateChild('smoother', p.label_smoothing)

  def _FPropSoftmax(self,
                    theta,
                    softmax_input,
                    target_labels,
                    target_weights,
                    target_paddings,
                    target_segment_ids=None):
    """Computes cross-entropy loss given the softmax input, labels and weights.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      softmax_input: A tensor of shape [time, batch, p.softmax.input_dim].
      target_labels: A matrix of tf.int32. [time, batch].
      target_weights: A matrix of params.dtype. [time, batch].
      target_paddings: A matrix of params.dtype. [time, batch].
      target_segment_ids: A matrix of params.dtype. [time, batch].

    Returns:
      A dictionary containing metrics for the xent loss and prediction accuracy.
    """
    p = self.params
    softmax_input = tf.reshape(softmax_input, [-1, p.softmax.input_dim])

    if p.label_smoothing is None:
      xent_loss = self.softmax.FProp(
          theta.softmax, [softmax_input],
          class_weights=tf.reshape(target_weights, [-1, 1]),
          class_ids=tf.reshape(target_labels, [-1, 1]))
    else:
      # [time, batch, num_classes]
      target_probs = tf.transpose(
          self.smoother.FProp(theta.smoother, tf.transpose(target_paddings),
                              tf.transpose(target_labels),
                              target_ids=None), [1, 0, 2])
      xent_loss = self.softmax.FProp(
          theta.softmax, [softmax_input],
          class_weights=tf.reshape(target_weights, [-1, 1]),
          class_probabilities=tf.reshape(target_probs,
                                         [-1, p.softmax.num_classes]))

    if p.per_word_avg_loss:
      final_loss = tf.identity(xent_loss.avg_xent, name='loss')
      loss_weight = tf.identity(xent_loss.total_weight, name='num_predictions')
    else:
      # NOTE: Per-sequence loss is the sum of each example's loss.  The
      # final loss for a training batch is the mean loss of sequences in
      # the batch.
      # [time, batch]
      per_example_loss = tf.reshape(xent_loss.per_example_xent,
                                    py_utils.GetShape(target_weights))
      per_sequence_loss = tf.reduce_sum(per_example_loss * target_weights, 0)
      if p.packed_input:
        assert target_segment_ids is not None, (
            'Need target segment ids for '
            'normalizing loss when training with packed inputs.')
        num_samples = tf.cast(
            tf.reduce_sum(
                tf.reduce_max(target_segment_ids, 0) -
                tf.reduce_min(target_segment_ids, 0) + 1),
            dtype=per_sequence_loss.dtype)
        final_loss = tf.reduce_sum(per_sequence_loss) / num_samples
      else:
        final_loss = tf.reduce_mean(per_sequence_loss)
      loss_weight = py_utils.GetShape(per_sequence_loss)[0]

    ret_dict = {
        'loss': (final_loss, loss_weight),
        'log_pplx': (xent_loss.avg_xent, xent_loss.total_weight)
    }

    # NOTE: tf.argmax is not implemented for the JF backend, see b/36093673
    # Skip the fraction_of_correct_next_step_preds during training.
    if p.is_eval:
      logits = xent_loss.logits
      correct_preds = tf.cast(
          tf.equal(
              tf.cast(tf.reshape(tf.argmax(logits, 1), [-1]), tf.int32),
              tf.reshape(target_labels, [-1])), p.dtype)
      correct_next_preds = tf.reduce_sum(
          correct_preds * tf.reshape(target_weights, [-1]))
      num_preds = tf.reduce_sum(target_weights)
      accuracy = tf.identity(
          correct_next_preds / num_preds,
          name='fraction_of_correct_next_step_preds')
      ret_dict['fraction_of_correct_next_step_preds'] = (accuracy, num_preds)
    return ret_dict

  def ComputeLoss(self, theta, predictions, targets):
    segment_id = None
    if self.params.packed_input:
      segment_id = tf.transpose(targets.segment_ids)
    return self._FPropSoftmax(theta, predictions, tf.transpose(targets.labels),
                              tf.transpose(targets.weights),
                              tf.transpose(targets.paddings), segment_id)

  def _TruncateTargetSequence(self, targets):
    """Truncate padded time steps from all sequences."""
    # The following tensors are all in the [batch, time] shape.
    p = self.params
    # Let's make a copy of targets.
    targets = targets.Pack(targets.Flatten())
    target_ids = targets.ids
    target_labels = targets.labels
    target_weights = targets.weights
    target_paddings = targets.paddings
    max_seq_length = tf.to_int32(
        tf.reduce_max(tf.reduce_sum(1.0 - target_paddings, 1)))
    summary_utils.scalar(p, 'max_seq_length', max_seq_length)
    # Assert to make sure after max_seq_length, all are padded steps for all
    # sequences.
    target_paddings = py_utils.with_dependencies([
        py_utils.assert_equal(
            tf.constant(True, tf.bool),
            tf.reduce_all(target_paddings[:, max_seq_length:] > 0.5))
    ], target_paddings)
    target_ids = py_utils.with_dependencies([
        AssertIdShape(
            py_utils.GetShape(target_ids), py_utils.GetShape(target_labels),
            py_utils.GetShape(target_paddings),
            py_utils.GetShape(target_weights))
    ], target_ids)
    targets.ids = target_ids[:, :max_seq_length]
    targets.labels = target_labels[:, :max_seq_length]
    targets.weights = target_weights[:, :max_seq_length]
    targets.paddings = target_paddings[:, :max_seq_length]
    return targets

  def _AddAttenProbsSummary(self, source_paddings, targets, atten_probs):
    """Add image summary of attention probs.

    Args:
      source_paddings: source padding, of shape [src_len, src_batch].
      targets: A dict of string to tensors representing the targets one try to
          predict. Each tensor in targets is of shape [tgt_batch, tgt_len].
      atten_probs: a list of attention probs, each element is of shape
          [tgt_len, tgt_batch, src_len].
    """
    if not self.cluster.add_summary:
      return

    num_rows = len(atten_probs)
    fig = plot.MatplotlibFigureSummary(
        'decoder_example',
        figsize=(6, 3 * num_rows),
        max_outputs=1,
        subplot_grid_shape=(num_rows, 1))

    def PlotAttention(fig, axes, cur_atten_probs, title, set_x_label):
      plot.AddImage(fig, axes, cur_atten_probs, title=title)
      axes.set_ylabel(plot.ToUnicode('Output sequence index'), wrap=True)
      if set_x_label:
        axes.set_xlabel(plot.ToUnicode('Input sequence index'), wrap=True)

    index = 0
    srclen = tf.cast(tf.reduce_sum(1 - source_paddings[:, index]), tf.int32)
    tgtlen = tf.cast(tf.reduce_sum(1 - targets.paddings[index, :]), tf.int32)

    for i, probs in enumerate(atten_probs):
      # Extract first entry in batch of attention prob matrices
      # [tgt_len, src_len]
      probs = probs[:, index, :]
      probs = tf.expand_dims(probs[:tgtlen, :srclen], 0)
      fig.AddSubplot(
          [probs],
          PlotAttention,
          title='atten_probs_%d' % i,
          set_x_label=(i == len(atten_probs) - 1))
    fig.Finalize()


class MTDecoderV1(MTBaseDecoder, quant_utils.QuantizableLayer):
  """MT decoder v1."""

  @classmethod
  def Params(cls):
    p = super(MTDecoderV1, cls).Params()
    # Shared embedding.
    p.Define('emb', layers.EmbeddingLayer.Params(), 'Embedding layer params.')
    p.Define('source_dim', 1024, 'Dimension of the source encoding.')
    p.Define('attention', attention.AdditiveAttention.Params(),
             'Additive attention params.')
    p.Define('atten_rnn_cell_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Attention RNNCell params template.')
    p.Define('rnn_cell_tpl', rnn_cell.LSTMCellSimple.Params(),
             'RNNCell params template.')
    p.Define('rnn_cell_dim', 1024, 'size of the rnn cells.')
    p.Define('rnn_layers', 8, 'Number of rnn layers.')
    p.Define('residual_start', 2, 'Start residual connections from this layer.')
    p.Define('atten_rnn_cls', rnn_layers.FRNNWithAttention,
             'Which atten rnn cls to use.')
    p.Define('use_prev_atten_ctx', False,
             'If True, all decoder layers use previous attention context as '
             'input. Otherwise, only first decoder layer uses previous '
             'attention context and the rest of the layers use current '
             'attention context.')
    p.Define('dropout_prob', 0.0, 'Prob at which we do dropout.')
    # Default value was mildly tuned. Could be further tuned in the future.
    p.Define('qlogsoftmax_range_min', -10.0, 'Quantization of the output of '
             'log softmax.')
    p.Define(
        'use_zero_atten_state', False, 'To use zero attention state '
        'instead of computing attention with zero query vector.')

    p.Define('cc_schedule', None, 'Clipping cap schedule.')

    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    default_params_init = py_utils.WeightInit.Uniform(0.04)

    # Default config for the embedding.
    p.emb.vn = disable_vn
    p.emb.vocab_size = 32000
    p.emb.embedding_dim = 1024
    p.emb.max_num_shards = 16
    p.emb.params_init = default_params_init

    # Default config for the attention model.
    p.attention.vn = disable_vn
    p.attention.hidden_dim = 1024
    p.attention.params_init = None  # Filled in after dims are known.
    # Default config for the attention rnn cell.
    p.atten_rnn_cell_tpl.vn = disable_vn
    p.atten_rnn_cell_tpl.params_init = default_params_init
    # Default config for the rnn cell.
    p.rnn_cell_tpl.vn = disable_vn
    p.rnn_cell_tpl.params_init = default_params_init
    # Default config for the softmax part.
    p.softmax.vn = disable_vn
    p.softmax.num_classes = 32000  # 32k
    p.softmax.num_shards = 16
    p.softmax.params_init = default_params_init

    # Default config for beam search.
    p.target_seq_len = 300
    p.beam_search.length_normalization = 0.2
    p.beam_search.coverage_penalty = 0.2

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTDecoderV1, self).__init__(params)
    p = self.params
    assert p.emb.vocab_size == p.softmax.num_classes

    with tf.variable_scope(p.name):
      if p.cc_schedule is None:
        self.cc_schedule = None
      else:
        self.CreateChild('cc_schedule', p.cc_schedule)

      if py_utils.use_tpu():
        emb_device = self.cluster.WorkerDeviceInModelSplit(0)
      else:
        emb_device = ''
      with tf.device(emb_device):
        self.CreateChild('emb', p.emb)

        p.attention.dtype = p.dtype
        p.attention.source_dim = p.source_dim
        p.attention.query_dim = p.rnn_cell_dim
        p.attention.packed_input = p.packed_input
        if p.attention.params_init is None:
          p.attention.params_init = py_utils.WeightInit.Gaussian(
              1. / math.sqrt(p.attention.source_dim + p.attention.query_dim))
        atten_params = p.attention.Copy()

        params = p.atten_rnn_cell_tpl.Copy()
        params.name = 'atten_rnn'
        params.dtype = p.dtype
        params.reset_cell_state = p.packed_input
        params.num_input_nodes = p.emb.embedding_dim + p.attention.source_dim
        params.num_output_nodes = p.rnn_cell_dim
        atten_rnn_cell = params.Copy()

        params = p.atten_rnn_cls.Params()
        params.name = 'frnn_with_atten'
        params.dtype = p.dtype
        params.cell = atten_rnn_cell
        params.attention = atten_params
        params.output_prev_atten_ctx = p.use_prev_atten_ctx
        params.packed_input = p.packed_input
        params.use_zero_atten_state = p.use_zero_atten_state
        params.atten_context_dim = p.attention.source_dim
        self.CreateChild('frnn_with_atten', params)

        # TODO(zhifengc): Avoid this?
        self._rnn_attn = self.frnn_with_atten.rnn_cell
        self._atten = self.frnn_with_atten.attention

        rnn_layers_params = []
        for i in range(1, p.rnn_layers):
          params = p.rnn_cell_tpl.Copy()
          params.name = 'rnn%d' % i
          params.dtype = p.dtype
          params.num_input_nodes = p.rnn_cell_dim + p.attention.source_dim
          params.num_output_nodes = p.rnn_cell_dim
          params.reset_cell_state = p.packed_input
          rnn_cell_p = params

          params = model_helper.CreateUnidirectionalRNNParams(
              self.params, rnn_cell_p)
          params.name = 'frnn%d' % i
          params.packed_input = p.packed_input
          rnn_layers_params.append(params)

        self.CreateChildren('frnn', rnn_layers_params)

      p.softmax.dtype = p.dtype
      if p.feed_attention_context_vec_to_softmax:
        p.softmax.input_dim = p.rnn_cell_dim + p.attention.source_dim
      else:
        p.softmax.input_dim = p.rnn_cell_dim
      self.CreateChild('softmax', p.softmax)

  def ApplyDropout(self, x_in):
    p = self.params
    assert 0 <= p.dropout_prob and p.dropout_prob < 1.0
    if p.is_eval or p.dropout_prob == 0.0:
      return x_in
    else:
      return tf.nn.dropout(x_in, 1.0 - p.dropout_prob)

  def ApplyClipping(self, theta, x):
    if self.cc_schedule:
      return self.cc_schedule.ApplyClipping(theta.cc_schedule, x)
    else:
      return x

  @py_utils.NameScopeDecorator('MTDecoderV1/ComputePredictions')
  def ComputePredictions(self, theta, source_encs, source_paddings, targets,
                         src_segment_id):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_encs: source encoding, of shape [time, batch, depth].
      source_paddings: source encoding's padding, of shape [time, batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, time].
      src_segment_id: source segment id, of shape [time, batch].

    Returns:
      A Tensor with shape [time, batch, params.softmax.input_dim].
    """
    p = self.params
    time, batch = py_utils.GetShape(source_paddings, 2)
    source_encs = py_utils.HasShape(source_encs, [time, batch, p.source_dim])
    with tf.name_scope(p.name):
      target_ids = tf.transpose(targets.ids)
      target_paddings = py_utils.HasRank(targets.paddings, 2)
      target_paddings = tf.expand_dims(tf.transpose(target_paddings), 2)
      if p.packed_input:
        target_segment_id = tf.expand_dims(tf.transpose(targets.segment_ids), 2)
      else:
        target_segment_id = tf.zeros_like(target_paddings)

      if py_utils.use_tpu():
        emb_device = self.cluster.WorkerDeviceInModelSplit(0)
      else:
        emb_device = ''
      with tf.device(emb_device):
        inputs = self.emb.EmbLookup(theta.emb, target_ids)
        inputs = self.ApplyClipping(theta, inputs)
        summary_utils.histogram(p, 'input_emb', inputs)
        inputs = self.ApplyDropout(inputs)
        self._emb_out = inputs

        # Layer 0 interwines with attention.
        (atten_ctxs, xs, atten_probs, _) = self.frnn_with_atten.FProp(
            theta.frnn_with_atten,
            source_encs,
            source_paddings,
            inputs,
            target_paddings,
            src_segment_id=src_segment_id,
            segment_id=target_segment_id)
        self._AddAttenProbsSummary(source_paddings, targets, [atten_probs])

        atten_ctxs = self.ApplyClipping(theta, atten_ctxs)
        summary_utils.histogram(p, 'atten_ctxs', atten_ctxs)

        for i, (layer, layer_theta) in enumerate(zip(self.frnn, theta.frnn)):
          # Forward through Layer-(i + 1) because Layer-0 handled before.
          ys, _ = layer.FProp(
              layer_theta,
              tf.concat([xs, atten_ctxs], 2),
              target_paddings,
              segment_id=target_segment_id)
          ys = self.ApplyDropout(ys)
          if 1 + i >= p.residual_start:
            xs += ys  # Residual skip
            xs = self.ApplyClipping(theta, xs)
          else:
            xs = ys
          summary_utils.histogram(p, 'layer_out_%s' % i, xs)

        if p.feed_attention_context_vec_to_softmax:
          xs = tf.concat([xs, atten_ctxs], 2)

        return xs

  @py_utils.NameScopeDecorator('MTDecoderV1/InitDecoder')
  def _InitDecoder(self, theta, source_encs, source_paddings, num_hyps):
    """Returns initial decoder states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
          its children layers.
      source_encs: source encoding, of shape [time, batch, depth].
      source_paddings: source encoding's padding, of shape [time, batch].
      num_hyps: Scalar Tensor of type int, Number of hypothesis maintained in
          beam search, equal to beam_size * num_hyps_per_beam.

    Returns:
      Tuple of initial model states.
    """
    p = self.params
    rnn_states = [self._rnn_attn.zero_state(num_hyps)]
    for layer in self.frnn:
      rnn_states.append(layer.rnn_cell.zero_state(num_hyps))

    if p.use_zero_atten_state:
      self._atten.InitForSourcePacked(theta.frnn_with_atten.atten, source_encs,
                                      source_encs, source_paddings)
      s_seq_len = tf.shape(source_encs)[0]
      atten_context = tf.zeros(
          [num_hyps, p.attention.source_dim], dtype=source_encs.dtype)
      atten_states = self._atten.ZeroAttentionState(s_seq_len, num_hyps)
      atten_probs = tf.zeros([num_hyps, s_seq_len], dtype=source_encs.dtype)
    else:
      self._atten.InitForSourcePacked(theta.frnn_with_atten.atten, source_encs,
                                      source_encs, source_paddings)

      src_seq_len = tf.shape(source_encs)[0]
      zero_atten_state = self._atten.ZeroAttentionState(src_seq_len, num_hyps)

      (atten_context, atten_probs,
       atten_states) = self._atten.ComputeContextVector(
           theta.frnn_with_atten.atten,
           tf.zeros([num_hyps, p.rnn_cell_dim], dtype=p.dtype),
           attention_state=zero_atten_state)

    assert atten_states is not None
    return rnn_states, atten_context, atten_probs, atten_states

  @py_utils.NameScopeDecorator('MTDecoderV1/DecodeStep')
  def _DecodeStep(self, theta, embs, step_paddings, prev_atten_context,
                  rnn_states, prev_atten_states):
    """Decode one step."""
    p = self.params
    new_rnn_states = []
    new_rnn_states_0, _ = self._rnn_attn.FProp(
        theta.frnn_with_atten.cell, rnn_states[0],
        py_utils.NestedMap(
            act=[tf.concat([embs, prev_atten_context], 1)],
            padding=step_paddings,
            reset_mask=tf.ones_like(step_paddings)))
    new_rnn_states.append(new_rnn_states_0)
    rnn_out = self._rnn_attn.GetOutput(new_rnn_states_0)
    cur_atten_context, atten_probs, atten_states = (
        self._atten.ComputeContextVector(
            theta.frnn_with_atten.atten,
            rnn_out,
            attention_state=prev_atten_states))
    assert atten_states is not None

    if p.use_prev_atten_ctx:
      atten_context = prev_atten_context
    else:
      atten_context = cur_atten_context

    for i, (layer, layer_theta) in enumerate(zip(self.frnn, theta.frnn)):
      new_rnn_states_i, _ = layer.rnn_cell.FProp(
          layer_theta.cell, rnn_states[1 + i],
          py_utils.NestedMap(
              act=[tf.concat([rnn_out, atten_context], 1)],
              padding=step_paddings,
              reset_mask=tf.ones_like(step_paddings)))
      new_rnn_states.append(new_rnn_states_i)
      new_rnn_out = layer.rnn_cell.GetOutput(new_rnn_states_i)
      if 1 + i >= p.residual_start:
        rnn_out += new_rnn_out
        rnn_out = self.ApplyClipping(theta, rnn_out)
      else:
        rnn_out = new_rnn_out
    # Concatenating atten_context vec to rnn output before softmax might help
    if p.feed_attention_context_vec_to_softmax:
      step_out = tf.concat([rnn_out, atten_context], 1)
    else:
      step_out = rnn_out
    return (cur_atten_context, atten_probs, new_rnn_states, step_out,
            atten_states)

  def _GetAttentionInitState(self):
    """Gets the attention initialization state.

    It is valid to call this after `_DecoderInit()`. Inference subclasses use
    this to split computation across subgraph boundaries.

    Returns:
      `.NestedMap` of attention source states.
    """
    return self._atten.GetInitializationSourceState()

  def _SetAttentionInitState(self, new_init_state):
    """Sets the attention initialization state.

    Args:
      new_init_state: `.NestedMap` compatible with that returned from
        `_GetAttentionSourceState`.
    """
    self._atten.SetInitializationSourceState(new_init_state)

  def _InitBeamSearchStateCallback(self,
                                   theta,
                                   source_encs,
                                   source_paddings,
                                   num_hyps_per_beam,
                                   additional_source_info=None):
    """Returns initial beams search states.

    Args:
      source_encs: A tensor of shape [src_len, src_batch, source_dim].
      source_paddings: A tensor of shape [src_len, src_batch].
      num_hyps_per_beam: An int, number hyps to keep for source sentence.
      additional_source_info: a `.NestedMap` of tensors containing extra context
          information about the source that may be useful for decoding.
    Returns:
      A tuple (initial_results, states).
        initial_results: a `.NestedMap` of initial results.
          atten_probs:
            The initial attention probs, of shape [tgt_batch, src_len].
        states: a `.NestedMap` of initial model states.
          rnn_states:
            Initial state of the RNN.
          atten_context:
            Initial attention context vector.
          atten_states:
            Initial attention state.
    """
    # additional_source_info is currently not used.
    del additional_source_info
    num_beams = py_utils.GetShape(source_encs)[1]
    num_hyps = num_beams * num_hyps_per_beam
    rnn_states, init_atten_context, atten_probs, atten_states = (
        self._InitDecoder(theta, source_encs, source_paddings, num_hyps))

    initial_results = py_utils.NestedMap({'atten_probs': atten_probs})

    return initial_results, py_utils.NestedMap({
        'rnn_states': rnn_states,
        'atten_context': init_atten_context,
        'atten_probs': atten_probs,
        'atten_states': atten_states,
    })

  @py_utils.NameScopeDecorator('MTDecoderV1/PreBeamSearchStepCallback')
  def _PreBeamSearchStepCallback(self,
                                 theta,
                                 source_encs,
                                 source_paddings,
                                 step_ids,
                                 states,
                                 num_hyps_per_beam,
                                 additional_source_info=None):
    """Returns logits for sampling ids and the next model states.

    Args:
      source_encs: A tensor of shape [src_len, src_batch, source_dim].
      source_paddings: A tensor of shape [src_len, src_batch].
      step_ids: A tensor of shape [tgt_batch, 1].
      states: A `.NestedMap` of tensors representing states that the clients
          would like to keep track of for each of the active hyps.
      num_hyps_per_beam: Beam size.
      additional_source_info: a `.NestedMap` of tensors containing extra context
          information about the source that may be useful for decoding.
    Returns:
      A tuple (results, out_states).
      results: A `.NestedMap` of beam search results.
        atten_probs:
          The updated attention probs, of shape [tgt_batch, src_len].
        log_probs:
          Log prob for each of the tokens in the target vocab. This is of shape
          [tgt_batch, vocab_size].
      out_states: A `.NestedMap`. The updated states.
        rnn_states:
          Last state of the RNN.
        atten_context:
          Updated attention context vector.
        atten_states:
          Updates attention states.
    """
    p = self.params
    # additional_source_info is currently not used.
    del additional_source_info

    prev_rnn_states = states['rnn_states']
    prev_atten_context = states['atten_context']
    prev_atten_probs = states['atten_probs']
    prev_atten_states = states['atten_states']
    step_paddings = tf.zeros(py_utils.GetShape(step_ids), dtype=p.dtype)
    embs = self.emb.EmbLookup(theta.emb, tf.reshape(step_ids, [-1]))
    embs = self.ApplyClipping(theta, embs)
    atten_context, atten_probs, rnn_states, step_out, atten_states = (
        self._DecodeStep(theta, embs, step_paddings, prev_atten_context,
                         prev_rnn_states, prev_atten_states))
    atten_probs = tf.reshape(atten_probs, tf.shape(prev_atten_probs))

    logits = self.softmax.Logits(theta.softmax, [step_out])
    log_probs = self.fns.qlogsoftmax(
        logits, qmin=p.qlogsoftmax_range_min, qmax=0.0)

    if p.use_prev_atten_ctx:
      cur_atten_probs = prev_atten_probs
    else:
      cur_atten_probs = atten_probs

    bs_results = py_utils.NestedMap({
        'atten_probs': cur_atten_probs,  # the probs exposed to beam search
        'log_probs': log_probs,
    })
    new_states = py_utils.NestedMap({
        'rnn_states': rnn_states,
        'atten_context': atten_context,
        'atten_probs': atten_probs,  # the updated attention probs
        'atten_states': atten_states,
    })

    return bs_results, new_states

  def _PostBeamSearchStepCallback(self,
                                  theta,
                                  source_encs,
                                  source_paddings,
                                  new_step_ids,
                                  states,
                                  additional_source_info=None):
    # There is nothing to do here.
    return states

  def BeamSearchDecode(self,
                       source_encs,
                       source_paddings,
                       num_hyps_per_beam_override=0,
                       additional_source_info=None):
    """Performs beam-search based decoding.

    Args:
      source_encs: source encoding, of shape [time, batch, depth].
      source_paddings: source encoding's padding, of shape [time, batch].
      num_hyps_per_beam_override: If set to a value <= 0, this parameter is
        ignored. If set to a value > 0, then this value will be used to
        override `p.num_hyps_per_beam`.
      additional_source_info: a `.NestedMap` of tensors containing extra context
          information about the source that may be useful for decoding.

    Returns:
      BeamSearchDecodeOutput, a namedtuple containing the decode results.
    """
    del additional_source_info  # Unused.
    return self.beam_search.BeamSearchDecode(
        self.theta, source_encs, source_paddings, num_hyps_per_beam_override,
        self._InitBeamSearchStateCallback, self._PreBeamSearchStepCallback,
        self._PostBeamSearchStepCallback)


class TransformerDecoder(MTBaseDecoder):
  """Transformer decoder.

  Implements the decoder of Transformer model:
  https://arxiv.org/abs/1706.03762.
  """

  @classmethod
  def Params(cls):
    p = super(TransformerDecoder, cls).Params()
    p.Define('token_emb', layers.EmbeddingLayer.Params(),
             'Token embedding layer params.')
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define('source_dim', 1024, 'Dimension of encoder outputs.')
    p.Define('model_dim', 1024, 'Model dimension that applies to embedding '
             'layers and all Transformer layers.')
    p.Define('num_trans_layers', 6, 'Number of Transformer layers.')
    p.Define('trans_tpl', layers_with_attention.TransformerLayer.Params(),
             'Transformer layer params.')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define(
        'is_transparent', False, 'If set, expects a tensor of shape '
        '[time, batch, source_dim, num_trans_layers] as source encodings.')

    # Default config for the token embedding.
    p.token_emb.vocab_size = 32000
    p.token_emb.embedding_dim = p.model_dim
    p.token_emb.max_num_shards = 16
    p.token_emb.params_init = py_utils.WeightInit.Gaussian(
        1.0 / math.sqrt(p.token_emb.embedding_dim))
    p.token_emb.scale_sqrt_depth = True

    # Default config for the position embedding.
    p.position_emb.embedding_dim = p.model_dim

    # Default config for the transformer layers.
    p.trans_tpl.source_dim = p.model_dim
    p.trans_tpl.is_decoder = True
    p.trans_tpl.tr_atten_tpl.source_dim = p.model_dim
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 8
    p.trans_tpl.tr_fflayer_tpl.input_dim = p.model_dim
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 2048

    # Default config for beam search.
    p.target_seq_len = 300
    p.beam_search.length_normalization = 0.5
    p.beam_search.coverage_penalty = 0.0

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerDecoder, self).__init__(params)
    p = self.params
    assert p.token_emb.vocab_size == p.softmax.num_classes
    assert p.token_emb.embedding_dim == p.position_emb.embedding_dim
    if p.model_dim != p.token_emb.embedding_dim:
      tf.logging.warning('token_emb.embedding_dim != model_dim (%s vs. %s), '
                         'creating a projection!')
      proj_p = layers.ProjectionLayer.Params().Copy()
      proj_p.name = 'emb_proj'
      proj_p.input_dim = p.token_emb.embedding_dim
      proj_p.output_dim = p.model_dim
      self.CreateChild('emb_proj', proj_p)

    with tf.variable_scope(p.name):
      self._token_emb = self.CreateChild('token_emb', p.token_emb)
      self.CreateChild('position_emb', p.position_emb)

      dropout_tpl = layers.DropoutLayer.Params()
      dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
      self.CreateChild('input_dropout', dropout_tpl)

      params_trans_layers = []
      for i in range(p.num_trans_layers):
        params = p.trans_tpl.Copy()
        params.name = 'trans_layer_%d' % i
        params.packed_input = p.packed_input
        params_trans_layers.append(params)
      self.CreateChildren('trans', params_trans_layers)

      p.softmax.input_dim = p.model_dim
      self.CreateChild('softmax', p.softmax)

  def _FProp(self, theta, source_encs, source_paddings, targets,
             src_segment_id):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_encs: source encoding. When `p.is_transparent` is False, it is a
        tensor of shape [time, batch, depth]. When `p.is_transparent` is True,
        it is a tensor of shape [time, batch, depth, num_trans_layers] if
        `p.is_eval` is True, and a list of `num_trans_layers` tensors of shape
        [time, batch, depth] if `p.is_eval` is False.
      source_paddings: source encoding's padding, of shape [time, batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, time].
      src_segment_id: source segment id, of shape [time, batch].

    Returns:
      Output of last decoder layer, [target_time, target_batch, source_dim].
    """
    p = self.params
    time, batch = py_utils.GetShape(source_paddings, 2)
    if p.is_transparent:
      if p.is_eval:
        source_encs = py_utils.HasShape(
            source_encs, [time, batch, p.source_dim, p.num_trans_layers])
        source_encs = tf.unstack(source_encs, axis=3)
      else:
        assert isinstance(source_encs, list)
        assert len(source_encs) == p.num_trans_layers
        for i in range(p.num_trans_layers):
          source_encs[i] = py_utils.HasShape(source_encs[i],
                                             [time, batch, p.source_dim])
    else:
      source_encs = py_utils.HasShape(source_encs, [time, batch, p.source_dim])
      source_encs = [source_encs] * p.num_trans_layers
    with tf.name_scope(p.name):
      # [batch, time]
      target_ids = targets.ids
      # [time, batch]
      target_paddings = tf.transpose(targets.paddings)
      target_segment_pos = None
      target_segment_id = None
      if p.packed_input:
        target_segment_id = tf.transpose(targets.segment_ids)
        target_segment_pos = targets.segment_pos
        assert src_segment_id is not None, ('Need to provide src_segment_id '
                                            'for packed input.')

      # Embedding layer
      # [batch, time, model_dim]
      token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      target_time = py_utils.GetShape(target_ids)[1]
      # [1, time, model_dim]
      if p.packed_input:
        posit_embs = self.position_emb.FPropWithPosition(
            theta.position_emb, target_segment_pos)
      else:
        posit_embs = tf.expand_dims(
            self.position_emb.FProp(theta.position_emb, target_time), 0)

      # [time, batch, model_dim]
      input_embs = token_embs + posit_embs

      if p.model_dim != p.token_emb.embedding_dim:
        input_embs = self.emb_proj.FProp(theta.emb_proj, input_embs)

      input_embs = tf.transpose(input_embs, [1, 0, 2])
      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

      atten_probs = []
      layer_in = input_embs
      for i, (layer, layer_theta) in enumerate(zip(self.trans, theta.trans)):
        # [time, batch, model_dim]
        layer_out, probs = layer.FProp(
            layer_theta,
            layer_in,
            target_paddings,
            source_encs[i],
            source_paddings,
            source_segment_id=target_segment_id,
            aux_segment_id=src_segment_id)
        layer_in = layer_out
        atten_probs.append(probs)

      self._AddAttenProbsSummary(source_paddings, targets, atten_probs)

      return layer_out

  def ExtendStep(self, theta, source_encs, source_paddings, new_ids,
                 t, prefix_states):
    """Extend prefix as represented by `prefix_states` by one more step.

    This function is expected to be called during fast decoding of Transformer
    models.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_encs: source encoding, of shape [time, batch, depth]. Can be [time,
        bs, depth, num_trans_layers] if is_transparent is set.
      source_paddings: source encoding's padding, of shape [time, batch].
      new_ids: new input ids, of shape [batch].
      t: a scalar, the current time step, 0-based.
      prefix_states: a `.NestedMap` representing the prefix that has already
        been decoded.

    Returns:
      A pair (last_decoder_out, prefix_states), where last_decoder_out is the
      output of the last decoder layer of shape [batch, model_dim], and
      `prefix_states` is the update prefix states.
    """
    p = self.params
    time, batch = py_utils.GetShape(source_paddings, 2)
    if p.is_transparent:
      source_encs = py_utils.HasShape(
          source_encs, [time, batch, p.source_dim, p.num_trans_layers])
      source_encs = tf.unstack(source_encs, axis=3)
    else:
      source_encs = py_utils.HasShape(source_encs, [time, batch, p.source_dim])
      source_encs = [source_encs] * p.num_trans_layers
    with tf.name_scope(p.name):
      # Embedding layer
      # [batch, time, model_dim]
      token_embs = self.token_emb.EmbLookup(theta.token_emb, new_ids)
      # [time, model_dim]
      posit_embs = self.position_emb.FProp(theta.position_emb, t + 1)[-1:, :]
      input_embs = token_embs + posit_embs

      if p.model_dim != p.token_emb.embedding_dim:
        input_embs = self.emb_proj.FProp(theta.emb_proj, input_embs)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      # Make a copy of the input.
      out_prefix_states = prefix_states.Pack(prefix_states.Flatten())

      layer_in = input_embs
      for i, (layer, layer_theta) in enumerate(zip(self.trans, theta.trans)):
        # [time, batch, model_dim]
        layer_prefix_states = prefix_states['layer_%i' % i]
        layer_out, _, updated_prefix_states = layer.ExtendStep(
            layer_theta, layer_in, layer_prefix_states, source_encs[i],
            source_paddings)
        out_prefix_states['layer_%i' % i] = updated_prefix_states
        layer_in = layer_out

      return layer_out, out_prefix_states

  def ComputePredictions(self, theta, source_encs, source_paddings, targets,
                         src_segment_id):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_encs: source encoding, of shape [time, batch, depth]. Can be [time,
        batch, depth, num_layers] if is_transparent is set.
      source_paddings: source encoding's padding, of shape [time, batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, time].
      src_segment_id: source segment id, of shape [time, batch].

    Returns:
      A Tensor with shape [time, batch, params.softmax.input_dim].
    """
    return self._FProp(theta, source_encs, source_paddings, targets,
                       src_segment_id)

  def _InitBeamSearchStateCallback(self,
                                   theta,
                                   source_encs,
                                   source_paddings,
                                   num_hyps_per_beam,
                                   additional_source_info=None):
    """Returns initial beams search states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_encs: A tensor of shape [src_len, src_batch, source_dim].
          Can be [time, batch, depth, num_layers] if is_transparent is set.
      source_paddings: A tensor of shape [src_len, src_batch].
      num_hyps_per_beam: An int, number hyps to keep for source sentence.
      additional_source_info: a `.NestedMap` of tensors containing extra context
          information about the source that may be useful for decoding.
    Returns:
      A tuple (initial_results, states).
        initial_results: a `.NestedMap` of initial results.
          atten_probs:
            The initial attention probs, of shape [tgt_batch, src_len].
        states: a `.NestedMap` of initial model states.
          source_encs:
            A tensor of shape [src_batch, src_len, source_dim].
          source_paddings:
            A tensor of shape [src_batch, src_len].
          target_ids:
            Initial empty list of decoded ids. [num_hyps, 0].
    """
    p = self.params
    # additional_source_info is currently not used.
    del additional_source_info

    num_hyps = py_utils.GetShape(source_encs)[1] * num_hyps_per_beam
    source_len = py_utils.GetShape(source_encs)[0]

    # Dummy attention probs
    atten_probs = tf.ones([num_hyps, source_len]) / tf.to_float(source_len)
    initial_results = py_utils.NestedMap({'atten_probs': atten_probs})

    batch_size = num_hyps
    key_channels = p.model_dim
    value_channels = p.model_dim

    prefix_states = py_utils.NestedMap({
        'layer_%d' % layer: py_utils.NestedMap({
            'key': tf.zeros([batch_size, 0, key_channels]),
            'value': tf.zeros([batch_size, 0, value_channels]),
        })
        for layer in range(p.num_trans_layers)
    })

    return initial_results, py_utils.NestedMap({
        'prefix_states': prefix_states,
        'time_step': 0
    })

  def _PreBeamSearchStepCallback(self,
                                 theta,
                                 source_encs,
                                 source_paddings,
                                 step_ids,
                                 states,
                                 num_hyps_per_beam,
                                 additional_source_info=None):
    """Returns logits for sampling ids and the next model states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_encs: A tensor of shape [src_len, src_batch, source_dim].
          Can be [time, batch, depth, num_layers] if is_transparent is set.
      source_paddings: A tensor of shape [src_len, src_batch].
      step_ids: A tensor of shape [tgt_batch, 1].
      states: A `.NestedMap` of tensors representing states that the clients
          would like to keep track of for each of the active hyps.
      num_hyps_per_beam: Beam size.
      additional_source_info: a `.NestedMap` of tensors containing extra context
          information about the source that may be useful for decoding.
    Returns:
      A tuple (results, out_states).
        results: A `.NestedMap` of beam search results.
          atten_probs:
            The updated attention probs, of shape [tgt_batch, src_len].
          log_probs:
            Log prob for each of the tokens in the target vocab. This is of
            shape [tgt_batch, vocab_size].
        out_states: A `.NestedMap`. The updated states.
           source_encs:
             A tensor of shape [src_batch, src_len, source_dim].
           source_paddings:
             A tensor of shape [src_batch, src_len].
           target_ids:
             Updated list of decoded ids. [num_hyps, Num of decoded ids].
    """
    p = self.params
    # additional_source_info is currently not used.
    del additional_source_info

    target_time = states.time_step
    prefix_states = states.prefix_states

    new_states = states.Pack(states.Flatten())

    layer_out, updated_prefix_states = self.ExtendStep(
        theta, source_encs, source_paddings, tf.squeeze(step_ids, 1),
        target_time, prefix_states)

    new_states.prefix_states = updated_prefix_states
    new_states.time_step = target_time + 1

    softmax_input = tf.reshape(layer_out, [-1, p.softmax.input_dim])
    logits = self.softmax.Logits(theta.softmax, [softmax_input])

    num_hyps = py_utils.GetShape(step_ids)[0]
    source_len = py_utils.GetShape(source_encs)[0]
    # [time * batch, num_classes] -> [time, batch, num_classes]
    logits = tf.reshape(logits, (-1, num_hyps, p.softmax.num_classes))
    # [time, batch, num_classes] -> [batch, time, num_classes]
    logits = tf.transpose(logits, (1, 0, 2))

    # Dummy attention probs
    atten_probs = tf.ones([num_hyps, source_len]) / tf.to_float(source_len)

    # Only return logits for the last ids
    log_probs = tf.nn.log_softmax(tf.squeeze(logits, axis=1))

    bs_results = py_utils.NestedMap({
        'atten_probs': atten_probs,
        'log_probs': log_probs,
    })

    return bs_results, new_states

  def _PostBeamSearchStepCallback(self,
                                  theta,
                                  source_encs,
                                  source_paddings,
                                  new_step_ids,
                                  states,
                                  additional_source_info=None):
    # There is nothing to do here.
    return states

  def BeamSearchDecode(self,
                       source_encs,
                       source_paddings,
                       num_hyps_per_beam_override=0):
    return self.beam_search.BeamSearchDecode(
        self.theta, source_encs, source_paddings, num_hyps_per_beam_override,
        self._InitBeamSearchStateCallback, self._PreBeamSearchStepCallback,
        self._PostBeamSearchStepCallback)
