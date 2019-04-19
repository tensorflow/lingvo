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
    """Populates a metrics dictionary based on the output of ComputePredictions.

    Args:
      theta: Nested map describing decoder model parameters.
      predictions: NestedMap describing the decoding process, requiring:
        .softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
      targets: NestedMap describing the target sequences.

    Returns:
      Two dicts:
      A map from metric name (a python string) to a tuple (value, weight).
      Both value and weight are scalar Tensors.
      A map from name to arbitrary tensors, where the first dimension must be
      the batch index.
    """
    segment_id = None
    if self.params.packed_input:
      segment_id = tf.transpose(targets.segment_ids)
    if isinstance(predictions, py_utils.NestedMap):
      predictions = predictions.softmax_input
    return self._FPropSoftmax(theta, predictions, tf.transpose(targets.labels),
                              tf.transpose(targets.weights),
                              tf.transpose(targets.paddings), segment_id), {}

  def _TruncateTargetSequence(self, targets):
    """Truncate padded time steps from all sequences."""
    # The following tensors are all in the [batch, time] shape.
    # Let's make a copy of targets.
    targets = targets.Pack(targets.Flatten())
    target_ids = targets.ids
    target_labels = targets.labels
    target_weights = targets.weights
    target_paddings = targets.paddings
    max_seq_length = tf.to_int32(
        tf.reduce_max(tf.reduce_sum(1.0 - target_paddings, 1)))
    summary_utils.scalar('max_seq_length', max_seq_length)
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
    """Add summary of attention probs.

    Args:
      source_paddings: source padding, of shape [src_len, src_batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [tgt_batch, tgt_len].
      atten_probs: a list of attention probs, each element is of shape [tgt_len,
        tgt_batch, src_len].
    """
    if not self.cluster.add_summary:
      return

    self._AddAttenProbsImageSummary(source_paddings, targets, atten_probs)
    self._AddAttenProbsHistogramSummary(atten_probs)

  def _AddAttenProbsHistogramSummary(self, atten_probs):
    """Add histogram summary of attention probs.

    Args:
      atten_probs: a list of attention probs, each element is of shape [tgt_len,
        tgt_batch, src_len].
    """
    for i, probs in enumerate(atten_probs):
      # a prefix from the context will be used, which looks like
      # fprop/wmt14_en_de_transformer/tower_0_0/dec/
      summary_utils.histogram('atten{}'.format(i + 1), probs)

  def _AddAttenProbsImageSummary(self, source_paddings, targets, atten_probs):
    """Add image summary of attention probs.

    Args:
      source_paddings: source padding, of shape [src_len, src_batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [tgt_batch, tgt_len].
      atten_probs: a list of attention probs, each element is of shape [tgt_len,
        tgt_batch, src_len].
    """
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
  def ComputePredictions(self, theta, encoder_outputs, targets):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder. Expected to contain:
        encoded - source encoding, of shape [time, batch, depth].
        padding - source encoding's padding, of shape [time, batch].
        segment_id - (optional) source segment id, of shape [time, batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, time].

    Returns:
      A `.NestedMap` containing information about the decoding process. At a
      minimum, this should contain:
        softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
        attention: `.NestedMap` of attention distributions of shape [batch,
                   time, source_len].
        source_enc_len: Lengths of source sentences. Tensor of shape [batch].
    """
    p = self.params
    source_paddings = encoder_outputs.padding
    time, batch = py_utils.GetShape(source_paddings, 2)
    source_encs = py_utils.HasShape(encoder_outputs.encoded,
                                    [time, batch, p.source_dim])
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
        summary_utils.histogram('input_emb', inputs)
        inputs = self.ApplyDropout(inputs)
        self._emb_out = inputs

        # Layer 0 intertwines with attention.
        (accumulated_states, _,
         side_info) = self.frnn_with_atten.AccumulateStates(
             theta.frnn_with_atten,
             source_encs,
             source_paddings,
             inputs,
             target_paddings,
             src_segment_id=getattr(encoder_outputs, 'segment_id', None),
             segment_id=target_segment_id)

        (atten_ctxs, xs, atten_probs) = self.frnn_with_atten.PostProcessStates(
            accumulated_states, side_info)

        self._AddAttenProbsSummary(source_paddings, targets, [atten_probs])

        atten_ctxs = self.ApplyClipping(theta, atten_ctxs)
        summary_utils.histogram('atten_ctxs', atten_ctxs)

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
          summary_utils.histogram('layer_out_%s' % i, xs)

        if p.feed_attention_context_vec_to_softmax:
          xs = tf.concat([xs, atten_ctxs], 2)

        # Get intermediate attention information
        atten_states = accumulated_states.atten_state
        if isinstance(atten_states, py_utils.NestedMap):
          additional_atten_probs = sorted(
              [(name, tensor)
               for name, tensor in atten_states.FlattenItems()
               if name.endswith('probs')])
        else:
          additional_atten_probs = []
        attention_map = py_utils.NestedMap(probs=accumulated_states.atten_probs)
        attention_map.update(additional_atten_probs)

        # Transpose attention probs from [target_length, batch, source_length]
        # to [batch, target_length, source_length]
        def _TransposeAttentions(x):
          return tf.transpose(x, [1, 0, 2])

        attention_map = attention_map.Transform(_TransposeAttentions)
        if isinstance(source_paddings, tf.Tensor):
          source_enc_len = tf.reduce_sum(1 - source_paddings, axis=0)

        return py_utils.NestedMap(
            softmax_input=xs,
            attention=attention_map,
            source_enc_len=source_enc_len)

  @py_utils.NameScopeDecorator('MTDecoderV1/InitDecoder')
  def _InitDecoder(self, theta, encoder_outputs, num_hyps):
    """Returns initial decoder states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
          its children layers.
      encoder_outputs: a NestedMap computed by encoder.
      num_hyps: Scalar Tensor of type int, Number of hypothesis maintained in
          beam search, equal to beam_size * num_hyps_per_beam.

    Returns:
      Tuple of initial model states.
    """
    p = self.params
    source_paddings = encoder_outputs.padding
    time, batch = py_utils.GetShape(source_paddings, 2)
    source_encs = py_utils.HasShape(encoder_outputs.encoded,
                                    [time, batch, p.source_dim])
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

  def _InitBeamSearchStateCallback(self, theta, encoder_outputs,
                                   num_hyps_per_beam):
    """Returns initial beams search states.

    Args:
      theta: a NestedMap of parameters.
      encoder_outputs: a NestedMap computed by encoder.
      num_hyps_per_beam: An int, number hyps to keep for source sentence.

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
    p = self.params
    num_beams = py_utils.GetShape(encoder_outputs.padding)[1]
    num_hyps = num_beams * num_hyps_per_beam
    rnn_states, init_atten_context, atten_probs, atten_states = (
        self._InitDecoder(theta, encoder_outputs, num_hyps))

    initial_results = py_utils.NestedMap(
        log_probs=tf.zeros([num_hyps, p.softmax.num_classes],
                           dtype=py_utils.FPropDtype(p)),
        atten_probs=atten_probs)

    return initial_results, py_utils.NestedMap({
        'rnn_states': rnn_states,
        'atten_context': init_atten_context,
        'atten_probs': atten_probs,
        'atten_states': atten_states,
    })

  @py_utils.NameScopeDecorator('MTDecoderV1/PreBeamSearchStepCallback')
  def _PreBeamSearchStepCallback(self, theta, encoder_outputs, step_ids, states,
                                 num_hyps_per_beam):
    """Returns logits for sampling ids and the next model states.

    Args:
      theta: a NestedMap of parameters.
      encoder_outputs: a NestedMap computed by encoder.
      step_ids: A tensor of shape [tgt_batch, 1].
      states: A `.NestedMap` of tensors representing states that the clients
          would like to keep track of for each of the active hyps.
      num_hyps_per_beam: Beam size.
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

  def _PostBeamSearchStepCallback(self, theta, encoder_outputs, new_step_ids,
                                  states):
    # There is nothing to do here.
    return states

  def BeamSearchDecode(self, encoder_outputs, num_hyps_per_beam_override=0):
    """Performs beam-search based decoding.

    Args:
      encoder_outputs: a NestedMap computed by encoder.
      num_hyps_per_beam_override: If set to a value <= 0, this parameter is
        ignored. If set to a value > 0, then this value will be used to
        override `p.num_hyps_per_beam`.

    Returns:
      BeamSearchDecodeOutput, a namedtuple containing the decode results.
    """
    return self.beam_search.BeamSearchDecode(
        self.theta, encoder_outputs, num_hyps_per_beam_override,
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
    p.Define(
        'add_multiheaded_attention_scalar_summary', False,
        'If set, will include scalar summaries for multi-headed attention'
        ' to visualize the sparsity statistics of attention weights.')

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
    p.trans_tpl.tr_atten_tpl.source_dim = p.model_dim
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 8
    p.trans_tpl.tr_fflayer_tpl.input_dim = p.model_dim
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 2048

    # Default config for beam search.
    p.target_seq_len = 300
    p.beam_search.length_normalization = 0.5
    p.beam_search.coverage_penalty = 0.0
    p.beam_search.batch_major_state = False

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
        params.has_aux_atten = True
        params.mask_self_atten = True
        params_trans_layers.append(params)
      self.CreateChildren('trans', params_trans_layers)

      p.softmax.input_dim = p.model_dim
      self.CreateChild('softmax', p.softmax)

  def _ExpandToNumHyps(self, source_enc_len, num_hyps_per_beam):
    """Repeat each value according to num hyps.

    Args:
      source_enc_len: source encoder length; int [batch].
      num_hyps_per_beam: number of hypotheses

    Returns:
      New version of source_enc_len; int [batch * num_hyps_per_beam].
      Target_batch is (num_hyps_per_beam * batch).
      Example: src_enc_len = [3, 2, 1] and num_hyps_per_beam = 2
      --> [3, 2, 1, 3, 2, 1]
    """
    x = tf.tile(input=source_enc_len, multiples=[num_hyps_per_beam])
    return x

  def _RemoveEOSProbs(self, p, probs, source_enc_len):
    """Remove the attention probs on EOS symbol and renormalize.

    Args:
      p: decoder params.
      probs: attention probs matrix; float [batch, target_len, source_len].
      source_enc_len: source encoder length; int [batch].

    Returns:
      probs with value on last actual token (EOS token) replaced by 0 and
      renormalized so that final dim (src_len) sums to 1 again; float
      [batch, target_len, source_len].
    """
    batch = py_utils.GetShape(probs)[0]
    source_enc_len = py_utils.HasShape(source_enc_len, [batch])

    # Set -1 values
    target_len = py_utils.GetShape(probs)[1]
    replacements = tf.ones([py_utils.GetShape(probs)[0], target_len],
                           dtype=py_utils.FPropDtype(p)) * (-1)

    index_0 = tf.reshape(tf.range(batch), shape=[batch, 1, 1])
    index_0 *= tf.ones(shape=[batch, target_len, 1], dtype=tf.int32)

    index_1 = tf.ones(shape=[batch, 1], dtype=tf.int32)
    index_1 *= tf.expand_dims(tf.range(target_len), 0)
    index_1 = tf.expand_dims(index_1, -1)

    index_2 = tf.reshape(source_enc_len, shape=[batch, 1, 1]) - 1  # Note the -1
    index_2 = tf.to_int32(index_2)
    index_2 *= tf.ones(shape=[batch, target_len, 1], dtype=tf.int32)

    index = tf.concat([index_0, index_1, index_2], axis=2)

    # Original update matrix contained -1 values. Change all to 1 except for
    # those positions coming from scatter which will be 0.
    updates = tf.scatter_nd(
        index, updates=replacements, shape=py_utils.GetShape(probs))
    updates += 1
    res = probs * updates

    # Normalize to that probs sum to 1.
    # Add eps to sum to deal with case where all probs except last one are 0.
    # In this case then, attention probs will not sum to 1 but this seems still
    # better then evenly distributing attention probs in this case.
    s = tf.reduce_sum(res, axis=2, keepdims=True)
    epsilon = tf.constant(value=1e-6, dtype=py_utils.FPropDtype(p))
    s += epsilon
    res /= s
    return res

  def _FProp(self, theta, encoder_outputs, targets):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder. Expected to contain:

        encoded - source encoding. When `p.is_transparent` is False, it is a
                  tensor of shape [time, batch, depth]. When `p.is_transparent`
                  is True, it is a tensor of shape
                  [time, batch, depth, num_trans_layers] if `p.is_eval` is True,
                  and a list of `num_trans_layers` tensors of shape
                  [time, batch, depth] if `p.is_eval` is False.

        padding - source encoding's padding, of shape [time, batch].
        segment_id - source segment id, of shape [time, batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, time].

    Returns:
      `.NestedMap` containing output of last decoder layer and attention probs:
        softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
        attention: `.NestedMap` of attention distributions of shape
        [batch, target_length, source_length].
    """
    p = self.params
    source_encs = encoder_outputs.encoded
    source_paddings = encoder_outputs.padding
    src_segment_id = getattr(encoder_outputs, 'segment_id', None)
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

      if not p.packed_input:
        src_enc_len = tf.reduce_sum(1 - source_paddings, axis=0)
        num_hyps_per_beam = tf.div(
            py_utils.GetShape(target_paddings)[1],
            py_utils.GetShape(source_paddings)[1])
        src_enc_len = self._ExpandToNumHyps(src_enc_len, num_hyps_per_beam)

      layer_in = input_embs
      per_layer_attn_probs = []
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
        pl_probs = tf.transpose(probs, [1, 0, 2])
        if p.packed_input:
          # For packed inputs we are currently not removing the EOS token.
          per_layer_attn_probs.append(pl_probs)
        else:
          # Remove attention weight on last (EOS) token and re-normalize
          # so that last dimension sums to 1. See b/129097156.
          # Original probs shape: [trg time, batch, src time]
          norma_atten_probs_3d = self._RemoveEOSProbs(p, pl_probs, src_enc_len)
          per_layer_attn_probs.append(norma_atten_probs_3d)

      # per_layer_attn_probs shape: [batch, trg time, src time]
      self._AddAttenProbsSummary(source_paddings, targets, per_layer_attn_probs)

      # Aggregate per-layer attention probs.
      aggregated_atten_probs = (
          tf.math.add_n(per_layer_attn_probs) / len(per_layer_attn_probs))

      attention_map = py_utils.NestedMap(probs=aggregated_atten_probs)
      return py_utils.NestedMap(
          softmax_input=layer_out, attention=attention_map)

  def ExtendStep(self, theta, encoder_outputs, new_ids, t, prefix_states):
    """Extend prefix as represented by `prefix_states` by one more step.

    This function is expected to be called during fast decoding of Transformer
    models.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder, containing:

        - encoded: source encoding, of shape [time, batch, depth]. Can be [time,
          bs, depth, num_trans_layers] if is_transparent is set.
        - padding: source encoding's padding, of shape [time, batch].
      new_ids: new input ids, of shape [batch].
      t: a scalar, the current time step, 0-based.
      prefix_states: a `.NestedMap` representing the prefix that has already
        been decoded.

    Returns:
      A tuple (last_decoder_out, prefix_states, atten_probs), where
      last_decoder_out is the output of the last decoder layer of
      shape [batch, model_dim], `prefix_states` is the update prefix states,
      and atten_probs contains attention in shape [batch, src_len] for the
      given target position.
    """
    p = self.params
    source_paddings = encoder_outputs.padding
    time, batch = py_utils.GetShape(source_paddings, 2)
    if p.is_transparent:
      source_encs = py_utils.HasShape(
          encoder_outputs.encoded,
          [time, batch, p.source_dim, p.num_trans_layers])
      source_encs = tf.unstack(source_encs, axis=3)
    else:
      source_encs = py_utils.HasShape(encoder_outputs.encoded,
                                      [time, batch, p.source_dim])
      source_encs = [source_encs] * p.num_trans_layers
    with tf.name_scope(p.name):
      # Embedding layer
      # [batch, time, model_dim]
      token_embs = self.token_emb.EmbLookup(theta.token_emb, new_ids)
      # [time, model_dim]
      posit_embs = tf.slice(
          self.position_emb.FProp(theta.position_emb, p.target_seq_len), [t, 0],
          [1, p.model_dim])
      input_embs = token_embs + posit_embs

      if p.model_dim != p.token_emb.embedding_dim:
        input_embs = self.emb_proj.FProp(theta.emb_proj, input_embs)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      # Make a copy of the input.
      out_prefix_states = prefix_states.Pack(prefix_states.Flatten())

      layer_in = input_embs

      # Infer num_hyps_per_beam: new_ids has orig_batch_size * num_hyps_per_beam
      # source_paddings has orig_batch_size.
      num_hyps_per_beam = tf.div(
          py_utils.GetShape(new_ids)[0],
          py_utils.GetShape(source_paddings)[1])

      # Infer true source encoder length from the padding.
      src_enc_len = tf.reduce_sum(1 - source_paddings, axis=0)

      # Need to expand src_enc_len to reflect multiple hypotheses.
      src_enc_len = self._ExpandToNumHyps(src_enc_len, num_hyps_per_beam)

      atten_probs = []
      for i, (layer, layer_theta) in enumerate(zip(self.trans, theta.trans)):
        # [time, batch, model_dim]
        layer_prefix_states = prefix_states['layer_%i' % i]
        layer_out, probs, updated_prefix_states = layer.ExtendStep(
            layer_theta, layer_in, layer_prefix_states, source_encs[i],
            source_paddings,
            t if p.beam_search.name == 'tpu_beam_search' else None)
        out_prefix_states['layer_%i' % i] = updated_prefix_states
        layer_in = layer_out

        # Enforce shape: [batch, src_len]
        probs = tf.squeeze(probs)

        # Remove attention weight on last (EOS) token and re-normalize
        # so that last dimension sums to 1. See b/129097156.
        probs_3d = tf.expand_dims(probs, axis=1)
        probs_3d = self._RemoveEOSProbs(p, probs_3d, src_enc_len)
        probs = tf.squeeze(probs_3d, axis=1)

        atten_probs.append(probs)

      # Aggregate per-layer attention probs.
      aggregated_atten_probs = tf.math.add_n(atten_probs) / len(atten_probs)
      return layer_out, out_prefix_states, aggregated_atten_probs

  def ComputePredictions(self, theta, encoder_outputs, targets):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder. Expected to contain:

        encoded - source encoding, of shape [time, batch, depth]. Can be [time,
                  batch, depth, num_layers] if is_transparent is set.

        padding - source encoding's padding, of shape [time, batch].
        segment_id - source segment id, of shape [time, batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, time].

    Returns:
      A `.NestedMap` containing utput of last decoder layer and attention probs:
        softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
        attention: `.NestedMap` of attention distributions of shape
        [batch, time, source_len].
    """
    return self._FProp(theta, encoder_outputs, targets)

  def _InitBeamSearchStateCallback(self, theta, encoder_outputs,
                                   num_hyps_per_beam):
    """Returns initial beams search states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder.
      num_hyps_per_beam: An int, number hyps to keep for source sentence.
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

    source_encs = encoder_outputs.encoded
    num_hyps = py_utils.GetShape(source_encs)[1] * num_hyps_per_beam
    source_len = py_utils.GetShape(source_encs)[0]

    # Dummy attention probs
    atten_probs = tf.ones([num_hyps, source_len]) / tf.to_float(source_len)
    initial_results = py_utils.NestedMap(
        log_probs=tf.zeros([num_hyps, p.softmax.num_classes],
                           dtype=py_utils.FPropDtype(p)),
        atten_probs=atten_probs)

    batch_size = num_hyps
    atten_hidden_dim = p.trans_tpl.tr_atten_tpl.atten_hidden_dim
    if not atten_hidden_dim:
      atten_hidden_dim = p.model_dim

    if p.beam_search.name == 'tpu_beam_search':
      seq_len = p.target_seq_len
    else:
      seq_len = 0

    prefix_states = py_utils.NestedMap({
        'layer_%d' % layer: py_utils.NestedMap({  # pylint:disable=g-complex-comprehension
            'key':
                tf.zeros([seq_len, batch_size, atten_hidden_dim],
                         dtype=py_utils.FPropDtype(p)),
            'value':
                tf.zeros([seq_len, batch_size, atten_hidden_dim],
                         dtype=py_utils.FPropDtype(p)),
        }) for layer in range(p.num_trans_layers)
    })

    return initial_results, py_utils.NestedMap({
        'prefix_states': prefix_states,
        'time_step': tf.constant(0)
    })

  def _PreBeamSearchStepCallback(self, theta, encoder_outputs, step_ids, states,
                                 num_hyps_per_beam):
    """Returns logits for sampling ids and the next model states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder.
      step_ids: A tensor of shape [tgt_batch, 1].
      states: A `.NestedMap` of tensors representing states that the clients
          would like to keep track of for each of the active hyps.
      num_hyps_per_beam: Beam size.
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

    target_time = states.time_step
    prefix_states = states.prefix_states

    new_states = states.Pack(states.Flatten())

    layer_out, updated_prefix_states, atten_probs = self.ExtendStep(
        theta, encoder_outputs, tf.squeeze(step_ids, 1), target_time,
        prefix_states)

    new_states.prefix_states = updated_prefix_states
    new_states.time_step = target_time + 1

    softmax_input = tf.reshape(layer_out, [-1, p.softmax.input_dim])
    logits = self.softmax.Logits(theta.softmax, [softmax_input])

    num_hyps = py_utils.GetShape(step_ids)[0]
    # [time * batch, num_classes] -> [time, batch, num_classes]
    logits = tf.reshape(logits, (-1, num_hyps, p.softmax.num_classes))
    # [time, batch, num_classes] -> [batch, time, num_classes]
    logits = tf.transpose(logits, (1, 0, 2))

    # Only return logits for the last ids
    log_probs = tf.nn.log_softmax(tf.squeeze(logits, axis=1))

    bs_results = py_utils.NestedMap({
        'atten_probs': atten_probs,
        'log_probs': log_probs,
    })

    return bs_results, new_states

  def _PostBeamSearchStepCallback(self, theta, encoder_outputs, new_step_ids,
                                  states):
    # There is nothing to do here.
    return states

  def BeamSearchDecode(self, encoder_outputs, num_hyps_per_beam_override=0):
    return self.beam_search.BeamSearchDecode(
        self.theta, encoder_outputs, num_hyps_per_beam_override,
        self._InitBeamSearchStateCallback, self._PreBeamSearchStepCallback,
        self._PostBeamSearchStepCallback)

  def _AddAttenProbsScalarSummary(self, source_paddings, targets, atten_probs):
    """Add scalar summary of multi-headed transformer attention probs.

    This summary is primarily used to show statistics of the multi-headed
    attention that reveals potential sparsity related properties. The
    multi-headed attention probability tensors are exposed by
    `MultiHeadedAttention.ComputeContextVectorWithSource` with the name
    `multi_headed_atten_prob`. The following statistics are summarized:

    - 1_v_2: margin of the largest value vs. the 2nd largest
    - 1_v_3: similar, but vs the 3rd largest
    - mean: mean of the attention probs. NOTE: the sequences in a mini-batch
        are not always of the same length. The attention probability for the
        padded time index in target sequences are removed. However, the padding
        for the source sequences are left unchanged. As a result, the atten
        probs vectors will have some extra zero entries, so the mean calculated
        here will be smaller than the true mean.
    - source_padding_ratio: as explained above, the source paddings are not
        handled when computing the mean. This summary show the average ratio
        of time-steps that are padded values in the source sequences, to give
        a reference of roughly how much the mean summarized above should be
        adjusted.
    - 1_v_mean: margin of the largest value vs the mean value.
    - sum: the sum of the attention prob vectors. Should always be 1, for sanity
        check only.

    The quantity above are computed for each sequence in the mini-batch, each
    valid (target) sequence index, and each attention head, and then the
    average value is reported to the tensorboard as a scalar summary.

    Args:
      source_paddings: source padding, of shape [src_len, src_batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [tgt_batch, tgt_len].
      atten_probs: a list of attention probs, each element is of shape [tgt_len,
        tgt_batch, src_len].
    """
    default_graph = tf.get_default_graph()
    # looks like fprop/wmt14_en_de_transformer/tower_0_0/dec
    name_scope = default_graph.get_name_scope()
    # NOTE: shapes
    # source_paddings: [src_len, src_batch]
    # targets.paddings: [tgt_batch, tgt_len].
    source_time = tf.shape(source_paddings)[0]
    source_batch = tf.shape(source_paddings)[1]
    target_time = tf.shape(targets.paddings)[1]
    target_batch = tf.shape(targets.paddings)[0]
    num_heads = self.trans[0].self_atten.params.num_attention_heads
    with tf.control_dependencies([tf.assert_equal(source_batch, target_batch)]):
      target_batch = tf.identity(target_batch)

    source_padding_ratio = tf.cast(
        tf.reduce_sum(source_paddings, axis=0), tf.float32)
    source_padding_ratio /= tf.cast(tf.shape(source_paddings)[0], tf.float32)
    summary_utils.scalar('source_padding_ratio',
                         tf.reduce_mean(source_padding_ratio))

    for i in range(len(atten_probs)):
      suffix = '_{}'.format(i) if i > 0 else ''
      # Tensor exported from MultiHeadedAttention.ComputeContextVectorWithSource
      # shape [target_time * batch_size, num_heads, source_time]
      try:
        mha_probs = default_graph.get_tensor_by_name(
            name_scope + ('/aux_atten{}/MultiHeadedAttention/'
                          'ComputeContextVectorWithSource/'
                          'multi_headed_atten_prob:0').format(suffix))
      except KeyError:
        # no such tensor found, stop here
        return

      mha_probs = tf.reshape(
          mha_probs, (target_time, target_batch, num_heads, source_time))

      # remove time padding from target_time
      # (tgt_t, batch, n_heads, src_t) => (n_valid, n_heads, src_t)
      # explicit reshape is used here to give masks static ndims, otherwise
      # tf.boolean_mask will fail
      masks = tf.reshape(
          tf.equal(targets.paddings, 0), (target_time, target_batch))
      mha_probs = tf.boolean_mask(mha_probs, masks)

      # note we did not remove invalid entries according to source_paddings,
      # because the result will no longer be a rectangular tensor, just
      # remember when interpreting some statistics like mean, there are some
      # padded zero entries due to non-uniform sequence lengths

      # (n_valid, n_heads, src_t) => (n_valid*n_heads, src_t)
      mha_probs = tf.reshape(mha_probs, (-1, tf.shape(mha_probs)[-1]))

      probs_top3, _ = tf.math.top_k(mha_probs, k=3)
      probs_mean = tf.math.reduce_mean(mha_probs, axis=1)
      probs_sum = tf.math.reduce_sum(mha_probs, axis=1)  # sanity check

      margins_12 = tf.reduce_mean(probs_top3[:, 0] - probs_top3[:, 1])
      margins_13 = tf.reduce_mean(probs_top3[:, 0] - probs_top3[:, 2])
      margins_1m = tf.reduce_mean(probs_top3[:, 0] - probs_mean)
      summary_utils.scalar('1_v_2/atten{}'.format(i), margins_12)
      summary_utils.scalar('1_v_3/atten{}'.format(i), margins_13)
      summary_utils.scalar('1_v_mean/atten{}'.format(i), margins_1m)
      summary_utils.scalar('mean/atten{}'.format(i), tf.reduce_mean(probs_mean))
      summary_utils.scalar('sum/atten{}'.format(i), tf.reduce_mean(probs_sum))

  def _AddAttenProbsSummary(self, source_paddings, targets, atten_probs):
    """Add summary of attention probs.

    Args:
      source_paddings: source padding, of shape [src_len, src_batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [tgt_batch, tgt_len].
      atten_probs: a list of attention probs, each element is of shape [tgt_len,
        tgt_batch, src_len].
    """
    super(TransformerDecoder,
          self)._AddAttenProbsSummary(source_paddings, targets, atten_probs)
    if self.cluster.add_summary and self.params.add_multiheaded_attention_scalar_summary:
      self._AddAttenProbsScalarSummary(source_paddings, targets, atten_probs)
