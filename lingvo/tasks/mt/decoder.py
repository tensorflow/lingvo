# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
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

import math
import lingvo.compat as tf
from lingvo.core import attention
from lingvo.core import base_decoder
from lingvo.core import batch_major_attention
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import model_helper
from lingvo.core import plot
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers
from lingvo.core import summary_utils


class MTBaseDecoder(base_decoder.BaseBeamSearchDecoder):
  """Base class for Lingvo MT decoders."""

  @classmethod
  def Params(cls):
    p = super().Params()
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
    p.Define('per_example_tensors', False, 'Return per example tensors')

    # Default config for the softmax part.
    p.softmax.num_classes = 32000  # 32k
    p.softmax.num_shards = 8

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.label_smoothing is not None:
      p.label_smoothing.name = 'smoother'
      p.label_smoothing.num_classes = p.softmax.num_classes
      self.CreateChild('smoother', p.label_smoothing)

  @classmethod
  def UpdateTargetVocabSize(cls, p, vocab_size, wpm_model=None):
    """Sets the params with the given vocab size and wpm model.

    Args:
      p: model params.
      vocab_size: size of the vocabulary.
      wpm_model: file name prefix pointing to a wordpiece model.

    Returns:
      Model params updated with the vocab size and wpm model.
    """
    p.softmax.num_classes = vocab_size
    return p

  def _ComputeXentLoss(self,
                       theta,
                       softmax_input,
                       target_labels,
                       target_weights,
                       target_paddings,
                       target_segment_ids=None,
                       time_axis=0):
    """Computes cross-entropy loss given the softmax input, labels and weights.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      softmax_input: A tensor of shape [time, batch, p.softmax.input_dim].
      target_labels: A matrix of tf.int32. [time, batch].
      target_weights: A matrix of params.dtype. [time, batch].
      target_paddings: A matrix of params.dtype. [time, batch].
      target_segment_ids: A matrix of params.dtype. [time, batch].
      time_axis: If 0, the inputs are time-major: [time, batch, ...]; if 1, the
        inputs are batch-major: [batch, time, ...].

    Returns:
      The cross entropy loss.
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
      if time_axis == 0:
        target_probs = tf.transpose(
            self.smoother.FProp(
                theta.smoother,
                tf.transpose(target_paddings),
                tf.transpose(target_labels),
                target_ids=None), [1, 0, 2])
      else:
        target_probs = self.smoother.FProp(
            theta.smoother, target_paddings, target_labels, target_ids=None)
      xent_loss = self.softmax.FProp(
          theta.softmax, [softmax_input],
          class_weights=tf.reshape(target_weights, [-1, 1]),
          class_probabilities=tf.reshape(target_probs,
                                         [-1, p.softmax.num_classes]))
    return xent_loss

  def _ComputeSoftmaxMetrics(self,
                             xent_loss,
                             target_labels,
                             target_weights,
                             target_segment_ids=None,
                             time_axis=0):
    """Computes cross-entropy metrics given the cross-entropy loss.

    Args:
      xent_loss: The output of `_ComputeXentLoss`.
      target_labels: A matrix of tf.int32. [time, batch].
      target_weights: A matrix of params.dtype. [time, batch].
      target_segment_ids: A matrix of params.dtype. [time, batch].
      time_axis: If 0, the inputs are time-major: [time, batch, ...]; if 1, the
        inputs are batch-major: [batch, time, ...].

    Returns:
      A tuple (metrics, per_example_tensors).
        metrics:
          A dictionary containing metrics for the xent loss and prediction
          accuracy.
        per_example_tensors:
          A dictionary of per-example tensors.
    """
    p = self.params
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
      per_sequence_loss = tf.reduce_sum(
          per_example_loss * target_weights, axis=time_axis)
      if p.packed_input:
        assert target_segment_ids is not None, (
            'Need target segment ids for '
            'normalizing loss when training with packed inputs.')
        num_samples_per_row = tf.math.reduce_max(
            target_segment_ids, axis=time_axis)
        num_samples = tf.reduce_sum(num_samples_per_row)
        final_loss = tf.reduce_sum(per_sequence_loss) / tf.cast(
            num_samples, per_sequence_loss.dtype)
      else:
        final_loss = tf.reduce_mean(per_sequence_loss)
      loss_weight = tf.cast(
          py_utils.GetShape(per_sequence_loss)[0], per_sequence_loss.dtype)

    metrics = {
        'loss': (final_loss, loss_weight),
        'log_pplx': (xent_loss.avg_xent, xent_loss.total_weight),
    }

    per_example_tensors = {}
    if p.per_example_tensors:
      per_example_tensors['per_example_loss'] = tf.reshape(
          xent_loss.per_example_xent, py_utils.GetShape(target_weights))
      per_example_tensors['per_sequence_loss'] = tf.reduce_sum(
          per_example_tensors['per_example_loss'] * target_weights,
          axis=time_axis)
      per_example_tensors['loss'] = per_example_tensors['per_sequence_loss']
      per_example_tensors['logits'] = tf.reshape(
          xent_loss.logits,
          tf.concat([py_utils.GetShape(target_weights), [-1]], 0))
      per_example_tensors['log_probs'] = tf.reshape(
          xent_loss.log_probs,
          tf.concat([py_utils.GetShape(target_weights), [-1]], 0))

    # NOTE: tf.argmax is not implemented for the JF backend, see b/36093673
    # Skip the fraction_of_correct_next_step_preds during training.
    if self.do_eval:
      logits = xent_loss.logits
      correct_preds = tf.cast(
          tf.equal(
              tf.cast(tf.reshape(tf.argmax(logits, 1), [-1]), tf.int32),
              tf.reshape(target_labels, [-1])), p.dtype)
      correct_next_preds = tf.reduce_sum(
          correct_preds * tf.reshape(tf.cast(target_weights, p.dtype), [-1]))
      num_preds = tf.reduce_sum(tf.cast(target_weights, p.dtype))
      accuracy = tf.identity(
          correct_next_preds / num_preds,
          name='fraction_of_correct_next_step_preds')
      metrics['fraction_of_correct_next_step_preds'] = (accuracy, num_preds)
    return metrics, per_example_tensors

  def _FPropSoftmax(self,
                    theta,
                    softmax_input,
                    target_labels,
                    target_weights,
                    target_paddings,
                    target_segment_ids=None,
                    time_axis=0):
    """Computes cross-entropy loss given the softmax input, labels and weights.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      softmax_input: A tensor of shape [time, batch, p.softmax.input_dim].
      target_labels: A matrix of tf.int32. [time, batch].
      target_weights: A matrix of params.dtype. [time, batch].
      target_paddings: A matrix of params.dtype. [time, batch].
      target_segment_ids: A matrix of params.dtype. [time, batch].
      time_axis: If 0, the inputs are time-major: [time, batch, ...]; if 1, the
        inputs are batch-major: [batch, time, ...].

    Returns:
      A tuple (metrics, per_example_tensors).
        metrics:
          A dictionary containing metrics for the xent loss and prediction
          accuracy.
        per_example_tensors:
          A dictionary of per-example tensors.
    """
    xent_loss = self._ComputeXentLoss(theta, softmax_input, target_labels,
                                      target_weights, target_paddings)
    return self._ComputeSoftmaxMetrics(xent_loss, target_labels, target_weights,
                                       target_segment_ids, time_axis=time_axis)

  def ComputeLoss(self, theta, predictions, targets):
    """Populates a metrics dictionary based on the output of ComputePredictions.

    Args:
      theta: Nested map describing decoder model parameters.
      predictions: NestedMap describing the decoding process, requiring:
        .softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
      targets: NestedMap describing the target sequences.

    Returns:
      Two dicts.

        - A map from metric name (a python string) to a tuple (value, weight).
          Both value and weight are scalar Tensors.
        - A map from name to arbitrary tensors, where the first dimension must
          be the batch index.
    """
    segment_id = None
    if self.params.packed_input:
      segment_id = tf.transpose(targets.segment_ids)
    if isinstance(predictions, py_utils.NestedMap):
      predictions = predictions.softmax_input
    return self._FPropSoftmax(theta, predictions, tf.transpose(targets.labels),
                              tf.transpose(targets.weights),
                              tf.transpose(targets.paddings), segment_id)

  def _TruncateTargetSequence(self, targets):
    """Truncate padded time steps from all sequences."""
    # The following tensors are all in the [batch, time] shape.
    # Let's make a copy of targets.
    targets = targets.Pack(targets.Flatten())
    target_ids = targets.ids
    target_labels = targets.labels
    target_weights = targets.weights
    target_paddings = targets.paddings
    max_seq_length = tf.cast(
        tf.round(tf.reduce_max(tf.reduce_sum(1.0 - target_paddings, 1))),
        tf.int32)
    summary_utils.scalar('max_seq_length', max_seq_length)
    # Assert to make sure after max_seq_length, all are padded steps for all
    # sequences.
    target_paddings = py_utils.with_dependencies([
        py_utils.assert_equal(
            tf.constant(True, tf.bool),
            tf.reduce_all(target_paddings[:, max_seq_length:] > 0.5))
    ], target_paddings)
    target_ids = py_utils.with_dependencies([
        py_utils.AssertIdShape(
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
    def PlotAttention(fig, axes, cur_atten_probs, title, set_x_label):
      plot.AddImage(fig, axes, cur_atten_probs, title=title)
      axes.set_ylabel(plot.ToUnicode('Output sequence index'), wrap=True)
      if set_x_label:
        axes.set_xlabel(plot.ToUnicode('Input sequence index'), wrap=True)

    index = 0
    srclen = tf.cast(
        tf.round(tf.reduce_sum(1 - source_paddings[:, index])), tf.int32)
    tgtlen = tf.cast(
        tf.round(tf.reduce_sum(1 - targets.paddings[index, :])), tf.int32)

    num_rows = len(atten_probs)
    with plot.MatplotlibFigureSummary(
        'decoder_example',
        figsize=(6, 3 * num_rows),
        max_outputs=1,
        subplot_grid_shape=(num_rows, 1)) as fig:
      for i, probs in enumerate(atten_probs):
        # Extract first entry in batch of attention prob matrices
        # [tgt_len, src_len]
        probs = probs[:, index, :]
        probs = tf.expand_dims(probs[:tgtlen, :srclen], 0)
        fig.AddSubplot([probs],
                       PlotAttention,
                       title='atten_probs_%d' % i,
                       set_x_label=(i == len(atten_probs) - 1))

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


class MTDecoderV1(MTBaseDecoder, quant_utils.QuantizableLayer):
  """MT decoder v1."""

  # We scale a float's dtype.max by this amount to stand in for a
  # sufficiently large number. This is chosen such that for log_prob values,
  # accumulating it over beam search steps will not cause numerical issues.
  _FLOAT_DTYPE_MAX_SCALER = 0.00001

  @classmethod
  def Params(cls):
    p = super().Params()
    # Shared embedding.
    p.Define('emb', layers.EmbeddingLayer.Params(), 'Embedding layer params.')
    p.Define('source_dim', 1024, 'Dimension of the source encoding.')
    p.Define('attention', attention.AdditiveAttention.Params(),
             'Additive attention params.')
    p.Define('atten_rnn_cell_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Attention RNNCell params template.')
    p.Define(
        'emb_projection_tpl', None,
        'Template for embedding projection layer. If set, the embeddings '
        'are projected to match the `rnn_cell_dim`, if they are different. '
        'It will also add a projection if `rnn_cell_dim` is not equal to '
        'softmax input_dim (shared embedding case). The clipping schedule, '
        'if enabled, is also applied after projection.'
        'See Factorized Embedding in ALBERT: https://arxiv.org/abs/1909.11942')
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
    p.Define(
        'init_step_ids', False,
        'Initializes beam search with first target id instead of <s>.'
        'Use this when decoding starts with target_lang id intead of <s> '
        'token at time step 0. Make sure the training data has '
        'target_lang id as the first token in target sequence.')
    p.Define(
        'force_alignment', False,
        'When input contains multiple sentences, adjusts the scores of '
        'p.sentence_boundary_token_id and EOS during beam search to force '
        'the hypothesis to contain the same number of sentences as the input '
        'source. Must also set p.sentence_boundary_token_id.')
    p.Define(
        'sentence_boundary_token_id', None,
        'None, or an int. The token id that separates different sentences.')
    p.Define(
        'single_token_fast_decode', False,
        'bool. When enabled, for input of length 1, decoding completes in 1 '
        'step. This reserves a special sentinel input with fast beam search. '
        'It can be used to pad inputs to a fixed batch size.')

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

  @classmethod
  def UpdateTargetVocabSize(cls, p, vocab_size, wpm_model=None):
    """Updates the params with the input vocab_size and WPM model.

    Args:
      p: model params.
      vocab_size: size of the vocabulary.
      wpm_model: file name prefix pointing to a wordpiece model.

    Returns:
      Model params updated with the vocab size and wpm model.
    """
    p = super().UpdateTargetVocabSize(p, vocab_size)
    p.emb.vocab_size = vocab_size
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.force_alignment and p.sentence_boundary_token_id is None:
      raise ValueError('When p.force_alignment is set, '
                       'must specify p.sentence_boundary_token_id.')
    if p.softmax.cls == layers.SharedSoftmaxLayer:
      self._share_sm_emb = True
    else:
      self._share_sm_emb = False

    if p.cc_schedule is None:
      self.cc_schedule = None
    else:
      self.CreateChild('cc_schedule', p.cc_schedule)

    if not self._share_sm_emb:
      assert p.emb.vocab_size == p.softmax.num_classes
      self.CreateChild('emb', p.emb)

    self._project_emb = False
    self._project_out = False
    if p.emb_projection_tpl:
      self._project_emb = (p.emb.embedding_dim != p.rnn_cell_dim)
      self._project_out = (self._project_emb and self._share_sm_emb)

    p.attention.dtype = p.dtype
    p.attention.source_dim = p.source_dim
    p.attention.query_dim = p.rnn_cell_dim
    p.attention.packed_input = p.packed_input
    if p.attention.params_init is None:
      p.attention.params_init = py_utils.WeightInit.Gaussian(
          1. / math.sqrt(p.attention.source_dim + p.attention.query_dim),
          seed=p.random_seed)
    atten_params = p.attention.Copy()

    if ('enable_ctx_post_proj' in p.attention and
        p.attention.enable_ctx_post_proj):
      atten_context_dim = p.attention.ctx_post_proj_dim
    else:
      atten_context_dim = p.attention.source_dim

    params = p.atten_rnn_cell_tpl.Copy()
    params.name = 'atten_rnn'
    params.dtype = p.dtype
    params.reset_cell_state = p.packed_input
    if self._project_emb:
      params.num_input_nodes = p.rnn_cell_dim + atten_context_dim
    else:
      params.num_input_nodes = p.emb.embedding_dim + atten_context_dim
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
    params.atten_context_dim = atten_context_dim
    self.CreateChild('frnn_with_atten', params)

    # TODO(zhifengc): Avoid this?
    self._atten = self.frnn_with_atten.attention

    rnn_layers_params = []
    for i in range(1, p.rnn_layers):
      params = p.rnn_cell_tpl.Copy()
      params.name = 'rnn%d' % i
      params.dtype = p.dtype
      params.num_input_nodes = p.rnn_cell_dim + atten_context_dim
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
      assert not self._share_sm_emb
      assert not self._project_emb
      assert not self._project_out
      p.softmax.input_dim = p.rnn_cell_dim + atten_context_dim
    else:
      p.softmax.input_dim = p.rnn_cell_dim

    # Factorized embedding:
    # An embedding matrix of O(vocab_size * emb_dim) is factorized into
    # two matrices O(vocab_size * hidden_dim) + O(hidden_dim * embedding_dim).
    # See ALBERT: https://arxiv.org/abs/1909.11942.
    if self._project_emb:
      tf.logging.info('Creating an embedding projection from '
                      f'{p.emb.embedding_dim} -> {p.rnn_cell_dim} '
                      'before feeding to rnn.')
      self._CreateProjection(p.emb_projection_tpl, 'emb_proj',
                             p.emb.embedding_dim, p.rnn_cell_dim)
    if self._project_out:
      tf.logging.info('Creating an out projection from '
                      f'{p.rnn_cell_dim} -> {p.emb.embedding_dim} '
                      'before feeding to softmax layer.')
      p.softmax.input_dim = p.emb.embedding_dim
      self._CreateProjection(p.emb_projection_tpl, 'out_proj', p.rnn_cell_dim,
                             p.emb.embedding_dim)
    self.CreateChild('softmax', p.softmax)

  def _CreateProjection(self, proj_tpl, name, input_dim, output_dim):
    """Creates a projection layer(projects from `input_dim` to `output_dim`)."""
    assert proj_tpl.cls == layers.ProjectionLayer
    proj_p = proj_tpl.Copy()
    proj_p.name = name
    proj_p.input_dim = input_dim
    proj_p.output_dim = output_dim
    self.CreateChild(name, proj_p)
    return proj_p

  def _CreateChildrenVariables(self):
    if self._share_sm_emb:
      # Taking shared emb/softmax layer out of the decoder variable scope so
      # that it can also be shared by encoder if needed.
      with tf.variable_scope('shared_emb', reuse=tf.AUTO_REUSE):
        self.softmax.InstantiateVariables()

    super()._CreateChildrenVariables()

  def ApplyDropout(self, x_in):
    p = self.params
    assert 0 <= p.dropout_prob and p.dropout_prob < 1.0
    if self.do_eval or p.dropout_prob == 0.0:
      return x_in
    else:
      return tf.nn.dropout(x_in, rate=p.dropout_prob)

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

      if self._share_sm_emb:
        inputs = self.softmax.EmbLookup(theta.softmax, target_ids)

      with tf.device(emb_device):
        if not self._share_sm_emb:
          inputs = self.emb.EmbLookup(theta.emb, target_ids)
        inputs = self.ApplyClipping(theta, inputs)
        summary_utils.histogram('input_emb', inputs)
        inputs = self.ApplyDropout(inputs)

        if self._project_emb:
          inputs = self.emb_proj.FProp(theta.emb_proj, inputs)
          summary_utils.histogram('emb_proj', inputs)
          inputs = self.ApplyClipping(theta, inputs)

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

        if self._project_out:
          xs = self.out_proj.FProp(theta.out_proj, xs)
          summary_utils.histogram('out_proj', xs)
          xs = self.ApplyClipping(theta, xs)

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

  def AddExtraDecodingInfo(self, encoder_outputs, targets):
    """Adds extra decoding information to encoded_outputs.

    Args:
      encoder_outputs: a NestedMap computed by encoder.
      targets: a NestedMap containing target input fields.

    Returns:
      encoder_ouputs with extra information used for decoding.
    """
    p = self.params
    if p.init_step_ids:
      encoder_outputs['init_step_ids'] = targets.ids[:, 0]
    return encoder_outputs

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
      Tuple of initial model states. Also inserts 'packed_src' to
      'encoder_outputs'.
    """
    p = self.params
    source_paddings = encoder_outputs.padding
    time, batch = py_utils.GetShape(source_paddings, 2)
    source_encs = py_utils.HasShape(encoder_outputs.encoded,
                                    [time, batch, p.source_dim])
    rnn_states = [
        self.frnn_with_atten.cell.zero_state(theta.frnn_with_atten.cell,
                                             num_hyps)
    ]
    for layer, layer_theta in zip(self.frnn, theta.frnn):
      rnn_states.append(layer.zero_state(layer_theta, num_hyps))

    if p.use_zero_atten_state:
      encoder_outputs.packed_src = self._atten.InitForSourcePacked(
          theta.frnn_with_atten.atten, source_encs, source_encs,
          source_paddings)
      s_seq_len = tf.shape(source_encs)[0]
      context_dim = tf.shape(source_encs)[2]
      atten_context = tf.zeros([num_hyps, context_dim], dtype=source_encs.dtype)
      atten_states = self._atten.ZeroAttentionState(s_seq_len, num_hyps)
      atten_probs = tf.zeros([num_hyps, s_seq_len], dtype=source_encs.dtype)
    else:
      encoder_outputs.packed_src = self._atten.InitForSourcePacked(
          theta.frnn_with_atten.atten, source_encs, source_encs,
          source_paddings)

      src_seq_len = tf.shape(source_encs)[0]
      zero_atten_state = self._atten.ZeroAttentionState(src_seq_len, num_hyps)

      (atten_context, atten_probs,
       atten_states) = self._atten.ComputeContextVectorWithSource(
           theta.frnn_with_atten.atten,
           encoder_outputs.packed_src,
           tf.zeros([num_hyps, p.rnn_cell_dim], dtype=py_utils.FPropDtype(p)),
           attention_state=zero_atten_state)

    assert atten_states is not None
    return rnn_states, atten_context, atten_probs, atten_states

  @py_utils.NameScopeDecorator('MTDecoderV1/DecodeStep')
  def _DecodeStep(self, theta, encoder_outputs, embs, step_paddings,
                  prev_atten_context, rnn_states, prev_atten_states):
    """Decode one step."""
    p = self.params
    if self._project_emb:
      embs = self.emb_proj.FProp(theta.emb_proj, embs)
      embs = self.ApplyClipping(theta, embs)
    new_rnn_states = []
    new_rnn_states_0, _ = self.frnn_with_atten.cell.FProp(
        theta.frnn_with_atten.cell, rnn_states[0],
        py_utils.NestedMap(
            act=[tf.concat([embs, prev_atten_context], 1)],
            padding=step_paddings,
            reset_mask=tf.ones_like(step_paddings)))
    new_rnn_states.append(new_rnn_states_0)
    rnn_out = self.frnn_with_atten.cell.GetOutput(new_rnn_states_0)
    cur_atten_context, atten_probs, atten_states = (
        self._atten.ComputeContextVectorWithSource(
            theta.frnn_with_atten.atten,
            encoder_outputs.packed_src,
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
    if self._project_out:
      step_out = self.out_proj.FProp(theta.out_proj, step_out)
      step_out = self.ApplyClipping(theta, step_out)
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

    if p.init_step_ids and hasattr(encoder_outputs, 'init_step_ids'):
      initial_results['step_ids'] = tf.expand_dims(
          self._ExpandToNumHyps(encoder_outputs.init_step_ids,
                                num_hyps_per_beam), 1)
    states = py_utils.NestedMap({
        'time_step': tf.constant(0),
        'rnn_states': rnn_states,
        'atten_context': init_atten_context,
        'atten_probs': atten_probs,
        'atten_states': atten_states,
    })
    if p.force_alignment:
      states['num_sentences'] = tf.ones([num_hyps], dtype=tf.int32)
    return initial_results, states

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
    if self._share_sm_emb:
      embs = self.softmax.EmbLookup(theta.softmax, tf.reshape(step_ids, [-1]))
    else:
      embs = self.emb.EmbLookup(theta.emb, tf.reshape(step_ids, [-1]))
    embs = self.ApplyClipping(theta, embs)
    atten_context, atten_probs, rnn_states, step_out, atten_states = (
        self._DecodeStep(theta, encoder_outputs, embs, step_paddings,
                         prev_atten_context, prev_rnn_states,
                         prev_atten_states))
    atten_probs = tf.reshape(atten_probs, tf.shape(prev_atten_probs))

    logits = self.softmax.Logits(theta.softmax, [step_out])
    log_probs = self.fns.qlogsoftmax(
        logits, qmin=p.qlogsoftmax_range_min, qmax=0.0)
    if p.force_alignment:
      if 'num_sentences' not in encoder_outputs:
        raise ValueError('Model does not support p.force_alignment as '
                         'key "num_sentences" is missing from encoder_outputs.')
      source_num_sentences = tf.tile(
          encoder_outputs['num_sentences'], multiples=[num_hyps_per_beam])
      log_probs = self._ForceAlignment(log_probs, source_num_sentences,
                                       states['num_sentences'])

    if p.single_token_fast_decode:
      input_lengths = tf.math.reduce_sum(1.0 - encoder_outputs.padding, 0)
      is_single_token = tf.math.less_equal(input_lengths,
                                           tf.ones_like(input_lengths))
      any_single_token = tf.math.reduce_any(is_single_token)

      def _NewLogProbs():
        return self._UpdateLogitsForSingleTokenFastDecode(
            log_probs, is_single_token, num_hyps_per_beam)

      log_probs = tf.cond(any_single_token, _NewLogProbs, lambda: log_probs)

    if p.use_prev_atten_ctx:
      cur_atten_probs = prev_atten_probs
    else:
      cur_atten_probs = atten_probs

    bs_results = py_utils.NestedMap({
        'atten_probs': cur_atten_probs,  # the probs exposed to beam search
        'log_probs': log_probs,
    })
    new_states = py_utils.NestedMap({
        'time_step': states.time_step + 1,
        'rnn_states': rnn_states,
        'atten_context': atten_context,
        'atten_probs': atten_probs,  # the updated attention probs
        'atten_states': atten_states,
    })
    if p.force_alignment:
      new_states['num_sentences'] = states['num_sentences']

    return bs_results, new_states

  def _PostBeamSearchStepCallback(self, theta, encoder_outputs, new_step_ids,
                                  states):
    p = self.params
    if p.force_alignment:
      add = tf.squeeze(
          tf.math.equal(new_step_ids, p.sentence_boundary_token_id), axis=1)
      states['num_sentences'] += tf.cast(
          add, dtype=states['num_sentences'].dtype)
    return states

  def _ForceAlignment(self, log_probs, source_num_sentences, hyp_num_sentences):
    """Update 'log_probs' to for alignment.

    We adjust 'log_probs' to disallow p.sentence_boundary_token_id or EOS if
    emitting it would result in a misaligned output with an unequal number
    of sentences.

    Args:
      log_probs: encoder's log_probs output.
      source_num_sentences: shape [beam_size * num_hyps_per_beam], int32. The
        number of sentences in source.
      hyp_num_sentences: shape [beam_size * num_hyps_per_beam], int32. The
        number of sentences in hyp.

    Returns:
      The adjusted log_probs (score used for beam search) which ensures
      aligned output in terms of number of sentences.
    """
    p = self.params
    eos_id = p.target_eos_id
    # We replace log_probs with a sufficiently large negative value where
    # the current hyp contains fewer sentences than expected to disallow
    # eos in such misaligned cases.
    large_negative_value = tf.ones_like(log_probs[:, eos_id]) * tf.constant(
        -self._FLOAT_DTYPE_MAX_SCALER,
        dtype=log_probs.dtype) * log_probs.dtype.max
    eos_log_probs = tf.where(
        tf.math.greater(source_num_sentences, hyp_num_sentences),
        large_negative_value, log_probs[:, eos_id])
    eos_log_probs = tf.expand_dims(eos_log_probs, axis=1)
    boundary_id = p.sentence_boundary_token_id
    boundary_id_log_probs = tf.where(
        tf.math.less_equal(source_num_sentences, hyp_num_sentences),
        large_negative_value, log_probs[:, boundary_id])
    boundary_id_log_probs = tf.expand_dims(boundary_id_log_probs, axis=1)
    new_log_probs = tf.concat(
        [log_probs[:, :eos_id], eos_log_probs, log_probs[:, eos_id + 1:]],
        axis=1)
    new_log_probs = tf.concat([
        new_log_probs[:, :boundary_id], boundary_id_log_probs,
        new_log_probs[:, boundary_id + 1:]
    ],
                              axis=1)
    return new_log_probs

  def _UpdateLogitsForSingleTokenFastDecode(self, log_probs, is_single_token,
                                            num_hyps_per_beam):
    """Update 'log_probs' to enable fast decode for single token inputs.

    Args:
      log_probs: encoder's log_probs output, shape [tgt_batch, vocab_size].
      is_single_token: [src_batch], whether the input contains only a single
        token.
      num_hyps_per_beam: int, num_hyps_per_beam * src_batch = tgt_batch.

    Returns:
      The updated log_probs (score used for beam search) which ensures
      fast decoding when input has a single token.
    """
    b, v = py_utils.GetShape(log_probs, 2)
    is_single_token = tf.tile(is_single_token, multiples=[num_hyps_per_beam])
    # Shape [tgt_batch, vocab_size]
    is_single_token_2d = tf.tile(tf.expand_dims(is_single_token, 1), [1, v])
    # Updated log_probs concentrates on eos_id entirely.
    eos_id = self.params.target_eos_id
    is_eos = tf.math.equal(tf.range(v), tf.ones_like(tf.range(v)) * eos_id)
    is_eos = tf.tile(tf.expand_dims(is_eos, 0), [b, 1])
    large_neg_probs = tf.ones_like(log_probs) * tf.constant(
        -self._FLOAT_DTYPE_MAX_SCALER,
        dtype=log_probs.dtype) * log_probs.dtype.max
    new_log_probs = tf.where(is_eos, tf.zeros_like(large_neg_probs),
                             large_neg_probs)
    return tf.where(is_single_token_2d, new_log_probs, log_probs)


class TransformerDecoder(MTBaseDecoder):
  """Transformer decoder.

  Implements the decoder of Transformer model:
  https://arxiv.org/abs/1706.03762.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('token_emb', layers.EmbeddingLayer.Params(),
             'Token embedding layer params.')
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define('source_dim', 1024, 'Dimension of encoder outputs.')
    p.Define('model_dim', 1024, 'Model dimension that applies to embedding '
             'layers and all Transformer layers.')
    p.Define('num_trans_layers', 6, 'Number of Transformer layers.')
    p.Define(
        'trans_tpl', layers_with_attention.TransformerLayer.Params(),
        'Transformer layer params. '
        ' Can be a list. num_trans_layers should be divisible by '
        'len(trans_tpl).')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define(
        'is_transparent', False, 'If set, expects a tensor of shape '
        '[time, batch, source_dim, num_trans_layers] as source encodings.')
    p.Define(
        'add_multiheaded_attention_scalar_summary', False,
        'If set, will include scalar summaries for multi-headed attention'
        ' to visualize the sparsity statistics of attention weights.')
    p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
    p.Define(
        'ln_output', False, 'If True, layer normalization is applied to the '
        'final output of the decoder.')

    # TODO(miachen): Extend this to more general logic of adding multiple
    # embedding fields.
    p.Define('task_emb', None, 'Task embedding layer params.')
    p.Define(
        'init_step_ids', False,
        'Initializes beam search with first target id instead of <s>.'
        'Use this when decoder has target language token intead of <s> '
        'token at time step 0.'
        'Make sure the training is done in similar manner.')
    # MASS pretraining related (https://github.com/microsoft/MASS)
    p.Define(
        'use_lang_dependent_atten', False, 'If True, attention between '
        'encoder and decoder is language dependent.')

    p.Define('zero_token_embs_first_time_step', False,
             'If True, the first time step uses zeros as the post-emb lookup.')

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

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if p.softmax.cls == layers.SharedSoftmaxLayer:
      self._token_emb_vocab_size = p.softmax.num_classes
      self._token_emb_dim = p.model_dim
      self._share_sm_emb = True
    else:
      self._token_emb_vocab_size = p.token_emb.vocab_size
      self._token_emb_dim = p.token_emb.embedding_dim
      self._share_sm_emb = False

    assert self._token_emb_vocab_size == p.softmax.num_classes
    assert self._token_emb_dim == p.position_emb.embedding_dim
    if p.model_dim != self._token_emb_dim:
      tf.logging.warning(
          'token_emb.embedding_dim != model_dim (%s vs. %s), '
          'creating a projection!')
      proj_p = layers.ProjectionLayer.Params().Copy()
      proj_p.name = 'emb_proj'
      proj_p.input_dim = p.token_emb.embedding_dim
      proj_p.output_dim = p.model_dim
      self.CreateChild('emb_proj', proj_p)

    if p.use_lang_dependent_atten and p.task_emb:
      p.trans_tpl.num_aux_atten_post_proj = p.task_emb.vocab_size

    if not self._share_sm_emb:
      self.CreateChild('token_emb', p.token_emb)
    self.CreateChild('position_emb', p.position_emb)
    if p.task_emb:
      assert p.task_emb.embedding_dim == self._token_emb_dim
      self.CreateChild('task_emb', p.task_emb)

    dropout_tpl = layers.DropoutLayer.Params()
    dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
    self.CreateChild('input_dropout', dropout_tpl)

    params_trans_layers = []
    denom = 1
    if isinstance(p.trans_tpl, list):
      denom = len(p.trans_tpl)
    assert p.num_trans_layers % denom == 0
    for i in range(p.num_trans_layers // denom):
      if isinstance(p.trans_tpl, list):
        for q in p.trans_tpl:
          params = q.Copy()
          params_trans_layers.append(params)
      else:
        params = p.trans_tpl.Copy()
        params_trans_layers.append(params)

    for i, params in enumerate(params_trans_layers):
      params.name = 'trans_layer_%d' % i
      params.packed_input = p.packed_input
      params.has_aux_atten = True
      params.mask_self_atten = True

    # Initialize decoder output layer norm
    if p.ln_output:
      params = p.ln_tpl.Copy()
      params.name = 'dec_out_ln'
      params.input_dim = p.model_dim
      self.CreateChild('layer_norm_out', params)

    self.CreateChildren('trans', params_trans_layers)

    p.softmax.input_dim = p.model_dim
    self.CreateChild('softmax', p.softmax)

  def _CreateChildrenVariables(self):
    if self._share_sm_emb:
      # Taking shared emb/softmax layer out of the decoder variable scope so
      # that it can also be shared by encoder if needed.
      with tf.variable_scope('shared_emb', reuse=tf.AUTO_REUSE):
        self.softmax.InstantiateVariables()
    super()._CreateChildrenVariables()

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
    index_2 = tf.cast(index_2, tf.int32)
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

  def _ZeroOutFirstTimeStep(self, token_embs, batch, target_time):
    """Zeroes out the first time step.

    Args:
      token_embs:  [batch, time, model_dim] embeding lookups
      batch: Batch size scalar
      target_time: Target sequence length scalar.

    Returns:
      modified token_embs with the first time step zeroed out.
    """
    p = self.params

    zero_out_index = tf.expand_dims(tf.constant([0]), axis=1)
    # [[[0]]]
    zero_out_index = tf.expand_dims(zero_out_index, axis=1)

    # [[0]...[target_time-1]]
    time_steps = tf.expand_dims(tf.range(target_time), axis=1)
    condition = tf.equal(zero_out_index, time_steps)
    mask = tf.logical_not(tf.tile(condition, [batch, 1, p.model_dim]))
    mask = tf.cast(mask, dtype=tf.float32)
    return token_embs * mask

  def _FProp(self, theta, encoder_outputs, targets):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder. Expected to contain:
        encoded - source encoding. When `p.is_transparent` is False, it is a
        tensor of shape [time, batch, depth]. When `p.is_transparent` is True,
        it is a tensor of shape [time, batch, depth, num_trans_layers] if
        `self.do_eval` is True, and a list of `num_trans_layers` tensors of
        shape [time, batch, depth] if `self.do_eval` is False.  padding - source
        encoding's padding, of shape [time, batch]. segment_id - source segment
        id, of shape [time, batch].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, time].

    Returns:
      A `.NestedMap` containing output of last decoder layer and attention probs

      - softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
      - attention: `.NestedMap` of attention distributions of shape
        [batch, target_length, source_length].
    """
    p = self.params
    source_encs = encoder_outputs.encoded
    source_paddings = encoder_outputs.padding
    src_segment_id = getattr(encoder_outputs, 'segment_id', None)
    time, batch = py_utils.GetShape(source_paddings, 2)
    if p.is_transparent:
      if self.do_eval:
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
      if not self._share_sm_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, target_ids)

      target_batch = py_utils.GetShape(target_ids)[0]
      target_time = py_utils.GetShape(target_ids)[1]

      if p.zero_token_embs_first_time_step:
        # For models that do not use an explicit start-of-sequence token
        # with associated embedding, but instead use zeros.
        token_embs = self._ZeroOutFirstTimeStep(token_embs, target_batch,
                                                target_time)

      # [1, time, model_dim]
      if p.packed_input:
        posit_embs = self.position_emb.FPropWithPosition(
            theta.position_emb, target_segment_pos)
      else:
        posit_embs = tf.expand_dims(
            self.position_emb.FProp(theta.position_emb, target_time), 0)

      # [time, batch, model_dim]
      input_embs = token_embs + posit_embs

      atten_idx = None
      if p.task_emb:
        if p.use_lang_dependent_atten:
          atten_idx = targets.task_ids
          # Works for both packed and unpacked inputs.
          atten_idx = tf.reshape(tf.transpose(atten_idx), [-1])
        input_embs += self.task_emb.EmbLookup(theta.task_emb, targets.task_ids)

      if p.model_dim != self._token_emb_dim:
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
        extra_kwargs = dict()
        if isinstance(layer, layers_with_attention.TransformerWithContextLayer):
          # If the encoder contains encodings for the context and the
          # transformer layer in the decoder is able to attend to it, we pass
          # them to the transformer layer.
          extra_kwargs['tertiary_vecs'] = encoder_outputs.context_encoded
          extra_kwargs['tertiary_paddings'] = encoder_outputs.context_padding
        # [time, batch, model_dim]
        layer_out, probs = layer.FProp(
            layer_theta,
            layer_in,
            target_paddings,
            source_encs[i],
            source_paddings,
            source_segment_id=target_segment_id,
            aux_segment_id=src_segment_id,
            atten_idx=atten_idx,
            **extra_kwargs)
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

      if p.ln_output:
        layer_out = self.layer_norm_out.FProp(theta.layer_norm_out, layer_out)

      # per_layer_attn_probs shape: [batch, trg time, src time]
      self._AddAttenProbsSummary(source_paddings, targets, per_layer_attn_probs)

      # Aggregate per-layer attention probs.
      aggregated_atten_probs = (
          tf.math.add_n(per_layer_attn_probs) / len(per_layer_attn_probs))

      attention_map = py_utils.NestedMap(probs=aggregated_atten_probs)
      return py_utils.NestedMap(
          softmax_input=layer_out, attention=attention_map)

  def AddExtraDecodingInfo(self, encoder_outputs, targets):
    """Adds extra decoding information to encoded_outputs.

    Args:
      encoder_outputs: a NestedMap computed by encoder.
      targets: a NestedMap containing target input fields.

    Returns:
      encoder_ouputs with extra information used for decoding.
    """
    p = self.params
    if p.task_emb:
      encoder_outputs['target_task_ids'] = targets.task_ids[:, 0]
    if p.init_step_ids:
      encoder_outputs['init_step_ids'] = targets.ids[:, 0]
    return encoder_outputs

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
      # [batch, model_dim]
      if not self._share_sm_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, new_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, new_ids)

      if p.zero_token_embs_first_time_step:
        # For models that do not use an explicit start-of-sequence token
        # with associated embedding, but instead use zeros.
        zeros = tf.zeros_like(token_embs)
        token_embs = tf.cond(tf.equal(t, 0), lambda: zeros, lambda: token_embs)

      # [1, model_dim]
      posit_embs = tf.slice(
          self.position_emb.FProp(theta.position_emb, p.target_seq_len), [t, 0],
          [1, p.model_dim])
      input_embs = token_embs + posit_embs

      # Infer num_hyps_per_beam: new_ids has orig_batch_size * num_hyps_per_beam
      # source_paddings has orig_batch_size.
      num_hyps_per_beam = tf.div(
          py_utils.GetShape(new_ids)[0],
          py_utils.GetShape(source_paddings)[1])

      atten_idx = None
      if p.task_emb:
        task_ids = self._ExpandToNumHyps(encoder_outputs.target_task_ids,
                                         num_hyps_per_beam)
        if p.use_lang_dependent_atten:
          atten_idx = task_ids
        input_embs += self.task_emb.EmbLookup(theta.task_emb, task_ids)

      if p.model_dim != self._token_emb_dim:
        input_embs = self.emb_proj.FProp(theta.emb_proj, input_embs)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      # Make a copy of the input.
      out_prefix_states = prefix_states.Pack(prefix_states.Flatten())

      layer_in = input_embs

      # Infer true source encoder length from the padding.
      src_enc_len = tf.reduce_sum(1 - source_paddings, axis=0)

      # Need to expand src_enc_len to reflect multiple hypotheses.
      src_enc_len = self._ExpandToNumHyps(src_enc_len, num_hyps_per_beam)

      atten_probs = []
      for i, (layer, layer_theta) in enumerate(zip(self.trans, theta.trans)):
        extra_kwargs = dict()
        if isinstance(layer, layers_with_attention.TransformerWithContextLayer):
          # If the encoder contains encodings for the context and the
          # transformer layer in the decoder is able to attend to it, we pass
          # them to the transformer layer.
          extra_kwargs['tertiary_vecs'] = encoder_outputs.context_encoded
          extra_kwargs['tertiary_paddings'] = encoder_outputs.context_padding
        # [time, batch, model_dim]
        layer_prefix_states = prefix_states['layer_%i' % i]
        layer_out, probs, updated_prefix_states = layer.ExtendStep(
            layer_theta,
            layer_in,
            layer_prefix_states,
            source_encs[i],
            source_paddings,
            t=t if p.beam_search.name == 'tpu_beam_search' else None,
            atten_idx=atten_idx,
            **extra_kwargs)
        out_prefix_states['layer_%i' % i] = updated_prefix_states
        layer_in = layer_out
        # Enforce shape: [batch, src_len]
        probs = tf.squeeze(probs, [0])
        # Remove attention weight on last (EOS) token and re-normalize
        # so that last dimension sums to 1. See b/129097156.
        probs_3d = tf.expand_dims(probs, axis=1)
        probs_3d = self._RemoveEOSProbs(p, probs_3d, src_enc_len)
        probs = tf.squeeze(probs_3d, axis=1)

        atten_probs.append(probs)

      if p.ln_output:
        layer_out = self.layer_norm_out.FProp(theta.layer_norm_out, layer_out)

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
      A `.NestedMap` containing output of last decoder layer and attention probs

      - softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
      - attention: `.NestedMap` of attention distributions of shape
        [batch, time, source_len].
    """
    return self._FProp(theta, encoder_outputs, targets)

  def SampleSequenceDecode(self, encoder_outputs):
    """Decode via sampling from softmax at each step.

    Args:
      encoder_outputs: the outputs of the encoder.

    Returns:
      BeamSearchDecodeOutput, same as what BeamSearchDecode returns.
    """
    p = self.params
    non_tpu = p.beam_search.name != 'tpu_beam_search'

    def InitCallback(theta, encoder_outputs, num_hyps_per_beam=1):
      """Wrapper for _InitBeamSearchStateCallback for sequence sampler.

      The main change is to ensure state tensors have fixed shapes.

      Args:
        theta: A `.NestedMap` object containing weights' values of this layer
          and its children layers.
        encoder_outputs: a NestedMap computed by encoder.
        num_hyps_per_beam: An int, number hyps to keep for source sentence.

      Returns:
        A NestedMap of

          - initial_results: a `.NestedMap` of initial results.
          - states: a `.NestedMap` of initial model states.
      """
      init_results, states = self._InitBeamSearchStateCallback(
          theta, encoder_outputs, num_hyps_per_beam)
      if non_tpu:
        prefix_states = states['prefix_states']
        for layer in range(p.num_trans_layers):
          key = prefix_states['layer_%d' % layer]['key']
          value = prefix_states['layer_%d' % layer]['value']
          key_shapes = py_utils.GetShape(key)
          bs = key_shapes[1]
          atten_dim = key_shapes[2]
          zeros = tf.zeros([p.target_seq_len, bs, atten_dim],
                           dtype=py_utils.FPropDtype(p))
          prefix_states['layer_%d' % layer]['key'] = tf.concat([key, zeros], 0)
          prefix_states['layer_%d' % layer]['value'] = tf.concat([value, zeros],
                                                                 0)
      return init_results, states

    def PreBeamSearchCallback(theta,
                              encoder_outputs,
                              step_ids,
                              states,
                              num_hyps_per_beam=1):
      """Wrapper for _PreBeamSearchStepCallback for sequence sampler.

      The main change is to ensure state tensors have fixed shapes.

      Args:
        theta: A `.NestedMap` object containing weights' values of this layer
          and its children layers.
        encoder_outputs: a NestedMap computed by encoder.
        step_ids: A tensor of shape [tgt_batch, 1].
        states: A `.NestedMap` of tensors representing states that the clients
          would like to keep track of for each of the active hyps.
        num_hyps_per_beam: Beam size.

      Returns:
        A NestedMap of

          - results: A `.NestedMap` of beam search results.
          - out_states: A `.NestedMap`. The updated states.
      """

      if non_tpu:
        # Strip off paddings.
        prefix_states = states['prefix_states']
        target_time = states.time_step
        for layer in range(p.num_trans_layers):
          key = prefix_states['layer_%d' % layer]['key']
          val = prefix_states['layer_%d' % layer]['value']
          prefix_states['layer_%d' % layer]['key'] = tf.slice(
              key, [0, 0, 0], [target_time, -1, -1])
          prefix_states['layer_%d' % layer]['value'] = tf.slice(
              val, [0, 0, 0], [target_time, -1, -1])

      bs_results, new_states = self._PreBeamSearchStepCallback(
          theta, encoder_outputs, step_ids, states, num_hyps_per_beam)

      if non_tpu:
        # Add back paddings (to maintain paddings shape).
        bs = tf.shape(new_states.prefix_states['layer_0']['key'])[1]
        dim = tf.shape(new_states.prefix_states['layer_0']['key'])[2]
        pad = tf.zeros([p.target_seq_len - new_states.time_step, bs, dim],
                       dtype=py_utils.FPropDtype(p))
        for layer in range(p.num_trans_layers):
          key = new_states.prefix_states['layer_%d' % layer]['key']
          val = new_states.prefix_states['layer_%d' % layer]['value']
          new_states.prefix_states['layer_%d' % layer]['key'] = tf.concat(
              [key, pad], axis=0)
          new_states.prefix_states['layer_%d' % layer]['value'] = tf.concat(
              [val, pad], axis=0)

      return bs_results, new_states

    random_seed = tf.random.uniform(
        shape=[], maxval=(2**31 - 1), dtype=tf.int32, seed=p.random_seed)
    sample = self.target_sequence_sampler.Sample(
        self.theta, encoder_outputs, random_seed, InitCallback,
        PreBeamSearchCallback, self._PostBeamSearchStepCallback)
    bs = tf.shape(sample.ids)[0]
    # Only need to make sure topk_hyps has the right shape
    # [bs, num_hyps_per_beam], where num_hyps_per_beam=1 for sampling.
    # TODO(yuancao): Support sampling multiple sequences and remove
    # num_hyps_per_beam constraint.
    assert self.params.beam_search.num_hyps_per_beam == 1
    sample.topk_hyps = tf.zeros([bs, 1], dtype=tf.string)
    sample.topk_ids = sample.ids
    weights = 1 - sample.paddings
    sample.topk_lens = tf.cast(tf.reduce_sum(weights, axis=1), dtype=tf.int32)
    sample.topk_scores = tf.reduce_sum(
        tf.math.log(tf.reduce_max(tf.nn.softmax(sample.logits), axis=2)) *
        weights,
        axis=1)
    return sample

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
    atten_probs = tf.ones([num_hyps, source_len]) / tf.cast(
        source_len, tf.float32)
    initial_results = py_utils.NestedMap(
        log_probs=tf.zeros([num_hyps, p.softmax.num_classes],
                           dtype=py_utils.FPropDtype(p)),
        atten_probs=atten_probs)

    if p.init_step_ids:
      initial_results['step_ids'] = tf.expand_dims(
          self._ExpandToNumHyps(encoder_outputs.init_step_ids,
                                num_hyps_per_beam), 1)

    batch_size = num_hyps
    if isinstance(p.trans_tpl, list):
      atten_hidden_dim = p.trans_tpl[0].tr_atten_tpl.atten_hidden_dim
      assert [tpl.tr_atten_tpl.atten_hidden_dim for tpl in p.trans_tpl
             ].count(atten_hidden_dim) == len(
                 p.trans_tpl), 'atten_hidden_dim must match'
    else:
      atten_hidden_dim = p.trans_tpl.tr_atten_tpl.atten_hidden_dim

    if not atten_hidden_dim:
      atten_hidden_dim = p.model_dim

    if p.beam_search.name == 'tpu_beam_search':
      seq_len = p.target_seq_len
    else:
      seq_len = 0

    prefix_states = py_utils.NestedMap()
    for layer in range(p.num_trans_layers):
      prefix_states['layer_%d' % layer] = py_utils.NestedMap({
          'key':
              tf.zeros([seq_len, batch_size, atten_hidden_dim],
                       dtype=py_utils.FPropDtype(p)),
          'value':
              tf.zeros([seq_len, batch_size, atten_hidden_dim],
                       dtype=py_utils.FPropDtype(p)),
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
    super()._AddAttenProbsSummary(source_paddings, targets, atten_probs)
    if self.cluster.add_summary and self.params.add_multiheaded_attention_scalar_summary:
      self._AddAttenProbsScalarSummary(source_paddings, targets, atten_probs)


class InsertionDecoder(base_decoder.BaseBeamSearchDecoder):
  """Basic Insertion decoder for MT (or any symbol based sequence).

  References:
    KERMIT: https://arxiv.org/pdf/1906.01604.pdf
    Insertion Transformer: https://arxiv.org/pdf/1902.03249.pdf
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('token_emb', layers.EmbeddingLayer.Params(),
             'Token embedding layer params.')
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define(
        'model_dim', 1024, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    p.Define('num_trans_layers', 6, 'Number of Transformer layers.')
    p.Define('trans_tpl', layers_with_attention.TransformerLayer.Params(),
             'Transformer layer params.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(), 'Softmax params.')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')

    # Default config for the token embeddings.
    p.token_emb.vocab_size = 32000 * 2
    p.token_emb.embedding_dim = p.model_dim
    p.token_emb.max_num_shards = 16
    p.token_emb.params_init = py_utils.WeightInit.Gaussian(
        1.0 / math.sqrt(p.token_emb.embedding_dim))
    p.token_emb.scale_sqrt_depth = True

    # Default config for the position embeddings.
    p.position_emb.embedding_dim = p.model_dim

    # Default config for the transformer layers.
    p.trans_tpl.source_dim = p.model_dim
    p.trans_tpl.tr_atten_tpl.source_dim = p.model_dim
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 8
    p.trans_tpl.tr_fflayer_tpl.input_dim = p.model_dim
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 4096

    # Default config for the softmax.
    p.softmax.num_classes = 32000
    p.softmax.num_shards = 8

    p.target_seq_len = 300

    return p

  @classmethod
  def UpdateTargetVocabSize(cls, p, vocab_size, wpm_model=None):
    """Sets the vocab size in the params.

    Args:
      p: model params.
      vocab_size: size of the vocabulary.
      wpm_model: file name prefix pointing to a wordpiece model.

    Returns:
      Model params updated with the vocab size and wpm model.
    """
    p.softmax.num_classes = vocab_size
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.token_emb.vocab_size % p.softmax.num_classes == 0
    assert p.token_emb.embedding_dim == p.position_emb.embedding_dim
    assert p.token_emb.embedding_dim == p.model_dim

    self.CreateChild('token_emb', p.token_emb)
    self.CreateChild('position_emb', p.position_emb)

    dropout_tpl = layers.DropoutLayer.Params()
    dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
    self.CreateChild('input_dropout', dropout_tpl)

    params_trans_layers = []
    for i in range(p.num_trans_layers):
      params = p.trans_tpl.Copy()
      params.name = 'trans_layer_%d' % i
      params.packed_input = p.packed_input
      params.has_aux_atten = False
      params.mask_self_atten = True
      params_trans_layers.append(params)
    self.CreateChildren('trans', params_trans_layers)

    p.softmax.input_dim = p.model_dim
    self.CreateChild('softmax', p.softmax)

  def ComputePredictions(self, theta, encoder_outputs, targets):
    """Compute 1-step of the insertion iteration.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: This should be None.
      targets: A `.NestedMap`.
        - ids: The target ids of shape [batch_size, time_dim].
        - paddings: The target paddings of shape [batch_size, time_dim].

    Returns:
      A `.NestedMap`.
        - outputs: The contextualized output vectors of shape
          [batch_size, time_dim, model_dim].
    """
    p = self.params

    # TODO(williamchan): Enable cross-attention.
    assert encoder_outputs is None

    with tf.name_scope(p.name):
      # [batch, time]
      target_ids = targets.ids
      # [time, batch]
      target_paddings = tf.transpose(targets.paddings)

      # Embedding layer
      # [batch, time, model_dim]
      token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      target_time = py_utils.GetShape(target_ids)[1]

      # [1, time, model_dim]
      posit_embs = tf.expand_dims(
          self.position_emb.FProp(theta.position_emb, target_time), 0)

      # [time, batch, model_dim]
      input_embs = token_embs + posit_embs

      input_embs = tf.transpose(input_embs, [1, 0, 2])
      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

      layer_in = input_embs
      for layer, layer_theta in zip(self.trans, theta.trans):
        # [time, batch, model_dim]
        layer_out, _ = layer.FProp(layer_theta, layer_in, target_paddings)
        layer_in = layer_out

      return py_utils.NestedMap(outputs=layer_out)

  def ComputeLoss(self, theta, predictions, targets):
    # pyformat: disable
    """Returns the insertion loss.

    Args:
      theta: A `.NestedMap` object capturing decoder model parameters.
      predictions: A `.NestedMap` describing the decoding process, requiring
        .outputs: Tensor of shape [time, batch, params.softmax.input_dim].
      targets: A `.NestedMap`.

        - target_indices: A Tensor capturing the relevant insertion tokens to
          tf.gather_nd the log-probs.

        - target_weights: A Tensor capturing the relevant insertion tokens'
          weights.

    Returns:
      Two dicts.
        - A map from metric name (a python string) to a tuple (value, weight).
          Both value and weight are scalar Tensors.
        - A map from name to arbitrary tensors, where the first dimension must
          be the batch index.
    """
    # pyformat: enable
    p = self.params

    batch_size = py_utils.GetShape(predictions.outputs)[0]

    state = tf.reshape(predictions.outputs, [-1, p.softmax.input_dim])
    logits = self.softmax.Logits(theta.softmax, state)
    logits = tf.reshape(
        logits,
        tf.concat([
            py_utils.GetShape(predictions.outputs)[:-1],
            [p.softmax.num_classes]
        ], 0))
    log_probs = tf.nn.log_softmax(logits)

    # `target_indices` are in the form [batch, time, vocab], where as `logits`
    # are in the form [time, batch, vocab]. We need to swap the columns.
    target_indices = tf.concat([
        predictions.tgt.target_indices[:, 1:2],
        predictions.tgt.target_indices[:, 0:1],
        predictions.tgt.target_indices[:, 2:3],
    ], 1)

    loss = tf.reduce_sum(
        tf.gather_nd(log_probs, target_indices) *
        predictions.tgt.target_weights)
    loss_weight = tf.cast(batch_size, tf.float32)

    return ({
        'loss': (loss, loss_weight)
    }, {
        'log_probs': log_probs,
        'logits': logits
    })


class TransformerBatchMajorDecoder(MTBaseDecoder):
  """Transformer decoder with batch major implementation.

  Implements the decoder of Transformer model:
  https://arxiv.org/abs/1706.03762.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('token_emb', layers.EmbeddingLayer.Params(),
             'Token embedding layer params.')
    p.Define('shared_emb', None, 'Embedding shared with softmax.')
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define('source_dim', 1024, 'Dimension of encoder outputs.')
    p.Define(
        'model_dim', 1024, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    p.Define('num_trans_layers', 6, 'Number of Transformer layers.')
    p.Define(
        'trans_decoder_tpl',
        batch_major_attention.TransformerDecoderLayer.Params(),
        'Transformer layer params. This can be a list of params '
        'of length equal to num_trans_layers or a factor of it, '
        'in which case the params are tiled as [a, a, ..., b, b, ...]')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define('input_dropout_tpl', layers.DropoutLayer.Params(),
             'Input dropout layer params.')
    p.Define('final_layer_norm', False,
             'Whether or not to apply layer norm after transformer stack.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm.')
    p.Define('use_fast_softmax', False,
             'Whether or not to use a faster softmax with label smoothing.')
    p.Define(
        'input_data_format', 'TBC', 'The data format of input features: '
        'TBC for [time, batch, feature_dim], '
        'BTC for [batch, time, feature_dim].')
    p.Define(
        'prediction_data_format', 'TBC',
        'The data format of predictions and per-example losses: '
        'TBC for [time, batch, ...], '
        'BTC for [batch, time, ...].')

    # Default config for the token embedding.
    p.token_emb.vocab_size = 32000
    p.token_emb.embedding_dim = p.model_dim
    p.token_emb.max_num_shards = 16
    p.token_emb.params_init = py_utils.WeightInit.Gaussian(
        1.0 / math.sqrt(p.token_emb.embedding_dim))
    p.token_emb.scale_sqrt_depth = True

    # Default config for the position embedding.
    p.position_emb.embedding_dim = p.model_dim

    # Default config for the transformer decoder layers.
    p.trans_decoder_tpl.input_dim = p.model_dim
    p.trans_decoder_tpl.tr_atten_tpl.input_dim = p.model_dim
    p.trans_decoder_tpl.tr_atten_tpl.num_heads = 8
    p.trans_decoder_tpl.tr_fflayer_tpl.input_dim = p.model_dim
    p.trans_decoder_tpl.tr_fflayer_tpl.hidden_dim = 2048

    # Default config for beam search.
    p.target_seq_len = 300
    p.beam_search.length_normalization = 0.5
    p.beam_search.coverage_penalty = 0.0
    p.beam_search.batch_major_state = False
    p.beam_search.batch_major_compute = True
    p.beam_search.short_seq_limit = 40

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if p.shared_emb:
      self.CreateChild('softmax', p.shared_emb)

    if not p.shared_emb:
      self.CreateChild('token_emb', p.token_emb)
    self.CreateChild('position_emb', p.position_emb)

    dropout_tpl = p.input_dropout_tpl.Copy()
    dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
    self.CreateChild('input_dropout', dropout_tpl)

    if isinstance(p.trans_decoder_tpl, list):
      if p.num_trans_layers % len(p.trans_decoder_tpl):
        raise ValueError('num_trans_layers should be divisible by '
                         'len(p.trans_decoder_tpl)')

    params_trans_layers = []
    for i in range(p.num_trans_layers):
      if isinstance(p.trans_decoder_tpl, list):
        idx = i // len(p.trans_decoder_tpl)
        params = p.trans_decoder_tpl[idx].Copy()
        params.packed_input = p.packed_input
      else:
        params = p.trans_decoder_tpl.Copy()
        params.packed_input = p.packed_input
      params.name = 'decoder_trans_layer_%d' % i
      params_trans_layers.append(params)
    self.CreateChildren('decoder_trans', params_trans_layers)

    p.softmax.input_dim = p.model_dim
    if not p.shared_emb:
      self.CreateChild('softmax', p.softmax)

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

  def _MaybeTransposeEncoderOutputs(self, encoder_outputs, target_data_format):
    p = self.params
    if p.input_data_format == target_data_format:
      return encoder_outputs
    transposed = py_utils.NestedMap(
        encoded=tf.transpose(encoder_outputs.encoded, [1, 0, 2]),
        padding=tf.transpose(encoder_outputs.padding))
    if getattr(encoder_outputs, 'segment_id', None) is None:
      transposed.segment_id = None
    else:
      transposed.segment_id = tf.transpose(encoder_outputs.segment_id)
    return transposed

  def _MaybeTransposeTargets(self, targets):
    p = self.params
    if p.prediction_data_format == 'BTC':
      return targets
    transposed = py_utils.NestedMap()
    for k, v in targets.items():
      if v is not None and k != 'transcripts' and k != 'strs':
        with tf.name_scope('transpose_%s' % k):
          v = tf.transpose(py_utils.HasShape(v, [-1, -1]))
      transposed[k] = v
    return transposed

  def _FProp(self, theta, encoder_outputs, targets):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder. * encoded -
        Source encoding of shape [source_time, source_batch, dim] or
        [source_batch, source_time, dim], depending on p.input_data_format. *
        paddings - Source encoding's padding of shape [source_time,
        source_batch] or [source_batch, source_time].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, target_time].

    Returns:
      softmax_input: Tensor of shape [target_time, batch, dim].
    """
    p = self.params
    # [batch, source_time, dim]
    encoder_out_bm = self._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')
    aux_vec = encoder_out_bm.encoded
    aux_paddings = encoder_out_bm.padding
    aux_segment_id = getattr(encoder_out_bm, 'segment_id', None)

    with tf.name_scope(p.name):
      # [batch, target_time]
      target_ids = targets.ids
      target_paddings = targets.paddings
      target_time = py_utils.GetShape(target_ids)[1]
      target_segment_pos = None
      target_segment_id = None
      if p.packed_input:
        target_segment_id = targets.segment_ids
        target_segment_pos = targets.segment_pos
        assert aux_segment_id is not None, ('Need to provide aux_segment_id '
                                            'for packed input.')

      # Embedding layer
      # [batch, target_time, dim]
      if not p.shared_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, target_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, target_ids)
      # [1, target_time, dim]
      if p.packed_input:
        posit_embs = self.position_emb.FPropWithPosition(
            theta.position_emb, target_segment_pos)
      else:
        posit_embs = tf.expand_dims(
            self.position_emb.FProp(theta.position_emb, target_time), 0)
      # [batch, target_time, dim]
      input_embs = token_embs + posit_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)
        target_paddings = tf.cast(target_paddings,
                                  p.input_dropout_tpl.fprop_dtype)

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      layer_in = input_embs
      # Explicitly set the input shape of Transformer layers, to avoid
      # unknown shape error occurred to tf.einsum on nonTPU devices.
      batch, _, dim = py_utils.GetShape(aux_vec, 3)
      layer_in = tf.reshape(layer_in, [batch, target_time, dim])
      if p.packed_input:
        segment_padding = batch_major_attention.SegmentMask(
            target_segment_id,
            target_segment_id,
            dtype=layer_in.dtype,
            apply_dtype_min=False)
        causal_padding = tf.expand_dims(
            tf.tile(
                tf.expand_dims(
                    batch_major_attention.CausalPadding(
                        target_time, dtype=layer_in.dtype), 0), [batch, 1, 1]),
            1)
        segment_padding = tf.math.maximum(causal_padding, segment_padding)
        segment_mask = segment_padding * batch_major_attention.GetDtypeMin(
            dtype=layer_in.dtype)
        aux_segment_mask = batch_major_attention.SegmentMask(
            target_segment_id, aux_segment_id, dtype=layer_in.dtype)
      for layer, layer_theta in zip(self.decoder_trans, theta.decoder_trans):
        # [batch, target_time, dim]
        shape = py_utils.GetShape(layer_in)
        batch_size = shape[0]
        seq_len = shape[1]
        target_paddings = tf.reshape(target_paddings, [batch_size, seq_len])
        layer_out, _ = layer.FProp(
            layer_theta,
            layer_in,
            target_paddings,
            aux_vec,
            aux_paddings,
            segment_mask=segment_mask if p.packed_input else None,
            aux_segment_mask=aux_segment_mask if p.packed_input else None)
        layer_in = layer_out

      if p.final_layer_norm:
        layer_out = self.final_ln.FProp(theta.final_ln, layer_out)
      if p.prediction_data_format == 'TBC':
        # Transpose the softmax_input to match the input requirement of
        # ComputePredictions.
        layer_out = tf.transpose(layer_out, [1, 0, 2])
      return layer_out

  def ExtendStep(self,
                 theta,
                 encoder_outputs,
                 new_ids,
                 time_step,
                 prefix_states,
                 use_short_seq_opt=False):
    """Extend prefix as represented by `prefix_states` by one more step.

    This function is expected to be called during fast decoding of Transformer
    models.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder.

        - encoded: Source encoding of shape [source_time, source_batch, dim] or
          [source_batch, source_time, dim], depending on p.input_data_format.
        - paddings: Source encoding's padding of shape
          [source_time, source_batch] or [source_batch, source_time].
      new_ids: New input ids, of shape [target_batch, 1].
      time_step: A scalar, the current decode step, 0-based.
      prefix_states: A `.NestedMap` representing the previous decoded states.

        - key: [target_time, target_batch, num_heads, dim_per_head].
        - value: [target_time, target_batch, num_heads, dim_per_head].
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      last_decoder_out: The last decoder layer of shape [target_batch, dim].
      updated_prefix_states: A `.NestedMap` representing the updated states.

        - key: [target_time, target_batch, num_heads, dim_per_head].
        - value: [target_time, target_batch, num_heads, dim_per_head].
    """
    p = self.params
    encoder_out_bm = self._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')
    # [source_batch, source_time, dim]
    aux_vec = encoder_out_bm.encoded
    # [source_batch, source_time]
    aux_paddings = encoder_out_bm.padding

    with tf.name_scope(p.name):
      # Embedding layer
      # [target_batch, 1, dim]
      if not p.shared_emb:
        token_embs = self.token_emb.EmbLookup(theta.token_emb, new_ids)
      else:
        token_embs = self.softmax.EmbLookup(theta.softmax, new_ids)
      # [1, 1, dim]
      if isinstance(time_step, tf.Tensor):
        time_step_t = tf.reshape(time_step, [1, 1])
      elif isinstance(time_step, int):
        time_step_t = tf.constant([[time_step]], dtype=tf.int32)
      else:
        raise ValueError('Unexpected input type `%s` for `time_step`.' %
                         type(time_step))
      posit_embs = self.position_emb.FPropWithPosition(theta.position_emb,
                                                       time_step_t)
      # [target_batch, 1, dim]
      input_embs = token_embs + posit_embs

      if p.input_dropout_tpl.fprop_dtype:
        input_embs = tf.cast(input_embs, p.input_dropout_tpl.fprop_dtype)

      # Make a copy of the input.
      updated_prefix_states = prefix_states.DeepCopy()

      input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
      layer_in = input_embs
      for i, (layer, layer_theta) in enumerate(
          zip(self.decoder_trans, theta.decoder_trans)):
        # [target_batch, 1, dim]
        layer_out, _, updated_states = layer.ExtendStep(
            layer_theta, layer_in, aux_vec, aux_paddings,
            prefix_states['layer_%i' % i], time_step, use_short_seq_opt)
        updated_prefix_states['layer_%i' % i] = updated_states
        layer_in = layer_out

      # [target_batch, dim]
      last_decoder_out = tf.squeeze(layer_out, 1)
      if p.final_layer_norm:
        last_decoder_out = self.final_ln.FProp(theta.final_ln, last_decoder_out)
      return last_decoder_out, updated_prefix_states

  def ComputePredictions(self, theta, encoder_outputs, targets):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder.

        - encoded: Source encoding of shape [source_time, source_batch, dim] or
          [source_batch, source_time, dim], depending on p.input_data_format.
        - paddings: Source encoding's padding of shape
          [source_time, source_batch] or [source_batch, source_time].
      targets: A dict of string to tensors representing the targets one try to
        predict. Each tensor in targets is of shape [batch, target_time].

    Returns:
      Output of the last decoder layer, of shape [target_time, batch, dim].
    """
    return self._FProp(theta, encoder_outputs, targets)

  def _FPropFastSoftmax(self,
                        theta,
                        softmax_input,
                        target_labels,
                        target_weights,
                        time_axis=0):
    """Computes cross-entropy loss with label smoothing.

    As compared to the _FPropSoftmax, this version is faster by removing the
    data formatting overheads and bias of the linear projection. A normalizing
    factor is also added to the xentropy result be better model quality.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      softmax_input: A tensor of shape [time, batch, p.softmax.input_dim].
      target_labels: A matrix of tf.int32. [time, batch].
      target_weights: A matrix of params.dtype. [time, batch].
      time_axis: If 0, the inputs are time-major: [time, batch, ...]; if 1, the
        inputs are batch-major: [batch, time, ...].

    Returns:
      A tuple (metrics, per_example_tensors).
        metrics:
          A dictionary containing metrics for the xent loss and prediction
          accuracy.
        per_example_tensors:
          A dictionary of per-example tensors.
    """
    p = self.params
    assert p.label_smoothing is not None
    assert p.per_word_avg_loss

    softmax_input = tf.reshape(softmax_input, [-1, p.softmax.input_dim])

    logits = self.softmax.SimpleLogits(theta.softmax, softmax_input)
    logits = tf.cast(logits, tf.float32)

    high_confidence = 1.0 - p.label_smoothing.uncertainty
    low_confidence = p.label_smoothing.uncertainty / tf.cast(
        p.label_smoothing.num_classes - 1, tf.float32)
    normalizing = -(
        high_confidence * tf.math.log(high_confidence) +
        tf.cast(p.softmax.num_classes - 1, tf.float32) * low_confidence *
        tf.math.log(low_confidence + 1e-20))

    target_labels = tf.reshape(target_labels, [-1])
    soft_targets = tf.one_hot(
        tf.cast(target_labels, tf.int32),
        depth=p.softmax.num_classes,
        on_value=high_confidence,
        off_value=low_confidence)

    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=soft_targets)
    xent = xentropy - normalizing

    target_weights_shape = py_utils.GetShape(target_weights)
    orig_target_weights = target_weights
    target_weights = tf.cast(tf.reshape(target_weights, [-1]), xent.dtype)
    total_xent = tf.reduce_sum(xent * target_weights)
    total_weights = tf.reduce_sum(target_weights)

    final_loss = total_xent / total_weights
    loss_weight = total_weights

    metrics = {
        'loss': (final_loss, loss_weight),
        'log_pplx': (final_loss, loss_weight),
    }

    per_example_tensors = {}
    if p.per_example_tensors:
      per_example_tensors['per_example_loss'] = tf.reshape(
          xent, target_weights_shape)
      per_example_tensors['per_sequence_loss'] = tf.reduce_sum(
          per_example_tensors['per_example_loss'] * orig_target_weights,
          axis=time_axis)
      per_example_tensors['loss'] = per_example_tensors['per_sequence_loss']
      per_example_tensors['logits'] = tf.reshape(
          logits, tf.concat([target_weights_shape, [-1]], 0))
      per_example_tensors['log_probs'] = tf.reshape(
          tf.nn.log_softmax(logits), tf.concat([target_weights_shape, [-1]], 0))

    # NOTE: tf.argmax is not implemented for the JF backend, see b/36093673
    # Skip the fraction_of_correct_next_step_preds during training.
    if self.do_eval:
      correct_preds = tf.cast(
          tf.equal(
              tf.cast(tf.reshape(tf.argmax(logits, 1), [-1]), tf.int32),
              tf.reshape(target_labels, [-1])), p.dtype)
      correct_next_preds = tf.reduce_sum(
          correct_preds * tf.reshape(tf.cast(target_weights, p.dtype), [-1]))
      num_preds = tf.reduce_sum(tf.cast(target_weights, p.dtype))
      accuracy = tf.identity(
          correct_next_preds / num_preds,
          name='fraction_of_correct_next_step_preds')
      metrics['fraction_of_correct_next_step_preds'] = (accuracy, num_preds)
    return metrics, per_example_tensors

  def ComputeLoss(self, theta, predictions, targets):
    """Populates a metrics dictionary based on the output of ComputePredictions.

    Args:
      theta: Nested map describing decoder model parameters.
      predictions: NestedMap describing the decoding process, requiring:
        .softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
      targets: NestedMap describing the target sequences.

    Returns:
      Two dicts.

        - A map from metric name (a python string) to a tuple (value, weight).
          Both value and weight are scalar Tensors.
        - A map from name to arbitrary tensors, where the first dimension must
          be the batch index.
    """
    p = self.params
    targets = self._MaybeTransposeTargets(targets)
    if isinstance(predictions, py_utils.NestedMap):
      predictions = predictions.softmax_input
    time_axis = {'TBC': 0, 'BTC': 1}.get(p.prediction_data_format)
    if p.use_fast_softmax:
      return self._FPropFastSoftmax(
          theta,
          predictions,
          targets.labels,
          targets.weights,
          time_axis=time_axis)
    else:
      return self._FPropSoftmax(
          theta,
          predictions,
          targets.labels,
          targets.weights,
          targets.paddings,
          targets.get('segment_ids', None),
          time_axis=time_axis)

  def _InitBeamSearchStateCallback(self, theta, encoder_outputs,
                                   num_hyps_per_beam):
    """Returns initial beams search states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder. * encoded -
        Source encoding of shape [source_time, source_batch, dim] or
        [source_batch, source_time, dim], depending on p.input_data_format. *
        paddings - Source encoding's padding of shape [source_time,
        source_batch] or [source_batch, source_time].
      num_hyps_per_beam: An int, number hyps to keep for source sentence.

    Returns:
      initial_results: A `.NestedMap` of initial beam search results.
        log_probs - Log prob for each of the tokens in the target vocab,
                    of shape [target_batch, vocab_size].
        atten_probs - The updated attention probs, of shape
                      [target_batch, source_time].
      states: A `.NestedMap` of initial model states.
        prefix_states - A `.NestedMap` representing the empty decoded states.
        key   - [target_time, target_batch, num_heads, dim_per_head].
        value - [target_time, target_batch, num_heads, dim_per_head].
        time_step - A scalar, the initial decode step (0).
    """
    p = self.params

    # [source_batch, source_time, dim]
    encoder_out_bm = self._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')
    aux_vec = encoder_out_bm.encoded
    target_batch = py_utils.GetShape(aux_vec)[0] * num_hyps_per_beam
    source_time = py_utils.GetShape(aux_vec)[1]
    target_time = p.target_seq_len

    log_probs = tf.zeros([target_batch, p.softmax.num_classes],
                         dtype=py_utils.FPropDtype(p))
    # Dummy attention probs
    atten_probs = (
        tf.ones([target_batch, source_time], dtype=py_utils.FPropDtype(p)) /
        tf.cast(source_time, py_utils.FPropDtype(p)))
    initial_results = py_utils.NestedMap(
        log_probs=log_probs, atten_probs=atten_probs)

    prefix_states = py_utils.NestedMap()
    for layer in range(p.num_trans_layers):
      prefix_states['layer_%d' % layer] = self.decoder_trans[layer].InitStates(
          theta.decoder_trans[layer], target_batch, target_time)

    return initial_results, py_utils.NestedMap({
        'prefix_states': prefix_states,
        'time_step': tf.constant(0)
    })

  def _PreBeamSearchStepCallback(self,
                                 theta,
                                 encoder_outputs,
                                 new_ids,
                                 states,
                                 num_hyps_per_beam,
                                 use_short_seq_opt=False):
    """Returns logits for sampling ids and the next model states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: A '.NestedMap' object computed by encoder.

        - encoded: Source encoding of shape [source_time, source_batch, dim] or
          [source_batch, source_time, dim], depending on p.input_data_format.
        - paddings: Source encoding's padding of shape
          [source_time, source_batch] or [source_batch, source_time].
      new_ids: A tensor of shape [target_batch, 1].
      states: A `.NestedMap` of tensors representing states that the clients
        would like to keep track of for each of the active hyps. prefix_states -
        A `.NestedMap` representing the previous decoded states.

          - key: [target_time, target_batch, num_heads, dim_per_head].
          - value: [target_time, target_batch, num_heads, dim_per_head].
          - time_step: A scalar, the current decode step, 0-based.
      num_hyps_per_beam: A scalar, beam size.
      use_short_seq_opt: A bool, whether using short sequence optimization.

    Returns:
      bs_results: A `.NestedMap` of beam search results.
        log_probs - Log prob for each of the tokens in the target vocab,
                    of shape [target_batch, vocab_size].
        atten_probs - The updated attention probs, of shape
                      [target_batch, source_time].
      new_states: A `.NestedMap` object. The updated states.
        prefix_states - A `.NestedMap` representing the updated decoded states.
        key   - [target_time, target_batch, num_heads, dim_per_head].
        value - [target_time, target_batch, num_heads, dim_per_head].
        time_step - A scalar, the current decode step, 0-based.
    """
    p = self.params
    # [source_batch, source_time, dim]
    encoder_out_bm = self._MaybeTransposeEncoderOutputs(encoder_outputs, 'BTC')

    target_batch = py_utils.GetShape(new_ids)[0]
    source_batch = target_batch // num_hyps_per_beam

    new_states = states.Pack(states.Flatten())
    time_step = states.time_step
    prefix_states = states.prefix_states

    # The inputs are ordered as num_hyps_per_beam by num_beams,
    # which needs to be transposed for the layer computation.
    # [num_hyps_per_beam, source_batch, 1]
    new_ids = tf.reshape(new_ids, [num_hyps_per_beam, source_batch, 1])
    # [source_batch, num_hyps_per_beam, 1]
    new_ids = tf.transpose(new_ids, [1, 0, 2])
    # [source_batch * num_hyps_per_beam, 1]
    new_ids = tf.reshape(new_ids, [-1, 1])

    softmax_input, updated_prefix_states = self.ExtendStep(
        theta, encoder_outputs, new_ids, time_step, prefix_states,
        use_short_seq_opt)

    # Transpose the outputs as num_beams by num_hyps_per_beam to match the
    # beam search requirement.
    # [source_batch, num_hyps_per_beam, dim]
    softmax_input = tf.reshape(softmax_input,
                               [source_batch, num_hyps_per_beam, -1])
    # [num_hyps_per_beam, source_batch, dim]
    softmax_input = tf.transpose(softmax_input, [1, 0, 2])
    # [num_hyps_per_beam * source_batch, dim]
    softmax_input = tf.reshape(softmax_input, [target_batch, -1])

    # [target_batch, vocab_size]
    logits = self.softmax.Logits(theta.softmax, [softmax_input])

    # Only return logits for the last ids
    log_probs = tf.nn.log_softmax(logits)

    # Dummy attention probs
    source_time = py_utils.GetShape(encoder_out_bm.padding)[1]
    atten_probs = (
        tf.ones([target_batch, source_time], dtype=py_utils.FPropDtype(p)) /
        tf.cast(source_time, py_utils.FPropDtype(p)))

    bs_results = py_utils.NestedMap({
        'log_probs': log_probs,
        'atten_probs': atten_probs,
    })

    new_states.prefix_states = updated_prefix_states
    new_states.time_step = time_step + 1

    return bs_results, new_states

  def _PostBeamSearchStepCallback(self, theta, encoder_outputs, new_step_ids,
                                  states):
    # There is nothing to do here.
    return states
