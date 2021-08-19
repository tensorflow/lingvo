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
"""LM models."""

import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.tasks.lm import layers


class LanguageModel(base_model.BaseTask):
  """Language model."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('lm', layers.RnnLm.Params(), 'LM layer.')
    p.Define('packed_input', False, 'Whether the inputs are packed.')

    tp = p.train
    tp.Define(
        'max_lstm_gradient_norm', 0.0,
        'Clip gradient for vars in lstm layers by setting this value to '
        'something > 0.')
    tp.Define(
        'sum_loss_across_tokens_in_batch', False,
        'Sum the logP across predicted tokens in batch when set to True; '
        'average across predicted tokens in batch o/w (default).')

    tp.lr_schedule = schedule.PiecewiseConstantSchedule.Params().Set(
        boundaries=[350000, 500000, 600000], values=[1.0, 0.1, 0.01, 0.001])
    tp.vn_start_step = 20000
    tp.vn_std = 0.0
    tp.learning_rate = 0.001
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    assert p.lm.vocab_size == p.input.tokenizer.vocab_size, (
        'lm.vocab_size does not match input.tokenizer.vocab_size: %d vs %d' %
        (p.lm.vocab_size, p.input.tokenizer.vocab_size))

    # Construct the model.
    lm_p = p.lm.Copy().Set(packed_input=p.packed_input)
    self.CreateChild('lm', lm_p)

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
    p.lm = p.lm.cls.UpdateTargetVocabSize(p.lm, vocab_size, wpm_model=wpm_model)
    return p

  def _TrimIfPossibleThenTranspose(self, input_batch):
    paddings = input_batch['paddings']
    max_seq_len = None
    if not py_utils.use_tpu():
      max_seq_len = tf.cast(
          tf.reduce_max(tf.reduce_sum(1.0 - paddings, 1)), tf.int32)

    def _TrimAndTranspose(name):
      x = input_batch[name]
      if name not in ('ids', 'paddings', 'labels', 'weights', 'segment_ids',
                      'segment_pos'):
        return x
      with tf.name_scope(f'trim_and_transpose_{name}'):
        x = py_utils.HasShape(x, tf.shape(paddings))
        if max_seq_len is not None:
          x = x[:, :max_seq_len]
        return tf.transpose(x)

    return py_utils.NestedMap(
        {name: _TrimAndTranspose(name) for name in input_batch})

  def FPropTower(self, theta, input_batch):
    p = self.params
    batch_size = tf.shape(input_batch.ids)[0]
    transposed_input_batch = self._TrimIfPossibleThenTranspose(input_batch)
    labels_ids = transposed_input_batch.labels
    weights = transposed_input_batch.weights

    state0 = self.lm.zero_state(theta.lm, batch_size)
    labels = py_utils.NestedMap(class_ids=labels_ids, class_weights=weights)
    fprop_kwargs = dict()
    if p.packed_input:
      # segment_id for FRNN should be of shape [time, batch, 1].
      fprop_kwargs.update(
          segment_id=tf.expand_dims(transposed_input_batch.segment_ids, -1))
    xent_output, _ = self.lm.FProp(theta.lm, transposed_input_batch.ids,
                                   transposed_input_batch.paddings, state0,
                                   labels, **fprop_kwargs)

    # +1 to account for the end of sequence symbol.
    if p.packed_input:
      num_sentences = input_batch.num_sentences
    else:
      num_sentences = tf.constant(1, dtype=tf.int32)
    num_words = tf.cast(
        # words and eos tokens.
        tf.reduce_sum(input_batch.word_count + num_sentences),
        tf.float32)
    predicted_labels = tf.cast(xent_output.per_example_argmax, labels_ids.dtype)

    num_preds = xent_output.total_weight
    mean_acc = tf.reduce_sum(
        tf.cast(tf.equal(labels_ids, predicted_labels), tf.float32) *
        weights) / (
            num_preds + 1e-4)

    loss = xent_output.avg_xent
    if p.train.sum_loss_across_tokens_in_batch:
      loss = xent_output.total_xent
    return {
        'loss': (loss, num_preds),
        'fraction_of_correct_next_step_preds': (mean_acc, num_preds),
        'log_pplx': (xent_output.avg_xent, num_preds),
        'log_pplx_per_word': (xent_output.total_xent / num_words, num_words),
        'num_predictions': (num_preds, 1),
        'num_words': (num_words, 1)
    }, {}

  def ComputePredictions(self, theta, input_batch):
    return self.FPropTower(theta, input_batch)

  def ComputeLoss(self, theta, predictions, input_batch):
    return predictions

  def AdjustGradients(self, var_grad):
    """Clip LSTM gradients.

    Args:
      var_grad: a `.NestedMap` of (variable, gradient). You can view `var_grad`
        as an ordered list of (key, (var, grad)) tuples. Every key of `var_grad`
        exists in `vmap`. Every variable in `vmap` that contributes to loss must
        exist in `var_grad`. Every var of `var_grad` must exist in `vmap`.
        `grad` is the corresponding gradient computed for `var`. `grad` is
        guaranteed to be not None.

    Returns:
      adjusted version of `var_grad` that has clipped the LSTM gradients
      if `self.params.max_lstm_gradient_norm` is set.
    """

    p = self.params
    if p.train.max_lstm_gradient_norm:
      lstm_var_grad = var_grad.lm.rnns
      lstm_vars = lstm_var_grad.Transform(lambda x: x[0]).Flatten()
      lstm_grads = lstm_var_grad.Transform(lambda x: x[1]).Flatten()
      clipped_lstm_grads, _ = tf.clip_by_global_norm(
          lstm_grads, p.train.max_lstm_gradient_norm)
      var_grad.lm.rnns = var_grad.lm.rnns.Pack(
          list(zip(lstm_vars, clipped_lstm_grads)))

    return var_grad

  def Inference(self):
    """Constructs the inference subgraphs.

    Returns:
      dict: ``{'subgraph_name': (fetches, feeds)}``
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    """Default inference subgraph.

    Returns:
      (fetches, feeds):

      - fetches: A dictionary of fetches, containing:

        - log_pplx_per_token: A matrix of shape [batch, time]. [i, j]
          is i-th input text's j-th token's log prob.
        - paddings: A matrix of shape [batch, time]. The padding mask.
        - log_pplx_per_sample: A vector of shape [batch]. [i]
          is i-th input text's log prob.
        - num_oovs_per_sample: A vector of shape [batch] counting the total
          number of out-of-vocabulary tokens in each input.
        - tokens_from_labels: A vector of shape [batch] returning the predicted
          tokens as a sequence after mapping them back to strings from ids using
          the vocabulary.
        - ids: A matrix of shape [batch, time]. [i, j]
          is i-th input text's j-th token's id.

      - feeds: A dictionary of feeds, containing:

        - text: A placeholder for a vector of strings.
    """
    text = tf.placeholder(tf.string, shape=[None])
    # [batch, time]
    ids, labels, paddings = self.input_generator.StringsToIds(text)
    weights = 1. - paddings
    lengths = tf.reduce_sum(tf.cast(1 - paddings, tf.int32), axis=1)
    tokens_from_labels = self.input_generator.IdsToStrings(labels, lengths)
    oovs = tf.equal(labels, self.input_generator.tokenizer.unk_id)
    num_oovs_per_sample = tf.cast(
        tf.round(
            tf.reduce_sum(tf.cast(oovs, tf.float32) * (1 - paddings), axis=1)),
        tf.int32)
    # [time, batch]
    transposed = self._TrimIfPossibleThenTranspose(
        py_utils.NestedMap({
            'ids': ids,
            'paddings': paddings,
            'labels': labels,
            'weights': weights,
        }))
    batch_size = tf.shape(ids)[0]
    xent_output, _ = self.lm.FPropDefaultTheta(
        inputs=transposed.ids,
        paddings=transposed.paddings,
        state0=self.lm.zero_state(self.theta.lm, batch_size),
        labels=py_utils.NestedMap(
            class_ids=transposed.labels, class_weights=transposed.weights))

    per_example_xent = py_utils.HasShape(xent_output.per_example_xent,
                                         tf.shape(transposed.ids))
    log_pplx_per_sample = tf.reduce_sum(
        per_example_xent * (1 - transposed.paddings), axis=0)
    fetches = {
        'log_pplx_per_token':  # [batch, time]
            tf.transpose(per_example_xent),
        'paddings':  # [batch, time]
            paddings,
        'lengths':  # [batch]
            lengths,
        'log_pplx_per_sample':  # [batch]
            log_pplx_per_sample,
        'num_oovs_per_sample':  # [batch], int32
            num_oovs_per_sample,
        'tokens_from_labels':  # [batch], string
            tokens_from_labels,
        'ids':  # [batch, time], int32
            ids,
    }
    feeds = {
        'text': text,
        'ids': ids,
        'paddings': paddings,
        'labels': labels,
        'weights': weights,
    }
    return fetches, feeds


class FixedShapeInputLanguageModel(LanguageModel):

  def _TrimIfPossibleThenTranspose(self, input_batch):
    if not py_utils.use_tpu() and self.do_eval:
      max_seq_len = tf.cast(
          tf.reduce_max(tf.reduce_sum(1.0 - input_batch.paddings, 1)), tf.int32)
      input_batch = input_batch.Transform(lambda x: x[:, :max_seq_len])
    input_batch = input_batch.Transform(tf.transpose)
    return input_batch


class BatchMajorLanguageModel(LanguageModel):
  """Batch major implementation of the language model."""

  def _TrimIfPossible(self, ids, paddings, labels, weights):
    p = self.params
    data = (ids, paddings, labels, weights)
    if not py_utils.use_tpu() and not p.packed_input:
      max_seq_len = tf.cast(
          tf.reduce_max(tf.reduce_sum(1.0 - paddings, 1)), tf.int32)
      data = (x[:, :max_seq_len] for x in data)
    return data

  def FPropTower(self, theta, input_batch):
    p = self.params
    tf.logging.info('input_batch=%r', input_batch)
    ids, paddings, labels_ids, weights = self._TrimIfPossible(
        input_batch.ids, input_batch.paddings, input_batch.labels,
        input_batch.weights)
    fprop_dtype = py_utils.FPropDtype(p)
    paddings = tf.cast(paddings, fprop_dtype)
    weights = tf.cast(weights, fprop_dtype)
    tf.logging.info('inputs={}'.format((ids, paddings, labels_ids, weights)))

    batch_size = tf.shape(ids)[0]
    state0 = None
    labels = py_utils.NestedMap(class_ids=labels_ids, class_weights=weights)
    fprop_kwargs = dict()
    if 'segment_ids' in input_batch:
      fprop_kwargs.update(
          segment_ids=input_batch.segment_ids,
          segment_pos=input_batch.segment_pos)
    xent_output, _ = self.lm.FProp(theta.lm, ids, paddings, state0, labels,
                                   **fprop_kwargs)

    if 'segment_ids' in input_batch:
      num_sentences = input_batch.num_sentences
    else:
      num_sentences = tf.ones(shape=[batch_size], dtype=tf.int32)
    # +num_sentences to account for the end of sequence symbol.
    num_words = tf.cast(
        tf.reduce_sum(input_batch.word_count + num_sentences), fprop_dtype)
    predicted_labels = tf.cast(xent_output.per_example_argmax, labels_ids.dtype)

    num_preds = xent_output.total_weight
    mean_acc = tf.reduce_sum(
        tf.cast(tf.equal(labels_ids, predicted_labels), fprop_dtype) *
        weights) / tf.math.maximum(num_preds, 1)
    loss = xent_output.avg_xent
    if p.train.sum_loss_across_tokens_in_batch:
      loss = xent_output.total_xent
    return {
        'loss': (loss, num_preds),
        'fraction_of_correct_next_step_preds': (mean_acc, num_preds),
        'log_pplx': (xent_output.avg_xent, num_preds),
        'log_pplx_per_word': (xent_output.total_xent / num_words, num_words),
        'num_predictions': (num_preds, 1),
        'num_words': (num_words, 1),
        'num_sentences': (tf.reduce_sum(num_sentences), 1),
    }, {}

  def _InferenceSubgraph_Default(self):
    """Default inference subgraph."""
    text = tf.placeholder(tf.string, shape=[None])
    # [batch, time]
    ids, labels, paddings = self.input_generator.StringsToIds(text)
    weights = 1. - paddings
    ids, paddings, labels, weights = self._TrimIfPossible(
        ids, paddings, labels, weights)
    lengths = tf.reduce_sum(tf.cast(1 - paddings, tf.int32), axis=1)
    tokens_from_labels = self.input_generator.IdsToStrings(labels, lengths)
    oovs = tf.equal(labels, self.input_generator.tokenizer.unk_id)
    num_oovs_per_sample = tf.cast(
        tf.round(
            tf.reduce_sum(tf.cast(oovs, tf.float32) * (1 - paddings), axis=1)),
        tf.int32)
    batch_size = tf.shape(ids)[0]
    xent_output, _ = self.lm.FPropDefaultTheta(
        inputs=ids,
        paddings=paddings,
        state0=self.lm.zero_state(self.theta.lm, batch_size),
        labels=py_utils.NestedMap(class_ids=labels, class_weights=weights))

    per_example_xent = py_utils.HasShape(xent_output.per_example_xent,
                                         tf.shape(ids))
    log_pplx_per_sample = tf.reduce_sum(
        per_example_xent * (1 - paddings), axis=1)
    fetches = {
        'log_pplx_per_token':  # [batch, time]
            per_example_xent,
        'paddings':  # [batch, time]
            paddings,
        'lengths':  # [batch]
            lengths,
        'log_pplx_per_sample':  # [batch]
            log_pplx_per_sample,
        'num_oovs_per_sample':  # [batch], int32
            num_oovs_per_sample,
        'tokens_from_labels':  # [batch], string
            tokens_from_labels,
        'ids':  # [batch, time], int32
            ids
    }
    feeds = {
        'text': text,
        'ids': ids,
        'paddings': paddings,
        'labels': labels,
        'weights': weights,
    }
    return fetches, feeds


class PackedBatchMajorLanguageModel(base_model.BaseTask):
  """Packed batch major implementation of the language model."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('aux_loss_weight', 0.0,
             'Weight of the auxiliary loss in the overall loss term.')
    p.Define('lm', layers.RnnLm.Params(), 'LM layer.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    # Construct the model.
    self.CreateChild('lm', p.lm)

  def FPropTower(self, theta, input_batch):
    with py_utils.AuxLossContext() as aux_loss_ctx:
      assert aux_loss_ctx is not None
      p = self.params
      fprop_dtype = py_utils.FPropDtype(p)
      tf.logging.info('input_batch=%r', input_batch)
      ids = input_batch.ids
      labels_ids = input_batch.labels
      paddings = tf.cast(input_batch.paddings, fprop_dtype)
      weights = tf.cast(input_batch.weights, fprop_dtype)
      tf.logging.info('inputs={}'.format((ids, paddings, labels_ids, weights)))

      batch_size = tf.shape(ids)[0]
      state0 = self.lm.zero_state(theta.lm, batch_size)
      labels = py_utils.NestedMap(class_ids=labels_ids, class_weights=weights)
      xent_output, _ = self.lm.FProp(
          theta.lm,
          ids,
          paddings,
          state0,
          labels,
          segment_ids=input_batch.segment_ids,
          segment_pos=input_batch.segment_pos)

      # +input_batch.num_sentences to account for the end of sequence symbol.
      num_words = tf.cast(
          tf.reduce_sum(input_batch.word_count +
                        tf.cast(input_batch.num_sentences, dtype=tf.int32)),
          fprop_dtype)
      predicted_labels = tf.cast(xent_output.per_example_argmax,
                                 labels_ids.dtype)
      num_sentences = tf.reduce_sum(input_batch.num_sentences)

      num_preds = tf.cast(xent_output.total_weight, fprop_dtype)
      mean_acc = tf.reduce_sum(
          tf.cast(tf.equal(labels_ids, predicted_labels), fprop_dtype) *
          weights) / tf.math.maximum(num_preds, 1)
      avg_xent = xent_output.avg_xent
      aux_loss_tensors = aux_loss_ctx.aux_losses
      if aux_loss_tensors:
        assert isinstance(aux_loss_tensors, list)
        assert len(aux_loss_tensors) >= 1
        # scalar
        assert p.aux_loss_weight > 0
        aux_loss = p.aux_loss_weight * tf.add_n(aux_loss_tensors)
      else:
        # scalar
        aux_loss = tf.zeros_like(avg_xent)

      loss = avg_xent + aux_loss
      return {
          'loss': (loss, num_preds),
          'avg_xent': (avg_xent, num_preds),
          'aux_loss': (aux_loss, num_preds),
          'fraction_of_correct_next_step_preds': (mean_acc, num_preds),
          'log_pplx': (xent_output.avg_xent, num_preds),
          'log_pplx_per_word': (xent_output.total_xent / num_words, num_words),
          'num_predictions': (num_preds, 1),
          'num_words': (num_words, 1),
          'num_sentences': (num_sentences, 1)
      }, {}

  def Inference(self):
    """Constructs the inference subgraphs.

    Returns:
      dict: ``{'subgraph_name': (fetches, feeds)}``
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    """Default inference subgraph."""
    batch_size = None
    seq_length = None
    fp_dtype = py_utils.FPropDtype(self.params)
    tshape = (batch_size, seq_length)
    input_ids = tf.placeholder(dtype=tf.int32, shape=tshape)
    targets = tf.placeholder(dtype=tf.int32, shape=tshape)
    paddings = tf.placeholder(dtype=fp_dtype, shape=tshape)
    weights = tf.placeholder(dtype=fp_dtype, shape=tshape)
    segment_ids = tf.placeholder(dtype=tf.int32, shape=tshape)
    segment_pos = tf.placeholder(dtype=tf.int32, shape=tshape)
    word_count = tf.placeholder(dtype=tf.int32, shape=(batch_size))
    num_sentences = tf.placeholder(dtype=tf.int32, shape=(batch_size))
    feeds = {
        'ids': input_ids,
        'labels': targets,
        'paddings': paddings,
        'weights': weights,
        'segment_ids': segment_ids,
        'segment_pos': segment_pos,
        'word_count': word_count,
        'num_sentences': num_sentences
    }
    input_batch = py_utils.NestedMap(feeds)
    loss, _ = self.FPropTower(self.theta, input_batch)
    fetches = {'loss': loss['loss'][0]}
    return fetches, feeds
