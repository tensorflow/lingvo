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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import lr_schedule
from lingvo.core import py_utils
from lingvo.tasks.lm import layers


class LanguageModel(base_model.BaseTask):
  """Language model."""

  @classmethod
  def Params(cls):
    p = super(LanguageModel, cls).Params()
    p.Define('lm', layers.RnnLm.Params(), 'LM layer.')

    tp = p.train
    tp.Define(
        'max_lstm_gradient_norm', 0.0,
        'Clip gradient for vars in lstm layers by setting this value to '
        'something > 0.')
    tp.Define(
        'sum_loss_across_tokens_in_batch', False,
        'Sum the logP across predicted tokens in batch when set to True; '
        'average across predicted tokens in batch o/w (default).')

    tp.lr_schedule = lr_schedule.PiecewiseConstantLearningRateSchedule.Params(
    ).Set(
        boundaries=[350000, 500000, 600000], values=[1.0, 0.1, 0.01, 0.001])
    tp.vn_start_step = 20000
    tp.vn_std = 0.0
    tp.learning_rate = 0.001
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LanguageModel, self).__init__(params)
    p = self.params

    assert p.lm.vocab_size == p.input.tokenizer.vocab_size, (
        'lm.vocab_size does not match input.tokenizer.vocab_size: %d vs %d' %
        (p.lm.vocab_size, p.input.tokenizer.vocab_size))

    with tf.variable_scope(p.name):
      # Construct the model.
      self.CreateChild('lm', p.lm)

  def _TrimIfPossibleThenTranspose(self, ids, paddings, labels, weights):
    data = (ids, paddings, labels, weights)
    if not py_utils.use_tpu():
      max_seq_len = tf.cast(
          tf.reduce_max(tf.reduce_sum(1.0 - paddings, 1)), tf.int32)
      data = (x[:, :max_seq_len] for x in data)
    return (tf.transpose(x) for x in data)

  def FPropTower(self, theta, input_batch):
    p = self.params
    ids, paddings, labels_ids, weights = self._TrimIfPossibleThenTranspose(
        input_batch.ids, input_batch.paddings, input_batch.labels,
        input_batch.weights)

    batch_size = tf.shape(ids)[1]
    state0 = self.lm.zero_state(batch_size)
    labels = py_utils.NestedMap(class_ids=labels_ids, class_weights=weights)
    xent_output, _ = self.lm.FProp(theta.lm, ids, paddings, state0, labels)

    # +1 to account for the end of sequence symbol.
    num_words = tf.cast(
        tf.reduce_sum(input_batch.word_count + tf.constant(1, dtype=tf.int32)),
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

  def AdjustGradients(self, var_grad):
    """Clip LSTM gradients.

    Args:
      var_grad: a `.NestedMap` of (variable, gradient). You can view
        `var_grad` as an ordered list of (key, (var, grad)) tuples. Every
        key of `var_grad` exists in `vmap`. Every variable in `vmap` that
        contributes to loss must exist in `var_grad`. Every var of `var_grad`
        must exist in `vmap`. `grad` is the corresponding gradient computed
        for `var`. `grad` is guaranteed to be not None.

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
      {'subgraph_name': (fetches, feeds)}
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    """Default inference subgraph.

    Returns:
      (fetches, feeds), with:

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
    lengths = tf.reduce_sum(tf.to_int32(1 - paddings), axis=1)
    tokens_from_labels = self.input_generator.IdsToStrings(labels, lengths)
    oovs = tf.equal(labels, self.input_generator.tokenizer.unk_id)
    num_oovs_per_sample = tf.to_int32(
        tf.reduce_sum(tf.to_float(oovs) * (1 - paddings), axis=1))
    # [time, batch]
    ids, paddings, labels, weights = self._TrimIfPossibleThenTranspose(
        ids, paddings, labels, 1.0 - paddings)
    batch_size = tf.shape(ids)[1]
    xent_output, _ = self.lm.FPropDefaultTheta(
        inputs=ids,
        paddings=paddings,
        state0=self.lm.zero_state(batch_size),
        labels=py_utils.NestedMap(class_ids=labels, class_weights=weights))

    per_example_xent = py_utils.HasShape(xent_output.per_example_xent,
                                         tf.shape(ids))
    log_pplx_per_sample = tf.reduce_sum(
        per_example_xent * (1 - paddings), axis=0)
    fetches = {
        'log_pplx_per_token':  # [batch, time]
            tf.transpose(per_example_xent),
        'paddings':  # [batch, time]
            tf.transpose(paddings),
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
    feeds = {'text': text}
    return fetches, feeds


class FixedShapeInputLanguageModel(LanguageModel):

  def _TrimIfPossibleThenTranspose(self, ids, paddings, labels, weights):
    data = (ids, paddings, labels, weights)
    if not py_utils.use_tpu() and self.params.is_eval:
      max_seq_len = tf.cast(
          tf.reduce_max(tf.reduce_sum(1.0 - paddings, 1)), tf.int32)
      data = (x[:, :max_seq_len] for x in data)
    return (tf.transpose(x) for x in data)
