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
"""Common layers for language models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import range
from six.moves import zip
import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import layers_with_gpipe
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers


class BaseLanguageModel(base_layer.BaseLayer):
  """Abstract base class for a language model layer."""

  @classmethod
  def Params(cls):
    p = super(BaseLanguageModel, cls).Params()
    p.Define('vocab_size', 0, 'Number of vocabulary tokens.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseLanguageModel, self).__init__(params)

  def zero_state(self, batch_size):
    raise NotImplementedError('Abstract method')

  def FProp(self, theta, inputs, paddings, state0, *args, **kwargs):
    """Computes xent loss given the language model inputs.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [time, batch] or [time, batch, dims].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A `.NestedMap` containing the initial recurrent state.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      (xent_output, state1). `xent_output` is a `.NestedMap` as defined by
      `SoftmaxLayer`'s return value and `state1` is the next recurrent state.
    """
    raise NotImplementedError('Abstract method')

  def Logits(self, theta, inputs, paddings, *args, **kwargs):
    """FProp and returns the logits for the whole sequence."""
    xent_output, _ = self.FProp(
        theta,
        inputs,
        paddings,
        state0=self.zero_state(tf.shape(inputs)[1]),
        *args,
        **kwargs)
    return xent_output.logits

  @classmethod
  def StepOutputDimension(cls, params):
    """Returns dimensions of `Step()`'s output dimension.

    Args:
      params: Params for this layer.

    Returns:
      A `.NestedMap` with fields
        logits: a python int.
            The vocab size.
        last_hidden: a python int.
            The last hidden layer's dimension.
    """
    raise NotImplementedError('Abstract method')

  def Step(self, theta, inputs, paddings, state0, *args, **kwargs):
    """FProp one step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [batch] or [batch, dims].
      paddings: a 0/1 tensor of shape [batch].
      state0: A `.NestedMap` containing the initial recurrent state.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      A tuple (output, state1).
        output: A `.NestedMap` with fields.
          logits:
            [batch, vocab_size].
          log_probs:
            [batch, vocab_size].
          last_hidden:
            [batch, dims].
        state1:
          The new recurrent state.
    """

    def ExpandTime(x):
      return tf.expand_dims(x, axis=0)

    xent_output, state1 = self.FProp(
        theta=theta,
        inputs=ExpandTime(inputs),
        paddings=ExpandTime(paddings),
        state0=state0,
        *args,
        **kwargs)

    output = py_utils.NestedMap()
    output.log_probs = tf.squeeze(xent_output.log_probs, axis=0)
    output.probs = tf.squeeze(xent_output.probs, axis=0)
    output.last_hidden = tf.squeeze(xent_output.last_hidden, axis=0)
    if 'logits' in xent_output:
      # FstLm doesn't return logits.
      output.logits = tf.squeeze(xent_output.logits, axis=0)
    return output, state1

  def GetFeedDict(self):
    """Returns an optional feed dict with str keys and Tensor values."""
    return {}

  def CombineStates(self, state0, state1, switch_cond):
    """Combines states based on a switch conditional.

    Args:
      state0: a NestedMap of states to use for batch elements where switch_cond
        is true.
      state1: a NestedMap of states to use for batch elements where switch_cond
        is false.
      switch_cond: bool tensor of shape [batch] on which to switch.

    Returns:
      state_combined: a NestedMap of states.
    """
    raise NotImplementedError('Abstract method')


class NullLm(BaseLanguageModel):
  """A trivial language model does nothing really."""

  def zero_state(self, batch_size):
    return py_utils.NestedMap(
        m=tf.zeros([batch_size, 0], dtype=self.params.dtype))

  def FProp(self, theta, inputs, paddings, state0, *args, **kwargs):
    p = self.params
    time = tf.shape(inputs)[0]
    batch = tf.shape(inputs)[1]
    logits = tf.zeros([time, batch, p.vocab_size], dtype=p.dtype)
    return py_utils.NestedMap(
        logits=logits,
        probs=tf.nn.softmax(logits),
        log_probs=tf.nn.log_softmax(logits),
        last_hidden=tf.zeros([time, batch, 0], dtype=p.dtype)), state0

  def Logits(self, theta, inputs, paddings, *args, **kwargs):
    """FProp and returns the logits for the whole sequence."""
    p = self.params
    del theta, paddings
    time, batch = tf.unstack(tf.shape(inputs)[:2])
    return tf.zeros([time, batch, p.vocab_size], dtype=p.dtype)

  @classmethod
  def StepOutputDimension(cls, params):
    """Returns dimensions of `Step()`'s output dimension."""
    return py_utils.NestedMap(logits=params.vocab_size, last_hidden=0)

  def Step(self, theta, inputs, paddings, state0, *args, **kwargs):
    """FProp one step."""
    p = self.params
    batch = tf.shape(inputs)[0]
    logits = tf.zeros([batch, p.vocab_size], dtype=p.dtype)
    return py_utils.NestedMap(
        logits=logits,
        log_probs=tf.nn.log_softmax(logits),
        probs=tf.nn.softmax(logits),
        last_hidden=tf.zeros([batch, 0], dtype=p.dtype)), state0

  def CombineStates(self, state0, state1, switch_cond):
    """Combines states based on a switch conditional.

    Args:
      state0: a NestedMap of states to use for batch elements where switch_cond
        is true.
      state1: a NestedMap of states to use for batch elements where switch_cond
        is false.
      switch_cond: bool tensor of shape [batch] on which to switch.

    Returns:
      state_combined: a NestedMap of states.
    """
    return state0


def _RnnOutputSize(rnns):
  cell = rnns.cell_tpl[-1]
  return cell.num_output_nodes


def ComputeXentOutput(softmax_layer,
                      softmax_theta,
                      activations,
                      labels,
                      num_samples=1):
  """Compute Softmax CrossEntropy output."""
  seqlen, batch, _ = tf.unstack(tf.shape(activations), num=3)
  if labels is None:
    # We can only compute the logits here.
    logits = softmax_layer.Logits(
        theta=softmax_theta,
        inputs=tf.reshape(activations, [seqlen * batch * num_samples, -1]))
    xent_output = py_utils.NestedMap(
        logits=tf.reshape(logits, [seqlen, batch, -1]))
  elif 'class_ids' in labels:
    # labels.class_ids: [len, batch]
    if num_samples > 1:
      class_ids = tf.tile(labels.class_ids, [1, num_samples])
      class_weights = tf.tile(labels.class_weights, [1, num_samples])
    else:
      class_ids = labels.class_ids
      class_weights = labels.class_weights
    xent_output = softmax_layer.FProp(
        theta=softmax_theta,
        inputs=activations,
        class_weights=class_weights,
        class_ids=class_ids)
  else:
    assert 'class_probabilities' in labels
    if num_samples > 1:
      class_probabilities = tf.tile(labels.class_probabilities,
                                    [1, num_samples])
      class_weights = tf.tile(labels.class_weights, [1, num_samples])
    else:
      class_probabilities = labels.class_probabilities
      class_weights = labels.class_weights
    xent_output = softmax_layer.FProp(
        theta=softmax_theta,
        inputs=activations,
        class_weights=class_weights,
        class_probabilities=class_probabilities)
  return xent_output


class RnnLmNoEmbedding(BaseLanguageModel):
  """Stacked RNN based language model layer."""

  @classmethod
  def Params(cls):
    p = super(RnnLmNoEmbedding, cls).Params()
    p.Define('rnns', rnn_layers.StackedFRNNLayerByLayer.Params(),
             'The stacked-RNNs layer params.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'The softmax layer params.')
    p.Define(
        'direct_features_dim', 0,
        'If > 0, then the number of dimensions of direct features '
        'that bypass the RNN and are provided directly to the softmax '
        'input.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RnnLmNoEmbedding, self).__init__(params)
    p = self.params
    if not isinstance(p.rnns.cell_tpl, (list, tuple)):
      p.rnns.cell_tpl = [p.rnns.cell_tpl]
    p.rnns.allow_implicit_capture = p.allow_implicit_capture

    cell_output_size = _RnnOutputSize(p.rnns)
    output_layer_size = cell_output_size + p.direct_features_dim

    if output_layer_size != p.softmax.input_dim:
      raise ValueError(
          'Output layer size %d does not match softmax input size %d! '
          'cell_output_size: %d direct_features_dim: %d ' %
          (output_layer_size, p.softmax.input_dim, cell_output_size,
           p.direct_features_dim))
    if p.softmax.num_classes != p.vocab_size:
      raise ValueError(
          'softmax num of classess %d does not match vocabulary size %d!' %
          (p.softmax.num_classes, p.vocab_size))

    with tf.variable_scope(p.name):
      self.CreateChild('rnns', p.rnns)
      self.CreateChild('softmax', p.softmax)

  def zero_state(self, batch_size):
    return self.rnns.zero_state(batch_size)

  @classmethod
  def StepOutputDimension(cls, params):
    return py_utils.NestedMap(
        logits=params.vocab_size, last_hidden=params.softmax.input_dim)

  def Step(self,
           theta,
           inputs,
           paddings,
           state0,
           direct_features=None,
           *args,
           **kwargs):
    """FProp one step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [batch] or [batch, dims].
      paddings: a 0/1 tensor of shape [batch].
      state0: A `.NestedMap` containing the initial recurrent state.
      direct_features: If not None, a tensor of [batch, direct_feature_dims]
        that is concatenated to the output of the last RNN layer.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      A tuple (output, state1).
        output: A `.NestedMap` with fields.
          logits:
            [batch, vocab_size].
          last_hidden:
            [batch, dims].
        state1:
          The new recurrent state.
    """

    def ExpandTime(x):
      return tf.expand_dims(x, axis=0)

    if direct_features is not None:
      direct_features = py_utils.HasRank(direct_features, 2)
      direct_features = ExpandTime(direct_features)

    xent_output, state1 = self.FProp(
        theta=theta,
        inputs=ExpandTime(inputs),
        paddings=ExpandTime(paddings),
        state0=state0,
        direct_features=direct_features,
        *args,
        **kwargs)

    output = py_utils.NestedMap()
    output.logits = tf.squeeze(xent_output.logits, axis=0)
    output.probs = tf.squeeze(xent_output.probs, axis=0)
    output.log_probs = tf.squeeze(xent_output.log_probs, axis=0)
    output.last_hidden = tf.squeeze(xent_output.last_hidden, axis=0)
    return output, state1

  def FProp(self,
            theta,
            inputs,
            paddings,
            state0,
            labels=None,
            direct_features=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: input activation. A tensor of shape [time, batch, dims].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A `.NestedMap` containing the initial recurrent state.
      labels: If not None, a `.NestedMap` containing the following fields.

        - class_weights, a tensor with shape [time, batch] containing the
          weights for each target word.
        - class_ids, a tensor with shape [time, batch] of int32 dtype containing
          the target class labels.
        - class_probabilities, a tensor with shape [time, batch, vocab_size] of
          float values indicating class-membership probabilities.
      direct_features:
        If not None, a tensor of [time, batch, direct_feature_dims] that is
        concatenated to the output of the last RNN layer.

    Returns:
      If `labels` is not None, returns (xent_output, state1), where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return
      value and `state1` is the next recurrent state. Otherwise,
      `xent_output` contains the softmax logits, probabilities (.probs) and
      log-probabilities (.log_probs).
    """
    inputs = py_utils.HasRank(inputs, 3)
    seqlen, batch, _ = tf.unstack(tf.shape(inputs), num=3)
    paddings = py_utils.HasShape(paddings, [seqlen, batch])
    assert state0 is not None
    activation, state1 = self.rnns.FProp(theta.rnns, inputs,
                                         tf.expand_dims(paddings, 2), state0)

    if direct_features is not None:
      direct_features = py_utils.HasRank(direct_features, 3)
      activation = tf.concat([activation, direct_features], axis=2)

    if labels is None:
      # We can only compute the logits here.
      logits = self.softmax.Logits(
          theta=theta.softmax,
          inputs=tf.reshape(activation, [seqlen * batch, -1]))
      xent_output = py_utils.NestedMap(
          logits=tf.reshape(logits, [seqlen, batch, -1]))
      xent_output.probs = tf.nn.softmax(xent_output.logits)
      xent_output.log_probs = tf.nn.log_softmax(xent_output.logits)
    elif 'class_ids' in labels:
      xent_output = self.softmax.FProp(
          theta=theta.softmax,
          inputs=activation,
          class_weights=labels.class_weights,
          class_ids=labels.class_ids)
    else:
      assert 'class_probabilities' in labels
      xent_output = self.softmax.FProp(
          theta=theta.softmax,
          inputs=activation,
          class_weights=labels.class_weights,
          class_probabilities=labels.class_probabilities)
    xent_output.last_hidden = activation
    return xent_output, state1

  def CombineStates(self, state0, state1, switch_cond):
    """Combines states based on a switch conditional.

    Args:
      state0: a NestedMap of states to use for batch elements where switch_cond
        is true.
      state1: a NestedMap of states to use for batch elements where switch_cond
        is false.
      switch_cond: bool tensor of shape [batch] on which to switch.

    Returns:
      state_combined: a NestedMap of states.
    """
    updated_rnn_states = []
    for i in range(self.params.rnns.num_layers):
      updated_rnn_states.append(
          py_utils.NestedMap({
              'c': tf.where(switch_cond, state0.rnn[i].c, state1.rnn[i].c),
              'm': tf.where(switch_cond, state0.rnn[i].m, state1.rnn[i].m)
          }))
    combined_state = py_utils.NestedMap({'rnn': updated_rnn_states})
    return combined_state


class RnnLm(RnnLmNoEmbedding):
  """Stacked RNN based language model layer."""

  @classmethod
  def Params(cls):
    p = super(RnnLm, cls).Params()
    p.Define('emb', layers.EmbeddingLayer.Params(),
             'The embedding layer params.')
    p.Define('embedding_dropout_keep_prob', 1.0, 'Embedding dropout keep prob.')
    p.Define('embedding_dropout_seed', None, 'Embedding dropout seed.')
    p.emb.max_num_shards = 1
    return p

  # TODO(zhifengc): Consider merge Params() and CommonParams().
  @classmethod
  def CommonParams(cls,
                   vocab_size,
                   emb_dim=1024,
                   num_layers=2,
                   rnn_dims=2048,
                   rnn_hidden_dims=0,
                   residual_start=1,
                   softmax_max_alloc=None):
    """A LM model parameterized by vocab size, etc.

    Args:
      vocab_size: Vocab size.
      emb_dim: Embedding dimension.
      num_layers: The number of rnn layers.
      rnn_dims: Each RNN layer has this many output nodes.
      rnn_hidden_dims: If > 0, each RNN layer has this many hidden nodes.
      residual_start: index of the first layer with a residual connection;
        higher index layers also have residuals.
      softmax_max_alloc: If set to a positive integer the soft-max
        computation is chunked into allocations of at most
        `softmax_max_alloc`; when left to its default value of None no
        chunking is done.

    Returns:
      A `RnnLm` parameter object.
    """
    p = cls.Params()
    p.vocab_size = vocab_size

    init_scale = 1.0 / math.sqrt(rnn_dims)

    # Embedding.
    p.emb.vocab_size = vocab_size
    p.emb.embedding_dim = emb_dim
    p.emb.scale_sqrt_depth = True
    p.emb.params_init = py_utils.WeightInit.Uniform(init_scale)

    # RNNs
    p.rnns.num_layers = num_layers
    # Which layer starts to have the residual connection.
    p.rnns.skip_start = residual_start
    if num_layers > 1:
      p.rnns.cell_tpl = [
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=emb_dim,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims),
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=rnn_dims,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims)
      ]
    else:
      p.rnns.cell_tpl = [
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=emb_dim,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims)
      ]

    # Softmax
    p.softmax.input_dim = rnn_dims
    p.softmax.num_classes = vocab_size
    p.softmax.params_init = py_utils.WeightInit.Uniform(init_scale)
    if softmax_max_alloc:
      # If the vocab is very large, computes the softmax chunk-by-chunk.
      p.softmax.chunk_size = max(1, int(softmax_max_alloc / vocab_size))

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RnnLm, self).__init__(params)
    p = self.params

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert p.emb.embedding_dim == p.rnns.cell_tpl[0].num_input_nodes, (
        '{} vs. {}'.format(p.emb.embedding_dim,
                           p.rnns.cell_tpl[0].num_input_nodes))

    with tf.variable_scope(p.name):
      self.CreateChild('emb', p.emb)

  def FProp(self,
            theta,
            inputs,
            paddings,
            state0,
            labels=None,
            direct_features=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: input ids. An int32 tensor of shape [time, batch].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A `.NestedMap` containing the initial recurrent state.
      labels: If not None, a `.NestedMap` containing the following fields:

        - class_weights, a tensor with shape [time, batch] containing the
          weights for each target word.
        - class_ids, a tensor with shape [time, batch] of int32 dtype containing
          the target class labels.
        - class_probabilities, a tensor with shape [time, batch, vocab_size] of
          float values indicating class-membership probabilities.
      direct_features:
        If not None, a tensor of [time, batch, direct_feature_dims] that is
        concatenated to the output of the last RNN layer.

    Returns:
      If `labels` is not None, returns (xent_output, state1), where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return
      value and `state1` is the next recurrent state. Otherwise,
      `xent_output` only contains the softmax logits.
    """
    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    assert state0
    activation = self.emb.EmbLookup(theta.emb, ids)
    # Dropout on embeddings is only applied in training.
    p = self.params
    if p.embedding_dropout_keep_prob < 1.0 and not p.is_eval:
      activation = tf.nn.dropout(
          activation,
          keep_prob=p.embedding_dropout_keep_prob,
          seed=p.embedding_dropout_seed)
    return super(RnnLm, self).FProp(theta, activation, paddings, state0, labels,
                                    direct_features)


class ConditionalRnnLm(RnnLmNoEmbedding):
  """RnnLm where looked up embedding is concatenated with a condition vector."""

  @classmethod
  def Params(cls):
    p = super(ConditionalRnnLm, cls).Params()
    p.Define('condition_dim', 128, 'The size of the condition vector.')
    p.Define('emb', layers.EmbeddingLayer.Params(),
             'The embedding layer params.')
    p.Define(
        'embedding_dropout_keep_prob', 1.0, 'Embedding dropout keep prob.'
        'Dropout is applied after concatenating with condition vector.')
    p.Define('embedding_dropout_seed', None, 'Embedding dropout seed.')
    p.emb.max_num_shards = 1
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ConditionalRnnLm, self).__init__(params)
    p = self.params

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert (p.emb.embedding_dim + p.condition_dim ==
            p.rnns.cell_tpl[0].num_input_nodes), ('{} vs. {}'.format(
                p.emb.embedding_dim, p.rnns.cell_tpl[0].num_input_nodes))

    with tf.variable_scope(p.name):
      self.CreateChild('emb', p.emb)

  def FProp(self,
            theta,
            inputs,
            paddings,
            state0,
            condition,
            labels=None,
            direct_features=None):
    """Computes xent loss given the language model input and condition.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: input ids. An int32 tensor of shape [time, batch].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A `.NestedMap` containing the initial recurrent state.
      condition: input condition. A tensor of shape [batch, condition_dim].
      labels: If not None, a `.NestedMap` containing the following fields:  -
        class_weights, a tensor with shape [time, batch] containing the weights
        for each target word. - class_ids, a tensor with shape [time, batch] of
        int32 dtype containing the target class labels. - class_probabilities, a
        tensor with shape [time, batch, vocab_size] of float values indicating
        class-membership probabilities.
      direct_features: If not None, a tensor of [time, batch,
        direct_feature_dims] that is concatenated to the output of the last RNN
        layer.

    Returns:
      If `labels` is not None, returns (xent_output, state1), where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return
      value and `state1` is the next recurrent state. Otherwise,
      `xent_output` only contains the softmax logits.
    """
    p = self.params
    # `condition` should have shape (batch_size, dim)
    condition = py_utils.HasShape(condition,
                                  [tf.shape(paddings)[1], p.condition_dim])
    # Expand the time dimension -> (time, batch_size, dim)
    condition = tf.tile(
        tf.expand_dims(condition, 0), [tf.shape(inputs)[0], 1, 1])

    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    activation = self.emb.EmbLookup(theta.emb, ids)
    activation = tf.concat([activation, tf.cast(condition, p.dtype)], -1)
    # Dropout on embeddings is only applied in training.
    if p.embedding_dropout_keep_prob < 1.0 and not p.is_eval:
      activation = tf.nn.dropout(
          activation,
          keep_prob=p.embedding_dropout_keep_prob,
          seed=p.embedding_dropout_seed)
    return super(ConditionalRnnLm, self).FProp(theta, activation, paddings,
                                               state0, labels, direct_features)


class MoeLm(BaseLanguageModel):
  """Mixture of experts language modeling class."""

  @classmethod
  def Params(cls):
    p = super(MoeLm, cls).Params()
    p.Define(
        'emb',
        layers.EmbeddingLayer.Params().Set(max_num_shards=1),
        'The embedding layer params.')
    p.Define('shared_emb', True, 'If true, uses a single embedding')
    p.Define(
        'add_postgating_rnn', True, 'If true, add an RNNLM post gating. '
        'If false, add only a softmax on top.')
    p.Define('rnns', rnn_layers.StackedFRNNLayerByLayer.Params(),
             'The stacked-RNNs layer params.')
    p.Define('number_of_experts', 7, 'Number of experts.')
    p.Define('merge', RnnLmNoEmbedding.Params(),
             'The LM to use for the merged LM')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MoeLm, self).__init__(params)
    p = self.params
    if not isinstance(p.rnns.cell_tpl, (list, tuple)):
      p.rnns.cell_tpl = [p.rnns.cell_tpl]

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert p.emb.embedding_dim == p.rnns.cell_tpl[0].num_input_nodes, (
        '{} vs. {}'.format(p.emb.embedding_dim,
                           p.rnns.cell_tpl[0].num_input_nodes))
    if p.add_postgating_rnn:
      assert p.merge.vocab_size == p.vocab_size, ('{} vs. {}'.format(
          p.merge.vocab_size, p.vocab_size))

    with tf.variable_scope(p.name):
      # Embeddings
      if p.shared_emb:
        self.CreateChild('emb', p.emb)
      else:
        # 0-th embedding is for the domain predictor.
        self.CreateChildren(
            'emb', [
                p.emb.Copy().Set(name='emb_%d' % i)
                for i in range(1 + p.number_of_experts)
            ])

      # Rnns
      # 0-th rnns is for the domain predictor.
      self.CreateChildren(
          'rnns', [p.rnns.Copy() for i in range(1 + p.number_of_experts)])

      # Softmax
      rnn_output_size = _RnnOutputSize(p.rnns)
      sm_params = layers.SimpleFullSoftmax.Params()
      sm_params.name = 'domain_predictor_softmax'
      sm_params.input_dim = rnn_output_size
      sm_params.num_classes = p.number_of_experts
      self.CreateChild('domain_predictor_softmax', sm_params)

      # Merge
      if p.add_postgating_rnn:
        self.CreateChild('merge', p.merge)
      else:
        output_sm_params = layers.SimpleFullSoftmax.Params()
        output_sm_params.name = 'output_softmax'
        output_sm_params.input_dim = rnn_output_size
        output_sm_params.num_classes = p.vocab_size
        self.CreateChild('output_softmax', output_sm_params)

  def zero_state(self, batch_size):
    p = self.params
    if p.add_postgating_rnn:
      return py_utils.NestedMap(
          rnns=[x.zero_state(batch_size) for x in self.rnns],
          merge=self.merge.zero_state(batch_size))
    else:
      return py_utils.NestedMap(
          rnns=[x.zero_state(batch_size) for x in self.rnns])

  def FProp(self, theta, inputs, paddings, state0, labels=None):
    """Forward compute."""
    p = self.params

    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    seqlen, batch = tf.unstack(tf.shape(inputs), num=2)
    assert state0

    paddings_3d = tf.expand_dims(paddings, axis=2)

    # RNNs
    if p.shared_emb:
      emb_act = [self.emb.EmbLookup(theta.emb, inputs)
                ] * (1 + p.number_of_experts)
    else:
      emb_act = [
          self.emb[i].EmbLookup(theta.emb[i], inputs)
          for i in range(1 + p.number_of_experts)
      ]
    state1 = py_utils.NestedMap(rnns=[])
    rnns_act = []
    for i, act in enumerate(emb_act):
      act, state = self.rnns[i].FProp(theta.rnns[i], act, paddings_3d,
                                      state0.rnns[i])
      act = py_utils.HasRank(act, 3)
      rnns_act += [act]
      state1.rnns += [state]

    # [time, batch, experts, dims].
    expert_stacked = tf.stack(rnns_act[1:], axis=2)

    # Compute gating softmax. The 0-th rnns is used as the expert
    # predictor.  Because SoftmaxLayer.Logits takes a matrix as input,
    # we reshape rnns_act[0], the domain predictor activation, to a
    # matrix here.
    act = tf.reshape(rnns_act[0], [seqlen * batch, -1])
    logits = self.domain_predictor_softmax.Logits(
        theta.domain_predictor_softmax, act)
    # [time, batch, experts]
    gating = tf.reshape(tf.nn.softmax(logits), [seqlen, batch, -1])

    # Mix the experts.
    # [time, batch, dims]
    combined = tf.squeeze(
        tf.matmul(
            # [time, batch, 1, experts]
            tf.expand_dims(gating, axis=2),
            # [time, batch, experts, dims]
            expert_stacked),
        axis=2)

    if p.add_postgating_rnn:
      # Note that this layer includes 1 or more RNN layers followed
      # by a softmax.
      xent_loss, state1.merge = self.merge.FProp(theta.merge, combined,
                                                 paddings, state0.merge, labels)
    else:
      xent_loss = self.output_softmax.FProp(
          theta=theta.output_softmax,
          inputs=combined,
          class_weights=labels.class_weights,
          class_ids=labels.class_ids)

    # return xent_loss, state1
    return xent_loss, state1


class TransformerLmNoEmbedding(BaseLanguageModel):
  """Transformer language model."""

  @classmethod
  def Params(cls):
    p = super(TransformerLmNoEmbedding, cls).Params()
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define(
        'model_dim', 512, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    p.Define('num_trans_layers', 6, 'Number of Transformer layers.')
    p.Define('trans_tpl', layers_with_attention.TransformerLayer.Params(),
             'Transformer Layer params.')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define(
        'residual_dropout_prob', 0.0, 'Dropout prob to the output of '
        'each sub-layer before it is added to the sub-layer input.')
    p.Define(
        'atten_dropout_prob', 0.0, 'Dropout prob to the attention '
        'weights in each Transformer attention sub-layer.')
    p.Define(
        'relu_dropout_prob', 0.0, 'Dropout prob to the inner layer '
        'output (ReLU activation) in each Transformer feed-forward '
        'sub-layer.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'The softmax layer params.')

    # Default config for the transformer layers.
    p.trans_tpl.has_aux_atten = False
    p.trans_tpl.mask_self_atten = True
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 8
    p.trans_tpl.tr_atten_tpl.atten_tpl.enable_ctx_pre_proj = True
    p.trans_tpl.tr_atten_tpl.atten_tpl.enable_ctx_post_proj = True
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 2048

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerLmNoEmbedding, self).__init__(params)
    p = self.params
    p.trans_tpl.tr_atten_tpl.residual_dropout_prob = p.residual_dropout_prob
    p.trans_tpl.tr_atten_tpl.atten_dropout_prob = p.atten_dropout_prob
    p.trans_tpl.tr_fflayer_tpl.residual_dropout_prob = p.residual_dropout_prob
    p.trans_tpl.tr_fflayer_tpl.relu_dropout_prob = p.relu_dropout_prob

    with tf.variable_scope(p.name):
      p.position_emb.embedding_dim = p.model_dim
      self.CreateChild('position_emb', p.position_emb)

      dropout_tpl = layers.DropoutLayer.Params().Set(
          keep_prob=(1.0 - p.input_dropout_prob))
      self.CreateChild('input_dropout', dropout_tpl)

      params_trans_layers = []
      for i in range(p.num_trans_layers):
        params = p.trans_tpl.Copy()
        params.source_dim = p.model_dim
        params.name = 'layer_%d' % i
        params_trans_layers.append(params)
      self.CreateChildren('trans', params_trans_layers)

      p.softmax.input_dim = p.model_dim
      p.softmax.num_classes = p.vocab_size
      self.CreateChild('softmax', p.softmax)

  def zero_state(self, batch_size):
    p = self.params
    return py_utils.NestedMap({
        'layer_%d' % layer: py_utils.NestedMap({
            'key': tf.zeros([0, batch_size, p.model_dim]),
            'value': tf.zeros([0, batch_size, p.model_dim]),
        }) for layer in range(p.num_trans_layers)
    })

  @classmethod
  def StepOutputDimension(cls, params):
    return py_utils.NestedMap(
        logits=params.vocab_size, last_hidden=params.softmax.input_dim)

  def Step(self, theta, inputs, paddings, state0, *args, **kwargs):
    """FProp one step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [batch, model_dim].
      paddings: a 0/1 tensor of shape [batch]. Unused here.
      state0: A `.NestedMap` containing the prefix states up to step t-1.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      A tuple (output, state1).
        output: A `.NestedMap` with fields.
          logits:
            [batch, vocab_size].
          last_hidden:
            [batch, model_dims].
        state1:
          The updated prefix states including step t.
    """

    prefix_len, _ = py_utils.GetShape(state0['layer_0'].key, 2)
    # [1, model_dim]
    posit_embs = self.position_emb.FProp(theta.position_emb,
                                         prefix_len + 1)[-1:, :]
    # [batch, model_dim]
    input_embs = inputs + posit_embs
    input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

    # Make a copy of the input.
    state1 = state0.Pack(state0.Flatten())

    layer_in = input_embs
    for i, (layer, layer_theta) in enumerate(zip(self.trans, theta.trans)):
      layer_prefix_states = state0['layer_%i' % i]
      # [batch, model_dim]
      layer_out, _, updated_prefix_states = layer.ExtendStep(
          layer_theta, layer_in, layer_prefix_states)
      state1['layer_%i' % i] = updated_prefix_states
      layer_in = layer_out

    # [batch, vocab_size]
    logits = self.softmax.Logits(theta=theta.softmax, inputs=layer_out)

    output = py_utils.NestedMap(logits=logits, last_hidden=layer_out)
    return output, state1

  def FProp(self, theta, inputs, paddings, state0=None, labels=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: Input activation. A tensor of shape [time, batch, model_dim].
      paddings: A 0/1 tensor of shape [time, batch].
      state0: Not used for Transformer.
      labels: If not None, a `.NestedMap` containing the following fields:

        - class_weights, a tensor with shape [time, batch] containing the
          weights for each target word.
        - class_ids, a tensor with shape [time, batch] of int32 dtype containing
          the target class labels.
        - class_probabilities, a tensor with shape [time, batch, vocab_size] of
          float values indicating class-membership probabilities.

    Returns:
      If `labels` is not None, returns (xent_output, None), where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return
      value. Otherwise, `xent_output` only contains the softmax logits.
    """
    p = self.params
    inputs = py_utils.HasRank(inputs, 3)
    seqlen, batch, _ = tf.unstack(tf.shape(inputs), num=3)
    inputs = py_utils.HasShape(inputs, [seqlen, batch, p.model_dim])
    paddings = py_utils.HasShape(paddings, [seqlen, batch])

    # [time, 1, model_dim]
    posit_embs = tf.expand_dims(
        self.position_emb.FProp(theta.position_emb, seqlen), 1)
    # [time, batch, model_dim]
    input_embs = inputs + posit_embs
    input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

    layer_in = input_embs
    for layer, layer_theta in zip(self.trans, theta.trans):
      # [time, batch, model_dim]
      layer_out, _ = layer.FProp(layer_theta, layer_in, paddings)
      layer_in = layer_out

    if labels is None:
      # We can only compute the logits here.
      logits = self.softmax.Logits(
          theta=theta.softmax,
          inputs=tf.reshape(layer_out, [seqlen * batch, -1]))
      xent_output = py_utils.NestedMap(
          logits=tf.reshape(logits, [seqlen, batch, -1]))
    elif 'class_ids' in labels:
      xent_output = self.softmax.FProp(
          theta=theta.softmax,
          inputs=layer_out,
          class_weights=labels.class_weights,
          class_ids=labels.class_ids)
    else:
      assert 'class_probabilities' in labels
      xent_output = self.softmax.FProp(
          theta=theta.softmax,
          inputs=layer_out,
          class_weights=labels.class_weights,
          class_probabilities=labels.class_probabilities)
    xent_output.last_hidden = layer_out
    return xent_output, None


class TransformerLm(TransformerLmNoEmbedding):
  """Stacked RNN based language model layer."""

  @classmethod
  def Params(cls):
    p = super(TransformerLm, cls).Params()
    p.Define('emb', layers.EmbeddingLayer.Params(),
             'The embedding layer params.')
    p.emb.max_num_shards = 1
    return p

  @classmethod
  def CommonParams(cls,
                   model_dim,
                   hidden_dim,
                   num_heads,
                   num_layers,
                   learning_rate,
                   warmup_steps,
                   vocab_size,
                   input_dropout_prob=0.0,
                   residual_dropout_prob=0.1,
                   atten_dropout_prob=0.0,
                   relu_dropout_prob=0.0,
                   softmax_max_alloc=None):
    """Common setup for Transformer language models.

    Args:
      model_dim: model dimension.
      hidden_dim: hidden dimension of feed-forward inner layer.
      num_heads: number of attention heads.
      num_layers: number of layers in the transformer LM.
      learning_rate: learning rate.
      warmup_steps: warmup steps for TransformerLearningRateSchedule.
      vocab_size: vocab size.
      input_dropout_prob: dropout prob to the sums of the token embeddings and
        the position embeddings.
      residual_dropout_prob: dropout prob to the output of each sub-layer before
        it is added to the sub-layer input.
      atten_dropout_prob: dropout prob to the attention weights in each
        Transformer attention sub-layer.
      relu_dropout_prob: dropout prob to the inner layer output (ReLU
        activation) in each Transformer feed-forward sub-layer.
      softmax_max_alloc: If set to a positive integer the soft-max
        computation is chunked into allocations of at most
        softmax_max_alloc; when left to its default value of None no
        chunking is done.

    Returns:
      A Params object containing the parameters that set up a Transformer LM.
    """
    p = cls.Params()
    p.name = 'transformerlm'

    p.model_dim = model_dim
    p.vocab_size = vocab_size
    p.num_trans_layers = num_layers
    p.input_dropout_prob = input_dropout_prob
    p.residual_dropout_prob = residual_dropout_prob
    p.atten_dropout_prob = atten_dropout_prob
    p.relu_dropout_prob = relu_dropout_prob

    default_params_init = py_utils.WeightInit.Xavier(1.0)
    emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(p.model_dim))
    p.emb.Set(
        vocab_size=vocab_size,
        embedding_dim=p.model_dim,
        max_num_shards=16,
        params_init=emb_params_init,
        scale_sqrt_depth=True)

    p.position_emb.Set(embedding_dim=p.model_dim, trainable_scaling=False)

    p.trans_tpl.has_aux_atten = False
    p.trans_tpl.mask_self_atten = True

    p.trans_tpl.tr_atten_tpl.Set(
        num_attention_heads=num_heads, params_init=default_params_init)

    p.trans_tpl.tr_atten_tpl.atten_tpl.Set(
        enable_ctx_pre_proj=True, enable_ctx_post_proj=True)

    p.trans_tpl.tr_fflayer_tpl.Set(
        hidden_dim=hidden_dim, params_init=default_params_init)

    p.softmax.Set(
        num_classes=vocab_size, num_shards=16, params_init=emb_params_init)

    if softmax_max_alloc:
      # If the vocab is very large, computes the softmax chunk-by-chunk.
      p.softmax.chunk_size = max(1, int(softmax_max_alloc / vocab_size))
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerLm, self).__init__(params)
    p = self.params

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert p.emb.embedding_dim == p.position_emb.embedding_dim, (
        '{} vs. {}'.format(p.emb.embedding_dim, p.position_emb.embedding_dim))
    assert p.emb.embedding_dim == p.model_dim, ('{} vs. {}'.format(
        p.emb.embedding_dim, p.model_dim))

    with tf.variable_scope(p.name):
      self.CreateChild('emb', p.emb)

  def FProp(self, theta, inputs, paddings, state0=None, labels=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: Input ids. An int32 tensor of shape [time, batch].
      paddings: A 0/1 tensor of shape [time, batch].
      state0: Not used for Transformer.
      labels: If not None, a `.NestedMap` containing the following fields:

        - class_weights, a tensor with shape [time, batch] containing the
          weights for each target word.
        - class_ids, a tensor with shape [time, batch] of int32 dtype containing
          the target class labels.
        - class_probabilities, a tensor with shape [time, batch, vocab_size] of
          float values indicating class-membership probabilities.

    Returns:
      If `labels` is not None, returns (xent_output, state1), where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return
      value and `state1` is the next recurrent state. Otherwise,
      `xent_output` only contains the softmax logits.
    """
    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    activation = self.emb.EmbLookup(theta.emb, ids)
    return super(TransformerLm, self).FProp(
        theta, activation, paddings, labels=labels)


class GPipeTransformerLmNoEmbedding(BaseLanguageModel):
  """GPipe Transformer language model."""

  @classmethod
  def Params(cls):
    p = super(GPipeTransformerLmNoEmbedding, cls).Params()
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define(
        'model_dim', 512, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    p.Define('stack', layers_with_gpipe.GPipeTransformerStack.Params(),
             'GPipeTransformerStack Layer params.')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define(
        'residual_dropout_prob', 0.0, 'Dropout prob to the output of '
        'each sub-layer before it is added to the sub-layer input.')
    p.Define(
        'atten_dropout_prob', 0.0, 'Dropout prob to the attention '
        'weights in each Transformer attention sub-layer.')
    p.Define(
        'relu_dropout_prob', 0.0, 'Dropout prob to the inner layer '
        'output (ReLU activation) in each Transformer feed-forward '
        'sub-layer.')
    p.Define('label_smoother', None, 'Label smoothing class.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'The softmax layer params.')

    # Default config for the transformer layers.
    trans_tpl = p.stack.encoder_tpl
    trans_tpl.has_aux_atten = False
    trans_tpl.mask_self_atten = True
    trans_tpl.tr_atten_tpl.is_masked = True
    trans_tpl.tr_atten_tpl.num_attention_heads = 8
    trans_tpl.tr_atten_tpl.atten_tpl.enable_ctx_pre_proj = True
    trans_tpl.tr_atten_tpl.atten_tpl.enable_ctx_post_proj = True
    trans_tpl.tr_fflayer_tpl.hidden_dim = 2048
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeTransformerLmNoEmbedding, self).__init__(params)
    p = self.params
    p.position_emb.embedding_dim = p.model_dim
    p.stack.name = p.name
    p.stack.model_dim = p.model_dim
    p.softmax.input_dim = p.model_dim
    p.softmax.num_classes = p.vocab_size
    trans_tpl = p.stack.encoder_tpl
    trans_tpl.tr_atten_tpl.residual_dropout_prob = p.residual_dropout_prob
    trans_tpl.tr_atten_tpl.atten_dropout_prob = p.atten_dropout_prob
    trans_tpl.tr_fflayer_tpl.residual_dropout_prob = p.residual_dropout_prob
    trans_tpl.tr_fflayer_tpl.relu_dropout_prob = p.relu_dropout_prob

    with tf.variable_scope(p.name):
      self.CreateChild('position_emb', p.position_emb)

      dropout_tpl = layers.DropoutLayer.Params().Set(
          keep_prob=(1.0 - p.input_dropout_prob))
      self.CreateChild('input_dropout', dropout_tpl)
      self.CreateChild('stack', p.stack)
      self.CreateChild('softmax', p.softmax)
      if p.label_smoother is not None:
        self.CreateChild('smoother', p.label_smoother)

  def FProp(self, theta, inputs, paddings, state0=None, labels=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Input activation. A tensor of shape [time, batch, model_dim].
      paddings: A 0/1 tensor of shape [time, batch].
      state0: Not used for Transformer.
      labels: If not None, a `.NestedMap` containing the following fields: -
        class_weights, a tensor with shape [time, batch] containing the weights
        for each target word. - class_ids, a tensor with shape [time, batch] of
        int32 dtype containing the target class labels. - class_probabilities, a
        tensor with shape [time, batch, vocab_size] of float values indicating
        class-membership probabilities.

    Returns:
      If `labels` is not None, returns (xent_output, None), where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return
      value. Otherwise, `xent_output` only contains the softmax logits.
    """
    p = self.params
    inputs = py_utils.HasRank(inputs, 3)
    tf.logging.info('input shape = {}'.format(inputs.shape))
    seqlen, batch, _ = tf.unstack(tf.shape(inputs), num=3)
    inputs = py_utils.HasShape(inputs, [seqlen, batch, p.model_dim])
    paddings = py_utils.HasShape(paddings, [seqlen, batch])

    # [time, 1, model_dim]
    posit_embs = tf.expand_dims(
        self.position_emb.FProp(theta.position_emb, seqlen), 1)
    # [time, batch, model_dim]
    input_embs = inputs + posit_embs
    input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)
    tf.logging.info('input_embs shape = {}'.format(input_embs.shape))

    layer_out = self.stack.FProp(theta.stack, input_embs, paddings)
    tf.logging.info('layer_out shape = {}'.format(layer_out.shape))

    if not (p.label_smoother is None or p.is_eval):
      # [time, batch, num_classes]
      labels.class_probabilities = self.smoother.FProp(
          theta.smoother, paddings, labels.class_ids, target_ids=None)
      labels.pop('class_ids', None)
    xent_output = ComputeXentOutput(self.softmax, theta.softmax, layer_out,
                                    labels)
    xent_output.last_hidden = layer_out
    return xent_output, None

  def zero_state(self, batch_size):
    return py_utils.NestedMap()


class GPipeTransformerLm(GPipeTransformerLmNoEmbedding):
  """GPipe Transformer based language model layer."""

  @classmethod
  def Params(cls):
    p = super(GPipeTransformerLm, cls).Params()
    p.Define('emb', layers.SimpleEmbeddingLayer.Params(),
             'The embedding layer params.')
    return p

  @classmethod
  def CommonParams(cls,
                   vocab_size,
                   model_dim,
                   hidden_dim=1024,
                   num_heads=8,
                   num_layers=6,
                   splits=1,
                   num_micro_batches=1,
                   num_shards=16,
                   input_dropout_prob=0.0,
                   residual_dropout_prob=0.1,
                   atten_dropout_prob=0.0,
                   relu_dropout_prob=0.0,
                   softmax_max_alloc=None):
    """Common setup for Transformer language models.

    Args:
      vocab_size: vocab size.
      model_dim: model dimension.
      hidden_dim: hidden dimension of feed-forward inner layer.
      num_heads: number of attention heads.
      num_layers: number of layers in the transformer LM.
      splits: list or number of partitions for GPipe.
      num_micro_batches: number of micro batches for GPipe.
      num_shards: num_shards for softmax. Assert vocab_size % num_shards == 0
      input_dropout_prob: dropout prob to the sums of the token embeddings and
        the position embeddings.
      residual_dropout_prob: dropout prob to the output of each sub-layer before
        it is added to the sub-layer input.
      atten_dropout_prob: dropout prob to the attention weights in each
        Transformer attention sub-layer.
      relu_dropout_prob: dropout prob to the inner layer output (ReLU
        activation) in each Transformer feed-forward sub-layer.
      softmax_max_alloc: If set to a positive integer the soft-max computation
        is chunked into allocations of at most softmax_max_alloc; when left to
        its default value of None no chunking is done.

    Returns:
      A Params object containing the parameters that set up a Transformer LM.
    """
    p = cls.Params()
    p.name = 'transformerlm'

    p.model_dim = model_dim
    p.vocab_size = vocab_size
    p.input_dropout_prob = input_dropout_prob
    p.residual_dropout_prob = residual_dropout_prob
    p.atten_dropout_prob = atten_dropout_prob
    p.relu_dropout_prob = relu_dropout_prob

    emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(p.model_dim))
    p.emb.Set(
        use_matmul=False,
        use_3d_weight_tensor=False,
        vocab_size=vocab_size,
        embedding_dim=p.model_dim,
        params_init=emb_params_init)

    p.position_emb.Set(embedding_dim=p.model_dim, trainable_scaling=False)

    p.stack.splits = splits
    p.stack.num_micro_batches = num_micro_batches
    p.stack.num_encoder_layers = num_layers
    trans_tpl = p.stack.encoder_tpl

    trans_tpl.is_decoder = False
    trans_tpl.has_aux_atten = False
    trans_tpl.mask_self_atten = True
    trans_tpl.tr_atten_tpl.is_masked = True
    trans_tpl.tr_atten_tpl.num_attention_heads = num_heads
    trans_tpl.tr_atten_tpl.atten_tpl.enable_ctx_pre_proj = True
    trans_tpl.tr_atten_tpl.atten_tpl.enable_ctx_post_proj = True
    trans_tpl.tr_fflayer_tpl.hidden_dim = hidden_dim
    p.softmax.Set(num_classes=vocab_size, num_shards=num_shards)

    if softmax_max_alloc:
      # If the vocab is very large, computes the softmax chunk-by-chunk.
      p.softmax.chunk_size = max(1, int(softmax_max_alloc / vocab_size))
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeTransformerLm, self).__init__(params)
    p = self.params
    p.emb.embedding_dim = p.model_dim

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert p.emb.embedding_dim == p.position_emb.embedding_dim, (
        '{} vs. {}'.format(p.emb.embedding_dim, p.position_emb.embedding_dim))
    assert p.emb.embedding_dim == p.model_dim, ('{} vs. {}'.format(
        p.emb.embedding_dim, p.model_dim))

    with tf.variable_scope(p.name):
      self.CreateChild('emb', p.emb)

  def FProp(self, theta, inputs, paddings, state0=None, labels=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: Input ids. An int32 tensor of shape [time, batch].
      paddings: A 0/1 tensor of shape [time, batch].
      state0: Not used for Transformer.
      labels: If not None, a `.NestedMap` containing the following fields:  -
        class_weights, a tensor with shape [time, batch] containing the weights
        for each target word. - class_ids, a tensor with shape [time, batch] of
        int32 dtype containing the target class labels. - class_probabilities, a
        tensor with shape [time, batch, vocab_size] of float values indicating
        class-membership probabilities.

    Returns:
      If `labels` is not None, returns (xent_output, state1), where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return
      value and `state1` is the next recurrent state. Otherwise,
      `xent_output` only contains the softmax logits.
    """
    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    activation = self.emb.EmbLookup(theta.emb, ids)
    return super(GPipeTransformerLm, self).FProp(
        theta, activation, paddings, labels=labels)
