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
"""Attention models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import function

from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import py_utils


def _ApplyAttentionDropout(params, x, step_state=None, prng_seed=None):
  """Apply attention dropout according to the given parameters.

  If params.atten_dropout_deterministic is set to True, the dropout will be
  fully deterministic (requires for `step_state` and `prng_seed`).

  Args:
    params: The parameters of attention layer.
    x: A float Tensor on which to apply dropout.
    step_state: (Optional) A NestedMap contains 'global_step' and 'time_step'.
      Required for deterministic dropout.
    prng_seed: (Optional) An int seed for pseudo random number generator.
      Required for deterministic dropout.

  Returns:
    A Tensor with the same shape as `x`.
  """
  if params.atten_dropout_prob == 0:
    return x

  if params.atten_dropout_deterministic:
    if isinstance(step_state, py_utils.NestedMap):
      assert 'global_step' in step_state, step_state.DebugString()
      assert 'time_step' in step_state, step_state.DebugString()
      assert prng_seed is not None
      seeds = prng_seed + tf.stack(
          [step_state.global_step, step_state.time_step])
    else:
      assert prng_seed is not None
      seeds = py_utils.GetOpSeedPair(prng_seed)

    return py_utils.DeterministicDropout(x, 1.0 - params.atten_dropout_prob,
                                         seeds)
  else:
    seed = None if not params.random_seed else prng_seed
    return tf.nn.dropout(x, 1.0 - params.atten_dropout_prob, seed=seed)


class BaseAttentionLayer(base_layer.LayerBase):
  """A base class for all attention layers."""

  @classmethod
  def Params(cls):
    p = super(BaseAttentionLayer, cls).Params()
    p.Define('atten_dropout_prob', 0.0,
             'Probability at which we apply dropout to the attention weights.')
    p.Define(
        'atten_dropout_deterministic', False,
        'Whether to dropout in a fully deterministic way, which is more '
        'suitable for TPU.')
    p.Define(
        'random_seed', None,
        'If set, this decides the random seed to apply in dropout. '
        'Only set this random_seed for unit tests.')
    p.Define('packed_input', False,
             'If True, each training example may pack multiple sequences.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a BaseAttentionLayer object."""
    if not params.name:
      raise ValueError('params.name is not set.')
    super(BaseAttentionLayer, self).__init__(params)

    p = self.params
    self._source_init_done = False
    self._prng_seed = py_utils.GenerateSeedFromName(p.name)
    if p.random_seed:
      self._prng_seed += p.random_seed

  def InitForSourcePacked(self,
                          theta,
                          source_vecs,
                          source_contexts,
                          source_padding,
                          source_segment_id=None):
    """Initialize attention for the given source vectors.

    Must set _source_init_done to True in the function.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A single tensor of shape [time, batch_size, source_dim].
      source_contexts: A single tensor of shape [time, batch_size, some_dim].
      source_padding: A tensor of shape [time, batch_size].
      source_segment_id: A tensor of shape [time, batch_size].
          source_segment_id is not None for packed inputs where one training
          example may pack multiple sequences.

    Note: source_segment_id, if present, should always have the same shape as
    source_padding.

    Returns:
      A tuple (concated_source_vecs, concated_source_contexts, source_padding,
               source_segment_id),
      where concated_source_vecs is a tensor of shape [time, batch_size,
      hidden_dim], concated_source_contexts is a tensor of shape [batch_size,
      time, some_dim], source_padding is a tensor of shape [time,
      batch_size], source_segment_id is a tensor of shape [time, batch_size].
      Note the mismatch between concated_source_vecs and
      concated_source_contexts. In concated_source_vecs, time is the first dim,
      while it is the second dim in concated_source_contexts.
    """
    raise NotImplementedError('Abstract method.')

  def ComputeContextVectorWithSource(self,
                                     theta,
                                     concated_source_vecs,
                                     concated_source_contexts,
                                     source_padding,
                                     source_segment_id,
                                     query_vec,
                                     attention_state=None,
                                     per_step_source_padding=None,
                                     step_state=None,
                                     query_segment_id=None):
    """Computes the context vector given the current query output.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      concated_source_vecs: Concated source vectors with shape [time,
        batch_size, hidden_dim].
      concated_source_contexts: Concated source contexts with shape [
        batch_size, time, context_dim].
      source_padding: Source padding with shape [time, batch_size].
      source_segment_id: Source segment ids with shape [time, batch_size].
      query_vec: a tensor of shape [batch_size, query_dim].
      attention_state: previous attention state.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch_size,
        source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: a tensor of shape [batch_size].

    Returns:
      The attention context vector.
      The attention probability vector.
      The new attention mechanism state: possibly nested tuple of tensors with
        dimensions [target_batch....]
    """
    raise NotImplementedError('Abstract method.')

  def ComputeContextVector(self,
                           theta,
                           query_vec,
                           attention_state=None,
                           per_step_source_padding=None,
                           step_state=None,
                           query_segment_id=None):
    """Computes the context vector given the current query output.

    Unlike ComputeContextVectorWithSource, which explicitly asks for the source
    tensors (concated_source_vecs, concated_source_contexts, source_padding),
    ComputeContextVector uses the class' internal variables.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      query_vec: a tensor of shape [batch_size, query_dim].
      attention_state: previous attention state.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch_size,
        source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: a tensor of shape [batch_size].

    Returns:
      The attention context vector.
      The attention probability vector.
      The new attention mechanism state: possibly nested tuple of tensors with
        dimensions [target_batch....]
    """
    assert self._source_init_done
    return self.ComputeContextVectorWithSource(
        theta, self._concated_source_vecs, self._concated_source_contexts,
        self._source_padding, self._source_segment_id, query_vec,
        attention_state, per_step_source_padding, step_state, query_segment_id)

  def GetInitializationSourceState(self):
    """Gets the attention initialization state.

    The base class only preserves the concated_source_vecs,
    concated_source_contexts and source_padding. If subclasses use more
    state than this and need to interact with inference code that must
    fetch and reload state, this and SetInitializationSourceState must
    be overridden.

    Returns:
      A NestedMap of Tensors that can be preserved and reset via
      SetInitializationSourceState() at a later point. This allows, for example,
      for attention computations to span session runs.
    """
    assert self._source_init_done
    return py_utils.NestedMap(
        concated_source_vecs=self._concated_source_vecs,
        concated_source_contexts=self._concated_source_contexts,
        source_padding=self._source_padding)

  def SetInitializationSourceState(self, new_init_state):
    """Sets the attention initialization state.

    Args:
      new_init_state: A NestedMap matching what was returned from
      GetInitializationSourceState, which will return this layer to that
      initialization state.
    """
    self._source_init_done = True
    self._concated_source_vecs = new_init_state.concated_source_vecs
    self._concated_source_contexts = new_init_state.concated_source_contexts
    self._source_padding = new_init_state.source_padding

  def _PaddedSoftmax(self, logits, padding):
    """Performs a softmax as if padding were applied after exponentiation.

    The default implementation uses numerical techniques to approximate this
    with a standard tf.nn.softmax (using large negative logits for padded
    values). It defers to a Defun that may be replaced on low-range
    implementations with a version that is numerically correct.

    Args:
      logits: Logits.
      padding: Padding (must be the same shape as logits).
    Returns:
      Result of the softmax.
    """
    assert logits.dtype.is_floating
    assert hasattr(logits.dtype, 'max')
    very_negative_logits = (
        tf.ones_like(logits) * logits.dtype.max * tf.constant(
            -0.7, dtype=logits.dtype))
    padded_logits = tf.where(padding > 0.0, very_negative_logits, logits)
    return tf.nn.softmax(padded_logits)

  def _UpdatePaddingWithPackedInputMask(self, padding, source_segment_ids,
                                        query_segment_ids):
    """Creates an attention mask based on source and query segment ids.

    This creates a mask that removes invalid attention, where the query vector
    might assign some weight to neighboring sequences in a packed input example.
    Assumes n = target_batch // source_batch.

    Args:
      padding: Padding for logits, a tensor of shape [time, n, source_batch].
      source_segment_ids: a tensor of shape [time, source_batch]
      query_segment_ids: a tensor of shape [target_batch]

    Returns:
      Logits with mask applied.
    """
    # Generating packed input mask for attention padding.
    source_segment_ids = tf.expand_dims(source_segment_ids, 1)
    query_segment_ids = tf.reshape(
        query_segment_ids, [1, -1, tf.shape(source_segment_ids)[2]])
    padding = tf.where(
        tf.equal(source_segment_ids, query_segment_ids), padding,
        tf.ones_like(padding))
    return padding


class AdditiveAttention(BaseAttentionLayer):
  """Implements attention model as described in this paper.

  http://arxiv.org/pdf/1409.0473v6.pdf
  """

  @classmethod
  def Params(cls):
    """Params for this AdditiveAttention class."""
    p = super(AdditiveAttention, cls).Params()
    p.Define('source_dim', 0, 'Number of source nodes.')
    p.Define('query_dim', 0, 'Number of query nodes.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    # Fill in reasonable default for params init
    p.params_init.method = 'gaussian_sqrt_dim'
    p.params_init.scale = 1.0
    p.Define(
        'same_batch_size', False,
        'True iff the source and target sequence has the same batch size.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs an AdditiveAttention object."""
    super(AdditiveAttention, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      pc = py_utils.WeightParams(
          shape=None,
          init=p.params_init,
          dtype=p.dtype,
          collections=['AdditiveAttention_vars'])
      source_var_shape = [p.source_dim, p.hidden_dim]
      pc.shape = source_var_shape
      self.CreateVariable('source_var', pc, self.AddGlobalVN)
      query_var_shape = [p.query_dim, p.hidden_dim]
      pc.shape = query_var_shape
      self.CreateVariable('query_var', pc, self.AddGlobalVN)
      hidden_var_shape = [p.hidden_dim]
      pc.shape = hidden_var_shape
      self.CreateVariable('hidden_var', pc, self.AddGlobalVN)

    # noinline and compiled cannot be set at the same time
    @function.Defun(
        *([layers.FPropDtype(p)] * 7), noinline=not py_utils.use_tpu())
    def AttenProbs(concated_source_vecs, source_padding, query_vec_reshaped, v,
                   per_step_source_padding, source_segment_id,
                   query_segment_id):
      """Generates probs."""
      source_batch = py_utils.GetShape(source_padding)[1]
      target_batch = py_utils.GetShape(per_step_source_padding)[0]
      multiplier = target_batch // source_batch

      # Shape of summed is [sl, tb/sb, sb, hidden_dim].
      summed = tf.tanh(concated_source_vecs + query_vec_reshaped)
      # logits is of shape [sl * tb/sb * sb, 1]. Computes dot product
      # between v with every rows in 'summed'. Then we reshape the
      # result to be of shape [sl, tb/sb, sb].
      #
      # Another equivalent way is to do:
      #  logits = tf.reduce_sum(summed *
      #                         tf.reshape(v, [1, 1, 1, hidden_dim]), 3)
      logits = py_utils.Matmul(
          tf.reshape(summed, [-1, p.hidden_dim]),
          tf.reshape(v, [p.hidden_dim, 1]))
      logits = tf.reshape(logits, tf.shape(summed)[:3])
      # Take out the padding states.
      # _source_padding is of shape [source_length, source_batch].
      # reshaped to [source_length, 1, source_batch].
      # per_step_source_padding is reshaped to the same but with 'multiplier'
      # for the second dim.
      source_padding = tf.expand_dims(source_padding, 1)
      per_step_source_padding = tf.reshape(
          tf.transpose(per_step_source_padding), [-1, multiplier, source_batch])
      source_padding += per_step_source_padding

      if p.packed_input:
        source_padding = self._UpdatePaddingWithPackedInputMask(
            source_padding, source_segment_id, query_segment_id)
      # Reshape logits to a matrix of shape [target_batch, source_length] and
      # takes the softmax to compute the probabilities.
      logits = tf.transpose(tf.reshape(logits, [-1, target_batch]))
      source_padding = tf.transpose(
          tf.reshape(source_padding, [-1, target_batch]))
      probs = self._PaddedSoftmax(logits, source_padding)
      return probs

    # Adds the atten function into the graph's library.
    def Atten(v, w, source_padding, source_segment_id, concated_source_vecs,
              concated_source_contexts, query_vec, query_segment_id,
              per_step_source_padding, step_state):
      """Computes the attention context vector.

      Args:
        v: hidden weight. [hidden_dim, 1].
        w: query weight. [query_dim, hidden_dim].
        source_padding: [source_length, source_batch].
        source_segment_id: [source_lentgh, source_batch]
        concated_source_vecs: [source_length, source_batch, hidden_dim].
        concated_source_contexts: [source_batch, source_length, context_dim]
        query_vec: [target_batch, query_dim]
        query_segment_id: [target_batch]
        per_step_source_padding: [target_batch, source_length]
        step_state: A NestedMap containing 'global_step' and 'time_step'.
          Required for deterministic dropout.

      Returns:
        attention context vectors and probabilities.
      """
      source_batch = py_utils.GetShape(concated_source_vecs)[1]
      target_batch = py_utils.GetShape(query_vec)[0]
      multiplier = target_batch // source_batch
      # concated_source_vecs is reshaped to
      # [source_length, 1, source_batch, hidden_dims]
      concated_source_vecs = tf.expand_dims(concated_source_vecs, 1)
      query_vec_transformed = py_utils.Matmul(query_vec, w)

      # query_vec is reshaped to
      # [1, target_batch/source_batch, source_batch, hidden_dims].
      query_vec_reshaped = tf.reshape(
          query_vec_transformed, [1, multiplier, source_batch, p.hidden_dim])
      # logits is of shape
      # [source_length, target_batch/source_batch, source_batch]
      probs = AttenProbs(concated_source_vecs, source_padding,
                         query_vec_reshaped, v, per_step_source_padding,
                         source_segment_id, query_segment_id)

      # Apply dropout to weights if applicable.
      if not p.is_eval:
        probs = _ApplyAttentionDropout(p, probs, step_state, self._prng_seed)

      # Reshape probs to be of shape
      # [target_batch/source_batch, source_batch, source_length]
      probs_reshaped = tf.reshape(probs, [multiplier, source_batch, -1])
      # Transpose probs to be of shape
      # [source_batch, target_batch/source_batch, source_length]
      probs_reshaped = tf.transpose(probs_reshaped, [1, 0, 2])
      # Batched matmul
      # [source_batch, target_batch/source_batch, source_length] *
      # [source_batch, source_length, context_dim] =
      # [source_batch, target_batch/source_batch, context_dim]
      summed = tf.matmul(probs_reshaped, concated_source_contexts)

      # summed is of shape
      # [target_batch/source_batch, source_batch, context_dim]
      summed = tf.transpose(summed, [1, 0, 2])

      return tf.reshape(summed, [target_batch, -1]), probs

    # The source batch size equals to the target batch size.
    def AttenSameBatchSize(v, w, source_padding, source_segment_id,
                           concated_source_vecs, concated_source_contexts,
                           query_vec, query_segment_id, per_step_source_padding,
                           step_state):
      """Computes the attention context vector.

      Args:
        v: hidden weight. [hidden_dim].
        w: query weight. [query_dim, hidden_dim].
        source_padding: [sl, b]
        source_segment_id: [sl, b]
        concated_source_vecs: [sl, b, hidden_dim].
        concated_source_contexts: [b, sl, hidden_dim]
        query_vec: [b, query_dim]
        query_segment_id: [b]
        per_step_source_padding: [b, sl]
        step_state: A NestedMap containing 'global_step' and 'time_step'.
          Required for deterministic dropout.

      Returns:
        attention context vectors and probabilities.
      """
      # TODO(jiaye): support dropout
      if p.atten_dropout_prob != 0:
        raise NotImplementedError('dropout is not supported')
      del step_state

      # [b, hidden_dim]
      query_vec = py_utils.Matmul(query_vec, w)
      # [sl, b]
      @function.Defun(*([p.dtype] * 7), noinline=not py_utils.use_tpu())
      def AttenProbs(x, source_padding, y, v, per_step_source_padding,
                     source_segment_id, query_segment_id):
        """Calculates atten probs with padding."""
        # tf.tanh(x+y) shape [sl, b, hidden_dim]
        summed = tf.tanh(x + y)
        # [-1, hidden_dim] * [hidden_dim, 1] = [-1, 1]
        res = py_utils.Matmul(
            tf.reshape(summed, [-1, p.hidden_dim]), tf.expand_dims(v, 1))
        # Reshape res to [sl, b]
        logits = tf.reshape(res, tf.shape(summed)[:2])
        # Take out the padding states. _source_padding is of shape [sl, b].
        source_padding += tf.transpose(per_step_source_padding)

        if p.packed_input:
          source_padding = self._UpdatePaddingWithPackedInputMask(
              tf.expand_dims(source_padding, 1), source_segment_id,
              query_segment_id)
          source_padding = tf.squeeze(source_padding, 1)
        # [b, sl]
        source_padding = tf.transpose(source_padding)
        logits = tf.transpose(logits)
        # softmax to compute the probabilities. [b, sl]
        probs = self._PaddedSoftmax(logits, source_padding)
        return probs

      probs = AttenProbs(concated_source_vecs, source_padding, query_vec, v,
                         per_step_source_padding, source_segment_id,
                         query_segment_id)

      # contexts[i, :] is a weighted (probs[i, :]) average of
      # concated_source_vecs[i, :, :].
      # Reshaped probs is of shape [b, 1, sl]
      reshaped_probs = tf.expand_dims(probs, 1)
      # [b, 1, sl] * [b, sl, hidden_dim] = [b, 1, hidden_dim]
      contexts = tf.matmul(reshaped_probs, concated_source_contexts)
      # Reshaped context is of shape [b, hidden_dim]
      contexts = tf.squeeze(contexts, axis=1)
      return contexts, probs

    if p.same_batch_size:
      self._ctx_vec = AttenSameBatchSize
    else:
      self._ctx_vec = Atten

    def EncodeSource(src_w, vecs, ctxs):
      time, batch = py_utils.GetShape(vecs, 2)
      ctxs = py_utils.HasShape(ctxs, [time, batch, -1])
      transformed_vecs = tf.reshape(
          py_utils.Matmul(tf.reshape(vecs, [-1, p.source_dim]), src_w),
          [time, batch, -1])
      transposed_ctxs = tf.transpose(ctxs, [1, 0, 2])
      return transformed_vecs, transposed_ctxs

    self._encode_source = EncodeSource

  def PackSource(self,
                 theta,
                 source_vecs,
                 source_contexts,
                 source_padding,
                 source_segment_id=None):
    """Packs source vectors. Does not change attention state.

    Unlike the InitForSource API above, this API takes in a single tensor
    representing the entire sequence.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A single tensor of shape [time, batch_size, source_dim].
      source_contexts: A single tensor of shape [time, batch_size, some_dim].
      source_padding: A tensor of shape [time, batch_size].
      source_segment_id: A tensor of shape [time, batch_size].

    Returns:
      Concated source vectors, concated source contexts, and source paddings.
    """
    with tf.name_scope(self.params.name):
      if source_segment_id is None:
        source_segment_id = tf.zeros_like(source_padding)

      (concated_source_vecs, concated_source_contexts) = (
          self._encode_source(theta.source_var, source_vecs, source_contexts))
    return (concated_source_vecs, concated_source_contexts, source_padding,
            source_segment_id)

  def InitForSourcePacked(self,
                          theta,
                          source_vecs,
                          source_contexts,
                          source_padding,
                          source_segment_id=None):
    """Initialize attention for the given source vectors.

    Unlike the InitForSource API above, this API takes in a single tensor
    representing the entire sequence.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A single tensor of shape [time, batch_size, source_dim].
      source_contexts: A single tensor of shape [time, batch_size, some_dim].
      source_padding: A tensor of shape [time, batch_size].
      source_segment_id: A tensor of shape [time, batch_size].

    Returns:
      Concated source vectors, concated source contexts, and source paddings.
    """
    self._source_init_done = True

    (self._concated_source_vecs, self._concated_source_contexts,
     self._source_padding, self._source_segment_id) = self.PackSource(
         theta, source_vecs, source_contexts, source_padding, source_segment_id)
    return (self._concated_source_vecs, self._concated_source_contexts,
            self._source_padding, self._source_segment_id)

  def ZeroAttentionState(self, source_seq_length, decoder_batch_size):
    p = self.params
    # This is just a dummy state. The first dimension of the state has to match
    # decoder_batch_size.
    zs = tf.zeros([decoder_batch_size, 1], dtype=layers.FPropDtype(p))
    return zs

  def ComputeContextVectorWithSource(self,
                                     theta,
                                     concated_source_vecs,
                                     concated_source_contexts,
                                     source_padding,
                                     source_segment_id,
                                     query_vec,
                                     attention_state=None,
                                     per_step_source_padding=None,
                                     step_state=None,
                                     query_segment_id=None):
    """Computes the context vector given the current query output.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      concated_source_vecs: Concated source vectors with shape [time,
        batch_size, hidden_dim].
      concated_source_contexts: Concated source contexts with shape [
        batch_size, time, context_dim].
      source_padding: Source padding with shape [time, batch_size].
      source_segment_id: Tensor of source segment ids, with shape [time,
        batch_size].
      query_vec: a tensor of shape [batch_size, query_dim].
      attention_state: previous attention state. It is not used in
          AdditiveAttention, and is simply passed through.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch_size,
        source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: a tensor of shape [batch_size]

    Returns:
      The attention context vector.
      The attention probability vector.
      The new attention mechanism state: possibly nested tuple of tensors with
        dimensions [target_batch....]
    """
    p = self.params
    query_batch_size = py_utils.GetShape(query_vec)[0]
    source_seq_length = py_utils.GetShape(source_padding)[0]
    if per_step_source_padding is None:
      zero = tf.constant(0.0, dtype=query_vec.dtype)
      per_step_source_padding = tf.fill([query_batch_size, source_seq_length],
                                        zero)
    per_step_source_padding = py_utils.HasShape(
        per_step_source_padding, [query_batch_size, source_seq_length])
    hidden = py_utils.AddPerStepVN(p, theta.hidden_var)
    query = py_utils.AddPerStepVN(p, theta.query_var)

    if source_segment_id is None:
      source_segment_id = tf.zeros_like(source_padding)
    if query_segment_id is None:
      query_segment_id = tf.zeros(
          tf.shape(query_vec)[0], dtype=source_padding.dtype)

    ctx_vec, prob = self._ctx_vec(
        hidden, query, source_padding, source_segment_id, concated_source_vecs,
        concated_source_contexts, query_vec, query_segment_id,
        per_step_source_padding, step_state)

    return ctx_vec, prob, attention_state


class DotProductAttention(BaseAttentionLayer):
  """Implements dot-product attention.
  """

  @classmethod
  def Params(cls):
    """Params for DotProductAttention."""
    p = super(DotProductAttention, cls).Params()
    p.Define('source_dim', 0, 'Number of source nodes.')
    p.Define('query_dim', 0, 'Number of query nodes.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a DotProductAttention object."""
    super(DotProductAttention, self).__init__(params)
    p = self.params
    # TODO(yonghui): relax these constraints.
    assert p.source_dim == p.query_dim
    assert p.source_dim == p.hidden_dim

    with tf.variable_scope(p.name):
      pc = py_utils.WeightParams(
          shape=[p.hidden_dim],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=['DotProductAttention_vars'])

      def ScaleFn(x):
        return tf.nn.softplus(x) / tf.nn.softplus(
            tf.constant(0.0, dtype=x.dtype))

      self.CreateVariable('per_dim_scale', pc, ScaleFn)

    @function.Defun(
        *[layers.FPropDtype(p)] * 7, noinline=not py_utils.use_tpu())
    def AttenProbs(per_dim_scale, source_padding, concated_source_vecs,
                   query_vec, per_step_source_padding, source_segment_id,
                   query_segment_id):
      """Main attention function.

      Args:
        per_dim_scale:            [source_dim], a vec to scale individual dims.
        source_padding:           [time, source_batch].
        concated_source_vecs:     [time, source_batch, source_dim].
        query_vec:                [target_batch, source_dim].
        per_step_source_padding:  [target_batch, source_seq_length]
        source_segment_id:        [time, source_batch].
        query_segment_id:         [target_batch].

      Returns:
        logits: [target_batch, source_time].

      target_batch = source_batch * n where n is an integer >= 1.
      In this case query_vec contains:
              -------------------------
              | instance    1         |
              | instance    2         |
           0  |          ...          |
              | instance source_batch |
              -------------------------
              | instance    1         |
              | instance    2         |
           1  |          ...          |
              | instance source_batch |
              -------------------------
                           ...
              -------------------------
              | instance    1         |
              | instance    2         |
          n-1 |          ...          |
              | instance source_batch |
              -------------------------
      One use case is beam search where n = beam size.
      """
      source_padding = tf.transpose(source_padding)
      concated_source_vecs = tf.transpose(concated_source_vecs, [1, 0, 2])

      logit_scale = tf.stop_gradient(
          tf.rsqrt(tf.cast(tf.shape(query_vec)[1], dtype=layers.FPropDtype(p))))
      source_batch = tf.shape(concated_source_vecs)[0]
      target_batch = tf.shape(query_vec)[0]
      query_vec *= per_dim_scale
      # The n here refers to the "n" described in the comment above.
      n = target_batch // source_batch
      query_vec = tf.reshape(query_vec, [n, source_batch, -1])
      # => [source_batch, source_dim, n]
      query_vec = tf.transpose(query_vec, [1, 2, 0])
      # => [n, source_batch, source_sequence_len]
      per_step_source_padding = tf.reshape(per_step_source_padding,
                                           [n, source_batch, -1])
      # => [source_batch, source_sequence_len, n]
      per_step_source_padding = tf.transpose(per_step_source_padding, [1, 2, 0])
      # Dot-product part.
      # Calls batch_mat_mul since dim > 2 for per-instance matmul.
      # [source_batch, time, source_dim] * [source_batch, source_dim, n]
      # => [source_batch, time, n]
      logits = tf.matmul(concated_source_vecs, query_vec)
      logits *= logit_scale
      # Exclude padding frames.
      # [source_batch, time] => [source_batch, time, 1]
      source_padding = tf.expand_dims(source_padding, 2)
      source_padding += per_step_source_padding
      if p.packed_input:
        source_padding = tf.transpose(source_padding, [1, 2, 0])
        source_padding = self._UpdatePaddingWithPackedInputMask(
            source_padding, source_segment_id, query_segment_id)
        source_padding = tf.transpose(source_padding, [1, 2, 0])
      else:
        source_padding = tf.transpose(source_padding, [2, 0, 1])

      # => [n, source_batch, time]
      logits = tf.transpose(logits, [2, 0, 1])

      # => [n * source_batch, time].
      # This makes logits store content in the same order as query_vec.
      logits = tf.reshape(logits, [target_batch, -1])
      source_padding = tf.reshape(source_padding, [target_batch, -1])
      probs = self._PaddedSoftmax(logits, source_padding)
      return probs

    def Atten(per_dim_scale, source_padding, source_segment_id,
              concated_source_vecs, concated_source_contexts, query_vec,
              query_segment_id, per_step_source_padding, step_state):
      """Main attention function.

      Args:
        per_dim_scale:            [source_dim], a vec to scale individual dims.
        source_padding:           [time, source_batch].
        source_segment_id:        [time, source_batch].
        concated_source_vecs:     [time, source_batch, source_dim].
        concated_source_contexts: [source_batch, time, context_dim].
        query_vec:                [target_batch, source_dim].
        query_segment_id:         [target_batch].
        per_step_source_padding:  [target_batch, source_seq_length]
        step_state:               A NestedMap containing 'global_step' and
                                  'time_step'. Required for deterministic
                                  dropout.

      Returns:
        context_vector: [target_batch, context_dim].
        probs:          [target_batch, time].
      """
      py_utils.assert_shape_match([tf.shape(concated_source_vecs)[2]],
                                  [tf.shape(query_vec)[1]])
      py_utils.assert_shape_match([tf.shape(concated_source_vecs)[2]],
                                  [p.source_dim])
      source_batch = tf.shape(concated_source_vecs)[1]
      target_batch = tf.shape(query_vec)[0]
      n = target_batch // source_batch
      returned_probs = AttenProbs(
          per_dim_scale, source_padding, concated_source_vecs, query_vec,
          per_step_source_padding, source_segment_id, query_segment_id)

      # => [n, source_batch, time].
      probs = tf.reshape(returned_probs, [n, source_batch, -1])
      # => [source_batch, n, time].
      probs = tf.transpose(probs, [1, 0, 2])

      # Apply dropout to weights if applicable.
      if not p.is_eval:
        probs = _ApplyAttentionDropout(p, probs, step_state, self._prng_seed)

      # Weight each frame with the probability and sum them.
      # [source_batch, n, time] * [source_batch, time, context_dim]
      # => [source_batch, n, context_dim].
      context_vector = tf.matmul(probs, concated_source_contexts)
      # => [n, source_batch, context_dim].
      context_vector = tf.transpose(context_vector, [1, 0, 2])
      # => [n * source_batch, context_dim].
      context_vector = tf.reshape(context_vector, [target_batch, -1])

      return context_vector, returned_probs

    self._ctx_vec = Atten

  def PackSource(self,
                 theta,
                 source_vecs,
                 source_contexts,
                 source_padding,
                 source_segment_id=None):
    """Packs source vectors. Does not change attention state.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A tensor of shape [time, source_batch, source_dim].
      source_contexts: A tensor of shape [time, source_batch, context_dim].
      source_padding: A tensor of shape [time, source_batch].
      source_segment_id: A tensor of shape [time, source_batch].

    Returns:
      A tuple (concated_source_vecs, concated_source_contexts, source_padding),
      where concated_source_vecs is a tensor of shape [time, batch_size,
      hidden_dim], concated_source_contexts is a tensor of shape [batch_size,
      time, some_dim] and source_padding is a tensor of shape [time,
      batch_size]. Note the mismatch between concated_source_vecs and
      concated_source_contexts. In concated_source_vecs, time is the first dim,
      while it is the second dim in concated_source_contexts.
    """
    concated_source_vecs = tf.identity(source_vecs)
    concated_source_contexts = tf.transpose(source_contexts, [1, 0, 2])
    if source_segment_id is None:
      source_segment_id = tf.zeros_like(source_padding)
    return (concated_source_vecs, concated_source_contexts, source_padding,
            source_segment_id)

  def InitForSourcePacked(self,
                          theta,
                          source_vecs,
                          source_contexts,
                          source_padding,
                          source_segment_id=None):
    """Initialize attention for the given source vectors.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A tensor of shape [time, source_batch, source_dim].
      source_contexts: A tensor of shape [time, source_batch, context_dim].
      source_padding: A tensor of shape [time, source_batch].
      source_segment_id: A tensor of shape [time, source_batch].

    Returns:
      A tuple (concated_source_vecs, concated_source_contexts, source_padding),
      where concated_source_vecs is a tensor of shape [time, batch_size,
      hidden_dim], concated_source_contexts is a tensor of shape [batch_size,
      time, some_dim] and source_padding is a tensor of shape [time,
      batch_size]. Note the mismatch between concated_source_vecs and
      concated_source_contexts. In concated_source_vecs, time is the first dim,
      while it is the second dim in concated_source_contexts.
    """
    self._source_init_done = True
    (self._concated_source_vecs, self._concated_source_contexts,
     self._source_padding, self._source_segment_id) = self.PackSource(
         theta, source_vecs, source_contexts, source_padding, source_segment_id)
    return (self._concated_source_vecs, self._concated_source_contexts,
            self._source_padding, self._source_segment_id)

  def ZeroAttentionState(self, source_seq_length, decoder_batch_size):
    p = self.params
    # No states to keep track of currently.
    return tf.zeros([decoder_batch_size, 1], dtype=p.dtype)

  def ComputeContextVectorWithSource(self,
                                     theta,
                                     concated_source_vecs,
                                     concated_source_contexts,
                                     source_padding,
                                     source_segment_id,
                                     query_vec,
                                     attention_state=None,
                                     per_step_source_padding=None,
                                     step_state=None,
                                     query_segment_id=None):
    """Computes the context vector given the current query output.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      concated_source_vecs: Concated source vectors with shape [time,
        source_batch, hidden_dim].
      concated_source_contexts: Concated source contexts with shape [
        source_batch, time, context_dim].
      source_padding: Source padding with shape [time, source_batch].
      source_segment_id: Source segment id with shape [time, source_batch].
      query_vec: a tensor of shape [target_batch, query_dim], where
        target_batch = n * source_batch (e.g., n = num_hyps_per_beam in
        beamsearch). Along the target_batch dimension, there are n groups of
        consecutive rows, each group containing source_batch rows.
      attention_state: previous attention state. It is not used in
          AdditiveAttention, and is simply passed through.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch, source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: Query segment id with shape [target_batch].

    Returns:
      The attention context vector.
      The attention probability vector.
      The new attention mechanism state: possibly nested tuple of tensors with
        dimensions [target_batch....]
    """
    query_batch_size = tf.shape(query_vec)[0]
    source_sequence_length = tf.shape(source_padding)[0]
    if per_step_source_padding is None:
      zero = tf.constant(0.0, dtype=query_vec.dtype)
      per_step_source_padding = tf.fill(
          [query_batch_size, source_sequence_length], zero)
    per_step_source_padding = py_utils.HasShape(
        per_step_source_padding, [query_batch_size, source_sequence_length])
    if source_segment_id is None:
      source_segment_id = tf.zeros_like(source_padding)
    if query_segment_id is None:
      query_segment_id = tf.zeros(
          tf.shape(query_vec)[0], dtype=source_padding.dtype)

    ctx_vec, prob = self._ctx_vec(
        theta.per_dim_scale, source_padding, source_segment_id,
        concated_source_vecs, concated_source_contexts, query_vec,
        query_segment_id, per_step_source_padding, step_state)
    return ctx_vec, prob, attention_state


def _RecursiveReshape(x, shape):
  if x is None:
    return None
  elif isinstance(x, py_utils.NestedMap):
    return x.Transform(lambda y: _RecursiveReshape(y, shape))
  else:
    return tf.reshape(x, shape) if x.shape.ndims == 2 else x


class MultiHeadedAttention(BaseAttentionLayer):
  """Attention with multiple attention heads.

  Conceptually, the algorithm works as follows:
    1. Source vectors (attention keys) are first projected to vectors of dim
    p.hidden_dim.
    2. Query vectors are projected to vectors of dim p.hidden_dim as well.
    3. Context vectors (attention values) are not projected.
    4. Source vectors, query vectors and context vectors are all split into
    p.num_attention_heads chunks.
    5. The inner atten mechanism is computed separately on each of the chunks.
    6. Attention contexts from each of the chunk are concatenated to form the
    final context.
    7. Attention probs from each of the chunk are averaged to form the final
    attention prob.
  """

  @classmethod
  def Params(cls):
    """Params for MultiHeadedAttention."""
    p = super(MultiHeadedAttention, cls).Params()
    p.Define('source_dim', 0, 'Number of source nodes.')
    p.Define('query_dim', 0, 'Number of query nodes.')
    p.Define('context_dim', 0, 'Number of context nodes.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    p.Define('num_attention_heads', 2, 'Num of attention heads.')
    p.Define(
        'use_source_vec_as_attention_value', True,
        'Whether or not to use source_vec as the attention value as well.'
        ' If True, we expect source_vec and source_contexts are the same.')
    p.Define('enable_source_proj', True,
             'If False, source side linear projection is disabled.')
    p.Define('enable_query_proj', True,
             'If False, query side linear projection is disabled.')
    p.Define('inner_atten_params', DotProductAttention.Params(),
             'Params for underlying attention mechanism.')
    p.Define(
        'enable_ctx_pre_proj', False,
        'If True, context is pre-projected before processing into'
        ' hidden_dim.')
    p.Define(
        'enable_ctx_post_proj', False,
        'If True, computed context is post projected into'
        ' ctx_post_proj_dim.')
    p.Define('ctx_post_proj_dim', 0, 'Number of post projection nodes.')
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0)

    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a MultiHeadedAttention object."""
    super(MultiHeadedAttention, self).__init__(params)
    p = self.params
    assert p.hidden_dim % p.num_attention_heads == 0

    pc_bias = py_utils.WeightParams(
        shape=[p.hidden_dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      if p.enable_source_proj:
        pc = py_utils.WeightParams(
            shape=[p.source_dim, p.hidden_dim],
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('source_proj', pc)
        self.CreateVariable('source_proj_b', pc_bias)
      else:
        assert p.source_dim == p.hidden_dim

      if p.enable_query_proj:
        pc = py_utils.WeightParams(
            shape=[p.query_dim, p.hidden_dim],
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('query_proj', pc)
        self.CreateVariable('query_proj_b', pc_bias)
      else:
        assert p.query_dim == p.hidden_dim

      if p.enable_ctx_pre_proj and not p.use_source_vec_as_attention_value:
        assert p.context_dim
        pc = py_utils.WeightParams(
            shape=[p.context_dim, p.hidden_dim],
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('ctx_proj', pc)
        self.CreateVariable('ctx_proj_b', pc_bias)

      if p.enable_ctx_post_proj:
        assert p.ctx_post_proj_dim
        pc = py_utils.WeightParams(
            shape=[p.hidden_dim, p.ctx_post_proj_dim],
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('ctx_post_proj', pc)
        pc_bias_post_proj = py_utils.WeightParams(
            shape=[p.ctx_post_proj_dim],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('ctx_post_proj_b', pc_bias_post_proj)

      att_dim = p.hidden_dim // p.num_attention_heads

      att_p = p.inner_atten_params.Set(
          source_dim=att_dim,
          query_dim=att_dim,
          hidden_dim=att_dim,
          dtype=p.dtype,
          random_seed=p.random_seed,
          atten_dropout_prob=p.atten_dropout_prob,
          atten_dropout_deterministic=p.atten_dropout_deterministic,
          packed_input=p.packed_input)
      py_utils.SetNameIfNone(att_p, 'inner_att')
      self.CreateChild('atten', att_p)

  def InitForSourcePacked(self,
                          theta,
                          source_vecs,
                          source_contexts,
                          source_padding,
                          source_segment_id=None):
    """Initialize attention for the given source vectors.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A tensor of shape [time, source_batch, source_dim].
      source_contexts: A tensor of shape [time, source_batch, context_dim].
      source_padding: A tensor of shape [time, source_batch].
      source_segment_id: A tensor of shape [time, source_batch].

    Returns:
      (concated_source_vecs, concated_source_contexts, source_padding,
      source_segment_id) tuple where concated_source_vecs is a tensor of shape
      [source_seq_len, batch_size * num_heads, orig_source_dim / num_heads],
      concated_source_contexts is a tensor of shape [source_batch_size *
      num_heads, source_seq_len,  orig_context_dim / num_heads],
      source_padding is a tensor of shape [source_seq_len, batch_size *
      num_heads] and source_segment_id is a tensor of shape
      [source_seq_len, batch_size * num_heads].
    """
    self._source_init_done = True
    (self._concated_source_vecs, self._concated_source_contexts,
     self._source_padding, self._source_segment_id) = self.PackSource(
         theta, source_vecs, source_contexts, source_padding, source_segment_id)
    return (self._concated_source_vecs, self._concated_source_contexts,
            self._source_padding, self._source_segment_id)

  def PackSource(self,
                 theta,
                 source_vecs,
                 source_contexts,
                 source_padding,
                 source_segment_id=None):
    """Packs source vectors. Does not change attention state.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A tensor of shape [time, source_batch, source_dim].
      source_contexts: A tensor of shape [time, source_batch, context_dim].
      source_padding: A tensor of shape [time, source_batch].
      source_segment_id: A tensor of shape [time, source_batch].

    Returns:
      (concated_source_vecs, concated_source_contexts, source_padding,
      source_segment_id) tuple where concated_source_vecs is a tensor of shape
      [source_seq_len, batch_size * num_heads, orig_source_dim / num_heads],
      concated_source_contexts is a tensor of shape [source_batch_size *
      num_heads, source_seq_len,  orig_context_dim / num_heads],
      source_padding is a tensor of shape [source_seq_len, batch_size *
      num_heads] and source_segment_id is a tensor of shape
      [source_seq_len, batch_size * num_heads].
    """

    p = self.params
    if not p.enable_source_proj:
      assert p.source_dim == p.hidden_dim
    if not p.enable_query_proj:
      assert p.query_dim == p.hidden_dim
    with tf.name_scope('init__0'):
      if p.use_source_vec_as_attention_value:
        source_vecs = py_utils.HasShape(source_vecs, tf.shape(source_contexts))
      time_steps = tf.shape(source_vecs)[0]
      batch_size = tf.shape(source_vecs)[1]
      # source_projected shape [time * source_batch, hidden]
      with tf.name_scope('init__0a'):
        source_vec_depth = tf.shape(source_vecs)[2]
      with tf.name_scope('init__0b'):
        if p.enable_source_proj:
          source_projected = (
              tf.matmul(
                  tf.reshape(source_vecs, [-1, source_vec_depth]),
                  theta.source_proj))
          source_projected += theta.source_proj_b
        else:
          source_projected = tf.reshape(source_vecs, [-1, source_vec_depth])
    with tf.name_scope('init__1'):
      hidden_depth = p.hidden_dim
      num_heads = p.num_attention_heads
      # => [time, source_batch * num_heads, hidden / num_heads]
      source_projected = tf.reshape(
          source_projected,
          [time_steps, batch_size * num_heads, hidden_depth // num_heads])
      if p.use_source_vec_as_attention_value:
        source_contexts_reshaped = source_projected
      else:
        if p.enable_ctx_pre_proj:
          source_context_depth = tf.shape(source_contexts)[2]
          source_contexts_projected = tf.matmul(
              tf.reshape(source_contexts, [-1, source_context_depth]),
              theta.ctx_proj)
          source_contexts_projected += theta.ctx_proj_b
        else:
          source_contexts_projected = source_contexts
        source_contexts_reshaped = tf.reshape(
            source_contexts_projected, [time_steps, batch_size * num_heads, -1])

    with tf.name_scope('init__2'):
      source_padding_replicated = tf.reshape(
          tf.tile(
              tf.reshape(source_padding, [time_steps, batch_size, 1]),
              [1, 1, num_heads]), [time_steps, batch_size * num_heads])
      if source_segment_id is None:
        source_segment_id_repl = tf.zeros_like(source_padding_replicated)
      else:
        source_segment_id_repl = tf.reshape(
            tf.tile(
                tf.reshape(source_segment_id, [time_steps, batch_size, 1]),
                [1, 1, num_heads]), [time_steps, batch_size * num_heads])

      (concated_source_vecs, concated_source_contexts,
       source_padding, source_segment_id) = self.atten.PackSource(
           theta.atten, source_projected, source_contexts_reshaped,
           source_padding_replicated, source_segment_id_repl)
      return (concated_source_vecs, concated_source_contexts, source_padding,
              source_segment_id)

  def ExtendSourcePacked(self, theta, new_source_vecs, new_source_contexts,
                         new_source_paddings, new_source_segment_ids,
                         cached_prev_source_vecs, cached_prev_source_contexts,
                         cached_prev_source_paddings,
                         cached_prev_source_segment_ids):
    """Extend cached source_vecs and source_contexts by one more timestep.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      new_source_vecs: A tensor of shape [source_batch, source_dim].
      new_source_contexts: A tensor of shape [source_batch, context_dim].
        new_source_vecs and new_source_contexts are source_vecs and
        source_contexts for the new timestep to be extended.
      new_source_paddings: If not None, a tensor of shape [source_batch].
        source_padding for the new timestep.
      new_source_segment_ids: If not None, a tensor of shape [source_batch].
        source_segment_id for the new timestep.

      cached_prev_source_vecs: A tensor of shape [source_batch,
        t - 1, hidden_dim].
      cached_prev_source_contexts: A tensor of shape [source_batch,
        t - 1, hidden_dim].
        'cached_prev_source_vecs' and 'cached_prev_source_contexts' are the
        already preprocessed source_vecs and source_contexts for the previous
        t-1 steps.
      cached_prev_source_paddings: If not None, a tensor of shape [source_batch,
        t - 1, num_heads], cached source padding for the previous t - 1
        timesteps.
      cached_prev_source_segment_ids: If not None, a tensor of shape
        [source_batch, t - 1, num_heads], cached source segment id for the
        previous t - 1 timesteps.
    Returns:
      Extended cached source_vecs, source_contexts and source_paddings.
      'extended_source_vec' is of shape [batch_size, t, num_heads * dim],
      'extended_source_context' is of shape [batch_size, t, num_heads * dim],
      source_padding is of shape [batch_size, t, num_heads], source_segment_id
      is of shape [batch_size, t, num_heads].
    """
    p = self.params
    batch_size = tf.shape(new_source_vecs)[0]
    hidden_dim = p.hidden_dim
    num_heads = p.num_attention_heads
    if new_source_paddings is None:
      new_source_paddings = tf.zeros([batch_size], dtype=new_source_vecs.dtype)
    if new_source_segment_ids is None:
      new_source_segment_ids = tf.zeros(
          [batch_size], dtype=new_source_vecs.dtype)
    (processed_source_vecs, processed_source_contexts,
     processed_source_paddings,
     processed_source_segment_ids) = self.InitForSourcePacked(
         theta, tf.expand_dims(new_source_vecs, 0),
         tf.expand_dims(new_source_contexts, 0),
         tf.expand_dims(new_source_paddings, 0),
         tf.expand_dims(new_source_segment_ids, 0))
    processed_source_vecs = tf.reshape(processed_source_vecs,
                                       [batch_size, 1, hidden_dim])
    processed_source_contexts = tf.reshape(processed_source_contexts,
                                           [batch_size, 1, -1])
    processed_source_paddings = tf.reshape(processed_source_paddings,
                                           [batch_size, 1, num_heads])
    processed_source_segment_ids = tf.reshape(processed_source_segment_ids,
                                              [batch_size, 1, num_heads])
    cached_source_vecs = tf.concat(
        [cached_prev_source_vecs, processed_source_vecs], axis=1)
    cached_source_contexts = tf.concat(
        [cached_prev_source_contexts, processed_source_contexts], axis=1)
    if cached_prev_source_paddings is None:
      cached_source_paddings = None
    else:
      cached_source_paddings = tf.concat(
          [cached_prev_source_paddings, processed_source_paddings], axis=1)
    if cached_prev_source_segment_ids is None:
      cached_source_segment_ids = None
    else:
      cached_source_segment_ids = tf.concat(
          [cached_prev_source_segment_ids, processed_source_segment_ids],
          axis=1)
    return (cached_source_vecs, cached_source_contexts, cached_source_paddings,
            cached_source_segment_ids)

  def ZeroAttentionState(self, source_seq_length, decoder_batch_size):
    zero_att_state = self.atten.ZeroAttentionState(
        source_seq_length, decoder_batch_size * self.params.num_attention_heads)
    # [batch * num_heads, length] => [batch, num_heads * length].
    zero_att_state = _RecursiveReshape(zero_att_state, [decoder_batch_size, -1])
    return zero_att_state

  def ComputeContextVectorWithSource(self,
                                     theta,
                                     concated_source_vecs,
                                     concated_source_contexts,
                                     source_padding,
                                     source_segment_id,
                                     query_vec,
                                     attention_state=None,
                                     per_step_source_padding=None,
                                     step_state=None,
                                     query_segment_id=None):
    """Computes the context vector given the current query output.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      concated_source_vecs: Concated source vectors with shape [time,
        batch_size, hidden_dim].
      concated_source_contexts: Concated source contexts with shape [
        batch_size, time, context_dim].
      source_padding: Source padding with shape [time, batch_size].
      source_segment_id: Source segment id with shape [time, batch_size].
      query_vec: a tensor of shape [target_batch, query_dim].
      attention_state: previous attention state. It is not used in
          AdditiveAttention, and is simply passed through.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch_size,
        source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: a tensor of shape [target_batch].

    Returns:
      The attention context vector:     [target_batch, source_dim]
      The attention probability vector: [target_batch, time]
      The new attention mechanism state: possibly nested tuple of tensors with
        dimensions [target_batch....]
    """
    p = self.params
    source_seq_len = tf.shape(source_padding)[0]
    num_heads = p.num_attention_heads
    batch_size = tf.shape(query_vec)[0]

    if p.enable_query_proj:
      query_vec_projected = tf.matmul(query_vec, theta.query_proj)
      query_vec_projected += theta.query_proj_b
      query_vec_projected = tf.reshape(
          query_vec_projected,
          [batch_size * num_heads, p.hidden_dim // num_heads])
    else:
      query_vec_projected = tf.reshape(
          query_vec, [batch_size * num_heads, p.hidden_dim // num_heads])

    query_batch_size = tf.shape(query_vec)[0]
    if query_segment_id is None:
      query_segment_id = tf.zeros(
          query_batch_size * num_heads, dtype=source_padding.dtype)
    else:
      query_segment_id_repl = tf.tile(
          tf.expand_dims(query_segment_id, 1), [1, num_heads])
      query_segment_id = tf.reshape(query_segment_id_repl, [-1])

    if per_step_source_padding is None:
      zero = tf.constant(0.0, dtype=query_vec.dtype)
      per_step_source_padding = tf.fill([query_batch_size, source_seq_len],
                                        zero)
    per_step_source_padding = py_utils.HasShape(
        per_step_source_padding, [query_batch_size, source_seq_len])
    per_step_source_padding = tf.reshape(
        tf.tile(per_step_source_padding, [1, num_heads]), [-1, source_seq_len])
    attention_state = _RecursiveReshape(attention_state,
                                        [batch_size * num_heads, -1])
    ctx_vec, prob, att_state = self.atten.ComputeContextVectorWithSource(
        theta.atten, concated_source_vecs, concated_source_contexts,
        source_padding, source_segment_id, query_vec_projected, attention_state,
        per_step_source_padding, step_state, query_segment_id)
    ctx_vec = tf.reshape(ctx_vec, [batch_size, -1])
    if p.enable_ctx_post_proj:
      ctx_vec = tf.matmul(ctx_vec, theta.ctx_post_proj)
      ctx_vec += theta.ctx_post_proj_b
    prob = tf.reduce_mean(tf.reshape(prob, [batch_size, num_heads, -1]), 1)
    att_state = _RecursiveReshape(att_state, [batch_size, -1])

    return ctx_vec, prob, att_state

  def ComputeContextVectorWithAttenProbs(self, theta, packed_context,
                                         atten_probs):
    """Computes the context vector given the attention probailities.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      packed_context: Concated source contexts with shape [
        batch_size * num_heads, time, context_dim // num_heads].
      atten_probs: The attention probability vector:
        [batch_size * num_heads, time].

    Returns:
      The attention context vector: [target_batch, source_dim]
      If p.enable_ctx_post_proj is false, source_dim = context_dim,
      otherwise, source_dim = p.ctx_post_proj_dim.
    """
    p = self.params
    num_heads = p.num_attention_heads
    # packed_context: [batch_size * num_head, num_style,
    # hidden_dim / num_head]
    # inp: [batch_size * num_head, num_style]
    packed_context = py_utils.with_dependencies([
        py_utils.assert_shape_match([tf.shape(packed_context)[0]],
                                    [tf.shape(atten_probs)[0]])
    ], packed_context)
    b_size = tf.shape(packed_context)[0] // num_heads
    ctx_vec = tf.reshape(
        tf.matmul(tf.expand_dims(atten_probs, 1), packed_context), [b_size, -1])
    if p.enable_ctx_post_proj:
      ctx_vec_proj = tf.matmul(ctx_vec, theta.ctx_post_proj)
      ctx_vec_proj += theta.ctx_post_proj_b
    else:
      ctx_vec_proj = ctx_vec
    return ctx_vec_proj, ctx_vec

  def ComputeContextVectorWithCachedSource(self,
                                           theta,
                                           concated_source_vecs,
                                           concated_source_contexts,
                                           source_padding,
                                           source_segment_id,
                                           query_vec,
                                           attention_state=None,
                                           per_step_source_padding=None,
                                           step_state=None,
                                           query_segment_id=None):
    """Same as the ComputeContextVectorWithSource api above, except values ...

    in source_vecs, source_contexts and source_padding are ordered differently.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      concated_source_vecs: Concated source vectors with shape [source_batch,
        time, hidden_dim].
      concated_source_contexts: Concated source contexts with shape [
        source_batch, time, context_dim].
      source_padding: Source padding with shape [source_batch, time, num_heads].
        If None, assume no padding.
      source_segment_id: Source segment id with shape
        [source_batch, time, num_heads].
      query_vec: a tensor of shape [target_batch, query_dim].
      attention_state: previous attention state. It is not used in
          AdditiveAttention, and is simply passed through.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch_size,
        source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: a tensor of shape [target_batch].

    Returns:
      The attention context vector:     [target_batch, source_dim]
      The attention probability vector: [target_batch, time]
      The new attention mechanism state: possibly nested tuple of tensors with
        dimensions [target_batch....]
    """
    p = self.params
    batch_size = tf.shape(concated_source_vecs)[0]
    src_seq_len = tf.shape(concated_source_vecs)[1]
    num_heads = p.num_attention_heads
    concated_source_vecs = tf.reshape(
        tf.transpose(concated_source_vecs, [1, 0, 2]),
        [src_seq_len, batch_size * num_heads, -1])
    # TODO(yonghui): Rewrite the following with just one transpose.
    concated_source_contexts = tf.transpose(
        tf.reshape(
            tf.transpose(concated_source_contexts, [1, 0, 2]),
            [src_seq_len, batch_size * num_heads, -1]), [1, 0, 2])
    if source_padding is not None:
      source_padding = tf.reshape(
          tf.transpose(source_padding, [1, 0, 2]),
          [src_seq_len, batch_size * num_heads])
    else:
      source_padding = tf.zeros([src_seq_len, batch_size * num_heads])
    if source_segment_id is None:
      source_segment_id = tf.zeros(
          [src_seq_len, batch_size * num_heads], dtype=source_padding.dtype)
    else:
      source_segment_id = tf.reshape(
          tf.transpose(source_segment_id, [1, 0, 2]),
          [src_seq_len, batch_size * num_heads])
    return self.ComputeContextVectorWithSource(
        theta, concated_source_vecs, concated_source_contexts, source_padding,
        source_segment_id, query_vec, attention_state, per_step_source_padding,
        step_state, query_segment_id)


class LocationSensitiveAttention(BaseAttentionLayer):
  """An attention that also takes into account previously attended locations.

  See section 2.2 of this paper for a description of this technique:
  http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf
  """

  @classmethod
  def Params(cls):
    """Params for this LocationSensitiveAttention class."""
    p = super(LocationSensitiveAttention, cls).Params()
    p.Define('source_dim', 0, 'Number of source nodes.')
    p.Define('location_filter_size', 0,
             'Location filter size, should be an odd number e.g. 31.')
    p.Define('location_num_filters', 0, 'Number of location filters, e.g. 32.')
    p.Define('query_dim', 0, 'Number of query nodes.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    p.Define(
        'same_batch_size', False,
        'True iff the source and target sequence has the same batch size.')
    p.Define(
        'location_features', ['PREV_PROBS'],
        'List signals to run the convolutions on. Possible options are: '
        'PREV_PROBS, CUMULATIVE_PROBS.')

    # Fill in reasonable default for params init
    p.params_init.method = 'gaussian_sqrt_dim'
    p.params_init.scale = 1.0
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs an LocationSensitiveAttention object."""
    super(LocationSensitiveAttention, self).__init__(params)
    p = self.params
    name = p.name
    assert p.packed_input is False, ('Packed input is not supported yet for '
                                     'LocationsensitiveAttention.')

    if p.atten_dropout_prob != 0:
      raise NotImplementedError('dropout is not supported')

    with tf.variable_scope(name):
      pc = py_utils.WeightParams(
          shape=None,
          init=p.params_init,
          dtype=p.dtype,
          collections=['LocationSensitiveAttention_vars'])
      source_var_shape = [p.source_dim, p.hidden_dim]
      pc.shape = source_var_shape
      self.CreateVariable('source_var', pc, self.AddGlobalVN)
      query_var_shape = [p.query_dim, p.hidden_dim]
      pc.shape = query_var_shape
      self.CreateVariable('query_var', pc, self.AddGlobalVN)
      hidden_var_shape = [p.hidden_dim]
      pc.shape = hidden_var_shape
      self.CreateVariable('hidden_var', pc, self.AddGlobalVN)

      assert p.location_filter_size % 2 == 1
      assert p.location_num_filters > 0

      location_filter_shape = [
          p.location_filter_size,
          len(p.location_features), p.location_num_filters
      ]
      # TODO(yonghui): Don't hard code how params are initialized.
      location_filter_pc = py_utils.WeightParams(
          shape=location_filter_shape,
          init=py_utils.WeightInit.Uniform(0.05),
          dtype=p.dtype,
          collections=['LocationSensitiveAttention_vars'])
      self.CreateVariable('location_filter_var', location_filter_pc,
                          self.AddGlobalVN)
      location_var_shape = [p.location_num_filters, p.hidden_dim]
      location_pc = py_utils.WeightParams(
          shape=location_var_shape,
          init=py_utils.WeightInit.Uniform(0.05),
          dtype=p.dtype,
          collections=['LocationSensitiveAttention_vars'])
      self.CreateVariable('location_var', location_pc, self.AddGlobalVN)

    @function.Defun(*[p.dtype] * 5, noinline=not py_utils.use_tpu())
    def AttenLogits(concated_source_vecs, query_vec_reshaped, hidden_v,
                    location_feats, location_var):
      """Generates logits."""

      def CollapseOutDim(x):
        return tf.reshape(x, [-1, tf.shape(x)[-1]])

      location_hidden = py_utils.Matmul(
          CollapseOutDim(location_feats), location_var)
      sl = tf.shape(location_feats)[1]
      tb = tf.shape(location_feats)[0]
      hd = tf.shape(location_var)[1]
      location_hidden = tf.reshape(location_hidden, [tb, sl, hd])
      location_hidden = tf.transpose(location_hidden, [1, 0, 2])
      sb = tf.shape(query_vec_reshaped)[2]
      bs_mult = tf.shape(query_vec_reshaped)[1]
      location_hidden = tf.reshape(location_hidden, [sl, bs_mult, sb, hd])

      # Shape of summed is [sl, tb/sb, sb, hidden_dim].
      summed = tf.tanh(concated_source_vecs + query_vec_reshaped +
                       location_hidden)
      # logits is of shape [sl * tb/sb * sb, 1]. Computes dot product
      # between v with every rows in 'summed'. Then we reshape the
      # result to be of shape [sl, tb/sb, sb].
      logits = py_utils.Matmul(
          tf.reshape(summed, [-1, p.hidden_dim]),
          tf.reshape(hidden_v, [p.hidden_dim, 1]))
      logits = tf.reshape(logits, tf.shape(summed)[:3])
      return logits

    @function.Defun(*[p.dtype] * 5, noinline=not py_utils.use_tpu())
    def AttenLogitsSameBatchSize(concated_source_vecs, query_vec_transformed,
                                 hidden_v, location_feats, location_var):
      """Generates logits.

      Optimized code path for when the target and the source have the same batch
      size.

      Args:
        concated_source_vecs: Tensor of shape [sl, batch, dim]
        query_vec_transformed: Tensor of shape [batch, dim]
        hidden_v: Tensor of shape [dim]
        location_feats: Tensor of shape [batch, sl, location_feature_dim]
        location_var: Tensor of shape [location_feature_dim, dim]

      Returns:
        logits in the shape [sl, batch_size].
      """

      def CollapseOutDim(x):
        return tf.reshape(x, [-1, tf.shape(x)[-1]])

      # => [sl, batch, hd]
      location_feats = tf.transpose(location_feats, [1, 0, 2])
      location_hidden = py_utils.Matmul(
          CollapseOutDim(location_feats), location_var)
      sl = tf.shape(location_feats)[0]
      tb = tf.shape(location_feats)[1]
      hd = tf.shape(location_var)[1]
      location_hidden = tf.reshape(location_hidden, [sl, tb, hd])

      # Shape of summed is [sl, sb, hidden_dim].
      summed = tf.tanh(concated_source_vecs +
                       tf.expand_dims(query_vec_transformed, 0) +
                       location_hidden)
      # logits is of shape [sl * sb, 1]. Computes dot product
      # between v with every rows in 'summed'. Then we reshape the
      # result to be of shape [sl, tb].
      logits = py_utils.Matmul(
          tf.reshape(summed, [-1, p.hidden_dim]),
          tf.reshape(hidden_v, [p.hidden_dim, 1]))
      logits = tf.reshape(logits, tf.shape(summed)[:2])
      # ==> of shape [sl, tb]
      return logits

    def Atten(hidden_var, query_var, source_padding, concated_source_vecs,
              concated_source_contexts, query_vec, attention_state,
              location_filter_var, location_var, per_step_source_padding):
      """Computes the attention context vector."""
      p = self.params
      # attention_state shape [batch, slen, len(p.location_features)]
      # it contains previous and accumulated attention probabilites.
      attention_state = py_utils.HasShape(
          attention_state, [-1, -1, len(p.location_features)])

      if p.dtype != tf.float32:
        location_feats = tf.nn.conv1d(
            tf.cast(attention_state, tf.float32),
            tf.cast(location_filter_var, tf.float32),
            1,
            'SAME',
            data_format='NHWC')
        location_feats = tf.cast(location_feats, p.dtype)
      else:
        location_feats = tf.nn.conv1d(
            attention_state, location_filter_var, 1, 'SAME', data_format='NHWC')
      # concated_source_vecs is of shape [sl, sb, dims]
      # concated_source_contexts is of shape [sb, sl, context_dim]
      # query_vec is of shape [tb, dims]
      sb = tf.shape(concated_source_vecs)[1]
      tb = tf.shape(query_vec)[0]
      multiplier = tb // sb
      # concated_source_vecs is reshaped to [sl, 1, sb, hidden_dims]
      concated_source_vecs = tf.expand_dims(concated_source_vecs, 1)
      query_vec_transformed = py_utils.Matmul(query_vec, query_var)
      # query_vec is reshaped to [1, tb/sb, sb, hidden_dims].
      query_vec_reshaped = tf.reshape(query_vec_transformed,
                                      [1, multiplier, sb, p.hidden_dim])
      # logits is of shape [sl, tb/sb, sb]
      logits = AttenLogits(concated_source_vecs, query_vec_reshaped, hidden_var,
                           location_feats, location_var)
      # Take out the padding states.
      # _source_padding is of shape [sl, sb].
      # reshaped to [sl, 1,  sb].
      source_padding = tf.expand_dims(source_padding, 1)
      per_step_source_padding = tf.reshape(
          tf.transpose(per_step_source_padding), [-1, multiplier, sb])
      source_padding += per_step_source_padding
      # Reshape logits to a matrix of shape [tb, sl] and takes the
      # softmax to compute the probabilities.
      logits = tf.transpose(tf.reshape(logits, [-1, tb]))
      source_padding = tf.transpose(tf.reshape(source_padding, [-1, tb]))
      probs = self._PaddedSoftmax(logits, source_padding)
      # Reshape probs to be of shape [tb/sb, sb, sl].
      probs_reshaped = tf.reshape(probs, [multiplier, sb, -1])
      # Transpose probs to be of shape [sb, tb/sb, sl]
      probs_reshaped = tf.transpose(probs_reshaped, [1, 0, 2])
      # [sb, tb/sb, sl] * [sb, sl, context_dim] = [sb, tb/sb, context_dim]
      summed = tf.matmul(probs_reshaped, concated_source_contexts)
      # summed is of shape [tb/sb, sb, context_dim]
      summed = tf.transpose(summed, [1, 0, 2])
      return tf.reshape(summed, [tb, -1]), probs

    def AttenSameBatchSize(hidden_var, query_var, source_padding,
                           concated_source_vecs, concated_source_contexts,
                           query_vec, attention_state, location_filter_var,
                           location_var, per_step_source_padding):
      """Computes the attention context vector.

      Optimized code path for when source and target have the same batch size.
      """
      del per_step_source_padding
      p = self.params
      # attention_state shape [batch, slen, len(p.location_features)]
      # it contains previous and accumulated attention probabilites.
      attention_state = py_utils.HasShape(
          attention_state, [-1, -1, len(p.location_features)])

      if p.dtype != tf.float32:
        location_feats = tf.nn.conv1d(
            tf.cast(attention_state, tf.float32),
            tf.cast(location_filter_var, tf.float32),
            1,
            'SAME',
            data_format='NHWC')
        location_feats = tf.cast(location_feats, p.dtype)
      else:
        location_feats = tf.nn.conv1d(
            attention_state, location_filter_var, 1, 'SAME', data_format='NHWC')
      query_vec_transformed = py_utils.Matmul(query_vec, query_var)
      # logits is of shape [sl, sb]
      logits = AttenLogitsSameBatchSize(concated_source_vecs,
                                        query_vec_transformed, hidden_var,
                                        location_feats, location_var)
      # => [sl, tb]
      logits.set_shape(source_padding.shape)
      # Reshape logits to a matrix of shape [tb, sl] and takes the
      # softmax to compute the probabilities.
      logits = tf.transpose(logits)
      source_padding = tf.transpose(source_padding)
      probs = self._PaddedSoftmax(logits, source_padding)
      summed = tf.matmul(tf.expand_dims(probs, 1), concated_source_contexts)
      return tf.squeeze(summed, 1), probs

    if p.same_batch_size:
      self._ctx_vec = AttenSameBatchSize
    else:
      self._ctx_vec = Atten

    def EncodeSource(src_w, vecs, ctxs):
      time, batch = py_utils.GetShape(vecs, 2)
      ctxs = py_utils.HasShape(ctxs, [time, batch, -1])
      transformed_vecs = tf.reshape(
          py_utils.Matmul(tf.reshape(vecs, [-1, p.source_dim]), src_w),
          [time, batch, -1])
      transposed_ctxs = tf.transpose(ctxs, [1, 0, 2])
      return transformed_vecs, transposed_ctxs

    self._encode_source = EncodeSource

  def InitForSourcePacked(self,
                          theta,
                          source_vecs,
                          source_contexts,
                          source_padding,
                          source_segment_id=None):
    """Initialize attention for the given source vectors.

    Unlike the InitForSource API above, this API takes in a single tensor
    representing the entire sequence.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A single tensor of shape [time, batch_size, source_dim].
      source_contexts: A single tensor of shape [time, batch_size, some_dim].
      source_padding: A tensor of shape [time, batch_size].
      source_segment_id: A tensor of shape [time, batch_size].

    Returns:
      Concated source vectors, concated source contexts, and source paddings.
    """
    with tf.name_scope(self.params.name):
      self._source_init_done = True
      self._source_padding = source_padding
      (self._concated_source_vecs, self._concated_source_contexts) = (
          self._encode_source(theta.source_var, source_vecs, source_contexts))
      if source_segment_id is None:
        source_segment_id = tf.zeros_like(source_padding)
      self._source_segment_id = source_segment_id
    return (self._concated_source_vecs, self._concated_source_contexts,
            self._source_padding, self._source_segment_id)

  def ZeroAttentionState(self, source_seq_length, decoder_batch_size):
    p = self.params
    dtype = p.dtype
    num_features = len(p.location_features)
    with tf.name_scope(p.name):
      state = tf.concat([
          tf.ones([decoder_batch_size, 1, num_features], dtype=dtype),
          tf.zeros(
              [decoder_batch_size, source_seq_length - 1, num_features],
              dtype=dtype)
      ], 1)
      # Having the last dim being 1 or 2 is very inefficient on tpu, and hence
      # we reshape to combine the last two dims.
      state = tf.reshape(state, [decoder_batch_size, -1])
      return state

  def ComputeContextVectorWithSource(self,
                                     theta,
                                     concated_source_vecs,
                                     concated_source_contexts,
                                     source_padding,
                                     source_segment_id,
                                     query_vec,
                                     attention_state=None,
                                     per_step_source_padding=None,
                                     step_state=None,
                                     query_segment_id=None):
    """Computes the context vector given the current query output.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      concated_source_vecs: Concated source vectors with shape [time,
        batch_size, hidden_dim].
      concated_source_contexts: Concated source contexts with shape [
        batch_size, time, context_dim].
      source_padding: Source padding with shape [time, batch_size].
      source_segment_id: Source segment id with shape [time, batch_size].
      query_vec: a tensor of shape [batch_size, query_dim].
      attention_state: If params().location_features == ['PREV_PROBS',
                                                         'CUMULATIVE_PROBS'],
        then attention_state is a tensor of shape [batch_size, src_len * 2]:
          attention_state[:, :, 0] contains previous attention probabilities
          attention_state[:, :, 1] contains a sum over previous timesteps of
          attention probabilities.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch_size,
        source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: Query segment id with shape [batch_size].

    Returns:
      The attention context vector.
      The attention probability vector.
      The new attention mechanism state: possibly nested tuple of tensors with
        dimensions [target_batch....]
    """
    del source_segment_id
    del query_segment_id
    p = self.params
    if p.same_batch_size:
      assert per_step_source_padding is None
    query_batch_size = tf.shape(query_vec)[0]
    source_seq_length = tf.shape(source_padding)[0]
    if per_step_source_padding is None:
      zero = tf.constant(0.0, dtype=query_vec.dtype)
      per_step_source_padding = tf.fill([query_batch_size, source_seq_length],
                                        zero)
    per_step_source_padding = py_utils.HasShape(
        per_step_source_padding, [query_batch_size, source_seq_length])

    hidden = py_utils.AddPerStepVN(p, theta.hidden_var)
    query = py_utils.AddPerStepVN(p, theta.query_var)
    location_filter = py_utils.AddPerStepVN(p, theta.location_filter_var)
    location = py_utils.AddPerStepVN(p, theta.location_var)

    bs = tf.shape(attention_state)[0]
    num_location_features = len(p.location_features)
    attention_state = tf.reshape(attention_state,
                                 [bs, -1, num_location_features])

    ctx_vec, prob = self._ctx_vec(
        hidden, query, source_padding, concated_source_vecs,
        concated_source_contexts, query_vec, attention_state, location_filter,
        location, per_step_source_padding)

    new_feats = {'PREV_PROBS': prob}
    if 'CUMULATIVE_PROBS' in p.location_features:
      new_feats['CUMULATIVE_PROBS'] = (
          prob + attention_state[:, :,
                                 p.location_features.index('CUMULATIVE_PROBS')])
    new_attention_state = tf.stack(
        [new_feats[f] for f in p.location_features], axis=2)
    new_attention_state = tf.reshape(new_attention_state, [bs, -1])
    return ctx_vec, prob, new_attention_state


def MergeSourcePaddingWithPerStepSourcePadding(source_padding,
                                               per_step_source_padding, tb):
  """Merges source padding with per-step source padding.

  Args:
    source_padding: [sl, sb].
    per_step_source_padding: [tb, sl].
    tb: target batch size.

  Returns:
    A tensor of shape [tb, sl].
  """
  # source_padding is of shape [sl, sb].
  sl = tf.shape(source_padding)[0]
  sb = tf.shape(source_padding)[1]

  if per_step_source_padding is None:
    zero = tf.constant(0.0, dtype=source_padding.dtype)
    per_step_source_padding = tf.fill([tb, sl], zero)
  per_step_source_padding = py_utils.HasShape(per_step_source_padding, [tb, sl])

  # Transpose and reshape source_padding to [1, sb,  sl].
  source_padding = tf.expand_dims(tf.transpose(source_padding), 0)
  # Merge source_padding and per_step_source_padding.
  source_padding = tf.maximum(source_padding,
                              tf.reshape(per_step_source_padding, [-1, sb, sl]))
  return tf.reshape(source_padding, [tb, -1])


class MonotonicAttention(BaseAttentionLayer):
  """An attention mechanism which enforces monotonic alignments.

  This layer implements the monotonic attention mechanism described in ``Online
  and Linear-Time Attention by Enforcing Mononotonic Alignments''
  (https://arxiv.org/abs/1704.00784).  It is used in exactly the same way as
  AdditiveAttention, but both the attention distribution and the energy function
  are different.
  Rather than using a softmax, this mechanism feeds the attention energy into a
  (hard or soft) sigmoid and treats the output as Bernoulli probabilities
  representing the probability of attending to a given entry in the input
  sequence, processed from left-to-right.  Based on this interpretation, the
  resulting distribution over input sequence entries is computed with a dynamic
  program.  The intended use is to train with soft sigmoids according to the
  expected output (setting param hard_sigmoid=False), then use hard sigmoids at
  test time to allow for online and linear-time decoding.  To encourge the train
  and test-time behavior to be similar, noise can optionally be added to the
  sigmoid activations during training (param pre_sigmoid_noise).  For the energy
  function, rather than computing
  energy = dot(v, tanh(dot(W, query) + dot(W, encoder_states)))
  it computes
  energy = dot(g*v/||v||, tanh(dot(W, query) + dot(W, encoder_states) + b)) + r
  where g and r are scalars and b is a vector, and ||v|| is the L2 norm of v.
  instead.  These modifications address the fact that the sigmoids in the
  monotonic attention mechanism are sensitive to offset and a bit harder to
  train compared to the softmax function.  It can be helpful to initialize the
  energy bias scalar r to a negative value (param hidden_bias_init).
  """

  @classmethod
  def Params(cls):
    """Params for this MonotonicAttention class."""
    p = super(MonotonicAttention, cls).Params()
    p.Define('source_dim', 0, 'Number of source nodes.')
    p.Define('query_dim', 0, 'Number of query nodes.')
    p.Define('hidden_dim', 0, 'Number of hidden nodes.')
    p.Define('pre_sigmoid_noise', 0, 'Standard deviation of pre-sigmoid noise.')
    p.Define('hidden_bias_init', -1, 'Initial value of hidden bias.')
    p.Define('hard_sigmoid', False, 'Whether to use a hard sigmoid.')
    # Fill in reasonable default for params init
    p.params_init.method = 'gaussian_sqrt_dim'
    p.params_init.scale = 1.0
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs an MonotonicAttention object."""
    super(MonotonicAttention, self).__init__(params)
    p = self.params
    assert p.packed_input is False, ('Packed input not supported for '
                                     'Monotonic Attention.')
    if p.atten_dropout_prob != 0:
      raise NotImplementedError('dropout is not supported')

    # When running eval, don't add pre-sigmoid noise, and use a hard sigmoid to
    # match behavior of online decoding.
    if p.is_eval:
      p.pre_sigmoid_noise = 0.
      p.hard_sigmoid = True

    with tf.variable_scope(p.name):
      pc = py_utils.WeightParams(
          shape=None,
          init=p.params_init,
          dtype=p.dtype,
          collections=['MonotonicAttention_vars'])
      # source is the weight matrix for the memory/encoder states
      source_var_shape = [p.source_dim, p.hidden_dim]
      pc.shape = source_var_shape
      self.CreateVariable('source_var', pc, self.AddGlobalVN)
      # query is the weight matrix for the query/decoder RNN state
      query_var_shape = [p.query_dim, p.hidden_dim]
      pc.shape = query_var_shape
      self.CreateVariable('query_var', pc, self.AddGlobalVN)
      # hidden is the pre-softmax vector which converts from tanh to scalar
      hidden_var_shape = [p.hidden_dim]
      pc.shape = hidden_var_shape
      self.CreateVariable('hidden_var', pc, self.AddGlobalVN)

      # energy_bias is the bias vector which appears inside of tanh
      energy_bias_shape = [p.hidden_dim]
      pc.shape = energy_bias_shape
      # Initialize the bias vector to all zeros
      pc.init.method = 'constant'
      pc.init.scale = 0.0
      self.CreateVariable('energy_bias_var', pc)
      # hidden_scale is the weight normalization scale for hidden
      hidden_scale_var_shape = []
      pc.shape = hidden_scale_var_shape
      # Initialize so that the initial scale is 1/sqrt(hidden_dim)
      pc.init.scale = 1 / np.sqrt(p.hidden_dim)
      self.CreateVariable('hidden_scale_var', pc)
      # hidden_bias is the bias scalar applied before the sigmoid
      hidden_bias_var_shape = []
      pc.shape = hidden_bias_var_shape
      # Use the hidden_bias_init hyperparam to set the initial value
      pc.init.scale = p.hidden_bias_init
      self.CreateVariable('hidden_bias_var', pc)

      # Create seeds for stateless random number generator.
      random_seed_dtype = tf.int32
      _, self._step_counter = py_utils.CreateVariable(
          name='atten_step_counter',
          params=py_utils.WeightParams([], py_utils.WeightInit.Constant(0),
                                       random_seed_dtype),
          trainable=False)
      vname = self._step_counter.name
      self._prng_seed = tf.constant(
          py_utils.GenerateSeedFromName(vname), dtype=random_seed_dtype)
      if p.random_seed:
        self._prng_seed += p.random_seed

    def EncodeSource(src_w, vecs, ctxs):
      time, batch = py_utils.GetShape(vecs, 2)
      ctxs = py_utils.HasShape(ctxs, [time, batch, -1])
      transformed_vecs = tf.reshape(
          py_utils.Matmul(tf.reshape(vecs, [-1, p.source_dim]), src_w),
          [time, batch, -1])
      transposed_ctxs = tf.transpose(ctxs, [1, 0, 2])
      return transformed_vecs, transposed_ctxs

    self._encode_source = EncodeSource

  def PackSource(self,
                 theta,
                 source_vecs,
                 source_contexts,
                 source_padding,
                 source_segment_id=None):
    """Packs source vectors. Does not change attention state.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A single tensor of shape [time, batch_size, source_dim].
      source_contexts: A single tensor of shape [time, batch_size, some_dim].
      source_padding: A tensor of shape [time, batch_size].
      source_segment_id: A tensor of shape [time, batch_size].

    Returns:
      Concated source vectors, concated source contexts, and source paddings.
    """
    with tf.name_scope(self.params.name):
      (concated_source_vecs, concated_source_contexts) = (
          self._encode_source(theta.source_var, source_vecs, source_contexts))
      if source_segment_id is None:
        source_segment_id = tf.zeros_like(source_padding)

    return (concated_source_vecs, concated_source_contexts, source_padding,
            source_segment_id)

  def InitForSourcePacked(self,
                          theta,
                          source_vecs,
                          source_contexts,
                          source_padding,
                          source_segment_id=None):
    """Initialize attention for the given source vectors.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A single tensor of shape [time, batch_size, source_dim].
      source_contexts: A single tensor of shape [time, batch_size, some_dim].
      source_padding: A tensor of shape [time, batch_size].
      source_segment_id: A tensor of shape [time, batch_size].

    Returns:
      Concated source vectors, concated source contexts, and source paddings.
    """
    self._source_init_done = True
    (self._concated_source_vecs, self._concated_source_contexts,
     self._source_padding, self._source_segment_id) = self.PackSource(
         theta, source_vecs, source_contexts, source_padding, source_segment_id)

    return (self._concated_source_vecs, self._concated_source_contexts,
            self._source_padding, self._source_segment_id)

  def ZeroAttentionState(self, source_seq_length, decoder_batch_size):
    p = self.params
    dtype = p.dtype
    with tf.name_scope(p.name):
      # Set initial previous attention to [1, 0, ... 0] to avoid special-casing
      emit_probs = tf.one_hot(
          tf.zeros((decoder_batch_size,), dtype=tf.int32),
          source_seq_length,
          dtype=dtype)
      return py_utils.NestedMap(
          emit_probs=emit_probs,
          # stateless.stateless_random_normal() requires seeds of shape [2].
          random_seed=tf.stack([self._prng_seed, self._step_counter]))

  def ComputeProbabilities(self, theta, concated_source_vecs,
                           merged_source_padding, query_vec, attention_state):
    """Computes probabilities of emissions."""

    # concated_source_contexts is of shape [sb, sl, context_dim]
    # query_vec is of shape [tb, dims]
    sb = tf.shape(concated_source_vecs)[1]
    tb = tf.shape(query_vec)[0]
    multiplier = tb // sb

    p = self.params
    # noinline and compiled cannot be set at the same time
    @function.Defun(*([p.dtype] * 7), noinline=not py_utils.use_tpu())
    def AttenLogits(concated_source_vecs, query_vec, query_v, energy_b,
                    hidden_v, hidden_g, hidden_b):
      """Computes logits from source, query, and variables.

      Args:
        concated_source_vecs: [sl, sb, hidden_dims].
        query_vec: [tb, query_dim].
        query_v: [query_dim, hidden_dim]
        energy_b: [hidden_dim].
        hidden_v: [hidden_dim].
        hidden_g: [].
        hidden_b: [].

      Returns:
        logits: [tb, sl].
      """
      # Apply query matrix to query. Becomes [tb, hidden_dim].
      query_vec_transformed = py_utils.Matmul(
          query_vec, query_v, name='query_transformation')
      # query_vec is reshaped to [1, tb/sb, sb, hidden_dim].
      query_vec_reshaped = tf.reshape(query_vec_transformed,
                                      [1, multiplier, sb, p.hidden_dim])

      # [sl, 1, sb, hidden_dim].
      concated_source_vecs = tf.expand_dims(concated_source_vecs, 1)
      energy_b = tf.reshape(energy_b, [1, 1, 1, -1])
      # Shape of summed is [sl, tb/sb, sb, hidden_dim].
      summed = tf.tanh(concated_source_vecs + query_vec_reshaped + energy_b)
      hidden_v = hidden_g * tf.nn.l2_normalize(hidden_v, axis=0)
      # logits is of shape [sl * tb/sb * sb, 1]. Computes dot product
      # between v with every rows in 'summed'. Then we reshape the
      # result to be of shape [sl, tb/sb, sb].
      #
      # Another equivalent way is to do:
      #  logits = tf.reduce_sum(summed *
      #                         tf.reshape(v, [1, 1, 1, hidden_dim]), 3)
      logits = py_utils.Matmul(
          tf.reshape(summed, [-1, p.hidden_dim]),
          tf.reshape(hidden_v, [p.hidden_dim, 1]))
      logits += hidden_b
      # [tb, sl].
      logits = tf.transpose(tf.reshape(logits, [-1, tb]), [1, 0])
      return logits

    with tf.name_scope('logits'):
      logits = AttenLogits(concated_source_vecs, query_vec, theta.query_var,
                           theta.energy_bias_var, theta.hidden_var,
                           theta.hidden_scale_var, theta.hidden_bias_var)

    previous_attention = attention_state.emit_probs
    with tf.name_scope('prob'):
      if self.params.hard_sigmoid:
        # If using a hard sigmoid, just compare against 0
        p_choose_i = tf.cast(tf.greater(logits, 0), logits.dtype)
        # Never choose padded values.
        p_choose_i = tf.where(merged_source_padding > 0.0,
                              tf.zeros_like(p_choose_i), p_choose_i)
        # Compute probability distribution assuming hard probabilities
        probs = tf.contrib.seq2seq.monotonic_attention(
            p_choose_i, previous_attention, 'hard')
      else:
        # Compute pre-sigmoid noise.
        activation_noise = tf.contrib.stateless.stateless_random_normal(
            py_utils.GetShape(logits),
            attention_state.random_seed,
            dtype=logits.dtype)
        # Compute sigmoid probabilities.
        p_choose_i = tf.nn.sigmoid(
            logits + self.params.pre_sigmoid_noise * activation_noise)
        # Never choose padded values.
        p_choose_i = tf.where(merged_source_padding > 0,
                              tf.zeros_like(p_choose_i), p_choose_i)
        # Compute attention distribution
        probs = tf.contrib.seq2seq.monotonic_attention(
            p_choose_i, previous_attention, 'parallel')

    # [tb, sl].
    return probs, py_utils.NestedMap(
        emit_probs=probs, random_seed=attention_state.random_seed)

  def ComputeContextVectorWithSource(self,
                                     theta,
                                     concated_source_vecs,
                                     concated_source_contexts,
                                     source_padding,
                                     source_segment_id,
                                     query_vec,
                                     attention_state,
                                     per_step_source_padding=None,
                                     step_state=None,
                                     query_segment_id=None):
    """Computes the context vector given the current query output.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      concated_source_vecs: Concated source vectors with shape [time,
        batch_size, hidden_dim].
      concated_source_contexts: Concated source contexts with shape [
        batch_size, time, context_dim].
      source_padding: Source padding with shape [time, batch_size].
      source_segment_id: Source segment id with shape [time, batch_size].
      query_vec: a tensor of shape [batch_size, query_dim].
      attention_state: The attention probs computed at the previous timestep.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch_size,
        source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: a tensor of shape [batch_size].

    Returns:
      The attention context vector.
      The attention probability vector.
      The attention probability vector (again, to be interpreted as state).
    """
    del source_segment_id
    del query_segment_id
    sb = tf.shape(concated_source_vecs)[1]
    tb = tf.shape(query_vec)[0]
    multiplier = tb // sb
    merged_source_padding = MergeSourcePaddingWithPerStepSourcePadding(
        source_padding, per_step_source_padding, tb)

    probs, new_state = self.ComputeProbabilities(theta, concated_source_vecs,
                                                 merged_source_padding,
                                                 query_vec, attention_state)

    with tf.name_scope('sum'):
      # Reshape probs to be of shape [tb/sb, sb, sl]
      probs_reshaped = tf.reshape(probs, [multiplier, sb, -1])
      # Transpose probs to be of shape [sb, tb/sb, sl]
      probs_reshaped = tf.transpose(probs_reshaped, [1, 0, 2])
      # Batched matmul
      # [sb, tb/sb, sl] * [sb, sl, context_dim] = [sb, tb/sb, context_dim]
      summed = tf.matmul(probs_reshaped, concated_source_contexts)
      # summed is of shape [tb/sb, sb, context_dim]
      summed = tf.transpose(summed, [1, 0, 2])
      ctx_vec = tf.reshape(summed, [tb, -1])

    return ctx_vec, probs, new_state

  def PostTrainingStepUpdate(self, global_step):
    """Update self._step_counter with the global_step value."""
    return self._step_counter.assign(
        tf.cast(global_step, self._step_counter.dtype))


class GmmMonotonicAttention(BaseAttentionLayer):
  """A GMM-based monotonic attention module.

  Based on "Generating Sequences With Recurrent Neural Networks" by Alex Graves.
  Eq [46-51] in https://arxiv.org/abs/1308.0850.
  """

  @classmethod
  def Params(cls):
    """Params for this MonotonicAttention class."""
    p = super(GmmMonotonicAttention, cls).Params()
    p.Define('source_dim', 0, 'Number of source nodes.')
    p.Define('query_dim', 0, 'Number of query nodes.')
    p.Define('gmm_mlp_hidden_dim', 128,
             'Number of hidden units for the MLP that predicts GMM params.')
    p.Define('max_offset', -1,
             'Max offset to move attention pointer, Enabled only when > 0.')
    p.Define('num_mixtures', 5, 'Number of location GMM components.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a GMM-based monotonic attention module."""
    super(GmmMonotonicAttention, self).__init__(params)
    p = self.params
    if p.atten_dropout_prob != 0:
      raise NotImplementedError('dropout is not supported.')

    # TODO(ngyuzh): find a good initialize for both TTS and ASR.
    # Consider split the layer if it's very sensitive to the initialization
    # Compare Sigmoid and other activation functions.
    with tf.variable_scope(p.name):
      gmm_params = layers.FeedForwardNet.Params().Set(
          name=p.name,
          input_dim=p.query_dim,
          hidden_layer_dims=[p.gmm_mlp_hidden_dim, p.num_mixtures * 3],
          activation=['SIGMOID', 'NONE'],
          init=py_utils.WeightInit.Xavier(0.1))
      self.CreateChild('GMM', gmm_params)

      # TODO(ngyuzh): change variance to scale to make it simpler.
      # noinline and compiled cannot be set at the same time
      @function.Defun(*[p.dtype] * 4, noinline=not py_utils.use_tpu())
      def EvalGmmPdfs(encoder_positions, priors, means, variances):
        """Evaluate the location GMMs on all encoder positions."""
        # encoder_positions: [batch, 1, timesteps, 1]
        # [batch, tb / sb, 1, num_mixtures]
        priors = tf.expand_dims(priors, 2)
        means = tf.expand_dims(means, 2)
        variances = tf.expand_dims(variances, 2)
        # [batch, tb / sb, timesteps, num_mixtures]
        pdfs = ((priors * tf.rsqrt(2 * np.pi * variances + 1e-8)) * tf.exp(
            -(encoder_positions - means)**2 / (2 * variances + 1e-8)))
        # pdfs sized [batch, tb / sb, timesteps].
        return tf.reduce_sum(pdfs, 3)

      # TODO(ngyuzh): remove unnecessary transpose.
      def Atten(source_padding, concated_source_vecs, concated_source_contexts,
                query_vec, priors, means, variances, encoder_positions,
                per_step_source_padding):
        """Computes the attention context vector."""
        # tb: target batch size
        # sb: source batch size
        # concated_source_vecs is of shape [sl, sb, context_dim]
        # query_vec is of shape [tb, dims]
        p = self.params
        sb = tf.shape(concated_source_vecs)[1]
        tb = tf.shape(query_vec)[0]
        multiplier = tb // sb
        # [sb, tb / sb, num_mixtures]
        priors = tf.reshape(priors, [-1, multiplier, p.num_mixtures])
        means = tf.reshape(means, [-1, multiplier, p.num_mixtures])
        variances = tf.reshape(variances, [-1, multiplier, p.num_mixtures])

        probs = EvalGmmPdfs(encoder_positions, priors, means, variances)
        # [sl, tb / sb, sb]
        probs = tf.reshape(tf.transpose(probs, [2, 0, 1]), [-1, multiplier, sb])

        source_padding = tf.expand_dims(source_padding, 1)
        per_step_source_padding = tf.reshape(
            tf.transpose(per_step_source_padding), [-1, multiplier, sb])
        source_padding += per_step_source_padding
        source_padding = tf.minimum(source_padding, 1.0)

        probs *= (1.0 - source_padding)
        probs = py_utils.AddDebugTensor(probs, name='atten_probs')
        probs = tf.transpose(tf.reshape(probs, [-1, tb]))
        # [tb/sb, sb, sl]
        probs_reshaped = tf.reshape(probs, [multiplier, sb, -1])
        # [sb, tb/sb, sl]
        probs_reshaped = tf.transpose(probs_reshaped, [1, 0, 2])
        # Batched matmul
        # [sb, tb/sb, sl] * [sb, sl, context_dim] = [sb, tb/sb, context_dim]
        context_vector = tf.matmul(probs_reshaped, concated_source_contexts)
        context_vector = tf.transpose(context_vector, [1, 0, 2])
        return tf.reshape(context_vector, [tb, -1]), probs

      self._ctx_vec = Atten

      def EncodeSource(vecs, ctxs):
        # TODO(ngyuzh): combine with content-base attention.
        time, batch = py_utils.GetShape(vecs, 2)
        ctxs = py_utils.HasShape(ctxs, [time, batch, -1])
        transposed_ctxs = tf.transpose(ctxs, [1, 0, 2])
        return vecs, transposed_ctxs

      self._encode_source = EncodeSource

  def InitForSourcePacked(self,
                          theta,
                          source_vecs,
                          source_contexts,
                          source_padding,
                          source_segment_id=None):
    """Initialize attention for the given source vectors.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      source_vecs: A single tensor of shape [time, batch_size, source_dim].
      source_contexts: A single tensor of shape [time, batch_size, some_dim].
      source_padding: A tensor of shape [time, batch_size].
      source_segment_id: A tensor of shape [time, batch_size].

    Returns:
      Concated source vectors, concated source contexts, source paddings
      and source_segment_id.
    """
    with tf.name_scope(self.params.name):
      self._source_init_done = True
      self._source_padding = source_padding
      (self._concated_source_vecs, self._concated_source_contexts) = (
          self._encode_source(source_vecs, source_contexts))
      if source_segment_id is None:
        source_segment_id = tf.zeros_like(source_padding, dtype=tf.int32)
      self._source_segment_id = source_segment_id
    return (self._concated_source_vecs, self._concated_source_contexts,
            self._source_padding, self._source_segment_id)

  def ZeroAttentionState(self, source_seq_length, decoder_batch_size):
    p = self.params
    position = tf.zeros([decoder_batch_size, p.num_mixtures], dtype=p.dtype)
    position_offsets = tf.zeros(
        [decoder_batch_size, p.num_mixtures], dtype=p.dtype)
    variances = tf.ones([decoder_batch_size, p.num_mixtures], dtype=p.dtype)
    priors = tf.zeros([decoder_batch_size, p.num_mixtures], dtype=p.dtype)
    atten_states = tf.stack(
        [position, position_offsets, variances, priors], axis=2)
    return atten_states

  def ComputeContextVectorWithSource(self,
                                     theta,
                                     concated_source_vecs,
                                     concated_source_contexts,
                                     source_padding,
                                     source_segment_id,
                                     query_vec,
                                     attention_state,
                                     per_step_source_padding=None,
                                     step_state=None,
                                     query_segment_id=None):
    """Computes the context vector given the current query output.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      concated_source_vecs: Concated source vectors with shape [time,
        batch_size, hidden_dim].
      concated_source_contexts: Concated source contexts with shape [
        batch_size, time, context_dim].
      source_padding: Source padding with shape [time, batch_size].
      source_segment_id: Tensor of source segment ids, with shape [time,
        batch_size].
      query_vec: a tensor of shape [batch_size, query_dim].
      attention_state: previous attention state,
        then attention_state is a tensor of shape [batch_size, num_mixtures, 4]:
          attention_state[:, :, 0] contains previous location
          attention_state[:, :, 1] contains previous offset.
          attention_state[:, :, 2] contains previous variance.
          attention_state[:, :, 3] contains previous prior.
      per_step_source_padding: Source sequence padding to apply at this step.
        If not None, it should be of shape [target_batch_size,
        source_seq_length].
      step_state: A NestedMap containing 'global_step' and 'time_step'.
        Required for deterministic dropout.
      query_segment_id: a tensor of shape [batch_size]

    Returns:
      The attention context vector.
      The attention probability vector.
      The new attention state vector.
    """
    del source_segment_id
    del query_segment_id
    p = self.params
    query_batch_size = tf.shape(query_vec)[0]
    source_seq_length = tf.shape(source_padding)[0]
    if per_step_source_padding is None:
      zero = tf.constant(0.0, dtype=query_vec.dtype)
      per_step_source_padding = tf.fill([query_batch_size, source_seq_length],
                                        zero)
    per_step_source_padding = py_utils.HasShape(
        per_step_source_padding, [query_batch_size, source_seq_length])
    out = self.GMM.FProp(theta.GMM, query_vec)
    priors_logits, position_offset_logits, log_variances = tf.split(
        out, 3, axis=1, name='GMM')
    log_variances = tf.minimum(log_variances, layers.LOG_SCALE_CLAMP_BOUND)
    variances = tf.exp(log_variances)
    priors = tf.nn.softmax(priors_logits)
    if p.max_offset > 0:
      position_offset = tf.nn.sigmoid(position_offset_logits)
      position_offset *= p.max_offset
    else:
      position_offset = tf.exp(position_offset_logits)
    new_position = attention_state[:, :, 0] + position_offset
    new_position = tf.minimum(new_position, tf.to_float(source_seq_length))
    variances = py_utils.AddDebugTensor(variances, name='variances')
    priors = py_utils.AddDebugTensor(priors, name='priors')
    # Tile and reshape encoder_positions to [batch, 1, timesteps, 1] so that
    # it can be evaluated by locations GMMs in a vectorized way.
    source_batch_size = tf.shape(source_padding)[1]
    encoder_positions = tf.expand_dims(
        tf.to_float(tf.range(source_seq_length)), 0)
    encoder_positions = tf.tile(encoder_positions, (source_batch_size, 1))
    # [batch, timesteps, 1].
    encoder_positions = tf.expand_dims(encoder_positions, 1)
    encoder_positions = tf.expand_dims(encoder_positions, 3)
    ctx_vec, prob = self._ctx_vec(source_padding, concated_source_vecs,
                                  concated_source_contexts, query_vec, priors,
                                  new_position, variances, encoder_positions,
                                  per_step_source_padding)
    new_atten_states = tf.stack(
        [new_position, position_offset, variances, priors], axis=2)
    return ctx_vec, prob, new_atten_states
