# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Step classes for embedding tables."""

from lingvo import compat as tf
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import step


class EmbeddingStep(step.Step):
  """A simple wrapper around EmbeddingLayer and its subclasses.

  This class can be used to insert an embedding lookup at the input side
  of a GraphStep or StackStep.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'emb_step'
    p.Define('emb',
             layers.EmbeddingLayer.Params().Set(max_num_shards=1),
             'Embedding layer params.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = params
    self.CreateChild('emb', p.emb)

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Looks up a list of embeddings from an EmbeddingLayer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: unused.
      step_inputs: A NestedMap containing a list called inputs. This list should
        contain a single integer tensor of shape [batch], where each integer
        represents an index into the embedding table. (By convention, all Steps
        that can be used with StackStep must store inputs in
        step_inputs.inputs[], but in this step it does not make sense for that
        list to have more than one tensor in it).
      padding: unused.
      state0: unused.

    Returns:
      A params.dtype tensor of shape [batch, embedding_dim].
    """
    del prepared_inputs
    del state0
    assert len(step_inputs.inputs) == 1

    output = self.emb.EmbLookup(theta.emb, step_inputs.inputs[0])
    return py_utils.NestedMap(output=output), py_utils.NestedMap()


class StatefulEmbeddingStep(EmbeddingStep):
  """Simple wrapper for keeping a state of the tokens previously emitted."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('target_sos_id', 1, 'ID of the start of sentence token.')
    p.Define('num_prev_tokens', 0,
             'The number of previous tokens to keep in state.')
    p.Define('include_current_token', False,
             'Include current token in embedding lookup')
    p.name = 'stateful_emb_step'
    return p

  def __init__(self, params):
    p = params
    self.num_tokens = p.num_prev_tokens + int(p.include_current_token)
    assert self.num_tokens > 0, 'Number of tokens is zero!'

    # If embedding supports multiple tokens, then it must have a num_tokens
    # param, otherwise the num_tokens must be 1
    if hasattr(p.emb, 'num_tokens'):
      p.emb.num_tokens = self.num_tokens
    else:
      assert self.num_tokens == 1, ("Since p.emb doesn't have the num_tokens "
                                    'param, p.num_prev_tokens and '
                                    'p.include_current_token must sum to 1')
    super().__init__(params)

  def ZeroState(self, theta, prepared_inputs, batch_size):
    p = self.params
    state0 = super().ZeroState(theta, prepared_inputs, batch_size)
    state0.prev_ids = tf.ones([batch_size, p.num_prev_tokens],
                              dtype=tf.int32) * p.target_sos_id
    return state0

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Calls an embedding lookup and updates the state of token history.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: unused.
      step_inputs: A NestedMap containing a list called inputs. This list should
        contain a single integer tensor of shape [batch], where each integer
        represents an index into the embedding table. (By convention, all Steps
        that can be used with StackStep must store inputs in
        step_inputs.inputs[], but in this step it does not make sense for that
        list to have more than one tensor in it).
      padding: unused.
      state0: A NestedMap containing the state of previous tokens.
      - prev_ids: A Tensor containing the n previous token ids. [batch,
        num_prev_tokens]. Each row is the token ids at t-1, ..., t-n.

    Returns:
      Embedding vectors. [batch, p.emb.embedding_dim]

    """
    p = self.params

    # prepare token ids
    if p.include_current_token:
      ids = tf.concat([step_inputs.inputs[0][:, None], state0.prev_ids],
                      axis=-1)
    else:
      ids = state0.prev_ids

    # lookup embedding. ids.shape is [batch, num_tokens]
    embedding = self.emb.EmbLookup(theta.emb, ids)
    embedding = tf.reshape(embedding, [-1, p.emb.embedding_dim])

    # update state
    state1 = state0.copy()
    if p.num_prev_tokens > 0:
      state1.prev_ids = tf.concat([
          tf.cast(step_inputs.inputs[0][:, None], tf.int32),
          tf.cast(state0.prev_ids[:, :-1], tf.int32)
      ],
                                  axis=-1)
    state1.prev_ids = tf.ensure_shape(
        state1.prev_ids, [None, p.num_prev_tokens],
        name='prev_ids_shape_validation')

    return py_utils.NestedMap(output=embedding), state1
