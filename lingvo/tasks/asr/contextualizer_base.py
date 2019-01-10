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
"""API for context injection into a speech decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo.core import base_layer


class ContextualizerBase(base_layer.BaseLayer):
  """Base class for a contextualizer.

  Typical usage of a contextualizer is::

    contextualizer.SetContextMap(context_map, theta)  # Set context map.
    contextualizer.InitAttention(...)             # Initalize attn sources.
    context = contextualizer.ZeroAttention(...)   # Initalize attn state.
    context = contextualizer.QueryAttention(...)  # Call on each decoding step.

  `context` is a per-decoding-step context vector that augments the standard
  LAS model with additional context.

  `context_map` can include data needed for initialization.

  After parameters of the contextualizer are set, these accessors can be used:

  - contextualizer.GetContextDim()
  """

  def SetContextMap(self, context_map, theta):
    """Set the context map.

    Args:
      context_map: A NestedMap object containing the context from which
        attention vectors will be computed.
      theta: NestedMap, parameters needed for embedding.
    """
    raise NotImplementedError('SetContextMap')

  def InitAttention(self, theta, packed_src, misc_states):
    """Initialized the contextualizer's attention.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      packed_src: A NestedMap object into which attention source vectors will be
        placed.
      misc_states: A NestedMap object into which attention states will be
        placed.
    """
    raise NotImplementedError('InitAttention')

  def ZeroAttention(self, theta, dec_bs, misc_states, audio_context,
                    packed_src):
    """Creates the contextualizer 'zero' context vector.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      dec_bs: int32, the decoding batch size.
      misc_states: A NestedMap object into which attention states will be
        placed.
      audio_context: Tensor of shape [dec_bs, aud_dim] representing the
        audio-based context vector.
      packed_src: A NestedMap object into which attention source vectors will be
        placed.

    Returns:
      A 'zero' context vector of shape [dec_bs, aud_dim + context_dim]
    """
    raise NotImplementedError('ZeroAttention')

  def QueryAttention(self, theta, attn_query, misc_states, audio_context,
                     packed_src):
    """Query the contextualizer's attention.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      attn_query: Tensor of shape [dec_bs, ...] representing the query vectors.
      misc_states: A NestedMap object into which attention states will be
        placed.
      audio_context: Tensor of shape [dec_bs, aud_dim] representing the
        audio-based context vector.
      packed_src: A NestedMap object into which attention source vectors will be
        placed.

    Returns:
      A context vector of shape [dec_bs, aud_dim + context_dim]
    """
    raise NotImplementedError('QueryAttention')

  def GetContextDim(self):
    """Returns the context dimension."""
    raise NotImplementedError('GetContextDim')


class NullContextualizer(ContextualizerBase):
  """An 'empty' or no-op contextualizer."""

  def SetContextMap(self, context_map, theta):
    pass

  def InitAttention(self, theta, packed_src):
    pass

  def ZeroAttention(self, theta, dec_bs, misc_states, audio_context,
                    packed_src):
    return audio_context

  def QueryAttention(self, theta, attn_query, misc_states, audio_context,
                     packed_src):
    return audio_context

  def GetContextDim(self):
    return 0
