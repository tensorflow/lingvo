# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import compat as tf
from lingvo.core import base_layer
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
    p = super(EmbeddingStep, cls).Params()
    p.name = 'emb_step'
    p.Define('emb', layers.EmbeddingLayer.Params(), 'Embedding layer params.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(EmbeddingStep, self).__init__(params)
    p = params
    with tf.variable_scope(p.name):
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
