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
"""Utilities for fusing language models with the decoder output."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.tasks.lm import layers as lm_layers


class FusionBase(base_layer.BaseLayer):
  """Base class for fusion with LMs."""

  @classmethod
  def Params(cls):
    p = super(FusionBase, cls).Params()
    p.Define('lm', lm_layers.NullLm.Params(), 'Language model params.')
    p.Define(
        'base_model_logits_dim', None,
        'Dimension of base (i.e., the model being fused with the LM) model\'s '
        'logits.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes Fusion class."""
    try:
      if isinstance(params.lm.emb, (layers.EmbeddingLayer)):
        params.lm.emb.on_ps = False
    except AttributeError:
      pass

    super(FusionBase, self).__init__(params)

    p = self.params
    self.CreateChild('lm', p.lm)

  def zero_state(self, batch_size):
    """Returns initial model state for fusion model."""
    state0 = py_utils.NestedMap()
    state0.lm_states = self.lm.zero_state(batch_size)
    return state0

  def _FPropLm(self, theta, state0, ids, paddings, misc=None):
    """LM FProp.

    Works for single step or entire seq.

    Args:
      theta: A NestedMap object containing weights for the layer and its
        children.
      state0: A NestedMap of states (specific to the layer).
      ids: Target ids, of shape [batch_size] for single step unrolling or
        [batch_size, base_model_logits_dim] for the entire sequence.
      paddings: Target paddings, of the same shape as 'ids'.
      misc: NestedMap of miscellaneous items, which might be needed during
        training.

    Returns:
      (lm_output, state1), with:

      - lm_output: A NestedMap containing lm output. If 'ids' is 1-D, then
        lm_output should have shape [batch_size, dim]; if it is 2-D then the
        shape should be [seq_len, batch_size, dim].
      - state1: A NestedMap of updated states.
    """
    state1 = state0.DeepCopy()
    is_single_step = (ids.shape.ndims == 1)
    if is_single_step:
      seq_len = 1
    else:
      seq_len = tf.shape(ids)[0]

    self._ModifyLmBeforeFProp(theta, state0, ids, paddings, misc)

    lm_output, state1.lm_states = self.lm.FProp(
        theta.lm, tf.reshape(ids, [seq_len, -1]),
        tf.reshape(paddings, [seq_len, -1]), state0.lm_states)

    if is_single_step:
      # lm outputs have dimension [time, batch, dim]. Since this is only one
      # step, remove time dimension.
      lm_output = lm_output.Transform(lambda v: tf.squeeze(v, axis=0))

    return lm_output, state1

  def FProp(self, theta, state0, am_output, ids, paddings, misc=None):
    """Real fusion logic happens here.

    Works for single step or for the entire sequence.

    Args:
      theta: A NestedMap object containing weights for the layer and its
        children.
      state0: A NestedMap of states (specific to the layer).
      am_output: The output from the speech model. 'am_output' can have shape
        [batch_size, base_model_logits_dim] for a single step unrolling or
        [seq_len, batch_size, base_model_logits_dim] for the entire sequence.
      ids: Target ids, of shape [batch_size] for single step unrolling or
        [batch_size, base_model_logits_dim] for the entire sequence.
      paddings: Target paddings, of the same shape as 'ids'.
      misc: NestedMap of miscellaneous items, which might be needed during
        training.

    Returns:
      (fused_output, state1), with:

      - fused_output: A tensor containing the fused result. If am_output is 2-D,
        then the fused_output should have shape [batch_size, dim]; if
        am_output is 3-D, then the shape should be [seq_len, batch_size, dim].
      - state1: a NestedMap of updated states (specific to the layer).
    """
    del theta, state0, am_output, ids, paddings
    raise NotImplementedError('Must be implemented by sub-classes.')

  def _ModifyLmBeforeFProp(self, theta, state0, ids, paddings, misc=None):
    """Perform any LM modifications before LM FProp (no-op by default)."""
    del theta, state0, ids, paddings, misc

  def ComputeLogitsWithLM(self, state, logits, is_eval=False):
    """Compute resulting logits based on the fusion method.

    Args:
      state: a NestedMap of states (specific to the layer).
      logits: a tensor corresponds to AM logits.
      is_eval: whether this is used in eval model (for example, beam search).

    Returns:
      Resulting logits after fusion with the LM.

    Raises:
      NotImplementedError: If method is not implemented.
    """
    del state, logits, is_eval
    raise NotImplementedError('Must be implemented by sub-classes.')

  def AddAdditionalDecoderSummaries(self, source_encs, source_paddings, targets,
                                    seq_out_tas, softmax_input):
    """Add any fusion related summaries (no-op by default).

    Args:
      source_encs: A tensor of shape [time, batch_size, source_dim].
      source_paddings: A tensor of shape [time, batch_size].
      targets: A NestedMap containing target info.
      seq_out_tas: A SequenceOutTensorArrays.
      softmax_input: A tensor of shape [batch, time, vocab_size].
    """
    del source_encs, source_paddings, targets, seq_out_tas, softmax_input


class NullFusion(FusionBase):
  """A trivial fusion layer which does nothing."""

  def FProp(self, theta, state0, am_output, ids, paddings, misc=None):
    del theta, ids, paddings
    return am_output, state0

  def ComputeLogitsWithLM(self, state, logits, is_eval=False):
    return tf.nn.log_softmax(logits) if is_eval else logits
