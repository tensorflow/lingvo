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
"""Steps for attention computation."""

from lingvo import compat as tf
from lingvo.core import attention
from lingvo.core import py_utils
from lingvo.core import step
from lingvo.core.steps import rnn_steps


class AttentionStep(step.Step):
  """AttentionStep wraps an attention layer in the Step interface.

  An attention algorithm outputs a targeted summary of a set of input vectors.

  At each step, the query vector (input to FProp as step_inputs.input)
  describes what data should be returned. The attention algorithm compares the
  query vector with the source vectors (external_inputs.src) to compute
  weights (attention_probs).

  The result is the weighted sum (using attention_probs) of a set of vectors.
  By default that set of vectors is also external_inputs.src, but it can
  optionally be external_inputs.context if a context tensor is specified.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'attention_step'
    p.Define('atten', attention.AdditiveAttention.Params(),
             'Params of a subclass of BaseAttentionLayer.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('atten', p.atten)

  def PrepareExternalInputs(self, theta, external_inputs):
    """Prepare encoded source data for processing.

    In some attention algorithms, this step will pre-process the source
    feature data so that FProp runs faster.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      external_inputs: A NestedMap containing tensors:

        - src: a [time, batch, depth] tensor that forms the input to the
          attention layer.
        - padding: a [time, batch] 0/1 tensor indicating which parts of
          src contain useful information.
        - context: Optional. See the class documentation for more details.

    Returns:
      state0, a state parameter to pass to FProp on its first invocation.
    """
    packed_inputs = external_inputs.DeepCopy()
    if 'context' in external_inputs:
      del packed_inputs['context']
      context = external_inputs.context
    else:
      context = external_inputs.src
    packed_inputs.packed_src = self.atten.InitForSourcePacked(
        theta.atten, external_inputs.src, context, external_inputs.padding)
    return packed_inputs

  def _GetMaxSeqLength(self, src_encs):
    """Compute the maximum sequence length of the encoded source sequence.

    Args:
      src_encs:  Encoded source sequence pre-processed by using
        PrepareExternalInputs. It can be either a [time, batch, depth] tensor
        (when there is only one source) or a NestedMap of [time, batch, depth]
        tensors (when there are more than one source).

    Returns:
      max_seq_length: the maximum sequence length of the encoded source
      sequence. It can be either a scalar (when there is only one source) or
      a NestedMap of scalars (when there are more than one source).
    """
    # TODO(shaojinding): Create a MultiSourceAttentionStep class for the
    # scenarios when there are multiple sources to avoid the use of if/else.
    if isinstance(src_encs, py_utils.NestedMap):
      max_seq_length = py_utils.NestedMap()
      for key in src_encs:
        max_seq_length[key] = py_utils.GetShape(src_encs[key], 3)[0]
    else:
      max_seq_length = py_utils.GetShape(src_encs, 3)[0]
    return max_seq_length

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Produce a zero state for this step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: A set of inputs pre-processed by using
        PrepareExternalInputs.
      batch_size: Number of elements in the batched input.

    Returns:
      state0, a state parameter to pass to FProp on its first invocation.
    """
    max_seq_length = self._GetMaxSeqLength(prepared_inputs.src)
    atten_state = self.atten.ZeroAttentionState(max_seq_length, batch_size)
    (new_atten_context, _,
     new_atten_states) = self.atten.ComputeContextVectorWithSource(
         theta.atten,
         prepared_inputs.packed_src,
         tf.zeros([batch_size, self.params.atten.query_dim],
                  dtype=py_utils.FPropDtype(self.params)),
         attention_state=atten_state)
    return py_utils.NestedMap(
        atten_context=new_atten_context, atten_state=new_atten_states)

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Produces a context vector from the attention algorithm.

    The context vector is a summary of the inputs from external_inputs
    which the attention algorithm has determined would be useful for decoding
    the next output.

    Args:
      theta: A NestedMap containing weights' values of this layer and its
        children layers.
      prepared_inputs: A set of encoded tensors that have been pre-processed by
        PrepareExternalInputs.
      step_inputs: A NestedMap containing an 'inputs' tensor with the query
        vector to use.
      padding: A [batch, 1] 0/1 float tensor, where 1.0 means that this batch
        slot is not used.
      state0: A NestedMap of state, either produced by ZeroState or a previous
        invocation of this graph.

    Returns:
      output, state1, defined as follows:
      - output: a NestedMap containing a query tensor, a context tensor, and
        cum_atten_probs, the log of attention probabilities for each input
        vector.
      - state1: a NestedMap of state to be used in subsequent invocations of
        this graph.
    """
    (new_atten_context, new_atten_probs,
     new_atten_states) = self.atten.ComputeContextVectorWithSource(
         theta.atten,
         prepared_inputs.packed_src,
         tf.concat(step_inputs.inputs, axis=1),
         attention_state=state0.atten_state)
    new_atten_probs = py_utils.ApplyPadding(padding, new_atten_probs)
    output = py_utils.NestedMap(
        context=new_atten_context, probs=new_atten_probs)
    state1 = py_utils.NestedMap(
        atten_context=new_atten_context, atten_state=new_atten_states)
    return output, state1


class AttentionBlockStep(step.Step):
  """Computes attention queries and context vectors.

  An attention algorithm produces a summary of a set of input vectors.
  A query vector is used as an input to the process; we can think of this as
  describing what information we're hoping to retrieve in the summary.
  The summary output is called a context vector.

  This class uses attention as a way to view the input to a decoder.
  In each step, we hope to read a summary of the encoded input that would be
  most useful for generating the next output from the decoder.

  To do this, we combine a query generator and an attention algorithm.
  The query generator's job is to build a query vector that represents the
  current decoder state. The attention algorithm then uses that query vector
  as a key to decide which encoded inputs are most important, and it combines
  those into a context vector.

  The query generator takes two inputs: one is the context vector output by
  the attention algorithm in the previous step, and the other is the label
  output by the decoder in the previous step. It combines this information
  in an implementation-dependent way to produce a query vector.

  In previous implementations, the query generator was just the first layer
  of a stack of RNN layers (rnn_cell[0]). In this implementation, the query
  generator can be any suitable Step, but in practice is one or more RNN layers.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'attention_block'
    p.Define('query_generator', rnn_steps.RnnStackStep.Params(),
             'Query generator params.')
    p.Define(
        'attention', AttentionStep.Params(),
        'Attention params. This can be either an AttentionStep.Params()'
        'or a list of them.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('query_generator', p.query_generator)
    self.CreateChild('attention', p.attention)

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Produce a zero state for this step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: A set of inputs pre-processed by using
        PrepareExternalInputs.
      batch_size: Number of elements in the batched input.

    Returns:
      state0, a state parameter to pass to FProp on its first invocation.
    """
    query_state0 = self.query_generator.ZeroState(
        theta.query_generator, prepared_inputs.query_generator, batch_size)
    atten_state0 = self.attention.ZeroState(theta.attention,
                                            prepared_inputs.attention,
                                            batch_size)
    state0 = py_utils.NestedMap(
        query_state=query_state0, atten_state=atten_state0)
    return state0

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Produces a query vector and a context vector for the next decoder step.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      prepared_inputs: A set of encoded tensors that have been pre-processed by
        PrepareExternalInputs.
      step_inputs: Unused. All of the input for this step comes from
        external_inputs and previous step state.
      padding: A [batch, 1] 0/1 float tensor, where 1.0 means that this batch
        slot is not used.
      state0: A NestedMap of state, either produced by ZeroState or a previous
        invocation of this graph.

    Returns:
      output, state1, are defined as follows.
      output, a NestedMap containing an atten_query tensor,
      an atten_context tensor, and atten_probs, attention probabilities for each
      input vector.
      state1, a NestedMap of state to be used in subsequent invocations of this
      graph.
    """
    query_output, query_state1 = self.query_generator.FProp(
        theta.query_generator, prepared_inputs.query_generator,
        py_utils.NestedMap(inputs=[state0.atten_state.atten_context]), padding,
        state0.query_state)
    atten_input = py_utils.NestedMap(inputs=[query_output.output])
    atten_output, atten_state1 = self.attention.FProp(theta.attention,
                                                      prepared_inputs.attention,
                                                      atten_input, padding,
                                                      state0.atten_state)
    state1 = py_utils.NestedMap(
        atten_state=atten_state1, query_state=query_state1)
    return py_utils.NestedMap(
        atten_context=atten_output.context,
        atten_query=query_output.output,
        atten_probs=atten_output.probs), state1
