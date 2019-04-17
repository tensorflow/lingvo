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
"""Helper class for implementing a beam search decoder.

Individual models just need to provide a few callback functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core.ops import py_x_ops

# TODO(yonghui):
#   1) Change the tensor shape [max_decoder_time_steps, batch_size *
#   num_hyps_per_beam] to [max_decoder_time_steps, num_hyps_per_beam,
#   batch_size] to avoid confusing and mis-interpretation of the results.

# Defines a namedtuple to store the results of BeamSearchDecode. It contains
# the following entries:
#  done_hyps: A string Tensor of shape
#    [max_decoder_time_steps, batch_size * num_hyps_per_beam] which can be
#    either an empty string, or a serialized Hypothesis proto. The non-empty
#    hyps in done_hyps are terminated hypotheses. The 'h'-th hyp for sample
#    'b' at time step 't' can be found at done_hyps[t, batch_size * h + b].
#  topk_hyps: A string Tensor of shape [batch_size, num_hyps_per_beam].
#    topk_hyps[b, h] is the h-th hypothesis for the sample 'b' in the
#    batch, which can either be an empty string or a serialized Hypothesis
#    proto.
#  topk_ids: Int32 Tensor of shape [batch_size * num_hyps_per_beam,
#    target_seq_len] which contains the IDs of the targets in each of the
#    hypotheses in the beam for the samples in the batch. For sample
#    'b' in the batch, the h-th hypothesis for this sample can be found at
#    position [b * num_hyps_per_beam + h, :].
#  topk_lens: Int32 Tensor of shape [batch_size * num_hyps_per_beam] which
#    indicates the length (>=0) of each of the hypotheses.
#  topk_scores: Float32 Tensor of shape [batch_size * num_hyps_per_beam]
#    containing the scores (negative log probabilities) of each of the
#    hypotheses in the beam.
#  topk_decoded: A string Tensor of shape [batch_size * num_hyps_per_beam] which
#    contains the decoded target strings in each of the hypotheses in the
#    beam for the samples in the batch. The 'h'-th hyp for sample 'b' can
#    be found at topk_decoded[b * num_hyps_per_beam + h]
BeamSearchDecodeOutput = collections.namedtuple(
    'BeamSearchDecodeOutput',
    [
        'done_hyps', 'topk_hyps', 'topk_ids', 'topk_lens', 'topk_scores',
        'topk_decoded', 'other_states'
    ],
)
# Make the last attribute default to None.
BeamSearchDecodeOutput.__new__.__defaults__ = (None,)


class BeamSearchHelper(base_layer.BaseLayer):
  """Helper class for performing beam search.

  The user of this helper class needs to implement three callbacks.

  This callback is called once only at the beginning of beam search:

  .. code-block:: none

      def InitBeamSearchState(theta, encoder_outputs, num_hyps_per_beam):
        Args:
          theta: A NestedMap object containing weights' values of this
              layer and its children layers.
          encoder_outputs: A NestedMap computed by encoder.
          num_hyps_per_beam: An int, number hyps to keep for source sentence.
        Returns:
          initial_results: a `.NestedMap` of initial results. It should contain
              the following tensors at the minimum.
              .log_probs: The initial log probs for each of the tokens in
                  the target vocab, of shape [num_hyps_per_beam * src_batch,
                  vocab_size]. src_batch "b" and hyp_per_beam "h" is
                  represented at index (h * src_batch + b).
              .atten_probs: The initial attention probs, of shape [
                  num_hyps_per_beam * src_batch, src_len]. src_batch "b"
                  and hyp_per_beam "h" is represented at index
                  (h * src_batch + b).
          states: a `.NestedMap` of tensors representing states that the client
              would like to keep track of for each hyp.

  This callback is called once every decoding time step before beam_search_step
  is called:

  .. code-block:: none

      def PreBeamSearchStepCallback(theta,
                                    encoder_outputs,
                                    step_ids,
                                    in_states,
                                    num_hyps_per_beam):
        Args:
          theta: A NestedMap object containing weights' values of this
              layer and its children layers.
          encoder_outputs: A NestedMap computed by encoder.
          step_ids: A tensor of shape [num_hyps_per_beam * src_batch, 1].
          in_states: A `.NestedMap` of tensors representing states that the
              clients would like to keep track of for each of the active hyps.
        Returns:
          results: A `.NestedMap` of beam search results. It should contain
              the 'atten_probs' and 'log_probs' tensors at the minimal.
              Optionally it may contain 'is_last_chunk' if it is decoding a
              neural transducer model.
              .atten_probs: The updated attention probs, of shape
                  [num_hyps_per_beam * src_batch, src_len]. src_batch "b" and
                  hyp_per_beam "h" is represented at index (h * src_batch + b).
              .log_probs: Log prob for each of the tokens in the target vocab.
                  This is of shape [num_hyps_per_beam * src_batch, vocab_size].
                  src_batch "b" and hyp_per_beam "h" is represented at index
                  (h * src_batch + b).
              .is_last_chunk: Whether or not each of the hyp is at the end of a
                  chunk. If non-empty, it is of shape
                  [num_hyps_per_beam * src_batch, 1].
          out_states: A `.NestedMap`. The updated states. This 'out_states'
              should be of the exact same structure as 'in_states'

  This callback is called once every decoding time step after beam_search_step
  is called:

  .. code-block:: none

      def PostBeamSearchStepCallback(theta,
                                     encoder_outputs,
                                     new_step_ids,
                                     other_states):
        Args:
          theta: A NestedMap object containing weights' values of this
              layer and its children layers.
          encoder_outputs: A NestedMap computed by encoder.
          new_step_ids: Token ids for the next beam search step.
          other_states: A `.NestedMap`.
        Returns:
          final_states, A `.NestedMap`.
  """

  @classmethod
  def Params(cls):
    p = super(BeamSearchHelper, cls).Params()
    p.Define('num_hyps_per_beam', 8,
             'Num of hyps to keep per beam during decoding.')
    p.Define(
        'target_seq_length_ratio', 1.0,
        'Ratio of the average target sequence length over the average '
        'source sequence length.')
    p.Define('length_normalization', 0.0,
             'Beam search length normalization ratio.')
    p.Define('coverage_penalty', 0.0, 'Beam search coverage penalty.')
    p.Define(
        'valid_eos_max_logit_delta', 5.0,
        'During beam search, allow </s> to terminate a hyp only if its '
        'logit is no more than than this value away from the logit of the '
        'best candidate.')
    p.Define(
        'beam_size', 3.0,
        'The maximum difference between best hyp and the worst in a beam.'
        ' This allows to prune our search when none of the active hyp is'
        ' close enough to the current best.')
    p.Define('target_sos_id', 1, 'Id of the start of sentence token.')
    p.Define('target_eos_id', 2, 'Id of the end of sentence token.')
    p.Define(
        'target_eoc_id', -1,
        'Id of the end of chunk token. Used by neural transducer only.'
        ' Set this id to a non-negative value only for NT.')
    p.Define(
        'target_seq_len', 0, 'Maximum allowed target seq length. Note '
        'that decoding terminates if an end of sentence token '
        'is not emitted after target_seq_len decode steps.')
    p.Define(
        'merge_paths', False, 'If true, hyps which are identical when '
        'epsilons are removed will be combined into a single hyp.  The '
        'probability for that combined hyp will be the sum of the '
        'probabilities of the component hyps.  This can only be applied '
        'for epsilon-emitting models (RNN-T and NT).')
    p.Define(
        'allow_empty_terminated_hyp', True, 'Whether it is okay to consider a '
        'hyp that consists only of epsilons as terminated.  By default this '
        'is true, as an utterance may consist of silence.  It should be set '
        'to false when EMBR training epsilon-emitting models (e.g., RNN-T), '
        'which are prone to emit all-epsilon hyps even in the presence of '
        'speech.  Note that a hyp that terminates in EOS is not considered '
        'empty, so this flag has no effect for non-epsilon-emitting models.')
    p.Define(
        'ensure_full_beam', False, 'If True, we will not terminate the search '
        'until both of these conditions are satisfied: we have found '
        'num_hyps_per_beam terminated hyps AND no active hyps have a score '
        'within beam_size of the best terminated hyp.  If False, only the '
        'second condition must be satisfied.  Note that in either case, we can '
        'also terminate if we have run for target_seq_len steps.  Generally '
        'this should be False unless beam search is being run as part of '
        'minimum word error rate training.')
    p.Define(
        'force_eos_in_last_step', False,
        'For all active hyps that are still on the beam after target_seq_len '
        'steps, return partial hyps with EOS set as the last token.')
    p.Define(
        'batch_major_state', True, 'If True, we use batch as the major '
        'dimension of the hyp states. Otherwise, timing becomes the major '
        'dimension, and the gathers are performed along the second-to-major '
        'dimension.')
    p.name = 'beam_search'
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BeamSearchHelper, self).__init__(params)
    p = self.params
    self._model_uses_eoc_id = p.target_eoc_id >= 0

  def _BeamSearchStep(self, theta, encoder_outputs, cur_step, step_ids,
                      core_bs_states, other_states, num_hyps_per_beam,
                      pre_beam_search_step_callback,
                      post_beam_search_step_callback):
    """Extend beam search hyps for one step.

      | num_beams = Number of source sequences to be decoded.
      | num_hyps_per_beam = Number of hyps to keep per source sequence.
      | num_hyps = num_beams * num_hyps_per_beam
      | src_seq_len = Number of time steps in the source sequence.
      | src_batch = Number of examples in the source sequence.
      | tgt_seq_len = Maximum allowed time steps in the target sequence.
      | tgt_batch = num_hyps_per_beam * src_batch

    Args:
      theta: A `.NestedMap` object containing weights' values of the decoder
          layer and its children layers.
      encoder_outputs: A `.NestedMap` containing encoder outputs to be passed
        to the callbacks.
      cur_step: A scalar int tensor, the current time step, 0-based.
      step_ids: An int tensor of shape [num_hyps, 1]. The input ids to the
          current search step.
      core_bs_states: A tuple of core beam search states. This list is
          maintained by this helper class.
      other_states: A `.NestedMap` of other beam search states.
          This `.NestedMap` is managed and updated by the client. It is
          expected that each of its member tensors are of rank >= 1. t[i, ...]
          is the state of the i-th hyp at the beginning of this search step.
      num_hyps_per_beam: Num of hyps to keep per beam.
      pre_beam_search_step_callback: The `PreBeamSearchStepCallback` callback.
          See class header comments for more details.
      post_beam_search_step_callback: The `PostBeamSearchStepCallback` callback.
          See class header comments for more details.
    Returns:
      A tuple of following elements for the next beam search step,
      (next step, all_done, step_ids, core_bs_states, other_states)
    """
    p = self.params

    bs_results, other_states = pre_beam_search_step_callback(
        theta, encoder_outputs, step_ids, other_states, num_hyps_per_beam)

    (best_scores, cumulative_scores, in_scores, in_hyps, in_prev_hyps,
     in_done_hyps, in_atten_probs) = core_bs_states

    (out_best_scores, out_cumulative_scores, out_scores, out_hyps,
     out_prev_hyps, out_done_hyps, out_atten_probs,
     all_done) = py_x_ops.beam_search_step(
         bs_results.log_probs,
         bs_results.atten_probs,
         best_scores,
         cumulative_scores,
         in_scores,
         in_hyps,
         in_prev_hyps,
         in_done_hyps,
         in_atten_probs,
         bs_results.is_last_chunk if self._model_uses_eoc_id else [],
         cur_step,
         eoc_id=p.target_eoc_id,
         eos_id=p.target_eos_id,
         beam_size=p.beam_size,
         num_hyps_per_beam=num_hyps_per_beam,
         valid_eos_max_logit_delta=p.valid_eos_max_logit_delta,
         merge_paths=p.merge_paths,
         allow_empty_terminated_hyp=p.allow_empty_terminated_hyp,
         ensure_full_beam=p.ensure_full_beam,
         force_eos_in_last_step=p.force_eos_in_last_step)

    new_step_ids = tf.reshape(out_hyps[cur_step, :], tf.shape(step_ids))
    new_step_ids.set_shape(step_ids.get_shape())

    old_hyp_ids = tf.reshape(
        tf.slice(out_prev_hyps, begin=[cur_step, 0], size=[1, -1]), [-1])

    new_bs_states = (out_best_scores, out_cumulative_scores, out_scores,
                     out_hyps, out_prev_hyps, out_done_hyps, out_atten_probs)

    def ReOrderHyps(x_in):
      """Reorders x_in based on prev hyp ids."""
      if (isinstance(x_in, tf.Tensor) and x_in.shape.ndims and
          x_in.shape.ndims > 0):
        if x_in.shape.ndims > 2 and not p.batch_major_state:
          x_out = tf.gather(x_in, old_hyp_ids, axis=1)
        else:
          x_out = tf.gather(x_in, old_hyp_ids)
        x_out.set_shape(x_in.get_shape())
        return x_out
      else:
        return x_in

    new_other_states = other_states.Transform(ReOrderHyps)

    final_other_states = post_beam_search_step_callback(
        theta, encoder_outputs, new_step_ids, new_other_states)

    return (cur_step + 1, all_done, new_step_ids, new_bs_states,
            final_other_states)

  def BeamSearchDecode(self,
                       theta,
                       encoder_outputs,
                       num_hyps_per_beam_override=0,
                       init_beam_search_state=None,
                       pre_beam_search_step_callback=None,
                       post_beam_search_step_callback=None,
                       max_steps=None):
    """Performs beam-search based decoding.

    Args:
      theta: A NestedMap object containing weights' values of the decoder
        layer and its children layers.
      encoder_outputs: A NestedMap containing encoder outputs to be passed
        to the callbacks.
      num_hyps_per_beam_override: If set to a value <= 0, this parameter is
        ignored. If set to a value > 0, then this value will be used to
        override `p.num_hyps_per_beam`.
      init_beam_search_state: The `InitBeamSearchState` callback. Please refer
          to the class header comments for more details.
      pre_beam_search_step_callback: The `PreBeamSearchStepCallback` callback.
          Please refer to the class header comments for more details.
      post_beam_search_step_callback: The `PostBeamSearchStepCallback` callback.
          Please refer to the class header comments for more details.
      max_steps: maximum beam search steps. If None,
          use self.params.target_seq_len.

    Returns:
      A `BeamSearchDecodeOutput`.
    """
    p = self.params
    num_hyps_per_beam = p.num_hyps_per_beam
    if num_hyps_per_beam_override > 0:
      num_hyps_per_beam = num_hyps_per_beam_override
    if max_steps is None:
      max_steps = p.target_seq_len

    initial_results, other_states = init_beam_search_state(
        theta, encoder_outputs, num_hyps_per_beam)

    num_hyps = tf.shape(initial_results.log_probs)[0]
    num_beams = num_hyps // num_hyps_per_beam

    step_ids = tf.fill([num_hyps, 1],
                       tf.constant(p.target_sos_id, dtype=tf.int32))
    min_score = -1e36
    best_scores = (tf.zeros(shape=[num_beams], dtype=p.dtype) + min_score)
    cumulative_scores = tf.zeros(shape=[num_hyps], dtype=p.dtype)
    in_scores = tf.zeros([max_steps, num_hyps], dtype=p.dtype)
    in_hyps = tf.zeros([max_steps, num_hyps], dtype=tf.int32)
    in_prev_hyps = tf.zeros([max_steps, num_hyps], dtype=tf.int32)
    in_done_hyps = tf.zeros([max_steps, num_hyps], dtype=tf.string)
    bs_atten_probs = tf.zeros(
        [max_steps, num_hyps,
         tf.shape(initial_results.atten_probs)[1]],
        dtype=p.dtype)
    cur_step = tf.constant(0, dtype=tf.int32)
    all_done = tf.constant(False, dtype=tf.bool)
    core_bs_states = (best_scores, cumulative_scores, in_scores, in_hyps,
                      in_prev_hyps, in_done_hyps, bs_atten_probs)

    def LoopContinue(cur_step, all_done, unused_step_ids, unused_core_bs_states,
                     unused_other_states_list):
      return tf.logical_and(cur_step < max_steps, tf.logical_not(all_done))

    def LoopBody(cur_step, unused_all_done, step_ids, core_bs_states,
                 other_states_list):
      (cur_step, all_done, new_step_ids, new_bs_states,
       new_other_states) = self._BeamSearchStep(
           theta, encoder_outputs, cur_step, step_ids, core_bs_states,
           other_states.Pack(other_states_list), num_hyps_per_beam,
           pre_beam_search_step_callback, post_beam_search_step_callback)
      return (cur_step, all_done, new_step_ids, new_bs_states,
              new_other_states.Flatten())

    flat_other_states = other_states.Flatten()
    _, _, _, final_bs_states, flat_final_other_states = tf.while_loop(
        LoopContinue,
        LoopBody,
        loop_vars=(cur_step, all_done, step_ids, core_bs_states,
                   flat_other_states),
        parallel_iterations=10,
        back_prop=False,
        swap_memory=False,
        shape_invariants=(tf.TensorShape(cur_step.get_shape()),
                          tf.TensorShape(all_done.get_shape()),
                          tf.TensorShape(step_ids.get_shape()),
                          _GetShapes(core_bs_states),
                          _GetShapes(flat_other_states, none_shapes=True)))
    # [target_seq_len, num_beams * num_hyps_per_beam].
    final_done_hyps = final_bs_states[5]
    final_other_states = other_states.Pack(flat_final_other_states)

    # TODO(rpang): avoid inspecting 'encoder_outputs'.
    source_paddings = encoder_outputs.padding
    if isinstance(source_paddings, py_utils.NestedMap):
      source_seq_lengths = tf.to_int32(
          tf.reduce_sum(1.0 - tf.transpose(source_paddings.Flatten()[0]), 1))
    else:
      source_seq_lengths = tf.to_int32(
          tf.reduce_sum(1.0 - tf.transpose(source_paddings), 1))

    # [num_beams, num_hyps_per_beam].
    topk_hyps = py_x_ops.top_k_terminated_hyps(
        final_done_hyps,
        source_seq_lengths,
        k=num_hyps_per_beam,
        num_hyps_per_beam=num_hyps_per_beam,
        length_normalization=p.length_normalization,
        coverage_penalty=p.coverage_penalty,
        target_seq_length_ratio=p.target_seq_length_ratio,
        eoc_id=p.target_eoc_id,
        merge_paths=p.merge_paths)
    # [num_beams * num_hyps_per_beam, ...].
    max_seq_length = 0 if isinstance(max_steps, tf.Tensor) else max_steps
    topk_ids, topk_lens, topk_scores = py_x_ops.unpack_hyp(
        tf.reshape(topk_hyps, [-1]), max_seq_length=max_seq_length)
    # [num_beams, num_hyps_per_beam].
    topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))

    return BeamSearchDecodeOutput(final_done_hyps, topk_hyps, topk_ids,
                                  topk_lens, topk_scores, None,
                                  final_other_states)


def _GetShapes(tensors, none_shapes=False):
  """Util for getting nested structure of shapes from structure of tensors.

  Args:
    tensors: Structure of Tensors to get shapes for.
    none_shapes: Returns None shapes if true.

  Returns:
    The same structure as tensors but of corresponding `TensorShape` objects.
  """
  shapes = []
  for t in tf.contrib.framework.nest.flatten(tensors):
    shape = t.get_shape() if isinstance(t, tf.Tensor) else None
    if none_shapes:
      if shape:
        shapes.append(tf.TensorShape([None] * len(shape)))
      else:
        shapes.append(tf.TensorShape(None))
    else:
      shapes.append(tf.TensorShape(shape))

  return type(tensors)(
      tf.contrib.framework.nest.pack_sequence_as(tensors, shapes))


def MergeBeamSearchOutputs(max_hyps_per_beam, beam_search_outputs):
  """Merges beam search hyps from multiple decoders.

  Args:
    max_hyps_per_beam: the number of top hyps in the merged results. Must be
      less than or equal to total number of input hyps.
    beam_search_outputs: a list of BeamSearchDecodeOutput objects. Must share
      the same source_batch and max sequence length.

  Returns:
    A BeamSearchDecodeOutput object containing max_hyps_per_beam hypotheses per
    beam.
  """
  source_batch = tf.shape(beam_search_outputs[0].topk_hyps)[0]
  value_dict = {}
  for output in beam_search_outputs:
    hyps_per_beam = py_utils.with_dependencies([
        py_utils.assert_equal(source_batch,
                              tf.shape(output.topk_hyps)[0]),
    ],
                                               tf.shape(output.topk_hyps)[1])
    for k, v in output._asdict().iteritems():
      if v is None:
        continue
      if k == 'done_hyps':
        v = tf.transpose(v)
      if k not in value_dict:
        value_dict[k] = []
      value_dict[k].append(tf.reshape(v, [source_batch, hyps_per_beam, -1]))

  # Concatenate the tensors along the 'num_hyps_per_beam' dimension.
  concatenated = {}
  for k, values in value_dict.iteritems():
    if len(values) != len(beam_search_outputs):
      raise ValueError(
          'Incomplete values for %s: %s' % (k, beam_search_outputs))
    concatenated[k] = tf.concat(values, axis=1)

  scores = concatenated['topk_scores']
  scores = tf.where(
      tf.equal(concatenated['topk_lens'], 0), tf.fill(tf.shape(scores), -1e6),
      scores)
  scores = tf.squeeze(scores, -1)

  # Select top max_hyps_per_beam indices per beam.
  _, top_indices = tf.nn.top_k(scores, max_hyps_per_beam)
  batch_ids = tf.tile(
      tf.expand_dims(tf.range(source_batch), -1), [1, max_hyps_per_beam])
  # [source_batch, max_hyps_per_beam, 2]
  gather_indices = tf.stack([batch_ids, top_indices], axis=-1)

  # Gather the merged top hyps according to 'gather_indices'.
  top = beam_search_outputs[0]._asdict()
  total_hyps = source_batch * max_hyps_per_beam
  for k, v in concatenated.iteritems():
    v = tf.gather_nd(v, gather_indices)
    if k == 'done_hyps':
      v = tf.transpose(tf.reshape(v, [total_hyps, -1]))
    elif k == 'topk_hyps':
      v = tf.reshape(v, [source_batch, max_hyps_per_beam])
    elif k == 'topk_ids':
      v = tf.reshape(v, [total_hyps, -1])
    elif k in ('topk_lens', 'topk_scores', 'topk_decoded'):
      v = tf.reshape(v, [total_hyps])
    else:
      raise ValueError('Unexpected field: %s' % k)
    top[k] = v
  return BeamSearchDecodeOutput(**top)
