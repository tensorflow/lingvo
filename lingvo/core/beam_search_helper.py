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
"""Helper class for implementing a beam search decoder.

Individual models just need to provide a few callback functions.
"""

import collections
import re
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import ops
from lingvo.core import py_utils

from tensorflow.python.ops import inplace_ops

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
#  topk_scores: Float32 Tensor of shape [batch_size, num_hyps_per_beam]
#    containing the scores (negative log probabilities) of each of the
#    hypotheses in the beam.
#  topk_decoded: A string Tensor of shape [batch_size * num_hyps_per_beam] which
#    contains the decoded target strings in each of the hypotheses in the
#    beam for the samples in the batch. The 'h'-th hyp for sample 'b' can
#    be found at topk_decoded[b * num_hyps_per_beam + h]
BeamSearchDecodeOutput = collections.namedtuple(
    'BeamSearchDecodeOutput',
    [
        'topk_hyps', 'topk_ids', 'topk_lens', 'topk_scores', 'topk_decoded',
        'other_states'
    ],
)
# Make the last attribute default to None.
BeamSearchDecodeOutput.__new__.__defaults__ = (None,)


# Keys in fusion state that can be two dimensional, with the batch element in
# the second dimension, requiring special treatment in hypothesis reordering.
POSSIBLY_TIME_MAJOR_STATE_KEYS = [
    'misc_states.fusion_states.lm_states.prev_ids',
    'misc_states.fusion_states.lm_states.prev_paddings',
    'fusion_states.lm_states.prev_ids',
    'fusion_states.lm_states.prev_paddings',
]


class BeamSearchSharedParams(base_layer.BaseLayer):
  """Class defining common beam search params."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_hyps_per_beam', 8,
             'Num of hyps to keep per beam during decoding.')
    p.Define(
        'target_seq_length_ratio', 1.0,
        'Ratio of the average target sequence length over the average '
        'source sequence length. Affects coverage penalty.')
    p.Define(
        'length_normalization', 0.0,
        'Beam search length normalization factor, typically in [0, 1]. '
        'This is the exponent on (len+5)/5 used to normalize '
        'global score. The larger this value is, the more likely '
        'longer sequences are produced. This value is alpha '
        'in https://arxiv.org/abs/1609.08144, equation 14.')
    p.Define(
        'coverage_penalty', 0.0,
        'Beam search coverage penalty. This value is beta in '
        'https://arxiv.org/abs/1609.08144, equation 14. The higher this '
        'value is, the more heavily low coverage is penalized.')
    p.Define(
        'valid_eos_max_logit_delta', 5.0,
        'During beam search, allow </s> to terminate a hyp only if its '
        'logit is no more than than this value away from the logit of the '
        'best candidate. The larger this value is, the easier hyps can '
        'terminate, and the more likely shorter sequences are produced.')
    p.Define(
        'local_eos_threshold', -100.0,
        'During beam search, allow </s> to terminate a hyp if the local score '
        'for </s> is greater than local_eos_threshold.')
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
        'force_eos_in_top_k', False,
        'Whether to always consider the eos token to be among the top k tokens '
        'for every step. When False, hyps can only terminate if the eos token '
        'is part of the top k. Note that p.valid_eos_max_logit_delta and '
        'p.local_eos_threshold always apply regardless of this.')
    p.Define(
        'batch_major_state', True, 'If True, we use batch as the major '
        'dimension of the hyp states. Otherwise, timing becomes the major '
        'dimension, and the gathers are performed along the second-to-major '
        'dimension.')
    p.Define(
        'batch_major_compute', False, 'If True, the target batch dimension '
        'is organized as num_beams by num_hyps_per_beam during the '
        'ExtendStep computation and the cache is stored following this order. '
        'So the topk indices into the cache for ReOrderHyps needs to be '
        'reordered before usage. Otherwise, the indices will be directly used '
        'without extra transformation. '
        'Setting batch_major_compute=True does not change the ordering of '
        'ids and logits of beam search callbacks. '
        'The target_batch dim for those tensors will remain num_hyps_per_beam '
        '* num_beams.')
    p.Define(
        'short_seq_limit', 0,
        'An integer, the sequence length limit for using early stop '
        'method in attention layer (batch-major implementation). The sequence '
        'is always treated as the default long sequence for decoding when the '
        'limit is set to 0. For typical mt transformer config '
        '(batch 16, sequence length 150), the break even point is around 40 '
        'on TPU V3, and 50 on TPU V2. This may slightly change for '
        'different batch size and sequence length, which requires more '
        'experiments to set the value.')
    p.Define(
        'terminate_beams_independently', False,
        'Whether each beam in the same batch can independently terminate. '
        'This controls whether the search termination criteria set by params '
        'like `p.beam_size` or `p.ensure_full_beam` are applied collectively '
        'to all beams, or individually to each beam. When False, all beams '
        'continue the search until each and every beam meets the termination '
        'criteria. When True, each beam individually, independent of each '
        'other, decides whether to terminate the search.')
    return p


class BeamSearchHelper(BeamSearchSharedParams):
  """Helper class for performing beam search.

  The user of this helper class needs to implement three callbacks.

  This callback is called once only at the beginning of beam search:

  .. code-block:: none

      def InitBeamSearchState(theta, encoder_outputs, num_hyps_per_beam):
        Args:
          theta: A NestedMap object containing weights' values of this layer and
            its children layers.
          encoder_outputs: A NestedMap computed by encoder.
          num_hyps_per_beam: An int, number hyps to keep for source sentence.

        Returns:
          A tuple (initial_results, states):

          - initial_results: a `.NestedMap` of initial results. It must contain
            the 'atten_probs' and 'log_probs' tensors. Optionally it may
            contain 'step_ids'.

            - log_probs: The initial log probs for each of the tokens in the
              target vocab of shape [num_hyps_per_beam * src_batch, vocab_size].
              src_batch "b" and hyp_per_beam "h" is represented at index
              ``(h * src_batch + b)``.
            - atten_probs: The initial attention probs, of shape
              [num_hyps_per_beam * src_batch, src_len]. src_batch "b" and
              hyp_per_beam "h" is represented at index ``(h * src_batch + b)``.
            - step_ids: Optional. The initial ids of shape [num_hyps_per_beam *
              src_batch, 1] for which to start the beam search. src_batch "b"
              hyp_per_beam "h" is represented at index ``(h * src_batch + b)``.
              If not specified, defaults to a tensor filled with target_sos_id.

          - states: a `.NestedMap` of tensors representing states that the
            client would like to keep track of for each hyp.

  This callback is called once every decoding time step before beam_search_step
  is called:

  .. code-block:: none

      def PreBeamSearchStepCallback(theta,
                                    encoder_outputs,
                                    step_ids,
                                    in_states,
                                    num_hyps_per_beam):
        Args:
          theta: A NestedMap object containing weights' values of this layer and
            its children layers.
          encoder_outputs: A NestedMap computed by encoder.
          step_ids: A tensor of shape [num_hyps_per_beam * src_batch, 1].
          in_states: A `.NestedMap` of tensors representing states that the
            clients would like to keep track of for each of the active hyps.

        Returns:
          A tuple (results, out_states):

          - results: A `.NestedMap` of beam search results. It should contain
            the 'atten_probs' and 'log_probs' tensors at the minimal.
            Optionally it may contain 'is_last_chunk' if it is decoding a
            neural transducer model.

            - atten_probs: The updated attention probs, of shape
              [num_hyps_per_beam * src_batch, src_len]. src_batch "b" and
              hyp_per_beam "h" is represented at index ``(h * src_batch + b)``.
            - log_probs: Log prob for each of the tokens in the target vocab.
              This is of shape [num_hyps_per_beam * src_batch, vocab_size].
              src_batch "b" and hyp_per_beam "h" is represented at index
              ``(h * src_batch + b)``.
            - is_last_chunk: Whether each of the hyp is at the end of a chunk.
              If non-empty, it has shape [num_hyps_per_beam * src_batch, 1].

          - out_states: A `.NestedMap`. The updated states. This 'out_states'
            should be of the exact same structure as 'in_states'

  This callback is called once every decoding time step after beam_search_step
  is called:

  .. code-block:: none

      def PostBeamSearchStepCallback(theta,
                                     encoder_outputs,
                                     new_step_ids,
                                     other_states):
        Args:
          theta: A NestedMap object containing weights' values of this layer and
            its children layers.
          encoder_outputs: A NestedMap computed by encoder.
          new_step_ids: Token ids for the next beam search step.
          other_states: A `.NestedMap`.

        Returns:
          final_states, A `.NestedMap`.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
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
        'until both of these conditions are satisfied: we have found at least '
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
        'atten_vecs_in_hypothesis_protos', True,
        'Whether to write atten_vecs fields in the Hypothesis protos. Setting '
        'this to False saves memory, and can be used when the protos become '
        'too large for long sequences, but requires p.coverage_penalty == 0.0.')
    p.name = 'beam_search'
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._model_uses_eoc_id = p.target_eoc_id >= 0
    if not p.atten_vecs_in_hypothesis_protos and p.coverage_penalty != 0.0:
      raise ValueError('p.atten_vecs_in_hypothesis_protos requires '
                       'p.coverage_penalty == 0.0.')

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
      encoder_outputs: A `.NestedMap` containing encoder outputs to be passed to
        the callbacks.
      cur_step: A scalar int tensor, the current time step, 0-based.
      step_ids: An int tensor of shape [num_hyps, 1]. The input ids to the
        current search step.
      core_bs_states: A tuple of core beam search states. This list is
        maintained by this helper class.
      other_states: A `.NestedMap` of other beam search states. This
        `.NestedMap` is managed and updated by the client. It is expected that
        each of its member tensors are of rank >= 1. t[i, ...] is the state of
        the i-th hyp at the beginning of this search step.
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
     in_done_hyps, in_atten_probs, in_beam_done) = core_bs_states

    (out_best_scores, out_cumulative_scores, out_scores, out_hyps,
     out_prev_hyps, out_done_hyps, out_atten_probs, out_beam_done,
     all_done) = ops.beam_search_step(
         tf.cast(bs_results.log_probs, dtype=p.dtype),
         tf.cast(bs_results.atten_probs, dtype=p.dtype),
         best_scores,
         cumulative_scores,
         in_scores,
         in_hyps,
         in_prev_hyps,
         in_done_hyps,
         in_atten_probs,
         in_beam_done,
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
         force_eos_in_last_step=p.force_eos_in_last_step,
         force_eos_in_top_k=p.force_eos_in_top_k,
         local_eos_threshold=p.local_eos_threshold,
         beam_independence=p.terminate_beams_independently,
         atten_vecs_in_hypothesis_protos=p.atten_vecs_in_hypothesis_protos)

    new_step_ids = tf.reshape(out_hyps[cur_step, :], tf.shape(step_ids))
    new_step_ids.set_shape(step_ids.get_shape())

    # [num_hyps_per_beam * num_beams].
    old_hyp_ids = tf.reshape(
        tf.slice(out_prev_hyps, begin=[cur_step, 0], size=[1, -1]), [-1])

    if p.batch_major_compute:
      # Transformed the indices into the key/value cache for fast decoding
      # (prefix_states in other_states) due to the num_hyps dimension of
      # cache is computed as num_beams by num_hyps_per_beam, which is different
      # from the old_hyp_ids assumption (num_hyps_per_beam by num_beams).
      # Both transpose and recomputation are required to correct the indices.
      num_beams = tf.shape(best_scores)[0]
      # [num_beams * num_hyps_per_beam].
      old_hyp_ids_in_cache_order = tf.reshape(
          tf.transpose(tf.reshape(old_hyp_ids, [num_hyps_per_beam, -1])), [-1])
      old_hyp_ids_in_cache_order = (
          (old_hyp_ids_in_cache_order % num_beams) * num_hyps_per_beam +
          old_hyp_ids_in_cache_order // num_beams)

    new_bs_states = (out_best_scores, out_cumulative_scores, out_scores,
                     out_hyps, out_prev_hyps, out_done_hyps, out_atten_probs,
                     out_beam_done)
    random_seed_regex = re.compile(r'rnn_states\[\d+\].r$')

    def ReOrderHyps(key, x_in):
      """Reorders x_in based on prev hyp ids."""
      if random_seed_regex.match(key):
        # For keys like rnn_states[0].r, it is a shape [2] random seeds tensor
        # used for deterministic behavior and should not be reordered.
        return py_utils.HasShape(x_in, [2])
      correct_old_hyp_ids = (
          old_hyp_ids_in_cache_order if p.batch_major_compute else old_hyp_ids)
      if (isinstance(x_in, tf.Tensor) and x_in.shape.ndims):
        if x_in.shape.ndims > 2 and not p.batch_major_state:
          # Use corrected indices only here for batch major compute as key/value
          # caches are the states being affected.
          x_out = tf.gather(x_in, correct_old_hyp_ids, axis=1)
        elif key in POSSIBLY_TIME_MAJOR_STATE_KEYS:
          x_out = tf.gather(x_in, old_hyp_ids, axis=-1)
        else:
          x_out = tf.gather(x_in, correct_old_hyp_ids)
        x_out.set_shape(x_in.get_shape())
        return x_out
      else:
        return x_in

    new_other_states = other_states.TransformWithKey(ReOrderHyps)

    final_other_states = post_beam_search_step_callback(theta, encoder_outputs,
                                                        new_step_ids,
                                                        new_other_states)

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
      theta: A NestedMap object containing weights' values of the decoder layer
        and its children layers.
      encoder_outputs: A NestedMap containing encoder outputs to be passed to
        the callbacks. Mostly opaque to BeamSearchHelper, except that it should
        contain either a 'seq_lengths' field of shape [source_batch_size] or
        a 'paddings' field of shape [source_max_lengths, source_batch_size].
      num_hyps_per_beam_override: If set to a value <= 0, this parameter is
        ignored. If set to a value > 0, then this value will be used to override
        `p.num_hyps_per_beam`.
      init_beam_search_state: The `InitBeamSearchState` callback. Please refer
        to the class header comments for more details.
      pre_beam_search_step_callback: The `PreBeamSearchStepCallback` callback.
        Please refer to the class header comments for more details.
      post_beam_search_step_callback: The `PostBeamSearchStepCallback` callback.
        Please refer to the class header comments for more details.
      max_steps: maximum beam search steps. If None, use
        self.params.target_seq_len.

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

    if 'step_ids' in initial_results:
      # [num_hyps, 1]
      step_ids = tf.ensure_shape(initial_results.step_ids, [None, 1])
    else:
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
    beam_done = tf.zeros([num_beams], dtype=tf.bool)
    cur_step = tf.constant(0, dtype=tf.int32)
    all_done = tf.constant(False, dtype=tf.bool)
    core_bs_states = (best_scores, cumulative_scores, in_scores, in_hyps,
                      in_prev_hyps, in_done_hyps, bs_atten_probs, beam_done)

    def LoopContinue(cur_step, all_done, unused_step_ids, unused_core_bs_states,
                     unused_other_states_list):
      return tf.math.logical_and(cur_step < max_steps,
                                 tf.math.logical_not(all_done))

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

    # Assume that `paddings` has shape [source_max_lengths, source_batch_size]
    # by default, and compute `encoded_seq_lengths` accordingly. This can be
    # overridden by directly passing `seq_lengths` in the `encoder_outputs`
    # NestedMap.
    encoded_seq_lengths = getattr(encoder_outputs, 'seq_lengths', None)
    if encoded_seq_lengths is None:
      source_paddings = encoder_outputs.padding
      if isinstance(source_paddings, py_utils.NestedMap):
        encoded_seq_lengths = tf.cast(
            tf.round(
                tf.reduce_sum(1.0 - tf.transpose(source_paddings.Flatten()[0]),
                              1)), tf.int32)
      else:
        encoded_seq_lengths = tf.cast(
            tf.round(
                tf.reduce_sum(
                    1.0 - tf.cast(tf.transpose(source_paddings), tf.float32),
                    1)), tf.int32)

    # [num_beams, num_hyps_per_beam].
    topk_hyps = ops.top_k_terminated_hyps(
        final_done_hyps,
        encoded_seq_lengths,
        k=num_hyps_per_beam,
        num_hyps_per_beam=num_hyps_per_beam,
        length_normalization=p.length_normalization,
        coverage_penalty=p.coverage_penalty,
        target_seq_length_ratio=p.target_seq_length_ratio)
    # [num_beams * num_hyps_per_beam, ...].
    max_seq_length = 0 if isinstance(max_steps, tf.Tensor) else max_steps
    topk_ids, topk_lens, topk_scores = ops.unpack_hyp(
        tf.reshape(topk_hyps, [-1]), max_seq_length=max_seq_length)
    # [num_beams, num_hyps_per_beam].
    topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))

    return BeamSearchDecodeOutput(topk_hyps, topk_ids, topk_lens, topk_scores,
                                  None, final_other_states)


def _GetShapes(tensors, none_shapes=False):
  """Util for getting nested structure of shapes from structure of tensors.

  Args:
    tensors: Structure of Tensors to get shapes for.
    none_shapes: Returns None shapes if true.

  Returns:
    The same structure as tensors but of corresponding `TensorShape` objects.
  """
  shapes = []
  for t in tf.nest.flatten(tensors):
    shape = t.get_shape() if isinstance(t, tf.Tensor) else None
    if none_shapes:
      if shape:
        shapes.append(tf.TensorShape([None] * len(shape)))
      else:
        shapes.append(tf.TensorShape(None))
    else:
      shapes.append(tf.TensorShape(shape))

  return type(tensors)(tf.nest.pack_sequence_as(tensors, shapes))


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
    for k, v in output._asdict().items():
      if v is None:
        continue
      if k == 'done_hyps':
        v = tf.transpose(v)
      if k not in value_dict:
        value_dict[k] = []
      value_dict[k].append(tf.reshape(v, [source_batch, hyps_per_beam, -1]))

  # Concatenate the tensors along the 'num_hyps_per_beam' dimension.
  concatenated = {}
  for k, values in value_dict.items():
    if len(values) != len(beam_search_outputs):
      raise ValueError('Incomplete values for %s: %s' %
                       (k, beam_search_outputs))
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
  for k, v in concatenated.items():
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


class GreedySearchHelper(base_layer.BaseLayer):
  """Helper class for performing greedy decoding.

  The user of this helper class needs to implement three callbacks just as in a
  beam search decoder.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('target_sos_id', 1, 'Id of the start of sentence token.')
    p.Define('target_eos_id', 2, 'Id of the end of sentence token.')
    p.Define(
        'target_seq_len', 0, 'Maximum allowed target seq length. Note '
        'that decoding terminates if an end of sentence token '
        'is not emitted after target_seq_len decode steps.')
    p.name = 'greedy_search'
    return p

  def _GreedySearchStep(self, theta, encoder_outputs, cur_step, step_ids,
                        hyp_ids, hyp_lens, done_hyps, other_states,
                        pre_beam_search_step_callback,
                        post_beam_search_step_callback):
    """Extend greedy search hyps for one step.

    Args:
      theta: A `.NestedMap` object containing weights' values of the decoder
        layer and its children layers.
      encoder_outputs: A `.NestedMap` containing encoder outputs to be passed to
        the callbacks.
      cur_step: A scalar int tensor, the current time step, 0-based.
      step_ids: An int tensor of shape [num_hyps, 1]. The input ids to the
        current search step.
      hyp_ids: An int tensor of shape [num_hyps, tgt_seq_len].
      hyp_lens: Valid length of all the hyps. Tokens after eos ids are not
        counted.
      done_hyps: Whether or not a hyp has finished.
      other_states: A `.NestedMap` of other beam search states. This
        `.NestedMap` is managed and updated by the client. It is expected that
        each of its member tensors are of rank >= 1. t[i, ...] is the state of
        the i-th hyp at the beginning of this search step.
      pre_beam_search_step_callback: The `PreBeamSearchStepCallback` callback.
        See class header comments for more details.
      post_beam_search_step_callback: The `PostBeamSearchStepCallback` callback.
        See class header comments for more details.

    Returns:
      A tuple of following elements for the next greedy search step,
      (next step, new_step_ids, hyp_ids, hyp_lens, done_hyps, other_states)
    """
    p = self.params
    # Increment hyp_lens by 1 if the hyp is not finished yet.
    hyp_lens = hyp_lens + (1 - tf.cast(done_hyps, tf.int32))

    bs_results, new_other_states = pre_beam_search_step_callback(
        theta, encoder_outputs, step_ids, other_states, 1)  # num_hyps_per_beam
    new_step_ids = tf.math.argmax(bs_results.log_probs, 1)
    new_step_ids = tf.cast(new_step_ids, tf.int32)
    new_step_ids = tf.reshape(new_step_ids, tf.shape(step_ids))
    final_other_states = post_beam_search_step_callback(theta, encoder_outputs,
                                                        new_step_ids,
                                                        new_other_states)

    # Stash new_step_ids into the right slot.
    new_step_ids_1d = tf.reshape(new_step_ids, [-1])
    hyp_ids = inplace_ops.alias_inplace_update(hyp_ids, cur_step,
                                               new_step_ids_1d)
    # Update done_hyps if the current step_ids is the end of sequence token.
    done_hyps = tf.math.logical_or(done_hyps,
                                   tf.equal(new_step_ids_1d, p.target_eos_id))

    return (cur_step + 1, new_step_ids, hyp_ids, hyp_lens, done_hyps,
            final_other_states)

  def GreedySearchDecode(self,
                         theta,
                         encoder_outputs,
                         init_beam_search_state=None,
                         pre_beam_search_step_callback=None,
                         post_beam_search_step_callback=None,
                         max_steps=None):
    """Performs greedy-search based decoding.

    Args:
      theta: A NestedMap object containing weights' values of the decoder layer
        and its children layers.
      encoder_outputs: A NestedMap containing encoder outputs to be passed to
        the callbacks.
      init_beam_search_state: The `InitBeamSearchState` callback. Please refer
        to the class header comments for more details.
      pre_beam_search_step_callback: The `PreBeamSearchStepCallback` callback.
        Please refer to the class header comments for more details.
      post_beam_search_step_callback: The `PostBeamSearchStepCallback` callback.
        Please refer to the class header comments for more details.
      max_steps: maximum beam search steps. If None, use
        self.params.target_seq_len.

    Returns:
      A tuple (hyp_ids, hyp_lens, done_hyps). Note that num_hyps is same as
      src_batch_size.

        - hyp_ids: [num_hyps, max_step]. Hyps end with <eos> token if the <eos>
          token is encountered during search.
        - hyp_lens: [num_hyps].
        - done_hyps: [num_hyps], whether or not an eos is encountered.
    """
    p = self.params
    if max_steps is None:
      max_steps = p.target_seq_len

    initial_results, other_states = init_beam_search_state(
        theta,
        encoder_outputs,
        1  # num_hyps_per_beam
    )

    num_hyps = tf.shape(initial_results.log_probs)[0]

    if 'step_ids' in initial_results:
      # [num_hyps, 1]
      step_ids = tf.ensure_shape(initial_results.step_ids, [None, 1])
    else:
      step_ids = tf.fill([num_hyps, 1],
                         tf.constant(p.target_sos_id, dtype=tf.int32))

    cur_step = tf.constant(0, dtype=tf.int32)
    done_hyps = inplace_ops.empty(shape=[num_hyps], dtype=tf.bool, init=True,
                                  name='done_hyps')
    hyp_lens = inplace_ops.empty(shape=[num_hyps], dtype=tf.int32, init=True,
                                 name='hyp_lens')
    hyp_ids = inplace_ops.empty(
        shape=[max_steps, num_hyps], dtype=tf.int32, init=True,
        name='hyp_ids')

    def LoopContinue(cur_step, unused_step_ids, unused_hyp_ids, unused_hyp_lens,
                     done_hyps, unused_other_states_list):
      return tf.math.logical_and(cur_step < max_steps,
                                 tf.math.logical_not(tf.reduce_all(done_hyps)))

    def LoopBody(cur_step, step_ids, hyp_ids, hyp_lens, done_hyps,
                 other_states_list):
      (cur_step, new_step_ids, hyp_ids, hyp_lens, done_hyps,
       new_other_states) = self._GreedySearchStep(
           theta, encoder_outputs, cur_step,
           step_ids, hyp_ids, hyp_lens, done_hyps,
           other_states.Pack(other_states_list), pre_beam_search_step_callback,
           post_beam_search_step_callback)
      return (cur_step, new_step_ids, hyp_ids, hyp_lens, done_hyps,
              new_other_states.Flatten())

    flat_other_states = other_states.Flatten()
    _, _, final_hyp_ids, final_hyp_lens, final_done_hyps, _ = tf.while_loop(
        LoopContinue,
        LoopBody,
        loop_vars=(cur_step, step_ids, hyp_ids, hyp_lens, done_hyps,
                   flat_other_states),
        parallel_iterations=10,
        back_prop=False,
        swap_memory=False,
        shape_invariants=(tf.TensorShape(cur_step.get_shape()),
                          tf.TensorShape(step_ids.get_shape()),
                          tf.TensorShape(hyp_ids.get_shape()),
                          tf.TensorShape(hyp_lens.get_shape()),
                          tf.TensorShape(done_hyps.get_shape()),
                          _GetShapes(flat_other_states, none_shapes=True)))

    # transpose hyp_ids so it matches BeamSearchDecode's output
    final_hyp_ids = tf.transpose(final_hyp_ids)
    return final_hyp_ids, final_hyp_lens, final_done_hyps
