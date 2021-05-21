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
"""A library to sample target sequences given a decoder and decoder inputs.

The sampled sequences can be used for training, e.g., with scheduled sampling,
OCD, second-pass deliberation.
"""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import recurrent


def _ComputePaddings(ids, eos_id):
  is_eos = tf.cast(tf.equal(ids, eos_id), tf.int32)
  # eos_in_prefix[i, j] = any(ids[i, k] == eos_id for k in range(j))
  eos_in_prefix = tf.cumsum(is_eos, axis=-1, exclusive=True)
  return tf.where(
      tf.equal(eos_in_prefix, 0), tf.zeros_like(ids), tf.ones_like(ids))


class TargetSequenceSampler(base_layer.BaseLayer):
  """Helper class for sampling target sequences with a decoder from inputs."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('target_sos_id', 1, 'Id of the start of sentence token.')
    p.Define('target_eos_id', 2, 'Id of the end of sentence token.')
    p.Define('target_eoc_id', -1, 'Id of the end of chunk token.')
    p.Define('target_seq_len', 0, 'Maximum allowed target seq length.')
    p.Define('top_k', 0, 'If > 0, use top-k sampling.')
    p.Define('top_k_renormalize', True, 'Renormalize top-k probabilities.')
    p.Define(
        'temperature', 1., 'If > 1, a smoother distribution than logits; '
        'if < 1, a sharper distribution than logits. '
        'Must be > 0.')
    p.Define(
        'use_stop_fn', False, 'If True, use a stop_fn that causes the '
        'sampler to early terminate when all samples in the batch end with '
        'the target_eos_id token.')
    p.name = 'target_sequence_sampler'
    return p

  def Sample(self,
             decoder_theta,
             encoder_outputs,
             random_seed,
             init_state_callback,
             pre_step_callback,
             post_step_callback,
             init_step_ids=None):
    """Samples target sequences, one target sequence per source sequence.

    (Please see beam_search_helper.py for description of decoder callbacks.)

    Args:
      decoder_theta: A NestedMap object containing weights' values of the
        decoder layer and its children layers, to be passed to decoder
        callbacks.
      encoder_outputs: the outputs of the encoder, to be passed to callbacks.
      random_seed: a scalar int32 tensor representing the random seed.
      init_state_callback: decoder._InitBeamSearchStateCallback.
      pre_step_callback: decoder._PreBeamSearchStepCallback.
      post_step_callback: decoder._PostBeamSearchStepCallback.
      init_step_ids: [batch], optional init step ids, default to SOS.

    Returns:
      A NestedMap containing the following tensors

      - 'logits': [batch, max_target_length, vocab_size], representing the
        distribution from which target sequences are sampled.
      - 'ids': [batch, max_target_length] of int32, representing the target
        sequence ids, not including target_sos_id, but maybe ending with
        target_eos_id if end-of-sequence is reached before target_seq_len.
      - 'paddings': [batch, max_target_length] of 0/1, where 1 represents
        a padded timestep.
    """
    p = self.params
    assert p.temperature > 0
    assert p.top_k >= 0
    if getattr(encoder_outputs, 'segment_id', 1) is None:
      # Remove None values, which are not supported by recurrent.
      del encoder_outputs['segment_id']
    # init_state_callback may modify 'encoder_outputs', e.g., by inserting
    # 'packed_src'.
    bs_result, bs_state = init_state_callback(
        decoder_theta, encoder_outputs, num_hyps_per_beam=1)
    # 'recurrent_theta' represents all cross-timestep information used by the
    # recurrent loop below, including layer theta and encoder outputs.
    recurrent_theta = py_utils.NestedMap(
        random_seed=random_seed,
        encoder_outputs=encoder_outputs)
    batch = tf.shape(bs_result.log_probs)[0]
    recurrent_state0 = py_utils.NestedMap(
        timestep=tf.zeros(shape=[], dtype=tf.int32),
        logits=bs_result.log_probs,
        # Start with target_sos_id.
        ids=init_step_ids if init_step_ids is not None else tf.fill(
            [batch], tf.cast(p.target_sos_id, tf.int32)),
        bs_state=bs_state)
    inputs = py_utils.NestedMap(dummy=tf.zeros([p.target_seq_len, batch]))

    def Step(recurrent_theta, state0, inputs):
      """Computes one decoder step."""
      del inputs
      with tf.name_scope('single_sampler_step'):
        # Compute logits and states.
        bs_result, bs_state1 = pre_step_callback(
            decoder_theta,
            recurrent_theta.encoder_outputs,
            tf.expand_dims(state0.ids, 1),  # [batch, 1].
            state0.bs_state,
            num_hyps_per_beam=1)
        batch = tf.shape(bs_result.log_probs)[0]
        state1 = py_utils.NestedMap(timestep=state0.timestep + 1)
        state1.logits = bs_result.log_probs

        if p.top_k > 0:
          topk_logits, topk_ids = tf.math.top_k(state1.logits, k=p.top_k)
          sample_logits = tf.nn.log_softmax(
              topk_logits) if p.top_k_renormalize else topk_logits
        else:
          sample_logits = state1.logits

        # Sample ids from logits. [batch].
        ids = tf.reshape(
            tf.random.stateless_categorical(
                sample_logits / p.temperature,
                num_samples=1,
                seed=tf.stack([recurrent_theta.random_seed, state0.timestep]),
                dtype=state0.ids.dtype,
                name='sample_next_id'), [batch])
        state1.ids = tf.gather(
            topk_ids, ids, axis=1, batch_dims=1) if p.top_k > 0 else ids

        if 'is_last_chunk' in bs_result and p.target_eoc_id >= 0:
          state1.ids = tf.where(
              tf.math.logical_and(bs_result.is_last_chunk,
                                  tf.equal(state1.ids, p.target_eoc_id)),
              tf.fill(tf.shape(state1.ids), p.target_eos_id), state1.ids)
        state1.bs_state = post_step_callback(decoder_theta,
                                             recurrent_theta.encoder_outputs,
                                             state1.ids, bs_state1)
      return state1, py_utils.NestedMap()

    def StopFn(t, theta, state):
      del t, theta  # Unused: this stop function only uses the state ids.
      return tf.equal(state.ids, p.target_eos_id)

    if p.use_stop_fn:
      stop_fn = StopFn
    else:
      stop_fn = None

    accumulated_states, _ = recurrent.Recurrent(
        recurrent_theta,
        recurrent_state0,
        inputs,
        Step,
        stop_fn=stop_fn,
        allow_implicit_capture=True)
    result = py_utils.NestedMap(
        logits=tf.transpose(accumulated_states.logits, [1, 0, 2]),
        ids=tf.transpose(accumulated_states.ids))
    result.paddings = tf.cast(
        _ComputePaddings(result.ids, p.target_eos_id), result.logits.dtype)
    # Force ids to be eos_id if the timestep is padded.
    result.ids = tf.where(
        tf.equal(result.paddings, 0), result.ids,
        tf.fill(tf.shape(result.ids), p.target_eos_id))
    static_batch_size = bs_result.log_probs.shape[0]
    result.ids.set_shape([static_batch_size, p.target_seq_len])
    result.paddings.set_shape([static_batch_size, p.target_seq_len])
    return result
