# Lint as: python2, python3
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
"""Common decoder interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import beam_search_helper
from lingvo.core import py_utils
from lingvo.core import target_sequence_sampler

# metrics: Dict[Text, Tuple[float, float]] A dict of named metrics, which must
#   include 'loss'. The value of the dict is (metric_val, count), where
#   metric_val is the sum of the metric over all examples, and count is the
#   number of examples seen. The mean value of the metric is metric_val/count.
#   This is the first output of ComputeLoss.
# predictions: Union[Tensor, Dict[Text, Tensor], NestedMap] This is the output
#   of ComputePredictions.
# per_sequence: Dict[Text, Tensor] This is the second output of ComputeLoss.
DecoderOutput = collections.namedtuple(
    'DecoderOutput',
    ['metrics', 'predictions', 'per_sequence'],
)


class BaseDecoder(base_layer.BaseLayer):
  """Base class for all decoders."""

  @classmethod
  def Params(cls):
    p = super(BaseDecoder, cls).Params()
    p.Define(
        'packed_input', False, 'If True, decoder and all layers support '
        'multiple examples in a single sequence.')
    return p

  @classmethod
  def UpdateTargetVocabSize(cls, p, vocab_size, wpm_model=None):
    """Sets the vocab size and wpm model in the params.

    Args:
      p: model params.
      vocab_size: size of the vocabulary.
      wpm_model: file name prefix pointing to a wordpiece model.

    Returns:
      Model target vocabulary params updated with the vocab size and wpm model.
    """
    raise NotImplementedError('Abstract method')

  def FProp(self, theta, encoder_outputs, targets):
    """Decodes `targets` given encoded source.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder.
      targets: A NestedMap containing additional inputs to the decoder,
        such as the targets being predicted.

    Returns:
      A DecoderOutput namedtuple.
    """
    predictions = self.ComputePredictions(theta, encoder_outputs, targets)
    metrics, per_sequence = self.ComputeLoss(theta, predictions, targets)
    return DecoderOutput(
        metrics=metrics, predictions=predictions, per_sequence=per_sequence)

  def ComputePredictions(self, theta, encoder_outputs, targets):
    raise NotImplementedError('Abstract method: %s' % type(self))

  def ComputeLoss(self, theta, predictions, targets):
    raise NotImplementedError('Abstract method: %s' % type(self))


class BaseBeamSearchDecoder(BaseDecoder):
  """Decoder that does beam search."""

  @classmethod
  def Params(cls):
    p = super(BaseBeamSearchDecoder, cls).Params()
    p.Define('target_sos_id', 1, 'Id of the target sequence sos symbol.')
    p.Define('target_eos_id', 2, 'Id of the target sequence eos symbol.')
    # TODO(rpang): remove target_seq_len and use beam_search.target_seq_len
    # instead.
    p.Define('target_seq_len', 0, 'Target seq length.')
    p.Define('beam_search', beam_search_helper.BeamSearchHelper.Params(),
             'BeamSearchHelper params.')
    p.Define('greedy_search', beam_search_helper.GreedySearchHelper.Params(),
             'GreedySearchHelper params.')
    p.Define('target_sequence_sampler',
             target_sequence_sampler.TargetSequenceSampler.Params(),
             'TargetSequenceSampler params.')
    p.Define(
        'bias_only_if_consistent', True, 'BeamSearchBiased bias is only'
        'applied if the hypothesis has been consistent with targets so far.')
    return p

  @classmethod
  def UpdateTargetVocabSize(cls, p, vocab_size, wpm_model=None):
    """Sets the vocab size and wpm model in the params.

    Args:
      p: model params.
      vocab_size: size of the vocabulary.
      wpm_model: file name prefix pointing to a wordpiece model.

    Returns:
      Model target vocabulary params updated with the vocab size and wpm model.
    """
    raise NotImplementedError('Abstract method')

  @base_layer.initializer
  def __init__(self, params):
    super(BaseBeamSearchDecoder, self).__init__(params)
    p = self.params
    p.beam_search.target_seq_len = p.target_seq_len
    p.beam_search.target_sos_id = p.target_sos_id
    p.beam_search.target_eos_id = p.target_eos_id
    self.CreateChild('beam_search', p.beam_search)
    p.greedy_search.target_seq_len = p.target_seq_len
    p.greedy_search.target_sos_id = p.target_sos_id
    p.greedy_search.target_eos_id = p.target_eos_id
    self.CreateChild('greedy_search', p.greedy_search)
    p.target_sequence_sampler.target_seq_len = p.target_seq_len
    p.target_sequence_sampler.target_sos_id = p.target_sos_id
    p.target_sequence_sampler.target_eos_id = p.target_eos_id
    self.CreateChild('target_sequence_sampler', p.target_sequence_sampler)

  def AddExtraDecodingInfo(self, encoder_outputs, targets):
    """Adds extra decoding information to encoded_outputs.

    Args:
      encoder_outputs: a NestedMap computed by encoder.
      targets: a NestedMap containing target input fields.

    Returns:
      encoder_ouputs with extra information used for decoding.
    """
    return encoder_outputs

  def BeamSearchDecode(self, encoder_outputs, num_hyps_per_beam_override=0):
    """Performs beam search based decoding.

    Args:
      encoder_outputs: the outputs of the encoder.
      num_hyps_per_beam_override: If set to a value <= 0, this parameter is
        ignored. If set to a value > 0, then this value will be used to override
        p.num_hyps_per_beam.

    Returns:
      `.BeamSearchDecodeOutput`, A namedtuple whose elements are tensors.
    """
    return self.BeamSearchDecodeWithTheta(self.theta, encoder_outputs,
                                          num_hyps_per_beam_override)

  def BeamSearchDecodeWithTheta(self,
                                theta,
                                encoder_outputs,
                                num_hyps_per_beam_override=0):
    return self.beam_search.BeamSearchDecode(theta, encoder_outputs,
                                             num_hyps_per_beam_override,
                                             self._InitBeamSearchStateCallback,
                                             self._PreBeamSearchStepCallback,
                                             self._PostBeamSearchStepCallback)

  def GreedySearchDecode(self, encoder_outputs):
    """Performs beam search based decoding.

    Args:
      encoder_outputs: the outputs of the encoder.

    Returns:
      greedy search decode output.
    """
    return self.GreedySearchDecodeWithTheta(self.theta, encoder_outputs)

  def GreedySearchDecodeWithTheta(self, theta, encoder_outputs):
    return self.greedy_search.GreedySearchDecode(
        theta, encoder_outputs,
        self._InitBeamSearchStateCallback,
        self._PreBeamSearchStepCallback,
        self._PostBeamSearchStepCallback)

  def SampleTargetSequences(self, theta, encoder_outputs, random_seed):
    """Performs target sequence sampling.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      encoder_outputs: a NestedMap computed by encoder.
      random_seed: a scalar int32 tensor representing the random seed.

    Returns:
      A NestedMap containing the following tensors

      - 'ids': [batch, max_target_length] of int32, representing the target
        sequence ids, not including target_sos_id, but maybe ending with
        target_eos_id if target_eos_id is sampled.
      - 'paddings': [batch, max_target_length] of 0/1, where 1 represents
        a padded timestep.
    """
    return self.target_sequence_sampler.Sample(
        theta, encoder_outputs, random_seed, self._InitBeamSearchStateCallback,
        self._PreBeamSearchStepCallback, self._PostBeamSearchStepCallback)

  def BeamSearchDecodeBiased(self,
                             encoder_outputs,
                             num_hyps_per_beam_override=0):
    """Performs beam-search decoding while biasing towards provided targets.

    Args:
      encoder_outputs: a NestedMap computed by encoder. Must include `targets`,
        which is used to bias beam search.
      num_hyps_per_beam_override: If set to a value <= 0, this parameter is
        ignored. If set to a value > 0, then this value will be used to override
        `p.num_hyps_per_beam`.

    Returns:
      BeamSearchDecodeOutput, a namedtuple containing the decode results.
    """
    p = self.params

    targets = encoder_outputs.targets
    targets.weights *= (1.0 - targets.paddings)

    def PadToTargetSeqLen(tensor, constant):
      length = tf.shape(tensor)[1]
      pad = tf.maximum(0, p.beam_search.target_seq_len - length)
      return tf.pad(tensor, [[0, 0], [0, pad]], constant_values=constant)

    targets.labels = PadToTargetSeqLen(targets.labels, 0)
    targets.weights = PadToTargetSeqLen(targets.weights, 0)

    def InitBeamSearchStateCallback(theta, encoder_outputs, num_hyps_per_beam):
      """Wrapper for adding bias to _InitBeamSearchStateCallback.

      Exapnds state to track consistency of hypothesis with provided target.

      Args:
        theta: A NestedMap object containing weights' values of this layer and
          its children layers.
        encoder_outputs: A NestedMap computed by encoder.
        num_hyps_per_beam: An int, number hyps to keep for source sentence.

      Returns:
        initial_results: a `.NestedMap` of initial results.
        states: a `.NestedMap` of initial model states that the client
          would like to keep track of for each hyp. The states relevant here
          are:
          time_step: A scalar indicating current step (=0 for initial state) of
            decoder.  Must be provided and maintained by super.
          consistent: A boolean tensor of shape [tgt_batch, 1] which tracks
              whether each hypothesis has exactly matched
              encoder_outputs.targets
              so far.
      """
      initial_results, states = self._InitBeamSearchStateCallback(
          theta, encoder_outputs, num_hyps_per_beam)
      assert hasattr(states, 'time_step')
      num_hyps = tf.shape(encoder_outputs.padding)[1] * num_hyps_per_beam
      # states.consistent is initially all True
      states.consistent = tf.ones([
          num_hyps,
      ], dtype=tf.bool)
      return initial_results, states

    def PreBeamSearchStepCallback(theta, encoder_outputs, step_ids, states,
                                  num_hyps_per_beam, *args, **kwargs):
      """Wrapper for adding bias to _PreBeamSearchStateCallback.

      Biases results.log_probs towards provided encoder_outputs.targets.

      Args:
        theta: a NestedMap of parameters.
        encoder_outputs: a NestedMap computed by encoder.
        step_ids: A tensor of shape [tgt_batch, 1].
        states: A `.NestedMap` of tensors representing states that the clients
          would like to keep track of for each of the active hyps.
        num_hyps_per_beam: Beam size.
        *args: additional arguments to _PreBeamSearchStepCallback.
        **kwargs: additional arguments to _PreBeamSearchStepCallback.

      Returns:
        A tuple (results, out_states).
        results: A `.NestedMap` of beam search results.
          atten_probs:
            The updated attention probs, of shape [tgt_batch, src_len].
          log_probs:
            Log prob for each of the tokens in the target vocab. This is of
            shape
            [tgt_batch, vocab_size].
        out_states: a `.NestedMap` The updated states. The states relevant here
          are:
          time_step: A scalar indicating current step of decoder.  Must be
            provided and maintained by subclass.
          consistent: A boolean vector of shape [tgt_batch, ] which tracks
              whether each hypothesis has exactly matched
              encoder_outputs.targets
              so far.
      """
      p = self.params
      time_step = states.time_step
      bs_results, out_states = self._PreBeamSearchStepCallback(
          theta, encoder_outputs, step_ids, states, num_hyps_per_beam, *args,
          **kwargs)
      labels = encoder_outputs.targets.labels
      weights = encoder_outputs.targets.weights

      def ApplyBias():
        """Bias and update log_probs and consistent."""

        def TileForBeamAndFlatten(tensor):
          tensor = tf.reshape(tensor, [1, -1])  # [1, src_batch]
          tensor = tf.tile(
              tensor, [num_hyps_per_beam, 1])  # [num_hyps_per_beam, src_batch]
          tgt_batch = tf.shape(step_ids)[0]  # num_hyps_per_beam*src_batch
          return tf.reshape(tensor, [tgt_batch])

        # Consistent if step_ids == labels from previous step
        # TODO(navari): Consider updating consistent only if weights > 0. Then
        # re-evaluate the need for bias_only_if_consistent=True.
        # Note that prev_label is incorrrect for step 0 but is overridden later
        prev_label = TileForBeamAndFlatten(
            tf.gather(labels, tf.maximum(time_step - 1, 0), axis=1))
        is_step0 = tf.equal(time_step, 0)
        local_consistence = tf.math.logical_or(
            is_step0, tf.equal(prev_label, tf.squeeze(step_ids, 1)))
        consistent = tf.math.logical_and(states.consistent, local_consistence)

        # get label, weight slices corresponding to current time_step
        label = TileForBeamAndFlatten(tf.gather(labels, time_step, axis=1))
        weight = TileForBeamAndFlatten(tf.gather(weights, time_step, axis=1))
        if p.bias_only_if_consistent:
          weight = weight * tf.cast(consistent, p.dtype)

        # convert from dense label to sparse label probs
        vocab_size = tf.shape(bs_results.log_probs)[1]
        uncertainty = tf.constant(
            1e-10, p.dtype)  # avoid 0 probs which may cause issues with log
        label_probs = tf.one_hot(
            label,
            vocab_size,
            on_value=1 - uncertainty,
            off_value=uncertainty / tf.cast(vocab_size - 1, p.dtype),
            dtype=p.dtype)  # [tgt_batch, vocab_size]
        pred_probs = tf.exp(bs_results.log_probs)

        # interpolate predicted probs and label probs
        weight = tf.expand_dims(weight, 1)
        probs = py_utils.with_dependencies([
            py_utils.assert_less_equal(weight, 1.),
            py_utils.assert_greater_equal(weight, 0.)
        ], (1.0 - weight) * pred_probs + weight * label_probs)
        return tf.math.log(probs), consistent

      def NoApplyBias():
        """No-op. Return original log_probs and consistent."""
        return bs_results.log_probs, states.consistent

      log_probs, consistent = tf.cond(
          tf.reduce_all(tf.equal(weights, 0.0)), NoApplyBias, ApplyBias)
      bs_results.log_probs = log_probs
      out_states.consistent = consistent

      return bs_results, out_states

    return self.beam_search.BeamSearchDecode(self.theta, encoder_outputs,
                                             num_hyps_per_beam_override,
                                             InitBeamSearchStateCallback,
                                             PreBeamSearchStepCallback,
                                             self._PostBeamSearchStepCallback)
