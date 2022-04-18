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
    p = super().Params()
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


# A finite value used to represent a negative infinity log-probability.
LARGE_NEGATIVE_NUMBER = -1e9


def _KeepTopP(sorted_log_probs, p):
  """Keeps the top-p probability mass of `sorted_log_probs`.

  For each row, elements that are not included in the first `p` probability mass
  are set to `LARGE_NEGATIVE_NUMBER`. The first element is always kept as-is.

  Args:
    sorted_log_probs: A float tensor of shape [batch, k] that represents
      log-probabilities sorted in descending order. The probabilities do not
      need to sum to 1.
    p: A float tensor of shape [batch] that represents a probability threshold
      for each batch item.

  Returns:
    A tensor like `sorted_log_probs` where elements outside the top-p
    probability mass are set to `LARGE_NEGATIVE_NUMBER`.
  """
  sorted_cum_probs = tf.math.cumsum(
      tf.exp(sorted_log_probs), exclusive=True, axis=-1)
  mask = tf.less(sorted_cum_probs, tf.expand_dims(p, axis=1))
  # Set mask[:, 0] = True to always keep the first element.
  batch_size = tf.shape(mask)[0]
  true = tf.ones([batch_size, 1], dtype=tf.bool)
  mask = tf.concat([true, mask[:, 1:]], axis=1)
  filtered_sorted_log_probs = tf.where(
      mask, sorted_log_probs,
      tf.fill(
          tf.shape(sorted_log_probs),
          tf.constant(LARGE_NEGATIVE_NUMBER, dtype=sorted_log_probs.dtype)))
  return filtered_sorted_log_probs


def _BatchScatter(default_tensor, indices, values):
  """Performs tf.tensor_scatter_nd_update for each batch item.

  Args:
    default_tensor: A float tensor of shape [batch, vocab] that contains the
      default values.
    indices: An int tensor of shape [batch, k] that represents the k indices of
      `default_tensor` to update.
    values: A float tensor of shape [batch, k] that represents the value to
      replace with for each corresponding element of `indices`.

  Returns:
    A tensor like `default_tensor` where the (i, indices[i][j]) element has been
    replaced with values[i][j].
  """
  batch_size = tf.shape(default_tensor)[0]
  # Prepend batch indices to `indices`.
  batch_indices = tf.range(batch_size, dtype=indices.dtype)
  batch_indices = tf.expand_dims(batch_indices, 1)
  batch_indices = tf.broadcast_to(batch_indices, tf.shape(indices))
  batch_indices = tf.stack([batch_indices, indices], axis=2)

  return tf.tensor_scatter_nd_update(default_tensor, batch_indices, values)


def _BatchLookup(keys, table_keys, table_values):
  """Looks up `keys` in a given key-value table.

  Args:
    keys: An int tensor of shape [batch, 1] that represents keys to look up.
    table_keys: An int tensor of shape [batch, k] that represents table keys.
    table_values: A float tensor of shape [batch, k] that represents table
      values.

  Returns:
    A float tensor of shape [batch, 1] that holds values from `table_values`
    whose corresponding elements of `table_keys` match `keys`.
  """
  match_indices = tf.math.argmax(tf.math.equal(keys, table_keys), axis=1)
  return tf.expand_dims(
      tf.gather_nd(
          table_values, tf.expand_dims(match_indices, 1), batch_dims=1),
      axis=1)


def _BatchSampleGumbel(batch_seed, time_step, src_ids, src_paddings, shape,
                       dtype):
  """Samples (standard) Gumbel noises of a given shape for each batch item.

  The random seed for the i-th batch item is determined by batch_seed[i],
  time_step, and the sum of non-padding elements of src_ids[i].

  Args:
    batch_seed: An int tensor of shape [batch] that holds a seed for each batch
      item.
    time_step: An int tensor used as a secondary seed.
    src_ids: An int tensor of shape [batch, src_seq] that represents source IDs.
      Used for turning the random seed into a function of source IDs.
    src_paddings: A 0/1 float tensor of shape [batch, src_seq] where 1 means
      that the corresponding element of src_ids is a padding.
    shape: A shape of the Gumbel noises to sample.
    dtype: A type of the Gumbel noises.

  Returns:
    A `dtype` tensor of shape [batch, ...] that holds Gumbel noises.
  """
  # Turn batch_seed into a function of the source IDs by adding the sum of the
  # source IDs. Without doing this, the same pattern of random noises would be
  # used no matter what the source sequence is, resulting in a systematic bias
  # among the output for a given seed value.
  # Mask padding IDs by 0.
  src_ids = src_ids * tf.cast(1.0 - src_paddings, dtype=src_ids.dtype)
  # Compute the sum of source IDs.
  src_ids_sum = tf.math.reduce_sum(src_ids, axis=1)  # shape: [src_batch]
  batch_seed_plus_src_ids_sum = batch_seed + src_ids_sum

  def SampleForBeam(seed):
    return -tf.math.log(-tf.math.log(
        tf.random.stateless_uniform(
            shape=shape, dtype=dtype, seed=tf.stack([seed, time_step]))))

  return tf.map_fn(SampleForBeam, batch_seed_plus_src_ids_sum, dtype=dtype)


def _SampleGumbelWithMax(phi, target_max, batch_seed, time_step, src_ids,
                         src_paddings):
  """Samples a set of Gumbel noises with a specified maximum value.

  A set of values are sampled from Gumbel distributions with location parameters
  `phi` under the condition that their maximum is equal to `target_max`.

  The numerical stable implementation from Appendix B.3 of
  https://arxiv.org/pdf/1903.06059.pdf is used.

  Args:
    phi: A float tensor of shape [tgt_batch, k] thtat represents location
      parameters of Gumbel distributions.
    target_max: A float tensor of shape [tgt_batch, 1] that represents the
      target max values.
    batch_seed: An int tensor of shape [src_batch] that holds a seed value for
      each batch item. src_batch must be equal to tgt_batch / num_hyps_per_beam.
      The same seed is used within each consecutive num_hyps_per_beam items
      along the tgt_batch axis.
    time_step: A float tensor used as a secondary seed.
    src_ids: An int tensor of shape [src_batch, src_seq] that represents source
      IDs. Used for turning the random seed into a function of source IDs.
    src_paddings: A 0/1 float tensor of shape [src_batch, src_seq] where 1 means
      that the corresponding element of src_ids is a padding.

  Returns:
    A float tensor like `phi` where their maximum values along the second axis
    is (almost) equal to `target_max`.
  """
  dtype = phi.dtype
  tgt_batch = tf.shape(phi)[0]
  k = tf.shape(phi)[1]
  src_batch = tf.shape(batch_seed)[0]
  num_hyps_per_beam = tgt_batch // src_batch

  # Sample noises from Gumbel distributions with location parameters `phi`.
  # shape: [src_batch, num_hyps_per_beam, k]
  gumbel_noises = _BatchSampleGumbel(batch_seed, time_step, src_ids,
                                     src_paddings, [num_hyps_per_beam, k],
                                     dtype)
  # shape: [num_hyps_per_beam, src_batch, k]
  gumbel_noises = tf.transpose(gumbel_noises, perm=[1, 0, 2])
  # shape: [tgt_batch, k]
  gumbel_noises = tf.reshape(gumbel_noises, tf.shape(phi))
  # shape: [tgt_batch, k]
  g_phi = phi + gumbel_noises

  # shape: [tgt_batch, 1]
  z = tf.reduce_max(g_phi, axis=1, keepdims=True)

  # Equation (23).
  # shape: [tgt_batch, k]
  v = target_max - g_phi + tf.math.log1p(
      # Without taking max, sometimes the result of log1p would become NaN on
      # TPU.
      tf.maximum(-tf.exp(g_phi - z), tf.constant(-1., dtype=dtype)))

  # Equation (24).
  return target_max - tf.nn.relu(v) - tf.math.log1p(tf.exp(-tf.abs(v)))


class BaseBeamSearchDecoder(BaseDecoder):
  """Decoder that does beam search."""

  @classmethod
  def Params(cls):
    p = super().Params()
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
    p.Define(
        'stochastic_beam_search_top_k', 8,
        'As a performance optimization, the stochastic beam search '
        'implementation first performs top-k filtering of the '
        'log-probabilities so that only k values need to be maintained for '
        'each hypothesis.')
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

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if hasattr(p.beam_search, 'target_seq_len'):
      p.beam_search.target_seq_len = p.target_seq_len
    if hasattr(p.beam_search, 'target_sos_id'):
      p.beam_search.target_sos_id = p.target_sos_id
    if hasattr(p.beam_search, 'target_eos_id'):
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

  def _PostprocessSample(self, sample, is_tpu):
    """Add topk_hyps, topk_ids, topk_lens, topk_scores tensors to `sample`.

    These features are required by `.BeamSearchDecodeOutput`.

    Args:
      sample: a NestedMap with `id`, `paddings`, and `logits` fields.
      is_tpu: whether inference is being run on TPU.

    Returns:
      sample with additional feature that matches `.BeamSearchDecodeOutput`
      requirements. `topk_hyps` is empty.
    """
    p = self.params
    bs = tf.shape(sample.ids)[0]
    num_hyps_per_beam = p.target_sequence_sampler.num_hyps_per_beam
    vocab_size = tf.shape(sample.logits)[2]

    # tf.string is not supported on tpu.
    sample.topk_hyps = tf.zeros([bs], dtype=tf.int32 if is_tpu else tf.string)
    sample.topk_hyps = tf.reshape(sample.topk_hyps, [-1, num_hyps_per_beam])

    sample.topk_ids = sample.ids
    weights = 1 - sample.paddings
    sample.topk_lens = tf.cast(tf.reduce_sum(weights, axis=1), dtype=tf.int32)
    # Computing the hypothesis scores based on the returned ids
    mask = tf.one_hot(
        sample.topk_ids, depth=vocab_size, axis=-1, dtype=sample.logits.dtype)
    token_log_probs = tf.einsum('ijk,ijk->ij', tf.nn.log_softmax(sample.logits),
                                mask)
    sample.topk_scores = tf.reduce_sum(token_log_probs * weights, axis=1)
    # At this point batch dimension is (batch_size*num_hyps_per_beam),
    # interleaved as [num_hyps_per_beam, batch_size].
    # This does not match the order expected by beam search post-processing.
    # Must transpose to [batch_size, num_hyps_per_beam] and flatten back.
    max_len = tf.shape(sample.topk_ids)[1]
    sample.topk_ids = tf.reshape(sample.topk_ids,
                                 [num_hyps_per_beam, -1, max_len])
    sample.topk_ids = tf.transpose(sample.topk_ids, perm=[1, 0, 2])
    sample.topk_ids = tf.reshape(sample.topk_ids, [bs, max_len])

    # The same for topk_lens and topk_scores
    sample.topk_lens = tf.reshape(sample.topk_lens, [num_hyps_per_beam, -1])
    sample.topk_lens = tf.transpose(sample.topk_lens, [1, 0])
    sample.topk_lens = tf.reshape(sample.topk_lens, [-1])

    sample.topk_scores = tf.reshape(sample.topk_scores, [num_hyps_per_beam, -1])
    sample.topk_scores = tf.transpose(sample.topk_scores, [1, 0])
    sample.topk_scores = tf.reshape(sample.topk_scores, [-1])
    return sample

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
    return self.StochasticBeamSearchDecodeBiased(
        encoder_outputs,
        biased=True,
        stochastic=False,
        num_hyps_per_beam_override=num_hyps_per_beam_override)

  def StochasticBeamSearchDecodeBiased(self,
                                       encoder_outputs,
                                       biased,
                                       stochastic,
                                       num_hyps_per_beam_override=0):
    """Performs beam search based decoding with optional advanced features.

    If `biased` is true, the target biasing feature is added. `encoder_outputs`
    must include the following auxiliary inputs:

    - targets.labels: An int tensor of shape [batch, seq] that represents target
      labels to bias beam search towards.
    - targets.paddings: A 0/1 float tensor of shape [batch, seq] where 1 means
      that the corresponding element of targets.labels is a padding.
    - targets.weights: A float tensor of shape [batch, seq] that represents
      biasing weights. 1.0 means forced-decoding.

    If `stochastic` is true, the stochastic beam search feature
    (https://arxiv.org/pdf/1903.06059.pdf) is added. Also, top-p filtering (i.e.
    sampling only from the top-p probability mass of the token distribution) is
    performed to ensure the quality of samples. Note that there are slight
    differences from the implementation in the original paper, e.g., length
    normalization and coverage penalty are applied to the perturbed
    probabilities. `encoder_outputs` must include the following auxiliary
    inputs:

    - stochastic_beam_search.top_p_threshold: A float tensor of shape [batch]
      that represents the thresholds of top-p filtering. Must satisfy
      0 < top_p_threshold <= 1. If the value is low, the quality of samples will
      be high but the diversity will be low. If the value is high, the quality
      of samples will be low but the diversity will be high. Stochastic beam
      search is performed only if top_p_threshold > 0 for some batch items.
    - stochastic_beam_search.seed: An int tensor of shape [batch] the represents
      the random seeds. If the seeds are the same, the same samples are drawn.
    - stochastic_beam_search.src_ids: An int tensor of shape [batch, src_seq]
      that represents source IDs. Used for turning the random seed into a
      function of source IDs.
    - stochastic_beam_search.src_paddings: A 0/1 float tensor of shape [batch,
      src_seq] where 1 means that the corresponding element of
      stochastic_beam_search.src_ids is a padding.

    Args:
      encoder_outputs: a NestedMap computed by encoder.
      biased: If true, add the target decoding feature.
      stochastic: If true, add the stochastic beam search feature.
      num_hyps_per_beam_override: If set to a value <= 0, this parameter is
        ignored. If set to a value > 0, then this value will be used to override
        `p.num_hyps_per_beam`.

    Returns:
      BeamSearchDecodeOutput, a namedtuple containing the decode results.
    """
    p = self.params

    if biased:
      targets = encoder_outputs.targets
      targets.weights *= (1.0 - targets.paddings)

      def PadToTargetSeqLen(tensor, constant):
        length = tf.shape(tensor)[1]
        pad = tf.maximum(0, p.beam_search.target_seq_len - length)
        return tf.pad(tensor, [[0, 0], [0, pad]], constant_values=constant)

      targets.labels = PadToTargetSeqLen(targets.labels, 0)
      targets.weights = PadToTargetSeqLen(targets.weights, 0)

    if stochastic:
      # Determine whether to perform stochastic beam search.
      stochastic_beam_search = encoder_outputs.stochastic_beam_search
      stochastic_beam_search.enable = tf.reduce_any(
          tf.greater(stochastic_beam_search.top_p_threshold, 0.0))

    return self.beam_search.BeamSearchDecode(
        self.theta, encoder_outputs, num_hyps_per_beam_override,
        self._WrapInitBeamSearchStateCallback(biased, stochastic),
        self._WrapPreBeamSearchStepCallback(biased, stochastic),
        self._WrapPostBeamSearchStepCallback(stochastic))

  def _WrapInitBeamSearchStateCallback(self, biased, stochastic):
    """Creates a new callback that wraps self._InitBeamSearchStateCallback.

    Used by the implementation of StochasticBeamSearchDecodeBiased().

    If `biased` is True, attaches the following field to the states:

    - consistent: A boolean tensor of shape [tgt_batch, 1] which tracks whether
      each hypothesis has exactly matched encoder_outputs.targets so far.

    If `stochastic_beam_search` is True, attaches the following fields to the
    states:

    - cumulative_log_probs: A float tensor of shape [tgt_batch, k] that
      represents the cumulative (unperturbed) log-probabilities.
    - perturbed_cumulative_log_probs: A float tensor of shape [tgt_batch, k]
      that represents the perturbed counterpart of `cumulative_log_probs`.
      Initialized with 0 (see the footnote 2 of the original paper), which
      ensures that perturbed_cumulative_log_probs is always non-positive, which
      is desirable since it gets divided by the length normalization at the end
      of the beam search.
    - tmp_states: A NestedMap that holds internal info passed from
      PreBeamSearchStepCallback to PostBeamSearchStepCallback.

    It is assumed that the wrapped callback provides states.time_step which is
    scalar indicating current step (=0 for initial state) of decoder.

    Args:
      biased: If true, add the target decoding feature.
      stochastic: If true, add the stochastic beam search feature.

    Returns:
      A new function that has the same interface as
      self._InitBeamSearchStateCallback.
    """
    k = self.params.stochastic_beam_search_top_k

    def Callback(theta, encoder_outputs, num_hyps_per_beam):
      initial_results, states = self._InitBeamSearchStateCallback(
          theta, encoder_outputs, num_hyps_per_beam)
      assert hasattr(states, 'time_step')
      if tf.is_tensor(encoder_outputs.padding):
        batch_size = tf.shape(encoder_outputs.padding)[1]
      else:  # Required for multisource models.
        batch_size = tf.shape(list(encoder_outputs.padding.values())[0])[1]
      num_hyps = batch_size * num_hyps_per_beam

      if biased:
        # states.consistent is initially all True
        states.consistent = tf.ones([
            num_hyps,
        ], dtype=tf.bool)

      if stochastic:
        dtype = py_utils.FPropDtype(self.params)
        states.cumulative_log_probs = tf.zeros([num_hyps, 1], dtype=dtype)
        states.perturbed_cumulative_log_probs = tf.zeros([num_hyps, 1],
                                                         dtype=dtype)
        # Temporary tensors that store information passed from
        # PreBeamSearchStepCallback to PostBeamSearchStepCallback. These are
        # used for updating states.cumulative_log_probs and
        # states.perturbed_cumulative_log_probs for the next step, which
        # requires the knowledge of the chosen IDs, which only becomes available
        # after PreBeamSearchStepCallback.
        states.tmp_states = py_utils.NestedMap(
            # Top-k (non-perturbed) log-probs. Used for updating
            # `cumulative_log_probs` in PostBeamSearchStepCallback.
            top_k_log_probs=tf.zeros([num_hyps, k], dtype=dtype),
            # Vocab ID of each item of `top_k_log_probs`.
            top_k_ids=tf.zeros([num_hyps, k], dtype=tf.int32),
            # Perturbed cumulative log-probs of the top-k IDs. Used for updating
            # `perturbed_cumulative_log_probs` in PostBeamSearchStepCallback.
            new_perturbed_cumulative_log_probs=tf.zeros([num_hyps, k],
                                                        dtype=dtype),
        )

      return initial_results, states

    return Callback

  def _WrapPreBeamSearchStepCallback(self, biased, stochastic):
    """Creates a new callback that wraps self._PreBeamSearchStepCallback.

    Used by the implementation of StochasticBeamSearchDecodeBiased().

    Modifies results.log_probs as follows:

    1. If `biased` is True, biases results.log_probs towards provided
       encoder_outputs.targets.
    2. If `stochastic_beam_search` is True, perturbs results.log_probs by Gumbel
       noises.

    It is assumed that the wrapped callback maintains states.time_step which is
    scalar indicating current step (=0 for initial state) of decoder.

    Args:
      biased: If true, add the target decoding feature.
      stochastic: If true, add the stochastic beam search feature.

    Returns:
      A new function that has the same interface as
      self._PreBeamSearchStepCallback.
    """
    k = self.params.stochastic_beam_search_top_k

    def Callback(theta, encoder_outputs, step_ids, states, num_hyps_per_beam,
                 cur_step, *args, **kwargs):
      p = self.params
      time_step = states.time_step
      bs_results, out_states = self._PreBeamSearchStepCallback(
          theta, encoder_outputs, step_ids, states, num_hyps_per_beam, cur_step,
          *args, **kwargs)

      def TileForBeamAndFlatten(tensor):
        tensor = tf.reshape(tensor, [1, -1])  # [1, src_batch]
        tensor = tf.tile(
            tensor, [num_hyps_per_beam, 1])  # [num_hyps_per_beam, src_batch]
        tgt_batch = tf.shape(step_ids)[0]  # num_hyps_per_beam*src_batch
        return tf.reshape(tensor, [tgt_batch])

      if biased:
        labels = encoder_outputs.targets.labels
        weights = encoder_outputs.targets.weights

        def ApplyBias():
          """Bias and update log_probs and consistent."""

          # Consistent if step_ids == labels from previous step
          # TODO(navari): Consider updating consistent only if weights > 0. Then
          # re-evaluate the need for bias_only_if_consistent=True.
          # Note that prev_label is incorrrect for step 0 but is overridden
          # later
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
            weight = weight * tf.cast(consistent, py_utils.FPropDtype(p))

          # convert from dense label to sparse label probs
          vocab_size = tf.shape(bs_results.log_probs)[1]
          label_probs = tf.one_hot(
              label, vocab_size,
              dtype=py_utils.FPropDtype(p))  # [tgt_batch, vocab_size]
          pred_probs = tf.exp(bs_results.log_probs)

          # interpolate predicted probs and label probs
          weight = tf.expand_dims(weight, 1)
          probs = py_utils.with_dependencies([
              py_utils.assert_less_equal(weight, 1.),
              py_utils.assert_greater_equal(weight, 0.)
          ], (1.0 - weight) * pred_probs + weight * label_probs)
          # Ensure that tf.math.log is applied to positive values.
          probs = tf.maximum(probs, tf.constant(1e-12, dtype=probs.dtype))
          return tf.math.log(probs), consistent

        def NoApplyBias():
          """No-op. Return original log_probs and consistent."""
          return bs_results.log_probs, states.consistent

        log_probs, consistent = tf.cond(
            tf.reduce_all(tf.equal(weights, 0.0)), NoApplyBias, ApplyBias)
        bs_results.log_probs = log_probs
        out_states.consistent = consistent

      if stochastic:
        log_probs = bs_results.log_probs

        def PerturbedLogProbs():
          # STEP 1: Perform top-k filtering. This is done as a performance
          # optimization of avoiding sorting the entire `log_probs`, which is
          # prohibitively slow.
          top_k = tf.math.top_k(log_probs, k, sorted=True)
          # shape: [tgt_batch, k]
          top_k_log_probs = top_k.values
          # shape: [tgt_batch, k]
          top_k_ids = top_k.indices

          # STEP 2: Perform top-p filtering.
          # shape: [tgt_batch]
          top_p_threshold = encoder_outputs.stochastic_beam_search.top_p_threshold
          top_p_threshold = tf.clip_by_value(top_p_threshold, 0., 1.)
          top_p_threshold = TileForBeamAndFlatten(top_p_threshold)
          # shape: [tgt_batch, k]
          filtered_top_k_log_probs = _KeepTopP(top_k_log_probs, top_p_threshold)

          # STEP 3: Perturb cumulative log-probs.
          # shape: [tgt_batch, 1]
          last_cumulative_log_probs = states.cumulative_log_probs
          # shape: [tgt_batch, 1]
          last_perturbed_cumulative_log_probs = states.perturbed_cumulative_log_probs
          # Compute cumulative log-probs of the current step.
          # shape: [tgt_batch, k]
          cumulative_log_probs = (
              last_cumulative_log_probs + filtered_top_k_log_probs)
          # Perturb cumulative log-probs by Gumbel noises under the condition
          # that the max of the new perturbed log-probs is equal to
          # perturbed_cumulative_log_probs of the previous step.
          # shape: [tgt_batch, k]
          new_perturbed_cumulative_log_probs = _SampleGumbelWithMax(
              cumulative_log_probs, last_perturbed_cumulative_log_probs,
              encoder_outputs.stochastic_beam_search.seed, time_step,
              encoder_outputs.stochastic_beam_search.src_ids,
              encoder_outputs.stochastic_beam_search.src_paddings)

          # STEP 4: Compute updated log_probs. This step is necessary because
          # the output of PreBeamSearchStepCallback must be "per-step"
          # log-probs, whereas so far "cumulative" log-probs have been computed.
          # shape: [tgt_batch, k]
          updated_top_k_log_probs = (
              new_perturbed_cumulative_log_probs -
              last_perturbed_cumulative_log_probs)
          # Convert to the shape [tgt_batch, vocab_size].
          updated_log_probs = tf.fill(
              tf.shape(log_probs),
              tf.constant(LARGE_NEGATIVE_NUMBER, dtype=log_probs.dtype))
          updated_log_probs = _BatchScatter(updated_log_probs, top_k_ids,
                                            updated_top_k_log_probs)

          return (
              updated_log_probs,
              py_utils.NestedMap(
                  new_perturbed_cumulative_log_probs=new_perturbed_cumulative_log_probs,
                  top_k_log_probs=top_k_log_probs,
                  top_k_ids=top_k_ids,
              ))

        (bs_results.log_probs, out_states.tmp_states) = tf.cond(
            encoder_outputs.stochastic_beam_search.enable,
            PerturbedLogProbs,
            # No-op.
            lambda: (bs_results.log_probs, states.tmp_states))
        # These states are not updated here but will be updated in
        # PostBeamSearchStepCallback since doing so requires the knowledge of
        # the next step IDs.
        out_states.cumulative_log_probs = states.cumulative_log_probs
        out_states.perturbed_cumulative_log_probs = states.perturbed_cumulative_log_probs

      return bs_results, out_states

    return Callback

  def _WrapPostBeamSearchStepCallback(self, stochastic):
    """Creates a new callback that wraps self._PostBeamSearchStateCallback.

    Used by the implementation of StochasticBeamSearchDecodeBiased().

    Args:
      stochastic: If true, add the stochastic beam search feature.

    Returns:
      A new function that has the same interface as
      self._PostBeamSearchStepCallback.
    """

    def Callback(theta, encoder_outputs, new_step_ids, other_states):
      final_states = self._PostBeamSearchStepCallback(theta, encoder_outputs,
                                                      new_step_ids,
                                                      other_states)

      if stochastic:
        # Update perturbed_cumulative_log_probs and cumulative_log_probs if
        # stochastic beam search is requested.

        perturbed_cumulative_log_probs = other_states.perturbed_cumulative_log_probs
        cumulative_log_probs = other_states.cumulative_log_probs
        tmp_states = other_states.tmp_states

        def UpdateCumulativeScores():
          new_perturbed_cumulative_log_probs = _BatchLookup(
              new_step_ids, tmp_states.top_k_ids,
              tmp_states.new_perturbed_cumulative_log_probs)
          new_log_probs = _BatchLookup(new_step_ids, tmp_states.top_k_ids,
                                       tmp_states.top_k_log_probs)
          new_cumulative_log_probs = cumulative_log_probs + new_log_probs
          return new_perturbed_cumulative_log_probs, new_cumulative_log_probs

        (final_states.perturbed_cumulative_log_probs,
         final_states.cumulative_log_probs) = tf.cond(
             encoder_outputs.stochastic_beam_search.enable,
             UpdateCumulativeScores,
             # No-op.
             lambda: (perturbed_cumulative_log_probs, cumulative_log_probs))

      return final_states

    return Callback

  def InferenceAdditionalEncoder(self, feeds):
    """Generate an inference graph for the additional encoder."""
    return py_utils.NestedMap(), py_utils.NestedMap()
