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
"""Speech model decoder metrics."""

import collections
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.tasks.asr import decoder_utils


# hyps: [num_beams, num_hyps_per_beam] of serialized Hypothesis protos.
# ids: [num_beams * num_hyps_per_beam, max_target_length].
# lens: [num_beams * num_hyps_per_beam].
# scores: [num_beams, num_hyps_per_beam].
# decoded: [num_beams, num_hyps_per_beam].
DecoderTopK = collections.namedtuple(
    'topk', ['hyps', 'ids', 'lens', 'scores', 'decoded'])  # pyformat: disable


def BeamSearchDecodeOutputToDecoderTopK(decoder_outs,
                                        *,
                                        ids_to_strings_fn,
                                        tag=''):
  """Converts BeamSearchDecodeOutput to DecoderTopK.

  As a side-effect, also creates TF nodes used by eval pipelines
  ("top_k_decoded" and "top_k_scores").

  Args:
    decoder_outs: a beam_search_helper.BeamSearchDecodeOutput instance.
    ids_to_strings_fn: a function of (ids, lens) -> strings, where ids has shape
      [batch, length], lens has shape [batch], and strings has shape [batch].
    tag: optional tag for tf.identity() names.

  Returns:
    A DecoderTopK instance.
  """
  hyps = decoder_outs.topk_hyps
  ids = decoder_outs.topk_ids
  lens = tf.identity(decoder_outs.topk_lens, name='TopKLabelLengths' + tag)
  scores = decoder_outs.topk_scores
  decoded = decoder_outs.topk_decoded

  if decoder_outs.topk_ids is not None:
    ids = tf.identity(ids, name='TopKLabelIds' + tag)
    decoded = ids_to_strings_fn(ids, lens - 1)
    decoded = tf.identity(decoded, name='top_k_decoded%s' % tag)
    decoded = tf.reshape(decoded, tf.shape(hyps))
  if scores is not None and hyps is not None:
    scores = tf.identity(
        tf.reshape(scores, tf.shape(lens)), name='top_k_scores%s' % tag)
    scores = tf.reshape(scores, tf.shape(hyps))
  return DecoderTopK(hyps, ids, lens, scores, decoded)


class DecoderMetrics(base_layer.BaseLayer):
  """Speech model decoder metrics."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'include_auxiliary_metrics', True,
        'In addition to simple WER, also computes oracle WER, SACC, TER, etc. '
        'Turning off this option will speed up the decoder job.')
    p.Define(
        'log_utf8', False,
        'If True, decodes reference and hypotheses bytes to UTF-8 for logging.')
    return p

  def __init__(self, params):
    if not params.name:
      raise ValueError('params.name not set.')
    super().__init__(params)
    p = self.params

  def GetTopK(self, decoder_outs, ids_to_strings_fn, tag=''):
    return BeamSearchDecodeOutputToDecoderTopK(
        decoder_outs, ids_to_strings_fn=ids_to_strings_fn, tag=tag)

  def ComputeNormalizedWER(self, hyps, refs, num_hyps_per_beam):
    # Filter out all '<epsilon>' tokens for norm_wer computation.
    hyps_no_epsilon = tf.strings.regex_replace(hyps, '(<epsilon>)+', ' ')
    # norm_wer is size [num_transcripts * hyps_per_beam, 2]
    norm_wer = decoder_utils.ComputeWer(hyps_no_epsilon, refs)
    # Split into two tensors of size [num_transcripts * hyps_per_beam, 1]
    norm_wer_errors, norm_wer_words = tf.split(norm_wer, [1, 1], 1)
    shape = [-1, num_hyps_per_beam]
    norm_wer_errors = tf.reshape(norm_wer_errors, shape)
    norm_wer_words = tf.reshape(norm_wer_words, shape)

    return norm_wer_errors, norm_wer_words

  def AddAdditionalDecoderMetricsToGraph(self, topk_hyps, filtered_hyps,
                                         filtered_refs, input_batch,
                                         decoder_outs):
    """Returns a dict of metrics which should be computed from decoded hyps."""
    # The base class implementation returns an empty dictionary. Sub-classes can
    # provide their own implementation.
    return {}

  def ComputeMetrics(self, decoder_outs, input_batch, ids_to_strings_fn):
    """Computes metrics on output from decoder.

    Args:
      decoder_outs: A `BeamSearchDecodeOutput`, a namedtuple containing the
        decode results.
      input_batch:  A `NestedMap` of tensors representing the source, target,
        and other components of the input batch.
      ids_to_strings_fn: a function of (ids, lens) -> strings, where ids has
        shape [batch, length], lens has shape [batch], and strings has shape
        [batch].

    Returns:
      A dict of Tensors containing decoder output and metrics.
    """
    topk = self.GetTopK(decoder_outs, ids_to_strings_fn=ids_to_strings_fn)
    tgt_batch = tf.shape(topk.scores)[0]
    num_hyps_per_beam = tf.shape(topk.scores)[1]
    tgt = input_batch.tgt
    tgt_lens = tf.cast(tf.round(tf.reduce_sum(1.0 - tgt.paddings, 1)), tf.int32)
    tgt_lens = py_utils.HasShape(tgt_lens, [tgt_batch])
    transcripts = ids_to_strings_fn(tgt.labels, tgt_lens - 1)

    # Filter out all isolated '<noise>' tokens.
    noise_pattern = ' <noise> |^<noise> | <noise>$|^<noise>$'
    filtered_refs = tf.strings.regex_replace(transcripts, noise_pattern, ' ')
    filtered_hyps = tf.strings.regex_replace(topk.decoded, noise_pattern, ' ')
    # Compute translation quality scores for all hyps.
    filtered_refs = tf.tile(
        tf.reshape(filtered_refs, [-1, 1]), [1, num_hyps_per_beam])
    filtered_hyps = tf.reshape(filtered_hyps, [-1])
    filtered_refs = tf.reshape(filtered_refs, [-1])
    tf.logging.info('filtered_refs=%s', filtered_refs)
    norm_wer_errors, norm_wer_words = self.ComputeNormalizedWER(
        filtered_hyps, filtered_refs, num_hyps_per_beam)

    ret_dict = {
        'target_ids': tgt.ids,
        'target_labels': tgt.labels,
        'target_weights': tgt.weights,
        'target_paddings': tgt.paddings,
        'transcripts': transcripts,
        'topk_decoded': topk.decoded,
        'topk_ids': topk.ids,
        'topk_lens': topk.lens,
        'topk_scores': topk.scores,
        'norm_wer_errors': norm_wer_errors,
        'norm_wer_words': norm_wer_words,
    }

    if not py_utils.use_tpu() and 'sample_ids' in input_batch:
      ret_dict['utt_id'] = input_batch.sample_ids

    ret_dict.update(
        self.AddAdditionalDecoderMetricsToGraph(topk, filtered_hyps,
                                                filtered_refs, input_batch,
                                                decoder_outs))
    return ret_dict

  def CreateMetrics(self):
    base_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'norm_wer': metrics.AverageMetric(),  # Normalized word error rate.
        'corpus_bleu': metrics.CorpusBleuMetric(),
    }

    if self.params.include_auxiliary_metrics:
      base_metrics.update({
          'wer': metrics.AverageMetric(),  # Word error rate.
          'sacc': metrics.AverageMetric(),  # Sentence accuracy.
          'ter': metrics.AverageMetric(),  # Token error rate.
          'oracle_norm_wer': metrics.AverageMetric(),
      })

    return base_metrics

  def PostProcess(self, dec_out_dict, dec_metrics_dict):
    p = self.params
    assert 'topk_scores' in dec_out_dict, list(dec_out_dict.keys())
    topk_scores = dec_out_dict['topk_scores']
    topk_decoded = dec_out_dict['topk_decoded']
    transcripts = dec_out_dict['transcripts']
    if not py_utils.use_tpu():
      utt_id = dec_out_dict['utt_id']
      assert len(utt_id) == len(transcripts)
    norm_wer_errors = dec_out_dict['norm_wer_errors']
    norm_wer_words = dec_out_dict['norm_wer_words']
    target_labels = dec_out_dict['target_labels']
    target_paddings = dec_out_dict['target_paddings']
    topk_ids = dec_out_dict['topk_ids']
    topk_lens = dec_out_dict['topk_lens']
    assert len(transcripts) == len(target_labels)
    assert len(transcripts) == len(target_paddings)
    assert len(transcripts) == len(topk_decoded)
    assert len(norm_wer_errors) == len(transcripts)
    assert len(norm_wer_words) == len(transcripts)

    num_samples_in_batch = len(transcripts)
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples_in_batch)

    def GetRefIds(ref_ids, ref_paddinds):
      assert len(ref_ids) == len(ref_paddinds)
      return_ids = []
      for i in range(len(ref_ids)):
        if ref_paddinds[i] == 0:
          return_ids.append(ref_ids[i])
      return return_ids

    total_norm_wer_errs = norm_wer_errors[:, 0].sum()
    total_norm_wer_words = norm_wer_words[:, 0].sum()

    dec_metrics_dict['norm_wer'].Update(
        total_norm_wer_errs / total_norm_wer_words, total_norm_wer_words)

    for ref_str, hyps in zip(transcripts, topk_decoded):
      filtered_ref = decoder_utils.FilterNoise(ref_str)
      filtered_ref = decoder_utils.FilterEpsilon(filtered_ref)
      filtered_hyp = decoder_utils.FilterNoise(hyps[0])
      filtered_hyp = decoder_utils.FilterEpsilon(filtered_hyp)
      dec_metrics_dict['corpus_bleu'].Update(filtered_ref, filtered_hyp)

    total_errs = 0
    total_oracle_errs = 0
    total_ref_words = 0
    total_token_errs = 0
    total_ref_tokens = 0
    total_accurate_sentences = 0
    key_value_pairs = []

    if p.include_auxiliary_metrics:
      for i in range(len(transcripts)):
        ref_str = transcripts[i]
        if not py_utils.use_tpu():
          tf.logging.info('utt_id: %s', utt_id[i])
        if self.cluster.add_summary:
          tf.logging.info('  ref_str: %s',
                          ref_str.decode('utf-8') if p.log_utf8 else ref_str)
        hyps = topk_decoded[i]
        num_hyps_per_beam = len(hyps)
        ref_ids = GetRefIds(target_labels[i], target_paddings[i])
        hyp_index = i * num_hyps_per_beam
        top_hyp_ids = topk_ids[hyp_index][:topk_lens[hyp_index]]
        if self.cluster.add_summary:
          tf.logging.info('  ref_ids: %s', ref_ids)
          tf.logging.info('  top_hyp_ids: %s', top_hyp_ids)
        total_ref_tokens += len(ref_ids)
        _, _, _, token_errs = decoder_utils.EditDistanceInIds(
            ref_ids, top_hyp_ids)
        total_token_errs += token_errs

        filtered_ref = decoder_utils.FilterNoise(ref_str)
        filtered_ref = decoder_utils.FilterEpsilon(filtered_ref)
        oracle_errs = norm_wer_errors[i][0]
        for n, (score, hyp_str) in enumerate(zip(topk_scores[i], hyps)):
          if self.cluster.add_summary:
            tf.logging.info('  %f: %s', score,
                            hyp_str.decode('utf-8') if p.log_utf8 else hyp_str)
          filtered_hyp = decoder_utils.FilterNoise(hyp_str)
          filtered_hyp = decoder_utils.FilterEpsilon(filtered_hyp)
          ins, subs, dels, errs = decoder_utils.EditDistance(
              filtered_ref, filtered_hyp)
          # Note that these numbers are not consistent with what is used to
          # compute normalized WER.  In particular, these numbers will be
          # inflated when the transcript contains punctuation.
          tf.logging.info('  ins: %d, subs: %d, del: %d, total: %d', ins, subs,
                          dels, errs)
          # Only aggregate scores of the top hypothesis.
          if n == 0:
            total_errs += errs
            total_ref_words += len(decoder_utils.Tokenize(filtered_ref))
            if norm_wer_errors[i, n] == 0:
              total_accurate_sentences += 1
          oracle_errs = min(oracle_errs, norm_wer_errors[i, n])
        total_oracle_errs += oracle_errs

      dec_metrics_dict['wer'].Update(total_errs / max(1., total_ref_words),
                                     total_ref_words)
      dec_metrics_dict['oracle_norm_wer'].Update(
          total_oracle_errs / max(1., total_ref_words), total_ref_words)
      dec_metrics_dict['sacc'].Update(
          total_accurate_sentences / len(transcripts), len(transcripts))
      dec_metrics_dict['ter'].Update(
          total_token_errs / max(1., total_ref_tokens), total_ref_tokens)

    return key_value_pairs
