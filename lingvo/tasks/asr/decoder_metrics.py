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
from typing import Any, Dict

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.tasks.asr import decoder_utils
from lingvo.tasks.asr import eos_normalization
from lingvo.tasks.asr import metrics_calculator
import numpy as np

# hyps: [num_beams, num_hyps_per_beam] of serialized Hypothesis protos.
# ids: [num_beams * num_hyps_per_beam, max_target_length].
# lens: [num_beams * num_hyps_per_beam].
# scores: [num_beams, num_hyps_per_beam].
# decoded: [num_beams, num_hyps_per_beam].
DecoderTopK = collections.namedtuple(
    'topk', ['hyps', 'ids', 'lens', 'scores', 'decoded'])  # pyformat: disable
PostProcessInputs = metrics_calculator.PostProcessInputs


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
    # With the assumption that ids[-1] is always EOS token.
    # TODO(b/195027707): remove EOS token in better way.
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

  def ComputeMetrics(self,
                     decoder_outs,
                     input_batch,
                     ids_to_strings_fn,
                     eos_id=None):
    """Computes metrics on output from decoder.

    Args:
      decoder_outs: A `BeamSearchDecodeOutput`, a namedtuple containing the
        decode results.
      input_batch:  A `NestedMap` of tensors representing the source, target,
        and other components of the input batch.
      ids_to_strings_fn: a function of (ids, lens) -> strings, where ids has
        shape [batch, length], lens has shape [batch], and strings has shape
        [batch].
      eos_id: A number, which is eos_id. The default is None as not all models
        have EOS concept.

    Returns:
      A dict of Tensors containing decoder output and metrics.
    """
    topk = self.GetTopK(decoder_outs, ids_to_strings_fn=ids_to_strings_fn)
    tgt_batch = tf.shape(topk.scores)[0]
    num_hyps_per_beam = tf.shape(topk.scores)[1]
    tgt = input_batch.tgt
    tgt_lens = tf.cast(tf.round(tf.reduce_sum(1.0 - tgt.paddings, 1)), tf.int32)
    tgt_lens = py_utils.HasShape(tgt_lens, [tgt_batch])
    if eos_id is not None:
      tgt_labels, tgt_lens = eos_normalization.NormalizeTrailingEos(
          tgt.labels, tgt_lens, need_trailing_eos=False, eos_id=eos_id)
    else:
      tgt_labels = tgt.labels
    transcripts = ids_to_strings_fn(tgt_labels, tgt_lens)

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

    if 'example_weights' in input_batch:
      example_weights = input_batch.example_weights
    else:
      example_weights = tf.ones([tgt_batch], tf.float32)
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
        'example_weights': example_weights
    }

    if not py_utils.use_tpu() and 'sample_ids' in input_batch:
      ret_dict['utt_id'] = input_batch.sample_ids

    if 'is_real' in input_batch:
      ret_dict['is_real'] = input_batch.is_real

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
          # TODO(xingwu): fully replace 'wer' with 'error_rates/wer'.
          'wer': metrics.AverageMetric(),  # Word error rate.
          'error_rates/ins': metrics.AverageMetric(),  # Insert error rate
          'error_rates/sub': metrics.AverageMetric(),  # Substitute error rate
          'error_rates/del': metrics.AverageMetric(),  # Deletion error rate
          'error_rates/wer': metrics.AverageMetric(),  # Word error rate.
          'sacc': metrics.AverageMetric(),  # Sentence accuracy.
          'ter': metrics.AverageMetric(),  # Token error rate.
          'oracle_norm_wer': metrics.AverageMetric(),
      })

    return base_metrics

  def FilterRealExamples(self, dec_out_dict: Dict[str, Any]) -> None:
    """Remove from input dec_out_dict data that do not refer to real utt."""
    is_real = dec_out_dict['is_real']
    for key in dec_out_dict.keys():
      if key in ['topk_ids', 'topk_lens']:
        # Array length should be the total # of hyps for all utts.
        # Do not assume constant num_hyps/utt. Instead count hyps for each utt.
        temp = []
        start_idx, end_idx = 0, 0
        for utt_idx, hyps_scores in enumerate(dec_out_dict['topk_scores']):
          end_idx = start_idx + len(hyps_scores)
          if is_real[utt_idx]:
            temp.extend(dec_out_dict[key][start_idx:end_idx])
          start_idx = end_idx
        dec_out_dict[key] = np.asarray(temp)
      elif len(dec_out_dict[key]) == len(is_real):
        dec_out_dict[key] = np.asarray(
            [dec_out_dict[key][i] for i in range(len(is_real)) if is_real[i]])

  def PreparePostProcess(self, dec_out_dict,
                         dec_metrics_dict) -> PostProcessInputs:
    """Prepare the objects for PostProcess metrics calculations."""
    assert 'topk_scores' in dec_out_dict, list(dec_out_dict.keys())
    # Filter out examples that is not real (dummy batch paddings).
    if 'is_real' in dec_out_dict:
      self.FilterRealExamples(dec_out_dict)

    topk_scores = dec_out_dict['topk_scores']
    topk_decoded = dec_out_dict['topk_decoded']
    transcripts = dec_out_dict['transcripts']
    utt_id = None
    if not py_utils.use_tpu():
      utt_id = dec_out_dict['utt_id']
      assert len(utt_id) == len(transcripts)
    norm_wer_errors = dec_out_dict['norm_wer_errors']
    norm_wer_words = dec_out_dict['norm_wer_words']
    target_labels = dec_out_dict['target_labels']
    target_paddings = dec_out_dict['target_paddings']
    topk_ids = dec_out_dict['topk_ids']
    topk_lens = dec_out_dict['topk_lens']
    if 'example_weights' in dec_out_dict:
      example_weights = dec_out_dict['example_weights']
    else:
      example_weights = np.ones([len(transcripts)], np.float32)
    assert len(transcripts) == len(target_labels)
    assert len(transcripts) == len(target_paddings)
    assert len(transcripts) == len(topk_decoded)
    assert len(norm_wer_errors) == len(transcripts)
    assert len(norm_wer_words) == len(transcripts)

    num_samples_in_batch = example_weights.sum()
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples_in_batch)

    total_norm_wer_errs = (norm_wer_errors[:, 0] * example_weights).sum()
    total_norm_wer_words = (norm_wer_words[:, 0] * example_weights).sum()

    dec_metrics_dict['norm_wer'].Update(
        total_norm_wer_errs / total_norm_wer_words, total_norm_wer_words)

    filtered_transcripts = []
    filtered_top_hyps = []
    for ref_str, hyps in zip(transcripts, topk_decoded):
      filtered_ref = decoder_utils.FilterNoise(ref_str)
      filtered_ref = decoder_utils.FilterEpsilon(filtered_ref)
      filtered_transcripts.append(filtered_ref)
      filtered_hyp = decoder_utils.FilterNoise(hyps[0])
      filtered_hyp = decoder_utils.FilterEpsilon(filtered_hyp)
      filtered_top_hyps.append(filtered_hyp)
      dec_metrics_dict['corpus_bleu'].Update(filtered_ref, filtered_hyp)

    return PostProcessInputs(transcripts, topk_decoded, filtered_transcripts,
                             filtered_top_hyps, topk_scores, utt_id,
                             norm_wer_errors, target_labels, target_paddings,
                             topk_ids, topk_lens)

  def PostProcess(self, dec_out_dict, dec_metrics_dict):
    key_value_pairs = []  # To store results per each utt, not used now.
    postprocess_inputs = self.PreparePostProcess(dec_out_dict, dec_metrics_dict)
    p = self.params
    if p.include_auxiliary_metrics:
      metrics_calculator.CalculateMetrics(postprocess_inputs, dec_metrics_dict,
                                          self.cluster.add_summary,
                                          py_utils.use_tpu(), p.log_utf8)
    return key_value_pairs
