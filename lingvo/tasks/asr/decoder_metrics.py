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
from lingvo.core import tokenizers
from lingvo.tasks.asr import decoder_utils
from lingvo.tasks.asr import eos_normalization
from lingvo.tasks.asr import metrics_calculator
import numpy as np

# hyps: [num_beams, num_hyps_per_beam] of serialized Hypothesis protos.
# ids: [num_beams * num_hyps_per_beam, max_target_length].
# lens: [num_beams * num_hyps_per_beam].
# scores: [num_beams, num_hyps_per_beam].
# decoded: [num_beams, num_hyps_per_beam].
# alignment: [num_beams * num_hyps_per_beam, max_src_length, 2]
DecoderTopK = collections.namedtuple(
    'topk', ['hyps', 'ids', 'lens', 'scores', 'decoded', 'alignment'],
    defaults=[{}])  # pyformat: disable
PostProcessInputs = metrics_calculator.PostProcessInputs


def BeamSearchDecodeOutputToDecoderTopK(decoder_outs,
                                        *,
                                        ids_to_strings_fn,
                                        feed_encoder_outs=False,
                                        encoder_outs=None,
                                        tag=''):
  """Converts BeamSearchDecodeOutput to DecoderTopK.

  As a side-effect, also creates TF nodes used by eval pipelines
  ("top_k_decoded" and "top_k_scores").

  Args:
    decoder_outs: a beam_search_helper.BeamSearchDecodeOutput instance.
    ids_to_strings_fn: a function of (ids, lens) -> strings, where ids has shape
      [batch, length], lens has shape [batch], and strings has shape [batch].
    feed_encoder_outs: whether feed encoder_outs to ids_to_strings_fn or not.
    encoder_outs: outputs derived from the encoder.
    tag: optional tag for tf.identity() names.

  Returns:
    A DecoderTopK instance.
  """
  hyps = decoder_outs.topk_hyps
  ids = decoder_outs.topk_ids
  lens = tf.identity(decoder_outs.topk_lens, name='TopKLabelLengths' + tag)
  scores = decoder_outs.topk_scores
  decoded = decoder_outs.topk_decoded
  alignment = getattr(decoder_outs, 'topk_alignment', {})

  if decoder_outs.topk_ids is not None:
    ids = tf.identity(ids, name='TopKLabelIds' + tag)
    # With the assumption that ids[-1] is always EOS token.
    # TODO(b/195027707): remove EOS token in better way.
    if feed_encoder_outs:
      decoded = ids_to_strings_fn(ids, lens - 1, encoder_outs=encoder_outs)
    else:
      decoded = ids_to_strings_fn(ids, lens - 1)
    decoded = tf.identity(decoded, name='top_k_decoded%s' % tag)
    decoded = tf.reshape(decoded, tf.shape(scores))
  if scores is not None and hyps is not None:
    scores = tf.identity(
        tf.reshape(scores, tf.shape(lens)), name='top_k_scores%s' % tag)
    scores = tf.reshape(scores, tf.shape(hyps))
  return DecoderTopK(hyps, ids, lens, scores, decoded, alignment)


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
    p.Define(
        'only_output_tpu_tensors', False,
        'If True, ComputeMetrics() will only output TPU compatible tensors, '
        'no string output, and PostProcess() will run Detokenize. '
        'Note: This requires tokenizer implemented IdsToStringsPython().')
    p.Define(
        'pass_through_transcript_field', None,
        'If None, we get ground truth transcripts for scoring by detokenizing '
        'input_batch.tgt. Otherwise, the transcript is from the specified '
        'field in the input batch.')
    return p

  def __init__(self, params):
    if not params.name:
      raise ValueError('params.name not set.')
    super().__init__(params)

  def GetTopK(self,
              decoder_outs,
              ids_to_strings_fn,
              feed_encoder_outs=False,
              encoder_outs=None,
              tag=''):
    return BeamSearchDecodeOutputToDecoderTopK(
        decoder_outs,
        ids_to_strings_fn=ids_to_strings_fn,
        feed_encoder_outs=feed_encoder_outs,
        encoder_outs=encoder_outs,
        tag=tag)

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

    if 'example_weights' in input_batch:
      example_weights = input_batch.example_weights
    else:
      example_weights = tf.ones([tgt_batch], tf.float32)
    ret_dict = {
        'topk_decoded': topk.decoded,
        'topk_ids': topk.ids,
        'topk_lens': topk.lens,
        'topk_scores': topk.scores,
        'example_weights': example_weights
    }

    if 'is_real' in input_batch:
      ret_dict['is_real'] = input_batch.is_real
    # For CPU run, we pass following string fields from input_batch to ret_dict.
    if not py_utils.use_tpu():
      if 'sample_ids' in input_batch:
        ret_dict['utt_id'] = input_batch.sample_ids
      if 'language' in input_batch:
        ret_dict['language'] = input_batch.language
      if 'testset_name' in input_batch:
        ret_dict['testset_name'] = input_batch.input_batch
      transcript_field = self.params.pass_through_transcript_field
      if transcript_field:
        ret_dict[transcript_field] = input_batch.Get(transcript_field, None)

    if not self.params.pass_through_transcript_field:
      tgt = input_batch.tgt
      tgt_lens = tf.cast(
          tf.round(tf.reduce_sum(1.0 - tgt.paddings, 1)), tf.int32)
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

      ret_dict.update({
          'target_ids': tgt.ids,
          'target_labels': tgt.labels,
          'target_weights': tgt.weights,
          'target_paddings': tgt.paddings,
          'transcripts': transcripts,
          'norm_wer_errors': norm_wer_errors,
          'norm_wer_words': norm_wer_words,
      })
      ret_dict.update(
          self.AddAdditionalDecoderMetricsToGraph(topk, filtered_hyps,
                                                  filtered_refs, input_batch,
                                                  decoder_outs))
    # Remove string outputs
    if self.params.only_output_tpu_tensors:
      ret_dict.pop('transcripts', None)
      ret_dict.pop('topk_decoded', None)
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
          'error_rates/ins': metrics.AverageMetric(),  # Insert error rate.
          'error_rates/sub': metrics.AverageMetric(),  # Substitute error rate.
          'error_rates/del': metrics.AverageMetric(),  # Deletion error rate.
          'error_rates/wer': metrics.AverageMetric(),  # Word error rate.
          'case_insensitive_error_rates/ins':
              metrics.AverageMetric(),  # Insert case-insensitive error rate.
          'case_insensitive_error_rates/sub': metrics.AverageMetric(
          ),  # Substitute case-insensitive error rate.
          'case_insensitive_error_rates/del':
              metrics.AverageMetric(),  # Deletion case-insensitive error rate.
          'case_insensitive_error_rates/wer':
              metrics.AverageMetric(),  # Case-insensitive Word error rate.
          'sacc': metrics.AverageMetric(),  # Sentence accuracy.
          'ter': metrics.AverageMetric(),  # Token error rate.
          'oracle_norm_wer': metrics.AverageMetric(),
          'oracle/ins': metrics.AverageMetric(),
          'oracle/sub': metrics.AverageMetric(),
          'oracle/del': metrics.AverageMetric(),
      })

    return base_metrics

  def FilterRealExamples(self, dec_out_dict: Dict[str, Any]) -> None:
    """Remove from input dec_out_dict data that do not refer to real utt."""
    is_real = dec_out_dict['is_real'].copy()
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

  def DeTokenizeTranscripts(self, dec_out_dict: Dict[str, Any],
                            tokenizer: tokenizers.BaseTokenizer) -> None:
    """Fill transcripts in dec_out_dict."""
    if self.params.pass_through_transcript_field:
      assert self.params.pass_through_transcript_field in dec_out_dict, (
          'p.pass_through_transcript_field specified as '
          f'{self.params.pass_through_transcript_field}, but '
          'this field not exist in dec_out_dict!')
      if self.params.pass_through_transcript_field != 'transcripts':
        dec_out_dict['transcripts'] = dec_out_dict[
            self.params.pass_through_transcript_field].copy()
        del dec_out_dict[self.params.pass_through_transcript_field]
    else:
      raise NotImplementedError(
          'DeTokenizeTranscripts not implemented when '
          'p.pass_through_transcript_field not specified!')

  def DeTokenizeTopkDecoded(self, dec_out_dict: Dict[str, Any],
                            tokenizer: tokenizers.BaseTokenizer) -> None:
    """Fill topk_decoded in dec_out_dict."""
    raise NotImplementedError('DeTokenizeTopkDecoded not implemented!')

  def PreparePostProcess(self,
                         dec_out_dict,
                         dec_metrics_dict,
                         tokenizer=None) -> PostProcessInputs:
    """Prepare the objects for PostProcess metrics calculations."""
    assert 'topk_scores' in dec_out_dict, list(dec_out_dict.keys())

    # Detokenize transcripts outside of tf session.
    if 'transcripts' not in dec_out_dict:
      self.DeTokenizeTranscripts(dec_out_dict, tokenizer)
    # Detokenize topk_decoded outside of tf session.
    if 'topk_decoded' not in dec_out_dict:
      self.DeTokenizeTopkDecoded(dec_out_dict, tokenizer)
    # Filter out examples that is not real (dummy batch paddings).
    if 'is_real' in dec_out_dict:
      self.FilterRealExamples(dec_out_dict)

    topk_scores = dec_out_dict['topk_scores']
    topk_decoded = dec_out_dict['topk_decoded']
    transcripts = dec_out_dict['transcripts']

    # If all filtered, early return.
    if not transcripts.size:
      return PostProcessInputs(
          np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
          np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
          np.array([]))

    utt_id = None
    if 'utt_id' in dec_out_dict:
      utt_id = dec_out_dict['utt_id']
    elif 'sample_ids' in dec_out_dict:
      utt_id = dec_out_dict['sample_ids']
    assert utt_id is None or len(utt_id) == len(transcripts)

    topk_ids = dec_out_dict['topk_ids']
    topk_lens = dec_out_dict['topk_lens']
    assert len(transcripts) == len(topk_decoded)

    if 'example_weights' in dec_out_dict:
      example_weights = dec_out_dict['example_weights']
    else:
      example_weights = np.ones([len(transcripts)], np.float32)
    num_samples_in_batch = example_weights.sum()
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples_in_batch)

    if self.params.pass_through_transcript_field:
      # When using pass through transcripts, we donot care token level metrics.
      # Fake these np arrays to avoid crash.
      norm_wer_errors = np.zeros_like(topk_scores, dtype=np.float32)
      target_labels = np.zeros(
          shape=[len(transcripts), topk_ids.shape[1]], dtype=np.int32)
      target_paddings = np.ones_like(target_labels, dtype=np.int32)
    else:
      norm_wer_errors = dec_out_dict['norm_wer_errors']
      norm_wer_words = dec_out_dict['norm_wer_words']
      target_labels = dec_out_dict['target_labels']
      target_paddings = dec_out_dict['target_paddings']
      assert len(norm_wer_errors) == len(transcripts)
      assert len(norm_wer_words) == len(transcripts)
      assert len(transcripts) == len(target_labels)
      assert len(transcripts) == len(target_paddings)
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

  def PostProcess(self, dec_out_dict, dec_metrics_dict, tokenizer=None):
    key_value_pairs = []  # To store results per each utt, not used now.
    postprocess_inputs = self.PreparePostProcess(dec_out_dict, dec_metrics_dict,
                                                 tokenizer)
    p = self.params
    if p.include_auxiliary_metrics:
      metrics_calculator.CalculateMetrics(postprocess_inputs, dec_metrics_dict,
                                          self.cluster.add_summary,
                                          py_utils.use_tpu(), p.log_utf8)
    return key_value_pairs
