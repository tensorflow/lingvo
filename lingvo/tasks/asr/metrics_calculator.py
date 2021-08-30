# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Calculate and update metrics for Decode Post-Process."""

import collections

from typing import Any, Dict

import lingvo.compat as tf
from lingvo.tasks.asr import decoder_utils

# transcripts: [num_utts]. A sequence of transcripts (references), one for
#   each utterance.
# topk_decoded: [num_utts, num_topk_hyps]. A sequence, for each utterance, of a
#   sequence of the top K hypotheses.
# filtered_transcripts: [num_utts]. Same as transcripts, but after simple
#   filtering.
# filtered_top_hyps: [num_utts]. A sequence of filtered top hypotheses, one for
#   each utterance.
# topk_scores: [num_utts, num_topk_hyps]. A sequence, for each utterance, of a
#   sequence of the top K decoder scores corresponding to the top K hypotheses.
# utt_id: [num_utts]. A sequence of utterance_ids.
# norm_wer_errors: [num_utts, num_topk_hyps]. A sequence, for each utterance, of
#   a sequence of the normalized wer corresponding to the top K hypotheses.
# target_labels: [num_utts, max_target_length]. A sequence, for each utterance,
#   of a sequence of labels for the target(reference) tokens (e.g. word-pieces).
# target_paddings: [num_utts, max_target_length]. A sequence, for each
#   utterance, of a sequence of padding values for the target(reference) token
#   sequence, used to determine non-padding labels.
# topk_ids: [num_utts * num_topk_hyps, max_target_length]. A sequence, for each
#   topK hypothesis of ALL utterances, of a sequence of token labels/ids
#   (up to max_target_length).
# topk_lens: [num_utts * num_topk_hyps]. A sequence, for each topK hypothesis of
#   ALL utterances, of the length of its token labels/ids sequence.
PostProcessInputs = collections.namedtuple('postprocess_input', [
    'transcripts', 'topk_decoded', 'filtered_transcripts', 'filtered_top_hyps',
    'topk_scores', 'utt_id', 'norm_wer_errors', 'target_labels',
    'target_paddings', 'topk_ids', 'topk_lens'
])


def GetRefIds(ref_ids, ref_paddings):
  assert len(ref_ids) == len(ref_paddings)
  return_ids = []
  for i in range(len(ref_ids)):
    if ref_paddings[i] == 0:
      return_ids.append(ref_ids[i])
  return return_ids


def CalculateMetrics(
    postprocess_inputs: PostProcessInputs,
    dec_metrics_dict: Dict[str, Any],
    add_summary: bool,
    use_tpu: bool,
    log_utf8: bool,
):
  """Calculate and update metrics.

  Args:
    postprocess_inputs: namedtuple of Postprocess input objects/tensors.
    dec_metrics_dict: A dictionary of metric names to metrics.
    add_summary: Whether to add detailed summary logging for processing each
      utterance.
    use_tpu: Whether TPU is used (for decoding).
    log_utf8: DecoderMetrics param. If True, decode reference and hypotheses
      bytes to UTF-8 for logging.
  """
  (transcripts, topk_decoded, filtered_transcripts, filtered_top_hyps,
   topk_scores, utt_id, norm_wer_errors, target_labels, target_paddings,
   topk_ids, topk_lens) = postprocess_inputs

  total_ins, total_subs, total_dels, total_errs = 0, 0, 0, 0
  total_oracle_errs = 0
  total_ref_words = 0
  total_token_errs = 0
  total_ref_tokens = 0
  total_accurate_sentences = 0

  for i in range(len(transcripts)):
    ref_str = transcripts[i]
    if not use_tpu:
      tf.logging.info('utt_id: %s', utt_id[i])
    if add_summary:
      tf.logging.info('  ref_str: %s',
                      ref_str.decode('utf-8') if log_utf8 else ref_str)
    hyps = topk_decoded[i]
    num_hyps_per_beam = len(hyps)
    ref_ids = GetRefIds(target_labels[i], target_paddings[i])
    hyp_index = i * num_hyps_per_beam
    top_hyp_ids = topk_ids[hyp_index][:topk_lens[hyp_index]]
    if add_summary:
      tf.logging.info('  ref_ids: %s', ref_ids)
      tf.logging.info('  top_hyp_ids: %s', top_hyp_ids)
    total_ref_tokens += len(ref_ids)
    _, _, _, token_errs = decoder_utils.EditDistanceInIds(ref_ids, top_hyp_ids)
    total_token_errs += token_errs

    filtered_ref = filtered_transcripts[i]
    oracle_errs = norm_wer_errors[i][0]
    for n, (score, hyp_str) in enumerate(zip(topk_scores[i], hyps)):
      oracle_errs = min(oracle_errs, norm_wer_errors[i, n])
      if add_summary:
        tf.logging.info('  %f: %s', score,
                        hyp_str.decode('utf-8') if log_utf8 else hyp_str)
      # Only aggregate scores of the top hypothesis.
      if n != 0:
        continue
      filtered_hyp = filtered_top_hyps[i]
      ins, subs, dels, errs = decoder_utils.EditDistance(
          filtered_ref, filtered_hyp)
      total_ins += ins
      total_subs += subs
      total_dels += dels
      total_errs += errs
      ref_words = len(decoder_utils.Tokenize(filtered_ref))
      total_ref_words += ref_words
      if norm_wer_errors[i, n] == 0:
        total_accurate_sentences += 1
      tf.logging.info(
          '  ins: %d, subs: %d, del: %d, total: %d, ref_words: %d, wer: %f',
          ins, subs, dels, errs, ref_words, errs / max(1, ref_words))

    total_oracle_errs += oracle_errs

  non_zero_total_ref_words = max(1., total_ref_words)
  dec_metrics_dict['wer'].Update(total_errs / non_zero_total_ref_words,
                                 total_ref_words)
  dec_metrics_dict['error_rates/ins'].Update(
      total_ins / non_zero_total_ref_words, total_ref_words)
  dec_metrics_dict['error_rates/sub'].Update(
      total_subs / non_zero_total_ref_words, total_ref_words)
  dec_metrics_dict['error_rates/del'].Update(
      total_dels / non_zero_total_ref_words, total_ref_words)
  dec_metrics_dict['error_rates/wer'].Update(
      total_errs / non_zero_total_ref_words, total_ref_words)
  dec_metrics_dict['oracle_norm_wer'].Update(
      total_oracle_errs / non_zero_total_ref_words, total_ref_words)
  dec_metrics_dict['sacc'].Update(total_accurate_sentences / len(transcripts),
                                  len(transcripts))
  dec_metrics_dict['ter'].Update(total_token_errs / max(1., total_ref_tokens),
                                 total_ref_tokens)
