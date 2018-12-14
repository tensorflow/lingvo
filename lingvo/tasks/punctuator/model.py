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
"""Punctuator model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.tasks.mt import model as mt_model


class TransformerModel(mt_model.TransformerModel):
  """Transformer model."""

  def CreateDecoderMetrics(self):
    decoder_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'corpus_bleu': metrics.CorpusBleuMetric(separator_type=None),
    }
    return decoder_metrics

  def Inference(self):
    """Constructs the inference subgraphs.

    Returns:
      {'subgraph_name': (fetches, feeds)}
    """
    subgraphs = dict()
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    with tf.name_scope('inference'):
      src_strings = tf.placeholder(tf.string, shape=[None])
      _, src_ids, src_paddings = self.input_generator.StringsToIds(
          src_strings, is_source=True)

      # Truncate paddings at the end.
      max_seq_length = tf.to_int32(
          tf.reduce_max(tf.reduce_sum(1.0 - src_paddings, 1)))
      src_paddings = py_utils.with_dependencies([
          py_utils.assert_equal(
              tf.constant(True, tf.bool),
              tf.reduce_all(src_paddings[:, max_seq_length:] > 0.5))
      ], src_paddings)
      src_ids = src_ids[:, :max_seq_length]
      src_paddings = src_paddings[:, :max_seq_length]

      src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      src_enc, src_enc_paddings, _ = self.enc.FPropDefaultTheta(src_input_map)
      decoder_outs = self.dec.BeamSearchDecode(src_enc, src_enc_paddings)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens

      topk_decoded = self.input_generator.IdsToStrings(topk_ids, topk_lens - 1)
      topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_hyps))

      feeds = py_utils.NestedMap({'src_strings': src_strings})
      fetches = py_utils.NestedMap({
          'src_ids': src_ids,
          'topk_decoded': topk_decoded,
          'topk_scores': decoder_outs.topk_scores,
          'topk_hyps': topk_hyps,
      })

      return fetches, feeds
