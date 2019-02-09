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

from lingvo.core import py_utils
from lingvo.tasks.mt import model as mt_model


class RNMTModel(mt_model.RNMTModel):
  """The MT model with an inference graph for punctuator."""

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

      src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens

      # topk_lens - 1 to remove the EOS id.
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
