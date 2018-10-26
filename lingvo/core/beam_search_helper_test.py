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
"""Tests for beam_search_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lingvo.core import beam_search_helper
from lingvo.core import py_utils


class BeamSearchHelperTest(tf.test.TestCase):

  # TODO(yonghui): Add more thorough tests.
  def testBeamSearchHelper(self):
    with self.session(use_gpu=False) as sess:
      np.random.seed(9384758)
      tf.set_random_seed(8274758)
      vocab_size = 12
      src_len = 5
      tgt_len = 7
      num_hyps_per_beam = 3
      src_batch_size = 2
      tgt_batch_size = src_batch_size * num_hyps_per_beam
      p = beam_search_helper.BeamSearchHelper.Params().Set(
          name='bsh', target_seq_len=tgt_len)
      bs_helper = p.cls(p)

      def InitBeamSearchCallBack(
          unused_theta, unused_source_encs, unused_source_padding,
          unused_num_hyps_per_beam, unused_additional_source_info):
        atten_probs = tf.constant(
            np.random.normal(size=(tgt_batch_size, src_len)), dtype=tf.float32)
        return (py_utils.NestedMap({
            'atten_probs': atten_probs
        }), py_utils.NestedMap({
            'atten_probs': atten_probs
        }))

      def PreBeamSearchStepCallback(unused_theta, unused_source_encs,
                                    unused_source_paddings, unused_step_ids,
                                    states, unused_num_hyps_per_beam,
                                    unused_additional_source_info):
        atten_probs = tf.identity(states.atten_probs)
        logits = tf.random_normal([tgt_batch_size, vocab_size], seed=8273747)
        return (py_utils.NestedMap({
            'atten_probs': atten_probs,
            'log_probs': logits
        }), states)

      def PostBeamSearchStepCallback(
          unused_theta, unused_source_encs, unused_source_paddings,
          unused_new_step_ids, states, unused_additional_source_info):
        return states

      src_enc = tf.random_normal([src_len, src_batch_size, 8], seed=982774838)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)

      theta = py_utils.NestedMap()
      decoder_output = bs_helper.BeamSearchDecode(
          theta, src_enc, src_enc_padding, num_hyps_per_beam,
          InitBeamSearchCallBack, PreBeamSearchStepCallback,
          PostBeamSearchStepCallback)

      topk_ids, topk_lens, topk_scores = sess.run([
          decoder_output.topk_ids, decoder_output.topk_lens,
          decoder_output.topk_scores
      ])
      print(np.array_repr(topk_ids))
      print(np.array_repr(topk_lens))
      print(np.array_repr(topk_scores))
      expected_topk_ids = [[4, 3, 4, 3, 2, 0, 0], [4, 3, 11, 2, 0, 0, 0],
                           [4, 3, 6, 2, 0, 0, 0], [6, 0, 4, 6, 6, 11, 2],
                           [6, 0, 4, 6, 1, 2, 0], [6, 0, 4, 6, 6, 2, 0]]
      expected_topk_lens = [5, 4, 4, 7, 6, 6]
      expected_topk_scores = [[8.27340603, 6.26949024, 5.59490776],
                              [9.74691486, 8.46679497, 7.14809656]]
      self.assertEqual(expected_topk_ids, topk_ids.tolist())
      self.assertEqual(expected_topk_lens, topk_lens.tolist())
      self.assertAllClose(expected_topk_scores, topk_scores)


if __name__ == '__main__':
  tf.test.main()
