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
from lingvo.core import test_utils


class BeamSearchHelperTest(test_utils.TestCase):

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

      def InitBeamSearchCallBack(unused_theta, unused_encoder_outputs,
                                 unused_num_hyps_per_beam):
        atten_probs = tf.constant(
            np.random.normal(size=(tgt_batch_size, src_len)), dtype=tf.float32)
        return (py_utils.NestedMap({
            'log_probs': tf.zeros([tgt_batch_size, vocab_size]),
            'atten_probs': atten_probs
        }), py_utils.NestedMap({'atten_probs': atten_probs}))

      def PreBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                    unused_step_ids, states,
                                    unused_num_hyps_per_beam):
        atten_probs = tf.identity(states.atten_probs)
        logits = tf.random_normal([tgt_batch_size, vocab_size], seed=8273747)
        return (py_utils.NestedMap({
            'atten_probs': atten_probs,
            'log_probs': logits
        }), states)

      def PostBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                     unused_new_step_ids, states):
        return states

      src_enc = tf.random_normal([src_len, src_batch_size, 8], seed=982774838)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)

      theta = py_utils.NestedMap()
      decoder_output = bs_helper.BeamSearchDecode(
          theta, encoder_outputs, num_hyps_per_beam, InitBeamSearchCallBack,
          PreBeamSearchStepCallback, PostBeamSearchStepCallback)

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


class MergeBeamSearchOutputsTest(test_utils.TestCase):

  def testMergeBeamSearchOutputs(self):
    with self.session():
      topk_scores_1 = [[1., 3., 5.], [-2., -1., 0.]]
      topk_ids_1 = [[[10, 11, 12], [30, 31, 32], [50, 51, 52]],
                    [[20, 21, 22], [10, 11, 12], [0, 0, 0]]]
      topk_lens_1 = [[3, 3, 2], [3, 3, 0]]
      topk_hyps_1 = [['one', 'three', 'five'], ['minus two', 'minus one', '']]
      topk_1 = beam_search_helper.BeamSearchDecodeOutput(
          None, tf.constant(topk_hyps_1),
          tf.reshape(tf.constant(topk_ids_1), [6, -1]),
          tf.reshape(tf.constant(topk_lens_1), [-1]),
          tf.reshape(tf.constant(topk_scores_1), [-1]), None, None)

      topk_scores_2 = [[2., 4.], [-3., 0.]]
      topk_ids_2 = [[[20, 21, 22], [40, 41, 42]], [[30, 31, 33], [0, 0, 0]]]
      topk_lens_2 = [[3, 2], [3, 0]]
      topk_hyps_2 = [['two', 'four'], ['minus three', '']]
      topk_2 = beam_search_helper.BeamSearchDecodeOutput(
          None, tf.constant(topk_hyps_2),
          tf.reshape(tf.constant(topk_ids_2), [4, -1]),
          tf.reshape(tf.constant(topk_lens_2), [-1]),
          tf.reshape(tf.constant(topk_scores_2), [-1]), None, None)

      topk = beam_search_helper.MergeBeamSearchOutputs(3, [topk_1, topk_2])
      self.assertIsNone(topk.done_hyps)
      self.assertIsNone(topk.topk_decoded)
      self.assertAllEqual([5., 4., 3., -1., -2., -3.], topk.topk_scores.eval())
      self.assertAllEqual([2, 2, 3, 3, 3, 3], topk.topk_lens.eval())
      self.assertAllEqual([[50, 51, 52], [40, 41, 42], [30, 31, 32],
                           [10, 11, 12], [20, 21, 22], [30, 31, 33]],
                          topk.topk_ids.eval())
      self.assertAllEqual([['five', 'four', 'three'],
                           ['minus one', 'minus two', 'minus three']],
                          topk.topk_hyps.eval())


if __name__ == '__main__':
  tf.test.main()
