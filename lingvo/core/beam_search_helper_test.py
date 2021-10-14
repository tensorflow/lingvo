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
# ==============================================================================
"""Tests for beam_search_helper."""

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import beam_search_helper
from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np


def GetBeamSearchHelperResults(sess,
                               num_hyps_per_beam,
                               pass_seq_lengths=False,
                               force_eos_in_top_k=False):
  np.random.seed(9384758)
  tf.random.set_seed(8274758)
  vocab_size = 12
  src_len = 5
  tgt_len = 7
  src_batch_size = 2
  tgt_batch_size = src_batch_size * num_hyps_per_beam
  p = beam_search_helper.BeamSearchHelper.Params().Set(
      name='bsh', target_seq_len=tgt_len, force_eos_in_top_k=force_eos_in_top_k)
  bs_helper = p.Instantiate()

  def InitBeamSearchState(unused_theta, unused_encoder_outputs,
                          unused_num_hyps_per_beam):
    atten_probs = tf.constant(
        np.random.normal(size=(tgt_batch_size, src_len)), dtype=tf.float32)
    return (py_utils.NestedMap({
        'log_probs': tf.zeros([tgt_batch_size, vocab_size]),
        'atten_probs': atten_probs,
    }), py_utils.NestedMap({'atten_probs': atten_probs}))

  def PreBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                unused_step_ids, states,
                                unused_num_hyps_per_beam):
    atten_probs = tf.identity(states.atten_probs)
    logits = tf.random.normal([tgt_batch_size, vocab_size], seed=8273747)
    return (py_utils.NestedMap({
        'atten_probs': atten_probs,
        'log_probs': logits
    }), states)

  def PostBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                 unused_new_step_ids, states):
    return states

  src_enc = tf.random.normal([src_len, src_batch_size, 8], seed=982774838)
  src_enc_padding = tf.constant(
      [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
      dtype=tf.float32)
  encoder_outputs = py_utils.NestedMap(encoded=src_enc, padding=src_enc_padding)
  if pass_seq_lengths:
    encoder_outputs['seq_lengths'] = tf.constant([4, 3], dtype=tf.int32)

  theta = py_utils.NestedMap()
  decoder_output = bs_helper.BeamSearchDecode(theta, encoder_outputs,
                                              num_hyps_per_beam,
                                              InitBeamSearchState,
                                              PreBeamSearchStepCallback,
                                              PostBeamSearchStepCallback)

  topk_ids, topk_lens, topk_scores = sess.run([
      decoder_output.topk_ids, decoder_output.topk_lens,
      decoder_output.topk_scores
  ])
  return topk_ids, topk_lens, topk_scores


class BeamSearchHelperTest(test_utils.TestCase, parameterized.TestCase):

  # TODO(yonghui): Add more thorough tests.
  def testBeamSearchHelper(self):
    with self.session(use_gpu=False) as sess:
      topk_ids, topk_lens, topk_scores = GetBeamSearchHelperResults(
          sess, num_hyps_per_beam=3)
      print(np.array_repr(topk_ids))
      print(np.array_repr(topk_lens))
      print(np.array_repr(topk_scores))
      expected_topk_ids = [[4, 3, 4, 3, 2, 0, 0], [4, 3, 11, 2, 0, 0, 0],
                           [4, 3, 6, 2, 0, 0, 0], [6, 0, 4, 6, 6, 11, 2],
                           [6, 0, 4, 6, 1, 2, 0], [6, 0, 4, 6, 6, 2, 0]]
      expected_topk_lens = [5, 4, 4, 7, 6, 6]
      expected_topk_scores = [[8.27340603, 6.26949024, 5.59490776],
                              [9.74691486, 8.46679497, 7.14809656]]
      self.assertAllEqual(expected_topk_ids, topk_ids.tolist())
      self.assertAllEqual(expected_topk_lens, topk_lens.tolist())
      self.assertAllClose(expected_topk_scores, topk_scores)

  def testBeamSearchHelperHypsOne(self):
    with self.session(use_gpu=False) as sess:
      topk_ids, topk_lens, topk_scores = GetBeamSearchHelperResults(
          sess, num_hyps_per_beam=1)
      print(np.array_repr(topk_ids))
      print(np.array_repr(topk_lens))
      print(np.array_repr(topk_scores))
      expected_topk_ids = [[9, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
      expected_topk_lens = [2, 0]
      expected_topk_scores = [[3.778749], [0.0]]
      self.assertAllEqual(expected_topk_ids, topk_ids.tolist())
      self.assertAllEqual(expected_topk_lens, topk_lens.tolist())
      self.assertAllClose(expected_topk_scores, topk_scores)

  def testBeamSearchHelperWithSeqLengths(self):
    with self.session(use_gpu=False) as sess:
      topk_ids, topk_lens, topk_scores = GetBeamSearchHelperResults(
          sess, num_hyps_per_beam=3, pass_seq_lengths=True)
      print(np.array_repr(topk_ids))
      print(np.array_repr(topk_lens))
      print(np.array_repr(topk_scores))
      expected_topk_ids = [[4, 3, 4, 3, 2, 0, 0], [4, 3, 11, 2, 0, 0, 0],
                           [4, 3, 6, 2, 0, 0, 0], [6, 0, 4, 6, 6, 11, 2],
                           [6, 0, 4, 6, 1, 2, 0], [6, 0, 4, 6, 6, 2, 0]]
      expected_topk_lens = [5, 4, 4, 7, 6, 6]
      expected_topk_scores = [[8.27340603, 6.26949024, 5.59490776],
                              [9.74691486, 8.46679497, 7.14809656]]
      self.assertAllEqual(expected_topk_ids, topk_ids.tolist())
      self.assertAllEqual(expected_topk_lens, topk_lens.tolist())
      self.assertAllClose(expected_topk_scores, topk_scores)

  def testBeamSearchHelperForceEos(self):
    with self.session(use_gpu=False) as sess:
      topk_ids, topk_lens, topk_scores = GetBeamSearchHelperResults(
          sess, num_hyps_per_beam=3, force_eos_in_top_k=True)
      print(np.array_repr(topk_ids))
      print(np.array_repr(topk_lens))
      print(np.array_repr(topk_scores))
      expected_topk_ids = [
          [4, 3, 11, 6, 9, 3, 2],
          [4, 3, 11, 6, 9, 7, 2],
          [4, 3, 4, 1, 4, 1, 2],
          [6, 0, 4, 6, 6, 11, 2],
          [6, 0, 4, 6, 3, 3, 2],
          [6, 0, 4, 6, 1, 2, 0],
      ]
      expected_topk_lens = [7, 7, 7, 7, 7, 6]
      expected_topk_scores = [[10.576365, 9.345996, 9.125197],
                              [9.746915, 8.905771, 8.466795]]
      self.assertAllEqual(expected_topk_ids, topk_ids.tolist())
      self.assertAllEqual(expected_topk_lens, topk_lens.tolist())
      self.assertAllClose(expected_topk_scores, topk_scores)

  @parameterized.named_parameters(
      ('eos_valid_in_topk', 100.0, True),
      ('eos_valid_not_in_topk', 100.0, False),
      ('eos_not_valid_in_topk', 0.5, True),
      ('eos_not_valid_not_in_topk', 0.5, False),
  )
  def testBeamSearchForceEosInTopK(self, valid_eos_max_logit_delta,
                                   force_eos_in_top_k):
    with self.session() as sess:
      vocab_size = 300
      tgt_len = 100
      num_hyps_per_beam = 3
      src_batch_size = 2
      tgt_batch_size = src_batch_size * num_hyps_per_beam
      p = beam_search_helper.BeamSearchHelper.Params().Set(
          name='bsh',
          target_seq_len=tgt_len,
          num_hyps_per_beam=num_hyps_per_beam,
          beam_size=100000.0,  # Beam search until the end.
          valid_eos_max_logit_delta=valid_eos_max_logit_delta,
          force_eos_in_top_k=force_eos_in_top_k,
      )
      bs_helper = p.Instantiate()

      def InitBeamSearchCallBack(unused_theta, unused_encoder_outputs,
                                 unused_num_hyps_per_beam):
        return py_utils.NestedMap(
            log_probs=tf.zeros([tgt_batch_size, vocab_size]),
            atten_probs=tf.zeros([tgt_batch_size, 0])), py_utils.NestedMap()

      def PreBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                    unused_step_ids, states,
                                    unused_num_hyps_per_beam):
        # Same probs for each id.
        logits = tf.zeros([tgt_batch_size, vocab_size])
        # Except eos is slightly lower prob.
        logits = logits - 1.0 * tf.expand_dims(
            tf.one_hot(p.target_eos_id, vocab_size), 0)
        return py_utils.NestedMap(
            atten_probs=tf.zeros([tgt_batch_size, 0]), log_probs=logits), states

      def PostBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                     unused_new_step_ids, states):
        return states

      encoder_outputs = py_utils.NestedMap(
          seq_lengths=tf.zeros([src_batch_size], dtype=tf.int32))
      theta = py_utils.NestedMap()

      beam_search_output = bs_helper.BeamSearchDecode(
          theta,
          encoder_outputs,
          init_beam_search_state=InitBeamSearchCallBack,
          pre_beam_search_step_callback=PreBeamSearchStepCallback,
          post_beam_search_step_callback=PostBeamSearchStepCallback)

      topk_lens = sess.run(beam_search_output.topk_lens)
      if not force_eos_in_top_k or valid_eos_max_logit_delta < 1.0:
        self.assertAllEqual(topk_lens, np.zeros_like(topk_lens))
      else:
        self.assertAllGreater(topk_lens, 0)

  @parameterized.named_parameters(
      # eos score is too low to terminate
      # 1 hyp terminated at first frame by eoc, and then two other
      # terminated at second frame by eoc
      ('last_chunk_eoc_in_topk', True, True, -10., [1, 2, 2, 1, 2, 2],
       [[-1., -1., -1.], [-1., -1., -1.]]),
      # Not last chunk or not forcing in topk, eoc can not terminate.
      # eos score is low, can not terminate either
      ('last_chunk_eoc_not_in_topk1', True, False, -10., [0, 0, 0, 0, 0, 0],
       [[-0., -0., -0.], [-0., -0., -0.]]),
      ('last_chunk_eoc_not_in_topk2', False, True, -10., [0, 0, 0, 0, 0, 0],
       [[-0., -0., -0.], [-0., -0., -0.]]),
      ('last_chunk_eoc_not_in_topk3', False, False, -10., [0, 0, 0, 0, 0, 0],
       [[-0., -0., -0.], [-0., -0., -0.]]),
      # eos score is high and can terminate
      # 1 hyp terminated at first frame by eos, and then two other
      # terminated at second frame by eos
      ('last_chunk_eoc_not_in_topk_eos_in_top_k', False, False, 1.,
       [1, 2, 2, 1, 2, 2], [[1., 1., 1.], [1., 1., 1.]]),
      # both can terminate at each step, use the lower score.
      ('last_chunk_eoc_in_topk_eos_in_top_k', True, True, 1.,
       [1, 2, 2, 1, 2, 2], [[-1., -1., -1.], [-1., -1., -1.]]),
  )
  def testBeamSearchForceLastChunkEocInTopK(self, is_last_chunk,
                                            force_last_chunk_eoc_in_top_k,
                                            eos_score, expected_topk_lens,
                                            expected_topk_scores):
    with self.session() as sess:
      vocab_size = 30
      tgt_len = 10
      num_hyps_per_beam = 3
      src_batch_size = 2
      tgt_batch_size = src_batch_size * num_hyps_per_beam
      p = beam_search_helper.BeamSearchHelper.Params().Set(
          name='bsh',
          target_eoc_id=0,
          target_seq_len=tgt_len,
          num_hyps_per_beam=num_hyps_per_beam,
          beam_size=100000.0,  # Beam search until the end.
          force_last_chunk_eoc_in_top_k=force_last_chunk_eoc_in_top_k,
      )
      bs_helper = p.Instantiate()

      def InitBeamSearchCallBack(unused_theta, unused_encoder_outputs,
                                 unused_num_hyps_per_beam):
        return py_utils.NestedMap(
            log_probs=tf.zeros([tgt_batch_size, vocab_size]),
            atten_probs=tf.zeros([tgt_batch_size, 0]),
            is_last_chunk=tf.zeros([tgt_batch_size],
                                   tf.bool)), py_utils.NestedMap()

      def PreBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                    unused_step_ids, states,
                                    unused_num_hyps_per_beam):
        # Same probs for each id.
        logits = tf.zeros([tgt_batch_size, vocab_size])
        # Except eoc has slightly lower score.
        logits = logits - 1.0 * tf.expand_dims(
            tf.one_hot(p.target_eoc_id, vocab_size), 0)
        # eos has very low score (can not terminate by eos)
        logits = logits + eos_score * tf.expand_dims(
            tf.one_hot(p.target_eos_id, vocab_size), 0)
        return py_utils.NestedMap(
            atten_probs=tf.zeros([tgt_batch_size, 0]),
            log_probs=logits,
            is_last_chunk=tf.fill([tgt_batch_size],
                                  value=is_last_chunk)), states

      def PostBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                     unused_new_step_ids, states):
        return states

      encoder_outputs = py_utils.NestedMap(
          seq_lengths=tf.zeros([src_batch_size], dtype=tf.int32))
      theta = py_utils.NestedMap()

      beam_search_output = bs_helper.BeamSearchDecode(
          theta,
          encoder_outputs,
          init_beam_search_state=InitBeamSearchCallBack,
          pre_beam_search_step_callback=PreBeamSearchStepCallback,
          post_beam_search_step_callback=PostBeamSearchStepCallback)

      topk_lens, topk_scores = sess.run(
          [beam_search_output.topk_lens, beam_search_output.topk_scores])
      self.assertAllEqual(topk_lens, expected_topk_lens)
      self.assertAllClose(topk_scores, expected_topk_scores, atol=1e-6)

  def testCustomStepIds(self):
    with self.session(use_gpu=False):
      np.random.seed(9384758)
      tf.random.set_seed(8274758)
      vocab_size = 12
      src_len = 5
      tgt_len = 7
      num_hyps_per_beam = 3
      src_batch_size = 2
      tgt_batch_size = src_batch_size * num_hyps_per_beam
      p = beam_search_helper.BeamSearchHelper.Params().Set(
          name='bsh', target_seq_len=tgt_len)
      bs_helper = p.Instantiate()

      def InitBeamSearchState(unused_theta, unused_encoder_outputs,
                              unused_num_hyps_per_beam):
        atten_probs = tf.constant(
            np.random.normal(size=(tgt_batch_size, src_len)), dtype=tf.float32)
        return (py_utils.NestedMap({
            'log_probs': tf.zeros([tgt_batch_size, vocab_size]),
            'atten_probs': atten_probs,
            'step_ids': tf.zeros([tgt_batch_size, 1], dtype=tf.int32)
        }), py_utils.NestedMap({'atten_probs': atten_probs}))

      def PreBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                    unused_step_ids, states,
                                    unused_num_hyps_per_beam):
        atten_probs = tf.identity(states.atten_probs)
        logits = tf.random.normal([tgt_batch_size, vocab_size], seed=8273747)
        return (py_utils.NestedMap({
            'atten_probs': atten_probs,
            'log_probs': logits
        }), states)

      def PostBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                     unused_new_step_ids, states):
        return states

      src_enc = tf.random.normal([src_len, src_batch_size, 8], seed=982774838)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)

      theta = py_utils.NestedMap()
      decoder_output = bs_helper.BeamSearchDecode(theta, encoder_outputs,
                                                  num_hyps_per_beam,
                                                  InitBeamSearchState,
                                                  PreBeamSearchStepCallback,
                                                  PostBeamSearchStepCallback)

      topk_ids, topk_lens, topk_scores = self.evaluate([
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
      self.assertAllEqual(expected_topk_ids, topk_ids.tolist())
      self.assertAllEqual(expected_topk_lens, topk_lens.tolist())
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
          tf.constant(topk_hyps_1),
          tf.reshape(tf.constant(topk_ids_1), [6, -1]),
          tf.reshape(tf.constant(topk_lens_1), [-1]),
          tf.reshape(tf.constant(topk_scores_1), [-1]), None, None)

      topk_scores_2 = [[2., 4.], [-3., 0.]]
      topk_ids_2 = [[[20, 21, 22], [40, 41, 42]], [[30, 31, 33], [0, 0, 0]]]
      topk_lens_2 = [[3, 2], [3, 0]]
      topk_hyps_2 = [['two', 'four'], ['minus three', '']]
      topk_2 = beam_search_helper.BeamSearchDecodeOutput(
          tf.constant(topk_hyps_2),
          tf.reshape(tf.constant(topk_ids_2), [4, -1]),
          tf.reshape(tf.constant(topk_lens_2), [-1]),
          tf.reshape(tf.constant(topk_scores_2), [-1]), None, None)

      topk = beam_search_helper.MergeBeamSearchOutputs(3, [topk_1, topk_2])
      self.assertIsNone(topk.topk_decoded)
      self.assertAllEqual([5., 4., 3., -1., -2., -3.], topk.topk_scores.eval())
      self.assertAllEqual([2, 2, 3, 3, 3, 3], topk.topk_lens.eval())
      self.assertAllEqual([[50, 51, 52], [40, 41, 42], [30, 31, 32],
                           [10, 11, 12], [20, 21, 22], [30, 31, 33]],
                          topk.topk_ids.eval())
      self.assertAllEqual([[b'five', b'four', b'three'],
                           [b'minus one', b'minus two', b'minus three']],
                          topk.topk_hyps.eval())


class GreedySearchHelperTest(test_utils.TestCase):

  def testGreedySearchHelper(self):
    with self.session(use_gpu=False):
      np.random.seed(9384758)
      tf.random.set_seed(8274758)
      vocab_size = 12
      src_len = 5
      tgt_len = 7
      src_batch_size = 2
      tgt_batch_size = src_batch_size
      p = beam_search_helper.GreedySearchHelper.Params().Set(
          name='gsh', target_seq_len=tgt_len)
      gs_helper = p.Instantiate()

      def InitGreedySearchState(unused_theta, unused_encoder_outputs,
                                unused_num_hyps_per_beam):
        atten_probs = tf.constant(
            np.random.normal(size=(tgt_batch_size, src_len)), dtype=tf.float32)
        return (py_utils.NestedMap({
            'log_probs': tf.zeros([tgt_batch_size, vocab_size]),
            'atten_probs': atten_probs,
        }), py_utils.NestedMap({'atten_probs': atten_probs}))

      def PreGreedySearchStepCallback(unused_theta, unused_encoder_outputs,
                                      unused_step_ids, states,
                                      unused_num_hyps_per_beam):
        atten_probs = tf.identity(states.atten_probs)
        logits = tf.random.normal([tgt_batch_size, vocab_size], seed=8273747)
        return (py_utils.NestedMap({
            'atten_probs': atten_probs,
            'log_probs': logits
        }), states)

      def PostGreedySearchStepCallback(unused_theta, unused_encoder_outputs,
                                       unused_new_step_ids, states):
        return states

      src_enc = tf.random.normal([src_len, src_batch_size, 8], seed=982774838)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)

      theta = py_utils.NestedMap()
      (final_hyp_ids, final_hyp_lens,
       final_done_hyps) = gs_helper.GreedySearchDecode(
           theta, encoder_outputs, InitGreedySearchState,
           PreGreedySearchStepCallback, PostGreedySearchStepCallback)

      (final_hyp_ids, final_hyp_lens, final_done_hyps) = self.evaluate(
          [final_hyp_ids, final_hyp_lens, final_done_hyps])

      print(np.array_repr(final_hyp_ids))
      print(np.array_repr(final_hyp_lens))
      print(np.array_repr(final_done_hyps))

      expected_hyp_ids = [[2, 2, 6, 7, 1, 9, 4], [3, 9, 3, 9, 6, 5, 10]]
      expected_hyp_lens = [1, 7]
      expected_done_hyps = [True, False]
      self.assertAllEqual(expected_hyp_ids, final_hyp_ids.tolist())
      self.assertAllEqual(expected_hyp_lens, final_hyp_lens.tolist())
      self.assertAllEqual(expected_done_hyps, final_done_hyps.tolist())


if __name__ == '__main__':
  tf.test.main()
