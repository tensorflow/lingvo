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
"""Tests for beam_search_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import zip
import tensorflow as tf

from google.protobuf import text_format
from lingvo.core.ops import hyps_pb2
from lingvo.core.ops import py_x_ops

_MIN_SCORE = -1e36


class BeamSearchOpTest(tf.test.TestCase):

  def setUp(self):
    super(BeamSearchOpTest, self).setUp()
    np.random.seed(12345)
    tf.set_random_seed(398849988)

  def _testBeamSearchOpHelper(
      self, b_size, num_beams, seq_len, lm_weight, init_best_score, probs,
      init_atten_probs, atten_probs, best_scores_expected, cum_scores_expected,
      scores_expected, hyps_expected, prev_hyps_expected, atten_probs_expected):
    eos_id = 2
    num_classes = 5
    num_hyps_per_beam = b_size / num_beams

    best_scores = tf.zeros([num_beams])
    cumulative_scores = tf.zeros([b_size])
    scores = tf.zeros([seq_len, b_size])
    hyps = tf.zeros([seq_len, b_size], dtype=tf.int32)
    prev_hyps = tf.zeros([seq_len, b_size], dtype=tf.int32)
    done_hyps = tf.as_string(tf.zeros([seq_len, b_size], dtype=tf.int32))
    lm_log_probs = tf.random_uniform([b_size, num_classes])
    best_scores += init_best_score

    for i, prob in enumerate(probs):
      (best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
       atten_probs, done) = py_x_ops.beam_search_step(
           prob,
           init_atten_probs,
           best_scores,
           cumulative_scores,
           scores,
           hyps,
           prev_hyps,
           done_hyps,
           atten_probs, [],
           i,
           lm_log_probs,
           eos_id=eos_id,
           beam_size=3.0,
           num_hyps_per_beam=num_hyps_per_beam,
           valid_eos_max_logit_delta=0.1,
           lm_weight=lm_weight)

    with self.test_session(use_gpu=False) as sess:
      (best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
       atten_probs, done, scores, atten_probs, lm_log_probs) = sess.run([
           best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
           atten_probs, done, scores, atten_probs, lm_log_probs
       ])

    tf.logging.info(np.array_repr(best_scores))
    tf.logging.info(np.array_repr(cumulative_scores))
    tf.logging.info(np.array_repr(scores))
    tf.logging.info(np.array_repr(hyps))
    tf.logging.info(np.array_repr(prev_hyps))
    tf.logging.info(np.array_repr(done_hyps))
    tf.logging.info(np.array_repr(atten_probs))
    tf.logging.info(np.array_repr(done))
    tf.logging.info(np.array_repr(scores))
    tf.logging.info(np.array_repr(atten_probs))
    tf.logging.info(np.array_repr(lm_log_probs))

    self.assertAllClose(best_scores_expected, best_scores)
    self.assertAllClose(cum_scores_expected, cumulative_scores)
    self.assertAllClose(scores_expected, scores)
    self.assertAllClose(hyps_expected, hyps)
    self.assertAllClose(prev_hyps_expected, prev_hyps)
    self.assertAllClose(atten_probs_expected, atten_probs)
    self.assertEqual(False, done)

    return done_hyps

  def testBeamSearchOp(self):
    b_size = 8
    num_beams = 2
    seq_len = 6
    num_classes = 5

    best_scores_expected = [1.895863, 1.54302]
    cum_scores_expected = [
        1.79580379, 1.95926094, 1.66166079, 1.73289895, 1.63867819, 1.5765177,
        1.57203436, 1.21810341
    ]
    scores_expected = [[
        0.89790189, 0.97963047, 0.7637589, 0.75326848, 0.67413247, 0.18659079,
        0.32619762, 0.0323503
    ], [
        0.89790189, 0.97963047, 0.7637589, 0.75326848, 0.96454573, 0.82324922,
        0.67413247, 0.46483493
    ], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.,
                                          0.], [0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0.]]
    hyps_expected = [[3, 4, 0, 0, 1, 1, 4, 3], [3, 4, 0, 0, 1, 1, 1, 3],
                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    prev_hyps_expected = [[0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 4, 3, 0, 3],
                          [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]

    hyp_str_expected = """
    beam_id: 1
    ids: 1
    ids: 2
      scores: 0.186590790749
      scores: 0.712740063667
    atten_vecs {
      prob: 0.689820885658
      prob: 0.216090679169
      prob: 0.40637075901
    }
    atten_vecs {
      prob: 0.824981451035
      prob: 0.774956822395
      prob: 0.944657206535
    }
    """
    atten_probs_expected = [
        [[0.85785675, 0.60858226,
          0.72539818], [0.68982089, 0.21609068,
                        0.40637076], [0.85785675, 0.60858226, 0.72539818],
         [0.68982089, 0.21609068,
          0.40637076], [0.85785675, 0.60858226,
                        0.72539818], [0.68982089, 0.21609068, 0.40637076],
         [0.85785675, 0.60858226,
          0.72539818], [0.68982089, 0.21609068, 0.40637076]],
        [[0.85785675, 0.60858226,
          0.72539818], [0.68982089, 0.21609068, 0.40637076],
         [0.85785675, 0.60858226,
          0.72539818], [0.68982089, 0.21609068,
                        0.40637076], [0.51121557, 0.80525708, 0.73596036], [
                            0.45252705, 0.37489808, 0.12745726
                        ], [0.85785675, 0.60858226, 0.72539818],
         [0.45252705, 0.37489808, 0.12745726]], [
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
         ], [
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
         ], [
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
         ], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
             [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    ]

    scores = [tf.random_uniform([b_size, num_classes])] * 2
    init_atten_probs = tf.random_uniform([b_size, 3])
    atten_probs = tf.zeros([seq_len, b_size, 3])
    done_hyps = self._testBeamSearchOpHelper(
        b_size, num_beams, seq_len, 0., 0, scores, init_atten_probs,
        atten_probs, best_scores_expected, cum_scores_expected, scores_expected,
        hyps_expected, prev_hyps_expected, atten_probs_expected)

    self._SameHyp(hyp_str_expected, done_hyps[1, 5])

  def testBeamSearchOpLMWeight(self):
    b_size = 8
    num_beams = 2
    seq_len = 6
    num_classes = 5

    best_scores_expected = [3.421918, 2.743461]
    cum_scores_expected = [
        3.110763, 2.753090, 3.000604, 2.466151, 2.642948, 2.358385, 2.464650,
        1.938061
    ]
    scores_expected = [[
        1.555382, 1.376545, 1.445222, 1.089605, 1.087567, 0.967116, 0.831732,
        0.856801
    ], [
        1.555382, 1.376545, 1.445222, 1.089605, 1.087567, 1.268780, 1.019428,
        1.081260
    ], [0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.,
                                          0.], [0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0.]]

    # hyps and prev_hyps are same as with no LM weight, except
    # for prev_hyps[1, 7], which is 7 instead of 3.  This is because for this
    # hyp, the weighted global score of the prefix is 0.42 and the unweighted
    # scores for element [7, 0] has a high prob, 0.85.  This doesn't happen
    # without LM weighting because the global score of the prefix would have
    # been only 0.03.  Once the local score is reweighted with the LM,
    # 0.85 gets replaced by 0.54, so the total score is only 0.96.
    hyps_expected = [[3, 4, 0, 0, 1, 1, 4, 3], [3, 4, 0, 0, 1, 1, 3, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    prev_hyps_expected = [[0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 3, 2, 7],
                          [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]

    hyp_str_expected = """
    beam_id: 1
    ids: 1
    ids: 2
    scores: 0.967116
    scores: 0.854650
    atten_vecs {
      prob: 0.689820885658
      prob: 0.216090679169
      prob: 0.40637075901
    }
    atten_vecs {
      prob: 0.824981451035
      prob: 0.774956822395
      prob: 0.944657206535
    }
    """

    # atten_probs are also the same except for [1, 7, :], due to the change
    # in hyps described above.
    atten_probs_expected = [
        [[0.85785675, 0.60858226,
          0.72539818], [0.68982089, 0.21609068,
                        0.40637076], [0.85785675, 0.60858226, 0.72539818],
         [0.68982089, 0.21609068,
          0.40637076], [0.85785675, 0.60858226,
                        0.72539818], [0.68982089, 0.21609068, 0.40637076],
         [0.85785675, 0.60858226,
          0.72539818], [0.68982089, 0.21609068, 0.40637076]],
        [[0.85785675, 0.60858226,
          0.72539818], [0.68982089, 0.21609068, 0.40637076],
         [0.85785675, 0.60858226,
          0.72539818], [0.68982089, 0.21609068,
                        0.40637076], [0.85785675, 0.60858226, 0.72539818], [
                            0.45252705, 0.37489808, 0.12745726
                        ], [0.48289251, 0.35160720, 0.99230611],
         [0.48067939, 0.30471754, 0.39703238]], [
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
         ], [
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
         ], [
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
         ], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
             [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    ]

    scores = [tf.random_uniform([b_size, num_classes])] * 2
    init_atten_probs = tf.random_uniform([b_size, 3])
    atten_probs = tf.zeros([seq_len, b_size, 3])
    done_hyps = self._testBeamSearchOpHelper(
        b_size, num_beams, seq_len, 1.0, 0, scores, init_atten_probs,
        atten_probs, best_scores_expected, cum_scores_expected, scores_expected,
        hyps_expected, prev_hyps_expected, atten_probs_expected)

    self._SameHyp(hyp_str_expected, done_hyps[1, 5])

  # The following 3 tests, test each step of this decoding tree.
  # Test that beam search finds the most probable sequence.
  # These probabilities represent the following search
  #
  #               G0 (0)
  #                  / \
  #                /     \
  #              /         \
  #            /             \
  #         0(0.6)          1(0.4)
  #           / \            / \
  #          /   \          /   \
  #         /     \        /     \
  #     0(0.55) 1(0.45) 0(0.05) 1(0.95)
  #
  # and these decoding probabilities
  # 000 - 0.6 * 0.55 = 0.33
  # 001 - 0.6 * 0.45 = 0.27
  # 010 - 0.4 * 0.05 = 0.02
  # 011 - 0.4 * 0.95 = 0.38
  #
  # Greedy would decode 000 since the first 0 is the most probable, but beam
  # should decode 011 since it's the highest probability then followed by 000.
  def _test_single_step_small_vocab_3(self):
    b_size = 2
    num_beams = 1
    seq_len = 3
    lm_weight = 0.

    probs = [np.log([[0.6, 0.4, 0.0000001], [0.6, 0.4, 0.0000001]])]
    done_hyps = self._testBeamSearchOpHelper(
        b_size,
        num_beams,
        seq_len,
        lm_weight,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([b_size, 0]),
        atten_probs=np.zeros([seq_len, b_size, 0]),
        best_scores_expected=[_MIN_SCORE],
        cum_scores_expected=np.log([0.6, 0.4]),
        scores_expected=[np.log([0.6, 0.4]), [0, 0], [0, 0]],
        hyps_expected=[[0, 1], [0, 0], [0, 0]],
        prev_hyps_expected=[[0, 0], [0, 0], [0, 0]],
        atten_probs_expected=np.zeros([seq_len, b_size, 0]))

    np.testing.assert_array_equal([['0', '0'], ['0', '0'], ['0', '0']],
                                  done_hyps)

  def test_two_steps_small_vocab_3(self):
    b_size = 2
    num_beams = 1
    seq_len = 3
    lm_weight = 0.

    probs = [
        np.log([[0.6, 0.4, 0.0000001], [0.6, 0.4, 0.0000001]]),
        np.log([[0.55, 0.45, 0.0000001], [0.05, 0.95, 0.0000001]]),
    ]
    done_hyps = self._testBeamSearchOpHelper(
        b_size,
        num_beams,
        seq_len,
        lm_weight,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([b_size, 0]),
        atten_probs=np.zeros([seq_len, b_size, 0]),
        best_scores_expected=[_MIN_SCORE],
        # Note, probabilites are swapped due to beams being swapped.
        cum_scores_expected=np.log([0.4 * 0.95, 0.6 * 0.55]),
        scores_expected=[np.log([0.6, 0.4]),
                         np.log([0.95, 0.55]), [0, 0]],
        hyps_expected=[[0, 1], [1, 0], [0, 0]],
        prev_hyps_expected=[[0, 0], [1, 0], [0, 0]],
        atten_probs_expected=np.zeros([seq_len, b_size, 0]))

    np.testing.assert_array_equal([['0', '0'], ['0', '0'], ['0', '0']],
                                  done_hyps)

  def test_three_steps_eos(self):
    b_size = 2
    num_beams = 1
    seq_len = 3
    lm_weight = 0.

    probs = [
        np.log([[0.6, 0.4, 0.0000001], [0.6, 0.4, 0.0000001]]),
        np.log([[0.55, 0.45, 0.0000001], [0.05, 0.95, 0.0000001]]),
        # Finish the beams with EOS
        np.log([[0.05, 0.05, 0.9], [0.05, 0.05, 0.9]]),
    ]

    done_hyps = self._testBeamSearchOpHelper(
        b_size,
        num_beams,
        seq_len,
        lm_weight,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([b_size, 0]),
        atten_probs=np.zeros([seq_len, b_size, 0]),
        best_scores_expected=np.log([0.4 * 0.95 * 0.9]),
        cum_scores_expected=np.log([0.4 * 0.95 * 0.05, 0.4 * 0.95 * 0.05]),
        scores_expected=[
            np.log([0.6, 0.4]),
            np.log([0.95, 0.55]),
            np.log([0.05, 0.05])
        ],
        hyps_expected=[[0, 1], [1, 0], [0, 1]],
        prev_hyps_expected=[[0, 0], [1, 0], [0, 0]],
        atten_probs_expected=np.zeros([seq_len, b_size, 0]))

    expected_for_beam_0 = """
      beam_id: 0
      ids: 1
      ids: 1
      ids: 2
      scores: -0.916290700436  # = log 0.4
      scores: -0.0512933060527 # = log 0.95
      scores: -0.105360545218  # = log 0.9
      atten_vecs {
      }
      atten_vecs {
      }
      atten_vecs {
      }
      """

    expected_for_beam_1 = """
      beam_id: 0
      ids: 0
      ids: 0
      ids: 2
      scores: -0.510825574398  # = log 0.6
      scores: -0.597836971283  # = log 0.55
      scores: -0.105360545218  # = log 0.9
      atten_vecs {
      }
      atten_vecs {
      }
      atten_vecs {
      }
      """

    self._SameHyp(expected_for_beam_0, done_hyps[2, 0])
    self._SameHyp(expected_for_beam_1, done_hyps[2, 1])

  def _SameHyp(self, expected_hyp_str, real_serialized_hyp):
    hyp1 = hyps_pb2.Hypothesis()
    text_format.Merge(expected_hyp_str, hyp1)
    hyp2 = hyps_pb2.Hypothesis()
    hyp2.ParseFromString(real_serialized_hyp)

    self.assertEqual(hyp1.beam_id, hyp2.beam_id)
    self.assertEqual(hyp1.ids, hyp2.ids)
    self.assertEqual(hyp1.normalized_score, hyp2.normalized_score)
    self.assertAllClose(hyp1.scores, hyp2.scores)
    self.assertEqual(len(hyp1.atten_vecs), len(hyp2.atten_vecs))
    for av1, av2 in zip(hyp1.atten_vecs, hyp2.atten_vecs):
      self.assertAllClose(av1.prob, av2.prob)

  def testTopKTerminatedHypsOp(self):
    with self.test_session(use_gpu=False) as sess:
      b_size = 8
      num_beams = 2
      num_hyps_per_beam = b_size / num_beams
      seq_len = 6
      scores = tf.random_uniform([b_size, 5])
      atten_probs = tf.random_uniform([b_size, 3])
      src_seq_lengths = [3, 3]
      best_scores = tf.zeros([num_beams])
      cumulative_scores = tf.zeros([b_size])
      in_scores = tf.zeros([seq_len, b_size])
      in_hyps = tf.zeros([seq_len, b_size], dtype=tf.int32)
      in_prev_hyps = tf.zeros([seq_len, b_size], dtype=tf.int32)
      in_done_hyps = tf.as_string(tf.zeros([seq_len, b_size], dtype=tf.int32))
      in_atten_probs = tf.zeros([seq_len, b_size, 3])

      (out_best_scores_0, out_cumulative_scores_0, out_scores_0, out_hyps_0,
       out_prev_hyps_0, out_done_hyps_0, out_atten_probs_0,
       _) = py_x_ops.beam_search_step(
           scores,
           atten_probs,
           best_scores,
           cumulative_scores,
           in_scores,
           in_hyps,
           in_prev_hyps,
           in_done_hyps,
           in_atten_probs, [],
           0, [],
           eos_id=2,
           beam_size=3.0,
           num_hyps_per_beam=num_hyps_per_beam)

      outputs = py_x_ops.beam_search_step(
          scores,
          atten_probs,
          out_best_scores_0,
          out_cumulative_scores_0,
          out_scores_0,
          out_hyps_0,
          out_prev_hyps_0,
          out_done_hyps_0,
          out_atten_probs_0, [],
          1, [],
          eos_id=2,
          beam_size=3.0,
          num_hyps_per_beam=num_hyps_per_beam)

      # Get the topk terminated hyps.
      in_done_hyps = outputs[5]
      topk_hyps = py_x_ops.top_k_terminated_hyps(
          in_done_hyps,
          src_seq_lengths,
          k=2,
          num_hyps_per_beam=num_hyps_per_beam,
          length_normalization=0.2,
          coverage_penalty=0.2,
          target_seq_length_ratio=1.0)
      seq_ids, seq_lens, seq_scores = py_x_ops.unpack_hyp(
          tf.reshape(topk_hyps, [-1]), max_seq_length=5)

      k1, k2, k3, k4 = sess.run([topk_hyps, seq_ids, seq_lens, seq_scores])
      print(np.array_repr(k1))
      assert k1.size == 4

      expected_top1_for_beam_0 = """
      beam_id: 0
      ids: 3
      ids: 2
      scores: 0.897901892662
      scores: 0.997961401939
      atten_vecs {
        prob: 0.857856750488
        prob: 0.608582258224
        prob: 0.725398182869
      }
      atten_vecs {
        prob: 0.857856750488
        prob: 0.608582258224
        prob: 0.725398182869
      }
      normalized_score: 1.35659193993
      """
      expected_top2_for_beam_1 = """
      beam_id: 1
      ids: 0
      ids: 2
      scores: 0.753268480301
      scores: 0.789751410484
      atten_vecs {
        prob: 0.689820885658
        prob: 0.216090679169
        prob: 0.40637075901
      }
      atten_vecs {
        prob: 0.452527046204
        prob: 0.374898076057
        prob: 0.127457261086
      }
      normalized_score: 1.02671170235
      """
      self._SameHyp(expected_top1_for_beam_0, k1[0, 0])
      self._SameHyp(expected_top2_for_beam_1, k1[1, 1])

      self.assertAllClose(
          k2,
          [[3, 2, 0, 0, 0], [0, 2, 0, 0, 0], [4, 2, 0, 0, 0], [0, 2, 0, 0, 0]])
      self.assertAllClose(k3, [2, 2, 2, 2])
      self.assertAllClose(k4, [1.35659194, 1.02759778, 1.21130753, 1.0267117])


if __name__ == '__main__':
  tf.test.main()
