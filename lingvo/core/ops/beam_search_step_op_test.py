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
"""Tests for beam_search_op."""

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import ops
from lingvo.core import test_utils
from lingvo.core.ops import hyps_pb2
import numpy as np

from google.protobuf import text_format

_MIN_SCORE = -1e36


class BeamSearchOpTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(12345)
    tf.random.set_seed(398849988)

  def _runBeamSearchOpHelper(self,
                             hyp_size,
                             num_beams,
                             seq_len,
                             init_best_score,
                             probs,
                             init_atten_probs,
                             atten_probs,
                             beam_size=3.0,
                             ensure_full_beam=False,
                             force_eos_in_last_step=False,
                             local_eos_threshold=-100.0,
                             independence=True,
                             use_v2=True,
                             atten_vecs_in_hypothesis_protos=True,
                             merge_paths=False,
                             is_last_chunk=None,
                             valid_eos_max_logit_delta=0.1):
    eos_id = 2
    num_hyps_per_beam = hyp_size / num_beams

    best_scores = tf.zeros([num_beams])
    cumulative_scores = tf.zeros([hyp_size])
    scores = tf.zeros([seq_len, hyp_size])
    hyps = tf.zeros([seq_len, hyp_size], dtype=tf.int32)
    prev_hyps = tf.zeros([seq_len, hyp_size], dtype=tf.int32)
    done_hyps = tf.constant('', shape=[seq_len, hyp_size], dtype=tf.string)
    best_scores += init_best_score
    beam_done = tf.zeros([num_beams], dtype=tf.bool)
    if is_last_chunk is None:
      is_last_chunk = (
          tf.ones([seq_len, hyp_size], dtype=tf.bool)
          if merge_paths else tf.zeros([seq_len, hyp_size], dtype=tf.bool))

    for i, prob in enumerate(probs):
      if use_v2:
        (best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
         atten_probs, beam_done, done) = ops.beam_search_step(
             prob,
             init_atten_probs,
             best_scores,
             cumulative_scores,
             scores,
             hyps,
             prev_hyps,
             done_hyps,
             atten_probs,
             beam_done,
             is_last_chunk[i],
             i,
             eos_id=eos_id,
             eoc_id=0 if merge_paths else -1,
             beam_size=beam_size,
             ensure_full_beam=ensure_full_beam,
             num_hyps_per_beam=num_hyps_per_beam,
             valid_eos_max_logit_delta=valid_eos_max_logit_delta,
             force_eos_in_last_step=force_eos_in_last_step,
             local_eos_threshold=local_eos_threshold,
             merge_paths=merge_paths,
             beam_independence=independence,
             atten_vecs_in_hypothesis_protos=atten_vecs_in_hypothesis_protos)
      else:
        (best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
         atten_probs, done) = ops.beam_search_step_deprecated(
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
             eos_id=eos_id,
             beam_size=beam_size,
             ensure_full_beam=ensure_full_beam,
             num_hyps_per_beam=num_hyps_per_beam,
             valid_eos_max_logit_delta=0.1,
             force_eos_in_last_step=force_eos_in_last_step,
             local_eos_threshold=local_eos_threshold)

    with self.session(use_gpu=False):
      (best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
       atten_probs, done, beam_done) = self.evaluate([
           best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
           atten_probs, done, beam_done
       ])

    return (best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
            atten_probs, done, beam_done)

  def _testBeamSearchOpHelper(self,
                              hyp_size,
                              num_beams,
                              seq_len,
                              init_best_score,
                              probs,
                              init_atten_probs,
                              atten_probs,
                              best_scores_expected,
                              cum_scores_expected,
                              scores_expected,
                              hyps_expected,
                              prev_hyps_expected,
                              atten_probs_expected,
                              force_eos_in_last_step=False,
                              local_eos_threshold=-100.0,
                              use_v2=True,
                              atten_vecs_in_hypothesis_protos=True):

    (best_scores, cumulative_scores, scores, hyps, prev_hyps, done_hyps,
     atten_probs, done, beam_done) = self._runBeamSearchOpHelper(
         hyp_size,
         num_beams,
         seq_len,
         init_best_score,
         probs,
         init_atten_probs,
         atten_probs,
         force_eos_in_last_step=force_eos_in_last_step,
         local_eos_threshold=local_eos_threshold,
         use_v2=use_v2,
         atten_vecs_in_hypothesis_protos=atten_vecs_in_hypothesis_protos)

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

    self.assertAllClose(best_scores_expected, best_scores)
    self.assertAllClose(cum_scores_expected, cumulative_scores)
    self.assertAllClose(scores_expected, scores)
    self.assertAllClose(hyps_expected, hyps)
    self.assertAllClose(prev_hyps_expected, prev_hyps)
    self.assertAllClose(atten_probs_expected, atten_probs)

    self.assertEqual(False, done)
    expected_beam_done = np.array([False] * num_beams)
    self.assertAllEqual(expected_beam_done, beam_done)

    return done_hyps

  @parameterized.parameters((False, True), (True, True), (True, False))
  def testBeamSearchOp(self, use_v2, atten_vecs_in_hypothesis_protos):
    hyp_size = 8
    num_beams = 2
    seq_len = 6
    num_classes = 5

    best_scores_expected = [1.769434, 1.640316]
    cum_scores_expected = [
        1.823942, 1.609159, 1.610366, 1.454234, 1.348811, 1.3167, 1.346274,
        1.045735
    ]
    scores_expected = [
        [
            0.86230338, 0.84442794, 0.45372832, 0.38127339, 0.42067075,
            0.25818801, 0.38612545, 0.18693292
        ],
        [
            0.96163845, 0.76473117, 0.74806261, 0.60980642, 0.9281404,
            0.47227204, 0.89254606, 0.20130682
        ],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ]
    hyps_expected = [[1, 0, 0, 3, 4, 1, 3, 4], [1, 4, 4, 1, 1, 3, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    prev_hyps_expected = [[0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 4, 1, 2, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]

    hyp_str_expected = """
    beam_id: 1
    ids: 1
    ids: 2
    scores: 0.25818801
    scores: 0.65319967
    """
    if atten_vecs_in_hypothesis_protos:
      hyp_str_expected += """
      atten_vecs {
        prob: 0.38612545
        prob: 0.42067075
        prob: 0.84442794
      }
      atten_vecs {
        prob: 0.45298624
        prob: 0.53518069
        prob: 0.57700801
      }
      """
    atten_probs_expected = [
        [
            [0.45372832, 0.86230338, 0.65504861],
            [0.38612545, 0.42067075, 0.84442794],
            [0.45372832, 0.86230338, 0.65504861],
            [0.38612545, 0.42067075, 0.84442794],
            [0.45372832, 0.86230338, 0.65504861],
            [0.38612545, 0.42067075, 0.84442794],
            [0.45372832, 0.86230338, 0.65504861],
            [0.38612545, 0.42067075, 0.84442794],
        ],
        [
            [0.45372832, 0.86230338, 0.65504861],
            [0.38612545, 0.42067075, 0.84442794],
            [0.45372832, 0.86230338, 0.65504861],
            [0.38612545, 0.42067075, 0.84442794],
            [0.0532794, 0.53777719, 0.07609642],
            [0.38612545, 0.42067075, 0.84442794],
            [0.25818801, 0.03645897, 0.38127339],
            [0.38612545, 0.42067075, 0.84442794],
        ],
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
         [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    ]

    scores = [
        tf.random.uniform([hyp_size, num_classes], seed=12345),
        tf.random.uniform([hyp_size, num_classes], seed=12346),
    ]
    init_atten_probs = tf.random.uniform([hyp_size, 3], seed=12345)
    atten_probs = tf.zeros([seq_len, hyp_size, 3])
    done_hyps = self._testBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        0.,
        scores,
        init_atten_probs,
        atten_probs,
        best_scores_expected,
        cum_scores_expected,
        scores_expected,
        hyps_expected,
        prev_hyps_expected,
        atten_probs_expected,
        use_v2=use_v2,
        atten_vecs_in_hypothesis_protos=atten_vecs_in_hypothesis_protos)

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
  @parameterized.parameters(False, True)
  def _test_single_step_small_vocab_3(self, use_v2):
    hyp_size = 2
    num_beams = 1
    seq_len = 3

    probs = [np.log([[0.6, 0.4, 0.0000001], [0.6, 0.4, 0.0000001]])]
    done_hyps = self._testBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        best_scores_expected=[_MIN_SCORE],
        cum_scores_expected=np.log([0.6, 0.4]),
        scores_expected=[np.log([0.6, 0.4]), [0, 0], [0, 0]],
        hyps_expected=[[0, 1], [0, 0], [0, 0]],
        prev_hyps_expected=[[0, 0], [0, 0], [0, 0]],
        atten_probs_expected=np.zeros([seq_len, hyp_size, 0]),
        use_v2=use_v2)

    np.testing.assert_array_equal([['0', '0'], ['0', '0'], ['0', '0']],
                                  done_hyps)

  @parameterized.parameters(False, True)
  def test_two_steps_small_vocab_3(self, use_v2):
    hyp_size = 2
    num_beams = 1
    seq_len = 3

    probs = [
        np.log([[0.6, 0.4, 0.0000001], [0.6, 0.4, 0.0000001]]),
        np.log([[0.55, 0.45, 0.0000001], [0.05, 0.95, 0.0000001]]),
    ]
    done_hyps = self._testBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        best_scores_expected=[_MIN_SCORE],
        # Note, probabilites are swapped due to beams being swapped.
        cum_scores_expected=np.log([0.4 * 0.95, 0.6 * 0.55]),
        scores_expected=[np.log([0.6, 0.4]),
                         np.log([0.95, 0.55]), [0, 0]],
        hyps_expected=[[0, 1], [1, 0], [0, 0]],
        prev_hyps_expected=[[0, 0], [1, 0], [0, 0]],
        atten_probs_expected=np.zeros([seq_len, hyp_size, 0]),
        use_v2=use_v2)

    np.testing.assert_array_equal([[b'', b''], [b'', b''], [b'', b'']],
                                  done_hyps)

  @parameterized.parameters(False, True)
  def test_three_steps_eos(self, use_v2):
    hyp_size = 2
    num_beams = 1
    seq_len = 3

    probs = [
        np.log([[0.6, 0.4, 0.0000001], [0.6, 0.4, 0.0000001]]),
        np.log([[0.55, 0.45, 0.0000001], [0.05, 0.95, 0.0000001]]),
        # Finish the beams with EOS
        np.log([[0.05, 0.05, 0.9], [0.05, 0.05, 0.9]]),
    ]

    done_hyps = self._testBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        best_scores_expected=np.log([0.4 * 0.95 * 0.9]),
        cum_scores_expected=np.log([0.4 * 0.95 * 0.05, 0.4 * 0.95 * 0.05]),
        scores_expected=[
            np.log([0.6, 0.4]),
            np.log([0.95, 0.55]),
            np.log([0.05, 0.05])
        ],
        hyps_expected=[[0, 1], [1, 0], [0, 1]],
        prev_hyps_expected=[[0, 0], [1, 0], [0, 0]],
        atten_probs_expected=np.zeros([seq_len, hyp_size, 0]),
        use_v2=use_v2)

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

  @parameterized.parameters(False, True)
  def test_three_steps_force_eos(self, use_v2):
    hyp_size = 2
    num_beams = 1
    seq_len = 3

    probs = [
        np.log([[0.6, 0.4, 0.0000001], [0.6, 0.4, 0.0000001]]),
        np.log([[0.55, 0.45, 0.0000001], [0.05, 0.95, 0.0000001]]),
        # EOS probability is still very low, so unless it is forced it will
        # not be in the beam.
        np.log([[0.45, 0.44, 0.01], [0.5, 0.5, 0.01]]),
    ]

    # Set expected values
    cum_scores_expected = np.log([0.4 * 0.95 * 0.45, 0.4 * 0.95 * 0.44])
    scores_expected = [
        np.log([0.6, 0.4]),
        np.log([0.95, 0.55]),
        np.log([0.45, 0.44])
    ]
    hyps_expected = [[0, 1], [1, 0], [0, 1]]
    prev_hyps_expected = [[0, 0], [1, 0], [0, 0]]

    # If force EOS is false, the we get empty hyps after beam search.
    done_hyps = self._testBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        best_scores_expected=[_MIN_SCORE],
        cum_scores_expected=cum_scores_expected,
        scores_expected=scores_expected,
        hyps_expected=hyps_expected,
        prev_hyps_expected=prev_hyps_expected,
        atten_probs_expected=np.zeros([seq_len, hyp_size, 0]),
        force_eos_in_last_step=False)
    np.testing.assert_array_equal([[b'', b''], [b'', b''], [b'', b'']],
                                  done_hyps)

    # If force eos is true, we get valid results as in test_three_step_eos,
    # but with lower probabilities (because of lower eos probs).
    done_hyps = self._testBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        best_scores_expected=np.log([0.4 * 0.95 * 0.01]),
        cum_scores_expected=cum_scores_expected,
        scores_expected=scores_expected,
        hyps_expected=hyps_expected,
        prev_hyps_expected=prev_hyps_expected,
        atten_probs_expected=np.zeros([seq_len, hyp_size, 0]),
        force_eos_in_last_step=True,
        use_v2=use_v2)

    expected_for_beam_0 = """
      beam_id: 0
      ids: 1
      ids: 1
      ids: 2
      scores: -0.916290700436  # = log 0.4
      scores: -0.0512933060527 # = log 0.95
      scores: -4.605170185988  # = log 0.01
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
      scores: -4.605170185988  # = log 0.01
      atten_vecs {
      }
      atten_vecs {
      }
      atten_vecs {
      }
      """

    self._SameHyp(expected_for_beam_0, done_hyps[2, 0])
    self._SameHyp(expected_for_beam_1, done_hyps[2, 1])

  def testBeamSearchOpV2SmokeTest(self):
    hyp_size = 2
    num_beams = 1
    seq_len = 3
    probs = [
        np.log([[0.6, 0.4, 0.0000001], [0.6, 0.4, 0.0000001]]),
    ]
    results = self._runBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]))
    expected_beam_done = np.array([False])
    self.assertAllEqual(results[-1], expected_beam_done)

  @parameterized.parameters(False, True)
  def testBeamSearchOpV2ThreeSteps(self, independence):
    """Similar setup as test_three_steps_eos above but for op V2."""
    hyp_size = 2
    num_beams = 1
    seq_len = 4
    small_prob = 1e-7
    probs = [
        np.log([[0.6, 0.4, small_prob], [0.6, 0.4, small_prob]]),
        np.log([[0.55, 0.45, small_prob], [0.05, 0.95, small_prob]]),
        # We insert id=1 here to make the decoded output with length 4.
        np.log([[small_prob, 1.0, small_prob], [small_prob, 1.0, small_prob]]),
        np.log([[0.05, 0.05, 0.9], [0.05, 0.05, 0.9]]),
    ]
    results = self._runBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        independence=independence,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]))
    done_hyps = results[-4]
    hyp = hyps_pb2.Hypothesis()
    hyp.ParseFromString(done_hyps[3, 0])
    self.assertAllEqual(0, hyp.beam_id)
    self.assertAllEqual([1, 1, 1, 2], hyp.ids)
    # [log(0.4), log(0.95), log(1), log(0.9)]
    self.assertAllClose([-0.91629070, -0.05129331, 0., -0.10536052], hyp.scores)
    hyp.ParseFromString(done_hyps[3, 1])
    self.assertAllEqual(0, hyp.beam_id)
    self.assertAllEqual([0, 0, 1, 2], hyp.ids)
    # [log(0.6), log(0.55), log(1), log(0.9)]
    self.assertAllClose([-0.51082557, -0.59783697, 0., -0.10536052], hyp.scores)

  @parameterized.parameters(False, True)
  def testBeamSearchOpV2Independence(self, independence):
    """Test for V2 op's beam independence mode.

    The setup is the following: we have two beams and hyp_per_beam=2.

    Beam 0 has the same probablity setup as test_three_steps_eos above,
    except that we add a step by inserting id=1 at t=2 so that it finishes
    decoding in 4 steps, to [1, 1, 1, 2] (best) and [0, 0, 1, 2] (second best).

    Beam 1 encounters a terminated hyp at t=1: [1, 2]. But it also contains
    longer terminated hyps at t=3: [1,1,1,2] and [0, 0, 1, 2].

    We verify that under beam independence mode, for beam 1 the longer
    terminated hyps are not present. We achieve this by setting beam_size to
    be very small for force beam_done to True for beam 1.

    Args:
      independence: whether beam independence mode is enabled.
    """
    hyp_size = 4
    num_beams = 2
    seq_len = 4

    small_prob = 1e-7
    probs = [
        np.log([[0.6, 0.4, small_prob], [0.2, 0.8, small_prob],
                [0.6, 0.4, small_prob], [0.2, 0.8, small_prob]]),
        np.log([[0.55, 0.45, small_prob], [small_prob, 0.3, 0.6],
                [0.05, 0.95, small_prob], [0.05, 0.9, 0.05]]),
        # We insert id=1 here to make the decoded output with length 4.
        np.log([[small_prob, 1.0, small_prob], [small_prob, 1.0, small_prob],
                [small_prob, 1.0, small_prob], [small_prob, 1.0, small_prob]]),
        np.log([[0.05, 0.05, 0.9], [0.05, 0.05, 0.9], [0.05, 0.05, 0.9],
                [0.05, 0.05, 0.9]]),
    ]
    results = self._runBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        beam_size=0.1,
        independence=independence)
    done_hyps = results[-4]
    self.assertAllEqual(done_hyps.shape, [4, 4])

    hyp = hyps_pb2.Hypothesis()
    hyp.ParseFromString(done_hyps[1, 1])
    self.assertAllEqual(1, hyp.beam_id)
    self.assertAllEqual([1, 2], hyp.ids)
    # [log(0.8), log(0.6)]
    self.assertAllClose([-0.223144, -0.510826], hyp.scores)

    if not independence:
      # For beam 1, we have 3 terminated hyps when not under beam independence
      # mode.
      hyp.ParseFromString(done_hyps[3, 1])
      self.assertAllEqual(1, hyp.beam_id)
      self.assertAllEqual([1, 1, 1, 2], hyp.ids)
      # [log(0.8), log(0.3), log(1), log(0.9)]
      self.assertAllClose([-0.22314355, -1.20397282, 0., -0.10536052],
                          hyp.scores)
      hyp.ParseFromString(done_hyps[3, 3])
      self.assertAllEqual([0, 1, 1, 2], hyp.ids)
      self.assertAllEqual(1, hyp.beam_id)
      # [log(0.2), log(0.9), log(1), log(0.9)]
      self.assertAllClose([-1.609438, -0.105361, 0., -0.105361], hyp.scores)
    else:
      # Under beam independence mode, no further terminated hyps are found.
      for step_t in [2, 3]:
        for hyp_idx in [1, 3]:
          hyp.ParseFromString(done_hyps[step_t, hyp_idx])
          self.assertEmpty(hyp.ids)

    # For beam 0, we have 2 terminated hyps, similar to in test_three_steps_eos.
    hyp.ParseFromString(done_hyps[3, 0])
    self.assertAllEqual(0, hyp.beam_id)
    self.assertAllEqual([1, 1, 1, 2], hyp.ids)
    # [log(0.4), log(0.95), log(1), log(0.9)]
    self.assertAllClose([-0.91629070, -0.05129331, 0., -0.10536052], hyp.scores)
    hyp.ParseFromString(done_hyps[3, 2])
    self.assertAllEqual(0, hyp.beam_id)
    self.assertAllEqual([0, 0, 1, 2], hyp.ids)
    # [log(0.6), log(0.55), log(1), log(0.9)]
    self.assertAllClose([-0.51082557, -0.59783697, 0., -0.10536052], hyp.scores)

    expected_beam_done = np.array([True, True])
    self.assertAllEqual(results[-1], expected_beam_done)
    for steps in range(1, 4):
      # We verify that beam_done[1] is True after 2 steps (but it has no affect
      # when indpendence=False).
      results_at_steps = self._runBeamSearchOpHelper(
          hyp_size,
          num_beams,
          seq_len,
          _MIN_SCORE,
          probs[:steps],
          init_atten_probs=tf.zeros([hyp_size, 0]),
          atten_probs=np.zeros([seq_len, hyp_size, 0]),
          beam_size=0.1,
          independence=False)
      expected_beam_done = np.array([False, steps >= 2])
      self.assertAllEqual(results_at_steps[-1], expected_beam_done)

  def _testBeamSearchStoppingHelper(self,
                                    beam_size,
                                    ensure_full_beam,
                                    local_eos_threshold=-100,
                                    use_v2=True):
    hyp_size = 2
    num_beams = 1
    seq_len = 3
    probs = [
        # Only finish one beam with EOS.
        np.log([[0.05, 0.05, 0.9], [0.05, 0.9, 0.05]]),
    ]

    results = self._runBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        beam_size=beam_size,
        ensure_full_beam=ensure_full_beam,
        local_eos_threshold=local_eos_threshold,
        use_v2=use_v2)
    all_done = results[7]
    if use_v2:
      self.assertAllEqual([all_done], results[8])
    return all_done

  @parameterized.parameters(False, True)
  def test_beam_size_large(self, use_v2):
    # With default beam size, we are not yet all done, because we still have an
    # active hyp within 3.0 of best done hyp.
    all_done = self._testBeamSearchStoppingHelper(3.0, False, use_v2=use_v2)
    self.assertEqual(False, all_done)

  @parameterized.parameters(False, True)
  def test_beam_size_small(self, use_v2):
    # With small beam size, we are all done, because the active hyp is not
    # within such a narrow margin of best done hyp.
    all_done = self._testBeamSearchStoppingHelper(0.1, False, use_v2=use_v2)
    self.assertEqual(True, all_done)

  @parameterized.parameters(False, True)
  def test_ensure_full_beam(self, use_v2):
    # With small beam size and ensure_full_beam, we are _not_ yet done,
    # because we require to have two done hyps before stopping, regardless of
    # beam size.
    all_done = self._testBeamSearchStoppingHelper(0.1, True, use_v2=use_v2)
    self.assertEqual(False, all_done)

  @parameterized.parameters(False, True)
  def test_small_eos_threshold(self, use_v2):
    # With a small eos_threshold, we are done because the active hyp produced,
    # </s>, independent of small beam size.
    all_done = self._testBeamSearchStoppingHelper(
        0.1, False, -100.0, use_v2=use_v2)
    self.assertTrue(all_done)

  @parameterized.parameters(False, True)
  def test_large_eos_threshold(self, use_v2):
    # With larger eos_threshold, we are _not_ yet done, because we do not hit
    # </s> criteria we we require to have two done hyps before stopping,
    # regardless of beam size.
    all_done = self._testBeamSearchStoppingHelper(
        0.1, False, -0.05, use_v2=use_v2)
    self.assertFalse(all_done)

  @parameterized.parameters(False, True)
  def test_ensure_full_beam_more_strict(self, ensure_full_beam):
    hyp_size = 2
    num_beams = 1
    seq_len = 4
    probs = [
        np.log([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]),
        np.log([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]]),
    ]
    common_args = dict(
        hyp_size=hyp_size,
        num_beams=num_beams,
        seq_len=seq_len,
        init_best_score=_MIN_SCORE,
        probs=probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        ensure_full_beam=ensure_full_beam,
        use_v2=True,
    )

    # After two steps, we found 2 terminated hyps.
    # Regardless of p.ensure_full_beam, we are not done because beam_size is
    # large.
    results = self._runBeamSearchOpHelper(
        beam_size=3.0, local_eos_threshold=-1.0, **common_args)
    all_done = results[7]
    self.assertAllEqual([all_done], results[8])
    self.assertFalse(all_done)

    # With a smaller beam_size, we are done.
    results = self._runBeamSearchOpHelper(
        beam_size=0.1, local_eos_threshold=-1.0, **common_args)
    all_done = results[7]
    self.assertTrue(all_done)

    # If we found 3 terminated hyps, we are similarly not done.
    results = self._runBeamSearchOpHelper(
        beam_size=3.0, local_eos_threshold=-100.0, **common_args)
    all_done = results[7]
    self.assertFalse(all_done)

  def test_ensure_full_beam_two_beams(self):
    hyp_size = 4
    num_beams = 2
    seq_len = 3
    # Beam 0: only has found 1 terminated hyp at step 0.
    # Beam 1: plenty of terminated hyps.
    probs = [
        np.log([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8],
                [0.1, 0.1, 0.8]]),
        np.log([[0.1, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.8]]),
    ]

    # After two steps, with strict beam_size, both beams would have been done,
    # except that ensure_full_beam kicks in, so beam 0 is not yet done.
    results = self._runBeamSearchOpHelper(
        hyp_size=hyp_size,
        num_beams=num_beams,
        seq_len=seq_len,
        init_best_score=_MIN_SCORE,
        probs=probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        ensure_full_beam=True,
        use_v2=True,
        beam_size=2.0,
        local_eos_threshold=-1.0)
    beam_done = results[8]
    self.assertAllEqual([False, True], beam_done)

  def test_merge_paths_terminate_with_eoc(self):
    hyp_size = 2
    num_beams = 1
    seq_len = 2

    probs = [
        # [3], [1]
        [[-10., -1., -10., 0.], [-10., -10., -10., -10.]],
        # We find two termianted hyps here.
        # [3, 0] is valid, so is [1, 0].
        [[-0.3, -1.5, -10., -10.], [-0.5, -2., -10., -10.]],
    ]

    results = self._runBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        merge_paths=True,
        valid_eos_max_logit_delta=5.0,
        use_v2=True)
    beam_done = results[-1]
    # We are not done, because [3, 1] is still pretty good.
    self.assertAllEqual([False], beam_done)
    done_hyps = results[5]
    hyp0 = hyps_pb2.Hypothesis()
    hyp0.ParseFromString(done_hyps[1][0])
    hyp1 = hyps_pb2.Hypothesis()
    hyp1.ParseFromString(done_hyps[1][1])
    self.assertAllEqual([3, 0], hyp0.ids)
    self.assertAllClose([-0.15, -0.15], hyp0.scores)
    self.assertAllEqual([1, 0], hyp1.ids)
    self.assertAllClose([-0.75, -0.75], hyp1.scores)

  def test_merge_paths_eoc_versus_eos(self):
    hyp_size = 2
    num_beams = 1
    seq_len = 3

    probs = [
        # [3], [0]
        [[-1., -10., -10., 0.], [-10., -10., -10., -10.]],
        # [3, 3], [3, 0]
        [[-0.6, -10, -10., -0.1], [-.5, -10, -10., -1.31]],
        # score very low on id=1 or 3, so we are done after this.
        # For hyp 0, EOC has higher score; for hyp 1, EOC has lower
        # score.
        [[-2.3, -10, -2.6, -9.7], [-2.4, -10, -2.1, -9.]],
    ]
    is_last_chunk = [
        [False, False],
        [False, False],
        [True, True],
    ]

    results = self._runBeamSearchOpHelper(
        hyp_size,
        num_beams,
        seq_len,
        _MIN_SCORE,
        probs,
        init_atten_probs=tf.zeros([hyp_size, 0]),
        atten_probs=np.zeros([seq_len, hyp_size, 0]),
        is_last_chunk=is_last_chunk,
        merge_paths=True,
        valid_eos_max_logit_delta=5.0,
        use_v2=True)
    beam_done = results[-1]
    # After 2 terminated beams are found, we are done.
    self.assertAllEqual([True], beam_done)
    done_hyps = results[5]
    for step in range(seq_len - 1):
      for i in range(2):
        self.assertEmpty(done_hyps[step][i])
    hyp0 = hyps_pb2.Hypothesis()
    hyp0.ParseFromString(done_hyps[2][0])
    hyp1 = hyps_pb2.Hypothesis()
    hyp1.ParseFromString(done_hyps[2][1])
    # Both EOC and EOS can theoretically terminate the hyp.
    # Note that we take the one with lower score.
    self.assertAllEqual([3, 3, 2], hyp0.ids)
    # `scores` is the per step average of global score: -2.7 = 0 - 0.1 - 2.6
    self.assertAllClose([-.9] * 3, hyp0.scores)
    self.assertAllEqual([3, 0, 0], hyp1.ids)
    # `scores` is the per step average of global score: -3 = 0 - 0.6 - 2.4
    self.assertAllClose([-1.] * 3, hyp1.scores)

  def _SameHyp(self, expected_hyp_str, real_serialized_hyp):
    hyp1 = hyps_pb2.Hypothesis()
    text_format.Parse(expected_hyp_str, hyp1)
    hyp2 = hyps_pb2.Hypothesis()
    hyp2.ParseFromString(real_serialized_hyp)

    self.assertEqual(hyp1.beam_id, hyp2.beam_id)
    self.assertEqual(hyp1.ids, hyp2.ids)
    self.assertNear(hyp1.normalized_score, hyp2.normalized_score, 1e-6)
    self.assertAllClose(hyp1.scores, hyp2.scores)
    self.assertEqual(len(hyp1.atten_vecs), len(hyp2.atten_vecs))
    for av1, av2 in zip(hyp1.atten_vecs, hyp2.atten_vecs):
      self.assertAllClose(av1.prob, av2.prob)

  def testTopKTerminatedHypsOp(self):
    with self.session(use_gpu=False):
      hyp_size = 8
      num_beams = 2
      num_hyps_per_beam = hyp_size / num_beams
      seq_len = 6
      scores = tf.random.uniform([hyp_size, 5], seed=12345)
      atten_probs = tf.random.uniform([hyp_size, 3], seed=12345)
      src_seq_lengths = [3, 3]
      best_scores = tf.zeros([num_beams])
      cumulative_scores = tf.zeros([hyp_size])
      in_scores = tf.zeros([seq_len, hyp_size])
      in_hyps = tf.zeros([seq_len, hyp_size], dtype=tf.int32)
      in_prev_hyps = tf.zeros([seq_len, hyp_size], dtype=tf.int32)
      in_done_hyps = tf.as_string(tf.zeros([seq_len, hyp_size], dtype=tf.int32))
      in_atten_probs = tf.zeros([seq_len, hyp_size, 3])
      beam_done = tf.zeros([num_beams], dtype=tf.bool)

      (out_best_scores_0, out_cumulative_scores_0, out_scores_0, out_hyps_0,
       out_prev_hyps_0, out_done_hyps_0, out_atten_probs_0, beam_done,
       _) = ops.beam_search_step(
           scores,
           atten_probs,
           best_scores,
           cumulative_scores,
           in_scores,
           in_hyps,
           in_prev_hyps,
           in_done_hyps,
           in_atten_probs,
           beam_done, [],
           0,
           eos_id=2,
           beam_size=3.0,
           num_hyps_per_beam=num_hyps_per_beam)

      outputs = ops.beam_search_step(
          scores,
          atten_probs,
          out_best_scores_0,
          out_cumulative_scores_0,
          out_scores_0,
          out_hyps_0,
          out_prev_hyps_0,
          out_done_hyps_0,
          out_atten_probs_0,
          beam_done, [],
          1,
          eos_id=2,
          beam_size=3.0,
          num_hyps_per_beam=num_hyps_per_beam)

      # Get the topk terminated hyps.
      in_done_hyps = outputs[5]
      topk_hyps = ops.top_k_terminated_hyps(
          in_done_hyps,
          src_seq_lengths,
          k=2,
          num_hyps_per_beam=num_hyps_per_beam,
          length_normalization=0.2,
          coverage_penalty=0.2,
          target_seq_length_ratio=1.0)
      seq_ids, seq_lens, seq_scores = ops.unpack_hyp(
          tf.reshape(topk_hyps, [-1]), max_seq_length=5)

      k1, k2, k3, k4 = self.evaluate([topk_hyps, seq_ids, seq_lens, seq_scores])
      self.assertEqual(k1.size, 4)

      expected_top1_for_beam_0 = """
      beam_id: 0
      ids: 1
      ids: 2
      scores: 0.86230338
      scores: 0.65504861
      atten_vecs {
        prob: 0.45372832
        prob: 0.86230338
        prob: 0.65504861
      }
      atten_vecs {
        prob: 0.45372832
        prob: 0.86230338
        prob: 0.65504861
      }
      normalized_score: 1.002714
      """
      expected_top2_for_beam_1 = """
      beam_id: 1
      ids: 3
      ids: 2
      scores: 0.38127339
      scores: 0.57700801
      atten_vecs {
        prob: 0.38612545
        prob: 0.42067075
        prob: 0.84442794
      }
      atten_vecs {
        prob: 0.18693292
        prob: 0.17821217
        prob: 0.66380036
      }
      normalized_score: 0.480028
      """
      self._SameHyp(expected_top1_for_beam_0, k1[0, 0])
      self._SameHyp(expected_top2_for_beam_1, k1[1, 1])

      self.assertAllClose(
          k2,
          [[1, 2, 0, 0, 0], [4, 2, 0, 0, 0], [4, 2, 0, 0, 0], [3, 2, 0, 0, 0]])
      self.assertAllClose(k3, [2, 2, 2, 2])
      self.assertAllClose(k4, [1.002714, 0.684296, 0.522484, 0.480028])


class TopKOpTest(test_utils.TestCase, parameterized.TestCase):

  def testHypsFromBeamSearchOut(self):
    with self.session(use_gpu=False) as sess:
      hyp_size = 4
      num_beams = 2
      num_hyps_per_beam = hyp_size / num_beams
      src_seq_len = 2
      tgt_seq_len = 3
      hyps = [
          [3, 4, 5, 6],
          [7, 8, 9, 10],
          [0, 0, 0, 0],  # unused row
      ]
      prev_hyps = [
          [4, 4, 4, 4],  # unused row
          [2, 3, 0, 1],
          [4, 4, 4, 4],  # unused row
      ]
      done_hyps = tf.ones([tgt_seq_len, hyp_size], dtype=tf.bool)
      scores = tf.zeros([tgt_seq_len, hyp_size], dtype=tf.float32)
      eos_scores = tf.zeros([tgt_seq_len, hyp_size], dtype=tf.float32)
      # hyp ids:      0,          1          2           3
      # step=0: [[ 0.,  1.], [ 2.,  3.], [ 4.,  5.], [ 6.,  7.]]
      # step=1: [[ 8.,  9.], [10., 11.], [12., 13.], [14., 15.]]
      atten_probs = np.reshape(
          np.arange(tgt_seq_len * hyp_size * src_seq_len, dtype=np.float32),
          [tgt_seq_len, hyp_size, src_seq_len])
      eos_atten_probs = tf.ones([tgt_seq_len, hyp_size, src_seq_len],
                                dtype=tf.float32)
      final_done_hyps = ops.hyps_from_beam_search_outs(
          hyps,
          prev_hyps,
          done_hyps,
          scores,
          atten_probs,
          eos_scores,
          eos_atten_probs,
          eos_id=2,
          num_hyps_per_beam=num_hyps_per_beam)
      final_done_hyps = sess.run(final_done_hyps)
    self.assertAllEqual(final_done_hyps.shape, [tgt_seq_len, hyp_size])
    # Focusing on hyps of length 3:
    # hyp 0: last step EOS, step 1 hyp 0, step 0 hyp 2 (prev_hyps[1, 0]=2).
    #        The output seq is [5, 7, 2].
    # hyp 1: step 1 hyp 1, step 0 hyp 3, outputs [6, 8, 2].
    # hyp 2: step 1 hyp 2, step 0 hyp 0, outputs [3, 9, 2].
    # hyp 3: step 1 hyp 3, step 0 hyp 1, outputs [4, 10, 2].
    expected_hyp_ids = [[5, 7, 2], [6, 8, 2], [3, 9, 2], [4, 10, 2]]
    expected_atten_probs = np.array(
        [
            [[4, 5], [8, 9], [1, 1]],
            [[6, 7], [10, 11], [1, 1]],
            [[0, 1], [12, 13], [1, 1]],
            [[2, 3], [14, 15], [1, 1]],
        ],
        dtype=np.float32,
    )
    for hyp_id in range(hyp_size):
      hyp = hyps_pb2.Hypothesis()
      hyp.ParseFromString(final_done_hyps[2, hyp_id])
      self.assertAllEqual(hyp.ids, expected_hyp_ids[hyp_id])
      for j in range(tgt_seq_len):
        self.assertAllClose(hyp.atten_vecs[j].prob,
                            expected_atten_probs[hyp_id][j])

  def _SameHyp(self, hyp1_pb, hyp2_pb):
    hyp1 = hyps_pb2.Hypothesis()
    hyp1.ParseFromString(hyp1_pb)
    hyp2 = hyps_pb2.Hypothesis()
    hyp2.ParseFromString(hyp2_pb)

    self.assertEqual(hyp1.beam_id, hyp2.beam_id)
    self.assertEqual(hyp1.ids, hyp2.ids)
    self.assertNear(hyp1.normalized_score, hyp2.normalized_score, 1e-6)
    self.assertAllClose(hyp1.scores, hyp2.scores)
    self.assertEqual(len(hyp1.atten_vecs), len(hyp2.atten_vecs))
    for av1, av2 in zip(hyp1.atten_vecs, hyp2.atten_vecs):
      self.assertAllClose(av1.prob, av2.prob)

  def testTopKFromBeamSearchOut(self):

    with self.session(use_gpu=False) as sess:
      hyp_size = 6
      num_beams = 2
      num_hyps_per_beam = hyp_size / num_beams
      seq_len = 3
      hyps = [
          [0, 1, 0, 1, 0, 1],
          [1, 0, 0, 1, 1, 0],
          [0, 1, 0, 1, 0, 0],
      ]
      prev_hyps = [
          [0, 1, 2, 3, 4, 5],
          [0, 1, 2, 3, 4, 5],
          [0, 1, 2, 3, 4, 5],
      ]
      done_hyps = tf.ones([seq_len, hyp_size], dtype=tf.bool)
      cumulative_scores = [
          [1., 1., 5., 2., 3., 6.],
          [1., 4., 2., 5., 6., 3.],
          [10., 10., 10., 10., 10., 10.],  # this row doesn't matter.
      ]
      eos_scores = [
          [6., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
      ]
      results = ops.top_k_from_beam_search_outs(
          hyps,
          prev_hyps,
          done_hyps,
          cumulative_scores,
          eos_scores,
          0,  # unused scores
          0,  # unused atten_probs
          0,  # unused eos_atten_probs
          0,  # unused cumulative_atten_probs
          length_normalization=0.0,
          coverage_penalty=0.0,
          num_hyps_per_beam=num_hyps_per_beam,
          max_seq_length=seq_len)
      outs = sess.run(results)
      self.assertAllEqual(outs[0].shape, [hyp_size, seq_len])
      self.assertAllEqual(outs[1].shape, [hyp_size])
      self.assertAllEqual(outs[2].shape, [hyp_size])
      self.assertAllEqual(outs[3].shape, [num_beams, num_hyps_per_beam])

      # For beam 0, the highest scores are:
      # [0, 0] (from eos_scores), [1, 4], [0, 2] (from cumulative_scores)
      # These map to sequences [2], [0, 1, 2], [0, 2].
      #
      # For beam 1, the highest scores are [0, 5], [1, 3], [1, 1] (from
      # cumulative_scores).
      # These map to sequences [1, 2], [1, 1, 2], [1, 0, 2]
      expected_ids = [[2, 0, 0], [0, 1, 2], [0, 2, 0], [1, 2, 0], [1, 1, 2],
                      [1, 0, 2]]
      expected_lengths = [1, 3, 2, 2, 3, 3]
      expected_scores = [6., 6., 5., 6., 5., 4.]
      self.assertAllEqual(outs[0], expected_ids)
      self.assertAllEqual(outs[1], expected_lengths)
      self.assertAllEqual(outs[2], expected_scores)

      results = ops.top_k_from_beam_search_outs(
          hyps,
          prev_hyps,
          done_hyps,
          cumulative_scores,
          eos_scores,
          0,  # unused scores
          0,  # unused atten_probs
          0,  # unused eos_atten_probs
          0,  # unused cumulative_atten_probs
          length_normalization=0.5,
          coverage_penalty=0.0,
          num_hyps_per_beam=num_hyps_per_beam,
          max_seq_length=seq_len)
      outs = sess.run(results)

      def normalize(score, length):
        f = np.power(length + 5.0, 0.5) / np.power(5.0, 0.5)
        return score / f

      expected_normalized_scores = [
          normalize(score, l)
          for l, score in zip(expected_lengths, expected_scores)
      ]
      self.assertAllEqual(outs[0], expected_ids)
      self.assertAllEqual(outs[1], expected_lengths)
      self.assertAllClose(outs[2], expected_normalized_scores)

  @parameterized.named_parameters(
      ('Base', 0., 0.0, 1.0, False),
      ('LengthNorm', .5, 0.0, 1.0, False),
      ('LengthNormHalfAndHyp', 0.5, 0.0, 1.0, True),
      ('LengthNormOneAndHyp', 1., 0.0, 1.0, True),
      ('Coverage', 0.0, 1.0, 1.0, False),
      ('Coverage2', 0.0, .1, 1.3, False),
      ('CoverageAndHyp', 0.0, .1, 1.3, True),
      ('All', 0.5, .2, 1.1, True),
  )
  def testTopKEquivalent(self, length_normalization, coverage_penalty,
                         length_ratio, populate_hyps):
    """Tests that top_k_from_beam_search_outs is indeed equivalent."""
    with self.session(use_gpu=False) as sess:
      hyp_size = 32
      num_beams = 8
      num_hyps_per_beam = hyp_size // num_beams
      seq_len = 10

      hyps = np.random.randint(3, 100, size=[seq_len, hyp_size])
      # We align all the hyps to make cumulative_score easy to compute.
      prev_hyps = np.tile(np.arange(hyp_size), [seq_len, 1])
      done_hyps = np.ones([seq_len, hyp_size], dtype=np.bool)
      scores = np.random.uniform(-0.5, 1, size=[seq_len, hyp_size])
      cumulative_scores = np.cumsum(scores, axis=0)
      eos_scores = np.random.uniform(-0.5, 1, size=[seq_len, hyp_size])
      atten_probs = np.random.uniform(0, .05, size=[seq_len, hyp_size, seq_len])
      eos_atten_probs = np.random.uniform(
          0, .05, size=[seq_len, hyp_size, seq_len])
      cum_atten_probs = np.cumsum(atten_probs, axis=0)
      cum_atten_probs = np.pad(cum_atten_probs, ((1, 0), (0, 0), (0, 0)),
                               'constant')[:seq_len, :, :]
      cum_atten_probs = cum_atten_probs + eos_atten_probs
      results = ops.top_k_from_beam_search_outs(
          hyps,
          prev_hyps,
          done_hyps,
          cumulative_scores,
          eos_scores,
          scores=scores if populate_hyps else 0,
          atten_probs=atten_probs if populate_hyps else 0,
          eos_atten_probs=eos_atten_probs if populate_hyps else 0,
          cumulative_atten_probs=cum_atten_probs if coverage_penalty > 0 else 0,
          length_normalization=length_normalization,
          coverage_penalty=coverage_penalty,
          num_hyps_per_beam=num_hyps_per_beam,
          max_seq_length=seq_len,
          target_seq_length_ratio=length_ratio,
          populate_topk_hyps=populate_hyps,
      )
      outs = sess.run(results)

      final_done_hyps = ops.hyps_from_beam_search_outs(
          hyps,
          prev_hyps,
          done_hyps,
          scores,
          atten_probs,
          eos_scores,
          eos_atten_probs,
          eos_id=2,
          num_hyps_per_beam=num_hyps_per_beam,
      )
      src_seq_lengths = np.ones([num_beams], dtype=np.int32) * seq_len
      topk_hyps = ops.top_k_terminated_hyps(
          final_done_hyps,
          src_seq_lengths,
          k=num_hyps_per_beam,
          num_hyps_per_beam=num_hyps_per_beam,
          length_normalization=length_normalization,
          coverage_penalty=coverage_penalty,
          target_seq_length_ratio=length_ratio)
      topk_ids, topk_lens, topk_scores = ops.unpack_hyp(
          topk_hyps, max_seq_length=seq_len)
      topk_hyps, topk_ids, topk_lens, topk_scores = sess.run(
          [topk_hyps, topk_ids, topk_lens, topk_scores])

    self.assertAllEqual(outs[0], topk_ids)
    self.assertAllEqual(outs[1], topk_lens)
    self.assertAllClose(outs[2], topk_scores)
    if populate_hyps:
      self.assertAllEqual(outs[3].shape, topk_hyps.shape)
      self.assertAllEqual(outs[3].shape, [num_beams, num_hyps_per_beam])
      for i in range(num_beams):
        for j in range(num_hyps_per_beam):
          self._SameHyp(outs[3][i, j], topk_hyps[i, j])


if __name__ == '__main__':
  tf.test.main()
