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
"""Tests for target_sequence_sampler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lingvo.core import py_utils
from lingvo.core import target_sequence_sampler


class TargetSequenceSamplerTest(tf.test.TestCase):

  def testTargetSequenceSampler(self):
    with self.session(use_gpu=False) as sess:
      np.random.seed(9384758)
      tf.set_random_seed(8274758)
      vocab_size = 12
      src_len = 5
      tgt_len = 7
      batch_size = 2
      p = target_sequence_sampler.TargetSequenceSampler.Params().Set(
          name='bsh', target_seq_len=tgt_len)
      seq_sampler = p.cls(p)

      def InitBeamSearchCallBack(unused_theta,
                                 unused_source_encs,
                                 unused_source_padding,
                                 num_hyps_per_beam,
                                 unused_additional_source_info=None):
        self.assertEqual(1, num_hyps_per_beam)
        logits = tf.zeros((batch_size, vocab_size), dtype=tf.float32)
        return (py_utils.NestedMap(log_probs=logits),
                py_utils.NestedMap(step=tf.constant(0)))

      def PreBeamSearchStepCallback(unused_theta,
                                    unused_source_encs,
                                    unused_source_paddings,
                                    unused_step_ids,
                                    states,
                                    num_hyps_per_beam,
                                    unused_additional_source_info=None):
        self.assertEqual(1, num_hyps_per_beam)
        logits = tf.random_normal([batch_size, vocab_size], seed=8273747)
        return (py_utils.NestedMap(log_probs=logits),
                py_utils.NestedMap(step=states.step + 1))

      def PostBeamSearchStepCallback(unused_theta,
                                     unused_source_encs,
                                     unused_source_paddings,
                                     unused_new_step_ids,
                                     states,
                                     unused_additional_source_info=None):
        return states

      src_enc = tf.random_normal([src_len, batch_size, 8], seed=982774838)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)

      theta = py_utils.NestedMap()
      random_seed = tf.constant(123)
      decoder_output = seq_sampler.Sample(
          theta, src_enc, src_enc_padding, random_seed, InitBeamSearchCallBack,
          PreBeamSearchStepCallback, PostBeamSearchStepCallback)

      ids, lens = sess.run([
          decoder_output.ids,
          tf.reduce_sum(1 - decoder_output.paddings, 1),
      ])
      print(np.array_repr(ids))
      print(np.array_repr(lens))
      expected_ids = [[10, 3, 4, 2, 2, 2, 2], [1, 1, 11, 6, 1, 0, 6]]
      expected_lens = [4, 7]
      self.assertAllEqual(expected_ids, ids)
      self.assertAllEqual(expected_lens, lens)


if __name__ == '__main__':
  tf.test.main()
