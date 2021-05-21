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
"""Tests for target_sequence_sampler."""

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import target_sequence_sampler
from lingvo.core import test_utils
import numpy as np


class TargetSequenceSamplerTest(test_utils.TestCase):

  def testTargetSequenceSampler(self):
    with self.session(use_gpu=False):
      np.random.seed(9384758)
      tf.random.set_seed(8274758)
      vocab_size = 12
      src_len = 5
      tgt_len = 7
      batch_size = 2

      def InitBeamSearchCallBack(unused_theta, unused_encoder_outputs,
                                 num_hyps_per_beam):
        self.assertEqual(1, num_hyps_per_beam)
        logits = tf.zeros((batch_size, vocab_size), dtype=tf.float32)
        return (py_utils.NestedMap(log_probs=logits),
                py_utils.NestedMap(step=tf.constant(0)))

      def PreBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                    unused_step_ids, states, num_hyps_per_beam):
        self.assertEqual(1, num_hyps_per_beam)
        logits = tf.random.stateless_normal([batch_size, vocab_size],
                                            seed=[8273747, 9])
        return (py_utils.NestedMap(log_probs=logits),
                py_utils.NestedMap(step=states.step + 1))

      def PostBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                     unused_new_step_ids, states):
        return states

      src_enc = tf.random.stateless_normal([src_len, batch_size, 8],
                                           seed=[982774838, 9])
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)

      theta = py_utils.NestedMap()
      random_seed = tf.constant(123)
      p = target_sequence_sampler.TargetSequenceSampler.Params().Set(
          name='bsh', target_seq_len=tgt_len)
      seq_sampler = p.Instantiate()
      decoder_output = seq_sampler.Sample(
          theta, encoder_outputs, random_seed, InitBeamSearchCallBack,
          PreBeamSearchStepCallback, PostBeamSearchStepCallback)

      ids, lens = self.evaluate([
          decoder_output.ids,
          tf.reduce_sum(1 - decoder_output.paddings, 1),
      ])
      print(np.array_repr(ids))
      print(np.array_repr(lens))
      expected_ids = [[9, 0, 2, 2, 2, 2, 2], [0, 0, 11, 8, 1, 0, 7]]
      expected_lens = [3, 7]
      self.assertAllEqual(expected_ids, ids)
      self.assertAllEqual(expected_lens, lens)

      p = target_sequence_sampler.TargetSequenceSampler.Params().Set(
          name='bsh', target_seq_len=tgt_len, top_k=1)
      seq_sampler = p.Instantiate()
      decoder_output = seq_sampler.Sample(theta, encoder_outputs, random_seed,
                                          InitBeamSearchCallBack,
                                          PreBeamSearchStepCallback,
                                          PostBeamSearchStepCallback)

      ids, lens = self.evaluate([
          decoder_output.ids,
          tf.reduce_sum(1 - decoder_output.paddings, 1),
      ])
      print(np.array_repr(ids))
      print(np.array_repr(lens))
      expected_ids = [[0, 0, 0, 0, 0, 0, 0], [7, 7, 7, 7, 7, 7, 7]]
      expected_lens = [7, 7]
      self.assertAllEqual(expected_ids, ids)
      self.assertAllEqual(expected_lens, lens)

      p = target_sequence_sampler.TargetSequenceSampler.Params().Set(
          name='bsh', target_seq_len=tgt_len, top_k=5)
      seq_sampler = p.Instantiate()
      decoder_output = seq_sampler.Sample(theta, encoder_outputs, random_seed,
                                          InitBeamSearchCallBack,
                                          PreBeamSearchStepCallback,
                                          PostBeamSearchStepCallback)

      ids, lens = self.evaluate([
          decoder_output.ids,
          tf.reduce_sum(1 - decoder_output.paddings, 1),
      ])
      print(np.array_repr(ids))
      print(np.array_repr(lens))
      expected_ids = [[5, 0, 0, 0, 8, 0, 6], [7, 7, 10, 0, 7, 7, 0]]
      expected_lens = [7, 7]
      self.assertAllEqual(expected_ids, ids)
      self.assertAllEqual(expected_lens, lens)

      p = target_sequence_sampler.TargetSequenceSampler.Params().Set(
          name='bsh', target_seq_len=tgt_len, temperature=0.2)
      seq_sampler = p.Instantiate()
      decoder_output = seq_sampler.Sample(
          theta, encoder_outputs, random_seed, InitBeamSearchCallBack,
          PreBeamSearchStepCallback, PostBeamSearchStepCallback)

      ids, lens = self.evaluate([
          decoder_output.ids,
          tf.reduce_sum(1 - decoder_output.paddings, 1),
      ])
      print(np.array_repr(ids))
      print(np.array_repr(lens))
      expected_ids = [[0, 0, 0, 0, 0, 0, 9], [0, 0, 11, 7, 1, 0, 7]]
      expected_lens = [7, 7]
      self.assertAllEqual(expected_ids, ids)
      self.assertAllEqual(expected_lens, lens)

  def testTargetSequenceSamplerWithEOC(self):
    with self.session(use_gpu=False):
      np.random.seed(9384758)
      tf.random.set_seed(8274758)
      vocab_size = 4
      src_len = 5
      tgt_len = 20
      batch_size = 2
      p = target_sequence_sampler.TargetSequenceSampler.Params().Set(
          name='bsh', target_seq_len=tgt_len, target_eoc_id=0)
      seq_sampler = p.Instantiate()

      def InitBeamSearchCallBack(unused_theta, unused_encoder_outputs,
                                 num_hyps_per_beam):
        self.assertEqual(1, num_hyps_per_beam)
        logits = tf.zeros((batch_size, vocab_size), dtype=tf.float32)
        is_last_chunk = tf.constant(False, shape=[batch_size])
        result = py_utils.NestedMap(
            log_probs=logits, is_last_chunk=is_last_chunk)
        states = py_utils.NestedMap(
            step=tf.constant(0),
            src_step=tf.zeros([batch_size], dtype=tf.int32))
        return result, states

      def PreBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                    unused_step_ids, states, num_hyps_per_beam):
        self.assertEqual(1, num_hyps_per_beam)
        logits = tf.random.stateless_normal([batch_size, vocab_size],
                                            seed=[8273747, 9])
        # Make it never predict <eos>.
        logits -= tf.one_hot([p.target_eos_id], vocab_size, 1e30)
        is_last_chunk = tf.equal(states.src_step, src_len - 1)
        result = py_utils.NestedMap(
            log_probs=logits, is_last_chunk=is_last_chunk)
        return result, states

      def PostBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                     new_step_ids, states):
        return py_utils.NestedMap(
            step=states.step + 1,
            src_step=states.src_step + tf.cast(
                tf.equal(new_step_ids, p.target_eoc_id), dtype=tf.int32))

      src_enc = tf.random.stateless_normal([src_len, batch_size, 8],
                                           seed=[982774838, 9])
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)

      theta = py_utils.NestedMap()
      random_seed = tf.constant(123)
      decoder_output = seq_sampler.Sample(
          theta, encoder_outputs, random_seed, InitBeamSearchCallBack,
          PreBeamSearchStepCallback, PostBeamSearchStepCallback)

      ids, lens = self.evaluate([
          decoder_output.ids,
          tf.reduce_sum(1 - decoder_output.paddings, 1),
      ])
      print(np.array_repr(ids))
      print(np.array_repr(lens))
      expected_ids = [
          [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
          [0, 0, 3, 3, 1, 0, 3, 0, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      ]
      expected_lens = [5, 11]
      self.assertAllEqual(expected_ids, ids)
      self.assertAllEqual(expected_lens, lens)

      # Now do the same, except with use_stop_fn=True.
      p = target_sequence_sampler.TargetSequenceSampler.Params().Set(
          name='bsh', target_seq_len=tgt_len, target_eoc_id=0, use_stop_fn=True)
      seq_sampler = p.Instantiate()
      decoder_output = seq_sampler.Sample(theta, encoder_outputs, random_seed,
                                          InitBeamSearchCallBack,
                                          PreBeamSearchStepCallback,
                                          PostBeamSearchStepCallback)

      ids, lens = self.evaluate([
          decoder_output.ids,
          tf.reduce_sum(1 - decoder_output.paddings, 1),
      ])
      print(np.array_repr(ids))
      print(np.array_repr(lens))
      expected_ids = [
          [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
          [0, 0, 3, 3, 1, 0, 3, 0, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      ]
      expected_lens = [5, 11]
      self.assertAllEqual(expected_ids, ids)
      self.assertAllEqual(expected_lens, lens)

  def testTargetSequenceSamplerWithVariables(self):
    with self.session(use_gpu=False):
      np.random.seed(9384758)
      tf.random.set_seed(8274758)
      hidden_dim = 8
      vocab_size = 12
      src_len = 5
      tgt_len = 7
      batch_size = 2
      keras_layer = py_utils.NestedMap(dense=None)

      def InitBeamSearchCallBack(unused_theta, unused_encoder_outputs,
                                 num_hyps_per_beam):
        self.assertEqual(1, num_hyps_per_beam)
        logits = tf.zeros((batch_size, vocab_size), dtype=tf.float32)
        return (py_utils.NestedMap(log_probs=logits),
                py_utils.NestedMap(step=tf.constant(0)))

      def PreBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                    unused_step_ids, states, num_hyps_per_beam):
        self.assertEqual(1, num_hyps_per_beam)
        hidden = tf.random.stateless_normal([batch_size, hidden_dim],
                                            seed=[8273747, 9])
        # A stateful 'dense' layer.
        if not keras_layer.dense:
          keras_layer.dense = tf.keras.layers.Dense(
              vocab_size, input_shape=hidden.shape)
        log_probs = tf.nn.log_softmax(keras_layer.dense(hidden))
        return (py_utils.NestedMap(log_probs=log_probs),
                py_utils.NestedMap(step=states.step + 1))

      def PostBeamSearchStepCallback(unused_theta, unused_encoder_outputs,
                                     unused_new_step_ids, states):
        return states

      src_enc = tf.random.stateless_normal([src_len, batch_size, 8],
                                           seed=[982774838, 9])
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)

      theta = py_utils.NestedMap()
      random_seed = tf.constant(123)
      p = target_sequence_sampler.TargetSequenceSampler.Params().Set(
          name='bsh', target_seq_len=tgt_len)
      seq_sampler = p.Instantiate()
      decoder_output = seq_sampler.Sample(theta, encoder_outputs, random_seed,
                                          InitBeamSearchCallBack,
                                          PreBeamSearchStepCallback,
                                          PostBeamSearchStepCallback)

      self.evaluate(tf.global_variables_initializer())
      ids, lens = self.evaluate([
          decoder_output.ids,
          tf.reduce_sum(1 - decoder_output.paddings, 1),
      ])
      print(np.array_repr(ids))
      print(np.array_repr(lens))


if __name__ == '__main__':
  tf.test.main()
