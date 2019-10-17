# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for third_party.py.lingvo.core.steps.attention_steps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core.steps import attention_steps
import numpy as np


class AttentionStepsTest(test_utils.TestCase):

  def testAttentionStep(self):
    with self.session(use_gpu=False) as sess:
      np.random.seed(12345)
      src_batch_size = 3
      target_batch_size = 6
      src_length = 5
      src_context_dim = 4
      query_dim = 5
      src_dim = 4
      source_vecs = tf.constant(
          np.random.rand(src_length, src_batch_size, src_dim), dtype=tf.float32)
      source_contexts = tf.constant(
          np.random.rand(src_length, src_batch_size, src_context_dim),
          dtype=tf.float32)
      source_padding = tf.zeros([src_length, target_batch_size],
                                dtype=tf.float32)
      query_vec = tf.constant(
          np.random.rand(target_batch_size, query_dim), dtype=tf.float32)

      p = attention_steps.AttentionStep.Params()
      p.atten.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      p.atten.source_dim = src_dim
      p.atten.query_dim = query_dim
      p.atten.hidden_dim = query_dim
      p.atten.vn.global_vn = False
      p.atten.vn.per_step_vn = False
      p.atten.packed_input = True
      step = p.Instantiate()

      external_inputs = py_utils.NestedMap(
          max_seq_length=13,
          src=source_vecs,
          context=source_contexts,
          padding=source_padding)
      packed = step.PrepareExternalInputs(step.theta, external_inputs)
      state0 = step.ZeroState(step.theta, packed, target_batch_size)

      step_inputs = py_utils.NestedMap(inputs=[query_vec])
      step_padding = tf.zeros([target_batch_size, 1], dtype=tf.float32)
      output, state1 = step.FProp(step.theta, packed, step_inputs, step_padding,
                                  state0)

      tf.global_variables_initializer().run()
      output, state1 = sess.run([output, state1])

      self.assertAllClose(
          output, {
              'context': [[0.41788787, 0.5865286, 0.58267754, 0.21218117],
                          [0.42178467, 0.5067202, 0.5413259, 0.6616881],
                          [0.71586907, 0.6303425, 0.52290946, 0.694283],
                          [0.41789612, 0.58647645, 0.5826333, 0.21220288],
                          [0.421697, 0.5068262, 0.5411844, 0.66167986],
                          [0.7156511, 0.63033843, 0.5228955, 0.69437]],
              'probs':
                  [[0.20118009, 0.19332525, 0.20120151, 0.2022583, 0.20203482],
                   [0.20019522, 0.20133461, 0.19572362, 0.2025276, 0.2002189],
                   [0.20116101, 0.20004824, 0.20221081, 0.19645905, 0.20012087],
                   [0.20123273, 0.19319996, 0.20131132, 0.20220752, 0.2020485],
                   [0.2002011, 0.2015253, 0.19534773, 0.20260131, 0.20032457],
                   [0.20097165, 0.19993119, 0.20225787, 0.19671878, 0.20012051]]
          })
      self.assertAllClose(
          state1, {
              'atten_state': [[0.], [0.], [0.], [0.], [0.], [0.]],
              'atten_context': [[0.41788787, 0.5865286, 0.58267754, 0.21218117],
                                [0.42178467, 0.5067202, 0.5413259, 0.6616881],
                                [0.71586907, 0.6303425, 0.52290946, 0.694283],
                                [0.41789612, 0.58647645, 0.5826333, 0.21220288],
                                [0.421697, 0.5068262, 0.5411844, 0.66167986],
                                [0.7156511, 0.63033843, 0.5228955, 0.69437]]
          })

  def testAttentionBlockStep(self):
    with self.session(use_gpu=False) as sess:
      np.random.seed(12345)
      src_batch_size = 3
      target_batch_size = 6
      src_length = 5
      query_dim = 5
      context_dim = 8
      hidden_dim = 7
      src_dim = context_dim
      source_vecs = tf.constant(
          np.random.rand(src_length, src_batch_size, src_dim), dtype=tf.float32)
      source_padding = tf.zeros([src_length, target_batch_size],
                                dtype=tf.float32)

      p = attention_steps.AttentionBlockStep.Params()
      p.attention.atten.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      p.attention.atten.source_dim = src_dim
      p.attention.atten.query_dim = query_dim
      p.attention.atten.hidden_dim = hidden_dim
      p.attention.atten.vn.global_vn = False
      p.attention.atten.vn.per_step_vn = False
      p.attention.atten.packed_input = True
      p.query_generator.step_input_dim = context_dim
      p.query_generator.rnn_cell_dim = query_dim
      step = p.Instantiate()

      external_inputs = py_utils.NestedMap(
          attention=py_utils.NestedMap(
              max_seq_length=14, src=source_vecs, padding=source_padding))
      packed = step.PrepareExternalInputs(step.theta, external_inputs)
      state0 = step.ZeroState(step.theta, packed, target_batch_size)

      step_padding = tf.zeros([target_batch_size, 1], dtype=tf.float32)
      output, state1 = step.FProp(step.theta, packed, None, step_padding,
                                  state0)

      tf.global_variables_initializer().run()
      output, state1 = sess.run([output, state1])

      self.assertAllClose(
          output, {
              'atten_query':
                  np.array([
                      [
                          0.1142175, 0.00020437, 0.02718649, -0.06030316,
                          0.02916641
                      ],
                      [
                          0.09362462, 0.07093287, 0.10184045, -0.0228882,
                          0.06189567
                      ],
                      [
                          0.12866478, 0.0121689, 0.05557573, -0.04107622,
                          0.0543875
                      ],
                      [
                          0.1142175, 0.00020437, 0.02718649, -0.06030316,
                          0.02916641
                      ],
                      [
                          0.09362462, 0.07093287, 0.10184045, -0.0228882,
                          0.06189567
                      ],
                      [
                          0.12866478, 0.0121689, 0.05557573, -0.04107622,
                          0.0543875
                      ],
                  ]),
              'atten_context':
                  np.array([
                      [
                          0.55453926, 0.55162865, 0.62239933, 0.26001987,
                          0.51269007, 0.555924, 0.54857075, 0.51340824
                      ],
                      [
                          0.6495046, 0.42096642, 0.605386, 0.79519784,
                          0.39852753, 0.30938083, 0.53797, 0.43651274
                      ],
                      [
                          0.66645885, 0.56522155, 0.67393464, 0.6224826,
                          0.66094846, 0.6098963, 0.52270895, 0.5319694
                      ],
                      [
                          0.55453926, 0.55162865, 0.62239933, 0.26001987,
                          0.51269007, 0.555924, 0.54857075, 0.51340824
                      ],
                      [
                          0.6495046, 0.42096642, 0.605386, 0.79519784,
                          0.39852753, 0.30938083, 0.53797, 0.43651274
                      ],
                      [
                          0.66645885, 0.56522155, 0.67393464, 0.6224826,
                          0.66094846, 0.6098963, 0.52270895, 0.5319694
                      ],
                  ]),
              'atten_probs':
                  np.array([
                      [
                          0.20132412, 0.19545832, 0.20277032, 0.19362292,
                          0.20682438
                      ],
                      [
                          0.20172212, 0.20001633, 0.20166671, 0.20218876,
                          0.19440602
                      ],
                      [
                          0.20540778, 0.20792785, 0.19377577, 0.19288684,
                          0.20000176
                      ],
                      [
                          0.20132412, 0.19545832, 0.20277032, 0.19362292,
                          0.20682438
                      ],
                      [
                          0.20172212, 0.20001633, 0.20166671, 0.20218876,
                          0.19440602
                      ],
                      [
                          0.20540778, 0.20792785, 0.19377577, 0.19288684,
                          0.20000176
                      ],
                  ])
          })


if __name__ == '__main__':
  tf.test.main()
