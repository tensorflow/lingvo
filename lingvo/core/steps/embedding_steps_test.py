# Lint as: python3
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
"""Tests for lingvo.core.steps.embedding_steps."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import recurrent
from lingvo.core import test_utils
from lingvo.core.steps import embedding_steps
import numpy as np


class EmbeddingStepsTest(test_utils.TestCase):

  def testEmbeddingStep(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      p = embedding_steps.EmbeddingStep.Params()
      p.name = 'emb_step'
      p.emb.vocab_size = 10
      p.emb.embedding_dim = 2
      p.emb.max_num_shards = 1
      p.emb.params_init = py_utils.WeightInit.Gaussian(0.01)
      p.emb.vn.global_vn = False
      p.emb.vn.per_step_vn = False
      emb = p.Instantiate()

      # Verify that nothing bad happens when these methods are called.
      packed = emb.PrepareExternalInputs(None, None)
      state0 = emb.ZeroState(None, None, None)

      out1, state1 = emb.FProp(
          emb.theta, packed,
          py_utils.NestedMap(inputs=[tf.constant([4, 3], tf.int32)]),
          tf.constant([0.0], dtype=tf.float32), state0)

      self.evaluate(tf.global_variables_initializer())
      out1, state1 = self.evaluate([out1, state1])

      self.assertEqual({}, state1)
      self.assertAllClose(
          out1.output,
          [[-5.9790569e-03, -8.7367110e-03], [-2.1643407e-06, 1.4426162e-02]])

  def testStatefulEmbeddingStep(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      p = embedding_steps.StatefulEmbeddingStep.Params().Set(
          name='emb_step',
          num_prev_tokens=1,
          include_current_token=False,
          target_sos_id=1,
          embedding_dim=3,
      )
      p.emb.Set(
          vocab_size=10,
          embedding_dim=3,
          max_num_shards=1,
          params_init=py_utils.WeightInit.Gaussian(0.01),
      )
      p.emb.vn.global_vn = False
      p.emb.vn.per_step_vn = False
      emb = p.Instantiate()

      # Verify that nothing bad happens when these methods are called.
      packed = emb.PrepareExternalInputs(None, None)
      state0 = emb.ZeroState(emb.theta, packed, 2)

      # Test FProp of the unit
      out1, state1 = emb.FProp(
          emb.theta, packed,
          py_utils.NestedMap(inputs=[tf.constant([4, 3], tf.int32)]),
          tf.constant([0.0], dtype=tf.float32), state0)
      self.evaluate(tf.global_variables_initializer())
      out1, state1 = self.evaluate([out1, state1])

      self.assertAllEqual(state1.prev_ids, np.array([[4], [3]]))
      self.assertAllClose(
          out1.output,
          np.array([[-0.00740041, -0.00746862, 0.00093992],
                    [-0.00740041, -0.00746862, 0.00093992]]))

      # Test FProp and BProp when integrated with Recurrent()
      def _FProp(theta, state0, inputs):
        embedding, state1 = emb.FProp(
            theta,
            None,
            inputs,
            None,
            state0,
        )
        state1.embedding = embedding.output
        return state1, py_utils.NestedMap()

      inputs = py_utils.NestedMap(inputs=[
          tf.constant([[1., 2.], [3., 2.], [0., 1.], [2., 3.], [3., 0.]])
      ])
      acc, _ = recurrent.Recurrent(
          emb.theta,
          state0,
          inputs,
          _FProp,
      )
      loss = tf.math.l2_normalize(acc.embedding)
      grad = tf.gradients(loss, emb.emb.theta.wm[0])
      self.evaluate(tf.global_variables_initializer())
      acc_, _, grad_ = self.evaluate([acc, emb.emb.theta.wm[0], grad])
      prev_ids_expected = np.array([
          [[1.], [2.]],
          [[3.], [2.]],
          [[0.], [1.]],
          [[2.], [3.]],
          [[3.], [0.]],
      ])
      grad_expected = np.array([[21.952698, 20.50312, 19.037958],
                                [79.622116, 72.15271, 106.34329],
                                [41.631985, 70.19292, 75.52608],
                                [53.644493, 36.28507, 36.64856], [0., 0., 0.],
                                [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                                [0., 0., 0.], [0., 0., 0.]])
      self.assertAllClose(acc_.prev_ids, prev_ids_expected)
      self.assertAllClose(grad_[0], grad_expected)


if __name__ == '__main__':
  tf.test.main()
