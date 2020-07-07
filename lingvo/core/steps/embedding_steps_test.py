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
from lingvo.core import test_utils
from lingvo.core.steps import embedding_steps


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


if __name__ == '__main__':
  tf.test.main()
