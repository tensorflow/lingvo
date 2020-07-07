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
"""Tests for third_party.py.lingvo.core.steps.attention_steps."""

from lingvo import compat as tf
from lingvo.core import attention
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core.steps import attention_steps
import numpy as np


class AttentionStepsTest(test_utils.TestCase):

  def testAttentionStep(self):
    with self.session(use_gpu=False):
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
          src=source_vecs,
          context=source_contexts,
          padding=source_padding)
      packed = step.PrepareExternalInputs(step.theta, external_inputs)
      state0 = step.ZeroState(step.theta, packed, target_batch_size)

      step_inputs = py_utils.NestedMap(inputs=[query_vec])
      step_padding = tf.zeros([target_batch_size, 1], dtype=tf.float32)
      output, state1 = step.FProp(step.theta, packed, step_inputs, step_padding,
                                  state0)

      self.evaluate(tf.global_variables_initializer())
      output, state1 = self.evaluate([output, state1])

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

  def testAttentionStepMultiSourceSame(self):
    with self.session(use_gpu=False):
      np.random.seed(12345)
      src_batch_size = 3
      target_batch_size = 6
      src_length = 5
      query_dim = 5
      src_dim = 4
      source_vecs_0 = tf.constant(
          np.random.rand(src_length, src_batch_size, src_dim), dtype=tf.float32)
      source_vecs_1 = tf.constant(
          np.random.rand(src_length, src_batch_size, src_dim), dtype=tf.float32)
      sources = py_utils.NestedMap(
          source_0=source_vecs_0, source_1=source_vecs_1)

      source_padding_0 = tf.zeros([src_length, src_batch_size],
                                  dtype=tf.float32)
      source_padding_1 = tf.zeros([src_length, src_batch_size],
                                  dtype=tf.float32)
      source_paddings = py_utils.NestedMap(
          source_0=source_padding_0, source_1=source_padding_1)
      query_vec = tf.constant(
          np.random.rand(target_batch_size, query_dim), dtype=tf.float32)

      p = attention_steps.AttentionStep.Params()

      # Setup MultiSourceAttention
      p.atten = attention.MultiSourceAttention.Params()
      p.atten.source_dim = src_dim
      p.atten.query_dim = query_dim

      add_atten_params = attention.AdditiveAttention.Params()
      add_atten_params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      add_atten_params.source_dim = src_dim
      add_atten_params.query_dim = query_dim
      add_atten_params.hidden_dim = query_dim
      add_atten_params.vn.global_vn = False
      add_atten_params.vn.per_step_vn = False
      add_atten_params.packed_input = True

      p.atten.source_atten_tpls = [('source_0', add_atten_params),
                                   ('source_1', add_atten_params)]

      step = p.Instantiate()

      external_inputs = py_utils.NestedMap(src=sources, padding=source_paddings)
      packed = step.PrepareExternalInputs(step.theta, external_inputs)
      state0 = step.ZeroState(step.theta, packed, target_batch_size)

      step_inputs = py_utils.NestedMap(inputs=[query_vec])
      step_padding = tf.zeros([target_batch_size, 1], dtype=tf.float32)
      output, state1 = step.FProp(step.theta, packed, step_inputs, step_padding,
                                  state0)

      self.evaluate(tf.global_variables_initializer())
      output, state1 = self.evaluate([output, state1])

      self.assertAllClose(
          output, {
              'context': [[0.9590156, 0.8653384, 1.1668519, 0.697219],
                          [1.175648, 1.1199431, 1.2219069, 1.1452408],
                          [1.3191833, 1.0350775, 1.1315871, 1.3297331],
                          [0.95910096, 0.86546516, 1.1669571, 0.6971649],
                          [1.175647, 1.1201943, 1.222264, 1.1451368],
                          [1.3188481, 1.034915, 1.1314276, 1.3297772]],
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
              'atten_state': {
                  'source_0': [[0.], [0.], [0.], [0.], [0.], [0.]],
                  'source_1': [[0.], [0.], [0.], [0.], [0.], [0.]]
              },
              'atten_context': [[0.9590156, 0.8653384, 1.1668519, 0.697219],
                                [1.175648, 1.1199431, 1.2219069, 1.1452408],
                                [1.3191833, 1.0350775, 1.1315871, 1.3297331],
                                [0.95910096, 0.86546516, 1.1669571, 0.6971649],
                                [1.175647, 1.1201943, 1.222264, 1.1451368],
                                [1.3188481, 1.034915, 1.1314276, 1.3297772]]
          })

  def testAttentionStepMultiSourceSameWithGmmAttention(self):
    with self.session(use_gpu=False):
      np.random.seed(12345)
      src_batch_size = 3
      target_batch_size = 6
      src_length = 5
      query_dim = 5
      src_dim = 4
      source_vecs_0 = tf.constant(
          np.random.rand(src_length, src_batch_size, src_dim), dtype=tf.float32)
      source_vecs_1 = tf.constant(
          np.random.rand(src_length, src_batch_size, src_dim), dtype=tf.float32)
      sources = py_utils.NestedMap(
          source_0=source_vecs_0, source_1=source_vecs_1)

      source_padding_0 = tf.zeros([src_length, src_batch_size],
                                  dtype=tf.float32)
      source_padding_1 = tf.zeros([src_length, src_batch_size],
                                  dtype=tf.float32)
      source_paddings = py_utils.NestedMap(
          source_0=source_padding_0, source_1=source_padding_1)
      query_vec = tf.constant(
          np.random.rand(target_batch_size, query_dim), dtype=tf.float32)

      p = attention_steps.AttentionStep.Params()

      # Setup MultiSourceAttention
      p.atten = attention.MultiSourceAttention.Params()
      p.atten.source_dim = src_dim
      p.atten.query_dim = query_dim

      gmm_atten_params = attention.GmmMonotonicAttention.Params()
      gmm_atten_params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      gmm_atten_params.source_dim = src_dim
      gmm_atten_params.query_dim = query_dim
      gmm_atten_params.hidden_dim = query_dim
      gmm_atten_params.vn.global_vn = False
      gmm_atten_params.vn.per_step_vn = False
      gmm_atten_params.packed_input = True

      p.atten.source_atten_tpls = [('source_0', gmm_atten_params),
                                   ('source_1', gmm_atten_params)]

      step = p.Instantiate()

      external_inputs = py_utils.NestedMap(src=sources, padding=source_paddings)
      packed = step.PrepareExternalInputs(step.theta, external_inputs)
      state0 = step.ZeroState(step.theta, packed, target_batch_size)

      step_inputs = py_utils.NestedMap(inputs=[query_vec])
      step_padding = tf.zeros([target_batch_size, 1], dtype=tf.float32)
      output, state1 = step.FProp(step.theta, packed, step_inputs, step_padding,
                                  state0)

      self.evaluate(tf.global_variables_initializer())
      output, state1 = self.evaluate([output, state1])

      self.assertAllClose(
          output, {
              'context': [[0.8048796, 0.9554154, 1.2422264, 0.82598877],
                          [1.1976988, 0.9226365, 1.1311831, 1.1287751],
                          [1.2583418, 0.96984935, 0.8972859, 1.2939383],
                          [0.8055052, 0.9545301, 1.2421954, 0.824931],
                          [1.1980952, 0.9227077, 1.1313919, 1.13009],
                          [1.2582378, 0.96980226, 0.8973369, 1.2938937]],
              'probs':
                  [[0.05302628, 0.20965888, 0.3661108, 0.26998273, 0.08293614],
                   [0.05321905, 0.20958655, 0.36570197, 0.270003, 0.08308904],
                   [0.05327733, 0.20919749, 0.36514452, 0.27033207, 0.08349889],
                   [0.05328987, 0.20906723, 0.3648241, 0.27042356, 0.08376145],
                   [0.05301215, 0.21013679, 0.36650375, 0.26960865, 0.08261178],
                   [0.05328071, 0.20917267, 0.36505368, 0.27032903, 0.08357814]]
          })
      self.assertAllClose(
          state1, {
              'atten_state': {
                  'source_0': [[[2.4243412, 1.2218076, 1.0122609, 0.18427502],
                                [1.9546769, 0.9721461, 1.0469768, 0.19244196],
                                [1.7934805, 0.8947478, 1.2158467, 0.18101364],
                                [2.2727895, 1.1433213, 0.969053, 0.21098366],
                                [2.1986299, 1.0997422, 1.3713341, 0.23128569]],
                               [[2.4298353, 1.227302, 1.0116383, 0.18391277],
                                [1.9476058, 0.96507514, 1.0462759, 0.19275317],
                                [1.793545, 0.89481235, 1.220826, 0.1817861],
                                [2.2800756, 1.1506072, 0.96794796, 0.21093304],
                                [2.194984, 1.0960963, 1.3741415, 0.23061496]],
                               [[2.4273272, 1.2247936, 1.0106387, 0.18302175],
                                [1.9522938, 0.96976304, 1.0510013, 0.19241981],
                                [1.7976122, 0.8988795, 1.2246737, 0.18208173],
                                [2.2875524, 1.1580843, 0.97309643, 0.21170339],
                                [2.1904838, 1.0915961, 1.3786552, 0.23077331]],
                               [[2.4339817, 1.2314482, 1.0118915, 0.18239658],
                                [1.9538436, 0.9713129, 1.050209, 0.19243228],
                                [1.7997689, 0.90103614, 1.2248727, 0.18208562],
                                [2.286818, 1.15735, 0.9776513, 0.2125737],
                                [2.1872034, 1.0883157, 1.3807379, 0.23051178]],
                               [[2.4258854, 1.223352, 1.0136935, 0.18414007],
                                [1.9573982, 0.9748675, 1.0445031, 0.19239089],
                                [1.7965381, 0.89780533, 1.2112961, 0.18159895],
                                [2.2637806, 1.1343125, 0.9743988, 0.21178932],
                                [2.1948628, 1.0959752, 1.366173, 0.23008086]],
                               [[2.435421, 1.2328876, 1.0118036, 0.18307444],
                                [1.9479709, 0.96544015, 1.0476727, 0.19277772],
                                [1.795729, 0.8969963, 1.224472, 0.18180896],
                                [2.2865427, 1.1570745, 0.9713619, 0.211611],
                                [2.1911612, 1.0922736, 1.3791639, 0.2307278]]],
                  'source_1': [[[2.4243412, 1.2218076, 1.0122609, 0.18427502],
                                [1.9546769, 0.9721461, 1.0469768, 0.19244196],
                                [1.7934805, 0.8947478, 1.2158467, 0.18101364],
                                [2.2727895, 1.1433213, 0.969053, 0.21098366],
                                [2.1986299, 1.0997422, 1.3713341, 0.23128569]],
                               [[2.4298353, 1.227302, 1.0116383, 0.18391277],
                                [1.9476058, 0.96507514, 1.0462759, 0.19275317],
                                [1.793545, 0.89481235, 1.220826, 0.1817861],
                                [2.2800756, 1.1506072, 0.96794796, 0.21093304],
                                [2.194984, 1.0960963, 1.3741415, 0.23061496]],
                               [[2.4273272, 1.2247936, 1.0106387, 0.18302175],
                                [1.9522938, 0.96976304, 1.0510013, 0.19241981],
                                [1.7976122, 0.8988795, 1.2246737, 0.18208173],
                                [2.2875524, 1.1580843, 0.97309643, 0.21170339],
                                [2.1904838, 1.0915961, 1.3786552, 0.23077331]],
                               [[2.4339817, 1.2314482, 1.0118915, 0.18239658],
                                [1.9538436, 0.9713129, 1.050209, 0.19243228],
                                [1.7997689, 0.90103614, 1.2248727, 0.18208562],
                                [2.286818, 1.15735, 0.9776513, 0.2125737],
                                [2.1872034, 1.0883157, 1.3807379, 0.23051178]],
                               [[2.4258854, 1.223352, 1.0136935, 0.18414007],
                                [1.9573982, 0.9748675, 1.0445031, 0.19239089],
                                [1.7965381, 0.89780533, 1.2112961, 0.18159895],
                                [2.2637806, 1.1343125, 0.9743988, 0.21178932],
                                [2.1948628, 1.0959752, 1.366173, 0.23008086]],
                               [[2.435421, 1.2328876, 1.0118036, 0.18307444],
                                [1.9479709, 0.96544015, 1.0476727, 0.19277772],
                                [1.795729, 0.8969963, 1.224472, 0.18180896],
                                [2.2865427, 1.1570745, 0.9713619, 0.211611],
                                [2.1911612, 1.0922736, 1.3791639, 0.2307278]]]
              },
              'atten_context': [[0.8048796, 0.9554154, 1.2422264, 0.82598877],
                                [1.1976988, 0.9226365, 1.1311831, 1.1287751],
                                [1.2583418, 0.96984935, 0.8972859, 1.2939383],
                                [0.8055052, 0.9545301, 1.2421954, 0.824931],
                                [1.1980952, 0.9227077, 1.1313919, 1.13009],
                                [1.2582378, 0.96980226, 0.8973369, 1.2938937]]
          })

  def testAttentionStepMultiSourceDifferent(self):
    with self.session(use_gpu=False):
      np.random.seed(12345)
      src_batch_size = 3
      target_batch_size = 6
      src_length = 5
      query_dim = 5
      src_dim = 4
      source_vecs_0 = tf.constant(
          np.random.rand(src_length, src_batch_size, src_dim), dtype=tf.float32)
      source_vecs_1 = tf.constant(
          np.random.rand(src_length, src_batch_size, src_dim), dtype=tf.float32)
      sources = py_utils.NestedMap(
          source_0=source_vecs_0, source_1=source_vecs_1)

      source_padding_0 = tf.zeros([src_length, src_batch_size],
                                  dtype=tf.float32)
      source_padding_1 = tf.zeros([src_length, src_batch_size],
                                  dtype=tf.float32)
      source_paddings = py_utils.NestedMap(
          source_0=source_padding_0, source_1=source_padding_1)
      query_vec = tf.constant(
          np.random.rand(target_batch_size, query_dim), dtype=tf.float32)

      p = attention_steps.AttentionStep.Params()

      # Setup MultiSourceAttention
      p.atten = attention.MultiSourceAttention.Params()
      p.atten.source_dim = src_dim
      p.atten.query_dim = query_dim

      add_atten_params = attention.AdditiveAttention.Params()
      add_atten_params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      add_atten_params.source_dim = src_dim
      add_atten_params.query_dim = query_dim
      add_atten_params.hidden_dim = query_dim
      add_atten_params.vn.global_vn = False
      add_atten_params.vn.per_step_vn = False
      add_atten_params.packed_input = True

      gmm_atten_params = attention.GmmMonotonicAttention.Params()
      gmm_atten_params.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      gmm_atten_params.source_dim = src_dim
      gmm_atten_params.query_dim = query_dim
      gmm_atten_params.hidden_dim = query_dim
      gmm_atten_params.vn.global_vn = False
      gmm_atten_params.vn.per_step_vn = False
      gmm_atten_params.packed_input = True

      p.atten.source_atten_tpls = [('source_0', add_atten_params),
                                   ('source_1', gmm_atten_params)]

      step = p.Instantiate()

      external_inputs = py_utils.NestedMap(src=sources, padding=source_paddings)
      packed = step.PrepareExternalInputs(step.theta, external_inputs)
      state0 = step.ZeroState(step.theta, packed, target_batch_size)

      step_inputs = py_utils.NestedMap(inputs=[query_vec])
      step_padding = tf.zeros([target_batch_size, 1], dtype=tf.float32)
      output, state1 = step.FProp(step.theta, packed, step_inputs, step_padding,
                                  state0)

      self.evaluate(tf.global_variables_initializer())
      output, state1 = self.evaluate([output, state1])

      self.assertAllClose(
          output, {
              'context': [[0.9140804, 0.8979037, 1.1033492, 0.70460725],
                          [1.1748682, 1.0488822, 1.2771418, 1.0938747],
                          [1.2568944, 1.0808113, 0.9878455, 1.4196949],
                          [0.9142588, 0.8978502, 1.1039352, 0.7042637],
                          [1.174994, 1.0493405, 1.2779118, 1.0942582],
                          [1.2567302, 1.0806134, 0.98783255, 1.4195559]],
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
              'atten_state': {
                  'source_0': [[0.], [0.], [0.], [0.], [0.], [0.]],
                  'source_1': [[[2.4243412, 1.2218076, 1.0122609, 0.18427502],
                                [1.9546769, 0.9721461, 1.0469768, 0.19244196],
                                [1.7934805, 0.8947478, 1.2158467, 0.18101364],
                                [2.2727895, 1.1433213, 0.969053, 0.21098366],
                                [2.1986299, 1.0997422, 1.3713341, 0.23128569]],
                               [[2.4298353, 1.227302, 1.0116383, 0.18391277],
                                [1.9476058, 0.96507514, 1.0462759, 0.19275317],
                                [1.793545, 0.89481235, 1.220826, 0.1817861],
                                [2.2800756, 1.1506072, 0.96794796, 0.21093304],
                                [2.194984, 1.0960963, 1.3741415, 0.23061496]],
                               [[2.4273272, 1.2247936, 1.0106387, 0.18302175],
                                [1.9522938, 0.96976304, 1.0510013, 0.19241981],
                                [1.7976122, 0.8988795, 1.2246737, 0.18208173],
                                [2.2875524, 1.1580843, 0.97309643, 0.21170339],
                                [2.1904838, 1.0915961, 1.3786552, 0.23077331]],
                               [[2.4339817, 1.2314482, 1.0118915, 0.18239658],
                                [1.9538436, 0.9713129, 1.050209, 0.19243228],
                                [1.7997689, 0.90103614, 1.2248727, 0.18208562],
                                [2.286818, 1.15735, 0.9776513, 0.2125737],
                                [2.1872034, 1.0883157, 1.3807379, 0.23051178]],
                               [[2.4258854, 1.223352, 1.0136935, 0.18414007],
                                [1.9573982, 0.9748675, 1.0445031, 0.19239089],
                                [1.7965381, 0.89780533, 1.2112961, 0.18159895],
                                [2.2637806, 1.1343125, 0.9743988, 0.21178932],
                                [2.1948628, 1.0959752, 1.366173, 0.23008086]],
                               [[2.435421, 1.2328876, 1.0118036, 0.18307444],
                                [1.9479709, 0.96544015, 1.0476727, 0.19277772],
                                [1.795729, 0.8969963, 1.224472, 0.18180896],
                                [2.2865427, 1.1570745, 0.9713619, 0.211611],
                                [2.1911612, 1.0922736, 1.3791639, 0.2307278]]]
              },
              'atten_context': [[0.9140804, 0.8979037, 1.1033492, 0.70460725],
                                [1.1748682, 1.0488822, 1.2771418, 1.0938747],
                                [1.2568944, 1.0808113, 0.9878455, 1.4196949],
                                [0.9142588, 0.8978502, 1.1039352, 0.7042637],
                                [1.174994, 1.0493405, 1.2779118, 1.0942582],
                                [1.2567302, 1.0806134, 0.98783255, 1.419555]]
          })

  def testAttentionBlockStep(self):
    with self.session(use_gpu=False):
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
          attention=py_utils.NestedMap(src=source_vecs, padding=source_padding))
      packed = step.PrepareExternalInputs(step.theta, external_inputs)
      state0 = step.ZeroState(step.theta, packed, target_batch_size)

      step_padding = tf.zeros([target_batch_size, 1], dtype=tf.float32)
      output, state1 = step.FProp(step.theta, packed, None, step_padding,
                                  state0)

      self.evaluate(tf.global_variables_initializer())
      output, state1 = self.evaluate([output, state1])

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
