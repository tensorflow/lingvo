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
"""Tests for lingvo.core.steps.rnn_steps."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import step
from lingvo.core import test_utils
from lingvo.core.steps import rnn_steps


class RnnStepsTest(test_utils.TestCase):

  def testRnnStep(self):
    with self.session(use_gpu=False):
      p = rnn_steps.RnnStep.Params()
      p.name = 'rnn_step'
      p.cell.name = 'rnn_step_cell'
      p.cell.output_nonlinearity = True
      p.cell.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.cell.bias_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.cell.vn.global_vn = False
      p.cell.vn.per_step_vn = False
      p.cell.num_input_nodes = 2
      p.cell.num_output_nodes = 2
      p.cell.inputs_arity = 2

      rnn_step = p.Instantiate()
      external = tf.constant([[3]], tf.float32)
      packed = rnn_step.PrepareExternalInputs(rnn_step.theta, external)

      state0 = rnn_step.ZeroState(rnn_step.theta, packed, 1)
      _, state1 = rnn_step.FProp(
          rnn_step.theta, packed,
          py_utils.NestedMap(inputs=[tf.constant([[4]], tf.float32)]),
          tf.constant([0.0], dtype=tf.float32), state0)
      out2, state2 = rnn_step.FProp(
          rnn_step.theta, packed,
          py_utils.NestedMap(inputs=[tf.constant([[4]], tf.float32)]),
          tf.constant([0.0], dtype=tf.float32), state1)

      self.evaluate(tf.global_variables_initializer())
      out2, state2 = self.evaluate([out2, state2])

      self.assertAllClose(state2.m, [[-0.32659757, 0.87739915]])
      self.assertAllClose(state2.c, [[-1.9628618, 1.4194499]])
      self.assertAllClose(out2.output, [[-0.32659757, 0.87739915]])
      self.assertAllClose(out2.padding, [0.0])

  def testRnnStepRecurrent(self):
    with self.session(use_gpu=False):
      p = rnn_steps.RnnStep.Params()
      p.name = 'rnn_step'
      p.cell.name = 'rnn_step_cell'
      p.cell.output_nonlinearity = True
      p.cell.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.cell.bias_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.cell.vn.global_vn = False
      p.cell.vn.per_step_vn = False
      p.cell.num_input_nodes = 2
      p.cell.num_output_nodes = 2
      p.cell.inputs_arity = 2

      recurrent_p = step.RecurrentStepWrapper.Params().Set(
          name='recurrent_wrapper', step=p)
      recurrent_step = recurrent_p.Instantiate()

      external = tf.constant([[3]], tf.float32)
      packed = recurrent_step.PrepareExternalInputs(recurrent_step.theta,
                                                    external)
      inputs = py_utils.NestedMap(
          inputs=[tf.constant([[[4]], [[4]]], tf.float32)])
      padding = tf.constant([[0.0], [0.0]], dtype=tf.float32)
      state0 = recurrent_step.ZeroState(recurrent_step.theta, packed, 1)

      output, state2 = recurrent_step.FProp(
          recurrent_step.theta,
          prepared_inputs=packed,
          inputs=inputs,
          padding=padding,
          state0=state0)

      self.evaluate(tf.global_variables_initializer())
      output, state2 = self.evaluate([output, state2])
      self.assertAllClose(state2.m,
                          [[[-0.20144, 0.741861]], [[-0.326598, 0.877399]]])
      self.assertAllClose(state2.c,
                          [[[-0.980946, 0.973391]], [[-1.962862, 1.41945]]])
      self.assertAllClose(output.output,
                          [[[-0.20144, 0.741861]], [[-0.326598, 0.877399]]])

  def testRnnStackStep(self):
    with self.session(use_gpu=False):
      p = rnn_steps.RnnStackStep.Params()
      p.name = 'rnn_stack_step'
      p.rnn_cell_tpl.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.rnn_cell_tpl.bias_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.rnn_cell_tpl.vn.global_vn = False
      p.rnn_cell_tpl.vn.per_step_vn = False
      p.external_input_dim = 2
      p.step_input_dim = 1
      p.context_input_dim = 1
      p.rnn_cell_dim = 3
      p.rnn_cell_hidden_dim = 3
      p.rnn_layers = 2
      rnn_stack = p.Instantiate()

      external = tf.constant([[1, 2]], tf.float32)
      packed = rnn_stack.PrepareExternalInputs(rnn_stack.theta, external)
      state0 = rnn_stack.ZeroState(rnn_stack.theta, packed, 1)
      output1, state1 = rnn_stack.FProp(
          rnn_stack.theta, packed,
          py_utils.NestedMap(
              inputs=[tf.constant([[4]], tf.float32)],
              context=tf.constant([[5]], tf.float32)),
          tf.constant([0.0], dtype=tf.float32), state0)

      self.evaluate(tf.global_variables_initializer())
      output1, state1 = self.evaluate([output1, state1])

      self.assertAllClose(output1.output,
                          [[0.43175745, -0.39472747, -0.36191428]])
      self.assertAllClose(
          state1, {
              'sub': [{
                  'm': [[-0.4587491, 0.56409806, 0.23025148]],
                  'c': [[5.6926787e-01, 6.3084178e-02, 4.1969700e-04]]
              }, {
                  'm': [[0.43175745, -0.39472747, -0.36191428]],
                  'c': [[0.00158867, 0.57818687, -0.98025495]]
              }]
          })

  def testRnnStackStepResidual(self):
    with self.session(use_gpu=False):
      p = rnn_steps.RnnStackStep.Params()
      p.name = 'rnn_stack_step'
      p.rnn_cell_tpl.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.rnn_cell_tpl.bias_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.rnn_cell_tpl.vn.global_vn = False
      p.rnn_cell_tpl.vn.per_step_vn = False
      p.external_input_dim = 2
      p.step_input_dim = 1
      p.context_input_dim = 1
      p.rnn_cell_dim = 3
      p.rnn_cell_hidden_dim = 3
      p.rnn_layers = 3
      p.residual_start = 0
      rnn_stack = p.Instantiate()

      external = tf.constant([[1, 2]], tf.float32)
      packed = rnn_stack.PrepareExternalInputs(rnn_stack.theta, external)
      state0 = rnn_stack.ZeroState(rnn_stack.theta, packed, 1)
      output1, state1 = rnn_stack.FProp(
          rnn_stack.theta, packed,
          py_utils.NestedMap(
              inputs=[tf.constant([[4]], tf.float32)],
              context=tf.constant([[5]], tf.float32)),
          tf.constant([0.0], dtype=tf.float32), state0)

      self.evaluate(tf.global_variables_initializer())
      output1, state1 = self.evaluate([output1, state1])

      # Because there are residual connections, we expect the output to
      # be equal to sum(rnn_state.m for rnn_state in state1.sub).
      self.assertAllClose(output1.output, [[3.5457525, 4.576743, 4.2290816]])
      self.assertAllClose(
          state1, {
              'sub': [{
                  'm': [[-0.4587491, 0.56409806, 0.23025148]],
                  'c': [[5.6926787e-01, 6.3084178e-02, 4.1969700e-04]]
              }, {
                  'm': [[0.00226134, 0.00633766, -0.00058896]],
                  'c': [[0.00088014, 0.07547389, 0.00625527]]
              }, {
                  'm': [[0.00224019, 0.00630755, -0.00058099]],
                  'c': [[0.00088081, 0.07505893, 0.00621293]]
              }]
          })

  def testRnnStackStepNoContext(self):
    with self.session(use_gpu=False):
      p = rnn_steps.RnnStackStep.Params()
      p.name = 'rnn_stack_step'
      p.rnn_cell_tpl.params_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.rnn_cell_tpl.bias_init = py_utils.WeightInit.Uniform(1.24, 429891685)
      p.rnn_cell_tpl.vn.global_vn = False
      p.rnn_cell_tpl.vn.per_step_vn = False
      p.step_input_dim = 1
      p.rnn_cell_dim = 3
      p.rnn_cell_hidden_dim = 3
      p.rnn_layers = 2
      p.residual_start = 0
      rnn_stack = p.Instantiate()

      packed = rnn_stack.PrepareExternalInputs(rnn_stack.theta,
                                               py_utils.NestedMap())
      state0 = rnn_stack.ZeroState(rnn_stack.theta, packed, 1)
      output1, state1 = rnn_stack.FProp(
          rnn_stack.theta, packed,
          py_utils.NestedMap(inputs=[tf.constant([[4]], tf.float32)]),
          tf.constant([0.0], dtype=tf.float32), state0)

      self.evaluate(tf.global_variables_initializer())
      output1, state1 = self.evaluate([output1, state1])

      self.assertAllClose(output1.output, [[5.900284, 3.0231729, 3.0207822]])
      self.assertAllClose(
          state1, {
              'sub': [{
                  'm': [[1.1416901, -0.32166323, -0.5909376]],
                  'c': [[-0.98086286, 0.9052862, 0.10041453]]
              }, {
                  'm': [[0.7585938, -0.655164, -0.3882802]],
                  'c': [[-8.3011830e-01, 1.8685710e-01, 1.0723456e-04]]
              }]
          })


if __name__ == '__main__':
  tf.test.main()
