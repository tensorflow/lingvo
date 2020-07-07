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
"""Tests for lingvo.core.step."""

from lingvo import compat as tf
from lingvo.core import builder_layers
from lingvo.core import py_utils
from lingvo.core import step
from lingvo.core import test_utils


class TextStep(step.Step):
  """TextStep is a fake step used for testing.

  A real step would perform numeric operations on tensors. Instead, we return
  text strings from these functions. This causes the result of our tests
  to be long text strings that encode the sequence of operations that
  were performed.
  """

  def PrepareExternalInputs(self, theta, external_inputs):
    return external_inputs

  def ZeroState(self, theta, prepared_inputs, batch_size):
    return self.params.name

  def FProp(self, theta, external_inputs, step_inputs, padding, state0):
    return py_utils.NestedMap(
        output=':'.join(step_inputs.inputs + [external_inputs]) + state0), (
            state0 + ':'.join(step_inputs.inputs))


class StepTest(test_utils.TestCase):

  def testStatelessLayerStep(self):
    with self.session():
      p = step.StatelessLayerStep.Params()
      p.name = 'sum'
      p.layer = builder_layers.FnLayer.Params().Set(fn=lambda x: x[0] + x[1])
      s = p.Instantiate()

      self.assertEqual(py_utils.NestedMap(),
                       s.PrepareExternalInputs(s.theta, None))
      self.assertEqual(py_utils.NestedMap(), s.ZeroState(s.theta, None, 8))

      out, state1 = s.FProp(
          s.theta, None,
          py_utils.NestedMap(inputs=[
              tf.constant([4], dtype=tf.int32),
              tf.constant([11], dtype=tf.int32)
          ]), None, None)
      self.assertEqual(py_utils.NestedMap(), state1)
      value = self.evaluate(out)
      self.assertEqual([15], value)

  def testGraphStep(self):
    p = step.GraphStep.Params()
    p.name = 'graphtest'
    p.sub = [
        ('(inputs=[step_inputs.a])->output0', 'external_inputs.x',
         TextStep.Params().Set(name='step0')),
        ('(inputs=[output0.output])->output1', 'external_inputs.y',
         TextStep.Params().Set(name='step1')),
    ]
    p.output_signature = 'output1'
    s = p.Instantiate()

    prepared = s.PrepareExternalInputs(s.theta,
                                       py_utils.NestedMap(x='1', y='2'))
    self.assertEqual({'step0': '1', 'step1': '2'}, prepared)

    state0 = s.ZeroState(s.theta, prepared, 8)
    self.assertEqual({'step0': 'step0', 'step1': 'step1'}, state0)

    output, state1 = s.FProp(s.theta, prepared,
                             py_utils.NestedMap(a='aa', b='bb'),
                             tf.constant([0.0], dtype=tf.float32), state0)
    self.assertEqual({'output': 'aa:1step0:2step1'}, output)
    self.assertEqual({'step1': 'step1aa:1step0', 'step0': 'step0aa'}, state1)

  def testStackStep(self):
    p = step.StackStep.Params()
    p.name = 'stack'
    p.sub = [
        TextStep.Params().Set(name='text0'),
        TextStep.Params().Set(name='text1'),
        TextStep.Params().Set(name='text2'),
    ]
    s = p.Instantiate()

    prepared = s.PrepareExternalInputs(s.theta, 'z')
    self.assertEqual({'sub': ['z', 'z', 'z']}, prepared)
    state0 = s.ZeroState(s.theta, prepared, 1)
    self.assertEqual({'sub': ['text0', 'text1', 'text2']}, state0)
    output, state1 = s.FProp(s.theta, prepared,
                             py_utils.NestedMap(inputs=['in']), [0.0], state0)
    # This output encodes the computations that were performed in the stack:
    #   input = in
    #   layer0: state=text0, external_inputs=z
    #      in -> in:ztext0
    #   layer1: state=text1, external_inputs=z
    #      in:ztext0 -> in:ztext0:ztext1
    #   layer2: state=text2, external_inputs=z
    #      in:ztext0:ztext1 -> in:ztext0:ztext1:ztext2
    self.assertEqual({'output': 'in:ztext0:ztext1:ztext2'}, output)
    # The state1 of each sub-step is equal to its input plus its current state.
    #   layer0: state=text0, input=in
    #   layer1: state=text1, input=in:ztext0
    #   layer2: state=text2, input=in:ztext0:ztext1
    self.assertEqual(
        {'sub': ['text0in', 'text1in:ztext0', 'text2in:ztext0:ztext1']}, state1)

  def testStackStepWithResidualConnections(self):
    p = step.StackStep.Params()
    p.name = 'stack'
    p.sub = [
        TextStep.Params().Set(name='text0'),
        TextStep.Params().Set(name='text1'),
        TextStep.Params().Set(name='text2'),
    ]
    p.residual_start = 1
    s = p.Instantiate()

    prepared = s.PrepareExternalInputs(s.theta, 'z')
    self.assertEqual({'sub': ['z', 'z', 'z']}, prepared)
    state0 = s.ZeroState(s.theta, prepared, 1)
    self.assertEqual({'sub': ['text0', 'text1', 'text2']}, state0)
    output, state1 = s.FProp(s.theta, prepared,
                             py_utils.NestedMap(inputs=['in']), [0.0], state0)
    # This output encodes the computations that were performed in the stack:
    #  input=in
    #  layer0: state=text0, external_inputs=z
    #    in -> in:ztext0
    #  layer1: state=text1, external_inputs=z
    #    in:ztext0 -> in:ztext0:ztext1 + in:ztext0 (residual)
    #   layer2: state=text2, external_inputs=z
    #    in:ztext0:ztext1in:ztext0 -> in:ztext0:ztext1in:ztext0:ztext2 +
    #        in:ztext0:ztext1in:ztext0 (residual)
    self.assertEqual(
        {'output': 'in:ztext0:ztext1in:ztext0:ztext2in:ztext0:ztext1in:ztext0'},
        output)
    # The state1 of each sub-step is equal to its input plus its current state.
    #   layer0: state=text0, input=in
    #   layer1: state=text1, input=in:ztext0
    #   layer2: state=text2, input=in:ztext0:ztext1in:ztext0
    self.assertEqual(
        {
            'sub':
                ['text0in', 'text1in:ztext0', 'text2in:ztext0:ztext1in:ztext0']
        }, state1)

  def testParallelStep(self):

    def PlusConstantParams(n):
      p = step.StatelessLayerStep.Params()
      p.name = 'sum'
      p.layer = builder_layers.FnLayer.Params().Set(fn=lambda x: x + n)
      return p

    with self.session():
      p = step.ParallelStep.Params()
      p.name = 'concat'
      p.sub = [PlusConstantParams(1), PlusConstantParams(2)]
      concat = p.Instantiate()

      prepared = concat.PrepareExternalInputs(concat.theta,
                                              py_utils.NestedMap())
      state0 = concat.ZeroState(concat.theta, prepared, 3)
      step_inputs = py_utils.NestedMap(
          inputs=tf.constant([[5], [10], [15]], dtype=tf.float32))
      output, _ = concat.FProp(concat.theta, prepared, step_inputs, None,
                               state0)
      output = self.evaluate(output)
      # Input is batch size 3: [[5], [10], [15]].
      # Two separate steps run on each input, +1 and +2, and the result is
      # concatenated.
      self.assertAllClose(output.output, [[6, 7], [11, 12], [16, 17]])

  def testIteratorStep(self):
    with self.session():
      p = step.IteratorStep.Params()
      p.name = 'iterator'
      iterator = p.Instantiate()

      prepared = iterator.PrepareExternalInputs(
          iterator.theta,
          py_utils.NestedMap(
              a=tf.constant(
                  [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 1], [2, 3]]],
                  dtype=tf.int32)))
      state = iterator.ZeroState(iterator.theta, prepared, 2)

      outputs = []
      for _ in range(3):
        out, state = iterator.FProp(iterator.theta, prepared,
                                    py_utils.NestedMap(), None, state)
        outputs.append(out)
      outputs = self.evaluate(outputs)
      self.assertAllClose([{
          'a': [[1, 2], [7, 8]]
      }, {
          'a': [[3, 4], [9, 1]]
      }, {
          'a': [[5, 6], [2, 3]]
      }], outputs)

if __name__ == '__main__':
  tf.test.main()
