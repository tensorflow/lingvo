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
"""Tests for recurrent."""

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import layers as lingvo_layers
from lingvo.core import py_utils
from lingvo.core import recurrent
from lingvo.core import symbolic
from lingvo.core import test_utils
import numpy as np

FLAGS = tf.flags.FLAGS


def _ApplyPadding(padding, v_no_pad, v_pad):
  if padding is not None:
    padding = tf.cast(padding, v_no_pad.dtype)
    return (1 - padding) * v_no_pad + padding * v_pad
  return v_no_pad


def _ReferenceStaticUnroll(theta, state0, inputs, cell_fn):
  """Statically unrolls a `cell_fn` wrt inputs as `recurrent.Recurrent()` does.

  This is not a complete implementation but should work reasonably for
  calls that do not have a custom cell_grad, extras or accumulators and
  can help check expectations.

  Args:
    theta: weights. A `.NestedMap`.
    state0: initial state. A `.NestedMap`.
    inputs: inputs. A `.NestedMap`. Tensors must have static shape.
    cell_fn: A python function, which computes::
      state1, extras = cell_fn(theta, state0, inputs[t, :])

  Returns:
    accumulate and final state.
  """
  flat_inputs = inputs.Flatten()
  flat_state = state0.Flatten()
  step_count = flat_inputs[0].shape[0]
  acc_state = [[None] * step_count for t in flat_state]
  for i in range(step_count):
    flat_step_inputs = [t[i] for t in flat_inputs]
    step_inputs = inputs.Pack(flat_step_inputs)
    state1, unused_extras = cell_fn(theta, state0, step_inputs)
    for j, state_step in zip(range(len(acc_state[0])), state1.Flatten()):
      acc_state[j][i] = state_step
    state0 = state1
  return state0.Pack(acc_state), state1


def _Poly(theta, state, inputs):
  next_state = py_utils.NestedMap()
  next_state.value = state.value + inputs.coeff * state.x_power
  next_state.x_power = state.x_power * theta.x
  return next_state, py_utils.NestedMap()


class _IncrementAccumulator(base_layer.Accumulator):

  def DefaultValue(self):
    return tf.convert_to_tensor(0.0)

  def Update(self, increment_by):
    initial = self.GetValue()
    self.SetValue(initial + tf.convert_to_tensor(increment_by))


class _SampleAccumulatorLayer(base_layer.BaseLayer):

  def __init__(self, params):
    super().__init__(params)
    self.accumulator_name = 'sample_accumulator'
    self.RegisterAccumulator(self.accumulator_name, _IncrementAccumulator())


def Elman(theta, state0, inputs):
  h0, w, b, x = state0.h, theta.w, theta.b, inputs.x
  xw = py_utils.Matmul(tf.concat([x, h0], axis=1), w)  # 1st part
  # 2nd part
  padding = inputs.get('padding', None)
  h1 = _ApplyPadding(padding, v_no_pad=tf.sigmoid(xw + b), v_pad=state0.h)

  state1 = py_utils.NestedMap(h=h1)
  if padding is not None:
    state1.padding = inputs.padding

  return (state1, py_utils.NestedMap(h=h1))


def ElmanGrad(theta, state0, inputs, extras, dstate1):

  @tf.function(autograph=False)
  def Grad(h0, w, b, x, padding, h1, dh1):
    del b
    dh1_orig = dh1
    dh1 = _ApplyPadding(padding, dh1, tf.zeros_like(dh1, dtype=dh1.dtype))

    # We hand-roll the gradient for the 2nd half of the cell as a demo.
    # h1 = tf.sigmoid(xw + b)
    # 𝛔'(x) = ((1 - 𝛔(x)) * 𝛔(x))
    dxwb = (dh1 * (1 - h1) * h1)
    dxw, db = dxwb, tf.reduce_sum(dxwb, axis=0)

    # Uses tf.gradient for the 1nd half of the cell as a demo.
    xw = py_utils.Matmul(tf.concat([x, h0], axis=1), w)
    dh0, dx, dw = tf.gradients(ys=[xw], xs=[h0, x, w], grad_ys=[dxw])

    dh0 = _ApplyPadding(padding, dh0, dh1_orig)

    return dh0, dx, dw, db

  dh0, dx, dw, db = Grad(state0.h, theta.w, theta.b, inputs.x,
                         inputs.get('padding', 0), extras.h, dstate1.h)
  dstate0 = py_utils.NestedMap(h=dh0)
  dinputs = py_utils.NestedMap(x=dx)
  if 'padding' in dstate1:
    dstate0.padding = dstate1.padding
    dinputs.padding = dstate1.padding
  return (py_utils.NestedMap(w=dw, b=db), dstate0, dinputs, None)


def ElmanOut(state1):
  return py_utils.NestedMap(x=state1.h, padding=state1.padding)


def ElmanOutGrad(dout):
  return py_utils.NestedMap(h=dout.x, padding=dout.padding)


class RecurrentTest(test_utils.TestCase, parameterized.TestCase):

  def testBasic(self):

    with self.session(use_gpu=True):

      theta = py_utils.NestedMap()
      theta.x = tf.constant(2.0)
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      state.x_power = tf.constant(1.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])

      # x = 2
      # 1 + 2*x + 3*x^2
      ret = recurrent.Recurrent(theta, state, inputs, _Poly)

      acc, state = self.evaluate(ret)
      self.assertAllClose(acc.value, [1., 5., 17.])
      self.assertAllClose(acc.x_power, [2., 4., 8.])
      self.assertAllClose(state.value, 17.)
      self.assertAllClose(state.x_power, 8.)

      y = ret[1].value
      dx, d_coeff = tf.gradients(ys=[y], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])

      # 2 + 6*x
      self.assertAllClose(dx_val, 14.)
      self.assertAllClose(d_coeff_val, [1., 2., 4.])

      # acc = [1, 1+2x, 1+2x+3x^2]
      # sum(acc) = 3 + 4x + 3x^2
      acc = ret[0].value
      dx, d_coeff = tf.gradients(
          ys=[tf.reduce_sum(acc)], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])
      # 4 + 6*x
      self.assertAllClose(dx_val, 16.)
      self.assertAllClose(d_coeff_val, [3., 4., 4.])

  def testBasicWithAccumulator(self):

    with self.session(use_gpu=True):

      p = _SampleAccumulatorLayer.Params()
      p.name = 'sample'
      accum_layer = _SampleAccumulatorLayer(p)
      accum_obj = accum_layer.accumulators[accum_layer.accumulator_name]

      theta = py_utils.NestedMap()
      theta.x = tf.constant(2.0)
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      state.x_power = tf.constant(1.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])

      def _CellFn(theta, state, inputs):
        print('TEST ACCUM WITHIN CellFn = ', accum_obj.GetValue())
        accum_obj.Update(inputs.coeff)
        return _Poly(theta, state, inputs)

      # By doing one accumulate prior to recurrent, we ensure that incoming
      # recurrent state is preserved.
      accum_obj.Update(10.)

      # x = 2
      # 1 + 2*x + 3*x^2
      ret = recurrent.Recurrent(
          theta, state, inputs, _CellFn, accumulator_layer=accum_layer)

      # Verify bprop.
      y = ret[1].value
      dx, d_coeff = tf.gradients(ys=[y], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])

      # 2 + 6*x
      self.assertAllClose(dx_val, 14.)
      self.assertAllClose(d_coeff_val, [1., 2., 4.])

      # acc = [1, 1+2x, 1+2x+3x^2]
      # sum(acc) = 3 + 4x + 3x^2
      acc = ret[0].value
      dx, d_coeff = tf.gradients(
          ys=[tf.reduce_sum(acc)], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])
      # 4 + 6*x
      self.assertAllClose(dx_val, 16.)
      self.assertAllClose(d_coeff_val, [3., 4., 4.])

      # Verify fprop.
      (acc, state), accum_obj_value = self.evaluate((ret, accum_obj.GetValue()))

      # Verify that accumulators don't change fprop results.
      self.assertAllClose(acc.value, [1., 5., 17.])
      self.assertAllClose(acc.x_power, [2., 4., 8.])
      self.assertAllClose(state.value, 17.)
      self.assertAllClose(state.x_power, 8.)

      # Verify accumulator (should be 10 (initial increment) + 1 + 2 + 3).
      self.assertEqual(0, accum_obj._disable_count)
      self.assertAllClose([accum_obj_value], [16.0])

  def testTimeBasedStopFn(self):

    with self.session(use_gpu=True):

      def StopFn(t, unused_theta, unused_state):
        # This stops after 3 iterations.
        return t >= 3

      theta = py_utils.NestedMap()
      theta.x = tf.constant(2.0)
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      state.x_power = tf.constant(1.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3., 4.])

      # x = 2
      # 1 + 2*x + 3*x^2
      ret = recurrent.Recurrent(theta, state, inputs, _Poly, stop_fn=StopFn)

      acc, state = self.evaluate(ret)
      self.assertAllClose([1., 5., 17., 0.], acc.value)
      self.assertAllClose([2., 4., 8., 0.], acc.x_power)
      self.assertAllClose(17., state.value)
      self.assertAllClose(8., state.x_power)

      y = ret[1].value
      dx, d_coeff = tf.gradients(ys=[y], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])

      # 2 + 6*x
      self.assertAllClose(14., dx_val)
      self.assertAllClose([1., 2., 4., 0.], d_coeff_val)

      # acc = [1, 1+2x, 1+2x+3x^2]
      # sum(acc) = 3 + 4x + 3x^2
      acc = ret[0].value
      dx, d_coeff = tf.gradients(
          ys=[tf.reduce_sum(acc)], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])
      # 4 + 6*x
      self.assertAllClose(16., dx_val)
      self.assertAllClose([3., 4., 4., 0.], d_coeff_val)

  def testStateBasedStopFn(self):

    with self.session(use_gpu=True):

      def StopFn(unused_t, unused_theta, state):
        # This stops after 3 iterations.
        return state.value >= 15.

      theta = py_utils.NestedMap()
      theta.x = tf.constant(2.0)
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      state.x_power = tf.constant(1.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3., 4.])

      # x = 2
      # 1 + 2*x + 3*x^2
      ret = recurrent.Recurrent(theta, state, inputs, _Poly, stop_fn=StopFn)

      acc, state = self.evaluate(ret)
      self.assertAllClose([1., 5., 17., 0.], acc.value)
      self.assertAllClose([2., 4., 8., 0.], acc.x_power)
      self.assertAllClose(17., state.value)
      self.assertAllClose(8., state.x_power)

      y = ret[1].value
      dx, d_coeff = tf.gradients(ys=[y], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])

      # 2 + 6*x
      self.assertAllClose(14., dx_val)
      self.assertAllClose([1., 2., 4., 0.], d_coeff_val)

      # acc = [1, 1+2x, 1+2x+3x^2]
      # sum(acc) = 3 + 4x + 3x^2
      acc = ret[0].value
      dx, d_coeff = tf.gradients(
          ys=[tf.reduce_sum(acc)], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])
      # 4 + 6*x
      self.assertAllClose(16., dx_val)
      self.assertAllClose([3., 4., 4., 0.], d_coeff_val)

  def testStopFnNotTriggeredBeforeEOS(self):

    with self.session(use_gpu=True):

      def StopFn(t, unused_theta, unused_state):
        # The input sequence is only length 4, so this is never true.
        # However, the Recurrent call should still terminate after iteration 4.
        return t >= 5

      theta = py_utils.NestedMap()
      theta.x = tf.constant(2.0)
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      state.x_power = tf.constant(1.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3., 4.])

      # x = 2
      # 1 + 2*x + 3*x^2 + 4*x^3
      ret = recurrent.Recurrent(theta, state, inputs, _Poly, stop_fn=StopFn)

      acc, state = self.evaluate(ret)
      self.assertAllClose([1., 5., 17., 49.], acc.value)
      self.assertAllClose([2., 4., 8., 16.], acc.x_power)
      self.assertAllClose(49., state.value)
      self.assertAllClose(16., state.x_power)

      y = ret[1].value
      dx, d_coeff = tf.gradients(ys=[y], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])

      # 2 + 6*x + 12*x^2
      self.assertAllClose(62., dx_val)
      self.assertAllClose([1., 2., 4., 8.], d_coeff_val)

      # acc = [1, 1+2x, 1+2x+3x^2, 1+2x+3x^2+4x^3]
      # sum(acc) = 4 + 6x + 6x^2 + 4x^3
      acc = ret[0].value
      dx, d_coeff = tf.gradients(
          ys=[tf.reduce_sum(acc)], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = self.evaluate([dx, d_coeff])
      # 6 + 12*x + 12*x^2
      self.assertAllClose(78., dx_val)
      self.assertAllClose([4., 6., 8., 8.], d_coeff_val)

  def testCapture(self):

    with self.session(use_gpu=True):

      theta = py_utils.NestedMap()
      theta.x = tf.constant(2.0)
      state0 = py_utils.NestedMap()
      state0.value = tf.constant(0.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])
      captured = tf.constant(1.2, name='captured_const')

      def CellFn(theta, state, inputs):
        next_state = py_utils.NestedMap()
        # Captured is pulled from outside of the function and captured via the
        # internal Defun.
        next_state.value = state.value + inputs.coeff * captured * theta.x
        return next_state, py_utils.NestedMap()

      # Run static fprop reference implementation.
      ref_acc, ref_staten = _ReferenceStaticUnroll(theta, state0, inputs,
                                                   CellFn)
      ref_acc_values, ref_staten_values = self.evaluate((ref_acc, ref_staten))
      print('Ref fprop: acc =', ref_acc, ', stateN =', ref_staten)
      print('Ref fprop values: acc =', ref_acc_values, ', stateN =',
            ref_staten_values)
      self.assertAllClose(ref_acc_values.value, [2.4, 7.2, 14.4])
      self.assertAllClose(ref_staten_values.value, 14.4)

      # Run real fprop implementation.
      real_acc, real_staten = recurrent.Recurrent(
          theta, state0, inputs, CellFn, allow_implicit_capture=True)
      real_acc_values, real_staten_values = self.evaluate(
          (real_acc, real_staten))
      print('Real fprop: acc =', real_acc, ', stateN =', real_staten)
      print('Real fprop values: acc =', real_acc_values, ', stateN =',
            real_staten_values)
      self.assertAllClose(ref_acc_values.value, real_acc_values.value)
      self.assertAllClose(ref_staten_values.value, real_staten_values.value)

      # BProp real vs ref of stateN.
      ref_dx, ref_dcaptured = tf.gradients(
          ys=[ref_staten.value], xs=[theta.x, captured])
      ref_dx_values, ref_dcaptured_values = self.evaluate(
          [ref_dx, ref_dcaptured])
      real_dx, real_dcaptured = tf.gradients(
          ys=[real_staten.value], xs=[theta.x, captured])
      real_dx_values, real_dcaptured_values = self.evaluate(
          [real_dx, real_dcaptured])
      print('Ref Dstate/[dx,dcaptured] =', ref_dx_values, ', ',
            ref_dcaptured_values)
      print('Real Dstate/[dx,dcaptured] =', real_dx_values, ', ',
            real_dcaptured_values)
      self.assertAllClose(ref_dx_values, 7.2)
      self.assertAllClose(ref_dcaptured_values, 12.0)
      self.assertAllClose(ref_dx_values, real_dx_values)
      self.assertAllClose(ref_dcaptured_values, real_dcaptured_values)

  def testCaptureDisallowed(self):

    with self.session(use_gpu=True):

      theta = py_utils.NestedMap()
      theta.x = tf.constant(2.0)
      state0 = py_utils.NestedMap()
      state0.value = tf.constant(0.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])
      captured = tf.constant(1.2, name='captured_const')

      def CellFn(theta, state, inputs):
        next_state = py_utils.NestedMap()
        # Captured is pulled from outside of the function and captured via the
        # internal Defun.
        next_state.value = state.value + inputs.coeff * captured * theta.x
        return next_state, py_utils.NestedMap()

      # Run real fprop implementation.
      with self.assertRaisesRegex(ValueError, 'implicit capture is disabled'):
        unused_real_acc, unused_real_staten = recurrent.Recurrent(
            theta, state0, inputs, CellFn, allow_implicit_capture=False)

  def testSymbolToTensorMap(self):
    """Tests that cell_fn can rely on the contextual symbol-to-tensor map."""

    x = symbolic.Symbol('x')
    y = symbolic.Symbol('y')

    def PlusWXT(theta, state, inputs):
      """state.value += theta.w * x * inputs.t."""
      next_state = py_utils.NestedMap()
      x_tensor = symbolic.EvalExpr(symbolic.TENSOR_VALUES, x)
      next_state.value = state.value + theta.w * x_tensor * inputs.t
      return next_state, py_utils.NestedMap()

    def PlusWXTGrad(theta, state0, inputs, extras, dstate1):
      """Gradient function for PlusWXT."""
      del state0, extras
      x_tensor = symbolic.EvalExpr(symbolic.TENSOR_VALUES, x)
      dtheta = py_utils.NestedMap(w=dstate1.value * x_tensor * inputs.t)
      dstate0 = py_utils.NestedMap(value=dstate1.value)
      dinputs = py_utils.NestedMap(t=dstate1.value * theta.w * x_tensor)
      return dtheta, dstate0, dinputs, None

    with self.session(use_gpu=True):
      theta = py_utils.NestedMap(w=tf.constant(1., name='w'))
      state0 = py_utils.NestedMap(value=tf.constant(0., name='value'))
      inputs = py_utils.NestedMap(t=tf.constant([1., 2., 3.], name='t'))

      # With automatic cell_grad.
      with symbolic.SymbolToValueMap(symbolic.TENSOR_VALUES, {
          x: tf.constant(7., name='x7'),
          y: 8
      }):
        x_tensor = symbolic.EvalExpr(symbolic.TENSOR_VALUES, x)
        _, state1 = recurrent.Recurrent(theta, state0, inputs, PlusWXT)
        dw = tf.gradients(ys=[state1.value], xs=[theta.w])[0]
        dx = tf.gradients(ys=[state1.value], xs=[x_tensor])[0]
        final_value, x_val, dx_val, dw_val = self.evaluate(
            [state1.value, x_tensor, dx, dw])
      self.assertEqual(x_val, 7)
      self.assertEqual(final_value, x_val * (1. + 2. + 3.))
      self.assertEqual(dw_val, x_val * (1. + 2. + 3.))
      self.assertEqual(dx_val, (1. + 2. + 3.))

      # With manual cell_grad.
      with symbolic.SymbolToValueMap(symbolic.TENSOR_VALUES,
                                     {x: tf.constant(5., name='x5')}):
        x_tensor = symbolic.EvalExpr(symbolic.TENSOR_VALUES, x)
        _, state1 = recurrent.Recurrent(
            theta, state0, inputs, PlusWXT, cell_grad=PlusWXTGrad)
        dw = tf.gradients(ys=[state1.value], xs=[theta.w])[0]
        dx = tf.gradients(ys=[state1.value], xs=[x_tensor])[0]
        final_value, x_val, dx_val, dw_val = self.evaluate(
            [state1.value, x_tensor, dx, dw])
      self.assertEqual(x_val, 5)
      self.assertEqual(final_value, x_val * (1. + 2. + 3.))
      self.assertEqual(dw_val, x_val * (1. + 2. + 3.))
      self.assertEqual(dx_val, (1. + 2. + 3.))

  def testStatefulCellFn(self):

    def Rand(theta, state, inputs):
      del theta
      next_state = py_utils.NestedMap()
      next_state.value = (
          state.value +
          inputs.coeff * tf.random.uniform(shape=[], dtype=state.value.dtype))
      return next_state, py_utils.NestedMap()

    with self.session(use_gpu=True):
      theta = py_utils.NestedMap()
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])

      with self.assertRaisesRegex(ValueError, 'stateful.*random_uniform'):
        recurrent.Recurrent(theta, state, inputs, Rand, check_stateful_ops=True)

  def testNestedCellFn(self):
    """Tests when cell_fn calls another function."""

    @tf.function(autograph=False)
    def RandWithCoeff(coeff):
      return coeff * tf.random.uniform(shape=[], dtype=coeff.dtype)

    def Rand(theta, state, inputs):
      del theta
      next_state = py_utils.NestedMap()
      next_state.value = state.value + RandWithCoeff(inputs.coeff)
      return next_state, py_utils.NestedMap()

    @tf.function(autograph=False)
    def Coeff(coeff):
      return coeff * 2

    def Deterministic(theta, state, inputs):
      del theta
      next_state = py_utils.NestedMap()
      next_state.value = state.value + Coeff(inputs.coeff)
      return next_state, py_utils.NestedMap()

    with self.session(use_gpu=True):
      theta = py_utils.NestedMap()
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])

      recurrent.Recurrent(
          theta, state, inputs, Deterministic, check_stateful_ops=True)
      with self.assertRaisesRegex(ValueError, 'cell_fn contains stateful ops:'):
        recurrent.Recurrent(theta, state, inputs, Rand, check_stateful_ops=True)

  def testSeqLenActual(self):
    for value, expected in [([[1.0], [1.0], [1.0]],
                             [0, 3]), ([[1.0], [1.0], [0.0]],
                                       [2, 0]), ([[1.0], [0.0], [1.0]], [1, 1]),
                            ([[1.0], [0.0], [0.0]],
                             [1, 0]), ([[0.0], [1.0], [1.0]],
                                       [0, 2]), ([[0.0], [1.0], [0.0]], [0, 0]),
                            ([[0.0], [0.0], [1.0]], [0, 1]), ([[0.0], [0.0],
                                                               [0.0]], [0, 0])]:
      with self.session(use_gpu=True):
        inputs = py_utils.NestedMap()
        inputs.padding = tf.constant(value)

        slen_op = recurrent._SeqPaddingLength(inputs)
        slen, = self.evaluate([slen_op])
        self.assertEqual(expected, slen)

  @staticmethod
  def Rand(shape):
    return tf.random.uniform(shape, minval=-0.2, maxval=0.2, dtype=tf.float64)

  def _testElmanHelper(self, seqlen, use_grad, stop_fn=None):
    with self.session(use_gpu=True):
      tf.random.set_seed(342462)

      batch = 3
      dims = 4
      theta = py_utils.NestedMap()
      theta.w = self.Rand([2 * dims, dims])
      theta.b = self.Rand([dims])
      state0 = py_utils.NestedMap()
      state0.h = self.Rand([batch, dims])
      inputs = py_utils.NestedMap()
      inputs.x = self.Rand([seqlen, batch, dims])

      # Static unrolled.
      s = state0
      out = []
      for i in range(seqlen):
        inp = py_utils.NestedMap()
        inp.x = inputs.x[i, :]
        s, _ = Elman(theta, s, inp)
        out += [s.h]
        if stop_fn and stop_fn(i + 1, theta, s):
          out += [tf.zeros_like(out[-1]) for _ in range(seqlen - i - 1)]
          break
      acc0, final0 = tf.stack(out), s.h
      loss0 = tf.reduce_sum(acc0) + tf.reduce_sum(final0)
      (dw0, db0, dh0,
       di0) = tf.gradients(loss0, [theta.w, theta.b, state0.h, inputs.x])

      # Uses the Recurrent() library.
      acc1, final1 = recurrent.Recurrent(
          theta=theta,
          state0=state0,
          inputs=inputs,
          cell_fn=Elman,
          cell_grad=ElmanGrad if use_grad else None,
          stop_fn=stop_fn)
      acc1, final1 = acc1.h, final1.h
      loss1 = tf.reduce_sum(acc1) + tf.reduce_sum(final1)
      (dw1, db1, dh1,
       di1) = tf.gradients(loss1, [theta.w, theta.b, state0.h, inputs.x])

      # Fetches a bunch of values and compare them.
      (acc0, acc1, final0, final1, dw0, dw1, db0, db1, dh0, dh1, di0,
       di1) = self.evaluate(
           [acc0, acc1, final0, final1, dw0, dw1, db0, db1, dh0, dh1, di0, di1])
      self.assertAllClose(acc0, acc1)
      self.assertAllClose(final0, final1)
      self.assertAllClose(dw0, dw1)
      self.assertAllClose(db0, db1)
      self.assertAllClose(dh0, dh1)
      self.assertAllClose(di0, di1)

  def testElman(self):
    self._testElmanHelper(1, False)
    self._testElmanHelper(1, True)
    self._testElmanHelper(7, False)
    self._testElmanHelper(7, True)

    def StopFn(t, unused_theta, unused_state):
      return t >= 4

    self._testElmanHelper(7, False, StopFn)
    self._testElmanHelper(7, True, StopFn)

  def testSetShape(self):
    dst = py_utils.NestedMap(
        a=tf.placeholder(tf.int32, shape=None),
        b=py_utils.NestedMap(
            b1=tf.placeholder(tf.int32, shape=None),
            b2=tf.placeholder(tf.int32, shape=None)))
    src = py_utils.NestedMap(
        a=tf.constant(0, shape=[2, 4], dtype=tf.int32),
        b=py_utils.NestedMap(
            b1=tf.constant(0, shape=[1, 3], dtype=tf.int32),
            b2=tf.constant(0, shape=[5, 8], dtype=tf.int32)))
    py_utils.SetShapes(dst, src)
    self.assertAllClose(
        [2, 4],
        py_utils.GetShape(dst.a, 2),
    )
    self.assertAllClose([1, 3], py_utils.GetShape(dst.b.b1, 2))
    self.assertAllClose([5, 8], py_utils.GetShape(dst.b.b2, 2))

  @parameterized.parameters(True, False)
  def testDropoutInRecurrent(self, graph_seed):
    with self.session() as sess:
      if graph_seed:
        tf.random.set_seed(12345)
      l = lingvo_layers.DeterministicDropoutLayer.Params().Set(
          name='dropout', keep_prob=0.7).Instantiate()
      # Input variable.
      w = tf.get_variable('w', shape=[9, 20], initializer=tf.ones_initializer())
      sess.run(tf.global_variables_initializer())
      prev_sum = np.sum(np.isclose(sess.run(w), 0.0))

      def Step(theta, state0, unused_inputs):
        w = l.FProp(theta.l, state0.w)
        state1 = py_utils.NestedMap(w=w)
        return state1, py_utils.NestedMap()

      acc, final = recurrent.Recurrent(
          theta=py_utils.NestedMap(l=l.theta),
          state0=py_utils.NestedMap(w=w),
          inputs=py_utils.NestedMap(x=tf.zeros([4])),
          cell_fn=Step)

      acc_w = sess.run(acc.w)
      self.assertLen(acc_w, 4)
      for acc_w_i in acc_w:
        next_sum = np.sum(np.isclose(acc_w_i, 0.0))
        self.assertGreater(next_sum, prev_sum)
        prev_sum = next_sum

      # Construct loss function such that gradients = final activation.
      loss = tf.reduce_sum(final.w)
      grads = py_utils.ComputeGradients(loss, py_utils.NestedMap(w=w))
      w_val, grads_val = sess.run([final.w, grads.w.grad])
      self.assertAllClose(w_val, grads_val)


class StackedRecurrentTest(test_utils.TestCase, parameterized.TestCase):

  @staticmethod
  def Poly(theta, state0, inputs):
    x = theta.x
    s = state0.s
    c = inputs.c
    return py_utils.NestedMap(s=s * x + c), py_utils.NestedMap()

  @staticmethod
  def Identity(theta, state0, inputs):
    del theta, state0
    return py_utils.NestedMap(s=inputs.s), py_utils.NestedMap()

  def testSimpleStacked(self):
    g = tf.Graph()
    with g.as_default():
      devices = ['/cpu:0'] * 3
      cell_fns = [self.Poly, self.Identity, self.Identity]
      cell_grads = [None] * 3
      cell_outs = [lambda x: x] * 3
      cell_out_grads = [lambda x: x] * 3
      w0 = tf.constant(2.)
      w1 = tf.constant(0.)
      w2 = tf.constant(0.)
      thetas = [
          py_utils.NestedMap(x=w0),
          py_utils.NestedMap(x=w1),
          py_utils.NestedMap(x=w2)
      ]
      init_states = [py_utils.NestedMap(s=tf.constant(0.))] * 3
      inputs = py_utils.NestedMap(
          c=tf.constant([1., 2., 1., 0.]),
          padding=tf.constant([0., 0., 0., 1.]))
      output, _ = recurrent.StackedRecurrent(
          devices=devices,
          cell_fns=cell_fns,
          cell_grads=cell_grads,
          cell_outs=cell_outs,
          cell_out_grads=cell_out_grads,
          thetas=thetas,
          init_states=init_states,
          inputs=inputs)
      dw0, dw1, dw2 = tf.gradients(tf.reduce_sum(output.s), [w0, w1, w2])

    with self.session(graph=g, use_gpu=True):
      (output, dw0, dw1, dw2) = self.evaluate([output.s, dw0, dw1, dw2])

    self.assertAllClose(output, [1., 4., 9., 0.])
    self.assertAllClose(dw2, 0.)
    self.assertAllClose(dw1, 0.)
    self.assertAllClose(dw0, 7.)

  def _BuildStackedRecurrentElman(self, seqlen, trailing_pad_len, batch, dims,
                                  layers):
    tf.random.set_seed(342462)
    np.random.seed(32540)

    seqlen += trailing_pad_len
    dtype = tf.float64

    def CreateTheta():
      return py_utils.NestedMap(
          w=tf.constant(
              np.random.uniform(0, 0.2, (2 * dims, dims)), dtype=dtype),
          b=tf.constant(np.random.uniform(0, 0.2, (dims,)), dtype=dtype))

    def CreateState0():
      return py_utils.NestedMap(
          h=tf.constant(np.random.uniform(0, 0.2, (batch, dims)), dtype=dtype),
          padding=tf.constant([[0]] * batch, dtype=dtype))

    devices = ['/cpu:0'] * layers
    cell_fns = [Elman] * layers
    cell_grads = [ElmanGrad] * layers
    cell_outs = [ElmanOut] * layers
    cell_out_grads = [ElmanOutGrad] * layers
    thetas = [CreateTheta() for _ in range(layers)]
    init_states = [CreateState0() for _ in range(layers)]
    padding = np.zeros((seqlen, batch, 1))
    padding[-trailing_pad_len:, :, :] = 1.
    padding[-trailing_pad_len - 3:-trailing_pad_len - 1, :, :] = 1.
    inputs = py_utils.NestedMap(
        x=tf.constant(
            np.random.uniform(0, 0.2, (seqlen, batch, dims)), dtype=dtype),
        padding=tf.constant(padding, dtype=dtype))
    output, _ = recurrent.StackedRecurrent(
        devices=devices,
        cell_fns=cell_fns,
        cell_grads=cell_grads,
        cell_outs=cell_outs,
        cell_out_grads=cell_out_grads,
        thetas=thetas,
        init_states=init_states,
        inputs=inputs)
    o = output.x
    if 'padding' in inputs:
      o *= (1 - inputs.padding)
    loss = tf.reduce_sum(tf.square(o))

    xs = py_utils.Flatten(thetas + [py_utils.NestedMap(x=inputs.x)])
    dxs = tf.gradients(ys=loss, xs=xs)

    # Reference implementation using Recurrent().
    ref = inputs
    for i in range(layers):
      ref = ElmanOut(
          recurrent.Recurrent(
              cell_fn=cell_fns[i],
              cell_grad=cell_grads[i],
              theta=thetas[i],
              state0=init_states[i],
              inputs=ref)[0])
    return ref.x, output.x, loss, xs, dxs

  def _LogDiff(self, x, y):
    tf.logging.info('max(abs(x - y)) = %s', np.max(np.abs(x - y)))

  def _CompareStackedElman(self, seqlen, batch, dims, layers):
    """Tests that StackedRecurrent computest the same output as Recurrent()."""
    trailing_pad_len = 2
    g = tf.Graph()
    with g.as_default():
      ref, output, _, _, _ = self._BuildStackedRecurrentElman(
          seqlen, trailing_pad_len, batch, dims, layers)
    ref = ref[:-trailing_pad_len]
    output = output[:-trailing_pad_len]
    with self.session(graph=g, use_gpu=True):
      ref_val, out_val = self.evaluate([ref, output])
    self._LogDiff(ref_val, out_val)
    self.assertAllClose(ref_val, out_val)

  def testStackedElman_2(self):
    self._CompareStackedElman(4, 3, 8, 2)

  def testStackedElman_4(self):
    self._CompareStackedElman(8, 5, 8, 4)

  def testStackedElman_8(self):
    self._CompareStackedElman(11, 1, 4, 8)

  def _TestStackedElmanGradient(self, num, seqlen=7, batch=5):
    """Tests a stacked Elman recurrent network with num layers."""
    g = tf.Graph()
    with g.as_default():
      # Sequence length, batdh size, hidden dimension
      trailing_pad_len, dims, layers = 2, 8, num
      _, _, loss, xs, dxs = self._BuildStackedRecurrentElman(
          seqlen, trailing_pad_len, batch, dims, layers)

    # Fetches all gradients (dxs) in one session run and compare
    # them with their respective numerical gradient.
    with self.session(graph=g, use_gpu=True) as sess:
      s_dxs = self.evaluate(dxs)
      for (x, s_dx) in zip(xs, s_dxs):
        n_dx = test_utils.ComputeNumericGradient(sess, loss, x)
        self._LogDiff(n_dx, s_dx)
        self.assertAllClose(n_dx, s_dx)

    # Randomly pick a few (x, dx) pairs, and fetch dx via one self.evaluate
    # and compare with its numerical gradient.
    xs_dxs = list(zip(xs, dxs))
    np.random.shuffle(xs_dxs)
    with self.session(graph=g, use_gpu=True) as sess:
      for (x, dx) in xs_dxs[:4]:
        s_dx = self.evaluate(dx)
        n_dx = test_utils.ComputeNumericGradient(sess, loss, x)
        self._LogDiff(n_dx, s_dx)
        self.assertAllClose(n_dx, s_dx)

  def testStackedElmanGrad_1(self):
    self._TestStackedElmanGradient(1)

  def testStackedElmanGrad_2(self):
    self._TestStackedElmanGradient(2)

  def testStackedElmanGrad_4(self):
    self._TestStackedElmanGradient(4)

  def testStackedElmanGrad_8(self):
    self._TestStackedElmanGradient(8, seqlen=5, batch=3)


if __name__ == '__main__':
  tf.test.main()
