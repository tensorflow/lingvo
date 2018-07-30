# -*- coding: utf-8 -*-
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.python.framework import function
from lingvo.core import py_utils
from lingvo.core import recurrent
from lingvo.core import test_utils


def _ApplyPadding(padding, v_no_pad, v_pad):
  if padding is not None:
    padding = tf.cast(padding, v_no_pad.dtype)
    return (1 - padding) * v_no_pad + padding * v_pad
  return v_no_pad


class RecurrentTest(tf.test.TestCase):

  def testBasic(self):

    def Poly(theta, state, inputs):
      next_state = py_utils.NestedMap()
      next_state.value = state.value + inputs.coeff * state.x_power
      next_state.x_power = state.x_power * theta.x
      return next_state, py_utils.NestedMap()

    with self.test_session() as sess:

      theta = py_utils.NestedMap()
      theta.x = tf.constant(2.0)
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      state.x_power = tf.constant(1.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])

      # x = 2
      # 1 + 2*x + 3*x^2
      ret = recurrent.Recurrent(theta, state, inputs, Poly)

      acc, state = sess.run(ret)
      self.assertAllClose(acc.value, [1., 5., 17.])
      self.assertAllClose(acc.x_power, [2., 4., 8.])
      self.assertAllClose(state.value, 17.)
      self.assertAllClose(state.x_power, 8.)

      y = ret[1].value
      dx, d_coeff = tf.gradients(ys=[y], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = sess.run([dx, d_coeff])

      # 2 + 6*x
      self.assertAllClose(dx_val, 14.)
      self.assertAllClose(d_coeff_val, [1., 2., 4.])

      # acc = [1, 1+2x, 1+2x+3x^2]
      # sum(acc) = 3 + 4x + 3x^2
      acc = ret[0].value
      dx, d_coeff = tf.gradients(
          ys=[tf.reduce_sum(acc)], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = sess.run([dx, d_coeff])
      # 4 + 6*x
      self.assertAllClose(dx_val, 16.)
      self.assertAllClose(d_coeff_val, [3., 4., 4.])

  def testStatefulCellFn(self):

    def Rand(theta, state, inputs):
      del theta
      next_state = py_utils.NestedMap()
      next_state.value = (
          state.value +
          inputs.coeff * tf.random_uniform(shape=[], dtype=state.value.dtype))
      return next_state, py_utils.NestedMap()

    with self.test_session():
      theta = py_utils.NestedMap()
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])

      with self.assertRaisesRegexp(ValueError, 'stateful.*random_uniform'):
        recurrent.Recurrent(theta, state, inputs, Rand, check_stateful_ops=True)

  def testNestedCellFn(self):
    """Tests when cell_fn calls another function."""

    @function.Defun(tf.float32)
    def RandWithCoeff(coeff):
      return coeff * tf.random_uniform(shape=[], dtype=coeff.dtype)

    def Rand(theta, state, inputs):
      del theta
      next_state = py_utils.NestedMap()
      next_state.value = state.value + RandWithCoeff(inputs.coeff)
      return next_state, py_utils.NestedMap()

    @function.Defun(tf.float32)
    def Coeff(coeff):
      return coeff * 2

    def Deterministic(theta, state, inputs):
      del theta
      next_state = py_utils.NestedMap()
      next_state.value = state.value + Coeff(inputs.coeff)
      return next_state, py_utils.NestedMap()

    with self.test_session():
      theta = py_utils.NestedMap()
      state = py_utils.NestedMap()
      state.value = tf.constant(0.0)
      inputs = py_utils.NestedMap()
      inputs.coeff = tf.constant([1., 2., 3.])

      recurrent.Recurrent(
          theta, state, inputs, Deterministic, check_stateful_ops=True)
      with self.assertRaisesRegexp(ValueError, 'stateful.*RandWithCoeff'):
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
      with self.test_session() as sess:
        inputs = py_utils.NestedMap()
        inputs.padding = tf.constant(value)

        slen_op = recurrent._SeqPaddingLength(inputs)
        slen, = sess.run([slen_op])
        self.assertEqual(expected, slen)

  @staticmethod
  def Rand(shape):
    return tf.random_uniform(shape, minval=-0.2, maxval=0.2, dtype=tf.float64)

  @staticmethod
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

  @staticmethod
  def ElmanGrad(theta, state0, inputs, extras, dstate1):

    @function.Defun()
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
    return (py_utils.NestedMap(w=dw, b=db), dstate0, dinputs)

  @staticmethod
  def ElmanOut(state1):
    return py_utils.NestedMap(x=state1.h, padding=state1.padding)

  @staticmethod
  def ElmanOutGrad(dout):
    return py_utils.NestedMap(h=dout.x, padding=dout.padding)

  def _testElmanHelper(self, seqlen, use_grad):
    with self.test_session() as sess:
      tf.set_random_seed(342462)

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
        s, _ = self.Elman(theta, s, inp)
        out += [s.h]
      acc0, final0 = tf.stack(out), s.h
      loss0 = tf.reduce_sum(acc0) + tf.reduce_sum(final0)
      (dw0, db0, dh0,
       di0) = tf.gradients(loss0, [theta.w, theta.b, state0.h, inputs.x])

      # Uses the Recurrent() library.
      acc1, final1 = recurrent.Recurrent(
          theta=theta,
          state0=state0,
          inputs=inputs,
          cell_fn=self.Elman,
          cell_grad=self.ElmanGrad if use_grad else None)
      acc1, final1 = acc1.h, final1.h
      loss1 = tf.reduce_sum(acc1) + tf.reduce_sum(final1)
      (dw1, db1, dh1,
       di1) = tf.gradients(loss1, [theta.w, theta.b, state0.h, inputs.x])

      # Fetches a bunch of values and compare them.
      (acc0, acc1, final0, final1, dw0, dw1, db0, db1, dh0, dh1, di0,
       di1) = sess.run(
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


if __name__ == '__main__':
  tf.test.main()
