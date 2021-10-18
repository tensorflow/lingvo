# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lingvo Jax recurrent layer."""

from absl import logging
from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax import test_util
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax.layers import recurrent
from lingvo.jax.layers import stochastics
import numpy as np
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap


class RecurrentTest(test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  def test_recurrent01(self):
    theta = NestedMap(proj=np.random.uniform(size=[3, 4]))
    inputs = NestedMap(x=np.random.uniform(size=[5, 3]))
    state0 = NestedMap(y=np.zeros([4]))

    prng_key = jnp.array([21230, 90230], dtype=jnp.uint32)
    global_step = jnp.array(0, dtype=jnp.uint64)

    def cell_fn(theta, state0, inputs_t):
      del state0
      y = jnp.einsum('x,xy->y', inputs_t.x, theta.proj)
      return NestedMap(y=y)

    def comp01(theta, state0, inputs):
      with base_layer.JaxContext.NewContext(
          prng_key=prng_key, global_step=global_step):
        final_state, cum_states = recurrent.recurrent_func(
            theta, state0, inputs, cell_fn)
        loss = jnp.sum(final_state.y) + jnp.sum(cum_states.y)
        return loss

    def comp02(theta, state0, inputs):
      with base_layer.JaxContext.NewContext(
          prng_key=prng_key, global_step=global_step):
        final_state, cum_states = recurrent.recurrent_static(
            theta, state0, inputs, cell_fn)
        loss = jnp.sum(final_state.y) + jnp.sum(cum_states.y)
        return loss

    logging.info('comp01_jaxpr: %s',
                 jax.make_jaxpr(comp01)(theta, state0, inputs))
    logging.info('comp02_jaxpr: %s',
                 jax.make_jaxpr(comp02)(theta, state0, inputs))
    loss1 = comp01(theta, state0, inputs)
    loss2 = comp02(theta, state0, inputs)

    def to_np(x):
      return np.asarray(x, dtype=np.float32)

    self.assertAllClose(to_np(loss1), to_np(loss2))

  def test_recurrent02(self):
    theta = NestedMap(delta=np.ones([3, 4]))
    inputs = NestedMap(x=np.ones([5, 3, 4]))
    state0 = NestedMap(y=np.ones([3, 4]))

    prng_key = jnp.array([21230, 90230], dtype=jnp.uint32)
    global_step = jnp.array(0, dtype=jnp.uint64)

    def cell_fn(theta, state0, inputs_t):
      y = theta.delta + inputs_t.x + state0.y
      return NestedMap(y=y)

    def comp01(theta, state0, inputs):
      final_state, cum_states = recurrent.recurrent_static(
          theta, state0, inputs, cell_fn)
      return final_state, cum_states

    def comp01_loss(theta, state0, inputs):
      final_state, cum_states = comp01(theta, state0, inputs)
      return jnp.sum(final_state.y) + jnp.sum(cum_states.y)

    def comp02(theta, state0, inputs):
      final_state, cum_states = recurrent.recurrent_func(
          theta, state0, inputs, cell_fn)
      return final_state, cum_states

    def comp02_loss(theta, state0, inputs):
      final_state, cum_states = comp02(theta, state0, inputs)
      return jnp.sum(final_state.y) + jnp.sum(cum_states.y)

    def to_np(x):
      return np.asarray(x, dtype=np.float32)

    grad_fn_01 = jax.grad(comp01_loss, [0, 1, 2])
    grad_fn_02 = jax.grad(comp02_loss, [0, 1, 2])

    def same_value(x, y):
      self.assertAllClose(to_np(x), to_np(y))

    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=global_step):
      final_states_01, cum_states_01 = comp01(theta, state0, inputs)
      logging.info('final_states_01: %s', final_states_01)
      logging.info('cum_states_01: %s', cum_states_01)
      expected_final_state = np.zeros([3, 4]) + 11.0
      self.assertAllClose(to_np(expected_final_state), to_np(final_states_01.y))
      final_states_02, cum_states_02 = comp02(theta, state0, inputs)
      tf.nest.map_structure(same_value, final_states_01, final_states_02)
      tf.nest.map_structure(same_value, cum_states_01, cum_states_02)
      grad01 = grad_fn_01(theta, state0, inputs)
      grad02 = grad_fn_02(theta, state0, inputs)
      expected_grad = np.zeros([3, 4]) + 20.0
      logging.info('grad01: %s', grad01)
      logging.info('grad02: %s', grad02)
      self.assertAllClose(expected_grad, to_np(grad01[0].delta))
      tf.nest.map_structure(same_value, grad01, grad02)

  def test_recurrent03(self):
    dropout_l = stochastics.DropoutLayer.Params().Set(
        name='dropout01', keep_prob=0.5)
    layer = dropout_l.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    dropout_layer_vars = layer.InstantiateVariables(prng_key)
    logging.info('dropout_layer_vars: %s', dropout_layer_vars)

    inputs = NestedMap(x=np.ones([5, 3, 4]))
    state0 = NestedMap(y=np.ones([3, 4]))
    prng_key = jax.random.PRNGKey(seed=123456)
    global_step = jnp.array(0, dtype=jnp.uint64)

    theta = NestedMap(delta=np.ones([3, 4]), dropout=dropout_layer_vars)

    def cell_fn(theta, state0, inputs_t):
      increment = theta.delta + inputs_t.x
      increment = layer.FProp(theta.dropout, increment)
      y = state0.y + increment
      return NestedMap(y=y)

    def comp(theta, state0, inputs):
      final_state, cum_states = recurrent.recurrent_static(
          theta, state0, inputs, cell_fn)
      return final_state, cum_states

    def comp_loss(theta, state0, inputs):
      final_state, cum_states = comp(theta, state0, inputs)
      return jnp.sum(final_state.y) + jnp.sum(cum_states.y)

    grad_fn = jax.grad(comp_loss)

    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=global_step):
      final_state, cum_states = comp(theta, state0, inputs)
      logging.info('final_state: %s', final_state)
      logging.info('cum_states: %s', cum_states)
      grad = grad_fn(theta, state0, inputs)
      logging.info('grad jaxpr: %s',
                   jax.make_jaxpr(grad_fn)(theta, state0, inputs))
      logging.info('grad: %s', grad)

  def test_recurrent04(self):
    theta = NestedMap(delta=np.ones([2]))
    inputs = NestedMap(
        x=np.ones([8, 2]),
        padding=np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0],
                          [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]))
    state0 = NestedMap(y=np.ones([2]))

    prng_key = jax.random.PRNGKey(seed=1234)
    global_step = jnp.array(0, dtype=jnp.uint64)

    def cell_fn(theta, state0, inputs_t):
      y = theta.delta + inputs_t.x + state0.y
      return NestedMap(y=y)

    def comp(theta, state0, inputs):
      final_state, cum_states = recurrent.recurrent_static(
          theta, state0, inputs, cell_fn)
      return final_state, cum_states

    def comp_loss(theta, state0, inputs):
      final_state, cum_states = comp(theta, state0, inputs)
      cum_states.y *= (1.0 - inputs.padding)
      return jnp.sum(final_state.y) + jnp.sum(cum_states.y)

    def to_np(x):
      return np.asarray(x, dtype=np.float32)

    grad_fn = jax.grad(comp_loss)

    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=global_step):
      final_state, cum_states = comp(theta, state0, inputs)
      logging.info('final_state: %s', final_state)
      logging.info('cum_states: %s', cum_states)
      expected_final_state = np.zeros([2]) + 11.0
      self.assertAllClose(to_np(expected_final_state), to_np(final_state.y))
      expected_cum_states = np.array(
          [[3., 3.], [3., 3.], [5., 5.], [7., 7.], [9., 9.], [11., 11.],
           [11., 11.], [11., 11.]],
          dtype=np.float32)
      self.assertAllClose(expected_cum_states, to_np(cum_states.y))

      grad = grad_fn(theta, state0, inputs)
      logging.info('grad jaxpr: %s',
                   jax.make_jaxpr(grad_fn)(theta, state0, inputs))
      expected_grad = np.zeros([2]) + 20.0
      self.assertAllClose(expected_grad, to_np(grad.delta))

  def test_scan_01(self):

    theta = NestedMap(delta=np.ones([3, 4]))
    xs = NestedMap(x=np.ones([5, 3, 4]))
    carry_initial = NestedMap(y=np.ones([3, 4]))

    def comp01(theta, carry, xs):

      def cell_fn(carry_0, xs_t):
        # theta is implicitly captured.
        y = theta.delta + xs_t.x + carry_0.y
        z = y + 1
        carry_1 = NestedMap(y=y)
        return carry_1, NestedMap(z=z)

      carry_final, ys = recurrent.scan(carry, xs, cell_fn)

      loss = jnp.sum(carry_final.y) + jnp.sum(ys.z)

      return loss, (carry_final, ys)

    grad_fn_01 = jax.value_and_grad(comp01, [0, 1, 2], has_aux=True)

    def to_np(x):
      return np.asarray(x, dtype=np.float32)

    prng_key = jax.random.PRNGKey(21230)
    global_step = jnp.array(0, dtype=jnp.uint32)

    with base_layer.JaxContext.NewContext(
        prng_key=prng_key, global_step=global_step):
      loss, (carry_final, ys) = comp01(theta, carry_initial, xs)
      logging.info('loss: %s', loss)
      logging.info('carry_final: %s', carry_final)
      logging.info('ys: %s', ys)
      expected_carry_final = np.zeros([3, 4]) + 11.0
      self.assertAllClose(to_np(expected_carry_final), to_np(carry_final.y))
      self.assertAllClose(612.0, loss)
      _, grad01 = grad_fn_01(theta, carry_initial, xs)
      logging.info('grad01: %s', grad01)
      grad_wrt_delta = grad01[0].delta
      expected_grad_wrt_delta = np.zeros([3, 4]) + 20.0
      self.assertAllClose(to_np(expected_grad_wrt_delta), to_np(grad_wrt_delta))


if __name__ == '__main__':
  absltest.main()
