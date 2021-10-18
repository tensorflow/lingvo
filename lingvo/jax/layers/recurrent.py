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
"""Jax recurrent layer implementation.

The main interface of this module is recurrent_func().
This expects the caller to describe the recurrent neural net by specifying:

  - theta: the "weights" each RNN uses.
  - states_0: the initial state of each RNN.
  - cell_fn: A python function describing RNN cell. It must have the following
    signature::

        cell_fn: (theta, states_0, inputs) -> states_1

    states_1 is the next RNN state.

recurrent_func computes, roughly::

    state = states_0
    t = 0
    while t < seq_length:
      state = cell_fn(theta, state, inputs[t, :])
      accumulate_state[t, :] = state
      t += 1
    return accumulate_state, state
"""

import enum
import functools
from typing import Callable, Optional, Tuple

import jax
from jax import ad_checkpoint
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
WeightParams = py_utils.WeightParams

ParamsT = pytypes.ParamsT
JTensor = pytypes.JTensor
CallableOrNone = Optional[Callable]
NestedMapOrNone = Optional[NestedMap]


@enum.unique
class AutodiffCheckpointType(str, enum.Enum):
  """jax.checkpoint policy types."""
  SAVE_EVERYTHING = 'save_everything'
  SAVE_NOTHING = 'save_nothing'
  SAVE_DOT_ONLY = 'save_dot_only'
  SAVE_DOT_WITH_NO_BATCH_DIM = 'save_dot_with_no_batch_dims'
  SAVE_DOT_FOR_MLPERF_200B = 'save_dot_for_mlperf_200b'


def recurrent_func(theta: NestedMap, states_0: NestedMap, inputs: NestedMap,
                   cell_fn: Callable[[NestedMap, NestedMap, NestedMap],
                                     NestedMap]):
  """Computes a recurrent neural net.

  Args:
    theta: weights. A `.NestedMap`.
    states_0: initial state. A `.NestedMap`.
    inputs: inputs. A `.NestedMap`.
    cell_fn: A python function which computes::
        states_1 = cell_fn(theta, states_0, inputs[t, :])

  Returns:
    `accumulate_state` and the final state.
  """
  input_seq_len = inputs.Flatten()[0].shape[0]

  def assert_not_none(x):
    assert x is not None

  tf.nest.map_structure(assert_not_none, states_0)
  tf.nest.map_structure(assert_not_none, inputs)
  tf.nest.map_structure(assert_not_none, theta)

  def new_cum_state(x):
    x1 = jnp.expand_dims(x, 0)
    # +1 so that we can store initial_states at position 0.
    return jnp.tile(x1, [input_seq_len + 1] + [1] * x.ndim)

  cumulative_states = states_0.Transform(new_cum_state)

  prng_key = base_layer.NextPrngKey()
  global_step = base_layer.CurGlobalStep()

  start_time = jnp.array(0, dtype=jnp.uint32)
  fwd_initial_loop_vars = NestedMap(
      cur_time=start_time,
      theta=theta,
      states_0=states_0,
      cumulative_states=cumulative_states,
      inputs=inputs)

  def same_type_shape(x, y):
    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    assert x.shape == y.shape, (x.shape, y.shape)

  def wrapped_cell_fn(fn_in):
    # fn_in is NestedMap containing the following elements:
    #    - t
    #    - theta
    #    - states_0
    #    - inputs_t
    # Start a chain of prng key that also takes into account of time steps.
    t = fn_in.t
    theta = fn_in.theta
    states_0 = fn_in.states_0
    inputs_t = fn_in.inputs_t
    with base_layer.JaxContext.NewContext(
        prng_key=jax.random.fold_in(prng_key, t), global_step=global_step):
      states_1 = cell_fn(theta, states_0, inputs_t)

      tf.nest.assert_same_structure(states_0, states_1)
      tf.nest.map_structure(same_type_shape, states_0, states_1)
    return states_1

  def wrapped_cell_fn_grad(fn_in, d_fn_out):
    # This is roughly the following:
    #
    # fn_out = wrapped_cell_fn(fn_in)
    # d_fn_in = tf.gradient(fn_out, fn_in, d_fn_out)
    # return d_fn_in
    #
    assert isinstance(fn_in, NestedMap)
    fn_out, vjp_fn = jax.vjp(wrapped_cell_fn, fn_in)
    del fn_out
    d_fn_in = vjp_fn(d_fn_out)
    assert isinstance(d_fn_in, tuple)
    assert len(d_fn_in) == 1
    d_fn_in_0 = d_fn_in[0]
    # Over-write gradient for t, the time step.
    d_fn_in_0.t = jnp.zeros_like(fn_in.t)
    tf.nest.assert_same_structure(fn_in, d_fn_in_0)
    tf.nest.map_structure(same_type_shape, fn_in, d_fn_in_0)
    return d_fn_in_0

  def fwd_comp_fn(loop_vars):
    # loop_vars is a NestedMap containing the following elements:
    #   - cur_time
    #   - theta
    #   - inputs
    #   - cumulative_states
    #   - states_0
    t = loop_vars.cur_time
    theta = loop_vars.theta
    inputs = loop_vars.inputs
    cumulative_states = loop_vars.cumulative_states
    states_0 = loop_vars.states_0
    inputs_t = inputs.Transform(lambda x: x[t])

    states_1 = wrapped_cell_fn(
        NestedMap(t=t, theta=theta, states_0=states_0, inputs_t=inputs_t))

    def set_t(x, x_t):
      return x.at[t + 1].set(x_t)

    cumulative_states = tf.nest.map_structure(set_t, cumulative_states,
                                              states_1)
    loop_out = NestedMap(
        cur_time=t + 1,
        theta=theta,
        inputs=inputs,
        states_0=states_1,
        cumulative_states=cumulative_states)
    return loop_out

  def fwd_continue_fn(loop_vars):
    return loop_vars.cur_time < input_seq_len

  # This custom_vjp implementation follows examples here:
  # https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  @jax.custom_vjp
  def fwd_loop(loop_vars):
    final_loop_vars = jax.lax.while_loop(fwd_continue_fn, fwd_comp_fn,
                                         loop_vars)
    return NestedMap(
        final_states=final_loop_vars.states_0,
        cumulative_states=final_loop_vars.cumulative_states)

  def loop_fn_vjp_fwd(loop_vars):
    loop_fn_out = fwd_loop(loop_vars)
    return loop_fn_out, (loop_vars, loop_fn_out.cumulative_states)

  def loop_fn_vjp_bwd(res, d_out):
    fwd_loop_vars, cumulative_states = res
    d_final_states = d_out.final_states
    d_cumulative_states = d_out.cumulative_states

    start_time = input_seq_len - 1
    d_states_1 = tf.nest.map_structure(lambda x, y: x[start_time + 1] + y,
                                       d_cumulative_states, d_final_states)
    bwd_loop_vars = NestedMap(
        cur_time=start_time,
        theta=fwd_loop_vars.theta,
        inputs=fwd_loop_vars.inputs,
        cumulative_states=cumulative_states,
        d_cumulative_states=d_cumulative_states,
        d_theta=fwd_loop_vars.theta.Transform(jnp.zeros_like),
        d_inputs=fwd_loop_vars.inputs.Transform(jnp.zeros_like),
        d_states_1=d_states_1)

    def bwd_comp_fn(loop_vars):
      t = loop_vars.cur_time
      inputs = loop_vars.inputs
      inputs_t = inputs.Transform(lambda x: x[t])
      states_0 = loop_vars.cumulative_states.Transform(lambda x: x[t])
      d_cell_in = wrapped_cell_fn_grad(
          NestedMap(
              t=t, theta=loop_vars.theta, states_0=states_0, inputs_t=inputs_t),
          loop_vars.d_states_1)
      d_theta = tf.nest.map_structure(lambda x, y: x + y, loop_vars.d_theta,
                                      d_cell_in.theta)
      d_states_0 = tf.nest.map_structure(lambda x, y: x + y[t],
                                         d_cell_in.states_0,
                                         loop_vars.d_cumulative_states)

      def set_t(x, x_t):
        return x.at[t].set(x_t)

      d_inputs = tf.nest.map_structure(set_t, loop_vars.d_inputs,
                                       d_cell_in.inputs_t)
      loop_vars_out = loop_vars.Transform(lambda x: x)
      loop_vars_out.d_inputs = d_inputs
      loop_vars_out.d_states_1 = d_states_0
      loop_vars_out.d_theta = d_theta
      loop_vars_out.cur_time = t - 1
      return loop_vars_out

    def bwd_continue_fn(loop_vars):
      return loop_vars.cur_time >= 0

    bwd_final_loop_vars = jax.lax.while_loop(bwd_continue_fn, bwd_comp_fn,
                                             bwd_loop_vars)
    d_out = fwd_loop_vars.Transform(jnp.zeros_like)

    tf.nest.map_structure(same_type_shape, d_out.states_0,
                          bwd_final_loop_vars.d_states_1)
    tf.nest.map_structure(same_type_shape, d_out.theta,
                          bwd_final_loop_vars.d_theta)
    tf.nest.map_structure(same_type_shape, d_out.inputs,
                          bwd_final_loop_vars.d_inputs)

    d_out.states_0 = bwd_final_loop_vars.d_states_1
    d_out.theta = bwd_final_loop_vars.d_theta
    d_out.inputs = bwd_final_loop_vars.d_inputs
    return (d_out,)

  fwd_loop.defvjp(loop_fn_vjp_fwd, loop_fn_vjp_bwd)

  # Finally, let's simply run the forward loop fn.
  fwd_final_loop_vars = fwd_loop(fwd_initial_loop_vars)
  fwd_cumulative_states = fwd_final_loop_vars.cumulative_states.Transform(
      lambda x: x[1:])
  return fwd_final_loop_vars.final_states, fwd_cumulative_states


def recurrent_static(theta: NestedMap,
                     states_0: NestedMap,
                     inputs: NestedMap,
                     cell_fn: Callable[[NestedMap, NestedMap, NestedMap],
                                       NestedMap],
                     root_layer: Optional[base_layer.BaseLayer] = None):
  """A simpler form of Recurrent where num of steps is known statically.

  Back-prop is availale through auto-diff.

  'padding' in inputs is used to skip certain steps dynamically. If the
  'padding' tensor exists, it is expected of a binary 0/1 tensor.

  Args:
    theta: weights. A `.NestedMap`.
    states_0: initial state. A `.NestedMap`.
    inputs: inputs. A `.NestedMap`. All inputs in time-major.
    cell_fn: A python function which computes::
        states_1 = cell_fn(theta, states_0, inputs[t, :])
    root_layer: The root layer within which this recurrent_static recurrent loop
      is carried out.

  Returns:
    `accumulate_state` and the final state.
  """

  assert 'time_step' not in states_0
  # The initial time step.
  time_step = jnp.array(0, dtype=jnp.uint32)
  # Make a copy of states_0 structure.
  states_0 = tf.nest.map_structure(lambda x: x, states_0)
  states_0.time_step = time_step

  prng_key = base_layer.NextPrngKey()
  global_step = base_layer.CurGlobalStep()

  # TODO(zhangqiaorjc): Switch to ad_checkpoint.checkpoint after mattjj bug fix.
  @jax.checkpoint
  def comp_fn(states_0, inputs_t):
    # Start a new prng_key branch that also depends on the time step.
    if root_layer is not None:
      forward_updated_vars_before = tf.nest.map_structure(
          lambda x: x, root_layer.forward_updated_vars)
    prng_key_t = jax.random.fold_in(prng_key, states_0.time_step)
    with base_layer.JaxContext.NewContext(
        prng_key=prng_key_t, global_step=global_step):
      # Whether or not we should skip this time step.
      if 'padding' in inputs_t:
        # We skip if all are padded steps.
        skip = jnp.all(inputs_t.padding > 0.5)
      else:
        skip = jnp.array(False)

      def carry_over(args):
        states_0, inputs_t = args
        del inputs_t
        # We simply carry over the states for this time step.
        states_1 = tf.nest.map_structure(lambda x: x, states_0)
        states_1.time_step = states_0.time_step + 1
        return states_1

      def do_compute(args):
        states_0, inputs_t = args
        # Actually carry out the computation.
        states_1 = cell_fn(theta, states_0, inputs_t)
        states_1.time_step = states_0.time_step + 1
        return states_1

      if 'padding' in inputs_t:
        states_1 = jax.lax.cond(skip, carry_over, do_compute,
                                (states_0, inputs_t))
      else:
        states_1 = do_compute((states_0, inputs_t))
      tf.nest.assert_same_structure(states_0, states_1)

      if root_layer is not None:
        forward_updated_vars_after = tf.nest.map_structure(
            lambda x: x, root_layer.forward_updated_vars)

        def assert_no_change(x, y):
          assert (x is None and y is None) or (x is not None and y is not None)

        tf.nest.map_structure(assert_no_change, forward_updated_vars_before,
                              forward_updated_vars_after)

      return states_1, states_1

  final_states, cumulative_states = jax.lax.scan(comp_fn, states_0, inputs)
  del final_states.time_step
  del cumulative_states.time_step
  return final_states, cumulative_states


def scan(carry_init: NestedMap,
         xs: NestedMap,
         fn: Callable[[NestedMap, NestedMap], Tuple[NestedMap, NestedMap]],
         root_layer: Optional[base_layer.BaseLayer] = None,
         checkpoint_policy: AutodiffCheckpointType = AutodiffCheckpointType
         .SAVE_NOTHING):
  """A simple wrap around jax.lax.scan.

  Back-prop is availale through auto-diff.

  Args:
    carry_init: initial state. A `.NestedMap`.
    xs: inputs. A `.NestedMap`. All inputs in time-major.
    fn: A python function which computes:
        carry, ys[t] = fn(carry, xs[t, :])
    root_layer: The root layer within which this jax.lax.scan based while_loop
      is carried out. If root_layer is provided, some basic-effort check is
      performed to make sure fn is side-effect free. Otherwise, no such checks
      are performed.
    checkpoint_policy: A AutodiffCheckpointType. How to checkpoint for BProp:
      SAVE_NOTHING, SAVE_DOT_ONLY, SAVE_DOT_WITH_NO_BATCH_DIM.

  Returns:
    final 'carry' as well as 'ys'.
  """
  assert isinstance(carry_init, py_utils.NestedMap)
  assert isinstance(xs, py_utils.NestedMap)
  # Make a copy of carry_init structure.
  carry_init = tf.nest.map_structure(lambda x: x, carry_init)
  # "carry" will be augmented with the following three tensors, so make sure
  # they don't already exist in the NestedMap.
  assert 'time_step' not in carry_init
  assert 'prng_key' not in carry_init
  assert 'global_step' not in carry_init

  def custom_policy(checkpoint_policy: AutodiffCheckpointType):

    if checkpoint_policy == AutodiffCheckpointType.SAVE_EVERYTHING:
      return jax.checkpoint_policies.everything_saveable
    if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_ONLY:
      return jax.checkpoint_policies.checkpoint_dots
    if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM:
      return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    # TODO(zhangqiaorjc): Configure custom checkpoint policy in expt config
    # without introducing enum.
    if checkpoint_policy == AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B:
      return jax.checkpoint_policies.save_only_these_names(
          'combined_qkv_proj', 'query_proj', 'value_proj', 'key_proj',
          'context', 'out_proj')
    assert checkpoint_policy == AutodiffCheckpointType.SAVE_NOTHING
    return jax.checkpoint_policies.nothing_saveable

  @functools.partial(
      ad_checkpoint.checkpoint,
      prevent_cse=False,
      policy=custom_policy(checkpoint_policy))
  def fn_wrap(carry, xs_t):
    # carry is augmented with time_step, prng_key, global_step three additional
    # tensors to make fn_wrap fully functional.
    if root_layer is not None:
      forward_updated_vars_before = tf.nest.map_structure(
          lambda x: x, root_layer.forward_updated_vars)
    # Start a new prng_key branch that also depends on the time step.
    prng_key_t = jax.random.fold_in(carry.prng_key, carry.time_step)
    with base_layer.JaxContext.NewContext(
        prng_key=prng_key_t, global_step=carry.global_step):

      carry_new, ys_t = fn(carry, xs_t)
      carry_new.time_step = carry.time_step + 1
      # copy over prng_key and global_step
      carry_new.prng_key = carry.prng_key
      carry_new.global_step = carry.global_step

      tf.nest.assert_same_structure(carry_new, carry)

      if root_layer is not None:
        forward_updated_vars_after = tf.nest.map_structure(
            lambda x: x, root_layer.forward_updated_vars)

        def assert_no_change(x, y):
          assert (x is None and y is None) or (x is not None and y is not None)

        # Make sure fn doesn't have side-effect, in particular it doesn't
        # update any forward-vars.
        tf.nest.map_structure(assert_no_change, forward_updated_vars_before,
                              forward_updated_vars_after)

      return carry_new, ys_t

  # The initial time step.
  time_step = jnp.array(0, dtype=jnp.uint32)
  prng_key = base_layer.NextPrngKey()
  global_step = base_layer.CurGlobalStep()
  carry_init.time_step = time_step
  carry_init.prng_key = prng_key
  carry_init.global_step = global_step

  carry_final, ys = jax.lax.scan(fn_wrap, carry_init, xs)

  del carry_final.time_step
  del carry_final.global_step
  del carry_final.prng_key
  return carry_final, ys
