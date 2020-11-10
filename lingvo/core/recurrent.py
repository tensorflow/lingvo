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
"""Recurrent neural nets.

The main interface of this module is Recurrent().
This expects the caller to describe the recurrent neural net by specifying:

  - theta: the "weights" each RNN uses.
  - state0: the initial state of each RNN.
  - cell_fn: A python function describing RNN cell. It must have the following
    signature::

        cell_fn: (theta, state0, inputs) -> (state1, extras)

    state1 is the next RNN state, extras are computed by cell_fn
    and the library forwards extras to cell_fn's gradient function.
  - cell_grad: An optional python function describing the backprop gradient
    function for the RNN cell. It must have the following signature::

        cell_grad: (theta, state0, inputs, extras, dstate1) ->
            (dtheta, dstate0, dinputs)

    dstate1 is what the backprop algorithm provides representing
    gradients of the final loss w.r.t. state1.

All of `theta`, `state0`, `inputs`, `extras` and `dstate1` are
`.NestedMap` so that they can carry a bunch of tensors around.

Recurrent computes, roughly::

    state = state0
    for t in inputs' sequence length:
      state = cell_fn(theta, state, inputs[t, :])
      accumulate_state[t, :] = state
    return accumulate_state, state

The main advantage to using Recurrent instead of tf.while_loop is in
memory savings. In order to compute the gradient for cell_fn, a tf.while_loop
implementation will try to save all of the intermediate tensor values in the
forward pass. For long input sequences this can add up to a very large amount
of memory space.

Recurrent saves only the state output from cell_fn, not any of the intermediate
tensors generated within cell_fn. This saves lots of memory in the forward
pass, but there is a cost: we have to recompute those intermediate tensors
in the backward pass in order to compute the gradient. This recomputation
is why we require that cell_fn be stateless: Recurrent calls cell_fn both
in the forward pass and in the backward pass, and both of those invocations
need to be the same in order for training to work properly.

When using Recurrent, then, we need to store state for the whole training
sequence (in accumulate_state), as well as all of the intermediate tensors
for a single step of cell_fn. Without Recurrent, we would store all of the
intermediate tensors for all of the steps.

We prefer that all of the inputs to cell_fn be passed in using theta, state0,
or inputs. But sometimes you may have code with other inputs; for instance, this
cell_fn references tensor my_tensor, even though it was never passed in as an
input::

    my_tensor = tf.constant(5)
    def cell_fn(inputs):
      return inputs.input * my_tensor

We say that my_tensor was implicitly captured by cell_fn. By default,
Recurrent doesn't allow this, but you can change that behavior by setting the
allow_implicit_captures flag.
"""

import collections
import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core import sendrecv
from lingvo.core import symbolic

DevicePair = collections.namedtuple('DevicePair', ['send', 'recv'])


def _AssertSameTensors(list_a, list_b):
  """Asserts that two lists of tensors are the same tensors."""
  assert len(list_a) == len(list_b), (
      'Expected equal tensor lists but different lengths: %r vs %r' %
      (list_a, list_b))
  for a, b in zip(list_a, list_b):
    assert a is b, (
        'Expected equal tensor lists but at least one differs: %r vs %r' %
        (list_a, list_b))


def _Index(nmap, index):
  """Returns a `.NestedMap` with x[index, :] for each tensor x in nmap.

  Args:
    nmap: A `.NestedMap` of tensors.
    index: A tf scalar integer. Performance is better if 'index' is on the host
      memory.

  Returns:
    A `.NestedMap` of tensors. For each key in nmap::

      rets.key = nmap.key[index, :]
  """
  index = tf.convert_to_tensor(index)
  index.get_shape().assert_has_rank(0)
  return nmap.Transform(lambda x: tf.gather(x, index))


def _Update(nmap_acc, nmap_x, t):
  """Updates t-th row in accumulators.

  Args:
    nmap_acc: A `.NestedMap` of tensors. The accumulators.
    nmap_x: A `.NestedMap` of tensors. The update values.
    t: A scalar integer. Performance is better if 't' is on the device memory.

  Returns:
    A `.NestedMap` of tensors. Say, ret is returned. For each key, we have::

        ret[key] = nmap_acc[key];
        ret[key][t, :] = nmap_x[key]
  """
  acc_lst = nmap_acc.Flatten()
  kx_lst = nmap_x.FlattenItems()
  t = tf.cast([t], tf.int32)  # tf.cast casts on-device tensors.
  lst = []
  for acc, (key, x) in zip(acc_lst, kx_lst):
    with tf.name_scope('update_%s' % py_utils.SanitizeScopeKey(key)):
      lst += [tf.InplaceUpdate(acc, t, tf.expand_dims(x, 0))]
  return nmap_acc.Pack(lst)


def _SeqLenDim(nmap):
  """Returns the 0-th dim size of tensors in nmap.

  This is the max sequence length according to the shape of the inputs.

  Args:
    nmap: A `.NestedMap` of tensors. Every tensor's 0-th dim has the same size.

  Returns:
    A scalar tensor which is the size of 0-th dim of every tensors in nmap.
  """
  keys, values = zip(*nmap.FlattenItems())
  assert values, 'nmap is empty.'
  with tf.control_dependencies([
      py_utils.assert_same_dim0(
          values, msg='recurrent._SeqLen: %s' % list(keys))
  ]):
    return tf.shape(values[0])[0]


def FlattenPadding(padding):
  """Returns padding reduced to have only the time dimension."""
  if padding is None:
    return padding
  r = tf.rank(padding)
  return tf.reduce_min(padding, axis=tf.range(1, r))


def _SeqPaddingLength(inputs_nmap):
  """Returns the lengths of paddings at the beginning and end of the sequence.

  Args:
    inputs_nmap: A `.NestedMap` of tensors that may have 'padding' Every
      tensor's 0-th dim has the same size.

  Returns:
    padding length at the beginning, padding length at the end
  """
  padding = inputs_nmap.get('padding')
  if padding is None:
    return [0, 0]
  time = tf.shape(padding)[0]
  pad_1d = FlattenPadding(padding)
  mask = tf.cast(tf.equal(pad_1d, 0), tf.int32)  # [time], 1s/0s
  mask_reverse = tf.cast(tf.equal(tf.reverse(pad_1d, [0]), 0), tf.int32)
  numbers = tf.range(1, time + 1)
  padding_end = time - tf.reduce_max(mask * numbers)
  padding_begin = tf.where(
      tf.equal(padding_end, time), 0,
      time - tf.reduce_max(mask_reverse * numbers))
  return [padding_begin, padding_end]


def _EmptyAcc(slen, nmap):
  """Creates a set of accumulators for tensors in nmap.

  Args:
    slen: A scalar tensor.
    nmap: A `.NestedMap` of tensors.

  Returns:
    A `.NestedMap` with the same keys as nmap. ret.key, a tensor, has the
    same dtype as nmap.key. The tensor's shape has 1 more dimension
    than the tensor nmap.key. The extra 0-th dimension is of size
    slen. E.g., if slen=10 and nmap.key's shape is [3, 5], then,
    ret.key's shape is [10, 3, 5].
  """

  def Fill(x):
    return tf.Empty(
        tf.concat([[slen], tf.shape(x)], axis=0), x.dtype, init=True)

  return nmap.Transform(Fill)


def _EmptyWithFixShape(shape, nmap):
  """Creates a set of empty initialized tensors with fixed shape.

  Args:
    shape: A list of integers to describe the output tensor shape.
    nmap: A `.NestedMap` of tensors.

  Returns:
    A `.NestedMap` with the same keys as nmap. ret.key, a tensor, has the
    same dtype as nmap.key, but with the fixed shape.
  """

  return nmap.Transform(lambda x: tf.Empty(shape, dtype=x.dtype, init=True))


def _EmptyLike(nmap):
  """Creates a set of empty initialized tensors.

  Args:
    nmap: A `.NestedMap` of tensors.

  Returns:
    A `.NestedMap` of tensors. Each tensor has the same shape and dtype as
    its corresponding tensor in nmap. And each tensor is initialized.
  """
  return nmap.Transform(lambda x: tf.EmptyLike(x, init=True))


def _Add(nmap_x, nmap_y):
  """Adds tensors in nmap_x with respective tensors in nmap_y.

  Args:
    nmap_x: A `.NestedMap` of tensors.
    nmap_y: A `.NestedMap` of tensors.

  Returns:
    A `.NestedMap` of tensors. ret.key = nmap_x.key + nmap_y.key for every key.
  """
  return py_utils.Transform(tf.add, nmap_x, nmap_y)


def _TransformDType(nmap):
  return nmap.Transform(lambda x: tf.cast(x, tf.int64)
                        if x.dtype == tf.int32 else x)


class _Recurrent:
  """A helper class to construct a recurrent neural net."""

  def __init__(self,
               cell_fn,
               cell_grad,
               stop_fn,
               theta,
               state0,
               inputs,
               extras,
               cell_type=None,
               accumulator_layer=None,
               implicit_captures=None,
               unused_acc_state=None):
    """RNN helper class.

    Args:
      cell_fn: A python function which computes:
         state1, extras = cell_fn(theta, state0, inputs[t, :])
      cell_grad: A python function which computes:
         dtheta, dstate0, dinputs[t, :] = cell_grad(
           theta, state0, inputs[t, :], extras, dstate1)
      stop_fn: A python function which computes: should_stop = stop_fn(t, theta,
        state0)
      theta: weights. A `.NestedMap`.
      state0: initial state. A `.NestedMap`.
      inputs: inputs. A `.NestedMap`.
      extras: A `.NestedMap` of Tensors. The 2nd return value of every
        invocation of cell_fn is a `.NestedMap` with matching keys and shapes of
        this 'extras'.
      cell_type: Cell type used in this class.
      accumulator_layer: If provided, then accumulators on this layer will be
        managed such that they carry to the final state in `FProp` and are
        disabled for gradients. Uses the state key `accumulators`.
      implicit_captures: A `.NestedMap` corresponding to implicit captures of
        the cell_fn. If empty/None, implicit captures are either not present or
        disallowed.
      unused_acc_state: If None, we assume every field of acc_state is consumed
        in the following timestamps. If True, None of the acc_state is consumed.
        And we reduce_sum each timestep's new state into a scalar. Note, this
        feature should be used with StackedRecurrent where we send out the new
        state to the other devices.
    """
    self._theta = theta
    self._state = state0
    self._inputs = inputs
    self._cell_fn = _DecorateCellFn(cell_fn, accumulator_layer)
    self._cell_grad = _DecorateCellGrad(cell_grad, accumulator_layer)
    self._stop_fn = stop_fn
    self._extras = extras
    if cell_type is not None:
      self._cell_type = cell_type
    else:
      self._cell_type = 'UnknownType'
    self._accumulator_layer = accumulator_layer
    self._implicit_captures = implicit_captures
    self._unused_acc_state = unused_acc_state

    # NOTE: TF Function (Fwd, Bak, ForwardLoopBody, BackwardLoopBody,
    # Forward and Backward defined below) simply takes a list of
    # Tensors and returns a list of Tensors. When we pass in a
    # structure (a list of NestedMap of Tensors), we use Flatten to
    # convert the structure into a list of tensor. Conversely, the
    # following code often uses Pack to formulate a structure from a
    # list of tensors based on a "template".

    compiled = py_utils.use_xla()
    noinline = not compiled

    # state1, extras = cell_fn(theta, state0, inputs)
    def Fwd(theta, state0, inputs):
      py_utils.SetShapes(theta, self._theta)
      state1, extras = self._cell_fn(theta, state0, inputs)
      py_utils.AssertIsCompatible(state1, self._state)
      py_utils.AssertIsCompatible(extras, self._extras)
      return state1, extras

    # Wraps cell_fn in a TF Function as a for-loop's body.
    #
    # The loop state is composed of:
    #  t: The loop variable on the device. Timestep id.
    #  theta: the recurrent net's weights.
    #  state0: the previous recurrent state.
    #  inputs: inputs to the recurrent net. inputs[t, :] are for the timestep t.
    #  acc_state: Each timestep's computed new state is also stashed into
    #    acc_state.
    #  acc_extras: Each timestep's computed extras is stashed into acc_extras

    def ForwardLoopCond(loop_state):
      """The condition of forward loop."""
      should_continue = loop_state.t < loop_state.limit
      if self._stop_fn:
        should_continue = tf.math.logical_and(
            should_continue,
            tf.reduce_any(
                tf.math.logical_not(
                    self._stop_fn(loop_state.t, loop_state.theta,
                                  loop_state.state0))))
      return should_continue

    def ForwardLoopBody(loop_state):
      """The body of forward loop."""
      t = loop_state.t
      # external input at time step t.
      inputs_t = _Index(loop_state.inputs, t)
      loop_state.state0, extras = Fwd(loop_state.theta, loop_state.state0,
                                      inputs_t)
      # Saves state1 and extras in their accumulators.
      if not self._unused_acc_state:
        loop_state.acc_state = _Update(loop_state.acc_state, loop_state.state0,
                                       t)
      loop_state.acc_extras = _Update(loop_state.acc_extras, extras, t)
      loop_state.t = tf.add(t, 1)
      return loop_state

    # Forward calls ForwardLoopBody n times. Each time computes one
    # time step of the recurrent net.
    def Forward(args):
      """Forward pass of the recurrent net."""
      # The sequence length.
      pad_begin, pad_end = _SeqPaddingLength(args.inputs)
      slen_dim = _SeqLenDim(args.inputs)
      limit = slen_dim - pad_end

      # Creates accumulators for state0 and extras.
      if self._unused_acc_state:
        acc_state = _EmptyWithFixShape([slen_dim], args.state0)
      else:
        acc_state = _EmptyAcc(slen_dim, args.state0)
      acc_extras = _EmptyAcc(slen_dim, args.extras)

      if compiled:
        t = tf.cast(pad_begin, tf.int32)
        limit = tf.cast(limit, tf.int32)
      else:
        t = tf.cast(pad_begin, tf.int64)
        limit = tf.cast(limit, tf.int64)

      with py_utils.RemoveAssertContext(remove=noinline):
        run = py_utils.WhileLoop(
            ForwardLoopCond,
            ForwardLoopBody,
            loop_state=py_utils.NestedMap(
                t=t,
                limit=limit,
                theta=args.theta,
                state0=args.state0,
                inputs=args.inputs,
                acc_state=acc_state,
                acc_extras=acc_extras))
      return py_utils.NestedMap(
          limit=run.t,
          final_state=run.state0,
          acc_state=run.acc_state,
          acc_extras=run.acc_extras)

    # The per-step backward computes:
    #    d_theta, d_state0, d_inputs = cell_grad(
    #        theta, state0, inputs, extras, d_state1)
    # where d_state1 is the backprop-ed gradient for state1, and
    # extras is the computed by the forward step to facilitate the
    # backward step.
    def Bak(theta, state0, inputs, extras, d_state1):
      """Backward step."""
      py_utils.SetShapes(theta, self._theta)
      (dtheta, dstate0, dinputs,
       dcaptures) = self._cell_grad(theta, state0, inputs, extras, d_state1)
      py_utils.AssertIsCompatible(dtheta, self._theta)
      py_utils.AssertIsCompatible(dstate0, self._state)
      py_utils.AssertIsCompatible(dinputs, self._inputs)
      if dcaptures is None:
        # NOTE: Custom gradient fns can return None if they do not support
        # captured tensors. The return value is reserved for the future when
        # that may be supported.
        dcaptures = _EmptyLike(self._implicit_captures)
      py_utils.AssertIsCompatible(dcaptures, self._implicit_captures)

      # Make sure this function didn't capture anything different than the
      # cell_fn when reflected on at the beginning. Must come after the call
      # to cell_grad() which adds to the captured list.
      _AssertSameTensors(py_utils.GetExtraInputs(),
                         self._implicit_captures.Flatten())

      return [dtheta, dstate0, dinputs, dcaptures]

    # Wraps cell_grad gradient function in a TF Function as a
    # for-loop's body for the Backward pass.
    #
    # The loop state is composed of:
    #  t: The loop variable on the device. Timestep id.
    #  state0: the initial state for the entire backward loop.
    #  theta: the recurrent net's weights.
    #  inputs: inputs to the recurrent net. inputs[t, :] are for the timestep t.
    #  acc_state: Each timestep's computed new state was stashed into
    #    acc_state by the Forward pass.
    #  acc_extras: Each timestep's computed extras was stashed into
    #    acc_extras by the Forward pass.
    #  d_theta: All timestep's gradient for theta is accumulated (added) into
    #      d_theta.
    #  d_state1: The backprop-ed gradient for the new stated computed by
    #      timestep t.
    #  d_inputs: d_inputs[t, :] is populated by the backward time step t.
    #  d_acc_state: The backprop-ed gradient for acc_state.
    #  d_captured: All timestep's gradient for theta is accumulated (added)
    #      into d_captured.

    def BackwardLoopCond(loop_state):
      """Backward loop condition function."""
      return loop_state.t >= loop_state.limit

    def BackwardLoopBody(loop_state):
      """Backward loop body function."""
      t = loop_state.t
      # The input recurrent state for time step t is previous time step's
      # output, or the original state0 when on time step 0.
      state_from_acc = _Index(loop_state.acc_state,
                              tf.maximum(tf.constant(0, t.dtype), t - 1))
      state0 = py_utils.If(
          tf.equal(t, tf.constant(0, t.dtype)),
          inputs=py_utils.NestedMap(
              orig_state0=loop_state.state0, state_from_acc=state_from_acc),
          then_branch=lambda nmap: nmap.orig_state0,
          else_branch=lambda nmap: nmap.state_from_acc)

      # The external inputs for time step t.
      inputs_t = _Index(loop_state.inputs, t)
      # The extras for time step t.
      extras_t = _Index(loop_state.acc_extras, t)

      d_state1 = _Add(_Index(loop_state.d_acc_state, t), loop_state.d_state1)
      (d_theta_t, loop_state.d_state1, d_inputs_t,
       d_captured_t) = Bak(loop_state.theta, state0, inputs_t, extras_t,
                           d_state1)

      if self._unused_acc_state:
        # XLA IF op requires the same shape for if and else branches.
        loop_state.d_state1 = loop_state.d_state1.Transform(tf.reduce_sum)
      loop_state.d_theta = _Add(loop_state.d_theta, d_theta_t)
      loop_state.d_inputs = _Update(loop_state.d_inputs, d_inputs_t, t)
      loop_state.d_captured = _Add(loop_state.d_captured, d_captured_t)
      loop_state.t = tf.subtract(t, 1)

      # Make sure this function didn't capture anything different than the
      # cell_fn when reflected on at the beginning. Must come after the call
      # to Bak() which adds to the captured list.
      _AssertSameTensors(py_utils.GetExtraInputs(),
                         self._implicit_captures.Flatten())

      return loop_state

    # Backward calls BackwardLoopBody n times. Each time computes the backprop
    # for one time step of the recurrent net.
    def Backward(xs, ys, dys):
      """Backward pass for the recurrent net.

      Args:
        xs: inputs to the forward operation.
        ys: outputs of the forward operation.
        dys: gradients to the outputs of the forward operation.

      Returns:
        Gradients to the inputs of the forward operation.
      """
      # Accumulators for gradients.
      d_theta = _EmptyLike(xs.theta)
      d_inputs = _EmptyLike(xs.inputs)
      d_captured = _EmptyLike(self._implicit_captures)

      # The sequence length.
      pad_begin, _ = _SeqPaddingLength(xs.inputs)
      limit = pad_begin

      if compiled:
        limit = tf.cast(limit, tf.int32)
      else:
        limit = tf.cast(limit, tf.int64)

      state0 = xs.state0
      d_state1 = dys.final_state
      if self._unused_acc_state:
        # XLA While op requires the same shape for the init and carry on
        # values.
        state0 = state0.Transform(tf.reduce_sum)
        d_state1 = d_state1.Transform(tf.reduce_sum)

      with py_utils.RemoveAssertContext(remove=noinline):
        run = py_utils.WhileLoop(
            cond=BackwardLoopCond,
            body=BackwardLoopBody,
            loop_state=py_utils.NestedMap(
                t=ys.limit - 1,
                limit=limit,
                theta=xs.theta,
                state0=state0,
                inputs=xs.inputs,
                acc_state=ys.acc_state,
                acc_extras=ys.acc_extras,
                d_theta=d_theta,
                d_state1=d_state1,
                d_inputs=d_inputs,
                d_acc_state=dys.acc_state,
                d_captured=d_captured))

      d_state0 = run.d_state1
      if self._unused_acc_state:
        # Match the shape of gradient of the init_state.
        d_state0 = self._state.Transform(tf.zeros_like)

      # The `extra` input in the Forward function is actually an output of the
      # function. It was supplied as an input only to create acc_extras with
      # proper shape, so its gradients should be zero.
      return py_utils.NestedMap(
          d_theta=run.d_theta,
          d_state0=d_state0,
          d_inputs=run.d_inputs,
          d_extras=_EmptyLike(self._extras)), run.d_captured

    # Forward arguments.
    self._fwd_args = py_utils.NestedMap(
        theta=self._theta,
        state0=self._state,
        inputs=self._inputs,
        extras=self._extras)

    # pylint: disable=protected-access
    device_funcs = tf.get_default_graph()._device_functions_outer_to_inner
    self._caller_device = device_funcs[-1] if device_funcs else None
    # pylint: enable=protected-access

    self._forward = Forward
    self._backward = Backward

  def Compute(self):
    """Run the computation."""
    run = py_utils.CallDefun(
        self._forward,
        self._fwd_args,
        bak=self._backward,
        device=self._caller_device)

    if self._accumulator_layer:
      # Restore the accumulators from the final recurrent state.
      self._accumulator_layer.SetAccumulatorValues(run.final_state.accumulators)
      del run.acc_state.accumulators
      del run.final_state.accumulators

    del run.acc_state['_step_seed']
    py_utils.ResetStepSeed(run.final_state.pop('_step_seed'))

    return run.acc_state, run.final_state


def _ReflectOnCellFn(cell_fn,
                     theta,
                     state0,
                     inputs,
                     accumulator_layer=None,
                     check_stateful_ops=False,
                     allow_implicit_capture=False,
                     allowed_tensor_captures=None):
  """Reflects on the cell_fn, applying asserts and returning needed info.

  Args:
    cell_fn: A python function that computes:
      state1, extras = cell_fn(theta, state0, inputs[t, :])
    theta: weights. A `.NestedMap`.
    state0: initial state. A `.NestedMap`.
    inputs: inputs. A `.NestedMap`.
    accumulator_layer: Whether the cell function must be run in the context of
      the given accumulator layer.
    check_stateful_ops: if True, raise a `ValueError` if cell_fn is stateful.
    allow_implicit_capture: Whether to allow the `cell_fn` to implicitly capture
      tensors.
    allowed_tensor_captures: A list of tensors that may be captured. If
      specified, overrides allow_implicit_capture.

  Returns:
    `.NestedMap` of implicit captures that the cell_fn takes.
  Raises:
    ValueError: cell_fn is stateful.
  """
  # Reset the augmented state entries as we may be running in a special
  # disabled context and we want state0 to reflect that.
  state0 = _AugmentState(
      state0.DeepCopy(), accumulator_layer, allow_overwrite=True)

  fwd_sig = [theta, state0, inputs]

  @py_utils.Function(
      # Remove shape information since it may be incompatible with cell_fn.
      fwd_sig=py_utils.TensorSpecs(fwd_sig, keep_shape=False),
      use_tf_function=True)
  def Fwd(args):
    theta, state0, inputs = args
    py_utils.SetShapes(theta, fwd_sig[0])
    state1, extras = cell_fn(theta, state0, inputs)
    return py_utils.Flatten([state1, extras])

  # Asserts about the function.
  if Fwd.stateful_ops:
    if check_stateful_ops:
      raise ValueError('cell_fn contains stateful ops: %s' % Fwd.stateful_ops)
    else:
      tf.logging.warning('cell_fn contains stateful ops: %s', Fwd.stateful_ops)

  if cluster_factory.Current().job in {'trainer', 'trainer_client'}:
    stateful_random_ops = py_utils.StatefulRandomOpsInDefun(Fwd.func)
    if stateful_random_ops:
      raise tf.errors.InvalidArgumentError(
          None, None, 'cell_fn depends on stateful random ops: {}'.format(
              stateful_random_ops))

  ret = py_utils.NestedMap()
  captured_inputs = list(Fwd.captured_inputs)
  if captured_inputs:
    if allowed_tensor_captures:
      allowed_tensor_names = [x.name for x in allowed_tensor_captures]
      for c in captured_inputs:
        if c.name not in allowed_tensor_names:
          raise ValueError('Recurrent cell_fn implicitly captured tensor: %r '
                           'but it is not an allowed captured tensor.' % c)
    elif not allow_implicit_capture:
      raise ValueError('Recurrent cell_fn implicitly captures tensors but '
                       'implicit capture is disabled or a custom cell_grad fn '
                       'is in use. Captured tensors: %r' % captured_inputs)
    ret.captured = captured_inputs
  return ret


def _GetCellGrad(cell_fn,
                 cell_grad,
                 theta,
                 state0,
                 inputs,
                 accumulator_layer,
                 check_stateful_ops=False,
                 allow_implicit_capture=False,
                 allowed_tensor_captures=None):
  """Returns the gradient function for cell_fn.

  Args:
    cell_fn: The recurrent neural net's cell function.
    cell_grad: If not None, cell_fn's gradient function.
    theta: weights. A `.NestedMap`.
    state0: initial state. A `.NestedMap`.
    inputs: inputs. A `.NestedMap`.
    accumulator_layer: Whether the cell function must be run in the context of
      the given accumulator layer.
    check_stateful_ops: if True, raise a `ValueError` if cell_fn is stateful.
    allow_implicit_capture: Whether to allow the `cell_fn` to implicitly capture
      tensors.
    allowed_tensor_captures: A list of tensors that may be captured. If
      specified, overrides allow_implicit_capture.

  Returns:
    Returns (cell_grad, implicit_captures). The passed in cell_grad is returned
    as-is if not None. Otherwise, assume cell_fn is a python function
    representing the recurrent neural net's cell function, i.e.::

      cell_fn: (theta, state0, inputs) -> (state1, extra)

    returns its default gradient python function, i.e.::

      cell_grad: (theta, state0, inputs, extras, captured, dstate1) ->
          (dtheta, dstate0, dinputs)
  """
  implicit_captures = _ReflectOnCellFn(cell_fn, theta, state0, inputs,
                                       accumulator_layer, check_stateful_ops,
                                       allow_implicit_capture,
                                       allowed_tensor_captures)

  if not cell_grad:

    def CellGrad(theta, state0, inputs, unused_extras, dstate1):
      """Default gradient function for cell_fn."""
      # NOTE: The default grad function recomputes the forward
      # function and does not take advantage of 'extras' returned by
      # the forward function.
      state1, _ = cell_fn(theta, state0, inputs)
      assert isinstance(state1, py_utils.NestedMap), ('%s' % state1)

      # Assert that if captured inputs were given, they match the actual
      # tensors passed to the function we are compiled into. Must come after
      # the call to cell_fn, which does the capture.
      _AssertSameTensors(py_utils.GetExtraInputs(), implicit_captures.Flatten())

      # Extract the internal captured tensor placeholders within the Defun
      # we are running in.
      captured = py_utils.Pack(implicit_captures, py_utils.GetExtraArgs())
      ys = py_utils.Flatten(state1)
      xs = py_utils.Flatten([theta, state0, inputs, captured])
      grad_ys = py_utils.Flatten(dstate1)
      grads = tf.gradients(ys=ys, xs=xs, grad_ys=grad_ys)
      grads = py_utils.ConvertNoneGradientToZeros(xs, grads)
      return py_utils.Pack([theta, state0, inputs, captured], grads)

    cell_grad = CellGrad

  return cell_grad, implicit_captures


def _AugmentState(state0, accumulator_layer, allow_overwrite=False):
  """Augments state0 with additional state."""
  if accumulator_layer:
    if 'accumulators' in state0 and not allow_overwrite:
      raise ValueError('accumulators is a private state key used by Recurrent.')
    state0.accumulators = accumulator_layer.GetAccumulatorValues()

  # _step_seed is used for seeding stateless random ops.
  # See py_utils.GenerateStepSeedPair for more details.
  if '_step_seed' in state0 and not allow_overwrite:
    raise ValueError('_step_seed is a private state key used by Recurrent.')
  state0['_step_seed'] = py_utils.GetStepSeed()

  return state0


def _WrapAccumulatorCellFn(accumulator_layer, cell_fn):
  """Wrap a cell_fn to propagate accumulators."""

  def WrappedCellFn(theta, state0, inputs):
    """cell_fn wrapped to propagate accumulators."""
    accumulator_layer.SetAccumulatorValues(state0.accumulators)
    # The underlying cell_fn has no knowledge of accumulator state so
    # delete it.
    state0_accumulators = state0.pop('accumulators')
    state1, extras = cell_fn(theta, state0, inputs)
    state0.accumulators = state0_accumulators
    # Propagate new accumulator state forward.
    state1.accumulators = accumulator_layer.GetAccumulatorValues()
    # Reset: make sure nothing escapes.
    accumulator_layer.accumulators.Transform(lambda x: x.Reset())
    return state1, extras

  return WrappedCellFn


def _WrapAccumulatorCellGradFn(accumulator_layer, cell_grad):
  """Wrap a cell grad function to disable accumulators."""

  def WrappedCellGradFn(theta, state0, inputs, extras, dstate1):
    """cell_grad wrapped to disable accumulators."""
    # Compute the cell grad function with accumulators disabled.
    accumulator_layer.accumulators.Transform(lambda x: x.Disable())
    # The underlying cell_grad has no knowledge of accumulator state so
    # delete it.
    state0_accumulators = state0.pop('accumulators')
    dstate1_accumulators = dstate1.pop('accumulators')
    dtheta, dstate0, dinputs, dcaptures = cell_grad(theta, state0, inputs,
                                                    extras, dstate1)
    state0.accumulators = state0_accumulators
    dstate0.accumulators = dstate1_accumulators
    dstate1.accumulators = dstate1_accumulators
    accumulator_layer.accumulators.Transform(lambda x: x.Enable())
    return dtheta, dstate0, dinputs, dcaptures

  return WrappedCellGradFn


def _WrapCellFnWithStepSeed(cell_fn):
  """Wrap a cell_fn to initialize the step seed."""

  def WrappedCellFn(theta, state0, *args, **kwargs):
    """The wrapper function."""
    # The _step_seed state should be transparent to cell_fn.
    state0_step_seed = state0.pop('_step_seed')
    py_utils.ResetStepSeed(state0_step_seed)
    state1, extras = cell_fn(theta, state0, *args, **kwargs)
    state0['_step_seed'] = state0_step_seed
    state1['_step_seed'] = py_utils.GetStepSeed()
    return state1, extras

  return WrappedCellFn


def _WrapCellGradFnWithStepSeed(cell_grad):
  """Wrap a cell grad function to handle step seed in state."""

  def WrappedCellGradFn(theta, state0, inputs, extras, dstate1):
    """The wrapper function."""
    # The _step_seed state should be transparent to cell_grad.
    state0_step_seed = state0.pop('_step_seed')
    dstep_seed = dstate1.pop('_step_seed')
    py_utils.ResetStepSeed(state0_step_seed)
    dtheta, dstate0, dinputs, dcaptures = cell_grad(theta, state0, inputs,
                                                    extras, dstate1)
    state0['_step_seed'] = state0_step_seed
    dstate0['_step_seed'] = dstep_seed
    dstate1['_step_seed'] = dstep_seed
    return dtheta, dstate0, dinputs, dcaptures

  return WrappedCellGradFn


def _WrapCellFnWithSymbolValues(cell_fn, symbol_to_tensor_map):
  """Wrap a cell_fn to propagate symbol values."""

  def WrappedCellFn(theta, state0, inputs):
    """cell_fn wrapped to propagate accumulators."""
    theta = theta.copy()
    symbols = list(symbol_to_tensor_map.keys())
    symbol_values = theta.pop('_symbol_values')
    inner_symbol_to_tensor_map = dict(zip(symbols, symbol_values))
    if symbols:
      tf.logging.info('_WrapCellFnWithSymbolValues: %s', symbols)
    with symbolic.SymbolToValueMap(symbolic.TENSOR_VALUES,
                                   inner_symbol_to_tensor_map):
      state1, extras = cell_fn(theta, state0, inputs)
    return state1, extras

  return WrappedCellFn


def _WrapCellGradFnWithSymbolValues(cell_grad, cell_fn, symbol_to_tensor_map):
  """Wrap a cell grad function to propagate symbol values."""

  def WrappedCellGradFn(theta, state0, inputs, extras, dstate1):
    """The wrapper function."""
    symbols = list(symbol_to_tensor_map.keys())
    symbol_values = theta['_symbol_values']
    inner_symbol_to_tensor_map = dict(zip(symbols, symbol_values))
    if symbols:
      tf.logging.info('_WrapCellGradFnWithSymbolValues: %s', symbols)
    with symbolic.SymbolToValueMap(symbolic.TENSOR_VALUES,
                                   inner_symbol_to_tensor_map):
      dtheta, dstate0, dinputs, dcaptures = cell_grad(theta, state0, inputs,
                                                      extras, dstate1)
      # cell_grad may have populated dtheta by applying tf.gradients() on
      # theta.Flatten().
      if '_symbol_values' not in dtheta:
        state1, _ = cell_fn(theta, state0, inputs)
        dxs = tf.gradients(
            ys=state1.Flatten(), xs=symbol_values, grad_ys=dstate1.Flatten())
        dtheta['_symbol_values'] = py_utils.ConvertNoneGradientToZeros(
            symbol_values, dxs)
    return dtheta, dstate0, dinputs, dcaptures

  return WrappedCellGradFn


def _DecorateCellFn(cell_fn, accumulator_layer):
  """Decorates cell_fn with additional state information."""
  if accumulator_layer:
    # Wrap the cell_fn so that it knows how to propagate accumulators.
    cell_fn = _WrapAccumulatorCellFn(accumulator_layer, cell_fn)
  cell_fn = _WrapCellFnWithStepSeed(cell_fn)
  return cell_fn


def _DecorateCellGrad(cell_grad, accumulator_layer):
  """Decorates cell_grad with additional state information."""
  if accumulator_layer:
    # Wrap the cell_grad so it disables accumulators.
    cell_grad = _WrapAccumulatorCellGradFn(accumulator_layer, cell_grad)
  cell_grad = _WrapCellGradFnWithStepSeed(cell_grad)
  return cell_grad


def _IsSingleTimeStep(inputs):
  """Returns True only if the time dimension of inputs is 1."""
  for x in inputs.Flatten():
    if x.shape.dims is None or x.shape.as_list()[0] != 1:
      return False
  return True


def _RecurrentSingleTimeStep(theta, state0, inputs, cell_fn):
  """Short-cut for the single timestep without explicit cell_grad case."""
  # The seqlen length is staticly known as 1. Hence, we just need to
  # call cell_fn once without putting it into a loop.
  # Since we are not looping, there is no need to specially manage
  # accumulators.
  inputs = inputs.Transform(lambda x: tf.squeeze(x, axis=0))
  state1, _ = cell_fn(theta, state0, inputs)
  acc_state = state1.Transform(lambda x: tf.expand_dims(x, axis=0))
  return acc_state, state1


def Recurrent(theta,
              state0,
              inputs,
              cell_fn,
              cell_grad=None,
              cell_type=None,
              stop_fn=None,
              extras=None,
              check_stateful_ops=False,
              accumulator_layer=None,
              allow_implicit_capture=False,
              allowed_tensor_captures=None):
  """Compute a recurrent neural net.

  Roughly, `Recurrent()` computes the following::

      state = state0
      for t in inputs' sequence length:
        state = cell_fn(theta, state, inputs[t, :])
        accumulate_state[t, :] = state
      return accumulate_state, state

  `theta`, `state`, `inputs` are all `.NestedMap` objects.

  `inputs[t, :]` means taking a slice out from every tensor in the
  `.NestedMap` `inputs`.

  `accumulate_state[t, :] = state` means that we stash every tensor in
  `state` into a slice of the corresponding tensor in
  `accumulate_state`.

  `cell_fn` is a python callable computing (building up a TensorFlow
  graph) the recurrent neural network's one forward step. `cell_fn` must not
  contain any stateful ops. Two calls of `cell_fn` must describe two identical
  computations.

  By construction, `Recurrent()`'s backward computation does not access
  any intermediate values computed by `cell_fn` during forward
  computation. We may extend `Recurrent()` to support that by taking a
  customized backward function of `cell_fn`.

  Args:
    theta: weights. A `.NestedMap`.
    state0: initial state. A `.NestedMap`.
    inputs: inputs. A `.NestedMap`.
    cell_fn: A python function which computes::
        state1, extras = cell_fn(theta, state0, inputs[t, :])
    cell_grad: A python function which computes::
        dtheta, dstate0, dinputs[t, :], dcaptured = cell_grad(
            theta, state0, inputs[t, :], extras, dstate1)  If there are no
              captured tensors in `cell_fn`, `dcaptured` can be returned as
              None. Captured tensors with custom `cell_grad` is currently
              unsupported so this return value is reserved for future expansion.
    cell_type: Cell name to be used.
    stop_fn: If not None, a python function which computes::  should_stop =
      stop_fn(t, theta, state0)  The function determines whether the recurrent
      loop should terminate.
    extras: A `.NestedMap` of Tensors. The 2nd return value of every invocation
      of `cell_fn` is a `.NestedMap` with matching keys and shapes of `extras`.
    check_stateful_ops: if True, raise a `ValueError` if `cell_fn` is stateful.
    accumulator_layer: If provided, then accumulators on this layer will be
      managed such that they carry to the final state in `FProp` and are
      disabled for gradients. Uses the state key `accumulators`.
    allow_implicit_capture: Whether to allow the `cell_fn` to implicitly capture
      tensors. Only allowed if an explicit `cell_grad` is not given.
    allowed_tensor_captures: A list of tensors that may be captured. If
      specified, overrides allow_implicit_capture.

  Returns:
    `accumulate_state` and the final state.
  """
  symbol_to_tensor_map = symbolic.SymbolToValueMap.Get(symbolic.TENSOR_VALUES)
  if symbol_to_tensor_map:
    theta = theta.copy()  # Do not modify the caller's 'theta'.
    theta['_symbol_values'] = list(
        tf.convert_to_tensor(v) for v in symbol_to_tensor_map.values())
    cell_fn = _WrapCellFnWithSymbolValues(cell_fn, symbol_to_tensor_map)
    if cell_grad:
      cell_grad = _WrapCellGradFnWithSymbolValues(cell_grad, cell_fn,
                                                  symbol_to_tensor_map)

  inputs = _TransformDType(inputs)
  if cell_grad is not None:
    allow_implicit_capture = False
    allowed_tensor_captures = None

  # Short-cut for the single timestep with default grad function case.
  if cell_grad is None and _IsSingleTimeStep(inputs):
    return _RecurrentSingleTimeStep(theta, state0, inputs, cell_fn)

  # Disable accumulators since cell_fn needs to be called a few times and those
  # aren't real calls to cell_fn. They will be re-enabled just prior to
  # calling _Recurrent.
  if accumulator_layer:
    accumulator_layer.accumulators.Transform(lambda x: x.Disable())

  cell_grad, implicit_captures = _GetCellGrad(cell_fn, cell_grad, theta, state0,
                                              inputs, accumulator_layer,
                                              check_stateful_ops,
                                              allow_implicit_capture,
                                              allowed_tensor_captures)

  with tf.name_scope('recurrent_cellfn_extras'):
    # Derives 'extras' so that we can allocate extras' accumulator.
    # Not a real call to cell_fn, so make sure it doesn't affect step_seed.
    step_seed = py_utils.GetStepSeed()
    # Make sure not to modify the original state0.
    _, actual_extras = cell_fn(theta, state0.DeepCopy(), _Index(inputs, 0))
    py_utils.ResetStepSeed(step_seed)
  if extras is None:
    extras = actual_extras.Transform(tf.zeros_like)
  else:
    if not extras:
      # Forces the extras to be an empty map if an empty 'extras' is provided.
      extras = py_utils.NestedMap()
    py_utils.AssertIsCompatible(extras, actual_extras)

  # Enable accumulators. Note that this must happen prior to the initial
  # _AugmentState() below or it will initialize with defaults.
  if accumulator_layer:
    accumulator_layer.accumulators.Transform(lambda x: x.Enable())

  acc_state, final_state = _Recurrent(
      cell_fn=cell_fn,
      cell_grad=cell_grad,
      cell_type=cell_type,
      stop_fn=stop_fn,
      theta=theta,
      state0=_AugmentState(state0.DeepCopy(), accumulator_layer),
      inputs=inputs,
      extras=extras,
      accumulator_layer=accumulator_layer,
      implicit_captures=implicit_captures).Compute()

  # TODO(b/129159299): The ResetStepSeed below is needed to work around this
  # bug, which is a problem with global tensors being shared by different
  # inference graphs. It should be removed once the bug is fixed.
  py_utils.MaybeResetStepSeedFromScope()

  return acc_state, final_state


class _Link:
  """A link is a pair of channels."""

  def __init__(self, t, dpair):
    # Uses a unique name scope to name the channel.
    with tf.name_scope('fwd') as scope:
      self.fwd = sendrecv.Channel(t.dtype, t.shape, dpair.send, dpair.recv,
                                  scope)
    with tf.name_scope('bak') as scope:
      self.bak = sendrecv.Channel(t.dtype, t.shape, dpair.recv, dpair.send,
                                  scope)


def _CreateLinks(nmap, dpair):
  """Creates links between the send/recv devices for every tensor in nmap."""
  return nmap.Transform(lambda t: _Link(t, dpair))


def _Join(nmap_x, nmap_y, fn):
  return py_utils.Transform(fn, nmap_x, nmap_y).Flatten()


class _Input:
  """Input layers."""

  def __init__(self,
               cell_fn,
               cell_out,
               cell_grad,
               cell_out_grad,
               theta,
               state0,
               accumulator_layer,
               inputs,
               extras,
               out_links,
               unused_acc_state=False):
    self._cell_fn = cell_fn
    self._cell_out = cell_out
    self._cell_grad, self._implicit_captures = _GetCellGrad(
        cell_fn,
        cell_grad,
        theta,
        state0,
        inputs,
        accumulator_layer,
        allow_implicit_capture=True)
    self._cell_out_grad = cell_out_grad
    self._theta = theta
    self._state0 = state0
    self._accumulator_layer = accumulator_layer
    self._inputs = inputs
    self._extras = extras
    self._out_links = out_links
    self._unused_acc_state = unused_acc_state
    assert self._extras is not None

  def Compute(self):
    """Compute the input layer."""

    def InputFn(theta, state0, inputs):
      state1, extras = self._cell_fn(theta, state0, inputs)
      py_utils.AssertIsCompatible(state1, state0)
      py_utils.AssertIsCompatible(extras, self._extras)
      out = self._cell_out(state1)
      sends = _Join(self._out_links, out, lambda l, x: l.fwd.Send(x))
      with tf.control_dependencies(sends):
        return state1.Transform(tf.identity), extras.Transform(tf.identity)

    def InputGrad(theta, state0, inputs, extras, dstate1):
      """Gradient function for InputFn."""
      recv_dout = self._out_links.Transform(lambda l: l.bak.Recv())
      dstate1 = _Add(dstate1, self._cell_out_grad(recv_dout))
      dtheta, dstate0, dinputs, dcaptures = self._cell_grad(
          theta, state0, inputs, extras, dstate1)  # pylint: disable=unbalanced-tuple-unpacking
      py_utils.AssertIsCompatible(dtheta, self._theta)
      py_utils.AssertIsCompatible(dstate0, state0)
      py_utils.AssertIsCompatible(dinputs, self._inputs)
      if dcaptures is None:
        # NOTE: Custom gradient fns can return None if they do not support
        # captured tensors. The return value is reserved for the future when
        # that may be supported.
        dcaptures = _EmptyLike(self._implicit_captures)
      py_utils.AssertIsCompatible(dcaptures, self._implicit_captures)
      return dtheta, dstate0, dinputs, dcaptures

    return _Recurrent(
        cell_fn=InputFn,
        cell_grad=InputGrad,
        stop_fn=None,
        theta=self._theta,
        state0=self._state0,
        inputs=self._inputs,
        extras=self._extras,
        accumulator_layer=self._accumulator_layer,
        implicit_captures=self._implicit_captures,
        unused_acc_state=self._unused_acc_state).Compute()


class _Middle:
  """Middle layers."""

  def __init__(self, cell_fn, cell_out, cell_grad, cell_out_grad, theta, state0,
               accumulator_layer, in_links, padding, slen_dim, per_step_inputs,
               extras, out_links, unused_acc_state):
    self._cell_fn = cell_fn
    self._cell_out = cell_out
    self._cell_grad, self._implicit_captures = _GetCellGrad(
        cell_fn,
        cell_grad,
        theta,
        state0,
        per_step_inputs,
        accumulator_layer,
        allow_implicit_capture=True)
    self._cell_out_grad = cell_out_grad
    self._theta = theta
    self._state0 = state0
    self._accumulator_layer = accumulator_layer
    self._in_links = in_links
    self._padding = padding
    self._slen_dim = slen_dim
    self._per_step_inputs = per_step_inputs
    self._extras = extras
    assert self._extras is not None
    self._out_links = out_links
    self._unused_acc_state = unused_acc_state

  def Compute(self):
    """Compute the middle layer."""

    def MiddleFn(theta, state0, inputs):
      del inputs
      inputs = self._in_links.Transform(lambda l: l.fwd.Recv())
      state1, extras = self._cell_fn(theta, state0, inputs)
      py_utils.AssertIsCompatible(state1, state0)
      py_utils.AssertIsCompatible(extras, self._extras)
      out = self._cell_out(state1)
      sends = _Join(self._out_links, out, lambda l, x: l.fwd.Send(x))
      with tf.control_dependencies(sends):
        return (state1.Transform(tf.identity),
                py_utils.NestedMap(inputs=inputs,
                                   cell_fn_extras=extras).Transform(
                                       tf.identity))

    def MiddleGrad(theta, state0, inputs, extras, dstate1):
      """Gradient function for MiddleFn."""
      recv_dout = self._out_links.Transform(lambda l: l.bak.Recv())
      dstate1 = _Add(dstate1, self._cell_out_grad(recv_dout))
      dtheta, dstate0, dinputs, dcaptures = self._cell_grad(
          theta, state0, extras.inputs, extras.cell_fn_extras, dstate1)  # pylint: disable=unbalanced-tuple-unpacking
      py_utils.AssertIsCompatible(dtheta, self._theta)
      py_utils.AssertIsCompatible(dstate0, state0)
      py_utils.AssertIsCompatible(dinputs, self._per_step_inputs)
      if dcaptures is None:
        # NOTE: Custom gradient fns can return None if they do not support
        # captured tensors. The return value is reserved for the future when
        # that may be supported.
        dcaptures = _EmptyLike(self._implicit_captures)
      py_utils.AssertIsCompatible(dcaptures, self._implicit_captures)
      sends = _Join(self._in_links, dinputs, lambda l, x: l.bak.Send(x))
      with tf.control_dependencies(sends):
        return (dtheta.Transform(tf.identity), dstate0.Transform(tf.identity),
                inputs.Transform(tf.zeros_like),
                dcaptures.Transform(tf.identity))

    fake_inputs = py_utils.NestedMap(
        fake_input=tf.zeros([self._slen_dim], tf.float32))
    if self._padding is not None:
      fake_inputs['padding'] = self._padding

    return _Recurrent(
        cell_fn=MiddleFn,
        cell_grad=MiddleGrad,
        stop_fn=None,
        theta=self._theta,
        state0=self._state0,
        inputs=fake_inputs,
        extras=py_utils.NestedMap(
            inputs=self._per_step_inputs, cell_fn_extras=self._extras),
        accumulator_layer=self._accumulator_layer,
        implicit_captures=self._implicit_captures,
        unused_acc_state=self._unused_acc_state).Compute()


class _Output:
  """Output layers."""

  def __init__(self, cell_fn, cell_grad, theta, state0, accumulator_layer,
               in_links, padding, slen_dim, per_step_inputs, extras):
    self._cell_fn = cell_fn
    self._cell_grad, self._implicit_captures = _GetCellGrad(
        cell_fn,
        cell_grad,
        theta,
        state0,
        per_step_inputs,
        accumulator_layer,
        allow_implicit_capture=True)
    self._theta = theta
    self._state0 = state0
    self._accumulator_layer = accumulator_layer
    self._in_links = in_links
    self._padding = padding
    self._slen_dim = slen_dim
    self._per_step_inputs = per_step_inputs
    self._extras = extras
    assert self._extras is not None

  def Compute(self):
    """Compute the output layer."""

    def OutputFn(theta, state0, inputs):
      del inputs
      inputs = self._in_links.Transform(lambda l: l.fwd.Recv())
      state1, extras = self._cell_fn(theta, state0, inputs)
      py_utils.AssertIsCompatible(state1, state0)
      py_utils.AssertIsCompatible(extras, self._extras)
      return state1, py_utils.NestedMap(inputs=inputs, cell_fn_extras=extras)

    def OutputGrad(theta, state0, inputs, extras, dstate1):
      """Gradient function for OutputFn."""
      dtheta, dstate0, dinputs, dcaptures = self._cell_grad(
          theta, state0, extras.inputs, extras.cell_fn_extras, dstate1)  # pylint: disable=unbalanced-tuple-unpacking
      py_utils.AssertIsCompatible(dtheta, self._theta)
      py_utils.AssertIsCompatible(dstate0, state0)
      py_utils.AssertIsCompatible(dinputs, self._per_step_inputs)
      if dcaptures is None:
        # NOTE: Custom gradient fns can return None if they do not support
        # captured tensors. The return value is reserved for the future when
        # that may be supported.
        dcaptures = _EmptyLike(self._implicit_captures)
      py_utils.AssertIsCompatible(dcaptures, self._implicit_captures)
      sends = _Join(self._in_links, dinputs, lambda l, x: l.bak.Send(x))
      with tf.control_dependencies(sends):
        return (dtheta.Transform(tf.identity), dstate0.Transform(tf.identity),
                inputs.Transform(tf.zeros_like),
                dcaptures.Transform(tf.identity))

    fake_inputs = py_utils.NestedMap(
        fake_input=tf.zeros([self._slen_dim], tf.float32))
    if self._padding is not None:
      fake_inputs['padding'] = self._padding

    return _Recurrent(
        cell_fn=OutputFn,
        cell_grad=OutputGrad,
        stop_fn=None,
        theta=self._theta,
        state0=self._state0,
        inputs=fake_inputs,
        extras=py_utils.NestedMap(
            inputs=self._per_step_inputs, cell_fn_extras=self._extras),
        accumulator_layer=self._accumulator_layer,
        implicit_captures=self._implicit_captures,
        unused_acc_state=False).Compute()


def _DependsOn(xs, ys):
  """Every x in xs should depend on every y in ys via a data edge."""

  # TODO(zhifengc): Using the following ops is likely more robust because
  # algebra simplifier may remove s - s, t + 0, etc.
  #   nil: list -> 0
  #   first: x, list -> x
  #
  # If we have nil & first, we can write
  #   zero = nil(py_utils.Flatten(ys))
  #   return [x.Transform(lambda t: first(t, zero)) for x in xs]
  def MakeZero(x):
    s = tf.reduce_sum(x)
    return tf.cast(s - s, tf.float32)

  def SumToZero(nmap_list):
    return tf.add_n([MakeZero(x) for x in py_utils.Flatten(nmap_list)])

  ys_zero = SumToZero(ys)
  return [x.Transform(lambda t: t + tf.cast(ys_zero, t.dtype)) for x in xs]


def StackedRecurrent(devices,
                     cell_fns,
                     cell_grads,
                     cell_outs,
                     cell_out_grads,
                     thetas,
                     init_states,
                     inputs,
                     accumulator_layers=None,
                     unused_acc_state=False):
  """Computes stacked recurrent neural nets placed on various devices.

  Conceptually, StackedRecurrent() computes the following::

    for (device, cell_fn, cell_out, cell_grad, theta, state0) in zip(
      (devices, cell_fns, cell_outs, cell_grads, thetas, init_states):
        with tf.device(device):
          state1, _ = Recurrent(theta, state0, inputs, cell_fn, cell_grad)
          outputs = cell_out(state1)
          inputs = outputs  # Next layer's input is this layer's output
    return outputs

  The only difference is that StackedRecurrent implements a model parallelism
  so that all layers computation can happen concurrently.

  Args:
    devices: A list of N tensorflow device names.
    cell_fns: If a list of N recurrent cell function, cell_fns[i] must meet the
      same requirement as Recurrent() requires its cell_fn argument.  Otherwise,
      applies to all layers.
    cell_grads: If a list of N recurrent cell gradient function, cell_grads[i]
      must meet the same requirement as Recurrent() requires its cell_grad
      argument.  Otherwise, applies to all layers.
    cell_outs: If a list of N function, cell_outs[i] takes the state computed by
      cell_fns[i] and returns the input for the next layer. These functions are
      expected to be simple and just do renaming of fields.  Otherwise, applies
      to all layers.
    cell_out_grads: If a list of N function, cell_out_grads[i] is often the
      reverse of cell_outs[i]. Otherwise, applies to all layers.
    thetas: A list of N weights NestedMap. thetas[i] must meet the same
      requirement as Recurrent() requires its theta argument.
    init_states: A list of N initial state NestedMap. init_states[i] must meet
      the same requirement as Recurrent() requires its state0 argument.
    inputs: Inputs to the 1st layer of the stacked recurrent neural nets.  A
      NestedMap.
    accumulator_layers: A list of layers whose accumulators will be managed such
      that they carry to the output state in `FProp` and are disabled for
      gradients. Uses the state key `accumulators`.  Default to None where no
      accumulator values will be carried.
    unused_acc_state: If True, we shink all the layer's acc_state to [num_ts]
      except the last layer(_Output).

  Returns:
    Tuple (output, states):

      - The last layer's output (accumulated states).
      - The list of final state NestedMap. One for each layer.
  """
  # Enable rendezvous sharing when using tf.function, since it needs to do
  # send/recv across function boundary.
  # pylint: disable=protected-access
  with py_utils._SharedRendezvousScope(shared_rendezvous=True):
    return _StackedRecurrent(devices, cell_fns, cell_grads, cell_outs,
                             cell_out_grads, thetas, init_states, inputs,
                             accumulator_layers, unused_acc_state)
  # pylint: enable=protected-access


def _StackedRecurrent(devices, cell_fns, cell_grads, cell_outs, cell_out_grads,
                      thetas, init_states, inputs, accumulator_layers,
                      unused_acc_state):
  """Implementation of StackedRecurrent, see StackedRecurrent for details."""
  num_layers = len(devices)
  assert num_layers

  def _MakeList(fns):
    if not isinstance(fns, (list, tuple)):
      return [fns] * num_layers
    else:
      assert num_layers == len(fns)
      return fns

  cell_fns = _MakeList(cell_fns)
  cell_grads = _MakeList(cell_grads)
  cell_outs = _MakeList(cell_outs)
  cell_out_grads = _MakeList(cell_out_grads)
  accumulator_layers = accumulator_layers or [None] * num_layers
  assert num_layers == len(thetas)
  assert all(isinstance(x, py_utils.NestedMap) for x in thetas)
  assert num_layers == len(init_states)
  assert all(isinstance(x, py_utils.NestedMap) for x in init_states)
  assert isinstance(inputs, py_utils.NestedMap)

  if py_utils.use_tpu():
    # If this error happens, the number of splits must be increased (e.g.
    # worker_split_size in trainer/tpu.sh), or the number of rnn layers
    # decreased.
    # TODO(cwhipkey): lift this restriction by grouping layers by device and
    # having a device handle a contiguous run of layers, and have them loop
    # over the layers in the cell fns.
    assert len(devices) == len(set(devices)), (
        'StackedRecurrent must provide a different device for each layer '
        'when run on TPU. devices passed were: %s' % str(devices))

  if num_layers == 1:
    # Simple case, just use Recurrent() directly.
    with tf.device(devices[0]):
      acc_states, final = Recurrent(
          theta=thetas[0],
          state0=init_states[0],
          inputs=inputs,
          cell_fn=cell_fns[0],
          cell_grad=cell_grads[0],
          accumulator_layer=accumulator_layers[0])
      # Just the accumulated states.
      return cell_outs[0](acc_states), final

  # We add explicit data dependencies between layer-i's theta/state0
  # and layer-(i-1)'s theta/state0, layer-0's theta/state0 has an
  # explicit data dependency on inputs.  These extra data dependencies
  # ensure that if layer-i's theta/state0 is used in tf.gradient, all
  # layers above's backprop are triggered.
  prev = [inputs]
  for i in range(num_layers):
    with tf.device(devices[i]):
      thetas[i], init_states[i] = _DependsOn([thetas[i], init_states[i]], prev)
    prev = [thetas[i], init_states[i]]

  def ExpectedOutputOfLayers():
    """Estimate what tensor dtypes and shapes output by each layer."""

    def ZerosLikeRequireShape(t):
      assert t.shape.is_fully_defined()
      return tf.zeros_like(t)

    if py_utils.use_tpu():
      transform_fn = ZerosLikeRequireShape
    else:
      transform_fn = tf.zeros_like

    expected_output_by_layers = []
    xs = _Index(inputs, 0)
    for i in range(num_layers):
      # Disable accumulators and step_seed since this is not a real call to
      # cell_fns[i]. They will be re-enabled in _Recurrent.<F24><F25>
      if accumulator_layers[i]:
        accumulator_layers[i].accumulators.Transform(lambda x: x.Disable())
      step_seed = py_utils.GetStepSeed()
      state1, extras = cell_fns[i](thetas[i], init_states[i], xs)
      py_utils.ResetStepSeed(step_seed)
      # only dtype and shape is needed.
      xs = cell_outs[i](state1)
      expected_output_by_layers += [
          py_utils.NestedMap(
              xs=xs.Transform(transform_fn),
              extras=extras.Transform(transform_fn))
      ]
    return expected_output_by_layers

  expected_output_by_layers = ExpectedOutputOfLayers()

  # Sequence length. We assume it's a grid we are building.
  slen_dim = _SeqLenDim(inputs)

  assert num_layers >= 2
  layers = []

  padding = FlattenPadding(inputs.get('padding', None))

  # Builds the input layer.
  out_links = _CreateLinks(expected_output_by_layers[0].xs,
                           DevicePair(devices[0], devices[1]))

  # Enable accumulators. Note that this must happen prior to the initial
  # _AugmentState() below or it will initialize with defaults.
  for accumulator_layer in accumulator_layers:
    if accumulator_layer:
      accumulator_layer.accumulators.Transform(lambda x: x.Enable())

  inp_l = _Input(
      cell_fn=cell_fns[0],
      cell_grad=cell_grads[0],
      cell_out=cell_outs[0],
      cell_out_grad=cell_out_grads[0],
      theta=thetas[0],
      state0=_AugmentState(init_states[0].DeepCopy(), accumulator_layers[0]),
      accumulator_layer=accumulator_layers[0],
      inputs=inputs,
      extras=expected_output_by_layers[0].extras,
      out_links=out_links,
      unused_acc_state=unused_acc_state)
  layers += [inp_l]

  # Builds the intermediate layers.
  for i in range(1, num_layers - 1):
    in_links = out_links
    out_links = _CreateLinks(expected_output_by_layers[i].xs,
                             DevicePair(devices[i], devices[i + 1]))
    mid_l = _Middle(
        cell_fn=cell_fns[i],
        cell_grad=cell_grads[i],
        cell_out=cell_outs[i],
        cell_out_grad=cell_out_grads[i],
        theta=thetas[i],
        state0=_AugmentState(init_states[i].DeepCopy(), accumulator_layers[i]),
        accumulator_layer=accumulator_layers[i],
        in_links=in_links,
        padding=padding,
        slen_dim=slen_dim,
        per_step_inputs=expected_output_by_layers[i - 1].xs,
        extras=expected_output_by_layers[i].extras,
        out_links=out_links,
        unused_acc_state=unused_acc_state)
    layers += [mid_l]

  # Builds the final output layer.
  in_links = out_links
  del out_links
  out_l = _Output(
      cell_fn=cell_fns[-1],
      cell_grad=cell_grads[-1],
      theta=thetas[-1],
      state0=_AugmentState(init_states[-1].DeepCopy(), accumulator_layers[-1]),
      accumulator_layer=accumulator_layers[-1],
      in_links=in_links,
      padding=padding,
      slen_dim=slen_dim,
      per_step_inputs=expected_output_by_layers[-2].xs,
      extras=expected_output_by_layers[-1].extras)
  layers += [out_l]

  assert len(layers) == num_layers

  anchor = 0
  final_states = []
  for (dev, layer) in zip(devices, layers):
    # Computes each layer on their designated device.
    with tf.device(dev):
      acc_states, final = layer.Compute()  # Don't care of final state yet.
      final_states.append(final)

      # We add every number output by the layer (s) and computes a
      # zero scalar: (s - s), as an anchor. Anchors are added
      # sequentially and added to the final layer's output. This way,
      # we ensure that the final output depends on every previous
      # layer through data dependencies. This is a hack to ensure that
      # tf.gradient will follow some data dependencies path to start
      # the Backward loop for each layer.
      #
      # TODO(zhifengc): We can write, if we have nil & first ops:
      #   anchor += [nil(py_utils.Flatten(acc_states))]
      # And finally,
      #   return acc_states.Transform(lambda x: first(x, anchor))
      def ComputeAnchor(x):
        # For each
        s = tf.add_n([tf.reduce_sum(_) for _ in x.Flatten()])
        return s - s

      anchor = ComputeAnchor(acc_states) + anchor

  # The last layer's output is the real output that matters.  However,
  # to make the previous layers backprop work, we need to make sure
  # the returned value has data dependencies on the previous layers.
  # 'anchor' is guaranteed to be a scalar 0 and hence adding it to the
  # final output does not change its numerical value.
  with tf.device(devices[-1]):
    outputs = cell_outs[-1](acc_states.Transform(lambda x: x + anchor))

  # TODO(b/129159299): The ResetStepSeed below is needed to work around this
  # bug, which is a problem with global tensors being shared by different
  # inference graphs. It should be removed once the bug is fixed.
  py_utils.MaybeResetStepSeedFromScope()

  return outputs, final_states
