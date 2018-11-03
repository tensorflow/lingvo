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
  - cell_fn: A python function describing RNN cell. It must has the following
    signature::

        cell_fn: (theta, state0, inputs) -> (state1, extras)

    state1 is the next RNN state, extras are computed by cell_fn
    and the library forwards extras to cell_fn's gradient function.
  - cell_grad: A python function describing the backprop gradient function
    for the RNN cell. It must has the following signature::

        cell_grad: (theta, state0, inputs, extras, dstate1) ->
            (dtheta, dstate0, dinputs)

    dstate1 is what the backprop algorithm provides representing
    gradients of state1 w.r.t. the final loss.

All of `theta`, `state0`, `inputs`, `extras` and `dstate1` are
`.NestedMap` so that they can carry a bunch of tensors around.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.python.framework import function
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import inplace_ops
from lingvo.core import py_utils


def _AssertIsCompatible(a, b):
  assert a.IsCompatible(b), ('%s vs %s' % (a, b))


def _AssertSameTensors(list_a, list_b):
  """Asserts that two lists of tensors are the same tensors."""
  assert len(list_a) == len(list_b), (
      'Expected equal tensor lists but different lengths: %r vs %r' % (list_a,
                                                                       list_b))
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
    t: A scalar integer. Performance is better if 't' is on the device
      memory.

  Returns:
    A `.NestedMap` of tensors. Say, ret is returned. For each key, we have::

        ret[key] = nmap_acc[key];
        ret[key][t, :] = nmap_x[key]
  """
  acc_lst = nmap_acc.Flatten()
  x_lst = nmap_x.Flatten()
  t = tf.to_int32([t])  # tf.to_int32 casts on-device tensors.
  lst = []
  for acc, x in zip(acc_lst, x_lst):
    lst += [inplace_ops.alias_inplace_update(acc, t, tf.expand_dims(x, 0))]
  return nmap_acc.Pack(lst)


def _SeqLenDim(nmap):
  """Returns the 0-th dim size of tensors in nmap.

  This is the max sequence length according to the shape of the inputs.

  Args:
    nmap: A `.NestedMap` of tensors. Every tensor's 0-th dim has the same size.

  Returns:
    A scalar tensor which is the size of 0-th dim of every tensors in nmap.
  """
  xs = nmap.Flatten()
  assert xs, 'nmap is empty.'
  with tf.control_dependencies(
      [py_utils.assert_same_dim0(xs, msg='recurrent._SeqLen')]):
    return tf.shape(xs[0])[0]


def _FlattenPadding(padding):
  """Returns padding reduced to have only the time dimension."""
  if padding is None:
    return padding
  r = tf.rank(padding)
  return tf.reduce_min(padding, axis=tf.range(1, r))


def _SeqPaddingLength(inputs_nmap):
  """Returns the lengths of paddings at the beginning and end of the sequence.

  Args:
    inputs_nmap: A `.NestedMap` of tensors that may have 'padding'
                 Every tensor's 0-th dim has the same size.

  Returns:
    padding length at the beginning, padding length at the end
  """
  padding = inputs_nmap.get('padding')
  if padding is None:
    return [0, 0]
  time = tf.shape(padding)[0]
  pad_1d = _FlattenPadding(padding)
  mask = tf.to_int32(tf.equal(pad_1d, 0))  # [time], 1s/0s
  mask_reverse = tf.to_int32(tf.equal(tf.reverse(pad_1d, [0]), 0))
  numbers = tf.range(1, time + 1)
  padding_end = time - tf.reduce_max(mask * numbers)
  padding_begin = tf.where(
      tf.equal(padding_end, time), 0,
      time - tf.reduce_max(mask_reverse * numbers))
  return [padding_begin, padding_end]


def _Flatten(nmap_list):
  """Flattens every `.NestedMap` in nmap_list and concatenate them."""
  ret = []
  for x in nmap_list:
    ret += x.Flatten()
  return ret


def _Pack(flatten, nmap_list):
  """Packs the list of tensors according to `.NestedMap` in `nmap_list`.

  `_Pack` is loosely the inverse of `_Flatten`.

  Args:
    flatten: A list of tensors.
    nmap_list: A list of `.NestedMap`.

  Returns:
    A list of `.NestedMap`, say ret is the returned list. We have

      1. len(ret) == len(nmap_list);
      2. recursively, ret[i] has the same keys as nmap_list[i];
      3. _Flatten(ret) == flatten;
  """
  if not isinstance(flatten, (list, tuple)):
    flatten = [flatten]
  ret = []
  for x in nmap_list:
    # x needs num values from the head of flatten.
    num = len(x.Flatten())
    ret += [x.Pack(flatten[:num])]
    flatten = flatten[num:]
  assert not flatten, ('flatten does not match nmap_list.')
  return ret


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
    return inplace_ops.empty(
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

  return nmap.Transform(
      lambda x: inplace_ops.empty(shape, dtype=x.dtype, init=True))


def _EmptyLike(nmap):
  """Creates a set of empty initialized tensors.

  Args:
    nmap: A `.NestedMap` of tensors.

  Returns:
    A `.NestedMap` of tensors. Each tensor has the same shape and dtype as
    its corresponding tensor in nmap. And each tensor is initialized.
  """
  return nmap.Transform(lambda x: inplace_ops.empty_like(x, init=True))


def _EmptyCaptures():
  """Creates an empty implicit capture map for when capture is not supported."""
  return py_utils.NestedMap(captures=[])


def _Add(nmap_x, nmap_y):
  """Adds tensors in nmap_x with respective tensors in nmap_y.

  Args:
    nmap_x: A `.NestedMap` of tensors.
    nmap_y: A `.NestedMap` of tensors.

  Returns:
    A `.NestedMap` of tensors. ret.key = nmap_x.key + nmap_y.key for every key.
  """
  x_lst = nmap_x.Flatten()
  y_lst = nmap_y.Flatten()
  z = []
  for x, y in zip(x_lst, y_lst):
    z += [tf.add(x, y)]
  return nmap_x.Pack(z)


def _Dtypes(nmap_list):
  """Returns all tensors' data types in a list."""
  flatten = []
  for x in nmap_list:
    flatten += x.Flatten()
  return [x.dtype for x in flatten]


def _ConvertNoneGradientToZeros(xs, dxs):
  """Sanitize dxs so that None becomes zeros appropriately.

  Args:
    xs: A list of tensors.
    dxs: A list of tensors. dxs[i] corresponds to xs[i]'s gradient.

  Returns:
    A `.NestedMap` same as dxs with None replaced by a zero tensor.
  """
  xs_lst = _Flatten(xs)
  dxs_lst = _Flatten(dxs)

  # If x does not get any backprop-ed gradient, propagate zeros.
  rets = []
  for (x, dx) in zip(xs_lst, dxs_lst):
    if dx is None:
      rets.append(tf.zeros_like(x))
    else:
      rets.append(dx)

  return _Pack(rets, dxs)


def _TransformDType(nmap):
  return nmap.Transform(
      lambda x: tf.cast(x, tf.int64) if x.dtype == tf.int32 else x)


class _Recurrent(object):
  """A helper class to construct a recurrent neural net."""

  def __init__(self,
               cell_fn,
               cell_grad,
               theta,
               state0,
               inputs,
               extras,
               implicit_captures=None,
               unused_acc_state=None):
    """RNN helper class.

    Args:
      cell_fn: A python function, which computes:
         state1, extras = cell_fn(theta, state0, inputs[t, :])
      cell_grad: A python function which computes:
         dtheta, dstate0, dinputs[t, :] = cell_grad(
           theta, state0, inputs[t, :], extras, dstate1)
      theta: weights. A `.NestedMap`.
      state0: initial state. A `.NestedMap`.
      inputs: inputs. A `.NestedMap`.
      extras: A `.NestedMap` of Tensors. The 2nd return value of every
        invocation of cell_fn is a `.NestedMap` with matching keys and shapes
        of this 'extras'.
      implicit_captures: A `.NestedMap` corresponding to implicit captures of
        the cell_fn. If empty/None, implicit captures are either not present
        or disallowed.
      unused_acc_state: If None, we assume every field of acc_state is consumed
        in the following timestamps. If True, None of the acc_state is consumed.
        And we reduce_sum each timestep's new state into a scalar.
        Note, this feature should be used with StackedRecurrent where we send
        out the new state to the other devices.
    """
    self._theta = theta
    self._state = state0
    self._inputs = inputs
    self._cell_fn = cell_fn
    self._cell_grad = cell_grad
    self._extras = extras
    self._implicit_captures = implicit_captures
    self._unused_acc_state = unused_acc_state

    if self._implicit_captures is None:
      self._implicit_captures = _EmptyCaptures()

    # pylint: disable=unbalanced-tuple-unpacking

    # NOTE: TF Function (Fwd, Bak, ForwardLoopBody, BackwardLoopBody,
    # Forward and Backward defined below) simply takes a list of
    # Tensors and returns a list of Tensors. When we pass in a
    # structure (a list of NestedMap of Tensors), we use _Flatten to
    # convert the structure into a list of tensor. Conversely, the
    # following code often uses _Pack to formulate a structure from a
    # list of tensors based on a "template".

    # Wraps cell_fn in a TF Function:
    #    state1 = cell_fn(theta, state0, inputs)
    fwd_sig = [self._theta, self._state, self._inputs]

    compiled = py_utils.use_tpu()
    noinline = not compiled
    dev_t_type = tf.int32 if py_utils.use_tpu() else tf.int64

    @function.Defun(*_Dtypes(fwd_sig))
    def Fwd(*args):
      (theta, state0, inputs) = _Pack(args, fwd_sig)
      state1, extras = self._cell_fn(theta, state0, inputs)
      _AssertIsCompatible(state1, self._state)
      _AssertIsCompatible(extras, self._extras)
      return _Flatten([state1, extras])

    # Wraps cell_fn in a TF Function as a for-loop's body.
    #
    # The loop state is composed of:
    #  t: The loop variable. Timestep id.
    #  dev_t: The loop variable mirrored on the device.
    #  theta: the recurrent net's weights.
    #  state0: the previous recurrent state.
    #  inputs: inputs to the recurrent net. inputs[t, :] are for the timestep t.
    #  acc_state: Each timestep's computed new state is also stashed into
    #    acc_state.
    #  acc_extras: Each timestep's computed extras is stashed into acc_extras
    fwdloop_sig = [
        self._theta, self._state, self._inputs, self._state, self._extras
    ]

    @function.Defun(tf.int32, dev_t_type, *_Dtypes(fwdloop_sig))
    def ForwardLoopBody(*args):
      """The body of forward loop."""
      t, dev_t = args[0], args[1]
      (theta, state0, inputs, acc_state, acc_extras) = _Pack(
          args[2:], fwdloop_sig)
      inputs_t = _Index(inputs, t)  # external input at time step t.
      state1, extras = _Pack(
          Fwd(*_Flatten([theta, state0, inputs_t])),
          [self._state, self._extras])
      # Saves state1 and extras in their accumulators.
      if not self._unused_acc_state:
        acc_state = _Update(acc_state, state1, dev_t)
      acc_extras = _Update(acc_extras, extras, dev_t)

      return [tf.add(dev_t, 1)] + _Flatten(
          [theta, state1, inputs, acc_state, acc_extras])

    def Grad(op, *args):
      """The python grad function for the Forward function.

      Flowchart:
      +------------------------------------------------------------+
      |  Backward() DEFUN -> [d_fwd..., acc_extras, dcaptured]     |
      |                          |                                 |
      |                          v                                 |
      |                For(BackwardLoopBody())                     |
      |                          |                                 |
      |                          v                                 |
      |                BackwardLoopBody() DEFUN ->                 |
      |             ..., d_theta, d_state0, d_inputs,              |
      |                 d_acc_state, d_captured                    |
      |                          |                                 |
      |                          v                                 |
      |          Bak(..., inputs[t], extras[t]) DEFUN ->           |
      |       d_theta_t, d_state0, d_inputs_t, d_captured_t        |
      |                          |                                 |
      |                          v                                 |
      |      CellGrad(theta, state0, inputs, extras, d_state1) ->  |
      |               dtheta, dstate0, dinputs, dcaptured          |
      |                                                            |
      +------------------------------------------------------------+

      The key thing is that this function must return a dx value for each of
      the inputs to the Fwd function (theta, state0, inputs, captured...).
      The tricky part is that implicitly captured inputs are carried through
      function boundaries implicitly by the function call as the last
      arguments. When assembling gradients, we must account for these implicit
      captures even though they are not passed explicitly from function to
      function.

      Args:
        op: The forward operation.
        *args: Args to the forward operation (includes implicit captures).
      Returns:
        Tuple of derivitives.
      Raises:
        ValueError: on argument mismatch issues.
      """
      expected_num_inputs = 0
      for nmap in [
          self._theta,
          self._state,
          self._inputs,
          self._extras,
          # Implicit captured tensors always come last
          self._implicit_captures
      ]:
        expected_num_inputs += len(nmap.Flatten())
      if len(op.inputs) != expected_num_inputs:
        if len(op.inputs) > expected_num_inputs:
          raise ValueError(
              ('Too many inputs. The most likely cause is that cell_fn '
               'captures additional tensors: extra inputs %r vs captures %r') %
              (list(op.inputs), list(self._implicit_captures.Flatten())))
        raise ValueError(
            ('Mismatched inputs to cell fn: Found %d vs expected %d: %r'
             '. Implicit captures(%d) = %r') %
            (len(op.inputs), expected_num_inputs, list(op.inputs),
             len(self._implicit_captures.Flatten()), self._implicit_captures))

      # NOTE: tf.gradient backprops None for int32/int64 while zeros
      # for float32/float64. For consistency, we always backprop
      # zeros.
      args = list(args)
      for i, dy in enumerate(args):
        if dy is None:
          args[i] = tf.zeros_like(op.outputs[i])
      (theta, state0, inputs, _, unused_captured) = _Pack(
          [x for x in op.inputs],
          [
              self._theta,
              self._state,
              self._inputs,
              self._extras,
              # Implicit captured tensors always come last
              self._implicit_captures,
          ])
      # acc_state and acc_extras are computed by the Forward pass and
      # needed by the Backward pass.
      acc_state, _, acc_extras = _Pack([x for x in op.outputs],
                                       [self._state, self._state, self._extras])

      # Forward computes acc_state, the final state and
      # acc_extras. tf.gradients gives us their gradients w.r.t. the
      # final loss. Because acc_extras are not exposed by Compute(),
      # it has no gradients w.r.t. the final loss (i.e., by
      # construction, it must be zeros).
      d_acc_state, d_state1, _ = _Pack(args,
                                       [self._state, self._state, self._extras])

      if self._unused_acc_state:
        # XLA While op requires the same shape for the init and carry on values.
        state0 = state0.Transform(tf.reduce_sum)
        d_state1 = d_state1.Transform(tf.reduce_sum)

      return Backward(*_Flatten([
          theta,
          state0,
          inputs,
          acc_state,
          acc_extras,
          d_acc_state,
          d_state1,
      ]))

    # Forward calls ForwardLoopBody n times. Each time computes one
    # time step of the recurrent net.
    forward_sig = [self._theta, self._state, self._inputs, self._extras]

    @function.Defun(
        *_Dtypes(forward_sig), python_grad_func=Grad, noinline=noinline)
    def Forward(*args):
      """Forward pass of the recurrent net."""
      theta, state0, inputs, extras = _Pack(args, forward_sig)

      # The sequence length.
      pad_begin, pad_end = _SeqPaddingLength(inputs)
      slen_dim = _SeqLenDim(inputs)

      # Creates accumulators for state0 and extras.
      if self._unused_acc_state:
        acc_state = _EmptyWithFixShape([slen_dim], state0)
      else:
        acc_state = _EmptyAcc(slen_dim, state0)
      acc_extras = _EmptyAcc(slen_dim, extras)

      if py_utils.use_tpu():
        dev_t = tf.to_int32(pad_begin)
      else:
        dev_t = tf.to_int64(pad_begin)
      run = functional_ops.For(
          start=pad_begin,
          limit=slen_dim - pad_end,
          delta=1,
          inputs=[dev_t] + _Flatten(
              [theta, state0, inputs, acc_state, acc_extras]),
          body=ForwardLoopBody,
          rewrite_with_while=compiled)
      _, state1, _, acc_state, acc_extras = _Pack(
          run[1:],
          [self._theta, self._state, self._inputs, self._state, self._extras])

      return _Flatten([acc_state, state1, acc_extras])

    # The per-step backward computes:
    #    d_theta, d_state0, d_inputs = cell_grad(
    #        theta, state0, inputs, extras, d_state1)
    # where d_state1 is the backprop-ed gradient for state1, and
    # extras is the computed by the forward step to facilitate the
    # backward step.
    bak_sig = [
        self._theta,
        self._state,
        self._inputs,
        self._extras,
        self._state,
    ]

    @function.Defun(*_Dtypes(bak_sig))
    def Bak(*args):
      """Backward step."""
      (theta, state0, inputs, extras, d_state1) = _Pack(args, bak_sig)
      (dtheta, dstate0, dinputs, dcaptures) = self._cell_grad(
          theta, state0, inputs, extras, d_state1)
      _AssertIsCompatible(dtheta, self._theta)
      _AssertIsCompatible(dstate0, self._state)
      _AssertIsCompatible(dinputs, self._inputs)
      if dcaptures is None:
        # NOTE: Custom gradient fns can return None if they do not support
        # captured tensors. The return value is reserved for the future when
        # that may be supported.
        dcaptures = _EmptyLike(self._implicit_captures)
      _AssertIsCompatible(dcaptures, self._implicit_captures)

      # Make sure this function didn't capture anything different than the
      # cell_fn when reflected on at the beginning. Must come after the call
      # to cell_grad() which adds to the captured list.
      _AssertSameTensors(function.get_extra_inputs(),
                         self._implicit_captures.Flatten())

      (captured,) = _Pack(function.get_extra_args(), [self._implicit_captures])
      return _Flatten(
          _ConvertNoneGradientToZeros([theta, state0, inputs, captured],
                                      [dtheta, dstate0, dinputs, dcaptures]))

    # Define defuns used by a functional.if in BackwardLoopBody.
    state_if_sig = [self._state, self._state]

    @function.Defun(*_Dtypes(state_if_sig))
    def ReturnOrigState0(*args):
      """Returns original state0 from inputs."""
      (_, orig_state0) = _Pack(args, state_if_sig)
      return orig_state0.Flatten()

    @function.Defun(*_Dtypes(state_if_sig))
    def ReturnAccState(*args):
      """Returns acc_state[t-1] from inputs."""
      (acc_state, _) = _Pack(args, state_if_sig)
      return acc_state.Flatten()

    # Wraps cell_grad gradient function in a TF Function as a
    # for-loop's body for the Backward pass.
    #
    # The loop state is composed of:
    #  t: The loop variable. Timestep id.
    #  state0: the initial state for the entire backward loop.
    #  dev_t: The loop variable mirrored on the device.
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
    bakloop_sig = [
        self._theta,
        self._state,
        self._inputs,
        self._state,
        self._extras,
        # End of forward params
        self._theta,
        self._state,
        self._inputs,
        self._state,
        self._implicit_captures,
    ]

    @function.Defun(tf.int32, dev_t_type, *_Dtypes(bakloop_sig))
    def BackwardLoopBody(*args):
      """Backward loop body function."""
      t, dev_t = args[0], args[1]
      (
          theta,
          orig_state0,
          inputs,
          acc_state,
          acc_extras,
          # End of forward params
          d_theta,
          d_state1,
          d_inputs,
          d_acc_state,
          d_captured) = (
              _Pack(args[2:], bakloop_sig))

      # The input recurrent state for time step t is previous time step's
      # output, or the original state0 when on time step 0.
      state_from_acc = _Index(acc_state, tf.maximum(0, t - 1))
      state0 = functional_ops.If(
          tf.equal(t, tf.constant(0, tf.int32)),
          _Flatten([state_from_acc, orig_state0]), ReturnOrigState0,
          ReturnAccState)
      state0 = orig_state0.Pack(state0)

      # The external inputs for time step t.
      inputs_t = _Index(inputs, t)
      # The extras for time step t.
      extras_t = _Index(acc_extras, t)

      d_state1 = _Add(_Index(d_acc_state, t), d_state1)
      (d_theta_t, d_state0, d_inputs_t, d_captured_t) = _Pack(
          Bak(*_Flatten([theta, state0, inputs_t, extras_t, d_state1])),
          [self._theta, self._state, self._inputs, self._implicit_captures])

      if self._unused_acc_state:
        # XLA IF op requires the same shape for if and else branches.
        d_state0 = d_state0.Transform(tf.reduce_sum)
      d_theta = _Add(d_theta, d_theta_t)
      d_inputs = _Update(d_inputs, d_inputs_t, dev_t)
      d_captured = _Add(d_captured, d_captured_t)

      # Make sure this function didn't capture anything different than the
      # cell_fn when reflected on at the beginning. Must come after the call
      # to Bak() which adds to the captured list.
      _AssertSameTensors(function.get_extra_inputs(),
                         self._implicit_captures.Flatten())

      return [tf.subtract(dev_t, 1)] + _Flatten([
          theta,
          orig_state0,
          inputs,
          acc_state,
          acc_extras,
          # End of forward params
          d_theta,
          d_state0,
          d_inputs,
          d_acc_state,
          d_captured,
      ])

    # Backward calls BackwardLoopBody n times.  Each time computes the backprop
    # for one time step of the recurrent net.
    backward_sig = [
        self._theta,
        self._state,
        self._inputs,
        self._state,
        self._extras,
        # End of forward params.
        self._state,
        self._state,
    ]

    @function.Defun(*_Dtypes(backward_sig), noinline=noinline)
    def Backward(*args):
      """Backward pass for the recurrent net."""
      # theta, state0, inputs are Forward's inputs.
      # acc_state is the accumulated 1st output of Forward.
      # acc_extras is the accumulated 2nd output of Forward.
      # d_acc_state is the gradient for acc_state.
      # d_state1 is the gradient for the final state computed by Forward.
      (theta, state0, inputs, acc_state, acc_extras, d_acc_state,
       d_state1) = _Pack(args, backward_sig)

      # Accumulators for gradients.
      d_theta = _EmptyLike(theta)
      d_inputs = _EmptyLike(inputs)
      d_captured = _EmptyLike(self._implicit_captures)

      # The sequence length.
      pad_begin, pad_end = _SeqPaddingLength(inputs)
      start = _SeqLenDim(inputs) - pad_end - 1

      if py_utils.use_tpu():
        dev_t = tf.to_int32(start)
      else:
        dev_t = tf.to_int64(start)
      run = functional_ops.For(
          start=start,
          limit=pad_begin - 1,
          delta=-1,
          inputs=[dev_t] + _Flatten([
              theta,
              state0,
              inputs,
              acc_state,
              acc_extras,
              d_theta,
              d_state1,
              d_inputs,
              d_acc_state,
              d_captured,
          ]),
          body=BackwardLoopBody,
          rewrite_with_while=compiled)

      (theta, state0, inputs, acc_state, acc_extras, d_theta, d_state0,
       d_inputs, d_acc_state, d_captured) = _Pack(run[1:], bakloop_sig)

      # Make sure this function didn't capture anything different than the
      # cell_fn when reflected on at the beginning. Must come after the
      # call to BackwardLoopBody, which adds to the captured list.
      _AssertSameTensors(function.get_extra_inputs(),
                         self._implicit_captures.Flatten())

      if self._unused_acc_state:
        # Match the shape of gradient of the init_state.
        d_state0 = self._state.Transform(tf.zeros_like)
      return _Flatten([d_theta, d_state0, d_inputs, acc_extras, d_captured])

    self._forward = Forward

  def Compute(self):
    return _Pack(
        self._forward(
            *_Flatten([self._theta, self._state, self._inputs, self._extras])),
        [self._state, self._state, self._extras])[:2]


def _GetCellGrad(cell_fn, cell_grad, implicit_captures=None):
  """Returns the gradient function for cell_fn.

  Args:
    cell_fn: The recurrent neural net's cell function.
    cell_grad: If not None, cell_fn's gradient function.
    implicit_captures: `.NestedMap` of implicit captures that cell_fn
        does. If None, then no captures are supported.

  Returns:
    Returns cell_grad if not None. Otherwise, assume cell_fn is a python
    function representing the recurrent neural net's cell function, i.e.::

      cell_fn: (theta, state0, inputs) -> (state1, extra)

    returns its default gradient python function, i.e.::

      cell_grad: (theta, state0, inputs, extras, captured, dstate1) ->
          (dtheta, dstate0, dinputs)
  """

  if cell_grad:
    return cell_grad

  if implicit_captures is None:
    # Matches the structure in the Recurrent constructor.
    implicit_captures = _EmptyCaptures()

  def CellGrad(theta, state0, inputs, extras, dstate1):
    """Default gradient function for cell_fn."""
    state1, extras = cell_fn(theta, state0, inputs)
    assert isinstance(state1, py_utils.NestedMap), ('%s' % state1)
    assert isinstance(extras, py_utils.NestedMap), ('%s' % extras)
    # NOTE: The default grad function recomputes the forward
    # function and does not take advantage of 'extras' returned by
    # the forward function.

    # Assert that if captured inputs were given, they match the actual
    # tensors passed to the function we are compiled into. Must come after
    # the call to cell_fn, which does the capture.
    _AssertSameTensors(function.get_extra_inputs(), implicit_captures.Flatten())

    # Extract the internal captured tensor placeholders within the Defun
    # we are running in.
    (captured,) = _Pack(function.get_extra_args(), [implicit_captures])
    ys = _Flatten([state1])
    xs = _Flatten([theta, state0, inputs, captured])
    grad_ys = _Flatten([dstate1])
    grads = tf.gradients(ys=ys, xs=xs, grad_ys=grad_ys)
    return _ConvertNoneGradientToZeros([theta, state0, inputs, captured],
                                       _Pack(grads,
                                             [theta, state0, inputs, captured]))

  return CellGrad


def _WrapAccumulatorCellFn(accumulator_layer, cell_fn):
  """Wrap a cell_fn to propagate accumulators."""

  def WrappedCellFn(theta, state0, inputs):
    accumulator_layer.SetAccumulatorValues(state0.accumulators)
    # The underlying cell_fn has no knowledge of accumulator state so
    # delete it.
    state0 = state0.DeepCopy()
    del state0.accumulators
    state1, extras = cell_fn(theta, state0, inputs)
    # Propagate new accumulator state forward.
    state1.accumulators = accumulator_layer.GetAccumulatorValues()
    # Reset: make sure nothing escapes.
    accumulator_layer.accumulators.Transform(lambda x: x.Reset())
    return state1, extras

  return WrappedCellFn


def _WrapAccumulatorCellGradFn(accumulator_layer, cell_grad):
  """Wrap a cell grad function to disable accumulators."""

  def WrappedCellGradFn(theta, state0, inputs, extras, dstate1):
    # Compute the cell grad function with accumulators disabled.
    accumulator_layer.accumulators.Transform(lambda x: x.Disable())
    dtheta, dstate0, dinputs, dcaptures = cell_grad(theta, state0, inputs,
                                                    extras, dstate1)
    accumulator_layer.accumulators.Transform(lambda x: x.Enable())
    return dtheta, dstate0, dinputs, dcaptures

  return WrappedCellGradFn


def _IsSingleTimeStep(inputs):
  """Returns True only if the time dimension of inputs is 1."""
  for x in inputs.Flatten():
    if x.shape.dims is None or x.shape[0].value != 1:
      return False
  return True


def _ReflectOnCellFn(cell_fn,
                     theta,
                     state0,
                     inputs,
                     accumulator_layer=None,
                     check_stateful_ops=False):
  """Reflects on the cell_fn, applying asserts and returning needed info.

  Args:
    cell_fn: A python function that computes:
      state1, extras = cell_fn(theta, state0, inputs[t, :])
    theta: weights. A `.NestedMap`.
    state0: initial state. A `.NestedMap`.
    inputs: inputs. A `.NestedMap`.
    accumulator_layer: Whether the cell function must be run in the context
        of the given accumulator layer.
    check_stateful_ops: if True, raise a `ValueError` if cell_fn is stateful.
  Returns:
    `.NestedMap` of implicit captures that the cell_fn takes.
  Raises:
    ValueError: cell_fn is stateful.
  """
  fwd_sig = [theta, state0, inputs]

  # Disable accumulators.
  if accumulator_layer:
    accumulator_layer.accumulators.Transform(lambda x: x.Disable())
  try:

    @function.Defun(*_Dtypes(fwd_sig))
    def Fwd(*args):
      (theta, state0, inputs) = _Pack(args, fwd_sig)
      state1, extras = cell_fn(theta, state0, inputs)
      return _Flatten([state1, extras])

    # This can trigger instantiating the function, so do it in the context
    # of accumulators disabled to be safe.
    captured_inputs = list(Fwd.captured_inputs)
  finally:
    # Enable accumulators.
    if accumulator_layer:
      accumulator_layer.accumulators.Transform(lambda x: x.Enable())

  # Asserts about the function.
  if Fwd.stateful_ops:
    if check_stateful_ops:
      raise ValueError('cell_fn contains stateful ops: %s' % Fwd.stateful_ops)
    else:
      tf.logging.warn('cell_fn contains stateful ops: %s', Fwd.stateful_ops)

  return py_utils.NestedMap(captured=captured_inputs)


def _NestedMapCopier(nmap):
  """Returns a function that will DeepCopy the map on each call."""

  def Copier():
    return nmap.DeepCopy()

  return Copier


def _RecurrentSingleTimeStep(theta, state0, inputs, cell_fn):
  """Short-cut for the single timestep without explicit cell_grad case."""
  # The seqlen length is staticly known as 1. Hence, we just need to
  # call cell_fn once without putting it into a loop.
  # Since we are not looping, there is no need to specially manage
  # accumulators.
  inputs = inputs.Transform(lambda x: tf.squeeze(x, axis=0))
  state1, _ = cell_fn(theta, state0.DeepCopy(), inputs)
  acc_state = state1.Transform(lambda x: tf.expand_dims(x, axis=0))
  return acc_state, state1


def Recurrent(theta,
              state0,
              inputs,
              cell_fn,
              cell_grad=None,
              extras=None,
              check_stateful_ops=False,
              accumulator_layer=None,
              allow_implicit_capture=False):
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
    cell_fn: A python function, which computes::

        state1, extras = cell_fn(theta, state0, inputs[t, :])

    cell_grad: A python function which computes::

        dtheta, dstate0, dinputs[t, :], dcaptured = cell_grad(
            theta, state0, inputs[t, :], extras, dstate1)

      If there are no captured tensors in `cell_fn`, `dcaptured` can be returned
      as None. Captured tensors with custom `cell_grad` is currently unsupported
      so this return value is reserved for future expansion.
    extras: A `.NestedMap` of Tensors. The 2nd return value of every
      invocation of `cell_fn` is a `.NestedMap` with matching keys and shapes
      of `extras`.
    check_stateful_ops: if True, raise a `ValueError` if `cell_fn` is stateful.
    accumulator_layer: If provided, then accumulators on this layer will be
      managed such that they carry to the final state in `FProp` and are
      disabled for gradients. Uses the state key `accumulators`.
    allow_implicit_capture: Whether to allow the `cell_fn` to implicitly
      capture tensors. Only allowed if an explicit `cell_grad` is not given.
  Returns:
    `accumulate_state` and the final state.
  """
  inputs = _TransformDType(inputs)

  # Short-cut for the single timestep with default grad function case.
  if cell_grad is None and _IsSingleTimeStep(inputs):
    return _RecurrentSingleTimeStep(theta, state0, inputs, cell_fn)

  # Whether we have accumulators to manage state for.
  has_accumulators = False
  if accumulator_layer:
    assert 'accumulators' not in state0, (
        'Duplicate "accumulators" key in state0.')
    accumulator_values = accumulator_layer.GetAccumulatorValues()
    if accumulator_values.Flatten():
      state0 = state0.DeepCopy()
      state0.accumulators = accumulator_values
      has_accumulators = True

  # Make it impossible to use a shared state0 since cell_fn can have
  # side-effects and we want each call below to have a fresh one.
  new_state0 = _NestedMapCopier(state0)
  del state0

  # Reflect on the cell_fn for assertions and info.
  implicit_captures = _ReflectOnCellFn(
      cell_fn,
      theta,
      new_state0(),
      inputs,
      accumulator_layer=accumulator_layer,
      check_stateful_ops=check_stateful_ops)

  assert (
      not implicit_captures.Flatten() or
      (allow_implicit_capture and cell_grad is None)), (
          ('Recurrent cell_fn implicitly captures tensors but '
           'implicit capture is disabled or a custom cell_grad fn '
           'is in use. Captured tensors: %r') % (implicit_captures.Flatten(),))

  # Wrap the cell_fn so that it knows how to propagate accumulators.
  if has_accumulators:
    cell_fn = _WrapAccumulatorCellFn(accumulator_layer, cell_fn)

  # If cell_grad is not given, derives the gradient function from
  # cell_fn.
  cell_grad = _GetCellGrad(cell_fn, cell_grad, implicit_captures)

  # Wrap the cell_grad so it disables accumulators.
  if has_accumulators:
    cell_grad = _WrapAccumulatorCellGradFn(accumulator_layer, cell_grad)

  if extras is None:
    with tf.name_scope('recurrent_cellfn_extras'):
      # Derives 'extras' so that we can allocate extras' accumulator.
      _, extras = cell_fn(theta, new_state0(), _Index(inputs, 0))
      extras = extras.Transform(tf.zeros_like)
  elif not extras:
    # Forces the extras to be an empty map if an empty 'extras' is provided.
    extras = py_utils.NestedMap()
  else:
    _, actual = cell_fn(theta, new_state0(), _Index(inputs, 0))
    _AssertIsCompatible(extras, actual)

  acc_state, final_state = _Recurrent(
      cell_fn=cell_fn,
      cell_grad=cell_grad,
      theta=theta,
      state0=new_state0(),
      inputs=inputs,
      extras=extras,
      implicit_captures=implicit_captures).Compute()

  if has_accumulators:
    # Restore the accumulators from the final recurrent state.
    accumulator_layer.SetAccumulatorValues(final_state.accumulators)
    del acc_state.accumulators
    del final_state.accumulators

  return acc_state, final_state
