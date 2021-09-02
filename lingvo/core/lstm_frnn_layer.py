# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Functional LSTM RNN layer with LSTM cells fused into FRNN to run input ...

projections in parallel.
"""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import recurrent
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers


class LSTMCellExt:
  """Extends LSTM-based cell classes with extra methods for parallelizing ...

  input projections across steps.
  """

  def ProjectInputSequence(self, theta, inputs):
    """Applies input projection for the entire sequence.

    Args:
      theta: a NestedMap of layer weights. Notably, it's expected to contain
        separate weight tensors for input and hidden state projections, for
        performance reasons, under the key 'wm_i' (input) and 'wm_h'
      inputs: A NestedMap with the following fields:
        - act: A list of Tensors of shape [seqlen, batch, input_dim].

    Returns:
      A Tensor of shape [seqlen, batch, 4 * hidden_dim].
    """
    assert isinstance(inputs.act, list)
    if len(inputs.act) > 1:
      x = tf.concat(inputs.act, -1)
    else:
      x = inputs.act[0]
    # [T, B, 4 * H]
    proj_inputs = tf.einsum('TBD,DH->TBH', x, theta.wm_i)
    return proj_inputs

  def _MixWithProjectedInput(self, theta, state0, inputs):
    """Computes _Mix() with inputs already projected.

    Args:
      theta: a NestedMap of layer weights. Notably, it's expected to contain
        separate weight tensors for input and hidden state projections, for
        performance reasons, under the key 'wm_i' (input) and 'wm_h'
      state0: A NestedMap with the same structure as return value of
        `self.zero_state()`.
      inputs: A Tensor of shape [batch, 4 * hidden_dim].

    Returns:
      A Tensor of the same shape as `inputs`.
    """
    proj_m = tf.matmul(state0.m, theta.wm_h)
    return inputs + proj_m

  def FPropWithProjectedInput(self, theta, state0, inputs):
    """FProp with inputs already projected.

    This method is for parallelizing the input projection across time steps to
    accelerate training.

    The following are equivalent:

    >>> inputs = <a tensor of [T, B, D]>
    >>> paddings = tf.zeros([T, B])
    >>> theta = cell.theta
    >>> state = cell.zero_state(theta, B)

    # a. Use FProp().
    >>> for i in range(T):
    ...  state, _ = cell.FProp(theta, inputs[i, :, :], paddings, state)

    # b. Use FPropWithProjectedInput().
    >>> proj_inputs = cell.ProjectInputSequence(theta, inputs)
    >>> for i in range(T):
    ...  state, _ = cell.FPropWithProjectedInputs(
    ...    theta, proj_inputs[i, :, :], paddings, state)

    Args:
      theta: a NestedMap of layer weights. Notably, it's expected to contain
        separate weight tensors for input and hidden state projections, for
        performance reasons, under the key 'wm_i' (input) and 'wm_h' (hidden
        state).
      state0: A NestedMap with the same structure as return value of
        `self.zero_state()`.
      inputs: A NestedMap with the following fields:
        - proj_inputs: A single Tensors of shape [batch, 4 * hidden_dim].
        - padding: A Tensor of shape [batch, 1].
        - reset_mask: A Tensor of shape [batch, 1].

    Returns:
      state1: A NestedMap of the same structure as `state0`.
      extras: Intermediate results to facilitate backprop. A NestedMap.
    """
    if self.params.reset_cell_state:
      state0_modified = self._ResetState(state0.DeepCopy(), inputs)
    else:
      state0_modified = state0
    xmw = self._MixWithProjectedInput(theta, state0_modified,
                                      inputs.proj_inputs)
    gates_input = inputs.copy()
    gates_input.act = [inputs.proj_inputs]
    state1 = self._Gates(xmw, theta, state0_modified, gates_input)
    return state1, py_utils.NestedMap()


class LSTMCellSimpleExt(rnn_cell.LSTMCellSimple, LSTMCellExt):
  """Extends LSTMCellSimple with extra methods for parallelizing ...

  input projections across steps.
  """
  pass


class LayerNormalizedLSTMCellSimpleExt(rnn_cell.LayerNormalizedLSTMCellSimple,
                                       LSTMCellExt):
  """Extends LayerNormalizedLSTMCellSimple with extra methods for ...

  parallelizing input projections across steps.
  """
  pass


class LayerNormalizedLSTMCellLeanExt(rnn_cell.LayerNormalizedLSTMCellLean,
                                     LSTMCellExt):
  """Extends LayerNormalizedLSTMCellLean with extra methods for parallelizing ...

  input projections across steps.
  """
  pass


class LstmFRNN(base_layer.BaseLayer):
  """A FRNN for LSTMCellSimple or LayerNormalizedLSTMCellLean cell.

  It exploits the parallelism in input projection across time steps, and is in
  general faster than the combination of LayerNormalizedLSTMCellLean and FRNN.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    p.Define(
        'cell', None,
        'Configs for the RNN cell. Supported classes are LSTMCellSimpleExt, '
        'LayerNormalizedLSTMCellLeanExt.')
    p.Define('reverse', False,
             'Whether or not to unroll the sequence in reversed order.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.cell.cls not in (LSTMCellSimpleExt, LayerNormalizedLSTMCellSimpleExt,
                          LayerNormalizedLSTMCellLeanExt):
      raise ValueError(
          'Only LSTMCellSimpleExt, LayerNormalizedLSTMCellSimpleExt and '
          'LayerNormalizedLSTMCellLeanExt are supported, got {}.'.format(
              p.cell.cls.__name__))
    self.CreateChild('cell', p.cell)

  def zero_state(self, theta, batch_size):
    return self.cell.zero_state(theta.cell, batch_size)

  def FProp(self, theta, inputs, paddings, state0=None, segment_id=None):
    """Computes LSTM forward pass.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A single tensor or a tuple of tensors with cardinality equal to
        rnn_cell.inputs_arity. For every input tensor, the first dimension is
        assumed to be time, second dimension batch, and third dimension depth.
      paddings: A tensor. First dim is time, second dim is batch, and third dim
        is expected to be 1.
      state0: If not None, the initial rnn state in a `.NestedMap`. Defaults to
        the cell's zero-state.
      segment_id: A tensor to support packed inputs. First dim is time, second
        dim is batch, and third dim is expected to be 1.

    Returns:
      A tensor of [time, batch, dims].
      The final recurrent state.
    """
    p = self.params
    assert isinstance(self.cell, rnn_cell.RNNCell)

    if not isinstance(inputs, (list, tuple)):
      inputs = [inputs]

    # Slicing wm to wm_{i,h} outside the loop to get 20% speedup over regular
    # LSTM baseline.
    # Keeping slicing within the loop gives only < 3% speedup.
    cell_theta = theta.cell.copy()
    num_input_nodes = p.cell.num_input_nodes
    cell_theta['wm_i'] = cell_theta.wm[:num_input_nodes, :]
    cell_theta['wm_h'] = cell_theta.wm[num_input_nodes:, :]
    tf.logging.vlog(1, 'cell_theta: %r', cell_theta)
    if p.packed_input:
      assert segment_id is not None
      reset_mask = rnn_layers.GeneratePackedInputResetMask(
          segment_id, is_reverse=False)
      reset_mask = py_utils.HasShape(reset_mask, tf.shape(paddings))
    else:
      reset_mask = tf.zeros_like(paddings)

    if p.reverse:
      inputs = [tf.reverse(x, [0]) for x in inputs]
      paddings = tf.reverse(paddings, [0])
      reset_mask = tf.reverse(reset_mask, [0])

    if not state0:
      batch_size = py_utils.GetShape(paddings)[1]
      state0 = self.cell.zero_state(cell_theta, batch_size)

    # [T, B, H]
    proj_inputs = self.cell.ProjectInputSequence(cell_theta,
                                                 py_utils.NestedMap(act=inputs))
    proj_inputs = py_utils.NestedMap(
        proj_inputs=proj_inputs, padding=paddings, reset_mask=reset_mask)

    acc_state, final_state = recurrent.Recurrent(
        theta=cell_theta,
        state0=state0,
        inputs=proj_inputs,
        cell_fn=self.cell.FPropWithProjectedInput,
        cell_type=self.cell.layer_type,
        accumulator_layer=self,
        allow_implicit_capture=p.allow_implicit_capture)

    act = self.cell.GetOutput(acc_state)
    if p.reverse:
      act = tf.reverse(act, [0])
    return act, final_state
