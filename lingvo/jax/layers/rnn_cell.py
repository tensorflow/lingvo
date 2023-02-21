# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""RNN-related layers."""

from typing import Sequence, Tuple

import jax
from jax import numpy as jnp
from lingvo.jax import asserts
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
weight_params = py_utils.weight_params

InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor

RNN_CELL_WT = 'rnn_cell_weight_variable'


class RNNCell(base_layer.BaseLayer):
  """RNN cells.

  RNNCell represents recurrent state in a `.NestedMap`.

  `init_states(batch_size)` returns the initial state, which is defined
  by each subclass. From the state, each subclass defines `get_output()`
  to extract the output tensor.

  `RNNCell.fprop` defines the forward function::

      (state0, inputs) -> state1, extras

  All arguments and return values are `.NestedMap`. Each subclass defines
  what fields these `.NestedMap` are expected to have. `extras` is a
  `.NestedMap` containing some intermediate results `fprop` computes to
  facilitate the backprop.

  `init_states(batch_size)`, `state0` and `state1` are all compatible
  `.NestedMap` (see `.NestedMap.IsCompatible`).
  I.e., they have the same keys recursively. Furthermore, the corresponding
  tensors in these `.NestedMap` have the same shape and dtype.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('inputs_arity', 1,
             'number of tensors expected for the inputs.act to fprop.')
    p.Define('num_input_nodes', 0, 'Number of input nodes.')
    p.Define(
        'num_output_nodes', 0,
        'Number of output nodes. If num_hidden_nodes is 0, also used as '
        'cell size.')
    p.Define(
        'reset_cell_state', False,
        'Set True to support resetting cell state in scenarios where multiple '
        'inputs are packed into a single training example. The RNN layer '
        'should provide reset_mask inputs in addition to act and padding if '
        'this flag is set.')
    p.Define(
        'cell_value_cap', 10.0, 'Cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap] if the value is not None. '
        'It can be a scalar, a scalar tensor or None. When set to None, '
        'no capping is applied.')
    return p

  @property
  def _variable_collections(self) -> Sequence[str]:
    return [RNN_CELL_WT, '%s_vars' % self.__class__.__name__]

  def init_states(self, batch_size: int) -> NestedMap:
    """Returns the initial state given the batch size."""
    raise NotImplementedError('Abstract method')

  def get_output(self, state: NestedMap) -> NestedMap:
    """Returns the output value given the current state."""
    raise NotImplementedError('Abstract method')

  def _maybe_reset_state(self, state: NestedMap, inputs: NestedMap) -> JTensor:
    """Reset state if needed."""
    if self.params.reset_cell_state:
      state_modified = self._reset_state(state.DeepCopy(), inputs)
    else:
      state_modified = state
    return state_modified  # pytype: disable=bad-return-type  # jax-ndarray

  def fprop(self, state0: NestedMap,
            inputs: NestedMap) -> Tuple[NestedMap, NestedMap]:
    """Forward function.

    `_reset_state` is optionally applied if `reset_cell_state` is True. The RNN
    layer should provide `reset_mask` inputs in addition to other inputs.
    `reset_mask` inputs are expected to be 0 at timesteps where state0 should be
    reset to default (zeros) before running `fprop`, and 1
    otherwise. This is meant to support use cases like packed inputs, where
    multiple samples are fed in a single input example sequence, and need to be
    masked from each other. For example, if the two examples packed together
    are ['good', 'day'] -> ['guten-tag'] and ['thanks'] -> ['danke']
    to produce ['good', 'day', 'thanks'] -> ['guten-tag', 'danke'], the
    source reset_masks would be [1, 1, 0] and target reset masks would be
    [1, 0]. These ids are meant to enable masking computations for
    different examples from each other.

    Args:
      state0: The previous recurrent state.
      inputs: The inputs to the cell.

    Returns:
      state1: The next recurrent state.
      extras: Intermediate results to faciliate backprop.
    """
    asserts.instance(inputs.act, list)
    asserts.eq(self.params.inputs_arity, len(inputs.act))

    state0 = self._maybe_reset_state(state0, inputs)

    concat = jnp.concatenate(inputs.act + [state0.m], 1)  # pytype: disable=attribute-error  # jax-ndarray
    wm = self.local_theta().wm
    xmw = jnp.einsum('bd,dc->bc', concat, wm)

    i_i, i_g, f_g, o_g = self._retrieve_and_split_gates(xmw)
    state1 = self._gates_internal(state0, inputs, i_i, i_g, f_g, o_g)
    return state1, NestedMap()

  def _zoneout_internal(self, prev_v: JTensor, cur_v: JTensor,
                        padding_v: JTensor, zo_prob: float, is_eval: bool,
                        random_uniform: JTensor) -> JTensor:
    """Apply ZoneOut regularlization to cur_v.

    Implements ZoneOut regularization as described in
    https://arxiv.org/abs/1606.01305

    Args:
      prev_v: Values from the previous timestep.
      cur_v: Values from the current timestep.
      padding_v: The paddings vector for the cur timestep.
      zo_prob: Probability at which to apply ZoneOut regularization.
      is_eval: Whether or not in eval mode.
      random_uniform: Random uniform numbers. This can be None if zo_prob=0.0

    Returns:
      cur_v after ZoneOut regularization has been applied.
    """
    prev_v = jnp.array(prev_v)
    cur_v = jnp.array(cur_v)
    padding_v = jnp.array(padding_v)
    if zo_prob == 0.0:
      # Special case for when ZoneOut is not enabled.
      return jnp.where(padding_v, prev_v, cur_v)

    if is_eval:
      mix_prev = jnp.full(prev_v.shape, zo_prob) * prev_v
      mix_curr = jnp.full(cur_v.shape, 1.0 - zo_prob) * cur_v
      mix = mix_prev + mix_curr

      # If padding_v is 1, it always carries over the previous state.
      return jnp.where(padding_v, prev_v, mix)
    else:
      asserts.not_none(random_uniform)
      zo_p = (random_uniform < zo_prob).astype(padding_v.dtype)
      zo_p += padding_v
      # If padding_v is 1, we always carry over the previous state.
      zo_p = jnp.minimum(zo_p, 1.0)
      zo_p = jax.lax.stop_gradient(zo_p)
      return jnp.where(zo_p, prev_v, cur_v)


class LSTMCellSimple(RNNCell):
  """Simple LSTM cell.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1]
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define('forget_gate_bias', 0.0, 'Bias to apply to the forget gate.')
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('bias_init', WeightInit.Constant(0.0),
             'Initialization parameters for bias')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Initializes LSTMCellSimple."""
    super().__init__(params)
    p = self.params
    if p.cell_value_cap is not None:
      asserts.instance(p.cell_value_cap, (int, float))

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    # Define weights.
    wm_pc = weight_params(
        shape=[
            p.num_input_nodes + self.output_size,
            self.num_gates * self.hidden_size
        ],
        init=p.params_init,
        dtype=p.dtype,
        collections=self._variable_collections)
    self.create_variable('wm', wm_pc)

    if p.num_hidden_nodes:
      w_proj = weight_params(
          shape=[self.hidden_size, self.output_size],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._variable_collections)
      self.create_variable('w_proj', w_proj)

    bias_pc = weight_params(
        shape=[self.num_gates * self.hidden_size],
        init=p.bias_init,
        dtype=p.dtype,
        collections=self._variable_collections)
    self.create_variable('b', bias_pc)

  @property
  def output_size(self) -> int:
    return self.params.num_output_nodes

  @property
  def hidden_size(self) -> int:
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  @property
  def num_gates(self) -> int:
    return 4

  def init_states(self, batch_size: int) -> NestedMap:
    zero_m = jnp.zeros((batch_size, self.output_size))
    zero_c = jnp.zeros((batch_size, self.hidden_size))

    return NestedMap(m=zero_m, c=zero_c)

  def _reset_state(self, state: NestedMap, inputs: NestedMap) -> NestedMap:
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def get_output(self, state: NestedMap) -> JTensor:  # pytype: disable=signature-mismatch  # jax-ndarray
    return state.m

  def get_adjustment(self) -> JTensor:
    adjustment = jnp.ones([4, self.hidden_size]) * jnp.expand_dims(
        jnp.array([0., 0., self.params.forget_gate_bias, 0.]), axis=1)
    adjustment = jnp.reshape(adjustment, [self.num_gates * self.hidden_size])
    return adjustment

  def _get_bias(self) -> JTensor:
    """Gets the bias vector to add.

    Includes adjustments like forget_gate_bias. Use this instead of the 'b'
    variable directly as including adjustments in this way allows const-prop
    to eliminate the adjustments at inference time.

    Returns:
      The bias vector.
    """
    p = self.params
    b = self.local_theta().b
    if p.forget_gate_bias != 0.0:
      b = b + self.get_adjustment()

    return b

  def _split_gate(self,
                  xmw: JTensor) -> Tuple[JTensor, JTensor, JTensor, JTensor]:
    div = xmw.shape[1] // self.num_gates
    div2, div3 = 2 * div, 3 * div
    return xmw[:, :div], xmw[:, div:div2], xmw[:, div2:div3], xmw[:, div3:]

  def _retrieve_and_split_gates(
      self, xmw: JTensor) -> Tuple[JTensor, JTensor, JTensor, JTensor]:
    b = jnp.expand_dims(self._get_bias(), 0)
    xmw = xmw + b

    return self._split_gate(xmw)

  def _compute_new_c(self, state0: NestedMap, i_i: JTensor, i_g: JTensor,
                     f_g: JTensor) -> JTensor:
    asserts.not_none(i_g)
    forget_gate = jax.nn.sigmoid(f_g) * state0.c
    input_gate = jax.nn.sigmoid(i_g) * jnp.tanh(i_i)
    return forget_gate + input_gate

  def _gates_internal(self, state0: NestedMap, inputs: NestedMap, i_i: JTensor,
                      i_g: JTensor, f_g: JTensor, o_g: JTensor) -> NestedMap:
    p = self.params
    new_c = self._compute_new_c(state0, i_i, i_g, f_g)

    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = jax.lax.clamp(
          jnp.full(new_c.shape, -p.cell_value_cap), new_c,
          jnp.full(new_c.shape, p.cell_value_cap))
    if p.output_nonlinearity:
      new_m = jax.nn.sigmoid(o_g) * jnp.tanh(new_c)
    else:
      new_m = jax.nn.sigmoid(o_g) * new_c
    if p.num_hidden_nodes:
      w_proj = self.local_theta().w_proj
      new_m = new_m * w_proj

    # Apply Zoneout.
    return self._apply_zoneout(state0, inputs, new_c, new_m)

  def _apply_zoneout(self, state0: NestedMap, inputs: NestedMap, new_c: JTensor,
                     new_m: JTensor) -> NestedMap:
    """Apply Zoneout and returns the updated states."""
    p = self.params

    if p.zo_prob > 0.0:
      c_random_uniform = jax.random.uniform(base_layer.next_prng_key(),
                                            new_c.shape)
      m_random_uniform = jax.random.uniform(base_layer.next_prng_key(),
                                            new_m.shape)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._zoneout_internal(state0.c, new_c, inputs.padding, p.zo_prob,
                                   self.do_eval, c_random_uniform)
    new_m = self._zoneout_internal(state0.m, new_m, inputs.padding, p.zo_prob,
                                   self.do_eval, m_random_uniform)

    return NestedMap(m=new_m, c=new_c)


class CIFGLSTMCellSimple(LSTMCellSimple):
  """CIFG variant LSTM which couple the input and output gate."""

  @property
  def num_gates(self) -> int:
    return 3

  def get_adjustment(self) -> JTensor:
    adjustment = jnp.ones([3, self.hidden_size]) * jnp.expand_dims(
        jnp.array([0., self.params.forget_gate_bias, 0.]), axis=1)
    adjustment = jnp.reshape(adjustment, [self.num_gates * self.hidden_size])
    return adjustment

  def _split_gate(self,
                  xmw: JTensor) -> Tuple[JTensor, JTensor, JTensor, JTensor]:
    div = xmw.shape[1] // self.num_gates
    return xmw[:, :div], None, xmw[:, div:2 * div], xmw[:, 2 * div:]  # pytype: disable=bad-return-type  # jax-ndarray

  def _compute_new_c(self, state0: NestedMap, i_i: JTensor, i_g: JTensor,
                     f_g: JTensor) -> JTensor:
    asserts.none(i_g)
    forget_gate = jax.nn.sigmoid(f_g) * state0.c

    tanh_i_i = jnp.tanh(i_i)
    input_gate = tanh_i_i - tanh_i_i * jax.nn.sigmoid(f_g)
    return forget_gate + input_gate
