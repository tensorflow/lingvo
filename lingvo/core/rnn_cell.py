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
"""RNN cells (e.g., LSTM, GRU) that the Lingvo model uses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

from lingvo.core import base_layer
from lingvo.core import cudnn_rnn_utils
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import summary_utils


def _HistogramSummary(p, name, v):
  """Adds a histogram summary for 'v' into the default tf graph."""
  summary_utils.histogram(name, tf.cast(v, tf.float32))


RNN_CELL_WT = 'rnn_cell_weight_variable'


class RNNCell(quant_utils.QuantizableLayer):
  # pylint: disable=line-too-long
  """RNN cells.

  RNNCell represents recurrent state in a `.NestedMap`.

  `zero_state(batch_size)` returns the initial state, which is defined
  by each subclass. From the state, each subclass defines `GetOutput()`
  to extract the output tensor.

  `RNNCell.FProp` defines the forward function::

      (theta, state0, inputs) -> state1, extras

  All arguments and return values are `.NestedMap`. Each subclass defines
  what fields these `.NestedMap` are expected to have. `extras` is a
  `.NestedMap` containing some intermediate results `FProp` computes to
  facilitate the backprop.

  `zero_state(batch_size)`, `state0` and `state1` are all compatible
  `.NestedMap` (see `.NestedMap.IsCompatible`).
  I.e., they have the same keys recursively. Furthermore, the corresponding
  tensors in these `.NestedMap` have the same shape and dtype.
  """
  # pylint: enable=line-too-long

  @classmethod
  def Params(cls):
    p = super(RNNCell, cls).Params()
    p.Define('inputs_arity', 1,
             'number of tensors expected for the inputs.act to FProp.')
    p.Define('num_input_nodes', 0, 'Number of input nodes.')
    p.Define(
        'num_output_nodes', 0,
        'Number of output nodes. If num_hidden_nodes is 0, also used as '
        'cell size.')
    p.Define(
        'reset_cell_state', False,
        ('Set True to support resetting cell state in scenarios where multiple '
         'inputs are packed into a single training example. The RNN layer '
         'should provide reset_mask inputs in addition to act and padding if '
         'this flag is set.'))
    p.Define(
        'zero_state_init_params', py_utils.DefaultRNNCellStateInit(),
        'Parameters that define how the initial state values are set '
        'for each cell. Must be one of the static functions defined in '
        'py_utils.RNNCellStateInit.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes RnnCell."""
    super(RNNCell, self).__init__(params)
    assert not self.params.vn.per_step_vn, (
        'We do not support per step VN in RNN cells.')

  def _VariableCollections(self):
    return [RNN_CELL_WT, '%s_vars' % (self.__class__.__name__)]

  def zero_state(self, batch_size):
    """Returns the initial state given the batch size."""
    raise NotImplementedError('Abstract method')

  def GetOutput(self, state):
    """Returns the output value given the current state."""
    raise NotImplementedError('Abstract method')

  def batch_size(self, inputs):
    """Given the inputs, returns the batch size."""
    raise NotImplementedError('Abstract method')

  def FProp(self, theta, state0, inputs):
    """Forward function.

    The default implementation here assumes the cell forward
    function is composed of two functions::

        _Gates(_Mix(theta, state0, inputs), theta, state0, inputs)

    The result of `_Mix` is stashed in `extras` to facilitate backprop.

    `_ResetState` is optionally applied if `reset_cell_state` is True. The RNN
    layer should provide `reset_mask` inputs in addition to other inputs.
    `reset_mask` inputs are expected to be 0 at timesteps where state0 should be
    reset to default (zeros) before running `_Mix()` and `_Gates()`, and 1
    otherwise. This is meant to support use cases like packed inputs, where
    multiple samples are fed in a single input example sequence, and need to be
    masked from each other. For example, if the two examples packed together
    are ['good', 'day'] -> ['guten-tag'] and ['thanks'] -> ['danke']
    to produce ['good', 'day', 'thanks'] -> ['guten-tag', 'danke'], the
    source reset_masks would be [1, 1, 0] and target reset masks would be
    [1, 0]. These ids are meant to enable masking computations for
    different examples from each other.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      state0: The previous recurrent state. A `.NestedMap`.
      inputs: The inputs to the cell. A `.NestedMap`.

    Returns:
      A tuple (state1, extras).
      - state1: The next recurrent state. A `.NestedMap`.
      - extras: Intermediate results to faciliate backprop. A `.NestedMap`.
    """
    assert isinstance(inputs.act, list)
    assert self.params.inputs_arity == len(inputs.act)
    if self.params.reset_cell_state:
      state0_modified = self._ResetState(state0.DeepCopy(), inputs)
    else:
      state0_modified = state0
    xmw = self._Mix(theta, state0_modified, inputs)
    state1 = self._Gates(xmw, theta, state0_modified, inputs)
    return state1, py_utils.NestedMap()

  def _ZoneOut(self,
               prev_v,
               cur_v,
               padding_v,
               zo_prob,
               is_eval,
               random_uniform,
               qt=None,
               qdomain=''):
    """Apply ZoneOut regularlization to cur_v.

    Implements ZoneOut regularization as described in
    https://arxiv.org/abs/1606.01305

    Args:
      prev_v: A tensor, values from the previous timestep.
      cur_v: A tensor, values from the current timestep.
      padding_v: A tensor, the paddings vector for the cur timestep.
      zo_prob: A float, probability at which to apply ZoneOut regularization.
      is_eval: A bool, whether or not in eval mode.
      random_uniform: a tensor of random uniform numbers. This can be None if
        zo_prob=0.0
      qt: A string, name of the qtensor for zone out math.
      qdomain: A string, name of the qdomain for quantized zone out math.

    Returns:
      cur_v after ZoneOut regularization has been applied.
    """
    prev_v = tf.convert_to_tensor(prev_v)
    cur_v = tf.convert_to_tensor(cur_v)
    padding_v = tf.convert_to_tensor(padding_v)
    if zo_prob == 0.0:
      # Special case for when ZoneOut is not enabled.
      return py_utils.ApplyPadding(padding_v, cur_v, prev_v)

    if is_eval:
      # We take expectation in the eval mode.
      #
      fns = self.fns
      # This quantized mixed operation should probably occur as fused kernel to
      # avoid quantized-math rounding errors. Current accuracy has not been
      # verified.
      prev_weight = self.QWeight(zo_prob, domain=qdomain)
      new_weight = self.QWeight(1.0 - prev_weight, domain=qdomain)
      if qt is None:
        mix_prev = tf.multiply(tf.fill(tf.shape(prev_v), prev_weight), prev_v)
        mix_curr = tf.multiply(tf.fill(tf.shape(cur_v), new_weight), cur_v)
        mix = tf.add(mix_prev, mix_curr)
      else:
        mix_prev = fns.qmultiply(
            self.QWeight(
                tf.fill(tf.shape(prev_v), prev_weight), domain=qdomain),
            prev_v,
            qt=qt)
        mix_curr = fns.qmultiply(
            self.QWeight(tf.fill(tf.shape(cur_v), new_weight), domain=qdomain),
            cur_v,
            qt=qt)
        mix = fns.qadd(mix_prev, mix_curr, qt=qt)

      # If padding_v is 1, it always carries over the previous state.
      return py_utils.ApplyPadding(padding_v, mix, prev_v)
    else:
      assert random_uniform is not None
      random_uniform = py_utils.HasShape(random_uniform, tf.shape(prev_v))
      zo_p = tf.cast(random_uniform < zo_prob, padding_v.dtype)
      zo_p += padding_v
      # If padding_v is 1, we always carry over the previous state.
      zo_p = tf.minimum(zo_p, 1.0)
      zo_p = tf.stop_gradient(zo_p)
      return py_utils.ApplyPadding(zo_p, cur_v, prev_v)


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
  def Params(cls):
    p = super(LSTMCellSimple, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define('cell_value_cap', 10.0, 'LSTM cell values are capped to be within '
             ' [-cell_value_cap, +cell_value_cap] if the value is not None. '
             'It can be a scalar, a scalar tensor or None. When set to None, '
             'no capping is applied.')
    p.Define('forget_gate_bias', 0.0, 'Bias to apply to the forget gate.')
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('enable_lstm_bias', True, 'Enable the LSTM Cell bias.')
    p.Define(
        'couple_input_forget_gates', False,
        'Whether to couple the input and forget gates. Just like '
        'tf.contrib.rnn.CoupledInputForgetGateLSTMCell')
    p.Define('apply_pruning', False, 'Whether to prune the weights while '
             'training')
    p.Define('bias_init', py_utils.WeightInit.Constant(0.0),
             'Initialization parameters for bias')

    # Non-default quantization behaviour.
    p.qdomain.Define('weight', None, 'Quantization for the weights')
    p.qdomain.Define('c_state', None, 'Quantization for the c-state.')
    p.qdomain.Define('m_state', None, 'Quantization for the m-state.')
    p.qdomain.Define('fullyconnected', None,
                     'Quantization for fully connected node.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LSTMCellSimple."""
    super(LSTMCellSimple, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert isinstance(p.cell_value_cap,
                      (int, float)) or p.cell_value_cap is None

    assert p.cell_value_cap is None or p.qdomain.default is None
    self.TrackQTensor(
        'zero_m',
        'm_output',
        'm_output_projection',
        'm_zoneout',
        domain='m_state')
    self.TrackQTensor(
        'zero_c',
        'mixed',
        'c_couple_invert',
        'c_input_gate',
        'c_forget_gate',
        'c_output_gate',
        'c_zoneout',
        domain='c_state')
    self.TrackQTensor('add_bias', domain='fullyconnected')

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[
              p.num_input_nodes + self.output_size,
              self.num_gates * self.hidden_size
          ],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      if p.apply_pruning:
        mask_pc = py_utils.WeightParams(wm_pc.shape,
                                        py_utils.WeightInit.Constant(1.0),
                                        p.dtype)
        threshold_pc = py_utils.WeightParams([],
                                             py_utils.WeightInit.Constant(0.0),
                                             tf.float32)
        self.CreateVariable('mask', mask_pc, theta_fn=None, trainable=False)
        self.CreateVariable(
            'threshold', threshold_pc, theta_fn=None, trainable=False)

        def MaskWeightFn(weight):
          return tf.multiply(
              self.AddGlobalVN(weight), self.vars.mask, 'masked_weights')

        self.CreateVariable('wm', wm_pc, theta_fn=MaskWeightFn)
        py_utils.AddToPruningCollections(self.vars.wm, self.vars.mask,
                                         self.vars.threshold)

      else:
        self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      if p.num_hidden_nodes:
        w_proj = py_utils.WeightParams(
            shape=[self.hidden_size, self.output_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('w_proj', w_proj, self.AddGlobalVN)

      if p.enable_lstm_bias:
        bias_pc = py_utils.WeightParams(
            shape=[self.num_gates * self.hidden_size],
            init=p.bias_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('b', bias_pc, self.AddGlobalVN)

      # Collect some stats.
      w = self.vars.wm
      if p.couple_input_forget_gates:
        i_i, f_g, o_g = tf.split(
            value=w, num_or_size_splits=self.num_gates, axis=1)
      else:
        i_i, i_g, f_g, o_g = tf.split(
            value=w, num_or_size_splits=self.num_gates, axis=1)
        _HistogramSummary(p, scope.name + '/wm_i_g', i_g)
      _HistogramSummary(p, scope.name + '/wm_i_i', i_i)
      _HistogramSummary(p, scope.name + '/wm_f_g', f_g)
      _HistogramSummary(p, scope.name + '/wm_o_g', o_g)

      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  @property
  def num_gates(self):
    return 3 if self.params.couple_input_forget_gates else 4

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    if p.is_inference:
      zero_m = self.QTensor('zero_m', zero_m)
      zero_c = self.QTensor('zero_c', zero_c)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def GetOutput(self, state):
    return state.m

  def _GetBias(self, theta):
    """Gets the bias vector to add.

    Includes adjustments like forget_gate_bias. Use this instead of the 'b'
    variable directly as including adjustments in this way allows const-prop
    to eliminate the adjustments at inference time.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.

    Returns:
      The bias vector.
    """
    p = self.params
    if p.enable_lstm_bias:
      b = theta.b
    else:
      b = tf.zeros([self.num_gates * self.hidden_size], dtype=p.dtype)
    if p.forget_gate_bias != 0.0:
      # Apply the forget gate bias directly to the bias vector.
      if not p.couple_input_forget_gates:
        # Normal 4 gate bias (i_i, i_g, f_g, o_g).
        adjustment = (
            tf.ones([4, self.hidden_size], dtype=p.dtype) * tf.expand_dims(
                tf.constant([0., 0., p.forget_gate_bias, 0.], dtype=p.dtype),
                axis=1))
      else:
        # 3 gates with coupled input/forget (i_i, f_g, o_g).
        adjustment = (
            tf.ones([3, self.hidden_size], dtype=p.dtype) * tf.expand_dims(
                tf.constant([0., p.forget_gate_bias, 0.], dtype=p.dtype),
                axis=1))
      adjustment = tf.reshape(adjustment, [self.num_gates * self.hidden_size])
      b += adjustment

    return b

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    wm = self.QWeight(theta.wm)
    concat = tf.concat(inputs.act + [state0.m], 1)
    # Defer quantization until after adding in the bias to support fusing
    # matmul and bias add during inference.
    return tf.matmul(concat, wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    fns = self.fns
    b = self.QWeight(tf.expand_dims(self._GetBias(theta), 0), domain='fc')
    xmw = fns.qadd(xmw, b, qt='add_bias')

    if not p.couple_input_forget_gates:
      i_i, i_g, f_g, o_g = tf.split(value=xmw, num_or_size_splits=4, axis=1)
      forget_gate = fns.qmultiply(tf.sigmoid(f_g), state0.c, qt='c_input_gate')
      # Sigmoid / tanh calls are not quantized under the assumption they share
      # the range with c_input_gate and c_forget_gate.
      input_gate = fns.qmultiply(
          tf.sigmoid(i_g), tf.tanh(i_i), qt='c_forget_gate')
      new_c = fns.qadd(forget_gate, input_gate, qt='c_output_gate')
    else:
      i_i, f_g, o_g = tf.split(value=xmw, num_or_size_splits=3, axis=1)
      # Sigmoid / tanh calls are not quantized under the assumption they share
      # the range with c_input_gate and c_forget_gate.
      forget_gate = fns.qmultiply(tf.sigmoid(f_g), state0.c, qt='c_input_gate')

      # input_gate = tanh(i_i) - tanh(i_i) * tf.sigmoid(f_g)
      # equivalent to (but more stable in fixed point):
      # (1.0 - sigmoid(f_g)) * tanh(i_i)
      tanh_i_i = tf.tanh(i_i)
      input_gate = fns.qsubtract(
          tanh_i_i,
          fns.qmultiply(tanh_i_i, tf.sigmoid(f_g), qt='c_couple_invert'),
          qt='c_forget_gate')

      new_c = fns.qadd(forget_gate, input_gate, qt='c_output_gate')
    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)
    if p.output_nonlinearity:
      new_m = fns.qmultiply(tf.sigmoid(o_g), tf.tanh(new_c), qt='m_output')
    else:
      new_m = fns.qmultiply(tf.sigmoid(o_g), new_c, qt='m_output')
    if p.num_hidden_nodes:
      w_proj = self.QWeight(theta.w_proj, domain='m_state')
      new_m = fns.qmatmul(new_m, w_proj, qt='m_output_projection')

    # Apply Zoneout.
    return self._ApplyZoneOut(state0, inputs, new_c, new_m)

  def _ApplyZoneOut(self, state0, inputs, new_c, new_m):
    """Apply Zoneout and returns the updated states."""
    p = self.params
    if p.zo_prob > 0.0:
      assert not py_utils.use_tpu(), (
          'LSTMCellSimple does not support zoneout on TPU. Switch to '
          'LSTMCellSimpleDeterministic instead.')
      c_random_uniform = tf.random_uniform(tf.shape(new_c), seed=p.random_seed)
      m_random_uniform = tf.random_uniform(tf.shape(new_m), seed=p.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(
        state0.c,
        new_c,
        self.QRPadding(inputs.padding),
        p.zo_prob,
        p.is_eval,
        c_random_uniform,
        qt='c_zoneout',
        qdomain='c_state')
    new_m = self._ZoneOut(
        state0.m,
        new_m,
        self.QRPadding(inputs.padding),
        p.zo_prob,
        p.is_eval,
        m_random_uniform,
        qt='m_zoneout',
        qdomain='m_state')
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LSTMCellGrouped(RNNCell):
  """LSTM cell with groups.

  Grouping: based on "Factorization tricks for LSTM networks".
  https://arxiv.org/abs/1703.10722.

  Shuffling: adapted from "ShuffleNet: An Extremely Efficient Convolutional
  Neural Network for Mobile Devices". https://arxiv.org/abs/1707.01083.

  theta:

  - groups: a list of child LSTM cells.

  state:

    A `.NestedMap` containing 'groups', a list of `.NestedMap`, each with:

    - m: the lstm output. [batch, cell_nodes // num_groups]
    - c: the lstm cell state. [batch, cell_nodes // num_groups]

  inputs:

  -  act: a list of input activations. [batch, input_nodes]
  -  padding: the padding. [batch, 1].
  -  reset_mask: optional 0/1 float input to support packed input training.
     Shape [batch, 1]
  """

  @classmethod
  def Params(cls, child_cell_cls=LSTMCellSimple):
    p = super(LSTMCellGrouped, cls).Params()
    p.Define('child_lstm_tpl', child_cell_cls.Params(),
             'Template of child LSTM cells.')
    p.Define('num_groups', 0, 'Number of LSTM cell groups.')
    p.Define('num_shuffle_shards', 1,
             'If > 1, number of shards for cross-group shuffling.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LSTMCellGrouped."""
    super(LSTMCellGrouped, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert p.num_input_nodes > 0
    assert p.num_output_nodes > 0
    assert p.num_groups > 0
    assert p.num_shuffle_shards > 0
    assert p.num_input_nodes % p.num_groups == 0
    assert p.num_output_nodes % (p.num_shuffle_shards * p.num_groups) == 0

    with tf.variable_scope(p.name):
      child_params = []
      for i in range(p.num_groups):
        child_p = self.params.child_lstm_tpl.Copy()
        child_p.name = 'group_%d' % i
        assert child_p.num_input_nodes == 0
        assert child_p.num_output_nodes == 0
        child_p.num_input_nodes = p.num_input_nodes // p.num_groups
        child_p.num_output_nodes = p.num_output_nodes // p.num_groups
        child_p.reset_cell_state = p.reset_cell_state
        child_params.append(child_p)
      self.CreateChildren('groups', child_params)

  def batch_size(self, inputs):
    return self.groups[0].batch_size(inputs)

  def zero_state(self, batch_size):
    return py_utils.NestedMap(
        groups=[child.zero_state(batch_size) for child in self.groups])

  # TODO(rpang): avoid split and concat between layers with the same number of
  # groups, if necessary.
  def GetOutput(self, state):
    p = self.params
    # Assuming that GetOutput() is stateless, we can just use the first child.
    outputs = [
        child.GetOutput(child_state)
        for child, child_state in zip(self.groups, state.groups)
    ]
    split_output = []
    # Split each output to num_shuffle_shards.
    for output in outputs:
      split_output.extend(
          py_utils.SplitRecursively(output, p.num_shuffle_shards))
    # Shuffle and concatenate shards.
    return py_utils.ConcatRecursively(self._ShuffleShards(split_output))

  def FProp(self, theta, state0, inputs):
    """Forward function.

    Splits state0 and inputs into N groups (N=num_groups), runs child
    LSTM cells on each group, and concatenates the outputs with optional
    shuffling between groups.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      state0: The previous recurrent state. A `.NestedMap`.
      inputs: The inputs to the cell. A `.NestedMap`.

    Returns:
      A tuple (state1, extras).
      - state1: The next recurrent state. A list.
      - extras: An empty `.NestedMap`.
    """
    p = self.params
    split_inputs_act = py_utils.SplitRecursively(inputs.act, p.num_groups)
    state1 = py_utils.NestedMap(groups=[])
    for child, child_theta, child_state0, child_inputs_act in zip(
        self.groups, theta.groups, state0.groups, split_inputs_act):
      child_inputs = inputs.copy()
      child_inputs.act = child_inputs_act
      child_state1, child_extras = child.FProp(child_theta, child_state0,
                                               child_inputs)
      assert not child_extras
      state1.groups.append(child_state1)
    return state1, py_utils.NestedMap()

  def _ShuffleShards(self, shards):
    """Shuffles shards across groups.

    Args:
      shards: a list of length num_shuffle_shards (S) * num_groups (G). The
        first S shards belong to group 0, the next S shards belong to group 1,
        etc.

    Returns:
      A shuffled list of shards such that shards from each input group are
      scattered across output groups.

      For example, if we have 3 groups, each with 4 shards:

      | Group 0: 0_0, 0_1, 0_2, 0_3
      | Group 1: 1_0, 1_1, 1_2, 1_3
      | Group 2: 2_0, 2_1, 2_2, 2_3

      The shuffled output will be:

      | Group 0: 0_0, 1_1, 2_2, 0_3
      | Group 1: 1_0, 2_1, 0_2, 1_3
      | Group 2: 2_0, 0_1, 1_2, 2_3
    """
    p = self.params
    assert len(shards) == (p.num_shuffle_shards * p.num_groups)
    shuffled_shards = []
    for group_i in range(p.num_groups):
      for shuffle_i in range(p.num_shuffle_shards):
        shuffled_shards.append(
            shards[((group_i + shuffle_i) % p.num_groups) * p.num_shuffle_shards
                   + shuffle_i])
    return shuffled_shards


# TODO(yonghui): Merge this cell with the LSTMCellSimple cell.
class LSTMCellSimpleDeterministic(LSTMCellSimple):
  """Same as LSTMCellSimple, except this cell is completely deterministic."""

  @classmethod
  def Params(cls):
    p = super(LSTMCellSimpleDeterministic, cls).Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LSTMCell."""
    super(LSTMCellSimpleDeterministic, self).__init__(params)
    p = self.params
    assert p.name
    with tf.variable_scope(p.name):
      _, self._step_counter = py_utils.CreateVariable(
          name='lstm_step_counter',
          params=py_utils.WeightParams([], py_utils.WeightInit.Constant(0),
                                       tf.int64),
          trainable=False)
      vname = self._step_counter.name
      self._prng_seed = tf.constant(
          py_utils.GenerateSeedFromName(vname), dtype=tf.int64)
      if p.random_seed:
        self._prng_seed += p.random_seed

  def zero_state(self, batch_size):
    p = self.params
    zero_m = tf.zeros((batch_size, self.output_size),
                      dtype=py_utils.FPropDtype(p))
    zero_c = tf.zeros((batch_size, self.hidden_size),
                      dtype=py_utils.FPropDtype(p))
    if p.is_inference:
      zero_m = self.QTensor('zero_m', zero_m)
      zero_c = self.QTensor('zero_c', zero_c)

    # The first random seed changes for different layers and training steps.
    random_seed1 = self._prng_seed + self._step_counter
    # The second random seed changes for different unroll time steps.
    random_seed2 = tf.constant(0, dtype=tf.int64)
    random_seeds = tf.stack([random_seed1, random_seed2])
    return py_utils.NestedMap(m=zero_m, c=zero_c, r=random_seeds)

  def _ApplyZoneOut(self, state0, inputs, new_c, new_m):
    """Apply Zoneout and returns the updated states."""
    p = self.params
    random_seed1 = state0.r[0]
    random_seed2 = state0.r[1]
    if p.zo_prob > 0.0:
      # Note(yonghui): It seems that currently TF only supports int64 as the
      # random seeds, however, TPU will support int32 as the seed.
      # TODO(yonghui): Fix me for TPU.
      c_seed = tf.stack([random_seed1, 2 * random_seed2])
      m_seed = tf.stack([random_seed1, 2 * random_seed2 + 1])
      if py_utils.use_tpu():
        c_random_uniform = tf.contrib.stateless.stateless_random_uniform(
            py_utils.GetShape(new_c, 2), tf.cast(c_seed, tf.int32))
        m_random_uniform = tf.contrib.stateless.stateless_random_uniform(
            py_utils.GetShape(new_m, 2), tf.cast(m_seed, tf.int32))
      else:
        c_random_uniform = tf.contrib.stateless.stateless_random_uniform(
            py_utils.GetShape(new_c, 2), c_seed)
        m_random_uniform = tf.contrib.stateless.stateless_random_uniform(
            py_utils.GetShape(new_m, 2), m_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(
        state0.c,
        new_c,
        inputs.padding,
        p.zo_prob,
        p.is_eval,
        c_random_uniform,
        qt='zero_c',
        qdomain='c_state')
    new_m = self._ZoneOut(
        state0.m,
        new_m,
        inputs.padding,
        p.zo_prob,
        p.is_eval,
        m_random_uniform,
        qt='zero_m',
        qdomain='m_state')
    # TODO(yonghui): stop the proliferation of tf.stop_gradient
    r = tf.stop_gradient(tf.stack([random_seed1, random_seed2 + 1]))
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    r.set_shape(state0.r.shape)
    return py_utils.NestedMap(m=new_m, c=new_c, r=r)

  def PostTrainingStepUpdate(self, global_step):
    """Update the global_step value."""
    p = self.params
    with tf.name_scope(p.name):
      summary_utils.scalar('step_counter', self._step_counter)
    return self._step_counter.assign(tf.cast(global_step, tf.int64))


class QuantizedLSTMCell(RNNCell):
  """Simplified LSTM cell used for quantized training.

  There is no forget_gate_bias, no output_nonlinearity and no bias. Right now
  only clipping is performed.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - cap: the cell value cap.

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    [batch, 1]
  """

  @classmethod
  def Params(cls):
    p = super(QuantizedLSTMCell, cls).Params()
    p.Define('cc_schedule', quant_utils.LinearClippingCapSchedule.Params(),
             'Clipping cap schedule.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes QuantizedLSTMCell."""
    super(QuantizedLSTMCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[
              p.num_input_nodes + p.num_output_nodes, 4 * p.num_output_nodes
          ],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      self.CreateChild('cc_schedule', p.cc_schedule)

      # Collect some stats
      i_i, i_g, f_g, o_g = tf.split(
          value=self.vars.wm, num_or_size_splits=4, axis=1)
      _HistogramSummary(p, scope.name + '/wm_i_i', i_i)
      _HistogramSummary(p, scope.name + '/wm_i_g', i_g)
      _HistogramSummary(p, scope.name + '/wm_f_g', f_g)
      _HistogramSummary(p, scope.name + '/wm_o_g', o_g)

      self._timestep = -1

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, p.num_output_nodes),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    zero_c = py_utils.InitRNNCellState((batch_size, p.num_output_nodes),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    return py_utils.Matmul(tf.concat(inputs.act + [state0.m], 1), theta.wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    i_i, i_g, f_g, o_g = tf.split(value=xmw, num_or_size_splits=4, axis=1)

    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)
    new_c = self.cc_schedule.ApplyClipping(theta.cc_schedule, new_c)
    new_m = tf.sigmoid(o_g) * new_c

    # Respect padding.
    new_m = state0.m * inputs.padding + new_m * (1 - inputs.padding)
    new_c = state0.c * inputs.padding + new_c * (1 - inputs.padding)

    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LSTMCellCuDNNCompliant(RNNCell):
  """LSTMCell compliant with variables with CuDNN-LSTM layout.

  theta:

  - wb: the cudnn LSTM weight.

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
  def Params(cls):
    p = super(LSTMCellCuDNNCompliant, cls).Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LSTMCellCuDNNCompliant, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      cudnn_init_helper = cudnn_rnn_utils.CuDNNLSTMInitializer(
          p.num_input_nodes, p.num_output_nodes)
      if not p.is_eval:
        wb_pc = py_utils.WeightParams(
            shape=None,
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'wb',
            wb_pc,
            self.AddGlobalVN,
            init_wrapper=cudnn_init_helper.InitOpaqueParams)
      else:
        # Run eval mode on CPU, use inferred static shape since dynamic shape
        # requires running a GPU kernel.
        # Also uses simplified initialization approach, since the vars would
        # be restored from checkpoints and initialization values don't matter.
        # TODO(jamesqin): save cudnn opaque params in canonical format.
        wb_pc = py_utils.WeightParams(
            shape=[cudnn_init_helper.weight_size + cudnn_init_helper.bias_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('wb', wb_pc, self.AddGlobalVN)

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self._params
    return py_utils.NestedMap(
        m=py_utils.InitRNNCellState([batch_size, p.num_output_nodes],
                                    init=p.zero_state_init_params,
                                    dtype=p.dtype),
        c=py_utils.InitRNNCellState([batch_size, p.num_output_nodes],
                                    init=p.zero_state_init_params,
                                    dtype=p.dtype))

  def GetOutput(self, state):
    return state.m

  def _WeightAndBias(self, theta):
    p = self.params
    return cudnn_rnn_utils.RecoverLSTMCellSimpleWeightsFromCuDNN(
        theta.wb, p.num_input_nodes, p.num_output_nodes,
        cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    wm, _ = self._WeightAndBias(theta)
    return py_utils.Matmul(tf.concat(inputs.act + [state0.m], 1), wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    _, b = self._WeightAndBias(theta)
    i_i, i_g, f_g, o_g = tf.split(
        value=xmw + tf.expand_dims(b, 0), num_or_size_splits=4, axis=1)
    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)
    new_m = tf.sigmoid(o_g) * tf.tanh(new_c)

    # Technically this is not the same as CuDNN impl which does not support
    # padding for performance reasons.
    # Yet padding is still done correctly here so that the backward direction of
    # BidirectionalCuDNNLSTM is done correctly in eval mode, when CuDNNLSTM
    # inherits DRNN.
    new_m = state0.m * inputs.padding + new_m * (1 - inputs.padding)
    new_c = state0.c * inputs.padding + new_c * (1 - inputs.padding)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LayerNormalizedLSTMCell(RNNCell):
  """DEPRECATED: use LayerNormalizedLSTMCellSimple instead.

  Simple LSTM cell with layer normalization.

  Implements normalization scheme as described in
  https://arxiv.org/pdf/1607.06450.pdf

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
  def Params(cls):
    p = super(LayerNormalizedLSTMCell, cls).Params()
    p.Define(
        'cell_value_cap', 10.0, 'LSTM cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap]. This can be a'
        ' scalar or a scalar tensor.')
    p.Define('forget_gate_bias', 0.0, 'Bias to apply to the forget gate.')
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm.')
    return p

  @tf.contrib.framework.deprecated(
      date=None,
      instructions='New models should use LayerNormalizedLSTMCellSimple.')
  @base_layer.initializer
  def __init__(self, params):
    """Initializes LayerNormalizedLSTMCell."""
    super(LayerNormalizedLSTMCell, self).__init__(params)
    params = self.params
    if not isinstance(params.cell_value_cap, (int, float)):
      raise ValueError('Cell value cap must of type int or float!')

    with tf.variable_scope(params.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[
              params.num_input_nodes + params.num_output_nodes,
              4 * params.num_output_nodes
          ],
          init=params.params_init,
          dtype=params.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)
      # This bias variable actually packs the initial lstm bias variables as
      # well as various layer norm scale and bias variables. We pack multiple
      # variables into one so that we can still unroll this lstm using the FRNN
      # layer defined in layers.py.
      bias_pc = py_utils.WeightParams(
          shape=[4 * params.num_output_nodes + 4 * params.num_output_nodes],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=params.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

      if params.cc_schedule:
        self.CreateChild('cc_schedule', params.cc_schedule)

      # Collect some stats
      i_i, i_g, f_g, o_g = tf.split(
          value=self.vars.wm, num_or_size_splits=4, axis=1)
      _HistogramSummary(params, scope.name + '/wm_i_i', i_i)
      _HistogramSummary(params, scope.name + '/wm_i_g', i_g)
      _HistogramSummary(params, scope.name + '/wm_f_g', f_g)
      _HistogramSummary(params, scope.name + '/wm_o_g', o_g)
      # TODO(yonghui): Add more summaries here.

      self._timestep = -1

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_output_nodes

  def zero_state(self, batch_size):
    p = self.params
    return py_utils.NestedMap(
        m=py_utils.InitRNNCellState([batch_size, p.num_output_nodes],
                                    init=p.zero_state_init_params,
                                    dtype=p.dtype),
        c=py_utils.InitRNNCellState([batch_size, p.num_output_nodes],
                                    init=p.zero_state_init_params,
                                    dtype=p.dtype))

  def GetOutput(self, state):
    return state.m

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def _Mix(self, theta, state0, inputs):
    if not isinstance(inputs.act, list):
      raise ValueError('Input activations must be of list type!')
    return py_utils.Matmul(tf.concat(inputs.act + [state0.m], 1), theta.wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    # Unpack the variables (weight and bias) into individual variables.
    params = self.params

    def BiasSlice(dim, num_dims, start_ind):
      s = []
      for i in range(num_dims):
        s.append(theta.b[start_ind + i * dim:start_ind + (i + 1) * dim])
      start_ind += dim * num_dims
      return s, start_ind

    # Unpack the bias variable.
    slice_start = 0
    bias_lstm, slice_start = BiasSlice(params.num_output_nodes, 4, slice_start)
    ln_scale, slice_start = BiasSlice(params.num_output_nodes, 4, slice_start)
    assert slice_start == 8 * params.num_output_nodes

    def _LayerNorm(x, last_dim):
      """Normalize the last dimension."""
      if params.use_fused_layernorm:
        counts, means_ss, variance_ss, _, = tf.nn.sufficient_statistics(
            x, axes=[last_dim], keep_dims=True)
        mean, variance = tf.nn.normalize_moments(counts, means_ss, variance_ss,
                                                 None)
      else:
        mean = tf.reduce_mean(x, axis=[last_dim], keepdims=True)
        variance = tf.reduce_mean(
            tf.square(x - mean), axis=[last_dim], keepdims=True)
      return (x - mean) * tf.rsqrt(variance + params.layer_norm_epsilon)

    state_split = tf.split(xmw, num_or_size_splits=4, axis=1)
    for i in range(4):
      state_split[i] = _LayerNorm(state_split[i], 1) * tf.expand_dims(
          ln_scale[i] + 1.0, 0) + tf.expand_dims(bias_lstm[i], 0)

    i_i, i_g, f_g, o_g = state_split

    if params.forget_gate_bias != 0.0:
      f_g += params.forget_gate_bias
    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)

    # Clip the cell states to reasonable value.
    if params.cc_schedule:
      cap = self.cc_schedule.GetState(theta.cc_schedule)
    else:
      cap = params.cell_value_cap
    new_c = py_utils.clip_by_value(new_c, -cap, cap)

    if params.output_nonlinearity:
      new_m = tf.sigmoid(o_g) * tf.tanh(new_c)
    else:
      new_m = tf.sigmoid(o_g) * new_c

    if params.zo_prob > 0.0:
      c_random_uniform = tf.random_uniform(
          tf.shape(new_c), seed=params.random_seed)
      m_random_uniform = tf.random_uniform(
          tf.shape(new_m), seed=params.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(state0.c, new_c, inputs.padding, params.zo_prob,
                          params.is_eval, c_random_uniform)
    new_m = self._ZoneOut(state0.m, new_m, inputs.padding, params.zo_prob,
                          params.is_eval, m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LayerNormalizedLSTMCellSimple(LSTMCellSimple):
  """An implementation of layer normalized LSTM based on LSTMCellSimple.

  Implements normalization scheme as described in
  https://arxiv.org/pdf/1607.06450.pdf

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
  def Params(cls):
    p = super(LayerNormalizedLSTMCellSimple, cls).Params()
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LayerNormalizedLSTMCellSimple."""
    super(LayerNormalizedLSTMCellSimple, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      ln_scale_pc = py_utils.WeightParams(
          shape=[self.num_gates * self.hidden_size],
          init=py_utils.WeightInit.Constant(1.0),
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('ln_scale', ln_scale_pc, self.AddGlobalVN)

  def _Mix(self, theta, state0, inputs):
    xmw = super(LayerNormalizedLSTMCellSimple, self)._Mix(theta, state0, inputs)
    p = self.params

    # TODO(dehao): refactor the code to remove reshape and use fused layernorm.
    def _LayerNorm(x):
      """Applies layer normalization on the last dimension of 'x'.

      Args:
        x: activation tensor, where the last dimension represents channels.

      Returns:
        Layer normalized 'x', with the same shape as the input.
      """
      last_dim = tf.rank(x) - 1
      mean = tf.reduce_mean(x, axis=[last_dim], keepdims=True)
      variance = tf.reduce_mean(
          tf.square(x - mean), axis=[last_dim], keepdims=True)
      return (x - mean) * tf.rsqrt(variance + p.layer_norm_epsilon)

    def _PerGateLayerNorm(x, scale):
      """Applies per-gate layer normalization 'x'.

      Args:
        x: a tensor of shape [B, self.hidden_size * self.num_gates], containing
          'num_gates' activations concatenated along the last dimension.
        scale: per-channel scaling factor, of shape
          [self.hidden_size * self.num_gates].

      Returns:
        Per-gate layer normalized 'x', with the same shape as the input.
      """
      x_reshaped = tf.reshape(
          x, tf.stack([tf.shape(x)[0], self.num_gates, self.hidden_size]))
      x_norm = _LayerNorm(x_reshaped)
      return tf.reshape(x_norm, tf.shape(x)) * tf.expand_dims(scale, 0)

    return _PerGateLayerNorm(xmw, theta.ln_scale)


class LayerNormalizedLSTMCellLean(RNNCell):
  """A very lean layer normalized LSTM cell.

  This version is around 20% faster on TPU than LayerNormalizedLSTMCellSimple as
  it avoids certain reshape ops which are not free on TPU.

  Note, this version doesn't support all the options as implemented in
  LayerNormalizedLSTMCellSimple, like quantization, zoneout regularization and
  etc. Please use the other version if you even need those options. Another
  difference is that in this version, c_state is also being layer-normalized.

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
  def Params(cls):
    p = super(LayerNormalizedLSTMCellLean, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqrt against.')

    # TODO(yonghui): Get rid of the following two params.
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LayerNormalizedLSTMCellLean."""
    super(LayerNormalizedLSTMCellLean, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params

    assert p.output_nonlinearity
    assert p.zo_prob == 0.0

    with tf.variable_scope(p.name):
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes + self.output_size, 4 * self.hidden_size],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      if p.num_hidden_nodes:
        w_proj = py_utils.WeightParams(
            shape=[self.hidden_size, self.output_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('w_proj', w_proj, self.AddGlobalVN)

      ln_params = ['i_g', 'i_i', 'f_g', 'o_g', 'c']
      for ln_name in ln_params:
        pc = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('ln_scale_' + ln_name, pc, self.AddGlobalVN)
        self.CreateVariable('bias_' + ln_name, pc, self.AddGlobalVN)

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    mixed = tf.matmul(tf.concat(inputs.act + [state0.m], 1), theta.wm)
    return mixed

  def _LayerNormGate(self, theta, gate_name, x):
    """Applies layer normalization on the last dimension of 'x'.

    Args:
      theta: a NestedMap of layer params.
      gate_name: the name of the gate, e.g., 'i_i', 'f_g', 'c', etc.
      x: activation tensor, where the last dimension represents channels.

    Returns:
      Layer normalized 'x', with the same shape as the input.
    """
    p = self.params
    mean = tf.reduce_mean(x, axis=[1], keepdims=True)
    centered = x - mean
    variance = tf.reduce_mean(tf.square(centered), axis=[1], keepdims=True)
    normed = centered * tf.rsqrt(variance + p.layer_norm_epsilon)
    scale = theta['ln_scale_%s' % gate_name] + 1.0
    bias = theta['bias_%s' % gate_name]
    return normed * scale + bias

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    i_i, i_g, f_g, o_g = tf.split(value=xmw, num_or_size_splits=4, axis=1)
    i_i = self._LayerNormGate(theta, 'i_i', i_i)
    i_g = self._LayerNormGate(theta, 'i_g', i_g)
    f_g = self._LayerNormGate(theta, 'f_g', f_g)
    o_g = self._LayerNormGate(theta, 'o_g', o_g)
    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)
    new_c_normed = self._LayerNormGate(theta, 'c', new_c)
    new_m = tf.sigmoid(o_g) * tf.tanh(new_c_normed)

    if p.num_hidden_nodes:
      new_m = tf.matmul(new_m, theta.w_proj)

    # Now take care of padding.
    padding = inputs.padding
    new_m = py_utils.ApplyPadding(padding, new_m, state0.m)
    new_c = py_utils.ApplyPadding(padding, new_c, state0.c)

    return py_utils.NestedMap(m=new_m, c=new_c)


class DoubleProjectionLSTMCell(RNNCell):
  """A layer normalized LSTM cell that support input and output projections.

  Note, this version doesn't support all the options as implemented in
  LayerNormalizedLSTMCellSimple, like quantization, zoneout regularization,
  etc. Please use the other version if you need those options and do not need
  input projection.

  It also uses separate variables for weight matrices between gates
  ('wm_{i_i, i_g, f_g, o_g}') instead of a single variable ('wm'). This allows
  the initialization to use the default GeoMeanXavier().

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
  def Params(cls):
    p = super(DoubleProjectionLSTMCell, cls).Params()
    p.Define(
        'num_input_hidden_nodes', 0,
        'Project all inputs, include m, to a hidden vector this size before '
        'projecting to num_gates * |c|. Must be > 0.')
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqrt against.')
    p.params_init = py_utils.WeightInit.GeoMeanXavier()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(DoubleProjectionLSTMCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert p.num_input_hidden_nodes > 0
    assert p.num_hidden_nodes > 0

    with tf.variable_scope(p.name):

      def _WeightInit(shape):
        return py_utils.WeightParams(
            shape=shape,
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())

      self.CreateVariable(
          'w_input_proj',
          _WeightInit(
              [p.num_input_nodes + self.output_size, p.num_input_hidden_nodes]),
          self.AddGlobalVN)

      self.CreateVariable('w_output_proj',
                          _WeightInit([self.hidden_size, self.output_size]),
                          self.AddGlobalVN)

      for gate_name in self.gates:
        self.CreateVariable(
            'wm_%s' % gate_name,
            _WeightInit([p.num_input_hidden_nodes, self.hidden_size]),
            self.AddGlobalVN)

      for ln_name in self.gates + ['c']:
        pc = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('ln_scale_' + ln_name, pc, self.AddGlobalVN)
        self.CreateVariable('bias_' + ln_name, pc, self.AddGlobalVN)

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  @property
  def gates(self):
    return ['i_g', 'i_i', 'f_g', 'o_g']

  def zero_state(self, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    concat = tf.concat(inputs.act + [state0.m], 1)
    input_proj = tf.matmul(concat, theta.w_input_proj)
    gate_map = {}
    for gate_name in self.gates:
      g = tf.matmul(input_proj, theta.get('wm_%s' % gate_name))
      g = self._LayerNorm(g,
                          theta.get('ln_scale_%s' % gate_name) + 1.0,
                          theta.get('bias_%s' % gate_name))
      gate_map[gate_name] = g
    return gate_map

  def _LayerNorm(self, x, scale, bias):
    """Applies layer normalization on the last dimension of 'x'.

    Args:
      x: activation tensor, where the last dimension represents channels.
      scale: multiples to the noramlized results
      bias: additions to the noramlized results for biasing

    Returns:
      Layer normalized 'x', with the same shape as the input.
    """
    p = self.params
    mean = tf.reduce_mean(x, axis=[1], keepdims=True)
    centered = x - mean
    variance = tf.reduce_mean(tf.square(centered), axis=[1], keepdims=True)
    normed = centered * tf.rsqrt(variance + p.layer_norm_epsilon)
    return normed * scale + bias

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    new_c = tf.sigmoid(xmw['f_g']) * state0.c + tf.sigmoid(
        xmw['i_g']) * tf.tanh(xmw['i_i'])
    new_c_normed = self._LayerNorm(new_c, theta.ln_scale_c + 1.0, theta.bias_c)
    new_m = tf.sigmoid(xmw['o_g']) * tf.tanh(new_c_normed)
    new_m = tf.matmul(new_m, theta.w_output_proj)

    # Now take care of padding.
    padding = inputs.padding
    new_m = py_utils.ApplyPadding(padding, new_m, state0.m)
    new_c = py_utils.ApplyPadding(padding, new_c, state0.c)

    return py_utils.NestedMap(m=new_m, c=new_c)


class ConvLSTMCell(RNNCell):
  """Convolution LSTM cells.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the lstm output. cell_shape
  - c: the lstm cell state. cell_shape

  inputs:

  - act: a list of input activations. input_shape.
  - padding: the padding. [batch].
  """

  @classmethod
  def Params(cls):
    p = super(ConvLSTMCell, cls).Params()
    p.Define(
        'inputs_shape', [None, None, None, None],
        'The shape of the input. It should be a list/tuple of size four.'
        ' Elements are in the order of batch, height, width, channel.')
    p.Define(
        'cell_shape', [None, None, None, None],
        'The cell shape. It should be a list/tuple of size four.'
        ' Elements are in the order of batch, height, width, channel.'
        ' Height and width of cell_shape should match that of'
        ' inputs_shape.')
    p.Define(
        'filter_shape', [None, None],
        'Shape of the convolution filter. This should be a pair, in the'
        ' order height and width.')
    p.Define(
        'cell_value_cap', 10.0, 'LSTM cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap]. This can be a'
        ' scalar or a scalar tensor.')
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ConvLSTMCell."""
    assert isinstance(params, hyperparams.Params)
    super(ConvLSTMCell, self).__init__(params)

    p = self.params
    assert p.reset_cell_state is False, ('ConvLSTMCell currently doesnt '
                                         'support resetting cell state.')
    assert p.inputs_shape[1] == p.cell_shape[1]
    assert p.inputs_shape[2] == p.cell_shape[2]
    assert isinstance(p.cell_value_cap, (int, float))

    in_channels = p.inputs_shape[3] + p.cell_shape[3]
    out_channels = p.cell_shape[3]
    with tf.variable_scope(p.name):
      # Define weights.
      var_shape = [
          p.filter_shape[0], p.filter_shape[1], in_channels, 4 * out_channels
      ]
      wm_pc = py_utils.WeightParams(
          shape=var_shape,
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      bias_pc = py_utils.WeightParams(
          shape=[4 * out_channels],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    height = p.inputs_shape[1]
    width = p.inputs_shape[2]
    out_channels = p.cell_shape[3]
    return py_utils.NestedMap(
        m=py_utils.InitRNNCellState(
            tf.stack([batch_size, height, width, out_channels]),
            init=p.zero_state_init_params,
            dtype=p.dtype),
        c=py_utils.InitRNNCellState(
            tf.stack([batch_size, height, width, out_channels]),
            init=p.zero_state_init_params,
            dtype=p.dtype))

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    # Concate on channels.
    xm = tf.concat(inputs.act + [state0.m], 3)
    # TODO(yonghui): Possibly change the data_format to NCHW to speed
    # up conv2d kernel on gpu.
    xmw = tf.nn.conv2d(xm, theta.wm, [1, 1, 1, 1], 'SAME', data_format='NHWC')
    return xmw

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # Bias is applied to channels.
    bias = tf.reshape(theta.b, [1, 1, 1, -1])
    i_i, i_g, f_g, o_g = tf.split(
        value=xmw + bias, num_or_size_splits=4, axis=3)
    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)
    # Clip the cell states to reasonable value.
    new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)
    if p.output_nonlinearity:
      new_m = tf.sigmoid(o_g) * tf.tanh(new_c)
    else:
      new_m = tf.sigmoid(o_g) * new_c
    padding = tf.reshape(inputs.padding, [-1, 1, 1, 1])
    new_c = state0.c * padding + new_c * (1.0 - padding)
    new_m = state0.m * padding + new_m * (1.0 - padding)
    if p.zo_prob > 0.0:
      c_random_uniform = tf.random_uniform(tf.shape(new_c), seed=p.random_seed)
      m_random_uniform = tf.random_uniform(tf.shape(new_m), seed=p.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None
    new_c = self._ZoneOut(state0.c, new_c, padding, p.zo_prob, p.is_eval,
                          c_random_uniform)
    new_m = self._ZoneOut(state0.m, new_m, padding, p.zo_prob, p.is_eval,
                          m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class SRUCell(RNNCell):
  """SRU cell.

  From this paper: https://arxiv.org/abs/1709.02755

  This is a simple implementation that can be used as a drop-in replacement for
  another RNN. It doesn't do the performance tricks that an SRU is capable of,
  like unrolling matrix computations over time. This is just a basic
  implementation. It does the 4-matrix implementation found in appendix C.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the sru output. [batch, cell_nodes]
  - c: the sru cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(SRUCell, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define(
        'cell_value_cap', 10.0, 'SRU cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap]. This can be a'
        ' scalar or a scalar tensor.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('couple_input_forget_gates', True,
             'Whether to couple the input and forget gates.')
    p.Define('apply_layer_norm', False, 'Apply layer norm to the variables')
    p.Define(
        'layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.'
        'value is necessary only if apply_layer_norm is True')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes SRUCell."""
    super(SRUCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert p.reset_cell_state is False, ('SRUCell currently doesnt support '
                                         'resetting cell state.')
    assert isinstance(p.cell_value_cap, (int, float))

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, self.num_gates * self.hidden_size],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      bias_pc = py_utils.WeightParams(
          shape=[self.num_gates * self.hidden_size],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

      if p.num_hidden_nodes:
        w_proj = py_utils.WeightParams(
            shape=[self.hidden_size, self.output_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('w_proj', w_proj, self.AddGlobalVN)

      if p.apply_layer_norm:
        f_t_ln_scale = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('f_t_ln_scale', f_t_ln_scale, self.AddGlobalVN)
        r_t_ln_scale = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('r_t_ln_scale', r_t_ln_scale, self.AddGlobalVN)
        c_t_ln_scale = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('c_t_ln_scale', c_t_ln_scale, self.AddGlobalVN)
        if not p.couple_input_forget_gates:
          i_t_ln_scale = py_utils.WeightParams(
              shape=[self.hidden_size],
              init=py_utils.WeightInit.Constant(0.0),
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable('i_t_ln_scale', i_t_ln_scale, self.AddGlobalVN)

      # Collect some stats
      if p.couple_input_forget_gates:
        x_t2, resized, f_t, r_t = tf.split(
            value=self.vars.wm, num_or_size_splits=self.num_gates, axis=1)
      else:
        x_t2, resized, i_t, f_t, r_t = tf.split(
            value=self.vars.wm, num_or_size_splits=self.num_gates, axis=1)
        _HistogramSummary(p, scope.name + '/wm_i_t', i_t)
      _HistogramSummary(p, scope.name + '/wm_x_t2', x_t2)
      _HistogramSummary(p, scope.name + '/wm_resized', resized)
      _HistogramSummary(p, scope.name + '/wm_f_t', f_t)
      _HistogramSummary(p, scope.name + '/wm_r_t', r_t)

      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  @property
  def num_gates(self):
    return 4 if self.params.couple_input_forget_gates else 5

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def LayerNorm(self, x, scale):
    """Applies layer normalization on the last dimension of 'x'.

    Args:
      x: activation tensor, where the last dimension represents channels.
      scale: the scale tensor of the layer normalization

    Returns:
      Layer normalized 'x', with the same shape as the input.
    """
    p = self.params
    mean = tf.reduce_mean(x, axis=[1], keepdims=True)
    centered = x - mean
    variance = tf.reduce_mean(tf.square(centered), axis=[1], keepdims=True)
    normed = centered * tf.rsqrt(variance + p.layer_norm_epsilon)
    return normed * scale

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    return py_utils.Matmul(tf.concat(inputs.act, 1), theta.wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    if p.couple_input_forget_gates:
      x_t2, resized, f_t, r_t = tf.split(
          value=xmw + tf.expand_dims(theta.b, 0), num_or_size_splits=4, axis=1)
      if p.apply_layer_norm:
        f_t = self.LayerNorm(f_t, theta.f_t_ln_scale + 1.0)
      f_t = tf.nn.sigmoid(f_t)
      i_t = 1.0 - f_t
    else:
      x_t2, resized, i_t, f_t, r_t = tf.split(
          value=xmw + tf.expand_dims(theta.b, 0), num_or_size_splits=5, axis=1)
      if p.apply_layer_norm:
        f_t = self.LayerNorm(f_t, theta.f_t_ln_scale + 1.0)
      f_t = tf.nn.sigmoid(f_t)
      if p.apply_layer_norm:
        i_t = self.LayerNorm(i_t, theta.i_t_ln_scale + 1.0)
      i_t = tf.nn.sigmoid(i_t)
    if p.apply_layer_norm:
      r_t = self.LayerNorm(r_t, theta.r_t_ln_scale + 1.0)
    r_t = tf.nn.sigmoid(r_t)
    c_t = f_t * state0.c + i_t * x_t2
    if p.apply_layer_norm:
      c_t = self.LayerNorm(c_t, theta.c_t_ln_scale + 1.0)
    g_c_t = tf.nn.tanh(c_t)
    h_t = r_t * g_c_t + (1.0 - r_t) * resized

    # Clip the cell states to reasonable value.
    c_t = py_utils.clip_by_value(c_t, -p.cell_value_cap, p.cell_value_cap)

    if p.num_hidden_nodes:
      h_t = tf.matmul(h_t, theta.w_proj)

    return self._ApplyZoneOut(state0, inputs, c_t, h_t)

  def _ApplyZoneOut(self, state0, inputs, new_c, new_m):
    """Apply ZoneOut and returns updated states."""
    p = self.params
    if p.zo_prob > 0.0:
      assert not py_utils.use_tpu(), (
          'SRUCell does not support zoneout on TPU yet.')
      c_random_uniform = tf.random_uniform(tf.shape(new_c), seed=p.random_seed)
      m_random_uniform = tf.random_uniform(tf.shape(new_m), seed=p.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(state0.c, new_c, inputs.padding, p.zo_prob, p.is_eval,
                          c_random_uniform)
    new_m = self._ZoneOut(state0.m, new_m, inputs.padding, p.zo_prob, p.is_eval,
                          m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class QRNNPoolingCell(RNNCell):
  """This implements just the "pooling" part of a quasi-RNN or SRU.

  From these papers:

  - https://arxiv.org/abs/1611.01576
  - https://arxiv.org/abs/1709.02755

  The pooling part implements gates for recurrence. These architectures split
  the transform (conv or FC) from the gating/recurrent part. This cell can
  do either the quasi-RNN style or SRU style pooling operation based on params.

  If you want all of the functionality in one RNN cell, use `SRUCell` instead.

  theta:

    Has the trainable zero state. Other weights are done outside the recurrent
    loop.

  state:

  - m: the qrnn output. [batch, cell_nodes]
  - c: the qrnn cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes * num_rnn_matrices]
  - padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(QRNNPoolingCell, cls).Params()
    p.Define(
        'cell_value_cap', 10.0, 'LSTM cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap] if the value is not None. '
        'It can be a scalar, a scalar tensor or None. When set to None, '
        'no capping is applied.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('pooling_formula', 'INVALID',
             'Options: quasi_ifo, sru. Which pooling math to use')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes quasi-RNN Cell."""
    super(QRNNPoolingCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert p.reset_cell_state is False, ('QRNNPoolingCell currently doesnt '
                                         'support resetting cell state.')
    assert p.pooling_formula in ('quasi_ifo', 'sru')
    assert isinstance(p.cell_value_cap,
                      (int, float)) or p.cell_value_cap is None

    self._timestep = -1

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, p.num_output_nodes),
                                       init=p.zero_state_init_params,
                                       dtype=p.dtype)
    zero_c = py_utils.InitRNNCellState((batch_size, p.num_output_nodes),
                                       init=p.zero_state_init_params,
                                       dtype=p.dtype)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    # Just do identity. The convolution part of the QRNN has to be done earlier.
    return inputs.act

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    if p.pooling_formula == 'quasi_ifo':
      z_t, i_t, f_t, o_t = tf.split(
          value=tf.concat(xmw, 1), num_or_size_splits=4, axis=1)
      # Quasi-RNN "ifo" pooling
      c_t = f_t * state0.c + i_t * z_t
      h_t = o_t * c_t
    elif p.pooling_formula == 'sru':
      x_t2, resized, f_t, r_t = tf.split(
          value=tf.concat(xmw, 1), num_or_size_splits=4, axis=1)
      c_t = f_t * state0.c + (1.0 - f_t) * x_t2
      # TODO(otaviogood): Optimization - Since state doesn't depend on these
      # ops, they can be moved outside the loop.
      g_c_t = tf.nn.tanh(c_t)
      h_t = r_t * g_c_t + (1.0 - r_t) * resized
    else:
      raise ValueError('Invalid pooling_formula: %s', p.pooling_formula)

    new_c = c_t
    new_m = h_t

    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)

    # Apply Zoneout.
    return self._ApplyZoneOut(state0, inputs, new_c, new_m)

  def _ApplyZoneOut(self, state0, inputs, new_c, new_m):
    """Apply Zoneout and returns the updated states."""
    p = self.params
    if p.zo_prob > 0.0:
      c_random_uniform = tf.random_uniform(tf.shape(new_c), seed=p.random_seed)
      m_random_uniform = tf.random_uniform(tf.shape(new_m), seed=p.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(state0.c, new_c, inputs.padding, p.zo_prob, p.is_eval,
                          c_random_uniform)
    new_m = self._ZoneOut(state0.m, new_m, inputs.padding, p.zo_prob, p.is_eval,
                          m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class GRUCell(RNNCell):
  """ Gated Recurrent Unit cell.

  implemented: layer normalization, gru_biasing, gru_cell cap,
  not yet implemented: pruning, quantization, zone-out (enforced to 0.0 now)
  reference: https://arxiv.org/pdf/1412.3555.pdf

  theta:

  - w_n: the parameter weight matrix for the input block.
  - w_u: the parameter weight matrix for the update gate
  - w_r: the parameter weight matrix for the reset gate
  - b_n: the bias vector for the input block
  - b_u: the bias vector for the update gate
  - b_r: the bias vector for the reset gate

  state:

  - m: the GRU output. [batch, output_cell_nodes]
  - c: the GRU cell state. [batch, hidden_cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1]
  """

  @classmethod
  def Params(cls):
    p = super(GRUCell, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define(
        'cell_value_cap', 10.0, 'GRU cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap] if the value is not None. '
        'It can be a scalar, a scalar tensor or None. When set to None, '
        'no capping is applied.')
    p.Define('enable_gru_bias', False, 'Enable the GRU Cell bias.')
    p.Define('bias_init', py_utils.WeightInit.Constant(0.0),
             'Initialization parameters for GRU Cell bias')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('apply_layer_norm', True, 'Apply layer norm to the variables')
    p.Define(
        'layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.'
        'value is necessary only if apply_layer_norm is True')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes GRUCell."""
    super(GRUCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert isinstance(p.cell_value_cap,
                      (int, float)) or p.cell_value_cap is None
    assert p.zo_prob == 0.0

    def CreateVarHelper(variable_name, shape_to_init, params_to_init):
      """Utility function to initialize variables.

      Args:
        variable_name: the name of the variable
        shape_to_init: shape of the variables to be initialized.
        params_to_init: p.params_init, p.bias_init, or otherwise specified
      returns: initialized variable with name "$variable_name"
      """
      return self.CreateVariable(
          variable_name,
          py_utils.WeightParams(
              shape=shape_to_init,
              init=params_to_init,
              dtype=p.dtype,
              collections=self._VariableCollections()), self.AddGlobalVN)

    with tf.variable_scope(p.name):
      # Define weights.
      # Weight for block input
      CreateVarHelper('w_n',
                      [p.num_input_nodes + self.output_size, self.hidden_size],
                      p.params_init)
      # Weight for update gate
      CreateVarHelper('w_u',
                      [p.num_input_nodes + self.output_size, self.hidden_size],
                      p.params_init)
      # Weight for reset gate
      CreateVarHelper('w_r',
                      [p.num_input_nodes + self.output_size, self.output_size],
                      p.params_init)

      if p.num_hidden_nodes:
        # Set up projection matrix
        CreateVarHelper('w_proj', [self.hidden_size, self.output_size],
                        p.params_init)
        CreateVarHelper('b_proj', [self.output_size], p.bias_init)

      if p.enable_gru_bias:
        # Bias for the block input
        CreateVarHelper('b_n', [self.hidden_size], p.bias_init)
        # Bias for update gate
        CreateVarHelper('b_u', [self.hidden_size], p.bias_init)
        # Bias for the reset gate
        CreateVarHelper('b_r', [self.output_size], p.bias_init)

      if p.apply_layer_norm:
        assert p.layer_norm_epsilon is not None
        ln_unit = py_utils.WeightInit.Constant(0.0)
        CreateVarHelper('bn_ln_scale', [self.hidden_size], ln_unit)
        CreateVarHelper('bu_ln_scale', [self.hidden_size], ln_unit)
        CreateVarHelper('br_ln_scale', [self.output_size], ln_unit)

      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p))
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def GetOutput(self, state):
    return state.m

  def LayerNorm(self, x, scale):
    """Applies layer normalization on the last dimension of 'x'.

    Args:
      x: activation tensor, where the last dimension represents channels.
      scale: the scale tensor of the layer normalization

    Returns:
      Layer normalized 'x', with the same shape as the input.
    """
    p = self.params
    mean = tf.reduce_mean(x, axis=[1], keepdims=True)
    centered = x - mean
    variance = tf.reduce_mean(tf.square(centered), axis=[1], keepdims=True)
    normed = centered * tf.rsqrt(variance + p.layer_norm_epsilon)
    return normed * scale

  def FProp(self, theta, state0, inputs):
    """Forward function.

    GRU has coupled reset gate in the candidate actiavation function for output.
    See equation 5 and above in https://arxiv.org/pdf/1412.3555.pdf.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      state0: The previous recurrent state. A `.NestedMap`.
      inputs: The inputs to the cell. A `.NestedMap`.

    Returns:
      A tuple (state1, extras).
      - state1: The next recurrent state. A `.NestedMap`.
      - extras: Intermediate results to faciliate backprop. A `.NestedMap`.
    """

    p = self.params
    assert isinstance(inputs.act, list)

    # Update all gates
    # Compute r_g. r_g has size [batch, output]
    r_g = tf.matmul(tf.concat(inputs.act + [state0.m], 1), theta.w_r)
    if p.apply_layer_norm:
      r_g = self.LayerNorm(r_g, theta.br_ln_scale + 1.0)
    if p.enable_gru_bias:
      r_g = r_g + theta.b_r
    r_g = tf.sigmoid(r_g)

    # Compute u_g and n_g. Both have size [batch, hidden].
    # u_g has size [batch, hidden]
    u_g = tf.matmul(tf.concat(inputs.act + [state0.m], 1), theta.w_u)
    # size of n_g is [batch, hidden]
    n_g = tf.matmul(
        tf.concat(inputs.act + [tf.multiply(r_g, state0.m)], 1), theta.w_n)
    if p.apply_layer_norm:
      u_g = self.LayerNorm(u_g, theta.bu_ln_scale + 1.0)
      n_g = self.LayerNorm(n_g, theta.bn_ln_scale + 1.0)
    if p.enable_gru_bias:  # Add biases to u_g and n_g if needed
      u_g = u_g + theta.b_u
      n_g = n_g + theta.b_n

    u_g = tf.sigmoid(u_g)
    n_g = tf.tanh(n_g)

    new_c = (1.0 - u_g) * (state0.c) + u_g * n_g

    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)

    # Apply non-linear output is necessary
    new_m = new_c
    # Apply projection matrix if necessary
    if p.num_hidden_nodes:
      new_m = tf.matmul(new_m, theta.w_proj) + theta.b_proj
    # Apply padding.
    new_m = py_utils.ApplyPadding(inputs.padding, new_m, state0.m)
    new_c = py_utils.ApplyPadding(inputs.padding, new_c, state0.c)
    return py_utils.NestedMap(m=new_m, c=new_c), py_utils.NestedMap()
