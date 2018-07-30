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

from tensorflow.contrib import stateless
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

from lingvo.core import base_layer
from lingvo.core import cudnn_rnn_utils
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import summary_utils


def _HistogramSummary(p, name, v):
  """Adds a histogram summary for 'v' into the default tf graph."""
  summary_utils.histogram(p, name, tf.cast(v, tf.float32))


RNN_CELL_WT = 'rnn_cell_weight_variable'


def _FPropDtype(params):
  return params.fprop_dtype if params.fprop_dtype is not None else params.dtype


class RNNCell(base_layer.LayerBase):
  """RNN cells.

  RNNCell represents recurrent state in a NestedMap.

  zero_state(batch_size) returns the initial state, which is defined
  by each subclass.  From the state, each subclass defines GetOutput()
  to extract the output tensor.

  RNNCell.FProp defines the forward function:
    (theta, state0, inputs) -> state1, extras

  All arguments and return values are NestedMap. Each subclass defines
  what fields these NestedMap are expected to have.  'extras' is a
  NestedMap containing some intermediate results FProp computes to
  facilitate the backprop.

  zero_state(batch_size), state0 and state1 are all compatible
  NestedMaps (see NestedMap.IsCompatible). I.e., they have the same
  keys recursively. Furthermore, the corresponding tensors in these
  NestedMaps have the same shape and dtype.
  """

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
    function is composed of two functions:
      _Gates(_Mix(theta, state0, inputs), theta, state0, inputs)
    The result of _Mix is stashed in extras to facilitate backprop.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      state0: The previous recurrent state. A NestedMap.
      inputs: The inputs to the cell. A NestedMap.

    Returns:
      state1: The next recurrent state. A NestedMap.
      extras: Intermediate results to faciliate backprop. A NestedMap.
    """
    assert isinstance(inputs.act, list)
    assert self.params.inputs_arity == len(inputs.act)
    xmw = self._Mix(theta, state0, inputs)
    state1 = self._Gates(xmw, theta, state0, inputs)
    return state1, py_utils.NestedMap()


def ZoneOut(prev_v, cur_v, padding_v, zo_prob, is_eval, random_uniform):
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
  Returns:
    cur_v after ZoneOut regularization has been applied.
  """
  prev_v = tf.convert_to_tensor(prev_v)
  cur_v = tf.convert_to_tensor(cur_v)
  padding_v = tf.convert_to_tensor(padding_v)
  if zo_prob == 0.0:
    # Special case for when ZoneOut is not enabled.
    return prev_v * padding_v + cur_v * (1.0 - padding_v)

  if is_eval:
    # We take expectation in the eval mode.
    #
    # If padding_v is 1, it always carries over the previous state.
    zo_p = tf.minimum(1.0, tf.fill(tf.shape(prev_v), zo_prob) + padding_v)
  else:
    assert random_uniform is not None
    random_uniform = py_utils.HasShape(random_uniform, tf.shape(prev_v))
    zo_p = tf.cast(random_uniform < zo_prob, prev_v.dtype)
    zo_p += padding_v
    # If padding_v is 1, we always carry over the previous state.
    zo_p = tf.minimum(zo_p, 1.0)
  zo_p = tf.stop_gradient(zo_p)
  return prev_v * zo_p + cur_v * (1.0 - zo_p)


class LSTMCellSimple(RNNCell):
  """Simple LSTM cell.

  theta:
    wm: the parameter weight matrix. All gates combined.
    b: the combined bias vector.

  state:
    m: the lstm output. [batch, cell_nodes]
    c: the lstm cell state. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
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
    p.Define('random_seed', None, 'Random seed. Useful for unittests.')
    p.Define('trainable_zero_state', False,
             'If true the zero_states are trainable variables.')
    p.Define(
        'couple_input_forget_gates', False,
        'Whether to couple the input and forget gates. Just like '
        'tf.contrib.rnn.CoupledInputForgetGateLSTMCell')
    p.Define('apply_pruning', False, 'Whether to prune the weights while '
             'training')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LSTMCellSimple."""
    super(LSTMCellSimple, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert isinstance(p.cell_value_cap,
                      (int, float)) or p.cell_value_cap is None

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
        w_proj = wm_pc.Copy()
        w_proj.shape = [self.hidden_size, self.output_size]
        self.CreateVariable('w_proj', w_proj, self.AddGlobalVN)

      bias_pc = wm_pc.Copy()
      bias_pc.shape = [self.num_gates * self.hidden_size]
      bias_pc.init = py_utils.WeightInit.Constant(0.0)
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

      if p.trainable_zero_state:
        zs_m_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('zero_state_m', zs_m_pc)
        zs_c_pc = zs_m_pc.Copy()
        zs_c_pc.shape = [1, self.hidden_size]
        self.CreateVariable('zero_state_c', zs_c_pc)

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
    if p.trainable_zero_state:
      zero_m = tf.tile(self.theta.zero_state_m, [batch_size, 1])
      zero_c = tf.tile(self.theta.zero_state_c, [batch_size, 1])
    else:
      zero_m = tf.zeros((batch_size, self.output_size), dtype=_FPropDtype(p))
      zero_c = tf.zeros((batch_size, self.hidden_size), dtype=_FPropDtype(p))
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    return py_utils.Matmul(tf.concat(inputs.act + [state0.m], 1), theta.wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    if not p.couple_input_forget_gates:
      i_i, i_g, f_g, o_g = tf.split(
          value=xmw + tf.expand_dims(theta.b, 0), num_or_size_splits=4, axis=1)
      if p.forget_gate_bias != 0.0:
        f_g += p.forget_gate_bias
      new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)
    else:
      i_i, f_g, o_g = tf.split(
          value=xmw + tf.expand_dims(theta.b, 0), num_or_size_splits=3, axis=1)
      if p.forget_gate_bias != 0.0:
        f_g += p.forget_gate_bias
      new_c = (
          tf.sigmoid(f_g) * state0.c + (1.0 - tf.sigmoid(f_g)) * tf.tanh(i_i))
    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = tf.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)
    if p.output_nonlinearity:
      new_m = tf.sigmoid(o_g) * tf.tanh(new_c)
    else:
      new_m = tf.sigmoid(o_g) * new_c
    if p.num_hidden_nodes:
      new_m = tf.matmul(new_m, theta.w_proj)

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

    new_c = ZoneOut(state0.c, new_c, inputs.padding, p.zo_prob, p.is_eval,
                    c_random_uniform)
    new_m = ZoneOut(state0.m, new_m, inputs.padding, p.zo_prob, p.is_eval,
                    m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LSTMCellGrouped(RNNCell):
  """LSTM cell with groups.

  Grouping: based on "Factorization tricks for LSTM networks":
    https://arxiv.org/abs/1703.10722.
  Shuffling: adapted from "ShuffleNet: An Extremely Efficient Convolutional
    Neural Network for Mobile Devices": https://arxiv.org/abs/1707.01083.

  theta:
    groups: a list of child LSTM cells.

  state:
    A NestedMap containing 'groups', a list of NestedMaps, each with:
      m: the lstm output. [batch, cell_nodes // num_groups]
      c: the lstm cell state. [batch, cell_nodes // num_groups]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
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
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      state0: The previous recurrent state. A NestedMap.
      inputs: The inputs to the cell. A NestedMap.

    Returns:
      state1: The next recurrent state. A list.
      extras: An empty NestedMap.
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
        Group 0: 0_0, 0_1, 0_2, 0_3
        Group 1: 1_0, 1_1, 1_2, 1_3
        Group 2: 2_0, 2_1, 2_2, 2_3

      The shuffled output will be:
        Group 0: 0_0, 1_1, 2_2, 0_3
        Group 1: 1_0, 2_1, 0_2, 1_3
        Group 2: 2_0, 0_1, 1_2, 2_3
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
    assert not p.trainable_zero_state
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
    # We don't support trainable zero_state.
    assert not p.trainable_zero_state
    zero_m = tf.zeros((batch_size, self.output_size), dtype=p.dtype)
    zero_c = tf.zeros((batch_size, self.hidden_size), dtype=p.dtype)
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
        c_random_uniform = stateless.stateless_random_uniform(
            py_utils.GetShape(new_c, 2), tf.cast(c_seed, tf.int32))
        m_random_uniform = stateless.stateless_random_uniform(
            py_utils.GetShape(new_m, 2), tf.cast(m_seed, tf.int32))
      else:
        c_random_uniform = stateless.stateless_random_uniform(
            py_utils.GetShape(new_c, 2), c_seed)
        m_random_uniform = stateless.stateless_random_uniform(
            py_utils.GetShape(new_m, 2), m_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = ZoneOut(state0.c, new_c, inputs.padding, p.zo_prob, p.is_eval,
                    c_random_uniform)
    new_m = ZoneOut(state0.m, new_m, inputs.padding, p.zo_prob, p.is_eval,
                    m_random_uniform)
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
      summary_utils.scalar(p, 'step_counter', self._step_counter)
    return self._step_counter.assign(tf.cast(global_step, tf.int64))


class QuantizedLSTMCell(RNNCell):
  """Simplified LSTM cell used for quantized training.

  There is no forget_gate_bias, no output_nonlinearity and no bias. Right now
  only clipping is performed.

  theta:
    wm: the parameter weight matrix. All gates combined.
    cap: the cell value cap.

  state:
    m: the lstm output. [batch, cell_nodes]
    c: the lstm cell state. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(QuantizedLSTMCell, cls).Params()
    p.Define('random_seed', None, 'Random seed. Useful for unittests.')
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
    zero_m = tf.zeros((batch_size, p.num_output_nodes), dtype=_FPropDtype(p))
    zero_c = tf.zeros((batch_size, p.num_output_nodes), dtype=_FPropDtype(p))
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

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


class FakeQuantizedLSTMCell(RNNCell):
  """Simplified LSTM cell used for quantized training.

  Performs quantized training for downstream inference by a reduced bit
  depth inference engine (i.e. tfmini/tflite). This works by using a schedule
  to implement warm-up periods for unclipped, full-precision training. Then
  clipping is gradually introduced. Finally, simulated quantization is
  introduced (i.e. "fake quant") to the weights and activations.

  Note that this cell has a subset of features of LSTMCellSimple. Features
  are added as needed for models actually being productionalized. It may be
  appropriate at some point to merge the clipping/quantization here with
  LSTMCellSimple, but they are kept separate so that experimentation with that
  implementation is not impeded.

  See for the general procedure: https://arxiv.org/abs/1712.05877

  This implementation deviates from the paper in key ways as described
  in the implementation of the tflite reference LSTM op:
      https://github.com/tensorflow/tensorflow/blob/69f229a56652f076454ce9f3cb99bba285604ebe/tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h#L2009
  Changes include:
    - Ranges are not learned but derived from the clip schedule.
    - Activations are clipped from -1..1
    - Output of the hidden layer MatMul is clipped to -8..8 because it is
      the input to a TANH and benefits from the additional range.
    - Key outputs that are the result of state accumulation are quantized
      to 16bit.
    - In order to guarantee that zero is zero, ranges are actually adjusted
      to -cap..cap*((dt_max-1)/dt_max).

  theta:
    wm: the parameter weight matrix. All gates combined.
    cap: the cell value cap.

  state:
    m: the lstm output. [batch, cell_nodes]
    c: the lstm cell state. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(FakeQuantizedLSTMCell, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define('random_seed', None, 'Random seed. Useful for unittests.')
    p.Define('cc_schedule', quant_utils.FakeQuantizationSchedule.Params(),
             'Fake quantization clipping schedule.')
    p.Define('fc_start_cap', 64.0,
             'Clipping start cap for fully connected MatMul.')
    p.Define('fc_end_cap', 8.0,
             'Clipping end cap for the output of the fully connected MatMul.')
    p.Define('fc_bits', 16,
             'Quantized bit depth of fully connected components.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes FakeQuantizedLSTMCell."""
    super(FakeQuantizedLSTMCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params

    num_gates = 4
    with tf.variable_scope(p.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[
              p.num_input_nodes + self.output_size, num_gates * self.hidden_size
          ],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      if p.num_hidden_nodes:
        w_proj = wm_pc.Copy()
        w_proj.shape = [self.hidden_size, self.output_size]
        self.CreateVariable('w_proj', w_proj, self.AddGlobalVN)

      self.CreateChild('cc_schedule', p.cc_schedule)
      assert isinstance(
          self.cc_schedule, quant_utils.FakeQuantizationSchedule), (
              'FakeQuantizedLSTMCell requires a FakeQuantizationSchedule '
              'cc_schedule')

      # Collect some stats
      i_i, i_g, f_g, o_g = tf.split(
          value=self.vars.wm, num_or_size_splits=4, axis=1)
      _HistogramSummary(p, scope.name + '/wm_i_i', i_i)
      _HistogramSummary(p, scope.name + '/wm_i_g', i_g)
      _HistogramSummary(p, scope.name + '/wm_f_g', f_g)
      _HistogramSummary(p, scope.name + '/wm_o_g', o_g)

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
    zero_m = tf.zeros((batch_size, self.output_size), dtype=p.dtype)
    zero_c = tf.zeros((batch_size, self.hidden_size), dtype=p.dtype)
    if p.is_inference:
      # Toco needs fake quants applied to these so that it knows the ranges
      # in the case that they are used as standalone values.
      # Note that since this is inference-only, we can use the default theta.
      zero_m = self.cc_schedule.ApplyClipping(self.cc_schedule.theta, zero_m)
      zero_c = self.cc_schedule.ApplyClipping(
          self.cc_schedule.theta, zero_c, bits=16)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    p = self.params
    assert isinstance(inputs.act, list)
    act = inputs.act
    m = state0.m
    wm = self.cc_schedule.ApplyClipping(theta.cc_schedule, theta.wm)
    mixed = py_utils.Matmul(tf.concat(act + [m], 1), wm)
    mixed = self.cc_schedule.ApplyClipping(
        theta.cc_schedule,
        mixed,
        start_cap=p.fc_start_cap,
        end_cap=p.fc_end_cap,
        bits=16)
    return mixed

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""

    def Clip(x):
      return self.cc_schedule.ApplyClipping(theta.cc_schedule, x, bits=8)

    def Clip16(x):
      return self.cc_schedule.ApplyClipping(theta.cc_schedule, x, bits=16)

    p = self.params
    i_i, i_g, f_g, o_g = tf.split(value=xmw, num_or_size_splits=4, axis=1)

    orig_c = state0.c
    new_c = (
        Clip16(tf.sigmoid(f_g) * orig_c) + Clip16(
            tf.sigmoid(i_g) * tf.tanh(i_i)))
    new_c = Clip16(new_c)

    orig_m = state0.m
    new_m = Clip(tf.sigmoid(o_g) * new_c)
    if p.num_hidden_nodes:
      w_proj = Clip(theta.w_proj)
      new_m = Clip(tf.matmul(new_m, w_proj))

    # Respect padding.
    new_m = py_utils.ApplyPadding(inputs.padding, new_m, orig_m)
    new_c = py_utils.ApplyPadding(inputs.padding, new_c, orig_c)

    new_m.set_shape(state0.m.shape)
    new_c.set_shape(state0.c.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LSTMCellCuDNNCompliant(RNNCell):
  """LSTMCell compliant with variables with CuDNN-LSTM layout.

  theta:
    wb: the cudnn LSTM weight.

  state:
    m: the lstm output. [batch, cell_nodes]
    c: the lstm cell state. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
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
        m=tf.zeros([batch_size, p.num_output_nodes], dtype=p.dtype),
        c=tf.zeros([batch_size, p.num_output_nodes], dtype=p.dtype))

  def GetOutput(self, state):
    return state.m

  def _WeightAndBias(self, theta):
    p = self.params
    return cudnn_rnn_utils.RecoverLSTMCellSimpleWeightsFromCuDNN(
        theta.wb, p.num_input_nodes, p.num_output_nodes,
        cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION)

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
    wm: the parameter weight matrix. All gates combined.
    b: the combined bias vector.

  state:
    m: the lstm output. [batch, cell_nodes]
    c: the lstm cell state. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
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
    p.Define('random_seed', None, 'Random seed. Useful for unittests.')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.')
    p.Define('trainable_zero_state', False,
             'If true the zero_states are trainable variables.')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')
    return p

  @tf.contrib.framework.deprecated(
      date=None,
      instructions='New models should use LayerNormalizedLSTMCellSimple.')
  @base_layer.initializer
  def __init__(self, params):
    """Initializes LayerNormalizedLSTMCell."""
    super(LayerNormalizedLSTMCell, self).__init__(params)
    params = self.params
    if params.trainable_zero_state:
      raise ValueError('Trainable inital state is not supported!')
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
      bias_pc = wm_pc.Copy()
      bias_pc.shape = [
          4 * params.num_output_nodes + 4 * params.num_output_nodes
      ]
      bias_pc.init = py_utils.WeightInit.Constant(0.0)
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
    params = self.params
    return py_utils.NestedMap(
        m=tf.zeros([batch_size, params.num_output_nodes], dtype=params.dtype),
        c=tf.zeros([batch_size, params.num_output_nodes], dtype=params.dtype))

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    if not isinstance(inputs.act, list):
      raise ValueError('Input activations must be of list type!')
    return py_utils.Matmul(tf.concat(inputs.act + [state0.m], 1), theta.wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    # Unpack the variables (weight and bias) into individual variables.
    params = self.params

    def BiasSlice(dim, start_ind):
      s = theta.b[start_ind:start_ind + dim]
      start_ind += dim
      return s, start_ind

    # Unpack the bias variable.
    slice_start = 0
    bias_lstm, slice_start = BiasSlice(4 * params.num_output_nodes, slice_start)
    ln_scale, slice_start = BiasSlice(4 * params.num_output_nodes, slice_start)
    assert slice_start == 8 * params.num_output_nodes

    def _LayerNorm(x):
      last_dim = tf.rank(x) - 1
      mean = tf.reduce_mean(x, axis=[last_dim], keepdims=True)
      variance = tf.reduce_mean(
          tf.square(x - mean), axis=[last_dim], keepdims=True)
      return (x - mean) * tf.rsqrt(variance + params.layer_norm_epsilon)

    # TODO(yonghui): define a function to reduce memory usage here.
    def _PerGateLayerNorm(x, scale):
      # Normalize each gate activation values separately.
      x_reshaped = tf.reshape(
          x, tf.stack([tf.shape(x)[0], 4, params.num_output_nodes]))
      x_norm = _LayerNorm(x_reshaped)
      return tf.reshape(x_norm, tf.shape(x)) * tf.expand_dims(scale, 0)

    i_i, i_g, f_g, o_g = tf.split(
        _PerGateLayerNorm(xmw, ln_scale + 1.0) + tf.expand_dims(bias_lstm, 0),
        num_or_size_splits=4,
        axis=1)

    if params.forget_gate_bias != 0.0:
      f_g += params.forget_gate_bias
    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)

    # Clip the cell states to reasonable value.
    if params.cc_schedule:
      cap = self.cc_schedule.CurrentCap(theta.cc_schedule)
    else:
      cap = params.cell_value_cap
    new_c = tf.clip_by_value(new_c, -cap, cap)

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

    new_c = ZoneOut(state0.c, new_c, inputs.padding, params.zo_prob,
                    params.is_eval, c_random_uniform)
    new_m = ZoneOut(state0.m, new_m, inputs.padding, params.zo_prob,
                    params.is_eval, m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LayerNormalizedLSTMCellSimple(LSTMCellSimple):
  """An implementation of layer normalized LSTM based on LSTMCellSimple.

  Implements normalization scheme as described in
  https://arxiv.org/pdf/1607.06450.pdf

  theta:
    wm: the parameter weight matrix. All gates combined.
    b: the combined bias vector.

  state:
    m: the lstm output. [batch, cell_nodes]
    c: the lstm cell state. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
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

    # TODO(yonghui): define a function to reduce memory usage here.
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


class ConvLSTMCell(RNNCell):
  """Convolution LSTM cells.

  theta:
    wm: the parameter weight matrix. All gates combined.
    b: the combined bias vector.

  state:
    m: the lstm output. cell_shape
    c: the lstm cell state. cell_shape

  inputs:
    act: a list of input activations. input_shape.
    padding: the padding. [batch].
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
    p.Define('random_seed', None, 'Random seed. Useful for unittests.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ConvLSTMCell."""
    assert isinstance(params, hyperparams.Params)
    super(ConvLSTMCell, self).__init__(params)

    p = self.params
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

      bias_pc = wm_pc.Copy()
      bias_pc.shape = [4 * out_channels]
      bias_pc.init = py_utils.WeightInit.Constant(0.0)
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    height = p.inputs_shape[1]
    width = p.inputs_shape[2]
    out_channels = p.cell_shape[3]
    return py_utils.NestedMap(
        m=tf.zeros(
            tf.stack([batch_size, height, width, out_channels]), dtype=p.dtype),
        c=tf.zeros(
            tf.stack([batch_size, height, width, out_channels]), dtype=p.dtype))

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
    new_c = tf.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)
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
    new_c = ZoneOut(state0.c, new_c, padding, p.zo_prob, p.is_eval,
                    c_random_uniform)
    new_m = ZoneOut(state0.m, new_m, padding, p.zo_prob, p.is_eval,
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
    wm: the parameter weight matrix. All gates combined.
    b: the combined bias vector.

  state:
    m: the sru output. [batch, cell_nodes]
    c: the sru cell state. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(SRUCell, cls).Params()
    p.Define(
        'cell_value_cap', 10.0, 'SRU cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap]. This can be a'
        ' scalar or a scalar tensor.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('random_seed', None, 'Random seed. Useful for unittests.')
    p.Define('trainable_zero_state', False,
             'If true the zero_states are trainable variables.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes SRUCell."""
    super(SRUCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert isinstance(p.cell_value_cap, (int, float))

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, 4 * p.num_output_nodes],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      bias_pc = wm_pc.Copy()
      bias_pc.shape = [4 * p.num_output_nodes]
      bias_pc.init = py_utils.WeightInit.Constant(0.0)
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

      if p.trainable_zero_state:
        zs_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('zero_state_m', zs_pc)
        self.CreateVariable('zero_state_c', zs_pc)

      # Collect some stats
      x_t2, resized, f_t, r_t = tf.split(
          value=self.vars.wm, num_or_size_splits=4, axis=1)
      _HistogramSummary(p, scope.name + '/wm_x_t2', x_t2)
      _HistogramSummary(p, scope.name + '/wm_resized', resized)
      _HistogramSummary(p, scope.name + '/wm_f_t', f_t)
      _HistogramSummary(p, scope.name + '/wm_r_t', r_t)

      self._timestep = -1

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    if p.trainable_zero_state:
      zero_m = tf.tile(self.theta.zero_state_m, [batch_size, 1])
      zero_c = tf.tile(self.theta.zero_state_c, [batch_size, 1])
    else:
      zero_m = tf.zeros((batch_size, p.num_output_nodes), dtype=p.dtype)
      zero_c = tf.zeros((batch_size, p.num_output_nodes), dtype=p.dtype)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    return py_utils.Matmul(tf.concat(inputs.act, 1), theta.wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    x_t2, resized, f_t, r_t = tf.split(
        value=xmw + tf.expand_dims(theta.b, 0), num_or_size_splits=4, axis=1)
    f_t = tf.nn.sigmoid(f_t)
    r_t = tf.nn.sigmoid(r_t)
    c_t = f_t * state0.c + (1.0 - f_t) * x_t2
    g_c_t = tf.nn.tanh(c_t)
    h_t = r_t * g_c_t + (1.0 - r_t) * resized

    # Clip the cell states to reasonable value.
    c_t = tf.clip_by_value(c_t, -p.cell_value_cap, p.cell_value_cap)

    c_t = ZoneOut(state0.c, c_t, inputs.padding, p.zo_prob, p.is_eval,
                  p.random_seed)
    h_t = ZoneOut(state0.m, h_t, inputs.padding, p.zo_prob, p.is_eval,
                  p.random_seed)
    c_t.set_shape(state0.c.shape)
    h_t.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=h_t, c=c_t)


class QRNNPoolingCell(RNNCell):
  """This implements just the "pooling" part of a quasi-RNN or SRU.

  From these papers:
   https://arxiv.org/abs/1611.01576
   https://arxiv.org/abs/1709.02755

  The pooling part implements gates for recurrence. These architectures split
  the transform (conv or FC) from the gating/recurrent part. This cell can
  do either the quasi-RNN style or SRU style pooling operation based on params.

  If you want all of the functionality in one RNN cell, use "SRUCell" instead.

  theta:
    Has the trainable zero state. Other weights are done outside the recurrent
    loop.

  state:
    m: the qrnn output. [batch, cell_nodes]
    c: the qrnn cell state. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes * num_rnn_matrices]
    padding: the padding. [batch, 1].
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
    p.Define('random_seed', None, 'Random seed. Useful for unittests.')
    p.Define('trainable_zero_state', False,
             'If true the zero_states are trainable variables.')
    p.Define('pooling_formula', 'INVALID',
             'Options: quasi_ifo, sru. Which pooling math to use')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes quasi-RNN Cell."""
    super(QRNNPoolingCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert p.pooling_formula in ('quasi_ifo', 'sru')
    assert isinstance(p.cell_value_cap,
                      (int, float)) or p.cell_value_cap is None

    with tf.variable_scope(p.name):
      if p.trainable_zero_state:
        zs_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('zero_state_m', zs_pc)
        self.CreateVariable('zero_state_c', zs_pc)

      self._timestep = -1

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, batch_size):
    p = self.params
    if p.trainable_zero_state:
      zero_m = tf.tile(self.theta.zero_state_m, [batch_size, 1])
      zero_c = tf.tile(self.theta.zero_state_c, [batch_size, 1])
    else:
      zero_m = tf.zeros((batch_size, p.num_output_nodes), dtype=p.dtype)
      zero_c = tf.zeros((batch_size, p.num_output_nodes), dtype=p.dtype)
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
      new_c = tf.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)

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

    new_c = ZoneOut(state0.c, new_c, inputs.padding, p.zo_prob, p.is_eval,
                    c_random_uniform)
    new_m = ZoneOut(state0.m, new_m, inputs.padding, p.zo_prob, p.is_eval,
                    m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)
