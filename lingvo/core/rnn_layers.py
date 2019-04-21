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
"""Lingvo RNN layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip
import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.framework import function

from lingvo.core import attention
from lingvo.core import base_layer
from lingvo.core import cudnn_rnn_utils
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import recurrent
from lingvo.core import rnn_cell


def _AssertCellParamsCuDNNCompatible(p_cell):
  if p_cell.cls == rnn_cell.LSTMCellCuDNNCompliant:
    return
  else:
    assert p_cell.cls == rnn_cell.LSTMCellSimple
    assert isinstance(p_cell, hyperparams.Params)
    assert p_cell.cell_value_cap is None
    assert p_cell.forget_gate_bias == 0
    assert p_cell.output_nonlinearity
    assert p_cell.zo_prob == 0.0


def _GeneratePackedInputResetMask(segment_id, is_reverse=False):
  """Generates mask inputs for RNN cells from segment_id.

  Args:
    segment_id: A tensor of shape [time, batch_size, 1].
    is_reverse: True if inputs are fed to the RNN in reverse order.

  Returns:
    reset_mask - a tensor of shape [time, batch_size, 1]. Set to 0 for samples
      where state needs to be reset (at example boundaries), and 1 otherwise.
  """
  segment_id_left = segment_id[:-1]
  segment_id_right = segment_id[1:]

  # Mask is a [t-1, bs, 1] tensor.
  reset_mask = tf.cast(
      tf.equal(segment_id_left, segment_id_right), dtype=segment_id.dtype)
  mask_padding_shape = tf.concat(
      [tf.ones([1], dtype=tf.int32),
       tf.shape(segment_id)[1:]], axis=0)
  mask_padding = tf.ones(mask_padding_shape, dtype=segment_id.dtype)
  if is_reverse:
    reset_mask = tf.concat([reset_mask, mask_padding], axis=0)
  else:
    reset_mask = tf.concat([mask_padding, reset_mask], axis=0)
  return reset_mask


class IdentitySeqLayer(base_layer.BaseLayer):
  """A no-op sequence layer."""

  @classmethod
  def Params(cls):
    p = super(IdentitySeqLayer, cls).Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(IdentitySeqLayer, self).__init__(params)

  def zero_state(self, batch_size):
    return py_utils.NestedMap()

  def FPropFullSequence(self, theta, inputs, paddings):
    return inputs


class RNN(base_layer.BaseLayer):
  """Statically unrolled RNN."""

  @classmethod
  def Params(cls):
    p = super(RNN, cls).Params()
    p.Define('cell', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the RNN cell.')
    p.Define(
        'sequence_length', 0,
        'Sequence length to unroll. If > 0, then will unroll to this fixed '
        'size. If 0, then will unroll to accommodate the size of the inputs '
        'for each call to FProp.')
    p.Define('reverse', False,
             'Whether or not to unroll the sequence in reversed order.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RNN, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed inputs are currently not supported by '
                                'Static RNN')
    p.cell.reset_cell_state = p.packed_input
    assert p.sequence_length >= 0
    self.CreateChild('cell', p.cell)

  def zero_state(self, batch_size):
    return self.cell.zero_state(batch_size)

  def FProp(self, theta, inputs, paddings, state0=None):
    """Compute RNN forward pass.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A single tensor or a tuple of tensors with cardinality equal to
          rnn_cell.inputs_arity. For every input tensor, the first dimension is
          assumed to be time, second dimension batch, and third dimension depth.
      paddings: A tensor. First dim is time, second dim is batch, and third dim
          is expected to be 1.
      state0: If not None, the initial rnn state in a `.NestedMap`. Defaults
        to the cell's zero-state.

    Returns:
      A tensor of [time, batch, dims].
      The final recurrent state.
    """
    p = self.params
    rcell = self.cell
    assert isinstance(rcell, (rnn_cell.RNNCell))
    if p.sequence_length == 0:
      if isinstance(inputs, (tuple, list)):
        sequence_length = len(inputs)
      else:
        sequence_length = py_utils.GetShape(inputs)[0]
    else:
      sequence_length = p.sequence_length
    assert sequence_length >= 1, ('Sequence length must be defined or inputs '
                                  'must have fixed shapes.')
    with tf.name_scope(p.name):
      inputs_sequence = tf.unstack(inputs, num=sequence_length)
      paddings_sequence = tf.unstack(paddings, num=sequence_length)
      # We start from all 0 states.
      if state0:
        state = state0
      else:
        inputs0 = py_utils.NestedMap(
            act=[inputs_sequence[0]], padding=paddings_sequence[0])
        state = rcell.zero_state(rcell.batch_size(inputs0))
      outputs = [None] * sequence_length
      if p.reverse:
        sequence = xrange(sequence_length - 1, -1, -1)
      else:
        sequence = xrange(0, sequence_length, 1)
      for idx in sequence:
        cur_input = py_utils.NestedMap(act=[inputs[idx]], padding=paddings[idx])
        state, _ = rcell.FProp(theta.cell, state, cur_input)
        outputs[idx] = rcell.GetOutput(state)
      return tf.stack(outputs), state


class StackedRNNBase(base_layer.BaseLayer):
  """Stacked RNN base class."""

  @classmethod
  def Params(cls):
    p = super(StackedRNNBase, cls).Params()
    p.Define('num_layers', 1, 'The number of RNN layers.')
    p.Define(
        'num_input_nodes', -1,
        'If >= 0, overrides cell_tpl.num_input_nodes for the first layer.')
    p.Define(
        'num_output_nodes', -1,
        'If >= 0, overrides cell_tpl.num_output_nodes for the last layer.')
    p.Define('skip_start', 1, 'The first layer start skip connection.')
    p.Define(
        'cell_tpl', rnn_cell.LSTMCellSimple.Params(),
        'Configs for the RNN cell(s). '
        'If cell_tpl is not a list/tuple, the same cell config is used '
        'for all layers. Otherwise, cell_tpl[i] is the config for '
        'i-th layer and cell_tpl[-1] is used for the rest of layers.')
    p.Define('dropout', layers.DropoutLayer.Params(),
             'Dropout applied to each layer.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(StackedRNNBase, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed inputs are currently not supported by '
                                'Static RNN Base')

  def _GetCellTpls(self):
    p = self.params
    if not isinstance(p.cell_tpl, (tuple, list)):
      cell_tpls = [p.cell_tpl] * p.num_layers
    else:
      cell_tpls = list(p.cell_tpl)
      assert len(p.cell_tpl) <= p.num_layers
      last = cell_tpls[-1]
      while len(cell_tpls) < p.num_layers:
        cell_tpls.append(last)
    for cell_tpl in cell_tpls:
      cell_tpl.reset_cell_state = p.packed_input
    return cell_tpls


class StackedFRNNLayerByLayer(StackedRNNBase, quant_utils.QuantizableLayer):
  """An implemention of StackedRNNBase which computes layer-by-layer."""

  @classmethod
  def Params(cls):
    p = super(StackedFRNNLayerByLayer, cls).Params()
    p.Define('rnn_tpl', FRNN.Params(), 'Rnn cell default params.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(StackedFRNNLayerByLayer, self).__init__(params)
    p = self.params

    rnn_params = []
    with tf.name_scope(p.name):
      for (i, cell_tpl) in enumerate(self._GetCellTpls()):
        params = p.rnn_tpl.Copy()
        params.packed_input = p.packed_input
        params.allow_implicit_capture = p.allow_implicit_capture
        params.name = 'frnn_%d' % i
        params.cell = cell_tpl.Copy()
        params.cell.name = '%s_%d' % (p.name, i)
        if p.num_input_nodes > 0 and i == 0:
          params.cell.num_input_nodes = p.num_input_nodes
        if p.num_output_nodes > 0 and i == p.num_layers - 1:
          params.cell.num_output_nodes = p.num_output_nodes
        rnn_params.append(params)

    for i in range(len(rnn_params) - 1):
      # Because one layer's output needs to be fed into the next layer's
      # input, hence, we have this assertion. We can relax it later by
      # allowing more parameterization of the layers.
      assert (rnn_params[i].cell.num_output_nodes == rnn_params[i + 1].cell
              .num_input_nodes)

    self.CreateChildren('rnn', rnn_params)
    self.CreateChild('dropout', p.dropout)
    self.TrackQTensor('residual')

  def zero_state(self, batch_size):
    p = self.params
    ret = py_utils.NestedMap(rnn=[])
    for i in range(p.num_layers):
      state0 = self.rnn[i].zero_state(batch_size)
      ret.rnn.append(state0)
    return ret

  def FProp(self, theta, inputs, paddings, state0=None):
    """Compute RNN forward pass.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A single tensor of shape [time, batch, dims].
      paddings: A single tensor of shape [time, batch, 1].
      state0: If not None, the initial rnn state in a `.NestedMap`. Defaults
        to the init state.

    Returns:
      (outputs, state1)
      outputs: A tensor of [time, batch, dims].
      state1: The final state.
    """
    p = self.params
    if not state0:
      state0 = self.zero_state(tf.shape(inputs)[1])
    xs = inputs
    state1 = py_utils.NestedMap(rnn=[None] * p.num_layers)
    for i in range(p.num_layers):
      ys, state1.rnn[i] = self.rnn[i].FProp(theta.rnn[i], xs, paddings,
                                            state0.rnn[i])
      ys = self.dropout.FProp(theta.dropout, ys)
      if (p.skip_start >= 0 and i >= p.skip_start and
          (p.num_input_nodes <= 0 or i != 0) and
          (p.num_output_nodes <= 0 or i != p.num_layers - 1)):
        ys = self.fns.qadd(ys, xs, qt='residual')
      xs = ys
    return xs, state1

  def FPropFullSequence(self, theta, inputs, paddings):
    return self.FProp(theta, inputs, paddings)[0]


class StackedBiFRNNLayerByLayer(StackedRNNBase, quant_utils.QuantizableLayer):
  """An implemention of StackedRNNBase with bidirection RNN layers."""

  @base_layer.initializer
  def __init__(self, params):
    super(StackedBiFRNNLayerByLayer, self).__init__(params)
    p = self.params

    rnn_params = []
    feature_dim = None
    with tf.name_scope(p.name):
      for (i, cell_tpl) in enumerate(self._GetCellTpls()):
        rnn_p = cell_tpl.Copy()
        if i > 0:
          rnn_p.num_input_nodes = feature_dim
        if p.num_input_nodes > 0 and i == 0:
          rnn_p.num_input_nodes = p.num_input_nodes
        if p.num_output_nodes > 0 and i == p.num_layers - 1:
          rnn_p.num_output_nodes = p.num_output_nodes // 2
        frnn_param = BidirectionalFRNN.Params()
        frnn_param.name = 'bidi_rnn_%d' % i
        frnn_param.fwd = rnn_p.Copy().Set(name='f_rnn_%d' % i)
        frnn_param.bak = rnn_p.Copy().Set(name='b_rnn_%d' % i)
        rnn_params.append(frnn_param)
        feature_dim = 2 * rnn_p.num_output_nodes

    self.CreateChildren('rnn', rnn_params)
    self.CreateChild('dropout', p.dropout)
    self.TrackQTensor('residual')

  def FProp(self, theta, inputs, paddings):
    """Compute the forward pass.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A single tensor of shape [time, batch, dims].
      paddings: A single tensor of shape [time, batch, 1].

    Returns:
      A tensor of [time, batch, dims].
    """
    p = self.params
    xs = inputs
    for i in range(p.num_layers):
      ys = self.rnn[i].FProp(theta.rnn[i], xs, paddings)
      ys = self.dropout.FProp(theta.dropout, ys)
      if (p.skip_start >= 0 and i >= p.skip_start and
          (p.num_input_nodes <= 0 or i != 0) and
          (p.num_output_nodes <= 0 or i != p.num_layers - 1)):
        ys = self.fns.qadd(ys, xs, qt='residual')
      xs = ys
    return xs

  def FPropFullSequence(self, theta, inputs, paddings):
    return self.FProp(theta, inputs, paddings)


class FRNN(base_layer.BaseLayer):
  """Functional while based RNN."""

  @classmethod
  def Params(cls):
    p = super(FRNN, cls).Params()
    p.Define('cell', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the RNN cell.')
    p.Define('reverse', False,
             'Whether or not to unroll the sequence in reversed order.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(FRNN, self).__init__(params)
    p = self.params
    p.cell.reset_cell_state = p.packed_input
    self.CreateChild('cell', p.cell)

  @property
  def rnn_cell(self):
    return self.cell

  def zero_state(self, batch_size):
    return self.cell.zero_state(batch_size)

  def FProp(self, theta, inputs, paddings, state0=None, segment_id=None):
    """Compute RNN forward pass.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A single tensor or a tuple of tensors with cardinality equal to
          rnn_cell.inputs_arity. For every input tensor, the first dimension is
          assumed to be time, second dimension batch, and third dimension depth.
      paddings: A tensor. First dim is time, second dim is batch, and third dim
          is expected to be 1.
      state0: If not None, the initial rnn state in a `.NestedMap`. Defaults
        to the cell's zero-state.
      segment_id: A tensor to support packed inputs. First dim is time, second
          dim is batch, and third dim is expected to be 1.

    Returns:
      A tensor of [time, batch, dims].
      The final recurrent state.
    """
    p = self.params
    rcell = self.cell
    assert isinstance(rcell, (rnn_cell.RNNCell))

    @function.Defun()
    def FlipUpDown(x):
      # Reverse the first dimension (time)
      return tf.reverse(x, [0])

    if not isinstance(inputs, (list, tuple)):
      inputs = [inputs]

    if p.packed_input:
      assert segment_id is not None
      reset_mask = _GeneratePackedInputResetMask(
          segment_id, is_reverse=p.reverse)
      reset_mask = py_utils.HasShape(reset_mask, tf.shape(paddings))
    else:
      reset_mask = tf.zeros_like(paddings)

    if p.reverse:
      inputs = [FlipUpDown(x) for x in inputs]
      paddings = FlipUpDown(paddings)
      reset_mask = FlipUpDown(reset_mask)

    if not state0:
      inputs0 = py_utils.NestedMap(
          act=[x[0] for x in inputs],
          padding=paddings[0, :],
          reset_mask=reset_mask[0, :])
      state0 = rcell.zero_state(rcell.batch_size(inputs0))

    inputs = py_utils.NestedMap(
        act=inputs, padding=paddings, reset_mask=reset_mask)

    acc_state, final_state = recurrent.Recurrent(
        theta=theta.cell,
        state0=state0,
        inputs=inputs,
        cell_fn=rcell.FProp,
        accumulator_layer=self,
        allow_implicit_capture=p.allow_implicit_capture)

    act = rcell.GetOutput(acc_state)
    if p.reverse:
      act = FlipUpDown(act)
    return act, final_state


class BidirectionalFRNN(base_layer.BaseLayer):
  """Bidirectional functional RNN."""

  @classmethod
  def Params(cls):
    p = super(BidirectionalFRNN, cls).Params()
    p.Define('fwd', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the forward RNN cell.')
    p.Define('bak', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the backward RNN cell.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BidirectionalFRNN, self).__init__(params)
    p = params
    cluster = self.cluster
    if py_utils.use_tpu():
      fwd_device = cluster.WorkerDeviceInModelSplit(0)
      bwd_device = cluster.WorkerDeviceInModelSplit(1)
    else:
      fwd_device = ''
      bwd_device = ''
    with tf.device(fwd_device):
      params_forward = FRNN.Params()
      params_forward.name = 'fwd'
      params_forward.dtype = p.dtype
      params_forward.reverse = False
      params_forward.packed_input = p.packed_input
      params_forward.cell = p.fwd.Copy()
      self.CreateChild('fwd_rnn', params_forward)

    with tf.device(bwd_device):
      params_backward = FRNN.Params()
      params_backward.name = 'bak'
      params_backward.dtype = p.dtype
      params_backward.reverse = True
      params_backward.packed_input = p.packed_input
      params_backward.cell = p.bak.Copy()
      self.CreateChild('bak_rnn', params_backward)

  def FProp(self, theta, inputs, paddings, segment_id=None):
    """Compute bidi-RNN forward pass.

    `rcell_forward` unroll the sequence in the forward direction and
    `rcell_backward` unroll the sequence in the backward direction. The
    outputs are concatenated in the last output dim.

    See `FRNN.FProp` for more details.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A single tensor or a tuple of tensors with cardinality equal to
          rnn_cell.inputs_arity. For every input tensor, the first dimension is
          assumed to be time, second dimension batch, and third dimension depth.
      paddings: A tensor. First dim is time, second dim is batch, and third dim
          is expected to be 1.
      segment_id: A tensor to support packed inputs. First dim is time, second
          dim is batch, and third dim is expected to be 1.

    Returns:
      A tensor of [time, batch, dims].
    """
    p = self.params
    with tf.name_scope(p.name):

      def Fwd():
        """Run the forward pass."""
        output_forward, _ = self.fwd_rnn.FProp(
            theta.fwd_rnn, inputs, paddings, segment_id=segment_id)
        return output_forward

      def Bwd():
        """Run the backward pass.

        Returns:
          A tensor of [time, batch, dims]. The final recurrent state.
        """
        output_backward, _ = self.bak_rnn.FProp(
            theta.bak_rnn, inputs, paddings, segment_id=segment_id)
        # TODO(yonghui/zhifengc): In the current implementation, we copy
        # output_forward from gpu:0 to gpu:1, and then copy the concatenated
        # output from gpu:1 to gpu:0 to enable next layer computation. It might
        # be more efficient to only copy output_backward from gpu:1 to gpu:0 to
        # reduce cross-gpu data transfer.
        return output_backward

      # On TPU, we run both direction's RNNs on one device to reduce memory
      # usage.
      cluster = self.cluster
      fwd_device = cluster.WorkerDeviceInModelSplit(0)
      bwd_device = cluster.WorkerDeviceInModelSplit(1)
      with tf.device(fwd_device):
        output_forward = Fwd()
      with tf.device(bwd_device):
        output_backward = Bwd()
      with tf.device(fwd_device):
        return tf.concat([output_forward, output_backward], -1)


class BidirectionalRNN(base_layer.BaseLayer):
  """Statically unrolled bidirectional RNN."""

  @classmethod
  def Params(cls):
    p = super(BidirectionalRNN, cls).Params()
    p.Define('fwd', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the forward RNN cell.')
    p.Define('bak', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the backward RNN cell.')
    p.Define('sequence_length', 0, 'Sequence length.')
    p.Define('rnn', RNN.Params(), 'Config for underlying RNNs')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BidirectionalRNN, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed input is currently not supported by '
                                'BiDirectionalRNN')
    params_forward = p.rnn.Copy()
    params_forward.name = '%s_forward' % p.name
    params_forward.cell = p.fwd.Copy()
    params_forward.sequence_length = p.sequence_length
    self.CreateChild('fwd_rnn', params_forward)
    params_backward = p.rnn.Copy()
    params_backward.name = '%s_backward' % p.name
    params_backward.cell = p.bak.Copy()
    params_backward.sequence_length = p.sequence_length
    params_backward.reverse = True
    self.CreateChild('bak_rnn', params_backward)

  def FProp(self, theta, inputs, paddings):
    """Compute bidi-RNN forward pass.

    `rcell_forward` is responsible for unrolling the sequence in the forward
    direction and `rcell_backward` in the backward direction. Output from
    forward and backward rnns are concatenated on the last output dim.

    See `RNN.FProp()` for more details.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A single tensor or a tuple of tensors with cardinality equal to
          rnn_cell.inputs_arity. For every input tensor, the first dimension is
          assumed to be time, second dimension batch, and third dimension depth.
      paddings: A tensor. First dim is time, second dim is batch, and third dim
          is expected to be 1.

    Returns:
      A tensor of [time, batch, dims].
    """
    p = self.params

    with tf.name_scope(p.name):
      outputs_forward, _ = self.fwd_rnn.FProp(theta.fwd_rnn, inputs, paddings)
      outputs_backward, _ = self.bak_rnn.FProp(theta.bak_rnn, inputs, paddings)
      return tf.concat([outputs_forward, outputs_backward], axis=-1)


class BidirectionalRNNV2(base_layer.BaseLayer):
  """Statically unrolled bidirectional RNN."""

  @classmethod
  def Params(cls):
    p = super(BidirectionalRNNV2, cls).Params()
    p.Define('fwd', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the forward RNN cell.')
    p.Define('bak', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the backward RNN cell.')
    p.Define('sequence_length', 0, 'Sequence length.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BidirectionalRNNV2, self).__init__(params)
    assert not self.params.packed_input, ('Packed input is currently not '
                                          'supported by BiDirectionalRNNV2')
    p = BidirectionalRNN.Params()
    p.packed_input = self.params.packed_input
    p.name = '%s_brnn' % self.params.name
    p.fwd = self.params.fwd.Copy()
    p.bak = self.params.bak.Copy()
    p.sequence_length = self.params.sequence_length
    self.CreateChild('brnn', p)

  def _PadSequenceToLength(self, t_input, length, pad_value):
    t_input = py_utils.with_dependencies(
        [py_utils.assert_less_equal(tf.shape(t_input)[0], length)], t_input)
    pad_shape = tf.concat([[length - tf.shape(t_input)[0]],
                           tf.shape(t_input)[1:]], 0)
    padding = tf.zeros(shape=pad_shape, dtype=t_input.dtype) + pad_value
    return tf.concat([t_input, padding], 0)

  def FProp(self, theta, inputs, paddings):
    """Compute bidi-RNN forward pass.

    rcell_forward is responsible for unrolling the sequence in the forward
    direction and rcell_backward in the backward direction. Output from forward
    and backward rnns are concatenated on the last output dim.

    See RNN.FProp() for more details.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A single tensor or a tuple of tensors with cardinality equal to
          rnn_cell.inputs_arity. For every input tensor, the first dimension is
          assumed to be time, second dimension batch, and third dimension depth.
      paddings: A tensor. First dim is time, second dim is batch, and third dim
          is expected to be 1.

    Returns:
      A tensor of [time, batch, dims].
    """
    p = self.params
    if not isinstance(inputs, (list, tuple)):
      inputs = [inputs]
    seq_len = tf.shape(paddings)[0]
    inputs = [
        self._PadSequenceToLength(x, p.sequence_length, 0) for x in inputs
    ]
    inputs_sequence = [
        list(tf.unstack(x, num=p.sequence_length)) for x in inputs
    ]
    if len(inputs_sequence) > 1:
      inputs_sequence = list(zip(inputs_sequence))
    else:
      inputs_sequence = inputs_sequence[0]
    paddings_sequence = list(
        tf.unstack(
            self._PadSequenceToLength(paddings, p.sequence_length, 1.0),
            num=p.sequence_length))
    assert len(inputs_sequence) == p.sequence_length
    assert len(paddings_sequence) == p.sequence_length
    out = self.brnn.FProp(theta.brnn, inputs_sequence, paddings_sequence)
    return tf.stack(out)[:seq_len,]


class CuDNNLSTM(base_layer.BaseLayer):
  """A single layer of unidirectional LSTM with Cudnn impl.

  Runs training with CuDNN on GPU, and eval using a FRNN with properly
  configured `LSTMCellSimple` cell.
  """

  @classmethod
  def Params(cls):
    p = super(CuDNNLSTM, cls).Params()
    p.Define('cell', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the RNN cell used in eval mode.')
    p.Define('reverse', False,
             'Whether or not to unroll the sequence in reversed order.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(CuDNNLSTM, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed input is currently not supported by '
                                'CuDNNLSTM')
    _AssertCellParamsCuDNNCompatible(p.cell)
    if not p.is_eval:
      # Use the cell's name as variable scope such that vars in train and eval
      # modes stay in the same scope.
      with tf.variable_scope(p.cell.name):
        cudnn_init_helper = cudnn_rnn_utils.CuDNNLSTMInitializer(
            p.cell.num_input_nodes, p.cell.num_output_nodes)
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
        self.vars.wb.approx_size = (
            cudnn_init_helper.weight_size + cudnn_init_helper.bias_size)
      # Create saveable in the outer_scope, thus saved canonicals are in
      # variable_scope: $outer_scope/p.cell.name
      saveable = cudnn_rnn_utils.CuDNNLSTMSaveable(
          self.vars.wb,
          p.cell.num_output_nodes,
          p.cell.num_input_nodes,
          p.cell.name,
          name=self.vars.wb.name + '_saveable')
      tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    else:
      frnn_p = FRNN.Params()
      frnn_p.name = 'rnn'
      frnn_p.cell = p.cell.Copy()
      frnn_p.reverse = p.reverse
      self.CreateChild('rnn', frnn_p)

  @property
  def rnn_cell(self):
    return self.rnn.cell

  def zero_state(self, batch_size):
    p = self.params
    if not p.is_eval:
      zero_m = tf.zeros([1, batch_size, p.cell.num_output_nodes], dtype=p.dtype)
      zero_c = tf.zeros([1, batch_size, p.cell.num_output_nodes], dtype=p.dtype)
      return py_utils.NestedMap(m=zero_m, c=zero_c)
    else:
      return self.rnn.cell.init_state(batch_size)

  def FProp(self, theta, inputs, paddings, state0=None):
    """Compute LSTM forward pass.

    Runs training pass with CuDNN. Eval is implemented without CuDNN to be
    compatible with different hardwares.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A tensor of [seq_length, batch_size, num_input_nodes]
      paddings: A tensor of [seq_length, batch_size, 1]
      state0: If not None, the initial rnn state in a `.NestedMap`. Defaults
        to the cell's zero-state.

    Returns:
      A tensor of [seq_length, batch_size, num_output_nodes] and the
      final recurrent state.  Because cudnn_rnn does not apply padding
      every step, the final state returned differs from other RNN
      layers even if the inputs, paddings and state0 given are
      identical.
    """
    p = self.params
    if p.is_eval:
      return self.rnn.FProp(theta.rnn, inputs, paddings, state0)

    if not state0:
      batch_dim = 1
      state0 = self.zero_state(tf.shape(paddings)[batch_dim])
    state_h, state_c = state0.m, state0.c
    if p.reverse:
      inputs = py_utils.ReversePaddedSequence(inputs, paddings)
    output, output_h, output_c = cudnn_rnn_ops.cudnn_lstm(
        inputs=inputs,
        input_h=state_h,
        input_c=state_c,
        params=theta.wb,
        is_training=True,
        input_mode='linear_input',
        direction='unidirectional',
        dropout=0.0)

    if p.reverse:
      output = py_utils.ReversePaddedSequence(output, paddings)
    return output, py_utils.NestedMap(m=output_h, c=output_c)


class BidirectionalNativeCuDNNLSTM(base_layer.BaseLayer):
  """A single layer of bidirectional LSTM with native Cudnn impl.
  """

  @classmethod
  def Params(cls):
    p = super(BidirectionalNativeCuDNNLSTM, cls).Params()
    p.Define('fwd', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the forward RNN cell used in eval mode.')
    p.Define('bak', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the backward RNN cell used in eval mode.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BidirectionalNativeCuDNNLSTM, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('Packed input is currently not supported by '
                                'BidirectionalNativeCuDNNLSTM')
    assert p.fwd.num_input_nodes == p.bak.num_input_nodes
    assert p.fwd.num_output_nodes == p.bak.num_output_nodes
    _AssertCellParamsCuDNNCompatible(p.fwd)
    _AssertCellParamsCuDNNCompatible(p.bak)
    if not p.is_eval:
      with tf.variable_scope(p.name):
        cudnn_init_helper = cudnn_rnn_utils.CuDNNLSTMInitializer(
            p.fwd.num_input_nodes,
            p.fwd.num_output_nodes,
            direction=cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION)
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
        self.vars.wb.approx_size = (
            cudnn_init_helper.weight_size + cudnn_init_helper.bias_size)
      # Create saveable in the outer_scope, thus saved canonicals are in
      # variable_scope: $outer_scope/p.cell.name
      saveable = cudnn_rnn_utils.BidiCuDNNLSTMSaveable(
          self.vars.wb,
          p.fwd.num_output_nodes,
          p.fwd.num_input_nodes,
          p.fwd.name,
          p.bak.name,
          name=self.vars.wb.name + '_saveable')
      tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    else:
      bidi_frnn_p = BidirectionalFRNN.Params()
      bidi_frnn_p.name = 'rnn'
      bidi_frnn_p.fwd = p.fwd.Copy()
      bidi_frnn_p.bak = p.bak.Copy()
      self.CreateChild('rnn', bidi_frnn_p)

  def zero_state(self, batch_size):
    p = self.params
    if not p.is_eval:
      zero_m = tf.zeros(
          [2, batch_size, p.fwd.num_output_nodes], dtype=p.fwd.dtype)
      zero_c = tf.zeros(
          [2, batch_size, p.fwd.num_output_nodes], dtype=p.fwd.dtype)
      return py_utils.NestedMap(m=zero_m, c=zero_c)
    else:
      fwd = self.fwd_rnn.cell.zero_state(batch_size)
      bak = self.bak_rnn.cell.zero_state(batch_size)
      return py_utils.NestedMap(
          m=tf.stack([fwd.m, bak.m], axis=0),
          c=tf.stack([fwd.c, bak.c], axis=0))

  def FProp(self, theta, inputs, paddings):
    """Compute bidirectional LSTM forward pass.

    Runs training pass with CuDNN. Eval is implemented without CuDNN to be
    compatible with different hardwares.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A tensor of [seq_length, batch_size, num_input_nodes]
      paddings: A tensor of [seq_length, batch_size, 1]
    Returns:
      A tensor of [seq_length, batch_size, 2 * num_output_nodes]

    See `DRNN.FProp()` for more details.
    """
    p = self.params
    if p.is_eval:
      return self.rnn.FProp(theta.rnn, inputs, paddings)

    with tf.name_scope(p.name):
      batch_dim = 1
      state0 = self.zero_state(tf.shape(inputs)[batch_dim])
      output, _, _ = cudnn_rnn_ops.cudnn_lstm(
          inputs=inputs,
          input_h=state0.m,
          input_c=state0.c,
          params=theta.wb,
          is_training=True,
          input_mode='linear_input',
          direction='bidirectional',
          dropout=0.0)
      return output


def _ConcatLastDim(*args):
  """Concatenates all args along the last dimension."""
  return tf.concat(args, tf.rank(args[0]) - 1)


def _ShiftRight(x0, xs):
  """Shifts xs[:-1] one step to the right and attaches x0 on the left."""
  return tf.concat([[x0], xs[:-1]], axis=0)


def _ShiftRightWithMasking(x0, xs, mask):
  """Shifts xs[:-1] one step to the right and attaches x0 on the left."""
  return tf.concat([[x0], xs[:-1] * mask[1:]], axis=0)


class FRNNWithAttention(base_layer.BaseLayer):
  """An RNN layer intertwined with an attention layer."""

  @classmethod
  def Params(cls):
    p = super(FRNNWithAttention, cls).Params()
    p.Define('cell', rnn_cell.LSTMCellSimple.Params(),
             'Configs for the RNN cell.')
    p.Define('attention', attention.AdditiveAttention.Params(),
             'Attention used by this layer.')
    p.Define(
        'output_prev_atten_ctx', False,
        'If True, output previous attention context for each position.'
        'Otherwise, output current attention context.')
    p.Define(
        'use_zero_atten_state', False,
        'To use zero attention state instead of computing attention with '
        'zero query vector.')
    p.Define('atten_context_dim', 0, 'Size of attention context.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(FRNNWithAttention, self).__init__(params)
    p = self.params
    if p.use_zero_atten_state:
      assert p.atten_context_dim > 0, (
          'atten_context_dim needs to be set when '
          'initializing attention state and context with 0.')
    if p.packed_input:
      assert p.use_zero_atten_state, (
          'Packed input is only supported when '
          'training with zero initial attention states.')
    p.cell.reset_cell_state = p.packed_input
    self.CreateChild('cell', p.cell)
    p.attention.packed_input = p.packed_input
    # Set p.attention.atten_dropout_deterministic to True by default.
    p.attention.atten_dropout_deterministic = True
    self.CreateChild('atten', p.attention)

  @property
  def rnn_cell(self):
    return self.cell

  @property
  def attention(self):
    return self.atten

  def InitForSourcePacked(self,
                          theta,
                          src_encs,
                          src_paddings,
                          src_contexts=None,
                          src_segment_id=None):
    """A wrapper of InitForSourcePacked of child attention layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      src_encs: A tensor of shape [source_seq_length, batch_size, source_dim].
      src_paddings: A tensor of shape [source_seq_length, batch_size].
      src_contexts: [Optional] If specified, must be a tensor of shape
        [source_seq_length, batch_size, some_dim]. When specified, this tensor
        will be used as the source context vectors when computing attention
        context, and src_ends will be only used to compute the attention score
        for each context. If set to None, the 'src_encs' will be used as
        source context.
      src_segment_id: A tensor of shape [source_seq_length, batch_size], to
        support packed inputs.

    Returns:
      packed_src - A `.NestedMap` containing packed source.
    """
    atten = self.atten

    if src_contexts is None:
      src_contexts = src_encs

    if self.params.packed_input:
      assert src_segment_id is not None

    # Initial attention state.
    return atten.InitForSourcePacked(
        theta=theta.atten,
        source_vecs=src_encs,
        source_contexts=src_contexts,
        source_padding=src_paddings,
        source_segment_id=src_segment_id)

  def zero_state(self,
                 theta,
                 src_encs,
                 packed_src,
                 batch_size,
                 atten_state_dim=1):
    """Initial state of this layer.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      src_encs: A tensor of shape [source_seq_length, batch_size, source_dim].
      packed_src: A `.NestedMap` containing packed source.
      batch_size: Batch size.
      atten_state_dim: Attention state dim when using zero_atten_state

    Returns:
      state0 - A `.NestedMap` containing initial states of RNN and attention.
    """

    p = self.params
    atten = self.atten
    # Initial RNN states.
    state0 = py_utils.NestedMap(rnn=self.cell.zero_state(batch_size))

    s_seq_len = tf.shape(src_encs)[0]

    zero_atten_state = atten.ZeroAttentionState(s_seq_len, batch_size)
    state0.step_state = py_utils.NestedMap(global_step=py_utils.GetGlobalStep())
    if p.use_zero_atten_state:
      zero_atten_context = tf.zeros([batch_size, p.atten_context_dim],
                                    dtype=py_utils.FPropDtype(p))
      state0.atten = zero_atten_context
      state0.atten_state = zero_atten_state
      state0.atten_probs = tf.zeros([batch_size, s_seq_len],
                                    dtype=py_utils.FPropDtype(p))
    else:
      state0.atten, state0.atten_probs, state0.atten_state = (
          atten.ComputeContextVectorWithSource(
              theta.atten, packed_src,
              tf.zeros([batch_size, p.cell.num_output_nodes],
                       dtype=self.cell.params.dtype), zero_atten_state))
    return state0

  def reset_atten_state(self, theta, state, inputs):
    state.atten = inputs.reset_mask * state.atten
    state.atten_state = inputs.reset_mask * state.atten_state
    state.atten_probs = inputs.reset_mask * state.atten_probs
    return state

  def AccumulateStates(self,
                       theta,
                       src_encs,
                       src_paddings,
                       inputs,
                       paddings,
                       src_contexts=None,
                       state0=None,
                       src_segment_id=None,
                       segment_id=None):
    """Sets up and runs the recurrence, returning the raw accumulated states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      src_encs: A tensor of shape [source_seq_length, batch_size, source_dim].
      src_paddings: A tensor of shape [source_seq_length, batch_size].
      inputs: A tensor of [time, batch, dims].
      paddings: A tensor of [time, batch, 1].
      src_contexts: [Optional] If specified, must be a tensor of shape
        [source_seq_length, batch_size, some_dim]. When specified, this tensor
        will be used as the source context vectors when computing attention
        context, and src_ends will be only used to compute the attention score
        for each context. If set to None, the 'src_encs' will be used as source
        context.
      state0: [Optional] If not None, the initial rnn state and attention
        context in a `.NestedMap`. Defaults to the cell's zero-state.
      src_segment_id: A tensor of shape [source_seq_length, batch_size] to
        support masking with packed inputs.
      segment_id: A tensor of [time, batch, 1].

    Returns:
      accumulated_state: a NestedMap of accumulated states from the recurrence.
      final_state: final_state: The final recurrent state.
      side_info: Nested map of intermediate results needed for post-processing
    """
    p = self.params
    dtype = p.dtype
    rcell = self.cell
    atten = self.atten
    assert dtype == rcell.params.dtype
    assert dtype == atten.params.dtype
    assert rcell.params.inputs_arity == 1 or rcell.params.inputs_arity == 2
    if segment_id is None:
      segment_id = tf.zeros_like(paddings)

    inputs_shape = tf.shape(inputs)
    _, batch = inputs_shape[0], inputs_shape[1]
    input_dim = tf.shape(inputs)[2]

    packed_src = self.InitForSourcePacked(theta, src_encs, src_paddings,
                                          src_contexts, src_segment_id)
    if state0 is None:
      zero_atten_state_dim = p.cell.num_input_nodes - input_dim
      state0 = self.zero_state(theta, src_encs, packed_src, batch,
                               zero_atten_state_dim)
    else:
      assert not p.packed_input, ('packed input is only supported with default '
                                  'initial states.')

    def CellFn(theta, state0, inputs):
      """Computes one step forward."""
      if p.packed_input:
        state0_mod = state0.DeepCopy()
        state0_mod = self.reset_atten_state(theta, state0_mod, inputs)
      else:
        state0_mod = state0
      state1 = py_utils.NestedMap(step_state=state0_mod.step_state)
      if rcell.params.inputs_arity == 1:
        act = [_ConcatLastDim(inputs.act, state0_mod.atten)]
      else:
        act = [inputs.act, state0_mod.atten]
      state1.rnn, _ = rcell.FProp(
          theta.rnn, state0_mod.rnn,
          py_utils.NestedMap(
              act=act, padding=inputs.padding, reset_mask=inputs.reset_mask))

      state1.atten, state1.atten_probs, state1.atten_state = (
          atten.ComputeContextVectorWithSource(
              theta.atten,
              theta.packed_src,
              rcell.GetOutput(state1.rnn),
              state0_mod.atten_state,
              query_segment_id=tf.squeeze(inputs.segment_id, 1)))
      return state1, py_utils.NestedMap()

    if p.packed_input:
      reset_mask = _GeneratePackedInputResetMask(segment_id)
    else:
      reset_mask = tf.zeros_like(paddings)

    acc_state, final_state = recurrent.Recurrent(
        theta=py_utils.NestedMap(
            rnn=theta.cell,
            packed_src=packed_src,
            atten=theta.atten,
            global_step=theta.global_step),
        state0=state0,
        inputs=py_utils.NestedMap(
            act=inputs,
            padding=paddings,
            reset_mask=reset_mask,
            segment_id=segment_id),
        cell_fn=CellFn,
        accumulator_layer=self,
        allow_implicit_capture=p.allow_implicit_capture)

    side_info = py_utils.NestedMap(state0=state0, reset_mask=reset_mask)
    return acc_state, final_state, side_info

  def PostProcessStates(self, acc_state, side_info):
    """Post-process accumulated states to fulfill FProp's interface.

    Args:
      acc_state: a NestedMap of the raw accumulated states from the recurrence.
      side_info: side-information collected by AccumulateStates.

    Returns:
      A tuple (atten_context, rnn_output, atten_probs).

      - atten_context: a tensor of [time, batch, attention.context_dim].
      - rnn_output: a tensor of [time, batch, rcell.num_output_nodes].
      - atten_probs: a tensor of [time, batch, source_seq_length].
    """

    p = self.params
    rcell = self.cell
    state0 = side_info.state0
    reset_mask = side_info.reset_mask
    if p.output_prev_atten_ctx:
      # Add the initial attention context in and drop the attention context
      # in the last position so that the output atten_ctx is previous
      # attention context for each target position.
      if p.packed_input:
        # Note: Assumes first element of mask is padding, as generated by
        # _GeneratePackedInputResetMask
        atten_ctx = _ShiftRightWithMasking(state0.atten, acc_state.atten,
                                           reset_mask)
        atten_probs = _ShiftRightWithMasking(state0.atten_probs,
                                             acc_state.atten_probs, reset_mask)
      else:
        atten_ctx = _ShiftRight(state0.atten, acc_state.atten)
        atten_probs = _ShiftRight(state0.atten_probs, acc_state.atten_probs)
    else:
      atten_ctx = acc_state.atten
      atten_probs = acc_state.atten_probs

    return atten_ctx, rcell.GetOutput(acc_state.rnn), atten_probs

  def FProp(self,
            theta,
            src_encs,
            src_paddings,
            inputs,
            paddings,
            src_contexts=None,
            state0=None,
            src_segment_id=None,
            segment_id=None):
    """Forward propagate through a rnn layer with attention.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      src_encs: A tensor of shape [source_seq_length, batch_size, source_dim].
      src_paddings: A tensor of shape [source_seq_length, batch_size].
      inputs: A tensor of [time, batch, dims].
      paddings: A tensor of [time, batch, 1].
      src_contexts: [Optional] If specified, must be a tensor of shape
        [source_seq_length, batch_size, some_dim]. When specified, this tensor
        will be used as the source context vectors when computing attention
        context, and src_ends will be only used to compute the attention score
        for each context. If set to None, the 'src_encs' will be used as source
        context.
      state0: [Optional] If not None, the initial rnn state and attention
        context in a `.NestedMap`. Defaults to the cell's zero-state.
      src_segment_id: A tensor of shape [source_seq_length, batch_size] to
        support masking with packed inputs.
      segment_id: A tensor of [time, batch, 1].

    Returns:
      A tuple (atten_context, rnn_output, atten_probs, final_state).

      - atten_context: a tensor of [time, batch, attention.context_dim].
      - rnn_output: a tensor of [time, batch, rcell.num_output_nodes].
      - atten_probs: a tensor of [time, batch, source_seq_length].
      - final_state: The final recurrent state.
    """

    acc_state, final_state, side_info = self.AccumulateStates(
        theta, src_encs, src_paddings, inputs, paddings, src_contexts, state0,
        src_segment_id, segment_id)

    atten_context, rnn_output, atten_probs = self.PostProcessStates(
        acc_state, side_info)
    return atten_context, rnn_output, atten_probs, final_state


class MultiSourceFRNNWithAttention(base_layer.BaseLayer):
  """RNN layer intertwined with an attention layer for multiple sources.

  Allows different attention params per source, if attention is not shared.
  """

  @classmethod
  def Params(cls):
    """Params for this MultiSourceFRNNWithAttention class."""
    p = super(MultiSourceFRNNWithAttention, cls).Params()
    p.Define(
        'cell',
        rnn_cell.LSTMCellSimple.Params().Set(
            params_init=py_utils.WeightInit.Uniform(0.04)),
        'Configs for the RNN cell.')
    p.Define(
        'attention_tpl', attention.AdditiveAttention.Params(),
        'Attention used by this attention layer, can be overridden by '
        'source_name_to_attention_params.')
    p.Define(
        'atten_merger', layers_with_attention.MergerLayer.Params(),
        'Merger layer config for combining context vectors computed for '
        'different source encodings.')
    p.Define('source_names', None, 'List of source names.')
    p.Define('share_attention', False, 'If set single attention layer shared.')
    p.Define(
        'source_name_to_attention_params', None,
        'Can be set if share_attention is False. Allows defining '
        'different attention params per source in a dictionary, eg. '
        '{"src1": atten_tpl1, "src2": atten_tpl2}')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @property
  def rnn_cell(self):
    """Reference to the RNN cell of this layer."""
    return self.cell

  @property
  def attention(self):
    """Reference to the attention layer(s) of this layer."""
    return self.attentions

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a MultiSourceFRNNWithAttention layer with params."""
    super(MultiSourceFRNNWithAttention, self).__init__(params)
    p = self.params
    assert not p.packed_input, ('packed input is not supported for '
                                'MultiSourceFRNNWithAttention')
    if p.atten_merger is None:
      raise ValueError('Merger layer cannot be none!')
    if not isinstance(p.source_names, list) or not p.source_names:
      raise ValueError('Source names must be a non-empty list.')
    if p.share_attention and p.source_name_to_attention_params:
      raise ValueError(
          'Cant specify source_name_to_attention_params with share_attention.')
    self.CreateChild('cell', p.cell)

    # Initialize attention layer(s).
    params_atten = []
    self._source_dims = []
    src_to_att = p.source_name_to_attention_params
    for src_name in p.source_names:
      if src_to_att and src_name in src_to_att:
        att_params = src_to_att[src_name]
        att_params.name = 'atten_%s' % src_name
      else:
        att_params = p.attention_tpl.Copy()
        if p.share_attention:
          att_params.name = 'atten_shared'
        else:
          att_params.name = 'atten_%s' % (src_name)
      if att_params.params_init is None:
        att_params.params_init = py_utils.WeightInit.Gaussian(
            1. / math.sqrt(att_params.source_dim + att_params.query_dim))
      att_params.atten_dropout_deterministic = True
      params_atten.append(att_params)
      self._source_dims.append(att_params.source_dim)
      if p.share_attention:
        break
    self.CreateChildren('attentions', params_atten)

    # Initialize merger layer for attention layer(s).
    params = p.atten_merger.Copy()
    params.name = 'atten_merger'
    self.CreateChild('atten_merger', params)

  def InitAttention(self, theta, src_encs, src_paddings, batch_size):
    """Computes initial states for attention layer(s).

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      src_encs: A `.NestedMap` object containing source encoding tensors,
        each of shape [source_seq_length, batch_size, source_dim]. Children
        names of the `.NestedMap` is defined by source_names.
      src_paddings: A `.NestedMap` object contraining source padding tensors,
        each of shape [source_seq_length, batch_size]. Children names of the
        `.NestedMap` is defined by source_names.
      batch_size: Scalar Tensor of type int, for initial state shape.

    Returns:
      state0 - Initial attention-rnn state in a `.NestedMap`. Zeros for the rnn
      initial state, and merger output for attention initial state.

      Transformed source vectors and transposed source vectors.
    """
    p = self._params
    dtype = p.dtype
    rcell = self.cell

    # Initial RNN states, theta and auxiliary variables.
    state0 = py_utils.NestedMap(rnn=rcell.zero_state(batch_size))
    query_vec0 = tf.zeros([batch_size, p.cell.num_output_nodes], dtype)

    ctxs0 = []
    packed_srcs = py_utils.NestedMap()
    state0.step_state = py_utils.NestedMap(global_step=py_utils.GetGlobalStep())
    for i, src_name in enumerate(p.source_names):
      att_idx = (0 if p.share_attention else i)

      packed_srcs[src_name] = self.attentions[att_idx].InitForSourcePacked(
          theta.attentions[att_idx], src_encs[src_name], src_encs[src_name],
          src_paddings[src_name])

      # Initial attention state.
      s_seq_len = tf.shape(src_encs[src_name])[0]
      zero_atten_state = self.attentions[att_idx].ZeroAttentionState(
          s_seq_len, batch_size)
      ctxs0.append(self.attentions[att_idx].ComputeContextVectorWithSource(
          theta.attentions[att_idx], packed_srcs[src_name], query_vec0,
          zero_atten_state)[0])

    # Initial attention state is the output of merger-op.
    state0.atten = self.atten_merger.FProp(theta.atten_merger, ctxs0,
                                           query_vec0)
    return state0, packed_srcs

  def FProp(self, theta, src_encs, src_paddings, inputs, paddings):
    """Forward propagate through a RNN layer with attention(s).

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      src_encs: A `.NestedMap` object containing source encoding tensors, each
        of shape [source_seq_length, batch_size, source_dim]. Children names of
        the `.NestedMap` is defined by source_names.
      src_paddings: A `.NestedMap` object contraining source padding tensors,
        each of shape [source_seq_length, batch_size]. Children names of the
        `.NestedMap` is defined by source_names.
      inputs: A tensor of [time, batch, dims].
      paddings: A tensor of [time, batch, 1].

    Returns:
      A tuple (atten_context, rnn_output)

      - atten_context: a tensor of [time, batch, attention.hidden_dim].
      - rnn_output: a tensor of [time, batch, p.cell.num_output_nodes].

    Raises:
      ValueError: dtype mismatch of attention layers.
    """
    p = self.params
    dtype = p.dtype
    rcell = self.cell
    attentions = self.attentions
    assert rcell.params.inputs_arity == 1 or rcell.params.inputs_arity == 2
    assert dtype == rcell.params.dtype
    for atten in attentions:
      if dtype != atten.params.dtype:
        raise ValueError('Data type mismatch!')

    # Check if all batch sizes and depths match for source encs and paddings.
    src_name_0 = p.source_names[0]
    src_encs[src_name_0] = py_utils.with_dependencies([
        py_utils.assert_shape_match(
            [tf.shape(src_encs[src_name_0])[1], source_dim],
            tf.shape(src_encs[src_name_i])[-2:])
        for src_name_i, source_dim in zip(p.source_names, self._source_dims)
    ], src_encs[src_name_0])
    src_paddings[src_name_0] = py_utils.with_dependencies([
        py_utils.assert_equal(
            tf.shape(src_paddings[src_name_0])[-1],
            tf.shape(src_paddings[src_name_i])[-1])
        for src_name_i in p.source_names[1:]
    ], src_paddings[src_name_0])

    # Compute source transformations and initial rnn states.
    state0, packed_src = self.InitAttention(theta, src_encs, src_paddings,
                                            tf.shape(inputs)[1])

    # Collect individual attention parameters for CellFn.
    attens_theta = py_utils.NestedMap({
        src_name: theta.attentions[0 if p.share_attention else i]
        for i, src_name in enumerate(p.source_names)
    })

    def CellFn(theta, state0, inputs):
      """Computes one step forward."""
      state1 = py_utils.NestedMap(step_state=state0.step_state)
      state1.rnn, _ = rcell.FProp(
          theta.rnn, state0.rnn,
          py_utils.NestedMap(
              act=[_ConcatLastDim(inputs.act, state0.atten)],
              padding=inputs.padding))

      # The ordering in local_ctxs follows p.source_names.
      local_ctxs = []
      query_vec = rcell.GetOutput(state1.rnn)
      for i, src_name in enumerate(p.source_names):
        att_idx = (0 if p.share_attention else i)
        local_ctxs.append(attentions[att_idx].ComputeContextVectorWithSource(
            theta.attens[src_name],
            theta.packed_src[src_name],
            query_vec,
            state0.atten,
        )[0])
      state1.atten = self.atten_merger.FProp(theta.atten_merger, local_ctxs,
                                             query_vec)
      return state1, py_utils.NestedMap()

    # Note that, we have a NestedMap for each parameter.
    acc_state, _ = recurrent.Recurrent(
        theta=py_utils.NestedMap(
            rnn=theta.cell,
            attens=attens_theta,
            packed_src=packed_src,
            atten_merger=theta.atten_merger),
        state0=state0,
        inputs=py_utils.NestedMap(act=inputs, padding=paddings),
        cell_fn=CellFn,
        accumulator_layer=self,
        allow_implicit_capture=p.allow_implicit_capture)

    return acc_state.atten, rcell.GetOutput(acc_state.rnn)


class BidirectionalFRNNQuasi(base_layer.BaseLayer):
  """Bidirectional functional Quasi-RNN.

  This is very similar to BidirectionalFRNN except the input is a list of the
  forward and backward inputs. It is split because quasi-rnns do the
  matrix/convolution unrolled over time outside of the recurrent part. Also,
  this uses quasi-rnn instead of LSTM.
  """

  @classmethod
  def Params(cls):
    p = super(BidirectionalFRNNQuasi, cls).Params()
    p.Define('fwd', rnn_cell.QRNNPoolingCell.Params(),
             'Configs for the forward RNN cell.')
    p.Define('bak', rnn_cell.QRNNPoolingCell.Params(),
             'Configs for the backward RNN cell.')
    p.Define('packed_input', False, 'To reset states for packed inputs.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BidirectionalFRNNQuasi, self).__init__(params)
    p = params
    assert not p.packed_input, ('packed input is not supported for '
                                'BidirectionalFRNNQuasi')
    params_forward = FRNN.Params()
    params_forward.name = 'fwd'
    params_forward.dtype = p.dtype
    params_forward.reverse = False
    params_forward.cell = p.fwd.Copy()
    self.CreateChild('fwd_rnn', params_forward)

    params_backward = FRNN.Params()
    params_backward.name = 'bak'
    params_backward.dtype = p.dtype
    params_backward.reverse = True
    params_backward.cell = p.bak.Copy()
    self.CreateChild('bak_rnn', params_backward)

  def FProp(self, theta, inputs, paddings):
    """Compute bidi-quasi-RNN forward pass.

    fwd_rnn unroll the sequence in the forward direction and
    bak_rnn unroll the sequence in the backward direction. The
    outputs are concatenated in the last output dim.

    See `FRNN.FProp` for more details.

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      inputs: A list of the fwd and bak tensors. Each item in the list should be
        A single tensor or a tuple of tensors with cardinality equal to
        rnn_cell.inputs_arity. For every input tensor, the first dimension is
        assumed to be time, second dimension batch, and third dimension depth.
      paddings: A tensor. First dim is time, second dim is batch, and third dim
          is expected to be 1.

    Returns:
      A tensor of [time, batch, dims].
    """
    p = self.params
    with tf.name_scope(p.name):
      cluster = self.cluster
      fwd_device = cluster.WorkerDeviceInModelSplit(0)
      bwd_device = cluster.WorkerDeviceInModelSplit(1)
      with tf.device(fwd_device):
        output_forward, _ = self.fwd_rnn.FProp(theta.fwd_rnn, inputs[0],
                                               paddings)
      with tf.device(bwd_device):
        output_backward, _ = self.bak_rnn.FProp(theta.bak_rnn, inputs[1],
                                                paddings)
        output_forward = py_utils.HasShape(output_forward,
                                           tf.shape(output_backward))
        out_rank = tf.rank(output_forward) - 1
        # TODO(yonghui/zhifengc): In the current implementation, we copy
        # output_forward from gpu:0 to gpu:1, and then copy the concatenated
        # output from gpu:1 to gpu:0 to enable next layer computation. It might
        # be more efficient to only copy output_backward from gpu:1 to gpu:0 to
        # reduce cross-gpu data transfer.
        return tf.concat([output_forward, output_backward], out_rank)
