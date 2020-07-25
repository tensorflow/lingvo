# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Step APIs for RNN layers."""

from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import step


class RnnStep(step.Step):
  """A step containing an RNNCell."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('cell', rnn_cell.LSTMCellSimple.Params(),
             'Params for the RNN cell.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = params
    self.CreateChild('cell', p.cell)

  def PrepareExternalInputs(self, theta, external_inputs):
    """Does not modify the external_inputs parameter.

    This parameter, if provided, is assumed to be a vector that should be
    concatenated with the other vectors in step_inputs.inputs.

    Args:
      theta: unused.
      external_inputs: Either a tensor or None.

    Returns:
      external_inputs, unmodified.
    """
    return external_inputs

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Returns the zero_state for the RNN cell.

    Args:
      theta: Variables used by the RNNCell.
      prepared_inputs: unused.
      batch_size: An int scalar representing the batch size of per-step inputs.

    Returns:
      The zero state of the RNNCell.
    """
    return self.cell.zero_state(theta.cell, batch_size)

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Performs one inference step on the RNN cell.

    If external_inputs is not None, it is added as another act input
    to the RNNCell.

    Args:
      theta: Variables used by the RNNCell.
      prepared_inputs: If not None, concatenated with step_inputs.input. A
        tensor of shape [batch_size, external_input_dim].
      step_inputs: A NestedMap containing an 'input' list of [batch_size, dim]
        where the sum of dim (including external_inputs) is
        p.cell.num_input_nodes.
      padding: A 0/1 float tensor of shape [batch_size]; 1.0 means that this
        batch element is empty in this step.
      state0: A NestedMap of state, either produced by ZeroState or a previous
        invocation of FProp.

    Returns:
      (output, state1), where output is the cell output (GetOutput(state1))
      of shape [batch_size, p.cell.num_output_nodes], and state1 is the cell's
      recurrent state.
    """
    cell_inputs = py_utils.NestedMap(act=step_inputs.inputs)
    # An empty NestedMap can act as a None value here.
    if prepared_inputs is not None and not isinstance(prepared_inputs,
                                                      py_utils.NestedMap):
      cell_inputs.act.append(prepared_inputs)
    cell_inputs.padding = padding
    state1, extra = self.cell.FProp(theta.cell, state0, cell_inputs)
    return py_utils.NestedMap(
        output=self.cell.GetOutput(state1), extra=extra,
        padding=padding), state1


class RnnStackStep(step.Step):
  """A stack of RnnSteps.

  Three types of inputs are supported:
    step_inputs.input: This is the standard input. It is expected to change
      on every step of the sequence, and it is fed only to the first layer.
    step_inputs.context: This input changes for each step of the sequence, but
      is fed to every layer.
    external_inputs: This input is fixed at the beginning of the sequence.
      It is fed to every layer.

  Residual connections are also supported. When residual_start >= 0, the output
  of layer i (i >= residual_start) is added to the output of layer
  i - residual_stride.
  """

  @classmethod
  def Params(cls):
    """Constructs Params for an RnnStackStep."""
    p = super().Params()
    p.Define(
        'rnn_cell_tpl', rnn_cell.LSTMCellSimple.Params(),
        'RNNCell params template. '
        'Can be a single param or '
        'a list of rnn_layers params, one for each layer.')
    p.Define(
        'external_input_dim', 0, 'Size of the external input. '
        'The external input is given at the start of the sequence '
        'and is given to every layer at every step.')
    p.Define(
        'step_input_dim', 0, 'Size of the step input. '
        'This input is only given to the first layer and is expected to '
        'be different for each step.')
    p.Define(
        'context_input_dim', 0, 'Size of the context input. '
        'This input is given to every layer and is expected to be '
        'different for each step.')
    p.Define(
        'rnn_cell_dim', 0, 'Size of the rnn cells. '
        'This may be overridden by parameters set in rnn_cell_tpl.')
    p.Define(
        'rnn_cell_hidden_dim', 0, 'internal size of the rnn cells. When '
        'set to > 0 it enables a projection layer at the output of the '
        'rnn cell. This may be overridden by parameters set in rnn_cell_tpl.')
    p.Define('rnn_layers', 1, 'Number of rnn layers.')
    p.Define(
        'residual_start', -1,
        'Start residual connections from this layer. For this and higher '
        'layers, the layer output is the sum of the RNN cell output and '
        'input; if the layer also normalizes its output, then the '
        'normalization is done over this sum. Set to -1 to disable '
        'residual connections.')
    p.Define('residual_stride', 1,
             'Number of lstm layers to skip per residual connection.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = params
    sub = []

    # Users can either provide a single rnn_cell_tpl or one per layer.
    # If only one is provided, we replicate it for each layer.
    rnn_cell_tpls = p.rnn_cell_tpl
    if not isinstance(rnn_cell_tpls, list):
      rnn_cell_tpls = [p.rnn_cell_tpl] * p.rnn_layers

    # We may provide up to three tensors as input to the RnnStep:
    # the normal input, the context input (from step_inputs.context),
    # and the external input (from external_inputs).
    arity = 1
    if p.context_input_dim:
      arity += 1
    if p.external_input_dim:
      arity += 1
    extra_dim = p.context_input_dim + p.external_input_dim

    # The first layer's input comes from step_inputs.input. Later layers
    # will get their inputs from the previous layer's output.
    input_nodes = p.step_input_dim
    for i in range(p.rnn_layers):
      step_i = RnnStep.Params()
      step_i.name = 'rnn_%d' % i
      step_i.cell = rnn_cell_tpls[i].Copy()
      step_i.cell.num_input_nodes = input_nodes + extra_dim
      step_i.cell.inputs_arity = arity
      # The dimensions of each cell may be specified in the cell template
      # but most users will specify them in the stack params.
      if step_i.cell.num_output_nodes == 0:
        step_i.cell.num_output_nodes = p.rnn_cell_dim
      if step_i.cell.num_hidden_nodes == 0:
        step_i.cell.num_hidden_nodes = p.rnn_cell_hidden_dim
      input_nodes = step_i.cell.num_output_nodes
      sub.append(step_i)

    stack_params = step.StackStep.Params()
    stack_params.name = p.name
    stack_params.sub = sub
    stack_params.residual_start = p.residual_start
    stack_params.residual_stride = p.residual_stride
    self.CreateChild('stack', stack_params)

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    self.stack.InstantiateVariables()
    super()._CreateChildrenVariables()

  def PrepareExternalInputs(self, theta, external_inputs):
    """Delegates external inputs preparation to sub-layers.

    Args:
      theta: A `.NestedMap` object containing weight values of this layer and
        its children layers.
      external_inputs: A `.NestedMap` object. The structure of the internal
        fields is defined by the sub-steps.

    Returns:
      A `.NestedMap` containing a pre-processed version of the external_inputs,
      one per sub-step.
    """
    return self.stack.PrepareExternalInputs(theta.stack, external_inputs)

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Computes a zero state for each sub-step.

    Args:
      theta: A `.NestedMap` object containing weight values of this layer and
        its children layers.
      prepared_inputs: An output from PrepareExternalInputs.
      batch_size: The number of items in the batch that FProp will process.

    Returns:
      A `.NestedMap` containing a state0 object for each sub-step.
    """
    return self.stack.ZeroState(theta.stack, prepared_inputs, batch_size)

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Performs inference on the stack of sub-steps.

    See the documentation for StackStep for the particulars of passing context
    information to layers.

    Args:
      theta: A `.NestedMap` object containing weight values of this layer and
        its children layers.
      prepared_inputs: An output from PrepareExternalInputs.
      step_inputs: A `.NestedMap` containing a list called 'inputs', an
        optionally a tensor called 'context'.
      padding: A 0/1 float tensor of shape [batch_size]; 1.0 means that this
        batch element is empty in this step.
      state0: The previous recurrent state.

    Returns:
      (output, state1):

      - output: A `.NestedMap` containing the output of the top-most step.
      - state1: The recurrent state to feed to next invocation of this graph.
    """
    return self.stack.FProp(theta.stack, prepared_inputs, step_inputs, padding,
                            state0)
