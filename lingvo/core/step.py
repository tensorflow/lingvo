# Lint as: python2, python3
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
"""An abstract layer for processing sequences step-by-step.

E.g.::

  def ProcessSeq(step, external_inputs, input_batch):
    prepared_external_inputs = step.PrepareExternalInputs(
        step.theta, external_inputs)
    batch_size, T = tf.shape(input_batch.paddings)[:2]
    state = step.ZeroState(
        step.theta, prepared_external_inputs, batch_size)
    for t in range(T):
      step_inputs = input_batch.Transform(lambda x: x[:, i, ...])
      step_outputs, state = step.FProp(
          step.theta, prepared_external_inputs, step_inputs, state)
      (processing step_outputs...)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo.core import base_layer


class Step(base_layer.BaseLayer):
  """A layer that processes input sequences step-by-step.

  This can be seen as an RNNCell extended with optional external inputs.
  """

  def PrepareExternalInputs(self, theta, external_inputs):
    """Returns the prepared external inputs, e.g., packed_src for attention."""
    raise NotImplementedError(type(self))

  def ZeroState(self, theta, external_inputs, batch_size):
    """Returns the initial state given external inputs and batch size.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      external_inputs: External inputs returned by PrepareExternalInputs().
      batch_size: An int scalar representing the batch size of per-step inputs.

    Returns:
      A `.NestedMap` representing the initial state, which can be passed to
      FProp() for processing the first time step.
    """
    raise NotImplementedError(type(self))

  def FProp(self, theta, external_inputs, step_inputs, padding, state0):
    """Forward function.

    step_inputs, state0, step_outputs, and state1 should each be a `.NestedMap`
    of tensor values. Each tensor must be of shape [batch_size ...]. The
    structure of NestedMaps are determined by the implementation. state0 and
    state1 must have exactly the same structure and tensor shapes.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      external_inputs: External inputs returned by PrepareExternalInputs().
      step_inputs: The inputs for this time step.
      padding: A 0/1 float tensor of shape [batch_size]; 1.0 means that this
        batch element is empty in this step.
      state0: The previous recurrent state.

    Returns:
      A tuple (step_outputs, state1).
      - outputs: The outputs of this step.
      - state1: The next recurrent state.
    """
    raise NotImplementedError(type(self))
