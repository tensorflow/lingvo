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
"""GSPMD pipeline parallelism implementations."""

from typing import Any

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import recurrent

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
weight_params = py_utils.weight_params
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


# Ported from LayerwiseShardablePipelinedLayer in gshard_layers.py.
# TODO(zhangqiaorjc): Use jax.vmap to avoid requiring inputs to have a leading
# num_microbatch dim.
class LayerwiseShardablePipelined(base_layer.BaseLayer):
  """A layer that implements pipelining across stages.

  It creates a loop over microbatches around a loop-body layer. The wrapped body
  layer should have an explicit leading num_stages dimension in the input/output
  data (required) and weights (to achieve real pipeline parallelism).

  It can run on a single core, or sharded using GSPMD annotations. If the stage
  dimension is sharded, GSPMD will produce a cross-core pipelining pattern.

  Inputs to LayerwiseShardablePipelined should have a leading
  num_microbatch dimension. Each microbatch will be send to each pipeline loop
  iteration.

  The high-level idea is to use a shifting buffer to communicate between stages,
  as shown below (although the real implementation uses recurrent.Recurrent() to
  manage accumulation buffers)::

      # shape: [num_microbatches, ...]
      input = ...
      # Insert a num_stages dimension after num_microbatches, then pad to shape:
      #   [num_microbatches + num_stages - 1, num_stages, ...]
      padded_input = pad(expand_dim(input, 1), ...)

      # Shifting buffer
      state = jnp.zeros([num_stages, ...])

      # Recurrent loop
      for i in range(num_microbatches + num_stages - 1):
        # shift state to the right by one stage
        shifted_state = jnp.pad(state, [[1, 0], ...])[:-1]
        in_mask = jnp.equal(jnp.arange(num_stages), 0)
        stages_in = jnp.where(in_mask, padded_input[i],  shifted_state)
        state = stage_parallel_body.FProp(theta.body, stages_in)
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for LayerwiseShardablePipelined."""
    p = super().Params()
    p.Define('num_stages', 1, 'Number of pipeline stages.')
    p.Define(
        'stage_parallel_body', None,
        'The param for the pipelined body. Its input data should have '
        'a leading dimension that corresponds to num_stages, and its '
        'computation should be parallel along this dimension to achieve '
        'real pipeline parallelism.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructs a LayerwiseShardablePipelined object."""
    super().__init__(params)
    p = self.params
    assert p.name
    self.create_child('body', p.stage_parallel_body)

  def fprop(self, theta: NestedMap, inputs: NestedJTensor) -> Any:
    """FProp inputs through the pipeline body.

    self.body.fprop is expected to be of the following signature:
    outputs = self.body.fprop(theta, inputs)

    outputs are expected to be of the same structure as inputs.

    Args:
      theta: The combined layer params for all pipeline stages.
      inputs: A NestedMap of inputs that goes through the pipeline body.

    Returns:
      Output from the last pipeline stage.
    """
    p = self.params
    L = p.num_stages  # pylint: disable=invalid-name

    # Pad the leading num_microbatches dimension by num_stages - 1 to match
    # loop iteration count, which corresponds to the bubbles between forward
    # and backward passes.
    # [num_microbatches, ...] -> [num_microbatches + (num_stages - 1), ...].
    padded_inputs = jax.tree_map(
        lambda x: jnp.pad(x, [[0, L - 1]] + [[0, 0]] * (len(x.shape) - 1)),
        inputs)

    # The loop state has shape [num_stages, ...]
    # Inputs are not the loop state: they are not changed during the loop. The
    # state (shifting buffer) does not have a num_microbatches dimension.
    state0 = jax.tree_map(
        lambda x: jnp.zeros((L,) + x.shape[1:], dtype=x.dtype), padded_inputs)

    def _ScanFn(carry, inputs):
      in_state, inp = carry.data, inputs.data
      # Shift state to the right by 1.
      padding = [[1, 0]] + [[0, 0]] * (len(in_state.shape) - 1)
      shifted_state = jnp.pad(in_state, padding)[0:L, ...]
      # Bring in the next microbatch.
      iota = jax.lax.broadcasted_iota('int32', shifted_state.shape, 0)
      stages_in = jnp.where(iota == 0, inp, shifted_state)
      # Run through pipeline body.
      out_state = self.body.fprop(theta.body, stages_in)
      py_utils.assert_same_shape_and_dtype(stages_in, out_state)
      # Accumulator saves out_state for final output retrieval.
      return NestedMap(data=out_state), NestedMap(data=out_state)

    # Loop over num_microbatches + (num_stages - 1), where input to each iter
    # has the same shape as the loop state.
    _, accum = recurrent.scan(
        NestedMap(data=state0),
        NestedMap(data=padded_inputs),
        _ScanFn,
        root_layer=self)

    # Extract output from the last stage after num_stages-1 bubbles.
    output = jax.tree_map(lambda x: x[L - 1:, -1, ...], accum.data)
    return output
