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
from jax.experimental import maps
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
class LayerwiseShardablePipelined(base_layer.BaseLayer):
  """A layer that implements pipelining across stages.

  It creates a loop over microbatches around a loop-body layer. The wrapped body
  layer represents a single stage, which will be added a leading num_stages
  dimension with xmap() in the input/output data and weights.

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
        state = xmap(single_stage_body.fprop)(theta.body, stages_in)
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for LayerwiseShardablePipelined."""
    p = super().Params()
    p.Define('num_stages', 1, 'Number of pipeline stages.')
    p.Define(
        'single_stage_body', None,
        'Single Stage body. A leading num_stages dimension will be added '
        'automatically by the pipeline layer.')
    wp = p.weight_split_dims_mapping
    wp.Define('stages', [-1], 'How the num_stages dimension should be sharded.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructs a LayerwiseShardablePipelined object."""
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.single_stage_body
    repeat_prefix = (p.repeat_prefix or []) + [p.num_stages]
    assert len(p.weight_split_dims_mapping.stages) == 1
    repeat_prefix_split_dims_mapping = tuple(
        p.repeat_prefix_split_dims_mapping or [-1] *
        (len(repeat_prefix) - 1)) + tuple(p.weight_split_dims_mapping.stages)
    body_params = p.single_stage_body.Copy().Set(
        name='body',
        repeat_prefix=repeat_prefix,
        repeat_prefix_split_dims_mapping=repeat_prefix_split_dims_mapping)
    self.create_child('body', body_params)

  def body_fprop(self, theta: NestedMap,
                 per_stage_inputs: NestedJTensor) -> NestedJTensor:
    """Runs the fprop function of the stages."""
    p = self.params
    axis_resources = {}
    if p.mesh_axis_names is not None:
      mesh_axis = base_layer.to_partition_spec(
          p.weight_split_dims_mapping.stages, p.mesh_axis_names)[0]
      if mesh_axis is not None:
        axis_resources = {'num_stages': mesh_axis}
    return maps.xmap(
        self.body.fprop,
        in_axes=['num_stages', ...],
        out_axes=['num_stages', ...],
        axis_resources=axis_resources)(theta.body, per_stage_inputs)

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
      stages_in = jax.tree_map(lambda x, s: jnp.where(iota == 0, x, s), inp,
                               shifted_state)
      # Run through pipeline body.
      out_state = self.body_fprop(theta, stages_in)
      py_utils.assert_same_shape_and_dtype(stages_in, out_state)
      # Accumulator saves out_state for final output retrieval.
      return NestedMap(data=out_state), NestedMap(data=out_state)

    # Loop over num_microbatches + (num_stages - 1), where input to each iter
    # has the same shape as the loop state.
    _, accum, summaries = recurrent.scan(
        NestedMap(data=state0),
        NestedMap(data=padded_inputs),
        _ScanFn,
        root_layer=self)
    # TODO(xxx): deal with summaries.
    del summaries

    # Extract output from the last stage after num_stages-1 bubbles.
    output = jax.tree_map(lambda x: x[L - 1:, -1, ...], accum.data)
    return output
