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

from absl import logging
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
  dimension with vmap() in the input/output data and weights.

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
        state = vmap(single_stage_body.fprop)(stages_in)
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
    # Set either num_microbatches or microbatch_size for input microbatching.
    p.Define(
        'num_microbatches', None,
        'If not None, the input is not yet microbatched, and will be reshaped '
        'to [num_microbatches, microbatch_size] here.')
    p.Define(
        'microbatch_size', None,
        'If not None, the input is not yet microbatched, and will be reshaped '
        'to [num_microbatches, microbatch_size] here.')
    p.Define(
        'unpack_summaries', False,
        'If true, unpack summaries to the individual values from each stage.')
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

  def _forward_summary(self, summaries):
    """Forwards summary from the inner JaxContext to the outer context."""
    p = self.params
    for summary_key, summary_value in summaries.items():
      logging.info((summary_key, summary_value))
      summary_type = base_layer.get_summary_type_from_key(summary_key)
      assert summary_value.shape[0] == p.num_stages
      if p.unpack_summaries:
        # unstack summary_value
        unstacked_values = jnp.split(summary_value, p.num_stages)
        for i, v in enumerate(unstacked_values):
          base_layer.add_summary(f'{summary_key}/{i}', v, summary_type)
      else:
        base_layer.add_summary('{summary_key}', summary_value, summary_type)

  def body_fprop(self, per_stage_inputs: JTensor, *per_stage_args,
                 **per_stage_kwargs) -> NestedJTensor:
    """Runs the fprop function of the stages."""
    p = self.params
    if p.mesh_axis_names is not None:

      def annotate(x):
        unconstrained_dims = list(range(1, x.ndim))
        dims_mapping = (
            p.weight_split_dims_mapping.stages + [None] * (x.ndim - 1))
        return base_layer.maybe_shard(x, dims_mapping, p.mesh_axis_names,
                                      unconstrained_dims)

      per_stage_inputs = jax.tree_map(annotate, per_stage_inputs)
      per_stage_args = jax.tree_map(annotate, per_stage_args)
      per_stage_kwargs = jax.tree_map(annotate, per_stage_kwargs)

    prng_key = base_layer.next_prng_key()
    global_step = base_layer.cur_global_step()

    # vmap self.body.fprop to get a leading stage dimension to handle per_stage
    # inputs and args.
    def _wrapped_fn(theta, per_stage_inputs, *per_stage_args,
                    **per_stage_kwargs):
      with base_layer.JaxContext.new_context(
          prng_key=prng_key, global_step=global_step) as jax_ctx:
        jax_ctx.bind(self.body, self.body.vars_to_flax_vars(theta),
                     [base_layer.SCOPE_AUX_LOSS])
        res = self.body.fprop(per_stage_inputs, *per_stage_args,
                              **per_stage_kwargs)
        summaries = base_layer.all_summaries()
        return res, summaries

    res, summaries = jax.vmap(_wrapped_fn)(self.body.local_theta(),
                                           per_stage_inputs, *per_stage_args,
                                           **per_stage_kwargs)

    self._forward_summary(summaries)
    return res

  def fprop(self, inputs: NestedJTensor, *broadcast_inputs,
            **broadcast_kwargs) -> NestedJTensor:
    """FProp inputs through the pipeline body.

    self.body.fprop is expected to be of the following signature:
    outputs = self.body.fprop(theta, inputs,
                              *broadcast_inputs, **broadcast_kwargs)

    outputs are expected to be of the same structure as inputs.

    Args:
      inputs: Inputs to body_fprop, same structure as outputs.
      *broadcast_inputs: Broadcasted args to body_fprop.
      **broadcast_kwargs: Broadcasted kwargs to body_fprop

    Returns:
      Output from the last pipeline stage.
    """
    p = self.params
    L = p.num_stages  # pylint: disable=invalid-name

    # Handle microbatching.
    needs_microbatching = False
    if p.num_microbatches is None:
      num_microbatches = inputs.shape[0]  # pytype: disable=attribute-error  # jax-ndarray
      if p.microbatch_size is not None:
        batch_size = num_microbatches
        assert batch_size % p.microbatch_size == 0
        num_microbatches = batch_size // p.microbatch_size
        needs_microbatching = True
    else:
      num_microbatches = p.num_microbatches
      needs_microbatching = True

    if needs_microbatching:

      def _to_microbatches(x):
        batch = x.shape[0]
        assert batch % num_microbatches == 0
        # We first put num_microbatches in the inner dimension then transpose
        # it. This allows the sharding on the batch (if any) to be propagated
        # to the microbatch dimension because otherwise XLA SPMD propagates
        # sharding to the major dimension (num_microbatches) when we split
        # batch to num_microbatches and microbatch_sizes. We cannot shard the
        # num_microbatches dimension since it's indexed by the loop iteration.
        reshaped = x.reshape([batch // num_microbatches, num_microbatches] +
                             list(x.shape[1:]))
        return reshaped.transpose([1, 0] + list(range(2, len(reshaped.shape))))

      inputs = jax.tree_map(_to_microbatches, inputs)
      broadcast_inputs = jax.tree_map(_to_microbatches, broadcast_inputs)
      broadcast_kwargs = jax.tree_map(_to_microbatches, broadcast_kwargs)

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

    theta = self.local_theta()
    def _scan_fn(carry, xs):
      jax_context = base_layer.cur_jax_context()
      flax_theta = self.vars_to_flax_vars(theta)
      jax_context.bind(self, flax_theta, [base_layer.SCOPE_AUX_LOSS])

      in_state, loop_iter, inp = carry.data, carry.loop_iter, xs.inputs

      # Bring in the next microbatch.
      def _select_state_or_input(x, s):
        return jnp.where(
            jax.lax.broadcasted_iota('int32', s.shape, 0) == 0, x, s)

      stages_in = jax.tree_map(_select_state_or_input, inp, in_state)

      # Different stages need args from different microbatches.
      microbatch_ids = (loop_iter - jnp.arange(L) +
                        num_microbatches) % num_microbatches

      def _gather(xs):

        def _gather_one(x, i):
          return x[i]

        if p.mesh_axis_names is not None:
          # When the stage dim is partitioned, we use xmap (with manual sharding
          # implementation) to make sure it's trivially partitioned on the stage
          # dim and work around some potential optimization problems in XLA.
          # TODO(yuanzx): Use xmap on the whole body fprop.
          mesh_axis = base_layer.to_partition_spec(
              p.weight_split_dims_mapping.stages, p.mesh_axis_names)[0]
          if mesh_axis is not None:
            axis_resources = {'num_stages': mesh_axis}
            return maps.xmap(
                _gather_one,
                # broadcast_inputs are replicated across stages, but IDs are
                # per-stage.
                in_axes=([...], ['num_stages', ...]),
                out_axes=['num_stages', ...],
                axis_resources=axis_resources)(xs, microbatch_ids)
        return xs[microbatch_ids]

      per_stage_args = jax.tree_map(_gather, broadcast_inputs)
      per_stage_kwargs = jax.tree_map(_gather, broadcast_kwargs)

      # Run through pipeline body.
      out_state = self.body_fprop(stages_in, *per_stage_args,
                                  **per_stage_kwargs)
      py_utils.assert_same_shape_and_dtype(stages_in, out_state)

      # Shift state to the right by 1.
      def _shift_right(x):
        padding = [[1, 0]] + [[0, 0]] * (x.ndim - 1)
        # Use lax.slice to guarantee the gradient is a pad.
        return jax.lax.slice(jnp.pad(x, padding), [0] * x.ndim, x.shape)

      shifted_out_state = jax.tree_map(_shift_right, out_state)
      # Accumulator saves out_state for final output retrieval.
      return NestedMap(
          data=shifted_out_state,
          loop_iter=loop_iter + 1), NestedMap(data=out_state)

    # Loop over num_microbatches + (num_stages - 1), where input to each iter
    # has the same shape as the loop state.
    _, accum, summaries = recurrent.scan(
        NestedMap(data=state0, loop_iter=0),
        NestedMap(inputs=padded_inputs),
        _scan_fn,
        root_layer=self)
    # TODO(xxx): deal with summaries.
    del summaries

    # Extract output from the last stage after num_stages-1 bubbles.
    def _extract_out(x):
      # Use lax.slice to guarantee the gradient is a pad.
      return jnp.squeeze(
          jax.lax.slice(x, [L - 1, x.shape[1] - 1] + [0] * (x.ndim - 2),
                        x.shape), 1)

    output = jax.tree_map(_extract_out, accum.data)

    if needs_microbatching:

      def _to_batches(x):
        transposed = x.transpose([1, 0] + list(range(2, len(x.shape))))
        return transposed.reshape([num_microbatches * x.shape[1]] +
                                  list(x.shape[2:]))

      output = jax.tree_map(_to_batches, output)

    return output
