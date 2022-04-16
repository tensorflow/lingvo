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
"""Generic repeat layer that stacks a sub-layer multiple times.

This simply passes input through the layer stack.
"""

from typing import Any

from absl import logging
import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import recurrent
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class Repeat(base_layer.BaseLayer):
  """A generic repeat layer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Parameterization of this model."""
    p = super().Params()
    p.Define('sub', None, 'The param of the sub-layer.')
    p.Define('x_times', 0, 'Num times to repeat sub.')
    p.Define(
        'unpack_summaries', False,
        'If true, unpack summaries to the individual values from each loop'
        ' iterations.')
    p.Define(
        'checkpoint_policy', recurrent.AutodiffCheckpointType.SAVE_NOTHING,
        'How to checkpoint residuals for BProp: save nothing, dot only or '
        'dot with no batch dimensions.')
    wp = p.weight_split_dims_mapping
    wp.Define('sub', None, 'How the list of subs should be sharded.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructor."""
    super().__init__(params)
    p = self.params
    wp = p.weight_split_dims_mapping
    assert p.x_times > 0
    assert p.sub is not None

    if wp.sub is not None:
      assert isinstance(wp.sub, (list, tuple))
      assert len(wp.sub) == 1
      wp_sub = tuple(wp.sub)
    else:
      wp_sub = (-1,)

    if p.repeat_prefix is not None:
      # This repeat layer is already part of another repeat layer.
      repeat_prefix = p.repeat_prefix + [p.x_times]
    else:
      repeat_prefix = [p.x_times]
    # TODO(yonghui): Propagate repeat_prefix_split_dims_mapping
    if p.repeat_prefix_split_dims_mapping is not None:
      repeat_prefix_split_dims_mapping = (
          tuple(p.repeat_prefix_split_dims_mapping) + wp_sub)
    else:
      repeat_prefix_split_dims_mapping = wp_sub
    sub_params = p.sub.Copy()
    sub_params.repeat_prefix = repeat_prefix
    sub_params.repeat_prefix_split_dims_mapping = (
        repeat_prefix_split_dims_mapping)

    self.create_child('sub', sub_params)

  def _forward_summary(self, summaries):
    """Forwards summary from the inner JaxContext to the outer context."""
    p = self.params
    for summary_key, summary_value in summaries.items():
      logging.info((summary_key, summary_value))
      summary_type = base_layer.get_summary_type_from_key(summary_key)
      assert summary_value.shape[0] == p.x_times
      if p.unpack_summaries:
        # unstack summary_value
        unstacked_values = jnp.split(summary_value, p.x_times)
        for i, v in enumerate(unstacked_values):
          base_layer.add_summary(f'{summary_key}/{i}', v, summary_type)
      else:
        base_layer.add_summary('{summary_key}', summary_value, summary_type)

  def fprop(self, fprop_fn, inputs: NestedJTensor, *args: Any,
            **kwargs: Any) -> Any:
    """FProp inputs through the sub layer stack.

    fprop_fn is expected to be of the following signature:
    outputs, extra = fprop_fn(sub, inputs, *args, **kwargs)

    outputs are expected to be of the same structure as inputs. extra can be any
    structure.

    Args:
      fprop_fn: The fprop fn to scan over. See comments above for the expected
        signature.
      inputs: A NestedMap of inputs that goes through the sub layer stack.
      *args: Positional args to be passed to sub.fprop method.
      **kwargs: Keyward args to be passed to sub.fprop method.

    Returns:
      Output from the last sub layer.
    """
    p = self.params

    # We wrap inputs in a NestedMap so that inputs to recurrent.scan will always
    # be a NestedMap, which is required by the recurrent.scan interface.
    inputs_mp = NestedMap(carry=inputs)

    def _scan_fn(layer_in, layer_vars):
      jax_context = base_layer.cur_jax_context()
      flax_variables = self.sub.vars_to_flax_vars(layer_vars)
      # properly setup scope.
      jax_context.bind(self.sub, flax_variables, [base_layer.SCOPE_AUX_LOSS])

      layer_out, extra = fprop_fn(self.sub, layer_in.carry, *args, **kwargs)
      tf.nest.assert_same_structure(layer_in.carry, layer_out)
      return NestedMap(carry=layer_out), py_utils.NestedMap(extra=extra)

    out_final, out_extra, summaries = recurrent.scan(
        inputs_mp,
        self.sub.local_theta(),
        _scan_fn,
        root_layer=self,
        checkpoint_policy=p.checkpoint_policy)

    self._forward_summary(summaries)

    return out_final.carry, out_extra.extra

  def init_states(self, init_fn, *args: Any, **kwargs: Any) -> Any:
    """Inits decoder states for all sub layers.

    sub.init_states() should be of signature

    init_fn should be of signature
    init_states = init_fn(self.sub, *args, **kwargs)

    Args:
      init_fn: A callback responsible for initializing the per-block states.
      *args: Positional args to pass to the sub.init_states() method.
      **kwargs: Keyward args to pass to the sub.init_states() method.

    Returns:
      Initial decoder states.
    """
    p = self.params
    init_states = init_fn(self.sub, *args, **kwargs)

    def tile_x(x):
      a = jnp.expand_dims(x, 0)
      return jnp.tile(a, [p.x_times] + [1] * len(x.shape))

    init_states = jax.tree_map(tile_x, init_states)

    # TODO(yonghui): Configure for spmd.
    return init_states

  def extend_step(self, extend_fn, cached_states: NestedMap,
                  step_inputs: NestedJTensor, *args: Any, **kwargs: Any) -> Any:
    """Extends decoder states by one step.

    extend_fn should have the following signature.

    extended_states, step_out = extend_fn(self.sub, states, step_input,
                                          *args, **kwargs)
    extended_states should have the same structure as states
    step_out should have the same structure as step_input

    Args:
      extend_fn: fn to extend cached_states for one step. It should be of the
        expected signature as described above.
      cached_states: The combined states for all sub-layers.
      step_inputs: Input to the bottom decoder layer.
      *args: Additional positional input.
      **kwargs: Additional keyword input.

    Returns:
      new_states, top_decoder_out, where new_states is the updated decoder
      states, and top_decoder_out is the output from the top decoder layer.
    """
    # Wrap inputs in a NestedMap to conform to recurrent.scan interface.
    step_inputs_mp = NestedMap(carry=step_inputs)

    def _scan_fn(layer_in, vars_and_states):
      layer_vars = vars_and_states.layer_vars
      layer_states = vars_and_states.layer_states
      # Properly setup context.
      jax_context = base_layer.cur_jax_context()
      flax_variables = self.sub.vars_to_flax_vars(layer_vars)
      jax_context.bind(self.sub, flax_variables, [base_layer.SCOPE_AUX_LOSS])

      extended_states, layer_out = extend_fn(self.sub, layer_states,
                                             layer_in.carry, *args, **kwargs)
      tf.nest.assert_same_structure(layer_in.carry, layer_out)
      tf.nest.assert_same_structure(extended_states, layer_states)
      return NestedMap(carry=layer_out), extended_states

    vars_and_states = py_utils.NestedMap(
        layer_vars=self.sub.local_theta(), layer_states=cached_states)

    final_out, new_states, summaries = recurrent.scan(
        step_inputs_mp, vars_and_states, _scan_fn, root_layer=self)
    # forward summaries to the out-context.
    self._forward_summary(summaries)
    return new_states, final_out.carry
