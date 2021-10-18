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
"""Generic repeat layer that stacks a sub-layer multiple times.

This simply passes input through the layer stack.
"""

from typing import Any

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import recurrent
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
WeightParams = py_utils.WeightParams
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class RepeatLayer(base_layer.BaseLayer):
  """A generic repeat layer."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Parameterization of this model."""
    p = super().Params()
    p.Define('sub', None, 'The param of the sub-layer.')
    p.Define('x_times', 0, 'Num times to repeat sub.')
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

    self.CreateChild('sub', sub_params)

  def FProp(self, theta: NestedMap, inputs: NestedJTensor, *args: Any,
            **kwargs: Any) -> Any:
    """FProp inputs through the sub layer stack.

    sub.FProp is expected to be of the following signature:
    outputs, extra = sub.FProp(theta, inputs, *args, **kwargs)

    outputs are expected to be of the same structure as inputs. extra can be any
    structure.

    Args:
      theta: The combined layer params for all layers.
      inputs: A NestedMap of inputs that goes through the sub layer stack.
      *args: Positional args to be passed to sub.FProp method.
      **kwargs: Keyward args to be passed to sub.FProp method.

    Returns:
      Output from the last sub layer.
    """

    # We wrap inputs in a NestedMap so that inputs to recurrent.scan will always
    # be a NestedMap, which is required by the recurrent.scan interface.
    inputs_mp = NestedMap(carry=inputs)

    def _ScanFn(layer_in, layer_vars):
      layer_out, extra = self.sub.FProp(layer_vars, layer_in.carry, *args,
                                        **kwargs)
      # TODO(yonghui): Maybe return stacked extra.
      del extra
      tf.nest.assert_same_structure(layer_in.carry, layer_out)
      return NestedMap(carry=layer_out), py_utils.NestedMap()

    out_final, _ = recurrent.scan(
        inputs_mp, theta.sub, _ScanFn, root_layer=self)

    return out_final.carry

  def InitStates(self, theta: NestedMap, *args: Any, **kwargs: Any) -> Any:
    """Inits decoder states for all sub layers.

    sub.InitStates() should be of signature

    init_states = sub.InitStates(theta, *args, **kwargs)

    Args:
      theta: The combined layer params for all layers.
      *args: Positional args to pass to the sub.InitStates() method.
      **kwargs: Keyward args to pass to the sub.InitStates() method.

    Returns:
      Initial decoder states.
    """
    p = self.params
    # TODO(yonghui): Fix me. We should pass in theta for one sub, instead of all
    # the subs.
    init_states = self.sub.InitStates(theta.sub, *args, **kwargs)

    def TileX(x):
      a = jnp.expand_dims(x, 0)
      return jnp.tile(a, [p.x_times] + [1] * len(x.shape))

    init_states = jax.tree_map(TileX, init_states)

    # TODO(yonghui): Configure for spmd.
    return init_states

  def ExtendStep(self, theta: NestedMap, cached_states: NestedMap,
                 step_inputs: NestedJTensor, *args: Any, **kwargs: Any) -> Any:
    """Extends decoder states by one step.

    sub.ExtendStep() should have the following signature.

    extended_states, step_out = sub.ExtendStep(theta, states, step_input,
                                              *args, **kwargs)
    extended_states should have the same structure as states
    step_out should have the same structure as step_input

    Args:
      theta: The combined layer params for all sub-layers.
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

    def _ScanFn(layer_in, vars_and_states):
      layer_vars = vars_and_states.layer_vars
      layer_states = vars_and_states.layer_states
      extended_states, layer_out = self.sub.ExtendStep(layer_vars, layer_states,
                                                       layer_in.carry, *args,
                                                       **kwargs)
      tf.nest.assert_same_structure(layer_in.carry, layer_out)
      tf.nest.assert_same_structure(extended_states, layer_states)
      return NestedMap(carry=layer_out), extended_states

    vars_and_states = py_utils.NestedMap(
        layer_vars=theta.sub, layer_states=cached_states)

    final_out, new_states = recurrent.scan(
        step_inputs_mp, vars_and_states, _ScanFn, root_layer=self)
    return new_states, final_out.carry
