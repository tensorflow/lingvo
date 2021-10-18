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
"""Flax layer wrapper."""

from typing import Any, Dict

import flax.linen as flax_nn
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import tensorflow.compat.v2 as tf

CreateLayerVariablesStatus = base_layer.CreateLayerVariablesStatus
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor


class FlaxModuleLayer(base_layer.BaseLayer):
  """An adaptor for a Flax Module."""

  def __init__(self, params: base_layer.BaseLayerParamsT) -> None:
    super().__init__(params)
    self._module = self._CreateFlaxModule()

  def _CreateFlaxModule(self) -> flax_nn.Module:
    """Creates and returns a Flax module.

    This function is expected to be called within the __init__ function. A
    concrete sub-class must implement this function.
    """
    raise NotImplementedError()

  def _InitModuleStates(self, prng_key: JTensor) -> Dict[str, JTensor]:
    """Initializes and returns module variables.

    A sub-class must implement this function.

    A typical user will simply do the following:

      prng_key, sub_key1 = jrandom.split(prng_key)
      jit_init = jax.jit(self._module.init)
      dummy_input = <dummy input to initialize the model variables>
      return jit_init({'params': sub_key1}, **dummy_input)

    Args:
      prng_key: The random key used to initialize the module states.

    Returns:
      A nested dictionary of module states.
    """
    raise NotImplementedError()

  def InstantiateVariableConfigs(self) -> None:
    assert (
        self._create_variables_status == CreateLayerVariablesStatus.NOT_ENABLED)
    self._create_variables_status = CreateLayerVariablesStatus.ENABLED
    # Note: it is not very efficient that we have to actually create the
    # variable in order to know their meta information.
    dummy_prng_key = jnp.array([0, 0], dtype=jnp.uint32)
    initial_vars = self._InitModuleStates(dummy_prng_key)
    initial_vars = NestedMap.FromNestedDict(initial_vars)

    def _GetWeightParams(init_var):
      wp = base_layer.WeightParams(
          init=None,
          dtype=init_var.dtype,
          shape=init_var.shape,
          collections=[base_layer.FLAX_VARIABLE])
      # We don't know how this variable was initialized.
      wp.init = None
      return wp

    self._private_vars = tf.nest.map_structure(_GetWeightParams, initial_vars)
    self._create_variables_status = CreateLayerVariablesStatus.COMPLETED

  def InstantiateVariables(self, prng_key: JTensor) -> NestedMap:
    """Initiates module states (params and other states)."""
    # NOTE(yonghui): Here we create variables twice.
    # TODO(yonghui): Optimize to reduce to creating variables only once.
    if self._create_variables_status != CreateLayerVariablesStatus.COMPLETED:
      self.InstantiateVariableConfigs()

    initial_vars = self._InitModuleStates(prng_key)
    initial_vars = NestedMap.FromNestedDict(initial_vars)
    return initial_vars

  @property
  def vars(self):
    return tf.nest.map_structure(lambda x: x, self._private_vars)

  @property
  def forward_updated_vars(self):
    return tf.nest.map_structure(lambda x: x, self._forward_updated_vars.dict)

  def FProp(self, theta: NestedMap, *args: Any, **kwargs: Any) -> Any:
    """Applies self._module to the inputs.

    A sub-class must implement this function.

    A typical implementation simply does the following:

      prng_key1 = base_layer.NextPrngKey()
      prng_key2 = base_layer.NextPrngKey()
      out = self._module.apply(
          theta,
          *args,
          rngs={
              'params': prng_key1,
              'dropout': prng_key2
          },
          **kwargs)
      return out

    TODO(yonghui): Figure out how we may more automatically register
    forward-updated variable, as well as psssing down random keys.

    Args:
      theta: params for this layer and its sub-layers.
      *args: positional args.
      **kwargs: keyword args:

    Returns:
      output from calling the correspondng module.apply function.
    """
    raise NotImplementedError()
