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
"""Python utility functions for JAX which contains minimal TF lingvo deps."""

import dataclasses
import functools
from typing import Any

from absl import logging
import flax
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from lingvo.core import cluster
from lingvo.core import hyperparams
from lingvo.core import py_utils
import numpy as np

infeed_context_scope = cluster.InfeedContextScope
# No more symbols from lingvo cluster should be accessed by JAX library.

WeightInit = py_utils.WeightInit
weight_params = py_utils.WeightParams
is_default_param_init = py_utils.IsDefaultParamInit
default_param_init = py_utils.DefaultParamInit
flatten = py_utils.Flatten
NestedMap = py_utils.NestedMap
ThreadLocalDict = py_utils.ThreadLocalDict
ThreadLocalStack = py_utils.ThreadLocalStack
AuxLossContext = py_utils.AuxLossContext
fprop_dtype = py_utils.FPropDtype
sharded_file_pattern_to_glob = py_utils.ShardedFilePatternToGlob
# No more symbols from lingvo py_utils should be accessed by JAX library.
del py_utils

InstantiableParams = hyperparams.InstantiableParams
Params = hyperparams.Params
# No more symbols from lingvo hyperparams should be accessed by JAX library.
del hyperparams

# No more imports from lingvo should be accessed by core JAX library.


# A utility function to flatten copied from third_party/py/jax/_src/util.py
def _unzip2(xys):
  xs = []
  ys = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)


jax.tree_util.register_pytree_node(NestedMap,
                                   lambda xs: _unzip2(sorted(xs.items()))[::-1],
                                   lambda keys, xs: NestedMap(zip(keys, xs)))


@functools.partial(functools.partial, jax.tree_map)
def assert_same_shape_and_dtype(x, y):
  assert x.shape == y.shape and x.dtype == y.dtype


def reshard(array: jnp.ndarray) -> np.ndarray:
  """Reshards an input tensor according to the number of local devices."""
  num_devices = jax.local_device_count()
  batch_size = array.shape[0]
  return np.reshape(array,
                    (num_devices, batch_size // num_devices) + array.shape[1:])


def unshard(array: jnp.ndarray) -> np.ndarray:
  """Undo the resharding to reshape away the local device count leading dim."""
  return np.reshape(array, (-1,) + array.shape[2:])


def maybe_unreplicate_gda(data):
  """Returns the first local shard in `data` if it is a GDA.

  Args:
    data: A Pytree of fully replicated GlobalDeviceArray or other arrays.
  Returns: A Pytree of DeviceArray or the original input.
  """
  return jax.tree_map(
      lambda x: x.addressable_data(0)  # pylint: disable=g-long-lambda
      if isinstance(x, jax.Array)
      else x,
      data,
  )


def extract_keys(n, p, key_separator, left_separator, right_separator):
  """Alias long function call with fixed separators."""
  return extract_prefixed_keys_from_nested_map(
      n,
      p,
      key_separator=key_separator,
      left_separator=left_separator,
      right_separator=right_separator)


def _handle_dict(
    node,
    prefix,
    key_separator,
    left_separator,
    right_separator,
    node_type=None,
):
  """Handles dictionaries."""
  result = {}
  for key, value in node.items():
    if prefix:
      path = f'{prefix}{key_separator}{key}'
    else:
      path = key
    result[key] = extract_keys(value, path, key_separator, left_separator,
                               right_separator)
  if node_type is not None:
    return node_type(**result)
  else:
    return type(node)(result)


def extract_prefixed_keys_from_nested_map(node: Any,
                                          prefix: str = '',
                                          key_separator: str = '/',
                                          left_separator: str = '[',
                                          right_separator: str = ']') -> Any:
  """Extracts a NestedMap with the nested prefix keys from its NestedMap node."""
  if isinstance(node, dict):  # NestedMap inherits from dict.
    return _handle_dict(node, prefix, key_separator, left_separator,
                        right_separator)
  # PartitionSpec is subclass of tuple.
  elif isinstance(node, jax.sharding.PartitionSpec):
    return prefix
  elif isinstance(node, (list, tuple)):
    # Check if it is a NamedTuple.
    if hasattr(node, '_fields'):
      if prefix:
        prefix += f'{key_separator}'
      return type(node)(**{
          field: extract_keys(
              getattr(node, field), f'{prefix}{field}', key_separator,
              left_separator, right_separator) for field in node._fields
      })
    # Convert back to list or tuple.
    return type(node)(
        extract_keys(v, f'{prefix}{left_separator}{i}{right_separator}',
                     key_separator, left_separator, right_separator)
        for i, v in enumerate(node))
  elif (dataclasses.is_dataclass(node) and
        node.__class__ in flax.serialization._STATE_DICT_REGISTRY):  # pylint: disable=protected-access
    if hasattr(node, '__dict__'):
      node_dict = node.__dict__
    else:
      node_dict = flax.serialization.to_state_dict(node)
    return _handle_dict(
        node_dict,
        prefix,
        key_separator,
        left_separator,
        right_separator,
        node_type=type(node),
    )
  if not prefix:
    return None
  return prefix


def sync_global_devices(name: str) -> None:
  """Sync across all hosts/devices."""
  global_device_count = jax.device_count()
  logging.info('Starting sync_global_devices %s across %s devices globally',
               name, global_device_count)
  multihost_utils.sync_global_devices(name)
  logging.info('Finished sync_global_devices %s across %s devices globally',
               name, global_device_count)


def make_array(
    host_arrays: np.ndarray,
    global_shapes: jax.ShapeDtypeStruct,
    global_mesh: jax.sharding.Mesh,
    pspecs: Any,
) -> jax.Array:
  """Makes a Jax Array from host array.

  Evenly partitioning x along axis 0 and device_put shards to local devices.

  Args:
    host_arrays: host-local arrays.
    global_shapes: global shapes of the resultant GDA.
    global_mesh: global mesh of the resultant GDA.
    pspecs: partition specs of the resultant GDA.

  Returns:
    A GDA with x as the host-local data.
  """

  local_devices = global_mesh.local_devices
  local_device_count = jax.local_device_count()

  def _put_to_devices(x):
    per_device_arrays = np.split(x, local_device_count, axis=0)
    device_buffers = [
        jax.device_put(arr, d)
        for arr, d in zip(per_device_arrays, local_devices)
    ]
    return device_buffers

  device_buffers = jax.tree_map(_put_to_devices, host_arrays)

  def _gda(global_shape, pspec, dbs):
    return jax.make_array_from_single_device_arrays(
        global_shape.shape, jax.sharding.NamedSharding(global_mesh, pspec), dbs
    )

  return jax.tree_map(_gda, global_shapes, pspecs, device_buffers)


def get_global_input_shape_dtype(x: jnp.ndarray) -> jax.ShapeDtypeStruct:
  """Get global input shape/dtype assuming fully sharded batch dim."""
  assert len(x.shape) >= 1
  # Assume fully sharded batch dim.
  x_shape = (x.shape[0] * jax.process_count(),) + x.shape[1:]
  return jax.ShapeDtypeStruct(x_shape, x.dtype)


def set_globally_use_rbg_prng_key() -> None:
  """Must call this before any JAX computation to set RBG PRNGKey globally."""
  jax.config.update('jax_default_prng_impl', 'rbg')
