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
"""Python utility functions for JAX which contains minimal TF lingvo deps."""

import dataclasses
import functools
from typing import Any
import zlib

from absl import logging
import flax
import jax
from jax.experimental import pjit
import jax.experimental.global_device_array as gda_lib
from jax.interpreters import pxla
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


def maybe_unreplicate_gda(data):
  """Returns the first local shard in `data`.

  This is typically used when `data` is fully replicated and we only want one
  unreplicated shard.

  Args:
    data: A Pytree of fully replicated GlobalDeviceArray or other arrays.
  Returns: A Pytree of DeviceArray or the original input.
  """
  leaves, _ = jax.tree_flatten(data)
  if all(
      isinstance(x, gda_lib.GlobalDeviceArray) and x.is_fully_replicated
      for x in leaves):
    return jax.tree_map(lambda x: x.local_data(0), data)
  else:
    return data


def maybe_gda_to_sda(data):
  """Returns a ShardedDeviceArray with local shards from GlobalDeviceArray.

  This is typically used when `data` is partially replicated and we only want
  one SDA gathering all local shards from GDA.

  Args:
    data: A Pytree of partially or fully replicated GlobalDeviceArray or other
      arrays.
  Returns: A Pytree of SDA or original input.
  """
  leaves, _ = jax.tree_flatten(data)
  if all(isinstance(x, gda_lib.GlobalDeviceArray) for x in leaves):

    def gda_to_sda(gda):
      mesh_axes = gda._mesh_axes  # pylint: disable=protected-access
      if not isinstance(mesh_axes, pjit.PartitionSpec):
        pspec = pjit.PartitionSpec(*mesh_axes)
      else:
        pspec = mesh_axes
      parsed_pspec, _, _ = pjit._prepare_axis_resources(pspec, 'mesh_axes')  # pylint: disable=protected-access
      array_mapping = pjit.get_array_mapping(parsed_pspec)
      global_aval = jax.core.ShapedArray(gda.shape, gda.dtype)
      # Will convert to host local shape
      global_mesh = gda._global_mesh  # pylint: disable=protected-access
      local_aval = global_mesh.global_to_local(array_mapping, global_aval)
      sharding_spec = pxla.mesh_sharding_specs(
          global_mesh.local_mesh.shape,
          global_mesh.local_mesh.axis_names)(local_aval, array_mapping)
      return pxla.make_sharded_device_array(local_aval, sharding_spec,
                                            [s.data for s in gda.local_shards])

    return jax.tree_map(gda_to_sda, data)
  else:
    return data


def extract_keys(n, p, key_separator, left_separator, right_separator):
  """Alias long function call with fixed separators."""
  return extract_prefixed_keys_from_nested_map(
      n,
      p,
      key_separator=key_separator,
      left_separator=left_separator,
      right_separator=right_separator)


def _handle_dict(node, prefix, key_separator, left_separator, right_separator):
  result = {}
  for key, value in node.items():
    if prefix:
      path = f'{prefix}{key_separator}{key}'
    else:
      path = key
    result[key] = extract_keys(value, path, key_separator, left_separator,
                               right_separator)
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
    node_dict = flax.serialization.to_state_dict(node)
    return _handle_dict(node_dict, prefix, key_separator, left_separator,
                        right_separator)
  if not prefix:
    return None
  return prefix


# Use top-level named function to avoid recompilation.
def _sync_global_devices_f(x):
  return jax.lax.psum(x, 'i')


def sync_global_devices(name: str) -> None:
  """Sync across all hosts/devices."""
  local_device_count = jax.local_device_count()
  global_device_count = jax.device_count()
  logging.info('sync_global_devices %s across %s devices globally', name,
               global_device_count)
  h = np.int32(zlib.crc32(name.encode()))
  # Round-down h to multiples of global_device_count.
  expected = h // global_device_count * global_device_count
  x = jnp.ones(
      shape=(local_device_count), dtype=np.int32) * (
          h // global_device_count)
  actual = jax.device_get(jax.pmap(_sync_global_devices_f, 'i')(x))
  if actual[0] != expected:
    raise ValueError(f'Sync point {name} expected: {expected}; got: {actual}')
  logging.info('Finished sync_global_devices %s across %s devices globally',
               name, global_device_count)
