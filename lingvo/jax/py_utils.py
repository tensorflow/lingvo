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

import functools
from typing import Any
import zlib

from absl import logging
import jax
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


def extract_prefixed_keys_from_nested_map(node: Any,
                                          prefix: str = '',
                                          left_separator: str = '[',
                                          right_separator: str = ']') -> Any:
  """Extracts a NestedMap with the nested prefix keys from its NestedMap node."""

  def extract_keys(n, p):
    """Alias long function call with fixed separators."""
    return extract_prefixed_keys_from_nested_map(
        n, p, left_separator=left_separator, right_separator=right_separator)

  if isinstance(node, dict):  # NestedMap inherits from dict.
    result = {}
    for key, value in node.items():
      if prefix:
        path = f'{prefix}/{key}'
      else:
        path = key
      result[key] = extract_keys(value, path)
    # Convert back to dict or NestedMap.
    return type(node)(result)
  elif isinstance(node, (list, tuple)):
    # Check if it is a NamedTuple.
    if hasattr(node, '_fields'):
      if prefix:
        prefix += '/'
      return type(node)(**{
          field: extract_keys(getattr(node, field), f'{prefix}{field}')
          for field in node._fields
      })
    # Convert back to list or tuple.
    return type(node)(
        extract_keys(v, f'{prefix}{left_separator}{i}{right_separator}')
        for i, v in enumerate(node))
  if not prefix:
    return None
  return prefix


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
  actual = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
  if actual[0] != expected:
    raise ValueError(f'Sync point {name} expected: {expected}; got: {actual}')
  logging.info('Finished sync_global_devices %s across %s devices globally',
               name, global_device_count)
