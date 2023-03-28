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
"""Utilities to vectorize optimizers based on variable shape prefixes.

In weight_params, repeat_prefix represents stacking variables from multiple
layers of the type. Optimizers should treat such a variable as a collection of
separate per-layer variables.

get_transformations_with_vectorized_repeat_prefix creates a wrapper of existing
optimizers to vectorize them on prefix dimensions, such that using stacked
variables does not affect optimizer behavior.

If there are variables that use repeat_prefix, the optimizer state will be a
NestedMap with fields like:
{
  'no_prefix': state for vars without prefixes,
  'p#2.3#i0.i-1': state for vars w/ shape prefix [2,3], sharding prefix [0,-1],
  'p#4#i1': state for vars with shape prefix [4], sharding prefix [1],
   ...
}
The 'i' prefix of each sharding dim indicates it's an integer.

If the sharding prefix dims are strings are tuples of strings, the prefix keys
are encoded as:
{
  'p#2.3#sdata.smdl': sharding prefix ['data','mdl'],
  'p#2.3#tsdata,smdl.': sharding prefix [('data','mdl'), None],
  ...
}
The 's' prefix of each sharding dim indicates it's a string, and the 't' prefix
indicates it's a tuple of elements separated by ','.

Stacking variables helps reduce the number of individual variables, which can be
beneficial for compilation time and the current GDA-based checkpointing.
"""

from typing import Callable, Optional, Sequence, Tuple, Union

import jax
from lingvo.jax import optimizers
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import optax

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedParams = pytypes.NestedParams
GeneralGradientTransformation = optimizers.GeneralGradientTransformation
ShardedGradientTransformation = optimizers.ShardedGradientTransformation

NO_PREFIX_KEY = 'no_prefix'


def has_no_prefix(dct: NestedMap) -> bool:
  return dct.keys() == set([NO_PREFIX_KEY])


def _vectorize_on_prefix_dims(fn: Callable[..., NestedJTensor],
                              num_dim: int) -> Callable[..., NestedJTensor]:
  """Vectorize fn on multiple dimensions."""
  if num_dim == 0:
    return fn
  v_fns = [fn]
  for _ in range(num_dim):
    inner_fn = v_fns[-1]
    v_fns.append(jax.vmap(inner_fn))
  return v_fns[-1]


def _encode_sharding_dim(d: Optional[Union[str, Sequence[str], int]]) -> str:
  """Encodes the sharding annotation into a string for one dimension."""
  if d is None:
    return ''
  if isinstance(d, int):
    return 'i%d' % d
  if isinstance(d, (list, tuple)):
    return 't' + ','.join([_encode_sharding_dim(e) for e in d])

  assert isinstance(d, str)
  # The original string should not contain separators.
  assert '.' not in d
  assert ',' not in d
  assert '#' not in d
  return 's' + d


def _decode_sharding_dim(d: str) -> Optional[Union[str, Sequence[str], int]]:
  """Decodes the sharding annotation from a string for one dimension."""
  if not d:
    return None
  if d.startswith('i'):
    return int(d[1:])
  if d.startswith('s'):
    return d[1:]

  assert d.startswith('t')
  if len(d) == 1:
    return ()
  tuple_elements = [_decode_sharding_dim(e) for e in d[1:].split(',')]
  return tuple(tuple_elements)  # pytype: disable=bad-return-type  # always-use-return-annotations


def _get_var_param_repeat_prefix_key(var_param: InstantiableParams) -> str:
  """Returns string keys that uniquely identify shape and sharding prefixes."""
  if not var_param.repeat_prefix:
    return NO_PREFIX_KEY

  sharding_prefix = var_param.repeat_prefix_split_dims_mapping
  if sharding_prefix is None:
    sharding_prefix = [-1] * len(var_param.repeat_prefix)
  assert len(sharding_prefix) == len(var_param.repeat_prefix)
  shape_str = '.'.join(str(d) for d in var_param.repeat_prefix)
  sharding_str = '.'.join(_encode_sharding_dim(d) for d in sharding_prefix)
  return f'p#{shape_str}#{sharding_str}'


def _parse_var_param_repeat_prefix_key(
    prefix: str) -> Tuple[Sequence[int], Sequence[int]]:
  """Parses shape and sharding prefixes from string keys."""
  if prefix == NO_PREFIX_KEY:
    return [], []

  _, shape_str, sharding_str = prefix.split('#')
  shape_prefix = [int(d) for d in shape_str.split('.')]
  sharding_prefix = [_decode_sharding_dim(d) for d in sharding_str.split('.')]
  return shape_prefix, sharding_prefix  # pytype: disable=bad-return-type  # always-use-return-annotations


def group_by_repeat_prefix(variables: NestedMap,
                           var_params: NestedParams) -> NestedMap:
  """Groups variables based on prefix keys."""
  var_params_flat, _ = jax.tree_flatten(var_params)
  key_set = set()
  for p in var_params_flat:
    key = _get_var_param_repeat_prefix_key(p)
    key_set.add(key)

  def _filter_key(key):

    def _filter_one(v, p):
      if key == _get_var_param_repeat_prefix_key(p):
        return v
      return ()

    return jax.tree_map(_filter_one, variables, var_params)

  groups = NestedMap()
  for key in key_set:
    groups[key] = _filter_key(key)

  return groups


def ungroup_by_repeat_prefix(groups: NestedMap,
                             var_params: NestedParams) -> NestedMap:
  """Converts grouped values to the original structure of var_params."""

  group_list = []
  group_index = {}
  for key, group in groups.items():
    group_index[key] = len(group_list)
    group_list.append(group)

  def _get_item(p, *group_vals):
    key = _get_var_param_repeat_prefix_key(p)
    return group_vals[group_index[key]]

  return jax.tree_map(_get_item, var_params, *group_list)


def init_with_vectorized_repeat_prefix(
    tx: GeneralGradientTransformation, var_vals: NestedJTensor,
    var_params: NestedParams) -> optax.OptState:
  """init function for vectorized optimizers based on var_params."""
  vmap_groups = group_by_repeat_prefix(var_vals, var_params)
  results = NestedMap()
  for prefix, group in vmap_groups.items():
    shape_prefix, _ = _parse_var_param_repeat_prefix_key(prefix)
    results[prefix] = _vectorize_on_prefix_dims(tx.init, len(shape_prefix))(
        group)

  if has_no_prefix(results):
    # Do not change the structure if no prefix exists.
    results = results[NO_PREFIX_KEY]
  return results


def update_with_vectorized_repeat_prefix(
    tx: GeneralGradientTransformation, updates: NestedJTensor,
    state: optax.OptState, old_vars: NestedJTensor,
    var_params: NestedParams) -> Tuple[NestedJTensor, optax.OptState]:
  """update function for vectorized optimizers based on var_params."""
  grouped_updates = group_by_repeat_prefix(updates, var_params)
  grouped_old_vars = group_by_repeat_prefix(old_vars, var_params)
  update_results = NestedMap()
  state_results = NestedMap()
  grouped_state = state
  if has_no_prefix(grouped_updates):
    # state structure did not change if no prefix exists.
    grouped_state = NestedMap()
    grouped_state[NO_PREFIX_KEY] = state
  for prefix, group in grouped_updates.items():
    shape_prefix, _ = _parse_var_param_repeat_prefix_key(prefix)
    new_updates, new_state = _vectorize_on_prefix_dims(
        tx.update, len(shape_prefix))(group, grouped_state[prefix],
                                      grouped_old_vars[prefix])
    update_results[prefix] = new_updates
    state_results[prefix] = new_state
  if has_no_prefix(state_results):
    # Do not change the structure if no prefix exists.
    state_results = state_results[NO_PREFIX_KEY]
  return ungroup_by_repeat_prefix(update_results, var_params), state_results


def init_partition_spec_with_vectorized_repeat_prefix(
    tx: ShardedGradientTransformation,
    var_params: NestedParams) -> NestedParams:
  """init_partition_spec for vectorized optimizers based on var_params."""

  def call_inner_on_group(group, shape_prefix, sharding_prefix):

    def _remove_prefix(p):
      p = p.Copy()
      p.repeat_prefix = None
      p.repeat_prefix_split_dims_mapping = None
      return p

    group = jax.tree_map(_remove_prefix, group)
    result = tx.init_partition_spec(group)

    def _add_prefix(p):
      p.repeat_prefix = shape_prefix
      p.repeat_prefix_split_dims_mapping = sharding_prefix
      return p

    return jax.tree_map(_add_prefix, result)

  # Use the same grouping as init_with_vectorized_repeat_prefix, in order to
  # produce compatible tree structures.
  vmap_groups = group_by_repeat_prefix(var_params, var_params)
  results = NestedMap()
  for prefix, group in vmap_groups.items():
    shape_prefix, sharding_prefix = _parse_var_param_repeat_prefix_key(prefix)
    results[prefix] = call_inner_on_group(group, shape_prefix, sharding_prefix)
  if has_no_prefix(results):
    # Do not change the structure if no prefix exists.
    results = results[NO_PREFIX_KEY]
  return results


def get_transformations_with_vectorized_repeat_prefix(
    tx: GeneralGradientTransformation,
    var_params: NestedParams) -> GeneralGradientTransformation:
  """Vectorizes a transformation on shape/sharding prefixes."""

  def _init(variables):
    return init_with_vectorized_repeat_prefix(tx, variables, var_params)

  def _update(updates, state, params=None):
    return update_with_vectorized_repeat_prefix(tx, updates, state, params,
                                                var_params)

  def _init_partition_spec(var_param_args):
    assert isinstance(tx, ShardedGradientTransformation)
    return init_partition_spec_with_vectorized_repeat_prefix(tx, var_param_args)

  if isinstance(tx, ShardedGradientTransformation):
    return ShardedGradientTransformation(
        init=_init, update=_update, init_partition_spec=_init_partition_spec)
  else:
    assert isinstance(tx, optax.GradientTransformation)
    return optax.GradientTransformation(init=_init, update=_update)
