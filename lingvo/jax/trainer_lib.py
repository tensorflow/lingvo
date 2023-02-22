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
"""Shared trainer lib utilities."""

import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import pjit
from lingvo.jax import base_layer
from lingvo.jax import base_model
from lingvo.jax import base_task
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax import summary_utils
from lingvo.jax import train_states
import numpy as np
import tensorflow.compat.v2 as tf

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedMap = py_utils.NestedMap
NestedShape = NestedMap
PRNGKey = pytypes.PRNGKey
ParamsT = pytypes.ParamsT
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
InstantiableParams = py_utils.InstantiableParams
TrainState = train_states.TrainState
SummaryDict = pytypes.SummaryDict
TrainStepFn = Callable[[TrainState, JTensor, NestedJTensor], Tuple[TrainState,
                                                                   ...]]
EvalStepFn = Callable[[NestedJTensor, JTensor, JTensor, NestedJTensor], Tuple]
DecodeFn = Callable[[NestedJTensor, JTensor, JTensor, NestedJTensor],
                    NestedJTensor]


def initialize_model_state(jax_task: base_task.SingleTask,
                           prng_key: PRNGKey,
                           discard_opt_states: bool = False) -> TrainState:
  """Initializes the model states."""
  model = jax_task.model
  logging.info('init_var prng_seed: %s', prng_key)
  initial_vars = model.instantiate_variables(prng_key)
  logging.debug('initial_vars: %s', initial_vars)
  learnable_vars = tf.nest.map_structure(
      lambda v: not base_layer.var_not_trainable(v), model.vars)
  tf.nest.assert_same_structure(initial_vars, learnable_vars)
  return jax_task.create_train_state(initial_vars, model.vars,
                                     discard_opt_states)


def replicate_model_state(model_states: TrainState) -> TrainState:
  """Replicates the model states."""
  return jax.device_put_replicated(model_states, jax.local_devices())


def initialize_replicate_model_state(
    jax_task: base_task.SingleTask,
    prng_key: PRNGKey,
    discard_opt_states: bool = False) -> TrainState:
  """Initializes and replicates the model states."""
  model_states = initialize_model_state(jax_task, prng_key, discard_opt_states)
  return replicate_model_state(model_states)


def _maybe_to_bfloat16(x: JTensor) -> JTensor:
  if x.dtype == jnp.float32:
    return x.astype(jnp.bfloat16)
  else:
    return x


def _get_uneven_sharding_paddings(
    partition_spec: jax.sharding.PartitionSpec, shape: Sequence[int],
    mesh_shape: Sequence[int], mesh_axis_names: Sequence[str]) -> Sequence[int]:
  """Returns the padding size on each dimension due to uneven sharding."""
  axes_sizes = {}
  for size, name in zip(mesh_shape, mesh_axis_names):
    axes_sizes[name] = size
  paddings = []
  for axes, dim_size in zip(partition_spec, shape):
    if isinstance(axes, str):
      axes = [axes]
    partitions = int(np.prod([axes_sizes[axis] for axis in (axes or ())]))
    padding = (partitions - dim_size % partitions) % partitions
    paddings.append(padding)
  return paddings


def _maybe_pad_uneven_sharding(
    x: JTensor,
    partition_spec: jax.sharding.PartitionSpec,
    shape: Sequence[int],
    mesh_shape: Sequence[int],
    mesh_axis_names: Sequence[str],
) -> JTensor:
  """Pads x to make it evenly shardable, if needed."""
  paddings = _get_uneven_sharding_paddings(partition_spec, shape, mesh_shape,
                                           mesh_axis_names)
  if all([p == 0 for p in paddings]):
    return x
  # Annotate before pad to make sure they have the same sharding. (Pad does not
  # have the highest sharding propgation priority.)
  x = base_layer.with_sharding_constraint(x, partition_spec)
  return jnp.pad(x, [[0, p] for p in paddings])


def _maybe_slice_uneven_sharding(
    x: JTensor, partition_spec: jax.sharding.PartitionSpec, shape: Sequence[int]
) -> JTensor:
  """Slices x to remove padding due to uneven sharding, if needed."""
  if list(shape) == list(x.shape):
    return x
  x = jax.lax.slice(x, [0] * x.ndim, shape)
  # Annotate after slice to make sure they have the same sharding. (Slice does
  # not have the highest sharding propgation priority.)
  return base_layer.with_sharding_constraint(x, partition_spec)


def train_step_single_learner(
    jax_task: base_task.SingleTask,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    data_parallel_axis_name: Optional[str] = 'batch',
    fprop_dtype: jnp.dtype = jnp.float32
) -> Tuple[TrainState, Any, Any, Any, SummaryDict]:
  """Trains a model for a single step.

  This function works for both pmap-ed model and pjit-ed model. When this
  function is called from pmap-ed trainer, data_parallel_axis_name has to be
  set. Otherwise, data_parallel_axis_name should be either an empty string or
  None.

  TODO(yonghui): Maybe refactor pmap and pjit into two functions.

  This utility is specialized for the singler learner case.

  Args:
    jax_task: An instance of base_task.SingleTask.
    states: An instance of model.TrainState.
    prng_key: A PRNGKey, of shape [2], of type np.uint32.
    inputs: Inputs to the mdl.fprop() function.
    data_parallel_axis_name: a string, the device axis to aggregate gradients
      over.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of the following elements.
    updated_states - updated states.
    loss - loss as computed by mdl.fprop.
    mean_metrics - a dict of metrics. Each element of the dict is a pair
    (metric, weight).
    per_example_out - auxilillary per-example output as computed in mdl.fprop.
    summary_tensors - A dict or nested map of summary tensors computed in
      forward as well as backward.
  """
  assert len(jax_task.learners) == 1
  learner = jax_task.learners[0]
  model = jax_task.model

  context_p = base_layer.JaxContext.Params().Set(do_eval=False)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, states.step)  # pytype: disable=wrong-arg-types  # jax-ndarray

  if data_parallel_axis_name:
    in_pmap = True
  else:
    in_pmap = False

  prng_key, subkey = jax.random.split(prng_key)

  def _loss_fn(
      mdl_vars: NestedJTensor, inputs: NestedMap
  ) -> Tuple[JTensor, Tuple[Any, NestedMap, SummaryDict, SummaryDict]]:
    """Computes loss as well as other auxiliary outputs."""
    if fprop_dtype == jnp.float32:
      pass
    elif fprop_dtype == jnp.bfloat16:
      mdl_vars = jax.tree_map(_maybe_to_bfloat16, mdl_vars)
      inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
    else:
      assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

    with base_layer.JaxContext.new_context(
        params=context_p, prng_key=subkey,
        global_step=states.step) as jax_context:
      jax_context.bind(model, model.vars_to_flax_vars(mdl_vars),
                       [base_layer.SCOPE_VARS, base_layer.SCOPE_AUX_LOSS])

      metrics, per_example_output = model.fprop(inputs)
      loss_name = learner.loss_name
      assert loss_name in metrics
      loss, loss_weight = metrics[loss_name]
      assert loss.ndim == 0, 'loss has to be a scalar.'
      assert loss_weight.ndim == 0, 'loss_weight has to be a scalar'
      loss_weight = jax.lax.stop_gradient(loss_weight)
      if in_pmap:
        # Renormalize loss weight by the total weight across all replicas.
        # This also takes care of dividing loss by num of data parallel
        # replicas.
        loss_weight /= jax.lax.psum(
            loss_weight, axis_name=data_parallel_axis_name)
      else:
        # loss_weight == 1 in spmd.
        loss_weight /= jnp.sum(loss_weight)
      weighted_loss = loss * loss_weight
      # Fetch forward-updated vars, which often include batch norm vars, other
      # misc stats, etc.
      forward_updated_vars = model.updated_vars
      # Finally, fetch all the summary tensors.
      summary_tensors = base_layer.all_summaries()
      if in_pmap:
        summary_tensors = summary_utils.aggregate_per_replica_summaries(
            summary_tensors, data_parallel_axis_name)
    if fprop_dtype == jnp.bfloat16 and weighted_loss.dtype == fprop_dtype:
      weighted_loss = weighted_loss.astype(jnp.float32)
    return weighted_loss, (metrics, forward_updated_vars, summary_tensors,
                           per_example_output)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

  (weighted_loss, (metrics, fwd_updated_vars, fwd_summary_tensors,
                   per_example_out)), grads = grad_fn(states.mdl_vars, inputs)

  if in_pmap:
    # Scale weighted_loss back after gradient computation.
    # This is the average loss across all replicas.
    weighted_loss = jax.lax.psum(
        weighted_loss, axis_name=data_parallel_axis_name)
  else:
    # No sum of loss over all the replicas.
    pass

  if in_pmap:
    mean_metrics = type(metrics)()
    for key in metrics:
      value, weight = metrics[key]
      sum_value = jax.lax.psum(
          value * weight, axis_name=data_parallel_axis_name)
      sum_weight = jax.lax.psum(weight, axis_name=data_parallel_axis_name)
      mean_metrics[key] = (sum_value / (sum_weight + 1e-8), sum_weight)
  else:
    # No aggregation is needed.
    mean_metrics = metrics

  var_weight_params = model.vars
  tf.nest.assert_same_structure(states.mdl_vars, var_weight_params)
  tf.nest.assert_same_structure(states.mdl_vars, grads)
  tf.nest.assert_same_structure(states.mdl_vars, fwd_updated_vars)
  var_is_learnable = tf.nest.map_structure(
      lambda x: not base_layer.var_not_trainable(x), var_weight_params)
  tf.nest.assert_same_structure(states.mdl_vars, var_is_learnable)

  def _maybe_zero_out_grad_fn(var_grad, var, var_learnable):
    if var_learnable:
      return var_grad
    else:
      return jnp.zeros_like(var)

  # Zero-out gradient for non-learnable vars.
  grads = tf.nest.map_structure(_maybe_zero_out_grad_fn, grads, states.mdl_vars,
                                var_is_learnable)

  if in_pmap:
    # Aggregate grads across different model replicas.
    grads = jax.lax.psum(grads, axis_name=data_parallel_axis_name)
  else:
    # No gradient aggregation is needed.
    pass

  # Carry out backward computation under a JaxContext.
  prng_key, subkey = jax.random.split(prng_key)
  with base_layer.JaxContext.new_context(
      params=context_p, prng_key=subkey,
      global_step=states.step) as jax_context:
    # Nothing is allowed to change, except for summaries.
    jax_context.bind(model, model.vars_to_flax_vars(states.mdl_vars),
                     [base_layer.SCOPE_AUX_LOSS])

    # Add a summary for learning rate
    learning_rate = learner.optimizer.get_learning_rate(states.step)
    base_layer.add_summary('learning_rate', learning_rate)

    # Apply gradient transformations.
    transformed_grads, new_opt_states = learner.update_states(
        grads, states.opt_states[0], states.mdl_vars, var_weight_params)

    # Gradient descent on learnable vars.
    mdl_vars = learner.apply_gradient(states.mdl_vars, transformed_grads,
                                      var_is_learnable)

    def _synchronize_vars_using_mean(new_var: NestedMap,
                                     old_var: NestedMap) -> NestedMap:
      """Synchronize variables across a replica by averaging."""
      delta = new_var - old_var
      delta_mean = jax.lax.pmean(delta, axis_name=data_parallel_axis_name)
      updated_var = old_var + delta_mean
      return updated_var

    def _update_non_learnable_var(old_var: NestedMap, new_var: NestedMap,
                                  var_params: ParamsT) -> NestedMap:
      """Update non-trainable variables, using cross-replica synchronization.

      Args:
        old_var: Nested map of old variables.
        new_var: Nested map of new variables.
        var_params: Nested map of var param attributes such as whether a
          variable is trainable or requires synchornization across replicas.

      Returns:
        Updated variables.

      Raises:
        ValueError if no synchronization method is provided for non-trainable
        variables.
      """
      if not base_layer.var_not_trainable(var_params):
        assert new_var is None
        return old_var
      elif not in_pmap:
        # No aggregation is needed.
        assert new_var is not None
        return new_var
      elif base_layer.var_requires_mean_sync(var_params):
        assert new_var is not None
        return _synchronize_vars_using_mean(new_var, old_var)
      else:
        raise ValueError('Non-trainable variables must have a cross-replica '
                         'synchronization method specified.')

    var_weight_params = model.vars
    tf.nest.assert_same_structure(mdl_vars, var_weight_params)
    mdl_vars = tf.nest.map_structure(_update_non_learnable_var, mdl_vars,
                                     fwd_updated_vars, var_weight_params)

    new_states = states.new_state(
        mdl_vars=mdl_vars, opt_states=[new_opt_states])
    # Finally fetch all backward summary tensors. We do not aggregate the scalar
    # summaries with pmean because the grads are already psum-ed.
    bwd_summary_tensors = base_layer.all_summaries()

  summary_tensors = NestedMap(
      fwd_summary_tensors=fwd_summary_tensors,
      bwd_summary_tensors=bwd_summary_tensors)

  return (new_states, weighted_loss, mean_metrics, per_example_out,
          summary_tensors)


def eval_step_single_learner(
    jax_task: base_task.SingleTask,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    data_parallel_axis_name: Optional[str] = 'batch',
    fprop_dtype: jnp.dtype = jnp.float32
) -> Tuple[TrainState, Any, Any, Any, SummaryDict]:
  """Evaluates a model for a single step.

  This utility is specialized for the single learner case.

  Args:
    jax_task: An instance of base_task.SingleTask.
    states: An instance of model.TrainState.
    prng_key: A prng seed, of shape [2], of type np.uint32.
    inputs: Inputs to the mdl.fprop() function.
    data_parallel_axis_name: a string, the device axis to aggregate gradients
      over.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of the following elements.
    loss - loss as computed by mdl.fprop.
    mean_metrics - a dict of metrics. Each element of the dict is a pair
    (metric, weight).
    per_example_out - auxilillary per-example output as computed in mdl.fprop.
    summary_tensors - A nested map or dict of summary tensors computed in
      forward as well as backward pass.
  """
  context_p = base_layer.JaxContext.Params().Set(do_eval=True)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, states.step)  # pytype: disable=wrong-arg-types  # jax-ndarray
  model = jax_task.model
  mdl_vars = states.mdl_vars
  assert not states.opt_states

  if fprop_dtype == jnp.float32:
    pass
  elif fprop_dtype == jnp.bfloat16:
    mdl_vars = jax.tree_map(_maybe_to_bfloat16, mdl_vars)
    inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
  else:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  with base_layer.JaxContext.new_context(
      params=context_p, prng_key=prng_key,
      global_step=states.step) as jax_context:
    # Prepares mdl for fprop. This clears all forward-updated vars that kept
    # locally in mdl.
    jax_context.bind(model, model.vars_to_flax_vars(mdl_vars),
                     [base_layer.SCOPE_AUX_LOSS])

    # Support multiple learners.
    assert len(jax_task.learners) == 1
    learner = jax_task.learners[0]

    metrics, per_example_out = model.fprop(inputs)
    loss_name = learner.loss_name
    assert loss_name in metrics
    loss, loss_weight = metrics[loss_name]
    assert loss.ndim == 0, 'loss has to be a scalar.'
    assert loss_weight.ndim == 0, 'loss_weight has to be a scalar.'

    summary_tensors = base_layer.all_summaries()

    if data_parallel_axis_name:
      # This is simple data-parallel training.
      # Renormalize loss weight by the total weight across all replicas.
      sum_loss = jax.lax.psum(
          loss * loss_weight, axis_name=data_parallel_axis_name)
      sum_loss_weight = jax.lax.psum(
          loss_weight, axis_name=data_parallel_axis_name)
      mean_loss = sum_loss / (sum_loss_weight + 1e-8)

      mean_metrics = type(metrics)()
      for key in metrics:
        value, weight = metrics[key]
        sum_value = jax.lax.psum(
            value * weight, axis_name=data_parallel_axis_name)
        sum_weight = jax.lax.psum(weight, axis_name=data_parallel_axis_name)
        mean_metrics[key] = (sum_value / (sum_weight + 1e-8), sum_weight)

      summary_tensors = summary_utils.aggregate_per_replica_summaries(
          summary_tensors, data_parallel_axis_name)
    else:
      # No data_parallel_axis_name is specified, most likely this is evaling an
      # spmd model.
      mean_metrics = metrics
      mean_loss = loss

  def _maybe_to_float32(x):
    if x.dtype == jnp.bfloat16:
      return x.astype(jnp.float32)
    else:
      return x

  if fprop_dtype == jnp.bfloat16:
    mean_loss, mean_metrics, per_example_out, summary_tensors = jax.tree_map(
        _maybe_to_float32,
        (mean_loss, mean_metrics, per_example_out, summary_tensors))

  # Adding the unchanged state to the return list so that both
  # eval_step_single_learner and train_step_single_learner have the same api to
  # facilitate some down-stream code.
  return states, mean_loss, mean_metrics, per_example_out, summary_tensors


def decode_step(
    model: base_model.BaseModel,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32) -> Tuple[NestedMap, NestedMap]:
  """Decodes a model for a single step.

  Args:
    model: An instance of models.BaseModel.
    states: An instance of TrainState..
    prng_key: A prng seed, of shape [2], of type np.uint32.
    inputs: A batch of inputs to model.decode().
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of (metrics, results) as computed by mdl.decode().
  """
  context_p = base_layer.JaxContext.Params().Set(do_eval=True)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, states.step)  # pytype: disable=wrong-arg-types  # jax-ndarray
  mdl_vars = states.mdl_vars
  assert not states.opt_states

  if fprop_dtype == jnp.bfloat16:
    mdl_vars = jax.tree_map(_maybe_to_bfloat16, mdl_vars)
    inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
  elif fprop_dtype != jnp.float32:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  with base_layer.JaxContext.new_context(
      params=context_p, prng_key=prng_key,
      global_step=states.step) as jax_context:
    jax_context.bind(model, model.vars_to_flax_vars(mdl_vars),
                     [base_layer.SCOPE_AUX_LOSS])

    return model.decode(inputs)


def initialize_partitioned_model_states(
    jax_task: base_task.SingleTask,
    prng_key: PRNGKey,
    discard_opt_states: bool = False,
) -> Tuple[TrainState, TrainState]:
  """Initializes model vars that are partitioned over TPU devices.

  This function is equivalent to calling a pjit-ted version of
  InitializesModelStates().

  Args:
    jax_task: The task which is an instance of base_task.SingleTask.
    prng_key: A PRNGKey.
    discard_opt_states: bool, When true, optimizer slot variables are skipped.

  Returns:
    The partitioned specs and the partitioned vars themselves.
  """
  model = jax_task.model
  model.instantiate_variable_configs()
  # At this point, variable specs are already known.
  var_specs = model.vars
  train_state_partition_specs = jax_task.create_train_state_partition_specs(
      var_specs, discard_opt_states)
  train_state_unpadded_shapes = jax_task.create_train_state_unpadded_shapes(
      var_specs, discard_opt_states)
  assert train_state_partition_specs is not None

  def _maybe_pad(x, pspec, shape):
    return _maybe_pad_uneven_sharding(x, pspec, shape,
                                      model.params.device_mesh.shape,
                                      model.params.mesh_axis_names)

  def init_model_from_seed(prng_key):
    outs = initialize_model_state(jax_task, prng_key, discard_opt_states)
    return jax.tree_map(_maybe_pad, outs, train_state_partition_specs,
                        train_state_unpadded_shapes)

  logging.info('unpadded_out_shape: %s', train_state_unpadded_shapes)
  logging.info('train_state_partition_specs: %s', train_state_partition_specs)
  tf.nest.assert_same_structure(train_state_unpadded_shapes,
                                train_state_partition_specs)

  init_fn = pjit.pjit(
      init_model_from_seed,
      in_shardings=(None,),
      out_shardings=train_state_partition_specs,
  )

  assert (
      base_layer.global_mesh_defined()
  ), 'must be inside jax.sharding.Mesh scope'
  partitioned_vars = init_fn(prng_key)

  return (train_state_partition_specs, partitioned_vars)


def shard_on_batch_dim_partition_spec(
    mesh_names: Sequence[str], x: jax.ShapeDtypeStruct
) -> jax.sharding.PartitionSpec:
  """Fully shards x on the batch dimension."""
  x_dim = len(x.shape)
  assert x_dim >= 1
  sharding = [-1] * x_dim
  # Assume the first dim is batch, and fully shard the batch dim over the entire
  # mesh.
  sharding[0] = tuple(mesh_names)
  return base_layer.to_partition_spec(sharding, mesh_names)


def reshard_input_based_on_rank_fn(
    mapping_dict: Dict[str, base_layer.SplitDimsMapping],
    mesh_names: Sequence[str],
    x: JTensor,
) -> JTensor:
  """Reshards input based on its rank.

  Args:
    mapping_dict: Dictionary which contains the split mapping for different
      shapes. For n-d shape, it must have an entry f'map_{n}d' which tells us
      how to partition tensors of this dimension.
    mesh_names: List of mesh axis names.
    x: JTensor which to shard.

  Returns:
    Resharded tensor.
  """
  key = f'map_{len(x.shape)}d'
  if key not in mapping_dict:
    raise ValueError(f'Split mapping must be provided for {len(x.shape)}-d'
                     f'in the form of key map_{len(x.shape)} in'
                     f'{mapping_dict}.')
  if mapping_dict[key] is not None:
    return base_layer.maybe_shard(x, mapping_dict[key], mesh_names)
  else:
    return x


def infer_partition_spec_based_on_rank_fn(
    mapping_dict: Dict[str, base_layer.SplitDimsMapping],
    mesh_names: Sequence[str],
    x: JTensor,
) -> Optional[jax.sharding.PartitionSpec]:
  """Infers PartitionSpec of input from the rank of corresponding JTensors.

  Args:
    mapping_dict: Dictionary which contains the split mapping for different
      shapes. For n-d shape, it must have an entry f'map_{n}d' which tells us
      how to partition tensors of this dimension.
    mesh_names: List of mesh axis names.
    x: JTensor which to shard.

  Returns:
    PartitionSpec or None (if everything is replicated).
  """
  key = f'map_{len(x.shape)}d'
  if key not in mapping_dict:
    raise ValueError(f'Split mapping must be provided for {len(x.shape)}-d'
                     f'in the form of key map_{len(x.shape)} in'
                     f'{mapping_dict}.')
  if mapping_dict[key] is not None:
    return base_layer.to_partition_spec(mapping_dict[key], mesh_names)


def get_input_partition_specs(mesh_axis_names, inputs_shape):
  # Compute inputs PartitionSpec from inputs_shape
  inputs_partition_spec_fn = functools.partial(
      shard_on_batch_dim_partition_spec, mesh_axis_names)
  return tf.nest.map_structure(inputs_partition_spec_fn, inputs_shape)


def train_state_for_eval_step(state_with_opt_states):
  return TrainState(
      step=state_with_opt_states.step,
      mdl_vars=state_with_opt_states.mdl_vars,
      opt_states={})


def partition_spmd_model(
    task_p: InstantiableParams,
    init_key: PRNGKey,
    inputs_shape: NestedShapeDtypeStruct,
) -> Tuple[TrainState, TrainState, TrainState, TrainStepFn, EvalStepFn, int]:
  """Setup the SPMD model and return sharded train and eval step function.

  For partitioning inputs, it is assumed the `task_p.train` has a field
  `inputs_split_mapping` which further contains keys `map_1d`, `map_2d`, ...,
  etc., which specifies how to shard inputs of that corresponding dimension.

  Args:
    task_p: Task parameters of type NestedMap.
    init_key: PRNGKey for initializing the model variables.
    inputs_shape: Shape of the inputs for use in pjit sharding.

  Returns:
    (train_model_states, train_state_partition_specs,
    inputs_partition_spec, train_step_fn, eval_step_fn, total_num_params):
    The partitioned TrainState, the corresponding partitioned TrainState specs,
    the partition spec for the inputs, the train step function, eval step
    function and total number of parameters.
  """
  jax_task = task_p.Instantiate()

  # Initialize the partitioned vars.
  model_state_partition_specs, model_states = (
      initialize_partitioned_model_states(jax_task, init_key))

  total_num_params = jax_task.model.total_num_vars

  train_step, inputs_partition_spec = get_partitioned_spmd_model_step_fn(
      jax_task,
      init_key,
      model_state_partition_specs,
      inputs_shape,
      is_eval=False)

  eval_step, _ = get_partitioned_spmd_model_step_fn(
      jax_task,
      init_key,
      train_state_for_eval_step(model_state_partition_specs),
      inputs_shape,
      is_eval=True)
  return (model_states, model_state_partition_specs, inputs_partition_spec,
          train_step, eval_step, total_num_params)


# TODO(pax): merge with get_partitioned_spmd_model_decode_fn
def get_partitioned_spmd_model_step_fn(jax_task: base_task.SingleTask,
                                       init_key: PRNGKey,
                                       model_state_partition_specs: TrainState,
                                       inputs_shape: NestedShapeDtypeStruct,
                                       is_eval: bool):
  """Return sharded train or eval step function of the SPMD Model.

  Args:
    jax_task: The task which is an instance of base_task.SingleTask.
    init_key: PRNGKey for initializing the model variables.
    model_state_partition_specs: A TrainState contains PartitionSpecs for all
      the variables.
    inputs_shape: Shape of the inputs for use in pjit sharding.
    is_eval: bool, indicating if it's a eval/decode task or not.

  Returns:
    (step_fn, inputs_partition_spec):
    The step function and the partition spec for the inputs.
  """
  task_p = jax_task.params
  model_p = task_p.model
  mesh_names = model_p.mesh_axis_names

  reshard_inputs_fn = functools.partial(reshard_input_based_on_rank_fn,
                                        task_p.train.inputs_split_mapping,
                                        mesh_names)
  inputs_partition_spec = get_input_partition_specs(mesh_names, inputs_shape)

  state_unpadded_shapes = jax_task.create_train_state_unpadded_shapes(
      jax_task.model.vars, discard_opt_states=is_eval)

  # TODO(bf-jax): prng_key is replicated. Would this be a problem?
  prng_key_partition_spec = base_layer.to_partition_spec((None,), mesh_names)

  def _maybe_pad(x, pspec, shape):
    return _maybe_pad_uneven_sharding(x, pspec, shape,
                                      model_p.device_mesh.shape,
                                      model_p.mesh_axis_names)

  def _step_fn(state, prng_key, inputs):
    # Reshard inputs.
    inputs = jax.tree_map(reshard_inputs_fn, inputs)
    # Vars are padded at program entry/exit to avoid uneven sharding. We slice
    # the vars to revome padding before the step computation, and pad them after
    # the step computation to make user code independent of paddings. Internal
    # uneven sharding in the step computation is supported by XLA.
    state = jax.tree_map(_maybe_slice_uneven_sharding, state,
                         model_state_partition_specs, state_unpadded_shapes)

    fn = eval_step_single_learner if is_eval else train_step_single_learner
    fn_out = fn(
        jax_task,
        state,
        prng_key,
        inputs,
        data_parallel_axis_name=None,
        fprop_dtype=model_p.fprop_dtype)

    assert len(fn_out) > 1

    new_states = jax.tree_map(_maybe_pad, fn_out[0],
                              model_state_partition_specs,
                              state_unpadded_shapes)
    return (new_states,) + fn_out[1:]

  def init_model_from_seed(init_key):
    outs = initialize_model_state(
        jax_task, init_key, discard_opt_states=is_eval)
    return jax.tree_map(_maybe_pad, outs, model_state_partition_specs,
                        state_unpadded_shapes)

  var_padded_shapes = jax.eval_shape(init_model_from_seed, init_key)

  out_padded_shapes = jax.eval_shape(_step_fn, var_padded_shapes, init_key,
                                     inputs_shape)

  fn_in_partition_specs = (model_state_partition_specs, prng_key_partition_spec,
                           inputs_partition_spec)
  # Currently, all the outputs are fully replicated.
  # TODO(yonghui): Somehow fetch the output sharding spec from _eval_step fn.
  fn_out_partition_specs = tf.nest.map_structure(lambda _: None,
                                                 out_padded_shapes)

  fn_out_partition_specs = tuple([model_state_partition_specs] +
                                 list(fn_out_partition_specs[1:]))

  tf.nest.assert_same_structure(fn_out_partition_specs, out_padded_shapes)

  # pjit-ed step function.
  step_fn = pjit.pjit(
      _step_fn,
      in_axis_resources=fn_in_partition_specs,
      out_axis_resources=fn_out_partition_specs,
      donate_argnums=() if is_eval else (0,))

  return step_fn, inputs_partition_spec


def get_partitioned_spmd_model_decode_fn(jax_task, init_key,
                                         model_state_partition_specs,
                                         inputs_shape: NestedShapeDtypeStruct):
  """Return sharded decode step function and input partition spec.

  Args:
    jax_task: Task instance.
    init_key: PRNGKey for initializing the model variables.
    model_state_partition_specs: A TrainState contains PartitionSpecs for all
      the variables.
    inputs_shape: Shape of the inputs for use in pjit sharding.

  Returns:
    (decode_step_fn, inputs_partition_spec):
    The decode step function, and input partition spec.
  """
  task_p = jax_task.params
  model_p = task_p.model
  mesh_names = task_p.model.mesh_axis_names
  model = jax_task.model

  # Compute inputs PartitionSpec from inputs_shape
  inputs_partition_spec_fn = functools.partial(
      shard_on_batch_dim_partition_spec, mesh_names)
  reshard_inputs_fn = functools.partial(reshard_input_based_on_rank_fn,
                                        task_p.train.inputs_split_mapping,
                                        mesh_names)

  inputs_partition_spec = tf.nest.map_structure(inputs_partition_spec_fn,
                                                inputs_shape)

  # TODO(b/198356509): Fix this so that prng_key is no longer replicated, as
  # we want each core to not have identical random behavior.
  prng_key_partition_spec = base_layer.to_partition_spec((None,), mesh_names)

  def _maybe_pad(x, pspec, shape):
    return _maybe_pad_uneven_sharding(x, pspec, shape,
                                      model_p.device_mesh.shape,
                                      model_p.mesh_axis_names)

  model_state_unpadded_shapes = jax_task.create_train_state_unpadded_shapes(
      model.vars, discard_opt_states=True)

  def _decode_step(states, prng_key, inputs):
    inputs = jax.tree_map(reshard_inputs_fn, inputs)
    states = jax.tree_map(_maybe_slice_uneven_sharding, states,
                          model_state_partition_specs,
                          model_state_unpadded_shapes)
    # Right now we only pad the vars, and decode doesn't output vars so we do
    # not need to pad at the end.
    return decode_step(
        model, states, prng_key, inputs, fprop_dtype=task_p.model.fprop_dtype)

  def init_model_from_seed(init_key):
    outs = initialize_model_state(jax_task, init_key, discard_opt_states=True)
    return jax.tree_map(_maybe_pad, outs, model_state_partition_specs,
                        model_state_unpadded_shapes)

  var_padded_shapes = jax.eval_shape(init_model_from_seed, init_key)

  decode_out_shapes = jax.eval_shape(_decode_step, var_padded_shapes, init_key,
                                     inputs_shape)

  decode_fn_in_partition_specs = (model_state_partition_specs,
                                  prng_key_partition_spec,
                                  inputs_partition_spec)
  # decoder output are always replicated at the moment.
  decode_fn_out_partition_specs = tf.nest.map_structure(lambda _: None,
                                                        decode_out_shapes)
  decode_step_fn = pjit.pjit(
      _decode_step,
      in_shardings=decode_fn_in_partition_specs,
      out_shardings=decode_fn_out_partition_specs,
  )

  return decode_step_fn, inputs_partition_spec


def partition_spmd_model_decode(
    task_p: InstantiableParams,
    init_key: PRNGKey,
    inputs_shape: NestedShapeDtypeStruct,
) -> Tuple[TrainState, TrainState, TrainState, DecodeFn]:
  """Setup the SPMD model and return sharded decode step function.

  For partitioning inputs, it is assumed the `task_p.train` has a field
  `inputs_split_mapping` which further contains keys `map_1d`, `map_2d`, ...,
  etc., which specifies how to shard inputs of that corresponding dimension.

  Args:
    task_p: Task parameters of type NestedMap.
    init_key: PRNGKey for initializing the model variables.
    inputs_shape: Shape of the inputs for use in pjit sharding.

  Returns:
    (partitioned_train_state, inputs_partition_spec,
    train_state_partition_specs, decode_step_fn):
    The partitioned TrainState, input partition spec, the corresponding
    partitioned TrainState
    specs, the decode step function.
  """
  jax_task = task_p.Instantiate()
  # Initialize the partitioned vars.
  model_state_partition_specs, partitioned_train_state = (
      initialize_partitioned_model_states(
          jax_task, init_key, discard_opt_states=True))

  decode_step_fn, inputs_partition_spec = get_partitioned_spmd_model_decode_fn(
      jax_task, init_key, model_state_partition_specs, inputs_shape)

  return (partitioned_train_state, inputs_partition_spec,
          model_state_partition_specs, decode_step_fn)
