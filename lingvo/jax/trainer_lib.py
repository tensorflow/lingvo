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
"""Shared trainer lib utilities."""

import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import pjit
from lingvo.jax import base_layer
from lingvo.jax import model
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax import train_states
import tensorflow.compat.v2 as tf

JTensor = pytypes.JTensor
Model = model.BaseTask
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
InitFn = Callable[[NestedJTensor, JTensor, JTensor], NestedJTensor]
ExtendFn = Callable[[NestedJTensor, NestedMap, JTensor, JTensor, NestedMap],
                    Tuple[NestedJTensor, NestedJTensor]]


def InitializesModelState(jax_model: model.BaseTask,
                          prng_key: PRNGKey) -> TrainState:
  """Initializes the model states."""
  logging.info('init_var prng_seed: %s', prng_key)
  initial_vars = jax_model.InstantiateVariables(prng_key)
  logging.debug('initial_vars: %s', initial_vars)
  learnable_vars = tf.nest.map_structure(
      lambda v: not base_layer.VarNotTrainable(v), jax_model.vars)
  tf.nest.assert_same_structure(initial_vars, learnable_vars)
  return jax_model.CreateTrainState(initial_vars, jax_model.vars)


def ReplicateModelState(model_states: TrainState) -> TrainState:
  """Replicates the model states."""
  return jax.device_put_replicated(model_states, jax.local_devices())


def InitializeReplicateModelState(jax_model: model.BaseTask,
                                  prng_key: PRNGKey) -> TrainState:
  """Initializes and replicates the model states."""
  model_states = InitializesModelState(jax_model, prng_key)
  return ReplicateModelState(model_states)


def _MaybeToBFloat16(x: JTensor) -> JTensor:
  if x.dtype == jnp.float32:
    return x.astype(jnp.bfloat16)
  else:
    return x


def TrainStepSingleLearner(
    mdl: Model,
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
    mdl: An instance of model.BaseTask.
    states: An instance of model.TrainState.
    prng_key: A PRNGKey, of shape [2], of type np.uint32.
    inputs: Inputs to the mdl.FProp() function.
    data_parallel_axis_name: a string, the device axis to aggregate gradients
      over.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of the following elements.
    updated_states - updated states.
    loss - loss as computed by mdl.FProp.
    mean_metrics - a dict of metrics. Each element of the dict is a pair
    (metric, weight).
    per_example_out - auxilillary per-example output as computed in mdl.FProp.
    summary_tensors - A dict or nested map of summary tensors computed in
      forward as well as backward.
  """
  assert len(mdl.learners) == 1
  learner = mdl.learners[0]

  context_p = base_layer.JaxContext.Params().Set(do_eval=False)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, states.step)

  if data_parallel_axis_name:
    in_pmap = True
  else:
    in_pmap = False

  prng_key, subkey = jax.random.split(prng_key)

  def _LossFn(
      mdl_vars: NestedJTensor, inputs: NestedMap
  ) -> Tuple[JTensor, Tuple[Any, NestedMap, SummaryDict, SummaryDict]]:
    """Computes loss as well as other auxiliary outputs."""
    if fprop_dtype == jnp.float32:
      pass
    elif fprop_dtype == jnp.bfloat16:
      mdl_vars = jax.tree_map(_MaybeToBFloat16, mdl_vars)
      inputs = jax.tree_map(_MaybeToBFloat16, inputs)
    else:
      assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

    with base_layer.JaxContext.NewContext(
        params=context_p, prng_key=subkey, global_step=states.step):
      # Prepares mdl for fprop. This clears all forward-updated vars that kept
      # locally in mdl.
      mdl.PrepareFProp()

      metrics, per_example_output = mdl.FProp(mdl_vars, inputs)
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
      forward_updated_vars = mdl.forward_updated_vars
      # Finally, fetch all the summary tensors.
      summary_tensors = base_layer.AllSummaries()
    if fprop_dtype == jnp.bfloat16 and weighted_loss.dtype == fprop_dtype:
      weighted_loss = weighted_loss.astype(jnp.float32)
    return weighted_loss, (metrics, forward_updated_vars, summary_tensors,
                           per_example_output)

  grad_fn = jax.value_and_grad(_LossFn, has_aux=True)

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

  var_weight_params = mdl.vars
  tf.nest.assert_same_structure(states.mdl_vars, var_weight_params)
  tf.nest.assert_same_structure(states.mdl_vars, grads)
  tf.nest.assert_same_structure(states.mdl_vars, fwd_updated_vars)
  var_is_learnable = tf.nest.map_structure(
      lambda x: not base_layer.VarNotTrainable(x), var_weight_params)
  tf.nest.assert_same_structure(states.mdl_vars, var_is_learnable)

  def _MaybeZeroOutGradFn(var_grad, var, var_learnable):
    if var_learnable:
      return var_grad
    else:
      return jnp.zeros_like(var)

  # Zero-out gradient for non-learnable vars.
  grads = tf.nest.map_structure(_MaybeZeroOutGradFn, grads, states.mdl_vars,
                                var_is_learnable)

  if in_pmap:
    # Aggregate grads across different model replicas.
    grads = jax.lax.psum(grads, axis_name=data_parallel_axis_name)
  else:
    # No gradient aggregation is needed.
    pass

  # Carry out backward computation under a JaxContext.
  prng_key, subkey = jax.random.split(prng_key)
  with base_layer.JaxContext.NewContext(
      params=context_p, prng_key=subkey, global_step=states.step):

    # Add a summary for learning rate
    learning_rate = learner.optimizer.GetLearningRate(states.step)
    base_layer.AddSummary('learning_rate', learning_rate)

    # Apply gradient transformations.
    transformed_grads, new_opt_states = learner.UpdateStates(
        grads, states.opt_states[0], states.mdl_vars)

    # Gradient descent on learnable vars.
    mdl_vars = learner.ApplyGradient(states.mdl_vars, transformed_grads,
                                     var_is_learnable)

    def _SynchronizeVarsUsingMean(new_var: NestedMap,
                                  old_var: NestedMap) -> NestedMap:
      """Synchronize variables across a replica by averaging."""
      delta = new_var - old_var
      delta_mean = jax.lax.pmean(delta, axis_name=data_parallel_axis_name)
      updated_var = old_var + delta_mean
      return updated_var

    def _UpdateNonLearnableVar(old_var: NestedMap, new_var: NestedMap,
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
      if not base_layer.VarNotTrainable(var_params):
        assert new_var is None
        return old_var
      elif not in_pmap:
        # No aggregation is needed.
        assert new_var is not None
        return new_var
      elif base_layer.VarRequiresMeanSync(var_params):
        assert new_var is not None
        return _SynchronizeVarsUsingMean(new_var, old_var)
      else:
        raise ValueError('Non-trainable variables must have a cross-replica '
                         'synchronization method specified.')

    var_weight_params = mdl.vars
    tf.nest.assert_same_structure(mdl_vars, var_weight_params)
    mdl_vars = tf.nest.map_structure(_UpdateNonLearnableVar, mdl_vars,
                                     fwd_updated_vars, var_weight_params)

    new_states = states.NewState(mdl_vars=mdl_vars, opt_states=[new_opt_states])
    # Finally fetch all backward summary tensors.
    bwd_summary_tensors = base_layer.AllSummaries()

  summary_tensors = NestedMap(
      fwd_summary_tensors=fwd_summary_tensors,
      bwd_summary_tensors=bwd_summary_tensors)

  return (new_states, weighted_loss, mean_metrics, per_example_out,
          summary_tensors)


def EvalStepSingleLearner(
    mdl: Model,
    mdl_vars: NestedJTensor,
    prng_key: JTensor,
    global_step: JTensor,
    inputs: Union[JTensor, NestedMap],
    data_parallel_axis_name: Optional[str] = 'batch',
    fprop_dtype: jnp.dtype = jnp.float32) -> Tuple[Any, Any, Any, SummaryDict]:
  """Evaluates a model for a single step.

  This utility is specialized for the single learner case.

  Args:
    mdl: An instance of model.BaseTask.
    mdl_vars: model variables to be used during eval.
    prng_key: A prng seed, of shape [2], of type np.uint32.
    global_step: A global step tensor indicating how many steps a model has been
      trained.
    inputs: Inputs to the mdl.FProp() function.
    data_parallel_axis_name: a string, the device axis to aggregate gradients
      over.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of the following elements.
    loss - loss as computed by mdl.FProp.
    mean_metrics - a dict of metrics. Each element of the dict is a pair
    (metric, weight).
    per_example_out - auxilillary per-example output as computed in mdl.FProp.
    summary_tensors - A nested map or dict of summary tensors computed in
      forward as well as backward pass.
  """
  context_p = base_layer.JaxContext.Params().Set(do_eval=True)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, global_step)

  if fprop_dtype == jnp.float32:
    pass
  elif fprop_dtype == jnp.bfloat16:
    mdl_vars = jax.tree_map(_MaybeToBFloat16, mdl_vars)
    inputs = jax.tree_map(_MaybeToBFloat16, inputs)
  else:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  with base_layer.JaxContext.NewContext(
      params=context_p, prng_key=prng_key, global_step=global_step):
    # Prepares mdl for fprop. This clears all forward-updated vars that kept
    # locally in mdl.
    mdl.PrepareFProp()

    # Support multiple learners.
    assert len(mdl.learners) == 1
    learner = mdl.learners[0]

    metrics, per_example_out = mdl.FProp(mdl_vars, inputs)
    loss_name = learner.loss_name
    assert loss_name in metrics
    loss, loss_weight = metrics[loss_name]
    assert loss.ndim == 0, 'loss has to be a scalar.'
    assert loss_weight.ndim == 0, 'loss_weight has to be a scalar.'

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

    else:
      # No data_parallel_axis_name is specified, most likely this is evaling an
      # spmd model.
      mean_metrics = metrics
      mean_loss = loss

    summary_tensors = base_layer.AllSummaries()

  def _MaybeToFloat32(x):
    if x.dtype == jnp.bfloat16:
      return x.astype(jnp.float32)
    else:
      return x

  if fprop_dtype == jnp.bfloat16:
    mean_loss, mean_metrics, per_example_out, summary_tensors = jax.tree_map(
        _MaybeToFloat32,
        (mean_loss, mean_metrics, per_example_out, summary_tensors))

  return mean_loss, mean_metrics, per_example_out, summary_tensors


def InitializePartitionedModelStates(
    mdl: model.BaseTask,
    prng_key: PRNGKey,
) -> Tuple[TrainState, NestedShape, TrainState]:
  """Initializes model vars that are partitioned over TPU devices.

  This function is equivalent to calling a pjit-ted version of
  InitializesModelStates().

  Args:
    mdl: The model which is an instance of model.BaseTask.
    prng_key: A PRNGKey.

  Returns:
    The partitioned specs, the shapes of the partitioned vars, and the
    partitioned vars themselves.
  """
  mdl.InstantiateVariableConfigs()
  # At this point, variable specs are already known.
  var_specs = mdl.vars
  train_state_partition_specs = mdl.CreateTrainStatePartitionSpecs(var_specs)
  assert train_state_partition_specs is not None

  init_model_from_seed = functools.partial(InitializesModelState, mdl)

  in_shape = jax.ShapeDtypeStruct((2,), jnp.uint32)
  out_shape = jax.eval_shape(init_model_from_seed, in_shape)

  logging.info('in_shape: %s', in_shape)
  logging.info('out_shape: %s', out_shape)
  logging.info('train_state_partition_specs: %s', train_state_partition_specs)
  tf.nest.assert_same_structure(train_state_partition_specs, out_shape)

  init_fn = pjit.pjit(
      init_model_from_seed,
      in_axis_resources=(None,),
      out_axis_resources=train_state_partition_specs)

  assert base_layer.GlobalMeshDefined(), 'must be inside maps.mesh scope'
  partitioned_vars = init_fn(prng_key)

  # Make sure output is of the expected structure.
  tf.nest.assert_same_structure(out_shape, partitioned_vars)

  return train_state_partition_specs, out_shape, partitioned_vars


def InitDecoderState(mdl: Model, mdl_vars: NestedJTensor, prng_key: JTensor,
                     global_step: JTensor, target_batch_size: int,
                     target_max_length: int,
                     fprop_dtype: jnp.dtype) -> NestedJTensor:
  """Initializes the decoder states.

  Args:
    mdl: An instance of model.BaseTask.
    mdl_vars: Model variables used during decoding.
    prng_key: A prng seed, of shape [2], of type np.uint32.
    global_step: A counter of how many steps mdl has been trained.
    target_batch_size: int, target batch size.
    target_max_length: int, target max sequence length.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    Initial cached decoder state.
  """
  if fprop_dtype == jnp.float32:
    pass
  elif fprop_dtype == jnp.bfloat16:
    mdl_vars = jax.tree_map(_MaybeToBFloat16, mdl_vars)
  else:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  context_p = base_layer.JaxContext.Params().Set(do_eval=True)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, global_step)
  with base_layer.JaxContext.NewContext(
      params=context_p, prng_key=prng_key, global_step=global_step):
    # Prepares mdl for fprop. This clears all forward-updated vars that kept
    # locally in mdl.
    mdl.PrepareFProp()

    initial_decoder_state = mdl.lm.InitStates(
        mdl_vars.lm,
        target_batch_size=target_batch_size,
        target_max_length=target_max_length)

  return initial_decoder_state


def ShardOnBatchDimPartitionSpec(mesh_names: Sequence[str],
                                 x: jax.ShapeDtypeStruct) -> pjit.PartitionSpec:
  """Fully shards x on the batch dimension."""
  x_dim = len(x.shape)
  assert x_dim >= 1
  sharding = [-1] * x_dim
  # Assume the first dim is batch, and fully shard the batch dim over the entire
  # mesh.
  sharding[0] = tuple(mesh_names)
  return base_layer.ToPartitionSpec(sharding, mesh_names)


def ReshardInputBasedOnRankFn(
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
    return base_layer.MaybeShard(x, mapping_dict[key], mesh_names)
  else:
    return x


def InferPartitionSpecBasedOnRankFn(
    mapping_dict: Dict[str, base_layer.SplitDimsMapping],
    mesh_names: Sequence[str],
    x: JTensor,
) -> Optional[pjit.PartitionSpec]:
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
    return base_layer.ToPartitionSpec(mapping_dict[key], mesh_names)


def ExtendDecoderState(
    mdl: Model, mdl_vars: NestedJTensor, decoder_state: NestedJTensor,
    prng_key: JTensor, global_step: JTensor, step_input: NestedJTensor,
    fprop_dtype: jnp.dtype) -> Tuple[NestedJTensor, NestedJTensor]:
  """Extends the decoder states.

  Args:
    mdl: An instance of model.BaseTask.
    mdl_vars: model variables used for decoding.
    decoder_state: Cached decoder state.
    prng_key: A prng seed, of shape [2], of type np.uint32.
    global_step: a global step tensor indicating how many steps a model has been
      trained.
    step_input: Input to the current decoder step.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A pair (xent_output, cached_state).
  """
  if fprop_dtype == jnp.float32:
    pass
  elif fprop_dtype == jnp.bfloat16:
    mdl_vars = jax.tree_map(_MaybeToBFloat16, mdl_vars)
    step_input = jax.tree_map(_MaybeToBFloat16, step_input)
  else:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  context_p = base_layer.JaxContext.Params().Set(do_eval=True)
  # Fold in decoder step as part of the random seed key, so that random
  # numbers depends on current decoding step.
  prng_key = jax.random.fold_in(prng_key, global_step)
  with base_layer.JaxContext.NewContext(
      params=context_p, prng_key=prng_key, global_step=global_step):
    # Prepares mdl for fprop. This clears all forward-updated vars that kept
    # locally in mdl.
    mdl.PrepareFProp()

    # TODO(yonghui): Fix me. ExtendDecoderState shouldn't try to access specific
    # member of the input arg (like mdl_vars.lm here).
    cached_state, xent_output = mdl.lm.ExtendStep(
        mdl_vars.lm, decoder_state, inputs=step_input)
  return cached_state, xent_output


def PartitionSpmdModel(
    mdl_params: InstantiableParams,
    init_key: PRNGKey,
    inputs_shape: NestedShapeDtypeStruct,
    decoder_mdl_params: Optional[InstantiableParams] = None,
    decoder_inputs_shape: Optional[NestedShapeDtypeStruct] = None,
    decoder_batch_size: Optional[int] = None,
    decoder_max_length: Optional[int] = None
) -> Tuple[TrainState, TrainState, TrainStepFn, EvalStepFn, Optional[InitFn],
           Optional[ExtendFn], int]:
  """Setup the SPMD model and return sharded train, eval, init and extend steps.

  For partitioning inputs, it is assumed the `mdl_params` has a field
  `inputs_split_mapping` which further contains keys `map_1d`, `map_2d`, ...,
  etc., which specifies how to shard inputs of that corresponding dimension.
  If `decoder_mdl_params` is not None, then it is assumed that it contains
  fields `decoder_inputs_split_mapping` and `decoder_states_split_mapping`,
  which similarly contain keys `map_0d`, `map_1d` etc., which specify how to
  shard decoder inputs and decoder states of that dimension respectively.

  Args:
    mdl_params: Model parameters of type NestedMap.
    init_key: PRNGKey for initializing the model variables.
    inputs_shape: Shape of the inputs for use in pjit sharding.
    decoder_mdl_params: Model parameters used to build the decoder functions
      (Optional). If this is not None, then decoder functions are returned.
    decoder_inputs_shape: Shape of decoder inputs (Optional).
    decoder_batch_size: Batch size to be used for sharding decoding step
      (Optional).
    decoder_max_length: Maximum sequence length to be used for sharding decoding
      step (Optional).

  Returns:
    (partitioned_train_state, train_state_partition_specs, train_step_fn,
    eval_step_fn, decoder_init_fn, decoder_extend_fn, total_num_params): The
    partitioned TrainState, the corresponding partitioned TrainState specs,
    the train step function, eval step function and total number of parameters.
    Optionally, if `decoder_mdl_params` is not None, then an init cache step
    and extend cache step are also returned, otherwise None values are returned
    in their place.
  """
  mesh_names = mdl_params.mesh_axis_names
  mdl = mdl_params.Instantiate()

  # Compute inputs PartitionSpec from inputs_shape
  inputs_partition_spec_fn = functools.partial(ShardOnBatchDimPartitionSpec,
                                               mdl_params.mesh_axis_names)
  reshard_inputs_fn = functools.partial(ReshardInputBasedOnRankFn,
                                        mdl_params.train.inputs_split_mapping,
                                        mdl_params.mesh_axis_names)

  inputs_partition_spec = tf.nest.map_structure(inputs_partition_spec_fn,
                                                inputs_shape)

  # Initialize the partitioned vars.
  train_state_partition_specs, var_shapes, partitioned_train_state = (
      InitializePartitionedModelStates(mdl, init_key))
  total_num_params = mdl.total_num_vars

  prng_key_shape = jax.ShapeDtypeStruct((2,), jnp.uint32)
  # TODO(bf-jax): prng_key is replicated. Would this be a problem?
  prng_key_partition_spec = base_layer.ToPartitionSpec((None,), mesh_names)

  def _TrainStep(state, prng_key, inputs):
    # Reshard inputs.
    inputs = jax.tree_map(reshard_inputs_fn, inputs)
    return TrainStepSingleLearner(
        mdl,
        state,
        prng_key,
        inputs,
        data_parallel_axis_name=None,
        fprop_dtype=mdl_params.fprop_dtype)

  def _EvalStep(mdl_vars, prng_key, global_step, inputs):
    # Reshard inputs.
    inputs = jax.tree_map(reshard_inputs_fn, inputs)
    return EvalStepSingleLearner(
        mdl,
        mdl_vars,
        prng_key,
        global_step,
        inputs,
        data_parallel_axis_name=None,
        fprop_dtype=mdl_params.fprop_dtype)

  train_out_shapes = jax.eval_shape(_TrainStep, var_shapes, prng_key_shape,
                                    inputs_shape)

  eval_out_shapes = jax.eval_shape(_EvalStep, var_shapes.mdl_vars,
                                   prng_key_shape, var_shapes.step,
                                   inputs_shape)

  def _PartitionSpecFromShape(x_shape):
    # Currently, all the outputs are fully replicated.
    # TODO(yonghui): Somehow fetch the output sharding spec from _EvalStep fn.
    del x_shape
    return None

  train_fn_in_partition_specs = (train_state_partition_specs,
                                 prng_key_partition_spec, inputs_partition_spec)
  train_fn_out_replicated_specs = tf.nest.map_structure(_PartitionSpecFromShape,
                                                        train_out_shapes)
  # Here we assume the first output is the train-state.
  # Expcept for the first train_state output, others outputs are explicitly
  # replicated.
  train_fn_out_partition_specs = tuple([train_state_partition_specs] +
                                       list(train_fn_out_replicated_specs[1:]))
  tf.nest.assert_same_structure(train_fn_out_replicated_specs,
                                train_fn_out_partition_specs)
  tf.nest.assert_same_structure(train_fn_out_partition_specs, train_out_shapes)

  eval_fn_in_partition_specs = (train_state_partition_specs.mdl_vars,
                                prng_key_partition_spec,
                                train_state_partition_specs.step,
                                inputs_partition_spec)
  eval_fn_out_partition_specs = tf.nest.map_structure(_PartitionSpecFromShape,
                                                      eval_out_shapes)

  # pjit-ed train step function.
  train_step = pjit.pjit(
      _TrainStep,
      in_axis_resources=train_fn_in_partition_specs,
      out_axis_resources=train_fn_out_partition_specs,
      donate_argnums=(0,))

  # pjit-ed eval step function.
  eval_step = pjit.pjit(
      _EvalStep,
      in_axis_resources=eval_fn_in_partition_specs,
      out_axis_resources=eval_fn_out_partition_specs)

  decoder_init_fn = None
  decoder_extend_fn = None
  if decoder_mdl_params is not None:
    assert decoder_inputs_shape is not None
    assert decoder_batch_size is not None
    assert decoder_max_length is not None
    decode_mdl = decoder_mdl_params.Instantiate()

    # Initializes all the meta data.
    decode_mdl.InstantiateVariableConfigs()

    def _InitDecoderState(mdl_vars, prng_key, global_step):
      return InitDecoderState(
          decode_mdl,
          mdl_vars,
          prng_key,
          global_step,
          target_batch_size=decoder_batch_size,
          target_max_length=decoder_max_length,
          fprop_dtype=decoder_mdl_params.fprop_dtype)

    reshard_decoder_inputs_fn = functools.partial(
        ReshardInputBasedOnRankFn,
        decoder_mdl_params.train.decoder_inputs_split_mapping,
        decoder_mdl_params.mesh_axis_names)

    def _ExtendDecoderState(mdl_vars, decoder_state, prng_key, global_step,
                            step_input):
      # Reshard the decoder inputs.
      step_input = jax.tree_map(reshard_decoder_inputs_fn, step_input)
      return ExtendDecoderState(decode_mdl, mdl_vars, decoder_state, prng_key,
                                global_step, step_input,
                                decoder_mdl_params.fprop_dtype)

    init_decoder_state_fn = _InitDecoderState
    extend_decoder_state_fn = _ExtendDecoderState

    decoder_state_shapes = jax.eval_shape(_InitDecoderState,
                                          var_shapes.mdl_vars, prng_key_shape,
                                          var_shapes.step)

    decoder_inputs_partition_spec_fn = functools.partial(
        ShardOnBatchDimPartitionSpec, decoder_mdl_params.mesh_axis_names)
    decoder_inputs_partition_spec = tf.nest.map_structure(
        decoder_inputs_partition_spec_fn, decoder_inputs_shape)

    # Compute the Decoder state partition specs from decoder state shapes.
    decoder_state_partition_spec_fn = functools.partial(
        InferPartitionSpecBasedOnRankFn,
        decoder_mdl_params.train.decoder_states_split_mapping,
        decoder_mdl_params.mesh_axis_names)
    decoder_state_partition_specs = tf.nest.map_structure(
        decoder_state_partition_spec_fn, decoder_state_shapes)

    init_fn_in_partition_specs = (train_state_partition_specs.mdl_vars,
                                  prng_key_partition_spec,
                                  train_state_partition_specs.step)
    init_fn_out_partition_specs = decoder_state_partition_specs

    _, xent_out_shapes = jax.eval_shape(extend_decoder_state_fn,
                                        var_shapes.mdl_vars,
                                        decoder_state_shapes, prng_key_shape,
                                        var_shapes.step, decoder_inputs_shape)
    # decoder output are always replicated at the moment.
    # TODO(yonghui): Fix me.
    xent_out_partition_specs = tf.nest.map_structure(lambda x: None,
                                                     xent_out_shapes)

    extend_fn_in_partition_specs = (train_state_partition_specs.mdl_vars,
                                    decoder_state_partition_specs,
                                    prng_key_partition_spec,
                                    train_state_partition_specs.step,
                                    decoder_inputs_partition_spec)

    extend_fn_out_partition_specs = (decoder_state_partition_specs,
                                     xent_out_partition_specs)

    # pjit-ed decoder init function.
    decoder_init_fn = pjit.pjit(
        init_decoder_state_fn,
        in_axis_resources=init_fn_in_partition_specs,
        out_axis_resources=init_fn_out_partition_specs)

    # pjit-ed decoder extend step function.
    decoder_extend_fn = pjit.pjit(
        extend_decoder_state_fn,
        in_axis_resources=extend_fn_in_partition_specs,
        out_axis_resources=extend_fn_out_partition_specs)

  return (partitioned_train_state, train_state_partition_specs, train_step,
          eval_step, decoder_init_fn, decoder_extend_fn, total_num_params)
