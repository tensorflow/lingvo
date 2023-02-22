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
"""Evaluation loop for lingvo Jax model."""

import contextlib
import functools
import hashlib
import os
import time
from typing import List, Optional, Sequence

from absl import logging
import jax
from jax.experimental import mesh_utils
from lingvo.jax import base_input
from lingvo.jax import base_layer
from lingvo.jax import base_metrics
from lingvo.jax import base_model_params
from lingvo.jax import base_task
from lingvo.jax import checkpoint_pb2
from lingvo.jax import checkpoints
from lingvo.jax import io_utils
from lingvo.jax import model_utils
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax import summary_utils
from lingvo.jax import train_states
from lingvo.jax import trainer_lib
import tensorflow.compat.v2 as tf

BaseModelParamsT = base_model_params.BaseModelParamsT
CheckpointType = checkpoint_pb2.CheckpointType
InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
TrainState = train_states.TrainState
SummaryWriter = tf.summary.SummaryWriter


def maybe_ema(model_states):
  """Finds the ema state from optimizer states."""
  if not model_states.opt_states:
    return model_states
  for i in range(len(model_states.opt_states[0])):
    if 'ema' in model_states.opt_states[0][i]:
      return TrainState(
          step=model_states.step,
          mdl_vars=model_states.opt_states[0][i].ema,  # pytype: disable=attribute-error  # jax-ndarray
          opt_states={})
  return model_states


def evaluate(
    model_name: str,
    job_log_dir: Optional[str],
    multi_host_checkpointing: Optional[bool],
    maybe_use_persistence_checkpointing: bool,
) -> None:
  """Runs the evaluation loop on the entire eval data set.

  Args:
    model_name: The name of the model from the registry to evaluate.
    job_log_dir: The directory for the job logs.
    multi_host_checkpointing: Whether to use multi-host checkpointing.
    maybe_use_persistence_checkpointing: If set, it will try to use
      persistence-based checkpointing if suitable.
  """
  model_config = model_utils.get_model(model_name)()
  task_p = model_config.task()
  model_p = task_p.model
  eval_input_p = [v for v in model_config.datasets() if not v.is_training]
  for inp in eval_input_p:
    inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  if model_p.device_mesh is not None:
    checkpoint_type = checkpoints.retrieve_checkpoint_type(
        multi_host_checkpointing, maybe_use_persistence_checkpointing, task_p)
    evaluate_spmd_model(task_p, eval_input_p, job_log_dir, checkpoint_type)
  else:
    evaluate_pmap_model(task_p, eval_input_p, job_log_dir)


def evaluate_pmap_model(
    task_p: InstantiableParams,
    eval_input_p: Sequence[InstantiableParams],
    job_log_dir: Optional[str],
) -> None:
  """Runs the evaluation loop on the entire test dataset for PMAP model.

  Args:
    task_p: Params for the task encapsulating the data parallel model.
    eval_input_p: List of params for the eval data input pipelines.
    job_log_dir: Directory for the job logs.
  """
  logging.info('Using pmap for data parallelism.')
  jax_task = task_p.Instantiate()
  eval_input_pipelines = [input_p.Instantiate() for input_p in eval_input_p]
  # TODO(shafey): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  checkpoint_dir = os.path.join(job_log_dir, 'checkpoints')
  # Restore flax checkpoints still required bak variables in TrainState
  # TODO(pax): add is_eval=True to initialize_model_state
  model_states = trainer_lib.initialize_model_state(jax_task, init_key)
  # Pmap does not use GDA, and so global_mesh and mesh_axes are None.
  model_states = checkpoints.restore_checkpoint(model_states, checkpoint_dir)
  replicated_model_states = trainer_lib.replicate_model_state(model_states)
  logging.info('replicated_model_states: %s',
               jax.tree_map(lambda x: x.shape, replicated_model_states))
  # From now on, different replicas should use different random seeds.
  # Here, each process will have its unique prng_key.
  # prng_key will be further split so that each core on a host will get
  # different prng_key.
  prng_key = jax.random.fold_in(prng_key, jax.process_index())
  logging.info('root prng_key: %s', prng_key)

  def eval_step(mdl_states, prng_key, inputs):
    mdl_states = trainer_lib.train_state_for_eval_step(mdl_states)
    return trainer_lib.eval_step_single_learner(
        jax_task,
        mdl_states,
        prng_key,
        inputs,
        data_parallel_axis_name='batch',
        fprop_dtype=jax_task.model.fprop_dtype)

  num_devices = jax.local_device_count()
  prng_key, eval_key = jax.random.split(prng_key)
  eval_prng_seed = jax.random.split(eval_key, num=num_devices)
  logging.info('eval prng_seed: %s', eval_prng_seed)

  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  logging.info('Evaluation loop starting...')
  summary_base_dir = os.path.join(job_log_dir, 'summaries')
  summary_eval_dirs = [
      os.path.join(summary_base_dir, f'eval_test_{split}')
      for split, _ in enumerate(eval_input_p)
  ]

  num_steps = [
      -1 if p.reset_for_eval else p.eval_loop_num_batches for p in eval_input_p
  ]
  last_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
  with contextlib.ExitStack() as exit_stack:
    eval_summary_writers = [
        exit_stack.enter_context(summary_utils.get_summary_writer(d))
        for d in summary_eval_dirs
    ]

    while True:
      step_i = int(jax.device_get(replicated_model_states.step)[0])
      eval_step = functools.partial(p_eval_step,
                                    maybe_ema(replicated_model_states),
                                    eval_prng_seed)
      # Run the eval loop.
      model_utils.run_eval_loop_over_test_splits(
          num_steps,
          eval_step,
          eval_summary_writers,
          step_i,
          eval_input_pipelines,
          reshard_inputs=True)
      # If the last check point evaluated matches max train steps, exit.
      if last_checkpoint is not None:
        last_ckpt_step = checkpoints.get_step_from_checkpoint_asset(
            last_checkpoint)
        exceeded_ckpt = last_ckpt_step + task_p.train.save_interval_steps
        if exceeded_ckpt >= task_p.train.num_train_steps:
          break
      # Release replicated_model_states.
      del replicated_model_states
      new_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
      while new_checkpoint == last_checkpoint:
        # Sleep for a minute.
        time.sleep(60)
        new_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
      # There must be a new checkpoint here.
      logging.info('Found new checkpoint: %s', new_checkpoint)
      model_states = checkpoints.restore_checkpoint(model_states,
                                                    checkpoint_dir)
      replicated_model_states = trainer_lib.replicate_model_state(model_states)
      last_checkpoint = new_checkpoint


def evaluate_spmd_model(
    task_p: InstantiableParams,
    eval_input_p: Sequence[InstantiableParams],
    job_log_dir: Optional[str],
    checkpoint_type: CheckpointType,
) -> None:
  """Runs the evaluation loop on the entire test dataset for SPMD model.

  Args:
    task_p: Params of the task encapsulating an SPMD model.
    eval_input_p: List of Params for the eval data pipelines.
    job_log_dir: Directory for the job logs.
    checkpoint_type: Type of model checkpointing method to use.
  """
  logging.info('Using SPMD sharding for model parallelism.')
  eval_input_pipelines = [input_p.Instantiate() for input_p in eval_input_p]
  # TODO(bf-jax): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  checkpoint_dir = os.path.join(job_log_dir, 'checkpoints')
  # Note that GDA checkpoint requires all processes to participate in
  # checkpointing but it does not require a separate checkpoint_dir per process.
  if checkpoint_type == CheckpointType.CHECKPOINT_MULTI_HOST_FLAX:
    checkpoint_task_dir = os.path.join(checkpoint_dir,
                                       f'{jax.process_index():03d}')
  else:
    checkpoint_task_dir = checkpoint_dir

  multi_host_checkpointing = bool(checkpoint_type in {
      CheckpointType.CHECKPOINT_MULTI_HOST_FLAX, CheckpointType.CHECKPOINT_GDA
  })

  def get_shape_dtype(x):
    y = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return y

  # Do not ues eval_input_pipelines[0] directly.
  sample_model_inputs = eval_input_p[0].Instantiate().get_next()
  inputs_shape = tf.nest.map_structure(get_shape_dtype, sample_model_inputs)

  jax_task = task_p.Instantiate()
  model_p = task_p.model
  mesh_shape = model_p.device_mesh.shape
  device_mesh = mesh_utils.create_device_mesh(mesh_shape)
  logging.info('device_mesh: %s', device_mesh)
  global_mesh = jax.sharding.Mesh(device_mesh, model_p.mesh_axis_names)
  use_gda_checkpoint = jax.config.jax_parallel_functions_output_gda
  with global_mesh:
    jax_task.model.instantiate_variable_configs()
    # Restore flax checkpoints still required backward variables in TrainState
    # TODO(pax): set is_eval=True for all ckpt types.
    if use_gda_checkpoint:
      partitioned_specs = jax_task.create_train_state_partition_specs(
          jax_task.model.vars, discard_opt_states=True)
      partitioned_train_state = checkpoints.restore_checkpoint(
          None,
          checkpoint_task_dir,
          global_mesh=global_mesh,
          checkpoint_type=checkpoint_type,
          state_specs=partitioned_specs)
      eval_step, inputs_partition_specs = (
          trainer_lib.get_partitioned_spmd_model_step_fn(
              jax_task,
              init_key,
              partitioned_specs,
              inputs_shape,
              is_eval=True))
    else:
      (partitioned_train_state, partitioned_specs, inputs_partition_specs, _,
       eval_step, _) = trainer_lib.partition_spmd_model(task_p, init_key,
                                                        inputs_shape)
      partitioned_train_state = checkpoints.restore_checkpoint(
          partitioned_train_state,
          checkpoint_task_dir,
          global_mesh=global_mesh,
          checkpoint_type=checkpoint_type,
          state_specs=partitioned_specs)

    logging.info('partitioned_train_state: %s',
                 jax.tree_map(lambda x: x.shape, partitioned_train_state))
    if multi_host_checkpointing:
      py_utils.sync_global_devices(f'checkpointer:restored:{checkpoint_dir}')

    # We do not fold in jax.process_index in contrast to the pmap version and
    # use a single global key instead to rely on pjit to split for different
    # replicas.
    logging.info('root prng_key: %s', prng_key)
    prng_key, eval_key = jax.random.split(prng_key)
    logging.info('eval prng_key: %s', eval_key)

    logging.info('Evaluation loop starting...')
    summary_base_dir = os.path.join(job_log_dir, 'summaries')
    summary_eval_dirs = [
        os.path.join(summary_base_dir, f'eval_{split}')
        for split, _ in enumerate(eval_input_p)
    ]

    num_steps = [-1 if p.reset_for_eval else 1 for p in eval_input_p]
    last_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
    with contextlib.ExitStack() as exit_stack:
      eval_summary_writers = [
          exit_stack.enter_context(summary_utils.get_summary_writer(d))
          for d in summary_eval_dirs
      ]
      while True:
        step_i = int(jax.device_get(partitioned_train_state.step))
        eval_step_fn = functools.partial(
            eval_step,
            trainer_lib.train_state_for_eval_step(partitioned_train_state),
            eval_key)
        # Run the eval loop.
        model_utils.run_eval_loop_over_test_splits(
            num_steps,
            eval_step_fn,
            eval_summary_writers,
            step_i,
            eval_input_pipelines,
            inputs_partition_specs,
            inputs_shape,
            global_mesh,
            reshard_inputs=False)
        # If the last check point evaluated matches max train steps, exit.
        if last_checkpoint is not None:
          last_ckpt_step = checkpoints.get_step_from_checkpoint_asset(
              last_checkpoint)
          exceeded_ckpt = last_ckpt_step + task_p.train.save_interval_steps
          if exceeded_ckpt >= task_p.train.num_train_steps:
            break
        new_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
        while new_checkpoint == last_checkpoint:
          # Sleep for a minute.
          time.sleep(60)
          new_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
        # There must be a new checkpoint here.
        logging.info('Found new checkpoint: %s', new_checkpoint)
        partitioned_train_state = checkpoints.restore_checkpoint(
            None if use_gda_checkpoint else partitioned_train_state,
            checkpoint_task_dir,
            global_mesh=global_mesh,
            checkpoint_type=checkpoint_type,
            state_specs=partitioned_specs)
        if multi_host_checkpointing:
          py_utils.sync_global_devices(
              f'checkpointer:restored:{checkpoint_dir}')
        last_checkpoint = new_checkpoint


def decode(
    model_name: str,
    job_log_dir: Optional[str],
    multi_host_checkpointing: Optional[bool],
    maybe_use_persistence_checkpointing: bool,
    restore_checkpoint_dir: Optional[str],
    restore_checkpoint_step: Optional[int],
    continuous_decode: bool,
) -> None:
  """Runs decoding once on the decoder datasets.

  Args:
    model_name: The name of the model from the registry to evaluate.
    job_log_dir: The directory for the job logs.
    multi_host_checkpointing: Whether to use multi-host checkpointing.
    maybe_use_persistence_checkpointing: If set, it will try to use
      persistence-based checkpointing if suitable.
    restore_checkpoint_dir: The directory from which to restore checkpoint.
    restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
      try to restore from the latest checkpoint if any.
    continuous_decode: whether to continuously decode on the latest ckpt.
  """
  logging.info('running decode_once on model %s restored from %s', model_name,
               restore_checkpoint_dir)
  model_config = model_utils.get_model(model_name)()
  task_p = model_config.task()
  model_p = task_p.model
  decoder_inputs = model_config.decoder_datasets()
  if not decoder_inputs:
    return
  for inp in decoder_inputs:
    inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  if model_p.device_mesh is not None:
    if continuous_decode:
      raise NotImplementedError('http://b/214589358: not supported')
    checkpoint_type = checkpoints.retrieve_checkpoint_type(
        multi_host_checkpointing, maybe_use_persistence_checkpointing, task_p)
    decode_once_spmd_model(task_p, decoder_inputs, job_log_dir, checkpoint_type,
                           restore_checkpoint_dir, restore_checkpoint_step)
  else:
    decode_pmap_model(task_p, decoder_inputs, job_log_dir,
                      restore_checkpoint_dir, restore_checkpoint_step,
                      continuous_decode)


def _get_dir_names(input_p: Sequence[InstantiableParams]) -> Sequence[str]:
  """Returns a list of same length for parent dir names for each dataset."""
  uniq_names = set()
  ret = []
  for idx, p in enumerate(input_p):
    name = p.name or f'decode_test_{idx}'
    if p.name and p.name in uniq_names:
      name = f'{p.name}_{idx}'
    if name in uniq_names:
      suffix = hashlib.md5(name.encode()).hexdigest()[-5:]
      name = f'{name}_{suffix}'
      assert name not in uniq_names
    uniq_names.add(name)
    ret.append(name)
  return ret


def _get_step(step: base_layer.JTensorOrPartitionSpec) -> int:
  """Returns an int for the current global step."""
  if step.ndim == 0:  # pytype: disable=attribute-error  # jax-ndarray
    return jax.device_get(step)
  if step.ndim == 1:  # pytype: disable=attribute-error  # jax-ndarray
    return jax.device_get(step[0])
  raise ValueError(
      f'Expecting a replicated 1D global step (got ndim=`{step.ndim}`).')  # pytype: disable=attribute-error  # jax-ndarray


def _get_filename(step: base_layer.JTensorOrPartitionSpec) -> str:
  """Returns a filename for the given step."""
  step_num = _get_step(step)
  return f'decoder_out_{step_num}_shard_{jax.process_index()}'


def decode_pmap_model(
    task_p: InstantiableParams,
    input_p: Sequence[InstantiableParams],
    job_log_dir: Optional[str],
    restore_checkpoint_dir: Optional[str],
    restore_checkpoint_step: Optional[int],
    continuous_decode: bool,
) -> None:
  """Runs the decoding on the entire decoder datasets for a PMAP model.

  Args:
    task_p: Params of the task encapsulating a the data parallel model.
    input_p: List of input params to be decoded.
    job_log_dir: Directory for the job logs.
    restore_checkpoint_dir: The directory from which to restore checkpoint. If
      None, uses job_log_dir.
    restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
      try to restore from the latest checkpoint if any.
    continuous_decode: whether to continuously decode on the latest ckpt.
  """
  if continuous_decode and restore_checkpoint_step is not None:
    raise ValueError('Continuous decoding mode requires restore_checkpoint_step'
                     '=None, actual restore_checkpoint_step='
                     f'{restore_checkpoint_step}')
  restore_checkpoint_dir = restore_checkpoint_dir or os.path.join(
      job_log_dir, 'checkpoints')

  # TODO(shafey): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  # From now on, different replicas should use different random seeds.
  # Here, each process will have its unique prng_key.
  # prng_key will be further split so that each core on a host will get
  # different prng_key.
  prng_key = jax.random.fold_in(prng_key, jax.process_index())
  logging.info('root prng_key: %s', prng_key)
  prng_key, eval_key = jax.random.split(prng_key)
  prng_seed = jax.random.split(eval_key, num=jax.local_device_count())
  logging.info('decoder prng_seed: %s', prng_seed)

  inputs = [p.Instantiate() for p in input_p]
  summary_base_dir = os.path.join(job_log_dir, 'summaries')
  dirnames = _get_dir_names(input_p)
  summary_decode_dirs = [
      os.path.join(summary_base_dir, f'decode_test_{dirnames[split]}')
      for split, _ in enumerate(input_p)
  ]
  with contextlib.ExitStack() as exit_stack:
    summary_writers = [
        exit_stack.enter_context(summary_utils.get_summary_writer(d))
        for d in summary_decode_dirs
    ]

    jax_task = task_p.Instantiate()
    # Restore flax checkpoints still required bak variables in TrainState
    # TODO(pax): add is_eval=True to initialize_model_state
    model_states = trainer_lib.initialize_model_state(jax_task, init_key)
    model_states = checkpoints.restore_checkpoint(
        model_states, restore_checkpoint_dir, step=restore_checkpoint_step)
    replicated_model_states = trainer_lib.replicate_model_state(model_states)
    logging.info('replicated_model_states: %s',
                 jax.tree_map(lambda x: x.shape, replicated_model_states))
    last_checkpoint = checkpoints.latest_checkpoint(restore_checkpoint_dir)

    while True:
      _decode_once_pmap_model(jax_task, task_p, inputs, input_p, prng_seed,
                              job_log_dir, replicated_model_states,
                              summary_writers)
      if not continuous_decode:
        break
      if last_checkpoint is not None:
        last_ckpt_step = int(last_checkpoint.split('_')[-1])
        exceeded_ckpt = last_ckpt_step + task_p.train.save_interval_steps
        if exceeded_ckpt >= task_p.train.num_train_steps:
          break
      # Release replicated_model_states.
      del replicated_model_states
      new_checkpoint = checkpoints.latest_checkpoint(restore_checkpoint_dir)
      while new_checkpoint == last_checkpoint:
        time.sleep(60)
        new_checkpoint = checkpoints.latest_checkpoint(restore_checkpoint_dir)
      logging.info('Found new checkpoint: %s', new_checkpoint)
      model_states = checkpoints.restore_checkpoint(model_states,
                                                    restore_checkpoint_dir)
      replicated_model_states = trainer_lib.replicate_model_state(model_states)
      last_checkpoint = new_checkpoint


def _decode_once_pmap_model(
    jax_task: base_task.SingleTask,
    task_p: InstantiableParams,
    inputs: List[base_input.BaseInput],
    input_p: Sequence[InstantiableParams],
    prng_seed: JTensor,
    job_log_dir: Optional[str],
    replicated_model_states: train_states.TrainState,
    summary_writers: List[SummaryWriter],
) -> None:
  """Runs the decoding on the entire decoder datasets for a PMAP model.

  Args:
    jax_task: instantiated model from task_p.
    task_p: Params for the task encapsulating a data parallel model.
    inputs: instantiated inputs.
    input_p: List of input params to be decoded.
    prng_seed: The prng seed used for decoding.
    job_log_dir: Directory for the job logs.
    replicated_model_states: A TrainState object.
    summary_writers: The summary writer objects to log summaries.
  """
  model = jax_task.model
  model_p = task_p.model
  metrics_p = task_p.metrics
  if not metrics_p:
    metrics_p = base_metrics.MeanMetrics.Params()
  decode_metrics = metrics_p.Instantiate()
  process_decode_metrics = metrics_p.Instantiate()

  step_i = _get_step(replicated_model_states.step)
  pmap_axis_name = 'batch'

  def decode_step(mdl_states, prng_key, inputs):
    mdl_states = trainer_lib.train_state_for_eval_step(mdl_states)
    metrics, out = trainer_lib.decode_step(model, mdl_states, prng_key, inputs,
                                           model_p.fprop_dtype)
    metrics = decode_metrics.aggregate(metrics)
    return metrics, out

  # As an example, suppose the output leaf from trainer_lib.decoder_step()
  # for each core has shape: [per_core_batch_size, decoding_length].
  # In the all_gather we set tiled=True, so the output chunks are all
  # concatenated into the existing batch axis, so we get shape
  # [num_cores x per_core_batch_size, decoding_length].
  # In the pmap call we set out_axes=None to not have to manually unreplicate,
  # so the output of pmap_decode_step() will have the same shape.
  #
  # Example code snippet showing this:
  #   # shape (8, 3, 2)
  #   x = jnp.tile(jnp.arange(8)[:, None, None],[1, 3, 2])
  #   # shape (24, 2)
  #   z = jax.pmap(
  #       lambda y: jax.lax.all_gather(y+1, axis_name='i', tiled=True),
  #       axis_name='i', out_axes=None)(x)
  #
  # We only aggregate metrics, not `out`, hence the tuple for out_axes.
  pmap_decode_step = jax.pmap(
      decode_step, axis_name=pmap_axis_name, out_axes=(None, 0))
  decode_step_func = functools.partial(pmap_decode_step,
                                       maybe_ema(replicated_model_states),
                                       prng_seed)

  num_steps = [
      -1 if p.reset_for_eval else p.eval_loop_num_batches for p in input_p
  ]
  decodes = [list() for _ in input_p]
  for split, num_split_steps in enumerate(num_steps):
    logging.info('Start decoding on input %s', input_p[split].name)
    step_num = 0
    while num_split_steps < 0 or step_num < num_split_steps:
      step_num += 1
      try:
        batch = inputs[split].get_next()
      except (tf.errors.OutOfRangeError, StopIteration):
        inputs[split].reset()
        break
      batch = tf.nest.map_structure(py_utils.reshard, batch)
      batch_metrics, out = decode_step_func(batch)
      # we store the metric directly as it has already been aggregated in
      # side decode_step_fun
      decode_metrics.store(batch_metrics)
      logging.info('Finished decoding input batch %d', step_num)

      out = tf.nest.map_structure(py_utils.unshard, out)
      process_metrics, processed = model.process_decode_out(inputs[split], out)
      decodes[split].extend(processed)
      logging.info('Finished processing decoded input batch %d', step_num)

      # Reshard the metrics for pmap.
      process_decode_metrics.update(process_metrics)

    with summary_writers[split].as_default():
      decode_metrics.summarize(step_i, 'decode_metrics')
      process_decode_metrics.summarize(step_i, 'process_decode_metrics')

  basedir = os.path.join(job_log_dir, 'decoder_out')
  dirnames = _get_dir_names(input_p)
  filename = _get_filename(replicated_model_states.step)
  for s in dirnames:
    dir_path = os.path.join(basedir, s)
    if not tf.io.gfile.exists(dir_path):
      tf.io.gfile.makedirs(dir_path)
  filenames = [os.path.join(basedir, s, filename) for s in dirnames]
  for split, output_file in enumerate(filenames):
    logging.info('Writing decoder output to %s with %d entries', output_file,
                 len(decodes[split]))
    io_utils.WriteKeyValuePairs(output_file, decodes[split])


def decode_once_spmd_model(
    task_p: InstantiableParams,
    input_p: Sequence[InstantiableParams],
    job_log_dir: Optional[str],
    checkpoint_type: CheckpointType,
    restore_checkpoint_dir: str,
    restore_checkpoint_step: Optional[int],
) -> None:
  """Runs the decoding once on the entire decoder datasets for SPMD model.

  Args:
    task_p: Params for the task that encapsulates an SPMD model.
    input_p: List of input params to be decoded.
    job_log_dir: Directory for the job logs.
    checkpoint_type: Type of model checkpointing method to use.
    restore_checkpoint_dir: The directory from which to restore checkpoint.
    restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
      try to restore from the latest checkpoint if any.
  """
  # TODO(bf-jax): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  if restore_checkpoint_dir:
    restore_checkpoint_parent_dir = restore_checkpoint_dir
    if checkpoint_type == CheckpointType.CHECKPOINT_MULTI_HOST_FLAX:
      # TODO(zhouwk): add sanity check on number of subdirs and number of
      # processes and fail early if unequal.
      restore_checkpoint_dir = os.path.join(restore_checkpoint_dir,
                                            f'{jax.process_index():03d}')

  multi_host_checkpointing = bool(checkpoint_type in {
      CheckpointType.CHECKPOINT_MULTI_HOST_FLAX, CheckpointType.CHECKPOINT_GDA
  })

  sample_inputs = input_p[0].Instantiate().get_next()
  inputs_shape = tf.nest.map_structure(py_utils.get_global_input_shape_dtype,
                                       sample_inputs)

  model_p = task_p.model
  # TODO(b/198356509): This is a hack for now as we need to change some
  # annotations for mode='decode'. A future cl will move this logic
  # to a more generic model_p.update_sharding_params_v1(mode='decode').
  model_p.lm = model_p.lm.cls.set_sharding_params_v1(
      model_p.lm,
      replica_axis=model_p.lm.mesh_axis_names[0],
      data_axis=model_p.lm.mesh_axis_names[1],
      mdl_axis=model_p.lm.mesh_axis_names[2],
      device_ids_mesh=model_p.lm.device_mesh,
      mesh_axis_names=model_p.lm.mesh_axis_names,
      mode='decode')

  mesh_shape = model_p.device_mesh.shape
  device_mesh = mesh_utils.create_device_mesh(mesh_shape)
  logging.info('device_mesh: %s', device_mesh)
  jax_task = task_p.Instantiate()
  global_mesh = jax.sharding.Mesh(device_mesh, model_p.mesh_axis_names)
  with global_mesh:
    if restore_checkpoint_dir:
      model = jax_task.model
      model.instantiate_variable_configs()
      # Get the metadata from variables instead of actually instantiating them.
      partitioned_specs = jax_task.create_train_state_partition_specs(
          model.vars, discard_opt_states=True)
      # Instantiate the TrainState directly from the checkpoint.
      partitioned_train_state = checkpoints.restore_checkpoint(
          None,
          restore_checkpoint_dir,
          global_mesh=global_mesh,
          checkpoint_type=checkpoint_type,
          state_specs=partitioned_specs,
          step=restore_checkpoint_step)
      if multi_host_checkpointing:
        py_utils.sync_global_devices(
            f'checkpointer:restored:{restore_checkpoint_parent_dir}')
      decode_step_fn, inputs_partition_spec = (
          trainer_lib.get_partitioned_spmd_model_decode_fn(
              jax_task, init_key, partitioned_specs, inputs_shape))
    else:
      # When restore is not specified, randomly initiate the train_state.
      (partitioned_train_state, inputs_partition_spec, partitioned_specs,
       decode_step_fn) = trainer_lib.partition_spmd_model_decode(
           task_p, init_key, inputs_shape)
    logging.info('partitioned_train_state: %s',
                 jax.tree_map(lambda x: x.shape, partitioned_train_state))
    # We do not fold in jax.process_index in contrast to the pmap version and
    # use a single global key instead to rely on pjit to split for different
    # replicas.
    logging.info('root prng_key: %s', prng_key)
    prng_key, decode_key = jax.random.split(prng_key)
    logging.info('eval prng_key: %s', decode_key)
    spmd_decode_step_fn = functools.partial(
        decode_step_fn,
        trainer_lib.train_state_for_eval_step(partitioned_train_state),
        decode_key)

    num_steps = [
        -1 if p.reset_for_eval else p.eval_loop_num_batches for p in input_p
    ]
    inputs = [p.Instantiate() for p in input_p]
    decodes = [list() for _ in input_p]
    process_id = jax.process_index()

    for split, num_split_steps in enumerate(num_steps):
      logging.info('Start decoding on input %s', input_p[split].name)
      step_num = 0
      while num_split_steps < 0 or step_num < num_split_steps:
        step_num += 1
        try:
          batch = inputs[split].get_next()
        except (tf.errors.OutOfRangeError, StopIteration):
          break
        if jax.config.jax_parallel_functions_output_gda:
          batch = py_utils.make_array(
              batch, inputs_shape, global_mesh, inputs_partition_spec
          )
        _, out = spmd_decode_step_fn(batch)
        # Output is fully replicated now, so it's ok to unreplicate it by
        # retrieving from device 0 only.
        out = py_utils.maybe_unreplicate_gda(out)
        global_batch_size = next(iter(out.values())).shape[0]
        logging.info('Finished decoding input batch %d with %d examples',
                     step_num, global_batch_size)
        # Manually shard the output per each jax process.
        # We require that all fields in the output is batch major.
        if global_batch_size % jax.process_count() != 0:
          raise ValueError(f'Global batch size {global_batch_size} must divide '
                           f'jax process count {jax.process_count()}')
        for k, v in out.items():
          if v.shape[0] != global_batch_size:
            raise ValueError('We require that all fields in the decode output '
                             'to have batch size as the first dim, got shape='
                             f'{v.shape} with key={k}, expect batch size = '
                             f'{global_batch_size}')
        per_process_batch_size = global_batch_size // jax.process_count()

        def shard(x, per_process_batch_size=per_process_batch_size):
          return x[(process_id *
                    per_process_batch_size):((process_id + 1) *
                                             per_process_batch_size)]

        out = jax.tree_map(shard, out)
        _, processed = jax_task.model.process_decode_out(inputs[split], out)
        decodes[split].extend(processed)
        logging.info('Finished processing decoded input batch %d', step_num)

  basedir = os.path.join(job_log_dir, 'decoder_out')
  dirnames = _get_dir_names(input_p)
  filename = _get_filename(
      py_utils.maybe_unreplicate_gda(partitioned_train_state.step))
  for s in dirnames:
    dir_path = os.path.join(basedir, s)
    if not tf.io.gfile.exists(dir_path):
      tf.io.gfile.makedirs(dir_path)
  filenames = [os.path.join(basedir, s, filename) for s in dirnames]
  for split, output_file in enumerate(filenames):
    logging.info('Writing decoder output to %s with %d entries', output_file,
                 len(decodes[split]))
    io_utils.WriteKeyValuePairs(output_file, decodes[split])
