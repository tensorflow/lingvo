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
"""Evaluation loop for lingvo Jax model."""

import contextlib
import functools
import os
import time
from typing import Optional, Sequence

from absl import logging
import jax
from jax.experimental import maps
from lingvo.jax import base_model_params
from lingvo.jax import model_utils
from lingvo.jax import partitioning
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax import summary_utils
from lingvo.jax import train_states
from lingvo.jax import trainer_lib
import tensorflow.compat.v2 as tf

from lingvo.jax import checkpoints

BaseModelParamsT = base_model_params.BaseModelParamsT
InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
TrainState = train_states.TrainState
SummaryWriter = tf.summary.SummaryWriter


def evaluate(
    model_name: str,
    job_log_dir: Optional[str],
    multi_host_checkpointing: Optional[bool],
    checkpoint_type: checkpoints.CheckpointType,
) -> None:
  """Runs the evaluation loop on the entire eval data set.

  Args:
    model_name: The name of the model from the registry to evaluate.
    job_log_dir: The directory for the job logs.
    multi_host_checkpointing: Whether to use multi-host checkpointing.
    checkpoint_type: Type of model checkpointing method to use.
  """
  model_config = model_utils.get_model(model_name)()
  model_p = model_config.task()
  eval_input_p = [v for v in model_config.datasets() if not v.is_training]
  for inp in eval_input_p:
    inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()
  if model_p.device_mesh is not None:
    evaluate_spmd_model(model_p, eval_input_p, job_log_dir,
                        multi_host_checkpointing, checkpoint_type)
  else:
    evaluate_pmap_model(model_p, eval_input_p, job_log_dir, checkpoint_type)


def evaluate_pmap_model(
    model_p: InstantiableParams,
    eval_input_p: Sequence[InstantiableParams],
    job_log_dir: Optional[str],
    checkpoint_type: checkpoints.CheckpointType,
) -> None:
  """Runs the evaluation loop on the entire test dataset for PMAP model.

  Args:
    model_p: Params for the data parallel model.
    eval_input_p: List of params for the eval data input pipelines.
    job_log_dir: Directory for the job logs.
    checkpoint_type: Type of model checkpointing method to use.
  """
  logging.info('Using pmap for data parallelism.')
  jax_model = model_p.Instantiate()
  eval_input_pipelines = [input_p.Instantiate() for input_p in eval_input_p]
  # TODO(shafey): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  checkpoint_dir = os.path.join(job_log_dir, 'checkpoints')
  model_states = trainer_lib.initialize_model_state(jax_model, init_key)
  model_states = checkpoints.restore_checkpoint(
      model_states, checkpoint_dir, checkpoint_type=checkpoint_type)
  replicated_model_states = trainer_lib.replicate_model_state(model_states)
  logging.info('replicated_model_states: %s',
               jax.tree_map(lambda x: x.shape, replicated_model_states))
  # From now on, different replicas should use different random seeds.
  # Here, each process will have its unique prng_key.
  # prng_key will be further split so that each core on a host will get
  # different prng_key.
  prng_key = jax.random.fold_in(prng_key, jax.process_index())
  logging.info('root prng_key: %s', prng_key)

  def eval_step(mdl_vars, prng_key, global_step, inputs):
    return trainer_lib.eval_step_single_learner(
        jax_model,
        mdl_vars,
        prng_key,
        global_step,
        inputs,
        data_parallel_axis_name='batch')

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

  num_steps = [-1 if p.reset_for_eval else 1 for p in eval_input_p]
  last_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
  with contextlib.ExitStack() as exit_stack:
    eval_summary_writers = [
        exit_stack.enter_context(summary_utils.get_summary_writer(d))
        for d in summary_eval_dirs
    ]

    while True:
      step_i = int(jax.device_get(replicated_model_states.step)[0])
      eval_step = functools.partial(p_eval_step,
                                    replicated_model_states.mdl_vars,
                                    eval_prng_seed,
                                    replicated_model_states.step)
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
        last_ckpt_step = int(last_checkpoint.split('_')[-1])
        exceeded_ckpt = last_ckpt_step + model_p.train.save_interval_steps
        if exceeded_ckpt >= model_p.train.num_train_steps:
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
      model_states = checkpoints.restore_checkpoint(
          model_states, checkpoint_dir, checkpoint_type=checkpoint_type)
      replicated_model_states = trainer_lib.replicate_model_state(model_states)
      last_checkpoint = new_checkpoint


def evaluate_spmd_model(
    model_p: InstantiableParams,
    eval_input_p: Sequence[InstantiableParams],
    job_log_dir: Optional[str],
    multi_host_checkpointing: bool,
    checkpoint_type: checkpoints.CheckpointType,
) -> None:
  """Runs the evaluation loop on the entire test dataset for SPMD model.

  Args:
    model_p: Params for the SPMD model.
    eval_input_p: List of Params for the eval data pipelines.
    job_log_dir: Directory for the job logs.
    multi_host_checkpointing: Whether to use multi-host checkpointing.
    checkpoint_type: Type of model checkpointing method to use.
  """
  logging.info('Using SPMD sharding for model parallelism.')
  eval_input_pipelines = [input_p.Instantiate() for input_p in eval_input_p]
  # TODO(bf-jax): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  checkpoint_dir = os.path.join(job_log_dir, 'checkpoints')
  if multi_host_checkpointing:
    checkpoint_task_dir = os.path.join(checkpoint_dir,
                                       f'{jax.process_index():03d}')
  else:
    checkpoint_task_dir = checkpoint_dir

  def get_shape_dtype(x):
    y = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return y

  model_inputs = eval_input_pipelines[0].get_next()
  inputs_shape = tf.nest.map_structure(get_shape_dtype, model_inputs)

  mesh_shape = model_p.device_mesh.shape
  device_mesh = partitioning.create_device_mesh(mesh_shape)
  logging.info('device_mesh: %s', device_mesh)
  with maps.mesh(device_mesh, model_p.mesh_axis_names):
    partitioned_train_state, partitioned_specs, _, eval_step, _, _, _ = (
        trainer_lib.partition_spmd_model(model_p, init_key, inputs_shape))
    partitioned_train_state = checkpoints.restore_checkpoint(
        partitioned_train_state,
        checkpoint_task_dir,
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
        eval_step_fn = functools.partial(eval_step,
                                         partitioned_train_state.mdl_vars,
                                         eval_key, partitioned_train_state.step)
        # Run the eval loop.
        model_utils.run_eval_loop_over_test_splits(
            num_steps,
            eval_step_fn,
            eval_summary_writers,
            step_i,
            eval_input_pipelines,
            reshard_inputs=False)
        # If the last check point evaluated matches max train steps, exit.
        if last_checkpoint is not None:
          last_ckpt_step = int(last_checkpoint.split('_')[-1])
          exceeded_ckpt = last_ckpt_step + model_p.train.save_interval_steps
          if exceeded_ckpt >= model_p.train.num_train_steps:
            break
        new_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
        while new_checkpoint == last_checkpoint:
          # Sleep for a minute.
          time.sleep(60)
          new_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
        # There must be a new checkpoint here.
        logging.info('Found new checkpoint: %s', new_checkpoint)
        partitioned_train_state = checkpoints.restore_checkpoint(
            partitioned_train_state,
            checkpoint_task_dir,
            checkpoint_type=checkpoint_type,
            state_specs=partitioned_specs)
        if multi_host_checkpointing:
          py_utils.sync_global_devices(
              f'checkpointer:restored:{checkpoint_dir}')
        last_checkpoint = new_checkpoint
