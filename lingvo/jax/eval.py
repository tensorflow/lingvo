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

import functools
import math
import os
import time
from typing import List, Optional

from absl import logging
import jax
from jax.experimental import maps
from lingvo.jax import base_layer
from lingvo.jax import base_model_params
from lingvo.jax import checkpoints
from lingvo.jax import model_utils
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax import summary_utils
from lingvo.jax import train_states
from lingvo.jax import trainer_lib
import tensorflow.compat.v2 as tf

BaseModelParamsT = base_model_params.BaseModelParamsT
InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
TrainState = train_states.TrainState
SummaryWriter = tf.summary.SummaryWriter


def evaluate(model_name: str, job_log_dir: Optional[str],
             multi_host_checkpointing: Optional[bool]) -> None:
  """Runs the evaluation loop on the entire eval data set.

  Args:
    model_name: The name of the model from the registry to evaluate.
    job_log_dir: The directory for the job logs.
    multi_host_checkpointing: Whether to use multi-host checkpointing.
  """
  model_config = model_utils.get_model(model_name)()
  model_p = model_config.Task()
  eval_input_p = [v for v in model_config.Datasets() if not v.is_training]
  eval_input_p = [eval_param.input_gen_params for eval_param in eval_input_p]
  if model_p.device_mesh is not None:
    evaluate_spmd_model(model_p, eval_input_p, job_log_dir,
                        multi_host_checkpointing)
  else:
    evaluate_pmap_model(model_p, eval_input_p, job_log_dir)


def evaluate_pmap_model(model_p: InstantiableParams,
                        eval_input_p: List[InstantiableParams],
                        job_log_dir: Optional[str]) -> None:
  """Runs the evaluation loop on the entire test dataset for PMAP model.

  Args:
    model_p: Params for the data parallel model.
    eval_input_p: List of params for the eval data input pipeline.
    job_log_dir: Directory for the job logs.
  """
  logging.info('Using pmap for data parallelism.')
  jax_model = model_p.Instantiate()
  eval_input_pipelines = [input_p.Instantiate() for input_p in eval_input_p]
  get_model_inputs = functools.partial(model_utils.get_model_inputs,
                                       eval_input_pipelines)
  # TODO(shafey): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  checkpoint_dir = os.path.join(job_log_dir, 'checkpoints')
  model_states = trainer_lib.InitializesModelState(jax_model, init_key)
  model_states = checkpoints.RestoreCheckpoint(model_states, checkpoint_dir)
  replicated_model_states = trainer_lib.ReplicateModelState(model_states)
  logging.info('replicated_model_states: %s',
               jax.tree_map(lambda x: x.shape, replicated_model_states))
  # From now on, different replicas should use different random seeds.
  # Here, each process will have its unique prng_key.
  # prng_key will be further split so that each core on a host will get
  # different prng_key.
  prng_key = jax.random.fold_in(prng_key, jax.process_index())
  logging.info('root prng_key: %s', prng_key)

  def eval_step(mdl_vars, prng_key, global_step, inputs):
    return trainer_lib.EvalStepSingleLearner(
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
  summary_writer = summary_utils.GetSummaryWriter

  # Eval batch size per replica defaults to 1.
  batch_size = [1] * len(eval_input_p)
  for i, input_p in enumerate(eval_input_p):
    if 'batch_size' in input_p:
      batch_size[i] = input_p.batch_size
    if 'bucket_batch_limit' in eval_input_p:
      batch_size[i] = input_p.bucket_batch_limit[0]
  num_steps = [
      math.ceil(input_p.num_samples / batch_size[i])
      for i, input_p in enumerate(eval_input_p)
  ]
  last_checkpoint = checkpoints.LatestCheckpoint(checkpoint_dir)
  while True:
    step_i = int(jax.device_get(replicated_model_states.step)[0])
    eval_step = functools.partial(p_eval_step, replicated_model_states.mdl_vars,
                                  eval_prng_seed, replicated_model_states.step)
    # Run the eval loop.
    model_utils.run_eval_loop_over_test_splits(
        num_steps,
        eval_step,
        summary_writer,
        summary_eval_dirs,
        step_i,
        get_model_inputs,
        reshard_inputs=True)
    # If the last check point evaluated matches max train steps, exit.
    if last_checkpoint is not None:
      last_ckpt_step = int(last_checkpoint.split('_')[-1])
      exceeded_ckpt = last_ckpt_step + model_p.train.save_interval_steps
      if exceeded_ckpt >= model_p.train.num_train_steps:
        break
    new_checkpoint = checkpoints.LatestCheckpoint(checkpoint_dir)
    while new_checkpoint == last_checkpoint:
      # Sleep for a minute.
      time.sleep(60)
      new_checkpoint = checkpoints.LatestCheckpoint(checkpoint_dir)
    # There must be a new checkpoint here.
    logging.info('Found new checkpoint: %s', new_checkpoint)
    model_states = checkpoints.RestoreCheckpoint(model_states, checkpoint_dir)
    replicated_model_states = trainer_lib.ReplicateModelState(model_states)
    last_checkpoint = new_checkpoint


def evaluate_spmd_model(model_p: InstantiableParams,
                        eval_input_p: InstantiableParams,
                        job_log_dir: Optional[str],
                        multi_host_checkpointing: bool) -> None:
  """Runs the evaluation loop on the entire test dataset for SPMD model.

  Args:
    model_p: Params for the SPMD model.
    eval_input_p: Params for the eval data pipeline.
    job_log_dir: Directory for the job logs.
    multi_host_checkpointing: Whether to use multi-host checkpointing.
  """
  logging.info('Using SPMD sharding for model parallelism.')
  eval_input_pipelines = [input_p.Instantiate() for input_p in eval_input_p]
  get_model_inputs = functools.partial(model_utils.get_model_inputs,
                                       eval_input_pipelines)
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

  model_inputs = tf.nest.map_structure(lambda x: x.numpy(),
                                       get_model_inputs(split=0))
  inputs_shape = tf.nest.map_structure(get_shape_dtype, model_inputs)

  mesh_shape = model_p.device_mesh.shape
  device_mesh = base_layer.CreateDeviceMesh(mesh_shape)
  logging.info('device_mesh: %s', device_mesh)
  with maps.mesh(device_mesh, model_p.mesh_axis_names):
    partitioned_train_state, _, _, eval_step, _, _, _ = (
        trainer_lib.PartitionSpmdModel(model_p, init_key,
                                       inputs_shape))
    partitioned_train_state = checkpoints.RestoreCheckpoint(
        partitioned_train_state, checkpoint_task_dir)
    logging.info('partitioned_train_state: %s',
                 jax.tree_map(lambda x: x.shape, partitioned_train_state))
    if multi_host_checkpointing:
      py_utils.SyncGlobalDevices(f'checkpointer:restored:{checkpoint_dir}')

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
    summary_writer = summary_utils.GetSummaryWriter

    # Eval batch size per replica defaults to 1.
    batch_size = [1] * len(eval_input_p)
    for i, input_p in enumerate(eval_input_p):
      if 'batch_size' in input_p:
        batch_size[i] = input_p.batch_size
      if 'bucket_batch_limit' in eval_input_p:
        batch_size[i] = input_p.bucket_batch_limit
    num_steps = [
        math.ceil(input_p.num_samples / batch_size[i])
        for i, input_p in enumerate(eval_input_p)
    ]
    last_checkpoint = checkpoints.LatestCheckpoint(checkpoint_dir)
    while True:
      step_i = int(jax.device_get(partitioned_train_state.step))
      eval_step_fn = functools.partial(eval_step,
                                       partitioned_train_state.mdl_vars,
                                       eval_key, partitioned_train_state.step)
      # Run the eval loop.
      model_utils.run_eval_loop_over_test_splits(
          num_steps,
          eval_step_fn,
          summary_writer,
          summary_eval_dirs,
          step_i,
          get_model_inputs,
          reshard_inputs=False)
      # If the last check point evaluated matches max train steps, exit.
      if last_checkpoint is not None:
        last_ckpt_step = int(last_checkpoint.split('_')[-1])
        exceeded_ckpt = last_ckpt_step + model_p.train.save_interval_steps
        if exceeded_ckpt >= model_p.train.num_train_steps:
          break
      new_checkpoint = checkpoints.LatestCheckpoint(checkpoint_dir)
      while new_checkpoint == last_checkpoint:
        # Sleep for a minute.
        time.sleep(60)
        new_checkpoint = checkpoints.LatestCheckpoint(checkpoint_dir)
      # There must be a new checkpoint here.
      logging.info('Found new checkpoint: %s', new_checkpoint)
      partitioned_train_state = checkpoints.RestoreCheckpoint(
          partitioned_train_state, checkpoint_task_dir)
      if multi_host_checkpointing:
        py_utils.SyncGlobalDevices(f'checkpointer:restored:{checkpoint_dir}')
      last_checkpoint = new_checkpoint
