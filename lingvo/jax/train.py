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
"""Training loop for lingvo Jax model."""

import functools
import os
import time
from typing import List, Optional

from absl import logging
import jax
from jax.experimental import maps
from lingvo.jax import base_layer
from lingvo.jax import checkpoints
from lingvo.jax import model_utils
from lingvo.jax import py_utils
from lingvo.jax import summary_utils
from lingvo.jax import trainer_lib
import tensorflow.compat.v2 as tf

InstantiableParams = py_utils.InstantiableParams


def train_and_evaluate(model_name: str, job_log_dir: Optional[str],
                       multi_host_checkpointing: Optional[bool],
                       restore_checkpoint_dir: Optional[str],
                       restore_checkpoint_step: Optional[int],
                       eval_on_test: Optional[bool]) -> None:
  """Runs the training and evaluation loop.

  Args:
    model_name: The name of the model from the registry to train.
    job_log_dir: The directory for the job logs.
    multi_host_checkpointing: Whether to use multi-host checkpointing.
    restore_checkpoint_dir: If set, the directory from which to restore
      checkpoint. If unset, use job_log_dir's `checkpoints` subdirectory
      instead.
    restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
      try to restore from the latest checkpoint if any.
    eval_on_test: Whether to eval on test as a part of the training loop.
  """
  model_config = model_utils.get_model(model_name)()

  if jax.process_index() == 0:
    # Write out the params file.
    params_fpath = os.path.join(job_log_dir, 'model_params.txt')
    if not tf.io.gfile.exists(job_log_dir):
      tf.io.gfile.makedirs(job_log_dir)
    with tf.io.gfile.GFile(params_fpath, 'w') as params_file:
      datasets = model_config.Datasets()
      for dataset in datasets:
        params_file.write(dataset.ToText())
        params_file.write('\n\n')
      params_file.write(model_config.Task().ToText())

  model_p = model_config.Task()
  train_input_p = [v for v in model_config.Datasets() if v.is_training]
  if len(train_input_p) != 1:
    raise ValueError(
        f'Expecting exactly one training split. Got `{len(train_input_p)}`.')
  train_input_p = train_input_p[0].input_gen_params
  eval_input_p = None
  if eval_on_test:
    eval_input_p = [
        v.input_gen_params for v in model_config.Datasets() if not v.is_training
    ]
  if 'bucket_batch_limit' in train_input_p:
    logging.info('train_input_p.bucket_batch_limit: %s',
                 train_input_p.bucket_batch_limit)
  if model_p.device_mesh is not None:
    train_and_evaluate_spmd_model(model_p, train_input_p, job_log_dir,
                                  multi_host_checkpointing,
                                  restore_checkpoint_dir,
                                  restore_checkpoint_step, eval_input_p)
  else:
    train_and_evaluate_pmap(model_p, train_input_p, job_log_dir,
                            restore_checkpoint_dir, restore_checkpoint_step,
                            eval_input_p)


def train_and_evaluate_pmap(
    model_p: InstantiableParams, train_input_p: InstantiableParams,
    job_log_dir: Optional[str], restore_checkpoint_dir: Optional[str],
    restore_checkpoint_step: Optional[int],
    eval_input_p: Optional[List[InstantiableParams]]) -> None:
  """Runs the training and evaluation loop.

  Args:
    model_p: Params for the data parallel model.
    train_input_p: Params for the train data input pipeline.
    job_log_dir: Directory for the job logs.
    restore_checkpoint_dir: If set, the directory from which to restore
      checkpoint. If unset, use job_log_dir's `checkpoints` subdirectory
      instead.
    restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
      try to restore from the latest checkpoint if any.
    eval_input_p: Optional list of params for the eval input pipeline.
  """
  logging.info('Using pmap for data parallelism.')
  jax_model = model_p.Instantiate()

  with py_utils.InfeedContextScope(
      infeed_host_index=jax.process_index(),
      num_infeed_hosts=jax.process_count()):
    train_input_pipeline = train_input_p.Instantiate()
    get_model_inputs = functools.partial(model_utils.get_model_inputs,
                                         train_input_pipeline)
    if eval_input_p is not None:
      eval_input_pipelines = [input_p.Instantiate() for input_p in eval_input_p]
      get_eval_model_inputs = functools.partial(model_utils.get_model_inputs,
                                                eval_input_pipelines)

  # TODO(shafey): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  checkpoint_dir = os.path.join(job_log_dir, 'checkpoints')
  restore_checkpoint_dir = restore_checkpoint_dir or checkpoint_dir
  model_states = trainer_lib.InitializesModelState(jax_model, init_key)
  model_states = checkpoints.RestoreCheckpoint(
      model_states, restore_checkpoint_dir, step=restore_checkpoint_step)
  total_num_params = jax_model.total_num_vars
  replicated_model_states = trainer_lib.ReplicateModelState(model_states)
  # Unreplicated model states are not needed anymore at that point.
  del model_states

  logging.info('replicated_model_states shapes: %s',
               jax.tree_map(lambda x: x.shape, replicated_model_states))
  # From now on, different replicas should use different random seeds.
  # Here, each process will have its unique prng_key.
  # prng_key will be further split so that each core on a host will get
  # different prng_key.
  prng_key = jax.random.fold_in(prng_key, jax.process_index())
  logging.info('root prng_key: %s', prng_key)

  fprop_dtype = model_p.fprop_dtype

  def train_step(states, prng_key, inputs):
    return trainer_lib.TrainStepSingleLearner(
        jax_model,
        states,
        prng_key,
        inputs,
        data_parallel_axis_name='batch',
        fprop_dtype=fprop_dtype)

  def eval_step(mdl_vars, prng_key, global_step, inputs):
    return trainer_lib.EvalStepSingleLearner(
        jax_model,
        mdl_vars,
        prng_key,
        global_step,
        inputs,
        data_parallel_axis_name='batch',
        fprop_dtype=fprop_dtype)

  num_devices = jax.local_device_count()
  prng_key, train_key, eval_key = jax.random.split(prng_key, 3)
  train_prng_seed = jax.random.split(train_key, num=num_devices)
  eval_prng_seed = jax.random.split(eval_key, num=num_devices)
  logging.info('train prng_seed: %s', train_prng_seed)
  logging.info('eval prng_seed: %s', eval_prng_seed)

  p_train_step = jax.pmap(train_step, donate_argnums=(0,), axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  train_p = model_p.train

  logging.info('Training loop starting...')
  summary_base_dir = os.path.join(job_log_dir, 'summaries')
  summary_train_dir = os.path.join(summary_base_dir, 'train')
  summary_eval_dir = os.path.join(summary_base_dir, 'eval_train')
  summary_writer = summary_utils.GetSummaryWriter
  if eval_input_p is not None:
    summary_eval_dirs = [
        os.path.join(summary_base_dir, f'eval_test_{split}')
        for split, _ in enumerate(eval_input_p)
    ]
    # Only run one eval step during training.
    # TODO(yonghui): Allow user to customize this.
    eval_num_steps = [-1 if p.resettable else 1 for p in eval_input_p]

  with summary_writer(
      summary_train_dir) as train_summary_writer, summary_writer(
          summary_eval_dir) as eval_summary_writer:

    summary_utils.WriteModelStructure(
        train_summary_writer, replicated_model_states, is_vars_replicated=True)
    summary_utils.WriteTotalNumParams(train_summary_writer, total_num_params)

    summary_last_time = time.time()
    summary_last_step = None

    step_i = int(jax.device_get(replicated_model_states.step)[0])
    while True:
      logging.debug('step=`%d`: Beginning', step_i)
      if step_i >= train_p.num_train_steps:
        logging.info(
            'Training loop completed (step (`%d`) greater than '
            'num_train_step (`%d`).', step_i, train_p.num_train_steps)
        break
      if summary_last_step is None:
        summary_last_step = step_i - 1

      if (jax.process_index() == 0 and
          step_i % train_p.save_interval_steps == 0):
        checkpoints.SaveCheckpoint(
            replicated_model_states,
            checkpoint_dir,
            max_checkpoints=train_p.save_max_to_keep)

      if step_i <= 5:
        logging.info('step=`%d`: Retrieving model inputs.', step_i)
      logging.debug('  Retrieving inputs.')
      model_inputs = tf.nest.map_structure(py_utils.Reshard, get_model_inputs())
      logging.debug('  Retrieved inputs.')
      logging.debug('  Performing train_step().')
      (replicated_model_states, loss, metrics, per_example_out,
       summary_tensors) = p_train_step(replicated_model_states, train_prng_seed,
                                       model_inputs)
      logging.debug('  Completed train_step().')

      logging.debug('  Writing summaries (attempt).')
      if summary_utils.WriteSummaryEveryNSteps(
          replicated_model_states,
          train_summary_writer,
          step_i,
          train_p.summary_interval_steps,
          loss,
          metrics,
          per_example_out,
          summary_tensors,
          train_p.norm_summary_interval_steps,
          summary_last_time,
          summary_last_step,
          unreplicate_mdl_vars=True,
          unreplicate_metrics=True):
        summary_last_time = time.time()
        summary_last_step = step_i
        # Synchronize step_i
        step_i = int(jax.device_get(replicated_model_states.step)[0])
      else:
        # Increment locally to avoid an explicit sync.
        step_i += 1
      logging.debug('  Wrote summaries (attempted).')

      # Run eval at regular step interval.
      if step_i % train_p.eval_interval_steps == 0:
        logging.debug('  Starting eval_step().')
        logging.debug('  Retrieving eval model_inputs.')
        eval_inputs = get_model_inputs()
        logging.debug('  Retrieved eval model_inputs.')
        logging.debug('  Performing eval_step() runs on training split.')
        eval_step_fn = functools.partial(p_eval_step,
                                         replicated_model_states.mdl_vars,
                                         eval_prng_seed,
                                         replicated_model_states.step)
        loss, mean_metrics, summary_tensors = model_utils.run_eval_one_step(
            eval_inputs, eval_step_fn, reshard_inputs=True)
        logging.debug('  Completed eval_step() runs on training split.')
        logging.info('step=`%d`', step_i)
        logging.info('  eval loss: %s', loss)
        logging.info('  mean_metrics: %s', mean_metrics)
        logging.info('  summary_tensors: %s', summary_tensors)
        if step_i % train_p.summary_interval_steps == 0:
          logging.debug('  Writing eval summaries.')
          summary_utils.WriteSummaryEntry(
              eval_summary_writer,
              step_i,
              loss,
              mean_metrics,
              summary_tensors,
              unreplicate_metrics=True)
          logging.debug('  Wrote eval summaries.')
        # Eval on the test sets.
        if eval_input_p is not None:
          logging.debug('  Performing eval_step() runs on test splits.')
          model_utils.run_eval_loop_over_test_splits(
              eval_num_steps,
              eval_step_fn,
              summary_writer,
              summary_eval_dirs,
              step_i,
              get_eval_model_inputs,
              reshard_inputs=True)
          for i, _ in enumerate(eval_input_pipelines):
            if eval_input_p[i].resettable:
              # We re-instantiate the input to reset it.
              with py_utils.InfeedContextScope(
                  infeed_host_index=jax.process_index(),
                  num_infeed_hosts=jax.process_count()):
                eval_input_pipelines[i] = eval_input_p[i].Instantiate()
          if any([p.resettable for p in eval_input_p]):
            get_eval_model_inputs = functools.partial(
                model_utils.get_model_inputs, eval_input_pipelines)
        logging.debug('  Completed eval_step() runs on test splits.')
      logging.debug('step=`%d`: End', step_i - 1)


def train_and_evaluate_spmd_model(
    model_p: InstantiableParams, train_input_p: InstantiableParams,
    job_log_dir: Optional[str], multi_host_checkpointing: bool,
    restore_checkpoint_dir: Optional[str],
    restore_checkpoint_step: Optional[int],
    eval_input_p: Optional[InstantiableParams]) -> None:
  """Runs the training and evaluation loop.

  Args:
    model_p: Params for the SPMD model.
    train_input_p: Params for the train data pipeline.
    job_log_dir: Directory for the job logs.
    multi_host_checkpointing: Whether to use multi-host checkpointing.
    restore_checkpoint_dir: If set, the directory from which to restore
      checkpoint. If unset, use job_log_dir's `checkpoints` subdirectory
      instead.
    restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
      try to restore from the latest checkpoint if any.
    eval_input_p: Optional list of params for the eval input pipeline.
  """
  logging.info('Using SPMD sharding for model parallelism.')
  with py_utils.InfeedContextScope(
      infeed_host_index=jax.process_index(),
      num_infeed_hosts=jax.process_count()):
    train_input_pipeline = train_input_p.Instantiate()
    get_model_inputs = functools.partial(model_utils.get_model_inputs,
                                         train_input_pipeline)
    if eval_input_p is not None:
      eval_input_pipelines = [input_p.Instantiate() for input_p in eval_input_p]
      get_eval_model_inputs = functools.partial(model_utils.get_model_inputs,
                                                eval_input_pipelines)

  # TODO(bf-jax): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  checkpoint_dir = os.path.join(job_log_dir, 'checkpoints')
  restore_checkpoint_dir = restore_checkpoint_dir or checkpoint_dir
  if multi_host_checkpointing:
    checkpoint_task_dir = os.path.join(checkpoint_dir,
                                       f'{jax.process_index():03d}')
    restore_checkpoint_task_dir = os.path.join(restore_checkpoint_dir,
                                               f'{jax.process_index():03d}')
  else:
    checkpoint_task_dir = checkpoint_dir
    restore_checkpoint_task_dir = restore_checkpoint_dir

  if jax.process_index() == 0:
    tf.io.gfile.makedirs(checkpoint_dir)
  if multi_host_checkpointing:
    # Block all hosts until directory is ready.
    py_utils.SyncGlobalDevices(f'checkpointer:makedirs:{checkpoint_dir}')

  logging.info('step=`0`: Retrieving model inputs.')
  model_inputs = tf.nest.map_structure(lambda x: x.numpy(), get_model_inputs())

  def get_shape_dtype(x):
    # We assume all the hosts infeed the same data.
    process_count = jax.process_count()
    assert len(x.shape) >= 1
    x_shape = (x.shape[0] * process_count,) + x.shape[1:]
    y = jax.ShapeDtypeStruct(x_shape, x.dtype)
    return y

  inputs_shape = tf.nest.map_structure(get_shape_dtype, model_inputs)

  mesh_shape = model_p.device_mesh.shape
  device_mesh = base_layer.CreateDeviceMesh(mesh_shape)
  logging.info('device_mesh: %s', device_mesh)
  with maps.mesh(device_mesh, model_p.mesh_axis_names):
    (partitioned_train_state, _, train_step, eval_step, _, _,
     total_num_params) = trainer_lib.PartitionSpmdModel(model_p, init_key,
                                                        inputs_shape)

    partitioned_train_state = checkpoints.RestoreCheckpoint(
        partitioned_train_state,
        restore_checkpoint_task_dir,
        step=restore_checkpoint_step)
    logging.info('partitioned_train_state shapes: %s',
                 jax.tree_map(lambda x: x.shape, partitioned_train_state))
    if multi_host_checkpointing:
      py_utils.SyncGlobalDevices(f'checkpointer:restored:{checkpoint_dir}')

    # We do not fold in jax.process_index in contrast to the pmap version and
    # use a single global key instead to rely on pjit to split for different
    # replicas.
    logging.info('root prng_key: %s', prng_key)
    prng_key, train_key, eval_key = jax.random.split(prng_key, 3)
    logging.info('train prng_key: %s', train_key)
    logging.info('eval prng_key: %s', eval_key)

    train_p = model_p.train

    logging.info('Training loop starting...')
    summary_base_dir = os.path.join(job_log_dir, 'summaries')
    summary_train_dir = os.path.join(summary_base_dir, 'train')
    summary_eval_dir = os.path.join(summary_base_dir, 'eval_train')
    summary_writer = summary_utils.GetSummaryWriter
    if eval_input_p is not None:
      summary_eval_dirs = [
          os.path.join(summary_base_dir, f'eval_test_{split}')
          for split, _ in enumerate(eval_input_p)
      ]
      # Eval batch size per replica defaults to 1 when not resettable,
      # otherwise we exhaust all eval data (num_steps=-1).
      # TODO(yonghui): Allow user to customize this.
      eval_num_steps = [-1 if p.resettable else 1 for p in eval_input_p]

    with summary_writer(
        summary_train_dir) as train_summary_writer, summary_writer(
            summary_eval_dir) as eval_summary_writer:

      # This only prints the view from the first host machine.
      summary_utils.WriteModelStructure(
          train_summary_writer,
          partitioned_train_state,
          is_vars_replicated=False)
      summary_utils.WriteTotalNumParams(train_summary_writer, total_num_params)

      summary_last_time = time.time()
      summary_last_step = None

      step_i = int(jax.device_get(partitioned_train_state.step))

      # Start the train loop. Make sure all at the same step.
      py_utils.SyncGlobalDevices(f'Start training loop from step: {step_i}')
      while True:
        logging.debug('step=`%d`: Beginning', step_i)
        if step_i >= train_p.num_train_steps:
          logging.info(
              'Training loop completed (step (`%d`) greater than '
              'num_train_step (`%d`).', step_i, train_p.num_train_steps)
          break

        if summary_last_step is None:
          summary_last_step = step_i - 1

        if step_i % train_p.save_interval_steps == 0:
          logging.info('Saving a ckpt at step: %d', step_i)
          if multi_host_checkpointing:
            py_utils.SyncGlobalDevices(
                f'checkpointer:saving:{checkpoint_dir}:step-{step_i}')
          if multi_host_checkpointing or jax.process_index() == 0:
            checkpoints.SaveCheckpoint(
                partitioned_train_state,
                checkpoint_task_dir,
                max_checkpoints=train_p.save_max_to_keep,
                unreplicate=False)
          if multi_host_checkpointing:
            py_utils.SyncGlobalDevices(
                f'checkpointer:saved:{checkpoint_dir}:step-{step_i}')

        logging.debug('  Performing train_step().')
        (partitioned_train_state, loss, metrics, per_example_out,
         summary_tensors) = train_step(partitioned_train_state, train_key,
                                       model_inputs)
        logging.debug('  Completed train_step().')

        logging.debug('  Writing summaries (attempt).')
        if summary_utils.WriteSummaryEveryNSteps(
            partitioned_train_state,
            train_summary_writer,
            step_i,
            train_p.summary_interval_steps,
            loss,
            metrics,
            per_example_out,
            summary_tensors,
            train_p.norm_summary_interval_steps,
            summary_last_time,
            summary_last_step,
            unreplicate_mdl_vars=False,
            unreplicate_metrics=False):
          summary_last_time = time.time()
          summary_last_step = step_i
          step_i = int(jax.device_get(partitioned_train_state.step))
        else:
          # Increment train step locally to avoid an explicit device sync.
          step_i += 1
        logging.debug('  Wrote summaries (attempted).')

        # Run eval at regular step interval.
        if step_i % train_p.eval_interval_steps == 0:
          logging.debug('  Starting eval_step().')
          logging.debug('  Retrieving eval model_inputs.')
          eval_inputs = get_model_inputs()
          logging.debug('  Retrieved eval model_inputs.')
          logging.debug('  Performing eval_step() runs on training split.')
          eval_step_fn = functools.partial(eval_step,
                                           partitioned_train_state.mdl_vars,
                                           eval_key,
                                           partitioned_train_state.step)
          loss, mean_metrics, summary_tensors = model_utils.run_eval_one_step(
              eval_inputs, eval_step_fn, reshard_inputs=False)
          logging.debug('  Completed eval_step() runs on training split.')

          logging.info('step=`%d`', step_i)
          logging.info('  eval loss: %s', loss)
          logging.info('  mean_metrics: %s', mean_metrics)
          logging.info('  summary_tensors: %s', summary_tensors)
          if step_i % train_p.summary_interval_steps == 0:
            logging.debug('  Writing eval summaries.')
            summary_utils.WriteSummaryEntry(
                eval_summary_writer,
                step_i,
                loss,
                mean_metrics,
                summary_tensors,
                unreplicate_metrics=False)
            logging.debug('  Wrote eval summaries.')
          # If we have eval test then also evaluate on test.
          if eval_input_p is not None:
            logging.debug('  Performing eval_step() runs on test splits.')
            model_utils.run_eval_loop_over_test_splits(
                eval_num_steps,
                eval_step_fn,
                summary_writer,
                summary_eval_dirs,
                step_i,
                get_eval_model_inputs,
                reshard_inputs=False)
            for i, _ in enumerate(eval_input_pipelines):
              if eval_input_p[i].resettable:
                # We re-instantiate the input to reset it.
                with py_utils.InfeedContextScope(
                    infeed_host_index=jax.process_index(),
                    num_infeed_hosts=jax.process_count()):
                  eval_input_pipelines[i] = eval_input_p[i].Instantiate()
            if any([p.resettable for p in eval_input_p]):
              get_eval_model_inputs = functools.partial(
                  model_utils.get_model_inputs, eval_input_pipelines)
            logging.debug('  Completed eval_step() runs on test splits.')

        # Get new model inputs
        if step_i <= 5:
          logging.info('step=`%d`: Retrieving model inputs.', step_i)
        logging.debug('  Retrieving inputs.')
        model_inputs = tf.nest.map_structure(lambda x: x.numpy(),
                                             get_model_inputs())
        logging.debug('  Retrieved inputs.')
        logging.debug('step=`%d`: End', step_i - 1)
