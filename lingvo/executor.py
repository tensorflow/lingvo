# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A multi-program TPU executor."""

import contextlib
import multiprocessing.dummy
import os
import time

from lingvo import compat as tf
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import cluster_factory
from lingvo.core import ml_perf_log as mlp_log
from lingvo.core import multitask_model
from lingvo.core import py_utils
from lingvo.core import task_scheduler
from lingvo.core import tpu_embedding_layers
import numpy as np

from lingvo import base_runner
from tensorflow.python.tpu import device_assignment as device_assignment_lib  # pylint: disable=g-direct-tensorflow-import

tf.flags.DEFINE_bool(
    'disable_meta_optimizer_in_executor', False,
    'Disabling the grappler meta_optimizer improves start-up time.')
FLAGS = tf.flags.FLAGS


def UnsetUnusedTrainParams(task_params):
  # Remove misleading train params
  task_params.train.tpu_steps_per_loop = None
  if 'task' in task_params:
    task_params.task.train.tpu_steps_per_loop = None
  return task_params


def GetExecutorParams(model_name, cluster_params, model_registry):
  """Get the params needed to instantiate the Executor.

  Args:
    model_name: A model name registered in the ModelRegistry.
    cluster_params: A cluster hyperparams object.
    model_registry: A ModelRegistry object.

  Returns:
    A tuple (dict, Params):

    - ps_params_dict: High-level task name -> ProgramScheduleParams
    - train_cfg: A SingleTaskModelParams or MultiTaskModelParams.

  Raises:
    ValueError if the model params is invalid.
  """

  ps_params_dict = {}
  with cluster_factory.Cluster(cluster_params):
    ps_cfg = model_registry.GetProgramSchedule(model_name)
    train_cfg = model_registry.GetParams(model_name, 'Train')
    train_cfg.cluster = cluster_params

    # Remove misleading train params
    train_cfg = UnsetUnusedTrainParams(train_cfg)

    if issubclass(train_cfg.cls, base_model.MultiTaskModel):
      multi_task_train_cfg = train_cfg

      if multi_task_train_cfg.train.ema_decay > 0:
        # Check that all subtasks use the same ema settings.
        for task_name, task_params in (
            multi_task_train_cfg.task_params.IterParams()):
          for field in ['ema_decay', 'ema_decay_moving_vars']:
            if (task_params.train.Get(field) !=
                multi_task_train_cfg.train.Get(field)):
              raise ValueError('Params did not match for field %s in task %s' %
                               (field, task_name))

      for k, _ in multi_task_train_cfg.task_params.IterParams():
        if multi_task_train_cfg.share_model_object:
          # Create MultiTaskSubModel params from a MultiTaskModelParams.
          train_task_params = base_model.MultiTaskSubModel.Params()
          train_task_params.task_name = k
          train_task_params.input = multi_task_train_cfg.input.Get(k).Copy()
        else:
          train_task_params = base_model.SingleTaskModel.Params()
          train_task_params.task = multi_task_train_cfg.task_params.Get(k)
          train_task_params.input = multi_task_train_cfg.input.Get(k)
        train_task_params.name = k + '_executor_train_task'
        train_task_params.cluster = multi_task_train_cfg.cluster
        train_task_params.train = multi_task_train_cfg.task_params.Get(k).train

        if k not in ps_cfg.program_schedule_dict:
          tf.logging.fatal(
              'Could not find %s in ps_cfg.program_schedule_dict: %s', k,
              ps_cfg)
        program_schedule_params = ps_cfg.program_schedule_dict[k]

        program_schedule_params.task_dict = {'Train': train_task_params}

        for eval_dataset_name in program_schedule_params.dataset_names:
          multi_task_eval_cfg = model_registry.GetParams(
              model_name, eval_dataset_name)
          if multi_task_train_cfg.share_model_object:
            eval_task_params = base_model.MultiTaskSubModel.Params()
            eval_task_params.task_name = k
            eval_task_params.input = multi_task_eval_cfg.input.Get(k).Copy()
          else:
            eval_task_params = base_model.SingleTaskModel.Params()
            eval_task_params.task = multi_task_eval_cfg.task_params.Get(k)
            eval_task_params.input = multi_task_eval_cfg.input.Get(k)
          eval_task_params.name = (
              k + '_' + eval_dataset_name + '_executor_eval_task')
          eval_task_params.cluster = multi_task_eval_cfg.cluster
          eval_task_params = UnsetUnusedTrainParams(eval_task_params)

          program_schedule_params.task_dict[
              eval_dataset_name] = eval_task_params
        ps_params_dict[k] = program_schedule_params
    else:
      program_schedule_params = ps_cfg
      program_schedule_params.task_dict = {'Train': train_cfg}
      for eval_dataset_name in program_schedule_params.dataset_names:
        task_eval_params = model_registry.GetParams(model_name,
                                                    eval_dataset_name)
        task_eval_params = UnsetUnusedTrainParams(task_eval_params)
        program_schedule_params.task_dict[eval_dataset_name] = task_eval_params

      ps_params_dict[''] = program_schedule_params

  return ps_params_dict, train_cfg


class ExecutorTpu(base_runner.BaseRunner):
  """An runner that does arbitrary multi-program execution on TPU.

  Overview of operation:

  - During construction, all programs construct their sub-graphs, in a sense
    creating a mega-graph.
  - A sequence of programs is then executed in-whole associated with that task.
    eg: [train x 1000 steps, checkpoint, eval 4 steps, decode 2 steps]
  - In this manner, programs and higher-level tasks cooperatively time-slice
    share the TPU.
  """

  def __init__(self, train_cfg, ps_params_dict, *args, **kwargs):
    """Construct an ExecutorTpu BaseRunner.

    Args:
      train_cfg: SingleTaskModelParams or MultiTaskModelParams
      ps_params_dict: A dict of top-level task name -> ProgramSchedule params,
        if train_cfg is a SingleTaskModelParams, we expect only one entry.
      *args: List args to pass through to BaseRunner.
      **kwargs: keyword args to pass through to BaseRunner.
    """
    super().__init__(train_cfg, *args, **kwargs)

    data_parallelism = self._cluster.num_splits_per_client
    assert data_parallelism
    num_devices_per_split = self._cluster.num_devices_per_split
    tf.logging.info('data_parallelism: %d, num_devices_per_split: %d',
                    data_parallelism, num_devices_per_split)

    self.task_scheduler = None
    self._checkpoint_dir = os.path.join(self._logdir, 'train')

    self._variable_renaming_rules = []

    self._ml_perf = None

    # If this is a multi-task model, grab the params for the TaskScheduler.
    if issubclass(train_cfg.cls, base_model.SingleTaskModel):
      tf.logging.info('single_task_model')
      assert len(ps_params_dict) == 1
      self._model_task_name = list(ps_params_dict.keys())[0]
      self._single_task_mode = True
    elif issubclass(train_cfg.cls, base_model.MultiTaskModel):
      tf.logging.info('multi_task_model')

      if issubclass(train_cfg.cls, multitask_model.RegExSharedVariableModel):
        self._variable_renaming_rules = train_cfg.variable_renaming_rules

      if train_cfg.task_schedule is None:
        task_schedule_params = task_scheduler.ConstantScheduler.Params()
        task_schedule_params.task_probs = sorted(
            list(train_cfg.task_probs.IterParams()))
      else:
        task_schedule_params = train_cfg.task_schedule
      self.task_scheduler = task_schedule_params.Instantiate()
      self._single_task_mode = False
    else:
      tf.logging.fatal(
          'Model %s is not a sub-class of SingleTaskModel or MultiTaskModel',
          train_cfg.cls)

    tf.logging.info('train_cfg.cls: %s', train_cfg.cls)

    self._WriteToLog(train_cfg.ToText(), self._checkpoint_dir,
                     'trainer_params.txt')
    if self._ml_perf is not None:
      self._ml_perf_log = True
      mlp_log.mlperf_print(key='benchmark', value=self._ml_perf.benchmark_name)
    else:
      self._ml_perf_log = False

    # BaseRunner legacy
    self.enqueue_ops = None

    train_cfg = self.params

    @py_utils.RetryOnTransientTfError()
    def _WaitTillInit(job=None):
      """Wait until the model is ready."""
      try:
        if py_utils.IsEagerMode():
          assert tf.executing_eagerly()
          tf.logging.info(f'FLAGS.tf_worker_address: {FLAGS.tf_worker_address}')

          # Connect to the TPU runtime.
          resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
              FLAGS.tf_worker_address, job_name=FLAGS.worker_job[len('/job:'):])
          tf.config.experimental_connect_to_cluster(resolver)
          topology = tf.tpu.experimental.initialize_tpu_system(resolver)
        else:
          # tpu.initialize_system() is called with None as embedding_config, as
          # embedding_config is not available yet. Later in _Loop, it is called
          # with the correct embedding_config. Since it cannot be called twice
          # in the same graph with different embedding_config, we use a
          # dummy_graph here.
          dummy_graph = tf.Graph()
          with dummy_graph.as_default():
            tpu_initialize_system_op = tf.tpu.initialize_system(
                embedding_config=None, job=job)

          with self._GetSession(graph=dummy_graph) as sess:
            topology = sess.run(tpu_initialize_system_op)

        if train_cfg.train.tpu_computation_shape is None:
          computation_shape = py_utils.ComputationShape(num_devices_per_split,
                                                        topology)
        else:
          computation_shape = train_cfg.train.tpu_computation_shape
          assert num_devices_per_split == np.prod(computation_shape)

        if train_cfg.train.tpu_device_order_mode is None:
          device_assignment = device_assignment_lib.device_assignment(
              topology,
              computation_shape=computation_shape,
              num_replicas=data_parallelism)
        else:
          device_assignment = device_assignment_lib.device_assignment(
              topology,
              computation_shape=computation_shape,
              num_replicas=data_parallelism,
              device_order_mode=train_cfg.train.tpu_device_order_mode)
        py_utils.SetTpuDeviceAssignment(device_assignment, job)
        tf.logging.info('device_assignment.core_assignment: %s',
                        str(device_assignment.core_assignment))
        tf.logging.info('device_assignment.topology.device_coordinates: %s',
                        str(device_assignment.topology.device_coordinates))
      except py_utils.transient_tf_errors as e:
        tf.logging.info('TPU initialization failed: %s', e)
        raise

    if self._ml_perf_log:
      mlp_log.mlperf_print(key='init_start', value=None)
    if len(self._cluster.all_worker_names) > 1:
      for worker in self._cluster.all_worker_names:
        _WaitTillInit(worker)
    else:
      _WaitTillInit(None)

    shared_model = self._MaybeConstructSharedModel(train_cfg)

    self._program_schedule_dict = {}
    self._programs = []

    for task_string, program_schedule_params in ps_params_dict.items():
      program_schedule_params.logdir = self._logdir
      program_schedule_params.num_splits_per_client = data_parallelism
      program_schedule_params.task_name = task_string
      # If the model was created above, we'll inject it here as a shared_model.
      ps = program_schedule_params.Instantiate(
          shared_model=shared_model,
          trial=self._trial,
          tf_master=self._tf_master)
      self._program_schedule_dict[task_string] = ps
      tf.logging.info('program_schedule_params: %s',
                      program_schedule_params.ToText())
      self._programs += ps.Programs()
      if program_schedule_params.ml_perf.benchmark_name is not None:
        self._ml_perf = program_schedule_params.ml_perf

    tf.logging.info('num_programs: %d', len(self._programs))

    # When running in a vizier trainer, the executor reports infeasiable runs
    # in case of errors. The programs report metrics and normal completions.
    for program in self._programs:
      if program._should_report_metrics:
        self._should_report_metrics = True

    with self._cluster, tf.container(
        self._container_id), contextlib.ExitStack() as stack:
      if not py_utils.IsEagerMode():
        stack.enter_context(self._graph.as_default())
        stack.enter_context(tf.device(self._cluster.GetPlacer()))
      with py_utils.VariableStore(), py_utils.VariableRenameScope(
          self._variable_renaming_rules):
        global_step = py_utils.GetOrCreateGlobalStepVar()
        if issubclass(train_cfg.cls, base_model.MultiTaskModel):
          if train_cfg.train.ema_decay > 0:
            # Create a global ExponentialMovingAverage object and share it
            # across all subtasks.
            ema = tf.train.ExponentialMovingAverage(
                decay=train_cfg.train.ema_decay, num_updates=global_step)
            py_utils.SetExponentialMovingAverage(ema)

        for program in self._programs:
          program.BuildTpuSubgraph()
          py_utils.ClearTpuSummaryTensors()

      if not py_utils.IsEagerMode():
        self._initialize_tables = tf.tables_initializer()
        self._initialize_local_vars = tf.local_variables_initializer()
        self._initialize_global_vars = tf.global_variables_initializer()

      for program in self._programs:
        program.SetStatusMessageFn(self._SetStatusMessage)
        if py_utils.IsEagerMode():
          # TODO(laigd): support checkpointing.
          pass
        else:
          program.CreateCheckpointer(init_op=self._initialize_global_vars)

      if py_utils.IsEagerMode():
        # TODO(laigd): support checkpointing.
        pass
      else:
        self.save_only_checkpointer = checkpointer.Checkpointer(
            self._checkpoint_dir,
            model=None,
            init_op=self._initialize_global_vars,
            train_params=train_cfg.train,
            save_only=True)

      tpu_embedding_collection = (
          tpu_embedding_layers.TpuEmbeddingCollection.Get())
      self._load_ops = tpu_embedding_collection.load_ops
      self._retrieve_ops = tpu_embedding_collection.retrieve_ops
      self._tpu_embedding = tpu_embedding_collection.tpu_embedding

  def _GetSession(self, **kwargs):
    if py_utils.IsEagerMode():
      raise ValueError('Eager mode does not support _GetSession.')
    return super()._GetSession(cluster_def=self._worker_cluster_def, **kwargs)

  def _MaybeConstructSharedModel(self, train_cfg):
    """Construct a single shared copy of the model if this is a MultiTaskModel.

    If the share_model_object parameter is set, for MultiTaskModels,
    we create a MultiTaskSubModel for each task, but construct the model only
    once.

    Args:
      train_cfg: The params for a SingleTaskModel or MultiTaskModel.

    Returns:
      A MultiTaskModel, if train_cfg is a MultiTaskModel params object.
    """
    if not issubclass(train_cfg.cls, base_model.MultiTaskModel):
      return None

    if not train_cfg.share_model_object:
      return None

    with self._cluster, tf.container(
        self._container_id), contextlib.ExitStack() as stack:
      if not py_utils.IsEagerMode():
        stack.enter_context(self._graph.as_default())
        stack.enter_context(tf.device(self._cluster.GetPlacer()))
      with py_utils.VariableStore(), py_utils.VariableRenameScope(
          self._variable_renaming_rules):
        _ = py_utils.GetOrCreateGlobalStepVar()
        shared_model = train_cfg.Instantiate()
        shared_model.InstantiateVariables()

    return shared_model

  def Start(self):
    # Run training.
    self._RunLoop('executor_tpu', self._Loop)

  def _Loop(self):
    with self._cluster, tf.container(
        self._container_id), contextlib.ExitStack() as stack:
      if py_utils.IsEagerMode():
        sess = None
      else:
        sess = self._GetSession(
            disable_meta_optimizer=FLAGS.disable_meta_optimizer_in_executor)
        stack.enter_context(sess)
        sess.reset(self._tf_master)
        config_proto = (
            self._tpu_embedding.config_proto
            if self._tpu_embedding is not None else None)
        for worker in self._cluster.all_worker_names:
          sess.run(
              tf.tpu.initialize_system(
                  embedding_config=config_proto, job=worker))

      # Initialize the variables first, if needed.
      compile_fns = []
      for program in self._programs:
        program.RestoreIfNeeded(sess)
        compile_fns += [program.Compile]

      # Run the compiles in parallel.
      threadpool = multiprocessing.dummy.Pool(len(compile_fns))
      futures = []
      tf.logging.info(f'Compiling {len(compile_fns)} programs in parallel.')
      for fn in compile_fns:
        futures += [threadpool.apply_async(fn, args=(sess,))]
      for future in futures:
        future.wait()

      if not py_utils.IsEagerMode():
        sess.run(self._initialize_tables)
        sess.run(self._initialize_local_vars)
        sess.run(self._load_ops)

      program_schedule = None
      # Threadpool to run code in programs async with TF Sessions (on TPUs).
      # This unblocks TPU from waiting for CPU processing on "main" thread, and
      # saves time for time-consuming CPU steps (e.g. PostProcessDecodeOut).
      program_threadpool = multiprocessing.dummy.Pool(1)
      start_time = time.time()
      while True:
        cycle_start_time = time.time()
        if py_utils.IsEagerMode():
          global_step = py_utils.GetGlobalStep().numpy()
        else:
          global_step = sess.run(py_utils.GetGlobalStep())

        async_checkpointing = False
        if py_utils.IsEagerMode():
          # TODO(laigd): save checkpoints.
          pass
        else:

          def RunSave(sess, global_step):
            # Run TPU embedding retrieve ops.
            # NOTE: this is expensive, so only run it when we're checkpointing.
            tf.logging.info('Retrieve params.')
            sess.run(self._retrieve_ops)
            tf.logging.info('Retrieve params done.')
            # Save program state first, so it's recoverable after we restore
            # from checkpoint.
            for program in self._programs:
              program.SaveProgramState(sess, global_step)
            # Save the checkpoints.
            self.save_only_checkpointer.Save(sess, global_step)

          if not self._ml_perf_log and self.save_only_checkpointer.ShouldSave(
              global_step):

            if self.save_only_checkpointer.async_checkpointing:
              tf.logging.info(
                  'Save checkpoint asynchronously AT YOUR OWN RISK.')
              threadpool = multiprocessing.dummy.Pool(1)
              saver_future = threadpool.apply_async(
                  RunSave, args=(sess, global_step))
              async_checkpointing = True
            else:
              RunSave(sess, global_step)

        # If a task is explicitly selected, only run the programs associated
        # with that task.
        if self._single_task_mode or self._model_task_name:
          tf.logging.info('Single task mode: %s', self._model_task_name)
          program_schedule = self._program_schedule_dict[self._model_task_name]
        else:
          # Otherwise, sample a task.
          model_task = self.task_scheduler.Sample(global_step)
          tf.logging.info('Sampled %s', model_task)
          program_schedule = self._program_schedule_dict[model_task]

        done, train_time_in_secs, eval_time_in_secs = program_schedule.Run(
            sess, program_threadpool)

        if async_checkpointing:
          saver_future.wait()

        executor_cycle_in_secs = time.time() - cycle_start_time
        self._ExportMetrics(
            executor_cycle_secs=executor_cycle_in_secs,
            executor_train_time_secs=train_time_in_secs,
            executor_eval_time_secs=eval_time_in_secs)

        def _ShutDown():
          program_threadpool.close()
          program_threadpool.join()
          tf.logging.info(
              'Program schedule told us to stop.\n'
              'Shutting down programs after running %f seconds.',
              time.time() - start_time)
          program_schedule.Shutdown()

        if done:
          tf.logging.info(
              'Program done after %f seconds. Waiting for threads to end.',
              time.time() - start_time)
          _ShutDown()
          return

        if py_utils.IsEagerMode():
          global_step = py_utils.GetGlobalStep().numpy()
        else:
          global_step = sess.run(py_utils.GetGlobalStep())
        if self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          if py_utils.IsEagerMode():
            # TODO(laigd): save checkpoints.
            pass
          else:
            if not self._ml_perf_log:
              RunSave(sess, global_step)
          tf.logging.info(
              'Program finished after %f seconds. Waiting for threads to end.',
              time.time() - start_time)
          _ShutDown()
          return
