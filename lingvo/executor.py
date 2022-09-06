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
from lingvo import pdb_wrapper
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
from google.protobuf import text_format
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.tpu import device_assignment as device_assignment_lib
# pylint: enable=g-direct-tensorflow-import

tf.flags.DEFINE_bool(
    'disable_meta_optimizer_in_executor', False,
    'Disabling the grappler meta_optimizer improves start-up time.')
tf.flags.DEFINE_bool(
    'use_tpu_mirrored_vars', False,
    'If set, use TPUStrategy / TPU mirrored variables to eliminate weight transfers. '
    'The trade-off here is that the Graph is larger. Disabling the meta optimizer '
    'might be needed for larger TPU slice topologies')

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

      for k, _ in multi_task_train_cfg.task_params.IterParams():
        if multi_task_train_cfg.share_model_object:
          # Create MultiTaskSubModel params from a MultiTaskModelParams.
          train_task_params = base_model.MultiTaskSubModel.Params()
          train_task_params.task_name = k
          train_task_params.input = multi_task_train_cfg.input.Get(k).Copy()
        else:
          task = multi_task_train_cfg.task_params.Get(k)
          train_task_params = base_model.SingleTaskModel.Params(task)
          train_task_params.input = multi_task_train_cfg.input.Get(k)
        train_task_params.name = k + '_executor_train_task'
        train_task_params.cluster = multi_task_train_cfg.cluster
        train_task_params.train = multi_task_train_cfg.task_params.Get(k).train

        if k not in ps_cfg.program_schedule_dict:
          tf.logging.fatal(
              'Could not find %s in ps_cfg.program_schedule_dict: %s', k,
              ps_cfg)
        # Add Copy in case a user is sharing the same ProgramSchedule params
        # instance across different tasks.
        program_schedule_params = ps_cfg.program_schedule_dict[k].Copy()

        program_schedule_params.task_dict = {'Train': train_task_params}

        for eval_dataset_name in program_schedule_params.dataset_names:
          multi_task_eval_cfg = model_registry.GetParams(
              model_name, eval_dataset_name)
          multi_task_eval_cfg.cluster = cluster_params
          if multi_task_train_cfg.share_model_object:
            eval_task_params = base_model.MultiTaskSubModel.Params()
            eval_task_params.task_name = k
            eval_task_params.input = multi_task_eval_cfg.input.Get(k).Copy()
          else:
            task = multi_task_eval_cfg.task_params.Get(k)
            eval_task_params = base_model.SingleTaskModel.Params(task)
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
        task_eval_params.cluster = cluster_params
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
    if py_utils.IsEagerMode():
      assert tf.executing_eagerly()
      tf.logging.info(f'FLAGS.tf_master: {FLAGS.tf_master}')

      # Connect to the TPU runtime.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tf_master, job_name=FLAGS.worker_job[len('/job:'):])
      tf.config.experimental_connect_to_cluster(resolver)

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
    self._WriteToLog(
        text_format.MessageToString(train_cfg.ToProto(), as_utf8=True),
        self._checkpoint_dir, 'trainer_params.pbtxt')
    if self._ml_perf is not None:
      self._ml_perf_log = True
      mlp_log.mlperf_print(key='benchmark', value=self._ml_perf.benchmark_name)
    else:
      self._ml_perf_log = False

    train_cfg = self.params

    @py_utils.RetryOnTransientTfError()
    def _WaitTillInit(job=None):
      """Wait until the model is ready."""
      try:
        if py_utils.IsEagerMode():
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
          self.device_assignment = device_assignment_lib.device_assignment(
              topology,
              computation_shape=computation_shape,
              num_replicas=data_parallelism)
        else:
          self.device_assignment = device_assignment_lib.device_assignment(
              topology,
              computation_shape=computation_shape,
              num_replicas=data_parallelism,
              device_order_mode=train_cfg.train.tpu_device_order_mode)
        py_utils.SetTpuDeviceAssignment(self.device_assignment, job)
        tf.logging.info('device_assignment.core_assignment: %s',
                        str(self.device_assignment.core_assignment))
        tf.logging.info('device_assignment.topology.device_coordinates: %s',
                        str(self.device_assignment.topology.device_coordinates))
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
    self._ckpt_programs = []

    self._checkpoint_to_load = None
    with self._cluster:
      # Create the ExponentialMovingAverage singleton shared by all programs, if
      # applicable.
      ema = py_utils.CreateEMAForModel(train_cfg, self._global_step_var)
      tf.logging.info('ps_params_dict=%s',
                      {k: v.ToText() for k, v in ps_params_dict.items()})
      for task_string, program_schedule_params in ps_params_dict.items():
        program_schedule_params.logdir = self._logdir
        program_schedule_params.num_splits_per_client = data_parallelism
        program_schedule_params.task_name = task_string
        # If the model was created above, we'll inject it here as a
        # shared_model.
        ps = program_schedule_params.Instantiate(
            shared_model=shared_model,
            trial=self._trial,
            ema=ema,
            tf_master=self._tf_master)
        self._program_schedule_dict[task_string] = ps
        tf.logging.info('program_schedule_params: %s',
                        program_schedule_params.ToText())
        self._programs += ps.Programs()
        if ps.train_program:
          self._ckpt_programs.append(ps.train_program)
        else:
          self._ckpt_programs += ps.Programs()
        if program_schedule_params.ml_perf.benchmark_name is not None:
          self._ml_perf = program_schedule_params.ml_perf
        if ('checkpoint_to_load' in program_schedule_params and
            program_schedule_params.checkpoint_to_load):
          if (self._checkpoint_to_load and
              (self._checkpoint_to_load !=
               program_schedule_params.checkpoint_to_load)):
            raise ValueError(f'Multiple values found for checkpoint_to_load: '
                             f'{self._checkpoint_to_load}, '
                             f'{program_schedule_params.checkpoint_to_load}.')
          self._checkpoint_to_load = program_schedule_params.checkpoint_to_load

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

        if FLAGS.use_tpu_mirrored_vars:
          resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
              FLAGS.tf_master, job_name=FLAGS.worker_job[len('/job:'):])
          self._tpu_strategy = tf.distribute.experimental.TPUStrategy(
              resolver, device_assignment=self.device_assignment)
          stack.enter_context(self._tpu_strategy.scope())
          stack.enter_context(
              tpu_strategy._TPUReplicaContext(self._tpu_strategy))
          if train_cfg.train.async_checkpointing:
            # TODO(b/228458924): reenable async checkpointing.
            tf.logging.warning(
                'Async checkpointing may not be compatible with TPU mirrored '
                'variables, so disabling it for now.')
            train_cfg.train.async_checkpointing = False
        else:
          stack.enter_context(tf.device(self._cluster.GetPlacer()))

      if FLAGS.pdb_on_exception:
        stack.enter_context(pdb_wrapper.catch_post_mortem())
      with py_utils.VariableStore(), py_utils.VariableRenameScope(
          self._variable_renaming_rules), py_utils.WarnOnGlobalStepAccess():
        # `BuildTpuSubgraph` has to be called before checkpoint restore, so that
        # the optimizer slot variables are guaranteed to be initialized before
        # they get loaded. Otherwise, the optimizers' slot variables will not
        # be properly loaded when V1 checkpoint is used.
        for program in self._programs:
          program.BuildTpuSubgraph()
          py_utils.ClearTpuSummaryTensors()

      checkpointer_models = [
          program.GetModel() for program in self._ckpt_programs
      ]
      if py_utils.IsEagerMode():
        if FLAGS.use_eager_v2_checkpoints:
          self._checkpointer = checkpointer.EagerCheckpointerV2(
              self._checkpoint_dir,
              models=checkpointer_models,
              train_params=train_cfg.train,
              save_only=False,
              experimental_enable_async_checkpoint=FLAGS
              .experimental_enable_async_checkpoint)
        else:
          self._checkpointer = checkpointer.EagerCheckpointerV1(
              self._checkpoint_dir,
              models=checkpointer_models,
              train_params=train_cfg.train,
              save_only=False)
      else:
        self._checkpointer = checkpointer.Checkpointer(
            self._checkpoint_dir,
            models=checkpointer_models,
            train_params=train_cfg.train,
            save_only=False)
        # Get the global_variables_initializer after creating the Checkpointer,
        # since it may create additional variables used by async checkpointing.
        self._initialize_tables = tf.tables_initializer()
        self._initialize_local_vars = tf.local_variables_initializer()
        self._initialize_global_vars = tf.global_variables_initializer()

      for program in self._programs:
        program.SetStatusMessageFn(self._SetStatusMessage)

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
        py_utils.GetOrCreateGlobalStepVar()
        shared_model = train_cfg.Instantiate()

    return shared_model

  def Start(self):
    super().Start()
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
      # Need to call create global step again because this is run in a thread.
      py_utils.GetOrCreateGlobalStepVar()

      if self._checkpoint_to_load:
        path = self._checkpointer.RestoreFromPath(
            sess, checkpoint_path=self._checkpoint_to_load)
      else:
        path = self._checkpointer.Restore(sess)

      # Run the compiles in parallel.
      compile_fns = []
      for program in self._programs:
        program.LoadProgramState(path, sess)
        compile_fns += [program.Compile]
      threadpool = multiprocessing.dummy.Pool(len(compile_fns))
      futures = []
      tf.logging.info(f'Compiling {len(compile_fns)} programs in parallel.')
      for fn in compile_fns:
        futures += [threadpool.apply_async(fn, args=(sess,))]
      for future in futures:
        future.get()

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

        def RunSave(sess, global_step):
          # Run TPU embedding retrieve ops.
          # NOTE: this is expensive, so only run it when we're checkpointing.
          if not py_utils.IsEagerMode():
            tf.logging.info('Retrieve params.')
            sess.run(self._retrieve_ops)
            tf.logging.info('Retrieve params done.')

          # Save program state first, so it's recoverable after we restore
          # from checkpoint.
          for program in self._programs:
            program.SaveProgramState(sess, global_step)
          # Save the checkpoints asynchronously.
          self._checkpointer.Save(sess, global_step, sync=False)

        checkpoint_write_secs = 0.0
        if not self._ml_perf_log and self._checkpointer.ShouldSave(global_step):
          checkpoint_write_start = time.time()
          RunSave(sess, global_step)
          checkpoint_write_secs = time.time() - checkpoint_write_start

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

        if hasattr(program_schedule, 'SetCheckpointer'):
          program_schedule.SetCheckpointer(self._checkpointer)

        done, train_time_in_secs, eval_time_in_secs = program_schedule.Run(
            sess, program_threadpool)

        executor_cycle_in_secs = time.time() - cycle_start_time
        if py_utils.IsEagerMode():
          global_step = py_utils.GetGlobalStep().numpy()
        else:
          global_step = sess.run(py_utils.GetGlobalStep())
        self._ExportMetrics(
            global_step=global_step,
            executor_cycle_secs=executor_cycle_in_secs,
            executor_train_time_secs=train_time_in_secs,
            executor_eval_time_secs=eval_time_in_secs,
            checkpoint_write_secs=checkpoint_write_secs,
        )

        def _ShutDown():
          # Wait for the save ops to finish before exit.
          self._checkpointer.Sync()
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

        if self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          if not self._ml_perf_log:
            RunSave(sess, global_step)
          tf.logging.info(
              'Program finished after %f seconds. Waiting for threads to end.',
              time.time() - start_time)
          _ShutDown()
          return
