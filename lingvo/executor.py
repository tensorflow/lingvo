# Lint as: python2, python3
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
"""An experimental new unified TPU executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from lingvo import compat as tf
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import cluster_factory
from lingvo.core import ml_perf_log as mlp_log
from lingvo.core import multitask_model
from lingvo.core import py_utils
from lingvo.core import task_scheduler

from lingvo import base_runner
from tensorflow.python.tpu import device_assignment as device_assignment_lib  # pylint: disable=g-direct-tensorflow-import

tf.flags.DEFINE_bool(
    'cluster_placer_in_executor', False,
    'If True, cluster.GetPlacer() is used in Executor. ' +
    'When running on TPU model weights can be distributed ' +
    'across TPU hosts, for outrageously large models this ' +
    'enables sharded checkpointing and reduces host memory ' +
    'requirements, see _LeastLoadedPlacer in cluster.py.')

tf.flags.DEFINE_bool(
    'disable_meta_optimizer_in_executor', False,
    'Disabling the grappler meta_optimizer improves start-up time.')

FLAGS = tf.flags.FLAGS


def GetExecutorParams(model_name, cluster_params, model_registry):
  """Get the params needed to instantiate the Executor.

  Args:
    model_name: A model name regsitered in the ModelRegistry.
    cluster_params: A cluster hyperparams object.
    model_registry: A ModelRegistry object.

  Returns:
    A tuple (dict, Params):

    - ps_params_dict: High-level task name -> ProgramScheduleParams
    - train_cfg: A SingleTaskModelParams or MultiTaskModelParams.
  """

  ps_params_dict = {}
  with cluster_factory.Cluster(cluster_params):
    ps_cfg = model_registry.GetProgramSchedule(model_name)
    train_cfg = model_registry.GetParams(model_name, 'Train')
    train_cfg.cluster = cluster_params

    if issubclass(train_cfg.cls, base_model.MultiTaskModel):
      multi_task_train_cfg = train_cfg
      # Create SingleTaskModelParams from a MultiTaskModelParams.
      for k, _ in multi_task_train_cfg.task_params.IterParams():
        single_task_params = base_model.SingleTaskModel.Params()
        single_task_params.cluster = multi_task_train_cfg.cluster
        single_task_params.input = multi_task_train_cfg.input.Get(k)
        single_task_params.task = multi_task_train_cfg.task_params.Get(k)
        single_task_params.train = single_task_params.task.train
        if k not in ps_cfg.program_schedule_dict:
          tf.logging.fatal(
              'Could not find %s in ps_cfg.program_schedule_dict: %s', k,
              ps_cfg)
        program_schedule_params = ps_cfg.program_schedule_dict[k]

        program_schedule_params.task_dict = {'Train': single_task_params}

        for eval_dataset_name in program_schedule_params.dataset_names:
          multi_task_eval_cfg = model_registry.GetParams(
              model_name, eval_dataset_name)
          eval_task_params = base_model.SingleTaskModel.Params()
          eval_task_params.cluster = single_task_params.cluster
          eval_task_params.input = multi_task_eval_cfg.input.Get(k)
          eval_task_params.task = multi_task_eval_cfg.task_params.Get(k)
          program_schedule_params.task_dict[
              eval_dataset_name] = eval_task_params
        ps_params_dict[k] = program_schedule_params
    else:
      program_schedule_params = ps_cfg
      program_schedule_params.task_dict = {'Train': train_cfg}
      for eval_dataset_name in program_schedule_params.dataset_names:
        task_eval_params = model_registry.GetParams(model_name,
                                                    eval_dataset_name)
        task_eval_params.cluster = train_cfg.cluster
        program_schedule_params.task_dict[eval_dataset_name] = task_eval_params
      ps_params_dict[''] = program_schedule_params

  return ps_params_dict, train_cfg


class ExecutorTpu(base_runner.BaseRunner):
  """An experimental runner that does arbitrary multi-program execution on TPU.

  Overview of operation:

  - During construction, all programs construct their sub-graphs, in a sense
    creating a mega-graph.
  - A sequence of programs is then executed in-whole associated with that task.
    eg: [train x 1000 steps, checkpoint, eval 4 steps, decode 2 steps]
  - In this manner, programs and higher-level tasks cooperatively time-slice
    share the TPU.
  """

  def __init__(self, train_cfg, ps_params_dict, model_task_name, logdir,
               tf_master, **kwargs):
    """Construct an ExecutorTpu BaseRunner.

    Args:
      train_cfg: SingleTaskModelParams or MultiTaskModelParams
      ps_params_dict: A dict of top-level task name -> ProgramSchedule params,
          if train_cfg is a SingleTaskModelParams, we expect only one entry.
      model_task_name: An override for multi-task models, currently unused.
      logdir:  String path to the log directory to output to.
      tf_master: String path to the master job, e.g. 'local'.
      **kwargs: keyword args to pass through to BaseRunner.
    """
    super(ExecutorTpu, self).__init__(train_cfg, model_task_name, logdir,
                                      tf_master, **kwargs)

    self._cluster_def = self._cluster.worker_cluster_def

    # There is a single Executor task
    assert self._cluster.num_replicas == 1
    data_parallelism = self._cluster.num_splits_per_client

    assert data_parallelism
    num_devices_per_split = self._cluster.num_devices_per_split
    tf.logging.info('data_parallelism: %d, num_devices_per_split: %d',
                    data_parallelism, num_devices_per_split)

    self.task_scheduler = None
    self._checkpoint_dir = os.path.join(logdir, 'train')

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

    self._WriteToLog(train_cfg.ToText(), self._checkpoint_dir, 'params.txt')
    self._program_schedule_dict = {}
    self._programs = []

    for task_string, program_schedule_params in ps_params_dict.items():
      program_schedule_params.logdir = logdir
      program_schedule_params.num_splits_per_client = data_parallelism
      program_schedule_params.task_name = task_string
      ps = program_schedule_params.Instantiate()
      self._program_schedule_dict[task_string] = ps
      tf.logging.info('program_schedule_params: %s',
                      program_schedule_params.ToText())
      self._programs += ps.Programs()
      if program_schedule_params.ml_perf.benchmark_name is not None:
        self._ml_perf = program_schedule_params.ml_perf

    tf.logging.info('num_programs: %d', len(self._programs))

    if self._ml_perf is not None:
      self._ml_perf_log = True
      mlp_log.mlperf_print(key='benchmark', value=self._ml_perf.benchmark_name)
    else:
      self._ml_perf_log = False

    # BaseRunner legacy
    self.enqueue_ops = None

    @py_utils.RetryOnTransientTfError()
    def _WaitTillInit():
      """Wait until the model is ready."""
      try:
        with self._graph.as_default(), self._GetSession(
            cluster_def=self._cluster_def,
            disable_meta_optimizer=FLAGS.disable_meta_optimizer_in_executor
        ) as sess:
          topology = sess.run(
              tf.tpu.initialize_system(embedding_config=None, job=None))
          device_assignment = device_assignment_lib.device_assignment(
              topology,
              computation_shape=py_utils.ComputationShape(
                  num_devices_per_split),
              num_replicas=data_parallelism)
          py_utils.SetTpuDeviceAssignment(device_assignment)
          tf.logging.info('device_assignment.core_assignment: %s',
                          str(device_assignment.core_assignment))
          tf.logging.info(
              'device_assignment.topology.device_coordinates: %s',
              str(device_assignment.topology.device_coordinates))
      except py_utils.transient_tf_errors as e:
        tf.logging.info('TPU initialization failed: %s', e)
        raise

    if self._ml_perf_log:
      mlp_log.mlperf_print(key='init_start', value=None)
    _WaitTillInit()

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(
          self._cluster.job_spec.name if not FLAGS.cluster_placer_in_executor
          else self._cluster.GetPlacer()):
        with py_utils.VariableRenameScope(self._variable_renaming_rules):
          for program in self._programs:
            program.BuildTpuSubgraph()
        for program in self._programs:
          program.SetStatusMessageFn(self._SetStatusMessage)
          program.CreateCheckpointer()
        self._initialize_tables = tf.tables_initializer()
        self._initialize_local_vars = tf.local_variables_initializer()

        self.save_only_checkpointer = checkpointer.Checkpointer(
            self._checkpoint_dir,
            model=None,
            train_params=train_cfg.train,
            save_only=True)

  def Start(self):
    # Run training.
    self._RunLoop('executor_tpu', self._Loop)

  def _Loop(self):
    with tf.container(self._container_id), self._GetSession(
        cluster_def=self._cluster_def,
        disable_meta_optimizer=FLAGS.disable_meta_optimizer_in_executor
    ) as sess:
      # Initialize the variables first, if needed.
      for program in self._programs:
        program.RestoreIfNeeded(sess)
        program.Compile(sess)
      sess.run(self._initialize_tables)
      sess.run(self._initialize_local_vars)

      while True:
        global_step = sess.run(py_utils.GetGlobalStep())
        if self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          if not self._ml_perf_log:
            self.save_only_checkpointer.Save(sess, global_step)
          return

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

        done = program_schedule.Run(sess)
        if done:
          tf.logging.info('Program schedule told us to stop.')
          return

        # TODO(blee): More complex saving rules. Currently, we assume
        # we save after every task's program schedule execution.
        #
        # global_step local variable above is a result of sess.run, not a
        # tf variable, so when we do save_only_checkpointer.Save(...) here
        # py_utils.GetGlobalStep() is ahead of it by
        #   (train_executions_per_eval * train_steps_per_loop)
        # steps ahead already, due to program_schedule.Run(sess).
        #
        if not self._ml_perf_log:
          self.save_only_checkpointer.Save(sess, py_utils.GetGlobalStep())
