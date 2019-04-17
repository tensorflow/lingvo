# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=line-too-long
"""Trainer.

To run locally:

.. code-block:: bash

  $ bazel build -c opt //lingvo:trainer
  $ bazel-bin/lingvo/trainer --logtostderr \
      --model=image.mnist.LeNet5 --mode=sync --logdir=/tmp/lenet5 --run_locally=cpu

To use GPU, add `--config=cuda` to build command and set `--run_locally=gpu`.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import re
import threading
import time

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip
import tensorflow as tf

from lingvo import base_runner
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.core.protobuf import config_pb2
from lingvo import base_trial
from lingvo import model_registry
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import cluster_factory
from lingvo.core import inference_graph_exporter
from lingvo.core import metrics
from lingvo.core import py_utils

tf.flags.DEFINE_string(
    'model', '', 'Name of the model class to train.'
    'Must be a model defined in the model_registry.')
tf.flags.DEFINE_string(
    'model_task_name', '', 'For multitask models: '
    'select task to train/evaluate/decode. '
    'Empty means to sample a task (training only).')
tf.flags.DEFINE_string('logdir', '', 'Log directory.')
tf.flags.DEFINE_bool(
    'interactive', False,
    'If True, enter interactive IPython for the controller job.')

tf.flags.DEFINE_string(
    'run_locally', '',
    'Can be empty, cpu, or gpu. If not empty, ignores cluster configuration '
    'flags and runs controller and trainer in a single local process.')

tf.flags.DEFINE_string('tf_master', '', 'TF runtime.')
tf.flags.DEFINE_string(
    'cluster_spec', '', 'A tf.train.ClusterSpec to override the master. '
    'The dict is specified as: job=host1:port1,host2:port2,'
    'host3:port3@job2=host3:port4,...')

tf.flags.DEFINE_string(
    'mode', 'async', 'How this trainer binary is used. '
    'async: used in an async training setup; '
    'sync: used in a sync training setup; '
    'shell: an interactive shell for development; '
    'inspect_evaler: print evaler dataset names; '
    'inspect_decoder: print decoder dataset names; '
    'write_inference_graph: write inference graphs to logdir.')
tf.flags.DEFINE_string('job', '', 'trainer/controller/eval, etc.')
tf.flags.DEFINE_integer('task', 0, 'Task id within the job.')

tf.flags.DEFINE_string('controller_job', '/job:controller', 'Job name.')
tf.flags.DEFINE_integer('controller_gpus', 0, 'Number of controller GPUs.')

tf.flags.DEFINE_string('worker_job', '/job:trainer', 'Job name.')
tf.flags.DEFINE_integer('worker_replicas', 1, 'Number of replicas.')
tf.flags.DEFINE_integer('worker_gpus', 0, 'Number of gpus to use per replica.')
tf.flags.DEFINE_integer('worker_tpus', 0, 'Number of tpus to use per replica.')
tf.flags.DEFINE_integer('worker_num_tpu_hosts', 0, 'Number of tpu hosts.')
tf.flags.DEFINE_integer('worker_split_size', 1,
                        'Number of devices for one split.')

tf.flags.DEFINE_string('ps_job', '/job:ps', 'Job name')
tf.flags.DEFINE_integer('ps_replicas', 1, 'Number of replicas.')
tf.flags.DEFINE_integer('ps_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_string('input_job', '/job:input', 'Job name')
tf.flags.DEFINE_integer('input_replicas', 0, 'Number of replicas.')

tf.flags.DEFINE_string('evaler_job', '/job:evaler', 'Job name')
tf.flags.DEFINE_integer('evaler_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_integer('evaler_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_string('decoder_job', '/job:decoder', 'Job name')
tf.flags.DEFINE_integer('decoder_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_integer('decoder_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_bool(
    'evaler_in_same_address_as_controller', False,
    'Whether or not evaler is in the same address space as '
    ' controller. This flag is meant for unittest only.')

tf.flags.DEFINE_string(
    'vizier_reporting_job', 'evaler',
    'Job reponsible for reporting metrics. This specifies a '
    'job prefix, evaler will match all evaler jobs, while '
    'evaler_dev and decoder_dev will only match the corresponding '
    'jobs that are on the dev set.')

FLAGS = tf.flags.FLAGS


# useful for debugging.
def _StartShell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython  # pylint: disable=g-import-not-at-top

  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def _ModelAnalysis(model):
  """Returns a text showing variable sizes and their total size."""

  class Analyzer(object):

    def __init__(self):
      self._seen_var = {}
      self.total = 0

    def __call__(self, v):
      assert isinstance(v, tf.Variable)
      # pylint: disable=protected-access
      if not v.shape.is_fully_defined():
        # Only Cudnn RNN params lack static shapes.
        if hasattr(v, 'approx_size'):
          size = v.approx_size
        else:
          return '%-20s %10s %s' % (v.shape, 'n/a', v._shared_name)
      else:
        size = v.shape.num_elements()
      if v._shared_name not in self._seen_var:
        self._seen_var[v._shared_name] = size
        self.total += size
      return '%-20s %10d %s' % (v.shape, size, v._shared_name)

  analyzer = Analyzer()
  output = '\n'
  output += model.vars.Transform(analyzer).DebugString()
  output += '\n'
  output += '=' * 100
  output += '\ntotal #params: %10d\n' % (analyzer.total)
  return output, analyzer.total


class Controller(base_runner.BaseRunner):
  """Controller for a training cluster."""

  def __init__(self, *args, **kwargs):
    super(Controller, self).__init__(*args, **kwargs)
    assert not self._model_task_name, 'Controller needs all tasks!'
    self._save_path = os.path.join(self._train_dir, 'ckpt')
    tf.gfile.MakeDirs(self._train_dir)
    self._control_dir = os.path.join(self._logdir, 'control')
    tf.gfile.MakeDirs(self._control_dir)
    self._summary_writer = self._CreateSummaryWriter(self._control_dir)
    self._time_steps = []  # A short history of (timestamp, global_step)

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._model = self.params.cls(self.params)
        self._params = self._model.params
        self._model.ConstructFPropBPropGraph()
        self._saver = self._GetSaver()
        self._summary_op = tf.summary.merge_all()
        self._vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self._uninitialized = tf.report_uninitialized_variables(self._vars)
        self._initialize_all = tf.global_variables_initializer()
        self.initialize_tables = tf.tables_initializer()
        self._initialize_local_vars = tf.local_variables_initializer()
        self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
        self.close_queue_ops = tf.get_collection(py_utils.CLOSE_QUEUE_OPS)

    self._ExportMetrics(params=self.params)
    self._model_analysis, self._total_num_params = _ModelAnalysis(self._model)
    py_utils.LogMultiLines('MODEL ANALYSIS', self._model_analysis)
    self._WriteToLog(self._model_analysis, self._control_dir,
                     'model_analysis.txt')
    self._WriteToLog(self.params.ToText(), self._control_dir, 'params.txt')
    tf.train.write_graph(self._graph.as_graph_def(), self._control_dir,
                         'train.pbtxt')

  def Start(self):
    self._RunLoop('controller', self._Loop)

  def StartEnqueueOp(self, op):
    self._RunLoop(
        'controller/enqueue_op/%s' % op.name, self._LoopEnqueue, loop_args=[op])

  def _Loop(self):
    self._summary_writer.add_graph(self._graph)
    with tf.container(self._container_id), self._GetSession() as sess:
      gsteps = py_utils.GetGlobalStep()
      examples = self._model.total_examples

      if FLAGS.interactive:
        # Into interactive debugging mode.
        _StartShell(locals())
        return

      # This initializes local tables
      sess.run(self.initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)

      # TODO(zhifengc): Moves these options into params.
      tp = self.params.train
      save_interval_seconds = tp.save_interval_seconds
      summary_interval_steps = tp.summary_interval_steps

      next_checkpoint_seconds = 0
      next_summary_step = 1

      while True:
        now = time.time()
        next_iteration_seconds = now + 10  # 10 seconds

        # Init/restore variable if needed.
        self._RestoreIfNeeded(sess)

        global_step, total_examples = sess.run([gsteps, examples])
        step_rate, example_rate = self._RecordStepRate(global_step,
                                                       total_examples)
        if self._trial.ShouldStop() or self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          self._saver.save(sess, self._save_path, gsteps)
          # Close all the queues so the enqueue threads can also finish.
          for close_op in self.close_queue_ops:
            sess.run(close_op)
          sess.close()
          return

        # Checkpoint.
        if now >= next_checkpoint_seconds:
          tf.logging.info('Save checkpoint')
          path = self._saver.save(sess, self._save_path, gsteps)
          tf.logging.info('Save checkpoint done: %s', path)
          next_checkpoint_seconds = now + save_interval_seconds

        # Summary.
        if self._summary_op is not None and global_step >= next_summary_step:
          tf.logging.info('Write summary @%s', global_step)
          summary_str = sess.run(self._summary_op)
          if isinstance(summary_str, np.ndarray) and summary_str.size == 0:
            tf.logging.info('Skipping summary: %s', summary_str)
          else:
            self._summary_writer.add_summary(summary_str, global_step)
          self._SummarizeValue(global_step, 'total_num_params',
                               self._total_num_params)
          next_summary_step = global_step + summary_interval_steps
          tf.logging.info('Write summary done: step %d', global_step)
          self._SetStatusMessage(
              'step:%6d, steps/sec: %0.2f, examples/sec: %0.2f' %
              (global_step, step_rate, example_rate))
          self._ExportMetrics(
              global_step=global_step,
              step_rate=step_rate,
              example_rate=example_rate)

        now = time.time()
        if now < next_iteration_seconds:
          time.sleep(next_iteration_seconds - now)

  def _RestoreIfNeeded(self, sess):
    uninitialized_var_names = list(sess.run(self._uninitialized))
    if not uninitialized_var_names:
      return

    tf.logging.info('Uninitialized var list: %s ', uninitialized_var_names)
    path = tf.train.latest_checkpoint(self._train_dir)
    if path:
      tf.logging.info('Load from checkpoint %s.', path)
      self._saver.restore(sess, path)
      tf.logging.info('Load checkpoint done.')
      return

    if (not any(task.params.train.init_from_checkpoint_rules
                for task in self._model.tasks) and
        not self._params.train.init_from_checkpoint_rules):
      tf.logging.info('Initialize ALL variables: %s', uninitialized_var_names)
      sess.run([self._initialize_all])
      tf.logging.info('Initialize variables done.')
      return

    # There was a race in local run. Another thread will get unblocked once
    # _initialize_all is called. OverrideVarsFromCheckpoints
    # might not happen at the right time.
    for task in self._model.tasks:
      tp = task.params.train
      if tp.init_from_checkpoint_rules:
        tf.logging.info('OverrideVarsFromCheckpoints %s',
                        tp.init_from_checkpoint_rules)
        py_utils.OverrideVarsFromCheckpoints(sess, self._vars,
                                             tp.init_from_checkpoint_rules)

    if self._params.train.init_from_checkpoint_rules:
      tp = self._params.train
      tf.logging.info('OverrideVarsFromCheckpoints %s',
                      tp.init_from_checkpoint_rules)
      py_utils.OverrideVarsFromCheckpoints(sess, self._vars,
                                           tp.init_from_checkpoint_rules)

    uninitialized_var_names = list(sess.run(self._uninitialized))
    if not uninitialized_var_names:
      return

    # uninitialized_var_names is a list of strings without ":0" suffix.
    assert all(isinstance(s, str) for s in uninitialized_var_names)

    # Need to retrieve vars, removing ":0" suffix from names.
    uninitialized_vars = [
        v for v in self._vars if v.name[:-2] in uninitialized_var_names
    ]
    tf.logging.info('Initialize variables: %s',
                    [v.name for v in uninitialized_vars])
    sess.run(tf.variables_initializer(uninitialized_vars))

  def _SummarizeValue(self, steps, tag, value):
    self._summary_writer.add_summary(
        metrics.CreateScalarSummary(tag, value), steps)

  def _RecordStepRate(self, current_steps, total_examples):
    """Computes the overall step rate and adds a summary."""
    self._time_steps.append((time.time(), current_steps, total_examples))
    # Keeps a relative long history to compute a smooth steps/second.
    # Removes duplicate stats for step = 0 to get rid of the warm-up period.
    while (self._time_steps[-1][1] - self._time_steps[0][1] > 10000 or
           (len(self._time_steps) > 1 and self._time_steps[-1][1] == 0 and
            self._time_steps[0][1] == 0)):
      del self._time_steps[0]
    (t0, s0, e0), (t1, s1, e1) = self._time_steps[0], self._time_steps[-1]
    rate = 0.0
    example_rate = 0.0
    if t1 > t0 + 1:
      elapsed_secs = t1 - t0
      rate = (s1 - s0) / elapsed_secs
      example_rate = (e1 - e0) / elapsed_secs
    tf.logging.info('Steps/second: %f, Examples/second: %f', rate, example_rate)
    self._SummarizeValue(current_steps, 'global_step/sec', rate)
    self._SummarizeValue(current_steps, 'examples/sec', example_rate)
    return rate, example_rate


class Trainer(base_runner.BaseRunner):
  """Trainer on non-TPU."""

  def __init__(self, *args, **kwargs):
    super(Trainer, self).__init__(*args, **kwargs)
    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._model = self.params.cls(self.params)
        self._params = self._model.params
        self._model.ConstructFPropBPropGraph()
      self.initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      self.close_queue_ops = tf.get_collection(py_utils.CLOSE_QUEUE_OPS)
      tf.logging.info('Trainer number of enqueue ops: %d',
                      len(self.enqueue_ops))

    try:
      self._task_probs_summary_writers = []
      for task in self._model.task_schedule.tasks:
        path = os.path.join(os.path.join(self._train_dir, task))
        tf.gfile.MakeDirs(path)
        self._task_probs_summary_writers.append(self._CreateSummaryWriter(path))
    except AttributeError:
      tf.logging.info('AttributeError. Expected for single task models.')
      self._task_probs_summary_writers = []

    # Saves the graph def.
    if self.params.cluster.task > 0:
      self._summary_writer = None
    else:
      self._summary_writer = self._CreateSummaryWriter(self._train_dir)
      tf.train.write_graph(self._graph.as_graph_def(), self._train_dir,
                           'train.pbtxt')
    worker_id = self.params.cluster.task
    self._start_up_delay_steps = (((worker_id + 1) * worker_id / 2) *
                                  self.params.train.start_up_delay_steps)

  def _SummarizeValue(self, steps, tag, value, writer):
    if writer:
      writer.add_summary(metrics.CreateScalarSummary(tag, value), steps)

  def Start(self):
    self._RunLoop('trainer', self._Loop)

  def StartEnqueueOp(self, op):
    self._RunLoop(
        'trainer/enqueue_op/%s' % op.name, self._LoopEnqueue, loop_args=[op])

  def _LoopEnqueue(self, op):
    # Evaler/Controller jobs may find that the trial is infeasible and report
    # done earlier. This is an important check since the trainer may retry
    # indefinitely without it.
    if self._trial.ShouldStop():
      tf.logging.info('Training skipped (trial requested to stop).')
      return
    return super(Trainer, self)._LoopEnqueue(op)

  def _Loop(self):
    # Evaler/Controller jobs may find that the trial is infeasible and report
    # done earlier. This is an important check since the trainer may retry
    # indefinitely without it.
    if self._trial.ShouldStop():
      tf.logging.info('Training skipped (trial requested to stop).')
      return
    with tf.container(self._container_id), self._GetSession() as sess:
      # This initializes local tables
      sess.run(self.initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      global_step = None

      @py_utils.Retry(retry_value=(tf.errors.FailedPreconditionError,))
      def _WaitTillInit():
        """Wait until the model is ready."""
        try:
          global_step = sess.run(py_utils.GetGlobalStep())
        except tf.errors.FailedPreconditionError as e:
          tf.logging.info('Probably the expected race on global_step: %s', e)
          raise
        msg = 'step:%6d' % global_step
        self._SetStatusMessage(msg)
        if global_step < self._start_up_delay_steps:
          msg = 'global step (%d) has not reached start up delay steps (%d)' % (
              global_step, self._start_up_delay_steps)
          tf.logging.info('%s', msg)
          raise tf.errors.FailedPreconditionError(
              node_def=None, op=None, message=msg)
        return global_step

      global_step = _WaitTillInit()

      status_interval_steps = 100
      next_status_step = 1
      eval_metrics = None
      while True:
        if (self._trial.ShouldStopAndMaybeReport(global_step, eval_metrics) or
            self._ShouldStop(sess, global_step)):
          tf.logging.info('Training finished.')
          # Close all the queues so the enque threads can also finish.
          for close_op in self.close_queue_ops:
            sess.run(close_op)
          if self._early_stop:
            time.sleep(300)  # controller hangs if it doesn't finish first
          return

        # If a task is explicitly specified, only train that task.
        if self._model_task_name:
          model_task = self._model.GetTask(self._model_task_name)
        else:
          # Note: This is a slightly stale global_step value from the previous
          # sess.run() call.
          # For multi-task models, `self._model.task_schedule.cur_probs` will
          # be updated.
          model_task = self._model.SampleTask(global_step)
          if self._task_probs_summary_writers:
            for index, prob in enumerate(self._model.task_schedule.cur_probs):
              self._SummarizeValue(global_step, 'task_probability', prob,
                                   self._task_probs_summary_writers[index])
            try:
              for index, task in enumerate(self._model.tasks):
                self._SummarizeValue(global_step, 'task_weight',
                                     sess.run(task.vars.task_weight),
                                     self._task_probs_summary_writers[index])
            except AttributeError:
              pass

        _, global_step, eval_metrics, per_example_tensors = sess.run([
            model_task.train_op,
            py_utils.GetGlobalStep(),
            model_task.eval_metrics,
            model_task.per_example_tensors,
        ])
        msg = 'step:%6d' % (global_step)
        for key, (val, _) in sorted(six.iteritems(eval_metrics)):
          msg += ' %s:%.8g' % (key, val)
          self._SummarizeValue(global_step, key, val, self._summary_writer)
        model_task.ProcessFPropResults(sess, global_step, eval_metrics,
                                       per_example_tensors)
        if global_step >= next_status_step:
          self._SetStatusMessage(msg)
          next_status_step = global_step + status_interval_steps
        else:
          tf.logging.info(msg)
        self._model.ProcessFPropResults(sess, global_step, eval_metrics,
                                        per_example_tensors)


class TrainerTpu(base_runner.BaseRunner):
  """Trainer on TPU."""

  def __init__(self, *args, **kwargs):
    super(TrainerTpu, self).__init__(*args, **kwargs)

    # Multiple TPU trainer tasks not tested/implemented.
    assert self._cluster.num_replicas == 1
    data_parallelism = self._cluster.num_splits_per_client
    assert data_parallelism
    num_devices_per_split = self._cluster.num_devices_per_split
    tf.logging.info('data_parallelism: %d, num_devices_per_split: %d',
                    data_parallelism, num_devices_per_split)

    def ComputationShape(split_size):
      """Decides the computation shape based on the split_size."""
      computation_shape = None
      if split_size == 1:
        computation_shape = [1, 1, 1]
      elif split_size == 2:
        computation_shape = [1, 1, 2]
      elif split_size == 4:
        computation_shape = [1, 2, 2]
      elif split_size == 8:
        computation_shape = [2, 2, 2]
      elif split_size == 16:
        computation_shape = [4, 2, 2]
      else:
        assert False, ('Model parallelism with %d devices is currently not'
                       ' supported.' % split_size)
      assert computation_shape is not None
      return computation_shape

    self._steps_per_loop = min(self.params.train.tpu_steps_per_loop,
                               self.params.train.max_steps)

    self._initialized = threading.Event()

    tf.logging.info(
        'Creating TrainerTpu using data parallelism %s '
        'and %s steps_per_loop', data_parallelism, self._steps_per_loop)

    @py_utils.RetryOnTransientTfError()
    def _WaitTillInit():
      """Wait until the model is ready."""
      try:
        with self._GetSession() as sess:
          topology = sess.run(
              tf.contrib.tpu.initialize_system(embedding_config=None, job=None))
          device_assignment = tf.contrib.tpu.device_assignment(
              topology,
              computation_shape=ComputationShape(num_devices_per_split),
              num_replicas=data_parallelism)
          py_utils.SetTpuDeviceAssignment(device_assignment)
          tf.logging.info('device_assignment.core_assignment: %s',
                          str(device_assignment.core_assignment))
          tf.logging.info('device_assignment.topology.device_coordinates: %s',
                          str(device_assignment.topology.device_coordinates))
      except py_utils.transient_tf_errors as e:
        tf.logging.info('TPU initialization failed: %s', e)
        raise

    _WaitTillInit()

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.job_spec.name):
        self._eval_metrics = metrics.TpuEvalMetrics()

        def TpuTrainStep(*args):
          """Train a shard of a batch on a single TPU core.

          Args:
            *args: metrics values from previous steps.

          Returns:
            New summed metrics values and a train_op.
          """
          self._model = self.params.cls(self.params)
          self._model.ConstructFPropBPropGraph()
          per_step_eval_metrics = self._eval_metrics.SetMetrics(
              self._model.GetTask().eval_metrics, args)
          outfeed_op = self._OutfeedEnqueue(
              self._model.GetTask().per_example_tensors)
          summed_metrics = []
          assert len(per_step_eval_metrics) == len(args)
          with tf.control_dependencies([outfeed_op]):
            for x, y in zip(per_step_eval_metrics, args):
              summed_metrics.append(x + y)
          return summed_metrics + [self._model.GetTask().train_op]

        @tpu_function.on_device_training_loop
        def TpuTrain():
          loop_result = tf.contrib.tpu.repeat(
              self._steps_per_loop,
              TpuTrainStep,
              inputs=self._eval_metrics.initial_values,
              name='train_loop')
          # Final metrics are the avg across self._steps_per_loop steps.
          return self._eval_metrics.FinalizeMetrics(loop_result)

        batch_parallel_res = tf.contrib.tpu.batch_parallel(
            TpuTrain,
            num_shards=data_parallelism,
            device_assignment=py_utils.GetTpuDeviceAssignment())
        outfeed_dequeue_op = self._OutfeedDequeueLoop(
            self._model.GetTask().per_example_tensors, self._steps_per_loop,
            self._cluster.num_splits_per_client)
        # Get metric result from a single replica; they are all same here.
        self._tpu_train_ops = [[t[0] for t in batch_parallel_res],
                               outfeed_dequeue_op]

      self.initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      assert not tf.get_collection(py_utils.CLOSE_QUEUE_OPS)
      tf.logging.info('Trainer number of enqueue ops: %d',
                      len(self.enqueue_ops))

    self._summary_writer = self._CreateSummaryWriter(self._train_dir)

    # Saves the graph def.
    tf.train.write_graph(self._graph.as_graph_def(), self._train_dir,
                         'train.pbtxt')

  def _OutfeedEnqueue(self, per_example_tensors):
    if not per_example_tensors:
      return tf.no_op()
    per_example_tensors = py_utils.NestedMap(per_example_tensors)
    return tf.contrib.tpu.outfeed_enqueue_tuple(per_example_tensors.Flatten())

  def _OutfeedDequeueLoop(self, per_example_tensors, num_loops, num_devices):
    """Process all per-example tensor outfeed data for a TPU sess.run.

    Args:
      per_example_tensors: dict of key -> tensor as generated by TpuTrainStep.
      num_loops: number of times that TpuTrainStep will be executed by TpuTrain.
      num_devices: number of TPU cores assigned to this process.

    Returns:
      A dict of per-example tensors from the latest TpuTrainStep.
    """
    if not per_example_tensors:
      return tf.no_op()

    tensor_shapes = [
        py_utils.GetShape(per_example_tensors[key])
        for key in sorted(per_example_tensors)
    ]
    tensor_types = [
        tf.as_dtype(per_example_tensors[key].dtype)
        for key in sorted(per_example_tensors)
    ]

    def LoopBody(i, *input_arrays):
      """Process outfeed data for a single TpuTrainStep.

      Args:
        i: current loop index.
        *input_arrays: One tf.TensorArray per outfeed tensor.

      Returns:
        i+1 (new index) plus post-write tf.TensorArray handles.
      """
      # Outfeed ops execute on each JF node, so they must be located on the
      # nodes.
      outfeed_devices = []
      device_assignment = py_utils.GetTpuDeviceAssignment()
      assert device_assignment
      for replica in xrange(device_assignment.num_replicas):
        for core in xrange(device_assignment.num_cores_per_replica):
          with tf.device(device_assignment.host_device(replica, core)):
            outfeed_devices.append(
                tf.contrib.tpu.outfeed_dequeue_tuple(
                    tensor_types,
                    tensor_shapes,
                    device_ordinal=device_assignment.tpu_ordinal(replica,
                                                                 core)))
      offset = i * num_devices
      output_arrays = list(input_arrays)
      # Each output_array holds a different per-example tensor. We get results
      # for each tensor from each TPU for each TpuTrainStep call.
      for j in range(len(output_arrays)):
        for k in range(len(outfeed_devices)):
          output_arrays[j] = output_arrays[j].write(offset + k,
                                                    outfeed_devices[k][j])

      return tuple([i + 1] + output_arrays)

    def LoopCond(i, *output_arrays):
      del output_arrays
      return i < num_loops

    output_arrays = [
        tf.TensorArray(
            tensor_types[i],
            size=num_loops * num_devices,
            element_shape=tensor_shapes[i]) for i in range(len(tensor_shapes))
    ]
    # Loop once for each time that TpuTrainStep runs.
    output_arrays = tf.while_loop(
        LoopCond, LoopBody, [0] + output_arrays, parallel_iterations=1)[1:]
    concatenated_arrays = [array.concat() for array in output_arrays]
    return dict(zip(sorted(per_example_tensors), concatenated_arrays))

  def Start(self):
    # Run training.
    self._RunLoop('trainer', self._Loop)

  def StartEnqueueOp(self, op):
    self._RunLoop(
        'trainer/enqueue_op/%s' % op.name, self._LoopEnqueue, loop_args=[op])

  def _SummarizeValue(self, steps, tag, value):
    self._summary_writer.add_summary(
        metrics.CreateScalarSummary(tag, value), steps)

  def _LoopEnqueue(self, op):
    # Evaler/Controller jobs may find that the trial is infeasible and report
    # done earlier. This is an important check since the trainer may retry
    # indefinitely without it.
    if self._trial.ShouldStop():
      tf.logging.info('Training skipped (trial requested to stop).')
      return
    # Wait for _Loop to initialize variables first before attempting to infeed.
    self._initialized.wait()
    return super(TrainerTpu, self)._LoopEnqueue(op)

  def _Loop(self):
    # Evaler/Controller jobs may find that the trial is infeasible and report
    # done earlier. This is an important check since the trainer may retry
    # indefinitely without it.
    if self._trial.ShouldStop():
      tf.logging.info('Training skipped (trial requested to stop).')
      return
    with tf.container(self._container_id), self._GetSession() as sess:
      sess.run(self.initialize_tables)
      sess.run(self._initialize_local_vars)
      sess.run(
          tf.contrib.tpu.initialize_system(embedding_config=None, job=None))
      if FLAGS.run_locally == 'tpu':
        sess.run(tf.global_variables_initializer())
      gsteps = py_utils.GetGlobalStep()
      global_step = sess.run(gsteps)
      self._initialized.set()
      eval_metrics = None

      while True:
        if self._trial.ShouldStopAndMaybeReport(global_step, eval_metrics):
          # Early terminate gracefully by setting a new max step horizon: three
          # more TPU steps to ensure that the enqueue ops can gracefully
          # terminate as well.
          if self._max_steps is None:
            self._max_steps = global_step + 3 * self._steps_per_loop
            tf.logging.info('Early stopping at step: %d', self._max_steps)

        if self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          return

        values, outfeeds = sess.run(self._tpu_train_ops)
        eval_metrics = self._eval_metrics.PackMetricsValues(values)

        # Note: global_step is incremented by self._steps_per_loop by the
        # previous sess.run call.
        global_step = sess.run(gsteps)

        msg = 'step:%6d' % global_step
        for key, (val, _) in sorted(six.iteritems(eval_metrics)):
          msg += ' %s:%.8g' % (key, val)
          self._SummarizeValue(global_step, key, val)

        self._SetStatusMessage(msg)

        task = self._model.GetTask()
        if not task.per_example_tensors:
          outfeeds = {}
        task.ProcessFPropResults(sess, global_step, eval_metrics, outfeeds)
        self._model.ProcessFPropResults(sess, global_step, eval_metrics,
                                        outfeeds)


class Evaler(base_runner.BaseRunner):
  """Evaler."""

  def __init__(self, eval_type, *args, **kwargs):
    super(Evaler, self).__init__(*args, **kwargs)
    self._job_name = 'evaler_' + eval_type
    self._output_name = 'eval_' + eval_type
    self.params.is_eval = True
    self._eval_dir = os.path.join(self._logdir, self._output_name)
    if self._model_task_name:
      self._eval_dir += '_' + str(self._model_task_name)
    tf.gfile.MakeDirs(self._eval_dir)
    self._summary_writer = self._CreateSummaryWriter(self._eval_dir)
    self._should_report_metrics = self._job_name.startswith(
        FLAGS.vizier_reporting_job)

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._model = self.params.cls(self.params)
        self._params = self._model.params
        # Always create the same graph to make sure node names are always
        # exactly the same.
        self._model.ConstructFPropGraph()
        self._model_task = self._model.GetTask(self._model_task_name)
        self._saver = self._GetSaver()
      self.initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      # No queues are allowed for eval models.
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      assert not self.enqueue_ops

    # Saves the graph def.
    self._WriteToLog(self.params.ToText(), self._eval_dir, 'params.txt')
    if self.params.cluster.task == 0:
      tf.train.write_graph(self._graph.as_graph_def(), self._eval_dir,
                           '%s.pbtxt' % self._output_name)

  def Start(self):
    self._RunLoop(self._job_name, self._Loop)

  def _Loop(self):
    """The main loop."""
    with tf.container(self._container_id), self._GetSession() as sess:
      # This initializes local tables
      sess.run(self.initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      path = None
      while True:
        path = self._FindNewCheckpoint(path, sess)
        if not path or self._EvalOnce(path, sess):
          break

    self.EvalLatestCheckpoint(path)
    if self._should_report_metrics:
      self._trial.ReportDone()
    tf.logging.info('Evaluation finished.')

  def EvalLatestCheckpoint(self, last_path=None):
    """Runs eval once on the latest checkpoint."""
    with tf.container(self._container_id), self._GetSession() as sess:
      # This initializes local tables
      sess.run(self.initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      path = tf.train.latest_checkpoint(self._train_dir)
      if not path:
        tf.logging.info('No checkpoint available.')
        return
      elif path == last_path:
        tf.logging.info('Latest checkpoint was already evaluated.')
        return
      self._EvalOnce(path, sess)

  def _EvalOnce(self, path, sess):
    """Runs evaluation for a batch of samples.

    Args:
      path: checkpoint path.
      sess: the tf Session.

    Returns:
      should_stop.
    """
    if not FLAGS.evaler_in_same_address_as_controller:
      self._LoadCheckpointForEval(sess, path)

    global_step = sess.run(py_utils.GetGlobalStep())
    metrics_dict = {
        name: metrics.AverageMetric() for name in self._model_task.eval_metrics
    }
    num_samples_metric = metrics_dict['num_samples_in_batch']
    while (num_samples_metric.total_value <
           self._model_task.params.eval.samples_per_summary):
      # NOTE: We intentionally do not let FProp generate summaries by default,
      # because evaler calls FProp multiple times for each checkpoint. Multiple
      # summaries at the same step is often confusing. Instead, models should
      # update eval_metrics and generate aggregate summaries.
      ans = sess.run(self._model_task.eval_metrics)
      for name, (value, weight) in six.iteritems(ans):
        metrics_dict[name].Update(value, weight)
      tf.logging.info('Total examples done: %d/%d',
                      num_samples_metric.total_value,
                      self._model_task.params.eval.samples_per_summary)

    # Replace average values with total values for certain metrics.
    if 'num_predictions' in metrics_dict:
      metrics_dict['num_predictions'].total_weight = 1.0
    if 'num_words' in metrics_dict:
      metrics_dict['num_words'].total_weight = 1.0

    # When we have evaluated so many samples, generate a summary.
    self._WriteSummaries(
        self._summary_writer,
        os.path.basename(self._eval_dir),
        global_step, {k: v.Summary(k) for k, v in six.iteritems(metrics_dict)},
        text_filename=os.path.join(self._eval_dir,
                                   'score-{:08d}.txt'.format(global_step)))

    should_stop = global_step >= self.params.train.max_steps
    if self._should_report_metrics:
      trial_should_stop = self._trial.ReportEvalMeasure(global_step,
                                                        metrics_dict, path)
      should_stop = should_stop or trial_should_stop
    return should_stop


def GetDecoderDir(logdir, decoder_type, model_task_name):
  if model_task_name:
    decoder_dir = '%s_%s' % (decoder_type, model_task_name)
  else:
    decoder_dir = decoder_type
  return os.path.join(logdir, decoder_dir)


def _GetCheckpointIdForDecodeOut(checkpoint_path, global_step):
  """Retrieve the checkpoint id for the decoder out file.

  Finds the checkpoint id in the checkpoint file name and compares to global
  step. If they diverge, uses the retrieved id and prints a warning.

  Args:
   checkpoint_path: path to checkpoint file.
   global_step: int specifying the global step of the model.

  Returns:
   Checkpoint id as int.
  """
  ckpt_id_from_file = int(re.sub(r'.*ckpt-', '', checkpoint_path))
  tf.logging.info('Loaded checkpoint is at global step: %d', global_step)
  tf.logging.info('Checkpoint path: %s', checkpoint_path)
  tf.logging.info('Checkpoint id according to checkpoint path: %d',
                  ckpt_id_from_file)
  if global_step != ckpt_id_from_file:
    tf.logging.warning(
        'Checkpoint id %d != global step %d. '
        'Will use checkpoint id from checkpoint file for '
        'writing decoder output.', ckpt_id_from_file, global_step)
  return ckpt_id_from_file


class Decoder(base_runner.BaseRunner):
  """Decoder."""

  def __init__(self, decoder_type, *args, **kwargs):
    super(Decoder, self).__init__(*args, **kwargs)
    self._job_name = 'decoder_' + decoder_type
    self.params.is_eval = True
    self._decoder_dir = GetDecoderDir(self._logdir, self._job_name,
                                      self._model_task_name)
    tf.gfile.MakeDirs(self._decoder_dir)
    self._summary_writer = self._CreateSummaryWriter(self._decoder_dir)
    self._should_report_metrics = self._job_name.startswith(
        FLAGS.vizier_reporting_job)

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._model = self.params.cls(self.params)
        self._params = self._model.params
        self._model_task = self._model.GetTask(self._model_task_name)
        # Note, different graphs are being constructed for different model
        # tasks, which may result in different node names being chosen.
        # Obviously, variable names has to be stay the same between train and
        # decode.
        input_batch = (
            self._model_task.input_generator.GetPreprocessedInputBatch())
        self._dec_output = self._model_task.Decode(input_batch)
        self._saver = self._GetSaver()
        self._summary_op = tf.summary.merge_all()
      self.initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      # No queues are allowed for decoder models.
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      assert not self.enqueue_ops

    # Saves the graph def.
    self._WriteToLog(self.params.ToText(), self._decoder_dir, 'params.txt')
    if self.params.cluster.task == 0:
      tf.train.write_graph(self._graph.as_graph_def(), self._decoder_dir,
                           '%s.pbtxt' % self._job_name)

  def Start(self):
    self._RunLoop(self._job_name, self._Loop)

  def _Loop(self):
    with tf.container(
        self._container_id), self._GetSession(inline=False) as sess:
      # This initializes local tables
      sess.run(self.initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)

      path = None
      while True:
        path = self._FindNewCheckpoint(path, sess)
        if not path or self.DecodeCheckpoint(sess, path):
          break

    self.DecodeLatestCheckpoint(path)

    if self._should_report_metrics:
      self._trial.ReportDone()
    tf.logging.info('Decoding finished.')

  @classmethod
  def GetDecodeOutPath(cls, decoder_dir, checkpoint_id):
    """Gets the path to decode out file."""
    out_dir = cls._GetTtlDir(decoder_dir, duration='7d')
    return os.path.join(out_dir, 'decoder_out_%09d' % checkpoint_id)

  def DecodeCheckpoint(self, sess, checkpoint_path):
    """Decodes `samples_per_summary` examples using `checkpoint_path`."""
    p = self._model_task.params
    samples_per_summary = p.eval.decoder_samples_per_summary
    if not samples_per_summary:
      samples_per_summary = p.eval.samples_per_summary
    self._LoadCheckpointForEval(sess, checkpoint_path)

    global_step = sess.run(py_utils.GetGlobalStep())
    dec_metrics = self._model_task.CreateDecoderMetrics()
    buffered_decode_out = []
    num_examples_metric = dec_metrics['num_samples_in_batch']
    start_time = time.time()
    while num_examples_metric.total_value < samples_per_summary:
      tf.logging.info('Fetching dec_output.')
      fetch_start = time.time()
      run_options = config_pb2.RunOptions(
          report_tensor_allocations_upon_oom=False)
      if self._summary_op is None:
        # No summaries were collected.
        dec_out = sess.run(self._dec_output, options=run_options)
      else:
        dec_out, summary = sess.run([self._dec_output, self._summary_op],
                                    options=run_options)
        self._summary_writer.add_summary(summary, global_step)
      post_process_start = time.time()
      tf.logging.info(
          'Done fetching (%f seconds)' % (post_process_start - fetch_start))
      decode_out = self._model_task.PostProcessDecodeOut(dec_out, dec_metrics)
      if decode_out:
        buffered_decode_out.extend(decode_out)
      tf.logging.info(
          'Total examples done: %d/%d '
          '(%f seconds decode postprocess)', num_examples_metric.total_value,
          samples_per_summary,
          time.time() - post_process_start)

    summaries = {k: v.Summary(k) for k, v in six.iteritems(dec_metrics)}
    elapsed_secs = time.time() - start_time
    example_rate = num_examples_metric.total_value / elapsed_secs
    summaries['examples/sec'] = metrics.CreateScalarSummary(
        'examples/sec', example_rate)
    self._WriteSummaries(
        self._summary_writer,
        os.path.basename(self._decoder_dir),
        global_step,
        summaries,
        text_filename=os.path.join(self._decoder_dir,
                                   'score-{:08d}.txt'.format(global_step)))
    self._ExportMetrics(
        decode_checkpoint=global_step,
        dec_metrics=dec_metrics,
        example_rate=example_rate)
    if buffered_decode_out:
      # global_step and the checkpoint id from the checkpoint file might be
      # different. For consistency of checkpoint filename and decoder_out
      # file, use the checkpoint id as derived from the checkpoint filename.
      checkpoint_id = _GetCheckpointIdForDecodeOut(checkpoint_path, global_step)
      decode_out_path = self.GetDecodeOutPath(self._decoder_dir, checkpoint_id)
      self._WriteKeyValuePairs(decode_out_path, buffered_decode_out)

    should_stop = global_step >= self.params.train.max_steps
    if self._should_report_metrics:
      trial_should_stop = self._trial.ReportEvalMeasure(
          global_step, dec_metrics, checkpoint_path)
      should_stop = should_stop or trial_should_stop
    return should_stop

  def DecodeLatestCheckpoint(self, last_path=None):
    """Runs decoder on the latest checkpoint."""
    with tf.container(self._container_id), self._GetSession() as sess:
      # This initializes local tables
      sess.run(self.initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      path = tf.train.latest_checkpoint(self._train_dir)
      if not path:
        tf.logging.info('No checkpoint available.')
        return
      elif path == last_path:
        tf.logging.info('Latest checkpoint was already decoded.')
        return
      self.DecodeCheckpoint(sess, path)


class RunnerManager(object):
  """Helper class for managing runners."""

  # This is a hack so these classes can be overridded with internal
  # non-public implementations.
  inference_graph_exporter = inference_graph_exporter
  model_registry = model_registry
  Controller = Controller
  Trainer = Trainer
  TrainerTpu = TrainerTpu
  Evaler = Evaler
  Decoder = Decoder

  def __init__(self, model):
    self._model_name = model

  def MaybeLaunchTensorFlow(self):
    """Starts TF machinary in this process."""
    if FLAGS.run_locally:
      return

    tf.logging.info('Launching tensorflow.')

    target = FLAGS.tf_master
    if not target.startswith('localhost'):
      # E.g., trainer_client is configured w/ FLAGS.tf_master pointing to
      # another job. In that case, start a local server.
      job_specs = FLAGS.cluster_spec.split('@')
      cluster_spec_dict = {}
      for job_spec in job_specs:
        # ps_host=worker1:1231,worker2:1234
        job_machines = job_spec.split('=')
        if len(job_machines) != 2:
          raise ValueError('Invalid job specification: %s', job_spec)
        cluster_spec_dict[job_machines[0]] = job_machines[1].split(',')
      self._tf_server = tf.train.Server(
          tf.train.ClusterSpec(cluster_spec_dict),
          job_name=FLAGS.job,
          task_index=FLAGS.task)
      target = self._tf_server.target
    if not FLAGS.tf_master:
      FLAGS.tf_master = target
    with tf.Session(target).as_default():
      value = (tf.constant(1.) + tf.constant(1.)).eval()
    assert value == 2.0, 'Something is really wrong.'
    tf.logging.info('Launched tensorflow.')

  def GetParamsForDataset(self, job_name, dataset_name):
    """Returns params for job `job_name` on the dataset `dataset_name`."""
    # Get the current cluster and update its params from flags.
    cluster = cluster_factory.Current()
    self.UpdateClusterParamsFromFlags(cluster.params, job_name)
    with cluster_factory.Cluster(cluster.params):
      try:
        cfg = self.model_registry.GetParams(self._model_name, dataset_name)
      except AttributeError as e:
        dataset_name_retry = dataset_name.title()
        tf.logging.warning(
            'Exception configuring dataset %s, retrying as %s: %s',
            dataset_name, dataset_name_retry, e)
        cfg = self.model_registry.GetParams(self._model_name,
                                            dataset_name_retry)
        tf.logging.warning(
            'Succeeded after retrying as %s.' % dataset_name_retry)
    cfg.cluster = cluster.params
    return cfg

  def MaybeConfigRunDistributed(self):
    """If given a `FLAGS.cluster_spec`, update flags for running distributed."""
    if not FLAGS.cluster_spec:
      return
    job_specs = FLAGS.cluster_spec.split('@')
    cluster_spec_dict = {}
    for job_spec in job_specs:
      # ps_host=worker1:1231,worker2:1234
      job_machines = job_spec.split('=')
      if len(job_machines) != 2:
        raise ValueError('Invalid job specification: %s', job_spec)
      cluster_spec_dict[job_machines[0]] = job_machines[1].split(',')
    if FLAGS.job == 'trainer_client':
      FLAGS.tf_master = 'grpc://%s' % cluster_spec_dict['worker'][FLAGS.task]
    for job in cluster_spec_dict.keys():
      if job.startswith('decoder_'):
        assert len(job_specs) == 1, 'Decoder jobs must run on their own'
        assert ',' not in job_specs[0], 'Only single machine supported'
        FLAGS.decoder_job = '/job:%s' % job
        FLAGS.decoder_replicas = 1
      if job.startswith('evaler_'):
        assert len(job_specs) == 1, 'Evaler jobs must run on their own'
        assert ',' not in job_specs[0], 'Only single machine supported'
        FLAGS.evaler_job = '/job:%s' % job
        FLAGS.evaler_replicas = 1
      if FLAGS.mode == 'sync' and FLAGS.job in ('controller', 'trainer_client',
                                                'worker'):
        FLAGS.worker_job = '/job:worker'
        FLAGS.worker_replicas = len(cluster_spec_dict['worker'])
        FLAGS.ps_job = '/job:worker'
        FLAGS.ps_replicas = FLAGS.worker_replicas
      if FLAGS.mode == 'async' and FLAGS.job in ('controller', 'trainer', 'ps'):
        FLAGS.worker_job = '/job:trainer'
        FLAGS.worker_replicas = len(cluster_spec_dict['trainer'])
        FLAGS.ps_job = '/job:ps'
        FLAGS.ps_replicas = len(cluster_spec_dict['ps'])

  def UpdateClusterParamsFromFlags(self, cluster, job_name):
    """Update `cluster` with a training cluster configuration from flags."""
    cluster.mode = FLAGS.mode
    cluster.job = job_name
    cluster.task = FLAGS.task

    cluster.controller.name = FLAGS.controller_job
    cluster.controller.gpus_per_replica = FLAGS.controller_gpus

    cluster.worker.name = FLAGS.worker_job
    cluster.worker.replicas = FLAGS.worker_replicas
    cluster.worker.gpus_per_replica = FLAGS.worker_gpus
    cluster.worker.tpus_per_replica = FLAGS.worker_tpus
    cluster.worker.num_tpu_hosts = FLAGS.worker_num_tpu_hosts
    cluster.worker.devices_per_split = FLAGS.worker_split_size

    cluster.ps.name = FLAGS.ps_job
    cluster.ps.replicas = FLAGS.ps_replicas
    cluster.ps.gpus_per_replica = FLAGS.ps_gpus

    cluster.input.name = FLAGS.input_job
    cluster.input.replicas = FLAGS.input_replicas

    cluster.evaler.name = FLAGS.evaler_job
    cluster.evaler.replicas = FLAGS.evaler_replicas
    cluster.evaler.gpus_per_replica = FLAGS.evaler_gpus

    cluster.decoder.name = FLAGS.decoder_job
    cluster.decoder.replicas = FLAGS.decoder_replicas
    cluster.decoder.gpus_per_replica = FLAGS.decoder_gpus

  def _CreateRunner(self, job, model_task_name, logdir, tf_master, trial):
    """Create a runner."""
    evaler_job_name_prefix = 'evaler_'
    decoder_job_name_prefix = 'decoder_'

    tf.logging.info('Job %s start', job)
    common_args = (model_task_name, logdir, tf_master, trial)
    if job == 'controller':
      cfg = self.GetParamsForDataset('controller', 'Train')
      return self.Controller(cfg, *common_args)
    elif job == 'trainer':
      cfg = self.GetParamsForDataset('trainer', 'Train')
      return self.Trainer(cfg, *common_args)
    elif job == 'trainer_client':
      cfg = self.GetParamsForDataset('trainer_client', 'Train')
      if py_utils.use_tpu():
        return self.TrainerTpu(cfg, *common_args)
      else:
        return self.Trainer(cfg, *common_args)
    elif job.startswith(evaler_job_name_prefix):
      dataset_name = job[len(evaler_job_name_prefix):]
      cfg = self.GetParamsForDataset('evaler', dataset_name)
      return self.Evaler(dataset_name.lower(), cfg, *common_args)
    elif job.startswith(decoder_job_name_prefix):
      dataset_name = job[len(decoder_job_name_prefix):]
      cfg = self.GetParamsForDataset('decoder', dataset_name)
      return self.Decoder(dataset_name.lower(), cfg, *common_args)
    elif job in ('ps', 'worker', 'input'):
      self._tf_server.join()
    else:
      raise ValueError('job %s is not supported' % job)

  def CreateRunners(self, jobs, logdir, trial=base_trial.NoOpTrial()):
    """Creates a list of runners based on `FLAGS.mode`.

    Args:
      jobs: a list of runner jobs.
      logdir: the directory used for logging, usually on CNS.
      trial: optional `Trial` object, used for reporting measures and early
        stopping.

    Returns:
      A list of `.BaseRunner`, one per job in `jobs`.
    """

    runners = []
    for j in jobs:
      tf_master = FLAGS.tf_master
      # Ensure that decoder or evaler threads do not clobber variables being
      # updated by trainer by forcing them to use independent sessions.
      if ('trainer' in jobs and
          (j.startswith('decoder') or j.startswith('evaler'))):
        tf_master = ''

      runner = self._CreateRunner(j, FLAGS.model_task_name, logdir, tf_master,
                                  trial)
      runners.append(runner)
    return runners

  def StartRunners(self, runners):
    """Runs `runners` in parallel threads.

    Returns when all of them finish.

    Args:
      runners: a list of `.BaseRunner`.

    Returns:
      None.
    """
    threads = []
    tf.logging.info('Starting runners')
    for runner in runners:
      t = threading.Thread(target=runner.Start)
      t.daemon = True
      t.start()
      threads.append(t)
      tf.logging.info('Total num runner.enqueue_ops: %d',
                      len(runner.enqueue_ops))
      for enqueue_op in runner.enqueue_ops:

        def StartEnqueue(runner, op):
          tf.logging.info('Starting enqueue op %s', op.name)
          return lambda: runner.StartEnqueueOp(op)

        tq = threading.Thread(target=StartEnqueue(runner, enqueue_op))
        tq.start()
        threads.append(tq)
    tf.logging.info('Waiting for runners to finish...')
    for t in threads:
      while True:
        t.join(1)
        if not t.isAlive():
          break
    tf.logging.info('All runners done.')

  def RunTrial(self, job, logdir, trial):
    """A wrapper function for running a trial."""
    if job == 'all':
      # For async mode: Run controller, trainer, evaler jobs in one process,
      # multiple threads.
      self.StartRunners(
          self.CreateRunners(['controller', 'trainer'], logdir, trial))
      evaler = self._CreateRunner('evaler_dev', FLAGS.model_task_name, logdir,
                                  FLAGS.tf_master, trial)
      evaler.EvalLatestCheckpoint()
    elif job == 'all_sync':
      # For sync mode: Run controller, trainer_client, evaler jobs in one
      # process, multiple threads.
      self.StartRunners(
          self.CreateRunners(['controller', 'trainer_client'], logdir, trial))
      evaler = self._CreateRunner('evaler_dev', FLAGS.model_task_name, logdir,
                                  FLAGS.tf_master, trial)
      evaler.EvalLatestCheckpoint()
    else:
      # Run each job in separate process/task
      # TODO(rpang): add support for running evaler_test and decoder.
      self.StartRunners(self.CreateRunners([job], logdir, trial))

  def MaybeConfigRunLocally(self):
    """Update flags if configured to run locally."""
    if not FLAGS.run_locally:
      # Do nothing
      return

    FLAGS.tf_master = tf.train.Server.create_local_server().target

    if not FLAGS.mode:
      FLAGS.mode = 'sync'

    if not FLAGS.job:
      if FLAGS.run_locally == 'tpu':
        FLAGS.job = 'trainer_client'
      else:
        FLAGS.job = 'controller,trainer_client'

    FLAGS.task = 0

    FLAGS.controller_job = '/job:local'

    FLAGS.worker_job = '/job:local'
    FLAGS.worker_replicas = 1
    if FLAGS.run_locally == 'gpu':
      if not FLAGS.worker_gpus:
        FLAGS.worker_gpus = 1
    else:
      FLAGS.worker_gpus = 0
    if FLAGS.run_locally == 'tpu':
      FLAGS.xla_device = 'tpu'
      FLAGS.enable_asserts = False
    else:
      FLAGS.worker_tpus = 0

    if not FLAGS.worker_split_size:
      FLAGS.worker_split_size = 1

    FLAGS.ps_job = '/job:local'
    FLAGS.ps_replicas = 1
    FLAGS.ps_gpus = 0

    FLAGS.input_job = '/job:local'
    FLAGS.input_replicas = 0

    FLAGS.evaler_job = '/job:local'
    FLAGS.evaler_replicas = 1
    if FLAGS.run_locally == 'gpu':
      FLAGS.evaler_gpus = 1
    else:
      FLAGS.evaler_gpus = 0

    FLAGS.decoder_job = '/job:local'
    FLAGS.decoder_replicas = 1
    if FLAGS.run_locally == 'gpu':
      FLAGS.decoder_gpus = 1
    else:
      FLAGS.decoder_gpus = 0

  def InspectModel(self):
    """Prints out model analysis for the model."""
    p = self.GetParamsForDataset('controller', 'Train')
    p.cluster.mode = 'sync'
    c = cluster_factory.Cluster(p.cluster)
    with tf.Graph().as_default(), c, tf.device(c.GetPlacer()):
      analysis, _ = _ModelAnalysis(p.cls(p))
    print(analysis)

  def InspectDatasets(self):
    """Prints out datasets configured for the model."""
    cls = self.model_registry.GetClass(self._model_name)
    datasets = []
    for name, _ in inspect.getmembers(cls, inspect.ismethod):
      if name not in ['GetDatasetParams', 'Model', 'Task'
                     ] and not name.startswith('_'):
        datasets += [name]
    print(','.join([_.lower() for _ in datasets]))

  def InspectDecoder(self):
    """Prints out datasets configured for the decoder."""
    cls = self.model_registry.GetClass(self._model_name)

    has_decoder = False
    if issubclass(cls, base_model_params.SingleTaskModelParams):
      has_decoder = cls.Task(
      ).cls.CreateDecoderMetrics != base_model.BaseTask.CreateDecoderMetrics
    else:
      for _, task_param in cls.Model().task_params.IterParams():
        has_decoder |= (
            task_param.cls.CreateDecoderMetrics !=
            base_model.BaseTask.CreateDecoderMetrics)
    if has_decoder:
      # We assume that the proper decoder is implemented.
      self.InspectDatasets()
    else:
      print('')

  def WriteInferenceGraph(self):
    """Generates the inference graphs for a given model."""
    inference_graph_dir = os.path.join(FLAGS.logdir, 'inference_graphs')
    tf.gfile.MakeDirs(inference_graph_dir)
    tf.logging.info('Writing inference graphs to dir: %s', inference_graph_dir)

    cfg = self.model_registry.GetParams(self._model_name, 'Test')
    if (issubclass(cfg.cls, base_model.MultiTaskModel) and
        not FLAGS.model_task_name):
      tf.logging.info('Cannot write inference graphs for multi-task model '
                      'when model_task_name is not specified.')
      return
    try:
      filename_prefix = 'inference'
      if FLAGS.model_task_name:
        filename_prefix = '%s_inference' % FLAGS.model_task_name
      filename_prefix = os.path.join(inference_graph_dir, filename_prefix)
      # Standard inference graph.
      self.inference_graph_exporter.InferenceGraphExporter.Export(
          model_cfg=cfg,
          model_task_name=FLAGS.model_task_name,
          export_path=filename_prefix + '.pbtxt')
    except NotImplementedError as e:
      tf.logging.error('Cannot write inference graph: %s', e)

    # TPU inference graph. Not all models support it so fail silently.
    try:
      self.inference_graph_exporter.InferenceGraphExporter.Export(
          model_cfg=cfg,
          model_task_name=FLAGS.model_task_name,
          device_options=self.inference_graph_exporter.InferenceDeviceOptions(
              device='tpu',
              retain_device_placement=False,
              var_options='ON_DEVICE',
              gen_init_op=True,
              dtype_override=None),
          export_path=filename_prefix + '_tpu.pbtxt')
    except Exception as e:  # pylint: disable=broad-except
      tf.logging.info('Error exporting TPU inference graph: %s' % e)

  def Start(self):
    """Start the process."""
    tf.logging.set_verbosity(tf.logging.INFO)

    assert self.model_registry.GetClass(
        self._model_name), ('Model %s is not found.' % FLAGS.model)

    if FLAGS.mode == 'inspect_model':
      self.InspectModel()
      return

    if FLAGS.mode == 'inspect_evaler':
      self.InspectDatasets()
      return

    if FLAGS.mode == 'inspect_decoder':
      self.InspectDecoder()
      return

    if FLAGS.mode == 'write_inference_graph':
      self.WriteInferenceGraph()
      return

    assert FLAGS.mode in ['sync', 'async']

    if FLAGS.mode == 'shell':
      _StartShell(locals())
      return

    self.MaybeConfigRunLocally()
    self.MaybeConfigRunDistributed()
    self.MaybeLaunchTensorFlow()
    self.StartRunners(self.CreateRunners(FLAGS.job.split(','), FLAGS.logdir))


def main(unused_argv):
  # pylint: disable=g-import-not-at-top
  # pylint: disable=unused-variable
  from lingvo import model_imports
  RunnerManager(FLAGS.model).Start()


if __name__ == '__main__':
  tf.app.run(main)
