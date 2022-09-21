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
"""Graph based runners."""

import contextlib
import os
import re
import threading
import time

from absl import flags
from lingvo import pdb_wrapper
import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import cluster_factory
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import summary_utils
from lingvo.core import tpu_embedding_layers
import numpy as np

from lingvo import base_runner
from google.protobuf import text_format

# pylint:disable=g-direct-tensorflow-import
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import training_loop as tpu_training_loop
from tensorflow.python.tpu.ops import tpu_ops
# pylint:enable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

tf.flags.DEFINE_bool(
    'checkpoint_in_trainer_cpu', False,
    'Whether to enable checkpointing in Trainer, allowing for '
    'operation without a separate Controller task.'
    'This flag also disables checkpointing from the Controller, '
    'but still allows it to write summaries.')


# useful for debugging.
def StartShell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython  # pylint: disable=g-import-not-at-top

  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


class Controller(base_runner.BaseRunner):
  """Controller for a training cluster."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if py_utils.IsEagerMode():
      raise RuntimeError('The controller job is not supported in eager mode.')
    self._job_name = 'controller'
    assert not self._model_task_name, 'Controller needs all tasks!'
    self._control_dir = os.path.join(self._logdir, 'control')
    tf.io.gfile.makedirs(self._control_dir)
    self._checkpoint_in_controller = True
    if FLAGS.checkpoint_in_trainer_tpu or FLAGS.checkpoint_in_trainer_cpu:
      self._checkpoint_in_controller = False
    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._summary_writer = self._CreateSummaryWriter(self._control_dir)
        self._model = self.params.Instantiate()
        self._params = self._model.params
        self._model.ConstructFPropBPropGraph()
        self._summary_op = tf.summary.merge_all()
        if self._checkpoint_in_controller:
          # TODO(laigd): consider disabling async checkpointing for controller
          # since it does not affect training efficiency.
          self._checkpointer = self._CreateCheckpointer(self._train_dir,
                                                        self._model)
        # Get the global_variables_initializer after creating the Checkpointer,
        # since it may create additional variables used by async checkpointing.
        self._initialize_tables = tf.tables_initializer()
        self._initialize_local_vars = tf.local_variables_initializer()
        self._initialize_global_vars = tf.global_variables_initializer()
        self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)

    self._ExportMetrics(params=self.params)
    self._model_analysis, self._total_num_params = summary_utils.ModelAnalysis(
        self._model, FLAGS.inspect_model_topn, FLAGS.inspect_model_part_regex)
    py_utils.LogMultiLines('MODEL ANALYSIS', self._model_analysis)
    self._WriteToLog(self._model_analysis, self._control_dir,
                     'model_analysis.txt')
    self._WriteToLog(self.params.ToText(), self._control_dir, 'params.txt')
    self._WriteToLog(
        text_format.MessageToString(self.params.ToProto(), as_utf8=True),
        self._control_dir, 'params.pbtxt')
    self._summary_writer.add_graph(self._graph)

  def Start(self):
    super().Start()
    self._RunLoop('controller', self._Loop)

  def StartEnqueueOp(self, op):
    self._RunLoop(
        'controller/enqueue_op/%s' % op.name, self._LoopEnqueue, loop_args=[op])

  def _Loop(self):
    with tf.container(self._container_id), self._GetSession() as sess:
      if FLAGS.interactive:
        # Into interactive debugging mode.
        StartShell(locals())
        return

      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      for task in self._model.tasks:
        task.input.Initialize(sess)

      # TODO(zhifengc): Moves these options into params.
      tp = self.params.train
      summary_interval_steps = tp.summary_interval_steps
      save_interval_seconds = tp.save_interval_seconds
      next_summary_step = 1

      if not self._checkpoint_in_controller:
        global_step = self._WaitUntilInit(sess)

      while True:
        now = time.time()
        next_iteration_seconds = now + min(
            10, save_interval_seconds)  # 10 seconds or less

        if self._checkpoint_in_controller:
          # Init/restore variable if needed.
          self._checkpointer.RestoreIfNeeded(sess)

        global_step = sess.run(self._model.global_step)
        if self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          if self._checkpoint_in_controller:
            self._checkpointer.Save(sess, global_step)
          sess.close()
          self._DequeueThreadComplete()
          return

        if self._checkpoint_in_controller:
          # Checkpoint if it's time.
          self._checkpointer.MaybeSave(sess, global_step)

        # Summary.
        if self._summary_op is not None and global_step >= next_summary_step:
          global_step, summary_str = sess.run(
              [self._model.global_step, self._summary_op])
          next_summary_step = global_step + summary_interval_steps

          if isinstance(summary_str, np.ndarray) and summary_str.size == 0:
            tf.logging.info('Skipping summary: %s', summary_str)
          else:
            self._summary_writer.add_summary(summary_str, global_step)
          tf.logging.info('Write summary @%s', global_step)
          self._SummarizeValue(global_step, 'total_num_params',
                               self._total_num_params)
          tf.logging.info('Write summary done: step %d', global_step)

        now = time.time()
        if now < next_iteration_seconds:
          time.sleep(next_iteration_seconds - now)

  def _SummarizeValue(self, step, tag, value):
    self._summary_writer.add_summary(
        metrics.CreateScalarSummary(tag, value), step)


class Trainer(base_runner.BaseRunner):
  """Trainer on non-TPU."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._job_name = 'trainer'
    with self._graph.as_default(), tf.container(self._container_id):
      try:
        self._task_probs_summary_writers = []
        for task in self._model.task_schedule.tasks:
          path = os.path.join(os.path.join(self._train_dir, task))
          tf.io.gfile.makedirs(path)
          self._task_probs_summary_writers.append(
              self._CreateSummaryWriter(path))
      except AttributeError:
        tf.logging.info('AttributeError. Expected for single task models.')
        self._task_probs_summary_writers = []

      if self.params.cluster.task == 0:
        self._summary_writer = self._CreateSummaryWriter(self._train_dir)
        self._CreateTF2SummaryWriter(self._train_dir)
      else:
        self._summary_writer = None

      with self._cluster, tf.device(
          self._cluster.GetPlacer()), self._TF2SummaryContext():
        self._model = self.params.Instantiate()
        self._params = self._model.params
        self._model.ConstructFPropBPropGraph()
      self._CreateTF2SummaryOps()
      if FLAGS.checkpoint_in_trainer_cpu:
        self._checkpointer = checkpointer.Checkpointer(self._train_dir,
                                                       self._model)

      self._initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      tf.logging.info('Trainer number of enqueue ops: %d',
                      len(self.enqueue_ops))

    self._step_rate_tracker = summary_utils.StepRateTracker()

    if self.params.cluster.task == 0:
      self._WriteToLog(self.params.ToText(), self._train_dir,
                       'trainer_params.txt')
    worker_id = self.params.cluster.task
    self._start_up_delay_steps = (((worker_id + 1) * worker_id / 2) *
                                  self.params.train.start_up_delay_steps)

  def _SummarizeValue(self, steps, tag, value, writer=None):
    if writer:
      writer.add_summary(metrics.CreateScalarSummary(tag, value), steps)
    elif self._summary_writer:
      self._summary_writer.add_summary(
          metrics.CreateScalarSummary(tag, value), steps)

  def Start(self):
    super().Start()
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
    return super()._LoopEnqueue(op)

  def _Loop(self):
    # Evaler/Controller jobs may find that the trial is infeasible and report
    # done earlier. This is an important check since the trainer may retry
    # indefinitely without it.
    if self._trial.ShouldStop():
      tf.logging.info('Training skipped (trial requested to stop).')
      return
    with tf.container(
        self._container_id), self._cluster, self._GetSession() as sess:
      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      self._InitializeTF2SummaryWriter(sess)
      for task in self._model.tasks:
        task.input.Initialize(sess)

      if FLAGS.checkpoint_in_trainer_cpu:
        self._checkpointer.Restore(sess, force_reinitialize=True)
        global_step = sess.run(py_utils.GetGlobalStep())
      else:
        global_step = self._WaitUntilInit(sess, self._start_up_delay_steps)

      status_interval_steps = 100
      next_status_step = 1
      eval_metrics = None
      while True:
        if (self._trial.ShouldStopAndMaybeReport(global_step, eval_metrics) or
            self._ShouldStop(sess, global_step)):
          tf.logging.info('Training finished.')
          if self._early_stop:
            time.sleep(300)  # controller hangs if it doesn't finish first
          self._DequeueThreadComplete()
          return

        # If a task is explicitly specified, only train that task.
        if self._model_task_name:
          task = self._model.GetTask(self._model_task_name)
        else:
          # Note: This is a slightly stale global_step value from the previous
          # sess.run() call.
          # For multi-task models, `self._model.task_schedule.cur_probs` will
          # be updated.
          task = self._model.SampleTask(global_step)
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

        (_, eval_metrics, per_example_tensors) = sess.run([
            task.train_op,
            task.eval_metrics,
            task.per_example_tensors,
        ])
        # Explicitly fetch global_step after running train_op.
        # TODO(b/151181934): Investigate this behavior further.
        task_global_step = sess.run(task.global_step)
        task.ProcessFPropResults(sess, task_global_step, eval_metrics,
                                 per_example_tensors)
        self._RunTF2SummaryOps(sess)
        global_step = sess.run(self._model.global_step)
        step_rate, example_rate, total_examples = (
            self._step_rate_tracker.ComputeStepRate(
                global_step, eval_metrics['num_samples_in_batch'][0]))
        self._SummarizeValue(global_step, 'global_step/sec', step_rate)
        self._SummarizeValue(global_step, 'examples/sec', example_rate)
        self._SummarizeValue(global_step, 'total_samples', total_examples)

        msg = 'step:%6d, steps/sec: %0.2f, examples/sec: %0.2f' % (
            global_step, step_rate, example_rate)
        for key, (val, _) in sorted(eval_metrics.items()):
          msg += ' %s:%.8g' % (key, val)
          self._SummarizeValue(global_step, key, val)
        if global_step >= next_status_step:
          self._SetStatusMessage(msg)
          self._ExportMetrics(
              # Metrics expects python int, but global_step is numpy.int64.
              global_step=int(global_step),
              step_rate=step_rate,
              example_rate=example_rate)
          next_status_step = global_step + status_interval_steps
        else:
          tf.logging.info(msg)
        self._model.ProcessFPropResults(sess, global_step, eval_metrics,
                                        per_example_tensors)
        if (FLAGS.checkpoint_in_trainer_cpu and
            self._checkpointer.ShouldSave(global_step)):
          self._checkpointer.Save(sess, global_step, sync=True)


class TrainerTpu(base_runner.BaseRunner):
  """Trainer on TPU."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if py_utils.IsEagerMode():
      raise RuntimeError('TrainerTpu is not supported in eager mode. '
                         'Please run with --use_executor '
                         '(or --job=executor_tpu if running locally).')
    self._job_name = 'trainer_tpu'

    # Multiple TPU trainer tasks not tested/implemented.
    assert self._cluster.num_replicas == 1
    data_parallelism = self._cluster.num_splits_per_client
    assert data_parallelism
    num_devices_per_split = self._cluster.num_devices_per_split
    tf.logging.info('data_parallelism: %d, num_devices_per_split: %d',
                    data_parallelism, num_devices_per_split)

    self._steps_per_loop = min(self.params.train.tpu_steps_per_loop,
                               self.params.train.max_steps)
    self._step_rate_tracker = summary_utils.StepRateTracker()

    self._compile_op = None

    self._initialized = threading.Event()

    tf.logging.info(
        'Creating TrainerTpu using data parallelism %s '
        'and %s steps_per_loop', data_parallelism, self._steps_per_loop)

    @py_utils.RetryOnTransientTfError()
    def _WaitUntilInitTpu():
      """Wait until the model is ready."""
      try:
        # tpu.initialize_system() is called with None as embedding_config, as
        # embedding_config is not available yet. Later in _Loop, it is called
        # with the correct embedding_config. Since it cannot be called twice in
        # the same graph with different embedding_config, we use a dummy_graph
        # here.
        dummy_graph = tf.Graph()
        with dummy_graph.as_default():
          tpu_initialize_system_op = tf.tpu.initialize_system(
              embedding_config=None, job=None)

        with self._GetSession(graph=dummy_graph) as sess:
          topology = sess.run(tpu_initialize_system_op)

        if self.params.train.tpu_computation_shape is None:
          computation_shape = py_utils.ComputationShape(num_devices_per_split,
                                                        topology)
        else:
          computation_shape = self.params.train.tpu_computation_shape
          assert num_devices_per_split == np.prod(computation_shape)

        if self.params.train.tpu_device_order_mode is None:
          device_assignment = device_assignment_lib.device_assignment(
              topology,
              computation_shape=computation_shape,
              num_replicas=data_parallelism)
        else:
          device_assignment = device_assignment_lib.device_assignment(
              topology,
              computation_shape=computation_shape,
              num_replicas=data_parallelism,
              device_order_mode=self.params.train.tpu_device_order_mode)
        py_utils.SetTpuDeviceAssignment(device_assignment)
        tf.logging.info('device_assignment.core_assignment: %s',
                        str(device_assignment.core_assignment))
        tf.logging.info('device_assignment.topology.device_coordinates: %s',
                        str(device_assignment.topology.device_coordinates))
      except py_utils.transient_tf_errors as e:
        tf.logging.info('TPU initialization failed: %s', e)
        raise

    _WaitUntilInitTpu()

    with self._graph.as_default(), tf.container(
        self._container_id), contextlib.ExitStack() as stack:
      if FLAGS.pdb_on_exception:
        stack.enter_context(pdb_wrapper.catch_post_mortem())
      self._summary_writer = self._CreateSummaryWriter(self._train_dir)
      self._CreateTF2SummaryWriter(self._train_dir)
      with self._cluster, tf.device(
          self._cluster.GetPlacer()), self._TF2SummaryContext():
        self._model = self.params.Instantiate()
        self._task = self._model.GetTask()
        self._task.input.InfeedSetupGraph()
        self._eval_metrics = metrics.TpuEvalMetrics()
        # Needed due to the AddExtraTheta() reference to global_step when
        # instantiating the InputGenerator.
        _ = py_utils.GetOrCreateGlobalStepVar()
        self._CreateTF2SummaryOps()

        self._input_stats_summary_interval_steps = (
            self._task.input.params.input_stats_summary_interval_steps)

        def TpuTrainStep(*args):
          """Train a shard of a batch on a single TPU core.

          Args:
            *args: metrics values from previous steps.

          Returns:
            New summed metrics values and a train_op.
          """
          self._model.ConstructFPropBPropGraph()
          tpu_embedding_collection = (
              tpu_embedding_layers.TpuEmbeddingCollection.Get())
          self._load_ops = tpu_embedding_collection.load_ops
          self._retrieve_ops = tpu_embedding_collection.retrieve_ops
          self._tpu_embedding = tpu_embedding_collection.tpu_embedding

          per_step_eval_metrics = self._eval_metrics.SetMetrics(
              self._task.eval_metrics, args)
          outfeed_op = self._OutfeedEnqueue(self._task.per_example_tensors)
          summed_metrics = []
          assert len(per_step_eval_metrics) == len(args)
          with tf.control_dependencies([outfeed_op]):
            for x, y in zip(per_step_eval_metrics, args):
              summed_metrics.append(x + y)
          return summed_metrics + [self._task.train_op]

        @tpu_function.on_device_training_loop
        def TpuTrain():
          loop_result = tpu_training_loop.repeat(
              self._steps_per_loop,
              TpuTrainStep,
              inputs=self._eval_metrics.initial_values,
              name='train_loop')
          # Final metrics are the avg across self._steps_per_loop steps.
          return self._eval_metrics.FinalizeMetrics(loop_result)

        self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
            TpuTrain,
            num_shards=data_parallelism,
            device_assignment=py_utils.GetTpuDeviceAssignment())
        outfeed_dequeue_op = self._OutfeedDequeueLoop(
            self._task.per_example_tensors, self._steps_per_loop,
            self._cluster.num_splits_per_client)

        def _ConstructPostTrainingLoop(train_loop_op, outfeed_dequeue_op):
          """Returns the op for tpu training with tail cpu computation."""
          # Adds a tail computation that is run after the tpu_training loop
          # step finishes. This allows us to run certain computation that
          # acts on the variable between tpu_train_loop iterations and
          # amortizing the cost of the operations. Alternative of running
          # tpu.outside_compilation & using tf.cond is expensive.
          with tf.control_dependencies(train_loop_op):
            self._model.ConstructPostTrainingLoop(outfeed_dequeue_op)
            with tf.control_dependencies([self._task.post_training_loop_op]):
              return ([[tf.identity(o) for o in train_loop_op],
                       outfeed_dequeue_op])

        # Get metric result from a single replica; they are all same here.
        all_tpu_ops = [t[0] for t in batch_parallel_res]
        self._tpu_train_ops = (
            _ConstructPostTrainingLoop(all_tpu_ops, outfeed_dequeue_op))

      if FLAGS.checkpoint_in_trainer_tpu:
        self._checkpointer = checkpointer.Checkpointer(self._train_dir,
                                                       self._model)
      # Get the global_variables_initializer after creating the Checkpointer,
      # since it may create additional variables used by async checkpointing.
      self._initialize_local_vars = tf.local_variables_initializer()
      self._initialize_global_vars = tf.global_variables_initializer()
      self._initialize_tables = tf.tables_initializer()

      self._tpu_infeed_op = self._task.input.tpu_infeed_op
      if not self._retrieve_ops:
        # When retrieve ops for TPU embedding is present, we use _InfeedLoop
        # instead of enqueue_ops so the training loop will be driven by the main
        # thread.
        self.enqueue_ops = self._tpu_infeed_op
      tf.logging.info('TrainerTpu number of enqueue ops: %d',
                      len(self._tpu_infeed_op))

    if self._task.input.input_data_summary_layout is not None:
      self._summary_writer.add_summary(
          self._task.input.input_data_summary_layout)

    if FLAGS.checkpoint_in_trainer_tpu:
      self._model_analysis, self._total_num_params = (
          summary_utils.ModelAnalysis(self._model, FLAGS.inspect_model_topn,
                                      FLAGS.inspect_model_part_regex))
      py_utils.LogMultiLines('MODEL ANALYSIS', self._model_analysis)
      self._WriteToLog(self._model_analysis, self._train_dir,
                       'model_analysis.txt')

    # Saves the trainer params.
    self._WriteToLog(self.params.ToText(), self._train_dir,
                     'trainer_params.txt')

  def _GetSession(self, **kwargs):
    return super()._GetSession(cluster_def=self._worker_cluster_def, **kwargs)

  def _OutfeedEnqueue(self, per_example_tensors):
    if not per_example_tensors:
      return tf.no_op()
    per_example_tensors = py_utils.NestedMap(per_example_tensors)
    return tpu_ops.outfeed_enqueue_tuple(per_example_tensors.Flatten())

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
      for replica in range(device_assignment.num_replicas):
        for core in range(device_assignment.num_cores_per_replica):
          with tf.device(device_assignment.host_device(replica, core)):
            outfeed_devices.append(
                tpu_ops.outfeed_dequeue_tuple(
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

    output_arrays = []
    for i in range(len(tensor_shapes)):
      output_arrays.append(
          tf.TensorArray(
              tensor_types[i],
              size=num_loops * num_devices,
              element_shape=tensor_shapes[i]))
    # Loop once for each time that TpuTrainStep runs.
    output_arrays = tf.while_loop(
        LoopCond, LoopBody, [0] + output_arrays, parallel_iterations=1)[1:]
    concatenated_arrays = [array.concat() for array in output_arrays]
    return dict(zip(sorted(per_example_tensors), concatenated_arrays))

  def _CleanUp(self):
    # If there's an exception, we want _LoopEnqueue to wait until
    # everything is initialized before starting up.
    self._initialized.clear()

  def Start(self):
    super().Start()
    # Run training.
    self._RunLoop('trainer', self._Loop, cleanup_func=self._CleanUp)

  def _InfeedLoop(self, sess):
    tf.logging.info('_InfeedLoop start')
    for _ in range(self._steps_per_loop):
      sess.run(self._tpu_infeed_op)

  def StartEnqueueOp(self, op):
    # When retrieve ops for TPU embedding is present, we use _InfeedLoop above
    # instead to make sure enqueue and retrieve does not happen at the same
    # time as required by TPU embedding.
    # We can remove this by using a tf.while_loop driven infeed op.
    assert not self._retrieve_ops
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
    tf.logging.info('_LoopEnqueue waiting for _initialized...')
    self._initialized.wait()
    tf.logging.info('_LoopEnqueue proceeding.')

    # The global step may not be initialized in this thread if the target server
    # uses session state isolation (e.g. Cloud TPUs).
    sess = self._GetSession()
    if FLAGS.checkpoint_in_trainer_tpu:
      self._checkpointer.RestoreGlobalStepIfNeeded(sess)

    # Get merged summary op for training related input data stats from the
    # tasks's input generator.
    self._merged_input_data_summary_op = (
        self._task.input.merged_input_data_summary_op)

    return super()._LoopEnqueue(op, sess)

  def _Loop(self):
    # Evaler/Controller jobs may find that the trial is infeasible and report
    # done earlier. This is an important check since the trainer may retry
    # indefinitely without it.
    if self._trial.ShouldStop():
      tf.logging.info('Training skipped (trial requested to stop).')
      self._DequeueThreadComplete()
      return
    with tf.container(
        self._container_id), self._cluster, self._GetSession() as sess:
      config_proto = (
          self._tpu_embedding.config_proto
          if self._tpu_embedding is not None else None)
      sess.run(
          tf.tpu.initialize_system(embedding_config=config_proto, job=None))
      sess.run(self._initialize_tables)
      sess.run(self._initialize_local_vars)
      self._InitializeTF2SummaryWriter(sess)

      if FLAGS.run_locally == 'tpu':
        sess.run(self._initialize_global_vars)

      self._SetStatusMessage('Compiling ...')
      compilation_result = sess.run(self._compile_op)
      comp_result_proto = tpu_compilation_result.CompilationResultProto()
      comp_result_proto.ParseFromString(compilation_result)
      if comp_result_proto.status_error_message:
        tf.logging.fatal('Compilation failed: {}'.format(
            comp_result_proto.status_error_message))
      self._SetStatusMessage('Compiling done.')

      if FLAGS.checkpoint_in_trainer_tpu:
        # For b/134415393 -- better to initialize to a known state than
        # rely on what's in the session on the trainer/TPU worker.
        tf.logging.info('TrainerTpu: Force restore or initialize.')
        self._checkpointer.Restore(sess, force_reinitialize=True)

      global_step = sess.run(self._model.global_step)
      self._initialized.set()
      eval_metrics = None
      if FLAGS.checkpoint_in_trainer_tpu and global_step == 0:
        # Always save a ckpt at step 0.
        self._checkpointer.MaybeSave(sess, global_step)

      if self._retrieve_ops:
        sess.run(self._load_ops)
        self._task.input.Initialize(sess)

      while True:
        train_steps_start = time.perf_counter()

        if self._trial.ShouldStopAndMaybeReport(
            global_step, eval_metrics) or self._ShouldEarlyStop(sess):
          # Early terminate gracefully by setting a new max step horizon: three
          # more TPU steps to ensure that the enqueue ops can gracefully
          # terminate as well. Otherwise, the enqueue thread may be stuck, e.g.,
          # when the queue is filled and the enqueue thread is blocked when
          # pushing new data to the queue, if the trainer thread decides to
          # early stop (i.e., `self._ShouldEarlyStop(sess)` is true), then the
          # enqueue thread could be blocked forever as the trainer thread would
          # never consume any new data from the queue. After setting the new
          # max step horizon, the trainer thread would continue run for 3 loops
          # (3K global steps usually), so the enqueue thread could get a chance
          # to move forward and run `_ShouldStop()` to stop gracefully.
          # Updated this to account for `tpu_infeed_parallelism` which could
          # allow for more enqueue threads to get further ahead of the traiiner
          # thread.
          if self._max_steps_for_early_stop is None:
            tpu_infeed_parallelism = self._task.input.params.tpu_infeed_parallelism
            self._max_steps_for_early_stop = global_step + 3 * tpu_infeed_parallelism * self._steps_per_loop
            tf.logging.info('Early stopping at step: %d',
                            self._max_steps_for_early_stop)

        def _RunSave(sync):
          if self._retrieve_ops:
            # Running retrieve ops is expensive, so do it only before
            # checkpointing.
            tf.logging.info('Retrieve params.')
            sess.run(self._retrieve_ops)
            tf.logging.info('Retrieve params done.')

          checkpoint_write_start = time.perf_counter()
          self._checkpointer.Save(sess, global_step, sync=sync)
          return time.perf_counter() - checkpoint_write_start

        if self._ShouldStop(sess, global_step, check_early_stop=False):
          tf.logging.info('Training finished.')
          if FLAGS.checkpoint_in_trainer_tpu:
            _RunSave(True)  # Wait for the save ops to finish before exit.
          self._DequeueThreadComplete()
          return

        if self._retrieve_ops:
          assert FLAGS.checkpoint_in_trainer_tpu, (
              'When retrieve ops for TPU embedding is present, checkpointing '
              'need to happen in TrainerTpu to avoid race conditions.')
          infeed_loop_thread = threading.Thread(
              target=self._InfeedLoop, args=(sess,))
          infeed_loop_thread.start()

        tpu_train_op_start = time.perf_counter()
        values, outfeeds = sess.run(self._tpu_train_ops)
        tpu_train_op_secs = time.perf_counter() - tpu_train_op_start

        if self._retrieve_ops:
          # Wait for infeed loop to finish to avoid running it in parallel with
          # retrieve ops.
          infeed_loop_thread.join()

        self._eval_metrics.PackMetricsValues(values)
        eval_metrics = self._eval_metrics.metrics

        # Note: global_step is incremented by self._steps_per_loop by the
        # previous sess.run call.
        task_global_step = sess.run(self._task.global_step)
        global_step = sess.run(self._model.global_step)

        if not self._task.per_example_tensors:
          outfeeds = {}
        self._task.ProcessFPropResults(sess, task_global_step, eval_metrics,
                                       outfeeds)
        self._model.ProcessFPropResults(sess, global_step, eval_metrics,
                                        outfeeds)

        step_rate, example_rate, total_examples = (
            self._step_rate_tracker.ComputeStepRate(
                global_step,
                eval_metrics['num_samples_in_batch'][0] * self._steps_per_loop))
        self._RunTF2SummaryOps(sess)
        self._SummarizeValue(global_step, 'global_step/sec', step_rate)
        self._SummarizeValue(global_step, 'examples/sec', example_rate)
        self._SummarizeValue(global_step, 'total_samples', total_examples)
        if FLAGS.checkpoint_in_trainer_tpu:
          self._SummarizeValue(global_step, 'total_num_params',
                               self._total_num_params)
        msg = 'step:%6d, steps/sec: %0.2f, examples/sec: %0.2f' % (
            global_step, step_rate, example_rate)
        for key, (val, _) in sorted(eval_metrics.items()):
          msg += ' %s:%.8g' % (key, val)
          self._SummarizeValue(global_step, key, val)

        self._SetStatusMessage(msg)

        # Add model eval metrics to early stop metric history.
        for metric_name, (metric_value, _) in eval_metrics.items():
          self._UpdateEarlyStopMetric('train', global_step, metric_name,
                                      metric_value)

        checkpoint_write_secs = 0.0
        if (FLAGS.checkpoint_in_trainer_tpu and
            self._checkpointer.ShouldSave(global_step)):
          checkpoint_write_secs = _RunSave(False)  # Save asynchronously.
        train_steps_secs = time.perf_counter() - train_steps_start
        self._ExportMetrics(
            # Metrics expects python int, but global_step is numpy.int64.
            global_step=int(global_step),
            step_rate=step_rate,
            example_rate=example_rate,
            tpu_train_op_secs=tpu_train_op_secs,
            checkpoint_write_secs=checkpoint_write_secs,
            total_train_steps_secs=train_steps_secs,
            **{k: v[0] for k, v in eval_metrics.items()})


class Evaler(base_runner.BaseRunner):
  """Evaler."""

  def __init__(self, eval_type, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._job_name = 'evaler_' + eval_type
    self._output_name = 'eval_' + eval_type
    self._export = eval_type == 'train'
    if not self._export:
      tf.logging.info(f'Job {self._job_name} will not export the model.')
    self.params.cluster.do_eval = True
    self._cluster = cluster_factory.Cluster(self.params.cluster)
    self._eval_dir = os.path.join(self._logdir, self._output_name)
    if self._model_task_name:
      self._eval_dir += '_' + str(self._model_task_name)
    tf.io.gfile.makedirs(self._eval_dir)

    self._eval_path = None
    # Multitask params doesn't have 'task'.
    if 'task' in self.params:
      self._eval_path = checkpointer.GetSpecificCheckpoint(
          self.params.task.eval.load_checkpoint_from)

    self._should_report_metrics = self._job_name.startswith(
        self._cluster.reporting_job)

    with self._graph.as_default(), tf.container(self._container_id):
      self._summary_writer = self._CreateSummaryWriter(self._eval_dir)
      self._CreateTF2SummaryWriter(self._eval_dir)
      with self._cluster, tf.device(
          self._cluster.GetPlacer()), self._TF2SummaryContext():
        self._model = self.params.Instantiate()
        self._params = self._model.params
        self._model.ConstructFPropGraph()
        self._task = self._model.GetTask(self._model_task_name)
        self._checkpointer = self._CreateCheckpointer(self._train_dir,
                                                      self._model)
      self._CreateTF2SummaryOps()
      self._summary_op = tf.summary.merge_all()
      self._initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      # No queues are allowed for eval models.
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      assert not self.enqueue_ops

      self._input_stats_summary_interval_steps = (
          self._task.input.params.input_stats_summary_interval_steps)

    # Saves the graph def.
    self._WriteToLog(self.params.ToText(), self._eval_dir, 'params.txt')
    if self.params.cluster.task == 0:
      tf.io.write_graph(self._graph.as_graph_def(), self._eval_dir,
                        '%s.pbtxt' % self._output_name)

  def Start(self):
    super().Start()
    self._RunLoop(self._job_name, self._Loop)

  def _Loop(self):
    """The main loop."""
    with tf.container(
        self._container_id), self._cluster, self._GetSession() as sess:
      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      self._InitializeTF2SummaryWriter(sess)
      self._task.input.Initialize(sess)

      if self._eval_path:
        self._EvalOnce(sess, self._eval_path)
        self._UpdateProcessedCheckpoints(self._eval_dir, self._eval_path)
      elif self._task.params.eval.eval_all_checkpoints:
        self._RunOnAllCheckpoints(sess, self._EvalOnce, self._eval_dir)
      else:
        self._RunOnLatestCheckpoints(sess, self._EvalOnce, self._eval_dir)

    if self._should_report_metrics:
      tf.logging.info('Reporting trial done.')
      self._trial.ReportDone()
    tf.logging.info('Evaluation finished.')

  def EvalLatestCheckpoint(self, last_path=None):
    """Runs eval once on the latest checkpoint."""
    with tf.container(
        self._container_id), self._cluster, self._GetSession() as sess:
      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      self._task.input.Initialize(sess)
      path = tf.train.latest_checkpoint(self._train_dir)
      if not path:
        tf.logging.info('No checkpoint available.')
        return
      elif path == last_path:
        tf.logging.info('Latest checkpoint was already evaluated.')
        return

      self._EvalOnce(sess, path)

  def EvalCheckpoint(self, ckpt_id):
    with tf.container(self._container_id), self._GetSession() as sess:
      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      self._task.input.Initialize(sess)
      path = '{}/ckpt-{:08d}'.format(self._train_dir, ckpt_id)
      self._EvalOnce(sess, path)

  def _RemoveScalarSummaries(self, summaries):
    proto = summary_pb2.Summary()
    proto.ParseFromString(summaries)
    for i, value in enumerate(proto.value):
      if value.WhichOneof('value') == 'simple_value':
        del proto.value[i]
    return proto.SerializeToString()

  def _EvalOnce(self, sess, path):
    """Runs evaluation for a batch of samples.

    Args:
      sess: the tf Session.
      path: checkpoint path.
    """

    if not FLAGS.evaler_in_same_address_as_controller:
      self._checkpointer.RestoreFromPath(sess, path)

    global_step = sess.run(py_utils.GetGlobalStep())
    # Save any additional information to disk before evaluation.
    if self._export:
      self._task.Export(path)

    # Check after how many steps checkpoint got saved.
    # And decide whether to run an evaluation.
    if global_step < self._task.params.eval.start_eval_after:
      return

    if self._task.input.params.resettable:
      tf.logging.info('Resetting input_generator.')
      self._task.input_generator.Reset(sess)

    metrics_dict = {
        name: metrics.AverageMetric() for name in self._task.eval_metrics
    }
    num_samples_metric = metrics_dict['num_samples_in_batch']
    samples_per_summary = self._task.params.eval.samples_per_summary
    if samples_per_summary == 0:
      assert self._task.input.params.resettable
    while samples_per_summary == 0 or (num_samples_metric.total_value <
                                       samples_per_summary):
      try:
        is_first_loop = num_samples_metric.total_value == 0
        # NOTE: We intentionally do not let FProp generate scalar summaries by
        # default, because evaler calls FProp multiple times for each
        # checkpoint. Multiple summaries at the same step is often confusing.
        # Instead, models should update eval_metrics and generate aggregate
        # summaries. Other types of summaries (images, audio etc.) will be
        # generated for the first eval batch.
        if self._summary_op is not None and is_first_loop:
          ans, summaries = sess.run([self._task.eval_metrics, self._summary_op])
          summaries = self._RemoveScalarSummaries(summaries)

          # Add non-scalar summaries only for the first batch of data.
          self._summary_writer.add_summary(summaries, global_step)
          self._summary_writer.flush()
        else:
          ans = sess.run(self._task.eval_metrics)

        for name, (value, weight) in ans.items():
          metrics_dict[name].Update(value, weight)
        tf.logging.info('Total examples done: %d/%d',
                        num_samples_metric.total_value, samples_per_summary)
      except tf.errors.OutOfRangeError:
        if not self._task.input.params.resettable:
          raise
        break

    # Replace average values with total values for certain metrics.
    if 'num_predictions' in metrics_dict:
      metrics_dict['num_predictions'].total_weight = 1.0
    if 'num_words' in metrics_dict:
      metrics_dict['num_words'].total_weight = 1.0

    self._RunTF2SummaryOps(sess)
    summaries = {k: v.Summary(k) for k, v in metrics_dict.items()}
    summaries['total_samples'] = metrics.CreateScalarSummary(
        'total_samples', num_samples_metric.total_value)

    # When we have evaluated so many samples, generate a summary.
    self._WriteSummaries(
        self._summary_writer,
        os.path.basename(self._eval_dir),
        global_step,
        summaries,
        text_filename=os.path.join(self._eval_dir,
                                   'score-{:08d}.txt'.format(global_step)))

    # Get merged summaries for input data stats logged by the tasks's input
    # generator and write summaries for the stats.
    if self._task.input.merged_input_data_summary_op is not None:
      input_stats_summary_str = sess.run(
          self._task.input.merged_input_data_summary_op)
      self._WriteInputDataStatSummaries(input_stats_summary_str, global_step)

    if self._should_report_metrics:
      tf.logging.info('Reporting eval measure for step %d.' % global_step)
      self._trial.ReportEvalMeasure(global_step, metrics_dict, path)


def GetDecoderDir(logdir, decoder_type, model_task_name):
  if model_task_name:
    decoder_dir = '%s_%s' % (decoder_type, model_task_name)
  else:
    decoder_dir = decoder_type
  return os.path.join(logdir, decoder_dir)


def _GetCheckpointIdForDecodeOut(ckpt_id_from_file, global_step):
  """Retrieve the checkpoint id for the decoder out file.

  Compares the checkpoint id found in the checkpoint file name to global
  step. If they diverge, uses the retrieved id and prints a warning.

  Args:
   ckpt_id_from_file: Checkpoint Id from the checkpoint file path.
   global_step: int specifying the global step of the model.

  Returns:
   Checkpoint id as int.
  """
  tf.logging.info('Loaded checkpoint is at global step: %d', global_step)
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
    super().__init__(*args, **kwargs)
    self._job_name = 'decoder_' + decoder_type
    self.params.cluster.do_eval = True
    self._cluster = cluster_factory.Cluster(self.params.cluster)
    self._decoder_dir = GetDecoderDir(self._logdir, self._job_name,
                                      self._model_task_name)
    tf.io.gfile.makedirs(self._decoder_dir)

    self._decode_path = None
    # Multitask params doesn't have 'task'.
    if 'task' in self.params:
      self._decode_path = checkpointer.GetSpecificCheckpoint(
          self.params.task.eval.load_checkpoint_from)

    self._should_report_metrics = self._job_name.startswith(
        self._cluster.reporting_job)

    with self._graph.as_default(), tf.container(self._container_id):
      self._summary_writer = self._CreateSummaryWriter(self._decoder_dir)
      self._CreateTF2SummaryWriter(self._decoder_dir)
      with self._cluster, tf.device(
          self._cluster.GetPlacer()), self._TF2SummaryContext():
        self._model = self.params.Instantiate()
        self._params = self._model.params
        self._task = self._model.GetTask(self._model_task_name)
        # Note, different graphs are being constructed for different model
        # tasks, which may result in different node names being chosen.
        # Obviously, variable names has to be stay the same between train and
        # decode.
        input_batch, self._dec_output = self._model.ConstructDecodeGraph(
            self._model_task_name)
        for key in self._task.input_generator.GetCpuPassthroughKeys():
          if key in input_batch:
            if key in self._dec_output:
              tf.logging.warning(f'Key {key} already present in decode output. '
                                 f'Not adding from input batch.')
            else:
              self._dec_output[key] = input_batch[key]

        self._summary_op = tf.summary.merge_all()
        self._checkpointer = self._CreateCheckpointer(self._train_dir,
                                                      self._model)
      self._CreateTF2SummaryOps()
      self._initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      # No queues are allowed for decoder models.
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      assert not self.enqueue_ops

    # Saves the graph def.
    self._WriteToLog(self.params.ToText(), self._decoder_dir, 'params.txt')
    if self.params.cluster.task == 0:
      tf.io.write_graph(self._graph.as_graph_def(), self._decoder_dir,
                        '%s.pbtxt' % self._job_name)

  def Start(self):
    super().Start()
    self._RunLoop(self._job_name, self._Loop)

  def _Loop(self):
    with tf.container(self._container_id), self._cluster, self._GetSession(
        inline=False) as sess:
      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      self._InitializeTF2SummaryWriter(sess)
      self._task.input.Initialize(sess)

      if self._decode_path:
        self.DecodeCheckpoint(sess, self._decode_path)
        py_utils.UpdateProcessedCheckpoints(self._decoder_dir,
                                            self._decode_path)
      elif self._task.params.eval.decode_all_checkpoints:
        self._RunOnAllCheckpoints(sess, self.DecodeCheckpoint,
                                  self._decoder_dir)
      else:
        self._RunOnLatestCheckpoints(sess, self.DecodeCheckpoint,
                                     self._decoder_dir)

    if self._should_report_metrics:
      tf.logging.info('Reporting trial done.')
      self._trial.ReportDone()
    tf.logging.info('Decoding finished.')

  @classmethod
  def GetDecodeOutPath(cls, decoder_dir, checkpoint_id):
    """Gets the path to decode out file."""
    out_dir = cls._GetTtlDir(decoder_dir, duration='7d')
    return os.path.join(out_dir, 'decoder_out_%09d' % checkpoint_id)

  def GetCkptIdFromFile(self, checkpoint_path):
    return int(re.sub(r'.*ckpt-', '', checkpoint_path))

  def _RemoveScalarSummaries(self, summaries):
    proto = tf.Summary()
    proto.ParseFromString(summaries)
    for i, value in enumerate(proto.value):
      if value.WhichOneof('value') == 'simple_value':
        del proto.value[i]
    return proto.SerializeToString()

  def DecodeCheckpoint(self, sess, checkpoint_path):
    """Decodes `samples_per_summary` examples using `checkpoint_path`."""
    p = self._task.params
    ckpt_id_from_file = self.GetCkptIdFromFile(checkpoint_path)
    if ckpt_id_from_file < p.eval.start_decoder_after:
      return

    samples_per_summary = p.eval.decoder_samples_per_summary
    if samples_per_summary is None:
      samples_per_summary = p.eval.samples_per_summary
    if samples_per_summary == 0:
      assert self._task.input.params.resettable
    self._checkpointer.RestoreFromPath(sess, checkpoint_path)

    global_step = sess.run(py_utils.GetGlobalStep())

    if self._task.input.params.resettable:
      tf.logging.info('Resetting input_generator.')
      self._task.input.Reset(sess)

    dec_metrics = self._task.CreateDecoderMetrics()
    if not dec_metrics:
      tf.logging.info('Empty decoder metrics')
      return
    buffered_decode_out = []
    num_examples_metric = dec_metrics['num_samples_in_batch']
    start_time = time.time()
    while samples_per_summary == 0 or (num_examples_metric.total_value <
                                       samples_per_summary):
      try:
        is_first_loop = num_examples_metric.total_value == 0
        tf.logging.info('Fetching dec_output.')
        fetch_start = time.time()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=False)

        # NOTE: We intentionally do not generate scalar summaries by
        # default, because decoder is run  multiple times for each
        # checkpoint. Multiple summaries at the same step is often confusing.
        # Instead, models should generate aggregate summaries using
        # PostProcessDecodeOut. Other types of summaries (images, audio etc.)
        # will be generated for the first eval batch.
        if self._summary_op is not None and is_first_loop:
          dec_out, summaries = sess.run([self._dec_output, self._summary_op],
                                        options=run_options)
          summaries = self._RemoveScalarSummaries(summaries)

          # Add non-scalar summaries only for the first batch of data.
          self._summary_writer.add_summary(summaries, global_step)
          self._summary_writer.flush()
        else:
          dec_out = sess.run(self._dec_output, options=run_options)

        self._RunTF2SummaryOps(sess)
        post_process_start = time.time()
        tf.logging.info('Done fetching (%f seconds)' %
                        (post_process_start - fetch_start))
        decode_out = self._task.PostProcessDecodeOut(dec_out, dec_metrics)
        if decode_out:
          if isinstance(decode_out, dict):
            decode_out = decode_out.items()

          if is_first_loop:
            # Add summaries only for the first batch of data.
            for key, value in decode_out:
              if isinstance(value, tf.Summary):
                tf.logging.info(f'Adding summary {key} with tags '
                                f'{[x.tag for x in value.value]}.')
                self._summary_writer.add_summary(value, global_step)
            self._summary_writer.flush()

          buffered_decode_out.extend(
              kv for kv in decode_out if not isinstance(kv[1], tf.Summary))
        tf.logging.info(
            'Total examples done: %d/%d '
            '(%f seconds decode postprocess)', num_examples_metric.total_value,
            samples_per_summary,
            time.time() - post_process_start)
      except tf.errors.OutOfRangeError:
        if not self._task.input.params.resettable:
          raise
        break
    tf.logging.info('Done decoding ckpt: %s', checkpoint_path)

    summaries = {k: v.Summary(k) for k, v in dec_metrics.items()}
    elapsed_secs = time.time() - start_time
    example_rate = num_examples_metric.total_value / elapsed_secs
    summaries['examples/sec'] = metrics.CreateScalarSummary(
        'examples/sec', example_rate)
    summaries['total_samples'] = metrics.CreateScalarSummary(
        'total_samples', num_examples_metric.total_value)
    self._WriteSummaries(
        self._summary_writer,
        os.path.basename(self._decoder_dir),
        global_step,
        summaries,
        text_filename=os.path.join(self._decoder_dir,
                                   'score-{:08d}.txt'.format(global_step)))
    self._ExportMetrics(
        # Metrics expects python int, but global_step is numpy.int64.
        decode_checkpoint=int(global_step),
        dec_metrics=dec_metrics,
        example_rate=example_rate)
    # global_step and the checkpoint id from the checkpoint file might be
    # different. For consistency of checkpoint filename and decoder_out
    # file, use the checkpoint id as derived from the checkpoint filename.
    checkpoint_id = _GetCheckpointIdForDecodeOut(ckpt_id_from_file, global_step)
    decode_out_path = self.GetDecodeOutPath(self._decoder_dir, checkpoint_id)

    decode_finalize_args = base_model.DecodeFinalizeArgs(
        decode_out_path=decode_out_path, decode_out=buffered_decode_out)
    self._task.DecodeFinalize(decode_finalize_args)

    if self._should_report_metrics:
      tf.logging.info('Reporting eval measure for step %d.' % global_step)
      self._trial.ReportEvalMeasure(global_step, dec_metrics, checkpoint_path)

  def DecodeLatestCheckpoint(self, last_path=None):
    """Runs decoder on the latest checkpoint."""
    with tf.container(
        self._container_id), self._cluster, self._GetSession() as sess:
      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      self._task.input.Initialize(sess)
      path = tf.train.latest_checkpoint(self._train_dir)
      if not path:
        tf.logging.info('No checkpoint available.')
        return
      elif path == last_path:
        tf.logging.info('Latest checkpoint was already decoded.')
        return
      self.DecodeCheckpoint(sess, path)
