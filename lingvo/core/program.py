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
"""Programs for interleaving execution on TPU."""

import _thread
import collections
import contextlib
import functools
import multiprocessing.dummy
import os
import queue
import time

from lingvo import base_trial
import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import cluster_factory
from lingvo.core import hyperparams
from lingvo.core import metrics
from lingvo.core import ml_perf_log as mlp_log
from lingvo.core import program_utils
from lingvo.core import py_utils
from lingvo.core import summary_utils

# pylint:disable=g-direct-tensorflow-import
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.eager import context
from tensorflow.python.eager import executor
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import training_loop as tpu_training_loop
from tensorflow.python.tpu.ops import tpu_ops

# pylint:enable=g-direct-tensorflow-import
FLAGS = tf.flags.FLAGS
# According to the Runtime team, by default (set to True), even if we use
# async executors locally, the remote host will still run the functions
# sequentially. We have to set this to False in order to enable parallelism
# in the remote hosts as well. As long as the infeed is independent of the
# training part, we are safe to do so.
# TODO(haoyuzhang): I have a question: this seems to be a very subtle behavior,
# because the formal docs only mentioned setting this to False for better
# debugging. Will this behavior get changed in the future?
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = 'False'


def _RewriteV1SummaryInTF2(summary, global_step):
  # TODO(laigd): make this work for other v1 summaries.
  if summary.value:
    for value in summary.value:
      if value.WhichOneof('value') != 'simple_value':
        return
      tf.compat.v2.summary.scalar(
          value.tag, value.simple_value, step=global_step)


class BaseProgram:
  """A Program associated with a Task.

  This is inspired by the "program" multi-tenancy that TPUs
  support. Essentially, each program corresponds with a
  sub-graph can exist in the same Graph/Session.

  Upon first execution, it is XLA/JIT compiled and is subsequently
  available to be executed on demand without significant delay.

  Program's provides the following functionality:

    - Builds a sub-graph
    - Writes summaries
    - Runs for pre-determined `steps_per_loop` steps with appropriate infeeds
  """

  @classmethod
  def Params(cls):
    """"Defaults parameters for Programs."""
    p = hyperparams.InstantiableParams(cls)
    p.Define('task', None, 'Underlying task')
    p.Define('logdir', None, 'Log directory')
    p.Define('num_splits_per_client', None, '')
    p.Define('steps_per_loop', None, 'Number of steps to run.')
    p.Define('dataset_name', None,
             'Dataset the program is operating on, eg: "Test"')
    p.Define('name', 'base_program', 'Program name.')
    p.Define('task_name', None,
             'If multi-task, what the high-level task name is')
    p.Define('num_threads', 1, 'Number of threads in multiprocessing pool.')
    p.Define('spmd', False, 'Whether program is running under SPMD mode.')
    p.Define('write_train_input_stats', False,
             'Whether to write input data stats during training.')
    p.Define('max_metrics', 256, 'Overrides TpuEvalMetrics.max_metrics')
    p.Define('ml_perf', None, 'MLPerf config')
    return p

  def __init__(self,
               params,
               shared_model=None,
               trial=base_trial.NoOpTrial(),
               ema=None,
               **kwargs):
    self.params = params.Copy()
    p = self.params
    p.task = trial.OverrideModelParams(p.task)
    self._task_params = p.task
    self._logdir = p.logdir
    self._task_name = p.task_name
    self._program_name = ''
    self._shared_model = shared_model
    self._tf_master = kwargs.pop('tf_master', None)
    self._write_train_input_stats = p.write_train_input_stats
    self._trial = trial
    self._ema = ema

    self._SetProgramDir()
    # Initialized on use; access via self._summary_writer property only.
    self._summary_writer_obj = None

    tf.io.gfile.makedirs(self._logdir)
    # Just a standard spot that all programs may restore from.
    self._checkpoint_dir = os.path.join(self._logdir, 'train')
    tf.io.gfile.makedirs(self._checkpoint_dir)

    self._steps_per_loop = p.steps_per_loop
    self.num_splits_per_client = p.num_splits_per_client
    self.data_parallelism = p.num_splits_per_client

    # Thread Pool for infeed.
    self._infeed_pool = multiprocessing.dummy.Pool(p.num_threads)

    self._compile_op = None
    self._status_msg_fn = None

    # Set input repeat_steps to steps_per_loop, if repeat_steps was undefined
    # but available, and also 'resettable' is False.
    # This allows a repeating input TF Dataset (without reset) to take, for each
    # repeat/loop, exactly steps_per_loop batches of data.
    if (hasattr(self._task_params, 'input') and
        not getattr(self._task_params.input, 'resettable', True) and
        hasattr(self._task_params.input, 'repeat_steps') and
        self._task_params.input.repeat_steps is None and
        self._steps_per_loop > 0):
      tf.logging.info('Setting input repeat_steps to %d', self._steps_per_loop)
      self._task_params.input.repeat_steps = self._steps_per_loop

    self._InitializeVizier()

  def _SetProgramDir(self):
    """Set program dir for output."""
    p = self.params
    # Program dirs are where the summaries are written to.
    if p.task_name:
      program_dir_name = (
          p.task_name + '_' + p.name + '_' + p.dataset_name.lower())
    else:
      program_dir_name = p.name + '_' + p.dataset_name.lower()
    self._program_dir = os.path.join(self._logdir, program_dir_name)
    tf.io.gfile.makedirs(self._program_dir)
    with tf.io.gfile.GFile(os.path.join(self._program_dir, 'params.txt'),
                           'w') as f:
      f.write(p.ToText())

  def _InitializeVizier(self):
    """Checks if this program should report metrics to vizier."""
    p = self.params
    self._should_report_metrics = False

    reporting_job = self._task_params.cluster.reporting_job
    job_split = self._task_params.cluster.reporting_job.split('/')

    if len(job_split) != 2:
      # The default setting for reporting job is 'evaler'. This is not valid
      # for use with program. We only warn only since we may not be in a vizier
      # setting.
      tf.logging.info('reporting_job should be of the form '
                      'program_name/dataset_name with exactly one / '
                      f'instead got {reporting_job}')
      return

    vizier_program_name, vizier_dataset_name = job_split
    if p.name == vizier_program_name and p.dataset_name == vizier_dataset_name:
      tf.logging.info(f'Adding reporting for {reporting_job}')
      self._should_report_metrics = True

  @property
  def _summary_writer(self):
    """Returns the FileWriter object to use for summaries."""
    # Initialize on first use, so that subclasses can override the
    # implementation without creating a default FileWriter in the constructor.
    if self._summary_writer_obj is None:
      if py_utils.IsEagerMode():
        self._summary_writer_obj = tf.compat.v2.summary.create_file_writer(
            self._program_dir)
      else:
        self._summary_writer_obj = tf.summary.FileWriter(self._program_dir)
        # Apply a custom Tensorboard layout for input data stats if writing
        # TF summaries for input data stats is enabled and a custom layout is
        # defined by the input generator.
        if (self._task.input.input_data_summary_layout is not None and
            self._write_train_input_stats):
          self._summary_writer_obj.add_summary(
              self._task.input.input_data_summary_layout)
    return self._summary_writer_obj

  def _SummarizeValue(self, steps, tag, value):
    if py_utils.IsEagerMode():
      with self._summary_writer.as_default():
        tf.compat.v2.summary.scalar(tag, value, step=steps)
    else:
      self._summary_writer.add_summary(
          metrics.CreateScalarSummary(tag, value), steps)

  def _WriteSummaries(self, job_name, global_step, summaries):
    """Write summaries to be viewed by TensorBoard.

    Args:
      job_name: The name of this job ('trainer', 'evaler', etc.)
      global_step: Integer number of trainer steps (not a tensor).
      summaries: Dict of {summary_name: tf.Summary()}.
    """
    if not summaries:
      return
    with contextlib.ExitStack() as stack:
      if py_utils.IsEagerMode():
        stack.enter_context(self._summary_writer.as_default())
      for unused_name, summary in sorted(summaries.items()):
        if py_utils.IsEagerMode():
          _RewriteV1SummaryInTF2(summary, global_step)
        else:
          self._summary_writer.add_summary(summary, global_step)
        if summary.value:
          for value in summary.value:
            if value.HasField('simple_value'):
              tf.logging.info('%s summary on checkpoint@%d %s = %.8g', job_name,
                              global_step, value.tag, value.simple_value)
        self._summary_writer.flush()

  def _WriteInputDataStats(self, sess=None, **unused_kwargs):
    """Write input data stats for model training as TF summaries.

    Args:
      sess: The Tensorflow session.
    """
    if (self._task.input.merged_input_data_summary_op is None or
        not self._write_train_input_stats):
      return

    global_step = sess.run(self._model.global_step)
    if (global_step %
        self._task.input.params.input_stats_summary_interval_steps == 0):
      summary_str = sess.run(self._task.input.merged_input_data_summary_op)
      self._summary_writer.add_summary(summary_str, global_step)
      self._summary_writer.flush()

  def _InfeedLoopForInput(self, dataset_name, inp_instance, sess=None):
    """Infeed loop for specified dataset and input instance."""
    tf.logging.info(f'_InfeedLoop start {self._program_name} '
                    f'on dataset {dataset_name}')
    try:
      for i in range(self._steps_per_loop):
        tf.logging.vlog(1, '_InfeedLoop %d', i)
        sess.run(inp_instance.tpu_infeed_op)
      self._WriteInputDataStats(
          sess, dataset_name=dataset_name, inp_instance=inp_instance)
      tf.logging.info('_InfeedLoop done')
    except Exception as e:
      tf.logging.info('_InfeedLoop exception %r %s', e, e)
      raise

  def _InfeedLoop(self, sess=None):
    """Infeed loop for program's own dataset and input instance."""
    self._InfeedLoopForInput(self.params.dataset_name, self._task.input, sess)

  def _ReportVizierMetrics(self, global_step, metrics_dict):
    """Report metrics to vizier service.

    Args:
      global_step: Int.
      metrics_dict: A dict of metric name -> metric values.

    Returns:
      vizier_early_stop: Boolean, indicates if early stopping has bee requested
        by vizier.
    """
    p = self.params
    if self._should_report_metrics:
      tf.logging.info(f'Reporting Vizier metrics for {p.name}/{p.dataset_name}')
      vizier_early_stop = self._trial.ReportEvalMeasure(global_step,
                                                        metrics_dict, '')
      if global_step >= self._task_params.train.max_steps or vizier_early_stop:
        self._trial.ReportDone()

    else:
      vizier_early_stop = False
    # Export cluster metrics as well.
    cluster_factory.Current().ExportMetrics(
        global_step, {k: metric.value for k, metric in metrics_dict.items()})
    return vizier_early_stop

  def BuildTpuSubgraph(self):
    """Sub classes should construct a model/graph to be executed by Run.

    Specific to TPU execution, this may involve a
    @tpu_function.on_device_training_loop etc.
    """
    raise NotImplementedError()

  def SetStatusMessageFn(self, fn):
    """Workaround since we instantiate programs via Params."""
    self._status_msg_fn = fn

  def SetStatusMessage(self, msg):
    """Write to borglet status."""
    if self._status_msg_fn:
      self._status_msg_fn(msg)
    else:
      tf.logging.info('Status: %s', msg)

  def InitInputs(self, sess=None):
    self.SetStatusMessage('Init inputs %s' % self._program_name)
    self._task.input.Initialize(sess)
    self.SetStatusMessage('Init inputs %s done.' % self._program_name)

  def Compile(self, sess=None):
    """Compile the program using the given session handle."""
    self.InitInputs(sess)
    if not py_utils.IsEagerMode() and self._compile_op is not None:
      self.SetStatusMessage('Compiling %s' % self._program_name)
      result = sess.run(self._compile_op)
      proto = tpu_compilation_result.CompilationResultProto()
      proto.ParseFromString(result)
      if proto.status_error_message:
        error_msg = 'Compilation of {} failed: {}'.format(
            self._program_name, proto.status_error_message)
        self.SetStatusMessage(error_msg)
        tf.logging.fatal(error_msg)

      self.SetStatusMessage('Compiling {} done.'.format(self._program_name))
      tf.logging.info('Compiling %s done.', self._program_name)

  def Run(self, sess=None, threadpool=None):
    """Execute the program using the given session handle.

    Args:
      sess: TF Session.
      threadpool: A ThreadPool on the executor for running async functions.

    Returns:
      done: Whether to end all execution.
    """
    raise NotImplementedError()

  def Shutdown(self):
    """Runs any necessary cleanup (potentially blocking)."""
    pass

  def SaveProgramState(self, sess=None, global_step=None):
    """Saves program state information that need to be loaded during restore."""
    pass

  def LoadProgramState(self, restored_checkpoint_path=None, sess=None):
    """Restore additional state before training starts.

    Args:
      restored_checkpoint_path: The path to the latest checkpoint that was
        restored. If None, this is a new run.
      sess: Optional session.
    """
    pass

  def _InstantiateTaskModel(self, task_params):
    """Instantiates a model object for a particular task.

    MultiTaskModels can accept a shared_model parameter, but SingleTaskModels
    cannot, so we handle them separately here.

    Args:
      task_params: An params instance that constructs either a SingleTaskModel
        or a MultiTaskSubModel.

    Returns:
      An instantiated object based on task_params.
    """
    if issubclass(task_params.cls, base_model.MultiTaskSubModel):
      return task_params.Instantiate(
          shared_model=self._shared_model, executor_ema=self._ema)
    return task_params.Instantiate(executor_ema=self._ema)

  def _OutfeedEnqueue(self, per_example_tensors):
    if not per_example_tensors:
      return tf.constant(0.0)
    per_example_tensors = py_utils.NestedMap(per_example_tensors)
    device = tpu.core(0) if self.spmd else ''
    with tf.device(device):
      return tpu_ops.outfeed_enqueue_tuple(per_example_tensors.Flatten())

  def GetModel(self):
    return self._model


class InputBenchmark(BaseProgram):
  """Measures input generation steps/sec depending on the params below."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('warmup_loops', 1,
             'How many loops to warmup before measuring elapsed time.')
    p.Define('measurement_loops', 5, 'How many loops to measure across.')
    return p

  def __init__(self, params, **kwargs):
    super().__init__(params, input_benchmark_only=True, **kwargs)
    self._program_name = 'InputBenchmark'

  def BuildTpuSubgraph(self):
    with py_utils.OpportunisticVariableReuseScope(True):
      self._model = self._InstantiateTaskModel(self._task_params)
    self._task = self._model.GetTask()
    self._task.input.CreateTpuEnqueueOps(benchmark_only=True)

  def Run(self, sess=None):
    p = self.params
    # Input benchmark doesn't work with eager yet.
    assert not py_utils.IsEagerMode()

    for _ in range(p.warmup_loops):
      self._InfeedLoop(sess)

    start_time = time.time()
    for _ in range(p.measurement_loops):
      self._InfeedLoop(sess)
    elapsed_secs = time.time() - start_time

    steps_per_sec = p.measurement_loops * self._steps_per_loop / elapsed_secs
    tf.logging.info('Input benchmark: steps/sec %f', steps_per_sec)
    return True


class TrainProgram(BaseProgram):
  """TrainProgram trains a single task and handles checkpoints."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'summary_interval_steps', None,
        'By default, we write summaries after each program execution. '
        'If this param is set, we write roughly every '
        '`summary_interval_steps`.')
    return p

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self._step_rate_tracker = summary_utils.StepRateTracker()
    self._program_name = 'TrainProgram'
    p = self.params
    self._summary_interval_steps = p.summary_interval_steps
    self._next_summary_step = None
    if (p.ml_perf is not None and p.ml_perf.benchmark_name is not None and
        p.ml_perf.steps_per_epoch is not None):
      self._ml_perf = p.ml_perf
    else:
      self._ml_perf = None

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
      return tf.constant(0.0)

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
        num_cores_per_replica = 1 if self.spmd else (
            device_assignment.num_cores_per_replica)
        for core in range(num_cores_per_replica):
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

  def TpuTrainStep(self, *args):
    """Train a shard of a batch on a single TPU core.

    Args:
      *args: metrics values from previous steps.

    Returns:
      New summed metrics values and a train_op.
    """
    with tf.name_scope('tpu_train'):
      with py_utils.OpportunisticVariableReuseScope(True):
        with contextlib.ExitStack() as stack:
          if py_utils.IsEagerMode():
            stack.enter_context(py_utils.GradientTape(persistent=True))
          self._model.ConstructFPropBPropGraph()
      per_step_eval_metrics = self._eval_metrics.SetMetrics(
          self._task.eval_metrics, args)
      outfeed_op = self._OutfeedEnqueue(self._task.per_example_tensors)
      summed_metrics = []
      assert len(per_step_eval_metrics) == len(args)
      with tf.control_dependencies([outfeed_op]):
        for x, y in zip(per_step_eval_metrics, args):
          summed_metrics.append(x + y)
      return summed_metrics + [self._task.train_op]

  def InfeedTFFunc(self):
    """Infeed function. Only needed in tf.function."""
    self._task.input.DeviceLoopSetupEager()

    def InfeedBody(i):
      self._task.input.CreateTpuEnqueueOps()
      return i + 1

    tf.while_loop(
        cond=lambda i: i < self._steps_per_loop,
        body=InfeedBody,
        loop_vars=[tf.constant(0)])

  def BuildTpuSubgraph(self):
    tf.logging.info('TrainProgram BuildTpuSubGraph')
    p = self.params
    self.spmd = (
        self.params.spmd or
        self._task_params.input.use_partitioned_infeed_queue)

    self._eval_metrics = metrics.TpuEvalMetrics(max_metrics=p.max_metrics)

    with py_utils.OpportunisticVariableReuseScope(True):
      self._model = self._InstantiateTaskModel(self._task_params)
    self._task = self._model.GetTask()
    # In Graph mode `InfeedSetupGraph` is called once to build the infeed ops.
    # In tf.function the relevant methods will be called in `InfeedTFFunc`.
    if not py_utils.IsEagerMode():
      self._task.input.InfeedSetupGraph()

    @tpu_function.on_device_training_loop
    def TpuTrainLoop():
      loop_result = tpu_training_loop.repeat(
          self._steps_per_loop,
          self.TpuTrainStep,
          inputs=self._eval_metrics.initial_values,
          name='train_loop')
      # Final metrics are the avg across self._steps_per_loop steps.
      return self._eval_metrics.FinalizeMetrics(loop_result)

    def TrainFunc():
      # TODO(laigd): investigate how to run compilation only to catch errors
      # earlier.
      self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
          TpuTrainLoop,
          num_shards=self.data_parallelism,
          device_assignment=py_utils.GetTpuDeviceAssignment())
      outfeed = self._OutfeedDequeueLoop(self._task.per_example_tensors,
                                         self._steps_per_loop,
                                         self.data_parallelism)

      def _ConstructPostTrainingLoop(metric_values, outfeed):
        """Returns the op for tpu training with tail cpu computation."""
        # Adds a tail computation that is run after the tpu_training loop
        # step finishes. This allows us to run certain computation that
        # acts on the variable between tpu_train_loop iterations and
        # amortizing the cost of the operations. Alternative of running
        # tpu.outside_compilation & using tf.cond is expenseive.
        with tf.control_dependencies(metric_values):
          self._model.ConstructPostTrainingLoop(outfeed)
          with tf.control_dependencies([self._task.post_training_loop_op]):
            return [[tf.identity(o) for o in metric_values], outfeed]

      # Get metric result from a single replica; they are all same here
      # because TpuEvalMetrics.FinalizeMetrics runs a cross_replica_sum.
      metric_values = [t[0] for t in batch_parallel_res]
      return _ConstructPostTrainingLoop(metric_values, outfeed)

    if py_utils.IsEagerMode():
      with self._summary_writer.as_default():
        self.infeed_fn = tf.function(autograph=False)(
            self.InfeedTFFunc).get_concrete_function()
        self.tpu_outs = (
            tf.function(autograph=False)(TrainFunc).get_concrete_function())
    else:
      self.tpu_outs = TrainFunc()

    # Write model analysis.
    self._model_analysis, self._total_num_params = summary_utils.ModelAnalysis(
        self._model)
    tf.logging.info('Total params=%d', self._total_num_params)
    try:
      with tf.io.gfile.GFile(
          os.path.join(self._program_dir, 'model_analysis.txt'), 'w') as f:
        f.write(self._model_analysis)
    except tf.errors.NotFoundError as e:
      tf.logging.info('Failed to write model analysis %s', e)

  def _ShouldWriteSummary(self, global_step):
    if not self._summary_interval_steps:
      return True
    if not self._next_summary_step:
      self._next_summary_step = global_step
    if global_step >= self._next_summary_step:
      self._next_summary_step = global_step + self._summary_interval_steps
      return True
    else:
      return False

  def Run(self, sess=None):
    # Prevent overtraining.
    if py_utils.IsEagerMode():
      task_global_step = self._task.global_step.numpy()
    else:
      task_global_step = sess.run(self._task.global_step)
    if self._ShouldStop(task_global_step):
      return True

    if self._ml_perf:
      mlp_log.mlperf_print(
          'block_start',
          None,
          metadata={
              'epoch_count': 1,
              'first_epoch_num': 1
          })

    if py_utils.IsEagerMode():
      async_executor = executor.new_executor(enable_async=True)
      with context.executor_scope(async_executor):
        self.infeed_fn()

      values, outfeeds = self.tpu_outs()
      # Ensure that the infeed ops are finished
      # This is necessary to ensure that any state in the infeed ops is
      # synchronized before the next device loop. Otherwise we might see that
      # a device loop still using the same data batches in the last device loop.
      async_executor.wait()

      values = py_utils.Transform(lambda x: x.numpy(), values)
      outfeeds = py_utils.Transform(lambda x: x.numpy(), outfeeds)
    else:
      infeed_future = self._infeed_pool.apply_async(
          self._InfeedLoop, args=(sess,))
      values, outfeeds = sess.run(self.tpu_outs)
      infeed_future.get()

    self._eval_metrics.PackMetricsValues(values)
    eval_metrics = self._eval_metrics.metrics

    if py_utils.IsEagerMode():
      global_step = self._model.global_step.numpy()
    else:
      global_step = sess.run(self._model.global_step)

    if self._ShouldWriteSummary(global_step):
      step_rate, example_rate, total_examples = (
          self._step_rate_tracker.ComputeStepRate(
              global_step,
              eval_metrics['num_samples_in_batch'][0] * self._steps_per_loop))
      self._SummarizeValue(global_step, 'global_step/sec', step_rate)
      self._SummarizeValue(global_step, 'examples/sec', example_rate)
      self._SummarizeValue(global_step, 'total_samples', total_examples)
      self._SummarizeValue(global_step, 'total_num_params',
                           self._total_num_params)
      status_strs = []
      for key, (val, _) in sorted(eval_metrics.items()):
        self._SummarizeValue(global_step, key, val)
        tf.logging.info((global_step, key, val))
        status_strs.append('%s=%s' % (key, val))
      self.SetStatusMessage('Executing train program at step %d %s' %
                            (global_step, ','.join(status_strs)))

      if py_utils.IsEagerMode():
        task_global_step = self._task.global_step.numpy()
        # TODO(laigd): Not all `ProcessFPropResults` work in Eager.
        if py_utils.RunProcessFPropResultsInEager():
          summaries = self._task.ProcessFPropResults(None, task_global_step,
                                                     eval_metrics, outfeeds)
      else:
        task_global_step = sess.run(self._task.global_step)
        summaries = self._task.ProcessFPropResults(sess, task_global_step,
                                                   eval_metrics, outfeeds)
        self._WriteSummaries(
            os.path.basename(self._program_dir), global_step, summaries)
      self._summary_writer.flush()

    if self._ml_perf:
      mlp_log.mlperf_print(
          'block_stop', None, metadata={
              'epoch_num': 1,
              'first_epoch_num': 1
          })

    vizier_early_stop = self._ReportVizierMetrics(
        global_step, self._eval_metrics.ToAverageMetrics())
    return self._ShouldStop(task_global_step) or vizier_early_stop

  def _ShouldStop(self, task_global_step):
    """Simpler version of _ShouldStop without early stopping."""
    if task_global_step >= self._task_params.train.max_steps:
      tf.logging.info('ShouldStop: step:%6d params.train.max_steps:%6d',
                      task_global_step, self._task_params.train.max_steps)
      return True

    return False


class EvalProgram(BaseProgram):
  """Evaluation program."""

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self._program_name = 'EvalProgram'
    p = self.params
    if (p.ml_perf is not None and p.ml_perf.benchmark_name is not None and
        p.ml_perf.steps_per_epoch is not None):
      self._ml_perf = p.ml_perf
      self._run_stop = None
    else:
      self._ml_perf = None

  def TpuEvalStep(self, *args):
    """Eval a shard of a batch on a single TPU core.

    Args:
      *args: metrics values from previous steps.

    Returns:
      Summed eval metrics.
    """
    with tf.name_scope('tpu_eval'):
      # Applies EMA if applicable to support running only eval/decode programs.
      self._model.ConstructFPropGraph(apply_ema=True)
      per_step_eval_metrics = self._eval_metrics.SetMetrics(
          self._task.eval_metrics, args)
      summed_metrics = []
      for x, y in zip(per_step_eval_metrics, args):
        summed_metrics.append(x + y)
      return summed_metrics

  def InfeedTFFunc(self):
    """Infeed function. Only needed in tf.function."""
    self._task.input.DeviceLoopSetupEager()

    def InfeedBody(i):
      self._task.input.CreateTpuEnqueueOps()
      return i + 1

    tf.while_loop(
        cond=lambda i: i < self._steps_per_loop,
        body=InfeedBody,
        loop_vars=[tf.constant(0)])

  def EvalFunc(self):
    """Eval function."""

    @tpu_function.on_device_training_loop
    def TpuEvalLoop():
      loop_result = tpu_training_loop.repeat(
          self._steps_per_loop,
          self.TpuEvalStep,
          inputs=self._eval_metrics.initial_values,
          name='eval_loop')
      # Final metrics are the avg across self._steps_per_loop steps.
      return self._eval_metrics.FinalizeMetrics(loop_result)

    # TODO(laigd): investigate how to run compilation only to catch errors
    # earlier.
    self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
        TpuEvalLoop,
        num_shards=self.data_parallelism,
        device_assignment=py_utils.GetTpuDeviceAssignment())

    # Get metric result from a single replica; they are all same here
    # because TpuEvalMetrics.FinalizeMetrics runs a cross_replica_sum.
    return [t[0] for t in batch_parallel_res]

  def BuildTpuSubgraph(self):
    tf.logging.info(f'EvalProgram {self.params.dataset_name} BuildTpuSubGraph')
    p = self.params
    with cluster_factory.SetEval(True):
      self._eval_metrics = metrics.TpuEvalMetrics(max_metrics=p.max_metrics)
      with py_utils.OpportunisticVariableReuseScope(True):
        self._model = self._InstantiateTaskModel(self._task_params)
      self._task = self._model.GetTask()
      # In Graph mode `InfeedSetupGraph` is called once to build the infeed ops.
      # In tf.function the relevant methods will be called in `InfeedTFFunc`.
      if not py_utils.IsEagerMode():
        self._task.input.InfeedSetupGraph()

      if py_utils.IsEagerMode():
        with self._summary_writer.as_default():
          self.infeed_fn = tf.function(autograph=False)(
              self.InfeedTFFunc).get_concrete_function()
          self.tpu_outs = (
              tf.function(autograph=False)(
                  self.EvalFunc).get_concrete_function())
      else:
        self.tpu_outs = self.EvalFunc()

  def Run(self, sess=None):
    if py_utils.IsEagerMode():
      global_step = self._model.global_step.numpy()
    else:
      global_step = sess.run(self._model.global_step)

    mlperf_epoch_num = None
    if self._ml_perf:
      mlperf_epoch_num = int(global_step / self._ml_perf.steps_per_epoch)
      mlp_log.mlperf_print(
          'eval_start', None, metadata={'epoch_num': mlperf_epoch_num})
    begin_time = time.time()
    if self._task.input.params.resettable:
      tf.logging.info('Resetting input_generator.')
      self._task.input.Reset(sess)
      if py_utils.IsEagerMode():
        with self._summary_writer.as_default():
          # In eager mode, after resetting the input generator, we need to
          # re-trace the infeed tf.function to ensure it uses the new iterator.
          self.infeed_fn = tf.function(autograph=False)(
              self.InfeedTFFunc).get_concrete_function()

    if py_utils.IsEagerMode():
      async_executor = executor.new_executor(enable_async=True)
      with context.executor_scope(async_executor):
        self.infeed_fn()

      values = self.tpu_outs()
      values = py_utils.Transform(lambda x: x.numpy(), values)
      # Ensure that the infeed ops are finished
      # This is necessary to ensure that any state in the infeed ops is
      # synchronized before the next device loop. Otherwise we might see that
      # a device loop still using the same data batches in the last device loop.
      async_executor.wait()
    else:
      infeed_future = self._infeed_pool.apply_async(
          self._InfeedLoop, args=(sess,))
      values = sess.run(self.tpu_outs)
      infeed_future.get()

    status_strs = []
    self._eval_metrics.PackMetricsValues(values)
    # Make a copy to avoid changing self._eval_metrics.metrics, which needs to
    # be fixed in order to run PackMetricsValues() correctly.
    eval_metrics = self._eval_metrics.metrics.copy()

    elapsed_secs = time.time() - begin_time
    # TODO(yanqiz): Here we assume a per replica example rate.
    example_rate = float(eval_metrics['num_samples_in_batch'][0] *
                         self._steps_per_loop / elapsed_secs)
    tf.logging.info((global_step, 'example_rate', example_rate))
    if 'example_rate' not in eval_metrics:
      eval_metrics['example_rate'] = (example_rate, 1.0)

    for key, (val, _) in sorted(eval_metrics.items()):
      self._SummarizeValue(global_step, key, val)
      tf.logging.info((global_step, key, val))
      status_strs.append('%s=%s' % (key, val))

    mlperf_done = False
    if self._ml_perf:
      mlperf_metric = self._ml_perf.decoder_metric_name
      if (mlperf_metric
          in eval_metrics) and (self._ml_perf.decoder_metric_success_threshold
                                is not None):
        mlperf_metric_value = eval_metrics[mlperf_metric][0]
        mlp_log.mlperf_print(
            'eval_accuracy',
            mlperf_metric_value,
            metadata={'epoch_num': mlperf_epoch_num})

        mlp_log.mlperf_print(
            'eval_stop', None, metadata={'epoch_num': mlperf_epoch_num})
        # Successful ML Perf run if we exceed target accuracy
        if mlperf_metric_value > self._ml_perf.decoder_metric_success_threshold:
          tf.logging.info('ml_perf_final_threshold: %f exceeded',
                          self._ml_perf.decoder_metric_success_threshold)
          if not self._run_stop:
            self._run_stop = mlp_log.mlperf_print(
                'run_stop', None, metadata={'status': 'success'})
            mlperf_done = True

        # Failed ML Perf run if we fail to reach target accuracy after
        # predefined number of steps.
        elif global_step >= self._ml_perf.max_steps_to_train:
          if not self._run_stop:
            self._run_stop = mlp_log.mlperf_print(
                'run_stop', None, metadata={'status': 'abort'})
            mlperf_done = True

    self.SetStatusMessage(
        f'Executing eval program on dataset {self.params.dataset_name} '
        f"at step {global_step}\n{','.join(status_strs)}")

    self._summary_writer.flush()

    if self._ml_perf:
      return mlperf_done
    else:
      average_metrics = self._eval_metrics.ToAverageMetrics()
      if 'example_rate' not in average_metrics:
        average_metrics['example_rate'] = (
            metrics.TpuEvalMetrics.ToAverageMetric(example_rate))
      return self._ReportVizierMetrics(global_step, average_metrics)


def _UpdateCpuPassThroughData(decode_out_dict, cpu_pt):
  """Combine cpu_pt into decode_out_dict."""
  if cpu_pt is None:
    cpu_pt = {}
  elif py_utils.IsEagerMode():
    cpu_pt = py_utils.Transform(lambda x: x.numpy(), cpu_pt)
  common_keys = decode_out_dict.keys() & cpu_pt.keys()
  if common_keys:
    raise ValueError('CPU passthrough keys already present in '
                     f'decode_out_dict keys: {common_keys}')

  decode_out_dict.update(cpu_pt)
  return decode_out_dict


def _FetchDecodeOut(tpu_outs, sess=None):
  """Fetch decoder outputs, combining with CPU passthrough tensors if needed.

  Args:
    tpu_outs: A list of decoded tensors and list of cpu passthrough tensors in
      graph mode, or a callable returning such in eager mode.
    sess: A session to use in graph mode.

  Returns:
    A dict containing merged decoded outputs.
  """
  if py_utils.IsEagerMode():
    # The CPU pass through data will be from the infeed function.
    # Here we will get an empty `cpu_pt`.
    decode_out_dict, cpu_pt = tpu_outs()
    decode_out_dict = py_utils.Transform(lambda x: x.numpy(), decode_out_dict)
  else:
    decode_tensors, cpu_passthrough_tensors = tpu_outs
    if cpu_passthrough_tensors is not None:
      decode_out_dict, cpu_pt = sess.run(
          [decode_tensors, cpu_passthrough_tensors])
    else:
      decode_out_dict = sess.run(decode_tensors)
      cpu_pt = {}
  return _UpdateCpuPassThroughData(decode_out_dict, cpu_pt)


class DecodeProgram(BaseProgram):
  """DecodeProgram."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('decode_until_out_of_range', False,
             ('If set, ignores steps_per_loop and Decode proceeds until an '
              'OutOfRangeError is triggered by hitting the end of dataset.'
              'DEPRECATED, pls set steps_per_loop to -1 instead.'))
    p.Define(
        'postprocess_all_at_once', False,
        'If set, decode_out_dict of all steps are accumulated into a list '
        'and passed to PostProcess to run only once at the end. Note that'
        ' the PostProcess of the Task should define logic for aggregating'
        'data from the list of decode_out_dict.')
    p.Define(
        'trigger_offset', 0, 'The program is only effectively triggered after '
        'this num of runs. The default is 0 where no runs is skipped. '
        'It is used with trigger_interval below to control trigger schedule.')
    p.Define(
        'trigger_interval', 1, 'The program is only effectively triggered '
        'every this num of runs, after trigger_offset is met.')
    return p

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self._program_name = 'DecodeProgram'
    self._decode_out_dict_lst = []
    self._ema_applied = False
    self._dataset_summaries = {}
    # TODO(xingwu): fully deprecate decode_until_out_of_range
    if self.params.decode_until_out_of_range:
      self.params.steps_per_loop = -1
    self._trigger_scheduler = program_utils.TriggerScheduler(
        self.params.trigger_offset, self.params.trigger_interval)

  def _DatasetSummaryWriter(self, unused_dataset_name):
    """Returns the FileWriter object to use for summaries."""
    return self._summary_writer

  def _WriteSummaries(self, job_name, dataset_name, global_step, summaries):
    """Write summaries to be viewed by TensorBoard.

    Args:
      job_name: The name of this job ('trainer', 'evaler', etc.)
      dataset_name: The dataset name that's decoded.
      global_step: Integer number of trainer steps (not a tensor).
      summaries: Dict of {summary_name: tf.Summary()}.
    """
    if not summaries:
      return

    sliced_summaries = collections.defaultdict(dict)
    for name, summary in summaries.items():
      arr = name.split(summary_utils.METRIC_SLICE_SEPARATOR)
      if len(arr) == 2:
        metric, slice_name = arr
        summary.value[0].tag = metric
        sliced_summaries[slice_name][metric] = summary
      else:
        metric = arr[0]
        sliced_summaries[''][metric] = summary
    with contextlib.ExitStack() as stack:
      if py_utils.IsEagerMode():
        stack.enter_context(
            self._DatasetSummaryWriter(dataset_name).as_default())
      for slice_name, summaries in sliced_summaries.items():
        if slice_name:
          slice_writer = tf.summary.FileWriter(
              os.path.join(self._program_dir, slice_name))
        else:
          slice_writer = self._DatasetSummaryWriter(dataset_name)
        for unused_name, summary in sorted(summaries.items()):
          if py_utils.IsEagerMode():
            _RewriteV1SummaryInTF2(summary, global_step)
          else:
            slice_writer.add_summary(summary, global_step)
          if summary.value:
            for value in summary.value:
              if value.HasField('simple_value'):
                tf.logging.info('%s@%s summary on checkpoint@%d %s:%s = %.8g',
                                job_name, dataset_name, global_step, slice_name,
                                value.tag, value.simple_value)
        slice_writer.flush()

  def _WriteInputDataStats(self, sess=None, **kwargs):
    """Write input data stats for model training as TF summaries.

    Args:
      sess: The Tensorflow session.
      **kwargs: dict of extra args, can include,
        - dataset_name: the dataset name to run infeed.
        - inp_instance: the input instance that will run infeed op.
    """
    if 'dataset_name' not in kwargs or 'inp_instance' not in kwargs:
      raise ValueError('dataset_name and inp_instance must be set for '
                       'DecodeProgram._WriteInputDataStats.')
    dataset_name = kwargs['dataset_name']
    inp_instance = kwargs['inp_instance']
    if (inp_instance.merged_input_data_summary_op is None or
        not self._write_train_input_stats):
      return
    global_step = sess.run(self._model.global_step)
    if (global_step %
        inp_instance.params.input_stats_summary_interval_steps == 0):
      summary_str = sess.run(inp_instance.merged_input_data_summary_op)
      self._DatasetSummaryWriter(dataset_name).add_summary(
          summary_str, global_step)
      self._DatasetSummaryWriter(dataset_name).flush()

  def FinalizeCallback(self, unused_finalize_ret):
    """Callback after _FinalizeDecode thread done."""
    tf.logging.info('DecodeProgram skip FinalizeCallback.')

  def Summary(self):
    return self._dataset_summaries

  def InfeedTFFunc(self, inp_instance):
    """Infeed function. Only needed in tf.function."""
    inp_instance.DeviceLoopSetupEager()
    inp_instance.CreateTpuEnqueueOps()
    # `CreateTpuEnqueueOps` and `CreateCpuPassthroughEnqueueOps` must be in the
    # same place, because the former enqueues `_per_host_passthrough_batches`,
    # while the latter consumes it.
    inp_instance.CreateCpuPassthroughEnqueueOps()
    # `CreateCpuPassthroughEnqueueOps` and `DequeueCpuPassthrough` must be in
    # the same place, because the former enqueues `_host_queues`,
    # while the latter consumes it.
    cpu_pt = inp_instance.DequeueCpuPassthrough()
    return cpu_pt

  def DecodeFunc(self, inp_instance):
    """Wrap the DecodeFn with split_compile_and_shard."""

    def _DecodeFn():
      """Decode call to be compiled for TPU."""
      # Applies EMA if applicable to support running only eval/decode programs.
      _, decode_dict = self._model.ConstructDecodeGraph(
          apply_ema=(not self._ema_applied),
          input_batch=inp_instance.TpuDequeueBatch())
      self._ema_applied = True
      self.decode_nm = py_utils.NestedMap(decode_dict)
      return self.decode_nm.Flatten()

    self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
        _DecodeFn,
        num_shards=self.data_parallelism,
        device_assignment=py_utils.GetTpuDeviceAssignment())

    if self.decode_nm:
      decode_tensors = self.decode_nm.Pack(batch_parallel_res)
    else:
      decode_tensors = py_utils.NestedMap()
    if py_utils.IsEagerMode():
      # The CPU pass through data will be from the infeed function.
      cpu_pt = {}
    else:
      cpu_pt = inp_instance.DequeueCpuPassthrough()
    return decode_tensors, cpu_pt

  def _DecodeUntilOutOfRangeInfeedLoop(self,
                                       dataset_name,
                                       inp_instance,
                                       sess=None,
                                       infeed_step_queue=None):
    """Infeed loop that stops when it runs out of data (OutOfRange error)."""
    tf.logging.info(f'_InfeedLoop start {self._program_name} '
                    f'on dataset {dataset_name}')

    def _HandleEndOfData():
      tf.logging.info(f'End of dataset {dataset_name}.')
      infeed_step_queue.put(-1)  # -1 signals reaching end of dataset.
      self._WriteInputDataStats(
          sess, dataset_name=dataset_name, inp_instance=inp_instance)
      tf.logging.info('_InfeedLoop done')

    try:
      loop_index = 0
      while True:
        tf.logging.vlog(1, '_InfeedLoop %d', loop_index)
        sess.run(inp_instance.tpu_infeed_op)
        infeed_step_queue.put(loop_index)
        loop_index += 1
    except tf.errors.OutOfRangeError:
      _HandleEndOfData()
    except tf.errors.InvalidArgumentError as e:
      if 'REPEAT_SENTINEL_' in e.message:
        # Sentinel in repeating dataset signaling end of one epoch.
        tf.logging.info('Detected end-of-data sentinel.')
        _HandleEndOfData()
      else:
        tf.logging.info('_InfeedLoop InvalidArgumentError %r: %s', e, e.message)
        raise
    except Exception as e:
      tf.logging.info('_InfeedLoop exception %r: %s', e, e.message)
      raise

  def BuildTpuSubgraph(self):
    tf.logging.info(
        f'DecodeProgram {self.params.dataset_name} BuildTpuSubGraph')
    with cluster_factory.SetEval(True):
      py_utils.ResetStepSeed()
      with py_utils.OpportunisticVariableReuseScope(True):
        self._model = self._InstantiateTaskModel(self._task_params)
      self._task = self._model.GetTask()
      # In Graph mode `InfeedSetupGraph` is called once to build the infeed ops.
      # In tf.function the relevant methods will be called in `InfeedTFFunc`.
      if not py_utils.IsEagerMode():
        self._task.input.InfeedSetupGraph(cpu_passthrough=True)

      if py_utils.IsEagerMode():
        with self._summary_writer.as_default():
          self.infeed_fn = tf.function(autograph=False)(functools.partial(
              self.InfeedTFFunc, self._task.input)).get_concrete_function()
          self.tpu_outs = (
              tf.function(autograph=False)(functools.partial(
                  self.DecodeFunc, self._task.input)).get_concrete_function())
      else:
        self.tpu_outs = self.DecodeFunc(self._task.input)

  def _DecodeStep(self,
                  sess,
                  step,
                  dec_metrics,
                  global_step,
                  buffered_decode_out,
                  postprocess_futures,
                  dataset_name,
                  threadpool=None):
    """Run one iteration of decode."""
    tf.logging.info(f'Decoding step {step}')
    fetch_start = time.time()
    if py_utils.IsEagerMode():
      async_executor = executor.new_executor(enable_async=True)
      with context.executor_scope(async_executor):
        cpu_pt = self.infeed_fn()

    if isinstance(self.tpu_outs, dict):
      tpu_out = self.tpu_outs[dataset_name]
    else:
      tpu_out = self.tpu_outs
    decode_out_dict = _FetchDecodeOut(tpu_out, sess)
    if py_utils.IsEagerMode():
      # Ensure that the infeed ops are finished
      # This is necessary to ensure that any state in the infeed ops is
      # synchronized before the next device loop. Otherwise we might see that
      # a device loop still using the same data batches in the last device loop.
      async_executor.wait()
      decode_out_dict = _UpdateCpuPassThroughData(decode_out_dict, cpu_pt)

    tf.logging.info(f'Finished TPU decoding on step {step}')
    dec_metrics['decode_secs'].Update(time.time() - fetch_start)
    if self.params.postprocess_all_at_once:
      # Accumulate decode_out_dicts and skip postprocess until the end.
      self._decode_out_dict_lst.append(decode_out_dict)
    else:
      self._RunPostProcess(threadpool, step, decode_out_dict, dec_metrics,
                           global_step, buffered_decode_out,
                           postprocess_futures)

  def _RunPostProcess(self, threadpool, step, decode_out_obj, dec_metrics,
                      global_step, buffered_decode_out, postprocess_futures):
    """Run postprocess in sync or async if a threadpool is provided."""
    if threadpool:
      # Run postprocess on separate CPU thread.
      postprocess_futures.append(
          threadpool.apply_async(
              self._PostProcessStep,
              args=(step, decode_out_obj, dec_metrics, global_step,
                    buffered_decode_out)))
    else:
      self._PostProcessStep(step, decode_out_obj, dec_metrics, global_step,
                            buffered_decode_out)

  def _PostProcessStep(self, idx, decode_out_obj, dec_metrics, global_step,
                       buffered_decode_out):
    """Run postprocess for a single decode step."""
    tf.logging.info(f'PostProcessStep {idx}')
    post_process_start = time.time()
    decode_out = self._task.PostProcessDecodeOut(decode_out_obj, dec_metrics)
    dec_metrics['postprocess_secs'].Update(time.time() - post_process_start)
    tf.logging.info('PostProcessed step: %d %f' %
                    (idx, dec_metrics['num_samples_in_batch'].total_value))
    if decode_out:
      if isinstance(decode_out, dict):
        decode_out = decode_out.items()

      if idx == 0:
        # Add summaries only for the first batch of data.
        for key, value in decode_out:
          if isinstance(value, tf.Summary):
            tags = f'{[x.tag for x in value.value]}'
            tf.logging.info(f'Adding summary {key} with tags {tags}.')
            if py_utils.IsEagerMode():
              # TODO(b/241113869): Logging summary in eager mode
              pass
            else:
              self._summary_writer.add_summary(value, global_step)
              self._summary_writer.flush()

      buffered_decode_out.extend(
          kv for kv in decode_out if not isinstance(kv[1], tf.Summary))

  def _FinalizeDecode(self,
                      dataset_name,
                      dec_metrics,
                      start_time,
                      global_step,
                      buffered_decode_out,
                      futures=None):
    """Finalize and summarize the results of this Decode program run."""
    if futures:
      # Wait for all async postprocessing jobs to finish.
      for future in futures:
        future.get()
    elapsed_secs = time.time() - start_time
    # TODO(xingwu): simplify summaries format.
    summaries = {k: v.Summary(k) for k, v in dec_metrics.items()}
    for k, v in dec_metrics.items():
      if k.startswith('num_samples_in_batch'):
        cumulative_key = 'cumulative_num_examples' + k.removeprefix(
            'num_samples_in_batch')
        summaries[cumulative_key] = tf.Summary(value=[
            tf.Summary.Value(tag=cumulative_key, simple_value=v.total_value)
        ])
        example_rate = v.total_value / elapsed_secs
        speed_key = 'examples/sec' + k.removeprefix('num_samples_in_batch')
        summaries[speed_key] = tf.Summary(
            value=[tf.Summary.Value(tag=speed_key, simple_value=example_rate)])

    self._WriteSummaries(
        os.path.basename(self._program_dir), dataset_name, global_step,
        summaries)
    decode_out_path = os.path.join(self._program_dir,
                                   'decoder_out_%09d' % global_step)
    decode_finalize_args = base_model.DecodeFinalizeArgs(
        decode_out_path=decode_out_path, decode_out=buffered_decode_out)
    self._task.DecodeFinalize(decode_finalize_args)

    # Result is not returned as a signal for "done", unlike for training.
    self._ReportVizierMetrics(global_step, dec_metrics)
    self._dataset_summaries[dataset_name] = summaries
    return dataset_name, summaries

  def RunForInput(self, dataset_name, inp_instance, sess=None, threadpool=None):
    """Setup and execute Decode program."""
    if py_utils.IsEagerMode():
      global_step = self._model.global_step.numpy()
    else:
      global_step = sess.run(self._model.global_step)
    self.SetStatusMessage(f'Executing decode program on dataset {dataset_name} '
                          f'at step {global_step}')

    if inp_instance.params.resettable:
      tf.logging.info('Resetting input_generator.')
      inp_instance.Reset(sess)
      if py_utils.IsEagerMode():
        with self._DatasetSummaryWriter(dataset_name).as_default():
          # In eager mode, after resetting the input generator, we need to
          # re-trace the infeed tf.function to ensure it uses the new iterator.
          self.infeed_fn = tf.function(autograph=False)(functools.partial(
              self.InfeedTFFunc, inp_instance)).get_concrete_function()

    # The infeed_step_queue synchronizes the _InfeedLoop with the Decoding loop
    # (that runs _DecodeStep). As an input batch is successfully fed through
    # the _InfeedLoop, a non-negative counter value is added to the queue.
    # _DecodeStep waits and only runs if it can successfully remove an item
    # from the queue (i.e. there is available data). If End of Dataset is
    # reached (OutOfRangeError), _InfeedLoop inserts a special value of "-1",
    # which will terminate the Decode loop once it's consumed from the queue.
    if self.params.steps_per_loop < 0:
      if py_utils.IsEagerMode():
        raise NotImplementedError(
            'p.steps_per_loop < 0 is not supported in eager mode.')
      infeed_step_queue = queue.Queue()
      infeed_future = self._infeed_pool.apply_async(
          self._DecodeUntilOutOfRangeInfeedLoop,
          args=(
              dataset_name,
              inp_instance,
              sess,
              infeed_step_queue,
          ))
    else:
      if not py_utils.IsEagerMode():
        infeed_future = self._infeed_pool.apply_async(
            self._InfeedLoopForInput, args=(
                dataset_name,
                inp_instance,
                sess,
            ))

    dec_metrics = self._task.CreateDecoderMetrics()
    if not dec_metrics:
      tf.logging.info('Empty decoder metrics')
      return

    dec_metrics.update({
        'decode_secs': metrics.AverageMetric(),
        'postprocess_secs': metrics.AverageMetric(),
    })

    buffered_decode_out = []
    postprocess_futures = []

    start_time = time.time()
    if self.params.steps_per_loop < 0:
      while True:
        step = infeed_step_queue.get()  # Blocks until an item is returned.
        if step == -1:
          tf.logging.info('Reached end of dataset. Stop decoding.')
          break
        infeed_step_queue.task_done()
        self._DecodeStep(sess, step, dec_metrics, global_step,
                         buffered_decode_out, postprocess_futures, dataset_name,
                         threadpool)
    else:
      for step in range(self._steps_per_loop):
        tf.logging.info('Starting step %d of %d', step, self._steps_per_loop)
        self._DecodeStep(sess, step, dec_metrics, global_step,
                         buffered_decode_out, postprocess_futures, dataset_name,
                         threadpool)
    # Run postprocess after the last step if postprocess_all_at_once.
    if self.params.postprocess_all_at_once and self._decode_out_dict_lst:
      self._RunPostProcess(threadpool, step, self._decode_out_dict_lst,
                           dec_metrics, global_step, buffered_decode_out,
                           postprocess_futures)
    if not py_utils.IsEagerMode():
      infeed_future.get()

    if threadpool:

      def _HandleError(e):
        tf.logging.exception(e)
        # Terminate the main thread.
        _thread.interrupt_main()

      # Async. TPU+host processing is done and can move on to Train.
      return threadpool.apply_async(
          self._FinalizeDecode,
          args=(
              dataset_name,
              dec_metrics,
              start_time,
              global_step,
              buffered_decode_out,
              postprocess_futures,
          ),
          callback=self.FinalizeCallback,
          error_callback=_HandleError)
    else:
      finalize_ret = self._FinalizeDecode(dataset_name, dec_metrics, start_time,
                                          global_step, buffered_decode_out)
      self.FinalizeCallback(finalize_ret)
      return None

  def Run(self, sess=None, threadpool=None):
    self._trigger_scheduler.Trigger()
    if not self._trigger_scheduler.ShouldRun():
      return
    return self.RunForInput(self.params.dataset_name, self._task.input, sess,
                            threadpool)


class MultiInputsDecodeProgram(DecodeProgram):
  """Decode program with multiple inputs."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Delete('dataset_name')
    p.Define('dataset_names', [], 'List of datasets to decode.')
    p.Define('input_params', [], 'List of input params map to the datasets')
    return p

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)  # _SetProgramDir called here.
    self._program_name = 'MultiInputsDecodeProgram'
    self._inputs = []
    self.tpu_outs = {}
    self._summary_writer_objs = {}  # one writer per dataset

  def _SetProgramDir(self):
    """Set program dir for output."""
    p = self.params
    # Program dirs are where the summaries are written to.
    if p.task_name:
      program_dir_name = p.task_name + '_' + p.name + '_multi_inputs'
    else:
      program_dir_name = p.name + '_multi_inputs'
    self._program_dir = os.path.join(self._logdir, program_dir_name)
    tf.io.gfile.makedirs(self._program_dir)
    with tf.io.gfile.GFile(os.path.join(self._program_dir, 'params.txt'),
                           'w') as f:
      f.write(p.ToText())

    self.status_cache = program_utils.DecodeStatusCache(self._program_dir)

  def _DatasetSummaryWriter(self, dataset_name):
    if not self._summary_writer_objs:
      self._summary_writer_objs = {}
      for ds_name in self.params.dataset_names:
        ds_dir = os.path.join(self._program_dir, ds_name)
        tf.io.gfile.makedirs(ds_dir)
        if py_utils.IsEagerMode():
          file_writer = tf.compat.v2.summary.create_file_writer(ds_dir)
        else:
          file_writer = tf.summary.FileWriter(ds_dir)
        self._summary_writer_objs[ds_name] = file_writer
    return self._summary_writer_objs[dataset_name]

  def FinalizeCallback(self, finalize_ret):
    dataset_name, summaries = finalize_ret
    self.status_cache.UpdateDataset(dataset_name, summaries)

  def InitInputs(self, sess=None):
    self.SetStatusMessage('Init inputs %s' % self._program_name)
    for inp_instance in self._inputs:
      inp_instance.Initialize(sess)
    self.SetStatusMessage('Init inputs %s done.' % self._program_name)

  def BuildTpuSubgraph(self):
    tf.logging.info('MultiInputsDecodeProgram BuildTpuSubGraph')
    with cluster_factory.SetEval(True):
      py_utils.ResetStepSeed()
      with py_utils.OpportunisticVariableReuseScope(True):
        self._model = self._InstantiateTaskModel(self._task_params)
      self._task = self._model.GetTask()
      self._inputs = [p.Instantiate() for p in self.params.input_params]
      # In Graph mode `InfeedSetupGraph` is called once to build the infeed ops.
      # In tf.function the relevant methods will be called in `InfeedTFFunc`.
      if not py_utils.IsEagerMode():
        for i in range(len(self._inputs)):
          dataset_name = self.params.dataset_names[i]
          self._inputs[i].InfeedSetupGraph(cpu_passthrough=True)
          self.tpu_outs[dataset_name] = self.DecodeFunc(self._inputs[i])

      else:
        # TODO(xingwu): Verify for the eager mode.
        tf.logging.warning('Eager mode MultiInputsDecodeProgram is not ready!')
        for i in range(len(self._inputs)):
          dataset_name = self.params.dataset_names[i]
          with self._DatasetSummaryWriter(dataset_name).as_default():
            self.infeed_fn = tf.function(autograph=False)(
                self.InfeedTFFunc).get_concrete_function()
            self.tpu_outs[dataset_name] = (
                tf.function(autograph=False)(functools.partial(
                    self.DecodeFunc, self._inputs[i])).get_concrete_function())

  def Run(self, sess=None, threadpool=None):
    futures = []
    # TODO(xingwu): move global_step to attribute.
    if py_utils.IsEagerMode():
      global_step = self._model.global_step.numpy()
    else:
      global_step = sess.run(self._model.global_step)
    ckpt_key = f'ckpt-{global_step}'
    self.status_cache.UpdateCkpt(ckpt_key)
    for i in range(len(self.params.dataset_names)):
      dataset_name = self.params.dataset_names[i]
      summaries = self.status_cache.TryLoadCache(ckpt_key, dataset_name)
      if summaries:
        self._dataset_summaries[dataset_name] = summaries
        continue
      inp_instance = self._inputs[i]
      future = self.RunForInput(dataset_name, inp_instance, sess, threadpool)
      if future:
        futures.append(future)
    return futures


def _InferStepsPerLoop(num_samples, global_batch_size, drop_remainder):
  if drop_remainder:
    return num_samples // global_batch_size
  else:
    return -(-num_samples // global_batch_size)


class ExperimentalDecodeProgram(DecodeProgram):
  """DecodeProgram in a tpu loop.

  Note that decoder outputs across cores are concatenated along the first
  dimension. The first dimension usually corresponds to batch size and as long
  as post process decode outputs have the same expectations, this will work.

  TODO(huangyp) test this for beam search decoders and replace the
  default DecodeProgram.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'decode_num_samples', None,
        'ONLY used when steps_per_loop<0, then the actual steps_per_loop will '
        'be inferred from decode_num_samples and global batch size.')
    p.Define(
        'drop_remainder', False,
        'When `decode_num_samples` specified, whether to drop the remainder to '
        'infer the actual steps_per_loop. If True, we infer steps_per_loop as '
        'floor(decode_num_samples/batch_size), otherwise, we infer it as '
        'ceil(decode_num_samples/batch_size). '
        'Note: when drop_remainder=False, it is user\'s repsonsibility to pad '
        'the data for the last batch.')
    p.num_threads = 2
    return p

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self._program_name = 'ExperimentalDecodeProgram'
    self._dataset_summaries = {}

  def Summary(self):
    return self._dataset_summaries

  def DecodeFunc(self):
    """Wrap the DecodeLoop with split_compile_and_shard."""

    def _DecodeStep():
      """Decode call to be compiled for TPU."""
      # Applies EMA if applicable to support running only eval/decode programs.
      _, decode_dict = self._model.ConstructDecodeGraph(apply_ema=True)
      self.decode_nm = py_utils.NestedMap(decode_dict)
      return [self._OutfeedEnqueue(decode_dict)]

    @tpu_function.on_device_training_loop
    def DecodeLoopFn():
      return tpu_training_loop.repeat(
          self._steps_per_loop, _DecodeStep, inputs=[])

    self._compile_op, self.decode_loop = tpu.split_compile_and_shard(
        DecodeLoopFn,
        num_shards=self.data_parallelism,
        device_assignment=py_utils.GetTpuDeviceAssignment())

    # Pack the list of outfeed ops with structure in decode_nm.
    decode_tensors = self.decode_nm.Pack(self._OutfeedDequeue(self.decode_nm))
    cpu_pt = self._task.input.DequeueCpuPassthrough()
    return decode_tensors, cpu_pt

  def BuildTpuSubgraph(self):
    p = self.params
    tf.logging.info(
        f'ExperimentalDecodeProgram {p.dataset_name} BuildTpuSubgraph')
    if py_utils.IsEagerMode():
      raise NotImplementedError(
          'ExperimentalDecodeProgram is not supported in eager mode.')
    self.spmd = (p.spmd or self._task_params.input.use_partitioned_infeed_queue)
    with cluster_factory.SetEval(True):
      py_utils.ResetStepSeed()
      with py_utils.OpportunisticVariableReuseScope(True):
        self._model = self._InstantiateTaskModel(self._task_params)
      self._task = self._model.GetTask()
      # In Graph mode `InfeedSetupGraph` is called once to build the infeed ops.
      # In tf.function the relevant methods will be called in `InfeedTFFunc`.
      if not py_utils.IsEagerMode():
        self._task.input.InfeedSetupGraph(cpu_passthrough=True)

      if p.steps_per_loop < 0:
        assert p.decode_num_samples, (
            'When steps_per_loop<0, ExperimentalDecodeProgram must set '
            'p.decode_num_samples!')
        p.steps_per_loop = _InferStepsPerLoop(
            p.decode_num_samples, self._task.input.GlobalBatchSize(),
            p.drop_remainder)
        self._steps_per_loop = p.steps_per_loop
      self.tpu_outs = self.DecodeFunc()

  def _OutfeedDequeue(self, decode_nm):
    """Collect outfeed dequeue from all devices.

    Args:
      decode_nm: A NestedMap containing decoded tensors.

    Returns:
      A list of tensors corresponding to stacked decoded outputs. The decoder
      outputs are stacked on the first dimension (usually corresponds to
      batch size).
    """
    num_decode_tensors = len(decode_nm.Flatten())
    outfeed_ops = [[]] * num_decode_tensors
    device_assignment = py_utils.GetTpuDeviceAssignment()
    assert device_assignment
    num_cores_per_replica = (1 if self.spmd else
                             (device_assignment.num_cores_per_replica))
    for replica in range(device_assignment.num_replicas):
      for core in range(num_cores_per_replica):
        with tf.device(device_assignment.host_device(replica, core)):
          outfeeds_per_core = tpu_ops.outfeed_dequeue_tuple(
              dtypes=[x.dtype for x in decode_nm.Flatten()],
              shapes=[x.shape for x in decode_nm.Flatten()],
              device_ordinal=device_assignment.tpu_ordinal(replica, core))
          for idx_outfeed, out_feed in enumerate(outfeeds_per_core):
            outfeed_ops[idx_outfeed] = outfeed_ops[idx_outfeed] + [out_feed]
    return [tf.concat(per_outfeed, axis=0) for per_outfeed in outfeed_ops]

  def FinalizeCallback(self, unused_finalize_ret):
    """Callback after _FinalizeDecode thread done."""
    tf.logging.info('ExperimentalDecodeProgram skip FinalizeCallback.')

  def _DecodeLoop(self, sess=None):
    sess.run(self.decode_loop)

  def _PostProcess(self, global_step, dec_metrics, elapsed_secs):
    """Postprocess decoded metrics."""
    summaries = {k: v.Summary(k) for k, v in dec_metrics.items()}
    for k, v in dec_metrics.items():
      if k.startswith('num_samples_in_batch'):
        cumulative_key = 'cumulative_num_examples' + k.removeprefix(
            'num_samples_in_batch')
        summaries[cumulative_key] = tf.Summary(value=[
            tf.Summary.Value(tag=cumulative_key, simple_value=v.total_value)
        ])
        example_rate = v.total_value / elapsed_secs
        speed_key = 'examples/sec' + k.removeprefix('num_samples_in_batch')
        summaries[speed_key] = tf.Summary(
            value=[tf.Summary.Value(tag=speed_key, simple_value=example_rate)])
    self._WriteSummaries(
        os.path.basename(self._program_dir), self.params.dataset_name,
        global_step, summaries)
    self._ReportVizierMetrics(global_step, dec_metrics)
    self._dataset_summaries[self.params.dataset_name] = summaries
    return self.params.dataset_name, summaries

  def _ThreadCall(self, func, args, callback=None, thread=None):
    """Call async thread with default Error callback to break main."""
    if thread is None:
      thread = self._infeed_pool

    def _ErrorCallback(e):
      tf.logging.exception(e)
      # Terminate the main thread.
      _thread.interrupt_main()

    return thread.apply_async(
        func, args, callback=callback, error_callback=_ErrorCallback)

  def WaitThread(self, futures):
    if not isinstance(futures, list):
      futures = [futures]
    for future in futures:
      future.get()

  def Run(self, sess=None, threadpool=None):
    self._trigger_scheduler.Trigger()
    if not self._trigger_scheduler.ShouldRun():
      return
    global_step = sess.run(self._model.global_step)
    self.SetStatusMessage(f'Executing experimental decode program on dataset '
                          f'{self.params.dataset_name} at step {global_step}, '
                          f'total steps {self._steps_per_loop}.')

    if self._task.input.params.resettable:
      tf.logging.info('Resetting input_generator.')
      self._task.input.Reset(sess)

    infeed_future = self._ThreadCall(self._InfeedLoop, args=(sess,))
    decode_future = self._ThreadCall(self._DecodeLoop, args=(sess,))

    dec_metrics = self._task.CreateDecoderMetrics()
    start_time = time.time()
    post_futures = []
    decode_out_dict_list = []
    for _ in range(self._steps_per_loop):
      decode_out_dict = _FetchDecodeOut(self.tpu_outs, sess)
      if self.params.postprocess_all_at_once:
        decode_out_dict_list.append(decode_out_dict)
      elif threadpool:
        post_futures.append(
            self._ThreadCall(
                self._task.PostProcessDecodeOut,
                args=(decode_out_dict, dec_metrics),
                thread=threadpool))
      else:
        self._task.PostProcessDecodeOut(decode_out_dict, dec_metrics)
    if self.params.postprocess_all_at_once:
      if threadpool:
        post_futures.append(
            self._ThreadCall(
                self._task.PostProcessDecodeOut,
                args=(decode_out_dict_list, dec_metrics),
                thread=threadpool))
      else:
        self._task.PostProcessDecodeOut(decode_out_dict_list, dec_metrics)

    self.WaitThread([infeed_future, decode_future] + post_futures)
    elapsed_secs = time.time() - start_time

    return_future = None
    if threadpool:
      return_future = self._ThreadCall(
          self._PostProcess,
          args=(global_step, dec_metrics, elapsed_secs),
          callback=self.FinalizeCallback,
          thread=threadpool)
    else:
      data_summaries = self._PostProcess(global_step, dec_metrics, elapsed_secs)
      self.FinalizeCallback(data_summaries)
    return return_future


class MLPerfTrainDecodeProgram(BaseProgram):
  """Run train/decode in a single session run."""

  @classmethod
  def Params(cls):
    """"Defaults parameters for Programs."""
    p = super().Params()
    p.Define('train_task', None, 'Underlying task')
    p.Define('decode_task', None, 'Underlying task')
    p.Define('train_dataset_name', None, '')
    p.Define('decode_dataset_name', None, '')
    p.Define('train_steps_per_loop', 0, '')
    p.Define('decode_steps_per_loop', 0, '')
    return p

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    if py_utils.IsEagerMode():
      raise NotImplementedError(
          'MLPerfTrainDecodeProgram is not supported in eager mode.')
    p = self.params
    if p.ml_perf is not None and p.ml_perf.benchmark_name is not None:
      self._ml_perf_log = True
      self._ml_perf = p.ml_perf
      self._ml_perf_epoch = -1
    else:
      self._ml_perf_log = False
    self._program_name = 'TrainAndDecodeProgram'
    self._train_steps_per_loop = params.train_steps_per_loop
    self._decode_steps_per_loop = params.decode_steps_per_loop
    assert self._decode_steps_per_loop == 1, ('Only supports a single decode '
                                              'step right now.')
    self._train_task_params = params.train_task
    self._decode_task_params = params.decode_task
    self._run_start = None
    self._run_stop = None
    self._train_pool = multiprocessing.dummy.Pool(1)
    self._warmup_seconds = 60

  def _InitializeVizier(self):
    """We never use vizier with MLPerfPrograms."""
    self._should_report_metrics = False

  def BuildTpuSubgraph(self):
    p = self.params
    if self._ml_perf_log:
      mlp_log.mlperf_print('global_batch_size', self._ml_perf.global_batch_size)
      mlp_log.mlperf_print('max_sequence_length',
                           self._ml_perf.max_sequence_length)
      mlp_log.mlperf_print('opt_name', self._ml_perf.optimizer_name)
      mlp_log.mlperf_print('opt_base_learning_rate',
                           self._ml_perf.base_learning_rate)
      mlp_log.mlperf_print('opt_learning_rate_warmup_steps',
                           self._ml_perf.warmup_steps)

    self._eval_metrics = metrics.TpuEvalMetrics(max_metrics=p.max_metrics)
    with py_utils.OpportunisticVariableReuseScope(True):
      self._train_model = self._train_task_params.Instantiate(
          executor_ema=self._ema)
    self._train_task = self._train_model.GetTask()
    self._train_task.input.InfeedSetupGraph()
    self._model = self._train_model
    self._task = self._model.GetTask()

    def TpuTrainStep():
      """Train a shard of a batch on a single TPU core.

      Do not calculate loss metrics.

      Returns:
       [train_op].
      """
      with py_utils.OpportunisticVariableReuseScope(True):
        self._train_model.ConstructFPropBPropGraph()
      return [self._train_task.train_op]

    def TpuTrain():
      loop_result = tpu_training_loop.repeat(
          self._train_steps_per_loop,
          TpuTrainStep,
          inputs=[],
          name='train_loop')
      return loop_result

    py_utils.ResetStepSeed()

    with py_utils.OpportunisticVariableReuseScope(True):
      self._decode_model = self._InstantiateTaskModel(self._decode_task_params)
    self._decode_task = self._decode_model.GetTask()
    self._decode_task.input.InfeedSetupGraph(cpu_passthrough=True)

    def _DecodeFn():
      """Decode call to be compiled for TPU."""
      with cluster_factory.SetEval(True):
        # Applies EMA if applicable to support running only eval/decode
        # programs.
        _, decode_dict = self._decode_model.ConstructDecodeGraph(apply_ema=True)
      self.decode_nm = py_utils.NestedMap(decode_dict)
      return self.decode_nm.Flatten()

    @tpu_function.on_device_training_loop
    def TrainAndDecode():
      with tf.control_dependencies([TpuTrain()]):
        return _DecodeFn()

    self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
        TrainAndDecode,
        num_shards=self.data_parallelism,
        device_assignment=py_utils.GetTpuDeviceAssignment())

    decode_tensors = self.decode_nm.Pack(batch_parallel_res)
    cpu_pt = self._decode_task.input.DequeueCpuPassthrough()
    self.tpu_outs = (decode_tensors, cpu_pt)

  def _InfeedLoop(self, sess=None):
    if py_utils.IsEagerMode():
      # Eager mode infeed is run as part of the device loop.
      return
    tf.logging.info('_InfeedLoop start')
    try:
      for i in range(self._train_steps_per_loop):
        tf.logging.vlog(1, '_InfeedLoop %d', i)
        sess.run(self._train_task.input.tpu_infeed_op)
      if self._ml_perf_log:
        mlp_log.mlperf_print(
            'eval_start',
            None,
            metadata={
                'first_epoch_num': self._ml_perf_epoch + 1,
                'epoch_count': 1
            })
      for i in range(self._decode_steps_per_loop):
        tf.logging.vlog(1, '_InfeedLoop %d', i)
        sess.run(self._decode_task.input.tpu_infeed_op)
      tf.logging.info('_InfeedLoop done')
    except Exception as e:
      tf.logging.info('_InfeedLoop exception %r %s', e, e)
      raise

  def _TrainAndDecode(self, sess=None):
    decode_out_dict = _FetchDecodeOut(self.tpu_outs, sess)
    self._decode_task.PostProcessDecodeOut(decode_out_dict, self.dec_metrics)

  def Run(self, sess=None):
    global_step = sess.run(self._model.global_step)
    self.dec_metrics = self._decode_task.CreateDecoderMetrics()
    # Start TPU program thread.
    train_future = self._train_pool.apply_async(
        self._TrainAndDecode, args=(sess,))

    if self._warmup_seconds > 0:
      # The first execution of the TPU program has a warm-up
      # so we delay feeding data yet as that's when the MLPerf timing
      # starts. This way, when we actually infeed, the TPU program
      # is immediately ready to execute/dequeue data.
      tf.logging.info('Waiting before first infeed.')
      time.sleep(self._warmup_seconds)
      self._warmup_seconds = 0

    if self._ml_perf_log:
      if not self._run_start:
        mlp_log.mlperf_print(key='init_stop', value=None)
        self._run_start = mlp_log.mlperf_print(key='run_start', value=None)
      steps_per_epoch = self._ml_perf.steps_per_epoch
      epoch = int(global_step) // steps_per_epoch
      if epoch > self._ml_perf_epoch:
        self._ml_perf_epoch = epoch
        mlp_log.mlperf_print(
            'block_start',
            None,
            metadata={
                'first_epoch_num': epoch + 1,
                'epoch_count': 1
            })
      self.SetStatusMessage('MLPerf epoch: %d' % self._ml_perf_epoch)
    # Start infeed thread.
    infeed_future = self._infeed_pool.apply_async(
        self._InfeedLoop, args=(sess,))

    infeed_future.get()
    train_future.get()

    if self._ml_perf_log:
      mlp_log.mlperf_print(
          'eval_stop', None, metadata={'epoch_num': (epoch + 1)})
      mlperf_metric = self._ml_perf.decoder_metric_name
      mlperf_metric_value = float(self.dec_metrics[mlperf_metric].value)
      mlp_log.mlperf_print(
          'eval_accuracy', mlperf_metric_value, metadata={'epoch_num': epoch})

      # Successful ML Perf run if we exceed target accuracy
      if mlperf_metric_value > self._ml_perf.decoder_metric_success_threshold:
        tf.logging.info('ml_perf_final_threshold: %f exceeded',
                        self._ml_perf.decoder_metric_success_threshold)
        if not self._run_stop:
          self._run_stop = mlp_log.mlperf_print(
              'run_stop', None, metadata={'status': 'success'})
          self.SetStatusMessage('MLPerf run_time: %.2f' %
                                (self._run_stop - self._run_start))
          return True

      # Failed ML Perf run if we fail to reach target accuracy after
      # predefined number of steps.
      elif global_step >= self._ml_perf.max_steps_to_train:
        if not self._run_stop:
          self._run_stop = mlp_log.mlperf_print(
              'run_stop', None, metadata={'status': 'abort'})
          self.SetStatusMessage('MLPerf run_time: %.2f' %
                                (self._run_stop - self._run_start))
          return True

    return False


class MultiTaskProgramSchedule:
  """Container for ProgramSchedules for a MultiTask model."""

  @classmethod
  def Params(cls):
    p = hyperparams.InstantiableParams(cls)
    p.Define('program_schedule_dict', {}, 'task_name -> ProgramScheduleParams')
    return p


class SimpleProgramSchedule:
  """A schedule of programs associated with a single task.

  Simple sequence is:
  Run train_executions_per_eval * train_program
  Run all the eval_programs
  """

  @classmethod
  def Params(cls):
    """Params for a SimpleProgramSchedule."""
    p = hyperparams.InstantiableParams(cls)
    p.Define('task_dict', None, 'dataset_name -> task params')
    p.Define('task_name', None, 'High level task name')
    p.Define('logdir', None, 'Log directory')
    p.Define('train_program', None, 'Train program params')
    p.Define('train_executions_per_eval', 1, '')
    p.Define('eval_programs', [], 'List of eval program params.')
    p.Define('num_splits_per_client', None, '')
    p.Define('dataset_names', [], 'List of all dataset names.')
    p.Define('emails', [], 'List of emails to send metrics.')
    p.Define('summary_exporter', None, 'The summary exporter Params.')
    p.Define('async_postprocess', True,
             'whether to CPU postprocess asynchronously with TPU train')
    p.Define(
        'checkpoint_to_load', None,
        'If set, the program will initially load from this checkpoint, '
        'ignoring train_dir. Typically used for oneoff decode.')

    # TODO(blee): Clean these up.
    p.Define('ml_perf', hyperparams.Params(), 'MlPerf configuration.')
    mlp = p.ml_perf
    mlp.Define('submission_metadata', None,
               'A dictionary of static submission metadata')
    mlp.Define('benchmark_name', None, 'Benchmark name for compliance log.')
    mlp.Define('steps_per_epoch', None, 'Number of training steps per epoch.')
    mlp.Define('decoder_metric_name', None,
               'Name of the decoder metric to report for compliance log.')
    mlp.Define('decoder_metric_success_threshold', None,
               'Benchmark run must exceed this value to succeed.')
    mlp.Define('max_steps_to_train', None,
               'Maximum number of steps to reach target accuracy')
    return p

  def __init__(self, params, **kwargs):
    self.params = params.Copy()
    p = self.params
    self._programs = []
    self.train_program = None

    # Propagate run-time parameters to programs:
    if p.train_executions_per_eval > 0:
      p.train_program.logdir = p.logdir
      if p.train_program.dataset_name not in p.task_dict:
        raise ValueError('could not find train dataset %s in %s' %
                         (p.train_program.dataset_name, p.task_dict))
      p.train_program.task = p.task_dict[p.train_program.dataset_name]
      p.train_program.num_splits_per_client = p.num_splits_per_client
      p.train_program.task_name = p.task_name
      self.train_program = p.train_program.Instantiate(**kwargs)
      self._programs.append(self.train_program)

    for eval_program_params in p.eval_programs:
      eval_program_params.logdir = p.logdir
      task_dataset = None
      if issubclass(eval_program_params.cls, MultiInputsDecodeProgram):
        eval_program_params.input_params = []
        for dataset_name in eval_program_params.dataset_names:
          eval_program_params.input_params.append(
              p.task_dict[dataset_name].input)
        # Use the 1st dataset's task params.
        task_dataset = eval_program_params.dataset_names[0]
      else:
        if eval_program_params.dataset_name not in p.task_dict:
          raise ValueError('could not find eval dataset %s in %s' %
                           (eval_program_params.dataset_name, p.task_dict))
        task_dataset = eval_program_params.dataset_name
      eval_program_params.task = p.task_dict[task_dataset]
      eval_program_params.task_name = p.task_name
      eval_program_params.num_splits_per_client = p.num_splits_per_client
      # For oneoff decode, we unset trigger_offset, trigger_interval to ignore.
      if p.train_executions_per_eval == 0 and issubclass(
          eval_program_params.cls, DecodeProgram):
        eval_program_params.trigger_offset = 0
        eval_program_params.trigger_interval = 1

    self.eval_programs = []
    for eval_program in p.eval_programs:
      self.eval_programs.append(eval_program.Instantiate(**kwargs))
    self._programs += self.eval_programs

    if p.summary_exporter:
      p.summary_exporter.emails = p.emails
      p.summary_exporter.logdir = p.logdir
      self._summary_exporter = p.summary_exporter.Instantiate()

  def Programs(self):
    return self._programs

  def Run(self, sess=None, threadpool=None):
    """Execute the program schedule."""
    p = self.params
    start_time = time.time()
    for _ in range(p.train_executions_per_eval):
      done = self.train_program.Run(sess)
      if done:
        break
    train_finish_time = time.time()
    train_time_in_secs = train_finish_time - start_time
    tf.logging.info('Train took %f seconds.', train_time_in_secs)

    program_futures = []
    for eval_program in self.eval_programs:
      futures = None
      eval_program.train_executions_per_eval = p.train_executions_per_eval
      if p.async_postprocess and isinstance(
          eval_program, (DecodeProgram, ExperimentalDecodeProgram)):
        futures = eval_program.Run(sess, threadpool)
        if not isinstance(futures, list):
          futures = [futures]
        program_futures.append((futures, eval_program))
      else:
        eval_program.Run(sess)
        program_futures.append(([], eval_program))

    if p.train_executions_per_eval == 0 and hasattr(self, '_summary_exporter'):
      dataset_summaries = {}
      for pf in program_futures:
        for x in pf[0]:
          x.get()
        if isinstance(pf[1], DecodeProgram):
          dataset_summaries.update(pf[1].Summary())
      self._summary_exporter.Export(dataset_summaries, p.checkpoint_to_load)
    eval_time_in_secs = time.time() - train_finish_time
    tf.logging.info('Eval took %f seconds.', eval_time_in_secs)
    should_exit = (p.train_executions_per_eval == 0)
    return should_exit, train_time_in_secs, eval_time_in_secs

  def Shutdown(self):
    if self.train_program:
      self.train_program.Shutdown()
    for eval_program in self.eval_programs:
      eval_program.Shutdown()


def _CreateProgramParams(cls,
                         program_name,
                         dataset_name,
                         steps_per_loop,
                         decode_num_samples=None,
                         spmd=False):
  """Create different program param instance per inputs."""
  p = cls.Params()
  p.name = program_name
  p.spmd = spmd
  if issubclass(cls, MultiInputsDecodeProgram):
    assert isinstance(dataset_name, list)
    p.dataset_names = dataset_name
  else:
    p.dataset_name = dataset_name
  p.steps_per_loop = steps_per_loop
  if 'decode_num_samples' in p and decode_num_samples:
    p.decode_num_samples = decode_num_samples
  return p


def _CheckLengthOrExpand(param_per_dataset, expected_len, param_name):
  """Check the length of param_per_dataset, if it's not list, expand to list."""
  if param_per_dataset is None:
    return None
  if isinstance(param_per_dataset, list):
    if len(param_per_dataset) != expected_len:
      raise ValueError(f'{param_name} doesn\'t match the size of '
                       f'eval_dataset_names: {len(param_per_dataset)} vs '
                       f'{expected_len}.')
  else:
    param_per_dataset = [param_per_dataset] * expected_len
  return param_per_dataset


def SimpleProgramScheduleForTask(train_dataset_name,
                                 train_steps_per_loop,
                                 eval_dataset_names,
                                 eval_steps_per_loop,
                                 decode_steps_per_loop=None,
                                 decode_num_samples=None,
                                 decode_trigger_offset=0,
                                 decode_trigger_interval=1,
                                 experimental_decoder=False,
                                 multi_inputs_decoder=False,
                                 train_program_cls=TrainProgram,
                                 eval_program_cls=EvalProgram,
                                 summary_exporter_cls=None,
                                 async_postprocess=True,
                                 experimental_async_postprocess=False,
                                 decode_until_out_of_range=False,
                                 postprocess_all_at_once=False,
                                 emails=None,
                                 train_summary_interval_steps=None,
                                 spmd=False):
  """Convenient helper method for common case.

  Args:
    train_dataset_name: Name of the training dataset, eg: 'Train'
    train_steps_per_loop: Number of steps to execute the training program.
    eval_dataset_names: List of eval dataset_name strings, eg: ['Train'].
    eval_steps_per_loop: Number of steps to execute the eval program. Can be a
      single value or a list of values corresponding to the entries in
      eval_dataset_names.
    decode_steps_per_loop: Number of steps to execute the decode program. Can be
      a single value or a list of values corresponding to the entries in
      eval_dataset_names. If the value is negative, DecodeProgram will
      automatically decoding until out of range; ExperimentalDecodeProgram will
      infer the steps_per_loop from `decode_num_samples`.
    decode_num_samples: Number of samples to run decode program. This is only
      used when decode_steps_per_loop<0, and ExperimentalDecodeProgram is used.
    decode_trigger_offset: The program is only effectively triggered after this
      many of runs. The default is 0 where no runs is skipped. It is used with
      trigger_interval below to control trigger schedule.
    decode_trigger_interval: The program is only effectively triggered every
      this num of runs, after trigger_offset is met.
    experimental_decoder: bool. Whether to use experimental deocder which is
      placed in a tpu loop.
    multi_inputs_decoder: bool. Whether to use multi inputs decoder for all
      datasets.
    train_program_cls: The class to use for training programs.  Defaults to
      TrainProgram.
    eval_program_cls: The class to use for eval programs.  Defaults to
      EvalProgram.
    summary_exporter_cls: The class to export summaries. Default to None. If
      not None, the class should implement Export().
    async_postprocess: bool. Whether to run CPU postprocessing for Decode in a
      separate thread to save time (i.e. concurrent with train). This avoids
      blocking training. But if the CPU postprocessing takes more time compared
      to Train, then multiple Train loops could complete before Decode finishes
      for an older global step. Then the latest Decode results do not correspond
      to the latest trained model.
    experimental_async_postprocess: bool. Whether to run CPU postprocessing in
      separate thread to save time. This flag is only used for
      ExperimentalDecodeProgram, when experimental_decoder=True.
    decode_until_out_of_range(DEPRECATED): bool. Whether to run Decode (and its
      Infeed loop) until there is no more data (OutOfRange error is thrown). If
      this is True, decode_steps_per_loop is ignored (and not required).
      Currently do not support ExperimentalDecodeProgram, which uses loop on
      TPU. So keep experimental_decoder=False
    postprocess_all_at_once: bool/List. Whether to postprocess the (combined)
      batches at once at the end of Decode program, instead of once per step.
      This is needed if one needs to reference/combine data across different
      batches/steps during postprocess. The PostProcess(DecodeOut) function
      should define the logic of aggregating across steps/batches. Can be a
      single value or a list of values corresponding to the entries in
      eval_dataset_names.
    emails: List of emails to email decode/scoring summaries.
    train_summary_interval_steps: Number of steps to wait before flushing
      summaries to disk.
    spmd: Wether all the programs are running in SPMD mode.

  Returns:
    A populated SimpleProgramSchedule.Params()
  """

  program_schedule_params = SimpleProgramSchedule.Params()
  program_schedule_params.train_program = _CreateProgramParams(
      train_program_cls,
      'train',
      train_dataset_name,
      train_steps_per_loop,
      spmd=spmd)
  if issubclass(train_program_cls, TrainProgram):
    program_schedule_params.train_program.summary_interval_steps = train_summary_interval_steps

  program_schedule_params.dataset_names = []

  if experimental_decoder:
    program_schedule_params.async_postprocess = experimental_async_postprocess
  else:
    program_schedule_params.async_postprocess = async_postprocess

  if emails:
    program_schedule_params.emails = emails

  if summary_exporter_cls:
    program_schedule_params.summary_exporter = summary_exporter_cls.Params()

  num_eval_datasets = len(eval_dataset_names)
  eval_steps_per_loop = _CheckLengthOrExpand(eval_steps_per_loop,
                                             num_eval_datasets,
                                             'eval_steps_per_loop')
  # TODO(xingwu): fully deprecate decode_until_out_of_range.
  if decode_until_out_of_range and decode_steps_per_loop is None:
    decode_steps_per_loop = [-1] * num_eval_datasets

  if decode_steps_per_loop is None:
    raise ValueError('decode_steps_per_loop must be specified as int or list!')

  decode_steps_per_loop = _CheckLengthOrExpand(decode_steps_per_loop,
                                               num_eval_datasets,
                                               'decode_steps_per_loop')
  decode_trigger_offset = _CheckLengthOrExpand(decode_trigger_offset,
                                               num_eval_datasets,
                                               'decode_trigger_offset')
  decode_trigger_interval = _CheckLengthOrExpand(decode_trigger_interval,
                                                 num_eval_datasets,
                                                 'decode_trigger_interval')

  if decode_num_samples is not None:
    if not isinstance(decode_num_samples, list):
      raise ValueError('decode_num_samples must be a list.')

  decode_num_samples = _CheckLengthOrExpand(decode_num_samples,
                                            num_eval_datasets,
                                            'decode_num_samples')
  postprocess_all_at_once = _CheckLengthOrExpand(postprocess_all_at_once,
                                                 num_eval_datasets,
                                                 'postprocess_all_at_once')

  for idx, dataset_name in enumerate(eval_dataset_names):
    program_schedule_params.dataset_names.append(dataset_name)
    if eval_steps_per_loop[idx] > 0:
      program_schedule_params.eval_programs.append(
          _CreateProgramParams(
              eval_program_cls,
              'eval_tpu',
              dataset_name,
              eval_steps_per_loop[idx],
              spmd=spmd))
    if not multi_inputs_decoder and decode_steps_per_loop[idx] != 0:
      decoder = (
          ExperimentalDecodeProgram if experimental_decoder else DecodeProgram)
      decoder_param = _CreateProgramParams(
          decoder,
          'decode_tpu',
          dataset_name,
          decode_steps_per_loop[idx],
          decode_num_samples[idx] if decode_num_samples else None,
          spmd=spmd)
      decoder_param.postprocess_all_at_once = postprocess_all_at_once[idx]
      decoder_param.trigger_offset = decode_trigger_offset[idx]
      decoder_param.trigger_interval = decode_trigger_interval[idx]
      program_schedule_params.eval_programs.append(decoder_param)

  if multi_inputs_decoder:
    program_schedule_params.eval_programs.append(
        _CreateProgramParams(
            MultiInputsDecodeProgram,
            'decode_tpu',
            eval_dataset_names,
            decode_steps_per_loop,
            spmd=spmd))

  return program_schedule_params


def _ClearSpecifiedProgram(program_list, program_cls_to_clear):
  ret_programs = []
  for program in program_list:
    if not issubclass(program.cls, program_cls_to_clear):
      ret_programs.append(program)
  return ret_programs


def UpdateProgramSchedule(ps_params,
                          dataset_list=None,
                          train_executions_per_eval=None,
                          train_steps_per_loop=None,
                          eval_steps_per_loop=None,
                          decode_steps_per_loop=None,
                          multi_inputs_decoder=None,
                          decode_summary_emails=None,
                          oneoff_checkpoint_to_load=None,
                          train_summary_interval_steps=None):
  """Update ProgramSchedule params with the given new configs.

  Currently this override only support EvalProgram, DecodeProgram and
  MultiInputsDecodeProgram.

  Args:
    ps_params: SimpleProgramSchedule.Params(), to be overriden.
    dataset_list: Optional[List[str]], if not None, it will override eval
      datasets in ps_params.
    train_executions_per_eval: Optional[int], if not None, it will override
      train_executions_per_eval in ps_params.
    train_steps_per_loop: Optional[int], if not None, it will override train
      program's steps_per_loop.
    eval_steps_per_loop: Optional[int], if not None, it will override all the
      eval programs steps_per_loop. Currently list not supported.
    decode_steps_per_loop: Optional[int], if not None, it will override all the
      decode programs steps_per_loop. Currently list not supported. If it's
      negative, e.g. -1, DecodeProgram will decode until out of range;
      ExperimentalDecodeProgram will infer steps from decode_num_samples.
    multi_inputs_decoder: Optional[bool], if not None, update all testsets to
      use MultiInputsDecodeProgram (if true) or DecodeProgram (if False).
    decode_summary_emails: List of emails to send Decode summary to.
    oneoff_checkpoint_to_load: Optional[str], if not None, it will override
      checkpoint_to_load.
    train_summary_interval_steps: Optional[int], if not None, it will override
      train program's summary_interval_steps.

  Returns:
    ps_params after overriden.
  """
  assert ps_params
  if issubclass(ps_params.cls, MultiTaskProgramSchedule):
    tf.logging.info(
        'UpdateProgramSchedule does not support MultiTaskProgramSchedule.')
    return ps_params
  if dataset_list is not None:
    ps_params.dataset_names = dataset_list
    eval_programs = {}
    decode_programs = {}
    default_eval_steps_per_loop = 0
    default_decode_steps_per_loop = 0
    for eval_program in ps_params.eval_programs:
      if issubclass(eval_program.cls, EvalProgram):
        default_eval_steps_per_loop = eval_program.steps_per_loop
      elif issubclass(eval_program.cls, DecodeProgram):
        default_decode_steps_per_loop = eval_program.steps_per_loop
        if multi_inputs_decoder is None:
          multi_inputs_decoder = issubclass(eval_program.cls,
                                            MultiInputsDecodeProgram)
      if ('dataset_name' in eval_program) and (eval_program.dataset_name
                                               in dataset_list):
        if issubclass(eval_program.cls, EvalProgram):
          eval_programs[eval_program.dataset_name] = eval_program
        elif issubclass(eval_program.cls, DecodeProgram):
          decode_programs[eval_program.dataset_name] = eval_program

    # If there's no decode program in original ps_param, and we do not update
    # it, we use DecodeProgram by default.
    if multi_inputs_decoder is None:
      multi_inputs_decoder = False
    ps_params.eval_programs = []
    # Reorder the eval/decode programs sequence.
    for dataset_name in dataset_list:
      if dataset_name not in eval_programs:
        eval_programs[dataset_name] = _CreateProgramParams(
            EvalProgram, 'eval_tpu', dataset_name, default_eval_steps_per_loop)
      ps_params.eval_programs.append(eval_programs[dataset_name])

      if not multi_inputs_decoder:
        if dataset_name not in decode_programs:
          decode_programs[dataset_name] = _CreateProgramParams(
              DecodeProgram, 'decode_tpu', dataset_name,
              default_decode_steps_per_loop)
        ps_params.eval_programs.append(decode_programs[dataset_name])

    if multi_inputs_decoder:
      ps_params.eval_programs.append(
          _CreateProgramParams(MultiInputsDecodeProgram, 'decode_tpu',
                               dataset_list, default_decode_steps_per_loop))

  if train_executions_per_eval is not None:
    ps_params.train_executions_per_eval = train_executions_per_eval

  if train_steps_per_loop is not None:
    ps_params.train_program.steps_per_loop = train_steps_per_loop

  if eval_steps_per_loop is not None:
    if eval_steps_per_loop == 0:
      ps_params.eval_programs = _ClearSpecifiedProgram(ps_params.eval_programs,
                                                       EvalProgram)
    else:
      for eval_program in ps_params.eval_programs:
        if issubclass(eval_program.cls, EvalProgram):
          eval_program.steps_per_loop = eval_steps_per_loop

  if decode_steps_per_loop is not None:
    if decode_steps_per_loop == 0:
      ps_params.eval_programs = _ClearSpecifiedProgram(ps_params.eval_programs,
                                                       DecodeProgram)
    else:
      for eval_program in ps_params.eval_programs:
        if issubclass(eval_program.cls, DecodeProgram):
          eval_program.steps_per_loop = decode_steps_per_loop

  if oneoff_checkpoint_to_load:
    if ps_params.train_executions_per_eval:
      tf.logging.warning(
          'Training with decoding does not suggest to set `checkpoint_to_load` '
          'for DecodeProgram!')
    ps_params.checkpoint_to_load = oneoff_checkpoint_to_load

  if decode_summary_emails:
    ps_params.emails = decode_summary_emails

  if train_summary_interval_steps is not None:
    ps_params.train_program.summary_interval_steps = train_summary_interval_steps

  return ps_params


class MLPerfProgramSchedule:
  """Program schedule for ML Perf benchmark."""

  @classmethod
  def Params(cls):
    """Params for a MLPerfProgramSchedule."""
    p = hyperparams.InstantiableParams(cls)

    p.Define('task_dict', None, 'dataset_name -> task params')
    p.Define('task_name', None, 'High level task name')
    p.Define('logdir', None, 'Log directory')
    p.Define('train_program', None, 'Train program params')
    p.Define('train_executions_per_eval', 1, '')
    p.Define('dataset_names', [], 'List of all dataset names.')
    p.Define('num_splits_per_client', None, '')

    p.Define('ml_perf', hyperparams.Params(), 'MlPerf configuration.')

    mlp = p.ml_perf
    mlp.Define('benchmark_name', None, 'Benchmark name for compliance log.')
    mlp.Define('decoder_metric_name', None,
               'Name of the decoder metric to report for compliance log.')
    mlp.Define('decoder_metric_success_threshold', None,
               'Benchmark run must exceed this value to succeed.')
    mlp.Define('max_steps_to_train', None,
               'Maximum number of steps to reach target accuracy')
    mlp.Define('steps_per_epoch', None, 'Number of training steps per epoch.')
    mlp.Define('global_batch_size', None, 'Global batch size.')
    mlp.Define('max_sequence_length', None, 'Maximum sequence length.')
    mlp.Define('optimizer_name', None, 'Optimizer used.')
    mlp.Define('base_learning_rate', None, 'Base learning rate.')
    mlp.Define('warmup_steps', None, 'Number of warm-up steps.')

    return p

  def __init__(self, params, **kwargs):
    self.params = params.Copy()
    p = self.params

    # Propagate run-time parameters to programs:
    p.train_program.logdir = p.logdir
    if p.train_program.train_dataset_name not in p.task_dict:
      tf.logging.error('could not find %s in %s' %
                       (p.train_program.train_dataset_name, p.task_dict))

    if p.train_program.decode_dataset_name not in p.task_dict:
      tf.logging.error('could not find %s in %s' %
                       (p.train_program.decode_dataset_name, p.task_dict))

    p.train_program.train_task = p.task_dict[p.train_program.train_dataset_name]
    p.train_program.decode_task = p.task_dict[
        p.train_program.decode_dataset_name]

    p.train_program.num_splits_per_client = p.num_splits_per_client
    p.train_program.task_name = p.task_name
    p.train_program.ml_perf = p.ml_perf.Copy()

    self.train_program = p.train_program.Instantiate(**kwargs)
    self._programs = []
    self._programs.append(self.train_program)

  def Programs(self):
    return self._programs

  def Run(self, sess=None, threadpool=None):
    """Execute the program schedule."""
    del threadpool  # Unused.
    p = self.params
    start_time = time.time()
    ret = False
    for _ in range(p.train_executions_per_eval):
      program_done = self.train_program.Run(sess)
      if program_done:
        ret = True
        break
    train_time_in_secs = time.time() - start_time
    eval_time_in_secs = 0
    return ret, train_time_in_secs, eval_time_in_secs

  def Shutdown(self):
    self.train_program.Shutdown()


def MLPerfProgramScheduleForTask(train_dataset_name, train_steps_per_loop,
                                 decode_dataset_name, decode_steps_per_loop):
  """Populate MLPerfProgramSchedule params.

  Args:
    train_dataset_name: Name of the training dataset, eg: 'Train'.
    train_steps_per_loop: Number of steps to execute the training program.
    decode_dataset_name:  Eg: 'Test'.
    decode_steps_per_loop: Number of steps to execute the decode program.

  Returns:
    A populated MLPerfProgramSchedule.Params()
  """

  program_schedule_params = MLPerfProgramSchedule.Params()
  train_program_params = MLPerfTrainDecodeProgram.Params()
  train_program_params.name = 'train_and_decode'
  train_program_params.train_steps_per_loop = train_steps_per_loop
  train_program_params.decode_steps_per_loop = decode_steps_per_loop

  train_program_params.dataset_name = train_dataset_name
  train_program_params.train_dataset_name = train_dataset_name
  train_program_params.decode_dataset_name = decode_dataset_name

  program_schedule_params.train_program = train_program_params

  program_schedule_params.dataset_names = [
      train_dataset_name, decode_dataset_name
  ]

  return program_schedule_params
