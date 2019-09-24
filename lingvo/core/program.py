# -*- coding: utf-8 -*-
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
"""Programs for interleaving execution on TPU."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing.dummy
import os
import time

import lingvo.compat as tf
from lingvo.core import checkpointer
from lingvo.core import hyperparams
from lingvo.core import metrics
from lingvo.core import py_utils
import six
from six.moves import xrange
from six.moves import zip

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import training_loop as tpu_training_loop
from tensorflow.python.tpu.ops import tpu_ops
# pylint:enable=g-direct-tensorflow-import


class BaseProgram(object):
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
    return p

  def __init__(self, params):
    self.params = params.Copy()
    p = self.params
    self._task_params = p.task
    self._logdir = p.logdir
    self._task_name = p.task_name

    # Program dirs are where the summaries are written to.
    if p.task_name:
      program_dir_name = p.task_name + '_' + p.name + '_' + p.dataset_name.lower(
      )
    else:
      program_dir_name = p.name + '_' + p.dataset_name.lower()
    self._program_dir = os.path.join(self._logdir, program_dir_name)
    self._summary_writer = tf.summary.FileWriter(self._program_dir)

    tf.gfile.MakeDirs(self._logdir)
    # Just a standard spot that all programs may restore from.
    self._checkpoint_dir = os.path.join(self._logdir, 'train')
    tf.gfile.MakeDirs(self._checkpoint_dir)

    self._steps_per_loop = p.steps_per_loop
    self.num_splits_per_client = p.num_splits_per_client
    self.data_parallelism = p.num_splits_per_client

    # Thread Pool for infeed.
    self._infeed_pool = multiprocessing.dummy.Pool(1)

  def _SummarizeValue(self, steps, tag, value):
    self._summary_writer.add_summary(
        metrics.CreateScalarSummary(tag, value), steps)

  def _InfeedLoop(self, sess):
    tf.logging.info('_InfeedLoop start')
    for i in range(self._steps_per_loop):
      tf.logging.info('_InfeedLoop %d', i)
      sess.run(self._model.GetTask().input_generator.tpu_infeed_op)

  def BuildTpuSubgraph(self):
    """Sub classes should construct a model/graph to be executed by Run.

    Specific to TPU execution, this may involve a
    @tpu_function.on_device_training_loop etc.
    """
    raise NotImplementedError()

  def Run(self, sess):
    """Execute the program using the given session handle."""
    raise NotImplementedError()

  def RestoreIfNeeded(self, sess):
    """Restore from checkpoint if necessary."""
    raise NotImplementedError()


class TrainProgram(BaseProgram):
  """TrainProgram trains a single task and handles checkpoints."""

  @classmethod
  def Params(cls):
    """Parameters for TrainProgram."""
    p = super(TrainProgram, cls).Params()

    p.Define(
        'always_checkpoint_after_execution', True,
        'Always save a checkpoint after running the TrainProgram '
        'for `steps_per_loop`. Note that overrides the normal '
        '`save_interval_seconds` parameter.')
    return p

  def __init__(self, params):
    super(TrainProgram, self).__init__(params)
    self._time_steps = []  # A short history of (timestamp, global_step)

  def _CreateCheckpointer(self, train_dir, model):
    return checkpointer.Checkpointer(train_dir, model)

  def _RecordStepRate(self, current_steps, total_examples):
    """Computes the overall step rate and adds a summary."""
    self._time_steps.append((time.time(), current_steps, total_examples))
    # Keeps a relative long history to compute a smooth steps/second.
    # Removes duplicate stats for step = 0 to get rid of the warm-up period.
    while (self._time_steps[-1][1] - self._time_steps[0][1] > 10000 or
           (len(self._time_steps) > 1 and
            self._time_steps[0][1] == self._time_steps[1][1])):
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
      for replica in xrange(device_assignment.num_replicas):
        for core in xrange(device_assignment.num_cores_per_replica):
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

  def BuildTpuSubgraph(self):
    tf.logging.info('TrainProgram BuildTpuSubGraph')
    with py_utils.OpportunisticVariableReuseScope(True):
      self._eval_metrics = metrics.TpuEvalMetrics()
      data_parallelism = self.data_parallelism

      def TpuTrainStep(*args):
        """Train a shard of a batch on a single TPU core.

        Args:
          *args: metrics values from previous steps.

        Returns:
          New summed metrics values and a train_op.
        """
        self._model = self._task_params.Instantiate()
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
        loop_result = tpu_training_loop.repeat(
            self._steps_per_loop,
            TpuTrainStep,
            inputs=self._eval_metrics.initial_values,
            name='train_loop')
        # Final metrics are the avg across self._steps_per_loop steps.
        return self._eval_metrics.FinalizeMetrics(loop_result)

      batch_parallel_res = tf.tpu.batch_parallel(
          TpuTrain,
          num_shards=data_parallelism,
          device_assignment=py_utils.GetTpuDeviceAssignment())
      outfeed_dequeue_op = self._OutfeedDequeueLoop(
          self._model.GetTask().per_example_tensors, self._steps_per_loop,
          self.num_splits_per_client)
      # Get metric result from a single replica; they are all same here.
      self.tpu_ops = [[t[0] for t in batch_parallel_res], outfeed_dequeue_op]

      self._checkpointer = self._CreateCheckpointer(self._checkpoint_dir,
                                                    self._model)
    return self.tpu_ops

  def RestoreIfNeeded(self, sess):
    self._checkpointer.RestoreIfNeeded(sess)

  def Run(self, sess):
    tf.logging.info('Executing train program for %s.', self._task_name)
    gsteps = py_utils.GetGlobalStep()

    infeed_future = self._infeed_pool.apply_async(
        self._InfeedLoop, args=(sess,))
    ary = sess.run(self.tpu_ops)
    infeed_future.wait()

    values = ary[0]
    outfeeds = ary[1]

    eval_metrics = self._eval_metrics.PackMetricsValues(values)

    task = self._model.GetTask()
    global_step, total_examples = sess.run([gsteps, task.total_examples_var])
    self._RecordStepRate(global_step, total_examples)

    msg = 'step:%6d' % global_step
    for key, (val, _) in sorted(six.iteritems(eval_metrics)):
      msg += ' %s:%.8g' % (key, val)
      self._SummarizeValue(global_step, key, val)

    task.ProcessFPropResults(sess, global_step, eval_metrics, outfeeds)


class EvalProgram(BaseProgram):
  """Evaluation program.

  Note that this currently has different infeed semantics compared to
  the existing Evaler as the input generator is not recreated
  per-eval. Thus different random samples are selected each
  evaluation.
  """

  def BuildTpuSubgraph(self):
    tf.logging.info('EvalProgram BuildTpuSubGraph')
    with py_utils.OpportunisticVariableReuseScope(True):
      self._eval_metrics = metrics.TpuEvalMetrics()
      data_parallelism = self.data_parallelism

      def TpuEvalStep(*args):
        """Eval a shard of a batch on a single TPU core.

        Args:
          *args: metrics values from previous steps.

        Returns:
          Per-step eval metrics.
        """
        self._model = self._task_params.Instantiate()
        self._model.ConstructFPropGraph()
        per_step_eval_metrics = self._eval_metrics.SetMetrics(
            self._model.GetTask().eval_metrics, args)
        return per_step_eval_metrics

      @tpu_function.on_device_training_loop
      def TpuEval():
        loop_result = tpu_training_loop.repeat(
            self._steps_per_loop,
            TpuEvalStep,
            inputs=self._eval_metrics.initial_values,
            name='eval_loop')
        # Final metrics are the avg across self._steps_per_loop steps.
        return self._eval_metrics.FinalizeMetrics(loop_result)

      batch_parallel_res = tf.tpu.batch_parallel(
          TpuEval,
          num_shards=data_parallelism,
          device_assignment=py_utils.GetTpuDeviceAssignment())
      # Get metric result from a single replica; they are all same here.
      self.tpu_ops = [[t[0] for t in batch_parallel_res]]
      self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                     self._model)

      return self.tpu_ops

  def RestoreIfNeeded(self, sess):
    self._checkpointer.RestoreIfNeeded(sess)

  def Run(self, sess):
    tf.logging.info('Executing eval program for %s.', self._task_name)
    gsteps = py_utils.GetGlobalStep()
    infeed_future = self._infeed_pool.apply_async(
        self._InfeedLoop, args=(sess,))
    ary = sess.run(self.tpu_ops)
    infeed_future.wait()
    values = ary[0]
    eval_metrics = self._eval_metrics.PackMetricsValues(values)
    global_step = sess.run(gsteps)
    for key, (val, _) in sorted(six.iteritems(eval_metrics)):
      self._SummarizeValue(global_step, key, val)


class DecodeProgram(BaseProgram):
  """DecodeProgram.

  Note that this currently has different infeed semantics compared to
  the existing Decoder as the input generator is not recreated
  per-eval. Thus different random samples are selected each
  decoder run.
  """

  def _WriteSummaries(self, job_name, global_step, summaries):
    for unused_name, summary in sorted(summaries.items()):
      self._summary_writer.add_summary(summary, global_step)
      if summary.value:
        for value in summary.value:
          if value.HasField('simple_value'):
            tf.logging.info('%s summary on checkpoint@%d %s = %.8g', job_name,
                            global_step, value.tag, value.simple_value)
      self._summary_writer.flush()

  def BuildTpuSubgraph(self):
    tf.logging.info('DecodeProgram BuildTpuSubGraph')
    py_utils.ResetStepSeed()

    def _DecodeFn():
      with py_utils.OpportunisticVariableReuseScope(True):
        self._model = self._task_params.Instantiate()
        self._model_task = self._model.GetTask()
        input_batch = self._model_task.GetInputBatch()
        metrics_dict = self._model_task.Decode(input_batch)
        self.metrics_nm = py_utils.NestedMap(metrics_dict)
        return self.metrics_nm.Flatten()

    batch_parallel_res = tf.tpu.batch_parallel(
        _DecodeFn,
        num_shards=self.data_parallelism,
        device_assignment=py_utils.GetTpuDeviceAssignment())

    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   self._model)

    self.metrics = py_utils.NestedMap(self.metrics_nm)
    self.metrics = self.metrics.Pack(batch_parallel_res)
    return None

  def RestoreIfNeeded(self, sess):
    self._checkpointer.RestoreIfNeeded(sess)

  def Run(self, sess):
    tf.logging.info('Executing decode program for %s.', self._task_name)
    gsteps = py_utils.GetGlobalStep()
    global_step = sess.run(gsteps)

    infeed_future = self._infeed_pool.apply_async(
        self._InfeedLoop, args=(sess,))
    dec_metrics = self._model_task.CreateDecoderMetrics()
    start_time = time.time()
    for i in range(self._steps_per_loop):
      metrics_values = sess.run(self.metrics)
      self._model_task.PostProcessDecodeOut(metrics_values, dec_metrics)
      tf.logging.info('step: %d %f' %
                      (i, dec_metrics['num_samples_in_batch'].total_value))
    infeed_future.wait()
    num_examples_metric = dec_metrics['num_samples_in_batch']
    summaries = {k: v.Summary(k) for k, v in six.iteritems(dec_metrics)}
    elapsed_secs = time.time() - start_time
    example_rate = num_examples_metric.total_value / elapsed_secs
    summaries['examples/sec'] = tf.Summary(
        value=[tf.Summary.Value(tag='examples/sec', simple_value=example_rate)])
    self._WriteSummaries(
        os.path.basename(self._program_dir), global_step, summaries)


class MultiTaskProgramSchedule(object):
  """Container for ProgramSchedules for a MultiTask model."""

  @classmethod
  def Params(cls):
    p = hyperparams.InstantiableParams(cls)
    p.Define('program_schedule_dict', None,
             'task_name -> ProgramScheduleParams')
    return p


class SimpleProgramSchedule(object):
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
    return p

  def __init__(self, params):
    self.params = params.Copy()
    p = self.params

    # Propagate run-time parameters to programs:
    p.train_program.logdir = p.logdir
    if p.train_program.dataset_name not in p.task_dict:
      tf.logging.error('could not find %s in %s' %
                       (p.train_program.dataset_name, p.task_dict))
    p.train_program.task = p.task_dict[p.train_program.dataset_name]
    p.train_program.num_splits_per_client = p.num_splits_per_client
    p.train_program.task_name = p.task_name

    for eval_program_params in p.eval_programs:
      eval_program_params.logdir = p.logdir
      eval_program_params.task = p.task_dict[eval_program_params.dataset_name]
      eval_program_params.task_name = p.task_name
      eval_program_params.num_splits_per_client = p.num_splits_per_client

    self.eval_programs = []
    self.train_program = p.train_program.Instantiate()
    for eval_program in p.eval_programs:
      self.eval_programs.append(eval_program.Instantiate())

    self._programs = []
    self._programs.append(self.train_program)
    self._programs += self.eval_programs

  def Programs(self):
    return self._programs

  def Run(self, sess):
    p = self.params
    for _ in range(p.train_executions_per_eval):
      self.train_program.Run(sess)
    for eval_program in self.eval_programs:
      eval_program.Run(sess)


def SimpleProgramScheduleForTask(train_dataset_name, train_steps_per_loop,
                                 eval_dataset_names, eval_steps_per_loop,
                                 decode_steps_per_loop):
  """Convenient helper method for common case.

  Args:
    train_dataset_name: Name of the training dataset, eg: 'Train'
    train_steps_per_loop: Number of steps to execute the training program.
    eval_dataset_names: List of eval dataset_name strings, eg: ['Train'].
    eval_steps_per_loop: Number of steps to execute the eval program.
    decode_steps_per_loop: Number of steps to execute the decode program.

  Returns:
    A populated SimpleProgramSchedule.Params()
  """

  program_schedule_params = SimpleProgramSchedule.Params()
  train_program_params = TrainProgram.Params()
  train_program_params.name = 'train'
  train_program_params.steps_per_loop = train_steps_per_loop
  train_program_params.dataset_name = train_dataset_name
  program_schedule_params.train_program = train_program_params

  program_schedule_params.dataset_names = []

  for dataset_name in eval_dataset_names:
    program_schedule_params.dataset_names.append(dataset_name)
    if eval_steps_per_loop > 0:
      eval_program_params = EvalProgram.Params()
      eval_program_params.name = 'eval_tpu'
      # TODO(blee): This should be derived from the Dataset size.
      eval_program_params.steps_per_loop = eval_steps_per_loop
      eval_program_params.dataset_name = dataset_name
      program_schedule_params.eval_programs.append(eval_program_params)

    if decode_steps_per_loop > 0:
      decode_program_params = DecodeProgram.Params()
      decode_program_params.name = 'decode_tpu'
      # TODO(blee): This should be derived from the Dataset size.
      decode_program_params.steps_per_loop = decode_steps_per_loop
      decode_program_params.dataset_name = dataset_name
      program_schedule_params.eval_programs.append(decode_program_params)

  return program_schedule_params
