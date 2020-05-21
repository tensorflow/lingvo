# Lint as: python2, python3
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
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import cluster_factory
from lingvo.core import hyperparams
from lingvo.core import metrics
from lingvo.core import ml_perf_log as mlp_log
from lingvo.core import py_utils
from lingvo.core import summary_utils
import six
from six.moves import range
from six.moves import zip

# pylint:disable=g-direct-tensorflow-import
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.tpu import tpu
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
    p.Define('num_threads', 1, 'Number of threads in multiprocessing pool.')
    return p

  def __init__(self, params):
    self.params = params.Copy()
    p = self.params
    self._task_params = p.task
    self._logdir = p.logdir
    self._task_name = p.task_name
    self._program_name = ''

    # Program dirs are where the summaries are written to.
    if p.task_name:
      program_dir_name = (
          p.task_name + '_' + p.name + '_' + p.dataset_name.lower())
    else:
      program_dir_name = p.name + '_' + p.dataset_name.lower()
    self._program_dir = os.path.join(self._logdir, program_dir_name)
    self._summary_writer = tf.summary.FileWriter(self._program_dir)

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

  def _SummarizeValue(self, steps, tag, value):
    self._summary_writer.add_summary(
        metrics.CreateScalarSummary(tag, value), steps)

  def _InfeedLoop(self, sess):
    tf.logging.info('_InfeedLoop start')
    try:
      for i in range(self._steps_per_loop):
        tf.logging.vlog(1, '_InfeedLoop %d', i)
        sess.run(self._model.GetTask().input_generator.tpu_infeed_op)
      tf.logging.info('_InfeedLoop done')
    except Exception as e:
      tf.logging.info('_InfeedLoop exception %r %s', e, e)
      raise

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
    tf.logging.info('Status: %s', msg)
    if self._status_msg_fn:
      self._status_msg_fn(msg)

  def Compile(self, sess):
    """Compile the program using the given session handle."""
    if self._compile_op is not None:
      self.SetStatusMessage('Compiling %s' % self._program_name)
      result = sess.run(self._compile_op)
      proto = tpu_compilation_result.CompilationResultProto()
      proto.ParseFromString(result)
      if proto.status_error_message:
        tf.logging.fatal('Compilation failed: {}'.format(
            proto.status_error_message))
      tf.logging.info('Compiling %s done.', self._program_name)

  def Run(self, sess):
    """Execute the program using the given session handle.

    Args:
      sess: TF Session.

    Returns:
      done: Whether to end all execution.
    """
    raise NotImplementedError()

  def CreateCheckpointer(self):
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   self._model)

  def RestoreIfNeeded(self, sess):
    self._checkpointer.RestoreIfNeeded(sess)


class TrainProgram(BaseProgram):
  """TrainProgram trains a single task and handles checkpoints."""

  @classmethod
  def Params(cls):
    """Parameters for TrainProgram."""
    p = super(TrainProgram, cls).Params()
    return p

  def __init__(self, params):
    super(TrainProgram, self).__init__(params)
    self._step_rate_tracker = summary_utils.StepRateTracker()
    self._program_name = 'TrainProgram'

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

  def BuildTpuSubgraph(self):
    tf.logging.info('TrainProgram BuildTpuSubGraph')

    with py_utils.OpportunisticVariableReuseScope(True):
      self._eval_metrics = metrics.TpuEvalMetrics()
      data_parallelism = self.data_parallelism

      # Instantiate input generator first.
      self._input = self._task_params.input.Instantiate()
      self._input.CreateTpuEnqueueOps()
      self._task_params.input.Define('skip_create_child', True, '')

      def TpuTrainStep(*args):
        """Train a shard of a batch on a single TPU core.

        Args:
          *args: metrics values from previous steps.

        Returns:
          New summed metrics values and a train_op.
        """
        self._model = self._task_params.Instantiate()
        self._task = self._model.GetTask()
        self._task.AddChild('input', self._input)
        self._model.ConstructFPropBPropGraph()
        per_step_eval_metrics = self._eval_metrics.SetMetrics(
            self._task.eval_metrics, args)
        outfeed_op = self._OutfeedEnqueue(self._task.per_example_tensors)
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

      self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
          TpuTrain,
          num_shards=data_parallelism,
          device_assignment=py_utils.GetTpuDeviceAssignment())
      outfeed_dequeue_op = self._OutfeedDequeueLoop(
          self._model.GetTask().per_example_tensors, self._steps_per_loop,
          self.num_splits_per_client)
      # Get metric result from a single replica; they are all same here.
      self.tpu_ops = [[t[0] for t in batch_parallel_res], outfeed_dequeue_op]

    return self.tpu_ops

  def Run(self, sess):
    gsteps = py_utils.GetGlobalStep()
    global_step = sess.run(gsteps)
    self.SetStatusMessage('Executing train program at step %d' % global_step)
    infeed_future = self._infeed_pool.apply_async(
        self._InfeedLoop, args=(sess,))
    ary = sess.run(self.tpu_ops)
    infeed_future.wait()

    values = ary[0]
    outfeeds = ary[1]

    self._eval_metrics.PackMetricsValues(values)
    eval_metrics = self._eval_metrics.metrics

    global_step = sess.run(gsteps)
    step_rate, example_rate, total_examples = (
        self._step_rate_tracker.ComputeStepRate(
            global_step,
            eval_metrics['num_samples_in_batch'][0] * self._steps_per_loop))
    self._SummarizeValue(global_step, 'global_step/sec', step_rate)
    self._SummarizeValue(global_step, 'examples/sec', example_rate)
    self._SummarizeValue(global_step, 'total_samples', total_examples)

    for key, (val, _) in sorted(six.iteritems(eval_metrics)):
      self._SummarizeValue(global_step, key, val)

    self._model.GetTask().ProcessFPropResults(sess, global_step, eval_metrics,
                                              outfeeds)
    return False


class EvalProgram(BaseProgram):
  """Evaluation program.

  Note that this currently has different infeed semantics compared to
  the existing Evaler as the input generator is not recreated
  per-eval. Thus different random samples are selected each
  evaluation.
  """

  def __init__(self, params):
    super(EvalProgram, self).__init__(params)
    self._program_name = 'EvalProgram'

  def BuildTpuSubgraph(self):
    tf.logging.info('EvalProgram BuildTpuSubGraph')
    with py_utils.OpportunisticVariableReuseScope(True):
      self._eval_metrics = metrics.TpuEvalMetrics()
      data_parallelism = self.data_parallelism

      self._input = self._task_params.input.Instantiate()
      self._input.CreateTpuEnqueueOps()
      self._task_params.input.Define('skip_create_child', True, '')

      def TpuEvalStep(*args):
        """Eval a shard of a batch on a single TPU core.

        Args:
          *args: metrics values from previous steps.

        Returns:
          Summed eval metrics.
        """
        with cluster_factory.SetEval(True):
          self._model = self._task_params.Instantiate()
          self._task = self._model.GetTask()
          self._task.AddChild('input', self._input)

          self._model.ConstructFPropGraph()
          per_step_eval_metrics = self._eval_metrics.SetMetrics(
              self._model.GetTask().eval_metrics, args)
          summed_metrics = []
          for x, y in zip(per_step_eval_metrics, args):
            summed_metrics.append(x + y)
          return summed_metrics

      @tpu_function.on_device_training_loop
      def TpuEval():
        loop_result = tpu_training_loop.repeat(
            self._steps_per_loop,
            TpuEvalStep,
            inputs=self._eval_metrics.initial_values,
            name='eval_loop')
        # Final metrics are the avg across self._steps_per_loop steps.
        return self._eval_metrics.FinalizeMetrics(loop_result)

      self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
          TpuEval,
          num_shards=data_parallelism,
          device_assignment=py_utils.GetTpuDeviceAssignment())
      # Get metric result from a single replica; they are all same here.
      self.tpu_ops = [[t[0] for t in batch_parallel_res]]

      return self.tpu_ops

  def Run(self, sess):
    gsteps = py_utils.GetGlobalStep()
    global_step = sess.run(gsteps)
    self.SetStatusMessage('Executing eval program at step %d' % global_step)

    infeed_future = self._infeed_pool.apply_async(
        self._InfeedLoop, args=(sess,))
    ary = sess.run(self.tpu_ops)
    infeed_future.wait()
    values = ary[0]
    self._eval_metrics.PackMetricsValues(values)
    for key, (val, _) in sorted(six.iteritems(self._eval_metrics.metrics)):
      self._SummarizeValue(global_step, key, val)
    return False


class DecodeProgram(BaseProgram):
  """DecodeProgram.

  Note that this currently has different infeed semantics compared to
  the existing Decoder as the input generator is not recreated
  per-eval. Thus different random samples are selected each
  decoder run.
  """

  def __init__(self, params):
    super(DecodeProgram, self).__init__(params)
    self._program_name = 'DecodeProgram'

  def _WriteSummaries(self, job_name, global_step, summaries):
    for unused_name, summary in sorted(summaries.items()):
      self._summary_writer.add_summary(summary, global_step)
      if summary.value:
        for value in summary.value:
          if value.HasField('simple_value'):
            tf.logging.info('%s summary on checkpoint@%d %s = %.8g',
                                 job_name, global_step, value.tag,
                                 value.simple_value)
      self._summary_writer.flush()

  def BuildTpuSubgraph(self):
    tf.logging.info('DecodeProgram BuildTpuSubGraph')
    py_utils.ResetStepSeed()

    # Instantiate input generator first.
    self._input = self._task_params.input.Instantiate()
    self._input.CreateTpuEnqueueOps()
    self._task_params.input.Define('skip_create_child', True, '')

    def _DecodeFn():
      """Decode call to be compiled for TPU."""
      with py_utils.OpportunisticVariableReuseScope(True):
        with cluster_factory.SetEval(True):
          self._model = self._task_params.Instantiate()
          self._model_task = self._model.GetTask()
          self._model_task.AddChild('input', self._input)
          input_batch = self._model_task.input_generator.TpuDequeueBatch()
          metrics_dict = self._model_task.Decode(input_batch)
          self.metrics_nm = py_utils.NestedMap(metrics_dict)
          return self.metrics_nm.Flatten()

    self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
        _DecodeFn,
        num_shards=self.data_parallelism,
        device_assignment=py_utils.GetTpuDeviceAssignment())

    self.metrics = py_utils.NestedMap(self.metrics_nm)
    self.metrics = self.metrics.Pack(batch_parallel_res)
    return None

  def Run(self, sess):
    gsteps = py_utils.GetGlobalStep()
    global_step = sess.run(gsteps)
    self.SetStatusMessage('Executing decode program at step %d' % global_step)
    infeed_future = self._infeed_pool.apply_async(
        self._InfeedLoop, args=(sess,))
    dec_metrics = self._model_task.CreateDecoderMetrics()
    start_time = time.time()
    buffered_decode_out = []
    for i in range(self._steps_per_loop):
      metrics_values = sess.run(self.metrics)
      decode_out = self._model_task.PostProcessDecodeOut(
          metrics_values, dec_metrics)
      tf.logging.info('step: %d %f' %
                           (i, dec_metrics['num_samples_in_batch'].total_value))
      if decode_out:
        buffered_decode_out.extend(decode_out)
    infeed_future.wait()

    num_examples_metric = dec_metrics['num_samples_in_batch']
    summaries = {k: v.Summary(k) for k, v in six.iteritems(dec_metrics)}
    elapsed_secs = time.time() - start_time
    example_rate = num_examples_metric.total_value / elapsed_secs
    summaries['examples/sec'] = tf.Summary(
        value=[tf.Summary.Value(tag='examples/sec', simple_value=example_rate)])
    self._WriteSummaries(
        os.path.basename(self._program_dir), global_step, summaries)
    decode_out_path = os.path.join(self._program_dir,
                                   'decoder_out_%09d' % global_step)
    decode_finalize_args = base_model.DecodeFinalizeArgs(
        decode_out_path=decode_out_path, decode_out=buffered_decode_out)
    self._model_task.DecodeFinalize(decode_finalize_args)
    return False


class ExperimentalDecodeProgram(DecodeProgram):
  """DecodeProgram in a tpu loop.

  TODO(huangyp) test this for beam search decoders and replace the
  default DecodeProgram.
  """

  @classmethod
  def Params(cls):
    p = super(ExperimentalDecodeProgram, cls).Params()
    p.num_threads = 2
    return p

  def BuildTpuSubgraph(self):
    tf.logging.info('DecodeProgram BuildTpuSubGraph')
    py_utils.ResetStepSeed()
    device_assignment = py_utils.GetTpuDeviceAssignment()
    self.spmd = self._task_params.input.use_partitioned_infeed_queue
    with py_utils.OpportunisticVariableReuseScope(True):
      with cluster_factory.SetEval(True):
        self._model = self._task_params.Instantiate()
        self._model_task = self._model.GetTask()
        self._model_task.input.CreateTpuEnqueueOps()

        def _DecodeStep():
          """Decode call to be compiled for TPU."""
          input_batch = self._model_task.input_generator.TpuDequeueBatch()
          metrics_dict = self._model_task.Decode(input_batch)
          self.metrics_nm = py_utils.NestedMap(metrics_dict)
          device = tpu.core(0) if self.spmd else ''
          with tf.device(device):
            outfeed_enqueue = tpu_ops.outfeed_enqueue_tuple(
                self.metrics_nm.Flatten())
            return [outfeed_enqueue]

    @tpu_function.on_device_training_loop
    def DecodeLoopFn():
      return tpu_training_loop.repeat(
          self._steps_per_loop, _DecodeStep, inputs=[])

    self._compile_op, self.decode_loop = tpu.split_compile_and_shard(
        DecodeLoopFn,
        num_shards=self.data_parallelism,
        device_assignment=device_assignment)
    # Get a list of outfeed ops.
    self.metrics = self._OutfeedDequeue()
    # Pack the list of outfeed ops with structure in self.metrics_nm.
    self.metrics = tf.nest.pack_sequence_as(self.metrics_nm, self.metrics)
    return

  def _OutfeedDequeue(self):
    """Collect outfeed dequeue from all devices."""
    num_outfeeds = len(self.metrics_nm.Flatten())
    outfeed_ops = [[]] * num_outfeeds
    device_assignment = py_utils.GetTpuDeviceAssignment()
    assert device_assignment
    for replica in range(device_assignment.num_replicas):
      num_cores_per_replica = 1 if self.spmd else (
          device_assignment.num_cores_per_replica)
      for core in range(num_cores_per_replica):
        with tf.device(device_assignment.host_device(replica, core)):
          outfeeds_per_core = tpu_ops.outfeed_dequeue_tuple(
              dtypes=[x.dtype for x in self.metrics_nm.Flatten()],
              shapes=[x.shape for x in self.metrics_nm.Flatten()],
              device_ordinal=device_assignment.tpu_ordinal(replica, core))
          for idx_outfeed, out_feed in enumerate(outfeeds_per_core):
            outfeed_ops[idx_outfeed] = outfeed_ops[idx_outfeed] + [out_feed]
    return [tf.concat(per_outfeed, 0) for per_outfeed in outfeed_ops]

  def _DecodeLoop(self, sess):
    sess.run(self.decode_loop)

  def Run(self, sess):
    gsteps = py_utils.GetGlobalStep()
    global_step = sess.run(gsteps)
    self.SetStatusMessage('Executing decode program at step %d' % global_step)
    infeed_future = self._infeed_pool.apply_async(
        self._InfeedLoop, args=(sess,))
    decode_future = self._infeed_pool.apply_async(
        self._DecodeLoop, args=(sess,))

    dec_metrics = self._model_task.CreateDecoderMetrics()
    start_time = time.time()
    for _ in range(self._steps_per_loop):
      metrics_values = sess.run(self.metrics)
      self._model_task.PostProcessDecodeOut(metrics_values, dec_metrics)
    decode_future.wait()
    infeed_future.wait()
    summaries = {k: v.Summary(k) for k, v in six.iteritems(dec_metrics)}
    elapsed_secs = time.time() - start_time
    num_examples_metric = dec_metrics['num_samples_in_batch']
    example_rate = num_examples_metric.total_value / elapsed_secs
    summaries['examples/sec'] = tf.Summary(
        value=[tf.Summary.Value(tag='examples/sec', simple_value=example_rate)])
    self._WriteSummaries(
        os.path.basename(self._program_dir), global_step, summaries)

    return False


class MLPerfTrainDecodeProgram(BaseProgram):
  """Run train/decode in a single session run."""

  @classmethod
  def Params(cls):
    """"Defaults parameters for Programs."""
    p = super(MLPerfTrainDecodeProgram, cls).Params()
    p.Define('train_task', None, 'Underlying task')
    p.Define('decode_task', None, 'Underlying task')
    p.Define('train_dataset_name', None, '')
    p.Define('decode_dataset_name', None, '')
    p.Define('train_steps_per_loop', 0, '')
    p.Define('decode_steps_per_loop', 0, '')
    p.Define('ml_perf', None, 'MLPerf config')
    return p

  def __init__(self, params):
    super(MLPerfTrainDecodeProgram, self).__init__(params)
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

  def BuildTpuSubgraph(self):
    if self._ml_perf_log:
      mlp_log.mlperf_print('global_batch_size', self._ml_perf.global_batch_size)
      mlp_log.mlperf_print('max_sequence_length',
                           self._ml_perf.max_sequence_length)
      mlp_log.mlperf_print('opt_name', self._ml_perf.optimizer_name)
      mlp_log.mlperf_print('opt_base_learning_rate',
                           self._ml_perf.base_learning_rate)
      mlp_log.mlperf_print('opt_learning_rate_warmup_steps',
                           self._ml_perf.warmup_steps)

    with py_utils.OpportunisticVariableReuseScope(True):
      self._eval_metrics = metrics.TpuEvalMetrics()
      data_parallelism = self.data_parallelism
      self._train_task_params.input.Define('skip_create_child', True, '')
      self._train_input = self._train_task_params.input.Instantiate()
      self._train_input.CreateTpuEnqueueOps()

      def TpuTrainStep():
        """Train a shard of a batch on a single TPU core.

        Do not calculate loss metrics.

        Returns:
         [train_op].
        """
        self._train_model = self._train_task_params.Instantiate()
        self._task = self._train_model.GetTask()
        self._task.AddChild('input', self._train_input)
        self._model = self._train_model
        self._train_model.ConstructFPropBPropGraph()
        return [self._train_model.GetTask().train_op]

      def TpuTrain():
        loop_result = tpu_training_loop.repeat(
            self._train_steps_per_loop,
            TpuTrainStep,
            inputs=[],
            name='train_loop')
        return loop_result

    py_utils.ResetStepSeed()

    self._decode_input = self._decode_task_params.input.Instantiate()
    self._decode_input.CreateTpuEnqueueOps()
    self._decode_task_params.input.Define('skip_create_child', True, '')

    def _DecodeFn():
      """Decode call to be compiled for TPU."""
      with py_utils.OpportunisticVariableReuseScope(True):
        with cluster_factory.SetEval(True):
          self._decode_model = self._decode_task_params.Instantiate()
          self._decode_model_task = self._decode_model.GetTask()
          self._decode_model_task.AddChild('input', self._decode_input)
          input_batch = self._decode_model_task.input_generator.TpuDequeueBatch(
          )
          metrics_dict = self._decode_model_task.Decode(input_batch)
          self.metrics_nm = py_utils.NestedMap(metrics_dict)
          return self.metrics_nm.Flatten()

    @tpu_function.on_device_training_loop
    def TrainAndDecode():
      with tf.control_dependencies([TpuTrain()]):
        return _DecodeFn()

    self._compile_op, batch_parallel_res = tpu.split_compile_and_shard(
        TrainAndDecode,
        num_shards=data_parallelism,
        device_assignment=py_utils.GetTpuDeviceAssignment())

    self.metrics = py_utils.NestedMap(self.metrics_nm)
    self.metrics = self.metrics.Pack(batch_parallel_res)
    return None

  def _InfeedLoop(self, sess):
    tf.logging.info('_InfeedLoop start')
    try:
      for i in range(self._train_steps_per_loop):
        tf.logging.vlog(1, '_InfeedLoop %d', i)
        sess.run(self._train_model.GetTask().input_generator.tpu_infeed_op)
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
        sess.run(self._decode_model.GetTask().input_generator.tpu_infeed_op)
      tf.logging.info('_InfeedLoop done')
    except Exception as e:
      tf.logging.info('_InfeedLoop exception %r %s', e, e)
      raise

  def _TrainAndDecode(self, sess):
    metrics_values = sess.run(self.metrics)
    self._decode_model_task.PostProcessDecodeOut(metrics_values,
                                                 self.dec_metrics)

  def Run(self, sess):
    gsteps = py_utils.GetGlobalStep()
    global_step = sess.run(gsteps)
    self.dec_metrics = self._decode_model_task.CreateDecoderMetrics()
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

    infeed_future.wait()
    train_future.wait()

    if self._ml_perf_log:
      mlp_log.mlperf_print(
          'eval_stop', None, metadata={'epoch_num': (epoch + 1)})
      mlperf_metric = self._ml_perf.decoder_metric_name
      mlperf_metric_value = float(self.dec_metrics[mlperf_metric].value)
      mlp_log.mlperf_print(
          'eval_accuracy', mlperf_metric_value, metadata={'epoch_num': epoch})
      if mlperf_metric_value > self._ml_perf.decoder_metric_success_threshold:
        tf.logging.info('ml_perf_final_threshold: %f exceeded',
                             self._ml_perf.decoder_metric_success_threshold)
        if not self._run_stop:
          self._run_stop = mlp_log.mlperf_print(
              'run_stop', None, metadata={'status': 'success'})
          self.SetStatusMessage('MLPerf run_time: %.2f' %
                                (self._run_stop - self._run_start))
          return True
    return False


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

    # TODO(blee): Clean these up.
    p.Define('ml_perf', hyperparams.Params(), 'MlPerf configuration.')
    mlp = p.ml_perf
    mlp.Define('benchmark_name', None, 'Benchmark name for compliance log.')
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
    return False


def SimpleProgramScheduleForTask(train_dataset_name,
                                 train_steps_per_loop,
                                 eval_dataset_names,
                                 eval_steps_per_loop,
                                 decode_steps_per_loop,
                                 experimental_decoder=False):
  """Convenient helper method for common case.

  Args:
    train_dataset_name: Name of the training dataset, eg: 'Train'
    train_steps_per_loop: Number of steps to execute the training program.
    eval_dataset_names: List of eval dataset_name strings, eg: ['Train'].
    eval_steps_per_loop: Number of steps to execute the eval program.
    decode_steps_per_loop: Number of steps to execute the decode program.
    experimental_decoder: bool. Whether to use experimental deocder which is
      placed in a tpu loop.

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
      decoder = (
          ExperimentalDecodeProgram if experimental_decoder else DecodeProgram)
      decode_program_params = decoder.Params()
      decode_program_params.name = 'decode_tpu'
      # TODO(blee): This should be derived from the Dataset size.
      decode_program_params.steps_per_loop = decode_steps_per_loop
      decode_program_params.dataset_name = dataset_name
      program_schedule_params.eval_programs.append(decode_program_params)

  return program_schedule_params


class MLPerfProgramSchedule(object):
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
               'Benchmark run must exceed this value to succeeed.')
    mlp.Define('steps_per_epoch', None, 'Number of training steps per epoch.')
    mlp.Define('global_batch_size', None, 'Global batch size.')
    mlp.Define('max_sequence_length', None, 'Maximum sequence length.')
    mlp.Define('optimizer_name', None, 'Optimizer used.')
    mlp.Define('base_learning_rate', None, 'Base learning rate.')
    mlp.Define('warmup_steps', None, 'Number of warm-up steps.')

    return p

  def __init__(self, params):
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

    self.train_program = p.train_program.Instantiate()
    self._programs = []
    self._programs.append(self.train_program)

  def Programs(self):
    return self._programs

  def Run(self, sess):
    p = self.params
    for _ in range(p.train_executions_per_eval):
      program_done = self.train_program.Run(sess)
      if program_done:
        return True
    return False


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
