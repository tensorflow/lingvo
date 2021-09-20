# Lint as: python3
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
r"""Trainer.

To run locally:

.. code-block:: bash

  $ bazel build -c opt //lingvo:trainer
  $ bazel-bin/lingvo/trainer --logtostderr \
      --model=image.mnist.LeNet5 --mode=sync --logdir=/tmp/lenet5 \
      --run_locally=cpu

To use GPU, add `--config=cuda` to build command and set `--run_locally=gpu`.
"""
import os
import re
import sys
import threading
import time

from lingvo import base_trial
from lingvo import datasets
from lingvo import executor
from lingvo import model_imports
from lingvo import model_registry
from lingvo import trainer_impl
import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import checkpointer
from lingvo.core import cluster_factory
from lingvo.core import inference_graph_exporter
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

tf.flags.DEFINE_string(
    'model', None, 'Name of the model class to train.'
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
tf.flags.DEFINE_list('additional_worker_jobs', [],
                     'Additional worker job names.')
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
tf.flags.DEFINE_string(
    'input_targets', '', 'Target network addresses for the '
    'input job. E.g., a single ip:port, or a list of '
    'comma-separated grpc://ip:port, etc.')

tf.flags.DEFINE_string('evaler_job', '/job:evaler', 'Job name')
tf.flags.DEFINE_integer('evaler_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_integer('evaler_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_string('decoder_job', '/job:decoder', 'Job name')
tf.flags.DEFINE_integer('decoder_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_integer('decoder_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_string('tf_data_service_address', '',
                       'The address of the tf.data service.')

tf.flags.DEFINE_string(
    'inference_graph_filename', None,
    'Output inference graph filename. If unspecified, output two inference '
    'graphs, one for CPU and one for TPU using the default settings.')
tf.flags.DEFINE_string(
    'inference_graph_device', None,
    'Type of device the output inference graph is for. This flag is applicable '
    'only when FLAGS.inference_graph_filename is specified.')
tf.flags.DEFINE_integer(
    'inference_graph_random_seed', None,
    'Random seed to fix when exporting inference graph. '
    'Not fixed when set to None.')
tf.flags.DEFINE_list(
    'graph_def_filename', [],
    'Output inference graph_def filenames. Defaults to CPU graph if '
    'inference_graph_filename and inference_graph_device are not specified.')
tf.flags.DEFINE_string(
    'inference_dataset_name', 'Test',
    'Name of the dataset whose params to be extracted inference graph with.')
tf.flags.DEFINE_bool(
    'inference_gen_tpu_init_op', True,
    'Whether the tpu_init_op subgraph is generated for TPU inference graph.')

tf.flags.DEFINE_bool(
    'evaler_in_same_address_as_controller', False,
    'Whether or not evaler is in the same address space as '
    'controller. This flag is meant for unittest only.')

tf.flags.DEFINE_string(
    'vizier_reporting_job', 'evaler',
    'Job responsible for reporting metrics. This specifies a '
    'job prefix, evaler will match all evaler jobs, while '
    'evaler_dev and decoder_dev will only match the corresponding '
    'jobs that are on the dev set.')

tf.flags.DEFINE_bool(
    'add_summary', None,
    'Whether we should output summaries. The default value "None", enables '
    'summaries based on the job type.')
tf.flags.DEFINE_bool('disable_tf2', False,
                     'Whether run on Tensorflow without V2 behaviors.')


@tf.flags.validator('vizier_reporting_job')
def _ValidateVizierReportingJob(value):
  if value in ['evaler', 'decoder']:
    return True
  if value.startswith('evaler_') or value.startswith('decoder_'):
    return True
  tf.logging.info('vizier_reporting_job should usually start with evaler or '
                  'decoder, unless in executor/program mode. '
                  f'vizier_reporting_job={value}')
  return True


tf.flags.DEFINE_integer(
    'enqueue_max_steps', None, 'Max enqueue steps. -1 meaning no limit.'
    ' This flag should be set for unit-test only.')

tf.flags.DEFINE_integer('saver_max_to_keep', None,
                        'Maximum number of recent checkpoints to keep.')

tf.flags.DEFINE_float('saver_keep_checkpoint_every_n_hours', None,
                      'How often to keep a checkpoint.')

tf.flags.DEFINE_bool(
    'checkpoint_in_trainer_tpu', False,
    'Whether to enable checkpointing in TrainerTpu, allowing for '
    'operation without a separate Controller task.'
    'This flag also disables checkpointing from the Controller, '
    'but still allows it to write summaries.')

tf.flags.DEFINE_string(
    'tpu', None,
    'The Cloud TPU on GCP to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url. If set, other cluster parameters (such as --cluster_spec) will be '
    'configured automatically with TPUClusterResolver.')
tf.flags.DEFINE_string(
    'gcp_project', None,
    'Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_zone', None,
    'GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Please consider adding model params instead of adding flags.

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


class Controller(base_runner.BaseRunner):
  """Controller for a training cluster."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._job_name = 'controller'
    assert not self._model_task_name, 'Controller needs all tasks!'
    self._control_dir = os.path.join(self._logdir, 'control')
    tf.io.gfile.makedirs(self._control_dir)
    self._checkpoint_in_controller = True
    if FLAGS.checkpoint_in_trainer_tpu:
      self._checkpoint_in_controller = False
      if self._early_stop:
        tf.logging.warning('Controller ignoring early_stop since '
                           'TrainerTpu is driving training.')
        self._early_stop = None

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._summary_writer = self._CreateSummaryWriter(self._control_dir)
        self._model = self.params.Instantiate()
        self._params = self._model.params
        self._model.ConstructFPropBPropGraph()
        self._summary_op = tf.summary.merge_all()
        self._initialize_tables = tf.tables_initializer()
        self._initialize_local_vars = tf.local_variables_initializer()
        self._initialize_global_vars = tf.global_variables_initializer()
        self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
        if self._checkpoint_in_controller:
          self.checkpointer = self._CreateCheckpointer(
              self._train_dir,
              self._model,
              init_op=self._initialize_global_vars)

    self._ExportMetrics(params=self.params)
    self._model_analysis, self._total_num_params = summary_utils.ModelAnalysis(
        self._model)
    py_utils.LogMultiLines('MODEL ANALYSIS', self._model_analysis)
    self._WriteToLog(self._model_analysis, self._control_dir,
                     'model_analysis.txt')
    self._WriteToLog(self.params.ToText(), self._control_dir, 'params.txt')
    self._WriteToLog(
        text_format.MessageToString(self.params.ToProto(), as_utf8=True),
        self._control_dir, 'params.pbtxt')

  def _CreateCheckpointer(self, train_dir, model, init_op=None):
    """Wrapper method for override purposes."""
    return checkpointer.Checkpointer(train_dir, model, init_op)

  def Start(self):
    self._RunLoop('controller', self._Loop)

  def StartEnqueueOp(self, op):
    self._RunLoop(
        'controller/enqueue_op/%s' % op.name, self._LoopEnqueue, loop_args=[op])

  def _Loop(self):
    self._summary_writer.add_graph(self._graph)
    with tf.container(self._container_id), self._GetSession() as sess:
      if FLAGS.interactive:
        # Into interactive debugging mode.
        _StartShell(locals())
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
          self.checkpointer.RestoreIfNeeded(sess)

        global_step = sess.run(self._model.global_step)
        if self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          if self._checkpoint_in_controller:
            self.checkpointer.Save(sess, global_step)
          sess.close()
          self._DequeueThreadComplete()
          return

        if self._checkpoint_in_controller:
          # Checkpoint if it's time.
          self.checkpointer.MaybeSave(sess, global_step)

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


Trainer = trainer_impl.Trainer


class TrainerTpu(base_runner.BaseRunner):
  """Trainer on TPU."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
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

    with self._graph.as_default(), tf.container(self._container_id):
      self._summary_writer = self._CreateSummaryWriter(self._train_dir)
      self._CreateTF2SummaryWriter(self._train_dir)
      with self._cluster, tf.device(
          self._cluster.GetPlacer()), self._TF2SummaryContext():
        self._model = self.params.Instantiate()
        self._task = self._model.GetTask()
        self._task.input.CreateTpuEnqueueOps()
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
        self._task.input.CreateTpuEmbeddingEnqueueOps()

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

      self._initialize_local_vars = tf.local_variables_initializer()
      self._initialize_global_vars = tf.global_variables_initializer()
      self._initialize_tables = tf.tables_initializer()

      if FLAGS.checkpoint_in_trainer_tpu:
        self.checkpointer = checkpointer.Checkpointer(
            self._train_dir, self._model, init_op=self._initialize_global_vars)

      self.enqueue_ops = self._task.input.tpu_infeed_op
      tf.logging.info('Trainer number of enqueue ops: %d',
                      len(self.enqueue_ops))

    if self._task.input.input_data_summary_layout is not None:
      self._summary_writer.add_summary(
          self._task.input.input_data_summary_layout)

    if FLAGS.checkpoint_in_trainer_tpu:
      self._model_analysis, self._total_num_params = (
          summary_utils.ModelAnalysis(self._model))
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

  def _CleanUp(self):
    # If there's an exception, we want _LoopEnqueue to wait until
    # everything is initialized before starting up.
    self._initialized.clear()

  def Start(self):
    # Run training.
    self._RunLoop('trainer', self._Loop, cleanup_func=self._CleanUp)

  def _InfeedLoop(self, sess):
    tf.logging.info('_InfeedLoop start')
    for _ in range(self._steps_per_loop):
      sess.run(self.enqueue_ops)

  def StartEnqueueOp(self, op):
    # When retrieve ops for TPU embedding is present, we use _InfeedLoop above
    # instead to make sure enqueue and retrieve does not happen at the same
    # time as required by TPU embedding.
    # We can remove this by using a tf.while_loop driven infeed op.
    if self._retrieve_ops:
      return
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
      self.checkpointer.RestoreGlobalStepIfNeeded(sess)

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
        self.checkpointer.Restore(sess, force_reinitialize=True)

      global_step = sess.run(self._model.global_step)
      self._initialized.set()
      eval_metrics = None
      if FLAGS.checkpoint_in_trainer_tpu and global_step == 0:
        # Always save a ckpt at step 0.
        self.checkpointer.MaybeSave(sess, global_step)

      sess.run(self._load_ops)
      while True:
        train_steps_start = time.perf_counter()
        if FLAGS.checkpoint_in_trainer_tpu:
          # Init/restore variable if needed.
          self.checkpointer.RestoreIfNeeded(sess)

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
          if self._max_steps_for_early_stop is None:
            self._max_steps_for_early_stop = global_step + 3 * self._steps_per_loop
            tf.logging.info('Early stopping at step: %d',
                            self._max_steps_for_early_stop)

        if self._ShouldStop(sess, global_step, check_early_stop=False):
          tf.logging.info('Training finished.')
          if FLAGS.checkpoint_in_trainer_tpu:
            self.checkpointer.Save(sess, global_step)
          self._DequeueThreadComplete()
          return

        if self._retrieve_ops:
          infeed_loop_thread = threading.Thread(
              target=self._InfeedLoop, args=(sess,))
          infeed_loop_thread.start()

        tpu_train_op_start = time.perf_counter()
        values, outfeeds = sess.run(self._tpu_train_ops)
        tpu_train_op_secs = time.perf_counter() - tpu_train_op_start

        if self._retrieve_ops:
          infeed_loop_thread.join()
          tf.logging.info('Retrieve params.')
          sess.run(self._retrieve_ops)
          tf.logging.info('Retrieve params done.')

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
        if FLAGS.checkpoint_in_trainer_tpu:
          checkpoint_write_start = time.perf_counter()
          checkpoint_saved = self.checkpointer.MaybeSave(sess, global_step)
          if checkpoint_saved:
            checkpoint_write_secs = time.perf_counter() - checkpoint_write_start
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
        self.checkpointer = self._CreateCheckpointer(self._train_dir,
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

  def _CreateCheckpointer(self, train_dir, model):
    """Wrapper method for override purposes."""
    return checkpointer.Checkpointer(train_dir, model)

  def Start(self):
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
      self.checkpointer.RestoreFromPath(sess, path)

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


Decoder = trainer_impl.Decoder
GetDecoderDir = trainer_impl.GetDecoderDir


def _GetClusterSpecDict():
  """Parses the cluster_spec flag and returns a dict."""
  job_specs = FLAGS.cluster_spec.split('@')
  cluster_spec_dict = {}
  for job_spec in job_specs:
    # ps_host=worker1:1231,worker2:1234
    job_machines = job_spec.split('=')
    if len(job_machines) != 2:
      raise ValueError(f'Invalid job specification: {job_spec}')
    cluster_spec_dict[job_machines[0]] = job_machines[1].split(',')

  return cluster_spec_dict


class RunnerManager:
  """Helper class for managing runners."""

  # This is a hack so these classes can be overridded with internal
  # non-public implementations.
  # pylint: disable=invalid-name
  inference_graph_exporter = inference_graph_exporter
  model_registry = model_registry
  Controller = Controller
  Trainer = Trainer
  TrainerTpu = TrainerTpu
  Evaler = Evaler
  Decoder = Decoder
  ExecutorTpu = executor.ExecutorTpu

  # pylint: enable=invalid-name

  def __init__(self, model):
    self._model_name = model

  def MaybeLaunchTensorFlow(self):
    """Starts TF machinery in this process."""
    if FLAGS.run_locally or FLAGS.tpu:
      return

    tf.logging.info('Launching tensorflow.')

    target = FLAGS.tf_master
    if not target.startswith('localhost'):
      # E.g., trainer_client is configured w/ FLAGS.tf_master pointing to
      # another job. In that case, start a local server.
      cluster_spec_dict = _GetClusterSpecDict()
      self._tf_server = tf.distribute.Server(
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

  def GetExecutorParams(self):
    """Get the params needed to instantiate the ExecutorTpu.

    Returns:
       Tuple (dict, params):

         - ps_params_dict: high_level task_name -> ProgramScheduleParams
         - train_cfg: Either a SingleTaskModelParams or MultiTaskModelParams.
    """
    cluster = cluster_factory.Current()
    self.UpdateClusterParamsFromFlags(cluster.params, 'executor_tpu')
    ps_params_dict, train_cfg = executor.GetExecutorParams(
        self._model_name, cluster.params, self.model_registry)

    return ps_params_dict, train_cfg

  def GetParamsForDataset(self, job_name, dataset_name):
    """Returns params for job `job_name` on the dataset `dataset_name`."""
    # Get the current cluster and update its params from flags.
    cluster = cluster_factory.Current()
    self.UpdateClusterParamsFromFlags(cluster.params, job_name)
    with cluster_factory.Cluster(cluster.params):
      try:
        cfg = self.model_registry.GetParams(self._model_name, dataset_name)
      except base_model_params.DatasetError as e:
        dataset_name_retry = dataset_name.title()
        tf.logging.warning(
            'Exception configuring dataset %s, retrying as %s: %s',
            dataset_name, dataset_name_retry, e)
        cfg = self.model_registry.GetParams(self._model_name,
                                            dataset_name_retry)
        tf.logging.warning('Succeeded after retrying as %s.' %
                           dataset_name_retry)
    cfg.cluster = cluster.params

    # Updates a few params based on flags.
    if FLAGS.enqueue_max_steps is not None:
      cfg.train.enqueue_max_steps = FLAGS.enqueue_max_steps
    if FLAGS.saver_max_to_keep is not None:
      cfg.train.save_max_to_keep = FLAGS.saver_max_to_keep
    if FLAGS.saver_keep_checkpoint_every_n_hours is not None:
      cfg.train.save_keep_checkpoint_every_n_hours = FLAGS.saver_keep_checkpoint_every_n_hours
    return cfg

  def MaybeConfigRunDistributed(self):
    """If given a `FLAGS.cluster_spec`, update flags for running distributed."""
    if not FLAGS.cluster_spec:
      return
    job_specs = FLAGS.cluster_spec.split('@')
    cluster_spec_dict = _GetClusterSpecDict()
    if FLAGS.job == 'trainer_client':
      FLAGS.tf_master = 'grpc://%s' % cluster_spec_dict['worker'][FLAGS.task]
    for job in cluster_spec_dict:
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
                                              'worker', 'executor_tpu'):
      FLAGS.worker_job = '/job:worker'
      FLAGS.worker_replicas = len(cluster_spec_dict['worker'])
      FLAGS.ps_job = '/job:worker'
      FLAGS.ps_replicas = FLAGS.worker_replicas
    if FLAGS.mode == 'async' and FLAGS.job in ('controller', 'trainer', 'ps'):
      FLAGS.worker_job = '/job:trainer'
      FLAGS.worker_replicas = len(cluster_spec_dict['trainer'])
      FLAGS.ps_job = '/job:ps'
      FLAGS.ps_replicas = len(cluster_spec_dict['ps'])

  def MaybeConfigCloudTpu(self):
    """If given `FLAGS.tpu`, update flags for running on a Cloud TPU."""
    if not FLAGS.tpu:
      return

    if not FLAGS.job:
      FLAGS.job = 'trainer_client'

    if FLAGS.job not in ('trainer_client', 'executor_tpu'):
      raise ValueError('Only trainer_client and executor_tpu jobs are '
                       'supported on TPU.')

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu,
        project=FLAGS.gcp_project,
        zone=FLAGS.tpu_zone,
        job_name=FLAGS.job)
    cluster_spec_dict = cluster_resolver.cluster_spec().as_dict()

    FLAGS.mode = 'sync'
    FLAGS.tf_master = cluster_resolver.master()

    FLAGS.worker_job = '/job:{}'.format(FLAGS.job)
    FLAGS.worker_replicas = 1
    FLAGS.worker_num_tpu_hosts = len(cluster_spec_dict[FLAGS.job])
    FLAGS.worker_tpus = (
        cluster_resolver.num_accelerators()['TPU'] * FLAGS.worker_num_tpu_hosts)
    FLAGS.ps_job = FLAGS.worker_job
    if FLAGS.job == 'trainer_client':
      FLAGS.ps_replicas = FLAGS.worker_replicas

    FLAGS.cluster_spec = ('@'.join('{}={}'.format(job, ','.join(hosts))
                                   for job, hosts in cluster_spec_dict.items()))

    FLAGS.xla_device = 'tpu'
    FLAGS.enable_asserts = False
    FLAGS.checkpoint_in_trainer_tpu = True

  def UpdateClusterParamsFromFlags(self, cluster, job_name):
    """Update `cluster` with a training cluster configuration from flags."""
    cluster.mode = FLAGS.mode
    cluster.job = job_name
    cluster.task = FLAGS.task
    cluster.do_eval = job_name in ['evaler', 'decoder']
    cluster.logdir = FLAGS.logdir

    cluster.controller.name = FLAGS.controller_job
    cluster.controller.gpus_per_replica = FLAGS.controller_gpus

    cluster.worker.name = FLAGS.worker_job
    cluster.worker.replicas = FLAGS.worker_replicas
    cluster.worker.gpus_per_replica = FLAGS.worker_gpus
    cluster.worker.tpus_per_replica = FLAGS.worker_tpus
    cluster.worker.num_tpu_hosts = FLAGS.worker_num_tpu_hosts
    cluster.worker.devices_per_split = FLAGS.worker_split_size
    if FLAGS.additional_worker_jobs:
      for additional_job in FLAGS.additional_worker_jobs:
        cluster.worker.additional_worker_names.append(additional_job)

    if FLAGS.tpu:
      job_name = cluster.worker.name.replace('/job:', '', 1)
      worker_hosts = _GetClusterSpecDict()[job_name]
      if FLAGS.additional_worker_jobs:
        for additional_job in cluster.worker.additional_worker_names:
          additional_job_name = additional_job.replace('/job:', '', 1)
          worker_hosts.extend(_GetClusterSpecDict()[additional_job_name])
      cluster.worker.targets = ','.join(
          'grpc://{}'.format(host) for host in worker_hosts)

    cluster.ps.name = FLAGS.ps_job
    cluster.ps.replicas = FLAGS.ps_replicas
    cluster.ps.gpus_per_replica = FLAGS.ps_gpus

    cluster.input.name = FLAGS.input_job
    cluster.input.replicas = FLAGS.input_replicas
    cluster.input.targets = FLAGS.input_targets

    cluster.evaler.name = FLAGS.evaler_job
    cluster.evaler.replicas = FLAGS.evaler_replicas
    cluster.evaler.gpus_per_replica = FLAGS.evaler_gpus

    cluster.decoder.name = FLAGS.decoder_job
    cluster.decoder.replicas = FLAGS.decoder_replicas
    cluster.decoder.gpus_per_replica = FLAGS.decoder_gpus

    cluster.tf_data_service_address = FLAGS.tf_data_service_address

    cluster.add_summary = FLAGS.add_summary
    cluster.reporting_job = FLAGS.vizier_reporting_job

  def _CreateRunner(self, job, model_task_name, logdir, tf_master, trial):
    """Create a runner."""
    evaler_job_name_prefix = 'evaler_'
    decoder_job_name_prefix = 'decoder_'

    tf.logging.info('Job %s start', job)
    common_args = (model_task_name, logdir, tf_master, trial)
    if job == 'controller':
      cfg = self.GetParamsForDataset('controller', 'Train')
      cfg.cluster.xla_device = 'cpu'
      return self.Controller(cfg, *common_args)
    elif job == 'trainer':
      cfg = self.GetParamsForDataset('trainer', 'Train')
      return self.Trainer(cfg, *common_args)
    elif job == 'trainer_client':
      cfg = self.GetParamsForDataset('trainer_client', 'Train')
      if py_utils.use_tpu():
        cfg.cluster.xla_device = 'tpu'
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
    elif job == 'executor_tpu':
      ps_cfg_dict, train_cfg = self.GetExecutorParams()
      return self.ExecutorTpu(train_cfg, ps_cfg_dict, *common_args)
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
    is_training = 'trainer' in jobs or 'trainer_client' in jobs
    for j in jobs:
      tf_master = FLAGS.tf_master
      # Ensure that decoder or evaler threads do not clobber variables being
      # updated by trainer by forcing them to use independent sessions.
      if (is_training and (j.startswith('decoder') or j.startswith('evaler'))):
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
      runner_class_name = str(runner)
      t = threading.Thread(target=runner.Start, name=runner_class_name)
      t.daemon = True
      t.start()
      threads.append(t)
      if runner.enqueue_ops:
        tf.logging.info('Total num runner.enqueue_ops: %d',
                        len(runner.enqueue_ops))
        for i, enqueue_op in enumerate(runner.enqueue_ops):

          def StartEnqueue(runner, op):
            tf.logging.info('Starting enqueue op %s', op.name)
            return lambda: runner.StartEnqueueOp(op)

          enqueue_name = '%s-enqueue-%d' % (runner_class_name, i)
          tq = threading.Thread(
              target=StartEnqueue(runner, enqueue_op), name=enqueue_name)
          tq.start()
          threads.append(tq)
    tf.logging.info('Waiting for runners to finish...')
    for t in threads:
      tf.logging.info('Waiting for thread to finish: %s' % t.name)
      while True:
        t.join(1)
        if not t.is_alive():
          break
    tf.logging.info('All runners done.')

  def RunTrial(self, job, logdir, trial):
    """A wrapper function for running a trial."""
    # Run each job in separate process/task
    # TODO(rpang): add support for running evaler_test and decoder.
    self.StartRunners(self.CreateRunners([job], logdir, trial))

  def MaybeConfigRunLocally(self):
    """Update flags if configured to run locally."""
    if not FLAGS.run_locally:
      # Do nothing
      return

    FLAGS.tf_master = tf.distribute.Server.create_local_server().target

    if not FLAGS.mode:
      FLAGS.mode = 'sync'

    if not FLAGS.job:
      if FLAGS.run_locally == 'tpu':
        FLAGS.job = 'trainer_client'
      elif FLAGS.mode == 'async':
        FLAGS.job = 'controller,trainer'
      else:
        FLAGS.job = 'controller,trainer_client'

    FLAGS.task = 0
    local_job = '/job:localhost'
    FLAGS.controller_job = local_job

    FLAGS.worker_job = local_job
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

    FLAGS.ps_job = local_job
    FLAGS.ps_replicas = 1
    FLAGS.ps_gpus = 0

    FLAGS.input_job = local_job
    FLAGS.input_replicas = 0

    FLAGS.evaler_job = local_job
    FLAGS.evaler_replicas = 1
    if FLAGS.run_locally == 'gpu':
      FLAGS.evaler_gpus = 1
    else:
      FLAGS.evaler_gpus = 0

    FLAGS.decoder_job = local_job
    FLAGS.decoder_replicas = 1
    if FLAGS.run_locally == 'gpu':
      FLAGS.decoder_gpus = 1
    else:
      FLAGS.decoder_gpus = 0

  def InspectParams(self):
    r"""Print out all the params.

    An example to run this mode:

    bazel-bin/lingvo/trainer --logtostderr \
      --model=image.mnist.LeNet5 --mode=inspect_params --logdir=/tmp/lenet5 \
      --run_locally=cpu
    """
    FLAGS.mode = 'sync'
    cls = self.model_registry.GetClass(self._model_name)
    tf.io.gfile.makedirs(FLAGS.logdir)
    for dataset in datasets.GetDatasets(cls):
      p = self.GetParamsForDataset('controller', dataset)
      outf = os.path.join(FLAGS.logdir, dataset.lower() + '-params.txt')
      tf.logging.info('Write all params for {} to {}'.format(dataset, outf))
      with tf.io.gfile.GFile(outf, 'w') as f:
        f.write(p.ToText())

  def InspectModel(self):
    """Prints out model analysis for the model."""
    FLAGS.mode = 'sync'
    p = self.GetParamsForDataset('controller', 'Train')
    c = cluster_factory.Cluster(p.cluster)
    with tf.Graph().as_default(), c, tf.device(c.GetPlacer()):
      analysis, _ = summary_utils.ModelAnalysis(p.Instantiate())
    print(analysis)

  def InspectDatasets(self):
    """Prints out datasets configured for the model."""
    cls = self.model_registry.GetClass(self._model_name)
    print(','.join([dataset.lower() for dataset in datasets.GetDatasets(cls)]))

  def InspectDecoder(self):
    """Prints out datasets configured for the decoder."""
    cls = self.model_registry.GetClass(self._model_name)
    params = cls()

    has_decoder = False
    if issubclass(cls, base_model_params.SingleTaskModelParams):
      has_decoder = params.Task(
      ).cls.CreateDecoderMetrics != base_model.BaseTask.CreateDecoderMetrics
    else:
      for _, task_param in params.Model().task_params.IterParams():
        has_decoder |= (
            task_param.cls.CreateDecoderMetrics !=
            base_model.BaseTask.CreateDecoderMetrics)
    if has_decoder:
      # We assume that the proper decoder is implemented.
      self.InspectDatasets()
    else:
      print('')

  def SetModelName(self, model_name):
    """Sets the model name."""
    self._model_name = model_name

  def WriteInferenceGraph(self, cfg=None, prune_graph=True):
    """Generates the inference graphs for a given model.

    Args:
      cfg: Full `~.hyperparams.Params` for the model class. If present,
        this cfg will be used instead of retrieving from model_registry.
      prune_graph: If true, prune the graph to just the parts we need.

    Returns:
      InferenceGraph proto for cpu.
    """
    inference_graph_dir = os.path.join(FLAGS.logdir, 'inference_graphs')
    tf.io.gfile.makedirs(inference_graph_dir)
    tf.logging.info('Writing inference graphs to dir: %s', inference_graph_dir)

    if not cfg:
      cfg = self.model_registry.GetParams(self._model_name,
                                          FLAGS.inference_dataset_name)

    task_names = [FLAGS.model_task_name]
    if (issubclass(cfg.cls, base_model.MultiTaskModel) and
        not FLAGS.model_task_name):
      task_names = base_model.MultiTaskModel.TaskNames(cfg)

    inference_graph_proto = None

    if FLAGS.inference_graph_filename:
      # Custom inference graph.
      for task_name in task_names:
        filename_prefix = FLAGS.inference_graph_filename
        if task_name:
          filename_prefix = '%s_inference' % task_name
        filename_prefix = os.path.join(inference_graph_dir, filename_prefix)

        device = ''
        var_options = None
        if FLAGS.inference_graph_device == 'tpu':
          device = 'tpu'
          var_options = 'ON_DEVICE'
        device_options = inference_graph_exporter.InferenceDeviceOptions(
            device=device,
            retain_device_placement=False,
            var_options=var_options,
            gen_init_op=FLAGS.inference_gen_tpu_init_op,
            dtype_override=None,
            fprop_dtype_override=None)
        inference_graph_proto = (
            self.inference_graph_exporter.InferenceGraphExporter.Export(
                model_cfg=cfg,
                model_task_name=task_name,
                device_options=device_options,
                export_path=filename_prefix + '.pbtxt',
                random_seed=FLAGS.inference_graph_random_seed,
                prune_graph=prune_graph))
    else:
      for task_name in task_names:
        filename_prefix = 'inference'
        if task_name:
          filename_prefix = '%s_inference' % task_name
        filename_prefix = os.path.join(inference_graph_dir, filename_prefix)

        # Standard inference graph.
        try:
          inference_graph_proto = (
              self.inference_graph_exporter.InferenceGraphExporter.Export(
                  model_cfg=cfg,
                  model_task_name=task_name,
                  export_path=filename_prefix + '.pbtxt',
                  random_seed=FLAGS.inference_graph_random_seed,
                  prune_graph=prune_graph))
        except NotImplementedError as e:
          tf.logging.error('Cannot write inference graph: %s', e)

        # TPU inference graph. Not all models support it so fail silently.
        try:
          device_options = self.inference_graph_exporter.InferenceDeviceOptions(
              device='tpu',
              retain_device_placement=False,
              var_options='ON_DEVICE',
              gen_init_op=FLAGS.inference_gen_tpu_init_op,
              dtype_override=None,
              fprop_dtype_override=None)
          self.inference_graph_exporter.InferenceGraphExporter.Export(
              model_cfg=cfg,
              model_task_name=task_name,
              device_options=device_options,
              export_path=filename_prefix + '_tpu.pbtxt',
              random_seed=FLAGS.inference_graph_random_seed,
              prune_graph=prune_graph)
        except Exception as e:  # pylint: disable=broad-except
          tf.logging.error('Error exporting TPU inference graph: %s' % e)

    if FLAGS.graph_def_filename and inference_graph_proto:
      for graph_def_filename in FLAGS.graph_def_filename:
        tf.logging.info('Writing graphdef: %s', graph_def_filename)
        dir_path = os.path.dirname(graph_def_filename)
        if (not tf.io.gfile.exists(dir_path) or
            not tf.io.gfile.isdir(dir_path)):
          tf.io.gfile.makedirs(dir_path)
        with tf.io.gfile.GFile(graph_def_filename, 'w') as f:
          f.write(text_format.MessageToString(inference_graph_proto.graph_def))

    return inference_graph_proto

  def RunEvalerOnce(self):
    """Run once evaler."""
    m = re.match(r'evaler_once_([^_@]+)@(\d+)', FLAGS.job)
    dataset_name, ckpt_id = m.group(1), int(m.group(2))
    cfg = self.GetParamsForDataset('evaler', dataset_name)
    evaler = self.Evaler(dataset_name.lower(), cfg, FLAGS.model_task_name,
                         FLAGS.logdir, FLAGS.tf_master)
    evaler.EvalCheckpoint(ckpt_id)

  def Start(self):
    """Start the process."""
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('tf_api_version: %s', tf.summarize_tf2_status())

    if FLAGS.mode == 'inspect_params':
      self.InspectParams()
      return

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

    if FLAGS.mode == 'shell':
      _StartShell(locals())
      return

    assert FLAGS.mode in ['sync', 'async']

    self.MaybeConfigRunLocally()
    self.MaybeConfigRunDistributed()
    self.MaybeConfigCloudTpu()
    self.MaybeLaunchTensorFlow()

    if FLAGS.job.startswith('evaler_once_'):
      # E.g., trainer --model=foo.bar.Model --logdir=...
      # --run_locally=cpu --mode=sync --job=evaler_once_test@65200
      self.RunEvalerOnce()
      return

    self.StartRunners(self.CreateRunners(FLAGS.job.split(','), FLAGS.logdir))


def main(unused_argv):
  RunnerManager(FLAGS.model).Start()


if __name__ == '__main__':
  py_utils.SetEagerMode(False)
  tf.flags.mark_flag_as_required('model')
  FLAGS(sys.argv, known_only=True)
  if FLAGS.disable_tf2:
    tf.disable_v2_behavior()
  model_imports.ImportParams(FLAGS.model)
  FLAGS.unparse_flags()
  tf.app.run(main)
