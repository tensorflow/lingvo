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
"""Base class for all jobs."""

import contextlib
import os
import time
import traceback

from lingvo import base_trial
import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import early_stop
from lingvo.core import py_utils


tf.flags.DEFINE_bool('disable_tf2_summary', True,
                     'If True, disables TF2 summary writing.')

FLAGS = tf.flags.FLAGS


class BaseRunner:
  """Base class for all jobs."""

  def __init__(self,
               params,
               model_task_name,
               logdir,
               tf_master,
               trial=base_trial.NoOpTrial()):
    """Construct a new BaseRunner.

    Args:
      params:  Params object containing model configuration.
      model_task_name:  String name of the task this runner should execute for
        multitask models only.  See flag for details.
      logdir:  String path to the log directory to output to.
      tf_master:  String path to the master job, e.g. 'local'.
      trial:   An optional hyperparameter trial. Used by Vizier studies.
    """
    p = params.Copy()
    # Set in subclasses.
    self._job_name = ''
    self._daemon = False
    self._verbose_enqueue_logging = False

    self._params = trial.OverrideModelParams(p)
    tf.logging.info('=' * 60)
    for line in self.params.ToText().split('\n'):
      tf.logging.info('%s', line)
    tf.logging.info('=' * 60)

    self._logdir = logdir
    self._tf_master = tf_master
    self._model_task_name = model_task_name
    self._trial = trial
    # If the runner is conducting a Vizier trial, scope all the variables
    # (e.g., global_step) by the trial id so that we do not share states across
    # trials.
    self._container_id = self._trial.Name()
    self._should_report_metrics = False

    # To early terminate a runner, we set max_steps here and that will trigger
    # appropriate ShouldStop behavior in the threads. This is used by Vizier
    # to early stop a trial and also EarlyStop to stop training based on
    # metrics.
    self._max_steps_for_early_stop = None

    self._train_dir = os.path.join(self._logdir, 'train')
    tf.io.gfile.makedirs(self._train_dir)
    if py_utils.IsEagerMode():
      self._graph = None
    else:
      self._graph = tf.Graph()
    self._summary_writer = None
    self._initialize_tables = None
    self._dequeue_thread_complete = False
    self.params.cluster.logdir = logdir

    early_stop.MetricHistory.SetLogdirInMetricHistories(p, logdir)
    # The actual EarlyStop object.
    self._early_stop = None
    if p.train.early_stop and p.train.early_stop.window:
      self._early_stop = p.train.early_stop.Instantiate()
      self._verbose_enqueue_logging = True

    self._SetStatusMessage('Starting ...')
    self._cluster = cluster_factory.Cluster(self.params.cluster)
    self._worker_cluster_def = self._cluster.worker_cluster_def
    if py_utils.IsEagerMode():
      self._cluster.InitDevicesEager()
    else:
      self._cluster.InitDevices(self._GetSession())

    # Merged TF scalar summaries for training related input data stats.
    self._merged_input_data_summary_op = None

  @property
  def params(self):
    return self._params

  def _InVizierStudy(self):
    return not isinstance(self._trial, base_trial.NoOpTrial)

  def _FormatStatusMessage(self, message, retrying):
    if self._trial.Name():
      message = 'Trial:{} {}'.format(self._trial.Name(), message)
    if retrying:
      message = f'Job {self._job_name}: <b>Retrying as expected</b>\n{message}'
    return message

  def _SetStatusMessage(self, message, retrying=False):
    """Update the status message for this task."""
    tf.logging.info(self._FormatStatusMessage(message, retrying))

  def _UpdateEarlyStopMetric(self, jobname, global_step, metric_name,
                             metric_value):
    if not self._early_stop:
      return

    if (self._early_stop.params.logging_interval and
        global_step % self._early_stop.params.logging_interval == 0):
      early_stop.MetricHistory.ConditionalAppend(
          os.path.join(self._logdir, jobname), metric_name, global_step,
          metric_value)

  def _ShouldEarlyStop(self, sess=None):
    return self._early_stop and self._early_stop.Stop(sess)

  def _ShouldStop(self, sess=None, step=None, check_early_stop=True):
    """Check if the runner should stop.

    Args:
      sess: tf.Session.
      step: The current GlobalStep.
      check_early_stop: Whether or not we want to check the EarlyStop condition.
        In TPU-training, we don't want to check this at the every step
        granularity in the enqueue thread, as this may starve the TPU training
        loop which by default operates at the 1000 steps granularity.

    Returns:
      Whether runner should stop.
    """
    if step is None:
      if py_utils.IsEagerMode():
        step = py_utils.GetGlobalStep().numpy
      else:
        step = sess.run(py_utils.GetGlobalStep())

    if step >= self.params.train.max_steps:
      tf.logging.info('ShouldStop: step:%6d params.train.max_steps:%6d', step,
                      self.params.train.max_steps)
      return True

    if self._max_steps_for_early_stop and step >= self._max_steps_for_early_stop:
      tf.logging.info('ShouldStop: step:%6d _max_steps_for_early_stop:%6d',
                      step, self._max_steps_for_early_stop)
      return True

    if check_early_stop and self._ShouldEarlyStop(sess):
      tf.logging.info('ShouldStop: Early stopping.')
      return True

    if self._trial.ShouldStop():
      tf.logging.info('ShouldStop: Trial finished.')
      return True

    return False

  def _WriteToLog(self, text, logdir, filename):
    """Logs `text` and saves it under `logdir/filename`."""
    with tf.io.gfile.GFile(os.path.join(logdir, filename), 'w') as f:
      f.write(text)

    if self._summary_writer is not None:
      # Force newlines to be rendered correctly by Markdown.
      text = text.replace('\n', '  \n')
      self._summary_writer.add_summary(
          tf.Summary(value=[
              tf.Summary.Value(
                  tag=filename, tensor=tf.make_tensor_proto([text]))
          ]))

  def _WaitUntilInit(self, sess=None, start_up_delay_steps=None):
    """Wait until the model is ready."""
    # Wait a fix amount of time at start.
    time.sleep(30)

    @py_utils.Retry(retry_value=(tf.errors.FailedPreconditionError,))
    def RetryLoop():
      try:
        if py_utils.IsEagerMode():
          global_step = py_utils.GetGlobalStep().numpy
        else:
          global_step = sess.run(py_utils.GetGlobalStep())
      except tf.errors.FailedPreconditionError as e:
        tf.logging.info('%s: Probably the expected race on global_step: %s',
                        self._job_name, e)
        raise
      msg = 'step:%6d' % global_step
      self._SetStatusMessage(msg)
      if start_up_delay_steps:
        if global_step < start_up_delay_steps:
          msg = 'global step (%d) has not reached start up delay steps (%d)' % (
              global_step, self._start_up_delay_steps)
          tf.logging.info('%s: %s', self._job_name, msg)
          raise tf.errors.FailedPreconditionError(
              node_def=None, op=None, message=msg)
      return global_step

    return RetryLoop()

  @py_utils.Retry(
      initial_delay_sec=1, delay_growth_factor=1.5, max_delay_sec=300)
  def _FindNewCheckpoint(self, sess=None, processed_ckpts=None):
    """Returns the path to a new checkpoint, or raises RuntimeError."""
    if self._ShouldStop(sess, step=0):
      # Check for early stopping or trail early stopping.
      return None
    path = tf.train.latest_checkpoint(self._train_dir)
    if not path:
      msg = 'No check point is found in %s' % self._train_dir
      tf.logging.info('%s: %s', self._job_name, msg)
      raise RuntimeError(msg)
    if path in processed_ckpts:
      msg = 'No new check point is found: %s' % path
      tf.logging.info('%s: %s', self._job_name, msg)
      raise RuntimeError(msg)
    return path

  @py_utils.Retry()
  def _RunLoop(self, job_name, loop_func, loop_args=(), cleanup_func=None):
    """Runs `loop_func`, retrying on expected errors.

    Args:
      job_name: string job name.
      loop_func: callable to run and retry on expected errors.
      loop_args: list or tuple of arguments to be passed to the loop_func.
      cleanup_func: callable to run before retry.
    """
    try:
      tf.logging.info('%s started.', job_name)
      loop_func(*loop_args)
      tf.logging.info('%s done.', job_name)

      if self._daemon:
        # In daemon mode, an external scheduler will retry the job on 0 status.
        # So exit with a non-zero status to prevent retry.
        self._SetStatusMessage(
            '%s completed successfully. Exiting with FAILURE to prevent retry.'
            % job_name)
        time.sleep(300)  # Wait a bit for other threads to complete.
        os._exit(4)  # pylint: disable=protected-access

      return
    except Exception as e:  # pylint:disable=broad-except
      fatal_error_msgs = [
          'Compilation failure',
          'Run-time shape mismatch for TPUExecute argument'
      ]
      if any([x in str(e) for x in fatal_error_msgs]):
        # Fatal error if failing to compile graph on TPU.
        retry = False
      elif isinstance(e, tf.errors.AbortedError):
        # AbortedError: is thrown when processes restarts.
        retry = True
        if self._InVizierStudy():
          # With Vizier studies, we want to avoid retrying under some error
          # conditions, these are captured here.
          # Do not retry (via raise/retry) if AbortedError with RecvTensor
          # message. This can happen if there are memory issues.
          if ('The same RecvTensor (WorkerServiceImpl) request was received '
              'twice' in str(e)):
            retry = False
            tf.logging.info('%s done (infeasible error).', job_name)
      elif isinstance(e, tf.errors.OutOfRangeError):
        #   OutOfRangeError: Test/dev datasets are exhausted.
        retry = self._cluster.do_eval
      elif isinstance(e, tf.errors.InvalidArgumentError):
        #   InvalidArgumentError: variables were not initialized. Comes from
        #       ResourceVariableOp.
        retry = True
        # Do not retry within Vizier study when NaNs cause InvalidArgumentError.
        if self._InVizierStudy():
          if 'Tensor had NaN values' in str(e):
            retry = False
            tf.logging.info('%s done (infeasible result due to NaN values).',
                            job_name)
      elif isinstance(
          e, py_utils.transient_tf_errors +
          (tf.errors.DataLossError, tf.errors.CancelledError)):
        # Retry on these errors.
        #   FailedPreconditionError: variables are not initialized.
        #   DataLossError: Race condition between evaler and trainer when saving
        #       or removing checkpoints.
        #   CancelledError: Node was closed (on TPU).
        retry = True
      else:
        retry = False

      if retry:
        # Retry indefinitely (error should be transient).
        self._SetStatusMessage(
            '%s exception: %s\n' % (job_name, e), retrying=True)

        for msg in traceback.format_exc().split('\n'):
          tf.logging.vlog(1, msg)

        if cleanup_func:
          cleanup_func()

        if self._daemon:
          # In daemon mode, retry will be handled by an external scheduler by
          # returning a 0 status.
          tf.logging.error('Execution stopped due to fatal error. '
                           'Returning 0 to be scheduled for retry.')
          tf.logging.flush()
          time.sleep(10)
          os._exit(0)  # pylint: disable=protected-access

        raise
      else:
        # Allow the job to complete on errors that are unlikely to be transient,
        # e.g. caused by a mis-configured model.
        if self._should_report_metrics:
          self._trial.ReportDone(
              infeasible=True, infeasible_reason='Fatal error encountered.')
        tf.logging.error('%s done (fatal error): %s', job_name, type(e))

        self._SetStatusMessage('%s exception: %s\n' % (job_name, e))

        # Prints the error message line by line to avoid message cropping.
        msgv = traceback.format_exc().split('\n')
        for msg in msgv:
          tf.logging.error(msg)

        # Check if we are potentially running within an experiment. If so,
        # the worker should continue to the next trial instead of terminating
        # the process.
        if self._InVizierStudy():
          return

        # tf.logging.fatal prints out stack traces. Typically, that's not
        # useful at all here. Here we want to exit the program
        # definitively. Because LOG(QFATAL) is not easily available via
        # python so far, we need a way to exit the program directly.
        # Because sys.exit(1) must be called from the main thread, and does
        # not cancel non-daemon threads anyway, we use os._exit instead.
        tf.logging.flush()
        time.sleep(10)
        os._exit(1)  # pylint: disable=protected-access

  def _DequeueThreadComplete(self):
    self._dequeue_thread_complete = True
    return

  def _LoopEnqueue(self, op, session_override=None):
    """Runs the enqueue op in a loop."""
    if py_utils.IsEagerMode():
      raise ValueError('_LoopEnqueue is not supported in eager mode.')
    p = self.params
    sess = session_override or self._GetSession()

    with tf.container(self._container_id), sess:
      if self._initialize_tables is not None:
        sess.run(self._initialize_tables)
      for task in self._model.tasks:
        task.input.Initialize(sess)
      local_enqueue_steps = 0

      # Global enqueue steps measures how many global steps have data enqueued
      # for already. We use this to terminate; note that the enqueue op may
      # hang in session.run if we do not terminate with this check.
      global_enqueue_steps = None

      tf.logging.info('params.train.max_steps: %d, enqueue_max_steps: %d',
                      p.train.max_steps, p.train.enqueue_max_steps)
      while True:
        if self._dequeue_thread_complete:
          tf.logging.info(
              'LoopEnqueue done since consuming thread is done.')
          return

        global_step = sess.run(py_utils.GetGlobalStep())
        if global_enqueue_steps is None:
          global_enqueue_steps = global_step
        if local_enqueue_steps % 1000 == 0 or self._verbose_enqueue_logging:
          tf.logging.info(
              'Current global_enqueue_steps: %d, '
              'local_enqueue_steps: %d, global_step: %d', global_enqueue_steps,
              local_enqueue_steps, global_step)

        if py_utils.use_tpu():
          global_steps_with_available_data = int(global_enqueue_steps //
                                                 p.train.tpu_steps_per_loop *
                                                 p.train.tpu_steps_per_loop)
          # In TPU Training, the training thread in TrainerTpu is responsible
          # for checking early stop via _ShouldEarlyStop.
          check_early_stop = False
        else:
          global_steps_with_available_data = global_enqueue_steps
          check_early_stop = True

        if (self._ShouldStop(sess, global_steps_with_available_data,
                             check_early_stop) or
            self._ShouldStop(sess, global_step, check_early_stop)):
          tf.logging.info('Enqueue loop: Done. ShouldStop is True. Sleeping')
          time.sleep(15)
          continue
        if (p.train.enqueue_max_steps > 0 and
            local_enqueue_steps >= p.train.enqueue_max_steps):
          tf.logging.info('Enqueue loop: Done. train.enqueue_max_steps '
                          'reached. Sleeping.')
          time.sleep(15)
          continue
        local_enqueue_steps += 1

        # There are tpu_infeed_parallelism parallel threads enqueuing.
        # We account for all of them when updating global_enqueue_steps.
        global_enqueue_steps += p.input.tpu_infeed_parallelism

        # Input data stats generated during training are collected and logged in
        # in input generators. The merged summary op for input data stats merges
        # all the scalar summaries for the stats logged from the input
        # generators. If merged scalar summaries for input data stats are
        # available write them to the training directory along with processing
        # the TPU infeed op.
        if self._merged_input_data_summary_op is not None:
          summary_str, _ = sess.run([self._merged_input_data_summary_op, op])
          self._WriteInputDataStatSummaries(summary_str, global_enqueue_steps)
        else:
          sess.run([op])

  def _GetSession(self, **kwargs):
    if py_utils.IsEagerMode():
      raise ValueError('_GetSession is not supported in eager mode.')
    graph = kwargs.pop('graph', self._graph)
    return tf.Session(
        self._tf_master, graph=graph, config=py_utils.SessionConfig(**kwargs))

  def GetTrainDir(self):
    return self._train_dir

  @classmethod
  def _GetTtlDir(cls, path, duration):
    """Returns a path to a time-limited directory under dir if required."""
    del duration
    return path

  def _CreateSummaryWriter(self, logdir):
    """Creates and returns a tf summary writer."""
    return tf.summary.FileWriter(logdir)

  def _CreateTF2SummaryWriter(self, logdir):
    """Creates a tf2 summary writer."""
    if FLAGS.disable_tf2_summary:
      return
    self._tf2_summary_writer = tf.compat.v2.summary.create_file_writer(logdir)

  def _TF2SummaryContext(self):
    if FLAGS.disable_tf2_summary:
      return contextlib.ExitStack()
    return self._tf2_summary_writer.as_default()

  def _CreateTF2SummaryOps(self):
    if FLAGS.disable_tf2_summary:
      return
    self._tf2_summary_op = tf.compat.v1.summary.all_v2_summary_ops()
    self._tf2_summary_flush = self._tf2_summary_writer.flush()

  def _InitializeTF2SummaryWriter(self, sess=None):
    if FLAGS.disable_tf2_summary:
      return
    if not py_utils.IsEagerMode():
      sess.run(self._tf2_summary_writer.init())

  def _RunTF2SummaryOps(self, sess):
    if FLAGS.disable_tf2_summary or py_utils.IsEagerMode():
      return
    sess.run(self._tf2_summary_op)
    sess.run(self._tf2_summary_flush)

  def _WriteSummaries(self,
                      summary_writer,
                      job_name,
                      global_step,
                      summaries,
                      text_filename=None):
    """Construct the summary and write them to the summary writer.

    Args:
      summary_writer: The summary writer to use.
      job_name: The name of the job that tries to write this summary.
      global_step: The checkpoint used for eval is generated at this step.
      summaries: a dict from keys to `tf.Summary` protobuf messages.
      text_filename: If not None, writes the summary to the text file.
    """
    status_metrics = []
    for name, summary in sorted(summaries.items()):
      if not isinstance(summary, tf.summary.Summary):
        tf.logging.warning(
            'Non tf.Summary args passed to _WriteSummaries, skipping: '
            'job:%s name:%s @%s', job_name, name, global_step)
        continue
      summary_writer.add_summary(summary, global_step)
      if summary.value:
        for value in summary.value:
          if value.HasField('simple_value'):
            tf.logging.info('%s summary on checkpoint@%d %s = %.8g', job_name,
                            global_step, value.tag, value.simple_value)
            status_metrics.append('%s: %.8g' % (value.tag, value.simple_value))
            early_stop.MetricHistory.ConditionalAppend(job_name, value.tag,
                                                       global_step,
                                                       value.simple_value)
          else:
            tf.logging.info('%s summary on checkpoint@%d %s', job_name,
                            global_step, value.tag)
    summary_writer.flush()
    self._SetStatusMessage('%s: step:%6d, %s' %
                           (job_name, global_step, ', '.join(status_metrics)))
    if text_filename is not None:
      with tf.io.gfile.GFile(text_filename, 'w') as f:
        f.write('\n'.join(status_metrics))

  def _WriteInputDataStatSummaries(self, summary_str, global_enqueue_steps):
    """Write input data stats as TF summaries to the training dir.

    Args:
      summary_str: Merged TF scalar summaries for training related input data
        stats.
      global_enqueue_steps: Measures how many global steps for which data has
        been enqueued.
    """
    if summary_str is None:
      return

    if global_enqueue_steps % self._input_stats_summary_interval_steps == 0:
      self._summary_writer.add_summary(summary_str, global_enqueue_steps)
      self._summary_writer.flush()

  def _ExportMetrics(self, **kwargs):
    """Exports metrics externally."""
    pass

  def _TrainerFinished(self, sess=None):
    """Infer if training finished using the latest training checkpoint."""
    latest_ckpt_path = tf.train.latest_checkpoint(self._train_dir)
    if latest_ckpt_path is None:
      return self._ShouldStop(sess, step=0)
    latest_ckpt = tf.train.load_checkpoint(latest_ckpt_path)
    return self._ShouldStop(sess, step=latest_ckpt.get_tensor('global_step'))

  def _GetProcessedCheckpoints(self, runner_dir):
    """Returns the list of checkpoints previously processed by this runner."""
    # Set up (or reload) a file storing the list of previously processed
    # checkpoints. This caching allows jobs to run on VMs which may be
    # interrupted without duplicating work.
    processed_ckpts_path = os.path.join(runner_dir, 'processed_ckpts.txt')
    if not tf.io.gfile.exists(processed_ckpts_path):
      with tf.io.gfile.GFile(processed_ckpts_path, 'w') as f:
        f.write('')
    with tf.io.gfile.GFile(processed_ckpts_path, 'r') as f:
      processed_ckpts = list(line.strip() for line in f.readlines())
    return processed_ckpts

  def _UpdateProcessedCheckpoints(self, runner_dir, ckpt_path):
    """Denotes 'ckpt_path' as having been processed by this runner."""
    processed_ckpts_path = os.path.join(runner_dir, 'processed_ckpts.txt')
    # Some file systems don't support append operations, so we rewrite whole
    # file to append the latest checkpoint.
    processed_ckpts = self._GetProcessedCheckpoints(runner_dir)
    processed_ckpts.append(ckpt_path)
    with tf.io.gfile.GFile(processed_ckpts_path, 'w') as f:
      f.write('\n'.join(processed_ckpts) + '\n')

  def _RunOnLatestCheckpoints(self, sess=None, runner_fn=None, runner_dir=None):
    """Executes 'runner_fn' on the latest checkpoints produced by the Trainer.

    Args:
      sess: the session to compute the metrics in.
      runner_fn: a callable taking a session and a checkpoint path to apply to
        checkpoints saved by the Trainer.
      runner_dir: the log directory for this runner.
    """
    # Check if the trainer finished before this job (re)started. global_step
    # will be 0 in this process until a checkpoint is restored, so we infer the
    # trainer global_step from the step of its latest checkpoint.
    trainer_finished_at_job_start = self._TrainerFinished(sess)
    processed_ckpts = set(self._GetProcessedCheckpoints(runner_dir))
    if (trainer_finished_at_job_start and
        tf.train.latest_checkpoint(self._train_dir) in processed_ckpts):
      tf.logging.warning(
          'Training has finished and the final checkpoint has already been '
          'evaluated. Specify a new eval/decode directory, or set '
          'p.eval.load_checkpoint_from to the final checkpoint to reanalyze it.'
      )
      return

    # Process the latest checkpoints produced by the Trainer until the
    # checkpoint for the final training step has been processed. If training
    # has already finished then only the last checkpoint will be processed.
    while True:
      ckpt_path = self._FindNewCheckpoint(sess, processed_ckpts)
      if ckpt_path is None:
        # Could potentially be None in the case of early stopping.
        break

      runner_fn(sess, ckpt_path)
      self._UpdateProcessedCheckpoints(runner_dir, ckpt_path)
      processed_ckpts.add(ckpt_path)
      if self._ShouldStop(sess):
        break

  def _RunOnAllCheckpoints(self, sess=None, runner_fn=None, runner_dir=None):
    """Executes 'runner_fn' on all checkpoints produced by the Trainer.

    Args:
      sess: the session to compute the metrics in.
      runner_fn: a callable taking a session and a checkpoint path to apply to
        checkpoints saved by the Trainer.
      runner_dir: the log directory for this runner.
    """
    # Check if the trainer finished before this job (re)started. global_step
    # will be 0 in this process until a checkpoint is restored, so we infer the
    # trainer global_step from the step of its latest checkpoint.
    trainer_finished_at_job_start = self._TrainerFinished(sess)
    processed_ckpts = set(self._GetProcessedCheckpoints(runner_dir))

    while True:
      # Checkpoints may be deleted while runner_fn is running, so we fetch the
      # checkpoint state every loop.
      state = tf.train.get_checkpoint_state(self._train_dir)
      ckpts = set() if state is None else state.all_model_checkpoint_paths
      unprocessed_ckpts = set(ckpts).difference(processed_ckpts)

      if unprocessed_ckpts:
        # Process the checkpoints sequentially.
        ckpt_path = sorted(unprocessed_ckpts)[0]
        try:
          runner_fn(sess, ckpt_path)
          self._UpdateProcessedCheckpoints(runner_dir, ckpt_path)
          processed_ckpts.add(ckpt_path)
        except tf.errors.NotFoundError as e:
          # Though it should be exceedingly rare in realistic cases, it's
          # technically possible for the checkpoint in ckpt_path to be deleted
          # by Trainer.checkpointer during the set and sort operations above.
          tf.logging.warning(
              'Ignorring NotFoundError resulting from rare race '
              'condition:\n%s', e)
      elif trainer_finished_at_job_start or self._ShouldStop(sess):
        # Exit if all checkpoints have been processed and training is done.
        break
      else:
        # Check for new checkpoints every 10 seconds if none are found. Spends
        # 1s worth of CPU cycles every ~3hrs looking for new checkpoints.
        time.sleep(10)
