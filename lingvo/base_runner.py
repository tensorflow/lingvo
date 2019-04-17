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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time
import traceback
import tensorflow as tf

from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import saver_pb2

from lingvo import base_trial
from lingvo.core import cluster_factory
from lingvo.core import early_stop
from lingvo.core import py_utils

tf.flags.DEFINE_integer(
    'enqueue_max_steps', -1, 'Max enqueue steps. -1 meaning no limit.'
    ' This flag should be set for unit-test only.')

tf.flags.DEFINE_integer('saver_max_to_keep', 100,
                        'Maximum number of recent checkpoints to keep.')

FLAGS = tf.flags.FLAGS


class BaseRunner(object):
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
      model_task_name:  String name of the task this runner should execute
        for multitask models only.  See flag for details.
      logdir:  String path to the log directory to output to.
      tf_master:  String path to the master job, e.g. 'local'.
      trial:   An optional hyperparameter trial. Used by Vizier studies.
    """
    p = params.Copy()

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
    # to early stop a trial.
    self._max_steps = None

    self._cluster = cluster_factory.Cluster(self.params.cluster)
    self._train_dir = os.path.join(self._logdir, 'train')
    self._graph = tf.Graph()
    self._summary_writer = None
    self.initialize_tables = None

    early_stop.MetricHistory.SetLogdirInMetricHistories(p, logdir)
    self._early_stop = None
    if p.train.early_stop and p.train.early_stop.window:
      self._early_stop = early_stop.EarlyStop(p.train.early_stop)
      with self._graph.as_default():
        self._early_stop.FProp(None)

    self._SetStatusMessage('Starting ...')

  @property
  def params(self):
    return self._params

  def _InVizierStudy(self):
    return not isinstance(self._trial, base_trial.NoOpTrial)

  def _SetStatusMessage(self, message, retrying=False):
    """Update the status message for this task."""
    if self._trial.Name():
      message = 'Trial:{} {}'.format(self._trial.Name(), message)
    if retrying:
      message = '<b>Retrying as expected</b> ' + str(message)
    tf.logging.info(message)
    return message

  def _ShouldStop(self, sess, step):
    """Check if the runner should stop."""
    if step >= self.params.train.max_steps:
      tf.logging.info('ShouldStop: step:%6d params.train.max_steps:%6d', step,
                      self.params.train.max_steps)
      return True

    if self._max_steps and step >= self._max_steps:
      tf.logging.info('ShouldStop: step:%6d _max_steps:%6d', step,
                      self._max_steps)
      return True

    if self._early_stop and self._early_stop.Stop(sess):
      tf.logging.info('ShouldStop: Early stopping.')
      return True

    return False

  def _WriteToLog(self, text, logdir, filename):
    """Logs `text` and saves it under `logdir/filename`."""
    with tf.gfile.FastGFile(os.path.join(logdir, filename), 'w') as f:
      f.write(text)

    if self._summary_writer is not None:
      # Force newlines to be rendered correctly by Markdown.
      text = text.replace('\n', '  \n')
      self._summary_writer.add_summary(
          tf.Summary(value=[
              tf.Summary.Value(
                  tag=filename, tensor=tf.make_tensor_proto([text]))
          ]))

  def _GetSaver(self):
    """Returns a saver."""
    assert tf.get_default_graph() == self._graph
    if self.params.is_eval and self._model.ema:
      tf.logging.info('Using EMA for evaluation.')
      return tf.train.Saver(self._model.ema.variables_to_restore())
    return tf.train.Saver(
        sharded=True,
        max_to_keep=FLAGS.saver_max_to_keep,
        keep_checkpoint_every_n_hours=0.5,  # one per 30 minutes
        pad_step_number=True,  # %08d
        write_version=saver_pb2.SaverDef.V2)

  def _LoadCheckpointForEval(self, sess, checkpoint_path):
    """Load the checkpoint for evaluation."""
    tf.logging.info('Load from checkpoint %s.', checkpoint_path)
    self._saver.restore(sess, checkpoint_path)
    tf.logging.info('Load checkpoint done.')

  @py_utils.Retry(initial_delay_sec=300, max_delay_sec=300)
  def _FindNewCheckpoint(self, prev_path, sess):
    """Returns the path to a new checkpoint, or raises RuntimeError."""
    if self._trial.ShouldStop() or self._ShouldStop(sess, 0):
      return None
    path = tf.train.latest_checkpoint(self._train_dir)
    if not path:
      msg = 'No check point is found in %s' % self._train_dir
      tf.logging.info('%s', msg)
      raise RuntimeError(msg)
    if path == prev_path:
      msg = 'No new check point is found: %s' % path
      tf.logging.info('%s', msg)
      raise RuntimeError(msg)
    return path

  @py_utils.Retry()
  def _RunLoop(self, job_name, loop_func, loop_args=()):
    """Runs `loop_func`, retrying on expected errors.

    Args:
      job_name: string job name.
      loop_func: callable to run and retry on expected errors.
      loop_args: list or tuple of arguments to be passed to the loop_func.
    """
    try:
      tf.logging.info('%s started.', job_name)
      loop_func(*loop_args)
      tf.logging.info('%s done.', job_name)
      return
    except py_utils.transient_tf_errors + (tf.errors.OutOfRangeError,
                                           tf.errors.DataLossError) as e:
      # Retry on these three errors.
      #   FailedPreconditionError: variables are not initialized.
      #   AbortedError: processes restarts.
      #   OutOfRangeError: Test/dev datasets are exhausted.
      #   DataLossError: Race condition between evaler and trainer when saving
      #       or removing checkpoints.
      self._SetStatusMessage(
          '%s exception: %s\n' % (job_name, e), retrying=True)

      for msg in traceback.format_exc().split('\n'):
        tf.logging.vlog(1, msg)

      # With Vizier studies, we want to avoid retrying under some error
      # conditions, these are captured here.
      if self._InVizierStudy():
        # Do not retry (via raise/retry) if AbortedError with RecvTensor
        # message. This can happen if there are memory issues.
        if (isinstance(e, tf.errors.AbortedError) and
            'The same RecvTensor (WorkerServiceImpl) request was received twice'
            in str(e)):
          if self._should_report_metrics:
            self._trial.ReportDone(
                infeasible=True,
                infeasible_reason='Infeasible error encountered.')
          tf.logging.info('%s done (infeasible error).', job_name)
          return

      # Retry indefinitely (error should be transient).
      raise
    except Exception as e:  # pylint: disable=broad-except
      # Allow the job to complete on errors that are unlikely to be transient,
      # e.g. caused by a mis-configured model.
      if self._should_report_metrics:
        self._trial.ReportDone(
            infeasible=True, infeasible_reason='Fatal error encountered.')
      tf.logging.info('%s done (fatal error).', job_name)

      self._SetStatusMessage('%s exception: %s\n' % (job_name, e))

      # Prints the error message line by line to avoid message cropping.
      msgv = traceback.format_exc().split('\n')
      for msg in msgv:
        tf.logging.error(msg)

      # Check if we are potentially running within an experiment. If so,
      # the worker should continue to the next trial instead of terminating the
      # process.
      if self._InVizierStudy():
        return

      # tf.logging.fatal prints out stack traces. Typically, that's not
      # useful at all here. Here we want to exit the program
      # definitively. Because LOG(QFATAL) is not easily available via
      # python so far, we need a way to exit the program directly.
      # Because sys.exit(1) must be called from the main thread, and does
      # not cancel non-daemon threads anyway, we use os._exit instead.
      # Because tf.logging.error() may return before the flush is complete,
      # we need an extra sleep before exit.
      time.sleep(15)
      os._exit(1)  # pylint: disable=protected-access

  def _LoopEnqueue(self, op):
    """Runs the enqueue op in a loop."""
    with tf.container(self._container_id), self._GetSession() as sess:
      if self.initialize_tables is not None:
        sess.run(self.initialize_tables)
      gsteps = py_utils.GetGlobalStep()
      local_enqueue_steps = 0

      # Global enqueue steps measures how many global steps have data enqueued
      # for already. We use this to terminate; note that the enqueue op may
      # hang in session.run if we do not terminate with this check.
      global_enqueue_steps = None

      tf.logging.info('params.train.max_steps: %d, enqueue_max_steps: %d',
                      self.params.train.max_steps, FLAGS.enqueue_max_steps)
      while True:
        global_step = sess.run(gsteps)
        if global_enqueue_steps is None:
          global_enqueue_steps = global_step
        if local_enqueue_steps % 1000 == 0:
          tf.logging.info(
              'Current global_enqueue_steps: %d, '
              'local_enqueue_steps: %d, global_step: %d', global_enqueue_steps,
              local_enqueue_steps, global_step)

        if py_utils.use_tpu():
          global_steps_with_available_data = int(
              global_enqueue_steps // self.params.train.tpu_steps_per_loop *
              self.params.train.tpu_steps_per_loop)
        else:
          global_steps_with_available_data = global_enqueue_steps

        if (self._ShouldStop(sess, global_steps_with_available_data) or
            self._ShouldStop(sess, global_step)):
          tf.logging.info('Done. ShouldStop is True.')
          return
        if (FLAGS.enqueue_max_steps > 0 and
            local_enqueue_steps > FLAGS.enqueue_max_steps):
          tf.logging.info('Done. FLAGS.enqueue_max_steps reached.')
          return
        local_enqueue_steps += 1

        # There are tpu_infeed_parallism parallel threads enqueuing.
        # We account for all of them when updating global_enqueue_steps.
        global_enqueue_steps += self.params.input.tpu_infeed_parallism

        sess.run([op])

  def _GetSession(self, **kwargs):
    return tf.Session(
        self._tf_master,
        graph=self._graph,
        config=py_utils.SessionConfig(**kwargs))

  @classmethod
  def _GetTtlDir(cls, path, duration):
    """Returns a path to a time-limited directory under dir if required."""
    del duration
    return path

  def _CreateSummaryWriter(self, logdir):
    """Creates and returns a tf summary writer."""
    return tf.summary.FileWriter(logdir)

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
      if not isinstance(summary, summary_pb2.Summary):
        tf.logging.warning(
            'Non tf.Summary args passed to _WriteSummaries, skipping: %s @%s',
            job_name, global_step)
        continue
      summary_writer.add_summary(summary, global_step)
      if summary.value[0].HasField('simple_value'):
        value = summary.value[0].simple_value
        tf.logging.info('%s summary on checkpoint@%d %s = %.8g', job_name,
                        global_step, name, value)
        status_metrics.append('%s: %.8g' % (name, value))
        early_stop.MetricHistory.ConditionalAppend(job_name, name, global_step,
                                                   value)
      else:
        tf.logging.info('%s summary on checkpoint@%d %s', job_name, global_step,
                        name)
    summary_writer.flush()
    self._SetStatusMessage(
        '%s: step:%6d, %s' % (job_name, global_step, ', '.join(status_metrics)))
    if text_filename is not None:
      with tf.gfile.FastGFile(text_filename, 'w') as f:
        f.write('\n'.join(status_metrics))

  def _WriteKeyValuePairs(self, filename, key_value_pairs):
    """Writes `key_value_pairs` to `filename`."""
    with open(filename, 'wb') as f:
      pickle.dump(key_value_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

  def _ExportMetrics(self, **kwargs):
    """Exports metrics externally."""
    pass
