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
# pylint: disable=line-too-long
r"""Trainer.

To run locally:

.. code-block:: bash

  $ bazel build -c opt //lingvo:trainer
  $ bazel-bin/lingvo/trainer --logtostderr \
      --model=image.mnist.LeNet5 --mode=sync --logdir=/tmp/lenet5 \
      --run_locally=cpu

To use GPU, add `--config=cuda` to build command and set `--run_locally=gpu`.
"""
# pylint: enable=line-too-long
import os
import re

import time

import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import cluster_factory
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import summary_utils

from lingvo import base_runner


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
      self._initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      tf.logging.info('Trainer number of enqueue ops: %d',
                      len(self.enqueue_ops))

    self._step_rate_tracker = summary_utils.StepRateTracker()

    # Saves the graph def.
    if self.params.cluster.task == 0:
      self._WriteToLog(self.params.ToText(), self._train_dir,
                       'trainer_params.txt')
      tf.io.write_graph(self._graph.as_graph_def(), self._train_dir,
                        'train.pbtxt')
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
        cluster = self._cluster
        with tf.device(cluster.input_device):
          input_batch = (self._task.input_generator.GetPreprocessedInputBatch())

        self._dec_output = self._task.Decode(input_batch)
        self._summary_op = tf.summary.merge_all()
        self.checkpointer = self._CreateCheckpointer(self._train_dir,
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

  def _CreateCheckpointer(self, train_dir, model):
    """Wrapper method for override purposes."""
    return checkpointer.Checkpointer(train_dir, model)

  def Start(self):
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
      else:
        path = None
        while True:
          path = self._FindNewCheckpoint(path, sess)
          if not path or self.DecodeCheckpoint(sess, path):
            break

    # Maybe decode the last checkpoint if we are not given a specific
    # checkpoint to decode.
    if self._decode_path is None:
      self.DecodeLatestCheckpoint(path)

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

  def DecodeCheckpoint(self, sess, checkpoint_path):
    """Decodes `samples_per_summary` examples using `checkpoint_path`."""
    p = self._task.params
    ckpt_id_from_file = self.GetCkptIdFromFile(checkpoint_path)
    if ckpt_id_from_file < p.eval.start_decoder_after:
      return False
    samples_per_summary = p.eval.decoder_samples_per_summary
    if samples_per_summary is None:
      samples_per_summary = p.eval.samples_per_summary
    if samples_per_summary == 0:
      assert self._task.params.input.resettable
    self.checkpointer.RestoreFromPath(sess, checkpoint_path)

    global_step = sess.run(py_utils.GetGlobalStep())

    if self._task.params.input.resettable:
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
        tf.logging.info('Fetching dec_output.')
        fetch_start = time.time()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=False)
        if self._summary_op is None:
          # No summaries were collected.
          dec_out = sess.run(self._dec_output, options=run_options)
        else:
          dec_out, summary = sess.run([self._dec_output, self._summary_op],
                                      options=run_options)
          self._summary_writer.add_summary(summary, global_step)
        self._RunTF2SummaryOps(sess)
        post_process_start = time.time()
        tf.logging.info('Done fetching (%f seconds)' %
                        (post_process_start - fetch_start))
        decode_out = self._task.PostProcessDecodeOut(dec_out, dec_metrics)
        if decode_out:
          buffered_decode_out.extend(decode_out)
        tf.logging.info(
            'Total examples done: %d/%d '
            '(%f seconds decode postprocess)', num_examples_metric.total_value,
            samples_per_summary,
            time.time() - post_process_start)
      except tf.errors.OutOfRangeError:
        if not self._task.params.input.resettable:
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

    should_stop = global_step >= self.params.train.max_steps
    if self._should_report_metrics:
      tf.logging.info('Reporting eval measure for step %d.' % global_step)
      trial_should_stop = self._trial.ReportEvalMeasure(global_step,
                                                        dec_metrics,
                                                        checkpoint_path)
      should_stop = should_stop or trial_should_stop
    return should_stop

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
