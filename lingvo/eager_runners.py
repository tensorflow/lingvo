# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Eager runners."""
import os
import time
from lingvo import compat as tf
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import cluster_factory
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import summary_utils

from lingvo import base_runner

FLAGS = tf.flags.FLAGS


class Trainer(base_runner.BaseRunner):
  """Trainer that runs in eager mode."""

  def Start(self):
    """Run training."""
    super().Start()

    with self._cluster:
      model = self._params.Instantiate()
      ckptr = self._CreateCheckpointer(self._train_dir, model)
      task = model.GetTask(self._model_task_name)

      @tf.function(autograph=False)
      def TrainFunc():
        with py_utils.GradientTape(persistent=True):
          model.ConstructFPropBPropGraph()
        return task.eval_metrics, task.per_example_tensors

      step_rate_tracker = summary_utils.StepRateTracker()
      summary_writer = tf.compat.v2.summary.create_file_writer(self._train_dir)

      # Attempt to restore the checkpoint
      # A 'dummy run' to initialze the optimizer and related slot variables
      # This is also needed for V2 checkpoint even though it supports delayed
      # loading, in case the checkpoint already exeeds max_steps. In that
      # scenario the slot variables will be lost without a dummy run due to
      # checkpoint overwrites.
      _, _ = TrainFunc()
      global_step = py_utils.GetOrCreateGlobalStepVar()
      # Reset global_step after the dummy run, before loading checkpoints.
      global_step.assign(0)
      path = ckptr.Restore()
      if path:
        tf.logging.info(f'Loaded checkpoints from {path}.')
      else:
        tf.logging.info('Did not find checkpoints in the current directory.')

      global_step = model.global_step.numpy()
      # Save at the beginning of training
      ckptr.Save(gsteps=global_step)
      while True:
        if self._ShouldStop(global_step):
          break

        tf.logging.info('Starting train function.')
        metrics_dict, outfeed = TrainFunc()
        tf.logging.info('Train function complete.')

        global_step = model.global_step.numpy()

        if not task.per_example_tensors:
          assert not outfeed
        else:
          # TODO(laigd): debugging only, remove later.
          tf.logging.info(f'outfeed: {outfeed}')

        ckptr.MaybeSave(gsteps=global_step)

        step_rate, example_rate, total_examples = (
            step_rate_tracker.ComputeStepRate(
                global_step, metrics_dict['num_samples_in_batch'][0].numpy()))

        msg = 'step:%6d, steps/sec: %0.2f, examples/sec: %0.2f' % (
            global_step, step_rate, example_rate)
        # Write summaries.
        with summary_writer.as_default():
          tf.compat.v2.summary.scalar(
              'global_step/sec', step_rate, step=global_step)
          tf.compat.v2.summary.scalar(
              'examples/sec', example_rate, step=global_step)
          tf.compat.v2.summary.scalar(
              'total_samples', total_examples, step=global_step)
          for key, (val, _) in sorted(metrics_dict.items()):
            msg += ' %s:%.8g' % (key, val)
            tf.compat.v2.summary.scalar(key, val, step=global_step)
          summary_writer.flush()

        # Log training progress.
        self._SetStatusMessage(msg)

      # Also save at the end of training
      ckptr.Save(gsteps=global_step)


class TrainSummaries(base_runner.BaseRunner):
  """Write training summaries."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    logdir = os.path.join(self._logdir, 'train_summaries')
    if self._model_task_name:
      logdir += '_' + self._model_task_name
    tf.io.gfile.makedirs(logdir)
    self._summary_writer = tf.compat.v2.summary.create_file_writer(logdir)

  def Start(self):
    """Start."""
    super().Start()

    with self._cluster:
      model = self._params.Instantiate()
      ckptr = self._CreateCheckpointer(self._train_dir, model)
      task = model.GetTask(self._model_task_name)

      next_summary_step = 1
      global_step = model.global_step.numpy()
      last_path = None

      # Initialze the datasets and iterators before `tf.function` because
      # `tf.function` does not trace python side effects.
      # https://www.tensorflow.org/guide/function#executing_python_side_effects
      _ = task.GetInputBatch()

      @tf.function(autograph=False)
      def ModelFunc():
        with self._summary_writer.as_default():
          with py_utils.GradientTape(persistent=True):
            model.ConstructFPropBPropGraph()
          return task.eval_metrics

      while True:
        if self._ShouldStop(global_step):
          break

        time.sleep(30)  # Wait some time between loops.

        path = tf.train.latest_checkpoint(ckptr.checkpoint_dir)
        if path == last_path:
          continue

        # Attempt to restore the checkpoint
        path = ckptr.Restore()
        if not path:
          continue

        last_path = path

        global_step = model.global_step.numpy()
        if global_step >= next_summary_step:
          _ = ModelFunc()
          self._SetStatusMessage(f'Write summary @{global_step}')
          self._summary_writer.flush()
          next_summary_step = (
              global_step + model.params.train.summary_interval_steps)


class Evaler(base_runner.BaseRunner):
  """Evaler."""

  def __init__(self, eval_type, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.params.cluster.do_eval = True
    self._cluster = cluster_factory.Cluster(self.params.cluster)

    self._eval_type = eval_type

    self._eval_dir = os.path.join(self._logdir, f'eval_{eval_type}')
    if self._model_task_name:
      self._eval_dir += '_' + self._model_task_name
    tf.io.gfile.makedirs(self._eval_dir)
    self._summary_writer = tf.compat.v2.summary.create_file_writer(
        self._eval_dir)

  def Start(self):
    """Start."""
    super().Start()

    with self._cluster:
      self._model = self._params.Instantiate()
      self._checkpointer = self._CreateCheckpointer(self._train_dir,
                                                    self._model)
      self._task = self._model.GetTask(self._model_task_name)
      self._eval_fn = self._GetEvalFunc()
      self._eval_fn_with_summary = self._GetEvalFunc(write_summary=True)

    self._eval_path = checkpointer.GetSpecificCheckpoint(
        self._task.params.eval.load_checkpoint_from)

    if self._eval_path:
      self._EvalOnce(path=self._eval_path)
      py_utils.UpdateProcessedCheckpoints(self._eval_dir, self._eval_path)
    elif self._task.params.eval.eval_all_checkpoints:
      self._RunOnAllCheckpoints(
          runner_fn=self._EvalOnce, runner_dir=self._eval_dir)
    else:
      self._RunOnLatestCheckpoints(
          runner_fn=self._EvalOnce, runner_dir=self._eval_dir)

  def _GetEvalFunc(self, write_summary=False):

    @tf.function(autograph=False)
    def EvalFunc():
      if write_summary:
        # TODO(jiaweix): Investigate how to only write non-scalar summaries.
        with self._summary_writer.as_default():
          self._model.ConstructFPropGraph()
      else:
        self._model.ConstructFPropGraph()
      return self._task.eval_metrics

    return EvalFunc

  def _EvalOnce(self, sess=None, path=''):
    """Eval a single checkpoint."""
    with self._cluster:
      # Attempt to restore the checkpoint
      self._checkpointer.RestoreFromPath(checkpoint_path=path)

      # Save any additional information to disk before evaluation.
      if self._eval_type == 'train':
        self._task.Export(path)

      global_step = self._model.global_step.numpy()
      if global_step < self._task.params.eval.start_eval_after:
        return

      if self._task.input.params.resettable:
        tf.logging.info('Resetting input_generator.')
        self._task.input_generator.Reset()
        # In eager mode, after resetting the input generator, we need to
        # re-trace the tf.function to ensure it uses the new iterator.
        self._eval_fn = self._GetEvalFunc().get_concrete_function()
        self._eval_fn_with_summary = self._GetEvalFunc(
            write_summary=True).get_concrete_function()

      metrics_dict = None
      num_samples_metric = None
      samples_per_summary = self._task.params.eval.samples_per_summary
      if samples_per_summary == 0:
        assert self._task.input.params.resettable
      while (samples_per_summary == 0 or metrics_dict is None or
             num_samples_metric.total_value < samples_per_summary):
        try:
          # Evaler calls FProp multiple times for each checkpoint. Multiple
          # summaries at the same step is often confusing.  Instead, models
          # should update eval_metrics and generate aggregate summaries. Other
          # types of summaries (images, audio etc.) will be generated for the
          # first batch only.
          eval_fn = (
              self._eval_fn_with_summary
              if metrics_dict is None else self._eval_fn)
          eval_metrics = eval_fn()

          if metrics_dict is None:
            metrics_dict = {
                name: metrics.AverageMetric() for name in eval_metrics
            }
            num_samples_metric = metrics_dict['num_samples_in_batch']

          eval_metrics = py_utils.Transform(lambda x: x.numpy(), eval_metrics)
          for name, (value, weight) in eval_metrics.items():
            metrics_dict[name].Update(value, weight)
          tf.logging.info('Total examples done: %d/%d',
                          num_samples_metric.total_value, samples_per_summary)
        except tf.errors.OutOfRangeError:
          if not self._task.input.params.resettable:
            raise
          break

      if metrics_dict is None:
        metrics_dict = {}

      # Replace average values with total values for certain metrics.
      if 'num_predictions' in metrics_dict:
        metrics_dict['num_predictions'].total_weight = 1.0
      if 'num_words' in metrics_dict:
        metrics_dict['num_words'].total_weight = 1.0

      msg = 'step:%6d' % global_step
      with self._summary_writer.as_default():
        tf.compat.v2.summary.scalar(
            'total_samples', num_samples_metric.total_value, step=global_step)
        for key, metric in sorted(metrics_dict.items()):
          msg += ' %s:%.8g' % (key, metric.value)
          tf.compat.v2.summary.scalar(key, metric.value, step=global_step)
        self._summary_writer.flush()
      self._SetStatusMessage(msg)


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
    self.params.cluster.do_eval = True
    self._cluster = cluster_factory.Cluster(self.params.cluster)

    self._decoder_dir = os.path.join(self._logdir, f'decoder_{decoder_type}')
    if self._model_task_name:
      self._decoder_dir += '_' + self._model_task_name
    tf.io.gfile.makedirs(self._decoder_dir)
    self._summary_writer = tf.compat.v2.summary.create_file_writer(
        self._decoder_dir)

  def Start(self):
    """Start."""
    super().Start()

    with self._cluster:
      self._model = self._params.Instantiate()
      self._checkpointer = self._CreateCheckpointer(self._train_dir,
                                                    self._model)
      self._task = self._model.GetTask(self._model_task_name)
      self._decode_fn = self._GetDecodeFunc()
      self._decode_fn_with_summary = self._GetDecodeFunc(write_summary=True)

    self._decode_path = checkpointer.GetSpecificCheckpoint(
        self._task.params.eval.load_checkpoint_from)

    if self._decode_path:
      self._DecodeOnce(path=self._decode_path)
      py_utils.UpdateProcessedCheckpoints(self._decoder_dir, self._decode_path)
    elif self._task.params.eval.decode_all_checkpoints:
      self._RunOnAllCheckpoints(
          runner_fn=self._DecodeOnce, runner_dir=self._decoder_dir)
    else:
      self._RunOnLatestCheckpoints(
          runner_fn=self._DecodeOnce, runner_dir=self._decoder_dir)

  def _GetDecodeFunc(self, write_summary=False):

    @tf.function(autograph=False)
    def DecodeFunc():
      if write_summary:
        # TODO(jiaweix): Investigate how to only write non-scalar summaries.
        with self._summary_writer.as_default():
          input_batch, dec_output = self._model.ConstructDecodeGraph(
              self._model_task_name)
      else:
        input_batch, dec_output = self._model.ConstructDecodeGraph(
            self._model_task_name)
      return input_batch, dec_output

    return DecodeFunc

  @classmethod
  def GetDecodeOutPath(cls, decoder_dir, checkpoint_id):
    """Gets the path to decode out file."""
    out_dir = cls._GetTtlDir(decoder_dir, duration='7d')
    return os.path.join(out_dir, 'decoder_out_%09d' % checkpoint_id)

  def _DecodeOnce(self, sess=None, path=''):
    """Decode a single checkpoint."""
    with self._cluster:
      # Attempt to restore the checkpoint
      self._checkpointer.RestoreFromPath(checkpoint_path=path)

      global_step = self._model.global_step.numpy()
      if global_step < self._task.params.eval.start_decoder_after:
        return

      if self._task.input.params.resettable:
        tf.logging.info('Resetting input_generator.')
        self._task.input_generator.Reset()
        # In eager mode, after resetting the input generator, we need to
        # re-trace the tf.function to ensure it uses the new iterator.
        self._decode_fn = self._GetDecodeFunc().get_concrete_function()
        self._decode_fn_with_summary = self._GetDecodeFunc(
            write_summary=True).get_concrete_function()

      dec_metrics = self._task.CreateDecoderMetrics()
      if not dec_metrics:
        tf.logging.info('Empty decoder metrics')
        return
      buffered_decode_out = []
      num_samples_metric = dec_metrics['num_samples_in_batch']

      samples_per_summary = self._task.params.eval.decoder_samples_per_summary
      if samples_per_summary is None:
        samples_per_summary = self._task.params.eval.samples_per_summary
      if samples_per_summary == 0:
        assert self._task.input.params.resettable

      start_time = time.time()
      while samples_per_summary == 0 or (num_samples_metric.total_value <
                                         samples_per_summary):
        try:
          tf.logging.info('Fetching dec_output.')
          fetch_start = time.time()
          # Decoder calls FProp multiple times for each checkpoint. Multiple
          # summaries at the same step is often confusing.  Instead, models
          # should generate aggregate summaries using PostProcessDecodeOut.
          # Other types of summaries (images, audio etc.) will be generated for
          # the first batch only.
          is_first_loop = num_samples_metric.total_value == 0
          decode_fn = (
              self._decode_fn_with_summary
              if is_first_loop else self._decode_fn)
          input_batch, dec_output = decode_fn()

          for key in self._task.input_generator.GetCpuPassthroughKeys():
            if key in input_batch:
              if key in dec_output:
                tf.logging.warning(
                    f'Key {key} already present in decode output. '
                    f'Not adding from input batch.')
              else:
                dec_output[key] = input_batch[key]

          dec_output = py_utils.Transform(lambda x: x.numpy(), dec_output)

          post_process_start = time.time()
          tf.logging.info('Done fetching (%f seconds)' %
                          (post_process_start - fetch_start))
          decode_out = self._task.PostProcessDecodeOut(dec_output, dec_metrics)

          if decode_out:
            if isinstance(decode_out, dict):
              decode_out = decode_out.items()

            if is_first_loop:
              # Add summaries only for the first batch of data.
              with self._summary_writer.as_default():
                for key, value in decode_out:
                  if isinstance(value, tf.Summary):
                    tf.logging.info(f'Adding summary {key} with tags '
                                    f'{[x.tag for x in value.value]}.')
                    tf.compat.v2.summary.experimental.write_raw_pb(
                        tf.constant(value.SerializeToString()), global_step)

            buffered_decode_out.extend(
                kv for kv in decode_out if not isinstance(kv[1], tf.Summary))

          tf.logging.info(
              'Total examples done: %d/%d '
              '(%f seconds decode postprocess)', num_samples_metric.total_value,
              samples_per_summary,
              time.time() - post_process_start)

        except tf.errors.OutOfRangeError:
          if not self._task.input.params.resettable:
            raise
          break

      tf.logging.info('Done decoding ckpt: %s', path)

      elapsed_secs = time.time() - start_time
      example_rate = num_samples_metric.total_value / elapsed_secs
      msg = 'step:%6d, elapsed_secs: %0.2f, examples/sec: %0.2f' % (
          global_step, elapsed_secs, example_rate)
      with self._summary_writer.as_default():
        tf.compat.v2.summary.scalar(
            'decode_secs', elapsed_secs, step=global_step)
        tf.compat.v2.summary.scalar(
            'examples/sec', example_rate, step=global_step)
        tf.compat.v2.summary.scalar(
            'total_samples', num_samples_metric.total_value, step=global_step)
        for key, metric in sorted(dec_metrics.items()):
          msg += ' %s:%.8g' % (key, metric.value)
          tf.compat.v2.summary.scalar(key, metric.value, step=global_step)
        self._summary_writer.flush()
      self._SetStatusMessage(msg)

      self._ExportMetrics(
          # Metrics expects python int, but global_step is numpy.int64.
          decode_checkpoint=int(global_step),
          dec_metrics=dec_metrics,
          example_rate=example_rate)

      decode_out_path = self.GetDecodeOutPath(self._decoder_dir, global_step)
      decode_finalize_args = base_model.DecodeFinalizeArgs(
          decode_out_path=decode_out_path, decode_out=buffered_decode_out)
      self._task.DecodeFinalize(decode_finalize_args)
