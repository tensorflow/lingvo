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
"""Tests for trainer."""

import os
import pathlib
import random
import re
import shutil

from absl.testing import flagsaver
from absl.testing import parameterized
from lingvo import base_trial
from lingvo import model_registry
from lingvo import trainer
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import hyperparams
from lingvo.core import inference_graph_pb2
from lingvo.core import test_utils
from lingvo.core import trainer_test_utils
from lingvo.tasks.image.input_generator import FakeMnistData
import numpy as np

FLAGS = tf.flags.FLAGS


class BaseTrainerTest(test_utils.TestCase):
  """Base class for the test cases."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._trial = base_trial.NoOpTrial()

  def setUp(self):
    super().setUp()
    FLAGS.model_params_override = ''
    FLAGS.tf_master = tf.distribute.Server.create_local_server().target
    FLAGS.vizier_reporting_job = 'decoder'

  def _CreateController(self, cfg):
    return trainer.Controller(cfg, FLAGS.model_task_name, FLAGS.logdir,
                              FLAGS.tf_master, self._trial)

  def _CreateTrainer(self, cfg):
    return trainer.Trainer(cfg, FLAGS.model_task_name, FLAGS.logdir,
                           FLAGS.tf_master, self._trial)

  def _CreateEvalerDev(self, cfg):
    return trainer.Evaler('dev', cfg, FLAGS.model_task_name, FLAGS.logdir,
                          FLAGS.tf_master, self._trial)

  def _CreateDecoderDev(self, cfg):
    return trainer.Decoder('dev', cfg, FLAGS.model_task_name, FLAGS.logdir,
                           FLAGS.tf_master, self._trial)

  def _HasFile(self, files, substr):
    for f in files:
      if substr in f:
        return True
    return False

  def _GetMatchedFileName(self, files, substr):
    for f in files:
      if substr in f:
        return f
    return None

  def _HasLine(self, filename, pattern):
    """Returns True iff one line in the given file matches the pattern."""
    with tf.io.gfile.GFile(filename, 'r') as f:
      lines = f.readlines()
    return any(re.search(pattern, _) for _ in lines)

  def _GetSimpleTestConfig(self):
    model_name = 'image.mnist.LeNet5'
    cfg = model_registry.GetParams(model_name, 'Train')
    cfg.cluster.task = 0
    cfg.cluster.mode = 'sync'
    cfg.cluster.job = 'trainer_client'
    cfg.cluster.worker.name = '/job:localhost'
    cfg.cluster.worker.replicas = 1
    cfg.cluster.worker.gpus_per_replica = 0
    cfg.cluster.ps.name = '/job:localhost'
    cfg.cluster.ps.replicas = 1
    cfg.cluster.ps.gpus_per_replica = 0
    cfg.cluster.reporting_job = FLAGS.vizier_reporting_job

    # Generate 2 inputs.
    cfg.input.ckpt = FakeMnistData(
        self.get_temp_dir(), train_size=2, test_size=2)
    cfg.input.num_samples = 2
    cfg.input.batch_size = 2
    cfg.train.max_steps = 2
    cfg.task.train.ema_decay = 0.9999
    return cfg


class EmptyTask(base_model.BaseTask):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'empty'
    return p

  def Inference(self):
    return inference_graph_pb2.InferenceGraph()


class EmptyMultiTaskModel(base_model.MultiTaskModel):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'empty'
    p.task_params.Define('a', EmptyTask.Params(), '')
    p.task_params.Define('b', EmptyTask.Params(), '')
    return p


@model_registry.RegisterMultiTaskModel
class EmptyMultiTaskParams(base_model_params.MultiTaskModelParams):

  def Test(self):
    p = base_input_generator.BaseInputGenerator.Params()
    inputs = hyperparams.Params()
    for task_name in ['a', 'b']:
      inputs.Define(task_name, p.Copy(), '')
    return inputs

  def Model(self):
    return EmptyMultiTaskModel.Params()


class TrainerTest(BaseTrainerTest, parameterized.TestCase):

  @flagsaver.flagsaver
  def testController(self):
    logdir = os.path.join(tf.test.get_temp_dir(),
                          'controller_test' + str(random.random()))
    FLAGS.logdir = logdir
    cfg = self._GetSimpleTestConfig()

    runner_manager = trainer.RunnerManager(cfg.name)

    runner_manager.StartRunners(
        [self._CreateController(cfg),
         self._CreateTrainer(cfg)])

    train_files = tf.io.gfile.glob(logdir + '/train/*')
    self.assertTrue(self._HasFile(train_files, 'ckpt'))
    self.assertTrue(self._HasFile(train_files, 'tfevents'))
    control_files = tf.io.gfile.glob(logdir + '/control/*')
    self.assertTrue(self._HasFile(control_files, 'params.txt'))
    self.assertTrue(self._HasFile(control_files, 'params.pbtxt'))
    self.assertTrue(self._HasFile(control_files, 'model_analysis.txt'))
    self.assertTrue(self._HasFile(control_files, 'tfevents'))

    # EvalerDev may not run concurrently with Controller in a single process
    # because EvalerDev loads checkpoints and overwrites states like global
    # steps.
    runner_manager.StartRunners([self._CreateEvalerDev(cfg)])

    dev_files = tf.io.gfile.glob(logdir + '/eval_dev/*')
    self.assertTrue(self._HasFile(dev_files, 'params.txt'))
    self.assertTrue(self._HasFile(dev_files, 'eval_dev.pbtxt'))
    self.assertTrue(self._HasFile(dev_files, 'tfevents'))
    self.assertTrue(self._HasFile(dev_files, 'score'))
    self.assertTrue(
        self._HasLine(self._GetMatchedFileName(dev_files, 'score'), 'log_pplx'))

  @flagsaver.flagsaver
  def testDecoder(self):
    logdir = os.path.join(tf.test.get_temp_dir(),
                          'decoder_test' + str(random.random()))
    FLAGS.logdir = logdir
    dec_dir = os.path.join(logdir, 'decoder_dev')
    cfg = self._GetSimpleTestConfig()
    runner_manager = trainer.RunnerManager(cfg.name)
    runner_manager.StartRunners([
        self._CreateController(cfg),
        self._CreateTrainer(cfg),
    ])

    # Test decoding with default settings.
    with self.subTest(name='DefaultDecoder'):
      runner_manager.StartRunners([self._CreateDecoderDev(cfg)])
      dec_files = tf.io.gfile.glob(os.path.join(dec_dir, '*'))
      self.assertTrue(self._HasFile(dec_files, 'params.txt'))
      self.assertTrue(self._HasFile(dec_files, 'decoder_dev.pbtxt'))
      self.assertTrue(self._HasFile(dec_files, 'tfevents'))
      self.assertTrue(self._HasFile(dec_files, 'processed_ckpts.txt'))
      # Only the score for the 2-step checkpoint should be present.
      self.assertFalse(self._HasFile(dec_files, 'score-00000000.txt'))
      self.assertTrue(self._HasFile(dec_files, 'score-00000002.txt'))
      self.assertTrue(
          self._HasLine(
              self._GetMatchedFileName(dec_files, 'score'), 'examples/sec'))

    # Test that checkpoints are not reevaluated when a job is interrupted.
    score_2_path = os.path.join(dec_dir, 'score-00000002.txt')
    score_2_mod_time = pathlib.Path(score_2_path).stat().st_mtime
    with self.subTest(name='DefaultDecoderNoOp'):
      cfg = self._GetSimpleTestConfig()
      runner_manager.StartRunners([self._CreateDecoderDev(cfg)])

      dec_files = tf.io.gfile.glob(os.path.join(dec_dir, '*'))
      self.assertFalse(self._HasFile(dec_files, 'score-00000000.txt'))
      self.assertEqual(score_2_mod_time,
                       pathlib.Path(score_2_path).stat().st_mtime)

    # Test decoding a specific checkpoint.
    with self.subTest(name='LoadCheckpointFrom'):
      cfg = self._GetSimpleTestConfig()
      cfg.task.eval.load_checkpoint_from = os.path.join(logdir,
                                                        'train/ckpt-00000000')
      runner_manager.StartRunners([self._CreateDecoderDev(cfg)])
      dec_files = tf.io.gfile.glob(os.path.join(dec_dir, '*'))

      # Scores for both checkpoints should be present...
      self.assertTrue(self._HasFile(dec_files, 'score-00000000.txt'))
      self.assertTrue(self._HasFile(dec_files, 'score-00000002.txt'))
      # ... but only the score for the 0-step checkpoint should be modified.
      self.assertEqual(score_2_mod_time,
                       pathlib.Path(score_2_path).stat().st_mtime)

    # Reset the decoder's cached state and test decoding all checkpoints.
    shutil.rmtree(dec_dir)
    with self.subTest(name='DecodeAllCheckpoints'):
      cfg = self._GetSimpleTestConfig()
      cfg.task.eval.decode_all_checkpoints = True
      runner_manager.StartRunners([self._CreateDecoderDev(cfg)])
      dec_files = tf.io.gfile.glob(os.path.join(dec_dir, '*'))
      self.assertTrue(self._HasFile(dec_files, 'score-00000000.txt'))
      self.assertTrue(self._HasFile(dec_files, 'score-00000002.txt'))

    # Test that decode_all_checkpoints on an already decoded dir is a no-op.
    score_2_mod_time = pathlib.Path(score_2_path).stat().st_mtime
    with self.subTest(name='DecodeAllCheckpointsNoOp'):
      runner_manager.StartRunners([self._CreateDecoderDev(cfg)])
      self.assertEqual(score_2_mod_time,
                       pathlib.Path(score_2_path).stat().st_mtime)

  @flagsaver.flagsaver
  def testWriteInferenceGraph(self):
    random.seed()
    logdir = os.path.join(tf.test.get_temp_dir(),
                          'inference_graphs' + str(random.random()))
    FLAGS.logdir = logdir
    cfg = 'punctuator.codelab.RNMTModel'
    trainer.RunnerManager(cfg).WriteInferenceGraph()
    inference_files = tf.io.gfile.glob(logdir + '/inference_graphs/*')
    self.assertTrue(self._HasFile(inference_files, 'inference.pbtxt'))
    self.assertTrue(self._HasFile(inference_files, 'inference_tpu.pbtxt'))

  @flagsaver.flagsaver(model_task_name='a')
  def testWriteOneOfMultiTaskInferenceGraph(self):
    random.seed()
    logdir = os.path.join(tf.test.get_temp_dir(),
                          'inference_graphs' + str(random.random()))
    FLAGS.logdir = logdir
    cfg = 'test.EmptyMultiTaskParams'
    trainer.RunnerManager(cfg).WriteInferenceGraph()
    inference_files = tf.io.gfile.glob(logdir + '/inference_graphs/*')
    self.assertTrue(self._HasFile(inference_files, 'a_inference.pbtxt'))
    self.assertTrue(self._HasFile(inference_files, 'a_inference_tpu.pbtxt'))
    self.assertFalse(self._HasFile(inference_files, 'b_inference.pbtxt'))
    self.assertFalse(self._HasFile(inference_files, 'b_inference_tpu.pbtxt'))

  @flagsaver.flagsaver
  def testWriteMultiTaskInferenceGraph(self):
    random.seed()
    logdir = os.path.join(tf.test.get_temp_dir(),
                          'inference_graphs' + str(random.random()))
    FLAGS.logdir = logdir
    cfg = 'test.EmptyMultiTaskParams'
    trainer.RunnerManager(cfg).WriteInferenceGraph()
    inference_files = tf.io.gfile.glob(logdir + '/inference_graphs/*')
    self.assertTrue(self._HasFile(inference_files, 'a_inference.pbtxt'))
    self.assertTrue(self._HasFile(inference_files, 'a_inference_tpu.pbtxt'))
    self.assertTrue(self._HasFile(inference_files, 'b_inference.pbtxt'))
    self.assertTrue(self._HasFile(inference_files, 'b_inference_tpu.pbtxt'))

  def testRunLocally(self):
    logdir = os.path.join(tf.test.get_temp_dir(),
                          'run_locally_test' + str(random.random()))
    FLAGS.logdir = logdir
    FLAGS.run_locally = 'cpu'
    FLAGS.mode = 'sync'
    FLAGS.model = 'image.mnist.LeNet5'
    FLAGS.model_params_override = (
        'train.max_steps: 2; input.num_samples: 2; input.ckpt: %s' %
        FakeMnistData(self.get_temp_dir(), train_size=2, test_size=2))
    trainer.main(None)

    train_files = tf.io.gfile.glob(logdir + '/train/*')
    self.assertTrue(self._HasFile(train_files, 'ckpt'))
    self.assertTrue(self._HasFile(train_files, 'tfevents'))
    control_files = tf.io.gfile.glob(logdir + '/control/*')
    self.assertTrue(self._HasFile(control_files, 'params.txt'))
    self.assertTrue(self._HasFile(control_files, 'model_analysis.txt'))
    self.assertTrue(self._HasFile(control_files, 'tfevents'))

  @parameterized.named_parameters(('Evaler', trainer.Evaler),
                                  ('Decoder', trainer.Decoder))
  def testEMA(self, cls):
    logdir = os.path.join(tf.test.get_temp_dir(),
                          'ema_test' + str(random.random()))
    FLAGS.logdir = logdir
    cfg = self._GetSimpleTestConfig()
    runner = cls('dev', cfg, FLAGS.model_task_name, FLAGS.logdir,
                 FLAGS.tf_master, self._trial)
    var_list = runner.checkpointer._saver._var_list  # pylint: disable=protected-access
    self.assertIsInstance(var_list, dict)
    for var_name in var_list.keys():
      if var_name != 'global_step':
        self.assertTrue(var_name.endswith('/ExponentialMovingAverage'))


class TrainerWithTrialTest(BaseTrainerTest):

  @flagsaver.flagsaver
  def testControllerTrainerEvaler(self):
    trial = tf.test.mock.create_autospec(base_trial.Trial, instance=True)
    self._trial = trial

    logdir = os.path.join(tf.test.get_temp_dir(),
                          'controller_test' + str(random.random()))
    FLAGS.logdir = logdir
    cfg = self._GetSimpleTestConfig()

    trial.Name.return_value = 'trial1'

    def override_model_params(model_params):
      model_params.task.softmax.num_classes = 20
      model_params.task.filter_shapes = [(5, 5, 1, 10), (5, 5, 10, 50)]
      model_params.task.train.lr_schedule.decay_start = 100
      return model_params

    trial.OverrideModelParams.side_effect = override_model_params
    trial.ShouldStop.return_value = False
    trial.ShouldStopAndMaybeReport.return_value = False
    # Stop trial once ReportEvalMeasure is called.
    trial.ReportEvalMeasure.return_value = True

    runners = [self._CreateController(cfg), self._CreateTrainer(cfg)]
    # Param override works.
    for runner in runners:
      self.assertEqual(runner.params.task.softmax.num_classes, 20)
      self.assertEqual(runner.params.task.filter_shapes, [(5, 5, 1, 10),
                                                          (5, 5, 10, 50)])
      self.assertEqual(runner.params.task.train.lr_schedule.decay_start, 100)

    runner_manager = trainer.RunnerManager(cfg.name)
    runner_manager.StartRunners(runners)
    # Controller and trainer check whether the trial is stopped.
    self.assertGreater(trial.OverrideModelParams.call_count, 0)
    self.assertGreater(trial.ShouldStop.call_count, 0)
    self.assertGreater(trial.ShouldStopAndMaybeReport.call_count, 0)
    # Controller and trainer do not call report_measure, request_trial_stop, or
    # report_done.
    self.assertEqual(trial.ReportEvalMeasure.call_count, 0)

    train_files = tf.io.gfile.glob(logdir + '/train/*')
    self.assertTrue(self._HasFile(train_files, 'params.txt'))
    self.assertTrue(self._HasFile(train_files, 'trainer_params.txt'))
    self.assertTrue(self._HasFile(train_files, 'ckpt'))
    self.assertTrue(self._HasFile(train_files, 'tfevents'))
    control_files = tf.io.gfile.glob(logdir + '/control/*')
    self.assertTrue(self._HasFile(control_files, 'params.txt'))
    self.assertTrue(self._HasFile(control_files, 'model_analysis.txt'))
    self.assertTrue(self._HasFile(control_files, 'tfevents'))

    # EvalerDev may not run concurrently with Controller in a single process
    # because EvalerDev loads checkpoints and overwrites states like global
    # steps.
    self._CreateEvalerDev(cfg).EvalLatestCheckpoint()
    # EvalerDev calls report_measure, request_trial_stop, and report_done.
    after_eval_count = trial.ReportEvalMeasure.call_count
    self.assertEqual(after_eval_count, 0)

    self._CreateDecoderDev(cfg).DecodeLatestCheckpoint()
    after_decoder_count = trial.ReportEvalMeasure.call_count
    self.assertGreater(after_decoder_count, 0)

    dev_files = tf.io.gfile.glob(logdir + '/eval_dev/*')
    self.assertTrue(self._HasFile(dev_files, 'params.txt'))
    self.assertTrue(self._HasFile(dev_files, 'eval_dev.pbtxt'))
    self.assertTrue(self._HasFile(dev_files, 'tfevents'))
    self.assertTrue(self._HasFile(dev_files, 'score'))
    self.assertTrue(
        self._HasLine(self._GetMatchedFileName(dev_files, 'score'), 'log_pplx'))


class ProcessFPropResultsTest(BaseTrainerTest):

  @flagsaver.flagsaver
  def testIdentityRegressionModel(self):
    logdir = os.path.join(tf.test.get_temp_dir(),
                          'identity_regression_test' + str(random.random()))
    FLAGS.logdir = logdir

    steps = 100
    cfg = trainer_test_utils.IdentityRegressionModel.Params()
    cfg.cluster.task = 0
    cfg.cluster.mode = 'sync'
    cfg.cluster.job = 'trainer_client'
    cfg.cluster.worker.name = '/job:localhost'
    cfg.cluster.worker.replicas = 1
    cfg.cluster.worker.gpus_per_replica = 0
    cfg.cluster.ps.name = '/job:localhost'
    cfg.cluster.ps.replicas = 1
    cfg.cluster.ps.gpus_per_replica = 0
    cfg.train.max_steps = steps
    cfg.task.train.learning_rate = 0.025

    runners = [self._CreateController(cfg), self._CreateTrainer(cfg)]

    runner_manager = trainer.RunnerManager(cfg.name)
    runner_manager.StartRunners(runners)
    train = runners[1]

    # ProcessFPropResults should have been called <steps> times on the task
    # and <steps> times on the model.

    # There are always 2 samples in the batch.
    expected_samples_in_batch = [(2, 1.0) for _ in range(steps)]

    self.assertAllEqual(
        expected_samples_in_batch,
        [m['num_samples_in_batch'] for m in train._model.metrics])
    self.assertAllEqual(
        expected_samples_in_batch,
        [m['num_samples_in_batch'] for m in train._model._task.metrics])

    # Global steps should increment by 1 for each batch.
    expected_global_steps = [i + 1 for i in range(steps)]
    self.assertAllEqual(expected_global_steps, train._model.global_steps)
    self.assertAllEqual(expected_global_steps, train._model._task.global_steps)

    # The CountingInputGenerator makes [2,2] inputs that increment for each
    # batch, like:
    #   [[0, 1], [2, 3]],
    #   [[4, 5], [6, 7]],
    #   ...
    expected_input_tensors = [{
        'input': np.array([[4 * i, 4 * i + 1], [4 * i + 2, 4 * i + 3]])
    } for i in range(steps)]

    def keep_input_tensors(tensors):
      return [{'input': d['input']} for d in tensors]

    self.assertAllClose(
        expected_input_tensors,
        keep_input_tensors(train._model.result_per_example_tensors))
    self.assertAllClose(
        expected_input_tensors,
        keep_input_tensors(train._model._task.result_per_example_tensors))

    # This model is training parameters m and b such that:
    #    m * (input[0] + input[1]) + b = (input[0] + input[1])
    # So we expect m = 1 and b = 0 after training.

    # m is more stable so that's the one we test with a tight tolerance.
    self.assertNear(
        1.0, train._model._task.result_per_example_tensors[-1]['m'][0], 0.1)
    self.assertNear(1.0, train._model.result_per_example_tensors[-1]['m'][0],
                    0.1)

    # b isn't so stable but shouldn't be too crazy in size.
    self.assertNear(
        0.0, train._model._task.result_per_example_tensors[-1]['b'][0], 10.0)
    self.assertNear(0.0, train._model.result_per_example_tensors[-1]['b'][0],
                    10.0)


if __name__ == '__main__':
  tf.test.main()
