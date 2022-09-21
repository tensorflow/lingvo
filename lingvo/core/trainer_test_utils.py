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
"""Toy models and input generation tools for testing trainer code."""

from lingvo import model_registry
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import metrics as metrics_lib
from lingvo.core import program
from lingvo.core import py_utils
from lingvo.core import summary_utils

import numpy as np


class CountingInputGenerator(base_input_generator.BaseInputGenerator):
  """Produces deterministic inputs for IdentityRegressionModel.

  src_ids increment by 1, so a 2x2 batch would look like:
      [[0, 1], [2, 3]]
  and the next batch would be:
      [[4, 5], [6, 7]]

  Targets are the sum of the src_ids:
      [1, 5]
  next batch:
      [9, 13]

  Since `sum(src_ids) = target`, we expect that the regression model of
  `target = sum(m * src_ids) + b` will learn `m = 1` and `b = 0`.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Delete('batch_size')
    p.Define('batch_size', 2, 'batch size')
    p.Define('shape', [2, 2], 'source shape.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.shape = params.shape

  def _InputBatch(self):
    length = tf.reduce_prod(self.shape)
    counter = summary_utils.StatsCounter('CountingInputGenerator')
    new_value = tf.cast(counter.IncBy(length), dtype=tf.int32) - length
    new_value = tf.stop_gradient(new_value)
    values = new_value + tf.range(length)
    shaped_values = tf.reshape(tf.cast(values, dtype=tf.float32), self.shape)
    targets = tf.reduce_sum(shaped_values, axis=0)
    return py_utils.NestedMap(src_ids=shaped_values, tgt_ids=targets)


class IdentityRegressionTask(base_model.BaseTask):
  """A simple regression task for testing."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('weight_init_value', 0.8, 'Initial value of the model weights.')
    p.name = 'identity_regression_task'
    return p

  def __init__(self, params):
    super().__init__(params)
    self.global_steps = []
    self.metrics = []
    self.result_per_example_tensors = []

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    self.CreateVariable(
        'm',
        py_utils.WeightParams(
            shape=[], init=py_utils.WeightInit.Constant(p.weight_init_value)))
    self.CreateVariable(
        'b',
        py_utils.WeightParams(
            shape=[], init=py_utils.WeightInit.Constant(p.weight_init_value)))

  def ComputePredictions(self, theta, input_batch):
    """sum(m * x) + b."""
    return tf.reduce_sum(theta.m * input_batch.src_ids, axis=1) + theta.b

  def ComputeLoss(self, theta, predicted, input_batch):
    diff = predicted - input_batch.tgt_ids
    per_example_loss = diff * diff
    batch_dim = py_utils.GetShape(per_example_loss)[0]

    def replicate_var(name):
      return tf.convert_to_tensor(
          [self._private_vars[name]] * batch_dim, dtype=tf.float32)

    metrics = {'loss': (tf.reduce_sum(per_example_loss), batch_dim)}
    per_example_tensors = {
        'input': input_batch.src_ids,
        'loss': per_example_loss,
        'diff': diff,
        'm': replicate_var('m'),
        'b': replicate_var('b'),
    }
    return metrics, per_example_tensors

  def FilterPerExampleTensors(self, per_example):
    return per_example

  def ProcessFPropResults(self, sess, global_step, metrics,
                          per_example_tensors):
    self.global_steps.append(global_step)
    self.metrics.append(metrics)
    self.result_per_example_tensors.append(per_example_tensors)

  def CreateDecoderMetrics(self):
    return {
        'num_samples_in_batch': metrics_lib.AverageMetric(),
        'diff': metrics_lib.AverageMetric(),
    }

  def DecodeWithTheta(self, theta, input_batch):
    diff = self.ComputePredictions(theta, input_batch) - input_batch.tgt_ids
    return {'diff': diff}

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    diff = dec_out_dict['diff']
    dec_metrics_dict['diff'].Update(np.mean(diff))
    dec_metrics_dict['num_samples_in_batch'].Update(len(diff))
    return []


class ModelTrackingFPropResults(base_model.SingleTaskModel):
  """Simple regression model."""

  def __init__(self, params, **kwargs):
    super().__init__(params, **kwargs)
    self.global_steps = []
    self.metrics = []
    self.result_per_example_tensors = []

  def ProcessFPropResults(self, sess, global_step, metrics,
                          per_example_tensors):
    self.global_steps.append(global_step)
    self.metrics.append(metrics)
    self.result_per_example_tensors.append(per_example_tensors)


def RegisterIdentityRegressionModel(  # pylint: disable=invalid-name
    batch_size=2,
    weight_init_value=0.8,
    optimizer=None,
    learning_rate=1.0,
    max_train_steps=10,
    train_steps_per_loop=2,
    eval_decode_steps_per_loop=2,
    eval_decode_samples_per_summary=10):
  """Register an IdentityRegressionTask model with given configuration.

  Args:
    batch_size: batch size of CountingInputGenerator.
    weight_init_value: constant init value for the model varialbes.
    optimizer: if set, the optimizer params to use.
    learning_rate: the learning rate to use.
    max_train_steps: maximum training steps.
    train_steps_per_loop: number of training steps per TPU loop.
    eval_decode_steps_per_loop: number of evaluation/decode steps per TPU loop.
    eval_decode_samples_per_summary: number of samples to eval/decode for each
      checkpoint.
  """

  class IdentityRegressionModel(base_model_params.SingleTaskModelParams):
    """Model params for IdentityRegressionTask."""

    def Train(self):
      return CountingInputGenerator.Params().Set(batch_size=batch_size)

    def Test(self):
      return CountingInputGenerator.Params().Set(batch_size=batch_size)

    def Task(self):
      p = IdentityRegressionTask.Params().Set(
          weight_init_value=weight_init_value)
      if optimizer:
        p.train.optimizer = optimizer
      p.train.learning_rate = learning_rate
      p.train.max_steps = max_train_steps
      p.train.tpu_steps_per_loop = train_steps_per_loop
      p.eval.samples_per_summary = eval_decode_samples_per_summary
      p.eval.decoder_samples_per_summary = eval_decode_samples_per_summary
      return p

    def Model(self):
      return ModelTrackingFPropResults.Params(self.Task())

    def ProgramSchedule(self):
      p = program.SimpleProgramScheduleForTask(
          train_dataset_name='Train',
          train_steps_per_loop=train_steps_per_loop,
          eval_dataset_names=['Test'],
          eval_steps_per_loop=eval_decode_steps_per_loop,
          decode_steps_per_loop=eval_decode_steps_per_loop)
      if max_train_steps == 0:
        p.train_executions_per_eval = 0
      return p

  model_registry.RegisterSingleTaskModel(IdentityRegressionModel)
