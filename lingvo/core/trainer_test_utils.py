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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import py_utils


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
    p = super(CountingInputGenerator, cls).Params()
    p.Delete('batch_size')
    p.Define('batch_size', 2, 'batch size')
    p.Define('shape', [2, 2], 'source shape.')
    return p

  def __init__(self, params):
    super(CountingInputGenerator, self).__init__(params)
    self.shape = params.shape

  def InputBatch(self):
    length = tf.reduce_prod(self.shape)
    counter = base_model.StatsCounter('CountingInputGenerator')
    new_value = tf.cast(counter.IncBy(None, length), dtype=tf.int32) - length
    new_value = tf.stop_gradient(new_value)
    values = new_value + tf.range(length)
    shaped_values = tf.reshape(tf.cast(values, dtype=tf.float32), self.shape)
    targets = tf.reduce_sum(shaped_values, axis=0)
    return py_utils.NestedMap(src_ids=shaped_values, tgt_ids=targets)


class IdentityRegressionTask(base_model.BaseTask):
  """A simple regression task for testing."""

  @base_layer.initializer
  def __init__(self, params):
    super(IdentityRegressionTask, self).__init__(params)
    with tf.variable_scope('IdentityRegressionTask'):
      self.CreateVariable(
          'm',
          py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Uniform()))
      self.CreateVariable(
          'b',
          py_utils.WeightParams(shape=[], init=py_utils.WeightInit.Uniform()))
    self.global_steps = []
    self.metrics = []
    self.result_per_example_tensors = []

  def ComputePredictions(self, theta, input_batch):
    """sum(m * x) + b."""
    return tf.reduce_sum(theta.m * input_batch.src_ids, axis=1) + theta.b

  def ComputeLoss(self, theta, input_batch, predicted):
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


class IdentityRegressionModel(base_model.SingleTaskModel):
  """Simple regression model for testing."""

  @base_layer.initializer
  def __init__(self, params):
    super(IdentityRegressionModel, self).__init__(params)
    self.global_steps = []
    self.metrics = []
    self.result_per_example_tensors = []

  @classmethod
  def Params(cls):
    p = super(IdentityRegressionModel, cls).Params()
    p.name = 'IdentityRegressionModel'
    p.input = CountingInputGenerator.Params()
    p.task = IdentityRegressionTask.Params()
    return p

  def ProcessFPropResults(self, sess, global_step, metrics,
                          per_example_tensors):
    self.global_steps.append(global_step)
    self.metrics.append(metrics)
    self.result_per_example_tensors.append(per_example_tensors)
