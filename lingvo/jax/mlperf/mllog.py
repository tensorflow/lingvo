# Lint as: python3
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
"""MLPerf-related utilities for the Lingvo Jax submission."""

from typing import List, Tuple, Optional

import jax
from lingvo.jax import model as model_lib
from lingvo.jax import py_utils
import numpy as np

from mlperf.logging.mlperf_logging.mllog import mllog

Metrics = model_lib.Metrics
InstantiableParams = py_utils.InstantiableParams


class MLLogger:
  """Wrapper around the MLPerf logger to reduce verbosity."""

  _UNDEFINED = 'TBD'

  def __init__(self, benchmark: str, model: str = 'lingvo_jax') -> None:
    """Constructor.

    Args:
      benchmark: The name of the submission benchmark.
      model: The name of the submission model.
    """
    self._mllogger = mllog.MLLogger()
    self._global_batch_size = None
    self._eval_interval_steps = None
    self._init_done = False
    self._num_train_samples = None
    self._max_train_steps = None
    self._target_accuracy = None

    device_kind_str = jax.devices()[0].device_kind.lower()
    device_kind_str = '-'.join(device_kind_str.split())
    platform_str = f'{device_kind_str}-{jax.device_count()}'
    self._mllogger.event('cache_clear')
    self._mllogger.start('init_start')
    self._mllogger.event('submission_org', 'Google')
    self._mllogger.event('submission_platform', platform_str)
    self._mllogger.event('submission_status', 'research')
    self._mllogger.event('submission_benchmark', benchmark)
    self._mllogger.event('submission_model', model)
    self._mllogger.event('submission_division', 'open')

  def initialize(self, model_p: InstantiableParams,
                 train_input_p: InstantiableParams,
                 eval_input_p: InstantiableParams,
                 max_train_steps: Optional[int],
                 target_accuracy: Optional[float]) -> None:
    """Initializes and logs constants."""
    if 'batch_size' in train_input_p:
      self._global_batch_size = train_input_p.batch_size * jax.process_count()
    else:
      raise ValueError('MLPerf logging requires to infer `global_batch_size`.')
    train_p = model_p.train
    if 'eval_interval_steps' in train_p:
      self._eval_interval_steps = train_p.eval_interval_steps
    else:
      raise ValueError(
          'MLPerf logging requires to extract `eval_interval_steps`.')
    if 'num_samples' in train_input_p:
      self._num_train_samples = train_input_p.num_samples
    else:
      self._num_train_samples = self._UNDEFINED
    if 'num_samples' in eval_input_p:
      num_eval_samples = eval_input_p.num_samples
    else:
      num_eval_samples = self._UNDEFINED
    self._max_train_steps = max_train_steps
    self._target_accuracy = target_accuracy
    self._mllogger.event('global_batch_size', self._global_batch_size)
    self._mllogger.event('train_samples', self._num_train_samples)
    self._mllogger.event('eval_samples', num_eval_samples)
    self._init_done = True

  def extract_mlperf_eval_pipeline(
      self,
      eval_input_p: List[InstantiableParams]) -> Tuple[int, InstantiableParams]:
    """Extracts the MLPerf eval input pipeline."""
    index = None
    eval_p = None
    for i, p in enumerate(eval_input_p):
      if p.name == 'test':
        if index is not None:
          raise ValueError('Found more than one MLPerf eval pipeline.')
        index = i
        eval_p = p
    if eval_p is None:
      raise ValueError('Could not find any MLPerf eval pipeline.')
    return index, eval_p

  @property
  def num_train_samples_between_eval(self) -> int:
    """Returns the number of training samples processed between eval runs."""
    if not self._init_done:
      raise ValueError('`initialize()` must be called first.')
    return self._global_batch_size * self._eval_interval_steps

  def get_train_samples_since_beginning(self, step_i: int) -> int:
    """Returns the number of training samples processed since beginning."""
    if not self._init_done:
      raise ValueError('`initialize()` must be called first.')
    return (step_i + 1) * self._global_batch_size

  def log_run_start(self) -> None:
    """Logs initialization completion and run start."""
    self._mllogger.start('init_stop')
    self._mllogger.start('run_start')

  def log_block_start(self, step_i: int) -> None:
    """Logs a `block_start."""
    # -1 since the examples have not yet been processed.
    train_samples_since_beginning = self.get_train_samples_since_beginning(
        step_i - 1)
    self._mllogger.start(
        'block_start',
        metadata={
            'first_epoch_num':
                (train_samples_since_beginning + self._num_train_samples - 1) //
                self._num_train_samples,
            'epoch_count':
                self.num_train_samples_between_eval // self._num_train_samples +
                1,
            'train_samples_since_beginning':
                train_samples_since_beginning,
            'train_samples_since_last_eval':
                self.num_train_samples_between_eval
        })

  def log_block_stop(self, step_i: int) -> None:
    """Logs a `block_stop."""
    train_samples_since_beginning = self.get_train_samples_since_beginning(
        step_i)
    self._mllogger.end(
        'block_stop',
        metadata={
            'first_epoch_num':
                (train_samples_since_beginning + self._num_train_samples - 1) //
                self._num_train_samples,
            'epoch_count':
                self.num_train_samples_between_eval // self._num_train_samples +
                1,
            'train_samples_since_beginning':
                train_samples_since_beginning,
            'train_samples_since_last_eval':
                self.num_train_samples_between_eval
        })

  def log_eval_start(self, step_i: int) -> None:
    """Logs a `eval_start."""
    train_samples_since_beginning = self.get_train_samples_since_beginning(
        step_i)
    self._mllogger.start(
        'eval_start',
        metadata={
            'epoch_num':
                (train_samples_since_beginning + self._num_train_samples - 1) //
                self._num_train_samples,
            'train_samples_since_beginning':
                train_samples_since_beginning
        })

  def log_eval_accuracy_stop(self, step_i: int, metrics: Metrics) -> float:
    """Logs a `eval_accuracy` and `eval_stop`."""
    if 'fraction_of_correct_preds' in metrics:
      # Metrics are 2-tuples (value, weight), values being an array.
      metric_values, metric_weights = metrics['fraction_of_correct_preds']
      sum_metric_weights = np.sum(metric_weights)
      eval_accuracy = np.sum(
          metric_values * metric_weights) / sum_metric_weights
    else:
      raise ValueError(
          'Could not find `fraction_of_correct_preds` in metrics dict.')
    train_samples_since_beginning = self.get_train_samples_since_beginning(
        step_i)
    self._mllogger.event(
        'eval_accuracy',
        eval_accuracy,
        metadata={
            'epoch_num':
                (train_samples_since_beginning + self._num_train_samples - 1) //
                self._num_train_samples,
            'train_samples_since_beginning':
                train_samples_since_beginning
        })
    self._mllogger.end(
        'eval_stop',
        metadata={
            'epoch_num':
                (train_samples_since_beginning + self._num_train_samples - 1) //
                self._num_train_samples,
            'train_samples_since_beginning':
                train_samples_since_beginning
        })
    return eval_accuracy

  def check_termination_criteria(self, step_i: int, accuracy: float) -> bool:
    """Checks whether we reach the termination criteria."""
    if self._target_accuracy is not None and accuracy >= self._target_accuracy:
      self._mllogger.end('run_stop', metadata={'status': 'success'})
      return True
    elif self._max_train_steps is not None and step_i >= self._max_train_steps:
      self._mllogger.end('run_stop', metadata={'status': 'abort'})
      return True
    return False
