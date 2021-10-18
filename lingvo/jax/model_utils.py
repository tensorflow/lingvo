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
# ==============================================================================
"""Utils for training and evaluation of lingvo Jax models."""

from typing import Any, Callable, List, Optional, Union

from absl import logging
import jax
from lingvo.jax import base_model_params
from lingvo.jax import model
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax import summary_utils
from lingvo.jax import train_states
import numpy as np
import tensorflow.compat.v2 as tf

from lingvo.jax import model_imports
from lingvo.jax import model_registry

BaseModelParamsT = base_model_params.BaseModelParamsT
InstantiableParams = py_utils.InstantiableParams
Metrics = model.Metrics
NestedMap = py_utils.NestedMap
InputPipeline = Union[InstantiableParams, List[InstantiableParams]]
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
TrainState = train_states.TrainState
SummaryWriter = tf.summary.SummaryWriter


def get_model(model_name: str) -> BaseModelParamsT:
  """Retrieves a model config from the global registry."""
  model_imports.ImportParams(model_name)
  model_class = model_registry.GetModel(model_name)
  if model_class is None:
    raise ValueError(f'Could not find model `{model_name}`.')
  return model_class


@tf.function
def get_model_inputs(input_pipeline: InputPipeline,
                     split: Optional[int] = None) -> NestedMap:
  """Gets the model inputs and filters keys of type `tf.string`.

  Args:
    input_pipeline: The input pipeline from which to fetch the inputs. This can
      be a single `InstantiableParams` object, or it may be a list of params.
    split: Optional split index to get the input batch from, if the input
      pipeline is a list.

  Returns:
    The input batch with preprocessing and filtering applied.

  Raises:
    ValueError if the split is not provided when the input pipeline is a list or
    if the split requested is larger than the length of the input pipeline.
  """
  pipeline = input_pipeline
  if isinstance(input_pipeline, list):
    if split is None:
      raise ValueError('The split index must be provided when pipeline is a'
                       'list of params.')
    if split > len(input_pipeline):
      raise ValueError('The length of the input pipeline is'
                       f'{len(input_pipeline)} but the split requested is'
                       f'{split}.')
    pipeline = input_pipeline[split]

  with py_utils.InfeedContextScope(
      infeed_host_index=jax.process_index(),
      num_infeed_hosts=jax.process_count()):
    inputs = pipeline.GetPreprocessedInputBatch()
  # Remove unsupported string (byte) array from input.
  inputs = inputs.Filter(lambda v: v.dtype != tf.string)
  return inputs


def run_eval_one_step(eval_inputs: NestedJTensor,
                      eval_step: Callable[[NestedJTensor], Any],
                      reshard_inputs: Optional[bool] = False):
  """Runs eval on entire batch of eval inputs or for one step.

  Args:
    eval_inputs: `NestedJTensor` of eval inputs.
    eval_step: The eval step which evaluates the model on eval inputs.
    reshard_inputs: Whether to reshard inputs (in pmap) or not.

  Returns:
    Tuple of eval loss, mean metrics and eval summaries.
  """
  if reshard_inputs:
    eval_inputs = tf.nest.map_structure(py_utils.Reshard, eval_inputs)
  else:
    eval_inputs = tf.nest.map_structure(lambda x: x.numpy(), eval_inputs)
  loss, mean_metrics, _, summary_tensors = eval_step(eval_inputs)
  return loss, mean_metrics, summary_tensors


def run_eval_loop_over_test_splits(
    num_steps: List[int],
    eval_step: Callable[[NestedJTensor], Any],
    summary_writer: SummaryWriter,
    summary_eval_dirs: List[str],
    step: int,
    model_inputs_fn: Callable[[Optional[int]], NestedMap],
    reshard_inputs: Optional[bool] = False) -> List[Metrics]:
  """Run evaluation in a loop over a list of test sets.

  Args:
    num_steps: A list of steps for each test split to evaluate on.
    eval_step: The eval step function which to call to evaluate the model.
    summary_writer: The summary writer object to log summaries.
    summary_eval_dirs: The list of summary directories corresponding to the
      different test sets.
    step: The step at which we are evaling the model.
    model_inputs_fn: Function which returns the inputs on being called.
    reshard_inputs: Whether to reshard inputs.

  Returns:
    A list of eval metrics dictionaries (same order as eval splits/pipelines).
  """
  # If reshard_inputs = True, meaning this is called from pmap, hence we need to
  # unreplicate metrics for reporting.
  unreplicate_metrics = reshard_inputs

  metrics_output = []
  for split, num_split_steps in enumerate(num_steps):
    logging.info('Starting eval data split=%d with num_steps=%d', split,
                 num_split_steps)
    # Reset loss and summary tensors for each test split.
    loss = []
    summary_tensors = {}
    metrics = {}
    step_num = 0
    # Use num_split_steps < 0 to indicate running all of the input until
    # out of range.
    while num_split_steps < 0 or step_num < num_split_steps:
      step_num += 1
      try:
        eval_loss, eval_metrics, eval_summary_tensors = run_eval_one_step(
            model_inputs_fn(split), eval_step, reshard_inputs=reshard_inputs)
      except tf.errors.OutOfRangeError:
        if num_split_steps > 0:
          raise
        logging.info('Exhausted eval data split=%d after %d steps', split,
                     step_num - 1)
        break
      if unreplicate_metrics:
        # In pmap, metrics has already been aggregated on tpu.
        eval_metrics = jax.tree_map(lambda x: x[0], eval_metrics)
      loss += [eval_loss]
      for k in eval_summary_tensors:
        if k in summary_tensors:
          summary_tensors[k] += [eval_summary_tensors[k]]
        else:
          summary_tensors[k] = [eval_summary_tensors[k]]
      for k in eval_metrics:
        if k in metrics:
          metrics[k] += [eval_metrics[k]]
        else:
          metrics[k] = [eval_metrics[k]]
    loss = np.array(loss)
    for k in summary_tensors:
      summary_tensors[k] = np.array(summary_tensors[k])
    for k in metrics:
      value = np.array([metric[0] for metric in metrics[k]])
      weight = np.array([metric[1] for metric in metrics[k]])
      metrics[k] = (value, weight)
    loss = np.mean(loss, axis=0)
    logging.info('step_i: %d, eval test split %s loss: %s', step, split, loss)

    for key, value in metrics.items():
      metric_values = value[0]
      metric_weights = value[1]
      sum_metric_weights = np.sum(metric_weights)
      weighted_average = np.sum(
          metric_values * metric_weights) / sum_metric_weights
      logging.info('  %s=%f (weight=%f)', key, weighted_average.item(),
                   sum_metric_weights.item())

    with summary_writer(summary_eval_dirs[split]) as eval_test_summary_writer:
      summary_utils.WriteSummaryEntry(
          eval_test_summary_writer,
          step,
          loss,
          metrics,
          summary_tensors,
          # Metrics have already been unreplicated above.
          unreplicate_metrics=False)
    metrics_output.append(metrics)
  return metrics_output
