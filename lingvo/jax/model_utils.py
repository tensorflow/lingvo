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
from lingvo.jax import base_input
from lingvo.jax import base_model
from lingvo.jax import base_model_params
from lingvo.jax import model_imports
from lingvo.jax import model_registry
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax import summary_utils
from lingvo.jax import train_states
import numpy as np
import tensorflow.compat.v2 as tf

BaseModelParamsT = base_model_params.BaseModelParamsT
InstantiableParams = py_utils.InstantiableParams
Metrics = base_model.Metrics
NestedMap = py_utils.NestedMap
InputPipeline = Union[InstantiableParams, List[InstantiableParams]]
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
TrainState = train_states.TrainState
SummaryWriter = tf.summary.SummaryWriter


def get_model(model_name: str) -> BaseModelParamsT:
  """Retrieves a model config from the global registry."""
  model_imports.import_params(model_name)
  model_class = model_registry.get_model(model_name)
  if model_class is None:
    raise ValueError(f'Could not find model `{model_name}`.')
  return model_class


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
    eval_inputs = tf.nest.map_structure(py_utils.reshard, eval_inputs)
  _, loss, mean_metrics, _, summary_tensors = eval_step(eval_inputs)
  return loss, mean_metrics, summary_tensors


def run_eval_loop_over_test_splits(
    num_steps: List[int],
    eval_step: Callable[[NestedJTensor], Any],
    summary_writers: List[SummaryWriter],
    step: int,
    model_inputs: List[base_input.BaseInput],
    eval_inputs_pspecs=None,
    eval_inputs_shape=None,
    global_mesh=None,
    reshard_inputs: Optional[bool] = False) -> List[Metrics]:
  """Run evaluation in a loop over a list of test sets.

  Args:
    num_steps: A list of steps for each test split to evaluate on.
    eval_step: The eval step function which to call to evaluate the model.
    summary_writers: The summary writer objects to log summaries.
    step: The step at which we are evaling the model.
    model_inputs: List of BaseInput instances.
    eval_inputs_pspecs: PartitionSpec for eval inputs.
    eval_inputs_shape: Global shape of eval inputs
    global_mesh: Device mesh used by pjit.
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
        eval_inputs = model_inputs[split].get_next()
        if jax.config.jax_parallel_functions_output_gda:
          py_utils.assert_same_shape_and_dtype(
              eval_inputs_shape,
              tf.nest.map_structure(py_utils.get_global_input_shape_dtype,
                                    eval_inputs))
          eval_inputs = py_utils.make_array(
              eval_inputs,
              eval_inputs_shape,  # pytype: disable=wrong-arg-types  # jax-ndarray
              global_mesh,
              eval_inputs_pspecs,
          )
        eval_loss, eval_metrics, eval_summary_tensors = run_eval_one_step(
            eval_inputs, eval_step, reshard_inputs=reshard_inputs)
        eval_loss = py_utils.maybe_unreplicate_gda(eval_loss)
        eval_metrics = py_utils.maybe_unreplicate_gda(eval_metrics)
        eval_summary_tensors = py_utils.maybe_unreplicate_gda(
            eval_summary_tensors)
      except (tf.errors.OutOfRangeError, StopIteration):
        if num_split_steps > 0:
          raise
        logging.info('Exhausted eval data split=%d after %d steps', split,
                     step_num - 1)
        model_inputs[split].reset()
        break
      if unreplicate_metrics:
        # In pmap, metrics has already been aggregated on tpu.
        eval_metrics = jax.tree_map(lambda x: x[0], eval_metrics)
        eval_summary_tensors = jax.tree_map(lambda x: x[0],
                                            eval_summary_tensors)
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
      value = np.stack([metric[0] for metric in metrics[k]])
      weight = np.stack([metric[1] for metric in metrics[k]])
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
    summary_utils.write_summary_entry(
        summary_writers[split],
        step,
        loss,
        metrics,
        summary_tensors,
        # Metrics have already been unreplicated above.
        unreplicate_metrics=False)
    metrics_output.append(metrics)
  return metrics_output  # pytype: disable=bad-return-type  # py310-upgrade
