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
"""Utils for TF Summaries."""

import collections
import contextlib
import operator
import textwrap
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from absl import logging
import jax
from jax import numpy as jnp
from lingvo.jax import pytypes
from lingvo.jax import train_states
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import summary as tf_summary

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
TrainState = train_states.TrainState
SummaryWriter = tf.summary.SummaryWriter


# Copied from flax.core.FrozenDict and customized for lists.
def PrettyRepr(values: NestedJTensor, num_spaces: int = 4) -> str:
  """Returns an indented representation of the nested dictionary."""

  def Indent(txt: str) -> str:
    return textwrap.indent(txt, ' ' * num_spaces)

  if isinstance(values, dict):
    rep = []
    for key, val in values.items():
      rep.append(f'{key}: {PrettyRepr(val)},\n')
    if rep:
      return '{\n' + Indent(''.join(rep)) + '}'
    else:
      return '{}'
  elif isinstance(values, (list, tuple)):
    rep = []
    for v in values:
      rep.append(f'{PrettyRepr(v)},\n')
    if rep:
      return '[\n' + Indent(''.join(rep)) + ']'
    else:
      return '[]'
  else:
    return repr(values)


def PrettyReprShapes(replicated_vars: NestedJTensor, is_vars_replicated) -> str:
  """Returns a pretty representation of the variable shapes."""

  def Pps(x: JTensor) -> str:
    """Remove leading dim from replicated model vars."""
    if is_vars_replicated:
      return 'x'.join(str(e) for e in x.shape[1:])
    else:
      # If var is not replicated, no need to remove the first dim.
      return 'x'.join(str(e) for e in x.shape)

  out = jax.tree_map(Pps, replicated_vars)
  out = PrettyRepr(out)
  for c in '{}(),[]':
    out = out.replace(c, '')
  return '\n'.join(l for l in out.splitlines() if (l and not l.isspace()))


def _YieldSubtrees(
    root: NestedJTensor,
    max_level: int,
    level: int = 0,
    name: Tuple[str, ...] = (),
) -> Generator[Tuple[Tuple[str, ...], NestedJTensor], None, None]:
  """Yields subtrees up to max_level."""
  if level < max_level:
    if isinstance(root, dict):
      for key in root:
        for out in _YieldSubtrees(root[key], max_level, level + 1,
                                  name + (key,)):
          yield out
    elif isinstance(root, (list, tuple)):
      list_len = len(root)
      for ii in range(list_len):
        for out in _YieldSubtrees(root[ii], max_level, level + 1,
                                  name + (str(ii),)):
          yield out
    else:
      # TODO(yonghui): Support other common composite types.
      yield (name, root)
  else:
    if root:
      yield (name, root)


def L2Norms(tree: NestedJTensor,
            prefix: str = '',
            max_level: int = 4,
            sep: str = '/') -> Dict[str, jnp.float32]:
  """L2 Norms over pytree."""
  squares = jax.tree_map(lambda x: jnp.array([x.size, jnp.sum(x**2)]), tree)
  names, squares = zip(*_YieldSubtrees(squares, max_level=max_level))
  names = [sep.join(name) for name in names]
  if prefix:
    names = [prefix + sep + n for n in names]

  def NormFn(tree: NestedJTensor) -> jnp.float32:
    out = jax.tree_util.tree_reduce(operator.add, tree)
    # NOTE(yonghui): Here we normalize out[1] by out[0], instead of sqrt(out[1])
    # by out[0] so that l2_norm is more semantically meaningful: it means the
    # average scale of params. In addition, this normalization makes sure norm
    # is invariant to number of model replicas (in pmap training).
    # TODO(yonghui): In the future, compute mean and std instead.
    return jnp.sqrt(out[1] / out[0])

  norms = [NormFn(tree) for tree in squares]
  return dict(zip(names, norms))


@contextlib.contextmanager
def GetSummaryWriter(summary_dir: str) -> SummaryWriter:
  """Context manager around Tensorflow's SummaryWriter."""
  if jax.process_index() == 0:
    logging.info('Opening SummaryWriter `%s`...', summary_dir)
    summary_writer = tf_summary.create_file_writer(summary_dir)
  else:
    # We create a dummy tf.summary.SummaryWriter() on non-zero tasks. This will
    # return a mock object, which acts like a summary writer, but does nothing,
    # such as writing event to disk.
    logging.info('Opening a mock-like SummaryWriter.')
    summary_writer = tf_summary.create_noop_writer()
  try:
    yield summary_writer
  finally:
    summary_writer.close()
    if jax.process_index() == 0:
      logging.info('Closed SummaryWriter `%s`.', summary_dir)
    else:
      logging.info('Closed a mock-like SummaryWriter.')


def FlattenSummaryDict(summary_dict: Dict[str, JTensor],
                       parent_key: Optional[str] = None) -> List[Any]:
  """Flattens a summary dictionary."""
  separator = '@'
  outputs = []
  for key, value in summary_dict.items():
    if parent_key is not None:
      key = f'{parent_key}{separator}{key}'
    if isinstance(value, collections.MutableMapping):
      outputs.extend(FlattenSummaryDict(value, key))
    else:
      outputs.append((key, value))
  return outputs


def WriteSummaryEntry(summary_writer: SummaryWriter,
                      step_i: int,
                      loss: JTensor,
                      metrics: Dict[str, JTensor],
                      summary_tensors: NestedJTensor,
                      unreplicate_metrics: bool,
                      steps_per_sec: Optional[float] = None) -> None:
  """Writes a summary entry into the provided SummaryWriter."""
  # Scalar values must be plain Python types rather than e.g. np.int / np.float.
  if unreplicate_metrics:
    # We already aggregate metrics in trainer/evaler loop, and hence metric
    # data are replicated over all cores. Here we explicitly unreplicate the
    # metric value before logging or being added to summaries.
    metrics = jax.tree_map(lambda x: x[0], metrics)

  mean_loss = np.mean(loss).item()
  with summary_writer.as_default():
    tf_summary.scalar('loss', mean_loss, step_i)
    if steps_per_sec is not None:
      tf_summary.scalar('Steps/sec', steps_per_sec, step_i)

    logging.info('Metrics values at step %d:', step_i)
    logging.info('  loss=%f', mean_loss)
    for key, value in metrics.items():
      assert len(value) == 2, (
          'Metric value should be a pair of (value, weight).')
      metric_values = value[0]
      metric_weights = value[1]
      sum_metric_weights = np.sum(metric_weights)
      weighted_average = np.sum(
          metric_values * metric_weights) / sum_metric_weights
      logging.info('  %s=%f (weight=%f)', key, weighted_average.item(),
                   sum_metric_weights.item())
      tf_summary.scalar(f'Metrics/{key}', weighted_average.item(), step_i)
      tf_summary.scalar(f'Metrics/{key}-weight', sum_metric_weights.item(),
                        step_i)
    summaries = FlattenSummaryDict(summary_tensors)
    # TODO(shafey): Add support for non-scalar summaries.
    for key, tensor in summaries:
      mean_tensor = np.mean(tensor).item()
      tf_summary.scalar(key, mean_tensor, step_i)
  # Lastly flush summaries.
  summary_writer.flush()
  logging.info('Wrote summary entry at step `%d` (loss=`%f`).', step_i,
               mean_loss)


def WriteModelStructure(train_summary_writer: SummaryWriter,
                        train_state: TrainState, is_vars_replicated):
  """Writes the Model Param structure to TB."""
  with train_summary_writer.as_default():
    out = PrettyReprShapes(train_state.mdl_vars, is_vars_replicated)
    tf_summary.text('Model', out, step=0)
  train_summary_writer.flush()


def WriteTotalNumParams(train_summary_writer: SummaryWriter,
                        total_num_params: int):
  """Writes the total number of parameters to TB."""
  with train_summary_writer.as_default():
    # Add whitespace every 3 digit for readability.
    num_params_str = '{:,}'.format(total_num_params).replace(',', ' ')
    tf_summary.text('Total Num Params', num_params_str, step=0)
  train_summary_writer.flush()


def WriteSummaryEveryNSteps(train_state: TrainState,
                            train_summary_writer: SummaryWriter, step_i: int,
                            summary_every_n_steps: int, loss: JTensor,
                            metrics: NestedJTensor,
                            per_example_out: NestedJTensor,
                            summary_tensors: NestedJTensor,
                            norm_summary_every_step: int,
                            summary_last_time: Optional[float],
                            summary_last_step: Optional[int],
                            unreplicate_mdl_vars: bool,
                            unreplicate_metrics: bool) -> bool:
  """Writes summaries at regular intervals."""
  result = False

  if step_i % summary_every_n_steps == summary_every_n_steps - 1:
    logging.info('step_i: %d, training loss: %s', step_i, loss)
    logging.info('metrics: %s', metrics)
    logging.info('per_example_out: %s', per_example_out)
    logging.info('summary_tensors: %s', summary_tensors)

    duration_sec = time.time() - summary_last_time
    num_steps = step_i - summary_last_step
    steps_per_sec = num_steps / duration_sec
    logging.info('steps/sec: %f', steps_per_sec)

    WriteSummaryEntry(train_summary_writer, step_i, loss, metrics,
                      summary_tensors, unreplicate_metrics, steps_per_sec)
    result = True

  # Write detailed Var norms to TB.
  if step_i % norm_summary_every_step == 0:
    mdl_vars = train_state.mdl_vars
    if unreplicate_mdl_vars:
      # Compute stats over the first replica only.
      mdl_vars = jax.tree_map(lambda x: x[0], train_state.mdl_vars)
    else:
      # This is an SPMD model, mdl_vars can be sharded, not replicated.
      mdl_vars = train_state.mdl_vars
    norms = L2Norms(mdl_vars, prefix='Vars', max_level=10)
    with train_summary_writer.as_default():
      for name in norms:
        tf_summary.scalar(name, norms[name], step_i)

  return result
