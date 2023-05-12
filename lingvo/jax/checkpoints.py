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
"""Checkpointing-related utilities to handle TrainState instances."""

from concurrent import futures
import datetime
import functools
import os
import re
from typing import Optional

from absl import logging
from flax import jax_utils
from flax.training import checkpoints
import jax
from jax import sharding
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serialization
# Internal import
from lingvo.jax import asserts
from lingvo.jax import checkpoint_pb2
from lingvo.jax import py_utils
from lingvo.jax import train_states
import numpy as np
import tensorflow.compat.v2 as tf

CHECKPOINT_PREFIX = 'checkpoint_'
TMP_PREFIX = 'tmp_'
CHECKPOINT_PATTERN_RE = re.compile(rf'{CHECKPOINT_PREFIX}[\d]+$')
TMP_CHECKPOINT_PATTERN_RE = re.compile(
    rf'{TMP_PREFIX}[\d]+.{CHECKPOINT_PREFIX}[\d]+$')
# Large value to disable flax-specific checkpoint management.
_MAX_CHECKPOINT_FLAX = 1000000

CheckpointType = checkpoint_pb2.CheckpointType
InstantiableParams = py_utils.InstantiableParams


def _is_checkpoint_asset(x: str) -> bool:
  return bool(CHECKPOINT_PATTERN_RE.match(os.path.basename(x)))


def _is_tmp_checkpoint_asset(x: str) -> bool:
  return bool(TMP_CHECKPOINT_PATTERN_RE.match(os.path.basename(x)))


def _to_timestamp(datetime_instance: datetime.datetime) -> int:
  """Converts a datetime instance into an int timestamp."""
  timedelta = datetime_instance - datetime.datetime.fromtimestamp(0)
  return int(round(timedelta.total_seconds()))


def _make_checkpoint_step_dir(
    checkpoint_dir: str,
    step: int,
) -> str:
  return os.path.join(checkpoint_dir, f'{CHECKPOINT_PREFIX}{step:08d}')


def _make_tmp_checkpoint_dir(checkpoint_dir: str,
                             step: int,
                             sync_timestamp: bool = False) -> str:
  timestamp = _to_timestamp(datetime.datetime.utcnow())
  if sync_timestamp:
    timestamp = multihost_utils.broadcast_one_to_all(np.array(timestamp))
    multihost_utils.assert_equal(timestamp,
                                 "Timestamps across hosts don't match.")
  tmp_prefix = f'{TMP_PREFIX}{timestamp}'
  return os.path.join(checkpoint_dir,
                      f'{tmp_prefix}.{CHECKPOINT_PREFIX}{step:08d}')


def get_step_from_checkpoint_asset(checkpoint_dir: str) -> int:
  if _is_tmp_checkpoint_asset(checkpoint_dir):
    end_of_tmp = checkpoint_dir.rfind('.')
    return int(checkpoint_dir[end_of_tmp + 1 + len(CHECKPOINT_PREFIX):])
  return int(os.path.basename(checkpoint_dir)[len(CHECKPOINT_PREFIX):])


def retrieve_checkpoint_type(multi_host_checkpointing: bool,
                             maybe_use_persistence_checkpointing,
                             task_p: InstantiableParams) -> CheckpointType:
  """Retrieves the CheckpointType given the input arguments."""
  if jax.config.jax_parallel_functions_output_gda:
    asserts.eq(multi_host_checkpointing, True)
    checkpoint_type = CheckpointType.CHECKPOINT_GDA
  elif maybe_use_persistence_checkpointing and task_p.model.device_mesh is not None:
    asserts.eq(multi_host_checkpointing, False)
    checkpoint_type = CheckpointType.CHECKPOINT_PERSISTENCE
  else:  # Flax-based checkpointing
    if multi_host_checkpointing:
      checkpoint_type = CheckpointType.CHECKPOINT_MULTI_HOST_FLAX
    else:
      checkpoint_type = CheckpointType.CHECKPOINT_FLAX
  return checkpoint_type


def save_checkpoint(
    train_state: train_states.TrainState,
    checkpoint_dir: str,
    overwrite: bool = False,
    unreplicate: bool = True,
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
    state_specs: Optional[train_states.TrainState] = None,
) -> None:
  """Saves a checkpoint into the provided base directory.

  This is typically called on a replicated TrainState instance.

  Args:
    train_state: The TrainState instance to save.
    checkpoint_dir: The base directory from where to retrieve checkpoints.
    overwrite: Whether to overwrite existing checkpoints files if a checkpoint
      at the current or a later step already exists.
    unreplicate: Whether to unreplicate variables (Optional). If using SPMD
      sharding, then this should be set to False.
    checkpoint_type: The checkpoint type (implementation) to save. Either
      `CHECKPOINT_FLAX`, `CHECKPOINT_MULTI_HOST_FLAX`, `CHECKPOINT_GDA` or
      `CHECKPOINT_PERSISTENCE`.
    state_specs: Currently unused.

  Raises:
    ValueError: If the global step has an unexpected shape, if `state_specs`
    is not specified for persistence-based checkpointing or if
    `checkpoint_type` is invalid.
  """
  del state_specs

  if jax.config.jax_parallel_functions_output_gda:
    asserts.eq(checkpoint_type, CheckpointType.CHECKPOINT_GDA)
    step = int(jax.device_get(py_utils.maybe_unreplicate_gda(train_state.step)))
    _save_checkpoint_gda(train_state, checkpoint_dir, overwrite, step)
    return

  if train_state.step.ndim == 0:  # pytype: disable=attribute-error  # jax-ndarray
    step = jax.device_get(train_state.step)
  elif train_state.step.ndim == 1:  # pytype: disable=attribute-error  # jax-ndarray
    step = jax.device_get(train_state.step[0])
  else:
    raise ValueError(
        f'Expecting a replicated 1D global step (got `{train_state.step.ndim}`).'  # pytype: disable=attribute-error  # jax-ndarray
    )

  if checkpoint_type in {
      CheckpointType.CHECKPOINT_FLAX, CheckpointType.CHECKPOINT_MULTI_HOST_FLAX
  }:
    use_multi_host = (
        checkpoint_type == CheckpointType.CHECKPOINT_MULTI_HOST_FLAX)
    _save_checkpoint_flax(train_state, checkpoint_dir, overwrite, unreplicate,
                          step, use_multi_host)
  else:
    raise ValueError(f'Unexpected checkpoint_type `{checkpoint_type}`.')


def latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
  """Gets the path to the latest checkpoint.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    Path to latest checkpoint or None if there is no checkpoint.
  """
  if not tf.io.gfile.exists(checkpoint_dir):
    return None
  # Note: _is_checkpoint_asset() already filters out flax temporary checkpoints
  # that would be ending with `tmp`.
  checkpoint_assets = [
      v for v in tf.io.gfile.listdir(checkpoint_dir) if _is_checkpoint_asset(v)
  ]
  if not checkpoint_assets:
    return None
  checkpoint_assets = sorted(
      checkpoint_assets, key=get_step_from_checkpoint_asset)
  return os.path.join(checkpoint_dir, checkpoint_assets[-1])


def restore_checkpoint(
    train_state: Optional[train_states.TrainState],
    checkpoint_dir: str,
    global_mesh: Optional[jax.sharding.Mesh] = None,
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
    state_specs: Optional[train_states.TrainState] = None,
    step: Optional[int] = None) -> train_states.TrainState:
  """Restores a checkpoint from the provided base directory.

  This is typically called on an unreplicated TrainState instance.

  Args:
    train_state: The TrainState instance to restore.
    checkpoint_dir: The base directory from where to retrieve checkpoints.
    global_mesh: The global mesh representing devices across multiple processes.
    checkpoint_type: The checkpoint type (implementation) to restore. Either
      `CHECKPOINT_FLAX`, `CHECKPOINT_MULTI_HOST_FLAX`, `CHECKPOINT_GDA` or
      `CHECKPOINT_PERSISTENCE`.
    state_specs: If using a GDA-based checkpoint, the partition specs
      corresponding to this TrainState instance to restore.
    step: Step number to load a checkpoint from or None to load the latest.

  Returns:
    A restored `TrainState` instance.

  Raises:
    ValueError: When a mismatch between the current checkpoint structure and
    the saved checkpoint one is detected.
  """
  if jax.config.jax_parallel_functions_output_gda:
    asserts.eq(checkpoint_type, CheckpointType.CHECKPOINT_GDA)
    return _restore_checkpoint_gda(train_state, checkpoint_dir, global_mesh,
                                   state_specs, step)

  if train_state is not None and train_state.step.ndim != 0:  # pytype: disable=attribute-error  # jax-ndarray
    raise ValueError('Expecting an unreplicated scalar global step (got '
                     f'`{train_state.step.ndim}`).')  # pytype: disable=attribute-error  # jax-ndarray

  if checkpoint_type in {
      CheckpointType.CHECKPOINT_FLAX, CheckpointType.CHECKPOINT_MULTI_HOST_FLAX
  }:
    return _restore_checkpoint_flax(train_state, checkpoint_dir, step)
  else:
    raise ValueError(f'Unexpected checkpoint_type `{checkpoint_type}`.')


def _save_checkpoint_flax(train_state: train_states.TrainState,
                          checkpoint_dir: str, overwrite: bool,
                          unreplicate: bool, step: int,
                          use_multi_host: bool) -> None:
  """Saves a checkpoint using Flax serialization mechanism."""
  if not overwrite:
    previous_filename = latest_checkpoint(checkpoint_dir)
    if previous_filename:
      previous_step = int(previous_filename.rsplit('_', 1)[-1])
      if previous_step >= step:
        logging.warning(
            'A more recent checkpoint `%d` has already been saved compared '
            'to the current timestep `%d`. Skip saving a checkpoint.',
            previous_step, step)
        return

  # Assume data parallel-only model for now and retrieve train states
  # from the first replica only.
  def maybe_unreplicate(data):
    if unreplicate:
      return jax.device_get(jax_utils.unreplicate(data))
    else:
      return jax.device_get(data)

  # Extract/flatten data structure to store to disk. Flax requires a flattened
  # data structure to be passed to the checkpointer.
  flattened_state, pytree_state = jax.tree_flatten(
      maybe_unreplicate(train_state))
  checkpoint_target = {
      'flattened_state': flattened_state,
      # Saves a serialized version of the pytree structure to detect potential
      # mismatch caused by different versions of saver/restorer.
      'str_pytree_state': str(pytree_state),
  }

  prefix = CHECKPOINT_PREFIX
  if use_multi_host:
    # Notes:
    # 1. We currently don't broadcast / synchronize the timestamp across
    #    all the JAX processes.
    # 2. Flax checkpointing already saves the checkpoint file into a temporary
    #    file that is ultimately moved. We, hence, don't need to add a second
    #    layer of temporary files for single-host checkpointing.
    timestamp = _to_timestamp(datetime.datetime.utcnow())
    prefix = f'{TMP_PREFIX}{timestamp}.{prefix}'

  checkpoints.save_checkpoint(
      checkpoint_dir,
      checkpoint_target,
      step,
      prefix=prefix,
      keep=_MAX_CHECKPOINT_FLAX,
      overwrite=overwrite)

  if use_multi_host:
    py_utils.sync_global_devices(
        f'Renaming temporary checkpoint files at step {step} into their final '
        'destination.')
    tmp_filename = os.path.join(checkpoint_dir, f'{prefix}{step}')
    new_filename = os.path.join(checkpoint_dir, f'{CHECKPOINT_PREFIX}{step}')
    logging.debug('Renaming %s to %s', tmp_filename, new_filename)
    tf.io.gfile.rename(tmp_filename, new_filename)


def _restore_checkpoint_flax(
    train_state: train_states.TrainState,
    checkpoint_dir: str,
    step: Optional[int] = None) -> train_states.TrainState:
  """Restores a checkpoint using Flax serialization mechanism."""
  # Input the same data structure as in save_checkpoint().
  flattened_state, pytree_state = jax.tree_flatten(train_state)
  str_pytree_state = str(pytree_state)
  input_target = {
      'flattened_state': flattened_state,
      'str_pytree_state': str_pytree_state,
  }
  restored_target = checkpoints.restore_checkpoint(
      checkpoint_dir, input_target, step=step)
  restored_state = restored_target['flattened_state']
  restored_str_pytree_state = restored_target['str_pytree_state']
  if restored_str_pytree_state != str_pytree_state:
    raise ValueError(
        'Unable to restore checkpoint. A mismatch between the saved '
        'checkpoint structure and the current one has been detected '
        f'(`{restored_str_pytree_state}` vs `{str_pytree_state}`).')
  return jax.tree_unflatten(pytree_state, restored_state)


def _extract_nested_prefix_names(
    state: train_states.TrainState) -> train_states.TrainState:
  """Extracts prefix names from a TrainState data structure."""
  # CNS doesn't support square bracket in filenames.
  key_separator = '.'
  left_separator = '_'
  right_separator = ''
  return train_states.TrainState(
      step=py_utils.extract_prefixed_keys_from_nested_map(
          state.step,
          'step',
          key_separator=key_separator,
          left_separator=left_separator,
          right_separator=right_separator),
      mdl_vars=py_utils.extract_prefixed_keys_from_nested_map(
          state.mdl_vars,
          'mdl_vars',
          key_separator=key_separator,
          left_separator=left_separator,
          right_separator=right_separator),
      opt_states=py_utils.extract_prefixed_keys_from_nested_map(
          state.opt_states,
          'opt_states',
          key_separator=key_separator,
          left_separator=left_separator,
          right_separator=right_separator))


def _mkdir_path(name: str, tmp_dir: str) -> str:
  # Tensorstore does not want a trailing / in dirname.
  path = os.path.join(tmp_dir, name).rstrip('/')
  # Make the paths only on process 0.
  if jax.process_index() == 0:
    # Avoid recursively create parent dir.
    tf.io.gfile.mkdir(path)
  return path


def _save_checkpoint_gda(train_state: train_states.TrainState,
                         checkpoint_dir: str, overwrite: bool,
                         step: int) -> None:
  """Saves a checkpoint using JAX GDA serialization mechanism.

  Note that all JAX processes must call _save_checkpoint_gda in sync because
  each process may only have a slice of the global data.

  Args:
    train_state: A partitioned train_state that is a Pytree of
      GlobalDeviceArray.
    checkpoint_dir: Full path to parent checkpoint_dir.
    overwrite: Whether to allow overwriting an existing target directory.
    step: Step to save checkpoint for.
  """
  if not overwrite:
    # Does not contain directory path, only dirname is returned.
    checkpoint_dirnames = tf.io.gfile.listdir(checkpoint_dir)
    # Delete tmp directories if any.
    if jax.process_index() == 0:
      tmp_checkpoint_dirnames = [
          x for x in checkpoint_dirnames if _is_tmp_checkpoint_asset(x)
      ]
      if tmp_checkpoint_dirnames:
        logging.warn('Found incompletely saved checkpoints %s; deleting them',
                     tmp_checkpoint_dirnames)
        for x in tmp_checkpoint_dirnames:
          tf.io.gfile.rmtree(os.path.join(checkpoint_dir, x))
    # Note we must barrier across all processes after the tmp directory delete.
    py_utils.sync_global_devices('Wait for checkpoint tmp dir deletions to '
                                 'finish.')

    sorted_dirnames = sorted(
        [x for x in checkpoint_dirnames if _is_checkpoint_asset(x)])
    if sorted_dirnames:
      latest_checkpoint_dirname = sorted_dirnames[-1]
      previous_step = get_step_from_checkpoint_asset(latest_checkpoint_dirname)
      if previous_step >= step:
        logging.warning(
            'A more recent checkpoint `%d` has already been saved compared '
            'to the current timestep `%d`. Skip saving a checkpoint.',
            previous_step, step)
        return

  checkpoint_step_dir = _make_checkpoint_step_dir(checkpoint_dir, step)
  checkpoint_step_tmp_dir = _make_tmp_checkpoint_dir(
      checkpoint_dir, step, sync_timestamp=True)
  logging.info('Saving to a tmp checkpoint dir %s', checkpoint_step_tmp_dir)

  nested_names = _extract_nested_prefix_names(train_state)
  flattened_nested_names, _ = jax.tree_util.tree_flatten(nested_names)

  if jax.process_index() == 0:
    # Create the tmp parent dir.
    tf.io.gfile.makedirs(checkpoint_step_tmp_dir)

  with futures.ThreadPoolExecutor() as executor:
    ckpt_paths = list(
        executor.map(
            _mkdir_path,
            flattened_nested_names,
            [checkpoint_step_tmp_dir] * len(flattened_nested_names),
        )
    )
  py_utils.sync_global_devices(
      'Wait for checkpoint tmp dir and subdirs '
      f'creation {checkpoint_step_tmp_dir} to finish.'
  )

  tspecs = jax.tree_map(
      array_serialization.get_tensorstore_spec, ckpt_paths
  )
  leaves, _ = jax.tree_util.tree_flatten(train_state)

  array_serialization.run_serialization(leaves, tspecs)

  # Note we must barrier across all processes before the directory rename.
  py_utils.sync_global_devices(
      'Wait for checkpoint chunk writes to '
      f'{checkpoint_step_tmp_dir} to finish.'
  )

  if jax.process_index() == 0:
    # Rename temporary checkpoint directory to its final location.
    logging.info(
        'Renaming %s to %s', checkpoint_step_tmp_dir, checkpoint_step_dir
    )
    tf.io.gfile.rename(checkpoint_step_tmp_dir, checkpoint_step_dir)

  logging.info(
      'Finished saving GDA checkpoint for step `%s` to `%s`.',
      step,
      checkpoint_step_dir,
  )


@functools.lru_cache()
def _cached_mesh_pspec_sharding(mesh, pspec):
  return sharding.NamedSharding(mesh, pspec)


def _restore_checkpoint_gda(
    train_state: Optional[train_states.TrainState],
    checkpoint_dir: str,
    global_mesh: Optional[jax.sharding.Mesh],
    state_specs: Optional[train_states.TrainState],
    step: Optional[int] = None) -> train_states.TrainState:
  """Restores a checkpoint using JAX GDA deserialization mechanism."""
  if not tf.io.gfile.exists(checkpoint_dir) or not tf.io.gfile.listdir(
      checkpoint_dir):
    if train_state is not None and step is None:
      logging.info(
          'GDA checkpoint restore did not find checkpoint_dir %s; '
          'Return train_state passed in', checkpoint_dir)
      return train_state
    raise FileNotFoundError(
        f'No checkpoint found for restore in {checkpoint_dir}')

  if step is None:
    checkpoint_dirnames = tf.io.gfile.listdir(checkpoint_dir)
    tmp_checkpoint_dirnames = [
        x for x in checkpoint_dirnames if _is_tmp_checkpoint_asset(x)
    ]
    if tmp_checkpoint_dirnames:
      logging.warn('Found incompletely saved checkpoints %s; skipping them',
                   tmp_checkpoint_dirnames)
    sorted_dirnames = sorted(
        [x for x in checkpoint_dirnames if _is_checkpoint_asset(x)])
    if not sorted_dirnames:
      raise FileNotFoundError(
          f'No checkpoint found for restore in {checkpoint_dir}')
    latest_checkpoint_dirname = sorted_dirnames[-1]
    step = get_step_from_checkpoint_asset(latest_checkpoint_dirname)
    checkpoint_step_dir = _make_checkpoint_step_dir(checkpoint_dir, step)
    logging.info('Found latest checkpoint: %s', checkpoint_step_dir)
  else:
    checkpoint_step_dir = _make_checkpoint_step_dir(checkpoint_dir, step)
    if not tf.io.gfile.exists(checkpoint_step_dir) or not tf.io.gfile.listdir(
        checkpoint_step_dir):
      raise FileNotFoundError(
          f'No checkpoint found for restore in {checkpoint_step_dir}')

  logging.info('GDA checkpoint restore started...')
  if train_state is not None:
    leaves, treedef = jax.tree_util.tree_flatten(train_state)
    partition_spec_leaves, _ = jax.tree_util.tree_flatten(state_specs)
    nested_names = _extract_nested_prefix_names(train_state)
    global_shapes = jax.tree_map(lambda x: x.shape, leaves)
  else:
    partition_spec_leaves, treedef = jax.tree_util.tree_flatten(state_specs)
    nested_names = _extract_nested_prefix_names(state_specs)
    global_shapes = None

  flattened_nested_names, _ = jax.tree_util.tree_flatten(nested_names)

  ckpt_paths = [
      os.path.join(checkpoint_step_dir, x).rstrip('/')
      for x in flattened_nested_names
  ]
  tspecs = jax.tree_map(
      array_serialization.get_tensorstore_spec, ckpt_paths
  )

  shardings = [
      _cached_mesh_pspec_sharding(global_mesh, s) for s in partition_spec_leaves
  ]
  train_state_gda = array_serialization.run_deserialization(
      shardings, tspecs, global_shapes=global_shapes
  )

  restored_train_state = jax.tree_util.tree_unflatten(treedef, train_state_gda)
  # Barrier across all processes to ensure all restore finish.
  py_utils.sync_global_devices(
      f'Wait for checkpoint restore from {checkpoint_step_dir} to finish.'
  )
  logging.info(
      'Successfully restored GDA checkpoint at %s!', checkpoint_step_dir
  )
  return restored_train_state
