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
"""Checkpointing-related utilities to handle TrainState instances."""

import enum
from typing import Optional

from absl import logging
from flax import jax_utils
from flax.training import checkpoints
import jax
from lingvo.jax import train_states


@enum.unique
class CheckpointType(str, enum.Enum):
  """Checkpointing types wrt. the underlying implementation used."""
  FLAX = 'flax'
  PERSISTENCE = 'persistence'


def SaveCheckpoint(train_state: train_states.TrainState,
                   checkpoint_dir: str,
                   overwrite: bool = False,
                   unreplicate: bool = True,
                   checkpoint_type: CheckpointType = CheckpointType.FLAX,
                   state_specs: Optional[train_states.TrainState] = None,
                   max_checkpoints: int = 10) -> None:
  """Saves a checkpoint into the provided base directory.

  This is typically called on a replicated TrainState instance.

  Args:
    train_state: The TrainState instance to save.
    checkpoint_dir: The base directory from where to retrieve checkpoints.
    overwrite: Whether to overwrite existing checkpoints files if a checkpoint
      at the current or a later step already exists.
    unreplicate: Whether to unreplicate variables (Optional). If using SPMD
      sharding, then this should be set to False.
    checkpoint_type: The checkpoint type (implementation) to save. Currently,
      it must be `CheckpointType.FLAX`.
    state_specs: Currently unused.
    max_checkpoints: The number of past checkpoint files to keep.

  Raises:
    ValueError: If the global step has an unexpected shape, if `state_specs`
    is not specified for persistence-based checkpointing or if
    `checkpoint_type` is invalid.
  """
  if train_state.step.ndim == 0:
    step = jax.device_get(train_state.step)
  elif train_state.step.ndim == 1:
    step = jax.device_get(train_state.step[0])
  else:
    raise ValueError(
        f'Expecting a replicated 1D global step (got `{train_state.step.ndim}`).'
    )

  if checkpoint_type == CheckpointType.FLAX:
    _SaveCheckpointFlax(train_state, checkpoint_dir, overwrite, unreplicate,
                        max_checkpoints, step)
  else:
    raise ValueError(f'Unexpected checkpoint_type `{checkpoint_type}`.')


def LatestCheckpoint(checkpoint_dir: str) -> Optional[str]:
  """Gets the path to the latest checkpoint.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    Path to latest checkpoint or None if there is no checkpoint.
  """
  return checkpoints.latest_checkpoint(checkpoint_dir)


def RestoreCheckpoint(train_state: train_states.TrainState,
                      checkpoint_dir: str,
                      checkpoint_type: CheckpointType = CheckpointType.FLAX,
                      state_specs: Optional[train_states.TrainState] = None,
                      step: Optional[int] = None) -> train_states.TrainState:
  """Restores a checkpoint from the provided base directory.

  This is typically called on an unreplicated TrainState instance.

  Args:
    train_state: The TrainState instance to restore.
    checkpoint_dir: The base directory from where to retrieve checkpoints.
    checkpoint_type: The checkpoint type (implementation) to restore. Currently,
      it must be `CheckpointType.FLAX`.
    state_specs: Currently unused.
    step: Step number to load a checkpoint from or None to load the latest.

  Returns:
    A restored `TrainState` instance.

  Raises:
    ValueError: When a mismatch between the current checkpoint structure and
    the saved checkpoint one is detected.
  """
  del state_specs  # Unused.

  if train_state.step.ndim != 0:
    raise ValueError('Expecting an unreplicated scalar global step (got '
                     f'`{train_state.step.ndim}`).')

  if checkpoint_type == CheckpointType.FLAX:
    return _RestoreCheckpointFlax(train_state, checkpoint_dir, step)
  else:
    raise ValueError(f'Unexpected checkpoint_type `{checkpoint_type}`.')


def _SaveCheckpointFlax(train_state: train_states.TrainState,
                        checkpoint_dir: str, overwrite: bool, unreplicate: bool,
                        max_checkpoints: int, step: int) -> None:
  """Saves a checkpoint using Flax serialization mechanism."""
  if not overwrite:
    previous_filename = LatestCheckpoint(checkpoint_dir)
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
  def _MaybeUnreplicate(data):
    if unreplicate:
      return jax.device_get(jax_utils.unreplicate(data))
    else:
      return jax.device_get(data)

  # Extract/flatten data structure to store to disk. Flax requires a flattened
  # data structure to be passed to the checkpointer.
  flattened_state, pytree_state = jax.tree_flatten(
      _MaybeUnreplicate(train_state))
  checkpoint_target = {
      'flattened_state': flattened_state,
      # Saves a serialized version of the pytree structure to detect potential
      # mismatch caused by different versions of saver/restorer.
      'str_pytree_state': str(pytree_state),
  }
  checkpoints.save_checkpoint(
      checkpoint_dir,
      checkpoint_target,
      step,
      keep=max_checkpoints,
      overwrite=overwrite)


def _RestoreCheckpointFlax(
    train_state: train_states.TrainState,
    checkpoint_dir: str,
    step: Optional[int] = None) -> train_states.TrainState:
  """Restores a checkpoint using Flax serialization mechanism."""
  # Input the same data structure as in SaveCheckpoint().
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
