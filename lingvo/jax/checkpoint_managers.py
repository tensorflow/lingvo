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
"""Module to manage checkpoint metadata and automatic checkpoint deletion."""

import datetime
import itertools
import os
from typing import Optional

from absl import logging
import jax
from lingvo.jax import checkpoint_pb2
from lingvo.jax import py_utils
import tensorflow.compat.v2 as tf

CheckpointType = checkpoint_pb2.CheckpointType

CHECKPOINT_PREFIX = 'checkpoint_'
CHECKPOINT_BASENAME = 'checkpoints.pb'


def to_timestamp(datetime_instance: datetime.datetime) -> int:
  """Converts a datetime instance into an int timestamp."""
  timedelta = datetime_instance - datetime.datetime.fromtimestamp(0)
  return int(round(timedelta.total_seconds()))


def from_timestamp(value: int) -> datetime.datetime:
  """Converts an int timestamp back into a datetime instance."""
  return datetime.timedelta(seconds=value) + datetime.datetime.fromtimestamp(0)


def read_checkpoint_file(filename: str) -> checkpoint_pb2.CheckpointHistory:
  """Reads a checkpoint file and returns the CheckpointHistory."""
  checkpoint_history = checkpoint_pb2.CheckpointHistory()
  with tf.io.gfile.GFile(filename, 'rb') as reader:
    checkpoint_history.ParseFromString(reader.read())
  return checkpoint_history


def extract_latest_checkpoint_id(
    checkpoint_history: checkpoint_pb2.CheckpointHistory) -> int:
  """Given a CheckpointHistory, returns the latest checkpoint id."""
  # Checkpoints are added in chronological order. Retrieve the last one.
  checkpoint_metadata = checkpoint_history.checkpoints[-1]
  return checkpoint_metadata.global_step_id


class CheckpointManager:
  """Manages multiple checkpoints handling automatic deletion.

  This currently follows a pretty similar policy when compared to the vanilla
  TensorFlow CheckpointManager class:
  http://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/checkpoint_management.py

  There are three main parameters to control how often to save and delete
  checkpoints:
  - save_interval_steps: This ensures that a checkpoint is saved every N steps.
  - max_to_keep: The maximum number of checkpoints to keep (unless preserved by
      keep_interval_timedelta).
  - keep_interval_timedelta: This ensures that a checkpoint is kept at the
      the provided minimum time period expressed as a datetime.timedelta
      instance. This is in addition to keeping the `max_to_keep` most recent
      checkpoint files.

  The CheckpointManager instance can be created on several JAX processes.
  However, the handling and update of the checkpoint file and checkpoint assets
  will solely be performed on `jax.process_index() == 0`.
  """

  def __init__(self,
               *,
               config_name: str,
               root_dir: str,
               checkpoint_type: CheckpointType,
               save_interval_steps=int,
               max_to_keep: Optional[int],
               keep_interval_timedelta: Optional[datetime.timedelta] = None,
               checkpoint_basename: str = CHECKPOINT_BASENAME,
               todelete_subdir: Optional[str] = None):
    """Constructor.

    Only a single CheckpointManager for a given training program is expected
    to be instantiated.

    When a new checkpoint manager is instantiated pointing to a `root_dir`
    with previously generated checkpoints, a few of them may be deleted by
    ensuring that the time period between `kept` checkpoints is larger than
    `keep_interval_timedelta` and by ensuring that there are less than
    `max_to_keep` checkpoint among the non-kept ones.

    Args:
      config_name: The name of the registered model configuration.
      root_dir: The root directory, where model checkpoints are saved.
      checkpoint_type: The type of checkpoint to use.
      save_interval_steps: This ensures that a checkpoint is saved every N
        steps.
      max_to_keep: The number of checkpoints to keep. Unless preserved by
        `keep_interval_timedelta`, checkpoints will be deleted from the active
        set, oldest first, until only `max_to_keep` checkpoints remain. If set
        to `None`, no checkpoints are deleted and everything stays in the active
        set. Note that `max_to_keep=None` will keep all checkpoints metadata in
        memory and in the checkpoints.pb file on disk.
      keep_interval_timedelta: Upon removal from the active set, a checkpoint
        will be preserved if it has been at least `keep_interval_timedelta`
        since the last preserved checkpoint. The default setting of `None` does
        not preserve any checkpoints in this way.
      checkpoint_basename: The basename of the checkpoint metadata file.
      todelete_subdir: If set, checkpoints to be deleted will be only renamed
        into a subdirectory with the provided string. Otherwise, they will be
        directly deleted from the file system. Useful if checkpoint deletion is
        time consuming. By default, delete the checkpoint assets.
    """
    self._config_name: str = config_name
    self._root_dir: str = root_dir
    self._checkpoint_type: CheckpointType = checkpoint_type

    self._save_interval_steps: int = save_interval_steps
    self._max_to_keep: Optional[int] = max_to_keep
    self._keep_interval_timedelta: Optional[datetime.timedelta] = (
        keep_interval_timedelta)
    self._checkpoint_basename: str = checkpoint_basename
    self._todelete_subdir: Optional[str] = todelete_subdir

    self._init_checkpoint_history()

  @property
  def use_multi_host_flax(self) -> bool:
    """Indicates whether the checkpoint type uses multi-host Flax."""
    return (self._checkpoint_type ==
            checkpoint_pb2.CheckpointType.CHECKPOINT_MULTI_HOST_FLAX)

  @property
  def checkpoint_filename(self) -> str:
    """Full checkpoints' filename."""
    if self.use_multi_host_flax:
      return os.path.join(self._root_dir, f'{jax.process_index():03d}',
                          self._checkpoint_basename)
    return os.path.join(self._root_dir, self._checkpoint_basename)

  def should_save(self, global_step_id: int) -> bool:
    """Indicates whether there is a need to save a checkpoint."""
    return (self._last_saved_checkpoint_step is None or
            global_step_id - self._last_saved_checkpoint_step >=
            self._save_interval_steps)

  def save_metadata(self, global_step_id: int) -> None:
    """Adds a new checkpoint to the manager.

    This function also deletes the old checkpoints if needed and generates a new
    checkpoints file.

    Args:
      global_step_id: The global step identifier of the checkpoint to be added.

    Raises:
      ValueError: When save_metadata() was not supposed to be called yet at the
        current time. This can be verified by calling should_save() first.
    """
    if not self.should_save(global_step_id):
      raise ValueError(
          f'Not expecting to call save_metadata() at step `{global_step_id}`'
          f'(last saved step: `{self._last_saved_checkpoint_step}` --'
          f' save interval steps: `{self._save_interval_steps}`).')

    current_time = datetime.datetime.utcnow()
    self._last_saved_checkpoint_step = global_step_id

    # Use datetime.datetime directly rather than timestamp.GetCurrentTime()
    # to simplify mocking datetime.datetime function calls in unit tests.
    timestamp = to_timestamp(current_time)
    self._checkpoint_history.checkpoints.add(
        global_step_id=global_step_id, timestamp_sec=timestamp)

    # First, save the checkpoints file before deleting anything.
    # This is useful if e.g. a pre-emption happens.
    self._save_checkpoint_file(global_step_id, 'pre')
    # Clean up old checkpoints if needed.
    self._sweep(global_step_id)
    # Update again the checkpoints file with the new version.
    self._save_checkpoint_file(global_step_id, 'post')

  def _create_checkpoint_history(self) -> checkpoint_pb2.CheckpointHistory:
    """Creates a CheckpointHistory instance with default fields set."""
    return checkpoint_pb2.CheckpointHistory(
        config_name=self._config_name,
        root_directory=self._root_dir,
        checkpoint_type=self._checkpoint_type)

  def _init_checkpoint_history(self) -> None:
    """Initializes the checkpoint history and sets related class attributes."""
    self._last_saved_checkpoint_step: int = None
    self._last_kept_checkpoint_datetime: Optional[datetime.datetime] = None

    if not tf.io.gfile.exists(self.checkpoint_filename):
      self._checkpoint_history = self._create_checkpoint_history()
      return

    # Read the previous checkpoints file and performs a sanity check.
    self._checkpoint_history = self._read_checkpoint_file()

    last_saved_timestamp = (
        self._checkpoint_history.checkpoints[-1].timestamp_sec)
    current_datetime = datetime.datetime.utcnow()
    if current_datetime < from_timestamp(last_saved_timestamp):
      # Time seems to have reversed itself.
      logging.warning(
          'datetime.datetime.utcnow() returned a value `%s` behind the last '
          'saved checkpoint timestamp.',
          from_timestamp(last_saved_timestamp) - current_datetime)

    # Add few of the checkpoints to the `kept` list.
    kept_checkpoints = []
    if self._keep_interval_timedelta is None:
      maybe_delete_checkpoints = list(self._checkpoint_history.checkpoints)
    else:
      maybe_delete_checkpoints = []
      oldest_kept_timestamp = None
      for checkpoint in self._checkpoint_history.checkpoints:
        if (oldest_kept_timestamp is None or
            ((from_timestamp(oldest_kept_timestamp) - from_timestamp(
                checkpoint.timestamp_sec)) >= self._keep_interval_timedelta)):
          oldest_kept_timestamp = checkpoint.timestamp_sec
          kept_checkpoints.append(checkpoint)
          if self._last_kept_checkpoint_datetime is None:
            self._last_kept_checkpoint_datetime = (
                from_timestamp(checkpoint.timestamp_sec))
        else:
          maybe_delete_checkpoints.append(checkpoint)

    # Only keep at most `max_to_keep` non-kept checkpoints. Delete the old ones.
    if not self.use_multi_host_flax:
      py_utils.sync_global_devices(
          'checkpoint_manager:begin_delete_checkpoints:'
          f'init_{self._checkpoint_history.checkpoints[-1].global_step_id}')
    for i, checkpoint in enumerate(reversed(maybe_delete_checkpoints)):
      if self._max_to_keep is None or i < self._max_to_keep:
        kept_checkpoints.append(checkpoint)
      else:
        self._delete_checkpoint(checkpoint)
    if not self.use_multi_host_flax:
      py_utils.sync_global_devices(
          'checkpoint_manager:end_delete_checkpoints:'
          f'init_{self._checkpoint_history.checkpoints[-1].global_step_id}')

    # Finally create a new CheckpointHistory and save a new checkpoint file.
    kept_checkpoints = sorted(
        kept_checkpoints, key=lambda c: from_timestamp(c.timestamp_sec))
    latest_global_step = kept_checkpoints[-1].global_step_id
    self._last_saved_checkpoint_step = latest_global_step
    self._checkpoint_history = self._create_checkpoint_history()
    for c in kept_checkpoints:
      self._checkpoint_history.checkpoints.add().CopyFrom(c)

    self._save_checkpoint_file(latest_global_step)

  def _delete_pattern_if_exists(self, root_dir: str, filepath: str) -> None:
    """Deletes everything under `filepath`."""
    # Note: This method may be called by different JAX processes. The
    # concurrency logic is handled in _delete_checkpoint() below.
    src = os.path.join(root_dir, filepath)
    logging.info('Deleting files with filepath: `%s`', src)
    if tf.io.gfile.exists(src):
      if self._todelete_subdir:
        rename_dir = os.path.join(root_dir, self._todelete_subdir)
        if not tf.io.gfile.exists(rename_dir):
          tf.io.gfile.mkdir(rename_dir)
        dst = os.path.join(rename_dir, filepath)
        # TODO(lingvo-team): Check if dst already exists?
        tf.io.gfile.rename(src, dst)
      else:
        tf.io.gfile.rmtree(src)

  def _delete_checkpoint(self,
                         checkpoint: checkpoint_pb2.CheckpointMetadata) -> None:
    """Deletes the checkpoint files for a given checkpoint."""
    logging.info('Deleting checkpoint: %s %s', checkpoint.timestamp_sec,
                 self._root_dir)
    if (self._checkpoint_history.checkpoint_type ==
        CheckpointType.CHECKPOINT_FLAX):
      if jax.process_index() != 0:
        return
      self._delete_pattern_if_exists(
          self._root_dir, f'{CHECKPOINT_PREFIX}{checkpoint.global_step_id}')
    elif (self._checkpoint_history.checkpoint_type ==
          CheckpointType.CHECKPOINT_MULTI_HOST_FLAX):
      root_dir = os.path.join(self._root_dir, f'{jax.process_index():03d}')
      self._delete_pattern_if_exists(
          root_dir, f'{CHECKPOINT_PREFIX}{checkpoint.global_step_id}')
    elif self._checkpoint_history.checkpoint_type in {
        CheckpointType.CHECKPOINT_PERSISTENCE,
        CheckpointType.CHECKPOINT_GDA,
    }:
      if jax.process_index() != 0:
        return
      self._delete_pattern_if_exists(
          self._root_dir, f'{CHECKPOINT_PREFIX}{checkpoint.global_step_id:08d}')

  def _sweep(self, global_step_id: int) -> None:
    """Deletes or preserves managed checkpoints."""
    if not self._max_to_keep:
      return

    kept_checkpoints = []
    maybe_delete_checkpoints = []
    for checkpoint in self._checkpoint_history.checkpoints:
      if (self._last_kept_checkpoint_datetime is not None and (from_timestamp(
          checkpoint.timestamp_sec) <= self._last_kept_checkpoint_datetime)):
        kept_checkpoints.append(checkpoint)
      else:
        maybe_delete_checkpoints.append(checkpoint)

    if not self.use_multi_host_flax:
      py_utils.sync_global_devices(
          'checkpoint_manager:begin_delete_checkpoints:'
          f'step_{global_step_id}')
    while len(maybe_delete_checkpoints) > self._max_to_keep:
      checkpoint = maybe_delete_checkpoints.pop(0)
      if (self._keep_interval_timedelta and
          (not kept_checkpoints or
           self._last_kept_checkpoint_datetime is None or
           ((from_timestamp(checkpoint.timestamp_sec) -
             self._last_kept_checkpoint_datetime) >=
            self._keep_interval_timedelta))):
        kept_checkpoints.append(checkpoint)
        self._last_kept_checkpoint_datetime = from_timestamp(
            checkpoint.timestamp_sec)
        continue
      self._delete_checkpoint(checkpoint)
    if not self.use_multi_host_flax:
      py_utils.sync_global_devices('checkpoint_manager:end_delete_checkpoints:'
                                   f'step_{global_step_id}')

    self._checkpoint_history = self._create_checkpoint_history()
    for c in itertools.chain(kept_checkpoints, maybe_delete_checkpoints):
      logging.info('Keeping checkpoint: (step: %d, timestamp: %s)',
                   c.global_step_id, c.timestamp_sec)
      self._checkpoint_history.checkpoints.add().CopyFrom(c)

  def _save_checkpoint_file(self, global_step_id: int, key: str = '') -> None:
    """Saves the checkpoint file with latest checkpoint metadata.

    Note: This method overrides previous version of the checkpoint file.

    Args:
      global_step_id: The current global step.
      key: A key to add to the synchronization string.
    """
    py_utils.sync_global_devices(
        f'checkpoint_manager:begin_save_checkpoint_file:{key}_{global_step_id}')
    if self.use_multi_host_flax or jax.process_index() == 0:
      with tf.io.gfile.GFile(self.checkpoint_filename, 'wb') as writer:
        writer.write(self._checkpoint_history.SerializeToString())
    py_utils.sync_global_devices(
        f'checkpoint_manager:end_save_checkpoint_file:{key}_{global_step_id}')

  def _read_checkpoint_file(self) -> checkpoint_pb2.CheckpointHistory:
    """Restores the checkpoint file with latest checkpoint metadata.

    Returns:
      The parsed CheckpointHistory proto.
    """
    return read_checkpoint_file(self.checkpoint_filename)
