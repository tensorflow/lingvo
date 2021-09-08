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
"""Checkpointing utilities for save/restore."""

import os
import time

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
import six


class Checkpointer:
  """Checkpointing utility class.

  Needs to be created within a graph context.
  """

  def __init__(self,
               train_dir,
               model,
               init_op=None,
               train_params=None,
               save_only=False):
    """Initialize Checkpointer.

    Args:
     train_dir: Training directory for saving checkpoints.
     model: A BaseModel instance or None.
     init_op: The initialize variables op. If unset, it will call
       tf.global_variables_initializer().
     train_params: If specified, use these training params instead of those in
       the `model`.
     save_only: This checkpointer is only intended for saving checkpoints.
    """
    self._train_dir = train_dir
    self._save_only = save_only
    if init_op:
      self._init_op = init_op
    else:
      self._init_op = tf.global_variables_initializer()

    self._save_path = os.path.join(self._train_dir, 'ckpt')

    if train_params:
      self._train_params = train_params
      self._model = None
    else:
      assert model
      self._train_params = model.params.train
      self._model = model

    if self._save_only:
      self._params = None
    else:
      self._params = model.params
      self._model_tasks = model.tasks
      self._model = model

    self._next_checkpoint_seconds = 0
    self._save_interval_seconds = self._train_params.save_interval_seconds
    self._save_interval_steps = self._train_params.save_interval_steps
    self._prev_ckpt_step = None
    self._saver = self._GetSaver()

    self._uninitialized_vars = tf.report_uninitialized_variables(
        tf.global_variables())

    # TODO(b/160786085): Move this logic into Overriding vars logic itself,
    # which requires refactoring things out of py_utils to avoid circular deps.
    def _ResolveCkptPath(ckpt_rules):
      return {GetSpecificCheckpoint(k): v for k, v in ckpt_rules.items()}

    self._restore_fns = []

    # Add graph nodes to restore specific variables based on
    # init_from_checkpoint_rules.
    # TODO(b/159267006): Move this back to Restore().
    if self._model:
      for task in self._model.tasks:
        tp = task.params.train
        if tp.init_from_checkpoint_rules:
          rules = _ResolveCkptPath(tp.init_from_checkpoint_rules)
          tf.logging.info('OverrideVarsFromCheckpoints %s', rules)
          fn = py_utils.OverrideVarsFromCheckpoints(tf.global_variables(),
                                                    rules)
          self._restore_fns.append(fn)

    if self._params and self._params.train.init_from_checkpoint_rules:
      tp = self._params.train
      rules = _ResolveCkptPath(tp.init_from_checkpoint_rules)
      tf.logging.info('OverrideVarsFromCheckpoints %s', rules)
      fn = py_utils.OverrideVarsFromCheckpoints(tf.global_variables(), rules)
      self._restore_fns.append(fn)

  def _GetSaver(self):
    """Returns a saver."""
    do_eval = cluster_factory.Current().do_eval
    if not self._save_only and self._model.ema and do_eval:
      tf.logging.info('Using EMA for evaluation.')
      saver = tf.train.Saver(
          self._model.ema.variables_to_restore(self._model.variables_for_ema))
    else:
      saver = tf.train.Saver(
          sharded=True,
          max_to_keep=self._train_params.save_max_to_keep,
          keep_checkpoint_every_n_hours=(
              self._train_params.save_keep_checkpoint_every_n_hours),
          pad_step_number=True,  # %08d
          write_version=tf.train.SaverDef.V2)
    previous_checkpoint_state = tf.train.get_checkpoint_state(self._train_dir)
    if previous_checkpoint_state is not None:
      saver.recover_last_checkpoints(
          previous_checkpoint_state.all_model_checkpoint_paths)
    return saver

  @property
  def async_checkpointing(self):
    return self._train_params.async_checkpointing

  def RestoreFromPath(self, sess, checkpoint_path):
    """Load the checkpoint from specified path."""
    assert not self._save_only
    tf.logging.info('Load from checkpoint %s.', checkpoint_path)
    self._saver.restore(sess, checkpoint_path)
    tf.logging.info('Load checkpoint done.')
    # Successfully restored from checkpoint.
    uninitialized_var_names = self._GetUninitializedVarNames(sess)
    assert not uninitialized_var_names, uninitialized_var_names

  def ShouldSave(self, gsteps):
    """Returns True if a checkpoint should be saved."""
    if self._prev_ckpt_step is None:
      # Always save the first checkpoint.
      return True
    elif self._prev_ckpt_step == gsteps:
      # Don't rewrite the same checkpoint.
      return False
    elif self._save_interval_steps is not None:
      # Use save_interval_steps if it is specified by the user.
      return gsteps - self._prev_ckpt_step >= self._save_interval_steps
    else:
      # Use save_interval_seconds otherwise.
      return time.time() >= self._next_checkpoint_seconds

  def MaybeSave(self, sess, gsteps):
    """If it's time to save, save the checkpoint.

    Args:
      sess: tf.Session.
      gsteps: Current global step.
    Returns:
      Whether a checkpoint was saved.
    """
    if self.ShouldSave(gsteps):
      self.Save(sess, gsteps)
      return True
    return False

  def Save(self, sess, gsteps):
    """Save the checkpoint.

    Args:
      sess: tf.Session.
      gsteps: Current global step.
    """
    tf.logging.info('Save checkpoint')
    path = self._saver.save(sess, self._save_path, gsteps)
    tf.logging.info('Save checkpoint done: %s', path)
    self._prev_ckpt_step = gsteps
    self._UpdateNextSaveTime()

  def _UpdateNextSaveTime(self):
    now = time.time()
    self._next_checkpoint_seconds = now + self._save_interval_seconds

  def _RestoreFromLatestCheckpoint(self, sess):
    assert not self._save_only
    path = tf.train.latest_checkpoint(self._train_dir)
    if path:
      self.RestoreFromPath(sess, path)
      self._prev_ckpt_step = int(path.split('-')[-1])  # path=.../ckpt-step
      return path
    return None

  def _GetUninitializedVarNames(self, sess):
    uninitialized_var_names = sorted(list(sess.run(self._uninitialized_vars)))
    # uninitialized_var_names is a list of strings without ":0" suffix.
    # tf.report_uninitialized_variables returns binary strings.
    assert all(isinstance(s, bytes) for s in uninitialized_var_names)
    return uninitialized_var_names

  def Restore(self, sess, force_reinitialize=False):
    """Restore from latest checkpoint if available, or initialize."""
    # Try and restore from the latest checkpoint.
    path = self._RestoreFromLatestCheckpoint(sess)
    if path:
      # Successfully restored from checkpoint.
      uninitialized_var_names = self._GetUninitializedVarNames(sess)
      assert not uninitialized_var_names, uninitialized_var_names
      return path

    # Otherwise we need to initialize.
    uninitialized_var_names = self._GetUninitializedVarNames(sess)
    tf.logging.info('Uninitialized var list: %s', uninitialized_var_names)
    if not force_reinitialize:
      # There should only be uninitialized variables if all variables are
      # uninitialized - with the exception of global_step due to
      # RestoreGlobalStepIfNeeded in the _LoopEnqueue of TrainerTpu.
      all_var_names = [
          six.ensure_binary(v.name[:-2]) for v in tf.global_variables()
      ]
      already_initialized_vars = (
          set(all_var_names) - set(uninitialized_var_names))
      already_initialized_vars.discard(b'global_step')
      assert not already_initialized_vars, ('Already initialized vars: %s' %
                                            sorted(already_initialized_vars))

    # At this point all variables are uninitialized, so it is safe to run a
    # global initializer.
    sess.run(self._init_op)
    tf.logging.info('Initialized all vars.')

    if self._restore_fns:
      for fn in self._restore_fns:
        fn(sess)
      tf.logging.info('Restored vars using checkpoint rules.')
    return None

  def RestoreIfNeeded(self, sess):
    """If vars are not initialized, restore from checkpoint."""
    assert not self._save_only
    uninitialized_var_names = self._GetUninitializedVarNames(sess)
    if not uninitialized_var_names:
      # All variables are already initialized.
      return None

    return self.Restore(sess)

  def RestoreGlobalStepIfNeeded(self, sess):
    """If global step is not initialized, load it from the checkpoint.

    Args:
      sess: tf.Session.
    """
    assert not self._save_only
    uninitialized_vars = self._GetUninitializedVarNames(sess)
    if six.ensure_binary('global_step') not in uninitialized_vars:
      return

    with sess.graph.as_default():
      gstep = py_utils.GetGlobalStep()

      path = tf.train.latest_checkpoint(self._train_dir)
      if path:
        reader = tf.train.NewCheckpointReader(path)
        value = reader.get_tensor('global_step')
        tf.logging.info('Restoring global step: %s', value)
        sess.run(gstep.assign(value))
      else:
        tf.logging.info('Initializing global step')
        sess.run(gstep.initializer)


def GetSpecificCheckpoint(load_checkpoint_from):
  """Returns a specific checkpoint given `load_checkpoint_from`.

  When `load_checkpoint_from` is a checkpoint (determined by the existence of
  `load_checkpoint_from` + '.index'), validate the path and return it.

  Otherwise, if `load_checkpoint_from` is a directory, we find the latest
  checkpoint in the directory and return that checkpoint.

  Args:
    load_checkpoint_from: If not None, specifies the directory or specific
      checkpoint to load.  If a directory, the latest checkpoint in the
      directory will be used.

  Raises:
    ValueError: if `load_checkpoint_from` is not a checkpoint or a directory
      containing checkpoints.
  """
  if not load_checkpoint_from:
    return None

  # Check validity of eval path by looking for the index file.
  if tf.io.gfile.exists(load_checkpoint_from + '.index'):
    return load_checkpoint_from

  # If load_checkpoint_from is a directory, return the latest
  # checkpoint in the directory.
  if tf.io.gfile.isdir(load_checkpoint_from):
    latest_checkpoint = tf.train.latest_checkpoint(load_checkpoint_from)
    if latest_checkpoint:
      return latest_checkpoint

  # Fail if we see an unexpected load_checkpoint_from.
  # This might happen if load_checkpoint_from refers to a checkpoint
  # but the index file cannot be found.
  raise ValueError('Invalid load_checkpoint_from: %s' % load_checkpoint_from)
