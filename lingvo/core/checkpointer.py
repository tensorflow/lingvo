# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
import six


class Checkpointer(object):
  """Checkpointing utility class.

  Needs to be created within a graph context.
  """

  def __init__(self, train_dir, model, train_params=None, save_only=False):
    """Initialize Checkpointer.

    Args:
     train_dir: Training directory for saving checkpoints.
     model: A BaseModel instance or None.
     train_params: If specified, use these training params instead of those in
       the `model`.
     save_only: This checkpointer is only intended for saving checkpoints.
    """
    self._train_dir = train_dir
    self._save_only = save_only

    self._save_path = os.path.join(self._train_dir, 'ckpt')

    if train_params:
      self._train_params = train_params
      self._model = None
    else:
      assert model
      self._train_params = model.params.train
      self._model = model

    if not self._save_only:
      self._params = model.params
      self._model_tasks = model.tasks
      self._model = model

    self._next_checkpoint_seconds = 0
    self._save_interval_seconds = self._train_params.save_interval_seconds
    self._saver = self._GetSaver()

    self._uninitialized_vars = tf.report_uninitialized_variables(
        tf.global_variables())

  def _GetSaver(self):
    """Returns a saver."""
    do_eval = cluster_factory.Current().do_eval
    if not self._save_only and self._model.ema and do_eval:
      tf.logging.info('Using EMA for evaluation.')
      return tf.train.Saver(
          self._model.ema.variables_to_restore(self._model.variables_for_ema))
    return tf.train.Saver(
        sharded=True,
        max_to_keep=self._train_params.save_max_to_keep,
        keep_checkpoint_every_n_hours=(
            self._train_params.save_keep_checkpoint_every_n_hours),
        pad_step_number=True,  # %08d
        write_version=tf.train.SaverDef.V2)

  def RestoreFromPath(self, sess, checkpoint_path):
    """Load the checkpoint from specified path."""
    assert not self._save_only
    tf.logging.info('Load from checkpoint %s.', checkpoint_path)
    self._saver.restore(sess, checkpoint_path)
    tf.logging.info('Load checkpoint done.')
    # Successfully restored from checkpoint.
    uninitialized_var_names = self._GetUninitializedVarNames(sess)
    assert not uninitialized_var_names, uninitialized_var_names

  def MaybeSave(self, sess, gsteps):
    """If it's time to save, save the checkpoint.

    Args:
      sess: tf.Session.
      gsteps: Current global step.
    """
    now = time.time()
    if now >= self._next_checkpoint_seconds:
      self.Save(sess, gsteps)
      self._next_checkpoint_seconds = now + self._save_interval_seconds

  def Save(self, sess, gsteps):
    """Save the checkpoint.

    Args:
      sess: tf.Session.
      gsteps: Current global step.
    """
    tf.logging.info('Save checkpoint')
    path = self._saver.save(sess, self._save_path, gsteps)
    tf.logging.info('Save checkpoint done: %s', path)

  def _RestoreFromLatestCheckpoint(self, sess):
    """Restore the latest checkpoint and return True, else return False."""
    assert not self._save_only
    path = tf.train.latest_checkpoint(self._train_dir)

    if path is None:
      return False
    # First recover the checkpoint state in the directory.
    #
    # NOTE: latest_checkpoint() already calls this but to avoid duplicating
    # v1 vs. v2 behavior here, we just query the state again.
    ckpt_state = tf.train.get_checkpoint_state(self._train_dir)
    self._saver.recover_last_checkpoints(ckpt_state.all_model_checkpoint_paths)

    # Now restore the checkpoint.
    self.RestoreFromPath(sess, path)
    return True

  def _GetUninitializedVarNames(self, sess):
    uninitialized_var_names = sorted(list(sess.run(self._uninitialized_vars)))
    # uninitialized_var_names is a list of strings without ":0" suffix.
    # tf.report_uninitialized_variables returns binary strings.
    assert all(isinstance(s, six.binary_type) for s in uninitialized_var_names)
    return uninitialized_var_names

  def Restore(self, sess, force_reinitialize=False):
    """Restore from latest checkpoint if available, or initialize."""
    # Try and restore from the latest checkpoint.
    if self._RestoreFromLatestCheckpoint(sess):
      # Successfully restored from checkpoint.
      uninitialized_var_names = self._GetUninitializedVarNames(sess)
      assert not uninitialized_var_names, uninitialized_var_names
      return

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
    sess.run(tf.global_variables_initializer())
    tf.logging.info('Initialized all vars.')

    # Restore specific variables based on init_from_checkpoint_rules.
    for task in self._model.tasks:
      tp = task.params.train
      if tp.init_from_checkpoint_rules:
        tf.logging.info('OverrideVarsFromCheckpoints %s',
                             tp.init_from_checkpoint_rules)
        py_utils.OverrideVarsFromCheckpoints(sess, tf.global_variables(),
                                             tp.init_from_checkpoint_rules)

    if self._params.train.init_from_checkpoint_rules:
      tp = self._params.train
      tf.logging.info('OverrideVarsFromCheckpoints %s',
                           tp.init_from_checkpoint_rules)
      py_utils.OverrideVarsFromCheckpoints(sess, tf.global_variables(),
                                           tp.init_from_checkpoint_rules)

  def RestoreIfNeeded(self, sess):
    """If vars are not initialized, restore from checkpoint."""
    assert not self._save_only
    uninitialized_var_names = self._GetUninitializedVarNames(sess)
    if not uninitialized_var_names:
      # All variables are already initialized.
      return

    self.Restore(sess)

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
