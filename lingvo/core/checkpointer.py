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
# pylint: disable=line-too-long
"""Checkpointing utilities for save/restore."""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import lingvo.compat as tf
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
     train_params: If specified, use these training params instead of those
       in the `model`.
     save_only: This checkpointer is only intended for saving checkpoints.
    """
    self._train_dir = train_dir
    self._save_only = save_only

    self._vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self._uninitialized_vars = tf.report_uninitialized_variables(self._vars)
    self._initialize_vars = tf.global_variables_initializer()

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

  def _GetSaver(self):
    """Returns a saver."""
    if not self._save_only and self._model.ema and self._params.is_eval:
      tf.logging.info('Using EMA for evaluation.')
      return tf.train.Saver(self._model.ema.variables_to_restore())
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

  def _Restore(self, sess):
    assert not self._save_only
    path = tf.train.latest_checkpoint(self._train_dir)
    if path:
      self.RestoreFromPath(sess, path)
    return path

  def RestoreIfNeeded(self, sess):
    """If vars are not initialized, restore frome checkpoint.

    Args:
      sess: tf.Session.
    """
    assert not self._save_only
    uninitialized_var_names = list(sess.run(self._uninitialized_vars))
    if not uninitialized_var_names:
      return

    tf.logging.info('Uninitialized var list: %s ', uninitialized_var_names)
    if self._Restore(sess):
      return

    if (not any(task.params.train.init_from_checkpoint_rules
                for task in self._model_tasks) and
        not self._params.train.init_from_checkpoint_rules):
      tf.logging.info('Initialize ALL variables: %s', uninitialized_var_names)
      sess.run([self._initialize_vars])
      tf.logging.info('Initialize variables done.')
      return

    # There was a race in local run. Another thread will get unblocked once
    # _initialize_all is called. OverrideVarsFromCheckpoints
    # might not happen at the right time.
    for task in self._model.tasks:
      tp = task.params.train
      if tp.init_from_checkpoint_rules:
        tf.logging.info('OverrideVarsFromCheckpoints %s',
                        tp.init_from_checkpoint_rules)
        py_utils.OverrideVarsFromCheckpoints(sess, self._vars,
                                             tp.init_from_checkpoint_rules)

    if self._params.train.init_from_checkpoint_rules:
      tp = self._params.train
      tf.logging.info('OverrideVarsFromCheckpoints %s',
                      tp.init_from_checkpoint_rules)
      py_utils.OverrideVarsFromCheckpoints(sess, self._vars,
                                           tp.init_from_checkpoint_rules)

    uninitialized_var_names = list(sess.run(self._uninitialized_vars))
    if not uninitialized_var_names:
      return

    # uninitialized_var_names is a list of strings without ":0" suffix.
    # tf.report_uninitialized_variables returns binary strings.
    assert all(isinstance(s, six.binary_type) for s in uninitialized_var_names)

    # Need to retrieve vars, removing ":0" suffix from names.
    uninitialized_vars = [
        v for v in self._vars if v.name[:-2] in uninitialized_var_names
    ]
    tf.logging.info('Initialize variables: %s',
                    [v.name for v in uninitialized_vars])
    sess.run(tf.variables_initializer(uninitialized_vars))

  def RestoreGlobalStepIfNeeded(self, sess):
    """If global step is not initialized, load it from the checkpoint.

    Args:
      sess: tf.Session.
    """
    assert not self._save_only
    uninitialized_vars = sess.run(self._uninitialized_vars)
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
