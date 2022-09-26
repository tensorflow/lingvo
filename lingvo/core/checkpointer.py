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

import copy
import os
import time

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core import saver as custom_saver
import six

tf.flags.DEFINE_boolean('use_custom_saver', True,
                        'Uses customized saver if True.')
FLAGS = tf.flags.FLAGS


def SortCheckpointPaths(ckpts):
  """Sorts checkpoints based on the step number in the paths."""
  return sorted(ckpts, key=lambda x: int(x.split('-')[-1]))


class SaverWrapper:
  """Wrapper interface between tf.train.Saver and the custom saver."""

  def __init__(self,
               logdir,
               train_params,
               variables_to_restore_dict=None,
               async_save=False):
    """Create a tf.train.Saver or a custom_saver.Saver.

    Args:
      logdir: The directory path to save checkpoints to.
      train_params: Training parameters.
      variables_to_restore_dict: A dictionary mapping names to Saveables.
        Typically, used in evaluation for substituting exponential moving
        average weights.  If this is set, then tf.train.Saver is used.
      async_save: Save asynchronously. Only works with custom saver.
    """
    self._logdir = logdir
    self._save_path = os.path.join(self._logdir, 'ckpt')
    self._use_custom_saver = (
        FLAGS.use_custom_saver and not variables_to_restore_dict)
    if async_save and not self._use_custom_saver:
      tf.logging.warning('Asynchronous saving only works with custom saver.')

    self._keep_latest_n = train_params.save_max_to_keep
    self._keep_every_n_hours = train_params.save_keep_checkpoint_every_n_hours
    self._max_steps = train_params.max_steps
    self._tpu_steps_per_loop = train_params.tpu_steps_per_loop

    if not self._use_custom_saver:
      tf.logging.info('Instantiating tf.train.Saver')
      self._saver = tf.train.Saver(
          variables_to_restore_dict,
          sharded=True,
          max_to_keep=self._keep_latest_n,
          keep_checkpoint_every_n_hours=self._keep_every_n_hours,
          pad_step_number=True,  # %08d
          write_version=tf.train.SaverDef.V2)
      self._var_list = self._saver._var_list  # pylint: disable=protected-access
    else:
      tf.logging.info('Instantiating custom Saver')
      gsv = py_utils.GetOrCreateGlobalStepVar()
      self._var_list = tf.all_variables()

      if self._max_steps and self._tpu_steps_per_loop:
        sanity_checks = [
            ([gsv],
             custom_saver.InRange(0,
                                  self._max_steps + self._tpu_steps_per_loop))
        ]
      else:
        sanity_checks = []

      if train_params.checkpoint_finite_check:
        for var in self._var_list:
          sanity_checks.append(([var], custom_saver.IsFinite()))

      self._saver = custom_saver.Saver(
          logdir,
          variables=self._var_list,
          sanity_checks=sanity_checks,
          keep_latest_n=self._keep_latest_n,
          keep_every_n_hours=self._keep_every_n_hours,
          async_save=async_save)

  def Save(self, sess, gsteps):
    """Save a checkpoint.

    Args:
      sess: tf.Session.
      gsteps: Current global step.

    Returns:
      Path prefix to the checkpoint.
    """
    if not self._use_custom_saver:
      path = self._saver.save(sess, self._save_path, gsteps)
    else:
      del gsteps
      gsteps, path = self._saver.Save(sess)
    return path

  def Restore(self, sess, path):
    """Restore from a checkpoint.

    Args:
      sess: tf.Session.
      path: Path prefix to the checkpoint.
    """
    if not self._use_custom_saver:
      self._saver.restore(sess, path)
    else:
      self._saver.Restore(sess, path=path)

  def Sync(self):
    """Wait for any outstanding async saving operations to finish."""
    if self._use_custom_saver:
      self._saver.Sync()


class Checkpointer:
  """Checkpointing utility class.

  Needs to be created within a graph context.
  """

  def __init__(self,
               train_dir,
               models,
               train_params=None,
               save_only=False,
               check_loading_status=True):
    """Initialize Checkpointer.

    Args:
     train_dir: Training directory for saving checkpoints.
     models: One or a list of BaseModel instances. Cannot be empty. If there are
       more than one models and `train_params` is None, the save intervals will
       be only determined by the first model.
     train_params: If specified, use these training params instead of those in
       the `model`.
     save_only: This checkpointer is only intended for saving checkpoints.
     check_loading_status: Set to True to have extra safety checks when loading
       a checkpoint file. The safety check should be skipped when a mismatch is
       expected between the model and the checkpoint file. For example, a evaler
       is not expected to load all the checkpointed variables from a trainer
       because the evaler does not have an optimizer.
    """
    tf.logging.info('Starting checkpointer')
    self._train_dir = train_dir
    self._save_only = save_only
    self._save_path = os.path.join(self._train_dir, 'ckpt')
    self._check_loading_status = check_loading_status

    if not isinstance(models, (list, tuple)):
      models = [models]
    self._models = models

    if train_params:
      self._train_params = train_params
    else:
      self._train_params = models[0].params.train

    self._next_checkpoint_seconds = 0
    self._save_interval_seconds = self._train_params.save_interval_seconds
    self._save_interval_steps = self._train_params.save_interval_steps
    self._prev_ckpt_step = None
    self._saver = self._GetSaver()
    # self._saver may create variables used by async checkpointing, so we
    # need to get the global_variables_initializer after its creation.
    self._init_op = tf.global_variables_initializer()

    if not py_utils.IsEagerMode():
      self._uninitialized_vars = tf.report_uninitialized_variables(
          tf.global_variables())

    # In Eager mode, the init rules are loaded lazily to ensure slot variables
    # can also be loaded successfully.
    if not py_utils.IsEagerMode():
      self._BuildInitFromCheckpointRules()

  @property
  def checkpoint_dir(self) -> str:
    return self._train_dir

  def _GetSaver(self):
    """Returns a saver."""
    do_eval = cluster_factory.Current().do_eval
    variables_to_restore = {}
    if not self._save_only and do_eval:
      for model in self._models:
        if model.ema:
          tf.logging.info('Using EMA for evaluation.')
          variables_to_restore.update(
              model.ema.variables_to_restore(model.variables_for_ema))
    if not variables_to_restore:
      variables_to_restore = None
    return SaverWrapper(
        self._train_dir,
        self._train_params,
        variables_to_restore_dict=variables_to_restore,
        async_save=self.async_checkpointing)

  @property
  def async_checkpointing(self):
    return self._train_params.async_checkpointing

  def _BuildInitFromCheckpointRules(self):
    """Build restore fns for init_from_checkpoint_rules."""
    self._restore_fns = []
    all_vars = list(_GetSaveableVariablesDict(self._models).values())

    # TODO(b/160786085): Move this logic into Overriding vars logic itself,
    # which requires refactoring things out of py_utils to avoid circular deps.
    def _ResolveCkptPath(ckpt_rules):
      res_rules = {}
      for k, v in ckpt_rules.items():
        new_k = GetSpecificCheckpoint(k)
        if not new_k:
          tf.logging.warning(
              f'Empty checkpoint path init rules are ignored, key={k}')
        else:
          res_rules.update({new_k: v})
      return res_rules

    def _MergeLoadingRules(a, b):
      res = copy.deepcopy(a)
      for k, (load_rules, ignore_rules) in b.items():
        if k in res:
          res_load, res_ignore = res[k]
          for load in load_rules:
            if load not in res_load:
              res_load.append(load)
          for ignore in ignore_rules:
            if ignore not in res_ignore:
              res_ignore.append(ignore)
        else:
          res[k] = (load_rules, ignore_rules)
      return res

    # Restore specific variables based on init_from_checkpoint_rules.
    rules = {}
    for model in self._models:
      for task in model.tasks:
        tp = task.params.train
        if tp.init_from_checkpoint_rules:
          rules = _MergeLoadingRules(
              rules, _ResolveCkptPath(tp.init_from_checkpoint_rules))

    if self._train_params.init_from_checkpoint_rules:
      rules = _MergeLoadingRules(
          rules,
          _ResolveCkptPath(self._train_params.init_from_checkpoint_rules))

    # Add graph nodes to restore specific variables based on
    # init_from_checkpoint_rules.
    # TODO(b/159267006): Move this back to Restore().
    self._restore_fns.append(
        (f'OverrideVarsFromCheckpoints {rules}',
         py_utils.OverrideVarsFromCheckpoints(all_vars, rules)))

  def RestoreFromPath(self, sess=None, checkpoint_path=None):
    """Load the checkpoint from specified path."""
    assert not self._save_only
    tf.logging.info('Load from checkpoint %s.', checkpoint_path)
    self._saver.Restore(sess, checkpoint_path)
    tf.logging.info('Load checkpoint done.')
    # Successfully restored from checkpoint.
    uninitialized_var_names = self._GetUninitializedVarNames(sess)
    if self._check_loading_status:
      assert not uninitialized_var_names, uninitialized_var_names

    return checkpoint_path

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

  def MaybeSave(self, sess=None, gsteps=None, sync=True):
    """If it's time to save, save the checkpoint.

    Args:
      sess: tf.Session.
      gsteps: Current global step.
      sync: Whether to wait for the saving operations to finish. Only applicable
        when async checkpointing is used.

    Returns:
      Whether a checkpoint was saved.
    """
    if self.ShouldSave(gsteps):
      self.Save(sess, gsteps, sync)
      return True
    return False

  def Save(self, sess=None, gsteps=None, sync=True):
    """Save the checkpoint.

    Args:
      sess: tf.Session.
      gsteps: Current global step.
      sync: Whether to wait for the saving operations to finish. Only applicable
        when async checkpointing is used.
    """
    tf.logging.info('Save checkpoint')
    path = self._saver.Save(sess, gsteps)
    tf.logging.info('Save checkpoint done: %s', path)
    self._prev_ckpt_step = gsteps
    self._UpdateNextSaveTime()
    if sync:
      self.Sync()

  def Sync(self):
    """Wait for any outstanding async operations to finish."""
    self._saver.Sync()

  def _UpdateNextSaveTime(self):
    now = time.time()
    self._next_checkpoint_seconds = now + self._save_interval_seconds

  def _RestoreFromLatestCheckpoint(self, sess=None):
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

  def Restore(self, sess=None, force_reinitialize=False):
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

    for msg, fn in self._restore_fns:
      tf.logging.info(msg)
      fn(sess)
    return None

  def RestoreIfNeeded(self, sess):
    """If vars are not initialized, restore from checkpoint."""
    assert not self._save_only
    uninitialized_var_names = self._GetUninitializedVarNames(sess)
    if not uninitialized_var_names:
      # All variables are already initialized.
      return None

    return self.Restore(sess)

  def RestoreGlobalStepIfNeeded(self, sess=None):
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


def _GetSaveableVariablesDict(models):
  """Get all variables of the model that should be saved.

  Args:
    models: a list of lingvo model objects.

  Returns:
    A map of the variables with their names as keys, trailing `:0` stripepd.

  Raises:
    RuntimeError: if there are variables with shared name.
  """
  res = {}
  for model in models:
    res = py_utils.MergeDictsWithValueCheck(res, model.GetVariablesDict())

  res_updated = {}
  for k in res:
    k_new = k
    # strip ':0' from variable names to be backwards compatible with graph mode
    # checkpoint keys
    if k[-2:] == ':0':
      k_new = k[:-2]
    res_updated[k_new] = res[k]

  res_updated['global_step'] = py_utils.GetGlobalStep()
  return res_updated


class _EagerCheckpointer(Checkpointer):
  """Eager mode checkpointer."""

  def __init__(self,
               train_dir,
               models,
               train_params=None,
               save_only=False,
               check_loading_status=True):
    """Initialize Checkpointer.

    Args:
     train_dir: Training directory for saving checkpoints.
     models: One or a list of BaseModel instances. Cannot be empty. If there are
       more than one models and `train_params` is None, the save intervals will
       be only determined by the first model.
     train_params: If specified, use these training params instead of those in
       the `model`.
     save_only: This checkpointer is only intended for saving checkpoints.
     check_loading_status: Set to True to have extra safety checks when loading
       a checkpoint file. The safety check should be skipped when a mismatch is
       expected between the model and the checkpoint file. For example, a evaler
       is not expected to load all the checkpointed variables from a trainer
       because the evaler does not have an optimizer.
    """
    # This cannot be None because in Eager mode the models are necessary to
    # get saveable variables.
    if not isinstance(models, (list, tuple)):
      models = [models]
    self._models = models
    self._init_rules_built = False
    super().__init__(train_dir, models, train_params, save_only,
                     check_loading_status)

  def _MaybeBuildInitFromCheckpointRules(self):
    """Build restore fns for init_from_checkpoint_rules."""
    if self._init_rules_built:
      return
    else:
      self._init_rules_built = True

    # In Eager mode this function is called lazily.
    # We use a guard to avoid repeated executions.
    self._BuildInitFromCheckpointRules()

  def RestoreIfNeeded(self, sess):
    raise TypeError('Not supported in Eager mode')

  def _MaybeOverwriteModelVariablesWithEMA(self):
    """Overwrite model variables with EMA shadow variables in eval mode."""
    do_eval = cluster_factory.Current().do_eval
    if not self._save_only and do_eval:
      for model in self._models:
        if not model.ema:
          continue
        tf.logging.info('Using EMA for evaluation.')
        # TODO(jiaweix): this implementation will load both the model variables
        # and EMA variables. As a result the memory usage will be higher than
        # the eval jobs in TF1 mode.
        ema = model.ema
        cur_vars = model.GetVariablesDict()
        for v in cur_vars.values():
          shadow_v = ema.average(v)
          if shadow_v is not None:
            v.assign(shadow_v)


class EagerCheckpointerV1(_EagerCheckpointer):
  """Eager mode V1 checkpointer."""

  def __init__(self,
               train_dir,
               models,
               train_params=None,
               save_only=False,
               check_loading_status=True):
    super().__init__(train_dir, models, train_params, save_only,
                     check_loading_status)
    tf.logging.info('Starting eager checkpointer v1')
    self._train_dir = self._train_dir
    if not tf.io.gfile.exists(self._train_dir):
      tf.io.gfile.makedirs(self._train_dir)

    # Set to None; delay the initialization after the model ran at least once
    self._saver = None
    self._save_path = os.path.join(self._train_dir, 'ckpt')
    self._restorer = None

  def _GetSaver(self):
    all_vars = _GetSaveableVariablesDict(self._models)
    # TODO(b/217920843): sharded=True is needed to avoid sending all variables
    # to one worker before saving the checkpoints, but it's not well supported
    # in eager mode due to the use of Tensor.op. Investigate and fix it.
    saver = tf.train.Saver(
        var_list=all_vars,
        max_to_keep=self._train_params.save_max_to_keep,
        keep_checkpoint_every_n_hours=(
            self._train_params.save_keep_checkpoint_every_n_hours),
        pad_step_number=True)  # %08d
    return saver

  def _GetRestorer(self):
    """Use Checkpoint to restore checkpoint files created by Saver."""
    all_vars = _GetSaveableVariablesDict(self._models)
    # Restore name-based tf.compat.v1.train.Saver checkpoints with Checkpoint.
    restorer = tf.train.Checkpoint(variables=all_vars)
    return restorer

  def _WrapRestoreErrorWithGraphModeWarning(self, err):
    return ValueError(
        'Could not restore V1 checkpoint during eager mode. If you are '
        'restoring optimizer slot variables from a non-eager checkpoint, '
        'please set p.train.optimizer.clear_variable_scope = True. See '
        'b/218397533 for more details.\n\nOriginal Error: {}'.format(err))

  def Restore(self, sess=None, force_reinitialize=None):
    """`sess` and `force_reinitialize` are unused in Eager context."""
    assert sess is None
    path = self._RestoreFromLatestCheckpoint(sess)
    if path:
      tf.logging.info('Eager checkpoint is restored with path: %s', path)
      return path
    else:
      # No checkpoint is loaded, we need to initialize the variables,
      # and apply the init_from_checkpoint_rules if applicable.
      try:
        self._MaybeBuildInitFromCheckpointRules()
        for msg, fn in self._restore_fns:
          tf.logging.info(msg)
          fn(sess)
        return path
      except tf.errors.NotFoundError as err:
        raise self._WrapRestoreErrorWithGraphModeWarning(err)

  def RestoreGlobalStepIfNeeded(self, sess=None):
    """`sess` is unused in Eager context."""
    assert sess is None
    assert not self._save_only

    gstep = py_utils.GetGlobalStep()
    path = tf.train.latest_checkpoint(self._train_dir)
    if path:
      reader = tf.train.load_checkpoint(path)
      value = reader.get_tensor('global_step')
      gstep.assign(value)
      tf.logging.info('Restoring global step: %s', value)
    else:
      tf.logging.info('Cannot find checkpoints, using existing global_step.')

  def RestoreFromPath(self, sess=None, checkpoint_path=None):
    """`sess` is unused in Eager context."""
    # TODO(jiaweix): `check_loading_status` is not supported in V1 checkpointer.
    # For robustness we need to use V2 checkpointer as much as posisble.
    assert sess is None
    assert not self._save_only

    # Calling this before `Save` because the optimizer and EMA variables are not
    # created until at least one training step in the Eager trainer.
    if not self._restorer:
      self._restorer = self._GetRestorer()

    assert not self._save_only
    tf.logging.info('Load from checkpoint (V1) %s.', checkpoint_path)

    try:
      self._restorer.restore(save_path=checkpoint_path)
    except tf.errors.NotFoundError as err:
      raise self._WrapRestoreErrorWithGraphModeWarning(err)

    tf.logging.info('Load checkpoint done.')
    self._MaybeOverwriteModelVariablesWithEMA()
    return checkpoint_path

  def Save(self, sess=None, gsteps=None, sync=True):
    """`sess` is unused in Eager context."""
    assert sess is None

    # Calling this before `Save` because the optimizer and EMA variables are not
    # created until at least one training step in the Eager trainer.
    if not self._saver:
      self._saver = self._GetSaver()

    tf.logging.info('Save checkpoint (V1)')
    path = self._saver.save(
        sess=None, save_path=self._save_path, global_step=gsteps)
    tf.logging.info('Save checkpoint (V1) done: %s', path)
    self._prev_ckpt_step = gsteps
    self._UpdateNextSaveTime()

  def Sync(self):
    # Async checkpointing is not implemented in eager mode.
    pass


class EagerCheckpointerV2(_EagerCheckpointer):
  """Eager mode V2 checkpointer."""

  def __init__(self,
               train_dir,
               models,
               train_params=None,
               save_only=False,
               check_loading_status=True,
               experimental_enable_async_checkpoint=False):
    super().__init__(train_dir, models, train_params, save_only,
                     check_loading_status)
    tf.logging.info('Starting eager checkpointer v2')
    # Distinct from EagerCheckpointerV1
    self._train_dir = os.path.join(self._train_dir, 'ckpt_V2')
    if not tf.io.gfile.exists(self._train_dir):
      tf.io.gfile.makedirs(self._train_dir)

    # Set to None; delay the initialization after the model ran at least once
    self._saver = None
    self._save_path = os.path.join(self._train_dir, 'ckpt')
    self._enable_async = experimental_enable_async_checkpoint

  def _GetSaver(self):
    saver = tf.train.Checkpoint(variables=self._models)
    # Use the manager to support features e.g. max number of checkpoints
    self._saver_mgr = tf.train.CheckpointManager(
        saver,
        directory=self._train_dir,
        max_to_keep=self._train_params.save_max_to_keep,
        keep_checkpoint_every_n_hours=(
            self._train_params.save_keep_checkpoint_every_n_hours),
        checkpoint_name='ckpt')
    return saver

  def Restore(self, sess=None, force_reinitialize=None):
    """`sess` and `force_reinitialize` are unused in Eager context."""
    assert sess is None
    path = self._RestoreFromLatestCheckpoint(sess)
    if path:
      tf.logging.info('Eager checkpoint is restored with path: %s', path)
      return path
    # No checkpoint is loaded, we need to initialize the variables,
    # and apply the init_from_checkpoint_rules if applicable.
    self._MaybeBuildInitFromCheckpointRules()
    for msg, fn in self._restore_fns:
      tf.logging.info(msg)
      fn(sess)
    return path

  def RestoreGlobalStepIfNeeded(self, sess=None):
    """`sess` is unused in Eager context."""
    assert sess is None
    assert not self._save_only

    gstep = py_utils.GetGlobalStep()
    path = tf.train.latest_checkpoint(self._train_dir)
    if path:
      reader = tf.train.load_checkpoint(path)
      shapes = reader.get_variable_to_shape_map()
      step_var_keys = [v for v in shapes if 'global_step' in v]
      # Expecting only one variable with the name ‘global_step’
      assert len(step_var_keys) == 1, len(step_var_keys)
      value = reader.get_tensor(step_var_keys[0])
      gstep.assign(value)
      tf.logging.info('Restoring global step: %s', value)
    else:
      tf.logging.info('Cannot find checkpoints, using existing global_step.')

  def RestoreFromPath(self, sess=None, checkpoint_path=None):
    """`sess` is unused in Eager context."""
    assert sess is None
    assert not self._save_only

    # Calling this before `Save` because the optimizer and EMA variables are not
    # created until at least one training step in the Eager trainer.
    if not self._saver:
      self._saver = self._GetSaver()

    assert not self._save_only
    tf.logging.info('Load from checkpoint (V2) %s.', checkpoint_path)
    load_status = self._saver.restore(checkpoint_path)
    tf.logging.info('Load checkpoint done.')
    if self._check_loading_status:
      load_status.assert_existing_objects_matched().assert_consumed()

    self._MaybeOverwriteModelVariablesWithEMA()
    return checkpoint_path

  def Save(self, sess=None, gsteps=None, sync=True):
    """`sess` is unused in Eager context."""
    assert sess is None

    # Calling this before `Save` because the optimizer and EMA variables are not
    # created until at least one training step in the Eager trainer.
    if not self._saver:
      self._saver = self._GetSaver()

    tf.logging.info('Save checkpoint (V2)')
    options = tf.train.CheckpointOptions(
        experimental_enable_async_checkpoint=self._enable_async)
    path = self._saver_mgr.save(checkpoint_number=gsteps, options=options)
    tf.logging.info('Save checkpoint (V2) done: %s', path)
    self._prev_ckpt_step = gsteps
    self._UpdateNextSaveTime()

  def Sync(self):
    # Async checkpointing is not implemented in eager mode.
    pass


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
