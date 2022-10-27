# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""A saver that we use to save and restore variables.

The implementation mimics tf.train.Saver. Meanwhile, it allows us
to carry out extra sanity checks on the checkpoint.
"""

import collections
import re
import threading
import time
from lingvo import compat as tf
from lingvo.core import py_utils
import numpy as np
from pkg_resources import parse_version
from google.protobuf import text_format
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import monitoring
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
# pylint: enable=g-direct-tensorflow-import

# Captures the timestamp of the first Saver object instantiation or end of a
# save operation. Can be accessed by multiple Saver instances.
_END_TIME_OF_LAST_WRITE = None
_END_TIME_OF_LAST_WRITE_LOCK = threading.Lock()

# API label for cell names used in TF1 async checkpoint metrics.
_ASYNC_CHECKPOINT_V1 = "async_checkpoint_v1"

_async_checkpoint_op_time_seconds = monitoring.Sampler(
    "/lingvo_lib/core/saver/async_checkpoint_op_secs",
    monitoring.ExponentialBuckets(0.5, 1.3, 40),
    "Distribution of the duration in seconds for async checkpoint ops.")

# The TF version (or newer) required for the metrics recording.
# TODO(lingvo-dev): Remove this and the related code once the TF build
#                      version is bumped to 2.11.0 or later.
_TF_METRIC_VERSION = "2.11.0"
_SHOULD_RECORD_METRIC = parse_version(
    tf.compat.v1.__version__) >= parse_version(_TF_METRIC_VERSION)


def _GetDurationMicroseconds(start_time_seconds, end_time_seconds):
  """Returns the duration between start and end time in microseconds."""
  return max(int((end_time_seconds - start_time_seconds) * 1000000), 0)


class SanityCheck:

  def Check(self, *args):
    """Returns true iff the sanity check passes."""
    raise NotImplementedError()


class InRange(SanityCheck):
  """Sanity check a value is within [low, high]."""

  def __init__(self, low, high):
    self._low = low
    self._high = high

  def __str__(self):
    return "InRange({}, {})".format(self._low, self._high)

  def Check(self, value):
    return self._low <= value <= self._high


class IsFinite(SanityCheck):

  def __str__(self):
    return "IsFinite"

  def Check(self, value):
    return np.all(np.isfinite(value))


def _VarKey(var):
  return var.name[:-2]  # strip :0


class Saver:
  """Simpler version of tf.train.Saver with extra sanity checks.

  This version of the Saver has support for sanity checking variable
  values before saving a checkpoint.  Additionally, it has more defined
  behavior with respect to pre-emptions. The tf.train.Saver by default
  creates a new checkpoint state file each run, which breaks the `keep_latest_n`
  functionality.  However, if the checkpoint state file is reloaded each run,
  it breaks the `keep_every_n_hours` functionality as old checkpoints
  that were meant to be persisted indefinitely will be deleted.


  The garbage collection policy only impacts checkpoints that
  are tracked in the checkpoint state file.

  This has the following potential issues:
  1) The checkpoint save might happen without the state getting updated
  (pre-emption in between), which means that this checkpoint will never
  be GCed since it never makes it to the state file.

  2) It's also possible that a pre-emption occurs after saving the
  checkpoint state file but before checkpointing, so checkpoints
  that would have otherwise been deleted will live forever.
  """

  RE_PATTERN = re.compile(r"^.*/ckpt-(\d+).*$")

  def __init__(self,
               logdir,
               variables,
               sanity_checks=None,
               keep_latest_n=None,
               keep_every_n_hours=None,
               async_save=False):
    self._logdir = logdir
    self._state_file = "{}/checkpoint".format(self._logdir)
    self._vars = variables
    var_graphs = set([v.graph for v in self._vars])
    assert len(var_graphs) == 1  # All vars should be in the same graph.
    self._var_graph = list(var_graphs)[0]

    self._async_save = async_save
    self._copied_vars = None
    self._copying_op = []
    self._async_save_thread = None
    self._async_exception = None
    assert not sanity_checks or all(
        isinstance(x[1], SanityCheck) for x in sanity_checks)
    self._sanity_checks = sanity_checks

    if not keep_latest_n:
      self._keep_latest_n = 0
    else:
      self._keep_latest_n = keep_latest_n
    self._keep_every_n_hours = keep_every_n_hours
    self._logdir_ph = tf.placeholder(tf.string, shape=[])
    self._restore_prefix_ph = tf.placeholder(tf.string, shape=[])
    self._BuildSave()
    self._BuildRestore()
    tf.logging.info("Saver: %s %s %s", self._logdir, self._keep_latest_n,
                    self._keep_every_n_hours)
    self._InitTrainingTimeSavedMetric()

  def _InitTrainingTimeSavedMetric(self):
    """Initialize the first timestamp for _END_TIME_OF_LAST_WRITE."""
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      if _END_TIME_OF_LAST_WRITE is None:
        _END_TIME_OF_LAST_WRITE = time.time()

  def _AddShardedSaveOps(self, variables, checkpoint_prefix, var_key_fn):
    """Adds per-device save ops to save `variables` to `checkpoint_prefix`."""
    with self._var_graph.as_default():
      per_device = collections.defaultdict(lambda: [])
      for var in variables:
        per_device[var.device].append(var)

      tmp_save_prefix = tf.strings.join([checkpoint_prefix, "_temp/part"])
      num_shards = tf.constant(len(per_device))
      sharded_saves = []
      sharded_prefixes = []

      for shard, (device, var_list) in enumerate(per_device.items()):
        with self._var_graph.device(device):
          sharded_filename = gen_io_ops.sharded_filename(
              tmp_save_prefix, shard, num_shards)
          sharded_prefixes.append(sharded_filename)
          save_op = io_ops.save_v2(
              prefix=sharded_filename,
              tensor_names=[var_key_fn(v) for v in var_list],
              tensors=[v.read_value() for v in var_list],
              shape_and_slices=[""] * len(var_list))
          sharded_saves.append(save_op)

      with tf.control_dependencies(sharded_saves):
        return gen_io_ops.merge_v2_checkpoints(
            sharded_prefixes, checkpoint_prefix, delete_old_dirs=True)

  def _BuildSave(self):
    """Builds save ops."""
    self._save_global_step = py_utils.GetGlobalStep()
    self._save_prefix = tf.strings.join([
        self._logdir_ph, "/ckpt-",
        tf.as_string(self._save_global_step, width=8, fill="0")
    ])
    self._copied_vars = []
    self._copying_op = None

    if not self._async_save:
      self._save_op = self._AddShardedSaveOps(self._vars, self._save_prefix,
                                              _VarKey)
      self._copied_vars_initializer = tf.no_op()
    else:
      # Creating a copy of the vars. We need to add the graph ops here (during
      # construction) to avoid adding duplicate functions during execution
      # (b/226390414).
      # Note: the copy will be created in self._var_graph regardless which graph
      # is set as default, so we need to apply the device context in
      # self._var_graph.
      copying_ops = []
      with self._var_graph.as_default():
        for v in self._vars:
          with self._var_graph.device(v.device):
            # Initialize with a constant scalar to avoid consuming large amount
            # of memory unnecessarily.
            # According to TF, use tf.TensorShape(None) (unspecified shape) so
            # that the variable can later be assigned with values of different
            # shapes.
            assert v.name.endswith(":0")
            copied_v = tf.compat.v2.Variable(
                tf.cast(0, v.dtype),
                trainable=False,
                name=f"async_ckpt/{v.name[:-2]}",
                dtype=v.dtype,
                shape=tf.TensorShape(None))
            assert copied_v.graph is v.graph
            if v.device:
              assert copied_v.device == v.device
            else:
              # When v.device is empty, the device of the copied variable is
              # decided by the placer.
              # TODO(laigd): this should not happen for model variables during
              # training, find a way to check that.
              pass
            self._copied_vars.append(copied_v)
            copying_ops.append(copied_v.assign(v))
      # Group the ops to avoid running them directly, which will generate
      # expensive send/recv operations.
      self._copying_op = tf.group(*copying_ops)
      self._copied_vars_initializer = tf.group(
          *[v.initializer for v in self._copied_vars])

      copied_var_map = {
          id(copied_var): var
          for copied_var, var in zip(self._copied_vars, self._vars)
      }
      self._save_op = self._AddShardedSaveOps(
          self._copied_vars, self._save_prefix,
          lambda copied_var: _VarKey(copied_var_map[id(copied_var)]))

  def _BuildRestore(self):
    """Builds restore ops."""
    assign_ops = []
    with self._var_graph.as_default():
      per_device = collections.defaultdict(lambda: [])
      for var in self._vars:
        per_device[var.device].append(var)

      for device, var_list in per_device.items():
        with self._var_graph.device(device):
          for var in var_list:
            val, = io_ops.restore_v2(
                prefix=self._restore_prefix_ph,
                tensor_names=[_VarKey(var)],
                shape_and_slices=[""],
                dtypes=[var.dtype])
            assign_ops.append(var.assign(val))

    self._restore_op = tf.group(*assign_ops)

  def _GetState(self):
    """Returns the latest checkpoint id."""
    state = CheckpointState()
    if file_io.file_exists(self._state_file):
      content = file_io.read_file_to_string(self._state_file)
      text_format.Parse(content, state)
    return state

  def _SetState(self, state):
    file_io.atomic_write_string_to_file(self._state_file,
                                        text_format.MessageToString(state))

  @staticmethod
  def GetCheckpointId(filename_prefix):
    match = Saver.RE_PATTERN.match(filename_prefix)
    assert match, "Unexpected {} does not match re({})".format(
        filename_prefix, Saver.RE_PATTERN)
    return int(match.group(1))

  def _GarbageCollect(self, ids_to_garbage_collect):
    """Garbage collect obsolete checkpoint files.

    Args:
      ids_to_garbage_collect: A set of int ids to delete.
    """

    existing_files = tf.io.gfile.glob(r"{}/ckpt-*".format(self._logdir))
    # Filter to make sure we catch only the ckpt files.
    existing_files = [f for f in existing_files if self.RE_PATTERN.match(f)]
    for filename in existing_files:
      if self.GetCheckpointId(filename) in ids_to_garbage_collect:
        # TODO(zhifengc): May need to find a bulk delete method.
        tf.logging.info("Garbage collecting %s", filename)
        tf.io.gfile.remove(filename)

  def _DoSanityCheck(self, prefix):
    """Sanity-check the content of the checkpoint."""
    if not self._sanity_checks:
      return
    reader = tf.train.NewCheckpointReader(prefix)
    checks = collections.defaultdict(lambda: [])
    for variables, rule in self._sanity_checks:
      for v in variables:
        key = _VarKey(v)
        checks[key].append(rule)
    for key, rules in checks.items():
      value = reader.get_tensor(key)
      for rule in rules:
        if not rule.Check(value):
          # TODO(zhifengc): Maybe should return an explicit signal
          # so that the caller (the controller loop) can Restore()
          # the latest checkpoint before raise the error.
          msg = f"Checkpoint sanity check failed: {prefix} {key} {rule}\n"
          # Also saves the error message into a file.
          file_io.write_string_to_file("{}.failed".format(prefix), msg)
          raise tf.errors.AbortedError(None, None, msg)

  def _SaveAsync(self, sess):
    """Saves the graph asynchronously.

    All the variables are first copied, synchronously, in memory to another set
    of vars, and then the saving to disk is done in a different thread.

    The function blocks till the previous saving is done.

    Args:
      sess: A session with tf.Graph under which this object is constructed.

    Returns:
      Returns the global step and file prefix.
    """
    # Waiting for the previous save to finish.
    self.Sync()
    if self._async_exception is not None:
      e = self._async_exception
      self._async_exception = None
      raise e

    sess.run(self._copying_op)
    global_step, prefix = sess.run(
        fetches=[self._save_global_step, self._save_prefix],
        feed_dict={self._logdir_ph: self._logdir})
    prefix = tf.compat.as_text(prefix)
    tf.logging.info("Saving asynchronously to %s", prefix)

    def _Async(global_step, prefix):
      checkpoint_start_time = time.perf_counter()
      start_time = time.time()
      try:
        # Use the provided prefix in case self._save_global_step is changed.
        _ = sess.run(
            fetches=self._save_op, feed_dict={self._save_prefix: prefix})
        self._FinalizeSave(global_step, prefix)
      except Exception as e:  # pylint: disable=broad-except
        self._async_exception = e
      end_time = time.time()
      _async_checkpoint_op_time_seconds.get_cell().add(time.perf_counter() -
                                                       checkpoint_start_time)

      if _SHOULD_RECORD_METRIC:
        metrics.AddAsyncCheckpointWriteDuration(
            api_label=_ASYNC_CHECKPOINT_V1,
            microseconds=_GetDurationMicroseconds(start_time, end_time))

        # Measure the elapsed time since the last checkpoint.
        # Due to the nature of async checkpoint, here it actually captures the
        # duration between the start_time of the previous checkpoint and the
        # start time of this checkpoint. As a result, the duration of the final
        # async checkpoint is excluded, which is fine since it does not take
        # much time.
        self._RecordTrainingTimeSavedMetric(start_time)

    self._async_save_thread = threading.Thread(
        target=_Async, args=(global_step, prefix))
    self._async_save_thread.start()
    return global_step, prefix

  def _SaveSync(self, sess):
    """Saves the graph."""
    _, global_step, prefix = sess.run(
        fetches=[self._save_op, self._save_global_step, self._save_prefix],
        feed_dict={self._logdir_ph: self._logdir})
    prefix = tf.compat.as_text(prefix)
    global_step, prefix = self._FinalizeSave(global_step, prefix)
    self._RecordTrainingTimeSavedMetric(time.time())

    return global_step, prefix

  def _FinalizeSave(self, global_step, prefix):
    """Runs sanity check and updates status."""
    if not tf.executing_eagerly():
      # Many users expect this as the tf.train.Saver does this by default.
      meta_graph_filename = prefix + ".meta"
      tf.train.export_meta_graph(filename=meta_graph_filename)

    # We can do extra sanity checks.
    self._DoSanityCheck(prefix)

    # Commit new state.
    self._UpdateState(prefix)

    tf.logging.info("Saved %d %s", global_step, prefix)
    return global_step, prefix

  def _RecordTrainingTimeSavedMetric(self, end_time):
    """Record the training_time_saved metric.

    Args:
      end_time: The end time of the training time saved.
    """
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      metrics.AddTrainingTimeSaved(
          api_label=_ASYNC_CHECKPOINT_V1,
          microseconds=_GetDurationMicroseconds(_END_TIME_OF_LAST_WRITE,
                                                end_time))
      _END_TIME_OF_LAST_WRITE = end_time

  def Save(self, sess):
    """Generate a new checkpoint.

    May raise exceptions for failures or sanity checks. If using async mode,
    exceptions are raised in the following Save call.

    Args:
      sess: A session with tf.Graph under which this object is constructed.

    Returns:
      Returns the global step and file prefix.
    """
    blocking_time_start = time.time()
    global_step, file_prefix = self._SaveAsync(
        sess) if self._async_save else self._SaveSync(sess)
    blocking_time_end = time.time()
    metrics.AddCheckpointWriteDuration(
        api_label=_ASYNC_CHECKPOINT_V1,
        microseconds=_GetDurationMicroseconds(blocking_time_start,
                                              blocking_time_end))
    return global_step, file_prefix

  def _UpdateState(self, prefix):
    """Updates the checkpoint state with the new checkpoint prefix."""
    # The checkpoint looks OK. Commit it to the state.
    state = self._GetState()

    # Latest checkpoint.
    state.model_checkpoint_path = prefix
    state.last_preserved_timestamp = time.time()

    # Checkpoints are kept based on two independent policies
    # 1) keep_latest_n: The most recent n are kept.
    # 2) keep_every_n_hours: These checkpoints are kept
    # indefinitely, in addition to those in 1).
    # If more policies are added, a more generic interface is probably
    # warranted.

    indefinite_retention = []
    recent_retention = []

    # For more convenient list slicing.
    all_model_checkpoint_pairs = list(
        zip(state.all_model_checkpoint_timestamps,
            state.all_model_checkpoint_paths))

    # Add current checkpoint.
    all_model_checkpoint_pairs.append(
        (state.last_preserved_timestamp, state.model_checkpoint_path))

    # Explicitly sort by timestamp ascending just to be safe.
    # sort() will sort ascending by first element in the tuple.
    all_model_checkpoint_pairs.sort()

    # Select checkpoints for indefinite retention
    latest_indefinite_ts = 0
    if self._keep_every_n_hours:
      keep_every_n_secs = 3600.0 * self._keep_every_n_hours
      for ts, path in all_model_checkpoint_pairs:
        if ts - latest_indefinite_ts > keep_every_n_secs:
          latest_indefinite_ts = ts
          indefinite_retention.append((ts, path))

    # pylint: disable=invalid-unary-operand-type
    # Apply recent retention policy.
    recent_retention = all_model_checkpoint_pairs[-self._keep_latest_n:]
    # pylint: enable=invalid-unary-operand-type

    retained_pairs = indefinite_retention + recent_retention
    # sort() will sort ascending by first element in the tuple.
    retained_pairs.sort()

    unique_paths = []
    unique_ts = []
    path_set = set()
    for ts, path in retained_pairs:
      if path not in path_set:
        unique_paths.append(path)
        unique_ts.append(ts)
        path_set.add(path)

    # Identify all checkpoints that were in the state file before
    # but will be garbage collected due to the policy.
    ids_to_garbage_collect = set()
    for ts, path in all_model_checkpoint_pairs:
      if path not in path_set:
        ids_to_garbage_collect.add(self.GetCheckpointId(path))

    # Serialize to state.
    state.all_model_checkpoint_paths[:] = unique_paths
    state.all_model_checkpoint_timestamps[:] = unique_ts

    self._SetState(state)
    self._GarbageCollect(ids_to_garbage_collect)

  def Restore(self, sess, path=None, checkpoint_id=None):
    """Restore variables from a checkpoint.

    Args:
      sess: A session with tf.Graph under which this object is constructed.
      path: If not None, restore from this path prefix.
      checkpoint_id: If None, restore from the latest checkpoint. Otherwise,
        restore from the specific checkpoint.

    Returns:
      If no checkpoint has been generated, returns (0, "").  Otherwise, try to
      restore from the specified or the latest checkpoint. If the checkpoint is
      successfully restored, returns the checkpoint's global step and file
      prefix. Otherwise, raises an error.
    """
    self.Sync()
    if path:
      prefix = path
    elif checkpoint_id:
      prefix = "{}/ckpt-{:08d}".format(self._logdir, checkpoint_id)
    else:
      prefix = self._GetState().model_checkpoint_path
      if not prefix:
        return 0, ""

    sess.run(
        fetches=[self._restore_op], feed_dict={self._restore_prefix_ph: prefix})
    global_step = self.GetCheckpointId(prefix)
    tf.logging.info("Restored %d %s", global_step, prefix)
    if self._async_save:
      # Initialize the copied variables after a successful restore, to make sure
      # all variables are initialized (since copied variables are not saved in
      # the checkpoints).
      sess.run(self._copied_vars_initializer)
    return global_step, prefix

  def Sync(self):
    """Wait for any outstanding async operations to finish."""
    if self._async_save_thread is not None:
      self._async_save_thread.join()
      self._async_save_thread = None


def WriteNpArrays(file_prefix, nmap):
  """Writes a NestedMap of numpy arrays into a TF checkpoint.

  Args:
    file_prefix: A TF checkpoint filename prefix.
    nmap: A NestedMap of numpy arrays.
  """
  g = tf.Graph()
  with g.as_default():

    def Wrap(val):
      dtype = tf.as_dtype(val.dtype)
      assert dtype != tf.string  # tf.string is not supported by py_func.
      return tf.py_func(lambda: val, [], dtype)

    names, values = [], []
    for k, v in nmap.FlattenItems():
      names.append(k)
      assert isinstance(v, np.ndarray)
      values.append(Wrap(v))

    save = io_ops.save_v2(
        prefix=file_prefix,
        tensor_names=names,
        tensors=values,
        shape_and_slices=[""] * len(names))

  with tf.Session(graph=g) as sess:
    sess.run(save)


def ReadNpArrays(file_prefix, nmap):
  """Reads from a tf checkpoint to fill in values of a NesteMap.

  Args:
    file_prefix: A TF checkpoint filename prefix.
    nmap: A NestedMap of numpy dtypes.

  Returns:
    A NestedMap with numpy arrays compatible w/ nmap.
  """
  g = tf.Graph()
  with g.as_default():
    reads = []
    for name, dtype in nmap.FlattenItems():
      reads.append(
          io_ops.restore_v2(
              prefix=file_prefix,
              tensor_names=[name],
              shape_and_slices=[""],
              dtypes=[dtype])[0])

  with tf.Session(graph=g) as sess:
    vals = sess.run(reads)

  return nmap.Pack(vals)
