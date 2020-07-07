# Lint as: python3
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

import re
import time
from lingvo import compat as tf
# pylint: enable=g-direct-tensorflow-import
from lingvo.core import py_utils
import numpy as np
from google.protobuf import text_format
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import io_ops
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState


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
  """Simpler version of tf.train.Saver with extra sanity checks."""

  def __init__(self,
               logdir,
               variables,
               sanity_checks=None,
               keep_latest_n=None,
               keep_every_n_hours=None):
    self._logdir = logdir
    self._state_file = "{}/checkpoint".format(self._logdir)
    self._vars = variables
    assert not sanity_checks or all(
        isinstance(x[1], SanityCheck) for x in sanity_checks)
    self._sanity_checks = sanity_checks
    self._keep_latest_n = keep_latest_n
    self._keep_every_n_hours = keep_every_n_hours
    self._re_pattern = re.compile(r"^.*/ckpt-(\d+).*$")
    self._logdir_ph = tf.placeholder(tf.string, shape=[])
    self._restore_prefix_ph = tf.placeholder(tf.string, shape=[])
    self._BuildSave()
    self._BuildRestore()
    tf.logging.info("Saver: %s %s %s", self._logdir, self._keep_latest_n,
                    self._keep_every_n_hours)

  def _BuildSave(self):
    """Builds save ops."""
    self._save_global_step = py_utils.GetGlobalStep()
    self._save_prefix = tf.strings.join([
        self._logdir_ph, "/ckpt-",
        tf.as_string(self._save_global_step, width=8, fill="0")
    ])
    self._save_op = io_ops.save_v2(
        prefix=self._save_prefix,
        tensor_names=[_VarKey(v) for v in self._vars],
        tensors=[v.read_value() for v in self._vars],
        shape_and_slices=[""] * len(self._vars))

  def _BuildRestore(self):
    """Builds restore ops."""
    assign_ops = []
    for var in self._vars:
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

  def _GetCheckpointId(self, filename_prefix):
    match = self._re_pattern.match(filename_prefix)
    assert match, "Unexpected {} does not match re({})".format(
        filename_prefix, self._re_pattern.pattern)
    return int(match.group(1))

  def _GarbageCollect(self):
    """Garbage collect obsolete checkpoint files."""
    state = self._GetState()

    valid_ids = set()
    if state.model_checkpoint_path:
      valid_ids.add(self._GetCheckpointId(state.model_checkpoint_path))
    for path in state.all_model_checkpoint_paths:
      valid_ids.add(self._GetCheckpointId(path))

    existing_files = tf.io.gfile.glob(r"{}/ckpt-*".format(self._logdir))
    # Filter to make sure we catch only the ckpt files.
    existing_files = [f for f in existing_files if self._re_pattern.match(f)]
    for filename in existing_files:
      if self._GetCheckpointId(filename) not in valid_ids:
        # TODO(zhifengc): May need to find a bulk delete method.
        tf.logging.info("Garbage collecting %s", filename)
        tf.io.gfile.remove(filename)

  def _DoSanityCheck(self, prefix):
    """Sanity-check the content of the checkpoint."""
    if not self._sanity_checks:
      return
    reader = tf.train.NewCheckpointReader(prefix)
    content = {}
    for variables, rule in self._sanity_checks:
      args = []
      for v in variables:
        key = _VarKey(v)
        if key in content:
          args.append(content[key])
        else:
          value = reader.get_tensor(key)
          content[key] = value
          args.append(value)
      if not rule.Check(*args):
        # TODO(zhifengc): Maybe should return an explicit signal
        # so that the caller (the controller loop) can Restore()
        # the latest checkpoint before raise the error.
        msg = "Checkpoint sanity check failed: {} {} {}\n".format(
            prefix, ",".join([_VarKey(v) for v in variables]), rule)
        # Also saves the error messge into a file.
        file_io.write_string_to_file("{}.failed".format(prefix), msg)
        raise tf.errors.AbortedError(None, None, msg)

  def Save(self, sess):
    """Generate a new checkpoint.

    Args:
      sess: A session with tf.Graph under which this object is constructred.

    Returns:
      If the checkpoint is successfully generated, returns its global step
      and file prefix. Otherwise, raises an Aborted error.
    """
    # Garbage collect. Do so before generates the checkpoint
    # in case we repeatedly fails the sanity checks.
    self._GarbageCollect()

    _, global_step, prefix = sess.run(
        fetches=[self._save_op, self._save_global_step, self._save_prefix],
        feed_dict={self._logdir_ph: self._logdir})
    prefix = tf.compat.as_text(prefix)

    # Many users expect this as the tf.train.Saver does this by default.
    meta_graph_filename = prefix + ".meta"
    tf.train.export_meta_graph(filename=meta_graph_filename)

    # We can do extra sanity checks.
    self._DoSanityCheck(prefix)

    # Commit new state.
    self._UpdateState(prefix)

    tf.logging.info("Saved %d %s", global_step, prefix)
    return global_step, prefix

  def _UpdateState(self, prefix):
    """Updates the checkpoint state with the new checkpoint prefix."""
    # The checkpoint looks OK. Commit it to the state.
    state = self._GetState()

    # If a previous checkpoint exists and it was generated long
    # after the checkpoint before it, we preserve it in
    # all_model_checkpoint_{paths,timestamps}.
    if state.model_checkpoint_path and (
        not state.all_model_checkpoint_timestamps or
        not self._keep_every_n_hours or
        (state.last_preserved_timestamp -
         state.all_model_checkpoint_timestamps[-1] >
         3600. * self._keep_every_n_hours)):
      state.all_model_checkpoint_paths.append(state.model_checkpoint_path)
      state.all_model_checkpoint_timestamps.append(
          state.last_preserved_timestamp)

    # Record the checkpoint we just generated.
    state.model_checkpoint_path = prefix
    state.last_preserved_timestamp = time.time()

    # Applies the count-based GC policy.
    if self._keep_latest_n:
      if self._keep_latest_n == 1:
        # Use [:] to prevent "Assignment not allowed to repeated field" error.
        state.all_model_checkpoint_paths[:] = []
        state.all_model_checkpoint_timestamps[:] = []
      else:
        n = self._keep_latest_n - 1
        state.all_model_checkpoint_paths[:] = (
            state.all_model_checkpoint_paths[-n:])
        state.all_model_checkpoint_timestamps[:] = (
            state.all_model_checkpoint_timestamps[-n:])
    self._SetState(state)

  def Restore(self, sess, checkpoint_id=None):
    """Restore variables from a checkpoint.

    Args:
      sess: A session with tf.Graph under which this object is constructred.
      checkpoint_id: If None, restore from the latest checkpoint. Otherwise,
        restore from the specific checkpoint.

    Returns:
      If no checkpoint has been generated, returns (0, "").  Otherwise, try to
      restore from the specified or the latest checkpoint. If the checkpoint is
      successfully restored, returns the checkpoint's global step and file
      prefix. Otherwise, raises an error.
    """

    if checkpoint_id:
      prefix = "{}/ckpt-{:08d}".format(self._logdir, checkpoint_id)
    else:
      prefix = self._GetState().model_checkpoint_path
      if not prefix:
        return 0, ""

    sess.run(
        fetches=[self._restore_op], feed_dict={self._restore_prefix_ph: prefix})
    global_step = self._GetCheckpointId(prefix)
    tf.logging.info("Restored %d %s", global_step, prefix)
    return global_step, prefix


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
