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
"""Wrappers of tf.Variable to achieve various temporary effects.

E.g., such effects include tracking assign ops, and converting between manual
and auto sharding modes in GShard. These wrappers are intended to be used in
limited scopes.
"""

from lingvo import compat as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
# pylint: enable=g-direct-tensorflow-import


class VarWrapperTrackAssign:
  """A wrapper of tf.Variable that tracks assignments."""

  def __init__(self, var):
    self._var = var
    self._assign_ops = []

  def previous_assigns(self):
    return self._assign_ops

  def control_after_assigns(self):
    if not self._assign_ops:
      return tf.no_op()
    with tf.control_dependencies(self._assign_ops):
      return tf.no_op()

  def __getattr__(self, attr):
    if attr.startswith('scatter') or attr.startswith('gather'):
      raise NotImplementedError('%s not implemented in VarWrapperTrackAssign.' %
                                attr)
    return getattr(self._var, attr)

  def __repr__(self):
    return 'VarWrapperTrackAssign(%r)' % self._var.__repr__()

  @property
  def raw_var(self):
    return self._var

  def assign(self, value, use_locking=False, name=None, read_value=True):
    op = self._var.assign(value, use_locking, name, read_value)
    self._assign_ops.append(op)
    return op

  def assign_add(self, delta, use_locking=False, name=None, read_value=True):
    op = self._var.assign_add(delta, use_locking, name, read_value)
    self._assign_ops.append(op)
    return op

  def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
    op = self._var.assign_sub(delta, use_locking, name, read_value)
    self._assign_ops.append(op)
    return op


tf.ops.register_tensor_conversion_function(VarWrapperTrackAssign,
                                           lambda v, *a, **kw: v.value())


class StackedVarWrapperWithManualSharding:
  """A wrapper of tf.Variable for stacked variables in manual-sharding mode.

  The variable is sharded on on the leading (stacking) dimension, and the shard
  size is 1. Example: the physical variable v has shape [N, a, b], which is
  stacked from N logical variables of shape [a, b] and annotated to be sharded
  on dim 0 in N ways::

    - With StackedVarWrapperWithManualSharding(v), read from the wrapper will
    have (manually sharded) shape [a, b].

  This wrapper internally converts between auto and manual sharding modes, which
  makes variable read/write compatible with the rest of manually sharded code.

  If the variable has other dimensions sharded, the auto/manual conversion will
  only happen partially on the stacking dimension only.
  """

  def __init__(self, var):
    self._var = var
    assert not isinstance(var, StackedVarWrapperWithManualSharding)
    self._sharding = xla_sharding.get_op_sharding(var.op)
    if not self._sharding:
      self._sharding = xla_sharding.Sharding.split(
          var, split_dimension=0,
          num_devices=var.shape[0]).proto.SerializeToString()
      self._maybe_partial_manual = False
    else:
      self._maybe_partial_manual = True

  def __getattr__(self, attr):
    if attr.startswith('scatter') or attr.startswith('gather'):
      raise NotImplementedError(
          '%s not implemented in StackedVarWrapperWithManualSharding.' % attr)
    return getattr(self._var, attr)

  def __repr__(self):
    return 'StackedVarWrapperWithManualSharding(%r)' % self._var.__repr__()

  @property
  def raw_var(self):
    return self._var

  def _to_manual(self, val):
    if self._maybe_partial_manual:
      return xla_sharding.auto_to_manual_spmd_partition(
          val, self._sharding, single_dim=0)
    else:
      # Do not use single_dim if not necessary. This is to avoid problems with
      # older TF versions.
      return xla_sharding.auto_to_manual_spmd_partition(val, self._sharding)

  def _to_auto(self, val):
    if self._maybe_partial_manual:
      return xla_sharding.manual_to_auto_spmd_partition(
          val, self._sharding, self._var.shape, single_dim=0)
    else:
      return xla_sharding.manual_to_auto_spmd_partition(val, self._sharding,
                                                        self._var.shape)

  def value(self):
    """Returns the variable and converts it to manually sharded mode.

    Returns:
      The return value has the shape of the individual elements of the stacked
      variable (shard shape with the stacking dimension collapsed).
    """
    val = self._var.value()
    val = self._to_manual(val)
    return tf.squeeze(val, 0)

  def read_value(self):
    """Reads the variable and converts it to manually sharded mode.

    Returns:
      The return value has the shape of the individual elements of the stacked
      variable (shard shape with the stacking dimension collapsed).
    """
    val = self._var.read_value()
    val = self._to_manual(val)
    return tf.squeeze(val, 0)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    """Implements the interface of tf.Variable.assign.

    Args:
      value: A manually sharded tensor that has the shape of the individual
        elements of the stacked variable (shard shape with the stacking
        dimension collapsed).
      use_locking: See tf.Variable.assign.
      name: See tf.Variable.assign.
      read_value: See tf.Variable.assign. If True, the returned value will be
        manually sharded.

    Returns:
      See tf.Variable.assign. If read_value is True, returns the updated value
      in the shard shape of the shape of the individual elements of the stacked
      variable (shard shape with the stacking dimension collapsed).
    """
    value = tf.expand_dims(value, 0)
    value = self._to_auto(value)
    res = self._var.assign(value, use_locking, name, read_value)
    if read_value:
      res = self._to_manual(res)
      res = tf.squeeze(res, 0)
    return res

  def assign_add(self, delta, use_locking=False, name=None, read_value=True):
    """Implements the interface of tf.Variable.assign_add."""
    delta = tf.expand_dims(delta, 0)
    delta = self._to_auto(delta)
    res = self._var.assign_add(delta, use_locking, name, read_value)
    if read_value:
      res = self._to_manual(res)
      res = tf.squeeze(res, 0)
    return res

  def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
    """Implements the interface of tf.Variable.assign_sub."""
    delta = tf.expand_dims(delta, 0)
    delta = self._to_auto(delta)
    res = self._var.assign_sub(delta, use_locking, name, read_value)
    if read_value:
      res = self._to_manual(res)
      res = tf.squeeze(res, 0)
    return res


tf.ops.register_tensor_conversion_function(StackedVarWrapperWithManualSharding,
                                           lambda v, *a, **kw: v.value())
