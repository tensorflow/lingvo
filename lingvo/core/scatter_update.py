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
"""Utilities for scatter updates."""

import contextlib
import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import thread_local_utils

_global_inplace_update_stack = thread_local_utils.ThreadLocalStack()


@contextlib.contextmanager
def SetInplaceUpdate(inplace_update):
  _global_inplace_update_stack.stack.append(inplace_update)
  try:
    yield
  finally:
    _global_inplace_update_stack.stack.pop()


def UseInplaceUpdate():
  if not _global_inplace_update_stack.stack:
    # TODO(rpang): set the default value to False in a follow-up CL.
    return True
  return _global_inplace_update_stack.stack[-1]


def Update(x, i, v, *, inplace_update=None):
  """Performs scatter update: x[i] = v.

  A drop-in replacement for inplace_ops.alias_inplace_update (
  aka tf.InplaceUpdate).

  Args:
    x: the source tensor.
    i: the index tensor. If None, do x = v. If a scalar, do x[i, ...] = v. If a
      vector, do x[j, ...] = v[j, ...] for j in i.
    v: the update value tensor.
    inplace_update: whether to perform inplace updates. If None, follows the
      current context set by SetInplaceUpdate.

  Returns:
    The updated tensor.
  """
  if inplace_update is None:
    inplace_update = UseInplaceUpdate()
  if inplace_update:
    return tf.InplaceUpdate(x, i, v)
  if i is None:
    return py_utils.HasShape(v, tf.shape(x))
  i = tf.convert_to_tensor(i)
  assert i.shape, i
  assert i.shape.rank in (0, 1), i
  if i.shape.rank == 0:
    y = tf.concat([x[:i, ...], v[None, ...], x[i + 1:, ...]], axis=0)
    y.set_shape(x.shape)
    return y
  return tf.tensor_scatter_nd_update(x, i[:, None], v)
