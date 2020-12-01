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
# ==============================================================================
"""Utilities for applying xla-sharding to a model."""

from lingvo import compat as tf
from lingvo.core import py_utils
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
# pylint: enable=g-direct-tensorflow-import


def Split(x,
          split_dimension,
          num_devices,
          use_sharding_op=True,
          input_shape=None):
  """Wrapper for xla_sharding.split.

  Args:
    x: Tensor to annotate.
    split_dimension: xla_sharding.split arg.
    num_devices: xla_sharding.split arg.
    use_sharding_op: If true, adds a sharding op to set the sharding:
      tensor = gen_xla_ops.xla_sharding(tensor)

      hyouklee@: use_sharding_op=False
        "It adds the sharding attribute to the op itself. The outcome is that,
        that information could be lost by TF graph transformations. Also,
        directly attaching the sharding annotation to the op caused some
        compilation failures in the past (due to incompatible shardings), so the
        plan is to make use_sharding_op to be the default."

        "The only case I would set it to False today is when annotating weights.
        Weight annotation does some special handling, so there may be some
        changes needed in that logic if we add separate sharding op."
    input_shape: The shape of the original tensor.

  Returns:
    Tensor conditionally annotated with sharding.
  """
  if not py_utils.use_tpu() or num_devices is None or not num_devices > 1:
    return x
  return xla_sharding.split(
      x,
      split_dimension,
      num_devices,
      input_shape=input_shape,
      use_sharding_op=use_sharding_op,
  )


def MeshSplit(x, device_mesh, tensor_split_dims_mapping, use_sharding_op=True):
  """Wrapper of xla_sharding.mesh_split()."""
  if (not py_utils.use_tpu() or tensor_split_dims_mapping is None or
      device_mesh.size <= 1):
    return x
  num_tiles = np.prod(
      [device_mesh.shape[i] for i in tensor_split_dims_mapping if i >= 0])
  if num_tiles <= 1:
    return x
  return xla_sharding.mesh_split(
      x,
      device_mesh,
      tensor_split_dims_mapping,
      use_sharding_op=use_sharding_op)


def ZigzagOrderOnDeviceMesh(device_mesh, zigzag_mesh_dim):
  """Permutes device_mesh to form zigzag order along zigzag_mesh_dim."""
  # Where there is no wrap-around links along one edge, we might
  # reduce all-reduce latency along that edge by permuting the device order:
  # instead of
  #   0 - 1 - 2 - 3 - 4 - 5 - 6 - 7
  #   |                           |
  #   +---------------------------+
  # it will be
  #   +-------+-------+-------+
  #   |       |       |       |
  #   0 - 7   1   6   2   5   3 - 4
  #       |       |       |       |
  #       +-------+-------+-------+
  xpose_dims = list(range(len(device_mesh.shape)))
  xpose_dims[0] = zigzag_mesh_dim
  xpose_dims[zigzag_mesh_dim] = 0
  device_mesh = np.transpose(device_mesh, xpose_dims)
  permuted_mesh = np.copy(device_mesh)
  for i in range(device_mesh.shape[0]):
    zigzag_i = i * 2 if i * 2 < device_mesh.shape[0] else (
        device_mesh.shape[0] - i) * 2 - 1
    permuted_mesh[i, ...] = device_mesh[zigzag_i, ...]
  return np.transpose(permuted_mesh, xpose_dims)


def GetNonPod2dMesh(device_mesh_shape, physical_mesh_shape):
  """Returns a 2D device mesh on slices smaller than a pod."""
  assert len(device_mesh_shape) == 2
  assert len(physical_mesh_shape) == 3
  if device_mesh_shape[1] != physical_mesh_shape[1] * physical_mesh_shape[2]:
    tf.logging.warning(
        'This only works when device_mesh_shape == [physical_mesh_shape[0], '
        ' physical_mesh_shape[1] * physical_mesh_shape[2]]. '
        'If device_mesh_shape is [32, 16] where physical_mesh_shape is '
        ' [16, 16, 2]. we can transpose the result of this function '
        'GetNonPod2dMesh([16, 32], [16, 16, 2]).')
  # Form a ring on inner mesh dim.
  device_mesh = np.reshape(
      np.arange(0, np.product(device_mesh_shape)), physical_mesh_shape)
  device_mesh = np.transpose(device_mesh, [0, 2, 1])
  device_mesh[:, 1, :] = device_mesh[:, 1, ::-1]
  # Next line: reshape back to mesh shape
  device_mesh = np.reshape(device_mesh, device_mesh_shape)
  # Next line: zigzag on outer mesh dim (8). It doesn't have wrap link, either.
  device_mesh = ZigzagOrderOnDeviceMesh(device_mesh, zigzag_mesh_dim=0)
  return device_mesh
