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

from typing import Dict, List, Optional, Sequence

from lingvo import compat as tf
from lingvo.core import py_utils_flags
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla import xla_data_pb2
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
    use_sharding_op: If true, adds a sharding op to set the sharding: tensor =
      gen_xla_ops.xla_sharding(tensor)

      hyouklee@: use_sharding_op=False "It adds the sharding attribute to the op
        itself. The outcome is that, that information could be lost by TF graph
        transformations. Also, directly attaching the sharding annotation to the
        op caused some compilation failures in the past (due to incompatible
        shardings), so the plan is to make use_sharding_op to be the default."
        "The only case I would set it to False today is when annotating weights.
        Weight annotation does some special handling, so there may be some
        changes needed in that logic if we add separate sharding op."

    input_shape: The shape of the original tensor.

  Returns:
    Tensor conditionally annotated with sharding.
  """
  if not py_utils_flags.use_tpu() or num_devices is None or not num_devices > 1:
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
  if (not py_utils_flags.use_tpu() or tensor_split_dims_mapping is None or
      device_mesh is None or device_mesh.size <= 1):
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


def ReshapeDim(x, dim, dim_reshape_segments=None):
  """Reshapes tensor x according to dim_reshape_segments.

  Args:
    x: A input Tensor of shape [..., x.shape[dim], ...].
    dim: The dim that needs to be reshaped.
    dim_reshape_segments: The leading dim size of the reshaped dims.

  Returns:
    A Tensor of shape [..., dim_reshape_segments,
    x.shape[dim] // dim_reshape_segments, ...].
  """
  if dim_reshape_segments is None:
    return x
  assert x.shape[dim] % dim_reshape_segments == 0
  new_shape = list(x.shape[0:dim])
  new_shape.append(dim_reshape_segments)
  new_shape.append(x.shape[dim] // dim_reshape_segments)
  new_shape.extend(d for d in x.shape[dim + 1:])
  return tf.reshape(x, new_shape)


class TensorShardingSpec:
  """Represents a sharding annotation for GShard/XLA."""

  def __init__(self,
               split_dims_mapping: Optional[List[int]] = None,
               device_mesh: Optional[np.ndarray] = None,
               uneven_padding: Optional[List[int]] = None):
    """Creates a sharding specification.

    Args:
      split_dims_mapping: a list of integers that map each tensor axis to the
        device mesh axis along which it is sharded. Its length is the tensor
        rank, and split_dims_mapping[i] is device mesh axis for tensor dimension
        i. Use -1 for tensor dimensions that are not sharded. If the list is set
        to None, the sharding will be treated as replicated.
      device_mesh: a numpy.ndarray describing the topology of the device mesh
        and each element is the ID of the device in the topology. Not needed for
        replicated sharding, where it can be set to None.
      uneven_padding: amount of padding applied to the right side of each tensor
        dimension due to uneven partitioning of the shape in SPMD.
    """
    self._split_dims_mapping: Optional[List[int]] = split_dims_mapping
    self._device_mesh: Optional[np.ndarray] = device_mesh
    self._uneven_padding = uneven_padding

  @classmethod
  def FromFullShape(cls, full_shape: Sequence[int],
                    split_dims_mapping: List[int], device_mesh: np.ndarray):
    """Creates tiled sharding spec with uneven padding computed from shape."""
    uneven_padding = [0] * len(split_dims_mapping)
    for i in range(len(split_dims_mapping)):
      if split_dims_mapping[i] >= 0:
        partitions = device_mesh.shape[split_dims_mapping[i]]
        shard_size = (full_shape[i] + partitions - 1) // partitions
        uneven_padding[i] = shard_size * partitions - full_shape[i]
    return TensorShardingSpec(split_dims_mapping, device_mesh, uneven_padding)

  def ApplyToTensor(self,
                    tensor: tf.Tensor,
                    use_sharding_op: bool = True) -> tf.Tensor:
    if self.is_replicated:
      return xla_sharding.replicate(tensor, use_sharding_op=use_sharding_op)
    return xla_sharding.mesh_split(
        tensor,
        self.device_mesh,
        self.split_dims_mapping,
        use_sharding_op=use_sharding_op)

  def ApplyToVariable(self, variable: tf.Variable) -> tf.Variable:
    if self.is_replicated:
      return xla_sharding.replicate(variable, use_sharding_op=False)
    return xla_sharding.mesh_split(
        variable,
        self.device_mesh,
        self.split_dims_mapping,
        use_sharding_op=False)

  def ShardShape(self, full_shape: Sequence[int]) -> Sequence[int]:
    """Returns the shape after applying this sharding to full_shape."""
    if self.is_replicated:
      return full_shape

    shard_shape = list(full_shape)
    for i in range(len(self._split_dims_mapping)):
      if self._split_dims_mapping[i] >= 0:
        partitions = self._device_mesh.shape[self._split_dims_mapping[i]]
        shard_shape[i] = (full_shape[i] + partitions - 1) // partitions
    return shard_shape

  def ManualToAutoPartitioning(self, tensor: tf.Tensor) -> tf.Tensor:
    """Converts manually sharded tensor to full-size for auto partitioning."""
    full_shape = list(tensor.shape)
    if not self.is_replicated:
      for i in range(len(self._split_dims_mapping)):
        if self._split_dims_mapping[i] >= 0:
          full_shape[i] *= self._device_mesh.shape[self._split_dims_mapping[i]]
        if self._uneven_padding is not None and self._uneven_padding[i] > 0:
          full_shape[i] -= self._uneven_padding[i]
    return xla_sharding.manual_to_auto_spmd_partition(
        tensor,
        self.ToXlaOpSharding().SerializeToString(), full_shape)

  def AutoToManualPartitioning(self, tensor: tf.Tensor) -> tf.Tensor:
    """Converts full-size tensor (auto partitioning) to manually sharded."""
    manual = xla_sharding.auto_to_manual_spmd_partition(
        tensor,
        self.ToXlaOpSharding().SerializeToString())
    xla_sharding.Sharding.manual().apply_to_tensor(manual)
    return manual

  def ToXlaOpSharding(self) -> xla_data_pb2.OpSharding:
    if self.is_replicated:
      return xla_sharding.Sharding.replicate().proto
    return xla_sharding.mesh_split_sharding(self.device_mesh,
                                            self.split_dims_mapping).proto

  @classmethod
  def FromXlaOpSharding(
      cls, op_sharding_proto: xla_data_pb2.OpSharding) -> 'TensorShardingSpec':
    """Parses from an XLA OpSharding proto."""
    if op_sharding_proto.type == xla_data_pb2.OpSharding.OTHER:
      device_mesh_shape = op_sharding_proto.tile_assignment_dimensions
      device_mesh = np.reshape(
          np.array(op_sharding_proto.tile_assignment_devices),
          device_mesh_shape)
      if op_sharding_proto.replicate_on_last_tile_dim:
        split_dims_mapping = list(range(len(device_mesh_shape) - 1))
      else:
        split_dims_mapping = list(range(len(device_mesh_shape)))
      return cls(split_dims_mapping, device_mesh)
    else:
      return cls.ReplicatedSpec()

  def AddLeadingDims(self, num_dims: int = 1) -> 'TensorShardingSpec':
    if self.is_replicated:
      return self
    new_padding = (None if self._uneven_padding is None else [0] * num_dims +
                   self._uneven_padding)
    return TensorShardingSpec([-1] * num_dims + self._split_dims_mapping,
                              self.device_mesh, new_padding)

  def RemoveLeadingDims(self, num_dims: int = 1) -> 'TensorShardingSpec':
    if self.is_replicated:
      return self
    new_padding = (None if self._uneven_padding is None else
                   self._uneven_padding[num_dims:])
    return TensorShardingSpec(self._split_dims_mapping[num_dims:],
                              self.device_mesh, new_padding)

  def RemoveDim(self, dim) -> 'TensorShardingSpec':
    """Returns a copy of self with dimension 'dim' removed."""
    if self.is_replicated:
      return self
    if dim < 0:
      num_dims = len(self._split_dims_mapping)
      dim = num_dims + dim
    assert dim >= 0 and dim < len(self._split_dims_mapping)
    new_padding = (None if self._uneven_padding is None else
                   self._uneven_padding[:dim] + self._uneven_padding[dim + 1:])
    split_dims_mapping = (
        self._split_dims_mapping[:dim] + self._split_dims_mapping[dim + 1:])
    return TensorShardingSpec(split_dims_mapping, self.device_mesh, new_padding)

  @classmethod
  def ReplicatedSpec(cls):
    return TensorShardingSpec()

  @property
  def split_dims_mapping(self) -> Optional[List[int]]:
    return self._split_dims_mapping

  @property
  def device_mesh(self) -> Optional[np.ndarray]:
    return self._device_mesh

  @property
  def is_replicated(self) -> bool:
    if self.device_mesh is None or self.split_dims_mapping is None:
      return True
    for mesh_dim in self.split_dims_mapping:
      if mesh_dim >= 0 and self.device_mesh.shape[mesh_dim] > 1:
        return False
    return True

  @property
  def mesh_dim_to_tensor_dim_mapping(self) -> Dict[int, int]:
    mapping = {}
    if self.is_replicated:
      return mapping
    for i in range(len(self.split_dims_mapping)):
      if self.split_dims_mapping[i] >= 0:
        mapping[self.split_dims_mapping[i]] = i
    return mapping

  @property
  def uneven_padding(self) -> Optional[List[int]]:
    return self._uneven_padding


def GetVarSharding(var: tf.Variable) -> TensorShardingSpec:
  """Returns the sharding directly attached to a variable."""
  sharding = xla_sharding.get_op_sharding(var.op)
  if not sharding:
    return TensorShardingSpec.ReplicatedSpec()

  proto = xla_data_pb2.OpSharding()
  proto.ParseFromString(sharding)
  spec_without_padding = TensorShardingSpec.FromXlaOpSharding(proto)
  # Consider uneven padding.
  return TensorShardingSpec.FromFullShape(
      [int(d) for d in var.shape], spec_without_padding.split_dims_mapping,
      spec_without_padding.device_mesh)
