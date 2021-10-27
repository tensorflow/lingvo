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
"""Utils for partitioning."""

from typing import Tuple

from absl import logging
import jax
import numpy as np

_TPU_V3 = 'TPU v3'
_TPU_V4 = 'TPU v4'


# TODO(zhangqiaorjc): Generalize to other topologies.
def CreateDeviceMesh(mesh_shape: Tuple[int, int, int]) -> np.ndarray:
  """Creates a device mesh."""

  def MaybeReorderMesh(mesh_shape: Tuple[int, int, int],
                       device_mesh: np.ndarray) -> np.ndarray:
    """Possibly re-orders device mesh for better performance."""
    device_kind = jax.devices()[0].device_kind
    logging.info('device_kind: "%s", mesh_shape: %s', device_kind, mesh_shape)

    if device_kind == _TPU_V3 and mesh_shape[-1] == 8:
      logging.info('Re-order DF device mesh for better performance.')
      perm = np.array([0, 1, 2, 3, 6, 7, 4, 5])
      device_mesh = device_mesh[:, :, perm]
    elif device_kind == _TPU_V4 and mesh_shape == (1, 16, 4):
      logging.info('Re-order PF device mesh for better performance.')
      perm = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
      device_mesh = device_mesh.reshape((4, 16))[:, perm].reshape(1, 16, 4)
    elif device_kind == _TPU_V4 and mesh_shape == (1, 32, 8):
      logging.info('Re-order PF device mesh for better performance.')
      perm = np.array([0, 1, 4, 5, 2, 3, 6, 7])
      # host-untiling
      device_mesh = device_mesh.reshape((-1, 8))[:, perm]
      # x (minor), z(major) for 32, and y for 8
      device_mesh = device_mesh.reshape((8, 8, 4)).transpose(
          (0, 2, 1)).reshape(mesh_shape)
    elif device_kind == _TPU_V4 and mesh_shape == (1, 64, 8):
      logging.info('Re-order PF device mesh for better performance.')
      perm = np.array([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15])
      device_mesh = device_mesh.reshape((-1, 16))[:, perm].reshape(1, 64, 8)
    elif device_kind == _TPU_V4 and mesh_shape == (1, 64, 16):
      logging.info('Re-order PF device mesh for better performance.')
      device_mesh = device_mesh.reshape((16, 64, 1)).transpose()
    return device_mesh

  device_mesh = np.asarray(jax.devices()).reshape(mesh_shape)
  return MaybeReorderMesh(mesh_shape, device_mesh)
