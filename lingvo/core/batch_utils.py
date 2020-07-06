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
"""Utilities to calculate batch sizes."""

from lingvo.core import cluster_factory
from lingvo.core import py_utils


def scale_infeed_to_global(infeed_batch_size, use_per_host_infeed):
  """Obtains a global batch size from an infeed batch size and cluster configs.

  Args:
    infeed_batch_size: int: Per-infeed batch size.
    use_per_host_infeed: bool: Whether to use an individual infeed for each
      host.

  Returns:
    int: Global batch size.
  """
  cluster = cluster_factory.Current()
  if use_per_host_infeed and cluster.num_tpu_hosts > 0:
    if not py_utils.use_tpu():
      raise ValueError('Scaling to TPU hosts without TPUs. {}'.format(
          cluster.num_tpu_hosts))
    return infeed_batch_size * cluster.num_tpu_hosts
  else:
    return infeed_batch_size


def scale_split_to_infeed(split_batch_size, use_per_host_infeed):
  """Obtains an infeed batch size from a split batch size and cluster configs.

  Args:
    split_batch_size: int: Per-split batch size.
    use_per_host_infeed: bool: Whether to use an individual infeed for each
      host.

  Returns:
    int: Per-infeed batch size.
  """
  cluster = cluster_factory.Current()
  global_batch_size = split_batch_size * cluster.num_splits_per_client
  # If use_per_host_infeed, each input op is only responsible
  # for generating a subset of the whole batch.
  if use_per_host_infeed and cluster.num_tpu_hosts > 0:
    return global_batch_size // cluster.num_tpu_hosts
  else:
    return global_batch_size
