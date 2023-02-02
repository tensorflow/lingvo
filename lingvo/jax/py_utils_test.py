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
"""Tests for Python utils."""

import collections
from typing import Any

from absl.testing import absltest
from flax import struct
import jax
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import test_utils
from lingvo.jax import train_states
import numpy as np
import tensorflow.compat.v2 as tf


class PyUtilsTest(test_utils.TestCase):

  def test_reshard_empty_array(self):
    batch_size = 128
    empty_inputs = tf.ones(shape=(batch_size, 0))
    sharded_inputs = py_utils.reshard(empty_inputs)
    # Check the shape of returned inputs.
    num_devices = jax.local_device_count()
    self.assertEqual(sharded_inputs.shape,
                     (num_devices, batch_size // num_devices, 0))

  def test_extract_prefixed_keys_from_state_specs(self):
    w_sepc = base_layer.var_partition_specs(
        {'w': py_utils.weight_params(shape=(4, 8))},
        device_mesh=np.arange(1).reshape([1, 1]),
        device_axis_names=['a', 'b'])
    train_state_partition_specs = train_states.TrainState(
        step=jax.sharding.PartitionSpec(), mdl_vars=w_sepc, opt_states={})
    nested_names = py_utils.extract_prefixed_keys_from_nested_map(
        train_state_partition_specs)
    flattened_names, _ = jax.tree_util.tree_flatten(nested_names)
    self.assertListEqual(['step', 'mdl_vars/w'], flattened_names)

  def test_extract_prefixed_keys_from_nested_map(self):
    Point = collections.namedtuple('Point', ['x', 'y'])

    inputs = {'a': [1, 2, Point(x=3, y=4), (5, 6)], 'b': ('c', 'd')}
    outputs = py_utils.extract_prefixed_keys_from_nested_map(inputs)
    self.assertEqual(
        {
            'a': [
                'a[0]', 'a[1]',
                Point(x='a[2]/x', y='a[2]/y'), ('a[3][0]', 'a[3][1]')
            ],
            'b': ('b[0]', 'b[1]')
        }, outputs)

  def test_extract_prefixed_keys_from_dataclass(self):

    @struct.dataclass
    class GlobalShardedParameterStats:
      statistics: np.ndarray  # Statistics
      preconditioners: np.ndarray  # Preconditioners
      exponents: np.ndarray  # exponents
      index_start: int = struct.field(pytree_node=False)
      sizes: Any = struct.field(pytree_node=False)

    stats0 = GlobalShardedParameterStats(
        statistics=np.array([0], dtype=np.float32),
        preconditioners=np.array([1, 1], dtype=np.float32),
        exponents=np.array([2, 2, 2], dtype=np.float32),
        index_start=0,
        sizes=0,
    )
    # Even though the `preconditioners` is first here, the order is decided
    # by the order in `GlobalShardedParameterStats` class.
    stats1 = GlobalShardedParameterStats(
        preconditioners=np.array([5, 5], dtype=np.float32),
        statistics=np.array([4], dtype=np.float32),
        exponents=np.array([6, 6, 6], dtype=np.float32),
        index_start=1,
        sizes=1,
    )

    nested_data = py_utils.NestedMap(stats0=stats0, stats1=stats1)
    nested_names = py_utils.extract_prefixed_keys_from_nested_map(nested_data)
    flattened_nested_names, _ = jax.tree_util.tree_flatten(nested_names)

    self.assertListEqual([
        'stats0/statistics', 'stats0/preconditioners', 'stats0/exponents',
        'stats1/statistics', 'stats1/preconditioners', 'stats1/exponents'
    ], flattened_nested_names)

  def test_sync_global_devices(self):
    py_utils.sync_global_devices('sync')


if __name__ == '__main__':
  absltest.main()
