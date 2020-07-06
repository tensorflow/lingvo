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
"""Tests for lingvo.core.batch_utils."""

import itertools

from absl.testing import flagsaver
from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import batch_utils
from lingvo.core import cluster_factory


class BatchUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters({'use_per_host_infeed': False},
                            {'use_per_host_infeed': True})
  def testScaleInfeedToGlobalCPU(self, use_per_host_infeed):
    with cluster_factory.ForTestingWorker(cpus=128):
      self.assertEqual(
          batch_utils.scale_infeed_to_global(1024, use_per_host_infeed), 1024)

  @parameterized.parameters({'use_per_host_infeed': False},
                            {'use_per_host_infeed': True})
  def testScaleInfeedToGlobalGPU(self, use_per_host_infeed):
    with cluster_factory.ForTestingWorker(gpus=128):
      self.assertEqual(
          batch_utils.scale_infeed_to_global(1024, use_per_host_infeed), 1024)

  @parameterized.parameters(
      itertools.product(
          (False, True),  # use_per_host_infeed
          (1, 4)))  # num_tpu_hosts
  def testScaleInfeedToGlobalTPU(self, use_per_host_infeed, num_tpu_hosts):
    with flagsaver.flagsaver(xla_device='tpu', enable_asserts=False):
      with cluster_factory.ForTestingWorker(
          tpus=128, num_tpu_hosts=num_tpu_hosts):
        num_infeeds = num_tpu_hosts if use_per_host_infeed else 1
        self.assertEqual(
            batch_utils.scale_infeed_to_global(1024, use_per_host_infeed),
            1024 * num_infeeds)

  @parameterized.parameters(
      itertools.product(
          (False, True),  # use_per_host_infeed
          (1, 8)))  # split_size
  def testScaleSplitToInfeedCPU(self, use_per_host_infeed, split_size):
    with cluster_factory.ForTestingWorker(
        cpus=128, split_size=split_size) as cluster:
      num_splits = 128 // split_size
      self.assertEqual(cluster.num_splits_per_client, num_splits)
      self.assertEqual(
          batch_utils.scale_split_to_infeed(1024, use_per_host_infeed),
          1024 * num_splits)

  @parameterized.parameters(
      itertools.product(
          (False, True),  # use_per_host_infeed
          (1, 8)))  # split_size
  def testScaleSplitToInfeedGPU(self, use_per_host_infeed, split_size):
    with cluster_factory.ForTestingWorker(
        gpus=128, split_size=split_size) as cluster:
      num_splits = 128 // split_size
      self.assertEqual(cluster.num_splits_per_client, num_splits)
      self.assertEqual(
          batch_utils.scale_split_to_infeed(1024, use_per_host_infeed),
          1024 * num_splits)

  @parameterized.parameters(
      itertools.product(
          (False, True),  # use_per_host_infeed
          (1, 8),  # split_size
          (1, 4)))  # num_tpu_hosts
  def testScaleSplitToInfeedTPU(self, use_per_host_infeed, split_size,
                                num_tpu_hosts):
    with cluster_factory.ForTestingWorker(
        tpus=128, split_size=split_size,
        num_tpu_hosts=num_tpu_hosts) as cluster:
      num_splits = 128 // split_size
      num_infeeds = num_tpu_hosts if use_per_host_infeed else 1
      self.assertEqual(cluster.num_splits_per_client, num_splits)
      self.assertEqual(
          batch_utils.scale_split_to_infeed(1024, use_per_host_infeed),
          1024 * num_splits // num_infeeds)


if __name__ == '__main__':
  tf.test.main()
