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
"""Tests for routing_config_helper."""

from lingvo import compat as tf
from lingvo.core import routing_config_helper
from lingvo.core import test_utils


class RoutingTransformerEncoderParamsTest(test_utils.TestCase):
  """Test for generating routing params."""

  def testOverride(self):
    routing_params = routing_config_helper.RoutingTransformerEncoderParams(
        seq_len=128)
    self.assertEqual(routing_params.block_size, 32)
    self.assertEqual(routing_params.left_context, 33)
    self.assertEqual(routing_params.right_context, 32)
    self.assertEqual(routing_params.attention_window, 64)
    self.assertEqual(routing_params.num_clusters, 2)
    self.assertEqual(routing_params.num_routing_layers, 1)
    self.assertEqual(routing_params.num_routing_heads, 1)
    routing_params.block_size = 64
    routing_params.num_clusters = 4
    routing_params.attention_window = 32
    self.assertEqual(routing_params.block_size, 64)
    self.assertEqual(routing_params.num_clusters, 4)
    self.assertEqual(routing_params.attention_window, 32)


if __name__ == '__main__':
  tf.test.main()
