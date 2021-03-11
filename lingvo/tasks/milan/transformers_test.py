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
"""Tests for transformers."""

from lingvo import compat as tf
from lingvo.core import test_utils

from lingvo.tasks.milan import transformers


class TransformersTest(test_utils.TestCase):

  def testStackWithEmbeddingInput(self):
    input_dim = 7
    output_dim = 16
    p = transformers.GetTransformerStackWithEmbeddingInput(
        input_dim=input_dim,
        num_layers=3,
        hidden_dim=32,
        num_attention_heads=4,
        output_dim=output_dim,
        name='embedding_transformer_stack')
    layer = p.Instantiate()

    batch_size = 3
    input_features = tf.zeros([batch_size, 11, input_dim])
    input_lengths = tf.constant([4, 11, 1])
    outputs = layer(input_features, input_lengths)
    self.assertEqual(outputs.shape, (batch_size, output_dim))


if __name__ == '__main__':
  tf.test.main()
