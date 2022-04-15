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
"""Implementation of combination functions for dual-encoder models."""

from lingvo import compat as tf
from lingvo.core import base_layer


class DotProductScoreFunction(base_layer.BaseLayer):
  """Performs dot product combination between two encoded vectors."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'dot_product_score_function'
    return p

  def FProp(self, theta, x, y):
    """Computes pair-wise dot product similarity.

    Args:
      theta: NestedMap of variables belonging to this layer and its children.
      x: batch of encoded representations from modality x. A float32 Tensor of
        shape [x_batch_size, encoded_dim]
      y: batch of encoded representations from modality y. A float32 Tensor of
        shape [y_batch_size, encoded_dim]

    Returns:
      Pairwise dot products. A float32 Tensor with shape
      `[x_batch_size, y_batch_size]`.
    """

    return tf.matmul(x, y, transpose_b=True)
