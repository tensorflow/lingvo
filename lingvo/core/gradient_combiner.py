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
"""Layers to combine gradients computed from multiple losses.

Multi-task learning can sometimes benefit from more sophisticated gradient
combination algorithms than simple linear aggregation, for example

* Gradient surgery: https://arxiv.org/abs/2001.06782
* GradNorm: https://arxiv.org/abs/1711.02257
"""

from lingvo.core import base_layer


class GradientCombiner(base_layer.BaseLayer):
  """An abstract class to combine gradients (from multiple loss functions)."""

  def Combine(self, vmap, losses_and_gradients):
    """Combines gradients on the variables.

    Args:
      vmap: a NestedMap containing the variables.
      losses_and_gradients: a Dict[str, loss_and_grads], where each key
        represents the loss name used to compute the gradients and each value is
        a NestedMap with the following entries, 'loss_metric', a (loss, weight)
        pair representing the loss; 'grads', a NestedMap containing the gradient
        tensors for variables, with the identical structure as 'vmap'.

    Returns:
      A NestedMap of combined gradients, with an identical structure as 'vmap'.
    """
    raise NotImplementedError(type(self))
