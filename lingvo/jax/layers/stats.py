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
"""Compute stats of tensors mostly for monitoring purposes."""

from typing import Optional

from jax import numpy as jnp
from lingvo.jax import py_utils
from lingvo.jax import pytypes

JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap


def compute_stats(inputs: JTensor,
                  padding: Optional[JTensor] = None) -> NestedMap:
  """Computes various stats over the valid data points in inputs."""
  # Let's compute stats in fp32
  inputs = inputs.astype(jnp.float32)
  if padding is None:
    padding = jnp.zeros_like(inputs)
  assert inputs.ndim == padding.ndim
  mask = 1.0 - padding

  sum_v = jnp.sum(inputs * mask)
  count_v = jnp.sum(jnp.ones_like(inputs) * mask)
  mean_v = sum_v / jnp.maximum(1.0, count_v)
  sum_v_squared = jnp.sum(jnp.square((inputs - mean_v) * mask))
  std_v = jnp.sqrt(sum_v_squared / jnp.maximum(1.0, count_v))
  max_v = jnp.max(jnp.abs(inputs * mask))

  return NestedMap(mean_v=mean_v, std_v=std_v, max_v=max_v)
