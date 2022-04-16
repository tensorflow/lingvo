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
"""A set of objective functions for building quantizers (e.g VQ-VAE)."""

from typing import Optional

import jax
import jax.numpy as jnp
from lingvo.jax import pytypes

JTensor = pytypes.JTensor


def scatter_nd(indices, updates, shape):
  zeros = jnp.zeros(shape, updates.dtype)
  key = tuple(jnp.moveaxis(indices, -1, 0))
  return zeros.at[key].add(updates)


def batch_pplx_entropy_from_codes(codes: JTensor,
                                  num_classes: int,
                                  *,
                                  paddings: Optional[JTensor] = None,
                                  data_parallel_axis: Optional[str] = None):
  """Calculates pplx and entropy from probs within batch.

  Args:
    codes:         [..., num_groups] with values in [0, num_classes).
    num_classes:   A Python int.
    paddings:      [...], 0/1 value tensor.
    data_parallel_axis: If this is set we sum over replicas to get the combined
      statistics

  Returns:
      A tuple of 3 tensors:
      - pplx:      scalar, avg_across_groups(avg(non-padded samples of a group))
      - entropy:   scalar, avg_across_groups(avg(non-padded samples of a group))
      - histogram: [g, c], code word counts.
  """
  rank = len(codes.shape)
  assert rank is not None
  assert rank >= 2

  is_in_pmap = data_parallel_axis is not None

  codes = codes.astype(jnp.int32)
  if paddings is None:
    paddings = jnp.zeros_like(codes[..., 0], dtype=codes.dtype)
  else:
    paddings = paddings.astype(codes.dtype)

  num_groups = codes.shape[-1]
  # [?, g]
  codes = jnp.reshape(codes, [-1, num_groups])
  paddings = jnp.reshape(paddings, [-1, 1])
  paddings = jnp.broadcast_to(paddings, codes.shape)

  # [g]
  indices_offset = jnp.arange(
      start=0, stop=num_groups * num_classes, step=num_classes, dtype=jnp.int32)
  # [?, g]
  indices = codes + indices_offset
  # [? * g, 1]
  indices = jnp.reshape(indices, [-1])[:, jnp.newaxis]

  # [? * g]
  mask = (1.0 - paddings).astype(jnp.float32)
  values = jnp.reshape(mask, [-1])

  # [g * c]
  histogram = scatter_nd(indices, values, [num_groups * num_classes])
  normalizer = jnp.sum(values) / num_groups

  if is_in_pmap:
    histogram = jax.lax.psum(histogram, data_parallel_axis)
    normalizer = jax.lax.psum(normalizer, data_parallel_axis)
  # [g, c]
  histogram = jnp.reshape(histogram, [num_groups, num_classes])

  # [g, c]
  probs = histogram / jnp.maximum(normalizer, 1.0)
  log_probs = jnp.log(jnp.maximum(1.0e-30, probs))
  # [g]
  sum_plogp = jnp.sum(log_probs * probs, -1)
  pplx = jnp.mean(jnp.exp(-sum_plogp))
  entropy = jnp.log(pplx)
  return pplx, entropy, histogram


def batch_codebook_coverage(codes: JTensor,
                            num_classes: int,
                            *,
                            paddings: JTensor,
                            data_parallel_axis: Optional[str] = None):
  """Computes codebook coverage within a batch.

  Args:
    codes:         [..., num_groups], values are in [0, num_classes).
    num_classes:   A Python int.
    paddings:      [...], 0/1 value tensor.
    data_parallel_axis: If set will psum() over the axis

  Returns:
    A scalar tf.Tensor, avg coverage across groups.
  """
  # [num_groups, num_classes]
  _, _, histogram = batch_pplx_entropy_from_codes(
      codes,
      num_classes,
      paddings=paddings,
      data_parallel_axis=data_parallel_axis)
  onehot = jnp.greater(histogram, 0).astype(jnp.float32)
  avg_num_covered_words = jnp.mean(jnp.sum(onehot, -1))
  return avg_num_covered_words / num_classes
