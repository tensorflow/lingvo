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
"""Utility functions for computing metrics."""

from typing import Optional

import jax
from jax import numpy as jnp
from lingvo.jax import py_utils
from lingvo.jax import pytypes

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor


def top_k_accuracy(top_k: int,
                   logits: JTensor,
                   label_ids: Optional[JTensor] = None,
                   label_probs: Optional[JTensor] = None,
                   weights: Optional[JTensor] = None) -> JTensor:
  """Computes the top-k accuracy given the logits and labels.

  Args:
    top_k: An int scalar, specifying the value of top-k.
    logits: A [..., C] float tensor corresponding to the logits.
    label_ids: A [...] int vector corresponding to the class labels. One of
      label_ids and label_probs should be presented.
    label_probs: A [..., C] float vector corresponding to the class
      probabilites. Must be presented if label_ids is None.
    weights: A [...] float vector corresponding to the weight to assign to each
      example.

  Returns:
    The top-k accuracy represented as a `JTensor`.

  Raises:
    ValueError if neither `label_ids` nor `label_probs` are provided.
  """
  if label_ids is None and label_probs is None:
    raise ValueError("One of label_ids and label_probs should be given.")
  if label_ids is None:
    label_ids = jnp.argmax(label_probs, axis=-1)

  values, _ = jax.lax.top_k(logits, k=top_k)
  threshold = jnp.min(values, axis=-1)

  # Reshape logits to [-1, C].
  logits_reshaped = jnp.reshape(logits, [-1, logits.shape[-1]])

  # Reshape label_ids to [-1, 1].
  label_ids_reshaped = jnp.reshape(label_ids, [-1, 1])
  logits_slice = jnp.take_along_axis(
      logits_reshaped, label_ids_reshaped, axis=-1)[..., 0]

  # Reshape logits_slice back to original shape to be compatible with weights.
  logits_slice = jnp.reshape(logits_slice, label_ids.shape)
  correct = jnp.greater_equal(logits_slice, threshold)
  correct_sum = jnp.sum(correct * weights)
  all_sum = jnp.maximum(1.0, jnp.sum(weights))
  return correct_sum / all_sum
