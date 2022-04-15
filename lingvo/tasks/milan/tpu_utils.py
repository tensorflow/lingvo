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
"""TPU-specific utilities."""

import functools

from absl import logging
import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils


def ConcatenateAcrossReplicas(tensors,
                              tpu_cores=None,
                              axis=0,
                              stop_cross_gradients=False):
  """Concatenates one or more local tensors across all TPU cores.

  Input `tensors` may be in any format supported by `tf.nest.flatten` (single
  Tensor, dict, etc.), or a `NestedMap`.

  In order to avoid having to pass a TPU core ID into the infeed, this
  implementation produces a different rotation of the concatenation for each
  core.  For example, core 0 will be arranged as [0, 1, 2, ...], whereas core 3
  will be arranged as [3, 4, ..., 0, 1, 2].

  If called from a non-TPU context, this function returns the `tensors`
  unchanged.

  Args:
    tensors: The local tensor or tensors to concatenate across cores.
    tpu_cores: The total number of TPU cores. If not set, the number of cores is
      inferred from `cluster_factory.Current()`.
    axis: The axis to concatenate.
    stop_cross_gradients: If true, stop gradients on cross-replica slices.

  Returns:
    The tensor(s) concatenated across all replicas.
  """
  if not py_utils.use_tpu():
    return tensors

  if tpu_cores is None:
    cluster = cluster_factory.Current()
    tpu_cores = cluster.tpus_per_replica * cluster.num_replicas
    assert tpu_cores, 'Unable to determine number of TPU cores from cluster.'

  concat_fn = functools.partial(
      CrossReplicaConcat,
      tpu_cores=tpu_cores,
      axis=axis,
      stop_cross_gradients=stop_cross_gradients)
  concatenated_tensors = [concat_fn(t) for t in tf.nest.flatten(tensors)]

  if isinstance(tensors, py_utils.NestedMap):
    return tensors.Pack(concatenated_tensors)
  else:
    return tf.nest.pack_sequence_as(tensors, concatenated_tensors)


# TODO(austinwaters): Update non-Milan callers to use the function above, then
# make this one private.
def CrossReplicaConcat(local_tensor,
                       tpu_cores: int,
                       axis: int = 0,
                       stop_cross_gradients: bool = False):
  """Concatenates a single local tensor across all TPU cores.

  This is mostly a fork of //nlp/neon/dual_encoder/utils/tpu_utils.py, with
  some additional functionality to support int64-typed inputs.

  Args:
    local_tensor: The local tensor to concatenate across cores.
    tpu_cores: The total number of TPU cores.
    axis: The axis to concatenate.
    stop_cross_gradients: Whether or not to stop gradients on cross-replica
      slices.

  Returns:
    The tensor concatenated across all replicas.
  """

  # Handle int64 inputs as a special case since collective_permute() doesn't
  # natively support them. At a high level, we break each int64 into two 32-bit
  # parts, concatenate each part separately, and then recombine the result.
  #
  # Implementation notes:
  #   - The "parts" have to be int32 because collective_permute doesn't support
  #     uint32 inputs, either.
  #   - uint64 <-> int64 casts also have to be avoided because XLA doesn't
  #     know how to compile them for TPU. (Error: "While rewriting computation
  #     to not contain X64 element types, XLA encountered an HLO for which this
  #     rewriting is not implemented...")
  if local_tensor.dtype == tf.int64:
    low32 = tf.cast(local_tensor, tf.int32)
    high32 = tf.cast(
        tf.bitwise.bitwise_and(
            tf.bitwise.right_shift(local_tensor, 32), 0xffffffff), tf.int32)

    # Concatenate each int32 part.
    low32 = CrossReplicaConcat(
        low32, tpu_cores, axis=axis, stop_cross_gradients=stop_cross_gradients)
    high32 = CrossReplicaConcat(
        high32, tpu_cores, axis=axis, stop_cross_gradients=stop_cross_gradients)

    # Recombine high and low parts. Make the low part unsigned before upcasting
    # to avoid propagating its sign bit.
    low32 = tf.cast(tf.cast(low32, tf.uint32), tf.int64)
    high32 = tf.cast(high32, tf.int64)
    return tf.cast(
        tf.bitwise.bitwise_or(low32, tf.bitwise.left_shift(high32, 32)),
        tf.int64)

  all_tensors = [local_tensor]
  for rotation_index in range(tpu_cores - 1):
    permutation = tuple((source, (source + rotation_index + 1) % tpu_cores)
                        for source in range(tpu_cores))
    permuted_tensor = tf.raw_ops.CollectivePermute(
        input=local_tensor, source_target_pairs=permutation)

    if stop_cross_gradients:
      permuted_tensor = tf.stop_gradient(permuted_tensor)

    all_tensors.append(permuted_tensor)

  result = tf.concat(all_tensors, axis=axis)
  logging.info('TPU concat across %d cores; result shape %s', tpu_cores,
               result.shape)
  return result
