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
"""Utilities to handle XLA sharding annotations."""

from absl import logging
import jax
import jax.numpy as jnp
from lingvo.jax import pytypes
import numpy as np

SplitDimsMapping = pytypes.SplitDimsMapping


def RemoveDim(dim: int,
              split_dims_mapping: SplitDimsMapping) -> SplitDimsMapping:
  """Returns a copy of split_dims_mapping with dimension 'dim' removed."""
  if dim < 0:
    num_dims = len(split_dims_mapping)
    dim = num_dims + dim
  assert dim >= 0 and dim < len(split_dims_mapping)
  return list(split_dims_mapping[:dim]) + list(split_dims_mapping[dim + 1:])


def CumSum(elements, axis=0, exclusive=False, reverse=False):
  """Same as jax.np.cumsum but with the extra options from tf.cumsum.

  Args:
    elements: A Jax array. The cumulative sum is computed along the 'axis'
      dimension.
    axis: The axis to compute the cumsum over.
    exclusive: If True, perform exclusive cumsum.
      With exclusive=False: cumsum([a, b, c]) --> [a, a + b, a + b + c]
      With exclusive=True: cumprod([a, b, c]) --> [0, a, a + b]
    reverse: A bool (default: False), perform the cumulative sum in reverse.

  Returns:
    The cumulative sum.
  """
  if reverse:
    elements = jnp.flip(elements, axis=axis)

  result = jnp.cumsum(elements, axis=axis)
  if exclusive:
    result = result - elements
  if reverse:
    return jnp.flip(result, axis=axis)
  else:
    return result


# Difference wrt. to lingvo/core/gshard_layers.py:
# Removed args:
#   - inputs: always use sharded logits across tpu cores
#   - use_xla_sharding: rely on sharding propagation instead
#   - num_devices: no longer needed without use_xla_sharding
def Top2GatingOnLogits(paddings,
                       logits,
                       experts_dim,
                       expert_capacity_dim,
                       fprop_dtype,
                       prng_key,
                       second_expert_policy='all',
                       second_expert_threshold=0.0,
                       legacy_mtf_behavior=True,
                       capacity_factor=None,
                       importance=None,
                       mask_dtype=None):
  """Computes Top-2 gating for Mixture-of-Experts.

  This function assumes sharded `logits` across tpu cores as inputs. The
  operations within this function are explicitly sharded/replicated across
  tpu cores with jax.with_sharding_constraint.

  The ` next to an axis indicates common ways of splitting along mesh dimension.

  Dimensions:

    G: group dim
    S: group size dim
    E: number of experts
    C: capacity per expert
    M: model_dim (same as input_dim and output_dim as in FF layer)
    B: original batch dim
    L: original seq len dim

  Note that for local_dispatch, the original batch BLM is reshaped to GSM, each
  group `g = 0..G-1` is being dispatched independently.

  Args:
    paddings: G`S tensor.
    logits: G`SE tensor.
    experts_dim: number of experts
    expert_capacity_dim: number of examples per minibatch/group per expert. Each
      example is typically a vector of size input_dim, representing embedded
      token or an element of Transformer layer output.
    fprop_dtype: activation dtype
    prng_key: jax.random.PRNGKey used for randomness.
    second_expert_policy: 'all', 'sampling' or 'random'
      - 'all': we greedily pick the 2nd expert
      - 'sampling': we sample the 2nd expert from the softmax
      - 'random': we optionally randomize dispatch to second-best expert in
        proportional to (weight / second_expert_threshold).
    second_expert_threshold: threshold for probability normalization when
      second_expert_policy == 'random'
    legacy_mtf_behavior: bool, True if to match legacy mtf behavior exactly.
    capacity_factor: if set, increases expert_capacity_dim to at least
      (group_size * capacity_factor) / experts_dim
    importance: input importance weights for routing (G`S tensor or None)
    mask_dtype: using bfloat16 for fprop_dtype could be problematic for mask
      tensors, mask_dtype overrides dtype for such tensors

  Returns:
    A tuple (aux_loss, combine_tensor, dispatch_tensor).

    - aux_loss: auxiliary loss, for equalizing the expert assignment ratios.
    - combine_tensor: a G`SEC tensor for combining expert outputs.
    - dispatch_tensor: a G`SEC tensor, scattering/dispatching inputs to experts.
  """
  if mask_dtype is None:
    mask_dtype = fprop_dtype

  raw_gates = jax.nn.softmax(logits, axis=-1)  # along E dim
  if raw_gates.dtype != fprop_dtype:
    raw_gates = raw_gates.asdtype(fprop_dtype)

  if capacity_factor is not None:
    # Determine expert capacity automatically depending on the input size
    group_size_dim = logits.shape[1]
    auto_expert_capacity = int(group_size_dim * capacity_factor / experts_dim)
    if expert_capacity_dim < auto_expert_capacity:
      expert_capacity_dim = auto_expert_capacity
      # Round up to a multiple of 4 to avoid possible padding.
      while expert_capacity_dim % 4:
        expert_capacity_dim += 1
      logging.info(
          'Setting expert_capacity_dim=%r (capacity_factor=%r '
          'group_size_dim=%r experts_dim=%r)', expert_capacity_dim,
          capacity_factor, group_size_dim, experts_dim)

  # top first and second gate value and expert index for each input
  #
  # GSK tensors, K=2
  def Split(x):
    # TODO(zhangqiaorjc): figure out the splits
    return x

  # TODO(zhangqiaorjc): Add summary.

  # top-1 index: GS tensor
  index_1 = jnp.argmax(raw_gates, axis=-1)
  index_1 = Split(index_1)

  # GSE
  mask_1 = jax.nn.one_hot(index_1, experts_dim, dtype=mask_dtype)
  mask_1 = Split(mask_1)
  density_1_proxy = raw_gates

  if importance is not None:
    importance_is_one = jnp.equal(importance, 1.0)
    mask_1 *= jnp.expand_dims(importance_is_one.astype(mask_1.dtype), -1)
    density_1_proxy *= jnp.expand_dims(
        importance_is_one.astype(density_1_proxy.dtype), -1)
  else:
    assert len(mask_1.shape) == 3
    importance = jnp.ones_like(mask_1[:, :, 0])
    if paddings is not None:
      nonpaddings = 1.0 - paddings
      mask_1 *= jnp.expand_dims(nonpaddings.astype(mask_1.dtype), -1)
      density_1_proxy *= jnp.expand_dims(
          nonpaddings.astype(density_1_proxy.dtype), -1)
      importance = nonpaddings

  gate_1 = jnp.einsum('GSE,GSE->GS', raw_gates, mask_1.astype(raw_gates.dtype))
  gates_without_top_1 = raw_gates * (1.0 - mask_1.astype(raw_gates.dtype))

  if second_expert_policy == 'sampling':
    # We directly sample the 2nd expert index from the softmax over of the 2nd
    # expert by getting rid of the 1st expert already selected above. To do so,
    # we set a very negative value to the logit corresponding to the 1st expert.
    # Then we sample from the softmax distribution using the Gumbel max trick.
    prng_key, subkey = jax.random.split(prng_key)
    noise = Split(jax.random.uniform(subkey, logits.shape, dtype=logits.dtype))
    # Generates standard Gumbel(0, 1) noise, GSE tensor.
    noise = -jnp.log(-jnp.log(noise))
    very_negative_logits = Split(
        jnp.ones_like(logits) * (-0.7) * np.finfo(logits.dtype).max)
    # Get rid of the first expert by setting its logit to be very negative.
    updated_logits = Split(
        jnp.where(mask_1 > 0.0, very_negative_logits, logits))
    # Add Gumbel noise to the updated logits.
    noised_logits = Split(updated_logits + noise)
    # Pick the index of the largest noised logits as the 2nd expert. This is
    # equivalent to sampling from the softmax over the 2nd expert.
    index_2 = jnp.argmax(noised_logits, axis=-1)
  else:
    # Greedily pick the 2nd expert.
    index_2 = jnp.argmax(gates_without_top_1, axis=-1)

  index_2 = Split(index_2)
  mask_2 = jax.nn.one_hot(index_2, experts_dim, dtype=mask_dtype)
  mask_2 = Split(mask_2)
  if paddings is not None:
    importance_is_nonzero = importance > 0.0
    mask_2 *= jnp.expand_dims(importance_is_nonzero.astype(mask_2.dtype), -1)
  gate_2 = jnp.einsum('GSE,GSE->GS', gates_without_top_1,
                      mask_2.astype(gates_without_top_1.dtype))

  # See notes in lingvo/core/gshard_layers.py.
  if legacy_mtf_behavior:
    # Renormalize.
    denom = gate_1 + gate_2 + 1e-9
    gate_1 /= denom
    gate_2 /= denom

  # We reshape the mask as [X*S, E], and compute cumulative sums of assignment
  # indicators for each expert index e \in 0..E-1 independently.
  # First occurrence of assignment indicator is excluded, see exclusive=True
  # flag below.
  # cumsum over S dim: mask_1 is GSE tensor.
  position_in_expert_1 = CumSum(mask_1, exclusive=True, axis=-2)

  # GE tensor (reduce S out of GSE tensor mask_1).
  # density_1[:, e] represents assignment ration (num assigned / total) to
  # expert e as top_1 expert without taking capacity into account.
  assert importance.dtype == fprop_dtype
  if legacy_mtf_behavior:
    density_denom = 1.0
  else:
    density_denom = jnp.mean(importance, axis=1)[:, jnp.newaxis] + 1e-6
  density_1 = jnp.mean(mask_1.astype(fprop_dtype), axis=-2) / density_denom
  # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
  # those of examples not assigned to e with top_k
  density_1_proxy = jnp.mean(density_1_proxy, axis=-2) / density_denom

  # Compute aux_loss
  aux_loss = jnp.mean(density_1_proxy * density_1)  # element-wise
  aux_loss *= (experts_dim * experts_dim)  # const coefficients

  mask_1 *= jnp.less(position_in_expert_1,
                     expert_capacity_dim).astype(mask_1.dtype)
  position_in_expert_1 = jnp.einsum('GSE,GSE->GS', position_in_expert_1, mask_1)

  # How many examples in this sequence go to this expert?
  mask_1_count = jnp.einsum('GSE->GE', mask_1)
  # [batch, group] - mostly ones, but zeros where something didn't fit.
  mask_1_flat = jnp.sum(mask_1, axis=-1)
  assert mask_1_count.dtype == mask_dtype
  assert mask_1_flat.dtype == mask_dtype

  if second_expert_policy == 'all' or second_expert_policy == 'sampling':
    pass
  else:
    assert second_expert_policy == 'random'
    # gate_2 is between 0 and 1, reminder:
    #
    #   raw_gates = jax.nn.softmax(logits)
    #   index_1 = jnp.argmax(raw_gates, axis=-1)
    #   mask_1 = jax.nn.one_hot(index_1, experts_dim, dtpe=fprop_dtype)
    #   gate_1 = jnp.einsum(`GSE,GSE->GS', raw_gates, mask_1)
    #
    # e.g., if gate_2 exceeds second_expert_threshold, then we definitely
    # dispatch to second-best expert. Otherwise, we dispatch with probability
    # proportional to (gate_2 / threshold).
    #
    prng_key, subkey = jax.random.split(prng_key)
    sampled_2 = jnp.less(
        Split(jax.random.uniform(subkey, gate_2.shape, dtype=gate_2.dtype)),
        gate_2 / max(second_expert_threshold, 1e-9))
    gate_2 *= sampled_2.astype(gate_2.dtype)
    mask_2 *= jnp.expand_dims(sampled_2, -1).astype(mask_2.dtype)

  position_in_expert_2 = CumSum(
      mask_2, exclusive=True, axis=-2) + jnp.expand_dims(mask_1_count, -2)

  mask_2 *= jnp.less(position_in_expert_2,
                     expert_capacity_dim).astype(mask_2.dtype)
  position_in_expert_2 = jnp.einsum('GSE,GSE->GS', position_in_expert_2, mask_2)
  mask_2_flat = jnp.sum(mask_2, axis=-1)

  gate_1 *= mask_1_flat.astype(gate_1.dtype)
  gate_2 *= mask_2_flat.astype(gate_2.dtype)

  if not legacy_mtf_behavior:
    denom = gate_1 + gate_2
    # To avoid divide by 0.
    denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
    gate_1 /= denom
    gate_2 /= denom

  # GSC tensor
  b = jax.nn.one_hot(
      position_in_expert_1.astype(np.int32),
      expert_capacity_dim,
      dtype=fprop_dtype)
  # GSE tensor
  a = jnp.expand_dims(
      gate_1 * mask_1_flat.astype(fprop_dtype), axis=-1) * jax.nn.one_hot(
          index_1, experts_dim, dtype=fprop_dtype)
  # GSEC tensor
  first_part_of_combine_tensor = jnp.einsum('GSE,GSC->GSEC', a, b)

  # GSC tensor
  b = jax.nn.one_hot(
      position_in_expert_2.astype(np.int32),
      expert_capacity_dim,
      dtype=fprop_dtype)
  # GSE tensor
  a = jnp.expand_dims(
      gate_2 * mask_2_flat.astype(fprop_dtype), axis=-1) * jax.nn.one_hot(
          index_2, experts_dim, dtype=fprop_dtype)
  second_part_of_combine_tensor = jnp.einsum('GSE,GSC->GSEC', a, b)

  # GSEC tensor
  combine_tensor = first_part_of_combine_tensor + second_part_of_combine_tensor
  combine_tensor = Split(combine_tensor)

  # GSEC tensor
  dispatch_tensor = combine_tensor.astype(bool).astype(fprop_dtype)
  dispatch_tensor = Split(dispatch_tensor)

  return aux_loss, combine_tensor, dispatch_tensor
