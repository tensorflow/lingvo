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
"""N-grammer layers from https://openreview.net/forum?id=GxjCYmQAody."""
from typing import Optional, Tuple

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import embedding_softmax
from lingvo.jax.layers import normalizations
import sympy

NestedMap = py_utils.NestedMap
weight_params = py_utils.weight_params
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


def get_bigram_ids(ids: JTensor,
                   vocab_size: int,
                   segment_pos: Optional[JTensor] = None) -> JTensor:
  """Generate bi-gram ids from uni-gram ids.

  Args:
    ids: An int32 JTensor of shape [B, L].
    vocab_size: Vocabulary size of `ids`, must be > 0.
    segment_pos: If not None (meaning `ids` is packed, i.e. each example
      containing multiple segments), an int32 tensor of shape [B, L], containing
      the position of each id in `ids` in a segment.

  Returns:
    ngram_ids: An int64 JTensor of shape [B, L].
  """
  assert vocab_size > 0
  batch_size = ids.shape[0]
  # Cast to int64 to avoid overflow, which would affect bucket collision
  # rate and model quality.
  ids = jnp.array(ids, dtype=jnp.int64)  # [batch, time]
  pad = jnp.zeros([batch_size, 1], dtype=ids.dtype)  # [batch, 1]

  # Mechanism: for bigrams, we shift ids by one position along the time
  # dimension, and compute:
  #   bigram_id = original_id + shifted_id * vocab_size.
  ids_0 = jnp.concatenate([ids, pad], 1)  # [batch, time+1]
  ids_1 = jnp.concatenate([pad, ids], 1)  # [batch, 1+time]

  if segment_pos is not None:
    # If input is packed, mask out the parts that cross the segment
    # boundaries.
    mask = jnp.array(jnp.equal(segment_pos, 0), dtype=ids_0.dtype)
    mask = 1 - mask
    mask = jnp.concatenate([mask, pad], 1)
    ids_1 *= mask

  ngram_ids = ids_0 + ids_1 * vocab_size  # Bigram ids.
  ngram_ids = ngram_ids[:, 0:-1]
  return ngram_ids


class VectorQuantization(base_layer.BaseLayer):
  """Implements vector quantization (VQ)/online k-means clustering.

  This layer computes a discrete latent representation of a sequence, in a
  manner similar to https://arxiv.org/abs/1805.11063, where each sequence
  position is assigned a cluster membership. This can be useful in 1) reducing
  the latency of decoding a sequence 2) reducing the vocabulary of a sequence
  which can be used to augment the sequence with n-grams and 3) for computing
  sparse attention over a long sequence as in
  https://transacl.org/ojs/index.php/tacl/article/view/2405. Note that this
  applies multi-head VQ, where each head has a separate set of centroids.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of clusters
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params."""
    p = super().Params()
    p.Define(
        'num_clusters', 0, 'Number of clusters, typically around the square'
        ' root of the sequence length.')
    p.Define('num_heads', 0, 'Number of attention heads.')
    p.Define('decay', 0.999, 'The decay with which to update centroids.')
    p.Define('epsilon', 1e-6, 'Tiny value to guard against divide by 0.')
    p.Define(
        'dim_per_head', 0, 'The last dimension of the inputs on which to'
        'apply Vector Quantization.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructs an instance which tracks its own set of centroids."""
    super().__init__(params)
    p = self.params
    assert p.num_clusters
    assert p.dim_per_head

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    means = weight_params(
        shape=[p.num_heads, p.num_clusters, p.dim_per_head],
        init=p.params_init,
        dtype=self.fprop_dtype,
        collections=[base_layer.REQUIRES_MEAN_SYNC])
    self.create_variable('means', means, trainable=False)

  def fprop(self,
            theta: NestedMap,
            inputs: JTensor,
            paddings: Optional[JTensor] = None) -> Tuple[JTensor, JTensor]:
    """Computes distances of the given input 'x' to all centroids.

    Args:
      theta: A `.NestedMap` of weights' values of this layer.
      inputs: Input tensor of shape [B, L, N, H] or [B, L, D].
      paddings: If not None, a tensor of shape [B, L]. The padding tensor is
        supplied when we want certain tokens to not affect the centroids.

    Returns:
      dists: "distances" of the given input 'x' to all centroids.
             Shape [B, L, N, K].
      nearest_centroid: The inputs with the input embeddings replaced by the
             centroid embeddings, it has the same shape as the inputs i.e.,
             [B, L, N, H].
    """
    p = self.params
    inputs = jnp.array(inputs, dtype=theta.means.dtype)
    inputs_shape = inputs.shape
    if len(inputs_shape) == 3:
      inputs = jnp.reshape(
          inputs,
          [inputs_shape[0], inputs_shape[1], p.num_heads, p.dim_per_head])

    if paddings is not None:
      # Shape [B, L, 1, 1]
      paddings_4d = paddings[:, :, jnp.newaxis, jnp.newaxis]

    dists = -2 * jnp.einsum('BLNH, NKH -> BLNK', inputs, theta.means)
    # [B, L, N, 1]
    inputs_norm_sq = jnp.sum(jnp.square(inputs), axis=-1, keepdims=True)
    # [N, K]
    means_norm_sq = jnp.sum(jnp.square(theta.means), axis=-1, keepdims=False)
    # [1, 1, N, K]
    means_norm_sq = means_norm_sq[jnp.newaxis, jnp.newaxis, :, :]
    dists += inputs_norm_sq + means_norm_sq

    # Shape [B, L, N, K], the same as 'dists' above.
    nearest_one_hot = jax.nn.one_hot(
        jnp.argmin(dists, axis=-1), p.num_clusters, dtype=theta.means.dtype)

    # Apply paddings.
    if paddings is not None:
      nearest_one_hot *= (1 - paddings_4d)

    # Same shape as the input [B, L, N, H].
    nearest_centroid = jnp.einsum('BLNK, NKH -> BLNH', nearest_one_hot,
                                  theta.means)

    means_norm = jnp.linalg.norm(theta.means, ord=2, axis=-1)
    base_layer.add_summary('k_means/centroid/l2_norm_avg', jnp.mean(means_norm))
    base_layer.add_summary('k_means/centroid/l2_norm_min', jnp.min(means_norm))
    base_layer.add_summary('k_means/centroid/l2_norm_max', jnp.max(means_norm))

    if not self.do_eval:
      # To update the centroids (self.vars.means), we apply gradient descent on
      # the mini-batch of input, which yields the following:
      #   new_centroid = centroid + (1 - decay) * (x_mean - centroid)
      # where x_mean is the average over all the input vectors closest to this
      # centroid.

      # Sum away batch and sequence length dimensions to get per cluster count.
      # Shape: [N, K]
      per_cluster_count = jnp.sum(nearest_one_hot, axis=[0, 1])
      base_layer.add_summary('k_means/centroid/avg_cluster_count',
                             jnp.mean(per_cluster_count))

      # Sum of the input per each closest centroid.
      sum_x = jnp.einsum('BLNK, BLNH -> NKH', nearest_one_hot, inputs)

      # Sum up cluster counts across replicas.

      # If per_cluster_count for a cluster is 0, then 'nearest_one_hot' in that
      # cluster's position will always be 0, hence 'sum_x' in that dimension
      # will be 0.
      new_means = sum_x / (
          p.epsilon + jnp.expand_dims(per_cluster_count, axis=-1))
      updated_means = (1.0 - p.decay) * new_means + p.decay * theta.means
      updated_means = jnp.array(updated_means, self.vars.means.dtype)
      self.forward_update_var('means', updated_means)
    return dists, nearest_centroid


class Ngrammer(base_layer.BaseLayer):
  """Implements a generic N-grammer layer which looks up latent bi-gram id.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of clusters
    D = total dimension which is H * N
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params."""
    p = super().Params()
    p.Define('ngram_vocab_size', 768 * 256, 'Size of the ngram vocabulary.')
    p.Define('unigram_vocab_size', 0, 'Size of the unigram vocabulary.')
    p.Define('ngram_emb_dim', 8, 'Size of the ngram dimension per head.')
    p.Define('concat_ngrams', True, 'If True, then concat ngrams.')
    p.Define('num_heads', 0, 'Number of attention heads.')
    p.Define('dim_per_head', 0, 'The dimension per each head of the input.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructs an instance which looks up ngrams."""
    super().__init__(params)
    p = self.params

    if p.concat_ngrams:
      # The ngram_emb_dim must be smaller than dim_per_head.
      assert p.ngram_emb_dim <= p.dim_per_head
    else:
      # If not concatenating ngram embeddings, check the dims are compatible.
      assert p.ngram_emb_dim == p.dim_per_head

    # Create a separate layer norm per head for embedding normalization.
    # Create a separate layer norm per head for ngram embedding normalization.
    emb_layer_norm_p = []
    ngram_emb_layer_norm_p = []
    ngram_emb_table_p = []
    for i in range(p.num_heads):
      layer_norm_p = normalizations.LayerNorm.Params().Copy()
      layer_norm_p.input_dims = p.dim_per_head
      layer_norm_p.name = f'layer_norm_{i}'

      emb_layer_norm_p.append(layer_norm_p)
      ngram_layer_norm_p = normalizations.LayerNorm.Params().Copy()
      ngram_layer_norm_p.input_dims = p.ngram_emb_dim
      ngram_emb_layer_norm_p.append(ngram_layer_norm_p)

      # Create embedding table for ngram lookup.
      embedding_p = (embedding_softmax.SingleShardEmbedding.Params().Copy())
      embedding_p.name = f'embedding_{i}'
      embedding_p.vocab_size = p.ngram_vocab_size
      embedding_p.embedding_dims = p.ngram_emb_dim
      ngram_emb_table_p.append(embedding_p)

    self.create_children('emb_layer_norm', emb_layer_norm_p)
    self.create_children('ngram_layer_norm', ngram_emb_layer_norm_p)
    self.create_children('ngram_table', ngram_emb_table_p)

  def fprop(self,
            theta: NestedMap,
            input_ids: JTensor,
            input_embs: JTensor,
            paddings: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> JTensor:
    """Augments the input embeddings with VQ n-gram layer embeddings.

    Args:
      theta: A `.NestedMap` of weights' values of this layer.
      input_ids: Input unigram id tensor of shape [B, L] or [B, L, N].
      input_embs: Input unigram embedding tensor of shape [B, L, D] to which to
        add the ngram embedding.
      paddings: If not None, a tensor of shape [B, L] corresponding to padding.
      segment_pos: If not None, a tensor of shape [B, L] corresponding to the
        position of an id in a packed sequence.

    Returns:
      outputs: Output with the ngram embeddings added of shape [B, L, D].
    """
    p = self.params
    if paddings is not None:
      # Shape [B, L, 1]
      paddings_3d = paddings[:, :, jnp.newaxis]

    inputs_shape = input_ids.shape
    batch_size = inputs_shape[0]
    seq_length = inputs_shape[1]

    # [B, L].
    if len(inputs_shape) == 2:
      input_ids_per_head = [input_ids] * p.num_heads
    else:
      input_ids_per_head = jnp.split(input_ids, p.num_heads, axis=-1)
      input_ids_per_head = [
          jnp.squeeze(ids, axis=-1) for ids in input_ids_per_head
      ]

    # Reshape to [B, L, N, H].
    input_embs = jnp.reshape(input_embs,
                             [batch_size, seq_length, p.num_heads, -1])

    def _multi_way_hash_ids(x, a, b, prime, buckets):
      return ((x * a + b) % prime) % buckets

    ngram_embs_to_concat = []
    vocab_size = p.ngram_vocab_size
    primes = list(
        sympy.primerange(p.ngram_vocab_size + 1,
                         2 * p.ngram_vocab_size))[0:p.num_heads]
    for i in range(p.num_heads):
      ngram_ids = get_bigram_ids(input_ids_per_head[i], p.unigram_vocab_size,
                                 segment_pos)

      ngram_ids_for_head = _multi_way_hash_ids(ngram_ids, i + 1, i + 1,
                                               primes[i], vocab_size)
      ngram_embs_to_concat.append(self.ngram_table[i].fprop(
          theta.ngram_table[i], jnp.reshape(ngram_ids_for_head, [-1])))
      # [B * L, H]
      ngram_embs_to_concat[i] = self.ngram_layer_norm[i].fprop(
          theta.ngram_layer_norm[i], ngram_embs_to_concat[i])

    # [B * L, N * H].
    ngram_embs = jnp.concatenate(ngram_embs_to_concat, 1)
    ngram_embs = jnp.reshape(
        ngram_embs, [batch_size, seq_length, p.num_heads, p.ngram_emb_dim])

    # Layer norm input embeddings independently for each head.
    input_embs_per_head = jnp.split(input_embs, p.num_heads, 2)
    for i in range(p.num_heads):
      # Reshape into [B * L, H]
      per_head_emb = jnp.reshape(input_embs_per_head[i], [-1, p.dim_per_head])
      input_embs_per_head[i] = self.emb_layer_norm[i].fprop(
          theta.emb_layer_norm[i], per_head_emb)
      # Reshape to [B, L, H]
      input_embs_per_head[i] = jnp.reshape(
          input_embs_per_head[i], [batch_size, seq_length, p.dim_per_head])

    # [B, L, N, H].
    input_embs = jnp.stack(input_embs_per_head, 2)

    if p.concat_ngrams:
      d = p.dim_per_head - p.ngram_emb_dim
      input_embs_slice = jax.lax.dynamic_slice_in_dim(
          input_embs, start_index=0, slice_size=d, axis=-1)
      input_embs = jnp.concatenate([input_embs_slice, ngram_embs], axis=-1)
    else:
      input_embs += ngram_embs

    # [B, L, D].
    input_embs = jnp.reshape(input_embs, [batch_size, seq_length, -1])

    # Apply paddings back.
    if paddings is not None:
      input_embs *= (1 - paddings_3d)
    return input_embs


class VQNgrammer(base_layer.BaseLayer):
  """Implements a VQ based ngrammer layer which looks up latent ngram id.

  We use the following capital letters to denote shape parameters:
    B = batch size
    L = length of the input sequence (referred to as S or T elsewhere)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of clusters
    D = total dimension which is H * N
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params."""
    p = super().Params()
    p.Define('ngram_vocab_size', 768 * 256, 'Size of the ngram vocabulary.')
    p.Define('ngram_emb_dim', 8, 'Size of the ngram dimension per head.')
    p.Define('concat_ngrams', False, 'If True, then concat ngrams.')
    p.Define('num_clusters', 0, 'Number of clusters.')
    p.Define('num_heads', 0, 'Number of attention heads.')
    p.Define('decay', 0.999, 'The decay with which to update centroids.')
    p.Define('epsilon', 1e-6, 'Tiny value to guard against divide by 0.')
    p.Define(
        'dim_per_head', 0, 'The last dimension of the inputs on which to'
        'apply Vector Quantization.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructs a VQ layer and an N-grammer layer."""
    super().__init__(params)
    p = self.params

    if p.concat_ngrams:
      # The ngram_emb_dim must be smaller than dim_per_head.
      assert p.ngram_emb_dim <= p.dim_per_head
    else:
      # If not concatenating ngram embeddings, check the dims are compatible.
      assert p.ngram_emb_dim == p.dim_per_head

    # Create VQ layer.
    vq_layer_p = VectorQuantization.Params().Set(
        num_clusters=p.num_clusters,
        num_heads=p.num_heads,
        dim_per_head=p.dim_per_head,
        decay=p.decay,
        epsilon=p.epsilon)
    self.create_child('vq_layer', vq_layer_p)

    # Create N-gram lookup layer.
    ngram_layer_p = Ngrammer.Params().Set(
        ngram_vocab_size=p.ngram_vocab_size,
        unigram_vocab_size=p.num_clusters,
        ngram_emb_dim=p.ngram_emb_dim,
        concat_ngrams=p.concat_ngrams,
        num_heads=p.num_heads,
        dim_per_head=p.dim_per_head,
    )
    self.create_child('ngram_layer', ngram_layer_p)

  def fprop(self,
            theta: NestedMap,
            input_ids: JTensor,
            input_embs: JTensor,
            paddings: Optional[JTensor] = None,
            segment_pos: Optional[JTensor] = None) -> JTensor:
    """Augments the input embeddings with VQ ngram layer embeddings.

    Args:
      theta: A `.NestedMap` of weights' values of this layer.
      input_ids: Input unigram id tensor of shape [B, L] or [B, L, N]. This is
        unused and is added here to be consistent with the Ngrammger API.
      input_embs: Input unigram embedding tensor of shape [B, L, D] to which to
        add the ngram embedding.
      paddings: If not None, a tensor of shape [B, L] corresponding to padding.
      segment_pos: If not None, a tensor of shape [B, L] corresponding to the
        position of an id in a packed sequence.

    Returns:
      outputs: Input embedding with the VQ ngram added of shape [B, L, D].
    """
    del input_ids

    # Distances of shape [B, L, N, K].
    distances, _ = self.vq_layer.fprop(
        theta.vq_layer, input_embs, paddings=paddings)

    # [B, L, N].
    cluster_ids = jnp.argmin(distances, -1)

    # [B, L, D].
    output_embs = self.ngram_layer.fprop(theta.ngram_layer, cluster_ids,
                                         input_embs, paddings, segment_pos)
    return output_embs
