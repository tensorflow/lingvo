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
"""Implementation of multiheaded FAVOR-attention & FAVOR-self-attention layers.

Prefix sum tf implementation by Valerii Likhosherstov.
"""
from lingvo import compat as tf
from lingvo.core import py_utils

BIG_CONSTANT = 1e8


def next_seed(current_seed):
  if current_seed is None:
    return None
  else:
    return current_seed + 1


def create_projection_matrix(nb_random_projections, dim, seed=0, scaling=0):
  r"""Constructs the matrix of random projections.

  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random and either deterministic length
  \sqrt{dim} or length taken from the \chi(dim) distribution (in the latter
  case marginal distributions of the projections are dim-dimensional Gaussian
  vectors with associated identity covariance matrix).

  Args:
    nb_random_projections: number of random projections.
    dim: dimensionality of each random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{dim}, 0 if the lengths of random projections should follow
      \chi(dim) distribution.

  Returns:
    The matrix of random projections of the shape [nb_random_projections, dim].
  """
  if nb_random_projections == 0:
    return None
  nb_full_blocks = nb_random_projections // dim
  block_list = []
  current_seed = seed
  for _ in range(nb_full_blocks):
    unstructured_block = tf.random.normal((dim, dim), seed=current_seed)
    q, _ = tf.linalg.qr(unstructured_block)
    q = tf.transpose(q)
    block_list.append(q)
    current_seed = next_seed(current_seed)
  remaining_rows = nb_random_projections - nb_full_blocks * dim
  if remaining_rows > 0:
    unstructured_block = tf.random.normal((dim, dim), seed=current_seed)
    q, _ = tf.linalg.qr(unstructured_block)
    q = tf.transpose(q)
    block_list.append(q[0:remaining_rows])
  final_matrix = tf.concat(block_list, 0)
  current_seed = next_seed(current_seed)

  if scaling == 0:
    squares = tf.math.square(
        tf.random.normal((nb_random_projections, dim), seed=current_seed))
    squared_lengths = tf.math.reduce_sum(squares, axis=1)
    multiplier = tf.math.sqrt(squared_lengths)
  elif scaling == 1:
    multiplier = tf.math.sqrt(float(dim)) * tf.ones((nb_random_projections))
  else:
    raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

  return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)


def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
  """Computes features for the ReLU-kernel.

  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  del is_query
  if projection_matrix is None:
    return tf.nn.relu(data) + numerical_stabilizer
  else:
    ratio = 1.0 / tf.math.sqrt(
        tf.dtypes.cast(projection_matrix.shape[0], projection_matrix.dtype))
    data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
    return tf.nn.relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
  """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  projection_matrix = tf.cast(projection_matrix, data.dtype)
  data_normalizer = 1.0 / tf.math.sqrt(
      (tf.math.sqrt(tf.dtypes.cast(data.shape[-1], data.dtype))))
  ratio = 1.0 / tf.math.sqrt(
      tf.dtypes.cast(projection_matrix.shape[0], data.dtype))
  data_dash = tf.einsum("blhd,md->blhm", data_normalizer * data,
                        projection_matrix)
  diag_data = tf.math.square(data)
  diag_data = tf.math.reduce_sum(
      diag_data, axis=tf.keras.backend.ndim(data) - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = tf.expand_dims(diag_data, axis=tf.keras.backend.ndim(data) - 1)
  if is_query:
    last_dims_t = (len(data_dash.shape) - 1,)
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t, keepdims=True)) + numerical_stabilizer)
  else:
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash)) +
        numerical_stabilizer)

  return data_dash


def cossim_kernel_transformation(data,
                                 is_query,
                                 projection_matrix=None,
                                 numerical_stabilizer=0.0,
                                 randomized=False):
  """Computes features for the softmax kernel with FAVOR+ cossim mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    randomized: whether randomized version of the cos similarity is used.

  Returns:
    Corresponding kernel feature map.
  """
  if is_query:
    r = tf.math.sqrt(tf.dtypes.cast(data.shape[-1], data.dtype))
    if randomized:
      projection_matrix = tf.cast(projection_matrix, data.dtype)
      ratio = 1.0 / tf.math.sqrt(
          tf.cast(tf.shape(projection_matrix)[0], data.dtype))
      return ratio * (
          tf.einsum("blhd,md->blhm", r * tf.math.l2_normalize(data, axis=[-1]),
                    projection_matrix) +
          tf.cast(numerical_stabilizer, data.dtype))
    else:
      return r * tf.math.l2_normalize(data, axis=[-1])
  else:
    if randomized:
      projection_matrix = tf.cast(projection_matrix, data.dtype)
      ratio = 1.0 / tf.math.sqrt(
          tf.cast(tf.shape(projection_matrix)[0], data.dtype))
      return ratio * tf.einsum("blhd,md->blhm",
                               tf.math.l2_normalize(data, axis=[-1]),
                               projection_matrix)
    else:
      return tf.math.l2_normalize(data, axis=[-1])


def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs)
  return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
  """Computes FAVOR normalizer in noncausal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in noncausal attention.
  """
  ks_sum = tf.reduce_sum(ks, axis=0)
  return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)


@tf.custom_gradient
def causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """

  result = []
  sums = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

  for index in range(qs.shape[0]):
    sums = sums + tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])
    result.append(tf.einsum("ijkl,ijk->ijl", sums, qs[index])[None, ...])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    grads = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    gr_sums = sums

    q_grads = []
    k_grads = []
    v_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijkl,ijl->ijk", gr_sums, res_grad[index])[None, ...])
      grads = grads + tf.einsum("ijk,ijl->ijkl", qs[index], res_grad[index])
      k_grads.append(tf.einsum("ijkl,ijl->ijk", grads, vs[index])[None, ...])
      v_grads.append(tf.einsum("ijkl,ijk->ijl", grads, ks[index])[None, ...])
      gr_sums = gr_sums - tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)
    v_grads = tf.concat(v_grads[::-1], axis=0)

    return q_grads, k_grads, v_grads

  return result, grad


@tf.custom_gradient
def causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in causal attention.
  """

  result = []
  sums = tf.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(tf.reduce_sum(qs[index] * sums, axis=2)[None, ...])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    k_grad = tf.zeros_like(ks[0])

    gr_sums = sums

    q_grads = []
    k_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijk,ij->ijk", gr_sums, res_grad[index])[None, ...])
      k_grad = k_grad + tf.einsum("ijk,ij->ijk", qs[index], res_grad[index])
      k_grads.append(k_grad[None, ...])
      gr_sums = gr_sums - ks[index]

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)

    return q_grads, k_grads

  return result, grad


def favor_attention(query,
                    key,
                    value,
                    paddings,
                    kernel_transformation,
                    causal,
                    projection_matrix=None):
  """Computes FAVOR normalized attention.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    paddings: paddings tensor.
    kernel_transformation: transformation used to get finite kernel features.
    causal: whether attention is causal or not.
    projection_matrix: projection matrix to be used.

  Returns:
    FAVOR normalized attention.
  """
  query_prime = kernel_transformation(query, True,
                                      projection_matrix)  # [B,L,H,M]
  key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
  if paddings is not None:
    b, l, h, m = py_utils.GetShape(key_prime, 4)
    paddings = tf.tile(tf.reshape(paddings, [b, l, 1, 1]), [1, 1, h, m])
    key_prime *= tf.cast(1.0 - paddings, key_prime.dtype)
  query_prime = tf.transpose(query_prime, [1, 0, 2, 3])  # [L,B,H,M]
  key_prime = tf.transpose(key_prime, [1, 0, 2, 3])  # [L,B,H,M]
  value = tf.transpose(value, [1, 0, 2, 3])  # [L,B,H,D]
  # TODO(kchoro): Get rid of the transpose operations, at least in the
  # bidirectional variant.

  if causal:
    av_attention = causal_numerator(query_prime, key_prime, value)
    attention_normalizer = causal_denominator(query_prime, key_prime)
  else:
    av_attention = noncausal_numerator(query_prime, key_prime, value)
    attention_normalizer = noncausal_denominator(query_prime, key_prime)
  # TODO(kchoro): Add more comments.
  av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
  attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])
  attention_normalizer = tf.expand_dims(attention_normalizer,
                                        len(attention_normalizer.shape))
  return av_attention / attention_normalizer
