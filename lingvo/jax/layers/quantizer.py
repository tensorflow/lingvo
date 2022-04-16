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
"""Vector Quantization layers."""

import jax
import jax.numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import quantizer_objectives as objectives

JTensor = jnp.ndarray
NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
WeightParams = py_utils.weight_params
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


# TODO(nanxinchen): merge this with ngrammer.VectorQuantization
def quantize_vector(latent: JTensor, codebook: JTensor):
  """Vector quantization.

  (port from TF impl of ... speech/quantizer/layers.py)

  Symbols in comments:
  B: batch_size.
  D: latent_dim.
  C: num_latent_classes per group
  G: num of codebook groups.

  Args:
    latent:   [B, D]
    codebook: [C, G, D // G]

  Returns:
    (quantized, codes, onehot).
    - quantized: [B, D]
    - codes:     [B, G]
    - onehot:    [B, G, C]
  """
  # For lower HBM footprint.
  assert len(codebook.shape) == 3
  b, d = latent.shape
  c, g = codebook.shape[:2]
  assert d % g == 0

  latent = jnp.reshape(latent, [b, g, d // g])

  # [B, G, C]
  distance = (
      # [b, g, 1]
      jnp.sum(latent**2, -1, keepdims=True) -
      # [b, g, c]
      2 * jnp.einsum('bgd,cgd->bgc', latent, codebook) +
      # [1, g, c]
      jnp.sum(jnp.transpose(codebook, [2, 1, 0])**2, 0, keepdims=True))
  # distance = py_utils.check_numerics(distance, 'quantization NaN')

  # [B, G]
  codes = jnp.argmin(distance, axis=-1)

  # [B, G, C]
  one_hot = jax.nn.one_hot(codes, c, axis=-1, dtype=jnp.float32)
  quantized = jnp.einsum('bgc,cgd->bgd', one_hot, codebook)
  quantized = jnp.reshape(quantized, [b, d])
  return quantized, codes, one_hot


class RandomVectorQuantizer(base_layer.BaseLayer):
  """Random quantization for BEST-RQ: https://arxiv.org/pdf/2202.01855.pdf."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('latent_dim', None, 'Input dimension.')
    p.Define('projection_dim', 16, 'Projection dimension.')
    p.Define('num_latent_classes', None, 'Number of random quantized classes.')
    return p

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params

    self.create_variable(
        'random_proj',
        WeightParams(
            shape=[p.latent_dim, p.projection_dim],
            init=p.params_init,
            dtype=jnp.float32))
    self.create_variable(
        'random_codebook',
        WeightParams(
            shape=[p.num_latent_classes, 1, p.projection_dim],
            init=p.params_init,
            dtype=jnp.float32))

  def fprop(self, z: JTensor, paddings: JTensor) -> NestedMap:
    del paddings

    p = self.params
    theta = self.local_theta()

    proj_vec = jnp.einsum('dh,btd->bth', theta.random_proj, z)

    batch_size, time_steps, dim = proj_vec.shape
    proj_vec = jnp.reshape(proj_vec, [batch_size * time_steps, dim])
    q, c, onehot = quantize_vector(proj_vec, theta.random_codebook)
    q = jnp.reshape(q, [batch_size, time_steps, dim])
    c = jnp.reshape(c, [batch_size, time_steps])
    onehot = jnp.reshape(onehot, [batch_size, time_steps, p.num_latent_classes])
    return NestedMap(z_q=q, z_codes=c, z_onehot=onehot)


class SeqVectorQuantizer(base_layer.BaseLayer):
  """The VQ-VAE sequence vector quantizer.

  This extends l/b/r/b/m/vqgan.VectorQuantizer by allowing padding inputs.

  Symbols in comments:
  B: batch_size.
  T: sequence length.
  D: latent_dim.
  C: num_latent_classes.
  G: num of codebook groups.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_latent_classes', None, 'Number of latent classes.')
    p.Define('latent_dim', None, 'Latent vector dimension.')
    p.Define('beta', None, 'Scale of the commitment loss.')
    p.Define('normalize_latent_vector', True,
             'Normalize the L2 norm of each latent input vector to 1')
    p.Define('normalize_codebook', True,
             'Normalize the L2 norm of each codebook vector to 1')
    p.Define('num_groups', 1, 'Num of codebook groups.')

    p.name = 'sequence_vector_quantizer'
    p.params_init = py_utils.WeightInit.UniformSqrtDim()
    return p

  def __init__(self, params):
    assert params.num_latent_classes
    assert params.latent_dim
    assert params.beta is not None
    super().__init__(params)
    p = self.params
    assert p.beta >= 0

  def create_layer_variables(self):
    super().create_layer_variables()
    p = self.params

    assert p.latent_dim % p.num_groups == 0
    weight_params = py_utils.weight_params(
        shape=[
            p.num_latent_classes, p.num_groups, p.latent_dim // p.num_groups
        ],
        init=p.params_init,
        dtype=jnp.float32)

    # [C, D]
    self.create_variable('w', weight_params)

  def _l2_normalize(self, x, axis, epsilon=1e-12):
    norm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + epsilon)
    return x / norm

  def _get_latent_embedding(self, theta):
    """Gets the latent embedding."""
    p = self.params
    w = theta.w
    if p.normalize_codebook:
      w = self._l2_normalize(w, -1)
    return w

  def _apply_mask(self, x, mask):
    x_rank = len(x.shape)
    mask_rank = len(mask.shape)
    mask = jnp.reshape(mask, mask.shape + tuple([1] * (x_rank - mask_rank)))
    return x * mask.astype(x.dtype)

  def fprop(self, z: JTensor, paddings: JTensor) -> NestedMap:
    """Quantizes 'z' of shape [B, T, D].

    The z_codes of padded locations are 0.

    Args:
      z:        [B, T, D].
      paddings: [B, T].

    Returns:
      A NestedMap of
        - z_q:               [B, T, D].
        - z_codes:           [B, T, G].
        - z_onehot:          [B, T, G, C].
        - loss:              [], weighted sum of quantization loss and
          commitment loss.
        - codebook_coverage: [], a float scalar tensor between [0, 1].
        - pplx:              [], pplx of quantized distribution over the
          codebook.
        - entropy:           [], exp(pplx).
    """
    p = self.params
    theta = self.local_theta()
    b, t, d = z.shape
    g, c = p.num_groups, p.num_latent_classes

    mask = 1.0 - paddings
    num_frames = jnp.sum(mask)
    z = self._apply_mask(z, mask)

    if p.normalize_latent_vector:
      z = self._l2_normalize(z, axis=-1)

    # [b * t, d], [b * t, g], [b * t, g, c]
    z_q, z_codes, z_onehot = quantize_vector(
        jnp.reshape(z, [b * t, d]), self._get_latent_embedding(theta))

    z_q = jnp.reshape(z_q, [b, t, d])
    z_codes = jnp.reshape(z_codes, [b, t, g])
    z_onehot = jnp.reshape(z_onehot, [b, t, g, c])

    # Padded locations are all 0s without any 1.
    z_q = self._apply_mask(z_q, mask)
    # [b, t, g]
    z_codes = self._apply_mask(z_codes, mask)
    # [b, t, g, c]
    z_onehot = self._apply_mask(z_onehot, mask)

    # Move z towards z_q.
    normalizer = 1e-7 + num_frames
    # [b, t, d]
    loss_c = (z - jax.lax.stop_gradient(z_q))**2
    # [b, t, d] -> [b, t] -> []
    loss_c = jnp.sum(jnp.mean(loss_c, -1)) / normalizer
    # loss_c = py_utils.check_numerics(loss_c, 'loss_c has NaN.')

    # Move z_q towards z.
    loss_z = (z_q - jax.lax.stop_gradient(z))**2
    loss_z = jnp.sum(jnp.mean(loss_z, -1)) / normalizer
    # loss_z = py_utils.check_numerics(loss_z, 'loss_z has NaN.')
    loss = loss_z + p.beta * loss_c

    # Straight-through estimator.
    # Doesn't look like this line does anyhing besides stopping gradient ??
    z_q = z + jax.lax.stop_gradient(z_q - z)

    # [], []
    pplx, entropy, _ = objectives.batch_pplx_entropy_from_codes(
        z_codes, c, paddings=paddings)
    # pplx = py_utils.check_numerics(pplx, f'{p.name} perplexity NaN')

    codebook_coverage = objectives.batch_codebook_coverage(
        z_codes, c, paddings=paddings)
    codebook_num_covered_words = codebook_coverage * c**g

    return py_utils.NestedMap(
        z_q=z_q,
        z_codes=z_codes,
        z_onehot=z_onehot,
        loss=loss,
        codebook_coverage=codebook_coverage,
        codebook_num_covered_words=codebook_num_covered_words,
        pplx=pplx,
        entropy=entropy)

  def look_up(self, z_codes):
    """Looks up latent vectors [B, T, D] by z_codes [B, T, G]."""
    p = self.params
    theta = self.local_theta()
    b, t = z_codes.shape[:2]
    latent = jnp.einsum('btgc,cgd->btgd',
                        jax.nn.one_hot(z_codes, p.num_latent_classes),
                        self._get_latent_embedding(theta))
    return jnp.reshape(latent, [b, t, -1])
