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
"""Tests for ngrammer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from lingvo.core import attention_util
from lingvo.jax import base_layer
from lingvo.jax import test_utils
from lingvo.jax.layers import ngrammer
import numpy as np
import tensorflow as tf

to_np = test_utils.to_np


class NgrammerTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)

  @parameterized.parameters(
      (10000),
      (1000),
      (320000),
      (500),
  )
  def test_get_bigram_ids(self, vocab_size):
    ids = np.random.randint(vocab_size, size=(2, 16), dtype=np.int64)
    ngram_ids = ngrammer.get_bigram_ids(ids, vocab_size)
    np_ngram_ids = to_np(ngram_ids)
    self.assertLess(np.max(np_ngram_ids), vocab_size**2)

  @parameterized.parameters(
      (10000),
      (1000),
      (320000),
      (500),
  )
  def test_get_bigram_ids_with_packing(self, vocab_size):
    ids = np.random.randint(vocab_size, size=(2, 8), dtype=np.int64)
    segment_pos = np.array([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 0, 1, 2, 3, 4]])
    ngram_ids = ngrammer.get_bigram_ids(ids, vocab_size, segment_pos)
    np_ngram_ids = to_np(ngram_ids)
    self.assertLess(np.max(np_ngram_ids), vocab_size**2)
    self.assertEqual(np_ngram_ids[0, 0], ids[0, 0])
    self.assertEqual(np_ngram_ids[1, 0], ids[1, 0])
    self.assertEqual(np_ngram_ids[0, 4], ids[0, 4])
    self.assertEqual(np_ngram_ids[1, 3], ids[1, 3])

  @parameterized.parameters(
      (16, 8, 32),
      (24, 4, 16),
      (32, 16, 8),
      (25, 2, 16),
  )
  def test_vq_layer_equivalence_with_tf(self, num_clusters, num_heads,
                                        dim_per_head):
    inputs = np.random.normal(1.5, 2.0, (2, 32, num_heads, dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    vq_layer_p = ngrammer.VectorQuantization.Params().Set(
        name='jax_vq_layer',
        num_clusters=num_clusters,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
    )
    vq_layer = vq_layer_p.Instantiate()
    initial_vars = vq_layer.instantiate_variables(init_key)

    jax_dists, _ = test_utils.apply(vq_layer, initial_vars, vq_layer.fprop,
                                    inputs)

    # Now run TF based computation.
    tf_vq_layer_p = attention_util.KMeansClusteringForAtten.Params().Set(
        name='tf_vq_layer',
        num_clusters=num_clusters,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        apply_layer_norm=False)
    tf_vq_layer = tf_vq_layer_p.Instantiate()
    tf_dists, _ = tf_vq_layer.FProp(initial_vars, tf.constant(inputs))
    self.assertAllClose(to_np(jax_dists), to_np(tf_dists), atol=1e-5)

  @parameterized.parameters(
      (16, 8, 2, 32, True),
      (24, 4, 4, 16, True),
      (32, 16, 1, 64, True),
      (25, 4, 2, 8, True),
      (16, 8, 2, 8, False),
      (24, 4, 4, 4, False),
      (32, 16, 1, 16, False),
      (25, 4, 2, 4, False),
  )
  def test_ngrammer_layer_exact_bigram(self, unigram_vocab_size, ngram_emb_dim,
                                       num_heads, dim_per_head, concat_ngrams):
    batch_size = 2
    seq_len = 8
    inputs = np.random.randint(
        unigram_vocab_size,
        size=[batch_size, seq_len, num_heads],
        dtype=np.int32)
    paddings = np.random.randint(1, size=[batch_size, seq_len])
    input_embs = np.random.normal(
        1.5, 2.0, (batch_size, seq_len, num_heads * dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    ngrammer_layer_p = ngrammer.Ngrammer.Params().Set(
        name='jax_ngrammer_layer',
        unigram_vocab_size=unigram_vocab_size,
        ngram_vocab_size=num_heads * unigram_vocab_size**2,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
    )
    ngrammer_layer = ngrammer_layer_p.Instantiate()
    initial_vars = ngrammer_layer.instantiate_variables(init_key)

    ngram_embs = test_utils.apply(ngrammer_layer, initial_vars,
                                  ngrammer_layer.fprop, inputs, input_embs,
                                  paddings)
    ngram_embs = np.reshape(ngram_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    input_embs = np.reshape(input_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    for i in range(num_heads):
      input_ids_per_head = inputs[:, :, i]
      ngram_ids_per_head = ngrammer.get_bigram_ids(input_ids_per_head,
                                                   unigram_vocab_size)
      ngram_ids_per_head *= (i + 1)
      ngram_ids_per_head += (i + 1)
      ngram_embs_expected = test_utils.apply(
          ngrammer_layer.ngram_table[i], initial_vars.ngram_table[i],
          ngrammer_layer.ngram_table[i].fprop,
          np.reshape(ngram_ids_per_head, [-1]))
      ngram_embs_expected = test_utils.apply(
          ngrammer_layer.ngram_layer_norm[i], initial_vars.ngram_layer_norm[i],
          ngrammer_layer.ngram_layer_norm[i].fprop, ngram_embs_expected)
      ngram_embs_expected = jnp.reshape(ngram_embs_expected,
                                        [batch_size, seq_len, ngram_emb_dim])
      ngram_embs_expected *= (1 - paddings[:, :, np.newaxis])
      if concat_ngrams:
        ngram_embs_slice = ngram_embs[:, :, i, -ngram_emb_dim:]
      else:
        input_embs_ln = test_utils.apply(ngrammer_layer.emb_layer_norm[i],
                                         initial_vars.emb_layer_norm[i],
                                         ngrammer_layer.emb_layer_norm[i].fprop,
                                         input_embs[:, :, i, :])
        ngram_embs_slice = ngram_embs[:, :, i, :] - input_embs_ln
      self.assertAllClose(to_np(ngram_embs_slice), to_np(ngram_embs_expected))

  @parameterized.parameters(
      (16, 8, 2, 32, True),
      (24, 4, 4, 16, True),
      (32, 16, 1, 64, True),
      (25, 4, 2, 8, True),
      (16, 8, 2, 8, False),
      (24, 4, 4, 4, False),
      (32, 16, 1, 16, False),
      (25, 4, 2, 4, False),
  )
  def test_ngrammer_layer_exact_bigram_2d(self, unigram_vocab_size,
                                          ngram_emb_dim, num_heads,
                                          dim_per_head, concat_ngrams):
    batch_size = 2
    seq_len = 8
    inputs = np.random.randint(
        unigram_vocab_size, size=[batch_size, seq_len], dtype=np.int32)
    paddings = np.random.randint(1, size=[batch_size, seq_len])
    input_embs = np.random.normal(
        1.5, 2.0, (batch_size, seq_len, num_heads * dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    ngrammer_layer_p = ngrammer.Ngrammer.Params().Set(
        name='jax_ngrammer_layer',
        unigram_vocab_size=unigram_vocab_size,
        ngram_vocab_size=num_heads * unigram_vocab_size**2,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
    )
    ngrammer_layer = ngrammer_layer_p.Instantiate()
    initial_vars = ngrammer_layer.instantiate_variables(init_key)

    ngram_embs = test_utils.apply(ngrammer_layer, initial_vars,
                                  ngrammer_layer.fprop, inputs, input_embs,
                                  paddings)
    ngram_embs = np.reshape(ngram_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    input_embs = np.reshape(input_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    for i in range(num_heads):
      input_ids_per_head = inputs
      ngram_ids_per_head = ngrammer.get_bigram_ids(input_ids_per_head,
                                                   unigram_vocab_size)
      ngram_ids_per_head *= (i + 1)
      ngram_ids_per_head += (i + 1)
      ngram_embs_expected = test_utils.apply(
          ngrammer_layer.ngram_table[i], initial_vars.ngram_table[i],
          ngrammer_layer.ngram_table[i].fprop,
          np.reshape(ngram_ids_per_head, [-1]))
      ngram_embs_expected = test_utils.apply(
          ngrammer_layer.ngram_layer_norm[i], initial_vars.ngram_layer_norm[i],
          ngrammer_layer.ngram_layer_norm[i].fprop, ngram_embs_expected)
      ngram_embs_expected = jnp.reshape(ngram_embs_expected,
                                        [batch_size, seq_len, ngram_emb_dim])
      ngram_embs_expected *= (1 - paddings[:, :, np.newaxis])
      if concat_ngrams:
        ngram_embs_slice = ngram_embs[:, :, i, -ngram_emb_dim:]
      else:
        input_embs_ln = test_utils.apply(ngrammer_layer.emb_layer_norm[i],
                                         initial_vars.emb_layer_norm[i],
                                         ngrammer_layer.emb_layer_norm[i].fprop,
                                         input_embs[:, :, i, :])
        ngram_embs_slice = ngram_embs[:, :, i, :] - input_embs_ln
      self.assertAllClose(to_np(ngram_embs_slice), to_np(ngram_embs_expected))

  @parameterized.parameters(
      (8, 2, 4, 32, True),
      (4, 4, 32, 16, True),
      (16, 2, 8, 64, True),
      (4, 2, 8, 8, True),
      (8, 2, 4, 8, False),
      (4, 4, 32, 4, False),
      (16, 4, 16, 16, False),
      (16, 8, 16, 16, False),
  )
  def test_vq_ngrammer_layer_exact_bigram(self, ngram_emb_dim, num_heads,
                                          num_clusters, dim_per_head,
                                          concat_ngrams):
    batch_size = 2
    seq_len = 8
    paddings = np.random.randint(1, size=[batch_size, seq_len])
    input_embs = np.random.normal(
        1.5, 2.0, (batch_size, seq_len, num_heads * dim_per_head))
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    vq_ngrammer_layer_p = ngrammer.VQNgrammer.Params().Set(
        name='jax_vq_ngrammer_layer',
        ngram_vocab_size=num_heads * num_clusters**2 + 1,
        ngram_emb_dim=ngram_emb_dim,
        num_heads=num_heads,
        num_clusters=num_clusters,
        dim_per_head=dim_per_head,
        concat_ngrams=concat_ngrams,
    )
    vq_ngrammer_layer = vq_ngrammer_layer_p.Instantiate()
    initial_vars = vq_ngrammer_layer.instantiate_variables(init_key)
    global_step = jnp.array(0, dtype=jnp.uint64)
    prng_key, compute_key = jax.random.split(prng_key)

    # compute vq ngrams function is fully functional.
    context_params = base_layer.JaxContext.Params().Set(do_eval=True)

    @jax.jit
    def compute_vq_ngrams(theta, prng_key, global_step, input_embs):
      with base_layer.JaxContext.new_context(
          params=context_params, prng_key=prng_key,
          global_step=global_step) as jax_context:
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.reset_prng_key(per_step_prng_key, global_step)
        jax_context.bind(vq_ngrammer_layer,
                         vq_ngrammer_layer.vars_to_flax_vars(theta))
        output = vq_ngrammer_layer.fprop(None, input_embs, paddings)
        distances, _ = vq_ngrammer_layer.vq_layer.fprop(input_embs)
        return output, distances

    ngram_embs, dists = compute_vq_ngrams(initial_vars, compute_key,
                                          global_step, input_embs)
    ngram_embs = np.reshape(ngram_embs,
                            [batch_size, seq_len, num_heads, dim_per_head])
    input_embs = jnp.reshape(input_embs,
                             [batch_size, seq_len, num_heads, dim_per_head])

    # [B, L, N].
    cluster_ids = jnp.argmin(dists, -1)
    for i in range(num_heads):
      input_ids_per_head = cluster_ids[:, :, i]
      ngram_ids_per_head = ngrammer.get_bigram_ids(input_ids_per_head,
                                                   num_clusters)
      ngram_ids_per_head *= (i + 1)
      ngram_ids_per_head += (i + 1)
      ngram_embs_expected = test_utils.apply(
          vq_ngrammer_layer.ngram_layer.ngram_table[i],
          initial_vars.ngram_layer.ngram_table[i],
          vq_ngrammer_layer.ngram_layer.ngram_table[i].fprop,
          np.reshape(ngram_ids_per_head, [-1]))
      ngram_embs_expected = test_utils.apply(
          vq_ngrammer_layer.ngram_layer.ngram_layer_norm[i],
          initial_vars.ngram_layer.ngram_layer_norm[i],
          vq_ngrammer_layer.ngram_layer.ngram_layer_norm[i].fprop,
          ngram_embs_expected)
      ngram_embs_expected = jnp.reshape(ngram_embs_expected,
                                        [batch_size, seq_len, ngram_emb_dim])
      ngram_embs_expected *= (1 - paddings[:, :, np.newaxis])
      if concat_ngrams:
        ngram_embs_slice = ngram_embs[:, :, i, -ngram_emb_dim:]
      else:
        input_embs_ln = test_utils.apply(
            vq_ngrammer_layer.ngram_layer.emb_layer_norm[i],
            initial_vars.ngram_layer.emb_layer_norm[i],
            vq_ngrammer_layer.ngram_layer.emb_layer_norm[i].fprop,
            input_embs[:, :, i, :])
        ngram_embs_slice = ngram_embs[:, :, i, :] - input_embs_ln
      self.assertAllClose(to_np(ngram_embs_slice), to_np(ngram_embs_expected))


if __name__ == '__main__':
  absltest.main()
