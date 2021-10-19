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
"""Tests for lingvo Jax attention layers."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import test_util
from lingvo.core import batch_major_attention
from lingvo.jax import base_layer
from lingvo.jax import test_utils
from lingvo.jax.layers import attentions
import numpy as np
import tensorflow.compat.v2 as tf


def VarStats(x):
  return np.mean(x), np.std(x)


def AssertVarStatsClose(map01, map02, test_case):

  map01_items = map01.FlattenItems()
  map02_items = map02.FlattenItems()

  def SimilarStats(x, y):
    mean1, std1 = VarStats(test_utils.ToNp(x))
    mean2, std2 = VarStats(test_utils.ToNp(y))
    delta_mean = np.abs(mean1 - mean2)
    delta_std = np.abs(std1 - std2)
    logging.info('mean1: %s, mean2: %s', mean1, mean2)
    logging.info('std1: %s, std2: %s', std1, std2)
    test_case.assertLess(delta_mean, 0.0002)
    test_case.assertLess(delta_std, 0.0002)

  for x, y in zip(map01_items, map02_items):
    assert x[0] == y[0]
    SimilarStats(x[1], y[1])


class AttentionsTest(test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def test_per_dim_scale(self):
    test_layer_p = attentions.PerDimScaleLayer.Params().Set(name='scale', dim=4)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)
    initial_vars.per_dim_scale = jnp.array([-0.5, 0.5, 1.0, 0.0],
                                           dtype=jnp.float32)
    logging.info('initial_vars: %s', initial_vars)

    inputs = np.random.normal(1.5, 2.0, [5, 4]).astype(np.float32)
    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    # comp function is fully functional.
    @jax.jit
    def Comp(theta, prng_key, global_step, inputs):
      with base_layer.JaxContext.NewContext(
          prng_key=prng_key, global_step=global_step):
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.ResetPrngKey(per_step_prng_key, global_step)
        layer.PrepareFProp()
        output = layer.FProp(theta, inputs)
        return output

    jax_out = Comp(initial_vars, compute_key, global_step, inputs)
    logging.info('jax_output: %s', jax_out)

    # Now run TF based computation.
    tf_layer_p = batch_major_attention.PerDimScaleLayer.Params().Set(
        name='scale', dim=4)
    tf_layer = tf_layer_p.Instantiate()
    tf_output1 = tf_layer.FProp(tf_layer.theta, inputs)
    logging.info('tf_output1: %s', tf_output1)
    tf_output2 = tf_layer.FProp(initial_vars, inputs)
    logging.info('tf_output2: %s', tf_output2)
    self.assertAllClose(test_utils.ToNp(jax_out), test_utils.ToNp(tf_output2))

  def test_mhd_projection_01(self):
    test_layer_p = attentions.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=16,
        num_heads=2,
        dim_per_head=5,
        is_output_projection=False)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)
    logging.info('initial_vars: %s', initial_vars)

    inputs = np.random.normal(1.5, 2.0, [5, 16]).astype(np.float32)
    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    # comp function is fully functional.
    @jax.jit
    def Comp(theta, prng_key, global_step, inputs):
      with base_layer.JaxContext.NewContext(
          prng_key=prng_key, global_step=global_step):
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.ResetPrngKey(per_step_prng_key, global_step)
        layer.PrepareFProp()
        output = layer.FProp(theta, inputs)
        return output

    jax_out = Comp(initial_vars, compute_key, global_step, inputs)
    logging.info('jax_output: %s', jax_out)

    # Now run TF based computation.
    tf_layer_p = batch_major_attention.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=16,
        num_heads=2,
        dim_per_head=5,
        is_output_projection=False)
    tf_layer = tf_layer_p.Instantiate()
    tf_output1 = tf_layer.FProp(tf_layer.theta, inputs)
    logging.info('tf_output1: %s', tf_output1)
    tf_output2 = tf_layer.FProp(initial_vars, inputs)
    logging.info('tf_output2: %s', tf_output2)
    self.assertGreater(
        np.sum(
            np.abs(test_utils.ToNp(tf_output1) - test_utils.ToNp(tf_output2))),
        0.1)
    self.assertAllClose(test_utils.ToNp(jax_out), test_utils.ToNp(tf_output2))

  def test_mhd_projection_02(self):
    test_layer_p = attentions.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=16,
        num_heads=2,
        dim_per_head=5,
        is_output_projection=True)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)
    logging.info('initial_vars: %s', initial_vars)

    inputs = np.random.normal(1.5, 2.0, [5, 2, 5]).astype(np.float32)
    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    # comp function is fully functional.
    @jax.jit
    def Comp(theta, prng_key, global_step, inputs):
      with base_layer.JaxContext.NewContext(
          prng_key=prng_key, global_step=global_step):
        per_step_prng_key = jax.random.fold_in(prng_key, global_step)
        base_layer.ResetPrngKey(per_step_prng_key, global_step)
        layer.PrepareFProp()
        output = layer.FProp(theta, inputs)
        return output

    jax_out = Comp(initial_vars, compute_key, global_step, inputs)
    logging.info('jax_output: %s', jax_out)

    # Now run TF based computation.
    tf_layer_p = batch_major_attention.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=16,
        num_heads=2,
        dim_per_head=5,
        is_output_projection=True)
    tf_layer = tf_layer_p.Instantiate()
    tf_output1 = tf_layer.FProp(tf_layer.theta, inputs)
    logging.info('tf_output1: %s', tf_output1)
    tf_output2 = tf_layer.FProp(initial_vars, inputs)
    logging.info('tf_output2: %s', tf_output2)
    self.assertGreater(
        np.sum(
            np.abs(test_utils.ToNp(tf_output1) - test_utils.ToNp(tf_output2))),
        0.1)
    self.assertAllClose(test_utils.ToNp(jax_out), test_utils.ToNp(tf_output2))

  def test_mhd_projection_var_stats(self):
    test_layer_p = attentions.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=256,
        num_heads=16,
        dim_per_head=16,
        is_output_projection=True)
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)

    # Now run TF based computation.
    tf_layer_p = batch_major_attention.MultiHeadedProjectionLayer.Params().Set(
        name='mh',
        input_dim=256,
        num_heads=16,
        dim_per_head=16,
        is_output_projection=True)
    tf_layer = tf_layer_p.Instantiate()

    tf_initial_vars = tf.nest.map_structure(lambda x: x.numpy(), tf_layer.theta)
    AssertVarStatsClose(initial_vars, tf_initial_vars, self)

  def test_mask(self):
    a = np.random.random_integers(0, 5, size=[2, 50])
    jax_mask = attentions.CausalSegmentMask(a, jnp.float32)
    tf_mask = batch_major_attention.CausalSegmentMask(a, tf.float32)
    self.assertAllClose(test_utils.ToNp(jax_mask), test_utils.ToNp(tf_mask))

  @parameterized.parameters([(False, True, 3),
                             (True, True, 3), (False, True, 4), (True, True, 4),
                             (False, False, 1), (True, False, 1),
                             (False, False, 1), (True, False, 1)])
  def test_mha_01(self, combine_qkv, dconv_qkv, dconv_kernel_size):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.MultiHeadedAttention.Params().Set(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        atten_logit_cap=20.0,
        combine_qkv=combine_qkv,
        dconv_qkv=dconv_qkv,
        dconv_kernel_size=dconv_kernel_size)
    layer = test_layer_p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)
    logging.info('initial_vars: %s', initial_vars)
    target_batch_size = 3
    source_max_length = 16
    target_max_length = 16
    initial_states = layer.InitStates(initial_vars, target_batch_size,
                                      target_max_length)
    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = query_vec
    value_vec = query_vec
    atten_mask = attentions.CausalMask(query_vec)

    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    with base_layer.JaxContext.NewContext(
        prng_key=compute_key, global_step=global_step):
      fprop_out, _ = layer.FProp(initial_vars, query_vec, key_vec, value_vec,
                                 atten_mask)

      decoder_output = jnp.zeros(
          shape=[target_max_length, target_batch_size, mdl_dim])
      atten_states = initial_states
      for t in range(target_max_length):
        if dconv_qkv:
          start = max(0, t + 1 - dconv_kernel_size)
          end = t + 1
          query_vec_prefix = query_vec[:, start:end, :]
          pad_width = dconv_kernel_size - end + start
          paddings = [(0, 0), (pad_width, 0), (0, 0)]
          query_vec_prefix = jnp.pad(query_vec_prefix, paddings)
        else:
          query_vec_prefix = query_vec[:, t, :]
        atten_states, encoded = layer.ExtendStep(
            initial_vars,
            atten_states,
            query_vec=query_vec_prefix,
            atten_mask=atten_mask[:, :, t, :],
            time_step=t)
        decoder_output = decoder_output.at[t].set(encoded)

    decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])

    logging.info('fprop_out: %s', fprop_out)
    logging.info('decoder_out: %s', decoder_output)
    self.assertAllClose(fprop_out, decoder_out_transposed)

  def test_mha_02(self):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    test_layer_p = attentions.MultiHeadedAttention.Params().Set(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        atten_logit_cap=20.0,
    )
    layer = test_layer_p.Instantiate()

    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = layer.InstantiateVariables(init_key)

    target_batch_size = 3
    source_max_length = 8
    target_max_length = 8

    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    key_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    value_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]).astype(np.float32)
    segment_ids = np.random.random_integers(
        0, 1, size=[target_batch_size, target_max_length]).astype(np.int32)
    atten_mask = attentions.CausalSegmentMask(segment_ids, np.float32)

    prng_key, compute_key = jax.random.split(prng_key)
    global_step = jnp.array(0, dtype=jnp.uint64)

    with base_layer.JaxContext.NewContext(
        prng_key=compute_key, global_step=global_step):
      jax_fprop_out, jax_atten_prob = layer.FProp(initial_vars, query_vec,
                                                  key_vec, value_vec,
                                                  atten_mask)

    tf_layer_p = batch_major_attention.MultiHeadedAttention.Params().Set(
        name='mh',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        atten_logit_cap=20.0,
        packed_input=True)
    tf_layer = tf_layer_p.Instantiate()
    tf_out, tf_atten_prob = tf_layer.FProp(
        initial_vars,
        query_vec,
        key_vec,
        value_vec,
        paddings=tf.zeros([target_batch_size, source_max_length]),
        segment_mask=atten_mask)

    logging.info('jax_layer_out: %s', jax_fprop_out)
    logging.info('jax_atten_probs: %s', jax_atten_prob)
    logging.info('tf_layer_out: %s', tf_out)
    logging.info('tf_atten_probs: %s', tf_atten_prob)
    self.assertAllClose(test_utils.ToNp(jax_fprop_out), test_utils.ToNp(tf_out))
    self.assertAllClose(
        test_utils.ToNp(jax_atten_prob), test_utils.ToNp(tf_atten_prob))

  @parameterized.parameters(
      ([1, 2, 3, 4, 5], 1, 0, [0, 1, 2, 3, 4]),
      ([1, 2, 3, 4, 5], -1, 0, [2, 3, 4, 5, 0]),
      ([1, 2, 3, 4, 5], 2, 0, [0, 0, 1, 2, 3]),
      ([1, 2, 3, 4, 5], -2, 0, [3, 4, 5, 0, 0]),
      ([[1, 2, 3, 4], [6, 7, 8, 9]], 1, 0, [[0, 0, 0, 0], [1, 2, 3, 4]]),
      ([[1, 2, 3, 4], [6, 7, 8, 9]], -1, 0, [[6, 7, 8, 9], [0, 0, 0, 0]]),
      ([[1, 2, 3, 4], [6, 7, 8, 9]], 1, 1, [[0, 1, 2, 3], [0, 6, 7, 8]]),
      ([[1, 2, 3, 4], [6, 7, 8, 9]], -1, 1, [[2, 3, 4, 0], [7, 8, 9, 0]]),
      ([1], 1, 0, [0]),
  )
  def test_shift1d(self, inputs, offset, axis, outputs):
    inputs = np.asarray(inputs)
    shift_outputs = attentions.Shift1D(inputs, offset, axis)
    self.assertArraysEqual(shift_outputs, np.asarray(outputs))

  @parameterized.parameters(
      ([8, 16, 32], 3, 1, 32),
      ([8, 8, 4, 34], 2, 0, [4, 34]),
      ([2, 32, 8, 16, 128], 3, 1, [8, 16, 128]),
  )
  def test_causal_depthwise_conv1d(self, shape, kernel_size, axis, hidden_dims):
    inputs = np.random.normal(1.5, 2.0, shape).astype(np.float32)
    p = attentions.CausalDepthwiseConv1D.Params().Set(
        name='causal_dconv', kernel_size=kernel_size, hidden_dims=hidden_dims)
    causal_dconv_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = causal_dconv_layer.InstantiateVariables(init_key)
    if isinstance(hidden_dims, list):
      kernel_shape = hidden_dims
    else:
      kernel_shape = [hidden_dims]
    for k in range(kernel_size):
      initial_vars[f'dconv_{k}'] = np.ones(kernel_shape)
    jax_dconv_out = causal_dconv_layer.FProp(initial_vars, inputs, axis=axis)
    jax_np_out = test_utils.ToNp(jax_dconv_out)
    outputs = inputs
    for _ in range(1, kernel_size):
      inputs = attentions.Shift1D(inputs, offset=1, axis=axis)
      outputs += inputs
    self.assertArraysEqual(jax_np_out, outputs)

  @parameterized.parameters(
      ([8, 16, 32], 3, 1, 32),
      ([8, 8, 4, 34], 2, 0, [4, 34]),
      ([2, 32, 8, 16, 128], 3, 1, [8, 16, 128]),
  )
  def test_causal_depthwise_conv1d_extend_step(self, shape, kernel_size, axis,
                                               hidden_dims):
    inputs = np.random.normal(1.5, 2.0, shape).astype(np.float32)
    p = attentions.CausalDepthwiseConv1D.Params().Set(
        name='causal_dconv', kernel_size=kernel_size, hidden_dims=hidden_dims)
    causal_dconv_layer = p.Instantiate()
    prng_key = jax.random.PRNGKey(seed=123)
    prng_key, init_key = jax.random.split(prng_key)
    initial_vars = causal_dconv_layer.InstantiateVariables(init_key)
    jax_dconv_out = causal_dconv_layer.FProp(initial_vars, inputs, axis=axis)
    jax_np_out = test_utils.ToNp(jax_dconv_out)
    jax_extend_step_out = jnp.zeros_like(jax_dconv_out)
    for i in range(shape[1]):
      jax_extend_step_out = causal_dconv_layer.ExtendStep(
          initial_vars, inputs, axis=axis, step=i)
      jax_np_extend_step_out = test_utils.ToNp(jax_extend_step_out)
      jax_extend_step_out_tensor = causal_dconv_layer.ExtendStep(
          initial_vars, inputs, axis=axis, step=jnp.array(i))
      jax_np_extend_step_out_tensor = test_utils.ToNp(
          jax_extend_step_out_tensor)
      jax_fprop_slice = jax.lax.dynamic_slice_in_dim(
          jax_np_out, start_index=i, slice_size=1, axis=axis)
      jax_fprop_slice = jnp.squeeze(jax_fprop_slice, axis)
      self.assertArraysEqual(jax_fprop_slice, jax_np_extend_step_out)
      self.assertArraysEqual(jax_fprop_slice, jax_np_extend_step_out_tensor)


if __name__ == '__main__':
  absltest.main()
