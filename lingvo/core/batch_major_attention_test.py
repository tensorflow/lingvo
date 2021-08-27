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
# ==============================================================================
"""Tests for batch_major_attention."""

import math
from absl.testing import flagsaver
from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import attention as tm_attention
from lingvo.core import attention_util
from lingvo.core import base_layer
from lingvo.core import batch_major_attention as attention
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import stream_step_test_base
from lingvo.core import test_utils
import numpy as np


class FAVORDotAttenTest(test_utils.TestCase, parameterized.TestCase):

  def test_favor_output(self):
    multiheadattention = attention.MultiHeadedFavorAttention.Params().Set(
        name='atten',
        input_dim=4,
        hidden_dim=4,
        enable_per_dim_scale=False,
        enable_scaling_code_motion=True,
        attention_type='softmax',
        num_random_features=1000).Instantiate()
    batch_size = 1
    length = 2
    num_heads = 1
    dim = 8
    query = tf.random.normal([batch_size, length, num_heads, dim])
    key = tf.random.normal([batch_size, length, num_heads, dim])
    value = tf.random.normal([batch_size, length, num_heads, dim])
    encoded, _ = multiheadattention._DotAtten(None, query, key, value, None,
                                              None)

    query = tf.multiply(query, 1.0 / math.sqrt(float(dim)))
    attention_scores = tf.einsum('BXHD,BYHD->BXYH', query, key)
    attention_scores = tf.nn.softmax(attention_scores, axis=2)
    exact_attention_block_output = tf.einsum('BXYH,BYHD->BXHD',
                                             attention_scores, value)
    max_error = 0.5
    with self.session(use_gpu=False) as sess:
      favor_output, groundtruth_output = sess.run(
          [exact_attention_block_output, encoded])
      error = np.max(
          np.abs((groundtruth_output - favor_output) / groundtruth_output))
      self.assertLess(error, max_error)


class MultiHeadSelfAttentionTest(test_utils.TestCase, parameterized.TestCase):
  """Test attention models."""

  def _AttentionInputs(self, input_dim=4, dtype=tf.float32):
    np.random.seed(6348575)
    batch_size = 6
    seq_len = 6
    input_vecs_p = [
        np.random.rand(seq_len, input_dim) for _ in range(batch_size)
    ]
    input_vecs = tf.stack([tf.constant(x, dtype=dtype) for x in input_vecs_p])
    # pyformat: disable
    input_padding_p = [[0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0],
                       [0, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 0],
                       [1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]]
    # pyformat: enable
    input_padding = tf.constant(input_padding_p, dtype=dtype)

    return input_vecs, input_padding, input_vecs_p, input_padding_p

  def testDotProductAttention(self):
    (input_vecs, input_padding, input_vecs_p,
     input_padding_p) = self._AttentionInputs()
    p = attention.MultiHeadedAttention.Params().Set(
        name='self_atten',
        input_dim=4,
        hidden_dim=4,
        enable_scaling_code_motion=True)
    l = p.Instantiate()

    probs, probs_sum = l.AttenProbs(
        l.theta,
        tf.expand_dims(input_vecs, 2),
        tf.expand_dims(input_vecs, 2),
        input_padding,
        segment_mask=None)

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      prob_out = sess.run(tf.squeeze(probs / probs_sum))

    # Use numpy to perform the same computation to generate expected results.
    input_vecs_p = np.array(input_vecs_p)
    target_vecs_p = np.transpose(input_vecs_p, (0, 2, 1))
    expected_logit = np.matmul(input_vecs_p, target_vecs_p)
    expected_logit = np.transpose(expected_logit, (0, 2, 1))
    elexp = np.exp(expected_logit)
    input_padding_p = np.array(input_padding_p)
    input_padding_p = np.expand_dims(input_padding_p, axis=1)
    input_padding_p = np.tile(input_padding_p, (1, 6, 1))
    elexp *= (1 - input_padding_p)
    expected_prob_out = elexp / np.expand_dims(np.sum(elexp, axis=-1), axis=-1)
    expected_prob_out = np.reshape(expected_prob_out, (6, 6, 6))
    self.assertAllClose(expected_prob_out, prob_out)

  @parameterized.parameters(1.0, 5.0, 10.0)
  def testAttenLogitCapping(self, atten_logit_cap):
    (input_vecs, input_padding, input_vecs_p,
     input_padding_p) = self._AttentionInputs()
    p = attention.MultiHeadedAttention.Params().Set(
        name='self_atten',
        input_dim=4,
        hidden_dim=4,
        enable_scaling_code_motion=True,
        atten_logit_cap=atten_logit_cap)
    l = p.Instantiate()

    probs, probs_sum = l.AttenProbs(
        l.theta,
        tf.expand_dims(input_vecs, 2),
        tf.expand_dims(input_vecs, 2),
        input_padding,
        segment_mask=None)

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      prob_out = sess.run(tf.squeeze(probs / probs_sum))

    # Use numpy to perform the same computation to generate expected results.
    input_vecs_p = np.array(input_vecs_p)
    target_vecs_p = np.transpose(input_vecs_p, (0, 2, 1))
    expected_logit = np.matmul(input_vecs_p, target_vecs_p)
    expected_logit = np.transpose(expected_logit, (0, 2, 1))
    expected_logit = atten_logit_cap * np.tanh(expected_logit / atten_logit_cap)
    elexp = np.exp(expected_logit)
    input_padding_p = np.array(input_padding_p)
    input_padding_p = np.expand_dims(input_padding_p, axis=1)
    input_padding_p = np.tile(input_padding_p, (1, 6, 1))
    elexp *= (1 - input_padding_p)
    expected_prob_out = elexp / np.expand_dims(np.sum(elexp, axis=-1), axis=-1)
    expected_prob_out = np.reshape(expected_prob_out, (6, 6, 6))
    self.assertAllClose(expected_prob_out, prob_out)

  @parameterized.named_parameters(('Two', 2), ('Three', 3))
  def testMultiHeadedProjectionLayerInputMode(self, batch_dims):
    with self.session(use_gpu=True) as sess:
      batch_sizes = list(np.arange(3, 3 + batch_dims))

      num_heads, dim_per_head = 4, 2
      model_dims = num_heads * dim_per_head

      input_tf = tf.random.normal(
          shape=batch_sizes + [model_dims], dtype=tf.float32)
      proj_p = attention.MultiHeadedProjectionLayer.Params().Set(
          input_dim=model_dims,
          num_heads=num_heads,
          dim_per_head=dim_per_head,
          is_output_projection=False,
          name='proj')

      proj = proj_p.Instantiate()
      tf.global_variables_initializer().run()
      result = proj.FPropDefaultTheta(input_tf)
      result_np = sess.run(result)
      self.assertEqual(result_np.shape,
                       tuple(batch_sizes + [num_heads, dim_per_head]))

  @parameterized.named_parameters(('Two', 2), ('Three', 3))
  def testMultiHeadedProjectionLayerOutputMode(self, batch_dims):
    with self.session(use_gpu=True) as sess:
      batch_sizes = list(np.arange(3, 3 + batch_dims))

      num_heads, dim_per_head = 4, 2
      model_dims = num_heads * dim_per_head

      input_tf = tf.random.normal(
          shape=batch_sizes + [num_heads, dim_per_head], dtype=tf.float32)

      proj_p = attention.MultiHeadedProjectionLayer.Params().Set(
          input_dim=model_dims,
          num_heads=num_heads,
          dim_per_head=dim_per_head,
          is_output_projection=True,
          name='proj')

      proj = proj_p.Instantiate()
      tf.global_variables_initializer().run()
      result = proj.FPropDefaultTheta(input_tf)
      result_np = sess.run(result)

      self.assertEqual(result_np.shape, tuple(batch_sizes + [model_dims]))

  @parameterized.named_parameters(
      # Use the default data types.
      ('dtype_default', [], 1e-06),
      # Set the post projection matrix to float16.
      ('dtype_post_float16', [('.*post/w', tf.float16)], 1e-04),
      # Set the 4 weight matrices, query, key, value and post, to float16.
      ('dtype_all_float16', [('.*w', tf.float16)], 1e-04))
  def testMultiHeadedAttentionDotProduct(self, list_regex_dtypes, atol):
    # input_batch:6, seq_len:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      input_vecs, input_padding, _, _ = self._AttentionInputs()
      p = attention.MultiHeadedAttention.Params().Set(
          name='self_atten', num_heads=2, input_dim=4, hidden_dim=4)

      # Use Gaussian() to have consistent init values for float32 and float16.
      p.params_init = py_utils.WeightInit.Gaussian(0.1)
      with py_utils.VariableListDtypeRegexScope(list_regex_dtypes):
        l = p.Instantiate()
      tf.global_variables_initializer().run()
      ctx_vec, _ = l.FProp(
          l.theta,
          input_vecs,
          input_vecs,
          input_vecs,
          input_padding,
          segment_mask=None)
      context_vec_out = sess.run(ctx_vec)
      context_vec_out = np.reshape(context_vec_out, (6, 24))
      self.assertAllClose(
          [-0.091584, 0.133402, 0.036773, -0.033578, 0.097802, 0.047879],
          np.sum(context_vec_out, axis=1),
          atol=atol)

  def testMultiHeadedCrossAttentionDotProduct(self):
    with self.session(use_gpu=True) as sess:
      input_vecs, input_padding, _, _ = self._AttentionInputs()
      # Set query input dim to 8 with value as concat of input_vecs.
      query_vecs = tf.concat([input_vecs, input_vecs], axis=-1)
      p = attention.MultiHeadedAttention.Params().Set(
          name='self_atten',
          num_heads=2,
          input_dim={
              'query': 8,
              'key': 4,
              'value': 4
          },
          hidden_dim=4)

      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

      l = p.Instantiate()
      tf.global_variables_initializer().run()
      ctx_vec, _ = l.FProp(
          l.theta,
          query_vecs,
          input_vecs,
          input_vecs,
          input_padding,
          segment_mask=None)
      context_vec_out = sess.run(ctx_vec)
      context_vec_out = np.reshape(context_vec_out, (12, 24))
      self.assertAllClose([
          11.009628, 10.825181, 12.373755, 12.3311825, 7.5814877, 7.620001,
          9.472344, 9.438789, 8.375568, 8.353212, 11.167051, 11.240829
      ], np.sum(context_vec_out, axis=1))

  def testCausalSegmentMask(self):
    # input_batch:6, seq_len:6. Test n = 2 case.
    with self.session(use_gpu=False) as sess:
      segment_ids = tf.constant([[1, 1, 1, 0]])
      mask = attention.CausalSegmentMask(segment_ids, tf.float32)
      mask_val = sess.run(mask)
      print(mask_val)
      atten_allowed = np.sum((mask_val >= 0.0).astype(np.float32))
      self.assertEqual(7.0, atten_allowed)

  def testMultiHeadedAttentionDotProductSegmentMask(self):
    # input_batch:6, seq_len:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      input_vecs, input_padding, _, _ = self._AttentionInputs()
      p = attention.MultiHeadedAttention.Params().Set(
          name='self_atten',
          num_heads=2,
          input_dim=4,
          hidden_dim=4,
          packed_input=True)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

      segment_id = tf.zeros([6, 6])
      segment_mask = attention.SegmentMask(segment_id, segment_id)
      padding = tf.tile(tf.reshape(input_padding, [6, 1, 1, 6]), [1, 1, 6, 1])
      padding_mask = padding * segment_mask.dtype.max * tf.constant(
          -0.7, dtype=segment_mask.dtype)
      segment_mask += padding_mask

      l = p.Instantiate()
      tf.global_variables_initializer().run()
      ctx_vec, _ = l.FProp(
          l.theta,
          input_vecs,
          input_vecs,
          input_vecs,
          input_padding,
          segment_mask=segment_mask)
      context_vec_out = sess.run(ctx_vec)
      context_vec_out = np.reshape(context_vec_out, (6, 24))
      self.assertAllClose(
          [27.417763, 31.783672, 19.99568, 23.907103, 21.078259, 28.429199],
          np.sum(context_vec_out, axis=1))


class MultiHeadedAttentionXLOracle:
  """Oracle layer used for computing ground truths for MultiHeadedAttention.

  Written in a non-vectorized way.
  """

  def __init__(self, u, v, pos_proj, sinusoid_emb):
    """Constructor.

    Args:
      u: A numpy ndarray of shape [N, H]
      v: A numpy ndarray of shape [N, H]
      pos_proj: A numpy ndarray of shape [embed_dim, N, H]
      sinusoid_emb: A numpy ndarray of shape [seqlen, emb_dim].
    """
    assert u.shape == v.shape
    assert u.shape == pos_proj.shape[1:]
    assert sinusoid_emb.shape[-1] == pos_proj.shape[0]
    # [N, H]
    self._u = u
    # [N, H]
    self._v = v
    # [?, N, H]
    self._pos_proj = pos_proj

    self._num_heads = u.shape[0]
    self._atten_dim = u.shape[-1]
    self._hidden_dim = u.shape[0] * u.shape[-1]
    self._sinusoid_emb = sinusoid_emb

  def _GetPositionEnc(self, tgt_t, src_t, head, seqlen):
    """Gets positional encoding.

    Args:
      tgt_t: A Python int, time step of target seq.
      src_t: A Python int, time step of source seq.
      head: A Python int, num of heads of the attention.
      seqlen: A Python int, sequence length of target/source seq.

    Returns:
      A numpy array of shape [head, emb_dim // head].
    """
    # [emb_dim]
    sinusoid_enc = self._sinusoid_emb[tgt_t - src_t + seqlen - 1]
    return np.einsum('DNH,D->NH', self._pos_proj, sinusoid_enc)[head]

  def AttenProbs(self, key, query, paddings, per_step_padding):
    """Computes attention probs in a non vectorized way.

    Args:
      key: A numpy ndarray of shape [batch, seqlen, heads, dim].
      query: A numpy ndarray of the same shape as `key`.
      paddings: A numpy ndarray of shape [batch, seqlen].
      per_step_padding: A numpy ndarray of shape [batch, seqlen, seqlen].

    Returns:
      A numpy ndarray of shape [batch, query_seqlen, key_seqlen]
    """

    assert query.ndim == 4
    assert paddings.ndim == 2
    assert key.shape == query.shape

    batch, seqlen = query.shape[:2]
    tgtlen, srclen = seqlen, seqlen
    assert query.shape[2] == self._num_heads
    assert query.shape[3] == self._atten_dim
    assert paddings.shape == query.shape[:2]

    logits = np.zeros((batch, self._num_heads, tgtlen, srclen))
    probs = np.zeros((batch, self._num_heads, tgtlen, srclen))

    def Normalize(vec):
      expx = np.exp(vec)
      expxsum = np.sum(expx, axis=-1)
      return expx / expxsum

    # [b, tgtlen, srclen]
    paddings = np.broadcast_to(
        np.reshape(paddings, (batch, 1, seqlen)), (batch, seqlen, seqlen))
    for b in range(batch):
      for h in range(self._num_heads):
        for i in range(tgtlen):
          for j in range(srclen):
            pos_enc = self._GetPositionEnc(i, j, h, seqlen)
            logits[b][h][i][j] = (
                np.dot(query[b][i][h], key[b][j][h]) +
                np.dot(query[b][i][h], pos_enc) +
                np.dot(self._u[h], key[b][j][h]) + np.dot(self._v[h], pos_enc))

          total_padding = paddings[b][i] + per_step_padding[b][i]
          logits[b][h][i] = np.where(total_padding > 0,
                                     np.finfo(np.float32).max * (-0.7),
                                     logits[b][h][i])
          probs[b][h][i] = Normalize(logits[b][h][i])
    return probs


def _AttentionInputs(input_dim=4, dtype=tf.float32, is_causal=True):
  np.random.seed(6348575)
  batch_size = 6
  seq_len = 6
  query_vec_p = [np.random.rand(seq_len, input_dim) for _ in range(batch_size)]
  query_vec_p = np.array(query_vec_p).astype(dtype.as_numpy_dtype)
  query_vec = tf.convert_to_tensor(query_vec_p)

  memory_vec_p = [np.random.rand(seq_len, input_dim) for _ in range(batch_size)]
  memory_vec_p = np.array(memory_vec_p).astype(dtype.as_numpy_dtype)
  memory_vec = tf.convert_to_tensor(memory_vec_p)
  # pyformat: disable
  paddings_p = np.array(
      [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1],
       [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1]]).astype(dtype.as_numpy_dtype)
  paddings = tf.convert_to_tensor(paddings_p)
  # causal padding.
  if is_causal:
    per_step_padding_p = [
        [0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]]
  else:
    per_step_padding_p = np.zeros((seq_len, seq_len))
  per_step_padding_p = [per_step_padding_p for _ in range(batch_size)]
  per_step_padding_p = np.array(per_step_padding_p).astype(dtype.as_numpy_dtype)
  per_step_padding = tf.convert_to_tensor(per_step_padding_p)

  # pyformat: enable
  return (query_vec, memory_vec, paddings, per_step_padding, query_vec_p,
          memory_vec_p, paddings_p, per_step_padding_p)


class MultiHeadedAttentionTest(test_utils.TestCase, parameterized.TestCase):
  """Test dot-product multiheaded attention."""

  def _AttentionExtendStepInputs(self,
                                 input_dim=4,
                                 num_heads=2,
                                 dtype=tf.float32):
    np.random.seed(6348575)
    batch_size = 6
    seq_len = 6
    query_vec_p = [np.random.rand(1, input_dim) for _ in range(batch_size)]
    query_vec = tf.stack([tf.constant(x, dtype=dtype) for x in query_vec_p])
    # pyformat: disable
    per_step_padding_p = [[0, 1, 1, 1, 1, 1]]
    per_step_padding_p = [per_step_padding_p for _ in range(batch_size)]
    # pyformat: enable
    per_step_padding = tf.stack(
        [tf.constant(x, dtype=dtype) for x in per_step_padding_p])
    source_vecs = tf.constant(
        np.random.normal(
            0.1, 0.5, [seq_len, batch_size, num_heads, input_dim // num_heads]),
        dtype=dtype)
    source_ctxs = tf.constant(
        np.random.normal(
            0.1, 0.5, [seq_len, batch_size, num_heads, input_dim // num_heads]),
        dtype=dtype)
    cached_states = py_utils.NestedMap(key=source_vecs, value=source_ctxs)
    return query_vec, cached_states, per_step_padding

  def testAttenProbs(self):
    (query_vec, key_vec, paddings, per_step_padding, query_vec_p, key_vec_p,
     paddings_p, per_step_padding_p) = _AttentionInputs()
    p = attention.MultiHeadedAttention.Params().Set(
        name='atten',
        input_dim=4,
        hidden_dim=4,
        enable_scaling_code_motion=True)
    l = p.Instantiate()
    probs, probs_sum = l.AttenProbs(
        l.theta,
        tf.expand_dims(query_vec, 2),
        tf.expand_dims(key_vec, 2),
        paddings,
        segment_mask=None,
        per_step_padding=per_step_padding)

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      prob_out = sess.run(tf.squeeze(probs / probs_sum))

    # Use numpy to perform the same computation to generate expected results.
    query_vec_p = np.array(query_vec_p)
    key_vec_p = np.array(key_vec_p)
    key_vec_p = np.transpose(key_vec_p, (0, 2, 1))
    expected_logit = np.matmul(query_vec_p, key_vec_p)
    paddings_p = np.array(paddings_p)
    paddings_p = np.expand_dims(paddings_p, axis=1)
    paddings_p = np.tile(paddings_p, (1, 6, 1))
    per_step_padding_p = np.array(per_step_padding_p)
    paddings_p = 1.0 * np.logical_or(paddings_p, per_step_padding_p)
    elexp = np.exp(expected_logit)
    elexp *= (1.0 - paddings_p)
    elexp += 1e-9
    expected_prob_out = elexp / np.expand_dims(np.sum(elexp, axis=-1), axis=-1)
    expected_prob_out = np.reshape(expected_prob_out, (6, 6, 6))
    self.assertAllClose(expected_prob_out, prob_out)

  def testCrossAttentionPaddingWithTimestamp(self):
    with self.session(use_gpu=False) as sess:
      # batch=2, max_target_len=6
      timestamp = tf.constant([[0, 1, 2, 3, 4, 4], [0, 1, 1, 2, 3, 2]],
                              dtype=tf.int32)
      # max_source_len=5
      source_paddings = tf.constant([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
                                    dtype=tf.float32)
      out_paddings = attention.CrossAttentionPaddingWithTimestamp(
          timestamp, source_paddings, 2, 1)
      paddings_val = sess.run(out_paddings)
      print(paddings_val)
      paddings_expected = tf.constant(
          [[[0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1], [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]],
           [[0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1], [1, 0, 0, 0, 1]]],
          dtype=tf.float32)
      self.assertAllEqual(paddings_val, paddings_expected)

  def testFPropCrossAttention(self):
    # input_batch:6, seq_len:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      query_vec, memory_vec, paddings, per_step_padding, _, _, _, _ = (
          _AttentionInputs())
      p = attention.MultiHeadedAttention.Params().Set(
          name='cross_atten', num_heads=2, input_dim=4, hidden_dim=4)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      tf.global_variables_initializer().run()
      ctx_vec, _ = l.FProp(
          l.theta,
          query_vec,
          memory_vec,
          memory_vec,
          paddings,
          segment_mask=None,
          per_step_padding=per_step_padding)
      context_vec_out = sess.run(ctx_vec)
      context_vec_out = np.reshape(context_vec_out, (6, 24))
      self.assertAllClose(
          [24.624561, 27.805634, 23.358835, 11.085404, 27.165989, 23.750813],
          np.sum(context_vec_out, axis=1))

  def testExtendStepAsyncTimeStepSelfAttention(self):
    use_short_seq_opt = False
    # input_batch:6, seq_len:6, query_len: 1. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      query_vec, cached_states, per_step_padding = self._AttentionExtendStepInputs(
      )
      p = attention.MultiHeadedAttention.Params().Set(
          name='atten', num_heads=2, input_dim=4, hidden_dim=4)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

      allzero_time_step = tf.constant([0] * 6)
      time_step = tf.constant([0, 1, 2, 3, 4, 5])
      l = p.Instantiate()
      tf.global_variables_initializer().run()
      ctx_vec, updated_states = l.ExtendStep(l.theta, query_vec, cached_states,
                                             None, None, per_step_padding, 0,
                                             use_short_seq_opt)
      ctx_vec_async, updated_states_async = l.ExtendStep(
          l.theta, query_vec, cached_states, None, None, per_step_padding,
          allzero_time_step, use_short_seq_opt)

      context_vec_out = sess.run(ctx_vec)
      new_source_vecs = sess.run(updated_states.key)
      context_vec_out_async = sess.run(ctx_vec_async)
      new_source_vecs_async = sess.run(updated_states_async.key)

      self.assertAllClose(
          np.sum(context_vec_out, axis=1),
          np.sum(context_vec_out_async, axis=1))
      self.assertAllClose(
          np.sum(new_source_vecs, axis=1),
          np.sum(new_source_vecs_async, axis=1))

      ctx_vec_async, updated_states_async = l.ExtendStep(
          l.theta, query_vec, cached_states, None, None, per_step_padding,
          time_step, use_short_seq_opt)
      _, updated_states_step1 = l.ExtendStep(l.theta, query_vec, cached_states,
                                             None, None, per_step_padding, 1,
                                             use_short_seq_opt)

      context_vec_out_async = sess.run(ctx_vec_async)
      new_source_vecs_async = sess.run(updated_states_async.key)

      new_source_vecs_async_step1 = sess.run(updated_states_step1.key)

      context_vec_out_async = np.reshape(context_vec_out_async, (6, 4))
      self.assertAllClose(
          [5.381485, -1.943824, 2.214111, 0.840045, -0.939259, 0.752783],
          np.sum(context_vec_out_async, axis=1))
      # Updated status are the same at step 0.
      self.assertAllClose(new_source_vecs_async[0][0], new_source_vecs[0][0])
      self.assertAllClose(new_source_vecs_async[1][1],
                          new_source_vecs_async_step1[1][1])

  def testMultipleExtendStepAsyncTimeStepSelfAttention(self):
    # input_batch:6, seq_len:6, query_len: 1. Test n = 2 case.
    num_heads, input_dim, hidden_dim, batch, seqlen = 2, 4, 4, 6, 6
    with self.session(use_gpu=True):
      tf.random.set_seed(12345)
      (query_vec, _, paddings, _, _, _, _, _) = _AttentionInputs()
      p = attention.MultiHeadedAttention.Params().Set(
          name='atten',
          num_heads=num_heads,
          input_dim=input_dim,
          hidden_dim=hidden_dim)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()

      tf.global_variables_initializer().run()

      # Verify ExtendStep() via compare N ExtendStep() with one FProp() call on
      # a seq with length N.
      per_step_padding = 1 - tf.linalg.band_part(
          tf.ones((seqlen, seqlen)), -1, 0)
      per_step_padding = tf.stack([per_step_padding] * batch)
      dims_per_head = hidden_dim // num_heads

      def _ResetCachedStates():
        cached_source_vecs = tf.constant(
            np.random.normal(0.1, 0.5,
                             [seqlen, batch, num_heads, dims_per_head]),
            dtype=tf.float32)
        cached_source_ctxs = tf.constant(
            np.random.normal(0.1, 0.5,
                             [seqlen, batch, num_heads, dims_per_head]),
            dtype=tf.float32)
        cached_states = py_utils.NestedMap(
            key=cached_source_vecs, value=cached_source_ctxs)
        return cached_states

      encoded_all = []
      cached_states = _ResetCachedStates()
      for i in range(seqlen):
        per_step_paddings = 1. - tf.cast(
            tf.sequence_mask([i + 1] * batch, seqlen), tf.float32)
        per_step_paddings = tf.expand_dims(per_step_paddings, 1)
        encoded, cached_states = l.ExtendStep(l.theta, query_vec[:, i:i + 1, :],
                                              cached_states, paddings, None,
                                              per_step_paddings, i)
        # [batch, 1, dims_per_head]
        encoded_all.append(encoded)

      encoded_all_async = []
      cached_states = _ResetCachedStates()
      for i in range(seqlen):
        # Sample 1 to batch -1 time step are synchoronized: 1 -> Seqlen
        # Sample batch, the time step are [0, 0, 0, 1, .., Seqlen-2]
        index = i - 3 if i > 2 else 0
        new_query_vec = tf.concat([
            query_vec[:(batch - 1), i:i + 1, :], query_vec[(batch - 1):,
                                                           index:index + 1, :]
        ],
                                  axis=0)
        time_step = tf.constant([i] * (batch - 1) + [index], dtype=tf.int32)
        per_step_paddings = 1. - tf.cast(
            tf.sequence_mask([i + 1] *
                             (batch - 1) + [index + 1], seqlen), tf.float32)
        per_step_paddings = tf.expand_dims(per_step_paddings, 1)
        encoded, cached_states = l.ExtendStep(l.theta, new_query_vec,
                                              cached_states, paddings, None,
                                              per_step_paddings, time_step)
        # [batch, 1, dims_per_head]
        encoded_all_async.append(encoded)
      # [batch, T, dims_per_head]
      actual_ctx_vec = tf.concat(encoded_all, axis=1)
      actual_ctx_vec_async = tf.concat(encoded_all_async, axis=1)

      self.assertAllClose(actual_ctx_vec_async.eval()[:-1],
                          actual_ctx_vec.eval()[:-1])
      # Sample batch move 3 step slower than the synchronized version.
      self.assertAllClose(actual_ctx_vec_async.eval()[-1][3:],
                          actual_ctx_vec.eval()[-1][:3])

  @parameterized.named_parameters(
      ('Short', 0.0, True, None), ('Long', 0.0, False, None),
      ('ShortSmallCap', 1.0, True, None), ('LongSmallCap', 1.0, False, None),
      ('ShortCap', 5.0, True, None), ('LongCap', 5.0, False, None),
      ('ExplicitDimPerHead', 0.0, False, 4))
  def testExtendStep(self, cap, short_seq, explicit_dim_per_head):
    num_heads, input_dim, hidden_dim, batch, seqlen = 2, 4, 4, 6, 6
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(12345)
      query_vec = tf.random.normal([batch, seqlen, input_dim])
      paddings = tf.zeros_like(query_vec[:, :, 0])
      p = attention.MultiHeadedAttention.Params().Set(
          name='atten',
          num_heads=num_heads,
          input_dim=input_dim,
          hidden_dim=hidden_dim,
          atten_logit_cap=cap)
      if explicit_dim_per_head:
        p.dim_per_head = explicit_dim_per_head
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      tf.global_variables_initializer().run()

      # Verify ExtendStep() via compare N ExtendStep() with one FProp() call on
      # a seq with length N.
      per_step_padding = 1 - tf.linalg.band_part(
          tf.ones((seqlen, seqlen)), -1, 0)
      per_step_padding = tf.stack([per_step_padding] * batch)
      expected_ctx_tensor, _ = l.FPropDefaultTheta(
          query_vec,
          query_vec,
          query_vec,
          paddings,
          segment_mask=None,
          per_step_padding=per_step_padding)

      states = l.InitStates(l.theta, batch, seqlen)
      encoded_all = []
      for i in range(seqlen):
        per_step_paddings = 1. - tf.cast(
            tf.sequence_mask([i + 1] * batch, seqlen), tf.float32)
        per_step_paddings = tf.expand_dims(per_step_paddings, 1)
        encoded, states = l.ExtendStep(l.theta, query_vec[:, i:i + 1, :],
                                       states, paddings, None,
                                       per_step_paddings, i, short_seq)
        # [batch, 1, dims_per_head]
        encoded_all.append(encoded)
      # [batch, T, dims_per_head]
      actual_ctx_tensor = tf.concat(encoded_all, axis=1)
      expected_ctx, actual_ctx = sess.run(
          [expected_ctx_tensor, actual_ctx_tensor])
    self.assertAllClose(expected_ctx, actual_ctx)


class MultiSourceMultiHeadedAttentionTest(test_utils.TestCase):

  def testAttenProbs(self):
    (query_vec, key_vec, paddings, per_step_padding, query_vec_p, key_vec_p,
     paddings_p, per_step_padding_p) = _AttentionInputs()

    # Two-source attention.
    mha_params = attention.MultiHeadedAttention.Params().Set(
        name='atten',
        input_dim=4,
        hidden_dim=4,
        enable_scaling_code_motion=True)
    atten_merger_p = tm_attention.MergerLayer.Params().Set(
        params_init=py_utils.WeightInit.Uniform(0.04),
        merger_op='concat',  # concatenate attention
        pre_proj_input_dims=[4, 4],
        pre_proj_output_dims=[4, 4])
    params = attention.MultiSourceAttention.Params().Set(
        name='two_source_atten',
        input_dim=4,
        hidden_dim=4,
        source_atten_tpls=[('src_1', mha_params),
                           ('src_2', mha_params.Copy().Set(name='atten2'))],
        primary_source_key='src_1',
        atten_merger_tpl=atten_merger_p)
    l = params.Instantiate()

    probs, probs_sum = l.AttenProbs(
        l.theta,
        tf.expand_dims(query_vec, 2),
        py_utils.NestedMap({
            'src_1': tf.expand_dims(key_vec, 2),
            'src_2': tf.expand_dims(key_vec, 2)
        }),
        py_utils.NestedMap({
            'src_1': paddings,
            'src_2': paddings
        }),
        segment_mask=None,
        per_step_padding=per_step_padding)

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      prob_out = sess.run(tf.squeeze(probs / probs_sum))

    # Use numpy to perform the same computation to generate expected results.
    query_vec_p = np.array(query_vec_p)
    key_vec_p = np.array(key_vec_p)
    key_vec_p = np.transpose(key_vec_p, (0, 2, 1))
    expected_logit = np.matmul(query_vec_p, key_vec_p)
    paddings_p = np.array(paddings_p)
    paddings_p = np.expand_dims(paddings_p, axis=1)
    paddings_p = np.tile(paddings_p, (1, 6, 1))
    per_step_padding_p = np.array(per_step_padding_p)
    paddings_p = 1.0 * np.logical_or(paddings_p, per_step_padding_p)
    elexp = np.exp(expected_logit)
    elexp *= (1.0 - paddings_p)
    elexp += 1e-9
    expected_prob_out = elexp / np.expand_dims(np.sum(elexp, axis=-1), axis=-1)
    expected_prob_out = np.reshape(expected_prob_out, (6, 6, 6))
    self.assertAllClose(expected_prob_out, prob_out)

  def testFPropCrossAttention(self):
    # input_batch:6, seq_len:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      query_vec, memory_vec, paddings, per_step_padding, _, _, _, _ = (
          _AttentionInputs())
      mha_params = attention.MultiHeadedAttention.Params().Set(
          name='cross_atten', num_heads=2, input_dim=4, hidden_dim=4)
      mha_params.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      atten_merger_p = tm_attention.MergerLayer.Params().Set(
          params_init=py_utils.WeightInit.Uniform(0.04),
          merger_op='concat',  # concatenate attention
          pre_proj_input_dims=[4, 4],
          pre_proj_output_dims=[4, 4])
      # Two-source attention.
      p = attention.MultiSourceAttention.Params().Set(
          name='two_source_atten',
          input_dim=4,
          hidden_dim=4,
          source_atten_tpls=[('src_1', mha_params),
                             ('src_2', mha_params.Copy().Set(name='atten2'))],
          primary_source_key='src_1',
          atten_merger_tpl=atten_merger_p)
      l = p.Instantiate()

      tf.global_variables_initializer().run()
      ctx_vec, _ = l.FProp(
          l.theta,
          query_vec,
          py_utils.NestedMap({
              'src_1': memory_vec,
              'src_2': memory_vec
          }),
          py_utils.NestedMap({
              'src_1': memory_vec,
              'src_2': memory_vec
          }),
          py_utils.NestedMap({
              'src_1': paddings,
              'src_2': paddings
          }),
          segment_mask=None,
          per_step_padding=per_step_padding)
      context_vec_out = sess.run(ctx_vec)
      context_vec_out = np.reshape(context_vec_out, (12, 24))
      self.assertAllClose([
          5.6162043, 5.0109887, 6.0565553, 6.0565553, 4.5718207, 5.253615,
          2.0541124, 2.490314, 6.049119, 5.5567484, 4.409875, 5.8939424
      ], np.sum(context_vec_out, axis=1))


class MultiHeadedAttentionXLTest(test_utils.TestCase, parameterized.TestCase):
  """Test dot-product multiheaded attention."""

  def _AttentionExtendStepInputs(self,
                                 input_dim,
                                 batch_size,
                                 seq_len,
                                 dtype=tf.float32):
    np.random.seed(6348575)
    query_vec_p = [
        np.random.rand(seq_len, input_dim) for _ in range(batch_size)
    ]
    query_vec = tf.stack([tf.constant(x, dtype=dtype) for x in query_vec_p])
    paddings_p = [[0] * seq_len] * batch_size
    paddings = tf.constant(paddings_p, dtype=dtype)
    return query_vec, paddings

  @parameterized.named_parameters(('OneHead', 1), ('OneHeadCausal', 1, True),
                                  ('MultiHead', 2),
                                  ('MultiHeadCausal', 2, True))
  def testAttenProbs(self, num_heads, is_causal=False):
    batch, slen = 6, 6
    atten_dim = 4
    input_dim = num_heads * atten_dim
    (input_vecs, _, input_padding, per_step_padding, input_vecs_p, _,
     input_padding_p, per_step_padding_p) = _AttentionInputs(
         input_dim=input_dim, is_causal=is_causal)
    p = attention.MultiHeadedAttentionXL.Params().Set(
        name='self_atten',
        input_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=input_dim,
        rel_pos_emb_dim=input_dim,
        enable_scaling_code_motion=True)

    l = p.Instantiate()
    query = tf.reshape(input_vecs, (batch, slen, num_heads, atten_dim))
    probs, probs_sum = l.AttenProbs(
        l.theta,
        query,
        query,
        input_padding,
        segment_mask=None,
        per_step_padding=per_step_padding)

    # [1, 2 * slen - 1]
    positions = np.expand_dims(np.arange(-(slen - 1), slen), 0)
    sinusoid_emb = l.pos_emb.FPropWithPosition(l.theta.pos_emb,
                                               tf.convert_to_tensor(positions))
    # [ 2 * slen - 1, emb_dim=input_dim]
    sinusoid_emb = tf.squeeze(sinusoid_emb, 0)

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      u, v, pos_proj = sess.run([l.vars.u, l.vars.v, l.pos_proj.vars.w])
      actual_probs = sess.run(probs / probs_sum)
      sinusoid_emb_p = sess.run(sinusoid_emb)

    # Compute ground truth with oracle class.

    # Use numpy to perform the same computation to generate expected results.
    # [B, tgt_t, H]
    input_vecs_p = np.array(input_vecs_p)
    # [B, tgt_t, N, H]
    input_vecs_p = np.reshape(input_vecs_p, (batch, slen, num_heads, atten_dim))
    input_padding_p = np.array(input_padding_p)
    oracle = MultiHeadedAttentionXLOracle(u, v, pos_proj, sinusoid_emb_p)
    expected_probs = oracle.AttenProbs(input_vecs_p, input_vecs_p,
                                       input_padding_p, per_step_padding_p)
    self.assertAllClose(expected_probs, actual_probs)

  def testFPropSelfAttention(self):
    # input_batch:6, seq_len:6. Test n = 2 case.
    with self.session(use_gpu=True) as sess:
      query_vec, _, paddings, _, _, _, _, _ = _AttentionInputs()
      num_heads, input_dim, hidden_dim = 2, 4, 4
      p = attention.MultiHeadedAttentionXL.Params().Set(
          name='self_atten',
          num_heads=num_heads,
          input_dim=input_dim,
          hidden_dim=hidden_dim,
          rel_pos_emb_dim=num_heads * hidden_dim)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

      l = p.Instantiate()
      ctx_vec, _ = l.FPropDefaultTheta(
          query_vec, query_vec, query_vec, paddings, segment_mask=None)

      tf.global_variables_initializer().run()
      context_vec_out = sess.run(ctx_vec)
      context_vec_out = np.reshape(context_vec_out, (6, 24))
      self.assertAllClose(
          [32.33513, 28.584404, 20.54517, 23.407812, 18.616188, 24.212755],
          np.sum(context_vec_out, axis=1))

  def testExtendStepAsyncTimeStepSelfAttention(self):
    num_heads, input_dim, hidden_dim, batch, seqlen = 2, 4, 4, 6, 6
    emb_dim = 4
    with self.session(use_gpu=True):
      tf.random.set_seed(12345)
      query_vec, paddings = self._AttentionExtendStepInputs(
          input_dim, batch, seqlen)
      p = attention.MultiHeadedAttentionXL.Params().Set(
          name='atten',
          num_heads=num_heads,
          input_dim=input_dim,
          hidden_dim=hidden_dim,
          rel_pos_emb_dim=emb_dim,
          random_seed=0)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()

      tf.global_variables_initializer().run()

      # Verify ExtendStep() via compare N ExtendStep() with one FProp() call on
      # a seq with length N.
      per_step_padding = 1 - tf.linalg.band_part(
          tf.ones((seqlen, seqlen)), -1, 0)
      per_step_padding = tf.stack([per_step_padding] * batch)
      dims_per_head = hidden_dim // num_heads

      def _ResetCachedStates():
        cached_source_vecs = tf.constant(
            np.random.normal(0.1, 0.5,
                             [seqlen, batch, num_heads, dims_per_head]),
            dtype=tf.float32)
        cached_source_ctxs = tf.constant(
            np.random.normal(0.1, 0.5,
                             [seqlen, batch, num_heads, dims_per_head]),
            dtype=tf.float32)
        cached_states = py_utils.NestedMap(
            key=cached_source_vecs, value=cached_source_ctxs)
        return cached_states

      encoded_all = []
      cached_states = _ResetCachedStates()
      for i in range(seqlen):
        per_step_paddings = 1. - tf.cast(
            tf.sequence_mask([i + 1] * batch, seqlen), tf.float32)
        per_step_paddings = tf.expand_dims(per_step_paddings, 1)
        encoded, cached_states = l.ExtendStep(l.theta, query_vec[:, i:i + 1, :],
                                              cached_states, paddings, None,
                                              per_step_paddings, i)
        # [batch, 1, dims_per_head]
        encoded_all.append(encoded)

      encoded_all_async = []
      cached_states = _ResetCachedStates()
      for i in range(seqlen):
        # Sample 1 to batch -1 time step are synchoronized: 1 -> Seqlen
        # Sample batch, the time step are [0, 0, 0, 1, .., Seqlen-2]
        index = i - 3 if i > 2 else 0
        new_query_vec = tf.concat([
            query_vec[:(batch - 1), i:i + 1, :], query_vec[(batch - 1):,
                                                           index:index + 1, :]
        ],
                                  axis=0)
        time_step = tf.constant([i] * (batch - 1) + [index], dtype=tf.int32)
        per_step_paddings = 1. - tf.cast(
            tf.sequence_mask([i + 1] *
                             (batch - 1) + [index + 1], seqlen), tf.float32)
        per_step_paddings = tf.expand_dims(per_step_paddings, 1)
        encoded, cached_states = l.ExtendStep(l.theta, new_query_vec,
                                              cached_states, paddings, None,
                                              per_step_paddings, time_step)
        # [batch, 1, dims_per_head]
        encoded_all_async.append(encoded)
      # [batch, T, dims_per_head]
      actual_ctx_vec = tf.concat(encoded_all, axis=1)
      actual_ctx_vec_async = tf.concat(encoded_all_async, axis=1)

      self.assertAllClose(actual_ctx_vec_async.eval()[:-1],
                          actual_ctx_vec.eval()[:-1])
      # Sample batch move 3 step slower than the synchronized version.
      self.assertAllClose(actual_ctx_vec_async.eval()[-1][3:],
                          actual_ctx_vec.eval()[-1][:3])

  def testExtendStepSelfAttention(self):
    num_heads, input_dim, hidden_dim, batch, seqlen = 2, 4, 4, 6, 6
    emb_dim = 4
    with self.session(use_gpu=True):
      tf.random.set_seed(12345)
      query_vec, paddings = self._AttentionExtendStepInputs(
          input_dim, batch, seqlen)
      p = attention.MultiHeadedAttentionXL.Params().Set(
          name='atten',
          num_heads=num_heads,
          input_dim=input_dim,
          hidden_dim=hidden_dim,
          rel_pos_emb_dim=emb_dim,
          random_seed=0)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      tf.global_variables_initializer().run()

      # Verify ExtendStep() via compare N ExtendStep() with one FProp() call on
      # a seq with length N.
      per_step_padding = 1 - tf.linalg.band_part(
          tf.ones((seqlen, seqlen)), -1, 0)
      per_step_padding = tf.stack([per_step_padding] * batch)
      expected_ctx_vec, _ = l.FPropDefaultTheta(
          query_vec,
          query_vec,
          query_vec,
          paddings,
          segment_mask=None,
          per_step_padding=per_step_padding)
      dims_per_head = hidden_dim // num_heads
      cached_source_vecs = tf.constant(
          np.random.normal(0.1, 0.5, [seqlen, batch, num_heads, dims_per_head]),
          dtype=tf.float32)
      cached_source_ctxs = tf.constant(
          np.random.normal(0.1, 0.5, [seqlen, batch, num_heads, dims_per_head]),
          dtype=tf.float32)
      cached_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_ctxs)

      encoded_all = []
      for i in range(seqlen):
        per_step_paddings = 1. - tf.cast(
            tf.sequence_mask([i + 1] * batch, seqlen), tf.float32)
        per_step_paddings = tf.expand_dims(per_step_paddings, 1)
        encoded, cached_states = l.ExtendStep(l.theta, query_vec[:, i:i + 1, :],
                                              cached_states, paddings, None,
                                              per_step_paddings, i)
        # [batch, 1, dims_per_head]
        encoded_all.append(encoded)
      # [batch, T, dims_per_head]
      actual_ctx_vec = tf.concat(encoded_all, axis=1)
      self.assertAllClose(expected_ctx_vec.eval(), actual_ctx_vec.eval())


class MultiHeadedAttentionRPEOracle:
  """Computes ground truths for MultiHeadedfAttentionRPE.

  Written in a non-vectorized way.
  """

  def __init__(self, num_heads, key_embs, value_embs):
    """Constructor.

    Args:
      num_heads: A Python int.
      key_embs: A numpy array of shape [2 * radius + 1, hidden_dim]
      value_embs: A numpy array of shape [2 * radius + 1, hidden_dim]
    """
    assert key_embs.shape == value_embs.shape
    self._num_heads = num_heads
    self._hidden_dim = key_embs.shape[-1]
    self._atten_dim = self._hidden_dim // self._num_heads
    assert self._atten_dim * self._num_heads == self._hidden_dim

    self._key_embs = np.reshape(
        key_embs, [key_embs.shape[0], self._num_heads, self._atten_dim])
    self._value_embs = np.reshape(
        value_embs, [value_embs.shape[0], self._num_heads, self._atten_dim])
    self._radius = key_embs.shape[0] // 2

  def _GetEmb(self, tgt_t, src_t, head, emb_wt):
    radius = self._radius
    distance = np.clip(src_t - tgt_t, -radius, radius)
    return emb_wt[distance][head]

  def GetKeyEmb(self, tgt_t, src_t, head):
    return self._GetEmb(tgt_t, src_t, head, self._key_embs)

  def GetValueEmb(self, tgt_t, src_t, head):
    return self._GetEmb(tgt_t, src_t, head, self._value_embs)

  def AttenProbs(self, key, query, paddings):
    assert query.ndim == 4
    assert paddings.ndim == 2
    assert key.shape == query.shape

    batch, seqlen = query.shape[:2]
    tgtlen, srclen = seqlen, seqlen
    assert query.shape[2] == self._num_heads
    assert query.shape[3] == self._atten_dim
    assert paddings.shape == query.shape[:2]

    # [B, N, T, T]
    logits = np.zeros((batch, self._num_heads, tgtlen, srclen))
    # [B, N, T, T]
    probs = np.zeros((batch, self._num_heads, tgtlen, srclen))

    paddings = np.broadcast_to(
        np.reshape(paddings, (batch, 1, 1, seqlen)),
        (batch, self._num_heads, seqlen, seqlen))

    def Normalize(vec):
      expx = np.exp(vec)
      expxsum = np.sum(expx, axis=-1)
      return expx / expxsum

    for b in range(batch):
      for h in range(self._num_heads):
        for i in range(tgtlen):
          for j in range(srclen):
            logits[b][h][i][j] = np.dot(query[b][i][h],
                                        key[b][j][h] + self.GetKeyEmb(i, j, h))
          logits[b][h][i] = np.where(paddings[b][h][i] > 0,
                                     np.finfo(np.float32).max * (-0.7),
                                     logits[b][h][i])
          probs[b][h][i] = Normalize(logits[b][h][i])
    return probs

  def AttenContext(self, probs, values):
    assert probs.ndim == 4
    assert values.ndim == 4

    assert probs.shape[0] == values.shape[0]  # batch
    assert probs.shape[1] == values.shape[2]  # head
    assert probs.shape[2] == values.shape[1]  # tgtlen
    assert probs.shape[3] == probs.shape[2]  # slen
    assert values.shape[-1] == self._atten_dim

    batch, _, tgtlen, srclen = probs.shape
    # [B, N, T, H]
    ctx = np.zeros((batch, self._num_heads, tgtlen, self._atten_dim))
    for b in range(batch):
      for h in range(self._num_heads):
        for i in range(tgtlen):
          for j in range(srclen):
            ctx[b][h][i] += probs[b][h][i][j] * (
                values[b][j][h] + self.GetValueEmb(i, j, h))
    # [B, T, N, H]
    return np.transpose(ctx, (0, 2, 1, 3))


class MultiHeadedAttentionRPETest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('OneHead', 1), ('MultiHead', 2))
  def testAttenProbs(self, num_heads):
    batch, slen = 6, 6
    atten_dim = 4
    radius = 3
    input_dim = num_heads * atten_dim
    (input_vecs, _, input_padding, _, input_vecs_p, _, input_padding_p,
     _) = _AttentionInputs(input_dim=input_dim)
    p = attention.MultiHeadedAttentionRPE.Params().Set(
        name='self_atten',
        input_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=input_dim,
        rel_pos_radius=radius,
        enable_scaling_code_motion=True)

    l = p.Instantiate()
    query = tf.reshape(input_vecs, (batch, slen, num_heads, atten_dim))
    probs, probs_sum = l.AttenProbs(
        l.theta, query, query, input_padding, segment_mask=None)

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      # [radius * 2 + 1, hidden_dim], [B, tgt_t, src_t]
      key_emb, value_emb, actual_probs = sess.run(
          [l.key_emb.vars.w, l.value_emb.vars.w, probs / probs_sum])

    oracle = MultiHeadedAttentionRPEOracle(num_heads, key_emb, value_emb)

    # Use numpy to perform the same computation to generate expected results.
    # [B, tgt_t, N, H]
    input_vecs_p = np.reshape(input_vecs_p, (batch, slen, num_heads, atten_dim))
    expected_probs = oracle.AttenProbs(input_vecs_p, input_vecs_p,
                                       input_padding_p)
    self.assertAllClose(expected_probs, actual_probs)

  @parameterized.named_parameters(('OneHead', 1), ('MultiHead', 2))
  def testAttenContext(self, num_heads):
    batch, slen = 6, 6
    atten_dim = 4
    radius = 3
    input_dim = num_heads * atten_dim
    (input_vecs, _, _, _, input_vecs_p, _, _,
     _) = _AttentionInputs(input_dim=input_dim)
    p = attention.MultiHeadedAttentionRPE.Params().Set(
        name='self_atten',
        input_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=input_dim,
        rel_pos_radius=radius)

    l = p.Instantiate()
    probs = np.random.rand(batch, num_heads, slen, slen).astype(np.float32)
    probs = np.exp(probs) / np.sum(np.exp(probs), axis=-1, keepdims=True)
    ctx = l._AttenContext(
        l.theta, tf.convert_to_tensor(probs),
        tf.reshape(input_vecs, (batch, slen, num_heads, atten_dim)))

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      key_emb, value_emb, actual_ctx = sess.run(
          [l.key_emb.vars.w, l.value_emb.vars.w, ctx])

    oracle = MultiHeadedAttentionRPEOracle(num_heads, key_emb, value_emb)

    # [B, tgt_t, N, H]
    input_vecs_p = np.reshape(input_vecs_p, (batch, slen, num_heads, atten_dim))
    expected_ctx = oracle.AttenContext(probs, input_vecs_p)
    self.assertAllClose(expected_ctx, actual_ctx)

  @parameterized.named_parameters(('OneHead', 1), ('MultiHead', 2))
  def testAttenLogitsOneStep(self, num_heads):
    batch, slen = 6, 6
    atten_dim = 4
    radius = 3
    input_dim = num_heads * atten_dim
    (input_vecs, _, _, _, _, _, _, _) = _AttentionInputs(
        input_dim=input_dim, is_causal=True)
    p = attention.MultiHeadedAttentionRPE.Params().Set(
        name='self_atten',
        input_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=input_dim,
        rel_pos_radius=radius)

    l = p.Instantiate()
    # [B, T, N, H]
    query = tf.reshape(input_vecs, (batch, slen, num_heads, atten_dim))

    # Causal self attention.
    # [B, N, T, S]
    logits = l._AttenLogits(
        l.theta,
        query,
        query,
    )

    one_step_logits = []
    # [S=T, B, N, H]
    key = tf.transpose(query, [1, 0, 2, 3])
    for i in range(slen):
      local_logits = l._AttenLogitsOneStep(l.theta, query[:, i, :, :], key, i)
      one_step_logits.append(local_logits)
    # [T, S, B, N]
    stacked_logits = tf.stack(one_step_logits)
    stacked_logits = tf.transpose(stacked_logits, [2, 3, 0, 1])

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      expected_logits, actual_logits = sess.run([logits, stacked_logits])
    self.assertAllClose(expected_logits, actual_logits)

  @parameterized.named_parameters(('OneHead', 1), ('MultiHead', 2))
  def testAttenContextsOneStep(self, num_heads):
    batch, slen = 6, 6
    atten_dim = 4
    radius = 3
    input_dim = num_heads * atten_dim
    (input_vecs, _, _, per_step_padding, _, _, _, _) = _AttentionInputs(
        input_dim=input_dim, is_causal=True)
    p = attention.MultiHeadedAttentionRPE.Params().Set(
        name='self_atten',
        input_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=input_dim,
        rel_pos_radius=radius)

    l = p.Instantiate()
    # [B, N, T, S=T]
    # Make causal attention probs.
    probs = np.random.rand(batch, num_heads, slen, slen).astype(np.float32)
    per_step_padding = 1 - np.tril(np.ones((slen, slen))).astype(np.float32)
    probs *= per_step_padding
    # Normalize
    probs = np.exp(probs) / np.sum(np.exp(probs), axis=-1, keepdims=True)

    # Causal self attention.
    # [B, N, T, S]
    ctx = l._AttenContext(
        l.theta, tf.convert_to_tensor(probs),
        tf.reshape(input_vecs, (batch, slen, num_heads, atten_dim)))

    one_step_ctx = []
    # [B, T, N, H] -> [S=T, B, N, H]
    value = tf.reshape(input_vecs, (batch, slen, num_heads, atten_dim))
    value = tf.transpose(value, [1, 0, 2, 3])
    for i in range(slen):
      # [B, N, S]
      local_prob = probs[:, :, i, :]
      # [S, B, N]
      local_prob = tf.transpose(local_prob, [2, 0, 1])
      # [B, N, H]
      local_ctx = l._AttenContextOneStep(l.theta, local_prob, value, i,
                                         atten_dim)
      one_step_ctx.append(local_ctx)
    # [T, B, N, H]
    stacked_ctx = tf.stack(one_step_ctx)
    stacked_ctx = tf.transpose(stacked_ctx, [1, 0, 2, 3])

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      expected_ctx, actual_ctx = sess.run([ctx, stacked_ctx])
    self.assertAllClose(expected_ctx, actual_ctx)


class LocalSelfAttentionTest(test_utils.TestCase, parameterized.TestCase):
  """Test local causual self attention."""

  def _LocalCasualPadding(self, b, t, l, r):
    padding = np.ones((b, t, t))
    for i in range(t):
      padding[:, i, max(0, i - l + 1):i + r + 1] = 0
    return tf.constant(padding, dtype=tf.float32)

  @parameterized.named_parameters(
      {
          'testcase_name': 'block_size_unspecified',
          'block_size': None,
          'left_context': 4,
          'right_context': 1
      }, {
          'testcase_name': 'block_size_long',
          'block_size': 5,
          'left_context': 3,
          'right_context': 4
      }, {
          'testcase_name': 'mimic_full_attention',
          'block_size': None,
          'left_context': 6,
          'right_context': 5
      }, {
          'testcase_name': 'left_context_only',
          'block_size': 3,
          'left_context': 4,
          'right_context': 0,
      }, {
          'testcase_name': 'right_context_only',
          'block_size': 4,
          'left_context': 1,
          'right_context': 4,
      }, {
          'testcase_name': 'block_longer_than_sequence',
          'block_size': 10,
          'left_context': 7,
          'right_context': 0,
      }, {
          'testcase_name': 'pos_emb_left_context_only',
          'block_size': 3,
          'left_context': 4,
          'right_context': 0,
          'pos_emb_dim': 8,
      }, {
          'testcase_name': 'pos_emb_left_and_right_context',
          'block_size': 3,
          'left_context': 4,
          'right_context': 2,
          'pos_emb_dim': 8,
      }, {
          'testcase_name': 'lite_pos_emb_left_and_right_context',
          'block_size': 3,
          'left_context': 4,
          'right_context': 2,
          'pos_emb_dim': 8,
          'skip_term_b': True,
      })
  def testFPropAgainstReference(self,
                                block_size,
                                left_context,
                                right_context,
                                pos_emb_dim=0,
                                num_heads=2,
                                input_dim=4,
                                hidden_dim=4,
                                skip_term_b=False,
                                use_additional_per_step_padding=False):
    tf.reset_default_graph()
    with self.session(use_gpu=True) as sess:
      query_vec, _, paddings, _, _, _, _, _ = _AttentionInputs(input_dim)
      if use_additional_per_step_padding:
        # Generate a random binary mask of shape [N, T, S].
        additional_per_step_padding_val = np.random.random_integers(
            low=0, high=1, size=(6, 6, 6))
        additional_per_step_padding = tf.constant(
            additional_per_step_padding_val, tf.float32)
      else:
        additional_per_step_padding = None

      # Use the reference implementation + local casual padding to verify
      # correctness.
      if pos_emb_dim == 0:
        p_cls = attention.LocalSelfAttention
        expected_p_cls = attention.MultiHeadedAttention
      else:
        p_cls = attention.LocalSelfAttentionXL
        expected_p_cls = attention.MultiHeadedAttentionXL
      p = p_cls.Params().Set(
          name='self_atten',
          num_heads=num_heads,
          input_dim=input_dim,
          hidden_dim=hidden_dim,
          block_size=block_size,
          left_context=left_context,
          right_context=right_context,
          force_consistent_probs_shape=True)
      expected_p = expected_p_cls.Params().Set(
          name='expected_self_atten',
          num_heads=num_heads,
          input_dim=input_dim,
          hidden_dim=hidden_dim)
      if pos_emb_dim != 0:
        p.rel_pos_emb_dim = pos_emb_dim
        expected_p.rel_pos_emb_dim = pos_emb_dim
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      expected_p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

      l = p.Instantiate()
      expected_l = expected_p.Instantiate()

      tf.global_variables_initializer().run()
      ctx_vec, probs = l.FProp(
          l.theta,
          query_vec,
          query_vec,
          query_vec,
          paddings,
          segment_mask=None,
          per_step_padding=additional_per_step_padding)
      context_vec_out, probs_out = sess.run([ctx_vec, probs])
      per_step_padding = self._LocalCasualPadding(6, 6, left_context,
                                                  right_context)
      if additional_per_step_padding is not None:
        per_step_padding += additional_per_step_padding
      expected_ctx_vec, expected_probs = expected_l.FProp(
          expected_l.theta, query_vec, query_vec, query_vec, paddings, None,
          per_step_padding)
      expected_context_vec_out, expected_probs_out = sess.run(
          [expected_ctx_vec, expected_probs])

      # Don't compare if the query position is padded, or if all key positions
      # are padded.
      paddings_val = sess.run(paddings)
      per_step_padding_val = sess.run(per_step_padding)
      per_step_padding_val += paddings_val[:, :, np.newaxis]
      per_step_padding_val += paddings_val[:, np.newaxis, :]

      dont_compare = np.sum(
          per_step_padding_val > 0, axis=-1) == per_step_padding_val.shape[-1]
      factor = (1 - dont_compare)[:, None, :, None]
      expected_probs_out *= factor
      probs_out *= factor
      self.assertAllClose(probs_out, expected_probs_out)
      expected_context_vec_out *= (1 - dont_compare)[..., np.newaxis]
      context_vec_out *= (1 - dont_compare)[..., np.newaxis]
      self.assertAllClose(context_vec_out, expected_context_vec_out)

  def testFPropWithDropout(self):
    with self.session(use_gpu=True) as sess:
      query_vec, _, paddings, _, _, _, _, _ = _AttentionInputs(input_dim=4)
      p = attention.LocalSelfAttention.Params().Set(
          name='self_atten',
          num_heads=2,
          input_dim=4,
          hidden_dim=4,
          block_size=2,
          left_context=2,
          right_context=0,
          atten_dropout_prob=0.3,
      )
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      tf.global_variables_initializer().run()
      ctx_vec, _ = l.FProp(
          l.theta, query_vec, query_vec, query_vec, paddings, segment_mask=None)
      ctx_vec_val = sess.run(ctx_vec)
      print(ctx_vec_val)

  def _AttentionExtendStepInputs(self,
                                 batch_size=6,
                                 input_dim=4,
                                 num_heads=2,
                                 dtype=tf.float32):
    np.random.seed(6348575)
    seq_len = 6
    query_vec_p = [np.random.rand(1, input_dim) for _ in range(batch_size)]
    query_vec = tf.stack([tf.constant(x, dtype=dtype) for x in query_vec_p])
    source_vecs = tf.constant(
        np.random.normal(
            0.1, 0.5, [seq_len, batch_size, num_heads, input_dim // num_heads]),
        dtype=dtype)
    source_ctxs = tf.constant(
        np.random.normal(
            0.1, 0.5, [seq_len, batch_size, num_heads, input_dim // num_heads]),
        dtype=dtype)
    cached_states = py_utils.NestedMap(key=source_vecs, value=source_ctxs)
    return query_vec, cached_states

  def testExtendStepSelfAttention(self):
    # input_batch:6, seq_len:6, query_len: 1. Test n = 2 case.
    batch_size = 6
    input_dim = 4
    num_heads = 2
    with self.session(use_gpu=True) as sess:
      query_vec, cached_states = (
          self._AttentionExtendStepInputs(
              batch_size=batch_size, input_dim=input_dim, num_heads=num_heads))
      p = attention.LocalSelfAttention.Params().Set(
          name='self_atten',
          num_heads=num_heads,
          input_dim=input_dim,
          hidden_dim=4,
          block_size=2,
          left_context=2,
          right_context=0,
          atten_dropout_prob=0.3,
      )
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      tf.global_variables_initializer().run()
      ctx_vec, updated_states = l.ExtendStep(
          l.theta,
          query_vec,
          cached_states,
          paddings=None,
          segment_mask=None,
          per_step_padding=None,
          time_step=3,
          use_short_seq_opt=False)
      context_vec_out = sess.run(ctx_vec)
      new_source_vecs = sess.run(updated_states.key)
      context_vec_out = np.reshape(context_vec_out, (6, 4))

      tf.logging.info(np.array_repr(np.sum(context_vec_out, axis=1)))
      self.assertAllClose(
          [3.303124, 3.90266, 2.971359, 2.486641, 3.109267, 1.54773],
          np.sum(context_vec_out, axis=1))
      new_source_vecs = np.reshape(new_source_vecs, (6, 24))
      tf.logging.info(np.array_repr(np.sum(new_source_vecs, axis=1)))
      self.assertAllClose(
          [5.135725, 1.340482, 1.065773, 4.116683, 4.928454, 3.161165],
          np.sum(new_source_vecs, axis=1))


class LocalSelfAttentionStreamStepTest(stream_step_test_base.StreamStepTestBase
                                      ):
  """Tests StreamStep()."""

  def _GetParams(self, **kwargs):
    num_heads = kwargs['num_heads']
    input_dim = kwargs['input_dim']
    hidden_dim = kwargs['hidden_dim']
    left_context = kwargs['left_context']
    right_context = kwargs['right_context']

    p_cls = kwargs.get('p_cls', attention.LocalSelfAttention)
    use_3d_recurrent_state = kwargs.get('use_3d_recurrent_state', False)
    inference_step_max_length = kwargs.get('inference_step_max_length', None)
    minimize_state_size = kwargs.get('minimize_state_size', False)

    p = p_cls.Params().Set(
        name='local_self_atten',
        num_heads=num_heads,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        left_context=left_context,
        right_context=right_context)
    if p_cls == attention.LocalSelfAttentionXL:
      p.Set(rel_pos_emb_dim=input_dim)
    p.minimize_state_size = minimize_state_size
    p.use_3d_recurrent_state = use_3d_recurrent_state
    p.inference_step_max_length = inference_step_max_length
    return p

  def _FProp(self, layer, inputs, paddings):
    return layer.FProp(layer.theta, inputs, inputs, inputs, paddings)

  def _GetFPropOutput(self, fprop_out):
    return fprop_out[0]

  @parameterized.named_parameters(
      ('Basic',),
      ('Basic3d', attention.LocalSelfAttention, False, 1, 1, True),
      ('Basic3dMin', attention.LocalSelfAttention, False, 1, 1, True, True),
      ('BasicS4', attention.LocalSelfAttention, False, 4, 4),
      ('BasicS4L8', attention.LocalSelfAttention, False, 4, 8),
      ('BasicS4L8Min', attention.LocalSelfAttention, False, 4, 8, False, True),
      ('BasicS4L83d', attention.LocalSelfAttention, False, 4, 8, True),
      ('BasicS4L83dMin', attention.LocalSelfAttention, False, 4, 8, True, True),
      ('BasicDynamic', attention.LocalSelfAttention, False, 1, None),
      ('BasicS4Dynamic', attention.LocalSelfAttention, False, 4, None),
      ('SkipNorm', attention.LocalSelfAttention, True),
      ('SkipNormS2', attention.LocalSelfAttention, True, 2, 2),
      ('SkipNormS2L3', attention.LocalSelfAttention, True, 2, 3),
      ('SkipNormDynamic', attention.LocalSelfAttention, True, 1, None),
      ('SkipNormS2Dynamic', attention.LocalSelfAttention, True, 2, None),
      ('BasicXL', attention.LocalSelfAttentionXL),
      ('BasicS4XL', attention.LocalSelfAttentionXL, False, 4, 4),
      ('BasicDynamicXL', attention.LocalSelfAttentionXL, False, 1, None),
      ('BasicS4DynamicXL', attention.LocalSelfAttentionXL, False, 4, None),
      ('SkipNormXL', attention.LocalSelfAttentionXL, True),
      ('SkipNormS2XL', attention.LocalSelfAttentionXL, True, 2, 2),
      ('SkipNormDynamicXL', attention.LocalSelfAttentionXL, True, 1, None),
      ('SkipNormS2DynamicXL', attention.LocalSelfAttentionXL, True, 2, None),
  )
  def testLeftContext(self,
                      p_cls=attention.LocalSelfAttention,
                      testonly_skip_norm_layers=False,
                      stride=1,
                      inference_step_max_length=1,
                      use_3d_recurrent_state=False,
                      minimize_state_size=False):
    tf.random.set_seed(2021)
    kwargs = dict(
        stride=stride,
        input_dim=4,
        num_heads=2,
        hidden_dim=4,
        left_context=3,
        right_context=0,
        p_cls=p_cls,
        minimize_state_size=minimize_state_size,
        use_3d_recurrent_state=use_3d_recurrent_state,
        inference_step_max_length=inference_step_max_length)
    with flagsaver.flagsaver(
        testonly_skip_norm_layers=testonly_skip_norm_layers):
      self._TestStreamStepHelper(**kwargs)

  def testRightContext(self):
    tf.random.set_seed(2021)
    kwargs = dict(
        stride=2,
        input_dim=4,
        num_heads=4,
        hidden_dim=4,
        left_context=9,
        right_context=5)
    self._TestStreamStepHelper(**kwargs)

  def testRightContextStackingLayers(self):
    tf.random.set_seed(2021)
    kwargs = dict(
        stride=2,
        input_dim=2,
        num_heads=2,
        hidden_dim=2,
        left_context=6,
        right_context=3,
        num_layers=5)
    self._TestRightContextStackingLayersHelper(**kwargs)


class RoutingAttentionTest(test_utils.TestCase, parameterized.TestCase):
  """Tests for RoutingAttention."""

  def testDotAttenSlow(self):
    batch_size = 7
    source_length = 6
    target_length = 4
    num_heads = 2
    dim_per_head = 5
    num_clusters = 3
    attention_window = 4
    q = np.random.rand(batch_size, target_length, num_heads,
                       dim_per_head).astype(np.float32)
    k = np.random.rand(batch_size, source_length, num_heads,
                       dim_per_head).astype(np.float32)
    v = np.random.rand(batch_size, source_length, num_heads,
                       dim_per_head).astype(np.float32)
    query_paddings = np.zeros([batch_size, target_length], dtype=np.float32)
    key_paddings = np.zeros([batch_size, source_length], dtype=np.float32)
    p = attention.RoutingAttention.Params().Set(
        name='routing_atten',
        input_dim=1,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads,
        num_clusters=num_clusters,
        attention_window=attention_window,
        fast_path=False)
    atten = p.Instantiate()
    with self.session() as sess:
      tf.global_variables_initializer().run()
      encoded, probs = sess.run(
          atten._DotAtten(
              atten.theta, q, k, v, key_paddings,
              query_paddings=query_paddings))
      self.assertEqual(encoded.shape,
                       (batch_size, target_length, num_heads, dim_per_head))
      self.assertEqual(probs.shape,
                       (batch_size, num_heads, target_length, source_length))
      # attention weights sum to 1.
      self.assertAllClose(
          np.sum(probs, axis=-1),
          np.ones([batch_size, num_heads, target_length]))

  def testDotAttenFast(self):
    batch_size = 6
    source_length = 8
    target_length = 7
    num_heads = 3
    dim_per_head = 5
    num_clusters = 2
    attention_window = source_length
    q = np.random.rand(batch_size, target_length, num_heads,
                       dim_per_head).astype(np.float32)
    k = np.random.rand(batch_size, source_length, num_heads,
                       dim_per_head).astype(np.float32)
    v = np.random.rand(batch_size, source_length, num_heads,
                       dim_per_head).astype(np.float32)

    q_paddings = np.zeros([batch_size, target_length], dtype=np.float32)
    k_paddings = np.zeros([batch_size, source_length], dtype=np.float32)
    p = attention.RoutingAttention.Params().Set(
        name='routing_atten',
        input_dim=1,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads,
        num_clusters=num_clusters,
        attention_window=attention_window,
        query_group_size_factor=1.5,  # each group has 6 queries: 8 / 2 * 1.5.
        fast_path=True)
    atten = p.Instantiate()
    # increase group size to 7.
    atten2 = p.Copy().Set(
        name='increase_group_size_routing_atten',
        query_group_size_factor=1.75).Instantiate()
    p = attention.MultiHeadedAttention.Params().Set(
        name='full_atten',
        input_dim=1,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads)
    full_atten = p.Instantiate()
    with self.session() as sess:
      tf.global_variables_initializer().run()
      encoded, probs = sess.run(
          atten._DotAtten(
              atten.theta, q, k, v, k_paddings, query_paddings=q_paddings))
      self.assertEqual(encoded.shape,
                       (batch_size, target_length, num_heads, dim_per_head))
      self.assertEqual(probs.shape,
                       (batch_size, num_heads, target_length, source_length))
      _, probs2 = sess.run(
          atten2._DotAtten(
              atten2.theta, q, k, v, k_paddings, query_paddings=q_paddings))
      # In order to match the full attention, we apply layer norm first.
      q_ln = attention_util.KMeansClusteringForAtten.LayerNorm(q)
      k_ln = attention_util.KMeansClusteringForAtten.LayerNorm(k)
      full_encoded_t, full_probs_t = full_atten._DotAtten(
          full_atten.theta, q_ln, k_ln, v, k_paddings, None)
      full_probs, full_encoded = sess.run([full_probs_t, full_encoded_t])

    # When we increase p.query_group_size_factor, the number of left out queries
    # decreases.
    self.assertLess(np.sum(probs), np.sum(probs2))
    for batch_idx in range(batch_size):
      for time_idx in range(target_length):
        for head_idx in range(num_heads):
          sub_probs = probs[batch_idx, head_idx, time_idx, :]
          sub_encoded = encoded[batch_idx, time_idx, head_idx, :]
          # encoded output is either 0 or matching full attention output
          # for each query position.
          if np.allclose(sub_probs, np.zeros_like(sub_probs)):
            self.assertAllClose(sub_encoded, np.zeros_like(sub_encoded))
            continue
          self.assertAllClose(sub_probs, full_probs[batch_idx, head_idx,
                                                    time_idx, :])
          self.assertAllClose(sub_encoded, full_encoded[batch_idx, time_idx,
                                                        head_idx, :])

  @parameterized.parameters((False, 0), (False, 1), (False, 2), (True, 0),
                            (True, 1), (True, 2))
  def testDotAttenFull(self, fast_path, num_padded):
    batch_size = 2
    source_length = 5
    target_length = 6
    num_heads = 2
    dim_per_head = 5
    # fast_path=True with multiple clusters might leave out some queries.
    # For the purpose of this test we only use a single cluster.
    num_clusters = 1 if fast_path else 3
    attention_window = source_length
    q = tf.random.normal(
        shape=[batch_size, target_length, num_heads, dim_per_head])
    k = tf.random.normal(
        shape=[batch_size, source_length, num_heads, dim_per_head])
    v = tf.random.normal(
        shape=[batch_size, source_length, num_heads, dim_per_head])

    q_paddings = np.zeros([batch_size, target_length], dtype=np.float32)
    k_paddings = np.zeros([batch_size, source_length], dtype=np.float32)
    if num_padded:
      # randomly pad elements.
      for i in range(batch_size):
        zero_index = np.random.choice(source_length, num_padded, False)
        for j in zero_index:
          k_paddings[i, j] = 1.
    p = attention.RoutingAttention.Params().Set(
        name='routing_atten',
        input_dim=1,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads,
        num_clusters=num_clusters,
        attention_window=attention_window,
        query_group_size_factor=1.0,
        fast_path=fast_path)
    atten = p.Instantiate()
    p = attention.MultiHeadedAttention.Params().Set(
        name='full_atten',
        input_dim=1,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads)
    full_atten = p.Instantiate()
    with self.session() as sess:
      tf.global_variables_initializer().run()
      encoded_t, probs_t = atten._DotAtten(
          atten.theta, q, k, v, k_paddings, query_paddings=q_paddings)
      gradients_t = tf.gradients(encoded_t, [q, k, v])
      # In order to match the full attention, we apply layer norm first.
      q_ln = attention_util.KMeansClusteringForAtten.LayerNorm(q)
      k_ln = attention_util.KMeansClusteringForAtten.LayerNorm(k)
      full_encoded_t, full_probs_t = full_atten._DotAtten(
          full_atten.theta, q_ln, k_ln, v, k_paddings, None)
      full_gradients_t = tf.gradients(full_encoded_t, [q, k, v])
      (encoded, probs, full_encoded, full_probs, gradients,
       full_gradients) = sess.run([
           encoded_t, probs_t, full_encoded_t, full_probs_t, gradients_t,
           full_gradients_t
       ])
      self.assertAllClose(probs, full_probs)
      self.assertAllClose(encoded, full_encoded)
      # The 3 gradients (dq, dk, dv) should also match
      self.assertAllClose(gradients, full_gradients)

  @parameterized.parameters(False, True)
  def testDotAttenCausalMasking(self, fast_path):
    batch_size = 3
    seq_length = 12
    num_heads = 2
    dim_per_head = 4
    num_clusters = 1 if fast_path else 3
    attention_window = seq_length
    q = np.random.rand(batch_size, seq_length, num_heads,
                       dim_per_head).astype(np.float32)
    k = np.random.rand(batch_size, seq_length, num_heads,
                       dim_per_head).astype(np.float32)
    v = np.random.rand(batch_size, seq_length, num_heads,
                       dim_per_head).astype(np.float32)

    q_paddings = np.zeros([batch_size, seq_length], dtype=np.float32)
    k_paddings = np.zeros([batch_size, seq_length], dtype=np.float32)
    p = attention.RoutingAttention.Params().Set(
        name='routing_atten',
        input_dim=1,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads,
        num_clusters=num_clusters,
        attention_window=attention_window,
        causal_masking=True,
        query_group_size_factor=1.0,
        fast_path=fast_path)
    atten = p.Instantiate()
    p = attention.MultiHeadedAttention.Params().Set(
        name='full_atten',
        input_dim=1,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads)
    full_atten = p.Instantiate()
    with self.session() as sess:
      tf.global_variables_initializer().run()
      encoded, probs = sess.run(
          atten._DotAtten(
              atten.theta, q, k, v, k_paddings, query_paddings=q_paddings))
      # In order to match the full attention, we apply layer norm first.
      q_ln = attention_util.KMeansClusteringForAtten.LayerNorm(q)
      k_ln = attention_util.KMeansClusteringForAtten.LayerNorm(k)
      # Manually apply causal padding to full attention.
      per_step_padding = tf.tile(
          tf.expand_dims(
              attention.CausalPadding(seq_length, dtype=q_ln.dtype), 0),
          [batch_size, 1, 1])
      full_encoded, full_probs = full_atten._DotAtten(
          full_atten.theta,
          q_ln,
          k_ln,
          v,
          k_paddings,
          segment_mask=None,
          per_step_padding=per_step_padding)
      self.assertAllClose(probs, full_probs.eval())
      self.assertAllClose(encoded, full_encoded.eval())

    # Verify that the first token only attends to position 0.
    first_token_probs = probs[:, :, 0, :]
    expected = np.zeros_like(first_token_probs)
    expected[:, :, 0] = 1.
    self.assertAllClose(first_token_probs, expected)

  @parameterized.parameters(False, True)
  def testSelfAtten(self, fast_path):
    batch_size = 4
    target_length = 8
    num_heads = 4
    dim_per_head = 5
    num_clusters = 3
    attention_window = 6
    q = tf.random.normal(
        shape=[batch_size, target_length, num_heads, dim_per_head])
    v = tf.random.normal(
        shape=[batch_size, target_length, num_heads, dim_per_head])
    q_copy = tf.identity(q)

    paddings = np.zeros([batch_size, target_length], dtype=np.float32)
    p = attention.RoutingAttention.Params().Set(
        name='routing_atten',
        input_dim=1,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads,
        num_clusters=num_clusters,
        attention_window=attention_window,
        query_group_size_factor=1.0,
        fast_path=fast_path)
    atten = p.Instantiate()
    with self.session() as sess, self.SetEval(True):
      tf.global_variables_initializer().run()
      # self attention path
      encoded_self_t, probs_self_t = atten._DotAtten(
          atten.theta, q, q, v, paddings, query_paddings=paddings)
      # computed as cross attention
      encoded_t, probs_t = atten._DotAtten(
          atten.theta, q, q_copy, v, paddings, query_paddings=paddings)
      encoded, probs, encoded_self, probs_self = sess.run(
          [encoded_t, probs_t, encoded_self_t, probs_self_t])
      self.assertAllClose(probs, probs_self)
      self.assertAllClose(encoded, encoded_self)

  def testExtendStep(self):
    batch_size = 8
    target_length = 10
    num_heads = 4
    dim_per_head = 5
    num_clusters = 6
    attention_window = target_length
    input_dim = 7
    q = np.random.rand(batch_size, target_length, input_dim).astype(np.float32)
    paddings = np.zeros([batch_size, target_length], dtype=np.float32)
    p = attention.RoutingAttention.Params().Set(
        name='routing_atten',
        input_dim=input_dim,
        hidden_dim=num_heads * dim_per_head,
        num_heads=num_heads,
        num_clusters=num_clusters,
        attention_window=attention_window,
        causal_masking=True,
        fast_path=False)
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
    atten = p.Instantiate()
    # We ensure that the encoded attention result is the same between FProp()
    # and sequential calls to ExtendStep().
    with self.session() as sess:
      # self attention path via ExtendStep
      encoded_all = []
      states = atten.InitStates(atten.theta, batch_size, target_length)
      self.assertEqual(states.key.shape,
                       (target_length, batch_size, num_heads, dim_per_head))
      self.assertEqual(states.value.shape,
                       (target_length, batch_size, num_heads, dim_per_head))
      self.assertEqual(states.key_dists.shape,
                       (target_length, batch_size, num_heads, num_clusters))
      for i in range(target_length):
        encoded, states = atten.ExtendStep(atten.theta, q[:, i:i + 1, :],
                                           states, paddings, i)
        self.assertEqual(encoded.shape, (batch_size, 1, input_dim))
        encoded_all.append(encoded)
      encoded_extend_t = tf.concat(encoded_all, axis=1)

      # self attention path via FProp
      encoded_fprop_t, _ = atten.FProp(atten.theta, q, q, q, paddings)
      self.assertEqual(encoded_fprop_t.shape,
                       (batch_size, target_length, input_dim))

      tf.global_variables_initializer().run()
      encoded_extend, encoded_fprop = sess.run(
          [encoded_extend_t, encoded_fprop_t])
      self.assertAllClose(encoded_extend, encoded_fprop)


class TransformerAttentionLayerTest(test_utils.TestCase,
                                    parameterized.TestCase):
  """Tests for TransformerAttentionLayer."""

  @parameterized.named_parameters(
      ('Basic',),
      ('BasicR1', False, 1, None, 1),
      ('BasicS4', False, 4, 4),
      ('BasicS4L8', False, 4, 8),
      ('SkipNorm', True),
      ('SkipNormS2', True, 2, 2),
      ('SkipNormS2L3', True, 2, 3),
      ('SkipNormS4R2', True, 4, None, 2),
  )
  def testStreamStep(self,
                     testonly_skip_norm_layers=False,
                     stride=1,
                     inference_step_max_length=1,
                     right_context=0):
    with flagsaver.flagsaver(
        testonly_skip_norm_layers=testonly_skip_norm_layers):
      self._TestStreamStepHelper(stride, inference_step_max_length,
                                 right_context)

  def _TestStreamStepHelper(self, stride, inference_step_max_length,
                            right_context):
    batch_size, max_seqlen, input_dim = 2, 32, 4
    num_heads = 2
    left_context = 3

    # Prepares inputs.
    np.random.seed(None)
    inputs = np.random.normal(
        0.5, 1, [batch_size, max_seqlen, input_dim]).astype(np.float32)
    print(f'np.sum(inputs): {np.sum(inputs)}')
    inputs = tf.convert_to_tensor(inputs)

    seqlen = np.random.randint(
        low=max_seqlen // 2,
        high=max_seqlen + 1,
        size=(batch_size,),
        dtype=np.int32)
    print(f'seqlen: {repr(seqlen)}')
    seqlen = tf.convert_to_tensor(seqlen)
    paddings = py_utils.PaddingsFromLengths(seqlen, max_seqlen)

    # Builds graph.
    p = attention.TransformerAttentionLayer.CommonParams(
        input_dim=input_dim,
        num_heads=num_heads,
        is_masked=True,
        left_context=left_context,
        right_context=right_context)
    p.name = 'transformer_atten'
    p.atten_tpl.inference_step_max_length = inference_step_max_length
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

    l = p.Instantiate()
    init_op = tf.global_variables_initializer()

    base_outputs, _ = l.FProp(l.theta, inputs, None, paddings)
    base_outputs *= tf.reshape(1. - paddings, [batch_size, max_seqlen, 1])

    state = l.zero_state(batch_size)
    outputs = []
    assert max_seqlen % stride == 0
    for i in range(max_seqlen // stride +
                   int(math.ceil(right_context / stride))):
      if i < max_seqlen // stride:
        step_inputs = inputs[:, stride * i:stride * (i + 1)]
        step_paddings = paddings[:, stride * i:stride * (i + 1)]
      else:
        step_inputs = tf.zeros_like(inputs[:, 0:stride])
        step_paddings = tf.ones_like(paddings[:, 0:stride])
      output, _, state = l.StreamStep(l.theta, step_inputs, step_paddings,
                                      state)
      outputs.append(output)

    outputs = tf.concat(outputs, axis=1)
    outputs = outputs[:, right_context:][:, :max_seqlen]
    outputs *= tf.reshape(1. - paddings, [batch_size, max_seqlen, 1])

    with self.session(use_gpu=False) as sess:
      sess.run(init_op)

      expected, actual = sess.run([base_outputs, outputs])
      print(repr(expected))
      print(repr(actual))
      print(f'np.sum(np.abs(expected)): {np.sum(np.abs(expected))}')
      print(f'np.sum(np.abs(actual)): {np.sum(np.abs(actual))}')
      self.assertAllClose(expected, actual)
      self.assertEqual(
          tuple(expected.shape), (batch_size, max_seqlen, input_dim))

  def testStreamStepDropout(self):
    batch_size, input_dim, num_heads, stride, left_context = 2, 4, 2, 8, 3

    # Prepares inputs.
    np.random.seed(None)
    inputs = np.random.normal(0.5, 1, [batch_size, stride, input_dim]).astype(
        np.float32)
    print(f'np.sum(inputs): {np.sum(inputs)}')
    inputs = tf.convert_to_tensor(inputs)

    seqlen = np.random.randint(
        low=4, high=stride + 1, size=(batch_size,), dtype=np.int32)
    seqlen = tf.convert_to_tensor(seqlen)
    paddings = py_utils.PaddingsFromLengths(seqlen, stride)

    # Builds graph.
    p = attention.TransformerAttentionLayer.CommonParams(
        input_dim=input_dim,
        num_heads=num_heads,
        is_masked=True,
        left_context=left_context,
        right_context=0,
        dropout_prob=0.5)
    p.name = 'transformer_atten'
    p.atten_tpl.inference_step_max_length = stride
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)

    l = p.Instantiate()
    output, _, _ = l.StreamStep(l.theta, inputs, paddings,
                                l.zero_state(batch_size))
    output *= tf.reshape(1. - paddings, [batch_size, stride, 1])
    init_op = tf.global_variables_initializer()

    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      res = []
      for _ in range(2):
        out = sess.run([output])
        res.append(out)
      self.assertNotAllClose(res[0], res[1])


class TransformerLayerTest(test_utils.TestCase, parameterized.TestCase):
  """Test Transformer decoder layers."""

  def _TransformerAttentionLayerInputs(self, input_dim=4, dtype=tf.float32):
    np.random.seed(6348575)
    query_vec = tf.transpose(
        tf.stack([
            tf.constant(np.random.rand(2, input_dim), dtype=dtype)
            for _ in range(5)
        ]), [1, 0, 2])
    paddings = tf.constant([[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]], dtype=dtype)
    aux_vec = tf.transpose(
        tf.stack([
            tf.constant(np.random.rand(2, input_dim), dtype=dtype)
            for _ in range(7)
        ]), [1, 0, 2])
    aux_paddings = tf.constant([[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]],
                               dtype=dtype)
    return query_vec, paddings, aux_vec, aux_paddings

  def testTransformerAttentionLayerFPropMaskedSelfAttention(self):
    with self.session(use_gpu=True) as sess:
      query_vec, paddings, _, _ = self._TransformerAttentionLayerInputs()

      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_masked_self_atten',
          input_dim=4,
          is_masked=True,
          num_heads=2)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(l.theta, query_vec, None, paddings)

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (10, 4))
      tf.logging.info(np.array_repr(actual_ctx))
      expected_ctx = [7.777687, 5.219166, 6.305151, 4.817311]
      self.assertAllClose(expected_ctx, np.sum(actual_ctx, axis=0))

  def testTransformerAttentionLayerMaskedSelfAttentionMixHeads(self):
    p = attention.TransformerAttentionLayer.Params().Set(
        name='transformer_masked_self_atten',
        input_dim=16,
        is_masked=True,
        num_heads=[4, 4])
    p.atten_tpl = [
        attention.LocalSelfAttention.Params().Set(
            left_context=2, right_context=2, block_size=4),
        attention.RoutingAttention.Params().Set(
            num_clusters=1, attention_window=2)
    ]
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
    l = p.Instantiate()
    self.assertIsInstance(l.atten[0], attention.LocalSelfAttention)
    self.assertIsInstance(l.atten[1], attention.RoutingAttention)

  def testTransformerAttentionLayerFPropMultiHeadedAttentionMixHeads(self):
    with self.session(use_gpu=True) as sess:
      query_vec, paddings, _, _ = self._TransformerAttentionLayerInputs()

      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_masked_self_atten_mix',
          input_dim=4,
          is_masked=True,
          num_heads=[2])
      p.atten_tpl = [attention.MultiHeadedAttention.Params().Set()]
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(l.theta, query_vec, None, paddings)

      p2 = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_masked_self_atten',
          input_dim=4,
          is_masked=True,
          num_heads=2)
      p2.atten_tpl = attention.MultiHeadedAttention.Params().Set()
      p2.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l2 = p2.Instantiate()
      ctx_vec2, _ = l2.FProp(l2.theta, query_vec, None, paddings)

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx2 = sess.run(ctx_vec2)
      self.assertAllClose(actual_ctx, actual_ctx2)

  def testTransformerAttentionLayerFPropMaskedSelfAttentionMixHeads(self):
    with self.session(use_gpu=True) as sess:
      query_vec, paddings, _, _ = self._TransformerAttentionLayerInputs()

      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_masked_self_atten',
          input_dim=4,
          hidden_dim=8,
          is_masked=True,
          num_heads=[2, 3])
      p.atten_tpl = [
          attention.MultiHeadedAttention.Params().Set(),
          attention.MultiHeadedAttention.Params().Set()
      ]
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(l.theta, query_vec, None, paddings)

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (10, 4))
      tf.logging.info(np.array_repr(actual_ctx))
      expected_ctx = [12.3041725, 5.4454093, 1.684509, 10.300517]
      self.assertAllClose(expected_ctx, np.sum(actual_ctx, axis=0))

  def testAttentionLayerFPropMaskedSelfAttentionPaddingOverride(self):
    with self.session(use_gpu=True) as sess:
      query_vec, paddings, _, _ = self._TransformerAttentionLayerInputs()

      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_masked_self_atten',
          input_dim=4,
          is_masked=True,
          num_heads=2)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      triangle_padding = 1.0 - tf.linalg.band_part(
          tf.ones([5, 5], dtype=query_vec.dtype), -1, 0)
      per_step_padding_override = tf.tile(
          tf.expand_dims(triangle_padding, 0), [2, 1, 1])

      ctx_vec1, _ = l.FProp(l.theta, query_vec, None, paddings,
                            per_step_padding_override)
      expected_ctx1, _ = l.FProp(l.theta, query_vec, None, paddings)
      per_step_padding_override = tf.zeros([2, 5, 5])
      ctx_vec2, _ = l.FProp(l.theta, query_vec, None, paddings,
                            per_step_padding_override)

      tf.global_variables_initializer().run()
      actual_ctx1, actual_ctx2, actual_expected_ctx1 = sess.run(
          [ctx_vec1, ctx_vec2, expected_ctx1])
      tf.logging.info(np.array_repr(actual_ctx1))
      tf.logging.info(np.array_repr(actual_ctx2))
      expected_ctx2 = [7.9491496, 5.2976646, 6.5383415, 5.0169916]
      self.assertAllClose(actual_expected_ctx1, ctx_vec1)
      self.assertAllClose(expected_ctx2,
                          np.sum(np.reshape(actual_ctx2, (10, 4)), axis=0))

  def testTransformerAttentionLayerFPropCrossAttention(self):
    with self.session(use_gpu=True) as sess:
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_cross_atten',
          input_dim=4,
          is_masked=False,
          num_heads=2)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(l.theta, query_vec, aux_vec, aux_paddings)

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (10, 4))
      expected_ctx = [19.345360, 15.057412, 13.744134, 13.387347]
      self.assertAllClose(expected_ctx, np.sum(actual_ctx, axis=0))

  def testTransformerAttentionLayerFPropCrossAttentionPaddingOverride(self):
    # We use self-attention to verify cross-attention padding works correctly.
    with self.session(use_gpu=True) as sess:
      query_vec, _, _, _ = self._TransformerAttentionLayerInputs()
      paddings = tf.convert_to_tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]],
                                      dtype=tf.float32)

      # Setup LocalSelfAttention.
      self_atten_tpl = attention.LocalSelfAttention.Params().Set(
          left_context=2, right_context=1)
      p1 = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_self_atten',
          input_dim=4,
          is_masked=False,
          num_heads=2,
          atten_tpl=self_atten_tpl)
      p1.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l1 = p1.Instantiate()

      # Setup MultiHeadedAttention.
      source_atten_tpl = attention.MultiHeadedAttention.Params()
      p2 = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_cross_atten',
          input_dim=4,
          is_masked=False,
          num_heads=2,
          atten_tpl=source_atten_tpl)
      l2 = p2.Instantiate()

      # LocalSelfAttention FProp
      self_ctx_vec, _ = l1.FProp(l1.theta, query_vec, query_vec, paddings)

      # timestamp includes valid indices to source.
      timestamp = tf.convert_to_tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 0]],
                                       dtype=tf.int32)
      per_step_padding = attention.CrossAttentionPaddingWithTimestamp(
          timestamp, paddings, left_context=2, right_context=1)
      # MultiHeadedAttention FProp with same theta and per_step_padding.
      cross_ctx_vec, _ = l2.FProp(
          l1.theta,
          query_vec,
          query_vec,
          paddings,
          per_step_padding_override=per_step_padding)

      tf.global_variables_initializer().run()
      act_self_ctx, act_cross_ctx = sess.run([self_ctx_vec, cross_ctx_vec])
      # They can only differ in padded output positions.
      self.assertAllClose(act_self_ctx[0, :4, :], act_cross_ctx[0, :4, :])
      self.assertAllClose(act_self_ctx[1, :3, :], act_cross_ctx[1, :3, :])

  def testTransformerAttentionLayerFPropCrossAttentionInputDimAsDict(self):
    with self.session(use_gpu=True) as sess:
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_cross_atten',
          input_dim={
              'query': 4,
              'key': 4,
              'value': 4
          },
          is_masked=False,
          num_heads=2)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(l.theta, query_vec, aux_vec, aux_paddings)

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (10, 4))
      expected_ctx = [19.345360, 15.057412, 13.744134, 13.387347]
      self.assertAllClose(expected_ctx, np.sum(actual_ctx, axis=0))

  def testMultiSourceTransformerAttentionLayerFPropCrossAttention(self):
    with self.session(use_gpu=True) as sess:
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      p = attention.TransformerMultiSourceAttentionLayer.Params().Set(
          name='transformer_multi_source_cross_atten',
          input_dim=4,
          is_masked=False,
          num_heads=2,
          num_source=2)
      p.multi_source_atten.atten_merger_tpl = (
          tm_attention.MergerLayer.Params().Set(merger_op='sum'))
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(
          l.theta, query_vec,
          py_utils.NestedMap({
              'source_0': aux_vec,
              'source_1': aux_vec
          }),
          py_utils.NestedMap({
              'source_0': aux_paddings,
              'source_1': aux_paddings
          }))

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (10, 4))
      tf.logging.info(np.array_repr(actual_ctx))
      expected_ctx = [32.4878, 25.145725, 21.534966, 22.007454]
      self.assertAllClose(expected_ctx, np.sum(actual_ctx, axis=0))

  @parameterized.named_parameters(
      {
          'testcase_name': '_short_seq',
          'use_short_seq_opt': True,
      }, {
          'testcase_name': '_long_seq',
          'use_short_seq_opt': False,
      })
  def testTransformerAttentionLayerExtendStep(self, use_short_seq_opt):
    with self.session(use_gpu=True) as sess:
      query_vec, _, _, _ = self._TransformerAttentionLayerInputs()
      paddings = tf.zeros([2, 5])
      cached_key = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      cached_value = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      prefix_states = py_utils.NestedMap(key=cached_key, value=cached_value)

      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_atten', input_dim=4, is_masked=True, num_heads=2)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()

      ctx_vec1, _ = l.FProp(l.theta, query_vec, None, paddings)

      ctx_vec2 = []
      for i in range(5):
        ctx_vec, prefix_states = l.ExtendStep(
            l.theta, tf.expand_dims(query_vec[:, i, :], 1), prefix_states, i,
            use_short_seq_opt)
        ctx_vec2.append(tf.squeeze(ctx_vec, 1))
      ctx_vec2 = tf.transpose(tf.stack(ctx_vec2), [1, 0, 2])

      tf.global_variables_initializer().run()
      ctx1, ctx2 = sess.run([ctx_vec1, ctx_vec2])
      self.assertAllClose(ctx1, ctx2)

  @parameterized.named_parameters(
      {
          'testcase_name': '_short_seq',
          'use_short_seq_opt': True,
      }, {
          'testcase_name': '_long_seq',
          'use_short_seq_opt': False,
      })
  def testTransformerAttentionLayerExtendStepMixHeads(self, use_short_seq_opt):
    with self.session(use_gpu=True) as sess:
      query_vec, _, _, _ = self._TransformerAttentionLayerInputs()
      paddings = tf.zeros([2, 5])
      cached_key = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 1, 2]), dtype=tf.float32)
      cached_value = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 1, 2]), dtype=tf.float32)
      prefix_states = py_utils.NestedMap(key=cached_key, value=cached_value)
      prefix_states = py_utils.NestedMap(atten=[prefix_states, prefix_states])

      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_atten', input_dim=4, is_masked=True)
      p.atten_tpl = [
          attention.MultiHeadedAttention.Params().Set(),
          attention.MultiHeadedAttention.Params().Set()
      ]
      p.num_heads = [1, 1]
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()

      ctx_vec1, _ = l.FProp(l.theta, query_vec, None, paddings)

      ctx_vec2 = []
      for i in range(5):
        ctx_vec, prefix_states = l.ExtendStep(
            l.theta, tf.expand_dims(query_vec[:, i, :], 1), prefix_states, i,
            use_short_seq_opt)
        ctx_vec2.append(tf.squeeze(ctx_vec, 1))
      ctx_vec2 = tf.transpose(tf.stack(ctx_vec2), [1, 0, 2])

      tf.global_variables_initializer().run()
      ctx1, ctx2 = sess.run([ctx_vec1, ctx_vec2])
      self.assertAllClose(ctx1, ctx2)

  def testTransformerAttentionLayerNoLayernorm(self):
    """Verify if Transformer attention allows no layernorm in FProp and Extend."""
    with self.session(use_gpu=True) as sess:
      query_vec, _, _, _ = self._TransformerAttentionLayerInputs()
      paddings = tf.zeros([2, 5])
      cached_key = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      cached_value = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      prefix_states = py_utils.NestedMap(key=cached_key, value=cached_value)

      p = attention.TransformerAttentionLayer.Params().Set(
          name='transformer_atten',
          input_dim=4,
          is_masked=True,
          num_heads=2,
          ln_tpl=None)  # Set ln_tpl to None.
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()

      ctx_vec1, _ = l.FProp(l.theta, query_vec, None, paddings)

      ctx_vec2 = []
      for i in range(5):
        ctx_vec, prefix_states = l.ExtendStep(
            l.theta, tf.expand_dims(query_vec[:, i, :], 1), prefix_states, i,
            False)
        ctx_vec2.append(tf.squeeze(ctx_vec, 1))
      ctx_vec2 = tf.transpose(tf.stack(ctx_vec2), [1, 0, 2])

      tf.global_variables_initializer().run()
      ctx1, ctx2 = sess.run([ctx_vec1, ctx_vec2])
      self.assertAllClose(ctx1, ctx2)

  def _ConstructTransformerDecoderLayer(self, use_relative_atten=False):
    p = attention.TransformerDecoderLayer.Params()
    p.name = 'transformer_decoder_layer'
    p.input_dim = 4
    p.tr_fflayer_tpl.hidden_dim = 7
    p.tr_atten_tpl.num_heads = 2
    if use_relative_atten:
      p = attention.UseRelativeAttentionInTransformerLayer(p, 4)
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
    return attention.TransformerDecoderLayer(p)

  def _ConstructTransformerDecoderLayerMixHeads(self, use_relative_atten=False):
    p = attention.TransformerDecoderLayer.Params()
    p.name = 'transformer_decoder_layer'
    p.input_dim = 4
    p.tr_fflayer_tpl.hidden_dim = 7
    p.tr_atten_tpl.num_heads = [1, 1]
    p.tr_atten_tpl.atten_tpl = [
        attention.MultiHeadedAttention.Params().Set(),
        attention.MultiHeadedAttention.Params().Set()
    ]
    if use_relative_atten:
      p = attention.UseRelativeAttentionInTransformerLayer(p, 4)
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
    return attention.TransformerDecoderLayer(p)

  def testTransformerLayerCommonParams(self):
    with self.session(use_gpu=True) as sess:
      input_dim, fflayer_hidden_dim, num_heads = 4, 7, 2
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs(
           input_dim=input_dim)
      query_vec = tf.tile(query_vec, [1, 1, 1])
      paddings = tf.zeros([2, 5])
      p = attention.TransformerLayer.CommonParams(
          input_dim=input_dim,
          atten_num_heads=num_heads,
          fflayer_hidden_dim=fflayer_hidden_dim)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(l.theta, query_vec, paddings, aux_vec, aux_paddings)

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (10, 4))
      tf.logging.info(np.array_repr(actual_ctx))
      expected_ctx = [
          4.7839108, 4.5303655, 5.5551023, 5.0657663, 5.0493064, 3.2142467,
          2.820018, 5.659971, 4.3814187, 2.60475
      ]
      self.assertAllClose(expected_ctx, np.sum(actual_ctx, axis=1))

  @parameterized.named_parameters(
      ('F32FPropF32Input', tf.float32, tf.float32),
      ('F32FPropBF16Input', tf.float32, tf.bfloat16),
      ('BF16FPropF32Input', tf.bfloat16, tf.float32),
      ('BF16FPropBF16Input', tf.bfloat16, tf.bfloat16),
      ('BF16AddNormalizedInput', tf.bfloat16, tf.bfloat16, False),
  )
  def testTransformerLayerFPropDtypes(self,
                                      fprop_dtype,
                                      input_dtype,
                                      add_unnormalized_input=True):
    with self.session(use_gpu=True) as sess:
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs(dtype=input_dtype)
      paddings = tf.zeros([2, 5])
      p = attention.TransformerDecoderLayer.Params()
      p.name = 'transformer_layer'
      p.input_dim = 4
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_heads = 2
      p.tr_atten_tpl.add_unnormalized_input = add_unnormalized_input
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      p.random_seed = 1234

      p.cls.SetFPropDtype(p, fprop_dtype)
      # fprop_dtype set accordingly.
      self.assertEqual(fprop_dtype, p.fprop_dtype)

      l = p.Instantiate()
      tf.global_variables_initializer().run()

      ctx_vec, _ = l.FProp(l.theta, query_vec, paddings, aux_vec, aux_paddings)

      tgt_batch, tgt_len = py_utils.GetShape(paddings)
      with tf.name_scope('init_states'):
        prefix_states = l.InitStates(l.theta, tgt_batch, tgt_len)
      extend_step_outputs = []
      for i in range(tgt_len):
        with tf.name_scope(f'extend_step_{i}'):
          layer_output, _, prefix_states = l.ExtendStep(
              l.theta, tf.expand_dims(query_vec[:, i, :], 1), aux_vec,
              aux_paddings, prefix_states, i)
        extend_step_outputs.append(layer_output)
      extend_step_outputs = tf.concat(extend_step_outputs, axis=1)

      ctx_sum, step_sum = sess.run(
          [tf.reduce_sum(ctx_vec),
           tf.reduce_sum(extend_step_outputs)])
      self.assertAllClose(ctx_sum, step_sum)

  @parameterized.named_parameters(('SingleBatch', 1), ('DoubleBatch', 2))
  def testTransformerLayerFPropWithCrossAttention(self, multiplier):
    with self.session(use_gpu=True) as sess:
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      query_vec = tf.tile(query_vec, [multiplier, 1, 1])
      paddings = tf.zeros([2 * multiplier, 5])
      p = attention.TransformerLayer.Params()
      p.name = 'transformer_layer'
      p.input_dim = 4
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_heads = 2
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(l.theta, query_vec, paddings, aux_vec, aux_paddings)

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (10 * multiplier, 4))
      tf.logging.info(np.array_repr(actual_ctx))
      expected_ctx = [
          4.7839108, 4.5303655, 5.5551023, 5.065767, 5.0493064, 3.2142467,
          2.8200178, 5.659971, 4.3814187, 2.60475
      ] * multiplier
      self.assertAllClose(expected_ctx, np.sum(actual_ctx, axis=1))

  def testTransformerLayerDecodeWithCrossAttention(self):
    np.random.seed(6348575)
    dtype = tf.float32
    b_size = 2
    input_dim = 4
    src_seq_len = 4
    tgt_seq_len = 3
    query_vec = np.random.rand(b_size, tgt_seq_len, input_dim)
    paddings = tf.constant([[0, 0, 0], [0, 0, 0]], dtype=dtype)
    aux_vec = np.random.rand(b_size, src_seq_len, input_dim)
    aux_paddings = tf.constant([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=dtype)
    segment_mask = tf.constant(
        [[0, -1e30, -1e30], [-1e30, 0, -1e30], [0, -1e30, 0]], dtype=dtype)
    segment_mask = tf.tile(segment_mask[tf.newaxis, tf.newaxis, :, :],
                           [b_size, 1, 1, 1])
    aux_segment_mask = tf.zeros([b_size, 1, tgt_seq_len, src_seq_len])

    with self.session(use_gpu=True) as sess:
      p = attention.TransformerLayer.Params()
      p.name = 'transformer_layer'
      p.input_dim = 4
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_heads = 2
      p.mask_self_atten = True
      p.packed_input = True
      p.has_aux_atten = True
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(
          l.theta,
          query_vec,
          paddings,
          aux_vec,
          aux_paddings,
          segment_mask=segment_mask,
          aux_segment_mask=aux_segment_mask)

      cached_states = l.InitStates(l.theta, b_size, tgt_seq_len)
      extend_step_outs = []
      for t in range(tgt_seq_len):
        out_t, _, cached_states = l.ExtendStep(
            l.theta,
            query_vec[:, t:t + 1, :],
            aux_vec,
            aux_paddings,
            cached_states,
            t,
            segment_mask=segment_mask[:, :, t, :],
            aux_segment_mask=aux_segment_mask[:, :, t, :])
        extend_step_outs.append(out_t[:, 0, :])

      decoder_out = tf.stack(extend_step_outs, axis=1)

      tf.global_variables_initializer().run()
      fprop_out_v, decoder_out_v = sess.run([ctx_vec, decoder_out])
      tf.logging.info(np.array_repr(fprop_out_v))
      tf.logging.info(np.array_repr(decoder_out_v))
      self.assertAllClose(fprop_out_v, decoder_out_v)

  def testReshapedTransformerLayerFPropNoCrossAttention(self):
    with self.session(use_gpu=True) as sess:
      query_vec, _, _, _ = self._TransformerAttentionLayerInputs()
      paddings = tf.zeros([2, 5])
      # default setup
      p = attention.TransformerLayer.Params()
      p.name = 'transformer_layer'
      p.has_aux_atten = False
      p.input_dim = 4
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_heads = 2
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(l.theta, query_vec, paddings)
      # reshaped setup
      reshaped_p = p.Copy()
      attention.TransformerLayer.SetReshapedLayers(reshaped_p)
      reshaped_p.device_mesh = np.reshape(np.arange(4), [2, 2])
      attention.TransformerLayer.SetCanonicalShardingParams(
          reshaped_p, reshape_dim=True)
      reshaped_p.name = 'reshaped_transformer_layer'
      reshaped_l = reshaped_p.Instantiate()
      # Use l.theta as it is compatible with reshaped_l.
      reshaped_ctx_vec, _ = reshaped_l.FProp(
          l.theta, tf.reshape(query_vec, [2, 5, 2, 2]), paddings)

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (2, 5, 4))
      reshaped_ctx = sess.run(reshaped_ctx_vec)
      reshaped_ctx = np.reshape(reshaped_ctx, (2, 5, 4))
      self.assertAllClose(actual_ctx, reshaped_ctx)

  def testReshapedTransformerLayerDecodeNoCrossAttention(self):
    np.random.seed(6348575)
    dtype = tf.float32
    b_size = 2
    input_dim = 4
    seq_len = 3
    query_vec = np.random.rand(b_size, seq_len, input_dim)
    paddings = tf.zeros(shape=[b_size, seq_len], dtype=dtype)
    segment_mask = tf.constant(
        [[0, -1e30, -1e30], [-1e30, 0, -1e30], [0, -1e30, 0]], dtype=dtype)
    segment_mask = tf.tile(segment_mask[tf.newaxis, tf.newaxis, :, :],
                           [b_size, 1, 1, 1])

    with self.session(use_gpu=True) as sess:
      p = attention.TransformerLayer.Params()
      p.name = 'reshaped_transformer_layer'
      p.input_dim = input_dim
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_heads = 2
      p.mask_self_atten = True
      p.packed_input = True
      p.has_aux_atten = False
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      attention.TransformerLayer.SetReshapedLayers(p)
      p.device_mesh = np.reshape(np.arange(4), [2, 2])
      attention.TransformerLayer.SetCanonicalShardingParams(p, reshape_dim=True)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(
          l.theta,
          tf.reshape(query_vec, [b_size, seq_len, 2, 2]),
          paddings,
          None,
          None,
          segment_mask=segment_mask)
      ctx_vec = tf.reshape(ctx_vec, [b_size, seq_len, input_dim])

      cached_states = l.InitStates(l.theta, b_size, seq_len)
      extend_step_outs = []
      for t in range(seq_len):
        out_t, _, cached_states = l.ExtendStep(
            l.theta,
            query_vec[:, t:t + 1, :],
            None,
            None,
            cached_states,
            t,
            segment_mask=segment_mask[:, :, t, :])
        extend_step_outs.append(out_t[:, 0, :])

      decoder_out = tf.stack(extend_step_outs, axis=1)

      tf.global_variables_initializer().run()
      fprop_out_v, decoder_out_v = sess.run([ctx_vec, decoder_out])
      tf.logging.info(np.array_repr(fprop_out_v))
      tf.logging.info(np.array_repr(decoder_out_v))
      self.assertAllClose(fprop_out_v, decoder_out_v)

  @parameterized.named_parameters(('SingleBatch', 1), ('DoubleBatch', 2))
  def testMultiSourceTransformerLayerFPropWithCrossAttention(self, multiplier):
    with self.session(use_gpu=True) as sess:
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      query_vec = tf.tile(query_vec, [multiplier, 1, 1])
      paddings = tf.zeros([2 * multiplier, 5])
      p = attention.TransformerLayer.Params()
      p.name = 'transformer_layer'
      p.input_dim = 4
      p.tr_fflayer_tpl.hidden_dim = 7
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      # multi-source cross attention
      p.tr_atten_tpl = (
          attention.TransformerMultiSourceAttentionLayer.Params().Set(
              num_source=2, primary_source_index=0, num_heads=2))
      p.tr_self_atten_tpl = attention.TransformerAttentionLayer.Params().Set(
          input_dim=4, num_heads=2)
      l = p.Instantiate()
      ctx_vec, _ = l.FProp(
          l.theta, query_vec, paddings,
          py_utils.NestedMap({
              'source_0': aux_vec,
              'source_1': aux_vec
          }),
          py_utils.NestedMap({
              'source_0': aux_paddings,
              'source_1': aux_paddings
          }))

      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      actual_ctx = np.reshape(actual_ctx, (10 * multiplier, 4))
      tf.logging.info(np.array_repr(actual_ctx))
      expected_ctx = [
          4.7839108, 4.5303655, 5.5551023, 5.0657663, 5.0493064, 3.2142467,
          2.820018, 5.659971, 4.3814187, 2.60475
      ] * multiplier
      self.assertAllClose(expected_ctx, np.sum(actual_ctx, axis=1))

  @parameterized.named_parameters(('Base', False), ('RelativeAtten', True))
  def testTransformerDecoderLayerConstruction(self, use_relative_atten):
    _ = self._ConstructTransformerDecoderLayer(
        use_relative_atten=use_relative_atten)

  def testTransformerDecoderLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      (query_vec, paddings, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      l = self._ConstructTransformerDecoderLayer()

      layer_output, _ = l.FProp(l.theta, query_vec, paddings, aux_vec,
                                aux_paddings)

      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(layer_output)
      actual_layer_output = np.reshape(actual_layer_output, (10, 4))
      tf.logging.info(np.array_repr(actual_layer_output))
      expected_layer_output = [16.939590, 24.121685, 19.975197, 15.924350]
      self.assertAllClose(expected_layer_output,
                          np.sum(actual_layer_output, axis=0))

  def testTransformerDecoderLayerFPropMixHeads(self):
    with self.session(use_gpu=True) as sess:
      (query_vec, paddings, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      l = self._ConstructTransformerDecoderLayerMixHeads()

      layer_output, _ = l.FProp(l.theta, query_vec, paddings, aux_vec,
                                aux_paddings)

      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(layer_output)
      actual_layer_output = np.reshape(actual_layer_output, (10, 4))
      tf.logging.info(np.array_repr(actual_layer_output))
      expected_layer_output = [6.2344074, 15.817548, 6.8874574, 4.879834]
      self.assertAllClose(expected_layer_output,
                          np.sum(actual_layer_output, axis=0))

  def _ConstructTransformerEncoderLayerStack(self):
    p = attention.StackedTransformerLayers.Params()
    p.name = 'encoder_layers'
    p.has_aux_atten = False
    p.mask_self_atten = False
    p.num_layers = 2
    p.mdl_dim = 4
    p.hidden_dim = 8
    p.num_atten_heads = 2
    p.dropout_prob = 0.2
    p.params_init = py_utils.WeightInit.Xavier()
    p.random_seed = 12345
    return p.Instantiate()

  def _ConstructTransformerDecoderLayerStack(self, dropout_prob=0.2):
    p = attention.StackedTransformerLayers.Params()
    p.name = 'decoder_layers'
    p.has_aux_atten = True
    p.mask_self_atten = True
    p.num_layers = 2
    p.mdl_dim = 4
    p.hidden_dim = 8
    p.num_atten_heads = 2
    p.dropout_prob = dropout_prob
    p.params_init = py_utils.WeightInit.Xavier()
    p.random_seed = 12345
    return p.Instantiate()

  def _ConstructTransformerParamsTplListMixHeadsStack(self):
    p = attention.StackedTransformerLayers.Params()
    p.name = 'encoder_layers'
    p.has_aux_atten = False
    p.mask_self_atten = False
    p.num_layers = 6
    params1 = attention.TransformerLayer.Params()
    params1.tr_atten_tpl.atten_tpl = (
        attention.LocalSelfAttention.Params().Set(
            left_context=2, right_context=2, block_size=4))
    params2 = attention.TransformerLayer.Params()
    params2.tr_atten_tpl.atten_tpl = (
        attention.RoutingAttention.Params().Set(
            num_clusters=1, attention_window=2))
    params3 = attention.TransformerLayer.Params()
    params3.tr_atten_tpl.atten_tpl = [
        attention.LocalSelfAttention.Params().Set(
            left_context=2, right_context=2, block_size=4),
        attention.RoutingAttention.Params().Set(
            num_clusters=1, attention_window=2)
    ]
    params3.num_heads = [1, 1]
    p.transformer_layer_params_tpl = [params1, params2, params3]
    p.mdl_dim = 4
    p.hidden_dim = 8
    p.num_atten_heads = 2
    p.dropout_prob = 0.2
    p.params_init = py_utils.WeightInit.Xavier()
    p.random_seed = 12345
    return p.Instantiate()

  def _ConstructRepeatedTransformerDecoderLayer(self,
                                                repeat,
                                                per_layer_vars=False):
    p = attention.RepeatedTransformerLayer.Params()
    p.name = 'repeated_decoder_layer'
    p.params_init = py_utils.WeightInit.Xavier()
    p.random_seed = 12345
    p.repeat = repeat
    p.per_layer_vars = per_layer_vars
    p.atten_prob_aggregation = 'mean'
    tp = p.body = attention.TransformerDecoderLayer.Params()
    tp.input_dim = 4
    tp.tr_fflayer_tpl.hidden_dim = 7
    tp.tr_atten_tpl.num_heads = 2
    return p.Instantiate()

  def testTransformerStackTplList(self):
    l = self._ConstructTransformerParamsTplListMixHeadsStack()
    self.assertIsInstance(l.x_layers[0].self_atten.atten,
                          attention.LocalSelfAttention)
    self.assertIsInstance(l.x_layers[1].self_atten.atten,
                          attention.LocalSelfAttention)
    self.assertIsInstance(l.x_layers[2].self_atten.atten,
                          attention.RoutingAttention)
    self.assertIsInstance(l.x_layers[3].self_atten.atten,
                          attention.RoutingAttention)
    self.assertIsInstance(l.x_layers[4].self_atten.atten[0],
                          attention.LocalSelfAttention)
    self.assertIsInstance(l.x_layers[4].self_atten.atten[1],
                          attention.RoutingAttention)
    self.assertIsInstance(l.x_layers[5].self_atten.atten[0],
                          attention.LocalSelfAttention)
    self.assertIsInstance(l.x_layers[5].self_atten.atten[1],
                          attention.RoutingAttention)

  def testStackedTransformerGetSplitForLayer(self):
    cls = attention.StackedTransformerLayers

    buckets = [2, 4, 5, 6, 9, 11, 15]
    ys = [cls.GetSplitForLayer(buckets, i) for i in range(16)]
    self.assertEqual(0, ys[0])
    self.assertEqual(0, ys[1])
    self.assertEqual(0, ys[2])
    self.assertEqual(1, ys[3])

    self.assertEqual(1, ys[4])
    self.assertEqual(2, ys[5])
    self.assertEqual(3, ys[6])
    self.assertEqual(4, ys[7])

    self.assertEqual(4, ys[8])
    self.assertEqual(4, ys[9])
    self.assertEqual(5, ys[10])
    self.assertEqual(5, ys[11])

    self.assertEqual(6, ys[12])
    self.assertEqual(6, ys[13])
    self.assertEqual(6, ys[14])
    self.assertEqual(6, ys[15])

  def testTransformerEncoderLayerStackFProp(self):
    with self.session(use_gpu=True) as sess:
      (query_vec, paddings, _, _) = self._TransformerAttentionLayerInputs()
      l = self._ConstructTransformerEncoderLayerStack()
      layer_output, _ = l.FProp(l.theta, query_vec=query_vec, paddings=paddings)
      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(layer_output)
      actual_layer_output = np.reshape(actual_layer_output, (10, 4))
      tf.logging.info(np.array_repr(actual_layer_output))
      expected_layer_output = [6.178955, -11.376661, 7.032681, -1.532627]
      self.assertAllClose(expected_layer_output,
                          np.sum(actual_layer_output, axis=0))

  def testTransformerDecoderLayerStackFProp(self):
    with self.session(use_gpu=True) as sess:
      (query_vec, paddings, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      l = self._ConstructTransformerDecoderLayerStack()
      layer_output, _ = l.FProp(
          l.theta,
          query_vec=query_vec,
          paddings=paddings,
          aux_vec=aux_vec,
          aux_paddings=aux_paddings)
      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(layer_output)
      actual_layer_output = np.reshape(actual_layer_output, (10, 4))
      tf.logging.info(np.array_repr(actual_layer_output))
      expected_layer_output = [9.926413, -4.491376, 27.051598, 2.112684]
      self.assertAllClose(expected_layer_output,
                          np.sum(actual_layer_output, axis=0))

  @parameterized.named_parameters(
      {
          'testcase_name': '_short_seq',
          'use_short_seq_opt': True,
      }, {
          'testcase_name': '_long_seq',
          'use_short_seq_opt': False,
      })
  def testTransformerDecoderLayerStackExtendStep(self, use_short_seq_opt):

    def _Rnd(seed):
      return tf.random.normal([5, 2, 2, 2], seed=seed)

    graph = tf.Graph()
    with graph.as_default():
      tf.random.set_seed(123456)
      query_vec, _, aux_vec, aux_paddings = (
          self._TransformerAttentionLayerInputs())
      paddings = tf.zeros([2, 5])
      layer_prefix_states_1 = py_utils.NestedMap(key=_Rnd(1), value=_Rnd(2))
      layer_prefix_states_2 = py_utils.NestedMap(key=_Rnd(3), value=_Rnd(4))
      prefix_states = py_utils.NestedMap(
          x_layers=[layer_prefix_states_1, layer_prefix_states_2])

      l = self._ConstructTransformerDecoderLayerStack(dropout_prob=0.)

      layer_output1, _ = l.FProp(l.theta, query_vec, paddings, aux_vec,
                                 aux_paddings)

      layer_output2 = []
      for i in range(5):
        layer_output, prefix_states = l.ExtendStep(
            l.theta, tf.expand_dims(query_vec[:, i, :], 1), aux_vec,
            aux_paddings, prefix_states, i, use_short_seq_opt)
        layer_output2.append(tf.squeeze(layer_output, 1))
      layer_output2 = tf.transpose(tf.stack(layer_output2), [1, 0, 2])

    with self.session(graph=graph, use_gpu=True) as sess:
      tf.global_variables_initializer().run()
      actual_layer_output1, actual_layer_output2 = sess.run(
          [layer_output1, layer_output2])

    self.assertAllClose(actual_layer_output1, actual_layer_output2)

  @parameterized.named_parameters(
      {
          'testcase_name': '_short_seq',
          'use_short_seq_opt': True,
      }, {
          'testcase_name': '_long_seq',
          'use_short_seq_opt': False,
      })
  def testTransformerDecoderLayerExtendStep(self, use_short_seq_opt):
    with self.session(use_gpu=True) as sess:
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      paddings = tf.zeros([2, 5])
      cached_key = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      cached_value = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      prefix_states = py_utils.NestedMap(key=cached_key, value=cached_value)

      l = self._ConstructTransformerDecoderLayer()

      layer_output1, layer_atten_probs1 = l.FProp(l.theta, query_vec, paddings,
                                                  aux_vec, aux_paddings)
      layer_atten_probs1 = layer_atten_probs1.aux_atten

      layer_output2 = []
      layer_atten_probs2 = []
      for i in range(5):
        layer_output, cross_atten_probs, prefix_states = l.ExtendStep(
            l.theta,
            tf.expand_dims(query_vec[:, i, :], 1),
            aux_vec,
            aux_paddings,
            prefix_states,
            i,
            use_short_seq_opt,
            compute_atten_probs=True)
        layer_output2.append(tf.squeeze(layer_output, 1))
        layer_atten_probs2.append(cross_atten_probs)
      layer_output2 = tf.transpose(tf.stack(layer_output2), [1, 0, 2])
      # [B, N, T, S].
      layer_atten_probs2 = tf.concat(layer_atten_probs2, axis=2)

      tf.global_variables_initializer().run()
      (actual_layer_output1, actual_layer_output2, actual_layer_atten_probs1,
       actual_layer_atten_probs2) = sess.run([
           layer_output1, layer_output2, layer_atten_probs1, layer_atten_probs2
       ])
      self.assertAllClose(actual_layer_output1, actual_layer_output2)
      self.assertAllClose(actual_layer_atten_probs1, actual_layer_atten_probs2)

  @parameterized.named_parameters(
      {
          'testcase_name': '_short_seq',
          'use_short_seq_opt': True,
      }, {
          'testcase_name': '_long_seq',
          'use_short_seq_opt': False,
      }, {
          'testcase_name': '_repeat',
          'repeat': 3,
      }, {
          'testcase_name': '_repeat_per_layer_var',
          'repeat': 3,
          'per_layer_var': True,
      })
  def testTransformerDecoderLayerExtendStepDifferentBatchSizes(
      self, use_short_seq_opt=False, repeat=None, per_layer_var=False):
    with self.session(use_gpu=True) as sess:
      if repeat:
        l = self._ConstructRepeatedTransformerDecoderLayer(
            repeat, per_layer_var)
      else:
        l = self._ConstructTransformerDecoderLayer()

      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      paddings = tf.zeros([2, 5])

      layer_output1, layer_atten_probs1 = l.FProp(
          l.theta,
          query_vec,
          paddings=paddings,
          aux_vec=aux_vec,
          aux_paddings=aux_paddings)
      layer_atten_probs1 = layer_atten_probs1.aux_atten

      source_batch, source_length = py_utils.GetShape(aux_paddings, 2)
      batch_multiplier = 2
      target_batch = source_batch * batch_multiplier
      num_heads = 2
      prefix_states = l.InitStates(
          l.theta, target_batch_size=target_batch, target_max_length=5)

      def _TileByBatchMultiplier(x):
        """Tile 'x' along the batch dim by batch_multiplier."""
        b, t, d = py_utils.GetShape(x)
        # [b, batch_multiplier, t, d].
        x = tf.tile(tf.expand_dims(x, axis=1), [1, batch_multiplier, 1, 1])
        return tf.reshape(x, [b * batch_multiplier, t, d])

      tiled_query_vec = _TileByBatchMultiplier(query_vec)

      layer_output2 = []
      layer_atten_probs2 = []
      for i in range(5):
        layer_output, cross_atten_probs, prefix_states = l.ExtendStep(
            l.theta,
            tiled_query_vec[:, i:i + 1, :],
            cached_states=prefix_states,
            aux_vec=aux_vec,
            aux_paddings=aux_paddings,
            time_step=i,
            use_short_seq_opt=use_short_seq_opt,
            compute_atten_probs=True)
        layer_output2.append(layer_output)
        layer_atten_probs2.append(
            py_utils.HasShape(cross_atten_probs,
                              [target_batch, num_heads, 1, source_length]))
      layer_output2 = tf.concat(layer_output2, axis=1)
      # [B, N, T, S].
      layer_atten_probs2 = tf.concat(layer_atten_probs2, axis=-2)

      tf.global_variables_initializer().run()
      (actual_layer_output1, actual_layer_output2, actual_layer_atten_probs1,
       actual_layer_atten_probs2) = sess.run([
           layer_output1, layer_output2, layer_atten_probs1, layer_atten_probs2
       ])
      for i in range(source_batch):
        for j in range(batch_multiplier):
          tf.logging.info('Expected (%s): %s', i, actual_layer_output1[i])
          tf.logging.info('Actual (%s, %s): %s', i, j,
                          actual_layer_output2[i * batch_multiplier + j])
          self.assertAllClose(actual_layer_output1[i],
                              actual_layer_output2[i * batch_multiplier + j])
          self.assertAllClose(
              actual_layer_atten_probs1[i],
              actual_layer_atten_probs2[i * batch_multiplier + j])

  def _ConstructMultiSourceTransformerDecoderLayer(self,
                                                   use_relative_atten=False):
    p = attention.MultiSourceTransformerDecoderLayer.Params().Set(num_source=2)
    p.name = 'multi_source_transformer_decoder_layer'
    p.input_dim = 4
    p.tr_fflayer_tpl.hidden_dim = 7
    # multi-source cross attention
    p.tr_atten_tpl = (
        attention.TransformerMultiSourceAttentionLayer.Params().Set(
            num_source=2, primary_source_index=0, num_heads=2))
    p.tr_self_atten_tpl = attention.TransformerAttentionLayer.Params().Set(
        input_dim=4, num_heads=2)
    p.tr_atten_tpl.multi_source_atten.atten_merger_tpl = (
        tm_attention.MergerLayer.Params().Set(merger_op='sum'))
    if use_relative_atten:
      p = attention.UseRelativeAttentionInTransformerLayer(p, 4)
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
    return attention.MultiSourceTransformerDecoderLayer(p)

  @parameterized.named_parameters(
      {
          'testcase_name': '_short_seq',
          'use_short_seq_opt': True,
      }, {
          'testcase_name': '_long_seq',
          'use_short_seq_opt': False,
      })
  def testMultiSourceTransformerDecoderLayerExtendStep(self, use_short_seq_opt):
    with self.session(use_gpu=True) as sess:
      (query_vec, _, aux_vec,
       aux_paddings) = self._TransformerAttentionLayerInputs()
      paddings = tf.zeros([2, 5])
      cached_key = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      cached_value = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      prefix_states = py_utils.NestedMap(key=cached_key, value=cached_value)

      l = self._ConstructMultiSourceTransformerDecoderLayer()

      ms_aux_vec = py_utils.NestedMap({
          'source_0': aux_vec,
          'source_1': aux_vec
      })
      ms_aux_paddings = py_utils.NestedMap({
          'source_0': aux_paddings,
          'source_1': aux_paddings
      })
      layer_output1, layer_atten_probs1 = l.FProp(l.theta, query_vec, paddings,
                                                  ms_aux_vec, ms_aux_paddings)
      layer_atten_probs1 = layer_atten_probs1.aux_atten

      layer_output2 = []
      layer_atten_probs2 = []
      for i in range(5):
        layer_output, cross_atten_probs, prefix_states = l.ExtendStep(
            l.theta,
            tf.expand_dims(query_vec[:, i, :], 1),
            ms_aux_vec,
            ms_aux_paddings,
            prefix_states,
            i,
            use_short_seq_opt,
            compute_atten_probs=True)
        layer_output2.append(tf.squeeze(layer_output, 1))
        layer_atten_probs2.append(cross_atten_probs)
      layer_output2 = tf.transpose(tf.stack(layer_output2), [1, 0, 2])
      # [B, N, T, S].
      layer_atten_probs2 = tf.concat(layer_atten_probs2, axis=2)

      tf.global_variables_initializer().run()
      (actual_layer_output1, actual_layer_output2, actual_layer_atten_probs1,
       actual_layer_atten_probs2) = sess.run([
           layer_output1, layer_output2, layer_atten_probs1, layer_atten_probs2
       ])
      self.assertAllClose(actual_layer_output1, actual_layer_output2)
      self.assertAllClose(actual_layer_atten_probs1, actual_layer_atten_probs2)

  def _testTransformerDecoderLayerInputs(self,
                                         depth=3,
                                         context_depth=3,
                                         dtype=tf.float32):
    source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(5)])
    source_padding = tf.transpose(
        tf.constant([[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]], dtype=dtype))
    aux_source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)])
    aux_source_paddings = tf.transpose(
        tf.constant([[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]],
                    dtype=dtype))
    context_vecs = tf.stack([
        tf.constant(np.random.rand(2, context_depth), dtype=dtype)
        for _ in range(7)
    ])
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings,
            context_vecs)

  def testPrefixTransformerLayerExtendStep(self):
    with self.session(use_gpu=False):
      np.random.seed(6348575)
      depth = 4
      p = attention.TransformerDecoderLayer.Params()
      p.name = 'TransformerDecoderLayer'
      p.input_dim = 4
      p.tr_fflayer_tpl.input_dim = 4
      p.tr_fflayer_tpl.hidden_dim = 8
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_atten_tpl = attention.TransformerAttentionLayer.Params().Set(
          num_heads=2, input_dim=4)
      transformer = p.Instantiate()

      (source_vecs, _, aux_vecs, aux_paddings,
       _) = self._testTransformerDecoderLayerInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      source_vecs = tf.transpose(source_vecs, [1, 0, 2])
      source_padding = tf.transpose(source_padding, [1, 0])
      aux_vecs = tf.transpose(aux_vecs, [1, 0, 2])
      aux_paddings = tf.transpose(aux_paddings, [1, 0])

      h1, _ = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vec=aux_vecs,
          aux_paddings=aux_paddings)

      h2 = []
      cached_source_vecs = tf.concat([
          tf.random.uniform((2, 2, 2, 2), 0.0, 1.0),
          tf.zeros((5, 2, 2, 2), dtype=tf.float32)
      ],
                                     axis=0)
      cached_source_contexts = tf.concat([
          tf.random.uniform((2, 2, 2, 2), 0.0, 1.0),
          tf.zeros((5, 2, 2, 2), dtype=tf.float32)
      ],
                                         axis=0)
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        # Ignore the first two timesteps in cached_source.
        per_step_padding = tf.concat([
            tf.ones([2, 2], dtype=tf.float32),
            tf.zeros([2, i + 1], dtype=tf.float32),
            tf.ones([2, 4 - i], dtype=tf.float32)
        ],
                                     axis=1)
        per_step_padding = tf.expand_dims(per_step_padding, axis=1)

        h, _, prefix_states = transformer.ExtendStep(
            transformer.theta,
            source_vecs[:, i:i + 1, :],
            aux_vecs,
            aux_paddings,
            prefix_states,
            time_step=i + 2,
            per_step_padding=per_step_padding)
        h2.append(h)

      h2 = tf.concat(h2, axis=1)

      self.evaluate(tf.global_variables_initializer())
      h1_v, h2_v = self.evaluate([h1, h2])
      self.assertAllClose(h1_v, h2_v, atol=1e-3)


class GPipeBatchMajorTransformerLayerTest(test_utils.TestCase,
                                          parameterized.TestCase):
  """Test GPipeBatchMajorTransformer layers."""

  def _ConstructGPipeBatchMajorTransformerLayer(self,
                                                decoder=False,
                                                packed=True,
                                                dropout=0.1):
    p = attention.GPipeBatchMajorTransformerLayer.Params()
    p.name = 'gpipe_transformer_layer'
    p.input_dim = 4
    p.tr_fflayer_tpl.hidden_dim = 7
    p.tr_atten_tpl.num_heads = 2
    p.tr_atten_tpl.residual_dropout_prob = dropout
    p.packed_input = packed
    if decoder:
      p.has_aux_atten = True
      p.mask_self_atten = True
    p.cls.SetupDeterministicDropout(p)
    layer = p.Instantiate()
    return p, layer

  def _GPipeBatchMajorTransformerLayerInputs(self,
                                             input_dim=4,
                                             dtype=tf.float32):
    np.random.seed(6348575)
    target_vec = tf.transpose(
        tf.stack([
            tf.constant(np.random.rand(2, input_dim), dtype=dtype)
            for _ in range(5)
        ]), [1, 0, 2])
    target_paddings = tf.constant([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
                                  dtype=dtype)
    aux_vec = tf.transpose(
        tf.stack([
            tf.constant(np.random.rand(2, input_dim), dtype=dtype)
            for _ in range(7)
        ]), [1, 0, 2])
    aux_paddings = tf.constant([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]],
                               dtype=dtype)
    aux_segment_ids = tf.constant(
        [[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1]], dtype=dtype)
    target_segment_ids = tf.constant([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]],
                                     dtype=dtype)
    target_sa_mask = attention.SegmentMask(
        target_segment_ids, target_segment_ids, apply_dtype_min=False)
    aux_sa_mask = attention.SegmentMask(
        aux_segment_ids, aux_segment_ids, apply_dtype_min=False)
    ca_mask = attention.SegmentMask(
        target_segment_ids, aux_segment_ids, apply_dtype_min=False)
    causal_padding = tf.expand_dims(
        tf.tile(
            tf.expand_dims(attention.CausalPadding(5, dtype=dtype), 0),
            [2, 1, 1]), 1)
    target_sa_mask = tf.math.maximum(causal_padding, target_sa_mask)
    return (target_vec, target_paddings, target_sa_mask, aux_vec, aux_paddings,
            aux_sa_mask, ca_mask)

  def testGPipeBatchMajorTransformerEncoderLayerConstruction(self):
    _, layer = self._ConstructGPipeBatchMajorTransformerLayer()
    self.assertEqual(0.1, layer.params.tr_atten_tpl.residual_dropout_prob)

  def testGPipeBatchMajorTransformerDecoderLayerConstruction(self):
    _, layer = self._ConstructGPipeBatchMajorTransformerLayer(decoder=True)
    self.assertEqual(0.1, layer.params.tr_atten_tpl.residual_dropout_prob)

  def testGPipeBatchMajorTransformerEncoderLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      (_, _, _, aux_vec, aux_paddings, aux_sa_mask,
       _) = self._GPipeBatchMajorTransformerLayerInputs()
      _, l = self._ConstructGPipeBatchMajorTransformerLayer()

      layer_output = l.FProp(l.theta, aux_vec, aux_paddings, None, None,
                             aux_sa_mask, None, None)[0]

      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(layer_output)
      actual_layer_output = np.reshape(actual_layer_output, (14, 4))
      tf.logging.info(np.array_repr(actual_layer_output))
      expected_layer_output = [7.616176, 8.611565, -0.932456, -4.5797]
      self.assertAllClose(expected_layer_output,
                          np.sum(actual_layer_output, axis=0))

  def testGPipeBatchMajorTransformerDecoderLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      (target_vec, target_paddings, target_sa_mask, aux_vec, aux_paddings,
       aux_sa_mask, ca_mask) = self._GPipeBatchMajorTransformerLayerInputs()
      _, l = self._ConstructGPipeBatchMajorTransformerLayer(decoder=True)

      layer_output = l.FProp(l.theta, aux_vec, aux_paddings, target_vec,
                             target_paddings, aux_sa_mask, target_sa_mask,
                             ca_mask)[2]

      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(layer_output)
      actual_layer_output = np.reshape(actual_layer_output, (10, 4))
      tf.logging.info(np.array_repr(actual_layer_output))
      expected_layer_output = [2.721037, 5.228053, 2.27512, 6.92945]
      self.assertAllClose(expected_layer_output,
                          np.sum(actual_layer_output, axis=0))

  def testGPipeBatchMajorTransformerDecoderLayerExtendStep(self):
    with self.session(use_gpu=True) as sess:
      (target_vec, _, _, aux_vec, aux_paddings, _,
       _) = self._GPipeBatchMajorTransformerLayerInputs()
      target_paddings = tf.zeros([2, 5])
      cached_key = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      cached_value = tf.constant(
          np.random.normal(0.1, 0.5, [5, 2, 2, 2]), dtype=tf.float32)
      prefix_states = py_utils.NestedMap(key=cached_key, value=cached_value)
      _, l = self._ConstructGPipeBatchMajorTransformerLayer(
          decoder=True, packed=False, dropout=0.0)

      layer_output1 = l.FProp(l.theta, aux_vec, aux_paddings, target_vec,
                              target_paddings, None, None, None)[2]

      layer_output2 = []
      for i in range(5):
        layer_output, _, prefix_states = l.ExtendStep(
            l.theta, tf.expand_dims(target_vec[:, i, :], 1), aux_vec,
            aux_paddings, prefix_states, i)
        layer_output2.append(tf.squeeze(layer_output, 1))
      layer_output2 = tf.transpose(tf.stack(layer_output2), [1, 0, 2])

      tf.global_variables_initializer().run()
      actual_layer_output1, actual_layer_output2 = sess.run(
          [layer_output1, layer_output2])
      self.assertAllClose(actual_layer_output1, actual_layer_output2)


class BuilderTest(test_utils.TestCase, parameterized.TestCase):

  def _testGraph(self, glu_with_tanh=False, dtype=tf.float32):
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    atten_builder = attention.Builder.Params().Set(
        model_dim=4, num_heads=2, ff_hidden_dim=16, glu_with_tanh=glu_with_tanh)
    params = atten_builder.Instantiate().LConvStack(
        name='lightconv', kernel_sizes=[3, 3])
    params.dtype = dtype
    params.random_seed = 0
    params.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
    l = params.Instantiate()
    l_in = tf.constant(np.random.rand(2, 3, 4), dtype=dtype)
    l_padding = tf.zeros([2, 3], dtype=dtype)
    l_out = l.FPropDefaultTheta(
        py_utils.NestedMap(vec=l_in, paddings=l_padding))
    return l_out.vec

  @parameterized.parameters((False, 38.163662), (True, 35.88797))
  def testFprop(self, glu_with_tanh, expected_result):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      l_out = self._testGraph(glu_with_tanh)
      l_out = tf.reduce_sum(l_out)
      tf.global_variables_initializer().run()
      l_out_eval = sess.run(l_out)
      self.assertAllClose(expected_result, l_out_eval)

  def testBProp(self):
    with self.session(use_gpu=True) as sess:
      output = self._testGraph(dtype=tf.float64)
      loss = tf.reduce_sum(output)
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      tf.global_variables_initializer().run()
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]
      for ng, sg in zip(num_grads, sym_grads):
        self.assertAllClose(ng, sg, rtol=5e-02, atol=5e-02)

  @parameterized.named_parameters(
      {
          'testcase_name': '_baseline',
          'strides': [1, 1],
      }, {
          'testcase_name': '_stride_2',
          'strides': [1, 2],
      }, {
          'testcase_name': '_first_token',
          'strides': [2, 0],
      }, {
          'testcase_name': '_stride_2_begin_intact_1_no_trunc',
          'strides': [1, 2],
          'begin_intact': 1,
          'trunc_seq': False,
      }, {
          'testcase_name': '_stride_2_begin_intact_1_trunc',
          'strides': [1, 2],
          'begin_intact': 1,
          'trunc_seq': True,
      }, {
          'testcase_name': '_gpipe',
          'strides': [1, 1],
          'num_splits': 2,
          'num_micro_batches': 2,
      })
  def testFunnelTransformerStack(self,
                                 strides,
                                 begin_intact=0,
                                 trunc_seq=True,
                                 num_splits=1,
                                 num_micro_batches=1):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      atten_builder_params = attention.Builder.Params().Set(
          num_splits=num_splits,
          num_micro_batches=num_micro_batches,
          deterministic_dropout=num_splits > 1 or num_micro_batches > 1,
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          funnel_pool_tpl=attention.FunnelPoolingLayer.Params().Set(
              begin_intact=begin_intact, trunc_seq=trunc_seq))
      atten_builder = atten_builder_params.Instantiate()
      layers = []
      accumulate_stride = 1
      for layer_i, stride in enumerate(strides):
        accumulate_stride *= stride
        layers.append(
            atten_builder.FunnelEncoderLayer(
                name='atten_{}'.format(layer_i), stride=stride))
      p = atten_builder.Stack('model', layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      enc_out = l_out.vec
      tf.global_variables_initializer().run()
      actual_enc_out = sess.run(enc_out)
      if accumulate_stride == 0:
        self.assertAllEqual([bs, 1, d], actual_enc_out.shape)
      elif (not begin_intact) or (begin_intact and trunc_seq):
        seq_len = sl // accumulate_stride
        self.assertAllEqual([bs, seq_len, d], actual_enc_out.shape)
      elif begin_intact and not trunc_seq:
        seq_len = sl
        for stride in strides:
          if stride > 1:
            seq_len = begin_intact + int(
                math.ceil((seq_len - begin_intact) / stride))
        self.assertAllEqual([bs, seq_len, d], actual_enc_out.shape)

  @parameterized.named_parameters(
      {
          'testcase_name': '_baseline',
          'strides': [1, 1],
      }, {
          'testcase_name': '_stride_2',
          'strides': [1, 2],
      }, {
          'testcase_name': '_first_token',
          'strides': [2, 0],
      })
  def testFunnelTransformerStackStochasticDepth(self,
                                                strides,
                                                begin_intact=0,
                                                trunc_seq=True):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      atten_builder_params = attention.Builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          survival_prob=0.9,
          funnel_pool_tpl=attention.FunnelPoolingLayer.Params().Set(
              begin_intact=begin_intact, trunc_seq=trunc_seq))
      atten_builder = atten_builder_params.Instantiate()
      layers = []
      accumulate_stride = 1
      for layer_i, stride in enumerate(strides):
        accumulate_stride *= stride
        layers.append(
            atten_builder.FunnelEncoderLayer(
                name='atten_{}'.format(layer_i), stride=stride))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      enc_out = l_out.vec
      tf.global_variables_initializer().run()
      actual_enc_out = sess.run(enc_out)
      if accumulate_stride == 0:
        self.assertAllEqual([bs, 1, d], actual_enc_out.shape)
      elif (not begin_intact) or (begin_intact and trunc_seq):
        seq_len = sl // accumulate_stride
        self.assertAllEqual([bs, seq_len, d], actual_enc_out.shape)
      elif begin_intact and not trunc_seq:
        seq_len = sl
        for stride in strides:
          if stride > 1:
            seq_len = begin_intact + int(
                math.ceil((seq_len - begin_intact) / stride))
        self.assertAllEqual([bs, seq_len, d], actual_enc_out.shape)

  @parameterized.named_parameters(
      {
          'testcase_name': '_avg_pool_exclude',
          'stride': 2,
          'pooling_type': 'AVG',
          'exclude_pad_effect': True,
      }, {
          'testcase_name': '_max_pool_exclude',
          'stride': 2,
          'pooling_type': 'MAX',
          'exclude_pad_effect': True,
      }, {
          'testcase_name': '_avg_pool',
          'stride': 2,
          'pooling_type': 'AVG',
          'exclude_pad_effect': False,
      }, {
          'testcase_name': '_max_pool',
          'stride': 2,
          'pooling_type': 'MAX',
          'exclude_pad_effect': False,
      })
  def testFunnelPoolingFixPaddingEffect(self, stride, pooling_type,
                                        exclude_pad_effect):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      funnel_pooling_params = attention.FunnelPoolingLayer.Params().Set(
          name='funnel_pool',
          stride=stride,
          pooling_type=pooling_type,
          exclude_pad_effect=exclude_pad_effect)
      l = funnel_pooling_params.Instantiate()

      inputs_np = np.random.random([bs, sl, d]) * 10
      non_pad_len = np.random.randint(sl // 2, sl, size=[bs])
      paddings_np = np.arange(sl)[None, :] >= non_pad_len[:, None]
      paddings_np = paddings_np.astype(np.float)

      inputs = tf.constant(inputs_np, dtype=np.float)
      paddings = tf.constant(paddings_np, dtype=np.float)

      pooled_tensor, pooled_paddings = l.FPropDefaultTheta(inputs, paddings)
      tf.global_variables_initializer().run()
      pooled_tensor_np, pooled_paddings_np = sess.run(
          [pooled_tensor, pooled_paddings])
      self.assertAllEqual([bs, sl // stride, d], pooled_tensor_np.shape)
      self.assertAllEqual([bs, sl // stride], pooled_paddings_np.shape)
      self.assertAllClose(paddings_np[:, ::stride], pooled_paddings_np)

      # construct groudtruth
      inputs_4d = inputs_np.copy().reshape([bs, sl // stride, stride, d])
      paddings_4d = paddings_np.copy().reshape([bs, sl // stride, stride, 1])
      if pooling_type == 'AVG':
        if exclude_pad_effect:
          not_padding_4d = 1.0 - paddings_4d
          target_tensor = np.sum(inputs_4d * not_padding_4d, axis=2)
          target_tensor /= 1e-8 + np.sum(not_padding_4d, axis=2)
        else:
          target_tensor = np.mean(inputs_4d, axis=2)
      elif pooling_type == 'MAX':
        if exclude_pad_effect:
          padding_mask = np.tile(paddings_4d > 0, [1, 1, 1, d])
          inputs_4d[padding_mask] = np.finfo(inputs_4d.dtype).min
        target_tensor = np.max(inputs_4d, axis=2)
      target_tensor *= (1.0 - paddings_np[:, ::stride, None])

      self.assertAllClose(target_tensor, pooled_tensor_np)

  @parameterized.named_parameters(
      {
          'testcase_name': '_avg_pool_no_paddings',
          'stride': 2,
          'pooling_type': 'AVG',
      }, {
          'testcase_name': '_max_pool_no_paddings',
          'stride': 2,
          'pooling_type': 'MAX',
      })
  def testFunnelPoolingNoPaddings(self, stride, pooling_type):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      funnel_pooling_params = attention.FunnelPoolingLayer.Params().Set(
          name='funnel_pool', stride=stride, pooling_type=pooling_type)
      l = funnel_pooling_params.Instantiate()
      inputs_np = np.random.random([bs, sl, d]) * 10
      inputs = tf.constant(inputs_np, dtype=np.float)
      pooled_tensor = l.FPropDefaultTheta(inputs)
      tf.global_variables_initializer().run()
      pooled_tensor_np = sess.run(pooled_tensor)
      with self.subTest('test_output_shape'):
        self.assertAllEqual([bs, sl // stride, d], pooled_tensor_np.shape)

      inputs_4d = inputs_np.copy().reshape([bs, sl // stride, stride, d])
      if pooling_type == 'AVG':
        target_tensor = np.sum(inputs_4d, axis=2) / 2
      elif pooling_type == 'MAX':
        target_tensor = np.max(inputs_4d, axis=2)
      with self.subTest('test_output_value'):
        self.assertAllClose(target_tensor, pooled_tensor_np)

  @parameterized.named_parameters(
      {
          'testcase_name': '_baseline',
          'split': 1,
          'num_micro_batches': 1,
      }, {
          'testcase_name': '_split',
          'split': 2,
          'num_micro_batches': 1,
      }, {
          'testcase_name': '_gpipe',
          'split': 2,
          'num_micro_batches': 2,
      })
  def testFunnelTransformerStackWithSplit(self, split, num_micro_batches):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      atten_builder_params = attention.Builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          num_splits=split,
          num_micro_batches=num_micro_batches,
          deterministic_dropout=split > 1 or num_micro_batches > 1,
          funnel_pool_tpl=attention.FunnelPoolingLayer.Params())
      atten_builder = atten_builder_params.Instantiate()
      layers = []
      accumulate_stride = 1
      for layer_i, stride in enumerate([1, 2]):
        accumulate_stride *= stride
        layers.append(
            atten_builder.FunnelEncoderLayer(
                name='atten_{}'.format(layer_i), stride=stride))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      enc_out = l_out.vec
      tf.global_variables_initializer().run()
      actual_enc_out = sess.run(enc_out)
      seq_len = sl // accumulate_stride
      self.assertAllEqual([bs, seq_len, d], actual_enc_out.shape)

  @parameterized.named_parameters(
      {
          'testcase_name': '_baseline',
          'strides': [1, 1],
      }, {
          'testcase_name': '_stride_2',
          'strides': [1, 2],
      }, {
          'testcase_name': '_stride_2_begin_intact_1_no_trunc',
          'strides': [1, 2],
          'begin_intact': 1,
          'trunc_seq': False,
      }, {
          'testcase_name': '_stride_2_begin_intact_1_trunc',
          'strides': [1, 2],
          'begin_intact': 1,
          'trunc_seq': True,
      })
  def testFunnelTransformerStackWithUpsampling(self,
                                               strides,
                                               begin_intact=0,
                                               trunc_seq=True):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      atten_builder_params = attention.Builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          funnel_pool_tpl=attention.FunnelPoolingLayer.Params().Set(
              begin_intact=begin_intact, trunc_seq=trunc_seq))
      atten_builder = atten_builder_params.Instantiate()
      layers = []
      accumulate_stride = 1
      for layer_i, stride in enumerate(strides):
        accumulate_stride *= stride
        layers.append(
            atten_builder.FunnelEncoderLayer(
                name='atten_{}'.format(layer_i), stride=stride))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()

      upsample_p = attention.FunnelUpsampleLayer.Params().Set(
          name='funnel_upsample',
          begin_intact=begin_intact,
          trunc_seq=trunc_seq,
          upsample_rate=accumulate_stride)
      l_upsample = upsample_p.Instantiate()

      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      enc_out = l_out.vec
      upsampled_out = l_upsample.FPropDefaultTheta(enc_out)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_upsample_out = sess.run([enc_out, upsampled_out])
      if (begin_intact == 0) or (begin_intact > 0 and trunc_seq):
        seq_len = sl // accumulate_stride
      elif begin_intact > 0 and not trunc_seq:
        seq_len = sl
        for stride in strides:
          if stride > 1:
            seq_len = begin_intact + int(
                math.ceil((seq_len - begin_intact) / stride))
      tf.logging.info('Pool out: %s, Upsample out: %s', actual_enc_out.shape,
                      actual_upsample_out.shape)
      self.assertAllEqual([bs, seq_len, d], actual_enc_out.shape)
      self.assertAllEqual([bs, sl, d], actual_upsample_out.shape)

  def testFunnelEncoderLayerWithPerLayerFfns(self):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      num_ffns_list = [2, 1, 3]
      strides = [1, 2, 2]
      tf.random.set_seed(12345)
      atten_builder_params = attention.Builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          funnel_pool_tpl=attention.FunnelPoolingLayer.Params().Set())
      atten_builder = atten_builder_params.Instantiate()
      layers = []

      for layer_i, stride in enumerate(strides):
        layers.append(
            atten_builder.FunnelEncoderLayer(
                name='atten_{}'.format(layer_i),
                stride=stride,
                num_ffns=num_ffns_list[layer_i]))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      out = tf.reduce_sum(l_out.vec)
      tf.global_variables_initializer().run()
      actual_out = sess.run(out)
      self.assertAllClose(actual_out, 79.52954)

  @parameterized.named_parameters(
      {
          'testcase_name': '_baseline',
          'strides': [1, 1],
      }, {
          'testcase_name': '_stride_2',
          'strides': [2, 1],
      }, {
          'testcase_name': '_first_token',
          'strides': [2, 0],
      })
  def testTransformerStackWithStride(self, strides):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      atten_builder = attention.Builder.Params().Set(
          model_dim=d, num_heads=2, ff_hidden_dim=5).Instantiate()
      layers = []
      accumulate_stride = 1
      for layer_i, stride in enumerate(strides):
        accumulate_stride *= stride
        layers.append(
            atten_builder.TransformerEncoderLayer(
                name='atten_{}'.format(layer_i), stride=stride))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      enc_out = l_out.vec
      tf.global_variables_initializer().run()
      actual_enc_out = sess.run(enc_out)
      seq_len = sl // accumulate_stride if accumulate_stride != 0 else 1
      self.assertAllEqual([bs, seq_len, d], actual_enc_out.shape)

  @parameterized.named_parameters(
      {
          'testcase_name': '_baseline',
          'strides': [1, 1],
      }, {
          'testcase_name': '_stride_2',
          'strides': [2, 1],
      }, {
          'testcase_name': '_first_token',
          'strides': [2, 0],
      })
  def testTransformerStackWithStochasticDepth(self, strides):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      atten_builder = attention.Builder.Params().Set(
          model_dim=d, num_heads=2, ff_hidden_dim=5,
          survival_prob=0.9).Instantiate()
      layers = []
      accumulate_stride = 1
      for layer_i, stride in enumerate(strides):
        accumulate_stride *= stride
        layers.append(
            atten_builder.TransformerEncoderLayer(
                name='atten_{}'.format(layer_i), stride=stride))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      enc_out = l_out.vec
      tf.global_variables_initializer().run()
      actual_enc_out = sess.run(enc_out)
      seq_len = sl // accumulate_stride if accumulate_stride != 0 else 1
      self.assertAllEqual([bs, seq_len, d], actual_enc_out.shape)

  @parameterized.named_parameters(
      {
          'testcase_name': '_baseline',
          'strides': [(1, 6), (1, 3), 3],
      }, {
          'testcase_name': '_stride_2',
          'strides': [(2, 4), (1, None), 2],
      }, {
          'testcase_name': '_first_token',
          'strides': [(2, 5), (0, None), 1],
      })
  def testTransformerStackWithStrideAndOutLength(self, strides):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      atten_builder = attention.Builder.Params().Set(
          model_dim=d, num_heads=2, ff_hidden_dim=5).Instantiate()
      layers = []
      out_seq_len = strides.pop()
      for layer_i, (stride, first_n) in enumerate(strides):
        layers.append(
            atten_builder.TransformerEncoderLayer(
                name='atten_{}'.format(layer_i), stride=stride,
                first_n=first_n))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      enc_out = l_out.vec
      tf.global_variables_initializer().run()
      actual_enc_out = sess.run(enc_out)
      self.assertAllEqual([bs, out_seq_len, d], actual_enc_out.shape)

  @parameterized.named_parameters({
      'testcase_name': '_baseline',
  }, {
      'testcase_name': '_first_token',
      'first_n': 1,
  }, {
      'testcase_name': '_pack_sequences',
      'pack_sequences': 2,
  }, {
      'testcase_name': '_pack_sequences_first_token',
      'pack_sequences': 2,
      'first_n': 1,
  })
  def testStridingWithPackedInput(self, pack_sequences=None, first_n=None):
    with self.session(use_gpu=False) as sess:
      np.random.seed(123)
      bs = 2
      sl = 10
      d = 16
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      segment_mask = None
      if pack_sequences:
        # Pack multiple original sequences into one, delineated with
        # segment_mask.
        input_embs = tf.reshape(input_embs,
                                [bs // pack_sequences, pack_sequences * sl, d])
        paddings = tf.reshape(paddings,
                              [bs // pack_sequences, pack_sequences * sl])
        segment_ids = tf.reshape(
            tf.cumsum(tf.ones([bs, sl]), axis=0),
            [bs // pack_sequences, pack_sequences * sl])
        segment_mask = attention.SegmentMask(segment_ids, segment_ids)
      tf.random.set_seed(12345)
      atten_builder = attention.Builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          packed_input=pack_sequences is not None).Instantiate()
      if first_n is None:
        stride, atten_first_n = (1, None)
      elif pack_sequences:
        stride, atten_first_n = (sl, None)
      else:
        stride, atten_first_n = (0, 1)
      p = atten_builder.TransformerEncoderLayer(
          name='trans', stride=stride, first_n=atten_first_n)
      p.random_seed = 1234
      l = p.Instantiate()
      l_in = py_utils.NestedMap(vec=input_embs, paddings=paddings)
      if segment_mask is not None:
        l_in.segment_mask = segment_mask
      l_out = l.FPropDefaultTheta(l_in)
      enc_out = l_out.vec
      # Get the first token outputs.
      if pack_sequences:
        out_segment_mask = l_out.segment_mask
        if first_n:
          enc_out = py_utils.HasShape(enc_out,
                                      [bs // pack_sequences, pack_sequences, d])
          enc_out = tf.reshape(enc_out, [bs, d])
          self.assertAllEqual(
              out_segment_mask.shape,
              [bs // pack_sequences, 1, pack_sequences, pack_sequences])
        else:
          enc_out = py_utils.HasShape(
              enc_out, [bs // pack_sequences, pack_sequences * sl, d])
          enc_out = tf.reshape(enc_out, [bs, sl, d])
          enc_out = enc_out[:, 0, :]
          self.assertAllEqual(out_segment_mask.shape, [
              bs // pack_sequences, 1, pack_sequences * sl, pack_sequences * sl
          ])
      else:
        if first_n:
          enc_out = py_utils.HasShape(enc_out, [bs, 1, d])
          enc_out = tf.reshape(enc_out, [bs, 1, d])
        else:
          enc_out = py_utils.HasShape(enc_out, [bs, sl, d])
          enc_out = enc_out[:, 0, :]
      tf.global_variables_initializer().run()
      self.assertAllClose(20.82248, sess.run(tf.reduce_sum(enc_out)))

  def testTransformerEncoderWithGatedGelu(self):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(12345)
      atten_builder = attention.Builder.Params().Set(
          model_dim=d, num_heads=2, ff_hidden_dim=5).Instantiate()
      # TODO(huangyp): Change to GatedGeluFeedforward once tf.nn.gelu is in
      # latest release of tensorflow.
      encoder_block = atten_builder.Seq(
          'block', atten_builder._StridedAttention('self_atten', num_heads=2),
          atten_builder.Feedforward('ff', ff_hidden_dim=5))
      layers = []
      for layer_i in range(2):
        layers.append(
            atten_builder.Seq('atten_{}'.format(layer_i), encoder_block))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      enc_out = l_out.vec
      tf.global_variables_initializer().run()
      actual_enc_out = sess.run(enc_out)
      self.assertAllEqual([bs, sl, d], actual_enc_out.shape)

  def testEncoderLayerWithPerLayerParam(self):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 16
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      heads = [1, 2, 4]
      ff_dims = [16, 32, 16]
      atten_builder = attention.Builder.Params().Set(
          model_dim=16, num_heads=heads, ff_hidden_dim=ff_dims).Instantiate()
      layers = []
      for layer_i, (head, ff_dim) in enumerate(zip(heads, ff_dims)):
        layers.append(
            atten_builder.TransformerEncoderLayer(
                name='atten_{}'.format(layer_i),
                ff_hidden_dim=ff_dim,
                num_heads=head,
                stride=1 if layer_i < 2 else 0))
      p = atten_builder.Seq('model', *layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      l_out = l.FPropDefaultTheta(
          py_utils.NestedMap(vec=input_embs, paddings=paddings))
      out = tf.reduce_sum(l_out.vec)
      tf.global_variables_initializer().run()
      actual_out = sess.run(out)
      self.assertAllClose(actual_out, 17.40516)

  def testSerialization(self):
    heads = [1, 2, 4]
    ff_dims = [16, 32, 16]
    atten_builder = attention.Builder.Params().Set(
        model_dim=16, num_heads=heads, ff_hidden_dim=ff_dims).Instantiate()
    layers = []
    for layer_i, (head, ff_dim) in enumerate(zip(heads, ff_dims)):
      layers.append(
          atten_builder.TransformerEncoderLayer(
              name='atten_{}'.format(layer_i),
              ff_hidden_dim=ff_dim,
              num_heads=head,
              stride=1 if layer_i < 2 else 0))
    p = atten_builder.Seq('model', *layers)

    serialized = p.ToProto()
    p2 = hyperparams.InstantiableParams.FromProto(serialized)
    self.assertLen(p2.sub, len(p.sub))


class LmBuilderTest(test_utils.TestCase):

  def _testGraph(self, dtype=tf.float32):
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    atten_builder = attention.LmBuilder.Params().Set(
        model_dim=4, num_heads=2, ff_hidden_dim=16, dtype=dtype)
    params = atten_builder.Instantiate().TransformerEncoderStack(
        name='xformer', num_layers=2)
    params.dtype = dtype
    params.random_seed = 0
    params.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
    l = params.Instantiate()
    l_in = tf.constant(np.random.rand(2, 3, 4), dtype=dtype)
    l_padding = tf.zeros([2, 3], dtype=dtype)
    l_out = l.FPropDefaultTheta(
        py_utils.NestedMap(vec=l_in, paddings=l_padding))
    return l_out.vec

  def testFprop(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      l_out = self._testGraph()
      l_out = tf.reduce_sum(l_out)
      tf.global_variables_initializer().run()
      l_out_eval = sess.run(l_out)
      self.assertAllClose(36.04808, l_out_eval)

  def testBProp(self):
    with self.session(use_gpu=True) as sess:
      output = self._testGraph(dtype=tf.float64)
      loss = tf.reduce_sum(output)
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      tf.global_variables_initializer().run()
      sym_grads = [sg.eval() for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]
      for ng, sg in zip(num_grads, sym_grads):
        self.assertAllClose(ng, sg, rtol=5e-02, atol=5e-02)


def _CreateDummyParams(field_names):
  p = hyperparams.Params()
  for name in field_names:
    p.Define(name, None, 'Dummy')
  return p


class DummyDecoderRNNT(base_layer.BaseLayer):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'dummy_decoder_rnnt'
    p.Define('emb', _CreateDummyParams(['vocab_size']), 'Dummy emb.')
    p.Define('target_seq_len', 20, 'Dummy target seq len.')
    p.Define('num_classes', None, 'Dummy num classes.')
    return p

  @classmethod
  def UpdateTargetVocabSize(cls, p, vocab_size, wpm_model=None):
    p.emb.vocab_size = vocab_size
    p.num_classes = vocab_size
    return p


class RelativeAttentionHelperTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('MultiHeadedAttentionXL', attention.MultiHeadedAttentionXL,
       attention.MultiHeadedAttention),
      ('LocalSelfAttentionXL', attention.LocalSelfAttentionXL,
       attention.LocalSelfAttention))
  def testClearRelativeAttentionInTransformerLayer(self, atten_cls,
                                                   expected_atten_cls):
    """Tests scenarios in clear relative attention in transformer layer."""
    trans_p = attention.TransformerLayer.Params()
    # set attention params in transformer layer.
    input_dim = 4
    rel_pos_emb_dim = 4
    # Set rel_pos_emb_dim in attention params.
    trans_p.tr_atten_tpl.atten_tpl = (
        atten_cls.Params().Set(
            input_dim=input_dim, rel_pos_emb_dim=rel_pos_emb_dim))
    new_trans_p = attention.ClearRelativeAttentionInTransformerLayer(trans_p)
    tr_atten_tpl = new_trans_p.tr_self_atten_tpl.atten_tpl
    self.assertEqual(tr_atten_tpl.cls, expected_atten_cls)
    self.assertEqual(tr_atten_tpl.input_dim, input_dim)

  def testClearRelativeAttentionTransformerLayerNotSupportedError(self):
    transformer_params = DummyDecoderRNNT.Params()
    with self.assertRaises(ValueError):
      _ = attention.ClearRelativeAttentionInTransformerLayer(transformer_params)

  def testClearRelativeAttentionAttentionParamsNotSupportedError(self):
    trans_p = attention.TransformerLayer.Params()
    # MultiHeadedAttention is not supported in ClearRelativeAttention.
    attention_params = attention.MultiHeadedAttention.Params()
    trans_p.tr_atten_tpl.atten_tpl = attention_params
    with self.assertRaises(ValueError):
      _ = attention.ClearRelativeAttentionInTransformerLayer(trans_p)

  @parameterized.named_parameters(
      ('AttentionParamsNotSupported', _CreateDummyParams(
          ['name', 'cls']), attention.ATTEN_TRANSFORMER_XL),
      ('AttentionTypeNotSupported', attention.MultiHeadedAttention.Params(),
       'unsupported_atten_type'))
  def testUseRelativeAttentionInTransformerLayerValueError(
      self, attention_params, attention_type):
    """Tests unsupported Use Relative Attention cases."""
    transformer_param = attention.TransformerLayer.Params()
    transformer_param.tr_atten_tpl.atten_tpl = attention_params
    rel_pos_emb_dim = 4
    with self.assertRaises(ValueError):
      _ = attention.UseRelativeAttentionInTransformerLayer(
          transformer_param, rel_pos_emb_dim, atten_type=attention_type)

  def testUseRelativeAttentionInTransformerLayerNotSupportedError(self):
    """Tests unsupported input transformer params in Use Relative Attention."""
    transformer_params = DummyDecoderRNNT.Params()
    with self.assertRaises(ValueError):
      _ = attention.UseRelativeAttentionInTransformerLayer(
          transformer_params, 4, atten_type=attention.ATTEN_TRANSFORMER_XL)

  @parameterized.named_parameters(
      ('MultiHeadedAttention', attention.MultiHeadedAttention,
       attention.MultiHeadedAttentionXL, attention.ATTEN_TRANSFORMER_XL),
      ('LocalSelfAttention', attention.LocalSelfAttention,
       attention.LocalSelfAttentionXL, attention.ATTEN_TRANSFORMER_XL),
      ('MultiHeadedAttentionRPE', attention.MultiHeadedAttention,
       attention.MultiHeadedAttentionRPE, attention.ATTEN_RPE))
  def testUseRelativeAttentionInTransformerLayer(self, atten_cls,
                                                 expected_atten_cls,
                                                 atten_type):
    """Tests different scenarios in Use Relative Attention."""
    trans_p = attention.TransformerLayer.Params()
    # set attenion params in transformer layer.
    input_dim = 4
    trans_p.tr_atten_tpl.atten_tpl = atten_cls.Params().Set(input_dim=input_dim)
    rel_pos_emb_dim = 4
    new_trans_p = attention.UseRelativeAttentionInTransformerLayer(
        trans_p, rel_pos_emb_dim, atten_type=atten_type)
    tr_atten_tpl = new_trans_p.tr_self_atten_tpl.atten_tpl
    self.assertEqual(tr_atten_tpl.cls, expected_atten_cls)
    self.assertEqual(tr_atten_tpl.rel_pos_emb_dim, rel_pos_emb_dim)
    self.assertEqual(tr_atten_tpl.input_dim, input_dim)


class ResidualAddLayerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'apply_residual',
          'apply_residual': True,
          'residual_weight': 1.0,
          'expected_output': [[0.3, 0.5, 0.7]]
      }, {
          'testcase_name': 'no_residual',
          'apply_residual': False,
          'residual_weight': 1.0,
          'expected_output': [[0.2, 0.3, 0.4]]
      }, {
          'testcase_name': 'apply_residual_w_weight',
          'apply_residual': True,
          'residual_weight': 0.5,
          'expected_output': [[0.2, 0.35, 0.5]]
      }, {
          'testcase_name': 'no_residual_w_weight',
          'apply_residual': False,
          'residual_weight': 0.5,
          'expected_output': [[0.1, 0.15, 0.2]]
      })
  def testClearRelativeAttentionInTransformerLayer(self, apply_residual,
                                                   residual_weight,
                                                   expected_output):
    x = tf.constant([[0.1, 0.2, 0.3]])
    fx = tf.constant([[0.2, 0.3, 0.4]])
    p = attention.ResidualAddLayer.Params().Set(
        name='residual_test',
        residual_weight=residual_weight,
        apply_residual=apply_residual)
    l = p.Instantiate()
    ret = l.FPropDefaultTheta(x, fx)
    init = tf.group(
        [tf.global_variables_initializer(),
         tf.local_variables_initializer()])
    with self.session(use_gpu=False) as sess:
      sess.run(init)
      ret_val = sess.run(ret)
    self.assertAllClose(ret_val, np.array(expected_output))


if __name__ == '__main__':
  tf.test.main()
