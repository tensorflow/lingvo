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
"""Tests for batch_major self attention."""

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import batch_major_attention as mt_attention
from lingvo.core import py_utils
from lingvo.core import self_attention_layer as self_attention
from lingvo.core import test_utils
import numpy as np


class BuilderTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '_baseline',
          'num_splits': 1,
          'num_micro_batches': 1
      }, {
          'testcase_name': '_two_splits',
          'num_splits': 2,
          'num_micro_batches': 2
      }, {
          'testcase_name': '_one_split_two_micro_batches',
          'num_splits': 1,
          'num_micro_batches': 2
      })
  def testTransformerStack(self, num_splits, num_micro_batches):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      d = 16
      tf.random.set_seed(12345)
      deterministic_dropout = num_splits > 1 or num_micro_batches > 1
      atten_builder = self_attention.Builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          deterministic_dropout=deterministic_dropout,
          num_splits=num_splits,
          num_micro_batches=num_micro_batches)
      p = atten_builder.Instantiate().TransformerStack('atten', 6)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      segment_mask = tf.zeros([bs, 1, sl, sl])

      out = l.FPropDefaultTheta(
          py_utils.NestedMap(
              vec=input_embs, paddings=paddings, segment_mask=segment_mask))
      enc_out = out.vec
      tf.logging.info('enc_out={}'.format(enc_out.shape))
      enc_out_sum = tf.reduce_sum(enc_out)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum = sess.run([enc_out, enc_out_sum])
      print('actual_enc_out_sum=', actual_enc_out_sum)

      self.assertAllEqual(actual_enc_out.shape, [bs, sl, d])
      self.assertAllClose(386.16741943359375, actual_enc_out_sum, atol=1e-5)

  @parameterized.named_parameters(
      {
          'testcase_name': '_v1_stack',
          'use_v1_stack': True,
      }, {
          'testcase_name': '_baseline',
          'first_n': None,
      }, {
          'testcase_name': '_first_1',
          'first_n': 1,
      }, {
          'testcase_name': '_first_2',
          'first_n': 2,
      }, {
          'testcase_name': '_stride_2',
          'stride': 2,
      })
  def testTransformerStackV2(self, use_v1_stack=False, stride=1, first_n=None):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      d = 16
      tf.random.set_seed(12345)
      atten_builder = self_attention.Builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          deterministic_dropout=False,
          num_splits=1,
          num_micro_batches=1)
      builder = atten_builder.Instantiate()
      if use_v1_stack:
        p = builder.TransformerStack('atten', num_layers=3)
      else:
        p = builder.TransformerStackV2(
            'atten',
            num_layers=3,
            final_layer_stride=stride,
            final_layer_first_n=first_n)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      self.assertAllEqual([
          'atten/iter_000/block/ff/feedforward/bias01/b/var',
          'atten/iter_000/block/ff/feedforward/bias02/b/var',
          'atten/iter_000/block/ff/feedforward/linear01/w/var',
          'atten/iter_000/block/ff/feedforward/linear02/w/var',
          'atten/iter_000/block/ff/feedforward/ln/bias/var',
          'atten/iter_000/block/ff/feedforward/ln/scale/var',
          'atten/iter_000/block/self_atten/LN/bias/var',
          'atten/iter_000/block/self_atten/LN/scale/var',
          'atten/iter_000/block/self_atten/atten/key/b/var',
          'atten/iter_000/block/self_atten/atten/key/w/var',
          'atten/iter_000/block/self_atten/atten/per_dim_scale/per_dim_scale/var',
          'atten/iter_000/block/self_atten/atten/post/b/var',
          'atten/iter_000/block/self_atten/atten/post/w/var',
          'atten/iter_000/block/self_atten/atten/query/b/var',
          'atten/iter_000/block/self_atten/atten/query/w/var',
          'atten/iter_000/block/self_atten/atten/value/b/var',
          'atten/iter_000/block/self_atten/atten/value/w/var',
          'atten/iter_001/block/ff/feedforward/bias01/b/var',
          'atten/iter_001/block/ff/feedforward/bias02/b/var',
          'atten/iter_001/block/ff/feedforward/linear01/w/var',
          'atten/iter_001/block/ff/feedforward/linear02/w/var',
          'atten/iter_001/block/ff/feedforward/ln/bias/var',
          'atten/iter_001/block/ff/feedforward/ln/scale/var',
          'atten/iter_001/block/self_atten/LN/bias/var',
          'atten/iter_001/block/self_atten/LN/scale/var',
          'atten/iter_001/block/self_atten/atten/key/b/var',
          'atten/iter_001/block/self_atten/atten/key/w/var',
          'atten/iter_001/block/self_atten/atten/per_dim_scale/per_dim_scale/var',
          'atten/iter_001/block/self_atten/atten/post/b/var',
          'atten/iter_001/block/self_atten/atten/post/w/var',
          'atten/iter_001/block/self_atten/atten/query/b/var',
          'atten/iter_001/block/self_atten/atten/query/w/var',
          'atten/iter_001/block/self_atten/atten/value/b/var',
          'atten/iter_001/block/self_atten/atten/value/w/var',
          'atten/iter_002/block/ff/feedforward/bias01/b/var',
          'atten/iter_002/block/ff/feedforward/bias02/b/var',
          'atten/iter_002/block/ff/feedforward/linear01/w/var',
          'atten/iter_002/block/ff/feedforward/linear02/w/var',
          'atten/iter_002/block/ff/feedforward/ln/bias/var',
          'atten/iter_002/block/ff/feedforward/ln/scale/var',
          'atten/iter_002/block/self_atten/LN/bias/var',
          'atten/iter_002/block/self_atten/LN/scale/var',
          'atten/iter_002/block/self_atten/atten/key/b/var',
          'atten/iter_002/block/self_atten/atten/key/w/var',
          'atten/iter_002/block/self_atten/atten/per_dim_scale/per_dim_scale/var',
          'atten/iter_002/block/self_atten/atten/post/b/var',
          'atten/iter_002/block/self_atten/atten/post/w/var',
          'atten/iter_002/block/self_atten/atten/query/b/var',
          'atten/iter_002/block/self_atten/atten/query/w/var',
          'atten/iter_002/block/self_atten/atten/value/b/var',
          'atten/iter_002/block/self_atten/atten/value/w/var',
      ], [var.op.name for var in tf.nest.flatten(l.vars)])
      input_embs = tf.constant(
          np.random.random(size=[bs, sl, d]), dtype=np.float)
      paddings = tf.zeros([bs, sl])
      segment_mask = tf.zeros([bs, 1, sl, sl])

      out = l.FPropDefaultTheta(
          py_utils.NestedMap(
              vec=input_embs, paddings=paddings, segment_mask=segment_mask))
      enc_out = out.vec
      if first_n is None:
        first_n = sl
      enc_out = py_utils.HasShape(enc_out,
                                  [bs, (first_n + stride - 1) // stride, d])
      # Only test the value of the first token.
      enc_out = enc_out[:, :1, :]
      tf.logging.info('enc_out={}'.format(enc_out.shape))
      enc_out_sum = tf.reduce_sum(enc_out)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum = sess.run([enc_out, enc_out_sum])
      print('actual_enc_out_sum=', actual_enc_out_sum)

      self.assertAllEqual(actual_enc_out.shape, [bs, 1, d])
      self.assertAllClose(21.429626, actual_enc_out_sum, atol=1e-5)


class TransformerLayerTest(test_utils.TestCase, parameterized.TestCase):

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

    segment_mask = tf.zeros([2, 1, 5, 5])
    return query_vec, paddings, aux_vec, aux_paddings, segment_mask

  def testTransformerAttentionLayerFPropCaseEncoder(self):
    with self.session(use_gpu=True) as sess:
      query_vec, paddings, _, _, segment_mask = self._TransformerAttentionLayerInputs(
      )

      expected_p = mt_attention.TransformerAttentionLayer.Params().Set(
          name='transformer_self_atten',
          input_dim=4,
          is_masked=False,
          num_heads=2)
      expected_p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      expected_l = expected_p.Instantiate()
      expected_ctx_vec, _ = expected_l.FProp(expected_l.theta, query_vec, None,
                                             paddings)

      atten_builder = self_attention.Builder.Params().Set(
          model_dim=4, num_heads=2, selfatten_add_unnormalized_input=True)
      p = atten_builder.Instantiate().SelfAttention('self_atten')
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      l_out = l.FProp(
          l.theta,
          py_utils.NestedMap(
              vec=query_vec, paddings=paddings, segment_mask=segment_mask))
      ctx_vec = l_out.vec
      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx_vec)
      expected_ctx = sess.run(expected_ctx_vec)
      self.assertAllClose(expected_ctx, actual_ctx)

  def testTransformerLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      src_vec, _, _, _, segment_mask = self._TransformerAttentionLayerInputs()
      paddings = tf.zeros([2, 5])
      p = mt_attention.TransformerLayer.Params()
      p.name = 'transformer_encoder_layer'
      p.input_dim = 4
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_heads = 2
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      enc_vec, _ = l.FProp(l.theta, src_vec, paddings, None, None)

      self_attention_p = self_attention.Builder.Params().Set(
          model_dim=4,
          num_heads=2,
          ff_hidden_dim=7,
          selfatten_add_unnormalized_input=True)
      self_attention_p = self_attention_p.Instantiate().TransformerStack(
          name='transformer_encoder_layer', num_layers=1)
      self_attention_p.params_init = py_utils.WeightInit.Xavier(
          scale=1.0, seed=0)
      self_attention_l = self_attention_p.Instantiate()
      self_attention_enc_out = self_attention_l.FProp(
          self_attention_l.theta,
          py_utils.NestedMap(
              vec=src_vec, paddings=paddings, segment_mask=segment_mask))
      self_attention_enc_vec = self_attention_enc_out.vec
      tf.global_variables_initializer().run()
      enc_vec = sess.run(enc_vec)
      self_attention_enc_vec = sess.run(self_attention_enc_vec)
      self.assertAllClose(enc_vec, self_attention_enc_vec)


if __name__ == '__main__':
  tf.test.main()
