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
from scipy.special import softmax


class BuilderTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '_baseline', 'num_splits': 1, 'num_micro_batches': 1},
      {'testcase_name': '_two_splits', 'num_splits': 2, 'num_micro_batches': 2},
      {
          'testcase_name': '_one_split_two_micro_batches',
          'num_splits': 1,
          'num_micro_batches': 2,
      },
      {
          'testcase_name': '_atten_tpl_list',
          'num_splits': 1,
          'num_micro_batches': 2,
          'atten_tpl': [
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
          ],
      },
      {
          'testcase_name': '_simplified_transformer',
          'num_splits': 1,
          'num_micro_batches': 1,
          'builder': self_attention.SimplifiedTransformerBuilder,
          'expected_output': 39.930980,
      },
      {
          'testcase_name': '_simplified_transformer_atten_tpl_list',
          'num_splits': 1,
          'num_micro_batches': 1,
          'builder': self_attention.SimplifiedTransformerBuilder,
          'atten_tpl': [
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
          ],
          'expected_output': 39.930980,
      },
      {
          'testcase_name': '_simplified_transformer_parallel',
          'num_splits': 1,
          'num_micro_batches': 1,
          'builder': self_attention.SimplifiedTransformerBuilder,
          'parallel_attention_mlp': True,
          'expected_output': 28.284629,
      },
  )
  def testTransformerStack(
      self,
      num_splits,
      num_micro_batches,
      builder=self_attention.Builder,
      parallel_attention_mlp=False,
      atten_tpl=None,
      expected_output=386.16742,
  ):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      d = 16
      num_layers = 6
      tf.random.set_seed(12345)
      deterministic_dropout = num_splits > 1 or num_micro_batches > 1
      atten_builder = builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          deterministic_dropout=deterministic_dropout,
          num_splits=num_splits,
          num_micro_batches=num_micro_batches,
          atten_tpl=atten_tpl or mt_attention.MultiHeadedAttention.Params(),
      )
      if builder is self_attention.SimplifiedTransformerBuilder:
        atten_builder.Set(
            parallel_attention_mlp=parallel_attention_mlp,
        )
        if isinstance(atten_builder.atten_tpl, list):
          atten_builder.atten_tpl = [
              atten_builder.atten_tpl[i].Set(enable_shaped_attention=True)
              for i in range(len(atten_builder.atten_tpl))
          ]
        else:
          atten_builder.atten_tpl.enable_shaped_attention = True
      p = atten_builder.Instantiate().TransformerStack('atten', num_layers)
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      input_embs = tf.constant(np.random.random(size=[bs, sl, d]), dtype=float)
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
      self.assertAllClose(expected_output, actual_enc_out_sum, atol=1e-5)

  @parameterized.named_parameters(
      {
          'testcase_name': '_v1_stack',
          'use_v1_stack': True,
      },
      {
          'testcase_name': '_v1_stack_atten_tpl_list',
          'use_v1_stack': True,
          'atten_tpl': [
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
          ],
      },
      {
          'testcase_name': '_baseline',
          'first_n': None,
      },
      {
          'testcase_name': '_baseline_atten_tpl_list',
          'first_n': None,
          'atten_tpl': [
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
          ],
      },
      {
          'testcase_name': '_first_1',
          'first_n': 1,
      },
      {
          'testcase_name': '_first_2',
          'first_n': 2,
      },
      {
          'testcase_name': '_stride_2',
          'stride': 2,
      },
  )
  def testTransformerStackV2(
      self,
      use_v1_stack=False,
      stride=1,
      first_n=None,
      atten_tpl=mt_attention.MultiHeadedAttention.Params(),
  ):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      d = 16
      num_layers = 3
      tf.random.set_seed(12345)
      atten_builder = self_attention.Builder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          deterministic_dropout=False,
          num_splits=1,
          num_micro_batches=1,
          atten_tpl=atten_tpl,
      )
      builder = atten_builder.Instantiate()
      if use_v1_stack:
        p = builder.TransformerStack('atten', num_layers=num_layers)
      else:
        p = builder.TransformerStackV2(
            'atten',
            num_layers=num_layers,
            final_layer_stride=stride,
            final_layer_first_n=first_n,
        )
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
      input_embs = tf.constant(np.random.random(size=[bs, sl, d]), dtype=float)
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

  @parameterized.named_parameters(
      {
          'testcase_name': '_v1_stack',
          'use_v1_stack': True,
      },
      {
          'testcase_name': '_v1_stack_atten_tpl_list',
          'use_v1_stack': True,
          'atten_tpl': [
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
          ],
      },
      {
          'testcase_name': '_baseline',
          'first_n': None,
      },
      {
          'testcase_name': '_baseline_atten_tpl_list',
          'first_n': None,
          'atten_tpl': [
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
              mt_attention.MultiHeadedAttention.Params(),
          ],
      },
      {
          'testcase_name': '_first_1',
          'first_n': 1,
      },
      {
          'testcase_name': '_first_2',
          'first_n': 2,
      },
      {
          'testcase_name': '_stride_2',
          'stride': 2,
      },
      {
          'testcase_name': '_stride_2_first_2',
          'stride': 2,
          'first_n': 2,
      },
  )
  def testTransformerStackV2WithSimplifiedTransformer(
      self,
      use_v1_stack=False,
      stride=1,
      first_n=None,
      atten_tpl=mt_attention.MultiHeadedAttention.Params(),
  ):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      d = 16
      num_layers = 3
      tf.random.set_seed(12345)
      atten_builder = self_attention.SimplifiedTransformerBuilder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          deterministic_dropout=False,
          num_splits=1,
          num_micro_batches=1,
          selfatten_enable_value_proj=False,
          atten_tpl=atten_tpl,
      )
      if isinstance(atten_builder.atten_tpl, list):
        atten_builder.atten_tpl = [
            atten_builder.atten_tpl[i].Set(
                enable_shaped_attention=True, enable_ctx_post_proj=False
            )
            for i in range(len(atten_builder.atten_tpl))
        ]
      else:
        atten_builder.atten_tpl.enable_shaped_attention = True
        atten_builder.atten_tpl.enable_ctx_post_proj = False
      builder = atten_builder.Instantiate()
      if use_v1_stack:
        p = builder.TransformerStack('atten', num_layers=num_layers)
      else:
        p = builder.TransformerStackV2(
            'atten',
            num_layers=num_layers,
            final_layer_stride=stride,
            final_layer_first_n=first_n,
        )
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      self.assertAllEqual(
          [
              'atten/iter_000/block/LN/bias/var',
              'atten/iter_000/block/LN/scale/var',
              'atten/iter_000/block/atten/key/b/var',
              'atten/iter_000/block/atten/key/w/var',
              'atten/iter_000/block/atten/per_dim_scale/per_dim_scale/var',
              'atten/iter_000/block/atten/query/b/var',
              'atten/iter_000/block/atten/query/w/var',
              'atten/iter_000/block/atten/shaped_attn_alpha/var',
              'atten/iter_000/block/atten/shaped_attn_beta/var',
              'atten/iter_000/block/atten/shaped_attn_gamma/var',
              'atten/iter_000/block/ff/feedforward/bias01/b/var',
              'atten/iter_000/block/ff/feedforward/bias02/b/var',
              'atten/iter_000/block/ff/feedforward/linear01/w/var',
              'atten/iter_000/block/ff/feedforward/linear02/w/var',
              'atten/iter_000/block/ff/feedforward/ln/bias/var',
              'atten/iter_000/block/ff/feedforward/ln/scale/var',
              'atten/iter_001/block/LN/bias/var',
              'atten/iter_001/block/LN/scale/var',
              'atten/iter_001/block/atten/key/b/var',
              'atten/iter_001/block/atten/key/w/var',
              'atten/iter_001/block/atten/per_dim_scale/per_dim_scale/var',
              'atten/iter_001/block/atten/query/b/var',
              'atten/iter_001/block/atten/query/w/var',
              'atten/iter_001/block/atten/shaped_attn_alpha/var',
              'atten/iter_001/block/atten/shaped_attn_beta/var',
              'atten/iter_001/block/atten/shaped_attn_gamma/var',
              'atten/iter_001/block/ff/feedforward/bias01/b/var',
              'atten/iter_001/block/ff/feedforward/bias02/b/var',
              'atten/iter_001/block/ff/feedforward/linear01/w/var',
              'atten/iter_001/block/ff/feedforward/linear02/w/var',
              'atten/iter_001/block/ff/feedforward/ln/bias/var',
              'atten/iter_001/block/ff/feedforward/ln/scale/var',
              'atten/iter_002/block/LN/bias/var',
              'atten/iter_002/block/LN/scale/var',
              'atten/iter_002/block/atten/key/b/var',
              'atten/iter_002/block/atten/key/w/var',
              'atten/iter_002/block/atten/per_dim_scale/per_dim_scale/var',
              'atten/iter_002/block/atten/query/b/var',
              'atten/iter_002/block/atten/query/w/var',
              'atten/iter_002/block/atten/shaped_attn_alpha/var',
              'atten/iter_002/block/atten/shaped_attn_beta/var',
              'atten/iter_002/block/atten/shaped_attn_gamma/var',
              'atten/iter_002/block/ff/feedforward/bias01/b/var',
              'atten/iter_002/block/ff/feedforward/bias02/b/var',
              'atten/iter_002/block/ff/feedforward/linear01/w/var',
              'atten/iter_002/block/ff/feedforward/linear02/w/var',
              'atten/iter_002/block/ff/feedforward/ln/bias/var',
              'atten/iter_002/block/ff/feedforward/ln/scale/var',
          ],
          [var.op.name for var in tf.nest.flatten(l.vars)],
      )
      input_embs = tf.constant(np.random.random(size=[bs, sl, d]), dtype=float)
      paddings = tf.zeros([bs, sl])
      segment_mask = tf.zeros([bs, 1, sl, sl])

      out = l.FPropDefaultTheta(
          py_utils.NestedMap(
              vec=input_embs, paddings=paddings, segment_mask=segment_mask
          )
      )
      enc_out = out.vec
      if first_n is None:
        first_n = sl
      enc_out = py_utils.HasShape(
          enc_out, [bs, (first_n + stride - 1) // stride, d]
      )
      # Only test the value of the first token.
      enc_out = enc_out[:, :1, :]
      tf.logging.info('enc_out={}'.format(enc_out.shape))
      enc_out_sum = tf.reduce_sum(enc_out)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum = sess.run([enc_out, enc_out_sum])
      print('actual_enc_out_sum=', actual_enc_out_sum)

      self.assertAllEqual(actual_enc_out.shape, [bs, 1, d])
      self.assertAllClose(2.468383, actual_enc_out_sum, atol=1e-5)

  @parameterized.named_parameters(
      {
          'testcase_name': '_v1_stack',
          'use_v1_stack': True,
      },
      {
          'testcase_name': '_baseline',
          'first_n': None,
      },
      {
          'testcase_name': '_first_1',
          'first_n': 1,
      },
      {
          'testcase_name': '_first_2',
          'first_n': 2,
      },
      {
          'testcase_name': '_stride_2',
          'stride': 2,
      },
      {
          'testcase_name': '_stride_2_first_2',
          'stride': 2,
          'first_n': 2,
      },
  )
  def testTransformerStackV2WithSimplifiedTransformerWithMQA(
      self, use_v1_stack=False, stride=1, first_n=None
  ):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      d = 16
      tf.random.set_seed(12345)
      atten_builder = self_attention.SimplifiedTransformerBuilder.Params().Set(
          model_dim=d,
          num_heads=2,
          ff_hidden_dim=5,
          deterministic_dropout=False,
          num_splits=1,
          num_micro_batches=1,
      )
      atten_builder.atten_tpl.enable_shaped_attention = True
      atten_builder.atten_tpl.enable_ctx_post_proj = False
      atten_builder.atten_tpl.use_mqa = True
      atten_builder.atten_tpl.num_kv_heads = 1
      builder = atten_builder.Instantiate()
      if use_v1_stack:
        p = builder.TransformerStack('atten', num_layers=3)
      else:
        p = builder.TransformerStackV2(
            'atten',
            num_layers=3,
            final_layer_stride=stride,
            final_layer_first_n=first_n,
        )
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()
      self.assertAllEqual(
          [
              'atten/iter_000/block/LN/bias/var',
              'atten/iter_000/block/LN/scale/var',
              'atten/iter_000/block/atten/kv/b/var',
              'atten/iter_000/block/atten/kv/w/var',
              'atten/iter_000/block/atten/per_dim_scale/per_dim_scale/var',
              'atten/iter_000/block/atten/query/b/var',
              'atten/iter_000/block/atten/query/w/var',
              'atten/iter_000/block/atten/shaped_attn_alpha/var',
              'atten/iter_000/block/atten/shaped_attn_beta/var',
              'atten/iter_000/block/atten/shaped_attn_gamma/var',
              'atten/iter_000/block/ff/feedforward/bias01/b/var',
              'atten/iter_000/block/ff/feedforward/bias02/b/var',
              'atten/iter_000/block/ff/feedforward/linear01/w/var',
              'atten/iter_000/block/ff/feedforward/linear02/w/var',
              'atten/iter_000/block/ff/feedforward/ln/bias/var',
              'atten/iter_000/block/ff/feedforward/ln/scale/var',
              'atten/iter_001/block/LN/bias/var',
              'atten/iter_001/block/LN/scale/var',
              'atten/iter_001/block/atten/kv/b/var',
              'atten/iter_001/block/atten/kv/w/var',
              'atten/iter_001/block/atten/per_dim_scale/per_dim_scale/var',
              'atten/iter_001/block/atten/query/b/var',
              'atten/iter_001/block/atten/query/w/var',
              'atten/iter_001/block/atten/shaped_attn_alpha/var',
              'atten/iter_001/block/atten/shaped_attn_beta/var',
              'atten/iter_001/block/atten/shaped_attn_gamma/var',
              'atten/iter_001/block/ff/feedforward/bias01/b/var',
              'atten/iter_001/block/ff/feedforward/bias02/b/var',
              'atten/iter_001/block/ff/feedforward/linear01/w/var',
              'atten/iter_001/block/ff/feedforward/linear02/w/var',
              'atten/iter_001/block/ff/feedforward/ln/bias/var',
              'atten/iter_001/block/ff/feedforward/ln/scale/var',
              'atten/iter_002/block/LN/bias/var',
              'atten/iter_002/block/LN/scale/var',
              'atten/iter_002/block/atten/kv/b/var',
              'atten/iter_002/block/atten/kv/w/var',
              'atten/iter_002/block/atten/per_dim_scale/per_dim_scale/var',
              'atten/iter_002/block/atten/query/b/var',
              'atten/iter_002/block/atten/query/w/var',
              'atten/iter_002/block/atten/shaped_attn_alpha/var',
              'atten/iter_002/block/atten/shaped_attn_beta/var',
              'atten/iter_002/block/atten/shaped_attn_gamma/var',
              'atten/iter_002/block/ff/feedforward/bias01/b/var',
              'atten/iter_002/block/ff/feedforward/bias02/b/var',
              'atten/iter_002/block/ff/feedforward/linear01/w/var',
              'atten/iter_002/block/ff/feedforward/linear02/w/var',
              'atten/iter_002/block/ff/feedforward/ln/bias/var',
              'atten/iter_002/block/ff/feedforward/ln/scale/var',
          ],
          [var.op.name for var in tf.nest.flatten(l.vars)],
      )
      input_embs = tf.constant(np.random.random(size=[bs, sl, d]), dtype=float)
      paddings = tf.zeros([bs, sl])
      segment_mask = tf.zeros([bs, 1, sl, sl])

      out = l.FPropDefaultTheta(
          py_utils.NestedMap(
              vec=input_embs, paddings=paddings, segment_mask=segment_mask
          )
      )
      enc_out = out.vec
      if first_n is None:
        first_n = sl
      enc_out = py_utils.HasShape(
          enc_out, [bs, (first_n + stride - 1) // stride, d]
      )
      # Only test the value of the first token.
      enc_out = enc_out[:, :1, :]
      tf.logging.info('enc_out={}'.format(enc_out.shape))
      enc_out_sum = tf.reduce_sum(enc_out)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum = sess.run([enc_out, enc_out_sum])
      print('actual_enc_out_sum=', actual_enc_out_sum)

      self.assertAllEqual(actual_enc_out.shape, [bs, 1, d])
      self.assertAllClose(0.740105, actual_enc_out_sum, atol=1e-5)


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


class BlockSparseAttentionTest(test_utils.TestCase, parameterized.TestCase):
  """Test block sparse attention."""

  def _AttentionInputs(self, batch_size, seq_len, input_dim, dtype=tf.float32):
    np.random.seed(6348575)
    input_vecs_p = [
        np.random.rand(seq_len, input_dim) for _ in range(batch_size)
    ]
    input_vecs = tf.stack([tf.constant(x, dtype=dtype) for x in input_vecs_p])
    input_padding_p = np.zeros([batch_size, seq_len])
    input_padding = tf.constant(input_padding_p, dtype=dtype)

    return input_vecs, input_padding, input_vecs_p, input_padding_p

  @parameterized.named_parameters(
      {
          'testcase_name': '_local_attention',
          'expected_output': 11.933180,
      },
      {
          'testcase_name': '_local_attention_mqa',
          'use_mqa': True,
          'num_kv_heads': 1,
          'expected_output': 4.652255,
      },
      {
          'testcase_name': '_local_attention_gqa',
          'use_mqa': True,
          'num_kv_heads': 2,
          'expected_output': 5.873354,
      },
  )
  def testBlockSparseAttentionFprop(
      self,
      use_mqa=False,
      num_kv_heads=None,
      expected_output=None,
  ):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 4
      ws = 2  # block size.
      tf.random.set_seed(12345)
      (input_vecs, input_padding, _, _) = self._AttentionInputs(bs, sl, d)
      segment_mask = None

      p = self_attention.BlockSparseAttention.Params().Set(
          name='self_atten',
          input_dim=d,
          hidden_dim=4,
          num_heads=4,
          proj_tpl=mt_attention.MultiHeadedProjectionLayer.Params().Set(
              input_proj_bias_rank_3=True
          ),
          src_block_size=ws,
          tgt_block_size=ws,
          use_mqa=use_mqa,
          num_kv_heads=num_kv_heads,
      )
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()

      encoded, _ = l.FPropDefaultTheta(
          query_vec=input_vecs,
          key_vec=input_vecs,
          value_vec=input_vecs,
          paddings=input_padding,
          segment_mask=segment_mask,
      )
      encoder_sum = tf.reduce_sum(encoded)
      tf.global_variables_initializer().run()
      encoded_out, encoded_out_sum = sess.run([encoded, encoder_sum])
      self.assertAllEqual(encoded_out.shape, (bs, sl, d))
      self.assertAllClose(expected_output, encoded_out_sum, atol=1e-5)

  def _ComputeLocalContext(
      self, query, key, value, band_mask, logits_eqn=None, ctx_eqn=None
  ):
    """Computing local context.

    Args:
      query: [b, n, l, t, h]
      key: [b, n, l, s, h]
      value: [b, n, l, s, h]
      band_mask: [b, 1, l, t, s]
      logits_eqn: Einsum string
      ctx_eqn: Einsum string

    Returns:
    """
    logits_eqn = logits_eqn or 'bnlth,bnlsh->bnlts'
    ctx_eqn = ctx_eqn or 'bnlts,bnlsh->bnlth'
    band_logits = np.einsum(logits_eqn, query, key)
    band_logits += (1.0 - band_mask) * -1e9
    band_probs = softmax(band_logits, axis=-1)
    band_context = np.einsum(
        ctx_eqn,
        band_probs,
        value,
    )
    return band_context

  @parameterized.named_parameters(
      {
          'testcase_name': '_local_attention',
      },
      {
          'testcase_name': '_local_attention_mqa',
          'use_mqa': True,
          'num_kv_heads': 1,
      },
      {
          'testcase_name': '_local_attention_gqa',
          'use_mqa': True,
          'num_kv_heads': 2,
      },
  )
  def testBlockSparseAttentionOutput(self, use_mqa=False, num_kv_heads=None):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 10
      d = 4
      num_heads = 4
      src_block_size = 2
      tgt_block_size = 2
      tf.random.set_seed(12345)
      (input_vecs, input_padding, input_vecs_p, input_padding_p) = (
          self._AttentionInputs(bs, sl, d)
      )

      p = self_attention.BlockSparseAttention.Params().Set(
          name='self_atten',
          input_dim=d,
          hidden_dim=d,
          num_heads=num_heads,
          use_mqa=use_mqa,
          num_kv_heads=num_kv_heads,
          src_block_size=src_block_size,
          tgt_block_size=tgt_block_size,
          enable_query_scale=False,
      )
      p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      l = p.Instantiate()

      src_proj = tf.reshape(input_vecs, [bs, num_heads, sl, d // num_heads])
      if num_kv_heads is not None and num_kv_heads < num_heads:
        tgt_proj = tf.split(src_proj, num_heads, axis=1)
        tgt_proj = tf.concat(tgt_proj[:num_kv_heads], axis=1)
      else:
        tgt_proj = src_proj
      input_mask = tf.cast(
          1.0 - tf.cast(input_padding, tf.float32), input_padding.dtype
      )
      block_mask = tf.reshape(
          input_mask, (-1, sl // tgt_block_size, tgt_block_size)
      )
      band_mask = l._DiagonalBandMaskFromInputs(block_mask, block_mask)
      input_mask = tf.reshape(input_mask, (bs, 1, 1, sl))

      encoded, _ = l.BlockSparseAttention(
          l.theta,
          query=src_proj,
          key=tgt_proj,
          value=tgt_proj,
          input_mask=input_mask,
          band_mask=band_mask,
      )
      tf.global_variables_initializer().run()
      encoded_out = sess.run(encoded)
      self.assertAllEqual(
          encoded_out.shape, (bs, num_heads, sl, d // num_heads)
      )
      # Use numpy to perform the same computation to generate expected results.
      input_vecs_p = np.reshape(
          input_vecs_p, (bs, num_heads, sl, d // num_heads)
      )
      input_mask_p = 1.0 - input_padding_p
      block_mask_p = np.reshape(
          input_mask_p, (-1, sl // tgt_block_size, tgt_block_size)
      )
      band_mask_p = np.einsum('blq,blk->blqk', block_mask_p, block_mask_p)
      band_mask_p = np.expand_dims(band_mask_p, 1)
      blocked_query = np.reshape(
          input_vecs_p,
          (
              bs,
              num_heads,
              sl // src_block_size,
              src_block_size,
              d // num_heads,
          ),
      )
      if num_kv_heads is not None:
        input_kv_vecs_p = blocked_query[:, :num_kv_heads, :, :, :]
        if num_kv_heads > 1:
          blocked_query = np.reshape(
              blocked_query,
              (
                  bs,
                  num_kv_heads,
                  num_heads // num_kv_heads,
                  sl // src_block_size,
                  src_block_size,
                  d // num_heads,
              ),
          )
          logits_eqn = 'BKnLTH,BKLSH->BnKLTS'
          ctx_eqn = 'BKnLTS,BKLSH->BnKLTH'
        else:
          input_kv_vecs_p = np.squeeze(input_kv_vecs_p, axis=1)
          logits_eqn = 'BNLTH,BLSH->BNLTS'
          ctx_eqn = 'BNLTS,BLSH->BNLTH'
      else:
        logits_eqn = None
        ctx_eqn = None
        input_kv_vecs_p = blocked_query

      expected_encoded = self._ComputeLocalContext(
          blocked_query,
          input_kv_vecs_p,
          input_kv_vecs_p,
          band_mask_p,
          logits_eqn=logits_eqn,
          ctx_eqn=ctx_eqn,
      )
      expected_encoded = np.reshape(
          expected_encoded, (bs, num_heads, sl, d // num_heads)
      )
      self.assertAllClose(expected_encoded, encoded_out, atol=1e-5)


if __name__ == '__main__':
  test_utils.main()
