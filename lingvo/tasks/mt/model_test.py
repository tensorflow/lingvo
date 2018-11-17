# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for MT Models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from six.moves import range

import tensorflow as tf
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import lr_schedule
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import test_helper
from lingvo.tasks.mt import decoder
from lingvo.tasks.mt import encoder
from lingvo.tasks.mt import input_generator
from lingvo.tasks.mt import model

FLAGS = tf.flags.FLAGS

_TF_RANDOM_SEED = 93820986


class TestInputGenerator(base_input_generator.BaseSequenceInputGenerator):

  @classmethod
  def Params(cls):
    p = super(TestInputGenerator, cls).Params()
    p.Define('split', True, '')
    return p

  def __init__(self, params):
    super(TestInputGenerator, self).__init__(params)
    self._step = 0

  def InputBatchSize(self):
    if self.params.split:
      return 10 / 2

    return 10

  def InputBatch(self):
    np.random.seed(1)
    bs, sl = 10, 7
    src_ids = tf.constant(
        np.random.randint(low=0, high=8192 - 1, size=[bs, sl], dtype=np.int32))
    tgt_ids = tf.constant(
        np.random.randint(low=0, high=8192 - 1, size=[bs, sl], dtype=np.int32))
    tgt_labels = tf.constant(
        np.random.randint(low=0, high=8192 - 1, size=[bs, sl], dtype=np.int32))
    tgt_weights = tf.constant(np.ones(shape=[bs, sl], dtype=np.float32))

    src_paddings = tf.zeros([bs, sl])
    tgt_paddings = tf.zeros([bs, sl])

    ret = py_utils.NestedMap()
    ret.src = py_utils.NestedMap()
    ret.tgt = py_utils.NestedMap()

    if self.params.split:
      src_ids = tf.split(src_ids, 2, 0)
      src_paddings = tf.split(src_paddings, 2, 0)
      tgt_ids = tf.split(tgt_ids, 2, 0)
      tgt_labels = tf.split(tgt_labels, 2, 0)
      tgt_paddings = tf.split(tgt_paddings, 2, 0)
      tgt_weights = tf.split(tgt_weights, 2, 0)

      ret.src.ids = tf.cond(
          tf.equal(tf.mod(py_utils.GetOrCreateGlobalStep(), 2), 0),
          lambda: src_ids[0], lambda: src_ids[1])
      ret.src.paddings = tf.cond(
          tf.equal(tf.mod(py_utils.GetOrCreateGlobalStep(), 2), 0),
          lambda: src_paddings[0], lambda: src_paddings[1])
      ret.tgt.ids = tf.cond(
          tf.equal(tf.mod(py_utils.GetOrCreateGlobalStep(), 2), 0),
          lambda: tgt_ids[0], lambda: tgt_ids[1])
      ret.tgt.labels = tf.cond(
          tf.equal(tf.mod(py_utils.GetOrCreateGlobalStep(), 2), 0),
          lambda: tgt_labels[0], lambda: tgt_labels[1])
      ret.tgt.paddings = tf.cond(
          tf.equal(tf.mod(py_utils.GetOrCreateGlobalStep(), 2), 0),
          lambda: tgt_paddings[0], lambda: tgt_paddings[1])
      ret.tgt.weights = tf.cond(
          tf.equal(tf.mod(py_utils.GetOrCreateGlobalStep(), 2), 0),
          lambda: tgt_weights[0], lambda: tgt_weights[1])
    else:
      ret.src.ids = src_ids
      ret.src.paddings = src_paddings
      ret.tgt.ids = tgt_ids
      ret.tgt.labels = tgt_labels
      ret.tgt.paddings = tgt_paddings
      ret.tgt.weights = tgt_weights

    return ret


class TransformerModelTest(tf.test.TestCase):

  def _InputParams(self):
    p = input_generator.NmtInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.tfrecord')
    vocab_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.vocab')
    p.file_pattern = 'tfrecord:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [20, 40]
    p.bucket_batch_limit = [4, 8]
    p.source_max_length = 200
    p.target_max_length = 200

    p.tokenizer.token_vocab_filepath = vocab_file
    p.tokenizer.vocab_size = 32000
    return p

  def _EncoderParams(self):
    p = encoder.TransformerEncoder.Params()
    p.name = 'encoder'
    p.random_seed = 1234
    p.model_dim = 4
    p.token_emb.embedding_dim = 4
    p.token_emb.max_num_shards = 1
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim(
        seed=p.random_seed)
    p.position_emb.embedding_dim = 4
    p.transformer_stack.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    p.transformer_stack.transformer_tpl.tr_fflayer_tpl.hidden_dim = 5
    return p

  def _DecoderParams(self):
    p = decoder.TransformerDecoder.Params()
    p.name = 'decoder'
    p.random_seed = 1234
    p.source_dim = 4
    p.model_dim = 4
    p.token_emb.embedding_dim = 4
    p.token_emb.max_num_shards = 1
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim(
        seed=p.random_seed)
    p.position_emb.embedding_dim = 4
    p.trans_tpl.source_dim = 4
    p.trans_tpl.tr_atten_tpl.source_dim = 4
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
    p.trans_tpl.tr_fflayer_tpl.input_dim = 4
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 8
    p.softmax.num_shards = 1
    p.target_seq_len = 5
    return p

  def _testParams(self):
    p = model.TransformerModel.Params()
    p.name = 'test_mdl'
    p.input = self._InputParams()
    p.encoder = self._EncoderParams()
    p.decoder = self._DecoderParams()
    p.train.learning_rate = 2e-4
    return p

  def testConstruction(self):
    with self.session():
      p = self._testParams()
      mdl = p.cls(p)
      print('vars = ', mdl.vars)
      flatten_vars = mdl.vars.Flatten()
      print('vars flattened = ', flatten_vars)
      self.assertEqual(len(flatten_vars), 238)

      # Should match tf.trainable_variables().
      self.assertEqual(len(tf.trainable_variables()), len(flatten_vars))

  def testFProp(self, dtype=tf.float32, fprop_dtype=tf.float32):
    with self.session() as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.dtype = dtype
      if fprop_dtype:
        p.fprop_dtype = fprop_dtype
        p.input.dtype = fprop_dtype
      mdl = p.cls(p)
      input_batch = mdl.GetInputBatch()
      mdl.FProp(mdl.theta, input_batch)
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      tf.global_variables_initializer().run()
      vals = []
      for _ in range(5):
        vals += [sess.run((loss, logp))]

      print('actual vals = %s' % np.array_repr(np.array(vals)))
      self.assertAllClose(
          vals, [(189.22296, 10.368382), (282.57202, 10.369616),
                 (142.55638, 10.367737), (139.9939, 10.369918),
                 (293.08011, 10.374517)],
          atol=1e-6, rtol=1e-6)

  def testFPropEvalMode(self):
    with self.session() as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.is_eval = True
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      tf.global_variables_initializer().run()
      vals = []
      for _ in range(5):
        vals += [sess.run((loss, logp))]
      print('actual vals = ', vals)
      self.assertAllClose(
          vals, [(189.22296, 10.368382), (282.57202, 10.369616),
                 (142.55638, 10.367737), (139.9939, 10.369918),
                 (293.08011, 10.374517)])

  def testBProp(self):
    with self.session() as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]

      tf.global_variables_initializer().run()
      vals = []
      for _ in range(5):
        vals += [sess.run((loss, logp, mdl.train_op))[:2]]
      print('BProp actual vals = ', vals)
      expected_vals = [(189.22296, 10.368382), (282.54092, 10.368474),
                       (142.48544, 10.362577), (139.91856, 10.364338),
                       (292.86707, 10.366976)]
      self.assertAllClose(vals, expected_vals)

  def testBPropWithAccumComparison(self):

    def _SetDefaults(p):
      p.random_seed = 12345
      p.decoder.input_dropout_prob = 0.0
      mp = p.encoder.transformer_stack.transparent_merger_tpl
      mp.weighted_merger_dropout_prob = 0.0
      disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
      for lp in base_layer.RecursiveFindLayerParams(p):
        # TODO(lepikhin): lp.dtype = dtype
        lp.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
        lp.vn = disable_vn

      tp = p.train
      assert tp.l2_regularizer_weight is None
      tp.clip_gradient_norm_to_value = False
      tp.grad_norm_to_clip_to_zero = False
      tp.optimizer = optimizer.SGD.Params()
      tp.learning_rate = 1e-2
      tp.lr_schedule = lr_schedule.ContinuousLearningRateSchedule.Params()
      for l in p.ToText().split('\n'):
        print(l)
      return p

    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.input = TestInputGenerator.Params()
      p.input.split = True
      p = _SetDefaults(p)
      p.train.optimizer = optimizer.Accumulator.Params().Set(
          accum_steps=2, optimizer_tpl=p.train.optimizer)
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]

      tf.global_variables_initializer().run()

      for _ in range(2):
        sess.run((py_utils.GetOrCreateGlobalStep(), loss, logp, mdl.train_op))

      expected = sess.run(mdl.dec.softmax.vars['weight_0'])

    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.input = TestInputGenerator.Params()
      p.input.split = False
      p = _SetDefaults(p)
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]

      tf.global_variables_initializer().run()

      sess.run((py_utils.GetOrCreateGlobalStep(), loss, logp, mdl.train_op))

      actual = sess.run(mdl.dec.softmax.vars['weight_0'])

    self.assertAllClose(expected, actual, rtol=1e-2, atol=1e-2)

  def testBatchSplit(self):

    def Run(num_splits):
      with self.session(use_gpu=False, graph=tf.Graph()) as sess:
        tf.set_random_seed(93820981)
        p = self._testParams()
        p.input.bucket_batch_limit = [
            b * 2 / num_splits for b in p.input.bucket_batch_limit
        ]
        with cluster_factory.ForTestingWorker(gpus=num_splits):
          mdl = p.cls(p)
          metrics = mdl.FPropDefaultTheta()
        tf.global_variables_initializer().run()
        return sess.run(metrics['loss'])

    res1, res2 = Run(1), Run(2)
    self.assertAllClose(res1[0], res2[0])
    self.assertAllEqual(res1[1], res2[1])

  def testBatchSizeInInputGenerator(self):
    with self.session() as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      with cluster_factory.ForTestingWorker(
          mode='sync', job='trainer_client', gpus=5):
        mdl = p.cls(p)
        mdl.FPropDefaultTheta()
      loss = mdl.loss
      tf.global_variables_initializer().run()
      _ = sess.run(loss)
      self.assertEqual(mdl.input_generator.scaled_bucket_batch_limit, [20, 40])

  def testDecode(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(93820985)
      p = self._testParams()
      mdl = p.cls(p)
      input_batch = mdl.input_generator.GetPreprocessedInputBatch()
      dec_out_dict = mdl.Decode(input_batch)
      tf.global_variables_initializer().run()
      dec_out = sess.run(dec_out_dict)
      metrics_dict = mdl.CreateDecoderMetrics()
      key_value_pairs = mdl.PostProcessDecodeOut(dec_out, metrics_dict)
      self.assertNear(0.0, metrics_dict['corpus_bleu'].value, 1.0e-5)
      self.assertEqual(4, len(key_value_pairs))
      for k, v in key_value_pairs:
        self.assertIn(k, v)


if __name__ == '__main__':
  tf.test.main()
