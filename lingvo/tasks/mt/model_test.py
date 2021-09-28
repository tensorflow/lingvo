# Lint as: python3
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

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import layers
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.mt import decoder
from lingvo.tasks.mt import encoder
from lingvo.tasks.mt import input_generator
from lingvo.tasks.mt import model
import numpy as np


FLAGS = tf.flags.FLAGS

_TF_RANDOM_SEED = 93820986


class TestInputGenerator(base_input_generator.BaseSequenceInputGenerator):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('split', True, '')
    return p

  def __init__(self, params):
    super().__init__(params)
    self._step = 0

  def InfeedBatchSize(self):
    if self.params.split:
      return 10 / 2

    return 10

  def _InputBatch(self):
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
          tf.equal(tf.math.floormod(py_utils.GetGlobalStep(), 2), 0),
          lambda: src_ids[0], lambda: src_ids[1])
      ret.src.paddings = tf.cond(
          tf.equal(tf.math.floormod(py_utils.GetGlobalStep(), 2), 0),
          lambda: src_paddings[0], lambda: src_paddings[1])
      ret.tgt.ids = tf.cond(
          tf.equal(tf.math.floormod(py_utils.GetGlobalStep(), 2), 0),
          lambda: tgt_ids[0], lambda: tgt_ids[1])
      ret.tgt.labels = tf.cond(
          tf.equal(tf.math.floormod(py_utils.GetGlobalStep(), 2), 0),
          lambda: tgt_labels[0], lambda: tgt_labels[1])
      ret.tgt.paddings = tf.cond(
          tf.equal(tf.math.floormod(py_utils.GetGlobalStep(), 2), 0),
          lambda: tgt_paddings[0], lambda: tgt_paddings[1])
      ret.tgt.weights = tf.cond(
          tf.equal(tf.math.floormod(py_utils.GetGlobalStep(), 2), 0),
          lambda: tgt_weights[0], lambda: tgt_weights[1])
    else:
      ret.src.ids = src_ids
      ret.src.paddings = src_paddings
      ret.tgt.ids = tgt_ids
      ret.tgt.labels = tgt_labels
      ret.tgt.paddings = tgt_paddings
      ret.tgt.weights = tgt_weights

    return ret


class TransformerModelTest(test_utils.TestCase):

  def _InputParams(self):
    p = input_generator.NmtInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.tfrecord')
    vocab_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.vocab')
    p.file_pattern = 'tfrecord:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [40]
    p.bucket_batch_limit = [8]
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
      mdl = p.Instantiate()
      print('vars = ', mdl.vars)
      flatten_vars = mdl.vars.Flatten()
      print('vars flattened = ', flatten_vars)
      self.assertEqual(len(flatten_vars), 238)

      # Should match tf.trainable_variables().
      self.assertEqual(len(tf.trainable_variables()), len(flatten_vars))

  def testFProp(self, dtype=tf.float32, fprop_dtype=tf.float32):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.dtype = dtype
      if fprop_dtype:
        p.fprop_dtype = fprop_dtype
        p.input.dtype = fprop_dtype
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp))]

      print('actual vals = %s' % np.array_repr(np.array(vals)))
      self.assertAllClose(vals, [[226.99771, 10.377038], [243.92978, 10.379991],
                                 [260.7751, 10.379107], [201.10846, 10.379791],
                                 [272.22006, 10.370288]])

  def testFPropEvalMode(self):
    with self.session(), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp))]
      print('actual vals = ', vals)
      self.assertAllClose(vals, [(226.99771, 10.377038), (243.92978, 10.379991),
                                 (260.7751, 10.379107), (201.10846, 10.379791),
                                 (272.22006, 10.370288)])

  def testBProp(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]

      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp, mdl.train_op))[:2]]
      print('BProp actual vals = ', vals)
      expected_vals = [(226.99771, 10.377038), (243.87854, 10.3778105),
                       (260.66788, 10.374841), (200.94312, 10.371258),
                       (271.9328, 10.3593445)]
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
      tp.lr_schedule = schedule.ContinuousSchedule.Params()
      for l in p.ToText().split('\n'):
        print(l)
      return p

    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.input = TestInputGenerator.Params()
      p.input.split = True
      p = _SetDefaults(p)
      p.train.optimizer = optimizer.Accumulator.Params().Set(
          accum_steps=2, optimizer_tpl=p.train.optimizer)
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      mdl.BProp()

      self.evaluate(tf.global_variables_initializer())

      for _ in range(2):
        self.evaluate(mdl.train_op)

      expected = self.evaluate(mdl.dec.softmax.vars['weight_0'])

    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.input = TestInputGenerator.Params()
      p.input.split = False
      p = _SetDefaults(p)
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      mdl.BProp()

      self.evaluate(tf.global_variables_initializer())

      self.evaluate(mdl.train_op)

      actual = self.evaluate(mdl.dec.softmax.vars['weight_0'])

    self.assertAllClose(expected, actual, rtol=1e-2, atol=1e-2)

  def testBatchSplit(self):

    def Run(num_splits):
      with self.session(use_gpu=False, graph=tf.Graph()):
        tf.random.set_seed(93820981)
        p = self._testParams()
        p.input.bucket_batch_limit = [
            b * 2 / num_splits for b in p.input.bucket_batch_limit
        ]
        with cluster_factory.ForTestingWorker(gpus=num_splits):
          mdl = p.Instantiate()
          metrics = mdl.FPropDefaultTheta()[0]
        self.evaluate(tf.global_variables_initializer())
        return self.evaluate(metrics['loss'])

    res1, res2 = Run(1), Run(2)
    self.assertAllClose(res1[0], res2[0])
    self.assertAllEqual(res1[1], res2[1])

  def testBatchSizeInInputGenerator(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      with cluster_factory.ForTestingWorker(
          mode='sync', job='trainer_client', gpus=5):
        mdl = p.Instantiate()
        mdl.FPropDefaultTheta()
        loss = mdl.loss
        self.evaluate(tf.global_variables_initializer())
        _ = self.evaluate(loss)
        self.assertEqual(mdl.input_generator.infeed_bucket_batch_limit, [40])

  def testDecode(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()
      input_batch = mdl.input_generator.GetPreprocessedInputBatch()
      dec_out_dict = mdl.Decode(input_batch)
      self.evaluate(tf.global_variables_initializer())
      dec_out = self.evaluate(dec_out_dict)
      metrics_dict = mdl.CreateDecoderMetrics()
      key_value_pairs = mdl.PostProcessDecodeOut(dec_out, metrics_dict)
      self.assertNear(0.0, metrics_dict['corpus_bleu'].value, 1.0e-5)
      self.assertLen(key_value_pairs, 8)
      for k, v in key_value_pairs:
        self.assertIn(k, v)


class RNMTModelTest(test_utils.TestCase):

  def _InputParams(self):
    p = input_generator.NmtInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.tfrecord')
    vocab_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.vocab')
    p.file_pattern = 'tfrecord:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [40]
    p.bucket_batch_limit = [8]
    p.source_max_length = 200
    p.target_max_length = 200

    p.tokenizer.token_vocab_filepath = vocab_file
    p.tokenizer.vocab_size = 32000
    return p

  def _EncoderParams(self):
    p = encoder.MTEncoderBiRNN.Params()
    p.name = 'encoder'
    p.emb.vocab_size = 32000
    p.emb.embedding_dim = 4
    p.emb.max_num_shards = 1
    p.lstm_cell_size = 4
    p.num_lstm_layers = 3
    p.encoder_out_dim = 4
    return p

  def _DecoderParams(self):
    p = decoder.MTDecoderV1.Params()
    p.name = 'decoder'
    p.source_dim = 4
    p.emb.vocab_size = 32000
    p.emb.embedding_dim = 4
    p.emb.max_num_shards = 1
    p.rnn_cell_dim = 4
    p.rnn_layers = 3
    p.attention.hidden_dim = 2
    p.softmax.num_classes = 32000
    p.softmax.num_shards = 1
    return p

  def _testParams(self):
    p = model.RNMTModel.Params()
    p.name = 'test_mdl'
    p.input = self._InputParams()
    p.encoder = self._EncoderParams()
    p.decoder = self._DecoderParams()
    p.train.learning_rate = 1.0
    return p

  def testConstruction(self):
    with self.session():
      p = self._testParams()
      mdl = p.Instantiate()
      flatten_vars = mdl.vars.Flatten()
      # encoder/embedding: 1
      # encoder/lstms: 2 * (3 (forward) + 3 (backward))
      # encoder/proj: 2
      # decoder/embedding: 1
      # decoder/atten: 3
      # decoder/lstms: 2 * 3
      # decoder/softmax: 2
      self.assertEqual(len(flatten_vars), 1 + 12 + 2 + 1 + 3 + 6 + 2)

      # Should match tf.trainable_variables().
      self.assertEqual(len(tf.trainable_variables()), len(flatten_vars))

  def testFProp(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp))]
      print('actual vals = %s' % np.array_repr(np.array(vals)))
      self.assertAllClose(vals, [[226.92014, 10.373492], [243.77704, 10.373491],
                                 [260.63403, 10.373494], [200.98639, 10.373491],
                                 [272.30417, 10.373492]])

  def testFPropEvalMode(self):
    with self.session(), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp))]
      print('actual vals = %s' % np.array_repr(np.array(vals)))
      self.assertAllClose(vals, [[226.92014, 10.373492], [243.77704, 10.373491],
                                 [260.63403, 10.373494], [200.98639, 10.373491],
                                 [272.30417, 10.373492]])

  def testBProp(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]

      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp, mdl.train_op))[:2]]
      print('bprop actual vals = %s' % np.array_repr(np.array(vals)))
      expected_vals = [
          [226.92014, 10.373492],
          [225.25146, 9.585169],
          [248.49757, 9.8904505],
          [212.02884, 10.943424],
          [314.57098, 11.983657],
      ]
      self.assertAllClose(vals, expected_vals, atol=1e-3)

  def testDecode(self):
    with self.session(use_gpu=False), self.SetEval(True):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()
      input_batch = mdl.input_generator.GetPreprocessedInputBatch()
      dec_out_dict = mdl.Decode(input_batch)
      self.evaluate(tf.global_variables_initializer())
      dec_out = self.evaluate(dec_out_dict)
      metrics_dict = mdl.CreateDecoderMetrics()
      key_value_pairs = mdl.PostProcessDecodeOut(dec_out, metrics_dict)
      self.assertNear(0.0, metrics_dict['corpus_bleu'].value, 1.0e-5)
      self.assertLen(key_value_pairs, 8)
      for k, v in key_value_pairs:
        self.assertIn(k, v)

  def testBatchSplit(self):

    def Run(num_splits):
      with self.session(use_gpu=False, graph=tf.Graph()):
        tf.random.set_seed(93820981)
        p = self._testParams()
        p.input.bucket_batch_limit = [
            b * 2 / num_splits for b in p.input.bucket_batch_limit
        ]
        with cluster_factory.ForTestingWorker(gpus=num_splits):
          mdl = p.Instantiate()
          metrics = mdl.FPropDefaultTheta()[0]
        self.evaluate(tf.global_variables_initializer())
        return self.evaluate(metrics['loss'])

    res1, res2 = Run(1), Run(2)
    self.assertAllClose(res1[0], res2[0])
    self.assertAllEqual(res1[1], res2[1])

  def testBatchSizeInInputGenerator(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      cluster_params = cluster_factory.Cluster.Params()
      cluster_params.mode = 'sync'
      cluster_params.job = 'trainer_client'
      cluster_params.worker.name = '/job:localhost'
      cluster_params.worker.gpus_per_replica = 5
      cluster_params.input.name = '/job:localhost'
      cluster_params.input.replicas = 1
      cluster_params.input.gpus_per_replica = 0
      with cluster_params.Instantiate():
        mdl = p.Instantiate()
        mdl.FPropDefaultTheta()
        loss = mdl.loss
        self.evaluate(tf.global_variables_initializer())
        _ = self.evaluate(loss)
        self.assertEqual(mdl.input_generator.infeed_bucket_batch_limit, [40])


class HybridModelTest(test_utils.TestCase):

  def _InputParams(self):
    p = input_generator.NmtInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.tfrecord')
    vocab_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.vocab')
    p.file_pattern = 'tfrecord:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [40]
    p.bucket_batch_limit = [8]
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
    p = decoder.MTDecoderV1.Params()
    p.name = 'decoder'
    p.source_dim = 4
    p.emb.vocab_size = 32000
    p.emb.embedding_dim = 4
    p.emb.max_num_shards = 1
    p.rnn_cell_dim = 4
    p.rnn_layers = 3
    p.attention.hidden_dim = 2
    p.softmax.num_classes = 32000
    p.softmax.num_shards = 1
    return p

  def _testParams(self):
    p = model.HybridModel.Params()
    p.name = 'test_mdl'
    p.input = self._InputParams()
    p.encoder = self._EncoderParams()
    p.decoder = self._DecoderParams()
    p.train.learning_rate = 1.0
    return p

  def testConstruction(self):
    with self.session():
      p = self._testParams()
      mdl = p.Instantiate()
      flatten_vars = mdl.vars.Flatten()
      print('vars flattened = ', flatten_vars)
      # encoder: 91 (1 + 36 + 54)
      # encoder/embedding: 1
      # encoder/ff_layer: 6 * 6
      # encoder/attention: 9 * 6
      # decoder: 12 (1 + 3 + 6 + 2)
      # decoder/embedding: 1
      # decoder/atten: 3
      # decoder/lstms: 2 * 3
      # decoder/softmax: 2
      self.assertEqual(len(flatten_vars), 91 + 12)

      # Should match tf.trainable_variables().
      self.assertEqual(len(tf.trainable_variables()), len(flatten_vars))

  def testFProp(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp))]
      print('actual vals = %s' % np.array_repr(np.array(vals)))
      self.assertAllClose(vals, [[226.91527, 10.373269], [243.76906, 10.373152],
                                 [260.62787, 10.373248], [200.98814, 10.373582],
                                 [272.297, 10.373219]])

  def testFPropEvalMode(self):
    with self.session(), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]
      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp))]
      print('actual vals = %s' % np.array_repr(np.array(vals)))
      self.assertAllClose(vals, [[226.91527, 10.373269], [243.76906, 10.373152],
                                 [260.62787, 10.373248], [200.98814, 10.373582],
                                 [272.297, 10.373219]])

  def testBProp(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      mdl.BProp()
      loss = mdl.loss
      logp = mdl.eval_metrics['log_pplx'][0]

      self.evaluate(tf.global_variables_initializer())
      vals = []
      for _ in range(5):
        vals += [self.evaluate((loss, logp, mdl.train_op))[:2]]
      print('bprop actual vals = %s' % np.array_repr(np.array(vals)))
      expected_vals = [[226.91527, 10.373269], [222.4018, 9.463906],
                       [248.72293, 9.89942], [181.65323, 9.37565],
                       [312.97754, 11.922954]]
      self.assertAllClose(vals, expected_vals, atol=1e-3)

  def testDecode(self):
    with self.session(use_gpu=False), self.SetEval(True):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()
      input_batch = mdl.input_generator.GetPreprocessedInputBatch()
      dec_out_dict = mdl.Decode(input_batch)
      self.evaluate(tf.global_variables_initializer())
      dec_out = self.evaluate(dec_out_dict)
      metrics_dict = mdl.CreateDecoderMetrics()
      key_value_pairs = mdl.PostProcessDecodeOut(dec_out, metrics_dict)
      self.assertNear(0.0, metrics_dict['corpus_bleu'].value, 1.0e-5)
      self.assertLen(key_value_pairs, 8)
      for k, v in key_value_pairs:
        self.assertIn(k, v)

  def testBatchSplit(self):

    def Run(num_splits):
      with self.session(use_gpu=False, graph=tf.Graph()):
        tf.random.set_seed(93820981)
        p = self._testParams()
        p.input.bucket_batch_limit = [
            b * 2 / num_splits for b in p.input.bucket_batch_limit
        ]
        with cluster_factory.ForTestingWorker(gpus=num_splits):
          mdl = p.Instantiate()
          metrics = mdl.FPropDefaultTheta()[0]
        self.evaluate(tf.global_variables_initializer())
        return self.evaluate(metrics['loss'])

    res1, res2 = Run(1), Run(2)
    self.assertAllClose(res1[0], res2[0])
    self.assertAllEqual(res1[1], res2[1])

  def testBatchSizeInInputGenerator(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      cluster_params = cluster_factory.Cluster.Params()
      cluster_params.mode = 'sync'
      cluster_params.job = 'trainer_client'
      cluster_params.worker.name = '/job:localhost'
      cluster_params.worker.gpus_per_replica = 5
      cluster_params.input.name = '/job:localhost'
      cluster_params.input.replicas = 1
      cluster_params.input.gpus_per_replica = 0
      with cluster_params.Instantiate():
        mdl = p.Instantiate()
        mdl.FPropDefaultTheta()
        loss = mdl.loss
        self.evaluate(tf.global_variables_initializer())
        _ = self.evaluate(loss)
        self.assertEqual(mdl.input_generator.infeed_bucket_batch_limit, [40])


class InsertionModelTest(test_utils.TestCase):

  def _InputParams(self):
    p = input_generator.NmtInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.tfrecord')
    vocab_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.vocab')
    p.file_pattern = 'tfrecord:' + input_file
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [40]
    p.bucket_batch_limit = [8]
    p.source_max_length = 200
    p.target_max_length = 200

    p.tokenizer.token_vocab_filepath = vocab_file
    p.tokenizer.vocab_size = 32000
    return p

  def _DecoderParams(self):
    p = decoder.InsertionDecoder.Params()
    p.name = 'decoder'
    return p

  def _testParams(self):
    p = model.InsertionModel.Params()
    p.name = 'insertion'
    p.input = self._InputParams()
    p.decoder = self._DecoderParams()
    p.random_seed = 12345
    return p

  def testSampleCanvasAndTargets(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)

      x = np.asarray([[10, 11, 12, 13, 14, 15, 2], [10, 11, 12, 13, 14, 15, 2],
                      [2, 0, 0, 0, 0, 0, 0], [10, 11, 12, 13, 14, 2, 0]],
                     np.int32)
      x_paddings = np.asarray([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1]],
                              np.float32)

      p = self._testParams()
      mdl = p.Instantiate()

      descriptor = mdl._SampleCanvasAndTargets(
          tf.convert_to_tensor(x), tf.convert_to_tensor(x_paddings))

      canvas, canvas_paddings, target_indices, target_weights = self.evaluate([
          descriptor.canvas, descriptor.canvas_paddings,
          descriptor.target_indices, descriptor.target_weights
      ])

      canvas_gold = np.asarray([[13, 15, 2, 0, 0], [10, 11, 14, 2, 0],
                                [2, 0, 0, 0, 0], [10, 11, 13, 14, 2]], np.int32)
      canvas_paddings_gold = np.asarray(
          [[0., 0., 0., 1., 1.], [0., 0., 0., 0., 1.], [0., 1., 1., 1., 1.],
           [0., 0., 0., 0., 0.]], np.float32)
      target_indices_gold = np.asarray(
          [[0, 0, 10], [0, 0, 11], [0, 0, 12], [0, 0, 2], [0, 1, 14], [0, 1, 2],
           [0, 2, 2], [1, 0, 2], [1, 1, 2], [1, 2, 12], [1, 2, 13], [1, 2, 2],
           [1, 3, 15], [1, 3, 2], [2, 0, 2], [3, 0, 2], [3, 1, 2], [3, 2, 12],
           [3, 2, 2], [3, 3, 2], [3, 4, 2]], np.int32)
      target_weights_gold = np.asarray([1, 1, 1, 0, 1, 0, 1] +
                                       [1, 1, 1, 1, 0, 1, 0] + [1] +
                                       [1, 1, 1, 0, 1, 1], np.float32)
      target_weights_gold = np.reshape(target_weights_gold,
                                       [target_weights_gold.shape[0], 1])

      self.assertAllEqual(canvas, canvas_gold)
      self.assertAllEqual(canvas_paddings, canvas_paddings_gold)
      self.assertAllEqual(target_indices, target_indices_gold)
      self.assertAllEqual(target_weights, target_weights_gold)

  def testCreateCanvasAndTargets(self):
    with self.session():
      tf.random.set_seed(_TF_RANDOM_SEED)
      batch = py_utils.NestedMap(
          src=py_utils.NestedMap(
              ids=tf.convert_to_tensor(
                  np.asarray([
                      [10, 11, 12, 14, 2, 0],
                      [20, 21, 22, 24, 25, 2],
                  ], np.int32)),
              paddings=tf.convert_to_tensor(
                  np.asarray([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]],
                             np.float32))),
          tgt=py_utils.NestedMap(
              ids=tf.convert_to_tensor(
                  np.asarray([[100, 101, 102, 104, 2, 0],
                              [200, 201, 202, 204, 205, 2]], np.int32)),
              paddings=tf.convert_to_tensor(
                  np.asarray([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]],
                             np.float32))))

      p = self._testParams()
      mdl = p.Instantiate()

      descriptor = mdl._CreateCanvasAndTargets(batch)

      canvas, canvas_paddings, target_indices, target_weights = self.evaluate([
          descriptor.canvas, descriptor.canvas_paddings,
          descriptor.target_indices, descriptor.target_weights
      ])

      canvas_gold = np.asarray([
          [32014, 32002, 104, 2, 0, 0, 0, 0],
          [32020, 32021, 32022, 32002, 200, 201, 202, 2],
      ], np.int32)
      canvas_paddings_gold = np.asarray(
          [[0., 0., 0., 0., 1., 1., 1., 1.], [0., 0., 0., 0., 0., 0., 0., 0.]],
          np.float32)
      target_indices_gold = np.asarray(
          [[0, 0, 10], [0, 0, 11], [0, 0, 12], [0, 0, 2], [0, 1, 2], [1, 0, 2],
           [1, 1, 2], [1, 2, 2], [1, 3, 24], [1, 3, 25], [1, 3, 2], [0, 2, 100],
           [0, 2, 101], [0, 2, 102], [0, 2, 2], [0, 3, 2], [1, 4, 2], [1, 5, 2],
           [1, 6, 2], [1, 7, 204], [1, 7, 205], [1, 7, 2]], np.int32)
      target_weights_gold = np.asarray([1, 1, 1, 0, 1] + [1, 1, 1, 1, 1, 0] +
                                       [1, 1, 1, 0, 1] + [1, 1, 1, 1, 1, 0],
                                       np.float32)
      target_weights_gold = np.reshape(target_weights_gold,
                                       [target_weights_gold.shape[0], 1])

      self.assertAllEqual(canvas, canvas_gold)
      self.assertAllEqual(canvas_paddings, canvas_paddings_gold)
      self.assertAllEqual(target_indices, target_indices_gold)
      self.assertAllEqual(target_weights, target_weights_gold)

  def testConstruction(self):
    with self.session():
      p = self._testParams()
      mdl = p.Instantiate()
      flatten_vars = mdl.vars.Flatten()
      self.assertEqual(len(flatten_vars), 122)
      self.assertEqual(len(tf.trainable_variables()), len(flatten_vars))

  def testFPropGraph(self):
    """Test the construction of the fprop graph, then fprop the graph."""
    with self.session():
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()

      self.evaluate(tf.global_variables_initializer())
      self.evaluate(mdl.loss)


class TransformerXEnDecTest(test_utils.TestCase):

  def _InputParams(self):
    p = input_generator.NmtDoubleInput.Params()
    input_file = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_doublebatch_test-000-001')
    p.file_pattern = 'tfrecord:' + input_file
    p.tokenizer.token_vocab_filepath = test_helper.test_src_dir_path(
        'tasks/mt/testdata/wmt14_ende_wpm_32k_test.vocab')
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [10, 20]
    p.bucket_batch_limit = [4, 2]
    p.source_mask_ratio = -1
    p.source_mask_ratio_beta = '2,6'
    p.mask_word_id = 31999
    p.pad_id = 31998
    p.mask_words_ratio = 0.25
    p.permutation_distance = 3
    p.vocab_file = p.tokenizer.token_vocab_filepath
    p.packed_input = False
    return p

  def _EncoderParams(self):
    p = encoder.TransformerXEncoder.Params()
    p.name = 'mix_encoder'
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim()
    p.token_emb.vocab_size = 32000
    p.token_emb.embedding_dim = 4
    p.token_emb.max_num_shards = 1
    p.token_emb.scale_sqrt_depth = True
    p.token_emb.vn = py_utils.VariationalNoiseParams(1.0, False, False)
    p.position_emb.embedding_dim = 4
    p.position_emb.trainable_scaling = False
    p.model_dim = 4
    ts = p.transformer_stack
    ts.model_dim = 4
    ts.num_transformer_layers = 6
    ts.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    ts.transformer_tpl.tr_fflayer_tpl.hidden_dim = 4
    p.random_seed = 54321
    return p

  def _DecoderParams(self):
    p = decoder.TransformerXDecoder.Params()
    p.name = 'mix_decoder'
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim()
    p.token_emb.vocab_size = 32000
    p.token_emb.embedding_dim = 4
    p.token_emb.max_num_shards = 1
    p.token_emb.scale_sqrt_depth = True
    p.token_emb.vn = py_utils.VariationalNoiseParams(1.0, False, False)
    p.position_emb.embedding_dim = 4
    p.position_emb.trainable_scaling = False
    p.model_dim = 4
    p.source_dim = 4
    p.num_trans_layers = 6
    p.trans_tpl.source_dim = p.model_dim
    p.trans_tpl.tr_atten_tpl.source_dim = p.model_dim
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
    p.trans_tpl.tr_atten_tpl.atten_hidden_dim = 4
    p.trans_tpl.tr_atten_tpl.atten_tpl.context_dim = p.model_dim
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 4
    p.trans_tpl.tr_fflayer_tpl.input_dim = p.model_dim
    p.label_smoothing = layers.UniformLabelSmoother.Params()
    p.label_smoothing.uncertainty = 0.1
    p.per_word_avg_loss = True
    p.softmax.num_classes = 32000
    p.softmax.num_shards = 1
    p.random_seed = 54321
    return p

  def _testParams(self):
    p = model.TransformerXEnDecModel.Params()
    p.name = 'xendec'
    p.input = self._InputParams()
    p.encoder = self._EncoderParams()
    p.decoder = self._DecoderParams()
    p.random_seed = 12345
    return p

  def testFProp(self, dtype=tf.float32, fprop_dtype=tf.float32):
    with self.session(use_gpu=False):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._testParams()
      p.dtype = dtype
      if fprop_dtype:
        p.fprop_dtype = fprop_dtype
        p.input.dtype = fprop_dtype
      mdl = p.Instantiate()
      dec_metrics, _ = mdl.FPropDefaultTheta()
      self.evaluate(tf.global_variables_initializer())
      vals = []
      print(mdl)
      for _ in range(5):
        vals += [
            self.evaluate(
                (dec_metrics['clean_loss'][0], dec_metrics['other_loss'][0],
                 dec_metrics['mix_loss_0'][0], dec_metrics['loss'][0]))
        ]

      print('actual vals = %s' % np.array_repr(np.array(vals)))
      self.assertAllClose(
          vals, [[10.373864, 10.371083, 10.372491, 31.11744],
                 [10.36428, 10.379262, 10.366394, 31.109936],
                 [10.369206, 10.372709, 10.369126, 31.111042],
                 [10.363656, 10.364362, 10.362683, 31.090702],
                 [10.371622, 10.374066, 10.371591, 31.11728]],
          rtol=1e-02,
          atol=1e-02)


if __name__ == '__main__':
  tf.test.main()
