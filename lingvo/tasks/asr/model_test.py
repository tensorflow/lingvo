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
"""Tests for Asr Model."""

from unittest import mock

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import gshard_builder
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import summary_utils
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.asr import decoder
from lingvo.tasks.asr import input_generator
from lingvo.tasks.asr import model
from lingvo.tasks.asr import model_test_input_generator as tig
import numpy as np


class DecoderForTest(decoder.AsrDecoder):
  """Unit test class for AsrDecoder with functional.for based unrolling."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.use_while_loop_based_unrolling = False
    return p


class AsrModelTest(test_utils.TestCase):

  def _testParams(self):
    input_shape = [2, 16, 8, 3]
    p = model.AsrModel.Params()
    p.decoder.target_seq_len = 5
    p.encoder.input_shape = input_shape
    p.input = tig.TestInputGenerator.Params()
    p.input.target_max_length = 5
    p.input.source_shape = input_shape
    p.input.target_shape = [2, 5]
    p.name = 'test_mdl'
    return p

  def testMakeDecoderTheta(self):
    # Test that decoder theta returns a copy of theta.decoder without changes.
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      decoder_theta = mdl._MakeDecoderTheta(theta=mdl.theta, input_batch=None)
      mdl.BProp()
      self.assertEqual(decoder_theta, mdl.theta.decoder)

  def testFProp(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 4.472597, mdl.loss.eval())

      actual_var_names = [_.name for _ in tf.trainable_variables()]
      print('all vars \n', '\n'.join(actual_var_names))
      expected_var_names = [
          'test_mdl/enc/conv_L0/w/var:0',
          'test_mdl/enc/conv_L0/beta/var:0',
          'test_mdl/enc/conv_L0/gamma/var:0',
          'test_mdl/enc/conv_L1/w/var:0',
          'test_mdl/enc/conv_L1/beta/var:0',
          'test_mdl/enc/conv_L1/gamma/var:0',
          'test_mdl/enc/f_conv_lstm_0/wm/var:0',
          'test_mdl/enc/f_conv_lstm_0/b/var:0',
          'test_mdl/enc/b_conv_lstm_0/wm/var:0',
          'test_mdl/enc/b_conv_lstm_0/b/var:0',
          'test_mdl/enc/conv_lstm_cnn_0/w/var:0',
          'test_mdl/enc/conv_lstm_cnn_0/beta/var:0',
          'test_mdl/enc/conv_lstm_cnn_0/gamma/var:0',
          'test_mdl/enc/fwd_rnn_L0/wm/var:0',
          'test_mdl/enc/fwd_rnn_L0/b/var:0',
          'test_mdl/enc/bak_rnn_L0/wm/var:0',
          'test_mdl/enc/bak_rnn_L0/b/var:0',
          'test_mdl/enc/proj_L0/w/var:0',
          'test_mdl/enc/proj_L0/beta/var:0',
          'test_mdl/enc/proj_L0/gamma/var:0',
          'test_mdl/enc/fwd_rnn_L1/wm/var:0',
          'test_mdl/enc/fwd_rnn_L1/b/var:0',
          'test_mdl/enc/bak_rnn_L1/wm/var:0',
          'test_mdl/enc/bak_rnn_L1/b/var:0',
          'test_mdl/enc/proj_L1/w/var:0',
          'test_mdl/enc/proj_L1/beta/var:0',
          'test_mdl/enc/proj_L1/gamma/var:0',
          'test_mdl/enc/fwd_rnn_L2/wm/var:0',
          'test_mdl/enc/fwd_rnn_L2/b/var:0',
          'test_mdl/enc/bak_rnn_L2/wm/var:0',
          'test_mdl/enc/bak_rnn_L2/b/var:0',
          'test_mdl/dec/emb/var_0/var:0',
          'test_mdl/dec/rnn_cell/wm/var:0',
          'test_mdl/dec/rnn_cell/b/var:0',
          'test_mdl/dec/atten/source_var/var:0',
          'test_mdl/dec/atten/query_var/var:0',
          'test_mdl/dec/atten/hidden_var/var:0',
          'test_mdl/dec/softmax/weight_0/var:0',
          'test_mdl/dec/softmax/bias_0/var:0',
      ]
      self.assertCountEqual(expected_var_names, actual_var_names)

  def testDecode(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()
      input_batch = mdl.input_generator.GetPreprocessedInputBatch()
      dec_out_dict = mdl.DecodeWithTheta(mdl.theta, input_batch)
      self.evaluate(tf.global_variables_initializer())
      dec_out = self.evaluate(dec_out_dict)
      print('dec_out', dec_out)
      metrics_dict = mdl.CreateDecoderMetrics()
      key_value_pairs = mdl.PostProcessDecodeOut(dec_out, metrics_dict)

      self.assertEqual(1.0, metrics_dict['error_rates/wer'].value)
      self.assertEqual(1.0, metrics_dict['norm_wer'].value)
      self.assertEqual(1.0, metrics_dict['ter'].value)
      self.assertEqual(0, len(key_value_pairs))

  def testFilterRealExamples(self):
    p = self._testParams()
    mdl = p.Instantiate()
    fake_dec_out = {
        'utt_id': ['utt1', 'utt2', 'utt3'],  # utt3 is dummy.
        'transcripts': ['a b c d', 'a', ''],
        'topk_decoded': [['a b c d', 'a b c d'], ['wrong', ''], ['', '']],
        'topk_scores': [[1.0, 0.9], [1.0, 0.9], [0.0, 0.0]],
        'topk_ids': [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7],
                     [0, 0, 0, 0], [0, 0, 0, 0]],
        'topk_lens': [2, 4, 4, 2, 0, 0],
        'target_labels': [[1, 2, 3, 4], [2, 3, 4, 5], [0, 0, 0, 0]],
        'target_paddings': [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
        'norm_wer_errors': [[0, 0], [1, 1], [0, 0]],
        'norm_wer_words': [[4, 4], [1, 1], [0, 0]],
        'is_real': [1.0, 1.0, 0.0]
    }
    expected_filtered = {
        'utt_id': ['utt1', 'utt2'],
        'transcripts': ['a b c d', 'a'],
        'topk_decoded': [['a b c d', 'a b c d'], ['wrong', '']],
        'topk_scores': [[1.0, 0.9], [1.0, 0.9]],
        'topk_ids': [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        'topk_lens': [2, 4, 4, 2],
        'target_labels': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'target_paddings': [[0, 0, 0, 1], [0, 0, 0, 1]],
        'norm_wer_errors': [[0, 0], [1, 1]],
        'norm_wer_words': [[4, 4], [1, 1]],
        'is_real': [1.0, 1.0]
    }
    mdl.decoder_metrics.FilterRealExamples(fake_dec_out)
    fake_dec_out = {
        key: np_arr.tolist() for key, np_arr in fake_dec_out.items()
    }
    self.assertDictEqual(fake_dec_out, expected_filtered)

  def testPostProcessDecodeOut(self):
    p = self._testParams()
    p.decoder.beam_search.num_hyps_per_beam = 2
    mdl = p.Instantiate()
    fake_dec_out = {
        'utt_id': ['utt1', 'utt2'],
        'transcripts': ['a b c d', 'a'],
        'topk_decoded': [['a b c d', 'a b c d'], ['wrong', '']],
        'topk_scores': [[1.0, 0.9], [1.0, 0.9]],
        'topk_ids': [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        'topk_lens': [2, 4, 4, 2],
        'target_labels': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'target_paddings': [[0, 0, 0, 1], [0, 0, 0, 1]],
        'norm_wer_errors': [[0, 0], [1, 1]],
        'norm_wer_words': [[4, 4], [1, 1]],
    }
    fake_dec_out = {k: np.array(v) for k, v in fake_dec_out.items()}
    metrics_dict = mdl.CreateDecoderMetrics()
    key_value_pairs = mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)

    self.assertEqual(0 + 1, metrics_dict['error_rates/wer'].total_value)
    self.assertEqual(4 + 1, metrics_dict['error_rates/wer'].total_weight)
    self.assertEqual(0 + 1, metrics_dict['norm_wer'].total_value)
    self.assertEqual(4 + 1, metrics_dict['norm_wer'].total_weight)
    self.assertEqual(4, metrics_dict['ter'].total_value)
    self.assertEqual(6, metrics_dict['ter'].total_weight)
    self.assertEqual(2, metrics_dict['num_samples_in_batch'].total_value)
    self.assertEqual(1.0, metrics_dict['num_samples_in_batch'].total_weight)
    self.assertEqual((4 / 5 * 3 / 3 * 2 / 2 * 1 / 1)**(1 / 4),
                     metrics_dict['corpus_bleu'].value)
    self.assertEqual((0 + 1) / 2, metrics_dict['sacc'].value)
    self.assertEqual((0 + 1) / (4 + 1), metrics_dict['oracle_norm_wer'].value)
    self.assertEqual(0, len(key_value_pairs))

  def testPostProcessLogUtf8(self):
    p = self._testParams()
    p.decoder_metrics.log_utf8 = True
    mdl = p.Instantiate()
    fake_dec_out = {
        'utt_id': ['utt1', 'utt2'],
        'transcripts': ['あいうえ'.encode('utf-8'), 'あ'.encode('utf-8')],
        'topk_decoded': [
            ['あいうえ'.encode('utf-8'), 'あいう'.encode('utf-8')],
            ['wrong'.encode('utf-8'), ''.encode('utf-8')],
        ],
        'topk_scores': [[1.0, 0.9], [1.0, 0.9]],
        'topk_ids': [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        'topk_lens': [2, 4, 4, 2],
        'target_labels': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'target_paddings': [[0, 0, 0, 1], [0, 0, 0, 1]],
        'norm_wer_errors': [[0, 0], [1, 1]],
        'norm_wer_words': [[4, 4], [1, 1]],
    }
    fake_dec_out = {k: np.array(v) for k, v in fake_dec_out.items()}
    metrics_dict = mdl.CreateDecoderMetrics()
    with cluster_factory.ForTestingWorker(add_summary=True):
      with mock.patch.object(tf.logging, 'info', autospec=True) as mock_info:
        mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)
        mock_info.assert_has_calls([
            mock.call('utt_id: %s', 'utt1'),
            mock.call('  ref_str: %s', 'あいうえ'),
            mock.call('  ref_ids: %s', [1, 2, 3]),
        ])
        mock_info.assert_has_calls([
            # Skips np.array values for ValueError from `inspect` module.
            # mock.call('  top_hyp_ids: %s', np.array([1, 2])),
            mock.call('  %f: %s', 1.0, 'あいうえ'),
            mock.call(
                '  ins: %d, subs: %d, del: %d, '
                'total: %d, ref_words: %d, wer: %f', 0, 0, 0, 0, 1, 0.0),
            mock.call(
                '  ci_ins: %d, ci_subs: %d, ci_del: %d, '
                'ci_total: %d, ref_words: %d, ci_wer: %f', 0, 0, 0, 0, 1, 0.0),
            mock.call('  %f: %s', 0.9, 'あいう'),
            mock.call('utt_id: %s', 'utt2'),
            mock.call('  ref_str: %s', 'あ'),
            mock.call('  ref_ids: %s', [2, 3, 4]),
        ])
        mock_info.assert_has_calls([
            # Skips np.array values for ValueError from `inspect` module.
            # mock.call('  top_hyp_ids: %s', np.array([3, 4, 5, 6])),
            mock.call('  %f: %s', 1.0, 'wrong'),
            mock.call(
                '  ins: %d, subs: %d, del: %d, '
                'total: %d, ref_words: %d, wer: %f', 0, 1, 0, 1, 1, 1.0),
            mock.call(
                '  ci_ins: %d, ci_subs: %d, ci_del: %d, '
                'ci_total: %d, ref_words: %d, ci_wer: %f', 0, 1, 0, 1, 1, 1.0),
            mock.call('  %f: %s', 0.9, ''),
        ])

  def testPostProcessDecodeOutFiltersEpsilonTokensForWER(self):
    p = self._testParams()
    p.decoder.beam_search.num_hyps_per_beam = 1
    mdl = p.Instantiate()
    fake_dec_out = {
        'utt_id': ['utt1', 'utt2'],
        'transcripts': ['a b c d', 'a b c'],
        'topk_decoded': [['a b<epsilon>c d'], ['<epsilon>a b<epsilon>']],
        'topk_scores': [[1.0], [1.0]],
        'topk_ids': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'topk_lens': [3, 4],
        'target_labels': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'target_paddings': [[0, 0, 0, 1], [0, 0, 1, 1]],
        'norm_wer_errors': [[0], [1]],
        'norm_wer_words': [[4], [3]],
    }
    fake_dec_out = {k: np.array(v) for k, v in fake_dec_out.items()}
    metrics_dict = mdl.CreateDecoderMetrics()
    kv_pairs = mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)

    self.assertEqual(0 + 1, metrics_dict['error_rates/wer'].total_value)
    self.assertEqual(7, metrics_dict['error_rates/wer'].total_weight)
    self.assertEqual(0 + 1, metrics_dict['norm_wer'].total_value)
    self.assertEqual(7, metrics_dict['norm_wer'].total_weight)
    self.assertEqual(0, len(kv_pairs))

  def testPostProcessDecodeOutFiltersNoiseTokensForWER(self):
    p = self._testParams()
    p.decoder.beam_search.num_hyps_per_beam = 1
    mdl = p.Instantiate()
    fake_dec_out = {
        'utt_id': ['utt1', 'utt2'],
        'transcripts': ['a b c d', 'a b c'],
        'topk_decoded': [['a b <noise> c d'], ['<noise> a b <noise>']],
        'topk_scores': [[1.0], [1.0]],
        'topk_ids': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'topk_lens': [3, 4],
        'target_labels': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'target_paddings': [[0, 0, 0, 1], [0, 0, 1, 1]],
        'norm_wer_errors': [[0], [1]],
        'norm_wer_words': [[4], [3]],
    }
    fake_dec_out = {k: np.array(v) for k, v in fake_dec_out.items()}
    metrics_dict = mdl.CreateDecoderMetrics()
    kv_pairs = mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)

    self.assertEqual(0 + 1, metrics_dict['error_rates/wer'].total_value)
    self.assertEqual(7, metrics_dict['error_rates/wer'].total_weight)
    self.assertEqual(0 + 1, metrics_dict['norm_wer'].total_value)
    self.assertEqual(7, metrics_dict['norm_wer'].total_weight)
    self.assertEqual(0, len(kv_pairs))

  def testPostProcessDecodeOutHandlesEmptyRef(self):
    p = self._testParams()
    p.decoder.beam_search.num_hyps_per_beam = 1
    mdl = p.Instantiate()
    fake_dec_out = {
        'utt_id': ['utt1', 'utt2'],
        'transcripts': ['', 'a b c d'],
        'topk_decoded': [['a'], ['a b c d']],
        'topk_scores': [[1.0], [1.0]],
        'topk_ids': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'topk_lens': [3, 4],
        'target_labels': [[1, 2, 3, 4], [2, 3, 4, 5]],
        'target_paddings': [[1, 1, 1, 1], [0, 0, 1, 1]],
        'norm_wer_errors': [[1], [0]],
        'norm_wer_words': [[0], [4]],
    }
    fake_dec_out = {k: np.array(v) for k, v in fake_dec_out.items()}
    metrics_dict = mdl.CreateDecoderMetrics()
    mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)

    self.assertEqual(1 + 0, metrics_dict['error_rates/wer'].total_value)
    self.assertEqual(0 + 4, metrics_dict['error_rates/wer'].total_weight)
    self.assertEqual(1 + 0, metrics_dict['norm_wer'].total_value)
    self.assertEqual(0 + 4, metrics_dict['norm_wer'].total_weight)

  def testBProp(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      mdl.BProp()
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 4.472597, mdl.loss.eval())
      mdl.train_op.run()

  def testBPropSmoothDecay(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams()
      p.train.lr_schedule = (
          schedule.ContinuousSchedule.Params().Set(
              start_step=350000, half_life_steps=45000))
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      mdl.BProp()
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 4.472597, mdl.loss.eval())
      mdl.train_op.run()

  def testAllLayerParams(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      lps = base_layer.RecursiveFindLayerParams(mdl.params)
      l_names = sorted([p.cls.__name__ for p in lps])
      expected_layers = sorted([
          'Adam',
          'AdditiveAttention',
          'AsciiTokenizer',
          'AsrDecoder',
          'AsrEncoder',
          'AsrModel',
          'BatchNormLayer',
          'BeamSearchHelper',
          'ConvLSTMCell',
          'Conv2DLayer',
          'Conv2DLayer',
          'DecoderMetrics',
          'EmbeddingLayer',
          'GreedySearchHelper',
          'HighwaySkipLayer',
          'LSTMCellSimple',
          'LSTMCellSimple',
          'LayerNorm',
          'Learner',
          'MultitaskAdapterLayer',
          'NullContextualizer',
          'NullFusion',
          'NullLm',
          'PiecewiseConstantSchedule',
          'ProjectionLayer',
          'SimpleFullSoftmax',
          'SpectrumAugmenter',
          'StackingOverTime',
          'TargetSequenceSampler',
          'TestInputGenerator',
      ])
      self.assertEqual(expected_layers, l_names)

  def testParamValueSumSquared(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      all_vars = tf.trainable_variables()
      py_utils.SumSquared(all_vars)

  def testCollectVarHistogram(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      var_grads = py_utils.ComputeGradients(mdl.loss, mdl.vars)
      summary_utils.CollectVarHistogram(var_grads)

  def testGradientMult(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      var_grads = py_utils.ComputeGradients(mdl.loss, mdl.vars)
      py_utils.ApplyGradMultiplier(var_grads, -1.1)

  def testLRDecay(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      tp = p.train
      tp.lr_schedule.boundaries = [300000, 400000, 500000]
      tp.lr_schedule.values = [1.0, 0.1, 0.01, 0.001]
      lrs = tp.lr_schedule.Instantiate()
      fetches = []
      for step in [299999, 300001, 399999, 400001, 499999, 500001]:
        with py_utils.GlobalStepContext(step):
          fetches.append(lrs.Value())
      values = self.evaluate(fetches)
      self.assertAllClose([1.0, 0.1, 0.1, 0.01, 0.01, 0.001], values)

  def testBatchSplit(self):

    def Run(num_splits):
      p = self._testParams()
      with self.session(use_gpu=False, graph=tf.Graph()):
        tf.random.set_seed(93820981)
        p.input.cur_iter_in_seed = False
        p.input.bucket_batch_limit = [
            b * 2 / num_splits for b in p.input.bucket_batch_limit
        ]
        with cluster_factory.ForTestingWorker(gpus=num_splits, do_eval=True):
          mdl = p.Instantiate()
          metrics = mdl.FPropDefaultTheta()[0]
        self.evaluate(tf.global_variables_initializer())
        return self.evaluate(metrics['loss'])

    res1, res2 = Run(1), Run(2)
    self.assertAllClose(res1[0], res2[0])
    self.assertAllEqual(res1[1], res2[1])

  def testFrontendAndEncoderFPropWithAuxLoss(self):

    class DummyAsrEncoder(base_layer.BaseLayer):
      """Speech encoder with aux loss."""

      @classmethod
      def Params(cls):
        p = super().Params()
        p.Define('proj', None, 'MoE network')
        p.Define('input_shape', [2, 16, 8, 1],
                 'Shape of the input. This should a TensorShape with rank 4.')
        return p

      def __init__(self, params):
        super().__init__(params)
        p = self.params
        self.CreateChild('proj', p.proj)

      @property
      def input_shape(self):
        return self.params.input_shape

      def zero_state(self, theta, batch_size):
        return py_utils.NestedMap()

      def FProp(self, theta, batch, state0=None):
        p = self.params
        inputs, paddings = batch.src.src_inputs, batch.src.paddings
        outputs = py_utils.NestedMap()
        with tf.name_scope(p.name):
          inputs = inputs[:, :, :, 0]
          encoded = self.proj.FProp(theta.proj, inputs, paddings)
          outputs['encoded'] = encoded
          outputs['padding'] = paddings
          return outputs

    with self.session():
      # Construct a network that adds aux loss.
      p = self._testParams()
      moe_builder = gshard_builder.MoEBuilder.Params().Set(
          num_devices=2,
          num_groups=2,
          e_dim=2,
          c_dim=2,
          model_dim=8,
          dropout_rate=0,
          moe_hidden_dim=8,
          moe_activation='SWISH')
      moe_ff = layers_with_attention.MoEFeedforwardLayer.Params().Set(
          moe_builder_p=moe_builder)
      p.encoder = DummyAsrEncoder.Params().Set(proj=moe_ff)
      p.input.source_shape = p.encoder.input_shape
      mdl = p.Instantiate()
      input_batch = mdl.input.GetPreprocessedInputBatch()
      encoder_outputs = mdl.FrontendAndEncoderFProp(mdl.theta, input_batch)
      self.assertIn('encoded', encoder_outputs)
      self.assertIn('aux_loss', encoder_outputs)

  def testComputePredictions(self):
    with self.session():
      p = self._testParams()
      mdl = p.Instantiate()
      input_batch = mdl.input.GetPreprocessedInputBatch()
      predictions = mdl.ComputePredictions(mdl.theta, input_batch)
      self.assertIn('encoder_outputs', predictions)
      self.assertIn('encoded', predictions.encoder_outputs)

  def testInference(self):

    def _CreateModelParamsForTest():
      p = model.AsrModel.Params()
      p.name = 'test_config'

      # Encoder params.
      ep = p.encoder
      ep.input_shape = [None, None, 80, 1]
      ep.lstm_cell_size = 16
      ep.num_lstm_layers = 2
      ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)]
      ep.conv_filter_strides = [(2, 2), (2, 2)]
      ep.num_conv_lstm_layers = 0
      # Initialize decoder params.
      dp = p.decoder
      dp.rnn_cell_dim = 16
      dp.rnn_layers = 2
      dp.source_dim = ep.lstm_cell_size * 2
      # Use functional while based unrolling.
      dp.use_while_loop_based_unrolling = False

      p.input = input_generator.AsrInput.Params()
      ip = p.input
      ip.frame_size = 80
      ip.append_eos_frame = True
      ip.pad_to_max_seq_length = False
      return p

    with self.session(
        use_gpu=False, graph=tf.Graph()) as sess, self.SetEval(True):
      p = _CreateModelParamsForTest()
      mdl = p.Instantiate()
      subgraphs = mdl.Inference()
      self.assertIn('default', subgraphs)

      fetches, feeds = subgraphs['default']
      self.assertIn('wav', feeds)
      for name in ['hypotheses', 'scores', 'src_frames', 'encoder_frames']:
        self.assertIn(name, fetches)

      with open(
          test_helper.test_src_dir_path('tools/testdata/gan_or_vae.16k.wav'),
          'rb') as f:
        wav = f.read()
      self.evaluate(tf.global_variables_initializer())
      fetches = sess.run(fetches, {feeds['wav']: wav})

      self.assertAllEqual((1, p.decoder.beam_search.num_hyps_per_beam),
                          fetches['hypotheses'].shape)
      self.assertAllEqual((1, p.decoder.beam_search.num_hyps_per_beam),
                          fetches['scores'].shape)
      self.assertAllEqual((1, 314, p.encoder.input_shape[2], 1),
                          fetches['src_frames'].shape)
      self.assertAllEqual((80, 1, 2 * p.encoder.lstm_cell_size),
                          fetches['encoder_frames'].shape)


if __name__ == '__main__':
  tf.test.main()
