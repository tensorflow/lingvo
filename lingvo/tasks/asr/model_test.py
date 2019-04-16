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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import numpy as np
import six
from six.moves import range

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import lr_schedule
from lingvo.core import py_utils
from lingvo.core import summary_utils
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.asr import decoder
from lingvo.tasks.asr import input_generator
from lingvo.tasks.asr import model
from lingvo.tasks.asr import model_test_input_generator as tig


class DecoderForTest(decoder.AsrDecoder):
  """Unit test class for AsrDecoder with functional.for based unrolling."""

  @classmethod
  def Params(cls):
    p = super(DecoderForTest, cls).Params()
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
      tf.set_random_seed(93820985)
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      decoder_theta = mdl._MakeDecoderTheta(mdl.theta)
      mdl.BProp()
      self.assertEqual(decoder_theta, mdl.theta.decoder)

  def testFPropBPropTargetKey(self, inline=True):
    # Compute loss for a model without target_key.
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(93820985)
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      tf.global_variables_initializer().run()
      mdl_loss = mdl.loss.eval()
      sess.run(mdl._train_op)
      mdl_global_step = mdl.global_step.eval()
      mdl_train_loss = mdl.loss.eval()

    # Compute loss for a model with target_key.
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(93820985)
      p_t = self._testParams()
      p_t.input.target_key = 'target_key'
      p_t.input.target_key_target_shape = [2, 5]
      p_t.target_key = 'target_key'
      mdl_t = p.cls(p_t)
      mdl_t.FPropDefaultTheta()
      mdl_t.BProp()
      tf.global_variables_initializer().run()
      mdl_t_loss = mdl_t.loss.eval()
      sess.run(mdl_t._train_op)
      mdl_t_global_step = mdl_t.global_step.eval()
      mdl_t_train_loss = mdl_t.loss.eval()

    # Verify that losses are identical in either case.
    self.assertAllClose(mdl_loss, mdl_t_loss)
    self.assertAllClose(mdl_global_step, mdl_t_global_step)
    self.assertAllClose(mdl_train_loss, mdl_t_train_loss)

  def testFProp(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(93820985)
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      tf.global_variables_initializer().run()
      test_utils.CompareToGoldenSingleFloat(self, 4.472597, mdl.loss.eval())

      actual_var_names = [_.name for _ in tf.all_variables()]
      print('all vars \n', '\n'.join(actual_var_names))
      expected_var_names = [
          'global_step:0', 'test_mdl/enc/conv_L0/w/var:0',
          'test_mdl/enc/conv_L0/beta/var:0', 'test_mdl/enc/conv_L0/gamma/var:0',
          'test_mdl/enc/conv_L0/moving_mean/var:0',
          'test_mdl/enc/conv_L0/moving_variance/var:0',
          'test_mdl/enc/conv_L1/w/var:0', 'test_mdl/enc/conv_L1/beta/var:0',
          'test_mdl/enc/conv_L1/gamma/var:0',
          'test_mdl/enc/conv_L1/moving_mean/var:0',
          'test_mdl/enc/conv_L1/moving_variance/var:0',
          'test_mdl/enc/f_conv_lstm_0/wm/var:0',
          'test_mdl/enc/f_conv_lstm_0/b/var:0',
          'test_mdl/enc/b_conv_lstm_0/wm/var:0',
          'test_mdl/enc/b_conv_lstm_0/b/var:0',
          'test_mdl/enc/conv_lstm_cnn_0/w/var:0',
          'test_mdl/enc/conv_lstm_cnn_0/beta/var:0',
          'test_mdl/enc/conv_lstm_cnn_0/gamma/var:0',
          'test_mdl/enc/conv_lstm_cnn_0/moving_mean/var:0',
          'test_mdl/enc/conv_lstm_cnn_0/moving_variance/var:0',
          'test_mdl/enc/fwd_rnn_L0/wm/var:0', 'test_mdl/enc/fwd_rnn_L0/b/var:0',
          'test_mdl/enc/bak_rnn_L0/wm/var:0', 'test_mdl/enc/bak_rnn_L0/b/var:0',
          'test_mdl/enc/proj_L0/w/var:0', 'test_mdl/enc/proj_L0/beta/var:0',
          'test_mdl/enc/proj_L0/gamma/var:0',
          'test_mdl/enc/proj_L0/moving_mean/var:0',
          'test_mdl/enc/proj_L0/moving_variance/var:0',
          'test_mdl/enc/fwd_rnn_L1/wm/var:0', 'test_mdl/enc/fwd_rnn_L1/b/var:0',
          'test_mdl/enc/bak_rnn_L1/wm/var:0', 'test_mdl/enc/bak_rnn_L1/b/var:0',
          'test_mdl/enc/proj_L1/w/var:0', 'test_mdl/enc/proj_L1/beta/var:0',
          'test_mdl/enc/proj_L1/gamma/var:0',
          'test_mdl/enc/proj_L1/moving_mean/var:0',
          'test_mdl/enc/proj_L1/moving_variance/var:0',
          'test_mdl/enc/fwd_rnn_L2/wm/var:0', 'test_mdl/enc/fwd_rnn_L2/b/var:0',
          'test_mdl/enc/bak_rnn_L2/wm/var:0', 'test_mdl/enc/bak_rnn_L2/b/var:0',
          'test_mdl/dec/emb/var_0/var:0', 'test_mdl/dec/rnn_cell/wm/var:0',
          'test_mdl/dec/rnn_cell/b/var:0',
          'test_mdl/dec/atten/source_var/var:0',
          'test_mdl/dec/atten/query_var/var:0',
          'test_mdl/dec/atten/hidden_var/var:0',
          'test_mdl/dec/softmax/weight_0/var:0',
          'test_mdl/dec/softmax/bias_0/var:0'
      ]
      self.assertEqual(sorted(expected_var_names), sorted(actual_var_names))

  def testDecode(self):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(93820985)
      p = self._testParams()
      mdl = p.cls(p)
      input_batch = mdl.input_generator.GetPreprocessedInputBatch()
      dec_out_dict = mdl.Decode(input_batch)
      tf.global_variables_initializer().run()
      dec_out = sess.run(dec_out_dict)
      print('dec_out', dec_out)
      metrics_dict = mdl.CreateDecoderMetrics()
      key_value_pairs = mdl.PostProcessDecodeOut(dec_out, metrics_dict)

      self.assertEqual(1.0, metrics_dict['wer'].value)
      self.assertEqual(1.0, metrics_dict['norm_wer'].value)
      self.assertEqual(1.0, metrics_dict['ter'].value)
      self.assertEqual(0, len(key_value_pairs))

  def testPostProcessDecodeOut(self):
    p = self._testParams()
    p.decoder.beam_search.num_hyps_per_beam = 2
    mdl = p.cls(p)
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
    metrics_dict = mdl.CreateDecoderMetrics()
    key_value_pairs = mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)

    self.assertEqual(0 + 1, metrics_dict['wer'].total_value)
    self.assertEqual(4 + 1, metrics_dict['wer'].total_weight)
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

  def testPostProcessDecodeOutFiltersEpsilonTokensForWER(self):
    p = self._testParams()
    p.decoder.beam_search.num_hyps_per_beam = 1
    mdl = p.cls(p)
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
    metrics_dict = mdl.CreateDecoderMetrics()
    kv_pairs = mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)

    self.assertEqual(0 + 1, metrics_dict['wer'].total_value)
    self.assertEqual(7, metrics_dict['wer'].total_weight)
    self.assertEqual(0 + 1, metrics_dict['norm_wer'].total_value)
    self.assertEqual(7, metrics_dict['norm_wer'].total_weight)
    self.assertEqual(0, len(kv_pairs))

  def testPostProcessDecodeOutFiltersNoiseTokensForWER(self):
    p = self._testParams()
    p.decoder.beam_search.num_hyps_per_beam = 1
    mdl = p.cls(p)
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
    metrics_dict = mdl.CreateDecoderMetrics()
    kv_pairs = mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)

    self.assertEqual(0 + 1, metrics_dict['wer'].total_value)
    self.assertEqual(7, metrics_dict['wer'].total_weight)
    self.assertEqual(0 + 1, metrics_dict['norm_wer'].total_value)
    self.assertEqual(7, metrics_dict['norm_wer'].total_weight)
    self.assertEqual(0, len(kv_pairs))

  def testPostProcessDecodeOutHandlesEmptyRef(self):
    p = self._testParams()
    p.decoder.beam_search.num_hyps_per_beam = 1
    mdl = p.cls(p)
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
    metrics_dict = mdl.CreateDecoderMetrics()
    mdl.PostProcessDecodeOut(fake_dec_out, metrics_dict)

    self.assertEqual(1 + 0, metrics_dict['wer'].total_value)
    self.assertEqual(0 + 4, metrics_dict['wer'].total_weight)
    self.assertEqual(1 + 0, metrics_dict['norm_wer'].total_value)
    self.assertEqual(0 + 4, metrics_dict['norm_wer'].total_weight)

  def testBProp(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(93820985)
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      tf.global_variables_initializer().run()
      test_utils.CompareToGoldenSingleFloat(self, 4.472597, mdl.loss.eval())
      mdl.train_op.run()

  def testBPropSmoothDecay(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(93820985)
      p = self._testParams()
      p.train.lr_schedule = (
          lr_schedule.ContinuousLearningRateSchedule.Params().Set(
              start_step=350000, half_life_steps=45000))
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      mdl.BProp()
      tf.global_variables_initializer().run()
      test_utils.CompareToGoldenSingleFloat(self, 4.472597, mdl.loss.eval())
      mdl.train_op.run()

  def testAllLayerParams(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      mdl = p.cls(p)
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
          'BeamSearchHelper',
          'TargetSequenceSampler',
          'ConvLSTMCell',
          'Conv2DLayer',
          'Conv2DLayer',
          'EmbeddingLayer',
          'HighwaySkipLayer',
          'LSTMCellSimple',
          'LSTMCellSimple',
          'NullContextualizer',
          'NullFusion',
          'NullLm',
          'PiecewiseConstantLearningRateSchedule',
          'ProjectionLayer',
          'SimpleFullSoftmax',
          'TestInputGenerator',
      ])
      self.assertEqual(expected_layers, l_names)

  def testParamValueSumSquared(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      all_vars = tf.trainable_variables()
      py_utils.SumSquared(all_vars)

  def testCollectVarHistogram(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      var_grads = py_utils.ComputeGradients(mdl.loss, mdl.vars)
      summary_utils.CollectVarHistogram(var_grads)

  def testGradientMult(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      p = self._testParams()
      mdl = p.cls(p)
      mdl.FPropDefaultTheta()
      var_grads = py_utils.ComputeGradients(mdl.loss, mdl.vars)
      py_utils.ApplyGradMultiplier(var_grads, -1.1)

  def testLRDecay(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      p = self._testParams()
      tp = p.train
      tp.lr_schedule.boundaries = [300000, 400000, 500000]
      tp.lr_schedule.values = [1.0, 0.1, 0.01, 0.001]
      lrs = tp.lr_schedule.cls(tp.lr_schedule)
      steps = [299999, 300001, 399999, 400001, 499999, 500001]
      fetches = [lrs.Value(_) for _ in steps]
      values = sess.run(fetches)
      self.assertAllClose([1.0, 0.1, 0.1, 0.01, 0.01, 0.001], values)

  def testBatchSplit(self):

    def Run(num_splits):
      p = self._testParams()
      with self.session(use_gpu=False, graph=tf.Graph()) as sess:
        tf.set_random_seed(93820981)
        p.is_eval = True
        p.input.cur_iter_in_seed = False
        p.input.bucket_batch_limit = [
            b * 2 / num_splits for b in p.input.bucket_batch_limit
        ]
        with cluster_factory.ForTestingWorker(gpus=num_splits):
          mdl = p.cls(p)
          metrics = mdl.FPropDefaultTheta()[0]
        tf.global_variables_initializer().run()
        return sess.run(metrics['loss'])

    res1, res2 = Run(1), Run(2)
    self.assertAllClose(res1[0], res2[0])
    self.assertAllEqual(res1[1], res2[1])

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

      p.is_eval = True
      return p

    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      p = _CreateModelParamsForTest()
      mdl = p.cls(p)
      subgraphs = mdl.Inference()
      self.assertTrue('default' in subgraphs)

      fetches, feeds = subgraphs['default']
      self.assertTrue('wav' in feeds)
      for name in ['hypotheses', 'scores', 'src_frames', 'encoder_frames']:
        self.assertTrue(name in fetches)

      with open(
          test_helper.test_src_dir_path('tools/testdata/gan_or_vae.16k.wav'),
          'r') as f:
        wav = f.read()
      sess.run(tf.global_variables_initializer())
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
