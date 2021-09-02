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
"""Tests for speech decoder."""

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import layers as lingvo_layers
from lingvo.core import py_utils
from lingvo.core import symbolic
from lingvo.core import test_utils
from lingvo.core.ops.hyps_pb2 import Hypothesis
from lingvo.tasks.asr import decoder
import numpy as np

from google.protobuf import text_format

FLAGS = tf.flags.FLAGS


def _DecoderParams(vn_config, num_classes=32, num_rnn_layers=1):
  """Create a small decoder for testing."""
  p = decoder.AsrDecoder.Params()
  p.random_seed = 12345

  p.name = 'decoder'
  uniform_init = py_utils.WeightInit.Uniform(0.1, seed=12345)

  # Set up embedding params.
  p.emb.vocab_size = num_classes
  p.emb.max_num_shards = 1
  p.emb.params_init = uniform_init

  # Set up decoder RNN layers.
  p.rnn_layers = num_rnn_layers
  rnn_params = p.rnn_cell_tpl
  rnn_params.params_init = uniform_init

  # Set up attention.
  p.attention.hidden_dim = 16
  p.attention.params_init = uniform_init

  # Set up final softmax layer.
  p.softmax.num_classes = num_classes
  p.softmax.params_init = uniform_init

  # Set up variational noise params.
  p.vn = vn_config
  p.vn.scale = 0.1

  p.target_seq_len = 5
  p.source_dim = 8
  p.emb_dim = 2
  p.rnn_cell_dim = 4

  return p


def _CreateSourceAndTargets(params):
  """Creates encoder outputs and targets from params for the decoder."""
  src_seq_len = 5
  src_enc = tf.random.normal([src_seq_len, 2, 8],
                             seed=982774838,
                             dtype=py_utils.FPropDtype(params))
  src_enc_padding = tf.constant(
      [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
      dtype=py_utils.FPropDtype(params))
  encoder_outputs = py_utils.NestedMap(encoded=src_enc, padding=src_enc_padding)
  # shape=[4, 5]
  target_ids = tf.transpose(
      tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15], [5, 6, 7, 8],
                   [10, 5, 2, 5]],
                  dtype=tf.int32))
  # shape=[4, 5]
  target_labels = tf.transpose(
      tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13], [5, 7, 8, 10],
                   [10, 5, 2, 4]],
                  dtype=tf.int32))
  # shape=[4, 5]
  target_paddings = tf.transpose(
      tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                   [1, 1, 1, 0]],
                  dtype=py_utils.FPropDtype(params)))
  target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
  target_weights = 1.0 - target_paddings
  # ids/labels/weights/paddings are all in [batch, time] shape.
  targets = py_utils.NestedMap({
      'ids': target_ids,
      'labels': target_labels,
      'weights': target_weights,
      'paddings': target_paddings,
      'transcripts': target_transcripts,
  })
  return encoder_outputs, targets


class DecoderTest(test_utils.TestCase):

  def _getDecoderFPropMetrics(self, params):
    """Creates decoder from params and computes metrics with random inputs."""
    dec = params.Instantiate()
    encoder_outputs, targets = _CreateSourceAndTargets(params)
    decoder_outputs = dec.FPropDefaultTheta(encoder_outputs, targets)
    return decoder_outputs.metrics, decoder_outputs.per_sequence['loss']

  def _testDecoderFPropHelper(self, params):
    metrics, per_sequence_loss = self._getDecoderFPropMetrics(params)
    return metrics['loss'], per_sequence_loss

  def _testDecoderFPropFloatHelper(self,
                                   func_inline=False,
                                   num_decoder_layers=1,
                                   target_seq_len=5,
                                   residual_start=0):
    """Computes decoder from params and computes loss with random inputs."""
    cluster = cluster_factory.ForTestingWorker(add_summary=True)
    config = tf.config_pb2.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                do_function_inlining=func_inline)))
    with cluster, self.session(use_gpu=False, config=config):
      tf.random.set_seed(8372749040)
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = _DecoderParams(vn_config)
      p.rnn_layers = num_decoder_layers
      p.residual_start = residual_start
      p.target_seq_len = target_seq_len
      dec = p.Instantiate()
      src_seq_len = 5
      src_enc = tf.random.normal([src_seq_len, 2, 8], seed=9283748)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)
      target_ids = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15],
                       [5, 6, 7, 8], [10, 5, 2, 5]],
                      dtype=tf.int32))
      target_labels = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13],
                       [5, 7, 8, 10], [10, 5, 2, 4]],
                      dtype=tf.int32))
      target_paddings = tf.transpose(
          tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                       [1, 1, 1, 1]],
                      dtype=tf.float32))
      target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
      target_weights = 1.0 - target_paddings
      targets = py_utils.NestedMap({
          'ids': target_ids,
          'labels': target_labels,
          'weights': target_weights,
          'paddings': target_paddings,
          'transcripts': target_transcripts,
      })
      metrics = dec.FPropDefaultTheta(encoder_outputs, targets).metrics
      loss = metrics['loss'][0]
      correct_predicts = metrics['fraction_of_correct_next_step_preds'][0]
      summaries = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

      self.evaluate(tf.global_variables_initializer())
      loss_v, _ = self.evaluate([loss, correct_predicts])

      summaries.eval()

      return loss_v

  # Actual tests follow.

  def testDecoderConstruction(self):
    """Test that decoder can be constructed from params."""
    p = _DecoderParams(
        vn_config=py_utils.VariationalNoiseParams(
            None, True, False, seed=12345))
    _ = decoder.AsrDecoder(p)

  def testDecoderFProp(self):
    """Create decoder with default params, and verify that FProp runs."""
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)

      p = _DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(
              None, True, False, seed=12345))

      metrics, per_sequence_loss = self._getDecoderFPropMetrics(params=p)
      self.assertIn('fraction_of_correct_next_step_preds', metrics)
      self.evaluate(tf.global_variables_initializer())
      metrics_val, per_sequence_loss_val = self.evaluate(
          [metrics, per_sequence_loss])
      tf.logging.info('metrics=%s, per_sequence_loss=%s', metrics_val,
                      per_sequence_loss_val)

      self.assertEqual(metrics_val['loss'], metrics_val['log_pplx'])
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testDecoderFPropWithMeanSeqLoss(self):
    """Create and fprop a decoder with different dims per layer."""
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)

      p = _DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(
              None, True, False, seed=12345))
      p.token_normalized_per_seq_loss = True
      p.per_token_avg_loss = False

      metrics, per_sequence_loss = self._getDecoderFPropMetrics(params=p)
      self.evaluate(tf.global_variables_initializer())
      metrics_val, per_sequence_loss_val = self.evaluate(
          [metrics, per_sequence_loss])
      tf.logging.info('metrics=%s, per_sequence_loss=%s', metrics_val,
                      per_sequence_loss_val)

      self.assertNotEqual(metrics_val['loss'][0], metrics_val['log_pplx'][0])
      self.assertAllClose(metrics_val['loss'], (3.39413, 4.0))
      self.assertAllClose(metrics_val['log_pplx'], (3.427874, 15.0))
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testDecoderFPropWithProjection(self):
    """Create decoder with projection layers, and verify that FProp runs."""
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)

      p = _DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(
              None, True, False, seed=12345))
      rnn_cell_tpl = p.rnn_cell_tpl
      p.rnn_cell_tpl = [
          rnn_cell_tpl.Copy().Set(
              num_output_nodes=i + 2, num_hidden_nodes=i + 5)
          for i in range(p.rnn_layers)
      ]
      p.rnn_cell_dim = -1
      p.rnn_cell_hidden_dim = -1

      loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
      self.evaluate(tf.global_variables_initializer())
      loss_val, per_sequence_loss_val = self.evaluate([loss, per_sequence_loss])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testDecoderFPropWithPerLayerDims(self):
    """Create and fprop a decoder with different dims per layer."""
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)

      p = _DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(
              None, True, False, seed=12345))
      p.rnn_cell_hidden_dim = 6

      loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
      self.evaluate(tf.global_variables_initializer())
      loss_val, per_sequence_loss_val = self.evaluate([loss, per_sequence_loss])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testDecoderFPropDtype(self):
    """Create decoder with different fprop_type, and verify that FProp runs."""
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)

      p = _DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(
              None, True, False, seed=12345))
      p.fprop_dtype = tf.float64

      loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
      self.evaluate(tf.global_variables_initializer())
      loss_val, per_sequence_loss_val = self.evaluate([loss, per_sequence_loss])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testDecoderFPropDeterministicAttentionDropout(self):
    """Verify that attention dropout is deterministic given fixed seeds."""
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)
      p = _DecoderParams(
          py_utils.VariationalNoiseParams(None, True, False, seed=1792))

      p.use_while_loop_based_unrolling = False
      p.attention.atten_dropout_prob = 0.5
      p.attention.atten_dropout_deterministic = True

      loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
      global_step = py_utils.GetGlobalStep()
      self.evaluate(tf.global_variables_initializer())
      loss_val, per_sequence_loss_val, global_steps_val = self.evaluate(
          [loss, per_sequence_loss, global_step])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      self.assertAllClose([3.332992, 15.0], loss_val)
      self.assertAllClose([13.942583, 9.632538, 9.677502, 16.742266],
                          per_sequence_loss_val)
      self.assertEqual(0, global_steps_val)

      # Run another step to test global_step and time_step are incremented
      # correctly.
      self.evaluate(tf.assign_add(global_step, 1))
      loss_val, per_sequence_loss_val, global_steps_val = self.evaluate(
          [loss, per_sequence_loss, global_step])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      self.assertAllClose([3.565631, 15.0], loss_val)
      self.assertAllClose([14.560061, 10.566417, 10.554007, 17.803982],
                          per_sequence_loss_val)
      self.assertEqual(1, global_steps_val)

  def testLabelSmoothing(self):
    """Verify that loss computation with label smoothing is as expected.."""
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)

      p = _DecoderParams(vn_config=py_utils.VariationalNoiseParams(None))
      p.label_smoothing = lingvo_layers.LocalizedLabelSmoother.Params()
      p.label_smoothing.offsets = [-2, -1, 1, 2]
      p.label_smoothing.weights = [0.015, 0.035, 0.035, 0.015]

      loss, _ = self._testDecoderFPropHelper(params=p)
      self.evaluate(tf.global_variables_initializer())
      loss_val = self.evaluate(loss[0])

      print('loss = ', loss_val)
      test_utils.CompareToGoldenSingleFloat(self, 3.471763, loss_val)

  def testDecoderFPropFloatNoInline(self):
    actual_value = self._testDecoderFPropFloatHelper(func_inline=False)
    test_utils.CompareToGoldenSingleFloat(self, 3.458980, actual_value)

  def testDecoderFPropFloatNoInlinePadTargetsToLongerLength(self):
    actual_value = self._testDecoderFPropFloatHelper(
        func_inline=False, target_seq_len=10)
    test_utils.CompareToGoldenSingleFloat(self, 3.458980, actual_value)

  def testDecoderFPropFloatInline(self):
    actual_value = self._testDecoderFPropFloatHelper(func_inline=True)
    test_utils.CompareToGoldenSingleFloat(self, 3.458980, actual_value)

  def testDecoderFPropFloatNoInline2Layers(self):
    actual_value = self._testDecoderFPropFloatHelper(
        func_inline=False, num_decoder_layers=2)
    test_utils.CompareToGoldenSingleFloat(self, 3.457761, actual_value)

  def testDecoderFPropFloatInline2Layers(self):
    actual_value = self._testDecoderFPropFloatHelper(
        func_inline=True, num_decoder_layers=2)
    test_utils.CompareToGoldenSingleFloat(self, 3.457761, actual_value)

  def testDecoderFPropFloat2LayersResidual(self):
    actual_value = self._testDecoderFPropFloatHelper(
        num_decoder_layers=2, residual_start=2)
    test_utils.CompareToGoldenSingleFloat(self, 3.458294, actual_value)

  def testDecoderFPropDouble(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)
      np.random.seed(827374)

      p = _DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(None, False, False))
      p.dtype = tf.float64

      dec = decoder.AsrDecoder(p)
      src_seq_len = 5
      src_enc = tf.constant(
          np.random.uniform(size=(src_seq_len, 2, 8)), tf.float64)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float64)
      target_ids = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15],
                       [5, 6, 7, 8], [10, 5, 2, 5]],
                      dtype=tf.int32))
      target_labels = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13],
                       [5, 7, 8, 10], [10, 5, 2, 4]],
                      dtype=tf.int32))
      target_paddings = tf.transpose(
          tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                       [1, 1, 1, 1]],
                      dtype=tf.float64))
      target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
      target_weights = 1.0 - target_paddings
      targets = py_utils.NestedMap({
          'ids': target_ids,
          'labels': target_labels,
          'weights': target_weights,
          'paddings': target_paddings,
          'transcripts': target_transcripts,
      })
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)
      metrics = dec.FPropDefaultTheta(encoder_outputs, targets).metrics
      loss = metrics['loss'][0]

      self.evaluate(tf.global_variables_initializer())

      test_utils.CompareToGoldenSingleFloat(self, 3.467679, loss.eval())
      # Second run to make sure the function is determistic.
      test_utils.CompareToGoldenSingleFloat(self, 3.467679, loss.eval())

  def _testDecoderFPropGradientCheckerHelper(self, func_inline=False):
    config = tf.config_pb2.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                do_function_inlining=func_inline)))
    with self.session(use_gpu=False, config=config) as sess:
      tf.random.set_seed(8372749040)
      np.random.seed(274854)
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = _DecoderParams(vn_config)
      p.dtype = tf.float64

      dec = p.Instantiate()
      src_seq_len = 5
      src_enc = tf.constant(
          np.random.uniform(size=(src_seq_len, 2, 8)), tf.float64)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float64)
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)
      target_ids = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15],
                       [5, 6, 7, 8], [10, 5, 2, 5]],
                      dtype=tf.int32))
      target_labels = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13],
                       [5, 7, 8, 10], [10, 5, 2, 4]],
                      dtype=tf.int32))
      target_paddings = tf.transpose(
          tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                       [1, 1, 1, 1]],
                      dtype=tf.float64))
      target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
      target_weights = 1.0 - target_paddings

      targets = py_utils.NestedMap({
          'ids': target_ids,
          'labels': target_labels,
          'weights': target_weights,
          'paddings': target_paddings,
          'transcripts': target_transcripts,
      })
      metrics = dec.FPropDefaultTheta(encoder_outputs, targets).metrics
      loss = metrics['loss'][0]
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)

      def DenseGrad(var, grad):
        if isinstance(grad, tf.Tensor):
          return grad
        elif isinstance(grad, tf.IndexedSlices):
          return tf.math.unsorted_segment_sum(grad.values, grad.indices,
                                              tf.shape(var)[0])

      dense_grads = [DenseGrad(x, y) for (x, y) in zip(all_vars, grads)]

      self.evaluate(tf.global_variables_initializer())

      test_utils.CompareToGoldenSingleFloat(self, 3.458078, loss.eval())
      # Second run to make sure the function is determistic.
      test_utils.CompareToGoldenSingleFloat(self, 3.458078, loss.eval())

      symbolic_grads = [x.eval() for x in dense_grads if x is not None]
      numerical_grads = []
      for v in all_vars:
        numerical_grads.append(test_utils.ComputeNumericGradient(sess, loss, v))

      for x, y in zip(symbolic_grads, numerical_grads):
        self.assertAllClose(x, y)

  def testDecoderFPropGradientCheckerNoInline(self):
    self._testDecoderFPropGradientCheckerHelper(func_inline=False)

  def testDecoderFPropGradientCheckerInline(self):
    self._testDecoderFPropGradientCheckerHelper(func_inline=True)

  def _testDecoderBeamSearchDecodeHelperWithOutput(self,
                                                   params,
                                                   src_seq_len=None,
                                                   src_enc_padding=None):
    config = tf.config_pb2.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(do_function_inlining=False)))
    p = params
    with self.session(use_gpu=False, config=config), self.SetEval(True):
      tf.random.set_seed(837274904)
      np.random.seed(837575)
      p.beam_search.num_hyps_per_beam = 4
      p.beam_search.length_normalization = 10.
      p.dtype = tf.float32
      p.target_seq_len = 5

      dec = p.Instantiate()
      if src_seq_len is None:
        src_seq_len = 5
      src_enc = tf.constant(
          np.random.uniform(size=(src_seq_len, 2, 8)), tf.float32)
      if src_enc_padding is None:
        src_enc_padding = tf.constant(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=tf.float32)

      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding)
      topk_hyps = dec.BeamSearchDecode(encoder_outputs).topk_hyps
      self.evaluate(tf.global_variables_initializer())

      softmax_wts = self.evaluate(dec.vars.softmax)
      print('softmax wts = ', softmax_wts)

      topk_hyps_serialized = self.evaluate([topk_hyps])[0]
      hyp = Hypothesis()
      hyp.ParseFromString(topk_hyps_serialized[1, 2])
      return hyp

  def _VerifyHypothesesMatch(self, hyp1, hyp2):
    tf.logging.info('hyp1 = %s', hyp1)
    tf.logging.info('hyp2 = %s', hyp2)
    self.assertEqual(hyp1.beam_id, hyp2.beam_id)
    self.assertEqual(list(hyp1.ids), list(hyp2.ids))
    self.assertAllClose(hyp1.scores, hyp2.scores)
    self.assertAllClose(hyp1.normalized_score, hyp2.normalized_score)
    self.assertEqual(len(hyp1.atten_vecs), len(hyp2.atten_vecs))
    for av1, av2 in zip(hyp1.atten_vecs, hyp2.atten_vecs):
      self.assertAllClose(av1.prob, av2.prob)

  def testDecoderBeamSearchDecode(self):
    np.random.seed(837575)

    p = _DecoderParams(
        vn_config=py_utils.VariationalNoiseParams(None, False, False),
        num_classes=8)
    p.beam_search.num_hyps_per_beam = 4
    p.dtype = tf.float32
    p.target_seq_len = 5

    expected_str = """
      beam_id: 1
      ids: 0
      ids: 6
      ids: 2
      scores: -2.021608
      scores: -2.000098
      scores: -2.036338
      normalized_score: -0.0550976
      atten_vecs {
        prob: 0.330158
        prob: 0.342596
        prob: 0.327246
        prob: 0.0
        prob: 0.0
      }
      atten_vecs {
        prob: 0.330158
        prob: 0.342597
        prob: 0.327245
        prob: 0.0
        prob: 0.0
      }
      atten_vecs {
        prob: 0.330158
        prob: 0.342597
        prob: 0.327245
        prob: 0.0
        prob: 0.0
      }
    """
    expected_hyp = Hypothesis()
    text_format.Parse(expected_str, expected_hyp)

    decoded_hyp = self._testDecoderBeamSearchDecodeHelperWithOutput(params=p)
    self._VerifyHypothesesMatch(expected_hyp, decoded_hyp)

  def testDecoderSampleTargetSequences(self):
    p = _DecoderParams(
        vn_config=py_utils.VariationalNoiseParams(None, False, False),
        num_classes=8)
    p.target_seq_len = 5
    p.random_seed = 1
    config = tf.config_pb2.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(do_function_inlining=False)))
    with self.session(use_gpu=False, config=config):
      tf.random.set_seed(8372740)
      np.random.seed(35315)
      dec = p.Instantiate()
      source_sequence_length = 5
      batch_size = 4
      source_encodings = tf.constant(
          np.random.normal(
              size=[source_sequence_length, batch_size, p.source_dim]),
          dtype=tf.float32)
      source_encoding_padding = tf.constant(
          [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0],
           [0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
          dtype=tf.float32)
      encoder_outputs = py_utils.NestedMap(
          encoded=source_encodings, padding=source_encoding_padding)
      sampled_sequences = dec.SampleTargetSequences(
          dec.theta, encoder_outputs, random_seed=tf.cast(123, tf.int32))
      self.assertAllEqual([batch_size, p.target_seq_len],
                          sampled_sequences.ids.shape)
      self.evaluate(tf.global_variables_initializer())
      decoder_output = self.evaluate(sampled_sequences)
      print('ids=%s' % np.array_repr(decoder_output.ids))
      lens = np.sum(1 - decoder_output.paddings, axis=1)
      print('lens=%s' % lens)
      # pyformat: disable
      # pylint: disable=bad-whitespace,bad-continuation
      expected_ids = [[6, 2, 2, 2, 2],
                      [0, 0, 7, 5, 1],
                      [6, 1, 5, 1, 5],
                      [6, 7, 7, 4, 4]]
      # pylint: enable=bad-whitespace,bad-continuation
      # pyformat: enable
      expected_lens = [2, 5, 5, 5]
      self.assertAllEqual(expected_lens, lens)
      self.assertAllEqual(expected_ids, decoder_output.ids)

      # Sample again with the same random seed.
      decoder_output2 = self.evaluate(
          dec.SampleTargetSequences(
              dec.theta, encoder_outputs, random_seed=tf.cast(123, tf.int32)))
      # Get the same output.
      self.assertAllEqual(decoder_output.ids, decoder_output2.ids)
      self.assertAllEqual(decoder_output.paddings, decoder_output2.paddings)

      # Sample again with a different random seed.
      decoder_output3 = self.evaluate(
          dec.SampleTargetSequences(
              dec.theta, encoder_outputs, random_seed=tf.cast(123456,
                                                              tf.int32)))
      # Get different sequences.
      self.assertNotAllClose(expected_ids, decoder_output3.ids)

  def testDecoderFPropWithSymbolicShape(self):
    """Create decoder with default params, and verify that FProp runs."""
    with self.session():
      p = _DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(
              None, True, False, seed=12345))
      p.rnn_cell_dim = symbolic.Symbol('rnn_cell_dim')

      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES,
                                     {p.rnn_cell_dim: 6}):
        loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
        self.evaluate(tf.global_variables_initializer())
        loss_val, per_sequence_loss_val = self.evaluate(
            [loss, per_sequence_loss])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testUpdateTargetVocabSize(self):
    p = _DecoderParams(
        vn_config=py_utils.VariationalNoiseParams(
            None, True, False, seed=12345))
    p.label_smoothing = lingvo_layers.LocalizedLabelSmoother.Params()
    p.label_smoothing.num_classes = p.softmax.num_classes

    vocab_size = 1024
    self.assertNotEqual(p.emb.vocab_size, vocab_size)
    self.assertNotEqual(p.softmax.num_classes, vocab_size)
    self.assertNotEqual(p.fusion.lm.vocab_size, vocab_size)
    self.assertNotEqual(p.label_smoothing.num_classes, vocab_size)
    p = p.cls.UpdateTargetVocabSize(p, vocab_size)
    dec = p.Instantiate()
    self.assertEqual(vocab_size, dec.params.emb.vocab_size)
    self.assertEqual(vocab_size, dec.params.softmax.num_classes)
    self.assertEqual(vocab_size, p.fusion.lm.vocab_size)
    self.assertEqual(vocab_size, p.label_smoothing.num_classes)

  def testDecoderFPropWithAdapters(self):
    """Create decoder with adapters, and verify that FProp runs."""
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)

      params = _DecoderParams(
          num_rnn_layers=2,
          vn_config=py_utils.VariationalNoiseParams(
              None, True, False, seed=12345))
      params.rnn_cell_dim = 3
      params.adapter_layer_tpl.Set(
          bottleneck_dim=4,
          num_tasks=16,
          projection_params_init=py_utils.WeightInit.Gaussian(0.01))
      params.adapter_task_id_field = 'domain_ids'

      dec = params.Instantiate()
      src_seq_len = 5
      src_enc = tf.random.normal([src_seq_len, 2, 8],
                                 seed=982774838,
                                 dtype=py_utils.FPropDtype(params))
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=py_utils.FPropDtype(params))
      domain_ids = tf.constant(np.random.randint(low=0, high=16, size=[2]))
      encoder_outputs = py_utils.NestedMap(
          encoded=src_enc, padding=src_enc_padding, domain_ids=domain_ids)
      # shape=[4, 5]
      target_ids = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15],
                       [5, 6, 7, 8], [10, 5, 2, 5]],
                      dtype=tf.int32))
      # shape=[4, 5]
      target_labels = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13],
                       [5, 7, 8, 10], [10, 5, 2, 4]],
                      dtype=tf.int32))
      # shape=[4, 5]
      target_paddings = tf.transpose(
          tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                       [1, 1, 1, 0]],
                      dtype=py_utils.FPropDtype(params)))
      target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
      target_weights = 1.0 - target_paddings
      # ids/labels/weights/paddings are all in [batch, time] shape.
      targets = py_utils.NestedMap({
          'ids': target_ids,
          'labels': target_labels,
          'weights': target_weights,
          'paddings': target_paddings,
          'transcripts': target_transcripts,
      })
      decoder_outputs = dec.FPropDefaultTheta(encoder_outputs, targets)
      metrics = decoder_outputs.metrics
      per_sequence_loss = decoder_outputs.per_sequence['loss']

      self.assertIn('fraction_of_correct_next_step_preds', metrics)
      self.evaluate(tf.global_variables_initializer())
      metrics_val, per_sequence_loss_val = self.evaluate(
          [metrics, per_sequence_loss])
      tf.logging.info('metrics=%s, per_sequence_loss=%s', metrics_val,
                      per_sequence_loss_val)

      self.assertEqual(metrics_val['loss'], metrics_val['log_pplx'])
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))


class DecoderWithConfidenceTest(test_utils.TestCase):

  def _testComputePredictionsHelper(self,
                                    use_while_loop_based_unrolling=False,
                                    confidence_module=False):
    """Create decoder and confidence prediction, and verify that FProp runs."""
    with self.session():
      p = _DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(
              None, True, False, seed=12345))
      p.use_while_loop_based_unrolling = use_while_loop_based_unrolling
      if confidence_module:
        p.confidence = lingvo_layers.FeedForwardNet.Params()
        p.confidence.hidden_layer_dims = [8, 1]
        p.confidence.activation = ['RELU', 'NONE']

      dec = p.Instantiate()
      encoder_outputs, targets = _CreateSourceAndTargets(p)
      predictions = dec.ComputePredictions(dec.theta, encoder_outputs, targets)

      self.evaluate(tf.global_variables_initializer())
      predictions_val = self.evaluate(predictions)
      self.assertAllEqual(predictions_val['logits'].shape, [4, 5, 32])
      self.assertAllEqual(predictions_val['softmax_input'].shape, [5, 4, 12])
      if p.confidence is not None:
        self.assertAllEqual(predictions_val['confidence_logits'].shape, [4, 5])

  def testComputePredictionsDynamic(self):
    self._testComputePredictionsHelper(use_while_loop_based_unrolling=True)

  def testComputePredictionsFunctional(self):
    self._testComputePredictionsHelper(use_while_loop_based_unrolling=False)

  def testComputePredictionsDynamicWithConfidence(self):
    self._testComputePredictionsHelper(
        use_while_loop_based_unrolling=True, confidence_module=True)

  def testComputePredictionsFunctionalWithConfidence(self):
    self._testComputePredictionsHelper(
        use_while_loop_based_unrolling=False, confidence_module=True)


if __name__ == '__main__':
  tf.test.main()
