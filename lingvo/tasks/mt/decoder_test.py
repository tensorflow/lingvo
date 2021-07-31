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
"""Tests for mt.decoder."""

import random
from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import input_generator_helper as ig_helper
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import test_utils
from lingvo.core.ops.hyps_pb2 import Hypothesis
from lingvo.core.test_utils import CompareToGoldenSingleFloat
from lingvo.tasks.mt import decoder
import numpy as np

FLAGS = tf.flags.FLAGS

_NUMPY_RANDOM_SEED = 9885784
_TF_RANDOM_SEED = 8372749040


class DecoderTestCaseBase(test_utils.TestCase):

  def _Inputs(self, dtype=tf.float32, init_step_ids=False):
    np.random.seed(_NUMPY_RANDOM_SEED)
    src_seq_len = 5
    # batch = 2
    src_enc = tf.constant(
        np.random.normal(size=[src_seq_len, 2, 4]), dtype=dtype)
    src_enc_padding = tf.constant(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=dtype)
    # batch = 4, time = 3.
    target_ids = tf.transpose(
        tf.constant([[0, 1, 2, 3], [0, 5, 6, 7], [0, 10, 11, 12]],
                    dtype=tf.int32))
    target_labels = tf.transpose(
        tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [10, 11, 12, 13]],
                    dtype=tf.int32))
    target_paddings = tf.transpose(
        tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=dtype))
    target_weights = 1.0 - target_paddings
    targets = py_utils.NestedMap({
        'ids': target_ids,
        'labels': target_labels,
        'weights': target_weights,
        'paddings': target_paddings
    })
    encoder_outputs = py_utils.NestedMap(
        encoded=src_enc, padding=src_enc_padding, segment_id=None)
    if init_step_ids:
      tgt_prefix = tf.constant(np.random.randint(4, size=[2]), dtype=tf.int32)
      encoder_outputs['init_step_ids'] = tgt_prefix
    return encoder_outputs, targets

  def _DecoderParams(
      self,
      per_word_avg_loss=False,
      dtype=tf.float32,
      fprop_dtype=None,
      decoder_cls=decoder.MTDecoderV1,
  ):
    p = decoder_cls.Params()
    p.name = 'decoder'
    p.source_dim = 4
    p.emb.vocab_size = 16
    p.emb.embedding_dim = 4
    p.emb.max_num_shards = 1
    p.rnn_cell_dim = 4
    p.rnn_layers = 3
    p.attention.hidden_dim = 2
    p.softmax.num_classes = 16
    p.softmax.num_shards = 1
    p.per_word_avg_loss = per_word_avg_loss
    p.dtype = dtype
    p.target_seq_len = 5
    p.random_seed = 12345
    p.emb.params_init = py_utils.WeightInit.Uniform(0.04, 12345)
    p.atten_rnn_cell_tpl.params_init = py_utils.WeightInit.Uniform(0.04, 12345)
    p.rnn_cell_tpl.params_init = py_utils.WeightInit.Uniform(0.04, 12345)
    p.softmax.params_init = py_utils.WeightInit.Uniform(0.04, 123)

    for lp in base_layer.RecursiveFindLayerParams(p):
      lp.dtype = dtype

    if fprop_dtype:
      py_utils.UpdateFpropDtype(p, fprop_dtype)

    return p

  def _DecoderFPropHelper(self,
                          decoder_cls,
                          dtype,
                          fprop_dtype,
                          feed_att_context_to_softmax,
                          expected_loss,
                          per_example_tensors=False,
                          use_deterministic_cell=False):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(
          dtype=dtype, fprop_dtype=fprop_dtype, decoder_cls=decoder_cls)
      p.per_example_tensors = per_example_tensors
      p.feed_attention_context_vec_to_softmax = feed_att_context_to_softmax
      if use_deterministic_cell:
        p.rnn_cell_tpl = rnn_cell.LSTMCellSimple.Params()
        p.rnn_cell_tpl.deterministic = True
      dec = p.Instantiate()
      encoder_outputs, targets = self._Inputs(dtype=fprop_dtype)
      fprop_out = dec.FPropDefaultTheta(encoder_outputs, targets)
      loss = fprop_out.metrics['loss'][0]

      self.evaluate(tf.global_variables_initializer())
      actual_loss = loss.eval()
      print('actual loss = ', actual_loss)
      CompareToGoldenSingleFloat(self, expected_loss, actual_loss)
      if per_example_tensors:
        per_example = fprop_out.per_sequence
        self.assertIn('loss', per_example)
        self.assertAllEqual(per_example['loss'].shape.as_list(), [4])

  def _DecoderGradientCheckerHelper(self,
                                    decoder_cls,
                                    feed_att_context_to_softmax=False):
    with self.session(use_gpu=True, graph=tf.Graph()) as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=tf.float64, decoder_cls=decoder_cls)
      p.feed_attention_context_vec_to_softmax = feed_att_context_to_softmax
      dec = p.Instantiate()
      encoder_outputs, targets = self._Inputs(dtype=tf.float64)
      loss, _ = dec.FPropDefaultTheta(encoder_outputs, targets).metrics['loss']
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      print('num of vars ', len(all_vars))

      def DenseGrad(var, grad):
        if isinstance(grad, tf.Tensor):
          return grad
        elif isinstance(grad, tf.IndexedSlices):
          return tf.math.unsorted_segment_sum(grad.values, grad.indices,
                                              tf.shape(var)[0])

      grads = [DenseGrad(x, y) for x, y in zip(all_vars, grads)]

      self.evaluate(tf.global_variables_initializer())
      symbolic_grads = [gd.eval() for gd in grads]
      numerical_grads = []
      for v in all_vars:
        numerical_grads.append(
            test_utils.ComputeNumericGradient(sess, loss, v, delta=1e-5))

      rets = {}
      for v, x, y in zip(all_vars, symbolic_grads, numerical_grads):
        print('symbolic_grads, numerical_grads :', v.name)
        print(x)
        print(y)
        self.assertAllClose(x, y)
        rets[v.name] = x

      return rets

  def _DecoderPerWordAvgLossFPropHelper(self,
                                        decoder_cls,
                                        feed_att_context_to_softmax=False):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(True, decoder_cls=decoder_cls)
      p.feed_attention_context_vec_to_softmax = feed_att_context_to_softmax
      dec = p.Instantiate()
      encoder_outputs, targets = self._Inputs()
      loss, _ = dec.FPropDefaultTheta(encoder_outputs, targets).metrics['loss']
      self.evaluate(tf.global_variables_initializer())
      actual_loss = loss.eval()
      print('actual loss = ', actual_loss)
      if p.feed_attention_context_vec_to_softmax:
        CompareToGoldenSingleFloat(self, 2.768977, actual_loss)
      else:
        CompareToGoldenSingleFloat(self, 2.772613, actual_loss)


class DecoderTest(DecoderTestCaseBase, parameterized.TestCase):

  def testDecoderConstruction(self):
    p = self._DecoderParams()
    _ = decoder.MTDecoderV1(p)

  def testDecoderFPropFunctional(self):
    self._DecoderFPropHelper(decoder.MTDecoderV1, tf.float64, tf.float64, False,
                             7.624605)

  def testDecoderFPropFunctionalFloat64Dtype(self):
    self._DecoderFPropHelper(decoder.MTDecoderV1, tf.float32, tf.float64, False,
                             7.624684)

  def testDecoderFPropFunctionalFloat64FpropDtype(self):
    self._DecoderFPropHelper(decoder.MTDecoderV1, tf.float64, tf.float32, False,
                             7.624604)

  def testDecoderFPropFunctionalDeterministic(self):
    self._DecoderFPropHelper(
        decoder.MTDecoderV1,
        tf.float64,
        tf.float64,
        False,
        7.624416,
        use_deterministic_cell=True)

  def testDecoderFPropFunctionalFeedingAttContext(self):
    self._DecoderFPropHelper(decoder.MTDecoderV1, tf.float64, tf.float64, True,
                             7.640674)

  def testDecoderFPropPerExampleTensors(self):
    self._DecoderFPropHelper(
        decoder.MTDecoderV1,
        tf.float64,
        tf.float64,
        False,
        7.624605,
        per_example_tensors=True)

  def testDecoderFPropFactorizedEmbedding(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(decoder_cls=decoder.MTDecoderV1)
      p.per_example_tensors = True
      p.emb.embedding_dim = 2
      p.rnn_cell_dim = 4
      proj_tpl = layers.ProjectionLayer.Params().Copy()
      proj_tpl.batch_norm = False
      proj_tpl.activation = 'NONE'
      proj_tpl.has_bias = True
      proj_tpl.params_init = py_utils.WeightInit.Uniform(0.04, 1234)
      p.emb_projection_tpl = proj_tpl
      p.softmax = layers.SharedSoftmaxLayer.Params().Set(
          softmax=layers.SimpleFullSoftmax.Params().Set(
              num_shards=p.softmax.num_shards),
          num_classes=p.softmax.num_classes,
          params_init=p.softmax.params_init.Copy(),
          embedding_dim=p.emb.embedding_dim,
          vocab_size=p.softmax.num_classes)
      dec = p.Instantiate()
      encoder_outputs, targets = self._Inputs()
      fprop_out = dec.FPropDefaultTheta(encoder_outputs, targets)
      loss = fprop_out.metrics['loss'][0]

      self.evaluate(tf.global_variables_initializer())
      actual_loss = loss.eval()
      expected_loss = 7.6245975
      CompareToGoldenSingleFloat(self, expected_loss, actual_loss)
      per_example = fprop_out.per_sequence
      self.assertIn('loss', per_example)
      self.assertAllEqual(per_example['loss'].shape.as_list(), [4])

  def testDecoderBPropFunctional(self):
    self._DecoderGradientCheckerHelper(decoder.MTDecoderV1)

  def testDecoderBPropFunctionalFeedingAttContext(self):
    self._DecoderGradientCheckerHelper(
        decoder.MTDecoderV1, feed_att_context_to_softmax=True)

  def testDecoderPerWordAvgLossFPropFunctional(self):
    self._DecoderPerWordAvgLossFPropHelper(decoder.MTDecoderV1)

  def testDecoderPerWordAvgLossFPropFunctionalFeedingAttContext(self):
    self._DecoderPerWordAvgLossFPropHelper(
        decoder.MTDecoderV1, feed_att_context_to_softmax=True)

  def testBeamSearchDecode(self, dtype=tf.float32):
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      src_batch = 2
      p = self._DecoderParams(dtype=dtype)
      src_time = p.target_seq_len
      p.beam_search.num_hyps_per_beam = 2
      p.rnn_cell_dim = 32
      dec = decoder.MTDecoderV1(p)
      encoder_outputs, _ = self._Inputs(dtype=dtype)
      decode = dec.BeamSearchDecode(encoder_outputs)
      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # self.evaluate(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

      self.evaluate(tf.global_variables_initializer())
      actual_decode = self.evaluate(decode)

    self.assertTupleEqual(
        (src_batch, p.beam_search.num_hyps_per_beam),
        actual_decode.topk_hyps.shape)
    self.assertTupleEqual(
        (src_batch * p.beam_search.num_hyps_per_beam, src_time),
        actual_decode.topk_ids.shape)
    self.assertTupleEqual(
        (src_batch * p.beam_search.num_hyps_per_beam,),
        actual_decode.topk_lens.shape)
    self.assertTupleEqual(
        (src_batch, p.beam_search.num_hyps_per_beam),
        actual_decode.topk_scores.shape)

    expected_topk_ids = [[2, 0, 0, 0, 0], [13, 2, 0, 0, 0], [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]

    expected_topk_lens = [1, 2, 0, 0]
    expected_topk_scores = [[-3.783162, -5.767723], [0., 0.]]

    self.assertAllEqual(expected_topk_ids, actual_decode.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decode.topk_lens)
    self.assertAllClose(expected_topk_scores, actual_decode.topk_scores)

  def testBeamSearchDecodeTgtPrefix(self, dtype=tf.float32):
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      src_batch = 2
      p = self._DecoderParams(dtype=dtype)
      p.init_step_ids = True  # initializes beam search with predefined ids.
      p.beam_search.num_hyps_per_beam = 2
      p.rnn_cell_dim = 32
      dec = decoder.MTDecoderV1(p)
      encoder_outputs, _ = self._Inputs(dtype=dtype, init_step_ids=True)
      decode = dec.BeamSearchDecode(encoder_outputs)
      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # self.evaluate(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))
      self.evaluate(tf.global_variables_initializer())
      actual_decode = self.evaluate(decode)

    num_hyps = src_batch * p.beam_search.num_hyps_per_beam
    self.assertTupleEqual((src_batch, p.beam_search.num_hyps_per_beam),
                          actual_decode.topk_hyps.shape)
    self.assertTupleEqual((num_hyps, p.target_seq_len),
                          actual_decode.topk_ids.shape)
    self.assertTupleEqual((num_hyps,), actual_decode.topk_lens.shape)
    self.assertTupleEqual((src_batch, p.beam_search.num_hyps_per_beam),
                          actual_decode.topk_scores.shape)

    expected_topk_ids = [[2, 0, 0, 0, 0], [13, 2, 0, 0, 0], [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]
    expected_topk_lens = [1, 2, 0, 0]
    expected_topk_scores = [[-3.783162, -5.767723], [0., 0.]]
    self.assertAllEqual(expected_topk_ids, actual_decode.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decode.topk_lens)
    self.assertAllClose(expected_topk_scores, actual_decode.topk_scores)

  @parameterized.named_parameters(
      ('Bias0ConsistentFalse', 0., False),
      ('Bias0ConsistentTrue', 0., True),
      ('Bias1ConsistentFalse', 1., False),
      ('Bias1ConsistentTrue', 1., True),
  )
  def testBeamSearchDecodeBiased(self, bias, bias_only_if_consistent):
    dtype = tf.float32
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      src_batch = 2
      p = self._DecoderParams(dtype=dtype)
      p.bias_only_if_consistent = bias_only_if_consistent
      p.target_seq_len = 6
      p.beam_search.num_hyps_per_beam = 2
      p.rnn_cell_dim = 32
      dec = p.Instantiate()
      encoder_outputs, _ = self._Inputs(dtype=dtype)
      encoder_outputs['targets'] = py_utils.NestedMap(
          labels=tf.constant([[1, 3, 0, 0], [3, 4, 5, 2]]),
          paddings=tf.constant([[0, 0, 1, 1], [0, 0, 0, 0]], dtype=dtype))
      encoder_outputs['targets']['weights'] = tf.fill(
          tf.shape(encoder_outputs.targets.labels), bias)
      decode = dec.BeamSearchDecodeBiased(encoder_outputs)

      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # self.evaluate(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

      self.evaluate(tf.global_variables_initializer())
      actual_decode = self.evaluate(decode)

    num_hyps = src_batch * p.beam_search.num_hyps_per_beam
    self.assertTupleEqual((src_batch, p.beam_search.num_hyps_per_beam),
                          actual_decode.topk_hyps.shape)
    self.assertTupleEqual((num_hyps, p.target_seq_len),
                          actual_decode.topk_ids.shape)
    self.assertTupleEqual((num_hyps,), actual_decode.topk_lens.shape)
    self.assertTupleEqual((src_batch, p.beam_search.num_hyps_per_beam),
                          actual_decode.topk_scores.shape)

    if bias == 0:
      expected_topk_ids = [[2, 0, 0, 0, 0, 0], [13, 2, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
      expected_topk_lens = [1, 2, 0, 0]
      expected_topk_scores = [[-3.783162, -5.767723], [0., 0.]]
    elif bias == 1 and bias_only_if_consistent:
      expected_topk_ids = [[1, 3, 2, 0, 0, 0], [1, 3, 13, 2, 0, 0],
                           [3, 4, 5, 2, 0, 0], [0, 0, 0, 0, 0, 0]]
      expected_topk_lens = [3, 4, 4, 0]
      expected_topk_scores = [[-3.073836, -5.474799], [-0.415888, 0.]]
    elif bias == 1 and (not bias_only_if_consistent):
      expected_topk_ids = [[1, 3, 2, 0, 0, 0], [1, 3, 13, 2, 0, 0],
                           [3, 4, 5, 2, 0, 0], [3, 4, 0, 2, 0, 0]]
      expected_topk_lens = [3, 4, 4, 4]
      expected_topk_scores = [[-3.073837, -5.474799], [-0.415888, -24.98234]]

    self.assertAllEqual(expected_topk_ids, actual_decode.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decode.topk_lens)
    self.assertAllClose(
        expected_topk_scores, actual_decode.topk_scores, rtol=1e-1)

  def testBeamSearchDecodeUseZeroAttenState(self, dtype=tf.float32):
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      src_batch = 2
      p = self._DecoderParams(dtype=dtype)
      src_time = p.target_seq_len
      p.beam_search.num_hyps_per_beam = 2
      p.use_zero_atten_state = True
      p.rnn_cell_dim = 32
      dec = decoder.MTDecoderV1(p)
      encoder_outputs, _ = self._Inputs(dtype=dtype)
      decode = dec.BeamSearchDecode(encoder_outputs)
      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # self.evaluate(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

      self.evaluate(tf.global_variables_initializer())
      actual_decode = self.evaluate(decode)

    self.assertTupleEqual((src_batch, p.beam_search.num_hyps_per_beam),
                          actual_decode.topk_hyps.shape)
    self.assertTupleEqual(
        (src_batch * p.beam_search.num_hyps_per_beam, src_time),
        actual_decode.topk_ids.shape)
    self.assertTupleEqual((src_batch * p.beam_search.num_hyps_per_beam,),
                          actual_decode.topk_lens.shape)
    self.assertTupleEqual((src_batch, p.beam_search.num_hyps_per_beam),
                          actual_decode.topk_scores.shape)

    expected_topk_ids = [[2, 0, 0, 0, 0], [13, 2, 0, 0, 0], [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]

    expected_topk_lens = [1, 2, 0, 0]
    expected_topk_scores = [[-3.783176, -5.767704], [0., 0.]]

    self.assertAllEqual(expected_topk_ids, actual_decode.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decode.topk_lens)
    self.assertAllClose(expected_topk_scores, actual_decode.topk_scores)

  def testBeamSearchDecodeFeedingAttContext(self, dtype=tf.float32):
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      src_batch = 2
      p = self._DecoderParams(dtype=dtype)
      src_time = p.target_seq_len
      p.beam_search.num_hyps_per_beam = 2
      p.rnn_cell_dim = 32
      p.feed_attention_context_vec_to_softmax = True
      dec = decoder.MTDecoderV1(p)
      encoder_outputs, _ = self._Inputs(dtype=dtype)
      decode = dec.BeamSearchDecode(encoder_outputs)
      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # self.evaluate(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))
      self.evaluate(tf.global_variables_initializer())
      actual_decode_feeding_att_context = self.evaluate(decode)

    self.assertTupleEqual(
        (src_batch, p.beam_search.num_hyps_per_beam),
        actual_decode_feeding_att_context.topk_hyps.shape)
    self.assertTupleEqual(
        (src_batch * p.beam_search.num_hyps_per_beam, src_time),
        actual_decode_feeding_att_context.topk_ids.shape)
    self.assertTupleEqual(
        (src_batch * p.beam_search.num_hyps_per_beam,),
        actual_decode_feeding_att_context.topk_lens.shape)
    self.assertTupleEqual(
        (src_batch, p.beam_search.num_hyps_per_beam),
        actual_decode_feeding_att_context.topk_scores.shape)

    expected_topk_ids = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [2, 0, 0, 0, 0],
                         [8, 2, 0, 0, 0]]

    expected_topk_lens = [0, 0, 1, 2]
    expected_topk_scores = [[0., 0.], [-3.292501, -5.533068]]

    self.assertAllEqual(expected_topk_ids,
                        actual_decode_feeding_att_context.topk_ids)
    self.assertAllEqual(expected_topk_lens,
                        actual_decode_feeding_att_context.topk_lens)
    self.assertAllClose(expected_topk_scores,
                        actual_decode_feeding_att_context.topk_scores)

  def testBeamSearchDisallowMisalignment(self):
    """Test p.force_alignment behaves correctly."""
    with self.session(use_gpu=True), self.SetEval(True), tf.variable_scope(
        'test', reuse=tf.AUTO_REUSE):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=tf.float32)
      p.rnn_cell_dim = 32
      # We reduce the vocab size so that the decoding has better chance of
      # success (i.e. hitting the boundary token and eos).
      p.emb.vocab_size = 6
      p.softmax.num_classes = 6
      p.beam_search.num_hyps_per_beam = 3
      p.target_seq_len = 5
      p.sentence_boundary_token_id = 1
      dec = p.Instantiate()
      p_disallow = p.Copy()
      p_disallow.force_alignment = True
      dec_disallow = p_disallow.Instantiate()

      encoder_outputs, _ = self._Inputs(dtype=tf.float32)
      # We regenerate the encoded input so that the output with the vanilla
      # decoder is regular sequences. If we get unlucky top_ids with the
      # base case can be all zeros, defeating this test. If this breaks,
      # e.g. due to _Inputs() changing, try adding a np seed here or just
      # hard code this random 'encoded'.
      encoder_outputs.encoded = tf.constant(
          np.random.uniform(size=[5, 2, 4]), dtype=tf.float32)
      # src_batch_size = 2, we require the output to have 2 and 3 sentences
      # respectively.
      encoder_outputs['num_sentences'] = tf.constant([2, 3], dtype=tf.int32)
      decode = dec.BeamSearchDecode(encoder_outputs)
      decode_disallow = dec_disallow.BeamSearchDecode(encoder_outputs)

      # A second instance of decoding with disallow enabled, but all
      # inputs are single sentences.
      encoder_outputs['num_sentences'] = tf.constant([1, 1], dtype=tf.int32)
      decode_disallow2 = dec_disallow.BeamSearchDecode(encoder_outputs)

      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # self.evaluate(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))
      decode_disallow = decode_disallow._replace(
          topk_decoded=tf.constant(0, tf.float32))
      decode_disallow2 = decode_disallow2._replace(
          topk_decoded=tf.constant(0, tf.float32))

      self.evaluate(tf.global_variables_initializer())
      actual_decoded = self.evaluate(decode)
      actual_decoded_disallow = self.evaluate(decode_disallow)
      actual_decoded_disallow2 = self.evaluate(decode_disallow2)

    # normal decoder outputs regular sequences.
    expected_topk_ids = [[2, 0, 0, 0, 0], [1, 2, 0, 0, 0], [5, 2, 0, 0, 0],
                         [2, 0, 0, 0, 0], [1, 2, 0, 0, 0], [5, 2, 0, 0, 0]]
    expected_topk_lens = [1, 2, 2, 1, 2, 2]
    self.assertAllEqual(expected_topk_ids, actual_decoded.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decoded.topk_lens)

    # With disallow misalignment enabled, decoder outputs contains at least
    # 2 and 3 sentences respectively, for all hypothesis.
    expected_topk_ids2 = [[1, 2, 0, 0, 0], [5, 1, 2, 0, 0], [3, 1, 2, 0, 0],
                          [1, 1, 2, 0, 0], [5, 1, 1, 2, 0], [3, 1, 1, 2, 0]]
    expected_topk_lens2 = [2, 3, 3, 3, 4, 4]
    self.assertAllEqual(expected_topk_ids2, actual_decoded_disallow.topk_ids)
    self.assertAllEqual(expected_topk_lens2, actual_decoded_disallow.topk_lens)

    # For single sentence inputs with disallow enabled decoding, the output
    # never contains the boundary id 1.
    expected_topk_ids3 = [[2, 0, 0, 0, 0], [5, 2, 0, 0, 0], [3, 2, 0, 0, 0],
                          [2, 0, 0, 0, 0], [5, 2, 0, 0, 0], [3, 2, 0, 0, 0]]
    self.assertAllEqual(expected_topk_ids3, actual_decoded_disallow2.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decoded_disallow2.topk_lens)

  def testBeamSearchDecodeFactorizedEmbeddingEnabled(self, dtype=tf.float32):
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      proj_tpl = layers.ProjectionLayer.Params().Copy()
      proj_tpl.batch_norm = False
      proj_tpl.activation = 'NONE'
      proj_tpl.has_bias = True
      proj_tpl.params_init = py_utils.WeightInit.Uniform(0.04, 1234)
      p.emb_projection_tpl = proj_tpl
      p.beam_search.num_hyps_per_beam = 2
      p.emb.embedding_dim = 4
      p.rnn_cell_dim = 32
      p.softmax = layers.SharedSoftmaxLayer.Params().Set(
          softmax=layers.SimpleFullSoftmax.Params().Set(
              num_shards=p.softmax.num_shards),
          num_classes=p.softmax.num_classes,
          params_init=p.softmax.params_init.Copy(),
          embedding_dim=p.emb.embedding_dim,
          vocab_size=p.softmax.num_classes)
      dec = decoder.MTDecoderV1(p)
      encoder_outputs, _ = self._Inputs(dtype=dtype)
      decode = dec.BeamSearchDecode(encoder_outputs)
      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # self.evaluate(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

      self.evaluate(tf.global_variables_initializer())
      actual_decode = self.evaluate(decode)

    expected_topk_ids = [[2, 0, 0, 0, 0], [8, 2, 0, 0, 0], [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]

    expected_topk_lens = [1, 2, 0, 0]
    expected_topk_scores = [[-3.785752, -5.774518], [0., 0.]]

    self.assertAllEqual(expected_topk_ids, actual_decode.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decode.topk_lens)
    self.assertAllClose(expected_topk_scores, actual_decode.topk_scores)

  def testSingleTokenFastDecode(self):
    """Test p.single_token_fast_decode."""
    # We randomly select a subset of rows to "pad" with single token inputs.
    # And for the test we assert that only the padded rows output single
    # token output hyps.
    batch_size = 6
    single_token_rows = []
    for i in range(batch_size):
      if random.randint(0, 1):
        single_token_rows.append(i)
    with self.session(use_gpu=True), self.SetEval(True), tf.variable_scope(
        'test', reuse=tf.AUTO_REUSE):
      p = self._DecoderParams(dtype=tf.float32)
      p.emb.vocab_size = 5
      p.softmax.num_classes = 5
      p.beam_search.num_hyps_per_beam = 4
      p.target_seq_len = 8
      p.single_token_fast_decode = True
      dec = p.Instantiate()

      encoder_outputs = py_utils.NestedMap()
      encoder_outputs.encoded = tf.constant(
          np.random.normal(size=[5, batch_size, 4]), dtype=tf.float32)
      paddings = np.zeros([5, batch_size], dtype=np.float32)
      for i in single_token_rows:
        paddings[1:, i] = 1.0
      encoder_outputs.padding = tf.constant(paddings, dtype=tf.float32)
      decode = dec.BeamSearchDecode(encoder_outputs)

      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # self.evaluate(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))
      self.evaluate(tf.global_variables_initializer())
      actual_decoded = self.evaluate(decode)

    # For rows contain single token inputs only.
    expected_eos_ids = np.zeros(
        [p.beam_search.num_hyps_per_beam, p.target_seq_len], dtype=np.int32)
    expected_eos_ids[0, 0] = 2  # first hyp is EOS
    expected_eos_lens = np.zeros([p.beam_search.num_hyps_per_beam],
                                 dtype=np.int32)
    expected_eos_lens[0] = 1
    for i in range(batch_size):
      # We assert that only padded rows return one hyp with EOS.
      if i in single_token_rows:
        fn = self.assertAllEqual
      else:
        fn = self.assertNotAllEqual

      fn(
          expected_eos_ids,
          actual_decoded.topk_ids[i * p.beam_search.num_hyps_per_beam:(i + 1) *
                                  p.beam_search.num_hyps_per_beam, :])
      fn(
          expected_eos_lens,
          actual_decoded.topk_lens[i * p.beam_search.num_hyps_per_beam:(i + 1) *
                                   p.beam_search.num_hyps_per_beam])

  def testSampleTargetSequences(self, dtype=tf.float32):
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      src_batch = 2
      p = self._DecoderParams(dtype=dtype)
      if p.cls != decoder.MTDecoderV1:
        tf.logging.info('Skipping testSampleTargetSequences for %s', p.cls)
        return
      p.rnn_cell_dim = 32
      dec = p.Instantiate()
      encoder_outputs, _ = self._Inputs(dtype=dtype)
      sample = dec.SampleTargetSequences(
          dec.theta,
          encoder_outputs,
          random_seed=tf.constant(1, dtype=tf.int32))
      self.evaluate(tf.global_variables_initializer())
      actual_sample = self.evaluate(sample)

    self.assertTupleEqual((src_batch, p.target_seq_len),
                          actual_sample.ids.shape)
    self.assertTupleEqual((src_batch, p.target_seq_len),
                          actual_sample.paddings.shape)

    expected_ids = [[0, 12, 12, 13, 5], [12, 10, 15, 1, 2]]
    self.assertAllEqual(expected_ids, actual_sample.ids)


class TransformerDecoderTestCaseBase(test_utils.TestCase):

  def _DecoderParams(self,
                     per_word_avg_loss=False,
                     is_transparent=False,
                     dtype=tf.float32,
                     fprop_dtype=None,
                     use_task_emb=False,
                     init_step_ids=False):
    p = decoder.TransformerDecoder.Params()
    p.name = 'decoder'
    p.source_dim = 4
    p.model_dim = 4
    p.num_trans_layers = 6
    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    p.token_emb.vn = disable_vn
    p.token_emb.vocab_size = 20
    p.token_emb.embedding_dim = 4
    p.token_emb.max_num_shards = 1
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim(seed=12345)
    p.position_emb.embedding_dim = 4
    if use_task_emb:
      p.task_emb = p.token_emb.Copy()
      p.task_emb.vocab_size = 4
    p.trans_tpl.vn = disable_vn
    p.init_step_ids = init_step_ids
    p.trans_tpl.source_dim = 4
    p.trans_tpl.tr_atten_tpl.source_dim = 4
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
    p.trans_tpl.tr_fflayer_tpl.input_dim = 4
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 8
    p.label_smoothing = layers.LocalizedLabelSmoother.Params()
    p.label_smoothing.offsets = [-2, -1, 1, 2]
    p.label_smoothing.weights = [0.015, 0.035, 0.035, 0.015]
    p.softmax.vn = disable_vn
    p.softmax.num_classes = 20
    p.softmax.num_shards = 1
    p.per_word_avg_loss = per_word_avg_loss
    p.random_seed = 1234
    p.dtype = dtype
    p.target_seq_len = 5
    p.is_transparent = is_transparent

    for lp in base_layer.RecursiveFindLayerParams(p):
      lp.dtype = dtype

    py_utils.UpdateFpropDtype(p, fprop_dtype)

    return p

  def _Inputs(self, dtype=tf.float32, has_task_ids=False, init_step_ids=False):
    np.random.seed(_NUMPY_RANDOM_SEED)
    src_time = 5
    src_batch = 4
    num_hyps = 2
    emb_dims = 4
    src_enc = tf.constant(
        np.random.normal(size=[src_time, src_batch, emb_dims]), dtype=dtype)
    src_paddings = tf.zeros([src_time, src_batch], dtype=dtype)
    tgt_time = 5
    tgt_batch = src_batch * num_hyps
    self.tgt_batch = tgt_batch

    tgt_ids = tf.constant(
        np.random.randint(20, size=[tgt_batch, tgt_time]), dtype=tf.int32)
    tgt_labels = tf.constant(
        np.random.randint(20, size=[tgt_batch, tgt_time]), dtype=tf.int32)
    tgt_paddings = tf.zeros([tgt_batch, tgt_time], dtype=dtype)
    tgt_weights = 1.0 - tgt_paddings
    tgts = py_utils.NestedMap({
        'ids': tgt_ids,
        'labels': tgt_labels,
        'weights': tgt_weights,
        'paddings': tgt_paddings
    })
    encoder_outputs = py_utils.NestedMap(
        encoded=src_enc, padding=src_paddings, segment_id=None)

    if has_task_ids:
      task_ids = tf.constant(
          np.random.randint(4, size=[src_batch]), dtype=tf.int32)
      tgts['task_ids'] = tf.tile(
          tf.expand_dims(tf.tile(task_ids, [num_hyps]), 1), [1, tgt_time])
      encoder_outputs['target_task_ids'] = task_ids
    if init_step_ids:
      tgt_prefix = tf.constant(
          np.random.randint(4, size=[src_batch]), dtype=tf.int32)
      encoder_outputs['init_step_ids'] = tgt_prefix
    return (encoder_outputs, tgts, num_hyps)

  def _InputsForAttentionTest(self, dtype=tf.float32, has_task_ids=False):
    np.random.seed(_NUMPY_RANDOM_SEED)
    src_time = 5
    src_batch = 2
    num_hyps = 2
    emb_dims = 4
    src_enc = tf.constant(
        np.random.normal(size=[src_time, src_batch, emb_dims]), dtype=dtype)
    src_paddings = tf.constant(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=dtype)
    tgt_time = 5
    tgt_batch = src_batch * num_hyps
    self.tgt_batch = tgt_batch

    tgt_ids = tf.constant(
        np.random.randint(20, size=[tgt_batch, tgt_time]), dtype=tf.int32)
    tgt_labels = tf.constant(
        np.random.randint(20, size=[tgt_batch, tgt_time]), dtype=tf.int32)
    tgt_paddings = tf.zeros([tgt_batch, tgt_time], dtype=dtype)
    tgt_weights = 1.0 - tgt_paddings
    tgts = py_utils.NestedMap({
        'ids': tgt_ids,
        'labels': tgt_labels,
        'weights': tgt_weights,
        'paddings': tgt_paddings
    })
    encoder_outputs = py_utils.NestedMap(
        encoded=src_enc, padding=src_paddings, segment_id=None)

    if has_task_ids:
      task_ids = tf.constant(
          np.random.randint(4, size=[src_batch]), dtype=tf.int32)
      tgts['task_ids'] = tf.tile(
          tf.expand_dims(tf.tile(task_ids, [num_hyps]), 1), [1, tgt_time])
      encoder_outputs['target_task_ids'] = task_ids

    return (encoder_outputs, tgts, num_hyps)


class TransformerDecoderTest(TransformerDecoderTestCaseBase):

  def testDecoderConstruction(self):
    p = self._DecoderParams()
    dec = decoder.TransformerDecoder(p)
    self.assertIsInstance(dec, p.cls)

  def testDecoderWithNgramMaskConstruction(self):
    p = self._DecoderParams()
    # Turn on N-gram masking in the TransformerLayer.
    # Before doing so though copy the self-attention params to avoid
    # the auxilliary attention being masked as well.
    p.trans_tpl.tr_aux_atten_tpl = p.trans_tpl.tr_atten_tpl.Copy()
    p.trans_tpl.tr_atten_tpl.is_masked = True
    p.trans_tpl.tr_atten_tpl.mask_ngram_order = 3
    p.trans_tpl.tr_atten_tpl.mask_type = 'ngram'
    dec = decoder.TransformerDecoder(p)
    self.assertIsInstance(dec, p.cls)

  def testDecoderConstructionWithTplList(self):
    p = self._DecoderParams()
    p.trans_tpl = [p.trans_tpl.Copy(), p.trans_tpl.Copy()]
    dec = decoder.TransformerDecoder(p)
    self.assertIsInstance(dec, p.cls)

  def testTransparentDecoderConstruction(self):
    p = self._DecoderParams(is_transparent=True)
    dec = decoder.TransformerDecoder(p)
    self.assertIsInstance(dec, p.cls)

  def _testPackedInputs(self, dtype=tf.float32):
    p = self._DecoderParams()
    np.random.seed(_NUMPY_RANDOM_SEED)
    src_time = 5
    batch = 2
    emb_dims = 4
    tgt_time = 5
    src_enc = tf.constant(
        np.random.normal(size=[src_time, batch, p.source_dim]), dtype=dtype)
    paddings = tf.zeros([src_time, batch], dtype=dtype)
    tgt_ids = tf.constant(
        np.random.randint(20, size=[batch, tgt_time]), dtype=tf.int32)
    tgt_labels = tf.constant(
        np.random.randint(20, size=[batch, tgt_time]), dtype=tf.int32)
    tgt_paddings = tf.zeros([batch, tgt_time], dtype=dtype)
    tgt_weights = 1.0 - tgt_paddings
    tgts = py_utils.NestedMap({
        'ids': tgt_ids,
        'labels': tgt_labels,
        'weights': tgt_weights,
        'paddings': tgt_paddings
    })

    src_enc_packed = tf.transpose(src_enc, [1, 0, 2])
    src_enc_packed = tf.reshape(src_enc_packed, [-1, 1, emb_dims])
    src_enc_padding_packed = tf.reshape(paddings, [-1, 1])
    target_packed = py_utils.NestedMap({
        'ids': tf.reshape(tgts.ids, [1, -1]),
        'labels': tf.reshape(tgts.labels, [1, -1]),
        'weights': tf.reshape(tgts.weights, [1, -1]),
        'paddings': tf.reshape(tgts.paddings, [1, -1])
    })
    src_segment_id = tf.transpose(
        tf.constant(
            np.asarray([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]), dtype=tf.float32))
    target_packed.segment_ids = tf.constant(
        np.asarray([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]), dtype=tf.float32)
    target_packed.segment_pos = tf.constant(
        np.asarray([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]))
    return (src_enc, paddings, tgts, src_enc_packed, src_enc_padding_packed,
            src_segment_id, target_packed)

  def _testTransparentInputs(self,
                             num_layers=6,
                             dtype=tf.float32,
                             is_eval_mode=False):
    src_time = 5
    src_batch = 4
    emb_dims = 4
    encoder_outputs, tgts, num_hyps = self._Inputs(dtype)
    src_enc = tf.constant(
        np.random.normal(size=[src_time, src_batch, emb_dims, num_layers]),
        dtype=dtype)
    if not is_eval_mode:
      src_enc = tf.unstack(src_enc, axis=3)
    encoder_outputs.encoded = src_enc
    return (encoder_outputs, tgts, num_hyps)

  def testDecoderFPropWithPacking(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        tf.random.set_seed(_TF_RANDOM_SEED)
        p = self._DecoderParams(per_word_avg_loss=True, dtype=dtype)
        # Localized label smoother messes up the loss with packing.
        p.label_smoothing = None
        dec = decoder.TransformerDecoder(p)
        p_packed = p.Copy()
        p_packed.packed_input = True
        dec_packed = decoder.TransformerDecoder(p_packed)

        (src_enc, paddings, tgts, src_enc_packed, src_enc_padding_packed,
         src_segment_id, target_packed) = self._testPackedInputs()
        encoder_outputs = py_utils.NestedMap(
            encoded=src_enc, padding=paddings, segment_id=None)
        loss, _ = dec.FProp(dec.theta, encoder_outputs, tgts).metrics['loss']
        encoder_outputs_packed = py_utils.NestedMap(
            encoded=src_enc_packed,
            padding=src_enc_padding_packed,
            segment_id=src_segment_id)
        loss_packed, _ = dec_packed.FProp(dec_packed.theta,
                                          encoder_outputs_packed,
                                          target_packed).metrics['loss']
        self.evaluate(tf.global_variables_initializer())
        actual_loss, packed_loss = self.evaluate([loss, loss_packed])
        self.assertAlmostEqual(
            np.float32(packed_loss), np.float32(actual_loss), delta=1e-4)

  def testTransparentDecoderFProp(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(is_transparent=True, dtype=dtype)
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, _ = self._testTransparentInputs(dtype=dtype)
      loss, _ = dec.FPropDefaultTheta(encoder_outputs, targets).metrics['loss']
      self.evaluate(tf.global_variables_initializer())
      actual_loss = loss.eval()
      print('actual loss = ', actual_loss)
      CompareToGoldenSingleFloat(self, 19.725393, actual_loss)

  def test_ExpandToNumHyps(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(is_transparent=True, dtype=dtype)
      dec = decoder.TransformerDecoder(p)

      src_enc_len = tf.constant([3, 2, 1])
      num_hyps = 2
      expected = tf.constant([3, 2, 1, 3, 2, 1])
      expanded = dec._ExpandToNumHyps(src_enc_len, num_hyps)
      expanded_v, expected_v = self.evaluate([expanded, expected])
      self.assertAllEqual(expanded_v, expected_v)

  def test_RemoveEOSProbs(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(is_transparent=True, dtype=dtype)
      dec = decoder.TransformerDecoder(p)

      src_enc_len = tf.constant([5, 3, 5])

      # [batch, target_len, source_len]
      probs = tf.constant([[[0.2, 0.2, 0.2, 0.2, 0.2]],
                           [[0.2, 0.3, 0.5, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0, 0.0, 1.0]]])
      new_probs = dec._RemoveEOSProbs(p, probs, src_enc_len)
      new_probs_v = self.evaluate([new_probs])

      expected_probs = tf.constant([[[0.25, 0.25, 0.25, 0.25, 0.0]],
                                    [[0.4, 0.6, 0.0, 0.0, 0.0]],
                                    [[0.0, 0.0, 0.0, 0.0, 0.0]]])

      new_probs_v, expected_probs_v = self.evaluate([new_probs, expected_probs])

      self.assertAllClose(expected_probs_v, new_probs_v)

  def testDecoderFPropWithTaskEmb(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype, use_task_emb=True)
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, _ = self._Inputs(dtype=dtype, has_task_ids=True)
      loss, _ = dec.FPropDefaultTheta(encoder_outputs, targets).metrics['loss']
      self.evaluate(tf.global_variables_initializer())
      actual_loss = loss.eval()
      CompareToGoldenSingleFloat(self, 18.374338, actual_loss)

  def testDecoderFPropWithLangDepAtten(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype, use_task_emb=True)
      # 4 tasks, 2 languages.
      p.use_lang_dependent_atten = True
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, _ = self._Inputs(dtype=dtype, has_task_ids=True)
      loss, _ = dec.FPropDefaultTheta(encoder_outputs, targets).metrics['loss']
      self.evaluate(tf.global_variables_initializer())
      actual_loss = loss.eval()
      CompareToGoldenSingleFloat(self, 16.200066, actual_loss)

  def testDecoderFPropWithContext(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        tf.random.set_seed(_TF_RANDOM_SEED)
        p = self._DecoderParams(per_word_avg_loss=True, dtype=dtype)
        p.trans_tpl = layers_with_attention.TransformerWithContextLayer.Params()
        p.trans_tpl.source_dim = 4
        p.trans_tpl.tr_atten_tpl.source_dim = 4
        p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
        p.trans_tpl.tr_fflayer_tpl.input_dim = 4
        p.trans_tpl.tr_fflayer_tpl.hidden_dim = 8
        dec = p.Instantiate()

        enc_outputs, targets, _ = self._Inputs()
        encoder_outputs = py_utils.NestedMap(
            encoded=enc_outputs.encoded,
            padding=enc_outputs.padding,
            context_encoded=enc_outputs.encoded,
            context_padding=enc_outputs.padding)
        loss, _ = dec.FProp(dec.theta, encoder_outputs, targets).metrics['loss']
        self.evaluate(tf.global_variables_initializer())
        CompareToGoldenSingleFloat(self, 3.816574, loss.eval())

  def testDecoderFPropWithZeroFirstStep(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      p.zero_token_embs_first_time_step = True
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, _ = self._Inputs(dtype=dtype, has_task_ids=True)
      loss, _ = dec.FPropDefaultTheta(encoder_outputs, targets).metrics['loss']
      self.evaluate(tf.global_variables_initializer())
      actual_loss = loss.eval()
      CompareToGoldenSingleFloat(self, 21.425932, actual_loss)

  def _testExtendStep(self, sess, dec, encoder_outputs, tgts, num_hyps):
    p = self._DecoderParams()

    # Infer true source encoder length from the padding.
    src_enc_len = tf.reduce_sum(1 - encoder_outputs.padding, axis=0)
    src_enc_len = dec._ExpandToNumHyps(src_enc_len, num_hyps)

    # Run Fprop
    fprop_out = dec._FProp(dec.theta, encoder_outputs, tgts)
    l_out1 = fprop_out.softmax_input
    attention_map_fprop = fprop_out.attention

    # run ExtendStep
    prefix_states = py_utils.NestedMap()
    for i in range(6):
      layer_i_states = py_utils.NestedMap()
      # The first dim is for the decode step (sequence length).
      # Here's 0 as placeholder
      layer_i_states.key = tf.zeros([0, self.tgt_batch, p.model_dim])
      layer_i_states.value = tf.zeros([0, self.tgt_batch, p.model_dim])
      prefix_states['layer_%i' % i] = layer_i_states

    l_out2 = []
    per_step_atten_probs = []
    for i in range(5):
      l_i_out, prefix_states, atten_probs = dec.ExtendStep(
          dec.theta, encoder_outputs, tgts.ids[:, i], i, prefix_states)
      l_out2.append(l_i_out)
      per_step_atten_probs.append(atten_probs)
    l_out2 = tf.stack(l_out2)
    bs_atten_probs = tf.stack(per_step_atten_probs)

    attention_map_bs = py_utils.NestedMap(probs=bs_atten_probs)

    def _TransposeAttentions(x):
      return tf.transpose(x, [1, 0, 2])

    attention_map_bs = attention_map_bs.Transform(_TransposeAttentions)

    self.evaluate(tf.global_variables_initializer())

    l_out1_v, l_out2_v, attention_map_fprop_v, attention_map_bs_v, src_enc_len_v = self.evaluate(
        [l_out1, l_out2, attention_map_fprop, attention_map_bs, src_enc_len])

    # Ensure that FProp and BeamSearch output are the same.
    self.assertAllClose(l_out1_v, l_out2_v, rtol=1e-05, atol=1e-05)

    # Ensure that FProp and BeamSearch attention matrix is the same.
    self.assertAllClose(attention_map_fprop_v.probs, attention_map_bs_v.probs)

    print('attention map', attention_map_fprop_v.probs)

    # End-to-end test attention probs -- ensure EOS symbol and positions
    # behind EOS have 0 probability.
    for i in range(0, len(src_enc_len_v)):
      pos = int(src_enc_len_v[i]) - 1
      self.assertEqual(
          np.count_nonzero(attention_map_fprop_v.probs[i][:, pos:]), 0)

  def testDecoderExtendStep(self, dtype=tf.float32):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, num_hyps = (
          self._InputsForAttentionTest(dtype=dtype))

      self._testExtendStep(sess, dec, encoder_outputs, targets, num_hyps)

  def testDecoderWithNgramMaskExtendStep(self, dtype=tf.float32):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      # Turn on N-gram masking in the TransformerLayer.
      # Before doing so though copy the self-attention params to avoid
      # the auxilliary attention being masked as well.
      p.trans_tpl.tr_aux_atten_tpl = p.trans_tpl.tr_atten_tpl.Copy()
      p.trans_tpl.tr_atten_tpl.is_masked = True
      p.trans_tpl.tr_atten_tpl.mask_ngram_order = 3
      p.trans_tpl.tr_atten_tpl.mask_type = 'ngram'
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, num_hyps = (
          self._InputsForAttentionTest(dtype=dtype))

      self._testExtendStep(sess, dec, encoder_outputs, targets, num_hyps)

  def testDecoderExtendStepWithTaskEmb(self, dtype=tf.float32):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype, use_task_emb=True)
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, num_hyps = (
          self._InputsForAttentionTest(dtype=dtype, has_task_ids=True))

      self._testExtendStep(sess, dec, encoder_outputs, targets, num_hyps)

  def testDecoderExtendStepZeroFirstTimeStep(self, dtype=tf.float32):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      p.zero_token_embs_first_time_step = True
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, num_hyps = (
          self._InputsForAttentionTest(dtype=dtype))
      self._testExtendStep(sess, dec, encoder_outputs, targets, num_hyps)

  def testTransparentDecoderExtendStep(self, dtype=tf.float32):
    with self.session(use_gpu=True) as sess, self.SetEval(True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(is_transparent=True, dtype=dtype)
      dec = decoder.TransformerDecoder(p)
      encoder_outputs, targets, num_hyps = self._testTransparentInputs(
          dtype=dtype, is_eval_mode=True)
      self._testExtendStep(sess, dec, encoder_outputs, targets, num_hyps)

  def testDecoderExtendStepWithContext(self, dtype=tf.float32):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      p.trans_tpl = layers_with_attention.TransformerWithContextLayer.Params()
      p.trans_tpl.source_dim = 4
      p.trans_tpl.tr_atten_tpl.source_dim = 4
      p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
      p.trans_tpl.tr_fflayer_tpl.input_dim = 4
      p.trans_tpl.tr_fflayer_tpl.hidden_dim = 8
      dec = p.Instantiate()
      enc_outputs, targets, num_hyps = (
          self._InputsForAttentionTest(dtype=dtype, has_task_ids=True))
      encoder_outputs = py_utils.NestedMap(
          encoded=enc_outputs.encoded,
          padding=enc_outputs.padding,
          context_encoded=enc_outputs.encoded,
          context_padding=enc_outputs.padding)
      self._testExtendStep(sess, dec, encoder_outputs, targets, num_hyps)

  def testDecoderFPropSplitBatch(self, dtype=tf.float32):
    with self.session(use_gpu=True):
      tf.random.set_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      dec = decoder.TransformerDecoder(p)

      encoder_outputs, targets, _ = self._Inputs(dtype=dtype)
      src_enc1, src_enc2 = tf.split(encoder_outputs.encoded, 2, 1)
      src_paddings1, src_paddings2 = tf.split(encoder_outputs.padding, 2, 1)

      # source idx <-> target idx:
      # 0 <-> (0, 4), 1 <-> (1, 5), 2 <-> (2, 6), 3 <-> (3, 7)
      tgts = ig_helper.SplitDictOfTensors(targets, 4)
      targets1 = py_utils.NestedMap({
          'ids': tf.concat([tgts[0]['ids'], tgts[2]['ids']], 0),
          'labels': tf.concat([tgts[0]['labels'], tgts[2]['labels']], 0),
          'weights': tf.concat([tgts[0]['weights'], tgts[2]['weights']], 0),
          'paddings': tf.concat([tgts[0]['paddings'], tgts[2]['paddings']], 0)
      })
      targets2 = py_utils.NestedMap({
          'ids': tf.concat([tgts[1]['ids'], tgts[3]['ids']], 0),
          'labels': tf.concat([tgts[1]['labels'], tgts[3]['labels']], 0),
          'weights': tf.concat([tgts[1]['weights'], tgts[3]['weights']], 0),
          'paddings': tf.concat([tgts[1]['paddings'], tgts[3]['paddings']], 0)
      })

      loss, _ = dec.FPropDefaultTheta(encoder_outputs, targets).metrics['loss']
      encoder_outputs1 = py_utils.NestedMap(
          encoded=src_enc1, padding=src_paddings1, segment_id=None)
      loss1, _ = dec.FPropDefaultTheta(encoder_outputs1,
                                       targets1).metrics['loss']
      encoder_outputs2 = py_utils.NestedMap(
          encoded=src_enc2, padding=src_paddings2, segment_id=None)
      loss2, _ = dec.FPropDefaultTheta(encoder_outputs2,
                                       targets2).metrics['loss']

      self.evaluate(tf.global_variables_initializer())
      actual_loss, actual_loss1, actual_loss2 = self.evaluate(
          [loss, loss1, loss2])
      print('actual loss = ', actual_loss)
      print('actual loss1 = ', actual_loss1)
      print('actual loss2 = ', actual_loss2)
      self.assertAlmostEqual(
          actual_loss, np.mean([actual_loss1, actual_loss2]), delta=0.0001)

  def _testBeamSearch(self,
                      expected_values,
                      dtype=tf.float32,
                      init_step_ids=False,
                      has_task_ids=False):
    tf.random.set_seed(_TF_RANDOM_SEED)
    src_batch = 4
    src_time = 5
    p = self._DecoderParams(dtype=dtype, init_step_ids=init_step_ids)
    p.beam_search.num_hyps_per_beam = 2
    p.beam_search.coverage_penalty = 0.0
    p.beam_search.length_normalization = 0
    dec = decoder.TransformerDecoder(p)
    encoder_outputs, _, _ = self._Inputs(
        dtype=dtype, has_task_ids=has_task_ids, init_step_ids=init_step_ids)
    decode = dec.BeamSearchDecode(encoder_outputs)
    # topk_decoded is None in MT decoder, set it to a fake tensor to pass
    # self.evaluate(decode).
    decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

    with self.session(use_gpu=True):
      self.evaluate(tf.global_variables_initializer())
      actual_decode = self.evaluate(decode)

    self.assertTupleEqual(
        (src_batch, p.beam_search.num_hyps_per_beam),
        actual_decode.topk_hyps.shape)
    self.assertTupleEqual(
        (src_batch * p.beam_search.num_hyps_per_beam, src_time),
        actual_decode.topk_ids.shape)
    self.assertTupleEqual(
        (src_batch * p.beam_search.num_hyps_per_beam,),
        actual_decode.topk_lens.shape)
    self.assertTupleEqual(
        (src_batch, p.beam_search.num_hyps_per_beam),
        actual_decode.topk_scores.shape)

    # Assert expected IDs etc
    self.assertAllEqual(expected_values['topk_ids'], actual_decode.topk_ids)
    self.assertAllEqual(expected_values['topk_lens'], actual_decode.topk_lens)
    self.assertAllClose(expected_values['topk_scores'],
                        actual_decode.topk_scores)

    # Assert expected attention probs.
    hypstr = actual_decode.topk_hyps.flatten()[1]
    hyp = Hypothesis()
    hyp.ParseFromString(hypstr)
    print('HYP:', hyp)

    atten_vec_0 = list(np.expand_dims(np.array(hyp.atten_vecs[0].prob), 0)[0])
    atten_vec_1 = list(np.expand_dims(np.array(hyp.atten_vecs[1].prob), 0)[0])

    self.assertAllClose(atten_vec_0, expected_values['atten_vec_0'])
    self.assertAllClose(atten_vec_1, expected_values['atten_vec_1'])

    # Test normalized scores of hypotheses.
    CompareToGoldenSingleFloat(self, expected_values['normalized_score'],
                               hyp.normalized_score)

  def testBeamSearchDecode(self, dtype=tf.float32):
    expected_values = {}
    expected_values['topk_ids'] = [[5, 2, 0, 0, 0], [17, 2, 0, 0, 0],
                                   [5, 2, 0, 0, 0], [17, 3, 2, 0, 0],
                                   [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    expected_values['topk_lens'] = [2, 2, 2, 3, 0, 0, 0, 0]
    expected_values['topk_scores'] = [[-3.821746, -3.980103],
                                      [-3.817123, -4.522634], [0., 0.],
                                      [0., 0.]]

    expected_values['atten_vec_0'] = [
        0.532658, 0.140424, 0.122954, 0.203961, 0.
    ]
    expected_values['atten_vec_1'] = [
        0.067983, 0.532731, 0.284223, 0.115062, 0.
    ]
    expected_values['normalized_score'] = -3.980103

    self._testBeamSearch(
        expected_values=expected_values,
        dtype=dtype,
        init_step_ids=False,
        has_task_ids=False)

  def testBeamSearchDecodeTgtPrefix(self, dtype=tf.float32):
    expected_values = {}
    expected_values['topk_ids'] = [[5, 2, 0, 0, 0], [1, 2, 0, 0, 0],
                                   [5, 2, 0, 0, 0], [5, 3, 2, 0, 0],
                                   [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    expected_values['topk_lens'] = [2, 2, 2, 3, 0, 0, 0, 0]
    expected_values['topk_scores'] = [[-3.837574, -4.1681],
                                      [-3.821537, -5.069403], [0., 0.],
                                      [0., 0.]]

    expected_values['atten_vec_0'] = [0.58009, 0.129324, 0.109171, 0.181414, 0.]
    expected_values['atten_vec_1'] = [
        0.072272, 0.474015, 0.330433, 0.123279, 0.
    ]
    expected_values['normalized_score'] = -4.1681

    self._testBeamSearch(
        expected_values=expected_values,
        dtype=dtype,
        init_step_ids=True,
        has_task_ids=False)

  def _testSampleSequence(self,
                          expected_values,
                          dtype=tf.float32,
                          init_step_ids=False,
                          has_task_ids=False):
    tf.random.set_seed(_TF_RANDOM_SEED)
    src_batch = 4
    src_time = 5
    p = self._DecoderParams(dtype=dtype, init_step_ids=init_step_ids)
    p.beam_search.num_hyps_per_beam = 1
    p.beam_search.coverage_penalty = 0.0
    p.beam_search.length_normalization = 0
    dec = p.Instantiate()
    encoder_outputs, _, _ = self._Inputs(
        dtype=dtype, has_task_ids=has_task_ids, init_step_ids=init_step_ids)
    decode = dec.SampleSequenceDecode(encoder_outputs)

    with self.session(use_gpu=True):
      self.evaluate(tf.global_variables_initializer())
      actual_decode = self.evaluate(decode)

    self.assertTupleEqual((src_batch, p.beam_search.num_hyps_per_beam),
                          actual_decode.topk_hyps.shape)
    self.assertTupleEqual(
        (src_batch * p.beam_search.num_hyps_per_beam, src_time),
        actual_decode.topk_ids.shape)
    self.assertTupleEqual((src_batch * p.beam_search.num_hyps_per_beam,),
                          actual_decode.topk_lens.shape)

    self.assertAllEqual(expected_values['topk_ids'], actual_decode.topk_ids)
    self.assertAllEqual(expected_values['topk_lens'], actual_decode.topk_lens)
    self.assertAllClose(expected_values['topk_scores'],
                        actual_decode.topk_scores)

  def testSampleSequenceDecode(self, dtype=tf.float32):
    expected_values = {}
    expected_values['topk_ids'] = [[4, 3, 12, 7, 11], [17, 7, 2, 2, 2],
                                   [17, 2, 2, 2, 2], [17, 1, 1, 16, 4]]
    expected_values['topk_lens'] = [5, 3, 2, 5]
    expected_values['topk_scores'] = [
        -6.985453, -3.714368, -2.899912, -8.281981
    ]
    self._testSampleSequence(
        expected_values=expected_values,
        dtype=dtype,
        init_step_ids=True,
        has_task_ids=False)


class InsertionDecoderTest(TransformerDecoderTestCaseBase):

  def testDecoderConstruction(self):
    p = decoder.InsertionDecoder.Params()
    p.name = 'insertion_decoder'
    dec = p.Instantiate()
    self.assertIsInstance(dec, decoder.InsertionDecoder)


class TransformerBatchMajorDecoderTest(test_utils.TestCase,
                                       parameterized.TestCase):
  """Test Transformer decoder."""

  def _ConstructTransformerBatchMajorDecoder(self,
                                             dtype=tf.float32,
                                             packed_input=False,
                                             **kwargs):
    p = decoder.TransformerBatchMajorDecoder.Params()
    p.name = 'decoder'
    p.packed_input = packed_input
    p.source_dim = 4
    p.model_dim = 4
    p.num_trans_layers = 6
    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    p.token_emb.vn = disable_vn
    p.token_emb.vocab_size = 20
    p.token_emb.embedding_dim = 4
    p.token_emb.max_num_shards = 1
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim()
    p.position_emb.embedding_dim = 4
    p.trans_decoder_tpl.vn = disable_vn
    p.trans_decoder_tpl.input_dim = 4
    p.trans_decoder_tpl.tr_atten_tpl.input_dim = 4
    p.trans_decoder_tpl.tr_atten_tpl.num_heads = 2
    p.trans_decoder_tpl.tr_fflayer_tpl.input_dim = 4
    p.trans_decoder_tpl.tr_fflayer_tpl.hidden_dim = 8
    p.trans_decoder_tpl.packed_input = packed_input
    p.softmax.vn = disable_vn
    p.softmax.num_classes = 20
    p.softmax.num_shards = 1
    p.per_word_avg_loss = False
    p.random_seed = 12345
    p.target_seq_len = 5
    p.beam_search.num_hyps_per_beam = 2
    p.beam_search.coverage_penalty = 0.0
    p.beam_search.length_normalization = 0
    p.dtype = dtype
    for k, v in kwargs.items():
      setattr(p, k, v)
    for lp in base_layer.RecursiveFindLayerParams(p):
      lp.dtype = dtype
    return decoder.TransformerBatchMajorDecoder(p)

  def _Inputs(self, dtype=tf.float32, packed_input=False):
    np.random.seed(9885784)
    source_time = 5
    batch = 4
    dim = 4
    encoded = tf.constant(
        np.random.normal(size=[source_time, batch, dim]), dtype=dtype)
    padding = tf.zeros([source_time, batch], dtype=dtype)
    target_time = 5
    target_ids = tf.constant(
        np.random.randint(20, size=[batch, target_time]), dtype=tf.int32)
    target_labels = tf.constant(
        np.random.randint(20, size=[batch, target_time]), dtype=tf.int32)
    target_paddings = tf.zeros([batch, target_time], dtype=dtype)
    target_weights = 1.0 - target_paddings
    targets = py_utils.NestedMap({
        'ids': target_ids,
        'labels': target_labels,
        'weights': target_weights,
        'paddings': target_paddings,
        'transcripts': tf.convert_to_tensor(['hello'] * batch),
    })
    encoder_outputs = py_utils.NestedMap(encoded=encoded, padding=padding)

    if packed_input:
      encoder_outputs.segment_id = tf.tile(
          tf.constant(np.asarray([[1, 1, 2, 2, 2]]), dtype=tf.float32),
          [batch, 1])
      encoder_outputs.segment_id = tf.transpose(encoder_outputs.segment_id)
      targets.segment_pos = tf.tile(
          tf.constant(np.asarray([[0, 1, 0, 1, 2]]), dtype=tf.float32),
          [batch, 1])
      targets.segment_ids = tf.tile(
          tf.constant(np.asarray([[1, 1, 2, 2, 2]]), dtype=tf.float32),
          [batch, 1])
    return encoder_outputs, targets

  def testDecoderConstruction(self):
    _ = self._ConstructTransformerBatchMajorDecoder()

  def testDecoderConstructionPackedInput(self):
    self._ConstructTransformerBatchMajorDecoder(packed_input=True)

  @parameterized.named_parameters(('TBC', 'TBC'), ('BTC', 'BTC'))
  def testDecoderFProp(self, prediction_data_format):
    with self.session(use_gpu=True) as sess:
      dec = self._ConstructTransformerBatchMajorDecoder(
          prediction_data_format=prediction_data_format,
          per_example_tensors=True)
      encoder_outputs, targets = self._Inputs()
      dec_out = dec.FPropDefaultTheta(encoder_outputs, targets)
      dec_out = tf.nest.map_structure(tf.convert_to_tensor, dec_out)
      tf.global_variables_initializer().run()
      actual_dec_out = sess.run(dec_out)
      print('actual decoder output =', actual_dec_out)
      self.assertAllClose(27.047781, actual_dec_out.metrics['loss'][0])

  def testDecoderFPropPackedInput(self):
    with self.session(use_gpu=True) as sess:
      dec = self._ConstructTransformerBatchMajorDecoder(packed_input=True)
      encoder_outputs, targets = self._Inputs(packed_input=True)
      loss, _ = dec.FPropDefaultTheta(encoder_outputs, targets).metrics['loss']
      tf.global_variables_initializer().run()
      actual_loss = sess.run(loss)
      print('actual loss = ', actual_loss)
      self.assertAllClose(15.041874, actual_loss)

  def testDecoderExtendStep(self):
    with self.session(use_gpu=True) as sess:
      dec = self._ConstructTransformerBatchMajorDecoder()
      encoder_outputs, targets = self._Inputs()

      layer_out1 = dec._FProp(dec.theta, encoder_outputs, targets)

      prefix_states = py_utils.NestedMap()
      for i in range(6):
        layer_i_states = py_utils.NestedMap()
        layer_i_states.key = tf.zeros([5, 4, 2, 2])
        layer_i_states.value = tf.zeros([5, 4, 2, 2])
        prefix_states['layer_%i' % i] = layer_i_states

      layer_out2 = []
      for i in range(5):
        layer_i_out, prefix_states = dec.ExtendStep(
            dec.theta, encoder_outputs, tf.expand_dims(targets.ids[:, i], 1), i,
            prefix_states)
        layer_out2.append(layer_i_out)
      layer_out2 = tf.stack(layer_out2)

      tf.global_variables_initializer().run()
      actual_layer_out1, actual_layer_out2 = sess.run([layer_out1, layer_out2])
      self.assertAllClose(actual_layer_out1, actual_layer_out2)

  def testBeamSearchDecode(self):
    with self.session(use_gpu=True) as sess:
      dec = self._ConstructTransformerBatchMajorDecoder()
      encoder_outputs, _ = self._Inputs()
      decode = dec.BeamSearchDecode(encoder_outputs)
      # topk_decoded is None in MT decoder, set it to a fake tensor to pass
      # sess.run(decode).
      decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

      tf.global_variables_initializer().run()
      actual_decode = sess.run(decode)

      source_batch = 4
      source_time = 5
      num_hyps_per_beam = 2
      self.assertTupleEqual((source_batch, num_hyps_per_beam),
                            actual_decode.topk_hyps.shape)
      self.assertTupleEqual((source_batch * num_hyps_per_beam, source_time),
                            actual_decode.topk_ids.shape)
      self.assertTupleEqual((source_batch * num_hyps_per_beam,),
                            actual_decode.topk_lens.shape)
      self.assertTupleEqual((source_batch, num_hyps_per_beam),
                            actual_decode.topk_scores.shape)

      expected_topk_ids = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [8, 16, 2, 0, 0],
                           [8, 11, 7, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                           [8, 8, 8, 2, 0], [8, 11, 7, 2, 0]]
      expected_topk_lens = [0, 0, 3, 4, 0, 0, 4, 4]
      expected_topk_scores = [[0., 0.], [-3.154168, -3.2543223], [0., 0.],
                              [-4.8691816, -4.9210157]]
      tf.logging.info(['testBeamSearchDecode actual_decode', actual_decode])

      self.assertAllEqual(expected_topk_ids, actual_decode.topk_ids)
      self.assertAllEqual(expected_topk_lens, actual_decode.topk_lens)
      self.assertAllClose(expected_topk_scores, actual_decode.topk_scores)


class TransformerXDecoderTest(test_utils.TestCase):

  def _DecoderParams(self):
    p = decoder.TransformerXDecoder.Params()
    p.name = 'cross_decoder'
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim()
    p.token_emb.vocab_size = 64
    p.token_emb.embedding_dim = 16
    p.token_emb.max_num_shards = 1
    p.token_emb.scale_sqrt_depth = True
    p.token_emb.vn = py_utils.VariationalNoiseParams(1.0, False, False)
    p.position_emb.embedding_dim = 16
    p.position_emb.trainable_scaling = False
    p.model_dim = 16
    p.source_dim = 16
    p.num_trans_layers = 6
    p.trans_tpl.source_dim = p.model_dim
    p.trans_tpl.tr_atten_tpl.source_dim = p.model_dim
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 2
    p.trans_tpl.tr_atten_tpl.atten_hidden_dim = 16
    p.trans_tpl.tr_atten_tpl.atten_tpl.context_dim = p.model_dim
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 5
    p.trans_tpl.tr_fflayer_tpl.input_dim = p.model_dim
    p.label_smoothing = layers.UniformLabelSmoother.Params()
    p.label_smoothing.uncertainty = 0.1
    p.per_word_avg_loss = True
    p.softmax.num_classes = 64
    p.random_seed = 54321
    return p

  def _Inputs(self, bs, src_seq_len, tgt_seq_len):
    src_enc = tf.constant(
        np.random.normal(size=[src_seq_len, bs, 16]), tf.float32)
    src_enc_padding = tf.zeros([src_seq_len, bs])

    target_labels = tf.constant(
        np.random.randint(
            low=0, high=64, size=[bs, tgt_seq_len], dtype=np.int32))
    target_ids = tf.concat([tf.ones([bs, 1], tf.int32), target_labels],
                           1)[:, :-1]
    paddings = []
    for _ in range(bs):
      zeros_len = np.random.randint(1, tgt_seq_len + 1)
      paddings.append([
          0.,
      ] * zeros_len + [1.] * (tgt_seq_len - zeros_len))
    target_paddings = tf.constant(paddings, tf.float32)
    target_weights = 1.0 - target_paddings
    targets = py_utils.NestedMap({
        'ids': target_ids,
        'labels': target_labels,
        'weights': target_weights,
        'paddings': target_paddings
    })

    encoder_outputs = py_utils.NestedMap(
        encoded=src_enc,
        padding=src_enc_padding,
        segment_id=None,
        embedded_inputs=None)
    return (encoder_outputs, targets)

  def testDecoderConstruction(self):
    p = self._DecoderParams()
    p.Instantiate()

  def testForwardPassWithSingleBatch(self):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(8372749040)
      p = self._DecoderParams()
      mt_dec = p.Instantiate()
      bs = 2
      tgt_seq_len = 16
      src_seq_len = 10
      encoder_outputs, targets = self._Inputs(bs, src_seq_len, tgt_seq_len)
      out = mt_dec.ComputePredictions(mt_dec.theta, encoder_outputs, targets)
      out_metrics, _ = mt_dec.ComputeLoss(mt_dec.theta, out, targets)
      dec_out_sum = tf.reduce_sum(out.softmax_input, 0)
      out_loss = out_metrics['loss'][0]
      tf.global_variables_initializer().run()
      actual_dec_out, actual_dec_out_sum, actual_loss = sess.run(
          [out.softmax_input, dec_out_sum, out_loss])
      expected_enc_out_sum = [
          [-23.059402, -32.492645, 9.186216, -48.663956, -43.15247,
           -18.73859, -19.683437, 3.3179564, 36.15105, 23.998373,
           40.686966, -0.5539336, -12.252099, 33.48251, -5.5264044,
           17.28962],
          [-20.640846, -11.11311, -8.342873, -12.426766, -48.050953,
           -7.918814, 12.720908, -0.44217646, 15.6574, 12.280106,
           33.245914, 9.623148, -0.75011516, 19.58214, 3.4654825,
           21.844471]]  # pyformat: disable
      expected_loss = 5.2651153
      self.assertAllEqual([tgt_seq_len, bs, p.model_dim], actual_dec_out.shape)
      self.assertAllClose(
          expected_enc_out_sum, actual_dec_out_sum, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_loss, actual_loss, rtol=1e-05, atol=1e-05)

  def testForwardPassWithDoubleBatch(self):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(8372749040)
      p = self._DecoderParams()
      mt_dec = p.Instantiate()
      bs = 2
      tgt_seq_len = 16
      src_seq_len = 10
      encoder_outputs, targets = self._Inputs(bs, src_seq_len, tgt_seq_len)
      other_targets = py_utils.NestedMap()
      other_targets.ids = tf.gather(targets.ids, [1, 0])
      other_targets.labels = tf.gather(targets.labels, [1, 0])
      other_targets.weights = tf.gather(targets.weights, [1, 0])
      other_targets.paddings = tf.gather(targets.paddings, [1, 0])
      lambdas = np.random.random((bs, tgt_seq_len))
      lambdas = tf.constant(lambdas, tf.float32)
      out = mt_dec.ComputePredictions(mt_dec.theta, encoder_outputs, targets,
                                      other_targets, [lambdas, 1 - lambdas])

      target_probs = np.random.random([bs, tgt_seq_len, 64])
      target_probs = target_probs / np.sum(target_probs, -1, keepdims=True)
      target_probs = tf.constant(target_probs, tf.float32)
      mix_targets = targets
      target_weights = targets.weights + other_targets.weights
      target_weights = tf.clip_by_value(target_weights, 0.0, 1.0)
      mix_targets.weights = target_weights

      out_metrics, _ = mt_dec.ComputeLoss(mt_dec.theta, out, mix_targets,
                                          target_probs)
      dec_out_sum = tf.reduce_sum(out.softmax_input, 0)
      out_loss = out_metrics['loss'][0]
      tf.global_variables_initializer().run()
      actual_dec_out, actual_dec_out_sum, actual_loss = sess.run(
          [out.softmax_input, dec_out_sum, out_loss])
      print(actual_loss)
      print(actual_dec_out_sum)
      expected_enc_out_sum = [
          [-34.133366, -43.741, -1.8258251, -38.077496, -41.201332,
           -24.28507, -6.2848973, -7.3005624, 49.394604, 30.846378,
           36.994316, -7.868125, -0.25746167, 41.251163, -7.427534, 28.979422],
          [-18.840004, -10.098586, -7.126487, -14.059292, -46.043896,
           -6.7827044, 12.584265, 1.0161059, 17.472107, 13.747282,
           31.181364, 6.1263213, 0.2827285, 20.319666, 0.05137509,
           22.13324]]  # pyformat: disable
      expected_loss = 5.2481875
      self.assertAllEqual(actual_dec_out.shape, [tgt_seq_len, bs, p.model_dim])
      self.assertAllClose(
          expected_enc_out_sum, actual_dec_out_sum, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_loss, actual_loss, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
