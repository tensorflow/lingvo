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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import input_generator_helper as ig_helper
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core.test_utils import CompareToGoldenSingleFloat
from lingvo.tasks.mt import decoder

FLAGS = tf.flags.FLAGS

_NUMPY_RANDOM_SEED = 9885784
_TF_RANDOM_SEED = 8372749040


class DecoderTest(tf.test.TestCase):

  def _DecoderParams(self, per_word_avg_loss=False, dtype=tf.float32):
    p = decoder.MTDecoderV1.Params()
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

    for lp in base_layer.RecursiveFindLayerParams(p):
      lp.dtype = dtype

    return p

  def testDecoderConstruction(self):
    p = self._DecoderParams()
    _ = decoder.MTDecoderV1(p)

  def _testInputs(self, dtype=tf.float32):
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
        tf.constant(
            [[0, 1, 2, 3], [0, 5, 6, 7], [0, 10, 11, 12]], dtype=tf.int32))
    target_labels = tf.transpose(
        tf.constant(
            [[1, 2, 3, 4], [5, 6, 7, 8], [10, 11, 12, 13]], dtype=tf.int32))
    target_paddings = tf.transpose(
        tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=dtype))
    target_weights = 1.0 - target_paddings
    targets = py_utils.NestedMap({
        'ids': target_ids,
        'labels': target_labels,
        'weights': target_weights,
        'paddings': target_paddings
    })

    return (src_enc, src_enc_padding, targets)

  def _testDecoderFPropHelper(self, decoder_cls, dtype,
                              feed_att_context_to_softmax):
    with self.session(use_gpu=False):
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)

      p.feed_attention_context_vec_to_softmax = feed_att_context_to_softmax
      dec = decoder_cls(p)
      src_enc, src_enc_padding, targets = self._testInputs(dtype=dtype)
      loss, _ = dec.FPropDefaultTheta(src_enc, src_enc_padding, targets,
                                      None)['loss']

      tf.global_variables_initializer().run()
      actual_loss = loss.eval()
      print('actual loss = ', actual_loss)
      if p.feed_attention_context_vec_to_softmax:
        CompareToGoldenSingleFloat(self, 7.613735, actual_loss)
      else:
        CompareToGoldenSingleFloat(self, 7.624220, actual_loss)

  def testDecoderFPropFixedAttentionSeed(self, dtype=tf.float64):
    with self.session(use_gpu=False):
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      p.feed_attention_context_vec_to_softmax = False
      p.attention.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
      dec = decoder.MTDecoderV1(p)
      src_enc, src_enc_padding, targets = self._testInputs(dtype=dtype)
      loss, _ = dec.FPropDefaultTheta(src_enc, src_enc_padding, targets,
                                      None)['loss']

      tf.global_variables_initializer().run()
      actual_loss = loss.eval()
      print('actual loss = ', actual_loss)
      CompareToGoldenSingleFloat(self, 7.624183, actual_loss)

  def testDecoderFPropFunctional(self):
    self._testDecoderFPropHelper(decoder.MTDecoderV1, tf.float64, False)

  def testDecoderFPropFunctionalFeedingAttContext(self):
    self._testDecoderFPropHelper(decoder.MTDecoderV1, tf.float64, True)

  def _DecoderGradientCheckerHelper(self,
                                    decoder_cls,
                                    feed_att_context_to_softmax=False):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=tf.float64)
      p.feed_attention_context_vec_to_softmax = feed_att_context_to_softmax
      dec = decoder_cls(p)
      src_enc, src_enc_padding, targets = self._testInputs(dtype=tf.float64)
      loss, _ = dec.FPropDefaultTheta(src_enc, src_enc_padding, targets,
                                      None)['loss']
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      print('num of vars ', len(all_vars))

      def DenseGrad(var, grad):
        if isinstance(grad, tf.Tensor):
          return grad
        elif isinstance(grad, tf.IndexedSlices):
          return tf.unsorted_segment_sum(grad.values, grad.indices,
                                         tf.shape(var)[0])

      grads = [DenseGrad(x, y) for x, y in zip(all_vars, grads)]

    with self.session(use_gpu=False, graph=g) as sess:
      tf.global_variables_initializer().run()
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

  def testDecoderBPropFunctional(self):
    self._DecoderGradientCheckerHelper(decoder.MTDecoderV1)

  def testDecoderBPropFunctionalFeedingAttContext(self):
    self._DecoderGradientCheckerHelper(
        decoder.MTDecoderV1, feed_att_context_to_softmax=True)

  def _testDecoderPerWordAvgLossFPropHelper(self,
                                            decoder_cls,
                                            feed_att_context_to_softmax=False):
    with self.session(use_gpu=False):
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(True)
      p.feed_attention_context_vec_to_softmax = feed_att_context_to_softmax
      dec = decoder_cls(p)
      src_enc, src_enc_padding, targets = self._testInputs()
      loss, _ = dec.FPropDefaultTheta(src_enc, src_enc_padding, targets,
                                      None)['loss']
      tf.global_variables_initializer().run()
      actual_loss = loss.eval()
      print('actual loss = ', actual_loss)
      if p.feed_attention_context_vec_to_softmax:
        CompareToGoldenSingleFloat(self, 2.769071, actual_loss)
      else:
        CompareToGoldenSingleFloat(self, 2.772190, actual_loss)

  def testDecoderPerWordAvgLossFPropFunctional(self):
    self._testDecoderPerWordAvgLossFPropHelper(decoder.MTDecoderV1)

  def testDecoderPerWordAvgLossFPropFunctionalFeedingAttContext(self):
    self._testDecoderPerWordAvgLossFPropHelper(
        decoder.MTDecoderV1, feed_att_context_to_softmax=True)

  def testBeamSearchDecode(self, dtype=tf.float32):
    tf.set_random_seed(_TF_RANDOM_SEED)
    src_batch = 2
    p = self._DecoderParams(dtype=dtype)
    p.is_eval = True
    src_time = p.target_seq_len
    p.beam_search.num_hyps_per_beam = 2
    p.rnn_cell_dim = 32
    dec = decoder.MTDecoderV1(p)
    src_enc, src_enc_padding, _ = self._testInputs(dtype=dtype)
    decode = dec.BeamSearchDecode(src_enc, src_enc_padding)
    # topk_decoded is None in MT decoder, set it to a fake tensor to pass
    # sess.run(decode).
    decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      actual_decode = sess.run(decode)

    self.assertTupleEqual(
        (src_time, src_batch * p.beam_search.num_hyps_per_beam),
        actual_decode.done_hyps.shape)
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

    expected_topk_ids = [[2, 0, 0, 0, 0], [11, 2, 0, 0, 0], [2, 0, 0, 0, 0],
                         [6, 2, 0, 0, 0]]

    expected_topk_lens = [1, 2, 1, 2]
    expected_topk_scores = [[-3.781308, -5.741293], [-3.332158, -5.597181]]

    self.assertAllEqual(expected_topk_ids, actual_decode.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decode.topk_lens)
    self.assertAllClose(expected_topk_scores, actual_decode.topk_scores)

  def testBeamSearchDecodeFeedingAttContext(self, dtype=tf.float32):
    tf.set_random_seed(_TF_RANDOM_SEED)
    src_batch = 2
    p = self._DecoderParams(dtype=dtype)
    p.is_eval = True
    src_time = p.target_seq_len
    p.beam_search.num_hyps_per_beam = 2
    p.rnn_cell_dim = 32
    p.feed_attention_context_vec_to_softmax = True
    dec = decoder.MTDecoderV1(p)
    src_enc, src_enc_padding, _ = self._testInputs(dtype=dtype)
    decode = dec.BeamSearchDecode(src_enc, src_enc_padding)
    # topk_decoded is None in MT decoder, set it to a fake tensor to pass
    # sess.run(decode).
    decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      actual_decode_feeding_att_context = sess.run(decode)

    self.assertTupleEqual(
        (src_time, src_batch * p.beam_search.num_hyps_per_beam),
        actual_decode_feeding_att_context.done_hyps.shape)
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

    expected_topk_ids = [[2, 0, 0, 0, 0], [12, 2, 0, 0, 0], [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]

    expected_topk_lens = [1, 2, 0, 0]
    expected_topk_scores = [[-3.7437, -5.654146], [0., 0.]]

    self.assertAllEqual(expected_topk_ids,
                        actual_decode_feeding_att_context.topk_ids)
    self.assertAllEqual(expected_topk_lens,
                        actual_decode_feeding_att_context.topk_lens)
    self.assertAllClose(expected_topk_scores,
                        actual_decode_feeding_att_context.topk_scores)


class TransformerDecoderTest(tf.test.TestCase):

  def _DecoderParams(self,
                     per_word_avg_loss=False,
                     is_transparent=False,
                     dtype=tf.float32,
                     fprop_dtype=None):
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
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim()
    p.position_emb.embedding_dim = 4
    p.trans_tpl.vn = disable_vn
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

  def testDecoderConstruction(self):
    p = self._DecoderParams()
    _ = decoder.TransformerDecoder(p)

  def testTransparentDecoderConstruction(self):
    p = self._DecoderParams(is_transparent=True)
    _ = decoder.TransformerDecoder(p)

  def _testInputs(self, dtype=tf.float32):
    np.random.seed(_NUMPY_RANDOM_SEED)
    src_time = 5
    src_batch = 4
    emb_dims = 4
    src_enc = tf.constant(
        np.random.normal(size=[src_time, src_batch, emb_dims]), dtype=dtype)
    src_paddings = tf.zeros([src_time, src_batch], dtype=dtype)
    tgt_time = 5
    tgt_batch = 8
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
    return (src_enc, src_paddings, tgts)

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
    _, paddings, tgts = self._testInputs(dtype)
    src_enc = tf.constant(
        np.random.normal(size=[src_time, src_batch, emb_dims, num_layers]),
        dtype=dtype)
    if not is_eval_mode:
      src_enc = tf.unstack(src_enc, axis=3)
    return (src_enc, paddings, tgts)

  def testDecoderFPropWithPacking(self, dtype=tf.float32):
    with self.session(use_gpu=False) as sess:
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        tf.set_random_seed(_TF_RANDOM_SEED)
        p = self._DecoderParams(per_word_avg_loss=True, dtype=dtype)
        # Localized label smoother messes up the loss with packing.
        p.label_smoothing = None
        dec = decoder.TransformerDecoder(p)
        p_packed = p.Copy()
        p_packed.packed_input = True
        dec_packed = decoder.TransformerDecoder(p_packed)

        (src_enc, paddings, tgts, src_enc_packed, src_enc_padding_packed,
         src_segment_id, target_packed) = self._testPackedInputs()

        loss, _ = dec.FProp(dec.theta, src_enc, paddings, tgts, None)['loss']

        loss_packed, _ = dec_packed.FProp(dec_packed.theta, src_enc_packed,
                                          src_enc_padding_packed, target_packed,
                                          src_segment_id)['loss']
        tf.global_variables_initializer().run()
        actual_loss, packed_loss = sess.run([loss, loss_packed])
        self.assertAlmostEqual(
            np.float32(packed_loss), np.float32(actual_loss), delta=1e-4)

  def testTransparentDecoderFProp(self, dtype=tf.float32):
    with self.session(use_gpu=False):
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(is_transparent=True, dtype=dtype)
      dec = decoder.TransformerDecoder(p)
      src_enc, src_enc_padding, targets = self._testTransparentInputs(
          dtype=dtype)
      loss, _ = dec.FPropDefaultTheta(src_enc, src_enc_padding, targets,
                                      None)['loss']
      tf.global_variables_initializer().run()
      actual_loss = loss.eval()
      print('actual loss = ', actual_loss)
      self.assertAlmostEqual(15.864315, actual_loss, delta=0.0001)

  def _testExtendStep(self, sess, dec, src_enc, src_padding, tgts):
    p = self._DecoderParams()
    l_out1 = dec._FProp(dec.theta, src_enc, src_padding, tgts, None)

    prefix_states = py_utils.NestedMap()
    for i in range(6):
      layer_i_states = py_utils.NestedMap()
      # the middle dim is for num of transformer layers. Here's 0 as placeholder
      layer_i_states.key = tf.zeros([self.tgt_batch, 0, p.model_dim])
      layer_i_states.value = tf.zeros([self.tgt_batch, 0, p.model_dim])
      prefix_states['layer_%i' % i] = layer_i_states

    l_out2 = []
    for i in range(5):
      l_i_out, prefix_states = dec.ExtendStep(dec.theta, src_enc, src_padding,
                                              tgts.ids[:, i], i, prefix_states)
      l_out2.append(l_i_out)

    l_out2 = tf.stack(l_out2)

    tf.global_variables_initializer().run()
    l_out1_v, l_out2_v = sess.run([l_out1, l_out2])
    self.assertAllClose(l_out1_v, l_out2_v)

  def testDecoderExtendStep(self, dtype=tf.float32):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      dec = decoder.TransformerDecoder(p)
      src_enc, src_enc_padding, targets = self._testInputs(dtype=dtype)
      self._testExtendStep(sess, dec, src_enc, src_enc_padding, targets)

  def testTransparentDecoderExtendStep(self, dtype=tf.float32):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(is_transparent=True, dtype=dtype)
      p.is_eval = True
      dec = decoder.TransformerDecoder(p)
      src_enc, src_enc_padding, targets = self._testTransparentInputs(
          dtype=dtype, is_eval_mode=True)
      self._testExtendStep(sess, dec, src_enc, src_enc_padding, targets)

  def testDecoderFPropSplitBatch(self, dtype=tf.float32):
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(_TF_RANDOM_SEED)
      p = self._DecoderParams(dtype=dtype)
      dec = decoder.TransformerDecoder(p)

      src_enc, src_paddings, targets = self._testInputs(dtype=dtype)
      src_enc1, src_enc2 = tf.split(src_enc, 2, 1)
      src_paddings1, src_paddings2 = tf.split(src_paddings, 2, 1)

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

      loss, _ = dec.FPropDefaultTheta(src_enc, src_paddings, targets,
                                      None)['loss']
      loss1, _ = dec.FPropDefaultTheta(src_enc1, src_paddings1, targets1,
                                       None)['loss']
      loss2, _ = dec.FPropDefaultTheta(src_enc2, src_paddings2, targets2,
                                       None)['loss']

      tf.global_variables_initializer().run()
      actual_loss, actual_loss1, actual_loss2 = sess.run([loss, loss1, loss2])
      print('actual loss = ', actual_loss)
      print('actual loss1 = ', actual_loss1)
      print('actual loss2 = ', actual_loss2)
      self.assertAlmostEqual(
          actual_loss, np.mean([actual_loss1, actual_loss2]), delta=0.0001)

  def testBeamSearchDecode(self, dtype=tf.float32):
    tf.set_random_seed(_TF_RANDOM_SEED)
    src_batch = 4
    src_time = 5
    p = self._DecoderParams(dtype=dtype)
    p.beam_search.num_hyps_per_beam = 2
    dec = decoder.TransformerDecoder(p)
    src_enc, src_enc_padding, _ = self._testInputs(dtype=dtype)
    decode = dec.BeamSearchDecode(src_enc, src_enc_padding)
    # topk_decoded is None in MT decoder, set it to a fake tensor to pass
    # sess.run(decode).
    decode = decode._replace(topk_decoded=tf.constant(0, tf.float32))

    with self.session(use_gpu=False) as sess:
      tf.global_variables_initializer().run()
      actual_decode = sess.run(decode)

    self.assertTupleEqual(
        (src_time, src_batch * p.beam_search.num_hyps_per_beam),
        actual_decode.done_hyps.shape)
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

    expected_topk_ids = [[2, 0, 0, 0, 0], [6, 2, 0, 0, 0], [2, 0, 0, 0, 0],
                         [14, 2, 0, 0, 0], [6, 2, 0, 0, 0], [6, 6, 2, 0, 0],
                         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    expected_topk_lens = [1, 2, 1, 2, 2, 3, 0, 0]
    expected_topk_scores = [[-2.03276896,
                             -3.56309867], [-2.03811574, -3.5735178],
                            [-4.40967178, -6.06508398], [0., 0.]]

    self.assertAllEqual(expected_topk_ids, actual_decode.topk_ids)
    self.assertAllEqual(expected_topk_lens, actual_decode.topk_lens)
    self.assertAllClose(expected_topk_scores, actual_decode.topk_scores)


if __name__ == '__main__':
  tf.test.main()
