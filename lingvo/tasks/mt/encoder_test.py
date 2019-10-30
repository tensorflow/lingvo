# Lint as: python2, python3
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
"""Tests for mt.encoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.mt import encoder
import numpy as np


class EncoderTest(test_utils.TestCase):

  def _EncoderParams(self):
    p = encoder.MTEncoderV1.Params()
    p.name = 'encoder'
    p.emb.vocab_size = 16
    p.emb.embedding_dim = 4
    p.emb.max_num_shards = 1
    p.lstm_cell_size = 4
    p.num_lstm_layers = 3
    p.random_seed = 837464
    return p

  def testEncoderConstruction(self):
    p = self._EncoderParams()
    _ = encoder.MTEncoderV1(p)

  def testForwardPass(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(8372749040)
      p = self._EncoderParams()
      mt_enc = encoder.MTEncoderV1(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      tf.global_variables_initializer().run()
      actual_enc_out = enc_out.eval()
      expected_enc_out = [
          [[1.5309354e-06, -1.7816075e-07, 3.8047763e-06, -5.6422067e-07],
           [1.9017770e-06, -2.9778969e-06, -4.5083775e-06, -1.7054812e-06]],
          [[-2.1852782e-06, -1.8208171e-06, -1.4747930e-06, -5.8206351e-06],
           [6.7667429e-07, -3.6828042e-06, -1.0916860e-05, -3.2522742e-06]],
          [[-3.2333378e-07, 3.2147584e-06, 5.0556650e-07, -7.0188378e-07],
           [-6.5340635e-07, 1.9502845e-06, -9.2459632e-06, 5.1955390e-06]],
          [[2.0232728e-06, 4.9331529e-06, 1.1346837e-06, 7.5571520e-06],
           [-5.8475212e-07, 3.5547487e-06, -3.9037773e-06, 8.9575424e-06]]
      ]
      self.assertAllClose(expected_enc_out, actual_enc_out)

  def _UniEncoderParams(self):
    p = encoder.MTEncoderUniRNN.Params()
    p.name = 'encoder'
    p.emb.vocab_size = 16
    p.emb.embedding_dim = 2
    p.emb.max_num_shards = 1
    p.lstm_cell_size = 2
    p.num_lstm_layers = 3
    p.random_seed = 837464
    return p

  def _BiEncoderParams(self):
    p = encoder.MTEncoderBiRNN.Params()
    p.name = 'encoder'
    p.emb.vocab_size = 16
    p.emb.embedding_dim = 4
    p.emb.max_num_shards = 1
    p.lstm_cell_size = 4
    p.num_lstm_layers = 3
    p.encoder_out_dim = 2
    p.random_seed = 837464
    return p

  def testBiEncoderForwardPassWithInputPacking(self):
    with self.session(use_gpu=False) as sess:
      with tf.variable_scope('bienc_test', reuse=tf.AUTO_REUSE):
        bs = 3
        sl = 3
        tf.set_random_seed(8372749040)
        p = self._BiEncoderParams()
        mt_enc = encoder.MTEncoderBiRNN(p)
        packed_params = p.Copy()
        packed_params.packed_input = True
        mt_enc_packed = encoder.MTEncoderBiRNN(packed_params)

        batch = py_utils.NestedMap()
        batch.ids = tf.constant(
            np.random.randint(low=0, high=15, size=[bs, sl], dtype=np.int32))
        batch.paddings = tf.zeros([bs, sl])

        packed_batch = py_utils.NestedMap()
        packed_batch.ids = tf.reshape(batch.ids, [1, -1])
        packed_batch.paddings = tf.reshape(batch.paddings, [1, -1])
        packed_batch.segment_ids = tf.constant(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2]], dtype=tf.float32)
        packed_batch.segment_pos = tf.constant(
            [[0, 1, 2, 0, 1, 2, 0, 1, 2]], dtype=tf.int32)
        enc_out = mt_enc.FPropDefaultTheta(batch).encoded
        enc_out = tf.transpose(enc_out, [1, 0, 2])

        packed_enc_out = mt_enc_packed.FPropDefaultTheta(packed_batch)
        packed_enc_out = tf.reshape(packed_enc_out.encoded, tf.shape(enc_out))

        tf.global_variables_initializer().run()
        actual_enc_out, actual_packed_enc_out = sess.run(
            [enc_out, packed_enc_out])
        self.assertAllClose(actual_packed_enc_out, actual_enc_out)

  def testTransparentEncoderConstruction(self):
    p = self._BiEncoderParams()
    p.is_transparent = True
    _ = encoder.MTEncoderBiRNN(p)

  def testUniEncoderForwardPass(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(8372749040)
      p = self._UniEncoderParams()
      mt_enc = encoder.MTEncoderUniRNN(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      tf.global_variables_initializer().run()
      actual_enc_out = enc_out.eval()
      expected_enc_out = [[[4.3564430e-07, -2.3954138e-07],
                           [-9.8339186e-08, 8.5873717e-07]],
                          [[1.4911951e-07, -3.4084817e-07],
                           [-7.1341539e-08, 2.1979017e-06]],
                          [[-6.4054962e-07, -1.1942817e-06],
                           [-6.6461860e-07, 1.0963676e-06]],
                          [[-1.3252994e-06, -1.5549912e-06],
                           [-1.4652252e-06, 1.4637867e-06]]]
      self.assertAllClose(expected_enc_out, actual_enc_out)

  def testBiEncoderForwardPass(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(8372749040)
      p = self._BiEncoderParams()
      mt_enc = encoder.MTEncoderBiRNN(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      tf.global_variables_initializer().run()
      actual_enc_out = enc_out.eval()
      expected_enc_out = [[[4.0744379e-07, -2.0108675e-06],
                           [-4.2056736e-06, 9.2221135e-06]],
                          [[1.2086311e-06, -2.2510878e-07],
                           [-2.2938407e-06, 9.3108029e-06]],
                          [[3.4632390e-06, -3.1495360e-06],
                           [9.1814104e-07, 1.9459947e-06]],
                          [[-9.0593801e-08, -1.2912932e-06],
                           [-5.8420886e-07, -6.5603672e-07]]]
      self.assertAllClose(expected_enc_out, actual_enc_out)

  def testBiEncoderForwardPassWithDropout(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(8372749040)
      p = self._BiEncoderParams()
      p.dropout_prob = 0.5
      mt_enc = encoder.MTEncoderBiRNN(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      tf.global_variables_initializer().run()
      actual_enc_out = enc_out.eval()
      print('bi_enc_actual_enc_out_with_dropout', np.array_repr(actual_enc_out))
      expected_enc_out = [[[1.60383240e-06, 1.22550023e-06],
                           [-7.21660126e-06, 1.05704457e-05]],
                          [[1.42539475e-05, -2.06075638e-05],
                           [-4.98754298e-06, 1.51066461e-05]],
                          [[-7.15192800e-06, -6.44075908e-06],
                           [5.02962678e-07, -3.40795486e-06]],
                          [[-6.54424548e-06, 9.88359807e-06],
                           [1.42836643e-06, -1.68607176e-06]]]
      self.assertAllClose(expected_enc_out, actual_enc_out)

  def testBiEncoderForwardPassWithTransparent(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(8372749040)
      p = self._BiEncoderParams()
      p.is_transparent = True
      mt_enc = encoder.MTEncoderBiRNN(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      tf.global_variables_initializer().run()
      actual_enc_out = enc_out.eval()
      expected_enc_out = [[[-7.4536911e-05, 8.8465633e-05],
                           [2.8940600e-05, 3.2297492e-05]],
                          [[-1.9775725e-05, 9.8312848e-05],
                           [5.1837378e-05, 1.2998647e-05]],
                          [[4.5528584e-05, -6.8125606e-05],
                           [1.0955606e-04, -2.1024598e-04]],
                          [[8.5454740e-05, -1.8263397e-04],
                           [5.2042866e-05, -1.6407830e-04]]]
      self.assertAllClose(expected_enc_out, actual_enc_out)


class TransformerEncoderTest(test_utils.TestCase):

  def _EncoderParams(self):
    p = encoder.TransformerEncoder.Params()
    p.name = 'transformer_encoder'
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim()
    p.token_emb.vocab_size = 64
    p.token_emb.embedding_dim = 16
    p.token_emb.max_num_shards = 1
    p.token_emb.vn = py_utils.VariationalNoiseParams(1.0, False, False)
    p.model_dim = 16
    p.position_emb.embedding_dim = 16
    ts = p.transformer_stack
    ts.num_transformer_layers = 6
    ts.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    ts.transformer_tpl.tr_fflayer_tpl.hidden_dim = 5
    return p

  def testEncoderConstruction(self):
    p = self._EncoderParams()
    _ = encoder.TransformerEncoder(p)

  def testTransparentEncoderConstruction(self):
    p = self._EncoderParams()
    p.transformer_stack.is_transparent = True
    p.transformer_stack.num_transparent_outputs = 2
    _ = encoder.TransformerEncoder(p)

  def testForwardPass(self):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      tf.set_random_seed(8372749040)
      p = self._EncoderParams()
      mt_enc = encoder.TransformerEncoder(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, sl], dtype=np.int32))
      batch.paddings = tf.zeros([bs, sl])
      out = mt_enc.FPropDefaultTheta(batch)
      enc_out_sum = tf.reduce_sum(out.encoded, 0)
      emb_out_sum = tf.reduce_sum(out.embedded_inputs, 0)
      enc_padding = out.padding

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum, actual_emb_out_sum, \
          actual_padding = sess.run(
              [out.encoded, enc_out_sum, emb_out_sum, enc_padding])

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_enc_out = [
          [ 49.45291519, -31.5743885 ,  39.43684387, -47.67513275,
            35.39754105,  14.41970444,  29.58752823, -43.06747055,
            24.09403419,  -7.62717247,  18.48112106,  20.42408371,
            5.1519866 , -19.66542244,  29.81095314,  56.90407944],
          [ 55.26333618, -30.39743614,  29.68314743, -37.61392975,
            43.02292252,  13.88345146,  15.73033905, -24.68696213,
            24.70776558, -29.18026161,  15.41469955,  27.77672577,
            -5.36326742, -22.78984642,  22.15843391,  22.7237072 ]]
      expected_emb_out_sum = [
          [ 3.11785889,  1.33086884, -1.96904886, -4.81911993,  1.25389254,
            1.52582073,  0.79906291,  4.07078457, -1.20546532, -2.97308111,
            0.22460097,  2.99702668, -2.29453254,  6.06631422,  1.68836212,
            5.35728741],
          [ 1.41723049, -1.39409399, -1.49569404, -0.24654561,  1.09658146,
            4.51638842,  2.72023368, -0.45651400,  3.46091199, -0.43925080,
            1.02091551,  3.89704037,  1.87841535, -0.27947778, -0.91630745,
            1.34230828]]
      # pylint: enable=bad-whitespace
      # pyformat: enable
      self.assertAllEqual(actual_enc_out.shape, [sl, bs, p.model_dim])
      self.assertAllEqual(actual_padding.shape, [sl, bs])
      self.assertAllClose(
          expected_enc_out, actual_enc_out_sum, rtol=1e-05, atol=1e-05)
      self.assertAllClose(
          expected_emb_out_sum, actual_emb_out_sum, rtol=1e-05, atol=1e-05)

  def testForwardPassWithTaskEmb(self):
    with self.session(use_gpu=False):
      bs = 2
      sl = 21
      tf.set_random_seed(8372749040)
      p = self._EncoderParams()
      p.task_emb = p.token_emb.Copy()
      p.task_emb.vocab_size = 4
      mt_enc = encoder.TransformerEncoder(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, sl], dtype=np.int32))
      batch.task_ids = tf.constant(
          np.random.randint(low=0, high=3, size=[bs, sl], dtype=np.int32))
      batch.paddings = tf.zeros([bs, sl])

      enc_out = mt_enc.FPropDefaultTheta(batch)
      enc_out_sum = tf.reduce_sum(enc_out.encoded, 0)

      tf.global_variables_initializer().run()
      actual_enc_out = enc_out_sum.eval()

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_enc_out = [
          [ 1.2796677,  -31.786997, -0.4054339, -32.61311 ,
            42.41403,   11.020338,  54.115948,  -61.322887,
            39.593548,  15.315696,  -20.373957, 1.8548622,
            -17.743631, 3.140956,   30.730812,  41.4348],
          [ -1.0373995, -31.306532, -2.6323462, -32.078648,
            45.80049,   16.409424,  55.00114,   -63.102333,
            40.4261,    14.198621,  -23.027012, 1.0839912,
            -20.739473, 0.7242553,  32.49956,   41.592197]]
      # pylint: enable=bad-whitespace
      # pyformat: enable
      self.assertAllClose(
          expected_enc_out, actual_enc_out, rtol=1e-05, atol=1e-05)

  def testForwardPassWithSourceMask(self):
    with self.session(use_gpu=False):
      bs = 2
      sl = 21
      tf.set_random_seed(8372749040)
      p = self._EncoderParams()
      p.task_emb = p.token_emb.Copy()
      p.task_emb.vocab_size = 4
      # 4 tasks, 2 languages.
      p.apply_source_mask = True
      mt_enc = encoder.TransformerEncoder(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, sl], dtype=np.int32))
      batch.task_ids = tf.constant(
          np.random.randint(low=0, high=3, size=[bs, sl], dtype=np.int32))
      batch.paddings = tf.zeros([bs, sl])

      enc_out = mt_enc.FPropDefaultTheta(batch)
      enc_out_sum = tf.reduce_sum(enc_out.encoded, 0)

      tf.global_variables_initializer().run()
      actual_enc_out = enc_out_sum.eval()
      # pyformat: disable
      # pylint: disable=bad-whitespace
      print(actual_enc_out)

      expected_enc_out = [
          [1.2796695, -31.786999, -0.4054371, -32.61311, 42.414032, 11.020337,
           54.11595, -61.322884, 39.59355, 15.315693, -20.373957, 1.8548615,
           -17.743631, 3.1409538, 30.730812, 41.4348],
          [-1.0374013, -31.306532, -2.6323478, -32.078648, 45.800484, 16.40942,
           55.001144, -63.10233, 40.4261, 14.19862, -23.027012, 1.0839913,
           -20.739471, 0.7242559, 32.499565, 41.592197]]
      # pylint: enable=bad-whitespace
      # pyformat: enable
      self.assertAllClose(
          expected_enc_out, actual_enc_out, rtol=1e-05, atol=1e-05)

  def testForwardPassWithInputPacking(self):
    with self.session(use_gpu=False) as sess:
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        bs = 3
        sl = 3
        tf.set_random_seed(8372749040)
        p = self._EncoderParams()
        mt_enc = encoder.TransformerEncoder(p)
        packed_params = p.Copy()
        packed_params.packed_input = True
        mt_enc_packed = encoder.TransformerEncoder(packed_params)

        batch = py_utils.NestedMap()
        batch.ids = tf.constant(
            np.random.randint(low=0, high=63, size=[bs, sl], dtype=np.int32))
        batch.paddings = tf.zeros([bs, sl])

        packed_batch = py_utils.NestedMap()
        packed_batch.ids = tf.reshape(batch.ids, [1, -1])
        packed_batch.paddings = tf.reshape(batch.paddings, [1, -1])
        packed_batch.segment_ids = tf.constant(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2]], dtype=tf.float32)
        packed_batch.segment_pos = tf.constant(
            [[0, 1, 2, 0, 1, 2, 0, 1, 2]], dtype=tf.int32)
        enc_out = mt_enc.FPropDefaultTheta(batch).encoded
        enc_out = tf.transpose(enc_out, [1, 0, 2])

        packed_enc_out = mt_enc_packed.FPropDefaultTheta(packed_batch)
        packed_enc_out = tf.reshape(packed_enc_out.encoded, tf.shape(enc_out))

        enc_out = tf.reduce_sum(enc_out, axis=0)
        packed_enc_out = tf.reduce_sum(packed_enc_out, axis=0)

        tf.global_variables_initializer().run()
        actual_enc_out, actual_packed_enc_out = sess.run(
            [enc_out, packed_enc_out])

        self.assertAllClose(actual_packed_enc_out, actual_enc_out)

  def testForwardPassSplitBatch(self):
    with self.session(use_gpu=False) as sess:
      bs = 8
      sl = 20
      tf.set_random_seed(8372749040)
      p = self._EncoderParams()
      p.random_seed = 1234
      mt_enc = encoder.TransformerEncoder(p)

      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, sl], dtype=np.int32))
      batch.paddings = tf.zeros([bs, sl])
      out = mt_enc.FPropDefaultTheta(batch)
      enc_out = out.encoded
      emb_out = out.embedded_inputs

      inputs1, inputs2 = tf.split(batch.ids, 2, 0)
      paddings1, paddings2 = tf.split(batch.paddings, 2, 0)

      batch.ids = inputs1
      batch.paddings = paddings1
      out1 = mt_enc.FPropDefaultTheta(batch)
      enc_out1 = out1.encoded
      emb_out1 = out1.embedded_inputs

      batch.ids = inputs2
      batch.paddings = paddings2
      out2 = mt_enc.FPropDefaultTheta(batch)
      enc_out2 = out2.encoded
      emb_out2 = out2.embedded_inputs

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out1, actual_enc_out2, \
          actual_emb_out, actual_emb_out1, actual_emb_out2 = sess.run(
              [enc_out, enc_out1, enc_out2, emb_out, emb_out1, emb_out2])
      self.assertAllClose(actual_enc_out,
                          np.concatenate([actual_enc_out1, actual_enc_out2], 1))
      self.assertAllClose(actual_emb_out,
                          np.concatenate([actual_emb_out1, actual_emb_out2], 1))

  def testEncoderVars(self):
    p = self._EncoderParams()
    mt_enc = encoder.TransformerEncoder(p)
    enc_vars = mt_enc.vars
    flatten_vars = enc_vars.Flatten()
    self.assertEqual(len(flatten_vars), 91)


if __name__ == '__main__':
  tf.test.main()
