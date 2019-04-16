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

import numpy as np
import tensorflow as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.mt import encoder


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
      expected_enc_out = [[[
          -7.51581979e-07, 1.55304758e-06, -3.39117889e-07, 2.79457527e-06
      ], [-1.06733505e-05, 7.56898862e-06, -4.18875834e-06, -9.10360086e-06]], [
          [1.58444971e-06, 5.11627661e-07, 1.33408967e-05, 1.81603957e-06],
          [-1.59942228e-05, 1.26068180e-05, 4.49321249e-07, -1.43790385e-05]
      ], [[5.56546365e-06, -8.01007627e-06, 8.96620350e-06, 3.96485439e-06], [
          -8.77006005e-06, 4.04282991e-06, -4.79895652e-06, -5.90156833e-06
      ]], [[-8.59513818e-07, -7.63760727e-06, -5.57065960e-06, 1.80756274e-06],
           [-2.96017470e-06, -1.51323195e-06, -1.03562079e-05, 1.23328198e-06]]]
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
      expected_enc_out = [[[-1.74790625e-06, -5.04228524e-07], [
          2.04836829e-06, 1.48639378e-06
      ]], [[-1.10486064e-06, -5.77133278e-07],
           [4.66779238e-06,
            3.72350723e-06]], [[-5.65139544e-07, -1.84634030e-06],
                               [3.99908731e-06, 1.90148887e-06]],
                          [[7.14102157e-07, -2.31352783e-06],
                           [7.05981620e-06, 2.68004328e-06]]]
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
      expected_enc_out = [[[1.42110639e-06, 1.31101151e-05], [
          -6.62138473e-06, -1.11313329e-06
      ]], [[1.14506956e-05, 2.98347204e-05], [-5.89276988e-06, 5.54328744e-06]],
                          [[1.35346390e-05, 1.00745674e-05],
                           [-4.80002745e-06, -1.23648788e-05]],
                          [[2.00507566e-06, -1.51463591e-05],
                           [-5.71241526e-06, -1.87959231e-05]]]
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
      # pylint: disable=bad-whitespace,bad-continuation
      # pyformat: disable
      expected_enc_out = [
          [[-2.25614094e-05, 1.19781353e-05],
           [-2.74532852e-07, 8.17993077e-06]],
          [[2.66865045e-05, 1.02941645e-04],
           [1.51371260e-05, 3.78371587e-05]],
          [[3.50117516e-05, 7.65562072e-06],
           [-1.30227636e-05, 3.01171349e-06]],
          [[2.27566215e-06, 1.42354111e-07],
           [1.04521234e-06, 2.50320113e-06]]
       ]
      # pyformat: enable
      # pylint: enable=bad-whitespace,bad-continuation
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
      # pylint: disable=bad-whitespace,bad-continuation
      # pyformat: disable
      expected_enc_out = [
         [[ -4.95129862e-05,   1.92920983e-04],
          [ -2.18278728e-10,  -1.56396545e-05]],
         [[  1.80083007e-04,   6.55014810e-05],
          [  8.51639197e-05,   6.82225800e-05]],
         [[  2.12642481e-05,  -2.92667974e-05],
          [ -8.88068098e-05,   5.24776005e-05]],
         [[ -1.33993672e-04,   3.61708371e-05],
          [ -1.35903974e-04,   1.31157576e-05]]
      ]
      # pyformat: enable
      # pylint: enable=bad-whitespace,bad-continuation
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
      enc_padding = out.padding

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum, actual_padding = sess.run(
          [out.encoded, enc_out_sum, enc_padding])

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
      # pylint: enable=bad-whitespace
      # pyformat: enable
      self.assertAllEqual(actual_enc_out.shape, [sl, bs, p.model_dim])
      self.assertAllEqual(actual_padding.shape, [sl, bs])
      self.assertAllClose(expected_enc_out, actual_enc_out_sum)

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
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      inputs1, inputs2 = tf.split(batch.ids, 2, 0)
      paddings1, paddings2 = tf.split(batch.paddings, 2, 0)

      batch.ids = inputs1
      batch.paddings = paddings1
      enc_out1 = mt_enc.FPropDefaultTheta(batch).encoded

      batch.ids = inputs2
      batch.paddings = paddings2
      enc_out2 = mt_enc.FPropDefaultTheta(batch).encoded

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out1, actual_enc_out2 = sess.run(
          [enc_out, enc_out1, enc_out2])
      self.assertAllClose(actual_enc_out,
                          np.concatenate([actual_enc_out1, actual_enc_out2], 1))

  def testEncoderVars(self):
    p = self._EncoderParams()
    mt_enc = encoder.TransformerEncoder(p)
    enc_vars = mt_enc.vars
    flatten_vars = enc_vars.Flatten()
    self.assertEqual(len(flatten_vars), 91)


if __name__ == '__main__':
  tf.test.main()
