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
"""Tests for mt.encoder."""

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import self_attention_layer
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
      tf.random.set_seed(8372749040)
      p = self._EncoderParams()
      mt_enc = encoder.MTEncoderV1(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      self.evaluate(tf.global_variables_initializer())
      actual_enc_out = enc_out.eval()
      tf.logging.info('testForwardPass actual_enc_out %r' % actual_enc_out)
      expected_enc_out = [
          [[-2.5584161e-06, -5.6742726e-07, -8.1548797e-06, 2.6712776e-06],
           [1.1781749e-06, -4.7786052e-08, 4.2439538e-06, -3.3840388e-06]],
          [[-2.6852279e-06, 2.0878532e-07, -1.0491179e-05, 5.9619756e-06],
           [2.0423495e-06, 3.1651740e-07, 5.7234793e-06, -3.8120934e-06]],
          [[3.0904158e-07, -1.2983286e-06, -1.2469604e-05, 6.6027828e-06],
           [-3.8620223e-07, 3.8890593e-07, 1.9976458e-06, 1.0078909e-06]],
          [[1.0130438e-07, -1.1145677e-06, -1.2745468e-05, 8.0924037e-06],
           [-1.3496270e-06, -3.2355717e-06, -3.0266469e-06, -3.9747570e-06]]
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
    with self.session(use_gpu=False):
      with tf.variable_scope('bienc_test', reuse=tf.AUTO_REUSE):
        bs = 3
        sl = 3
        tf.random.set_seed(8372749040)
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

        self.evaluate(tf.global_variables_initializer())
        actual_enc_out, actual_packed_enc_out = self.evaluate(
            [enc_out, packed_enc_out])
        self.assertAllClose(actual_packed_enc_out, actual_enc_out)

  def testTransparentEncoderConstruction(self):
    p = self._BiEncoderParams()
    p.is_transparent = True
    _ = encoder.MTEncoderBiRNN(p)

  def testUniEncoderForwardPass(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)
      p = self._UniEncoderParams()
      mt_enc = encoder.MTEncoderUniRNN(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      self.evaluate(tf.global_variables_initializer())
      actual_enc_out = enc_out.eval()
      tf.logging.info('testUniEncoderForwardPass actual_enc_out %r' %
                      actual_enc_out)
      expected_enc_out = [[[-4.3304257e-07, 5.4100457e-07],
                           [-4.0170832e-07, -2.6441572e-07]],
                          [[-1.7024040e-07, -1.8555815e-07],
                           [-6.4563977e-07, -3.7835261e-07]],
                          [[-2.4001852e-07, 5.1114228e-07],
                           [-3.4349023e-07, -1.0049351e-06]],
                          [[1.8068013e-07, -6.8982729e-08],
                           [3.3005003e-07, -8.8834116e-07]]]
      self.assertAllClose(expected_enc_out, actual_enc_out)

  def testBiEncoderForwardPass(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)
      p = self._BiEncoderParams()
      mt_enc = encoder.MTEncoderBiRNN(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      self.evaluate(tf.global_variables_initializer())
      actual_enc_out = enc_out.eval()
      tf.logging.info('testBiEncoderForwardPass actual_enc_out %r' %
                      actual_enc_out)
      expected_enc_out = [[[-2.47998378e-06, 7.36457878e-06],
                           [7.89248020e-07, -2.67464316e-06]],
                          [[-2.98803275e-06, 8.20233890e-06],
                           [1.00139073e-06, -2.24554151e-06]],
                          [[-5.06675951e-06, 1.15983785e-05],
                           [-4.58391014e-07, -2.99553108e-07]],
                          [[-4.34937465e-06, 8.58816838e-06],
                           [-1.74859031e-06, 3.99598093e-06]]]
      self.assertAllClose(expected_enc_out, actual_enc_out)

  def testBiEncoderForwardPassWithDropout(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)
      p = self._BiEncoderParams()
      p.dropout_prob = 0.5
      mt_enc = encoder.MTEncoderBiRNN(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      self.evaluate(tf.global_variables_initializer())
      actual_enc_out = enc_out.eval()
      print('bi_enc_actual_enc_out_with_dropout', np.array_repr(actual_enc_out))
      expected_enc_out = [[[-1.8358192e-05, 1.2103478e-05],
                           [2.9347059e-06, -3.0652325e-06]],
                          [[-8.1282624e-06, 4.5443494e-06],
                           [3.0826509e-06, -5.2950490e-06]],
                          [[-4.6669629e-07, 2.4246765e-05],
                           [-1.5221613e-06, -1.9654153e-06]],
                          [[-1.1511075e-05, 1.9061190e-05],
                           [-5.7250163e-06, 9.2785704e-06]]]
      self.assertAllClose(expected_enc_out, actual_enc_out)

  def testBiEncoderForwardPassWithTransparent(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(8372749040)
      p = self._BiEncoderParams()
      p.is_transparent = True
      mt_enc = encoder.MTEncoderBiRNN(p)
      batch = py_utils.NestedMap()
      batch.ids = tf.transpose(tf.reshape(tf.range(0, 8, 1), [4, 2]))
      batch.paddings = tf.zeros([2, 4])
      enc_out = mt_enc.FPropDefaultTheta(batch).encoded

      self.evaluate(tf.global_variables_initializer())
      actual_enc_out = enc_out.eval()
      tf.logging.info(
          'testBiEncoderForwardPassWithTransparent actual_enc_out %r' %
          actual_enc_out)
      expected_enc_out = [[[1.53976856e-04, -1.66475205e-04],
                           [-1.02031634e-04, 1.39693424e-04]],
                          [[1.62726530e-04, -2.22654475e-04],
                           [-4.89080339e-05, 1.10912690e-04]],
                          [[1.28586107e-04, -1.62333992e-04],
                           [7.22907062e-05, -9.17545694e-05]],
                          [[9.02724860e-05, -1.71898617e-04],
                           [-9.77059244e-06, 7.55862275e-05]]]
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
    with self.session(use_gpu=False):
      bs = 2
      sl = 21
      tf.random.set_seed(8372749040)
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

      self.evaluate(tf.global_variables_initializer())
      actual_enc_out, actual_enc_out_sum, actual_emb_out_sum, \
          actual_padding = self.evaluate(
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
      tf.random.set_seed(8372749040)
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

      self.evaluate(tf.global_variables_initializer())
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
      tf.random.set_seed(8372749040)
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

      self.evaluate(tf.global_variables_initializer())
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
    with self.session(use_gpu=False):
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        bs = 3
        sl = 3
        tf.random.set_seed(8372749040)
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

        self.evaluate(tf.global_variables_initializer())
        actual_enc_out, actual_packed_enc_out = self.evaluate(
            [enc_out, packed_enc_out])

        self.assertAllClose(actual_packed_enc_out, actual_enc_out)

  def testForwardPassWithIndividuallyTaggedTokens(self):
    with self.session(use_gpu=False):
      bs = 3
      sl = 3
      tf.random.set_seed(8372749040)
      p = self._EncoderParams()
      p.packed_input = False
      p.individually_tagged_input = True
      mt_enc_tagged = encoder.TransformerEncoder(p)

      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, sl], dtype=np.int32))
      batch.paddings = tf.zeros([bs, sl])

      tagged_batch = py_utils.NestedMap()
      tagged_batch.ids = tf.reshape(batch.ids, [1, -1])
      tagged_batch.paddings = tf.reshape(batch.paddings, [1, -1])
      tagged_batch.segment_ids = tf.constant([[0, 0, 0, 1, 1, 1, 2, 2, 2]],
                                             dtype=tf.int32)

      tagged_enc_out = mt_enc_tagged.FPropDefaultTheta(tagged_batch)
      tagged_enc_out_sum = tf.reduce_sum(tagged_enc_out.encoded, 0)

      self.evaluate(tf.global_variables_initializer())
      actual_tagged_enc_out = tagged_enc_out_sum.eval()
      print(actual_tagged_enc_out)

      expected_enc_out = [[
          19.668077, -11.905859, 7.9366484, -16.66984, 23.359558, 13.41925,
          13.443447, -14.168186, 9.430209, -16.471195, 2.6439285, 11.756948,
          -4.6066704, -10.32788, 13.434055, 8.899297
      ]]
      self.assertAllClose(actual_tagged_enc_out, expected_enc_out, atol=1.0e-4)

  def testForwardPassSplitBatch(self):
    with self.session(use_gpu=False):
      bs = 8
      sl = 20
      tf.random.set_seed(8372749040)
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

      self.evaluate(tf.global_variables_initializer())
      actual_enc_out, actual_enc_out1, actual_enc_out2, \
          actual_emb_out, actual_emb_out1, actual_emb_out2 = self.evaluate(
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


class TransformerBatchMajorEncoderTest(test_utils.TestCase):

  def _TestBuilder(self, model_dim, ff_hidden_dim, num_heads, packed_input):
    return self_attention_layer.Builder.Params().Set(
        model_dim=model_dim,
        ff_hidden_dim=ff_hidden_dim,
        num_heads=num_heads,
        selfatten_add_unnormalized_input=False,
        selfatten_enable_value_proj=False,
        packed_input=packed_input).Instantiate()

  def _EncoderParams(self, packed_input=False):
    p = encoder.TransformerBatchMajorEncoder.Params()
    p.name = 'transformer_encoder'
    p.packed_input = packed_input
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim()
    p.token_emb.vocab_size = 64
    p.token_emb.embedding_dim = 16
    p.token_emb.max_num_shards = 1
    p.token_emb.vn = py_utils.VariationalNoiseParams(1.0, False, False)
    p.model_dim = 16
    p.position_emb.embedding_dim = 16
    stack = self._TestBuilder(16, 5, 2, packed_input).TransformerStack(
        'transformer_stack', 6)
    p.transformer_stack = (
        self_attention_layer.StackedTransformerEncoderLayers.Cast(stack))
    p.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
    return p

  def testEncoderConstruction(self):
    p = self._EncoderParams()
    p.Instantiate()

  def testForwardPass(self):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      d = 16
      tf.random.set_seed(8372749040)
      p = self._EncoderParams()
      mt_enc = p.Instantiate()
      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, sl], dtype=np.int32))
      batch.paddings = tf.zeros([bs, sl])
      out = mt_enc.FPropDefaultTheta(batch)
      enc_out_sum = tf.reduce_sum(out.encoded)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum = sess.run([out.encoded, enc_out_sum])

      self.assertAllEqual([sl, bs, d], actual_enc_out.shape)
      self.assertAllClose(306.010132, actual_enc_out_sum)

  def testForwardPassPackedInput(self):
    with self.session(use_gpu=False) as sess:
      bs = 2
      sl = 21
      d = 16
      tf.random.set_seed(8372749040)
      p = self._EncoderParams(packed_input=True)

      mt_enc = p.Instantiate()
      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, sl], dtype=np.int32))

      # Pack these into a single batch
      packed_bs = 1
      packed_sl = 2 * sl
      batch.ids = tf.reshape(batch.ids, [packed_bs, packed_sl])

      batch.paddings = tf.zeros([packed_bs, packed_sl])
      batch.segment_pos = [
          list(range(sl)) + list(range(sl)),
      ]
      batch.segment_ids = [
          [0 for i in range(sl)] + [1 for i in range(sl)],
      ]

      out = mt_enc.FPropDefaultTheta(batch)
      enc_out_sum = tf.reduce_sum(out.encoded)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum = sess.run([out.encoded, enc_out_sum])

      self.assertAllEqual([packed_sl, packed_bs, d], actual_enc_out.shape)
      self.assertAllClose(306.010132, actual_enc_out_sum)

  def testEncoderVars(self):
    p = self._EncoderParams()
    mt_enc = p.Instantiate()
    enc_vars = mt_enc.vars
    flatten_vars = enc_vars.Flatten()
    self.assertLen(flatten_vars, 91)


class TransformerXEncoderTest(test_utils.TestCase):

  def _EncoderParams(self):
    p = encoder.TransformerXEncoder.Params()
    p.name = 'cross_encoder'
    p.token_emb.params_init = py_utils.WeightInit.GaussianSqrtDim()
    p.token_emb.vocab_size = 64
    p.token_emb.embedding_dim = 16
    p.token_emb.max_num_shards = 1
    p.token_emb.scale_sqrt_depth = True
    p.token_emb.vn = py_utils.VariationalNoiseParams(1.0, False, False)
    p.position_emb.embedding_dim = 16
    p.position_emb.trainable_scaling = False
    p.model_dim = 16
    ts = p.transformer_stack
    ts.model_dim = 16
    ts.num_transformer_layers = 6
    ts.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    ts.transformer_tpl.tr_fflayer_tpl.hidden_dim = 5
    p.random_seed = 54321
    return p

  def testEncoderConstruction(self):
    p = self._EncoderParams()
    p.Instantiate()

  def testForwardPassWithSingleBatch(self):
    with self.session(use_gpu=False) as sess:
      p = self._EncoderParams()
      bs = 2
      seq_len = 16
      tf.random.set_seed(8372749040)
      mt_enc = p.Instantiate()
      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, seq_len], dtype=np.int32))
      batch.paddings = tf.zeros([bs, seq_len])
      out = mt_enc.FPropDefaultTheta(batch)
      enc_out_sum = tf.reduce_sum(out.encoded, 0)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum = sess.run([out.encoded, enc_out_sum])
      expected_enc_out_sum = [
          [-32.978672, -10.379181, 8.519216, -38.483955, -50.17593,
           8.633557, 31.622324, 20.088394, 26.17001, 26.835281,
           13.492112, -20.732882, 3.8693893, 47.24115, -31.376581,
           16.140488],
          [-40.31649, -12.325925, -0.41645133, -44.82979, -48.05,
           -2.8912444, 20.666195, 9.539089, 38.2287, 32.584393,
           18.059973, -25.717854, 6.607798, 51.795433, -28.735636,
           22.340008]]  # pyformat: disable

      self.assertAllEqual([seq_len, bs, p.model_dim], actual_enc_out.shape)
      self.assertAllClose(
          expected_enc_out_sum, actual_enc_out_sum, rtol=1e-05, atol=1e-05)

  def testForwardPassWithDoubleBatch(self):
    with self.session(use_gpu=False) as sess:
      p = self._EncoderParams()
      bs = 2
      seq_len = 16
      tf.random.set_seed(8372749040)
      mt_enc = p.Instantiate()
      batch = py_utils.NestedMap()
      batch.ids = tf.constant(
          np.random.randint(low=0, high=63, size=[bs, seq_len], dtype=np.int32))
      paddings = []
      for _ in range(bs):
        zeros_len = np.random.randint(1, seq_len + 1)
        paddings.append([
            0.,
        ] * zeros_len + [1.] * (seq_len - zeros_len))
      batch.paddings = tf.zeros([bs, seq_len])

      other_batch = py_utils.NestedMap()
      other_batch.ids = tf.gather(batch.ids, [1, 0])
      other_batch.paddings = tf.gather(batch.paddings, [1, 0])
      lambdas = np.random.random((bs, seq_len))
      lambdas = tf.constant(lambdas, tf.float32)
      out = mt_enc.FProp(
          mt_enc.theta,
          batch,
          interpolation_batch=other_batch,
          lambdas=[lambdas, 1 - lambdas])
      enc_out_sum = tf.reduce_sum(out.encoded, 0)

      tf.global_variables_initializer().run()
      actual_enc_out, actual_enc_out_sum = sess.run([out.encoded, enc_out_sum])

      expected_enc_out_sum = [
          [-38.089085, -22.181915, 3.3765068, -45.2483, -58.186905,
           -3.4464571, 24.461462, 12.014615, 33.08178, 34.02244,
           23.391253, -15.515911, 0.72847706, 50.45283, -26.36325, 21.799355],
          [-37.716507, -12.993027, 7.148979, -39.70747, -57.864025,
           2.2049172, 29.571432, 18.955816, 30.406136, 33.270325,
           21.685469, -17.21592, 1.3697424, 49.33187, -30.023928,
           22.915518]]  # pyformat: disable

      self.assertAllEqual([seq_len, bs, p.model_dim], actual_enc_out.shape)
      self.assertAllClose(
          expected_enc_out_sum, actual_enc_out_sum, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
