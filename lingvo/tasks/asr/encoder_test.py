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
"""Tests for ASR encoder."""

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.asr import encoder
import numpy as np


class EncoderTest(test_utils.TestCase):

  def _EncoderParams(self, vn_config):
    p = encoder.AsrEncoder.Params()
    p.name = 'encoder'
    vn_config.scale = 0.1
    params_init = py_utils.WeightInit.Uniform(0.05)

    rnn_params = p.lstm_tpl
    rnn_params.vn = vn_config
    rnn_params.params_init = params_init

    cnn_params = p.cnn_tpl
    cnn_params.vn = vn_config
    cnn_params.params_init = params_init

    proj_params = p.proj_tpl
    proj_params.vn = vn_config
    proj_params.params_init = params_init

    p.conv_filter_shapes = [[3, 3, 3, 6], [3, 3, 6, 6]]
    p.conv_filter_strides = [[2, 2], [2, 2]]
    p.input_shape = [None, None, 16, 3]

    p.conv_lstm_tpl.vn = vn_config
    p.after_conv_lstm_cnn_tpl.vn = vn_config

    p.num_cnn_layers = 2
    p.num_lstm_layers = 2
    p.lstm_cell_size = 16
    p.num_conv_lstm_layers = 0

    return p

  def _ForwardPass(self, p):
    tf.random.set_seed(8372749040)
    stt_enc = encoder.AsrEncoder(p)
    batch = py_utils.NestedMap()
    batch.src_inputs = tf.random.normal(
        [2, 20] + p.input_shape[2:], seed=92837472)
    batch.paddings = tf.zeros([2, 20])
    return stt_enc.FPropDefaultTheta(batch)

  def testEncoderConstruction(self):
    vn_config = py_utils.VariationalNoiseParams(None, True, False, seed=12345)
    p = self._EncoderParams(vn_config)
    _ = encoder.AsrEncoder(p)

  def testForwardPass(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out, 0)
      self.evaluate(tf.global_variables_initializer())

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_enc_out = [
          [-2.63900943e-02,  -4.88980189e-02,   1.78375337e-02,
           -9.66763496e-03,  -1.45432353e-02,   3.63842538e-03,
           -4.93378285e-03,   9.87463910e-03,  -1.98941268e-02,
           -2.31636949e-02,  -6.76718354e-03,  -1.01988772e-02,
           4.81432397e-03,   9.02220048e-03,   1.31793215e-03,
           -1.39696691e-02,  -2.36637704e-02,  -5.25583047e-04,
           -3.79295787e-03,   1.09998491e-02,   8.54234211e-03,
           -2.43989471e-02,  -6.27756910e-03,  -1.64192859e-02,
           1.54568311e-02,   3.69091239e-03,   1.27634332e-02,
           2.50437222e-02,   3.77510749e-02,   1.71656217e-02,
           1.94890760e-02,   4.31961473e-03],
          [-1.61839426e-02,   1.27755934e-02,  -1.96352396e-02,
           1.04363225e-02,   6.10197056e-03,  -5.08408714e-03,
           -9.20344493e-04,   2.55419128e-02,  -3.58198807e-02,
           -4.18110676e-02,   9.45025682e-03,  -7.00431701e-04,
           2.31945589e-02,  -6.53471798e-05,  -1.94577798e-02,
           -1.53421704e-02,  -1.50274234e-02,   1.06492080e-03,
           8.32110923e-03,  -1.38334394e-03,   2.02696323e-02,
           2.13975199e-02,   2.23143250e-02,  -1.54133392e-02,
           1.83746461e-02,   8.25020485e-03,  -1.64317098e-02,
           1.46762179e-02,   1.89543713e-03,  -3.36170895e-03,
           3.14423591e-02,  -2.64923554e-02 ]]
      # pylint: enable=bad-whitespace
      # pyformat: enable
      enc_out_sum_val = enc_out_sum.eval()
      print('enc_out_sum_val', np.array_repr(enc_out_sum_val))
      self.assertAllClose(expected_enc_out, enc_out_sum_val)

  def testForwardPassLstmOnly(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.num_cnn_layers = 0
      p.num_conv_lstm_layers = 0
      p.lstm_cell_size = 1024
      p.num_lstm_layers = 4
      p.conv_filter_strides = []
      p.conv_filter_shapes = []
      p.input_shape = [None, None, 96, 1]
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      self.evaluate(tf.global_variables_initializer())

      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      test_utils.CompareToGoldenSingleFloat(self, -280.61724853515625,
                                            enc_out_sum_val)

  def testForwardPassWithConvLSTM(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.num_conv_lstm_layers = 1
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      self.evaluate(tf.global_variables_initializer())

      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      test_utils.CompareToGoldenSingleFloat(
          self, -0.1091638132929802, enc_out_sum_val)

  def testForwardPassWithConvLSTM2Layers(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.num_conv_lstm_layers = 2
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      self.evaluate(tf.global_variables_initializer())

      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      test_utils.CompareToGoldenSingleFloat(
          self, 0.024680551141500473, enc_out_sum_val)

  def testForwardPassWithResidualStart1(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.residual_start = 1
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      self.evaluate(tf.global_variables_initializer())

      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      test_utils.CompareToGoldenSingleFloat(
          self, 119.61833190917969, enc_out_sum_val)

  def testForwardPassWithResidualStart2(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.residual_start = 2
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      self.evaluate(tf.global_variables_initializer())

      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      test_utils.CompareToGoldenSingleFloat(
          self, 13.103919982910156, enc_out_sum_val)

  def testForwardPassWithResidualStart1Interval2(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.residual_start = 1
      p.residual_stride = 2
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      self.evaluate(tf.global_variables_initializer())

      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      test_utils.CompareToGoldenSingleFloat(
          self, 92.58077239990234, enc_out_sum_val)

  def testForwardPassWithStackingAfterFinalLayer(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.stacking_layer_tpl.left_context = 1
      p.stacking_layer_tpl.right_context = 0
      p.stacking_layer_tpl.stride = 2
      p.layer_index_before_stacking = 1
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out, 0)
      self.evaluate(tf.global_variables_initializer())
      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_enc_out = [
          [-1.25796525e-02, -2.32883729e-02, 7.40477070e-03, -4.51436592e-03,
           -5.84740378e-03, 2.30195466e-03, -3.08505213e-03, 4.05658083e-03,
           -8.12252797e-03, -1.08030904e-02, -4.17955732e-03, -3.73707339e-03,
           6.97144482e-04, 2.79850606e-03, 8.33133236e-04, -5.75614115e-03,
           -1.10648498e-02, -1.20132393e-03, -1.69872947e-03, 6.97519444e-03,
           2.46211258e-03, -1.28190573e-02, -8.66306946e-05, -6.09322963e-03,
           7.14540575e-03, -5.67986863e-05, 5.17684873e-03, 1.18097477e-02,
           1.74862407e-02, 9.13049746e-03, 7.31027778e-03, 4.83186450e-05,
           -1.38104409e-02, -2.56096497e-02, 1.04327593e-02, -5.15327370e-03,
           -8.69584084e-03, 1.33647269e-03, -1.84873224e-03, 5.81806153e-03,
           -1.17716007e-02, -1.23606063e-02, -2.58761784e-03, -6.46180846e-03,
           4.11718246e-03, 6.22369815e-03, 4.84800315e-04, -8.21352564e-03,
           -1.25989169e-02, 6.75740885e-04, -2.09423108e-03, 4.02465323e-03,
           6.08023722e-03, -1.15798926e-02, -6.19094400e-03, -1.03260633e-02,
           8.31142440e-03, 3.74771934e-03, 7.58658582e-03, 1.32339774e-02,
           2.02648211e-02, 8.03512800e-03, 1.21787926e-02, 4.27130330e-03],
          [-5.94401825e-03, 4.23503201e-03, -7.39302021e-03, 3.84659087e-03,
           2.92047067e-03, -2.28955783e-03, 7.80778937e-05, 7.74920732e-03,
           -1.29534695e-02, -1.44997425e-02, 3.00848205e-03, -1.33561785e-04,
           7.31927902e-03, -2.24683899e-03, -6.27679843e-03, -5.35295857e-03,
           -5.39031485e-03, -4.90641687e-05, 4.03603073e-03, -1.08133641e-03,
           9.59445070e-03, 9.81783494e-03, 8.77558347e-03, -5.13678743e-03,
           7.19959754e-03, 3.93835502e-03, -6.01979066e-03, 6.13247836e-03,
           1.39782019e-03, 4.60287556e-04, 1.04263611e-02, -9.61792190e-03,
           -1.02399308e-02, 8.54056142e-03, -1.22422148e-02, 6.58972748e-03,
           3.18149826e-03, -2.79453350e-03, -9.98417381e-04, 1.77927073e-02,
           -2.28664111e-02, -2.73113251e-02, 6.44177478e-03, -5.66864444e-04,
           1.58752780e-02, 2.18148530e-03, -1.31809842e-02, -9.98921506e-03,
           -9.63711366e-03, 1.11398206e-03, 4.28507291e-03, -3.02007422e-04,
           1.06751733e-02, 1.15796775e-02, 1.35387452e-02, -1.02765551e-02,
           1.11750513e-02, 4.31185029e-03, -1.04119312e-02, 8.54373723e-03,
           4.97616245e-04, -3.82199232e-03, 2.10159980e-02, -1.68744288e-02]]
      # pylint: enable=bad-whitespace
      # pyformat: enable
      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      self.assertAllClose(expected_enc_out, enc_out_sum_val)

  def testForwardPassWithStackingAfterMiddleLayer(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.stacking_layer_tpl.left_context = 1
      p.stacking_layer_tpl.right_context = 0
      p.stacking_layer_tpl.stride = 2
      p.layer_index_before_stacking = 0
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out, 0)

      self.evaluate(tf.global_variables_initializer())

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_enc_out = [
          [0.00102275, -0.02697385, 0.01709868, -0.00939053, -0.01576837,
           0.0070826, -0.00626193, 0.01143604, -0.01742513, -0.00529445,
           0.00284249, -0.01362027, -0.00490865, 0.0216262, -0.01344598,
           -0.00460993, -0.01329017, 0.01379208, -0.00850593, 0.0193335,
           0.01134925, -0.00131254, 0.00375953, -0.00588882, 0.01347932,
           -0.00252493, 0.01274828, 0.01027388, 0.02657663, 0.02644286,
           0.0286899, -0.00833998],
          [-0.01801126, 0.0115137, 0.01355767, 0.00113954, 0.00986663,
           -0.0128988, 0.00794239, -0.00524312, 0.00246279, -0.00575782,
           -0.00213567, -0.01528412, 0.00186096, 0.00253562, -0.00411006,
           -0.00390748, -0.01001569, -0.00344393, -0.01211706, 0.00387725,
           0.02194905, 0.02578988, -0.00255773, 0.00690117, 0.00976908,
           0.01935913, 0.01131854, 0.0013859, -0.01567556, 0.01858256,
           0.02251371, -0.0185001]]
      # pylint: enable=bad-whitespace
      # pyformat: enable
      enc_out_sum_val = enc_out_sum.eval()
      print('enc_out_sum_val', np.array_repr(enc_out_sum_val))
      self.assertAllClose(expected_enc_out, enc_out_sum_val)

  def testForwardPassWithHighwaySkip(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.residual_start = 1
      p.residual_stride = 2
      p.highway_skip = True
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      self.evaluate(tf.global_variables_initializer())

      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      test_utils.CompareToGoldenSingleFloat(
          self, 65.77313995361328, enc_out_sum_val)


  def testForwardPassWithExtraPerLayerOutputs(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.num_conv_lstm_layers = 1
      p.extra_per_layer_outputs = True
      enc_out = self._ForwardPass(p)
      regular_encoded_sum = tf.reduce_sum(enc_out.encoded)
      conv_0_encoded_sum = tf.reduce_sum(enc_out.conv_0.encoded)
      conv_1_encoded_sum = tf.reduce_sum(enc_out.conv_1.encoded)
      conv_lstm_0_encoded_sum = tf.reduce_sum(enc_out.conv_lstm_0.encoded)
      rnn_0_encoded_sum = tf.reduce_sum(enc_out.rnn_0.encoded)

      self.evaluate(tf.global_variables_initializer())

      # pyformat: disable
      self.assertAllEqual(tf.shape(enc_out.conv_0.encoded).eval(),
                          [13, 2, 8, 6])
      self.assertAllEqual(tf.shape(enc_out.conv_0.padding).eval(),
                          [13, 2])
      self.assertAllEqual(tf.shape(enc_out.conv_1.encoded).eval(),
                          [7, 2, 4, 6])
      self.assertAllEqual(tf.shape(enc_out.conv_1.padding).eval(),
                          [7, 2])
      self.assertAllEqual(tf.shape(enc_out.conv_lstm_0.encoded).eval(),
                          [7, 2, 4, 6])
      self.assertAllEqual(tf.shape(enc_out.conv_lstm_0.padding).eval(),
                          [7, 2])
      self.assertAllEqual(tf.shape(enc_out.rnn_0.encoded).eval(),
                          [7, 2, 32])
      self.assertAllEqual(tf.shape(enc_out.rnn_0.padding).eval(),
                          [7, 2])
      self.assertAllEqual(tf.shape(enc_out.encoded).eval(),
                          [7, 2, 32])
      self.assertAllEqual(tf.shape(enc_out.padding).eval(),
                          [7, 2])
      # pyformat: enable

      test_utils.CompareToGoldenSingleFloat(self, 371.75390625,
                                            conv_0_encoded_sum.eval())
      test_utils.CompareToGoldenSingleFloat(self, 92.5332946777,
                                            conv_1_encoded_sum.eval())
      test_utils.CompareToGoldenSingleFloat(self, 80.975112915,
                                            conv_lstm_0_encoded_sum.eval())
      test_utils.CompareToGoldenSingleFloat(self, 10.9648704529,
                                            rnn_0_encoded_sum.eval())
      test_utils.CompareToGoldenSingleFloat(self, 0.0522322505713,
                                            regular_encoded_sum.eval())


if __name__ == '__main__':
  tf.test.main()
