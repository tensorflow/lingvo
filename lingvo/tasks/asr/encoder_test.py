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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.asr import encoder


class EncoderTest(test_utils.TestCase):

  def _EncoderParams(self, vn_config):
    p = encoder.AsrEncoder.Params()
    p.name = 'encoder'
    vn_config.scale = tf.constant(0.1)
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
    tf.set_random_seed(8372749040)
    stt_enc = encoder.AsrEncoder(p)
    batch = py_utils.NestedMap()
    batch.src_inputs = tf.random_normal([2, 20, 16, 3], seed=92837472)
    batch.paddings = tf.zeros([2, 20])
    return stt_enc.FPropDefaultTheta(batch)

  def testEncoderConstruction(self):
    vn_config = py_utils.VariationalNoiseParams(None, True, False)
    p = self._EncoderParams(vn_config)
    _ = encoder.AsrEncoder(p)

  def testForwardPass(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out, 0)
      tf.global_variables_initializer().run()

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

  def testForwardPassWithConvLSTM(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.num_conv_lstm_layers = 1
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      tf.global_variables_initializer().run()

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
      tf.global_variables_initializer().run()

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
      tf.global_variables_initializer().run()

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
      tf.global_variables_initializer().run()

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
      tf.global_variables_initializer().run()

      enc_out_sum_val = enc_out_sum.eval()
      print('expected enc_out_sum_val', enc_out_sum_val)
      test_utils.CompareToGoldenSingleFloat(
          self, 92.58077239990234, enc_out_sum_val)

  def testForwardPassWithHighwaySkip(self):
    with self.session(use_gpu=False):
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._EncoderParams(vn_config)
      p.residual_start = 1
      p.residual_stride = 2
      p.highway_skip = True
      enc_out = self._ForwardPass(p).encoded
      enc_out_sum = tf.reduce_sum(enc_out)
      tf.global_variables_initializer().run()

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

      tf.global_variables_initializer().run()

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
