# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for layers_with_gpipe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from lingvo.core import py_utils
from lingvo.core.gpipe import FeatureExtractionLayer
from lingvo.core.gpipe import PipeliningLayer
from lingvo.core.layers_with_gpipe import DeterministicDropoutLayer
from lingvo.core.layers_with_gpipe import GPipeTransformerLayer
from lingvo.core.layers_with_gpipe import GPipeTransformerStack


class GPipeTransformerLayerTest(tf.test.TestCase):

  def _testInputs(self, depth=3, dtype=tf.float32):
    np.random.seed(505837249)
    source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(5)])
    source_padding = tf.constant([[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]],
                                 dtype=dtype)
    aux_source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)])
    aux_source_paddings = tf.constant(
        [[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]], dtype=dtype)
    source_padding = tf.transpose(source_padding)
    aux_source_paddings = tf.transpose(aux_source_paddings)
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings)

  def testTransformerLayerExtendStep(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      np.random.seed(6348575)
      p = GPipeTransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = GPipeTransformerLayer(p)

      (source_vecs, _, aux_vecs, aux_paddings) = self._testInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      output1 = transformer.FPropDefaultTheta(
          aux_vecs, aux_paddings, source_vecs, source_padding, None, None)
      h1 = output1[2]

      h2 = []
      cached_source_vecs = tf.zeros([0, 2, 4])
      cached_source_contexts = tf.zeros([0, 2, 4])
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        h, _, prefix_states = transformer.ExtendStep(
            transformer.theta, source_vecs[i, :, :], prefix_states, aux_vecs,
            aux_paddings)
        h2.append(h)

      h2 = tf.stack(h2)

      tf.global_variables_initializer().run()
      h1_v, h2_v = sess.run([h1, h2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(h1_v[2][1],
                          [1.10429943, -1.64884555, 0.15726769, -0.00250494])


class GPipeTransformerStackTest(tf.test.TestCase):
  """Tests for GPipeTransformerStack layer."""

  def _TransformerParams(self,
                         num_decoder_layers=0,
                         num_encoder_layers=4,
                         splits=1,
                         num_micro_batches=1):
    model_dim = 2
    params = GPipeTransformerStack.Params()
    params.name = 'transformer'
    params.model_dim = model_dim
    params.num_decoder_layers = num_decoder_layers
    params.decoder_tpl.tr_atten_tpl.num_attention_heads = 1
    params.decoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
    params.num_encoder_layers = num_encoder_layers
    params.encoder_tpl.tr_atten_tpl.num_attention_heads = 1
    params.encoder_tpl.tr_fflayer_tpl.hidden_dim = model_dim
    params.num_micro_batches = num_micro_batches
    params.splits = splits
    params.random_seed = 0
    return params

  def _random_inputs(self, batch):
    input_arr = np.array([
        [[0, 1]] * batch,
        [[1, -1]] * batch,
    ])
    paddings_arr = np.array([[0] * batch] * 2)
    tgt_input_arr = np.array([
        [[1, 2]] * batch,
        [[1, -1]] * batch,
        [[2, 1]] * batch,
    ])
    tgt_paddings_arr = np.array([[0] * batch] * 3)
    inputs = tf.constant(input_arr.tolist(), dtype=tf.float32)
    paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
    tgt_inputs = tf.constant(tgt_input_arr.tolist(), dtype=tf.float32)
    tgt_paddings = tf.constant(tgt_paddings_arr.tolist(), dtype=tf.float32)
    return inputs, paddings, tgt_inputs, tgt_paddings

  def _testGPipeTransformerEncoderFPropDefaultTheta(self,
                                                    splits=1,
                                                    num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=4,
          num_encoder_layers=4,
          splits=splits,
          num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      params.fprop_dtype = tf.float32
      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch)

      output = xformer.EncoderFPropDefaultTheta(inputs, paddings)

      tf.global_variables_initializer().run()
      output = sess.run(output)

      self.assertAllCloseAccordingToType([[[0.21085747, 0.60925347]] * batch,
                                          [[0.21085747, 0.60925347]] * batch],
                                         output)

  def _testGPipeTransformerStackFProp(self, splits=1, num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          splits=splits, num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      params.fprop_dtype = tf.float32
      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch)

      output = xformer.FProp(xformer.theta, inputs, paddings)

      tf.global_variables_initializer().run()
      output = sess.run(output)

      self.assertAllCloseAccordingToType([[[0.21085747, 0.60925347]] * batch,
                                          [[0.21085747, 0.60925347]] * batch],
                                         output)

  def _testGPipeTransformerFPropPackedInput(self, splits=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = self._TransformerParams(splits=splits)
        params.dtype = tf.float32
        params.fprop_dtype = tf.float32
        packed_params = params.Copy()
        packed_params.packed_input = True
        xformer = GPipeTransformerStack(params)
        packed_xformer = GPipeTransformerStack(packed_params)
        # Prepare inputs
        inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs(batch)
        packed_inputs = tf.reshape(inputs, [-1, 1, 2])
        packed_tgt_inputs = tf.reshape(tgt_inputs, [-1, 1, 2])
        packed_paddings = tf.reshape(paddings, [-1, 1])
        packed_tg_paddings = tf.reshape(tgt_paddings, [-1, 1])
        segment_ids = tf.transpose(
            tf.constant([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=tf.float32))
        tgt_segment_id = tf.transpose(
            tf.constant([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]],
                        dtype=tf.float32))

        output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                               tgt_paddings)
        packed_output = packed_xformer.FProp(
            packed_xformer.theta, packed_inputs, packed_paddings,
            packed_tgt_inputs, packed_tg_paddings, segment_ids, tgt_segment_id)
        packed_output = tf.reshape(packed_output, output.shape)

        tf.global_variables_initializer().run()
        output, packed_output = sess.run([output, packed_output])
        self.assertAllClose(output, packed_output)

  def _testGPipeTransformerStackTrainTransparentFProp(self,
                                                      splits=1,
                                                      num_micro_batches=1):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          splits=splits,
          num_micro_batches=num_micro_batches,
          num_decoder_layers=3,
          num_encoder_layers=1)
      params.is_transparent = True
      params.num_transparent_outputs = 3
      params.transparent_merger_dropout_prob = 0.0
      xformer = GPipeTransformerStack(params)

      inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs(
          batch=batch)
      py_utils.GetOrCreateGlobalStep()
      tf.set_random_seed(1234)
      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      dec_output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                                 tgt_paddings)
      enc_out_1, enc_out_2, enc_out_3 = sess.run(enc_outputs)
      dec_out = sess.run(dec_output)
      self.assertAllClose(enc_out_1, enc_out_2)
      self.assertAllClose(enc_out_2, enc_out_3)
      self.assertAllClose(enc_out_1, [[[-0.27896273, 1.46589136]] * batch,
                                      [[1.03141928, -0.847896]] * batch])
      self.assertAllClose(
          dec_out,
          [[[2.926736, -4.090812]] * batch, [[-1.69508219, 1.75891459]] * batch,
           [[-1.6950829, 1.75891507]] * batch])

  def _testGPipeTransformerStackTrainEncoderTransparentFProp(
      self, splits=1, num_micro_batches=1):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          splits=splits,
          num_micro_batches=num_micro_batches,
          num_decoder_layers=2,
          num_encoder_layers=2)
      params.is_transparent = True
      params.num_transparent_outputs = 1
      params.transparent_merger_dropout_prob = 0.0
      xformer = GPipeTransformerStack(params)

      inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs(
          batch=batch)
      py_utils.GetOrCreateGlobalStep()
      tf.set_random_seed(1234)
      tf.global_variables_initializer().run()
      enc_output = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      dec_output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                                 tgt_paddings)
      enc_out = sess.run(enc_output)
      dec_out = sess.run(dec_output)
      self.assertAllClose(
          enc_out,
          [[[-0.118476, 1.031626]] * batch, [[0.643884, -1.02581167]] * batch])
      self.assertAllClose(
          dec_out,
          [[[-2.8764534, 1.00808454]] * batch, [[1.02129495, -0.78406084]] *
           batch, [[1.02129495, -0.78406084]] * batch])

  def testGPipeTransformerStackTrainTransparentFPropEval(self):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=3, num_encoder_layers=1)
      params.is_transparent = True
      params.num_transparent_outputs = 3
      params.is_eval = True

      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch=batch)

      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      enc_out = sess.run(enc_outputs)
      self.assertAllClose(enc_out,
                          [[[[-0.27896273] * 3, [1.46589136] * 3]] * batch,
                           [[[1.03141928] * 3, [-0.847896] * 3]] * batch])

  def testGPipeTransformerStackTrainEncoderTransparentFPropEval(self):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=3, num_encoder_layers=3)
      params.is_transparent = True
      params.num_transparent_outputs = 1
      params.is_eval = True

      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch=batch)

      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      enc_out = sess.run(enc_outputs)
      self.assertAllClose(enc_out, [[[0.18823329, 0.71548849]] * batch,
                                    [[0.76032472, -0.82791042]] * batch])

  def _testGPipeTransformerDecoderStackFProp(self,
                                             splits=1,
                                             num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=4,
          num_encoder_layers=0,
          splits=splits,
          num_micro_batches=num_micro_batches)
      params.dtype = tf.float32
      params.fprop_dtype = tf.float32
      xformer = GPipeTransformerStack(params)

      inputs, paddings, tgt_inputs, tgt_paddings = self._random_inputs(batch)

      output = xformer.FProp(xformer.theta, inputs, paddings, tgt_inputs,
                             tgt_paddings)

      tf.global_variables_initializer().run()
      output_val = sess.run(output)
      self.assertAllCloseAccordingToType([[[1.03550637, -1.3199079]] * batch,
                                          [[-3.36382699, -0.74492991]] * batch,
                                          [[-3.36382723, -0.74492997]] * batch],
                                         output_val)

  def testGPipeTransformerEncoderFProp(self):
    self._testGPipeTransformerEncoderFPropDefaultTheta(
        splits=2, num_micro_batches=2)

  def testGPipeTransformerEncoderFPropWithManualSplits(self):
    self._testGPipeTransformerEncoderFPropDefaultTheta(
        splits=[4, 8], num_micro_batches=2)

  def testGPipeTransformerFPropPackedInput(self):
    self._testGPipeTransformerFPropPackedInput()

  def testGPipeTransformerFPropPackedInputTwoSplits(self):
    self._testGPipeTransformerFPropPackedInput(splits=2)

  def testGPipeTransformerStackFPropNoSplit(self):
    self._testGPipeTransformerStackFProp()

  def testGPipeTransformerStackFPropNoSplitTwoMicroBatches(self):
    self._testGPipeTransformerStackFProp(num_micro_batches=2)

  def testGPipeTransformerStackFPropNoSplitFourMicroBatches(self):
    self._testGPipeTransformerStackFProp(num_micro_batches=4)

  def testGPipeTransformerStackFPropTwoSplits(self):
    self._testGPipeTransformerStackFProp(splits=2)

  def testGPipeTransformerStackFPropTwoSplitsTwoMicroBatches(self):
    self._testGPipeTransformerStackFProp(splits=2, num_micro_batches=2)

  def testGPipeTransformerStackFPropTwoSplitsFourMicroBatches(self):
    self._testGPipeTransformerStackFProp(splits=2, num_micro_batches=4)

  def testGPipeTransformerStackFPropFourSplits(self):
    self._testGPipeTransformerStackFProp(splits=4)

  def testGPipeTransformerStackFPropFourSplitsTwoMicroBatches(self):
    self._testGPipeTransformerStackFProp(splits=4, num_micro_batches=2)

  def testGPipeTransformerStackFPropFourSplitsFourMicroBatches(self):
    self._testGPipeTransformerStackFProp(splits=4, num_micro_batches=4)

  def testGPipeTransformerDecoderStackFPropNoSplit(self):
    self._testGPipeTransformerDecoderStackFProp()

  def testGPipeTransformerDecoderStackFPropNoSplitTwoMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(num_micro_batches=2)

  def testGPipeTransformerDecoderStackFPropNoSplitFourMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(num_micro_batches=4)

  def testGPipeTransformerDecoderStackFPropTwoSplits(self):
    self._testGPipeTransformerDecoderStackFProp(splits=2)

  def testGPipeTransformerDecoderStackFPropTwoSplitsTwoMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(splits=2, num_micro_batches=2)

  def testGPipeTransformerDecoderStackFPropTwoSplitsFourMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(splits=2, num_micro_batches=4)

  def testGPipeTransformerDecoderStackFPropFourSplits(self):
    self._testGPipeTransformerDecoderStackFProp(splits=4)

  def testGPipeTransformerDecoderStackFPropFourSplitsTwoMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(splits=4, num_micro_batches=2)

  def testGPipeTransformerDecoderStackFPropFourSplitsFourMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(splits=4, num_micro_batches=4)

  def testGPipeTransformerStackTrainTransparentFProp(self):
    self._testGPipeTransformerStackTrainTransparentFProp()

  def testGPipeTransformerStackTrainTransparentFPropMB2(self):
    self._testGPipeTransformerStackTrainTransparentFProp(num_micro_batches=2)

  def testGPipeTransformerStackTrainTransparentFPropMB4(self):
    self._testGPipeTransformerStackTrainTransparentFProp(num_micro_batches=4)

  def testGPipeTransformerStackTrainTransparentFPropTwoSplitsMB2(self):
    self._testGPipeTransformerStackTrainTransparentFProp(
        splits=2, num_micro_batches=2)

  def testGPipeTransformerStackTrainTransparentFPropTwoManualSplitsMB2(self):
    self._testGPipeTransformerStackTrainTransparentFProp(
        splits=[3, 4], num_micro_batches=2)

  def testGPipeTransformerStackTrainTransparentFPropTwoSplits2MB4(self):
    self._testGPipeTransformerStackTrainTransparentFProp(
        splits=2, num_micro_batches=4)

  def testGPipeTransformerStackTrainTransparentFPropFourSplitsMB4(self):
    self._testGPipeTransformerStackTrainTransparentFProp(
        splits=4, num_micro_batches=4)

  def testGPipeTransformerStackTrainEncoderTransparentFProp(self):
    self._testGPipeTransformerStackTrainEncoderTransparentFProp()

  def testGPipeTransformerStackTrainEncoderTransparentFPropMB2(self):
    self._testGPipeTransformerStackTrainEncoderTransparentFProp(
        num_micro_batches=2)

  def testGPipeTransformerStackTrainEncoderTransparentFPropMB4(self):
    self._testGPipeTransformerStackTrainEncoderTransparentFProp(
        num_micro_batches=4)

  def testGPipeTransformerStackTrainEncoderTransparentFPropTwoSplitsMB2(self):
    self._testGPipeTransformerStackTrainEncoderTransparentFProp(
        splits=2, num_micro_batches=2)

  def testGPipeTransformerStackTrainEncoderTransparentFPropManualSplitMB2(self):
    self._testGPipeTransformerStackTrainEncoderTransparentFProp(
        splits=[1, 4], num_micro_batches=2)

  def testGPipeTransformerStackTrainEncoderTransparentFPropTwoSplits2MB4(self):
    self._testGPipeTransformerStackTrainEncoderTransparentFProp(
        splits=2, num_micro_batches=4)

  def testGPipeTransformerStackTrainEncoderTransparentFPropFourSplitsMB4(self):
    self._testGPipeTransformerStackTrainEncoderTransparentFProp(
        splits=4, num_micro_batches=4)


class DeterministicDropoutTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for DeterministicDropoutLayer."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'splits': 1,
          'num_micro_batches': 1
      },
      {
          'testcase_name': 'OneSplitTwoMicroBatches',
          'splits': 1,
          'num_micro_batches': 2
      },
      {
          'testcase_name': 'TwoSplitsOneMicroBatch',
          'splits': 2,
          'num_micro_batches': 1
      },
      {
          'testcase_name': 'TwoSplitsTwoMicroBatches',
          'splits': 2,
          'num_micro_batches': 2
      },
  )
  def testDropoutInRecurrent(self, splits=1, num_micro_batches=1):
    assert splits in [1, 2, 4]
    with self.session() as sess:
      tf.set_random_seed(12345)
      num_layers = 4
      py_utils.GetOrCreateGlobalStep()
      # Build a model with 4 dropout layers.
      layers = []
      for l in range(num_layers):
        layers.append(DeterministicDropoutLayer.Params().Set(
            name='dropout_{}'.format(l), keep_prob=0.7))
      # Divide the model into splits partitions.
      cell_tpl = []
      layers_per_split = num_layers // splits
      for i in range(splits):
        sub = layers[i * layers_per_split:(i + 1) * layers_per_split]
        cell_tpl.append(FeatureExtractionLayer.Params().Set(
            name='cell_{}'.format(i), sub=sub))
      # Parallelize partitions using pipeline.
      p = PipeliningLayer.Params().Set(
          name='pipeline',
          num_micro_batches=num_micro_batches,
          cell_tpl=cell_tpl)
      # Fake input
      x = tf.ones([2, 3])
      # Construct weights.
      w = tf.get_variable(
          'w', shape=[2, 3], initializer=tf.constant_initializer([[1] * 3] * 2))
      mdl = p.cls(p)
      y = mdl.FPropDefaultTheta(x * w)
      # Construct loss function such that gradients = final activation.
      loss = tf.reduce_sum(y)
      grads = py_utils.ComputeGradients(loss, py_utils.NestedMap(w=w))
      tf.global_variables_initializer().run()
      y_val = sess.run(y)
      grads_val = sess.run(grads)['w'][1]
      self.assertAllClose(y_val, grads_val)


if __name__ == '__main__':
  tf.test.main()
