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


class LayersWithGPipeTest(tf.test.TestCase):

  def _TransformerParams(self,
                         num_decoder_layers=0,
                         num_encoder_layers=4,
                         num_splits=1,
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
    params.num_splits = num_splits
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
                                                    num_splits=1,
                                                    num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=4,
          num_encoder_layers=4,
          num_splits=num_splits,
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

  def _testGPipeTransformerStackFProp(self, num_splits=1, num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          num_splits=num_splits, num_micro_batches=num_micro_batches)
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

  def _testGPipeTransformerFPropPackedInput(self, num_splits=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      with tf.variable_scope('transformer_test', reuse=tf.AUTO_REUSE):
        params = self._TransformerParams(num_splits=num_splits)
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
                                                      num_splits=1,
                                                      num_micro_batches=1):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          num_splits=num_splits,
          num_micro_batches=num_micro_batches,
          num_decoder_layers=3,
          num_encoder_layers=1)
      params.is_transparent = True
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

  def testDeterministicDropoutInsideFunctionalWhile(self):
    with self.session() as sess:
      cells = FeatureExtractionLayer.Params().Set(
          name='cell',
          sub=[
              DeterministicDropoutLayer.Params().Set(
                  name='dropout', keep_prob=0.7)
          ])
      p = PipeliningLayer.Params().Set(name='pipe', cell_tpl=[cells])
      x = tf.ones([2, 3], dtype=tf.float32)
      model = p.cls(p)
      y = model.FPropDefaultTheta(x)
      py_utils.GetOrCreateGlobalStep()
      tf.global_variables_initializer().run()
      y_val = sess.run(y)
      self.assertAllClose([
          [1.0 / 0.7, 1.0 / 0.7, 1.0 / 0.7],
          [0.0, 0.0, 1.0 / 0.7],
      ], y_val)
      self.assertAllClose(5.7142859, np.sum(y_val))

  def testGPipeTransformerStackTrainTransparentFPropEval(self):
    # time = 2,
    batch = 4
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=3, num_encoder_layers=1)
      params.is_transparent = True
      params.is_eval = True

      xformer = GPipeTransformerStack(params)

      inputs, paddings, _, _ = self._random_inputs(batch=batch)

      tf.global_variables_initializer().run()
      enc_outputs = xformer.EncoderFPropDefaultTheta(inputs, paddings)
      enc_out = sess.run(enc_outputs)
      self.assertAllClose(enc_out,
                          [[[[-0.27896273] * 3, [1.46589136] * 3]] * batch,
                           [[[1.03141928] * 3, [-0.847896] * 3]] * batch])

  def _testGPipeTransformerDecoderStackFProp(self,
                                             num_splits=1,
                                             num_micro_batches=1):
    batch = 4
    tf.flags.FLAGS.tpu_compatible = True
    with self.session() as sess:
      params = self._TransformerParams(
          num_decoder_layers=4,
          num_encoder_layers=0,
          num_splits=num_splits,
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
        num_splits=2, num_micro_batches=2)

  def testGPipeTransformerFPropPackedInput(self):
    self._testGPipeTransformerFPropPackedInput()

  def testGPipeTransformerFPropPackedInputTwoSplits(self):
    self._testGPipeTransformerFPropPackedInput(num_splits=2)

  def testGPipeTransformerStackFPropNoSplit(self):
    self._testGPipeTransformerStackFProp()

  def testGPipeTransformerStackFPropNoSplitTwoMicroBatches(self):
    self._testGPipeTransformerStackFProp(num_micro_batches=2)

  def testGPipeTransformerStackFPropNoSplitFourMicroBatches(self):
    self._testGPipeTransformerStackFProp(num_micro_batches=4)

  def testGPipeTransformerStackFPropTwoSplits(self):
    self._testGPipeTransformerStackFProp(num_splits=2)

  def testGPipeTransformerStackFPropTwoSplitsTwoMicroBatches(self):
    self._testGPipeTransformerStackFProp(num_splits=2, num_micro_batches=2)

  def testGPipeTransformerStackFPropTwoSplitsFourMicroBatches(self):
    self._testGPipeTransformerStackFProp(num_splits=2, num_micro_batches=4)

  def testGPipeTransformerStackFPropFourSplits(self):
    self._testGPipeTransformerStackFProp(num_splits=4)

  def testGPipeTransformerStackFPropFourSplitsTwoMicroBatches(self):
    self._testGPipeTransformerStackFProp(num_splits=4, num_micro_batches=2)

  def testGPipeTransformerStackFPropFourSplitsFourMicroBatches(self):
    self._testGPipeTransformerStackFProp(num_splits=4, num_micro_batches=4)

  def testGPipeTransformerDecoderStackFPropNoSplit(self):
    self._testGPipeTransformerDecoderStackFProp()

  def testGPipeTransformerDecoderStackFPropNoSplitTwoMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(num_micro_batches=2)

  def testGPipeTransformerDecoderStackFPropNoSplitFourMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(num_micro_batches=4)

  def testGPipeTransformerDecoderStackFPropTwoSplits(self):
    self._testGPipeTransformerDecoderStackFProp(num_splits=2)

  def testGPipeTransformerDecoderStackFPropTwoSplitsTwoMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(
        num_splits=2, num_micro_batches=2)

  def testGPipeTransformerDecoderStackFPropTwoSplitsFourMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(
        num_splits=2, num_micro_batches=4)

  def testGPipeTransformerDecoderStackFPropFourSplits(self):
    self._testGPipeTransformerDecoderStackFProp(num_splits=4)

  def testGPipeTransformerDecoderStackFPropFourSplitsTwoMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(
        num_splits=4, num_micro_batches=2)

  def testGPipeTransformerDecoderStackFPropFourSplitsFourMicroBatches(self):
    self._testGPipeTransformerDecoderStackFProp(
        num_splits=4, num_micro_batches=4)

  def testGPipeTransformerStackTrainTransparentFProp(self):
    self._testGPipeTransformerStackTrainTransparentFProp()

  def testGPipeTransformerStackTrainTransparentFPropMB2(self):
    self._testGPipeTransformerStackTrainTransparentFProp(num_micro_batches=2)

  def testGPipeTransformerStackTrainTransparentFPropMB4(self):
    self._testGPipeTransformerStackTrainTransparentFProp(num_micro_batches=4)

  def testGPipeTransformerStackTrainTransparentFPropTwoSplitsMB2(self):
    self._testGPipeTransformerStackTrainTransparentFProp(
        num_splits=2, num_micro_batches=2)

  def testGPipeTransformerStackTrainTransparentFPropTwoSplits2MB4(self):
    self._testGPipeTransformerStackTrainTransparentFProp(
        num_splits=2, num_micro_batches=4)

  def testGPipeTransformerStackTrainTransparentFPropFourSplitsMB4(self):
    self._testGPipeTransformerStackTrainTransparentFProp(
        num_splits=4, num_micro_batches=4)


if __name__ == '__main__':
  tf.test.main()
