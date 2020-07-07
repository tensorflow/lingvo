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
"""Tests for mt.layers."""

import lingvo.compat as tf
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.mt import layers as mt_layers
import numpy as np


NUMPY_RANDOM_SEED = 505837249


class LayersTest(test_utils.TestCase):

  def _TransformerParams(self, layer=mt_layers.TransformerStack):
    model_dim = 2
    params = layer.Params()
    params.name = 'transformer'
    params.model_dim = model_dim
    params.num_transformer_layers = 1
    # Note: hidden_dim % num_attention_heads == 0
    params.transformer_tpl.tr_atten_tpl.num_attention_heads = 1
    params.transformer_tpl.tr_fflayer_tpl.hidden_dim = model_dim
    params.random_seed = 0
    return params

  def _ContextualTransformerParams(self,
                                   layer=mt_layers.TransformerStack):
    params = self._TransformerParams(layer)
    params.has_aux_attention = True
    return params

  def _MaskedTransformerParams(self,
                               layer=mt_layers.TransformerStack):
    params = self._TransformerParams(layer)
    params.mask_self_atten = True
    return params

  def _TransformerStackFProp(self, dtype, fprop_dtype, layer):
    # time = 2,
    batch = 3
    tf.flags.FLAGS.tpu_compatible = True
    with self.session(use_gpu=False):
      params = self._TransformerParams(layer=layer)
      params.dtype = dtype
      params.fprop_dtype = fprop_dtype
      xformer = layer(params)

      input_arr = np.array(
          [
              [[0, 1]] * batch,
              [[1, -1]] * batch,
          ], dtype=int)
      paddings_arr = np.array([[0] * batch, [0] * batch], dtype=int)
      inputs = tf.constant(input_arr.tolist(), dtype=fprop_dtype)
      paddings = tf.constant(paddings_arr.tolist(), dtype=fprop_dtype)

      output, _, _ = xformer.FProp(xformer.theta, inputs, paddings)

      self.evaluate(tf.global_variables_initializer())
      output = self.evaluate(output)

      self.assertAllCloseAccordingToType(
          [[[2.761079, -3.756719]] * batch, [[-2.623049, 3.3679538]] * batch],
          output)

  def testTransformerStackFPropFp32Fp32(self):
    self._TransformerStackFProp(tf.float32, tf.float32,
                                mt_layers.TransformerStack)

  def testTransformerStackAlternateLayers(self):
    batch = 3
    tf.flags.FLAGS.tpu_compatible = True
    with self.session(use_gpu=False):
      model_dim = 2
      num_transformer_layers = 2
      transformer_tpl = layers_with_attention.TransformerLayer.Params()
      transformer_tpl.tr_atten_tpl.num_attention_heads = 1
      transformer_tpl.tr_fflayer_tpl.hidden_dim = 2

      params = mt_layers.TransformerStack.Params().Set(
          name='transformer',
          model_dim=model_dim,
          num_transformer_layers=num_transformer_layers,
          transformer_tpl=[
              transformer_tpl.Copy() for _ in range(num_transformer_layers)
          ],
          random_seed=123456)

      xformer = mt_layers.TransformerStack(params)
      input_arr = np.array([
          [[0, 1]] * batch,
          [[1, -1]] * batch,
      ], dtype=int)
      paddings_arr = np.array([[0] * batch, [0] * batch], dtype=int)
      inputs = tf.constant(
          input_arr.tolist(), dtype=py_utils.FPropDtype(params))
      paddings = tf.constant(
          paddings_arr.tolist(), dtype=py_utils.FPropDtype(params))
      output, _, _ = xformer.FProp(xformer.theta, inputs, paddings)

      self.evaluate(tf.global_variables_initializer())
      output = self.evaluate(output)
      print(repr(output))
      self.assertAllCloseAccordingToType(
          np.array([[[-0.940543, 1.479253]] * batch,
                    [[-0.413938, -2.550903]] * batch]), output)

  def testTransformerStackFPropWithPackedInputs(self):
    # batch = 2. time = 2, depth = 2
    with self.session(use_gpu=True):
      with tf.variable_scope('packing_test', reuse=tf.AUTO_REUSE):
        params = self._TransformerParams()
        xformer = mt_layers.TransformerStack(params)
        packed_params = params.Copy()
        packed_params.packed_input = True
        xformer_packed = mt_layers.TransformerStack(packed_params)

        input_arr = np.array([[[0, 1], [1, -1]], [[1, 2], [-2, -1]]], dtype=int)
        paddings_arr = np.array([[0, 0], [0, 0]], dtype=int)
        seg_id_arr = np.array([[0, 1, 0, 1]], dtype=int)

        inputs = tf.constant(input_arr.tolist(), dtype=tf.float32)
        paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)
        inputs_packed = tf.reshape(inputs, [-1, 1, 2])
        paddings_packed = tf.reshape(paddings, [-1, 1])
        seg_id = tf.transpose(
            tf.constant(seg_id_arr.tolist(), dtype=tf.float32))

        output, _, _ = xformer.FProp(xformer.theta, inputs, paddings, seg_id)

        output_packed, _, _ = xformer_packed.FProp(
            xformer_packed.theta, inputs_packed, paddings_packed, seg_id)
        output_packed = tf.reshape(output_packed, tf.shape(output))

        self.evaluate(tf.global_variables_initializer())
        output, output_packed = self.evaluate([output, output_packed])

        self.assertAllClose(output_packed, output)

  def testTransparentTransformerStackTrainFProp(self):
    # time = 2, batch = 1
    with self.session(use_gpu=True):
      params = self._TransformerParams()
      params.is_transparent = True
      params.num_transparent_outputs = 2

      xformer = mt_layers.TransformerStack(params)

      input_arr = np.array([[[0, 1]], [[1, -1]]], dtype=int)
      paddings_arr = np.array([[0], [0]], dtype=int)

      inputs = tf.constant(input_arr.tolist(), dtype=tf.float32)
      paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)

      self.evaluate(tf.global_variables_initializer())
      outputs, _, _ = xformer.FPropDefaultTheta(inputs, paddings)
      out_1, out_2 = self.evaluate(outputs)
      self.assertAllClose([[[1.38054, -1.37836]], [[-0.811525, 1.183977]]],
                          out_1)
      self.assertAllClose([[[1.38054, -1.37836]], [[-0.811525, 1.183977]]],
                          out_2)

  def testTransparentTransformerStackEvalFProp(self):
    # time = 2, batch = 1
    with self.session(use_gpu=True), self.SetEval(True):
      params = self._TransformerParams()
      params.is_transparent = True
      params.num_transparent_outputs = 2

      xformer = mt_layers.TransformerStack(params)

      input_arr = np.array([[[0, 1]], [[1, -1]]], dtype=int)
      paddings_arr = np.array([[0], [0]], dtype=int)

      inputs = tf.constant(input_arr.tolist(), dtype=tf.float32)
      paddings = tf.constant(paddings_arr.tolist(), dtype=tf.float32)

      self.evaluate(tf.global_variables_initializer())
      outputs, _, _ = xformer.FPropDefaultTheta(inputs, paddings)
      out = self.evaluate(outputs)
      self.assertAllClose([[[1.38054, -1.37836]], [[-0.811525, 1.183977]]],
                          out[:, :, :, 0])
      self.assertAllClose([[[1.38054, -1.37836]], [[-0.811525, 1.183977]]],
                          out[:, :, :, 1])

  def _TransformerSingleSourceInputs(self, depth=3, dtype=tf.float32):
    np.random.seed(NUMPY_RANDOM_SEED)
    source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(5)])
    source_padding = tf.transpose(
        tf.constant([[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]], dtype=dtype))
    aux_source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)])
    aux_source_paddings = tf.transpose(
        tf.constant(
            [[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]], dtype=dtype))
    return source_vecs, source_padding, aux_source_vecs, aux_source_paddings

  def _TransformerMultiSourceInputs(self, depth=3, dtype=tf.float32):
    np.random.seed(NUMPY_RANDOM_SEED)
    src_names = ['en1', 'en2', 'de']
    slens = [11, 10, 9]
    sbatch = 3
    tlen = 5
    source_vecs = tf.constant(
        np.random.uniform(size=(tlen, sbatch*2, depth)), dtype)
    source_padding = tf.constant(np.zeros([tlen, sbatch*2, 1]), dtype)
    aux_source_vecs = py_utils.NestedMap()
    aux_source_paddings = py_utils.NestedMap()
    for slen, sname in zip(slens, src_names):
      aux_source_vecs[sname] = tf.constant(
          np.random.uniform(size=[slen, sbatch, depth]), dtype)
      aux_source_paddings[sname] = tf.constant(np.zeros([slen, sbatch]), dtype)
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings)

  def _ExpectedSingleSourceResults(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_ctx = [
        [[-1.2441386 ,  0.79554689,  0.92997617, -0.11617106],
         [-0.35435563, -1.3054322 ,  1.22436929,  0.78819978]],
        [[ 0.25911736,  1.59861779, -0.79328805, -0.69870573],
         [-0.07478544, -0.97613871, -0.28578228,  1.68864977]],
        [[ 1.70767629,  0.47948313, -1.04493523, -0.77614671],
         [-0.7304405 , -0.97698098,  1.06615043,  0.99392742]],
        [[-0.8379035 , -0.4325709 ,  1.64835262, -0.01260686],
         [ 1.9789927 , -0.0406196 , -0.70901859, -0.87701303]],
        [[-0.31843656, -0.88714617, -0.10433094,  1.6746937 ],
         [-0.75752175,  0.063187  , -0.50349152,  1.54967213]]]
    expected_probs = [
        [[ 0.2506268 ,  0.,  0.2491456 ,  0.,  0.24775073, 0., 0.25247693],
         [ 0.,  0.33213311,  0.,  0.3369872 ,  0., 0.33087969, 0.        ]],
        [[ 0.2478824 ,  0.,  0.24961777,  0.,  0.24944726, 0., 0.25305259],
         [ 0.,  0.32697931,  0.,  0.33510387,  0., 0.33791685, 0.        ]],
        [[ 0.24868786,  0.,  0.2501137 ,  0.,  0.25135297, 0., 0.24984553],
         [ 0.,  0.33173615,  0.  , 0.3365739,  0., 0.33168989, 0.        ]],
        [[ 0.25286067,  0.,  0.24919504,  0.,  0.24831679, 0., 0.24962756],
         [ 0.,  0.33384994,  0., 0.33144677,   0., 0.33470333, 0.        ]],
        [[ 0.24830991,  0.,  0.25117707,  0.,  0.25028574, 0., 0.25022724],
         [ 0.,  0.32832602,  0., 0.33358034,   0., 0.33809367, 0.        ]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    return expected_ctx, expected_probs

  def testTransformerAttentionLayerReference(self):
    depth = 4
    p = layers_with_attention.TransformerAttentionLayer.Params()
    p.name = 'transformer_atten'
    p.source_dim = depth
    p.is_masked = False
    p.num_attention_heads = 2
    p.atten_tpl.params_init = py_utils.WeightInit.Gaussian(0.1, 12345)
    transformer_atten_ref = layers_with_attention.TransformerAttentionLayer(p)

    (query_vec, _, aux_vecs,
     aux_paddings) = self._TransformerSingleSourceInputs(depth)

    ctx_ref, probs_ref = transformer_atten_ref.FPropDefaultTheta(
        query_vec, aux_paddings, aux_vecs)

    expected_ctx, expected_probs = self._ExpectedSingleSourceResults()
    with self.session(use_gpu=True):
      self.evaluate(tf.global_variables_initializer())
      actual_ctx_ref, actual_probs_ref = self.evaluate([ctx_ref, probs_ref])
      tf.logging.info(np.array_repr(actual_ctx_ref))
      tf.logging.info(np.array_repr(actual_probs_ref))
      self.assertAllClose(expected_ctx, actual_ctx_ref)
      self.assertAllClose(expected_probs, actual_probs_ref)

  def testContextualTransformerStackFProp(self):
    # time = 2,
    batch = 3
    dtype = tf.float32
    fprop_dtype = tf.float32
    layer = mt_layers.TransformerStack
    tf.flags.FLAGS.tpu_compatible = True
    with self.session(use_gpu=False):
      params = self._ContextualTransformerParams(layer=layer)
      params.dtype = dtype
      params.fprop_dtype = fprop_dtype
      xformer = layer(params)

      input_arr = np.array([
          [[0, 1]] * batch,
          [[1, -1]] * batch,
      ], dtype=int)

      context_arr = np.array([
          [[0, 1]] * batch,
          [[1, -1]] * batch,
          [[-1, 1]] * batch,
      ],
                             dtype=int)
      paddings_arr = np.array([[0] * batch, [0] * batch], dtype=int)
      context_paddings_arr = np.array([[0] * batch, [0] * batch, [0] * batch],
                                      dtype=int)
      inputs = tf.constant(input_arr.tolist(), dtype=fprop_dtype)
      paddings = tf.constant(paddings_arr.tolist(), dtype=fprop_dtype)
      context = tf.constant(context_arr.tolist(), dtype=fprop_dtype)
      context_paddings = tf.constant(
          context_paddings_arr.tolist(), dtype=fprop_dtype)

      output, _, _ = xformer.FProp(
          xformer.theta,
          inputs,
          paddings,
          aux_vecs=context,
          aux_paddings=context_paddings)

      self.evaluate(tf.global_variables_initializer())
      output = self.evaluate(output)
      self.assertAllCloseAccordingToType(
          [[[-2.622666, 3.367474]] * batch, [[2.719156, -3.707336]] * batch],
          output)

  def testMaskedTransformerStackFProp(self):
    batch = 3
    dtype = tf.float32
    fprop_dtype = tf.float32
    layer = mt_layers.TransformerStack
    tf.flags.FLAGS.tpu_compatible = True
    with self.session(use_gpu=False):
      params = self._MaskedTransformerParams(layer=layer)
      params.dtype = dtype
      params.fprop_dtype = fprop_dtype
      xformer = layer(params)

      input_arr = np.array([
          [[0, 1]] * batch,
          [[1, -1]] * batch,
      ], dtype=int)

      paddings_arr = np.array([[0] * batch, [0] * batch], dtype=int)

      inputs = tf.constant(input_arr.tolist(), dtype=fprop_dtype)
      paddings = tf.constant(paddings_arr.tolist(), dtype=fprop_dtype)

      output, _, _ = xformer.FProp(xformer.theta, inputs, paddings)

      self.evaluate(tf.global_variables_initializer())
      output = self.evaluate(output)
      self.assertAllClose(
          [[[2.761103, -3.756748]] * batch, [[-2.623049, 3.367954]] * batch],
          output)


if __name__ == '__main__':
  tf.test.main()
