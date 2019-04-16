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
# ==============================================================================
"""Tests for layers_with_attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from lingvo.core import attention
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core.test_utils import CompareToGoldenSingleFloat


class LayersWithAttentionTest(test_utils.TestCase):

  def testTransformerFeedForwardLayer(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      inputs = tf.random_normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers_with_attention.TransformerFeedForwardLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 3
      p.hidden_dim = 7
      transformer_fflayer = layers_with_attention.TransformerFeedForwardLayer(p)

      h = transformer_fflayer.FPropDefaultTheta(inputs, paddings)
      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(h)
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[-0.88366592, -0.05049637,  0.01003706],
           [-0.10550675,  1.68050027,  2.29110384]],
          [[-1.30083609, -0.40521634,  0.1911681 ],
           [ 1.2597878 ,  1.45850968,  1.58734488]],
          [[ 0.10373873, -0.2716777 ,  0.2314173 ],
           [ 0.46293864, -0.06359965,  1.20189023]],
          [[ 0.3673597 , -0.1691664 ,  0.78656065],
           [-1.51081395, -0.70281881, -0.9093715 ]],
          [[-1.04800868, -0.70610946, -0.35321558],
           [-1.92480004,  0.08361804,  0.62713993]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testTransformerFeedForwardLayerSpecOutDim(self):
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      inputs = tf.random_normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers_with_attention.TransformerFeedForwardLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 3
      p.output_dim = 5
      p.hidden_dim = 7
      transformer_fflayer = layers_with_attention.TransformerFeedForwardLayer(p)

      h = transformer_fflayer.FPropDefaultTheta(inputs, paddings)
      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(h)
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ 1.42697251,  0.79269135, -0.85500956, -0.8122285 , -1.56555367],
           [-1.7876718 ,  0.26025945, -3.18244219,  1.34756351,  0.25739765]],
          [[ 1.27962363,  0.88677615, -1.23556185, -1.06855559, -1.27293301],
           [ 0.89336467,  2.46229172,  0.11302143,  1.19385004, -2.37805009]],
          [[ 2.80146003, -0.66912627,  1.50160134, -2.30645609, -1.18872762],
           [ 1.61967182, -0.51639485,  0.24441491, -1.0871532 , -0.95539457]],
          [[ 2.03333473, -0.78205228,  0.71245927, -1.63276744, -0.91654319],
           [ 1.54542768, -0.30343491,  0.10666496, -1.67965126, -0.15671858]],
          [[ 1.60873222, -1.88402128,  0.79040933, -1.97199082,  0.4778356 ],
           [-0.13516766, -0.42583361, -1.86275542, -1.09650302,  0.83263111]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def _testTransformerAttentionLayerInputs(self, depth=3, dtype=tf.float32):
    np.random.seed(505837249)
    source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(5)])
    source_padding = tf.transpose(
        tf.constant([[0, 0, 1, 1, 0], [1, 0, 0, 0, 1]], dtype=dtype))
    aux_source_vecs = tf.stack(
        [tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)])
    aux_source_paddings = tf.transpose(
        tf.constant([[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]],
                    dtype=dtype))
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings)

  def testTransformerAttentionLayerCase1(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (source_vecs, source_padding, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

      ctx, probs = transformer_atten.FPropDefaultTheta(source_vecs,
                                                       source_padding)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs = sess.run([ctx, probs])
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-1.47126436,  1.46579707,  0.39105844, -0.88563323],
           [-1.29514003, -1.08241224,  1.49894714,  2.5935874 ]],
          [[-0.00313053,  1.17399275, -1.28071034, -1.6311729 ],
           [-0.77028418, -0.18855178, -0.75814998,  2.19872856]],
          [[ 1.72851753, -0.40323859, -1.19053328, -1.39761829],
           [-1.72141743, -0.78715289,  1.28404212,  2.78338313]],
          [[-0.8881942 ,  0.33776048,  1.28791749, -0.45082122],
           [ 1.4362365 ,  0.46009994, -1.45436597, -1.90602148]],
          [[-0.51681399, -0.70075679, -0.48352116,  1.93754733],
           [-1.44486678,  0.81801879, -1.03079689,  1.86697066]]]
      expected_probs = [
          [[ 0.21387868,  0.22080734,  0.        ,  0.        ,  0.56531399],
           [ 0.        ,  0.30584112,  0.24723588,  0.44692296,  0.        ]],
          [[ 0.25358215,  0.50932312,  0.        ,  0.        ,  0.23709476],
           [ 0.        ,  0.56834149,  0.2632803 ,  0.16837817,  0.        ]],
          [[ 0.38519409,  0.55454361,  0.        ,  0.        ,  0.06026226],
           [ 0.        ,  0.33708778,  0.21976741,  0.4431448 ,  0.        ]],
          [[ 0.27139962,  0.12790371,  0.        ,  0.        ,  0.60069668],
           [ 0.        ,  0.31849149,  0.28174096,  0.39976761,  0.        ]],
          [[ 0.16272782,  0.15781289,  0.        ,  0.        ,  0.67945927],
           [ 0.        ,  0.55003977,  0.26049581,  0.18946445,  0.        ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)

  def testTransformerAttentionLayerCase2(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = True
      p.num_attention_heads = 2
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (source_vecs, source_padding, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)
      ctx, probs = transformer_atten.FPropDefaultTheta(source_vecs,
                                                       source_padding)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs = sess.run([ctx, probs])
      tf.logging.info(np.array_repr(actual_ctx))
      tf.logging.info(np.array_repr(actual_probs))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-0.14429152,  1.15510106,  1.11930299, -1.19245839],
           [-0.69580591, -0.47006619,  0.82592297,  0.69593251]],
          [[ 0.24164687,  0.53328454, -1.02119482, -1.49412084],
           [-0.82601064,  0.024203  , -1.11880171,  1.80784416]],
          [[ 1.7644347 , -0.53346401, -1.1461122 , -1.42797422],
           [-0.95326459,  0.39580142,  0.39262164,  0.67513674]],
          [[-0.28252155, -0.95237327,  2.08757687, -0.21231559],
           [ 1.4362365 ,  0.46009994, -1.45436597, -1.90602148]],
          [[-0.51681399, -0.70075679, -0.48352116,  1.93754733],
           [-1.44486678,  0.81801879, -1.03079689,  1.86697066]]]
      expected_probs = [
          [[ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.2       ,  0.2       ,  0.2       ,  0.2       ,  0.2       ]],
          [[ 0.3966811 ,  0.60331887,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  1.        ,  0.        ,  0.        ,  0.        ]],
          [[ 0.41050252,  0.58949745,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.5245893 ,  0.4754107 ,  0.        ,  0.        ]],
          [[ 0.58882225,  0.41117775,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.31849149,  0.28174096,  0.39976761,  0.        ]],
          [[ 0.16272782,  0.15781289,  0.        ,  0.        ,  0.67945927],
           [ 0.        ,  0.55003977,  0.26049581,  0.18946445,  0.        ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx)
      self.assertAllClose(expected_probs, actual_probs)

  def testTransformerAttentionLayerDeterministicDropout(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2

      p.residual_dropout_tpl = layers.DeterministicDropoutLayer.Params()
      p.residual_dropout_prob = 0.1

      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (source_vecs, source_padding, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

      ctx, probs = transformer_atten.FProp(transformer_atten.theta, source_vecs,
                                           source_padding)

      tf.global_variables_initializer().run()
      actual_ctx, actual_probs = sess.run([ctx, probs])

      # pylint: disable=bad-whitespace
      # pyformat: disable
      print(np.array_repr(actual_ctx))
      expected_ctx = np.array([
          [[-1.45762944,  1.5337404 ,  0.34037334, -0.97208667],
           [-1.35992002, -1.06530988,  1.53705895,  2.79370689]],
          [[ 0.00657134,  1.12030125, -1.32564592, -1.73569465],
           [-0.80793667, -0.10877949, -0.80295694,  2.25494242]],
          [[ 1.76956046, -0.50777751, -1.19745886, -1.46751583],
           [-1.79178905, -0.77374339,  1.31586027,  2.98173356]],
          [[-0.85498607, -0.37413225,  1.25707364, -0.50043333],
           [ 1.62276983,  0.50820369, -1.52967572, -2.02076197]],
          [[-0.66754031, -0.68657839, -0.51643699,  1.96581018],
           [-1.4816376 ,  0.89419198, -0.57226259,  1.90177512]]
      ], dtype=np.float32)

      print(np.array_repr(actual_probs))
      expected_probs = np.array([
          [[ 0.21387868,  0.22080734,  0.        ,  0.        ,  0.56531399],
           [ 0.        ,  0.30584112,  0.24723588,  0.44692296,  0.        ]],
          [[ 0.25358215,  0.50932312,  0.        ,  0.        ,  0.23709476],
           [ 0.        ,  0.56834149,  0.2632803 ,  0.16837817,  0.        ]],
          [[ 0.38519409,  0.55454361,  0.        ,  0.        ,  0.06026226],
           [ 0.        ,  0.33708778,  0.21976741,  0.4431448 ,  0.        ]],
          [[ 0.27139962,  0.12790371,  0.        ,  0.        ,  0.60069668],
           [ 0.        ,  0.31849149,  0.28174096,  0.39976761,  0.        ]],
          [[ 0.16272782,  0.15781289,  0.        ,  0.        ,  0.67945927],
           [ 0.        ,  0.55003977,  0.26049581,  0.18946445,  0.        ]]
      ], dtype=np.float32)
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)

  def testTransformerAttentionLayerStepByStep(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = True
      p.num_attention_heads = 2
      x_atten = layers_with_attention.TransformerAttentionLayer(p)

      (source_vecs, _, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      ctx1, probs1 = x_atten.FPropDefaultTheta(source_vecs, source_padding)
      ctx2 = []
      probs2 = []
      cached_source_vecs = tf.zeros([0, 2, 4])
      cached_source_contexts = tf.zeros([0, 2, 4])
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        ctx, probs, prefix_states = x_atten.ExtendStep(
            x_atten.theta, source_vecs[i, :, :], prefix_states)
        probs_pad = tf.zeros([2, 5 - i - 1])
        padded_probs = tf.concat([probs, probs_pad], 1)
        ctx2.append(ctx)
        probs2.append(padded_probs)

      ctx2 = tf.stack(ctx2)
      probs2 = tf.stack(probs2)

      tf.global_variables_initializer().run()
      ctx1_v, probs1_v, ctx2_v, probs2_v = sess.run(
          [ctx1, probs1, ctx2, probs2])
      tf.logging.info(np.array_repr(ctx1_v))
      tf.logging.info(np.array_repr(probs1_v))
      tf.logging.info(np.array_repr(ctx2_v))
      tf.logging.info(np.array_repr(probs2_v))
      self.assertAllClose(ctx1_v, ctx2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testTransformerAttentionLayerCase3(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (query_vec, _, aux_vecs,
       aux_paddings) = self._testTransformerAttentionLayerInputs(depth=depth)

      ctx, probs = transformer_atten.FPropDefaultTheta(query_vec, aux_paddings,
                                                       aux_vecs)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs = sess.run([ctx, probs])
      tf.logging.info(np.array_repr(actual_ctx))
      tf.logging.info(np.array_repr(actual_probs))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-1.42420077,  1.19024372,  1.35146523,  0.85896158],
           [-0.44974625, -1.00108492,  1.63387251,  1.678146  ]],
          [[ 0.1134335 ,  1.97617495, -0.35918081,  0.26396495],
           [-0.19688171, -0.71197301,  0.0659425 ,  2.5417304 ]],
          [[ 1.58169425,  0.81259179, -0.58948535,  0.20254248],
           [-0.84438968, -0.65845209,  1.45584249,  1.87587976]],
          [[-1.01532316, -0.05166581,  2.07901478,  0.97540361],
           [ 2.08563352,  0.34328598, -0.23240227, -0.19035631]],
          [[-0.53881919, -0.60117185,  0.29170275,  2.6474514 ],
           [-0.88318163,  0.37149727, -0.16098523,  2.3810885 ]]]
      expected_probs = [
          [[ 0.32392544,  0.,  0.27218491,  0.,  0.19574419,  0.,  0.20814547],
           [ 0.,  0.273045  ,  0.,  0.43572819,  0.,  0.2912268 ,  0.]],
          [[ 0.24094662,  0.,  0.23919827,  0.,  0.26563686,  0.,  0.25421822],
           [ 0.,  0.21680018,  0.,  0.33962148,  0.,0.44357836  ,  0.]],
          [[ 0.20083594,  0.,  0.20683075,  0.,  0.28931937,  0.,  0.30301392],
           [ 0.,  0.24710922,  0.,  0.453915  ,  0.,0.29897571  ,  0.]],
          [[ 0.32845193,  0.,  0.26491433,  0.,  0.18304622,  0.,  0.22358747],
           [ 0.,  0.39426237,  0.,  0.19774443,  0.,0.4079932   ,  0.]],
          [[ 0.23542665,  0.,  0.27910906,  0.,  0.30036426,  0.,  0.18510005],
           [ 0.,  0.20147586,  0.,  0.37759233,  0.,  0.42093182,  0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)

  def testTransformerLayerConstruction(self):
    p = layers_with_attention.TransformerLayer.Params()
    p.name = 'transformer'
    p.source_dim = 4
    p.tr_fflayer_tpl.hidden_dim = 7
    p.tr_atten_tpl.num_attention_heads = 2
    p.has_aux_atten = True
    p.mask_self_atten = True
    _ = layers_with_attention.TransformerLayer(p)

  def testTransformerLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = layers_with_attention.TransformerLayer(p)

      (source_vecs, source_padding, aux_vecs,
       aux_paddings) = self._testTransformerAttentionLayerInputs(depth=depth)

      h, probs = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings)

      tf.global_variables_initializer().run()
      actual_layer_output, actual_prob_output = sess.run([h, probs])
      tf.logging.info(np.array_repr(actual_layer_output))
      tf.logging.info(np.array_repr(actual_prob_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[ 0.68134278,  0.74287307,  0.04602078,  1.99463582],
           [ 0.20382279, -1.50973201,  1.33421206,  0.53317755]],
          [[ 2.46715426,  2.84406185, -0.60359633,  0.51742059],
           [ 1.06444919, -1.45264888, -0.06196141,  0.35242724]],
          [[ 2.3442452 , -0.56243378, -1.1149826 ,  0.50276589],
           [ 1.04868603, -1.68515253,  0.3093726 , -0.19512933]],
          [[-0.11517292, -1.21290886,  1.31996512,  1.14821553],
           [ 3.14395714, -1.07060659,  0.27842081, -1.81273639]],
          [[ 1.39219522, -0.81882864, -0.32732445,  1.36851478],
           [-0.79119539, -0.28148842,  0.29963702,  1.37034667]]]
      expected_prob_output = [
          [[ 0.21795762,  0.,  0.26612395,  0.,  0.31251648, 0.,  0.20340192],
           [ 0.,  0.2677784 ,  0.,  0.32895881,  0., 0.40326279,  0.]],
          [[ 0.25721505,  0.,  0.24116731,  0.,  0.25138181, 0.,  0.2502358 ],
           [ 0.,  0.25691482,  0.,  0.31076014,  0., 0.43232504,  0.]],
          [[ 0.24550268,  0.,  0.25128055,  0.,  0.25109866, 0.,  0.25211811],
           [ 0.,  0.26769161,  0.,  0.32481128,  0., 0.40749705,  0.]],
          [[ 0.22675318,  0.,  0.26633731,  0.,  0.28919035, 0.,  0.21771915],
           [ 0.,  0.35955882,  0.,  0.36869824,  0., 0.271743  ,  0.]],
          [[ 0.21504655,  0.,  0.26958644,  0.,  0.30847484, 0.,  0.20689213],
           [ 0.,  0.29516917,  0.,  0.29359812,  0., 0.41123265,  0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_layer_output, actual_layer_output)
      self.assertAllClose(expected_prob_output, actual_prob_output)

  def testTransformerLayerWithInputPackingFProp(self):
    with self.session(use_gpu=True) as sess:
      with tf.variable_scope('transformer_packed_test', reuse=tf.AUTO_REUSE):
        np.random.seed(6348575)
        depth = 4
        p = layers_with_attention.TransformerLayer.Params()
        p.name = 'transformer'
        p.source_dim = depth
        p.has_aux_atten = True
        p.mask_self_atten = True
        p.tr_fflayer_tpl.hidden_dim = 7
        p.tr_atten_tpl.num_attention_heads = 2
        packed_params = p.Copy()
        transformer = layers_with_attention.TransformerLayer(p)
        packed_params.packed_input = True
        transformer_packed = layers_with_attention.TransformerLayer(
            packed_params)

        dtype = tf.float32
        source_vecs = tf.stack([
            tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(5)
        ])
        source_padding = tf.transpose(
            tf.constant([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=dtype))
        aux_vecs = tf.stack([
            tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)
        ])
        aux_paddings = tf.transpose(
            tf.constant([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1]],
                        dtype=dtype))

        source_vecs_packed = tf.reshape(source_vecs, [-1, 1, depth])
        aux_vecs_packed = tf.reshape(aux_vecs, [-1, 1, depth])
        source_padding_packed = tf.reshape(source_padding, [-1, 1])
        aux_padding_packed = tf.reshape(aux_paddings, [-1, 1])
        source_segment_id = tf.transpose(
            tf.constant([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=tf.float32))
        aux_segment_id = tf.transpose(
            tf.constant([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
                        dtype=tf.float32))

        h, _ = transformer.FPropDefaultTheta(
            source_vecs,
            source_padding,
            aux_vecs=aux_vecs,
            aux_paddings=aux_paddings,
            source_segment_id=None,
            aux_segment_id=None)

        h_packed, _ = transformer_packed.FPropDefaultTheta(
            source_vecs_packed,
            source_padding_packed,
            aux_vecs=aux_vecs_packed,
            aux_paddings=aux_padding_packed,
            source_segment_id=source_segment_id,
            aux_segment_id=aux_segment_id)
        h_packed = tf.reshape(h_packed, tf.shape(h))

        tf.global_variables_initializer().run()
        actual_layer, p_layer = sess.run([h, h_packed])
        self.assertAllClose(actual_layer, p_layer)

  def testTransformerLayerExtendStep(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = layers_with_attention.TransformerLayer(p)

      (source_vecs, _, aux_vecs,
       aux_paddings) = self._testTransformerAttentionLayerInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      h1, probs1 = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings)

      h2 = []
      probs2 = []
      cached_source_vecs = tf.zeros([0, 2, 4])
      cached_source_contexts = tf.zeros([0, 2, 4])
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        h, probs, prefix_states = transformer.ExtendStep(
            transformer.theta, source_vecs[i, :, :], prefix_states, aux_vecs,
            aux_paddings)
        h2.append(h)
        probs2.append(probs)

      h2 = tf.stack(h2)
      probs2 = tf.concat(probs2, 0)

      tf.global_variables_initializer().run()
      h1_v, probs1_v, h2_v, probs2_v = sess.run([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testEvolvedTransformerEncoderBranchedConvsLayer(self):
    layer = layers_with_attention.EvolvedTransformerEncoderBranchedConvsLayer
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      inputs = tf.random_normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layer.Params()
      p.name = 'et_encoder_branched_convs'
      p.input_dim = 3
      et_branched_convs = layer(p)

      h = et_branched_convs.FPropDefaultTheta(inputs, paddings)
      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(h)
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[-0.13232423, -0.46060669,  0.72598207],
           [ 0.6725747 ,  1.58664441,  2.64087844]],
          [[-0.21702465, -0.68267912,  1.20886588],
           [ 1.69793618,  0.53306532,  1.02958691]],
          [[-0.46037287, -0.42950529, -1.68443251],
           [ 0.21459752,  0.42246291, -0.01271994]],
          [[-0.23293658,  0.15300342, -0.83518255],
           [-0.48914853, -0.44239512, -0.2328119 ]],
          [[-0.57934833,  0.24165238, -1.05392623],
           [-0.8292231 ,  0.06175411,  1.28672981]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testEvolvedTransformerDecoderBranchedConvsLayer(self):
    layer = layers_with_attention.EvolvedTransformerDecoderBranchedConvsLayer
    with self.session(use_gpu=True) as sess:
      tf.set_random_seed(3980847392)
      inputs = tf.random_normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layer.Params()
      p.name = 'et_decoder_branched_convs'
      p.input_dim = 3
      et_branched_convs = layer(p)

      h = et_branched_convs.FPropDefaultTheta(inputs, paddings)
      tf.global_variables_initializer().run()
      actual_layer_output = sess.run(h)
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[-0.31987068, -0.65715098,  0.90350437],
           [ 0.00773269,  1.07779562,  4.11094666]],
          [[-0.84862059, -0.93186408,  1.16371167],
           [ 1.31467259,  0.03560367,  2.36822462]],
          [[ 0.02183507, -0.0799394 , -1.68870354],
           [ 0.77921551,  1.30145741, -0.86353606]],
          [[ 0.31672907,  0.50000876, -0.93973017],
           [-0.54707348,  0.19211179, -1.45307386]],
          [[-0.46405494,  0.65833056, -1.09345317],
           [-1.17221224, -0.08027397,  0.84021652]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testEvolvedTransformerEncoderLayerConstruction(self):
    p = layers_with_attention.EvolvedTransformerEncoderLayer.Params()
    p.name = 'evolved_transformer_encoder'
    p.source_dim = 4
    p.transformer_tpl.tr_fflayer_tpl.hidden_dim = 7
    p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    _ = layers_with_attention.EvolvedTransformerEncoderLayer(p)

  def testEvolvedTransformerEncoderLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.EvolvedTransformerEncoderLayer.Params()
      p.name = 'evolved_transformer_encoder'
      p.source_dim = depth
      p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
      transformer = layers_with_attention.EvolvedTransformerEncoderLayer(p)

      (source_vecs, source_padding, aux_vecs,
       aux_paddings) = self._testTransformerAttentionLayerInputs(depth=depth)

      h, probs = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings)

      tf.global_variables_initializer().run()
      actual_layer_output, actual_prob_output = sess.run([h, probs])
      tf.logging.info(np.array_repr(actual_layer_output))
      tf.logging.info(np.array_repr(actual_prob_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[-1.66072488, -0.68993098,  2.21474361, -1.19416285],
           [-1.19632852, -1.68216848,  0.81411338,  1.22243011]],
          [[-1.63495326, -0.59461731,  2.21768641, -1.27701926],
           [-1.21189928,  0.10466897, -0.2177283 ,  0.55320591]],
          [[ 2.01548862,  0.57699752, -0.19467634, -1.54167104],
           [-0.7504791 , -0.24882942, -1.03441   ,  1.34467971]],
          [[-0.70480233, -0.51531398,  2.22327709, -0.40050077],
           [ 1.80162501, -1.46674573, -1.71554327,  0.16294499]],
          [[-1.31785309,  0.02877033,  0.77593923,  0.23810911],
           [-1.5033375 , -0.3106221 , -0.83974278,  1.92515957]]]
      expected_prob_output = [
          [[ 0.25908554,  0.25745451,  0.        ,  0.        ,  0.48345995],
           [ 0.        ,  0.24002703,  0.24501085,  0.51496214,  0.        ]],
          [[ 0.26010525,  0.2584973 ,  0.        ,  0.        ,  0.48139751],
           [ 0.        ,  0.25460899,  0.4237237 ,  0.32166725,  0.        ]],
          [[ 0.3834559 ,  0.38570607,  0.        ,  0.        ,  0.23083803],
           [ 0.        ,  0.18320528,  0.39236429,  0.42443043,  0.        ]],
          [[ 0.30031765,  0.29874057,  0.        ,  0.        ,  0.40094173],
           [ 0.        ,  0.45309049,  0.3099916 ,  0.23691791,  0.        ]],
          [[ 0.18247566,  0.18200508,  0.        ,  0.        ,  0.63551933],
           [ 0.        ,  0.16233809,  0.33563358,  0.50202835,  0.        ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_layer_output, actual_layer_output)
      self.assertAllClose(expected_prob_output, actual_prob_output)

  def testEvolvedTransformerDecoderLayerConstruction(self):
    p = layers_with_attention.EvolvedTransformerDecoderLayer.Params()
    p.name = 'evolved_transformer_decoder'
    p.source_dim = 16
    p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
    p.has_aux_atten = True
    p.mask_self_atten = True
    _ = layers_with_attention.EvolvedTransformerDecoderLayer(p)

  def testEvolvedTransformerDecoderLayerFProp(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.EvolvedTransformerDecoderLayer.Params()
      p.name = 'evolved_transformer_decoder'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_double_heads_atten_tpl.num_attention_heads = 2
      p.tr_atten_tpl.num_attention_heads = 2
      p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
      transformer = layers_with_attention.EvolvedTransformerDecoderLayer(p)

      (source_vecs, source_padding, aux_vecs,
       aux_paddings) = self._testTransformerAttentionLayerInputs(depth=depth)

      h, probs = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings)

      tf.global_variables_initializer().run()
      actual_layer_output, actual_prob_output = sess.run([h, probs])
      tf.logging.info(np.array_repr(actual_layer_output))
      tf.logging.info(np.array_repr(actual_prob_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[-2.15843987,  0.54941475,  1.01636434,  0.13751736],
           [-1.31648636, -0.9490751 ,  0.87473369,  0.5825901 ]],
          [[-0.48339468,  2.73935509, -0.7249794 ,  0.38313258],
           [-1.10127831, -1.39807224,  0.34523556,  0.42135555]],
          [[ 0.55578727,  0.45714682, -0.5104562 , -1.37361968],
           [-1.25782788, -1.21873033,  0.93250239,  0.03656423]],
          [[-1.52875996, -0.97135425,  1.28484297,  0.32869172],
           [ 0.20500244,  2.30189896,  0.24345911, -0.75997925]],
          [[-1.27760804, -1.51032686,  0.2560831 ,  0.66362542],
           [-1.63565814, -0.27384362, -0.42035246,  1.58936501]]]
      expected_prob_output = [
          [[ 0.28604817, 0., 0.24327257, 0., 0.26117378, 0., 0.20950545],
           [ 0., 0.26642066, 0., 0.38120884, 0., 0.3523705 , 0.]],
          [[ 0.24503553, 0., 0.24042624, 0., 0.2301898, 0., 0.28434837],
           [ 0., 0.27049744, 0., 0.36453664, 0., 0.36496598, 0.]],
          [[ 0.25672671, 0., 0.2508592, 0., 0.25038037, 0., 0.24203378],
           [ 0., 0.27020746, 0., 0.38153058, 0., 0.34826195, 0.]],
          [[ 0.27227223, 0., 0.25547835, 0., 0.27728963, 0., 0.19495982],
           [ 0., 0.34053475, 0., 0.35592028, 0., 0.30354494, 0.]],
          [[ 0.23994856, 0., 0.25427216, 0., 0.26202756, 0., 0.24375173],
           [ 0., 0.30927902, 0., 0.32368731, 0., 0.36703369, 0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_layer_output, actual_layer_output)
      self.assertAllClose(expected_prob_output, actual_prob_output)

  def testEvolvedTransformerDecoderLayerExtendStep(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.EvolvedTransformerDecoderLayer.Params()
      p.name = 'evolved_transformer_decoder'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_double_heads_atten_tpl.num_attention_heads = 2
      p.tr_atten_tpl.num_attention_heads = 2
      p.transformer_tpl.tr_atten_tpl.num_attention_heads = 2
      et_decoder = layers_with_attention.EvolvedTransformerDecoderLayer(p)

      (source_vecs, _, aux_vecs,
       aux_paddings) = self._testTransformerAttentionLayerInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      h1, probs1 = et_decoder.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings)

      h2 = []
      probs2 = []

      double_head_attention_states = py_utils.NestedMap(
          key=tf.zeros([0, 2, 4]), value=tf.zeros([0, 2, 4]))
      transformer_layer_states = py_utils.NestedMap(
          key=tf.zeros([0, 2, 4]), value=tf.zeros([0, 2, 4]))
      branched_convs_input = tf.zeros([0, 2, 4])

      prefix_states = py_utils.NestedMap(
          double_head_attention_states=double_head_attention_states,
          transformer_layer_states=transformer_layer_states,
          branched_convs_input=branched_convs_input)

      for i in range(5):
        h, probs, prefix_states = et_decoder.ExtendStep(
            et_decoder.theta, source_vecs[i, :, :], prefix_states, aux_vecs,
            aux_paddings)
        h2.append(h)
        probs2.append(probs)

      h2 = tf.stack(h2)
      probs2 = tf.concat(probs2, 0)

      tf.global_variables_initializer().run()
      h1_v, probs1_v, h2_v, probs2_v = sess.run([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testMergerLayerMean(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 5
      n_sources = 3
      p_ctxs = [
          np.random.rand(batch, depth).astype('float32')
          for _ in range(n_sources)
      ]
      ctxs = [tf.constant(ctx, dtype=tf.float32) for ctx in p_ctxs]

      p = layers_with_attention.MergerLayer.Params()
      p.name = 'merger_layer'
      p.merger_op = 'mean'
      p.source_dim = depth
      merger = p.cls(p)

      ctx = merger.FProp(merger.theta, ctxs)
      tf.global_variables_initializer().run()
      actual_ctx = sess.run([ctx])[0]

      expected_ctx = np.mean(p_ctxs, axis=0)
      self.assertEqual(actual_ctx.shape, (batch, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testMergerLayerAdditiveAttention(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 5
      query_dim = 7
      n_sources = 3
      ctxs = [
          tf.constant(np.random.rand(batch, depth), dtype=tf.float32)
          for _ in range(n_sources)
      ]
      query_vec = tf.constant(
          np.random.rand(batch * 2, query_dim), dtype=tf.float32)
      p = layers_with_attention.MergerLayer.Params()
      p.name = 'merger_layer'
      p.merger_op = 'atten'
      p.source_dim = depth
      p.query_dim = query_dim
      p.hidden_dim = depth
      merger = p.cls(p)

      ctx = merger.FProp(merger.theta, ctxs, query_vec)
      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [ 0.40796196,  0.50855637,  0.92564321,  0.72608167],
          [ 0.34300309,  0.17305931,  0.64801621,  0.4161588 ],
          [ 0.40570667,  0.28166312,  0.07109687,  0.07077176],
          [ 0.44923055,  0.56033343,  0.70899796,  0.73256713],
          [ 0.56362778,  0.42331296,  0.47032064,  0.76701462],
          [ 0.40873578,  0.50516003,  0.92537481,  0.72435796],
          [ 0.33702248,  0.17404726,  0.65101075,  0.41883218],
          [ 0.40316698,  0.28128177,  0.0709244 ,  0.07073996],
          [ 0.44036126,  0.53640223,  0.68623006,  0.75264776],
          [ 0.54324883,  0.42487082,  0.4616943 ,  0.77234119]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch * 2, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testMergerLayerDotProductAttention(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 5
      n_sources = 3
      ctxs = [
          tf.constant(np.random.rand(batch, depth), dtype=tf.float32)
          for _ in range(n_sources)
      ]
      query_vec = tf.constant(
          np.random.rand(batch * 2, depth), dtype=tf.float32)
      p = layers_with_attention.MergerLayer.Params()
      p.name = 'merger_layer'
      p.merger_op = 'atten'
      p.source_dim = depth
      p.query_dim = depth
      p.hidden_dim = depth
      p.attention_tpl = attention.DotProductAttention.Params()
      merger = p.cls(p)

      ctx = merger.FProp(merger.theta, ctxs, query_vec)
      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [ 0.40122974,  0.53032947,  0.92722446,  0.73408204],
          [ 0.37834394,  0.16492322,  0.6284582 ,  0.40583336],
          [ 0.43172807,  0.28519249,  0.07334236,  0.07126588],
          [ 0.48187545,  0.56433642,  0.7028234 ,  0.77750808],
          [ 0.59640014,  0.46689704,  0.47688526,  0.74523771],
          [ 0.41653261,  0.50926942,  0.92638767,  0.74147904],
          [ 0.34954029,  0.16965927,  0.64286244,  0.41876066],
          [ 0.44629157,  0.28723121,  0.07451884,  0.07151417],
          [ 0.509902  ,  0.62019253,  0.75361776,  0.74199384],
          [ 0.56122077,  0.42407531,  0.46921006,  0.76747787]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch * 2, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testMergerLayerConcat(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 5
      n_sources = 3
      ctxs = [
          tf.constant(np.random.rand(batch, depth), dtype=tf.float32)
          for _ in range(n_sources)
      ]
      p = layers_with_attention.MergerLayer.Params()
      p.name = 'merger_layer'
      p.merger_op = 'concat'
      p.source_dim = depth
      merger = p.cls(p)

      ctx = merger.FProp(merger.theta, ctxs)
      tf.global_variables_initializer().run()
      actual_ctx = sess.run([ctx])[0]

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [ 0.1177848 ,  0.94777811,  0.94537693,  0.6216979 ,  0.51051533,
            0.5474115 ,  0.93749231,  0.93760508,  0.5904724 ,  0.05267439,
            0.89581013,  0.63010913],
          [ 0.25139269,  0.13851869,  0.65362513,  0.57537138,  0.05093541,
            0.28593501,  0.84663856,  0.39284077,  0.79584485,  0.07670615,
            0.40381077,  0.26504567],
          [ 0.1108813 ,  0.23381528,  0.05560364,  0.06867393,  0.77289224,
            0.32918185,  0.10567363,  0.07876136,  0.35448784,  0.28477612,
            0.05394353,  0.06531866],
          [ 0.82317245,  0.78475511,  0.82936037,  0.99494314,  0.07920805,
            0.02165302,  0.25108394,  0.92048419,  0.44413447,  0.81940264,
            0.98786688,  0.35846332],
          [ 0.86243463,  0.75607926,  0.54042   ,  0.58698255,  0.13624814,
            0.47994047,  0.28561282,  0.87185597,  0.66811442,  0.07942203,
            0.56781054,  0.83598584]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch, n_sources * depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testMergerLayerWeightedSum(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 2
      n_sources = 3
      ctxs = [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
              [[3.0, 4.0, 5.0, 6.0], [6.0, 7.0, 8.0, 9.0]],
              [[4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 1.0, 2.0]]]
      p = layers_with_attention.MergerLayer.Params()
      p.name = 'merger_layer'
      p.merger_op = 'weighted_sum'
      p.source_dim = depth
      p.num_sources = n_sources
      merger = p.cls(p)

      ctxs = [tf.expand_dims(i, 2) for i in ctxs]
      ctx = tf.squeeze(merger.FProp(merger.theta, ctxs), 2)
      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [[ 2.66666675,  3.66666675,  4.66666698,  5.66666698],
                      [ 5.0,         6.0,         4.33333349,  5.33333349]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testMergerLayerGatedAvg(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(505837249)
      depth = 4
      batch = 2
      n_sources = 3

      inp_1 = np.asarray([[0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      inp_2 = np.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      inp_3 = np.asarray([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      p = layers_with_attention.MergerLayer.Params()
      p.name = 'merger_layer'
      p.merger_op = 'gated_avg'
      p.source_dim = depth
      p.num_sources = n_sources
      merger = p.cls(p)

      ctx = merger.FProp(merger.theta, [inp_1, inp_2, inp_3])
      tf.global_variables_initializer().run()
      actual_ctx = sess.run(ctx)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [ 0.365041,  0.365041,  0.365041,  0.365041],
          [ -1.0, -1.0, 1.0 , 1.0]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testStyleLayer(self):
    with self.session(use_gpu=False) as sess:
      p = layers_with_attention.StyleLayer.Params().Set(
          name='style_layer',
          input_dim=10,
          output_dim=8,
          num_styles=16,
          random_seed=28384)

      tf.set_random_seed(8372749040)
      np.random.seed(12345)
      sl = p.cls(p)
      features = tf.random_normal([2, 10], seed=28384)
      latent, atten_probs = sl.FPropDefaultTheta(features)
      tf.global_variables_initializer().run()
      latent_v, atten_probs_v = sess.run([latent, atten_probs])
      CompareToGoldenSingleFloat(self, 0.122573, np.sum(latent_v))
      CompareToGoldenSingleFloat(self, 2.0, np.sum(atten_probs_v))

  def testStyleLayerWithFeedinAttenProbs(self):
    with self.session(use_gpu=False) as sess:
      p = layers_with_attention.StyleLayer.Params().Set(
          name='style_layer',
          input_dim=10,
          output_dim=8,
          num_styles=16,
          num_heads=4,
          enable_ctx_post_proj=False,
          random_seed=28384)

      tf.set_random_seed(8372749040)
      np.random.seed(12345)
      sl = p.cls(p)
      atten_probs = tf.constant([[1.0] + [0.0] * 15] * 2, dtype=tf.float32)
      ids = tf.constant([0, 0], dtype=tf.int32)
      latent_from_probs = sl.StyleEmbFromProbs(sl.theta, atten_probs)
      latent_from_lookup = sl.EmbLookup(sl.theta, ids)
      tf.global_variables_initializer().run()
      latent_p, latent_l = sess.run([latent_from_probs, latent_from_lookup])
      self.assertAllClose(latent_p, latent_l)

  def testStyleLayer02(self):
    with self.session(use_gpu=False) as sess:
      p = layers_with_attention.StyleLayer.Params().Set(
          name='style_layer',
          input_dim=10,
          output_dim=8,
          num_styles=16,
          random_seed=72738)
      tf.set_random_seed(8372749040)
      np.random.seed(12345)
      sl = p.cls(p)
      features = tf.random_normal([2, 10])
      features = tf.concat([features, features], 0)
      latent, _ = sl.FPropDefaultTheta(features)
      tf.global_variables_initializer().run()
      latent_v = sess.run(latent)
      # Makes sure identical input results in identical style output.
      self.assertAllClose(latent_v[:2], latent_v[2:])


if __name__ == '__main__':
  tf.test.main()
