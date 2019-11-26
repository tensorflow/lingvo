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
# ==============================================================================
"""Tests for layers_with_attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import attention
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core.test_utils import CompareToGoldenSingleFloat
import numpy as np
from six.moves import range


class LayersWithAttentionTest(test_utils.TestCase):

  def testTransformerFeedForwardLayerConstruction(self):
    p = layers_with_attention.TransformerFeedForwardLayer.Params()
    p.name = 'transformer_fflayer_1'
    p.input_dim = 3
    p.hidden_dim = 7
    transformer_fflayer = layers_with_attention.TransformerFeedForwardLayer(p)
    self.assertEqual(0, p.output_dim)
    # output_dim = p.input_dim when p.output_dim is zero.
    self.assertEqual(p.input_dim, transformer_fflayer.output_dim)

    # output_dim equals p.output_dim when p.output_dim is non zero.
    p.output_dim = 10
    p.name = 'transformer_fflayer_2'
    transformer_fflayer = p.Instantiate()
    self.assertEqual(p.output_dim, transformer_fflayer.output_dim)

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

  def _testTransformerAttentionLayerInputs(self,
                                           depth=3,
                                           context_depth=3,
                                           dtype=tf.float32):
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
    context_vecs = tf.stack([
        tf.constant(np.random.rand(2, context_depth), dtype=dtype)
        for _ in range(7)
    ])
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings,
            context_vecs)

  def testTransformerAttentionLayerCase1(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (source_vecs, source_padding, _, _,
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

      (source_vecs, source_padding, _, _,
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

      (source_vecs, source_padding, _, _,
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

      (source_vecs, _, _, _,
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
        ctx, probs, prefix_states = x_atten.ExtendStep(x_atten.theta,
                                                       source_vecs[i, :, :],
                                                       prefix_states)
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

      (query_vec, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

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

  def testTransformerAttentionLayerSourceContext(self):
    # Equivalent: Passing no context vecs and source vecs as context vecs.
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (query_vec, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(
           depth=depth, context_depth=depth)

      ctx1, probs1 = transformer_atten.FPropDefaultTheta(
          query_vec=query_vec,
          source_paddings=aux_paddings,
          source_vecs=aux_vecs,
          context_vecs=aux_vecs)

      ctx2, probs2 = transformer_atten.FPropDefaultTheta(
          query_vec=query_vec,
          source_paddings=aux_paddings,
          source_vecs=aux_vecs)

      tf.global_variables_initializer().run()
      actual_ctx1, actual_probs1, actual_ctx2, actual_probs2 = sess.run(
          [ctx1, probs1, ctx2, probs2])
      self.assertAllEqual(actual_ctx1, actual_ctx2)
      self.assertAllEqual(actual_probs1, actual_probs2)

  def testTransformerAttentionLayerCase4a(self):
    # Distinct key and value vectors of the same size.
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (query_vec, _, aux_vecs, aux_paddings,
       context_vecs) = self._testTransformerAttentionLayerInputs(
           depth=depth, context_depth=depth)

      ctx, probs = transformer_atten.FPropDefaultTheta(
          query_vec=query_vec,
          source_paddings=aux_paddings,
          source_vecs=aux_vecs,
          context_vecs=context_vecs)

      tf.global_variables_initializer().run()
      actual_ctx, actual_probs = sess.run([ctx, probs])
      tf.logging.info(np.array_repr(actual_ctx))
      tf.logging.info(np.array_repr(actual_probs))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-1.20854747,  1.25685954,  1.39818001,  0.558267  ],
           [-0.39904317, -0.85738903,  1.45404375,  1.16389585]],
          [[ 0.27544549,  1.93070388, -0.24477535,  0.12131107],
           [-0.07007086, -0.53334039, -0.01144788,  2.03883505]],
          [[ 1.72718525,  0.73558617, -0.45405889,  0.1063388 ],
           [-0.76255953, -0.52610761,  1.30195093,  1.3571732 ]],
          [[-0.79346895,  0.03049853,  2.11432981,  0.64747918],
           [ 1.86823332,  0.3250314 , -0.50979781, -0.40038702]],
          [[-0.30053592, -0.53348505,  0.41098642,  2.43903708],
           [-0.75298154,  0.50427407, -0.23542863,  1.89634883]]]
      expected_probs = [
          [[ 0.32392544,  0.,  0.27218491,  0.,  0.19574417, 0.,  0.20814548],
           [ 0.,  0.273045  ,  0.,  0.43572825,  0., 0.2912268 ,  0.]],
          [[ 0.24094665,  0.,  0.23919825,  0.,  0.26563686, 0.,  0.25421822],
           [ 0.,  0.21680018,  0.,  0.33962148,  0., 0.44357836,  0.]],
          [[ 0.20083596,  0.,  0.20683077,  0.,  0.28931937, 0.,  0.30301392],
           [ 0.,  0.24710923,  0.,  0.45391506,  0., 0.29897574,  0.]],
          [[ 0.32845187,  0.,  0.26491439,  0.,  0.18304622, 0.,  0.22358751],
           [ 0.,  0.39426237,  0.,  0.1977444 ,  0., 0.4079932 ,  0.]],
          [[ 0.23542665,  0.,  0.27910906,  0.,  0.30036426, 0.,  0.18510005],
           [ 0.,  0.20147583,  0.,  0.37759233,  0., 0.42093182,  0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)

  def testTransformerAttentionLayerCase4b(self):
    # Distinct key and value vectors of different sizes.
    with self.session(use_gpu=True) as sess:
      depth = 4
      context_depth = 3
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      print(p)
      p.num_attention_heads = 2
      p.atten_tpl.enable_ctx_pre_proj = True  # Project values first.
      p.context_dim = context_depth
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (query_vec, _, aux_vecs, aux_paddings,
       context_vecs) = self._testTransformerAttentionLayerInputs(
           depth=depth, context_depth=context_depth)

      ctx, probs = transformer_atten.FPropDefaultTheta(
          query_vec=query_vec,
          source_paddings=aux_paddings,
          source_vecs=aux_vecs,
          context_vecs=context_vecs)

      tf.global_variables_initializer().run()
      actual_ctx, actual_probs = sess.run([ctx, probs])
      tf.logging.info(np.array_repr(actual_ctx))
      tf.logging.info(np.array_repr(actual_probs))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-1.78694427,  0.47923172,  0.89032698,  0.05556235],
           [-0.91133636, -2.05677342,  1.30821121,  1.17388368]],
          [[-0.24106422,  1.27436733, -0.84274787, -0.58437365],
           [-0.58214164, -1.7144506 , -0.21780583,  2.03152227]],
          [[ 1.22925639,  0.15926462, -1.10279834, -0.69442266],
           [-1.2955091 , -1.72805309,  1.15411568,  1.39945638]],
          [[-1.38178754, -0.7436831 ,  1.60785818,  0.16023314],
           [ 1.5662415 , -0.77094424, -0.63392496, -0.6477108 ]],
          [[-0.83664525, -1.20021605, -0.15795891,  1.81301379],
           [-1.27991939, -0.67706013, -0.42443359,  1.92405224]]]
      # Probabilities are unaffected by change of value vectors.
      expected_probs = [
          [[ 0.32392544,  0.,  0.27218491,  0.,  0.19574417, 0.,  0.20814548],
           [ 0.,  0.273045  ,  0.,  0.43572825,  0., 0.2912268 ,  0.]],
          [[ 0.24094665,  0.,  0.23919825,  0.,  0.26563686, 0.,  0.25421822],
           [ 0.,  0.21680018,  0.,  0.33962148,  0., 0.44357836,  0.]],
          [[ 0.20083596,  0.,  0.20683077,  0.,  0.28931937, 0.,  0.30301392],
           [ 0.,  0.24710923,  0.,  0.45391506,  0., 0.29897574,  0.]],
          [[ 0.32845187,  0.,  0.26491439,  0.,  0.18304622, 0.,  0.22358751],
           [ 0.,  0.39426237,  0.,  0.1977444 ,  0., 0.4079932 ,  0.]],
          [[ 0.23542665,  0.,  0.27910906,  0.,  0.30036426, 0.,  0.18510005],
           [ 0.,  0.20147583,  0.,  0.37759233,  0., 0.42093182,  0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)

  def testTransformerAttentionLayerCase5(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = True
      p.mask_type = 'eye'
      p.num_attention_heads = 2
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (source_vecs, source_padding, _, _,
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
          [[-1.89149332,  1.18417633,  0.09695292, -0.83397102],
           [-1.29514003, -1.08241224,  1.49894726,  2.59358764]],
          [[ 0.79232693,  2.47633171, -0.90657401, -1.5221628 ],
           [-0.14457735,  0.09040731, -0.12422991,  2.13300467]],
          [[ 1.72851753, -0.40323859, -1.19053328, -1.39761829],
           [-2.15129089, -1.16594994,  1.1004864 ,  3.07194686]],
          [[-0.88819426,  0.3377606 ,  1.28791749, -0.45082125],
           [1.97874951,  1.50414598, -1.15547466, -1.18697572]],
          [[ 0.10235745, -1.51675844,  0.13308235,  1.26194644],
           [-1.44486666,  0.81801897, -1.03079677,  1.86697078]]]
      expected_probs = [
          [[ 0.        ,  0.33807203,  0.        ,  0.        ,  0.661928  ],
           [ 0.        ,  0.30584112,  0.24723586,  0.44692296,  0.        ]],
          [[ 0.63300228,  0.        ,  0.        ,  0.        ,  0.36699772],
           [ 0.        ,  0.        ,  0.70683479,  0.29316518,  0.        ]],
          [[ 0.38519406,  0.55454367,  0.        ,  0.        ,  0.06026225],
           [ 0.        ,  0.51602799,  0.        ,  0.48397198,  0.        ]],
          [[ 0.27139962,  0.12790368,  0.        ,  0.        ,  0.60069668],
           [ 0.        ,  0.46712866,  0.53287131,  0.        ,  0.        ]],
          [[ 0.55518425,  0.4448157 ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.55003977,  0.26049584,  0.18946445,  0.        ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx)
      self.assertAllClose(expected_probs, actual_probs)

  def testTransformerAttentionLayerCase6(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = True
      p.mask_type = 'ngram'
      p.mask_ngram_order = 3
      p.num_attention_heads = 2
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (source_vecs, source_padding, _, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)
      ctx, probs = transformer_atten.FPropDefaultTheta(source_vecs,
                                                       source_padding)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs = sess.run([ctx, probs])
      tf.logging.info(np.array_repr(actual_ctx))
      tf.logging.info('actual_probs=%r', np.array_repr(actual_probs))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-0.14429152, 1.155101, 1.119303, -1.1924583],
           [-0.6958059, -0.47006613, 0.8259231, 0.6959326]],
          [[0.24164662, 0.5332843, -1.0211949, -1.4941208],
           [-0.8260106, 0.024203, -1.1188016, 1.807844]],
          [[1.7644346, -0.533464, -1.1461123, -1.4279743],
           [-0.95326424, 0.39580172, 0.39262217, 0.6751373]],
          [[-1.3441969, -2.3305228, 1.7523124, 0.15416345],
           [1.4362367, 0.46009994, -1.4543657, -1.9060212]],
          [[-0.8291472, 0.21259767, -0.9077787, 1.6243731],
           [-1.0709695, 0.74920934, -0.5950014, 1.5919089]]]
      expected_probs = [
          [[1.        , 0.        , 0.        , 0.        , 0.        ],
           [0.2       , 0.2       , 0.2       , 0.2       , 0.2       ]],
          [[0.3966811 , 0.6033189 , 0.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        , 0.        , 0.        ]],
          [[0.41050246, 0.5894975 , 0.        , 0.        , 0.        ],
           [0.        , 0.5245893 , 0.4754107 , 0.        , 0.        ]],
          [[0.        , 1.        , 0.        , 0.        , 0.        ],
           [0.        , 0.31849146, 0.28174093, 0.39976764, 0.        ]],
          [[0.        , 0.        , 0.        , 0.        , 1.        ],
           [0.        , 0.        , 0.5881755 , 0.41182452, 0.        ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx)
      self.assertAllClose(expected_probs, actual_probs)

  def testTransformerLayerConstruction(self):
    p = layers_with_attention.TransformerLayer.Params()
    p.name = 'transformer_1'
    p.source_dim = 4
    p.tr_fflayer_tpl.hidden_dim = 7
    p.tr_atten_tpl.num_attention_heads = 2
    p.has_aux_atten = True
    p.mask_self_atten = True
    layer = layers_with_attention.TransformerLayer(p)
    # output_dim is equal to source_dim when p.output_dim == 0
    self.assertEqual(0, p.output_dim)
    self.assertEqual(p.source_dim, layer.output_dim)
    # output_dim corresponds to p.output_dim when it is non-zero.
    p.output_dim = 6
    p.name = 'transformer_2'
    layer = p.Instantiate()
    self.assertEqual(p.output_dim, layer.output_dim)

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

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

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

  def testTransformerLayerOutputLayerNormFProp(self):
    """Test post-layernorm Fprop."""
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.tr_post_ln_tpl = layers.LayerNorm.Params()
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = layers_with_attention.TransformerLayer(p)

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

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
          [[-0.2617511,  -0.17463534, -1.1612566,   1.5976431],
           [ 0.06115358, -1.5903126,   1.1505843,   0.37857458]],
          [[ 0.821784,    1.0885929,  -1.351966,   -0.5584109],
           [ 1.1864979,  -1.5562507,  -0.04089222,  0.41064504]],
          [[ 1.5548539,  -0.6477773,  -1.0664893,   0.15941268],
           [ 1.1784918,  -1.5536082,   0.43964866, -0.06453241]],
          [[-0.38961875, -1.4583365,   1.0075824,   0.84037286],
           [ 1.5903242,  -0.6370207,   0.07592358, -1.0292271]],
          [[ 0.99643826, -1.232215,   -0.73679215,  0.972569],
           [-1.1702524,  -0.5360445,   0.18702725,  1.5192697]]]
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

  def testTransformerLayerFPropMultiPostProj(self):
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
      p.num_aux_atten_post_proj = 2
      transformer = layers_with_attention.TransformerLayer(p)

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

      # Duplicate atten_idx n=2 times.
      atten_idx = tf.constant([0, 1, 1, 0, 1] * 2, dtype=tf.int32)
      h, probs = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings,
          atten_idx=atten_idx)

      tf.global_variables_initializer().run()
      actual_layer_output, actual_prob_output = sess.run([h, probs])
      tf.logging.info(np.array_repr(actual_layer_output))
      tf.logging.info(np.array_repr(actual_prob_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[-0.77411413,  0.86493313,  0.08914688,  1.4910977 ],
           [-1.0093606 , -1.7337079 ,  1.2784883 ,  0.49974248]],
          [[ 1.0396315 ,  2.902943  , -1.1812847 ,  0.19860795],
           [-0.37676954, -0.79837584,  0.6419263 ,  0.45496815]],
          [[ 1.0858665 , -0.6838142 , -1.2464247 ,  0.14764154],
           [-0.45331526, -1.0229169 ,  1.0660815 , -0.06151289]],
          [[-1.3433903 , -1.3154784 ,  1.1818855 ,  0.790216  ],
           [ 1.8400799 , -1.5192697 ,  0.05896807, -1.94113   ]],
          [[-0.11429042, -0.24730963,  0.06099784,  1.0156208 ],
           [-1.9910344 , -0.5176018 ,  0.2490384 ,  1.3254449 ]]]
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

      (source_vecs, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)
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

  def testTransformerLayerWithNgramMaskExtendStep(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_atten_tpl.num_attention_heads = 2
      # Turn on N-gram masking in the TransformerLayer.
      # Before doing so though copy the self-attention params to avoid
      # the auxilliary attention being masked as well.
      p.tr_aux_atten_tpl = p.tr_atten_tpl.Copy()
      p.tr_atten_tpl.is_masked = True
      p.tr_atten_tpl.mask_ngram_order = 3
      p.tr_atten_tpl.mask_type = 'ngram'
      transformer = layers_with_attention.TransformerLayer(p)

      (source_vecs, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)
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

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

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
          [[-1.69196141e+00, -5.03859818e-01, 2.42652583e+00, -1.47606134e+00],
           [-1.27175665e+00, -1.81504273e+00, 7.16031432e-01, 1.40093648e+00]],
          [[-1.83027089e-02, 8.73535872e-04, 2.04444170e+00, -2.74493122e+00],
           [-6.90074801e-01, -2.06015229e-01, -1.21154499e+00, 1.17104244e+00]],
          [[1.84662449e+00, 6.33037746e-01, -2.02951849e-01, -1.70747042e+00],
           [-9.46833491e-01, -3.77074480e-01, -8.77807617e-01, 1.56822240e+00]],
          [[-7.63426960e-01, -3.30413729e-01, 2.38541245e+00, -8.12596083e-01],
           [1.91653550e+00, -1.47251439e+00, -2.19180465e+00, 5.32990336e-01]],
          [[-1.46879780e+00, 5.38376629e-01, 1.50257730e+00, -8.52106392e-01],
           [-1.55632758e+00, -3.48120153e-01, -9.21136498e-01, 2.02043033e+00]]]
      expected_prob_output = [
          [[ 0.27934468,  0.28112975,  0.,          0.,          0.43952557],
           [ 0.,          0.24881637,  0.25068569,  0.50049794,  0.        ]],
          [[ 0.32433772,  0.33424711,  0.,          0.,          0.34141517],
           [ 0.,          0.33490175,  0.29024804,  0.37485027,  0.        ]],
          [[ 0.38673952,  0.33638299,  0.,          0.,          0.27687752],
           [ 0.,          0.30134204,  0.25906932,  0.43958867,  0.        ]],
          [[ 0.30161232,  0.33053303,  0.,          0.,          0.36785468],
           [ 0.,          0.36827549,  0.39080781,  0.24091662,  0.        ]],
          [[ 0.21496946,  0.22309673,  0.,          0.,          0.56193388],
           [ 0.,          0.27062884,  0.23510428,  0.4942669,   0.        ]]]
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

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

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
      expected_layer_output =[
          [[-2.15844011, 0.54941475,  1.01636434,  0.13751738],
           [-1.31499887, -0.9501676,   0.874282,    0.58270419]],
          [[-0.49268177,  2.71167898, -0.78087997,  0.43936318],
           [-1.11428595, -1.38933206,  0.34404463,  0.43363893]],
          [[ 0.57303172,  0.42080224, -0.50416583, -1.36097562],
           [-1.26460135, -1.21081781,  0.9377467,   0.03642488]],
          [[-1.52767372, -0.93615997,  1.33185053,  0.24640131],
           [ 0.16062447,  2.39912128,  0.1896024,  -0.70986807]],
          [[-1.27725732, -1.51283062,  0.26704332,  0.65503371],
           [-1.64287043, -0.30310085, -0.36987182,  1.57325172]]]
      expected_prob_output = [
          [[0.28604817,  0., 0.24327257, 0., 0.26117378, 0., 0.20950545],
           [0., 0.26639479, 0., 0.38120365, 0., 0.35240155, 0.]],
          [[0.24309734, 0., 0.24040565, 0., 0.22922358, 0., 0.2872735],
           [0., 0.27082229, 0., 0.36431897, 0., 0.36485875, 0.]],
          [[0.25640261, 0., 0.25117433, 0., 0.25067171, 0., 0.24175137],
           [0., 0.27037328, 0., 0.38163245, 0., 0.34799421, 0.]],
          [[0.27474535, 0., 0.25523224, 0., 0.27800021, 0., 0.19202216],
           [0., 0.34553668, 0., 0.35240823, 0., 0.30205506, 0.]],
          [[0.24020916, 0., 0.25431803, 0., 0.26219654, 0., 0.24327625],
           [0., 0.30723149, 0., 0.32563132, 0., 0.36713719, 0.]]]
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

      (source_vecs, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)
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
        h, probs, prefix_states = et_decoder.ExtendStep(et_decoder.theta,
                                                        source_vecs[i, :, :],
                                                        prefix_states, aux_vecs,
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
      merger = p.Instantiate()

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
      merger = p.Instantiate()

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
      merger = p.Instantiate()

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
      merger = p.Instantiate()

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

  def testMergerLayerConcatPreProjections(self):
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
      # We down project all of the sources to dimensionality 1.
      p.pre_proj_input_dims = [4, 4, 4]
      p.pre_proj_output_dims = [1, 1, 1]
      p.name = 'merger_layer'
      p.merger_op = 'concat'
      p.source_dim = depth
      merger = p.Instantiate()

      ctx = merger.FProp(merger.theta, ctxs)
      tf.global_variables_initializer().run()
      actual_ctx = sess.run([ctx])[0]

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [ 0.,          0.72890908,  0.        ],
          [ 0.4647972,   0.28266785,  0.        ],
          [ 0.,          0.74580085,  0.09588336],
          [ 0.46080768,  0.,          0.66402191],
          [ 0.19947493,  0.38837075,  0.        ],
      ]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      tf.logging.info(np.array_repr(actual_ctx))
      # The final context vector will have shape (5, 3) since each source
      # has dimensionality 1 after the down projection above.
      self.assertEqual(actual_ctx.shape, (batch, n_sources))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testInvalidPreProjections(self):
    with self.session(use_gpu=True):
      np.random.seed(505837249)
      depth = 4
      p = layers_with_attention.MergerLayer.Params()
      # We intentionally set output_dims to be of a different
      # length. This should cause a ValueError to be raised
      # during init.
      p.pre_proj_input_dims = [4, 4, 4]
      p.pre_proj_output_dims = [1, 1]
      p.name = 'merger_layer'
      p.merger_op = 'concat'
      p.source_dim = depth
      with self.assertRaisesRegex(
          ValueError, 'Output dims should be the same length as input dims.*'):
        _ = p.Instantiate()

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
      merger = p.Instantiate()

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
      merger = p.Instantiate()

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
      sl = p.Instantiate()
      features = tf.random_normal([2, 10], seed=28384)
      latent, atten_probs = sl.FPropDefaultTheta(features)
      tf.global_variables_initializer().run()
      latent_v, atten_probs_v = sess.run([latent, atten_probs])
      CompareToGoldenSingleFloat(self, -1.208686, np.sum(latent_v))
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
      sl = p.Instantiate()
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
      sl = p.Instantiate()
      features = tf.random_normal([2, 10])
      features = tf.concat([features, features], 0)
      latent, _ = sl.FPropDefaultTheta(features)
      tf.global_variables_initializer().run()
      latent_v = sess.run(latent)
      # Makes sure identical input results in identical style output.
      self.assertAllClose(latent_v[:2], latent_v[2:])

  def _testTransformerMultitaskLayerInputs(self, depth=3, dtype=tf.float32):
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
    source_task_id = tf.constant([[2, 3]], dtype=tf.int32)
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings,
            source_task_id)

  def testTransformerLayerWithMultitaskAdaptersConstruction(self):
    p = layers_with_attention.TransformerLayerWithMultitaskAdapters.Params()
    p.name = 'transformer_with_adapters'
    p.source_dim = 4
    p.tr_fflayer_tpl.hidden_dim = 7
    p.tr_atten_tpl.num_attention_heads = 2
    p.has_aux_atten = True
    p.mask_self_atten = True
    p.adapter_tpl.input_dim = 4
    p.adapter_tpl.num_tasks = 4
    p.adapter_tpl.bottleneck_dim = 2
    _ = layers_with_attention.TransformerLayerWithMultitaskAdapters(p)

  def testTransformerLayerWithMultitaskAdaptersFProp(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayerWithMultitaskAdapters.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      p.adapter_tpl.input_dim = 4
      p.adapter_tpl.num_tasks = 4
      p.adapter_tpl.bottleneck_dim = 2
      transformer = layers_with_attention.TransformerLayerWithMultitaskAdapters(
          p)

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       source_task_id) = self._testTransformerMultitaskLayerInputs(depth=depth)

      h, probs = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings,
          source_task_id=source_task_id)

      tf.global_variables_initializer().run()
      actual_layer_output, actual_prob_output = sess.run([h, probs])
      tf.logging.info(np.array_repr(actual_layer_output))
      tf.logging.info(np.array_repr(actual_prob_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[ 0.02441728,  0.26923186,  0.68582684,  1.1531992 ],
           [ 0.69027936, -1.94770098,  2.00558615,  0.17057157]],
          [[ 1.81022859,  2.37042093,  0.03620988, -0.32401592],
           [ 1.66707945, -1.95131969,  0.64937419,  0.05853128]],
          [[ 1.53475547, -0.60239077, -0.05797344, -0.48760295],
           [ 1.53514266, -2.1231215 ,  0.98074663, -0.5577352 ]],
          [[-1.32504404, -1.28702664,  2.597996  ,  0.24809647],
           [ 3.7842629 , -1.46549737,  0.91363102, -2.37071466]],
          [[ 0.52196532, -0.73371518,  0.86030912,  0.33838278],
           [ 0.01923725, -0.8887378 ,  1.08245265,  1.19935369]]
      ]
      expected_prob_output = [
          [[ 0.21795765,  0,  0.26612395,  0,  0.31251645, 0,  0.20340192],
           [ 0,  0.2677784 ,  0,  0.32895881,  0, 0.40326279,  0]],
          [[ 0.25721508,  0,  0.24116732,  0,  0.25138181, 0,  0.2502358 ],
           [ 0,  0.25691482,  0,  0.31076014,  0, 0.43232504,  0]],
          [[ 0.24550268,  0,  0.25128055,  0,  0.25109866, 0,  0.25211811],
           [ 0,  0.26769164,  0,  0.32481131,  0, 0.40749705,  0]],
          [[ 0.22675318,  0,  0.26633731,  0,  0.28919035, 0,  0.21771917],
           [ 0,  0.35955882,  0,  0.36869821,  0, 0.271743  ,  0]],
          [[ 0.21504655,  0,  0.26958644,  0,  0.30847484, 0,  0.20689213],
           [ 0,  0.29516917,  0,  0.29359812,  0, 0.41123268,  0]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_layer_output, actual_layer_output)
      self.assertAllClose(expected_prob_output, actual_prob_output)

  def testTransformerLayerWithMultitaskAdaptersWithInputPackingFProp(self):
    with self.session(use_gpu=True) as sess:
      with tf.variable_scope('transformer_packed_test', reuse=tf.AUTO_REUSE):
        np.random.seed(6348575)
        depth = 4
        p = layers_with_attention.TransformerLayerWithMultitaskAdapters.Params()
        p.name = 'transformer_with_adapters'
        p.source_dim = depth
        p.has_aux_atten = True
        p.mask_self_atten = True
        p.tr_fflayer_tpl.hidden_dim = 7
        p.tr_atten_tpl.num_attention_heads = 2
        p.adapter_tpl.input_dim = 4
        p.adapter_tpl.num_tasks = 4
        p.adapter_tpl.bottleneck_dim = 2
        packed_params = p.Copy()
        transformer = layers_with_attention.TransformerLayerWithMultitaskAdapters(
            p)
        packed_params.packed_input = True
        transformer_packed = layers_with_attention.TransformerLayerWithMultitaskAdapters(
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
        source_task_id = tf.constant([[2, 3]], dtype=tf.int32)

        source_vecs_packed = tf.reshape(source_vecs, [-1, 1, depth])
        aux_vecs_packed = tf.reshape(aux_vecs, [-1, 1, depth])
        source_padding_packed = tf.reshape(source_padding, [-1, 1])
        aux_padding_packed = tf.reshape(aux_paddings, [-1, 1])
        source_task_id_packed = tf.transpose(
            tf.constant([[2, 3, 2, 3, 2, 3, 2, 3, 2, 3]], dtype=tf.int32))
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
            aux_segment_id=None,
            source_task_id=source_task_id)

        h_packed, _ = transformer_packed.FPropDefaultTheta(
            source_vecs_packed,
            source_padding_packed,
            aux_vecs=aux_vecs_packed,
            aux_paddings=aux_padding_packed,
            source_segment_id=source_segment_id,
            aux_segment_id=aux_segment_id,
            source_task_id=source_task_id_packed)
        h_packed = tf.reshape(h_packed, tf.shape(h))

        tf.global_variables_initializer().run()
        actual_layer, p_layer = sess.run([h, h_packed])
        self.assertAllClose(actual_layer, p_layer)

  def testTransformerLayerWithMultitaskAdaptersExtendStep(self):
    with self.session(use_gpu=True) as sess:
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayerWithMultitaskAdapters.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_atten_tpl.num_attention_heads = 2
      p.adapter_tpl.input_dim = 4
      p.adapter_tpl.num_tasks = 4
      p.adapter_tpl.bottleneck_dim = 2
      transformer = layers_with_attention.TransformerLayerWithMultitaskAdapters(
          p)

      (source_vecs, _, aux_vecs, aux_paddings,
       source_task_id) = self._testTransformerMultitaskLayerInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      h1, probs1 = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings,
          source_task_id=source_task_id)

      h2 = []
      probs2 = []
      cached_source_vecs = tf.zeros([0, 2, 4])
      cached_source_contexts = tf.zeros([0, 2, 4])
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        h, probs, prefix_states = transformer.ExtendStep(
            transformer.theta,
            source_vecs[i, :, :],
            prefix_states,
            aux_vecs,
            aux_paddings,
            source_task_id=source_task_id[0, :])
        h2.append(h)
        probs2.append(probs)

      h2 = tf.stack(h2)
      probs2 = tf.concat(probs2, 0)

      tf.global_variables_initializer().run()
      h1_v, probs1_v, h2_v, probs2_v = sess.run([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)


if __name__ == '__main__':
  tf.test.main()
