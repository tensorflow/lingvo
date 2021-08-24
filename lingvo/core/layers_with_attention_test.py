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
# ==============================================================================
"""Tests for layers_with_attention."""

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import gshard_builder
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import symbolic
from lingvo.core import test_utils
from lingvo.core.test_utils import CompareToGoldenSingleFloat
import numpy as np


class LayersWithAttentionTest(test_utils.TestCase, parameterized.TestCase):

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
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers_with_attention.TransformerFeedForwardLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 3
      p.hidden_dim = 7
      transformer_fflayer = layers_with_attention.TransformerFeedForwardLayer(p)

      h = transformer_fflayer.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
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

  @parameterized.named_parameters(('_3D', 3), ('_4D', 4))
  def testReshapedTransformerFeedForwardLayer(self, rank):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      input_dim = 6
      if rank == 3:
        dims = [input_dim]
      else:
        self.assertEqual(rank, 4)
        dims = [2, input_dim // 2]
      shape = [5, 2] + dims
      inputs = tf.random.normal(shape, seed=948387483)
      paddings = tf.zeros([5, 2])

      p = layers_with_attention.ReshapedTransformerFeedForwardLayer.Params()
      p.name = 'reshaped_transformer_fflayer'
      p.input_dim = input_dim
      p.hidden_dim = 7
      p.fflayer_tpl.weight_split_dims_mapping_list = [[-1, -1], [-1, -1]]
      p.fflayer_tpl.activation_split_dims_mapping_list = [[-1, -1], [-1, -1]]
      p.device_mesh = np.reshape(np.arange(4), [2, 2])
      l = p.Instantiate()

      outputs = l.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      outputs = self.evaluate(outputs)
      self.assertAllClose(outputs.shape, inputs.shape)

  def testHybridFeedforwardLayer(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      symbol_sub_key = symbolic.Symbol('sub_key')

      # create a basic fflayer.
      fflayer_p = (layers_with_attention.TransformerFeedForwardLayer.Params())
      fflayer_p.name = 'fflayer'
      fflayer_p.input_dim = 3
      fflayer_p.hidden_dim = 7

      # create a moe layer.
      moe_p = layers_with_attention.MoEFeedforwardLayer.Params()
      moe_p.name = 'moe'
      moe_p.moe_builder_p = gshard_builder.MoEBuilder.Params().Set(
          model_dim=3,
          num_devices=2,
          num_groups=2,
          e_dim=2,
          c_dim=4,
          moe_hidden_dim=7)

      # create a hybrid layer.
      hybrid_p = layers_with_attention.HybridFeedforwardLayer.Params()
      hybrid_p.name = 'hybrid'
      hybrid_p.sub = py_utils.NestedMap({'ff': fflayer_p, 'moe': moe_p})
      hybrid_p.sub_key = symbol_sub_key

      hybrid_fflayer = layers_with_attention.HybridFeedforwardLayer(hybrid_p)

      with py_utils.AuxLossContext() as aux_loss_ctx:
        with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES,
                                       {symbol_sub_key: 'ff'}):
          outputs_ff = hybrid_fflayer.FPropDefaultTheta(inputs, paddings)
          self.assertEmpty(aux_loss_ctx.aux_losses)
        with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES,
                                       {symbol_sub_key: 'moe'}):
          outputs_moe = hybrid_fflayer.FPropDefaultTheta(inputs, paddings)
          self.assertNotEmpty(aux_loss_ctx.aux_losses)

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output_ff = self.evaluate(outputs_ff)
      actual_layer_output_moe = self.evaluate(outputs_moe)
      # pylint: disable=bad-whitespace
      expected_output_ff = ([[[-0.05825481, -0.07296887, 0.04780552],
                              [0.40495688, 1.3521885, 1.9623209]],
                             [[-0.538299, -0.51939666, 0.14743209],
                              [2.0082633, 0.41585845, 1.2604249]],
                             [[-0.16540301, -0.588541, -0.68776536],
                              [0.22190702, 0.32639492, 0.5300334]],
                             [[0.06300206, -0.01546569, 0.0259212],
                              [-0.9785279, -0.96456575, -1.2386773]],
                             [[-0.8001151, -0.08313039, -0.7068999],
                              [-1.4299163, -0.22745167, 0.2734915]]])
      # pylint: disable=bad-whitespace
      expected_output_moe = ([[[0.4632624, -0.08097249, -0.10976761],
                               [-1.1534482, 0.20076305, 2.2456918]],
                              [[0.42073604, 1.262385, -0.47051585],
                               [1.0274936, 1.9002852, 1.4712151]],
                              [[-0.03316217, 0.38010496, 0.24893013],
                               [0.34987167, -0.6271608, 1.3136444]],
                              [[-0.68526286, 0.08780301, -0.9903437],
                               [0.39456585, 0.1792891, 0.84773403]],
                              [[0.08420426, -1.4146113, 0.9402321],
                               [0.22846438, -1.857454, -0.59214497]]])
      print(np.array_repr(actual_layer_output_ff))
      print(np.array_repr(actual_layer_output_moe))
      self.assertAllClose(actual_layer_output_ff, expected_output_ff)
      self.assertAllClose(actual_layer_output_moe, expected_output_moe)

  def testTransformerShardedMoeLayer(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers_with_attention.TransformerShardedMoeLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 3
      p.hidden_dim = 7
      p.output_dim = 3
      p.num_groups = 2
      p.num_experts = 4
      p.expert_capacity_factor = 2
      moe_fflayer = layers_with_attention.TransformerShardedMoeLayer(p)

      h = moe_fflayer.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
      # pylint: disable=bad-whitespace
      expected_output = [[[-0.34213868, -0.1577737, 0.15908651],
                          [0.0995039, 2.0593567, 2.422616]],
                         [[-0.9544622, -0.289206, 0.3745581],
                          [2.7121983, 0.49732625, 0.98936653]],
                         [[-0.22911909, -0.52321994, -1.3037556],
                          [0.29460418, 0.14727175, 0.3075519]],
                         [[-0.03022301, 0.00274765, -0.4092078],
                          [-1.0508028, 0.11724383, -0.70965374]],
                         [[-0.3473336, -0.4793697, -0.26441547],
                          [-1.6704988, 0.60920537, 0.7469079]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testTransformerShardedMoeLayerShardedWeights(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 4], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers_with_attention.TransformerShardedMoeLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 4
      p.hidden_dim = 7
      p.output_dim = 4
      p.num_groups = 2
      p.num_experts = 4
      p.expert_capacity_factor = 2
      p.expert_weight_shards = 2
      moe_fflayer = layers_with_attention.TransformerShardedMoeLayer(p)

      h = moe_fflayer.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
      # pylint: disable=bad-whitespace
      expected_output = [[[-1.6894771, -0.6188934, 1.3259739, 0.6954013],
                          [2.0653946, 2.946611, -0.8549718, -0.5904686]],
                         [[0.15020806, 1.439679, 0.54579806, 2.0817866],
                          [-0.08175106, -0.7739575, -0.9843587, 0.46894]],
                         [[0.8291634, -0.58913743, 0.6789296, 0.08628751],
                          [-0.431438, -1.3788042, -0.8718487, -0.6101668]],
                         [[0.32909858, 0.5900509, -0.7350087, -1.3075548],
                          [0.46176028, 1.3289857, -1.640419, -0.9618089]],
                         [[-1.2423284, -0.26266062, 2.591324, 0.13978946],
                          [-0.10520535, -0.00721201, -0.44894043, 1.3547784]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  @parameterized.named_parameters(
      ('F32FPropF32Input', tf.float32, tf.float32, 7.182965),
      ('F32FPropBF16Input', tf.float32, tf.bfloat16, 7.183718),
      ('BF16FPropF32Input', tf.bfloat16, tf.float32, 7.15625),
      ('BF16FPropBF16Input', tf.bfloat16, tf.bfloat16, 7.15625),
  )
  def testTransformerFeedForwardLayerFPropDtype(self,
                                                fprop_dtype,
                                                input_dtype,
                                                expected_sum=0.):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.cast(
          tf.random.normal([5, 2, 3], seed=948387483), dtype=input_dtype)
      paddings = tf.zeros([5, 2], dtype=input_dtype)
      p = layers_with_attention.TransformerFeedForwardLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 3
      p.hidden_dim = 7
      p.random_seed = 1234
      p.cls.SetFPropDtype(p, fprop_dtype)

      # fprop_dtype set accordingly.
      self.assertEqual(fprop_dtype, p.fprop_dtype)

      transformer_fflayer = layers_with_attention.TransformerFeedForwardLayer(p)
      h = transformer_fflayer.FPropDefaultTheta(inputs, paddings)
      h *= tf.cast(1 - paddings[:, :, tf.newaxis], h.dtype)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(expected_sum, tf.reduce_sum(h).eval())

  def testTransformerFeedForwardLayerSpecOutDim(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers_with_attention.TransformerFeedForwardLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 3
      p.output_dim = 5
      p.hidden_dim = 7
      transformer_fflayer = layers_with_attention.TransformerFeedForwardLayer(p)

      h = transformer_fflayer.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
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
    with self.session(use_gpu=True):
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
      self.evaluate(tf.global_variables_initializer())
      actual_ctx, actual_probs = self.evaluate([ctx, probs])
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

  def testTransformerAttentionLayerCase1GatedResidualConnection(self):
    with self.session(use_gpu=True):
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      p.add_unnormalized_input = True
      p.residual_function = layers.HighwaySkipLayer.Params().Set(
          carry_bias_init=100, couple_carry_transform_gates=True)
      transformer_atten = layers_with_attention.TransformerAttentionLayer(p)

      (source_vecs, source_padding, _, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

      ctx, probs = transformer_atten.FPropDefaultTheta(source_vecs,
                                                       source_padding)
      self.evaluate(tf.global_variables_initializer())
      actual_ctx, _, actual_source_vecs = self.evaluate(
          [ctx, probs, source_vecs])
      # Due to the high bias, the gated residual connection is saturated and
      # returns the original (unnormalized) input.
      self.assertAllClose(actual_source_vecs, actual_ctx, rtol=1e-4, atol=1e-4)

  def testTransformerAttentionLayerCase2(self):
    with self.session(use_gpu=True):
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
      self.evaluate(tf.global_variables_initializer())
      actual_ctx, actual_probs = self.evaluate([ctx, probs])
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
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_ctx, actual_probs = self.evaluate([ctx, probs])

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
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      ctx1_v, probs1_v, ctx2_v, probs2_v = self.evaluate(
          [ctx1, probs1, ctx2, probs2])
      tf.logging.info(np.array_repr(ctx1_v))
      tf.logging.info(np.array_repr(probs1_v))
      tf.logging.info(np.array_repr(ctx2_v))
      tf.logging.info(np.array_repr(probs2_v))
      self.assertAllClose(ctx1_v, ctx2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testTransformerAttentionLayerGatedResidualConnectionStepByStep(self):
    with self.session(use_gpu=True):
      depth = 4
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = True
      p.num_attention_heads = 2
      p.residual_function = layers.HighwaySkipLayer.Params().Set(
          couple_carry_transform_gates=True)
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

      self.evaluate(tf.global_variables_initializer())
      ctx1_v, probs1_v, ctx2_v, probs2_v = self.evaluate(
          [ctx1, probs1, ctx2, probs2])
      self.assertAllClose(ctx1_v, ctx2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testTransformerAttentionLayerCase3(self):
    with self.session(use_gpu=True):
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
      self.evaluate(tf.global_variables_initializer())
      actual_ctx, actual_probs = self.evaluate([ctx, probs])
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

  def _testTransformerAttentionLayerInputsMultiAuxSource(
      self, aux_source_list, depth=3, context_depth=3, dtype=tf.float32):
    (source_vecs, source_padding, _, _, _) = (
        self._testTransformerAttentionLayerInputs(depth, context_depth, dtype))
    np.random.seed(505837249)
    aux_source_vecs = py_utils.NestedMap()
    for aux_src_key in aux_source_list:
      aux_source_vecs[aux_src_key] = tf.stack([
          tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)
      ])
    aux_source_paddings = py_utils.NestedMap({
        aux_src_key: tf.transpose(
            tf.constant([[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]],
                        dtype=dtype)) for aux_src_key in aux_source_list
    })
    context_vecs = py_utils.NestedMap()
    for aux_src_key in aux_source_list:
      context_vecs[aux_src_key] = tf.stack([
          tf.constant(np.random.rand(2, context_depth), dtype=dtype)
          for _ in range(7)
      ])
    return (source_vecs, source_padding, aux_source_vecs, aux_source_paddings,
            context_vecs)

  def testTransformerAttentionLayerCase3MultiSource(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerMultiSourceAttentionLayer.Params()
      p.name = 'transformer_atten_multisource'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      p.num_source = 2
      transformer_atten = (
          layers_with_attention.TransformerMultiSourceAttentionLayer(p))

      (query_vec, _, aux_vecs, aux_paddings, _) = (
          self._testTransformerAttentionLayerInputsMultiAuxSource(
              ['source_0', 'source_1'], depth=depth))

      ctx, probs = transformer_atten.FPropDefaultTheta(query_vec, aux_paddings,
                                                       aux_vecs)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs = sess.run([ctx, probs])
      tf.logging.info(np.array_repr(actual_ctx))
      tf.logging.info(np.array_repr(actual_probs))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-1.9893163 ,  0.8076348 , -0.33805895, -0.20369706],
           [-1.4164762 , -1.0597495 , -0.3834126 ,  0.3456189 ]],
          [[-0.32503036,  1.4952568 , -1.9324137 , -0.77024114],
           [-0.9230547 , -0.89096445, -1.7928462 ,  1.0901089 ]],
          [[ 1.2240632 ,  0.26689315, -2.0940783 , -0.9101793 ],
           [-1.805772  , -0.74725944, -0.5485071 ,  0.5403221 ]],
          [[-1.5880606 , -0.43595213,  0.3818947 , -0.15712431],
           [ 0.968494  ,  0.19423638, -2.308594  , -1.4253062 ]],
          [[-0.8178122 , -1.1570994 , -1.1993079 ,  1.4127911 ],
           [-1.7231476 ,  0.17116357, -2.0703826 ,  0.96320933]]]
      expected_probs = [
          [[0.16679956, 0., 0.2122806 , 0., 0.23512313, 0., 0.38579667],
           [0., 0.28562695, 0., 0.3442661 , 0., 0.370107  , 0.]],
          [[0.28629708, 0., 0.18837643, 0., 0.2644571 , 0., 0.26086944],
           [0., 0.5590873 , 0., 0.22519027, 0., 0.21572247, 0.]],
          [[0.3374045 , 0., 0.21468817, 0., 0.25822428, 0., 0.18968314],
           [0., 0.2896077 , 0., 0.34381902, 0., 0.36657327, 0.]],
          [[0.14310986, 0., 0.2507791 , 0., 0.22308563, 0., 0.3830254 ],
           [0., 0.43070328, 0., 0.2930708 , 0., 0.27622598, 0.]],
          [[0.30523974, 0., 0.30610216, 0., 0.2248916 , 0., 0.1637665 ],
           [0., 0.49082592, 0., 0.26013914, 0., 0.24903494, 0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)

  def testTransformerAttentionLayerCase3MultiSourceMatchSingle(self):
    with self.session(use_gpu=True) as sess:
      # Prepare inputs.
      depth = 4
      (query_vec, _, aux_vecs, aux_paddings, _) = (
          self._testTransformerAttentionLayerInputsMultiAuxSource(
              ['source_0', 'source_1'], depth=depth))

      # Create two source inputs but use single-source attention
      p = layers_with_attention.TransformerMultiSourceAttentionLayer.Params()
      p.random_seed = 123
      p.name = 'transformer_atten_multisource_single'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      p.num_source = 1
      msa = layers_with_attention.TransformerMultiSourceAttentionLayer(p)
      msa_ctx, msa_probs = (
          msa.FPropDefaultTheta(query_vec, aux_paddings, aux_vecs))

      # Original single source attention layer.
      p = layers_with_attention.TransformerAttentionLayer.Params()
      p.random_seed = 123
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      ssa = layers_with_attention.TransformerAttentionLayer(p)
      ssa_ctx, ssa_probs = ssa.FPropDefaultTheta(query_vec,
                                                 aux_paddings['source_0'],
                                                 aux_vecs['source_0'])

      # Compare two context vectors and probabilities.
      tf.global_variables_initializer().run()
      actual_msa_ctx, actual_msa_probs, actual_ssa_ctx, actual_ssa_probs = (
          sess.run([msa_ctx, msa_probs, ssa_ctx, ssa_probs]))

      # pylint: disable=bad-whitespace
      # pyformat: disable
      self.assertAllClose(actual_msa_ctx, actual_ssa_ctx,
                          rtol=1e-05, atol=1e-05)
      self.assertAllClose(actual_msa_probs, actual_ssa_probs,
                          rtol=1e-05, atol=1e-05)

  def testTransformerAttentionLayerSourceContext(self):
    # Equivalent: Passing no context vecs and source vecs as context vecs.
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_ctx1, actual_probs1, actual_ctx2, actual_probs2 = self.evaluate(
          [ctx1, probs1, ctx2, probs2])
      self.assertAllEqual(actual_ctx1, actual_ctx2)
      self.assertAllEqual(actual_probs1, actual_probs2)

  def testTransformerAttentionLayerCase4a(self):
    # Distinct key and value vectors of the same size.
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_ctx, actual_probs = self.evaluate([ctx, probs])
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

  def testTransformerAttentionLayerCase4aMultiSource(self):
    # Distinct key and value vectors of the same size.
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.TransformerMultiSourceAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      p.num_source = 2
      transformer_atten = (
          layers_with_attention.TransformerMultiSourceAttentionLayer(p))

      (query_vec, _, aux_vecs, aux_paddings,
       context_vecs) = self._testTransformerAttentionLayerInputsMultiAuxSource(
           ['source_0', 'source_1'], depth=depth, context_depth=depth)

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
          [[-2.263544  , -0.6288333 ,  0.56436384,  0.01389617],
           [-1.2714428 , -2.6551175 ,  1.2088637 ,  0.48963785]],
          [[-0.7530552 ,  0.2863059 , -1.0583341 , -0.62887365],
           [-0.96861804, -2.3108015 , -0.32213187,  1.4070555 ]],
          [[ 0.6888912 , -0.83782226, -1.3349627 , -0.69250315],
           [-1.646423  , -2.3046758 ,  1.0617565 ,  0.6768545 ]],
          [[-1.8710074 , -1.9080507 ,  1.2318314 ,  0.14334393],
           [ 0.92007947, -1.775676  , -1.1390316 , -0.9541185 ]],
          [[-1.375605  , -2.3637016 , -0.5955716 ,  1.8448071 ],
           [-1.6682272 , -1.2519215 , -0.5330956 ,  1.2296966 ]]]
      expected_probs = [
          [[0.22346233, 0., 0.27624047, 0., 0.18855348, 0., 0.31174374],
           [0., 0.17387941, 0., 0.4642802 , 0., 0.36184043, 0.]],
          [[0.23724607, 0., 0.24033949, 0., 0.3725937 , 0., 0.14982074],
           [0., 0.15892553, 0., 0.4639521 , 0., 0.37712237, 0.]],
          [[0.25570837, 0., 0.21216837, 0., 0.40378904, 0., 0.12833425],
           [0., 0.16656096, 0., 0.47455215, 0., 0.3588869 , 0.]],
          [[0.22077632, 0., 0.27379048, 0., 0.14691363, 0., 0.35851952],
           [0., 0.5620029 , 0., 0.21104112, 0., 0.22695602, 0.]],
          [[0.20673111, 0., 0.22832122, 0., 0.12665181, 0., 0.43829578],
           [0., 0.17881572, 0., 0.45228398, 0., 0.36890027, 0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)

  def testTransformerAttentionLayerCase4b(self):
    # Distinct key and value vectors of different sizes.
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_ctx, actual_probs = self.evaluate([ctx, probs])
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

  def testTransformerAttentionLayerCase4bMultiSource(self):
    # Distinct key and value vectors of different sizes.
    with self.session(use_gpu=True) as sess:
      depth = 4
      context_depth = 3
      p = layers_with_attention.TransformerMultiSourceAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      print(p)
      p.num_attention_heads = 2
      p.atten_tpl.enable_ctx_pre_proj = True  # Project values first.
      p.context_dim = context_depth
      p.num_source = 2
      transformer_atten = (
          layers_with_attention.TransformerMultiSourceAttentionLayer(p))

      (query_vec, _, aux_vecs, aux_paddings,
       context_vecs) = self._testTransformerAttentionLayerInputsMultiAuxSource(
           ['source_0', 'source_1'], depth=depth, context_depth=context_depth)

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
          [[-0.52144265,  1.7370229 ,  0.09479183,  1.3142197 ],
           [ 0.48182625, -0.41524518,  0.2950616 ,  2.3245158 ]],
          [[ 1.0139368 ,  2.6589985 , -1.528513  ,  0.6880791 ],
           [ 0.7810391 , -0.05419022, -1.227257  ,  3.2472034 ]],
          [[ 2.4781933 ,  1.5413835 , -1.7759092 ,  0.6057711 ],
           [ 0.11952043, -0.07813096,  0.12346762,  2.5386043 ]],
          [[-0.12219751,  0.46310303,  0.7768879 ,  1.4295386 ],
           [ 2.8404353 ,  0.901297  , -1.5073049 ,  0.60736287]],
          [[ 0.37801886, -0.05114734, -1.003877  ,  3.0894797 ],
           [ 0.10942292,  0.975695  , -1.4856565 ,  3.1215234 ]]]
      # Probabilities are unaffected by change of value vectors.
      expected_probs = [
          [[0.22346234, 0., 0.27624047, 0., 0.18855348, 0., 0.31174374],
           [0., 0.17387941, 0., 0.4642802 , 0., 0.36184043, 0.]],
          [[0.23724607, 0., 0.24033949, 0., 0.3725937 , 0., 0.14982076],
           [0., 0.15892553, 0., 0.4639521 , 0., 0.3771224 , 0.]],
          [[0.2557084 , 0., 0.21216837, 0., 0.403789  , 0., 0.12833424],
           [0., 0.16656098, 0., 0.47455215, 0., 0.3588869 , 0.]],
          [[0.22077632, 0., 0.27379048, 0., 0.14691365, 0., 0.35851952],
           [0., 0.5620028 , 0., 0.21104114, 0., 0.22695604, 0.]],
          [[0.20673111, 0., 0.22832122, 0., 0.12665181, 0., 0.43829578],
           [0., 0.17881574, 0., 0.45228398, 0., 0.36890027, 0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)

  def testTransformerAttentionLayerCase5(self):
    with self.session(use_gpu=True):
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
      self.evaluate(tf.global_variables_initializer())
      actual_ctx, actual_probs = self.evaluate([ctx, probs])
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
    with self.session(use_gpu=True):
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
      self.evaluate(tf.global_variables_initializer())
      actual_ctx, actual_probs = self.evaluate([ctx, probs])
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
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, actual_prob_output = self.evaluate([h, probs])
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

  def testMultiAuxSourceTransformerLayerFProp(self):
    with self.session(use_gpu=True):
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.tr_aux_atten_tpl = (
          layers_with_attention.TransformerMultiSourceAttentionLayer.Params()
          .Set(
              source_dim=p.source_dim,
              num_source=2,
              primary_source_index=0,
              num_attention_heads=4))
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = layers_with_attention.TransformerLayer(p)

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputsMultiAuxSource(
           ['source_0', 'source_1'], depth=depth)

      h, probs = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings)

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, actual_prob_output = self.evaluate([h, probs])
      tf.logging.info(np.array_repr(actual_layer_output))
      tf.logging.info(np.array_repr(actual_prob_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[-0.06297368,  0.75025094, -0.18167767,  2.27935   ],
           [-0.22771487, -1.9459789 ,  0.758848  ,  1.2273839 ]],
          [[ 1.6866916 ,  2.9894042 , -1.2287276 ,  0.8018402 ],
           [ 0.656631  , -1.2074132 , -0.41612232,  1.4099871 ]],
          [[ 1.6463919 , -0.493517  , -1.3494966 ,  0.6977608 ],
           [ 0.49527422, -1.5192728 , -0.1677584 ,  0.781141  ]],
          [[-0.86701846, -1.2044021 ,  1.0710557 ,  1.4103888 ],
           [ 3.0039275 , -0.98788637, -0.48796502, -0.90612394]],
          [[ 0.6298464 , -0.33676302, -0.22484902,  1.8341833 ],
           [-1.2259507 , -0.716857  , -0.1336647 ,  1.9020087 ]]]
      expected_prob_output = [
          [[0.23055646, 0., 0.270754  , 0., 0.20824522, 0., 0.2904443 ],
           [0., 0.34072176, 0., 0.34083408, 0., 0.31844413, 0.]],
          [[0.25588194, 0., 0.21465777, 0., 0.26527345, 0., 0.26418683],
           [0., 0.31694067, 0., 0.35715103, 0., 0.32590824, 0.]],
          [[0.24147315, 0., 0.22742277, 0., 0.2734162 , 0., 0.25768787],
           [0., 0.33686832, 0., 0.34380934, 0., 0.31932235, 0.]],
          [[0.22445586, 0., 0.29794338, 0., 0.20764738, 0., 0.26995337],
           [0., 0.3731808 , 0., 0.29736063, 0., 0.32945853, 0.]],
          [[0.2221506 , 0., 0.2830769 , 0., 0.21007922, 0., 0.2846933 ],
           [0., 0.3024338 , 0., 0.36399618, 0., 0.33357003, 0.]]]
      # # pyformat: enable
      # # pylint: enable=bad-whitespace
      self.assertAllClose(expected_layer_output, actual_layer_output)
      self.assertAllClose(expected_prob_output, actual_prob_output)

  def testMultiAuxSourceTransformerLayerFPropMatchSingle(self):
    with self.session(use_gpu=True):
      np.random.seed(6348575)
      depth = 4
      # Multi-source transformer layer
      p = layers_with_attention.TransformerLayer.Params().Set(
          name='multi_source_trans', random_seed=123)
      p.tr_atten_tpl.num_attention_heads = 4
      p.source_dim = depth
      p.has_aux_atten = True
      p.tr_aux_atten_tpl = (
          layers_with_attention.TransformerMultiSourceAttentionLayer.Params()
          .Set(
              source_dim=p.source_dim,
              num_source=1,
              primary_source_index=0,
              num_attention_heads=4))
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      msa_trans = layers_with_attention.TransformerLayer(p)

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputsMultiAuxSource(
           ['source_0', 'source_1'], depth=depth)

      msa_h, msa_probs = msa_trans.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings)

      # Original single-source transformer decoder.
      p = layers_with_attention.TransformerLayer.Params().Set(
          name='single_source_trans', random_seed=123)
      p.tr_atten_tpl.num_attention_heads = 4
      p.tr_atten_tpl.random_seed = 123
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      ssa_trans = layers_with_attention.TransformerLayer(p)
      ssa_h, ssa_probs = ssa_trans.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs['source_0'],
          aux_paddings=aux_paddings['source_0'])

      self.evaluate(tf.global_variables_initializer())
      msa_layer_output, msa_prob_output, ssa_layer_output, ssa_prob_output = (
          self.evaluate([msa_h, msa_probs, ssa_h, ssa_probs]))

      self.assertAllClose(
          msa_layer_output, ssa_layer_output, rtol=1e-05, atol=1e-05)
      self.assertAllClose(
          msa_prob_output, ssa_prob_output, rtol=1e-05, atol=1e-05)

  def testTransformerLayerOutputLayerNormFProp(self):
    """Test post-layernorm Fprop."""
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, actual_prob_output = self.evaluate([h, probs])
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
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, actual_prob_output = self.evaluate([h, probs])
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
    with self.session(use_gpu=True):
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

        self.evaluate(tf.global_variables_initializer())
        actual_layer, p_layer = self.evaluate([h, h_packed])
        self.assertAllClose(actual_layer, p_layer)

  def testTransformerLayerExtendStep(self):
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      h1_v, probs1_v, h2_v, probs2_v = self.evaluate([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testMultiAuxSourceTransformerLayerExtendStep(self):
    with self.session(use_gpu=True):
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.tr_aux_atten_tpl = (
          layers_with_attention.TransformerMultiSourceAttentionLayer.Params()
          .Set(
              source_dim=p.source_dim,
              num_source=2,
              primary_source_index=0,
              num_attention_heads=4))
      p.mask_self_atten = True
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = layers_with_attention.TransformerLayer(p)

      (source_vecs, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputsMultiAuxSource(
           ['source_0', 'source_1'], depth=depth)
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

      self.evaluate(tf.global_variables_initializer())
      h1_v, probs1_v, h2_v, probs2_v = self.evaluate([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testMultiAuxSourceTransformerLayerExtendStepMatchSingle(self):
    with self.session(use_gpu=True):
      # Prepare inputs
      np.random.seed(6348575)
      depth = 4
      (source_vecs, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputsMultiAuxSource(
           ['source_0', 'source_1'], depth=depth)

      # Multi-source transformer layer
      p = layers_with_attention.TransformerLayer.Params().Set(
          name='multi_source_trans', random_seed=123)
      p.tr_atten_tpl.num_attention_heads = 4
      p.source_dim = depth
      p.has_aux_atten = True
      p.tr_aux_atten_tpl = (
          layers_with_attention.TransformerMultiSourceAttentionLayer.Params()
          .Set(
              source_dim=p.source_dim,
              num_source=1,
              primary_source_index=0,
              num_attention_heads=4))
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      msa_trans = layers_with_attention.TransformerLayer(p)

      h_msa = []
      probs_msa = []
      cached_source_vecs = tf.zeros([0, 2, 4])
      cached_source_contexts = tf.zeros([0, 2, 4])
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        h, probs, prefix_states = msa_trans.ExtendStep(msa_trans.theta,
                                                       source_vecs[i, :, :],
                                                       prefix_states, aux_vecs,
                                                       aux_paddings)
        h_msa.append(h)
        probs_msa.append(probs)
      h_msa = tf.stack(h_msa)
      probs_msa = tf.concat(probs_msa, 0)

      # Original single-source transformer decoder.
      p = layers_with_attention.TransformerLayer.Params().Set(
          name='single_source_trans', random_seed=123)
      p.tr_atten_tpl.num_attention_heads = 4
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_fflayer_tpl.hidden_dim = 7
      ssa_trans = layers_with_attention.TransformerLayer(p)

      h_ssa = []
      probs_ssa = []
      cached_source_vecs = tf.zeros([0, 2, 4])
      cached_source_contexts = tf.zeros([0, 2, 4])
      prefix_states = py_utils.NestedMap(
          key=cached_source_vecs, value=cached_source_contexts)
      for i in range(5):
        h, probs, prefix_states = ssa_trans.ExtendStep(ssa_trans.theta,
                                                       source_vecs[i, :, :],
                                                       prefix_states,
                                                       aux_vecs['source_0'],
                                                       aux_paddings['source_0'])
        h_ssa.append(h)
        probs_ssa.append(probs)
      h_ssa = tf.stack(h_ssa)
      probs_ssa = tf.concat(probs_ssa, 0)

      self.evaluate(tf.global_variables_initializer())
      h_msa_v, h_ssa_v, probs_msa_v, probs_ssa_v = self.evaluate(
          [h_msa, h_ssa, probs_msa, probs_ssa])
      tf.logging.info(np.array_repr(h_msa_v))
      tf.logging.info(np.array_repr(h_ssa_v))
      self.assertAllClose(h_msa_v, h_ssa_v)
      self.assertAllClose(probs_msa_v, probs_ssa_v)

  def testTransformerLayerWithNgramMaskExtendStep(self):
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      h1_v, probs1_v, h2_v, probs2_v = self.evaluate([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testTransformerLayerWithPostLayernormExtendStep(self):
    with self.session(use_gpu=True):
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.has_aux_atten = True
      p.mask_self_atten = True
      p.tr_atten_tpl.num_attention_heads = 2
      p.tr_post_ln_tpl = layers.LayerNorm.Params()
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

      self.evaluate(tf.global_variables_initializer())
      h1_v, probs1_v, h2_v, probs2_v = self.evaluate([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testEvolvedTransformerEncoderBranchedConvsLayer(self):
    layer = layers_with_attention.EvolvedTransformerEncoderBranchedConvsLayer
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layer.Params()
      p.name = 'et_encoder_branched_convs'
      p.input_dim = 3
      et_branched_convs = layer(p)

      h = et_branched_convs.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
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
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layer.Params()
      p.name = 'et_decoder_branched_convs'
      p.input_dim = 3
      et_branched_convs = layer(p)

      h = et_branched_convs.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
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
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, actual_prob_output = self.evaluate([h, probs])
      tf.logging.info(np.array_repr(actual_layer_output))
      tf.logging.info(np.array_repr(actual_prob_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[-1.6823182 , -0.33362526,  2.3092952 , -1.2768047 ],
           [-1.2375467 , -1.7528018 ,  0.6906311 ,  1.4148781 ]],
          [[-0.3703399 , -0.8586656 ,  2.4906673 , -2.2977662 ],
           [-0.60055196, -0.23450398, -1.2372489 ,  1.1125396 ]],
          [[ 2.0659933 ,  0.82173675, -0.17450655, -1.7258614 ],
           [-0.9853776 , -0.37829524, -0.77619284,  1.516935  ]],
          [[-0.5684509 , -0.15367106,  2.3549438 , -0.7618298 ],
           [ 1.9434962 , -1.6360642 , -2.0586298 ,  0.6888489 ]],
          [[-1.4064629 ,  0.5313531 ,  1.5535516 , -1.0066429 ],
           [-1.5438917 , -0.40709162, -0.8882869 ,  2.037459  ]]]
      expected_prob_output = [
          [[0.3098957 , 0.21260454, 0.        , 0.        , 0.47749978],
           [0.        , 0.24464089, 0.24325356, 0.5121056 , 0.        ]],
          [[0.27023065, 0.43278426, 0.        , 0.        , 0.29698506],
           [0.        , 0.35950065, 0.2941079 , 0.3463914 , 0.        ]],
          [[0.350026  , 0.38011283, 0.        , 0.        , 0.26986116],
           [0.        , 0.32311335, 0.25958124, 0.41730544, 0.        ]],
          [[0.31028467, 0.31974676, 0.        , 0.        , 0.36996856],
           [0.        , 0.34648925, 0.38719398, 0.2663167 , 0.        ]],
          [[0.28063056, 0.15659373, 0.        , 0.        , 0.5627757 ],
           [0.        , 0.28404602, 0.23116755, 0.4847864 , 0.        ]]]
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
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, actual_prob_output = self.evaluate([h, probs])
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
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      h1_v, probs1_v, h2_v, probs2_v = self.evaluate([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testStyleLayer(self):
    with self.session(use_gpu=False):
      p = layers_with_attention.StyleLayer.Params().Set(
          name='style_layer',
          input_dim=10,
          output_dim=8,
          num_styles=16,
          random_seed=28384)

      tf.random.set_seed(8372749040)
      np.random.seed(12345)
      sl = p.Instantiate()
      features = tf.random.normal([2, 10], seed=28384)
      latent, atten_probs = sl.FPropDefaultTheta(features)
      self.evaluate(tf.global_variables_initializer())
      latent_v, atten_probs_v = self.evaluate([latent, atten_probs])
      CompareToGoldenSingleFloat(self, -1.208686, np.sum(latent_v))
      CompareToGoldenSingleFloat(self, 2.0, np.sum(atten_probs_v))

  def testStyleLayerWithFeedinAttenProbs(self):
    with self.session(use_gpu=False):
      p = layers_with_attention.StyleLayer.Params().Set(
          name='style_layer',
          input_dim=10,
          output_dim=8,
          num_styles=16,
          num_heads=4,
          enable_ctx_post_proj=False,
          random_seed=28384)

      tf.random.set_seed(8372749040)
      np.random.seed(12345)
      sl = p.Instantiate()
      atten_probs = tf.constant([[1.0] + [0.0] * 15] * 2, dtype=tf.float32)
      ids = tf.constant([0, 0], dtype=tf.int32)
      latent_from_probs = sl.StyleEmbFromProbs(sl.theta, atten_probs)
      latent_from_lookup = sl.EmbLookup(sl.theta, ids)
      self.evaluate(tf.global_variables_initializer())
      latent_p, latent_l = self.evaluate(
          [latent_from_probs, latent_from_lookup])
      self.assertAllClose(latent_p, latent_l)

  def testStyleLayer02(self):
    with self.session(use_gpu=False):
      p = layers_with_attention.StyleLayer.Params().Set(
          name='style_layer',
          input_dim=10,
          output_dim=8,
          num_styles=16,
          random_seed=72738)
      tf.random.set_seed(8372749040)
      np.random.seed(12345)
      sl = p.Instantiate()
      features = tf.random.normal([2, 10])
      features = tf.concat([features, features], 0)
      latent, _ = sl.FPropDefaultTheta(features)
      self.evaluate(tf.global_variables_initializer())
      latent_v = self.evaluate(latent)
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
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, actual_prob_output = self.evaluate([h, probs])
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
    with self.session(use_gpu=True):
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

        self.evaluate(tf.global_variables_initializer())
        actual_layer, p_layer = self.evaluate([h, h_packed])
        self.assertAllClose(actual_layer, p_layer)

  def testTransformerLayerWithMultitaskAdaptersExtendStep(self):
    with self.session(use_gpu=True):
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

      self.evaluate(tf.global_variables_initializer())
      h1_v, probs1_v, h2_v, probs2_v = self.evaluate([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testCCTFeedForwardLayerConstruction(self):
    p = layers_with_attention.CCTFeedForwardLayer.Params()
    p.name = 'cct_fflayer_1'
    p.input_dim = 3
    p.hidden_dim = 7
    p.num_blocks = 2
    p.gating_tpl.hidden_layer_dim = 2
    p.gating_tpl.noise_std = 5.0
    p.gating_tpl.noise_warmup_steps = 100
    _ = layers_with_attention.CCTFeedForwardLayer(p)

  def testCCTFeedForwardLayerTraining(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers_with_attention.CCTFeedForwardLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 3
      p.hidden_dim = 7
      p.num_blocks = 2
      p.gating_tpl.hidden_layer_dim = 2
      p.gating_tpl.noise_std = 5.0
      p.gating_tpl.noise_warmup_steps = 100
      cct_fflayer = layers_with_attention.CCTFeedForwardLayer(p)

      h, p_c = cct_fflayer.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, p_c_val = self.evaluate([h, p_c])
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ 0.49714983, -1.1684668 ,  0.4889576 ],
           [ 1.7869478 ,  1.4456576 ,  1.4123362 ]],
          [[ 0.10564739, -1.5359519 ,  0.67742175],
           [ 1.6211604 ,  0.583192  ,  1.056936  ]],
          [[-0.01121134, -0.78554434, -0.84111285],
           [ 0.45078042,  0.63005054,  0.08024757]],
          [[ 0.162924  ,  0.14500974, -0.32797086],
           [ 0.41885388, -0.5852693 , -1.7245001 ]],
          [[-0.6601118 ,  0.30835745, -0.48543385],
           [-0.04813027, -0.04633661, -0.21723843]]]
      expected_p_c = [
          [[0.5607947 , 0.49624035],
           [0.72082597, 0.50216115]],
          [[0.6352798 , 0.49843985],
           [0.5       , 0.5       ]],
          [[0.5       , 0.5       ],
           [0.5       , 0.5       ]],
          [[0.5       , 0.5       ],
           [0.7562946 , 0.50510687]],
          [[0.62267053, 0.50738835],
           [0.73273706, 0.5029184 ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(actual_layer_output, expected_output)
      self.assertAllClose(p_c_val, expected_p_c)

  def testCCTFeedForwardLayerInference(self):
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers_with_attention.CCTFeedForwardLayer.Params()
      p.name = 'transformer_fflayer'
      p.input_dim = 3
      p.hidden_dim = 7
      p.num_blocks = 2
      p.gating_tpl.hidden_layer_dim = 2
      p.gating_tpl.noise_std = 5.0
      p.gating_tpl.noise_warmup_steps = 100
      cct_fflayer = layers_with_attention.CCTFeedForwardLayer(p)

      h, p_c = cct_fflayer.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, p_c_val = self.evaluate([h, p_c])
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ 1.1921753 , -0.78980637, -0.58472836],
           [ 2.5051842 ,  1.6491661 ,  0.49059153]],
          [[ 0.6877271 , -1.1452659 , -0.29534382],
           [ 1.5774723 ,  0.6462606 ,  1.0375552 ]],
          [[ 0.12175584, -1.2262938 , -0.5333306 ],
           [ 0.4632102 ,  0.7119628 , -0.01409443]],
          [[ 0.16090955,  0.06721614, -0.24816278],
           [ 0.9799552 , -0.2861529 , -2.5847178 ]],
          [[-0.48719   ,  0.18763718, -0.53763545],
           [ 0.5886377 ,  0.21293162, -1.1132748 ]]
      ]
      expected_p_c = [
          [[1., 0.],
           [1., 1.]],
          [[1., 0.],
           [1., 1.]],
          [[1., 1.],
           [1., 1.]],
          [[1., 1.],
           [1., 1.]],
          [[1., 1.],
           [1., 1.]]
      ]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(actual_layer_output, expected_output, atol=2e-6)
      self.assertAllClose(p_c_val, expected_p_c)

  def testTransformerWithContextLayerConstruction(self):
    p = layers_with_attention.TransformerWithContextLayer.Params()
    p.name = 'transformer_1'
    p.source_dim = 4
    p.tr_fflayer_tpl.hidden_dim = 7
    p.tr_atten_tpl.num_attention_heads = 2
    layer = p.Instantiate()
    # output_dim is equal to source_dim when p.output_dim == 0
    self.assertEqual(0, p.output_dim)
    self.assertEqual(p.source_dim, layer.fflayer.output_dim)

  def testTransformerWithContextLayerFProp(self):
    with self.session(use_gpu=True):
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerWithContextLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.tr_fflayer_tpl.hidden_dim = 7
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = p.Instantiate()

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth)

      h, probs = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings,
          tertiary_vecs=aux_vecs,
          tertiary_paddings=aux_paddings)

      self.evaluate(tf.global_variables_initializer())
      actual_layer_output, actual_prob_output = self.evaluate([h, probs])
      tf.logging.info(np.array_repr(actual_layer_output))
      tf.logging.info(np.array_repr(actual_prob_output))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_layer_output = [
          [[ 0.55129296, -0.7571765 ,  0.281192  ,  0.8710322 ],
           [ 0.5072957 , -1.3714458 ,  1.5689826 , -0.0971924 ]],
          [[ 2.2560897 ,  2.7890472 ,  0.016873  , -0.5172725 ],
           [ 1.4128124 , -2.0595124 ,  0.37241971, -0.6075135 ]],
          [[ 2.57011   , -0.8678784 , -0.33203793, -0.18508816],
           [ 1.3549538 , -2.0990794 ,  0.62103236, -0.9975941 ]],
          [[ 0.15144205, -1.1681134 ,  1.7113727 ,  0.4682465 ],
           [ 2.9454587 , -1.4413761 ,  0.5215157 , -2.1541023 ]],
          [[ 1.5092299 , -1.7608491 ,  0.21144068,  0.22785848],
           [-0.766488  , -0.487573  ,  1.0574573 ,  0.81118184]]]
      expected_prob_output = [
          [[0.223735  , 0.        , 0.26685917, 0.        , 0.2968173 ,
            0.        , 0.2125885 ],
           [0.        , 0.28585374, 0.        , 0.35088098, 0.        ,
            0.36326528, 0.        ]],
          [[0.2703818 , 0.        , 0.23092957, 0.        , 0.2249705 ,
            0.        , 0.27371815],
           [0.        , 0.26997963, 0.        , 0.33745134, 0.        ,
            0.39256904, 0.        ]],
          [[0.25208434, 0.        , 0.24830116, 0.        , 0.23168065,
            0.        , 0.26793382],
           [0.        , 0.2847324 , 0.        , 0.3477454 , 0.        ,
            0.36752218, 0.        ]],
          [[0.23778549, 0.        , 0.26169604, 0.        , 0.26542395,
            0.        , 0.23509452],
           [0.        , 0.3603859 , 0.        , 0.37519425, 0.        ,
            0.26441985, 0.        ]],
          [[0.22522289, 0.        , 0.26782405, 0.        , 0.28599125,
            0.        , 0.22096181],
           [0.        , 0.29979968, 0.        , 0.31155068, 0.        ,
            0.38864967, 0.        ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_layer_output, actual_layer_output)
      self.assertAllClose(expected_prob_output, actual_prob_output)

  def testTransformerWithContextLayerPackedInputFProp(self):
    with self.session(use_gpu=True):
      with tf.variable_scope('transformer_packed_test', reuse=tf.AUTO_REUSE):
        np.random.seed(6348575)
        depth = 4
        p = layers_with_attention.TransformerLayer.Params()
        p.name = 'transformer'
        p.source_dim = depth
        p.tr_fflayer_tpl.hidden_dim = 7
        p.tr_atten_tpl.num_attention_heads = 2
        transformer = p.Instantiate()
        packed_params = p.Copy()
        packed_params.packed_input = True
        transformer_packed = packed_params.Instantiate()

        dtype = tf.float32
        source_vecs = tf.stack([
            tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(5)
        ])
        source_padding = tf.transpose(
            tf.constant([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=dtype))
        aux_vecs = tf.stack([
            tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)
        ])
        tertiary_vecs = tf.stack([
            tf.constant(np.random.rand(2, depth), dtype=dtype) for _ in range(7)
        ])
        aux_paddings = tf.transpose(
            tf.constant([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1]],
                        dtype=dtype))

        source_vecs_packed = tf.reshape(source_vecs, [-1, 1, depth])
        aux_vecs_packed = tf.reshape(aux_vecs, [-1, 1, depth])
        tertiary_vecs_packed = tf.reshape(tertiary_vecs, [-1, 1, depth])
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
            tertiary_vecs=tertiary_vecs,
            tertiary_paddings=aux_paddings)

        h_packed, _ = transformer_packed.FPropDefaultTheta(
            source_vecs_packed,
            source_padding_packed,
            aux_vecs=aux_vecs_packed,
            aux_paddings=aux_padding_packed,
            source_segment_id=source_segment_id,
            aux_segment_id=aux_segment_id,
            tertiary_vecs=tertiary_vecs_packed,
            tertiary_paddings=aux_padding_packed,
            tertiary_segment_id=aux_segment_id)
        h_packed = tf.reshape(h_packed, tf.shape(h))

        self.evaluate(tf.global_variables_initializer())
        actual_layer, p_layer = self.evaluate([h, h_packed])
        self.assertAllClose(actual_layer, p_layer)

  def testTransformerWithContextLayerExtendStep(self):
    with self.session(use_gpu=True):
      np.random.seed(6348575)
      depth = 4
      p = layers_with_attention.TransformerWithContextLayer.Params()
      p.name = 'transformer'
      p.source_dim = depth
      p.tr_atten_tpl.num_attention_heads = 2
      transformer = p.Instantiate()

      (source_vecs, source_padding, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth)
      source_padding = tf.zeros([5, 2])

      h1, probs1 = transformer.FPropDefaultTheta(
          source_vecs,
          source_padding,
          aux_vecs=aux_vecs,
          aux_paddings=aux_paddings,
          tertiary_vecs=aux_vecs,
          tertiary_paddings=aux_paddings)

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
            tertiary_vecs=aux_vecs,
            tertiary_paddings=aux_paddings)
        h2.append(h)
        probs2.append(probs)

      h2 = tf.stack(h2)
      probs2 = tf.concat(probs2, 0)

      self.evaluate(tf.global_variables_initializer())
      h1_v, probs1_v, h2_v, probs2_v = self.evaluate([h1, probs1, h2, probs2])
      self.assertAllClose(h1_v, h2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testCCTAttentionLayerSelfAttentionTraining(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.CCTAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = True
      p.num_attention_heads = 2
      p.gating_tpl.hidden_layer_dim = 2
      p.gating_tpl.noise_std = 5.0
      p.gating_tpl.noise_warmup_steps = 100
      transformer_atten = layers_with_attention.CCTAttentionLayer(p)

      (source_vecs, source_padding, _, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

      ctx, probs, qpc, spc = transformer_atten.FPropDefaultTheta(
          source_vecs, source_padding)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs, actual_qpc, actual_spc = sess.run(
          [ctx, probs, qpc, spc])
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-0.9170906 ,  0.89127994,  0.8682031 , -0.8423924 ],
           [-1.2874005 , -0.76474655,  0.5771928 ,  1.4749541 ]],
          [[ 0.34465155,  0.74996084, -0.48622286, -0.6083897 ],
           [-0.7486481 , -0.07628638, -0.99187833,  1.8168143 ]],
          [[ 1.6986014 , -0.44173932, -0.7130059 , -0.5438557 ],
           [-1.3927674 , -0.09861529,  0.3361559 ,  1.1552272 ]],
          [[-0.5439662 , -1.0707575 ,  1.8813989 , -0.26667514],
           [ 1.1484473 ,  0.9964316 , -1.2344118 , -0.91046673]],
          [[-0.06898946, -1.5815425 , -0.45298773,  2.1035194 ],
           [-1.7475295 ,  0.27231437, -0.8034381 ,  2.2786536 ]]]
      expected_probs = [
          [[1.        , 0.        , 0.        , 0.        , 0.        ],
           [0.2       , 0.2       , 0.2       , 0.2       , 0.2       ]],
          [[0.4238176 , 0.57618237, 0.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        , 0.        , 0.        ]],
          [[0.34105754, 0.65894246, 0.        , 0.        , 0.        ],
           [0.        , 0.55719167, 0.44280833, 0.        , 0.        ]],
          [[0.6528083 , 0.34719166, 0.        , 0.        , 0.        ],
           [0.        , 0.32477915, 0.36445653, 0.31076428, 0.        ]],
          [[0.28325003, 0.21873125, 0.        , 0.        , 0.49801874],
           [0.        , 0.43867606, 0.2793855 , 0.28193837, 0.        ]]]
      expected_qpc = [
          [[0.5       ],
           [0.5818492 ]],
          [[0.5411409 ],
           [0.55023897]],
          [[0.56948507],
           [0.5499979 ]],
          [[0.5166038 ],
           [0.58645904]],
          [[0.54155153],
           [0.5       ]]]
      expected_spc = [
          [[0.21472901],
           [0.06997871]],
          [[0.53207266],
           [0.39812705]],
          [[0.5217048 ],
           [0.07829338]],
          [[0.06743541],
           [0.5       ]],
          [[0.32987863],
           [0.5442441 ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_qpc, actual_qpc, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_spc, actual_spc, rtol=1e-05, atol=1e-05)

  def testCCTAttentionLayerSelfAttentionEval(self):
    with self.session(use_gpu=True) as sess, self.SetEval(True):
      depth = 4
      p = layers_with_attention.CCTAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = True
      p.num_attention_heads = 2
      p.gating_tpl.hidden_layer_dim = 2
      p.gating_tpl.noise_std = 5.0
      p.gating_tpl.noise_warmup_steps = 100
      transformer_atten = layers_with_attention.CCTAttentionLayer(p)

      (source_vecs, source_padding, _, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

      ctx, probs, qpc, spc = transformer_atten.FPropDefaultTheta(
          source_vecs, source_padding)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs, actual_qpc, actual_spc = sess.run(
          [ctx, probs, qpc, spc])
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-1.5939784e+00,  8.5430717e-01,  8.4722424e-01, -1.0755297e-01],
           [-1.6199683e+00, -1.9144357e+00,  1.0950426e+00,  2.4393613e+00]],
          [[ 2.0492536e-01, -5.0217152e-02, -1.5521961e-01,  5.1122904e-04],
           [-4.3141130e-01, -9.0650195e-01, -3.5488802e-01,  1.6928028e+00]],
          [[ 1.7034934e+00, -1.1774492e+00, -4.2603785e-01, -1.0000569e-01],
           [-1.0880733e+00, -9.0783793e-01,  9.9768031e-01,  9.9823117e-01]],
          [[-1.1584746e+00, -2.0163212e+00,  2.3776212e+00,  7.9717481e-01],
           [ 1.3303024e+00, -1.4763023e+00,  2.6441175e-01, -1.1841190e-01]],
          [[-3.0323851e-01, -2.5461116e+00,  5.0698155e-01,  2.3423686e+00],
           [-2.0771229e+00, -8.0027932e-01, -7.4258000e-02,  2.9516606e+00]]]
      expected_probs = [
          [[1.        , 0.        , 0.        , 0.        , 0.        ],
           [0.2       , 0.2       , 0.2       , 0.2       , 0.2       ]],
          [[0.35538384, 0.6446162 , 0.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        , 0.        , 0.        ]],
          [[0.18125553, 0.8187444 , 0.        , 0.        , 0.        ],
           [0.        , 0.5       , 0.5       , 0.        , 0.        ]],
          [[0.7752405 , 0.22475953, 0.        , 0.        , 0.        ],
           [0.        , 0.36166608, 0.36166608, 0.27666792, 0.        ]],
          [[0.40603536, 0.18792923, 0.        , 0.        , 0.40603536],
           [0.        , 0.32476988, 0.32476988, 0.35046023, 0.        ]]]
      expected_qpc = [
          [[1.],
           [1.]],
          [[1.],
           [1.]],
          [[1.],
           [1.]],
          [[1.],
           [1.]],
          [[1.],
           [1.]]]
      expected_spc = [
          [[0.],
           [0.]],
          [[1.],
           [0.]],
          [[1.],
           [0.]],
          [[0.],
           [1.]],
          [[0.],
           [1.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_qpc, actual_qpc, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_spc, actual_spc, rtol=1e-05, atol=1e-05)

  def testCCTAttentionLayerStepByStep(self):
    with self.session(use_gpu=True) as sess, self.SetEval(True):
      depth = 4
      p = layers_with_attention.CCTAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = True
      p.num_attention_heads = 2
      p.gating_tpl.hidden_layer_dim = 2
      p.gating_tpl.noise_std = 5.0
      p.gating_tpl.noise_warmup_steps = 100
      x_atten = layers_with_attention.CCTAttentionLayer(p)

      (source_vecs, _, _, _,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)
      source_padding = tf.zeros([5, 2])

      ctx1, probs1, _, _ = x_atten.FPropDefaultTheta(source_vecs,
                                                     source_padding)
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
      self.assertAllClose(ctx1_v, ctx2_v)
      self.assertAllClose(probs1_v, probs2_v)

  def testCCTAttentionLayerCrossAttenTraining(self):
    with self.session(use_gpu=True) as sess:
      depth = 4
      p = layers_with_attention.CCTAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      p.gating_tpl.hidden_layer_dim = 2
      p.gating_tpl.noise_std = 5.0
      p.gating_tpl.noise_warmup_steps = 100
      transformer_atten = layers_with_attention.CCTAttentionLayer(p)

      (query_vec, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

      ctx, probs, qpc, spc = transformer_atten.FPropDefaultTheta(
          query_vec, aux_paddings, aux_vecs)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs, actual_qpc, actual_spc = sess.run(
          [ctx, probs, qpc, spc])
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-1.9043474 ,  1.6999874 ,  0.4292767 , -0.22491673],
           [-0.84242177, -0.50577486,  0.29762083,  1.0505756 ]],
          [[-0.33607534,  2.5800223 , -1.3375163 , -0.90643084],
           [-0.4973639 , -0.17019022, -1.1589761 ,  1.8265318 ]],
          [[ 1.1859869 ,  1.5021455 , -1.6327672 , -1.0553647 ],
           [-1.2359238 , -0.22244841,  0.19330817,  1.2650642 ]],
          [[-1.5131142 ,  0.49699292,  1.129034  , -0.11291274],
           [ 2.1162672 ,  0.6308829 , -1.0373113 , -1.7098385 ]],
          [[-0.9935959 ,  0.07386243, -0.6836246 ,  1.6033579 ],
           [-1.0807116 ,  0.85268646, -1.2622242 ,  1.4902495 ]]]
      expected_probs = [
          [[0.24303743, 0.        , 0.30685946, 0.        , 0.25564623,
            0.        , 0.1944569 ],
           [0.        , 0.28801104, 0.        , 0.34431183, 0.        ,
            0.36767715, 0.        ]],
          [[0.2644446 , 0.        , 0.23458862, 0.        , 0.23393473,
            0.        , 0.26703206],
           [0.        , 0.22837642, 0.        , 0.2820819 , 0.        ,
            0.4895417 , 0.        ]],
          [[0.2599384 , 0.        , 0.19412258, 0.        , 0.21307275,
            0.        , 0.33286628],
           [0.        , 0.27514488, 0.        , 0.35259444, 0.        ,
            0.3722607 , 0.        ]],
          [[0.24153353, 0.        , 0.3045342 , 0.        , 0.2569951 ,
            0.        , 0.19693717],
           [0.        , 0.36325702, 0.        , 0.26765382, 0.        ,
            0.36908916, 0.        ]],
          [[0.21663833, 0.        , 0.28198314, 0.        , 0.29308724,
            0.        , 0.20829134],
           [0.        , 0.2337277 , 0.        , 0.319759  , 0.        ,
            0.44651327, 0.        ]]]
      expected_qpc = [
          [[0.5       ],
           [0.5818492 ]],
          [[0.541141  ],
           [0.55023897]],
          [[0.56948507],
           [0.5499979 ]],
          [[0.5166038 ],
           [0.58645904]],
          [[0.54155153],
           [0.5       ]]]
      expected_spc = [
          [[0.09838167],
           [0.5       ]],
          [[0.51203823],
           [0.22011107]],
          [[0.27349436],
           [0.5230051 ]],
          [[0.5       ],
           [0.0911701 ]],
          [[0.2730832 ],
           [0.5       ]],
          [[0.54982626],
           [0.44889307]],
          [[0.10193098],
           [0.11123485]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_qpc, actual_qpc, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_spc, actual_spc, rtol=1e-05, atol=1e-05)

  def testCCTAttentionLayerCrossAttenEval(self):
    with self.session(use_gpu=True) as sess, self.SetEval(True):
      depth = 4
      p = layers_with_attention.CCTAttentionLayer.Params()
      p.name = 'transformer_atten'
      p.source_dim = depth
      p.is_masked = False
      p.num_attention_heads = 2
      p.gating_tpl.hidden_layer_dim = 2
      p.gating_tpl.noise_std = 5.0
      p.gating_tpl.noise_warmup_steps = 100
      transformer_atten = layers_with_attention.CCTAttentionLayer(p)

      (query_vec, _, aux_vecs, aux_paddings,
       _) = self._testTransformerAttentionLayerInputs(depth=depth)

      ctx, probs, qpc, spc = transformer_atten.FPropDefaultTheta(
          query_vec, aux_paddings, aux_vecs)
      tf.global_variables_initializer().run()
      actual_ctx, actual_probs, actual_qpc, actual_spc = sess.run(
          [ctx, probs, qpc, spc])
      tf.logging.info(np.array_repr(actual_ctx))
      tf.logging.info(np.array_repr(actual_probs))
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [
          [[-1.5939784 ,  0.8543072 ,  0.84722424, -0.10755297],
           [-0.7121205 , -1.2363338 ,  1.1559415 ,  0.7925127 ]],
          [[-0.09044743,  1.6572162 , -0.87628996, -0.69047904],
           [-0.4314113 , -0.90650195, -0.35488802,  1.6928028 ]],
          [[ 1.3591317 ,  0.5376119 , -1.1282029 , -0.7685402 ],
           [-1.0880733 , -0.9078379 ,  0.9976803 ,  0.9982312 ]],
          [[-1.1870676 , -0.37413225,  1.5655125 , -0.00431258],
           [ 1.62277   ,  0.02716666, -0.7765793 , -0.87335706]],
          [[-0.6675403 , -0.8283625 , -0.18727894,  1.6831816 ],
           [-1.113929  ,  0.13246097, -0.57226247,  1.5537308 ]]]
      expected_probs = [
          [[0.25      , 0.        , 0.25      , 0.        , 0.25      ,
            0.        , 0.25      ],
           [0.        , 0.33333334, 0.        , 0.33333334, 0.        ,
            0.33333334, 0.        ]],
          [[0.25      , 0.        , 0.25      , 0.        , 0.25      ,
            0.        , 0.25      ],
           [0.        , 0.33333334, 0.        , 0.33333334, 0.        ,
            0.33333334, 0.        ]],
          [[0.25      , 0.        , 0.25      , 0.        , 0.25      ,
            0.        , 0.25      ],
           [0.        , 0.33333334, 0.        , 0.33333334, 0.        ,
            0.33333334, 0.        ]],
          [[0.25      , 0.        , 0.25      , 0.        , 0.25      ,
            0.        , 0.25      ],
           [0.        , 0.33333334, 0.        , 0.33333334, 0.        ,
            0.33333334, 0.        ]],
          [[0.25      , 0.        , 0.25      , 0.        , 0.25      ,
            0.        , 0.25      ],
           [0.        , 0.33333334, 0.        , 0.33333334, 0.        ,
            0.33333334, 0.        ]]]
      expected_qpc = [
          [[1.],
           [1.]],
          [[1.],
           [1.]],
          [[1.],
           [1.]],
          [[1.],
           [1.]],
          [[1.],
           [1.]]]
      expected_spc = [
          [[0.],
           [1.]],
          [[1.],
           [0.]],
          [[0.],
           [1.]],
          [[1.],
           [0.]],
          [[0.],
           [1.]],
          [[1.],
           [0.]],
          [[0.],
           [0.]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_probs, actual_probs, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_qpc, actual_qpc, rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_spc, actual_spc, rtol=1e-05, atol=1e-05)


class SelfAttentiveLayerTest(test_utils.TestCase):

  def testFPropForTrain(self):
    with self.session(use_gpu=False) as session:
      # time = 5, batch = 4, depth = 2
      features = tf.constant(np.random.normal(size=(5, 4, 2)), dtype=tf.float32)
      paddings = tf.constant(
          [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0],
           [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
          dtype=tf.float32)
      features = tf.transpose(features, [1, 0, 2])
      paddings = tf.transpose(paddings, [1, 0])
      # init parameters for the pooling layer
      params = layers_with_attention.SelfAttentiveLayer.Params()
      params.name = 'self_attentive_pooling'
      params.num_heads = 3
      params.input_dim = 2
      params.hidden_dim = 7
      params.penalty_coef = 1.0
      params.penalty_terms = [1.0, 0.33, 0.01]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      # forward through the layer
      with py_utils.AuxLossContext() as aux_loss_ctx:
        att_layer = layers_with_attention.SelfAttentiveLayer(params)
        outputs = att_layer.FProp(att_layer.theta, features, paddings=paddings)
        tf.global_variables_initializer().run()
        outputs, aux_loss = session.run([outputs, aux_loss_ctx.aux_losses[0]])
        # check the shapes of the resulted tensors
        self.assertEqual(
            outputs.shape,
            (features.shape[0], params.num_heads, params.input_dim))
        self.assertEqual(aux_loss.shape, (features.shape[0],))


if __name__ == '__main__':
  tf.test.main()
