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
"""Tests for layers."""

import math

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import bn_layers
from lingvo.core import cluster_factory
from lingvo.core import gpipe
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import symbolic
from lingvo.core import test_utils
from lingvo.core import tshape
import numpy as np


class BatchNormLayerTest(test_utils.TestCase, parameterized.TestCase):

  def testBatchNormLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn'
      params.dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.add_stats_to_moving_average_variables = True
      layers.BatchNormLayer(params)
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'bn/beta/var:0', 'bn/gamma/var:0', 'bn/moving_mean/var:0',
          'bn/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)
      self.assertEqual(['bn/moving_mean/var:0', 'bn/moving_variance/var:0'],
                       [x.name for x in tf.moving_average_variables()])

  def testBatchNormLayerMoments(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)

      in_padding1 = tf.zeros([2, 2, 8, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 2, 8, 2]), dtype=tf.float32)
      mean1, var1 = bn_layers.ComputeMoments(
          bn_in1, in_padding1, reduce_over_dims=[0, 1, 2])
      mean2, var2 = tf.nn.moments(bn_in1, [0, 1, 2])

      in_padding2 = tf.ones([2, 2, 8, 1], dtype=tf.float32)
      bn_in2 = tf.constant(
          np.random.normal(-0.3, 1.0, [2, 2, 8, 2]), dtype=tf.float32)
      in_padding3 = tf.concat([in_padding1, in_padding2], 1)
      bn_in3 = tf.concat([bn_in1, bn_in2], 1)
      mean3, var3 = bn_layers.ComputeMoments(
          bn_in3, in_padding3, reduce_over_dims=[0, 1, 2])
      mean4, var4 = tf.nn.moments(bn_in3, [0, 1, 2])

      mean_diff = tf.reduce_sum(tf.square(mean3 - mean4))
      var_diff = tf.reduce_sum(tf.square(var3 - var4))

      self.evaluate(tf.global_variables_initializer())

      self.assertAllClose(self.evaluate(mean2), self.evaluate(mean1))
      self.assertAllClose(self.evaluate(var2), self.evaluate(var1))
      self.assertAllClose(self.evaluate(mean3), self.evaluate(mean1))
      self.assertAllClose(self.evaluate(var3), self.evaluate(var1))
      # Since tf.nn.moments() doesn't support padding, it is expected to produce
      # different results than our own implementation (of moments).
      self.assertAllClose(0.095987, self.evaluate(mean_diff))
      self.assertAllClose(0.364456, self.evaluate(var_diff))

  @parameterized.named_parameters(('F32Input', tf.float32, 47.8371887),
                                  ('BF16Input', tf.bfloat16, 47.8373))
  def testBatchNormLayerFProp(self, input_dtype, expected_sig2):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn'
      params.dim = 3
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      bn_layer = layers.BatchNormLayer(params)
      in_padding1 = tf.zeros([2, 8, 1], dtype=input_dtype)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=input_dtype)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(0.0, self.evaluate(sig1), atol=1e-5)
      self.assertAllClose(expected_sig2, self.evaluate(sig2))

  def testBatchNormLayerFPropUseGlobalStatsForTraining(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn'
      params.dim = 3
      params.use_moving_avg_in_training = True
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      bn_layer = layers.BatchNormLayer(params)
      in_padding1 = tf.zeros([2, 8, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(2.6593573, self.evaluate(sig1), atol=1e-5)
      self.assertAllClose(15.464208, self.evaluate(sig2))

  def testBatchNormLayerFPropWithUpdateUseGlobalStatsForTraining(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn'
      params.dim = 3
      params.use_moving_avg_in_training = True
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      bn_layer = layers.BatchNormLayer(params)
      in_padding1 = tf.zeros([2, 8, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)

      # get updates which should be invoked during training step
      # but we call them here, so that UpdateBatchNormVars is tested too
      bn_update_dict = py_utils._get_batch_norm_updates_dict()
      bn_update_list = list(bn_update_dict.keys())

      self.evaluate(tf.global_variables_initializer())
      self.evaluate(bn_update_list)
      # IMPORTANT: Keep these values consistent with the corresponding
      # test in py_utils_eager_test.py
      self.assertAllClose(2.6575434, self.evaluate(sig1), atol=1e-5)
      self.assertAllClose(15.473802, self.evaluate(sig2))

  def testBatchNormLayerMomentsForConv(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)

      in_padding1 = tf.zeros([2, 8, 1, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 4, 3]), dtype=tf.float32)
      mean1, var1 = bn_layers.ComputeMoments(
          bn_in1, in_padding1, reduce_over_dims=[0, 1, 2])
      mean2, var2 = tf.nn.moments(bn_in1, [0, 1, 2])

      in_padding2 = tf.ones([2, 8, 1, 1], dtype=tf.float32)
      bn_in2 = tf.constant(
          np.random.normal(-0.3, 1.0, [2, 8, 4, 3]), dtype=tf.float32)
      in_padding3 = tf.concat([in_padding1, in_padding2], 1)
      bn_in3 = tf.concat([bn_in1, bn_in2], 1)
      mean3, var3 = bn_layers.ComputeMoments(
          bn_in3, in_padding3, reduce_over_dims=[0, 1, 2])
      mean4, var4 = tf.nn.moments(bn_in3, [0, 1, 2])

      mean_diff = tf.reduce_sum(tf.square(mean3 - mean4))
      var_diff = tf.reduce_sum(tf.square(var3 - var4))

      self.evaluate(tf.global_variables_initializer())

      self.assertAllClose(self.evaluate(mean2), self.evaluate(mean1))
      self.assertAllClose(self.evaluate(var2), self.evaluate(var1))
      self.assertAllClose(self.evaluate(mean3), self.evaluate(mean1))
      self.assertAllClose(self.evaluate(var3), self.evaluate(var1))
      self.assertAllClose(0.1726295, self.evaluate(mean_diff))
      self.assertAllClose(0.5592572093009949, self.evaluate(var_diff))

  def testBatchNormLayerFPropForConv(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn_conv'
      params.dim = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      bn_layer = layers.BatchNormLayer(params)
      in_padding1 = tf.zeros([2, 8, 1, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 4, 32]), dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(0.0, self.evaluate(sig1), atol=1e-4)
      self.assertAllClose(2039.398681, self.evaluate(sig2))

  @parameterized.named_parameters(
      ('FuseEvalNoFreeze', True, True, False),
      ('NoFuseEvalNoFreeze', False, True, False),
      ('FuseTrainingFreeze', True, False, True),
      ('NoFuseTrainingFreeze', False, False, True),
  )
  def testBatchNormLayerFPropForConvWithFusedEvalWithFreezeBNStats(
      self, use_fused_batch_norm_for_eval, do_eval, freeze_bn_stats):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayer.Params()
      params.name = 'bn_conv'
      params.dim = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.use_fused_batch_norm_for_eval = use_fused_batch_norm_for_eval
      params.freeze_bn_stats = freeze_bn_stats
      with cluster_factory.ForTestingWorker(
          mode='sync', job='trainer_client', do_eval=True):
        bn_layer = params.Instantiate()
        bn_layer._epsilon = 0.0  # Enables a lower tolerance in the test check.
        in_padding1 = tf.zeros([2, 4, 1, 1], dtype=tf.float32)
        np_in1 = np.random.normal(0.1, 0.5, [2, 4, 1, 32])
        bn_in1 = tf.constant(np_in1, dtype=tf.float32)
        bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1)
        self.evaluate(tf.global_variables_initializer())
        # Moving mean and variance are set to defaults, we set gamma and beta
        # through assignment such that the outputs are inputs * 2 + 1.
        moving_mean_init = np.zeros(bn_layer.vars.moving_mean.shape.as_list())
        moving_variance_init = np.ones(
            bn_layer.vars.moving_variance.shape.as_list())
        self.evaluate([
            tf.assign(bn_layer.vars.gamma,
                      np.ones(bn_layer.vars.gamma.shape.as_list())),
            tf.assign(bn_layer.vars.beta,
                      np.ones(bn_layer.vars.beta.shape.as_list())),
            tf.assign(bn_layer.vars.moving_mean, moving_mean_init),
            tf.assign(bn_layer.vars.moving_variance, moving_variance_init),
        ])
        self.assertAllClose(
            np_in1 * 2. + 1., self.evaluate(bn_out), atol=1e-5, rtol=1e-5)
        # check moving stats modified or not
        moving_mean = self.evaluate(bn_layer.vars.moving_mean)
        moving_variance = self.evaluate(bn_layer.vars.moving_variance)
        if do_eval or freeze_bn_stats:
          self.assertAllClose(moving_mean, moving_mean_init)
          self.assertAllClose(moving_variance, moving_variance_init)


class CategoricalBNTest(test_utils.TestCase, parameterized.TestCase):

  def testConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = bn_layers.CategoricalBN.Params()
      params.name = 'bn'
      params.dim = 2
      params.class_emb_dim = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      bn_layers.CategoricalBN(params)
      bn_vars = tf.get_collection('CategoricalBN_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'bn/beta/var:0', 'bn/gamma/var:0', 'bn/moving_mean/var:0',
          'bn/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)
      self.assertEqual(['bn/moving_mean/var:0', 'bn/moving_variance/var:0'],
                       [x.name for x in tf.moving_average_variables()])

  def testFPropSameClass(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = bn_layers.CategoricalBN.Params()
      params.name = 'bn'
      params.dim = 3
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.class_emb_dim = 4

      bn_layer = params.Instantiate()
      in_padding1 = tf.zeros([2, 8, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)
      domain_in1 = tf.one_hot([0, 0],
                              depth=params.class_emb_dim,
                              dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1, domain_in1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(0.0, self.evaluate(sig1), atol=1e-5)
      self.assertAllClose(47.8371887, self.evaluate(sig2))

  def testFPropDifferentClasses(self):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = bn_layers.CategoricalBN.Params()
      params.name = 'bn'
      params.dim = 3
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.class_emb_dim = 4

      bn_layer = params.Instantiate()
      in_padding1 = tf.zeros([4, 8, 1], dtype=tf.float32)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [4, 8, 3]), dtype=tf.float32)
      domain_in1 = tf.one_hot([0, 1, 1, 2],
                              depth=params.class_emb_dim,
                              dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1, in_padding1, domain_in1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      self.evaluate(tf.global_variables_initializer())

      sig1_v, sig2_v = sess.run([sig1, sig2])
      self.assertAllClose(0.0, sig1_v, atol=1e-5)
      self.assertAllClose(95.6266, sig2_v)


class GroupNormLayerTest(test_utils.TestCase, parameterized.TestCase):

  def testConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = bn_layers.GroupNormLayer.Params()
      params.name = 'gn'
      params.dim = 2
      params.num_groups = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      bn_layers.GroupNormLayer(params)
      gn_vars = tf.get_collection('GroupNormLayer_vars')
      gn_var_names = [x.name for x in gn_vars]
      expected_var_names = ['gn/beta/var:0', 'gn/gamma/var:0']
      self.assertEqual(expected_var_names, gn_var_names)

  @parameterized.named_parameters(
      ('Default', None),
      ('1e_3', 1e-3),
      ('1e_6', 1e-6),
  )
  def testFProp(self, epsilon=None):
    with self.session(use_gpu=True):
      params = bn_layers.GroupNormLayer.Params()
      params.name = 'gn'
      params.dim = 4
      params.num_groups = 2
      if epsilon is not None:
        params.epsilon = epsilon
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      gn_in = tf.reshape(np.arange(32, dtype=np.float32), [2, 2, 2, 4])

      gn_layer = bn_layers.GroupNormLayer(params)
      gn_out = gn_layer.FPropDefaultTheta(gn_in)

      tf.global_variables_initializer().run()
      if epsilon == 1e-6:
        base_block = np.array([[[-1.444444, -1.222222], [-0.555555, -0.333333]],
                               [[0.333333, 0.555555], [1.222222, 1.444444]]])
      else:
        base_block = np.array([[[-1.44440889, -1.22219217],
                                [-0.55554187, -0.33332515]],
                               [[0.33332515, 0.55554187],
                                [1.22219217, 1.44440889]]])
      expected_out = np.array([
          np.concatenate((base_block, base_block), -1),
          np.concatenate((base_block, base_block), -1)
      ])
      self.assertAllClose(expected_out, self.evaluate(gn_out), atol=1e-5)

  def testFPropWithPaddings(self):
    with self.session(use_gpu=True):
      params = bn_layers.GroupNormLayer.Params()
      params.name = 'gn'
      params.dim = 4
      params.num_groups = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      gn_in = tf.cast(
          tf.reshape(np.arange(32, dtype=np.float32), [2, 2, 2, 4]), tf.float32)
      paddings = tf.convert_to_tensor([[0, 0], [0, 1]], dtype=tf.float32)

      gn_layer = bn_layers.GroupNormLayer(params)
      gn_out, paddings_out = gn_layer.FPropDefaultTheta(gn_in, paddings)

      tf.global_variables_initializer().run()
      base_block1 = np.array([[[-1.44440889, -1.22219217],
                               [-0.55554187, -0.33332515]],
                              [[0.33332515, 0.55554187],
                               [1.22219217, 1.44440889]]])

      base_block2 = np.array([[[-1.2125355, -0.7275213], [0.7275213,
                                                          1.2125355]],
                              [[2.6675782, 3.1525922], [4.607635, 5.092649]]])

      expected_out = np.array([
          np.concatenate((base_block1, base_block1), -1),
          np.concatenate((base_block2, base_block2), -1)
      ])
      print(self.evaluate(gn_out))
      self.assertAllClose(expected_out, self.evaluate(gn_out), atol=1e-5)
      self.assertAllEqual(self.evaluate(paddings), self.evaluate(paddings_out))

  @parameterized.named_parameters(
      ('F32FPropF32Input', tf.float32, tf.float32, 31.040909),
      ('F32FPropBF16Input', tf.float32, tf.bfloat16, 31.040909),
      ('BF16FPropF32Input', tf.bfloat16, tf.float32, 31.),
      ('BF16FPropBF16Input', tf.bfloat16, tf.bfloat16, 31.),
  )
  def testFPropDtypes(self, fprop_dtype, input_dtype, expected_sum=0.):
    with self.session(use_gpu=True):
      params = bn_layers.GroupNormLayer.Params()
      params.name = 'gn'
      params.dim = 4
      params.num_groups = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.fprop_dtype = fprop_dtype
      params.random_seed = 123
      gn_in = tf.cast(
          tf.reshape(np.arange(32, dtype=np.float32), [2, 2, 2, 4]),
          input_dtype)
      paddings = tf.convert_to_tensor([[0, 0], [0, 1]], dtype=input_dtype)

      gn_layer = bn_layers.GroupNormLayer(params)
      gn_out, _ = gn_layer.FPropDefaultTheta(gn_in, paddings)

      tf.global_variables_initializer().run()
      self.assertAllClose(expected_sum, self.evaluate(tf.reduce_sum(gn_out)))

  def testFPropWithPaddings3DInput(self):
    with self.session(use_gpu=True):
      params = bn_layers.GroupNormLayer.Params()
      params.name = 'gn'
      params.dim = 4
      params.num_groups = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      gn_in = np.reshape(np.arange(16, dtype=np.float32), [2, 2, 1, 4])
      paddings = np.array([[0, 0], [0, 1]], dtype=np.float32)

      gn_layer = bn_layers.GroupNormLayer(params)
      gn_out, paddings_out = gn_layer.FPropDefaultTheta(
          tf.convert_to_tensor(gn_in), tf.convert_to_tensor(paddings))

      params_3d = params.Copy().Set(input_rank=3, name='gn_3d')
      gn_layer_3d = bn_layers.GroupNormLayer(params_3d)
      gn_out_3d, paddings_out_3d = gn_layer_3d.FPropDefaultTheta(
          tf.convert_to_tensor(gn_in.reshape([2, 2, 4])),
          tf.convert_to_tensor(paddings))

      tf.global_variables_initializer().run()
      print(self.evaluate(gn_out))
      # Tests against reference.
      self.assertAllClose(
          self.evaluate(gn_out_3d),
          self.evaluate(gn_out).reshape([2, 2, 4]),
          atol=1e-5)
      self.assertAllEqual(
          self.evaluate(paddings_out_3d), self.evaluate(paddings_out))

  @parameterized.named_parameters(('4D',), ('3D', 3))
  def testFPropCumulativeMode(self, input_rank=4):
    with self.session(use_gpu=True):
      params = bn_layers.GroupNormLayer.Params()
      params.name = 'gn'
      params.dim = 2
      params.num_groups = 2
      params.cumulative = True
      params.input_rank = input_rank
      # gn_in[0]: [[0, 1], [2, 3], [4, 5], [6, 7]]
      # gn_in[1]: [[8, 9], [10, 11], [12, 13], [14, 15]]
      input_shape = [2, 4, 1, 2] if input_rank == 4 else [2, 4, 2]
      gn_in = tf.reshape(np.arange(16, dtype=np.float32), input_shape)
      paddings = tf.zeros([2, 4], tf.float32)
      gn_layer = bn_layers.GroupNormLayer(params)
      gn_out, _ = gn_layer.FPropDefaultTheta(gn_in, paddings)

      tf.global_variables_initializer().run()
      base_block = np.array([[0., 0.], [1.4128014, 1.4128014],
                             [1.5487288, 1.5487288], [1.6033384, 1.6033384]])

      expected_out = np.stack([base_block, base_block],
                              axis=0).reshape(input_shape)
      self.assertAllClose(expected_out, self.evaluate(gn_out), atol=1e-5)

  @parameterized.named_parameters(
      ('Basic',),
      ('Group1', 1, 1),
      ('Stride2', 2),
      ('Stride2Group1', 2, 1),
      ('Stride4', 4),
      ('TfLiteCompatible', 1, 2, True),
  )
  def testStreamStep(self, stride=1, num_groups=2, tflite_compatible=False):
    py_utils.FLAGS.tflite_compatible = tflite_compatible
    batch, max_seqlen, input_dim = 2, 16, 4
    p = bn_layers.GroupNormLayer.Params().Set(
        name='gn',
        dim=input_dim,
        num_groups=num_groups,
        cumulative=True,
        input_rank=4)

    l = p.Instantiate()
    init_op = tf.global_variables_initializer()

    np.random.seed(None)
    inputs = np.random.normal(
        0.1, 0.5, [batch, max_seqlen, 1, input_dim]).astype(np.float32)
    print(f'np.sum(inputs): {np.sum(inputs)}')
    inputs = tf.convert_to_tensor(inputs)

    seqlen = np.random.randint(
        low=1, high=max_seqlen + 1, size=(batch,), dtype=np.int32)
    print(repr(seqlen))
    seqlen = tf.convert_to_tensor(seqlen)
    paddings = py_utils.PaddingsFromLengths(seqlen, max_seqlen)
    expanded_paddings = tf.reshape(paddings,
                                   py_utils.GetShape(paddings) + [1, 1])
    base_outs, _ = l.FProp(l.theta, inputs, paddings)
    base_outs *= 1. - expanded_paddings

    # Runs N//stride step each with input seqlen=stride.
    assert max_seqlen % stride == 0
    actual_outs = []
    state = l.zero_state(batch)
    for i in range(max_seqlen // stride):
      output, _, state = l.StreamStep(
          l.theta, inputs[:, stride * i:stride * (i + 1), :, :],
          paddings[:, stride * i:stride * (i + 1)], state)
      actual_outs.append(output)
    actual_outs = tf.concat(actual_outs, axis=1)
    actual_outs *= 1. - expanded_paddings

    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      expected, actual = sess.run([base_outs, actual_outs])
      print(repr(expected))
      print(repr(actual))
      print(f'np.sum(np.abs(expected)): {np.sum(np.abs(expected))}')
      print(f'np.sum(np.abs(actual)): {np.sum(np.abs(actual))}')
      self.assertAllClose(expected, actual)

  def testStreamStepLeadingPaddings(self):
    """Tests leading paddings case, useful for local atten with right ctx."""
    stride, num_groups = 2, 2
    batch, max_seqlen, input_dim = 2, 8, 4
    p = bn_layers.GroupNormLayer.Params().Set(
        name='gn',
        dim=input_dim,
        num_groups=num_groups,
        cumulative=True,
        input_rank=4)

    l = p.Instantiate()
    init_op = tf.global_variables_initializer()

    np.random.seed(None)
    inputs = np.random.normal(
        0.1, 0.5, [batch, max_seqlen, 1, input_dim]).astype(np.float32)
    print(f'np.sum(inputs): {np.sum(inputs)}')
    inputs_t = tf.convert_to_tensor(inputs)

    # The upperbound is always max_seqlen-1, so the batch is always padded.
    seqlen = np.random.randint(
        low=1, high=max_seqlen, size=(batch,), dtype=np.int32)
    print('seqlen: {seqlen}')
    paddings = py_utils.PaddingsFromLengths(
        tf.convert_to_tensor(seqlen), max_seqlen)

    shift_inputs = np.array(inputs)
    for i in range(batch):
      shift_inputs[i] = np.roll(shift_inputs[i], max_seqlen - seqlen[i], axis=0)
    shift_inputs_t = tf.convert_to_tensor(shift_inputs)

    # Has the same number of tokens as paddings per example
    leading_paddings = 1 - py_utils.PaddingsFromLengths(
        max_seqlen - tf.convert_to_tensor(seqlen), max_seqlen)

    def expand_pad(pad):  # pylint:disable=invalid-name
      return py_utils.AppendDims(pad, 2)

    def stream(l, inputs, paddings):  # pylint:disable=invalid-name
      state = l.zero_state(batch)
      all_outs = []
      for i in range(max_seqlen // stride):
        step_inputs = inputs[:, stride * i:stride * (i + 1)]
        step_paddings = paddings[:, stride * i:stride * (i + 1)]
        output, _, state = l.StreamStep(l.theta, step_inputs, step_paddings,
                                        state)
        all_outs.append(output)
      all_outs = tf.concat(all_outs, axis=1)
      return all_outs * (1. - expand_pad(paddings))

    base_outs = stream(l, inputs_t, paddings)
    actual_outs = stream(l, shift_inputs_t, leading_paddings)

    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      expected, actual = sess.run([base_outs, actual_outs])
      for i in range(batch):
        actual[i] = np.roll(actual[i], -(max_seqlen - seqlen[i]), axis=0)
      print(f'expected: {repr(expected)}')
      print(f'actual: {repr(actual)}')
      print(f'np.sum(np.abs(expected)): {np.sum(np.abs(expected))}')
      print(f'np.sum(np.abs(actual)): {np.sum(np.abs(actual))}')
      self.assertAllClose(expected, actual)


class ConvLayerTest(test_utils.TestCase):
  """Tests conv layers.

  Note that there are multiple subclasses of BaseConv2DLayer and most cases
  are tested via the concrete Conv2DLayer. Other tests are done against
  other subclasses to cover key differences.
  """

  def testConv2DLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.Conv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      layers.Conv2DLayer(params)
      conv_vars = tf.get_collection('Conv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/w/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'conv/beta/var:0', 'conv/gamma/var:0', 'conv/moving_mean/var:0',
          'conv/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)

  def testDepthwiseConv2DLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      layers.DepthwiseConv2DLayer(params)
      conv_vars = tf.get_collection('DepthwiseConv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/w/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'conv/beta/var:0', 'conv/gamma/var:0', 'conv/moving_mean/var:0',
          'conv/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)

  def testDepthwiseConv2DLayerModuleInterface(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv1'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv1 = layers.DepthwiseConv2DLayer(params)
      params.name = 'conv2'
      conv2 = layers.DepthwiseConv2DLayer(params)

      def ModuleName(m):
        return m.name

      conv1_variables = [v.name for v in conv1.variables]
      conv1_submodules = [ModuleName(v) for v in conv1.submodules]
      conv2_variables = [v.name for v in conv2.variables]
      conv2_submodules = [ModuleName(v) for v in conv2.submodules]
      expected_conv1_vars = [
          'conv1/w/var:0', 'conv1/moving_mean/var:0',
          'conv1/moving_variance/var:0', 'conv1/beta/var:0', 'conv1/gamma/var:0'
      ]
      expected_conv2_vars = [
          'conv2/w/var:0', 'conv2/moving_mean/var:0',
          'conv2/moving_variance/var:0', 'conv2/beta/var:0', 'conv2/gamma/var:0'
      ]
      expected_conv1_modules = ['bbf_BatchNormLayer_conv1']
      expected_conv2_modules = ['bbf_BatchNormLayer_conv2']
      self.assertCountEqual(expected_conv1_vars, conv1_variables)
      self.assertCountEqual(expected_conv2_vars, conv2_variables)
      self.assertCountEqual(expected_conv1_modules, conv1_submodules)
      self.assertCountEqual(expected_conv2_modules, conv2_submodules)

  def testSeparableConv2DLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.SeparableConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      params.Instantiate()
      # Vars for the outer conv layer.
      conv_vars = tf.get_collection('SeparableConv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/w/var:0']
      self.assertSetEqual(set(expected_var_names), set(conv_var_names))
      # Vars for the inner depthwise layer.
      conv_vars = tf.get_collection('DepthwiseConv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      expected_var_names = ['conv/depthwise_conv/w/var:0']
      self.assertSetEqual(set(expected_var_names), set(conv_var_names))
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          # Outer conv batchnorm.
          'conv/beta/var:0',
          'conv/gamma/var:0',
          'conv/moving_mean/var:0',
          'conv/moving_variance/var:0',
          # Inner depthwise batchnorm.
          'conv/depthwise_conv/beta/var:0',
          'conv/depthwise_conv/gamma/var:0',
          'conv/depthwise_conv/moving_mean/var:0',
          'conv/depthwise_conv/moving_variance/var:0',
      ]
      self.assertSetEqual(set(expected_var_names), set(bn_var_names))

  def testConv2DLayerWithBiasConstruction(self):
    """Tests Conv2DLayer with only bias and without batch normalization."""
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.Conv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      params.bias = True
      params.batch_norm = False
      layers.Conv2DLayer(params)
      conv_vars = tf.get_collection('Conv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      # Has both 'w' and 'b'.
      expected_var_names = ['conv/w/var:0', 'conv/b/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      # No BatchNorm variables.
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = []
      self.assertEqual(expected_var_names, bn_var_names)

  def testDepthwiseConv2DLayerWithBiasConstruction(self):
    """Tests DepthwiseConv2D with only bias and without batch normalization."""
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      params.bias = True
      params.batch_norm = False
      layers.DepthwiseConv2DLayer(params)
      conv_vars = tf.get_collection('DepthwiseConv2DLayer_vars')
      conv_var_names = [x.name for x in conv_vars]
      # Has both 'w' and 'b'.
      expected_var_names = ['conv/w/var:0', 'conv/b/var:0']
      self.assertEqual(expected_var_names, conv_var_names)
      # No BatchNorm variables.
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = []
      self.assertEqual(expected_var_names, bn_var_names)

  def testConv2DLayerOutShape(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.Conv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = layers.Conv2DLayer(params)
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 5, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 10, 5, 32])

  def testDepthwiseConv2DLayerOutShape(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = layers.DepthwiseConv2DLayer(params)
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 5, 96])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 10, 5, 96])

  def testSeparableConv2DLayerOutShape(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.SeparableConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = params.Instantiate()
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 5, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 10, 5, 32])

  def testConv2DLayerWithDilationOutShape(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.Conv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [1, 1]
      params.dilation_rate = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = layers.Conv2DLayer(params)
      # dilation_rate does not change output shape.
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 10, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 20, 10, 32])

  def testDepthwiseConv2DLayerWithDilationOutShape(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.DepthwiseConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [1, 1]
      params.dilation_rate = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = layers.DepthwiseConv2DLayer(params)
      # dilation_rate does not change output shape.
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 10, 96])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 20, 10, 96])

  def testSeparableConv2DLayerWithDilationOutShape(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.SeparableConv2DLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 32]
      params.filter_stride = [1, 1]
      params.dilation_rate = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = params.Instantiate()
      # dilation_rate does not change output shape.
      in_shape = [None, None, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, None, 10, 32])
      in_shape = [None, 20, 10, 3]
      out_shape = conv_layer.OutShape(in_shape)
      self.assertEqual(out_shape, [None, 20, 10, 32])

  def testConvPoolComputeOutPadding(self):
    with self.session(use_gpu=True):
      in_padding = tf.constant(
          [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]],
          dtype=tf.float32)
      out_padding = layers._ComputeConvOutputPadding(in_padding, 2, 2)
      expected_out_padding = [[1, 1, 0, 0, 0, 1, 1, 0],
                              [1, 1, 0, 0, 0, 1, 1, 0]]

      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(expected_out_padding,
                          self.evaluate(out_padding).tolist())

  def testConvPoolComputeOutPaddingUnevenStride(self):
    with self.session(use_gpu=True):
      in_padding = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
                               dtype=tf.float32)
      out_padding = layers._ComputeConvOutputPadding(in_padding, 3, 3)
      expected_out_padding = [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]

      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(expected_out_padding,
                          self.evaluate(out_padding).tolist())

  def _checkConvLayerShapes(self,
                            input_shape,
                            filter_shape,
                            filter_stride,
                            dilation_rate=None,
                            depth_multiplier=None,
                            params_builder=layers.Conv2DLayer.Params):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = params_builder()
      params.name = 'conv'
      params.filter_shape = filter_shape
      params.filter_stride = filter_stride
      if dilation_rate:
        params.dilation_rate = dilation_rate
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      if depth_multiplier is not None:
        params.depth_multiplier = depth_multiplier
      conv_layer = params.Instantiate()

      inp = tf.random.uniform(input_shape)
      inp_pad = tf.floor(0.5 + tf.random.uniform(input_shape[:2]))
      out, out_pad = conv_layer.FPropDefaultTheta(inp, inp_pad)

    with self.session(use_gpu=True, graph=g):
      self.evaluate(tf.global_variables_initializer())
      out, out_pad = self.evaluate([out, out_pad])
      print(out.shape, out_pad.shape)
      # We expect conv_layer.OutShape can compute the actual output shape.
      self.assertAllEqual(out.shape, conv_layer.OutShape(inp.shape.as_list()))
      # We expect out_pad.shape matches the 1st 2 dimensions of out.
      self.assertAllEqual(out.shape[:2], out_pad.shape)

  def testConv2DLayerOutputShapes(self):
    self._checkConvLayerShapes([2, 4, 4, 3], [3, 3, 3, 32], [1, 1])
    self._checkConvLayerShapes([2, 4, 4, 3], [3, 3, 3, 32], [2, 2])
    self._checkConvLayerShapes([2, 10, 4, 3], [3, 3, 3, 32], [3, 3])

    self._checkConvLayerShapes([2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
                               dilation_rate=[2, 2])
    self._checkConvLayerShapes([2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
                               dilation_rate=[3, 3])

  def testDepthwiseConv2DLayerOutputShapes(self):
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [1, 1],
        params_builder=layers.DepthwiseConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [2, 2],
        params_builder=layers.DepthwiseConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [3, 3],
        params_builder=layers.DepthwiseConv2DLayer.Params)

    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
        dilation_rate=[2, 2],
        params_builder=layers.DepthwiseConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
        dilation_rate=[3, 3],
        params_builder=layers.DepthwiseConv2DLayer.Params)

  def testSeparableConv2DLayerOutputShapes(self):
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [1, 1],
        params_builder=layers.SeparableConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [2, 2],
        params_builder=layers.SeparableConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [3, 3],
        params_builder=layers.SeparableConv2DLayer.Params)
    # Dilations.
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
        dilation_rate=[2, 2],
        params_builder=layers.SeparableConv2DLayer.Params)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [1, 1],
        dilation_rate=[3, 3],
        params_builder=layers.SeparableConv2DLayer.Params)
    # Depth multiplier.
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [1, 1],
        params_builder=layers.SeparableConv2DLayer.Params,
        depth_multiplier=2)
    self._checkConvLayerShapes(
        [2, 4, 4, 3], [3, 3, 3, 32], [2, 2],
        params_builder=layers.SeparableConv2DLayer.Params,
        depth_multiplier=6)
    self._checkConvLayerShapes(
        [2, 10, 4, 3], [3, 3, 3, 32], [3, 3],
        params_builder=layers.SeparableConv2DLayer.Params,
        depth_multiplier=12)

  def _evalConvLayerFProp(self,
                          params_builder=layers.Conv2DLayer.Params,
                          batch_norm=True,
                          weight_norm=False,
                          bias=False,
                          activation='RELU',
                          conv_last=False,
                          strides=(2, 2),
                          dilation_rate=(1, 1),
                          bn_fold_weights=False,
                          is_eval=False,
                          quantized=False):
    self._ClearCachedSession()
    tf.reset_default_graph()
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = params_builder()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.filter_stride = strides
      params.dilation_rate = dilation_rate
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.conv_last = conv_last
      params.batch_norm = batch_norm
      params.bn_fold_weights = bn_fold_weights
      params.weight_norm = weight_norm
      params.bias = bias
      params.activation = activation

      if quantized:
        params.qdomain.default = quant_utils.PassiveAsymQDomain.Params()

      conv_layer = params.Instantiate()
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)

      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      output2, _ = conv_layer.FPropDefaultTheta(inputs1)
      self.evaluate(tf.global_variables_initializer())
      v1, v2 = self.evaluate([output1, output2])
      self.assertAllClose(v1, v2)
      return v1

  def testConv2DLayerFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.36669245,  0.91488785],
          [ 0.07532132,  0.        ]],
         [[ 0.34952009,  0.        ],
          [ 1.91783941,  0.        ]]],
        [[[ 0.28304493,  0.        ],
          [ 0.        ,  0.        ]],
         [[ 0.        ,  0.86575812],
          [ 0.        ,  1.60203481]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp()
    print('actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testDepthwiseConv2DLayerFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.93514717,  0.35602099,  0.        ,  0.51261222,  0.        ,
            1.4310323 ],
          [ 0.        ,  0.        ,  0.49176404,  0.        ,  1.01494753,
            0.51337928]],
         [[ 0.62087697,  0.34572476,  0.        ,  0.19352221,  0.47142431,
            0.        ],
          [ 0.81119895,  1.00890303,  0.90471351,  0.        ,  1.22736526,
            0.        ]]],
        [[[ 0.        ,  0.        ,  0.48927376,  0.        ,  0.74019426,
            0.        ],
          [ 0.        ,  0.        ,  1.49952257,  0.        ,  0.        ,
            0.        ]],
         [[ 0.29156703,  0.        ,  0.        ,  1.14509106,  0.        ,
            0.74238932],
          [ 0.91312039,  1.39783907,  0.        ,  1.47650909,  0.        ,
            0.37969294]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        params_builder=layers.DepthwiseConv2DLayer.Params)
    print('actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testSeparableConv2DLayerFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 =[
        [[[ 0.39866772,  0.        ],
          [ 1.36471784,  0.        ]],
         [[ 0.        ,  0.        ],
          [ 0.        ,  0.        ]]],
        [[[ 1.15356529,  0.1036691 ],
          [ 0.12865055,  0.61244327]],
         [[ 0.03609803,  1.81620765],
          [ 0.        ,  0.23052886]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        params_builder=layers.SeparableConv2DLayer.Params)
    print('actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testConv2DLayerWithDilationFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.        ,  0.48857123],
          [ 1.07320869,  0.        ],
          [ 0.        ,  0.1550007 ],
          [ 0.        ,  1.59097648]],
         [[ 0.        ,  0.        ],
          [ 0.20024362,  0.        ],
          [ 0.        ,  0.64265913],
          [ 1.52903616,  0.        ]],
         [[ 0.099805  ,  0.        ],
          [ 0.        ,  0.61720949],
          [ 1.31608474,  0.        ],
          [ 0.        ,  0.        ]],
         [[ 0.0175612 ,  0.        ],
          [ 0.        ,  0.17234094],
          [ 0.21719536,  0.        ],
          [ 1.68514931,  0.        ]]],
        [[[ 1.45240796,  0.        ],
          [ 0.        ,  0.        ],
          [ 0.72675145,  1.971596  ],
          [ 0.        ,  0.01062769]],
         [[ 0.        ,  1.70299017],
          [ 1.36936104,  1.29897082],
          [ 1.40132439,  1.74345171],
          [ 0.02585058,  0.29061913]],
         [[ 0.        ,  0.        ],
          [ 0.32962656,  0.05025356],
          [ 0.        ,  0.        ],
          [ 0.        ,  0.        ]],
         [[ 0.97244394,  0.        ],
          [ 0.23401484,  0.5722279 ],
          [ 0.        ,  0.40940297],
          [ 0.        ,  0.52711827]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(strides=[1, 1], dilation_rate=[2, 2])
    print('testConvLayerWithDilationFProp actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual, atol=1e-5)

  def testSeparableConv2DLayerWithDilationFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.21535617,  0.86965537],
          [ 2.11499524,  1.2463783 ],
          [ 0.        ,  0.39275286],
          [ 0.        ,  0.        ]],
         [[ 1.12706482,  1.37450278],
          [ 0.        ,  0.        ],
          [ 0.        ,  0.        ],
          [ 1.2390101 ,  0.22932449]],
         [[ 0.        ,  0.        ],
          [ 0.15051894,  1.32616639],
          [ 0.        ,  0.        ],
          [ 0.72912866,  0.47753802]],
         [[ 0.91655868,  0.        ],
          [ 0.88526261,  0.26690534],
          [ 0.        ,  0.26084688],
          [ 0.42923039,  0.        ]]],
        [[[ 0.82440329,  0.        ],
          [ 0.49015623,  0.52662987],
          [ 0.        ,  0.        ],
          [ 0.35344127,  0.        ]],
         [[ 0.        ,  0.        ],
          [ 0.        ,  0.        ],
          [ 0.43848675,  0.        ],
          [ 0.        ,  1.21124518]],
         [[ 1.1026746 ,  1.39578998],
          [ 0.        ,  0.        ],
          [ 0.34652925,  0.        ],
          [ 0.        ,  1.26868236]],
         [[ 0.91519427,  0.09030763],
          [ 0.        ,  0.59271163],
          [ 0.        ,  0.54207176],
          [ 0.        ,  0.        ]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        strides=[1, 1],
        dilation_rate=[2, 2],
        params_builder=layers.SeparableConv2DLayer.Params)
    print('testConvLayerWithDilationFProp actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  def testConv2DLayerConvFirstVsLastFProp(self):
    """Compare results of conv first vs. last."""
    # ... with batch_norm and activation disabled.
    self.assertAllClose(
        self._evalConvLayerFProp(
            batch_norm=False, activation='NONE', conv_last=False),
        self._evalConvLayerFProp(
            batch_norm=False, activation='NONE', conv_last=True))

  def testConv2DLayerFPropConvLast(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output1 = [
        [[[ 0.22165056,  0.20731729],
          [ 0.09577402, -0.15359652]],
         [[ 0.07151584,  0.03027298],
          [ 0.05370769,  0.0143405 ]]],
        [[[-0.08854639,  0.06143938],
          [-0.37708873,  0.00889082]],
         [[-0.58154356,  0.30798748],
          [-0.37575331,  0.54729235]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(conv_last=True)
    print(['ConvLast actual = ', np.array_repr(actual)])
    self.assertAllClose(expected_output1, actual)

  def testConv2DLayerConvWithBias(self):
    """Compare results with bias vs. with neither batch_norm nor bias."""
    # Results should match since bias is initialized to be 0.
    self.assertAllClose(
        self._evalConvLayerFProp(batch_norm=False, bias=False),
        self._evalConvLayerFProp(batch_norm=False, bias=True))

  def testConv2DLayerWeightNormFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [
        [[[ 0.37172362, 0.92405349],
          [ 0.07635488, 0.]],
         [[ 0.35431579, 0.],
          [ 1.94415355, 0.]]],
        [[[ 0.28692839, 0.],
          [ 0.        , 0.]],
         [[ 0.        , 0.87443149],
          [ 0.        , 1.61808443]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(weight_norm=True)
    print('actual1 = ', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testDepthwiseConv2DLayerWeightNormFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [
        [[[ 0.97023201,  0.37429881,  0.        ,  0.53157473,  0.        ,
            1.60764372],
          [ 0.        ,  0.        ,  0.50401598,  0.        ,  1.07683432,
            0.57673818]],
         [[ 0.644171  ,  0.36347377,  0.        ,  0.20068097,  0.50016963,
            0.        ],
          [ 0.8416335 ,  1.06069875,  0.92725372,  0.        ,  1.30220449,
            0.        ]]],
        [[[ 0.        ,  0.        ,  0.50146359,  0.        ,  0.78532791,
            0.        ],
          [ 0.        ,  0.        ,  1.53688192,  0.        ,  0.        ,
            0.        ]],
         [[ 0.302506  ,  0.        ,  0.        ,  1.18745029,  0.        ,
            0.83401161],
          [ 0.94737887,  1.46960247,  0.        ,  1.53112805,  0.        ,
            0.42655289]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        weight_norm=True, params_builder=layers.DepthwiseConv2DLayer.Params)
    print('actual1 = ', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testSeparableConv2DLayerWeightNormFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [
        [[[ 0.41837293,  0.        ],
          [ 1.39592457,  0.        ]],
         [[ 0.        ,  0.        ],
          [ 0.        ,  0.        ]]],
        [[[ 1.20513153,  0.11938372],
          [ 0.1284119 ,  0.6927582 ]],
         [[ 0.0227453 ,  2.05591369],
          [ 0.        ,  0.26530063]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalConvLayerFProp(
        weight_norm=True, params_builder=layers.SeparableConv2DLayer.Params)
    print('actual1 = ', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testConv2DLayerFoldedBatchNormFProp(self):
    actual_unfolded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=False)
    actual_folded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=True)
    print('testConvLayerFoldedBatchNormFProp folded = ',
          np.array_repr(actual_folded))
    print('testConvLayerFoldedBatchNormFProp unfolded = ',
          np.array_repr(actual_unfolded))
    self.assertAllClose(actual_folded, actual_unfolded)

  def testDepthwiseConv2DLayerFoldedBatchNormFProp(self):
    actual_unfolded = self._evalConvLayerFProp(
        batch_norm=True,
        bn_fold_weights=False,
        params_builder=layers.DepthwiseConv2DLayer.Params)
    actual_folded = self._evalConvLayerFProp(
        batch_norm=True,
        bn_fold_weights=True,
        params_builder=layers.DepthwiseConv2DLayer.Params)
    print('testDepthwiseConvLayerFoldedBatchNormFProp folded = ',
          np.array_repr(actual_folded))
    print('testDepthwiseConvLayerFoldedBatchNormFProp unfolded = ',
          np.array_repr(actual_unfolded))
    self.assertAllClose(actual_folded, actual_unfolded)

  def testSeparableConv2DLayerFoldedBatchNormFProp(self):
    actual_unfolded = self._evalConvLayerFProp(
        batch_norm=True,
        bn_fold_weights=False,
        params_builder=layers.SeparableConv2DLayer.Params)
    actual_folded = self._evalConvLayerFProp(
        batch_norm=True,
        bn_fold_weights=True,
        params_builder=layers.SeparableConv2DLayer.Params)
    print('testSeparableConvLayerFoldedBatchNormFProp folded = ',
          np.array_repr(actual_folded))
    print('testSeparableConvLayerFoldedBatchNormFProp unfolded = ',
          np.array_repr(actual_unfolded))
    self.assertAllClose(actual_folded, actual_unfolded)

  def testConvLayerFoldedBatchNormFPropEval(self):
    actual_unfolded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=False, is_eval=True)
    actual_folded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=True, is_eval=True)
    print('testConvLayerFoldedBatchNormFPropEval folded = ',
          np.array_repr(actual_folded))
    print('testConvLayerFoldedBatchNormFPropEval unfolded = ',
          np.array_repr(actual_unfolded))
    self.assertAllClose(actual_folded, actual_unfolded)

  def testConv2DLayerNoPadding(self):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(24332)
      p = layers.Conv2DLayerNoPadding.Params().Set(
          name='test', filter_shape=(3, 3, 3, 5), filter_stride=(2, 2))
      l = p.Instantiate()
      x = tf.random.normal(shape=[17, 64, 64, 3])
      y = l.FPropDefaultTheta(x)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      y_val = self.evaluate(y)

    self.assertEqual(y_val.shape, (17, 32, 32, 5))

  def testConvLayerFoldedBatchNormFPropQuantized(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [
        [[[ 0.36997819,  0.91361964],
          [ 0.07550576,  0.        ]],

         [[ 0.35487702,  0.        ],
          [ 1.92539668,  0.        ]]],
        [[[ 0.27937129,  0.        ],
          [ 0.        ,  0.        ]],

         [[ 0.        ,  0.86831617],
          [ 0.        ,  1.59317136]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace

    actual_folded = self._evalConvLayerFProp(
        batch_norm=True, bn_fold_weights=True, quantized=True)
    print('testConvLayerFoldedBatchNormFPropQuantized folded = ',
          np.array_repr(actual_folded))
    self.assertAllClose(actual_folded, expected_output)

  def testCausalConvLayerFProp(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [2, 1, 3, 2]
      params.filter_stride = [1, 1]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      params.causal_convolution = True
      params.activation = 'NONE'
      params.batch_norm = False

      conv_layer = layers.ConvLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3, 3]), dtype=tf.float32)
      # Change the input for the last two steps.
      inputs2 = tf.concat([inputs1[:, :2, :, :], inputs1[:, 2:, :, :] + 0.5], 1)

      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      output2, _ = conv_layer.FPropDefaultTheta(inputs2, in_padding1)
      self.evaluate(tf.global_variables_initializer())
      v1, v2 = self.evaluate([output1, output2])
      tf.logging.info('CausalConv output: %s', np.array_repr(v1))
      # pylint: disable=bad-whitespace,bad-continuation,line-too-long
      self.assertAllClose(v1, [
          [[[-0.01093466,  0.00369835],
            [ 0.03474921,  0.01418608],
            [ 0.01887876, -0.00763734]],
           [[-0.06922598, -0.04526342],
            [-0.02428233,  0.02042499],
            [-0.04504267, -0.01260209]],
           [[-0.14253227, -0.11353028],
            [-0.09067881,  0.03742362],
            [ 0.01281691,  0.00644186]],
           [[-0.06524619, -0.0555004 ],
            [-0.18850081, -0.05325979],
            [ 0.04960757,  0.05512709]]],
          [[[-0.01077277,  0.03013588],
            [ 0.00325067, -0.0223705 ],
            [-0.00895232,  0.03310337]],
           [[ 0.03113075, -0.02388876],
            [ 0.03238059,  0.00590346],
            [ 0.12839797, -0.02194144]],
           [[-0.09115655, -0.06798521],
            [-0.09801255, -0.01440183],
            [-0.04321899,  0.00340509]],
           [[-0.089603  , -0.07257183],
            [-0.04469771, -0.0389927 ],
            [-0.01747611,  0.00903451]]]
      ])  # pyformat: disable
      # pylint: enable=bad-whitespace,bad-continuation,line-too-long
      self.assertAllClose(v1[:, :2, :, :], v2[:, :2, :, :])
      with self.assertRaises(AssertionError):
        self.assertAllClose(v1[:, 2:, :, :], v2[:, 2:, :, :])

  def testCausalConv2DLayerFProp(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'causal_conv'
      params.filter_shape = [2, 2, 3, 2]
      params.filter_stride = [1, 1]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      params.causal_convolution = True
      params.activation = 'NONE'
      params.batch_norm = False

      conv_layer = layers.ConvLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3, 3]), dtype=tf.float32)

      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      self.evaluate(tf.global_variables_initializer())
      v1 = self.evaluate(output1)
      tf.logging.info('CausalConv output: %s', np.array_repr(v1))
      # pylint: disable=bad-whitespace,bad-continuation,line-too-long
      self.assertAllClose(v1, [
          [[[-0.065196  , -0.0597635 ],
            [ 0.02871699, -0.02915794],
            [-0.00529849, -0.02677475]],
           [[-0.0227601 ,  0.06118587],
            [ 0.25884673, -0.13917476],
            [ 0.03899311, -0.06894699]],
           [[-0.28780231, -0.12121122],
            [ 0.2447218 ,  0.09553684],
            [-0.07054863,  0.12110104]],
           [[ 0.17036264, -0.00258163],
            [ 0.28644818, -0.02746056],
            [ 0.06173857, -0.11599959]]],
          [[[ 0.1468567 ,  0.12725323],
            [-0.00131077, -0.03644447],
            [ 0.0266833 ,  0.01140832]],
           [[-0.23816   , -0.07873908],
            [-0.07348203,  0.25653225],
            [-0.21931274, -0.0569509 ]],
           [[-0.06972647, -0.03123237],
            [ 0.07432974, -0.03340006],
            [ 0.10474236,  0.00807726]],
           [[ 0.07581483,  0.25381109],
            [ 0.07091375, -0.14229891],
            [ 0.05247882, -0.08783717]]]
      ])  # pyformat: disable
      # pylint: enable=bad-whitespace,bad-continuation,line-too-long

  def testCausalConv2DEqualsConv2DWithPadding(self):
    # Causal conv is equivalent to regular conv with zero pre-padding.
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'causal_conv'
      params.filter_shape = [2, 2, 3, 2]
      params.filter_stride = [1, 1]
      params.params_init = py_utils.WeightInit.Gaussian(0.1, seed=12345)

      params.causal_convolution = True
      params.activation = 'NONE'
      params.batch_norm = False
      causal_conv_layer = layers.ConvLayer(params)

      normal_conv_params = params.Copy()
      normal_conv_params.name = 'conv'
      normal_conv_params.causal_convolution = False
      normal_conv_layer = layers.ConvLayer(normal_conv_params)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3, 3]), dtype=tf.float32)
      # Causal conv with kernel height (time) = 2 requires prepadding size 1.
      inputs1_pad = tf.concat([tf.zeros([2, 1, 3, 3]), inputs1], axis=1)

      output_causal, _ = causal_conv_layer.FPropDefaultTheta(inputs1)
      output_normal, _ = normal_conv_layer.FPropDefaultTheta(inputs1_pad)
      self.evaluate(tf.global_variables_initializer())
      v_causal, v_normal = self.evaluate([output_causal, output_normal])
      # Normal conv would produce an extra timestep due to SAME padding at the
      # end.
      self.assertAllClose(v_causal, v_normal[:, :-1])

  def testCausalConv2DEqualsConv2DWithKernelHeightOne(self):
    # When kernel height (time) = 1, causal convolution regresses to normal
    # convolution with SAME padding.
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'causal_conv'
      params.filter_shape = [1, 2, 3, 2]
      params.filter_stride = [1, 1]
      params.params_init = py_utils.WeightInit.Gaussian(0.1, seed=12345)

      params.causal_convolution = True
      params.activation = 'NONE'
      params.batch_norm = False
      causal_conv_layer = layers.ConvLayer(params)

      normal_conv_params = params.Copy()
      normal_conv_params.name = 'conv'
      normal_conv_params.causal_convolution = False
      normal_conv_layer = layers.ConvLayer(normal_conv_params)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3, 3]), dtype=tf.float32)

      output_causal, _ = causal_conv_layer.FPropDefaultTheta(inputs1)
      output_normal, _ = normal_conv_layer.FPropDefaultTheta(inputs1)
      self.evaluate(tf.global_variables_initializer())
      v_causal, v_normal = self.evaluate([output_causal, output_normal])

      self.assertAllClose(v_causal, v_normal)

  def testConvLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = layers.ConvLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)
      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      loss = tf.reduce_sum(output1)

      all_vars = tf.trainable_variables()
      self.assertEqual(3, len(all_vars))

      grads = tf.gradients(loss, all_vars)
      self.evaluate(tf.global_variables_initializer())
      sym_grads = [self.evaluate(sg) for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-02, atol=1e-02)

  def testConvLayerFPropTanh(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvLayer.Params()
      params.activation = 'TANH'
      params.name = 'conv'
      params.filter_shape = [3, 3, 3, 2]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = layers.ConvLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 3]), dtype=tf.float32)

      output1, _ = conv_layer.FPropDefaultTheta(inputs1, in_padding1)
      self.evaluate(tf.global_variables_initializer())

      # pyformat: disable
      # pylint: disable=bad-whitespace
      expected_output1 = [
          [[[ 0.35109526,  0.72346997],
            [ 0.0751792 , -0.84315312]],
           [[ 0.33594984, -0.18976833],
            [ 0.95773894, -0.28015777]]],
          [[[ 0.27572086, -0.26577294],
            [-0.38503852, -0.88501388]],
           [[-0.92332661,  0.69921255],
            [-0.75103623,  0.9219743 ]]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      actual = self.evaluate(output1)
      print(['actual = ', actual])
      self.assertAllClose(expected_output1, actual)

  def testConvSetLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvSetLayer.Params()
      params.name = 'conv_set'
      params.filter_shapes = [[3, 3, 3, 32], [8, 5, 3, 64]]
      params.cnn_tpl.filter_stride = [2, 2]
      params.cnn_tpl.params_init = py_utils.WeightInit.Gaussian(0.1)

      layers.ConvSetLayer(params)

  def _evalConvSetLayerFProp(self,
                             batch_norm=True,
                             bn_fold_weights=False,
                             weight_norm=False,
                             bias=False,
                             activation='RELU',
                             conv_last=False,
                             strides=(2, 2),
                             dilation_rate=(1, 1),
                             quantized=False,
                             dump_graphdef=False):
    self._ClearCachedSession()
    tf.reset_default_graph()
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ConvSetLayer.Params()
      params.name = 'conv_set'
      params.filter_shapes = [[2, 2, 6, 1], [3, 5, 6, 3]]
      params.cnn_tpl.filter_stride = strides
      params.cnn_tpl.dilation_rate = dilation_rate
      params.cnn_tpl.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.cnn_tpl.conv_last = conv_last
      params.cnn_tpl.batch_norm = batch_norm
      params.cnn_tpl.bn_fold_weights = bn_fold_weights
      params.cnn_tpl.weight_norm = weight_norm
      params.cnn_tpl.bias = bias
      params.cnn_tpl.activation = activation

      if quantized:
        params.qdomain.default = quant_utils.PassiveAsymQDomain.Params()

      conv_set_layer = layers.ConvSetLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 6]), dtype=tf.float32)

      output1, _ = conv_set_layer.FPropDefaultTheta(inputs1, in_padding1)
      self.evaluate(tf.global_variables_initializer())

      if dump_graphdef:
        print('ConvSet GraphDef:', sess.graph.as_graph_def())
        assert False, 'Disable "dump_graphdef" before submit'

      return self.evaluate(output1)

  def testConvSetLayerFProp(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace,bad-continuation
    expected_output1 = [
        [[[ 1.04307961,  0.        ,  1.27613628,  0.        ],
        [ 0.          ,  0.        ,  0.        ,  1.21081829 ]],
        [[ 0.         ,  0.18475296,  0.        ,  0.        ],
        [ 1.34087086  ,  2.2726357 ,  0.        ,  0.         ]]],
        [[[ 0.        ,  0.25231963,  0.        ,  0.       ],
        [ 1.13677704  ,  0.        ,  0.996117  ,  1.836285   ]],
        [[ 0.         ,  0.        ,  1.04101253,  0.        ],
        [ 0.12628449  ,  0.37599814,  0.3134549 ,  0.51208746 ]]]
    ]
    # pyformat: enable
    # pylint: enable=bad-whitespace,bad-continuation
    actual = self._evalConvSetLayerFProp()
    print(['actual = ', np.array_repr(actual)])
    self.assertAllClose(expected_output1, actual)

  def testConvSetLayerFPropQuantized(self):
    # pyformat: disable
    # pylint: disable=bad-whitespace,bad-continuation
    expected_output1 = [
        [[[ 1.04016984,  0.        ,  1.28103447,  0.        ],
          [ 0.        ,  0.        ,  0.        ,  1.20986581]],
         [[ 0.        ,  0.18681753,  0.        ,  0.        ],
          [ 1.35328221,  2.26849842,  0.        ,  0.        ]]],
        [[[ 0.        ,  0.24909003,  0.        ,  0.        ],
          [ 1.14100266,  0.        ,  0.98746401,  1.83259094]],
         [[ 0.        ,  0.        ,  1.04084051,  0.        ],
          [ 0.12736773,  0.38253111,  0.32025862,  0.5159722 ]]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace,bad-continuation
    actual = self._evalConvSetLayerFProp(bn_fold_weights=True, quantized=True)
    # Note that we don't have many ways to verify in a unit test that the
    # quant nodes were added properly; however, if their placement changes,
    # it will very likely perturb the golden values above. If digging deeper,
    # add 'dump_graphdef=True' to the above call and inspect the graphdef:
    # There should be one layer of fake_quant* nodes before the ConcatV2.
    print('actual = ', np.array_repr(actual))
    self.assertAllClose(expected_output1, actual)

  # TODO(yonghui): more test for convolution layer


class PoolingLayerTest(test_utils.TestCase, parameterized.TestCase):

  # TODO(lingvo): fix 'VALID' padding in pooling.
  @parameterized.named_parameters(
      {
          'testcase_name': 'max_pooling_same_padding',
          'pooling_type': 'MAX',
          'padding_algorithm': 'SAME',
          'window_shape': (3, 1),
          'window_stride': (1, 1),
          'inputs': np.array([-2, -10, -3, -4, 0, 0]),
          'input_paddings': np.array([0, 0, 0, 0, 1, 1]),
          'expected_output': np.array([-2, -2, -3, -3, 0, 0]),
          'expected_output_padding': np.array([0, 0, 0, 0, 1, 1]),
          'expected_output_without_padding': np.array([-2, -2, -3, 0, 0, 0]),
      }, {
          'testcase_name': 'avg_pooling_same_padding',
          'pooling_type': 'AVG',
          'padding_algorithm': 'SAME',
          'window_shape': (3, 1),
          'window_stride': (1, 1),
          'inputs': np.array([-2, 0, 2, 4, 0, 0]),
          'input_paddings': np.array([0, 0, 0, 0, 1, 1]),
          'expected_output': np.array([-1, 0, 2, 3, 0, 0]),
          'expected_output_padding': np.array([0, 0, 0, 0, 1, 1]),
          'expected_output_without_padding': np.array([-1, 0, 2, 2, 4 / 3, 0]),
      })
  def testSimpleCases(self, inputs, input_paddings, pooling_type, window_shape,
                      window_stride, padding_algorithm, expected_output,
                      expected_output_padding, expected_output_without_padding):
    inputs = inputs[np.newaxis, :, np.newaxis, np.newaxis]
    input_paddings = input_paddings[np.newaxis, :]
    param = layers.PoolingLayer.Params().Set(
        name='test_layer',
        pooling_type=pooling_type,
        window_shape=window_shape,
        window_stride=window_stride,
        padding_algorithm=padding_algorithm)
    pooling_layer = param.Instantiate()
    with self.session(use_gpu=True) as sess:
      inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
      input_paddings = tf.convert_to_tensor(input_paddings, dtype=tf.float32)
      output, output_paddings = pooling_layer.FPropDefaultTheta(
          inputs, input_paddings)
      output_without_padding = pooling_layer.FPropDefaultTheta(inputs)
      tf.global_variables_initializer().run()
      output_val, output_paddings_val, output_without_padding_val = sess.run(
          [output, output_paddings, output_without_padding])

    self.assertAllClose(expected_output, output_val.flatten())
    self.assertAllEqual(expected_output_padding, output_paddings_val.flatten())
    self.assertAllClose(expected_output_without_padding,
                        output_without_padding_val.flatten())

  def testPoolLayerFProp(self):
    with self.session(use_gpu=True):
      params = layers.PoolingLayer.Params()
      params.name = 'pool'
      params.window_shape = [3, 3]
      params.window_stride = [1, 2]

      pool_layer = layers.PoolingLayer(params)
      in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
      inputs1 = tf.constant(
          np.arange(96, dtype='float32').reshape([2, 4, 4, 3]),
          dtype=tf.float32)

      output1, _ = pool_layer.FPropDefaultTheta(inputs1, in_padding1)
      self.evaluate(tf.global_variables_initializer())
      print([np.array_repr(self.evaluate(output1))])
      expected_output1 = [[[[18., 19., 20.], [21., 22., 23.]],
                           [[30., 31., 32.], [33., 34., 35.]],
                           [[42., 43., 44.], [45., 46., 47.]],
                           [[42., 43., 44.], [45., 46., 47.]]],
                          [[[66., 67., 68.], [69., 70., 71.]],
                           [[78., 79., 80.], [81., 82., 83.]],
                           [[90., 91., 92.], [93., 94., 95.]],
                           [[90., 91., 92.], [93., 94., 95.]]]]
      self.assertAllClose(expected_output1, self.evaluate(output1))

  def test2DPoolLayerNoPaddingsFProp(self):
    with self.session(use_gpu=True):
      params = layers.PoolingLayer.Params()
      params.name = 'pool'
      params.window_shape = [3, 3]
      params.window_stride = [1, 2]

      pool_layer = layers.PoolingLayer(params)
      inputs = tf.constant(
          np.arange(96, dtype='float32').reshape([2, 4, 4, 3]),
          dtype=tf.float32)

      outputs = pool_layer.FPropDefaultTheta(inputs)
      self.evaluate(tf.global_variables_initializer())
      expected_outputs = [[[[18., 19., 20.], [21., 22., 23.]],
                           [[30., 31., 32.], [33., 34., 35.]],
                           [[42., 43., 44.], [45., 46., 47.]],
                           [[42., 43., 44.], [45., 46., 47.]]],
                          [[[66., 67., 68.], [69., 70., 71.]],
                           [[78., 79., 80.], [81., 82., 83.]],
                           [[90., 91., 92.], [93., 94., 95.]],
                           [[90., 91., 92.], [93., 94., 95.]]]]
      self.assertAllClose(expected_outputs, self.evaluate(outputs))

  def testPoolLayerMoreShapes(self):
    with self.session(use_gpu=True):
      for window_shape, window_stride in [
          [[3, 3], [1, 2]],
          [[2, 2], [1, 2]],
          [[3, 4], [1, 3]],
      ]:
        params = layers.PoolingLayer.Params()
        params.name = 'pool'
        params.window_shape = window_shape
        params.window_stride = window_stride

        pool_layer = layers.PoolingLayer(params)
        in_padding1 = tf.zeros([2, 4], dtype=tf.float32)
        inputs1 = tf.constant(
            np.arange(96, dtype='float32').reshape([2, 4, 4, 3]),
            dtype=tf.float32)

        output1, _ = pool_layer.FPropDefaultTheta(inputs1, in_padding1)

        output2 = tf.nn.max_pool(inputs1, [1] + params.window_shape + [1],
                                 [1] + params.window_stride + [1], 'SAME')

        predicted_out_shape = pool_layer.OutShape(inputs1.shape.as_list())

        self.evaluate(tf.global_variables_initializer())
        output1_v = self.evaluate(output1)
        self.assertAllClose(self.evaluate(output2), output1_v)
        self.assertAllClose(predicted_out_shape, output1_v.shape)


class BlurPoolLayerTest(test_utils.TestCase):

  def _testBlurPool(self,
                    subsample_type,
                    blur_filter,
                    expected_output,
                    has_paddings=True):
    with self.session(use_gpu=True):
      p = layers.BlurPoolLayer.Params().Set(
          name='blur_pool',
          input_channels=3,
          subsample_type=subsample_type,
          blur_filter=blur_filter)

      layer = p.Instantiate()
      inputs1 = tf.constant(
          np.arange(24, dtype='float32').reshape([2, 4, 1, 3]),
          dtype=tf.float32)

      if has_paddings:
        in_padding1 = tf.convert_to_tensor([[0, 0, 0, 1], [0, 0, 1, 1]],
                                           dtype=tf.float32)
        expected_out_padding = [[0, 1], [0, 1]]
        output1, out_padding1 = layer.FPropDefaultTheta(inputs1, in_padding1)
        self.evaluate(tf.global_variables_initializer())
        with self.subTest('test_output_values'):
          self.assertAllClose(expected_output, self.evaluate(output1))

        with self.subTest('test_output_paddings'):
          self.assertAllClose(expected_out_padding, self.evaluate(out_padding1))
      else:
        output1 = layer.FPropDefaultTheta(inputs1)
        self.evaluate(tf.global_variables_initializer())
        with self.subTest('test_output_values'):
          self.assertAllClose(expected_output, self.evaluate(output1))

  def testBlurPool1D(self):
    expected_output = np.array([[[[1.125, 1.8125, 2.5]], [[0, 0, 0]]],
                                [[[8.25, 8.875, 9.5]], [[0, 0, 0]]]],
                               dtype=np.float32)
    self._testBlurPool('1D', 'B5', expected_output)

  def testBlurPool2D(self):
    expected_output = np.array([[[[0.421875, 0.6796875, 0.9375]], [[0, 0, 0]]],
                                [[[3.09375, 3.328125, 3.5625]], [[0, 0, 0]]]],
                               dtype=np.float32)
    self._testBlurPool('2D', 'B5', expected_output)

  def testBlurPool1DNoPaddings(self):
    expected_output = np.array(
        [[[[1.125, 1.8125, 2.5]], [[5.25, 6.1875, 7.125]]],
         [[[9.375, 10.0625, 10.75]], [[16.5, 17.4375, 18.375]]]],
        dtype=np.float32)
    self._testBlurPool('1D', 'B5', expected_output, has_paddings=False)

  def testBlurPool2DNoPaddings(self):
    expected_output = np.array(
        [[[[0.421875, 0.6796875, 0.9375]], [[1.96875, 2.320312, 2.671875]]],
         [[[3.515625, 3.7734375, 4.03125]], [[6.1875, 6.5390625, 6.890625]]]],
        dtype=np.float32)
    self._testBlurPool('2D', 'B5', expected_output, has_paddings=False)


class ProjectionLayerTest(test_utils.TestCase, parameterized.TestCase):

  def testProjectionLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.input_dim = 2
      params.output_dim = 3
      params.batch_norm = True
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      layers.ProjectionLayer(params)
      proj_vars = tf.get_collection('ProjectionLayer_vars')
      proj_var_names = [x.name for x in proj_vars]
      self.assertEqual(['proj/w/var:0'], proj_var_names)
      bn_vars = tf.get_collection('BatchNormLayer_vars')
      bn_var_names = [x.name for x in bn_vars]
      expected_var_names = [
          'proj/beta/var:0', 'proj/gamma/var:0', 'proj/moving_mean/var:0',
          'proj/moving_variance/var:0'
      ]
      self.assertEqual(expected_var_names, bn_var_names)

  def _evalProjectionLayer(self,
                           reshape_to_2d=False,
                           batch_norm=True,
                           weight_norm=False,
                           activation='RELU',
                           affine_last=False,
                           input_dim=3,
                           output_dim=2,
                           quantized=False,
                           has_bias=False,
                           bn_fold_weights=None,
                           expect_bn_fold_weights=None,
                           is_eval=False,
                           layer_callback=None,
                           bn_decay=0.999,
                           bn_use_moving_avg_in_training=False,
                           use_einsum=True,
                           block_dim=0,
                           input_dtype=tf.float32,
                           fprop_dtype=None):
    self._ClearCachedSession()
    tf.reset_default_graph()
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.input_dim = input_dim
      params.output_dim = output_dim
      params.has_bias = has_bias
      if has_bias:
        params.bias_init = 5.0
      params.activation = activation
      params.batch_norm = batch_norm
      params.weight_norm = weight_norm
      params.affine_last = affine_last
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.bn_fold_weights = bn_fold_weights
      params.bn_params.decay = bn_decay
      params.bn_params.use_moving_avg_in_training = bn_use_moving_avg_in_training
      params.use_einsum = use_einsum
      params.block_dim = block_dim
      params.use_blocked_matmul = True if block_dim > 0 else False
      params.fprop_dtype = fprop_dtype

      if quantized:
        cc_schedule = quant_utils.FakeQuantizationSchedule.Params().Set(
            clip_end_step=1, quant_start_step=1)
        qdomain_default = quant_utils.SymmetricScheduledClipQDomain.Params(
        ).Set(cc_schedule=cc_schedule.Copy())
        params.qdomain.default = qdomain_default.Copy()

      in_padding = tf.zeros([2, 4, 1], dtype=input_dtype)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, input_dim]), dtype=input_dtype)
      if reshape_to_2d:
        in_padding = tf.reshape(in_padding, [-1, 1])
        inputs = tf.reshape(inputs, [-1, input_dim])

      proj_layer = layers.ProjectionLayer(params)
      if layer_callback:
        layer_callback(proj_layer)
      if expect_bn_fold_weights is not None:
        self.assertEqual(expect_bn_fold_weights, proj_layer._is_bn_folded)

      output = proj_layer.FPropDefaultTheta(inputs, in_padding)
      self.evaluate(tf.global_variables_initializer())
      if quantized:
        # Put it in the fully quantized range.
        self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 5))
      return self.evaluate(output)

  @parameterized.named_parameters(
      ('dtype_default', [], 1e-06),
      ('dtype_float16', [('.*w', tf.float16)], 1e-03),
  )
  def testProjectionLayerFProp(self, list_regex_dtype, atol):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[ 0.        ,  0.33779466],
         [ 0.4527415 ,  0.99911398],
         [ 0.44320837,  0.        ],
         [ 0.        ,  0.04557215]],
        [[ 0.69273949,  0.        ],
         [ 0.30908319,  0.        ],
         [ 0.        ,  0.        ],
         [ 0.        ,  1.54578114]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    with py_utils.VariableListDtypeRegexScope(list_regex_dtype):
      for reshape_to_2d in (False, True):
        actual = self._evalProjectionLayer(
            reshape_to_2d=reshape_to_2d, expect_bn_fold_weights=False)
        if reshape_to_2d:
          expected_output = np.reshape(np.array(expected_output), (-1, 2))
        tf.logging.info('expected = %s', expected_output)
        tf.logging.info('actual = %s', np.array_repr(actual))
        self.assertAllClose(expected_output, actual, atol=atol)

  def testProjectionLayerFPropWithBias(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[ 4.98987579,  5.03493643],
         [ 5.01192808,  5.0917592 ],
         [ 5.01156807,  4.99741936],
         [ 4.96849394,  5.00982761]],
        [[ 5.02098131,  4.98014927],
         [ 5.00650883,  4.87676954],
         [ 4.98995209,  4.91770315],
         [ 4.95948696,  5.138731  ]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # Tested without batch_norm because batch_norm will mostly cancel out the
    # affect of bias.
    actual = self._evalProjectionLayer(
        has_bias=True,
        batch_norm=False,
        expect_bn_fold_weights=False,
        activation='RELU6')
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testProjectionLayerExplicitFolding(self):
    unfolded = self._evalProjectionLayer(
        bn_fold_weights=False, expect_bn_fold_weights=False)
    folded = self._evalProjectionLayer(
        bn_fold_weights=True, expect_bn_fold_weights=True)
    tf.logging.info('unfolded = %s', np.array_repr(unfolded))
    tf.logging.info('folded = %s', np.array_repr(folded))
    self.assertAllClose(folded, unfolded)

  def testProjectionLayerExplicitFoldingEval(self):
    unfolded = self._evalProjectionLayer(
        bn_fold_weights=False, expect_bn_fold_weights=False, is_eval=True)
    folded = self._evalProjectionLayer(
        bn_fold_weights=True, expect_bn_fold_weights=True, is_eval=True)
    tf.logging.info('unfolded = %s', np.array_repr(unfolded))
    tf.logging.info('folded = %s', np.array_repr(folded))
    self.assertAllClose(folded, unfolded)

  def testProjectionLayerExplicitFoldingNoBatchNorm(self):
    unfolded = self._evalProjectionLayer(
        batch_norm=False, bn_fold_weights=False, expect_bn_fold_weights=False)
    # Note that weight folding will report as disabled because batch norm is
    # disabled.
    folded = self._evalProjectionLayer(
        batch_norm=False, bn_fold_weights=True, expect_bn_fold_weights=False)
    tf.logging.info('unfolded = %s', np.array_repr(unfolded))
    tf.logging.info('folded = %s', np.array_repr(folded))
    self.assertAllClose(folded, unfolded)

  def testProjectionLayerExplicitFoldingWithWeightNorm(self):
    unfolded = self._evalProjectionLayer(
        weight_norm=True, bn_fold_weights=False, expect_bn_fold_weights=False)
    folded = self._evalProjectionLayer(
        weight_norm=True, bn_fold_weights=True, expect_bn_fold_weights=True)
    tf.logging.info('unfolded = %s', np.array_repr(unfolded))
    tf.logging.info('folded = %s', np.array_repr(folded))
    self.assertAllClose(folded, unfolded)

  def testProjectionLayerWeightNorm(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[ 0.        ,  0.36285588],
         [ 0.82909501,  1.07323885],
         [ 0.81163716,  0.        ],
         [ 0.        ,  0.04895319]],
        [[ 1.26859784,  0.        ],
         [ 0.56601691,  0.        ],
         [ 0.        ,  0.        ],
         [ 0.        ,  1.66046333]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    for reshape_to_2d in (False, True):
      actual = self._evalProjectionLayer(
          reshape_to_2d=reshape_to_2d, weight_norm=True)
      if reshape_to_2d:
        expected_output = np.reshape(np.array(expected_output), (-1, 2))
      tf.logging.info('expected = %s', expected_output)
      tf.logging.info('actual = %s', np.array_repr(actual))
      self.assertAllClose(expected_output, actual)

  def testProjectionLayerAffineFirstVsLastFProp(self):
    """Compare results of affine first vs. last."""
    # ... with batch_norm and activation disabled.
    self.assertAllClose(
        self._evalProjectionLayer(
            batch_norm=False, activation='NONE', affine_last=False),
        self._evalProjectionLayer(
            batch_norm=False, activation='NONE', affine_last=True))

  def testProjectionLayerAffineLastFProp(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output1 = [
        [[ 0.        ,  0.        ],
         [ 0.03410175,  0.04741348],
         [ 0.02665393, -0.02072855],
         [-0.01116518, -0.06280501]],
        [[ 0.04615254, -0.03589247],
         [-0.00376316, -0.0464084 ],
         [-0.01111402, -0.13706152],
         [-0.02596203,  0.16340451]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    actual = self._evalProjectionLayer(affine_last=True)
    print(['actual = ', np.array_repr(actual)])
    self.assertAllClose(expected_output1, actual)

  def testProjectionLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.dtype = tf.float64
      params.input_dim = 3
      params.output_dim = 2
      params.batch_norm = True
      params.params_init = py_utils.WeightInit.Gaussian(0.01)

      proj_layer = layers.ProjectionLayer(params)
      in_padding1 = tf.zeros([2, 4, 1], dtype=tf.float64)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float64)
      output1 = proj_layer.FPropDefaultTheta(inputs1, in_padding1)
      loss = tf.reduce_sum(output1)

      all_vars = tf.trainable_variables()
      self.assertLen(all_vars, 3)

      grads = tf.gradients(loss, all_vars)
      self.evaluate(tf.global_variables_initializer())
      sym_grads = [self.evaluate(sg) for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v, 1e-6)
          for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-06, atol=1e-06)

  def testProjectionLayerBackPropDTypeFloat16(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.dtype = tf.float64
      params.input_dim = 3
      params.output_dim = 2
      params.batch_norm = False
      params.params_init = py_utils.WeightInit.Constant(0.00443641)

      with py_utils.VariableListDtypeRegexScope([('.*w', tf.float16)]):
        proj_layer = layers.ProjectionLayer(params)
      in_padding1 = tf.zeros([2, 4, 1], dtype=tf.float64)
      inputs1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float64)
      output1 = proj_layer.FPropDefaultTheta(inputs1, in_padding1)
      loss = tf.reduce_sum(output1)

      all_vars = tf.trainable_variables()
      self.assertLen(all_vars, 1)

      grads = tf.gradients(loss, all_vars)
      self.evaluate(tf.global_variables_initializer())
      sym_grads = [self.evaluate(sg) for sg in grads]

      expected_grads = [[2.1999533, 2.1999533], [4.08647324, 4.08647324],
                        [0.76935822, 0.76935822]]

      self.assertAllClose(sym_grads[0], expected_grads, rtol=1e-03, atol=1e-03)

  def testProjectionLayerFPropQuantizedWithUnfusedActivation(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[-0.1328125,  0.3125   ],
         [ 0.421875 ,  0.734375 ],
         [ 0.421875 , -0.109375 ],
         [-0.6015625,  0.0078125]],
        [[ 0.6015625, -0.3046875],
         [ 0.3046875, -0.7578125],
         [-0.125    , -0.7578125],
         [-0.734375 ,  0.7578125]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    def CheckLayer(proj_layer):
      # Should not error because this qtensor is defined.
      proj_layer.QTensor('activation', tf.convert_to_tensor(0.))
      # The intermediate tensor should be defined.
      proj_layer.QTensor('affine_matmul', tf.convert_to_tensor(0.))

    # When quantization enabled, batchnorm folding should auto enable.
    # TANH is unfused.
    actual = self._evalProjectionLayer(
        activation='TANH',
        quantized=True,
        expect_bn_fold_weights=True,
        layer_callback=CheckLayer)
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testProjectionLayerFPropQuantizedWithFusedActivation(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[ 0.       ,  0.3203125],
         [ 0.453125 ,  0.9375   ],
         [ 0.4453125,  0.       ],
         [ 0.       ,  0.0078125]],
        [[ 0.6953125,  0.       ],
         [ 0.3125   ,  0.       ],
         [ 0.       ,  0.       ],
         [ 0.       ,  0.9921875]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    def CheckLayer(proj_layer):
      # Should not error because this qtensor is defined.
      proj_layer.QTensor('activation', tf.convert_to_tensor(0.))
      with self.assertRaises(AssertionError):
        # The intermediate tensor should *not* be quantized.
        proj_layer.QTensor('affine_matmul', tf.convert_to_tensor(0.))

    # When quantization enabled, batchnorm folding should auto enable.
    # RELU6 is fused.
    actual = self._evalProjectionLayer(
        activation='RELU6',
        quantized=True,
        expect_bn_fold_weights=True,
        layer_callback=CheckLayer)
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testProjectionLayerFPropQuantizedOnlyMatmul(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [
        [[-0.0078125,  0.0390625],
         [ 0.0078125,  0.09375  ],
         [ 0.0078125,  0.       ],
         [-0.03125  ,  0.015625 ]],
        [[ 0.015625 , -0.015625 ],
         [ 0.0078125, -0.125    ],
         [-0.0078125, -0.078125 ],
         [-0.0390625,  0.1484375]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    def CheckLayer(proj_layer):
      # Should not error because this qtensor is defined.
      proj_layer.QTensor('affine_matmul', tf.convert_to_tensor(0.))

    actual = self._evalProjectionLayer(
        activation='NONE',
        quantized=True,
        batch_norm=False,
        expect_bn_fold_weights=False,
        layer_callback=CheckLayer)
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testProjectionLayerFPropQuantizedOnlyMatmulBias(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    # Saturated because of the out of range bias.
    expected_output = [[[0.9921875, 0.9921875], [0.9921875, 0.9921875],
                        [0.9921875, 0.9921875], [0.9921875, 0.9921875]],
                       [[0.9921875, 0.9921875], [0.9921875, 0.9921875],
                        [0.9921875, 0.9921875], [0.9921875, 0.9921875]]]

    # pyformat: enable
    # pylint: enable=bad-whitespace
    def CheckLayer(proj_layer):
      # Should not error because this qtensor is defined.
      proj_layer.QTensor('affine_matmul', tf.convert_to_tensor(0.))

    actual = self._evalProjectionLayer(
        activation='NONE',
        quantized=True,
        has_bias=True,
        batch_norm=False,
        expect_bn_fold_weights=False,
        layer_callback=CheckLayer)
    tf.logging.info('expected = %s', expected_output)
    tf.logging.info('actual = %s', np.array_repr(actual))
    self.assertAllClose(expected_output, actual)

  def testFCLayerConstruction(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.FCLayer.Params()
      params.name = 'fc'
      params.input_dim = 2
      params.output_dim = 3
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      layers.FCLayer(params)
      proj_vars = tf.get_collection('FCLayer_vars')
      proj_var_names = [x.name for x in proj_vars]
      expected_var_names = ['fc/w/var:0', 'fc/b/var:0']
      self.assertEqual(expected_var_names, proj_var_names)

  def testFCLayerFProp(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.FCLayer.Params()
      params.name = 'fc'
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      proj_layer = layers.FCLayer(params)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float32)

      output = proj_layer.FPropDefaultTheta(inputs)
      self.evaluate(tf.global_variables_initializer())

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ 0.        ,  0.04883499],
           [ 0.17094055,  0.        ],
           [ 0.09287541,  0.        ],
           [ 0.        ,  0.19471419]],
          [[ 0.15290432,  0.        ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.10548697],
           [ 0.        ,  0.22610095]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      actual = self.evaluate(output)
      print(['actual = ', np.array_repr(actual)])
      self.assertAllClose(expected_output, actual)

  @parameterized.named_parameters(
      ('F32FPropF32Input', tf.float32, tf.float32, 0.668211),
      ('F32FPropBF16Input', tf.float32, tf.bfloat16, 0.669565),
      ('BF16FPropF32Input', tf.bfloat16, tf.float32, 0.667969),
      ('BF16FPropBF16Input', tf.bfloat16, tf.bfloat16, 0.667969),
  )
  def testFCLayerDtypes(self, fprop_dtype, input_dtype, expected_sum=0.):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.FCLayer.Params()
      params.name = 'fc'
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.fprop_dtype = fprop_dtype
      params.random_seed = 123

      proj_layer = layers.FCLayer(params)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=input_dtype)

      output = proj_layer.FPropDefaultTheta(inputs)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(expected_sum, self.evaluate(tf.reduce_sum(output)))

  def testFCLayerBackProp(self):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.FCLayer.Params()
      params.name = 'fc'
      params.dtype = tf.float64
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.01)

      proj_layer = layers.FCLayer(params)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float64)
      output = proj_layer.FPropDefaultTheta(inputs)
      loss = tf.reduce_sum(output)

      all_vars = tf.trainable_variables()
      self.assertLen(all_vars, 2)

      grads = tf.gradients(loss, all_vars)
      self.evaluate(tf.global_variables_initializer())
      sym_grads = [self.evaluate(sg) for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v, 1e-6)
          for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-06, atol=1e-06)

  def testProjectionLayerFPropUsingMovingAvgInTraining(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [[[0.        , 0.03491905],
                        [0.01192194, 0.09171353],
                        [0.01156251, 0.        ],
                        [0.        , 0.00982281]],
                       [[0.02097072, 0.        ],
                        [0.00650552, 0.        ],
                        [0.        , 0.        ],
                        [0.        , 0.13866161]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    for reshape_to_2d in (False, True):
      actual = self._evalProjectionLayer(
          reshape_to_2d=reshape_to_2d,
          expect_bn_fold_weights=False,
          bn_use_moving_avg_in_training=True)
      if reshape_to_2d:
        expected_output = np.reshape(np.array(expected_output), (-1, 2))
      tf.logging.info('expected = %s', expected_output)
      tf.logging.info('actual = %s', np.array_repr(actual))
      self.assertAllClose(expected_output, actual)

  def testProjectionLayerFPropEinsum(self):
    output_with_einsum = self._evalProjectionLayer(use_einsum=True)
    output_without_einsum = self._evalProjectionLayer(use_einsum=False)
    self.assertAllClose(output_with_einsum, output_without_einsum)

  def testProjectionLayerFPropBlockMatmul(self):
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output = [[[0.,         0.,         0.,         0.],
                        [0.,         0.03153732, 0.01891183, 0.02154643],
                        [0.01373154, 0.09479743, 0.,         0.03639744],
                        [0.05870617, 0.19162716, 0.05567084, 0.15362406]],

                       [[0.06591958, 0.2321615,  0.,         0.14735883],
                        [0.1003774,  0.33637568, 0.,         0.22276597],
                        [0.02362791, 0.10728893, 0.01922233, 0.07299631],
                        [0.,         0.,         0.,         0.        ]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    output_with_block_matmul = self._evalProjectionLayer(
        input_dim=4,
        output_dim=4,
        batch_norm=False,
        use_einsum=False,
        block_dim=2)
    tf.logging.info(output_with_block_matmul)
    self.assertAllClose(output_with_block_matmul, expected_output)
    # pylint: disable=bad-whitespace
    # pyformat: disable
    expected_output =  [[[0.,    0.,      0.,         0.,         0.],
                         [0.,    0.,      0.06118044, 0.,         0.02968791],
                         [0.,    0.,      0.09687695, 0.,         0.],
                         [0.10228965, 0., 0.01826946, 0.0219113,  0.16076824]],

                        [[0.03506518, 0.27432495, 0.25932777, 0.,         0.],
                         [0.,         0.00882578, 0.06655132, 0.,         0.],
                         [0.,         0.,         0.,         0.10196716, 0.],
                         [0.,         0.,         0.08253407, 0.,         0.]]]
    # pyformat: enable
    # pylint: enable=bad-whitespace
    # Test case for odd input and output dimensions.
    output_with_block_matmul = self._evalProjectionLayer(
        input_dim=5,
        output_dim=5,
        batch_norm=False,
        use_einsum=False,
        block_dim=2)
    tf.logging.info(output_with_block_matmul)
    self.assertAllClose(output_with_block_matmul, expected_output)

  @parameterized.named_parameters(
      {
          'testcase_name': 'RELU',
          'activation': 'RELU',
          'input_dims': [2, 4, 3],
          'expected_extra_flops': 0,
          'expected_per_fn_flops': 1
      },
      {
          'testcase_name': 'SIGMOID',
          'activation': 'SIGMOID',
          'input_dims': [2048, 10],
          'expected_extra_flops': 0,
          'expected_per_fn_flops': 4
      },
      {
          'testcase_name': 'BatchNorm',
          'activation': 'RELU',
          'input_dims': [2048, 10],
          'output_dim': 8,
          'expected_extra_flops': 2048 * 8 * 10,  # 10 flops per element.
          'expected_per_fn_flops': 1,
          'batch_norm': True
      },
      {
          'testcase_name': 'WeightNorm',
          'activation': 'RELU',
          'input_dims': [2048, 10],
          'output_dim': 100,
          'expected_extra_flops': 2 * 10 + 2 * 10 * 100 + 2,
          'expected_per_fn_flops': 1,
          'weight_norm': True
      },
      {
          'testcase_name': 'Bias',
          'activation': 'RELU',
          'input_dims': [2048, 10],
          'output_dim': 100,
          'expected_extra_flops': 100,
          'expected_per_fn_flops': 1,
          'has_bias': True
      })
  # Extra flops are for bias, batch norm, weight norm, etc.
  def testProjectionLayerMeta(self,
                              input_dims,
                              expected_per_fn_flops,
                              expected_extra_flops,
                              activation='RELU',
                              batch_norm=False,
                              weight_norm=False,
                              has_bias=False,
                              output_dim=2):
    with self.session(use_gpu=True):
      params = layers.ProjectionLayer.Params()
      params.name = 'fc'
      params.input_dim = input_dims[-1]
      params.output_dim = output_dim
      params.params_init = py_utils.WeightInit.Gaussian(0.1)
      params.activation = activation
      params.batch_norm = batch_norm
      params.weight_norm = weight_norm
      params.has_bias = has_bias

      meta = params.cls.FPropMeta(params, tshape.Shape(input_dims))
      self.assertEqual(
          meta.flops,
          expected_extra_flops + (np.prod(input_dims[:-1]) *
                                  (2 * params.input_dim * params.output_dim +
                                   params.output_dim * expected_per_fn_flops)))
      self.assertEqual(meta.out_shapes[0].ToTensorShape().as_list(),
                       input_dims[:-1] + [params.output_dim])

  def testProjectionLayerFPropFullSequence(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.ProjectionLayer.Params()
      params.name = 'proj'
      params.batch_norm = False
      params.has_bias = True
      params.input_dim = 3
      params.output_dim = 2
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      proj_layer = layers.ProjectionLayer(params)
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float32)
      padding = tf.zeros([2, 4, 1], dtype=tf.float32)

      output = proj_layer.FPropFullSequence(proj_layer.theta, inputs, padding)
      self.evaluate(tf.global_variables_initializer())

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ 0.        ,  0.0349365 ],
           [ 0.0119279 ,  0.09175937],
           [ 0.01156829,  0.        ],
           [ 0.        ,  0.00982772]],
          [[ 0.0209812 ,  0.        ],
           [ 0.00650877,  0.        ],
           [ 0.        ,  0.],
           [ 0.        ,  0.13873091]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      actual = self.evaluate(output)
      print(['actual = ', np.array_repr(actual)])
      self.assertAllClose(expected_output, actual)


class StackingOverTimeLayerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'pad_with_left_frame',
          'pad_with_left_frame': True
      },
      {
          'testcase_name': 'pad_with_zeros',
          'pad_with_left_frame': False
      },
  )
  def testStackingOverTimeFProp(self, pad_with_left_frame):
    with self.session(use_gpu=True):
      params = layers.StackingOverTime.Params()
      params.name = 'stackingOverTime'
      params.left_context = 2
      params.right_context = 0
      params.stride = 2
      params.pad_with_left_frame = pad_with_left_frame

      stacker = layers.StackingOverTime(params)
      self.assertEqual(stacker.window_size, 3)

      inputs = tf.constant([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                            [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0], [0, 0]]],
                           dtype=tf.float32)
      paddings = tf.constant(
          [[[0], [0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1], [1]]],
          dtype=tf.float32)

      outputs, output_paddings = stacker.FProp(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      print([np.array_repr(self.evaluate(outputs))])
      if pad_with_left_frame:
        expected_outputs = [
            [[1, 1, 1, 1, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
            [[7, 7, 7, 7, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
        ]
      else:
        expected_outputs = [
            [[0, 0, 0, 0, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
            [[0, 0, 0, 0, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
        ]

      self.assertAllClose(expected_outputs, self.evaluate(outputs))

      expected_output_paddings = [[[0], [0], [0]], [[0], [0], [1]]]
      self.assertAllClose(expected_output_paddings,
                          self.evaluate(output_paddings))

  @parameterized.named_parameters(
      {
          'testcase_name': 'pad_with_right_frame',
          'pad_with_right_frame': True
      },
      {
          'testcase_name': 'pad_with_zeros',
          'pad_with_right_frame': False
      },
  )
  def testStackingOverTimePadWithRightFrameFProp(self, pad_with_right_frame):
    with self.session(use_gpu=True):
      params = layers.StackingOverTime.Params()
      params.name = 'stackingOverTime'
      params.left_context = 0
      params.right_context = 1
      params.stride = 2
      params.pad_with_right_frame = pad_with_right_frame

      stacker = layers.StackingOverTime(params)
      self.assertEqual(stacker.window_size, 2)

      # input shape [2, 5, 2]
      inputs = tf.constant([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                            [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0]]],
                           dtype=tf.float32)
      paddings = tf.constant(
          [[[0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1]]],
          dtype=tf.float32)

      outputs, output_paddings = stacker.FProp(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      print([np.array_repr(self.evaluate(outputs))])
      if pad_with_right_frame:
        # output shape [2, 3, 4]
        # [5, 5] is duplication of the last input frame.
        expected_outputs = [
            [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 5, 5]],
            [[7, 7, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
      else:
        expected_outputs = [
            [[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 0, 0]],
            [[7, 7, 8, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]

      self.assertAllClose(expected_outputs, self.evaluate(outputs))

      expected_output_paddings = [[[0], [0], [0]], [[0], [1], [1]]]
      self.assertAllClose(expected_output_paddings,
                          self.evaluate(output_paddings))

  def testStackingOverTimeFPropReduceMaxPadding(self):
    with self.session(use_gpu=True):
      params = layers.StackingOverTime.Params()
      params.name = 'stackingOverTime'
      params.left_context = 2
      params.right_context = 0
      params.stride = 2
      params.padding_reduce_option = 'reduce_max'

      stacker = layers.StackingOverTime(params)
      self.assertEqual(stacker.window_size, 3)

      inputs = tf.constant([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
                            [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0], [0, 0]]],
                           dtype=tf.float32)
      paddings = tf.constant(
          [[[0], [0], [0], [0], [0], [0]], [[0], [0], [1], [1], [1], [1]]],
          dtype=tf.float32)

      outputs, output_paddings = stacker.FProp(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      print([np.array_repr(self.evaluate(outputs))])
      expected_outputs = [
          [[0, 0, 0, 0, 1, 1], [1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]],
          [[0, 0, 0, 0, 7, 7], [7, 7, 8, 8, 0, 0], [0, 0, 0, 0, 0, 0]],
      ]

      self.assertAllClose(expected_outputs, self.evaluate(outputs))

      expected_output_paddings = [[[1], [0], [0]], [[1], [1], [1]]]
      self.assertAllClose(expected_output_paddings,
                          self.evaluate(output_paddings))

  def testStackingOverTimeFProp2(self):
    with self.session(use_gpu=True):
      params = layers.StackingOverTime.Params()
      params.name = 'stackingOverTime'
      params.left_context = 0
      params.right_context = 1
      params.stride = 2

      stacker = layers.StackingOverTime(params)
      self.assertEqual(stacker.window_size, 2)

      inputs = tf.random.normal([2, 21, 16], seed=78123)
      paddings = 1.0 - tf.sequence_mask([9, 14], 21, tf.float32)
      paddings = tf.expand_dims(paddings, -1)

      outputs, output_paddings = stacker.FProp(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())

      inputs_v, outputs_v, paddings_v = self.evaluate(
          [inputs, outputs, output_paddings])

      # length
      self.assertAllEqual([5, 7], np.sum(1.0 - paddings_v, (1, 2)))
      # input and output sums are equal
      self.assertAllClose(np.sum(inputs_v, (1, 2)), np.sum(outputs_v, (1, 2)))

  def testStackingOverTimeIdentityFProp(self):
    with self.session(use_gpu=True):
      params = layers.StackingOverTime.Params()
      params.name = 'stackingOverTime'
      params.left_context = 0
      params.right_context = 0
      params.stride = 1

      stacker = layers.StackingOverTime(params)
      self.assertEqual(stacker.window_size, 1)
      inputs = tf.constant([[[1], [2], [3], [4], [5]]], dtype=tf.float32)
      paddings = tf.zeros([1, 5, 1], dtype=tf.float32)

      outputs, output_paddings = stacker.FProp(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      print([np.array_repr(self.evaluate(outputs))])
      expected_outputs = [[[1], [2], [3], [4], [5]]]
      self.assertAllClose(expected_outputs, self.evaluate(outputs))
      expected_output_paddings = [[[0], [0], [0], [0], [0]]]
      self.assertAllClose(expected_output_paddings,
                          self.evaluate(output_paddings))

  def _testUnstack(self, inputs, **kwargs):
    params = layers.StackingOverTime.Params().Set(
        name='stackingOverTime', **kwargs)
    with self.session(use_gpu=True):
      stacker = params.Instantiate()
      stacked, _ = stacker.FProp(inputs)
      unstacked = stacker.Unstack(stacked)
      inputs, stacked, unstacked = self.evaluate([inputs, stacked, unstacked])

      batch, input_length, depth = inputs.shape
      stacked_length = stacked.shape[1]
      stride = stacker.params.stride
      right_context = stacker.params.right_context

      self.assertAllEqual(
          unstacked.shape,
          [batch, (stacked_length - 1) * stride + right_context + 1, depth])
      if right_context + 1 >= stride:
        self.assertGreaterEqual(unstacked.shape[1], input_length)
        self.assertAllClose(inputs, unstacked[:, :input_length])
      else:
        self.assertLessEqual(unstacked.shape[1], input_length)
        # The final up to stride - right_context - 1 values are missing.
        self.assertLessEqual(input_length - unstacked.shape[1],
                             stride - right_context - 1)
        self.assertAllClose(inputs[:, :unstacked.shape[1]], unstacked)

  def testStackingOverTimeUnstack(self):
    batch_size = 2
    length = 7
    depth = 3
    inputs = tf.reshape(
        tf.cast(tf.range(batch_size * length * depth), tf.float32),
        [batch_size, length, depth])
    self._testUnstack(inputs, left_context=2, stride=1)
    with self.assertRaises(ValueError):
      self._testUnstack(inputs, stride=2)
    self._testUnstack(inputs, stride=2, right_context=3)
    self._testUnstack(inputs, left_context=2, stride=3)
    self._testUnstack(inputs, stride=4, right_context=3)
    self._testUnstack(inputs, stride=4, left_context=1, right_context=2)


class SingleShardEmbeddingLayerTest(test_utils.TestCase):

  def testSingleShardEmbeddingLayer(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.SingleShardEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = params.Instantiate()
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 0.126485, self.evaluate(embs_sum))  # pylint: disable=line-too-long

  def testCheckedIds(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.SingleShardEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 16
      params.embedding_dim = 128
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = params.Instantiate()

      neg_ids = tf.constant([[-1]])
      oov_ids = tf.constant([[params.vocab_size]])
      self.evaluate(tf.global_variables_initializer())

      with self.assertRaises(tf.errors.InvalidArgumentError):
        neg_embs = emb_layer.EmbLookupDefaultTheta(neg_ids)
        self.evaluate(neg_embs)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        oov_embs = emb_layer.EmbLookupDefaultTheta(oov_ids)
        self.evaluate(oov_embs)

  def testEmbeddingLayerScaling(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.SingleShardEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.scale_sqrt_depth = True
      emb_layer = params.Instantiate()
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(0.126485 * params.embedding_dim**0.5,
                          self.evaluate(embs_sum))

  def testEmbeddingLayerWithVN(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.SimpleEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.params_init = py_utils.WeightInit.Gaussian(0.01, seed=398847392)
      params.vn.global_vn = True
      params.vn.per_step_vn = False
      params.vn.scale = 0.5
      params.vn.seed = 398847392
      params.random_seed = 12345
      emb_layer = params.Instantiate()
      self.assertEqual(len(emb_layer.vars.Flatten()), 1)
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 1.886353, self.evaluate(embs_sum))  # pylint: disable=line-too-long


class EmbeddingLayerTest(test_utils.TestCase):

  def testEmbeddingLayer(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.EmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.max_num_shards = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = layers.EmbeddingLayer(params)
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 0.234941, self.evaluate(embs_sum))  # pylint: disable=line-too-long

  def testCheckedIds(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.EmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 16
      params.embedding_dim = 128
      params.max_num_shards = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = layers.EmbeddingLayer(params)

      neg_ids = tf.constant([[-1]])
      oov_ids = tf.constant([[params.vocab_size]])
      self.evaluate(tf.global_variables_initializer())

      with self.assertRaises(tf.errors.InvalidArgumentError):
        neg_embs = emb_layer.EmbLookupDefaultTheta(neg_ids)
        self.evaluate(neg_embs)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        oov_embs = emb_layer.EmbLookupDefaultTheta(oov_ids)
        self.evaluate(oov_embs)

  def testEmbeddingLayerScaling(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.EmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.max_num_shards = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.scale_sqrt_depth = True
      emb_layer = layers.EmbeddingLayer(params)
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(0.23494134843349457 * params.embedding_dim**0.5,
                          self.evaluate(embs_sum))

  def testEmbeddingLayerWithVN(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.EmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 80000
      params.embedding_dim = 128
      params.max_num_shards = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.01, seed=398847392)
      params.vn.global_vn = True
      params.vn.per_step_vn = False
      params.vn.scale = 0.5
      params.vn.seed = 398847392
      emb_layer = layers.EmbeddingLayer(params)
      self.assertEqual(len(emb_layer.vars.Flatten()), 4)
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 0.466322, self.evaluate(embs_sum), atol=1e-5)  # pylint: disable=line-too-long

  def _testSimpleEmbeddingLayer(self,
                                use_matmul,
                                use_3d_weight_tensor,
                                fprop_mode,
                                scale_sqrt_depth=False):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(398847392)
      params = layers.SimpleEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 8000
      params.embedding_dim = 128
      params.use_matmul = use_matmul
      params.fprop_mode = fprop_mode
      params.use_3d_weight_tensor = use_3d_weight_tensor
      params.scale_sqrt_depth = scale_sqrt_depth
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False

      emb_layer = layers.SimpleEmbeddingLayer(params)
      expected_fprop_mode = fprop_mode
      if expected_fprop_mode is None:
        expected_fprop_mode = 'matmul' if use_matmul else 'gather'
      self.assertEqual(emb_layer._fprop_mode, expected_fprop_mode)

      emb_matrix = emb_layer.vars.wm
      ids = tf.constant([[89], [100]])
      outputs = emb_layer.EmbLookupDefaultTheta(ids)
      fast_outputs = emb_layer.EmbLookupDefaultThetaOnCpu(ids)

    with self.session(use_gpu=True, graph=g):
      self.evaluate(tf.global_variables_initializer())
      emb_matrix_val, ids_val, outputs_val, fast_outputs_val = self.evaluate(
          [emb_matrix, ids, outputs, fast_outputs])
      if scale_sqrt_depth:
        emb_matrix_val *= params.embedding_dim**0.5

      self.assertEqual(emb_matrix_val.shape, (8000, 128))
      self.assertEqual(ids_val.shape, (2, 1))

      self.assertEqual(outputs_val.shape, (2, 1, 128))
      self.assertAllClose(emb_matrix_val[89, :], outputs_val[0, 0, :])
      self.assertAllClose(emb_matrix_val[100, :], outputs_val[1, 0, :])

      self.assertEqual(fast_outputs_val.shape, (2, 1, 128))
      self.assertAllClose(emb_matrix_val[89, :], fast_outputs_val[0, 0, :])
      self.assertAllClose(emb_matrix_val[100, :], fast_outputs_val[1, 0, :])

  def testSimpleEmbeddingLayerForLoop(self):
    self._testSimpleEmbeddingLayer(False, True, None)

  def testSimpleEmbeddingLayerForLoop2D(self):
    self._testSimpleEmbeddingLayer(False, False, None)

  def testSimpleEmbeddingLayerMatmul(self):
    self._testSimpleEmbeddingLayer(True, False, None)

  def testSimpleEmbeddingLayerGather(self):
    self._testSimpleEmbeddingLayer(False, False, 'gather')

  def testSimpleEmbeddingLayerScaling(self):
    self._testSimpleEmbeddingLayer(True, False, None, True)

  def _testKmeansSimpleEmbeddingLayer(self,
                                      use_matmul,
                                      use_3d_weight_tensor,
                                      fprop_mode,
                                      scale_sqrt_depth=False):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(398847392)
      params = layers.SimpleEmbeddingLayer.Params()
      params.name = 'kmeans_emb'
      params.dtype = tf.float32
      params.vocab_size = 8000
      params.embedding_dim = 128
      params.use_matmul = use_matmul
      params.fprop_mode = fprop_mode
      params.use_3d_weight_tensor = use_3d_weight_tensor
      params.scale_sqrt_depth = scale_sqrt_depth
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.pruning_hparams_dict = {
          'name': 'embedding_pruning',
          'prune_option': 'weight',
          'begin_pruning_step': 1,
          'end_pruning_step': 200,
          'sparsity_function_begin_step': 1,
          'sparsity_function_end_step': 200,
          'pruning_frequency': 10,
      }

      emb_layer = layers.SimpleEmbeddingLayer(params)
      expected_fprop_mode = fprop_mode
      if expected_fprop_mode is None:
        expected_fprop_mode = 'matmul' if use_matmul else 'gather'
      self.assertEqual(emb_layer._fprop_mode, expected_fprop_mode)

      emb_matrix = emb_layer.vars.wm
      ids = tf.constant([[89], [100]])
      outputs = emb_layer.EmbLookupDefaultTheta(ids)
      fast_outputs = emb_layer.EmbLookupDefaultThetaOnCpu(ids)

    with self.session(use_gpu=True, graph=g):
      self.evaluate(tf.global_variables_initializer())
      emb_matrix_val, ids_val, outputs_val, fast_outputs_val = self.evaluate(
          [emb_matrix, ids, outputs, fast_outputs])
      if scale_sqrt_depth:
        emb_matrix_val *= params.embedding_dim**0.5

      self.assertEqual(emb_matrix_val.shape, (8000, 128))
      self.assertEqual(ids_val.shape, (2, 1))

      self.assertEqual(outputs_val.shape, (2, 1, 128))
      self.assertEqual(fast_outputs_val.shape, (2, 1, 128))

  def testKmeansSimpleEmbeddingLayerForLoop(self):
    self._testKmeansSimpleEmbeddingLayer(False, True, None)

  def testKmeansSimpleEmbeddingLayerForLoop2D(self):
    self._testKmeansSimpleEmbeddingLayer(False, False, None)

  def testKmeansSimpleEmbeddingLayerMatmul(self):
    self._testKmeansSimpleEmbeddingLayer(True, False, None)

  def testKmeansSimpleEmbeddingLayerGather(self):
    self._testKmeansSimpleEmbeddingLayer(False, False, 'gather')

  def testKmeansSimpleEmbeddingLayerScaling(self):
    self._testKmeansSimpleEmbeddingLayer(True, False, None, True)

  def testSimpleEmbeddingLayerMasked(self):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(398847392)
      params = layers.SimpleEmbeddingLayer.Params()
      params.name = 'emd'
      params.dtype = tf.float32
      params.vocab_size = 10
      params.embedding_dim = 5
      params.fprop_mode = 'gather'
      params.use_3d_weight_tensor = False
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      params.apply_pruning = True

      emb_layer = layers.SimpleEmbeddingLayer(params)
      emb_matrix = emb_layer.vars.wm
      ids = tf.constant([[1], [2]])
      outputs = emb_layer.EmbLookupDefaultTheta(ids)

      self.assertIn('wm', emb_layer.vars.wm.name)
      self.assertIn('mask', emb_layer.vars.mask.name)
      self.assertIn('threshold', emb_layer.vars.threshold.name)

      self.assertEqual(emb_layer.theta.wm.get_shape(), tf.TensorShape([10, 5]))
      self.assertEqual(emb_layer.theta.mask.get_shape(), tf.TensorShape([10,
                                                                         5]))
      self.assertEqual(emb_layer.theta.threshold.get_shape(),
                       tf.TensorShape([]))

      embedding_var_count = 1
      wts = tf.get_collection('SimpleEmbeddingLayer_vars')
      self.assertEqual(embedding_var_count, len(wts))

      embedding_mask_count = 1
      masks = tf.get_collection('masks')
      self.assertEqual(embedding_mask_count, len(masks))

      emebdding_threshold_count = 1
      threshold = tf.get_collection('thresholds')
      self.assertEqual(emebdding_threshold_count, len(threshold))

    with self.session(use_gpu=False, graph=g):
      self.evaluate(tf.global_variables_initializer())
      emb_matrix_val, _, outputs_val = self.evaluate([emb_matrix, ids, outputs])

      self.assertAllClose(emb_matrix_val[1:3], outputs_val[:, 0, :])

  def _testSimpleEmbeddingLayerGrad(self,
                                    use_matmul,
                                    use_3d_weight_tensor,
                                    scale_sqrt_depth=False):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.SimpleEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 8000
      params.embedding_dim = 128
      params.use_matmul = use_matmul
      params.use_3d_weight_tensor = use_3d_weight_tensor
      params.scale_sqrt_depth = scale_sqrt_depth
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = layers.SimpleEmbeddingLayer(params)
      ids = tf.constant([89, 100, 89, 89])
      embs = emb_layer.EmbLookupDefaultTheta(ids) * tf.constant([[0.1], [0.2],
                                                                 [0.3], [0.4]])
      embs_sum = tf.reduce_sum(embs)
      emb_weight = emb_layer.vars.wm
      emb_grad, = tf.gradients(ys=[embs_sum], xs=[emb_weight])
      self.evaluate(tf.global_variables_initializer())
      emb_grad_val = self.evaluate(emb_grad)

    if not use_matmul:
      # tf.embedding_lookup's gradient is a sparse representation.
      # For testing, we convert it to a dense representation.
      o_grad_matrix = np.zeros((8000, 128))
      for i in range(emb_grad_val.indices.shape[0]):
        o_grad_matrix[emb_grad_val.indices[i], :] += emb_grad_val.values[i, :]
      emb_grad_val = o_grad_matrix

    expected_emb_grad = np.zeros(shape=(8000, 128))
    expected_emb_grad[89, :] = 0.8
    expected_emb_grad[100, :] = 0.2
    if scale_sqrt_depth:
      expected_emb_grad *= params.embedding_dim**0.5
    self.assertAllClose(expected_emb_grad, emb_grad_val)

  def testSimpleEmbeddingLayerGradForLoop(self):
    self._testSimpleEmbeddingLayerGrad(False, True)

  def testSimpleEmbeddingLayerGradForLoop2D(self):
    self._testSimpleEmbeddingLayerGrad(False, False)

  def testSimpleEmbeddingLayerGradMatmul(self):
    self._testSimpleEmbeddingLayerGrad(True, False)

  def testSimpleEmbeddingLayerGradScaling(self):
    self._testSimpleEmbeddingLayerGrad(True, False, True)

  def testCompareEmbeddingLayers(self):
    classes = 8000
    dims = 128
    g = tf.Graph()
    with g.as_default():
      ids = tf.placeholder(tf.int32)

      def CreateSimple():
        tf.random.set_seed(398847392)
        p = layers.SimpleEmbeddingLayer.Params()
        p.name = 'emb'
        p.dtype = tf.float32
        p.vocab_size = classes
        p.embedding_dim = dims
        p.params_init = py_utils.WeightInit.Gaussian(0.01)
        p.vn.global_vn = False
        p.vn.per_step_vn = False
        return layers.SimpleEmbeddingLayer(p)

      simple = CreateSimple()
      simple_outs = simple.EmbLookupDefaultTheta(ids)
      simple_grad = tf.gradients(simple_outs, simple.vars.wm)[0]

      def CreateOriginal():
        tf.random.set_seed(398847392)
        p = layers.EmbeddingLayer.Params()
        p.name = 'emb'
        p.dtype = tf.float32
        p.vocab_size = classes
        p.embedding_dim = dims
        p.max_num_shards = 1
        p.params_init = py_utils.WeightInit.Gaussian(0.01)
        p.vn.global_vn = False
        p.vn.per_step_vn = False
        return layers.EmbeddingLayer(p)

      original = CreateOriginal()
      weight = tf.identity(simple.vars.wm)
      theta = py_utils.NestedMap()
      theta.wm = [weight]
      original_outs = original.EmbLookup(theta, ids)
      original_grad = tf.gradients(original_outs, weight)[0]

    ids_val = np.random.randint(0, high=classes, size=(4000,))
    with self.session(graph=g) as sess:
      self.evaluate(tf.global_variables_initializer())
      s_outs, s_grad, o_outs, o_grad = sess.run(
          [simple_outs, simple_grad, original_outs, original_grad],
          feed_dict={ids: ids_val})
      self.assertAllClose(s_outs, o_outs)
      self.assertAllClose(s_grad, o_grad)

  def testCompareMatMulEmbeddingLayers(self):
    classes = 8000
    dims = 128
    g = tf.Graph()
    with g.as_default():
      ids = tf.placeholder(tf.int32)

      def CreateSimpleMatMul():
        tf.random.set_seed(398847392)
        p = layers.SimpleEmbeddingLayer.Params()
        p.name = 's_emb'
        p.dtype = tf.float32
        p.vocab_size = classes
        p.embedding_dim = dims
        p.fprop_mode = 'matmul'
        p.params_init = py_utils.WeightInit.Gaussian(0.01)
        p.vn.global_vn = False
        p.vn.per_step_vn = False
        return layers.SimpleEmbeddingLayer(p)

      simple = CreateSimpleMatMul()
      simple_outs = simple.EmbLookupDefaultTheta(ids)
      simple_grad = tf.gradients(simple_outs, simple.vars.wm)[0]

      def CreateEinsum():
        tf.random.set_seed(398847392)
        p = layers.EinsumEmbeddingLayer.Params()
        p.name = 'e_emb'
        p.dtype = tf.float32
        p.vocab_size = classes
        p.embedding_dim = dims
        p.params_init = py_utils.WeightInit.Gaussian(0.01)
        p.vn.global_vn = False
        p.vn.per_step_vn = False
        return p.Instantiate()

      einsum = CreateEinsum()
      weight = tf.identity(simple.vars.wm)
      einsum_outs = einsum.EmbLookup(py_utils.NestedMap(wm=weight), ids)
      einsum_grad = tf.gradients(einsum_outs, weight)[0]
      tf.logging.info('simple_grad=%s einsum_grad=%s', simple_grad, einsum_grad)

    ids_val = np.random.randint(0, high=classes, size=(4000,))
    with self.session(graph=g) as sess:
      self.evaluate(tf.global_variables_initializer())
      s_outs, s_grad, e_outs, e_grad = sess.run(
          [simple_outs, simple_grad, einsum_outs, einsum_grad],
          feed_dict={ids: ids_val})
      self.assertAllClose(s_outs, e_outs)
      self.assertAllClose(s_grad, e_grad)

  def testPositionalEmbeddingLayer(self):
    with self.session(use_gpu=False):
      p = layers.PositionalEmbeddingLayer.Params()
      p.name = 'position_emb'
      p.min_timescale = 1
      p.max_timescale = 7
      p.embedding_dim = 4
      seq_length = 11

      pos_emb_layer = layers.PositionalEmbeddingLayer(p)
      position_embs = pos_emb_layer.FPropDefaultTheta(seq_length)
      actual_position_embs, = self.evaluate([position_embs])

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032],
          [-0.7568025 ,  0.54083425, -0.65364361,  0.84112918],
          [-0.95892417,  0.65507787,  0.28366217,  0.75556135],
          [-0.27941549,  0.75597537,  0.96017027,  0.65460002],
          [ 0.65698659,  0.84147096,  0.7539022 ,  0.54030228],
          [ 0.98935831,  0.90982294, -0.14550003,  0.41499668],
          [ 0.41211855,  0.9596386 , -0.91113025,  0.28123617],
          [-0.54402113,  0.98990309, -0.83907151,  0.14174587]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print('expected_position_embs:', expected_output)
      print('actual_position_embs:', actual_position_embs)
      self.assertAllClose(actual_position_embs, expected_output)

  def testPositionalEmbeddingLayerWithPosition(self):
    with self.session(use_gpu=False):
      p = layers.PositionalEmbeddingLayer.Params()
      p.name = 'position_emb'
      p.min_timescale = 1
      p.max_timescale = 7
      p.embedding_dim = 4
      pos_tensor = tf.constant(
          np.asarray([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3],
                      [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 0]]),
          dtype=tf.int32)

      pos_emb_layer = layers.PositionalEmbeddingLayer(p)
      position_embs = pos_emb_layer.FPropWithPosition(pos_emb_layer.theta,
                                                      pos_tensor)
      actual_position_embs, = self.evaluate([position_embs])

      # pylint: disable=bad-whitespace,bad-continuation
      # pyformat: disable
      expected_output = [
          [[ 0.        ,  0.        ,  1.        ,  1.       ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032],
          [-0.7568025 ,  0.54083425, -0.65364361,  0.84112918],
          [-0.95892417,  0.65507787,  0.28366217,  0.75556135],
          [-0.27941549,  0.75597537,  0.96017027,  0.65460002],
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032]],
          [[ 0.        ,  0.        ,  1.        ,  1.       ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032],
          [-0.7568025 ,  0.54083425, -0.65364361,  0.84112918],
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.        ,  0.        ,  1.        ,  1.        ]]
      ]
      # pyformat: enable
      # pylint: enable=bad-whitespace,bad-continuation
      print('expected_position_embs:', expected_output)
      print('actual_position_embs:', actual_position_embs)
      self.assertAllClose(actual_position_embs, expected_output)

  def testPositionalEmbeddingLayerWithScaling(self):
    with self.session(use_gpu=False):
      p = layers.PositionalEmbeddingLayer.Params()
      p.name = 'position_emb'
      p.min_timescale = 1
      p.max_timescale = 7
      p.embedding_dim = 4
      p.trainable_scaling = True
      p.trainable_scaling_init = 1.0 / np.sqrt(p.embedding_dim)
      seq_length = 11

      pos_emb_layer = layers.PositionalEmbeddingLayer(p)
      position_embs = pos_emb_layer.FPropDefaultTheta(seq_length)
      self.evaluate(tf.global_variables_initializer())
      actual_position_embs, = self.evaluate([position_embs])

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [ 0.        ,  0.        ,  1.        ,  1.        ],
          [ 0.84147096,  0.14237173,  0.54030228,  0.98981327],
          [ 0.90929741,  0.28184283, -0.41614676,  0.95946062],
          [ 0.14112   ,  0.4155719 , -0.9899925 ,  0.90956032],
          [-0.7568025 ,  0.54083425, -0.65364361,  0.84112918],
          [-0.95892417,  0.65507787,  0.28366217,  0.75556135],
          [-0.27941549,  0.75597537,  0.96017027,  0.65460002],
          [ 0.65698659,  0.84147096,  0.7539022 ,  0.54030228],
          [ 0.98935831,  0.90982294, -0.14550003,  0.41499668],
          [ 0.41211855,  0.9596386 , -0.91113025,  0.28123617],
          [-0.54402113,  0.98990309, -0.83907151,  0.14174587]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertAllClose(expected_output / np.sqrt(p.embedding_dim),
                          actual_position_embs)

  def testRelativePositionalEmbeddingLayer(self):
    with self.session(use_gpu=False):
      radius = 3
      p = layers.RelativePositionalEmbeddingLayer.Params().Set(
          name='rel_position_emb', radius=radius, dim=4)
      layer = p.Instantiate()
      indices = np.array([-5, -2, 0, 1, 4], dtype=np.int32)
      pos_emb = layer.FPropDefaultTheta(tf.convert_to_tensor(indices))

      self.evaluate(tf.global_variables_initializer())
      actual_pos_emb, full_emb = self.evaluate([pos_emb, layer.vars.w])

      clipped_indices = np.vectorize(lambda x: max(-radius, min(radius, x)))(
          indices) + radius
      expected_output = np.take_along_axis(full_emb,
                                           np.expand_dims(clipped_indices, -1),
                                           0)
      print('expected_position_embs:', expected_output)
      print('actual_position_embs:', actual_pos_emb)
      self.assertAllClose(actual_pos_emb, expected_output)

  def testSinusoidalPositionalEmbeddingLayer(self):
    with self.session(use_gpu=False):
      p = layers.SinusoidalPositionalEmbeddingLayer.Params()
      p.name = 'position_emb'
      p.embedding_dim = 2
      seq_length = 4

      pos_emb_layer = layers.SinusoidalPositionalEmbeddingLayer(p)
      position_embs = pos_emb_layer.FPropDefaultTheta(seq_length)
      actual_position_embs, = self.evaluate([position_embs])
      expected_output = [[math.sin(p / 2 * math.pi),
                          math.cos(p / 2 * math.pi)] for p in range(4)]
      self.assertAllClose(actual_position_embs, expected_output)

  def testOneHotEmbeddingLayer(self):
    with self.session(use_gpu=True):
      params = layers.OneHotEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 4
      params.embedding_dim = 4
      emb_layer = layers.OneHotEmbeddingLayer(params)
      ids = tf.constant([[0], [2]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      self.evaluate(tf.global_variables_initializer())
      expected_output = [[[1., 0., 0., 0.]], [[0., 0., 1., 0.]]]
      self.assertAllClose(expected_output, self.evaluate(embs))

  def testOneHotEmbeddingLayerWithUncertainty(self):
    with self.session(use_gpu=True):
      params = layers.OneHotEmbeddingLayer.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 4
      params.embedding_dim = 4
      params.uncertainty = 0.3
      emb_layer = layers.OneHotEmbeddingLayer(params)
      ids = tf.constant([[0], [2]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      self.evaluate(tf.global_variables_initializer())
      expected_output = [[[0.7, 0.1, 0.1, 0.1]], [[0.1, 0.1, 0.7, 0.1]]]
      self.assertAllClose(expected_output, self.evaluate(embs))


class SoftmaxLayerTest(test_utils.TestCase):

  def _RunSimpleFullSoftmax(self,
                            num_shards=1,
                            chunk_size=0,
                            inputs=None,
                            class_ids=None,
                            class_weights=None,
                            class_probabilities=None,
                            num_samples=0,
                            default_qdomain=None,
                            logits_qdomain=None,
                            training_step=-1,
                            seed=None,
                            dtype=tf.float32,
                            fprop_dtype=None,
                            apply_pruning=False,
                            use_bias=True):
    if fprop_dtype is None:
      fprop_dtype = dtype
    with self.session(use_gpu=True, graph=tf.Graph()):
      if seed is not None:
        tf.random.set_seed(seed)
      if class_ids is None:
        class_ids = tf.constant([[1], [5], [10]], dtype=tf.int32)
      else:
        class_ids = tf.constant(class_ids)
      if class_weights is None:
        class_weights = tf.constant([1.0, 0.4, 0.8], dtype=fprop_dtype)
      else:
        class_weights = tf.constant(class_weights)
      np.random.seed(12345)
      if inputs is None:
        inputs = [tf.constant(np.random.rand(3, 10), dtype=fprop_dtype)]
      else:
        inputs = [tf.constant(inputs, dtype=fprop_dtype)]

      params = layers.SimpleFullSoftmax.Params()
      params.dtype = dtype
      params.fprop_dtype = fprop_dtype
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.num_shards = num_shards
      params.chunk_size = chunk_size
      params.apply_pruning = apply_pruning
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.random_seed = 12345678
      params.use_bias = use_bias

      if default_qdomain is not None:
        params.qdomain.default = default_qdomain
      if logits_qdomain is not None:
        params.qdomain.logits = logits_qdomain

      if num_samples > 0:
        # Turn on sampled soft-max; the asserts need to hold for it to be used.
        params.num_sampled = num_samples
        assert class_probabilities is None
        assert chunk_size == 0

      params.vn.global_vn = False
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.FProp(
          softmax.theta,
          inputs,
          class_weights=class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)

      all_vars = tf.get_collection('SimpleFullSoftmax_vars')
      expected_var_names = []
      for i in range(num_shards):
        expected_var_names.append(u'softmax/weight_%d/var:0' % i)
        if use_bias:
          expected_var_names.append(u'softmax/bias_%d/var:0' % i)

      all_var_names = [v.name for v in all_vars]
      self.assertCountEqual(expected_var_names, all_var_names)

      self.evaluate(tf.global_variables_initializer())
      if training_step >= 0:
        self.evaluate(
            tf.assign(py_utils.GetOrCreateGlobalStepVar(), training_step))
      return self.evaluate(xent_loss)

  def testSimpleFullSoftmaxMasked(self):
    num_shards = 2
    apply_pruning = True
    params = layers.SimpleFullSoftmax.Params()
    params.name = 'softmax'
    params.dtype = tf.float32
    params.input_dim = 10
    params.num_classes = 32
    params.fprop_dtype = tf.float32
    params.num_shards = num_shards
    params.apply_pruning = apply_pruning
    params.random_seed = 12345678
    softmax_layer = layers.SimpleFullSoftmax(params)

    self.assertIn('weight_0', softmax_layer.vars.weight_0.name)
    self.assertIn('weight_1', softmax_layer.vars.weight_1.name)
    self.assertIn('mask_0', softmax_layer.vars.mask_0.name)
    self.assertIn('mask_1', softmax_layer.vars.mask_1.name)
    self.assertIn('threshold_0', softmax_layer.vars.threshold_0.name)
    self.assertIn('threshold_1', softmax_layer.vars.threshold_1.name)

    self.assertEqual(softmax_layer.theta.weight_0.get_shape(),
                     tf.TensorShape([10, 16]))
    self.assertEqual(softmax_layer.theta.weight_1.get_shape(),
                     tf.TensorShape([10, 16]))
    self.assertEqual(softmax_layer.theta.mask_0.get_shape(),
                     tf.TensorShape([10, 16]))
    self.assertEqual(softmax_layer.theta.mask_1.get_shape(),
                     tf.TensorShape([10, 16]))
    self.assertEqual(softmax_layer.theta.threshold_0.get_shape(),
                     tf.TensorShape([]))
    self.assertEqual(softmax_layer.theta.threshold_0.get_shape(),
                     tf.TensorShape([]))

    softmax_var_count = 4  # 2 each for weights and biases (we have 2 shards)
    wts = tf.get_collection('SimpleFullSoftmax_vars')
    self.assertEqual(softmax_var_count, len(wts))

    softmax_mask_count = 2
    masks = tf.get_collection('masks')
    self.assertEqual(softmax_mask_count, len(masks))

    softmax_threshold_count = 2
    threshold = tf.get_collection('thresholds')
    self.assertEqual(softmax_threshold_count, len(threshold))

    # Sampled and Masked
    xent_loss = self._RunSimpleFullSoftmax(
        num_samples=32, seed=12345, apply_pruning=True)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.681571, 1e-5)
    self.assertNear(log_perplexity, 3.946169, 1e-5)

    # Sharded and Masked
    xent_loss = self._RunSimpleFullSoftmax(num_shards=2, apply_pruning=True)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 6.14888, 1e-5)
    self.assertNear(log_perplexity, 2.79495, 1e-5)

    # Non_2D and Masked
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_ids=np.random.randint(32, size=(4, 3)),
        apply_pruning=True)
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_probabilities=np.random.uniform(size=(4, 3, 32)),
        apply_pruning=True)
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    # Chunked and Masked
    for chunk_size in (0, 1, 2, 3, 4, 5):
      print('chunk_size = ', chunk_size)
      xent_output = self._RunSimpleFullSoftmax(
          chunk_size=chunk_size, apply_pruning=True)
      loss = xent_output.total_xent
      log_perplexity = xent_output.avg_xent
      print('xent_output ', xent_output)
      print('xent_output.per_example_argmax.dtype ',
            xent_output.per_example_argmax.dtype)
      self.assertAllClose(loss, 6.22425)
      self.assertAllClose(log_perplexity, 2.82920)
      self.assertAllEqual(xent_output.per_example_argmax,
                          np.argmax(xent_output.logits, axis=1))

  def testSimpleFullSoftmax_Sampled(self):
    xent_loss = self._RunSimpleFullSoftmax(num_samples=32, seed=12345)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.681571, 1e-5)
    self.assertNear(log_perplexity, 3.946169, 1e-5)

  def testSimpleFullSoftmax_NoBias(self):
    xent_loss = self._RunSimpleFullSoftmax(seed=12345, use_bias=False)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    err = 1e-5
    self.assertNear(loss, 12.476410, err=err)
    self.assertNear(log_perplexity, 5.671095, err=err)

  def testSimpleFullSoftmax_SampledAndSharded(self):
    xent_loss = self._RunSimpleFullSoftmax(
        num_shards=4, num_samples=32, seed=12345)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    self.assertNear(loss, 8.510439, 1e-5)
    self.assertNear(log_perplexity, 3.868381, 1e-5)

  def testSimpleFullSoftmax_Non2D(self):
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_ids=np.random.randint(32, size=(4, 3)))
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3)),
        class_probabilities=np.random.uniform(size=(4, 3, 32)))
    self.assertEqual(xent_loss.logits.shape, (4, 3, 32))
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

  def _testSimpleFullSoftmax_Basic_Helper(self, dtype, fprop_dtype):
    xent_loss = self._RunSimpleFullSoftmax(dtype=dtype, fprop_dtype=fprop_dtype)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    err = 1e-5
    if fprop_dtype == tf.float16 or fprop_dtype == tf.bfloat16:
      err = 1e-2
    self.assertNear(loss, 6.22425, err=err)
    self.assertNear(log_perplexity, 2.8292, err=err)
    self.assertAllEqual(xent_loss.per_example_argmax,
                        np.argmax(xent_loss.logits, axis=1))

  def testSimpleFullSoftmax_Basic_Float32(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float32)

  def testSimpleFullSoftmax_Basic_Float32Float16(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float16)

  def testSimpleFullSoftmax_Sharded(self):
    xent_loss = self._RunSimpleFullSoftmax(2)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    self.assertNear(loss, 6.14888, 1e-5)
    self.assertNear(log_perplexity, 2.79495, 1e-5)

  def testSimpleFullSoftmax_Chunked(self):
    for chunk_size in (0, 1, 2, 3, 4, 5):
      print('chunk_size = ', chunk_size)
      xent_output = self._RunSimpleFullSoftmax(chunk_size=chunk_size)
      loss = xent_output.total_xent
      log_perplexity = xent_output.avg_xent
      print('xent_output ', xent_output)
      print('xent_output.per_example_argmax.dtype ',
            xent_output.per_example_argmax.dtype)
      self.assertAllClose(loss, 6.22425)
      self.assertAllClose(log_perplexity, 2.82920)
      self.assertAllEqual(xent_output.per_example_argmax,
                          np.argmax(xent_output.logits, axis=1))

  def testSimpleFullSoftmax_Basic_Distributions(self):
    with self.session(use_gpu=False):
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = False
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs,
          class_weights=class_weights,
          class_probabilities=tf.one_hot(class_ids, params.num_classes))
      self.evaluate(tf.global_variables_initializer())
      loss = self.evaluate(xent_loss.total_xent)
      log_perplexity = self.evaluate(xent_loss.avg_xent)
      print(['loss', loss])
      print(['log_perplexity', log_perplexity])
      self.assertNear(loss, 6.22425, 1e-5)
      self.assertNear(log_perplexity, 2.8292, 1e-5)

  def testSimpleFullSoftmax_GlobalVN(self):
    with self.session(use_gpu=False):
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = True
      params.vn.seed = 23456
      params.vn.scale = 1.0
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs, class_weights=class_weights, class_ids=class_ids)
      self.evaluate(tf.global_variables_initializer())
      loss = self.evaluate(xent_loss.total_xent)
      log_perplexity = self.evaluate(xent_loss.avg_xent)
      print(['testSimpleFullSoftmax_GlobalVN loss', loss])
      print(['testSimpleFullSoftmax_GlobalVN log_perplexity', log_perplexity])
      self.assertNear(loss, 16.186937, 1e-4)
      self.assertNear(log_perplexity, 7.35769, 1e-4)

  def testSimpleFullSoftmax_PerStepVN(self):
    with self.session(use_gpu=False):
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.vn.global_vn = False
      params.vn.per_step_vn = True
      params.vn.seed = 23456
      params.vn.scale = 1.0
      softmax = layers.SimpleFullSoftmax(params)
      xent_loss = softmax.XentLoss(
          inputs, class_weights=class_weights, class_ids=class_ids)
      self.evaluate(tf.global_variables_initializer())
      loss = self.evaluate(xent_loss.total_xent)
      log_perplexity = self.evaluate(xent_loss.avg_xent)
      print(['testShardedFullSoftmax_PerStepVN loss', loss])
      print(['testShardedFullSoftmax_PerStepVN log_perplexity', log_perplexity])
      self.assertNear(loss, 8.315969, 1e-4)
      self.assertNear(log_perplexity, 3.779986, 1e-4)

  def testSimpleFullSoftmax_FakeQuantized(self):
    default_qdomain = quant_utils.SymmetricScheduledClipQDomain.Params()
    default_qdomain.cc_schedule = quant_utils.FakeQuantizationSchedule.Params(
    ).Set(
        clip_start_step=0, clip_end_step=2, quant_start_step=2)
    logits_qdomain = default_qdomain.Copy()
    xent_loss = self._RunSimpleFullSoftmax(
        default_qdomain=default_qdomain,
        logits_qdomain=logits_qdomain,
        training_step=5)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    self.assertNear(loss, 6.285590, 1e-5)
    self.assertNear(log_perplexity, 2.857086, 1e-5)

  def _RunSimpleFullSoftmaxGradientChecker(self, batch_size, num_classes,
                                           chunk_size, num_shards):
    for (dtype, use_gpu, tolerance) in [(tf.float32, True, 1e-2),
                                        (tf.float64, False, 1e-6)]:
      tf.logging.info('dtype %s tolerance %g', dtype, tolerance)
      with self.session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
        input_dim = 10
        np.random.seed(12345)
        class_ids = tf.constant(
            np.random.randint(num_classes, size=(batch_size, 1)),
            dtype=tf.int32)
        class_weights = tf.constant(np.random.rand(batch_size), dtype=dtype)
        inputs = [
            tf.constant(np.random.rand(batch_size, input_dim), dtype=dtype)
        ]

        params = layers.SimpleFullSoftmax.Params()
        params.name = 'softmax'
        params.dtype = dtype
        params.input_dim = input_dim
        params.num_classes = num_classes
        params.num_shards = num_shards
        params.chunk_size = chunk_size
        params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
        params.vn.global_vn = False
        softmax = layers.SimpleFullSoftmax(params)
        xent_loss = softmax.XentLoss(
            inputs, class_weights=class_weights, class_ids=class_ids)
        softmax_vars = softmax.vars.Flatten()
        # Now add the backward graph.
        grads = tf.gradients(xent_loss.total_xent, softmax_vars)

        self.evaluate(tf.global_variables_initializer())
        assert len(softmax_vars) == len(grads)
        for x, grad_x in zip(softmax_vars, grads):
          grad_symbolic = self.evaluate(grad_x)
          grad_numeric = test_utils.ComputeNumericGradient(
              sess, xent_loss.total_xent, x)
          self.assertAllClose(
              grad_symbolic, grad_numeric, atol=tolerance, rtol=tolerance)

  def testSimpleFullSoftmaxGradientChecker(self):
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 0, 1)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 0, 2)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 2, 2)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 5, 2)

  def testSimpleFullSoftmax_SymbolicShape(self):
    with self.session(use_gpu=False):
      class_ids = tf.constant([1, 5, 10], dtype=tf.int32)
      class_weights = tf.constant([1.0, 0.4, 0.8], dtype=tf.float32)
      np.random.seed(12345)
      inputs = [tf.constant(np.random.rand(3, 10), dtype=tf.float32)]

      # Use a symbol to represent the input dim.
      input_dim = symbolic.Symbol('input_dim')
      params = layers.SimpleFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = input_dim
      params.num_classes = 32
      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES, {input_dim: 10}):
        softmax = layers.SimpleFullSoftmax(params)
        xent_loss = softmax.XentLoss(
            inputs, class_weights=class_weights, class_ids=class_ids)
        self.evaluate(tf.global_variables_initializer())
        self.evaluate(xent_loss.total_xent)


class SingleShardSoftmaxLayerTest(test_utils.TestCase):

  def _RunSimpleFullSoftmax(self,
                            inputs=None,
                            class_ids=None,
                            class_weights=None,
                            class_probabilities=None,
                            chunk_size=0,
                            dtype=tf.float32,
                            fprop_dtype=None):
    if fprop_dtype is None:
      fprop_dtype = dtype
    with self.session(use_gpu=True, graph=tf.Graph()):
      inputs = tf.constant(inputs, dtype=fprop_dtype)
      if class_ids is not None:
        class_ids = tf.constant(class_ids, dtype=tf.int32)
      if class_weights is not None:
        class_weights = tf.constant(class_weights, dtype=dtype)
      if class_probabilities is not None:
        class_probabilities = tf.constant(class_probabilities, dtype=dtype)

      params = layers.SingleShardFullSoftmax.Params()
      params.dtype = dtype
      params.fprop_dtype = fprop_dtype
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.chunk_size = chunk_size
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.random_seed = 12345678

      params.vn.global_vn = False
      softmax = params.Instantiate()
      xent_loss = softmax.FProp(
          softmax.theta,
          inputs,
          class_weights=class_weights,
          class_ids=class_ids,
          class_probabilities=class_probabilities)

      self.evaluate(tf.global_variables_initializer())
      return self.evaluate(xent_loss)

  def testSoftmaxCapping(self):
    with self.session(use_gpu=True, graph=tf.Graph()):
      inputs = tf.constant(np.random.rand(4, 3, 10), dtype=tf.float32)
      class_weights = tf.constant(np.ones((4, 3, 1)), dtype=tf.float32)
      class_ids = tf.constant(
          np.random.randint(32, size=(4, 3, 1)), dtype=tf.int32)

      params = layers.SingleShardFullSoftmax.Params()
      params.name = 'softmax'
      params.input_dim = 10
      params.num_classes = 32
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      params.logits_soft_max = 1.0
      params.random_seed = 12345678

      params.vn.global_vn = False
      softmax = params.Instantiate()
      xent_loss = softmax.FProp(
          softmax.theta,
          inputs,
          class_weights=class_weights,
          class_ids=class_ids)

      self.evaluate(tf.global_variables_initializer())
      return self.evaluate(xent_loss)

  def testSimpleFullSoftmax_Non2D_ClassId(self):
    np.random.seed(1234578)
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3, 1)),
        class_ids=np.random.randint(32, size=(4, 3, 1)),
        chunk_size=2)
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

  def testSimpleFullSoftmax_Non2D_ClassProb(self):
    np.random.seed(12345)
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=np.random.rand(4, 3, 10),
        class_weights=np.ones((4, 3, 1)),
        class_probabilities=np.random.randint(32, size=(4, 3, 32)),
        chunk_size=1)
    self.assertEqual(xent_loss.per_example_xent.shape, (4, 3))
    self.assertEqual(xent_loss.per_example_weight.shape, (4, 3))

  def _testSimpleFullSoftmax_Basic_Helper(self, dtype, fprop_dtype):
    np.random.seed(12345)
    class_ids = [[1], [5], [10]]
    class_weights = [[1.0], [0.4], [0.8]]
    inputs = np.random.rand(3, 10)
    xent_loss = self._RunSimpleFullSoftmax(
        inputs=inputs,
        class_weights=class_weights,
        class_ids=class_ids,
        dtype=dtype,
        fprop_dtype=fprop_dtype)
    loss = xent_loss.total_xent
    log_perplexity = xent_loss.avg_xent
    print(['loss', loss])
    print(['log_perplexity', log_perplexity])
    err = 1e-5
    if fprop_dtype == tf.float16 or fprop_dtype == tf.bfloat16:
      err = 1e-2
    self.assertNear(loss, 6.22425, err=err)
    self.assertNear(log_perplexity, 2.8292, err=err)
    self.assertAllEqual(xent_loss.per_example_argmax,
                        np.argmax(xent_loss.logits, axis=1))

  def testSimpleFullSoftmax_Basic_Float32(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float32)

  def testSimpleFullSoftmax_Basic_Float32Float16(self):
    self._testSimpleFullSoftmax_Basic_Helper(
        dtype=tf.float32, fprop_dtype=tf.float16)

  def testSimpleFullSoftmax_Chunked(self):
    np.random.seed(12345)
    class_ids = [[1], [5], [10]]
    class_weights = [[1.0], [0.4], [0.8]]
    inputs = np.random.rand(3, 10)
    per_example_xent = None
    per_example_argmax = None
    for chunk_size in (0, 1, 3):
      xent_output = self._RunSimpleFullSoftmax(
          inputs=inputs,
          class_weights=class_weights,
          class_ids=class_ids,
          chunk_size=chunk_size)
      loss = xent_output.total_xent
      log_perplexity = xent_output.avg_xent
      print('xent_output ', xent_output)
      print('xent_output.per_example_argmax.dtype ',
            xent_output.per_example_argmax.dtype)
      self.assertAllClose(loss, 6.22425)
      self.assertAllClose(log_perplexity, 2.82920)
      if per_example_xent is None:
        per_example_xent = xent_output.per_example_xent
        per_example_argmax = xent_output.per_example_argmax
      else:
        self.assertAllClose(per_example_xent, xent_output.per_example_xent)
        self.assertAllClose(per_example_argmax, xent_output.per_example_argmax)

  def _RunSimpleFullSoftmaxGradientChecker(self, batch_size, num_classes,
                                           chunk_size):
    for (dtype, use_gpu, tolerance) in [(tf.float32, True, 1e-2),
                                        (tf.float64, False, 1e-6)]:
      tf.logging.info('dtype %s tolerance %g', dtype, tolerance)
      with self.session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
        input_dim = 10
        np.random.seed(12345)
        class_ids = tf.constant(
            np.random.randint(num_classes, size=(batch_size, 1)),
            dtype=tf.int32)
        class_weights = tf.constant(np.random.rand(batch_size, 1), dtype=dtype)
        inputs = tf.constant(np.random.rand(batch_size, input_dim), dtype=dtype)

        params = layers.SingleShardFullSoftmax.Params()
        params.name = 'softmax'
        params.dtype = dtype
        params.input_dim = input_dim
        params.num_classes = num_classes
        params.chunk_size = chunk_size
        params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
        params.vn.global_vn = False
        softmax = params.Instantiate()
        xent_loss = softmax.FProp(
            softmax.theta,
            inputs,
            class_weights=class_weights,
            class_ids=class_ids)
        softmax_vars = softmax.vars.Flatten()
        # Now add the backward graph.
        grads = tf.gradients(xent_loss.total_xent, softmax_vars)

        self.evaluate(tf.global_variables_initializer())
        assert len(softmax_vars) == len(grads)
        for x, grad_x in zip(softmax_vars, grads):
          grad_symbolic = self.evaluate(grad_x)
          grad_numeric = test_utils.ComputeNumericGradient(
              sess, xent_loss.total_xent, x)
          self.assertAllClose(
              grad_symbolic, grad_numeric, atol=tolerance, rtol=tolerance)

  def testSimpleFullSoftmaxGradientChecker(self):
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 0)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 1)
    self._RunSimpleFullSoftmaxGradientChecker(3, 4, 3)


class SoftmaxLayerLogitsTest(test_utils.TestCase):
  """Testing SoftmaxLayer.Logits()."""

  def _Logits(self, params, batch_size=2, seq_length=None):
    with self.session(use_gpu=True, graph=tf.Graph()):
      np.random.seed(12345)
      tf.random.set_seed(1234)

      params.name = 'softmax'
      if not params.input_dim:
        params.input_dim = 3
      if not params.num_classes:
        params.num_classes = 4
      params.params_init = py_utils.WeightInit.Gaussian(0.5, 123456)
      softmax = params.Instantiate()

      input_dim = params.input_dim
      if seq_length:
        inputs = np.random.rand(batch_size, seq_length, input_dim)
      else:
        inputs = np.random.rand(batch_size, input_dim)
      inputs = tf.constant(inputs, dtype=py_utils.FPropDtype(params))
      logits = softmax.Logits(softmax.theta, inputs)

      if seq_length:
        logits = py_utils.HasShape(logits,
                                   [batch_size, seq_length, params.num_classes])
      else:
        logits = py_utils.HasShape(logits, [batch_size, params.num_classes])
      self.evaluate(tf.global_variables_initializer())
      return self.evaluate(logits)

  def testConvSoftmaxLogits(self):
    params = layers.ConvSoftmax.Params()
    self.assertAllClose([[0.52536774, -0.17598523, 0.38314393, -0.36068222],
                         [0.75792629, -0.18001975, 0.42298675, -0.35423514]],
                        self._Logits(params))

  def testSimpleFullSoftmax(self):
    params = layers.SimpleFullSoftmax.Params()
    self.assertAllClose([[0.52536774, -0.17598523, 0.38314393, -0.36068222],
                         [0.75792629, -0.18001975, 0.42298675, -0.35423514]],
                        self._Logits(params))

  def testConvSoftmaxLogitsWith3DInputs(self):
    params = layers.ConvSoftmax.Params()
    logits = self._Logits(params, seq_length=5)
    self.assertAllClose(6.9934864, np.sum(logits))


class SharedSoftmaxLayerTest(SoftmaxLayerTest):

  def _testSharedSoftmaxLayerEmbLookup(self, scale_sqrt_depth=False):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(398847392)
      params = layers.SharedSoftmaxLayer.Params().Set(
          softmax=layers.SimpleFullSoftmax.Params().Set(
              num_shards=1, chunk_size=0, apply_pruning=False),
          dtype=tf.float32,
          fprop_dtype=None,
          name='shared_layer',
          input_dim=128,
          num_classes=8000,
          params_init=py_utils.WeightInit.Gaussian(0.5, 123456),
          scale_sqrt_depth=scale_sqrt_depth,
          random_seed=12345678)

      emb_layer = layers.SharedSoftmaxLayer(params)

      emb_matrix = tf.einsum(
          'ji',
          emb_layer.softmax.DenseWeights(emb_layer.softmax.theta).wm)
      ids = tf.constant([[89], [100]])
      outputs = emb_layer.EmbLookup(emb_layer.theta, ids)

    with self.session(use_gpu=True, graph=g):
      self.evaluate(tf.global_variables_initializer())
      emb_matrix_val, ids_val, outputs_val = self.evaluate(
          [emb_matrix, ids, outputs])
      self.assertEqual(emb_matrix_val.shape, (8000, 128))
      self.assertEqual(ids_val.shape, (2, 1))
      self.assertEqual(outputs_val.shape, (2, 1, 128))
      if scale_sqrt_depth:
        emb_matrix_val *= params.input_dim**0.5
      self.assertAllClose(emb_matrix_val[89, :], outputs_val[0, 0, :])
      self.assertAllClose(emb_matrix_val[100, :], outputs_val[1, 0, :])

  def testSharedSoftmaxLayerEmbLookup(self):
    self._testSharedSoftmaxLayerEmbLookup()

  def testSharedSoftmaxLayerEmbLookupScaling(self):
    self._testSharedSoftmaxLayerEmbLookup(True)


class EinsumSoftmaxLayerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'no_label_smoothing',
          'expected_loss': 27.981373
      },
      {
          'testcase_name': 'bfloat16_input',
          'expected_loss': 27.980932,
          'input_dtype': tf.bfloat16
      },
      {
          'testcase_name': 'with_label_smoothing',
          'expected_loss': 28.038475,
          'label_smoothing': True
      },
      {
          'testcase_name': 'focal_loss',
          'expected_loss': 27.539188,
          'focal_loss_gamma': 0.5
      },
  )
  def testEinsumSoftmax(self,
                        expected_loss,
                        label_smoothing=False,
                        input_dtype=tf.float32,
                        focal_loss_gamma=None):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(123)
      input_dim = 10
      num_classes = 32
      params = layers.EinsumSoftmax.Params().Set(
          name='softmax',
          input_dim=input_dim,
          num_classes=num_classes,
          focal_loss_gamma=focal_loss_gamma)
      params.random_seed = 12345678
      softmax = params.Instantiate()
      sess.run(tf.global_variables_initializer())
      np.random.seed(12345)
      inputs = tf.constant(np.random.rand(2, 4, 10), dtype=input_dtype)
      logits = softmax.Logits(softmax.theta, inputs)
      self.assertAllEqual([2, 4, num_classes], py_utils.GetShape(logits))
      class_ids = tf.constant([[3, 4, 5, 2], [4, 5, 6, 2]], dtype=tf.int32)
      class_weights = tf.ones_like(class_ids, dtype=tf.float32)
      if label_smoothing:
        class_onehots = tf.one_hot(
            class_ids, depth=num_classes, dtype=tf.float32)
        class_probabilities = (0.1 / (num_classes - 1) * (1. - class_onehots) +
                               0.9 * class_onehots)
      else:
        class_probabilities = None
      per_example_loss, per_example_argmax = softmax.XentLossFromLogits(
          softmax.theta, logits, class_weights, class_ids, class_probabilities)
      self.assertAllEqual([2, 4], py_utils.GetShape(per_example_loss))
      self.assertAllClose(expected_loss,
                          sess.run(tf.reduce_sum(per_example_loss)))
      self.assertAllEqual([2, 4], py_utils.GetShape(per_example_argmax))


class FocalFullSoftmaxLayerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_smooth_no_focal', False, None, 27.981373),
      ('no_smooth_focal', False, 0.5, 27.539188),
      ('smooth_no_focal', True, None, 28.038475),
      ('smooth_focal', True, 0.5, 27.5981063),
  )
  def testFocalFullSoftmax(self, label_smoothing, gamma, expected_loss):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(123)
      input_dim = 10
      num_classes = 32
      params = layers.FocalFullSoftmax.Params().Set(
          name='softmax',
          input_dim=input_dim,
          num_classes=num_classes,
          focal_loss_gamma=gamma)
      params.random_seed = 12345678
      softmax = params.Instantiate()
      sess.run(tf.global_variables_initializer())
      np.random.seed(12345)
      inputs = tf.constant(np.random.rand(8, 10), dtype=tf.float32)
      logits = softmax.Logits(softmax.theta, inputs)
      self.assertAllEqual([8, num_classes], py_utils.GetShape(logits))
      class_ids = tf.constant([3, 4, 5, 2, 4, 5, 6, 2], dtype=tf.int32)
      class_weights = tf.ones_like(class_ids, dtype=tf.float32)
      if label_smoothing:
        class_onehots = tf.one_hot(
            class_ids, depth=num_classes, dtype=tf.float32)
        class_probabilities = (0.1 / (num_classes - 1) * (1. - class_onehots) +
                               0.9 * class_onehots)
      else:
        class_probabilities = None
      per_example_loss, per_example_argmax = softmax.XentLossFromLogits(
          softmax.theta, logits, class_weights, class_ids, class_probabilities)
      self.assertAllEqual([8], py_utils.GetShape(per_example_loss))
      self.assertAllClose(expected_loss,
                          sess.run(tf.reduce_sum(per_example_loss)))
      self.assertAllEqual([8], py_utils.GetShape(per_example_argmax))


class FeedForwardNetTest(test_utils.TestCase):

  def testFeedForwardNetConstruction(self):
    with self.session(use_gpu=False):
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=True,
          activation='TANH',
          params_init=py_utils.WeightInit.Uniform(1.0))
      p.dropout.keep_prob = 0.5
      proj_l = p.Instantiate()
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)
      # check output_dim equals last hidden layer dim.
      self.assertEqual(p.hidden_layer_dims[-1], proj_l.output_dim)

      p = layers.FeedForwardNet.Params().Set(
          name='ffn2',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=True,
          activation='TANH',
          params_init=py_utils.WeightInit.Uniform(1.0))
      p.dropout = [
          layers.DropoutLayer.Params().Set(keep_prob=0.5),
          layers.DropoutLayer.Params().Set(keep_prob=0.9)
      ]
      proj_l = p.Instantiate()
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)

      p = layers.FeedForwardNet.Params().Set(
          name='ffn3',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=[True, False],
          activation=['TANH', 'RELU'],
          params_init=py_utils.WeightInit.Uniform(1.0))
      p.dropout = [
          layers.DropoutLayer.Params().Set(keep_prob=0.5),
          layers.DropoutLayer.Params().Set(keep_prob=0.9)
      ]
      proj_l = p.Instantiate()
      a = tf.constant(1.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a)

  def testFeedForwardNet(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=False,
          activation=['RELU', 'NONE'])
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.Instantiate()

      p1 = layers.ProjectionLayer.Params().Set(
          name='p1',
          input_dim=10,
          output_dim=20,
          activation='RELU',
          batch_norm=False)
      p1.params_init = params_init
      p1_l = p1.Instantiate()

      p2 = layers.ProjectionLayer.Params().Set(
          name='p2',
          input_dim=20,
          output_dim=30,
          activation='NONE',
          batch_norm=False)
      p2.params_init = params_init
      p2_l = p2.Instantiate()

      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out1 = feedforward_net.FPropAllLayers(feedforward_net.theta, a)

      out2 = [a, p1_l.FPropDefaultTheta(a)]
      out2.append(p2_l.FPropDefaultTheta(out2[-1]))

      self.evaluate(tf.global_variables_initializer())
      out1_v, out2_v = self.evaluate([out1, out2])
      self.assertAllClose(out1_v, out2_v)

  def testFeedForwardNetQuantized(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      np.random.seed(12345)

      cc_schedule = quant_utils.FakeQuantizationSchedule.Params().Set(
          clip_start_step=1,
          clip_end_step=2,
          quant_start_step=2,
          start_cap=8.0,
          end_cap=2.0)
      proj_qdomain = quant_utils.SymmetricScheduledClipQDomain.Params().Set(
          cc_schedule=cc_schedule)

      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=False,
          activation=['RELU', 'NONE'])
      p.qdomain.default = proj_qdomain.Copy()
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.Instantiate()

      p1 = layers.ProjectionLayer.Params().Set(
          name='p1',
          input_dim=10,
          output_dim=20,
          activation='RELU',
          batch_norm=False)
      p1.qdomain.default = proj_qdomain.Copy()
      p1.params_init = params_init
      p1_l = p1.Instantiate()

      p2 = layers.ProjectionLayer.Params().Set(
          name='p2',
          input_dim=20,
          output_dim=30,
          activation='NONE',
          batch_norm=False)
      p2.params_init = params_init
      p2.qdomain.default = proj_qdomain.Copy()
      p2_l = p2.Instantiate()

      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out1 = feedforward_net.FPropDefaultTheta(a)
      out2 = p2_l.FPropDefaultTheta(p1_l.FPropDefaultTheta(a))

      self.evaluate(tf.global_variables_initializer())

      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 5))
      out1_v, out2_v = self.evaluate([out1, out2])
      self.assertAllClose(out1_v, out2_v)

  def testFeedForwardNetBnFolded(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          batch_norm=True,
          bn_fold_weights=True,
          activation=['RELU', 'NONE'])
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.Instantiate()

      p1 = layers.ProjectionLayer.Params().Set(
          name='p1',
          input_dim=10,
          output_dim=20,
          activation='RELU',
          batch_norm=True,
          bn_fold_weights=True)
      p1.params_init = params_init
      p1_l = p1.Instantiate()

      p2 = layers.ProjectionLayer.Params().Set(
          name='p2',
          input_dim=20,
          output_dim=30,
          activation='NONE',
          batch_norm=True,
          bn_fold_weights=True)
      p2.params_init = params_init
      p2_l = p2.Instantiate()

      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out1 = feedforward_net.FPropDefaultTheta(a)

      out2 = p2_l.FPropDefaultTheta(p1_l.FPropDefaultTheta(a))

      self.evaluate(tf.global_variables_initializer())
      out1_v, out2_v = self.evaluate([out1, out2])
      self.assertAllClose(out1_v, out2_v)

  def testFeedForwardNetSmokeTest(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.FeedForwardNet.Params().Set(
          name='ffn',
          input_dim=10,
          hidden_layer_dims=[20, 30],
          activation=['RELU', 'NONE'])
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      feedforward_net = p.Instantiate()
      a = tf.constant(np.random.rand(5, 10), dtype=tf.float32)
      out = tf.reduce_sum(feedforward_net.FPropDefaultTheta(a))
      out_abs = tf.reduce_sum(tf.abs(feedforward_net.FPropDefaultTheta(a)))

      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 8.190775, self.evaluate(out), atol=1e-5)  # pylint: disable=line-too-long
      test_utils.CompareToGoldenSingleFloat(self, 36.773586, self.evaluate(out_abs))  # pylint: disable=line-too-long

  def testDropoutLayerTrain(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      p = layers.DropoutLayer.Params()
      p.keep_prob = 0.5
      p.random_seed = 1234
      p.name = 'dropout'

      dl = p.Instantiate()

      x = tf.random.normal([10, 10, 10, 3])
      xd = dl.FPropDefaultTheta(x)
      x, xd = self.evaluate([x, xd])
      self.assertGreater((xd == 0).mean(), 0.3)
      self.assertLess((xd == 0).mean(), 0.7)
      self.assertAllClose(xd[xd != 0], x[xd != 0] / p.keep_prob)

  def testDropoutLayerEval(self):
    with self.session(use_gpu=True), self.SetEval(True):
      tf.random.set_seed(3980847392)
      p = layers.DropoutLayer.Params()
      p.keep_prob = 0.5
      p.random_seed = 1234
      p.name = 'dropout'
      dl = p.Instantiate()

      x = tf.random.normal([10, 10, 10, 3])
      xd = dl.FPropDefaultTheta(x)

      x, xd = self.evaluate([x, xd])

      self.assertAllEqual(xd, x)

  def testDeterministicSerialize(self):
    p = layers.FeedForwardNet.Params().Set(
        input_dim=4,
        projection=layers.ProjectionLayer.Params().Set(
            has_bias=True,
            params_init=py_utils.WeightInit.KaimingUniformFanInRelu()),
        activation='TANH',
        hidden_layer_dims=[5, 5, 1],
        batch_norm=True,
        weight_norm=False)
    base_serialized = p.ToTextWithTypes()
    for _ in range(10):
      serialized = p.ToTextWithTypes()
      serialized_copy = p.Copy().ToTextWithTypes()
      self.assertEqual(serialized, base_serialized)
      self.assertEqual(serialized_copy, base_serialized)
      for x in [serialized, serialized_copy]:
        deserialized = layers.FeedForwardNet.Params()
        deserialized.FromTextWithTypes(x)
        self.assertEqual(p, deserialized)

  def testFeedForwardNetMeta(self):
    p = layers.FeedForwardNet.Params().Set(
        name='ffn',
        input_dim=10,
        hidden_layer_dims=[20, 30],
        activation=['RELU', 'NONE'])
    meta = p.cls.FPropMeta(p, tshape.Shape([5, 10]))
    self.assertEqual(
        meta.flops,
        # Last layer has no activation fns but need to add bias.
        5 * 2 * (10 * 20 + 20 * 30) + 5 * 20 + (20 + 30))
    self.assertEqual(meta.out_shapes[0].ToTensorShape().as_list(),
                     [5, p.hidden_layer_dims[-1]])


class AddingAccumulatorTest(test_utils.TestCase):
  """Test for AddingAccumulator."""

  def testAddingAccumulator(self):
    with self.session():
      layer_p = layers.IdentityLayer.Params()
      layer_p.name = 'test'
      layer = layer_p.Instantiate()

      layer.RegisterAccumulator('acc1', layers.AddingAccumulator([],
                                                                 tf.float32))

      # Initial value.
      self.assertEqual(0.0, self.evaluate(layer.accumulators.acc1.GetValue()))

      # Update/merge.
      layer.accumulators.acc1.Update(1.0)
      layer.accumulators.acc1.Update(1.0)
      self.assertEqual(2.0, self.evaluate(layer.accumulators.acc1.GetValue()))

      # Reset.
      layer.accumulators.Transform(lambda acc: acc.Reset())
      self.assertEqual(0.0, self.evaluate(layer.accumulators.acc1.GetValue()))


class BatchNormLayerNoPaddingTest(test_utils.TestCase, parameterized.TestCase):

  def testBatchNormLayerNoPaddingConstruction(self):
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayerNoPadding.Params()
    params.name = 'bn'
    params.dim = 2
    params.params_init = py_utils.WeightInit.Gaussian(0.1)
    layers.BatchNormLayerNoPadding(params)
    bn_vars = tf.get_collection('BatchNormLayerNoPadding_vars')
    bn_var_names = [x.name for x in bn_vars]
    expected_var_names = [
        'bn/beta/var:0', 'bn/gamma/var:0', 'bn/moving_mean/var:0',
        'bn/moving_variance/var:0'
    ]
    self.assertEqual(expected_var_names, bn_var_names)

  @parameterized.named_parameters({
      'testcase_name': '_eval',
      'is_eval': True,
  }, {
      'testcase_name': '_train',
      'is_eval': False,
  })
  def testBatchNormLayerNoPaddingFProp(self, is_eval):
    with self.session(use_gpu=True), self.SetEval(is_eval):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.BatchNormLayerNoPadding.Params()
      params.name = 'bn'
      params.dim = 3
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      bn_layer = layers.BatchNormLayerNoPadding(params)
      bn_in1 = tf.constant(
          np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

      bn_out = bn_layer.FPropDefaultTheta(bn_in1)
      sig1 = tf.reduce_sum(bn_out)
      sig2 = tf.reduce_sum(bn_out * bn_out)
      expected_sig1 = 2.6593573 if is_eval else 0
      expected_sig2 = 15.4642076 if is_eval else 47.850193

      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(expected_sig1, self.evaluate(sig1), atol=1e-5)
      self.assertAllClose(expected_sig2, self.evaluate(sig2), atol=1e-5)

  def testBatchNormLayerNoPaddingFPropUseGlobalStatsForTraining(self):
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayerNoPadding.Params()
    params.name = 'bn'
    params.dim = 3
    params.params_init = py_utils.WeightInit.Gaussian(0.1)

    bn_layer = layers.BatchNormLayerNoPadding(params)
    bn_in1 = tf.constant(
        np.random.normal(0.1, 0.5, [2, 8, 3]), dtype=tf.float32)

    bn_out = bn_layer.FPropDefaultTheta(bn_in1)
    sig1 = tf.reduce_sum(bn_out)
    sig2 = tf.reduce_sum(bn_out * bn_out)
    with self.session(use_gpu=True):
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(1.19209289551e-06, self.evaluate(sig1), atol=1e-5)
      self.assertAllClose(47.8501930237, self.evaluate(sig2), atol=1e-5)

  def testBatchNormLayerNoPaddingPostTrainingStepUpdate(self):
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayerNoPadding.Params()
    params.name = 'bn'
    params.dim = 2
    params.params_init = py_utils.WeightInit.Gaussian(0.1)

    bn_layer = layers.BatchNormLayerNoPadding(params)
    bn_layer.accumulators.counts.Update(0.0)
    bn_layer.accumulators.mean_ss.Update([1.0, 1.0])
    bn_layer.accumulators.variance_ss.Update([5.0, 5.0])
    with py_utils.GlobalStepContext(tf.constant(100)):
      bn_updates = bn_layer.PostTrainingStepUpdate()

    with self.session(use_gpu=True):
      self.evaluate(tf.global_variables_initializer())
      self.evaluate(bn_updates)
      moving_mean = self.evaluate(bn_layer.vars.moving_mean)
      moving_std = self.evaluate(bn_layer.vars.moving_variance)
      self.assertAllClose([0.0, 0.0], moving_mean)
      self.assertAllClose([1.0, 1.0], moving_std)

  def testBatchNormLayerNoPaddingFPropForConv(self):
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    params = layers.BatchNormLayerNoPadding.Params()
    params.name = 'bn_conv'
    params.dim = 32
    params.params_init = py_utils.WeightInit.Gaussian(0.1)

    bn_layer = layers.BatchNormLayerNoPadding(params)
    bn_in1 = tf.constant(
        np.random.normal(0.1, 0.5, [2, 8, 4, 32]), dtype=tf.float32)

    bn_out = bn_layer.FPropDefaultTheta(bn_in1)
    sig1 = tf.reduce_sum(bn_out)
    sig2 = tf.reduce_sum(bn_out * bn_out)
    with self.session(use_gpu=True):
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(0.0, self.evaluate(sig1), atol=1e-4)
      self.assertAllClose(2039.398681, self.evaluate(sig2))

  def _BuildDummyStackedBNLayer(self, splits):
    num_micro_batches = 8
    if splits == 0:
      endpoint = layers.BatchNormLayerNoPadding.Params().Set(
          decay=0.997, name='bn', dim=1)
    else:
      cell_tpl = []
      for split in range(splits):
        nets_to_split = [
            layers.BatchNormLayerNoPadding.Params().Set(
                decay=0.997, name='bn_{}'.format(split), dim=1),
        ]
        split_layer = gpipe.FeatureExtractionLayer.Params().Set(
            name='split_{}'.format(split), sub=nets_to_split)
        cell_tpl.append(split_layer)
      endpoint = gpipe.PipeliningLayer.Params().Set(
          name='pipeline',
          num_micro_batches=num_micro_batches,
          cell_tpl=cell_tpl,
          before_tpl=[])
    layer = endpoint.Instantiate()
    return layer

  @parameterized.named_parameters({
      'testcase_name': '_baseline',
      'splits': 0,
  }, {
      'testcase_name': '_two_splits',
      'splits': 2,
  }, {
      'testcase_name': '_four_splits',
      'splits': 4,
  })
  def testBatchNormLayerNoPaddingAccumulators(self, splits):
    batch_size = 1024
    with self.session(graph=tf.Graph()):
      # Construct a network where loss = w * x + b
      inputs = tf.concat([
          tf.ones([batch_size // 2, 1, 1, 1]),
          tf.zeros([batch_size // 2, 1, 1, 1])
      ],
                         axis=0)
      net = self._BuildDummyStackedBNLayer(splits)
      logits = net.FPropDefaultTheta(inputs)
      loss = tf.reduce_mean(logits)
      grads = tf.gradients(loss, tf.trainable_variables())
      # Check the accumulator values
      counts = []
      means = []
      variances = []
      for i in range(splits):
        l = net.children['split_{}'.format(i)].children['bn_{}'.format(i)]
        counts.append(l.accumulators.counts.GetValue())
        means.append(l.accumulators.mean_ss.GetValue())
        variances.append(l.accumulators.variance_ss.GetValue())
      if splits == 0:
        counts.append(net.accumulators.counts.GetValue())
        means.append(net.accumulators.mean_ss.GetValue())
        variances.append(net.accumulators.variance_ss.GetValue())
      post_training_step_updates = net.PostTrainingStepUpdate()

      self.evaluate(tf.global_variables_initializer())
      _, count_vals, mean_vals, var_vals = self.evaluate(
          [grads, counts, means, variances])

      self.assertSameElements(count_vals, {batch_size})

      self.assertEqual(batch_size // 2, mean_vals[0])
      if len(mean_vals) > 1:
        self.assertSameElements(mean_vals[1:], {0})

      self.assertEqual(batch_size // 2, var_vals[0])
      if len(var_vals) > 1:
        self.assertSameElements(var_vals[1:], {0})
      self.evaluate(post_training_step_updates)
      moving_vars = self.evaluate(tf.get_collection('moving_vars'))
    self.assertEqual(0.0015, moving_vars[0])
    self.assertNear(0.997750, moving_vars[1], err=1.0e-6)


class LayerNormTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'gpu',
      },
      {
          'testcase_name': 'nogpu',
          'use_gpu': False
      },
      {
          'testcase_name': 'fused_ln',
          'use_gpu': False,
          'use_fused_layernorm': True,
          'atol': 5e-5,
      },
      {
          'testcase_name': 'no_defun',
          'use_gpu': False,
          'use_defun': False,
      },
  )
  def testLayerNormFProp(self,
                         use_gpu=True,
                         use_fused_layernorm=False,
                         use_defun=True,
                         atol=None):
    with self.session(use_gpu=use_gpu):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.LayerNorm.Params()
      p.name = 'ln'
      p.input_dim = 3
      p.use_fused_layernorm = use_fused_layernorm
      p.use_defun = use_defun
      layer_norm = layers.LayerNorm(p)
      npy_input = np.random.normal(1.0, 0.5,
                                   [2, 4, 4, p.input_dim]).astype('float32')
      inputs = tf.constant(npy_input, dtype=tf.float32)
      output = layer_norm.FPropDefaultTheta(inputs)

      self.evaluate(tf.global_variables_initializer())
      sym_output = self.evaluate(output)

      # Mean should be zero and variance should be close to one.
      self.assertNear(0.0, sym_output.sum(), 1e-5)
      self.assertNear(1.0, np.var(sym_output), 1e-4)

      # Compare with numpy.
      mean = npy_input.mean(-1, keepdims=True)
      variance = np.mean(np.square(npy_input - mean), -1, keepdims=True)
      npy_output = (npy_input - mean) / np.sqrt(variance + p.epsilon)
      if atol is None:
        self.assertAllClose(sym_output, npy_output)
      else:
        self.assertAllClose(sym_output, npy_output, atol=atol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'F32Input',
          'input_dtype': tf.float32,
      },
      {
          'testcase_name': 'BF16Input',
          'input_dtype': tf.bfloat16,
      },
  )
  def testLayerNormDtypes(self, input_dtype):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.LayerNorm.Params()
      p.name = 'ln'
      p.random_seed = 123
      p.input_dim = 3
      layer_norm = layers.LayerNorm(p)
      npy_input = np.random.normal(1.0, 0.5,
                                   [2, 4, 4, p.input_dim]).astype('float32')
      inputs = tf.constant(npy_input, dtype=input_dtype)
      output = layer_norm.FPropDefaultTheta(inputs)

      self.evaluate(tf.global_variables_initializer())
      output = self.evaluate(output)

      # Mean should be zero and variance should be close to one.
      self.assertNear(0.0, output.sum(), 1e-5)
      self.assertNear(1.0, np.var(output), 1e-4)

  def testLayerNormFPropDirectScale(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.LayerNorm.Params()
      p.name = 'ln'
      p.input_dim = 3
      p.direct_scale = True
      layer_norm = layers.LayerNorm(p)
      npy_input = np.random.normal(1.0, 0.5,
                                   [2, 4, 4, p.input_dim]).astype('float32')
      inputs = tf.constant(npy_input, dtype=tf.float32)
      output = layer_norm.FPropDefaultTheta(inputs)

      self.evaluate(tf.global_variables_initializer())
      sym_output = self.evaluate(output)

      # Mean should be zero and variance should be close to one.
      self.assertNear(0.0, sym_output.sum(), 1e-5)
      self.assertNear(1.0, np.var(sym_output), 1e-4)

      # Compare with numpy.
      mean = npy_input.mean(-1, keepdims=True)
      variance = np.mean(np.square(npy_input - mean), -1, keepdims=True)
      npy_output = (npy_input - mean) / np.sqrt(variance + p.epsilon)
      self.assertAllClose(sym_output, npy_output)

  def testLayerNormFPropNoCenter(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.LayerNorm.Params()
      p.name = 'ln'
      p.input_dim = 3
      p.center = False
      layer_norm = p.Instantiate()
      npy_input = np.random.normal(1.0, 0.5,
                                   [2, 4, 4, p.input_dim]).astype('float32')
      inputs = tf.constant(npy_input, dtype=tf.float32)
      output = layer_norm.FPropDefaultTheta(inputs)

      self.evaluate(tf.global_variables_initializer())
      sym_output = self.evaluate(output)

      # Mean should be non-zero.
      self.assertNotAlmostEqual(0.0, sym_output.sum())
      # Mean of squared should be close to one.
      self.assertNear(1.0, np.mean(sym_output**2), 1e-4)

  def testLayerNormFPropNoCenterFused(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.LayerNorm.Params()
      p.name = 'ln'
      p.input_dim = 3
      p.center = False
      p.use_fused_layernorm = True
      layer_norm = p.Instantiate()
      npy_input = np.random.normal(1.0, 0.5,
                                   [2, 4, 4, p.input_dim]).astype('float32')
      inputs = tf.constant(npy_input, dtype=tf.float32)
      with self.assertRaisesRegex(ValueError, 'does not support center=false'):
        layer_norm.FPropDefaultTheta(inputs)

  @parameterized.named_parameters(('_3D', 3), ('_4D', 4))
  def testReshapedLayerNorm(self, rank):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.ReshapedLayerNorm.Params()
      p.name = 'reshaped_layer_norm'
      p.random_seed = 123
      p.device_mesh = np.reshape(np.arange(4), [2, 2])
      p.input_dim = 6
      if rank == 3:
        dims = [p.input_dim]
      else:
        self.assertEqual(rank, 4)
        dims = [2, p.input_dim // 2]
      l = p.Instantiate()
      shape = [2, 4] + dims
      npy_input = np.random.normal(1.0, 0.5, shape).astype(np.float32)
      inputs = tf.constant(npy_input, dtype=tf.float32)
      output = l.FPropDefaultTheta(inputs)

      self.evaluate(tf.global_variables_initializer())
      output = self.evaluate(output)

      self.assertEqual(npy_input.shape, output.shape)
      # Mean should be zero and variance should be close to one.
      self.assertNear(0.0, output.sum(), 1e-5)
      self.assertNear(1.0, np.var(output), 1e-4)

  def testLayerNormBProp(self):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.LayerNorm.Params()
      p.name = 'ln'
      p.input_dim = 3
      layer_norm = layers.LayerNorm(p)

      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, p.input_dim]), dtype=tf.float32)
      output = layer_norm.FPropDefaultTheta(inputs)
      loss = tf.reduce_sum(output)

      all_vars = tf.trainable_variables()
      self.assertLen(all_vars, 2)

      grads = tf.gradients(loss, all_vars)
      self.evaluate(tf.global_variables_initializer())
      sym_grads = [self.evaluate(sg) for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-02, atol=1e-02)


class CategoricalLayerNormTest(test_utils.TestCase):

  def testCategoricalLayerNormFProp(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.CategoricalLayerNorm.Params()
      p.name = 'cat_ln'
      p.input_dim = 3
      p.num_classes = 2
      layer_norm = layers.CategoricalLayerNorm(p)
      npy_input = np.random.normal(1.0, 0.5,
                                   [2, 4, 4, p.input_dim]).astype('float32')

      inputs = tf.constant(npy_input, dtype=tf.float32)
      output = layer_norm.FPropDefaultTheta(inputs)

      self.evaluate(tf.global_variables_initializer())
      # Set different bias and scale for different copy of ln params
      self.evaluate(tf.assign(layer_norm.vars.scale_0, [0.0] * 3))
      self.evaluate(tf.assign(layer_norm.vars.scale_1, [1.0] * 3))
      self.evaluate(tf.assign(layer_norm.vars.bias_0, [0.0] * 3))
      self.evaluate(tf.assign(layer_norm.vars.bias_1, [1.0] * 3))

      output = layer_norm.FPropDefaultTheta(inputs)
      sym_output_c1 = self.evaluate(output)

      # Redefine output to use the value of new theta
      theta = layer_norm.theta
      theta.class_index = tf.constant(1, dtype=tf.int32)
      output = layer_norm.FProp(theta, inputs)
      sym_output_c2 = self.evaluate(output)

      # Mean should be zero and variance should be close to one.
      self.assertNotAllClose(sym_output_c1, sym_output_c2)
      self.assertNear(0.0, sym_output_c1.mean(), 1e-5)
      self.assertNear(1.0, np.var(sym_output_c1), 1e-4)
      # Mean should be 1 and variance should be close to 2^2
      self.assertNear(1.0, sym_output_c2.mean(), 1e-5)
      self.assertNear(4.0, np.var(sym_output_c2), 1e-4)

      # Compare with numpy.
      mean = npy_input.mean(-1, keepdims=True)
      variance = np.mean(np.square(npy_input - mean), -1, keepdims=True)
      npy_output = (npy_input - mean) / np.sqrt(variance + p.epsilon)
      self.assertAllClose(sym_output_c1, npy_output)

      variance = np.mean(np.square(npy_input - mean), -1, keepdims=True)
      npy_output = 2 * (npy_input - mean) / np.sqrt(variance + p.epsilon) + 1
      self.assertAllClose(sym_output_c2, npy_output)

  def testCategoricalLayerNormBProp(self):
    with self.session(use_gpu=True) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.CategoricalLayerNorm.Params()
      p.name = 'cat_ln'
      p.input_dim = 3
      p.num_classes = 2
      layer_norm = layers.CategoricalLayerNorm(p)

      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, p.input_dim]), dtype=tf.float32)
      output = layer_norm.FPropDefaultTheta(inputs)
      loss = tf.reduce_sum(output)

      all_vars = tf.trainable_variables()
      self.assertEqual(4, len(all_vars))
      grads = tf.gradients(loss, all_vars)
      self.evaluate(tf.global_variables_initializer())
      print('grads = {}'.format(grads))
      sym_grads = [self.evaluate(sg) for sg in grads]
      num_grads = [
          test_utils.ComputeNumericGradient(sess, loss, v) for v in all_vars
      ]

      for sg, ng in zip(sym_grads, num_grads):
        self.assertAllClose(sg, ng, rtol=1e-02, atol=1e-02)


class DeterministicDropoutTest(test_utils.TestCase, parameterized.TestCase):

  def testDeterministicDropoutLayer(self):

    with self.session(graph=tf.Graph()):
      tf.random.set_seed(12345)
      params = layers.DeterministicDropoutLayer.Params().Set(
          name='drop', keep_prob=0.7)
      dropout = params.Instantiate()
      x = tf.ones([4, 6], dtype=tf.float32)

      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 1234))
      step_seed = self.evaluate(py_utils.GetStepSeed())
      x_val = self.evaluate(dropout.FPropDefaultTheta(x))
      x_golden = x_val
      self.assertEqual(step_seed + 1, self.evaluate(py_utils.GetStepSeed()))

      # Check that x_val is either 0 or 1.0/0.7.
      self.assertTrue(
          np.all(
              np.logical_or(
                  np.isclose(x_val, 0.0), np.isclose(x_val, 1.0 / 0.7))))
      # Check that values contain 0 but are not all 0.
      self.assertTrue(
          0 < np.sum(np.cast[np.int32](np.isclose(x_val, 0.0))) < x_val.size)

      # Different step seed gives different result.
      x_val = self.evaluate(dropout.FPropDefaultTheta(x))
      self.assertNotAllClose(x_golden, x_val)

    with self.session(graph=tf.Graph()):
      tf.random.set_seed(12345)
      params = layers.DeterministicDropoutLayer.Params().Set(
          name='drop', keep_prob=0.7)
      dropout = params.Instantiate()
      x = tf.ones([4, 6], dtype=tf.float32)

      # Different global step gives different result
      self.assertEqual(step_seed, self.evaluate(py_utils.GetStepSeed()))
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 1235))
      x_val = self.evaluate(dropout.FPropDefaultTheta(x))
      self.assertNotAllClose(x_golden, x_val)

    # The same seeds is consistent.
    with self.session(graph=tf.Graph()):
      tf.random.set_seed(12345)
      params = layers.DeterministicDropoutLayer.Params().Set(
          name='drop', keep_prob=0.7)
      dropout = params.Instantiate()
      x = tf.ones([4, 6], dtype=tf.float32)

      self.assertEqual(step_seed, self.evaluate(py_utils.GetStepSeed()))
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 1234))
      x_val = self.evaluate(dropout.FPropDefaultTheta(x))
      self.assertAllClose(x_golden, x_val)

  def testNoiseShapeBroadcastDims(self):
    params = layers.DeterministicDropoutLayer.Params().Set(
        name='drop', keep_prob=0.7, noise_shape_broadcast_dims=[-1])
    dropout = params.Instantiate()

    x = tf.ones([6, 6])

    with self.session():
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 1234))
      self.assertEqual(1234, self.evaluate(py_utils.GetGlobalStep()))
      step_seed = self.evaluate(py_utils.GetStepSeed())
      x_val = self.evaluate(dropout.FPropDefaultTheta(x))
      self.assertEqual(step_seed + 1, self.evaluate(py_utils.GetStepSeed()))

    # Check that x_val is either 0 or 1.0/0.7.
    self.assertTrue(
        np.all(
            np.logical_or(np.isclose(x_val, 0.0), np.isclose(x_val,
                                                             1.0 / 0.7))))
    # Check that values contain 0 but are not all 0.
    self.assertTrue(
        0 < np.sum(np.cast[np.int32](np.isclose(x_val, 0.0))) < x_val.size)
    # Check that each row has the same value.
    self.assertAllClose(np.broadcast_to(x_val[:, :1], x_val.shape), x_val)

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
    """Test to verify the drop mask used in fprop and bprop is identical."""
    assert splits in [1, 2, 4]
    with self.session():
      tf.random.set_seed(12345)
      num_layers = 4
      # Build a model with 4 dropout layers.
      blocks = []
      for l in range(num_layers):
        blocks.append(layers.DeterministicDropoutLayer.Params().Set(
            name='dropout_{}'.format(l), keep_prob=0.7))
      # Divide the model into splits partitions.
      cell_tpl = []
      blocks_per_split = num_layers // splits
      for i in range(splits):
        sub = blocks[i * blocks_per_split:(i + 1) * blocks_per_split]
        cell_tpl.append(gpipe.FeatureExtractionLayer.Params().Set(
            name='cell_{}'.format(i), sub=sub))
      # Parallelize partitions using pipeline.
      p = gpipe.PipeliningLayer.Params().Set(
          name='pipeline',
          num_micro_batches=num_micro_batches,
          cell_tpl=cell_tpl)
      # Fake input
      x = tf.ones([2, 3])
      # Construct weights.
      w = tf.get_variable(
          'w', shape=[2, 3], initializer=tf.constant_initializer([[1] * 3] * 2))
      mdl = p.Instantiate()
      y = mdl.FPropDefaultTheta(x * w)
      # Construct loss function such that gradients = final activation.
      loss = tf.reduce_sum(y)
      grads = py_utils.ComputeGradients(loss, py_utils.NestedMap(w=w))
      self.evaluate(tf.global_variables_initializer())
      y_val = self.evaluate(y)
      grads_val = self.evaluate(grads.w.grad)
      self.assertAllClose(y_val, grads_val)


class GradNormTrackerTest(test_utils.TestCase):

  def testGradNormTracker(self):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.GradNormTracker.Params().Set(
          name='grad_norm_tracker', clip_threshold=3.0)
      grad_norm_tracker = p.Instantiate()
      grad_norm = tf.placeholder(tf.float32)
      grad_norm_clip = grad_norm_tracker.FPropDefaultTheta(grad_norm)

      self.evaluate(tf.global_variables_initializer())

      random_normal = np.exp(np.random.normal(5.0, 1.0, size=10000))
      # We are expected to reject 16% of the outliers.
      outliers = np.exp(np.random.normal(7.0, 1.0, size=100))
      total_rejections = 0
      for i in range(100):
        for j in range(100):
          sess.run([grad_norm_clip], {grad_norm: random_normal[i * 100 + j]})
        clip = sess.run([grad_norm_clip], {grad_norm: outliers[i]})[0]
        if clip == 0.0:
          total_rejections += 1
      # Q(yonghui): Why is total_rejections not deterministic?
      print('total_rejections', total_rejections)
      self.assertGreater(total_rejections, 5)

  def testGradNormTrackerClipCapMin(self):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.GradNormTracker.Params().Set(
          name='grad_norm_tracker',
          clip_threshold=3.0,
          grad_norm_clip_cap_min=math.exp(10.0))
      grad_norm_tracker = p.Instantiate()
      grad_norm = tf.placeholder(tf.float32)
      grad_norm_clip = grad_norm_tracker.FPropDefaultTheta(grad_norm)

      self.evaluate(tf.global_variables_initializer())

      random_normal = np.exp(np.random.normal(5.0, 1.0, size=10000))
      # We expect no outliers being rejected due to the grad_norm_clip_cap_min.
      outliers = np.exp(np.random.normal(7.0, 1.0, size=100))
      total_rejections = 0
      for i in range(100):
        for j in range(100):
          sess.run([grad_norm_clip], {grad_norm: random_normal[i * 100 + j]})
        clip = sess.run([grad_norm_clip], {grad_norm: outliers[i]})[0]
        if clip == 0.0:
          total_rejections += 1
      print('total_rejections', total_rejections)
      self.assertEqual(total_rejections, 0)

  def testGradNormTrackerHasNan(self):
    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.GradNormTracker.Params().Set(
          name='grad_norm_tracker', clip_threshold=3.0)
      grad_norm_tracker = p.Instantiate()
      grad_norm = tf.placeholder(tf.float32)
      has_nan = tf.cast(tf.ones([]), dtype=tf.bool)
      grad_norm_clip = grad_norm_tracker.FPropDefaultTheta(grad_norm, has_nan)

      self.evaluate(tf.global_variables_initializer())

      random_normal = np.exp(np.random.normal(5.0, 1.0, size=10000))
      outliers = np.exp(np.random.normal(7.0, 1.0, size=100))
      total_rejections = 0
      for i in range(100):
        for j in range(100):
          sess.run([grad_norm_clip], {grad_norm: random_normal[i * 100 + j]})
        clip = sess.run([grad_norm_clip], {grad_norm: outliers[i]})[0]
        if clip == 0.0:
          total_rejections += 1
      self.assertEqual(total_rejections, 100)


class HighwaySkipLayerTest(test_utils.TestCase):

  def testHighwaySkipLayerConstruction(self):
    with self.session(use_gpu=False):
      p = layers.HighwaySkipLayer.Params().Set(
          name='gffn',
          input_dim=10,
          carry_bias_init=1.0,
          couple_carry_transform_gates=True,
          batch_norm=False,
          params_init=py_utils.WeightInit.Uniform(1.0))
      proj_l = p.Instantiate()
      a = tf.constant(1.0, shape=[20, 10])
      b = tf.constant(-2.0, shape=[20, 10])
      proj_l.FPropDefaultTheta(a, b)

  def testHighwaySkipLayerCarryGate(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.HighwaySkipLayer.Params().Set(
          name='gffn',
          input_dim=10,
          carry_bias_init=1000.0,
          couple_carry_transform_gates=True,
          batch_norm=False,
          params_init=py_utils.WeightInit.Uniform(1.0))
      proj_l = p.Instantiate()
      a = tf.constant(1.0, shape=[20, 10])
      b = tf.constant(-2.0, shape=[20, 10])
      out = proj_l.FPropDefaultTheta(a, b)
      self.evaluate(tf.global_variables_initializer())
      a, out = self.evaluate([a, out])
      self.assertAllClose(a, out)


class GatingLayerTest(test_utils.TestCase):

  def testGatingLayerConstruction(self):
    with self.session(use_gpu=False):
      p = layers.GatingLayer.Params().Set(
          name='gating',
          input_dim=10,
          carry_bias_init=1.0,
          params_init=py_utils.WeightInit.Uniform(1.0))
      gate = p.Instantiate()
      a = tf.constant(1.0, shape=[20, 10])
      b = tf.constant(-2.0, shape=[20, 10])
      gate.FPropDefaultTheta(a, b)

  def testGatingLayerFProp(self):
    with self.session(use_gpu=True):
      p = layers.GatingLayer.Params().Set(
          name='gate', input_dim=6, has_bias=False)
      gate = p.Instantiate()
      a = tf.constant(np.random.uniform(size=[10, 6]), dtype=tf.float32)
      b = tf.constant(np.random.uniform(size=[10, 6]), dtype=tf.float32)
      out = gate.FPropDefaultTheta(a, b)
      self.evaluate(tf.global_variables_initializer())
      actual_out, w = self.evaluate([out, gate.theta.carry_gate.w])
      self.assertAllEqual(np.shape(w), [12, 6])
      w = np.matmul(self.evaluate(a), w[:6, :]) + np.matmul(
          self.evaluate(b), w[6:, :])
      sigmoid_w = 1 / (1 + np.exp(-w))
      expected_out = a * sigmoid_w + b * (1 - sigmoid_w)
      self.assertAllClose(actual_out, expected_out)

  def testGatingLayerFPropSaturated(self):
    with self.session(use_gpu=True):
      p = layers.GatingLayer.Params().Set(
          name='gate', input_dim=6, has_bias=True, carry_bias_init=100)
      gate = p.Instantiate()
      a = tf.constant(np.random.uniform(size=[10, 6]), dtype=tf.float32)
      b = tf.constant(np.random.uniform(size=[10, 6]), dtype=tf.float32)
      out = gate.FPropDefaultTheta(a, b)
      self.evaluate(tf.global_variables_initializer())
      # High initial bias, causing the carry gate to saturate and the
      # output will be very close to a.
      self.assertAllClose(self.evaluate(out), self.evaluate(a))


class UniformLabelSmootherTest(test_utils.TestCase):

  def testUniformLabelSmoother(self):
    with self.session(use_gpu=False):
      params = layers.UniformLabelSmoother.Params()
      params.name = 'uls'
      params.num_classes = 5
      params.uncertainty = 0.1

      smooth_layer = layers.UniformLabelSmoother(params)
      target_labels = tf.constant([[0, 1, 2, 3, 3, 3, 4]], dtype=tf.int32)
      target_ids = tf.constant([[0, 0, 1, 2, 3, 3, 3]], dtype=tf.int32)
      target_paddings = tf.zeros(tf.shape(target_ids))
      output = smooth_layer.FPropDefaultTheta(target_paddings, target_labels,
                                              target_ids)
      self.evaluate(tf.global_variables_initializer())
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [[
          [0.89999998,  0.025     ,  0.025     ,  0.025     ,  0.025     ],
          [0.025     ,  0.89999998,  0.025     ,  0.025     ,  0.025     ],
          [0.025     ,  0.025     ,  0.89999998,  0.025     ,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.89999998,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.89999998,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.89999998,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.025     ,  0.89999998]
      ]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      output_v = self.evaluate(output)
      self.assertAllClose(expected_output, output_v, atol=1e-2, rtol=1e-2)
      self.assertAllClose(np.ones(output_v.shape[:-1]), output_v.sum(-1))

  def testUniformLabelSmootherLargerToken(self):
    with self.session(use_gpu=False):
      params = layers.UniformLabelSmoother.Params()
      params.name = 'uls'
      params.num_classes = 5
      params.uncertainty = 0.1
      params.uncertainty_larger = 0.2
      params.token_id_uncertainty_larger = 4

      smooth_layer = layers.UniformLabelSmoother(params)
      target_labels = tf.constant([[0, 1, 2, 3, 3, 3, 3]], dtype=tf.int32)
      target_ids = tf.constant([[0, 0, 1, 2, 4, 4, 4]], dtype=tf.int32)
      target_paddings = tf.zeros(tf.shape(target_ids))
      output = smooth_layer.FPropDefaultTheta(target_paddings, target_labels,
                                              target_ids)
      self.evaluate(tf.global_variables_initializer())
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [[
          [0.89999998,  0.025     ,  0.025     ,  0.025     ,  0.025     ],
          [0.025     ,  0.89999998,  0.025     ,  0.025     ,  0.025     ],
          [0.025     ,  0.025     ,  0.89999998,  0.025     ,  0.025     ],
          [0.025     ,  0.025     ,  0.025     ,  0.89999998,  0.025     ],
          [0.05      ,  0.05      ,  0.05      ,  0.80000001,  0.05      ],
          [0.05      ,  0.05      ,  0.05      ,  0.80000001,  0.05      ],
          [0.05      ,  0.05      ,  0.05      ,  0.80000001,  0.05      ]
      ]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      output_v = self.evaluate(output)
      self.assertAllClose(expected_output, output_v, atol=1e-2, rtol=1e-2)
      self.assertAllClose(np.ones(output_v.shape[:-1]), output_v.sum(-1))


class WeightedSumLayerTest(test_utils.TestCase):

  def testWeightedSumLayer(self):
    with self.session(use_gpu=True):
      np.random.seed(505837249)
      depth = 4
      batch = 2
      n_sources = 3
      ctxs = [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
              [[3.0, 4.0, 5.0, 6.0], [6.0, 7.0, 8.0, 9.0]],
              [[4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 1.0, 2.0]]]
      p = layers.WeightedSumLayer.Params()
      p.name = 'transparent_layer'
      p.num_sources = n_sources
      p.random_seed = 505837249
      merger = p.Instantiate()

      ctxs = [tf.expand_dims(i, 2) for i in ctxs]
      ctx = tf.squeeze(merger.FProp(merger.theta, ctxs), 2)
      self.evaluate(tf.global_variables_initializer())
      actual_ctx = self.evaluate(ctx)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [[ 2.66666675,  3.66666675,  4.66666698,  5.66666698],
                      [ 5.0,         6.0,         4.33333349,  5.33333349]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)

  def testWeightedSumLayerGlobalWeightAndMinimalProb(self):
    with self.session(use_gpu=True):
      np.random.seed(505837249)
      depth = 4
      batch = 2
      n_sources = 3
      ctxs = [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
              [[3.0, 4.0, 5.0, 6.0], [6.0, 7.0, 8.0, 9.0]],
              [[4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 1.0, 2.0]]]
      p = layers.WeightedSumLayer.Params()
      p.name = 'transparent_layer'
      p.num_sources = n_sources
      p.random_seed = 505837249
      p.minimal_prob = 0.01
      p.global_weight_scale = 10.0
      merger = p.Instantiate()

      ctxs = [tf.expand_dims(i, 2) for i in ctxs]
      ctx = tf.squeeze(merger.FProp(merger.theta, ctxs), 2)
      self.evaluate(tf.global_variables_initializer())
      actual_ctx = self.evaluate(ctx)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_ctx = [[ 2.66666675,  3.66666675,  4.66666698,  5.66666698],
                      [ 5.0,         6.0,         4.33333349,  5.33333349]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_ctx.shape, (batch, depth))
      self.assertAllClose(expected_ctx, actual_ctx, rtol=1e-05, atol=1e-05)


class DeconvLayerTest(test_utils.TestCase):

  def testDeconvLayerFProp(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      params = layers.DeconvLayer.Params()
      params.name = 'deconv'
      params.filter_shape = [3, 3, 2, 8]
      params.filter_stride = [2, 2]
      params.params_init = py_utils.WeightInit.Gaussian(0.1)

      conv_layer = params.Instantiate()
      inputs = tf.constant(
          np.random.normal(0.1, 0.5, [2, 4, 4, 8]), dtype=tf.float32)

      out = conv_layer.FPropDefaultTheta(inputs)
      out_shape = conv_layer.OutShape(tf.shape(inputs))

      self.evaluate(tf.global_variables_initializer())
      out_v, shape_v = self.evaluate([out, out_shape])
      self.assertAllEqual(shape_v, [2, 8, 8, 2])
      self.assertAllEqual(out_v.shape, shape_v)

      summary = np.sum(np.square(out_v), axis=(1, 2, 3))
      tf.logging.info('testDeconvLaye rFProp actual = %s',
                      np.array_repr(summary))
      self.assertAllClose([4.77159977, 5.47860432], summary)


class GatedAverageLayerTest(test_utils.TestCase):

  def testGatedAverageLayer(self):
    with self.session(use_gpu=True):
      np.random.seed(505837249)
      depth = 4
      batch = 2
      num_inputs = 3

      inp_1 = np.asarray([[0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      inp_2 = np.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      inp_3 = np.asarray([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]],
                         dtype=np.float32)
      p = layers.GatedAverageLayer.Params()
      p.name = 'gated_avg_layer'
      p.num_inputs = num_inputs
      p.num_nodes = depth
      p.random_seed = 505837249
      g_avg = p.Instantiate()

      avg = g_avg.FProp(g_avg.theta, [inp_1, inp_2, inp_3])
      self.evaluate(tf.global_variables_initializer())
      actual_avg = self.evaluate(avg)

      expected_avg = [[0.092766, 0.092766, 0.092766, 0.092766],
                      [-1., -1., 1., 1.]]
      self.assertEqual(actual_avg.shape, (batch, depth))
      self.assertAllClose(expected_avg, actual_avg, rtol=1e-05, atol=1e-05)


class LHUCLayerTest(test_utils.TestCase):

  def testLHUCLayer(self):
    with self.session(use_gpu=True):
      np.random.seed(505837249)
      depth = 4
      batch = 2

      inp = np.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
                       dtype=np.float32)
      p = layers.LHUCLayer.Params()
      p.name = 'lhuc_layer'
      p.input_dim = depth
      p.random_seed = 505837249
      lhuc = p.Instantiate()

      lhuc = lhuc.FProp(lhuc.theta, inp)
      self.evaluate(tf.global_variables_initializer())
      actual_avg = self.evaluate(lhuc)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_avg = [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_avg.shape, (batch, depth))
      self.assertAllClose(expected_avg, actual_avg, rtol=1e-05, atol=1e-05)


class ResidualAdapterLayerTest(test_utils.TestCase):

  def testResidualAdapterLayer(self):
    with self.session(use_gpu=True):
      np.random.seed(505837249)
      depth = 4
      batch = 2

      inp = np.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
                       dtype=np.float32)
      p = layers.ResidualAdapterLayer.Params()
      p.name = 'resadap_layer'
      p.input_dim = depth
      p.bottleneck_dim = 2
      p.random_seed = 505837249
      resadap = p.Instantiate()

      resadap = resadap.FProp(resadap.theta, inp)
      self.evaluate(tf.global_variables_initializer())
      actual_avg = self.evaluate(resadap)

      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_avg = [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      self.assertEqual(actual_avg.shape, (batch, depth))
      self.assertAllClose(expected_avg, actual_avg, rtol=1e-05, atol=1e-05)


class GluLayerTest(test_utils.TestCase):

  def testGlu(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers.GluLayer.Params()
      p.name = 'glu_layers'
      p.input_dim = 3
      glu_layer = layers.GluLayer(p)

      h = glu_layer.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[-0.627102  , -0.61167496,  0.86975265],
           [ 0.26028383,  1.2781785 ,  3.2620153 ]],
          [[-1.0547485 , -0.8914927 ,  1.221562  ],
           [ 1.5992979 ,  0.59807897,  1.1403995 ]],
          [[ 0.32643905,  0.2473597 , -1.6090155 ],
           [ 0.7710849 ,  1.0708375 , -0.7306549 ]],
          [[ 0.57057625,  0.7954664 , -1.1305015 ],
           [-0.9524991 , -0.75179183, -0.25037393]],
          [[-0.38881764,  0.584502  , -1.7211256 ],
           [-1.4992303 , -0.23748302,  1.5525813 ]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)

  def testGluWithoutResidual(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(3980847392)
      inputs = tf.random.normal([5, 2, 3], seed=948387483)
      paddings = tf.zeros([5, 2])
      p = layers.GluLayer.Params()
      p.name = 'glu_layers'
      p.input_dim = 3
      p.output_dim = 4
      p.apply_residual = False
      glu_layer = layers.GluLayer(p)

      h = glu_layer.FPropDefaultTheta(inputs, paddings)
      self.evaluate(tf.global_variables_initializer())
      actual_layer_output = self.evaluate(h)
      # pylint: disable=bad-whitespace
      # pyformat: disable
      expected_output = [
          [[ 0.2498899 , -0.11702066,  0.6268383 , -0.45065   ],
           [ 0.341157  , -0.36893576,  0.3802086 , -0.5135959 ]],
          [[ 0.3014423 , -0.19727892,  0.59274423, -0.5131605 ],
           [-0.35804138,  0.35897657,  0.29084033,  0.03678071]],
          [[-0.6129734 ,  0.7878639 , -0.52119696,  0.38839644],
           [-0.38819426,  0.44012898, -0.8382209 ,  0.41553053]],
          [[-0.49943203,  0.61838603, -0.71468747,  0.41521466],
           [ 0.3411708 , -0.550296  ,  0.0372162 , -0.39770594]],
          [[-0.01931547, -0.19615713, -0.9648195 ,  0.281362  ],
           [ 0.34413674, -0.41213694,  0.30943182, -0.49557427]]]
      # pyformat: enable
      # pylint: enable=bad-whitespace
      print(np.array_repr(actual_layer_output))
      self.assertAllClose(actual_layer_output, expected_output)


class MultitaskAdapterLayerTest(test_utils.TestCase, parameterized.TestCase):

  def _MultitaskAdapterParams(self, data_format='TBC'):
    return layers.MultitaskAdapterLayer.Params().Set(
        name='multi_adapter',
        input_dim=4,
        bottleneck_dim=2,
        num_tasks=3,
        data_format=data_format,
        random_seed=505837249)

  @parameterized.parameters('TBC', 'BTC')
  def testSingleStepFProp(self, data_format):
    with self.session(use_gpu=True):
      np.random.seed(1234567)
      # Inputs are of shape [1, batch, input_dim] (single time step)
      # Batch elements 0, 2, and 3 are identical, but 0 and 2 have the same
      # task ID where as 3 has a different task ID.
      inputs = tf.constant([[[0.5, 0.3, -0.2, 0.0], [0.0, 0.7, -1.0, 2.0],
                             [0.5, 0.3, -0.2, 0.0], [0.5, 0.3, -0.2, 0.0]]],
                           dtype=tf.float32)
      tasks = tf.constant([1, 0, 1, 0], dtype=tf.int32)
      if data_format == 'BTC':
        inputs = tf.transpose(inputs, [1, 0, 2])
      p = self._MultitaskAdapterParams(data_format)
      adapter = p.Instantiate()
      output = adapter.FProp(adapter.theta, inputs, tasks)
      self.evaluate(tf.global_variables_initializer())
      actual = self.evaluate(output)
      if data_format == 'BTC':
        actual = tf.transpose(actual, [1, 0, 2])
      tf.logging.info('testSingleStepFProp actual=%r' % actual)
      expected = [[[1.1579462, 1.2241995, -0.6177901, 0.23089096],
                   [0.05022227, 0.9056754, -0.3771479, 2.1245508],
                   [1.1579462, 1.2241995, -0.6177901, 0.23089096],
                   [0.55389655, 0.5207228, 0.46842042, 0.13366304]]]
      self.assertEqual(actual.shape, (1, 4, 4))
      # Batch elements 0 and 2 are equal because they had the same input
      # and the same task ID.
      self.assertAllClose(actual[0][0], actual[0][2], rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected, actual, rtol=1e-05, atol=1e-05)

  @parameterized.parameters('TBC', 'BTC')
  def testMultiStepFProp(self, data_format):
    with self.session(use_gpu=True):
      np.random.seed(1234567)
      # Inputs are same as above but of shape [time, batch, input_dim]
      inputs = tf.constant([[[0.5, 0.3, -0.2, 0.0], [0.0, 0.7, -1.0, 2.0]],
                            [[0.5, 0.3, -0.2, 0.0], [0.5, 0.3, -0.2, 0.0]]],
                           dtype=tf.float32)
      # tasks is of shape [batch] indicating one task for each sequence.
      tasks = tf.constant([1, 0], dtype=tf.int32)
      if data_format == 'BTC':
        inputs = tf.transpose(inputs, [1, 0, 2])
      p = self._MultitaskAdapterParams(data_format)
      adapter = p.Instantiate()
      output = adapter.FProp(adapter.theta, inputs, tasks)
      self.evaluate(tf.global_variables_initializer())
      actual = self.evaluate(output)
      if data_format == 'BTC':
        actual = tf.transpose(actual, [1, 0, 2])
      tf.logging.info('testMultiStepFProp actual=%r' % actual)
      # Output is same as above but with shape same as input.
      expected = [[[1.1579462, 1.2241995, -0.6177901, 0.23089096],
                   [0.05022227, 0.9056754, -0.3771479, 2.1245508]],
                  [[1.1579462, 1.2241995, -0.6177901, 0.23089096],
                   [0.55389655, 0.5207228, 0.46842045, 0.13366304]]]
      self.assertEqual(actual.shape, (2, 2, 4))
      self.assertAllClose(expected, actual, rtol=1e-05, atol=1e-05)

  @parameterized.parameters('TBC', 'BTC')
  def testSpecifyTaskPerTimestepFProp(self, data_format):
    with self.session(use_gpu=True):
      np.random.seed(1234567)
      inputs = tf.constant([[[0.5, 0.3, -0.2, 0.0], [0.0, 0.7, -1.0, 2.0]],
                            [[0.5, 0.3, -0.2, 0.0], [0.5, 0.3, -0.2, 0.0]]],
                           dtype=tf.float32)
      # tasks are same as above but of shape [time, batch] indicating that
      # we should look up adapter params per timestep.  In this example we
      # still have the task ID consistent across timesteps in order to
      # replicate the previous test's output.
      tasks = tf.constant([[1, 0], [1, 0]], dtype=tf.int32)
      if data_format == 'BTC':
        inputs = tf.transpose(inputs, [1, 0, 2])
        tasks = tf.transpose(tasks, [1, 0])
      p = self._MultitaskAdapterParams(data_format)
      adapter = p.Instantiate()
      output = adapter.FProp(adapter.theta, inputs, tasks)
      self.evaluate(tf.global_variables_initializer())
      actual = self.evaluate(output)
      if data_format == 'BTC':
        actual = tf.transpose(actual, [1, 0, 2])
      tf.logging.info('testSpecifyTaskPerTimestepFProp actual=%r' % actual)
      # Output is same as above.
      expected = [[[1.1579462, 1.2241995, -0.6177901, 0.23089096],
                   [0.05022227, 0.9056754, -0.3771479, 2.1245508]],
                  [[1.1579462, 1.2241995, -0.6177901, 0.23089096],
                   [0.55389655, 0.5207228, 0.46842042, 0.13366304]]]
      self.assertEqual(actual.shape, (2, 2, 4))
      self.assertAllClose(expected, actual, rtol=1e-05, atol=1e-05)

  @parameterized.parameters('TBC', 'BTC')
  def testDifferentTaskPerTimestepFProp(self, data_format):
    with self.session(use_gpu=True):
      np.random.seed(1234567)
      inputs = tf.constant([[[0.5, 0.3, -0.2, 0.0], [0.0, 0.7, -1.0, 2.0]],
                            [[0.5, 0.3, -0.2, 0.0], [0.5, 0.3, -0.2, 0.0]]],
                           dtype=tf.float32)
      # tasks are again of shape [time, batch] but with different tasks
      # for each timestep.
      tasks = tf.constant([[1, 0], [2, 1]], dtype=tf.int32)
      if data_format == 'BTC':
        inputs = tf.transpose(inputs, [1, 0, 2])
        tasks = tf.transpose(tasks, [1, 0])
      p = self._MultitaskAdapterParams(data_format)
      adapter = p.Instantiate()
      output = adapter.FProp(adapter.theta, inputs, tasks)
      self.evaluate(tf.global_variables_initializer())
      actual = self.evaluate(output)
      if data_format == 'BTC':
        actual = tf.transpose(actual, [1, 0, 2])
      tf.logging.info('testDifferentTaskPerTimestepFProp actual=%r' % actual)
      expected = [[[1.1579462, 1.2241995, -0.6177901, 0.23089096],
                   [0.05022227, 0.9056754, -0.3771479, 2.1245508]],
                  [[0.6961179, 0.06690431, -0.08757646, 0.40129724],
                   [1.1579462, 1.2241995, -0.6177901, 0.23089096]]]
      self.assertEqual(actual.shape, (2, 2, 4))
      self.assertAllClose(expected, actual, rtol=1e-05, atol=1e-05)

  @parameterized.parameters('TBC', 'BTC')
  def testWithClipping(self, data_format):
    with self.session(use_gpu=True):
      np.random.seed(1234567)
      # Inputs are of shape [1, batch, input_dim] (single time step)
      # Batch elements 0, 2, and 3 are identical, but 0 and 2 have the same
      # task ID where as 3 has a different task ID.
      inputs = tf.constant([[[0.5, 0.3, -0.2, 0.0], [0.0, 0.7, -1.0, 2.0],
                             [0.5, 0.3, -0.2, 0.0], [0.5, 0.3, -0.2, 0.0]]],
                           dtype=tf.float32)
      tasks = tf.constant([1, 0, 1, 0], dtype=tf.int32)
      if data_format == 'BTC':
        inputs = tf.transpose(inputs, [1, 0, 2])
      p = self._MultitaskAdapterParams(data_format)
      p.clip_task_ids = True
      p.num_tasks = 1
      adapter = p.Instantiate()
      output = adapter.FProp(adapter.theta, inputs, tasks)
      self.evaluate(tf.global_variables_initializer())
      actual = self.evaluate(output)
      if data_format == 'BTC':
        actual = tf.transpose(actual, [1, 0, 2])
      tf.logging.info('testWithClipping actual=%r' % actual)
      expected = [[[0.573645, 0.578965, 0.592506, 0.151613],
                   [0.059424, 0.943358, -0.263031, 2.147371],
                   [0.573645, 0.578965, 0.592506, 0.151613],
                   [0.573645, 0.578965, 0.592506, 0.151613]]]
      self.assertEqual(actual.shape, (1, 4, 4))
      # Batch elements 0 and 2 are equal because they had the same input
      # and the same task ID.
      self.assertAllClose(actual[0][0], actual[0][2], rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected, actual, rtol=1e-05, atol=1e-05)

  def testGradientChecker(self):
    with self.session(use_gpu=True):
      np.random.seed(1234567)
      inputs = tf.constant([[[0.5, 0.3, -0.2, 0.0], [0.0, 0.7, -1.0, 2.0]],
                            [[0.5, 0.3, -0.2, 0.0], [0.5, 0.3, -0.2, 0.0]]],
                           dtype=tf.float32)
      tasks = tf.constant([1, 0], dtype=tf.int32)
      p = self._MultitaskAdapterParams()
      adapter = p.Instantiate()
      output = adapter.FProp(adapter.theta, inputs, tasks)
      loss = tf.reduce_sum(output)
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      self.evaluate(tf.global_variables_initializer())

      def DenseGrad(var, grad):
        if isinstance(grad, tf.Tensor):
          return grad
        elif isinstance(grad, tf.IndexedSlices):
          return tf.math.unsorted_segment_sum(grad.values, grad.indices,
                                              tf.shape(var)[0])

      dense_grads = [DenseGrad(x, y) for (x, y) in zip(all_vars, grads)]
      dense_grad_sums = [tf.reduce_sum(g) for g in dense_grads]
      grad_vs = self.evaluate(dense_grad_sums)
      self.assertAllClose([
          -5.364418e-07, 3.405262e+00, 1.252710e+01, 1.600000e+01, 1.335246e+00,
          2.513876e-01
      ],
                          grad_vs,
                          rtol=1e-05,
                          atol=1e-05)

  @parameterized.parameters((None, 3.675607), (tf.bfloat16, 3.671875))
  def testEinsumLayer(self, fprop_dtype, expected_sum):
    with self.session(use_gpu=True):
      np.random.seed(1234567)
      # Inputs are of shape [batch, 1, input_dim] (single time step)
      # Batch elements 0, 2, and 3 are identical, but 0 and 2 have the same
      # task ID where as 3 has a different task ID.
      inputs = tf.constant([[[0.5, 0.3, -0.2, 0.0]], [[0.0, 0.7, -1.0, 2.0]],
                            [[0.5, 0.3, -0.2, 0.0]], [[0.5, 0.3, -0.2, 0.0]]],
                           dtype=tf.float32)
      tasks = tf.constant([1, 0, 1, 0], dtype=tf.int32)
      p = layers.MultitaskAdapterEinsumLayer.Params().Set(
          name='multi_adapter',
          input_dim=4,
          bottleneck_dim=2,
          num_tasks=3,
          fprop_dtype=fprop_dtype,
          random_seed=505837249)
      adapter = p.Instantiate()
      output = adapter.FProp(adapter.theta, inputs, tasks)
      self.evaluate(tf.global_variables_initializer())
      actual, actual_sum = self.evaluate((output, tf.reduce_sum(output)))
      tf.logging.info('testSingleStepFProp actual=%r' % actual)
      self.assertEqual(actual.shape, (4, 1, 4))
      # Batch elements 0 and 2 are equal because they had the same input
      # and the same task ID.
      self.assertAllClose(actual[0][0], actual[2][0], rtol=1e-05, atol=1e-05)
      self.assertAllClose(expected_sum, actual_sum, rtol=1e-05, atol=1e-05)


class CCTGatingNetworkTest(test_utils.TestCase):

  def testCCTGatingNetworkConstruction(self):
    with self.session(use_gpu=False):
      p = layers.CCTGatingNetwork.Params().Set(
          name='cct_gating',
          input_dim=10,
          hidden_layer_dim=20,
          num_outputs=3,
          noise_std=5.0,
          noise_warmup_steps=300)
      cct_l = p.Instantiate()
      a = tf.constant(1.0, shape=[20, 10])
      cct_l.FPropDefaultTheta(a)

  def testCCTGatingNetworkTraining(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.CCTGatingNetwork.Params().Set(
          name='cct_gating',
          input_dim=10,
          hidden_layer_dim=20,
          num_outputs=3,
          noise_std=5.0,
          noise_warmup_steps=300)
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      p.is_inference = False
      cct_net = p.Instantiate()

      a = tf.constant(np.random.rand(3, 10), dtype=tf.float32)
      out = cct_net.FPropDefaultTheta(a)

      self.evaluate(tf.global_variables_initializer())
      out_v = self.evaluate([out])
      self.assertAllClose(
          out_v,
          [[[0.412904, 0.520129, 0.694699], [0.395485, 0.47316, 0.632451],
            [0.404144, 0.502593, 0.644338]]])

  def testCCTGatingNetworkInference(self):
    with self.session(use_gpu=False), self.SetEval(True):
      tf.random.set_seed(398847392)
      np.random.seed(12345)
      p = layers.CCTGatingNetwork.Params().Set(
          name='cct_gating',
          input_dim=10,
          hidden_layer_dim=20,
          num_outputs=3,
          noise_std=5.0,
          noise_warmup_steps=300)
      params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=837465638)
      p.params_init = params_init
      cct_net = p.Instantiate()

      a = tf.constant(np.random.rand(3, 10), dtype=tf.float32)
      out = cct_net.FPropDefaultTheta(a)

      self.evaluate(tf.global_variables_initializer())
      out_v = self.evaluate([out])
      self.assertAllClose(out_v, [[[0., 1., 1.], [0., 0., 1.], [0., 1., 1.]]])


class CondScaleShiftFFNLayerTest(test_utils.TestCase):

  def testCondScaleShiftFFNLayerConstruction(self):
    with self.session(use_gpu=False):
      params = layers.CondScaleShiftFFNLayer.Params().Set(
          name='ss_ffn',
          input_dim=10,
          output_dim=7,
          scale_fn='NONE',
          shift_fn='NONE')
      params.ffn.hidden_layer_dims = [5, 5]
      layer_ss = params.Instantiate()
      time_c, batch_c, in_dim_c = 15, 2, 10
      a = tf.constant(1.0, shape=[time_c, batch_c, in_dim_c])
      layer_ss.FPropDefaultTheta(a)

  def testCondScaleShiftFFNLayerFprop(self):
    with self.session(use_gpu=False):
      time_c, batch_c, in_dim_c, out_dim_c = 15, 2, 10, 7
      params = layers.CondScaleShiftFFNLayer.Params().Set(
          name='ss_ffn',
          input_dim=in_dim_c,
          output_dim=out_dim_c,
          scale_fn='NONE',
          shift_fn='NONE')
      params.ffn.hidden_layer_dims = [5, 5]
      layer_ss = params.Instantiate()

      a = tf.constant(1.0, shape=[time_c, batch_c, in_dim_c])
      scale_out, shift_out = layer_ss.FPropDefaultTheta(a)
      self.evaluate(tf.global_variables_initializer())
      scale_out, shift_out = self.evaluate([scale_out, shift_out])
      self.assertEqual(scale_out.shape, (time_c, batch_c, out_dim_c))
      self.assertEqual(shift_out.shape, (time_c, batch_c, out_dim_c))


class IdentityLayerTest(test_utils.TestCase):

  def testIdentityLayerNestedMap(self):
    with self.session(use_gpu=False):
      p = layers.IdentityLayer.Params().Set(name='Nested')
      layer = p.Instantiate()
      a = tf.constant(1.0, shape=[20, 10])
      b = tf.constant(-2.0, shape=[20, 10])
      inputs = py_utils.NestedMap(a=a, b=b)
      outputs = layer.FPropDefaultTheta(inputs)
      self.assertAllEqual(self.evaluate(inputs.a), self.evaluate(outputs.a))
      self.assertAllEqual(self.evaluate(inputs.b), self.evaluate(outputs.b))
      a_copy = layer.FPropDefaultTheta(a)
      self.assertAllEqual(self.evaluate(a), self.evaluate(a_copy))


class SingleShardSharedEmbeddingSoftmaxLayerTest(test_utils.TestCase):

  def testSingleShardSharedEmbeddingSoftmaxLayer(self):
    with self.session(use_gpu=True):
      tf.random.set_seed(398847392)
      params = layers.SingleShardSharedEmbeddingSoftmax.Params()
      params.name = 'emb'
      params.dtype = tf.float32
      params.vocab_size = 128
      params.num_classes = 128
      params.embedding_dim = 8
      params.input_dim = 8
      params.emb_with_matmul = True
      params.params_init = py_utils.WeightInit.Gaussian(0.01)
      params.vn.global_vn = False
      params.vn.per_step_vn = False
      emb_layer = params.Instantiate()
      ids = tf.constant([[89], [100]])
      embs = emb_layer.EmbLookupDefaultTheta(ids)
      embs_sum = tf.reduce_sum(embs)
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, -0.031068, self.evaluate(embs_sum))  # pylint: disable=line-too-long


class StatisticalPoolingLayerTest(test_utils.TestCase):

  def testFProp(self):
    with self.session(use_gpu=False) as sess:
      # time = 5, batch = 4, depth = 2
      features = tf.constant(np.random.normal(size=(5, 4, 2)), dtype=tf.float32)
      paddings = tf.constant(
          [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0],
           [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
          dtype=tf.float32)
      features = tf.transpose(features, [1, 0, 2])
      paddings = tf.transpose(paddings, [1, 0])
      # test fprop with both mean & stddev
      params = layers.StatisticalPoolingLayer.Params()
      params.name = 'mean_stddev_pooling'
      params.has_stddev = True
      layer1 = layers.StatisticalPoolingLayer(params)
      results1 = layer1.FProp(features, paddings)
      # test fprop with only mean
      params.has_stddev = False
      params.name = 'mean_pooling'
      layer2 = layers.StatisticalPoolingLayer(params)
      results2 = layer2.FProp(features, paddings)
      # check the results
      tf.global_variables_initializer().run()
      results1, results2 = sess.run([results1, results2])
      self.assertEqual(results1.shape,
                       (features.shape[0], 2 * features.shape[2]))
      self.assertEqual(results2.shape, (features.shape[0], features.shape[2]))


class MaskedLmDataAugmenterTest(test_utils.TestCase):

  def testFProp(self):
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      np.random.seed(12345)
      tf.random.set_seed(12345)
      aug_p = layers.MaskedLmDataAugmenter.Params()
      aug_p.vocab_size = 128
      aug_p.mask_token_id = 0
      aug_p.name = 'lm_aug'
      aug_layer = aug_p.Instantiate()

      input_ids = tf.random.uniform([2, 100], 0, 128, tf.int32)
      paddings = tf.zeros([2, 100])

      auged_ids, mask_pos = aug_layer.FPropDefaultTheta(input_ids, paddings)
      v0, v1, v2 = sess.run([input_ids, auged_ids, mask_pos])
      self.assertAllEqual(v0 * (1.0 - v2), v1 * (1.0 - v2))
      tf.logging.info('orig_ids: %s', np.array_repr(v0))
      tf.logging.info('auged_ids: %s', np.array_repr(v1))
      tf.logging.info('masked_pos: %s', np.array_repr(v2))


if __name__ == '__main__':
  test_utils.main()
