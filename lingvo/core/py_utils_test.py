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
"""Tests for py_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf
from tensorflow.python.framework import function
from lingvo import model_registry
from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core import test_helper
from lingvo.tasks.image.params import mnist  # pylint: disable=unused-import

FLAGS = tf.flags.FLAGS


class PyUtilsTest(tf.test.TestCase):

  def testIsDefaultParamInit(self):
    p = py_utils.DefaultParamInit()
    self.assertTrue(py_utils.IsDefaultParamInit(p))

  def testCreateVariableBasics(self):
    with self.test_session(use_gpu=False, graph=tf.Graph()):
      methods = [
          py_utils.WeightInit.Gaussian,
          py_utils.WeightInit.Uniform,
          py_utils.WeightInit.Constant,
          py_utils.WeightInit.TruncatedGaussian,
          py_utils.WeightInit.GaussianSqrtDim,
          py_utils.WeightInit.UniformSqrtDim,
          py_utils.WeightInit.UniformUnitScaling,
          py_utils.WeightInit.TruncatedGaussianSqrtDim,
      ]
      dtypes = [tf.float32, tf.float64]
      shapes = [[], [3], [2, 4]]
      collections = ['col1', 'col2']

      all_vars = []
      for i, (m, dt, sp) in enumerate(
          itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt, collections)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc)[0])

      # To reuse existing variables
      tf.get_variable_scope().reuse_variables()

      self.assertEqual(len(tf.all_variables()), len(all_vars))

      all_vars_copy = []
      for i, (m, dt, sp) in enumerate(
          itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt, collections)
        all_vars_copy.append(py_utils.CreateVariable('var_%d' % i, pc)[0])

      tf.global_variables_initializer().run()
      for v1, v2 in zip(all_vars, all_vars_copy):
        v1_v = v1.eval()
        v2_v = v2.eval()
        self.assertAllEqual(v1_v, v2_v)

  def testCreateVariableUniform(self):
    with self.test_session(use_gpu=False, graph=tf.Graph()):
      tf.set_random_seed(12345678)
      methods = [
          py_utils.WeightInit.Uniform,
          py_utils.WeightInit.UniformSqrtDim,
          py_utils.WeightInit.UniformUnitScaling,
      ]
      dtypes = [tf.float32]
      shapes = [[2, 3]]
      all_vars = []
      for i, (m, dt, sp) in enumerate(
          itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(0.1), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc)[0])

      v1_v_expted = [[0.069674, -0.072278, -0.021777],
                     [-0.052155, -0.050274, 0.086218]]
      v2_v_expted = [[0.005361, 0.036109, -0.036575],
                     [0.058314, 0.031438, 0.049196]]

      tf.global_variables_initializer().run()
      v1_v = all_vars[0].eval()
      v2_v = all_vars[1].eval()
      self.assertAllClose(v1_v_expted, v1_v.tolist())
      self.assertAllClose(v2_v_expted, v2_v.tolist())

  def testCreateVariableNormal(self):
    with self.test_session(use_gpu=False, graph=tf.Graph()):
      tf.set_random_seed(832124)
      methods = [
          py_utils.WeightInit.Gaussian,
          py_utils.WeightInit.GaussianSqrtDim,
      ]
      dtypes = [tf.float32]
      shapes = [[2, 3]]
      all_vars = []
      for i, (m, dt, sp) in enumerate(
          itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc)[0])

      v1_v_expted = [[-1.472208, 0.960204, -0.192588],
                     [-0.461884, 1.018134, 0.063719]]
      v2_v_expted = [[-0.862255, -0.688153, 0.82515],
                     [-0.07671, 0.613031, -0.020327]]

      tf.global_variables_initializer().run()
      v1_v = all_vars[0].eval()
      v2_v = all_vars[1].eval()
      self.assertAllClose(v1_v_expted, v1_v.tolist())
      self.assertAllClose(v2_v_expted, v2_v.tolist())

  def testCreateVariableException(self):
    with self.test_session(use_gpu=False, graph=tf.Graph()):
      tf.set_random_seed(832124)
      pc = py_utils.WeightParams([2, 3], py_utils.WeightInit.Gaussian())
      var1 = py_utils.CreateVariable('var1', pc)[0]

      tf.get_variable_scope().reuse_variables()
      # Reuses an existing variable.
      var2 = py_utils.CreateVariable('var1', pc)[0]

      # An exception should be thrown in this case.
      pc.init.scale = 2.0
      with self.assertRaises(AssertionError):
        py_utils.CreateVariable('var1', pc)

      tf.global_variables_initializer().run()
      self.assertAllEqual(var1.eval(), var2.eval())

  def testCreateVariableDifferentSeed(self):
    with self.test_session(use_gpu=False) as sess:
      tf.set_random_seed(3251343)
      pc = py_utils.WeightParams([2, 3], py_utils.WeightInit.Gaussian())
      with tf.variable_scope('layer0'):
        w0, _ = py_utils.CreateVariable('w', pc)
      with tf.variable_scope('layer1'):
        w1, _ = py_utils.CreateVariable('w', pc)
      sess.run(tf.global_variables_initializer())

      # w0_val, w1_val should be sufficient different.
      w0_val, w1_val = sess.run([w0, w1])
      print(['diff = ', w0_val - w1_val])
      self.assertTrue(np.max(np.abs(w0_val - w1_val)) > 0.1)

  def testXavier(self):
    with self.test_session(use_gpu=False, graph=tf.Graph()):
      tf.set_random_seed(1618)
      methods = [py_utils.WeightInit.Xavier]
      dtypes = [tf.float32, tf.float16]
      shapes = [[2, 3]]
      all_vars = []
      for i, (m, dt, sp) in enumerate(
          itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc)[0])

      v1_v_expted = [[1.051236, -0.959198, 0.796091],
                     [-0.685691, 0.230933, -1.006293]]

      tf.global_variables_initializer().run()
      v1_v = all_vars[0].eval()
      self.assertAllClose(v1_v_expted, v1_v.tolist())

  def testXavier1D(self):
    with self.test_session(use_gpu=False, graph=tf.Graph()):
      tf.set_random_seed(1618)
      methods = [py_utils.WeightInit.Xavier]
      dtypes = [tf.float32, tf.float16]
      shapes = [[2]]
      all_vars = []
      for i, (m, dt, sp) in enumerate(
          itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc)[0])

      v1_v_expted = [1.175317, -1.072416]

      tf.global_variables_initializer().run()
      v1_v = all_vars[0].eval()
      self.assertAllClose(v1_v_expted, v1_v.tolist())

  def testXavier3D(self):
    with self.test_session(use_gpu=False, graph=tf.Graph()):
      tf.set_random_seed(1618)
      methods = [py_utils.WeightInit.Xavier]
      dtypes = [tf.float32, tf.float16]
      shapes = [[1, 1, 2]]
      all_vars = []
      for i, (m, dt, sp) in enumerate(
          itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc)[0])

      v1_v_expted = [[[1.357139, -1.23832]]]

      tf.global_variables_initializer().run()
      v1_v = all_vars[0].eval()
      self.assertAllClose(v1_v_expted, v1_v.tolist())

  def testCheckNumerics(self):
    xv = [[1, 2], [3, 4]]
    yv = [10] * 4
    with self.test_session() as sess:
      x = tf.constant(xv, tf.float32)
      y = tf.constant(yv)
      z = tf.reduce_mean(tf.constant([], tf.float32))
      self.assertAllClose(xv, sess.run(py_utils.CheckNumerics(x)))
      self.assertAllClose(yv, sess.run(py_utils.CheckNumerics(y)))
      actual_xv, actual_yv = sess.run(py_utils.CheckNumerics([x, y]))
      self.assertAllClose(xv, actual_xv)
      self.assertAllClose(yv, actual_yv)
      actual_xv, actual_yv = sess.run(py_utils.CheckNumerics((x, y)))
      self.assertAllClose(xv, actual_xv)
      self.assertAllClose(yv, actual_yv)

      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, 'NaN'):
        sess.run(py_utils.CheckNumerics(z))

  def testGetShape(self):
    a = tf.constant([1])
    self.assertEqual(py_utils.GetShape(a), [1])
    self.assertEqual(py_utils.GetShape(a, 1), [1])
    self.assertEqual(py_utils.GetShape(a, 3), [1])

    b = tf.constant([[1, 2]])
    self.assertEqual(py_utils.GetShape(b), [1, 2])
    self.assertEqual(py_utils.GetShape(b, 1), [1])
    self.assertEqual(py_utils.GetShape(b, 2), [1, 2])
    self.assertEqual(py_utils.GetShape(b, 3), [1, 2])

    c = tf.zeros([1, a[0], a.shape[0].value, tf.shape(a)[0]])
    self.assertEqual(py_utils.GetShape(c)[0], 1)
    self.assertEqual(py_utils.GetShape(c)[1], 1)
    self.assertEqual(py_utils.GetShape(c)[2], 1)
    self.assertEqual(py_utils.GetShape(c)[3], 1)

    d = tf.placeholder(tf.float32, shape=(1, None))
    self.assertEqual(py_utils.GetShape(d)[0], 1)
    self.assertIsInstance(py_utils.GetShape(d)[1], tf.Tensor)

    e = tf.zeros([d.shape[0].value, tf.shape(d)[0], tf.shape(d)[1]])
    self.assertEqual(py_utils.GetShape(e)[0], 1)
    self.assertIsInstance(py_utils.GetShape(e)[1], tf.Tensor)
    self.assertIsInstance(py_utils.GetShape(e)[2], tf.Tensor)

    @function.Defun(tf.float32)
    def Identity(x):
      return x

    f = Identity(e)
    # Function return value does not have shape info.
    self.assertIsNone(f.shape.ndims)
    # GetShape() will return a Tensor.
    self.assertIsInstance(py_utils.GetShape(f), tf.Tensor)

  def testRenamingRules(self):
    pc = py_utils.WeightParams([3, 3])
    with tf.variable_scope('model'):
      _, v1 = py_utils.CreateVariable('v1', pc)
      with py_utils.VariableRenameScope([('model/(.*)', 'data/%s')]):
        _, v2 = py_utils.CreateVariable('v2', pc)
      _, v3 = py_utils.CreateVariable('v3', pc)

    self.assertTrue(v1.name == 'model/v1/var:0')
    self.assertTrue(v2.name == 'data/v2/var:0')
    self.assertTrue(v3.name == 'model/v3/var:0')

  def testOpportunisticReuse(self):
    pc = py_utils.WeightParams([3, 3])
    _, v1 = py_utils.CreateVariable('v1', pc)
    with self.assertRaises(Exception):
      _ = py_utils.CreateVariable('v1', pc)
    with py_utils.OpportunisticVariableReuseScope(True):
      _, v2 = py_utils.CreateVariable('v1', pc)
      _, x1 = py_utils.CreateVariable('x1', pc)
      with py_utils.OpportunisticVariableReuseScope(False):
        with self.assertRaises(Exception):
          _ = py_utils.CreateVariable('v1', pc)
      _, v3 = py_utils.CreateVariable('v1', pc)
    with self.assertRaises(Exception):
      _ = py_utils.CreateVariable('v1', pc)

    for v in [v2, v3]:
      self.assertTrue(v1 is v)
    self.assertTrue(v1 is not x1)

  def testGetOrCreateGlobalStep(self):
    with tf.variable_scope('s1'):
      with tf.name_scope('s2'):
        gs1 = py_utils.GetOrCreateGlobalStep()
        gs2 = tf.train.get_global_step()
      gs3 = py_utils.GetOrCreateGlobalStep()
      gs4 = tf.train.get_global_step()
    gs5 = py_utils.GetOrCreateGlobalStep()
    gs6 = tf.train.get_global_step()
    for gs in [gs2, gs3, gs4, gs5, gs6]:
      self.assertTrue(gs1 is gs)
    self.assertEqual(gs1.name, 'global_step:0')

  def testCreateLocalTheta(self):
    methods = [py_utils.WeightInit.Gaussian, py_utils.WeightInit.Uniform]
    dtypes = [tf.float32]
    shapes = [[2, 4], [3]]

    test_vars = py_utils.NestedMap()
    for i, (m, dt, sp) in enumerate(itertools.product(methods, dtypes, shapes)):
      pc = py_utils.WeightParams(sp, m(), dt, 'col1')
      test_vars['var_%d' % i] = py_utils.CreateVariable('var_%d' % i, pc)[0]

    test_devices = [
        '/job:worker/replica:0/device:GPU:0',
        '/job:worker/replica:0/device:GPU:1'
    ]

    sharded_local_vars = py_utils.CreateLocalTheta(test_vars, test_devices)
    sharded_local_vars_list = sharded_local_vars.Flatten()

    # assert the name is now Identity*
    for v in sharded_local_vars_list:
      self.assertTrue('Identity' in v.name)

    # assert proper device placement
    for i, v in enumerate(sharded_local_vars_list):
      expected_device = test_devices[i % len(test_devices)]
      self.assertEqual(v.device, expected_device)

  def testComputeGradient(self):
    with self.test_session(use_gpu=False):
      a = tf.get_variable('a', [])
      b = tf.get_variable('b', [], trainable=False)
      c = tf.get_variable('c', [])
      e = tf.get_variable('e', [])
      l = a + b + tf.stop_gradient(c)
      vmap = py_utils.NestedMap(
          a=a, b=b, c=c, d=None, n=py_utils.NestedMap(aa=a, e=e))
      var_grads = py_utils.ComputeGradients(l, vmap)
      print('var_grads = ', var_grads.DebugString())
      # Only 'a' matters. b is not trainable; c has stop_gradient; d
      # is None; e is not computed by l and aa is a duplicated.
      self.assertEqual([_[0] for _ in var_grads.FlattenItems()], ['a'])
      self.assertEqual(var_grads.a[0].name, 'a:0')

  def testAdjustGradientsWithL2Loss(self):
    with self.test_session(use_gpu=False) as sess:
      emb = tf.get_variable(
          'emb',
          initializer=tf.constant(np.arange(100).reshape([10, 10]), tf.float32))
      act = tf.gather(emb, [2, 5, 2, 2, 5])
      weight = tf.get_variable(
          'w', initializer=tf.constant(np.ones([10, 1]), tf.float32))
      bias = tf.get_variable('b', initializer=tf.constant([0.217]))
      pred = tf.matmul(act, weight) + tf.stop_gradient(bias)
      loss = tf.reduce_sum(pred)
      vmap = py_utils.NestedMap(emb=emb, weight=weight, bias=bias)
      var_grads = py_utils.ComputeGradients(loss, vmap)
      self.assertEqual(sorted(var_grads.keys()), ['emb', 'weight'])
      l2_loss, var_grads_with_l2 = py_utils.AdjustGradientsWithL2Loss(
          var_grads, 0.1)

      sess.run(tf.global_variables_initializer())
      var_grads_vals, l2_loss_val, var_grads_with_l2_vals = sess.run(
          [var_grads, l2_loss, var_grads_with_l2])
      print('var_grads_vals = ', var_grads_vals)
      print('var_grads_with_l2_vals = ', var_grads_with_l2_vals)
      self.assertAllEqual(var_grads_vals.emb[0], var_grads_with_l2_vals.emb[0])
      self.assertAllEqual(var_grads_vals.weight[0],
                          var_grads_with_l2_vals.weight[0])
      self.assertAllEqual(
          l2_loss_val,
          0.5 * 0.1 * (np.sum(np.square(var_grads_vals.weight[0])) + np.sum(
              np.square(var_grads_vals.emb[0][2, :])) + np.sum(
                  np.square(var_grads_vals.emb[0][5, :]))))

      # With l2, gradients of emb and weight are adjusted.
      self.assertAllClose(
          var_grads_with_l2_vals.weight[1],
          var_grads_vals.weight[1] + 0.1 * var_grads_vals.weight[0])
      self.assertAllClose(var_grads_with_l2_vals.emb[1].indices,
                          var_grads_vals.emb[1].indices)
      self.assertAllClose(var_grads_with_l2_vals.emb[1].indices,
                          [2, 5, 2, 2, 5])
      self.assertAllClose(
          var_grads_with_l2_vals.emb[1].values, var_grads_vals.emb[1].values +
          0.1 * np.array([[1 / 3.], [1 / 2.], [1 / 3.], [1 / 3.], [1 / 2.]
                         ]) * var_grads_vals.emb[0][[2, 5, 2, 2, 5], :])

  def testFindNeeded(self):
    phs = [
        tf.placeholder('float32', shape=(), name='p%d' % (i + 1,))
        for i in range(4)
    ]
    p1, p2, p3, p4 = phs

    z1 = p1 + p2
    z2 = z1 * p3

    z1_needed = set(py_utils.FindNeededInList(phs, z1))
    z2_needed = set(py_utils.FindNeededInList(phs, [z2]))
    z2_p4_needed = set(py_utils.FindNeededInList(phs, [z2, p4]))

    self.assertTrue(set([p1, p2]) == z1_needed)
    self.assertTrue(set([p1, p2, p3]) == z2_needed)
    self.assertTrue(set([p1, p2, p3, p4]) == z2_p4_needed)

  def testStatsCounter(self):
    with self.test_session() as sess:
      foo = py_utils.StatsCounter('foo')
      val = foo.Value()
      params = base_layer.LayerBase.Params()
      inc = foo.IncBy(params, 100)

      tf.global_variables_initializer().run()
      self.assertAllEqual(0, val.eval())
      self.assertAllEqual(100, sess.run(inc))
      self.assertAllEqual(100, val.eval())
      self.assertAllEqual([100, 200], sess.run([val, inc]))
      self.assertAllEqual([200, 300], sess.run([val, inc]))

  def testModelSplit(self):
    with py_utils.ModelSplit(2):
      assert py_utils.GetModelSplit() == 2
      with py_utils.ModelSplit(3):
        assert py_utils.GetModelSplit() == 3
    assert py_utils.GetModelSplit() == 0

  def testPiecewiseConstant(self):
    boundaries = (1000, 2000, 3000)
    values = (1e-3, 2e-4, 3e-5, 4e-6)

    def _Eval(x):
      with self.test_session(use_gpu=False) as sess:
        result = py_utils.PiecewiseConstant(
            x, boundaries, values, vdtype=tf.float32)
        return sess.run(result)

    self.assertAlmostEqual(1e-3, _Eval(0))
    self.assertAlmostEqual(1e-3, _Eval(1000))
    self.assertAlmostEqual(2e-4, _Eval(1001))
    self.assertAlmostEqual(2e-4, _Eval(2000))
    self.assertAlmostEqual(3e-5, _Eval(3000))
    self.assertAlmostEqual(4e-6, _Eval(4000))


class WeightedAvgTest(tf.test.TestCase):

  def testWeightedAvg(self):
    with self.test_session(use_gpu=False) as sess:
      losses = tf.constant([5.6, 4.6, 1.5, 3.4])
      weights = tf.constant([10, 9, 2, 8])
      loss, weight = py_utils.WeightedAvg(losses, weights)
      expected = [4.4, 29]
      actual = sess.run([loss, weight])
      self.assertAllClose(actual, expected)

  def testWeightedAvgOfMetrics(self):
    with self.test_session(use_gpu=False) as sess:
      metrics = [{
          'a': (2.0, 0.5),
          'b': (5.0, 1.5)
      }, {
          'a': (9.0, 3.0),
          'b': (4.0, 0.5)
      }]
      expected = {'a': (8.0, 3.5), 'b': (4.75, 2.0)}
      weighted_avg = py_utils.WeightedAvgOfMetrics(metrics)
      actual = sess.run(weighted_avg)
      self.assertDictEqual(actual, expected)

  def testCombineMetrics(self):
    a = py_utils.NestedMap()
    a['a'] = (1, 1)
    a['loss'] = (100, 10)
    b = py_utils.NestedMap()
    b['b'] = (2, 2)
    b['loss'] = (50, 20)
    c = py_utils.NestedMap()
    c['loss'] = (60, 15)
    combined = py_utils.CombineMetrics([(a, 0.7), (b, 0.3), (c, 1.5)])
    self.assertEqual(combined['a'], (1, 1))
    self.assertEqual(combined['b'], (2, 2))
    total_loss = combined['loss'][0] * combined['loss'][1]
    self.assertEqual(total_loss, 100 * 10 * 0.7 + 50 * 20 * 0.3 + 60 * 15 * 1.5)

  def testCombineMetricsKeyNotInAllMetrics(self):
    a = py_utils.NestedMap()
    a['a'] = (1, 1)
    b = py_utils.NestedMap()
    b['b'] = (2, 2)
    b['loss'] = (50, 20)
    c = py_utils.NestedMap()
    c['loss'] = (60, 15)
    with self.assertRaises(ValueError):
      py_utils.CombineMetrics([(a, 0.7), (b, 0.3), (c, 1.5)])


class OverrideVarsFromCheckpointsTest(tf.test.TestCase):

  def _GetLeNetVarsFirstVal(self, sess):
    with tf.variable_scope('lenet5', reuse=True):
      conv0 = tf.get_variable('conv0/w/var')
      conv1 = tf.get_variable('conv1/w/var')
      fc_bias = tf.get_variable('fc/b/var')
    conv0_val, conv1_val, fc_bias_val = sess.run([conv0, conv1, fc_bias])
    return conv0_val[0][0][0][0], conv1_val[0][0][0][0], fc_bias_val[0]

  def testOverrideVarsFromCheckpoint(self):

    with self.test_session(use_gpu=False) as sess:
      tf.set_random_seed(8372749040)
      cfg = model_registry.GetParams('image.mnist.LeNet5', 'Train')
      with cluster_factory.ForTestingWorker(mode='sync', job='trainer_client'):
        cfg.cls(cfg)
      tf.global_variables_initializer().run()
      self.assertAllClose(
          # These are initialized values before overriding with checkpoint.
          self._GetLeNetVarsFirstVal(sess),
          [-0.005945, -0.036722, 0.0])
      checkpoint_path = test_helper.test_src_dir_path(
          'core/testdata/lenet_test_model')
      variable_loading_rules = [('lenet5/conv0/w/var', 'lenet5/conv0/w/var'),
                                ('lenet5/conv1/w/var', 'lenet5/conv1/w/var')]
      variable_ignore_rules = []
      py_utils._OverrideVarsFromCheckpoint(
          sess, tf.all_variables(), checkpoint_path, variable_loading_rules,
          variable_ignore_rules)
      self.assertAllClose(
          # Now conv weights have been overwritten but fc bias has not.
          self._GetLeNetVarsFirstVal(sess),
          [0.043092, -0.024082, 0.0])

  def testOverrideVarsFromCheckpointWithIgnoreRules(self):

    with self.test_session(use_gpu=False) as sess:
      tf.set_random_seed(8372749040)
      cfg = model_registry.GetParams('image.mnist.LeNet5', 'Train')
      with cluster_factory.ForTestingWorker(mode='sync', job='trainer_client'):
        cfg.cls(cfg)
      tf.global_variables_initializer().run()
      self.assertAllClose(
          # These are initialized values before overriding with checkpoint.
          self._GetLeNetVarsFirstVal(sess),
          [-0.005945, -0.036722, 0.0])
      checkpoint_path = test_helper.test_src_dir_path(
          'core/testdata/lenet_test_model')
      variable_loading_rules = [('lenet5/conv0/w/var', 'lenet5/conv0/w/var'),
                                ('lenet5/conv1/w/var', 'lenet5/conv1/w/var')]
      variable_ignore_rules = ['lenet5/conv1/w/var']
      py_utils._OverrideVarsFromCheckpoint(
          sess, tf.all_variables(), checkpoint_path, variable_loading_rules,
          variable_ignore_rules)
      self.assertAllClose(
          # Now only conv0 weights have been overridden.
          self._GetLeNetVarsFirstVal(sess),
          [0.043092, -0.036722, 0.0])


class NestedMapTest(tf.test.TestCase):

  def testBasic(self):
    x = py_utils.NestedMap()
    self.assertEqual(0, len(list(x.keys())))
    x['foo'] = 100
    self.assertEqual(100, x.foo)
    self.assertEqual(100, x['foo'])
    x.bar = py_utils.NestedMap({'baz': 200})
    self.assertEqual(200, x.bar.baz)
    self.assertFalse('flatten' in x)

  def testPrint(self):
    m = py_utils.NestedMap()
    m.foo = py_utils.NestedMap()
    m.foo.bar = 100
    m.x = py_utils.NestedMap()
    m.x.y = py_utils.NestedMap()
    m.x.y.z = 'abc'
    m.lst = [py_utils.NestedMap({'l': i}) for i in range(2)]
    # pyformat: disable
    self.assertEqual(m.DebugString(), '\n'.join([
        'foo.bar     100',
        'lst[0].l    0',
        'lst[1].l    1',
        'x.y.z       abc']))
    # pyformat: enable

  def testTransform(self):
    m = py_utils.NestedMap()
    m.foo = [1, 20, 32]
    m.bar = py_utils.NestedMap()
    m.bar.x = 100
    m.bar.y = [200, 201]
    m.z = (123, 321)
    n = m.Transform(lambda x: x if isinstance(x, tuple) else 1 + x)
    # pyformat: disable
    self.assertEqual(n.DebugString(), '\n'.join(
        ['bar.x       101',
         'bar.y[0]    201',
         'bar.y[1]    202',
         'foo[0]      2',
         'foo[1]      21',
         'foo[2]      33',
         'z           (123, 321)']))
    # pyformat: enable

  def testPack(self):
    m = py_utils.NestedMap()
    m.foo = [1, 20, 32]
    m.bar = py_utils.NestedMap()
    m.bar.x = 100
    m.bar.y = [200, 201]
    m.x = (123, 321)
    n = m.Pack(list(range(7)))
    # pyformat: disable
    self.assertEqual(n.DebugString(), '\n'.join(
        ['bar.x       0',
         'bar.y[0]    1',
         'bar.y[1]    2',
         'foo[0]      3',
         'foo[1]      4',
         'foo[2]      5',
         'x           6']))
    # pyformat: enable

  def testEmpty(self):
    m = py_utils.NestedMap()
    self.assertEqual(m.Flatten(), [])
    self.assertEqual(m.DebugString(), '')
    m1 = m.Pack([])
    self.assertEqual(m1.Flatten(), [])
    self.assertEqual(m1.DebugString(), '')

  def testIsCompatible(self):
    x = py_utils.NestedMap(
        a='a', b='b', c=py_utils.NestedMap(d='d', e=[1, 2, 4]))
    y = py_utils.NestedMap(a=1, b=2, c=py_utils.NestedMap(d=3, e=[10, 20, 30]))
    self.assertTrue(x.IsCompatible(y))
    z = py_utils.NestedMap(
        a=1, b=[10, 20, 30], c=py_utils.NestedMap(d=3, e=['x', 'y', 'z']))
    self.assertFalse(x.IsCompatible(z))

  def testFlattenItems(self):
    x = py_utils.NestedMap(
        a='a', b='b', c=py_utils.NestedMap(d='d', e=[1, 2, 4]))
    flat_x = x.FlattenItems()
    expected = [('a', 'a'), ('b', 'b'), ('c.d', 'd'), ('c.e', 1), ('c.e', 2),
                ('c.e', 4)]
    self.assertEqual(expected, flat_x)

  def testFilter(self):
    x = py_utils.NestedMap(
        a=100,
        b=200,
        c=300,
        d=py_utils.NestedMap(foo=38, bar=192, ok=[200, 300], ko=[10, 20]))
    y = x.Filter(lambda v: v > 150)
    self.assertEqual(y.FlattenItems(), [('b', 200), ('c', 300), ('d.bar', 192),
                                        ('d.ok', 200), ('d.ok', 300)])

  def testCopy(self):
    # This is not a copy.
    x = py_utils.NestedMap(
        a='a', b='b', c=py_utils.NestedMap(d='d', e=[1, 2, 4]))
    y = x
    y.a = 'y'
    self.assertEqual('y', y.a)
    self.assertEqual('y', x.a)

    # This is a (shallow) copy.
    x = py_utils.NestedMap(
        a='a', b='b', c=py_utils.NestedMap(d='d', e=[1, 2, 4]))
    y = py_utils.NestedMap(x)
    self.assertNotEqual(id(x), id(y))
    y.a = 'y'
    y.c.d = 'z'
    self.assertEqual('y', y.a)
    self.assertEqual('a', x.a)
    self.assertEqual('z', y.c.d)
    self.assertEqual('z', x.c.d)

    # This is also a (shallow) copy.
    x = py_utils.NestedMap(
        a='a', b='b', c=py_utils.NestedMap(d='d', e=[1, 2, 4]))
    y = x.copy()
    self.assertNotEqual(id(x), id(y))
    y.a = 'y'
    y.c.d = 'z'
    self.assertEqual('y', y.a)
    self.assertEqual('a', x.a)
    self.assertEqual('z', y.c.d)
    self.assertEqual('z', x.c.d)


class ReadOnlyAttrDictViewTest(tf.test.TestCase):

  def testWrapping(self):
    backing = dict()
    view = py_utils.ReadOnlyAttrDictView(backing)
    backing['test'] = 1

    self.assertEquals(1, view['test'])
    self.assertEquals(1, view.test)
    # Item assign.
    with self.assertRaises(AttributeError):
      view['test'] = 2
    self.assertEquals(1, view['test'])
    # Attr assign.
    with self.assertRaises(AttributeError):
      view.test = 2
    self.assertEquals(1, view['test'])
    # Delete attr.
    with self.assertRaises(AttributeError):
      del view.test
    self.assertEquals(1, view['test'])
    # Delete item.
    with self.assertRaises(AttributeError):
      del view['test']
    self.assertEquals(1, view['test'])


class ApplyPaddingTest(tf.test.TestCase):

  def testApplyPaddingToZeroWithBroadcast(self):
    with self.test_session():
      y = py_utils.ApplyPadding([[0.0], [1.0], [0.0]],
                                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).eval()
      self.assertAllClose(y, [[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])

  def testApplyPaddingToConstWithBroadcast(self):
    with self.test_session():
      y = py_utils.ApplyPadding([[0.0], [1.0], [0.0]],
                                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                [[1.0, 2.0], [9.0, 10.0], [5.0, 6.0]]).eval()
      self.assertAllClose(y, [[1.0, 2.0], [9.0, 10.0], [5.0, 6.0]])

  def testApplyPaddingToZeroWithoutBroadcast(self):
    with self.test_session():
      y = py_utils.ApplyPadding([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).eval()
      self.assertAllClose(y, [[1.0, 2.0], [0.0, 4.0], [5.0, 0.0]])


if __name__ == '__main__':
  tf.test.main()
