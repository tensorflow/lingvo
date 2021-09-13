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
"""Tests for py_utils."""

import collections
import contextlib
import copy
import itertools
import math
import os
import sys

from absl.testing import flagsaver
from absl.testing import parameterized
from lingvo import model_registry
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import builder_layers
from lingvo.core import cluster_factory
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import py_utils_flags
from lingvo.core import recurrent
from lingvo.core import symbolic
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.image.params import mnist  # pylint: disable=unused-import

import mock
import numpy as np

from tensorflow.python.ops import functional_ops  # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.ops import init_ops  # pylint:disable=g-direct-tensorflow-import

FLAGS = tf.flags.FLAGS


class PyUtilsTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    with contextlib.ExitStack() as stack:
      self._store = stack.enter_context(py_utils.VariableStore())
      self.addCleanup(stack.pop_all().close)

  def testEnableAssertFlagOverrideFromCluster(self):
    cluster_params = cluster_factory.Current().params.Copy()
    cluster_params.enable_asserts = True
    with cluster_factory.Cluster(cluster_params):
      self.assertTrue(py_utils_flags.enable_asserts())
    cluster_params.enable_asserts = False
    with cluster_factory.Cluster(cluster_params):
      self.assertFalse(py_utils_flags.enable_asserts())

  def testEnableCheckNumericsFlagOverrideFromCluster(self):
    cluster_params = cluster_factory.Current().params.Copy()
    cluster_params.enable_check_numerics = True
    with cluster_factory.Cluster(cluster_params):
      self.assertTrue(py_utils_flags.enable_check_numerics())
    cluster_params.enable_check_numerics = False
    with cluster_factory.Cluster(cluster_params):
      self.assertFalse(py_utils_flags.enable_check_numerics())

  def testIsDefaultParamInit(self):
    p = py_utils.DefaultParamInit()
    self.assertTrue(py_utils.IsDefaultParamInit(p))
    p = hyperparams.Params.FromProto(p.ToProto())
    self.assertTrue(py_utils.IsDefaultParamInit(p))
    p = py_utils.WeightInit.Xavier(scale=1.)
    self.assertFalse(py_utils.IsDefaultParamInit(p))

  def testNestedMapFromNestedDict(self):
    a = {'a1': 1, 'a2': 2}
    b = {'b1': 1, 'b2': 2}
    c = {'a': a, 'b': b, 'ab': [a, b]}
    d = py_utils.NestedMap(c)
    e = py_utils.NestedMap.FromNestedDict(c)
    self.assertIsInstance(d, py_utils.NestedMap)
    self.assertNotIsInstance(d.a, py_utils.NestedMap)
    self.assertIsInstance(e.a, py_utils.NestedMap)
    self.assertIsInstance(e.ab[0], py_utils.NestedMap)
    self.assertEqual(e.ab[0], e.a)
    self.assertEqual(e.ab[1], e.b)

  def testNestedMapToNestedDict(self):
    a = py_utils.NestedMap(a1=1, a2=2)
    b = {'b1': 1, 'b2': 2}
    c = py_utils.NestedMap(a=a, b=b, ab=[a, b])
    d = c.ToNestedDict()
    self.assertIsInstance(c, py_utils.NestedMap)
    self.assertNotIsInstance(d, py_utils.NestedMap)
    self.assertIsInstance(c.a, py_utils.NestedMap)
    self.assertNotIsInstance(d['a'], py_utils.NestedMap)
    self.assertNotIsInstance(d['ab'][0], py_utils.NestedMap)

  def testNestedMapGetItem(self):
    nested_map = py_utils.NestedMap()
    nested_map['a'] = py_utils.NestedMap({'x': 3})
    nested_map['b'] = py_utils.NestedMap({'y_0': 4, 'y': [3]})
    self.assertEqual(nested_map.GetItem('a.x'), nested_map.a.x)
    self.assertEqual(nested_map.GetItem('b.y_0'), 4)
    self.assertEqual(nested_map.GetItem('b.y[0]'), 3)
    with self.assertRaises(KeyError):
      nested_map.GetItem('o')
    with self.assertRaises(IndexError):
      nested_map.GetItem('b.y[1]')
    with self.assertRaises(TypeError):
      nested_map.GetItem('b.y.c')

  def testNestedMapGet(self):
    nested_map = py_utils.NestedMap({'a': {'b': 0}})
    self.assertEqual(nested_map.Get('a.b'), 0)
    self.assertIsNone(nested_map.Get('a.b.c'))
    self.assertIsNone(nested_map.Get('x'))
    self.assertEqual(nested_map.Get('x', 0), 0)

  def testNestedMapGetFromNestedList(self):
    nested_map = py_utils.NestedMap({'a': {'b': [0, 1, {'c': 2}]}})
    self.assertEqual(nested_map.Get('a.b'), [0, 1, {'c': 2}])
    self.assertEqual(nested_map.Get('a.b[1]'), 1)
    self.assertEqual(nested_map.Get('a.b[2].c'), 2)
    self.assertIsNone(nested_map.Get('a.b[3]'))
    self.assertIsNone(nested_map.Get('a.b.c'))

  def testNestedMapDir(self):
    nested_map = py_utils.NestedMap({'a': {'b': 0}})
    nested_map.c = '1'
    nested_map_dir = dir(nested_map)
    # Flattened keys are in the dir
    self.assertIn('a.b', nested_map_dir)
    self.assertIn('c', nested_map_dir)
    # So are method APIs.
    self.assertIn('Get', nested_map_dir)

  def testNestedMapSet(self):
    nested_map = py_utils.NestedMap.FromNestedDict({'a': {'b': 0}})
    self.assertEqual(nested_map.a.b, 0)
    # Test if overriding an existing value works.
    nested_map.Set('a.b', 1)
    self.assertEqual(nested_map.a.b, 1)
    # Test if ValueError is raised if an existing intermediate value is not a
    # NestedMap.
    with self.assertRaises(ValueError):
      nested_map.Set('a.b.c', 2)
    nested_map.Set('a.b', py_utils.NestedMap())
    # Verify that non-existing intermediate keys are set to NestedMap.
    nested_map.Set('a.b.x.y', 2)
    self.assertIsInstance(nested_map.a.b.x, py_utils.NestedMap)
    self.assertEqual(nested_map.a.b.x.y, 2)
    # Verify that non-existing intermediate list keys are set to list.
    nested_map.Set('a.b.z[0].y', 3)
    nested_map.Set('a.b.z[1].y', 4)
    self.assertIsInstance(nested_map.a.b.z, list)
    self.assertEqual(nested_map.a.b.z[0].y, 3)
    self.assertEqual(nested_map.a.b.z[1].y, 4)
    # Verify that using a index > len(list) leads to ValueError.
    with self.assertRaises(ValueError):
      nested_map.Set('a.b.z[3]', 5)

  def testNestedMapSetIndexed(self):
    nested_map = py_utils.NestedMap()
    nested_map.x = [43, 44]
    nested_map.y = 14
    nested_map.z = [py_utils.NestedMap(b=13, c=[18, 8]), 4]
    another_map = py_utils.NestedMap()
    for k, v in nested_map.FlattenItems():
      another_map.Set(k, v)
    self.assertDictEqual(another_map, nested_map)

  @parameterized.named_parameters(
      ('_stateful', False, [tf.float32, tf.float64, tf.complex64, tf.int8]),
      ('_stateless', True, [tf.float32, tf.float64]))
  def testCreateVariableBasics(self, stateless_vars_init, dtypes):
    with flagsaver.flagsaver(stateless_vars_init=stateless_vars_init):
      with self.session(use_gpu=False):
        methods = [
            py_utils.WeightInit.Gaussian,
            py_utils.WeightInit.Uniform,
            py_utils.WeightInit.Constant,
            py_utils.WeightInit.TruncatedGaussian,
            py_utils.WeightInit.GaussianSqrtDim,
            py_utils.WeightInit.UniformSqrtDim,
            py_utils.WeightInit.UniformUnitScaling,
            py_utils.WeightInit.UniformUnitScalingFanAvg,
            py_utils.WeightInit.TruncatedGaussianSqrtDim,
            py_utils.WeightInit.TruncatedGaussianSqrtFanIn,
            py_utils.WeightInit.TruncatedGaussianSqrtFanOut,
            py_utils.WeightInit.GaussianSqrtFanAvg,
        ]
        shapes = [[], [3], [2, 4], [3, 3, 2, 4]]
        col = ['col1', 'col2']

        all_vars = []
        for i, (m, dt,
                sp) in enumerate(itertools.product(methods, dtypes, shapes)):
          pc = py_utils.WeightParams(sp, m(), dt, col)
          all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

        # To reuse existing variables
        tf.get_variable_scope().reuse_variables()

        self.assertEqual(len(tf.trainable_variables()), len(all_vars))

        all_vars_copy = []
        for i, (m, dt,
                sp) in enumerate(itertools.product(methods, dtypes, shapes)):
          pc = py_utils.WeightParams(sp, m(), dt, col)
          all_vars_copy.append(py_utils.CreateVariable('var_%d' % i, pc))

        self.evaluate(tf.global_variables_initializer())
        for v1, v2 in zip(all_vars, all_vars_copy):
          v1_v = self.evaluate(v1)
          v2_v = self.evaluate(v2)
          self.assertAllEqual(v1_v, v2_v)

  def testCreateVariableWithSymbols(self):
    with self.session(use_gpu=False):
      dim_symbol = symbolic.Symbol('dim')
      shape = [2, 3, dim_symbol * 2]

      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES, {dim_symbol: 2}):
        pc = py_utils.WeightParams(shape, py_utils.WeightInit.Gaussian(),
                                   tf.float32, ['col1', 'col2'])
        var = py_utils.CreateVariable('var', pc)

      # To reuse existing variables
      tf.get_variable_scope().reuse_variables()

      new_dim_symbol = symbolic.Symbol('new_dim')
      # Same shape as above but from different symbol.
      shape = [2, 3, new_dim_symbol * 2]
      with symbolic.SymbolToValueMap(symbolic.STATIC_VALUES,
                                     {new_dim_symbol: 2}):
        pc = py_utils.WeightParams(shape, py_utils.WeightInit.Gaussian(),
                                   tf.float32, ['col1', 'col2'])
        var_copy = py_utils.CreateVariable('var', pc)

      self.evaluate(tf.global_variables_initializer())
      self.assertAllEqual(self.evaluate(var), self.evaluate(var_copy))

  def testCreateVariableWithRegexDTypes(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      methods = [
          py_utils.WeightInit.Uniform,
          py_utils.WeightInit.UniformSqrtDim,
          py_utils.WeightInit.UniformUnitScaling,
      ]
      dtypes = [tf.float32, tf.complex64]
      all_vars = []
      for i, (dt_orig, m) in enumerate(itertools.product(dtypes, methods)):
        pc = py_utils.WeightParams([2, 3], m(0.1), dt_orig)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

      regex_dtypes = [
          # var_100 does not exist. So it changes nothing.
          ('var_100', tf.float16),
          # reset var_new_0 to tf.float16.
          ('var_new_0$', tf.float16),
          # reset var_new_3 to tf.loat16.
          ('.*3', tf.float16)
      ]
      all_vars_new = []
      with py_utils.VariableListDtypeRegexScope(regex_dtypes):
        for i, (dt_orig, m) in enumerate(itertools.product(dtypes, methods)):
          pc = py_utils.WeightParams([2, 3], m(0.1), dt_orig)
          all_vars_new.append(py_utils.CreateVariable('var_new_%d' % i, pc))
      self.evaluate(tf.global_variables_initializer())

      v1_v = self.evaluate(all_vars[0])
      v2_v = self.evaluate(all_vars[1])
      v4_v = self.evaluate(all_vars[3])
      v1_v_expted = [[0.069674, -0.072278, -0.021777],
                     [-0.052155, -0.050274, 0.086218]]
      v2_v_expted = [[0.005361, 0.036109, -0.036575],
                     [0.058314, 0.031438, 0.049196]]
      v4_v_expted = [
          [0.015448 + 0.068295j, -0.098710 - 0.054435j, 0.037030 - 0.048017j],
          [-0.047435 + 0.035301j, 0.041994 + 0.000279j, -0.029097 + 0.084902j],
      ]
      tf.assert_type(all_vars[0], tf.float32)
      self.assertAllClose(v1_v_expted, v1_v.tolist())
      tf.assert_type(all_vars[1], tf.float32)
      self.assertAllClose(v2_v_expted, v2_v.tolist())
      tf.assert_type(all_vars[3], tf.complex64)
      self.assertAllClose(v4_v_expted, v4_v.tolist())

      v1_v = self.evaluate(all_vars_new[0])
      v4_v = self.evaluate(all_vars_new[3])
      v1_v_expted = [[0.089233, 0.016235, -0.053497],
                     [0.077759, -0.00177, -0.08728]]
      v4_v_expted = [[0.028931, -0.064636, 0.016968],
                     [0.093506, 0.020142, -0.087646]]
      tf.assert_type(all_vars_new[0], tf.float16)
      self.assertAllClose(v1_v_expted, v1_v.tolist())
      tf.assert_type(all_vars_new[3], tf.float16)
      self.assertAllClose(v4_v_expted, v4_v.tolist())

  def testCreateVariableUniform(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      methods = [
          py_utils.WeightInit.Uniform,
          py_utils.WeightInit.UniformSqrtDim,
          py_utils.WeightInit.UniformUnitScaling,
      ]
      dtypes = [tf.float32, tf.complex64]
      shapes = [[2, 3]]
      all_vars = []
      for i, (dt, m,
              sp) in enumerate(itertools.product(dtypes, methods, shapes)):
        pc = py_utils.WeightParams(sp, m(0.1), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

      v1_v_expted = [[0.069674, -0.072278, -0.021777],
                     [-0.052155, -0.050274, 0.086218]]
      v2_v_expted = [[0.005361, 0.036109, -0.036575],
                     [0.058314, 0.031438, 0.049196]]
      v4_v_expted = [
          [0.015448 + 0.068295j, -0.098710 - 0.054435j, 0.037030 - 0.048017j],
          [-0.047435 + 0.035301j, 0.041994 + 0.000279j, -0.029097 + 0.084902j],
      ]

      self.evaluate(tf.global_variables_initializer())
      v1_v = self.evaluate(all_vars[0])
      v2_v = self.evaluate(all_vars[1])
      v4_v = self.evaluate(all_vars[3])
      self.assertAllClose(v1_v_expted, v1_v.tolist())
      self.assertAllClose(v2_v_expted, v2_v.tolist())
      self.assertAllClose(v4_v_expted, v4_v.tolist())

  def testCreateVariableNormal(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(832124)
      methods = [
          py_utils.WeightInit.Gaussian,
          py_utils.WeightInit.GaussianSqrtDim,
      ]
      dtypes = [tf.float32, tf.complex64]
      shapes = [[2, 3]]
      all_vars = []
      for i, (dt, m,
              sp) in enumerate(itertools.product(dtypes, methods, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

      v1_v_expted = [[-1.472208, 0.960204, -0.192588],
                     [-0.461884, 1.018134, 0.063719]]
      v2_v_expted = [[-0.862255, -0.688153, 0.82515],
                     [-0.07671, 0.613031, -0.020327]]
      v3_v_expted = [
          [1.005469 + 0.827639j, 1.249896 + 0.802671j, -0.026286 - 0.813836j],
          [0.865386 + 0.301172j, 0.876698 - 0.907293j, 1.996337 + 1.840192j],
      ]

      self.evaluate(tf.global_variables_initializer())
      v1_v = self.evaluate(all_vars[0])
      v2_v = self.evaluate(all_vars[1])
      v3_v = self.evaluate(all_vars[2])
      self.assertAllClose(v1_v_expted, v1_v.tolist())
      self.assertAllClose(v2_v_expted, v2_v.tolist())
      self.assertAllClose(v3_v_expted, v3_v.tolist())

  def testCreateVariableCustomVarInit(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(832124)

      var = py_utils.CreateVariable(
          'var',
          py_utils.WeightParams(
              shape=(2, 2),
              dtype=tf.float32,
              init=py_utils.WeightInit.CustomVarInit(
                  init_ops.constant_initializer(3))))

      v_expected = [[3, 3], [3, 3]]
      self.evaluate(tf.global_variables_initializer())
      v = self.evaluate(var)
      self.assertAllClose(v_expected, v.tolist())

  def testCreateVariableSqrtFanInOut(self):
    with self.session():
      tf.random.set_seed(832124)
      methods = [
          py_utils.WeightInit.GaussianSqrtFanIn,
          py_utils.WeightInit.TruncatedGaussianSqrtFanIn,
          py_utils.WeightInit.GaussianSqrtFanOut,
          py_utils.WeightInit.TruncatedGaussianSqrtFanOut,
          py_utils.WeightInit.GaussianSqrtFanAvg,
      ]
      dtypes = [tf.float32]
      shapes = [[1, 1, 2, 3]]
      all_vars = []
      for i, (dt, m,
              sp) in enumerate(itertools.product(dtypes, methods, shapes)):
        pc = py_utils.WeightParams(sp, m(scale=2), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

      self.evaluate(tf.global_variables_initializer())
      var_values = self.evaluate(all_vars)
      tf.logging.info('var_values=%s', var_values)
      self.assertAllClose(
          [
              # GaussianSqrtFanIn.
              [[[[-2.08201575, 1.35793388, -0.27236053],
                 [-0.65320235, 1.43985856, 0.09011276]]]],
              # TruncatedGaussianSqrtFanIn.
              [[[[-1.72450912, -1.37630582, 1.65029943],
                 [-0.15342039, -0.7636584, -0.97026265]]]],
              # GaussianSqrtFanOut.
              [[[[1.16101539, 1.4432559, -0.03035267],
                 [0.9992612, 1.01232362, 2.30517101]]]],
              # TruncatedGaussianSqrtFanOut.
              [[[[-0.049076, -0.25183302, -1.79192507],
                 [0.93166995, -0.83121753, -1.40264213]]]],
              # GaussianSqrtFanAvg.
              [[[[-0.59545106, 1.232773, -0.2630677],
                 [0.11635535, 0.9317614, 0.16670291]]]],
          ],
          var_values)

  def testCreateVariableException(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(832124)
      pc = py_utils.WeightParams([2, 3], py_utils.WeightInit.Gaussian())
      var1 = py_utils.CreateVariable('var1', pc)

      tf.get_variable_scope().reuse_variables()
      # Reuses an existing variable.
      var2 = py_utils.CreateVariable('var1', pc)

      # An exception should be thrown in this case.
      pc = py_utils.WeightParams([2, 3], py_utils.WeightInit.Gaussian(2.0))
      with self.assertRaises(AssertionError):
        self.evaluate(py_utils.CreateVariable('var1', pc))

      self.evaluate(tf.global_variables_initializer())
      self.assertAllEqual(self.evaluate(var1), self.evaluate(var2))

  def testCreateVariableDifferentSeed(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(3251343)
      pc = py_utils.WeightParams([2, 3], py_utils.WeightInit.Gaussian())
      with tf.variable_scope('layer0'):
        w0 = py_utils.CreateVariable('w', pc)
      with tf.variable_scope('layer1'):
        w1 = py_utils.CreateVariable('w', pc)
      self.evaluate(tf.global_variables_initializer())

      # w0_val, w1_val should be sufficient different.
      w0_val, w1_val = self.evaluate([w0, w1])
      print(['diff = ', w0_val - w1_val])
      self.assertGreater(np.max(np.abs(w0_val - w1_val)), 0.1)

  def testXavier(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(1618)
      methods = [py_utils.WeightInit.Xavier]
      dtypes = [tf.float32, tf.float16, tf.complex64]
      shapes = [[2, 3]]
      all_vars = []
      for i, (m, dt,
              sp) in enumerate(itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

      v1_v_expted = [[1.051236, -0.959198, 0.796091],
                     [-0.685691, 0.230933, -1.006293]]
      v3_v_expted = [
          [0.149996 - 0.064369j, 0.689145 + 0.017257j, -0.502070 - 0.367683j],
          [0.519782 + 0.470412j, 0.738902 - 0.054006j, 0.028603 + 0.471832j],
      ]

      self.evaluate(tf.global_variables_initializer())
      v1_v = self.evaluate(all_vars[0])
      v3_v = self.evaluate(all_vars[2])
      self.assertAllClose(v1_v_expted, v1_v.tolist())
      self.assertAllClose(v3_v_expted, v3_v.tolist())

  def testXavier1D(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(1618)
      methods = [py_utils.WeightInit.Xavier]
      dtypes = [tf.float32, tf.float16, tf.complex64]
      shapes = [[2]]
      all_vars = []
      for i, (m, dt,
              sp) in enumerate(itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

      v1_v_expted = [1.175317, -1.072416]

      self.evaluate(tf.global_variables_initializer())
      v1_v = self.evaluate(all_vars[0])
      self.assertAllClose(v1_v_expted, v1_v.tolist())

  def testXavier3D(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(1618)
      methods = [py_utils.WeightInit.Xavier]
      dtypes = [tf.float32, tf.float16, tf.complex64]
      shapes = [[1, 1, 2]]
      all_vars = []
      for i, (m, dt,
              sp) in enumerate(itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

      v1_v_expted = [[[1.357139, -1.23832]]]

      self.evaluate(tf.global_variables_initializer())
      v1_v = self.evaluate(all_vars[0])
      self.assertAllClose(v1_v_expted, v1_v.tolist())

  def testVariableShapePrefix(self):
    with self.session(use_gpu=False):
      shape = [3, 2]
      pc = py_utils.WeightParams(
          shape=shape, init=py_utils.WeightInit.Constant(0.0), dtype=tf.float32)
      with py_utils.VariableShapePrefixContext(5):
        with py_utils.VariableShapePrefixContext(4):
          var = py_utils.CreateVariable('var', pc)
      self.assertEqual([5, 4, 3, 2], var.shape.as_list())
      self.assertEqual(2, py_utils.GetVarLeadingDimsAsCombinedLayers(var))

  def testGeoMeanXavier(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(1618)
      methods = [py_utils.WeightInit.GeoMeanXavier]
      dtypes = [tf.float32, tf.float16, tf.complex64]
      shapes = [[2, 3]]
      all_vars = []
      for i, (m, dt,
              sp) in enumerate(itertools.product(methods, dtypes, shapes)):
        pc = py_utils.WeightParams(sp, m(), dt)
        all_vars.append(py_utils.CreateVariable('var_%d' % i, pc))

      v1_v_expted = [[1.062019, -0.969037, 0.804257],
                     [-0.692724, 0.233301, -1.016615]]
      v3_v_expted = [[
          0.151534 - 0.065029j, 0.696214 + 0.017434j, -0.507220 - 0.371455j
      ], [0.525114 + 0.475238j, 0.746481 - 0.05456j, 0.028896 + 0.476672j]]

      self.evaluate(tf.global_variables_initializer())
      v1_v = self.evaluate(all_vars[0])
      v3_v = self.evaluate(all_vars[2])
      self.assertAllClose(v1_v_expted, v1_v.tolist())
      self.assertAllClose(v3_v_expted, v3_v.tolist())

  def testCheckNumerics(self):
    xv = [[1, 2], [3, 4]]
    yv = [10] * 4
    with self.session():
      x = tf.constant(xv, tf.float32)
      y = tf.constant(yv)
      z = tf.reduce_mean(tf.constant([], tf.float32))
      self.assertAllClose(xv, self.evaluate(py_utils.CheckNumerics(x)))
      self.assertAllClose(yv, self.evaluate(py_utils.CheckNumerics(y)))
      actual_xv, actual_yv = self.evaluate(py_utils.CheckNumerics([x, y]))
      self.assertAllClose(xv, actual_xv)
      self.assertAllClose(yv, actual_yv)
      actual_xv, actual_yv = self.evaluate(py_utils.CheckNumerics((x, y)))
      self.assertAllClose(xv, actual_xv)
      self.assertAllClose(yv, actual_yv)

      with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'NaN'):
        self.evaluate(py_utils.CheckNumerics(z))

  def testLog(self):
    with self.session():
      x = tf.constant([[1, 2], [3, 4]])
      y = tf.constant([10] * 4)
      x = py_utils.Log(x, 'testLog', x=x, y=y)
      self.assertAllEqual(self.evaluate(x), [[1, 2], [3, 4]])

  def testDebug(self):
    with self.session():
      x = tf.constant([[1, 2], [3, 4]])
      y = tf.constant([11] * 4)
      z = tf.constant([22] * 4)
      x = py_utils.Debug(x, 'msg')
      self.assertAllEqual(self.evaluate(x), [[1, 2], [3, 4]])

      x = py_utils.Debug(x, 'msg', more=[y, z])
      self.assertAllEqual(self.evaluate(x), [[1, 2], [3, 4]])

  def testSave(self):
    with self.session():
      x = tf.constant([[1, 2], [3, 4]])
      y = tf.constant([10] * 4)
      x = py_utils.Save(x, '%s/test' % self.get_temp_dir(), x=x, y=y)
      self.evaluate(tf.global_variables_initializer())
      self.assertAllEqual(self.evaluate(x), [[1, 2], [3, 4]])

    # Reads npy files and check the values.
    read_x = np.load('%s/test.%08d.x.npy' % (self.get_temp_dir(), 0))
    read_y = np.load('%s/test.%08d.y.npy' % (self.get_temp_dir(), 0))
    self.assertAllEqual(read_x, [[1, 2], [3, 4]])
    self.assertAllEqual(read_y, [10] * 4)

  def testTensorRank(self):
    a = tf.constant([1])
    self.assertIsInstance(py_utils.HasRank(a, 1), tf.Tensor)
    self.assertIsInstance(py_utils.HasAtLeastRank(a, 1), tf.Tensor)

    b = tf.constant([[1, 2]])
    self.assertIsInstance(py_utils.HasRank(b, 2), tf.Tensor)
    self.assertIsInstance(py_utils.HasAtLeastRank(b, 1), tf.Tensor)
    self.assertIsInstance(py_utils.HasAtLeastRank(b, 2), tf.Tensor)
    with self.assertRaises(Exception):
      py_utils.HasAtLeastRank(b, 3)

    c = tf.placeholder(tf.int32, shape=None)
    d = py_utils.HasAtLeastRank(c, 3)
    with self.session() as sess:
      d_v = sess.run(d, feed_dict={c: np.array([[[1, 2]]])})
      self.assertAllEqual([[[1, 2]]], d_v)
      with self.assertRaises(Exception):
        sess.run(d, feed_dict={c: np.array([[1, 2]])})

  def testTensorRankDisableAsserts(self):
    with flagsaver.flagsaver(enable_asserts=False):
      c = tf.placeholder(tf.int32, shape=None)
      d = py_utils.HasAtLeastRank(c, 3)
      with self.session() as sess:
        d_v = sess.run(d, feed_dict={c: np.array([[1, 2]])})
        self.assertAllEqual([[1, 2]], d_v)

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

    c = tf.zeros([1, a[0], a.shape[0], tf.shape(a)[0]])
    self.assertEqual(py_utils.GetShape(c)[0], 1)
    self.assertEqual(py_utils.GetShape(c)[1], 1)
    self.assertEqual(py_utils.GetShape(c)[2], 1)
    self.assertEqual(py_utils.GetShape(c)[3], 1)

    d = tf.placeholder(tf.float32, shape=(1, None))
    self.assertEqual(py_utils.GetShape(d)[0], 1)
    self.assertIsInstance(py_utils.GetShape(d)[1], tf.Tensor)

    e = tf.zeros([d.shape[0], tf.shape(d)[0], tf.shape(d)[1]])
    self.assertEqual(py_utils.GetShape(e)[0], 1)
    # TODO(b/167426925): re-enable once cl/380675625 is submitted
    # self.assertEqual(py_utils.GetShape(e)[1], 1)
    self.assertIsInstance(py_utils.GetShape(e)[2], tf.Tensor)

    f = tf.placeholder(tf.float32)
    self.assertIsNone(f.shape.ndims)
    # GetShape() will return a Tensor.
    self.assertIsInstance(py_utils.GetShape(f), tf.Tensor)

  def testGetSize(self):
    a = tf.constant([1])
    self.assertEqual(py_utils.GetSize(a), 1)

    b = tf.constant([[1, 2]])
    self.assertEqual(py_utils.GetSize(b), 2)

    d = tf.placeholder(tf.float32, shape=(1, None))
    self.assertIsInstance(py_utils.GetSize(d), tf.Tensor)

    shape = tf.placeholder(tf.int32)
    f = py_utils.GetSize(tf.reshape(d, shape))
    self.assertIsInstance(f, tf.Tensor)

    with self.session() as sess:
      f_v = sess.run(f, feed_dict={d: np.array([[1, 2]]), shape: [2]})
      self.assertEqual(2, f_v)

  def testUpdateFpropDtype(self):
    network_p = builder_layers.SequentialLayer.Params()
    linear_layer_p = builder_layers.LinearLayer.Params()
    linear_layer_p.input_dims = 5
    linear_layer_p.output_dims = 6
    network_p.sub.append(linear_layer_p)

    py_utils.UpdateFpropDtype(network_p, tf.bfloat16)
    self.assertEqual(network_p.sub[0].fprop_dtype, tf.bfloat16)

  def testUpdateDtype(self):
    network_p = builder_layers.SequentialLayer.Params()
    linear_layer_p = builder_layers.LinearLayer.Params()
    linear_layer_p.input_dims = 5
    linear_layer_p.output_dims = 6
    network_p.sub.append(linear_layer_p)

    py_utils.UpdateDtype(network_p, tf.bfloat16)
    self.assertEqual(network_p.sub[0].dtype, tf.bfloat16)

  def testGetRank(self):
    a = tf.constant([1])
    self.assertEqual(py_utils.GetRank(a), 1)

    b = tf.constant([[1, 2]])
    self.assertEqual(py_utils.GetRank(b), 2)

    c = tf.zeros([1, a[0], a.shape[0], tf.shape(a)[0]])
    self.assertEqual(py_utils.GetRank(c), 4)

    d = tf.placeholder(tf.float32, shape=(1, None))
    self.assertEqual(py_utils.GetRank(d), 2)

    e = tf.zeros([d.shape[0], tf.shape(d)[0], tf.shape(d)[1]])
    self.assertEqual(py_utils.GetRank(e), 3)

    f = tf.placeholder(tf.float32)
    self.assertIsNone(f.shape.ndims)
    # GetRank() will return a Tensor.
    self.assertIsInstance(py_utils.GetRank(f), tf.Tensor)

  def testRenamingRules(self):
    pc = py_utils.WeightParams([3, 3])
    with tf.variable_scope('model'):
      v1 = py_utils.CreateVariable('v1', pc)
      with py_utils.VariableRenameScope([('model/(.*)', 'data/%s')]):
        v2 = py_utils.CreateVariable('v2', pc)
      v3 = py_utils.CreateVariable('v3', pc)

    self.assertEqual(v1.name, 'model/v1/var:0')
    self.assertEqual(v2.name, 'data/v2/var:0')
    self.assertEqual(v3.name, 'model/v3/var:0')

  def testOpportunisticReuse(self):
    with self.session():
      pc = py_utils.WeightParams([3, 3])
      v1 = py_utils.CreateVariable('v1', pc)
      with self.assertRaises(Exception):
        self.evaluate(py_utils.CreateVariable('v1', pc))
      with py_utils.OpportunisticVariableReuseScope(True):
        v2 = py_utils.CreateVariable('v1', pc)
        x1 = py_utils.CreateVariable('x1', pc)
        with py_utils.OpportunisticVariableReuseScope(False):
          with self.assertRaises(Exception):
            self.evaluate(py_utils.CreateVariable('v1', pc))
        v3 = py_utils.CreateVariable('v1', pc)
      with self.assertRaises(Exception):
        self.evaluate(py_utils.CreateVariable('v1', pc))

      for v in [v2, v3]:
        self.assertIs(v1, v)
      self.assertIsNot(v1, x1)

  def testGetOrCreateGlobalStepVar(self):
    with tf.variable_scope('s1'):
      with tf.name_scope('s2'):
        gs1 = py_utils.GetOrCreateGlobalStepVar()
        gs2 = tf.train.get_global_step()
      gs3 = py_utils.GetOrCreateGlobalStepVar()
      gs4 = tf.train.get_global_step()
    gs5 = py_utils.GetOrCreateGlobalStepVar()
    gs6 = tf.train.get_global_step()
    for gs in [gs2, gs3, gs4, gs5, gs6]:
      self.assertIs(gs1, gs)
    self.assertEqual(gs1.name, 'global_step:0')

  def testCreateLocalTheta(self):
    methods = [py_utils.WeightInit.Gaussian, py_utils.WeightInit.Uniform]
    dtypes = [tf.float32, tf.complex64]
    shapes = [[2, 4], [3]]

    test_vars = py_utils.NestedMap()
    for i, (m, dt, sp) in enumerate(itertools.product(methods, dtypes, shapes)):
      pc = py_utils.WeightParams(sp, m(), dt, 'col1')
      var = py_utils.CreateVariable('var_%d' % i, pc)
      with tf.device(var.device):
        test_vars['var_%d' % i] = tf.identity(var)

    test_devices = [
        '/job:worker/replica:0/device:GPU:0',
        '/job:worker/replica:0/device:GPU:1'
    ]

    sharded_local_vars = py_utils.CreateLocalTheta(test_vars, test_devices)
    sharded_local_vars_list = sharded_local_vars.Flatten()

    # assert proper device placement
    for i, v in enumerate(sharded_local_vars_list):
      expected_device = test_devices[i % len(test_devices)]
      self.assertEqual(v.device, expected_device)

  def testComputeGradient(self):
    with self.session(use_gpu=False):
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
      self.assertEqual(var_grads.a.var.name, 'a:0')

  def testVarGradNestFlatten(self):
    a = tf.get_variable('a', [])
    b = tf.get_variable('b', [])
    vs_gs = py_utils.NestedMap(
        a=py_utils.VarGrad(a,
                           tf.ones_like(a) * 10.0),
        b=py_utils.VarGrad(b,
                           tf.ones_like(b) * 0.5))
    flattened = tf.nest.flatten(vs_gs)
    self.assertLen(flattened, 2)
    for x in flattened:
      self.assertIsInstance(x, py_utils.VarGrad)

  def testClipSingleTensorGradients(self):
    a = tf.get_variable('a', [])
    b = tf.get_variable('b', [])
    vs_gs = py_utils.NestedMap(
        a=py_utils.VarGrad(a,
                           tf.ones_like(a) * 10.0),
        b=py_utils.VarGrad(b,
                           tf.ones_like(b) * 0.5))
    clipped = py_utils.ApplyGradNormClipping(vs_gs, norm=1.0)
    with self.session(use_gpu=False):
      self.evaluate(tf.global_variables_initializer())
      clipped_np = self.evaluate(clipped.Transform(tuple))
      # Each variable is clipped indipendently to grad scale of 1.
      self.assertAllClose(clipped_np.a[1], 1.0)
      self.assertAllClose(clipped_np.b[1], 0.5)

  def testMaskGradient(self):
    with self.session(use_gpu=False):
      a = tf.get_variable('a', [])
      b = tf.get_variable('b', [])
      c = tf.get_variable('c', [])
      d = tf.get_variable('d', [])
      e = tf.get_variable('e', [])
      l = a + b + c + d
      zeros = tf.zeros(3, dtype=tf.float32)
      select = tf.one_hot(1, 3, dtype=tf.float32)
      vmap = py_utils.NestedMap(
          a=a, b=b, c=c, d=d, n=py_utils.NestedMap(aa=a, e=e))
      grad_mask = py_utils.NestedMap()
      grad_mask['a:0'] = zeros
      grad_mask['b:0'] = zeros
      grad_mask['c:0'] = select
      grad_mask['d:0'] = select
      grad_onehot = tf.one_hot(1, 3, dtype=tf.float32)
      grad_mask = {
          k: tf.tensordot(v, grad_onehot, 1) for k, v in grad_mask.items()
      }
      var_grads = py_utils.ComputeGradients(l, vmap)
      var_grads_mask = py_utils.MaskGradients(var_grads, grad_mask)
      self.evaluate(tf.global_variables_initializer())
      _, var_grads_mask_vals = self.evaluate(
          [var_grads.Transform(tuple),
           var_grads_mask.Transform(tuple)])
      # 'a' and 'b' are masked, while 'c' and 'd' are not.
      self.assertEqual(var_grads_mask_vals['a'][1], 0)
      self.assertEqual(var_grads_mask_vals['b'][1], 0)
      self.assertEqual(var_grads_mask_vals['c'][1], 1)
      self.assertEqual(var_grads_mask_vals['d'][1], 1)

  def testSkipL2Regularization(self):
    with self.session(use_gpu=False):
      beta = tf.get_variable(
          'beta',
          initializer=tf.constant(np.arange(10).reshape([1, 10]), tf.float32))
      tf.add_to_collection(py_utils.SKIP_LP_REGULARIZATION, beta)
      gamma = tf.get_variable(
          'gamma',
          initializer=tf.constant(np.arange(10).reshape([1, 10]), tf.float32))
      act = tf.constant(np.arange(10).reshape([1, 10]), tf.float32)
      pred = act * gamma + beta
      loss = tf.reduce_sum(pred)
      vmap = py_utils.NestedMap(beta=beta, gamma=gamma)
      var_grads = py_utils.ComputeGradients(loss, vmap)
      self.assertCountEqual(var_grads.keys(), ['beta', 'gamma'])
      l2_loss, var_grads_with_l2 = py_utils.AdjustGradientsWithLpLoss(
          var_grads, 0.1, p=2.0)

      self.evaluate(tf.global_variables_initializer())
      var_grads_vals, l2_loss_val, var_grads_with_l2_vals = self.evaluate([
          var_grads.Transform(tuple), l2_loss,
          var_grads_with_l2.Transform(tuple)
      ])
      print('var_grads_vals = ', var_grads_vals)
      print('var_grads_with_l2_vals = ', var_grads_with_l2_vals)
      self.assertAllEqual(var_grads_vals.beta[0],
                          var_grads_with_l2_vals.beta[0])
      self.assertAllEqual(var_grads_vals.gamma[0],
                          var_grads_with_l2_vals.gamma[0])
      self.assertAllEqual(
          l2_loss_val, 0.5 * 0.1 * np.sum(np.square(var_grads_vals.gamma[0])))

      # With l2, gradients of be gamma are adjusted.
      self.assertAllClose(
          var_grads_with_l2_vals.gamma[1],
          var_grads_vals.gamma[1] + 0.1 * var_grads_vals.gamma[0])
      self.assertAllClose(var_grads_with_l2_vals.beta[1],
                          var_grads_vals.beta[1])

  def testAdjustGradientsWithL2Loss(self):
    with self.session(use_gpu=False):
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
      self.assertCountEqual(var_grads.keys(), ['emb', 'weight'])
      for mode in ('NestedMap', 'list'):
        if mode == 'NestedMap':
          l2_loss, var_grads_with_l2 = py_utils.AdjustGradientsWithLpLoss(
              var_grads, 0.1, p=2.0)
        else:
          l2_loss, var_grads_with_l2 = py_utils.AdjustGradientsWithLpLoss(
              var_grads.Flatten(), 0.1, p=2.0)
          var_grads_with_l2 = py_utils.Pack(var_grads, var_grads_with_l2)

        self.evaluate(tf.global_variables_initializer())
        var_grads_vals, l2_loss_val, var_grads_with_l2_vals = self.evaluate([
            var_grads.Transform(tuple), l2_loss,
            var_grads_with_l2.Transform(tuple)
        ])
        print('var_grads_vals = ', var_grads_vals)
        print('var_grads_with_l2_vals = ', var_grads_with_l2_vals)
        self.assertAllEqual(var_grads_vals.emb[0],
                            var_grads_with_l2_vals.emb[0])
        self.assertAllEqual(var_grads_vals.weight[0],
                            var_grads_with_l2_vals.weight[0])
        self.assertAllEqual(
            l2_loss_val,
            0.5 * 0.1 * (np.sum(np.square(var_grads_vals.weight[0])) +
                         np.sum(np.square(var_grads_vals.emb[0][2, :])) +
                         np.sum(np.square(var_grads_vals.emb[0][5, :]))))

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
            0.1 * np.array([[1 / 3.], [1 / 2.], [1 / 3.], [1 / 3.], [1 / 2.]]) *
            var_grads_vals.emb[0][[2, 5, 2, 2, 5], :])

  def testSkipL1Regularization(self):
    with self.session(use_gpu=False):
      beta = tf.get_variable(
          'beta',
          initializer=tf.constant(np.arange(10).reshape([1, 10]), tf.float32))
      tf.add_to_collection(py_utils.SKIP_LP_REGULARIZATION, beta)
      gamma = tf.get_variable(
          'gamma',
          initializer=tf.constant(np.arange(10).reshape([1, 10]), tf.float32))
      act = tf.constant(np.arange(10).reshape([1, 10]), tf.float32)
      pred = act * gamma + beta
      loss = tf.reduce_sum(pred)
      vmap = py_utils.NestedMap(beta=beta, gamma=gamma)
      var_grads = py_utils.ComputeGradients(loss, vmap)
      self.assertCountEqual(var_grads.keys(), ['beta', 'gamma'])
      l1_loss, var_grads_with_l1 = py_utils.AdjustGradientsWithLpLoss(
          var_grads, 0.1, p=1.0)

      self.evaluate(tf.global_variables_initializer())
      var_grads_vals, l1_loss_val, var_grads_with_l1_vals = self.evaluate([
          var_grads.Transform(tuple), l1_loss,
          var_grads_with_l1.Transform(tuple)
      ])
      print('var_grads_vals = ', var_grads_vals)
      print('var_grads_with_l1_vals = ', var_grads_with_l1_vals)
      self.assertAllEqual(var_grads_vals.beta[0],
                          var_grads_with_l1_vals.beta[0])
      self.assertAllEqual(var_grads_vals.gamma[0],
                          var_grads_with_l1_vals.gamma[0])
      self.assertAllEqual(l1_loss_val,
                          0.1 * np.sum(np.abs(var_grads_vals.gamma[0])))

  def testAdjustGradientsWithL1Loss(self):
    with self.session(use_gpu=False):
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
      self.assertCountEqual(var_grads.keys(), ['emb', 'weight'])
      l1_loss, var_grads_with_l1 = py_utils.AdjustGradientsWithLpLoss(
          var_grads, 0.1, p=1.0)

      self.evaluate(tf.global_variables_initializer())
      var_grads_vals, l1_loss_val, var_grads_with_l1_vals = self.evaluate([
          var_grads.Transform(tuple), l1_loss,
          var_grads_with_l1.Transform(tuple)
      ])
      print('var_grads_vals = ', var_grads_vals)
      print('var_grads_with_l1_vals = ', var_grads_with_l1_vals)
      self.assertAllEqual(var_grads_vals.emb[0], var_grads_with_l1_vals.emb[0])
      self.assertAllEqual(var_grads_vals.weight[0],
                          var_grads_with_l1_vals.weight[0])
      self.assertAllEqual(
          l1_loss_val, 0.1 * (np.sum(np.abs(var_grads_vals.weight[0])) +
                              np.sum(np.abs(var_grads_vals.emb[0][2, :])) +
                              np.sum(np.abs(var_grads_vals.emb[0][5, :]))))

      # With l1, gradients of emb and weight are adjusted.
      self.assertAllClose(
          var_grads_with_l1_vals.weight[1],
          var_grads_vals.weight[1] + 0.1 * var_grads_vals.weight[0])
      self.assertAllClose(var_grads_with_l1_vals.emb[1].indices,
                          var_grads_vals.emb[1].indices)

  def testSplitAndConcat(self):
    with self.session():
      # Split a Tensor.
      m3x4 = tf.constant(np.arange(12).reshape([3, 4]))
      splits = py_utils.SplitRecursively(m3x4, 2)
      self.assertLen(splits, 2)
      for split in splits:
        self.assertIsInstance(split, tf.Tensor)
      self.assertAllClose([[0, 1], [4, 5], [8, 9]], self.evaluate(splits[0]))
      self.assertAllClose([[2, 3], [6, 7], [10, 11]], self.evaluate(splits[1]))
      concatenated = py_utils.ConcatRecursively(splits)
      self.assertAllClose(self.evaluate(m3x4), self.evaluate(concatenated))

      # Split along axis 0.
      splits = py_utils.SplitRecursively(m3x4, 3, axis=0)
      self.assertLen(splits, 3)
      concatenated = py_utils.ConcatRecursively(splits, axis=0)
      self.assertAllClose(self.evaluate(m3x4), self.evaluate(concatenated))
      self.assertAllClose([[0, 1, 2, 3]], self.evaluate(splits[0]))

      # Split a list.
      list_3 = [m3x4] * 3
      splits = py_utils.SplitRecursively(list_3, 2)
      for split in splits:
        self.assertIsInstance(split, list)
      for x in splits[0]:
        self.assertAllClose([[0, 1], [4, 5], [8, 9]], self.evaluate(x))
      for x in splits[1]:
        self.assertAllClose([[2, 3], [6, 7], [10, 11]], self.evaluate(x))
      concatenated = py_utils.ConcatRecursively(splits)
      self.assertAllClose([self.evaluate(x) for x in list_3],
                          [self.evaluate(x) for x in concatenated])

      # Split a NestedMap.
      map_ab = py_utils.NestedMap(a=m3x4, b=list_3)
      splits = py_utils.SplitRecursively(map_ab, 2)
      for split in splits:
        self.assertIsInstance(split, py_utils.NestedMap)
        self.assertIsInstance(split.a, tf.Tensor)
        self.assertIsInstance(split.b, list)
      for x in splits[0].b:
        self.assertAllClose([[0, 1], [4, 5], [8, 9]], self.evaluate(x))
      concatenated = py_utils.ConcatRecursively(splits)
      self.assertAllClose(
          self.evaluate(map_ab.a), self.evaluate(concatenated.a))
      self.assertAllClose([self.evaluate(x) for x in map_ab.b],
                          [self.evaluate(x) for x in concatenated.b])

  def testFindNeeded(self):
    phs = [
        tf.zeros((), dtype=tf.float32, name='p%d' % (i + 1)) for i in range(4)
    ]
    p1, p2, p3, p4 = phs

    z1 = p1 + p2
    z2 = z1 * p3

    z1_needed = set([t for t in phs if t.name in py_utils.FindNeeded(z1)])
    z2_needed = set([t for t in phs if t.name in py_utils.FindNeeded([z2])])
    z2_p4_needed = set(
        [t for t in phs if t.name in py_utils.FindNeeded([z2, p4])])

    self.assertEqual(set([p1, p2]), z1_needed)
    self.assertEqual(set([p1, p2, p3]), z2_needed)
    self.assertEqual(set([p1, p2, p3, p4]), z2_p4_needed)

  def testArgMax(self):

    def Compute(x):
      with self.session(graph=tf.Graph()):
        x = tf.constant(x)
        y = py_utils.ArgMax(x)
        return self.evaluate([x, y])

    np.random.seed(426421)
    x, y = Compute(np.random.uniform(size=(3, 5, 10)))
    self.assertAllEqual(np.argmax(x, axis=-1), y)

    x, y = Compute(np.array([[1, 5, 3, 4, 5], [1, 5, 3, 5, 0]]))  # Has dups.
    self.assertAllEqual(np.argmax(x, axis=-1), y)

  @parameterized.named_parameters(
      ('scalar_tensor', []),
      ('small_tensor', (2, 5, 10)),
      ('big_tensor', (1000, 1000, 25)),
  )
  def testReduceRms(self, size):
    x = np.ones(size) * 0.5
    with self.session():
      x = tf.constant(x)
      y = py_utils.ReduceRms(x)
      y_val = self.evaluate(y)
    self.assertAllClose(0.5, y_val, atol=1e-12)

  def testReduceRmsNotFullyDefined(self):
    with self.session():
      x = tf.placeholder(tf.float32, shape=(None, 5))
      with self.assertRaisesRegex(ValueError,
                                  'Shape of x must be fully defined.'):
        py_utils.ReduceRms(x)

  def testPiecewiseConstant(self):
    boundaries = (1000, 2000, 3000)
    values = (1e-3, 2e-4, 3e-5, 4e-6)

    def _Eval(x):
      with self.session(use_gpu=False):
        result = py_utils.PiecewiseConstant(
            x, boundaries, values, vdtype=tf.float32)
        return self.evaluate(result)

    self.assertAlmostEqual(1e-3, _Eval(0))
    self.assertAlmostEqual(1e-3, _Eval(999))
    self.assertAlmostEqual(2e-4, _Eval(1000))
    self.assertAlmostEqual(2e-4, _Eval(1001))
    self.assertAlmostEqual(2e-4, _Eval(1999))
    self.assertAlmostEqual(3e-5, _Eval(2000))
    self.assertAlmostEqual(4e-6, _Eval(3000))
    self.assertAlmostEqual(4e-6, _Eval(4000))

  def testRepeatDim(self):
    # Create a tensor shaped [time (2), batch(2), depth(3)]
    x = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    # [batch, time, depth]
    y = tf.transpose(x, [1, 0, 2])
    # [depth, batch, time]
    z = tf.transpose(x, [2, 1, 0])
    repeat_inner_dim0 = py_utils.RepeatDim(x, 2, 0)
    repeat_inner_dim1 = py_utils.RepeatDim(y, 2, 1)
    repeat_inner_dim2 = py_utils.RepeatDim(z, 2, 2)

    with self.session(use_gpu=False):
      [repeat_inner_dim0, repeat_inner_dim1, repeat_inner_dim2] = self.evaluate(
          [repeat_inner_dim0, repeat_inner_dim1, repeat_inner_dim2])
      self.assertAllEqual(
          repeat_inner_dim0,
          [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]],
           [[7, 8, 9], [10, 11, 12]], [[7, 8, 9], [10, 11, 12]]])
      repeat_inner_dim = np.transpose(repeat_inner_dim0, [1, 0, 2])
      self.assertAllEqual(repeat_inner_dim1, repeat_inner_dim)
      repeat_inner_dim = np.transpose(repeat_inner_dim0, [2, 1, 0])
      self.assertAllEqual(repeat_inner_dim2, repeat_inner_dim)

  def testStackTensorsRecursively(self):
    with self.session(use_gpu=False):
      stacked = py_utils.StackTensorsRecursively([
          py_utils.NestedMap(
              x=tf.constant([1, 2]),
              y=py_utils.NestedMap(),
              z=py_utils.NestedMap(a=tf.constant([1, 2]),),
          ),
          py_utils.NestedMap(
              x=tf.constant([3, 4]),
              y=py_utils.NestedMap(),
              z=py_utils.NestedMap(a=tf.constant([10, 20]),),
          ),
      ])
      self.evaluate(tf.global_variables_initializer())
      self.assertAllEqual(stacked.x, tf.constant([[1, 2], [3, 4]]))
      self.assertAllEqual(stacked.z.a, tf.constant([[1, 2], [10, 20]]))

  @parameterized.named_parameters(
      ('tensordot', False),
      ('einsum', True),
  )
  def testCumSum(self, use_einsum):
    with self.session(use_gpu=False), mock.patch(
        'lingvo.core.py_utils.use_tpu', return_value=True):
      np.random.seed(12345)
      x = tf.constant(np.random.rand(2, 4, 8), dtype=tf.float32)

      # If rank is non-static, py_utils.CumSum falls back to tf.cumsum. Make
      # sure it's not the case.
      rank = py_utils.GetRank(x)
      self.assertIsInstance(rank, int)

      def _CumSum(x, *args, **kwargs):
        return py_utils.CumSum(x, *args, **kwargs, use_einsum=use_einsum)

      self.assertAllClose(
          self.evaluate(_CumSum(x, 0)), self.evaluate(tf.cumsum(x, 0)))
      self.assertAllClose(
          self.evaluate(_CumSum(x, 1)), self.evaluate(tf.cumsum(x, 1)))
      self.assertAllClose(
          self.evaluate(_CumSum(x, 2)), self.evaluate(tf.cumsum(x, 2)))
      self.assertAllClose(
          self.evaluate(_CumSum(x, -1)), self.evaluate(tf.cumsum(x, -1)))
      self.assertAllClose(
          self.evaluate(_CumSum(x, -2)), self.evaluate(tf.cumsum(x, -2)))
      self.assertAllClose(
          self.evaluate(_CumSum(x, -3)), self.evaluate(tf.cumsum(x, -3)))
      with self.assertRaises(ValueError):
        self.evaluate(_CumSum(x, -4))

  def testProjectLastDim(self):
    np.random.seed(12345)
    input_dim = 4
    output_dim = 6
    inputs_p = np.random.rand(2, 5, input_dim)
    weight_p = np.random.rand(input_dim, output_dim)

    with self.session(use_gpu=False), mock.patch(
        'lingvo.core.py_utils.use_tpu', return_value=True):
      inputs = tf.constant(inputs_p, dtype=tf.float32)
      weight = tf.constant(weight_p, dtype=tf.float32)
      outputs = py_utils.ProjectLastDim(inputs, weight, input_dim, output_dim)

      self.assertAllClose(
          self.evaluate(outputs), np.einsum('bti,io->bto', inputs_p, weight_p))

  def testAssertEvenDivide(self):
    with self.session():
      self.evaluate(py_utils.assert_even_divide(4, 2))

      with self.assertRaises(tf.errors.InvalidArgumentError):  # pylint: disable=g-error-prone-assert-raises
        self.evaluate(py_utils.assert_even_divide(4, 3))

  @mock.patch.object(tf.tpu, 'outside_compilation', autospec=True)
  def testTpuHostDecorator(self, mock_outside_compilation):

    with self.session(use_gpu=False), mock.patch(
        'lingvo.core.py_utils.use_tpu', return_value=True):

      def noop_outside_compilation(func, *args, **kwargs):  # pylint:disable=invalid-name
        return func(*args, **kwargs)

      mock_outside_compilation.side_effect = noop_outside_compilation

      @py_utils.tpu_host
      def foo(x):  # pylint:disable=invalid-name
        return bar(x) * x

      @py_utils.tpu_host
      def bar(x):  #  pylint:disable=invalid-name
        return tf.math.log(x)

      x = tf.random.uniform([])
      y = foo(x)
      unused_z = y * y

      self.assertTrue(py_utils.use_tpu())
      self.assertEqual(1, mock_outside_compilation.call_count)

  def testRemoveAssertContext(self):

    @WrapFunction(tf.float32, noinline=True)
    def Op(x):
      with py_utils.RemoveAssertContext(remove=True):
        x = py_utils.with_dependencies(
            [tf.assert_equal(0, 1, message='assert not removed')], x)
        x = py_utils.with_dependencies(
            [tf.check_ops.assert_equal(0, 1, message='assert not removed')], x)
      return x

    with self.session(use_gpu=True):

      x = tf.ones((2, 2))
      y = Op(x)
      _ = self.evaluate(y)

  def testDefaultVnParams(self):
    default_vn = py_utils.DefaultVN()
    disable_vn = py_utils.DisableVN()
    self.assertNotEqual(default_vn, disable_vn)

  def testShardedFilePatternToGlob(self):
    file_pattern = '/some/path/to/file@8'
    self.assertEqual('/some/path/to/file-?????-of-00008',
                     py_utils.ShardedFilePatternToGlob(file_pattern))

    file_pattern = '/some/path/to/file@000008'
    self.assertEqual('/some/path/to/file-?????-of-00008',
                     py_utils.ShardedFilePatternToGlob(file_pattern))

    file_pattern = '/some/path/to/file@888888'
    self.assertEqual('/some/path/to/file-?????-of-888888',
                     py_utils.ShardedFilePatternToGlob(file_pattern))

    file_pattern = '/some/path/to/file'
    self.assertEqual('/some/path/to/file',
                     py_utils.ShardedFilePatternToGlob(file_pattern))

    file_pattern = '/some/path/to/file*'
    self.assertEqual('/some/path/to/file*',
                     py_utils.ShardedFilePatternToGlob(file_pattern))

    file_pattern = '/some/path/to/file1@8,/some/path/to/file2@8'
    with self.assertRaises(ValueError):
      py_utils.ShardedFilePatternToGlob(file_pattern)

  def testComputeNceAndAuc(self):
    probs = tf.constant([[1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                         [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])
    targets = tf.constant([[1., 1., 1., 0., 0., 1.], [1., 0., 1., 0., 0., 0.]])
    mask = tf.constant([[1., 1., 1., 1., 1., 0.], [1., 1., 1., 1., 1., 1.]])
    nce, auc = py_utils.ComputeNceAndAuc(probs, targets, mask)
    with self.session():
      self.evaluate(tf.local_variables_initializer())
      nce_val, auc_val = self.evaluate([nce, auc])
      print(nce_val, auc_val)
      self.assertAllClose(0.315853, nce_val)
      self.assertAllClose(0.846309, auc_val)

  def testGetSoftmaxProbsBySeqIndices(self):
    logits = tf.constant(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
         [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]],
        dtype=tf.float32)
    indices = tf.constant([[0, 1, 2], [1, 2, 3]])
    y = py_utils.GetSoftmaxProbsBySeqIndices(logits, indices)
    with self.session():
      y_val = self.evaluate(y)
      self.assertAllClose([[0.0320586, 0.08714432, 0.2368828],
                           [0.08714432, 0.2368828, 0.6439142]], y_val)


class DeterministicDropoutTest(test_utils.TestCase):

  def testDeterministicDropoutTest(self):
    x = tf.ones([4, 6], dtype=tf.float32)
    x = py_utils.DeterministicDropout(x, keep_prob=0.7, seeds=[1234, 5678])
    with self.session():
      x_val = self.evaluate(x)
      self.assertAllClose([
          [1.0 / 0.7, 0.0000000, 0.0000000, 0.0000000, 1.0 / 0.7, 1.0 / 0.7],
          [1.0 / 0.7, 1.0 / 0.7, 1.0 / 0.7, 1.0 / 0.7, 1.0 / 0.7, 1.0 / 0.7],
          [1.0 / 0.7, 0.0000000, 0.0000000, 1.0 / 0.7, 1.0 / 0.7, 0.0000000],
          [1.0 / 0.7, 0.0000000, 0.0000000, 1.0 / 0.7, 1.0 / 0.7, 1.0 / 0.7],
      ], x_val)
      self.assertAllClose(22.85714, np.sum(x_val))
      self.assertEqual(x_val.dtype, np.float32)


class WeightedAvgTest(test_utils.TestCase):

  def testWeightedAvg(self):
    with self.session(use_gpu=False):
      losses = tf.constant([5.6, 4.6, 1.5, 3.4])
      weights = tf.constant([10, 9, 2, 8])
      loss, weight = py_utils.WeightedAvg(losses, weights)
      expected = [4.4, 29]
      actual = self.evaluate([loss, weight])
      self.assertAllClose(actual, expected)

  def testWeightedAvgOfMetrics(self):
    with self.session(use_gpu=False):
      metrics = [{
          'a': (2.0, 0.5),
          'b': (5.0, 1.5)
      }, {
          'a': (9.0, 3.0),
          'b': (4.0, 0.5)
      }]
      expected = {'a': (8.0, 3.5), 'b': (4.75, 2.0)}
      weighted_avg = py_utils.WeightedAvgOfMetrics(metrics)
      actual = self.evaluate(weighted_avg)
      self.assertDictEqual(actual, expected)

  def testConcatPerExampleTensors(self):
    with self.session(use_gpu=False):
      per_example_1 = {
          'a':
              tf.constant([[1.0, 2.0, 3.0], [12.0, 13.0, 14.0]],
                          dtype=tf.float32),
          'b':
              tf.constant([[1.5, 2.5, 3.5, 4.5]], dtype=tf.float32),
      }
      per_example_2 = {
          'a':
              tf.constant([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]],
                          dtype=tf.float32),
          'b':
              tf.constant([[3.5, 4.5, 5.5, 6.5]], dtype=tf.float32),
      }
      expected = {
          'a': [[1.0, 2.0, 3.0], [12.0, 13.0, 14.0], [3.0, 4.0, 5.0],
                [9.0, 10.0, 11.0]],
          'b': [[1.5, 2.5, 3.5, 4.5], [3.5, 4.5, 5.5, 6.5]]
      }
      stacked = py_utils.ConcatPerExampleTensors([per_example_1, per_example_2])
      actual = self.evaluate(stacked)
      self.assertAllClose(actual['a'], expected['a'])
      self.assertAllClose(actual['b'], expected['b'])
      self.assertEqual(2, len(actual))

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


class OverrideVarsFromCheckpointsTest(test_utils.TestCase):

  def _GetLeNetVarsFirstVal(self, sess):
    with tf.variable_scope('lenet5', reuse=True):
      conv0 = tf.get_variable('conv0/w/var')
      conv1 = tf.get_variable('conv1/w/var')
      fc_bias = tf.get_variable('fc/b/var')
    conv0_val, conv1_val, fc_bias_val = self.evaluate([conv0, conv1, fc_bias])
    return conv0_val[0][0][0][0], conv1_val[0][0][0][0], fc_bias_val[0]

  def testOverrideVarsFromCheckpoint(self):

    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(8372749040)
      cfg = model_registry.GetParams('image.mnist.LeNet5', 'Train')
      with cluster_factory.ForTestingWorker(mode='sync', job='trainer_client'):
        cfg.Instantiate()
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(
          # These are initialized values before overriding with checkpoint.
          self._GetLeNetVarsFirstVal(sess),
          [-0.005945, -0.036722, 0.0])
      checkpoint_path = test_helper.test_src_dir_path(
          'core/testdata/lenet_test_model')
      variable_loading_rules = [('lenet5/conv0/w/var', 'lenet5/conv0/w/var'),
                                ('lenet5/conv1/w/var', 'lenet5/conv1/w/var')]
      variable_ignore_rules = []
      py_utils.OverrideVarsFromCheckpoint(tf.all_variables(), checkpoint_path,
                                          variable_loading_rules,
                                          variable_ignore_rules)(
                                              sess)
      self.assertAllClose(
          # Now conv weights have been overwritten but fc bias has not.
          self._GetLeNetVarsFirstVal(sess),
          [0.043092, -0.024082, 0.0])

  def testOverrideVarsFromCheckpointWithIgnoreRules(self):

    with self.session(use_gpu=False) as sess:
      tf.random.set_seed(8372749040)
      cfg = model_registry.GetParams('image.mnist.LeNet5', 'Train')
      with cluster_factory.ForTestingWorker(mode='sync', job='trainer_client'):
        cfg.Instantiate()
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(
          # These are initialized values before overriding with checkpoint.
          self._GetLeNetVarsFirstVal(sess),
          [-0.005945, -0.036722, 0.0])
      checkpoint_path = test_helper.test_src_dir_path(
          'core/testdata/lenet_test_model')
      variable_loading_rules = [('lenet5/conv0/w/var', 'lenet5/conv0/w/var'),
                                ('lenet5/conv1/w/var', 'lenet5/conv1/w/var')]
      variable_ignore_rules = ['lenet5/conv1/w/var']
      py_utils.OverrideVarsFromCheckpoint(tf.all_variables(), checkpoint_path,
                                          variable_loading_rules,
                                          variable_ignore_rules)(
                                              sess)
      self.assertAllClose(
          # Now only conv0 weights have been overridden.
          self._GetLeNetVarsFirstVal(sess),
          [0.043092, -0.036722, 0.0])


def _AddOne(x):
  return None if x is None else x + type(x)(1)


class NestedMapTest(test_utils.TestCase):

  _TUPLE = collections.namedtuple('Tuple', ['x', 'y'])

  def _get_basic_test_inputs(self):
    m = py_utils.NestedMap()
    m.foo = [1, 20, [32]]
    m.bar = py_utils.NestedMap()
    m.bar.x = 100
    m.bar.y = [200, py_utils.NestedMap(z='abc')]
    return m

  def _get_advanced_test_inputs(self):
    m = py_utils.NestedMap()
    m.w = None
    m.y = (200, py_utils.NestedMap(z='abc'))
    m.x = {'foo': 1, 'bar': 'def'}
    m.z = self._TUPLE(5, 'xyz')
    m.zz = []
    return m

  def testBasic(self):
    x = py_utils.NestedMap()
    self.assertLen(x, 0)
    x['foo'] = 100
    self.assertEqual(100, x.foo)
    self.assertEqual(100, x['foo'])
    x.bar = py_utils.NestedMap({'baz': 200})
    self.assertEqual(200, x.bar.baz)
    self.assertNotIn('flatten', x)

  def testPrint(self):
    self.assertEqual(py_utils.NestedMap().DebugString(), '')

    expected = """bar.x         100
bar.y[0]      200
bar.y[1].z    abc
foo[0]        1
foo[1]        20
foo[2][0]     32"""
    m = self._get_basic_test_inputs()
    self.assertEqual(m.DebugString(), expected)

    m = self._get_advanced_test_inputs()
    res = m.DebugString()
    w, x1, x2, y, z = res.split('\n')
    self.assertEqual(w, 'w        None')
    self.assertEqual(x1, 'x.bar    def')
    self.assertEqual(x2, 'x.foo    1')
    self.assertEqual(y, "y        (200, {'z': 'abc'})")
    self.assertEqual(z, "z        Tuple(x=5, y='xyz')")

  def testTransformBasic(self):
    n = py_utils.Transform(_AddOne, py_utils.NestedMap())
    self.assertEqual(n.DebugString(), '')
    n = py_utils.NestedMap().Transform(_AddOne)
    self.assertEqual(n.DebugString(), '')

    expected = """bar.x         101
bar.y[0]      201
bar.y[1].z    abc1
foo[0]        2
foo[1]        21
foo[2][0]     33"""
    m = self._get_basic_test_inputs()
    n = py_utils.Transform(_AddOne, m)
    self.assertEqual(n.DebugString(), expected)
    n = m.Transform(_AddOne)
    self.assertEqual(n.DebugString(), expected)

    # Original has not been modified.
    expected = """bar.x         100
bar.y[0]      200
bar.y[1].z    abc
foo[0]        1
foo[1]        20
foo[2][0]     32"""
    self.assertEqual(m.DebugString(), expected)

  def testTransformAdvanced(self):
    m = self._get_advanced_test_inputs()
    original = [
        ('w', None),
        ('x.bar', 'def'),
        ('x.foo', 1),
        ('y', (200, {
            'z': 'abc'
        })),
        ('z', self._TUPLE(x=5, y='xyz')),
    ]
    self.assertEqual(m.FlattenItems(), original)

    expected = [
        ('w', None),
        ('x.bar', 'def1'),
        ('x.foo', 2),
        ('y', (201, {
            'z': 'abc1'
        })),
        ('z', self._TUPLE(x=6, y='xyz1')),
    ]
    n = py_utils.Transform(_AddOne, m)
    self.assertEqual(n.zz, [])
    self.assertNotEqual(expected, original)
    self.assertEqual(n.FlattenItems(), expected)

    with self.assertRaises(TypeError):
      m.Transform(_AddOne)

    def _AddOneIgnoreError(x):
      try:
        return _AddOne(x)
      except TypeError:
        return x

    expected = [
        ('w', None),
        ('x.bar', 'def1'),
        ('x.foo', 2),
        ('y', (200, {
            'z': 'abc'
        })),
        ('z', self._TUPLE(x=5, y='xyz')),
    ]
    n = m.Transform(_AddOneIgnoreError)
    self.assertEqual(n.zz, [])
    self.assertNotEqual(expected, original)
    self.assertEqual(n.FlattenItems(), expected)

    # Original has not been modified.
    self.assertEqual(m.FlattenItems(), original)

  def testFlattenBasic(self):
    self.assertEqual(py_utils.Flatten(py_utils.NestedMap()), [])
    self.assertEqual(py_utils.NestedMap().Flatten(), [])
    self.assertEqual(py_utils.NestedMap().FlattenItems(), [])

    expected = [100, 200, 'abc', 1, 20, 32]
    m = self._get_basic_test_inputs()
    self.assertEqual(py_utils.Flatten(m), expected)
    self.assertEqual(m.Flatten(), expected)

    expected_keys = [
        'bar.x', 'bar.y[0]', 'bar.y[1].z', 'foo[0]', 'foo[1]', 'foo[2][0]'
    ]
    self.assertEqual(m.FlattenItems(), list(zip(expected_keys, expected)))

  def testFlattenAdvanced(self):
    m = self._get_advanced_test_inputs()

    expected = [None, 'def', 1, 200, 'abc', 5, 'xyz']
    self.assertEqual(py_utils.Flatten(m), expected)

    expected = [
        None,
        'def',
        1,
        (200, {
            'z': 'abc'
        }),
        self._TUPLE(x=5, y='xyz'),
    ]
    self.assertEqual(m.Flatten(), expected)

    expected = [
        ('w', None),
        ('x.bar', 'def'),
        ('x.foo', 1),
        ('y', (200, {
            'z': 'abc'
        })),
        ('z', self._TUPLE(x=5, y='xyz')),
    ]
    self.assertEqual(m.FlattenItems(), expected)

  def testPackBasic(self):
    n = py_utils.Pack(py_utils.NestedMap(), [])
    self.assertEqual(n.DebugString(), '')
    n = py_utils.NestedMap().Pack([])
    self.assertEqual(n.DebugString(), '')

    expected = """bar.x         0
bar.y[0]      1
bar.y[1].z    2
foo[0]        3
foo[1]        4
foo[2][0]     5"""
    m = self._get_basic_test_inputs()
    n = py_utils.Pack(m, list(range(6)))
    self.assertEqual(n.DebugString(), expected)
    n = m.Pack(list(range(6)))
    self.assertEqual(n.DebugString(), expected)

    # Original has not been modified.
    expected = """bar.x         100
bar.y[0]      200
bar.y[1].z    abc
foo[0]        1
foo[1]        20
foo[2][0]     32"""
    self.assertEqual(m.DebugString(), expected)

  def testPackAdvanced(self):
    m = self._get_advanced_test_inputs()

    expected = [
        ('w', 0),
        ('x.bar', 1),
        ('x.foo', 2),
        ('y', (3, {
            'z': 4
        })),
        ('z', self._TUPLE(x=5, y=None)),
    ]
    n = py_utils.Pack(m, list(range(6)) + [None])
    self.assertEqual(n.zz, [])
    self.assertEqual(n.FlattenItems(), expected)

    expected = [('w', 0), ('x.bar', 1), ('x.foo', 2), ('y', 3), ('z', None)]
    n = m.Pack(list(range(4)) + [None])
    self.assertEqual(n.zz, [])
    self.assertEqual(n.FlattenItems(), expected)

    # Original has not been modified.
    expected = [
        ('w', None),
        ('x.bar', 'def'),
        ('x.foo', 1),
        ('y', (200, {
            'z': 'abc'
        })),
        ('z', self._TUPLE(x=5, y='xyz')),
    ]
    self.assertEqual(m.FlattenItems(), expected)

  def testIsCompatible(self):
    empty = py_utils.NestedMap()
    self.assertTrue(empty.IsCompatible(empty))
    self.assertTrue(py_utils.IsCompatible(empty, empty))
    self.assertTrue(empty.IsCompatible(py_utils.NestedMap(x=[])))
    self.assertFalse(py_utils.IsCompatible(empty, py_utils.NestedMap(x=[])))
    self.assertTrue(empty.IsCompatible(py_utils.NestedMap(x=empty)))
    self.assertFalse(py_utils.IsCompatible(empty, py_utils.NestedMap(x=empty)))
    self.assertTrue(empty.IsCompatible(py_utils.NestedMap(x={})))
    self.assertFalse(py_utils.IsCompatible(empty, py_utils.NestedMap(x={})))
    x = py_utils.NestedMap(
        a='a', b='b', c=py_utils.NestedMap(d='d', e=[1, 2, 4]))
    y = py_utils.NestedMap(a=1, b=2, c=py_utils.NestedMap(d=3, e=[10, 20, 30]))
    z = py_utils.NestedMap(
        a=1, b=[10, 20, 30], c=py_utils.NestedMap(d=3, e=['x', 'y', 'z']))
    self.assertTrue(x.IsCompatible(y))
    self.assertTrue(py_utils.IsCompatible(x, y))
    self.assertFalse(x.IsCompatible(z))
    self.assertFalse(py_utils.IsCompatible(x, z))

  def testFilter(self):
    x = py_utils.NestedMap(
        a=100,
        b=200,
        c=300,
        d=py_utils.NestedMap(foo=38, bar=192, ok=[200, 300], ko=[10, 20]))
    y = x.Filter(lambda v: v > 150)
    self.assertEqual(y.FlattenItems(), [('b', 200), ('c', 300), ('d.bar', 192),
                                        ('d.ok[0]', 200), ('d.ok[1]', 300)])
    self.assertNotIn('ko', y.d)

    y = x.Filter(lambda v: v > 500)
    self.assertLen(y.FlattenItems(), 0)

  def testFilterKeyVal(self):
    x = py_utils.NestedMap(
        a=100,
        b=200,
        c=300,
        d=py_utils.NestedMap(foo=38, bar=192, ok=[200, 300], ko=[10, 20]))
    selected = {'a', 'd.foo', 'd.ok[1]'}

    def Sel(k, _):
      return k in selected

    y = x.FilterKeyVal(Sel)
    self.assertEqual(y.FlattenItems(), [('a', 100), ('d.foo', 38),
                                        ('d.ok[0]', 300)])

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

  def testDeepCopy(self):

    class SomeObj:

      def __init__(self):
        self.foo = 'foo'

    x = py_utils.NestedMap(
        a='a',
        b='b',
        c=py_utils.NestedMap(d='d', e=[1, 2, 4], obj=SomeObj()),
        f=[],
        g={},
        h=py_utils.NestedMap(),
        i=None)
    # Perform a deep copy.
    y = copy.deepcopy(x)
    # Objects are different.
    self.assertNotEqual(id(x), id(y))

    # modify deep copy, even nested version.
    y.a = 'x'
    y.c.e[0] = 'y'
    y.c.obj.foo = 'bar'
    y.f.append(5)
    y.h.foo = 'bar'

    # x values are the originals.
    self.assertEqual('a', x.a)
    self.assertEqual(1, x.c.e[0])
    self.assertEqual([], x.f)
    self.assertEqual({}, x.g)
    self.assertLen(x.h, 0)

    # y values are updated.
    self.assertEqual('x', y.a)
    self.assertEqual('y', y.c.e[0])
    self.assertEqual([5], y.f)
    self.assertEqual({}, y.g)
    self.assertLen(y.h, 1)
    self.assertEqual('bar', y.h.foo)
    self.assertIsNone(y.i)

    # but leaf objects are the shared.
    self.assertEqual('bar', x.c.obj.foo)
    self.assertEqual('bar', y.c.obj.foo)
    self.assertEqual(id(x.c.obj), id(y.c.obj))

  def testAttrAccess(self):
    a = py_utils.NestedMap()
    a.a1 = 10
    self.assertEqual(10, a.a1)
    self.assertEqual(10, a['a1'])
    self.assertEqual(10, a.get('a1'))
    self.assertEqual(10, a.get(u'a1'))
    self.assertEqual(10, getattr(a, 'a1'))

    a[u'a1'] = 20
    self.assertEqual(20, a.a1)
    self.assertEqual(20, a['a1'])
    self.assertEqual(20, a.get('a1'))
    self.assertEqual(20, a.get(u'a1'))
    self.assertEqual(20, getattr(a, 'a1'))

    with self.assertRaisesRegex(AttributeError, 'available attributes'):
      print(a.a2)
    with self.assertRaises(KeyError):
      print(a['a2'])

    # 'get' is a reserved key.
    with self.assertRaisesRegex(AssertionError, 'is a reserved key'):
      a.get = 10
    with self.assertRaisesRegex(AssertionError, 'is a reserved key'):
      a['get'] = 10
    with self.assertRaisesRegex(AssertionError, 'is a reserved key'):
      _ = py_utils.NestedMap(get=2)

    del a.a1
    with self.assertRaisesRegex(AttributeError, 'available attributes'):
      print(a.a1)

    with self.assertRaisesRegex(AttributeError, 'available attributes'):
      del a.a2


class ReadOnlyAttrDictViewTest(test_utils.TestCase):

  def testWrapping(self):
    backing = dict()
    view = py_utils.ReadOnlyAttrDictView(backing)
    backing['test'] = 1

    self.assertEqual(1, view['test'])
    self.assertEqual(1, view.test)
    # Item assign.
    with self.assertRaises(AttributeError):
      view['test'] = 2
    self.assertEqual(1, view['test'])
    # Attr assign.
    with self.assertRaises(AttributeError):
      view.test = 2
    self.assertEqual(1, view['test'])
    # Delete attr.
    with self.assertRaises(AttributeError):
      del view.test
    self.assertEqual(1, view['test'])
    # Delete item.
    with self.assertRaises(AttributeError):
      del view['test']
    self.assertEqual(1, view['test'])


class PadPadSequenceToTest(test_utils.TestCase):

  def test2DInputs(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(3, 3), seed=123456)
      padding = tf.constant([[0, 0, 0], [0, 0, 1], [0, 1, 1]], tf.float32)
      length = 6
      new_xs, new_padding = py_utils.PadSequenceTo([x, x], padding, length, 0)

      real_xs, real_padding = self.evaluate([new_xs, new_padding])
      expected_x = [
          [0.38615, 2.975221, -0.852826, 0., 0., 0.],
          [-0.571142, -0.432439, 0.413158, 0., 0., 0.],
          [0.255314, -0.985647, 1.461641, 0., 0., 0.],
      ]
      expected_padding = [
          [0., 0., 0., 1., 1., 1.],
          [0., 0., 1., 1., 1., 1.],
          [0., 1., 1., 1., 1., 1.],
      ]
      self.assertAllClose([expected_x, expected_x], real_xs)
      self.assertAllClose(expected_padding, real_padding)

  def testSingleInput(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(3, 3), seed=123456)
      padding = tf.constant([[0, 0, 0], [0, 0, 1], [0, 1, 1]], tf.float32)
      length = 6
      new_x, new_padding = py_utils.PadSequenceTo(x, padding, length, 0)

      real_x, real_padding = self.evaluate([new_x, new_padding])
      expected_x = [
          [0.38615, 2.975221, -0.852826, 0., 0., 0.],
          [-0.571142, -0.432439, 0.413158, 0., 0., 0.],
          [0.255314, -0.985647, 1.461641, 0., 0., 0.],
      ]
      expected_padding = [
          [0., 0., 0., 1., 1., 1.],
          [0., 0., 1., 1., 1., 1.],
          [0., 1., 1., 1., 1., 1.],
      ]
      self.assertAllClose(expected_x, real_x)
      self.assertAllClose(expected_padding, real_padding)


class PadSequenceDimensionTest(test_utils.TestCase):

  def testPadSequenceDimension_2D(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(3, 3), seed=123456)
      length = 6
      padded_x = py_utils.PadSequenceDimension(x, length, 0)
      self.assertEqual(padded_x.shape.as_list(), [3, 6])
      real_x = self.evaluate(padded_x)
      expected_x = [
          [0.38615, 2.975221, -0.852826, 0., 0., 0.],
          [-0.571142, -0.432439, 0.413158, 0., 0., 0.],
          [0.255314, -0.985647, 1.461641, 0., 0., 0.],
      ]
      self.assertAllClose(expected_x, real_x)

  def testPadSequenceDimension_2D_axis0(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(3, 3), seed=123456)
      length = 6
      padded_x = py_utils.PadSequenceDimension(x, length, 0, axis=0)
      self.assertEqual(padded_x.shape.as_list(), [6, 3])
      real_x = self.evaluate(padded_x)
      expected_x = [[0.38615, 2.975221, -0.852826],
                    [-0.571142, -0.432439, 0.413158],
                    [0.255314, -0.985647, 1.461641], [0., 0., 0.], [0., 0., 0.],
                    [0., 0., 0.]]
      self.assertAllClose(expected_x, real_x)

  def testPadSequenceDimension_2D_UnknownShape(self):
    with self.session(use_gpu=False) as sess:
      shape = tf.placeholder(tf.int32)
      x = tf.random.normal(shape=shape, seed=123456)
      length = 6
      padded_x = py_utils.PadSequenceDimension(x, length, 0)
      self.assertEqual(padded_x.shape, tf.TensorShape(None))
      real_x = sess.run(padded_x, feed_dict={shape: [3, 3]})
      expected_x = [
          [0.38615, 2.975221, -0.852826, 0., 0., 0.],
          [-0.571142, -0.432439, 0.413158, 0., 0., 0.],
          [0.255314, -0.985647, 1.461641, 0., 0., 0.],
      ]
      self.assertAllClose(expected_x, real_x)

  def testPadSequenceDimension_ShortPaddingLength(self):
    x = tf.random.normal(shape=(3, 8), seed=123456)
    length = 6
    with self.assertRaisesRegex(ValueError, 'Paddings must be non-negative'):
      py_utils.PadSequenceDimension(x, length, 0)

  def testPadSequenceDimension_4D(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(2, 2, 2, 2), seed=123456)
      length = 4
      padded_x = py_utils.PadSequenceDimension(x, length, 1)
      real_x = self.evaluate(padded_x)
      expected_x = [
          [[[0.38614973, 2.97522092], [-0.85282576, -0.57114178]],
           [[-0.43243945, 0.41315758], [0.2553139, -0.98564667]],
           [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
          [[[1.46164131, 0.12003655], [-0.0986772, 0.60644895]],
           [[0.03092973, -0.96897006], [-1.27853918, -0.44018385]],
           [[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]],
      ]
      self.assertAllClose(expected_x, real_x)

  def testPadSequenceDimension_UnmatchedShape(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(2, 2, 2, 2), seed=123456)
      length = 4
      self.assertRaises(ValueError, py_utils.PadSequenceDimension, x, length, 0,
                        (32, 3, 4, 5))


class ShiftLeftTest(test_utils.TestCase):

  def testShiftLeft_2D_axis1(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(3, 3), seed=123456)
      shift_size = 2
      shifted_x = py_utils.ShiftLeft(x, shift_size, axis=1)
      real_x = self.evaluate(shifted_x)
      expected_x = [
          [-0.852826, 0., 0.],
          [0.413158, 0., 0.],
          [1.461641, 0., 0.],
      ]
      self.assertAllClose(expected_x, real_x)

  def testShiftLeft_4D_axis0(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(2, 2, 2, 2), seed=123456)
      shift_size = 1
      shifted_x = py_utils.ShiftLeft(x, shift_size, axis=0)
      real_x = self.evaluate(shifted_x)
      expected_x = [
          [[[1.46164131, 0.12003655], [-0.0986772, 0.60644895]],
           [[0.03092973, -0.96897006], [-1.27853918, -0.44018385]]],
          [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]],
      ]
      self.assertAllClose(expected_x, real_x)


class PadOrTrimToTest(test_utils.TestCase):

  def test2DConstantShapePad(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(3, 3), seed=123456)
      shape = [4, 6]
      padded_x_right = py_utils.PadOrTrimTo(x, shape, pad_val=0)
      padded_x_left = py_utils.PadOrTrimTo(
          x, shape, pad_val=0, pad_after_contents=False)
      self.assertEqual(padded_x_right.shape.as_list(), [4, 6])
      self.assertEqual(padded_x_left.shape.as_list(), [4, 6])
      real_x_right, real_x_left = self.evaluate([padded_x_right, padded_x_left])
      expected_x_right = [
          [0.38615, 2.975221, -0.852826, 0., 0., 0.],
          [-0.571142, -0.432439, 0.413158, 0., 0., 0.],
          [0.255314, -0.985647, 1.461641, 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
      ]
      self.assertAllClose(expected_x_right, real_x_right)
      expected_x_left = [
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0.38615, 2.975221, -0.852826],
          [0., 0., 0., -0.571142, -0.432439, 0.413158],
          [0., 0., 0., 0.255314, -0.985647, 1.461641],
      ]
      self.assertAllClose(expected_x_left, real_x_left)

  def test2DConstantShapeTrim(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(3, 3), seed=123456)
      shape = [1, 3]
      trimmed_x_right = py_utils.PadOrTrimTo(x, shape, pad_val=0)
      trimmed_x_left = py_utils.PadOrTrimTo(
          x, shape, pad_val=0, pad_after_contents=False)
      self.assertEqual(trimmed_x_right.shape.as_list(), [1, 3])
      self.assertEqual(trimmed_x_left.shape.as_list(), [1, 3])
      real_x_right, real_x_left = self.evaluate(
          [trimmed_x_right, trimmed_x_left])
      expected_x_right = [[0.38615, 2.975221, -0.852826]]
      self.assertAllClose(expected_x_right, real_x_right)
      expected_x_left = [[0.255314, -0.985647, 1.461641]]
      self.assertAllClose(expected_x_left, real_x_left)

  def test2DStaticShape(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(3, 3), seed=123456)
      y = tf.zeros(shape=(4, 6))
      padded_x = py_utils.PadOrTrimTo(x, y.shape, pad_val=0)
      self.assertEqual(padded_x.shape.as_list(), [4, 6])
      real_x = self.evaluate(padded_x)
      expected_x = [
          [0.38615, 2.975221, -0.852826, 0., 0., 0.],
          [-0.571142, -0.432439, 0.413158, 0., 0., 0.],
          [0.255314, -0.985647, 1.461641, 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
      ]
      self.assertAllClose(expected_x, real_x)

  def test2DDynamicShape(self):
    with self.session(use_gpu=False) as sess:
      x = tf.random.normal(shape=(3, 3), seed=123456)
      y = tf.placeholder(dtype=tf.float32)
      padded_x = py_utils.PadOrTrimTo(x, tf.shape(y), pad_val=0)
      self.assertEqual(padded_x.shape, tf.TensorShape(None))
      real_x = sess.run(padded_x, feed_dict={y: np.zeros((4, 6))})
      expected_x = [
          [0.38615, 2.975221, -0.852826, 0., 0., 0.],
          [-0.571142, -0.432439, 0.413158, 0., 0., 0.],
          [0.255314, -0.985647, 1.461641, 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
      ]
      self.assertAllClose(expected_x, real_x)

  def testDynamicTensorShapeRaises(self):
    tensor = tf.zeros(shape=[3, 2])
    shape = tf.TensorShape([3, None])
    with self.assertRaises(ValueError):
      py_utils.PadOrTrimTo(tensor, shape)

  def test4D(self):
    with self.session(use_gpu=False):
      x = tf.random.normal(shape=(2, 2, 2, 2), seed=123456)
      shape = (1, 1, 3, 3)
      padded_x = py_utils.PadOrTrimTo(x, shape, pad_val=1)
      real_x = self.evaluate(padded_x)
      expected_x = [[[
          [0.38615, 2.975221, 1.],
          [-0.852826, -0.571142, 1.],
          [1., 1., 1.],
      ]]]
      self.assertAllClose(expected_x, real_x)


class ApplyPaddingTest(test_utils.TestCase):

  def testApplyPaddingToZeroWithBroadcast(self):
    with self.session():
      y = self.evaluate(
          py_utils.ApplyPadding(
              tf.convert_to_tensor([[0.0], [1.0], [0.0]]),
              tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))
      self.assertAllClose(y, [[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])

  def testApplyPaddingToConstWithBroadcast(self):
    with self.session():
      y = self.evaluate(
          py_utils.ApplyPadding(
              tf.convert_to_tensor([[0.0], [1.0], [0.0]]),
              tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
              tf.convert_to_tensor([[1.0, 2.0], [9.0, 10.0], [5.0, 6.0]])))
      self.assertAllClose(y, [[1.0, 2.0], [9.0, 10.0], [5.0, 6.0]])

  def testApplyPaddingToZeroWithoutBroadcast(self):
    with self.session():
      y = self.evaluate(
          py_utils.ApplyPadding(
              tf.convert_to_tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
              tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))
      self.assertAllClose(y, [[1.0, 2.0], [0.0, 4.0], [5.0, 0.0]])

  def testApplyPaddingToZeroWithBroadcastArithmetic(self):
    with self.session():
      y = self.evaluate(
          py_utils.ApplyPadding(
              tf.convert_to_tensor([[0.0], [1.0], [0.0]]),
              tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
              use_select=False))
      self.assertAllClose(y, [[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])

  def testApplyPaddingToZeroWithoutBroadcastArithmetic(self):
    with self.session():
      y = self.evaluate(
          py_utils.ApplyPadding(
              tf.convert_to_tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
              tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
              use_select=False))
      self.assertAllClose(y, [[1.0, 2.0], [0.0, 4.0], [5.0, 0.0]])


class LengthsFromPaddingsTest(test_utils.TestCase):

  def testBasic(self):
    with self.session():
      paddings = np.array([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ])
      lengths = self.evaluate(
          py_utils.LengthsFromPaddings(tf.convert_to_tensor(paddings)))
      self.assertAllEqual([6, 3, 5, 0], lengths)

  def testZeroLength(self):
    with self.session():
      paddings = np.zeros([4, 0])
      lengths = self.evaluate(
          py_utils.LengthsFromPaddings(tf.convert_to_tensor(paddings)))
      self.assertAllEqual([0, 0, 0, 0], lengths)

  def testBFloat16(self):
    with self.session():
      actual_lengths = [1, 255, 256, 1024, 2048]
      paddings = 1.0 - tf.sequence_mask(
          actual_lengths, maxlen=actual_lengths[-1], dtype=tf.bfloat16)
      lengths = self.evaluate(py_utils.LengthsFromPaddings(paddings))
      self.assertAllEqual(actual_lengths, lengths)


class PaddingsFromLengthsTest(test_utils.TestCase):

  def testBasic(self):
    with self.session():
      expected = np.array([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ])
      lengths = np.array([6, 3, 5, 0])
      actual = self.evaluate(
          py_utils.PaddingsFromLengths(tf.convert_to_tensor(lengths)))
      self.assertAllEqual(expected, actual)

  def testZeroLength(self):
    with self.session():
      expected = np.zeros([4, 0])
      lengths = np.array([0, 0, 0, 0])
      actual = self.evaluate(
          py_utils.PaddingsFromLengths(tf.convert_to_tensor(lengths)))
      self.assertAllEqual(expected, actual)

  def testMaxLen(self):
    with self.session():
      expected = np.array([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ])
      lengths = np.array([6, 3, 5, 0])
      actual = self.evaluate(
          py_utils.PaddingsFromLengths(tf.convert_to_tensor(lengths), maxlen=8))
      self.assertAllEqual(expected, actual)

  def testMaxLenTooShort(self):
    with self.session():
      lengths = np.array([6, 3, 5, 0])
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError, ''):
        self.evaluate(
            py_utils.PaddingsFromLengths(
                tf.convert_to_tensor(lengths), maxlen=4))


class TrimTrailingPaddingsTest(test_utils.TestCase):

  def test2D(self):
    with self.session(use_gpu=False):
      np.random.seed(123456)
      x = np.random.normal(size=(3, 6))
      padding = np.array([
          [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
      ])
      trimmed_x, trimmed_padding = self.evaluate(
          py_utils.TrimTrailingPaddings(x, tf.convert_to_tensor(padding)))
      self.assertAllEqual(x[:, :5], trimmed_x)
      self.assertAllEqual(padding[:, :5], trimmed_padding)

  def test2D_UnknownShape(self):
    with self.session(use_gpu=False) as sess:
      shape = tf.placeholder(tf.int32)
      x = tf.random.normal(shape=shape, seed=123456)
      padding = np.array([
          [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
      ])
      actual_x, (trimmed_x, trimmed_padding) = sess.run(
          [x,
           py_utils.TrimTrailingPaddings(x, tf.convert_to_tensor(padding))],
          feed_dict={shape: [3, 6]})
      self.assertAllEqual(actual_x[:, :5], trimmed_x)
      self.assertAllEqual(padding[:, :5], trimmed_padding)

  def test4D(self):
    with self.session(use_gpu=False):
      np.random.seed(123456)
      x = np.random.normal(size=(3, 6, 3, 3))
      padding = np.array([
          [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
      ])
      trimmed_x, trimmed_padding = self.evaluate(
          py_utils.TrimTrailingPaddings(x, tf.convert_to_tensor(padding)))
      self.assertAllEqual(x[:, :5], trimmed_x)
      self.assertAllEqual(padding[:, :5], trimmed_padding)

  def testNoPadding(self):
    with self.session(use_gpu=False):
      np.random.seed(123456)
      x = np.random.normal(size=(3, 6))
      padding = np.array([
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ])
      trimmed_x, trimmed_padding = self.evaluate(
          py_utils.TrimTrailingPaddings(x, tf.convert_to_tensor(padding)))
      self.assertAllEqual(x, trimmed_x)
      self.assertAllEqual(padding, trimmed_padding)

  def testLeadingPaddingOnly(self):
    with self.session(use_gpu=False):
      np.random.seed(123456)
      x = np.random.normal(size=(3, 6))
      padding = np.array([
          [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ])
      trimmed_x, trimmed_padding = self.evaluate(
          py_utils.TrimTrailingPaddings(x, tf.convert_to_tensor(padding)))
      self.assertAllEqual(x, trimmed_x)
      self.assertAllEqual(padding, trimmed_padding)

  def testAllPadded(self):
    with self.session(use_gpu=False):
      np.random.seed(123456)
      x = np.random.normal(size=(3, 6))
      padding = np.array([
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ])
      trimmed_x, trimmed_padding = self.evaluate(
          py_utils.TrimTrailingPaddings(x, tf.convert_to_tensor(padding)))
      self.assertAllEqual([3, 1], trimmed_x.shape)
      self.assertAllEqual(padding[:, :1], trimmed_padding)


class ReversePaddedSequenceTest(test_utils.TestCase):

  def testReversePaddedSequence(self):
    with self.session(use_gpu=False):
      # inputs is [seq_length, batch_size, input_dim] = [4, 3, 2]
      # The length of each batch is [2, 3, 4]
      inputs = tf.constant(
          [[[1, 2], [3, 4], [5, 6]], [[11, 12], [13, 14], [15, 16]],
           [[0, 0], [23, 24], [25, 26]], [[0, 0], [0, 0], [35, 36]]],
          dtype=tf.float32)
      paddings = tf.constant(
          [[[0], [0], [0]], [[0], [0], [0]], [[1], [0], [0]], [[1], [1], [0]]],
          dtype=tf.float32)
      actual_output = self.evaluate(
          py_utils.ReversePaddedSequence(inputs, paddings))
      expected_output = np.array([[[11, 12], [23, 24], [35, 36]],
                                  [[1, 2], [13, 14], [25, 26]],
                                  [[0, 0], [3, 4], [15, 16]],
                                  [[0, 0], [0, 0], [5, 6]]]).astype('float32')
      self.assertAllClose(expected_output, actual_output)


class ConcatenatePaddedSequencesTest(test_utils.TestCase):

  def _ComputeFloatOutputAndVerify(self,
                                   input0,
                                   input1,
                                   seq_lens0,
                                   seq_lens1,
                                   tranpose_input=False):
    with self.session(use_gpu=False):
      expected_output_seq_lens = seq_lens0 + seq_lens1
      batch_size, input0_seq_dim = input0.shape
      input1_seq_dim = input1.shape[1]
      padding0 = 1.0 - tf.sequence_mask(
          seq_lens0, maxlen=input0_seq_dim, dtype=tf.float32)
      padding1 = 1.0 - tf.sequence_mask(
          seq_lens1, maxlen=input1_seq_dim, dtype=tf.float32)

      if tranpose_input:
        seq_dim = 0
        tf_input0 = tf.constant(np.transpose(input0))
        tf_input1 = tf.constant(np.transpose(input1))
        tf_padding0 = tf.transpose(padding0)
        tf_padding1 = tf.transpose(padding1)
      else:
        seq_dim = 1
        tf_input0 = tf.constant(input0)
        tf_input1 = tf.constant(input1)
        tf_padding0 = padding0
        tf_padding1 = padding1

      actual_outputs = self.evaluate(
          py_utils.ConcatenatePaddedSequences(
              tf_input0,
              tf_input1,
              padding0=tf_padding0,
              padding1=tf_padding1,
              seq_dim=seq_dim))

      if tranpose_input:
        actual_outputs = (np.transpose(actual_outputs[0]),
                          np.transpose(actual_outputs[1]))

      for batch in range(batch_size):
        expected_output = np.concatenate((input0[batch, :seq_lens0[batch]],
                                          input1[batch, :seq_lens1[batch]]))
        self.assertAllClose(
            expected_output,
            actual_outputs[0][batch, :expected_output_seq_lens[batch]])
        expected_padding = np.ones(
            (input0_seq_dim + input1_seq_dim,)).astype('float32')
        expected_padding[:(seq_lens0[batch] + seq_lens1[batch])] = 0.0
        self.assertAllClose(expected_padding, actual_outputs[1][batch, :])

  def testConcatenateFloatFeatures(self):
    input0 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).astype('float32')
    seq_lens0 = np.array([2, 3])
    input1 = np.array([[11, 12, 13, 14, 15, 16], [16, 17, 18, 19, 20,
                                                  21]]).astype('float32')
    seq_lens1 = np.array([4, 5])
    batch_size, input0_seq_dim = input0.shape
    input1_seq_dim = input1.shape[1]

    no_padding_seq_lens0 = np.array([input0_seq_dim] * batch_size)
    no_padding_seq_lens1 = np.array([input1_seq_dim] * batch_size)

    self._ComputeFloatOutputAndVerify(input0, input1, no_padding_seq_lens0,
                                      no_padding_seq_lens1, False)
    self._ComputeFloatOutputAndVerify(input0, input1, no_padding_seq_lens0,
                                      no_padding_seq_lens1, True)

    self._ComputeFloatOutputAndVerify(input0, input1, seq_lens0, seq_lens1,
                                      False)
    self._ComputeFloatOutputAndVerify(input0, input1, seq_lens0, seq_lens1,
                                      True)


class RetryTest(test_utils.TestCase):

  def testRetry(self):
    max_retries = 5

    @py_utils.Retry(max_retries=max_retries)
    def Foo(state):
      tf.logging.error('foo retried %s', state)
      state['count'] += 1
      raise ValueError('test')

    try:
      state = {'count': 0, 'msg': 'test'}
      Foo(state)
    except Exception as e:  # pylint: disable=broad-except
      tf.logging.error('%s', e)

    self.assertEqual(1 + max_retries, state['count'])


class MixByWeightTest(test_utils.TestCase):

  def testMixByWeight(self):
    var_a = tf.get_variable('a', trainable=False, initializer=0)
    var_b = tf.get_variable('b', trainable=False, initializer=0)

    with self.session():
      self.evaluate(tf.global_variables_initializer())

      def _AddFn(var):
        return lambda: tf.assign_add(var, 1)

      op, _ = py_utils.MixByWeight([_AddFn(var_a), _AddFn(var_b)], [0.7, 0.3],
                                   seed=12345)
      for _ in range(100):
        self.evaluate(op)
      a, b = self.evaluate([var_a, var_b])
      self.assertEqual(100, a + b)
      self.assertGreater(a, 50)
      self.assertLess(b, 50)

  def testMixByWeightWithDynamicWeights(self):
    var_a = tf.get_variable('a', trainable=False, initializer=0)
    var_b = tf.get_variable('b', trainable=False, initializer=0)
    var_w = tf.get_variable('w', trainable=False, dtype=tf.float32, shape=[2])

    with self.session():
      self.evaluate(tf.global_variables_initializer())

      def _AddFn(var):
        return lambda: tf.assign_add(var, 1)

      op, _ = py_utils.MixByWeight([_AddFn(var_a), _AddFn(var_b)], var_w)

      # all weight goes to 'a'
      self.evaluate([tf.assign(var_w, [1.0, 0.0])])
      for _ in range(10):
        self.evaluate(op)
      a, b = self.evaluate([var_a, var_b])
      self.assertEqual(10, a)
      self.assertEqual(0, b)

      # all weight goes to 'b'
      self.evaluate([tf.assign(var_w, [0.0, 1.0])])
      for _ in range(10):
        self.evaluate(op)
      a, b = self.evaluate([var_a, var_b])
      self.assertEqual(10, a)
      self.assertEqual(10, b)

  def testMixByWeightAndBpropType(self):
    var_a = tf.get_variable('a', trainable=False, initializer=0)
    var_b = tf.get_variable('b', trainable=False, initializer=0)

    with self.session():
      self.evaluate(tf.global_variables_initializer())

      def _AddFn(var):
        return lambda: tf.assign_add(var, 1)

      op, bprop = py_utils.MixByWeight(
          [_AddFn(var_a), _AddFn(var_b)], [1.0, 0.0])
      for _ in range(10):
        self.evaluate(op)
      bprop_v, a, b = self.evaluate([bprop, var_a, var_b])
      self.assertEqual(10, a)
      self.assertEqual(0, b)
      self.assertAllClose(np.array([1, 0]), np.squeeze(bprop_v))

      op, bprop = py_utils.MixByWeight(
          [_AddFn(var_a), _AddFn(var_b)], [0.0, 1.0])
      for _ in range(10):
        self.evaluate(op)
      bprop_v, a, b = self.evaluate([bprop, var_a, var_b])
      self.assertEqual(10, a)
      self.assertEqual(10, b)
      self.assertAllClose(np.array([0, 1]), np.squeeze(bprop_v))


class SequencesToDebugStrings(test_utils.TestCase):

  def testSequencesToDebugStrings(self):
    with self.session():
      self.assertAllEqual([b'[1 2 3]', b'[100 200]'],
                          self.evaluate(
                              py_utils.SequencesToDebugStrings(
                                  tf.constant([[1, 2, 3], [100, 200, 300]],
                                              dtype=tf.int32),
                                  tf.constant([3, 2], dtype=tf.int32))))


class StepSeedTest(test_utils.TestCase, parameterized.TestCase):

  def _testStepSeedHelper(self, sess, step_fn):
    state0 = py_utils.NestedMap(
        input=tf.constant(0, dtype=tf.int64),
        seed_pair=tf.zeros(2, dtype=tf.int64))
    inputs = py_utils.NestedMap(input=tf.range(10, dtype=tf.int64))

    p = base_layer.BaseLayer.Params().Set(name='test')
    accumulated_states, _ = recurrent.Recurrent(p.Instantiate().theta, state0,
                                                inputs, step_fn)

    self.evaluate(tf.global_variables_initializer())
    accumulated_states = accumulated_states.Pack(
        self.evaluate(accumulated_states.Flatten()))
    self.assertAllEqual(np.arange(10), accumulated_states.input)
    global_steps, step_seeds = zip(*accumulated_states.seed_pair)
    self.assertAllEqual(np.zeros(10), global_steps)
    self.assertAllEqual(np.arange(10), step_seeds - step_seeds[0])
    return step_seeds[0]

  def testStepSeed(self):
    p = base_layer.BaseLayer.Params()

    def RecurrentStep(unused_theta, unused_state0, inputs):
      state1 = py_utils.NestedMap()
      state1.input = inputs.input
      state1.seed_pair = py_utils.GenerateStepSeedPair(p)
      return state1, py_utils.NestedMap()

    with self.session(graph=tf.Graph()) as sess:
      step_seed = self._testStepSeedHelper(sess, RecurrentStep)
      # Second recurrent inside the same graph has different step_seeds.
      step_seed2 = self._testStepSeedHelper(sess, RecurrentStep)
      self.assertNotEqual(step_seed, step_seed2)

    # After a reset, the step_seeds are the same even with a slightly
    # different RecurrentStep function.
    def RecurrentStep2(theta, state0, inputs):
      with tf.control_dependencies([tf.no_op()]):
        return RecurrentStep(theta, state0, inputs)

    with self.session(graph=tf.Graph()) as sess:
      step_seed3 = self._testStepSeedHelper(sess, RecurrentStep2)
      step_seed4 = self._testStepSeedHelper(sess, RecurrentStep2)
      self.assertEqual(step_seed, step_seed3)
      self.assertEqual(step_seed2, step_seed4)

    with self.session(graph=tf.Graph()) as sess:
      # But a different name_scope changes it.
      with tf.name_scope('test'):
        step_seed5 = self._testStepSeedHelper(sess, RecurrentStep2)
        step_seed6 = self._testStepSeedHelper(sess, RecurrentStep2)
        self.assertEqual(step_seed, step_seed5)
        self.assertNotEqual(step_seed2, step_seed6)

  @parameterized.named_parameters(('Base', False), ('Tpu', True))
  def testGenerateStepSeedPair(self, use_tpu):
    with self.session(use_gpu=False), mock.patch(
        'lingvo.core.py_utils.use_tpu', return_value=use_tpu):
      tf.random.set_seed(12345678)
      seed_dtype = tf.int32 if use_tpu else tf.int64
      p = base_layer.BaseLayer.Params()
      global_step = py_utils.GetGlobalStep()
      global_step = tf.assign_add(global_step, 10)
      with tf.control_dependencies([global_step]):
        seed1 = py_utils.GenerateStepSeedPair(p) + tf.cast(
            [global_step, global_step + 1], seed_dtype)
        seed2 = py_utils.GenerateStepSeedPair(p, global_step)
        self.evaluate(tf.global_variables_initializer())
        seeds = self.evaluate([seed1, seed2])
        self.assertAllClose(seeds[0], seeds[1])


class WeightParamsTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    with contextlib.ExitStack() as stack:
      self._store = stack.enter_context(py_utils.VariableStore())
      self.addCleanup(stack.pop_all().close)

  def testShapeModification(self):
    """Tests that WeightParams.shape can be modified."""
    pc = py_utils.WeightParams([20, 30],
                               py_utils.WeightInit.UniformPositive(1.0),
                               tf.float32)
    pc.shape = [10, 30]
    var = py_utils.CreateVariable('var', pc)
    self.assertEqual(var.shape, [10, 30])


class WeightInitTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    with contextlib.ExitStack() as stack:
      self._store = stack.enter_context(py_utils.VariableStore())
      self.addCleanup(stack.pop_all().close)

  def testModification(self):
    """Tests that WeightInit cannot be modified."""
    w_init = py_utils.WeightInit.UniformPositive(1.0)
    with self.assertRaisesRegex(TypeError, 'immutable'):
      w_init.scale = 2.0

  def testUniformPositive(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      pc = py_utils.WeightParams([20, 30],
                                 py_utils.WeightInit.UniformPositive(1.0),
                                 tf.float32)
      var = py_utils.CreateVariable('var', pc)
      self.evaluate(tf.global_variables_initializer())
      var_v = self.evaluate(var)
      self.assertTrue(np.all(var_v >= 0.0))
      self.assertTrue(np.all(var_v <= 1.0))

  def testCategory(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      pc = py_utils.WeightParams([30, 30], py_utils.WeightInit.Category(3),
                                 tf.float32)
      var = py_utils.CreateVariable('var', pc)
      self.evaluate(tf.global_variables_initializer())
      var_v = self.evaluate(var)
      self.assertEqual({0.0, 1.0, 2.0}, set(np.unique(var_v)))

  def testKaimingUniformRelu(self):
    with self.session(use_gpu=False):
      pc = py_utils.WeightParams(
          [2, 10, 30], py_utils.WeightInit.KaimingUniformFanInRelu(1.0),
          tf.float32)
      var = py_utils.CreateVariable('var', pc)
      self.evaluate(tf.global_variables_initializer())
      var_v = self.evaluate(var)
      # With Relu initialization, uniform bounds are
      # sqrt(3) * sqrt(2) / sqrt(fan_in)
      bound = np.sqrt(3.) * np.sqrt(2.) / np.sqrt(20)
      self.assertTrue(np.all(var_v >= -bound))
      self.assertTrue(np.all(var_v <= bound))

  def testKaimingUniformLeakyRelu(self):
    with self.session(use_gpu=False):
      pc = py_utils.WeightParams(
          [2, 10, 30], py_utils.WeightInit.KaimingUniformFanInLeakyRelu(),
          tf.float32)
      var = py_utils.CreateVariable('var', pc)
      self.evaluate(tf.global_variables_initializer())
      var_v = self.evaluate(var)
      # With LeakyRelu initialization, uniform bounds are
      # sqrt(3) * sqrt(2 / (1 + scale**2)) / sqrt(fan_in)
      #
      # scale = sqrt(5) by default.
      bound = np.sqrt(3.) * np.sqrt(2. / 6.) / np.sqrt(20)
      self.assertTrue(np.all(var_v >= -bound))
      self.assertTrue(np.all(var_v <= bound))


class RNNCellStateInitTest(test_utils.TestCase):

  def testZeros(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      zero_state = py_utils.InitRNNCellState(
          [2, 3], init=py_utils.RNNCellStateInit.Zeros(), dtype=tf.float32)
      self.evaluate(tf.global_variables_initializer())
      zero_state_v = self.evaluate(zero_state)
      expected_zero_state = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      self.assertAllClose(zero_state_v, expected_zero_state)

  def testRandomNormal(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      zero_state = py_utils.InitRNNCellState(
          [2, 3],
          init=py_utils.RNNCellStateInit.RandomNormal(seed=12345),
          dtype=tf.float32)
      self.evaluate(tf.global_variables_initializer())
      zero_state_v = self.evaluate(zero_state)
      expected_zero_state = [[1.621003, -1.097501, 0.493424],
                             [-1.048426, 2.73048, 0.091445]]
      self.assertAllClose(zero_state_v, expected_zero_state)

  @flagsaver.flagsaver(stateless_vars_init=True)
  def testRandomNormalStatelessVarsInit(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      zero_state = py_utils.InitRNNCellState(
          [2, 3],
          name='RNNCellStateInit',
          init=py_utils.RNNCellStateInit.RandomNormal(seed=12345),
          dtype=tf.float32)
      self.evaluate(tf.global_variables_initializer())
      zero_state_v = self.evaluate(zero_state)
      expected_zero_state = [[-0.887855, 0.993745, -0.439152],
                             [0.312563, 0.923067, -1.952364]]
      self.assertAllClose(zero_state_v, expected_zero_state)
      zero_state_v_bis = self.evaluate(zero_state)
      self.assertAllClose(zero_state_v_bis, expected_zero_state)

  def testRandomNormalInEval(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      zero_state = py_utils.InitRNNCellState(
          [2, 3],
          init=py_utils.RNNCellStateInit.RandomNormal(seed=12345),
          dtype=tf.float32,
          is_eval=True)
      self.evaluate(tf.global_variables_initializer())
      zero_state_v = self.evaluate(zero_state)
      expected_zero_state = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      self.assertAllClose(zero_state_v, expected_zero_state)


class RematerializeFnTest(test_utils.TestCase):

  def testRandomNormal(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(12345678)
      a = tf.random.normal([2, 3])
      b = tf.random.normal([3, 4])

      def Fn(a, b):
        c = tf.matmul(a, b)
        d = tf.nn.sigmoid(c)
        e = tf.nn.tanh(c)
        return d, e

      d1, e1 = Fn(a, b)
      d2, e2 = py_utils.RematerializeFn(Fn, a, b)
      self.assertEqual(d2.shape.as_list(), [2, 4])
      self.assertEqual(e2.shape.as_list(), [2, 4])
      da1, db1 = tf.gradients([d1, e1], [a, b])
      da2, db2 = tf.gradients([d2, e2], [a, b])
      self.evaluate(tf.global_variables_initializer())
      v1, v2, v3, v4 = self.evaluate([da1, db1, da2, db2])
      self.assertAllEqual(v1, v3)
      self.assertAllEqual(v2, v4)


def WrapFunction(*dtypes, noinline=False):
  """Wrap a python function as a Defun or tf.function."""
  if py_utils._UseTfFunction():

    def Decorated(fn):

      @tf.function(
          input_signature=[tf.TensorSpec(shape=None, dtype=t) for t in dtypes])
      def Fn(*args):
        return fn(*args)

      return Fn.get_concrete_function()

    return Decorated
  else:
    return tf.Defun(*dtypes, noinline=noinline)


class StatefulRandomOpsInDefunTest(test_utils.TestCase, parameterized.TestCase):

  def testFunctionWithStatelessOp(self):

    @WrapFunction()
    def FunctionWithStatelessOp():
      return tf.constant(42.0)

    self.assertAllEqual(
        [], py_utils.StatefulRandomOpsInDefun(FunctionWithStatelessOp))

  def testFunctionWithStatefulOp(self):

    @WrapFunction()
    def FunctionWithStatefulOp():
      return tf.random.uniform([100], maxval=10, dtype=tf.int32, name='rand_10')

    self.assertAllEqual(
        ['rand_10'], py_utils.StatefulRandomOpsInDefun(FunctionWithStatefulOp))

  def testFunctionWithStatelessFunctionCall(self):

    @WrapFunction()
    def FunctionWithStatelessOp():
      return tf.constant(42.0)

    @WrapFunction()
    def FunctionWithStatelessFunctionCall():
      return FunctionWithStatelessOp()

    self.assertAllEqual(
        [],
        py_utils.StatefulRandomOpsInDefun(FunctionWithStatelessFunctionCall))

  def testFunctionWithStatefulFunctionCall(self):

    @WrapFunction()
    def FunctionWithStatefulOp():
      return tf.random.uniform([100], maxval=10, dtype=tf.int32, name='rand_10')

    @WrapFunction()
    def FunctionWithStatefulFunctionCall():
      with tf.name_scope('func'):
        return FunctionWithStatefulOp()

    self.assertAllEqual(
        ['rand_10'],
        py_utils.StatefulRandomOpsInDefun(FunctionWithStatefulFunctionCall))

  def testFunctionWithStatefulFunctionalWhile(self):

    @WrapFunction()
    def FunctionWithStatefulFunctionalWhile():

      @WrapFunction(tf.float32, tf.int32)
      def Cond(result, i):
        del result
        return tf.less(i, 4)

      @WrapFunction(tf.float32, tf.int32)
      def Body(result, i):
        with tf.name_scope('body'):
          return (result + tf.random.uniform(tf.shape(result), name='rand'),
                  i + 1)

      return functional_ops.While([tf.zeros([2, 2]), 0],
                                  cond=Cond,
                                  body=Body,
                                  name='while')

    self.assertAllEqual(
        ['body/rand/RandomUniform'],
        py_utils.StatefulRandomOpsInDefun(FunctionWithStatefulFunctionalWhile))

  def testFunctionWithStatefulFunctionalIf(self):

    @WrapFunction()
    def FunctionWithStatefulFunctionalIf():

      @WrapFunction(tf.float32)
      def ThenFn(x):
        return tf.abs(x)

      @WrapFunction(tf.float32)
      def ElseFn(x):
        with tf.name_scope('else'):
          return tf.random.uniform(tf.shape(x), name='rand')

      return functional_ops.If(
          tf.greater(tf.eye(2), 0.5), [tf.eye(2)], ThenFn, ElseFn)

    self.assertAllEqual(
        ['else/rand/RandomUniform'],
        py_utils.StatefulRandomOpsInDefun(FunctionWithStatefulFunctionalIf))

  def testFunctionWithStatefulFunctionalFor(self):

    @WrapFunction()
    def FunctionWithStatefulFunctionalFor():

      @WrapFunction(tf.float32)
      def Body(result):
        with tf.name_scope('body'):
          return [
              result +
              tf.random.uniform(tf.shape(result), name='rand_uniform') +
              tf.random.poisson(
                  shape=tf.shape(result), lam=[0.5, 1.5], name='rand_poisson')
          ]

      return functional_ops.For(
          start=0, limit=4, delta=1, inputs=[tf.eye(2)], body=Body, name='for')

    self.assertAllEqual([
        'body/rand_poisson/RandomPoissonV2',
        'body/rand_uniform/RandomUniform',
    ],
                        sorted(
                            py_utils.StatefulRandomOpsInDefun(
                                FunctionWithStatefulFunctionalFor)))

  def testFunctionWithStatelessFunctionalFor(self):

    @WrapFunction()
    def FunctionWithStatelessFunctionalFor():

      @WrapFunction(tf.float32)
      def Body(result):
        return [
            result +
            tf.random.stateless_normal(tf.shape(result), seed=tf.stack([0, 1]))
        ]

      return functional_ops.For(
          start=0, limit=4, delta=1, inputs=[tf.eye(2)], body=Body)

    self.assertAllEqual(
        [],
        py_utils.StatefulRandomOpsInDefun(FunctionWithStatelessFunctionalFor))


class RecordFormatTest(test_utils.TestCase):

  def testRecordFormatFromFilePattern(self):
    record_format, path = py_utils.RecordFormatFromFilePattern(
        'tfrecord:/path/to/bar')
    self.assertEqual(record_format, 'tfrecord')
    self.assertEqual(path, '/path/to/bar')

    record_format, path = py_utils.RecordFormatFromFilePattern(
        'custom_FORMAT:/path/to/baz')
    self.assertEqual(record_format, 'custom_FORMAT')
    self.assertEqual(path, '/path/to/baz')


class ReadFileLinesTest(test_utils.TestCase):

  def testReadFileLines(self):
    contents = [
        'hello',
        'world',
        'foo',
        'bar',
    ]
    outpath = os.path.join(tf.test.get_temp_dir(), 'test.txt')
    with tf.io.gfile.GFile(outpath, 'w') as f:
      f.write('\n'.join(contents))

    lines = [line.strip() for line in py_utils.ReadFileLines(outpath)]
    self.assertAllEqual(lines, contents)

  def testReadFilesLinesFromPackage(self):
    # py_utils.py is at lingvo/core relative to the working
    # directory, so it will load from the lingvo package instead
    lines = py_utils.ReadFileLines('core/py_utils.py')
    self.assertIsNotNone(lines)

  def testReadFileLinesWithInvalidFile(self):
    path = os.path.join(tf.test.get_temp_dir(), 'fake.txt')

    with self.assertRaises(tf.errors.NotFoundError):
      py_utils.ReadFileLines(path)


class FocalLossTest(parameterized.TestCase, test_utils.TestCase):

  def _testNpFL(self, logits, labels, alpha, gamma):
    self.assertEqual(logits.shape, labels.shape)
    shape = labels.shape
    logits = logits.reshape([-1])
    labels = labels.reshape([-1])

    def _Sigmoid(x):
      return 1.0 / (1.0 + np.exp(-x))

    def _CrossEntropy(prob, label):
      if label > 0:
        return -np.log(prob)
      else:
        return -np.log(1 - prob)

    probabilities = _Sigmoid(logits)
    ans = np.empty(probabilities.shape)
    for i, (l, p) in enumerate(zip(labels, probabilities)):
      ce = _CrossEntropy(p, l)
      pt = (l * p) + ((1 - l) * (1 - p))
      if alpha is not None:
        ce *= (l * alpha) + ((1 - l) * (1 - alpha))
      if gamma is not None:
        ce *= np.power(1 - pt, gamma)
      ans[i] = ce
    return ans.reshape(shape)

  def _testTfFL(self, logits, labels, alpha, gamma):
    with self.session() as sess:
      x = tf.placeholder(tf.float32)
      y = tf.placeholder(tf.float32)
      z = py_utils.SigmoidCrossEntropyFocalLoss(x, y, alpha, gamma)
      return sess.run(z, feed_dict={x: logits, y: labels})

  def testSigmoidCrossEntropyFocalLoss(self):
    logits = np.random.normal(scale=10, size=(2, 3, 5))
    labels = np.floor(np.random.uniform(size=(2, 3, 5)) + 0.2)
    for (alpha, gamma) in [(None, None), (0.25, 2), (0.1, 0), (1, 5)]:
      np_result = self._testNpFL(logits, labels, alpha, gamma)
      tf_lingvo_result = self._testTfFL(logits, labels, alpha, gamma)
      self.assertAllClose(np_result, tf_lingvo_result)

  def _testNpSCEFL(self, logits, labels, alpha, gamma):
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    shape = probs.shape[:-1]
    probs = probs.reshape([-1, probs.shape[-1]])
    ans = np.empty(probs.shape[:-1])

    if labels.shape != logits.shape:
      # convert labels to class probabilities
      label_probs = np.zeros(probs.shape)
      label_probs[np.arange(labels.size), labels.reshape([-1])] = 1.0
    else:
      label_probs = labels.reshape([-1, labels.shape[-1]])
    for i, (lp, p) in enumerate(zip(label_probs, probs)):
      ce = lp * -np.log(p)
      if alpha is not None:
        ce *= alpha
      if gamma is not None:
        ce *= np.power(1 - p, gamma)
      ans[i] = ce.sum()
    ans = ans.reshape(shape)
    return ans

  def _testTfSCEFLLabelIds(self, logits, labels, alpha, gamma):
    with self.session() as sess:
      x = tf.placeholder(tf.float32)
      y = tf.placeholder(tf.int32)
      z = py_utils.SoftmaxCrossEntropyFocalLoss(
          x, label_ids=y, alpha=alpha, gamma=gamma)
      return sess.run(z, feed_dict={x: logits, y: labels})

  def _testTfSCEFLLabelProbs(self, logits, labels, alpha, gamma):
    with self.session() as sess:
      x = tf.placeholder(tf.float32)
      y = tf.placeholder(tf.float32)
      z = py_utils.SoftmaxCrossEntropyFocalLoss(
          x, label_probs=y, alpha=alpha, gamma=gamma)
      return sess.run(z, feed_dict={x: logits, y: labels})

  def testSoftmaxCrossEntropyFocalLoss(self):
    num_classes = 7
    logits = np.random.normal(scale=10, size=(2, 3, 5, num_classes))
    label_ids = np.random.randint(num_classes, size=(2, 3, 5))
    label_probs = np.random.uniform(size=(2, 3, 5, num_classes))
    label_probs /= label_probs.sum(axis=-1, keepdims=True)
    for (alpha, gamma) in [
        (None, None),
        (np.random.uniform(size=[num_classes]).astype(np.float32), 2),
        (np.random.uniform(size=[num_classes]).astype(np.float32), 0),
        (np.random.uniform(size=[num_classes]).astype(np.float32), 5)
    ]:
      self.assertAllClose(
          self._testNpSCEFL(logits, label_ids, alpha, gamma),
          self._testTfSCEFLLabelIds(logits, label_ids, alpha, gamma))
      self.assertAllClose(
          self._testNpSCEFL(logits, label_probs, alpha, gamma),
          self._testTfSCEFLLabelProbs(logits, label_probs, alpha, gamma))

  @parameterized.named_parameters(
      ('NoStopGradientOnFocalLossCoefficient', False),
      ('StopGradientOnFocalLossCoefficient', True),
  )
  def testSoftmaxCrossEntropyFocalLossGradients(self,
                                                stop_gradient_on_coefficient):
    label_ids = [0, 1]
    logits = tf.constant([[1e30, 0., 0.], [0., -1e30, 0.]], dtype=tf.float32)
    loss = py_utils.SoftmaxCrossEntropyFocalLoss(
        logits,
        label_ids,
        label_probs=None,
        gamma=.5,
        stop_gradient_on_focal_loss_coefficient=stop_gradient_on_coefficient)
    dlogits, = tf.gradients(ys=loss, xs=[logits])
    with self.session():
      dlogits = self.evaluate(dlogits)
      if stop_gradient_on_coefficient:
        self.assertAllClose([[0., 0., 0.], [0.5, -1., 0.5]], dlogits)
      else:
        # Gradients will contain nan.
        self.assertTrue(any(math.isnan(x)) for x in dlogits.flatten())


class UniformSamplerTest(test_utils.TestCase):

  def testUniformSamplerSamples(self):
    sampler = py_utils.UniformSampler(5)
    for i in range(5):
      sampler.Add(i)
    # Up to the total number of samples, no sampling is performed.
    self.assertEqual([0, 1, 2, 3, 4], sampler.samples)

  def testUniformSampler(self):
    # Run a bunch of trials sampling 10 items out of 100 ids.
    np.random.seed(123456)
    state_space = 100
    num_samples = 10
    num_trials = 10000
    counts = np.zeros([state_space])
    for _ in range(num_trials):
      sampler = py_utils.UniformSampler(num_samples)
      # Add an element for each item in the state space.
      for i in range(state_space):
        sampler.Add(i)
      samples = sampler.samples
      self.assertEqual(num_samples, len(samples))
      for value in samples:
        counts[value] += 1
    distribution = counts / np.sum(counts)

    # We expect that over the course of many trials, each item in
    # the state space gets selected roughly an equal number of times,
    # implying that the reservoir sampler is not biased based on the order
    # in which items were added to the sampler.
    self.assertGreater(min(distribution), 0.009)
    self.assertLess(max(distribution), 0.011)


class FromGlobalTest(test_utils.TestCase):

  def testAccessAssertFlagWhenUnparsed(self):
    tf.flags.FLAGS.unparse_flags()
    # Accessing the flag value directly fails.
    with self.assertRaises(tf.flags._exceptions.UnparsedFlagAccessError):
      result = FLAGS.enable_asserts
    result = py_utils._FromGlobal('enable_asserts')
    # Default value of this flag is True.
    self.assertTrue(result)
    # Reparse args.
    tf.flags.FLAGS(sys.argv)


def FunctionTestParameters(test_fn):
  suffix = '_function' if py_utils._UseTfFunction() else '_defun'
  decorator = parameterized.named_parameters(
      (suffix, False),
      (suffix + '_bakasfunction', True),
  )
  return decorator(test_fn)


class FunctionTest(test_utils.TestCase, parameterized.TestCase):

  def testNoInputs(self):
    with self.session():

      @py_utils.Function()
      def Fwd():
        return tf.constant(1.0)

      self.assertEqual(tf.float32, Fwd.output_dtypes)
      ys = Fwd()
      self.assertEqual(1.0, self.evaluate(ys))

  @FunctionTestParameters
  def testScalarInput(self, bak_as_function):
    with self.session():
      sig = tf.TensorSpec(None, tf.float32)

      def Bak(x, y, dy):
        del y
        return 4 * x * dy

      @py_utils.Function(fwd_sig=sig)
      def Fwd(x):
        return x * x * 2

      @py_utils.Function(fwd_sig=sig, bak=Bak, bak_as_function=bak_as_function)
      def FwdWithBak(x):
        return x * x * 2

      x = tf.constant(3.0)
      for fwd in [Fwd, FwdWithBak]:
        self.assertEqual(tf.float32, fwd.output_dtypes)
        y = fwd(x)
        dx = tf.gradients(ys=[y], xs=[x], grad_ys=[5.0])
        self.assertEqual([18.0, 60.0], self.evaluate([y] + dx))

  @FunctionTestParameters
  def testListInput(self, bak_as_function):
    with self.session():
      sig = [tf.TensorSpec((2, 2), tf.float32)] * 2

      def Bak(xs, ys, dys):
        del ys
        w, x = xs
        return [
            tf.matmul(dys[0], tf.transpose(x)) + 100.,
            tf.matmul(tf.transpose(w), dys[0]) + 200.
        ]

      @py_utils.Function(fwd_sig=sig)
      def Fwd(args):
        w, x = args
        return [tf.matmul(w, x)]

      @py_utils.Function(fwd_sig=sig, bak=Bak, bak_as_function=bak_as_function)
      def FwdWithBak(args):
        w, x = args
        return [tf.matmul(w, x)]

      a = np.array([[1.0, 2.0], [0.0, -3.0]], dtype=np.float32)
      b = np.array([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
      xs = [tf.constant(a), tf.constant(b)]
      for fwd in [Fwd, FwdWithBak]:
        self.assertEqual([tf.float32], fwd.output_dtypes)
        ys = fwd(xs)
        self.assertIsInstance(ys, list)
        loss = tf.reduce_sum(tf.square(ys[0]))
        dw, dx, dy = tf.gradients(ys=loss, xs=xs + ys)
        y, dw, dx, dy = self.evaluate(ys + [dw, dx, dy])
        self.assertAllEqual(y, a.dot(b))
        self.assertAllEqual(dy, 2 * y)
        self.assertAllEqual(dw, (2 * y).dot(b.T) +
                            (100 if fwd is FwdWithBak else 0))
        self.assertAllEqual(dx,
                            a.T.dot(2 * y) + (200 if fwd is FwdWithBak else 0))

  @FunctionTestParameters
  def testNestedMapInput(self, bak_as_function):
    with self.session():
      spec = tf.TensorSpec((2, 2), tf.float32)
      sig = py_utils.NestedMap(w=spec, x=spec)

      def Bak(xs, ys, dys):
        del ys
        return py_utils.NestedMap(
            w=tf.matmul(dys.y, tf.transpose(xs.x)) + 100.,
            x=tf.matmul(tf.transpose(xs.w), dys.y) + 200.)

      @py_utils.Function(fwd_sig=sig)
      def Fwd(xs):
        return py_utils.NestedMap(y=tf.matmul(xs.w, xs.x))

      @py_utils.Function(fwd_sig=sig, bak=Bak, bak_as_function=bak_as_function)
      def FwdWithBak(xs):
        return py_utils.NestedMap(y=tf.matmul(xs.w, xs.x))

      a = np.array([[1.0, 2.0], [0.0, -3.0]], dtype=np.float32)
      b = np.array([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
      xs = py_utils.NestedMap(w=tf.constant(a), x=tf.constant(b))
      for fwd in [Fwd, FwdWithBak]:
        self.assertEqual(py_utils.NestedMap(y=tf.float32), fwd.output_dtypes)
        ys = fwd(xs)
        loss = tf.reduce_sum(tf.square(ys.y))
        dw, dx, dy = tf.gradients(xs=xs.Flatten() + ys.Flatten(), ys=loss)
        y, dw, dx, dy = self.evaluate([ys.y, dw, dx, dy])
        self.assertAllEqual(y, a.dot(b))
        self.assertAllEqual(dy, 2 * y)
        self.assertAllEqual(dw, (2 * y).dot(b.T) +
                            (100 if fwd is FwdWithBak else 0))
        self.assertAllEqual(dx,
                            a.T.dot(2 * y) + (200 if fwd is FwdWithBak else 0))

  @FunctionTestParameters
  def testImplicitInput(self, bak_as_function):
    with self.session() as sess:
      w = tf.placeholder(tf.float32)
      sig = py_utils.NestedMap(x=tf.TensorSpec((2, 2), tf.float32))

      def Bak(xs, ys, dys):
        del ys
        dw = tf.matmul(dys.y, tf.transpose(xs.x)) + 100.
        ret = py_utils.NestedMap(x=tf.matmul(tf.transpose(w), dys.y) + 200.)
        if bak_as_function:
          assert py_utils.GetExtraArgs()
        return ret, dw

      @py_utils.Function(fwd_sig=sig)
      def Fwd(xs):
        ret = py_utils.NestedMap(y=tf.matmul(w, xs.x))
        assert py_utils.GetExtraArgs()
        return ret

      @py_utils.Function(fwd_sig=sig, bak=Bak, bak_as_function=bak_as_function)
      def FwdWithBak(xs):
        ret = py_utils.NestedMap(y=tf.matmul(w, xs.x))
        assert py_utils.GetExtraArgs()
        return ret

      a = np.array([[1.0, 2.0], [0.0, -3.0]], dtype=np.float32)
      b = np.array([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
      xs = py_utils.NestedMap(x=tf.constant(b, dtype=tf.float32))
      for fwd in [Fwd, FwdWithBak]:
        self.assertEqual([w], fwd.captured_inputs)
        self.assertEqual(py_utils.NestedMap(y=tf.float32), fwd.output_dtypes)
        ys = fwd(xs)
        loss = tf.reduce_sum(tf.square(ys.y))
        dw, dx, dy = tf.gradients(xs=[w] + xs.Flatten() + ys.Flatten(), ys=loss)
        y, dw, dx, dy = sess.run([ys.y, dw, dx, dy], feed_dict={w: a})
        self.assertAllEqual(y, a.dot(b))
        self.assertAllEqual(dy, 2 * y)
        self.assertAllEqual(dw, (2 * y).dot(b.T) +
                            (100 if fwd is FwdWithBak else 0))
        self.assertAllEqual(dx,
                            a.T.dot(2 * y) + (200 if fwd is FwdWithBak else 0))

  @FunctionTestParameters
  def testPreserveStaticShape(self, bak_as_function):
    with self.session():

      def Bak(x, y, dy):
        del x, y
        return dy

      def Fwd(x):
        shape = py_utils.GetShape(x)
        if isinstance(shape, tf.Tensor):
          return tf.ones_like(x)
        else:
          for dim in shape:
            if isinstance(dim, tf.Tensor):
              return tf.ones_like(x) + 1
          return tf.zeros_like(x)

      a = np.array([[1.0, 2.0], [0.0, -3.0]], dtype=np.float32)
      x = tf.constant(a)
      sig = tf.TensorSpec((2, 2), tf.float32)
      for fwd in [
          py_utils.Function(fwd_sig=sig)(fwd=Fwd),
          py_utils.Function(
              fwd_sig=sig, bak=Bak, bak_as_function=bak_as_function)(fwd=Fwd)
      ]:
        y = self.evaluate(fwd(x))
        self.assertAllEqual(y, np.zeros_like(a))

  def testStatefulOps(self):

    @py_utils.Function()
    def Stateless():
      return tf.constant(1.0)

    @py_utils.Function()
    def Stateful():
      return tf.random.uniform([1])

    @py_utils.Function()
    def StatelessCall():
      return Stateless()

    @py_utils.Function()
    def StatefulCall():
      return Stateful()

    self.assertEmpty(Stateless.stateful_ops)
    self.assertEqual(['RandomUniform'], [op[1] for op in Stateful.stateful_ops])
    self.assertEmpty(StatelessCall.stateful_ops)
    self.assertLen(StatefulCall.stateful_ops, 1)

  def testFuncType(self):
    defun_type = type(tf.Defun()(lambda: 1))
    function_type = type(tf.function(lambda: 1).get_concrete_function())

    @py_utils.Function(fwd_sig=tf.TensorSpec(None, tf.float32))
    def Fwd(xs):
      return xs * 2

    self.assertIsInstance(
        Fwd.func, function_type if py_utils._UseTfFunction() else defun_type)

  def testEmptyInputWithGlobalStepContext(self):
    """Test that global step is pass as input iff it's in the signature."""
    # Unset the global step tensor so it's not in the signature.
    with py_utils.GlobalStepContext(None), self.session():

      @py_utils.Function()
      def Fwd():
        return tf.constant(1.0)

      self.assertEqual(tf.float32, Fwd.output_dtypes)
      # Set the global step tensor when calling the function.
      with py_utils.GlobalStepContext(tf.constant(1, dtype=tf.int32)):
        ys = Fwd()
        self.assertEqual(1.0, self.evaluate(ys))

  def testNonemptyInputWithGlobalStepContext(self):
    """Test that global step is pass as input iff it's in the signature."""
    # Unset the global step tensor so it's not in the signature.
    with py_utils.GlobalStepContext(None), self.session():
      sig = tf.TensorSpec(None, tf.float32)

      def Bak(x, y, dy):
        del y
        return 4 * x * dy

      @py_utils.Function(fwd_sig=sig)
      def Fwd(x):
        return x * x * 2

      @py_utils.Function(fwd_sig=sig, bak=Bak)
      def FwdWithBak(x):
        return x * x * 2

      x = tf.constant(3.0)
      for fwd in [Fwd, FwdWithBak]:
        self.assertEqual(tf.float32, fwd.output_dtypes)
        # Set the global step tensor when calling the function.
        with py_utils.GlobalStepContext(tf.constant(1, dtype=tf.int32)):
          y = fwd(x)
          dx = tf.gradients(ys=[y], xs=[x], grad_ys=[5.0])
          self.assertEqual([18.0, 60.0], self.evaluate([y] + dx))


class IfTest(test_utils.TestCase, parameterized.TestCase):

  def testNestedMapInput(self):
    with self.session():

      def ThenBody(nmap):
        nmap.value -= 1.
        return nmap

      def ElseBody(nmap):
        nmap.value += 1.
        return nmap

      inputs = py_utils.NestedMap(value=tf.constant(0.))
      true_out = py_utils.If(True, inputs, ThenBody, ElseBody)
      false_out = py_utils.If(False, inputs, ThenBody, ElseBody)

      true_out = self.evaluate(true_out)
      false_out = self.evaluate(false_out)

    self.assertEqual(-1., true_out.value)
    self.assertEqual(1., false_out.value)

  def testScalarInput(self):
    with self.session():

      def ThenBody(value):
        return value - 1.

      def ElseBody(value):
        return value + 1.

      inputs = tf.constant(0.)
      true_out = py_utils.If(True, inputs, ThenBody, ElseBody)
      false_out = py_utils.If(False, inputs, ThenBody, ElseBody)

      true_out = self.evaluate(true_out)
      false_out = self.evaluate(false_out)

    self.assertEqual(-1., true_out)
    self.assertEqual(1., false_out)

  def testListInput(self):
    with self.session():

      def ThenBody(values):
        return values[0] - 1., values[1] + 1.

      def ElseBody(values):
        return values[0] + 1., values[1] - 1.

      inputs = [tf.constant(0.), tf.constant(0.)]
      true_out = py_utils.If(True, inputs, ThenBody, ElseBody)
      false_out = py_utils.If(False, inputs, ThenBody, ElseBody)

      true_out = self.evaluate(true_out)
      false_out = self.evaluate(false_out)

    self.assertEqual((-1., 1.), true_out)
    self.assertEqual((1., -1.), false_out)

  def testEmptyInput(self):
    with self.session():

      def ThenBody():
        return tf.constant(1)

      def ElseBody():
        return tf.constant(0)

      true_out = py_utils.If(True, None, ThenBody, ElseBody)
      false_out = py_utils.If(False, None, ThenBody, ElseBody)

      true_out = self.evaluate(true_out)
      false_out = self.evaluate(false_out)

    self.assertEqual(1, true_out)
    self.assertEqual(0, false_out)

  def testCapturedInput(self):
    with self.session():
      a = tf.constant(0)
      b = tf.constant(1)

      def ThenBody():
        return a + b

      def ElseBody():
        return a - b

      true_out = py_utils.If(True, None, ThenBody, ElseBody)
      false_out = py_utils.If(False, None, ThenBody, ElseBody)

      true_out = self.evaluate(true_out)
      false_out = self.evaluate(false_out)

    self.assertEqual(1, true_out)
    self.assertEqual(-1, false_out)

  def testCapturedInputsMismatch(self):
    with self.session():
      a = tf.constant(0)
      b = tf.constant(1)
      c = tf.constant(2)

      def OneCapture():
        return a

      def TwoCapture():
        return a - b

      def TwoCaptureReverse():
        return b - a

      def TwoCapture2():
        return a + c

      with self.assertRaises(ValueError):
        py_utils.If(True, None, OneCapture, TwoCapture)

      with self.assertRaises(ValueError):
        py_utils.If(True, None, TwoCapture, OneCapture)

      with self.assertRaises(ValueError):
        py_utils.If(True, None, TwoCapture, TwoCapture2)

      with self.assertRaises(ValueError):
        py_utils.If(True, None, TwoCapture, TwoCaptureReverse)


class ForLoopTest(test_utils.TestCase):

  def testSimple(self):
    with self.session():

      # Basel problem. \sum_1 1/i^2 = pi ^ 2 / 6. A slow convergent series.
      def Body(i, state):
        state.value = state.value + 1.0 / tf.square(tf.cast(i, tf.float32))
        return state

      state = py_utils.NestedMap(value=tf.constant(0.))
      state = py_utils.ForLoop(Body, 1, 10000, 1, state)

      value = self.evaluate(state.value)

    self.assertAllClose(np.pi * np.pi / 6, value, rtol=1e-3)


class TopKTest(test_utils.TestCase):

  def test_top_2(self):
    with self.session():
      x_in = tf.random.normal([4, 5, 6, 8])
      top2_value_a, top2_index_a = py_utils.TopK(x_in, 2)
      top2_value_b, top2_index_b = tf.math.top_k(x_in, 2)
      v1, v2 = self.evaluate([top2_value_a, top2_value_b])
      v3, v4 = self.evaluate([top2_index_a, top2_index_b])
      self.assertAllEqual(v1, v2)
      self.assertAllEqual(v3, v4)

  def test_top_1(self):
    with self.session():
      x_in = tf.random.normal([4, 5, 6, 8])
      top1_value_a, top1_index_a = py_utils.TopK(x_in, 1)
      top1_value_b, top1_index_b = tf.math.top_k(x_in, 1)
      v1, v2 = self.evaluate([top1_value_a, top1_value_b])
      v3, v4 = self.evaluate([top1_index_a, top1_index_b])
      self.assertAllEqual(v1, v2)
      self.assertAllEqual(v3, v4)


class TpuSummaryTensorsTest(test_utils.TestCase):

  def testTpuSummaryTensors(self):
    with self.session():
      with tf.name_scope('fprop'):
        with tf.name_scope('tower_0_0'):
          with tf.name_scope('fprop'):
            with tf.name_scope('my_model'):
              with tf.name_scope('layer_001'):
                with tf.name_scope('fprop'):
                  x = tf.constant(0., name='inputs')
                  py_utils.AddTpuSummaryTensor('mean_x', tf.reduce_mean(x))
              with tf.name_scope('layer_002'):
                with tf.name_scope('fprop'):
                  x = tf.identity(x, name='inputs')
                  py_utils.AddTpuSummaryTensor('mean_x', tf.reduce_mean(x))
      tpu_summary_tensors = py_utils.GetTpuSummaryTensors()
      actual_value = self.evaluate([tpu_summary_tensors])
      expected_value = [{
          'mean_x/fprop/tower_0_0/fprop/my_model/layer_001/fprop': (0., 1.),
          'mean_x/fprop/tower_0_0/fprop/my_model/layer_002/fprop': (0., 1.),
      }]
      self.assertAllEqual(expected_value, actual_value)


class HasShapeTest(test_utils.TestCase):

  def testFullyDynamicShapesMatchesOk(self):
    x_pl = tf.placeholder(tf.float32)
    y_pl = tf.placeholder(tf.float32)
    x = py_utils.HasShape(x_pl, py_utils.GetShape(y_pl))
    with self.session() as sess:
      sess.run(
          x,
          feed_dict={
              x_pl: np.random.rand(1, 2, 3),
              y_pl: np.random.rand(1, 2, 3),
          })

  def testFullyDynamicShapesMismatchRaisesError(self):
    x_pl = tf.placeholder(tf.float32)
    y_pl = tf.placeholder(tf.float32)
    x = py_utils.HasShape(x_pl, py_utils.GetShape(y_pl))
    with self.session() as sess:
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  r'.*mismatch shape:.*'):
        sess.run(
            x,
            feed_dict={
                x_pl: np.random.rand(1, 2, 3),
                y_pl: np.random.rand(4, 5, 6),
            })

  def testFullyDynamicRankMismatchRaisesError(self):
    x_pl = tf.placeholder(tf.float32)
    y_pl = tf.placeholder(tf.float32)
    x = py_utils.HasShape(x_pl, py_utils.GetShape(y_pl))
    with self.session() as sess:
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  r'.*mismatch shape:.*'):
        sess.run(
            x,
            feed_dict={
                x_pl: np.random.rand(1, 2),
                y_pl: np.random.rand(4, 5, 6),
            })

  def testFullyConstantShapesMatchesOk(self):
    x_pl = tf.placeholder(tf.float32)
    x = py_utils.HasShape(x_pl, tf.constant([1, 2, -1]))
    with self.session() as sess:
      sess.run(
          x, feed_dict={
              x_pl: np.random.rand(1, 2, 3),
          })

  def testFullyConstantShapesMismatchRaisesError(self):
    x_pl = tf.placeholder(tf.float32)
    x = py_utils.HasShape(x_pl, tf.constant([1, 2, -1]))
    with self.session() as sess:
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  r'.*mismatch shape:.*'):
        sess.run(
            x, feed_dict={
                x_pl: np.random.rand(2, 2, 3),
            })

  def testRankMismatchRaisesError(self):
    with self.session():
      with self.assertRaisesRegex(
          ValueError, r'Tensor does not match rank of expected shape.*'):
        self.evaluate(py_utils.HasShape(tf.random.uniform((1, 2, 3)), [1, 2]))

  def testTensorRankLessThanNDimsRaisesError(self):
    with self.session():
      with self.assertRaisesRegex(ValueError,
                                  r'Tensor has fewer dimensions than ndims.*'):
        self.evaluate(
            py_utils.HasShape(
                tf.random.uniform((1, 2, 3)), [1, 2, 3, 4], ndims=4))

  def testExpectedShapeRankLessThanNDimsRaisesError(self):
    with self.session():
      with self.assertRaisesRegex(
          ValueError,
          r'Expected shape must have number of dimensions equal to ndims.*'):
        self.evaluate(
            py_utils.HasShape(
                tf.random.uniform((1, 2, 3, 4)), [1, 2, 3], ndims=4))

  def testTensorStaticShapeMismatchRaisesError(self):
    x_pl = tf.placeholder(tf.float32, (None, 2, 3))
    y_pl = tf.placeholder(tf.float32, (3, 1, None))
    with self.assertRaisesRegex(
        ValueError, r'Tensor does not match expected shape on dimension 1.*'):
      py_utils.HasShape(x_pl, py_utils.GetShape(y_pl))

  def testTensorShapeMatchesOk(self):
    x_pl = tf.placeholder(tf.float32, (None, 2, 3, None))
    y_pl = tf.placeholder(tf.float32, (3, 2, None, None))
    x = py_utils.HasShape(x_pl, py_utils.GetShape(y_pl))
    with self.session() as sess:
      sess.run(
          x,
          feed_dict={
              x_pl: np.random.rand(3, 2, 3, 4),
              y_pl: np.random.rand(3, 2, 3, 4),
          })

  def testTensorShapeMatchesWithMinus1Ok(self):
    x_pl = tf.placeholder(tf.float32, (None, 2, 3, None))
    x = py_utils.HasShape(x_pl, [-1, -1, 3, -1])
    with self.session() as sess:
      sess.run(
          x, feed_dict={
              x_pl: np.random.rand(3, 2, 3, 4),
          })

  def testTensorShapeWithMinus1MismatchRaises(self):
    x_pl = tf.placeholder(tf.float32, (None, 2, 3, None))
    x = py_utils.HasShape(x_pl, [-1, -1, 3, 5])
    with self.session() as sess:
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  r'.*mismatch shape:.*'):
        sess.run(
            x, feed_dict={
                x_pl: np.random.rand(3, 2, 3, 4),
            })

  def testTensorShapeMatchesWithTensorExpectedShape(self):
    x_pl = tf.placeholder(tf.float32, (None, 2, 3, None))
    x = py_utils.HasShape(x_pl, tf.constant([-1, -1, 3, -1]))
    with self.session() as sess:
      sess.run(
          x, feed_dict={
              x_pl: np.random.rand(3, 2, 3, 4),
          })

  def testTensorShapeMismatchWithTensorExpectedShapeRaises(self):
    x_pl = tf.placeholder(tf.float32, (None, 2, 3, None))
    x = py_utils.HasShape(x_pl, [-1, tf.constant(-1), 3, tf.constant(5)])
    with self.session() as sess:
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  r'.*mismatch shape:.*'):
        sess.run(
            x, feed_dict={
                x_pl: np.random.rand(3, 2, 3, 4),
            })

  def testTensorStaticShapeMatchDynamicMismatchRaises(self):
    x_pl = tf.placeholder(tf.float32, (None, None, 2, 3, None))
    y_pl = tf.placeholder(tf.float32, (None, 3, 2, None, None))
    x = py_utils.HasShape(x_pl, py_utils.GetShape(y_pl))
    with self.session() as sess:
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  r'.*mismatch shape:.*'):
        sess.run(
            x,
            feed_dict={
                x_pl: np.random.rand(1, 3, 2, 3, 4),
                y_pl: np.random.rand(2, 3, 2, 5, 4),
            })


class SoftmaxTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Base',), ('ExtraLogit', 0.))
  def testCompute(self, extra_logit=None):
    tf.random.set_seed(123)
    x = np.array([
        [10, 20, 1e10, 1e30],
        [-1e30, -1e10, -20, 10],
        [-20, -10, 1, 2],
    ])
    y = py_utils.Softmax(x, extra_logit=extra_logit)

    if extra_logit is None:
      expected = np.array([
          [0., 0., 0., 1.],
          [0., 0., 0., 1.],
          [0., 0., 0.268940, 0.731055],
      ])

    else:
      expected = np.array([
          [0., 0., 0., 1.],
          [0., 0., 0., 0.999954],
          [0., 0., 0.244727, 0.665238],
      ])
    with self.session():
      self.assertAllClose(expected, self.evaluate(y), atol=1e-5, rtol=1e-5)


class DivideNoNanTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple', 5., 2., 2.5), ('zero', 5., 0., 0.0),
      ('np', np.array([-2., 1., 0.]), np.array([-2., 0., 1.
                                               ]), np.array([1.0, 0., 0.])),
      ('np_bfloat16', np.array([-2., 1., 0.]), np.array(
          [-2., 0., 1.]), np.array([1.0, 0., 0.]), tf.bfloat16))
  def testDivide(self, x, y, expected, dtype=tf.float32):
    with self.session():
      res = py_utils.DivideNoNan(tf.cast(x, dtype), tf.cast(y, dtype))
      orig = tf.math.divide_no_nan(x, y)
      res, orig = self.evaluate([res, orig])
      self.assertAllEqual(res, orig)
      self.assertAllEqual(res, expected)

  def testGradient(self):
    with self.session(use_gpu=False):
      x = tf.get_variable('x', initializer=[-2., 1., 0., 0.])
      y = tf.get_variable('y', initializer=[-2., 0., 1., 0.])
      res = py_utils.DivideNoNan(x, y)
      dys = tf.where(res > 0., tf.ones_like(res), tf.ones_like(res) * -.5)
      self.evaluate(tf.global_variables_initializer())
      dx, dy = self.evaluate(tf.gradients(xs=[x, y], ys=res, grad_ys=dys))
      tf.logging.info('dx=%r, dy=%r', dx, dy)
      self.assertAllClose([-0.5, 0., -0.5, 0.], dx)
      self.assertAllClose([0.5, 0., 0.0, 0.], dy)


if __name__ == '__main__':
  test_utils.main()
