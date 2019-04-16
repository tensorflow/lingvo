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
"""Tests for base_layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import test_utils


class AddingAccumulator(base_layer.Accumulator):

  def DefaultValue(self):
    return tf.convert_to_tensor(0.0)

  def Update(self, new_value):
    self.SetValue(self.GetValue() + new_value)


def EvalAndFlatten(nmap):
  return nmap.Transform(lambda x: x.eval()).FlattenItems()


class TestLayer(base_layer.BaseLayer):

  @base_layer.initializer
  def __init__(self, params):
    super(TestLayer, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      self.CreateVariable(
          'w',
          py_utils.WeightParams(
              shape=[4, 4],
              dtype=p.dtype,
              init=p.params_init,
              collections=[self.__class__.__name__ + '_vars']))
      self.CreateVariable(
          'b',
          py_utils.WeightParams(
              shape=[4],
              dtype=p.dtype,
              init=py_utils.WeightInit.Constant(),
              collections=[
                  self.__class__.__name__ + '_vars',
                  py_utils.SKIP_LP_REGULARIZATION
              ]))


class BaseLayerTest(test_utils.TestCase):

  def testCopyBaseParams(self):
    # CopyBaseParams should only overwrite is_eval/vn setting when target use
    # default is_eval/vn config.
    layer_base_p = base_layer.BaseLayer.Params()
    from_param = layer_base_p.Copy()
    to_param = layer_base_p.Copy()
    from_param.is_eval = False
    from_param.vn.global_vn = True
    from_param.random_seed = 1234
    from_param.skip_lp_regularization = True
    # Target use default, overwrite.
    base_layer.BaseLayer.CopyBaseParams(from_param, to_param)
    self.assertEqual(False, to_param.is_eval)
    self.assertTrue(to_param.vn.global_vn)
    self.assertEqual(1234, to_param.random_seed)
    self.assertTrue(to_param.skip_lp_regularization)
    to_param = layer_base_p.Copy()
    to_param.is_eval = True
    to_param.vn.per_step_vn = True
    to_param.random_seed = 4321
    to_param.skip_lp_regularization = False
    # Target does not use default, should not overwrite.
    base_layer.BaseLayer.CopyBaseParams(from_param, to_param)
    self.assertEqual(True, to_param.is_eval)
    self.assertTrue(to_param.vn.per_step_vn)
    self.assertFalse(to_param.vn.global_vn)
    self.assertEqual(4321, to_param.random_seed)
    self.assertFalse(to_param.skip_lp_regularization)

  def testCreateChildren(self):
    layer_p = base_layer.BaseLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.cls(layer_p)
    layer.CreateChildren('a', [layer_p, [layer_p, layer_p], layer_p])
    self.assertEqual(len(layer.a), 3)
    self.assertEqual(len(layer.a[1]), 2)
    self.assertEqual(len(layer.vars.a), 3)
    self.assertEqual(len(layer.vars.a[1]), 2)
    self.assertEqual(len(layer.theta.a), 3)
    self.assertEqual(len(layer.theta.a[1]), 2)

  def testCreateVariable(self):
    layer_p = TestLayer.Params().Set(name='test')
    layer = layer_p.cls(layer_p)
    self.assertEqual('test/w/var:0', layer.vars.w.name)
    self.assertEqual('test/b/var:0', layer.vars.b.name)
    self.assertNotIn(layer.vars.w,
                     tf.get_collection(py_utils.SKIP_LP_REGULARIZATION))
    # 'b' always skips Lp regularization.
    self.assertIn(layer.vars.b,
                  tf.get_collection(py_utils.SKIP_LP_REGULARIZATION))

  def testCreateVariableSkipLpRegularization(self):
    layer_p = TestLayer.Params().Set(name='test', skip_lp_regularization=True)
    layer = layer_p.cls(layer_p)
    self.assertIn(layer.vars.w,
                  tf.get_collection(py_utils.SKIP_LP_REGULARIZATION))
    self.assertIn(layer.vars.b,
                  tf.get_collection(py_utils.SKIP_LP_REGULARIZATION))

  def testGetDescendant(self):
    q = base_layer.BaseLayer.Params()
    q.name = 'test'
    l = q.cls(q)
    p = base_layer.BaseLayer.Params()
    l.CreateChild('a', p)
    l.CreateChild('b', p)
    l.a.CreateChild('c', p)
    l.a.c.CreateChild('d', p)
    l.b.CreateChild('e', p)
    self.assertEqual(l, l.GetDescendant(''))
    self.assertEqual(l.a, l.GetDescendant('a'))
    self.assertEqual(l.b, l.GetDescendant('b'))
    self.assertEqual(l.a.c, l.GetDescendant('a.c'))
    self.assertEqual(l.a.c.d, l.GetDescendant('a.c.d'))
    self.assertEqual(l.b.e, l.GetDescendant('b.e'))

  def testCreateAccumulator(self):
    layer_p = base_layer.BaseLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.cls(layer_p)
    layer.CreateChild('child', layer_p)

    # First accumulator should succeed.
    acc1 = AddingAccumulator()
    layer.RegisterAccumulator('acc1', acc1)

    # Name of existing child should fail.
    with self.assertRaises(AssertionError):
      layer.RegisterAccumulator('child', AddingAccumulator())

    # Duplicate should fail.
    with self.assertRaises(AssertionError):
      layer.RegisterAccumulator('acc1', AddingAccumulator())

    # Child with the same name should fail.
    with self.assertRaises(AssertionError):
      layer.CreateChild('acc1', layer_p)

    self.assertEqual(acc1, layer.accumulators.acc1)

    # Get of not created accumulator should fail.
    with self.assertRaises(AttributeError):
      layer.accumulators.notexist

  def testGetUpdateAccumulator(self):
    with self.session():
      layer_p = base_layer.BaseLayer.Params()
      layer_p.name = 'test'
      layer = layer_p.cls(layer_p)

      layer.RegisterAccumulator('acc1', AddingAccumulator())

      # Initial value.
      self.assertEqual(0.0, layer.accumulators.acc1.GetValue().eval())

      # Update/merge.
      layer.accumulators.acc1.Update(1.0)
      layer.accumulators.acc1.Update(1.0)
      self.assertEqual(2.0, layer.accumulators.acc1.GetValue().eval())

      # Reset.
      layer.accumulators.Transform(lambda acc: acc.Reset())
      self.assertEqual(0.0, layer.accumulators.acc1.GetValue().eval())

  def testAccumulatorDisableEnable(self):
    with self.session():
      layer_p = base_layer.BaseLayer.Params()
      layer_p.name = 'test'
      layer = layer_p.cls(layer_p)

      layer.RegisterAccumulator('acc1', AddingAccumulator())
      layer.accumulators.acc1.Update(1.0)

      # Disable should force value to 0 and reject updates.
      layer.accumulators.Transform(lambda acc: acc.Disable())
      self.assertEqual(0.0, layer.accumulators.acc1.GetValue().eval())
      layer.accumulators.acc1.Update(3.0)
      self.assertEqual(0.0, layer.accumulators.acc1.GetValue().eval())
      layer.accumulators.Transform(lambda acc: acc.Enable())

      # Should restore.
      self.assertEqual(1.0, layer.accumulators.acc1.GetValue().eval())

  def testGetSetAccumulatorValues(self):
    with self.session():
      layer_p = base_layer.BaseLayer.Params()
      layer_p.name = 'test'
      layer1 = layer_p.cls(layer_p)
      layer1.CreateChild('layer1a', layer_p)
      layer1.CreateChild('layer1b', layer_p)
      layer1.layer1b.CreateChild('layer1b1', layer_p)

      # Create nested accumulators:
      #   acc1: layer1
      #   acc2: layer1.layer1a
      #   acc3: layer1.layer1b.layer1b1
      layer1.RegisterAccumulator('acc1', AddingAccumulator())
      layer1.layer1a.RegisterAccumulator('acc2', AddingAccumulator())
      layer1.layer1b.layer1b1.RegisterAccumulator('acc3', AddingAccumulator())

      # Pack with initial values.
      initial_pack = layer1.GetAccumulatorValues()
      initial_pack_eval = EvalAndFlatten(initial_pack)
      print('Initial values pack =', initial_pack_eval)
      self.assertEqual(initial_pack_eval, [('acc1', 0.0), ('layer1a.acc2', 0.0),
                                           ('layer1b.layer1b1.acc3', 0.0)])

      # Update to a new known state.
      layer1.accumulators.acc1.Update(1.0)
      layer1.layer1a.accumulators.acc2.Update(2.0)
      layer1.layer1b.layer1b1.accumulators.acc3.Update(3.0)
      updated_pack = layer1.GetAccumulatorValues()
      updated_pack_eval = EvalAndFlatten(updated_pack)
      print('Updated values pack =', updated_pack_eval)
      self.assertEqual(updated_pack_eval, [('acc1', 1.0), ('layer1a.acc2', 2.0),
                                           ('layer1b.layer1b1.acc3', 3.0)])

      # Save and reset.
      saved_pack = layer1.GetAccumulatorValues()
      layer1.accumulators.Transform(lambda acc: acc.Reset())
      self.assertEqual(0.0, layer1.accumulators.acc1.GetValue().eval())
      self.assertEqual(0.0, layer1.layer1a.accumulators.acc2.GetValue().eval())
      self.assertEqual(
          0.0,
          layer1.layer1b.layer1b1.accumulators.acc3.GetValue().eval())

      # Set and check.
      layer1.SetAccumulatorValues(saved_pack)
      self.assertEqual(1.0, layer1.accumulators.acc1.GetValue().eval())
      self.assertEqual(2.0, layer1.layer1a.accumulators.acc2.GetValue().eval())
      self.assertEqual(
          3.0,
          layer1.layer1b.layer1b1.accumulators.acc3.GetValue().eval())

  def testAddFunction(self):
    layer_p = base_layer.BaseLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.cls(layer_p)

    layer.AddFunction('test1', lambda: 1)
    with self.assertRaises(AttributeError):
      layer.AddFunction('test1', lambda: 2)
    self.assertEquals(1, layer.fns.test1())

  def testAttributeErrorInPropertyGetter(self):

    class BadLayer(base_layer.BaseLayer):

      @classmethod
      def Params(cls):
        return super(BadLayer, cls).Params()

      @base_layer.initializer
      def __init__(self, params):
        super(BadLayer, self).__init__(params)

      @property
      def bad_property(self):
        raise AttributeError('INTERNAL')

    layer_p = BadLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.cls(layer_p)

    with self.assertRaisesRegexp(AttributeError, 'bad_sub_layer'):
      _ = layer.bad_sub_layer

    with self.assertRaisesRegexp(AttributeError, 'INTERNAL'):
      _ = layer.bad_property


if __name__ == '__main__':
  tf.test.main()
