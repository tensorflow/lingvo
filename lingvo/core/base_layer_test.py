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

from typing import Any
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import test_utils


class AddingAccumulator(base_layer.Accumulator):

  def DefaultValue(self):
    return tf.convert_to_tensor(0.0)

  def Update(self, new_value):
    self.SetValue(self.GetValue() + new_value)


class TestLayer(base_layer.BaseLayer):

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
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


class TestParentLayer(base_layer.BaseLayer):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('child', TestLayer.Params(), 'The child layer params.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('child_0', p.child)
    self.CreateChild('child_1', p.child)


class BaseLayerTest(test_utils.TestCase):

  def _EvalAndFlatten(self, nmap: py_utils.NestedMap) -> list[tuple[str, Any]]:
    return nmap.Transform(self.evaluate).FlattenItems()

  def testCopyBaseParams(self):
    # CopyBaseParams should only overwrite vn setting when target use
    # default vn config.
    layer_base_p = base_layer.BaseLayer.Params()
    from_param = layer_base_p.Copy()
    to_param = layer_base_p.Copy()
    from_param.vn.global_vn = True
    from_param.random_seed = 1234
    from_param.skip_lp_regularization = True
    from_param.fprop_dtype = tf.bfloat16
    # Target use default, overwrite.
    base_layer.BaseLayer.CopyBaseParams(from_param, to_param)
    self.assertTrue(to_param.vn.global_vn)
    self.assertEqual(1234, to_param.random_seed)
    self.assertTrue(to_param.skip_lp_regularization)
    self.assertEqual(tf.bfloat16, to_param.fprop_dtype)
    to_param = layer_base_p.Copy()
    to_param.vn.per_step_vn = True
    to_param.random_seed = 4321
    to_param.skip_lp_regularization = False
    to_param.fprop_dtype = tf.float32
    # Target does not use default, should not overwrite.
    base_layer.BaseLayer.CopyBaseParams(from_param, to_param)
    self.assertTrue(to_param.vn.per_step_vn)
    self.assertFalse(to_param.vn.global_vn)
    self.assertEqual(4321, to_param.random_seed)
    self.assertFalse(to_param.skip_lp_regularization)
    self.assertEqual(tf.float32, to_param.fprop_dtype)

  def testCreateChild(self):
    layer_p = TestParentLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.Instantiate()
    self.assertEqual(
        {
            'child_0': {
                'w': [4, 4],
                'b': [4]
            },
            'child_1': {
                'w': [4, 4],
                'b': [4]
            },
        }, tf.nest.map_structure(lambda v: v.shape.as_list(), layer.vars))

  def testChildVariableScopeOverride(self):

    class TestParentLayerWithScopeOverride(TestParentLayer):

      def _child_variable_scope_override(self):
        return {'child_0': ['new_scope1']}

    # Check default scope in the base case.
    layer_p = TestParentLayer.Params().Set(name='test0')
    layer = layer_p.Instantiate()
    self.assertEqual('test0/child_0/w/var:0', layer.child_0.vars.w.name)
    self.assertEqual('test0/child_1/w/var:0', layer.child_1.vars.w.name)

    # Check updating scope with _child_variable_scope_override().
    layer_p = TestParentLayerWithScopeOverride.Params().Set(name='test1')
    layer = layer_p.Instantiate()
    self.assertEqual('new_scope1/child_0/w/var:0', layer.child_0.vars.w.name)
    self.assertEqual('test1/child_1/w/var:0', layer.child_1.vars.w.name)

    # Check updating scope with p.child_variable_scope_override.
    layer_p = TestParentLayer.Params().Set(
        name='test2', child_variable_scope_override={'child_1': ['new_scope2']})
    layer = layer_p.Instantiate()
    self.assertEqual('test2/child_0/w/var:0', layer.child_0.vars.w.name)
    self.assertEqual('new_scope2/child_1/w/var:0', layer.child_1.vars.w.name)

  def testCreateChildren(self):
    layer_p = base_layer.BaseLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.Instantiate()
    layer._disable_create_child = False  # pylint: disable=protected-access
    layer.CreateChildren('a', [
        layer_p,
        [layer_p, layer_p],
        {
            'b': layer_p,
            'c': {
                'd': [layer_p, layer_p]
            }
        },
    ])
    self.assertEqual(len(layer.a), 3)
    self.assertEqual(len(layer.a[1]), 2)
    self.assertEqual(len(layer.a[2]), 2)
    self.assertEqual(len(layer.a[2]['c']['d']), 2)
    self.assertEqual(len(layer.vars.a), 3)
    self.assertEqual(len(layer.vars.a[1]), 2)
    self.assertEqual(len(layer.vars.a[2]), 2)
    self.assertEqual(len(layer.vars.a[2]['c']['d']), 2)
    self.assertEqual(len(layer.theta.a), 3)
    self.assertEqual(len(layer.theta.a[1]), 2)
    self.assertEqual(len(layer.theta.a[2]), 2)
    self.assertEqual(len(layer.theta.a[2]['c']['d']), 2)

  def testCreateVariable(self):
    layer_p = TestLayer.Params().Set(name='test')
    layer = layer_p.Instantiate()
    self.assertEqual('test/w/var:0', layer.vars.w.name)
    self.assertEqual('test/b/var:0', layer.vars.b.name)
    self.assertFalse(
        py_utils._VarInCollection(
            layer.vars.w, tf.get_collection(py_utils.SKIP_LP_REGULARIZATION)))
    # 'b' always skips Lp regularization.
    self.assertTrue(
        py_utils._VarInCollection(
            layer.vars.b, tf.get_collection(py_utils.SKIP_LP_REGULARIZATION)))

  def testVariableThetaValue(self):
    with self.session():
      layer_p = TestLayer.Params().Set(name='test')
      layer = layer_p.Instantiate()
      self.evaluate(tf.global_variables_initializer())
      self.assertAllClose(
          self.evaluate(layer.vars.w), self.evaluate(layer.theta.w))
      b_eval = self.evaluate(layer.vars.b)
      self.assertAllClose(b_eval, self.evaluate(layer.theta.b))
      self.assertAllClose(b_eval, self.evaluate(layer._private_theta['b']))

      # theta reflects the vars change.
      new_b = layer.vars.b.assign(tf.ones_like(layer.vars.b) * 3.)
      with tf.control_dependencies([new_b]):
        self.assertAllClose(b_eval * 3., self.evaluate(new_b))
        self.assertAllClose(self.evaluate(layer.vars.b), self.evaluate(new_b))
        self.assertAllClose(
            self.evaluate(layer.vars.b), self.evaluate(layer.theta.b))

  def testCreateVariableSkipLpRegularization(self):
    layer_p = TestLayer.Params().Set(name='test', skip_lp_regularization=True)
    layer = layer_p.Instantiate()
    self.assertTrue(
        py_utils._VarInCollection(
            layer.vars.w, tf.get_collection(py_utils.SKIP_LP_REGULARIZATION)))
    self.assertTrue(
        py_utils._VarInCollection(
            layer.vars.b, tf.get_collection(py_utils.SKIP_LP_REGULARIZATION)))

  def testGetDescendant(self):
    q = base_layer.BaseLayer.Params()
    q.name = 'test'
    # pylint: disable=protected-access
    l = q.Instantiate()
    p = base_layer.BaseLayer.Params()
    l._disable_create_child = False
    l.CreateChild('a', p)
    l.CreateChild('b', p)
    l.a._disable_create_child = False
    l.a.CreateChild('c', p)
    l.a.c._disable_create_child = False
    l.a.c.CreateChild('d', p)
    l.b._disable_create_child = False
    l.b.CreateChild('e', p)
    l.b.e._disable_create_child = False
    l.b.e.CreateChildren('list', [p, p])
    l.b.e.list[1]._disable_create_child = False
    l.b.e.list[1].CreateChild('g', p)
    # pylint: enable=protected-access
    self.assertEqual(l, l.GetDescendant(''))
    self.assertEqual(l.a, l.GetDescendant('a'))
    self.assertEqual(l.b, l.GetDescendant('b'))
    self.assertEqual(l.a.c, l.GetDescendant('a.c'))
    self.assertEqual(l.a.c.d, l.GetDescendant('a.c.d'))
    self.assertEqual(l.b.e, l.GetDescendant('b.e'))
    self.assertEqual(l.b.e.list, l.GetDescendant('b.e.list'))
    self.assertEqual(l.b.e.list[0], l.GetDescendant('b.e.list[0]'))
    self.assertEqual(l.b.e.list[1], l.GetDescendant('b.e.list[-1]'))
    self.assertEqual(l.b.e.list[1].g, l.GetDescendant('b.e.list[1].g'))

    with self.assertRaisesRegex(KeyError, "'l' not found in sub-layer 'b.e'"):
      l.GetDescendant('b.e.l[0]')

    with self.assertRaisesRegex(TypeError, "Attempted to index 'b.e' of type"):
      l.GetDescendant('b.e[0]')

  def testCreateAccumulator(self):
    layer_p = base_layer.BaseLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.Instantiate()
    layer._disable_create_child = False  # pylint: disable=protected-access
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
      layer = layer_p.Instantiate()

      layer.RegisterAccumulator('acc1', AddingAccumulator())

      # Initial value.
      self.assertEqual(0.0, self.evaluate(layer.accumulators.acc1.GetValue()))

      # Update/merge.
      layer.accumulators.acc1.Update(1.0)
      layer.accumulators.acc1.Update(1.0)
      self.assertEqual(2.0, self.evaluate(layer.accumulators.acc1.GetValue()))

      # Reset.
      layer.accumulators.Transform(lambda acc: acc.Reset())
      self.assertEqual(0.0, self.evaluate(layer.accumulators.acc1.GetValue()))

  def testAccumulatorDisableEnable(self):
    with self.session():
      layer_p = base_layer.BaseLayer.Params()
      layer_p.name = 'test'
      layer = layer_p.Instantiate()

      layer.RegisterAccumulator('acc1', AddingAccumulator())
      layer.accumulators.acc1.Update(1.0)

      # Disable should force value to 0 and reject updates.
      layer.accumulators.Transform(lambda acc: acc.Disable())
      self.assertEqual(0.0, self.evaluate(layer.accumulators.acc1.GetValue()))
      layer.accumulators.acc1.Update(3.0)
      self.assertEqual(0.0, self.evaluate(layer.accumulators.acc1.GetValue()))
      layer.accumulators.Transform(lambda acc: acc.Enable())

      # Should restore.
      self.assertEqual(1.0, self.evaluate(layer.accumulators.acc1.GetValue()))

  def testGetSetAccumulatorValues(self):
    with self.session():
      layer_p = base_layer.BaseLayer.Params()
      layer_p.name = 'test'
      layer1 = layer_p.Instantiate()
      layer1._disable_create_child = False  # pylint: disable=protected-access
      layer1.CreateChild('layer1a', layer_p)
      layer1.CreateChild('layer1b', layer_p)
      layer1.layer1b._disable_create_child = False  # pylint: disable=protected-access
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
      initial_pack_eval = self._EvalAndFlatten(initial_pack)
      print('Initial values pack =', initial_pack_eval)
      self.assertEqual(initial_pack_eval, [('acc1', 0.0), ('layer1a.acc2', 0.0),
                                           ('layer1b.layer1b1.acc3', 0.0)])

      # Update to a new known state.
      layer1.accumulators.acc1.Update(1.0)
      layer1.layer1a.accumulators.acc2.Update(2.0)
      layer1.layer1b.layer1b1.accumulators.acc3.Update(3.0)
      updated_pack = layer1.GetAccumulatorValues()
      updated_pack_eval = self._EvalAndFlatten(updated_pack)
      print('Updated values pack =', updated_pack_eval)
      self.assertEqual(updated_pack_eval, [('acc1', 1.0), ('layer1a.acc2', 2.0),
                                           ('layer1b.layer1b1.acc3', 3.0)])

      # Save and reset.
      saved_pack = layer1.GetAccumulatorValues()
      layer1.accumulators.Transform(lambda acc: acc.Reset())
      self.assertEqual(0.0, self.evaluate(layer1.accumulators.acc1.GetValue()))
      self.assertEqual(
          0.0, self.evaluate(layer1.layer1a.accumulators.acc2.GetValue()))
      self.assertEqual(
          0.0,
          self.evaluate(layer1.layer1b.layer1b1.accumulators.acc3.GetValue()))

      # Set and check.
      layer1.SetAccumulatorValues(saved_pack)
      self.assertEqual(1.0, self.evaluate(layer1.accumulators.acc1.GetValue()))
      self.assertEqual(
          2.0, self.evaluate(layer1.layer1a.accumulators.acc2.GetValue()))
      self.assertEqual(
          3.0,
          self.evaluate(layer1.layer1b.layer1b1.accumulators.acc3.GetValue()))

  def testAddFunction(self):
    layer_p = base_layer.BaseLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.Instantiate()

    layer.AddFunction('test1', lambda: 1)
    with self.assertRaises(AttributeError):
      layer.AddFunction('test1', lambda: 2)
    self.assertEqual(1, layer.fns.test1())

  def testAttributeErrorInPropertyGetter(self):

    class BadLayer(base_layer.BaseLayer):

      @classmethod
      def Params(cls):
        return super().Params()

      def __init__(self, params):
        super().__init__(params)

      @property
      def bad_property(self):
        raise AttributeError('INTERNAL')

    layer_p = BadLayer.Params()
    layer_p.name = 'test'
    layer = layer_p.Instantiate()

    with self.assertRaisesRegex(AttributeError, 'bad_sub_layer'):
      _ = layer.bad_sub_layer

    with self.assertRaisesRegex(AttributeError, 'INTERNAL'):
      _ = layer.bad_property

  def testIsLayerParams(self):
    self.assertTrue(base_layer.IsLayerParams(base_layer.BaseLayer.Params()))
    self.assertTrue(base_layer.IsLayerParams(TestLayer.Params()))
    self.assertFalse(base_layer.IsLayerParams(None))
    self.assertFalse(base_layer.IsLayerParams(hyperparams.Params()))
    self.assertFalse(
        base_layer.IsLayerParams(
            hyperparams.InstantiableParams(base_layer.Accumulator)))


if __name__ == '__main__':
  test_utils.main()
