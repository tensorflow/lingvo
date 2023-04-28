# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Test suite for nested_map functionality."""

import collections
import copy

from lingvo.core import nested_map
from lingvo.core import py_utils
from lingvo.core import test_utils


def _AddOne(x):
  return None if x is None else x + type(x)(1)


class NestedMapTest(test_utils.TestCase):
  _TUPLE = collections.namedtuple('Tuple', ['x', 'y'])

  def testNestedMapFromNestedDict(self):
    a = {'a1': 1, 'a2': 2}
    b = {'b1': 1, 'b2': 2}
    c = {'a': a, 'b': b, 'ab': [a, b]}
    d = nested_map.NestedMap(c)
    e = nested_map.NestedMap.FromNestedDict(c)
    self.assertIsInstance(d, nested_map.NestedMap)
    self.assertNotIsInstance(d.a, nested_map.NestedMap)
    self.assertIsInstance(e.a, nested_map.NestedMap)
    self.assertIsInstance(e.ab[0], nested_map.NestedMap)
    self.assertEqual(e.ab[0], e.a)
    self.assertEqual(e.ab[1], e.b)

  def testNestedMapToNestedDict(self):
    a = nested_map.NestedMap(a1=1, a2=2)
    b = {'b1': 1, 'b2': 2}
    c = nested_map.NestedMap(a=a, b=b, ab=[a, b])
    d = c.ToNestedDict()
    self.assertIsInstance(c, nested_map.NestedMap)
    self.assertNotIsInstance(d, nested_map.NestedMap)
    self.assertIsInstance(c.a, nested_map.NestedMap)
    self.assertNotIsInstance(d['a'], nested_map.NestedMap)
    self.assertNotIsInstance(d['ab'][0], nested_map.NestedMap)

  def testNestedMapGetItem(self):
    nm = nested_map.NestedMap()
    nm['a'] = nested_map.NestedMap({'x': 3})
    nm['b'] = nested_map.NestedMap({'y_0': 4, 'y': [3]})
    self.assertEqual(nm.GetItem('a.x'), nm.a.x)
    self.assertEqual(nm.GetItem('b.y_0'), 4)
    self.assertEqual(nm.GetItem('b.y[0]'), 3)
    with self.assertRaises(KeyError):
      nm.GetItem('o')
    with self.assertRaises(IndexError):
      nm.GetItem('b.y[1]')
    with self.assertRaises(TypeError):
      nm.GetItem('b.y.c')

  def testNestedMapGet(self):
    nm = nested_map.NestedMap({'a': {'b': 0}})
    self.assertEqual(nm.Get('a.b'), 0)
    self.assertIsNone(nm.Get('a.b.c'))
    self.assertIsNone(nm.Get('x'))
    self.assertEqual(nm.Get('x', 0), 0)

  def testNestedMapGetFromNestedList(self):
    nm = nested_map.NestedMap({'a': {'b': [0, 1, {'c': 2}]}})
    self.assertEqual(nm.Get('a.b'), [0, 1, {'c': 2}])
    self.assertEqual(nm.Get('a.b[1]'), 1)
    self.assertEqual(nm.Get('a.b[2].c'), 2)
    self.assertIsNone(nm.Get('a.b[3]'))
    self.assertIsNone(nm.Get('a.b.c'))

  def testNestedMapDir(self):
    nm = nested_map.NestedMap({'a': {'b': 0}})
    nm.c = '1'
    nested_map_dir = dir(nm)
    # Flattened keys are in the dir
    self.assertIn('a.b', nested_map_dir)
    self.assertIn('c', nested_map_dir)
    # So are method APIs.
    self.assertIn('Get', nested_map_dir)

  def testNestedMapSet(self):
    nm = nested_map.NestedMap.FromNestedDict({'a': {'b': 0}})
    self.assertEqual(nm.a.b, 0)
    # Test if overriding an existing value works.
    nm.Set('a.b', 1)
    self.assertEqual(nm.a.b, 1)
    # Test if ValueError is raised if an existing intermediate value is not a
    # NestedMap.
    with self.assertRaises(ValueError):
      nm.Set('a.b.c', 2)
    nm.Set('a.b', nested_map.NestedMap())
    # Verify that non-existing intermediate keys are set to NestedMap.
    nm.Set('a.b.x.y', 2)
    self.assertIsInstance(nm.a.b.x, nested_map.NestedMap)
    self.assertEqual(nm.a.b.x.y, 2)
    # Verify that non-existing intermediate list keys are set to list.
    nm.Set('a.b.z[0].y', 3)
    nm.Set('a.b.z[1].y', 4)
    self.assertIsInstance(nm.a.b.z, list)
    self.assertEqual(nm.a.b.z[0].y, 3)
    self.assertEqual(nm.a.b.z[1].y, 4)
    # Verify that using a index > len(list) leads to ValueError.
    with self.assertRaises(ValueError):
      nm.Set('a.b.z[3]', 5)

  def testNestedMapSetIndexed(self):
    nm = nested_map.NestedMap()
    nm.x = [43, 44]
    nm.y = 14
    nm.z = [nested_map.NestedMap(b=13, c=[18, 8]), 4]
    another_map = nested_map.NestedMap()
    for k, v in nm.FlattenItems():
      another_map.Set(k, v)
    self.assertDictEqual(another_map, nm)

  def testGetSlice(self):
    m = nested_map.NestedMap()
    m.foo = nested_map.NestedMap(a=123, b=345, c=nested_map.NestedMap(d=654))
    self.assertEqual(
        m.GetSlice({'foo.a', 'foo.c.d'}),
        nested_map.NestedMap(
            foo=nested_map.NestedMap(a=123, c=nested_map.NestedMap(d=654))
        ),
    )

  def testGetSliceArray(self):
    """Tests that slicing out array values works.

    Note: Get itself needs to be fixed so that setting array values beyond the
    last element + 1 can be set.
    """
    m = nested_map.NestedMap()
    m.foo = [
        nested_map.NestedMap(a=123, b=345, c=nested_map.NestedMap(d=654)),
        nested_map.NestedMap(a=987, b=654, c=nested_map.NestedMap(d=354)),
    ]
    self.assertEqual(
        m.GetSlice(['foo[0].a', 'foo[1].c.d']),
        nested_map.NestedMap(
            foo=[
                nested_map.NestedMap(a=123),
                nested_map.NestedMap(c=nested_map.NestedMap(d=354)),
            ]
        ),
    )

  def testHas(self):
    nm = nested_map.NestedMap.FromNestedDict(
        {'a': {'b': 0, 'c': [{'d': 0}, {'e': 1}]}}
    )
    self.assertTrue(nm.Has('a.b'))
    self.assertTrue(nm.Has('a.c[1].e'))
    self.assertFalse(nm.Has('a.f'))

  def testUnion(self):
    x = nested_map.NestedMap(a=123, b=345, c=nested_map.NestedMap(d=654))
    y = nested_map.NestedMap(aa=987, bb=654, c=nested_map.NestedMap(dd=354))
    self.assertEqual(
        x.Union(y),
        nested_map.NestedMap(
            a=123,
            b=345,
            c=nested_map.NestedMap(d=654, dd=354),
            aa=987,
            bb=654,
        ),
    )

  def testUpdate(self):
    x = nested_map.NestedMap(a=123, b=345, c=nested_map.NestedMap(d=654))
    y = nested_map.NestedMap(aa=987, bb=654, c=nested_map.NestedMap(dd=354))
    x.Update(y)
    self.assertEqual(
        x,
        nested_map.NestedMap(
            a=123,
            b=345,
            c=nested_map.NestedMap(d=654, dd=354),
            aa=987,
            bb=654,
        ),
    )

  def _get_basic_test_inputs(self):
    m = nested_map.NestedMap()
    m.foo = [1, 20, [32]]
    m.bar = nested_map.NestedMap()
    m.bar.x = 100
    m.bar.y = [200, nested_map.NestedMap(z='abc')]
    return m

  def _get_advanced_test_inputs(self):
    m = nested_map.NestedMap()
    m.w = None
    m.y = (200, nested_map.NestedMap(z='abc'))
    m.x = {'foo': 1, 'bar': 'def'}
    m.z = self._TUPLE(5, 'xyz')
    m.zz = []
    return m

  def testBasic(self):
    x = nested_map.NestedMap()
    self.assertLen(x, 0)
    x['foo'] = 100
    self.assertEqual(100, x.foo)
    self.assertEqual(100, x['foo'])
    x.bar = nested_map.NestedMap({'baz': 200})
    self.assertEqual(200, x.bar.baz)
    self.assertNotIn('flatten', x)

  def testPrint(self):
    self.assertEqual(nested_map.NestedMap().DebugString(), '')

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
    n = py_utils.Transform(_AddOne, nested_map.NestedMap())
    self.assertEqual(n.DebugString(), '')
    n = nested_map.NestedMap().Transform(_AddOne)
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
        ('y', (200, {'z': 'abc'})),
        ('z', self._TUPLE(x=5, y='xyz')),
    ]
    self.assertEqual(m.FlattenItems(), original)

    expected = [
        ('w', None),
        ('x.bar', 'def1'),
        ('x.foo', 2),
        ('y', (201, {'z': 'abc1'})),
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
        ('y', (200, {'z': 'abc'})),
        ('z', self._TUPLE(x=5, y='xyz')),
    ]
    n = m.Transform(_AddOneIgnoreError)
    self.assertEqual(n.zz, [])
    self.assertNotEqual(expected, original)
    self.assertEqual(n.FlattenItems(), expected)

    # Original has not been modified.
    self.assertEqual(m.FlattenItems(), original)

  def testFlattenBasic(self):
    self.assertEqual(py_utils.Flatten(nested_map.NestedMap()), [])
    self.assertEqual(nested_map.NestedMap().Flatten(), [])
    self.assertEqual(nested_map.NestedMap().FlattenItems(), [])

    expected = [100, 200, 'abc', 1, 20, 32]
    m = self._get_basic_test_inputs()
    self.assertEqual(py_utils.Flatten(m), expected)
    self.assertEqual(m.Flatten(), expected)

    expected_keys = [
        'bar.x',
        'bar.y[0]',
        'bar.y[1].z',
        'foo[0]',
        'foo[1]',
        'foo[2][0]',
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
        (200, {'z': 'abc'}),
        self._TUPLE(x=5, y='xyz'),
    ]
    self.assertEqual(m.Flatten(), expected)

    expected = [
        ('w', None),
        ('x.bar', 'def'),
        ('x.foo', 1),
        ('y', (200, {'z': 'abc'})),
        ('z', self._TUPLE(x=5, y='xyz')),
    ]
    self.assertEqual(m.FlattenItems(), expected)

  def testPackBasic(self):
    n = py_utils.Pack(nested_map.NestedMap(), [])
    self.assertEqual(n.DebugString(), '')
    n = nested_map.NestedMap().Pack([])
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
        ('y', (3, {'z': 4})),
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
        ('y', (200, {'z': 'abc'})),
        ('z', self._TUPLE(x=5, y='xyz')),
    ]
    self.assertEqual(m.FlattenItems(), expected)

  def testIsCompatible(self):
    empty = nested_map.NestedMap()
    self.assertTrue(empty.IsCompatible(empty))
    self.assertTrue(py_utils.IsCompatible(empty, empty))
    self.assertTrue(empty.IsCompatible(nested_map.NestedMap(x=[])))
    self.assertFalse(py_utils.IsCompatible(empty, nested_map.NestedMap(x=[])))
    self.assertTrue(empty.IsCompatible(nested_map.NestedMap(x=empty)))
    self.assertFalse(
        py_utils.IsCompatible(empty, nested_map.NestedMap(x=empty))
    )
    self.assertTrue(empty.IsCompatible(nested_map.NestedMap(x={})))
    self.assertFalse(py_utils.IsCompatible(empty, nested_map.NestedMap(x={})))
    x = nested_map.NestedMap(
        a='a', b='b', c=nested_map.NestedMap(d='d', e=[1, 2, 4])
    )
    y = nested_map.NestedMap(
        a=1, b=2, c=nested_map.NestedMap(d=3, e=[10, 20, 30])
    )
    z = nested_map.NestedMap(
        a=1, b=[10, 20, 30], c=nested_map.NestedMap(d=3, e=['x', 'y', 'z'])
    )
    self.assertTrue(x.IsCompatible(y))
    self.assertTrue(py_utils.IsCompatible(x, y))
    self.assertFalse(x.IsCompatible(z))
    self.assertFalse(py_utils.IsCompatible(x, z))

  def testFilter(self):
    x = nested_map.NestedMap(
        a=100,
        b=200,
        c=300,
        d=nested_map.NestedMap(foo=38, bar=192, ok=[200, 300], ko=[10, 20]),
    )
    y = x.Filter(lambda v: v > 150)
    self.assertEqual(
        y.FlattenItems(),
        [
            ('b', 200),
            ('c', 300),
            ('d.bar', 192),
            ('d.ok[0]', 200),
            ('d.ok[1]', 300),
        ],
    )
    self.assertNotIn('ko', y.d)

    y = x.Filter(lambda v: v > 500)
    self.assertLen(y.FlattenItems(), 0)

  def testFilterKeyVal(self):
    x = nested_map.NestedMap(
        a=100,
        b=200,
        c=300,
        d=nested_map.NestedMap(foo=38, bar=192, ok=[200, 300], ko=[10, 20]),
    )
    selected = {'a', 'd.foo', 'd.ok[1]'}

    def Sel(k, _):
      return k in selected

    y = x.FilterKeyVal(Sel)
    self.assertEqual(
        y.FlattenItems(), [('a', 100), ('d.foo', 38), ('d.ok[0]', 300)]
    )

  def testCopy(self):
    # This is not a copy.
    x = nested_map.NestedMap(
        a='a', b='b', c=nested_map.NestedMap(d='d', e=[1, 2, 4])
    )
    y = x
    y.a = 'y'
    self.assertEqual('y', y.a)
    self.assertEqual('y', x.a)

    # This is a (shallow) copy.
    x = nested_map.NestedMap(
        a='a', b='b', c=nested_map.NestedMap(d='d', e=[1, 2, 4])
    )
    y = nested_map.NestedMap(x)
    self.assertNotEqual(id(x), id(y))
    y.a = 'y'
    y.c.d = 'z'
    self.assertEqual('y', y.a)
    self.assertEqual('a', x.a)
    self.assertEqual('z', y.c.d)
    self.assertEqual('z', x.c.d)

    # This is also a (shallow) copy.
    x = nested_map.NestedMap(
        a='a', b='b', c=nested_map.NestedMap(d='d', e=[1, 2, 4])
    )
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

    x = nested_map.NestedMap(
        a='a',
        b='b',
        c=nested_map.NestedMap(d='d', e=[1, 2, 4], obj=SomeObj()),
        f=[],
        g={},
        h=nested_map.NestedMap(),
        i=None,
    )
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
    a = nested_map.NestedMap()
    a.a1 = 10
    self.assertEqual(10, a.a1)
    self.assertEqual(10, a['a1'])
    self.assertEqual(10, a.get('a1'))
    self.assertEqual(10, a.get('a1'))
    self.assertEqual(10, getattr(a, 'a1'))

    a['a1'] = 20
    self.assertEqual(20, a.a1)
    self.assertEqual(20, a['a1'])
    self.assertEqual(20, a.get('a1'))
    self.assertEqual(20, a.get('a1'))
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
      _ = nested_map.NestedMap(get=2)

    del a.a1
    with self.assertRaisesRegex(AttributeError, 'available attributes'):
      print(a.a1)

    with self.assertRaisesRegex(AttributeError, 'available attributes'):
      del a.a2


if __name__ == '__main__':
  test_utils.main()
