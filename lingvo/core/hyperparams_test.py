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
"""Tests for hyperparams."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf
from lingvo.core import hyperparams as _params
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


class TestClass1(object):
  """This class is used in ParamsToSimpleTextTest as a value of a variable."""
  pass


class TestClass2(object):
  """This class is used in ParamsToSimpleTextTest as a value of a variable."""
  pass


class ParamsTest(test_utils.TestCase):

  def testEquals(self):
    params1 = _params.Params()
    params2 = _params.Params()
    self.assertTrue(params1 == params2)
    params1.Define('first', 'firstvalue', '')
    self.assertFalse(params1 == params2)
    params2.Define('first', 'firstvalue', '')
    self.assertTrue(params1 == params2)
    some_object = object()
    other_object = object()
    params1.Define('second', some_object, '')
    params2.Define('second', other_object, '')
    self.assertFalse(params1 == params2)
    params2.second = some_object
    self.assertTrue(params1 == params2)
    params1.Define('third', _params.Params(), '')
    params2.Define('third', _params.Params(), '')
    self.assertTrue(params1 == params2)
    params1.third.Define('fourth', 'x', '')
    params2.third.Define('fourth', 'y', '')
    self.assertFalse(params1 == params2)
    params2.third.fourth = 'x'
    self.assertTrue(params1 == params2)

  def testDeepCopy(self):
    inner = _params.Params()
    inner.Define('alpha', 2, '')
    outer = _params.Params()
    outer.Define('beta', 1, '')
    outer.Define('inner', inner, '')
    outer_copy = outer.Copy()
    self.assertTrue(outer is not outer_copy)
    self.assertTrue(outer == outer_copy)
    self.assertTrue(outer.inner is not outer_copy.inner)
    self.assertTrue(outer.inner == outer_copy.inner)

  def testDefineExisting(self):
    p = _params.Params()
    p.Define('foo', 1, '')
    self.assertRaisesRegexp(AttributeError, 'already defined',
                            lambda: p.Define('foo', 1, ''))

  def testLegalParamNames(self):
    p = _params.Params()
    self.assertRaises(AssertionError, lambda: p.Define(None, 1, ''))
    self.assertRaises(AssertionError, lambda: p.Define('', 1, ''))
    self.assertRaises(AssertionError, lambda: p.Define('_foo', 1, ''))
    self.assertRaises(AssertionError, lambda: p.Define('Foo', 1, ''))
    self.assertRaises(AssertionError, lambda: p.Define('1foo', 1, ''))
    self.assertRaises(AssertionError, lambda: p.Define('foo$', 1, ''))
    p.Define('foo_bar', 1, '')
    p.Define('foo9', 1, '')

  def testSetAndGet(self):
    p = _params.Params()
    self.assertRaisesRegexp(AttributeError, 'foo', lambda: p.Set(foo=4))
    # We use setattr() because lambda cannot contain explicit assignment.
    self.assertRaisesRegexp(AttributeError, 'foo', lambda: setattr(p, 'foo', 4))
    p.Define('foo', 1, '')
    self.assertEqual(p.foo, 1)
    self.assertEqual(p.Get('foo'), 1)
    p.Set(foo=2)
    self.assertEqual(p.foo, 2)
    self.assertEqual(p.Get('foo'), 2)
    p.foo = 3
    self.assertEqual(p.foo, 3)
    self.assertEqual(p.Get('foo'), 3)
    p.Delete('foo')
    self.assertRaisesRegexp(AttributeError, 'foo', lambda: p.foo)
    self.assertRaisesRegexp(AttributeError, 'foo', p.Get, 'foo')

  def testSetAndGetNestedParam(self):
    innermost = _params.Params()
    innermost.Define('delta', 22, '')
    innermost.Define('zeta', 5, '')

    inner = _params.Params()
    inner.Define('alpha', 2, '')
    inner.Define('innermost', innermost, '')

    outer = _params.Params()
    outer.Define('beta', 1, '')
    outer.Define('inner', inner, '')
    outer.Define('d', dict(foo='bar'), '')

    self.assertEqual(inner.alpha, 2)
    self.assertEqual(outer.beta, 1)
    self.assertEqual(outer.d['foo'], 'bar')
    self.assertEqual(outer.inner.alpha, 2)
    self.assertEqual(outer.inner.innermost.delta, 22)
    self.assertEqual(outer.inner.innermost.zeta, 5)

    self.assertEqual(inner.Get('alpha'), 2)
    self.assertEqual(outer.Get('beta'), 1)
    self.assertEqual(outer.Get('d')['foo'], 'bar')
    self.assertEqual(outer.Get('inner.alpha'), 2)
    self.assertEqual(outer.Get('inner.innermost.delta'), 22)
    self.assertEqual(outer.Get('inner.innermost.zeta'), 5)

    outer.Set(**{'inner.alpha': 3})
    outer.Set(d=dict(foo='baq'))
    outer.Delete('beta')
    outer.Delete('inner.innermost.zeta')

    self.assertEqual(inner.alpha, 3)
    self.assertRaisesRegexp(AttributeError, 'beta', lambda: outer.beta)
    self.assertEqual(outer.d['foo'], 'baq')
    self.assertEqual(outer.inner.alpha, 3)
    self.assertEqual(outer.inner.innermost.delta, 22)
    self.assertRaisesRegexp(AttributeError, 'zeta',
                            lambda: outer.inner.innermost.zeta)

    self.assertEqual(inner.Get('alpha'), 3)
    self.assertRaisesRegexp(AttributeError, 'beta', outer.Get, 'beta')
    self.assertEqual(outer.Get('d')['foo'], 'baq')
    self.assertEqual(outer.Get('inner.alpha'), 3)
    self.assertEqual(outer.Get('inner.innermost.delta'), 22)
    self.assertRaisesRegexp(AttributeError, 'inner.innermost.zeta', outer.Get,
                            'inner.innermost.zeta')

    # NOTE(igushev): Finding nested Param object is shared between Get, Set and
    # Delete, so we test only Set.
    self.assertRaisesRegexp(AttributeError, 'inner\.gamma',
                            lambda: outer.Set(**{'inner.gamma': 5}))
    self.assertRaisesRegexp(AttributeError, 'inner\.innermost\.bad',
                            lambda: outer.Set(**{'inner.innermost.bad': 5}))
    self.assertRaisesRegexp(AssertionError, '^Cannot introspect',
                            lambda: outer.Set(**{'d.foo': 'baz'}))

  def testFreeze(self):
    p = _params.Params()
    self.assertRaises(AssertionError, lambda: p.Define('_immutable', 1, ''))
    self.assertRaisesRegexp(AttributeError, 'foo', lambda: p.Set(foo=4))
    # We use setattr() because lambda cannot contain explicit assignment.
    self.assertRaisesRegexp(AttributeError, 'foo', lambda: setattr(p, 'foo', 4))
    p.Define('foo', 1, '')
    p.Define('nested', p.Copy(), '')
    self.assertEqual(p.foo, 1)
    self.assertEqual(p.Get('foo'), 1)
    self.assertEqual(p.nested.foo, 1)
    p.Freeze()

    self.assertRaises(TypeError, lambda: p.Set(foo=2))
    self.assertEqual(p.Get('foo'), 1)
    self.assertRaises(TypeError, lambda: setattr(p, 'foo', 3))
    self.assertEqual(p.foo, 1)
    self.assertRaises(TypeError, lambda: p.Delete('foo'))
    self.assertEqual(p.foo, 1)
    self.assertRaises(TypeError, lambda: p.Define('bar', 1, ''))
    self.assertRaisesRegexp(AttributeError, 'bar', p.Get, 'bar')

    p.nested.foo = 2
    self.assertEqual(p.foo, 1)
    self.assertEqual(p.nested.foo, 2)

    self.assertRaises(TypeError, lambda: setattr(p, '_immutable', False))

    # Copies are still immutable.
    q = p.Copy()
    self.assertRaises(TypeError, lambda: q.Set(foo=2))

  def testToString(self):
    outer = _params.Params()
    outer.Define('foo', 1, '')
    inner = _params.Params()
    inner.Define('bar', 2, '')
    outer.Define('inner', inner, '')
    outer.Define('list', [1, inner, 2], '')
    outer.Define('dict', {'a': 1, 'b': inner}, '')
    self.assertEqual(
        '\n' + str(outer), """
{
  dict: {'a': 1, 'b': {'bar': 2}}
  foo: 1
  inner: {
    bar: 2
  }
  list: [1, {'bar': 2}, 2]
}""")

  def testIterParams(self):
    keys, values = ['a', 'b', 'c', 'd', 'e'], [True, None, 'zippidy', 78.5, 5]
    p = _params.Params()
    for k, v in zip(keys, values):
      p.Define(k, v, 'description of %s' % k)

    k_set, v_set = set(keys), set(values)
    number_of_params = 0
    for k, v in p.IterParams():
      self.assertTrue(k in k_set)
      self.assertTrue(v in v_set)
      number_of_params += 1
    self.assertEqual(number_of_params, len(keys))

  def testToText(self):
    outer = _params.Params()
    outer.Define('foo', 1, '')
    inner = _params.Params()
    inner.Define('bar', 2.71, '')
    inner.Define('baz', 'hello', '')
    outer.Define('inner', inner, '')
    outer.Define('tau', False, '')
    outer.Define('dtype', tf.float32, '')
    outer.Define('dtype2', tf.int32, '')
    outer.Define('seqlen', [10, inner, 30], '')
    outer.Define('tuple', (1, None), '')
    outer.Define('list_of_params', [inner.Copy()], '')
    outer.Define('class', TestClass1, '')
    outer.Define('plain_dict', {'a': 10}, '')
    outer.Define('complex_dict', {'a': 10, 'b': inner}, '')
    outer.Define('complex_dict_escape', {'a': 'abc"\'\ndef'}, '')
    outer.Define('some_class', complex(0, 1), '')
    outer.Define('optional_bool', None, '')

    self.assertEqual(
        '\n' + outer.ToText(), r"""
class : type/__main__/TestClass1
complex_dict : {'a': 10, 'b': {'bar': 2.71, 'baz': 'hello'}}
complex_dict_escape : {'a': 'abc"\'\ndef'}
dtype : float32
dtype2 : int32
foo : 1
inner.bar : 2.71
inner.baz : 'hello'
list_of_params[0].bar : 2.71
list_of_params[0].baz : 'hello'
optional_bool : NoneType
plain_dict : {'a': 10}
seqlen : [10, {'bar': 2.71, 'baz': 'hello'}, 30]
some_class : complex
tau : False
tuple : (1, 'NoneType')
""")

    outer.FromText("""
        dtype2 : float32
        inner.baz : 'world'
        # foo : 123
        optional_bool : true
        list_of_params[0].bar : 2.72
        seqlen : [1, 2.0, '3', [4]]
        plain_dict : {'x': 0.3}
        class : type/__main__/TestClass2
        tau : true
        tuple : (2, 3)
        """)

    # Note that the 'hello' has turned into 'world'!
    self.assertEqual(
        '\n' + outer.ToText(), r"""
class : type/__main__/TestClass2
complex_dict : {'a': 10, 'b': {'bar': 2.71, 'baz': 'world'}}
complex_dict_escape : {'a': 'abc"\'\ndef'}
dtype : float32
dtype2 : float32
foo : 1
inner.bar : 2.71
inner.baz : 'world'
list_of_params[0].bar : 2.72
list_of_params[0].baz : 'hello'
optional_bool : True
plain_dict : {'x': 0.3}
seqlen : [1, 2.0, '3', [4]]
some_class : complex
tau : True
tuple : (2, 3)
""")

  def testStringEscaping(self):
    p = _params.Params()
    p.Define('bs_end_quote', 'Single\\', '')
    p.Define('embedded_newlines', 'Split\nAcross\nLines', '')
    p.Define('empty', '', '')
    p.Define('empty_first_line', '\nNext', '')
    p.Define('end_escape_quote', '""Split\'\nLine', '')
    p.Define('escaping_single', 'In "quotes"', '')
    p.Define('escaping_double', 'In \\\'quotes\'', '')

    # Make sure it escapes properly.
    text_value = p.ToText()
    self.assertEqual(
        '\n' + text_value, r"""
bs_end_quote : 'Single\\'
embedded_newlines : 'Split
Across
Lines'
empty : ''
empty_first_line : '
Next'
end_escape_quote : '""Split\'
Line'
escaping_double : "In \\'quotes'"
escaping_single : 'In "quotes"'
""")

    # Reset the values and make sure that reading back in parses.
    p.bs_end_quote = ''
    p.embedded_newlines = ''
    p.empty_first_line = ''
    p.end_escape_quote = ''
    p.escaping_double = ''
    p.escaping_single = ''

    p.FromText(text_value)
    self.assertEqual(p.bs_end_quote, 'Single\\')
    self.assertEqual(p.embedded_newlines, 'Split\nAcross\nLines')
    self.assertEqual(p.empty_first_line, '\nNext')
    self.assertEqual(p.end_escape_quote, '""Split\'\nLine')
    self.assertEqual(p.escaping_single, 'In "quotes"')
    self.assertEqual(p.escaping_double, 'In \\\'quotes\'')


if __name__ == '__main__':
  tf.test.main()
