# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utils.py."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.milan import utils


class SelectorTest(test_utils.TestCase):

  def testSelector(self):
    inputs = py_utils.NestedMap()
    inputs.foo = py_utils.NestedMap()
    inputs.foo.bar = py_utils.NestedMap()
    inputs.foo.bar.a = '<a>'
    inputs.foo.bar.b = '<b>'
    inputs.foo.c = '<c>'

    # Select a single field from 'inputs'.
    selector = utils.Selector('foo.bar.a')
    self.assertEqual('<a>', selector(inputs))

    # Select multiple fields arranged in various structures.
    selector = utils.Selector(('foo.bar.a', 'foo.bar.b'))
    self.assertEqual(('<a>', '<b>'), selector(inputs))

    selector = utils.Selector(('foo.bar.a', ('foo.bar.b', 'foo.c')))
    self.assertEqual(('<a>', ('<b>', '<c>')), selector(inputs))
    selector = utils.Selector({1: 'foo.bar.b', 2: 'foo.bar.a'})
    self.assertEqual({1: '<b>', 2: '<a>'}, selector(inputs))

    spec = py_utils.NestedMap(
        features='foo.bar.a',
        lengths='foo.c',
        nested=py_utils.NestedMap(baz='foo.bar.b'))
    selector = utils.Selector(spec)
    self.assertEqual(
        py_utils.NestedMap(
            features='<a>', lengths='<c>',
            nested=py_utils.NestedMap(baz='<b>')), selector(inputs))


class BatchFlattenerTest(test_utils.TestCase):

  def testTensorInput(self):
    batch_shape = [2, 3]
    flattener = utils.BatchFlattener(batch_shape)

    t = tf.ones(batch_shape + [5, 7])
    flattened = flattener.Flatten(t)
    self.assertAllEqual([6, 5, 7], flattened.shape.as_list())
    unflattened = flattener.Unflatten(flattened)
    self.assertAllEqual(t, unflattened)

  def testStructuredInput(self):
    batch_shape = [2, 3]
    flattener = utils.BatchFlattener(batch_shape)
    nested_map = py_utils.NestedMap(
        foo=tf.ones(batch_shape + [7, 9]),
        bar=tf.ones(batch_shape),
        baz=tf.ones(batch_shape + [1]))
    flattened = flattener.Flatten(nested_map)
    self.assertSameStructure(
        py_utils.NestedMap(foo=[6, 7, 9], bar=[6], baz=[6, 1]),
        flattened.Transform(lambda t: t.shape.as_list()))
    unflattened = flattener.Unflatten(flattened)
    for k in nested_map.keys():
      self.assertAllEqual(nested_map.get(k), unflattened.get(k))


if __name__ == '__main__':
  tf.test.main()
