# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for comparing NestedMap in Python.

When comparing NestedMap variables, instead of getting this type of cryptic
error message in your unit test:

self.assertEqual(expected, actual)
> AssertionError: {'src[35 chars]ape=(16, 512), dtype=float32),
>   'src_inputs': S[305 chars]32)}} != {'src[35 chars]ape=(8, 512),
>   dtype=float32), 'src_inputs': Sh[299 chars]32)}}

You get:
self.assertNestedMapEqual(expected, actual)
> AssertionError:
- src.paddings      ShapeDtypeStruct(shape=(16, 512), dtype=float32)
?                                           ^^
+ src.paddings      ShapeDtypeStruct(shape=(8, 512), dtype=float32)
?                                           ^
- src.src_inputs    ShapeDtypeStruct(shape=(16, 512, 240, 1), dtype=float32)
?                                           ^^
+ src.src_inputs    ShapeDtypeStruct(shape=(8, 512, 240, 1), dtype=float32)
?                                           ^
- src.video         ShapeDtypeStruct(shape=(16, 512, 128, 128), dtype=float32)
?                                           ^^
+ src.video         ShapeDtypeStruct(shape=(8, 512, 128, 128), dtype=float32)
?                                           ^
- tgt.ids           ShapeDtypeStruct(shape=(16, 128), dtype=int32)
?                                           ^^
+ tgt.ids           ShapeDtypeStruct(shape=(8, 128), dtype=int32)
?                                           ^
- tgt.labels        ShapeDtypeStruct(shape=(16, 128), dtype=int32)
?                                           ^^
+ tgt.labels        ShapeDtypeStruct(shape=(8, 128), dtype=int32)
?                                           ^
- tgt.paddings      ShapeDtypeStruct(shape=(16, 128), dtype=float32)
?                                           ^^
+ tgt.paddings      ShapeDtypeStruct(shape=(8, 128), dtype=float32)
?
"""

from typing import Any, Union
import unittest
from lingvo.core import py_utils


# pyformat: disable
def assertNestedMapEqual(  # pylint: disable=invalid-name
    self: unittest.TestCase,
    expected: Union[dict[str, Any], py_utils.NestedMap],
    actual: py_utils.NestedMap):

  if not hasattr(expected, 'DebugString'):
    expected = py_utils.NestedMap(expected)

  self.assertMultiLineEqual(expected.DebugString(), actual.DebugString())
# pyformat: enable


class NestedMapAssertions(unittest.TestCase):
  """Mix this into a googletest.TestCase class to get NestedMap asserts.

  Usage:

  class SomeTestCase(compare.NestedMapAssertions, googletest.TestCase):
    ...
      def testSomething(self):
        ...
        self.assertNestedMapEqual(expected, actual):
  """

  # pylint: disable=invalid-name
  def assertNestedMapEqual(self, *args, **kwargs):
    return assertNestedMapEqual(self, *args, **kwargs)
