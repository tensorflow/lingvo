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
"""Tests for asserts."""

from absl.testing import absltest
from absl.testing import parameterized
from lingvo.jax import asserts


class AssertsTest(parameterized.TestCase):

  def test_none(self):
    value = None
    asserts.none(value)

  @parameterized.parameters(0, 1.2, 'hello')
  def test_none_raises(self, value):
    with self.assertRaisesRegex(ValueError,
                                f'`value={value}` must be `None`.$'):
      asserts.none(value)
    with self.assertRaisesRegex(ValueError,
                                f'`custom_value={value}` must be `None`.$'):
      asserts.none(value, value_str=f'custom_value={value}')
    custom_error_msg = 'This is a custom error message.'
    with self.assertRaisesRegex(ValueError, f'{custom_error_msg}$'):
      asserts.none(value, msg=custom_error_msg)

  @parameterized.parameters(0, 1.2, 'hello')
  def test_not_none(self, value):
    asserts.not_none(value)

  def test_not_none_raises(self):
    value = None
    with self.assertRaisesRegex(ValueError,
                                f'`value={value}` must not be `None`.$'):
      asserts.not_none(value)
    with self.assertRaisesRegex(ValueError,
                                f'`custom_value={value}` must not be `None`.$'):
      asserts.not_none(value, value_str=f'custom_value={value}')
    custom_error_msg = 'This is a custom error message.'
    with self.assertRaisesRegex(ValueError, f'{custom_error_msg}$'):
      asserts.not_none(value, msg=custom_error_msg)

  @parameterized.parameters(1234, 1.2, 'a')
  def test_eq(self, value1):
    value2 = value1
    asserts.eq(value1, value2)

  @parameterized.parameters((0, 1234), (1., 'hello'), ('a', 'b'))
  def test_eq_raises(self, value1, value2):
    with self.assertRaisesRegex(
        ValueError, f'`value1={value1}` must be equal to `value2={value2}`.$'):
      asserts.eq(value1, value2)
    with self.assertRaisesRegex(
        ValueError,
        f'`custom_value={value1}` must be equal to `value2={value2}`.$'):
      asserts.eq(value1, value2, value_str1=f'custom_value={value1}')
    with self.assertRaisesRegex(
        ValueError,
        f'`value1={value1}` must be equal to `custom_value={value2}`.$'):
      asserts.eq(value1, value2, value_str2=f'custom_value={value2}')
    custom_error_msg = 'This is a custom error message.'
    with self.assertRaisesRegex(ValueError, f'{custom_error_msg}$'):
      asserts.eq(value1, value2, msg=custom_error_msg)

  @parameterized.parameters((0, 1234), (1., 'hello'), ('a', 'b'))
  def test_ne(self, value1, value2):
    asserts.ne(value1, value2)

  def test_ne_raises(self):
    value1 = 1234
    value2 = value1
    with self.assertRaisesRegex(
        ValueError,
        f'`value1={value1}` must not be equal to `value2={value2}`.$'):
      asserts.ne(value1, value2)
    with self.assertRaisesRegex(
        ValueError,
        f'`custom_value={value1}` must not be equal to `value2={value2}`.$'):
      asserts.ne(value1, value2, value_str1=f'custom_value={value1}')
    with self.assertRaisesRegex(
        ValueError,
        f'`value1={value1}` must not be equal to `custom_value={value2}`.$'):
      asserts.ne(value1, value2, value_str2=f'custom_value={value2}')
    custom_error_msg = 'This is a custom error message.'
    with self.assertRaisesRegex(ValueError, f'{custom_error_msg}$'):
      asserts.ne(value1, value2, msg=custom_error_msg)

  @parameterized.parameters((0, int), (1.2, (int, float)), ('hello', str),
                            ([1, 2], list))
  def test_instance(self, value, instance):
    asserts.instance(value, instance)

  @parameterized.parameters((0, float), (1.2, (int, str)), ('hello', list),
                            ([1, 2], int))
  def test_instance_raises(self, value, instance):
    with self.assertRaisesRegex(ValueError,
                                '`value=.*` must be of type `.*`.$'):
      asserts.instance(value, instance)

  def test_subclass(self):

    class A:
      pass

    class B(A):
      pass

    asserts.subclass(B, A)

  def test_subclass_raises(self):

    class A:
      pass

    class B(A):
      pass

    with self.assertRaisesRegex(ValueError,
                                '`.*` must be a subclass of `.*`.$'):
      asserts.subclass(A, B)

  @parameterized.parameters((0, 0), (1.2, 3.4), (10, 30.))
  def test_le(self, value1, value2):
    asserts.le(value1, value2)

  @parameterized.parameters((1, 0), (3.4, 1.2), (30., 10))
  def test_le_raises(self, value1, value2):
    with self.assertRaisesRegex(
        ValueError,
        f'`value1={value1}` must be less than or equal to `value2={value2}`.$'):
      asserts.le(value1, value2)

  @parameterized.parameters((0, 1), (1.2, 3.4), (10, 30.))
  def test_lt(self, value1, value2):
    asserts.lt(value1, value2)

  @parameterized.parameters((0, 0), (3.4, 1.2), (30., 10))
  def test_lt_raises(self, value1, value2):
    with self.assertRaisesRegex(
        ValueError,
        f'`value1={value1}` must be strictly less than `value2={value2}`.$'):
      asserts.lt(value1, value2)

  @parameterized.parameters((0, 0), (3.4, 1.2), (30., 10))
  def test_ge(self, value1, value2):
    asserts.ge(value1, value2)

  @parameterized.parameters((0, 1), (1.2, 3.4), (10, 30.))
  def test_ge_raises(self, value1, value2):
    with self.assertRaisesRegex(
        ValueError,
        f'`value1={value1}` must be greater than or equal to `value2={value2}`.$'
    ):
      asserts.ge(value1, value2)

  @parameterized.parameters((1, 0), (3.4, 1.2), (30., 10))
  def test_gt(self, value1, value2):
    asserts.gt(value1, value2)

  @parameterized.parameters((0, 0), (1.2, 3.4), (10, 30.))
  def test_gt_raises(self, value1, value2):
    with self.assertRaisesRegex(
        ValueError,
        f'`value1={value1}` must be strictly greater than `value2={value2}`.$'):
      asserts.gt(value1, value2)

  @parameterized.parameters((0, list(range(5))), ('tanh', ('tanh', 'relu')))
  def test_in_set(self, value, elements):
    asserts.in_set(value, elements)

  @parameterized.parameters((10, list(range(5))), ('sigmoid', ('tanh', 'relu')))
  def test_in_set_raises(self, value, elements):
    with self.assertRaisesRegex(ValueError,
                                f'`value={value}` must be within `.*`.$'):
      asserts.in_set(value, elements)

  @parameterized.parameters((1, 0, 2, False, False),
                            (1.2, 1.2, 5.6, False, True),
                            (100., 10, 100., True, False))
  def test_between(self, value, min_value, max_value, left_strict,
                   right_strict):
    asserts.between(
        value,
        min_value,
        max_value,
        left_strict=left_strict,
        right_strict=right_strict,
        value_str=f'value={value}')

  @parameterized.parameters((2, 0, 1, False, False),
                            (1.2, 1.2, 3.4, True, False),
                            (100., 10, 100., False, True))
  def test_between_raises(self, value, min_value, max_value, left_strict,
                          right_strict):
    with self.assertRaisesRegex(ValueError,
                                f'`value={value}` must be in the range `.*`.$'):
      asserts.between(
          value,
          min_value,
          max_value,
          left_strict=left_strict,
          right_strict=right_strict,
          value_str=f'value={value}')

  def test_multiline_invalid_label(self):
    value = 1
    min_value = 0
    max_value = 2
    left_strict = True
    right_strict = True
    asserts.between(
        value,
        min_value,
        max_value,
        left_strict=left_strict,
        right_strict=right_strict)

  def test_multiline_invalid_label_raises(self):
    value = 10
    min_value = 0
    max_value = 2
    left_strict = True
    right_strict = True
    with self.assertRaisesRegex(ValueError,
                                f'`.*=?{value}` must be in the range `.*`.$'):
      asserts.between(
          value,
          min_value,
          max_value,
          left_strict=left_strict,
          right_strict=right_strict)


if __name__ == '__main__':
  absltest.main()
