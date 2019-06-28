# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Symbolic representation of tensor shapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
import six
import sympy


class Shape(object):
  """Shape represents a tensor's symbolic shape."""

  def __init__(self, dims):
    """Constructs a shape whose i-th dim is dims[i].

    Each dim can be one of the following types:
      integer: represents the dimension is a known and fixed.
      string: represents the dimension is an unknown and a sympy dummy symbol is
        used to represent it. Also note that contents of strings only matter for
        logging/printing. Even if the same string is given on multiple
        dimensions, it doesn't mean that they are the same.
      sympy expression: represents a dimension which possibly
        depends on dimensions of other shapes.

    Args:
      dims: A list of either integer, string or sympy.Symbol.
    """
    self._shape = []
    for x in dims:
      assert x is not None, str(dims)
      if isinstance(x, six.string_types):
        # NOTE: Dummy(x) creates a unique symbol. I.e., the value of x has no
        # meaning except for printing, etc.
        self._shape.append(sympy.Dummy(x, integer=True))
      else:
        # Converts x to a sympy type. E.g., int to sympy.Integer.
        self._shape.append(sympy.sympify(x))
    self._size = sympy.prod(self._shape)

  @property
  def rank(self):
    """Returns the rank of the tensor."""
    return len(self._shape)

  @property
  def size(self):
    """Returns the size (num of elements) of the tensor."""
    return self._size

  def num_elements(self):  # pylint: disable=invalid-name
    """Returns the size (num of elements) of the tensor."""
    return self.size

  def __getitem__(self, key):
    """Returns one dimension or a shape from a slice of dimensions."""
    if isinstance(key, int):
      return self._shape[key]
    elif isinstance(key, slice):
      return Shape(self._shape[key])
    else:
      raise TypeError("Invalid argument type.")

  def __add__(self, other):
    """Concatenates two shapes into one."""
    # pylint: disable=protected-access
    if isinstance(other, Shape):
      return Shape(self._shape + other._shape)
    elif isinstance(other, list):
      return Shape(self._shape + Shape(other)._shape)
    else:
      raise NotImplementedError

  def __radd__(self, other):
    """Concatenates two shapes into one."""
    # pylint: disable=protected-access
    if isinstance(other, Shape):
      return Shape(other._shape + self._shape)
    elif isinstance(other, list):
      return Shape(Shape(other)._shape + self._shape)
    else:
      raise NotImplementedError

  def __str__(self):
    return str(self._shape)

  def Subs(self, bindings):
    """Substitute symbols with new values.

    Args:
      bindings: key/value items correspond to old/new pairs for substitution.

    Returns:
      The Shape with symbols substituted according to bindings.
    """
    return Shape([x.subs(bindings) for x in self._shape])

  def ToTensorShape(self):
    """Converts to a possibly partially specified tf.TensorShape."""
    dims = []
    for d in self._shape:
      if d.is_number and d.is_integer:
        dims.append(int(d))
      else:
        dims.append(None)
    return tf.TensorShape(dims)
