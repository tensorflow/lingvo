# Lint as: python2, python3
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
"""Utilities for symbolic computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
from six.moves import zip
import sympy


class Symbol(sympy.Dummy):
  pass


def IsSymbol(x):
  return isinstance(x, Symbol)


def IsExpr(x):
  return isinstance(x, sympy.Expr)


STATIC_VALUES = 'static'
TENSOR_VALUES = 'tensor'
VALUE_TYPES = (STATIC_VALUES, TENSOR_VALUES)


class _LocalSymbolToValueStack(threading.local):
  """A thread-local stack of symbol-to-value dicts."""

  def __init__(self):
    super(_LocalSymbolToValueStack, self).__init__()
    self.stack = {}
    for value_type in VALUE_TYPES:
      self.stack[value_type] = [{}]


class SymbolToValueMap(object):
  """A symbol-to-value mapping.

  Usage:

  with SymbolToValueMap('static', {symbol1: value1, symbol2: value2, ...}):
    with SymbolToValueMap('tensor', {symbol1: value1, symbol2: value2, ...}):
      ... = EvalExpr(value_type, symbolic_expr)

  Multiple SymbolToValueMap context can be nested inside one another. The inner
  contexts take precedence over outer ones when multiple contexts provide
  values for the same symbol.
  """

  _local_stack = _LocalSymbolToValueStack()

  def __init__(self, value_type, symbol_to_value_map):
    """Creates a new symbol to value map.

    Args:
      value_type: the type of values in 'symbol_to_value_map'.
      symbol_to_value_map: a dict from Symbol to values.
    """
    assert value_type in VALUE_TYPES
    self.value_type = value_type
    self.merged = dict(self.Stack(value_type)[-1])
    self.merged.update(symbol_to_value_map)

  @staticmethod
  def Stack(value_type):
    return SymbolToValueMap._local_stack.stack[value_type]

  def __enter__(self):
    self.Stack(self.value_type).append(self.merged)

  def __exit__(self, type_arg, value_arg, traceback_arg):
    stack = self.Stack(self.value_type)
    assert stack
    assert stack[-1] is self.merged
    stack.pop()

  @staticmethod
  def Get(value_type):
    """Returns a symbol-to-value mapping merged from Stack()."""
    return SymbolToValueMap.Stack(value_type)[-1]


def EvalExpr(value_type, x):
  """Evaluates x with symbol_to_value_map within the current context.

  Args:
    value_type: the target value type (see VALUE_TYPE).
    x: a sympy.Expr, an object, or a list/tuple of Exprs and objects.

  Returns:
    Evaluation result of 'x'.
  """
  if isinstance(x, (list, tuple)):
    return type(x)(EvalExpr(value_type, y) for y in x)
  elif isinstance(x, sympy.Expr):
    symbol_to_value_map = SymbolToValueMap.Get(value_type)
    if not symbol_to_value_map:
      return x
    # In theory the below should be equivalent to:
    #   y = x.subs(symbol_to_value_map).
    # In practice subs() doesn't work for when values are Tensors.
    k, v = list(zip(*(list(symbol_to_value_map.items()))))
    y = sympy.lambdify(k, x)(*v)
    return y
  else:
    return x


def ToStatic(expr):
  return EvalExpr(STATIC_VALUES, expr)


def ToTensor(expr):
  return EvalExpr(TENSOR_VALUES, expr)
