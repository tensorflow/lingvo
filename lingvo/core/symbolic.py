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
import sympy


class _Symbol(sympy.Dummy):
  pass


def NewSymbol(name):
  return _Symbol(name)


def IsSymbol(x):
  return isinstance(x, _Symbol)


def IsExpr(x):
  return isinstance(x, sympy.Expr)


class _LocalSymbolToValueStack(threading.local):
  """A thread-local stack of symbol-to-value dicts."""

  def __init__(self):
    super(_LocalSymbolToValueStack, self).__init__()
    self.stack = [{}]


class SymbolToValueMap(object):
  """A symbol-to-value mapping.

  Usage:

  with SymbolToValueMap({symbol1: value1, symbol2: value2, ...}):
    ... = EvalExpr(symbolic_expr)

  Multiple SymbolToValueMap context can be nested inside one another. The inner
  contexts take precedence over outer ones when multiple contexts provide
  values for the same symbol.
  """

  _local_stack = _LocalSymbolToValueStack()

  def __init__(self, symbol_to_value_map):
    """Creates a new symbol to value map.

    Args:
      symbol_to_value_map: a dict from _Symbol to values.
    """
    self.merged = dict(self.Stack()[-1])
    self.merged.update(symbol_to_value_map)

  @staticmethod
  def Stack():
    return SymbolToValueMap._local_stack.stack

  def __enter__(self):
    self.Stack().append(self.merged)

  def __exit__(self, type_arg, value_arg, traceback_arg):
    stack = self.Stack()
    assert stack
    assert stack[-1] is self.merged
    stack.pop()

  @staticmethod
  def Get():
    """Returns a symbol-to-value mapping merged from Stack()."""
    return SymbolToValueMap.Stack()[-1]


def EvalExpr(x):
  """Evaluates x with symbol_to_value_map within the current context.

  Args:
    x: a sympy.Expr, an object, or a list/tuple of Exprs and objects.

  Returns:
    Evaluation result of 'x'.
  """
  if isinstance(x, (list, tuple)):
    return type(x)(EvalExpr(y) for y in x)
  elif isinstance(x, sympy.Expr):
    # In theory the below should be equivalent to:
    #   y = x.subs(symbol_to_value_map).
    # In practice subs() doesn't work for when values are Tensors.
    k, v = zip(*(SymbolToValueMap.Get().items()))
    y = sympy.lambdify(k, x)(*v)
    return y
  else:
    return x
